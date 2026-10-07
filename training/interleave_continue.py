"""Continue an existing LoRA trunk with **interleaved** primitive batches (the B-14 fix).

The trainer's epoch runs sequential phases — all `noul`, then all `choice`, then all `score`
(`score` last) — and those ~490-step pure phases are what invert a cell through the shared
trunk (B-14's driver test: `noul`-only continuation recovers the cell the joint run kills).
This script resumes a trained adapter and consumes batches round-robin over the three
primitives, changing *only* the order in which batches reach the optimizer: same loss math
(`batch_loss` below mirrors ``training.finetune_rlcd._train``), same split, same per-epoch
per-primitive shuffle seeds, same heads.

Resume fidelity rules (both learned the hard way in B-14):

- ``PeftModel.from_pretrained`` defaults to ``is_trainable=False`` — always pass
  ``is_trainable=True``, and hash ``adapter_model.safetensors`` before and after the run;
  identical hashes mean nothing trained (the run fails loudly below).
- The `choice` heads live in the adapter's ``choice_head.json`` (keyed bank:
  ``{shared, domains: {name: {head, signatures}}}``) and are warm-started from it, not
  re-initialized.

Torch/transformers/peft are imported lazily (the ``train`` extra).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.finetune_rlcd import (
    choice_head_modules,
    choice_head_payload,
    choice_target_index,
    criterion_texts,
    question_text,
    read_records,
    scorer_payload,
    shuffled,
    split_records,
)
from training.fit_choice_bank import warm_start

_TRAIN_HINT = "the train extra is required for fine-tuning: uv sync --extra train"

#: The fixed cycle order. Every optimizer window sees all three primitives' gradients.
PRIMITIVE_ORDER: tuple[str, ...] = ("noul", "choice", "score")

#: Recipe keys continued from the trunk's own ``finetune_config.json`` when the CLI omits them.
_HYPER_KEYS: tuple[str, ...] = (
    "model_id",
    "batch_size",
    "grad_accum",
    "learning_rate",
    "max_len",
    "seed",
    "choice_rank",
)

DEFAULTS: dict[str, Any] = {
    "model_id": "jhu-clsp/mmBERT-base",
    "batch_size": 8,
    "grad_accum": 4,
    "learning_rate": 1e-4,
    "max_len": 1024,
    "seed": 42,
    "choice_rank": 128,
}


def interleaved_batches(
    grouped: Mapping[str, Sequence[dict[str, Any]]],
    *,
    batch_size: int,
    seed: int | str,
    order: Sequence[str] = PRIMITIVE_ORDER,
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Round-robin the per-primitive batches: noul, choice, score, noul, ... (B-14).

    Each primitive keeps its own seeded shuffle (``seed:<kind>``) and its own batch slicing —
    only the delivery order changes: the cursor cycles through ``order`` and skips exhausted
    primitives, so every record reaches the optimizer exactly once per epoch while no
    primitive ever gets a long pure phase through the shared trunk.
    """
    queues: dict[str, list[list[dict[str, Any]]]] = {}
    for kind in order:
        ordered = shuffled(grouped.get(kind, ()), seed=f"{seed}:{kind}")
        queues[kind] = [
            ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)
        ]
    cursors: dict[str, int] = dict.fromkeys(queues, 0)
    while any(cursors[kind] < len(queues[kind]) for kind in order):
        for kind in order:
            if cursors[kind] >= len(queues[kind]):
                continue
            batch = queues[kind][cursors[kind]]
            cursors[kind] += 1
            yield kind, batch


def sha256(path: Path) -> str:
    """Hash of a file's bytes (B-14's rule: prove the adapter actually moved)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_hyperparameters(adapter: Path) -> dict[str, Any]:
    """Recipe keys from the trunk's ``finetune_config.json`` (absent → empty)."""
    path = adapter / "finetune_config.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: data[key] for key in _HYPER_KEYS if key in data}


def resolve_model_id(adapter: Path, override: str | None) -> str:
    """``--model-id`` > the trunk's ``finetune_config.json`` > the PEFT adapter config.

    Never infer from the directory name: ``predict``'s substring heuristic silently picked the
    English trunk for ``tmp_il_*`` directories and every load died with a size mismatch.
    """
    if override:
        return override
    hyper = adapter_hyperparameters(adapter)
    if isinstance(hyper.get("model_id"), str):
        return str(hyper["model_id"])
    peft_config = adapter / "adapter_config.json"
    if peft_config.is_file():
        base = json.loads(peft_config.read_text(encoding="utf-8")).get("base_model_name_or_path")
        if isinstance(base, str) and base:
            return base
    raise SystemExit(f"cannot resolve the base model for {adapter}: pass --model-id")


def resolve_setting(cli_value: Any, hyper: Mapping[str, Any], key: str) -> Any:
    """CLI override > the trunk's recorded recipe > the built-in default."""
    if cli_value is not None:
        return cli_value
    if key in hyper:
        return hyper[key]
    return DEFAULTS[key]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tachyone-interleave-continue", description=__doc__)
    parser.add_argument("--adapter", required=True, help="trunk adapter directory to resume")
    parser.add_argument("--out", required=True, help="output adapter directory (must be new)")
    parser.add_argument(
        "--epochs",
        required=True,
        help="comma-separated epoch numbers to run, e.g. 8,9 (they seed the shuffles)",
    )
    parser.add_argument("--data", default="data/train_multi_domains.jsonl")
    parser.add_argument(
        "--model-id", default=None, help="base model (resolved from the adapter otherwise)"
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--choice-rank", type=int, default=None)
    parser.add_argument(
        "--val-fraction", type=float, default=0.1, help="same split rule as the trainer"
    )
    return parser


def continue_interleaved(args: argparse.Namespace) -> dict[str, Any]:
    """Resume ``args.adapter`` for the epochs in ``args.epochs`` with round-robin batches."""
    try:
        import torch  # pyright: ignore[reportMissingImports]
        import torch.nn.functional as F  # pyright: ignore[reportMissingImports]
        from peft import PeftModel  # pyright: ignore[reportMissingImports]
        from transformers import (  # pyright: ignore[reportMissingImports]
            AutoModel,
            AutoTokenizer,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(_TRAIN_HINT) from exc

    adapter = Path(args.adapter)
    output = Path(args.out)
    if not adapter.is_dir():
        raise SystemExit(f"adapter directory not found: {adapter}")
    if output.resolve() == adapter.resolve():
        raise SystemExit("--out must differ from --adapter: the run hashes both to prove training")
    epochs = [int(item) for item in str(args.epochs).split(",") if item.strip()]
    if not epochs:
        raise SystemExit(f"--epochs must name at least one epoch, got {args.epochs!r}")

    hyper = adapter_hyperparameters(adapter)
    settings: dict[str, Any] = {
        "model_id": resolve_model_id(adapter, args.model_id),
        "batch_size": resolve_setting(args.batch_size, hyper, "batch_size"),
        "grad_accum": resolve_setting(args.grad_accum, hyper, "grad_accum"),
        "learning_rate": resolve_setting(args.learning_rate, hyper, "learning_rate"),
        "max_len": resolve_setting(args.max_len, hyper, "max_len"),
        "seed": resolve_setting(args.seed, hyper, "seed"),
        "choice_rank": resolve_setting(args.choice_rank, hyper, "choice_rank"),
    }
    batch_size = int(settings["batch_size"])
    grad_accum = int(settings["grad_accum"])
    seed = int(settings["seed"])
    rank = int(settings["choice_rank"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    before = sha256(adapter / "adapter_model.safetensors")

    tokenizer = AutoTokenizer.from_pretrained(str(settings["model_id"]))
    encoder = AutoModel.from_pretrained(str(settings["model_id"]))
    encoder.gradient_checkpointing_enable()
    # The trap: PEFT defaults to is_trainable=False and would train nothing at all.
    model = PeftModel.from_pretrained(encoder, str(adapter), is_trainable=True)
    model.enable_input_require_grads()  # type: ignore[attr-defined]
    model.to(device)
    model.train()  # type: ignore[attr-defined]

    # The split mirrors the trainer's: the train membership must be exactly what the trunk
    # saw (the val half is unused here — this continuation has no validation step).
    train_records, _val_records = split_records(
        list(read_records(args.data)), val_fraction=args.val_fraction, seed=seed
    )
    grouped: dict[str, list[dict[str, Any]]] = {kind: [] for kind in PRIMITIVE_ORDER}
    for record in train_records:
        grouped[record["type"]].append(record)
    domains = tuple(
        sorted(
            {
                str(record["domain"])
                for record in grouped["choice"]
                if isinstance(record.get("domain"), str)
            }
        )
    )
    languages = tuple(sorted({str(record["lang"]) for record in train_records}))

    hidden = int(encoder.config.hidden_size)
    shared_head, domain_heads = choice_head_modules(
        torch, hidden_size=hidden, rank=rank, init_std=0.0, domains=domains
    )
    payload = json.loads((adapter / "choice_head.json").read_text(encoding="utf-8"))
    shared_payload = payload.get("shared", payload)
    warm_start(shared_head, shared_payload, torch, name="shared")
    domain_payloads = payload.get("domains", {})
    for name, head in domain_heads.items():
        entry = domain_payloads.get(name)
        scorer = entry["head"] if isinstance(entry, dict) and "head" in entry else entry
        warm_start(head, scorer if scorer is not None else shared_payload, torch, name=name)
    shared_head = shared_head.to(device)
    domain_heads = {name: head.to(device) for name, head in domain_heads.items()}
    domain_params = [p for name in domains for p in domain_heads[name].parameters()]

    temperature_file = json.loads((adapter / "temperature.json").read_text(encoding="utf-8"))
    log_temperature = torch.nn.Parameter(
        torch.tensor(math.log(float(temperature_file["temperature"])), device=device)
    )
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *shared_head.parameters(), *domain_params, log_temperature],
        lr=float(settings["learning_rate"]),
    )

    def encode(texts: list[str]) -> Any:
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(settings["max_len"]),
            return_tensors="pt",
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            hidden_states = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return pooled.float()

    def batch_loss(kind: str, batch: list[dict[str, Any]]) -> Any:
        """Verbatim mirror of ``training.finetune_rlcd._train``'s loss (same math, same heads)."""
        temperature = torch.exp(log_temperature) + 1e-3
        states = encode([record["state"] for record in batch])
        questions = encode([question_text(record) for record in batch])
        if kind == "noul":
            similarity = F.cosine_similarity(questions, states, dim=-1)
            probabilities = torch.sigmoid(similarity / temperature)
            targets = torch.tensor([float(r["target"]) for r in batch], device=device)
            return ((probabilities - targets) ** 2).mean()
        criterion_rows = [text for record in batch for text in criterion_texts(record)]
        criteria = encode(criterion_rows)
        shared_context = shared_head.context(states, questions) if kind == "choice" else None
        slots: dict[int, tuple[Any, Any]] = {}
        if kind == "choice":
            contexts: dict[str, Any] = {}
            for index, record in enumerate(batch):
                name = record.get("domain")
                if isinstance(name, str) and name in domain_heads:
                    if name not in contexts:
                        contexts[name] = domain_heads[name].context(states, questions)
                    slots[index] = (domain_heads[name], contexts[name][index])
        total = torch.zeros((), device=device)
        offset = 0
        for index, record in enumerate(batch):
            count = len(criterion_texts(record))
            block = criteria[offset : offset + count]
            offset += count
            state = states[index].expand(count, -1)
            question = questions[index].expand(count, -1)
            base = (
                F.cosine_similarity(block, state, dim=-1)
                + F.cosine_similarity(block, question, dim=-1)
            ) / temperature
            target_index = (
                int(record["target"])
                if isinstance(record["target"], int)
                else choice_target_index(record)
            )
            one_hot = torch.zeros(count, device=device)
            one_hot[target_index] = 1.0
            if shared_context is not None:
                residual = (shared_context[index] * shared_head.project(block)).sum(dim=-1)
                total = (
                    total
                    + (
                        (torch.softmax(base + residual / math.sqrt(rank), dim=0) - one_hot) ** 2
                    ).sum()
                )
            slot = slots.get(index)
            if slot is not None:
                head, context_row = slot
                residual = (context_row * head.project(block)).sum(dim=-1)
                scaled = base + residual / math.sqrt(rank)
                total = total + ((torch.softmax(scaled, dim=0) - one_hot) ** 2).sum()
            if kind == "score":
                total = total + ((torch.softmax(base, dim=0) - one_hot) ** 2).sum()
        return total / len(batch)

    losses: list[dict[str, Any]] = []
    for epoch in epochs:
        model.train()  # type: ignore[attr-defined]
        optimizer.zero_grad(set_to_none=True)
        accumulations = 0
        epoch_loss: dict[str, float] = dict.fromkeys(PRIMITIVE_ORDER, 0.0)
        epoch_seen: dict[str, int] = dict.fromkeys(PRIMITIVE_ORDER, 0)
        for kind, chunk in interleaved_batches(
            grouped, batch_size=batch_size, seed=f"{seed}:{epoch}"
        ):
            loss = batch_loss(kind, chunk) / grad_accum
            epoch_loss[kind] += float(loss.detach()) * grad_accum * len(chunk)
            epoch_seen[kind] += len(chunk)
            loss.backward()
            accumulations += 1
            if accumulations % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if accumulations % grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        means = " ".join(
            f"{kind}={epoch_loss[kind] / max(1, epoch_seen[kind]):.4f}" for kind in PRIMITIVE_ORDER
        )
        print(f"epoch {epoch} interleaved train_loss {means}", flush=True)
        losses.append({"epoch": epoch, **epoch_loss, "seen": dict(epoch_seen)})

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output))
    bank = choice_head_payload(
        shared=scorer_payload(shared_head, rank=rank),
        domains={name: scorer_payload(head, rank=rank) for name, head in domain_heads.items()},
        languages=languages,
    )
    (output / "choice_head.json").write_text(json.dumps(bank, sort_keys=True), encoding="utf-8")
    (output / "temperature.json").write_text(
        json.dumps(
            {"temperature": float(torch.exp(log_temperature).item()), "val_loss": -1.0},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    after = sha256(output / "adapter_model.safetensors")
    if after == before:
        raise SystemExit(
            "the output adapter is byte-identical to the input: nothing trained "
            "(check is_trainable=True and that the epoch actually ran)"
        )
    provenance = {
        "adapter_in": str(adapter),
        "adapter_sha256_before": before,
        "adapter_sha256_after": after,
        "data_path": args.data,
        "epochs": epochs,
        "model_id": settings["model_id"],
        "settings": {key: value for key, value in settings.items() if key != "model_id"},
        "val_fraction": args.val_fraction,
        "train_loss": losses,
    }
    (output / "interleave_config.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    return provenance


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = continue_interleaved(args)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
