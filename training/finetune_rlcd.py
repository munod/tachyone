"""LoRA/QLoRA fine-tuning of the encoder with an RLCD proper-scoring objective.

The training head mirrors the inference math in ``tachyone.backends.encoder`` (cosine similarity
between the state, question, and criterion embeddings, softmaxed with a learned temperature),
so fine-tuning the shared encoder directly improves the local backend without a separate
head format. Torch/transformers/peft are imported lazily (the ``train`` extra); a ``--dry-run``
validates the config and data without them.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from training.generate_data import DOMAINS, domain_signatures
from training.rlcd import categorical_loss

_TRAIN_HINT = "the train extra is required for fine-tuning: uv sync --extra train"


def choice_head_modules(
    torch: Any, *, hidden_size: int, rank: int, init_std: float, domains: Sequence[str]
) -> tuple[Any, dict[str, Any]]:
    """Build the shared head plus one head per domain (ADR-0016 §1).

    The single definition of the head math, shared by the joint trainer and the frozen-trunk
    fitter (`training/fit_choice_bank.py`) so the two can never drift apart. ``torch`` is
    injected so this module stays importable without the ``train`` extra.

    Creation order matters: the shared head draws first (legacy runs — ``domains`` empty —
    therefore consume exactly the RNG the pre-bank trainer consumed) and domain heads draw
    after it (L-011, ADR-0016 §1).
    """

    class ChoiceScorer(torch.nn.Module):
        """Low-rank choice residual; small init keeps training at the cosine baseline at first."""

        def __init__(self) -> None:
            super().__init__()
            self.w1 = torch.nn.Linear(2 * hidden_size, rank, bias=False)
            self.w2 = torch.nn.Linear(hidden_size, rank, bias=False)
            torch.nn.init.normal_(self.w1.weight, std=init_std)
            torch.nn.init.normal_(self.w2.weight, std=init_std)

        def context(self, states: Any, questions: Any) -> Any:
            return self.w1(torch.cat([states, questions], dim=-1))

        def project(self, block: Any) -> Any:
            return self.w2(block)

    shared = ChoiceScorer()
    domain_heads = {name: ChoiceScorer() for name in domains}
    return shared, domain_heads


#: Candidate attention/MLP projections, tried in order. ModernBERT uses Wqkv/Wo/Wi;
#: BERT/mmBERT-style trunks use query/key/value/dense.
_LORA_CANDIDATES: tuple[str, ...] = (
    "Wqkv",
    "Wo",
    "Wi",
    "query",
    "key",
    "value",
    "dense",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)


@dataclass(frozen=True, slots=True)
class FinetuneConfig:
    """Hyperparameters and paths for one fine-tuning run."""

    model_id: str = "answerdotai/ModernBERT-large"
    data_path: str = "data/train.jsonl"
    out_dir: str = "checkpoints/en"
    seed: int = 42
    epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 4
    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    choice_rank: int = 16
    choice_init_std: float = 0.01
    max_len: int = 512
    val_split: float = 0.1
    bf16: bool = True
    gradient_checkpointing: bool = True
    use_4bit: bool = False
    max_records: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FinetuneConfig:
        known = {field.name for field in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> FinetuneConfig:
    """Load a :class:`FinetuneConfig` from a JSON file."""
    return FinetuneConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def read_records(path: str | Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield JSONL records, optionally capped at ``limit``."""
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def split_records(
    records: Sequence[dict[str, Any]], *, val_fraction: float, seed: int | str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split into train/validation with a **seeded shuffle** (B-5).

    The split used to take the tail of the file, and generated data is ordered by domain and then
    primitive — so on a multi-domain dataset the "validation" set was the tail of a single domain,
    and the published ``val_loss`` 0.326 turned out to be `support`/`score` alone. Shuffling makes
    it representative of what is being reported while staying reproducible.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be within [0, 1), got {val_fraction!r}")
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    cut = int(len(records) * (1.0 - val_fraction))
    return [records[index] for index in order[:cut]], [records[index] for index in order[cut:]]


def shuffled[T](items: Sequence[T], *, seed: int | str) -> list[T]:
    """Return ``items`` in a seeded random order (a new list; the input is untouched).

    Optimizer steps are built from consecutive batches, so feeding records in file order makes
    every step a single (domain, primitive) gradient — 32 correlated records per step, domains
    fighting each other from step to step. Shuffling inside each primitive keeps the per-kind
    batching the encoder needs while mixing the domains across steps (B-5).
    """
    order = list(items)
    random.Random(seed).shuffle(order)
    return order


def summarize_dataset(path: str | Path, *, limit: int | None = None) -> dict[str, Any]:
    """Count records per primitive and language (pure; no torch)."""
    per_type: dict[str, int] = {}
    per_lang: dict[str, int] = {}
    total = 0
    for record in read_records(path, limit=limit):
        total += 1
        per_type[record["type"]] = per_type.get(record["type"], 0) + 1
        per_lang[record["lang"]] = per_lang.get(record["lang"], 0) + 1
    return {"total": total, "per_type": per_type, "per_lang": per_lang}


def _ground_truth_loss(record: dict[str, Any], probabilities: Any) -> float:
    """Pure-python reference loss used to sanity-check the torch path."""
    if record["type"] == "noul":
        return (float(probabilities) - float(record["target"])) ** 2
    return categorical_loss(probabilities, record["target"])


def _train(config: FinetuneConfig, report: dict[str, Any]) -> dict[str, Any]:
    """Run LoRA/QLoRA training with batched forward passes.

    The head math mirrors the runtime encoder (cosine similarity, softmaxed with a learned
    temperature), so only the shared trunk is adapted. Batches are grouped by primitive and the
    encoder runs once per batch; imports are lazy behind the ``train`` extra.
    """
    try:
        import torch  # pyright: ignore[reportMissingImports]
        import torch.nn.functional as F  # pyright: ignore[reportMissingImports]
        from peft import LoraConfig, get_peft_model  # pyright: ignore[reportMissingImports]
        from transformers import (  # pyright: ignore[reportMissingImports]
            AutoModel,
            AutoTokenizer,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(_TRAIN_HINT) from exc

    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = config.bf16 and device == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    encoder = AutoModel.from_pretrained(config.model_id)
    if config.gradient_checkpointing:
        encoder.gradient_checkpointing_enable()
    linear_names = {
        name.split(".")[-1]
        for name, module in encoder.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    target_modules = [candidate for candidate in _LORA_CANDIDATES if candidate in linear_names]
    if not target_modules:  # last resort: adapt every linear projection
        target_modules = sorted(linear_names)
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(encoder, lora)
    if config.gradient_checkpointing:
        model.enable_input_require_grads()  # type: ignore[attr-defined]
    model.to(device)
    model.train()

    hidden_size = int(encoder.config.hidden_size)

    records = list(read_records(config.data_path, limit=config.max_records))
    train_records, val_records = split_records(
        records, val_fraction=config.val_split, seed=config.seed
    )
    grouped: dict[str, list[dict[str, Any]]] = {kind: [] for kind in ("noul", "choice", "score")}
    for record in train_records:
        grouped[record["type"]].append(record)

    # Fail before training, not after: signatures come from committed domain files (L-011).
    # ADR-0016 §1: the bank exists only when the *training* records carry `domain` fields, so
    # legacy single-domain configs keep the original RNG stream (shared head init draws at the
    # same point) and the original code path below — byte-identical artifacts (L-011). Domains
    # were already validated against the committed files in `run`.
    domains = choice_domains(grouped["choice"])
    shared_head, domain_heads = choice_head_modules(
        torch,
        hidden_size=hidden_size,
        rank=config.choice_rank,
        init_std=config.choice_init_std,
        domains=domains,
    )
    shared_head = shared_head.to(device)
    for name in domains:  # drawn after shared: legacy untouched
        domain_heads[name] = domain_heads[name].to(device)
    log_temperature = torch.zeros((), requires_grad=True, device=device)
    domain_params = [parameter for name in domains for parameter in domain_heads[name].parameters()]
    optimizer = torch.optim.AdamW(
        [
            *model.parameters(),
            *shared_head.parameters(),
            *domain_params,
            log_temperature,
        ],
        lr=config.learning_rate,
    )

    def encode(texts: list[str]) -> Any:
        batch = tokenizer(
            texts, padding=True, truncation=True, max_length=config.max_len, return_tensors="pt"
        ).to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return pooled.float()

    def batch_loss(kind: str, batch: list[dict[str, Any]]) -> Any:
        temperature = torch.exp(log_temperature) + 1e-3
        states = encode([record["state"] for record in batch])
        questions = encode([question_text(record) for record in batch])
        if kind == "noul":
            similarity = F.cosine_similarity(questions, states, dim=-1)
            probabilities = torch.sigmoid(similarity / temperature)
            targets = torch.tensor([float(record["target"]) for record in batch], device=device)
            return ((probabilities - targets) ** 2).mean()
        criterion_rows = [text for record in batch for text in criterion_texts(record)]
        criteria = encode(criterion_rows)
        # ADR-0016 §1: every choice record trains the shared head; one carrying a `domain`
        # field additionally trains its own domain head. Records without a domain never touch
        # a domain head, so a legacy config's gradient stream is unchanged.
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
                probabilities = torch.softmax(
                    base + residual / math.sqrt(config.choice_rank), dim=0
                )
                total = total + ((probabilities - one_hot) ** 2).sum()
            slot = slots.get(index)
            if slot is not None:
                head, context_row = slot
                residual = (context_row * head.project(block)).sum(dim=-1)
                scaled = base + residual / math.sqrt(config.choice_rank)
                total = total + ((torch.softmax(scaled, dim=0) - one_hot) ** 2).sum()
            if kind == "score":
                probabilities = torch.softmax(base, dim=0)
                total = total + ((probabilities - one_hot) ** 2).sum()
        return total / len(batch)

    def evaluate_loss(items: list[dict[str, Any]]) -> float:
        if not items:
            return float("nan")
        model.eval()  # type: ignore[attr-defined]
        total, count = 0.0, 0
        with torch.no_grad():
            for kind in ("noul", "choice", "score"):
                subset = [record for record in items if record["type"] == kind]
                for start in range(0, len(subset), config.batch_size):
                    chunk = subset[start : start + config.batch_size]
                    total += float(batch_loss(kind, chunk).item()) * len(chunk)
                    count += len(chunk)
        return total / count if count else float("nan")

    epochs_run = 0
    for epoch in range(config.epochs):
        model.train()  # type: ignore[attr-defined]
        optimizer.zero_grad(set_to_none=True)
        accumulations = 0
        # per-primitive epoch loss kept as detached tensors: one sync per epoch, not per batch
        epoch_loss: dict[str, Any] = dict.fromkeys(("noul", "choice", "score"), 0.0)
        epoch_seen = dict.fromkeys(epoch_loss, 0)
        for kind in ("noul", "choice", "score"):
            # Mixed-domain data must not arrive domain by domain: every optimizer step would then
            # be one (domain, primitive) gradient (B-5, see `shuffled`).
            items = shuffled(grouped[kind], seed=f"{config.seed}:{epoch}:{kind}")
            for start in range(0, len(items), config.batch_size):
                chunk = items[start : start + config.batch_size]
                loss = batch_loss(kind, chunk) / config.grad_accum
                # keep the per-primitive epoch loss as a detached tensor: one sync per epoch,
                # not one per batch
                epoch_loss[kind] = epoch_loss[kind] + loss.detach() * config.grad_accum * len(chunk)
                epoch_seen[kind] += len(chunk)
                loss.backward()
                accumulations += 1
                if accumulations % config.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        if accumulations % config.grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        epochs_run += 1
        # Train loss per primitive, every epoch: it costs nothing (the values are already computed)
        # and it is the only way to see whether a primitive is still improving when the run ends.
        means = " ".join(
            f"{kind}={epoch_loss[kind] / max(1, epoch_seen[kind]):.4f}" for kind in epoch_loss
        )
        print(f"epoch {epochs_run}/{config.epochs} train_loss {means}", flush=True)

    output_dir = Path(config.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))

    languages = tuple(sorted({str(record["lang"]) for record in train_records}))
    bank_payload = choice_head_payload(
        shared=scorer_payload(shared_head, rank=config.choice_rank),
        # Keyed bank (ADR-0016 §1): shared + one head per training domain, each carrying the
        # signature terms the runtime gate matches questions against. With no domains the
        # payload is the legacy single-scorer shape — byte-identical to what this writer
        # always produced, so existing configs reproduce today's artifact.
        domains={
            name: scorer_payload(domain_heads[name], rank=config.choice_rank)
            for name in domains  # sorted, deterministic
        },
        languages=languages,
    )
    (output_dir / "choice_head.json").write_text(
        json.dumps(bank_payload, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "finetune_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    fitted_temperature = float(torch.exp(log_temperature).item())
    (output_dir / "temperature.json").write_text(
        json.dumps(
            {"temperature": fitted_temperature, "val_loss": evaluate_loss(val_records)},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report["train_records"] = len(train_records)
    report["val_records"] = len(val_records)
    report["epochs"] = epochs_run
    report["batch_size"] = config.batch_size
    report["target_modules"] = target_modules
    report["choice_rank"] = config.choice_rank
    report["checkpoint"] = str(output_dir)
    report["temperature"] = fitted_temperature
    report["val_loss"] = evaluate_loss(val_records)
    return report


def question_text(record: dict[str, Any]) -> str:
    """The text the encoder sees as the question: instructions, JSON-encoded when not a string."""
    instructions = record["instructions"]
    if isinstance(instructions, str):
        return instructions
    return json.dumps(instructions, ensure_ascii=False)


def choice_domains(records: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    """Sorted domain keys carried by ``choice`` records (ADR-0016 §1).

    Empty for data without ``domain`` fields — the legacy single-domain case, where the run
    trains and ships only the shared head.
    """
    return tuple(
        sorted(
            {
                str(record["domain"])
                for record in records
                if record.get("domain") is not None and record.get("type") == "choice"
            }
        )
    )


def require_committed_domains(domains: Iterable[str]) -> tuple[str, ...]:
    """Fail fast (before training, L-011) when records name a domain with no committed file."""
    unknown = [name for name in domains if name not in DOMAINS]
    if unknown:
        known = ", ".join(sorted(DOMAINS))
        raise SystemExit(f"unknown domain(s): {', '.join(unknown)}; committed domains: {known}")
    return tuple(domains)


def scorer_payload(head: Any, *, rank: int) -> dict[str, Any]:
    """The ``choice_head.json`` scorer dict ``{rank, w1, w2}`` (the L-002 asset shape).

    The single definition of the asset's bytes, shared by the joint trainer and the
    frozen-trunk fitter (``training/fit_choice_bank.py``) so neither can drift from the
    runtime's :class:`tachyone.backends.encoder.ChoiceScorer`.
    """
    return {
        "rank": rank,
        "w1": head.w1.weight.detach().cpu().tolist(),
        "w2": head.w2.weight.detach().cpu().tolist(),
    }


def choice_head_payload(
    shared: dict[str, Any],
    domains: Mapping[str, dict[str, Any]],
    languages: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the ``choice_head.json`` payload: keyed bank when domains exist, else legacy.

    With no ``domains`` the payload *is* the shared scorer dict, so a run over data without
    ``domain`` fields writes byte-identically to the pre-ADR-0016 writer (ADR-0016 §1).
    With domains, each head ships its signature terms so the runtime gate can match
    questions without the committed data files (ADR-0016 §2.2).
    """
    if not domains:
        return shared
    return {
        "shared": shared,
        "domains": {
            name: {
                "head": head,
                "signatures": list(domain_signatures(name, languages)),
            }
            for name, head in sorted(domains.items())
        },
    }


def criterion_texts(record: dict[str, Any]) -> list[str]:
    """One text per option/level, matching what the runtime encoder feeds the heads."""
    criteria = record["criteria"]
    if isinstance(criteria, dict):
        return [
            f"{key}: {value}" if isinstance(value, str) and value else str(key)
            for key, value in criteria.items()
        ]
    return [str(level) for level in criteria]


def choice_target_index(record: dict[str, Any]) -> int:
    """Index of the target option for a record whose ``target`` names its key (not its index)."""
    keys = list(record["criteria"])
    return keys.index(record["target"])


def run(config: FinetuneConfig, *, dry_run: bool = False) -> dict[str, Any]:
    """Validate data and either summarize (dry run) or train (needs the ``train`` extra)."""
    report: dict[str, Any] = {
        "config": config.to_dict(),
        "dataset": summarize_dataset(config.data_path),
    }
    # Resolve and validate the bank's domain keys before anything else — including
    # `--dry-run`, whose whole job is to catch a bad config/data cheaply (L-011).
    domains = choice_domains(read_records(config.data_path, limit=config.max_records))
    require_committed_domains(domains)
    report["choice_domains"] = list(domains)
    if dry_run:
        report["mode"] = "dry-run"
        return report
    report["mode"] = "train"
    return _train(config, report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tachyone-finetune", description=__doc__)
    parser.add_argument("--config", required=True, help="path to a JSON FinetuneConfig")
    parser.add_argument("--dry-run", action="store_true", help="validate config and data only")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = run(load_config(args.config), dry_run=args.dry_run)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
