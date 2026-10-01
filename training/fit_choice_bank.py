"""Fit the per-domain `choice` head bank on a **frozen** trunk — the ADR-0016 isolate.

The first experiment of ADR-0016 §5 is deliberately cheap: the run-5 trunk is loaded once
through the *runtime* encoder (so the fit sees exactly what inference sees), every `choice`
record's state, question and criterion texts are encoded a single time, and two arms are fitted
from that one cache:

* **control** — today's single shared head, refit with the same budget and warm start;
* **bank** — shared head plus one head per domain, every head warm-started from the trunk's own
  `choice_head.json`, so at step 0 the bank answers exactly like the artifact it is measured
  against (a wrong gate can at worst reproduce today's numbers).

Only the heads move — no gradient reaches the trunk — so the head-capacity / gradient-share
axis is isolated before any joint retrain is considered. Each arm is written as its own
loadable adapter dir: the trunk's adapter files copied bit-for-bit, the arm's `choice_head.json`
(legacy shared-only for the control, keyed bank for the bank arm) and `choice_bank_fit.json`,
the recipe that produced it (L-011). Torch/transformers are imported lazily behind the `train`
extra; `--dry-run` validates config and data without them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from tachyone.backends.encoder import CHOICE_HEAD_ASSET, load_encoder
from tachyone.router import CheckpointInfo
from training.finetune_rlcd import (
    choice_domains,
    choice_head_modules,
    choice_head_payload,
    choice_target_index,
    criterion_texts,
    question_text,
    read_records,
    require_committed_domains,
    scorer_payload,
    shuffled,
    split_records,
    summarize_dataset,
)

_TRAIN_HINT = "the train extra is required for fitting the choice-head bank: uv sync --extra train"

#: Adapter files copied from the frozen trunk into every arm dir (the trunk itself does not move).
_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
#: Recipe asset each arm ships beside its head — the artifact carries its own recipe (L-011).
FIT_RECIPE_ASSET = "choice_bank_fit.json"
#: Texts per encoder call while building the cache — padding efficiency, not a model knob.
_ENCODE_BATCH = 256

_DEFAULT_MODELS_DIR = os.path.join(os.path.expanduser("~"), ".cache", "tachyone", "models")


@dataclass(frozen=True, slots=True)
class FitBankConfig:
    """Hyperparameters and paths for one frozen-trunk bank fit (the isolate)."""

    model_id: str = "answerdotai/ModernBERT-large"
    data_path: str = "data/train_en_domains.jsonl"
    trunk_adapter: str = "checkpoints/en_domains_r5"
    control_dir: str = "checkpoints/en_domains_bank_ctrl"
    bank_dir: str = "checkpoints/en_domains_bank"
    seed: int = 2
    epochs: int = 4
    batch_size: int = 64
    learning_rate: float = 1e-3
    choice_rank: int = 128
    max_len: int = 512
    val_split: float = 0.1
    max_records: int | None = None
    device: str = "auto"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FitBankConfig:
        known = {field.name for field in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> FitBankConfig:
    """Load a :class:`FitBankConfig` from a JSON file."""
    return FitBankConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def trunk_shared_head(trunk_adapter: str | Path) -> dict[str, Any]:
    """The trunk's shared scorer payload, structurally validated (fail before fitting — L-011).

    Accepts both asset shapes: the legacy single scorer that every published adapter ships and
    the keyed bank, whose ``shared`` entry is the trunk's head. Only the directory form is
    usable — the isolate needs bytes on this machine, not a Hub download.
    """
    source = Path(trunk_adapter)
    if not source.is_dir():
        raise SystemExit(f"trunk adapter must be a local directory, got: {trunk_adapter}")
    path = source / CHOICE_HEAD_ASSET
    if not path.is_file():
        raise SystemExit(f"trunk ships no {CHOICE_HEAD_ASSET}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"unreadable {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must be a JSON object, found {type(payload).__name__}")
    if "shared" in payload or "domains" in payload:
        shared = payload.get("shared")
        if not isinstance(shared, dict):
            raise SystemExit(f"{path} is a keyed bank without a usable 'shared' head")
    else:
        shared = payload
    _validate_scorer(shared, path)
    return shared


def _validate_scorer(scorer: Any, path: Path) -> None:
    """Check the scorer's shape — the isolate fails before it encodes, not while it fits."""
    if not isinstance(scorer, dict):
        raise SystemExit(f"{path}: scorer must be a JSON object, found {type(scorer).__name__}")
    rank = scorer.get("rank")
    w1, w2 = scorer.get("w1"), scorer.get("w2")
    if not isinstance(rank, int) or not isinstance(w1, list) or not isinstance(w2, list):
        raise SystemExit(f"{path}: scorer needs an integer 'rank' and 'w1'/'w2' matrices")
    if not w1 or not w2 or len(w1) != rank or len(w2) != rank:
        raise SystemExit(f"{path}: 'rank' {rank} disagrees with {len(w1)}/{len(w2)} rows")
    if not all(isinstance(row, list) and row for row in (*w1, *w2)):
        raise SystemExit(f"{path}: 'w1'/'w2' must be non-empty row matrices")
    if len(w1[0]) != 2 * len(w2[0]):
        raise SystemExit(
            f"{path}: w1 rows ({len(w1[0])}) must be twice w2 rows ({len(w2[0])}) — "
            "the context is [state; question]"
        )


def write_arm(
    out_dir: str | Path,
    *,
    trunk: str | Path,
    shared: Mapping[str, Any],
    domains: Mapping[str, Mapping[str, Any]],
    languages: Sequence[str],
    recipe: Mapping[str, Any],
) -> Path:
    """Materialize one arm as a loadable adapter dir and return its path.

    The trunk is frozen, so an arm copies its adapter files bit-for-bit and differs only in the
    head asset it writes — legacy shared-only for the control (an empty ``domains`` makes
    :func:`training.finetune_rlcd.choice_head_payload` emit the pre-ADR-0016 shape) and the
    keyed bank for the bank arm — plus ``choice_bank_fit.json``, its own recipe (L-011).
    """
    source, target = Path(trunk), Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name in _ADAPTER_FILES:
        candidate = source / name
        if not candidate.is_file():
            raise SystemExit(f"trunk adapter is missing {name}: {candidate}")
        shutil.copy2(candidate, target / name)
    payload = choice_head_payload(
        dict(shared),
        {name: dict(head) for name, head in domains.items()},
        languages,
    )
    (target / CHOICE_HEAD_ASSET).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    (target / FIT_RECIPE_ASSET).write_text(
        json.dumps(dict(recipe), indent=2, sort_keys=True), encoding="utf-8"
    )
    return target


def fit_recipe(
    config: FitBankConfig,
    *,
    arm: str,
    domains: Sequence[str],
    languages: Sequence[str],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """The ``choice_bank_fit.json`` an arm ships: what produced *this* artifact, in full (L-011)."""
    return {
        "arm": arm,
        "config": config.to_dict(),
        "trunk_adapter": config.trunk_adapter,
        "domains": list(domains),
        "languages": list(languages),
        "fit": dict(metrics),
    }


@dataclass(slots=True)
class ChoiceCache:
    """Frozen-trunk embeddings for one set of `choice` records — the isolate's single pass."""

    #: ``[N, H]`` state and question embeddings, one row per record.
    states: Any
    questions: Any
    #: ``[M, H]`` criterion embeddings; record ``i`` owns rows ``[offsets[i], offsets[i + 1])``.
    criteria: Any
    offsets: list[int]
    sizes: list[int]
    targets: list[int]
    domains: list[str]

    def __len__(self) -> int:
        return len(self.sizes)


def _encode_matrix(
    texts: Sequence[Any],
    encode: Callable[[list[str]], list[list[float]]],
    torch: Any,
) -> Any:
    """Encode ``texts`` in chunks into one float32 CPU tensor.

    Chunking is deliberate: the runtime encoder returns Python floats, and tens of thousands of
    rows held as objects cost gigabytes before a tensor exists.
    """
    chunks: list[Any] = []
    for start in range(0, len(texts), _ENCODE_BATCH):
        rows = encode([str(text) for text in texts[start : start + _ENCODE_BATCH]])
        chunks.append(torch.tensor(rows, dtype=torch.float32))
    if not chunks:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(chunks)


def encode_choice(
    records: Sequence[dict[str, Any]],
    encode: Callable[[list[str]], list[list[float]]],
    torch: Any,
) -> ChoiceCache:
    """Encode every state, question and criterion text **once** (ADR-0016 §5)."""
    sizes = [len(criterion_texts(record)) for record in records]
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    states = _encode_matrix([record["state"] for record in records], encode, torch)
    questions = _encode_matrix([question_text(record) for record in records], encode, torch)
    criteria = _encode_matrix(
        [text for record in records for text in criterion_texts(record)], encode, torch
    )
    targets = [
        int(record["target"]) if isinstance(record["target"], int) else choice_target_index(record)
        for record in records
    ]
    return ChoiceCache(
        states=states,
        questions=questions,
        criteria=criteria,
        offsets=offsets,
        sizes=sizes,
        targets=targets,
        domains=[str(record.get("domain", "")) for record in records],
    )


def warm_start(head: Any, scorer: Mapping[str, Any], torch: Any, *, name: str) -> None:
    """Overwrite a head's weights with the trunk's shared head — the isolate's step-0 state.

    Every arm starts from the artifact it is measured against, so a fit that learns nothing
    reproduces today's numbers instead of inventing a worse one.
    """
    for key in ("w1", "w2"):
        weight = getattr(head, key).weight
        source = torch.tensor(scorer[key], dtype=weight.dtype, device=weight.device)
        if tuple(source.shape) != tuple(weight.shape):
            raise SystemExit(
                f"trunk head '{key}' shape {tuple(source.shape)} does not fit the '{name}' head "
                f"({tuple(weight.shape)}) — hidden size or choice_rank disagrees with the trunk"
            )
        weight.data.copy_(source)


def arm_loss(
    torch: Any,
    F: Any,
    *,
    cache: ChoiceCache,
    indices: Sequence[int],
    shared: Any,
    heads: Mapping[str, Any],
    log_temperature: Any,
    rank: int,
    device: str,
    bank: bool,
) -> Any:
    """One batch's loss, mirroring the joint trainer's `choice` branch term for term.

    Both arms pay the shared-head term; the bank arm pays its domain head's term as well, so
    the *only* difference between the arms is the capacity and gradient share the bank adds.
    """
    temperature = torch.exp(log_temperature) + 1e-3
    states = cache.states[list(indices)].to(device)
    questions = cache.questions[list(indices)].to(device)
    rows = [
        row for index in indices for row in range(cache.offsets[index], cache.offsets[index + 1])
    ]
    criteria = cache.criteria[rows].to(device)
    shared_context = shared.context(states, questions)
    batch_domains = {cache.domains[index] for index in indices if cache.domains[index] in heads}
    contexts = {name: heads[name].context(states, questions) for name in batch_domains}
    total = torch.zeros((), device=device)
    offset = 0
    for position, index in enumerate(indices):
        count = cache.sizes[index]
        block = criteria[offset : offset + count]
        offset += count
        state = states[position].expand(count, -1)
        question = questions[position].expand(count, -1)
        base = (
            F.cosine_similarity(block, state, dim=-1) + F.cosine_similarity(block, question, dim=-1)
        ) / temperature
        one_hot = torch.zeros(count, device=device)
        one_hot[cache.targets[index]] = 1.0
        residual = (shared_context[position] * shared.project(block)).sum(dim=-1)
        probabilities = torch.softmax(base + residual / math.sqrt(rank), dim=0)
        total = total + ((probabilities - one_hot) ** 2).sum()
        name = cache.domains[index]
        if name in contexts:
            context_row = contexts[name][position]
            domain_residual = (context_row * heads[name].project(block)).sum(dim=-1)
            domain_probabilities = torch.softmax(base + domain_residual / math.sqrt(rank), dim=0)
            total = total + ((domain_probabilities - one_hot) ** 2).sum()
    return total / max(1, len(indices))


def _evaluate_loss(
    torch: Any, F: Any, *, cache: ChoiceCache, batch_size: int, **kwargs: Any
) -> float:
    """Mean arm loss over ``cache`` with no gradients (the held-out split)."""
    if not len(cache):
        return float("nan")
    total, seen = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(cache), batch_size):
            indices = list(range(start, min(start + batch_size, len(cache))))
            loss = arm_loss(torch, F, cache=cache, indices=indices, **kwargs)
            total += float(loss) * len(indices)
            seen += len(indices)
    return total / max(1, seen)


def fit_arm(
    torch: Any,
    F: Any,
    *,
    arm: str,
    bank: bool,
    cache: ChoiceCache,
    val_cache: ChoiceCache,
    config: FitBankConfig,
    trunk_head: Mapping[str, Any],
    domains: Sequence[str],
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit one arm from the frozen encodings; returns ``(metrics, head payloads)``."""
    hidden_size = int(cache.states.shape[1])
    # Both arms start from the same stream; the warm start then overwrites every weight, so
    # init_std 0 documents that initialization itself is irrelevant here.
    torch.manual_seed(config.seed)
    shared, domain_heads = choice_head_modules(
        torch,
        hidden_size=hidden_size,
        rank=config.choice_rank,
        init_std=0.0,
        domains=domains if bank else (),
    )
    warm_start(shared, trunk_head, torch, name="shared")
    for name, head in domain_heads.items():
        warm_start(head, trunk_head, torch, name=name)
    shared = shared.to(device)
    domain_heads = {name: head.to(device) for name, head in domain_heads.items()}
    log_temperature = torch.zeros((), requires_grad=True, device=device)
    parameters: list[Any] = [*shared.parameters(), log_temperature]
    parameters.extend(
        parameter for head in domain_heads.values() for parameter in head.parameters()
    )
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)

    loss_kwargs = {
        "shared": shared,
        "heads": domain_heads,
        "log_temperature": log_temperature,
        "rank": config.choice_rank,
        "device": device,
        "bank": bank,
    }
    epochs_run, train_loss = 0, float("nan")
    for epoch in range(config.epochs):
        order = shuffled(list(range(len(cache))), seed=f"{config.seed}:{epoch}:choice")
        optimizer.zero_grad(set_to_none=True)
        total, seen = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            loss = arm_loss(torch, F, cache=cache, indices=indices, **loss_kwargs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(indices)
            seen += len(indices)
        epochs_run += 1
        train_loss = total / max(1, seen)
        print(f"{arm} epoch {epochs_run}/{config.epochs} train_loss {train_loss:.4f}", flush=True)

    metrics = {
        "arm": arm,
        "epochs": epochs_run,
        "train_records": len(cache),
        "val_records": len(val_cache),
        "train_loss": train_loss,
        "val_loss": _evaluate_loss(
            torch, F, cache=val_cache, batch_size=config.batch_size, **loss_kwargs
        ),
        "temperature": float(torch.exp(log_temperature).item()),
    }
    payloads = {
        "shared": scorer_payload(shared, rank=config.choice_rank),
        "domains": {
            name: scorer_payload(head, rank=config.choice_rank)
            for name, head in domain_heads.items()
        },
    }
    return metrics, payloads


def _resolve_device(torch: Any, device: str) -> str:
    if device in {"cpu", "cuda", "mps"}:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _fit(
    config: FitBankConfig,
    report: dict[str, Any],
    *,
    trunk_head: Mapping[str, Any],
    records: Sequence[dict[str, Any]],
    domains: Sequence[str],
    languages: Sequence[str],
) -> dict[str, Any]:
    """Encode once through the frozen trunk, fit both arms from the cache, write both dirs."""
    try:
        import torch  # pyright: ignore[reportMissingImports]
        import torch.nn.functional as F  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(_TRAIN_HINT) from exc

    torch.manual_seed(config.seed)
    device = _resolve_device(torch, config.device)
    train_records, val_records = split_records(
        list(records), val_fraction=config.val_split, seed=config.seed
    )
    if not train_records:
        raise SystemExit(
            f"no choice records left to fit after the {config.val_split} validation split"
        )

    info = CheckpointInfo(
        id="fit-bank",
        languages=["*"],
        context=config.max_len,
        size_params=0,
        base_model=config.model_id,
        adapter=config.trunk_adapter,
    )
    encode = load_encoder(info, models_dir=_DEFAULT_MODELS_DIR, device=config.device)
    print(
        f"encode: {len(train_records)} train + {len(val_records)} val choice records "
        f"through the frozen trunk ({config.trunk_adapter})",
        flush=True,
    )
    train_cache = encode_choice(train_records, encode, torch)
    val_cache = encode_choice(val_records, encode, torch)

    arms: dict[str, Any] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for arm, bank in (("control", False), ("bank", True)):
        metrics, payload = fit_arm(
            torch,
            F,
            arm=arm,
            bank=bank,
            cache=train_cache,
            val_cache=val_cache,
            config=config,
            trunk_head=trunk_head,
            domains=domains,
            device=device,
        )
        arms[arm] = metrics
        payloads[arm] = payload

    write_arm(
        config.control_dir,
        trunk=config.trunk_adapter,
        shared=payloads["control"]["shared"],
        domains={},
        languages=languages,
        recipe=fit_recipe(
            config, arm="control", domains=(), languages=languages, metrics=arms["control"]
        ),
    )
    write_arm(
        config.bank_dir,
        trunk=config.trunk_adapter,
        shared=payloads["bank"]["shared"],
        domains=payloads["bank"]["domains"],
        languages=languages,
        recipe=fit_recipe(
            config, arm="bank", domains=domains, languages=languages, metrics=arms["bank"]
        ),
    )
    for arm, directory in (("control", config.control_dir), ("bank", config.bank_dir)):
        arms[arm]["dir"] = str(directory)
        Path(directory, "temperature.json").write_text(
            json.dumps(
                {"temperature": arms[arm]["temperature"], "val_loss": arms[arm]["val_loss"]},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    report["arms"] = arms
    report["encoded_records"] = len(train_cache) + len(val_cache)
    return report


def run(config: FitBankConfig, *, dry_run: bool = False) -> dict[str, Any]:
    """Validate config, data and trunk; fit both arms unless ``dry_run``."""
    report: dict[str, Any] = {
        "config": config.to_dict(),
        "dataset": summarize_dataset(config.data_path, limit=config.max_records),
    }
    records = [
        record
        for record in read_records(config.data_path, limit=config.max_records)
        if record["type"] == "choice"
    ]
    domains = choice_domains(records)
    require_committed_domains(domains)
    if not domains:
        raise SystemExit(
            "fit_choice_bank needs `choice` records carrying a `domain` field; "
            f"{config.data_path} has none, so the bank would have no keys to fit"
        )
    trunk_head = trunk_shared_head(config.trunk_adapter)
    if int(trunk_head["rank"]) != config.choice_rank:
        raise SystemExit(
            f"choice_rank {config.choice_rank} disagrees with the trunk head's rank "
            f"{trunk_head['rank']} ({Path(config.trunk_adapter) / CHOICE_HEAD_ASSET}) — "
            "the arms warm-start from the trunk, so the rank must match it"
        )
    languages = sorted({str(record["lang"]) for record in records})
    report["choice_domains"] = list(domains)
    report["languages"] = languages
    report["trunk"] = {
        "adapter": config.trunk_adapter,
        "choice_head_rank": trunk_head["rank"],
    }
    report["plan"] = {
        "control": config.control_dir,
        "bank": config.bank_dir,
        "trunk": config.trunk_adapter,
        "records": len(records),
    }
    if dry_run:
        report["mode"] = "dry-run"
        return report
    report["mode"] = "fit"
    return _fit(
        config,
        report,
        trunk_head=trunk_head,
        records=records,
        domains=tuple(domains),
        languages=tuple(languages),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tachyone-fit-choice-bank", description=__doc__)
    parser.add_argument("--config", required=True, help="path to a JSON FitBankConfig")
    parser.add_argument("--dry-run", action="store_true", help="validate config and data only")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = run(load_config(args.config), dry_run=args.dry_run)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
