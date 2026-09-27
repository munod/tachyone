"""Head-to-head comparison: Tachyone vs an open System-One scorer vs a small LLM.

Three engines answer **the same rows of the same public dataset**
(``pngwn/system-one-decisions``) through one set of metrics, so the numbers are comparable by
construction rather than by narrative:

- ``tachyone``  — the local encoder backend, through ``tachyone.wire.answer``.
- ``systemone`` — `pngwn/system-one-qwen3.5-4b-scorer` (CC-BY-NC-4.0), loaded from a local
  download through **its own** ``system_one.py``. Nothing third-party is vendored in this
  repository: the script is imported from a path you pass in, and it is never redistributed.
- ``llm``       — any OpenAI-compatible server (``llama-server``), through Tachyone's own
  ``llm`` backend, so schema failures are counted against the engine that produced them.

Every engine is scored with the *same* implementation of accuracy, 10-bin ECE and Brier, on
rows capped per task family (``--per-task``, default 64 — the same cap the scorer's model card
uses), with temperature fitted on the capped **validation** split and applied to test.

Two evaluation sets, because a head-to-head on someone else's training data proves little:

- ``--format probe`` (default) — a public dataset chosen by ``--probe``:
  ``system-one-decisions`` (default) is the peer scorer's **own** distribution (it trained on its
  train split; Tachyone is zero-shot there), while ``typed-decisions``, ``massive`` and ``xnli``
  are the public probes of BACKLOG **B-7**, loaded by ``benchmarks/probes.py``.
- ``--format jsonl`` — Tachyone's eval records, where **we** are in-domain and the peer and the
  LLMs are visitors.

One process per engine, then render:

    uv pip install pyarrow                    # parquet reader, lazily imported
    python -m benchmarks.compare run --engine tachyone \
        --data data/system-one-decisions --per-task 64 \
        --out benchmarks/results/compare_tachyone.json
    python -m benchmarks.compare run --engine tachyone --format jsonl \
        --data data/eval_en.jsonl --val-data data/train_en.jsonl --per-task 64 \
        --out benchmarks/results/compare_home_tachyone.json
    python -m benchmarks.compare run --engine systemone \
        --systemone-script /path/to/scorer/system_one.py \
        --systemone-base /path/to/base --systemone-adapter /path/to/scorer \
        --data data/system-one-decisions --per-task 64 \
        --out benchmarks/results/compare_systemone.json
    python -m benchmarks.compare run --engine llm \
        --llm-base-url http://127.0.0.1:8081/v1 --llm-model small.gguf \
        --data data/system-one-decisions --per-task 16 \
        --out benchmarks/results/compare_llm_small.json
    python -m benchmarks.compare render benchmarks/results/compare_*.json

Results and method notes are published in ``benchmarks/report.md`` and ``docs/compare.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
#: Rows per task family (``--per-task``); 64 matches the scorer's published evaluation.
DEFAULT_PER_TASK = 64
DEFAULT_WARMUP = 3
#: Datasets ``--format probe`` accepts. The first is the peer scorer's own distribution; the
#: other three are the public probes of BACKLOG B-7, loaded by ``benchmarks/probes.py``.
PROBE_CHOICES = ("system-one-decisions", "typed-decisions", "massive", "xnli")
#: Split used to fit temperature when ``--val-split`` is omitted: each dataset's own development
#: split, so the fit never touches the split being reported.
PROBE_VAL_SPLITS = {
    "system-one-decisions": "val",
    "typed-decisions": "train",
    "massive": "dev",
    "xnli": "validation",
}
#: Repositories the probes are read from (licences and citations: ``benchmarks/probes.py``).
PROBE_REPOS = {
    "system-one-decisions": "pngwn/system-one-decisions",
    "typed-decisions": "LocalLLaMA/typed-decisions",
    "massive": "AmazonScience/massive",
    "xnli": "facebook/xnli",
}
#: 0.25 .. 19.75 in 0.05 steps. Their sweep stops at 6.0; a wider, still-symmetric grid avoids
#: capping an over-confident engine at the boundary (Tachyone needs T > 6 on this probe).
TEMPERATURE_GRID = [round(0.25 + index * 0.05, 2) for index in range(396)]


# --------------------------------------------------------------------------- data
def _read_parquet(path: Path) -> list[dict[str, Any]]:
    """Read one parquet file into plain dicts (pyarrow is an opt-in benchmark dependency)."""
    try:
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - depends on the local environment
        raise SystemExit(
            "reading the public evaluation split needs pyarrow: `uv pip install pyarrow`"
        ) from exc
    return pq.read_table(path).to_pylist()


def load_rows(data: Path, split: str, per_task: int, limit: int = 0) -> list[dict[str, Any]]:
    """Load a split of ``pngwn/system-one-decisions``, optionally capped per task family."""
    directory = data / "data"
    files = sorted(directory.glob(f"{split}-*.parquet"))
    if not files:
        raise SystemExit(
            f"no {split!r} parquet under {directory}; download it first:\n"
            f"  hf download pngwn/system-one-decisions --repo-type dataset --local-dir {data}"
        )
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(_read_parquet(path))
    return _cap(rows, per_task, limit)


def load_tachyone_records(path: Path, per_task: int, limit: int = 0) -> list[dict[str, Any]]:
    """Load Tachyone's own JSONL eval records (``training/generate_data.py`` output).

    Each record keeps its **native** question type, so Tachyone answers it exactly as it was
    trained to (``noul``/``choice``/``score``) while a peer scorer that only knows
    ``(state, question, options)`` sees the same information flattened into options: for a
    ``noul`` the two criteria descriptions become a two-option set.
    """
    if not path.exists():
        raise SystemExit(
            f"{path} not found; regenerate it with training/generate_data.py (see AGENTS.md)"
        )
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(_record_to_row(json.loads(line)))
    return _cap(rows, per_task, limit)


def _record_to_row(record: dict[str, Any]) -> dict[str, Any]:
    kind = str(record["type"])
    instructions = record["instructions"]
    criteria = record["criteria"]
    if kind == "noul":
        assert isinstance(criteria, dict)
        options = [str(criteria.get("false", "false")), str(criteria.get("true", "true"))]
        answer = int(record["target"])
        wire: dict[str, Any] = {"type": "noul", "instructions": instructions, "criteria": criteria}
    elif kind == "choice":
        assert isinstance(criteria, dict)
        labels = list(criteria)
        options = [str(label) for label in labels]
        target = record["target"]
        answer = labels.index(str(target)) if str(target) in labels else int(target)
        wire = {"type": "choice", "instructions": instructions, "criteria": criteria}
    else:
        assert isinstance(criteria, list)
        options = [str(level) for level in criteria]
        answer = int(record["target"])
        wire = {"type": "score", "instructions": instructions, "criteria": criteria}
    return {
        "state": record["state"],
        "task": kind,
        "question": str(instructions),
        "options": options,
        "answer_index": answer,
        "wire": wire,
    }


def _cap(rows: list[dict[str, Any]], per_task: int, limit: int) -> list[dict[str, Any]]:
    """Take the first ``per_task`` rows of each family (stable order), then an optional cap."""
    if per_task and per_task > 0:
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_task[str(row["task"])].append(row)
        rows = [row for task in sorted(by_task) for row in by_task[task][:per_task]]
    if limit and limit > 0:
        rows = rows[:limit]
    if not rows:
        raise SystemExit("no rows loaded")
    return rows


# ------------------------------------------------------------------------ metrics
def _softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _probabilities_from_log(log_probs: list[float], temperature: float) -> list[float]:
    """Rescale an already-normalized distribution by ``temperature`` (softmax(log p / T))."""
    return _softmax([value / temperature for value in log_probs])


def accuracy(probabilities: list[list[float] | None], answers: list[int]) -> float:
    """Top-1 accuracy; a failed answer counts as wrong."""
    correct = 0
    for probs, answer in zip(probabilities, answers, strict=True):
        if probs is not None and max(range(len(probs)), key=probs.__getitem__) == answer:
            correct += 1
    return correct / len(answers) if answers else float("nan")


def expected_calibration_error(
    probabilities: list[list[float]], answers: list[int], bins: int = 10
) -> float:
    """10-bin ECE over the answered rows, using the selected mass as confidence."""
    if not probabilities:
        return float("nan")
    confidence = [max(probs) for probs in probabilities]
    correct = [
        max(range(len(probs)), key=probs.__getitem__) == answer
        for probs, answer in zip(probabilities, answers, strict=True)
    ]
    edges = [index / bins for index in range(bins + 1)]
    total = 0.0
    for index in range(bins):
        members = [
            position
            for position, value in enumerate(confidence)
            if edges[index] < value <= edges[index + 1]
        ]
        if not members:
            continue
        hit_rate = sum(correct[position] for position in members) / len(members)
        mean_conf = statistics.fmean(confidence[position] for position in members)
        total += (len(members) / len(confidence)) * abs(hit_rate - mean_conf)
    return total


def brier(probabilities: list[list[float]], answers: list[int]) -> float:
    """Multi-class Brier score over the answered rows."""
    if not probabilities:
        return float("nan")
    total = 0.0
    for probs, answer in zip(probabilities, answers, strict=True):
        total += sum(
            (value - (1.0 if index == answer else 0.0)) ** 2 for index, value in enumerate(probs)
        )
    return total / len(probabilities)


def fit_temperature(
    distributions: list[list[float]], answers: list[int], grid: list[float] | None = None
) -> float:
    """Grid-search the scalar temperature minimizing NLL on the validation split.

    Works from probabilities alone: ``softmax(log p / T)`` equals the logit rescaling, because
    softmax is shift-invariant.
    """
    log_probs = [[math.log(max(value, 1e-12)) for value in probs] for probs in distributions]
    best_temperature, best_nll = 1.0, float("inf")
    for temperature in grid or TEMPERATURE_GRID:
        nll = 0.0
        for logs, answer in zip(log_probs, answers, strict=True):
            scaled = _probabilities_from_log(logs, temperature)
            nll -= math.log(max(scaled[answer], 1e-12))
        if nll < best_nll:
            best_nll, best_temperature = nll, temperature
    return best_temperature


def _summary(probabilities: list[list[float] | None], answers: list[int]) -> dict[str, float | int]:
    answered = [
        (probs, answer) for probs, answer in zip(probabilities, answers, strict=True) if probs
    ]
    return {
        "n": len(probabilities),
        "answered": len(answered),
        "accuracy": accuracy(probabilities, answers),
        "ece": expected_calibration_error([p for p, _ in answered], [a for _, a in answered]),
        "brier": brier([p for p, _ in answered], [a for _, a in answered]),
        "mean_confidence": statistics.fmean(max(p) for p, _ in answered)
        if answered
        else float("nan"),
    }


def _by_task(
    rows: list[dict[str, Any]], probability_rows: list[list[float] | None]
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[list[float] | None]] = defaultdict(list)
    answers: dict[str, list[int]] = defaultdict(list)
    for row, probs in zip(rows, probability_rows, strict=True):
        task = str(row["task"])
        grouped[task].append(probs)
        answers[task].append(int(row["answer_index"]))
    return {task: _summary(grouped[task], answers[task]) for task in sorted(grouped)}


# ------------------------------------------------------------------------- engine
@dataclass
class Engine:
    """One decision engine bound to a data mapping."""

    name: str
    decide: Callable[[dict[str, Any]], list[float] | None]
    describe: dict[str, Any] = field(default_factory=dict)
    gpu: bool = True


def _question_payload(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Map one dataset row onto a canonical ``/v1/systemone`` question.

    Returns the question and the ordered keys that align with ``row["options"]``, so the
    response maps back to the dataset's option order regardless of serialization.

    Rows that already carry their native question (``--format jsonl``) keep it verbatim —
    that is what Tachyone was trained on. Rows built from a foreign option set (the public
    probe) become ``score`` when they are ordered (indices are positional) and ``choice``
    otherwise, so both engines see identical option sets and identical question text.
    """
    wire = row.get("wire")
    if isinstance(wire, dict):
        kind = wire.get("type")
        if kind == "noul":
            return wire, ["noul"]  # a single probability, decoded as [false, true]
        if kind == "score":
            return wire, [str(index) for index in range(len(wire["criteria"]))]
        return wire, [str(label) for label in wire["criteria"]]

    options = [str(option) for option in row["options"]]
    instructions = str(row["question"])
    if bool(row.get("ordered")) and 2 <= len(options) <= 10:
        payload = {"type": "score", "instructions": instructions, "criteria": options}
        return payload, [str(index) for index in range(len(options))]
    criteria = _unique_options(options)
    payload = {"type": "choice", "instructions": instructions, "criteria": criteria}
    return payload, list(criteria)


def _unique_options(options: list[str]) -> dict[str, str | None]:
    """Option text → ``None`` (the encoder renders ``option`` when there is no description).

    Duplicate option strings would collide as dict keys, so repeats get a stable suffix; the
    reverse mapping uses the insertion order, which JSON preserves.
    """
    criteria: dict[str, str | None] = {}
    seen: dict[str, int] = {}
    for option in options:
        count = seen.get(option, 0) + 1
        seen[option] = count
        key = option if count == 1 else f"{option} ({count})"
        while key in criteria:
            count += 1
            key = f"{option} ({count})"
        criteria[key] = None
    return criteria


def _wire_engine(backend: Any, model: str) -> Engine:
    """Drive any Tachyone backend (``encoder`` or ``llm``) through the frozen wire path."""
    from tachyone.wire import answer, parse_request  # first-party, always present

    loop = asyncio.new_event_loop()

    def decide(row: dict[str, Any]) -> list[float] | None:
        options = [str(option) for option in row["options"]]
        payload, keys = _question_payload(row)
        try:
            request = parse_request(
                {"state": row["state"], "model": model, "questions": {"q": payload}}
            )
            response = loop.run_until_complete(answer(request, backend))
        except Exception:  # a failed answer is a metric, not a crash
            return None
        dumped = response.answers["q"].model_dump(mode="json")
        if keys == ["noul"]:
            value = float(dumped.get("noul", 0.0))
            return [1.0 - value, value]
        probabilities = dumped.get("probabilities")
        if not isinstance(probabilities, dict):
            return None
        values = [float(probabilities.get(key, 0.0)) for key in keys]
        if len(values) != len(options):
            return None
        total = sum(values)
        return [value / total for value in values] if total > 0 else None

    return Engine(name="tachyone", decide=decide, gpu=True)


def _tachyone_engine(args: argparse.Namespace) -> Engine:
    from tachyone.backends import build_backend
    from tachyone.config import Config

    env = {**os.environ, "TACHYONE_BACKEND": args.backend}
    engine = _wire_engine(build_backend(Config.from_env(env)), model="tachyone-latest")
    engine.name = f"tachyone ({args.backend})"
    engine.describe = {"backend": args.backend}
    return engine


def _llm_engine(args: argparse.Namespace) -> Engine:
    """The same wire path, but answered by an OpenAI-compatible server (llama-server)."""
    from tachyone.backends import build_backend
    from tachyone.config import Config

    if not args.llm_base_url:
        raise SystemExit("--llm-base-url is required for --engine llm")
    env = {
        **os.environ,
        "TACHYONE_BACKEND": "llm",
        "TACHYONE_LLM_BASE_URL": args.llm_base_url,
        "TACHYONE_LLM_MODEL": args.llm_model,
        "TACHYONE_LLM_TIMEOUT": str(args.llm_timeout),
        "TACHYONE_LLM_RETRIES": str(args.llm_retries),
    }
    engine = _wire_engine(build_backend(Config.from_env(env)), model="tachyone-latest")
    engine.name = f"llm ({args.llm_model})"
    engine.gpu = False
    engine.describe = {
        "backend": "llm",
        "base_url": args.llm_base_url,
        "model": args.llm_model,
        "timeout_s": args.llm_timeout,
        "retries": args.llm_retries,
    }
    return engine


def _systemone_engine(args: argparse.Namespace) -> Engine:
    """The open peer scorer, evaluated with **its own** scoring code, loaded from disk."""
    if not (args.systemone_script and args.systemone_base and args.systemone_adapter):
        raise SystemExit(
            "--engine systemone needs --systemone-script, --systemone-base and "
            "--systemone-adapter (paths outside this repository; the model is CC-BY-NC-4.0)"
        )
    import importlib.util

    import torch  # pyright: ignore[reportMissingImports]

    spec = importlib.util.spec_from_file_location("_system_one", args.systemone_script)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {args.systemone_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from peft import PeftModel  # pyright: ignore[reportMissingImports]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok, model = module.load_model(args.systemone_base, lora=False, dtype=dtype)
    model = PeftModel.from_pretrained(model, args.systemone_adapter)
    model.to(device).eval()

    def decide(row: dict[str, Any]) -> list[float] | None:
        logits = module.score_options(
            model,
            tok,
            row["state"],
            row["question"],
            [str(option) for option in row["options"]],
            args.max_len,
            args.option_batch,
            device,
        )
        return _softmax([float(value) for value in logits])

    return Engine(
        name="systemone-qwen3.5-4b",
        decide=decide,
        gpu=device.type == "cuda",
        describe={
            "script": args.systemone_script,
            "base": args.systemone_base,
            "adapter": args.systemone_adapter,
            "max_len": args.max_len,
            "option_batch": args.option_batch,
            "device": str(device),
        },
    )


def build_engine(args: argparse.Namespace) -> Engine:
    if args.engine == "tachyone":
        return _tachyone_engine(args)
    if args.engine == "systemone":
        return _systemone_engine(args)
    if args.engine == "llm":
        return _llm_engine(args)
    raise SystemExit(f"unknown engine: {args.engine}")


# ----------------------------------------------------------------------- measuring
def _sync(engine: Engine) -> None:
    if not engine.gpu:
        return
    try:
        import torch  # pyright: ignore[reportMissingImports]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:  # pragma: no cover - torch is optional
        return


def _percentile(values: list[float], fraction: float) -> float:
    """Linearly interpolated percentile (the estimator numpy uses by default)."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _peak_rss_mb(pid: int | None = None) -> float:
    """Peak resident set in MiB — of this process, or of ``pid`` (Linux ``VmHWM``)."""
    source = Path("/proc/self/status") if pid is None else Path(f"/proc/{pid}/status")
    try:
        status = source.read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - non-Linux or a vanished process
        return float("nan")
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            return float(line.split()[1]) / 1024.0
    return float("nan")


def _find_llm_server_pid(base_url: str) -> int | None:
    """Locate the ``llama-server`` process listening on ``base_url``'s port, if any."""
    port = base_url.rsplit(":", 1)[-1].split("/")[0]
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:  # pragma: no cover - non-Linux
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if "llama-server" in cmdline and port in cmdline:
            return int(entry.name)
    return None


def _server_vram_mb(pid: int) -> float:
    """VRAM held by another CUDA process, as reported by ``nvidia-smi``."""
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return float("nan")
    for line in output.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            return float(parts[1])
    return float("nan")


def _peak_vram_mb(pid: int | None = None) -> float:
    if pid is not None:
        return _server_vram_mb(pid)
    try:
        import torch  # pyright: ignore[reportMissingImports]

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:  # pragma: no cover
        pass
    return 0.0


def _environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
    }
    try:
        import torch  # pyright: ignore[reportMissingImports]

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_total_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**2)
            )
    except ImportError:
        info["torch"] = None
    return info


def _evaluate(
    engine: Engine, rows: list[dict[str, Any]], warmup: int
) -> tuple[list[list[float] | None], list[float], int]:
    """Run every row once; latency is measured around ``decide`` only, warmup excluded.

    One discarded call happens first so lazy costs (kernel selection, HTTP keep-alive, first
    token) never land in the reported percentiles.
    """
    if rows:
        engine.decide(rows[0])
    probabilities: list[list[float] | None] = []
    latencies: list[float] = []
    failures = 0
    for index, row in enumerate(rows):
        _sync(engine)
        started = time.perf_counter()
        result = engine.decide(row)
        _sync(engine)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if result is None:
            failures += 1
        probabilities.append(result)
        if index >= warmup:
            latencies.append(elapsed_ms)
    return probabilities, latencies, failures


def _calibrated(
    distributions: list[list[float] | None], temperature: float
) -> list[list[float] | None]:
    return [
        None
        if probs is None
        else _probabilities_from_log([math.log(max(p, 1e-12)) for p in probs], temperature)
        for probs in distributions
    ]


def _probe_split(args: argparse.Namespace, *, validation: bool) -> str:
    """Split to read: ``--split``/``--val-split`` when given, else the dataset's own default."""
    if not validation:
        return str(args.split) if args.split else "test"
    if args.val_split:
        return str(args.val_split)
    return PROBE_VAL_SPLITS.get(str(args.probe), "val")


def _probe_languages(args: argparse.Namespace) -> tuple[str, ...] | None:
    """Language tags evaluated (recorded in the artifact), or ``None`` if not applicable."""
    probe = str(args.probe)
    if probe not in {"massive", "xnli"}:
        return None
    if args.languages:
        return tuple(tag for tag in str(args.languages).split(",") if tag)
    from benchmarks.probes import MASSIVE_LANGUAGES  # probes imports this module: keep it lazy

    return MASSIVE_LANGUAGES if probe == "massive" else ("en",)


def _select_rows(args: argparse.Namespace, *, validation: bool) -> list[dict[str, Any]] | None:
    """Rows for the reported split, or for the temperature-fitting split (``None`` if absent)."""
    if args.format == "jsonl":
        path = args.val_data if validation else args.data
        if validation and path is None:
            return None
        return load_tachyone_records(Path(path), args.per_task, args.limit)
    split = _probe_split(args, validation=validation)
    if validation and not split:
        return None
    probe = str(args.probe)
    if probe == "system-one-decisions":
        return load_rows(args.data, split, args.per_task, args.limit)
    from benchmarks.probes import load_probe  # imported here: probes imports this module

    return load_probe(
        probe,
        args.data,
        split=split,
        per_task=args.per_task,
        limit=args.limit,
        languages=_probe_languages(args),
    )


def run(args: argparse.Namespace) -> None:
    engine = build_engine(args)
    rows = _select_rows(args, validation=False)
    assert rows is not None

    # Temperature is fitted on the validation split (never on the split being reported).
    val_rows = _select_rows(args, validation=True) if args.temperature is None else None
    fit_on_val = val_rows is not None
    if val_rows is not None:
        val_probabilities, _, val_failures = _evaluate(engine, val_rows, args.warmup)
        answered = [
            (p, int(r["answer_index"]))
            for p, r in zip(val_probabilities, val_rows, strict=True)
            if p
        ]
        if not answered:
            raise SystemExit("validation split produced no usable answers; cannot fit temperature")
        temperature = fit_temperature([p for p, _ in answered], [a for _, a in answered])
        print(
            f"temperature {temperature:.2f} fitted on {len(answered)} validation rows "
            f"({val_failures} failures)",
            flush=True,
        )
    else:
        temperature = args.temperature if args.temperature is not None else 1.0

    probabilities, latencies, failures = _evaluate(engine, rows, args.warmup)
    answers = [int(row["answer_index"]) for row in rows]
    raw = _summary(probabilities, answers)
    calibrated = _calibrated(probabilities, temperature)
    cal = _summary(calibrated, answers)
    answered_pairs = [(p, a) for p, a in zip(probabilities, answers, strict=True) if p]

    total_seconds = sum(latencies) / 1000.0
    server_pid = _find_llm_server_pid(str(args.llm_base_url)) if args.engine == "llm" else None
    is_probe = args.format == "probe"
    languages = _probe_languages(args)
    artifact = {
        "engine": engine.name,
        "dataset": {
            "kind": args.format,
            "probe": str(args.probe) if is_probe else None,
            "repo": PROBE_REPOS.get(str(args.probe)) if is_probe else None,
            "path": str(args.data),
            "split": _probe_split(args, validation=False),
            "val_split": _probe_split(args, validation=True) if fit_on_val else None,
            "per_task": args.per_task,
            "limit": args.limit,
            "languages": list(languages) if languages else None,
            "rows": len(rows),
            "warmup_excluded": min(args.warmup, len(rows)),
        },
        "temperature": {
            "fitted": fit_on_val,
            "value": round(temperature, 4),
            "source": "validation split (capped)" if fit_on_val else "given/fixed",
        },
        "raw": raw,
        "calibrated": cal,
        "per_task_raw": _by_task(rows, probabilities),
        "per_task_calibrated": _by_task(rows, calibrated),
        "latency_ms": {
            "n": len(latencies),
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
            "mean": round(statistics.fmean(latencies), 3) if latencies else float("nan"),
        },
        "throughput": {
            "items_per_s": round(len(latencies) / total_seconds, 3)
            if total_seconds
            else float("nan"),
            "note": "sequential, one question per item, batch=1",
        },
        "compliance": {
            "answered": len(answered_pairs),
            "failed": failures,
            "rate": round(len(answered_pairs) / len(rows), 6) if rows else float("nan"),
        },
        "memory": {
            "peak_rss_mb": round(_peak_rss_mb(server_pid), 1),
            "peak_vram_mb": round(_peak_vram_mb(server_pid), 1),
            "measured_on": f"llama-server pid {server_pid}" if server_pid else "this process",
        },
        "engine_config": engine.describe,
        "environment": _environment(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "engine": artifact["engine"],
                "rows": len(rows),
                "temperature": artifact["temperature"],
                "accuracy": round(raw["accuracy"], 4),
                "ece_raw": round(float(raw["ece"]), 4),
                "ece_calibrated": round(float(cal["ece"]), 4),
                "p50_ms": artifact["latency_ms"]["p50"],
                "items_per_s": artifact["throughput"]["items_per_s"],
                "compliance": artifact["compliance"]["rate"],
                "peak_rss_mb": artifact["memory"]["peak_rss_mb"],
                "peak_vram_mb": artifact["memory"]["peak_vram_mb"],
                "out": str(args.out),
            },
            indent=2,
        )
    )


# -------------------------------------------------------------------------- render
def _row(artifact: dict[str, Any], kind: str) -> str:
    raw = artifact["raw"]
    cal = artifact["calibrated"]
    latency = artifact["latency_ms"]
    memory = artifact["memory"]
    if kind == "quality":
        return (
            f"| {artifact['engine']} | {artifact['dataset']['rows']} | {cal['answered']} "
            f"| {cal['accuracy']:.3f} | {float(raw['ece']):.3f} | {float(cal['ece']):.3f} "
            f"| {float(cal['brier']):.3f} | {float(cal['mean_confidence']):.3f} |"
        )
    compliance = artifact["compliance"]
    return (
        f"| {artifact['engine']} "
        f"| {latency['p50']:.2f} | {latency['p95']:.2f} "
        f"| {artifact['throughput']['items_per_s']:.1f} "
        f"| {compliance['rate']:.3f} "
        f"| {memory['peak_rss_mb']:.0f} | {memory['peak_vram_mb']:.0f} |"
    )


def _header(kind: str) -> str:
    if kind == "quality":
        return (
            "| Engine | n | answered | Accuracy | ECE raw | ECE cal | Brier | Conf |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
    return (
        "| Engine | p50 (ms) | p95 (ms) | items/s | JSON ok | RSS (MiB) | VRAM (MiB) |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )


def _per_task_table(artifacts: list[dict[str, Any]]) -> str:
    tasks = sorted(
        {task for artifact in artifacts for task in artifact.get("per_task_calibrated", {})}
    )
    engines = [artifact["engine"] for artifact in artifacts]
    lines = [
        "| Task family | " + " | ".join(engines) + " |",
        "| --- | " + " | ".join(["---:"] * len(engines)) + " |",
    ]
    for task in tasks:
        cells = []
        for artifact in artifacts:
            entry = artifact.get("per_task_calibrated", {}).get(task)
            cells.append(f"{entry['accuracy']:.3f}" if entry else "—")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render(paths: list[Path], out: Path | None) -> None:
    artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    artifacts.sort(key=lambda item: str(item["engine"]))

    def table(kind: str) -> str:
        return "\n".join([_header(kind), *(_row(artifact, kind) for artifact in artifacts)])

    body = ["### Quality", "", table("quality"), ""]
    body += [
        "`answered` is how many questions produced a contract-valid payload at all. **Accuracy "
        "counts a failed answer as wrong**; `ECE raw`, `ECE cal`, `Brier` and `Conf` are computed "
        "over `answered` rows only, so a low count there also means those calibration numbers rest "
        "on few samples. `ECE raw` is the engine as shipped; `ECE cal` after a scalar temperature "
        "fitted to minimize NLL on the capped validation split. Fit temperature is **not** an ECE "
        "optimizer, so an engine that ships already-calibrated confidence can score worse after "
        "it — read the four columns together. `Conf` is mean calibrated confidence: an engine can "
        "flatten its distributions to lower ECE, which shows up here as low `Conf` and is "
        "captured by `Brier`.",
        "",
        "### Performance",
        "",
        table("performance"),
        "",
        "Latency is p50/p95 **per question**, sequential, batch=1, warmup excluded; "
        "`items/s` is throughput for that same sequential loop. `JSON ok` is the share of "
        "questions answered with a contract-valid payload (1.000 by construction for the "
        "encoders, measured for the LLM). Memory is peak RSS and peak VRAM of the process "
        "that did the inference (for the LLM that is the `llama-server`, not the client).",
        "",
        "### Per-task accuracy (calibrated)",
        "",
        _per_task_table(artifacts),
        "",
    ]
    text = "\n".join(body)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)


# ---------------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks.compare", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="evaluate one engine and write a JSON artifact")
    run_parser.add_argument("--engine", choices=["tachyone", "systemone", "llm"], required=True)
    run_parser.add_argument(
        "--format",
        choices=["probe", "jsonl"],
        default="probe",
        help="probe: a public dataset selected by --probe; "
        "jsonl: Tachyone's own eval records (our distribution)",
    )
    run_parser.add_argument(
        "--probe",
        choices=list(PROBE_CHOICES),
        default="system-one-decisions",
        help="dataset behind --format probe: system-one-decisions is the peer scorer's own "
        "distribution, the other three are the public probes of BACKLOG B-7",
    )
    run_parser.add_argument(
        "--languages",
        default=None,
        help="massive: comma-separated language tags (default: English plus the six trained "
        "multilingual languages); xnli: comma-separated codes (default: en)",
    )
    run_parser.add_argument(
        "--data",
        type=Path,
        default=_ROOT / "data" / "system-one-decisions",
        help="probe: dataset directory; jsonl: the eval records file",
    )
    run_parser.add_argument(
        "--split", default=None, help="probe: split name (default: test for every dataset)"
    )
    run_parser.add_argument(
        "--val-split",
        default=None,
        help="probe: split used to fit temperature (default: each dataset's own dev split)",
    )
    run_parser.add_argument(
        "--val-data", type=Path, default=None, help="jsonl: records used to fit temperature"
    )
    run_parser.add_argument("--per-task", type=int, default=DEFAULT_PER_TASK)
    run_parser.add_argument("--limit", type=int, default=0)
    run_parser.add_argument("--temperature", type=float, default=None)
    run_parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    run_parser.add_argument("--out", type=Path, required=True)
    # tachyone
    run_parser.add_argument(
        "--backend", default="encoder", help="local backend for --engine tachyone"
    )
    # systemone (all paths live outside this repository)
    run_parser.add_argument("--systemone-script", default=None)
    run_parser.add_argument("--systemone-base", default=None)
    run_parser.add_argument("--systemone-adapter", default=None)
    run_parser.add_argument("--max-len", type=int, default=384)
    run_parser.add_argument("--option-batch", type=int, default=16)
    # llm
    run_parser.add_argument("--llm-base-url", default=None)
    run_parser.add_argument("--llm-model", default="local")
    run_parser.add_argument("--llm-timeout", type=float, default=120.0)
    run_parser.add_argument(
        "--llm-retries",
        type=int,
        default=1,
        help="attempts per question beyond the first (shipped default is 2)",
    )

    render_parser = sub.add_parser("render", help="render artifacts as a markdown table")
    render_parser.add_argument("artifacts", nargs="+", type=Path)
    render_parser.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "run":
        run(args)
    else:
        render(args.artifacts, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
