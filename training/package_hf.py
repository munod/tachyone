"""Package a trained adapter as a Hugging Face-ready folder.

HF model repos serve the model card from ``README.md`` and store the PEFT adapter files
alongside a fitted temperature. This assembles that folder locally so publishing is one
``hf upload`` (or ``git push``) away. No network access here.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

#: Files copied verbatim from the adapter directory when present.
_ADAPTER_FILES: tuple[str, ...] = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "finetune_config.json",
)


def _model_card(card_path: Path, checkpoint_name: str) -> str:
    text = card_path.read_text(encoding="utf-8")
    return text.replace(
        "# jeba (System One decision engine)",
        f"# jeba-{checkpoint_name} (System One decision engine)",
        1,
    )


def package(
    adapter_dir: str | Path,
    out_dir: str | Path,
    *,
    checkpoint_name: str,
    card_path: str | Path = "docs/model-card.md",
    temperature: str | Path | None = None,
) -> Path:
    """Assemble the HF upload folder and return its path."""
    source = Path(adapter_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"adapter directory not found: {source}")
    target = Path(out_dir)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    copied = 0
    for name in _ADAPTER_FILES:
        candidate = source / name
        if candidate.exists():
            shutil.copy2(candidate, target / name)
            copied += 1
    if copied == 0:
        raise FileNotFoundError(f"no adapter files found in {source}")

    temperature_path = Path(temperature) if temperature else source / "temperature_calibration.json"
    if temperature_path.exists():
        shutil.copy2(temperature_path, target / "temperature_calibration.json")

    (target / "README.md").write_text(
        _model_card(Path(card_path), checkpoint_name), encoding="utf-8"
    )
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jeba-package-hf", description=__doc__)
    parser.add_argument(
        "--adapter", required=True, help="trained adapter directory (checkpoints/en)"
    )
    parser.add_argument("--out", required=True, help="output folder to upload")
    parser.add_argument("--name", required=True, help="checkpoint name, e.g. 'en' or 'multi'")
    parser.add_argument("--card", default="docs/model-card.md")
    parser.add_argument("--temperature", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    target = package(
        args.adapter,
        args.out,
        checkpoint_name=args.name,
        card_path=args.card,
        temperature=args.temperature,
    )
    print(f"packaged HF upload folder: {target}")
    print(f"publish with: hf upload <user>/jeba-{args.name} {target} --repo-type model")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
