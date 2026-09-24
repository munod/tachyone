"""``jeba`` command-line entry point.

- ``jeba "text" --preset triage`` prints the canonical questions (offline, no model).
- ``jeba "text" --preset triage --predict`` answers via a local backend (env-configured).
- ``jeba "text" --preset triage --predict --url http://host:8000`` answers via a server.
- ``jeba --serve`` starts the HTTP server (requires the ``serve`` extra).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import replace

from jeba import __version__
from jeba.backends import build_backend
from jeba.client import JebaClient
from jeba.config import BACKENDS, Config
from jeba.presets import PRESETS, get_preset
from jeba.primitives import Question
from jeba.wire import SystemOneRequest, answer

_DEFAULT_MODEL = "jeba-latest"


def build_parser() -> argparse.ArgumentParser:
    """Build the ``jeba`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="jeba",
        description="Local-first System One decision engine (Jev /v1/systemone compatible).",
    )
    parser.add_argument("text", nargs="?", help="state text to evaluate")
    parser.add_argument(
        "--preset", choices=sorted(PRESETS), help="answer a ready-made question set"
    )
    parser.add_argument("--questions", help="questions as a JSON object (alternative to --preset)")
    parser.add_argument("--predict", action="store_true", help="run inference and print answers")
    parser.add_argument("--url", help="answer against a running jeba server instead of locally")
    parser.add_argument("--backend", choices=BACKENDS, help="override JEBA_BACKEND for local runs")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="model id sent in the request")
    parser.add_argument(
        "--list-presets", action="store_true", help="list available presets and exit"
    )
    parser.add_argument("--serve", action="store_true", help="start the HTTP server and exit")
    parser.add_argument("--version", action="version", version=f"jeba {__version__}")
    return parser


def _resolve_questions(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> dict[str, Question]:
    if args.questions:
        try:
            raw = json.loads(args.questions)
        except json.JSONDecodeError as exc:
            parser.error(f"--questions is not valid JSON: {exc}")
        request = SystemOneRequest.model_validate(
            {"state": "", "model": args.model, "questions": raw}
        )
        return request.questions
    if args.preset:
        return get_preset(args.preset)
    parser.error("provide --preset or --questions")
    raise AssertionError("unreachable")  # pragma: no cover - parser.error exits


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``jeba`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_presets:
        print("\n".join(sorted(PRESETS)))
        return 0
    if args.serve:
        from jeba import serve

        return serve.main()

    questions = _resolve_questions(args, parser)
    if not args.predict:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "questions": {qid: q.model_dump(mode="json") for qid, q in questions.items()},
                },
                indent=2,
            )
        )
        return 0

    if not args.text:
        parser.error("a state text is required with --predict")

    if args.url:
        client = JebaClient(args.url, api_key=os.environ.get("JEBA_API_KEY"))
        response = client.system_one(args.text, questions, model=args.model)
    else:
        config = Config.from_env()
        if args.backend:
            config = replace(config, backend=args.backend)
        backend = build_backend(config)
        request = SystemOneRequest(state=args.text, model=args.model, questions=questions)
        response = asyncio.run(answer(request, backend))

    json.dump(response.model_dump(mode="json"), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
