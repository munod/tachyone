"""HTTP surface: the canonical ``POST /v1/systemone`` plus additive extensions.

FastAPI/uvicorn live behind the ``serve`` extra and are imported lazily, so ``import
jeba.serve`` succeeds on a base install and only fails with an actionable message when the
server is actually started without the extra (NFR-M06). Transport delegates to
``wire.answer``; it adds auth, JSON parsing, and error-status mapping only.
"""

from __future__ import annotations

import secrets
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from jeba import __version__
from jeba.backends import build_backend
from jeba.backends.base import Backend
from jeba.config import Config
from jeba.primitives import Question, State
from jeba.wire import (
    SystemOneRequest,
    Unauthorized,
    UnprocessableEntity,
    WireError,
    answer,
    parse_request,
)

if TYPE_CHECKING:
    from fastapi import FastAPI, Request

_SERVE_HINT = "the serve extra is required: uv sync --extra serve"


class _ExtensionBase(BaseModel):
    model_config = ConfigDict(extra="ignore")


class PredictRequest(_ExtensionBase):
    """Body for the ``/predict`` extension: a wire request with a default model."""

    state: State
    questions: dict[str, Question]
    model: str = "jeba-latest"


class BatchPredictRequest(_ExtensionBase):
    """Body for the ``/predict/batch`` extension."""

    requests: list[PredictRequest]


def _fastapi() -> tuple[Any, Any, Any]:
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(_SERVE_HINT) from exc
    return FastAPI, Request, JSONResponse


def _authorize(request: Request, api_key: str | None) -> None:
    if not api_key:
        return
    header = request.headers.get("authorization", "")
    # Constant-time comparison; documented as dev-grade (single shared key) in NFR-S06.
    if not secrets.compare_digest(header, f"Bearer {api_key}"):
        raise Unauthorized("missing or invalid API key")


def create_app(config: Config, backend: Backend) -> FastAPI:
    """Build the FastAPI application bound to a specific backend."""
    FastAPI, Request, JSONResponse = _fastapi()
    # FastAPI resolves route annotations from module globals; expose the real class here
    # without importing it at module import time (keeps the base install serve-free).
    globals()["Request"] = Request
    app = FastAPI(title="jeba", version=__version__)

    @app.exception_handler(WireError)
    async def _wire_error_handler(_request: Request, exc: WireError) -> Any:
        return JSONResponse(status_code=exc.status_code, content=exc.to_body())

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> Any:
        _authorize(request, config.api_key)
        try:
            payload = await request.json()
        except Exception as exc:
            raise UnprocessableEntity("request body is not valid JSON") from exc
        validated = parse_request(payload)
        response = await answer(validated, backend)
        return response.model_dump(mode="json")

    @app.post("/predict")
    async def predict(request: Request, body: PredictRequest) -> Any:
        _authorize(request, config.api_key)
        validated = SystemOneRequest(state=body.state, model=body.model, questions=body.questions)
        response = await answer(validated, backend)
        return response.model_dump(mode="json")

    @app.post("/predict/batch")
    async def predict_batch(request: Request, body: BatchPredictRequest) -> dict[str, Any]:
        _authorize(request, config.api_key)
        results = []
        for item in body.requests:
            validated = SystemOneRequest(
                state=item.state, model=item.model, questions=item.questions
            )
            results.append((await answer(validated, backend)).model_dump(mode="json"))
        return {"results": results}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "backend": backend.name,
            "device": config.device,
            "models": list(config.models),
        }

    return app


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``jeba-serve`` console script."""
    del argv  # configuration is environment-driven
    try:
        import uvicorn
    except ImportError as exc:
        print(_SERVE_HINT, file=sys.stderr)
        raise SystemExit(1) from exc

    config = Config.from_env()
    backend = build_backend(config)
    uvicorn.run(create_app(config, backend), host=config.host, port=config.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
