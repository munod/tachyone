# jeba server image: Python 3.12 + uv, `serve` extra only, non-root.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (cache-friendly), from the committed lockfile.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --extra serve

# Default to the offline, model-free backend so the image runs without keys.
ENV JEBA_BACKEND=fake \
    JEBA_HOST=0.0.0.0 \
    JEBA_PORT=8000

RUN useradd --create-home --uid 10001 jeba && chown -R jeba:jeba /app
USER jeba

EXPOSE 8000

# `uv run` uses the project environment; `--no-sync` avoids re-resolving at runtime.
CMD ["uv", "run", "--no-sync", "jeba-serve"]
