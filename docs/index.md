# jeba

**Local-first, multilingual System One decision engine** that answers typed `choice` / `score`
/ `noul` questions and speaks the TypeSafe Jev `/v1/systemone` wire protocol as a drop-in.

- **Local-first:** the base install runs offline with no API key. Heavy or remote backends live
  behind pip extras (`serve`, `onnx`, `fast`, `langchain`, `mcp`, `train`).
- **One request, many atomic questions:** primitives are evaluated independently and returned
  as typed values with probabilities and `confidence`.
- **Drop-in contract:** existing Jev clients work unchanged; extensions are additive.
- **Trainable:** a reproducible pipeline (data → LoRA/RLCD → calibration → evaluation) targets a
  single RTX 3060 12GB.

## Start here

- [Overview](overview.md) — vision, personas, non-goals.
- [Protocol](protocol.md) — the frozen `POST /v1/systemone` contract.
- [Architecture](architecture.md) — components, flows, backend strategy.
- [Roadmap](roadmap.md) and [Tasks](tasks.md) — the M0–M6 plan.
- [Testing](testing.md) and [Training](training.md) — quality gates and the ML pipeline.

## Install

```bash
uv sync                 # core (offline, no key)
uv sync --extra serve   # HTTP server
uv run jeba --predict --preset triage --backend fake "refund please"
```
