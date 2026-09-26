"""Contract tests for the documented error shapes (M1-T7, WIRE-04 / PRIM-02 / PRIM-03).

The HTTP layer arrives in M2; here the statuses are produced by the Python-level wire
(``parse_request`` and ``WireError``) so the error contract is frozen before transport.
"""

from __future__ import annotations

import pytest

from tachyone.wire import (
    BackendError,
    Overloaded,
    RateLimited,
    Unauthorized,
    UnprocessableEntity,
    WireError,
    parse_request,
)

pytestmark = pytest.mark.contract


def _body(overrides: dict[str, object]) -> dict[str, object]:
    base: dict[str, object] = {
        "state": "some text",
        "model": "tachyone-latest",
        "questions": {"q": {"type": "noul", "instructions": "Is it so?"}},
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "tachyone-latest", "questions": {}},  # missing state
        {"state": "x", "questions": {}},  # missing model
        {"state": "x", "model": "m"},  # missing questions
        _body({"questions": {"q": {"type": "unknown", "instructions": "x"}}}),
        _body({"questions": {"q": {"type": "noul"}}}),  # missing instructions
    ],
)
def test_validation_failures_map_to_422(payload: dict[str, object]) -> None:
    with pytest.raises(UnprocessableEntity) as excinfo:
        parse_request(payload)
    assert excinfo.value.status_code == 422
    assert excinfo.value.to_body()["error"]["code"] == "unprocessable_entity"


def test_choice_too_many_options_maps_to_422() -> None:
    payload = _body(
        {
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "pick",
                    "criteria": {f"o{i}": None for i in range(256)},
                }
            }
        }
    )
    with pytest.raises(UnprocessableEntity):
        parse_request(payload)


@pytest.mark.parametrize("levels", [[], ["only-one"], [f"l{i}" for i in range(11)]])
def test_score_bad_level_counts_map_to_422(levels: list[str]) -> None:
    payload = _body(
        {"questions": {"q": {"type": "score", "instructions": "rate", "criteria": levels}}}
    )
    with pytest.raises(UnprocessableEntity):
        parse_request(payload)


def test_boundary_limits_are_accepted() -> None:
    parse_request(
        _body(
            {
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "pick",
                        "criteria": {f"o{i}": None for i in range(255)},
                    }
                }
            }
        )
    )
    for levels in (["a", "b"], [f"l{i}" for i in range(10)]):
        parse_request(
            _body(
                {"questions": {"q": {"type": "score", "instructions": "rate", "criteria": levels}}}
            )
        )


def test_single_option_choice_is_accepted() -> None:
    # Jev imposes a maximum (255), not a minimum, on choice options.
    request = parse_request(
        _body(
            {
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "pick",
                        "criteria": {"only": None},
                    }
                }
            }
        )
    )
    assert set(request.questions["q"].criteria) == {"only"}  # type: ignore[union-attr]


def test_validation_error_details_are_structured() -> None:
    with pytest.raises(UnprocessableEntity) as excinfo:
        parse_request({"model": "m", "questions": {}})
    details = excinfo.value.to_body()["error"]["details"]
    assert isinstance(details, list) and details
    assert set(details[0]) == {"loc", "msg", "type"}


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (Unauthorized("no key"), 401, "unauthorized"),
        (UnprocessableEntity("bad body"), 422, "unprocessable_entity"),
        (RateLimited("slow down"), 429, "rate_limited"),
        (Overloaded("busy"), 529, "overloaded"),
        (BackendError("boom"), 500, "internal_error"),
    ],
)
def test_error_status_and_body(error: WireError, status: int, code: str) -> None:
    assert error.status_code == status
    body = error.to_body()
    assert body == {"error": {"code": code, "message": str(error)}}
