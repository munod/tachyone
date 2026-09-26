"""Python SDK client for the tachyone ``/v1/systemone`` endpoint.

Stdlib-only (ADR-0004): the base install can call a local server with no extra dependency.
Retries ``429``/``529`` with exponential backoff plus jitter (WIRE-05); other non-2xx
statuses raise :class:`TachyoneAPIError` carrying the server's JSON body.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from tachyone.primitives import Question, State
from tachyone.wire import SystemOneRequest, SystemOneResponse

#: Sends a request and returns ``(status_code, body_text)``.
type Transport = Callable[[str, bytes, dict[str, str], float], tuple[int, str]]

_RETRYABLE = (429, 529)


class TachyoneError(Exception):
    """Base class for SDK errors."""


class TachyoneConnectionError(TachyoneError):
    """The server could not be reached."""


class TachyoneAPIError(TachyoneError):
    """The server returned a non-2xx status other than a retryable one."""

    def __init__(self, status_code: int, body: Any) -> None:
        super().__init__(f"tachyone returned HTTP {status_code}")
        self.status_code = status_code
        self.body = body


def _http_transport(
    url: str, data: bytes, headers: dict[str, str], timeout: float
) -> tuple[int, str]:
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise TachyoneConnectionError(f"could not reach tachyone at {url}: {exc.reason}") from exc


def _parse_body(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


class TachyoneClient:
    """Synchronous client speaking the exact ``/v1/systemone`` wire request."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff: float = 0.5,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max(0, max_retries)
        self._backoff = backoff
        self._transport = transport or _http_transport
        self._sleep = sleep

    def system_one(
        self,
        state: State,
        questions: dict[str, Question],
        *,
        model: str = "tachyone-latest",
    ) -> SystemOneResponse:
        """Ask one or more typed questions and return the parsed response."""
        request = SystemOneRequest(state=state, model=model, questions=questions)
        data = json.dumps(request.model_dump(mode="json")).encode("utf-8")
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        url = f"{self._base_url}/v1/systemone"

        attempt = 0
        while True:
            status, text = self._transport(url, data, headers, self._timeout)
            if status in _RETRYABLE and attempt < self._max_retries:
                self._sleep(self._backoff * (2**attempt) + random.uniform(0.0, self._backoff))
                attempt += 1
                continue
            if 200 <= status < 300:
                return SystemOneResponse.model_validate_json(text)
            raise TachyoneAPIError(status, _parse_body(text))


__all__ = [
    "TachyoneAPIError",
    "TachyoneClient",
    "TachyoneConnectionError",
    "TachyoneError",
    "Transport",
]
