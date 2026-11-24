"""Shared-code access control for an API that is reachable from the internet.

The demo deployment puts the API behind a public tunnel so a hosted frontend
can reach a home GPU. Without a gate, anyone who finds the URL can queue work
on that GPU. This middleware is the gate: when ``SATQUERY_ACCESS_CODE`` is set,
every request under the API prefix must carry a matching code.

The code travels in the ``X-SatQuery-Code`` header for everything ``fetch``
sends, and as a ``?code=`` query parameter for the URLs the browser loads by
itself (an ``<img src>`` cannot set a header). Comparison is constant-time.

Wrong codes are counted per client IP; past ``access_max_failures`` inside
``access_lockout_s`` that IP gets 429 until the window passes, which turns a
brute-force search of the code space into a very slow one.

CORS preflights (``OPTIONS``) pass untouched: a browser never sends custom
headers on a preflight, so requiring the code there would block every
cross-origin request before it started.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Final

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from satquery.core.config import Settings
from satquery.core.logging import get_logger
from satquery.schemas.api import ApiError, ApiErrorResponse

ACCESS_HEADER: Final[str] = "X-SatQuery-Code"
ACCESS_QUERY_PARAM: Final[str] = "code"

log = get_logger(__name__)


def parse_codes(raw: str | None) -> tuple[str, ...]:
    """Split a comma-separated code list, dropping blanks and surrounding space."""
    if not raw:
        return ()
    return tuple(code.strip() for code in raw.split(",") if code.strip())


@dataclass
class AccessGuard:
    """Decides whether one request may proceed, and remembers who keeps guessing."""

    codes: tuple[str, ...]
    max_failures: int
    lockout_s: float
    clock: Callable[[], float] = time.monotonic
    _failures: dict[str, list[float]] = field(default_factory=dict, repr=False)

    @property
    def enabled(self) -> bool:
        """True when at least one code is configured."""
        return bool(self.codes)

    def matches(self, candidate: str | None) -> bool:
        """Constant-time comparison against every configured code."""
        if not candidate:
            return False
        given = candidate.strip().encode()
        # Compare against all codes, always: stopping at the first match would
        # leak which position in the list matched through timing.
        found = False
        for code in self.codes:
            found |= secrets.compare_digest(given, code.encode())
        return found

    def _recent(self, client: str) -> list[float]:
        cutoff = self.clock() - self.lockout_s
        recent = [t for t in self._failures.get(client, []) if t > cutoff]
        if recent:
            self._failures[client] = recent
        else:
            self._failures.pop(client, None)
        return recent

    def locked_out(self, client: str) -> bool:
        """True while *client* has used up its wrong guesses for this window."""
        return len(self._recent(client)) >= self.max_failures

    def record_failure(self, client: str) -> None:
        """Count one wrong (not missing) code from *client*."""
        self._recent(client)
        self._failures.setdefault(client, []).append(self.clock())


def _error(status: int, code: str, message: str, hint: str) -> JSONResponse:
    body = ApiErrorResponse(
        error=ApiError(code=code, http_status=status, message=message, hint=hint)
    )
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


def gpu_busy_response(active: int) -> JSONResponse:
    """The 429 an analysis gets when the line for the GPU is already full."""
    return _error(
        429,
        "GPU_BUSY",
        f"The GPU is busy with {active} analyses right now.",
        "Wait about a minute, then ask again.",
    )


def install_access_guard(app: FastAPI, settings: Settings) -> AccessGuard:
    """Register the access middleware on *app* and return the guard it uses.

    Call this **before** adding ``CORSMiddleware``: middleware added later wraps
    middleware added earlier, and CORS has to be the outer layer so that a 401
    still carries ``Access-Control-Allow-Origin``. Without it the browser hides
    the status and the frontend sees an opaque network failure instead of
    "enter your code".
    """
    raw = settings.access_code.get_secret_value() if settings.access_code else None
    guard = AccessGuard(
        codes=parse_codes(raw),
        max_failures=settings.access_max_failures,
        lockout_s=settings.access_lockout_s,
    )
    prefix = settings.api_prefix.rstrip("/") + "/"

    @app.middleware("http")
    async def require_access_code(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if (
            not guard.enabled
            or request.method == "OPTIONS"
            or not request.url.path.startswith(prefix)
        ):
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        if guard.locked_out(client):
            return _error(
                429,
                "ACCESS_LOCKED",
                "Too many wrong access codes from this network.",
                f"Wait {int(guard.lockout_s // 60)} minutes, then try again.",
            )

        candidate = request.headers.get(ACCESS_HEADER) or request.query_params.get(
            ACCESS_QUERY_PARAM
        )
        if guard.matches(candidate):
            return await call_next(request)

        if candidate:
            guard.record_failure(client)
            log.warning("access.denied", client=client, path=request.url.path)
            return _error(
                401,
                "ACCESS_CODE_INVALID",
                "That access code is not valid.",
                "Check the code with the person running this demo.",
            )
        return _error(
            401,
            "ACCESS_CODE_REQUIRED",
            "This SatQuery instance needs an access code.",
            "Enter the access code you were given.",
        )

    if guard.enabled:
        log.info("access.enabled", codes=len(guard.codes))
    return guard


__all__ = [
    "ACCESS_HEADER",
    "ACCESS_QUERY_PARAM",
    "AccessGuard",
    "gpu_busy_response",
    "install_access_guard",
    "parse_codes",
]
