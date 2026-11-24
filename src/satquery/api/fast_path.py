"""Discover the faster public URL this API is also reachable at, if any.

The demo box is published two ways. A Tailscale Funnel URL is stable but its
ingress can sit a continent away; a Cloudflare quick tunnel lands at a nearby
edge and is several times faster, but its hostname changes on every restart.
So the stable URL serves as the signpost: ``/v1/health`` reports the quick
tunnel's current hostname, read from ``cloudflared``'s local metrics server
(``GET /quicktunnel`` -> ``{"hostname": "..."}``), and the frontend moves the
rest of its traffic there.

A quick tunnel buffers streamed responses, so a client on it reads job events
with the short-poll form of ``/v1/jobs/{id}/events`` (``?poll=true``).
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Final

from satquery.core.logging import get_logger

CACHE_SECONDS: Final[float] = 10.0
"""How long one lookup is trusted. Health is polled every 15 s per tab."""

LOOKUP_TIMEOUT_SECONDS: Final[float] = 0.5

log = get_logger(__name__)

_cache: dict[str, tuple[float, str | None]] = {}


def _lookup(metrics_url: str) -> str | None:
    try:
        with urllib.request.urlopen(metrics_url, timeout=LOOKUP_TIMEOUT_SECONDS) as response:  # noqa: S310 - operator-configured localhost URL
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError) as error:
        log.debug("fast_path.unavailable", detail=str(error))
        return None
    hostname = body.get("hostname") if isinstance(body, dict) else None
    if not isinstance(hostname, str) or not hostname.strip():
        return None
    hostname = hostname.strip().rstrip("/")
    return hostname if hostname.startswith("https://") else f"https://{hostname}"


def fast_base_url(metrics_url: str | None) -> str | None:
    """The quick tunnel's public base URL, or None when there is no live tunnel.

    Blocking (one localhost HTTP call at most every :data:`CACHE_SECONDS`);
    call it off the event loop.
    """
    if not metrics_url:
        return None
    now = time.monotonic()
    cached = _cache.get(metrics_url)
    if cached is not None and now - cached[0] < CACHE_SECONDS:
        return cached[1]
    url = _lookup(metrics_url)
    _cache[metrics_url] = (now, url)
    return url


def clear_cache() -> None:
    """Forget cached lookups (tests)."""
    _cache.clear()


__all__ = ["CACHE_SECONDS", "clear_cache", "fast_base_url"]
