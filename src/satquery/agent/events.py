"""The optional progress channel from the pipeline to a streaming transport.

``POST /v1/analyze`` is synchronous and needs none of this. ``POST /v1/jobs``
does: the frontend draws the tool DAG from the ``plan`` event and fills it in
from ``step_started`` / ``step_completed`` as they arrive (API_CONTRACT §5).

The channel is a plain callback rather than a queue or a bus so the pipeline
stays synchronous-looking and testable, and so a run with no listener costs one
``is None`` check per event. Two rules make it safe to hand to a network
transport:

* **A dead listener never fails an analysis.** :func:`emit_to` swallows and logs
  anything the sink raises. A judge closing the browser tab mid-run must not
  turn a working analysis into a 500.
* **The sink must not block.** It is called from inside the executor's event
  loop, so it puts onto a queue and returns; it does not await a socket.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final, TypeAlias

from satquery.core.logging import get_logger

log = get_logger(__name__)

Emit: TypeAlias = Callable[[str, dict[str, Any]], None]
"""``(event_name, data) -> None``. Event names are the §5 SSE event types."""

STAGE_PCT: Final[dict[str, int]] = {
    "ingesting": 5,
    "validating": 10,
    "rendering": 20,
    "planning": 30,
    "executing": 45,
    "aggregating": 85,
    "done": 100,
}
"""Coarse progress per stage, so the client has something monotonic to animate.

Execution is the long pole and spans 45-85: the job store interpolates inside
that band from ``step_completed`` counts, because only it knows the step total.
"""


def emit_to(emit: Emit | None, event: str, data: dict[str, Any]) -> None:
    """Send one event, tolerating both an absent sink and a broken one."""
    if emit is None:
        return
    try:
        emit(event, data)
    except Exception:  # noqa: BLE001 - a listener must never break the analysis
        log.warning("events.sink_failed", sse_event=event, exc_info=True)


def stage(emit: Emit | None, name: str) -> None:
    """Emit a ``stage`` event with its canonical percentage."""
    emit_to(emit, "stage", {"stage": name, "pct": STAGE_PCT.get(name, 0)})


__all__ = ["STAGE_PCT", "Emit", "emit_to", "stage"]
