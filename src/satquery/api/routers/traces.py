"""``GET /v1/traces/{trace_id}`` — the requirement-5 deliverable, addressable.

A trace outlives the request that produced it (Master.md §6.4): a judge has to be
able to ask for the run they watched twenty minutes ago, and the frontend's run
history rehydrates from here. :class:`~satquery.trace.store.TraceStore` already
persists the whole :class:`~satquery.schemas.trace.AuditTrace` as canonical JSON
keyed by ``trace_id``, so this is a lookup rather than a feature.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from satquery.api.dependencies import get_trace_store
from satquery.schemas.trace import AuditTrace
from satquery.trace.store import TraceStore

router = APIRouter(tags=["traces"])


@router.get(
    "/traces/{trace_id}",
    response_model=AuditTrace,
    summary="Fetch one stored audit trace",
    responses={404: {"description": "Unknown trace."}},
)
async def get_trace(
    trace_id: str,
    traces: Annotated[TraceStore, Depends(get_trace_store)],
) -> AuditTrace:
    """Return the full audit trace for *trace_id*."""
    trace = traces.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"No trace {trace_id}.")
    return trace


__all__: list[str] = ["router"]
