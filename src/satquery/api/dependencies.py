"""Shared FastAPI dependencies."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request

from satquery.agent.concurrency import DeviceGates
from satquery.core.config import Settings, get_settings
from satquery.render.artifact_store import ArtifactStore
from satquery.trace.store import TraceStore


@lru_cache(maxsize=8)
def _store_for(root: str) -> ArtifactStore:
    """Cache one store per root so the index is not re-opened per request."""
    return ArtifactStore(root)


def get_artifact_store(settings: Annotated[Settings, Depends(get_settings)]) -> ArtifactStore:
    """Provide the process-wide artifact store."""
    return _store_for(str(settings.artifact_root))


@lru_cache(maxsize=8)
def _traces_at(path: str) -> TraceStore:
    """Cache one trace store per database path."""
    return TraceStore(path)


def get_trace_store(settings: Annotated[Settings, Depends(get_settings)]) -> TraceStore:
    """Provide the process-wide audit trace store."""
    return _traces_at(str(settings.trace_db))


def get_device_gates(request: Request) -> DeviceGates:
    """Provide the process-wide device gates the lifespan created.

    A test client that bypasses the lifespan (``TestClient(app)`` without a
    ``with``) has no ``app.state.gates``; it gets one made on the spot, stored
    so the next request in that process shares it.
    """
    gates: DeviceGates | None = getattr(request.app.state, "gates", None)
    if gates is None:
        gates = DeviceGates()
        request.app.state.gates = gates
    return gates
