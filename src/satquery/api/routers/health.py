"""``GET /v1/health`` — liveness plus device and model state."""

from __future__ import annotations

import asyncio
import os
from typing import Annotated

from fastapi import APIRouter, Depends

from satquery.agent.concurrency import DeviceGates
from satquery.api.dependencies import get_device_gates
from satquery.api.fixtures import mock_registry
from satquery.api.jobs import JobStore, get_job_store
from satquery.core.config import Settings, get_settings
from satquery.models.loader import (
    BackendKind,
    available_backend,
    backend_config,
    current_backend,
    device_name,
    igpu_masked,
    resolve_torch_device,
    vram_snapshot,
)
from satquery.schemas.api import DeviceInfo, HealthResponse, ModelInfo
from satquery.schemas.enums import HealthStatus

router = APIRouter(tags=["health"])

_MB: int = 1024**2


def _device_info(settings: Settings) -> DeviceInfo:
    """Probe the accelerator, rather than describing the one we hoped for.

    Every field here is read from the driver at request time. The previous
    fixture reported a masked 7900 XTX unconditionally, which made
    ``igpu_masked`` a check that could not fail — the one property a pre-demo
    assertion must not have.
    """
    torch_device = resolve_torch_device(settings.vlm_device)
    snapshot = vram_snapshot(torch_device)
    # §1's nulls rule: a CPU-only box has no VRAM to report, and that is `null`
    # rather than 0 — a zero would assert a measurement nobody took.
    total_mb = round(snapshot.total / _MB) if snapshot.available else None
    used_mb = round(snapshot.used / _MB) if snapshot.available else None
    return DeviceInfo(
        backend="rocm" if snapshot.available else "cpu",
        name=device_name(torch_device),
        hip_visible_devices=os.environ.get("HIP_VISIBLE_DEVICES"),
        vram_total_mb=total_mb,
        vram_used_mb=used_mb,
        igpu_masked=igpu_masked(),
    )


def _models(settings: Settings) -> list[ModelInfo]:
    """The serving path this process would actually use, and whether it is warm.

    ``loaded`` is read from the backend only if one already exists:
    :func:`current_backend` never constructs, so polling health every 15 s
    cannot be the thing that pulls 16 GB onto the card.
    """
    config = backend_config(settings)
    kind = config.kind or available_backend(config)
    if kind is BackendKind.NONE:
        return []
    backend = current_backend()
    name = (
        config.gguf_path.name
        if kind is BackendKind.LLAMACPP and config.gguf_path
        else config.model_id
    )
    return [
        ModelInfo(
            name=name,
            backend=kind.value,
            loaded=bool(backend is not None and backend.is_loaded),
            # Nothing in Settings names the adapter, so there is nothing honest
            # to report: `null` says "unknown", where the fixture's "sq-lora-v3"
            # would name a specific adapter this process may not be serving.
            adapter=None,
        )
    ]


@router.get("/health", response_model=HealthResponse, summary="Service, device and model health")
async def health(
    settings: Annotated[Settings, Depends(get_settings)],
    jobs: Annotated[JobStore, Depends(get_job_store)],
    gates: Annotated[DeviceGates, Depends(get_device_gates)],
) -> HealthResponse:
    """Report service health.

    ``device.igpu_masked`` asserts that ``HIP_VISIBLE_DEVICES=0`` took effect. A
    ``false`` there is a red flag before any demo. ``leaked_gpu_permits`` is the
    other one: a timed-out GPU tool whose thread is still on the card.
    """
    registry = mock_registry()
    # The device probe is torch on first call — ~10 s of ROCm initialisation —
    # and the frontend polls this every 15 s. Off the loop, so the first poll
    # after a restart does not freeze every job's SSE stream for its duration.
    device = await asyncio.to_thread(_device_info, settings)
    models = await asyncio.to_thread(_models, settings)
    return HealthResponse(
        status=HealthStatus.DEGRADED if gates.degraded else HealthStatus.OK,
        version=settings.version,
        schema_version=settings.schema_version,
        device=device,
        models=models,
        tools_available=sum(1 for tool in registry.tools if tool.available),
        tools_total=len(registry.tools),
        jobs_running=jobs.running,
        jobs_retained=len(jobs),
        leaked_gpu_permits=gates.leaked_gpu_permits,
    )
