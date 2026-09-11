"""``GET /v1/health`` — liveness plus device and model state."""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import APIRouter, Depends

from satquery.api.fixtures import mock_registry
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
async def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report service health.

    ``device.igpu_masked`` asserts that ``HIP_VISIBLE_DEVICES=0`` took effect. A
    ``false`` there is a red flag before any demo.
    """
    registry = mock_registry()
    return HealthResponse(
        status=HealthStatus.OK,
        version=settings.version,
        schema_version=settings.schema_version,
        device=_device_info(settings),
        models=_models(settings),
        tools_available=sum(1 for tool in registry.tools if tool.available),
        tools_total=len(registry.tools),
    )
