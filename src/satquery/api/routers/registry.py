"""``GET /v1/registry`` — the live tool catalogue behind the capabilities panel."""

from __future__ import annotations

from fastapi import APIRouter

from satquery.api.fixtures import ADAPTER_VERSION
from satquery.registry.registry import default_registry
from satquery.schemas.api import RegistryResponse
from satquery.schemas.enums import TaskType

router = APIRouter(tags=["registry"])


@router.get("/registry", response_model=RegistryResponse, summary="List every registered tool")
async def registry() -> RegistryResponse:
    """Return every tool, available or not.

    Entries with ``available: false`` must be rendered as disabled rather than
    hidden — the registry being real, and honest about what is not loaded yet,
    is part of the story.
    """
    loaded = default_registry()
    return RegistryResponse(
        registry_version=loaded.version,
        tools=list(loaded.tools.values()),
        task_types=list(TaskType),
        adapter_version=ADAPTER_VERSION,
    )
