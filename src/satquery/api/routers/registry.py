"""``GET /v1/registry`` — the live tool catalogue behind the capabilities panel."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from satquery.core.config import Settings, get_settings
from satquery.registry.registry import default_registry
from satquery.schemas.api import RegistryResponse
from satquery.schemas.enums import TaskType

router = APIRouter(tags=["registry"])


@router.get("/registry", response_model=RegistryResponse, summary="List every registered tool")
async def registry(settings: Annotated[Settings, Depends(get_settings)]) -> RegistryResponse:
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
        adapter_version=settings.adapter_name,
    )
