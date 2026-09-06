"""``GET /v1/health`` — liveness plus device and model state."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from satquery.api.fixtures import mock_health
from satquery.core.config import Settings, get_settings
from satquery.schemas.api import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service, device and model health")
async def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report service health.

    ``device.igpu_masked`` asserts that ``HIP_VISIBLE_DEVICES=0`` took effect. A
    ``false`` there is a red flag before any demo.
    """
    return mock_health(version=settings.version, schema_version=settings.schema_version)
