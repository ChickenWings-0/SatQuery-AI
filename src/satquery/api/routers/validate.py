"""``POST /v1/validate`` — GPU-free ingestion and compatibility pre-flight."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from satquery.api.uploads import parse_options, spooled_uploads
from satquery.core.config import Settings, get_settings
from satquery.ingest.pipeline import ingest
from satquery.schemas.api import ValidateResponse
from satquery.schemas.enums import PairType

router = APIRouter(tags=["validate"])


@router.post(
    "/validate",
    response_model=ValidateResponse,
    summary="Check whether the uploaded images can be analysed together",
)
async def validate(
    settings: Annotated[Settings, Depends(get_settings)],
    images: list[UploadFile] = File(description="1-2 files. GeoTIFF preferred; PNG/JPEG accepted."),
    options: str | None = Form(
        default=None, description="JSON; only pair_type_hint and roles are honoured here."
    ),
) -> ValidateResponse:
    """Run ingestion and compatibility checks only — no tools, no GPU.

    Call this on file select, before the user types: ``supported_tasks`` tells the
    UI which questions to offer.
    """
    parsed = parse_options(options)
    hint = PairType(parsed.pair_type_hint) if parsed.pair_type_hint else None

    with spooled_uploads(images, max_bytes=settings.max_upload_mb << 20) as sources:
        result = ingest(
            sources,
            pair_type_hint=hint,
            roles=parsed.roles,
            max_images=settings.max_images,
        )

    return ValidateResponse(
        inputs=result.inputs,
        compatibility=result.compatibility,
        supported_tasks=result.supported_tasks,
        warnings=result.warnings,
    )
