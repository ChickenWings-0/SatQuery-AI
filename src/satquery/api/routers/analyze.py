"""``POST /v1/analyze`` — the full pipeline, with a full audit trace."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from satquery.agent.concurrency import DeviceGates
from satquery.agent.pipeline import AnalysisRequest
from satquery.agent.pipeline import analyze as run_analysis
from satquery.api.dependencies import get_artifact_store, get_device_gates, get_trace_store
from satquery.api.uploads import parse_options, spooled_uploads_async
from satquery.core.config import Settings, get_settings
from satquery.evidence.citation_validator import CitationPolicy
from satquery.ingest.pipeline import ingest
from satquery.render.artifact_store import ArtifactStore
from satquery.schemas.api import AnalyzeResponse
from satquery.trace.store import TraceStore

QUERY_FORM = Form(min_length=1, max_length=1000, description="1-1000 chars, UTF-8.")
"""The contract's stated bounds, enforced. They used to be description only, so an
empty query reached the classifier."""

router = APIRouter(tags=["analyze"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Answer a question about one or two images",
)
async def analyze(
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[ArtifactStore, Depends(get_artifact_store)],
    traces: Annotated[TraceStore, Depends(get_trace_store)],
    gates: Annotated[DeviceGates, Depends(get_device_gates)],
    images: list[UploadFile] = File(description="1-2 files. GeoTIFF preferred; PNG/JPEG accepted."),
    query: str = QUERY_FORM,
    options: str | None = Form(default=None, description="JSON, see AnalyzeOptions."),
) -> AnalyzeResponse:
    """Run the full pipeline and return the answer, its evidence and the trace.

    A tool failure never produces a 5xx: it produces ``200`` with a ``DEGRADED``
    or ``SKIPPED`` execution, a populated ``errors[]`` and a reduced confidence.
    The single exception is a step the policy entry declares non-optional — only
    the renderer today — because without rendered views there is no evidence to
    reason over at all.
    """
    parsed = parse_options(options)

    async with spooled_uploads_async(images, max_bytes=settings.max_upload_mb << 20) as sources:
        # rasterio reads plus phase correlation, off the loop: one large GeoTIFF
        # pair used to stall every SSE heartbeat in the process.
        result = await asyncio.to_thread(
            ingest,
            sources,
            pair_type_hint=parsed.pair_type,
            roles=parsed.roles,
            max_images=settings.max_images,
        )
        analysis = await run_analysis(
            AnalysisRequest(
                query=query.strip(),
                sources=sources,
                ingest=result,
                seed=parsed.seed,
                citation_policy=CitationPolicy(settings.citation_policy),
                allow_generic_fallback=settings.allow_generic_fallback,
                enable_tools=parsed.enable_tools,
                disable_tools=parsed.disable_tools,
                max_latency_ms=parsed.max_latency_ms,
                ignored_options=parsed.ignored(),
            ),
            store=store,
            traces=traces,
            gates=gates,
        )

    trace = analysis.trace
    return AnalyzeResponse(
        trace_id=trace.trace_id,
        answer=trace.answer,
        artifacts=trace.artifacts if parsed.include_rendered_views else [],
        confidence=trace.confidence,
        compatibility=trace.compatibility,
        resolved_task=trace.resolved_task,
        trace=trace if parsed.include_trace else None,
    )
