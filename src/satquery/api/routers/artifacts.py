"""``GET /v1/artifacts/{trace_id}/{artifact_id}.{ext}`` — raw evidence bytes.

Artifacts are content-addressed and never mutate, so they are served with a
one-year immutable cache and a strong ETag. That is what lets the frontend
preload every ``ArtifactRef.url`` without a revalidation storm (API_CONTRACT §4.6).
"""

from __future__ import annotations

from typing import Annotated, Final

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import FileResponse

from satquery.api.dependencies import get_artifact_store
from satquery.ingest.errors import IngestError
from satquery.render.artifact_store import (
    MIME_TYPES,
    ArtifactNotFoundError,
    ArtifactStore,
    TraceEvictedError,
)

router = APIRouter(tags=["artifacts"])

CACHE_CONTROL: Final[str] = "public, max-age=31536000, immutable"
"""Frozen by API_CONTRACT §4.6."""


class ArtifactUnknownError(IngestError):
    """The requested artifact does not exist (HTTP 404)."""

    def __init__(self, detail: str) -> None:
        """Build the 404 payload."""
        super().__init__(
            code="ARTIFACT_NOT_FOUND",
            http_status=404,
            message=detail,
            hint="Check the artifact id against the trace's artifacts array.",
        )


class ArtifactEvictedError(IngestError):
    """The trace's artifacts have been evicted (HTTP 410)."""

    def __init__(self, trace_id: str) -> None:
        """Build the 410 payload."""
        super().__init__(
            code="ARTIFACT_EVICTED",
            http_status=410,
            message="The artifacts for this analysis have been evicted from the store.",
            hint="Re-run the analysis to regenerate its evidence.",
            ref=trace_id,
        )


@router.get(
    "/artifacts/{trace_id}/{artifact_id}.{ext}",
    summary="Fetch one artifact's raw bytes",
    response_class=FileResponse,
    responses={
        200: {"content": {mime: {} for mime in sorted(set(MIME_TYPES.values()))}},
        304: {"description": "Not modified"},
        404: {"description": "Unknown trace or artifact"},
        410: {"description": "The trace has been evicted"},
    },
)
async def get_artifact(
    request: Request,
    trace_id: str,
    artifact_id: str,
    ext: str,
    store: Annotated[ArtifactStore, Depends(get_artifact_store)],
) -> Response:
    """Serve one artifact with immutable caching and conditional-request support."""
    extension = ext.lower()
    if extension not in MIME_TYPES:
        raise ArtifactUnknownError(f"'{ext}' is not a supported artifact format.")

    try:
        blob = store.resolve(trace_id, artifact_id, extension)
    except TraceEvictedError as exc:
        raise ArtifactEvictedError(trace_id) from exc
    except ArtifactNotFoundError as exc:
        raise ArtifactUnknownError("No such artifact.") from exc

    headers = {"Cache-Control": CACHE_CONTROL, "ETag": blob.etag}
    if request.headers.get("if-none-match") == blob.etag:
        return Response(status_code=304, headers=headers)

    return FileResponse(path=blob.path, media_type=blob.mime, headers=headers)
