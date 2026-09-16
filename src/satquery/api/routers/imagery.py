"""``POST /v1/imagery/fetch`` and ``GET /v1/imagery/{fetch_id}/{name}`` (Track 4.3).

Additive to the frozen 1.0 contract: a new router, new schemas, nothing
changed downstream. The frontend asks for one or two STAC items over a bbox,
the server writes clipped GeoTIFFs under the artifact root, and the frontend
downloads them and uploads them through ``/v1/validate`` exactly as it would a
dropped file — so the fetched scene is not a special path.
"""

from __future__ import annotations

import asyncio
import secrets
from pathlib import Path
from typing import Annotated, Final

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from satquery.api.routers.artifacts import ArtifactUnknownError
from satquery.core.config import Settings, get_settings
from satquery.ingest.stac_fetch import PlanetaryComputer, fetch_item
from satquery.schemas.api import ImageryFetchRequest, ImageryFetchResponse, ImageryFile
from satquery.schemas.trace import WarningItem

router = APIRouter(tags=["imagery"])

SUBDIR: Final[str] = "imagery"

_pc: PlanetaryComputer | None = None


def _client(settings: Settings) -> PlanetaryComputer:
    global _pc  # noqa: PLW0603 - one HTTP client and one token cache per process
    if _pc is None or _pc.stac_url != settings.imagery_stac_url.rstrip("/"):
        _pc = PlanetaryComputer(settings.imagery_stac_url, timeout_s=settings.imagery_timeout_s)
    return _pc


def _fetch_dir(settings: Settings, fetch_id: str) -> Path:
    return Path(settings.artifact_root) / SUBDIR / fetch_id


@router.post(
    "/imagery/fetch",
    response_model=ImageryFetchResponse,
    summary="Clip one or two Planetary Computer scenes to a bbox as GeoTIFFs",
)
async def fetch_imagery(
    body: ImageryFetchRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ImageryFetchResponse:
    """Read a window out of each item's COGs and write it as a GeoTIFF.

    Sentinel-2 comes back as the twelve BigEarthNet-v2 bands, Sentinel-1 as
    VV/VH — the band orders the sensor fingerprints recognise — so the files go
    through pre-flight exactly like an upload.
    """
    fetch_id = secrets.token_hex(8)
    out_dir = _fetch_dir(settings, fetch_id)
    max_px = min(body.max_px or settings.imagery_max_px, settings.imagery_max_px)
    bbox = (body.bbox[0], body.bbox[1], body.bbox[2], body.bbox[3])
    pc = _client(settings)

    fetched = await asyncio.gather(
        *(
            asyncio.to_thread(fetch_item, pc, item.collection, item.id, bbox, max_px, out_dir)
            for item in body.items
        )
    )

    files: list[ImageryFile] = []
    warnings: list[WarningItem] = []
    for raster in fetched:
        files.append(
            ImageryFile(
                name=raster.path.name,
                url=f"{settings.api_prefix}/imagery/{fetch_id}/{raster.path.name}",
                size_bytes=raster.path.stat().st_size,
                collection=raster.collection,
                item_id=raster.item_id,
                datetime=raster.datetime,
                crs=raster.crs,
                gsd_m=raster.gsd_m,
                width=raster.width,
                height=raster.height,
                band_count=raster.band_count,
                bounds_wgs84=list(raster.bounds_wgs84),
                orbit_state=raster.orbit_state,
            )
        )
        warnings.extend(
            WarningItem(code="IMAGERY_BAND_MISSING", message=message, ref=raster.item_id)
            for message in raster.warnings
        )
    return ImageryFetchResponse(fetch_id=fetch_id, files=files, warnings=warnings)


@router.get(
    "/imagery/{fetch_id}/{name}",
    summary="Download one fetched GeoTIFF",
    response_class=FileResponse,
)
async def download_imagery(
    fetch_id: str,
    name: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> FileResponse:
    """Serve a file written by :func:`fetch_imagery`; ids are opaque hex, names are sanitised."""
    if not (fetch_id.isalnum() and len(fetch_id) == 16) or "/" in name or ".." in name:
        raise ArtifactUnknownError("No such imagery.")
    path = _fetch_dir(settings, fetch_id) / name
    if not path.is_file():
        raise ArtifactUnknownError("No such imagery.")
    return FileResponse(path, media_type="image/tiff", filename=name)
