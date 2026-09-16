"""Window reads out of Planetary Computer COGs (ROADMAP Track 4.3).

A Sentinel-2 tile is ~1 GB; the pipeline wants a ≤ 1024 px window. ``rasterio``
reads that window over HTTP in a handful of range requests, which the browser
cannot do, so the fetch is server-side and the result is an ordinary GeoTIFF
the frontend uploads through ``/v1/validate`` like any other file.

The band order written is deliberately the one the sensor fingerprints in
:mod:`satquery.ingest.modality` recognise:

* ``sentinel-2-l2a`` → the twelve BigEarthNet-v2 bands (B01…B12 without B10)
  as ``uint16``, so the file fingerprints as the sensor the adapter trained on.
* ``sentinel-1-rtc`` / ``sentinel-1-grd`` → ``VV``, ``VH`` as ``float32`` with the
  polarisation in the band description, which is what the SAR fingerprint reads.

No STAC client library: the two requests involved are a ``GET`` for the item
and a ``GET`` for the collection's SAS token.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import httpx
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from satquery.ingest.errors import IngestError

log = logging.getLogger(__name__)

USER_AGENT: Final[str] = "SatQuery-AI/0.1 (+https://github.com/ChickenWings-0/SatQuery-AI)"

#: Asset keys per collection, in the band order the file is written in.
COLLECTION_ASSETS: Final[dict[str, tuple[str, ...]]] = {
    "sentinel-2-l2a": (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ),
    "sentinel-1-rtc": ("vv", "vh"),
    "sentinel-1-grd": ("vv", "vh"),
}

#: The band that sets the output grid: the finest resolution one.
REFERENCE_ASSET: Final[dict[str, str]] = {
    "sentinel-2-l2a": "B04",
    "sentinel-1-rtc": "vv",
    "sentinel-1-grd": "vv",
}

_GDAL_ENV: Final[dict[str, Any]] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_USERAGENT": USER_AGENT,
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": str(64 << 20),
}


class ImageryError(IngestError):
    """A fetch that failed for a reason the user can act on."""


class UpstreamUnreachableError(ImageryError):
    """Planetary Computer did not answer (HTTP 502)."""

    def __init__(self, what: str) -> None:
        """Build the 502 payload."""
        super().__init__(
            code="IMAGERY_UPSTREAM_UNREACHABLE",
            http_status=502,
            message=f"Planetary Computer did not answer while fetching {what}.",
            hint="Check the venue's internet connection, or load a sample scene instead.",
        )


class ItemNotFoundError(ImageryError):
    """The STAC item does not exist (HTTP 404)."""

    def __init__(self, collection: str, item_id: str) -> None:
        """Build the 404 payload."""
        super().__init__(
            code="IMAGERY_ITEM_NOT_FOUND",
            http_status=404,
            message=f"No item {item_id} in {collection}.",
            hint="Search again; the catalogue may have re-issued the scene.",
            ref=item_id,
        )


class UnsupportedCollectionError(ImageryError):
    """A collection this fetcher does not know how to stack (HTTP 422)."""

    def __init__(self, collection: str) -> None:
        """Build the 422 payload."""
        super().__init__(
            code="IMAGERY_COLLECTION_UNSUPPORTED",
            http_status=422,
            message=f"Collection {collection!r} is not supported.",
            hint=f"Use one of: {', '.join(sorted(COLLECTION_ASSETS))}.",
        )


class WindowEmptyError(ImageryError):
    """The requested bbox does not intersect the item (HTTP 422)."""

    def __init__(self, item_id: str) -> None:
        """Build the 422 payload."""
        super().__init__(
            code="IMAGERY_WINDOW_EMPTY",
            http_status=422,
            message="The requested area is not covered by this scene's data.",
            hint="Pick a scene whose footprint covers the place you searched for.",
            ref=item_id,
        )


@dataclass(frozen=True)
class FetchedRaster:
    """What :func:`fetch_item` wrote, with the manifest-shaped facts the UI wants."""

    path: Path
    collection: str
    item_id: str
    datetime: str
    crs: str
    gsd_m: float
    width: int
    height: int
    band_count: int
    bounds_wgs84: tuple[float, float, float, float]
    orbit_state: str | None
    warnings: list[str] = field(default_factory=list)


@dataclass
class SasToken:
    """A signed-URL token for one collection, with its expiry."""

    token: str
    expires_at: float


class PlanetaryComputer:
    """The two HTTP calls this module makes, with the SAS token cached per collection."""

    def __init__(self, stac_url: str, *, timeout_s: float = 30.0) -> None:
        """Bind to a STAC root; the SAS endpoint is its sibling."""
        self.stac_url = stac_url.rstrip("/")
        self.sas_url = self.stac_url.rsplit("/stac/", 1)[0] + "/sas/v1/token"
        self._client = httpx.Client(timeout=timeout_s, headers={"User-Agent": USER_AGENT})
        self._tokens: dict[str, SasToken] = {}

    def item(self, collection: str, item_id: str) -> dict[str, Any]:
        """Fetch one STAC item as JSON."""
        try:
            response = self._client.get(f"{self.stac_url}/collections/{collection}/items/{item_id}")
        except httpx.HTTPError as error:
            raise UpstreamUnreachableError(f"item {item_id}") from error
        if response.status_code == 404:
            raise ItemNotFoundError(collection, item_id)
        if response.status_code >= 500:
            raise UpstreamUnreachableError(f"item {item_id}")
        response.raise_for_status()
        return dict(response.json())

    def token(self, collection: str) -> str:
        """Return a SAS token for *collection*, refreshed a minute before expiry."""
        cached = self._tokens.get(collection)
        if cached and cached.expires_at - 60 > time.time():
            return cached.token
        try:
            response = self._client.get(f"{self.sas_url}/{collection}")
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise UpstreamUnreachableError(f"a token for {collection}") from error
        payload = response.json()
        expiry = payload.get("msft:expiry")
        expires_at = time.time() + 3600
        if isinstance(expiry, str):
            try:
                from datetime import datetime

                expires_at = datetime.fromisoformat(expiry.replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass
        token = SasToken(token=str(payload["token"]), expires_at=expires_at)
        self._tokens[collection] = token
        return token.token

    def sign(self, href: str, collection: str) -> str:
        """Append the collection's SAS token to an asset href."""
        return f"{href}{'&' if '?' in href else '?'}{self.token(collection)}"

    def close(self) -> None:
        """Release the HTTP client."""
        self._client.close()


def _asset_hrefs(item: dict[str, Any], collection: str) -> dict[str, str]:
    keys = COLLECTION_ASSETS.get(collection)
    if keys is None:
        raise UnsupportedCollectionError(collection)
    assets = item.get("assets") or {}
    hrefs: dict[str, str] = {}
    for key in keys:
        asset = assets.get(key)
        if isinstance(asset, dict) and isinstance(asset.get("href"), str):
            hrefs[key] = asset["href"]
    return hrefs


def _all_nodata(data: np.ndarray, nodata: float | None) -> bool:
    if data.size == 0:
        return True
    if np.issubdtype(data.dtype, np.floating) and np.isnan(data).all():
        return True
    return nodata is not None and bool((data == nodata).all())


def _out_shape(window: Window, max_px: int) -> tuple[int, int]:
    height = max(1, int(round(window.height)))
    width = max(1, int(round(window.width)))
    scale = min(1.0, max_px / max(width, height))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def read_stack(
    hrefs: dict[str, str],
    order: tuple[str, ...],
    reference: str,
    bbox_wgs84: tuple[float, float, float, float],
    max_px: int,
    out_path: Path,
    *,
    item_id: str = "",
    tags: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Read every band in *order* over *bbox_wgs84* onto the reference band's grid.

    Works on any rasterio-openable href, local or ``https://`` — which is what
    keeps the unit test offline: a synthetic COG on disk goes through exactly
    the code path a Planetary Computer asset does.

    Returns:
        The band descriptions written, and the profile facts the caller reports.
    """
    if reference not in hrefs:
        raise WindowEmptyError(item_id)
    warnings: list[str] = []
    with rasterio.Env(**_GDAL_ENV):
        with rasterio.open(hrefs[reference]) as ref:
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", ref.crs, *bbox_wgs84, densify_pts=21
            )
            # Clip to the raster.
            rb = ref.bounds
            left, right = max(left, rb.left), min(right, rb.right)
            bottom, top = max(bottom, rb.bottom), min(top, rb.top)
            if right <= left or top <= bottom:
                raise WindowEmptyError(item_id)
            window = from_bounds(left, bottom, right, top, transform=ref.transform)
            window = window.round_offsets().round_lengths()
            out_h, out_w = _out_shape(window, max_px)
            transform = ref.window_transform(window) * rasterio.Affine.scale(
                window.width / out_w, window.height / out_h
            )
            crs = ref.crs
            dtype = ref.dtypes[0]
            nodata = ref.nodata
            ref_bounds_geo = (left, bottom, right, top)

        bands: list[np.ndarray] = []
        names: list[str] = []
        for key in order:
            href = hrefs.get(key)
            if href is None:
                warnings.append(f"asset {key} missing; band filled with nodata")
                bands.append(np.zeros((out_h, out_w), dtype=dtype))
                names.append(key)
                continue
            with rasterio.open(href) as src:
                if src.crs != crs:
                    l2, b2, r2, t2 = transform_bounds(crs, src.crs, *ref_bounds_geo, densify_pts=21)
                else:
                    l2, b2, r2, t2 = ref_bounds_geo
                win = from_bounds(l2, b2, r2, t2, transform=src.transform)
                data = src.read(
                    1,
                    window=win,
                    out_shape=(out_h, out_w),
                    resampling=Resampling.bilinear,
                    boundless=True,
                    fill_value=src.nodata if src.nodata is not None else 0,
                )
                if key == reference and _all_nodata(data, src.nodata):
                    # The bbox is inside the item's rectangle but outside its
                    # swath — a rotated SAR footprint's empty corner.
                    raise WindowEmptyError(item_id)
                bands.append(data.astype(dtype, copy=False))
                names.append(key)

        profile = {
            "driver": "GTiff",
            "height": out_h,
            "width": out_w,
            "count": len(bands),
            "dtype": dtype,
            "crs": crs,
            "transform": transform,
            "compress": "deflate",
            "predictor": 2 if np.issubdtype(np.dtype(dtype), np.integer) else 3,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        if nodata is not None:
            profile["nodata"] = nodata
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            if tags:
                dst.update_tags(**tags)
            for index, (band, name) in enumerate(zip(bands, names, strict=True), start=1):
                dst.write(band, index)
                dst.set_band_description(index, name.upper() if name in {"vv", "vh"} else name)
        gsd = float(abs(transform.a))
        bounds_wgs84 = transform_bounds(crs, "EPSG:4326", *ref_bounds_geo, densify_pts=21)
    facts = {
        "crs": str(crs),
        "gsd_m": gsd,
        "width": out_w,
        "height": out_h,
        "band_count": len(bands),
        "bounds_wgs84": tuple(float(v) for v in bounds_wgs84),
        "warnings": warnings,
    }
    return names, facts


def fetch_item(
    pc: PlanetaryComputer,
    collection: str,
    item_id: str,
    bbox_wgs84: tuple[float, float, float, float],
    max_px: int,
    out_dir: Path,
) -> FetchedRaster:
    """Resolve one item's assets, sign them, and write the clipped stack."""
    order = COLLECTION_ASSETS.get(collection)
    if order is None:
        raise UnsupportedCollectionError(collection)
    item = pc.item(collection, item_id)
    hrefs = {key: pc.sign(href, collection) for key, href in _asset_hrefs(item, collection).items()}
    props = item.get("properties") or {}
    suffix = "s2" if collection.startswith("sentinel-2") else "s1"
    safe_id = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in item_id)[:120]
    out_path = out_dir / f"{suffix}_{safe_id}.tif"
    started = time.monotonic()
    acquired = str(props.get("datetime") or "")
    # The acquisition time rides in the file, so the temporal-ordering check
    # can tell T1 from T2 without the user naming them.
    tags = {
        "ACQUISITION_DATETIME": acquired,
        "STAC_ITEM_ID": item_id,
        "STAC_COLLECTION": collection,
    }
    if props.get("sat:orbit_state"):
        tags["SAT_ORBIT_STATE"] = str(props["sat:orbit_state"])
    names, facts = read_stack(
        hrefs,
        order,
        REFERENCE_ASSET[collection],
        bbox_wgs84,
        max_px,
        out_path,
        item_id=item_id,
        tags=tags,
    )
    log.info(
        "imagery.fetched",
        extra={
            "item": item_id,
            "bands": len(names),
            "seconds": round(time.monotonic() - started, 1),
        },
    )
    return FetchedRaster(
        path=out_path,
        collection=collection,
        item_id=item_id,
        datetime=acquired,
        orbit_state=props.get("sat:orbit_state"),
        **facts,
    )
