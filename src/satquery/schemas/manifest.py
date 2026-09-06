"""Per-image ingestion manifest (API_CONTRACT §3.1)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from satquery.schemas.enums import ImageRole, Modality


class InputManifest(BaseModel):
    """Everything the ingestion layer learned about one uploaded image.

    A field that is *inapplicable* is ``None``. A field that is *unknown* is
    ``None`` **and** carries an entry in :attr:`warnings`.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable within a trace: img_0, img_1, ... in upload order.")
    role: ImageRole | None = Field(
        default=None, description="Server-assigned unless overridden via options.roles."
    )
    filename: str = Field(description="Original client filename, sanitised.")
    sha256: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$", description="Of the raw bytes."
    )
    size_bytes: int = Field(ge=0)
    driver: str = Field(description="rasterio driver: GTiff, PNG, JPEG.")
    modality: Modality
    modality_confidence: float = Field(ge=0.0, le=1.0)
    sensor_guess: str | None = Field(
        default=None, description="e.g. 'Sentinel-2 L2A', 'Sentinel-1 GRD', 'Cartosat-2S'."
    )
    crs: str | None = Field(
        default=None, description="e.g. 'EPSG:32643'; None for non-georeferenced input."
    )
    transform: list[float] | None = Field(
        default=None,
        min_length=6,
        max_length=6,
        description="rasterio/GDAL affine 6-tuple [a, b, c, d, e, f].",
    )
    bounds_native: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="[minx, miny, maxx, maxy] in the raster's own CRS.",
    )
    bounds_wgs84: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="[min_lon, min_lat, max_lon, max_lat], lon first, EPSG:4326.",
    )
    gsd_m: float | None = Field(default=None, gt=0.0, description="Ground sample distance, metres.")
    width: int = Field(gt=0, description="Pixels.")
    height: int = Field(gt=0, description="Pixels.")
    band_count: int = Field(gt=0)
    dtype: str = Field(description="uint8, uint16, int16, float32.")
    band_names: list[str] | None = Field(
        default=None, description='["B02","B03",...] or ["VV","VH"].'
    )
    nodata_pct: float = Field(ge=0.0, le=100.0)
    acquisition_time: datetime | None = Field(
        default=None, description="From TIFF/GDAL metadata when present."
    )
    is_georeferenced: bool = Field(description="crs is not None and transform is not None.")
    warnings: list[str] = Field(default_factory=list, description="May be empty, never None.")
