"""A raster below the tools' 32 px floor is refused at ingest with a reason."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from satquery.ingest.errors import ImageTooSmallError
from satquery.ingest.pipeline import SourceImage, ingest


def _tiny(path: Path, size: int) -> Path:
    with rasterio.open(
        path, "w", driver="GTiff", width=size, height=size, count=3, dtype="uint8",
        crs="EPSG:32643", transform=from_origin(500000, 2500000, 10, 10),
    ) as dst:
        dst.write(np.full((3, size, size), 120, dtype=np.uint8))
    return path


def test_a_one_pixel_image_is_refused_with_its_size(tmp_path: Path) -> None:
    path = _tiny(tmp_path / "dot.tif", 1)
    with pytest.raises(ImageTooSmallError) as caught:
        ingest([SourceImage(path=path, filename="dot.tif")])
    assert caught.value.http_status == 422
    assert "1 x 1 px" in caught.value.message


def test_a_32_pixel_image_is_accepted(tmp_path: Path) -> None:
    path = _tiny(tmp_path / "ok.tif", 32)
    result = ingest([SourceImage(path=path, filename="ok.tif")])
    assert result.inputs[0].width == 32
