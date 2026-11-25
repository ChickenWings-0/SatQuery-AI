"""Pre-flight returns a quicklook for every input, GeoTIFFs included."""

from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from satquery.api.app import create_app
from satquery.core.config import Settings


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(Settings(access_code=None))) as test_client:
        yield test_client


def _s2(path: Path) -> Path:
    rng = np.random.default_rng(3)
    with rasterio.open(
        path, "w", driver="GTiff", width=96, height=80, count=12, dtype="uint16",
        crs="EPSG:32643", transform=from_origin(500000, 2500000, 10, 10),
    ) as dst:
        for band in range(1, 13):
            dst.write(rng.integers(200, 3000, (80, 96)).astype(np.uint16), band)
    return path


def test_a_sentinel2_geotiff_gets_a_jpeg_preview(client: TestClient, tmp_path: Path) -> None:
    path = _s2(tmp_path / "scene.tif")
    with path.open("rb") as handle:
        response = client.post(
            "/v1/validate", files={"images": ("scene.tif", handle, "image/tiff")}
        )
    assert response.status_code == 200, response.text
    previews = response.json()["previews"]
    assert set(previews) == {"img_0"}
    header, payload = previews["img_0"].split(",", 1)
    assert header == "data:image/jpeg;base64"
    assert base64.b64decode(payload)[:3] == b"\xff\xd8\xff"  # JPEG magic
