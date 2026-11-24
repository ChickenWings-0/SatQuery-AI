"""``class_outliner`` end to end: "Outline the water bodies" must outline water.

Synthetic scenes with a known lake — a rectangle covering exactly 20 % of the
frame — on a Sentinel-2 12-band stack and on a Sentinel-1 VV/VH pair. The
outline must find it at about that size, in the centre, as one region, draw an
overlay, and lead the answer with those measured numbers.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from satquery.agent.aggregator import outline_sentences
from satquery.api.app import create_app
from satquery.core.config import Settings
from satquery.evidence.fact_sheet import FactSheet
from satquery.tools.class_outliner import canonical_class, location_of

SIZE = 200
LAKE = (slice(60, 140), slice(50, 150))  # 80 x 100 px = 20 % of 200 x 200


def _write(path: Path, bands: list[np.ndarray], names: list[str] | None = None) -> Path:
    profile = {
        "driver": "GTiff",
        "height": SIZE,
        "width": SIZE,
        "count": len(bands),
        "dtype": bands[0].dtype.name,
        "crs": "EPSG:32643",
        "transform": from_origin(500000.0, 2550000.0, 10.0, 10.0),
    }
    with rasterio.open(path, "w", **profile) as dst:
        for index, band in enumerate(bands, start=1):
            dst.write(band, index)
            if names:
                dst.set_band_description(index, names[index - 1])
    return path


@pytest.fixture
def s2_lake(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    bands = []
    for index in range(1, 13):
        land = {8: 2500, 11: 2500, 12: 2200}.get(index, 1500)
        water = {2: 900, 3: 1200, 4: 600, 8: 300, 11: 200, 12: 150}.get(index, 400)
        band = np.full((SIZE, SIZE), land, dtype=np.float32) + rng.normal(0, 30, (SIZE, SIZE))
        band[LAKE] = water + rng.normal(0, 20, band[LAKE].shape)
        bands.append(np.clip(band, 1, 10000).astype(np.uint16))
    return _write(tmp_path / "s2_lake.tif", bands)


@pytest.fixture
def s1_lake(tmp_path: Path) -> Path:
    rng = np.random.default_rng(8)
    vv = np.full((SIZE, SIZE), 0.10, dtype=np.float32) * rng.gamma(4.0, 0.25, (SIZE, SIZE))
    vh = np.full((SIZE, SIZE), 0.02, dtype=np.float32) * rng.gamma(4.0, 0.25, (SIZE, SIZE))
    vv[LAKE] = 0.004
    vh[LAKE] = 0.001
    bands = [vv.astype(np.float32), vh.astype(np.float32)]
    return _write(tmp_path / "s1_lake.tif", bands, ["VV", "VH"])


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(Settings(access_code=None))) as test_client:
        yield test_client


def _outline(client: TestClient, path: Path) -> dict:
    with path.open("rb") as handle:
        response = client.post(
            "/v1/analyze",
            files={"images": (path.name, handle, "image/tiff")},
            data={
                "query": "Outline the water bodies",
                "options": json.dumps({"include_trace": True}),
            },
        )
    assert response.status_code == 200, response.text
    return response.json()


def _scalars(body: dict) -> dict:
    (execution,) = [e for e in body["trace"]["executions"] if e["tool"] == "class_outliner"]
    assert execution["status"] == "OK", execution
    return execution["scalars"]


@pytest.mark.parametrize("scene", ["s2_lake", "s1_lake"])
def test_outline_the_water_bodies_finds_the_lake(
    client: TestClient, scene: str, request: pytest.FixtureRequest
) -> None:
    body = _outline(client, request.getfixturevalue(scene))
    assert body["resolved_task"]["primary"] == "SEGMENTATION"
    scalars = _scalars(body)
    assert scalars["outline_class"] == "water"
    assert scalars["water_regions"] == 1
    assert 17.0 <= scalars["water_outline_fraction_pct"] <= 23.0
    assert scalars["water_largest_region_location"] == "centre"
    overlays = [a for a in body["trace"]["artifacts"] if a["type"] == "OVERLAY_PNG"]
    assert overlays and "outlined" in overlays[0]["label"]
    assert overlays[0]["url"]
    assert body["answer"]["text"].startswith("Open water covers")
    assert not body["answer"]["uncited_numeric_spans"]


def test_canonical_class_maps_query_words() -> None:
    assert canonical_class("water bodies") == "water"
    assert canonical_class("floods") == "water"
    assert canonical_class("Trees") == "vegetation"
    assert canonical_class("buildings") == "built_up"
    assert canonical_class("aircraft") is None
    assert canonical_class(None) is None


def test_location_of_names_the_third() -> None:
    mask = np.zeros((9, 9), dtype=bool)
    mask[0:2, 0:2] = True
    assert location_of(mask) == "top-left"
    mask[:] = False
    mask[4, 4] = True
    assert location_of(mask) == "centre"
    mask[:] = False
    mask[4, 8] = True
    assert location_of(mask) == "middle-right"
    assert location_of(np.zeros((3, 3), dtype=bool)) == "nowhere"


def test_outline_sentences_say_none_found_honestly() -> None:
    sheet = FactSheet()
    assert outline_sentences(sheet) == ""
