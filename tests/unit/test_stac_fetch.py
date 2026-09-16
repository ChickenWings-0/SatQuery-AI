"""``ingest.stac_fetch`` against a synthetic local COG — no network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from satquery.ingest.stac_fetch import WindowEmptyError, read_stack


@pytest.fixture
def synthetic_cog(tmp_path: Path) -> dict[str, str]:
    """Three single-band uint16 rasters, 512 px at 10 m in UTM 43N, like S2 bands."""
    hrefs: dict[str, str] = {}
    transform = from_origin(712340.0, 3161780.0, 10.0, 10.0)
    for index, key in enumerate(("B02", "B04", "B08"), start=1):
        path = tmp_path / f"{key}.tif"
        data = np.full((512, 512), index * 1000, dtype=np.uint16)
        data[:256, :256] = index * 2000
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=512,
            width=512,
            count=1,
            dtype="uint16",
            crs="EPSG:32643",
            transform=transform,
            nodata=0,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(data, 1)
        hrefs[key] = str(path)
    return hrefs


def _bbox_wgs84(hrefs: dict[str, str]) -> tuple[float, float, float, float]:
    from rasterio.warp import transform_bounds

    with rasterio.open(hrefs["B04"]) as src:
        return tuple(transform_bounds(src.crs, "EPSG:4326", *src.bounds))  # type: ignore[return-value]


def test_reads_a_window_onto_the_reference_grid(
    synthetic_cog: dict[str, str], tmp_path: Path
) -> None:
    """A 256 px window over the whole raster is written with every band in order."""
    out = tmp_path / "out" / "stack.tif"
    names, facts = read_stack(
        synthetic_cog,
        ("B02", "B04", "B08"),
        "B04",
        _bbox_wgs84(synthetic_cog),
        256,
        out,
        item_id="x",
    )
    assert names == ["B02", "B04", "B08"]
    assert facts["width"] == 256 and facts["height"] == 256
    assert facts["band_count"] == 3
    assert facts["crs"] == "EPSG:32643"
    assert facts["gsd_m"] == pytest.approx(20.0)
    with rasterio.open(out) as src:
        assert src.count == 3
        assert src.dtypes[0] == "uint16"
        assert list(src.descriptions) == ["B02", "B04", "B08"]
        band = src.read(2)
        assert band[10, 10] == 4000  # the bright quadrant of B04
        assert band[200, 200] == 2000


def test_missing_band_is_filled_and_reported(synthetic_cog: dict[str, str], tmp_path: Path) -> None:
    """An absent asset becomes a nodata band and a warning, never a crash."""
    hrefs = {k: v for k, v in synthetic_cog.items() if k != "B08"}
    names, facts = read_stack(
        hrefs, ("B02", "B04", "B08"), "B04", _bbox_wgs84(hrefs), 128, tmp_path / "o.tif"
    )
    assert names == ["B02", "B04", "B08"]
    assert facts["warnings"] == ["asset B08 missing; band filled with nodata"]


def test_disjoint_bbox_is_a_422(synthetic_cog: dict[str, str], tmp_path: Path) -> None:
    """A bbox that misses the raster raises the user-facing error, not a rasterio one."""
    with pytest.raises(WindowEmptyError):
        read_stack(synthetic_cog, ("B04",), "B04", (0.0, 0.0, 1.0, 1.0), 128, tmp_path / "o.tif")
