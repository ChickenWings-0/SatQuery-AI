#!/usr/bin/env python
"""Generate synthetic GeoTIFFs with known geometry for the ingestion tests.

Every fixture has a *known* answer: a declared CRS, an exact ground sample
distance, and — where relevant — an injected pixel shift or a footprint overlap
computed to land in a specific band of the compatibility thresholds. That is what
makes the assertions in ``tests/unit/test_ingest.py`` meaningful rather than
circular.

The scene content is deterministic (seeded) textured noise plus hard-edged
structures, because phase correlation needs edges to lock onto.

Usage:  uv run python scripts/make_synthetic_fixtures.py [output_dir]
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject

UTM43N: Final[str] = "EPSG:32643"
WGS84: Final[str] = "EPSG:4326"

# Origin of the reference scene, in EPSG:32643 metres (Delhi NCR).
ORIGIN_X: Final[float] = 712_340.0
ORIGIN_Y: Final[float] = 3_161_780.0

S2_GSD: Final[float] = 10.0
CARTOSAT_GSD: Final[float] = 0.65
SIZE: Final[int] = 256
INJECTED_SHIFT_PX: Final[int] = 3

DEFAULT_OUTPUT_DIR: Final[Path] = Path("data/fixtures/synthetic")


@dataclass(frozen=True)
class Scene:
    """A generated fixture and the ground truth a test can assert against."""

    name: str
    path: Path
    description: str
    truth: dict[str, Any]


def _texture(size: int, seed: int) -> npt.NDArray[np.float32]:
    """Deterministic textured field: smooth background plus hard-edged structures.

    Phase correlation locks onto edges, so pure smooth noise would make the
    shift-recovery test unreliable for reasons that have nothing to do with the
    code under test.
    """
    rng = np.random.default_rng(seed)
    field = rng.normal(0.5, 0.12, size=(size, size)).astype(np.float32)

    # Low-frequency background: an upsampled coarse grid, box-smoothed.
    coarse = rng.normal(0.5, 0.25, size=(size // 16, size // 16)).astype(np.float32)
    background = np.kron(coarse, np.ones((16, 16), dtype=np.float32))[:size, :size]
    field = 0.4 * field + 0.6 * background

    # Hard structures: rectangles (buildings) and a diagonal ribbon (a road).
    for _ in range(12):
        y = int(rng.integers(8, size - 40))
        x = int(rng.integers(8, size - 40))
        h = int(rng.integers(10, 32))
        w = int(rng.integers(10, 32))
        field[y : y + h, x : x + w] = float(rng.uniform(0.75, 1.0))

    rows, cols = np.indices((size, size))
    road = np.abs((rows - cols) % size - size // 2) < 4
    field[road] = 0.15

    return np.clip(field, 0.0, 1.0)


def _shifted(field: npt.NDArray[np.float32], dy: int, dx: int) -> npt.NDArray[np.float32]:
    """Roll the scene content by a whole number of pixels, wrapping at the edges."""
    return np.roll(np.roll(field, dy, axis=0), dx, axis=1)


def _write(
    path: Path,
    bands: Sequence[npt.NDArray[Any]],
    transform: Affine,
    crs: str | None,
    dtype: str,
    band_names: Sequence[str] | None = None,
    nodata: float | None = None,
    tags: dict[str, str] | None = None,
    driver: str = "GTiff",
) -> Path:
    """Write a raster with the given geometry, band names and GDAL tags."""
    path.parent.mkdir(parents=True, exist_ok=True)
    profile: dict[str, Any] = {
        "driver": driver,
        "width": bands[0].shape[1],
        "height": bands[0].shape[0],
        "count": len(bands),
        "dtype": dtype,
    }
    if driver == "GTiff":
        profile["compress"] = "deflate"
    if crs is not None:
        profile["crs"] = CRS.from_string(crs)
        profile["transform"] = transform
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dataset:
        for index, band in enumerate(bands, start=1):
            dataset.write(band.astype(dtype), index)
            if band_names:
                dataset.set_band_description(index, band_names[index - 1])
        if tags:
            dataset.update_tags(**tags)
    return path


def _s2_bands(field: npt.NDArray[np.float32], count: int = 12) -> list[npt.NDArray[np.uint16]]:
    """Fan one scene out into Sentinel-2-like uint16 reflectance bands.

    Each band is a distinct linear function of the scene so band statistics differ
    but the geometry stays identical — exactly what a multispectral stack looks like.
    """
    bands = []
    for index in range(count):
        gain = 2_000.0 + 380.0 * index
        offset = 120.0 * (index % 4)
        bands.append(np.clip(field * gain + offset, 0, 10_000).astype(np.uint16))
    return bands


def _transform(gsd: float, origin_x: float = ORIGIN_X, origin_y: float = ORIGIN_Y) -> Affine:
    """North-up affine at *gsd* metres per pixel."""
    return Affine(gsd, 0.0, origin_x, 0.0, -gsd, origin_y)


def _warp_to_wgs84(source: Path, destination: Path) -> Path:
    """Reproject an existing fixture into EPSG:4326.

    Warping the reference scene for real — rather than re-rasterising it on an
    invented lon/lat grid — means this fixture differs from its source in CRS and
    nothing else, so a test can attribute any WARN to the reprojection alone.
    """
    with rasterio.open(source) as dataset:
        transform, width, height = calculate_default_transform(
            dataset.crs, CRS.from_string(WGS84), dataset.width, dataset.height, *dataset.bounds
        )
        profile = dataset.profile | {
            "crs": CRS.from_string(WGS84),
            "transform": transform,
            "width": width,
            "height": height,
        }
        with rasterio.open(destination, "w", **profile) as out:
            for index in range(1, dataset.count + 1):
                reproject(
                    source=rasterio.band(dataset, index),
                    destination=rasterio.band(out, index),
                    resampling=Resampling.bilinear,
                )
    return destination


def build_fixtures(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Scene]:
    """Generate every fixture into *output_dir* and return their ground truth."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base = _texture(SIZE, seed=20260906)
    changed = base.copy()
    changed[40:110, 150:220] = 0.95  # a new built-up block in the post image
    scenes: list[Scene] = []

    scenes.append(
        Scene(
            name="s2_pre",
            path=_write(
                output_dir / "s2_pre.tif",
                _s2_bands(base),
                _transform(S2_GSD),
                UTM43N,
                "uint16",
                tags={"TIFFTAG_DATETIME": "2019:04:12 05:26:41"},
            ),
            description="12-band Sentinel-2-like optical scene, EPSG:32643, 10 m.",
            truth={
                "crs": UTM43N,
                "gsd_m": S2_GSD,
                "band_count": 12,
                "modality": "optical",
                "sensor_guess": "Sentinel-2 (BigEarthNet-v2, 12-band)",
                "width": SIZE,
                "height": SIZE,
            },
        )
    )

    scenes.append(
        Scene(
            name="s2_post",
            path=_write(
                output_dir / "s2_post.tif",
                _s2_bands(changed),
                _transform(S2_GSD),
                UTM43N,
                "uint16",
                tags={"TIFFTAG_DATETIME": "2024:04:08 05:26:33"},
            ),
            description="Same grid as s2_pre, with a new built-up block. Perfectly aligned.",
            truth={"crs": UTM43N, "gsd_m": S2_GSD, "offset_px": 0.0, "iou": 1.0},
        )
    )

    scenes.append(
        Scene(
            name="s2_shift3",
            path=_write(
                output_dir / "s2_shift3.tif",
                _s2_bands(_shifted(base, INJECTED_SHIFT_PX, 0)),
                _transform(S2_GSD),
                UTM43N,
                "uint16",
                tags={"TIFFTAG_DATETIME": "2024:04:08 05:26:33"},
            ),
            description=(
                f"s2_pre content displaced by {INJECTED_SHIFT_PX} px in y, same geotransform."
            ),
            truth={"injected_shift_px": float(INJECTED_SHIFT_PX), "shift_axis": "y"},
        )
    )

    # Land squarely inside the WARN band (0.30 <= iou < 0.80), not on its edge:
    # for two equal squares, iou = i / (2a - i), so i/a = 2/3 gives iou = 0.50.
    overlap_fraction = 2.0 / 3.0
    offset_px = int(round(SIZE * (1.0 - overlap_fraction)))
    scenes.append(
        Scene(
            name="s2_partial_overlap",
            path=_write(
                output_dir / "s2_partial_overlap.tif",
                # Roll the content west by exactly the eastward origin shift, so the
                # overlapping ground carries identical pixels: the footprint is the only
                # thing wrong with this pair.
                _s2_bands(_shifted(base, 0, -offset_px)),
                _transform(S2_GSD, origin_x=ORIGIN_X + offset_px * S2_GSD),
                UTM43N,
                "uint16",
            ),
            description="Shifted east so the footprint IoU lands in the WARN band (0.30-0.80).",
            truth={"expected_iou_band": "WARN", "expected_iou": 0.5, "offset_px": offset_px},
        )
    )

    scenes.append(
        Scene(
            name="s2_disjoint",
            path=_write(
                output_dir / "s2_disjoint.tif",
                _s2_bands(base),
                _transform(S2_GSD, origin_x=ORIGIN_X + 50_000.0, origin_y=ORIGIN_Y + 50_000.0),
                UTM43N,
                "uint16",
            ),
            description="50 km away: no overlap at all, so bounds_overlap_iou must FAIL.",
            truth={"expected_iou": 0.0, "expected_status": "FAIL"},
        )
    )

    scenes.append(
        Scene(
            name="s2_wgs84",
            path=_warp_to_wgs84(
                source=output_dir / "s2_pre.tif",
                destination=output_dir / "s2_wgs84.tif",
            ),
            description="The same scene in EPSG:4326, to exercise automatic reprojection.",
            truth={"crs": WGS84, "expected_crs_match": "WARN", "expected_iou_min": 0.9},
        )
    )

    vv = (-12.0 + 6.0 * base).astype(np.float32)
    vh = (-19.0 + 6.0 * base).astype(np.float32)
    scenes.append(
        Scene(
            name="s1_vvvh",
            path=_write(
                output_dir / "s1_vvvh.tif",
                [vv, vh],
                _transform(S2_GSD),
                UTM43N,
                "float32",
                band_names=["VV", "VH"],
                tags={"TIFFTAG_DATETIME": "2024:04:09 00:41:12"},
            ),
            description="2-band VV/VH backscatter in dB, co-located with the S2 scene.",
            truth={"modality": "sar", "sensor_guess": "Sentinel-1 GRD", "band_count": 2},
        )
    )

    scenes.append(
        Scene(
            name="risat_single_pol",
            path=_write(
                output_dir / "risat_single_pol.tif",
                [(-14.0 + 7.0 * base).astype(np.float32)],
                _transform(CARTOSAT_GSD),
                UTM43N,
                "float32",
                tags={"TIFFTAG_DATETIME": "2024:04:10 06:02:00"},
            ),
            description="Single-band dB-like SAR at 0.65 m — the RISAT-1 fingerprint.",
            truth={"modality": "sar", "sensor_guess": "RISAT-1", "gsd_m": CARTOSAT_GSD},
        )
    )

    scenes.append(
        Scene(
            name="cartosat_mx",
            path=_write(
                output_dir / "cartosat_mx.tif",
                _s2_bands(base, count=4),
                _transform(CARTOSAT_GSD),
                UTM43N,
                "uint16",
            ),
            description="4-band multispectral at 0.65 m — the Cartosat-2S MX fingerprint.",
            truth={
                "modality": "optical",
                "sensor_guess": "Cartosat-2S MX",
                "gsd_m": CARTOSAT_GSD,
                "band_count": 4,
            },
        )
    )

    scenes.append(
        Scene(
            name="cartosat_pan",
            path=_write(
                output_dir / "cartosat_pan.tif",
                [np.clip(base * 4_000, 0, 10_000).astype(np.uint16)],
                _transform(CARTOSAT_GSD),
                UTM43N,
                "uint16",
            ),
            description="Single-band uint16 at 0.65 m — the Cartosat-2S PAN fingerprint.",
            truth={"modality": "panchromatic", "sensor_guess": "Cartosat-2S PAN"},
        )
    )

    # Nodata must be 0 in *every* band: a pixel only counts as empty when no band
    # carries a value there.
    half_nodata = _s2_bands(base)
    for band in half_nodata:
        band[:, : SIZE // 2] = 0
    scenes.append(
        Scene(
            name="s2_half_nodata",
            path=_write(
                output_dir / "s2_half_nodata.tif",
                half_nodata,
                _transform(S2_GSD),
                UTM43N,
                "uint16",
                nodata=0.0,
            ),
            description="Half the scene is declared nodata, so nodata_extent must FAIL (>40 %).",
            truth={"expected_nodata_pct": 50.0, "expected_status": "FAIL"},
        )
    )

    rgb = [np.clip(base * 255 * gain, 0, 255).astype(np.uint8) for gain in (1.0, 0.9, 0.8)]
    scenes.append(
        Scene(
            name="benchmark_rgb",
            path=_write(
                output_dir / "benchmark_rgb.png",
                rgb,
                Affine.identity(),
                None,
                "uint8",
                driver="PNG",
            ),
            description="Non-georeferenced 3-band PNG — the benchmark path.",
            truth={"is_georeferenced": False, "modality": "optical", "band_count": 3},
        )
    )

    return scenes


def main(argv: Sequence[str] | None = None) -> int:
    """Write every fixture and print a one-line summary of each."""
    args = list(argv if argv is not None else sys.argv[1:])
    output_dir = Path(args[0]) if args else DEFAULT_OUTPUT_DIR
    scenes = build_fixtures(output_dir)
    for scene in scenes:
        size_kb = scene.path.stat().st_size / 1024
        print(f"{scene.name:22s} {size_kb:8.1f} KiB  {scene.description}")
    print(f"\n{len(scenes)} fixtures written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
