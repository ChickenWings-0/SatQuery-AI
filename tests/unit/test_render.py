"""Unit tests for the spectral rendering and evidence artifact layer.

Two families of assertion carry most of the weight:

* **Analytic value tests** — a synthetic raster with known reflectances must
  produce the index value the arithmetic says it should, exactly.
* **Fixed-domain tests** — the same index value must render to the same colour in
  every scene. This is the core of the domain-adaptation strategy, so the tests
  also demonstrate what per-image normalisation *would* do, to keep the reason
  for the rule visible.

Golden images are computed analytically inside the tests rather than committed as
blobs: a reference you can derive is one a future reader can check, and a
committed PNG that was regenerated wrongly still passes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import rasterio
from affine import Affine
from fastapi.testclient import TestClient
from rasterio.crs import CRS
from skimage.metrics import structural_similarity

from satquery.api.app import app
from satquery.api.dependencies import get_artifact_store
from satquery.ingest.bands import resolve_bands
from satquery.ingest.modality import classify
from satquery.ingest.reader import open_raster
from satquery.render.artifact_store import (
    MIME_TYPES,
    ArtifactNotFoundError,
    ArtifactStore,
    TraceEvictedError,
    encode_geotiff,
    encode_image,
    encode_json,
)
from satquery.render.colormaps import (
    COLORMAPS,
    apply_colormap,
    get_colormap,
    normalise_fixed,
)
from satquery.render.composites import (
    false_colour_ir,
    index_view,
    panchromatic,
    percentile_stretch,
    sar_backscatter,
    sar_false_colour,
    stretch_rgb,
    true_colour,
)
from satquery.render.indices import (
    EPS,
    INDEX_DOMAIN,
    SARDB_DOMAIN,
    VV_DB_CLIP,
    index_statistics,
    nbr,
    ndbi,
    ndvi,
    ndwi,
    to_db,
    vv_vh_ratio_db,
)
from satquery.render.overlays import (
    CHANGE_ALPHA,
    CHANGE_COLOUR,
    change_overlay,
    draw_boxes,
    overlay_heatmap,
    overlay_mask,
)
from satquery.render.renderer import (
    RenderSource,
    render_views,
    unavailable_views,
)
from satquery.render.tiling import (
    VIEW_SIZE_PX,
    fit_dimensions,
    iter_windows,
    pad_to_square,
    read_downsampled,
    render_source,
    view_geometry,
)
from satquery.render.view_labels import (
    SCALE_NOTES,
    VIEW_NAMES,
    build_view_label,
    label_for_view,
    short_sensor_name,
)
from satquery.render.views import (
    CATALOGUE,
    MAX_VIEWS,
    ImageFormat,
    ImageViewContext,
    ViewId,
    select_views,
)
from satquery.schemas.enums import ArtifactType, ImageRole, Modality, PairType

UTM43N = "EPSG:32643"


def write_raster(
    path: Path,
    bands: list[npt.NDArray[Any]],
    gsd: float = 10.0,
    crs: str | None = UTM43N,
    dtype: str = "float32",
    nodata: float | None = None,
    names: list[str] | None = None,
) -> Path:
    """Write a small georeferenced raster for a test to read back."""
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "width": bands[0].shape[1],
        "height": bands[0].shape[0],
        "count": len(bands),
        "dtype": dtype,
    }
    if crs:
        profile["crs"] = CRS.from_string(crs)
        profile["transform"] = Affine(gsd, 0.0, 700_000.0, 0.0, -gsd, 3_160_000.0)
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dataset:
        for index, band in enumerate(bands, start=1):
            dataset.write(band.astype(dtype), index)
            if names:
                dataset.set_band_description(index, names[index - 1])
    return path


# =========================================================================== indices


def test_ndvi_matches_the_analytic_value() -> None:
    """(nir - red) / (nir + red), to the last float32 bit."""
    nir = np.array([[0.5, 0.8, 0.1, 0.25]], dtype=np.float32)
    red = np.array([[0.1, 0.2, 0.3, 0.25]], dtype=np.float32)
    expected = np.array([[0.4 / 0.6, 0.6 / 1.0, -0.2 / 0.4, 0.0]], dtype=np.float32)
    assert np.allclose(ndvi(nir, red), expected, atol=1e-6)


@pytest.mark.parametrize(
    ("function", "a", "b", "expected"),
    [
        (ndvi, 0.6, 0.2, 0.5),  # (0.6-0.2)/0.8
        (ndwi, 0.3, 0.1, 0.5),  # green, nir
        (ndbi, 0.4, 0.2, 1 / 3),  # swir1, nir
        (nbr, 0.5, 0.1, 2 / 3),  # nir, swir2
    ],
)
def test_every_index_matches_its_definition(
    function: Any, a: float, b: float, expected: float
) -> None:
    result = function(np.array([[a]], dtype=np.float32), np.array([[b]], dtype=np.float32))
    assert result[0, 0] == pytest.approx(expected, abs=1e-6)


def test_index_is_nan_where_the_denominator_vanishes() -> None:
    """Both bands zero means nodata or deep shadow — undefined, not 0.0.

    Returning 0.0 there would render as bare soil and pollute every statistic.
    """
    zeros = np.zeros((1, 3), dtype=np.float32)
    assert np.all(np.isnan(ndvi(zeros, zeros)))


def test_index_values_stay_inside_the_fixed_domain() -> None:
    rng = np.random.default_rng(0)
    nir = rng.uniform(0, 10_000, (64, 64)).astype(np.float32)
    red = rng.uniform(0, 10_000, (64, 64)).astype(np.float32)
    values = ndvi(nir, red)
    assert np.nanmin(values) >= INDEX_DOMAIN[0]
    assert np.nanmax(values) <= INDEX_DOMAIN[1]


def test_ndvi_is_recovered_from_a_real_raster(tmp_path: Path) -> None:
    """End to end: known reflectances written to disk, read back, index computed."""
    red = np.full((32, 32), 0.2, dtype=np.float32)
    nir = np.full((32, 32), 0.6, dtype=np.float32)
    path = write_raster(tmp_path / "known.tif", [red, nir])

    with rasterio.open(path) as dataset:
        bands, valid = read_downsampled(dataset, [2, 1], out_width=32, out_height=32)
    assert valid.all()
    assert np.allclose(ndvi(bands[0], bands[1]), 0.5, atol=1e-6)


def test_db_conversion_is_exact() -> None:
    linear = np.array([[1.0, 0.1, 0.01]], dtype=np.float32)
    assert np.allclose(to_db(linear, already_db=False), [[0.0, -10.0, -20.0]], atol=1e-4)


def test_db_conversion_floors_at_eps_rather_than_diverging() -> None:
    result = to_db(np.array([[0.0]], dtype=np.float32), already_db=False)
    assert result[0, 0] == pytest.approx(10.0 * np.log10(EPS), abs=1e-4)


def test_a_band_already_in_db_is_not_converted_twice() -> None:
    already = np.array([[-12.0, -18.0]], dtype=np.float32)
    assert np.allclose(to_db(already), already)


def test_vv_vh_ratio_is_a_difference_in_the_log_domain() -> None:
    vv = np.array([[-12.0]], dtype=np.float32)
    vh = np.array([[-18.0]], dtype=np.float32)
    assert vv_vh_ratio_db(vv, vh)[0, 0] == pytest.approx(6.0)


def test_statistics_exclude_nodata() -> None:
    values = np.array([[0.5, np.nan, 0.5, np.nan]], dtype=np.float32)
    stats = index_statistics(values, "ndvi")
    assert stats["ndvi_mean"] == pytest.approx(0.5)
    assert stats["ndvi_valid_pct"] == pytest.approx(50.0)
    assert index_statistics(np.full((2, 2), np.nan, dtype=np.float32), "ndvi") == {}


# ========================================================================= colormaps


def test_colormaps_are_256_entry_rgb_tables() -> None:
    for name, colormap in COLORMAPS.items():
        assert colormap.lut.shape == (256, 3), name
        assert colormap.lut.dtype == np.uint8


def test_colormap_endpoints_match_the_colorbrewer_anchors() -> None:
    """RdYlGn runs dark red to dark green; its reverse is the mirror image."""
    forward = get_colormap("RdYlGn").lut
    assert tuple(forward[0]) == (165, 0, 38)
    assert tuple(forward[255]) == (0, 104, 55)
    reverse = get_colormap("RdYlGn_r").lut
    assert tuple(reverse[0]) == (0, 104, 55)
    assert tuple(reverse[255]) == (165, 0, 38)


def test_gray_colormap_is_a_linear_ramp() -> None:
    lut = get_colormap("gray").lut
    assert tuple(lut[0]) == (0, 0, 0)
    assert tuple(lut[255]) == (255, 255, 255)
    assert np.allclose(lut[:, 0], lut[:, 1]) and np.allclose(lut[:, 1], lut[:, 2])


def test_unknown_colormap_is_rejected() -> None:
    with pytest.raises(KeyError):
        get_colormap("viridis")


def test_normalise_fixed_clips_rather_than_rescaling() -> None:
    values = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
    normalised = normalise_fixed(values, -1.0, 1.0)
    assert np.allclose(normalised, [[0.0, 0.0, 0.5, 1.0, 1.0]])


def test_nodata_renders_pure_black_in_a_colormapped_view() -> None:
    values = np.array([[0.0, np.nan]], dtype=np.float32)
    rgb = apply_colormap(normalise_fixed(values, -1.0, 1.0), "RdYlGn")
    assert tuple(rgb[0, 1]) == (0, 0, 0)
    assert tuple(rgb[0, 0]) != (0, 0, 0)


def test_invalid_mask_renders_black_even_for_finite_values() -> None:
    values = np.zeros((1, 2), dtype=np.float32)
    valid = np.array([[True, False]])
    rgb = apply_colormap(normalise_fixed(values, -1.0, 1.0), "RdYlGn", valid)
    assert tuple(rgb[0, 1]) == (0, 0, 0)


# ======================================================== the fixed-domain invariant


def _index_pair() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Two NDVI fields with genuinely different values but the same *shape*."""
    base = np.linspace(-0.4, 0.4, 64, dtype=np.float32).reshape(8, 8)
    return base, (base + 0.4).astype(np.float32)


def test_identical_index_values_render_identically_across_scenes() -> None:
    """The whole point: NDVI 0.3 is the same colour in every image, forever."""
    dull = np.full((8, 8), 0.3, dtype=np.float32)
    vivid = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    vivid[0, 0] = 0.3

    dull_rgb = index_view(dull, "RdYlGn", INDEX_DOMAIN)
    vivid_rgb = index_view(vivid, "RdYlGn", INDEX_DOMAIN)
    assert tuple(dull_rgb[0, 0]) == tuple(vivid_rgb[0, 0])


def test_different_index_values_render_differently() -> None:
    low, high = _index_pair()
    assert not np.array_equal(
        index_view(low, "RdYlGn", INDEX_DOMAIN), index_view(high, "RdYlGn", INDEX_DOMAIN)
    )


def test_per_image_stretch_would_erase_the_difference_a_fixed_domain_keeps() -> None:
    """Demonstrates the failure the fixed domain exists to prevent.

    Two scenes whose NDVI differs by a constant 0.4 are visibly different on the
    fixed domain. Per-image normalisation collapses them onto the same picture —
    the model would then have no way to learn that a colour means a value.
    """
    low, high = _index_pair()

    fixed_low = index_view(low, "RdYlGn", INDEX_DOMAIN)
    fixed_high = index_view(high, "RdYlGn", INDEX_DOMAIN)
    fixed_similarity = structural_similarity(fixed_low, fixed_high, channel_axis=2)

    def per_image(values: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        return index_view(values, "RdYlGn", (float(values.min()), float(values.max())))

    stretched_similarity = structural_similarity(per_image(low), per_image(high), channel_axis=2)

    assert stretched_similarity == pytest.approx(1.0, abs=1e-6)
    assert fixed_similarity < 0.9


def test_sar_backscatter_uses_the_fixed_db_domain() -> None:
    values = np.array([[-25.0, -12.5, 0.0, 5.0]], dtype=np.float32)
    rgb = sar_backscatter(values)
    assert tuple(rgb[0, 0]) == (0, 0, 0)  # domain floor
    assert tuple(rgb[0, 1]) == (128, 128, 128)  # midpoint
    assert tuple(rgb[0, 2]) == (255, 255, 255)  # domain ceiling
    assert tuple(rgb[0, 3]) == (255, 255, 255)  # above the ceiling, clipped
    assert SARDB_DOMAIN == (-25.0, 0.0)


def test_sar_false_colour_maps_each_channel_on_its_own_fixed_clip() -> None:
    vv = np.array([[VV_DB_CLIP[0], VV_DB_CLIP[1]]], dtype=np.float32)
    vh = np.array([[-30.0, -5.0]], dtype=np.float32)
    rgb = sar_false_colour(vv, vh)
    assert rgb[0, 0, 0] == 0 and rgb[0, 1, 0] == 255  # VV channel spans its clip
    assert rgb[0, 0, 1] == 0 and rgb[0, 1, 1] == 255  # VH channel spans its clip


# ======================================================================== composites


def test_percentile_stretch_maps_p2_and_p98_to_the_endpoints() -> None:
    band = np.linspace(0.0, 100.0, 101, dtype=np.float32).reshape(1, 101)
    stretched = percentile_stretch(band)
    assert stretched[0, 2] == 0
    assert stretched[0, 98] == 255
    assert stretched[0, 0] == 0 and stretched[0, 100] == 255  # tails clip


def test_percentile_stretch_is_invariant_under_a_positive_linear_transform() -> None:
    """A stretch is scale- and offset-free by construction — the golden property.

    Doubling the brightness of a scene must produce the *same* composite, because
    the percentiles move with it. This is the analytic reference the golden-image
    comparison below is checked against.
    """
    rng = np.random.default_rng(7)
    band = rng.uniform(100, 4_000, (64, 64)).astype(np.float32)
    original = percentile_stretch(band)
    transformed = percentile_stretch((band * 3.0 + 250.0).astype(np.float32))

    assert structural_similarity(original, transformed) == pytest.approx(1.0, abs=1e-3)
    assert np.abs(original.astype(int) - transformed.astype(int)).max() <= 1


def test_composite_matches_an_analytically_computed_golden_image() -> None:
    """Render a known ramp and compare against the stretch computed by hand."""
    rng = np.random.default_rng(11)
    bands = [rng.uniform(0, 5_000, (48, 48)).astype(np.float32) for _ in range(3)]

    expected_channels = []
    for band in bands:
        low, high = np.percentile(band, [2, 98])
        scaled = np.clip((band - low) / (high - low), 0.0, 1.0)
        expected_channels.append(np.round(scaled * 255.0).astype(np.uint8))
    golden = np.stack(expected_channels, axis=-1)

    rendered = true_colour(bands[0], bands[1], bands[2])
    assert np.array_equal(rendered, golden)
    assert structural_similarity(rendered, golden, channel_axis=2) == pytest.approx(1.0)


def test_flat_band_renders_mid_grey_not_black() -> None:
    """A constant band has no contrast; black would be indistinguishable from nodata."""
    flat = np.full((4, 4), 42.0, dtype=np.float32)
    assert np.all(percentile_stretch(flat) == 128)


def test_composites_render_nodata_as_pure_black() -> None:
    rng = np.random.default_rng(3)
    bands = [rng.uniform(0, 1_000, (8, 8)).astype(np.float32) for _ in range(3)]
    valid = np.ones((8, 8), dtype=bool)
    valid[0, 0] = False

    for rendered in (
        true_colour(*bands, valid),
        false_colour_ir(*bands, valid),
        panchromatic(bands[0], valid),
    ):
        assert tuple(rendered[0, 0]) == (0, 0, 0)


def test_panchromatic_replicates_one_band_across_three_channels() -> None:
    band = np.linspace(0, 1_000, 64, dtype=np.float32).reshape(8, 8)
    rgb = panchromatic(band)
    assert np.array_equal(rgb[..., 0], rgb[..., 1])
    assert np.array_equal(rgb[..., 1], rgb[..., 2])


def test_stretch_rgb_rejects_the_wrong_channel_count() -> None:
    band = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="exactly 3 channels"):
        stretch_rgb([band, band])


# ============================================================================ tiling


def test_fit_dimensions_preserves_aspect_ratio() -> None:
    assert fit_dimensions(1_000, 500, 448) == (448, 224)
    assert fit_dimensions(500, 1_000, 448) == (224, 448)
    assert fit_dimensions(100, 100, 448) == (448, 448)


def test_windows_cover_the_grid_exactly_once() -> None:
    covered = np.zeros((300, 500), dtype=int)
    for window in iter_windows(500, 300, tile=128):
        covered[
            int(window.row_off) : int(window.row_off + window.height),
            int(window.col_off) : int(window.col_off + window.width),
        ] += 1
    assert np.all(covered == 1)


def test_padding_is_invalid_rather_than_zero_valued() -> None:
    """A black bar must never be read as 'reflectance 0'."""
    bands = np.ones((2, 4, 8), dtype=np.float32)
    valid = np.ones((4, 8), dtype=bool)
    padded, padded_valid, pad = pad_to_square(bands, valid, 8)

    assert padded.shape == (2, 8, 8)
    assert pad.top == 2 and pad.bottom == 2
    assert not padded_valid[0, 0]
    assert np.isnan(padded[0, 0, 0])
    assert padded_valid[2:6, :].all()


def test_view_geometry_transform_accounts_for_scale_and_padding() -> None:
    source = Affine(10.0, 0.0, 700_000.0, 0.0, -10.0, 3_160_000.0)
    geometry = view_geometry(1_000, 500, source, UTM43N, size=448)

    assert (geometry.content_width, geometry.content_height) == (448, 224)
    assert geometry.pad.top == 112
    transform = geometry.transform
    assert transform is not None
    # 1000 source px across 448 view px at 10 m -> 22.32 m per view pixel.
    assert transform.a == pytest.approx(10_000 / 448)
    # The canvas origin sits 112 padded rows above the content origin.
    assert transform.c == pytest.approx(700_000.0)
    assert transform.f == pytest.approx(3_160_000.0 + 112 * (10_000 / 448))


def test_windowed_read_equals_a_whole_raster_read(tmp_path: Path) -> None:
    """Tiling is an optimisation, not a different answer."""
    rng = np.random.default_rng(5)
    band = rng.uniform(0, 10_000, (300, 200)).astype(np.float32)
    path = write_raster(tmp_path / "tiled.tif", [band])

    with rasterio.open(path) as dataset:
        tiled, _ = read_downsampled(dataset, [1], out_width=100, out_height=150, tile=32)
        whole = dataset.read(
            [1], out_shape=(1, 150, 100), resampling=rasterio.enums.Resampling.bilinear
        ).astype(np.float32)
    assert np.allclose(tiled, whole, atol=1e-3)


def test_large_raster_renders_without_loading_it_whole(tmp_path: Path) -> None:
    """Peak memory must track the tile size, not the file size (Master §8 Phase 2)."""
    import tracemalloc

    width = height = 2_048
    rng = np.random.default_rng(13)
    row = rng.integers(0, 10_000, (1, width), dtype=np.uint16)
    path = tmp_path / "big.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint16",
        crs=CRS.from_string(UTM43N),
        transform=Affine(10.0, 0.0, 700_000.0, 0.0, -10.0, 3_160_000.0),
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dataset:
        for band in (1, 2, 3):
            for y in range(height):
                dataset.write(row * band // 3, band, window=rasterio.windows.Window(0, y, width, 1))

    tracemalloc.start()
    bands, valid, geometry = render_source(path, [1, 2, 3], size=VIEW_SIZE_PX)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert bands.shape == (3, VIEW_SIZE_PX, VIEW_SIZE_PX)
    assert valid.any()
    assert geometry.crs == UTM43N
    # A whole-raster float32 read would be 3 * 2048^2 * 4 = 48 MiB.
    assert peak < 16 * 1024 * 1024, f"peak was {peak / 1e6:.1f} MB"


# ========================================================================== overlays


def test_change_overlay_uses_the_frozen_red_at_45_percent() -> None:
    base = np.full((4, 4, 3), 200, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True

    result = change_overlay(base, mask)
    expected_red = round(200 * (1 - CHANGE_ALPHA) + CHANGE_COLOUR[0] * CHANGE_ALPHA)
    assert result[1, 1, 0] == pytest.approx(expected_red, abs=1)
    assert result[1, 1, 1] == pytest.approx(round(200 * (1 - CHANGE_ALPHA)), abs=1)
    assert tuple(result[0, 0]) == (200, 200, 200)  # untouched outside the mask


def test_overlay_does_not_modify_its_base() -> None:
    base = np.full((4, 4, 3), 100, dtype=np.uint8)
    overlay_mask(base, np.ones((4, 4), dtype=bool))
    assert np.all(base == 100)


def test_overlay_rejects_a_mismatched_mask() -> None:
    base = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not match"):
        overlay_mask(base, np.ones((5, 5), dtype=bool))


def test_overlay_rejects_an_out_of_range_alpha() -> None:
    base = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="alpha"):
        overlay_mask(base, np.ones((2, 2), dtype=bool), alpha=1.5)


def test_heatmap_overlay_uses_a_fixed_domain() -> None:
    base = np.zeros((4, 4, 3), dtype=np.uint8)
    values = np.full((4, 4), 1.0, dtype=np.float32)
    result = overlay_heatmap(base, values, "RdYlGn", (-1.0, 1.0), alpha=1.0)
    assert tuple(result[0, 0]) == (0, 104, 55)  # the domain ceiling's colour


def test_boxes_are_clipped_to_the_canvas() -> None:
    base = np.zeros((8, 8, 3), dtype=np.uint8)
    drawn = draw_boxes(base, [(-4, -4, 3, 3), (100, 100, 120, 120)], width=1)
    assert drawn[0, 0, 0] == 255  # the clipped box still draws
    assert drawn[6, 6, 0] == 0  # the off-canvas box draws nothing


# ==================================================================== view catalogue


@pytest.mark.parametrize(
    ("view_id", "image_format"),
    [
        (ViewId.TC, ImageFormat.JPEG),
        (ViewId.FCIR, ImageFormat.JPEG),
        (ViewId.SWIR, ImageFormat.JPEG),
        (ViewId.SARFC, ImageFormat.JPEG),
        (ViewId.PAN, ImageFormat.JPEG),
        (ViewId.NDVI, ImageFormat.PNG),
        (ViewId.NDWI, ImageFormat.PNG),
        (ViewId.NDBI, ImageFormat.PNG),
        (ViewId.SARDB, ImageFormat.PNG),
        (ViewId.CHANGE, ImageFormat.PNG),
    ],
)
def test_format_rules_are_frozen(view_id: ViewId, image_format: ImageFormat) -> None:
    """Measurement views are PNG; JPEG chroma subsampling would shift their values."""
    assert CATALOGUE[view_id].image_format is image_format


def test_every_fixed_domain_view_is_png_and_declares_its_scale() -> None:
    for view_id, spec in CATALOGUE.items():
        if spec.is_fixed_domain:
            assert spec.image_format is ImageFormat.PNG, view_id
            assert spec.scale_note, view_id


def test_index_views_are_heatmaps_and_composites_are_rendered_views() -> None:
    assert CATALOGUE[ViewId.NDVI].artifact_type is ArtifactType.HEATMAP
    assert CATALOGUE[ViewId.TC].artifact_type is ArtifactType.RENDERED_VIEW
    assert CATALOGUE[ViewId.CHANGE].artifact_type is ArtifactType.OVERLAY_PNG


S2_BANDS = {"blue": 2, "green": 3, "red": 4, "nir": 8, "swir1": 11, "swir2": 12}
CARTOSAT_BANDS = {"blue": 1, "green": 2, "red": 3, "nir": 4}
RGB_BANDS = {"red": 1, "green": 2, "blue": 3}
S1_BANDS = {"vv": 1, "vh": 2}
RISAT_BANDS = {"vv": 1}
PAN_BANDS = {"pan": 1}


@pytest.mark.parametrize(
    ("pair_type", "contexts", "expected"),
    [
        (
            PairType.SINGLE,
            [(Modality.OPTICAL, S2_BANDS)],
            ["TC", "FCIR", "SWIR", "NDVI", "NDBI", "NDWI"],
        ),
        (PairType.SINGLE, [(Modality.OPTICAL, CARTOSAT_BANDS)], ["TC", "FCIR", "NDVI", "NDWI"]),
        (PairType.SINGLE, [(Modality.OPTICAL, RGB_BANDS)], ["TC"]),
        (PairType.SINGLE, [(Modality.SAR, S1_BANDS)], ["SARFC", "SARDB"]),
        (PairType.SINGLE, [(Modality.SAR, RISAT_BANDS)], ["SARDB"]),
        (PairType.SINGLE, [(Modality.PANCHROMATIC, PAN_BANDS)], ["PAN"]),
        (
            PairType.CROSS_MODAL,
            [(Modality.OPTICAL, S2_BANDS), (Modality.SAR, S1_BANDS)],
            ["TC", "FCIR", "NDVI", "NDBI", "SARFC", "SARDB"],
        ),
        (
            PairType.BI_TEMPORAL,
            [(Modality.OPTICAL, S2_BANDS), (Modality.OPTICAL, S2_BANDS)],
            ["TC", "TC", "FCIR", "FCIR", "NDBI", "NDBI"],
        ),
        (
            PairType.BI_TEMPORAL,
            [(Modality.SAR, S1_BANDS), (Modality.SAR, S1_BANDS)],
            ["SARFC", "SARFC", "SARDB", "SARDB"],
        ),
    ],
)
def test_view_selection_matches_the_frozen_policy(
    pair_type: PairType, contexts: list[tuple[Modality, dict[str, int]]], expected: list[str]
) -> None:
    """Every row of DATA_ADAPTATION_PLAN §2.4, in slot order."""
    requests = select_views(
        pair_type,
        [ImageViewContext(modality=m, resolved_bands=b) for m, b in contexts],
    )
    assert [request.view_id.value for request in requests] == expected


def test_selection_never_exceeds_the_view_cap() -> None:
    contexts = [ImageViewContext(Modality.OPTICAL, S2_BANDS)] * 2
    assert len(select_views(PairType.BI_TEMPORAL, contexts)) <= MAX_VIEWS


def test_cartosat_drops_swir_views_rather_than_substituting_a_band() -> None:
    """Unavailability is explicit — the gap the cross-modal SAR path exists to fill."""
    requests = select_views(PairType.SINGLE, [ImageViewContext(Modality.OPTICAL, CARTOSAT_BANDS)])
    produced = {request.view_id for request in requests}
    assert ViewId.SWIR not in produced and ViewId.NDBI not in produced


# ====================================================================== view labels


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "view_id": "TC",
                "slot": 1,
                "modality": Modality.OPTICAL,
                "sensor": "Sentinel-2 (BigEarthNet-v2, 12-band)",
            },
            "Image 1 (optical true colour, Sentinel-2)",
        ),
        (
            {"view_id": "NDVI", "slot": 4, "modality": Modality.OPTICAL},
            "Image 4 (optical NDVI heatmap, fixed scale -1 to +1)",
        ),
        (
            {
                "view_id": "SARFC",
                "slot": 5,
                "modality": Modality.SAR,
                "sensor": "Sentinel-1 GRD",
            },
            "Image 5 (SAR false colour VV/VH/ratio, Sentinel-1, fixed scale)",
        ),
    ],
)
def test_labels_reproduce_the_frozen_examples(kwargs: dict[str, Any], expected: str) -> None:
    """Verbatim from DATA_ADAPTATION_PLAN §2.5 — training and inference must match."""
    assert label_for_view(**kwargs) == expected


def test_temporal_labels_reproduce_the_frozen_examples() -> None:
    from datetime import UTC, datetime

    assert (
        build_view_label(
            slot=1,
            modality=Modality.OPTICAL,
            view_name="true colour",
            temporal_role="pre-change",
            date="2019-03-14",
        )
        == "Image 1 (optical true colour, pre-change, 2019-03-14)"
    )
    assert (
        label_for_view(
            view_id="TC",
            slot=2,
            modality=Modality.OPTICAL,
            role=ImageRole.POST,
            acquisition_time=datetime(2021, 7, 2, tzinfo=UTC),
        )
        == "Image 2 (optical true colour, post-change, 2021-07-02)"
    )


@pytest.mark.parametrize(
    ("sensor", "expected"),
    [
        ("Sentinel-2 (BigEarthNet-v2, 12-band)", "Sentinel-2"),
        ("Sentinel-2 L2A", "Sentinel-2"),
        ("Sentinel-1 GRD", "Sentinel-1"),
        ("Cartosat-2S MX", "Cartosat-2S"),
        ("RISAT-1", "RISAT-1"),
        ("Generic RGB", None),
        (None, None),
    ],
)
def test_sensor_names_are_shortened_to_the_platform(
    sensor: str | None, expected: str | None
) -> None:
    assert short_sensor_name(sensor) == expected


def test_every_catalogue_view_has_a_label_name() -> None:
    for view_id in CATALOGUE:
        assert view_id.value in VIEW_NAMES


def test_fixed_domain_views_carry_a_scale_note_in_their_label() -> None:
    for view_id, spec in CATALOGUE.items():
        if spec.is_fixed_domain:
            assert SCALE_NOTES[view_id.value] in label_for_view(view_id.value, 1, Modality.OPTICAL)


def test_unknown_view_is_rejected() -> None:
    with pytest.raises(KeyError):
        label_for_view("NDXX", 1, Modality.OPTICAL)


def test_slot_numbers_are_one_indexed() -> None:
    with pytest.raises(ValueError, match="1-indexed"):
        build_view_label(slot=0, modality=Modality.OPTICAL, view_name="true colour")


# ==================================================================== artifact store


def test_store_is_content_addressed(tmp_path: Path) -> None:
    """Identical pixels land on one blob, whichever trace rendered them."""
    store = ArtifactStore(tmp_path)
    rgb = np.full((16, 16, 3), 42, dtype=np.uint8)

    first = store.put_image("a" * 32, "art_0", rgb, ImageFormat.PNG)
    second = store.put_image("b" * 32, "art_9", rgb, ImageFormat.PNG)

    assert first.digest == second.digest
    assert first.path == second.path
    assert first.url != second.url


def test_store_round_trips_every_supported_format(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    trace = "c" * 32
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    png = store.put_image(trace, "art_0", rgb, ImageFormat.PNG)
    jpeg = store.put_image(trace, "art_1", rgb, ImageFormat.JPEG)
    geojson = store.put_geojson(trace, "art_2", {"type": "FeatureCollection", "features": []})
    inline = store.put_json(trace, "art_3", {"ndvi_mean": 0.42})

    assert png.mime == "image/png" and jpeg.mime == "image/jpeg"
    assert geojson.mime == "application/geo+json" and inline.mime == "application/json"
    assert store.list_artifacts(trace) == [
        "art_0.png",
        "art_1.jpg",
        "art_2.geojson",
        "art_3.json",
    ]
    assert store.resolve(trace, "art_0", "png").digest == png.digest


def test_stored_geotiff_reopens_with_the_right_crs_and_transform(tmp_path: Path) -> None:
    """Master §8 Phase 2's acceptance criterion for georeferenced companions."""
    store = ArtifactStore(tmp_path)
    transform = Affine(22.32, 0.0, 700_000.0, 0.0, -22.32, 3_162_500.0)
    values = np.linspace(-1, 1, 64, dtype=np.float32).reshape(8, 8)

    blob = store.put_geotiff("d" * 32, "art_0", values, transform, UTM43N, nodata=float("nan"))
    with rasterio.open(blob.path) as dataset:
        assert dataset.crs.to_string() == UTM43N
        assert dataset.transform.a == pytest.approx(22.32)
        assert dataset.transform.c == pytest.approx(700_000.0)
        assert np.allclose(dataset.read(1), values, atol=1e-6)


def test_json_encoding_is_deterministic() -> None:
    assert encode_json({"b": 1, "a": 2}) == encode_json({"a": 2, "b": 1})


def test_png_and_jpeg_encodings_are_distinguishable() -> None:
    rgb = np.random.default_rng(2).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    assert encode_image(rgb, ImageFormat.PNG).startswith(b"\x89PNG")
    assert encode_image(rgb, ImageFormat.JPEG).startswith(b"\xff\xd8\xff")


def test_png_round_trips_exactly_but_jpeg_is_lossy() -> None:
    """Why measurement views are PNG: JPEG moves the pixel values."""
    import io

    from PIL import Image

    rgb = np.random.default_rng(4).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    png = np.array(Image.open(io.BytesIO(encode_image(rgb, ImageFormat.PNG))))
    jpeg = np.array(Image.open(io.BytesIO(encode_image(rgb, ImageFormat.JPEG))))

    assert np.array_equal(png, rgb)
    assert not np.array_equal(jpeg, rgb)


def test_encode_image_rejects_a_non_rgb_array() -> None:
    with pytest.raises(ValueError, match="RGB"):
        encode_image(np.zeros((8, 8), dtype=np.uint8), ImageFormat.PNG)


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="unsupported"):
        store.put("e" * 32, "art_0", "bmp", b"")


def test_unknown_trace_and_artifact_are_distinguished_from_eviction(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    trace = "f" * 32
    store.put_image(trace, "art_0", np.zeros((4, 4, 3), dtype=np.uint8), ImageFormat.PNG)

    with pytest.raises(ArtifactNotFoundError):
        store.resolve("0" * 32, "art_0", "png")
    with pytest.raises(ArtifactNotFoundError):
        store.resolve(trace, "art_7", "png")

    store.evict(trace)
    with pytest.raises(TraceEvictedError):
        store.resolve(trace, "art_0", "png")
    assert store.list_artifacts(trace) == []


def test_geotiff_without_georeferencing_still_encodes() -> None:
    """The benchmark PNG path has no CRS; the companion must not crash."""
    data = encode_geotiff(np.zeros((4, 4), dtype=np.uint8), None, None)
    assert data.startswith(b"II") or data.startswith(b"MM")


# ========================================================================== renderer


def _render_source(path: Path, image_id: str, role: ImageRole | None = None) -> RenderSource:
    """Build a renderer input from a fixture, going through the real ingest path."""
    info = open_raster(path)
    verdict = classify(info)
    return RenderSource(
        image_id=image_id,
        path=info.path,
        modality=verdict.modality,
        sensor=verdict.sensor_guess,
        resolved_bands=resolve_bands(verdict.sensor_guess, info.band_count, info.band_names),
        role=role,
        acquisition_time=info.acquisition_time,
    )


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    """A fresh artifact store per test."""
    return ArtifactStore(tmp_path / "artifacts")


def test_renderer_produces_the_bitemporal_view_set(
    scene_paths: dict[str, Path], store: ArtifactStore
) -> None:
    trace = "1" * 32
    views = render_views(
        [
            _render_source(scene_paths["s2_pre"], "img_0", ImageRole.PRE),
            _render_source(scene_paths["s2_post"], "img_1", ImageRole.POST),
        ],
        PairType.BI_TEMPORAL,
        trace,
        store,
    )

    assert [view.view_id.value for view in views] == ["TC", "TC", "FCIR", "FCIR", "NDBI", "NDBI"]
    assert [view.slot for view in views] == [1, 2, 3, 4, 5, 6]
    assert [view.artifact.id for view in views] == [f"art_{i}" for i in range(6)]
    assert all(view.rgb.shape == (VIEW_SIZE_PX, VIEW_SIZE_PX, 3) for view in views)


def test_rendered_artifacts_follow_the_format_rules(
    scene_paths: dict[str, Path], store: ArtifactStore
) -> None:
    views = render_views(
        [_render_source(scene_paths["s2_pre"], "img_0")], PairType.SINGLE, "2" * 32, store
    )
    by_view = {view.view_id: view.artifact for view in views}

    assert by_view[ViewId.TC].mime == "image/jpeg"
    assert by_view[ViewId.TC].url.endswith(".jpg")
    assert by_view[ViewId.NDVI].mime == "image/png"
    assert by_view[ViewId.NDVI].url.endswith(".png")
    # Measurement views carry a georeferenced companion (API_CONTRACT §2.5).
    assert by_view[ViewId.NDVI].geotiff_url is not None
    assert by_view[ViewId.TC].geotiff_url is None


def test_rendered_index_artifacts_carry_statistics_and_geo(
    scene_paths: dict[str, Path], store: ArtifactStore
) -> None:
    views = render_views(
        [_render_source(scene_paths["s2_pre"], "img_0")], PairType.SINGLE, "3" * 32, store
    )
    ndvi_view = next(view for view in views if view.view_id is ViewId.NDVI)

    assert "ndvi_mean" in ndvi_view.scalars
    assert -1.0 <= ndvi_view.scalars["ndvi_mean"] <= 1.0
    assert ndvi_view.artifact.geo is not None
    assert ndvi_view.artifact.geo.crs == UTM43N
    assert ndvi_view.artifact.stats is not None
    assert ndvi_view.artifact.stats["fixed_domain"] == [-1.0, 1.0]


def test_renderer_renumbers_slots_when_a_view_is_unavailable(
    scene_paths: dict[str, Path], store: ArtifactStore
) -> None:
    """Cartosat has no SWIR: four views, contiguously numbered 1-4."""
    views = render_views(
        [_render_source(scene_paths["cartosat_mx"], "img_0")], PairType.SINGLE, "4" * 32, store
    )
    assert [view.view_id.value for view in views] == ["TC", "FCIR", "NDVI", "NDWI"]
    assert [view.slot for view in views] == [1, 2, 3, 4]
    assert [f"Image {i}" in view.label for i, view in enumerate(views, start=1)] == [True] * 4


def test_unavailable_views_are_reported_with_the_missing_band(
    scene_paths: dict[str, Path],
) -> None:
    report = unavailable_views(
        [_render_source(scene_paths["cartosat_mx"], "img_0")], PairType.SINGLE
    )
    gaps = " ".join(report["img_0"])
    assert "NDBI: missing swir1" in gaps
    assert "SWIR: missing swir2" in gaps


def test_sar_scene_renders_sar_views(scene_paths: dict[str, Path], store: ArtifactStore) -> None:
    views = render_views(
        [_render_source(scene_paths["s1_vvvh"], "img_0")], PairType.SINGLE, "5" * 32, store
    )
    assert [view.view_id.value for view in views] == ["SARFC", "SARDB"]
    assert "Sentinel-1" in views[0].label


def test_non_georeferenced_input_renders_without_a_geotiff(
    scene_paths: dict[str, Path], store: ArtifactStore
) -> None:
    views = render_views(
        [_render_source(scene_paths["benchmark_rgb"], "img_0")], PairType.SINGLE, "6" * 32, store
    )
    assert [view.view_id.value for view in views] == ["TC"]
    assert views[0].artifact.geo is None
    assert views[0].artifact.geotiff_url is None


# ================================================================= artifact endpoint


@pytest.fixture
def artifact_client(store: ArtifactStore) -> Any:
    """A TestClient whose artifact store is the test's temporary one."""
    app.dependency_overrides[get_artifact_store] = lambda: store
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.pop(get_artifact_store, None)


def test_endpoint_serves_a_rendered_png_with_immutable_caching(
    artifact_client: TestClient, store: ArtifactStore, scene_paths: dict[str, Path]
) -> None:
    trace = "7" * 32
    views = render_views(
        [_render_source(scene_paths["s2_pre"], "img_0")], PairType.SINGLE, trace, store
    )
    ndvi_view = next(view for view in views if view.view_id is ViewId.NDVI)

    response = artifact_client.get(ndvi_view.artifact.url)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert response.headers["etag"]
    assert response.content.startswith(b"\x89PNG")

    import io

    from PIL import Image

    decoded = np.array(Image.open(io.BytesIO(response.content)))
    assert decoded.shape == (VIEW_SIZE_PX, VIEW_SIZE_PX, 3)
    assert np.array_equal(decoded, ndvi_view.rgb)


def test_endpoint_serves_a_jpeg_composite(
    artifact_client: TestClient, store: ArtifactStore, scene_paths: dict[str, Path]
) -> None:
    trace = "8" * 32
    views = render_views(
        [_render_source(scene_paths["s2_pre"], "img_0")], PairType.SINGLE, trace, store
    )
    composite = next(view for view in views if view.view_id is ViewId.TC)

    response = artifact_client.get(composite.artifact.url)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


def test_endpoint_honours_conditional_requests(
    artifact_client: TestClient, store: ArtifactStore
) -> None:
    trace = "9" * 32
    blob = store.put_image(trace, "art_0", np.zeros((8, 8, 3), dtype=np.uint8), ImageFormat.PNG)

    first = artifact_client.get(blob.url)
    again = artifact_client.get(blob.url, headers={"If-None-Match": first.headers["etag"]})
    assert again.status_code == 304
    assert again.headers["cache-control"] == "public, max-age=31536000, immutable"


def test_endpoint_returns_404_for_an_unknown_artifact(artifact_client: TestClient) -> None:
    response = artifact_client.get(f"/v1/artifacts/{'a' * 32}/art_0.png")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ARTIFACT_NOT_FOUND"


def test_endpoint_returns_410_for_an_evicted_trace(
    artifact_client: TestClient, store: ArtifactStore
) -> None:
    trace = "e" * 32
    blob = store.put_image(trace, "art_0", np.zeros((4, 4, 3), dtype=np.uint8), ImageFormat.PNG)
    store.evict(trace)

    response = artifact_client.get(blob.url)
    assert response.status_code == 410
    assert response.json()["error"]["code"] == "ARTIFACT_EVICTED"


def test_endpoint_rejects_an_unserveable_extension(artifact_client: TestClient) -> None:
    response = artifact_client.get(f"/v1/artifacts/{'a' * 32}/art_0.exe")
    assert response.status_code == 404
    assert "exe" not in MIME_TYPES
