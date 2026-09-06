"""Unit tests for the geospatial ingestion and compatibility engine.

Every assertion is anchored to a fixture whose geometry is known by construction
(see ``scripts/make_synthetic_fixtures.py``) — an injected shift of exactly 3 px,
a footprint offset computed to land in a named threshold band, a declared CRS.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from make_synthetic_fixtures import INJECTED_SHIFT_PX, Scene
from satquery.ingest.bands import missing_bands, resolve_bands
from satquery.ingest.capabilities import supported_tasks
from satquery.ingest.compatibility import assess, derive_overall, detect_pair_type
from satquery.ingest.coregistration import (
    bounds_iou,
    common_grid,
    estimate_offset,
    gsd_ratio,
    intersection_bounds,
    resample_to_grid,
)
from satquery.ingest.errors import (
    InvalidRasterError,
    MissingGeoreferenceError,
    TooManyImagesError,
    UnsupportedFormatError,
)
from satquery.ingest.manifest_builder import assign_roles, build_manifest, sanitise_filename
from satquery.ingest.modality import classify
from satquery.ingest.pipeline import SourceImage, ingest
from satquery.ingest.reader import RasterInfo, open_raster
from satquery.schemas.compatibility import CheckResult
from satquery.schemas.enums import (
    CheckName,
    CheckStatus,
    ImageRole,
    Modality,
    Overall,
    PairType,
    TaskType,
)
from satquery.schemas.manifest import InputManifest

# --------------------------------------------------------------------------- helpers


@pytest.fixture(scope="module")
def read(scenes_module: dict[str, Scene]) -> dict[str, RasterInfo]:
    """Parse every fixture once per module — rasterio I/O is the slow part here."""
    return {name: open_raster(scene.path) for name, scene in scenes_module.items()}


@pytest.fixture(scope="module")
def scenes_module(scenes: dict[str, Scene]) -> dict[str, Scene]:
    """Module-scoped alias of the session fixture, so `read` can depend on it."""
    return scenes


def report_for(read: dict[str, RasterInfo], *names: str, hint: PairType | None = None):
    """Run the full check battery over the named fixtures."""
    infos = [read[name] for name in names]
    return assess(infos, [classify(info) for info in infos], pair_type_hint=hint)


def check_of(checks: list[CheckResult], name: CheckName) -> CheckResult:
    """Pull one named check out of a report."""
    return next(check for check in checks if check.name is name)


# ---------------------------------------------------------------------------- reader


def test_reader_extracts_declared_geometry(read: dict[str, RasterInfo]) -> None:
    info = read["s2_pre"]
    assert info.crs == "EPSG:32643"
    assert info.is_georeferenced is True
    assert info.gsd_m == pytest.approx(10.0)
    assert info.width == 256 and info.height == 256
    assert info.band_count == 12
    assert info.dtype == "uint16"
    assert info.transform_list is not None
    assert len(info.transform_list) == 6


def test_reader_reports_bounds_in_both_crs(read: dict[str, RasterInfo]) -> None:
    info = read["s2_pre"]
    assert info.bounds_native is not None and info.bounds_wgs84 is not None

    minx, miny, maxx, maxy = info.bounds_native
    assert maxx - minx == pytest.approx(256 * 10.0)
    assert maxy - miny == pytest.approx(256 * 10.0)

    min_lon, min_lat, max_lon, max_lat = info.bounds_wgs84
    assert min_lon < max_lon and min_lat < max_lat
    # Lon first, and plausibly over the Delhi NCR footprint the fixture describes.
    assert 76.0 < min_lon < 78.0
    assert 27.0 < min_lat < 30.0


def test_reader_parses_acquisition_time(read: dict[str, RasterInfo]) -> None:
    assert read["s2_pre"].acquisition_time == datetime(2019, 4, 12, 5, 26, 41, tzinfo=UTC)
    assert read["s2_post"].acquisition_time == datetime(2024, 4, 8, 5, 26, 33, tzinfo=UTC)
    assert read["s2_partial_overlap"].acquisition_time is None


def test_reader_computes_nodata_percentage(read: dict[str, RasterInfo]) -> None:
    """The fixture blanks exactly half the scene and declares 0 as nodata."""
    assert read["s2_half_nodata"].nodata_pct == pytest.approx(50.0, abs=0.5)
    assert read["s2_pre"].nodata_pct == pytest.approx(0.0)


def test_reader_handles_a_non_georeferenced_png(read: dict[str, RasterInfo]) -> None:
    info = read["benchmark_rgb"]
    assert info.driver == "PNG"
    assert info.is_georeferenced is False
    assert info.crs is None
    assert info.transform_list is None
    assert info.bounds_wgs84 is None
    assert info.gsd_m is None
    assert any("not georeferenced" in warning for warning in info.warnings)


def test_reader_reads_band_names(read: dict[str, RasterInfo]) -> None:
    assert read["s1_vvvh"].band_names == ["VV", "VH"]
    assert read["s2_pre"].band_names is None


def test_reader_rejects_a_file_that_is_not_a_raster(tmp_path: Path) -> None:
    junk = tmp_path / "notes.tif"
    junk.write_bytes(b"this is definitely not a GeoTIFF")
    with pytest.raises(InvalidRasterError) as excinfo:
        open_raster(junk, ref="img_0")
    assert excinfo.value.http_status == 400
    assert excinfo.value.code == "INVALID_RASTER"
    assert str(tmp_path) not in excinfo.value.message  # never leak paths


def test_reader_rejects_an_unsupported_driver(tmp_path: Path) -> None:
    """A readable raster in a format outside the allow-list is a 415, not a 400."""
    import rasterio
    from affine import Affine

    path = tmp_path / "scene.gif"
    with rasterio.open(
        path,
        "w",
        driver="GIF",
        width=8,
        height=8,
        count=1,
        dtype="uint8",
        transform=Affine.identity(),
    ) as dataset:
        dataset.write(np.zeros((8, 8), dtype=np.uint8), 1)

    with pytest.raises(UnsupportedFormatError) as excinfo:
        open_raster(path)
    assert excinfo.value.http_status == 415


def test_structure_band_is_finite_and_two_dimensional(read: dict[str, RasterInfo]) -> None:
    band = read["s2_pre"].structure_band()
    assert band.ndim == 2
    assert np.isfinite(band).all()


# -------------------------------------------------------------------------- modality


@pytest.mark.parametrize(
    ("scene", "modality", "sensor"),
    [
        ("s2_pre", Modality.OPTICAL, "Sentinel-2 (BigEarthNet-v2, 12-band)"),
        ("s1_vvvh", Modality.SAR, "Sentinel-1 GRD"),
        ("cartosat_mx", Modality.OPTICAL, "Cartosat-2S MX"),
        ("cartosat_pan", Modality.PANCHROMATIC, "Cartosat-2S PAN"),
        ("risat_single_pol", Modality.SAR, "RISAT-1"),
        ("benchmark_rgb", Modality.OPTICAL, "Generic RGB"),
    ],
)
def test_modality_and_sensor_fingerprints(
    read: dict[str, RasterInfo], scene: str, modality: Modality, sensor: str
) -> None:
    verdict = classify(read[scene])
    assert verdict.modality is modality
    assert verdict.sensor_guess == sensor
    assert 0.0 <= verdict.confidence <= 1.0
    assert verdict.reasons


def test_sar_is_distinguished_by_its_value_distribution(read: dict[str, RasterInfo]) -> None:
    """RISAT-1 and Cartosat PAN are both single-band; only the histogram separates them."""
    risat, pan = read["risat_single_pol"], read["cartosat_pan"]
    assert risat.band_count == pan.band_count == 1
    assert classify(risat).modality is Modality.SAR
    assert classify(pan).modality is Modality.PANCHROMATIC


# ------------------------------------------------------------------ manifest builder


def test_manifest_is_built_from_the_raster(read: dict[str, RasterInfo]) -> None:
    manifest = build_manifest(read["s2_pre"], index=0)
    assert isinstance(manifest, InputManifest)
    assert manifest.id == "img_0"
    assert manifest.modality is Modality.OPTICAL
    assert manifest.sensor_guess == "Sentinel-2 (BigEarthNet-v2, 12-band)"
    assert manifest.crs == "EPSG:32643"
    assert manifest.transform is not None and len(manifest.transform) == 6
    assert manifest.bounds_wgs84 is not None and len(manifest.bounds_wgs84) == 4
    assert len(manifest.sha256) == 64
    assert manifest.size_bytes > 0
    assert manifest.is_georeferenced is True
    # Round-trips through the frozen schema unchanged.
    assert InputManifest.model_validate(manifest.model_dump()) == manifest


def test_manifest_ids_follow_upload_order(read: dict[str, RasterInfo]) -> None:
    manifests = [build_manifest(read["s2_pre"], 0), build_manifest(read["s2_post"], 1)]
    assert [m.id for m in manifests] == ["img_0", "img_1"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("../../etc/passwd", "passwd"), ("C:\\temp\\a.tif", "a.tif"), ("   ", "unnamed")],
)
def test_filenames_are_sanitised(raw: str, expected: str) -> None:
    assert sanitise_filename(raw) == expected


def test_roles_follow_acquisition_time_not_upload_order(read: dict[str, RasterInfo]) -> None:
    """The later image is `post` even when it was uploaded first."""
    manifests = [build_manifest(read["s2_post"], 0), build_manifest(read["s2_pre"], 1)]
    assign_roles(manifests, PairType.BI_TEMPORAL)
    by_id = {m.id: m for m in manifests}
    assert by_id["img_1"].role is ImageRole.PRE  # 2019
    assert by_id["img_0"].role is ImageRole.POST  # 2024


def test_cross_modal_roles_follow_modality(read: dict[str, RasterInfo]) -> None:
    manifests = [build_manifest(read["s2_pre"], 0), build_manifest(read["s1_vvvh"], 1)]
    assign_roles(manifests, PairType.CROSS_MODAL)
    assert [m.role for m in manifests] == [ImageRole.OPTICAL, ImageRole.SAR]


def test_explicit_role_overrides_win(read: dict[str, RasterInfo]) -> None:
    manifests = [build_manifest(read["s2_pre"], 0), build_manifest(read["s2_post"], 1)]
    assign_roles(
        manifests,
        PairType.BI_TEMPORAL,
        overrides={manifests[0].filename: "post", manifests[1].filename: "pre"},
    )
    assert manifests[0].role is ImageRole.POST
    assert manifests[1].role is ImageRole.PRE


# --------------------------------------------------------------------- geometry maths


def test_bounds_iou_is_exact_for_known_boxes() -> None:
    assert bounds_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert bounds_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # Half-overlapping equal squares: i = 50, union = 150.
    assert bounds_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_intersection_bounds_returns_none_when_disjoint() -> None:
    assert intersection_bounds((0, 0, 1, 1), (2, 2, 3, 3)) is None
    assert intersection_bounds((0, 0, 10, 10), (5, 5, 15, 15)) == (5, 5, 10, 10)


def test_gsd_ratio_is_always_at_least_one() -> None:
    assert gsd_ratio(10.0, 0.65) == pytest.approx(10.0 / 0.65)
    assert gsd_ratio(0.65, 10.0) == pytest.approx(10.0 / 0.65)
    assert gsd_ratio(10.0, None) is None


def test_common_grid_uses_the_coarser_resolution(read: dict[str, RasterInfo]) -> None:
    """Neither image is upsampled past its real resolution."""
    grid = common_grid(read["cartosat_mx"], read["s2_pre"])
    assert grid is not None
    assert grid.gsd_m == pytest.approx(10.0)
    assert grid.crs == "EPSG:32643"


def test_common_grid_is_none_for_disjoint_footprints(read: dict[str, RasterInfo]) -> None:
    assert common_grid(read["s2_pre"], read["s2_disjoint"]) is None


def test_common_grid_is_none_without_georeferencing(read: dict[str, RasterInfo]) -> None:
    assert common_grid(read["s2_pre"], read["benchmark_rgb"]) is None


def test_resample_to_grid_lands_on_the_requested_shape(read: dict[str, RasterInfo]) -> None:
    grid = common_grid(read["s2_pre"], read["s2_post"])
    assert grid is not None
    array = resample_to_grid(read["s2_pre"], grid)
    assert array.shape == (grid.height, grid.width)
    assert np.isfinite(array).any()


# ------------------------------------------------------------------- co-registration


def test_injected_three_pixel_shift_is_recovered(read: dict[str, RasterInfo]) -> None:
    """The plan's acceptance criterion: a 3 px injected shift, recovered to +/-0.5 px."""
    offset = estimate_offset(read["s2_pre"], read["s2_shift3"])
    assert offset is not None and offset.reliable
    assert offset.magnitude_px == pytest.approx(INJECTED_SHIFT_PX, abs=0.5)
    # The shift was injected along y only.
    assert abs(offset.dy_px) == pytest.approx(INJECTED_SHIFT_PX, abs=0.5)
    assert offset.dx_px == pytest.approx(0.0, abs=0.5)


def test_aligned_images_report_no_shift(read: dict[str, RasterInfo]) -> None:
    offset = estimate_offset(read["s2_pre"], read["s2_post"])
    assert offset is not None
    assert offset.magnitude_px == pytest.approx(0.0, abs=0.5)


def test_reprojected_image_still_registers(read: dict[str, RasterInfo]) -> None:
    """A CRS difference must not masquerade as a registration error."""
    offset = estimate_offset(read["s2_pre"], read["s2_wgs84"])
    assert offset is not None
    assert offset.magnitude_px == pytest.approx(0.0, abs=0.5)


def test_offset_is_none_without_a_common_grid(read: dict[str, RasterInfo]) -> None:
    assert estimate_offset(read["s2_pre"], read["s2_disjoint"]) is None


# ------------------------------------------------------------------------ pair typing


def test_pair_type_detection(read: dict[str, RasterInfo]) -> None:
    optical, sar = Modality.OPTICAL, Modality.SAR
    assert detect_pair_type([read["s2_pre"]], [optical]) is PairType.SINGLE
    assert (
        detect_pair_type([read["s2_pre"], read["s2_post"]], [optical, optical])
        is PairType.BI_TEMPORAL
    )
    assert (
        detect_pair_type([read["s2_pre"], read["s1_vvvh"]], [optical, sar]) is PairType.CROSS_MODAL
    )


def test_pair_type_hint_is_recorded_as_user_declared(read: dict[str, RasterInfo]) -> None:
    """A declared pair type overrides detection and changes which checks apply."""
    report = report_for(read, "s2_pre", "s1_vvvh", hint=PairType.BI_TEMPORAL)
    assert report.pair_type is PairType.BI_TEMPORAL  # detection alone would say CROSS_MODAL
    assert report.pair_type_source.value == "user_declared"
    assert check_of(report.checks, CheckName.MODALITY_DISTINCT).status is CheckStatus.SKIP
    assert check_of(report.checks, CheckName.TEMPORAL_ORDERING).status is not CheckStatus.SKIP


def test_a_failing_pair_is_marked_incompatible_even_when_declared(
    read: dict[str, RasterInfo],
) -> None:
    """A user-declared type cannot make an unusable pair usable."""
    report = report_for(read, "s2_pre", "s2_disjoint", hint=PairType.BI_TEMPORAL)
    assert report.overall is Overall.FAIL
    assert report.pair_type is PairType.INCOMPATIBLE


# --------------------------------------------------------------- compatibility rules


def test_derive_overall_follows_the_frozen_rule() -> None:
    def result(status: CheckStatus) -> CheckResult:
        return CheckResult(name=CheckName.GSD_RATIO, status=status, detail="synthetic")

    assert derive_overall([result(CheckStatus.PASS), result(CheckStatus.SKIP)]) is Overall.PASS
    assert (
        derive_overall([result(CheckStatus.PASS), result(CheckStatus.WARN)])
        is Overall.PASS_WITH_WARNINGS
    )
    assert derive_overall([result(CheckStatus.WARN), result(CheckStatus.FAIL)]) is Overall.FAIL


def test_clean_bitemporal_pair_passes_every_check(read: dict[str, RasterInfo]) -> None:
    report = report_for(read, "s2_pre", "s2_post")
    assert report.pair_type is PairType.BI_TEMPORAL
    assert report.overall is Overall.PASS
    assert {check.name for check in report.checks} == set(CheckName)
    assert check_of(report.checks, CheckName.TEMPORAL_ORDERING).status is CheckStatus.PASS
    assert report.common_grid is not None
    assert report.common_grid.gsd_m == pytest.approx(10.0)


def test_every_check_reports_a_displayable_detail(read: dict[str, RasterInfo]) -> None:
    for check in report_for(read, "s2_pre", "s2_post").checks:
        assert check.detail and check.detail[0].isupper() and check.detail.endswith(".")


def test_partial_overlap_warns_and_disjoint_fails(read: dict[str, RasterInfo]) -> None:
    """The IoU bands are 0.80 / 0.30, frozen by API_CONTRACT §3.2."""
    warned = check_of(
        report_for(read, "s2_pre", "s2_partial_overlap").checks, CheckName.BOUNDS_OVERLAP_IOU
    )
    assert warned.status is CheckStatus.WARN
    assert isinstance(warned.value, float) and 0.30 <= warned.value < 0.80

    failed_report = report_for(read, "s2_pre", "s2_disjoint")
    failed = check_of(failed_report.checks, CheckName.BOUNDS_OVERLAP_IOU)
    assert failed.status is CheckStatus.FAIL
    assert failed.value == 0.0
    assert failed_report.overall is Overall.FAIL
    assert failed_report.pair_type is PairType.INCOMPATIBLE
    assert failed_report.common_grid is None


def test_full_overlap_passes(read: dict[str, RasterInfo]) -> None:
    check = check_of(report_for(read, "s2_pre", "s2_post").checks, CheckName.BOUNDS_OVERLAP_IOU)
    assert check.status is CheckStatus.PASS
    assert check.value == pytest.approx(1.0)


def test_mismatched_crs_is_reprojected_and_the_action_is_reported(
    read: dict[str, RasterInfo],
) -> None:
    report = report_for(read, "s2_pre", "s2_wgs84")
    crs_check = check_of(report.checks, CheckName.CRS_MATCH)
    assert crs_check.status is CheckStatus.WARN
    assert "EPSG:4326" in str(crs_check.value)
    assert any("reprojected" in action for action in report.actions_taken)
    # Reprojection succeeded, so the pair is still usable.
    assert report.overall is Overall.PASS_WITH_WARNINGS
    assert report.common_grid is not None
    assert report.common_grid.crs == "EPSG:32643"


def test_three_pixel_shift_is_reported_as_a_coregistration_warning(
    read: dict[str, RasterInfo],
) -> None:
    check = check_of(
        report_for(read, "s2_pre", "s2_shift3").checks, CheckName.COREGISTRATION_OFFSET_PX
    )
    assert check.status is CheckStatus.WARN  # 1.0 < 3 px <= 5.0
    assert isinstance(check.value, float)
    assert check.value == pytest.approx(INJECTED_SHIFT_PX, abs=0.5)


def test_gsd_mismatch_beyond_four_times_fails(read: dict[str, RasterInfo]) -> None:
    """10 m against 0.65 m is a 15x ratio — past the frozen 4x limit."""
    report = report_for(read, "s2_pre", "cartosat_mx")
    check = check_of(report.checks, CheckName.GSD_RATIO)
    assert check.status is CheckStatus.FAIL
    assert isinstance(check.value, float) and check.value > 4.0
    assert report.overall is Overall.FAIL


def test_excessive_nodata_fails(read: dict[str, RasterInfo]) -> None:
    check = check_of(report_for(read, "s2_pre", "s2_half_nodata").checks, CheckName.NODATA_EXTENT)
    assert check.status is CheckStatus.FAIL  # 50 % > 40 %
    assert isinstance(check.value, float) and check.value == pytest.approx(50.0, abs=0.5)


def test_cross_modal_pair_is_detected_and_modalities_are_distinct(
    read: dict[str, RasterInfo],
) -> None:
    report = report_for(read, "s2_pre", "s1_vvvh")
    assert report.pair_type is PairType.CROSS_MODAL
    assert report.overall is Overall.PASS
    assert check_of(report.checks, CheckName.MODALITY_DISTINCT).status is CheckStatus.PASS
    assert check_of(report.checks, CheckName.TEMPORAL_ORDERING).status is CheckStatus.SKIP


def test_identical_modalities_are_not_a_cross_modal_pair(read: dict[str, RasterInfo]) -> None:
    """Forcing CROSS_MODAL on two optical images must FAIL that check, not pass silently."""
    report = report_for(read, "s2_pre", "s2_post", hint=PairType.CROSS_MODAL)
    assert check_of(report.checks, CheckName.MODALITY_DISTINCT).status is CheckStatus.FAIL
    assert report.overall is Overall.FAIL


def test_single_image_skips_every_pairwise_check(read: dict[str, RasterInfo]) -> None:
    report = report_for(read, "s2_pre")
    assert report.pair_type is PairType.SINGLE
    assert report.overall is Overall.PASS
    assert report.common_grid is None
    for name in (
        CheckName.CRS_MATCH,
        CheckName.BOUNDS_OVERLAP_IOU,
        CheckName.GSD_RATIO,
        CheckName.COREGISTRATION_OFFSET_PX,
        CheckName.SIZE_RATIO,
    ):
        assert check_of(report.checks, name).status is CheckStatus.SKIP


def test_non_georeferenced_input_warns_rather_than_failing(read: dict[str, RasterInfo]) -> None:
    """The benchmark PNG path must stay usable."""
    report = report_for(read, "benchmark_rgb")
    assert check_of(report.checks, CheckName.GEOREFERENCE_PRESENT).status is CheckStatus.WARN
    assert report.overall is Overall.PASS_WITH_WARNINGS


def test_mixed_georeferencing_fails(read: dict[str, RasterInfo]) -> None:
    report = report_for(read, "s2_pre", "benchmark_rgb")
    assert check_of(report.checks, CheckName.GEOREFERENCE_PRESENT).status is CheckStatus.FAIL
    assert report.overall is Overall.FAIL


# ------------------------------------------------------------------------ band aliases


def test_band_aliases_resolve_per_sensor() -> None:
    s2 = resolve_bands("Sentinel-2 (BigEarthNet-v2, 12-band)", band_count=12)
    assert s2["red"] == 4 and s2["nir"] == 8 and s2["swir1"] == 11

    s1 = resolve_bands("Sentinel-1 GRD", band_count=2, band_names=["VV", "VH"])
    assert s1 == {"vv": 1, "vh": 2}


def test_cartosat_has_no_swir_and_says_so() -> None:
    """Unavailability is explicit: NDBI is impossible on Cartosat and must not be faked."""
    cartosat = resolve_bands("Cartosat-2S MX", band_count=4)
    assert "swir1" not in cartosat
    assert missing_bands(["red", "nir", "swir1"], cartosat) == ["swir1"]


def test_declared_polarisation_names_override_the_table() -> None:
    resolved = resolve_bands(None, band_count=2, band_names=["VH", "VV"])
    assert resolved == {"vh": 1, "vv": 2}


def test_band_numbers_beyond_the_file_are_dropped() -> None:
    """A truncated stack keeps only the aliases that really exist in it."""
    assert resolve_bands("Sentinel-2 L2A", band_count=4) == {"blue": 2, "green": 3, "red": 4}
    assert resolve_bands("Sentinel-2 L2A", band_count=1) == {}


# ------------------------------------------------------------------------ capabilities


def test_supported_tasks_track_the_pair_type() -> None:
    bitemporal = supported_tasks(PairType.BI_TEMPORAL, 2, Overall.PASS)
    assert set(bitemporal) == {TaskType.CHANGE_VQA, TaskType.CHANGE_CAPTION, TaskType.CHANGE_MAP}

    single = supported_tasks(PairType.SINGLE, 1, Overall.PASS)
    assert TaskType.VQA in single and TaskType.CHANGE_VQA not in single

    cross = supported_tasks(PairType.CROSS_MODAL, 2, Overall.PASS)
    assert TaskType.CROSS_MODAL_VQA in cross


def test_a_failing_report_supports_nothing() -> None:
    assert supported_tasks(PairType.BI_TEMPORAL, 2, Overall.FAIL) == []
    assert supported_tasks(PairType.INCOMPATIBLE, 2, Overall.PASS) == []


def test_unsupported_is_never_offered() -> None:
    for pair_type in (PairType.SINGLE, PairType.BI_TEMPORAL, PairType.CROSS_MODAL):
        assert TaskType.UNSUPPORTED not in supported_tasks(pair_type, 2, Overall.PASS)


# --------------------------------------------------------------------------- pipeline


def source(scene: Scene) -> SourceImage:
    """Wrap a fixture as a pipeline input."""
    return SourceImage(path=scene.path, filename=scene.path.name)


def test_pipeline_end_to_end_on_a_bitemporal_pair(scenes: dict[str, Scene]) -> None:
    result = ingest([source(scenes["s2_pre"]), source(scenes["s2_post"])])
    assert [m.id for m in result.inputs] == ["img_0", "img_1"]
    assert [m.role for m in result.inputs] == [ImageRole.PRE, ImageRole.POST]
    assert result.compatibility.overall is Overall.PASS
    assert TaskType.CHANGE_VQA in result.supported_tasks


def test_pipeline_rejects_more_than_two_images(scenes: dict[str, Scene]) -> None:
    with pytest.raises(TooManyImagesError) as excinfo:
        ingest([source(scenes["s2_pre"])] * 3)
    assert excinfo.value.http_status == 400
    assert excinfo.value.code == "TOO_MANY_IMAGES"


def test_pipeline_rejects_a_half_georeferenced_pair(scenes: dict[str, Scene]) -> None:
    with pytest.raises(MissingGeoreferenceError) as excinfo:
        ingest([source(scenes["s2_pre"]), source(scenes["benchmark_rgb"])])
    assert excinfo.value.http_status == 422
    assert excinfo.value.ref == "img_1"


def test_pipeline_surfaces_check_warnings(scenes: dict[str, Scene]) -> None:
    result = ingest([source(scenes["s2_pre"]), source(scenes["s2_partial_overlap"])])
    codes = {warning.code for warning in result.warnings}
    assert "CHECK_BOUNDS_OVERLAP_IOU" in codes
