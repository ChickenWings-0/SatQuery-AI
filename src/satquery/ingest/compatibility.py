"""The compatibility check battery (API_CONTRACT §3.2, §3.3).

This module owns *policy*: it turns the measurements produced by
:mod:`satquery.ingest.coregistration` into PASS/WARN/FAIL verdicts using the
frozen threshold table, derives the pair type, and computes the common grid.

The thresholds are declared once, in :data:`THRESHOLDS`, so the numbers a check
enforces and the numbers it prints in ``threshold`` can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from satquery.ingest.bands import missing_bands, resolve_bands
from satquery.ingest.coregistration import (
    GridSpec,
    OffsetEstimate,
    common_grid,
    crs_equivalent,
    estimate_offset,
    gsd_ratio,
    overlap_iou_wgs84,
    reproject_bounds,
    size_ratio,
)
from satquery.ingest.modality import ModalityResult
from satquery.ingest.reader import RasterInfo
from satquery.schemas.compatibility import CheckResult, CommonGrid, CompatibilityReport
from satquery.schemas.enums import (
    CheckName,
    CheckStatus,
    Modality,
    Overall,
    PairType,
    PairTypeSource,
    Resampling,
)


@dataclass(frozen=True)
class Band:
    """A PASS/WARN band for a scalar measurement, with its printable rule."""

    pass_max: float
    warn_max: float
    rule: str


THRESHOLDS: Final[dict[CheckName, Band]] = {
    CheckName.BOUNDS_OVERLAP_IOU: Band(pass_max=0.80, warn_max=0.30, rule="iou >= 0.80"),
    CheckName.GSD_RATIO: Band(pass_max=1.5, warn_max=4.0, rule="ratio <= 1.5"),
    CheckName.COREGISTRATION_OFFSET_PX: Band(pass_max=1.0, warn_max=5.0, rule="offset <= 1.0 px"),
    CheckName.SIZE_RATIO: Band(pass_max=1.5, warn_max=4.0, rule="ratio <= 1.5"),
    CheckName.NODATA_EXTENT: Band(pass_max=5.0, warn_max=40.0, rule="nodata < 5 %"),
}
"""Frozen by API_CONTRACT §3.2. Changing a number here changes the contract."""

_OPTICAL_LIKE: Final[frozenset[Modality]] = frozenset({Modality.OPTICAL, Modality.PANCHROMATIC})

# Logical bands each modality needs before any tool can do useful work.
_MINIMUM_BANDS: Final[dict[Modality, list[str]]] = {
    Modality.OPTICAL: ["red", "green", "blue"],
    Modality.PANCHROMATIC: ["pan"],
    Modality.SAR: ["vv"],
    Modality.UNKNOWN: [],
}


def _descending(value: float, band: Band) -> CheckStatus:
    """Grade a measurement where smaller is better (ratios, offsets, nodata)."""
    if value <= band.pass_max:
        return CheckStatus.PASS
    if value <= band.warn_max:
        return CheckStatus.WARN
    return CheckStatus.FAIL


def _ascending(value: float, band: Band) -> CheckStatus:
    """Grade a measurement where larger is better (overlap IoU)."""
    if value >= band.pass_max:
        return CheckStatus.PASS
    if value >= band.warn_max:
        return CheckStatus.WARN
    return CheckStatus.FAIL


def derive_overall(checks: list[CheckResult]) -> Overall:
    """Apply the frozen derivation rule of API_CONTRACT §2.7."""
    statuses = {check.status for check in checks}
    if CheckStatus.FAIL in statuses:
        return Overall.FAIL
    if CheckStatus.WARN in statuses:
        return Overall.PASS_WITH_WARNINGS
    return Overall.PASS


def detect_pair_type(infos: list[RasterInfo], modalities: list[Modality]) -> PairType:
    """Decide how the uploaded images relate to one another.

    A single image is ``SINGLE``. Two images are ``CROSS_MODAL`` when exactly one
    of them is radar, and ``BI_TEMPORAL`` otherwise — two acquisitions of the
    same kind of sensor are a time series regardless of whether timestamps were
    recoverable.
    """
    if len(infos) <= 1:
        return PairType.SINGLE
    sar = [m is Modality.SAR for m in modalities]
    optical = [m in _OPTICAL_LIKE for m in modalities]
    if any(sar) and any(optical):
        return PairType.CROSS_MODAL
    return PairType.BI_TEMPORAL


# --------------------------------------------------------------------------- checks


def _skip(name: CheckName, detail: str, threshold: str | None = None) -> CheckResult:
    return CheckResult(
        name=name, status=CheckStatus.SKIP, value=None, threshold=threshold, detail=detail
    )


def check_georeference_present(infos: list[RasterInfo]) -> CheckResult:
    """PASS both georeferenced, WARN neither (benchmark path), FAIL exactly one."""
    flags = [info.is_georeferenced for info in infos]
    if all(flags):
        return CheckResult(
            name=CheckName.GEOREFERENCE_PRESENT,
            status=CheckStatus.PASS,
            value="both" if len(flags) > 1 else "present",
            threshold="both georeferenced",
            detail="Every image carries a CRS and an affine transform.",
        )
    if not any(flags):
        return CheckResult(
            name=CheckName.GEOREFERENCE_PRESENT,
            status=CheckStatus.WARN,
            value="neither",
            threshold="both georeferenced",
            detail=(
                "No image is georeferenced. Analysis continues in pixel space; areas "
                "cannot be reported in square metres."
            ),
        )
    return CheckResult(
        name=CheckName.GEOREFERENCE_PRESENT,
        status=CheckStatus.FAIL,
        value="exactly one",
        threshold="both georeferenced",
        detail=(
            "One image is georeferenced and the other is not, so they cannot be placed "
            "on a common grid."
        ),
    )


def check_crs_match(infos: list[RasterInfo], actions: list[str]) -> CheckResult:
    """PASS on identical CRS, WARN when a reprojection is needed and possible."""
    if len(infos) < 2:
        return _skip(CheckName.CRS_MATCH, "Only one image was supplied.", "identical CRS")
    a, b = infos[0], infos[1]
    if not (a.is_georeferenced and b.is_georeferenced):
        return _skip(
            CheckName.CRS_MATCH, "At least one image is not georeferenced.", "identical CRS"
        )
    if crs_equivalent(a, b):
        return CheckResult(
            name=CheckName.CRS_MATCH,
            status=CheckStatus.PASS,
            value=a.crs,
            threshold="identical CRS",
            detail=f"Both images are in {a.crs}; no reprojection was needed.",
        )

    src, dst = b.crs_obj, a.crs_obj
    try:
        if src is None or dst is None or b.bounds_native is None:
            raise ValueError("missing CRS or bounds")
        reproject_bounds(b.bounds_native, src, dst)
    except (ValueError, RuntimeError):
        return CheckResult(
            name=CheckName.CRS_MATCH,
            status=CheckStatus.FAIL,
            value=f"{a.crs} vs {b.crs}",
            threshold="identical CRS",
            detail=f"{b.crs} cannot be reprojected to {a.crs}.",
        )

    actions.append(f"reprojected img_1 {b.crs}->{a.crs} (bilinear)")
    return CheckResult(
        name=CheckName.CRS_MATCH,
        status=CheckStatus.WARN,
        value=f"{a.crs} vs {b.crs}",
        threshold="identical CRS",
        detail=f"The images differ in CRS; img_1 was reprojected from {b.crs} to {a.crs}.",
    )


def check_bounds_overlap(infos: list[RasterInfo]) -> CheckResult:
    """Grade the WGS84 footprint IoU against the frozen 0.80 / 0.30 bands."""
    band = THRESHOLDS[CheckName.BOUNDS_OVERLAP_IOU]
    if len(infos) < 2:
        return _skip(CheckName.BOUNDS_OVERLAP_IOU, "Only one image was supplied.", band.rule)
    iou = overlap_iou_wgs84(infos[0], infos[1])
    if iou is None:
        return _skip(
            CheckName.BOUNDS_OVERLAP_IOU,
            "At least one image is not georeferenced, so footprints cannot be compared.",
            band.rule,
        )
    status = _ascending(iou, band)
    if status is CheckStatus.PASS:
        detail = f"The two footprints overlap by {iou:.0%} of their combined extent."
    elif status is CheckStatus.WARN:
        detail = (
            f"The two footprints overlap by only {iou:.0%} of their combined extent; "
            "results outside the shared area are not comparable."
        )
    else:
        detail = (
            f"The two images overlap by {iou:.0%} of their combined extent. Pairwise "
            "analysis needs at least 30%."
        )
    return CheckResult(
        name=CheckName.BOUNDS_OVERLAP_IOU,
        status=status,
        value=round(iou, 4),
        threshold=band.rule,
        detail=detail,
    )


def check_gsd_ratio(infos: list[RasterInfo]) -> CheckResult:
    """Grade the coarser/finer ground sample distance ratio."""
    band = THRESHOLDS[CheckName.GSD_RATIO]
    if len(infos) < 2:
        return _skip(CheckName.GSD_RATIO, "Only one image was supplied.", band.rule)
    ratio = gsd_ratio(infos[0].gsd_m, infos[1].gsd_m)
    if ratio is None:
        return _skip(
            CheckName.GSD_RATIO,
            "Ground sample distance is unknown for at least one image.",
            band.rule,
        )
    status = _descending(ratio, band)
    detail = (
        f"Resolutions are {infos[0].gsd_m:.2f} m and {infos[1].gsd_m:.2f} m (ratio {ratio:.2f})."
        if status is CheckStatus.PASS
        else (
            f"Resolutions differ by {ratio:.2f}x ({infos[0].gsd_m:.2f} m vs "
            f"{infos[1].gsd_m:.2f} m); the finer image is resampled to the coarser grid."
        )
    )
    if status is CheckStatus.FAIL:
        detail = f"Resolutions differ by {ratio:.2f}x, beyond the 4x limit for joint analysis."
    return CheckResult(
        name=CheckName.GSD_RATIO,
        status=status,
        value=round(ratio, 4),
        threshold=band.rule,
        detail=detail,
    )


def check_coregistration(
    infos: list[RasterInfo], offset: OffsetEstimate | None, actions: list[str]
) -> CheckResult:
    """Grade the residual sub-pixel misalignment recovered by phase correlation."""
    band = THRESHOLDS[CheckName.COREGISTRATION_OFFSET_PX]
    if len(infos) < 2:
        return _skip(CheckName.COREGISTRATION_OFFSET_PX, "Only one image was supplied.", band.rule)
    if offset is None:
        return _skip(
            CheckName.COREGISTRATION_OFFSET_PX,
            "The images could not be placed on a common grid, so alignment was not measured.",
            band.rule,
        )
    if not offset.reliable:
        return CheckResult(
            name=CheckName.COREGISTRATION_OFFSET_PX,
            status=CheckStatus.WARN,
            value=None,
            threshold=band.rule,
            detail=offset.detail,
        )
    status = _descending(offset.magnitude_px, band)
    if status is not CheckStatus.FAIL:
        actions.append(
            f"measured a {offset.magnitude_px:.2f} px residual shift "
            f"(dx {offset.dx_px:+.2f}, dy {offset.dy_px:+.2f})"
        )
    detail = offset.detail
    if status is CheckStatus.FAIL:
        detail = (
            f"The images are misaligned by {offset.magnitude_px:.2f} px, beyond the 5 px "
            "limit; per-pixel comparison would be meaningless."
        )
    return CheckResult(
        name=CheckName.COREGISTRATION_OFFSET_PX,
        status=status,
        value=offset.magnitude_px,
        threshold=band.rule,
        detail=detail,
    )


def check_band_sufficiency(infos: list[RasterInfo], verdicts: list[ModalityResult]) -> CheckResult:
    """Confirm each image resolves the logical bands its modality requires.

    A recognised sensor resolves bands through its alias map; an unrecognised one
    falls back to positional resolution, which is usable but is reported as a
    substitution rather than a clean PASS.
    """
    resolved_all: list[str] = []
    missing_all: list[str] = []
    substituted = False

    for info, verdict in zip(infos, verdicts, strict=True):
        required = _MINIMUM_BANDS.get(verdict.modality, [])
        resolved = resolve_bands(
            sensor=verdict.sensor_guess,
            band_count=info.band_count,
            band_names=info.band_names,
        )
        gaps = missing_bands(required, resolved)
        missing_all.extend(gaps)
        resolved_all.extend(sorted(resolved))
        if not gaps and verdict.sensor_guess is None:
            substituted = True

    if missing_all:
        return CheckResult(
            name=CheckName.BAND_SUFFICIENCY,
            status=CheckStatus.FAIL,
            value=",".join(sorted(set(missing_all))),
            threshold="required bands present",
            detail=(
                "The following bands are required but absent with no substitute: "
                + ", ".join(sorted(set(missing_all)))
                + "."
            ),
        )
    if substituted:
        return CheckResult(
            name=CheckName.BAND_SUFFICIENCY,
            status=CheckStatus.WARN,
            value=",".join(sorted(set(resolved_all))),
            threshold="required bands present",
            detail=(
                "The sensor was not recognised, so bands were resolved by position rather "
                "than by a known alias map."
            ),
        )
    return CheckResult(
        name=CheckName.BAND_SUFFICIENCY,
        status=CheckStatus.PASS,
        value=",".join(sorted(set(resolved_all))),
        threshold="required bands present",
        detail="Every band required by the detected modality is present.",
    )


def check_modality_distinct(pair_type: PairType, modalities: list[Modality]) -> CheckResult:
    """Only meaningful for a cross-modal pair: the two sensors must actually differ."""
    if pair_type is not PairType.CROSS_MODAL:
        return _skip(
            CheckName.MODALITY_DISTINCT,
            "Not applicable outside a cross-modal pair.",
            "CROSS_MODAL only",
        )
    if Modality.UNKNOWN in modalities:
        return CheckResult(
            name=CheckName.MODALITY_DISTINCT,
            status=CheckStatus.WARN,
            value=",".join(m.value for m in modalities),
            threshold="modalities differ",
            detail="One image's modality could not be determined with confidence.",
        )
    if len(set(modalities)) < 2:
        return CheckResult(
            name=CheckName.MODALITY_DISTINCT,
            status=CheckStatus.FAIL,
            value=modalities[0].value,
            threshold="modalities differ",
            detail="Both images have the same modality, so there is nothing to fuse.",
        )
    return CheckResult(
        name=CheckName.MODALITY_DISTINCT,
        status=CheckStatus.PASS,
        value=",".join(m.value for m in modalities),
        threshold="modalities differ",
        detail="The two images come from different sensing modalities.",
    )


def check_temporal_ordering(infos: list[RasterInfo], pair_type: PairType) -> CheckResult:
    """PASS when timestamps exist and are ordered; FAIL when they are identical."""
    if pair_type is not PairType.BI_TEMPORAL or len(infos) < 2:
        return _skip(
            CheckName.TEMPORAL_ORDERING,
            "Not applicable outside a bi-temporal pair.",
            "timestamps present and ordered",
        )
    first, second = infos[0].acquisition_time, infos[1].acquisition_time
    if first is None or second is None:
        return CheckResult(
            name=CheckName.TEMPORAL_ORDERING,
            status=CheckStatus.WARN,
            value=None,
            threshold="timestamps present and ordered",
            detail=(
                "No acquisition times were found in the file metadata; upload order was "
                "assumed to be before then after."
            ),
        )
    if first == second:
        return CheckResult(
            name=CheckName.TEMPORAL_ORDERING,
            status=CheckStatus.FAIL,
            value=first.isoformat(),
            threshold="timestamps present and ordered",
            detail=(
                "Both images were acquired at the same instant; there is no interval to compare."
            ),
        )
    ordered = sorted([first, second])
    return CheckResult(
        name=CheckName.TEMPORAL_ORDERING,
        status=CheckStatus.PASS,
        value=f"{ordered[0].date().isoformat()} -> {ordered[1].date().isoformat()}",
        threshold="timestamps present and ordered",
        detail="Acquisition times were read from metadata and define a clear interval.",
    )


def check_size_ratio(infos: list[RasterInfo]) -> CheckResult:
    """Grade the ratio of the two images' longest edges."""
    band = THRESHOLDS[CheckName.SIZE_RATIO]
    if len(infos) < 2:
        return _skip(CheckName.SIZE_RATIO, "Only one image was supplied.", band.rule)
    ratio = size_ratio(infos[0], infos[1])
    status = _descending(ratio, band)
    return CheckResult(
        name=CheckName.SIZE_RATIO,
        status=status,
        value=round(ratio, 4),
        threshold=band.rule,
        detail=(
            f"The images are {infos[0].width}x{infos[0].height} and "
            f"{infos[1].width}x{infos[1].height} pixels (ratio {ratio:.2f})."
        ),
    )


def check_nodata_extent(infos: list[RasterInfo]) -> CheckResult:
    """Grade the worst nodata fraction across the inputs."""
    band = THRESHOLDS[CheckName.NODATA_EXTENT]
    worst = max(info.nodata_pct for info in infos)
    status = _descending(worst, band)
    return CheckResult(
        name=CheckName.NODATA_EXTENT,
        status=status,
        value=round(worst, 4),
        threshold=band.rule,
        detail=(
            f"At most {worst:.2f}% of any image is nodata."
            if status is CheckStatus.PASS
            else (
                f"{worst:.2f}% of an image is nodata; statistics over the empty area are "
                "excluded and confidence is reduced."
            )
        ),
    )


# ------------------------------------------------------------------------- battery


def assess(
    infos: list[RasterInfo],
    verdicts: list[ModalityResult],
    pair_type_hint: PairType | None = None,
) -> CompatibilityReport:
    """Run every applicable check and assemble the :class:`CompatibilityReport`.

    Args:
        infos: Parsed rasters, in upload order.
        verdicts: Classifier output per image, aligned with *infos*.
        pair_type_hint: A user-declared pair type, which wins over detection and
            sets ``pair_type_source`` to ``user_declared``.

    Returns:
        A report whose ``overall`` follows the frozen derivation rule and whose
        ``common_grid`` is populated for any georeferenced, non-failing pair.
    """
    actions: list[str] = []
    modalities = [verdict.modality for verdict in verdicts]
    detected = detect_pair_type(infos, modalities)
    pair_type = pair_type_hint or detected
    source = PairTypeSource.USER_DECLARED if pair_type_hint else PairTypeSource.METADATA

    offset: OffsetEstimate | None = None
    grid: GridSpec | None = None
    if len(infos) == 2:
        grid = common_grid(infos[0], infos[1], Resampling.BILINEAR)
        if grid is not None:
            offset = estimate_offset(infos[0], infos[1], grid)

    checks = [
        check_crs_match(infos, actions),
        check_bounds_overlap(infos),
        check_gsd_ratio(infos),
        check_coregistration(infos, offset, actions),
        check_band_sufficiency(infos, verdicts),
        check_modality_distinct(pair_type, modalities),
        check_temporal_ordering(infos, pair_type),
        check_georeference_present(infos),
        check_size_ratio(infos),
        check_nodata_extent(infos),
    ]

    overall = derive_overall(checks)
    if overall is Overall.FAIL and len(infos) > 1:
        pair_type = PairType.INCOMPATIBLE

    common: CommonGrid | None = None
    if grid is not None and overall is not Overall.FAIL and pair_type is not PairType.SINGLE:
        common = CommonGrid(
            crs=grid.crs,
            transform=grid.transform_list,
            width=grid.width,
            height=grid.height,
            gsd_m=round(grid.gsd_m, 6),
            resampling=grid.resampling,
        )
        actions.append(
            f"resampled both images onto a common {grid.gsd_m:.2f} m grid in {grid.crs} "
            f"({grid.width}x{grid.height} px, {grid.resampling.value})"
        )

    return CompatibilityReport(
        pair_type=pair_type,
        pair_type_source=source,
        checks=checks,
        overall=overall,
        actions_taken=actions,
        common_grid=common,
    )
