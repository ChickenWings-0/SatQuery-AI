"""Turn a :class:`~satquery.ingest.reader.RasterInfo` into a frozen :class:`InputManifest`."""

from __future__ import annotations

from pathlib import PurePosixPath

from satquery.ingest.modality import ModalityResult, classify
from satquery.ingest.reader import RasterInfo
from satquery.schemas.enums import ImageRole, Modality, PairType
from satquery.schemas.manifest import InputManifest

_UNSAFE_FILENAME_CHARS = str.maketrans({"/": "_", "\\": "_", "\x00": "_"})


def sanitise_filename(name: str) -> str:
    """Strip directory components and control characters from a client filename."""
    base = PurePosixPath(name.replace("\\", "/")).name
    cleaned = base.translate(_UNSAFE_FILENAME_CHARS).strip().lstrip(".")
    return cleaned or "unnamed"


def image_id(index: int) -> str:
    """Return the contract's stable input id for upload position *index*."""
    return f"img_{index}"


def build_manifest(
    info: RasterInfo,
    index: int,
    modality: ModalityResult | None = None,
    role: ImageRole | None = None,
    filename: str | None = None,
) -> InputManifest:
    """Build the manifest for one image.

    Args:
        info: The parsed raster.
        index: Zero-based upload position, which fixes the ``img_{n}`` id.
        modality: Pre-computed classification; classified here when omitted.
        role: Role override. Roles are otherwise assigned once the pair type is
            known, by :func:`assign_roles`.
        filename: Original client filename, when it differs from the temp file's.

    Returns:
        A populated :class:`InputManifest`.
    """
    verdict = modality or classify(info)
    warnings = list(info.warnings)
    warnings.extend(verdict.reasons)
    if info.nodata_pct > 0:
        warnings.append(f"{info.nodata_pct:.2f}% of the image is nodata.")
    if verdict.sensor_guess is None:
        warnings.append("The sensor could not be identified from the file's structure.")

    return InputManifest(
        id=image_id(index),
        role=role,
        filename=sanitise_filename(filename or info.filename),
        sha256=info.sha256,
        size_bytes=info.size_bytes,
        driver=info.driver,
        modality=verdict.modality,
        modality_confidence=round(verdict.confidence, 4),
        sensor_guess=verdict.sensor_guess,
        crs=info.crs,
        transform=info.transform_list,
        bounds_native=list(info.bounds_native) if info.bounds_native else None,
        bounds_wgs84=list(info.bounds_wgs84) if info.bounds_wgs84 else None,
        gsd_m=info.gsd_m,
        width=info.width,
        height=info.height,
        band_count=info.band_count,
        dtype=info.dtype,
        band_names=info.band_names,
        nodata_pct=info.nodata_pct,
        acquisition_time=info.acquisition_time,
        is_georeferenced=info.is_georeferenced,
        warnings=warnings,
    )


def assign_roles(
    manifests: list[InputManifest],
    pair_type: PairType,
    overrides: dict[str, str] | None = None,
) -> None:
    """Assign :class:`ImageRole` in place, once the pair type is known.

    ``BI_TEMPORAL`` orders by acquisition time when both timestamps are present
    and differ, otherwise by upload order. ``CROSS_MODAL`` labels by modality.
    Explicit ``options.roles`` overrides always win.
    """
    if len(manifests) == 1:
        manifests[0].role = ImageRole.SINGLE
    elif pair_type is PairType.BI_TEMPORAL and len(manifests) == 2:
        first, second = manifests
        times = [first.acquisition_time, second.acquisition_time]
        if all(times) and times[0] != times[1] and times[0] > times[1]:  # type: ignore[operator]
            first, second = second, first
        first.role = ImageRole.PRE
        second.role = ImageRole.POST
    elif pair_type is PairType.CROSS_MODAL:
        for manifest in manifests:
            manifest.role = (
                ImageRole.SAR if manifest.modality is Modality.SAR else ImageRole.OPTICAL
            )

    for manifest in manifests:
        override = (overrides or {}).get(manifest.filename)
        if override:
            try:
                manifest.role = ImageRole(override)
            except ValueError:
                manifest.warnings.append(f"Ignored unrecognised role override '{override}'.")
