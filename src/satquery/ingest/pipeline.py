"""The ingestion pipeline: files in, manifests plus a compatibility report out.

This is the single entry point the API layer uses. It owns the order of
operations — read, classify, assess, assign roles — so no router has to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from satquery.ingest.capabilities import supported_tasks
from satquery.ingest.compatibility import assess
from satquery.ingest.errors import (
    MissingGeoreferenceError,
    NoImagesError,
    TooManyImagesError,
)
from satquery.ingest.manifest_builder import assign_roles, build_manifest
from satquery.ingest.modality import ModalityResult, classify
from satquery.ingest.reader import RasterInfo, open_raster
from satquery.schemas.compatibility import CompatibilityReport
from satquery.schemas.enums import CheckName, CheckStatus, PairType, TaskType
from satquery.schemas.manifest import InputManifest
from satquery.schemas.trace import WarningItem

MAX_IMAGES = 2
"""API_CONTRACT §4.1: analyses relate at most two images."""


@dataclass
class SourceImage:
    """One uploaded file, already written to local disk."""

    path: Path
    filename: str


@dataclass
class IngestResult:
    """Everything ingestion produced, ready to be shaped into any response."""

    inputs: list[InputManifest]
    compatibility: CompatibilityReport
    supported_tasks: list[TaskType]
    warnings: list[WarningItem] = field(default_factory=list)
    rasters: list[RasterInfo] = field(default_factory=list)
    verdicts: list[ModalityResult] = field(default_factory=list)


def _collect_warnings(
    manifests: list[InputManifest], report: CompatibilityReport
) -> list[WarningItem]:
    """Lift per-image and per-check warnings into the response's warnings array."""
    warnings: list[WarningItem] = []
    for manifest in manifests:
        for message in manifest.warnings:
            warnings.append(WarningItem(code="INPUT_QUALITY", message=message, ref=manifest.id))
    for check in report.checks:
        if check.status is CheckStatus.WARN:
            warnings.append(
                WarningItem(code=f"CHECK_{check.name.value.upper()}", message=check.detail)
            )
    return warnings


def ingest(
    images: list[SourceImage],
    pair_type_hint: PairType | None = None,
    roles: dict[str, str] | None = None,
    max_images: int = MAX_IMAGES,
) -> IngestResult:
    """Read every image, classify it, and assess whether the set hangs together.

    Args:
        images: Files on local disk, in upload order.
        pair_type_hint: Forces the pair type and marks it ``user_declared``.
        roles: ``filename -> ImageRole`` overrides from ``options.roles``.
        max_images: Upper bound on the number of images accepted.

    Returns:
        An :class:`IngestResult` carrying manifests, the compatibility report and
        the tasks these inputs can support.

    Raises:
        NoImagesError: No files were supplied.
        TooManyImagesError: More than *max_images* files were supplied.
        MissingGeoreferenceError: Exactly one image of a pair is georeferenced.
    """
    if not images:
        raise NoImagesError()
    if len(images) > max_images:
        raise TooManyImagesError(count=len(images), maximum=max_images)

    rasters = [open_raster(image.path, ref=f"img_{index}") for index, image in enumerate(images)]
    verdicts = [classify(raster) for raster in rasters]
    report = assess(rasters, verdicts, pair_type_hint=pair_type_hint)

    georeference = next(
        check for check in report.checks if check.name is CheckName.GEOREFERENCE_PRESENT
    )
    if georeference.status is CheckStatus.FAIL:
        ungeoreferenced = next(
            f"img_{i}" for i, raster in enumerate(rasters) if not raster.is_georeferenced
        )
        raise MissingGeoreferenceError(ref=ungeoreferenced)

    manifests = [
        build_manifest(raster, index=index, modality=verdict, filename=image.filename)
        for index, (raster, verdict, image) in enumerate(
            zip(rasters, verdicts, images, strict=True)
        )
    ]
    assign_roles(manifests, report.pair_type, overrides=roles)

    return IngestResult(
        inputs=manifests,
        compatibility=report,
        supported_tasks=supported_tasks(
            pair_type=report.pair_type,
            image_count=len(manifests),
            overall=report.overall,
            modalities=[verdict.modality for verdict in verdicts],
        ),
        warnings=_collect_warnings(manifests, report),
        rasters=rasters,
        verdicts=verdicts,
    )
