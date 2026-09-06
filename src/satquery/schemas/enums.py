"""Closed string enumerations for the frozen SatQuery AI API contract.

Every enum here is closed: a client receiving an unknown value must treat it as
a contract violation. See ``DOCS/API_CONTRACT.md`` §2.
"""

from __future__ import annotations

from enum import StrEnum


class TaskType(StrEnum):
    """Resolved analysis task (API_CONTRACT §2.1)."""

    VQA = "VQA"
    CAPTION = "CAPTION"
    GROUNDING = "GROUNDING"
    SEGMENTATION = "SEGMENTATION"
    COUNT = "COUNT"
    SCENE_CLASSIFY = "SCENE_CLASSIFY"
    CHANGE_VQA = "CHANGE_VQA"
    CHANGE_CAPTION = "CHANGE_CAPTION"
    CHANGE_MAP = "CHANGE_MAP"
    CROSS_MODAL_VQA = "CROSS_MODAL_VQA"
    CROSS_MODAL_COMPARE = "CROSS_MODAL_COMPARE"
    UNSUPPORTED = "UNSUPPORTED"


class PairType(StrEnum):
    """Relationship between the uploaded images (API_CONTRACT §2.2)."""

    SINGLE = "SINGLE"
    CROSS_MODAL = "CROSS_MODAL"
    BI_TEMPORAL = "BI_TEMPORAL"
    INCOMPATIBLE = "INCOMPATIBLE"


class Modality(StrEnum):
    """Sensing modality of a single input (API_CONTRACT §2.3)."""

    OPTICAL = "optical"
    SAR = "sar"
    PANCHROMATIC = "panchromatic"
    UNKNOWN = "unknown"


class ImageRole(StrEnum):
    """Role an image plays within its pair (API_CONTRACT §2.4)."""

    SINGLE = "single"
    PRE = "pre"
    POST = "post"
    OPTICAL = "optical"
    SAR = "sar"


class ArtifactType(StrEnum):
    """Kind of evidence produced by a tool (API_CONTRACT §2.5)."""

    RENDERED_VIEW = "RENDERED_VIEW"
    CHANGE_MASK = "CHANGE_MASK"
    SEGMENTATION = "SEGMENTATION"
    BBOX_SET = "BBOX_SET"
    HEATMAP = "HEATMAP"
    OVERLAY_PNG = "OVERLAY_PNG"
    GEOJSON = "GEOJSON"
    SCALARS = "SCALARS"
    TEXT = "TEXT"


class CheckName(StrEnum):
    """Identity of a compatibility check (API_CONTRACT §2.6)."""

    CRS_MATCH = "crs_match"
    BOUNDS_OVERLAP_IOU = "bounds_overlap_iou"
    GSD_RATIO = "gsd_ratio"
    COREGISTRATION_OFFSET_PX = "coregistration_offset_px"
    BAND_SUFFICIENCY = "band_sufficiency"
    MODALITY_DISTINCT = "modality_distinct"
    TEMPORAL_ORDERING = "temporal_ordering"
    GEOREFERENCE_PRESENT = "georeference_present"
    SIZE_RATIO = "size_ratio"
    NODATA_EXTENT = "nodata_extent"


class CheckStatus(StrEnum):
    """Outcome of one compatibility check (API_CONTRACT §2.7)."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


class Overall(StrEnum):
    """Aggregate compatibility verdict (API_CONTRACT §2.7).

    Derivation: any FAIL -> FAIL; else any WARN -> PASS_WITH_WARNINGS; else PASS.
    """

    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FAIL = "FAIL"


class ToolStatus(StrEnum):
    """Outcome of a single tool execution (API_CONTRACT §2.8)."""

    OK = "OK"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ToolCategory(StrEnum):
    """Registry grouping for a tool (API_CONTRACT §2.9)."""

    ANALYSIS = "analysis"
    VLM = "vlm"
    CV = "cv"
    GEO = "geo"
    FUSION = "fusion"


class Device(StrEnum):
    """Compute placement (API_CONTRACT §2.10)."""

    ROCM_0 = "rocm:0"
    CPU = "cpu"
    AUTO = "auto"


class PairTypeSource(StrEnum):
    """How ``CompatibilityReport.pair_type`` was decided (API_CONTRACT §3.3)."""

    METADATA = "metadata"
    HEURISTIC = "heuristic"
    USER_DECLARED = "user_declared"


class Resampling(StrEnum):
    """Resampling kernel used to build the common grid (API_CONTRACT §3.3)."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CUBIC = "cubic"


class HealthStatus(StrEnum):
    """Service-level health verdict (API_CONTRACT §4.9)."""

    OK = "ok"
    DEGRADED = "degraded"
