"""Tool registry, planning and execution schemas (API_CONTRACT §3.5-§3.8)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from satquery.schemas.enums import (
    ArtifactType,
    Device,
    Modality,
    PairType,
    ToolCategory,
    ToolStatus,
)


class InputContract(BaseModel):
    """What a tool is willing to accept — the basis of capability matching."""

    model_config = ConfigDict(extra="forbid")

    pair_types: list[PairType]
    modalities: list[Modality]
    min_images: int = Field(ge=1)
    max_images: int = Field(ge=1)
    required_bands: list[str] = Field(
        default_factory=list, description="Logical band names, resolved per-sensor."
    )
    optional_bands: list[str] = Field(default_factory=list)
    gsd_range_m: list[float] | None = Field(
        default=None, min_length=2, max_length=2, description="[min_gsd_m, max_gsd_m]."
    )
    min_size_px: int | None = Field(default=None, gt=0)
    max_size_px: int | None = Field(default=None, gt=0)
    requires_georeference: bool = False


class ToolSpec(BaseModel):
    """One registry entry, as served by ``GET /v1/registry``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Registry key, snake_case, stable.")
    version: str = Field(description="'1.0.0' or '1.0.0+ckpt:9a3f'.")
    category: ToolCategory
    description: str = Field(description="One line, shown in the capabilities panel.")
    accepts: InputContract
    produces: list[ArtifactType]
    scalars_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema of this tool's FactSheet contribution."
    )
    device: Device
    est_ms: int = Field(ge=0, description="Typical wall time, for the progress bar.")
    fallback: str | None = Field(
        default=None, description="Name of the tool used when this one fails."
    )
    available: bool = Field(description="Runtime: false when weights/backend are missing.")
    unavailable_reason: str | None = None


class PlanStep(BaseModel):
    """One node of the execution DAG, in topological order."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1, description="1-indexed position in topological order.")
    tool: str
    depends_on: list[int] = Field(default_factory=list)
    input_refs: list[str] = Field(description="img_* and/or art_* references.")
    reason: str = Field(description="e.g. 'policy_table:CHANGE_VQA|BI_TEMPORAL|optical'.")


class Execution(BaseModel):
    """The record of actually running one :class:`PlanStep` — the audit payload."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1, description="Matches PlanStep.step.")
    tool: str
    version: str
    status: ToolStatus
    device_used: Device
    duration_ms: int = Field(ge=0)
    params: dict[str, Any] = Field(
        default_factory=dict, description="Exact effective parameters — the auditability payload."
    )
    input_refs: list[str] = Field(default_factory=list)
    output_refs: list[str] = Field(default_factory=list)
    scalars: dict[str, float | str] = Field(
        default_factory=dict, description="Merged into fact_sheet under this tool's namespace."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    cache_hit: bool = False
    fallback_of: str | None = Field(
        default=None, description="Set when this execution replaced a failed tool."
    )
    error: str | None = None
