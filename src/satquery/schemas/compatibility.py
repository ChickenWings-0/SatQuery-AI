"""Compatibility checking schemas (API_CONTRACT §3.2, §3.3)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from satquery.schemas.enums import (
    CheckName,
    CheckStatus,
    Overall,
    PairType,
    PairTypeSource,
    Resampling,
)


class CheckResult(BaseModel):
    """One compatibility check, including the ones that were skipped."""

    model_config = ConfigDict(extra="forbid")

    name: CheckName
    status: CheckStatus
    value: float | str | None = Field(default=None, description="Measured value.")
    threshold: str | None = Field(
        default=None, description="Human-readable rule, e.g. 'iou >= 0.80'."
    )
    detail: str = Field(description="One sentence, safe to display verbatim.")


class CommonGrid(BaseModel):
    """The grid both images were resampled onto for pairwise analysis."""

    model_config = ConfigDict(extra="forbid")

    crs: str
    transform: list[float] = Field(min_length=6, max_length=6)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    gsd_m: float = Field(gt=0.0)
    resampling: Resampling


class CompatibilityReport(BaseModel):
    """Whether the inputs can be analysed together, and what it cost to get there."""

    model_config = ConfigDict(extra="forbid")

    pair_type: PairType
    pair_type_source: PairTypeSource
    checks: list[CheckResult] = Field(description="Every applicable check, including SKIP.")
    overall: Overall = Field(
        description="any FAIL -> FAIL; else any WARN -> PASS_WITH_WARNINGS; else PASS."
    )
    actions_taken: list[str] = Field(
        default_factory=list,
        description="e.g. 'reprojected img_1 EPSG:4326->EPSG:32643 (bilinear)'.",
    )
    common_grid: CommonGrid | None = Field(
        default=None, description="None for SINGLE and for FAIL."
    )
