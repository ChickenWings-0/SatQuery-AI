"""Answer, evidence and audit trace schemas (API_CONTRACT §3.4, §3.9-§3.11)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from satquery.schemas.compatibility import CompatibilityReport
from satquery.schemas.enums import ArtifactType, TaskType
from satquery.schemas.manifest import InputManifest
from satquery.schemas.tool import Execution, PlanStep
from satquery.schemas.version import SCHEMA_VERSION


class ArtifactGeo(BaseModel):
    """Georeferencing of a raster artifact."""

    model_config = ConfigDict(extra="forbid")

    crs: str
    transform: list[float] = Field(min_length=6, max_length=6)


class ArtifactRef(BaseModel):
    """A piece of evidence the pipeline produced. Immutable and cacheable forever."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="art_0, art_1, ... stable within a trace.")
    type: ArtifactType
    mime: str = Field(description="image/png, image/tiff, application/geo+json, application/json.")
    label: str = Field(description="Display name, e.g. 'Change mask (binary)'.")
    url: str | None = Field(default=None, description="None for SCALARS/TEXT.")
    geotiff_url: str | None = None
    geojson_url: str | None = None
    geo: ArtifactGeo | None = Field(
        default=None, description="None when the artifact is not georeferenced."
    )
    width: int | None = None
    height: int | None = None
    stats: dict[str, Any] | None = Field(default=None, description="Artifact-type specific.")
    inline: dict[str, Any] | None = Field(
        default=None, description="Payload for SCALARS/BBOX_SET/TEXT — avoids a second round trip."
    )
    produced_by_step: int = Field(ge=1, description="Index into plan.steps.")


class Citation(BaseModel):
    """Binds one claim in the answer text to one measured scalar."""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(description="The exact span of answer text this supports.")
    source: str = Field(
        description="Grammar: 'step:{n}/scalars.{dotted.path}'; join two with '|'.",
        examples=["step:2/scalars.changed_area_pct"],
    )
    value: float | str = Field(description="The cited value, as measured.")


class Answer(BaseModel):
    """The synthesised natural-language answer plus its honesty signals."""

    model_config = ConfigDict(extra="forbid")

    text: str
    citations: list[Citation] = Field(default_factory=list)
    uncited_numeric_spans: list[str] = Field(
        default_factory=list,
        description="Numbers the CitationValidator could not resolve. Non-empty is a visible "
        "honesty signal — the frontend must render it.",
    )
    generator: str = Field(description="e.g. 'vlm_change_vqa@1.0.0+adapter:sq-lora-v3'.")
    template_fallback: bool = Field(
        default=False, description="True when a deterministic template wrote the text."
    )


class Confidence(BaseModel):
    """Overall confidence and the components it was derived from."""

    model_config = ConfigDict(extra="forbid")

    overall: float = Field(ge=0.0, le=1.0)
    method: str = Field(description="e.g. 'weighted_tool_agreement_v1'.")
    components: dict[str, float] = Field(default_factory=dict)
    caps_applied: list[str] = Field(
        default_factory=list, description="Names of any rule that clamped `overall`."
    )


class QuerySpec(BaseModel):
    """The user's question, as received and as normalised."""

    model_config = ConfigDict(extra="forbid")

    raw: str
    normalized: str
    language: str = Field(default="en", description="ISO 639-1.")


class ResolvedTask(BaseModel):
    """What the query classifier decided the user is asking for."""

    model_config = ConfigDict(extra="forbid")

    primary: TaskType
    secondary: list[TaskType] = Field(default_factory=list)
    slots: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    classifier: str = Field(description="e.g. 'rules_v1+llm_slotfill_v1'.")


class Plan(BaseModel):
    """The tool DAG chosen for this query."""

    model_config = ConfigDict(extra="forbid")

    planner: str = Field(description="e.g. 'policy_table_v1'.")
    policy_key: str = Field(description="e.g. 'CHANGE_VQA|BI_TEMPORAL|optical'.")
    steps: list[PlanStep] = Field(default_factory=list)


class WarningItem(BaseModel):
    """A non-fatal condition worth surfacing to the user."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    ref: str | None = Field(default=None, description="img_*, art_* or None.")


class ErrorItem(BaseModel):
    """A failure that did not prevent a partial result."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    step: int | None = None


class AuditTrace(BaseModel):
    """The full, replayable record of one analysis — the requirement-5 deliverable.

    Top-level key order is frozen by API_CONTRACT §3.11 and is preserved here.
    """

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(
        min_length=32,
        max_length=32,
        pattern=r"^[0-9a-f]{32}$",
        description="32-char lowercase hex.",
    )
    schema_version: str = SCHEMA_VERSION
    created_at: datetime
    duration_ms: int = Field(ge=0)
    query: QuerySpec
    resolved_task: ResolvedTask
    inputs: list[InputManifest]
    compatibility: CompatibilityReport
    plan: Plan
    executions: list[Execution]
    artifacts: list[ArtifactRef]
    fact_sheet: dict[str, float | str] = Field(
        default_factory=dict, description="Keys are namespaced '<tool>.<scalar>'."
    )
    answer: Answer
    confidence: Confidence
    warnings: list[WarningItem] = Field(default_factory=list)
    errors: list[ErrorItem] = Field(default_factory=list)
