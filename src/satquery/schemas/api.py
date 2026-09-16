"""Endpoint request/response envelopes (API_CONTRACT §4, §6)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from satquery.schemas.compatibility import CompatibilityReport
from satquery.schemas.enums import HealthStatus, PairType, TaskType
from satquery.schemas.manifest import InputManifest
from satquery.schemas.tool import ToolSpec
from satquery.schemas.trace import (
    Answer,
    ArtifactRef,
    AuditTrace,
    Confidence,
    ResolvedTask,
    WarningItem,
)
from satquery.schemas.version import SCHEMA_VERSION


class AnalyzeOptions(BaseModel):
    """Parsed form of the ``options`` multipart part (API_CONTRACT §4.1)."""

    model_config = ConfigDict(extra="forbid")

    pair_type_hint: str | None = Field(
        default=None,
        description="Forces PairType; sets pair_type_source='user_declared'. "
        "Must be one of the PairType values.",
    )
    roles: dict[str, str] | None = Field(
        default=None, description="filename -> ImageRole override."
    )
    include_trace: bool = True
    include_rendered_views: bool = True
    artifact_format: str = Field(
        default="png+geotiff",
        description="'png' | 'png+geotiff'. Accepted for forward compatibility; the "
        "server currently writes both and reports OPTION_IGNORED if this is changed.",
    )
    vlm_backend: str = Field(
        default="auto",
        description="'auto' | 'hf' | 'llamacpp' | 'none'. Accepted for forward "
        "compatibility; the backend is chosen per process from SATQUERY_VLM_BACKEND "
        "and a non-default value here reports OPTION_IGNORED.",
    )
    adapter_version: str | None = Field(
        default=None,
        description="None = server default. Accepted for forward compatibility; the "
        "adapter is chosen per process and a value here reports OPTION_IGNORED.",
    )
    enable_tools: list[str] | None = Field(default=None, description="Allow-list; None = all.")
    disable_tools: list[str] = Field(default_factory=list)
    max_latency_ms: int = Field(
        default=30_000,
        gt=0,
        description="Soft budget for the tool DAG. Overrunning it does not cancel "
        "anything; the response carries a LATENCY_BUDGET_EXCEEDED warning.",
    )
    seed: int = Field(default=0, description="Determinism; 0 = server default.")

    @field_validator("pair_type_hint")
    @classmethod
    def _known_pair_type(cls, value: str | None) -> str | None:
        """Refuse a hint that is not a PairType, here, as a 400 — not as a 500 later."""
        if value is None:
            return None
        try:
            PairType(value)
        except ValueError as error:
            raise ValueError(
                f"pair_type_hint must be one of {[member.value for member in PairType]}"
            ) from error
        return value

    @property
    def pair_type(self) -> PairType | None:
        """The hint as the enum the pipeline consumes."""
        return PairType(self.pair_type_hint) if self.pair_type_hint else None

    def ignored(self) -> list[str]:
        """The options this server accepts but does not act on, when set off-default."""
        names: list[str] = []
        if self.artifact_format != "png+geotiff":
            names.append("artifact_format")
        if self.vlm_backend != "auto":
            names.append("vlm_backend")
        if self.adapter_version is not None:
            names.append("adapter_version")
        return names


class AnalyzeResponse(BaseModel):
    """``POST /v1/analyze`` 200 body.

    ``compatibility`` and ``resolved_task`` are duplicated at the top level so the
    UI can render them without walking into the trace.
    """

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")
    answer: Answer
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    confidence: Confidence
    compatibility: CompatibilityReport
    resolved_task: ResolvedTask
    trace: AuditTrace | None = Field(
        default=None, description="None when options.include_trace is false."
    )


class JobAccepted(BaseModel):
    """``POST /v1/jobs`` 202 body (API_CONTRACT §4.2).

    ``job_id`` *is* the eventual ``trace_id``: the id is minted before the
    pipeline runs, so a client can address the trace before it exists.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")
    status: Literal["queued"] = "queued"
    poll_url: str = Field(description="e.g. '/v1/jobs/b3f1...'.")
    events_url: str = Field(description="SSE endpoint, e.g. '/v1/jobs/b3f1.../events'.")


class JobStatusResponse(BaseModel):
    """``GET /v1/jobs/{job_id}`` 200 body (API_CONTRACT §4.3).

    ``result`` is populated only once ``status`` is ``succeeded``, ``error``
    only once it is ``failed``; they are mutually exclusive, mirroring the
    ``done``/``error`` terminal SSE events.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")
    status: Literal["queued", "running", "succeeded", "failed"]
    stage: str = Field(
        description="queued|ingesting|validating|rendering|planning|executing|aggregating|done."
    )
    step: int = Field(ge=0)
    total_steps: int = Field(ge=0)
    pct: int = Field(ge=0, le=100)
    created_at: datetime
    updated_at: datetime
    result: AnalyzeResponse | None = None
    error: ApiError | None = None


class ValidateResponse(BaseModel):
    """``POST /v1/validate`` 200 body (API_CONTRACT §4.5). No tools run, no GPU touched."""

    model_config = ConfigDict(extra="forbid")

    inputs: list[InputManifest]
    compatibility: CompatibilityReport
    supported_tasks: list[TaskType] = Field(
        description=(
            "Capability matching over the registry — lets the UI grey out impossible questions."
        )
    )
    warnings: list[WarningItem] = Field(default_factory=list)


class ImageryItemRequest(BaseModel):
    """One STAC item to fetch — the server resolves its assets itself."""

    model_config = ConfigDict(extra="forbid")

    collection: str = Field(
        description="'sentinel-2-l2a' | 'sentinel-1-rtc' | 'sentinel-1-grd'.", max_length=64
    )
    id: str = Field(description="The STAC item id.", min_length=1, max_length=200)


class ImageryFetchRequest(BaseModel):
    """``POST /v1/imagery/fetch`` body (API_CONTRACT §4.11, additive)."""

    model_config = ConfigDict(extra="forbid")

    items: list[ImageryItemRequest] = Field(min_length=1, max_length=2)
    bbox: list[float] = Field(
        min_length=4,
        max_length=4,
        description="[min_lon, min_lat, max_lon, max_lat], EPSG:4326. Clipped to the item.",
    )
    max_px: int | None = Field(
        default=None,
        ge=64,
        le=4096,
        description="Longest side in pixels; capped by the server's own ceiling.",
    )


class ImageryFile(BaseModel):
    """One fetched GeoTIFF, ready to be uploaded to ``/v1/validate``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Filename to upload under; carries the item id.")
    url: str = Field(description="``GET`` here for the bytes; same origin as the API.")
    size_bytes: int = Field(ge=0)
    collection: str
    item_id: str
    datetime: str = Field(description="The item's acquisition time, ISO 8601.")
    crs: str
    gsd_m: float = Field(gt=0.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    band_count: int = Field(gt=0)
    bounds_wgs84: list[float] = Field(min_length=4, max_length=4)
    orbit_state: str | None = Field(default=None, description="From ``sat:orbit_state``.")


class ImageryFetchResponse(BaseModel):
    """``POST /v1/imagery/fetch`` 200 body."""

    model_config = ConfigDict(extra="forbid")

    fetch_id: str = Field(description="Groups the files; part of every ``url``.")
    files: list[ImageryFile]
    warnings: list[WarningItem] = Field(default_factory=list)


class RegistryResponse(BaseModel):
    """``GET /v1/registry`` 200 body (API_CONTRACT §4.8)."""

    model_config = ConfigDict(extra="forbid")

    registry_version: str
    tools: list[ToolSpec]
    task_types: list[TaskType]
    adapter_version: str | None = None


class DeviceInfo(BaseModel):
    """Accelerator state, as reported by the ROCm runtime."""

    model_config = ConfigDict(extra="forbid")

    backend: str = Field(description="'rocm' | 'cpu'.")
    name: str
    hip_visible_devices: str | None = None
    vram_total_mb: int | None = None
    vram_used_mb: int | None = None
    igpu_masked: bool = Field(
        description="Asserts HIP_VISIBLE_DEVICES=0 took effect. False is a red flag before a demo."
    )


class ModelInfo(BaseModel):
    """One loadable model and its adapter."""

    model_config = ConfigDict(extra="forbid")

    name: str
    backend: str = Field(description="'hf' | 'llamacpp'.")
    loaded: bool
    adapter: str | None = None


class HealthResponse(BaseModel):
    """``GET /v1/health`` 200 body (API_CONTRACT §4.9)."""

    model_config = ConfigDict(extra="forbid")

    status: HealthStatus
    version: str
    schema_version: str = SCHEMA_VERSION
    device: DeviceInfo
    models: list[ModelInfo] = Field(default_factory=list)
    tools_available: int = Field(ge=0)
    tools_total: int = Field(ge=0)
    jobs_running: int = Field(
        default=0, ge=0, description="Async jobs that have not reached a terminal event."
    )
    jobs_retained: int = Field(
        default=0, ge=0, description="Jobs the in-memory store still holds, running or finished."
    )
    leaked_gpu_permits: int = Field(
        default=0,
        ge=0,
        description="GPU tool threads that overran their timeout and are still running. "
        "Non-zero means the card is busy with work nobody is waiting for; past the "
        "process's limit, status is 'degraded'.",
    )


class ApiError(BaseModel):
    """The error payload (API_CONTRACT §6).

    ``message`` and ``hint`` are user-facing and safe to display verbatim: never
    leak file paths or stack traces into either.
    """

    model_config = ConfigDict(extra="forbid")

    code: str
    http_status: int = Field(ge=400, le=599)
    message: str
    hint: str | None = Field(default=None, description="An actionable next step.")
    ref: str | None = None
    trace_id: str | None = None


class ApiErrorResponse(BaseModel):
    """Wire envelope for :class:`ApiError` — every non-2xx body has this shape."""

    model_config = ConfigDict(extra="forbid")

    error: ApiError
