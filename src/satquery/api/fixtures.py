"""Phase 0 mock fixtures.

Every endpoint returns schema-valid data from here so the frontend can build a
typed client and start integrating before any model is loaded. The bi-temporal
built-up scenario is the canonical worked example of API_CONTRACT §7.4.

Nothing in this module touches a GPU, the filesystem, or the network. It is
replaced tool-by-tool as the real pipeline lands in later phases.
"""

from __future__ import annotations

from datetime import UTC, datetime

from satquery.schemas.api import (
    AnalyzeResponse,
    DeviceInfo,
    HealthResponse,
    ModelInfo,
    RegistryResponse,
    ValidateResponse,
)
from satquery.schemas.compatibility import CheckResult, CommonGrid, CompatibilityReport
from satquery.schemas.enums import (
    ArtifactType,
    CheckName,
    CheckStatus,
    Device,
    HealthStatus,
    ImageRole,
    Modality,
    Overall,
    PairType,
    PairTypeSource,
    Resampling,
    TaskType,
    ToolCategory,
    ToolStatus,
)
from satquery.schemas.manifest import InputManifest
from satquery.schemas.tool import Execution, InputContract, PlanStep, ToolSpec
from satquery.schemas.trace import (
    Answer,
    ArtifactGeo,
    ArtifactRef,
    AuditTrace,
    Citation,
    Confidence,
    Plan,
    QuerySpec,
    ResolvedTask,
    WarningItem,
)

MOCK_TRACE_ID = "b3f1c7d2e4a5486f9b0c1d2e3f4a5b6c"
MOCK_CREATED_AT = datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC)
REGISTRY_VERSION = "1.0.0"
ADAPTER_VERSION = "sq-lora-v3"

_UTM43N = "EPSG:32643"
_TRANSFORM = [10.0, 0.0, 712_340.0, 0.0, -10.0, 3_161_780.0]
_BOUNDS_NATIVE = [712_340.0, 3_151_540.0, 722_580.0, 3_161_780.0]
_BOUNDS_WGS84 = [77.0912, 28.4981, 77.2961, 28.6905]


# --------------------------------------------------------------------------- inputs


def mock_inputs() -> list[InputManifest]:
    """A co-registered pre/post Sentinel-2 pair over Delhi NCR."""
    common = {
        "driver": "GTiff",
        "modality": Modality.OPTICAL,
        "modality_confidence": 0.98,
        "sensor_guess": "Sentinel-2 L2A",
        "crs": _UTM43N,
        "transform": _TRANSFORM,
        "bounds_native": _BOUNDS_NATIVE,
        "bounds_wgs84": _BOUNDS_WGS84,
        "gsd_m": 10.0,
        "width": 1024,
        "height": 1024,
        "band_count": 12,
        "dtype": "uint16",
        "band_names": ["B02", "B03", "B04", "B08", "B11", "B12"],
        "is_georeferenced": True,
    }
    return [
        InputManifest(
            id="img_0",
            role=ImageRole.PRE,
            filename="delhi_2019_04_12.tif",
            sha256="9a3f" + "0" * 60,
            size_bytes=24_117_248,
            nodata_pct=0.4,
            acquisition_time=datetime(2019, 4, 12, 5, 26, 41, tzinfo=UTC),
            **common,
        ),
        InputManifest(
            id="img_1",
            role=ImageRole.POST,
            filename="delhi_2024_04_08.tif",
            sha256="7c21" + "0" * 60,
            size_bytes=24_117_248,
            nodata_pct=1.2,
            acquisition_time=datetime(2024, 4, 8, 5, 26, 33, tzinfo=UTC),
            **common,
        ),
    ]


# ------------------------------------------------------------------- compatibility


def mock_compatibility() -> CompatibilityReport:
    """A clean bi-temporal pair: everything passes, one sub-pixel coregistration nudge."""
    return CompatibilityReport(
        pair_type=PairType.BI_TEMPORAL,
        pair_type_source=PairTypeSource.METADATA,
        checks=[
            CheckResult(
                name=CheckName.CRS_MATCH,
                status=CheckStatus.PASS,
                value=_UTM43N,
                threshold="identical CRS",
                detail="Both images are in EPSG:32643; no reprojection was needed.",
            ),
            CheckResult(
                name=CheckName.BOUNDS_OVERLAP_IOU,
                status=CheckStatus.PASS,
                value=0.994,
                threshold="iou >= 0.80",
                detail="The two footprints overlap almost exactly.",
            ),
            CheckResult(
                name=CheckName.GSD_RATIO,
                status=CheckStatus.PASS,
                value=1.0,
                threshold="ratio <= 1.5",
                detail="Both images have a 10 m ground sample distance.",
            ),
            CheckResult(
                name=CheckName.COREGISTRATION_OFFSET_PX,
                status=CheckStatus.PASS,
                value=0.31,
                threshold="offset <= 1.0 px",
                detail="Phase-correlation found a 0.31 px shift, corrected before analysis.",
            ),
            CheckResult(
                name=CheckName.BAND_SUFFICIENCY,
                status=CheckStatus.PASS,
                value="red,nir,swir1",
                threshold="required bands present",
                detail="All bands required for NDBI are present in both images.",
            ),
            CheckResult(
                name=CheckName.MODALITY_DISTINCT,
                status=CheckStatus.SKIP,
                value=None,
                threshold="CROSS_MODAL only",
                detail="Not applicable to a bi-temporal pair.",
            ),
            CheckResult(
                name=CheckName.TEMPORAL_ORDERING,
                status=CheckStatus.PASS,
                value="2019-04-12 -> 2024-04-08",
                threshold="timestamps present and ordered",
                detail="Acquisition times were read from metadata and are correctly ordered.",
            ),
            CheckResult(
                name=CheckName.GEOREFERENCE_PRESENT,
                status=CheckStatus.PASS,
                value="both",
                threshold="both georeferenced",
                detail="Both images carry a CRS and an affine transform.",
            ),
            CheckResult(
                name=CheckName.SIZE_RATIO,
                status=CheckStatus.PASS,
                value=1.0,
                threshold="ratio <= 1.5",
                detail="Both images are 1024 x 1024 pixels.",
            ),
            CheckResult(
                name=CheckName.NODATA_EXTENT,
                status=CheckStatus.PASS,
                value=1.2,
                threshold="nodata < 5 %",
                detail="At most 1.2 % of either scene is nodata.",
            ),
        ],
        overall=Overall.PASS,
        actions_taken=["coregistered img_1 to img_0 (shift 0.31 px, bilinear)"],
        common_grid=CommonGrid(
            crs=_UTM43N,
            transform=_TRANSFORM,
            width=1024,
            height=1024,
            gsd_m=10.0,
            resampling=Resampling.BILINEAR,
        ),
    )


# ----------------------------------------------------------------------- registry


def mock_registry() -> RegistryResponse:
    """A representative slice of the tool registry, including one unavailable tool."""
    tools = [
        ToolSpec(
            name="spectral_renderer",
            version="1.0.0",
            category=ToolCategory.GEO,
            description=(
                "Renders named 3-channel views (true colour, false colour IR, NDWI) "
                "for the VLM and the UI."
            ),
            accepts=InputContract(
                pair_types=[PairType.SINGLE, PairType.BI_TEMPORAL, PairType.CROSS_MODAL],
                modalities=[Modality.OPTICAL, Modality.SAR, Modality.PANCHROMATIC],
                min_images=1,
                max_images=2,
                required_bands=[],
                min_size_px=64,
                max_size_px=20_000,
                requires_georeference=False,
            ),
            produces=[ArtifactType.RENDERED_VIEW],
            scalars_schema={"type": "object", "properties": {}},
            device=Device.CPU,
            est_ms=420,
            fallback=None,
            available=True,
        ),
        ToolSpec(
            name="siamese_change_detector",
            version="1.0.0+ckpt:9a3f",
            category=ToolCategory.CV,
            description="Detects per-pixel change between a co-registered bi-temporal pair.",
            accepts=InputContract(
                pair_types=[PairType.BI_TEMPORAL],
                modalities=[Modality.OPTICAL],
                min_images=2,
                max_images=2,
                required_bands=["red", "green", "blue"],
                optional_bands=["nir"],
                gsd_range_m=[0.3, 30.0],
                min_size_px=256,
                max_size_px=20_000,
                requires_georeference=True,
            ),
            produces=[ArtifactType.CHANGE_MASK, ArtifactType.OVERLAY_PNG],
            scalars_schema={
                "type": "object",
                "properties": {"changed_pixel_count": {"type": "integer"}},
            },
            device=Device.ROCM_0,
            est_ms=1_800,
            fallback="spectral_index_analyzer",
            available=True,
        ),
        ToolSpec(
            name="change_statistics",
            version="1.0.0",
            category=ToolCategory.ANALYSIS,
            description="Converts a change mask into georeferenced areas and percentages.",
            accepts=InputContract(
                pair_types=[PairType.BI_TEMPORAL],
                modalities=[Modality.OPTICAL, Modality.SAR],
                min_images=2,
                max_images=2,
                requires_georeference=True,
            ),
            produces=[ArtifactType.SCALARS],
            scalars_schema={
                "type": "object",
                "properties": {
                    "changed_area_pct": {"type": "number"},
                    "changed_area_km2": {"type": "number"},
                },
            },
            device=Device.CPU,
            est_ms=90,
            fallback=None,
            available=True,
        ),
        ToolSpec(
            name="spectral_index_analyzer",
            version="1.0.0",
            category=ToolCategory.ANALYSIS,
            description="Computes NDVI, NDWI and NDBI statistics per image.",
            accepts=InputContract(
                pair_types=[PairType.SINGLE, PairType.BI_TEMPORAL, PairType.CROSS_MODAL],
                modalities=[Modality.OPTICAL],
                min_images=1,
                max_images=2,
                required_bands=["red", "nir"],
                optional_bands=["swir1"],
                requires_georeference=False,
            ),
            produces=[ArtifactType.SCALARS, ArtifactType.HEATMAP],
            scalars_schema={
                "type": "object",
                "properties": {"ndbi_mean_pre": {"type": "number"}},
            },
            device=Device.CPU,
            est_ms=260,
            fallback=None,
            available=True,
        ),
        ToolSpec(
            name="vlm_change_vqa",
            version="1.0.0+adapter:sq-lora-v3",
            category=ToolCategory.VLM,
            description="Answers bi-temporal questions from rendered views and the fact sheet.",
            accepts=InputContract(
                pair_types=[PairType.BI_TEMPORAL],
                modalities=[Modality.OPTICAL, Modality.SAR],
                min_images=2,
                max_images=2,
                requires_georeference=False,
            ),
            produces=[ArtifactType.TEXT],
            scalars_schema={"type": "object", "properties": {}},
            device=Device.ROCM_0,
            est_ms=4_200,
            fallback="template_answer_writer",
            available=True,
        ),
        ToolSpec(
            name="physics_agreement",
            version="1.0.0",
            category=ToolCategory.FUSION,
            description="Applies deterministic optical/SAR physical consistency rules.",
            accepts=InputContract(
                pair_types=[PairType.CROSS_MODAL],
                modalities=[Modality.OPTICAL, Modality.SAR],
                min_images=2,
                max_images=2,
                requires_georeference=False,
            ),
            produces=[ArtifactType.SCALARS, ArtifactType.HEATMAP],
            scalars_schema={"type": "object", "properties": {}},
            device=Device.CPU,
            est_ms=150,
            fallback=None,
            available=False,
            unavailable_reason="Rule table ships in Phase 6.",
        ),
    ]
    return RegistryResponse(
        registry_version=REGISTRY_VERSION,
        tools=tools,
        task_types=list(TaskType),
        adapter_version=ADAPTER_VERSION,
    )


# ----------------------------------------------------------------------- validate


def mock_validate() -> ValidateResponse:
    """Pre-flight result for the canonical bi-temporal pair."""
    return ValidateResponse(
        inputs=mock_inputs(),
        compatibility=mock_compatibility(),
        supported_tasks=[TaskType.CHANGE_VQA, TaskType.CHANGE_CAPTION, TaskType.CHANGE_MAP],
        warnings=[
            WarningItem(
                code="MOCK_RESPONSE",
                message="Phase 0 stub: no raster was opened and no check was actually run.",
                ref=None,
            )
        ],
    )


# ------------------------------------------------------------------------- trace


def _mock_plan() -> Plan:
    return Plan(
        planner="policy_table_v1",
        policy_key="CHANGE_VQA|BI_TEMPORAL|optical",
        steps=[
            PlanStep(
                step=1,
                tool="spectral_renderer",
                depends_on=[],
                input_refs=["img_0", "img_1"],
                reason="policy_table:CHANGE_VQA|BI_TEMPORAL|optical",
            ),
            PlanStep(
                step=2,
                tool="siamese_change_detector",
                depends_on=[1],
                input_refs=["img_0", "img_1"],
                reason="policy_table:CHANGE_VQA|BI_TEMPORAL|optical",
            ),
            PlanStep(
                step=3,
                tool="spectral_index_analyzer",
                depends_on=[1],
                input_refs=["img_0", "img_1"],
                reason="built-up query slot requires NDBI evidence",
            ),
            PlanStep(
                step=4,
                tool="vlm_change_vqa",
                depends_on=[2, 3],
                input_refs=["art_0", "art_1", "art_2"],
                reason="task CHANGE_VQA requires natural-language synthesis",
            ),
        ],
    )


def _mock_executions() -> list[Execution]:
    return [
        Execution(
            step=1,
            tool="spectral_renderer",
            version="1.0.0",
            status=ToolStatus.OK,
            device_used=Device.CPU,
            duration_ms=412,
            params={"views": ["true_colour", "false_colour_ir"], "stretch": "percentile_2_98"},
            input_refs=["img_0", "img_1"],
            output_refs=["art_0", "art_1"],
            scalars={},
            confidence=0.99,
            cache_hit=False,
        ),
        Execution(
            step=2,
            tool="siamese_change_detector",
            version="1.0.0+ckpt:9a3f",
            status=ToolStatus.OK,
            device_used=Device.ROCM_0,
            duration_ms=1_830,
            params={"threshold": 0.5, "tile_px": 512, "overlap_px": 64, "seed": 0},
            input_refs=["img_0", "img_1"],
            output_refs=["art_2", "art_3"],
            scalars={
                "changed_area_pct": 7.4,
                "changed_area_km2": 1.07,
                "changed_pixel_count": 10_700,
            },
            confidence=0.88,
            cache_hit=False,
        ),
        Execution(
            step=3,
            tool="spectral_index_analyzer",
            version="1.0.0",
            status=ToolStatus.OK,
            device_used=Device.CPU,
            duration_ms=268,
            params={"indices": ["ndbi", "ndvi"]},
            input_refs=["img_0", "img_1"],
            output_refs=["art_4"],
            scalars={"ndbi_mean_pre": -0.14, "ndbi_mean_post": 0.03},
            confidence=0.95,
            cache_hit=False,
        ),
        Execution(
            step=4,
            tool="vlm_change_vqa",
            version="1.0.0+adapter:sq-lora-v3",
            status=ToolStatus.OK,
            device_used=Device.ROCM_0,
            duration_ms=4_180,
            params={"max_new_tokens": 256, "temperature": 0.2, "seed": 0},
            input_refs=["art_0", "art_1", "art_2"],
            output_refs=["art_5"],
            scalars={},
            confidence=0.84,
            cache_hit=False,
        ),
    ]


def _mock_artifacts() -> list[ArtifactRef]:
    base = f"/v1/artifacts/{MOCK_TRACE_ID}"
    geo = ArtifactGeo(crs=_UTM43N, transform=_TRANSFORM)
    return [
        ArtifactRef(
            id="art_0",
            type=ArtifactType.RENDERED_VIEW,
            mime="image/png",
            label="Image 1 (2019, true colour)",
            url=f"{base}/art_0.png",
            geo=geo,
            width=1024,
            height=1024,
            produced_by_step=1,
        ),
        ArtifactRef(
            id="art_1",
            type=ArtifactType.RENDERED_VIEW,
            mime="image/png",
            label="Image 2 (2024, true colour)",
            url=f"{base}/art_1.png",
            geo=geo,
            width=1024,
            height=1024,
            produced_by_step=1,
        ),
        ArtifactRef(
            id="art_2",
            type=ArtifactType.CHANGE_MASK,
            mime="image/png",
            label="Change mask (binary)",
            url=f"{base}/art_2.png",
            geotiff_url=f"{base}/art_2.tif",
            geo=geo,
            width=1024,
            height=1024,
            stats={"changed_pixel_count": 10_700, "total_pixel_count": 1_048_576},
            produced_by_step=2,
        ),
        ArtifactRef(
            id="art_3",
            type=ArtifactType.OVERLAY_PNG,
            mime="image/png",
            label="Change overlaid on the 2024 view",
            url=f"{base}/art_3.png",
            geo=geo,
            width=1024,
            height=1024,
            produced_by_step=2,
        ),
        ArtifactRef(
            id="art_4",
            type=ArtifactType.SCALARS,
            mime="application/json",
            label="NDBI statistics (pre / post)",
            url=None,
            inline={"ndbi_mean_pre": -0.14, "ndbi_mean_post": 0.03, "delta": 0.17},
            produced_by_step=3,
        ),
        ArtifactRef(
            id="art_5",
            type=ArtifactType.TEXT,
            mime="application/json",
            label="Answer text",
            url=None,
            inline={"text": ANSWER_TEXT},
            produced_by_step=4,
        ),
    ]


ANSWER_TEXT = (
    "Approximately 7.4% of the scene (1.07 km2) transitioned to built-up between "
    "April 2019 and April 2024, concentrated along the northern edge of the "
    "existing urban core. The mean NDBI rose from -0.14 to 0.03 over the same "
    "period, which is consistent with new impervious surface rather than seasonal "
    "vegetation loss."
)


def mock_trace() -> AuditTrace:
    """Return the canonical bi-temporal trace: "How much built-up area appeared?"."""
    return AuditTrace(
        trace_id=MOCK_TRACE_ID,
        created_at=MOCK_CREATED_AT,
        duration_ms=6_690,
        query=QuerySpec(
            raw="How much built-up area appeared?",
            normalized="how much built-up area appeared between the two acquisitions",
            language="en",
        ),
        resolved_task=ResolvedTask(
            primary=TaskType.CHANGE_VQA,
            secondary=[TaskType.CHANGE_MAP],
            slots={"target_class": "built_up", "quantity": "area", "unit": "pct"},
            confidence=0.94,
            classifier="rules_v1+llm_slotfill_v1",
        ),
        inputs=mock_inputs(),
        compatibility=mock_compatibility(),
        plan=_mock_plan(),
        executions=_mock_executions(),
        artifacts=_mock_artifacts(),
        fact_sheet={
            "change_statistics.changed_area_pct": 7.4,
            "change_statistics.changed_area_km2": 1.07,
            "change_statistics.changed_pixel_count": 10_700,
            "spectral_index_analyzer.ndbi_mean_pre": -0.14,
            "spectral_index_analyzer.ndbi_mean_post": 0.03,
            "siamese_change_detector.checkpoint": "9a3f",
        },
        answer=Answer(
            text=ANSWER_TEXT,
            citations=[
                Citation(
                    claim="7.4% of the scene",
                    source="step:2/scalars.changed_area_pct",
                    value=7.4,
                ),
                Citation(
                    claim="1.07 km2",
                    source="step:2/scalars.changed_area_km2",
                    value=1.07,
                ),
                Citation(
                    claim="mean NDBI rose from -0.14 to 0.03",
                    source="step:3/scalars.ndbi_mean_pre|ndbi_mean_post",
                    value="-0.14 -> 0.03",
                ),
            ],
            uncited_numeric_spans=[],
            generator="vlm_change_vqa@1.0.0+adapter:sq-lora-v3",
            template_fallback=False,
        ),
        confidence=Confidence(
            overall=0.86,
            method="weighted_tool_agreement_v1",
            components={
                "task_classification": 0.94,
                "input_quality": 0.97,
                "tool_mean": 0.88,
                "cross_tool_agreement": 0.81,
            },
            caps_applied=[],
        ),
        warnings=[
            WarningItem(
                code="MOCK_RESPONSE",
                message="Phase 0 stub: this trace is a fixture, no tool was executed.",
                ref=None,
            )
        ],
        errors=[],
    )


def mock_analyze(query: str | None = None, include_trace: bool = True) -> AnalyzeResponse:
    """Build the mocked ``/v1/analyze`` body, echoing the caller's query into the trace."""
    trace = mock_trace()
    if query:
        trace.query = QuerySpec(raw=query, normalized=query.strip().lower(), language="en")
    return AnalyzeResponse(
        trace_id=trace.trace_id,
        answer=trace.answer,
        artifacts=trace.artifacts,
        confidence=trace.confidence,
        compatibility=trace.compatibility,
        resolved_task=trace.resolved_task,
        trace=trace if include_trace else None,
    )


# -------------------------------------------------------------------------- health


def mock_health(version: str, schema_version: str) -> HealthResponse:
    """Health payload with the iGPU masking assertion the demo depends on."""
    registry = mock_registry()
    return HealthResponse(
        status=HealthStatus.OK,
        version=version,
        schema_version=schema_version,
        device=DeviceInfo(
            backend="rocm",
            name="AMD Radeon RX 7900 XTX",
            hip_visible_devices="0",
            vram_total_mb=24_576,
            vram_used_mb=9_210,
            igpu_masked=True,
        ),
        models=[
            ModelInfo(name="qwen3-vl-8b", backend="hf", loaded=True, adapter=ADAPTER_VERSION),
        ],
        tools_available=sum(1 for tool in registry.tools if tool.available),
        tools_total=len(registry.tools),
    )
