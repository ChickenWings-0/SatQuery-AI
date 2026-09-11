"""Contract tests: every Phase 0 endpoint answers 200 with a schema-valid body.

These tests are the guard on the frozen contract. A failure here means either a
handler drifted from ``DOCS/API_CONTRACT.md`` or the contract was changed
without a version bump.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from satquery.api.app import TRACE_ID_HEADER, app
from satquery.api.dependencies import get_artifact_store, get_trace_store
from satquery.core.config import get_settings
from satquery.models import loader
from satquery.registry.registry import default_registry
from satquery.render.artifact_store import ArtifactStore
from satquery.schemas.api import (
    AnalyzeResponse,
    ApiErrorResponse,
    HealthResponse,
    RegistryResponse,
    ValidateResponse,
)
from satquery.schemas.compatibility import CompatibilityReport
from satquery.schemas.enums import (
    CheckStatus,
    HealthStatus,
    Overall,
    PairType,
    TaskType,
    ToolStatus,
)
from satquery.schemas.trace import AuditTrace
from satquery.schemas.version import SCHEMA_VERSION
from satquery.trace.store import TraceStore

AUDIT_TRACE_KEYS = [
    "trace_id",
    "schema_version",
    "created_at",
    "duration_ms",
    "query",
    "resolved_task",
    "inputs",
    "compatibility",
    "plan",
    "executions",
    "artifacts",
    "fact_sheet",
    "answer",
    "confidence",
    "warnings",
    "errors",
]


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def _image_parts(count: int = 2) -> list[tuple[str, tuple[str, bytes, str]]]:
    """Minimal multipart image parts — Phase 0 never opens the bytes."""
    return [
        ("images", (f"img_{i}.tif", b"\x49\x49\x2a\x00" + bytes(16), "image/tiff"))
        for i in range(count)
    ]


# ----------------------------------------------------------------------- health


def _fake_torch(device_count: int, total: int = 24 * 1024**3, free: int = 15 * 1024**3) -> object:
    """A torch stand-in for the device probe: no ROCm on CI, but the mapping is real."""
    return SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: device_count,
            mem_get_info=lambda device: (free, total),
            memory_allocated=lambda device: 0,
            get_device_name=lambda device: "AMD Radeon RX 7900 XTX",
        )
    )


def test_health_returns_200_and_parses(client: TestClient) -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200

    health = HealthResponse.model_validate(response.json())
    assert health.status is HealthStatus.OK
    assert health.schema_version == SCHEMA_VERSION
    assert health.tools_available <= health.tools_total


def test_health_reports_igpu_masking_from_the_environment(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A false here is a red flag before any demo (API_CONTRACT §4.9).

    Asserted as a *mapping* rather than as a constant: the old test pinned
    ``igpu_masked is True`` against a fixture, so it passed on a CPU-only
    runner with nothing masked at all — the check could not fail, which is
    exactly what a pre-demo assertion must not be.
    """
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(loader, "torch_cuda_available", lambda: True)
    fake_torch = _fake_torch(device_count=1)
    monkeypatch.setattr(loader, "_torch", lambda: fake_torch)

    health = HealthResponse.model_validate(client.get("/v1/health").json())
    assert health.device.igpu_masked is True
    assert health.device.hip_visible_devices == "0"


def test_health_reports_unmasked_igpu_when_a_second_device_is_visible(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The iGPU enumerating as a second HIP device is the failure being caught."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(loader, "torch_cuda_available", lambda: True)
    fake_torch = _fake_torch(device_count=2)
    monkeypatch.setattr(loader, "_torch", lambda: fake_torch)

    health = HealthResponse.model_validate(client.get("/v1/health").json())
    assert health.device.igpu_masked is False


def test_health_reports_unmasked_when_the_variable_is_unset(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unset HIP_VISIBLE_DEVICES means nothing was masked."""
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)

    health = HealthResponse.model_validate(client.get("/v1/health").json())
    assert health.device.igpu_masked is False
    assert health.device.hip_visible_devices is None


def test_every_response_carries_a_trace_id_header(client: TestClient) -> None:
    response = client.get("/v1/health")
    trace_id = response.headers[TRACE_ID_HEADER]
    assert len(trace_id) == 32
    assert int(trace_id, 16) >= 0


# --------------------------------------------------------------------- registry


def test_registry_returns_200_and_parses(client: TestClient) -> None:
    response = client.get("/v1/registry")
    assert response.status_code == 200

    registry = RegistryResponse.model_validate(response.json())
    assert registry.tools, "the registry must not be empty"
    assert len({tool.name for tool in registry.tools}) == len(registry.tools)
    assert TaskType.CHANGE_VQA in registry.task_types


def test_registry_exposes_unavailable_tools_with_a_reason(client: TestClient) -> None:
    registry = RegistryResponse.model_validate(client.get("/v1/registry").json())
    for tool in registry.tools:
        if not tool.available:
            assert tool.unavailable_reason


# --------------------------------------------------------------------- validate
#
# /v1/validate runs the real Phase 1 ingestion engine, so these post genuine
# rasters from the synthetic corpus rather than placeholder bytes.


def test_validate_returns_200_and_parses(client: TestClient, upload_files) -> None:
    response = client.post("/v1/validate", files=upload_files("s2_pre", "s2_post"))
    assert response.status_code == 200

    payload = response.json()
    report = ValidateResponse.model_validate(payload)
    assert [manifest.id for manifest in report.inputs] == ["img_0", "img_1"]
    assert report.supported_tasks

    # The nested compatibility object is itself a valid CompatibilityReport.
    compatibility = CompatibilityReport.model_validate(payload["compatibility"])
    assert compatibility.pair_type is PairType.BI_TEMPORAL
    assert compatibility.checks


def test_validate_reports_real_measurements(client: TestClient, upload_files) -> None:
    """The manifests must describe the uploaded files, not a fixture."""
    report = ValidateResponse.model_validate(
        client.post("/v1/validate", files=upload_files("s2_pre", "s2_post")).json()
    )
    first = report.inputs[0]
    assert first.crs == "EPSG:32643"
    assert first.gsd_m == pytest.approx(10.0)
    assert first.band_count == 12
    assert len(first.sha256) == 64
    assert report.inputs[0].sha256 != report.inputs[1].sha256
    assert report.compatibility.common_grid is not None


def test_validate_overall_matches_the_derivation_rule(client: TestClient, upload_files) -> None:
    """Any FAIL -> FAIL; else any WARN -> PASS_WITH_WARNINGS; else PASS (§2.7)."""
    report = ValidateResponse.model_validate(
        client.post("/v1/validate", files=upload_files("s2_pre", "s2_post")).json()
    )
    statuses = {check.status for check in report.compatibility.checks}
    if CheckStatus.FAIL in statuses:
        expected = Overall.FAIL
    elif CheckStatus.WARN in statuses:
        expected = Overall.PASS_WITH_WARNINGS
    else:
        expected = Overall.PASS
    assert report.compatibility.overall is expected


def test_validate_offers_no_tasks_for_an_unusable_pair(client: TestClient, upload_files) -> None:
    report = ValidateResponse.model_validate(
        client.post("/v1/validate", files=upload_files("s2_pre", "s2_disjoint")).json()
    )
    assert report.compatibility.overall is Overall.FAIL
    assert report.supported_tasks == []


def test_validate_accepts_a_single_image(client: TestClient, upload_files) -> None:
    report = ValidateResponse.model_validate(
        client.post("/v1/validate", files=upload_files("s2_pre")).json()
    )
    assert report.compatibility.pair_type is PairType.SINGLE
    assert TaskType.CAPTION in report.supported_tasks


def test_validate_honours_a_pair_type_hint(client: TestClient, upload_files) -> None:
    response = client.post(
        "/v1/validate",
        files=upload_files("s2_pre", "s1_vvvh"),
        data={"options": json.dumps({"pair_type_hint": "BI_TEMPORAL"})},
    )
    report = ValidateResponse.model_validate(response.json())
    assert report.compatibility.pair_type is PairType.BI_TEMPORAL
    assert report.compatibility.pair_type_source.value == "user_declared"


# ------------------------------------------------------------------- error taxonomy


def test_unreadable_file_is_a_400_in_the_frozen_envelope(client: TestClient) -> None:
    response = client.post(
        "/v1/validate", files=[("images", ("notes.tif", b"not a raster at all", "image/tiff"))]
    )
    assert response.status_code == 400

    body = ApiErrorResponse.model_validate(response.json())
    assert body.error.code == "INVALID_RASTER"
    assert body.error.http_status == 400
    assert body.error.hint


def test_too_many_images_is_a_400(client: TestClient, upload_files) -> None:
    response = client.post("/v1/validate", files=upload_files("s2_pre", "s2_post", "s2_shift3"))
    assert response.status_code == 400
    assert ApiErrorResponse.model_validate(response.json()).error.code == "TOO_MANY_IMAGES"


def test_mixed_georeferencing_is_a_422(client: TestClient, upload_files) -> None:
    response = client.post("/v1/validate", files=upload_files("s2_pre", "benchmark_rgb"))
    assert response.status_code == 422

    error = ApiErrorResponse.model_validate(response.json()).error
    assert error.code == "MISSING_GEOREFERENCE"
    assert error.ref == "img_1"


# ---------------------------------------------------------------------- analyze
#
# These exercise the real Phase 3 pipeline over the synthetic corpus. The Phase 0
# versions posted a few fake TIFF bytes, which the ingestion layer now correctly
# rejects — a nice demonstration that the mock was doing less than it looked like.


@pytest.fixture
def deterministic_only(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the analyze contract tests to the deterministic-only baseline.

    These tests assert what *our* handlers do — a skipped VLM step, a templated
    answer, a fully cited trace. Whether the VLM step runs is decided by
    ``catalog.vlm_servable()``, which probes the machine: on a box where the
    Qwen weights have since been downloaded, the same assertions start
    describing a live 8B generation instead, and the contract suite becomes both
    slow and nondeterministic. ``SATQUERY_VLM_DISABLED`` is the switch the
    settings already document for exactly this ("the deterministic-only
    baseline"), so the tests state their premise rather than inheriting it from
    whatever happens to be in the HF cache.

    Both caches are cleared on the way in and on the way out: the registry and
    the settings are process-wide singletons, so a test that changes the
    environment without clearing them changes nothing, and one that clears only
    on entry leaks the pin into every test that follows.
    """
    monkeypatch.setenv("SATQUERY_VLM_DISABLED", "true")
    get_settings.cache_clear()
    default_registry.cache_clear()
    yield
    get_settings.cache_clear()
    default_registry.cache_clear()


@pytest.fixture
def analyze_client(tmp_path: Path, deterministic_only: None) -> Iterator[TestClient]:
    """A client whose artifact and trace stores are scoped to one test."""
    artifacts = ArtifactStore(tmp_path / "artifacts")
    traces = TraceStore(tmp_path / "traces.sqlite3")
    app.dependency_overrides[get_artifact_store] = lambda: artifacts
    app.dependency_overrides[get_trace_store] = lambda: traces
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_artifact_store, None)
    app.dependency_overrides.pop(get_trace_store, None)


def test_analyze_returns_200_and_parses(analyze_client: TestClient, upload_files) -> None:
    response = analyze_client.post(
        "/v1/analyze",
        files=upload_files("s2_pre", "s2_post"),
        data={"query": "How much built-up area appeared?"},
    )
    assert response.status_code == 200

    result = AnalyzeResponse.model_validate(response.json())
    assert result.trace is not None
    assert result.trace_id == result.trace.trace_id
    assert result.answer.text
    assert result.artifacts, "the analysis must produce evidence, not just prose"


def test_analyze_trace_has_exactly_the_frozen_top_level_keys(
    analyze_client: TestClient, upload_files
) -> None:
    payload = analyze_client.post(
        "/v1/analyze",
        files=upload_files("s2_pre", "s2_post"),
        data={"query": "How much built-up area appeared?"},
    ).json()

    trace = payload["trace"]
    assert list(trace.keys()) == AUDIT_TRACE_KEYS
    AuditTrace.model_validate(trace)


def test_analyze_citations_resolve_into_the_executions(
    analyze_client: TestClient, upload_files
) -> None:
    """Every citation source must point at a scalar an execution actually emitted."""
    result = AnalyzeResponse.model_validate(
        analyze_client.post(
            "/v1/analyze",
            files=upload_files("s2_pre", "s2_post"),
            data={"query": "what changed between these two images"},
        ).json()
    )
    assert result.trace is not None
    scalars_by_step = {execution.step: execution.scalars for execution in result.trace.executions}

    assert result.answer.citations, "a change analysis must cite its measurements"
    for citation in result.answer.citations:
        step_part, _, scalar_part = citation.source.partition("/")
        step = int(step_part.removeprefix("step:"))
        for name in scalar_part.removeprefix("scalars.").split("|"):
            assert name in scalars_by_step[step], f"unresolved citation source: {citation.source}"


def test_analyze_never_leaves_a_number_unaccounted_for(
    analyze_client: TestClient, upload_files
) -> None:
    """A templated answer is fully grounded by construction: nothing may be uncited."""
    result = AnalyzeResponse.model_validate(
        analyze_client.post(
            "/v1/analyze",
            files=upload_files("s2_pre", "s2_post"),
            data={"query": "what changed between these two images"},
        ).json()
    )
    assert result.answer.template_fallback is True
    assert result.answer.uncited_numeric_spans == []


def test_analyze_records_a_tool_failure_as_200_not_500(
    analyze_client: TestClient, upload_files
) -> None:
    """API_CONTRACT §4.1: a tool-level failure never becomes a 5xx."""
    response = analyze_client.post(
        "/v1/analyze",
        files=upload_files("s2_pre", "s2_post"),
        data={"query": "what changed between these two images"},
    )
    assert response.status_code == 200

    result = AnalyzeResponse.model_validate(response.json())
    assert result.trace is not None
    statuses = {execution.status for execution in result.trace.executions}
    # The VLM is pinned off for this test (see `deterministic_only`), so its step
    # is skipped and the answer is templated — visibly, in the trace, at a
    # reduced confidence. That degraded mode is part of the contract, and it is
    # what a machine with no weights does on its own.
    assert ToolStatus.SKIPPED in statuses
    assert result.confidence.overall < 1.0


def test_analyze_persists_the_trace(analyze_client: TestClient, upload_files) -> None:
    result = AnalyzeResponse.model_validate(
        analyze_client.post(
            "/v1/analyze",
            files=upload_files("s2_pre"),
            data={"query": "describe this scene"},
        ).json()
    )
    stored = app.dependency_overrides[get_trace_store]().get(result.trace_id)
    assert stored is not None
    assert stored.trace_id == result.trace_id
    assert stored.plan.policy_key == "CAPTION|SINGLE|optical"


def test_analyze_echoes_the_query_and_honours_include_trace(
    analyze_client: TestClient, upload_files
) -> None:
    query = "What changed between these two images?"
    response = analyze_client.post(
        "/v1/analyze",
        files=upload_files("s2_pre", "s2_post"),
        data={"query": query, "options": json.dumps({"include_trace": False})},
    )
    result = AnalyzeResponse.model_validate(response.json())
    assert result.trace is None

    with_trace = AnalyzeResponse.model_validate(
        analyze_client.post(
            "/v1/analyze", files=upload_files("s2_pre", "s2_post"), data={"query": query}
        ).json()
    )
    assert with_trace.trace is not None
    assert with_trace.trace.query.raw == query


def test_analyze_refuses_incompatible_inputs_with_the_frozen_envelope(
    analyze_client: TestClient, upload_files
) -> None:
    response = analyze_client.post(
        "/v1/analyze",
        files=upload_files("s2_pre", "s2_disjoint"),
        data={"query": "what changed between these two images"},
    )
    assert response.status_code == 422

    error = ApiErrorResponse.model_validate(response.json()).error
    assert error.code == "INSUFFICIENT_OVERLAP"
    assert error.hint


def test_analyze_requires_a_query(analyze_client: TestClient, upload_files) -> None:
    response = analyze_client.post("/v1/analyze", files=upload_files("s2_pre"))
    assert response.status_code == 422


# ----------------------------------------------------------------------- schema


def test_openapi_document_is_generated(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    for path in ("/v1/health", "/v1/registry", "/v1/validate", "/v1/analyze"):
        assert path in schema["paths"], f"{path} missing from the OpenAPI document"
