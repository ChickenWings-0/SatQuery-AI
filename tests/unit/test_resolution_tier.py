"""Medium-resolution inputs get the change plan that works at their scale."""

from __future__ import annotations

import pytest

from satquery.agent.planner import default_table, resolution_tier, resolve_key
from satquery.schemas.enums import PairType, TaskType
from satquery.schemas.manifest import InputManifest


def _m(gsd: float | None) -> InputManifest:
    return InputManifest.model_construct(gsd_m=gsd)


@pytest.fixture(autouse=True)
def tiers_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SATQUERY_RESOLUTION_TIERS", "true")


def test_tier_needs_every_input_coarse_and_known() -> None:
    assert resolution_tier([_m(10.0), _m(10.0)]) == "medium_res"
    assert resolution_tier([_m(10.0), _m(0.5)]) is None
    assert resolution_tier([_m(10.0), _m(None)]) is None
    assert resolution_tier([]) is None


def test_tier_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SATQUERY_RESOLUTION_TIERS", "false")
    assert resolution_tier([_m(10.0)]) is None


CHANGE_TASKS = [TaskType.CHANGE_VQA, TaskType.CHANGE_CAPTION, TaskType.CHANGE_MAP]


@pytest.mark.parametrize("task", CHANGE_TASKS)
@pytest.mark.parametrize("modality", ["optical", "sar"])
def test_medium_res_change_plans_use_spectral_differencing(task: TaskType, modality: str) -> None:
    key, entry, _ = resolve_key(default_table(), task, PairType.BI_TEMPORAL, modality, "medium_res")
    assert key.endswith(":medium_res")
    tools = [step.tool for step in entry.steps]
    assert "image_diff_change" in tools
    assert "siamese_change_detector" not in tools


def test_aerial_pairs_keep_the_trained_detector() -> None:
    key, entry, _ = resolve_key(
        default_table(), TaskType.CHANGE_VQA, PairType.BI_TEMPORAL, "optical", None
    )
    assert key == "CHANGE_VQA|BI_TEMPORAL|optical"
    assert "siamese_change_detector" in [step.tool for step in entry.steps]
