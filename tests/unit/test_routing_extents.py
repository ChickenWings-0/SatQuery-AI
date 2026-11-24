"""Routing for the questions the console suggests, pinned.

Flood wording and land-cover "where/locate" questions are extents to outline;
objects are still boxed; a sensor comparison is not mistaken for grounding.
"""

from __future__ import annotations

import pytest

from satquery.agent.task_classifier import classify
from satquery.schemas.enums import PairType, TaskType

CASES = [
    ("Find floods in my area", PairType.SINGLE, TaskType.SEGMENTATION),
    ("Locate built-up areas", PairType.SINGLE, TaskType.SEGMENTATION),
    ("Where is the water?", PairType.SINGLE, TaskType.SEGMENTATION),
    ("Show me the forest", PairType.SINGLE, TaskType.SEGMENTATION),
    ("Outline the water bodies", PairType.SINGLE, TaskType.SEGMENTATION),
    ("Locate the aircraft", PairType.SINGLE, TaskType.GROUNDING),
    ("Find the airport", PairType.SINGLE, TaskType.GROUNDING),
    ("Where is the bridge?", PairType.SINGLE, TaskType.GROUNDING),
    ("Where do optical and SAR disagree?", PairType.CROSS_MODAL, TaskType.CROSS_MODAL_COMPARE),
    ("How many aircraft are parked?", PairType.SINGLE, TaskType.COUNT),
    ("Detect new constructions", PairType.BI_TEMPORAL, TaskType.CHANGE_VQA),
]


@pytest.mark.parametrize(("query", "pair", "task"), CASES)
def test_suggested_questions_route_to_the_tool_that_answers_them(
    query: str, pair: PairType, task: TaskType
) -> None:
    assert classify(query, pair).primary is task
