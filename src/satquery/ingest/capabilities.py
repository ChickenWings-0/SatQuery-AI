"""Which tasks the uploaded images can support (API_CONTRACT §2.1, §4.5).

``/v1/validate`` returns this so the UI can grey out impossible questions before
the user types one. It is the input-shaped half of capability matching; the
tool-shaped half (``AGENT_POLICY_DAG.md`` §5) arrives with the registry in a
later phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from satquery.schemas.enums import Modality, Overall, PairType, TaskType


@dataclass(frozen=True)
class TaskRequirement:
    """The input conditions one task needs, straight from the contract's table."""

    min_images: int
    pair_types: frozenset[PairType]
    modalities: frozenset[Modality] | None = None


REQUIREMENTS: Final[dict[TaskType, TaskRequirement]] = {
    TaskType.VQA: TaskRequirement(1, frozenset({PairType.SINGLE, PairType.CROSS_MODAL})),
    TaskType.CAPTION: TaskRequirement(1, frozenset({PairType.SINGLE, PairType.CROSS_MODAL})),
    TaskType.GROUNDING: TaskRequirement(1, frozenset({PairType.SINGLE, PairType.CROSS_MODAL})),
    TaskType.SEGMENTATION: TaskRequirement(1, frozenset({PairType.SINGLE})),
    TaskType.COUNT: TaskRequirement(1, frozenset({PairType.SINGLE})),
    TaskType.SCENE_CLASSIFY: TaskRequirement(1, frozenset({PairType.SINGLE, PairType.CROSS_MODAL})),
    TaskType.CHANGE_VQA: TaskRequirement(2, frozenset({PairType.BI_TEMPORAL})),
    TaskType.CHANGE_CAPTION: TaskRequirement(2, frozenset({PairType.BI_TEMPORAL})),
    TaskType.CHANGE_MAP: TaskRequirement(2, frozenset({PairType.BI_TEMPORAL})),
    TaskType.CROSS_MODAL_VQA: TaskRequirement(2, frozenset({PairType.CROSS_MODAL})),
    TaskType.CROSS_MODAL_COMPARE: TaskRequirement(2, frozenset({PairType.CROSS_MODAL})),
}
"""``UNSUPPORTED`` is deliberately absent: it is a classifier outcome, never an offer."""


def supported_tasks(
    pair_type: PairType,
    image_count: int,
    overall: Overall,
    modalities: list[Modality] | None = None,
) -> list[TaskType]:
    """Return the tasks these inputs can support, in enum declaration order.

    A ``FAIL`` compatibility verdict supports nothing: the images cannot be
    analysed together at all, so offering a task would be a lie.
    """
    if overall is Overall.FAIL or pair_type is PairType.INCOMPATIBLE:
        return []
    present = set(modalities or [])
    offered: list[TaskType] = []
    for task, requirement in REQUIREMENTS.items():
        if image_count < requirement.min_images:
            continue
        if pair_type not in requirement.pair_types:
            continue
        if requirement.modalities is not None and not present <= requirement.modalities:
            continue
        offered.append(task)
    return offered
