"""``vlm_change_vqa`` — answer a bi-temporal question from the measured change.

The change path is where an ungrounded model is at its most confident and least
useful. Handed two satellite images and asked what changed, a language model will
describe plausible urban expansion whether or not any occurred, because "some
development has taken place between the two acquisitions" is true of almost every
real pair and costs nothing to assert.

What makes this tool different from that is upstream of it. By the time it runs,
``siamese_change_detector`` has produced a mask and ``change_statistics`` has
turned that mask into an area in square metres and a component count, and those
numbers are the only ones the prompt permits. The model's job is to say where the
change is and what it looks like — which needs the views — while every quantity
it states comes from the FactSheet.

The serving stack is shared with Phase 4's ``vlm_vqa`` and ``vlm_caption``:
identical backends, identical prompt library, identical citation check. Only the
task instruction differs, and it differs in one place —
:mod:`satquery.models.prompts.templates` — rather than in a second prompt here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from satquery.models.loader import VlmBackend
from satquery.schemas.enums import TaskType
from satquery.tools.base import ToolContext, ToolResult
from satquery.tools.vlm_runtime import synthesise

NAME: Final[str] = "vlm_change_vqa"

LABEL: Final[str] = "Change answer (grounded in the FactSheet)"

_CAPTION_MODES: Final[frozenset[str]] = frozenset({"caption", "change_caption"})
"""The policy table drives both CHANGE_VQA and CHANGE_CAPTION through this one
tool, distinguished by ``params.mode`` — entries 13 and 14."""


class VlmChangeVqa:
    """Synthesises the bi-temporal FactSheet into an answer or a description."""

    name = NAME

    def __init__(self, backend: VlmBackend | None = None) -> None:
        """Optionally pin a backend; production resolves the process-wide one."""
        self.backend = backend

    def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
        """Answer the change question, or describe the change when captioning.

        Raises:
            ToolError: No VLM backend was reachable, or generation failed. The
                executor degrades the step and the aggregator falls back to the
                templated answer — which, for change, is already fully cited by
                construction.
        """
        mode = str(params.get("mode", "vqa")).strip().lower()
        default_task = (
            TaskType.CHANGE_CAPTION if mode in _CAPTION_MODES else TaskType.CHANGE_VQA
        )
        return synthesise(
            ctx=ctx,
            params=params,
            tool_name=NAME,
            default_mode=mode or "vqa",
            default_task=default_task,
            label=LABEL,
            backend=self.backend,
        )


__all__ = ["LABEL", "NAME", "VlmChangeVqa"]
