"""``vlm_caption`` — describe a scene, constrained by the FactSheet.

Captioning is where an ungrounded model is most dangerous, because nobody asked
it a question it could decline to answer: handed a satellite image and told to
describe it, a language model will happily produce "approximately 40 % tree
cover" from nothing at all. Running captions through the same FactSheet
constraint as VQA is what keeps the description free to be vivid about shape and
arrangement while remaining unable to invent a quantity.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from satquery.models.loader import VlmBackend
from satquery.schemas.enums import TaskType
from satquery.tools.base import ToolContext, ToolResult
from satquery.tools.vlm_runtime import synthesise

NAME: Final[str] = "vlm_caption"

LABEL: Final[str] = "Scene description (grounded in the FactSheet)"


class VlmCaption:
    """Describes the scene from rendered views and measured scalars."""

    name = NAME

    def __init__(self, backend: VlmBackend | None = None) -> None:
        """Optionally pin a backend; production resolves the process-wide one."""
        self.backend = backend

    def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
        """Generate a description of the scene.

        Raises:
            ToolError: No VLM backend was reachable, or generation failed.
        """
        return synthesise(
            ctx=ctx,
            params=params,
            tool_name=NAME,
            default_mode="caption",
            default_task=TaskType.CAPTION,
            label=LABEL,
            backend=self.backend,
        )


__all__ = ["LABEL", "NAME", "VlmCaption"]
