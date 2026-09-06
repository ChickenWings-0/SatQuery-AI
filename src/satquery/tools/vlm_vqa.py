"""``vlm_vqa`` — answer one question about a scene, constrained by the FactSheet.

The last step of nearly every plan, and the only one that writes prose. It reads
what the deterministic steps upstream of it measured and what the renderer drew,
and turns them into a sentence. It does not measure, it does not choose what runs
next, and it cannot introduce a number: Master.md §3.1 puts control flow in the
policy table and evidence in the tools, and this tool is the synthesiser at the
end of both.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from satquery.models.loader import VlmBackend
from satquery.schemas.enums import TaskType
from satquery.tools.base import ToolContext, ToolResult
from satquery.tools.vlm_runtime import synthesise

NAME: Final[str] = "vlm_vqa"

LABEL: Final[str] = "Answer (grounded in the FactSheet)"


class VlmVqa:
    """Answers the analyst's question from rendered views and measured scalars."""

    name = NAME

    def __init__(self, backend: VlmBackend | None = None) -> None:
        """Optionally pin a backend; production resolves the process-wide one."""
        self.backend = backend

    def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
        """Generate the answer to :attr:`ToolContext.question`.

        Raises:
            ToolError: No VLM backend was reachable, or generation failed. The
                executor degrades the step and the aggregator writes the
                templated answer, which is why this is never fatal.
        """
        return synthesise(
            ctx=ctx,
            params=params,
            tool_name=NAME,
            default_mode="vqa",
            default_task=TaskType.VQA,
            label=LABEL,
            backend=self.backend,
        )


__all__ = ["LABEL", "NAME", "VlmVqa"]
