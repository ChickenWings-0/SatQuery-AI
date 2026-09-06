"""Tools: one protocol, one measurement each (Master.md §5).

Every tool but the ``vlm_*`` pair is deterministic. Those two synthesise the
prose at the end of a plan and produce no evidence of their own — their registry
entries declare no scalars, so the language model cannot introduce a number the
deterministic layer did not measure.
"""

from satquery.tools.base import (
    ArtifactDraft,
    Bands,
    ClassMap,
    ImageBundle,
    MaskPayload,
    MissingInputError,
    PixelReader,
    RenderDraft,
    Tool,
    ToolContext,
    ToolError,
    ToolResult,
)
from satquery.tools.catalog import (
    BUILTIN_TOOLS,
    CHECKPOINT_TOOLS,
    VLM_TOOLS,
    change_detector_servable,
    implementations,
    runnable_tools,
    vlm_servable,
)

__all__ = [
    "BUILTIN_TOOLS",
    "CHECKPOINT_TOOLS",
    "VLM_TOOLS",
    "ArtifactDraft",
    "Bands",
    "ClassMap",
    "ImageBundle",
    "MaskPayload",
    "MissingInputError",
    "PixelReader",
    "RenderDraft",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolResult",
    "change_detector_servable",
    "implementations",
    "runnable_tools",
    "vlm_servable",
]
