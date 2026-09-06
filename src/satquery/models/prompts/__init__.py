"""Versioned prompt library. The FactSheet constraint lives in ``templates``.

``box_format`` is re-exported here because it is a *contract* rather than a
helper: the Phase 7 corpus builder and ``tools/text_grounding.py`` must import
the same serialiser, and DATA_ADAPTATION_PLAN §4.5 makes a drift between them a
silent, total failure. One import path makes that harder to get wrong.
"""

from satquery.models.prompts.box_format import (
    BOX_SCALE,
    BoxFormatError,
    NormalisedBox,
    from_pixels,
    parse_boxes,
    serialise_boxes,
)
from satquery.models.prompts.builder import (
    CITATION_MARKER,
    MAX_VIEWS,
    BuiltPrompt,
    MarkedAnswer,
    ViewInput,
    build_prompt,
    build_system_prompt,
    build_user_prompt,
    render_fact_sheet,
    render_view_index,
    strip_citation_markers,
)
from satquery.models.prompts.templates import (
    DEFAULT_VERSION,
    GROUNDED_V1,
    TEMPLATES,
    PromptTemplate,
    get_template,
)

__all__ = [
    "BOX_SCALE",
    "CITATION_MARKER",
    "DEFAULT_VERSION",
    "GROUNDED_V1",
    "MAX_VIEWS",
    "TEMPLATES",
    "BoxFormatError",
    "BuiltPrompt",
    "MarkedAnswer",
    "NormalisedBox",
    "PromptTemplate",
    "ViewInput",
    "build_prompt",
    "build_system_prompt",
    "build_user_prompt",
    "from_pixels",
    "get_template",
    "parse_boxes",
    "render_fact_sheet",
    "render_view_index",
    "serialise_boxes",
    "strip_citation_markers",
]
