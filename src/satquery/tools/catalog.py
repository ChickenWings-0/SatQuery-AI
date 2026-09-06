"""The name -> implementation table.

``configs/registry.yaml`` declares what a tool *accepts*; this maps the same key
onto the code that runs it. A registry entry with no implementation here is a
tool whose weights or backend have not landed yet: capability matching treats it
as unavailable and substitutes its fallback.

Having an implementation is necessary but not sufficient. The ``vlm_*`` tools are
fully implemented and still cannot run on a machine with no model weights on it,
so :func:`runnable_tools` intersects the table with what the environment can
actually serve. That is the same honesty rule the registry loader already applies
— a capabilities panel that promises a tool which cannot run is worse than one
that admits the gap.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from satquery.tools.base import Tool
from satquery.tools.change_detect import SiameseChangeDetector
from satquery.tools.change_statistics import ChangeStatistics
from satquery.tools.crossmodal_dofa import CrossModalConsistency
from satquery.tools.image_diff_change import ImageDiffChange
from satquery.tools.object_counter import ObjectCounter
from satquery.tools.physics_agreement import PhysicsAgreement
from satquery.tools.raster_statistics import RasterStatistics
from satquery.tools.sar_backscatter_analyzer import SarBackscatterAnalyzer
from satquery.tools.semantic_segmenter import SemanticSegmenter
from satquery.tools.spectral_index_analyzer import SpectralIndexAnalyzer
from satquery.tools.spectral_renderer import SpectralRenderer
from satquery.tools.text_grounding import TextGrounding
from satquery.tools.vlm_caption import VlmCaption
from satquery.tools.vlm_change_vqa import VlmChangeVqa
from satquery.tools.vlm_vqa import VlmVqa

BUILTIN_TOOLS: Final[Mapping[str, Tool]] = {
    tool.name: tool
    for tool in (
        SpectralRenderer(),
        RasterStatistics(),
        SpectralIndexAnalyzer(),
        SarBackscatterAnalyzer(),
        ImageDiffChange(),
        ChangeStatistics(),
        ObjectCounter(),
        PhysicsAgreement(),
        CrossModalConsistency(),
        SemanticSegmenter(),
        TextGrounding(),
        SiameseChangeDetector(),
        VlmVqa(),
        VlmCaption(),
        VlmChangeVqa(),
    )
}
"""Every tool with code behind it. Frozen at import; the executor takes an
override mapping rather than mutating this, so a test cannot leak a fake tool
into another test."""

VLM_TOOLS: Final[frozenset[str]] = frozenset(
    {"vlm_vqa", "vlm_caption", "vlm_change_vqa", "text_grounding"}
)
"""Tools whose availability depends on a servable VLM, not just on code.
``text_grounding`` belongs here rather than with the checkpoint tools: its boxes
come out of Qwen3-VL itself, so it is exactly as servable as the synthesisers
are. It differs from them in having a declared fallback — ``semantic_segmenter``
— so a grounding question on a VLM-less box degrades rather than skipping."""

CHECKPOINT_TOOLS: Final[frozenset[str]] = frozenset(
    {"siamese_change_detector", "semantic_segmenter"}
)
"""Tools whose availability depends on trained weights being on this machine.
Unlike the VLM tools, these have a declared ``fallback`` — ``image_diff_change``
and ``spectral_index_analyzer`` — so an untrained checkout still answers change
and segmentation questions, visibly DEGRADED."""

DOFA_TOOLS: Final[frozenset[str]] = frozenset({"crossmodal_consistency"})
"""Tools needing the DOFA encoder, which arrives with torchgeo rather than with a
checkpoint of ours. Falls back to ``physics_agreement``, whose rules need no
weights at all — which is why requirement 4 is answerable on any machine."""


def vlm_servable() -> bool:
    """True when some backend on this machine can actually answer a prompt.

    Probed offline and cheaply — no weights are read, no network call is made
    (see :func:`satquery.models.loader.available_backend`) — because this runs
    once at registry load, on every process, including the ones that will never
    ask the model anything.
    """
    from satquery.models.loader import BackendKind, available_backend

    return available_backend() is not BackendKind.NONE


def change_detector_servable() -> bool:
    """True when a trained change-detection checkpoint is on this machine.

    Probed by looking for the file, never by loading it: this runs once per
    process at registry load, and deserialising a checkpoint to find out whether
    it exists would put tens of megabytes of tensor reads in the startup path of
    every process, including the ones that will never detect anything.
    """
    from satquery.tools.change_detect import checkpoint_available

    return checkpoint_available()


def segmenter_servable() -> bool:
    """True when a segmentation checkpoint is configured and present.

    Probed by looking for the file, never by loading it — the same rule the
    change detector follows, and for the same reason: this runs at registry load
    in every process, including those that will never segment anything.
    """
    from satquery.tools.semantic_segmenter import checkpoint_available

    return checkpoint_available()


def dofa_servable() -> bool:
    """True when the DOFA encoder could be constructed here.

    An import check only. Whether the *weights* are cached is not knowable
    without a network call, and making registry load depend on the network is
    exactly the failure this probe exists to avoid — a missing download surfaces
    at request time as a ToolError and degrades to physics_agreement there.
    """
    from satquery.tools.crossmodal_dofa import available

    return available()


def _probe() -> dict[str, bool]:
    """Every availability probe, run once so the callers below cannot disagree."""
    return {
        "vlm": vlm_servable(),
        "checkpoint": change_detector_servable(),
        "segmenter": segmenter_servable(),
        "dofa": dofa_servable(),
    }


def runnable_tools() -> list[str]:
    """The tools that can run right now, weights and backends included.

    This is what :func:`satquery.registry.registry.default_registry` intersects
    the declared availability against. On a machine with no VLM downloaded the
    ``vlm_*`` entries drop out and answers are templated; on one with no trained
    checkpoint ``siamese_change_detector`` drops out and capability matching
    substitutes ``image_diff_change``. Both are the honest degraded mode, and
    both are visible in the trace rather than silent.
    """
    return _runnable(_probe())


def _runnable(probe: Mapping[str, bool]) -> list[str]:
    """The runnable set under an already-taken probe."""
    return [
        name
        for name in BUILTIN_TOOLS
        if (name not in VLM_TOOLS or probe["vlm"])
        and (name != "siamese_change_detector" or probe["checkpoint"])
        and (name != "semantic_segmenter" or probe["segmenter"])
        and (name not in DOFA_TOOLS or probe["dofa"])
    ]


NO_VLM_REASON: Final[str] = (
    "The tool is implemented, but no Qwen3-VL backend is servable here: neither a "
    "local snapshot nor a llama.cpp server was found."
)

NO_CHECKPOINT_REASON: Final[str] = (
    "The tool is implemented, but no trained change-detection checkpoint was found. "
    "Train one with scripts/train_cd.py or set SATQUERY_CD_CHECKPOINT. Change "
    "questions are still answered by the image_diff_change fallback."
)

NO_SEGMENTER_REASON: Final[str] = (
    "The tool is implemented, but no segmentation checkpoint was found. Point "
    "SATQUERY_SEG_CHECKPOINT at a SegFormer checkpoint; land-cover fractions are "
    "still measured by the spectral_index_analyzer fallback."
)

NO_GROUNDING_REASON: Final[str] = (
    "The tool is implemented, but grounding boxes come from Qwen3-VL and no "
    "backend is servable here. Boxes still come from the semantic_segmenter "
    "fallback where a segmentation checkpoint is present."
)

NO_DOFA_REASON: Final[str] = (
    "The tool is implemented, but the DOFA encoder needs torchgeo, which is not "
    "installed (`uv sync --extra cd`). Optical/SAR agreement is still measured by "
    "the deterministic physics_agreement fallback, which needs no weights."
)


def unavailable_reasons() -> tuple[list[str], dict[str, str]]:
    """The runnable tools, paired with why each of the others is not.

    Returned together because they are one decision: the same probe that removes
    a tool from the runnable set is the only thing that knows why it was removed,
    and splitting them lets the two answers disagree.
    """
    probe = _probe()
    reasons: dict[str, str] = {}
    if not probe["vlm"]:
        reasons.update(dict.fromkeys(VLM_TOOLS - {"text_grounding"}, NO_VLM_REASON))
        reasons["text_grounding"] = NO_GROUNDING_REASON
    if not probe["checkpoint"]:
        reasons["siamese_change_detector"] = NO_CHECKPOINT_REASON
    if not probe["segmenter"]:
        reasons["semantic_segmenter"] = NO_SEGMENTER_REASON
    if not probe["dofa"]:
        reasons.update(dict.fromkeys(DOFA_TOOLS, NO_DOFA_REASON))
    return _runnable(probe), reasons


def implementations(overrides: Mapping[str, Tool] | None = None) -> dict[str, Tool]:
    """Return the tool table, with *overrides* layered on top."""
    return {**BUILTIN_TOOLS, **(overrides or {})}


__all__ = [
    "BUILTIN_TOOLS",
    "CHECKPOINT_TOOLS",
    "DOFA_TOOLS",
    "VLM_TOOLS",
    "change_detector_servable",
    "dofa_servable",
    "implementations",
    "runnable_tools",
    "segmenter_servable",
    "unavailable_reasons",
    "vlm_servable",
]
