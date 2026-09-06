"""``weighted_tool_agreement_v1`` (AGENT_POLICY_DAG.md §7.2).

    overall = 0.20*task_classification
            + 0.20*input_quality
            + 0.35*tool_mean
            + 0.25*cross_tool_agreement

then a ladder of caps, applied in order and each recorded by name. Three of the
choices here are the ones worth arguing about, so they are stated rather than
buried:

* **VLM steps are excluded from ``tool_mean``.** The VLM synthesises evidence, it
  does not produce it. Including it would let a fluent paragraph raise the
  system's confidence in a weak measurement, which is precisely backwards.
* **A single uncorroborated tool scores 0.75, not 1.0.** One measurement nobody
  checked is not the same as two that agree, and scoring it as if it were would
  make every single-tool plan systematically overconfident.
* **Caps clamp, they do not subtract.** ``overall`` is the *smaller* of the
  weighted sum and every applicable ceiling, so a plan that hit three problems
  cannot be dragged below the floor by arithmetic — the worst ceiling wins and
  the trace names it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import yaml

from satquery.evidence.fact_sheet import FactSheet
from satquery.schemas.enums import Overall, ToolCategory, ToolStatus
from satquery.schemas.manifest import InputManifest
from satquery.schemas.tool import Execution

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    # Runtime would close the loop evidence -> registry -> capability_match
    # -> tools.base -> evidence, which exists because a tool now receives
    # the FactSheet. Only the annotations need the type.
    from satquery.registry.registry import ToolRegistry
from satquery.schemas.trace import Confidence

CONFIG_PATH: Final[Path] = (
    Path(__file__).resolve().parents[3] / "configs" / "agreement_pairs.yaml"
)

METHOD: Final[str] = "weighted_tool_agreement_v1"

WEIGHTS: Final[dict[str, float]] = {
    "task_classification": 0.20,
    "input_quality": 0.20,
    "tool_mean": 0.35,
    "cross_tool_agreement": 0.25,
}

EVIDENCE_CATEGORIES: Final[frozenset[ToolCategory]] = frozenset(
    {ToolCategory.ANALYSIS, ToolCategory.CV, ToolCategory.FUSION}
)
"""Categories that *produce* evidence. ``vlm`` and ``geo`` are excluded."""

NEUTRAL_AGREEMENT: Final[float] = 0.75
"""What one uncorroborated measurement is worth — deliberately not 1.0."""

NO_EVIDENCE_TOOL_MEAN: Final[float] = 0.50
"""What a plan with no evidence step at all is worth."""

DEGRADED_WEIGHT: Final[float] = 0.50
GEOREF_PENALTY: Final[float] = 0.85
GSD_PENALTY: Final[float] = 0.80
PAIR_WARNING_PENALTY: Final[float] = 0.85

PRECISION: Final[int] = 4


@dataclass(frozen=True)
class AgreementPair:
    """Two scalars from different tools that measure the same quantity."""

    a: str
    b: str
    scale: float


@lru_cache(maxsize=1)
def load_agreement_pairs(path: Path | None = None) -> tuple[AgreementPair, ...]:
    """Load the frozen agreement set from ``configs/agreement_pairs.yaml``."""
    raw: dict[str, Any] = yaml.safe_load((path or CONFIG_PATH).read_text(encoding="utf-8"))
    return tuple(
        AgreementPair(a=str(entry["a"]), b=str(entry["b"]), scale=float(entry["scale"]))
        for entry in (raw.get("pairs") or [])
    )


@dataclass(frozen=True)
class Cap:
    """One ceiling on the final score."""

    name: str
    limit: float


CAPS: Final[tuple[Cap, ...]] = (
    Cap("compatibility_warnings", 0.80),
    Cap("llm_task_proposal", 0.75),
    Cap("degraded_execution", 0.70),
    Cap("template_fallback", 0.65),
    Cap("uncited_claims", 0.60),
    Cap("failed_execution", 0.50),
    Cap("generic_plan", 0.45),
)
"""Applied in this order. The order is what the trace reports, so it is frozen."""


@dataclass(frozen=True)
class Conditions:
    """Everything the caps key off, gathered in one place."""

    overall_compat: Overall
    task_proposed_by_llm: bool
    template_fallback: bool
    has_uncited_claims: bool
    generic_plan: bool
    caps_waived: frozenset[str] = frozenset()

    def triggered(self, executions: Sequence[Execution]) -> dict[str, bool]:
        """Which caps apply to this analysis."""
        statuses = {execution.status for execution in executions}
        return {
            "compatibility_warnings": self.overall_compat is Overall.PASS_WITH_WARNINGS,
            "llm_task_proposal": self.task_proposed_by_llm,
            "degraded_execution": ToolStatus.DEGRADED in statuses,
            "template_fallback": self.template_fallback,
            "uncited_claims": self.has_uncited_claims,
            "failed_execution": ToolStatus.FAILED in statuses,
            "generic_plan": self.generic_plan,
        }


def input_quality(
    manifests: Sequence[InputManifest],
    overall_compat: Overall,
    gsd_out_of_range: bool = False,
) -> float:
    """``pair_factor * mean_i(q_i)`` — how much the inputs themselves support an answer."""
    if not manifests:
        return 0.0
    scores: list[float] = []
    for manifest in manifests:
        nodata = 1.0 - min(max(manifest.nodata_pct, 0.0), 100.0) / 100.0
        georef = 1.0 if manifest.is_georeferenced else GEOREF_PENALTY
        gsd = GSD_PENALTY if gsd_out_of_range else 1.0
        scores.append(nodata * georef * gsd)
    pair_factor = (
        PAIR_WARNING_PENALTY if overall_compat is Overall.PASS_WITH_WARNINGS else 1.0
    )
    return pair_factor * (sum(scores) / len(scores))


def tool_mean(executions: Sequence[Execution], registry: ToolRegistry) -> float:
    """Mean confidence over the evidence-producing steps only.

    ``OK`` contributes its confidence, ``DEGRADED`` half of it, ``FAILED`` zero,
    and ``SKIPPED`` is excluded from the mean entirely — a step that never ran is
    not a measurement that went badly.
    """
    contributions: list[float] = []
    for execution in executions:
        spec = registry.get(execution.tool)
        if spec is None or spec.category not in EVIDENCE_CATEGORIES:
            continue
        if execution.status is ToolStatus.SKIPPED:
            continue
        if execution.status is ToolStatus.FAILED:
            contributions.append(0.0)
        elif execution.status is ToolStatus.DEGRADED:
            contributions.append(DEGRADED_WEIGHT * execution.confidence)
        else:
            contributions.append(execution.confidence)
    if not contributions:
        return NO_EVIDENCE_TOOL_MEAN
    return sum(contributions) / len(contributions)


def cross_tool_agreement(
    sheet: FactSheet, pairs: Iterable[AgreementPair] | None = None
) -> tuple[float, list[str]]:
    """How far independent tools agree about the same quantity.

    Returns:
        ``(score, corroborated)``. With fewer than two comparable measurements
        the score is the neutral prior and the list is empty.
    """
    comparable = pairs if pairs is not None else load_agreement_pairs()
    scores: list[float] = []
    corroborated: list[str] = []
    for pair in comparable:
        a = sheet.number(pair.a)
        b = sheet.number(pair.b)
        if a is None or b is None or pair.scale <= 0:
            continue
        scores.append(1.0 - min(1.0, abs(a - b) / pair.scale))
        corroborated.append(f"{pair.a} vs {pair.b}")
    if not scores:
        return NEUTRAL_AGREEMENT, corroborated
    return sum(scores) / len(scores), corroborated


def score(
    task_confidence: float,
    manifests: Sequence[InputManifest],
    executions: Sequence[Execution],
    registry: ToolRegistry,
    sheet: FactSheet,
    conditions: Conditions,
    gsd_out_of_range: bool = False,
) -> Confidence:
    """Compute the overall confidence and the components behind it.

    Components are reported at the same precision as ``overall``, so a frontend
    can display the arithmetic and it will visibly add up — which is the only
    reason to show a confidence number at all.
    """
    components = {
        "task_classification": task_confidence,
        "input_quality": input_quality(manifests, conditions.overall_compat, gsd_out_of_range),
        "tool_mean": tool_mean(executions, registry),
        "cross_tool_agreement": cross_tool_agreement(sheet)[0],
    }
    overall = sum(WEIGHTS[name] * value for name, value in components.items())

    applied: list[str] = []
    triggered = conditions.triggered(executions)
    for cap in CAPS:
        if not triggered.get(cap.name):
            continue
        if cap.name in conditions.caps_waived:
            # Entry 15 waives template_fallback: its templated answer is the
            # design, not a failure, and capping it would punish the right call.
            continue
        if overall > cap.limit:
            overall = cap.limit
        applied.append(cap.name)

    return Confidence(
        overall=round(max(0.0, min(1.0, overall)), PRECISION),
        method=METHOD,
        components={name: round(value, PRECISION) for name, value in components.items()},
        caps_applied=applied,
    )
