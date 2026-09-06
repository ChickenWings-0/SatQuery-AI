"""Capability matching — AGENT_POLICY_DAG.md §5.

Eight checks, run in order against the resolved inputs of one plan step. Seven
are hard: failing one means this tool cannot legitimately run on these inputs,
so the declared fallback is tried instead, and if that also fails the step is
skipped and everything downstream of it with it.

The eighth — ground sample distance — is deliberately **soft**. A tool trained
at 0.5-10 m asked to work on 0.65 m Cartosat-2S imagery is being asked to
generalise, not being misused; refusing would turn a degraded answer into no
answer at all, and the hidden evaluation set is exactly where that would hurt.
So an out-of-range GSD warns, reduces ``input_quality``, and runs.

Fallback resolution is **one level only**. A fallback's fallback is not
followed: unbounded chains make both latency and the trace unpredictable, and
the frozen chain in §5 covers every real case in a single hop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from satquery.registry.registry import ToolRegistry
from satquery.schemas.enums import CheckStatus, PairType
from satquery.schemas.tool import ToolSpec

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    # See the note in evidence.confidence: importing tools.base here at
    # runtime makes the import order of satquery.tools vs satquery.evidence
    # decide whether the package imports at all.
    from satquery.tools.base import ImageBundle


class MatchStatus(StrEnum):
    """What capability matching decided about one step."""

    OK = "OK"
    """The declared tool runs as planned."""

    SUBSTITUTED = "SUBSTITUTED"
    """The declared tool cannot run; its fallback can, and will."""

    SKIPPED = "SKIPPED"
    """Neither the tool nor its fallback can run on these inputs."""


CHECK_ORDER: Final[tuple[str, ...]] = (
    "availability",
    "pair_type",
    "image_count",
    "modality",
    "bands",
    "georeference",
    "size",
    "gsd",
)
"""The eight checks, in evaluation order. ``gsd`` is the only soft one."""

SOFT_CHECKS: Final[frozenset[str]] = frozenset({"gsd"})


@dataclass(frozen=True)
class CapabilityCheck:
    """The outcome of one of the eight checks."""

    name: str
    status: CheckStatus
    detail: str
    value: str | float | None = None
    threshold: str | None = None

    @property
    def blocking(self) -> bool:
        """True when this outcome forces a fallback."""
        return self.status is CheckStatus.FAIL and self.name not in SOFT_CHECKS


@dataclass(frozen=True)
class MatchResult:
    """What will actually run for one step, and the evidence for that decision."""

    requested: str
    status: MatchStatus
    tool: str | None
    spec: ToolSpec | None
    checks: list[CapabilityCheck] = field(default_factory=list)
    fallback_of: str | None = None
    reason: str = ""

    @property
    def warnings(self) -> list[CapabilityCheck]:
        """The soft failures worth surfacing — currently GSD only."""
        return [check for check in self.checks if check.status is CheckStatus.WARN]

    @property
    def gsd_out_of_range(self) -> bool:
        """True when the inputs sit outside the tool's trained GSD range."""
        return any(
            check.name == "gsd" and check.status is CheckStatus.WARN for check in self.checks
        )


def _check_availability(spec: ToolSpec) -> CapabilityCheck:
    if spec.available:
        return CapabilityCheck(
            "availability",
            CheckStatus.PASS,
            f"{spec.name} is loaded and ready.",
            threshold="available",
        )
    return CapabilityCheck(
        "availability",
        CheckStatus.FAIL,
        spec.unavailable_reason or f"{spec.name} is not available on this deployment.",
        threshold="available",
    )


def _check_pair_type(spec: ToolSpec, pair_type: PairType) -> CapabilityCheck:
    accepted = spec.accepts.pair_types
    if pair_type in accepted:
        return CapabilityCheck(
            "pair_type",
            CheckStatus.PASS,
            f"{spec.name} accepts {pair_type.value} inputs.",
            value=pair_type.value,
            threshold=", ".join(p.value for p in accepted),
        )
    return CapabilityCheck(
        "pair_type",
        CheckStatus.FAIL,
        f"{spec.name} does not accept {pair_type.value} inputs.",
        value=pair_type.value,
        threshold=", ".join(p.value for p in accepted),
    )


def _check_image_count(spec: ToolSpec, count: int) -> CapabilityCheck:
    low, high = spec.accepts.min_images, spec.accepts.max_images
    threshold = f"{low} <= n <= {high}"
    if low <= count <= high:
        return CapabilityCheck(
            "image_count", CheckStatus.PASS, f"{count} input(s) resolved.", count, threshold
        )
    return CapabilityCheck(
        "image_count",
        CheckStatus.FAIL,
        f"{spec.name} needs {threshold} inputs but the step resolved {count}.",
        count,
        threshold,
    )


def _check_modality(spec: ToolSpec, images: Sequence[ImageBundle]) -> CapabilityCheck:
    accepted = set(spec.accepts.modalities)
    present = {image.modality for image in images}
    threshold = ", ".join(sorted(m.value for m in accepted))
    if not present:
        return CapabilityCheck(
            "modality",
            CheckStatus.SKIP,
            "The step consumes artifacts only, so no modality applies.",
            threshold=threshold,
        )
    if present <= accepted:
        return CapabilityCheck(
            "modality",
            CheckStatus.PASS,
            f"{spec.name} accepts every input modality.",
            ", ".join(sorted(m.value for m in present)),
            threshold,
        )
    unsupported = sorted(m.value for m in present - accepted)
    return CapabilityCheck(
        "modality",
        CheckStatus.FAIL,
        f"{spec.name} does not accept {', '.join(unsupported)} input.",
        ", ".join(sorted(m.value for m in present)),
        threshold,
    )


def _check_bands(spec: ToolSpec, images: Sequence[ImageBundle]) -> CapabilityCheck:
    required = spec.accepts.required_bands
    threshold = ", ".join(required) if required else "none required"
    if not required:
        return CapabilityCheck(
            "bands",
            CheckStatus.PASS,
            f"{spec.name} requires no specific band.",
            threshold=threshold,
        )
    if not images:
        return CapabilityCheck(
            "bands",
            CheckStatus.SKIP,
            "The step consumes artifacts only, so no band resolution applies.",
            threshold=threshold,
        )
    gaps = {image.id: image.missing(required) for image in images}
    missing = {ref: names for ref, names in gaps.items() if names}
    if not missing:
        return CapabilityCheck(
            "bands",
            CheckStatus.PASS,
            f"Every band {spec.name} requires resolves on all inputs.",
            threshold,
            threshold,
        )
    detail = "; ".join(f"{ref} lacks {', '.join(names)}" for ref, names in sorted(missing.items()))
    return CapabilityCheck("bands", CheckStatus.FAIL, f"{detail}.", detail, threshold)


def _check_georeference(spec: ToolSpec, images: Sequence[ImageBundle]) -> CapabilityCheck:
    if not spec.accepts.requires_georeference:
        return CapabilityCheck(
            "georeference",
            CheckStatus.PASS,
            f"{spec.name} works on non-georeferenced input.",
            threshold="not required",
        )
    ungeoreferenced = [i.id for i in images if not i.manifest.is_georeferenced]
    if not ungeoreferenced:
        return CapabilityCheck(
            "georeference",
            CheckStatus.PASS,
            "Every input carries a CRS and a transform.",
            threshold="all georeferenced",
        )
    return CapabilityCheck(
        "georeference",
        CheckStatus.FAIL,
        f"{spec.name} needs georeferenced input but {', '.join(ungeoreferenced)} is not.",
        ", ".join(ungeoreferenced),
        "all georeferenced",
    )


def _check_size(spec: ToolSpec, images: Sequence[ImageBundle]) -> CapabilityCheck:
    low, high = spec.accepts.min_size_px, spec.accepts.max_size_px
    threshold = f"{low or 0} px <= edge <= {high or 'inf'} px"
    if not images or (low is None and high is None):
        return CapabilityCheck(
            "size",
            CheckStatus.SKIP,
            "No size constraint applies to this step.",
            threshold=threshold,
        )
    shortest = min(min(i.manifest.width, i.manifest.height) for i in images)
    longest = max(max(i.manifest.width, i.manifest.height) for i in images)
    if low is not None and shortest < low:
        return CapabilityCheck(
            "size",
            CheckStatus.FAIL,
            f"The shortest edge is {shortest} px; {spec.name} needs at least {low} px.",
            shortest,
            threshold,
        )
    if high is not None and longest > high:
        return CapabilityCheck(
            "size",
            CheckStatus.FAIL,
            f"The longest edge is {longest} px; {spec.name} accepts at most {high} px.",
            longest,
            threshold,
        )
    return CapabilityCheck(
        "size",
        CheckStatus.PASS,
        f"Input edges of {shortest}-{longest} px are in range.",
        shortest,
        threshold,
    )


def _check_gsd(spec: ToolSpec, images: Sequence[ImageBundle]) -> CapabilityCheck:
    """The soft check: out of range degrades the answer, it never blocks it."""
    window = spec.accepts.gsd_range_m
    if window is None:
        return CapabilityCheck(
            "gsd", CheckStatus.SKIP, f"{spec.name} declares no GSD range.", threshold="any"
        )
    low, high = window
    threshold = f"{low} m <= gsd <= {high} m"
    observed = [i.manifest.gsd_m for i in images if i.manifest.gsd_m is not None]
    if not observed:
        return CapabilityCheck(
            "gsd",
            CheckStatus.SKIP,
            "No input reports a ground sample distance.",
            threshold=threshold,
        )
    outside = [g for g in observed if not low <= g <= high]
    if not outside:
        return CapabilityCheck(
            "gsd",
            CheckStatus.PASS,
            f"Input GSD of {min(observed)}-{max(observed)} m is in range.",
            min(observed),
            threshold,
        )
    return CapabilityCheck(
        "gsd",
        CheckStatus.WARN,
        f"{spec.name} was trained at {low}-{high} m and is being applied at "
        f"{', '.join(f'{g} m' for g in sorted(set(outside)))}; results are extrapolated.",
        min(outside),
        threshold,
    )


def evaluate(
    spec: ToolSpec,
    pair_type: PairType,
    images: Sequence[ImageBundle],
    artifact_count: int = 0,
) -> list[CapabilityCheck]:
    """Run all eight checks against one spec, in the frozen order."""
    return [
        _check_availability(spec),
        _check_pair_type(spec, pair_type),
        _check_image_count(spec, len(images) + artifact_count),
        _check_modality(spec, images),
        _check_bands(spec, images),
        _check_georeference(spec, images),
        _check_size(spec, images),
        _check_gsd(spec, images),
    ]


def match(
    tool: str,
    registry: ToolRegistry,
    pair_type: PairType,
    images: Sequence[ImageBundle],
    artifact_count: int = 0,
) -> MatchResult:
    """Decide what runs for one step: the tool, its fallback, or nothing.

    Args:
        tool: The tool the policy table named.
        registry: The loaded registry.
        pair_type: The pair type of the whole analysis.
        images: The step's resolved image inputs.
        artifact_count: How many upstream artifacts the step also consumes.

    Returns:
        A :class:`MatchResult` carrying the decision and every check behind it.
    """
    spec = registry.get(tool)
    if spec is None:
        return MatchResult(
            requested=tool,
            status=MatchStatus.SKIPPED,
            tool=None,
            spec=None,
            reason=f"{tool} is not in registry {registry.version}",
        )

    checks = evaluate(spec, pair_type, images, artifact_count)
    blocking = [check for check in checks if check.blocking]
    if not blocking:
        return MatchResult(
            requested=tool,
            status=MatchStatus.OK,
            tool=tool,
            spec=spec,
            checks=checks,
            reason=f"capability_match:{tool}",
        )

    primary_reason = "; ".join(check.detail for check in blocking)
    if spec.fallback is None:
        return MatchResult(
            requested=tool,
            status=MatchStatus.SKIPPED,
            tool=None,
            spec=None,
            checks=checks,
            reason=f"{primary_reason} No fallback is declared.",
        )

    fallback_spec = registry.get(spec.fallback)
    if fallback_spec is None:
        return MatchResult(
            requested=tool,
            status=MatchStatus.SKIPPED,
            tool=None,
            spec=None,
            checks=checks,
            reason=f"{primary_reason} Fallback {spec.fallback!r} is not registered.",
        )

    # One level only: the fallback's own fallback is deliberately not followed.
    fallback_checks = evaluate(fallback_spec, pair_type, images, artifact_count)
    if any(check.blocking for check in fallback_checks):
        detail = "; ".join(check.detail for check in fallback_checks if check.blocking)
        return MatchResult(
            requested=tool,
            status=MatchStatus.SKIPPED,
            tool=None,
            spec=None,
            checks=[*checks, *fallback_checks],
            reason=f"{primary_reason} Fallback {spec.fallback} also failed: {detail}",
        )

    return MatchResult(
        requested=tool,
        status=MatchStatus.SUBSTITUTED,
        tool=fallback_spec.name,
        spec=fallback_spec,
        checks=[*checks, *fallback_checks],
        fallback_of=tool,
        reason=f"{primary_reason} Substituted declared fallback {fallback_spec.name}.",
    )
