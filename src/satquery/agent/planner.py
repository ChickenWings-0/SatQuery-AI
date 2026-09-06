"""The deterministic planner (AGENT_POLICY_DAG.md §4).

One table lookup, no search, no LLM. ``configs/policy_table.yaml`` maps
``"{TaskType}|{PairType}|{ModalityKey}"`` onto a fixed DAG, and this module does
nothing except resolve that key, load the entry and stamp it with the resolved
inputs.

That is the point rather than a limitation. A ReAct loop choosing its own tools
produces a different trace on every run, and the trace is what the rubric
scores. Here, two runs over the same images with the same query produce a
byte-identical plan — and when the router *was* guessing, the generic fallback
says so in the trace instead of hiding it.

**Extension rule (Master.md §5):** adding a tool must never require touching this
file. New behaviour arrives as a registry entry plus one or more table rows.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import yaml

from satquery.schemas.enums import ArtifactType, ImageRole, Modality, PairType, TaskType
from satquery.schemas.manifest import InputManifest
from satquery.schemas.tool import PlanStep
from satquery.schemas.trace import Plan

CONFIG_PATH: Final[Path] = Path(__file__).resolve().parents[3] / "configs" / "policy_table.yaml"

PLANNER_VERSION: Final[str] = "policy_table_v1"
GENERIC_PLANNER: Final[str] = "fallback_generic_v1"
GENERIC_KEY: Final[str] = "*|*|*"
MIXED_MODALITY: Final[str] = "mixed"

_ARTIFACT_TOKEN: Final[re.Pattern[str]] = re.compile(r"^@(\d+)(?::([A-Z_]+))?$")

_ROLE_TOKENS: Final[dict[str, ImageRole]] = {
    "pre": ImageRole.PRE,
    "post": ImageRole.POST,
    "optical": ImageRole.OPTICAL,
    "sar": ImageRole.SAR,
    "single": ImageRole.SINGLE,
}


class PolicyTableError(ValueError):
    """The policy table is malformed or references something that does not exist."""


@dataclass(frozen=True)
class ArtifactSelector:
    """An ``@N`` or ``@N:TYPE`` reference to an upstream step's output."""

    step: int
    artifact_type: ArtifactType | None = None

    def __str__(self) -> str:
        """The token as it appears in the table and in the plan."""
        return f"@{self.step}" + (f":{self.artifact_type.value}" if self.artifact_type else "")


@dataclass(frozen=True)
class PolicyStep:
    """One row of a policy entry, before inputs are resolved."""

    step: int
    tool: str
    inputs: tuple[str, ...]
    depends_on: tuple[int, ...]
    params: dict[str, Any]
    optional: bool = True

    @property
    def artifact_selectors(self) -> list[ArtifactSelector]:
        """The ``@N`` references among this step's inputs."""
        selectors = []
        for token in self.inputs:
            found = _ARTIFACT_TOKEN.match(token)
            if found:
                selectors.append(
                    ArtifactSelector(
                        step=int(found.group(1)),
                        artifact_type=ArtifactType(found.group(2)) if found.group(2) else None,
                    )
                )
        return selectors

    @property
    def image_tokens(self) -> list[str]:
        """The role tokens among this step's inputs."""
        return [token for token in self.inputs if not _ARTIFACT_TOKEN.match(token)]


@dataclass(frozen=True)
class PolicyEntry:
    """One key and the DAG it selects."""

    key: str
    steps: tuple[PolicyStep, ...]
    caps_waived: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyTable:
    """The parsed, validated table."""

    version: str
    entries: dict[str, PolicyEntry]

    def get(self, key: str) -> PolicyEntry | None:
        """Return the entry for an exact key."""
        return self.entries.get(key)


def _parse_step(index: int, raw: dict[str, Any], key: str) -> PolicyStep:
    tool = raw.get("tool")
    if not isinstance(tool, str):
        raise PolicyTableError(f"{key} step {index}: 'tool' is required")
    depends_on = tuple(int(d) for d in raw.get("depends_on", ()))
    for dependency in depends_on:
        if dependency >= index:
            raise PolicyTableError(
                f"{key} step {index} depends on step {dependency}, which is not earlier; "
                "the table must be listed in topological order"
            )
        if dependency < 1:
            raise PolicyTableError(f"{key} step {index} has a non-positive dependency")
    return PolicyStep(
        step=index,
        tool=tool,
        inputs=tuple(str(i) for i in raw.get("inputs", ())),
        depends_on=depends_on,
        params=dict(raw.get("params") or {}),
        optional=bool(raw.get("optional", True)),
    )


def reachable_from(entry: PolicyEntry, step: PolicyStep) -> set[int]:
    """Every step *step* transitively depends on."""
    by_number = {candidate.step: candidate for candidate in entry.steps}
    seen: set[int] = set()
    frontier = list(step.depends_on)
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        upstream = by_number.get(current)
        if upstream is not None:
            frontier.extend(upstream.depends_on)
    return seen


def _validate_selectors(entry: PolicyEntry) -> None:
    """Reject a step that reads an artifact it is not ordered after.

    Reachability, not a direct edge: the frozen table's own change entry has the
    terminal step read ``@2:CHANGE_MASK`` while depending on ``[1, 3, 4]``, and
    that is correct — step 3 depends on step 2, so the mask exists and step 2's
    status has already propagated down the chain. What would be a bug is reading
    a step nothing orders you after, because then the artifact may not exist yet.

    Raises:
        PolicyTableError: A step reads an artifact from an unordered step.
    """
    for step in entry.steps:
        ordered = reachable_from(entry, step)
        for selector in step.artifact_selectors:
            if selector.step not in ordered:
                raise PolicyTableError(
                    f"{entry.key} step {step.step} reads {selector} but does not depend on "
                    f"step {selector.step}, directly or transitively"
                )


def _validate_suffixes(entry: PolicyEntry) -> None:
    """Reject an entry where one tool could overwrite its own scalars.

    The FactSheet namespaces by tool, not by step (§6.4), so two invocations of
    the same tool in one plan must carry distinct ``suffixes`` or the second
    silently overwrites the first — and the FactSheet is the citation ground
    truth, so a silent overwrite is a correctness bug, not a cosmetic one.
    """
    seen: dict[str, list[tuple[str, ...]]] = {}
    for step in entry.steps:
        suffixes = tuple(str(s) for s in step.params.get("suffixes", ()))
        seen.setdefault(step.tool, []).append(suffixes)
    for tool, invocations in seen.items():
        if len(invocations) > 1 and len(set(invocations)) != len(invocations):
            raise PolicyTableError(
                f"{entry.key}: {tool} appears {len(invocations)} times without distinct "
                "params.suffixes, so its scalars would collide in the FactSheet"
            )


def parse_table(raw: dict[str, Any]) -> PolicyTable:
    """Build a :class:`PolicyTable` from parsed YAML.

    Raises:
        PolicyTableError: The table is malformed.
    """
    entries: dict[str, PolicyEntry] = {}
    for raw_entry in raw.get("entries") or []:
        key = str(raw_entry.get("key", ""))
        if not key:
            raise PolicyTableError("every entry needs a 'key'")
        if key in entries:
            raise PolicyTableError(f"duplicate policy key {key!r}")
        steps = tuple(
            _parse_step(index, step, key)
            for index, step in enumerate(raw_entry.get("steps") or [], start=1)
        )
        if not steps:
            raise PolicyTableError(f"{key} has no steps")
        entry = PolicyEntry(
            key=key,
            steps=steps,
            caps_waived=tuple(str(c) for c in raw_entry.get("caps_waived") or ()),
        )
        _validate_selectors(entry)
        _validate_suffixes(entry)
        entries[key] = entry

    if GENERIC_KEY not in entries:
        raise PolicyTableError(f"the table must define the generic entry {GENERIC_KEY!r}")
    return PolicyTable(version=str(raw.get("version", "unknown")), entries=entries)


def load_table(path: Path | None = None) -> PolicyTable:
    """Load and validate the policy table from disk."""
    source = path or CONFIG_PATH
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
    return parse_table(raw)


@lru_cache(maxsize=1)
def default_table() -> PolicyTable:
    """The process-wide policy table."""
    return load_table()


def lint_against_registry(table: PolicyTable, tool_names: Sequence[str]) -> list[str]:
    """Return every policy-table tool that the registry does not declare.

    Golden test 14 runs this in CI: a table row naming a tool nobody registered
    is a plan that cannot be executed, and it should fail the build rather than
    the request.
    """
    known = set(tool_names)
    missing = {
        step.tool
        for entry in table.entries.values()
        for step in entry.steps
        if step.tool not in known
    }
    return sorted(missing)


def modality_key(pair_type: PairType, manifests: Sequence[InputManifest]) -> tuple[str, PairType]:
    """Derive the ``ModalityKey`` half of the routing key.

    Returns:
        ``(key, pair_type)``. A bi-temporal pair whose two images turn out to be
        different modalities is re-classified ``CROSS_MODAL`` here, because
        comparing an optical epoch against a SAR epoch is a fusion problem, not
        a change problem, and routing it as change would produce confident
        nonsense.
    """
    modalities = [manifest.modality for manifest in manifests]
    if pair_type is PairType.CROSS_MODAL:
        return MIXED_MODALITY, pair_type
    if pair_type is PairType.BI_TEMPORAL:
        distinct = set(modalities)
        if len(distinct) > 1:
            return MIXED_MODALITY, PairType.CROSS_MODAL
        return (distinct.pop().value if distinct else Modality.UNKNOWN.value), pair_type
    return (modalities[0].value if modalities else Modality.UNKNOWN.value), pair_type


def resolve_key(
    table: PolicyTable, task: TaskType, pair_type: PairType, modality: str
) -> tuple[str, PolicyEntry, str]:
    """Resolve the routing key, first hit wins.

    Returns:
        ``(key, entry, planner)``. A key that only resolves at the generic entry
        comes back with ``planner=fallback_generic_v1``, which is what makes the
        router's uncertainty visible — and self-limiting, since the
        ``generic_plan`` confidence cap keys off exactly that string.
    """
    exact = f"{task.value}|{pair_type.value}|{modality}"
    entry = table.get(exact)
    if entry is not None:
        return exact, entry, PLANNER_VERSION

    wildcard = f"{task.value}|{pair_type.value}|*"
    entry = table.get(wildcard)
    if entry is not None:
        return wildcard, entry, PLANNER_VERSION

    generic = table.entries[GENERIC_KEY]
    return GENERIC_KEY, generic, GENERIC_PLANNER


def resolve_images(token: str, manifests: Sequence[InputManifest]) -> list[str]:
    """Resolve one role token to the ``img_*`` ids it names.

    An unmatched role token resolves to nothing rather than to a guess: a step
    addressed at ``pre`` when no image holds that role has no legitimate input,
    and capability matching will skip it on the image-count check.
    """
    if token == "all":
        return [manifest.id for manifest in manifests]
    role = _ROLE_TOKENS.get(token)
    if role is None:
        return []
    matched = [manifest.id for manifest in manifests if manifest.role is role]
    if matched:
        return matched
    # A lone unroled image is what "single" means, whatever the manifest says.
    if role is ImageRole.SINGLE and len(manifests) == 1:
        return [manifests[0].id]
    return []


@dataclass(frozen=True)
class PlannedStep:
    """One plan step with its inputs resolved and its params fixed."""

    step: int
    tool: str
    depends_on: tuple[int, ...]
    image_refs: tuple[str, ...]
    selectors: tuple[ArtifactSelector, ...]
    params: dict[str, Any]
    optional: bool
    reason: str

    @property
    def input_refs(self) -> list[str]:
        """Plan-time input references: concrete ``img_*`` plus ``@N`` selectors.

        Upstream artifacts are still selectors here because their ``art_*`` ids
        are assigned during execution. The concrete ids land in
        ``Execution.input_refs``, which is the field an auditor replays from.
        """
        return [*self.image_refs, *(str(selector) for selector in self.selectors)]


@dataclass(frozen=True)
class PlanResult:
    """The chosen plan and everything the trace needs to explain it."""

    plan: Plan
    entry: PolicyEntry
    steps: tuple[PlannedStep, ...]
    pair_type: PairType
    modality_key: str
    warnings: list[tuple[str, str]] = field(default_factory=list)

    @property
    def is_generic(self) -> bool:
        """True when the router fell through to the generic entry."""
        return self.plan.planner == GENERIC_PLANNER

    @property
    def caps_waived(self) -> tuple[str, ...]:
        """Confidence caps this entry waives (§7.2)."""
        return self.entry.caps_waived


def plan_for(
    task: TaskType,
    pair_type: PairType,
    manifests: Sequence[InputManifest],
    table: PolicyTable | None = None,
) -> PlanResult:
    """Select and instantiate the DAG for one classified query.

    Args:
        task: The resolved primary task.
        pair_type: The compatibility report's pair type.
        manifests: The ingested inputs, in upload order.
        table: Override for tests; defaults to the committed table.

    Returns:
        A :class:`PlanResult` carrying the contract-shaped :class:`Plan` and the
        richer step records the executor needs.
    """
    policy = table or default_table()
    modality, effective_pair_type = modality_key(pair_type, manifests)
    warnings: list[tuple[str, str]] = []

    if effective_pair_type is not pair_type:
        warnings.append(
            (
                "PAIR_RECLASSIFIED_CROSS_MODAL",
                "The two images are different modalities, so the pair was routed as "
                "cross-modal rather than bi-temporal.",
            )
        )

    key, entry, planner = resolve_key(policy, task, effective_pair_type, modality)
    if planner == GENERIC_PLANNER:
        warnings.append(
            (
                "GENERIC_PLAN_USED",
                f"No policy entry matched {task.value}|{effective_pair_type.value}|{modality}; "
                "a generic plan ran and the confidence is capped accordingly.",
            )
        )

    planned: list[PlannedStep] = []
    steps: list[PlanStep] = []
    for step in entry.steps:
        image_refs: list[str] = []
        for token in step.image_tokens:
            for ref in resolve_images(token, manifests):
                if ref not in image_refs:
                    image_refs.append(ref)
        selectors = tuple(step.artifact_selectors)
        reason = f"policy_table:{key}"
        planned.append(
            PlannedStep(
                step=step.step,
                tool=step.tool,
                depends_on=step.depends_on,
                image_refs=tuple(image_refs),
                selectors=selectors,
                params=dict(step.params),
                optional=step.optional,
                reason=reason,
            )
        )
        steps.append(
            PlanStep(
                step=step.step,
                tool=step.tool,
                depends_on=list(step.depends_on),
                input_refs=planned[-1].input_refs,
                reason=reason,
            )
        )

    return PlanResult(
        plan=Plan(planner=planner, policy_key=key, steps=steps),
        entry=entry,
        steps=tuple(planned),
        pair_type=effective_pair_type,
        modality_key=modality,
        warnings=warnings,
    )
