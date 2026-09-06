"""The FactSheet: the single source of truth every claim is checked against.

Two rules make it trustworthy (AGENT_POLICY_DAG.md §6.4):

* **Namespacing is unconditional.** Keys are always ``{tool}.{scalar}``, even in
  a single-tool plan, so a citation never becomes ambiguous when a plan grows a
  second tool later.
* **Only schema-conforming scalars are merged.** A tool that emits a scalar its
  registry entry does not declare has that scalar *dropped* and a
  ``SCALAR_SCHEMA_VIOLATION`` warning recorded. It must never reach the answer,
  because this sheet is what the citation validator resolves against — an
  undeclared number here would launder a hallucination into a citation.

The namespace is by tool, not by step. A tool invoked twice in one plan
disambiguates at the scalar level through the policy table's ``suffixes``
(``ndbi_mean_pre`` / ``ndbi_mean_post``), which the table loader validates
statically. Namespacing by step would make citation keys shift every time
someone reorders a plan.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from satquery.schemas.enums import ToolStatus
from satquery.schemas.tool import Execution

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    # A runtime import would close the loop registry -> capability_match ->
    # tools.base -> evidence, which exists because a tool now receives the sheet.
    from satquery.registry.registry import ToolRegistry

_EVIDENCE_STATUSES: Final[frozenset[ToolStatus]] = frozenset(
    {ToolStatus.OK, ToolStatus.DEGRADED}
)
"""A FAILED or SKIPPED step contributes nothing: there is no measurement to cite."""


@dataclass(frozen=True)
class Fact:
    """One measured scalar and where it came from."""

    key: str
    tool: str
    scalar: str
    step: int
    value: float | str

    @property
    def source(self) -> str:
        """The citation grammar of API_CONTRACT §3.9."""
        return f"step:{self.step}/scalars.{self.scalar}"

    @property
    def is_numeric(self) -> bool:
        """True when this fact can back a numeric claim."""
        return isinstance(self.value, int | float)


@dataclass
class FactSheet:
    """Every scalar the plan measured, namespaced by the tool that measured it."""

    facts: dict[str, Fact] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    def __contains__(self, key: object) -> bool:
        """True when a namespaced key is present."""
        return key in self.facts

    def __len__(self) -> int:
        """Number of merged facts."""
        return len(self.facts)

    def get(self, key: str) -> Fact | None:
        """Return one fact by its namespaced key."""
        return self.facts.get(key)

    def value(self, key: str) -> float | str | None:
        """Return one fact's value by its namespaced key."""
        fact = self.facts.get(key)
        return fact.value if fact else None

    def number(self, key: str) -> float | None:
        """Return one fact's value as a float, or None when it is not numeric."""
        fact = self.facts.get(key)
        if fact is None or not fact.is_numeric:
            return None
        return float(fact.value)

    def numeric_facts(self) -> list[Fact]:
        """Every numeric fact, in insertion order."""
        return [fact for fact in self.facts.values() if fact.is_numeric]

    def by_tool(self, tool: str) -> dict[str, Fact]:
        """Every fact one tool contributed, keyed by its unqualified scalar name."""
        return {
            fact.scalar: fact for fact in self.facts.values() if fact.tool == tool
        }

    def as_dict(self) -> dict[str, float | str]:
        """The contract-shaped ``fact_sheet`` mapping."""
        return {key: fact.value for key, fact in self.facts.items()}


def _type_matches(value: Any, declared: str | None) -> bool:
    """Check one scalar against a JSON Schema primitive type.

    ``Execution.scalars`` is typed ``float | str`` by the frozen contract, so a
    count arrives here as ``25.0`` rather than ``25``. ``integer`` therefore means
    *integral*, not *of Python type int* — anything else would reject every count
    the system measures.
    """
    if declared is None:
        return True
    if isinstance(value, bool):
        return declared == "boolean"
    if declared == "string":
        return isinstance(value, str)
    if declared == "integer":
        return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    if declared == "number":
        return isinstance(value, int | float)
    if declared == "boolean":
        return False
    return False


def validate_scalar(name: str, value: Any, schema: Mapping[str, Any]) -> str | None:
    """Validate one scalar against a tool's ``scalars_schema``.

    Supports the subset of JSON Schema the registry actually uses: ``properties``
    for fixed names and ``patternProperties`` for the suffixed families a
    bi-temporal invocation produces. A name matching neither is undeclared, and
    undeclared is a violation rather than a permission.

    Returns:
        None when the scalar is valid, otherwise a one-line reason.
    """
    properties = schema.get("properties") or {}
    if name in properties:
        declared = properties[name].get("type")
        if _type_matches(value, declared):
            return None
        return f"{name} should be {declared}, got {type(value).__name__}"

    for pattern, subschema in (schema.get("patternProperties") or {}).items():
        if re.fullmatch(pattern, name):
            declared = subschema.get("type")
            if _type_matches(value, declared):
                return None
            return f"{name} should be {declared}, got {type(value).__name__}"

    return f"{name} is not declared in the tool's scalars_schema"


def build(
    executions: Sequence[Execution], registry: ToolRegistry, strict: bool = True
) -> FactSheet:
    """Merge every conforming scalar into one namespaced sheet.

    Args:
        executions: The plan's execution records, in step order.
        registry: Supplies each tool's ``scalars_schema``.
        strict: When False, scalars are merged without schema validation. Only
            useful for exercising the validator itself.

    Returns:
        The populated :class:`FactSheet`, with every dropped scalar explained in
        :attr:`FactSheet.violations`.
    """
    sheet = FactSheet()
    for execution in sorted(executions, key=lambda e: e.step):
        if execution.status not in _EVIDENCE_STATUSES or not execution.scalars:
            continue
        spec = registry.get(execution.tool)
        schema: Mapping[str, Any] = spec.scalars_schema if spec else {}

        for name, value in execution.scalars.items():
            if strict:
                problem = validate_scalar(name, value, schema)
                if problem is not None:
                    sheet.violations.append(f"{execution.tool}: {problem}")
                    continue

            key = f"{execution.tool}.{name}"
            existing = sheet.facts.get(key)
            if existing is not None:
                if existing.value != value:
                    # The table loader is supposed to prevent this; if it ever
                    # slips through, the first measurement wins and the clash is
                    # visible rather than silently overwritten.
                    sheet.violations.append(
                        f"{execution.tool}: {name} was measured twice with different values "
                        f"({existing.value} then {value}); the policy table must supply "
                        "distinct params.suffixes"
                    )
                continue

            sheet.facts[key] = Fact(
                key=key,
                tool=execution.tool,
                scalar=name,
                step=execution.step,
                value=value,
            )
    return sheet
