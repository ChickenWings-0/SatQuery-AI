"""The YAML-driven tool registry (Master.md §5, API_CONTRACT §4.8).

``configs/registry.yaml`` is the single declaration of what every tool accepts
and produces. Loading it here rather than restating it in Python is what makes
the extension rule enforceable: a new capability is a registry entry plus a
policy-table row, and ``planner.py`` never changes.

The loader also *reality-checks* the declaration. A spec that says
``available: true`` but has no implementation behind it would make the
capabilities panel lie and would make capability matching plan a step that
cannot run, so availability is intersected with the implementation table at load
time and the reason is recorded.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import yaml

from satquery.ingest.bands import load_table
from satquery.schemas.tool import ToolSpec

CONFIG_PATH: Final[Path] = Path(__file__).resolve().parents[3] / "configs" / "registry.yaml"


class RegistryError(ValueError):
    """The registry declaration is internally inconsistent."""


@dataclass(frozen=True)
class ToolRegistry:
    """Every declared tool, keyed by its stable registry name."""

    version: str
    tools: dict[str, ToolSpec]

    def __contains__(self, name: object) -> bool:
        """True when *name* is a declared tool."""
        return name in self.tools

    def __getitem__(self, name: str) -> ToolSpec:
        """Return one spec.

        Raises:
            KeyError: No tool by that name is declared.
        """
        try:
            return self.tools[name]
        except KeyError:
            raise KeyError(f"no tool named {name!r} in registry {self.version}") from None

    def get(self, name: str) -> ToolSpec | None:
        """Return one spec, or None when it is not declared."""
        return self.tools.get(name)

    @property
    def names(self) -> list[str]:
        """Declared tool names, in declaration order."""
        return list(self.tools)

    @property
    def available(self) -> list[ToolSpec]:
        """The specs that can actually run right now."""
        return [spec for spec in self.tools.values() if spec.available]

    def with_availability(self, overrides: Mapping[str, bool]) -> ToolRegistry:
        """Return a copy with some tools forced available or unavailable.

        Tests use this to exercise the fallback paths (golden test 18) without
        editing the committed registry, and the executor uses it to honour
        ``options.disable_tools``.
        """
        tools = {}
        for name, spec in self.tools.items():
            if name in overrides and overrides[name] != spec.available:
                available = overrides[name]
                tools[name] = spec.model_copy(
                    update={
                        "available": available,
                        "unavailable_reason": (
                            None if available else "Disabled for this request."
                        ),
                    }
                )
            else:
                tools[name] = spec
        return ToolRegistry(version=self.version, tools=tools)


def _lint(tools: dict[str, ToolSpec]) -> None:
    """Reject a declaration that cannot produce a runnable plan.

    Both checks are golden tests 14/15 promoted into the loader, so a bad
    registry fails at startup with a clear message rather than at request time
    with a KeyError.

    Raises:
        RegistryError: A fallback names an unknown tool, or a required band
            resolves for no sensor in the alias table.
    """
    aliases = load_table()
    known_bands = {band for bands in aliases.sensors.values() for band in bands}
    known_bands |= {band for bands in aliases.fallback_by_band_count.values() for band in bands}

    for name, spec in tools.items():
        if spec.fallback is not None and spec.fallback not in tools:
            raise RegistryError(f"{name}.fallback names unknown tool {spec.fallback!r}")
        if spec.fallback == name:
            raise RegistryError(f"{name}.fallback points at itself")
        unknown = sorted(set(spec.accepts.required_bands) - known_bands)
        if unknown:
            raise RegistryError(
                f"{name}.accepts.required_bands {unknown} resolve for no sensor "
                f"in band_aliases.yaml v{aliases.version}"
            )
        if spec.accepts.max_images < spec.accepts.min_images:
            raise RegistryError(f"{name}.accepts has max_images < min_images")


DEFAULT_UNAVAILABLE_REASON: Final[str] = "No implementation is registered for this tool."


def parse_registry(
    raw: Mapping[str, Any],
    implemented: Iterable[str] | None = None,
    reasons: Mapping[str, str] | None = None,
) -> ToolRegistry:
    """Build a :class:`ToolRegistry` from parsed YAML.

    Args:
        raw: The document loaded from ``registry.yaml``.
        implemented: Names that can actually run here. A tool declared available
            that is not in this set is downgraded rather than trusted.
        reasons: Why a specific tool was downgraded. Without one the reason is
            "no implementation", which stops being true from Phase 4 on: a
            ``vlm_*`` tool or the change detector has code behind it and is
            unavailable because the *weights* are not on this machine. A
            capabilities panel that says the wrong thing is worse than one that
            says nothing.

    Raises:
        RegistryError: The declaration is inconsistent.
    """
    entries = raw.get("tools") or []
    tools: dict[str, ToolSpec] = {}
    for entry in entries:
        spec = ToolSpec.model_validate(entry)
        if spec.name in tools:
            raise RegistryError(f"duplicate tool name {spec.name!r}")
        tools[spec.name] = spec

    if implemented is not None:
        have = set(implemented)
        for name, spec in tools.items():
            if spec.available and name not in have:
                tools[name] = spec.model_copy(
                    update={
                        "available": False,
                        "unavailable_reason": (reasons or {}).get(
                            name, DEFAULT_UNAVAILABLE_REASON
                        ),
                    }
                )

    _lint(tools)
    return ToolRegistry(version=str(raw.get("version", "unknown")), tools=tools)


def load_registry(
    path: Path | None = None,
    implemented: Iterable[str] | None = None,
    reasons: Mapping[str, str] | None = None,
) -> ToolRegistry:
    """Load, validate and return the registry from disk."""
    source = path or CONFIG_PATH
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
    return parse_registry(raw, implemented=implemented, reasons=reasons)


@lru_cache(maxsize=1)
def default_registry() -> ToolRegistry:
    """The process-wide registry, checked against what this machine can run.

    Not merely against the implementation table: a ``vlm_*`` tool has code behind
    it from Phase 4 on and still cannot run without servable weights, so the
    reality check asks the catalogue what is *runnable* rather than what is
    written (see :func:`satquery.tools.catalog.runnable_tools`).
    """
    from satquery.tools.catalog import unavailable_reasons

    runnable, reasons = unavailable_reasons()
    return load_registry(implemented=runnable, reasons=reasons)
