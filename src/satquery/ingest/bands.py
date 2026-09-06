"""Logical band resolution against ``configs/band_aliases.yaml``.

The alias table is the shared vocabulary between capability matching
(``AGENT_POLICY_DAG.md`` §5) and the renderer (``DATA_ADAPTATION_PLAN.md`` §2.2).
It is loaded from the YAML rather than restated here so the two consumers cannot
drift apart.

Unavailability is explicit: a sensor that does not carry SWIR resolves ``swir1``
to ``None``, and callers must degrade rather than substitute another band.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import yaml

CONFIG_PATH: Final[Path] = Path(__file__).resolve().parents[3] / "configs" / "band_aliases.yaml"

_POLARISATIONS: Final[frozenset[str]] = frozenset({"vv", "vh", "hh", "hv"})


@dataclass(frozen=True)
class BandAliasTable:
    """The parsed alias table: sensor name -> {logical band -> 1-indexed number}."""

    version: str
    sensors: dict[str, dict[str, int]]
    fallback_by_band_count: dict[int, dict[str, int]]

    def for_sensor(self, sensor: str | None, band_count: int) -> dict[str, int]:
        """Return the alias map for *sensor*, falling back to a positional guess.

        Args:
            sensor: A ``sensor_guess`` from the modality classifier, or None.
            band_count: Used to pick a positional fallback when the sensor is
                unknown or absent from the table.
        """
        if sensor and sensor in self.sensors:
            return dict(self.sensors[sensor])
        return dict(self.fallback_by_band_count.get(band_count, {}))


@lru_cache(maxsize=1)
def load_table(path: Path | None = None) -> BandAliasTable:
    """Load and cache the alias table from disk."""
    source = path or CONFIG_PATH
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
    sensors = {
        str(name): {str(k): int(v) for k, v in (entry.get("bands") or {}).items()}
        for name, entry in (raw.get("sensors") or {}).items()
    }
    fallback = {
        int(count): {str(k): int(v) for k, v in (entry or {}).items()}
        for count, entry in (raw.get("fallback_by_band_count") or {}).items()
    }
    return BandAliasTable(
        version=str(raw.get("version", "unknown")),
        sensors=sensors,
        fallback_by_band_count=fallback,
    )


def resolve_bands(
    sensor: str | None,
    band_count: int,
    band_names: list[str] | None = None,
) -> dict[str, int]:
    """Resolve logical band names to 1-indexed band numbers for one image.

    Declared band names win over the table when they are recognisable
    polarisations, because a file that says ``VH`` in band 1 means it.

    Args:
        sensor: The guessed sensor, or None.
        band_count: Number of bands actually present.
        band_names: Band descriptions read off the file, if any.

    Returns:
        A mapping of logical name to band number, containing only bands that
        genuinely exist in the file.
    """
    aliases = load_table().for_sensor(sensor, band_count)

    for index, name in enumerate(band_names or [], start=1):
        key = name.strip().lower()
        if key in _POLARISATIONS:
            aliases[key] = index

    return {name: number for name, number in aliases.items() if 1 <= number <= band_count}


def missing_bands(required: list[str], resolved: dict[str, int]) -> list[str]:
    """Return the required logical bands that *resolved* does not provide."""
    return [name for name in required if name not in resolved]
