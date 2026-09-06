"""The inference bundle: weights, normalisation, calibrated threshold, provenance.

A bare ``state_dict`` is not enough to serve a detector, and every field this
bundle adds exists because leaving it out produces a silent wrong answer rather
than an error:

* **The architecture config.** Rebuilding with a different backbone and loading
  the weights anyway runs, scores nothing like the training run, and raises
  nothing.
* **The normalisation.** A model trained on per-channel standardised input and
  served raw reflectance produces a plausible-looking mask that is unrelated to
  the scene. There is no way to detect this from the mask.
* **The calibrated threshold.** ``0.5`` is the threshold that minimises training
  loss, not the one that maximises F1 — on class-imbalanced change data, where
  under 5 % of pixels change, the two are far apart. The value that was actually
  swept on the validation split travels with the weights.
* **The provenance.** ``dataset``, ``gsd_m`` and the metrics are what let the
  trace say which model answered, and what let the cross-resolution ablation
  compare two rows honestly.

Only ``torch`` is needed to read one of these, so a serving checkout loads a
checkpoint without Lightning or torchgeo installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

DEFAULT_CHECKPOINT_DIR: Final[Path] = Path("data/checkpoints/cd")
"""Where ``scripts/train_cd.py`` writes and ``tools/change_detect.py`` looks."""

CHECKPOINT_ENV: Final[str] = "SATQUERY_CD_CHECKPOINT"
"""Overrides the search entirely. The demo machine points this at one file."""

BUNDLE_SUFFIX: Final[str] = ".ckpt.pt"

FORMAT_VERSION: Final[int] = 1


class CheckpointError(RuntimeError):
    """A checkpoint could not be read, or does not describe a usable model."""


@dataclass(frozen=True)
class Normalisation:
    """Per-channel standardisation, in the order the model's bands are declared.

    Reflectance rather than digital numbers: everything upstream of the model
    reads through :func:`satquery.render.tiling.render_source`, which already
    scales to ``[0, 1]``, so these statistics are computed on that scale.
    """

    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        """Reject statistics that cannot standardise anything.

        Raises:
            CheckpointError: The vectors disagree, are empty, or contain a zero
                standard deviation.
        """
        if not self.mean or len(self.mean) != len(self.std):
            raise CheckpointError(
                f"normalisation needs matching non-empty vectors, got "
                f"{len(self.mean)} means and {len(self.std)} stds"
            )
        if any(value == 0.0 for value in self.std):
            raise CheckpointError("a zero standard deviation would divide by zero")

    @classmethod
    def identity(cls, channels: int) -> Normalisation:
        """Pass-through statistics, for a model trained on unscaled input."""
        return cls(mean=(0.0,) * channels, std=(1.0,) * channels)

    def to_dict(self) -> dict[str, list[float]]:
        """A JSON-safe record."""
        return {"mean": list(self.mean), "std": list(self.std)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Normalisation:
        """Rebuild from a checkpoint record."""
        return cls(
            mean=tuple(float(v) for v in raw["mean"]),
            std=tuple(float(v) for v in raw["std"]),
        )


@dataclass
class CheckpointBundle:
    """Everything needed to serve one trained detector."""

    config: dict[str, Any]
    """The :class:`~satquery.training.cd.model.SiameseConfig` record."""

    state_dict: dict[str, Any]
    normalisation: Normalisation
    threshold: float = 0.5
    """Calibrated on the validation split. See :func:`calibrate_threshold`."""

    dataset: str = "unknown"
    gsd_m: float | None = None
    """Ground sample distance the model was trained at. The cross-resolution
    ablation is exactly the comparison of two bundles differing in this field."""

    metrics: dict[str, float] = field(default_factory=dict)
    encoder_source: str = "unknown"
    trained_at: str = ""
    format_version: int = FORMAT_VERSION
    notes: str = ""

    @property
    def version(self) -> str:
        """The registry-shaped version string this checkpoint answers to."""
        resolution = f"@{self.gsd_m:g}m" if self.gsd_m else ""
        return f"1.0.0+ckpt:{self.dataset}{resolution}"

    def metadata(self) -> dict[str, Any]:
        """Everything except the weights — safe to put in a trace."""
        return {
            "config": self.config,
            "normalisation": self.normalisation.to_dict(),
            "threshold": self.threshold,
            "dataset": self.dataset,
            "gsd_m": self.gsd_m,
            "metrics": self.metrics,
            "encoder_source": self.encoder_source,
            "trained_at": self.trained_at,
            "format_version": self.format_version,
            "notes": self.notes,
        }


def save_bundle(bundle: CheckpointBundle, path: Path) -> Path:
    """Write a bundle, plus a sidecar JSON of everything but the weights.

    The sidecar is not redundancy. It lets the training script, the demo
    inventory and a human ``cat`` read a checkpoint's provenance without loading
    torch and without deserialising 50 MB of tensors.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    payload = {**bundle.metadata(), "state_dict": bundle.state_dict}
    torch.save(payload, path)
    path.with_suffix(".json").write_text(
        json.dumps(bundle.metadata(), indent=2, default=str), encoding="utf-8"
    )
    return path


def load_bundle(path: Path, map_location: str = "cpu") -> CheckpointBundle:
    """Read a bundle from disk.

    Raises:
        CheckpointError: The file is missing, unreadable, or not a bundle this
            code can serve.
    """
    if not path.is_file():
        raise CheckpointError(f"no checkpoint at {path}")
    import torch

    try:
        payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as error:  # noqa: BLE001 - a corrupt file is a data problem
        raise CheckpointError(f"{path} could not be read: {error}") from error

    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise CheckpointError(f"{path} is not a SatQuery change-detection bundle")

    version = int(payload.get("format_version", 0))
    if version > FORMAT_VERSION:
        raise CheckpointError(
            f"{path} is format version {version}; this build reads up to {FORMAT_VERSION}"
        )

    return CheckpointBundle(
        config=dict(payload.get("config") or {}),
        state_dict=payload["state_dict"],
        normalisation=Normalisation.from_dict(
            payload.get("normalisation") or {"mean": [0.0], "std": [1.0]}
        ),
        threshold=float(payload.get("threshold", 0.5)),
        dataset=str(payload.get("dataset", "unknown")),
        gsd_m=(float(payload["gsd_m"]) if payload.get("gsd_m") is not None else None),
        metrics={str(k): float(v) for k, v in (payload.get("metrics") or {}).items()},
        encoder_source=str(payload.get("encoder_source", "unknown")),
        trained_at=str(payload.get("trained_at") or datetime.now(UTC).isoformat()),
        format_version=version or FORMAT_VERSION,
        notes=str(payload.get("notes", "")),
    )


def default_checkpoint_path() -> Path:
    """The directory checkpoints are written to and searched for."""
    return DEFAULT_CHECKPOINT_DIR


def find_checkpoint(search: Path | None = None) -> Path | None:
    """Locate a servable checkpoint without loading it.

    Resolution order: the ``SATQUERY_CD_CHECKPOINT`` environment variable, then
    the newest bundle in the checkpoint directory. Returns None rather than
    raising, because "no checkpoint on this machine" is a normal state that makes
    the tool unavailable — not an error that should fail a request.
    """
    override = os.environ.get(CHECKPOINT_ENV)
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None

    directory = search or DEFAULT_CHECKPOINT_DIR
    if not directory.is_dir():
        return None
    bundles = sorted(
        (path for path in directory.glob(f"*{BUNDLE_SUFFIX}") if path.is_file()),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    return bundles[0] if bundles else None


__all__ = [
    "BUNDLE_SUFFIX",
    "CHECKPOINT_ENV",
    "DEFAULT_CHECKPOINT_DIR",
    "FORMAT_VERSION",
    "CheckpointBundle",
    "CheckpointError",
    "Normalisation",
    "default_checkpoint_path",
    "find_checkpoint",
    "load_bundle",
    "save_bundle",
]
