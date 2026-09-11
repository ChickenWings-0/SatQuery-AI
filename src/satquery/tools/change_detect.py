"""``siamese_change_detector`` — the trained detector behind the tool protocol.

What this tool owns, beyond calling a model:

* **Tiled inference.** A change map is only useful at a resolution where the
  change is visible, which for Cartosat-2S is far more pixels than fit on the
  card at once. :mod:`satquery.tools.tiled_inference` does the cutting and the
  overlap blending; this tool decides what a tile *is* and hands the model over.
* **Threshold calibration.** The operating point that maximises F1 on
  class-imbalanced change data is not 0.5, so the value swept on the validation
  split travels in the checkpoint and is used in preference to any default. A
  caller may override it, and ``threshold: "otsu"`` re-derives it from this
  scene's own probability field — which is what a sensor the model was never
  trained on needs.
* **Honest degradation.** No checkpoint, no torch, or a model that raises means
  :class:`ToolError`, which the executor turns into the declared
  ``image_diff_change`` fallback and a ``DEGRADED`` execution. A change question
  is still answered; the trace says by what.

The output contract is deliberately identical to the fallback's — same artifact
types, same labels, same basemap — because the executor may substitute one for
the other and the evidence gallery must not change shape when it does. That
identity lives in :mod:`satquery.tools.change_common`.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from skimage.filters import threshold_otsu

from satquery.schemas.enums import Device, Modality, ToolStatus
from satquery.tools.base import (
    Array,
    ImageBundle,
    Mask,
    MissingInputError,
    ToolContext,
    ToolError,
    ToolResult,
)
from satquery.tools.change_common import change_drafts
from satquery.tools.tiled_inference import (
    DEFAULT_OVERLAP_PX,
    DEFAULT_TILE_PX,
    TilePlan,
    TilingError,
    peak_tile_bytes,
    predict_tiled,
)

log = logging.getLogger(__name__)

NAME: Final[str] = "siamese_change_detector"

INFERENCE_SIZE_PX: Final[int] = 1024
"""The canvas the detector runs on, independent of the 448 px the renderer uses
for the VLM's views. A change map is a *spatial* product: at 448 px a Cartosat
scene is 0.5 m data resampled to roughly 5 m, and every building-scale change the
model was trained to find is gone before it is asked. Tiling is what makes the
larger canvas affordable."""

MAX_INFERENCE_SIZE_PX: Final[int] = 8192
"""Upper bound on ``size_px``. Not a memory limit — tiling handles that — but a
latency one: the tile count grows quadratically and the demo has a budget."""

OTSU: Final[str] = "otsu"
"""``threshold: otsu`` re-derives the operating point from this scene."""

SELF_CHECK_TOLERANCE: Final[float] = 0.02
"""Default absolute tolerance on the self-test's changed fraction.

Across thirty loads a healthy detector returned 0.1152-0.1163 on the stored pair
— a spread of 0.001 — while the corrupt loads returned 0.0000, 0.0000, 0.2872 and
0.5661. Two percentage points is twenty times the observed healthy spread and an
order of magnitude inside the nearest failure."""

SELF_CHECK_ATTEMPTS: Final[int] = 4
"""How many times to rebuild a model that fails its own known-answer test.

Corruption is per-load, not per-process: it appeared in 4 of 30 loads on this
card, and reloading after one cleared it every time. Retrying therefore turns a
detected fault into a recovered one, which is the difference between a demo that
degrades one run in eight and a demo that does not. Four attempts takes the
residual failure rate from 13% to about one run in three thousand."""

_OPTICAL_BANDS: Final[tuple[str, ...]] = ("red", "green", "blue")
_SAR_BANDS: Final[tuple[str, ...]] = ("vv",)


class DetectorUnavailableError(ToolError):
    """No trained checkpoint could be served on this machine."""


# ------------------------------------------------------------------- the model


@dataclass
class LoadedModel:
    """A checkpoint brought up on a device, ready to predict."""

    model: Any
    bundle: Any
    device: str
    torch: Any

    @property
    def bands(self) -> tuple[str, ...]:
        """Logical bands the model expects, in channel order."""
        return tuple(str(band) for band in self.bundle.config.get("bands") or _OPTICAL_BANDS)

    def predict(self, pre: Array, post: Array) -> Array:
        """Run one tile pair through the model and return probabilities."""
        torch = self.torch
        mean = np.asarray(self.bundle.normalisation.mean, dtype=np.float32)[:, None, None]
        std = np.asarray(self.bundle.normalisation.std, dtype=np.float32)[:, None, None]

        with torch.no_grad():
            tensors = [
                torch.from_numpy(
                    np.nan_to_num((stack - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
                )
                .unsqueeze(0)
                .to(self.device)
                for stack in (pre, post)
            ]
            probabilities = self.model.predict_probabilities(*tensors)
        result: Array = probabilities[0, 0].float().cpu().numpy().astype(np.float32)
        return result


def _self_check(loaded: LoadedModel) -> None:
    """Verify a freshly loaded detector against the known answer in its bundle.

    Both failure modes this system has produced are silent. A checkpoint loaded
    onto a busy card comes up numerically corrupt and marks a third of every scene
    changed, for the life of the process; a training/serving normalisation skew
    once made it mark nothing changed, equally confidently. Neither raises. Both
    produce a mask, a percentage and a fluent sentence quoting it, and nothing
    downstream can tell the number is invented — so this is the last place the
    fault can be caught at all.

    On this machine it is not rare: 4 of 30 loads were corrupt.

    Raises:
        DetectorUnavailableError: The weights loaded but do not reproduce the
            answer this checkpoint is known to give.
    """
    probe = loaded.bundle.self_test
    if not probe or probe.get("pre") is None or probe.get("post") is None:
        # A checkpoint from before the probe existed. Nothing to check against,
        # and refusing to serve it would be worse than serving it unverified.
        return

    pre = np.asarray(probe["pre"], dtype=np.float32)
    post = np.asarray(probe["post"], dtype=np.float32)
    expected = float(probe["changed_fraction"])
    tolerance = float(probe.get("tolerance") or SELF_CHECK_TOLERANCE)

    probabilities = loaded.predict(pre, post)
    if not np.isfinite(probabilities).all():
        raise DetectorUnavailableError(
            f"the detector returned non-finite probabilities on its own self-test "
            f"after loading onto {loaded.device}"
        )

    measured = float((probabilities >= loaded.bundle.threshold).mean())
    if abs(measured - expected) > tolerance:
        raise DetectorUnavailableError(
            f"the detector marked {measured:.2%} of its own self-test pair changed, "
            f"where this checkpoint is known to mark {expected:.2%} "
            f"(tolerance {tolerance:.2%}). The weights did not survive the load "
            f"onto {loaded.device}."
        )


class ModelCache:
    """Loads the checkpoint once per process and keeps it on the device.

    A change plan calls the detector once, but a demo answers question after
    question over the same scene, and reloading 11 M parameters and re-running
    the CUDA context setup for each of them is most of the wall-clock a judge
    sees. The cache is keyed on the checkpoint path so pointing
    ``SATQUERY_CD_CHECKPOINT`` somewhere else takes effect without a restart.
    """

    def __init__(self) -> None:
        """Create an empty cache."""
        self._lock = threading.Lock()
        self._loaded: dict[tuple[str, str], LoadedModel] = {}

    def get(self, path: Path, device: str) -> LoadedModel:
        """Return the model for *path* on *device*, loading it if needed.

        Raises:
            DetectorUnavailableError: torch is missing, or the checkpoint cannot
                be loaded into the architecture it names.
        """
        key = (str(path), device)
        with self._lock:
            cached = self._loaded.get(key)
            if cached is not None:
                return cached
            loaded = self._build_verified(path, device)
            self._loaded[key] = loaded
            return loaded

    def _build_verified(self, path: Path, device: str) -> LoadedModel:
        """Load until the weights pass their own known-answer test.

        The corruption this defends against is introduced *by* the load, not by
        the checkpoint, so building again is a real fix rather than a retry in
        hope. Nothing is cached until a model passes, so a caller never inherits
        one that failed.

        Raises:
            DetectorUnavailableError: Every attempt produced unusable weights.
        """
        last: DetectorUnavailableError | None = None
        for attempt in range(1, SELF_CHECK_ATTEMPTS + 1):
            loaded = self._load(path, device)
            try:
                _self_check(loaded)
            except DetectorUnavailableError as error:
                last = error
                log.warning(
                    "change detector failed its self-test on attempt %d/%d: %s",
                    attempt,
                    SELF_CHECK_ATTEMPTS,
                    error,
                )
                continue
            if attempt > 1:
                log.info("change detector loaded cleanly on attempt %d", attempt)
            return loaded
        raise DetectorUnavailableError(
            f"the change detector failed its self-test on all {SELF_CHECK_ATTEMPTS} "
            f"load attempts: {last}"
        )

    def _load(self, path: Path, device: str) -> LoadedModel:
        """Read a bundle and instantiate its model."""
        try:
            import torch
        except ImportError as error:
            raise DetectorUnavailableError(
                f"PyTorch is required to run {NAME}: {error}"
            ) from error

        from satquery.training.cd.checkpoint import CheckpointError, load_bundle
        from satquery.training.cd.model import SiameseChangeDetector, SiameseConfig

        try:
            bundle = load_bundle(path, map_location="cpu")
        except CheckpointError as error:
            raise DetectorUnavailableError(str(error)) from error

        config = SiameseConfig.from_dict(bundle.config)
        # Never re-fetch pretrained weights at serving time: the checkpoint
        # already contains them, and a hub call here would make inference depend
        # on the network.
        config.pretrained = False
        model = SiameseChangeDetector(config)
        try:
            model.load_state_dict(bundle.state_dict, strict=True)
        except (RuntimeError, KeyError) as error:
            raise DetectorUnavailableError(
                f"{path} does not fit a {config.backbone} detector over "
                f"{list(config.bands)}: {error}"
            ) from error

        model.eval().to(device)
        return LoadedModel(model=model, bundle=bundle, device=device, torch=torch)

    def clear(self) -> None:
        """Drop every loaded model."""
        with self._lock:
            self._loaded.clear()


MODELS: Final[ModelCache] = ModelCache()


def resolve_device(preferred: str | None = None) -> str:
    """Pick the device to run on, falling back to CPU rather than failing.

    A detector that refuses to run without a GPU would make the whole change path
    unavailable on a laptop, which is the wrong trade: 25 tiles on CPU is slow,
    and slow is a great deal better than absent.
    """
    if preferred and preferred != "auto":
        return preferred
    try:
        import torch
    except ImportError:
        return "cpu"
    try:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except (RuntimeError, AssertionError):  # pragma: no cover - driver-dependent
        return "cpu"


def checkpoint_available() -> bool:
    """True when a servable checkpoint exists, probed without loading it.

    Cheap and offline, like the VLM's backend probe: it runs once per process at
    registry load, and it must not read 50 MB of tensors to answer.
    """
    try:
        from satquery.training.cd.checkpoint import find_checkpoint
    except ImportError:  # pragma: no cover - broken install
        return False
    return find_checkpoint() is not None


# --------------------------------------------------------------------- the tool


def _bands_for(pre: ImageBundle, post: ImageBundle, wanted: Sequence[str]) -> list[str]:
    """The model's bands, if both epochs carry them.

    Raises:
        MissingInputError: An epoch cannot supply a band the model needs. There
            is no substitution here on purpose — feeding green into the channel a
            model learned as red produces a confident, wrong mask.
    """
    missing = sorted({band for band in wanted if not (pre.has(band) and post.has(band))})
    if missing:
        raise MissingInputError(
            f"the model needs {', '.join(wanted)} on both epochs; "
            f"{', '.join(missing)} is not resolvable"
        )
    return list(wanted)


def _otsu_threshold(probabilities: Array, valid: Mask) -> float:
    """Re-derive the operating point from this scene's own probability field."""
    finite = probabilities[valid & np.isfinite(probabilities)]
    if finite.size < 2 or float(finite.max() - finite.min()) < 1e-6:
        return 0.5
    return float(threshold_otsu(finite))  # type: ignore[no-untyped-call]


def _resolve_threshold(
    requested: Any, calibrated: float, probabilities: Array, valid: Mask
) -> tuple[float, str]:
    """Decide the operating point and say where it came from.

    Precedence: an explicit numeric override, then ``otsu`` recalibration, then
    the value the checkpoint was calibrated to. The last is the default because
    it is the only one measured against ground truth.
    """
    if isinstance(requested, str) and requested.strip().lower() == OTSU:
        return _otsu_threshold(probabilities, valid), OTSU
    if isinstance(requested, int | float) and not isinstance(requested, bool):
        return float(requested), "explicit"
    return float(calibrated), "checkpoint_calibrated"


class SiameseChangeDetector:
    """Trained bi-temporal change detection with tiled, blended inference."""

    name = NAME

    def __init__(self, checkpoint: Path | None = None, models: ModelCache | None = None) -> None:
        """Optionally pin a checkpoint and a cache; production resolves both."""
        self.checkpoint = checkpoint
        self.models = models or MODELS

    def _resolve_checkpoint(self) -> Path:
        """Locate the checkpoint to serve.

        Raises:
            DetectorUnavailableError: There is no checkpoint on this machine.
        """
        if self.checkpoint is not None:
            return self.checkpoint
        from satquery.training.cd.checkpoint import CHECKPOINT_ENV, find_checkpoint

        found = find_checkpoint()
        if found is None:
            raise DetectorUnavailableError(
                f"no change-detection checkpoint found. Train one with "
                f"scripts/train_cd.py, or point {CHECKPOINT_ENV} at a bundle."
            )
        return found

    def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
        """Detect change across the pair and emit the mask, overlay and payload.

        Raises:
            MissingInputError: The pair does not resolve the model's bands, or
                the two epochs did not render onto a common grid.
            DetectorUnavailableError: No checkpoint could be served.
            ToolError: The tiling was misconfigured, or the model raised.
        """
        pre, post = ctx.pair()
        device = resolve_device(str(params.get("device", "auto")))
        loaded = self.models.get(self._resolve_checkpoint(), device)

        size_px = min(int(params.get("size_px", INFERENCE_SIZE_PX)), MAX_INFERENCE_SIZE_PX)
        plan = self._plan(params, size_px)

        sar = pre.modality is Modality.SAR and post.modality is Modality.SAR
        wanted = loaded.bands if not sar or "vv" not in loaded.bands else _SAR_BANDS
        bands = _bands_for(pre, post, wanted)

        pre_stack = ctx.pixels.read(pre, bands, size=size_px)
        post_stack = ctx.pixels.read(post, bands, size=size_px)
        if pre_stack.stack.shape != post_stack.stack.shape:
            raise MissingInputError("the two epochs did not render onto a common grid")

        try:
            probabilities = predict_tiled(
                pre_stack.stack, post_stack.stack, loaded.predict, plan
            )
        except TilingError as error:
            raise ToolError(f"tiled inference could not run: {error}") from error

        valid: Mask = pre_stack.valid & post_stack.valid
        threshold, rule = _resolve_threshold(
            params.get("threshold"), loaded.bundle.threshold, probabilities, valid
        )
        mask: Mask = valid & np.isfinite(probabilities) & (probabilities >= threshold)

        scalars, notes = self._measure(probabilities, mask, valid, threshold)
        drafts, payload = change_drafts(
            mask=mask,
            post=post,
            post_stack=post_stack,
            stats={
                "method": "siamese",
                "threshold": round(threshold, 6),
                "threshold_rule": rule,
                "changed_pixel_count": int(mask.sum()),
                "valid_pixel_count": int(valid.sum()),
                "checkpoint": loaded.bundle.version,
            },
        )

        return ToolResult(
            status=ToolStatus.OK,
            scalars=scalars,
            artifacts=drafts,
            data={"change_mask": payload},
            params={
                "threshold": round(threshold, 6),
                "threshold_rule": rule,
                "tile": plan.tile,
                "overlap": plan.overlap,
                "size_px": size_px,
                "tiles": plan.count(size_px, size_px),
                "tile_bytes": peak_tile_bytes(plan, len(bands)),
                "bands": bands,
                "device": loaded.device,
                "checkpoint": loaded.bundle.version,
                "encoder_source": loaded.bundle.encoder_source,
                "trained_gsd_m": loaded.bundle.gsd_m,
            },
            confidence=self._confidence(loaded, pre, post),
            device=Device.ROCM_0 if loaded.device.startswith("cuda") else Device.CPU,
            notes=notes,
        )

    def _plan(self, params: Mapping[str, Any], size_px: int) -> TilePlan:
        """Build the tile plan, clamping a tile larger than the canvas.

        Raises:
            ToolError: The tiling parameters cannot cover a raster.
        """
        tile = min(int(params.get("tile", DEFAULT_TILE_PX)), size_px)
        overlap = int(params.get("overlap", DEFAULT_OVERLAP_PX))
        try:
            return TilePlan(tile=tile, overlap=min(overlap, max(0, tile - 1)))
        except TilingError as error:
            raise ToolError(str(error)) from error

    def _measure(
        self, probabilities: Array, mask: Mask, valid: Mask, threshold: float
    ) -> tuple[dict[str, float | str], list[str]]:
        """Summarise the probability field into the scalars the registry declares."""
        valid_px = int(valid.sum())
        changed_px = int(mask.sum())
        inside = probabilities[valid & np.isfinite(probabilities)]

        # The registry declares mean_change_logit, so a logit is what it gets:
        # reporting a probability under that name would let a citation resolve a
        # number against a scalar that does not mean what it says.
        clipped = np.clip(inside, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)) if clipped.size else np.zeros(0, np.float32)

        scalars: dict[str, float | str] = {
            "changed_area_pct": (
                round(100.0 * changed_px / valid_px, 4) if valid_px else 0.0
            ),
            "changed_pixel_count": changed_px,
            "threshold": round(threshold, 6),
            "mean_change_logit": round(float(logits.mean()), 6) if logits.size else 0.0,
        }

        notes: list[str] = []
        if valid_px == 0:
            notes.append("the two epochs share no valid pixel; the mask is empty")
        elif changed_px == 0:
            notes.append(f"no pixel reached the {threshold:.3f} threshold")
        return scalars, notes

    def _confidence(self, loaded: LoadedModel, pre: ImageBundle, post: ImageBundle) -> float:
        """Score the run against what the checkpoint was actually validated on.

        A model trained at 0.5 m and applied to 10 m imagery is doing something
        the F1 in its own metrics does not describe, and the confidence has to say
        so — that resolution gap is the whole point of training two of these.
        """
        base = float(loaded.bundle.metrics.get("f1", 0.85))
        trained_gsd = loaded.bundle.gsd_m
        gsds = [g for g in (pre.manifest.gsd_m, post.manifest.gsd_m) if g]
        if trained_gsd and gsds:
            ratio = max(max(gsds) / trained_gsd, trained_gsd / min(gsds))
            if ratio > 4.0:
                base *= 0.75
            elif ratio > 2.0:
                base *= 0.90
        return round(min(0.99, max(0.10, base)), 4)


__all__ = [
    "INFERENCE_SIZE_PX",
    "SELF_CHECK_ATTEMPTS",
    "SELF_CHECK_TOLERANCE",
    "MAX_INFERENCE_SIZE_PX",
    "NAME",
    "OTSU",
    "DetectorUnavailableError",
    "LoadedModel",
    "ModelCache",
    "SiameseChangeDetector",
    "checkpoint_available",
    "resolve_device",
]
