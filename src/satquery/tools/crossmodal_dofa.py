"""``crossmodal_consistency`` — DOFA embedding agreement between optical and SAR.

Master.md §6.4 picks **DOFA** (``torchgeo.models.dofa_base_patch16_224``) for the
cross-modal half of mandatory requirement 4. What makes it the right encoder here
is its patch embedding: a hypernetwork generates the projection weights *from the
wavelength of each channel*, so one set of weights encodes Sentinel-2's thirteen
optical bands and Sentinel-1's two radar channels into the same representation
space. A conventional encoder needs one head per sensor and cannot compare across
them at all; DOFA compares them by construction, and the same property is what
lets it accept Cartosat-2S and RISAT-1 without retraining.

The measurement is a **cosine similarity between the two acquisitions' patch
embeddings**, taken per patch and resampled back onto the canvas. High similarity
means both sensors encode the same surface; low similarity means they encode
different ones, and *that* is the interesting signal — it is where radar sees
something optical does not.

Two design choices carry the tool's honesty:

* **The wavelengths are physical constants, not learned.** They are stated in
  :data:`WAVELENGTH_UM` in micrometres, including the C-band figure derived from
  Sentinel-1's 5.405 GHz carrier. A checkpoint trained against a different
  convention can override them per-request rather than silently mismatching.
* **The encoder is injected, never constructed at import.** The weights are a
  ~350 MB download this repository does not vendor, so an environment without
  them raises :class:`CrossModalUnavailableError` and the executor substitutes
  the declared ``physics_agreement`` fallback — a `DEGRADED` cross-modal answer
  rather than none. The same seam is what lets the tests exercise every line of
  the tiling, similarity and artifact logic against a stub.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

from satquery.render.colormaps import apply_colormap, normalise_fixed
from satquery.schemas.enums import ArtifactType, Device, Modality, ToolStatus
from satquery.tools.base import (
    Array,
    ArtifactDraft,
    ImageBundle,
    Mask,
    MissingInputError,
    ToolContext,
    ToolError,
    ToolResult,
)

NAME: Final[str] = "crossmodal_consistency"

MODEL_ID: Final[str] = "dofa_base_patch16_224"
PATCH_PX: Final[int] = 16
INPUT_PX: Final[int] = 224
"""DOFA-base is a ViT-B/16 over a 224 px input: 14x14 patches, 768 dimensions."""

WAVELENGTH_UM: Final[dict[str, float]] = {
    # Sentinel-2 band centres, MSI spec, in micrometres.
    "coastal": 0.443,
    "blue": 0.490,
    "green": 0.560,
    "red": 0.665,
    "rededge1": 0.705,
    "rededge2": 0.740,
    "rededge3": 0.783,
    "nir": 0.842,
    "nir08": 0.865,
    "watervapour": 0.945,
    "swir1": 1.610,
    "swir2": 2.190,
    # Sentinel-1 is C-band at 5.405 GHz, so lambda = c/f = 5.546 cm. Both
    # polarisations are the same carrier: what distinguishes VV from VH is the
    # transmit/receive polarisation, not the wavelength, and DOFA conditions on
    # wavelength alone — so they enter with identical values, exactly as they do
    # in the DOFA authors' own Sentinel-1 configuration.
    "vv": 55_465.0,
    "vh": 55_465.0,
    # Panchromatic sensors integrate the visible; Cartosat-2S PAN is 0.50-0.85.
    "pan": 0.675,
}
"""Channel centre wavelengths in micrometres, keyed by the logical band names the
alias table resolves (DATA_ADAPTATION_PLAN §2.2)."""

_OPTICAL_PREFERENCE: Final[tuple[str, ...]] = (
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
)
_SAR_PREFERENCE: Final[tuple[str, ...]] = ("vv", "vh")

AGREEMENT_LABEL: Final[str] = "DOFA cross-modal embedding agreement"


class CrossModalUnavailableError(ToolError):
    """DOFA could not be served here — no torchgeo, no torch, or no weights."""


@runtime_checkable
class Embedder(Protocol):
    """A wavelength-conditioned encoder, reduced to the one call this tool makes."""

    def embed(self, stack: Array, wavelengths: Sequence[float]) -> Array:
        """Encode a ``(C, H, W)`` stack into a ``(h, w, D)`` grid of patch tokens."""
        ...


# ------------------------------------------------------------------ the encoder


class DofaEmbedder:
    """The real encoder: torchgeo's DOFA, loaded once and kept on the device."""

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None) -> None:
        """Record what to load; the weights come up on the first :meth:`embed`."""
        self.model_id = model_id
        self.device = device or _resolve_device()
        self._lock = threading.Lock()
        self._model: Any = None
        self._torch: Any = None

    def _load(self) -> tuple[Any, Any]:
        """Bring the weights up once per process.

        Raises:
            CrossModalUnavailableError: torch or torchgeo is not installed, or
                the pretrained weights are not present on this machine.
        """
        with self._lock:
            if self._model is not None:
                return self._model, self._torch
            try:
                import torch
                from torchgeo.models import dofa_base_patch16_224
            except ImportError as error:
                raise CrossModalUnavailableError(
                    f"DOFA needs torch and torchgeo, which are not installed "
                    f"(`uv sync --extra cd`): {error}"
                ) from error
            try:
                model = dofa_base_patch16_224(weights=_dofa_weights())
            except Exception as error:  # noqa: BLE001 - any load failure degrades
                raise CrossModalUnavailableError(
                    f"could not load {self.model_id} weights: {error}"
                ) from error
            model.eval().to(self.device)
            self._model, self._torch = model, torch
            return model, torch

    def embed(self, stack: Array, wavelengths: Sequence[float]) -> Array:
        """Encode one stack into its patch-token grid.

        Raises:
            CrossModalUnavailableError: The encoder could not run.
        """
        model, torch = self._load()
        with torch.no_grad():
            tensor = torch.from_numpy(np.ascontiguousarray(stack)).unsqueeze(0).to(self.device)
            tokens = model.forward_features(tensor, wavelengths=list(wavelengths))
        array = tokens[0].float().cpu().numpy()
        return _as_grid(array)


def _dofa_weights() -> Any:
    """The pretrained weight enum, or None when torchgeo does not expose it."""
    try:
        from torchgeo.models import DOFABase16_Weights
    except ImportError:  # pragma: no cover - torchgeo version drift
        return None
    return DOFABase16_Weights.DOFA_MAE


def _resolve_device() -> str:
    """Prefer the accelerator, fall back to CPU rather than refusing to run."""
    try:
        import torch
    except ImportError:
        return "cpu"
    try:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except (RuntimeError, AssertionError):  # pragma: no cover - driver-dependent
        return "cpu"


def _as_grid(tokens: Array) -> Array:
    """Reshape a ViT token sequence into its square spatial grid.

    A leading class token is dropped when the remaining count is a perfect
    square, which is how a ViT-B/16 at 224 px presents itself: 197 tokens, one of
    them global.

    Raises:
        CrossModalUnavailableError: The sequence is not a square grid, meaning
            the model returned something other than patch tokens.
    """
    if tokens.ndim == 3:
        return tokens.astype(np.float32)
    count = tokens.shape[0]
    for candidate in (count, count - 1):
        side = int(round(candidate**0.5))
        if side * side == candidate and side > 0:
            return tokens[count - candidate :].reshape(side, side, -1).astype(np.float32)
    raise CrossModalUnavailableError(
        f"DOFA returned {count} tokens, which is not a square patch grid"
    )


def available() -> bool:
    """True when DOFA could be served here, probed without loading the weights.

    Import-only, like every other availability probe in the catalogue: it runs at
    registry load in every process, and must not pull 350 MB of tensors to answer.
    """
    try:
        import torchgeo.models  # noqa: F401
    except ImportError:
        return False
    return True


# ------------------------------------------------------------------- similarity


def cosine_agreement(optical: Array, radar: Array) -> Array:
    """Per-patch cosine similarity between two token grids, rescaled to ``[0, 1]``.

    Cosine rather than a distance because the embeddings are not calibrated to a
    common magnitude — DOFA normalises direction, not length, so the angle is the
    only part of the comparison that means anything across two modalities.

    Raises:
        MissingInputError: The two grids differ in shape, which would make the
            similarity a comparison of unrelated ground.
    """
    if optical.shape != radar.shape:
        raise MissingInputError(
            f"the two embeddings must share a patch grid, got "
            f"{optical.shape} and {radar.shape}"
        )
    left = optical / np.maximum(np.linalg.norm(optical, axis=-1, keepdims=True), 1e-8)
    right = radar / np.maximum(np.linalg.norm(radar, axis=-1, keepdims=True), 1e-8)
    similarity = np.sum(left * right, axis=-1)
    # Cosine spans [-1, 1]; the artifact and the scalar are both a 0-1 agreement,
    # so the rescale happens once, here, rather than at each consumer.
    agreement: Array = np.clip((similarity + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
    return agreement


def upsample(field: Array, height: int, width: int) -> Array:
    """Resample a patch-resolution field onto the canvas by nearest neighbour.

    Nearest rather than bilinear on purpose: each value *is* one patch's verdict,
    and interpolating between two patches would invent an agreement level that no
    patch produced.
    """
    if field.shape == (height, width):
        return field
    rows = np.minimum((np.arange(height) * field.shape[0]) // height, field.shape[0] - 1)
    cols = np.minimum((np.arange(width) * field.shape[1]) // width, field.shape[1] - 1)
    resampled: Array = field[np.ix_(rows, cols)].astype(np.float32)
    return resampled


def _channels(image: ImageBundle, preference: Sequence[str]) -> list[str]:
    """The bands to encode, in wavelength order, from what this image resolves.

    Raises:
        MissingInputError: The image resolves none of the bands DOFA knows a
            wavelength for.
    """
    chosen = [band for band in preference if image.has(band) and band in WAVELENGTH_UM]
    if not chosen:
        raise MissingInputError(
            f"{image.id} resolves none of {', '.join(preference)}, so DOFA has no "
            f"wavelength-tagged channel to encode"
        )
    return chosen


# --------------------------------------------------------------------- the tool


class CrossModalConsistency:
    """DOFA embedding agreement between a co-registered optical/SAR pair."""

    name = NAME

    def __init__(self, embedder: Embedder | None = None) -> None:
        """Optionally inject an encoder; production constructs DOFA lazily."""
        self._embedder = embedder

    def embedder(self) -> Embedder:
        """The encoder to use, constructed on first use and then reused."""
        if self._embedder is None:
            self._embedder = DofaEmbedder()
        return self._embedder

    def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
        """Embed both acquisitions and report where they encode the same surface.

        Raises:
            MissingInputError: The step did not resolve an optical/SAR pair, or
                the two did not render onto a common grid.
            ToolError: DOFA could not be served, which the executor turns into
                the declared ``physics_agreement`` fallback.
        """
        optical, sar = self._split(ctx)
        size = int(params.get("input_px", INPUT_PX))
        overrides = dict(params.get("wavelengths_um") or {})

        optical_bands = _channels(optical, _OPTICAL_PREFERENCE)
        sar_bands = _channels(sar, _SAR_PREFERENCE)
        opt = ctx.pixels.read(optical, optical_bands, size=size)
        rad = ctx.pixels.read(sar, sar_bands, size=size)
        if opt.stack.shape[1:] != rad.stack.shape[1:]:
            raise MissingInputError("the two modalities did not render onto a common grid")

        encoder = self.embedder()
        embeddings = [
            encoder.embed(
                np.nan_to_num(bands.stack, nan=0.0, posinf=0.0, neginf=0.0),
                [float(overrides.get(band, WAVELENGTH_UM[band])) for band in names],
            )
            for bands, names in ((opt, optical_bands), (rad, sar_bands))
        ]
        patches = cosine_agreement(embeddings[0], embeddings[1])

        canvas = ctx.pixels.read(optical, optical_bands)
        field = upsample(patches, canvas.geometry.height, canvas.geometry.width)
        valid: Mask = canvas.valid & upsample(
            _resize_mask(opt.valid & rad.valid, patches.shape),
            canvas.geometry.height,
            canvas.geometry.width,
        ).astype(bool)
        field = np.where(valid, field, np.nan).astype(np.float32)

        finite = field[np.isfinite(field)]
        scalars: dict[str, float | str] = {
            "agreement_mean": round(float(finite.mean()), 6) if finite.size else 0.0,
            "agreement_std": round(float(finite.std()), 6) if finite.size else 0.0,
            "agreement_min": round(float(finite.min()), 6) if finite.size else 0.0,
            "agreement_max": round(float(finite.max()), 6) if finite.size else 0.0,
            "patch_grid": f"{patches.shape[0]}x{patches.shape[1]}",
            "embedding_dim": int(embeddings[0].shape[-1]),
        }
        if finite.size:
            scalars["divergent_area_pct"] = round(
                100.0 * float((finite < 0.5).mean()), 4
            )

        return ToolResult(
            status=ToolStatus.OK,
            scalars=scalars,
            artifacts=[
                ArtifactDraft(
                    key="agreement",
                    type=ArtifactType.HEATMAP,
                    label=AGREEMENT_LABEL,
                    image=apply_colormap(
                        normalise_fixed(field, 0.0, 1.0), "RdYlGn", np.isfinite(field)
                    ),
                    raster=field,
                    geometry=canvas.geometry,
                    stats={
                        "fixed_domain": [0.0, 1.0],
                        "optical_image": optical.id,
                        "sar_image": sar.id,
                        "patch_grid": list(patches.shape),
                    },
                )
            ],
            params={
                "embedding": MODEL_ID,
                "input_px": size,
                "patch_px": int(params.get("patch_px", PATCH_PX)),
                "optical_bands": optical_bands,
                "sar_bands": sar_bands,
                "wavelengths_um": {
                    band: float(overrides.get(band, WAVELENGTH_UM[band]))
                    for band in (*optical_bands, *sar_bands)
                },
            },
            confidence=0.85,
            device=Device.ROCM_0,
        )

    def _split(self, ctx: ToolContext) -> tuple[ImageBundle, ImageBundle]:
        """Return (optical, sar) from the step's two inputs, in either order.

        Raises:
            MissingInputError: The step did not resolve one image of each
                modality — which for this tool is not a degradation but a
                category error, since there is no cross-modal comparison to make.
        """
        optical = next((i for i in ctx.images if i.modality is Modality.OPTICAL), None)
        sar = next((i for i in ctx.images if i.modality is Modality.SAR), None)
        if optical is None or sar is None:
            raise MissingInputError(f"{NAME} needs one optical and one SAR input")
        return optical, sar


def _resize_mask(valid: Mask, shape: tuple[int, ...]) -> Array:
    """Reduce a pixel validity mask onto the patch grid.

    A patch counts as valid when any of its pixels are: DOFA has already mixed
    the whole patch into one token, so a token backed by a partly valid patch is
    still a measurement of real ground.
    """
    rows = np.minimum((np.arange(valid.shape[0]) * shape[0]) // valid.shape[0], shape[0] - 1)
    cols = np.minimum((np.arange(valid.shape[1]) * shape[1]) // valid.shape[1], shape[1] - 1)
    grid = np.zeros(shape[:2], dtype=bool)
    np.logical_or.at(grid, (rows[:, None], cols[None, :]), valid)
    return grid.astype(np.float32)


__all__ = [
    "AGREEMENT_LABEL",
    "INPUT_PX",
    "MODEL_ID",
    "NAME",
    "PATCH_PX",
    "WAVELENGTH_UM",
    "CrossModalConsistency",
    "CrossModalUnavailableError",
    "DofaEmbedder",
    "Embedder",
    "available",
    "cosine_agreement",
    "upsample",
]
