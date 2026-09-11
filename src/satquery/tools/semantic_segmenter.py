"""``semantic_segmenter`` — SegFormer land cover, mapped onto the shared vocabulary.

Master.md §5 lists SegFormer/UPerNet trained on LoveDA and OpenEarthMap for the
segmentation half of requirement 2's option B. The model is one half of this
tool; the other half is the part that makes its output usable by everything else:

* **The label space is translated, not passed through.** LoveDA's seven classes
  are its own; ``change_statistics`` joins a class map against a change mask, the
  planner normalises query terms, and the FactSheet is keyed by class name — all
  three read ``configs/class_vocabulary.yaml``. A segmenter emitting "agriculture"
  where the rest of the system says "vegetation" produces per-class change numbers
  that silently never match. :data:`LOVEDA_TO_CANONICAL` is that translation, and
  it is the reason this tool is not simply a model call.
* **It is the declared fallback target for grounding.** ``text_grounding`` falls
  back here (AGENT_POLICY_DAG §5), because connected components over a class mask
  give boxes when the VLM cannot be reached — which is why :func:`class_boxes`
  lives here rather than in the counter.
* **It degrades to index thresholding.** Its own fallback is
  ``spectral_index_analyzer``: NDVI/NDWI/NDBI thresholds are a coarse class map,
  and a coarse one beats none.

The checkpoint is not vendored. Without it the tool raises
:class:`SegmenterUnavailableError`, capability matching substitutes the fallback,
and the trace says so.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from skimage.measure import label, regionprops

from satquery.schemas.enums import ArtifactType, Device, ToolStatus
from satquery.tools.base import (
    Array,
    ArtifactDraft,
    Bands,
    ClassMap,
    ImageBundle,
    MissingInputError,
    ToolContext,
    ToolError,
    ToolResult,
)

NAME: Final[str] = "semantic_segmenter"

CHECKPOINT_ENV: Final[str] = "SATQUERY_SEG_CHECKPOINT"
MODEL_ID: Final[str] = "nvidia/segformer-b2-finetuned-ade-512-512"
INPUT_PX: Final[int] = 512
MIN_REGION_PX: Final[int] = 32

IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)
DEFAULT_RESCALE: Final[float] = 1.0 / 255.0
"""The preprocessing a SegFormer checkpoint is trained under, used when the
checkpoint ships no ``preprocessor_config.json`` of its own. These are the
ImageNet statistics every published SegFormer uses, and the 1/255 that turns an
8-bit view into the unit range they were computed on."""

LOVEDA_CLASSES: Final[tuple[str, ...]] = (
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agriculture",
)
"""LoveDA's own label order, which is what a checkpoint trained on it emits."""

LOVEDA_TO_CANONICAL: Final[dict[str, str | None]] = {
    "background": None,
    "building": "built_up",
    "road": "road",
    "water": "water",
    "barren": "bare_soil",
    "forest": "vegetation",
    "agriculture": "vegetation",
}
"""LoveDA to ``configs/class_vocabulary.yaml``. ``None`` means unlabelled: LoveDA's
"background" is *everything the annotators did not label*, and folding it into a
named class would invent land cover. It becomes ``-1`` in the :class:`ClassMap`,
which every consumer already excludes from its per-class figures."""

CANONICAL_CLASSES: Final[tuple[str, ...]] = (
    "built_up",
    "road",
    "water",
    "bare_soil",
    "vegetation",
)
"""The label space this tool emits, in a fixed order so ``ClassMap.labels`` means
the same thing across runs and across checkpoints."""

PALETTE: Final[dict[str, tuple[int, int, int]]] = {
    "built_up": (228, 96, 78),
    "road": (140, 140, 148),
    "water": (58, 122, 204),
    "bare_soil": (198, 166, 106),
    "vegetation": (74, 158, 88),
}

SEGMENTATION_LABEL: Final[str] = "Land-cover segmentation"

_VIEW_BANDS: Final[tuple[str, ...]] = ("red", "green", "blue")


class SegmenterUnavailableError(ToolError):
    """No segmentation checkpoint could be served on this machine."""


@runtime_checkable
class Segmenter(Protocol):
    """A semantic segmentation model, reduced to the one call this tool makes."""

    classes: tuple[str, ...]

    def segment(self, rgb: Array) -> npt.NDArray[np.int16]:
        """Label a ``(3, H, W)`` stack, returning ``(H, W)`` indices into :attr:`classes`."""
        ...


# ------------------------------------------------------------------- the model


def _read_preprocessing(checkpoint: str) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """The rescale factor, mean and std the *checkpoint itself* declares.

    Taken from the checkpoint's ``preprocessor_config.json`` rather than
    hard-coded, because it is the checkpoint's own contract: a model fine-tuned
    under different statistics than we feed it produces a confident,
    systematically wrong map rather than an obvious failure. Falls back to the
    ImageNet constants every published SegFormer is trained under.
    """
    config = Path(checkpoint) / "preprocessor_config.json"
    if not config.is_file():
        return DEFAULT_RESCALE, IMAGENET_MEAN, IMAGENET_STD
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DEFAULT_RESCALE, IMAGENET_MEAN, IMAGENET_STD
    scale = float(payload.get("rescale_factor", DEFAULT_RESCALE))
    if not payload.get("do_rescale", True):
        scale = 1.0
    mean = tuple(float(v) for v in payload.get("image_mean", IMAGENET_MEAN))
    std = tuple(float(v) for v in payload.get("image_std", IMAGENET_STD))
    if not payload.get("do_normalize", True):
        mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    return scale, mean, std


class SegformerSegmenter:
    """The real model: a SegFormer checkpoint loaded once and kept on the device."""

    classes = LOVEDA_CLASSES

    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None) -> None:
        """Record what to load; the weights come up on the first :meth:`segment`."""
        self.checkpoint = str(checkpoint or os.environ.get(CHECKPOINT_ENV) or MODEL_ID)
        self.device = device or _resolve_device()
        self._lock = threading.Lock()
        self._model: Any = None
        self._torch: Any = None
        self._preprocessing: tuple[float, tuple[float, ...], tuple[float, ...]] | None = None

    def _load(self) -> tuple[Any, Any]:
        """Bring the weights up once per process.

        Raises:
            SegmenterUnavailableError: torch or transformers is absent, or the
                checkpoint could not be read. Never a network fetch: a tool that
                downloads on first request makes inference depend on the link.
        """
        with self._lock:
            if self._model is not None:
                return self._model, self._torch
            try:
                import torch
                from transformers import SegformerForSemanticSegmentation
            except ImportError as error:
                raise SegmenterUnavailableError(
                    f"the segmenter needs torch and transformers "
                    f"(`uv sync --extra vlm`): {error}"
                ) from error
            try:
                model = SegformerForSemanticSegmentation.from_pretrained(
                    self.checkpoint, local_files_only=True
                )
            except Exception as error:  # noqa: BLE001 - any load failure degrades
                raise SegmenterUnavailableError(
                    f"could not load a segmentation checkpoint from "
                    f"{self.checkpoint!r}: {error}"
                ) from error
            loaded: Any = model  # torch is an optional extra; see hf_backend.
            loaded.eval().to(self.device)
            self._preprocessing = _read_preprocessing(self.checkpoint)
            self._model, self._torch = model, torch
            return model, torch

    def segment(self, rgb: Array) -> npt.NDArray[np.int16]:
        """Label one stack, at the resolution it came in on.

        Raises:
            SegmenterUnavailableError: The model could not be run.
        """
        model, torch = self._load()
        scale, mean, std = self._preprocessing or (DEFAULT_RESCALE, IMAGENET_MEAN, IMAGENET_STD)
        with torch.no_grad():
            tensor = (
                torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32))
                .unsqueeze(0)
                .to(self.device)
            )
            # The view stack carries native raster values — 0..255 for the 8-bit
            # RGB renders this tool actually receives — while SegFormer was
            # trained on unit-scaled, ImageNet-standardised input. Feeding the
            # raw stack is a ~250x distribution shift: the model still returns a
            # full map, which is exactly why the error is worth being explicit
            # about here rather than leaving to the caller.
            shape = (1, -1, 1, 1)
            tensor = tensor * scale
            tensor = (tensor - torch.tensor(mean, device=self.device).reshape(shape)) / (
                torch.tensor(std, device=self.device).reshape(shape)
            )
            logits = model(pixel_values=tensor).logits
            # SegFormer decodes at a quarter of the input stride, so the logits
            # come back smaller than the image they describe.
            upsampled = torch.nn.functional.interpolate(
                logits, size=rgb.shape[1:], mode="bilinear", align_corners=False
            )
            predicted = upsampled.argmax(dim=1)[0]
        labels: npt.NDArray[np.int16] = predicted.cpu().numpy().astype(np.int16)
        return labels


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


def checkpoint_available() -> bool:
    """True when a segmentation checkpoint is present, probed without loading it.

    Only an explicitly configured checkpoint counts. The Hugging Face id in
    :data:`MODEL_ID` is a *default target*, not evidence of anything on disk, and
    treating it as availability would advertise a tool whose first request fails.
    """
    configured = os.environ.get(CHECKPOINT_ENV)
    return configured is not None and Path(configured).exists()


# ------------------------------------------------------------------ translation


def to_canonical(
    labels: npt.NDArray[np.int16], source_classes: Sequence[str]
) -> npt.NDArray[np.int16]:
    """Remap a model's own label indices onto :data:`CANONICAL_CLASSES`.

    Unmapped and background pixels become ``-1``, which every consumer of a
    :class:`~satquery.tools.base.ClassMap` already excludes rather than
    attributing to class 0.
    """
    lookup = np.full(len(source_classes), -1, dtype=np.int16)
    for index, name in enumerate(source_classes):
        canonical = LOVEDA_TO_CANONICAL.get(name, name if name in CANONICAL_CLASSES else None)
        if canonical in CANONICAL_CLASSES:
            lookup[index] = CANONICAL_CLASSES.index(canonical)
    clipped = np.clip(labels, 0, len(source_classes) - 1)
    remapped = lookup[clipped]
    return np.where(
        (labels >= 0) & (labels < len(source_classes)), remapped, np.int16(-1)
    ).astype(np.int16)


def colourise(labels: npt.NDArray[np.int16]) -> npt.NDArray[np.uint8]:
    """Render a canonical class map as an RGB image for the evidence gallery.

    Unlabelled pixels are left black rather than given a colour of their own: a
    palette entry for "we do not know" reads as a class in the legend.
    """
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for index, name in enumerate(CANONICAL_CLASSES):
        out[labels == index] = PALETTE[name]
    return out


def class_boxes(
    labels: npt.NDArray[np.int16], min_region_px: int = MIN_REGION_PX
) -> list[dict[str, Any]]:
    """Connected-component boxes per class — the grounding fallback path.

    ``text_grounding`` degrades to this tool, and a grounding step that produced
    a raster instead of boxes would break every consumer downstream of it. So the
    segmenter emits boxes too, from the components of each class mask.
    """
    boxes: list[dict[str, Any]] = []
    for index, name in enumerate(CANONICAL_CLASSES):
        mask = labels == index
        if not mask.any():
            continue
        components = label(mask, connectivity=2)  # type: ignore[no-untyped-call]
        for region in regionprops(components):  # type: ignore[no-untyped-call]
            if region.area < min_region_px:
                continue
            min_row, min_col, max_row, max_col = (int(v) for v in region.bbox)
            boxes.append(
                {
                    "id": f"box_{len(boxes)}",
                    "label": name,
                    "score": None,
                    "bbox_px": [min_col, min_row, max_col, max_row],
                    "area_px": int(region.area),
                }
            )
    return boxes


# --------------------------------------------------------------------- the tool


class SemanticSegmenter:
    """Land-cover segmentation onto the shared class vocabulary."""

    name = NAME

    def __init__(self, segmenter: Segmenter | None = None) -> None:
        """Optionally inject a model; production constructs SegFormer lazily."""
        self._segmenter = segmenter

    def segmenter(self) -> Segmenter:
        """The model to use, constructed on first use and then reused.

        Raises:
            SegmenterUnavailableError: No checkpoint is configured on this
                machine, which the executor turns into the declared
                ``spectral_index_analyzer`` fallback.
        """
        if self._segmenter is None:
            if not checkpoint_available():
                raise SegmenterUnavailableError(
                    f"no segmentation checkpoint found. Point {CHECKPOINT_ENV} at "
                    f"one; land-cover fractions still come from the "
                    f"spectral_index_analyzer fallback."
                )
            self._segmenter = SegformerSegmenter()
        return self._segmenter

    def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
        """Label the scene and report what fraction each class covers.

        Raises:
            MissingInputError: No input resolves the three bands a view needs.
            ToolError: No checkpoint could be served.
        """
        image = self._image(ctx)
        min_region_px = int(params.get("min_region_px", MIN_REGION_PX))
        bands = ctx.pixels.read(image, list(_VIEW_BANDS))

        raw = self.segmenter().segment(
            np.nan_to_num(bands.stack, nan=0.0, posinf=0.0, neginf=0.0)
        )
        if raw.shape != bands.valid.shape:
            raise ToolError(
                f"the segmenter returned {raw.shape} for a {bands.valid.shape} canvas"
            )
        labels = np.where(
            bands.valid, to_canonical(raw, self.segmenter().classes), np.int16(-1)
        ).astype(np.int16)

        class_map = ClassMap(
            labels=labels,
            classes=CANONICAL_CLASSES,
            geometry=bands.geometry,
            pixel_area_m2=bands.pixel_area_m2,
        )
        scalars = self._fractions(labels, bands, min_region_px)
        boxes = class_boxes(labels, min_region_px)

        return ToolResult(
            status=ToolStatus.OK,
            scalars=scalars,
            artifacts=[
                ArtifactDraft(
                    key="segmentation",
                    type=ArtifactType.SEGMENTATION,
                    label=SEGMENTATION_LABEL,
                    image=colourise(labels),
                    raster=labels,
                    geometry=bands.geometry,
                    stats={
                        "classes": list(CANONICAL_CLASSES),
                        "palette": {name: list(PALETTE[name]) for name in CANONICAL_CLASSES},
                        "unlabelled_value": -1,
                        "source_image": image.id,
                    },
                ),
                ArtifactDraft(
                    key="regions",
                    type=ArtifactType.BBOX_SET,
                    label="Land-cover regions",
                    inline={
                        "boxes": boxes,
                        "frame": {
                            "width": bands.geometry.width,
                            "height": bands.geometry.height,
                        },
                        "source_image": image.id,
                    },
                    geometry=bands.geometry,
                ),
            ],
            data={"segmentation": class_map, "boxes": boxes},
            params={
                "taxonomy": str(params.get("taxonomy", "loveda")),
                "min_region_px": min_region_px,
                "classes": list(CANONICAL_CLASSES),
            },
            confidence=0.87,
            device=Device.ROCM_0,
        )

    def _image(self, ctx: ToolContext) -> ImageBundle:
        """The optical image to label.

        Raises:
            MissingInputError: No input resolves red, green and blue. There is no
                SAR path here: a model trained on reflectance applied to
                backscatter produces a confident, meaningless map.
        """
        for image in ctx.images:
            if image.has(*_VIEW_BANDS):
                return image
        raise MissingInputError(
            f"{NAME} needs an image resolving {', '.join(_VIEW_BANDS)}"
        )

    def _fractions(
        self, labels: npt.NDArray[np.int16], bands: Bands, min_region_px: int
    ) -> dict[str, float | str]:
        """Per-class coverage, over the valid pixels only.

        Denominator is the valid extent rather than the canvas, so the padding a
        non-square raster is rendered into cannot dilute a class fraction — the
        same convention the index analyser uses (DATA_ADAPTATION_PLAN §2.3).
        """
        total = int(bands.valid.sum())
        scalars: dict[str, float | str] = {}
        present = 0
        for index, name in enumerate(CANONICAL_CLASSES):
            count = int((labels == index).sum())
            if count:
                present += 1
            scalars[f"{name}_fraction_pct"] = (
                round(100.0 * count / total, 4) if total else 0.0
            )
            if bands.pixel_area_m2 is not None and count:
                scalars[f"{name}_area_m2"] = round(count * bands.pixel_area_m2, 2)
        scalars["class_count"] = present
        scalars["unlabelled_pct"] = (
            round(100.0 * float((labels[bands.valid] < 0).sum()) / total, 4) if total else 0.0
        )
        scalars["min_region_px"] = min_region_px
        return scalars


__all__ = [
    "CANONICAL_CLASSES",
    "CHECKPOINT_ENV",
    "LOVEDA_CLASSES",
    "LOVEDA_TO_CANONICAL",
    "MODEL_ID",
    "NAME",
    "PALETTE",
    "SEGMENTATION_LABEL",
    "SegformerSegmenter",
    "Segmenter",
    "SegmenterUnavailableError",
    "SemanticSegmenter",
    "checkpoint_available",
    "class_boxes",
    "colourise",
    "to_canonical",
]
