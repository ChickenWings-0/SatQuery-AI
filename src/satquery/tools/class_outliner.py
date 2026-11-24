"""``class_outliner`` — outline one land-cover class from physics, and measure it.

"Outline the water bodies" needs two things a land-cover segmenter does not give
on its own: a mask that is right at the sensor's resolution, and a picture of
that mask *on the scene*. The LoveDA segmenter is trained on 0.3 m aerial RGB;
on a 10 m Sentinel-2 scene it smears classes together, and its class map is a
separate picture the analyst has to compare by eye. This tool takes the
physics route instead, the same fixed-threshold indices the rest of the system
already cites:

* **water** — optical: MNDWI (green vs SWIR1) above 0 with NDVI below 0.1 (the
  SWIR band is what keeps dark roofs and shadow out, which NDWI alone does
  not); NDWI above 0 where there is no SWIR band. SAR: VV backscatter below
  -18 dB after a 3x3 median filter (specular open water), the same threshold as
  ``sar_backscatter_analyzer``.
* **vegetation** — NDVI above 0.35.
* **built_up** — NDBI above 0 with NDVI below 0.2.

The mask is cleaned (one opening, regions under ``min_region_px`` dropped) and
drawn as an outline with a translucent fill over the true-colour view (or the
backscatter view for SAR) on the same 448 px canvas every other tool reads, so
it lines up with the evidence views exactly. Every number it reports — area,
fraction, region count, the largest region and where it sits — is measured
from that mask, so the answer can cite it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from skimage.filters import median
from skimage.measure import label
from skimage.morphology import dilation, erosion, opening

from satquery.render.composites import sar_backscatter, true_colour
from satquery.render.indices import ndbi, ndvi, ndwi, normalised_difference, to_db
from satquery.schemas.enums import ArtifactType, Device, Modality, ToolStatus
from satquery.tools.base import (
    ArtifactDraft,
    Draft,
    ImageBundle,
    Mask,
    MissingInputError,
    Rgb,
    ToolContext,
    ToolResult,
)

NAME: Final[str] = "class_outliner"

SAR_WATER_DB: Final[float] = -18.0
_SQUARE: Final[npt.NDArray[np.bool_]] = np.ones((3, 3), dtype=bool)

MIN_REGION_PX: Final[int] = 12
"""On the 448 px canvas: a pond of a few hundred metres at 10 m, never one pixel of noise."""

ALIASES: Final[dict[str, str]] = {
    "water": "water",
    "water_body": "water",
    "water_bodies": "water",
    "flood": "water",
    "floods": "water",
    "flooding": "water",
    "river": "water",
    "lake": "water",
    "reservoir": "water",
    "vegetation": "vegetation",
    "forest": "vegetation",
    "trees": "vegetation",
    "tree": "vegetation",
    "green": "vegetation",
    "mangrove": "vegetation",
    "mangroves": "vegetation",
    "crops": "vegetation",
    "built_up": "built_up",
    "building": "built_up",
    "buildings": "built_up",
    "urban": "built_up",
    "settlement": "built_up",
}

COLOURS: Final[dict[str, tuple[int, int, int]]] = {
    "water": (0, 210, 255),
    "vegetation": (120, 240, 60),
    "built_up": (255, 90, 60),
}

NOUNS: Final[dict[str, str]] = {
    "water": "Water bodies",
    "vegetation": "Vegetation",
    "built_up": "Built-up or bare areas",
}

CONFIDENCE: Final[dict[str, float]] = {
    "water_mndwi": 0.92,
    "water_ndwi": 0.85,
    "water_sar": 0.85,
    "vegetation_ndvi": 0.88,
    "built_up_ndbi": 0.72,
}


def canonical_class(raw: Any) -> str | None:
    """Map a slot value such as ``"water bodies"`` onto an outlinable class."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(key)


def location_of(mask: Mask) -> str:
    """Name the third of the canvas, ``"top-left"`` … ``"centre"``, a mask's mass sits in."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return "nowhere"
    height, width = mask.shape
    row = ("top", "middle", "bottom")[min(2, int(3 * rows.mean() / height))]
    col = ("left", "centre", "right")[min(2, int(3 * cols.mean() / width))]
    if row == "middle" and col == "centre":
        return "centre"
    if row == "middle":
        return f"middle-{col}"
    return f"{row}-{col}"


def clean(mask: Mask, valid: Mask, min_region_px: int) -> tuple[Mask, npt.NDArray[np.int32], int]:
    """Open once, drop small regions; return the mask, its labels and the region count."""
    opened = np.asarray(opening(mask & valid, _SQUARE), dtype=bool)
    labels = np.asarray(label(opened, connectivity=2), dtype=np.int32)  # type: ignore[no-untyped-call]
    count = int(labels.max())
    if count == 0:
        return opened, labels, 0
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    kept = np.isin(labels, 1 + np.flatnonzero(sizes >= min_region_px))
    relabelled = np.asarray(label(kept, connectivity=2), dtype=np.int32)  # type: ignore[no-untyped-call]
    return kept, relabelled, int(relabelled.max())


def draw_outline(base: Rgb, mask: Mask, colour: tuple[int, int, int]) -> Rgb:
    """Translucent fill plus a solid two-pixel outline of *mask* over *base*."""
    out = base.copy()
    tint = np.array(colour, dtype=np.float32)
    out[mask] = (0.55 * out[mask].astype(np.float32) + 0.45 * tint).astype(np.uint8)
    edge = mask & ~np.asarray(erosion(mask, _SQUARE), dtype=bool)
    edge = np.asarray(dilation(edge, _SQUARE), dtype=bool)
    out[edge] = colour
    return out


def _optical(ctx: ToolContext, image: ImageBundle, target: str) -> tuple[Mask, Mask, Rgb, Any, str]:
    if not image.has("red", "green", "blue"):
        raise MissingInputError(f"{image.id} lacks red, green and blue to draw on")
    wanted = ["blue", "green", "red"] + [b for b in ("nir", "swir1") if image.has(b)]
    stack = ctx.pixels.read(image, wanted)
    red, green, blue = stack.band("red"), stack.band("green"), stack.band("blue")
    base = true_colour(red, green, blue, stack.valid)
    with np.errstate(invalid="ignore", divide="ignore"):
        if target == "water":
            if image.has("swir1", "nir"):
                mndwi = normalised_difference(green, stack.band("swir1"))
                veg = ndvi(stack.band("nir"), red)
                return (mndwi > 0.0) & (veg < 0.1), stack.valid, base, stack, "water_mndwi"
            if image.has("nir"):
                return ndwi(green, stack.band("nir")) > 0.0, stack.valid, base, stack, "water_ndwi"
            raise MissingInputError(f"{image.id} lacks nir: water cannot be separated from shadow")
        if not image.has("nir"):
            raise MissingInputError(f"{image.id} lacks nir: no vegetation or built-up index")
        veg = ndvi(stack.band("nir"), red)
        if target == "vegetation":
            return veg > 0.35, stack.valid, base, stack, "vegetation_ndvi"
        if not image.has("swir1"):
            raise MissingInputError(f"{image.id} lacks swir1: no built-up index")
        built = ndbi(stack.band("swir1"), stack.band("nir"))
        return (built > 0.0) & (veg < 0.2), stack.valid, base, stack, "built_up_ndbi"


def _sar(ctx: ToolContext, image: ImageBundle, target: str) -> tuple[Mask, Mask, Rgb, Any, str]:
    if target != "water":
        raise MissingInputError(f"SAR backscatter does not separate {target} on its own")
    if not image.has("vv"):
        raise MissingInputError(f"{image.id} carries no VV channel")
    stack = ctx.pixels.read(image, ["vv"])
    vv_db = to_db(stack.band("vv"))
    filled = np.where(np.isfinite(vv_db), vv_db, 0.0)
    smooth = np.asarray(median(filled.astype(np.float32), _SQUARE), dtype=np.float32)  # type: ignore[no-untyped-call]
    base = sar_backscatter(vv_db, stack.valid)
    return (smooth < SAR_WATER_DB) & np.isfinite(vv_db), stack.valid, base, stack, "water_sar"


class ClassOutliner:
    """Outline the requested class on the scene and measure every region of it."""

    name = NAME

    def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
        """Build the class mask, draw it, and report its extent.

        Raises:
            MissingInputError: The requested class has no physics rule here, or
                the image lacks the bands that rule needs.
        """
        raw = ctx.slots.get("target_class") or params.get("target_class") or "water"
        target = canonical_class(raw)
        if target is None:
            raise MissingInputError(f"no physics rule outlines {raw!r}")
        min_px = int(params.get("min_region_px", MIN_REGION_PX))
        image = ctx.images[0]

        if image.modality is Modality.SAR:
            mask, valid, base, stack, method = _sar(ctx, image, target)
        else:
            mask, valid, base, stack, method = _optical(ctx, image, target)

        kept, labels, regions = clean(mask, valid, min_px)
        pixel_area = stack.pixel_area_m2
        valid_px = int(valid.sum()) or int(valid.size)
        sizes = (
            np.bincount(labels.ravel(), minlength=regions + 1)[1:].astype(np.float64)
            if regions
            else np.zeros(0)
        )

        prefix = target
        scalars: dict[str, float | str] = {
            f"{prefix}_outline_fraction_pct": round(100.0 * float(kept.sum()) / valid_px, 4),
            f"{prefix}_regions": float(regions),
            "outline_class": target,
            "outline_method": method,
        }
        if regions:
            largest = int(np.argmax(sizes)) + 1
            scalars[f"{prefix}_largest_region_location"] = location_of(labels == largest)
            scalars[f"{prefix}_extent_location"] = location_of(kept)
        if pixel_area is not None:
            scalars[f"{prefix}_outline_area_m2"] = round(float(kept.sum()) * pixel_area, 2)
            if regions:
                scalars[f"{prefix}_largest_region_m2"] = round(float(sizes.max()) * pixel_area, 2)

        noun = NOUNS[target]
        over = "backscatter" if image.modality is Modality.SAR else "true colour"
        drafts: list[Draft] = [
            ArtifactDraft(
                key="outline",
                type=ArtifactType.OVERLAY_PNG,
                label=f"{noun} outlined over {over} ({image.id})",
                image=draw_outline(base, kept, COLOURS[target]),
                geometry=stack.geometry,
                stats={
                    "source_image": image.id,
                    "outline_class": target,
                    "method": method,
                    "regions": regions,
                },
            )
        ]
        notes = [] if regions else [f"no {noun.lower()} found above {min_px} px"]
        return ToolResult(
            status=ToolStatus.OK,
            scalars=scalars,
            artifacts=drafts,
            params={"target_class": target, "min_region_px": min_px},
            confidence=CONFIDENCE[method],
            device=Device.CPU,
            notes=notes,
        )


__all__ = ["NAME", "ClassOutliner", "canonical_class", "draw_outline", "location_of"]
