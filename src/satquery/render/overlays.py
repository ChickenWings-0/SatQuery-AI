"""Compositing evidence onto a basemap view (Master.md §8 Phase 2).

An overlay is the artifact that makes a result legible: a change mask alone is a
black-and-white rectangle, but the same mask at 45 % alpha over the post-change
true-colour view is the picture a judge can read in one glance.

Overlays composite in uint8 RGB space over an already-rendered view, so they
inherit that view's geometry exactly and can be stacked.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

from satquery.render.colormaps import apply_colormap, normalise_fixed

Rgb = npt.NDArray[np.uint8]
Mask = npt.NDArray[np.bool_]

CHANGE_COLOUR: Final[tuple[int, int, int]] = (255, 0, 0)
CHANGE_ALPHA: Final[float] = 0.45
"""Frozen by DATA_ADAPTATION_PLAN §2.1: the CHANGE view is red at 45 % alpha."""


def _check_alpha(alpha: float) -> float:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return alpha


def _blend(base: Rgb, colour: npt.NDArray[np.float32], where: Mask, alpha: float) -> Rgb:
    """Alpha-blend *colour* into *base* only where *where* is true."""
    out = base.astype(np.float32, copy=True)
    selected = np.broadcast_to(where[..., None], out.shape)
    blended = out * (1.0 - alpha) + colour * alpha
    out = np.where(selected, blended, out)
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_mask(
    base: Rgb,
    mask: Mask,
    colour: tuple[int, int, int] = CHANGE_COLOUR,
    alpha: float = CHANGE_ALPHA,
) -> Rgb:
    """Tint the pixels of *base* selected by *mask*.

    Args:
        base: An ``(H, W, 3)`` uint8 view to composite onto.
        mask: Boolean selection of the same height and width.
        colour: Fill colour, defaulting to the frozen change red.
        alpha: Opacity, defaulting to the frozen 0.45.

    Returns:
        A new ``(H, W, 3)`` uint8 image; *base* is not modified.
    """
    _check_alpha(alpha)
    if mask.shape != base.shape[:2]:
        raise ValueError(f"mask {mask.shape} does not match base {base.shape[:2]}")
    fill = np.array(colour, dtype=np.float32)
    return _blend(base, fill, mask.astype(bool), alpha)


def change_overlay(post_view: Rgb, change_mask: Mask) -> Rgb:
    """The ``CHANGE`` artifact: the change mask in red over the post-change view.

    Runtime-only — it is evidence for the frontend, never a training input.
    """
    return overlay_mask(post_view, change_mask, CHANGE_COLOUR, CHANGE_ALPHA)


def overlay_heatmap(
    base: Rgb,
    values: npt.NDArray[np.float32],
    colormap: str,
    domain: tuple[float, float],
    alpha: float = 0.6,
    valid: Mask | None = None,
) -> Rgb:
    """Composite a continuous field over a basemap on a **fixed** domain.

    The domain is required rather than inferred: an overlay whose colours shifted
    with each scene's contrast would be unreadable as evidence.
    """
    _check_alpha(alpha)
    if values.shape != base.shape[:2]:
        raise ValueError(f"heatmap {values.shape} does not match base {base.shape[:2]}")
    coloured = apply_colormap(normalise_fixed(values, *domain), colormap, valid)
    finite = np.isfinite(values)
    where = finite if valid is None else (finite & valid)
    return _blend(base, coloured.astype(np.float32), where, alpha)


def draw_boxes(
    base: Rgb,
    boxes: list[tuple[int, int, int, int]],
    colour: tuple[int, int, int] = (255, 255, 0),
    width: int = 2,
) -> Rgb:
    """Stroke axis-aligned boxes onto a view.

    Boxes are ``(x_min, y_min, x_max, y_max)`` in pixels, top-left origin, with
    the maxima exclusive — the contract's pixel-box convention (§1). Boxes are
    clipped to the canvas rather than rejected, so a detection that runs off the
    edge still draws the part that is visible.
    """
    if width < 1:
        raise ValueError(f"stroke width must be at least 1, got {width}")
    out = base.copy()
    height, canvas_width = base.shape[:2]
    fill = np.array(colour, dtype=np.uint8)

    for x_min, y_min, x_max, y_max in boxes:
        x0 = max(0, min(int(x_min), canvas_width))
        x1 = max(0, min(int(x_max), canvas_width))
        y0 = max(0, min(int(y_min), height))
        y1 = max(0, min(int(y_max), height))
        if x1 <= x0 or y1 <= y0:
            continue
        out[y0 : min(y0 + width, y1), x0:x1] = fill  # top
        out[max(y1 - width, y0) : y1, x0:x1] = fill  # bottom
        out[y0:y1, x0 : min(x0 + width, x1)] = fill  # left
        out[y0:y1, max(x1 - width, x0) : x1] = fill  # right
    return out
