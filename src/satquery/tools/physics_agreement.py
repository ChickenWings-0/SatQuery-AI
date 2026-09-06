"""``physics_agreement`` — deterministic optical/SAR consistency rules.

Master.md §6.3 asks for a rule layer over the spectral indices and sigma-nought
that produces explainable statements like *"NDVI 0.72 with sigma-nought VV
-18 dB -> vegetated, not built-up; sensors agree."* It is deterministic,
sensor-independent, needs no weights, and cannot hallucinate — which is why it is
also the declared fallback for the learned ``crossmodal_consistency`` tool.

**The bare-soil / built-up confusion is the reason this tool exists.** NDBI is
the normalised difference of SWIR and NIR, and *both* dry bare soil and concrete
are bright in SWIR and dark in NIR. Optical alone therefore cannot separate a
quarry from a housing estate: ``spectral_index_analyzer`` reports the union of
the two as ``built_up_fraction_pct`` and has no way to do better. Radar can, and
for a physical reason rather than a statistical one:

* **Built-up** surfaces are vertical structures over a flat ground plane, which
  form a dihedral. Double-bounce returns most of the incident energy to the
  sensor with its polarisation preserved: sigma-nought VV is high, and so is the
  VV/VH ratio.
* **Bare soil** is a rough single surface. Surface scattering is co-polarised
  too, but far weaker — sigma-nought VV sits well below the built-up range.
* **Vegetation** is a volume of randomly oriented scatterers. Volume scattering
  *depolarises*, so VH rises towards VV and the ratio collapses, which is what
  distinguishes a canopy from a rough field at similar backscatter.

So the rules split the optically ambiguous set on backscatter, and report both
halves. That is the judging demo: the same pixels optical calls built-up, radar
resolves into a built-up part and a bare-soil part, with the physics stated.

Every threshold below is frozen rather than derived per scene. "The sensors
agree" has to mean the same thing on every acquisition the system has ever seen,
and a per-scene threshold would make it mean "agree relative to this image".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from satquery.render.colormaps import apply_colormap, normalise_fixed
from satquery.render.indices import ndbi, ndvi, ndwi, to_db, vv_vh_ratio_db
from satquery.schemas.enums import ArtifactType, Device, Modality, ToolStatus
from satquery.tools.base import (
    Array,
    ArtifactDraft,
    Bands,
    ImageBundle,
    Mask,
    MissingInputError,
    ToolContext,
    ToolResult,
)

NAME: Final[str] = "physics_agreement"

NDVI_VEGETATED: Final[float] = 0.30
"""Above this, a pixel is photosynthesising. The same level
``spectral_index_analyzer`` uses for ``vegetation_fraction_pct``, deliberately:
two tools disagreeing about what "vegetated" means makes their agreement
meaningless (AGENT_POLICY_DAG §7.2)."""

NDBI_BUILT_LIKE: Final[float] = 0.0
"""Above this, SWIR exceeds NIR — the *ambiguous* signature that covers concrete
and dry soil alike. Never read as "built-up" on its own."""

SIGMA0_WATER_DB: Final[float] = -18.0
"""Specular reflection away from the sensor. Open water is the darkest thing in
a radar scene, which is why this rule is the most reliable one here."""

SIGMA0_BUILT_UP_DB: Final[float] = -5.0
"""Double-bounce floor. Below it, a SWIR-bright pixel is a rough surface rather
than a structure."""

RATIO_VOLUME_DB: Final[float] = 5.0
"""VV/VH below this is volume scattering — a depolarising canopy rather than a
surface. Used to confirm vegetation, and to veto a built-up call on a pixel whose
brightness comes from a forest edge rather than a wall."""

class _Rule:
    """One surface class, as an optical claim and an independent radar claim.

    The two predicates must be *independent*: the whole value of an agreement
    figure is that two sensors reached the same conclusion separately, and a rule
    whose radar half was derived from its optical half would always agree.
    """

    __slots__ = ("name", "optical", "radar")

    def __init__(self, name: str, optical: Mask, radar: Mask) -> None:
        """Bind the two per-pixel predicates."""
        self.name = name
        self.optical = optical
        self.radar = radar


@dataclass(frozen=True)
class Confusion:
    """How radar resolved the pixels optical could not separate.

    ``ambiguous`` is the SWIR-bright, non-vegetated set — everything optical
    would report as built-up. ``built_up`` and ``bare_soil`` partition it on
    backscatter, and their sum is ``ambiguous`` minus whatever fell in neither
    because a pixel was not valid in both modalities.
    """

    ambiguous: Mask
    built_up: Mask
    bare_soil: Mask


def _split(ctx: ToolContext) -> tuple[ImageBundle, ImageBundle]:
    """Return (optical, sar) from the step's two inputs, whichever order they came in.

    Raises:
        MissingInputError: The step did not resolve one image of each modality.
    """
    optical = next((i for i in ctx.images if i.modality is Modality.OPTICAL), None)
    sar = next((i for i in ctx.images if i.modality is Modality.SAR), None)
    if optical is None or sar is None:
        raise MissingInputError("physics_agreement needs one optical and one SAR input")
    return optical, sar


def resolve_confusion(
    built_like: Mask,
    vegetation_index: Array,
    sigma0: Array,
    ratio: Array | None,
    valid: Mask,
    vegetated: float,
    built_db: float,
) -> Confusion:
    """Split the optically ambiguous set into built-up and bare soil on radar.

    Args:
        built_like: NDBI above its threshold — SWIR-bright, the ambiguous set.
        vegetation_index: NDVI, used only to exclude vegetation from the split.
        sigma0: Sigma-nought VV in dB.
        ratio: VV/VH in dB, or None on a single-polarisation acquisition.
        valid: Pixels finite and in-scene in both modalities.
        vegetated: NDVI level above which a pixel is not a candidate at all.
        built_db: Sigma-nought floor for double-bounce.

    Returns:
        The ambiguous set and its two radar-resolved halves.

    A single-polarisation SAR image still resolves the confusion: sigma-nought
    alone carries the double-bounce signal, and the ratio only corroborates it.
    That matters for RISAT-1, which the Cartosat/RISAT transfer case will hand us
    in single-pol form more often than not.
    """
    ambiguous: Mask = valid & built_like & (vegetation_index <= vegetated)
    bright: Mask = ambiguous & (sigma0 > built_db)
    if ratio is not None:
        # A dihedral is co-polarised. A SWIR-bright, radar-bright pixel that is
        # nonetheless depolarising is a canopy over a bright understorey, not a
        # wall, so it is not counted as built-up.
        bright = bright & (ratio >= RATIO_VOLUME_DB)
    return Confusion(ambiguous=ambiguous, built_up=bright, bare_soil=ambiguous & ~bright)


def _share(part: Mask, whole: Mask) -> float:
    """Percentage of *whole* covered by *part*, or 0.0 when *whole* is empty."""
    total = int(whole.sum())
    return round(100.0 * float(part.sum()) / float(total), 4) if total else 0.0


def _mean(values: Array, where: Mask) -> float | None:
    """Mean over the selected finite pixels, or None when there are none."""
    selected = values[where & np.isfinite(values)]
    return float(selected.mean()) if selected.size else None


def _statements(
    scalars: Mapping[str, float | str],
    confusion: Confusion,
    valid: Mask,
    has_vh: bool,
    built_db: float,
) -> list[str]:
    """The explainable sentences this tool exists to produce.

    Written here rather than by the VLM on purpose: these are the joint
    conclusions, and a conclusion assembled by a language model from two numbers
    is exactly the step the FactSheet architecture removes. The VLM may quote
    them; it does not derive them.
    """
    lines: list[str] = []
    ndvi_mean = scalars.get("ndvi_mean")
    ndbi_mean = scalars.get("ndbi_mean")
    sigma_mean = scalars.get("sigma0_vv_mean_db")
    ratio_mean = scalars.get("vv_vh_ratio_mean_db")

    if isinstance(ndvi_mean, float) and isinstance(sigma_mean, float):
        if ndvi_mean > NDVI_VEGETATED:
            verdict = (
                "vegetated, not built-up"
                if sigma_mean <= built_db
                else "vegetated, though radar is brighter than a canopy alone explains"
            )
        elif ndvi_mean < 0.0:
            verdict = "non-vegetated"
        else:
            verdict = "sparsely vegetated"
        ratio_clause = (
            f" and a VV/VH ratio of {ratio_mean:.1f} dB"
            if isinstance(ratio_mean, float)
            else ""
        )
        lines.append(
            f"NDVI {ndvi_mean:.2f} with sigma-nought VV {sigma_mean:.1f} dB"
            f"{ratio_clause} -> {verdict}."
        )

    ambiguous_px = int(confusion.ambiguous.sum())
    if ambiguous_px:
        built_share = _share(confusion.built_up, confusion.ambiguous)
        bare_share = _share(confusion.bare_soil, confusion.ambiguous)
        detail = (
            f"NDBI {ndbi_mean:.2f} flags {_share(confusion.ambiguous, valid):.1f}% of the "
            f"scene as SWIR-bright"
            if isinstance(ndbi_mean, float)
            else f"{_share(confusion.ambiguous, valid):.1f}% of the scene is SWIR-bright"
        )
        if bare_share > built_share:
            lines.append(
                f"{detail}, which optical alone reads as built-up. Radar resolves "
                f"{bare_share:.1f}% of it as bare soil (sigma-nought VV at or below "
                f"{built_db:.0f} dB, too weak for double-bounce) against "
                f"{built_share:.1f}% as genuinely built-up."
            )
        elif built_share > 0.0:
            lines.append(
                f"{detail}, and radar confirms {built_share:.1f}% of it as built-up: "
                f"sigma-nought VV above {built_db:.0f} dB is the double-bounce return "
                f"of vertical structure, which bare soil does not produce."
            )
        else:
            lines.append(
                f"{detail}, and radar confirms none of it as built-up — every "
                f"SWIR-bright pixel backscatters like a rough surface."
            )
        if not has_vh:
            lines.append(
                "The acquisition is single-polarisation, so the split rests on "
                "sigma-nought VV alone without VV/VH corroboration."
            )

    agreement = scalars.get("agreement_pct")
    if isinstance(agreement, float):
        verdict = (
            "the sensors agree"
            if agreement >= 70.0
            else "the sensors partly agree"
            if agreement >= 40.0
            else "the sensors disagree"
        )
        lines.append(f"Across the rules that fired, {verdict} on {agreement:.1f}% of claims.")
    return lines


class PhysicsAgreement:
    """Rule-based cross-modal consistency between an optical and a SAR scene."""

    name = NAME

    def run(self, ctx: ToolContext, params: Mapping[str, Any]) -> ToolResult:
        """Score how far the two sensors agree, and resolve what optical cannot.

        Raises:
            MissingInputError: The step did not resolve to an optical/SAR pair,
                the bands the rules need are absent, or the two modalities did
                not render onto a common grid.
        """
        optical, sar = _split(ctx)
        vegetated = float(params.get("ndvi_vegetated", NDVI_VEGETATED))
        water_db = float(params.get("sigma0_water_db", SIGMA0_WATER_DB))
        built_db = float(params.get("sigma0_built_up_db", SIGMA0_BUILT_UP_DB))

        if not optical.has("red", "nir"):
            raise MissingInputError("the optical input needs red and NIR to run the rules")
        if not sar.has("vv"):
            raise MissingInputError("the SAR input needs a VV channel to run the rules")

        opt, rad = self._read(ctx, optical, sar)
        has_vh = sar.has("vh")

        vegetation_index = ndvi(opt.band("nir"), opt.band("red"))
        water_index = ndwi(opt.band("green"), opt.band("nir")) if optical.has("green") else None
        built_index = ndbi(opt.band("swir1"), opt.band("nir")) if optical.has("swir1") else None
        sigma0 = to_db(rad.band("vv"))
        ratio = vv_vh_ratio_db(sigma0, to_db(rad.band("vh"))) if has_vh else None

        valid: Mask = (
            opt.valid & rad.valid & np.isfinite(vegetation_index) & np.isfinite(sigma0)
        )
        if ratio is not None:
            valid = valid & np.isfinite(ratio)

        rules = self._rules(vegetation_index, water_index, sigma0, ratio, vegetated, water_db)
        scalars, field = self._score(rules, valid)

        confusion = Confusion(
            ambiguous=np.zeros_like(valid),
            built_up=np.zeros_like(valid),
            bare_soil=np.zeros_like(valid),
        )
        if built_index is not None:
            confusion = resolve_confusion(
                built_like=np.isfinite(built_index) & (built_index > NDBI_BUILT_LIKE),
                vegetation_index=vegetation_index,
                sigma0=sigma0,
                ratio=ratio,
                valid=valid,
                vegetated=vegetated,
                built_db=built_db,
            )
            scalars["ndbi_ambiguous_pct"] = _share(confusion.ambiguous, valid)
            scalars["built_up_confirmed_pct"] = _share(confusion.built_up, confusion.ambiguous)
            scalars["bare_soil_reassigned_pct"] = _share(
                confusion.bare_soil, confusion.ambiguous
            )

        # The numbers the statements quote have to be in the FactSheet, or the
        # CitationValidator will strip the sentences that quote them (§4.4).
        for key, value in (
            ("ndvi_mean", _mean(vegetation_index, valid)),
            ("ndbi_mean", _mean(built_index, valid) if built_index is not None else None),
            ("ndwi_mean", _mean(water_index, valid) if water_index is not None else None),
            ("sigma0_vv_mean_db", _mean(sigma0, valid)),
            ("vv_vh_ratio_mean_db", _mean(ratio, valid) if ratio is not None else None),
        ):
            if value is not None:
                scalars[key] = round(value, 6)

        statements = _statements(scalars, confusion, valid, has_vh, built_db)
        notes: list[str] = []
        if built_index is None:
            notes.append(
                "no SWIR band on the optical input, so the bare-soil/built-up "
                "confusion could not be resolved"
            )
        if not has_vh:
            notes.append("single-polarisation SAR: no VV/VH ratio rule was applied")

        return ToolResult(
            status=ToolStatus.OK,
            scalars=scalars,
            artifacts=self._artifacts(field, opt, optical, sar, statements),
            params={
                "ndvi_vegetated": vegetated,
                "sigma0_water_db": water_db,
                "sigma0_built_up_db": built_db,
                "ratio_volume_db": RATIO_VOLUME_DB if ratio is not None else None,
                "rules": sorted(rule.name for rule in rules),
                "polarisations": "vv,vh" if has_vh else "vv",
            },
            data={"statements": statements},
            confidence=0.90,
            device=Device.CPU,
            notes=notes,
        )

    def _read(
        self, ctx: ToolContext, optical: ImageBundle, sar: ImageBundle
    ) -> tuple[Bands, Bands]:
        """Read both modalities onto the shared canvas.

        Raises:
            MissingInputError: The two did not render onto a common grid, which
                means co-registration did not run and every per-pixel rule below
                would be comparing unrelated ground.
        """
        optical_bands = ["red", "nir"] + [
            band for band in ("green", "swir1") if optical.has(band)
        ]
        sar_bands = ["vv"] + (["vh"] if sar.has("vh") else [])
        opt = ctx.pixels.read(optical, optical_bands)
        rad = ctx.pixels.read(sar, sar_bands)
        if opt.stack.shape[1:] != rad.stack.shape[1:]:
            raise MissingInputError("the two modalities did not render onto a common grid")
        return opt, rad

    def _rules(
        self,
        vegetation_index: Array,
        water_index: Array | None,
        sigma0: Array,
        ratio: Array | None,
        vegetated: float,
        water_db: float,
    ) -> list[_Rule]:
        """Assemble the rules the available bands support.

        A rule that cannot be evaluated is omitted rather than assumed true: an
        agreement score inflated by rules that never ran is worse than a score
        over fewer rules, and ``rules_fired`` says which it was.
        """
        rules = [
            _Rule(
                "vegetated",
                vegetation_index > vegetated,
                (ratio < RATIO_VOLUME_DB) if ratio is not None else (sigma0 >= water_db),
            )
        ]
        if water_index is not None:
            rules.append(
                _Rule(
                    "water",
                    water_index > 0.0,
                    sigma0 < water_db,
                )
            )
        return rules

    def _score(self, rules: list[_Rule], valid: Mask) -> tuple[dict[str, float | str], Array]:
        """Per-rule agreement percentages and the per-pixel agreement field."""
        scalars: dict[str, float | str] = {"rules_fired": len(rules)}
        agreements: list[Array] = []
        for rule in rules:
            claimed = valid & (rule.optical | rule.radar)
            agreed = valid & rule.optical & rule.radar
            scalars[f"{rule.name}_agree_pct"] = _share(agreed, claimed)
            agreements.append(np.where(claimed, agreed.astype(np.float32), np.nan))

        # The share of the fired rules that concurred, per pixel. A pixel no rule
        # claims stays nodata rather than counting as agreement — an empty scene
        # must not score 100 %. That is an all-NaN column, which is exactly what
        # nanmean warns about, so the mean is taken by hand: the executor runs
        # tools on a thread pool and warnings filters are process-global, so
        # suppressing the warning would be both racy and somebody else's problem.
        stacked = np.stack(agreements)
        claimed = np.count_nonzero(np.isfinite(stacked), axis=0)
        field: Array = np.where(
            claimed > 0, np.nansum(stacked, axis=0) / np.maximum(claimed, 1), np.nan
        ).astype(np.float32)
        finite = field[np.isfinite(field)]
        score = float(finite.mean()) if finite.size else 0.0
        scalars["agreement_score"] = round(score, 6)
        scalars["agreement_pct"] = round(100.0 * score, 4)
        return scalars, field

    def _artifacts(
        self,
        field: Array,
        opt: Bands,
        optical: ImageBundle,
        sar: ImageBundle,
        statements: list[str],
    ) -> list[Any]:
        """The agreement heatmap plus the statements, as citable evidence."""
        heatmap = apply_colormap(normalise_fixed(field, 0.0, 1.0), "RdYlGn", np.isfinite(field))
        drafts: list[Any] = [
            ArtifactDraft(
                key="agreement",
                type=ArtifactType.HEATMAP,
                label="Optical/SAR physical agreement",
                image=heatmap,
                raster=field,
                geometry=opt.geometry,
                stats={
                    "fixed_domain": [0.0, 1.0],
                    "optical_image": optical.id,
                    "sar_image": sar.id,
                },
            )
        ]
        if statements:
            drafts.append(
                ArtifactDraft(
                    key="statements",
                    type=ArtifactType.TEXT,
                    label="Joint optical/SAR findings",
                    inline={"statements": statements},
                )
            )
        return drafts


__all__ = [
    "NAME",
    "NDBI_BUILT_LIKE",
    "NDVI_VEGETATED",
    "RATIO_VOLUME_DB",
    "SIGMA0_BUILT_UP_DB",
    "SIGMA0_WATER_DB",
    "Confusion",
    "PhysicsAgreement",
    "resolve_confusion",
]
