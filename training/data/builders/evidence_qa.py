#!/usr/bin/env python
"""Generate the synthetic ``evidence_qa`` source (DATA_ADAPTATION_PLAN §4.6).

    uv run python training/data/builders/evidence_qa.py --count 3000
    uv run python training/data/builders/evidence_qa.py --count 3000 --append

*Why this source exists:* none of the five public corpora teach citation. They
teach remote-sensing vocabulary and task formats, and every answer in them is
unsourced prose. Without explicit supervision the adapter learns to *sound* like
an analyst while still inventing numbers — which defeats the FactSheet-constrained
generation the whole architecture is built on. Because these samples are
generated *from* a FactSheet, every number in every answer is correct and
attributable by construction.

**What is synthetic here, and what is not.** §4.6 generates this source from the
FactSheets the Phase 2 renderer measures for real patches. That pass has not run,
so the sheets below are *plausible mock measurements* drawn from scene
archetypes — dense forest, cropland, urban, open water, bare soil, and
bi-temporal change — with internally consistent physics: a sheet that says NDVI
0.78 does not also say NDBI 0.30, and open water carries both a negative NDVI and
a σ⁰_VV near -20 dB. The *citation behaviour* this teaches is exactly right: the
answer copies a number from the sheet it was given and tags it with that sheet's
key, and the CitationValidator agrees.

**What it cannot teach is physics.** The views these samples point at are the
solid-black placeholders from ``PLACEHOLDER_VIEWS.json``, so the pairing between
a rendered NDVI heatmap and the value 0.78 is absent — worse, a corpus of black
squares paired with confident numbers actively teaches the model that the image
is irrelevant to the answer. Use this to exercise the pipeline and to hold the
citation format; regenerate it from real FactSheets, against real rendered views,
before any adapter that gets submitted.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from satquery.render.views import ViewId  # noqa: E402
from satquery.schemas.enums import ImageRole, Modality, PairType  # noqa: E402
from satquery.training.corpus_builder import (  # noqa: E402
    CorpusSample,
    CorpusSource,
    SampleMeta,
    SourceView,
    build_evidence_qa,
    write_jsonl,
)

DEFAULT_COUNT: Final[int] = 3_000
"""§5's target for this source: 4.6 % of the 65,000-sample corpus."""

DEFAULT_MANIFEST: Final[Path] = Path("data/processed/views/PLACEHOLDER_VIEWS.json")
DEFAULT_OUT_DIR: Final[Path] = Path("data/processed/corpus")
REFUSAL_PROBABILITY: Final[float] = 0.4
"""How often a scene also yields a refusal sample, landing them near 15 % of the
source. Rule 3 of the system prompt says an unmeasured quantity is declined
rather than guessed, and a corpus where every question has a number behind it
teaches the opposite — but one where a third of the answers decline teaches a
model to reach for "this system did not measure it" when the sheet in front of it
would have supported an answer. The headline metric is cited answers, not
abstentions."""

VAL_EVERY: Final[int] = 10
"""One scene in ten feeds ``val.jsonl``. Deterministic rather than random so two
builds of one seed produce the same split."""

OPTICAL_SENSOR: Final[str] = "Sentinel-2 (BigEarthNet-v2, 12-band)"
SAR_SENSOR: Final[str] = "Sentinel-1 GRD"
GSD_M: Final[float] = 10.0

CROSS_MODAL_ORDER: Final[tuple[str, ...]] = (
    "TC",
    "FCIR",
    "NDVI",
    "NDBI",
    "SARFC",
    "SARDB",
)
"""§2.4's CROSS_MODAL slot order, exactly — six views, optical then radar."""

BI_TEMPORAL_ORDER: Final[tuple[str, ...]] = ("TC", "FCIR", "NDBI")
"""§2.4's BI_TEMPORAL optical stack, each rendered pre and post — six views."""

_SAR_SLOTS: Final[frozenset[str]] = frozenset({"SARFC", "SARDB"})

_NDVI: Final[str] = "spectral_index_analyzer.ndvi_mean"
_NDBI: Final[str] = "spectral_index_analyzer.ndbi_mean"
_NDWI: Final[str] = "spectral_index_analyzer.ndwi_mean"
_VV: Final[str] = "sar_backscatter_analyzer.sigma0_vv_db_mean"
_RATIO: Final[str] = "sar_backscatter_analyzer.vv_vh_ratio_db_mean"
_CHANGED: Final[str] = "change_statistics.changed_area_pct"


class Archetype:
    """One scene type and the ranges its measurements fall in.

    Ranges rather than fixed values so 3,000 samples are not 6 sheets repeated,
    and *coupled* ranges rather than independent ones because the point of the
    agreement template is that the optical and radar evidence must be readable as
    the same scene. An NDVI of 0.8 beside a σ⁰_VV of -3 dB is not a hard case for
    the model to reason about; it is a sheet no instrument would produce.
    """

    def __init__(
        self,
        name: str,
        ndvi: tuple[float, float],
        ndbi: tuple[float, float],
        ndwi: tuple[float, float],
        vv_db: tuple[float, float],
        ratio_db: tuple[float, float],
    ) -> None:
        """Record the archetype's coupled ranges."""
        self.name = name
        self.ndvi = ndvi
        self.ndbi = ndbi
        self.ndwi = ndwi
        self.vv_db = vv_db
        self.ratio_db = ratio_db

    def sheet(self, rng: random.Random) -> dict[str, float | str]:
        """Draw one plausible FactSheet from this archetype."""
        return {
            _NDVI: round(rng.uniform(*self.ndvi), 2),
            _NDBI: round(rng.uniform(*self.ndbi), 2),
            _NDWI: round(rng.uniform(*self.ndwi), 2),
            _VV: round(rng.uniform(*self.vv_db), 1),
            _RATIO: round(rng.uniform(*self.ratio_db), 1),
        }


ARCHETYPES: Final[tuple[Archetype, ...]] = (
    # Dense canopy: high NDVI, volume scattering gives a moderate σ⁰ and a
    # comparatively low VV/VH ratio because VH is not weak.
    Archetype("dense_forest", (0.62, 0.86), (-0.34, -0.14), (-0.42, -0.20),
              (-12.0, -7.5), (4.0, 7.0)),
    Archetype("cropland", (0.34, 0.60), (-0.20, -0.04), (-0.35, -0.10),
              (-13.5, -9.0), (4.5, 8.0)),
    # Urban: strong double-bounce returns, and the one archetype whose NDBI and
    # radar agree that the scene is built up.
    Archetype("urban", (0.04, 0.22), (0.12, 0.36), (-0.40, -0.18),
              (-6.0, -1.5), (3.0, 6.0)),
    # Open water: the specular case — near-zero return, NDVI below zero.
    Archetype("open_water", (-0.38, -0.08), (-0.40, -0.18), (0.38, 0.72),
              (-22.0, -16.5), (6.0, 10.0)),
    # Bare soil: NDBI is ambiguous here, which is precisely the case §4.6's
    # agreement template exists to make the model reason about out loud.
    Archetype("bare_soil", (0.06, 0.24), (-0.02, 0.16), (-0.36, -0.14),
              (-11.5, -7.0), (5.0, 9.0)),
    Archetype("coastal_wetland", (0.18, 0.44), (-0.28, -0.08), (0.05, 0.40),
              (-17.0, -11.0), (5.5, 9.5)),
)

CHANGE_SHARE: Final[float] = 0.30
"""Fraction of scenes that are bi-temporal, and so carry the change template."""


def load_placeholder_dirs(manifest: Path) -> list[tuple[str, dict[str, str]]]:
    """Group the placeholder manifest into per-scene view stacks.

    Returns:
        ``(scene_id, {view_id: path})`` pairs, ordered, for scenes that have at
        least one view.

    Raises:
        SystemExit: The manifest is absent — run
            ``scripts/patch_dummy_images.py`` first.
    """
    if not manifest.is_file():
        raise SystemExit(
            f"{manifest} does not exist. Generate the placeholder views first:\n"
            "  uv run python scripts/patch_dummy_images.py"
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    scenes: dict[str, dict[str, str]] = {}
    for entry in payload.get("files", []):
        path = Path(str(entry["path"]))
        scenes.setdefault(path.parent.name, {})[path.stem] = str(path)
    return [(name, views) for name, views in sorted(scenes.items()) if views]


def views_for(
    pool: Sequence[str], pair_type: PairType, offset: int = 0
) -> list[SourceView]:
    """Build the view stack a sheet's claims require, from the placeholder pool.

    Every slot in §2.4's order is filled, drawing paths from *pool* and reusing
    them when the pool is smaller than the stack. Reuse is sound precisely
    because these are placeholders: the files carry no information, so pointing
    two slots at one black PNG loses nothing.

    What it buys is the property that does matter. The manifest's scenes are
    uneven — a VRSBench directory holds one ``TC.png``, a BigEarthNet one holds
    seven views — and drawing the stack from whatever a directory happened to
    contain gave 63 % of samples a single optical view beneath an answer
    discussing σ⁰_VV. A label index that names radar the model was never shown is
    a worse lesson than a blank image: it teaches that the prose need not follow
    from the views at all, which is the failure this whole source exists to
    prevent.
    """
    if not pool:
        raise SystemExit("no placeholder view files to draw from")

    def path(index: int) -> str:
        return pool[(offset + index) % len(pool)]

    if pair_type is PairType.BI_TEMPORAL:
        # §2.4's BI_TEMPORAL optical order: each view pre and post, adjacent, so
        # the two halves of a comparison sit next to one another.
        views: list[SourceView] = []
        for index, slot in enumerate(BI_TEMPORAL_ORDER):
            for half, (role, when) in enumerate(
                (
                    (ImageRole.PRE, datetime(2019, 3, 14, tzinfo=UTC)),
                    (ImageRole.POST, datetime(2021, 7, 2, tzinfo=UTC)),
                )
            ):
                views.append(
                    SourceView(
                        ViewId(slot),
                        path(index * 2 + half),
                        Modality.OPTICAL,
                        OPTICAL_SENSOR,
                        role,
                        when,
                    )
                )
        return views
    return [
        SourceView(
            ViewId(slot),
            path(index),
            Modality.SAR if slot in _SAR_SLOTS else Modality.OPTICAL,
            SAR_SENSOR if slot in _SAR_SLOTS else OPTICAL_SENSOR,
            ImageRole.SAR if slot in _SAR_SLOTS else ImageRole.OPTICAL,
        )
        for index, slot in enumerate(CROSS_MODAL_ORDER)
    ]


def change_sheet(archetype: Archetype, rng: random.Random) -> dict[str, float | str]:
    """A bi-temporal sheet: a changed fraction plus the NDBI either side of it.

    The radar scalars are dropped. These scenes are an optical pair, so no tool
    measured a backscatter for them — and leaving σ⁰ in the sheet let the
    agreement template fire on a sample whose six views are all optical, which
    is the model being shown prose about radar it was never given. Unavailability
    is explicit, never substituted (§2.2), and that rule applies to the synthetic
    sheets as much as to a Cartosat scene with no SWIR.
    """
    sheet = archetype.sheet(rng)
    sheet.pop(_VV, None)
    sheet.pop(_RATIO, None)
    before = round(rng.uniform(-0.30, 0.02), 2)
    after = round(before + rng.uniform(0.05, 0.35), 2)
    if rng.random() < 0.3:  # some scenes lose built-up surface rather than gain it
        before, after = after, before
    sheet[_CHANGED] = round(rng.uniform(1.5, 26.0), 2)
    sheet[f"{_NDBI}_pre"] = before
    sheet[f"{_NDBI}_post"] = after
    return sheet


def generate(
    count: int = DEFAULT_COUNT,
    manifest: Path = DEFAULT_MANIFEST,
    seed: int = 42,
) -> list[CorpusSample]:
    """Generate *count* citation-supervision samples.

    Scenes are drawn until the sample budget is met — each yields between two and
    four samples, depending on which templates its sheet supports.
    """
    rng = random.Random(seed)
    stacks = load_placeholder_dirs(manifest)
    if not stacks:
        raise SystemExit(f"{manifest} lists no usable view stacks")
    pool = [path for _, views in stacks for path in sorted(views.values())]

    samples: list[CorpusSample] = []
    scene_index = 0
    while len(samples) < count:
        archetype = ARCHETYPES[scene_index % len(ARCHETYPES)]
        is_change = rng.random() < CHANGE_SHARE
        pair_type = PairType.BI_TEMPORAL if is_change else PairType.CROSS_MODAL
        views = views_for(pool, pair_type, offset=scene_index * len(CROSS_MODAL_ORDER))
        sheet = change_sheet(archetype, rng) if is_change else archetype.sheet(rng)

        meta = SampleMeta(
            sensor=f"{OPTICAL_SENSOR} + {SAR_SENSOR}",
            gsd_m=GSD_M,
            labels=[archetype.name],
            split="val" if scene_index % VAL_EVERY == 0 else "train",
            source_split="synthetic",
        )
        samples.extend(
            build_evidence_qa(
                sample_id=f"evidence_qa:{archetype.name}:{scene_index:05d}",
                views=views,
                fact_sheet=sheet,
                pair_type=pair_type,
                meta=meta,
                rng=rng,
                refusal_probability=REFUSAL_PROBABILITY,
            )
        )
        scene_index += 1

    return samples[:count]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append into the corpus's train.jsonl / val.jsonl instead of writing "
        "evidence_qa.{train,val}.jsonl beside them.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate, report and write. Returns a process exit code."""
    args = parse_args(argv)
    samples = generate(args.count, args.manifest, args.seed)

    train = [sample for sample in samples if sample.meta.split == "train"]
    val = [sample for sample in samples if sample.meta.split == "val"]
    tasks: dict[str, int] = {}
    cited = 0
    for sample in samples:
        tasks[str(sample.task)] = tasks.get(str(sample.task), 0) + 1
        if "[" in sample.assistant:
            cited += 1

    print(f"generated       : {len(samples)} samples ({len(train)} train / {len(val)} val)")
    print(f"carrying a citation: {cited} ({cited / len(samples):.1%})")
    for task, number in sorted(tasks.items()):
        print(f"  {task:20s} {number:>5d}")

    if args.append:
        for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
            path = args.out_dir / name
            existing = path.read_text(encoding="utf-8") if path.is_file() else ""
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                handle.write(existing)
                for sample in rows:
                    handle.write(sample.to_json_line() + "\n")
            print(f"appended {len(rows):>5d} -> {path}")
    else:
        for name, rows in (("evidence_qa.train.jsonl", train), ("evidence_qa.val.jsonl", val)):
            written = write_jsonl(args.out_dir / name, rows)
            print(f"wrote    {written:>5d} -> {args.out_dir / name}")

    print(
        "\nThese sheets are plausible mock measurements against placeholder views: "
        "they hold the citation format, they do not teach what an NDVI heatmap "
        "looks like. Regenerate from the Phase 2 render pass before the submitted "
        "adapter."
    )
    assert all(sample.source is CorpusSource.EVIDENCE_QA for sample in samples)
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
