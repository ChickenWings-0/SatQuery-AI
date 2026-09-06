#!/usr/bin/env python
"""Build the Phase 7 instruction corpus from the six sources (§4, §5).

    uv run python scripts/build_corpus.py --out data/processed/corpus \
        --views-root data/processed/views
    uv run python scripts/build_corpus.py --sources vrsbench --limit 50 --dry-run

Each source is read through ``datasets.load_dataset`` and converted by the
adapter in :mod:`satquery.training.corpus_builder`, which is where every rule
that matters lives — the shared label builder, the shared box serialiser, and the
citation audit that fails the build rather than the eval. This script is only the
wiring: which split to read, where the pre-rendered views live, and what to do
with the report.

**Views are read, not rendered.** DATA_ADAPTATION_PLAN §7.2 pre-renders them in a
separate CPU pass so the exact bytes trained on are on disk and hashable, and so
the same files serve as Phase 8 eval fixtures. This script resolves each sample's
view paths by convention and, with ``--require-views``, refuses to write a corpus
line pointing at a file that does not exist.

Nothing downloads implicitly: ``--download`` must be passed for a source that is
not already in the local Hugging Face cache.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.schemas.enums import Modality  # noqa: E402
from satquery.training.corpus_builder import (  # noqa: E402
    COMPOSITION,
    CorpusError,
    CorpusSample,
    CorpusSource,
    Deduplicator,
    build_corpus,
    build_evidence_qa,
    from_bigearthnet,
    from_bigearthnet_txt,
    from_cdvqa,
    from_dior_rsvg,
    from_rsvqa,
    from_vrsbench,
    phash,
    sha256_of,
)

HF_DATASETS: dict[CorpusSource, str] = {
    CorpusSource.BIGEARTHNET_V2: "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt",
    CorpusSource.VRSBENCH: "xiang709/VRSBench",
    CorpusSource.RSVQA_HR: "flax-sentence-embeddings/RSVQA-HR",
    CorpusSource.CDVQA: "SECOND-CDVQA/cdvqa",
    CorpusSource.DIOR_RSVG: "danielz01/DIOR-RSVG",
}
"""Where each public source is read from. ``evidence_qa`` has no entry: it is
generated from the FactSheets the BigEarthNet render pass already produced."""

UNRESOLVED_SOURCES: dict[CorpusSource, str] = {
    CorpusSource.RSVQA_HR: (
        "'flax-sentence-embeddings/RSVQA-HR' does not exist on the Hub. The Hub's "
        "closest match, 'dmarsili/RSVQA-HR-2k', publishes a validation split only "
        "and is an evaluation subset, not a training corpus. The canonical RSVQA-HR "
        "release is hosted off-Hub (zenodo)"
    ),
    CorpusSource.CDVQA: (
        "'SECOND-CDVQA/cdvqa' does not exist on the Hub. 'ljx620/CDVQA' does, with "
        "train/validation/test splits, but publishes WebDataset shards of image "
        "bytes plus LLaVA-style conversation turns — a different adapter from the "
        "one in corpus_builder.from_cdvqa"
    ),
    CorpusSource.DIOR_RSVG: (
        "'danielz01/DIOR-RSVG' is gated: it loads once the account has accepted the "
        "terms and HF_TOKEN is set. The ungated mirrors are not substitutes — "
        "'pzhang1990/DIOR-RSVG' publishes a test split only, and "
        "'LittleCollections/DIOR-RSVG' does not carry the referring expressions"
    ),
}
"""Sources whose Hugging Face id could not be verified against a usable release.

Recorded rather than silently swapped for a mirror. §4.7 quarantines each
benchmark's test split by id, and a third-party mirror of unknown provenance is
exactly where a quarantined image reappears under a new name — "a leaked
benchmark is worse than no benchmark". Resolving these is a data-provenance
decision, so the builder states the problem and declines to guess."""

RESOLVED_SOURCES: tuple[CorpusSource, ...] = tuple(
    source for source in CorpusSource if source not in UNRESOLVED_SOURCES
)
"""The default ``--sources``. An unresolved source can still be named
explicitly, which reports its specific problem instead of a Hub 404."""

VIEW_IDS_BY_SOURCE: dict[CorpusSource, tuple[str, ...]] = {
    CorpusSource.BIGEARTHNET_V2: ("TC", "FCIR", "SWIR", "NDVI", "NDBI", "SARFC", "SARDB"),
}
"""BigEarthNet renders the full cross-modal stack; every other source is a
3-channel image and renders ``TC`` alone (§2.4)."""

BUNDLED_SPLIT: dict[CorpusSource, str] = {
    CorpusSource.BIGEARTHNET_V2: "train",
}
"""Sources that publish one Hugging Face split holding every row.

reBEN's parquet export puts all ~549 k patches under a split *named* ``train``,
and carries the official train / validation / test assignment in a column
instead. The name is therefore a container, not a claim: asking for
``validation`` raises ``Bad split``, and taking the ``train`` split at face value
would fold the official validation and test rows into training. Both are avoided
the same way — load the container, then filter on the column. §4.1 is explicit
that the assignment is never re-derived, because the published baselines depend
on it."""

SPLIT_FIELDS: tuple[str, ...] = ("split", "original_split", "set", "subset", "partition")
"""Column names reBEN's metadata table has used for the split assignment."""

SPLIT_ALIASES: dict[str, str] = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
    "testing": "test",
    "bench": "bench",
    "benchmark": "bench",
}
"""Normalises whatever the column says onto the names we filter by.

``bench`` is reBEN's fourth split and the reason the loader kept failing. The
dataset's own ``ben_txt_datamodule.py`` groups it with ``test``
(``splits=['test', 'bench']`` builds the test dataset), so it is held-out
evaluation data. It is mapped rather than rejected, and then excluded by the
filter along with ``test`` — a value we merely fail to *recognise* is a different
situation from one we recognise and decline."""


def row_split(row: Mapping[str, Any]) -> str | None:
    """The official split a bundled row belongs to, normalised.

    Returns None when the row carries no recognisable split column, which the
    caller treats as fatal rather than as permission to keep the row.
    """
    for field in SPLIT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return SPLIT_ALIASES.get(value.strip().lower())
    return None


def split_column_problem(
    source: CorpusSource, split: str, row: Mapping[str, Any]
) -> str:
    """Explain *why* a bundled row's split could not be read.

    Two very different faults reach this point and they need different fixes:
    the column is absent (wrong builder, wrong file, renamed schema), or the
    column is present and holds a value :data:`SPLIT_ALIASES` does not know
    (a new split, as ``bench`` was). Reporting both as "no split column" is what
    turned a one-line alias addition into a long hunt, so the message now says
    which one happened and shows the value and the columns that exist.
    """
    present = [field for field in SPLIT_FIELDS if row.get(field) is not None]
    if not present:
        return (
            f"{HF_DATASETS[source]} rows carry no split column (looked for "
            f"{list(SPLIT_FIELDS)}); the row's columns are {sorted(row)}. Refusing "
            "to guess, because guessing folds the official test split into training"
        )
    values = {field: row[field] for field in present}
    return (
        f"{HF_DATASETS[source]} row has a split column with an unrecognised value: "
        f"{values}. Known values are {sorted(set(SPLIT_ALIASES))}. Add it to "
        "SPLIT_ALIASES — mapping it to 'train'/'validation' to use it, or to any "
        "other name to hold it out — rather than dropping the check, which would "
        "fold held-out data into training"
    )


def hub_parquet_glob(source: CorpusSource) -> str:
    """The ``hf://`` glob a packaged builder needs to read a Hub repository.

    Built from :data:`HF_DATASETS` rather than written out, so the repository id
    lives in exactly one place: a builder pointed at a stale copy of it fails
    with an empty dataset rather than with a name it can be traced back to.
    """
    return f"hf://datasets/{HF_DATASETS[source]}/**/*.parquet"


def hf_split(source: CorpusSource, split: str) -> str:
    """The split name to *ask* Hugging Face for, which is not always the one we want."""
    return BUNDLED_SPLIT.get(source, split)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/processed/corpus"))
    parser.add_argument("--views-root", type=Path, default=Path("data/processed/views"))
    parser.add_argument(
        "--sources",
        nargs="*",
        choices=[source.value for source in CorpusSource],
        default=[source.value for source in RESOLVED_SOURCES],
        help="Defaults to the sources with a verified Hugging Face release. "
        "Naming an unresolved one reports what is wrong with it.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Rows per source, for a smoke run.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow Hugging Face to fetch a dataset that is not cached.",
    )
    parser.add_argument(
        "--require-views",
        action="store_true",
        help="Fail on a sample whose pre-rendered views are not on disk.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Build and report, but write nothing."
    )
    return parser.parse_args(argv)


def view_paths(source: CorpusSource, item_id: str, views_root: Path) -> dict[str, str]:
    """The conventional pre-rendered view paths for one item (§7.2)."""
    return {
        view_id: str(views_root / source.value / item_id / f"{view_id}.png")
        for view_id in VIEW_IDS_BY_SOURCE.get(source, ("TC",))
    }


def check_views(sample: CorpusSample, require: bool) -> None:
    """Refuse a corpus line that points at views nobody rendered.

    Raises:
        CorpusError: A referenced view file is absent.
    """
    if not require:
        return
    missing = [view.path for view in sample.views if not Path(view.path).is_file()]
    if missing:
        raise CorpusError(f"{sample.id}: pre-rendered views missing: {missing}")


def load_rows(
    source: CorpusSource, split: str, limit: int | None, download: bool
) -> Iterator[Mapping[str, Any]]:
    """Stream rows of one source's official split.

    A source listed in :data:`BUNDLED_SPLIT` is loaded whole and filtered on its
    own split column; every other source is asked for the split by name, which
    is what the five standard corpora support.

    ``limit`` counts rows *yielded*, not rows read, so a smoke run over a bundled
    source still returns the requested number instead of however many happened to
    precede the first validation row.

    A bundled source also names its builder explicitly. ``datasets`` infers the
    builder from the repository id, and this one ends in ``.txt``: inferred as a
    generic text dataset, it happily reads the repository's ``README.md`` and
    yields one row per *line of prose*, which is why the split column went
    missing rather than the load failing outright. Naming ``parquet`` stops the
    guess — and the files are addressed by their ``hf://`` URL, because a
    packaged builder resolves ``data_dir`` against the local filesystem and
    would look for a directory of that name beside the working directory.

    Raises:
        CorpusError: A bundled source's rows carry no recognisable split column.
            Keeping them would fold the official test split into training, and
            §4.7 makes a leaked benchmark a build error rather than a warning.
    """
    from datasets import load_dataset

    bundled = source in BUNDLED_SPLIT
    if bundled:
        dataset = load_dataset(
            "parquet",
            data_files=hub_parquet_glob(source),
            split=hf_split(source, split),
            streaming=True,
            download_mode=None if download else "reuse_cache_if_exists",
        )
    else:
        dataset = load_dataset(
            HF_DATASETS[source],
            split=hf_split(source, split),
            streaming=True,
            download_mode=None if download else "reuse_cache_if_exists",
        )
    wanted = SPLIT_ALIASES.get(split.lower(), split.lower())
    yielded = 0
    for row in dataset:
        row = dict(row)
        if bundled:
            assignment = row_split(row)
            if assignment is None:
                raise CorpusError(split_column_problem(source, split, row))
            if assignment != wanted:
                continue
        if limit is not None and yielded >= limit:
            return
        yielded += 1
        yield row


def hashes_for(row: Mapping[str, Any]) -> dict[str, Any]:
    """Image hashes for the dedup index, where the row carries a readable image."""
    path = row.get("image_path") or row.get("pre_path")
    if not path or not Path(str(path)).is_file():
        return {}
    return {"image_sha256": sha256_of(Path(str(path))), "phash": f"{phash(path):016x}"}


def build_source(
    source: CorpusSource, args: argparse.Namespace, rng: random.Random
) -> list[CorpusSample]:
    """Convert one source's official train and val splits into samples."""
    problem = UNRESOLVED_SOURCES.get(source)
    if problem is not None:
        raise CorpusError(
            f"{source.value} has no verified source: {problem}. Set "
            f"HF_DATASETS[{source.value!r}] to a release you have vetted against the "
            "§4.7 quarantine, or drop it from --sources and accept the composition gap"
        )

    samples: list[CorpusSample] = []
    for split in ("train", "validation"):
        for row in load_rows(source, split, args.limit, args.download):
            item_id = str(
                row.get("patch_id") or row.get("image_id") or row.get("pair_id") or len(samples)
            )
            enriched = dict(row, split=split, **hashes_for(row))
            enriched.setdefault("view_paths", view_paths(source, item_id, args.views_root))
            if source is not CorpusSource.BIGEARTHNET_V2:
                enriched.setdefault(
                    "image_path", enriched["view_paths"]["TC"]
                )
            samples.extend(convert(source, enriched, rng))
    for sample in samples:
        check_views(sample, args.require_views)
    return samples


def convert(
    source: CorpusSource, row: Mapping[str, Any], rng: random.Random
) -> list[CorpusSample]:
    """Dispatch one row to its adapter."""
    if source is CorpusSource.BIGEARTHNET_V2:
        # Two reBEN exports, dispatched on the row's shape. The label table
        # carries a 19-class vector and builds SCENE_CLASSIFY and presence
        # questions from it; the text export carries one question/answer pair per
        # row and no labels at all. Neither builder can read the other's rows.
        if row.get("labels") is not None:
            return from_bigearthnet(row, rng=rng)
        return from_bigearthnet_txt(row, rng=rng)
    if source is CorpusSource.VRSBENCH:
        return from_vrsbench(row, rng=rng)
    if source is CorpusSource.RSVQA_HR:
        return [from_rsvqa(row)]
    if source is CorpusSource.CDVQA:
        return [from_cdvqa(row)]
    if source is CorpusSource.DIOR_RSVG:
        return [from_dior_rsvg(row, rng=rng)]
    raise CorpusError(f"{source} has no dataset adapter; evidence_qa is generated")


def build_evidence_source(
    bigearthnet: Sequence[CorpusSample], rng: random.Random
) -> list[CorpusSample]:
    """Generate ``evidence_qa`` from the FactSheets BigEarthNet already carries.

    Free and exact supervision: every number in every answer was measured by the
    Phase 2 render pass, so the corpus teaches citation without a single human
    annotation (§4.6).
    """
    from satquery.training.corpus_builder import SourceView

    samples: list[CorpusSample] = []
    for sample in bigearthnet:
        if not sample.fact_sheet:
            continue
        samples.extend(
            build_evidence_qa(
                sample_id=f"evidence_qa:{sample.id.split(':', 1)[1].rsplit(':', 1)[0]}",
                views=[
                    SourceView(
                        view_id=view.view_id,
                        path=view.path,
                        modality=(
                            Modality.SAR
                            if view.view_id.startswith("SAR")
                            else Modality.OPTICAL
                        ),
                    )
                    for view in sample.views
                ],
                fact_sheet=sample.fact_sheet,
                pair_type=sample.pair_type,
                meta=sample.meta.model_copy(deep=True),
                rng=rng,
            )
        )
    return samples


def main(argv: Sequence[str] | None = None) -> int:
    """Build the corpus. Returns a process exit code."""
    args = parse_args(argv)
    rng = random.Random(args.seed)
    requested = {CorpusSource(name) for name in args.sources}

    excluded = [source for source in UNRESOLVED_SOURCES if source not in requested]
    if excluded:
        print("WARNING — these sources are not in this build:")
        for source in excluded:
            print(
                f"  {source.value:16s} target {COMPOSITION[source]:>6d} samples "
                f"({COMPOSITION[source] / sum(COMPOSITION.values()):.1%} of §5): "
                f"{UNRESOLVED_SOURCES[source]}"
            )
        print()

    built: dict[CorpusSource, list[CorpusSample]] = {}
    for source in CorpusSource:
        if source not in requested or source is CorpusSource.EVIDENCE_QA:
            continue
        built[source] = build_source(source, args, rng)
        print(f"{source.value:16s} {len(built[source]):>7d} samples")

    if CorpusSource.EVIDENCE_QA in requested:
        built[CorpusSource.EVIDENCE_QA] = build_evidence_source(
            built.get(CorpusSource.BIGEARTHNET_V2, []), rng
        )
        print(f"{'evidence_qa':16s} {len(built[CorpusSource.EVIDENCE_QA]):>7d} samples")
        if not built[CorpusSource.EVIDENCE_QA]:
            # Worth saying out loud: this is the only source that teaches
            # citation, and it is generated from measurements rather than from
            # annotations. No FactSheets, no citation supervision (§4.6).
            print(
                "  note: evidence_qa is generated from the FactSheets the Phase 2 "
                "render pass measures for each patch, and no BigEarthNet sample "
                "carried one. Run the render pass first; the text export publishes "
                "annotations, not measurements."
            )

    if args.dry_run:
        print("dry run: nothing written.")
        return 0

    report = build_corpus(
        built,
        args.out,
        composition={source: COMPOSITION[source] for source in requested},
        dedup=Deduplicator(),
        seed=args.seed,
    )
    for row in report.sources:
        print(
            f"{row.source.value:16s} track {row.track}  built {row.built:>6d}  "
            f"kept {row.kept:>6d}  train {row.train:>6d}  val {row.val:>5d}"
        )
    print(
        f"corpus: {report.train} train / {report.val} val "
        f"({report.dropped_exact} exact and {report.dropped_near} near duplicates dropped)"
    )
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
