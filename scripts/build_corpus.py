#!/usr/bin/env python
"""Build the Phase 7 instruction corpus from the six sources (§4, §5).

    uv run python scripts/build_corpus.py --out data/processed/corpus \
        --views-root data/processed/views
    uv run python scripts/build_corpus.py --sources vrsbench --limit 50 --dry-run

Each source is read — through ``datasets.load_dataset`` for the ones the Hub
publishes as a table, and through :mod:`satquery.training.local_sources` for the
three it does not (see :data:`LOCAL_SOURCES`) — and converted by the adapter in
:mod:`satquery.training.corpus_builder`, which is where every rule that matters
lives: the shared label builder, the shared box serialiser, and the citation
audit that fails the build rather than the eval. This script is only the wiring:
which split to read, where the pre-rendered views live, and what to do with the
report.

**Views are read, not rendered.** DATA_ADAPTATION_PLAN §7.2 pre-renders them in a
separate CPU pass so the exact bytes trained on are on disk and hashable, and so
the same files serve as Phase 8 eval fixtures. This script resolves each sample's
view paths by convention and, with ``--require-views``, refuses to write a corpus
line pointing at a file that does not exist.

Nothing downloads implicitly: ``--download`` must be passed for a source that is
not already in the local Hugging Face cache.

**The build streams end to end.** BigEarthNet.txt is a single 467 MB parquet of
9.6 M annotation rows; reading it with ``streaming=True`` and then collecting the
converted samples into a list still needs tens of gigabytes, which is what got an
earlier run OOM-killed. So rows are read a record batch at a time, converted one
at a time, and fed straight into
:func:`~satquery.training.corpus_builder.build_corpus_streaming`, which
deduplicates as they pass and keeps a reservoir sample capped at each source's §5
target. Peak memory is one corpus, not one dataset.

Measured on the real 9.6 M-row export: ~33 MB of resident heap per 1 000 retained
samples, plus a fixed ~1.2 GB for the ``datasets``/``pyarrow`` streaming reader.
The full 65 k §5 budget therefore peaks near 3.3 GB and does not grow with the
size of the source. Pass ``--limit`` to stop each split early as well, for a
smoke run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import UTC, datetime
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.render.views import CATALOGUE, ImageFormat, ViewId  # noqa: E402
from satquery.schemas.enums import Modality  # noqa: E402
from satquery.training.corpus_builder import (  # noqa: E402
    COMPOSITION,
    CorpusError,
    CorpusSample,
    CorpusSource,
    Deduplicator,
    build_corpus_streaming,
    build_evidence_qa,
    from_bigearthnet,
    from_bigearthnet_txt,
    from_cdvqa,
    from_dior_rsvg,
    from_rsvqa,
    from_vrsbench,
    noncanonical_grounding,
    phash,
    read_jsonl,
    sha256_of,
)
from satquery.training.local_sources import READERS as LOCAL_READERS  # noqa: E402
from satquery.training.local_sources import is_available as local_is_available  # noqa: E402

HF_DATASETS: dict[CorpusSource, str] = {
    CorpusSource.BIGEARTHNET_V2: "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt",
    CorpusSource.VRSBENCH: "xiang709/VRSBench",
    CorpusSource.RSVQA_HR: "zenodo.org/records/6344367",
    CorpusSource.CDVQA: "ljx620/CDVQA",
    CorpusSource.DIOR_RSVG: "danielz01/DIOR-RSVG",
}
"""Where each public source is read from. ``evidence_qa`` has no entry: it is
generated from the FactSheets the BigEarthNet render pass already produced.

Three of these are not *loaded* from here — see :data:`LOCAL_SOURCES`. The id is
still recorded, because it is the provenance a §4.7 audit is traced back to, and
``rsvqa_hr``'s is a Zenodo record rather than a Hub name precisely because the
canonical release is not on the Hub at all."""

LOCAL_SOURCES: frozenset[CorpusSource] = frozenset(LOCAL_READERS)
"""Sources read from ``data/raw`` by :mod:`satquery.training.local_sources`.

Not a convenience. Each of these three defeats ``load_dataset`` in its own way,
and one of them defeats it *silently*:

``vrsbench``
    The repository publishes four archives and no table. Hugging Face's
    data-file inference sweeps every name containing ``train`` into the train
    split, so the JSON packaged builder is handed ``Images_train.zip`` — 8.4 GB
    of PNG — and pyarrow dies with ``ArrowInvalid: JSON parse error: Invalid
    value. in row 0``. It dies *late*: the annotation zip is read first, so a
    ``--limit 20`` smoke run passes and the full pass fails 20 264 rows in. That
    is the crash that got VRSBench excluded from the last corpus.
``rsvqa_hr``
    Canonical release is Zenodo record 6344367, off-Hub entirely.
``cdvqa``
    A WebDataset of raw image bytes and LLaVA turns, not the flat rows
    ``from_cdvqa`` reads.

``scripts/fetch_sources.py`` puts the first two on disk; VRSBench is the two
annotation zips unpacked in place."""

UNRESOLVED_SOURCES: dict[CorpusSource, str] = {
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

HASH_CACHE_SIZE: int = 65_536
"""How many images' hashes :func:`hashes_for` remembers.

One entry is two short strings, so the whole cache is a few megabytes — set
above the largest per-source image count (VRSBench's 29 614) so a full pass over
any one source never evicts an entry it will ask for again."""

PROGRESS_EVERY: int = 25_000
"""How often the build says where it is.

A streaming build over 9.6 M rows shows nothing for a long time, and silence
is what a stalled build and a working one look like alike."""

VIEW_IDS_BY_SOURCE: dict[CorpusSource, tuple[str, ...]] = {
    CorpusSource.BIGEARTHNET_V2: ("TC", "FCIR", "SWIR", "NDVI", "NDBI", "SARFC", "SARDB"),
    CorpusSource.CDVQA: ("TC_pre", "TC_post"),
}
"""BigEarthNet renders the full cross-modal stack; every other source is a
3-channel image and renders ``TC`` alone (§2.4).

``cdvqa`` is the exception among the 3-channel sources: it is bi-temporal, so
one item is two true-colour renders of the same scene at two dates (§4.4). They
are suffixed rather than given new catalogue ids — both really are ``TC``, and
:func:`view_suffix` reads the format off the ``TC`` spec for exactly that
reason."""

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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Rows per source per split, for a smoke run. Stops the scan early.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow Hugging Face to fetch a dataset that is not cached.",
    )
    parser.add_argument(
        "--require-views",
        action="store_true",
        help="Shorthand for --on-missing-views fail.",
    )
    parser.add_argument(
        "--on-missing-views",
        choices=("keep", "skip", "fail"),
        default=None,
        help="What to do with a sample whose pre-rendered views are not on disk: "
        "keep the line anyway (default), skip the sample and carry on, or fail "
        "the build. Use 'skip' to build a corpus over the patches the render "
        "pass has actually covered.",
    )
    parser.add_argument(
        "--composition",
        nargs="*",
        default=None,
        metavar="SOURCE=N",
        help="Override a source's §5 target for this build, e.g. bigearthnet_v2=4000 "
        "evidence_qa=2500. Unnamed sources keep their COMPOSITION target. This is "
        "how a run is budgeted against measured throughput without editing the "
        "table that records the plan.",
    )
    parser.add_argument(
        "--factsheets",
        type=Path,
        default=None,
        help="factsheets.jsonl from scripts/render_views.py. Binds each BigEarthNet "
        "patch's measured FactSheet and the views the render pass actually wrote, "
        "which is what lets evidence_qa be generated in-band (§4.6). Defaults to "
        "<views-root>/bigearthnet_v2/factsheets.jsonl when that file exists.",
    )
    parser.add_argument(
        "--require-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After the build, refuse (exit 5) if any requested source contributed "
        "no train samples, or fewer than half of min(target, built) — the way a "
        "source quietly loses 90 %% of itself (ML_PIPELINE_RECOVERY_PLAN §5.3). "
        "--no-require-sources only for a deliberate partial build.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Build and report, but write nothing."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Cap the pyarrow CPU and I/O thread pools that decode the streamed "
        "parquet record batches. pyarrow sizes both to the machine's core count, "
        "and on a 16-thread box each in-flight batch of BigEarthNet.txt carries its "
        "own decode buffers — which is what froze a 32 GB machine. The build "
        "itself is single-process either way; 2 is a safe floor.",
    )
    return parser.parse_args(argv)


def cap_reader_threads(workers: int | None) -> None:
    """Pin pyarrow's thread pools before ``datasets`` opens anything.

    Must run before the first ``load_dataset`` call: pyarrow creates the pools
    lazily and honours ``set_cpu_count`` only for threads not yet spawned.
    """
    if workers is None:
        return
    if workers < 1:
        raise CorpusError(f"--workers must be >= 1, got {workers}")
    import os

    # Threads spawned by BLAS / tokenizers inside the adapters follow the same cap.
    os.environ.setdefault("OMP_NUM_THREADS", str(workers))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import pyarrow as pa

    pa.set_cpu_count(workers)
    pa.set_io_thread_count(workers)
    print(f"reader threads: pyarrow cpu={pa.cpu_count()} io={pa.io_thread_count()}")


def view_suffix(view_id: str) -> str:
    """The file extension ``scripts/render_views.py`` writes for one view.

    Taken from the same catalogue the renderer encodes against, not hardcoded:
    §2.1 puts the measurement views (NDVI, NDBI, NDWI, SARDB, SWIR) in lossless
    PNG and the composites (TC, FCIR, SARFC) in JPEG q92. This function used to
    return ``.png`` for every view, so a corpus built after a successful render
    still pointed at three files that were never written — and because
    ``--require-views`` is off by default, that resolved to the black
    placeholders instead of failing.
    """
    try:
        spec = CATALOGUE[ViewId(view_id.split("_", 1)[0])]
    except (ValueError, KeyError):
        return "png"
    return "png" if spec.image_format is ImageFormat.PNG else "jpg"


def view_paths(source: CorpusSource, item_id: str, views_root: Path) -> dict[str, str]:
    """The conventional pre-rendered view paths for one item (§7.2)."""
    return {
        view_id: str(
            views_root / source.value / item_id / f"{view_id}.{view_suffix(view_id)}"
        )
        for view_id in VIEW_IDS_BY_SOURCE.get(source, ("TC",))
    }


def bind_view_paths(
    source: CorpusSource,
    row: dict[str, Any],
    item_id: str,
    views_root: Path,
    facts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Point one row's image fields at the files the render pass wrote (§7.2).

    Each adapter reads the pixels it needs under a different key, and getting
    this wrong is invisible until training: ``--on-missing-views`` defaults to
    ``keep``, so a row pointed at a path nobody rendered produces a corpus line
    that resolves to a black placeholder rather than an error.

    - ``bigearthnet_v2`` carries its own band paths; only the view map is added.
    - ``cdvqa`` is bi-temporal (§4.4) and ``from_cdvqa`` reads ``pre_path`` and
      ``post_path``, so it gets the pair the VHR render pass writes as
      ``TC_pre`` / ``TC_post`` rather than a single ``TC``.
    - every other source is a single 3-channel image read from ``image_path``.

    CDVQA's two paths are *overwritten* rather than defaulted, because its rows
    arrive from ``fetch_sources`` already naming the raw ``data/raw`` pair they
    were unpacked from. Those are the render pass's input, not the corpus's:
    §7.2 trains on the bytes on disk under ``data/processed/views``, and a line
    pointing back at the raw file would train on pixels no view spec ever
    normalised.
    """
    if source is CorpusSource.CDVQA:
        paths = view_paths(source, item_id, views_root)
        row.setdefault("view_paths", paths)
        row["pre_path"] = paths["TC_pre"]
        row["post_path"] = paths["TC_post"]
        return row
    sheet = facts.get(item_id) if facts and source is CorpusSource.BIGEARTHNET_V2 else None
    if sheet is not None:
        # The render pass knows which views it actually wrote — a patch with no
        # S1 scene gets no SAR views rather than a path to a file nobody made —
        # and its FactSheet is the only measured supervision evidence_qa has.
        row.setdefault("view_paths", dict(sheet.get("views") or {}))
        row.setdefault("fact_sheet", dict(sheet.get("fact_sheet") or {}))
        if sheet.get("labels") is not None:
            row.setdefault("labels", list(sheet["labels"]))
    row.setdefault("view_paths", view_paths(source, item_id, views_root))
    if source is not CorpusSource.BIGEARTHNET_V2:
        row.setdefault("image_path", row["view_paths"]["TC"])
    return row


def load_factsheet_index(path: Path) -> dict[str, dict[str, Any]]:
    """Read ``factsheets.jsonl`` into ``patch_id -> record``.

    A bounded side table, not a second stream: ~25 k records of a few hundred
    bytes each. The *rows* still stream; this is what lets each of them pick up
    the measurements the render pass took for its patch (§4.6), which the build
    never did before — so ``evidence_qa`` was only ever produced by the
    standalone builder, with its own split logic and no image hashes.

    Raises:
        CorpusError: The file is missing, or a line is not a record with a
            ``patch_id``.
    """
    if not path.is_file():
        raise CorpusError(f"--factsheets {path} does not exist; run scripts/render_views.py first")
    index: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise CorpusError(f"{path}:{number}: not JSON ({error})") from error
            patch_id = record.get("patch_id") if isinstance(record, dict) else None
            if not patch_id:
                raise CorpusError(f"{path}:{number}: record carries no patch_id")
            index[str(patch_id)] = record
    return index


def resolve_factsheets(args: argparse.Namespace) -> Path | None:
    """The factsheet file to bind, explicit or by convention, or None."""
    explicit: Path | None = getattr(args, "factsheets", None)
    if explicit is not None:
        return explicit
    conventional = Path(args.views_root) / CorpusSource.BIGEARTHNET_V2.value / "factsheets.jsonl"
    return conventional if conventional.is_file() else None


def view_policy(args: argparse.Namespace) -> str:
    """The effective ``--on-missing-views`` mode for this invocation.

    ``--require-views`` predates the three-way choice and stays as its own flag,
    because it is what the overnight script and the docs already say. It simply
    resolves to ``fail``; naming both, with the newer flag saying something
    else, is the one combination worth rejecting outright.

    Raises:
        CorpusError: Both flags were given and they disagree.
    """
    explicit: str | None = getattr(args, "on_missing_views", None)
    if explicit is None:
        return "fail" if args.require_views else "keep"
    if args.require_views and explicit != "fail":
        raise CorpusError(
            f"--require-views means --on-missing-views fail, but "
            f"--on-missing-views {explicit} was given. Pass one of them."
        )
    return explicit


def missing_views(sample: CorpusSample) -> list[str]:
    """The referenced view files that are not on disk."""
    return [view.path for view in sample.views if not Path(view.path).is_file()]


def check_views(sample: CorpusSample, require: bool) -> None:
    """Refuse a corpus line that points at views nobody rendered.

    Raises:
        CorpusError: A referenced view file is absent.
    """
    if not require:
        return
    absent = missing_views(sample)
    if absent:
        raise CorpusError(f"{sample.id}: pre-rendered views missing: {absent}")


def keep_sample(sample: CorpusSample, mode: str) -> bool:
    """Apply the ``--on-missing-views`` policy to one sample.

    Three genuinely different situations, and conflating them is what produced a
    corpus where 98% of the lines pointed at files the render pass never wrote:

    ``keep``
        The historical default. Write the line anyway — correct while the render
        pass is still catching up, and the reason ``--require-views`` exists.
    ``skip``
        Drop the sample and keep building. This is what makes a corpus out of
        whatever *has* been rendered: §7.2 renders a subset, and a run that
        wants real pixels needs the corpus restricted to that subset rather than
        the whole build refused.
    ``fail``
        Stop the build. What ``--require-views`` did, kept under its own name.

    Raises:
        CorpusError: *mode* is ``fail`` and a view file is absent.
    """
    if mode == "keep":
        return True
    absent = missing_views(sample)
    if not absent:
        return True
    if mode == "fail":
        raise CorpusError(f"{sample.id}: pre-rendered views missing: {absent}")
    return False


def open_dataset(source: CorpusSource, split: str, download: bool) -> Any:
    """Open one source's Hugging Face split as a streaming iterable.

    ``streaming=True`` is what keeps the 467 MB / 9.6 M-row BigEarthNet.txt
    parquet off the heap: ``datasets`` pulls it a record batch at a time over
    HTTP range requests instead of downloading and decoding the file whole.

    A bundled source also names its builder explicitly. ``datasets`` infers the
    builder from the repository id, and this one ends in ``.txt``: inferred as a
    generic text dataset, it happily reads the repository's ``README.md`` and
    yields one row per *line of prose*, which is why the split column went
    missing rather than the load failing outright. Naming ``parquet`` stops the
    guess — and the files are addressed by their ``hf://`` URL, because a
    packaged builder resolves ``data_dir`` against the local filesystem and
    would look for a directory of that name beside the working directory.
    """
    from datasets import load_dataset

    if source in BUNDLED_SPLIT:
        return load_dataset(
            "parquet",
            data_files=hub_parquet_glob(source),
            split=split,
            streaming=True,
            download_mode=None if download else "reuse_cache_if_exists",
        )
    return load_dataset(
        HF_DATASETS[source],
        split=split,
        streaming=True,
        download_mode=None if download else "reuse_cache_if_exists",
    )


def load_rows_by_split(
    source: CorpusSource, splits: Sequence[str], limit: int | None, download: bool
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Stream one source's rows, each tagged with the split it belongs to.

    One pass over the data, not one per split. A source listed in
    :data:`BUNDLED_SPLIT` is read once and routed on its own split column;
    opening it once per split downloaded and decoded 9.6 M rows twice to keep a
    different subset each time. Every other source has no such column and is
    asked for each split by name, which is what the five standard corpora
    support.

    Nothing is accumulated here and nothing is accumulated by the caller: rows
    are yielded one at a time and converted one at a time, because materialising
    this stream is what exhausted RAM and got the build OOM-killed.

    ``limit`` counts rows *yielded* per split, not rows read, so a smoke run over
    a bundled source still returns the requested number instead of however many
    happened to precede the first validation row. Once every requested split has
    hit its limit the iteration stops, rather than scanning to end of file.

    Raises:
        CorpusError: A bundled source's rows carry no recognisable split column.
            Keeping them would fold the official test split into training, and
            §4.7 makes a leaked benchmark a build error rather than a warning.
    """
    wanted = {SPLIT_ALIASES.get(split.lower(), split.lower()): split for split in splits}
    yielded = dict.fromkeys(splits, 0)

    def full(split: str) -> bool:
        return limit is not None and yielded[split] >= limit

    if source in LOCAL_SOURCES:
        # Read from disk, one official split at a time. The reader already
        # stamps each row with the split it came from, which is what
        # `_guard_quarantine` checks; the tag is repeated here because the
        # caller routes on it.
        reader = LOCAL_READERS[source]
        for split in splits:
            for row in reader(split, limit):
                yielded[split] += 1
                yield split, dict(row)
        return

    if source in BUNDLED_SPLIT:
        for row in open_dataset(source, hf_split(source, splits[0]), download):
            if all(full(split) for split in splits):
                return
            row = dict(row)
            assignment = row_split(row)
            if assignment is None:
                raise CorpusError(split_column_problem(source, splits[0], row))
            split = wanted.get(assignment)
            if split is None or full(split):
                continue
            yielded[split] += 1
            yield split, row
        return

    for split in splits:
        for row in open_dataset(source, hf_split(source, split), download):
            if full(split):
                break
            yielded[split] += 1
            yield split, dict(row)


def load_rows(
    source: CorpusSource, split: str, limit: int | None, download: bool
) -> Iterator[Mapping[str, Any]]:
    """Stream rows of one source's official split.

    A single-split view of :func:`load_rows_by_split`, kept because most callers
    want exactly one split and should not have to unpack a tag to get it.
    """
    for _, row in load_rows_by_split(source, [split], limit, download):
        yield row


@lru_cache(maxsize=HASH_CACHE_SIZE)
def _hashes_of(path: str) -> tuple[str, str] | None:
    """One image's ``(sha256, phash)``, or None when it is not on disk.

    Raises:
        CorpusError: The file exists but is not an image PIL can open. A corrupt
            render is a render-pass bug and must not enter training under any
            ``--on-missing-views`` policy — and it must be reported as the file
            it is, not as a PIL traceback twenty hours into a build.
    """
    if not Path(path).is_file():
        return None
    try:
        return sha256_of(Path(path)), f"{phash(path):016x}"
    except (OSError, ValueError) as error:
        raise CorpusError(f"{path}: cannot be hashed for dedup: {error}") from error


class HashState(StrEnum):
    """Why a row did or did not get an image hash."""

    HASHED = "hashed"
    MISSING = "missing"
    """The row names an image, and it is not on disk."""
    UNHASHABLE = "unhashable"
    """The row names no image at all."""


@dataclass(frozen=True)
class ImageHashes:
    """The dedup identity of one row's image, and how it was (not) obtained."""

    state: HashState
    path: str | None
    sha256: str | None = None
    phash: str | None = None

    def as_row_fields(self) -> dict[str, str]:
        """The keys :func:`~satquery.training.corpus_builder._meta_for` reads."""
        if self.state is not HashState.HASHED:
            return {}
        assert self.sha256 is not None and self.phash is not None
        return {"image_sha256": self.sha256, "phash": self.phash}


@dataclass
class HashTally:
    """How many rows of a build got a hash, per state. Reported at the end."""

    counts: dict[HashState, int] = dc_field(default_factory=lambda: dict.fromkeys(HashState, 0))

    def add(self, hashes: ImageHashes) -> None:
        """Count one outcome."""
        self.counts[hashes.state] += 1

    @property
    def unhashed(self) -> int:
        """Rows that passed the dedup index without an identity."""
        return self.counts[HashState.MISSING] + self.counts[HashState.UNHASHABLE]


def dedup_image_path(source: CorpusSource, row: Mapping[str, Any]) -> str | None:
    """The file whose bytes identify this row's image for §4.7 dedup.

    Resolved from the same place the corpus line will point at, in this order:
    ``image_path`` (single-image sources), ``pre_path`` (CDVQA — the pair shares
    an id, so the pre image stands for both), then the first canonical view in
    ``view_paths`` — ``TC`` for BigEarthNet. That last fallback is what was
    missing: BigEarthNet carries its own band paths and never sets
    ``image_path``, so every one of its rows hashed to nothing and the dedup ran
    on an 18 k-line corpus without checking a single image. The ``TC`` path is
    also what :func:`~satquery.training.corpus_builder.image_key_of` returns for
    the built sample, so the index's owner key and its image key agree.
    """
    path = row.get("image_path") or row.get("pre_path")
    if path:
        return str(path)
    views: Mapping[str, Any] = row.get("view_paths") or {}
    for view_id in VIEW_IDS_BY_SOURCE.get(source, ("TC",)):
        if views.get(view_id):
            return str(views[view_id])
    return None


def hashes_for(source: CorpusSource, row: Mapping[str, Any]) -> ImageHashes:
    """Image hashes for the dedup index, from the image the row will train on.

    Memoised by path, because the sources publish rows per *annotation* while
    the hashes are per *image*: RSVQA-HR asks around a hundred questions of each
    tile and CDVQA around forty of each pair, so hashing per row reads and
    perceptually hashes the same JPEG a hundred times. On the §5 budget that is
    the difference between a minute and an hour, and it changes no result — the
    file does not move under us mid-build.

    Returns an :class:`ImageHashes` rather than a bare dict so "no image named"
    and "image named but absent" stay distinguishable: the first is a source
    with no pixels, the second is a render gap, and only one of them should be
    allowed to enter a corpus that claims strict image-level dedup.

    Raises:
        CorpusError: The image exists and cannot be read.
    """
    path = dedup_image_path(source, row)
    if path is None:
        return ImageHashes(HashState.UNHASHABLE, None)
    hashes = _hashes_of(path)
    if hashes is None:
        return ImageHashes(HashState.MISSING, path)
    return ImageHashes(HashState.HASHED, path, sha256=hashes[0], phash=hashes[1])


def iter_source_samples(
    source: CorpusSource,
    args: argparse.Namespace,
    rng: random.Random,
    tally: HashTally | None = None,
    facts: Mapping[str, Mapping[str, Any]] | None = None,
) -> Iterator[tuple[CorpusSource, CorpusSample]]:
    """Convert one source's official train and val splits into a *stream*.

    A generator rather than a list, and the reason this script no longer gets
    OOM-killed: BigEarthNet.txt is 9.6 M rows, and holding every ``CorpusSample``
    it converts to — three message turns, seven view records and a meta block
    apiece — needs tens of gigabytes before the §5 subsample ever runs. Nothing
    downstream needs the population, only a bounded sample of it, so rows are
    converted one at a time and handed straight to
    :func:`~satquery.training.corpus_builder.build_corpus_streaming`.

    ``evidence_qa`` is yielded inline, from the samples it derives from, because
    a stream cannot be walked twice (§4.6).
    """
    if source in LOCAL_SOURCES and not local_is_available(source):
        # Checked before the generator is built, for the same reason an
        # unresolved source is: "which source is missing" is a far cheaper
        # question to answer now than three sources into a streaming build.
        raise CorpusError(
            f"{source.value} is read from disk and has not been fetched. See "
            "scripts/fetch_sources.py, or this source's reader in "
            "satquery.training.local_sources for the exact command"
        )
    problem = UNRESOLVED_SOURCES.get(source)
    if problem is not None:
        # Raised before the generator is built, not on its first `next()`: an
        # unusable --sources should fail the invocation, not three sources in.
        raise CorpusError(
            f"{source.value} has no verified source: {problem}. Set "
            f"HF_DATASETS[{source.value!r}] to a release you have vetted against the "
            "§4.7 quarantine, or drop it from --sources and accept the composition gap"
        )
    return _iter_source_samples(source, args, rng, tally, facts)


def _iter_source_samples(
    source: CorpusSource,
    args: argparse.Namespace,
    rng: random.Random,
    tally: HashTally | None = None,
    facts: Mapping[str, Mapping[str, Any]] | None = None,
) -> Iterator[tuple[CorpusSource, CorpusSample]]:
    """The body of :func:`iter_source_samples`, once the source is known good."""
    mode = view_policy(args)
    tally = tally if tally is not None else HashTally()
    derive_evidence = (
        source is CorpusSource.BIGEARTHNET_V2
        and CorpusSource.EVIDENCE_QA in {CorpusSource(name) for name in args.sources}
    )
    rows = load_rows_by_split(source, ("train", "validation"), args.limit, args.download)
    # BigEarthNet patches whose label-table samples have been emitted. The text
    # export publishes ~20 annotation rows per patch; the patch's own labels and
    # FactSheet (bound from factsheets.jsonl) support one set of label-table
    # samples and one evidence_qa derivation, not twenty.
    labelled_patches: set[str] = set()
    for seen, (split, row) in enumerate(rows):
        item_id = str(
            row.get("patch_id") or row.get("image_id") or row.get("pair_id") or seen
        )
        enriched = bind_view_paths(
            source, dict(row, split=split), item_id, args.views_root, facts
        )
        # Hashed *after* binding, so the sha256/pHash pair the dedup index sees
        # is the rendered view the corpus line actually points at, not whatever
        # raw file the source happened to name. §4.7 dedups across sources, and
        # two sources agreeing only after a render is the case it exists for.
        hashes = hashes_for(source, enriched)
        tally.add(hashes)
        if hashes.state is HashState.MISSING and mode == "fail":
            # check_views() runs after conversion; the hash runs before it, so
            # without this a --require-views build could index nothing and pass.
            raise CorpusError(f"{source.value}:{item_id}: dedup image missing: {hashes.path}")
        enriched.update(hashes.as_row_fields())
        for sample in convert(source, enriched, rng):
            if not keep_sample(sample, mode):
                continue
            yield source, sample
            if derive_evidence:
                for evidence in iter_evidence_qa(sample, rng):
                    yield CorpusSource.EVIDENCE_QA, evidence
        if (
            source is CorpusSource.BIGEARTHNET_V2
            and is_ben_txt_row(enriched)
            and enriched.get("labels") is not None
            and item_id not in labelled_patches
        ):
            # The text row went to the text builder above. The patch it sits
            # on also carries a label vector and a FactSheet, which the
            # label-table builder turns into SCENE_CLASSIFY, presence and
            # cross-modal samples — and which is the only measured supervision
            # evidence_qa has (§4.6). Once per patch.
            labelled_patches.add(item_id)
            for sample in from_bigearthnet(enriched, rng=rng):
                if not keep_sample(sample, mode):
                    continue
                yield source, sample
                if derive_evidence:
                    for evidence in iter_evidence_qa(sample, rng):
                        yield CorpusSource.EVIDENCE_QA, evidence


BEN_TXT_COLUMNS: tuple[str, ...] = ("input", "output", "type")
"""What makes a BigEarthNet row a text-export annotation rather than a label row."""


def is_ben_txt_row(row: Mapping[str, Any]) -> bool:
    """True for a BigEarthNet.txt annotation row, whatever else was bound onto it."""
    return all(row.get(column) is not None for column in BEN_TXT_COLUMNS)


def convert(
    source: CorpusSource, row: Mapping[str, Any], rng: random.Random
) -> list[CorpusSample]:
    """Dispatch one row to its adapter."""
    if source is CorpusSource.BIGEARTHNET_V2:
        # Two reBEN exports, dispatched on the row's *annotation* shape. The
        # text export carries one question/answer pair per row (`input`,
        # `output`, `type`); the label table carries a 19-class vector and no
        # questions. Dispatching on `labels` — as this used to — sent every
        # text row to the label-table builder once factsheet binding started
        # attaching the patch's labels to it, and the corpus silently lost all
        # of BEN-txt's VQA and GROUNDING supervision. The bound labels stay on
        # the row: the text builder uses them to name a grounding box.
        if is_ben_txt_row(row):
            return from_bigearthnet_txt(row, rng=rng)
        return from_bigearthnet(row, rng=rng)
    if source is CorpusSource.VRSBENCH:
        return from_vrsbench(row, rng=rng)
    if source is CorpusSource.RSVQA_HR:
        return [from_rsvqa(row)]
    if source is CorpusSource.CDVQA:
        return [from_cdvqa(row)]
    if source is CorpusSource.DIOR_RSVG:
        return [from_dior_rsvg(row, rng=rng)]
    raise CorpusError(f"{source} has no dataset adapter; evidence_qa is generated")


def iter_evidence_qa(
    sample: CorpusSample, rng: random.Random
) -> Iterator[CorpusSample]:
    """Generate ``evidence_qa`` from the FactSheet one BigEarthNet sample carries.

    Free and exact supervision: every number in every answer was measured by the
    Phase 2 render pass, so the corpus teaches citation without a single human
    annotation (§4.6).

    Per sample rather than per corpus, so it composes with a stream: the old
    version took the whole built BigEarthNet list, which is exactly the object
    this script no longer builds.
    """
    from satquery.training.corpus_builder import SourceView

    if not sample.fact_sheet:
        return
    yield from build_evidence_qa(
        sample_id=f"evidence_qa:{sample.id.split(':', 1)[1].rsplit(':', 1)[0]}",
        views=[
            SourceView(
                view_id=view.view_id,
                path=view.path,
                modality=(
                    Modality.SAR if view.view_id.startswith("SAR") else Modality.OPTICAL
                ),
            )
            for view in sample.views
        ],
        fact_sheet=sample.fact_sheet,
        pair_type=sample.pair_type,
        meta=sample.meta.model_copy(deep=True),
        rng=rng,
    )


def build_evidence_source(
    bigearthnet: Iterable[CorpusSample], rng: random.Random
) -> list[CorpusSample]:
    """:func:`iter_evidence_qa` over a whole collection, as a list.

    For callers that already hold the samples in memory; the build itself goes
    through the generator.
    """
    return [
        evidence
        for sample in bigearthnet
        for evidence in iter_evidence_qa(sample, rng)
    ]


def composition_for(
    requested: set[CorpusSource], overrides: Sequence[str] | None
) -> dict[CorpusSource, int]:
    """The per-source targets for this build: §5's table, with ``--composition`` applied.

    Raises:
        CorpusError: An override does not parse, or names a source not requested.
    """
    targets = {source: COMPOSITION[source] for source in requested}
    for item in overrides or ():
        name, sep, number = item.partition("=")
        if not sep or not number.isdigit():
            raise CorpusError(f"--composition expects SOURCE=N, got {item!r}")
        try:
            source = CorpusSource(name)
        except ValueError as error:
            raise CorpusError(f"--composition names an unknown source {name!r}") from error
        if source not in requested:
            raise CorpusError(f"--composition names {name}, which is not in --sources")
        targets[source] = int(number)
    return targets


def main(argv: Sequence[str] | None = None) -> int:
    """Build the corpus. Returns a process exit code."""
    args = parse_args(argv)
    cap_reader_threads(args.workers)
    rng = random.Random(args.seed)
    requested = {CorpusSource(name) for name in args.sources}
    tally = HashTally()

    facts: dict[str, dict[str, Any]] | None = None
    factsheets = resolve_factsheets(args)
    if factsheets is not None and CorpusSource.BIGEARTHNET_V2 in requested:
        facts = load_factsheet_index(factsheets)
        print(f"factsheets: {len(facts)} BigEarthNet patches bound from {factsheets}")
    elif CorpusSource.EVIDENCE_QA in requested:
        print(
            "WARNING — no factsheets.jsonl found; evidence_qa cannot be generated "
            "in-band. Pass --factsheets or run scripts/render_views.py first."
        )

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

    ordered = [
        source
        for source in CorpusSource
        if source in requested and source is not CorpusSource.EVIDENCE_QA
    ]

    def stream() -> Iterator[tuple[CorpusSource, CorpusSample]]:
        """Every source's samples, one at a time, nothing held in between."""
        for source in ordered:
            counted = 0
            for tagged in iter_source_samples(source, args, rng, tally, facts):
                counted += tagged[0] is source
                yield tagged
                if counted % PROGRESS_EVERY == 0 and counted:
                    print(f"  {source.value:16s} {counted:>9d} samples built", flush=True)
            print(f"{source.value:16s} {counted:>7d} samples", flush=True)

    if args.dry_run:
        # Still a full pass, still bounded: counting is what a dry run is for.
        counts: dict[CorpusSource, int] = dict.fromkeys(CorpusSource, 0)
        for source, _ in stream():
            counts[source] += 1
        for source, count in counts.items():
            if count or source in requested:
                print(f"{source.value:16s} {count:>7d} samples")
        print("dry run: nothing written.")
        return 0

    report = build_corpus_streaming(
        stream(),
        args.out,
        composition=composition_for(requested, args.composition),
        dedup=Deduplicator(),
        seed=args.seed,
    )
    if CorpusSource.EVIDENCE_QA in requested and not any(
        row.built for row in report.sources if row.source is CorpusSource.EVIDENCE_QA
    ):
        # Worth saying out loud: this is the only source that teaches citation,
        # and it is generated from measurements rather than from annotations. No
        # FactSheets, no citation supervision (§4.6).
        print(
            "  note: evidence_qa is generated from the FactSheets the Phase 2 "
            "render pass measures for each patch, and no BigEarthNet sample "
            "carried one. Run the render pass first; the text export publishes "
            "annotations, not measurements."
        )

    for row in report.sources:
        print(
            f"{row.source.value:16s} track {row.track}  built {row.built:>6d}  "
            f"kept {row.kept:>6d}  train {row.train:>6d}  val {row.val:>5d}"
        )
    composition_path = write_composition(args.out, report, requested)
    print(f"composition: {composition_path}")
    print(
        f"corpus: {report.train} train / {report.val} val "
        f"({report.dropped_exact} exact and {report.dropped_near} near duplicates dropped)"
    )
    print(
        f"dedup:  {tally.counts[HashState.HASHED]} rows hashed, "
        f"{tally.counts[HashState.MISSING]} with a missing image, "
        f"{tally.counts[HashState.UNHASHABLE]} naming no image"
    )
    if tally.unhashed:
        # Said out loud, because the alternative was an 18 k-line corpus that
        # claimed strict image-level dedup and had checked nothing.
        print(
            f"WARNING — {tally.unhashed} row(s) entered the corpus without an image "
            "hash; the dedup index did not check them"
        )
    append_manifest(args, report, tally)
    print(f"written to {args.out}")

    # Two post-build assertions, both exit 5 (plan §5.3, §2.5). The files are
    # already written — a failed build is inspectable — but a non-zero exit is
    # what stops run_overnight.sh from training on it.
    short = composition_shortfalls(report, requested)
    if short and args.require_sources:
        print("\nREFUSED — sources below their floor:", file=sys.stderr)
        for line in short:
            print(f"  {line}", file=sys.stderr)
        return EXIT_COMPOSITION
    if short:
        print("\nWARNING — sources below their floor (--no-require-sources):")
        for line in short:
            print(f"  {line}")
    bare = [
        (split, sample_id, answer)
        for split in ("train", "val")
        for sample_id, answer in noncanonical_grounding(read_jsonl(args.out / f"{split}.jsonl"))
    ]
    if bare:
        print(
            f"\nREFUSED — {len(bare)} GROUNDING answer(s) are not in the canonical tagged "
            f"form, e.g. {bare[0]}",
            file=sys.stderr,
        )
        return EXIT_COMPOSITION
    print("grounding: every GROUNDING answer is canonical (tagged boxes or NONE)")
    return 0


EXIT_COMPOSITION: int = 5
"""The build wrote its files but the corpus must not be trained on."""

COMPOSITION_NAME: str = "composition.json"
"""Per-source counts of what was written, beside the JSONL. What
``train_vlm.py`` counts from the lines must agree with this."""


def composition_shortfalls(report: Any, requested: set[CorpusSource]) -> list[str]:
    """Every requested source that contributed nothing, or under half its floor.

    The floor is ``0.5 × min(target, built)``: a source with fewer candidates
    than its target is allowed to be small, but not to lose more than half of
    what it *did* build to dedup and view gaps without someone noticing.
    """
    lines: list[str] = []
    for row in report.sources:
        if row.source not in requested:
            continue
        floor = 0.5 * min(row.target, row.built) if row.target else 0
        if row.train == 0:
            lines.append(
                f"{row.source.value}: 0 train samples (built {row.built}, target {row.target})"
            )
        elif row.train < floor:
            lines.append(
                f"{row.source.value}: {row.train} train samples < floor {floor:.0f} "
                f"(built {row.built}, target {row.target})"
            )
    return lines


def write_composition(out: Path, report: Any, requested: set[CorpusSource]) -> Path:
    """Write ``composition.json``: per-source counts and the build's totals."""
    payload = {
        "built_at": datetime.now(UTC).isoformat(),
        "seed": report.seed,
        "prompt_version": report.prompt_version,
        "requested": sorted(source.value for source in requested),
        "train": report.train,
        "val": report.val,
        "dropped_exact": report.dropped_exact,
        "dropped_near": report.dropped_near,
        "sources": {
            row.source.value: {
                "track": row.track,
                "target": row.target,
                "built": row.built,
                "kept": row.kept,
                "train": row.train,
                "val": row.val,
            }
            for row in report.sources
            if row.built or row.source in requested
        },
    }
    path = Path(out) / COMPOSITION_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


MANIFEST_NAME: str = "MANIFEST.md"
"""Written beside the corpus files: which build produced which JSONL, and how."""


def append_manifest(args: argparse.Namespace, report: Any, tally: HashTally) -> Path:
    """Record this build in ``<out>/MANIFEST.md`` so the directory explains itself.

    The corpus directory has accumulated several overlapping JSONL sets with no
    record of which run consumed which. ``run_manifest.json`` says it per run;
    this says it per corpus.
    """
    path = Path(args.out) / MANIFEST_NAME
    header = (
        "# Corpus manifest\n\n"
        "One row per build, appended by `scripts/build_corpus.py`. "
        "`hashed` is the number of rows the dedup index actually checked.\n\n"
        "| built (UTC) | sources | seed | train | val | hashed | unhashed | dropped | command |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---|\n"
    )
    command = "build_corpus.py " + " ".join(sys.argv[1:])
    row = (
        f"| {datetime.now(UTC).strftime('%Y-%m-%d %H:%M')} "
        f"| {','.join(args.sources)} | {args.seed} | {report.train} | {report.val} "
        f"| {tally.counts[HashState.HASHED]} | {tally.unhashed} "
        f"| {report.dropped_exact + report.dropped_near} | `{command}` |\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(header, encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row)
    return path


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
