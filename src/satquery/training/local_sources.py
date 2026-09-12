"""Readers for the corpus sources that live on disk rather than behind ``load_dataset``.

``scripts/build_corpus.py`` opens most sources with ``datasets.load_dataset``,
which is right when the Hub publishes a tabular export. Three of the six §5
sources it is wrong for, each in its own way:

``vrsbench``
    ``xiang709/VRSBench`` publishes *archives*, not tables — two annotation zips
    and two 4-8 GB image zips. Hugging Face's data-file inference sweeps every
    file whose name contains ``train`` or ``val`` into the split of that name and
    hands the lot to the JSON packaged builder, so the builder is asked to parse
    ``Images_train.zip`` as JSON and pyarrow raises ``ArrowInvalid: JSON parse
    error: Invalid value. in row 0`` on the first PNG header it meets. That is
    the crash that got VRSBench excluded, and it only fires on a full pass: a
    small ``--limit`` stops inside the annotation zip and never reaches the
    images. Reading the extracted annotation directory sidesteps the inference
    entirely.

``rsvqa_hr``
    Not on the Hub at all. See :mod:`scripts.fetch_sources`.

``cdvqa``
    On the Hub as a WebDataset of image bytes and LLaVA turns, which
    ``fetch_sources`` flattens into the row shape ``from_cdvqa`` reads.

Every reader here yields **plain rows**, tagged with the official split they came
from, and converts nothing: conversion stays in
:mod:`satquery.training.corpus_builder`, which is where the rules that matter
live. Rows are yielded one at a time, because the build these feed is streaming
and materialising 625 000 RSVQA rows is exactly the shape of the run that got
OOM-killed.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from satquery.training.corpus_builder import CorpusError, CorpusSource

RAW_ROOT: Final[Path] = Path("data/raw")
"""Where ``fetch_sources.py`` and the VRSBench zips land."""

SPLIT_DIRS: Final[dict[str, str]] = {"train": "train", "validation": "val"}
"""Corpus split name -> the suffix these releases spell it with on disk."""

RSVQA_TYPES: Final[tuple[str, ...]] = ("presence", "comp", "count", "area")
"""RSVQA-HR's four question types, in the order §4.3 balances them.

The raw distribution is wildly uneven — 209 k comparison against 91 k area in
the train split — and a straight prefix of the file teaches the model to guess
the majority type. Rows are emitted round-robin across these four, so *any*
prefix of the stream is balanced and ``--limit`` inherits the balance for free."""

RSVQA_SENSOR: Final[str] = "USGS High Resolution Orthoimagery"
RSVQA_GSD_M: Final[float] = 0.15
"""RSVQA-HR's tiles are 512x512 at 15 cm (the release's ``res_x`` is ``.1524m``)."""


def raw_dir(source: CorpusSource, root: Path | None = None) -> Path:
    """Where one source's downloaded files live."""
    return (root or RAW_ROOT) / source.value


def is_available(source: CorpusSource, root: Path | None = None) -> bool:
    """True when *source* has been fetched and can be read from disk."""
    probe = {
        CorpusSource.VRSBENCH: lambda base: (base / "Annotations_train").is_dir(),
        CorpusSource.RSVQA_HR: lambda base: (
            base / "USGS_split_train_questions.json"
        ).is_file(),
        CorpusSource.CDVQA: lambda base: (base / "train.jsonl").is_file(),
    }.get(source)
    return probe is not None and probe(raw_dir(source, root))


# ---------------------------------------------------------------------- VRSBench


def iter_vrsbench(
    split: str, limit: int | None = None, root: Path | None = None
) -> Iterator[Mapping[str, Any]]:
    """Yield one row per VRSBench image in *split*, read from the extracted zips.

    ``Annotations_{train,val}/`` holds one JSON object per image — ``caption``,
    ``qa_pairs``, ``objects`` and the image file name — which is already the row
    ``from_vrsbench`` reads. The 29 614 images divide 20 264 / 9 350 between the
    two directories, so these two directories *are* the whole published dataset
    and the quarantined test split is simply not among the files.

    Files are walked in sorted order rather than in directory order, so a
    ``--limit`` run selects the same images on every machine.

    Raises:
        CorpusError: The annotation directory for *split* is not on disk.
    """
    base = raw_dir(CorpusSource.VRSBENCH, root)
    suffix = SPLIT_DIRS.get(split, split)
    directory = base / f"Annotations_{suffix}"
    if not directory.is_dir():
        raise CorpusError(
            f"{directory} does not exist. Fetch and extract VRSBench first:\n"
            f"  hf download xiang709/VRSBench --repo-type dataset --local-dir {base} "
            f'--include "Annotations_*.zip" --include "Images_*.zip"\n'
            f"  for z in {base}/*.zip; do unzip -q -d {base} \"$z\"; done"
        )
    for index, path in enumerate(sorted(directory.glob("*.json"))):
        if limit is not None and index >= limit:
            return
        row = json.loads(path.read_text(encoding="utf-8"))
        # The published object names the image but not the split it sits in, and
        # `_guard_quarantine` reads `split` off the row. Stamping it here is what
        # makes the quarantine check meaningful rather than a default.
        yield dict(row, split=suffix, image_id=Path(str(row.get("image") or path.stem)).stem)


# ---------------------------------------------------------------------- RSVQA-HR


def _rsvqa_payload(base: Path, suffix: str, kind: str) -> list[dict[str, Any]]:
    """One of RSVQA-HR's three per-split tables, unwrapped from its envelope."""
    path = base / f"USGS_split_{suffix}_{kind}.json"
    if not path.is_file():
        raise CorpusError(
            f"{path} does not exist. Fetch RSVQA-HR first:\n"
            "  uv run python scripts/fetch_sources.py --source rsvqa_hr"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise CorpusError(f"{path}: expected a one-key JSON object wrapping the table")
    table = payload[next(iter(payload))]
    if not isinstance(table, list):
        raise CorpusError(f"{path}: expected the wrapped table to be a JSON array")
    return cast(list[dict[str, Any]], table)


def iter_rsvqa_hr(
    split: str,
    limit: int | None = None,
    root: Path | None = None,
    seed: int = 42,
) -> Iterator[Mapping[str, Any]]:
    """Yield RSVQA-HR rows for *split*, balanced across the four question types.

    **The ``active`` flag is the split.** Each ``USGS_split_<split>_*.json`` file
    contains *every* question in the release — 955 664 of them in the train file
    — and marks the ones belonging to that split with ``active: true``. Reading
    the file at face value therefore pulls the official test and Philadelphia
    splits straight into training: 330 324 of the train file's rows are not
    train rows. §4.7 makes that a build error, so it is filtered here, once, and
    the same flag is applied to the image table so an id that leaks past the
    question filter still has nowhere to resolve.

    Questions and answers are joined on ``question_id``; a question whose answer
    is inactive is dropped rather than guessed at.

    Args:
        split: ``"train"`` or ``"validation"``.
        limit: Stop after this many rows. The round-robin means a limited run is
            still balanced across the four types.
        root: Raw data root; defaults to ``data/raw``.
        seed: Shuffles within each type, so a limited run is a random sample of
            that type rather than the first images in id order.
    """
    base = raw_dir(CorpusSource.RSVQA_HR, root)
    suffix = SPLIT_DIRS.get(split, split)

    answers = {
        int(row["question_id"]): str(row["answer"])
        for row in _rsvqa_payload(base, suffix, "answers")
        if row.get("active")
    }
    images = {
        int(row["id"]): row
        for row in _rsvqa_payload(base, suffix, "images")
        if row.get("active")
    }

    buckets: dict[str, list[tuple[str, int, int, str, str]]] = {
        kind: [] for kind in RSVQA_TYPES
    }
    for row in _rsvqa_payload(base, suffix, "questions"):
        if not row.get("active"):
            continue
        kind = str(row.get("type"))
        bucket = buckets.get(kind)
        if bucket is None:
            continue
        question_id = int(row["id"])
        image_id = int(row["img_id"])
        answer = answers.get(question_id)
        if answer is None or image_id not in images:
            continue
        bucket.append((kind, question_id, image_id, str(row["question"]), answer))
    del answers

    rng = random.Random(seed)
    for entries in buckets.values():
        rng.shuffle(entries)

    for yielded, (kind, question_id, image_id, question, answer) in enumerate(
        _round_robin([buckets[name] for name in RSVQA_TYPES])
    ):
        if limit is not None and yielded >= limit:
            return
        record = images[image_id]
        yield {
            "image_id": str(image_id),
            "question_id": question_id,
            # `from_rsvqa` switches on `type == "count"`; RSVQA spells the other
            # three `presence` / `comp` / `area`, all of which are plain VQA.
            "type": kind,
            "question": question,
            "answer": answer,
            "split": suffix,
            "sensor": str(record.get("sensor") or RSVQA_SENSOR),
            "gsd_m": RSVQA_GSD_M,
        }


def _round_robin(groups: Sequence[Sequence[Any]]) -> Iterator[Any]:
    """Interleave *groups*, one element at a time, until all are exhausted."""
    depth = 0
    remaining = True
    while remaining:
        remaining = False
        for group in groups:
            if depth < len(group):
                remaining = True
                yield group[depth]
        depth += 1


# ------------------------------------------------------------------------ CDVQA


CDVQA_UNCITABLE_TYPES: Final[frozenset[str]] = frozenset(
    {"change_ratio", "change_ratio_types"}
)
"""CDVQA question types withheld from the corpus, and why.

Both answer with a **binned percentage** — ``"10_to_20"``, ``"0_to_10"``, or a
bare ``"0"`` — meaning "between a tenth and a fifth of the scene changed".
§4.6's citation audit rejects an assistant turn stating a number the FactSheet
does not support, and there is nothing here to support one: the bin is a range,
the sheet holds scalars, and ``change_statistics`` measures a single
``changed_area_pct`` at inference. The three ways to keep these rows are all
worse than dropping them — train on an uncited percentage, which is exactly the
hallucination ``evidence_qa`` exists to suppress; invent a
``changed_area_pct_min`` / ``_max`` pair no tool emits, so the model cites keys
the server never injects; or write the bin's midpoint into the sheet as if it
had been measured, which is a fabricated fact.

They are 19.6 % of the fetched train rows. ``--shards`` is what makes that back
up; a silently uncitable fifth of a source is not."""


def iter_cdvqa(
    split: str, limit: int | None = None, root: Path | None = None
) -> Iterator[Mapping[str, Any]]:
    """Yield CDVQA rows for *split* from the JSONL ``fetch_sources`` unpacked.

    Each row already carries ``pair_id``, ``question``, ``answer`` and the
    ``pre_path`` / ``post_path`` of the SECOND image pair it asks about, which is
    the shape ``from_cdvqa`` reads. Only the ``train`` and ``val`` shards are
    ever unpacked, so the quarantined ``test`` / ``test2`` splits are absent by
    construction rather than by filter — but the row's own ``split`` field is
    still what ``_guard_quarantine`` checks, so a mis-fetched shard fails the
    build instead of entering it.

    The two question types in :data:`CDVQA_UNCITABLE_TYPES` are withheld. The
    filter is here rather than in the adapter deliberately: which rows enter the
    corpus is a data-selection decision, and ``from_cdvqa`` should keep failing
    loudly on an uncitable answer rather than learning to shrug at one.

    ``limit`` counts rows *yielded*, so a smoke run gets the number it asked for
    instead of however many survive the filter.

    Raises:
        CorpusError: The split's JSONL is not on disk.
    """
    base = raw_dir(CorpusSource.CDVQA, root)
    suffix = SPLIT_DIRS.get(split, split)
    path = base / f"{suffix}.jsonl"
    if not path.is_file():
        raise CorpusError(
            f"{path} does not exist. Fetch CDVQA first:\n"
            "  uv run python scripts/fetch_sources.py --source cdvqa"
        )
    yielded = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and yielded >= limit:
                return
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("question_type")) in CDVQA_UNCITABLE_TYPES:
                continue
            yielded += 1
            yield row


READERS: Final[dict[CorpusSource, Any]] = {
    CorpusSource.VRSBENCH: iter_vrsbench,
    CorpusSource.RSVQA_HR: iter_rsvqa_hr,
    CorpusSource.CDVQA: iter_cdvqa,
}
"""The sources ``build_corpus.py`` reads from disk instead of from the Hub."""
