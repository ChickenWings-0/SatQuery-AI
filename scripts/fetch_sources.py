#!/usr/bin/env python
"""Fetch the two corpus sources that ``datasets.load_dataset`` cannot reach (§4.3, §4.4).

    uv run python scripts/fetch_sources.py --source rsvqa_hr
    uv run python scripts/fetch_sources.py --source cdvqa --shards 90 --val-shards 20

Four of the six §5 sources are one ``load_dataset`` call away. Two are not, and
they are not for opposite reasons:

``rsvqa_hr``
    The canonical RSVQA-HR release is **not on the Hub at all** — it is Zenodo
    record 6344367, seventeen loose files: one 13.5 GB ``Images.tar`` of 512x512
    RGB tiles plus a question / answer / image-id triple per official split. The
    Hub's only RSVQA-HR entry, ``dmarsili/RSVQA-HR-2k``, publishes a *validation*
    split of 2 000 rows: an evaluation subset, and training on it would train on
    the thing we measure with. So the split assignment is taken from Zenodo's own
    ``USGS_split_*_images.json``, which is what every published RSVQA number is
    computed against.

``cdvqa``
    ``ljx620/CDVQA`` is on the Hub but is a **WebDataset**: 1 533 tar shards of
    raw image bytes, three members per sample (``<key>.0.img`` the pre image,
    ``<key>.1.img`` the post image, ``<key>.json`` the turn plus its metadata).
    ``load_dataset`` on it yields LLaVA conversation dicts and inline bytes,
    which is neither the row shape ``from_cdvqa`` reads nor something the render
    pass can address by path. Shards are therefore unpacked here, once, into the
    pre/post PNG pairs and the flat record rows the rest of the pipeline expects.

Both land under ``data/raw/<source>/`` and both are **idempotent**: a file whose
size already matches the manifest is not re-fetched, so an interrupted 13.5 GB
download resumes at the shard it stopped on rather than from zero.

Nothing here is quarantine-blind. Only ``train`` and ``val`` shards are fetched;
CDVQA's ``test`` / ``test2`` and RSVQA-HR's ``test`` / ``test_phili`` splits stay
on the server, except for their *image id lists*, which are downloaded precisely
so §4.7 can assert that none of those ids reached the corpus.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import urllib.request
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Final

ZENODO_RECORD: Final[str] = "6344367"
"""The canonical RSVQA-HR deposit — "Remote Sensing VQA - High Resolution"."""

RSVQA_SPLIT_FILES: Final[tuple[str, ...]] = (
    "USGS_split_train_questions.json",
    "USGS_split_train_answers.json",
    "USGS_split_train_images.json",
    "USGS_split_val_questions.json",
    "USGS_split_val_answers.json",
    "USGS_split_val_images.json",
)
"""The train and val halves of the official split, question / answer / image."""

RSVQA_QUARANTINE_FILES: Final[tuple[str, ...]] = (
    "USGS_split_test_images.json",
    "USGS_split_test_phili_images.json",
)
"""The held-out image id lists — fetched, but only so §4.7 can exclude them.

RSVQA-HR publishes two test sets: ``test`` (same geography as train) and
``test_phili`` (Philadelphia, a held-out city). Their *questions* are deliberately
not downloaded; their image ids are, because an assertion that no training image
appears in a test split needs the test split's ids to assert against."""

RSVQA_IMAGES_TAR: Final[str] = "Images.tar"

CDVQA_REPO: Final[str] = "ljx620/CDVQA"
CDVQA_MANIFEST: Final[str] = "manifest.json"

HF_RESOLVE: Final[str] = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
ZENODO_FILE: Final[str] = "https://zenodo.org/records/{record}/files/{name}?download=1"

CHUNK: Final[int] = 1 << 20

SHARD_WORKERS: Final[int] = 8
"""Concurrent CDVQA shard downloads. The Hub throttles a single connection well
below the link, and shards are independent 79 MB objects."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        required=True,
        choices=("rsvqa_hr", "cdvqa"),
        help="Which off-Hub source to fetch.",
    )
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--shards",
        type=int,
        default=90,
        help="CDVQA train shards to unpack. 100 samples each; the §5 target is "
        "8,000, so 90 leaves headroom for the dedup pass to drop into.",
    )
    parser.add_argument(
        "--val-shards",
        type=int,
        default=20,
        help="CDVQA validation shards to unpack.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="RSVQA-HR: fetch the annotation JSON only, not the 13.5 GB Images.tar. "
        "Enough to build and count a corpus; not enough to render one.",
    )
    return parser.parse_args(argv)


# ------------------------------------------------------------------ downloading


def _remote_size(url: str) -> int | None:
    """Content length of *url*, or None when the server will not say."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except OSError:
        return None


def download(url: str, destination: Path, *, expect: int | None = None) -> Path:
    """Fetch *url* to *destination*, skipping a file that is already complete.

    "Already complete" means the size on disk matches the size the server
    reports. A half-written 13.5 GB tar is the realistic interruption here, and
    treating "the path exists" as "the file is there" is what turns that into a
    tar error four hours later instead of a resumed download now.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = expect if expect is not None else _remote_size(url)
    if destination.exists() and size is not None and destination.stat().st_size == size:
        print(f"  have  {destination.name} ({size:,} bytes)")
        return destination
    if destination.exists() and size is None and destination.stat().st_size:
        print(f"  have  {destination.name} (size unverifiable, keeping)")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"  fetch {destination.name}" + (f" ({size:,} bytes)" if size else ""))
    written = 0
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        while chunk := response.read(CHUNK):
            handle.write(chunk)
            written += len(chunk)
            if size and written % (256 * CHUNK) < CHUNK:
                print(f"    {written / size:6.1%}  {written:,} / {size:,}", flush=True)
    partial.replace(destination)
    return destination


# ------------------------------------------------------------------- RSVQA-HR


def fetch_rsvqa(out: Path, skip_images: bool) -> int:
    """Download RSVQA-HR's official splits, and its imagery unless told not to."""
    root = out / "rsvqa_hr"
    for name in RSVQA_SPLIT_FILES + RSVQA_QUARANTINE_FILES:
        download(ZENODO_FILE.format(record=ZENODO_RECORD, name=name), root / name)

    if skip_images:
        print("\n--skip-images: Images.tar not fetched; renders will have no pixels.")
    else:
        tar_path = download(
            ZENODO_FILE.format(record=ZENODO_RECORD, name=RSVQA_IMAGES_TAR),
            root / RSVQA_IMAGES_TAR,
        )
        extract_rsvqa_images(tar_path, root)

    counts = {}
    for name in RSVQA_SPLIT_FILES:
        payload = json.loads((root / name).read_text(encoding="utf-8"))
        key = next(iter(payload))
        counts[name] = len(payload[key])
    print()
    for name, count in counts.items():
        print(f"  {name:38s} {count:>9,} rows")
    return 0


def extract_rsvqa_images(tar_path: Path, root: Path) -> None:
    """Unpack ``Images.tar`` into ``<root>/Data/<image_id>.png``.

    The archive nests its tiles under ``Data/`` and ships each one **twice** —
    ``<id>.tif``, the georeferenced original, and ``<id>.png``, the same 512x512
    RGB pixels already decoded. Only the PNG is extracted: §4.3 renders ``TC``
    and nothing else, so the GeoTIFF's projection is 8 GB of disk bought for
    metadata no view in the catalogue reads. Members are extracted flat so the
    id in ``USGS_split_*_images.json`` addresses the file directly, and
    extraction is skipped when the directory is already populated, because
    re-extracting 10 659 tiles on every invocation is twenty wasted minutes.
    """
    data_dir = root / "Data"
    if data_dir.is_dir():
        have = sum(1 for _ in data_dir.glob("*.png"))
        if have >= 10_000:
            print(f"  have  {data_dir} ({have:,} tiles)")
            return
    print(f"  unpack {tar_path.name} -> {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with tarfile.open(tar_path) as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if not name.lower().endswith(".png"):
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            (data_dir / name).write_bytes(source.read())
            written += 1
            if written % 1000 == 0:
                print(f"    {written:,} tiles", flush=True)
    print(f"    {written:,} tiles")


# ---------------------------------------------------------------------- CDVQA


def cdvqa_manifest() -> dict[str, Any]:
    """The shard index the repository publishes at its root."""
    url = HF_RESOLVE.format(repo=CDVQA_REPO, path=CDVQA_MANIFEST)
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def _fetch_shard(path: str) -> bytes:
    """One CDVQA shard's bytes, whole. Small enough to hold; 1 533 are not."""
    url = HF_RESOLVE.format(repo=CDVQA_REPO, path=path)
    with urllib.request.urlopen(url) as response:
        return response.read()


def iter_shard_samples(blob: bytes) -> Iterator[tuple[str, dict[str, Any], bytes, bytes]]:
    """Group one WebDataset shard's members into whole samples.

    A shard is a flat tar of ``<key>.0.img`` / ``<key>.1.img`` / ``<key>.json``
    triples. They are grouped by key rather than assumed to arrive in order, and
    a key missing any of its three members is dropped: a half sample is a pair
    with one image, which renders to a bi-temporal view with nothing to compare.
    """
    members: dict[str, dict[str, bytes]] = {}
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = Path(member.name).name
            key, _, suffix = name.partition(".")
            source = archive.extractfile(member)
            if source is not None:
                members.setdefault(key, {})[suffix] = source.read()
    for key, parts in members.items():
        if {"0.img", "1.img", "json"} <= set(parts):
            yield key, json.loads(parts["json"]), parts["0.img"], parts["1.img"]


def cdvqa_record(key: str, payload: dict[str, Any], root: Path) -> dict[str, Any] | None:
    """Flatten one WebDataset sample into the row ``from_cdvqa`` reads.

    The published sample is a LLaVA turn — ``conversations`` alternating
    ``user`` / ``gpt``, the question carrying ``Image 1: <image>`` preamble — and
    ``from_cdvqa`` wants ``pair_id``, ``question``, ``answer`` and the two image
    paths. The ``<image>`` placeholders are stripped here rather than in the
    adapter: the corpus's own prompt builder inserts image references from the
    view list, and leaving a second set in the question text trains the model to
    emit tokens the server never sends.
    """
    meta = payload.get("meta") or {}
    turns = payload.get("conversations") or []
    question = next((t["value"] for t in turns if t.get("from") == "user"), None)
    answer = next((t["value"] for t in turns if t.get("from") == "gpt"), None)
    if question is None or answer is None:
        return None
    question = " ".join(
        line
        for line in str(question).splitlines()
        if "<image>" not in line
    ).strip()
    pair_id = Path(str(meta.get("image_id") or key)).stem
    return {
        "pair_id": pair_id,
        "question_id": meta.get("question_id"),
        "question": question,
        "answer": str(answer).strip(),
        "question_type": meta.get("question_type"),
        "split": str(meta.get("split") or "train"),
        "sensor": "SECOND (aerial, bi-temporal)",
        "gsd_m": 0.5,
        "pre_path": str(root / "images" / f"{pair_id}.pre.png"),
        "post_path": str(root / "images" / f"{pair_id}.post.png"),
    }


def fetch_cdvqa(out: Path, shards: int, val_shards: int) -> int:
    """Unpack CDVQA's train and val shards into pre/post pairs and record rows."""
    root = out / "cdvqa"
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest = cdvqa_manifest()

    totals: dict[str, int] = {}
    for split, wanted in (("train", shards), ("val", val_shards)):
        spec = manifest["splits"][split]
        files = spec["shard_files"][:wanted]
        rows: list[dict[str, Any]] = []
        seen_pairs: set[str] = set()
        print(f"{split}: {len(files)} of {spec['shards']} shards")
        # Fetched in parallel, unpacked in order. A shard is ~79 MB and one
        # connection to the Hub runs well under a megabyte a second, so the
        # 90-shard default is the difference between three hours and twenty
        # minutes. Results are consumed as an ordered map rather than as they
        # complete: the record file is the corpus's input, and a reproducible
        # build wants it byte-identical between runs.
        with ThreadPoolExecutor(max_workers=SHARD_WORKERS) as pool:
            for index, blob in enumerate(pool.map(_fetch_shard, files), start=1):
                for key, payload, pre, post in iter_shard_samples(blob):
                    record = cdvqa_record(key, payload, root)
                    if record is None:
                        continue
                    if record["pair_id"] not in seen_pairs:
                        # One pair carries ~22 questions and the shard repeats
                        # its bytes for each. Writing them once is a 20x saving
                        # on both disk and the pass that reads them back.
                        Path(record["pre_path"]).write_bytes(pre)
                        Path(record["post_path"]).write_bytes(post)
                        seen_pairs.add(record["pair_id"])
                    rows.append(record)
                if index % 10 == 0 or index == len(files):
                    print(
                        f"  {index}/{len(files)} shards  {len(rows):,} rows  "
                        f"{len(seen_pairs):,} pairs",
                        flush=True,
                    )
        path = root / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        totals[split] = len(rows)
        print(f"  wrote {path}  ({len(rows):,} rows, {len(seen_pairs):,} image pairs)")

    print()
    for split, count in totals.items():
        print(f"  cdvqa {split:5s} {count:>9,} rows")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch one source. Returns a process exit code."""
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(line_buffering=True)
    if args.source == "rsvqa_hr":
        return fetch_rsvqa(args.out, args.skip_images)
    return fetch_cdvqa(args.out, args.shards, args.val_shards)


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
