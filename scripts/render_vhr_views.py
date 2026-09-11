#!/usr/bin/env python
"""The Phase 2 pre-render pass for the 3-channel VHR sources (§2.4, §7.2).

    uv run python scripts/render_vhr_views.py --source vrsbench --split train
    uv run python scripts/render_vhr_views.py --source cdvqa --workers 16
    uv run python scripts/render_vhr_views.py --source rsvqa_hr --limit 200 --dry-run

``scripts/render_views.py`` is the reBEN pass: it stacks twelve Sentinel-2 bands
and two Sentinel-1 bands onto a common grid, renders the seven-view cross-modal
stack, and measures the NDVI / NDBI / σ⁰ scalars ``evidence_qa`` is generated
from. None of that applies to VRSBench, RSVQA-HR or CDVQA, which publish ordinary
8-bit RGB: there is no NIR to index against and no radar to compare with, so §4.2
- §4.4 ask for ``TC`` and nothing else. Hence a second, much smaller pass rather
than a ``--source`` flag bolted onto the first.

**It renders nothing itself either.** Views come from
:func:`satquery.render.renderer.render_views`, the same function the API calls per
request, driven by the same :func:`~satquery.render.views.select_views` policy: a
single RGB image resolves to ``TC`` alone and a bi-temporal pair to ``TC(pre)``
and ``TC(post)``, because those are the only catalogue entries whose required
bands are present. A pass that cropped and re-encoded the JPEG itself would put a
second renderer behind the corpus, and §2.5 is explicit about what that costs.

**No FactSheets.** ``render_views.py`` writes ``factsheets.jsonl`` because it can
measure real scalars. Here there is nothing to measure — an RGB tile supports no
spectral index — and writing an empty or invented sheet would let a citation
audit pass against numbers no tool produced. The corpus lines these views back
are unsourced by construction (§4.2), and ``evidence_qa`` stays what teaches
citation.

Prerequisites, per source::

    vrsbench   data/raw/vrsbench/Images_{train,val}/    (the published zips, unpacked)
    rsvqa_hr   data/raw/rsvqa_hr/Data/                  (fetch_sources.py --source rsvqa_hr)
    cdvqa      data/raw/cdvqa/images/                   (fetch_sources.py --source cdvqa)

Output is the layout ``build_corpus.py`` resolves by convention::

    data/processed/views/<source>/<item_id>/TC.jpg
    data/processed/views/cdvqa/<pair_id>/TC_pre.jpg
    data/processed/views/cdvqa/<pair_id>/TC_post.jpg
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import traceback
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.render.artifact_store import ArtifactStore  # noqa: E402
from satquery.render.renderer import RenderSource, render_views  # noqa: E402
from satquery.render.views import ImageFormat  # noqa: E402
from satquery.schemas.enums import ImageRole, Modality, PairType  # noqa: E402
from satquery.training.corpus_builder import CorpusSource  # noqa: E402
from satquery.training.local_sources import RAW_ROOT, SPLIT_DIRS  # noqa: E402

RGB_BANDS: Final[dict[str, int]] = {"red": 1, "green": 2, "blue": 3}
"""Band order of an ordinary 8-bit RGB file, which is what all three sources ship.

This is the whole reason ``select_views`` resolves these images to ``TC`` and
stops: every other optical view in the catalogue requires ``nir`` or a SWIR band,
and :meth:`ViewSpec.is_available_for` finds neither here."""

SENSORS: Final[dict[CorpusSource, str]] = {
    CorpusSource.VRSBENCH: "VHR aerial (DOTA-v2 / DIOR, ~0.1-3 m)",
    CorpusSource.RSVQA_HR: "USGS High Resolution Orthoimagery (0.15 m)",
    CorpusSource.CDVQA: "SECOND (aerial, bi-temporal, ~0.5 m)",
}
"""What the label under each rendered view says the image came from (§2.5)."""

FAILURES_SHOWN: Final[int] = 20
EARLY_ABORT_AFTER: Final[int] = 50
"""If the first this-many items all fail, the pass is broken rather than the
data — a wrong ``--input``, most likely. Stop there instead of writing a
directory of nothing and reporting it as a successful render."""

MANIFEST_FILE: Final[str] = "rendered.jsonl"
"""One line per item rendered, beside the views. Not a FactSheet: it records
which files exist so a later ``--on-missing-views skip`` build can be reasoned
about without walking the tree."""


@dataclass(frozen=True)
class Item:
    """One thing to render: its id, its split, and the file(s) it reads."""

    item_id: str
    split: str
    paths: tuple[Path, ...]

    @property
    def is_pair(self) -> bool:
        """True for a bi-temporal item, which renders two views instead of one."""
        return len(self.paths) == 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        required=True,
        choices=[source.value for source in SENSORS],
        help="Which 3-channel source to render.",
    )
    parser.add_argument("--input", type=Path, default=RAW_ROOT)
    parser.add_argument("--out", type=Path, default=Path("data/processed/views"))
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "validation", "all"],
        help="Official split to render. 'all' does train then validation.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Items per split.")
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render an item whose views are already on disk. Off by default, "
        "so an interrupted pass resumes instead of restarting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Locate the items, then stop without rendering.",
    )
    return parser.parse_args(argv)


# ------------------------------------------------------------------- selection


def iter_items(
    source: CorpusSource, split: str, root: Path, limit: int | None
) -> Iterator[Item]:
    """The items of one source and split, in a stable order.

    Sorted rather than taken in directory order, so ``--limit`` selects the same
    items on every machine and a resumed pass covers the same set as the run it
    resumes.

    Raises:
        SystemExit: The source's image directory is not on disk. Raised here, on
            the first item, rather than as 20 000 identical per-item failures.
    """
    suffix = SPLIT_DIRS.get(split, split)
    if source is CorpusSource.VRSBENCH:
        directory = root / source.value / f"Images_{suffix}"
        _require_dir(directory, source)
        found = sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
        for index, path in enumerate(found):
            if limit is not None and index >= limit:
                return
            yield Item(path.stem, suffix, (path,))
        return

    if source is CorpusSource.RSVQA_HR:
        # The split lives in the annotation tables, not the image directory:
        # Zenodo ships one flat `Data/` of all 10 659 tiles and marks split
        # membership with `active` in `USGS_split_<split>_images.json`. Reading
        # the directory alone would render — and later train on — the held-out
        # Philadelphia tiles.
        directory = root / source.value / "Data"
        _require_dir(directory, source)
        for index, image_id in enumerate(_rsvqa_image_ids(root, suffix)):
            if limit is not None and index >= limit:
                return
            path = directory / f"{image_id}.png"
            if path.is_file():
                yield Item(image_id, suffix, (path,))
        return

    directory = root / source.value / "images"
    _require_dir(directory, source)
    wanted = _cdvqa_pair_ids(root, suffix)
    for index, pair_id in enumerate(sorted(wanted)):
        if limit is not None and index >= limit:
            return
        pre = directory / f"{pair_id}.pre.png"
        post = directory / f"{pair_id}.post.png"
        if pre.is_file() and post.is_file():
            yield Item(pair_id, suffix, (pre, post))


_IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})


def _require_dir(directory: Path, source: CorpusSource) -> None:
    """Fail early, and say which command puts the directory there.

    Raises:
        SystemExit: *directory* does not exist.
    """
    if directory.is_dir():
        return
    hint = (
        "unpack the published zips into data/raw/vrsbench"
        if source is CorpusSource.VRSBENCH
        else f"uv run python scripts/fetch_sources.py --source {source.value}"
    )
    raise SystemExit(f"{directory} does not exist. To create it:\n  {hint}")


def _rsvqa_image_ids(root: Path, suffix: str) -> list[str]:
    """The image ids RSVQA-HR's own split table marks as belonging to *suffix*."""
    path = root / CorpusSource.RSVQA_HR.value / f"USGS_split_{suffix}_images.json"
    if not path.is_file():
        raise SystemExit(
            f"{path} does not exist. Fetch RSVQA-HR's split tables first:\n"
            "  uv run python scripts/fetch_sources.py --source rsvqa_hr"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        str(row["id"]) for row in payload[next(iter(payload))] if row.get("active")
    ]


def _cdvqa_pair_ids(root: Path, suffix: str) -> set[str]:
    """The pair ids that appear in CDVQA's unpacked records for *suffix*.

    Read from the records rather than from the image directory because the
    directory is shared between splits: ``fetch_sources`` writes one copy of each
    pair's pixels and lets both split files reference it.
    """
    path = root / CorpusSource.CDVQA.value / f"{suffix}.jsonl"
    if not path.is_file():
        raise SystemExit(
            f"{path} does not exist. Fetch CDVQA first:\n"
            "  uv run python scripts/fetch_sources.py --source cdvqa"
        )
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(str(json.loads(line)["pair_id"]))
    return ids


# --------------------------------------------------------------------- render


def view_names(item: Item) -> tuple[str, ...]:
    """The file stems this item's views are written under.

    A bi-temporal item renders ``TC`` twice — once per date — so the two are
    suffixed rather than given new catalogue ids: both really are true colour,
    and ``build_corpus.view_paths`` reads the format off the ``TC`` spec for
    exactly that reason.
    """
    return ("TC_pre", "TC_post") if item.is_pair else ("TC",)


def already_rendered(item: Item, out_root: Path, source: CorpusSource) -> bool:
    """True when every view this item would write is already on disk."""
    target = out_root / source.value / item.item_id
    return all((target / f"{name}.jpg").is_file() for name in view_names(item))


def render_item(item: Item, out_root: Path, source: CorpusSource, size: int) -> dict[str, Any]:
    """Render one item's views through the production renderer.

    Returns:
        The manifest record for this item.

    Raises:
        RenderError: The image could not be read, or the policy selected no view
            for it — which means the file is not the 3-channel RGB this pass
            assumes and should be looked at, not skipped silently.
    """
    sensor = SENSORS[source]
    sources = [
        RenderSource(
            image_id=f"{item.item_id}:{index}" if item.is_pair else item.item_id,
            path=path,
            modality=Modality.OPTICAL,
            resolved_bands=dict(RGB_BANDS),
            sensor=sensor,
            role=(ImageRole.PRE, ImageRole.POST)[index] if item.is_pair else ImageRole.SINGLE,
        )
        for index, path in enumerate(item.paths)
    ]
    pair_type = PairType.BI_TEMPORAL if item.is_pair else PairType.SINGLE

    target = out_root / source.value / item.item_id
    target.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as scratch:
            # `render_views` stores every view it renders, because at inference
            # the artifact is what the UI fetches. Here the corpus reads the
            # files this pass writes, so the store is scratch: pointing it at
            # the view directory left a second encoded copy of all 46 000
            # images beside the first, doubling the pass's disk for nothing.
            views = render_views(
                sources=sources,
                pair_type=pair_type,
                trace_id=item.item_id,
                store=ArtifactStore(Path(scratch)),
                size=size,
            )
    except Exception as error:  # noqa: BLE001 - re-raised as this pass's own type
        raise RenderError(f"{item.paths[0].name}: {type(error).__name__}: {error}") from error

    names = view_names(item)
    if len(views) != len(names):
        raise RenderError(
            f"{item.item_id}: expected {len(names)} view(s) but the policy selected "
            f"{len(views)} ({[v.view_id.value for v in views]}). The file is probably "
            "not 3-channel RGB"
        )

    written: dict[str, str] = {}
    for name, view in zip(names, views, strict=True):
        path = target / f"{name}.{'png' if view.view.image_format is ImageFormat.PNG else 'jpg'}"
        _write_image(view.rgb, path, view.view.image_format)
        written[name] = str(path)

    return {
        "item_id": item.item_id,
        "split": item.split,
        "source": source.value,
        "pair_type": pair_type.value,
        "views": written,
        "view_labels": {name: view.label for name, view in zip(names, views, strict=True)},
        "sensor": sensor,
    }


class RenderError(RuntimeError):
    """One item could not be rendered. Carries the item and the reason, both."""


def _write_image(rgb: np.ndarray, path: Path, image_format: ImageFormat) -> None:
    """Encode one view, PNG for measurement views and JPEG q92 for composites (§2.1)."""
    from PIL import Image

    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    if image_format is ImageFormat.PNG:
        image.save(path, format="PNG", optimize=True)
    else:
        image.save(path, format="JPEG", quality=92, subsampling=0)


def run(
    items: Sequence[Item], out_root: Path, source: CorpusSource, size: int, workers: int
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Render every item, reporting progress and every failure as it goes.

    Returns:
        ``(records, failures)`` where each failure is ``(item_id, reason)``.
    """
    records: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(render_item, item, out_root, source, size): item for item in items
        }
        for index, future in enumerate(list(futures), start=1):
            item = futures[future]
            try:
                records.append(future.result())
            except RenderError as error:
                failures.append((item.item_id, str(error)))
                if len(failures) <= FAILURES_SHOWN:
                    print(f"  !! {error}", file=sys.stderr)
            except Exception as error:  # noqa: BLE001 - reported, then re-raised
                print(
                    f"\n{item.item_id} failed with an error this pass does not know "
                    f"how to interpret; aborting so it is not mistaken for a bad "
                    f"image:\n{traceback.format_exc()}",
                    file=sys.stderr,
                )
                raise SystemExit(f"{item.item_id}: {type(error).__name__}: {error}") from error
            if index == EARLY_ABORT_AFTER and not records:
                raise SystemExit(
                    f"the first {EARLY_ABORT_AFTER} items all failed to render — that "
                    "is a bug in this pass or a wrong --input, not bad data"
                )
            if index % 500 == 0 or index == len(futures):
                print(f"  rendered {index}/{len(futures)}  ({len(failures)} unreadable)")
    return records, failures


def main(argv: Sequence[str] | None = None) -> int:
    """Render one source's views. Returns a process exit code."""
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(line_buffering=True)

    source = CorpusSource(args.source)
    splits = ["train", "validation"] if args.split == "all" else [args.split]

    all_records: list[dict[str, Any]] = []
    all_failures: list[tuple[str, str]] = []
    for split in splits:
        items = list(iter_items(source, split, args.input, args.limit))
        pending = (
            items
            if args.overwrite
            else [i for i in items if not already_rendered(i, args.out, source)]
        )
        print(
            f"{source.value} {split}: {len(items)} items, {len(pending)} to render "
            f"({len(items) - len(pending)} already on disk)"
        )
        if args.dry_run:
            for item in items[:5]:
                print(f"  {item.item_id}  {[p.name for p in item.paths]}")
            continue
        if not pending:
            continue
        records, failures = run(pending, args.out, source, args.size, args.workers)
        all_records.extend(records)
        all_failures.extend(failures)

    if args.dry_run:
        print("dry run: nothing rendered.")
        return 0

    manifest = args.out / source.value / MANIFEST_FILE
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    views_written = sum(len(record["views"]) for record in all_records)
    print(f"\nitems rendered : {len(all_records)}")
    print(f"items failed   : {len(all_failures)}")
    print(f"views written  : {views_written}")
    print(f"manifest       : {manifest}")
    if all_failures:
        report = args.out / source.value / "failures.txt"
        report.write_text(
            "".join(f"{item_id}\t{reason}\n" for item_id, reason in all_failures),
            encoding="utf-8",
        )
        print(f"failure reasons: {report}")
    return 0 if all_records or not all_failures else 1


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
