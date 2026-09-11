#!/usr/bin/env python
"""The Phase 2 pre-render pass (DATA_ADAPTATION_PLAN §2, §7.2).

    uv run python scripts/render_views.py --limit 20000 --split train
    uv run python scripts/render_views.py --limit 200 --split validation --workers 8

Walks the reBEN patches on disk, renders the named views §2.4 selects, writes
them where the corpus expects to find them, and records each patch's measured
scalars so ``evidence_qa`` can be regenerated from real numbers instead of mock
sheets.

**It renders nothing itself.** Views come from
:func:`satquery.render.renderer.render_views` — the same function the API calls
per request — and the scalars from :mod:`satquery.render.indices`, the same
module ``spectral_index_analyzer`` and ``sar_backscatter_analyzer`` use, with the
same prefixes. That is deliberate: a pre-render pass that computed its own NDVI
would put a second implementation behind the corpus, and the FactSheets it wrote
would drift from the ones the server measures at inference — the same class of
silent, total failure as a divergent view label (§2.5).

**Layout.** reBEN ships one GeoTIFF per band *at its native resolution* — the
10 m bands are 120 x 120, the 20 m bands 60 x 60, the 60 m bands 20 x 20 — and
the renderer reads one multi-band raster. So each patch's bands are resampled
onto the 10 m grid and stacked into a temporary GeoTIFF that is deleted once its
views are written. The stack is the only intermediate; nothing else is
materialised.

Prerequisites, in order::

    hf download torchgeo/bigearthnet --repo-type dataset --local-dir data/raw/ben \
        --include "V2/*"
    cat data/raw/ben/V2/BigEarthNet-S2.tar.gza* > data/raw/ben/V2/S2.tar.gz
    cat data/raw/ben/V2/BigEarthNet-S1.tar.gza* > data/raw/ben/V2/S1.tar.gz
    tar -xzf data/raw/ben/V2/S2.tar.gz -C data/raw/ben
    tar -xzf data/raw/ben/V2/S1.tar.gz -C data/raw/ben

``torchgeo/bigearthnet`` V2 is the mirror this was verified against: its
``patch_id`` and ``s1_name`` match the annotation export byte for byte, and its
split assignment agrees with the export on every patch checked except the ones
the export calls ``bench``, which it calls ``test`` — held-out either way.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.ingest.bands import load_table  # noqa: E402
from satquery.render.artifact_store import ArtifactStore  # noqa: E402
from satquery.render.indices import (  # noqa: E402
    index_statistics,
    ndbi,
    ndvi,
    ndwi,
    to_db,
    vv_vh_ratio_db,
)
from satquery.render.renderer import RenderSource, render_views  # noqa: E402
from satquery.render.views import ImageFormat  # noqa: E402
from satquery.schemas.enums import ImageRole, Modality, PairType  # noqa: E402

MIRROR: Final[str] = "torchgeo/bigearthnet"
METADATA_FILE: Final[str] = "V2/metadata.parquet"
SOURCE_NAME: Final[str] = "bigearthnet_v2"

S2_SENSOR: Final[str] = "Sentinel-2 (BigEarthNet-v2, 12-band)"
S1_SENSOR: Final[str] = "Sentinel-1 GRD"

S2_BAND_FILES: Final[tuple[str, ...]] = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12",
)
"""The 12 bands reBEN ships, in the order the alias table indexes them (B10 is
excluded from the product, which is why ``swir1`` is band 11 and not 12)."""

S1_BAND_FILES: Final[tuple[str, ...]] = ("VV", "VH")

FACTSHEET_FILE: Final[str] = "factsheets.jsonl"
"""Written beside the views. One line per patch: the measured scalars, namespaced
by the tool that would have measured them at inference."""

EARLY_ABORT_AFTER: Final[int] = 50
"""If the first this-many patches yield nothing, the pass is broken rather than
the data. Stop there — the run this replaced reported 20,000 "unreadable"
patches over six hours and then wrote an empty corpus."""

FAILURES_SHOWN: Final[int] = 20
"""How many per-patch failures are printed in full before the log is spared the
rest. The count is always exact; the reasons are capped."""

SPECTRAL_TOOL: Final[str] = "spectral_index_analyzer"
SAR_TOOL: Final[str] = "sar_backscatter_analyzer"


@dataclass(frozen=True)
class Patch:
    """One reBEN patch: where its bands are, and what the metadata says it is."""

    patch_id: str
    s1_name: str
    split: str
    labels: tuple[str, ...]
    s2_dir: Path
    s1_dir: Path | None

    @property
    def has_radar(self) -> bool:
        """True when the S1 half of the pair was found on disk."""
        return self.s1_dir is not None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("data/raw/ben"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/views"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=f"reBEN metadata parquet. Downloaded from {MIRROR} when omitted.",
    )
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Patches to render.")
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select and locate the patches, then stop without rendering.",
    )
    return parser.parse_args(argv)


def metadata_path(explicit: Path | None) -> Path:
    """The reBEN metadata table, fetched from the verified mirror if needed."""
    if explicit is not None:
        return explicit
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(MIRROR, METADATA_FILE, repo_type="dataset"))


def index_patch_dirs(root: Path) -> dict[str, Path]:
    """Map every patch directory under *root* by its directory name.

    Indexed once by a single walk rather than globbed per patch: reBEN nests
    patches two levels deep under a tile directory, and 550,000 globs over a
    cold page cache is the difference between minutes and hours.
    """
    found: dict[str, Path] = {}
    for entry in root.rglob("*"):
        if entry.is_dir():
            found.setdefault(entry.name, entry)
    return found


def select_patches(
    metadata: Path,
    split: str,
    limit: int | None,
    s2_index: Mapping[str, Path],
    s1_index: Mapping[str, Path],
    seed: int = 42,
) -> list[Patch]:
    """Choose the patches to render, stratified over their label combinations.

    §4.1 asks for iterative stratification over the 19-class label vector. This
    is its cheap cousin: group by the exact label combination, then round-robin
    across groups so rare combinations survive a 20,000-patch subsample instead
    of being washed out by ``Arable land`` alone. The official split is filtered
    on, never re-derived.
    """
    import pandas as pd

    frame = pd.read_parquet(metadata)
    frame = frame[frame["split"] == split]

    groups: dict[tuple[str, ...], list[tuple[str, str, tuple[str, ...]]]] = {}
    for patch_id, labels, s1_name in zip(
        frame["patch_id"], frame["labels"], frame["s1_name"], strict=True
    ):
        key = tuple(sorted(str(label) for label in labels))
        groups.setdefault(key, []).append((str(patch_id), str(s1_name), key))

    rng = np.random.default_rng(seed)
    for entries in groups.values():
        rng.shuffle(entries)

    ordered: list[tuple[str, str, tuple[str, ...]]] = []
    keys = sorted(groups)
    depth = 0
    while len(ordered) < len(frame):
        added = False
        for key in keys:
            entries = groups[key]
            if depth < len(entries):
                ordered.append(entries[depth])
                added = True
        if not added:
            break
        depth += 1

    patches: list[Patch] = []
    for patch_id, s1_name, labels in ordered:
        s2_dir = s2_index.get(patch_id)
        if s2_dir is None:
            continue
        patches.append(
            Patch(
                patch_id=patch_id,
                s1_name=s1_name,
                split=split,
                labels=labels,
                s2_dir=s2_dir,
                s1_dir=s1_index.get(s1_name),
            )
        )
        if limit is not None and len(patches) >= limit:
            break
    return patches


class PatchError(RuntimeError):
    """A patch could not be read. Carries the patch and the reason, both."""


def _band_file(directory: Path, band: str) -> Path:
    """The GeoTIFF for one band inside a patch directory.

    Raises:
        PatchError: No file in *directory* carries that band suffix.
    """
    for candidate in sorted(directory.glob(f"*_{band}.tif*")):
        return candidate
    raise PatchError(f"{directory}: no file matching *_{band}.tif*")


def stack_bands(directory: Path, bands: Sequence[str], destination: Path) -> None:
    """Write one multi-band GeoTIFF from reBEN's per-band files.

    reBEN v2 ships each band at its **native** ground sampling distance, not on a
    common grid: a 120 x 120 patch is 120 x 120 only in the 10 m bands (B02-B04,
    B08); the 20 m bands (B05-B07, B8A, B11, B12) are 60 x 60 and the 60 m bands
    (B01, B09) are 20 x 20. So the bands are resampled onto the finest grid
    present — bilinear, because these are continuous reflectances — and the
    stack's georeferencing is taken from a band already on that grid rather than
    from whichever band happened to be read first.

    Raises:
        PatchError: A band is missing, or a band's grid does not divide into the
            finest one, which means the patch is not the layout we think it is.
    """
    import rasterio
    from rasterio.enums import Resampling as RioResampling

    paths = [_band_file(directory, band) for band in bands]

    shapes: list[tuple[int, int]] = []
    for path in paths:
        with rasterio.open(path) as handle:
            shapes.append((handle.height, handle.width))

    height = max(shape[0] for shape in shapes)
    width = max(shape[1] for shape in shapes)
    for path, shape in zip(paths, shapes, strict=True):
        if height % shape[0] or width % shape[1]:
            raise PatchError(
                f"{path.name}: {shape[1]}x{shape[0]} does not divide the patch "
                f"grid {width}x{height}"
            )

    reference = paths[shapes.index((height, width))]
    with rasterio.open(reference) as handle:
        profile = dict(handle.profile)

    arrays = []
    for path, shape in zip(paths, shapes, strict=True):
        with rasterio.open(path) as handle:
            if shape == (height, width):
                arrays.append(handle.read(1))
            else:
                arrays.append(
                    handle.read(
                        1,
                        out_shape=(height, width),
                        resampling=RioResampling.bilinear,
                    )
                )

    profile.update(count=len(arrays), driver="GTiff", height=height, width=width)
    with rasterio.open(destination, "w", **profile) as handle:
        for index, array in enumerate(arrays, start=1):
            handle.write(array, index)


def measure(optical: np.ndarray | None, radar: np.ndarray | None) -> dict[str, float | str]:
    """The scalars the runtime tools would measure for this patch.

    Computed with the tools' own functions and prefixes, then namespaced by the
    tool that owns them, so a sheet written here resolves under the same keys the
    CitationValidator sees at inference.
    """
    facts: dict[str, float | str] = {}
    if optical is not None:
        blue, green, red, nir, swir1, swir2 = optical
        for values, name in (
            (ndvi(nir, red), "ndvi"),
            (ndwi(green, nir), "ndwi"),
            (ndbi(swir1, nir), "ndbi"),
        ):
            for key, value in index_statistics(values, name).items():
                facts[f"{SPECTRAL_TOOL}.{key}"] = round(value, 6)
    if radar is not None:
        vv_db = to_db(radar[0])
        for key, value in index_statistics(vv_db, "sigma0_vv_db").items():
            facts[f"{SAR_TOOL}.{key}"] = round(value, 6)
        if len(radar) > 1:
            vh_db = to_db(radar[1])
            for key, value in index_statistics(vh_db, "sigma0_vh_db").items():
                facts[f"{SAR_TOOL}.{key}"] = round(value, 6)
            for key, value in index_statistics(
                vv_vh_ratio_db(vv_db, vh_db), "vv_vh_ratio_db"
            ).items():
                facts[f"{SAR_TOOL}.{key}"] = round(value, 6)
    return facts


def _read_stack(path: Path, indices: Sequence[int]) -> np.ndarray:
    """Read selected 1-indexed bands as float32."""
    import rasterio

    with rasterio.open(path) as handle:
        return np.stack([handle.read(index).astype(np.float32) for index in indices])


def render_patch(patch: Patch, out_root: Path, size: int) -> dict[str, Any]:
    """Render one patch's views and measure its scalars.

    Returns:
        The FactSheet record.

    Raises:
        PatchError: The patch's bands are missing or malformed. Every other
            exception propagates as itself — this pass has no business deciding
            that a rasterio error or an out-of-disk is an "unreadable patch".
    """
    aliases = load_table()
    optical_bands = aliases.sensors[S2_SENSOR]
    radar_bands = aliases.sensors[S1_SENSOR]
    target = out_root / SOURCE_NAME / patch.patch_id
    target.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)
        s2_stack = scratch_dir / "s2.tif"
        stack_bands(patch.s2_dir, S2_BAND_FILES, s2_stack)

        sources = [
            RenderSource(
                image_id=patch.patch_id,
                path=s2_stack,
                modality=Modality.OPTICAL,
                resolved_bands=dict(optical_bands),
                sensor=S2_SENSOR,
                role=ImageRole.OPTICAL,
            )
        ]
        radar: np.ndarray | None = None
        if patch.s1_dir is not None:
            s1_stack = scratch_dir / "s1.tif"
            # A missing radar half degrades the pair to SINGLE rather than
            # failing the patch — but it is reported, never swallowed.
            try:
                stack_bands(patch.s1_dir, S1_BAND_FILES, s1_stack)
            except PatchError as error:
                print(f"  no radar for {patch.patch_id}: {error}", file=sys.stderr)
            else:
                sources.append(
                    RenderSource(
                        image_id=patch.s1_name,
                        path=s1_stack,
                        modality=Modality.SAR,
                        resolved_bands=dict(radar_bands),
                        sensor=S1_SENSOR,
                        role=ImageRole.SAR,
                    )
                )
                radar = _read_stack(s1_stack, [radar_bands["vv"], radar_bands["vh"]])

        pair_type = PairType.CROSS_MODAL if len(sources) > 1 else PairType.SINGLE
        store = ArtifactStore(scratch_dir / "artifacts")
        views = render_views(
            sources=sources,
            pair_type=pair_type,
            trace_id=patch.patch_id,
            store=store,
            size=size,
        )

        optical = _read_stack(
            s2_stack,
            [optical_bands[name] for name in ("blue", "green", "red", "nir", "swir1", "swir2")],
        )

    written: dict[str, str] = {}
    for view in views:
        suffix = "png" if view.view.image_format is ImageFormat.PNG else "jpg"
        path = target / f"{view.view_id.value}.{suffix}"
        _write_image(view.rgb, path, view.view.image_format)
        written[view.view_id.value] = str(path)

    return {
        "patch_id": patch.patch_id,
        "s1_name": patch.s1_name if patch.has_radar else None,
        "split": patch.split,
        "labels": list(patch.labels),
        "pair_type": pair_type.value,
        "views": written,
        "view_labels": {v.view_id.value: v.label for v in views},
        "fact_sheet": measure(optical, radar),
    }


def _write_image(rgb: np.ndarray, path: Path, image_format: ImageFormat) -> None:
    """Encode one view, PNG for measurement views and JPEG q92 for composites (§2.1)."""
    from PIL import Image

    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    if image_format is ImageFormat.PNG:
        image.save(path, format="PNG", optimize=True)
    else:
        image.save(path, format="JPEG", quality=92, subsampling=0)


def run(
    patches: Sequence[Patch], out_root: Path, size: int, workers: int
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Render every patch, reporting progress and every failure as it goes.

    A patch that cannot be read is recorded with the reason it could not be
    read, and the first few reasons are printed in full. A pass that silently
    counts 20,000 "unreadable" patches and then writes an empty corpus is worse
    than one that crashes, because the crash is the only thing that reaches the
    person reading the log at 3 a.m.

    Returns:
        ``(records, failures)`` where each failure is ``(patch_id, reason)``.
    """
    records: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(render_patch, patch, out_root, size): patch for patch in patches
        }
        ordered = list(futures)
        for index, future in enumerate(ordered, start=1):
            patch = futures[future]
            try:
                records.append(future.result())
            except PatchError as error:
                failures.append((patch.patch_id, str(error)))
                if len(failures) <= FAILURES_SHOWN:
                    print(f"  !! {patch.patch_id}: {error}", file=sys.stderr)
            except Exception as error:  # noqa: BLE001 - reported, then re-raised
                print(
                    f"\n{patch.patch_id} failed with an error this pass does not "
                    f"know how to interpret; aborting so it is not mistaken for a "
                    f"bad patch:\n{traceback.format_exc()}",
                    file=sys.stderr,
                )
                raise SystemExit(f"{patch.patch_id}: {type(error).__name__}: {error}") from error
            if index == EARLY_ABORT_AFTER and not records:
                raise SystemExit(
                    f"the first {EARLY_ABORT_AFTER} patches all failed to render — "
                    "this is a bug in the pass or a wrong --input, not bad data. "
                    "Stopping rather than burning six hours to write an empty file."
                )
            if index % 250 == 0 or index == len(ordered):
                print(f"  rendered {index}/{len(ordered)}  ({len(failures)} unreadable)")
    return records, failures


def main(argv: Sequence[str] | None = None) -> int:
    """Select, render, and write the FactSheets. Returns a process exit code."""
    args = parse_args(argv)
    # run_overnight.sh pipes this through tee, and a pipe is block-buffered, so
    # `tail -f` on the log shows nothing for minutes at a time. The progress
    # lines are the only signal a six-hour pass gives.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(line_buffering=True)

    s2_root = args.input / "BigEarthNet-S2"
    s1_root = args.input / "BigEarthNet-S1"
    for root in (s2_root, s1_root):
        if not root.is_dir():
            print(
                f"{root} does not exist. Download and extract the patches first — "
                "see this script's docstring for the exact commands.",
                file=sys.stderr,
            )
            return 2

    print(f"indexing patch directories under {args.input} ...")
    s2_index = index_patch_dirs(s2_root)
    s1_index = index_patch_dirs(s1_root)
    print(f"  {len(s2_index)} S2 directories, {len(s1_index)} S1 directories")

    patches = select_patches(
        metadata_path(args.metadata), args.split, args.limit, s2_index, s1_index, args.seed
    )
    with_radar = sum(1 for patch in patches if patch.has_radar)
    print(f"selected {len(patches)} {args.split} patches ({with_radar} with radar)")
    if args.dry_run:
        for patch in patches[:5]:
            print(f"  {patch.patch_id}  radar={patch.has_radar}  labels={list(patch.labels)}")
        print("dry run: nothing rendered.")
        return 0

    records, failures = run(patches, args.out, args.size, args.workers)

    if not records:
        print(
            f"\nEvery one of the {len(failures)} selected patches failed to render. "
            "Nothing was written — see the reasons above.",
            file=sys.stderr,
        )
        return 1

    # One sheet file per split, then a merged index under the name the corpus
    # builders expect. Both splits used to write factsheets.jsonl with mode "w",
    # so the validation pass silently truncated the train pass's 20,000 rows.
    split_path = args.out / SOURCE_NAME / f"factsheets.{args.split}.jsonl"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    facts_path = args.out / SOURCE_NAME / FACTSHEET_FILE
    merged = 0
    with facts_path.open("w", encoding="utf-8") as handle:
        for sheet in sorted((args.out / SOURCE_NAME).glob("factsheets.*.jsonl")):
            for line in sheet.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    handle.write(line + "\n")
                    merged += 1

    views_written = sum(len(record["views"]) for record in records)
    print(f"\npatches rendered : {len(records)}")
    print(f"patches failed   : {len(failures)}")
    print(f"views written    : {views_written}")
    print(f"fact sheets      : {split_path}")
    print(f"merged index     : {facts_path}  ({merged} rows, all splits)")
    if failures:
        report = args.out / SOURCE_NAME / f"failures.{args.split}.txt"
        report.write_text(
            "".join(f"{patch_id}\t{reason}\n" for patch_id, reason in failures),
            encoding="utf-8",
        )
        print(f"failure reasons  : {report}")
    print(
        "\nRegenerate the citation corpus from these measured sheets:\n"
        f"  uv run python training/data/builders/evidence_qa.py --count 3000 "
        f"--factsheets {facts_path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
