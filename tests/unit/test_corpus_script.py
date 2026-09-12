"""``scripts/build_corpus.py`` — the seam between the sources and the builder.

The library (``corpus_builder``) has always been well tested. What was not is
the script-level plumbing that decides *whether the Deduplicator sees a hash at
all*: every BigEarthNet row used to hash to nothing, so an 18 k-line corpus
claimed strict image-level dedup and had checked no image. These tests pin the
plumbing at the level that broke.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import build_corpus  # noqa: E402 - tests/conftest.py puts scripts/ on sys.path
from satquery.training import corpus_builder as cb


def _write_png(path: Path, seed: int = 0, size: int = 16) -> Path:
    """A small, deterministic, non-blank image."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)
    return path


def _render_ben_views(views_root: Path, patch_id: str, seed: int = 0) -> dict[str, Path]:
    """Write every view the BigEarthNet render pass would, with real pixels."""
    written: dict[str, Path] = {}
    for view_id in build_corpus.VIEW_IDS_BY_SOURCE[cb.CorpusSource.BIGEARTHNET_V2]:
        suffix = build_corpus.view_suffix(view_id)
        written[view_id] = _write_png(
            views_root / "bigearthnet_v2" / patch_id / f"{view_id}.{suffix}", seed=seed
        )
    return written


def _unbound(row: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """The row as the loader yields it: before the script binds any view path."""
    return {key: value for key, value in dict(row, **changes).items() if key != "view_paths"}


def _args(views_root: Path, **overrides: Any) -> argparse.Namespace:
    """The namespace ``_iter_source_samples`` reads, with the CLI's defaults."""
    values: dict[str, Any] = {
        "sources": ["bigearthnet_v2"],
        "views_root": views_root,
        "limit": None,
        "download": False,
        "require_views": False,
        "on_missing_views": None,
        "seed": 0,
        "factsheets": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


# ------------------------------------------------------------ hashes_for (1.1)


def test_bigearthnet_rows_are_hashed_from_their_true_colour_view(
    tmp_path: Path, ben_row: dict[str, Any]
) -> None:
    """The seam that broke: BEN sets no image_path, so the TC view must stand in."""
    views = _render_ben_views(tmp_path, ben_row["patch_id"])
    row = build_corpus.bind_view_paths(
        cb.CorpusSource.BIGEARTHNET_V2, _unbound(ben_row), ben_row["patch_id"], tmp_path
    )
    assert "image_path" not in row, "BEN carries its own band paths; the contract is unchanged"

    hashes = build_corpus.hashes_for(cb.CorpusSource.BIGEARTHNET_V2, row)

    assert hashes.state is build_corpus.HashState.HASHED
    assert hashes.path == str(views["TC"])
    assert hashes.sha256 == cb.sha256_of(views["TC"])
    assert hashes.phash is not None and len(hashes.phash) == 16
    assert set(hashes.as_row_fields()) == {"image_sha256", "phash"}


def test_single_image_and_bitemporal_sources_hash_their_own_paths(tmp_path: Path) -> None:
    """The two paths that already worked keep working, and are pinned."""
    single = _write_png(tmp_path / "vrs" / "TC.png", seed=1)
    pre = _write_png(tmp_path / "cd" / "TC_pre.jpg", seed=2)
    _write_png(tmp_path / "cd" / "TC_post.jpg", seed=3)

    vrs = build_corpus.hashes_for(cb.CorpusSource.VRSBENCH, {"image_path": str(single)})
    assert vrs.state is build_corpus.HashState.HASHED
    assert vrs.sha256 == cb.sha256_of(single)

    cdvqa = build_corpus.hashes_for(
        cb.CorpusSource.CDVQA, {"pre_path": str(pre), "post_path": str(tmp_path / "cd/TC_post.jpg")}
    )
    assert cdvqa.state is build_corpus.HashState.HASHED
    assert cdvqa.sha256 == cb.sha256_of(pre), "the pre image stands for the pair"


def test_the_dedup_image_is_the_view_the_corpus_line_points_at(tmp_path: Path) -> None:
    """Hash after binding: the rendered view, never the raw file the source named."""
    raw = _write_png(tmp_path / "raw" / "x.png", seed=4)
    rendered = _write_png(tmp_path / "views" / "cdvqa" / "p1" / "TC_pre.jpg", seed=5)
    _write_png(tmp_path / "views" / "cdvqa" / "p1" / "TC_post.jpg", seed=6)
    row = build_corpus.bind_view_paths(
        cb.CorpusSource.CDVQA,
        {"pre_path": str(raw), "post_path": str(raw), "pair_id": "p1"},
        "p1",
        tmp_path / "views",
    )
    hashes = build_corpus.hashes_for(cb.CorpusSource.CDVQA, row)
    assert hashes.path == str(rendered)
    assert hashes.sha256 == cb.sha256_of(rendered) != cb.sha256_of(raw)


# ---------------------------------------------------- end-to-end binding (1.2)


def _stream_ben(
    monkeypatch: pytest.MonkeyPatch, rows: list[tuple[str, dict[str, Any]]]
) -> None:
    """Replace the Hub reader with an in-memory split-tagged stream."""

    def fake_load(source: cb.CorpusSource, splits: Any, limit: Any, download: Any) -> Any:
        yield from rows

    monkeypatch.setattr(build_corpus, "load_rows_by_split", fake_load)


def test_iter_source_samples_binds_a_hash_to_every_bigearthnet_sample(
    tmp_path: Path, ben_row: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Built through the script's own stream, every sample carries its hashes."""
    _render_ben_views(tmp_path, ben_row["patch_id"])
    _stream_ben(monkeypatch, [("train", _unbound(ben_row))])

    samples = list(
        build_corpus.iter_source_samples(
            cb.CorpusSource.BIGEARTHNET_V2, _args(tmp_path, require_views=True), random.Random(0)
        )
    )

    assert samples, "the row must convert"
    for _, sample in samples:
        assert sample.meta.image_sha256 is not None, sample.id
        assert sample.meta.phash is not None, sample.id


def test_the_dedup_now_fires_on_bigearthnet(
    tmp_path: Path, ben_row: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two patches with identical pixels: the second is an exact duplicate, dropped.

    Before the fix, both entered the corpus — the index had nothing to compare.
    """
    _render_ben_views(tmp_path, "patch-a", seed=7)
    _render_ben_views(tmp_path, "patch-b", seed=7)  # same seed, same bytes
    rows = [
        ("train", _unbound(ben_row, patch_id="patch-a")),
        ("train", _unbound(ben_row, patch_id="patch-b")),
    ]
    _stream_ben(monkeypatch, rows)

    stream = build_corpus.iter_source_samples(
        cb.CorpusSource.BIGEARTHNET_V2, _args(tmp_path), random.Random(0)
    )
    report = cb.build_corpus_streaming(
        stream,
        tmp_path / "out",
        composition={cb.CorpusSource.BIGEARTHNET_V2: 100},
        dedup=cb.Deduplicator(),
    )

    assert report.dropped_exact > 0
    row = next(r for r in report.sources if r.source is cb.CorpusSource.BIGEARTHNET_V2)
    assert row.kept < row.built


# ------------------------------------------------- loud unhashed rows (1.3)


def test_a_corrupt_view_is_a_build_error_not_a_traceback(tmp_path: Path) -> None:
    """A truncated JPEG names itself; it never enters a corpus silently."""
    broken = tmp_path / "views" / "vrsbench" / "1" / "TC.png"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"not an image at all")

    with pytest.raises(cb.CorpusError, match=str(broken)):
        build_corpus.hashes_for(cb.CorpusSource.VRSBENCH, {"image_path": str(broken)})


def test_missing_views_are_counted_and_reported_under_keep(
    tmp_path: Path, ben_row: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``keep`` still writes the line, but the build now knows it was unhashed."""
    _stream_ben(monkeypatch, [("train", _unbound(ben_row))])
    tally = build_corpus.HashTally()

    samples = list(
        build_corpus.iter_source_samples(
            cb.CorpusSource.BIGEARTHNET_V2, _args(tmp_path), random.Random(0), tally
        )
    )

    assert samples
    assert tally.counts[build_corpus.HashState.MISSING] == 1
    assert tally.unhashed == 1
    assert all(sample.meta.image_sha256 is None for _, sample in samples)


def test_a_missing_dedup_image_fails_a_require_views_build(
    tmp_path: Path, ben_row: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under ``fail``, the hash check runs before conversion and refuses first."""
    _stream_ben(monkeypatch, [("train", _unbound(ben_row))])
    with pytest.raises(cb.CorpusError, match="dedup image missing"):
        list(
            build_corpus.iter_source_samples(
                cb.CorpusSource.BIGEARTHNET_V2,
                _args(tmp_path, require_views=True),
                random.Random(0),
            )
        )


# ------------------------------------------------- factsheet binding (1.4)


def _factsheet_record(
    patch_id: str, views: dict[str, Path], split: str = "train"
) -> dict[str, Any]:
    return {
        "patch_id": patch_id,
        "s1_name": f"S1_{patch_id}",
        "split": split,
        "labels": ["Pastures"],
        "pair_type": "CROSS_MODAL",
        "views": {view_id: str(path) for view_id, path in views.items()},
        "fact_sheet": {
            "spectral_index_analyzer.ndvi_mean": 0.61,
            "spectral_index_analyzer.ndbi_mean": -0.1,
            "sar_backscatter_analyzer.sigma0_vv_db_mean": -8.0,
            "sar_backscatter_analyzer.vv_vh_ratio_db_mean": 6.0,
        },
    }


def test_factsheet_index_attaches_measurements_to_bigearthnet_rows(tmp_path: Path) -> None:
    """A patch in the index gets the render pass's views and its FactSheet."""
    views = _render_ben_views(tmp_path, "p1")
    subset = {k: v for k, v in views.items() if k in ("TC", "NDVI")}  # no S1 this patch
    sheets = tmp_path / "factsheets.jsonl"
    sheets.write_text(json.dumps(_factsheet_record("p1", subset)) + "\n", encoding="utf-8")

    facts = build_corpus.load_factsheet_index(sheets)
    row = build_corpus.bind_view_paths(
        cb.CorpusSource.BIGEARTHNET_V2, {"patch_id": "p1", "split": "train"}, "p1", tmp_path, facts
    )

    assert row["fact_sheet"]["spectral_index_analyzer.ndvi_mean"] == 0.61
    assert set(row["view_paths"]) == {"TC", "NDVI"}, "only the views actually rendered"
    assert row["labels"] == ["Pastures"]


def test_factsheet_index_refuses_a_record_without_a_patch_id(tmp_path: Path) -> None:
    sheets = tmp_path / "factsheets.jsonl"
    sheets.write_text('{"split": "train"}\n', encoding="utf-8")
    with pytest.raises(cb.CorpusError, match="patch_id"):
        build_corpus.load_factsheet_index(sheets)


def test_evidence_qa_is_generated_in_band_when_factsheets_are_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4.6 supervision comes out of the same stream, hashed like its parent."""
    views = _render_ben_views(tmp_path, "p1")
    sheets = tmp_path / "factsheets.jsonl"
    sheets.write_text(json.dumps(_factsheet_record("p1", views)) + "\n", encoding="utf-8")
    facts = build_corpus.load_factsheet_index(sheets)
    # The text export's row shape: annotations, no labels, no fact sheet.
    _stream_ben(
        monkeypatch,
        [
            (
                "train",
                {
                    "ID": "1",
                    "patch_id": "p1",
                    "input": "What is in this scene?",
                    "output": "Pastures.",
                    "type": "caption",
                },
            )
        ],
    )
    args = _args(tmp_path, sources=["bigearthnet_v2", "evidence_qa"])

    samples = list(
        build_corpus.iter_source_samples(
            cb.CorpusSource.BIGEARTHNET_V2, args, random.Random(0), None, facts
        )
    )

    evidence = [s for source, s in samples if source is cb.CorpusSource.EVIDENCE_QA]
    assert evidence, "the bound FactSheet must produce evidence_qa in-band"
    assert all(s.meta.image_sha256 is not None for s in evidence)
    parent = next(s for source, s in samples if source is cb.CorpusSource.BIGEARTHNET_V2)
    assert cb.image_key_of(evidence[0]) == cb.image_key_of(parent), (
        "evidence samples share their parent's image key, so the index treats "
        "them as annotations of one admitted image, not as duplicates"
    )


def test_resolve_factsheets_falls_back_to_the_conventional_path(tmp_path: Path) -> None:
    conventional = tmp_path / "bigearthnet_v2" / "factsheets.jsonl"
    assert build_corpus.resolve_factsheets(_args(tmp_path)) is None
    conventional.parent.mkdir(parents=True)
    conventional.write_text("", encoding="utf-8")
    assert build_corpus.resolve_factsheets(_args(tmp_path)) == conventional
    explicit = tmp_path / "elsewhere.jsonl"
    assert build_corpus.resolve_factsheets(_args(tmp_path, factsheets=explicit)) == explicit


# ------------------------------------------------------- composition (1.6)


def test_composition_overrides_apply_only_to_requested_sources() -> None:
    requested = {cb.CorpusSource.BIGEARTHNET_V2, cb.CorpusSource.EVIDENCE_QA}
    targets = build_corpus.composition_for(requested, ["bigearthnet_v2=4000"])
    assert targets[cb.CorpusSource.BIGEARTHNET_V2] == 4000
    assert targets[cb.CorpusSource.EVIDENCE_QA] == cb.COMPOSITION[cb.CorpusSource.EVIDENCE_QA]
    with pytest.raises(cb.CorpusError, match="not in --sources"):
        build_corpus.composition_for(requested, ["vrsbench=10"])
    with pytest.raises(cb.CorpusError, match="SOURCE=N"):
        build_corpus.composition_for(requested, ["bigearthnet_v2"])


def test_the_manifest_records_every_build(tmp_path: Path) -> None:
    args = _args(tmp_path, out=tmp_path / "corpus", sources=["vrsbench"])
    report = cb.CorpusReport(sources=[], train=3, val=1, dropped_exact=1, dropped_near=0, seed=0)
    tally = build_corpus.HashTally()
    path = build_corpus.append_manifest(args, report, tally)
    build_corpus.append_manifest(args, report, tally)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# Corpus manifest")
    assert sum(line.startswith("| 20") for line in lines) == 2, "one row per build, appended"
    assert "vrsbench" in lines[-1]
