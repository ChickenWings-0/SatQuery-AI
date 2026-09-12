"""``training/data/builders/evidence_qa.py`` — the standalone citation corpus.

The builder had no test file, and its one real defect was a split bug: the
merged ``factsheets.jsonl`` lists every train record before the first
validation record, and a sequential walk that stopped at ``--count`` never
reached one — so ``evidence_qa.val.jsonl`` was empty on every real build.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "data" / "builders"))

import evidence_qa  # noqa: E402


def _record(patch_id: str, split: str) -> dict[str, Any]:
    return {
        "patch_id": patch_id,
        "s1_name": f"S1_{patch_id}",
        "split": split,
        "labels": ["Pastures"],
        "pair_type": "CROSS_MODAL",
        "views": {
            "TC": f"views/{patch_id}/TC.jpg",
            "NDVI": f"views/{patch_id}/NDVI.png",
            "SARDB": f"views/{patch_id}/SARDB.png",
        },
        "fact_sheet": {
            "spectral_index_analyzer.ndvi_mean": 0.61,
            "spectral_index_analyzer.ndbi_mean": -0.1,
            "sar_backscatter_analyzer.sigma0_vv_db_mean": -8.0,
            "sar_backscatter_analyzer.vv_vh_ratio_db_mean": 6.0,
        },
    }


def _write_sheets(path: Path, n_train: int, n_val: int) -> Path:
    """Train records first, then validation — the order the render pass merges in."""
    records = [_record(f"t{i}", "train") for i in range(n_train)]
    records += [_record(f"v{i}", "validation") for i in range(n_val)]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_evidence_qa_val_split_is_populated_when_factsheets_carry_validation_rows(
    tmp_path: Path,
) -> None:
    """The bug: 9 train records ahead of 3 validation ones and a count of 4."""
    sheets = _write_sheets(tmp_path / "factsheets.jsonl", n_train=9, n_val=3)
    samples = evidence_qa.generate_from_factsheets(sheets, count=4, seed=0)
    assert len(samples) == 4
    assert any(sample.meta.split == "val" for sample in samples)
    assert any(sample.meta.split == "train" for sample in samples)


def test_split_quotas_are_proportional_with_a_floor_of_one() -> None:
    assert evidence_qa.split_quotas(100, 90, 10) == {"train": 90, "val": 10}
    assert evidence_qa.split_quotas(4, 9, 3) == {"train": 3, "val": 1}
    assert evidence_qa.split_quotas(10, 100, 0) == {"train": 10, "val": 0}
    # A tiny share on either side still gets a slot when the count allows.
    assert evidence_qa.split_quotas(50, 10_000, 5)["val"] == 1
    assert evidence_qa.split_quotas(50, 5, 10_000)["train"] == 1
    # Proportional when both can fill their share.
    assert evidence_qa.split_quotas(40, 2, 20) == {"train": 4, "val": 36}
    # A split that cannot fill its share, even over every pass, hands the rest over.
    passes = evidence_qa.PASSES_OVER_RECORDS
    assert evidence_qa.split_quotas(100, 20, 1) == {"train": 20 * passes, "val": 1 * passes}


def test_a_count_that_would_leave_val_empty_is_refused(tmp_path: Path) -> None:
    """``count=1`` cannot cover both splits; that is an error, not an empty file."""
    sheets = _write_sheets(tmp_path / "factsheets.jsonl", n_train=5, n_val=2)
    with pytest.raises(SystemExit, match="no validation samples"):
        evidence_qa.generate_from_factsheets(sheets, count=1, seed=0)


def test_factsheets_without_validation_records_warn_but_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheets = _write_sheets(tmp_path / "factsheets.jsonl", n_train=5, n_val=0)
    samples = evidence_qa.generate_from_factsheets(sheets, count=3, seed=0)
    assert len(samples) == 3
    assert "no validation records" in capsys.readouterr().out


def test_the_mock_path_needs_explicit_permission(tmp_path: Path) -> None:
    """A training corpus is never built from placeholder pixels by accident."""
    with pytest.raises(SystemExit, match="--allow-mock"):
        evidence_qa.main(["--count", "2", "--out-dir", str(tmp_path)])
