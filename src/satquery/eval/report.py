"""Aggregate the per-sample scores into the tables the slides are built from.

Three files, one directory, always the same names so a run is addressable:

* ``results.md`` — the per-source table (the one that goes on the slide), a
  per-task table, and the run's provenance: adapter, backend, git SHA, seed,
  sample count, wall time. Paste-ready.
* ``results.json`` — every number in ``results.md`` plus the per-sample
  scores, for the baseline comparison and for anyone who wants to re-cut.
* ``results.csv`` — the per-source rows, for a spreadsheet.

And a ``confusion/`` folder with the worst cases, because the honest-failure
slide is worth more than another decimal on the good one: the 20 lowest
grounding IoUs rendered as ``view + reference box + predicted box`` PNGs,
and the 20 wrong VQA answers as a Markdown list of question, reference,
prediction.

Means are over the samples for which a metric applies (``None`` is skipped,
not counted as zero), and every cell carries its ``n`` in the JSON so a
number backed by three samples is never mistaken for one backed by three
hundred.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from satquery.eval.runner import Prediction
from satquery.eval.scorers import SampleScore, boxes_of
from satquery.training.corpus_builder import CorpusSample

__all__ = [
    "Aggregate",
    "ScoredSample",
    "aggregate",
    "render_markdown",
    "write_report",
    "write_worst_cases",
]

SOURCE_TITLES: Final[dict[str, str]] = {
    "bigearthnet_v2": "BigEarthNet-v2",
    "vrsbench": "VRSBench",
    "rsvqa_hr": "RSVQA-HR",
    "cdvqa": "CDVQA",
    "evidence_qa": "Evidence QA",
}

METRICS: Final[tuple[tuple[str, str], ...]] = (
    ("exact_match", "Accuracy"),
    ("set_f1", "Label F1"),
    ("grounding_recall", "Grounding R@0.5"),
    ("grounding_iou", "Mean IoU"),
    ("grounding_format_ok", "Box format OK"),
    ("citation_precision", "Citation precision"),
    ("uncited_rate", "Uncited-number rate"),
    ("fact_recall", "Fact recall"),
    ("bleu4", "BLEU-4"),
    ("rouge_l", "ROUGE-L"),
)
"""Column order and slide labels. ``count_abs_error`` is reported in the JSON
only — an MAE next to nine rates reads as a rate."""

SLIDE_COLUMNS: Final[tuple[str, ...]] = (
    "exact_match",
    "grounding_recall",
    "grounding_iou",
    "citation_precision",
    "uncited_rate",
)
"""The per-source table on the slide is these five; the rest are in the
per-task table below it."""

WORST_N: Final[int] = 20


@dataclass
class ScoredSample:
    """One prediction with its score, the unit the report aggregates."""

    prediction: Prediction
    score: SampleScore


@dataclass
class Aggregate:
    """Mean and count per metric for one group of samples."""

    n: int
    means: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    latency_ms_mean: float = 0.0
    errors: int = 0
    truncated: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Plain mapping for JSON."""
        return {
            "n": self.n,
            "means": self.means,
            "counts": self.counts,
            "latency_ms_mean": self.latency_ms_mean,
            "errors": self.errors,
            "truncated": self.truncated,
        }


def _aggregate_group(rows: Sequence[ScoredSample]) -> Aggregate:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for key, value in row.score.as_dict().items():
            if key in {"markers", "spans", "notes"} or value is None:
                continue
            sums[key] += float(value)
            counts[key] += 1
    means = {key: sums[key] / counts[key] for key in counts}
    return Aggregate(
        n=len(rows),
        means=means,
        counts=dict(counts),
        latency_ms_mean=(
            sum(row.prediction.latency_ms for row in rows) / len(rows) if rows else 0.0
        ),
        errors=sum(1 for row in rows if row.prediction.error),
        truncated=sum(1 for row in rows if row.prediction.truncated),
    )


def aggregate(rows: Sequence[ScoredSample]) -> dict[str, dict[str, Aggregate]]:
    """Group by source, by task, and overall.

    Returns ``{"source": {...}, "task": {...}, "all": {"all": Aggregate}}``.
    """
    by_source: dict[str, list[ScoredSample]] = defaultdict(list)
    by_task: dict[str, list[ScoredSample]] = defaultdict(list)
    for row in rows:
        by_source[row.prediction.source].append(row)
        by_task[row.prediction.task].append(row)
    return {
        "source": {name: _aggregate_group(group) for name, group in sorted(by_source.items())},
        "task": {name: _aggregate_group(group) for name, group in sorted(by_task.items())},
        "all": {"all": _aggregate_group(rows)},
    }


# ------------------------------------------------------------- rendering


def _cell(group: Aggregate, key: str) -> str:
    value = group.means.get(key)
    if value is None:
        return "—"
    if key == "count_abs_error":
        return f"{value:.2f}"
    return f"{100 * value:.1f} %"


def _table(
    groups: Mapping[str, Aggregate],
    columns: Sequence[str],
    label: str,
    titles: Mapping[str, str] | None = None,
) -> list[str]:
    header = [label, "n", *[name for key, name in METRICS if key in columns]]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, group in groups.items():
        cells = [(titles or {}).get(name, name), str(group.n)]
        cells.extend(_cell(group, key) for key, _ in METRICS if key in columns)
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_markdown(
    groups: Mapping[str, Mapping[str, Aggregate]], provenance: Mapping[str, Any]
) -> str:
    """The paste-ready report."""
    overall = groups["all"]["all"]
    source_rows = {**groups["source"], "all": overall}
    titles = {**SOURCE_TITLES, "all": "**All**"}
    lines = [
        f"# SatQuery AI — validation benchmark · {provenance.get('name', '')}",
        "",
        f"*{provenance.get('finished_at', '')} · {overall.n} held-out samples · "
        f"backend `{provenance.get('backend', '?')}` · "
        f"adapter `{provenance.get('adapter') or 'none (base model)'}` · "
        f"git `{provenance.get('git_sha', '?')[:12]}` · seed {provenance.get('seed', '?')}*",
        "",
        "## Per source (slide table)",
        "",
        *_table(source_rows, SLIDE_COLUMNS, "Source", titles),
        "",
        "Accuracy is exact match on closed answers (VQA, change VQA, count, scene labels); "
        "grounding recall is at IoU ≥ 0.5 with mean IoU over reference boxes; citation "
        "precision is the share of `[tool.scalar]` markers the validator bound to that "
        "measurement; the uncited-number rate is the share of numbers in the answer that "
        "resolved to nothing. `—` means the metric does not apply to that group.",
        "",
        "## Per task",
        "",
        *_table(groups["task"], [key for key, _ in METRICS], "Task"),
        "",
        "## Run",
        "",
        f"- mean latency {overall.latency_ms_mean:.0f} ms/sample · "
        f"{overall.errors} generation errors · {overall.truncated} truncated decodes",
        f"- corpus `{provenance.get('val', '')}` · "
        f"{provenance.get('per_source', '?')} per source · "
        f"wall {provenance.get('wall_s', 0):.0f} s",
        f"- predictions `{provenance.get('predictions', 'predictions.jsonl')}` · "
        f"sample ids `{provenance.get('sample_ids', 'sample_ids.json')}`",
    ]
    if provenance.get("baseline_of"):
        lines.append(f"- baseline for `{provenance['baseline_of']}` (same sample ids)")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------- writing


def write_report(
    rows: Sequence[ScoredSample],
    out: Path,
    provenance: Mapping[str, Any],
) -> dict[str, dict[str, Aggregate]]:
    """Write ``results.md``, ``results.json`` and ``results.csv`` under *out*."""
    out.mkdir(parents=True, exist_ok=True)
    groups = aggregate(rows)
    provenance = {"finished_at": datetime.now(UTC).isoformat(timespec="seconds"), **provenance}

    (out / "results.md").write_text(render_markdown(groups, provenance), encoding="utf-8")

    payload = {
        "provenance": dict(provenance),
        "aggregate": {
            level: {name: group.as_dict() for name, group in named.items()}
            for level, named in groups.items()
        },
        "samples": [
            {
                "id": row.prediction.id,
                "source": row.prediction.source,
                "task": row.prediction.task,
                "score": row.score.as_dict(),
                "latency_ms": row.prediction.latency_ms,
                "error": row.prediction.error,
            }
            for row in rows
        ],
    }
    (out / "results.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    with (out / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "n", *[key for key, _ in METRICS], "latency_ms_mean"])
        for name, group in {**groups["source"], "all": groups["all"]["all"]}.items():
            writer.writerow(
                [
                    name,
                    group.n,
                    *[
                        f"{group.means[key]:.4f}" if key in group.means else ""
                        for key, _ in METRICS
                    ],
                    f"{group.latency_ms_mean:.0f}",
                ]
            )
    return groups


def write_worst_cases(
    rows: Iterable[ScoredSample],
    samples: Mapping[str, CorpusSample],
    out: Path,
    root: Path,
    limit: int = WORST_N,
) -> dict[str, int]:
    """Render the honest-failure evidence under ``out/confusion/``.

    Grounding: the *limit* lowest mean-IoU samples as PNG strips — the first
    view with the reference box in green and every predicted box in red.
    Closed answers: the first *limit* exact-match failures as a Markdown list.

    Returns how many of each were written.
    """
    from PIL import Image, ImageDraw

    folder = out / "confusion"
    folder.mkdir(parents=True, exist_ok=True)
    scored = list(rows)

    grounding = sorted(
        (
            row
            for row in scored
            if row.score.grounding_iou is not None and row.score.grounding_iou < 0.999
        ),
        key=lambda row: row.score.grounding_iou or 0.0,
    )[:limit]
    written = 0
    for row in grounding:
        sample = samples.get(row.prediction.id)
        if sample is None or not sample.views:
            continue
        view = sorted(sample.views, key=lambda v: v.slot)[0]
        path = root / view.path
        if not path.exists():
            continue
        with Image.open(path) as image:
            canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for box in boxes_of(row.prediction.reference):
            draw.rectangle(box.to_pixels(canvas.width, canvas.height), outline=(0, 220, 0), width=3)
        for box in boxes_of(row.prediction.prediction):
            draw.rectangle(
                box.to_pixels(canvas.width, canvas.height), outline=(230, 40, 40), width=3
            )
        draw.text(
            (6, 6),
            f"IoU {row.score.grounding_iou:.2f} · {row.prediction.id[:48]}",
            fill=(255, 255, 0),
        )
        safe = row.prediction.id.replace("/", "_").replace(":", "_")[:120]
        canvas.save(folder / f"grounding_{written:02d}_{safe}.png")
        written += 1

    wrong = [
        row for row in scored if row.score.exact_match is not None and row.score.exact_match < 1.0
    ][:limit]
    lines = ["# Wrong closed answers", ""]
    for row in wrong:
        sample = samples.get(row.prediction.id)
        question = sample.user if sample is not None else ""
        lines.extend(
            [
                f"## {row.prediction.id}",
                "",
                f"- **task** {row.prediction.task} · **source** {row.prediction.source}",
                f"- **question** {question.strip()[:300]}",
                f"- **reference** {row.prediction.reference.strip()[:300]}",
                f"- **prediction** {row.prediction.prediction.strip()[:300] or '*(empty)*'}",
                "",
            ]
        )
    (folder / "wrong_answers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"grounding_png": written, "wrong_answers": len(wrong)}
