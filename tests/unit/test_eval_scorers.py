"""The benchmark scorers, against hand-written answers with known scores.

Each scorer is a pure function, so every case here is "this reference, this
prediction, this number". The cases are the ones that would silently move a
slide if a scorer regressed: a box in a non-canonical format, a citation with
the right number under the wrong key, a label list in a different order.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from satquery.eval import report, sampler
from satquery.eval.runner import Prediction, reference_backend, run
from satquery.eval.scorers import (
    SampleScore,
    bleu4,
    citation_metrics,
    count_of,
    exact_match,
    fact_recall,
    iou,
    labels_of,
    match_boxes,
    normalise,
    score,
    set_f1,
)
from satquery.models.prompts.box_format import NormalisedBox, serialise_answer
from satquery.schemas.enums import TaskType
from satquery.training.corpus_builder import CorpusSample, fact_sheet_from

FACTS: dict[str, float | str] = {
    "spectral_index_analyzer.ndvi_mean": 0.82,
    "spectral_index_analyzer.ndbi_mean": -0.06,
    "sar_backscatter_analyzer.sigma0_vv_db_mean": -18.98,
    "object_counter.count": 3,
}


# ------------------------------------------------------------------ text


def test_normalise_drops_markers_case_and_terminal_punctuation() -> None:
    assert normalise("The mean NDVI is 0.82 [spectral_index_analyzer.ndvi_mean].") == (
        "the mean ndvi is 0.82"
    )
    assert normalise("  Yes. ") == "yes"


def test_exact_match_ignores_citation_markers_and_case() -> None:
    assert exact_match("Top-left", "top-left.")
    assert exact_match(
        "The measured count is 3 [object_counter.count].", "The measured count is 3."
    )
    assert not exact_match("Yes.", "No.")


def test_scene_labels_are_a_set() -> None:
    reference = "The scene contains Coniferous forest, Pastures and Inland waters."
    assert labels_of(reference) == {"coniferous forest", "pastures", "inland waters"}
    reordered = "The scene contains Inland waters, Coniferous forest and Pastures."
    assert set_f1(labels_of(reference), labels_of(reordered)) == 1.0
    partial = "The scene contains Pastures."
    assert set_f1(labels_of(reference), labels_of(partial)) == pytest.approx(0.5)
    assert set_f1(frozenset(), frozenset()) == 1.0
    assert set_f1(labels_of(reference), frozenset({"urban fabric"})) == 0.0


def test_count_of_reads_the_first_integer() -> None:
    assert count_of("The measured count is 12 [object_counter.count].") == 12
    assert count_of("There are 1,204 buildings.") == 1204
    assert count_of("None visible.") is None


# ------------------------------------------------------------- grounding


def box(x1: int, y1: int, x2: int, y2: int, label: str = "airplane") -> NormalisedBox:
    return NormalisedBox(x1, y1, x2, y2, label=label)


def test_iou_of_hand_drawn_boxes() -> None:
    a, b = box(0, 0, 100, 100), box(50, 0, 150, 100)
    assert iou(a, b) == pytest.approx(50 * 100 / (2 * 100 * 100 - 50 * 100))
    assert iou(a, a) == 1.0
    assert iou(a, box(200, 200, 300, 300)) == 0.0


def test_match_boxes_recall_and_mean_iou_over_references() -> None:
    reference = [box(0, 0, 100, 100), box(500, 500, 600, 600)]
    # One perfect box, one missing: recall 0.5, mean IoU 0.5.
    recall, mean_iou, pairs = match_boxes(reference, [box(0, 0, 100, 100)])
    assert recall == 0.5
    assert mean_iou == 0.5
    assert pairs == [(0, 0, 1.0)]
    # A near miss below the threshold counts toward IoU but not recall.
    recall, mean_iou, _ = match_boxes([box(0, 0, 100, 100)], [box(60, 0, 160, 100)])
    assert recall == 0.0
    assert 0.0 < mean_iou < 0.5
    # One prediction cannot claim two references.
    recall, _, pairs = match_boxes(reference, [box(0, 0, 100, 100), box(0, 0, 100, 100)])
    assert recall == 0.5
    assert len(pairs) == 1


def test_none_answers() -> None:
    assert match_boxes([], [])[:2] == (1.0, 1.0)
    assert match_boxes([], [box(0, 0, 10, 10)])[:2] == (0.0, 0.0)


def test_grounding_score_uses_the_serving_parser() -> None:
    reference = serialise_answer([box(620, 530, 1000, 1000)])
    perfect = score(TaskType.GROUNDING, reference, reference, {})
    assert perfect.grounding_recall == 1.0
    assert perfect.grounding_iou == 1.0
    assert perfect.grounding_format_ok == 1.0

    # Qwen2.5-style JSON is parsed (tolerant parser) but is not the canonical format.
    json_form = json.dumps([{"bbox_2d": [620, 530, 1000, 1000], "label": "airplane"}])
    tolerant = score(TaskType.GROUNDING, reference, json_form, {})
    assert tolerant.grounding_recall == 1.0
    assert tolerant.grounding_format_ok == 0.0
    assert tolerant.notes

    prose = score(TaskType.GROUNDING, reference, "The airplane is in the bottom right.", {})
    assert prose.grounding_recall == 0.0
    assert prose.grounding_iou == 0.0


# ------------------------------------------------------------- citations


def test_citation_metrics_bind_marker_to_key_and_value() -> None:
    sheet = fact_sheet_from(FACTS)
    good = citation_metrics("The mean NDVI is 0.82 [spectral_index_analyzer.ndvi_mean].", sheet)
    assert (good.markers, good.bound, good.spans, good.uncited) == (1, 1, 1, 0)
    assert good.precision == 1.0
    assert good.uncited_rate == 0.0
    assert good.cited_keys == {"spectral_index_analyzer.ndvi_mean"}


def test_right_number_under_the_wrong_key_is_not_a_citation() -> None:
    sheet = fact_sheet_from(FACTS)
    wrong = citation_metrics("The mean NDVI is 0.82 [spectral_index_analyzer.ndbi_mean].", sheet)
    assert wrong.markers == 1
    assert wrong.bound == 0
    assert wrong.precision == 0.0
    assert wrong.uncited == 1


def test_unknown_key_and_bare_numbers() -> None:
    sheet = fact_sheet_from(FACTS)
    unknown = citation_metrics("Cloud cover is 12% [cloud_masker.cover_pct].", sheet)
    assert unknown.precision == 0.0
    assert unknown.uncited_rate == 1.0

    bare = citation_metrics("Roughly 40% of the scene is forest and NDVI averages 0.82.", sheet)
    assert bare.markers == 0
    assert bare.precision is None
    # 0.82 resolves by value search; 40% resolves to nothing.
    assert bare.spans == 2
    assert bare.uncited == 1
    assert bare.uncited_rate == 0.5

    silent = citation_metrics("Yes.", sheet)
    assert silent.precision is None
    assert silent.uncited_rate is None


def test_fact_recall_is_over_the_reference_facts() -> None:
    sheet = fact_sheet_from(FACTS)
    reference = citation_metrics(
        "NDBI averages -0.06 [spectral_index_analyzer.ndbi_mean], while σ⁰_VV averages "
        "-18.98 dB [sar_backscatter_analyzer.sigma0_vv_db_mean].",
        sheet,
    )
    assert len(reference.cited_keys) == 2
    half = citation_metrics("NDBI averages -0.06 [spectral_index_analyzer.ndbi_mean].", sheet)
    assert fact_recall(reference, half) == 0.5
    assert fact_recall(citation_metrics("Yes.", sheet), half) is None


# -------------------------------------------------------------- captions


def test_bleu4_is_one_for_identity_and_zero_for_disjoint() -> None:
    caption = "A dense urban area with narrow roads and a small ground track field."
    assert bleu4(caption, caption) == pytest.approx(1.0)
    assert bleu4(caption, "Calm open water, nothing built, nothing moving") == 0.0
    assert 0.0 < bleu4(caption, "A dense urban area with narrow roads.") < 1.0


# ------------------------------------------------------------ per-sample


def test_score_dispatches_by_task() -> None:
    sheet = dict(FACTS)
    vqa = score(TaskType.VQA, "Yes.", "yes", sheet)
    assert vqa.exact_match == 1.0
    assert vqa.grounding_recall is None

    count = score(
        TaskType.COUNT,
        "The measured count is 3 [object_counter.count].",
        "The measured count is 5 [object_counter.count].",
        sheet,
    )
    assert count.exact_match == 0.0
    assert count.count_abs_error == 2.0
    # 5 is not what the counter measured: the marker fails the value check.
    assert count.citation_precision == 0.0

    classify = score(
        TaskType.SCENE_CLASSIFY,
        "The scene contains Pastures and Inland waters.",
        "The scene contains Inland waters and Pastures.",
        sheet,
    )
    assert classify.exact_match == 0.0
    assert classify.set_f1 == 1.0

    caption = score(TaskType.CAPTION, "A river.", "A river.", sheet)
    assert caption.exact_match is None
    assert caption.bleu4 == pytest.approx(1.0)

    compare = score(
        TaskType.CROSS_MODAL_COMPARE,
        "Optically the mean NDVI is 0.82 [spectral_index_analyzer.ndvi_mean]; on the radar "
        "side σ⁰_VV averages -18.98 dB [sar_backscatter_analyzer.sigma0_vv_db_mean].",
        "The mean NDVI is 0.82 [spectral_index_analyzer.ndvi_mean].",
        sheet,
    )
    assert compare.exact_match is None, "a paragraph is scored on overlap, not exact match"
    assert compare.fact_recall == 0.5
    assert compare.citation_precision == 1.0
    assert compare.bleu4 is not None


# --------------------------------------------------------------- sampler


def _sample(index: int, source: str, task: str, views: int = 1) -> CorpusSample:
    return CorpusSample.model_validate(
        {
            "id": f"{source}:{index}",
            "source": source,
            "task": task,
            "pair_type": "SINGLE",
            "views": [
                {
                    "slot": slot + 1,
                    "view_id": "TC",
                    "path": f"views/{index}/{slot}.png",
                    "label": "Image 1",
                }
                for slot in range(views)
            ],
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": "Yes."},
            ],
            "fact_sheet": {},
            "meta": {},
        }
    )


def _corpus() -> list[CorpusSample]:
    rows: list[CorpusSample] = []
    index = 0
    for task, count in (("VQA", 90), ("GROUNDING", 8), ("CAPTION", 2)):
        for _ in range(count):
            rows.append(_sample(index, "vrsbench", task))
            index += 1
    for _ in range(30):
        rows.append(_sample(index, "cdvqa", "CHANGE_VQA"))
        index += 1
    random.Random(1).shuffle(rows)
    return rows


def test_stratified_sample_balances_tasks_and_is_deterministic() -> None:
    corpus = _corpus()
    chosen = sampler.stratified_sample(corpus, per_source=20, seed=42)
    by_source: dict[str, list[CorpusSample]] = {}
    for row in chosen:
        by_source.setdefault(row.source.value, []).append(row)
    assert len(by_source["vrsbench"]) == 20
    assert len(by_source["cdvqa"]) == 20
    tasks = [row.task.value for row in by_source["vrsbench"]]
    # Round-robin: every task is represented before any task is repeated
    # beyond its share, and the small groups are exhausted, not ignored.
    assert tasks.count("CAPTION") == 2
    assert tasks.count("GROUNDING") == 8
    assert tasks.count("VQA") == 10
    again = sampler.stratified_sample(corpus, per_source=20, seed=42)
    assert [row.id for row in again] == [row.id for row in chosen]
    other = sampler.stratified_sample(corpus, per_source=20, seed=7)
    assert [row.id for row in other] != [row.id for row in chosen]


def test_pinned_ids_round_trip_and_refuse_missing(tmp_path: Path) -> None:
    corpus = _corpus()
    chosen = sampler.stratified_sample(corpus, per_source=5, seed=0)
    sampler.write_ids(chosen, tmp_path / "ids.json")
    ids = sampler.read_ids(tmp_path / "ids.json")
    assert [row.id for row in sampler.select_ids(corpus, ids)] == ids
    with pytest.raises(KeyError):
        sampler.select_ids(corpus, [*ids, "vrsbench:does-not-exist"])


# ---------------------------------------------------------------- runner


def test_reference_backend_runs_scores_perfectly_and_resumes(tmp_path: Path) -> None:
    from PIL import Image

    corpus = _corpus()[:6]
    for row in corpus:
        for view in row.views:
            path = tmp_path / view.path
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8)).save(path)

    out = tmp_path / "predictions.jsonl"
    first = list(run(corpus[:3], reference_backend(corpus), out, root=tmp_path))
    assert [row.prediction for row in first] == [row.assistant for row in corpus[:3]]
    assert all(row.error is None for row in first)

    seen: list[str] = []

    def progress(done: int, total: int, prediction: Prediction) -> None:
        seen.append(prediction.id)

    resumed = list(
        run(corpus, reference_backend(corpus), out, root=tmp_path, resume=True, progress=progress)
    )
    assert len(resumed) == 6
    assert seen == [row.id for row in corpus]
    assert len(out.read_text().splitlines()) == 6, "resume appended only the missing rows"


def test_a_missing_view_is_an_error_row_not_a_crash(tmp_path: Path) -> None:
    corpus = _corpus()[:1]
    rows = list(run(corpus, reference_backend(corpus), tmp_path / "p.jsonl", root=tmp_path))
    assert rows[0].error is not None and "FileNotFoundError" in rows[0].error
    assert rows[0].prediction == ""


# ---------------------------------------------------------------- report


def test_report_aggregates_skip_none_and_write_three_files(tmp_path: Path) -> None:
    def scored(source: str, task: str, **fields: float | None) -> report.ScoredSample:
        return report.ScoredSample(
            prediction=Prediction(
                id=f"{source}:{task}",
                source=source,
                task=task,
                pair_type="SINGLE",
                reference="",
                prediction="",
                latency_ms=100,
            ),
            score=SampleScore(**fields),  # type: ignore[arg-type]
        )

    rows = [
        scored("vrsbench", "VQA", exact_match=1.0, citation_precision=1.0),
        scored("vrsbench", "VQA", exact_match=0.0, citation_precision=None),
        scored("vrsbench", "GROUNDING", grounding_recall=0.5, grounding_iou=0.25),
        scored("cdvqa", "CHANGE_VQA", exact_match=1.0),
    ]
    groups = report.write_report(rows, tmp_path, {"name": "t", "seed": 1, "git_sha": "abc"})
    vrs = groups["source"]["vrsbench"]
    assert vrs.n == 3
    assert vrs.means["exact_match"] == 0.5
    assert vrs.counts["exact_match"] == 2
    assert vrs.means["citation_precision"] == 1.0, "None is skipped, not counted as zero"
    assert vrs.counts["citation_precision"] == 1
    assert groups["all"]["all"].means["grounding_recall"] == 0.5

    markdown = (tmp_path / "results.md").read_text()
    assert "| VRSBench | 3 | 50.0 % | 50.0 % | 25.0 % | 100.0 % | — |" in markdown
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["aggregate"]["source"]["cdvqa"]["n"] == 1
    assert len(payload["samples"]) == 4
    csv_lines = (tmp_path / "results.csv").read_text().splitlines()
    assert csv_lines[0].startswith("source,n,exact_match")
    assert len(csv_lines) == 4  # header + 2 sources + all
