"""Phase 7: the instruction corpus and the QLoRA training engine.

Nothing here downloads a dataset or a checkpoint. Dataset rows are mock mappings
of exactly the shape the real loaders yield, and the model is a *list of module
names* recorded from Qwen3-VL's tree — which is all the target-module resolver
ever looks at. That is not a compromise: the failures this phase is exposed to
are not failures of Qwen or of ``datasets``, they are ours, and all four of them
are visible without a GPU —

* a corpus label that does not match the label the server writes (§2.5),
* a box format that the grounding parser cannot read back (§4.5),
* an answer citing a number the FactSheet does not contain (§4.6),
* a LoRA that attached to the language model and missed the vision tower (§6.1).
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from satquery.evidence.citation_validator import validate
from satquery.models.prompts import box_format
from satquery.models.prompts.builder import (
    ViewInput,
    build_prompt,
    render_fact_sheet,
    strip_citation_markers,
)
from satquery.models.prompts.templates import get_template
from satquery.render.view_labels import label_for_view
from satquery.render.views import ViewId
from satquery.schemas.enums import Modality, PairType, TaskType
from satquery.training import corpus_builder as cb
from satquery.training import local_sources
from satquery.training.vlm import qlora

# ------------------------------------------------------------------- fixtures

BEN_FACTS: dict[str, float | str] = {
    "spectral_index_analyzer.ndvi_mean": 0.62,
    "spectral_index_analyzer.ndbi_mean": -0.08,
    "sar_backscatter_analyzer.sigma0_vv_db_mean": -8.4,
    "sar_backscatter_analyzer.vv_vh_ratio_db_mean": 6.1,
}

BEN_VIEW_IDS = ("TC", "FCIR", "SWIR", "NDVI", "NDBI", "SARFC", "SARDB")


@pytest.fixture
def ben_row() -> dict[str, Any]:
    """One reBEN metadata row, as the BigEarthNet-v2 loader yields it."""
    return {
        "patch_id": "S2A_MSIL2A_20180413T95029_37_58",
        "labels": ["Broad-leaved forest", "Pastures"],
        "split": "train",
        "season": "spring",
        "view_paths": {
            view_id: f"views/ben2/S2A_37_58/{view_id}.png" for view_id in BEN_VIEW_IDS
        },
        "fact_sheet": dict(BEN_FACTS),
    }


@pytest.fixture
def optical_views() -> list[cb.SourceView]:
    """A two-view optical stack, labelled the way the renderer labels one."""
    return [
        cb.SourceView(ViewId.TC, "views/x/TC.jpg", Modality.OPTICAL, "Sentinel-2 L2A"),
        cb.SourceView(ViewId.NDVI, "views/x/NDVI.png", Modality.OPTICAL, "Sentinel-2 L2A"),
    ]


def qwen_module_names(vision_blocks: int = 27, llm_layers: int = 36) -> list[str]:
    """The module paths a Qwen3-VL checkpoint exposes, to the depth we target."""
    names = ["model", "model.visual", "model.language_model", "lm_head", "model.embed_tokens"]
    for block in range(vision_blocks):
        prefix = f"model.visual.blocks.{block}"
        names += [
            prefix,
            f"{prefix}.attn",
            f"{prefix}.attn.qkv",
            f"{prefix}.attn.proj",
            f"{prefix}.mlp",
            f"{prefix}.mlp.fc1",
            f"{prefix}.mlp.fc2",
            f"{prefix}.norm1",
        ]
    for layer in range(llm_layers):
        prefix = f"model.language_model.layers.{layer}"
        names += [prefix] + [
            f"{prefix}.self_attn.{leaf}" for leaf in ("q_proj", "k_proj", "v_proj", "o_proj")
        ] + [f"{prefix}.mlp.{leaf}" for leaf in ("gate_proj", "up_proj", "down_proj")]
    names += ["model.visual.merger", "model.visual.merger.mlp.0"]
    return names


# ----------------------------------------------------- labels and the prompt


def test_corpus_labels_are_the_inference_labels(optical_views: list[cb.SourceView]) -> None:
    """The corpus writes no label of its own — §2.5's single source of truth."""
    records = cb.label_views(optical_views)
    assert [record.label for record in records] == [
        label_for_view("TC", 1, Modality.OPTICAL, "Sentinel-2 L2A"),
        label_for_view("NDVI", 2, Modality.OPTICAL, "Sentinel-2 L2A"),
    ]
    # The NDVI label must carry its fixed-domain note: it is how the model learns
    # the colour is absolute rather than stretched to the scene.
    assert "fixed scale -1 to +1" in records[1].label


def test_training_system_prompt_is_byte_identical_to_inference(
    optical_views: list[cb.SourceView],
) -> None:
    """The FactSheet the trainer injects is the one the Phase 4 VLM is served."""
    sample = cb.assemble(
        sample_id="evidence_qa:1",
        source=cb.CorpusSource.EVIDENCE_QA,
        task=TaskType.VQA,
        pair_type=PairType.SINGLE,
        views=optical_views,
        question="What is the mean NDVI?",
        answer=f"The mean NDVI is {cb.cite('spectral_index_analyzer.ndvi_mean', 0.62)}.",
        fact_sheet={"spectral_index_analyzer.ndvi_mean": 0.62},
    )
    served = build_prompt(
        task=TaskType.VQA,
        pair_type=PairType.SINGLE,
        sheet=cb.fact_sheet_from({"spectral_index_analyzer.ndvi_mean": 0.62}),
        views=[
            ViewInput(label=record.label, rgb=np.zeros((4, 4, 3), dtype=np.uint8))
            for record in sample.views
        ],
        question="What is the mean NDVI?",
    )
    assert sample.system == served.system
    assert sample.user == served.user


def test_fact_sheet_block_matches_the_phase_four_renderer() -> None:
    """The injected block is rendered by the inference renderer, key for key."""
    sheet = cb.fact_sheet_from(BEN_FACTS)
    block = render_fact_sheet(sheet, get_template())
    sample = cb.assemble(
        sample_id="ben2:x",
        source=cb.CorpusSource.BIGEARTHNET_V2,
        task=TaskType.SCENE_CLASSIFY,
        pair_type=PairType.CROSS_MODAL,
        views=[cb.SourceView(ViewId.TC, "a.jpg", Modality.OPTICAL)],
        question="Which land-cover classes are present in this scene?",
        answer="The scene contains broad-leaved forest and pastures.",
        fact_sheet=BEN_FACTS,
    )
    assert block in sample.system
    for key in BEN_FACTS:
        assert f"  {key} = " in sample.system
    assert sample.fact_sheet == BEN_FACTS


def test_fact_keys_must_be_namespaced() -> None:
    """An unnamespaced key would train citations the validator cannot resolve."""
    with pytest.raises(cb.CorpusError):
        cb.fact_sheet_from({"ndvi_mean": 0.62})


# ------------------------------------------------------- conversation shaping


def test_bigearthnet_row_becomes_conversation_turns(ben_row: dict[str, Any]) -> None:
    """A mock BEN item translates into system -> user -> assistant turns."""
    samples = cb.from_bigearthnet(ben_row, rng=random.Random(0), augment=False)
    by_task = {sample.task: sample for sample in samples}

    assert TaskType.SCENE_CLASSIFY in by_task
    assert TaskType.CROSS_MODAL_VQA in by_task
    for sample in samples:
        assert [message.role for message in sample.messages] == [
            "system",
            "user",
            "assistant",
        ]
        assert sample.source is cb.CorpusSource.BIGEARTHNET_V2
        assert sample.pair_type is PairType.CROSS_MODAL
        assert [view.slot for view in sample.views] == list(range(1, len(sample.views) + 1))
        assert sample.meta.gsd_m == cb.BEN_GSD_M
        assert sample.meta.split == "train"

    classify = by_task[TaskType.SCENE_CLASSIFY]
    assert classify.assistant == "The scene contains Broad-leaved forest and Pastures."
    assert "Which land-cover classes" in classify.user


def test_radar_answer_carries_tool_scalar_citations(ben_row: dict[str, Any]) -> None:
    """Every number in a generated answer is tagged with the key it came from."""
    samples = cb.from_bigearthnet(ben_row, rng=random.Random(0), augment=False)
    radar = next(s for s in samples if s.task is TaskType.CROSS_MODAL_VQA)

    assert "[sar_backscatter_analyzer.sigma0_vv_db_mean]" in radar.assistant
    marked = strip_citation_markers(radar.assistant, cb.fact_sheet_from(radar.fact_sheet))
    assert marked.unknown_keys == []
    assert set(marked.cited_keys) <= set(radar.fact_sheet)


def test_every_generated_number_resolves_to_the_fact_sheet(ben_row: dict[str, Any]) -> None:
    """Every answer we supervise passes the evidence audit.

    The §8.3 metric scores 1.0 on our own supervision by construction; if it does
    not, the corpus is teaching the failure it exists to prevent.
    """
    samples = cb.from_bigearthnet(ben_row, rng=random.Random(1), augment=False)
    samples += cb.build_evidence_qa(
        sample_id="evidence_qa:x",
        views=[cb.SourceView(ViewId.TC, "a.jpg", Modality.OPTICAL)],
        fact_sheet=dict(
            BEN_FACTS,
            **{
                "change_statistics.changed_area_pct": 7.42,
                "spectral_index_analyzer.ndbi_mean_pre": -0.12,
                "spectral_index_analyzer.ndbi_mean_post": 0.08,
            },
        ),
        rng=random.Random(2),
    )
    for sample in samples:
        sheet = cb.fact_sheet_from(sample.fact_sheet)
        marked = strip_citation_markers(sample.assistant, sheet)
        result = validate(marked.text, sheet)
        assert result.uncited_numeric_spans == [], sample.id


def test_an_invented_number_fails_the_build(optical_views: list[cb.SourceView]) -> None:
    """A number with no measurement behind it is a build error, not a warning."""
    with pytest.raises(cb.CitationError, match="no measurement supports"):
        cb.assemble(
            sample_id="bad:1",
            source=cb.CorpusSource.EVIDENCE_QA,
            task=TaskType.VQA,
            pair_type=PairType.SINGLE,
            views=optical_views,
            question="How much changed?",
            answer="About 41.7% of the scene changed.",
            fact_sheet={"spectral_index_analyzer.ndvi_mean": 0.62},
        )


def test_a_citation_to_an_absent_key_fails_the_build(
    optical_views: list[cb.SourceView],
) -> None:
    """Citing a key the sheet does not carry is the same defect, one step earlier."""
    with pytest.raises(cb.CitationError, match="absent from its FactSheet"):
        cb.assemble(
            sample_id="bad:2",
            source=cb.CorpusSource.EVIDENCE_QA,
            task=TaskType.VQA,
            pair_type=PairType.SINGLE,
            views=optical_views,
            question="What is the mean NDBI?",
            answer="The mean NDBI is 0.62 [spectral_index_analyzer.ndbi_mean].",
            fact_sheet={"spectral_index_analyzer.ndvi_mean": 0.62},
        )


def test_units_are_written_so_the_validator_can_bind_them() -> None:
    """A percentage claim must say ``%``, or it may not cite a ``*_pct`` scalar."""
    assert cb.cite("change_statistics.changed_area_pct", 7.42).startswith("7.42%")
    assert cb.cite("sar_backscatter_analyzer.sigma0_vv_db_mean", -8.4).startswith("-8.4 dB")
    assert cb.cite("spectral_index_analyzer.ndvi_mean", 0.62).startswith("0.62 [")


def test_messages_must_be_system_user_assistant() -> None:
    """The schema refuses a conversation the trainer could not consume."""
    with pytest.raises(ValueError, match="system/user/assistant"):
        cb.CorpusSample(
            id="x",
            source=cb.CorpusSource.VRSBENCH,
            task=TaskType.VQA,
            pair_type=PairType.SINGLE,
            views=[],
            messages=[
                cb.Message(role="user", content="q"),
                cb.Message(role="assistant", content="a"),
            ],
        )


# ------------------------------------------------------------ the other five


VRSBENCH_ROW: dict[str, Any] = {
    # The published shape: one image, three nested annotation sets.
    "image": "00002_0000.png",
    "image_path": "views/vrsbench/00002_0000/TC.jpg",
    "split": "train",
    "caption": "A rural expressway toll station with several small vehicles beside it.",
    "qa_pairs": [
        {
            "ques_id": 1,
            "question": "What is the main structure in the center of the image?",
            "type": "scene type",
            "answer": "expressway-toll-station",
        },
        {
            "ques_id": 3,
            "question": "How many expressway-toll-stations are visible in the image?",
            "type": "object quantity",
            "answer": "1",
        },
    ],
    "objects": [
        {
            "obj_id": 0,
            "referring_sentence": "The toll station is positioned at the center.",
            "obj_cls": "expressway-toll-station",
            "obj_coord": [0.45, 0.43, 0.59, 0.59],
        }
    ],
}


def test_vrsbench_yields_caption_vqa_and_grounding_from_one_row() -> None:
    """One VRSBench image carries a caption, N questions and M referred objects."""
    samples = cb.from_vrsbench(VRSBENCH_ROW, augment=False)
    by_task = {sample.task: sample for sample in samples}
    assert set(by_task) == {TaskType.CAPTION, TaskType.VQA, TaskType.COUNT,
                            TaskType.GROUNDING}

    assert by_task[TaskType.CAPTION].assistant == VRSBENCH_ROW["caption"]
    assert by_task[TaskType.VQA].assistant == "Expressway-toll-station"
    assert all(s.id.startswith("vrsbench:00002_0000:") for s in samples)
    assert by_task[TaskType.CAPTION].views[0].label == label_for_view(
        "TC", 1, Modality.OPTICAL, None
    )


def test_vrsbench_counts_are_cited_rather_than_stated_bare() -> None:
    """A bare "1" would be a number with no measurement behind it."""
    count = next(
        s for s in cb.from_vrsbench(VRSBENCH_ROW, augment=False) if s.task is TaskType.COUNT
    )
    assert count.fact_sheet == {"object_counter.count": 1.0}
    assert "[object_counter.count]" in count.assistant


def test_vrsbench_grounding_uses_the_shared_serialiser() -> None:
    """``obj_coord`` is normalised 0-1; the corpus emits the 0-1000 canonical form."""
    ground = next(
        s
        for s in cb.from_vrsbench(VRSBENCH_ROW, augment=False)
        if s.task is TaskType.GROUNDING
    )
    parsed = box_format.parse(ground.assistant)
    assert len(parsed) == 1
    assert (parsed[0].x_min, parsed[0].y_min, parsed[0].x_max, parsed[0].y_max) == (
        450,
        430,
        590,
        590,
    )
    assert parsed[0].label == "expressway-toll-station"
    assert ground.assistant == box_format.serialise(parsed)


def test_vrsbench_test_split_is_quarantined() -> None:
    """The official test split is Phase 8's, and the builder says so loudly."""
    with pytest.raises(cb.LeakageError):
        cb.from_vrsbench(dict(VRSBENCH_ROW, split="test"))


def test_rsvqa_count_becomes_a_cited_measurement() -> None:
    """A count is a measurement, and is supervised in the same citation form."""
    sample = cb.from_rsvqa(
        {
            "image_id": "7",
            "image_path": "views/rsvqa/7/TC.jpg",
            "question": "How many buildings are there in this image?",
            "answer": "12",
            "type": "count",
            "split": "train",
        }
    )
    assert sample.task is TaskType.COUNT
    assert sample.fact_sheet == {"object_counter.count": 12.0}
    assert "[object_counter.count]" in sample.assistant
    assert validate(
        strip_citation_markers(sample.assistant).text, cb.fact_sheet_from(sample.fact_sheet)
    ).uncited_numeric_spans == []


def test_cdvqa_is_bi_temporal_with_dated_pre_and_post_labels() -> None:
    """The two epochs are distinguished by their labels, dates included."""
    sample = cb.from_cdvqa(
        {
            "pair_id": "p1",
            "question": "What replaced the farmland?",
            "answer": "new buildings",
            "pre_path": "views/cdvqa/p1/TC_pre.jpg",
            "post_path": "views/cdvqa/p1/TC_post.jpg",
            "pre_time": datetime(2019, 3, 14, tzinfo=UTC),
            "post_time": datetime(2021, 7, 2, tzinfo=UTC),
            "split": "train",
            "fact_sheet": {"change_statistics.changed_area_pct": 7.42},
        }
    )
    assert sample.pair_type is PairType.BI_TEMPORAL
    assert sample.task is TaskType.CHANGE_VQA
    assert sample.views[0].label == "Image 1 (optical true colour, pre-change, 2019-03-14)"
    assert sample.views[1].label == "Image 2 (optical true colour, post-change, 2021-07-02)"
    assert "[change_statistics.changed_area_pct]" in sample.assistant


def test_grounding_answers_round_trip_through_the_shared_parser() -> None:
    """The corpus emits what ``tools/text_grounding.py`` parses — §4.5's whole point."""
    sample = cb.from_dior_rsvg(
        {
            "image_id": "00123",
            "image_path": "views/dior/00123/TC.jpg",
            "expression": "the white aircraft on the left apron",
            "bbox": [200, 400, 400, 600],
            "width": 800,
            "height": 800,
            "split": "train",
        },
        augment=False,
    )
    assert sample.task is TaskType.GROUNDING
    assert sample.assistant.startswith(box_format.REF_START)
    parsed = box_format.parse(sample.assistant)
    assert len(parsed) == 1
    assert (parsed[0].x_min, parsed[0].y_min, parsed[0].x_max, parsed[0].y_max) == (
        250,
        500,
        500,
        750,
    )
    assert parsed[0].label == "the white aircraft on the left apron"
    assert parsed[0].to_pixels(800, 800) == (200, 400, 400, 600)
    # And it is the serialiser's own bytes, not a reconstruction of them.
    assert sample.assistant == box_format.serialise(parsed)
    assert sample.user == "Locate the white aircraft on the left apron."


def test_grounding_boxes_need_the_frame_they_were_measured_on() -> None:
    """Pixel boxes without a frame cannot be normalised, and are refused."""
    with pytest.raises(cb.CorpusError, match="width and height"):
        cb.from_dior_rsvg(
            {
                "image_id": "1",
                "image_path": "a.jpg",
                "expression": "a ship",
                "bbox": [1, 2, 3, 4],
                "split": "train",
            },
            augment=False,
        )


# --------------------------------------------------------------- evidence_qa


def test_evidence_qa_generates_citation_behaviour(
    optical_views: list[cb.SourceView],
) -> None:
    """The synthetic source supervises exactly what no public corpus teaches."""
    samples = cb.build_evidence_qa(
        sample_id="evidence_qa:ben2:x",
        views=optical_views,
        fact_sheet=dict(
            BEN_FACTS,
            **{
                "change_statistics.changed_area_pct": 7.42,
                "spectral_index_analyzer.ndbi_mean_pre": -0.12,
                "spectral_index_analyzer.ndbi_mean_post": 0.08,
            },
        ),
        rng=random.Random(0),
    )
    answers = {sample.id.rsplit(":", 1)[-1]: sample.assistant for sample in samples}

    assert answers["ndvi"].startswith("The mean NDVI is 0.62 [spectral_index_analyzer.ndvi_mean]")
    assert "dense, healthy vegetation" in answers["ndvi"]
    assert "[sar_backscatter_analyzer.sigma0_vv_db_mean]" in answers["agreement"]
    assert "7.42% [change_statistics.changed_area_pct]" in answers["change"]
    assert "[spectral_index_analyzer.ndbi_mean_post]" in answers["change"]
    assert all(sample.source is cb.CorpusSource.EVIDENCE_QA for sample in samples)


def test_evidence_qa_teaches_refusal_as_well_as_citation(
    optical_views: list[cb.SourceView],
) -> None:
    """Rule 3: an unmeasured quantity is declined, not guessed."""
    samples = cb.build_evidence_qa(
        sample_id="evidence_qa:y",
        views=optical_views,
        fact_sheet={"spectral_index_analyzer.ndvi_mean": 0.62},
        rng=random.Random(0),
    )
    refusal = next(sample for sample in samples if sample.id.endswith(":unmeasured"))
    assert "did not" in refusal.assistant
    assert "[" not in refusal.assistant
    assert validate(
        refusal.assistant, cb.fact_sheet_from(refusal.fact_sheet)
    ).uncited_numeric_spans == []


def test_index_readings_follow_the_fixed_domain() -> None:
    """The interpretation language is a function of the value, not of the scene."""
    assert "dense" in cb.describe_ndvi(0.75)
    assert "unvegetated" in cb.describe_ndvi(0.0)
    assert "water" in cb.describe_ndvi(-0.4)
    assert "low built-up" in cb.describe_ndbi(-0.3)
    assert "water" in cb.describe_backscatter(-20.0)


# -------------------------------------------------------------- augmentation


def test_swir_dropout_removes_the_views_and_the_facts_together() -> None:
    """A dropped view must take its measurements with it (§7.3)."""
    views = [
        cb.SourceView(ViewId.TC, "TC.jpg", Modality.OPTICAL),
        cb.SourceView(ViewId.SWIR, "SWIR.jpg", Modality.OPTICAL),
        cb.SourceView(ViewId.NDBI, "NDBI.png", Modality.OPTICAL),
    ]
    facts = {
        "spectral_index_analyzer.ndvi_mean": 0.62,
        "spectral_index_analyzer.ndbi_mean": -0.08,
        "spectral_index_analyzer.built_up_fraction_pct": 3.0,
    }
    kept_views, kept_facts, tags = cb.apply_augmentations(
        [(cb.Augmentation.SWIR_DROPOUT, "swir_dropout")], views, facts
    )
    assert [str(view.view_id) for view in kept_views] == ["TC"]
    assert set(kept_facts) == {"spectral_index_analyzer.ndvi_mean"}
    assert tags == ["swir_dropout"]


def test_single_pol_drops_the_dual_pol_view_and_its_ratio() -> None:
    """RISAT has one polarisation, so VH-derived evidence must disappear with it."""
    views = [
        cb.SourceView(ViewId.SARFC, "SARFC.jpg", Modality.SAR),
        cb.SourceView(ViewId.SARDB, "SARDB.png", Modality.SAR),
    ]
    facts = {
        "sar_backscatter_analyzer.sigma0_vv_db_mean": -8.4,
        "sar_backscatter_analyzer.sigma0_vh_db_mean": -14.5,
        "sar_backscatter_analyzer.vv_vh_ratio_db_mean": 6.1,
    }
    kept_views, kept_facts, _ = cb.apply_augmentations(
        [(cb.Augmentation.SINGLE_POL, "single_pol")], views, facts
    )
    assert [str(view.view_id) for view in kept_views] == ["SARDB"]
    assert set(kept_facts) == {"sar_backscatter_analyzer.sigma0_vv_db_mean"}


def test_augmentation_is_reproducible_from_the_seed(ben_row: dict[str, Any]) -> None:
    """Two builds of one seed produce the same corpus, tags included."""
    first = cb.from_bigearthnet(ben_row, rng=random.Random(7))
    second = cb.from_bigearthnet(ben_row, rng=random.Random(7))
    assert [s.model_dump(mode="json") for s in first] == [
        s.model_dump(mode="json") for s in second
    ]
    assert all(
        tag.startswith(("rescale_", "swir_dropout", "single_pol"))
        for sample in first
        for tag in sample.meta.augmentations
    )


# ------------------------------------------------------- dedup and quarantine


def _scene(shift: int = 0, noise: int = 0, seed: int = 1) -> np.ndarray:
    """A smooth synthetic scene, optionally brightened and speckled.

    Smooth rather than a checkerboard on purpose: a pHash reads low-frequency
    structure, and a pattern that is pure high frequency hashes to noise — which
    would make this test measure the fixture rather than the hash.
    """
    row, column = np.mgrid[0:128, 0:128]
    field = 120 + 80 * np.sin(column / 17.0) * np.cos(row / 23.0) + shift
    if noise:
        field = field + np.random.default_rng(seed).integers(-noise, noise + 1, field.shape)
    return np.clip(field, 0, 255).astype(np.uint8)


def _other_scene() -> np.ndarray:
    """A visibly different scene, for the far end of the comparison."""
    row, _ = np.mgrid[0:128, 0:128]
    return np.clip(120 + 80 * np.sin(row / 9.0), 0, 255).astype(np.uint8)


def test_phash_is_near_for_a_perturbed_image_and_far_for_a_different_one() -> None:
    """Near-duplicate detection is what catches the VRSBench/DIOR overlap (§4.7)."""
    original = cb.phash(_scene())
    rescaled = cb.phash(_scene(shift=8, noise=3))
    cropped = cb.phash(_scene()[2:126, 2:126])
    unrelated = cb.phash(_other_scene())

    # A re-encoded, slightly brightened crop of one scene — the shape the
    # VRSBench/DIOR-RSVG overlap actually takes — stays inside the threshold.
    assert cb.hamming(original, rescaled) <= cb.NEAR_DUP_DISTANCE
    assert cb.hamming(original, cropped) <= cb.NEAR_DUP_DISTANCE
    assert cb.hamming(original, unrelated) > cb.NEAR_DUP_DISTANCE


def test_deduplicator_drops_exact_and_near_duplicates() -> None:
    """SHA-256 first, then pHash, across all sources."""
    dedup = cb.Deduplicator()
    base = cb.phash(_scene())

    assert dedup.check("a", "sha-1", base).verdict is cb.DedupVerdict.NEW
    assert dedup.check("b", "sha-1", base).verdict is cb.DedupVerdict.EXACT_DUPLICATE
    near = dedup.check("c", "sha-2", cb.phash(_scene(shift=8, noise=3)))
    assert near.verdict is cb.DedupVerdict.NEAR_DUPLICATE
    assert near.collided_with == "a"
    assert dedup.check("d", "sha-3", cb.phash(_other_scene())).accepted


def test_many_samples_of_one_image_are_not_duplicates_of_each_other() -> None:
    """§4.7 dedups images; four of the six sources publish many samples per image.

    VRSBench averages ten annotations per tile and CDVQA around forty questions
    per pair. Checked as if every sample were its own image, the second
    annotation collides with the first on an exact hash and is dropped — which
    is 205 000 VRSBench samples collapsing to 20 264 and every §5 target going
    out of reach. A *different* image with the same bytes is still a duplicate.
    """
    dedup = cb.Deduplicator()
    base = cb.phash(_scene())

    assert dedup.check("vrsbench:t1:cap", "sha-1", base, image_key="t1").accepted
    assert dedup.check("vrsbench:t1:qa0", "sha-1", base, image_key="t1").accepted
    assert dedup.check("vrsbench:t1:ref3", "sha-1", base, image_key="t1").accepted
    assert dedup.dropped == []

    collision = dedup.check("dior_rsvg:t2:0", "sha-1", base, image_key="t2")
    assert collision.verdict is cb.DedupVerdict.EXACT_DUPLICATE
    assert collision.collided_with == "t1"


def test_the_image_key_is_the_view_a_corpus_line_points_at() -> None:
    """Not the sample id: the path is what the line actually resolves to."""
    sample = _sample(1, cb.CorpusSource.VRSBENCH)
    assert cb.image_key_of(sample) == sample.views[0].path


def test_quarantined_test_images_are_a_build_error() -> None:
    """A leaked benchmark is worse than no benchmark, so the build fails."""
    dedup = cb.Deduplicator()
    dedup.quarantine("vrsbench_test:1", "sha-test", cb.phash(_scene()))

    verdict = dedup.check("dior_rsvg:9", "sha-other", cb.phash(_scene(shift=8, noise=3)))
    assert verdict.verdict is cb.DedupVerdict.QUARANTINED
    with pytest.raises(cb.LeakageError, match="quarantined test split"):
        dedup.assert_clean()


def test_sha256_of_bytes_and_of_a_file_agree(tmp_path: Path) -> None:
    """One hash function, whether the pixels are in memory or on disk."""
    payload = b"pretend this is a GeoTIFF"
    path = tmp_path / "scene.tif"
    path.write_bytes(payload)
    assert cb.sha256_of(payload) == cb.sha256_of(path)


# ------------------------------------------------------------- corpus wiring


def _sample(index: int, source: cb.CorpusSource, split: str = "train") -> cb.CorpusSample:
    """A minimal valid sample, for the composition and IO tests."""
    return cb.assemble(
        sample_id=f"{source.value}:{index}",
        source=source,
        task=TaskType.VQA,
        pair_type=PairType.SINGLE,
        views=[cb.SourceView(ViewId.TC, f"views/{index}/TC.jpg", Modality.OPTICAL)],
        question="What is in this scene?",
        answer="Farmland.",
        meta=cb.SampleMeta(split=split, source_split=split),
    )


def test_a_reservoir_caps_what_it_holds_and_keeps_arrival_order() -> None:
    """The whole point: a uniform draw whose memory is the target, not the stream.

    ``take`` needs the population as a Sequence, and materialising 9.6 M reBEN
    samples to choose 18 k of them is what got the build OOM-killed.
    """
    reservoir = cb.Reservoir(10, random.Random(0))
    for index in range(10_000):
        reservoir.offer(_sample(index, cb.CorpusSource.VRSBENCH))

    kept = reservoir.collect()
    assert len(kept) == 10
    assert reservoir.seen == 10_000
    # Arrival order survives, so two builds of one seed stay diffable.
    order = [int(sample.id.split(":")[1]) for sample in kept]
    assert order == sorted(order)
    # And it is a draw from the whole stream, not just the head.
    assert max(order) > 10


def test_a_reservoir_without_a_capacity_keeps_everything() -> None:
    """``None`` is 'no target' — unbounded, which is the caller's choice to make."""
    reservoir = cb.Reservoir(None, random.Random(0))
    for index in range(5):
        reservoir.offer(_sample(index, cb.CorpusSource.VRSBENCH))
    assert len(reservoir.collect()) == 5
    assert cb.Reservoir(0, random.Random(0)).collect() == []


def test_streaming_build_never_materialises_the_population(tmp_path: Path) -> None:
    """``build_corpus_streaming`` reads an iterator once and holds only the budget.

    The generator counts what it was asked for: if the builder listed its input,
    the count would be the population rather than the corpus.
    """
    produced = 0

    def stream() -> Any:
        nonlocal produced
        for index in range(2_000):
            produced += 1
            split = "train" if index % 2 else "val"
            yield cb.CorpusSource.VRSBENCH, _sample(index, cb.CorpusSource.VRSBENCH, split)

    report = cb.build_corpus_streaming(
        stream(), tmp_path, composition={cb.CorpusSource.VRSBENCH: 40}, seed=7
    )

    assert produced == 2_000
    row = next(r for r in report.sources if r.source is cb.CorpusSource.VRSBENCH)
    assert row.built == 2_000
    assert row.kept == 2_000
    assert row.train + row.val == 40
    assert report.train + report.val == 40
    assert len(cb.read_jsonl(tmp_path / "train.jsonl")) == report.train
    assert len(cb.read_jsonl(tmp_path / "val.jsonl")) == report.val


def test_streaming_build_still_refuses_a_leaked_image(tmp_path: Path) -> None:
    """§4.7 is not relaxed by streaming: the quarantine check runs as rows pass."""
    sample = _sample(1, cb.CorpusSource.VRSBENCH)
    sample.meta.image_sha256 = "abc"
    dedup = cb.Deduplicator()
    dedup.quarantine("test:1", sha256="abc")
    with pytest.raises(cb.LeakageError):
        cb.build_corpus_streaming(
            iter([(cb.CorpusSource.VRSBENCH, sample)]), tmp_path, dedup=dedup
        )


def test_corpus_round_trips_through_jsonl(tmp_path: Path) -> None:
    """The written line is the §3 schema, and reads back into the same object."""
    sample = _sample(1, cb.CorpusSource.VRSBENCH)
    path = tmp_path / "train.jsonl"
    assert cb.write_jsonl(path, [sample]) == 1

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["source"] == "vrsbench"
    assert payload["task"] == "VQA"
    assert set(payload) == {
        "id",
        "source",
        "task",
        "pair_type",
        "views",
        "messages",
        "fact_sheet",
        "meta",
    }
    assert cb.read_jsonl(path)[0] == sample


def test_build_corpus_applies_the_composition_and_splits(tmp_path: Path) -> None:
    """Per-source targets, the official split, and a report of both (§5)."""
    sources = {
        cb.CorpusSource.VRSBENCH: [_sample(i, cb.CorpusSource.VRSBENCH) for i in range(10)],
        cb.CorpusSource.RSVQA_HR: [
            _sample(i, cb.CorpusSource.RSVQA_HR, split="val") for i in range(4)
        ],
    }
    report = cb.build_corpus(
        sources,
        tmp_path,
        composition={cb.CorpusSource.VRSBENCH: 6, cb.CorpusSource.RSVQA_HR: 4},
        seed=42,
    )
    assert report.train == 6
    assert report.val == 4
    assert len(cb.read_jsonl(tmp_path / "train.jsonl")) == 6
    assert len(cb.read_jsonl(tmp_path / "val.jsonl")) == 4
    vrsbench = next(row for row in report.sources if row.source is cb.CorpusSource.VRSBENCH)
    assert (vrsbench.built, vrsbench.kept, vrsbench.train, vrsbench.track) == (10, 10, 6, "B")


def test_build_corpus_refuses_a_leaked_image(tmp_path: Path) -> None:
    """Step 3 of §4.7 is an assertion, not a warning."""
    dedup = cb.Deduplicator()
    dedup.quarantine("vrsbench_test:1", "sha-leak", None)
    sample = _sample(1, cb.CorpusSource.VRSBENCH)
    sample.meta.image_sha256 = "sha-leak"

    with pytest.raises(cb.LeakageError):
        cb.build_corpus({cb.CorpusSource.VRSBENCH: [sample]}, tmp_path, dedup=dedup)


def test_composition_totals_the_planned_corpus() -> None:
    """§5's table: 65,000 samples, both tracks present."""
    assert sum(cb.COMPOSITION.values()) == 65_000
    assert set(cb.TRACK.values()) == {"A", "B"}
    assert cb.TRACK[cb.CorpusSource.EVIDENCE_QA] == "A"


def test_ben6_merge_covers_the_official_nineteen() -> None:
    """The reference-replication metric needs every 19-class label mapped (§8.2)."""
    assert len(cb.BEN19_CLASSES) == 19
    assert set(cb.BEN6_MERGE) == set(cb.BEN19_CLASSES)
    assert cb.ben6_labels(["Broad-leaved forest", "Coniferous forest"]) == ["Forest"]


# ----------------------------------------------------------- QLoRA: the model


def test_profile_a_loads_with_the_frozen_recipe() -> None:
    """Profile A is the §6.1 YAML, validated rather than trusted."""
    profile = qlora.load_profile(qlora.DEFAULT_PROFILE_PATH)
    assert profile.base_model == "Qwen/Qwen3-VL-8B-Instruct"
    assert profile.attn_implementation == "sdpa"
    assert profile.quantization.load_in_4bit
    assert profile.quantization.bnb_4bit_quant_type == "nf4"
    assert profile.quantization.bnb_4bit_use_double_quant
    assert profile.lora.vision_block_count == 8
    assert profile.train.effective_batch == 16
    assert profile.env["HIP_VISIBLE_DEVICES"] == "0"
    assert tuple(profile.lora.target_modules_llm) == qlora.LLM_TARGET_MODULES
    assert tuple(profile.lora.target_modules_vision) == qlora.VISION_TARGET_MODULES


def test_every_profile_carries_the_rdna3_safeguards() -> None:
    """The three settings a gfx1100 run cannot start without.

    ``sdpa`` because flash-attention's CK path is unstable on RDNA3;
    ``HIP_VISIBLE_DEVICES=0`` because the 7800X3D iGPU otherwise enumerates as
    device 0; ``expandable_segments`` because the allocator fragments badly
    across a run with six variable-size views per sample.
    """
    for path in (qlora.DEFAULT_PROFILE_PATH, qlora.FALLBACK_PROFILE_PATH):
        profile = qlora.load_profile(path)
        assert profile.attn_implementation == "sdpa", path
        assert profile.attn_implementation != "flash_attention_2"
        assert profile.env["HIP_VISIBLE_DEVICES"] == "0", path
        assert profile.env["PYTORCH_HIP_ALLOC_CONF"] == "expandable_segments:True", path
        # The colon inside the value must survive YAML parsing as a string.
        assert isinstance(profile.env["PYTORCH_HIP_ALLOC_CONF"], str)


def test_the_defaults_are_the_rdna3_safeguards_too() -> None:
    """A profile written without an ``env`` block still hides the iGPU."""
    default = qlora.QLoraProfile()
    assert default.attn_implementation == "sdpa"
    assert default.env["HIP_VISIBLE_DEVICES"] == "0"
    assert default.env["PYTORCH_HIP_ALLOC_CONF"] == "expandable_segments:True"


def test_profile_b_is_a_config_switch_not_a_rewrite() -> None:
    """The fallback differs only where §6.2 says it does."""
    fallback = qlora.load_profile(qlora.FALLBACK_PROFILE_PATH)
    assert fallback.base_model == "Qwen/Qwen3-VL-4B-Instruct"
    assert not fallback.quantization.load_in_4bit
    assert fallback.train.optim == "adamw_torch_fused"
    assert fallback.train.effective_batch == 16
    assert fallback.lora.vision_block_count == 8


def test_unknown_vision_block_setting_is_fatal() -> None:
    """A profile that cannot be understood does not get to run."""
    with pytest.raises(qlora.ProfileError):
        _ = qlora.LoraSpec(vision_blocks="most_of_them").vision_block_count


def test_lora_targets_the_llm_projections_and_the_last_eight_vision_blocks() -> None:
    """The §6.1 target set, and no more of the vision tower than that."""
    names = qwen_module_names()
    targets = qlora.resolve_target_modules(names, qlora.LoraSpec())
    llm, vision = qlora.split_targets(targets)

    assert {name.rsplit(".", 1)[-1] for name in llm} == set(qlora.LLM_TARGET_MODULES)
    assert len(llm) == 36 * 7
    assert {qlora.vision_block_index(name) for name in vision} == set(range(19, 27))
    assert {name.rsplit(".", 1)[-1] for name in vision} == set(qlora.VISION_TARGET_MODULES)
    assert not any("blocks.18." in name for name in targets)
    assert not any(name.endswith("lm_head") or name.endswith("embed_tokens") for name in targets)


def test_vision_adaptation_can_be_widened_or_switched_off() -> None:
    """``all`` and ``none`` are the two ends of the same knob."""
    names = qwen_module_names()
    assert qlora.select_vision_blocks(names, -1) == set(range(27))
    assert qlora.select_vision_blocks(names, 0) == set()
    all_blocks = qlora.resolve_target_modules(names, qlora.LoraSpec(vision_blocks="all"))
    assert len(qlora.split_targets(all_blocks)[1]) == 27 * 4


def test_a_renamed_vision_subtree_is_still_found() -> None:
    """Qwen has shipped three names for this subtree; all three must resolve."""
    names = [
        "vision_tower.encoder.layers.0.attn.qkv",
        "vision_tower.encoder.layers.1.attn.qkv",
        "vision_tower.encoder.layers.1.mlp.fc2",
        "language_model.layers.0.self_attn.q_proj",
    ]
    targets = qlora.resolve_target_modules(names, qlora.LoraSpec(vision_blocks="last_1"))
    _, vision = qlora.split_targets(targets)
    assert sorted(vision) == [
        "vision_tower.encoder.layers.1.attn.qkv",
        "vision_tower.encoder.layers.1.mlp.fc2",
    ]


def test_a_checkpoint_that_matches_nothing_is_fatal() -> None:
    """Training zero parameters is a silent success; refusing is the honest outcome."""
    with pytest.raises(qlora.ProfileError, match="no LoRA target modules"):
        qlora.resolve_target_modules(["model.norm", "lm_head"], qlora.LoraSpec())


def test_target_resolution_accepts_a_torch_style_module() -> None:
    """The resolver reads ``named_modules`` when it is given a real model."""

    class FakeModel:
        """The one method the resolver uses."""

        def named_modules(self) -> list[tuple[str, object]]:
            """Mimic torch's ``(name, module)`` pairs."""
            return [(name, object()) for name in ["", *qwen_module_names(vision_blocks=9,
                                                                        llm_layers=2)]]

    targets = qlora.resolve_target_modules(FakeModel(), qlora.LoraSpec())
    assert len(qlora.split_targets(targets)[0]) == 2 * 7
    assert {qlora.vision_block_index(n) for n in qlora.split_targets(targets)[1]} == set(
        range(1, 9)
    )


# ------------------------------------------------------------ QLoRA: the box


def test_the_profile_fits_the_twenty_two_gigabyte_guard() -> None:
    """Both profiles are inside the budget the serving path also honours."""
    for path in (qlora.DEFAULT_PROFILE_PATH, qlora.FALLBACK_PROFILE_PATH):
        estimate = qlora.estimate_footprint(qlora.load_profile(path))
        assert estimate.fits, f"{path}: {estimate.total_gib:.1f} GiB"
        assert estimate.budget_bytes == 22 * 1024**3


def test_the_estimator_reproduces_the_measured_peak() -> None:
    """The calibration point of :data:`MEASURED_PEAKS`, held to within 1 GiB.

    The first version of this estimator predicted 12.0 GiB for a run that peaked
    at 21.49, because it modelled the language model alone: neither
    ``max_views_per_sample`` nor ``max_pixels`` reached the arithmetic at all,
    so the two knobs §6.3 says to turn first moved nothing. This pins the fit so
    that regression is visible.
    """
    measured = qlora.MEASURED_PEAKS[0]
    profile = qlora.load_profile(qlora.DEFAULT_PROFILE_PATH)
    profile.data.max_pixels = measured.max_pixels
    profile.data.max_views_per_sample = measured.views
    profile.data.max_seq_len = measured.max_seq_len
    profile.train.per_device_train_batch_size = measured.batch

    estimate = qlora.estimate_footprint(profile)
    error_gib = abs(estimate.total_bytes - measured.peak_bytes) / 1024**3
    measured_gib = measured.peak_bytes / 1024**3
    assert error_gib < 1.0, (
        f"predicted {estimate.total_gib:.2f} GiB for a measured {measured_gib:.2f} GiB"
    )


def test_the_vision_knobs_actually_move_the_estimate() -> None:
    """Views and pixels dominate, because the ViT's attention is quadratic in patches."""
    profile = qlora.load_profile(qlora.DEFAULT_PROFILE_PATH)
    base = qlora.estimate_footprint(profile)

    fewer_views = profile.model_copy(deep=True)
    fewer_views.data.max_views_per_sample = 4
    smaller = profile.model_copy(deep=True)
    smaller.data.max_pixels = 320 * 320

    assert qlora.estimate_footprint(fewer_views).total_bytes < base.total_bytes
    assert qlora.estimate_footprint(smaller).total_bytes < base.total_bytes
    # Quadratic, not linear: two thirds of the patches is well under two thirds
    # of the vision term.
    assert qlora.estimate_footprint(fewer_views).vision_bytes < base.vision_bytes * 0.67


def test_the_shipped_profiles_keep_a_margin_under_the_guard() -> None:
    """Fitting inside 22 GiB is not enough; the corpus's longest samples need room.

    The measured sanity run cleared the guard by 0.5 GiB, and a sanity run is 100
    samples at a capped length. 65,000 samples will produce longer ones.
    """
    for path in (qlora.DEFAULT_PROFILE_PATH, qlora.FALLBACK_PROFILE_PATH):
        estimate = qlora.estimate_footprint(qlora.load_profile(path))
        assert estimate.fits, f"{path}: {estimate.total_gib:.2f} GiB"
        assert estimate.within_margin, f"{path}: {estimate.total_gib:.2f} GiB"
        assert estimate.safe_bytes == 19 * 1024**3
        assert estimate.safe_bytes < estimate.budget_bytes


def test_the_adapter_is_about_fifty_million_parameters() -> None:
    """§6.1's expectation, and the reason the adapter ships at ~200 MB."""
    estimate = qlora.estimate_footprint(qlora.load_profile(qlora.DEFAULT_PROFILE_PATH))
    assert 40e6 < estimate.adapter_params < 60e6
    assert estimate.adapter_bytes < 250 * 1024**2


def test_vision_adapters_are_a_real_share_of_the_trainable_parameters() -> None:
    """Vision adapters are a visible share of the trainable parameters.

    Turning the vision tower off must change the count; if it does not, the
    vision targets never attached.
    """
    profile = qlora.load_profile(qlora.DEFAULT_PROFILE_PATH)
    with_vision = qlora.estimate_footprint(profile).adapter_params
    profile.lora.vision_blocks = "none"
    without = qlora.estimate_footprint(profile).adapter_params
    assert with_vision > without


def test_an_oversized_context_is_refused_before_the_run_starts() -> None:
    """The guard exists to fail in seconds rather than forty minutes in."""
    profile = qlora.load_profile(qlora.DEFAULT_PROFILE_PATH)
    profile.data.max_seq_len = 32_768
    estimate = qlora.estimate_footprint(profile)
    assert not estimate.fits
    with pytest.raises(Exception, match="VRAM|budget|exceed"):
        qlora.guard_budget(estimate)


def test_sanity_check_truncates_to_one_hundred_items_for_one_epoch() -> None:
    """``--sanity-check`` is 100 samples, 1 epoch, and the steps that implies."""
    profile = qlora.load_profile(qlora.DEFAULT_PROFILE_PATH)
    plan = qlora.plan_run(profile, corpus_size=65_000, sanity_check=True)
    assert (plan.samples, plan.epochs, plan.steps) == (100, 1.0, 6)
    assert plan.sanity_check

    full = qlora.plan_run(profile, corpus_size=65_000)
    assert (full.samples, full.steps) == (65_000, 4_062)


def test_resume_is_automatic_only_when_a_checkpoint_exists(tmp_path: Path) -> None:
    """An interrupted overnight run restarts with the same command line."""
    assert qlora.resume_target(tmp_path, "auto") is None
    checkpoint = tmp_path / "checkpoint-250"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    assert qlora.resume_target(tmp_path, "auto") is True
    assert qlora.resume_target(tmp_path, "false") is None
    assert qlora.resume_target(tmp_path, str(checkpoint)) == str(checkpoint)

    # A named checkpoint is verified here, not by the trainer: a typo otherwise
    # surfaces after the 4-bit weights are on the card.
    with pytest.raises(qlora.ProfileError, match="not a resumable checkpoint"):
        qlora.resume_target(tmp_path, "runs/x/checkpoint-500")


def profile_save_steps() -> int:
    """The checkpoint cadence the shipped profile declares."""
    return qlora.load_profile(qlora.DEFAULT_PROFILE_PATH).train.save_steps


def test_checkpoints_carry_the_optimiser_state_that_makes_them_resumable() -> None:
    """``save_only_model`` is the difference between servable and resumable.

    True writes an adapter you can serve and cannot resume; the optimiser
    momentum and the LR schedule position are simply absent, and Adam silently
    restarts from zero moment estimates on the next run.
    """
    kwargs = qlora.sft_config_kwargs(
        profile=qlora.load_profile(qlora.DEFAULT_PROFILE_PATH),
        output_dir=Path("runs/x"),
        has_eval=False,
    )
    assert kwargs["save_only_model"] is False
    assert kwargs["save_strategy"] == "steps"
    # The profile's own value, not a number repeated here: what this test defends
    # is that the checkpoint is *resumable*, and the cadence is a separate
    # decision the profile documents (50, ~54 min apart at the measured
    # throughput). Restating it turned a deliberate profile change into a test
    # failure that said nothing about resumability.
    assert kwargs["save_steps"] == profile_save_steps()
    assert kwargs["save_total_limit"] == 3


def test_the_newest_checkpoint_is_found_by_step_not_by_mtime(tmp_path: Path) -> None:
    """``save_total_limit`` prunes, and copied trees carry meaningless mtimes."""
    assert qlora.latest_checkpoint(tmp_path) is None
    for step in (250, 1_000, 750):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "checkpoint-not-a-number").mkdir()
    (tmp_path / "adapter").mkdir()

    found = qlora.latest_checkpoint(tmp_path)
    assert found is not None
    assert qlora.checkpoint_step(found) == 1_000


def test_an_interrupt_asks_the_trainer_to_save_and_stop() -> None:
    """Ctrl+C must checkpoint at a step boundary, not raise from mid-backward."""
    pytest.importorskip("transformers", reason="transformers is the vlm-train extra")

    request = qlora.InterruptRequest()
    callback = qlora.graceful_stop_callback(request)

    class _Control:
        should_save = False
        should_training_stop = False

    control = _Control()
    callback.on_step_end(None, None, control)
    assert (control.should_save, control.should_training_stop) == (False, False)

    request.request("SIGINT")
    callback.on_step_end(None, None, control)
    assert control.should_save
    assert control.should_training_stop


def test_the_interrupt_handler_restores_itself_so_a_second_ctrl_c_aborts() -> None:
    """An operator who has decided not to wait must always be able to leave."""
    import signal

    request = qlora.InterruptRequest()
    original = signal.getsignal(signal.SIGINT)
    restore = qlora.install_interrupt_handler(request, signals=(signal.SIGINT,))
    try:
        assert signal.getsignal(signal.SIGINT) is not original
        signal.raise_signal(signal.SIGINT)
        assert request.requested
        assert request.signal_name == "SIGINT"
        # The first signal put the default handler back.
        assert signal.getsignal(signal.SIGINT) is original
    finally:
        restore()
        signal.signal(signal.SIGINT, original)


def test_profile_env_is_applied_without_clobbering_an_explicit_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hiding the iGPU matters; overriding an operator who hid something else does not."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    monkeypatch.delenv("PYTORCH_HIP_ALLOC_CONF", raising=False)
    applied = qlora.load_profile(qlora.DEFAULT_PROFILE_PATH).apply_env()
    assert applied["HIP_VISIBLE_DEVICES"] == "1"
    assert applied["PYTORCH_HIP_ALLOC_CONF"] == "expandable_segments:True"


# ----------------------------------------------------------- QLoRA: the data


def test_chat_records_carry_the_views_in_slot_order(ben_row: dict[str, Any]) -> None:
    """The image blocks and the system turn's labels describe the same views."""
    sample = cb.from_bigearthnet(ben_row, rng=random.Random(0), augment=False)[0]
    record = qlora.sample_to_chat(sample, max_views=6)

    roles = [turn["role"] for turn in record["messages"]]
    assert roles == ["system", "user", "assistant"]
    images = [block for block in record["messages"][1]["content"] if block["type"] == "image"]
    assert len(images) == 6  # capped at the §6.1 budget, from seven rendered views
    assert [block["label"] for block in images] == [
        view.label for view in sample.views[:6]
    ]
    assert record["images"] == [view.path for view in sample.views[:6]]
    assert record["messages"][0]["content"][0]["text"] == sample.system
    assert record["messages"][2]["content"][0]["text"] == sample.assistant
    assert record["messages"][1]["content"][-1]["text"] == sample.user


def test_chat_records_satisfy_trls_placeholder_contract(ben_row: dict[str, Any]) -> None:
    """Every image block is an unfilled placeholder, and ``images`` matches its count.

    ``trl.data_utils.prepare_multimodal_messages`` counts blocks of type
    ``image`` that carry no ``image`` payload and refuses the example unless that
    count equals ``len(images)``; ``SFTTrainer`` separately decides a dataset is
    vision *only* if an ``image``/``images`` key is present. Break either and the
    processor expands placeholders against an empty iterator, which surfaces as a
    bare ``StopIteration`` deep inside ``get_text_with_replacements`` — an error
    that names nothing about the cause.
    """
    for sample in cb.from_bigearthnet(ben_row, rng=random.Random(0), augment=False):
        record = qlora.sample_to_chat(sample, max_views=6)
        assert "images" in record
        assert qlora.image_placeholder_count(record) == len(record["images"])
        blocks = [
            block
            for message in record["messages"]
            for block in message["content"]
            if block["type"] == "image"
        ]
        assert blocks, sample.id
        assert all("image" not in block for block in blocks)


def test_placeholder_count_ignores_filled_blocks() -> None:
    """A block that already carries its payload is not a placeholder to fill."""
    filled = {
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": "a.png"},
                                         {"type": "image"},
                                         {"type": "text", "text": "q"}]}
        ]
    }
    assert qlora.image_placeholder_count(filled) == 1
    assert qlora.image_placeholder_count({"messages": []}) == 0


def test_loading_a_corpus_truncates_and_prefixes(tmp_path: Path) -> None:
    """The 100-item sanity path and the image-root override, on one file."""
    path = tmp_path / "train.jsonl"
    cb.write_jsonl(path, [_sample(i, cb.CorpusSource.VRSBENCH) for i in range(5)])

    records = qlora.load_corpus(path, limit=2, image_root=Path("/data"))
    assert len(records) == 2
    assert records[0]["images"][0].startswith("/data/views/")
    assert qlora.image_placeholder_count(records[0]) == len(records[0]["images"])

    with pytest.raises(qlora.ProfileError, match="does not exist"):
        qlora.load_corpus(tmp_path / "missing.jsonl")


def test_the_training_package_pulls_in_no_serving_stack() -> None:
    """Importing the trainer must not drag transformers into the API's process.

    Checked in a fresh interpreter rather than against this one's ``sys.modules``:
    once the ``vlm`` extra is installed, another test in the same session can
    import transformers for its own reasons, and an in-process assertion would
    then be measuring the suite's import order instead of this package's imports.
    """
    import subprocess
    import sys

    probe = (
        "import sys;"
        "import satquery.training.vlm.qlora as q;"
        "leaked=[m for m in ('transformers','peft','trl','datasets','bitsandbytes')"
        " if m in sys.modules];"
        "print(','.join(leaked))"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "", f"qlora imported {result.stdout.strip()}"


# --------------------------------------------------------------- the scripts


def test_train_script_sanity_flag_rewrites_the_profile() -> None:
    """``--sanity-check`` shortens the schedule as well as the corpus."""
    import train_vlm

    args = train_vlm.parse_args(["--sanity-check"])
    profile = train_vlm.resolve_profile(args)
    assert profile.run_name.endswith("-sanity")
    assert profile.train.num_train_epochs == 1
    assert profile.train.logging_steps == 1
    assert train_vlm.output_dir_for(args, profile) == Path("runs/sq-lora-v1-sanity")


def test_train_script_overrides_reach_the_profile(tmp_path: Path) -> None:
    """A corpus somewhere else, and a resume the operator chose."""
    import train_vlm

    args = train_vlm.parse_args(
        ["--corpus", str(tmp_path / "t.jsonl"), "--resume", "false", "--output-dir", "runs/x"]
    )
    profile = train_vlm.resolve_profile(args)
    assert profile.data.corpus == tmp_path / "t.jsonl"
    assert profile.train.resume_from_checkpoint == "false"
    assert train_vlm.output_dir_for(args, profile) == Path("runs/x")


def test_merge_export_composes_the_gguf_commands(tmp_path: Path) -> None:
    """The offline path is two llama.cpp calls, printed before they are run."""
    import merge_export

    convert, quantise = merge_export.gguf_commands(
        tmp_path / "sq-lora-v1-merged", Path("/opt/llama.cpp"), "Q4_K_M"
    )
    assert convert[1].endswith("convert_hf_to_gguf.py")
    assert convert[-1] == "f16"
    assert quantise[0].endswith("llama-quantize")
    assert quantise[-1] == "Q4_K_M"
    assert quantise[1].endswith("sq-lora-v1-merged-f16.gguf")
    assert quantise[2].endswith("sq-lora-v1-merged-Q4_K_M.gguf")


def test_merge_export_finds_the_base_model_from_the_run_manifest(tmp_path: Path) -> None:
    """Merging into the wrong base produces a model that loads and is nonsense."""
    import merge_export

    adapter = tmp_path / "run" / "adapter"
    adapter.mkdir(parents=True)
    (tmp_path / "run" / "run_manifest.json").write_text(
        json.dumps({"profile": {"base_model": "Qwen/Qwen3-VL-4B-Instruct"}}), encoding="utf-8"
    )
    assert (
        merge_export.resolve_base_model(adapter, qlora.DEFAULT_PROFILE_PATH, None)
        == "Qwen/Qwen3-VL-4B-Instruct"
    )
    assert (
        merge_export.resolve_base_model(adapter, qlora.DEFAULT_PROFILE_PATH, "other/model")
        == "other/model"
    )


def test_merge_export_checks_the_llama_cpp_checkout(tmp_path: Path) -> None:
    """A missing converter is reported before an hour of merging, not after."""
    import merge_export

    with pytest.raises(merge_export.ProfileError, match="missing"):
        merge_export.check_llama_cpp(tmp_path)


def test_corpus_script_dispatches_each_source_to_its_adapter(tmp_path: Path) -> None:
    """The driver is wiring only: every rule lives in the corpus builder."""
    import build_corpus

    paths = build_corpus.view_paths(cb.CorpusSource.BIGEARTHNET_V2, "patch-1", tmp_path)
    assert set(paths) == set(build_corpus.VIEW_IDS_BY_SOURCE[cb.CorpusSource.BIGEARTHNET_V2])
    assert paths["NDVI"].endswith("bigearthnet_v2/patch-1/NDVI.png")
    assert set(build_corpus.view_paths(cb.CorpusSource.VRSBENCH, "img-1", tmp_path)) == {"TC"}

    converted = build_corpus.convert(
        cb.CorpusSource.RSVQA_HR,
        {
            "image_id": "3",
            "image_path": "views/rsvqa_hr/3/TC.png",
            "question": "Is there a road?",
            "answer": "yes",
            "type": "presence",
            "split": "train",
        },
        random.Random(0),
    )
    assert len(converted) == 1
    assert converted[0].source is cb.CorpusSource.RSVQA_HR


def test_corpus_script_refuses_a_sample_whose_views_were_never_rendered() -> None:
    """Pre-rendering is a separate pass, and a dangling path is a build error."""
    import build_corpus

    sample = _sample(1, cb.CorpusSource.VRSBENCH)
    build_corpus.check_views(sample, require=False)
    with pytest.raises(cb.CorpusError, match="views missing"):
        build_corpus.check_views(sample, require=True)


def test_evidence_qa_is_generated_from_bigearthnet_fact_sheets(
    ben_row: dict[str, Any],
) -> None:
    """§4.6's supervision costs nothing because the measurements already exist."""
    import build_corpus

    bigearthnet = cb.from_bigearthnet(ben_row, rng=random.Random(0), augment=False)
    generated = build_corpus.build_evidence_source(bigearthnet, random.Random(0))
    assert generated
    assert all(sample.source is cb.CorpusSource.EVIDENCE_QA for sample in generated)
    assert any("[spectral_index_analyzer.ndvi_mean]" in s.assistant for s in generated)


class _FakeDatasets:
    """A stand-in for ``datasets``, recording what the loader asked for.

    ``load_rows`` imports ``datasets`` inside its body, so injecting a module
    here exercises the real split plumbing without a download.
    """

    def __init__(self, rows: dict[str, list[dict[str, Any]]]) -> None:
        """Hold the rows each split should yield, a call log and a read counter."""
        self.rows = rows
        self.calls: list[dict[str, Any]] = []
        self.rows_read = 0

    def load_dataset(self, name: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        """Mimic ``load_dataset``, raising the error a bad split name gives.

        Yields rather than returns, because ``streaming=True`` does: a caller
        that stops early must be able to leave rows unread, and ``rows_read``
        is how a test sees that it did.
        """
        self.calls.append({"name": name, **kwargs})
        split = kwargs["split"]
        if split not in self.rows:
            raise ValueError(f"Bad split: {split}. Available splits: {list(self.rows)}")

        def stream() -> Iterator[dict[str, Any]]:
            for row in self.rows[split]:
                self.rows_read += 1
                yield row

        return stream()


def _install_datasets(
    monkeypatch: pytest.MonkeyPatch, rows: dict[str, list[dict[str, Any]]]
) -> _FakeDatasets:
    """Put a fake ``datasets`` module in front of the loader."""
    import sys
    import types

    fake = _FakeDatasets(rows)
    module = types.ModuleType("datasets")
    module.load_dataset = fake.load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)
    return fake


REBEN_ROWS: list[dict[str, Any]] = [
    {"patch_id": "p1", "split": "train", "labels": ["Pastures"]},
    {"patch_id": "p2", "split": "validation", "labels": ["Inland waters"]},
    {"patch_id": "p3", "split": "test", "labels": ["Urban fabric"]},
    {"patch_id": "p4", "split": "train", "labels": ["Mixed forest"]},
]


def test_bigearthnet_is_loaded_as_one_bundle_and_filtered_by_its_split_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bundle is loaded whole and the split is applied by filter.

    reBEN's parquet export puts every patch under a split named ``train``, with
    the official assignment in a column. Asking for ``validation`` raises
    ``Bad split``; taking ``train`` at face value would pull the official
    validation and test rows into training. §4.1 forbids re-deriving either.
    """
    import build_corpus

    fake = _install_datasets(monkeypatch, {"train": REBEN_ROWS})

    train = list(
        build_corpus.load_rows(cb.CorpusSource.BIGEARTHNET_V2, "train", None, False)
    )
    # The builder is named, not inferred: a repo id ending in `.txt` is read as
    # a generic text dataset, which yields the README's prose lines and no split
    # column at all.
    assert fake.calls[0]["name"] == "parquet"
    # Addressed by hf:// URL: a packaged builder resolves `data_dir` against the
    # local filesystem, which is an EmptyDatasetError, not a download.
    assert (
        fake.calls[0]["data_files"]
        == "hf://datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/**/*.parquet"
    )
    assert "data_dir" not in fake.calls[0]
    assert fake.calls[0]["split"] == "train"
    assert fake.calls[0]["download_mode"] == "reuse_cache_if_exists"
    assert [row["patch_id"] for row in train] == ["p1", "p4"]

    validation = list(
        build_corpus.load_rows(cb.CorpusSource.BIGEARTHNET_V2, "validation", None, False)
    )
    # The container is still the split named "train" — only the filter changes.
    assert fake.calls[1]["split"] == "train"
    assert [row["patch_id"] for row in validation] == ["p2"]

    # The quarantined split is reachable by neither request.
    assert "p3" not in {row["patch_id"] for row in train + validation}


def test_a_bundled_source_is_read_once_for_both_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One pass, not one per split.

    BigEarthNet.txt is a single 467 MB parquet of 9.6 M rows; opening it once per
    split downloaded and decoded all of it twice to keep a different subset each
    time. The split column is right there in the row, so the routing is free.
    """
    import build_corpus

    fake = _install_datasets(monkeypatch, {"train": REBEN_ROWS})

    tagged = list(
        build_corpus.load_rows_by_split(
            cb.CorpusSource.BIGEARTHNET_V2, ("train", "validation"), None, False
        )
    )

    assert len(fake.calls) == 1
    assert [(split, row["patch_id"]) for split, row in tagged] == [
        ("train", "p1"),
        ("validation", "p2"),
        ("train", "p4"),
    ]
    # The quarantined split is reachable by neither request.
    assert "p3" not in {row["patch_id"] for _, row in tagged}


def test_a_bundled_smoke_run_stops_instead_of_scanning_to_the_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once every requested split has its rows there is nothing left to read."""
    import build_corpus

    rows = [{"patch_id": f"p{i}", "split": "train", "labels": []} for i in range(100)]
    fake = _install_datasets(monkeypatch, {"train": rows})

    tagged = list(
        build_corpus.load_rows_by_split(
            cb.CorpusSource.BIGEARTHNET_V2, ("train",), 2, False
        )
    )
    assert [row["patch_id"] for _, row in tagged] == ["p0", "p1"]
    assert fake.rows_read < len(rows)


def test_source_samples_are_streamed_not_collected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``iter_source_samples`` is lazy: nothing is read until it is iterated."""
    import argparse

    import build_corpus

    fake = _install_datasets(monkeypatch, {"train": REBEN_ROWS})
    args = argparse.Namespace(
        limit=None,
        download=False,
        views_root=Path("views"),
        require_views=False,
        sources=["bigearthnet_v2"],
    )

    stream = build_corpus.iter_source_samples(
        cb.CorpusSource.BIGEARTHNET_V2, args, random.Random(0)
    )
    assert fake.calls == []

    source, first = next(iter(stream))
    assert source is cb.CorpusSource.BIGEARTHNET_V2
    assert first.source is cb.CorpusSource.BIGEARTHNET_V2


def test_missing_views_can_be_kept_skipped_or_fatal(tmp_path: Path) -> None:
    """Three situations that were previously two, and the middle one is the useful one.

    A render pass covers a subset of the patches the annotation export mentions
    (§7.2). ``keep`` writes lines pointing at files nobody rendered; ``fail``
    refuses the whole build. Neither of those builds a corpus over the pixels
    that do exist, which is what a training run actually needs.
    """
    import build_corpus

    sample = _sample(1, cb.CorpusSource.VRSBENCH)
    assert build_corpus.missing_views(sample) == [sample.views[0].path]

    assert build_corpus.keep_sample(sample, "keep") is True
    assert build_corpus.keep_sample(sample, "skip") is False
    with pytest.raises(cb.CorpusError, match="pre-rendered views missing"):
        build_corpus.keep_sample(sample, "fail")

    # A sample whose views are on disk survives every policy.
    present = tmp_path / "TC.jpg"
    present.write_bytes(b"x")
    real = sample.model_copy(deep=True)
    real.views[0] = real.views[0].model_copy(update={"path": str(present)})
    assert build_corpus.missing_views(real) == []
    assert all(build_corpus.keep_sample(real, mode) for mode in ("keep", "skip", "fail"))


def test_require_views_is_the_old_name_for_the_fail_policy() -> None:
    """The flag the overnight script and the docs already say, kept working."""
    import argparse

    import build_corpus

    def args(require: bool, mode: str | None) -> argparse.Namespace:
        return argparse.Namespace(require_views=require, on_missing_views=mode)

    assert build_corpus.view_policy(args(False, None)) == "keep"
    assert build_corpus.view_policy(args(True, None)) == "fail"
    assert build_corpus.view_policy(args(False, "skip")) == "skip"
    assert build_corpus.view_policy(args(True, "fail")) == "fail"

    # Naming both, disagreeing, is the one combination worth refusing.
    with pytest.raises(cb.CorpusError, match="Pass one of them"):
        build_corpus.view_policy(args(True, "skip"))


def test_the_hub_glob_is_derived_from_the_one_repository_id() -> None:
    """The repo id lives in ``HF_DATASETS`` and nowhere else."""
    import build_corpus

    glob = build_corpus.hub_parquet_glob(cb.CorpusSource.BIGEARTHNET_V2)
    assert glob.startswith("hf://datasets/")
    assert build_corpus.HF_DATASETS[cb.CorpusSource.BIGEARTHNET_V2] in glob
    assert glob.endswith("/**/*.parquet")


def test_the_other_five_sources_still_ask_for_their_split_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source read from the Hub asks for its split by name, untouched."""
    import build_corpus

    rows = {"train": [{"image_id": "a"}], "validation": [{"image_id": "b"}]}
    fake = _install_datasets(monkeypatch, rows)

    assert [r["image_id"] for r in build_corpus.load_rows(
        cb.CorpusSource.DIOR_RSVG, "train", None, False
    )] == ["a"]
    # Named by repo id, split by name, and still gated by --download.
    assert fake.calls[0]["name"] == "danielz01/DIOR-RSVG"
    assert fake.calls[0]["split"] == "train"
    assert "data_dir" not in fake.calls[0]
    assert fake.calls[0]["download_mode"] == "reuse_cache_if_exists"
    assert build_corpus.hf_split(cb.CorpusSource.DIOR_RSVG, "validation") == "validation"
    assert build_corpus.hf_split(cb.CorpusSource.BIGEARTHNET_V2, "validation") == "train"


def test_the_archive_sources_never_reach_the_hub_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The regression that excluded VRSBench from the last corpus.

    ``xiang709/VRSBench`` publishes four archives and no table, so Hugging Face's
    data-file inference hands ``Images_train.zip`` — 8.4 GB of PNG — to the JSON
    packaged builder and pyarrow dies with ``ArrowInvalid: JSON parse error``.
    It dies *late*, after the annotation zip is exhausted, which is why a small
    ``--limit`` smoke run passed and the full pass did not. The fix is that these
    three sources are read from disk and ``load_dataset`` is never called for
    them at all — so the assertion is on the *absence* of a call.
    """
    import build_corpus

    fake = _install_datasets(monkeypatch, {"train": [{"image_id": "from-the-hub"}]})
    annotations = tmp_path / "vrsbench" / "Annotations_train"
    annotations.mkdir(parents=True)
    (annotations / "P0001_0001.json").write_text(
        json.dumps({"image": "P0001_0001.png", "caption": "a field", "qa_pairs": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(local_sources, "RAW_ROOT", tmp_path)

    rows = list(build_corpus.load_rows(cb.CorpusSource.VRSBENCH, "train", None, False))
    assert [row["image_id"] for row in rows] == ["P0001_0001"]
    assert rows[0]["split"] == "train"
    assert fake.calls == []
    assert {cb.CorpusSource.VRSBENCH, cb.CorpusSource.RSVQA_HR, cb.CorpusSource.CDVQA} == (
        build_corpus.LOCAL_SOURCES
    )


def test_a_bundled_source_without_a_split_column_is_a_build_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guessing here would fold the official test split into training."""
    import build_corpus

    _install_datasets(monkeypatch, {"train": [{"patch_id": "p1", "labels": []}]})
    with pytest.raises(cb.CorpusError, match="no split column"):
        list(build_corpus.load_rows(cb.CorpusSource.BIGEARTHNET_V2, "train", None, False))


def test_limit_counts_rows_kept_not_rows_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """A smoke run over a bundled source still gets the rows it asked for."""
    import build_corpus

    padded = [{"patch_id": f"x{i}", "split": "test"} for i in range(20)] + REBEN_ROWS
    _install_datasets(monkeypatch, {"train": padded})

    train = list(build_corpus.load_rows(cb.CorpusSource.BIGEARTHNET_V2, "train", 1, False))
    assert [row["patch_id"] for row in train] == ["p1"]


def test_split_column_aliases_are_normalised() -> None:
    """The column has been spelled several ways across reBEN metadata releases."""
    import build_corpus

    assert build_corpus.row_split({"split": "Train"}) == "train"
    assert build_corpus.row_split({"original_split": "val"}) == "validation"
    assert build_corpus.row_split({"set": "TESTING"}) == "test"
    assert build_corpus.row_split({"patch_id": "p"}) is None


def test_a_corpus_pointing_at_unrendered_views_is_refused_before_the_weights(
    tmp_path: Path,
) -> None:
    """The images decode lazily, so this otherwise surfaces hours into a run.

    It happened: a sanity check spent seven minutes loading a 17 GB model and
    five optimiser steps before the first eval batch hit a FileNotFoundError
    from inside PIL.
    """
    present = tmp_path / "TC.jpg"
    present.write_bytes(b"x")

    good = [{"images": [str(present)], "messages": []}]
    assert qlora.missing_images(good) == []
    qlora.guard_images_present(good, "good.jsonl")

    bad = [{"images": [str(present), str(tmp_path / "gone.png")], "messages": []}]
    assert qlora.missing_images(bad) == [str(tmp_path / "gone.png")]
    with pytest.raises(qlora.ProfileError, match="referenced view file"):
        qlora.guard_images_present(bad, "bad.jsonl")


def test_warmup_is_expressed_in_steps_because_trl_dropped_the_ratio() -> None:
    """§6.1 states warmup as a ratio; ``SFTConfig`` (trl 1.x) takes only steps."""
    profile = qlora.load_profile(qlora.DEFAULT_PROFILE_PATH)
    assert profile.train.warmup_ratio == 0.03

    assert qlora.warmup_steps(profile, 4_062) == 122
    assert qlora.warmup_steps(profile, 0) == 0
    # A short sanity run still warms up: starting a cosine schedule at the peak
    # learning rate on a 4-bit backbone undoes the initialisation on step one.
    assert qlora.warmup_steps(profile, 6) == 1


def test_trainer_arguments_exist_on_the_installed_trl() -> None:
    """Guard the whole kwarg set against trl's field renames.

    Every removed or renamed field is a ``TypeError`` raised after the weights
    are on the card and before the first batch — the most expensive place to
    find one, and the cheapest to prevent.
    """
    trl = pytest.importorskip("trl", reason="trl is the vlm-train extra")
    fields = set(trl.SFTConfig.__dataclass_fields__)

    kwargs = qlora.sft_config_kwargs(
        profile=qlora.load_profile(qlora.DEFAULT_PROFILE_PATH),
        output_dir=Path("runs/sq-lora-v1"),
        has_eval=True,
        max_steps=None,
        total_steps=4_062,
    )
    assert not set(kwargs) - fields, f"trl {trl.__version__} does not accept these"
    assert "warmup_ratio" not in kwargs
    assert kwargs["max_steps"] == -1
    assert kwargs["eval_strategy"] == "steps"
    assert kwargs["gradient_checkpointing_kwargs"] == {"use_reentrant": False}


def test_trainer_arguments_track_the_profile() -> None:
    """The mapping is the profile, restated in trl's vocabulary and nothing more."""
    profile = qlora.load_profile(qlora.FALLBACK_PROFILE_PATH)
    kwargs = qlora.sft_config_kwargs(
        profile=profile,
        output_dir=Path("runs/x"),
        has_eval=False,
        max_steps=6,
        total_steps=6,
    )
    assert kwargs["optim"] == "adamw_torch_fused"
    assert kwargs["per_device_train_batch_size"] == 1
    assert kwargs["gradient_accumulation_steps"] == 16
    assert kwargs["max_length"] == profile.data.max_seq_len
    assert kwargs["eval_strategy"] == "no"
    assert kwargs["max_steps"] == 6


def test_bigearthnet_txt_rows_become_samples_by_annotation_type() -> None:
    """The text export is question/answer rows, not the 19-class label table."""
    base = {
        "ID": 1,
        "patch_id": "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57",
        "s1_name": "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57",
        "split": "train",
        "view_paths": {v: f"views/ben2/p/{v}.png" for v in ("TC", "SARFC")},
    }
    binary = cb.from_bigearthnet_txt(
        dict(base, input="Does the image depict permanent crops?", output="no",
             type="binary", category="presence"),
        augment=False,
    )
    assert len(binary) == 1
    assert binary[0].task is TaskType.VQA
    assert binary[0].assistant == "No."
    assert binary[0].pair_type is PairType.CROSS_MODAL
    # No measurements in this export, so the prompt forbids stating any number.
    assert binary[0].fact_sheet == {}
    assert "FACT SHEET — empty" in binary[0].system

    mcq = cb.from_bigearthnet_txt(
        dict(base, ID=2, input="Which classes share a boundary? a) X, b) Y", output="b",
             type="mcq", category="adjacency"),
        augment=False,
    )
    assert mcq[0].assistant == "b"


def test_bigearthnet_txt_boxes_are_rescaled_onto_the_canonical_frame() -> None:
    """The export writes ``[x1 y1, x2 y2]`` in 0-1; we emit the 0-1000 form."""
    boxes = cb.parse_ben_txt_box("[0.64 0.0, 1.0 0.71]", label="pastures")
    assert len(boxes) == 1
    assert (boxes[0].x_min, boxes[0].y_min, boxes[0].x_max, boxes[0].y_max) == (
        640,
        0,
        1000,
        710,
    )

    sample = cb.from_bigearthnet_txt(
        {
            "ID": 3,
            "patch_id": "p1",
            "split": "train",
            "view_paths": {"TC": "views/ben2/p1/TC.png"},
            "input": "Identify the location of the <ref>largest connected region of "
            "pastures</ref>.",
            "output": "[0.0 0.33, 0.28 0.8]",
            "type": "bounding box",
            "category": "reference",
        },
        augment=False,
    )[0]
    assert sample.task is TaskType.GROUNDING
    parsed = box_format.parse(sample.assistant)
    assert parsed[0].label == "largest connected region of pastures"
    assert sample.assistant == box_format.serialise(parsed)


def test_bigearthnet_txt_skips_an_answer_it_cannot_attribute() -> None:
    """A caption quoting an unmeasured percentage is dropped, not laundered."""
    row = {
        "ID": 4,
        "patch_id": "p1",
        "split": "train",
        "view_paths": {"TC": "views/ben2/p1/TC.png"},
        "input": "Describe the scene.",
        "output": "Arable land covers 43% of this scene.",
        "type": "captioning",
        "category": "None",
    }
    assert cb.from_bigearthnet_txt(row, augment=False) == []
    with pytest.raises(cb.CitationError):
        cb.from_bigearthnet_txt(row, augment=False, skip_uncitable=False)


def test_the_bench_split_is_held_out_like_test() -> None:
    """The fourth split is held-out evaluation data.

    reBEN's own datamodule builds its test dataset from ``['test', 'bench']``.
    """
    import build_corpus

    assert build_corpus.row_split({"split": "bench"}) == "bench"
    assert build_corpus.SPLIT_ALIASES["bench"] == "bench"
    # Recognised, and then excluded: neither request keeps it.
    for wanted in ("train", "validation"):
        assert build_corpus.SPLIT_ALIASES[wanted] != "bench"


def test_an_unknown_split_value_says_so_precisely(monkeypatch: pytest.MonkeyPatch) -> None:
    """The message distinguishes a missing column from an unrecognised value.

    Conflating the two is what turned a one-line alias addition into a long hunt.
    """
    import build_corpus

    with_value = build_corpus.split_column_problem(
        cb.CorpusSource.BIGEARTHNET_V2, "train", {"split": "bench", "ID": 1}
    )
    assert "unrecognised value" in with_value
    assert "'bench'" in with_value

    without = build_corpus.split_column_problem(
        cb.CorpusSource.BIGEARTHNET_V2, "train", {"ID": 1, "input": "q"}
    )
    assert "no split column" in without
    assert "'ID'" in without


def test_unresolved_sources_are_declared_rather_than_mirrored() -> None:
    """A mirror of unknown provenance is where a quarantined image comes back."""
    import build_corpus

    # RSVQA-HR and CDVQA left this set by being *resolved* — to Zenodo record
    # 6344367 and to the ljx620/CDVQA WebDataset, both fetched by
    # scripts/fetch_sources.py — not by being pointed at a convenient mirror.
    assert set(build_corpus.UNRESOLVED_SOURCES) == {cb.CorpusSource.DIOR_RSVG}
    for source in (
        cb.CorpusSource.BIGEARTHNET_V2,
        cb.CorpusSource.VRSBENCH,
        cb.CorpusSource.RSVQA_HR,
        cb.CorpusSource.CDVQA,
    ):
        assert source in build_corpus.RESOLVED_SOURCES
    for source in build_corpus.UNRESOLVED_SOURCES:
        assert source not in build_corpus.RESOLVED_SOURCES


def test_requesting_an_unresolved_source_reports_its_actual_problem() -> None:
    """Naming one explicitly explains what is wrong instead of raising a Hub 404."""
    import argparse

    import build_corpus

    args = argparse.Namespace(
        limit=1,
        download=False,
        views_root=Path("views"),
        require_views=False,
        sources=[cb.CorpusSource.DIOR_RSVG.value],
    )
    # Eagerly, not on the first `next()`: a bad --sources fails the invocation.
    with pytest.raises(cb.CorpusError, match="has no verified source"):
        build_corpus.iter_source_samples(cb.CorpusSource.DIOR_RSVG, args, random.Random(0))


def test_an_unfetched_local_source_says_so_before_the_build_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """"Which source is missing" is far cheaper to answer now than three in."""
    import argparse

    import build_corpus

    monkeypatch.setattr(local_sources, "RAW_ROOT", tmp_path)
    args = argparse.Namespace(
        limit=1,
        download=False,
        views_root=Path("views"),
        require_views=False,
        sources=[cb.CorpusSource.CDVQA.value],
    )
    with pytest.raises(cb.CorpusError, match="has not been fetched"):
        build_corpus.iter_source_samples(cb.CorpusSource.CDVQA, args, random.Random(0))


def _evidence_qa_builder() -> Any:
    """Load the ``training/data/builders`` module by path.

    It lives outside the package on purpose — DATA_ADAPTATION_PLAN §4 puts the
    per-source builders there — so it is imported by location rather than name.
    """
    import importlib.util

    path = (
        Path(__file__).resolve().parents[2]
        / "training"
        / "data"
        / "builders"
        / "evidence_qa.py"
    )
    spec = importlib.util.spec_from_file_location("evidence_qa_builder", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evidence_qa_never_narrates_a_view_it_did_not_attach(tmp_path: Path) -> None:
    """An answer may only discuss evidence the prompt actually shows.

    The first build of this source drew each sample's views from whichever
    placeholder directory came next, so most samples carried one optical view
    beneath an answer discussing σ⁰_VV, and every bi-temporal scene carried a
    radar sheet with no radar views. A label index naming instruments the model
    was never shown teaches that the prose need not follow from the images —
    which is the failure the whole source exists to prevent.
    """
    builder = _evidence_qa_builder()
    manifest = tmp_path / "PLACEHOLDER_VIEWS.json"
    files = []
    for index in range(4):
        image = tmp_path / f"view_{index}.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")  # never opened; only the path is used
        files.append({"path": str(image)})
    manifest.write_text(json.dumps({"files": files}), encoding="utf-8")

    samples = builder.generate(count=60, manifest=manifest, seed=7)
    assert len(samples) == 60

    for sample in samples:
        views = {view.view_id for view in sample.views}
        assert len(sample.views) == 6, sample.id
        if "σ⁰_VV" in sample.assistant:
            assert any(view.startswith("SAR") for view in views), sample.id
        # Radar scalars only appear on a sheet whose scene has radar views.
        if any(key.startswith("sar_backscatter") for key in sample.fact_sheet):
            assert any(view.startswith("SAR") for view in views), sample.id


def test_evidence_qa_answers_are_citable_and_refusals_stay_a_minority(
    tmp_path: Path,
) -> None:
    """Every number resolves, and abstentions do not dominate the source."""
    builder = _evidence_qa_builder()
    manifest = tmp_path / "PLACEHOLDER_VIEWS.json"
    image = tmp_path / "TC.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    manifest.write_text(json.dumps({"files": [{"path": str(image)}]}), encoding="utf-8")

    samples = builder.generate(count=200, manifest=manifest, seed=3)
    for sample in samples:
        sheet = cb.fact_sheet_from(sample.fact_sheet)
        marked = strip_citation_markers(sample.assistant, sheet)
        assert marked.unknown_keys == [], sample.id
        assert validate(marked.text, sheet).uncited_numeric_spans == [], sample.id

    refusals = [s for s in samples if s.id.endswith(":unmeasured")]
    assert 0 < len(refusals) < len(samples) * 0.25
