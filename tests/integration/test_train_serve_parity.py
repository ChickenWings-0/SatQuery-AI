"""Train/serve parity at the token level (ML_PIPELINE_RECOVERY_PLAN §2.3, level L2).

Needs the Qwen3-VL processor — a 2 MB config, not the weights — in the local
Hugging Face cache, and the ``vlm-train`` extra (trl's collator). Skipped
cleanly anywhere else. No GPU: everything here runs on CPU in seconds.

What L2 turns from "expected" into "known":

(a) the templated prompt *string* is byte-identical between the training
    record and the serving request;
(b) the ``input_ids`` are identical: the training side through
    ``DataCollatorForVisionLanguageModeling._collate_prompt_completion`` (the
    prompt width before the completion is appended), the serving side through
    ``apply_chat_template(tokenize=True)`` — Qwen adds no BOS and both paths
    pass ``add_special_tokens=False``;
(c) a 448 px view is 144 ``<|image_pad|>`` tokens under the training
    processor, and the mask audit's four assertions hold on real records.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from satquery.models import hf_backend
from satquery.models.loader import GenerationRequest, PromptImage
from satquery.models.prompts import layout
from satquery.training import corpus_builder as cb
from satquery.training.vlm import qlora

pytestmark = pytest.mark.integration

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


@pytest.fixture(scope="module")
def processor() -> Any:
    """The training processor, from the cache only — never a download."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("trl")
    profile = qlora.load_profile(qlora.DEFAULT_PROFILE_PATH)
    try:
        return transformers.AutoProcessor.from_pretrained(
            BASE_MODEL,
            local_files_only=True,
            max_pixels=profile.data.max_pixels,
            min_pixels=profile.data.view_size_px * profile.data.view_size_px // 4,
        )
    except (OSError, ValueError) as error:  # pragma: no cover - cache-dependent
        pytest.skip(f"{BASE_MODEL} processor is not in the local HF cache: {error}")


@pytest.fixture
def rendered_views(tmp_path: Path) -> list[Path]:
    """Six 448 px views on disk, as the render pass leaves them."""
    from PIL import Image

    paths: list[Path] = []
    rng = np.random.default_rng(0)
    for index in range(6):
        path = tmp_path / f"view{index}.png"
        Image.fromarray(rng.integers(0, 255, (448, 448, 3), dtype=np.uint8)).save(path)
        paths.append(path)
    return paths


def _sample_with_views(ben_row: dict[str, Any], paths: list[Path]) -> cb.CorpusSample:
    sample = cb.from_bigearthnet(ben_row, rng=random.Random(0), augment=False)[0]
    views = [
        view.model_copy(update={"path": str(paths[i])}) for i, view in enumerate(sample.views[:6])
    ]
    return sample.model_copy(update={"views": views})


def _request(sample: cb.CorpusSample, paths: list[Path]) -> GenerationRequest:
    from PIL import Image

    return GenerationRequest(
        system=sample.system,
        user=sample.user,
        images=tuple(
            PromptImage(view.label, np.asarray(Image.open(path).convert("RGB")))
            for view, path in zip(sample.views, paths, strict=True)
        ),
    )


def test_l2a_templated_prompt_strings_are_byte_identical(
    processor: Any, ben_row: dict[str, Any], rendered_views: list[Path]
) -> None:
    sample = _sample_with_views(ben_row, rendered_views)
    record = qlora.sample_to_chat(sample, max_views=6)
    served = hf_backend.build_messages(_request(sample, rendered_views))

    training_text = processor.apply_chat_template(
        record["prompt"], tokenize=False, add_generation_prompt=True
    )
    serving_text = processor.apply_chat_template(
        served, tokenize=False, add_generation_prompt=True
    )
    assert training_text == serving_text
    # In the user turn, each label is immediately followed by its own image.
    user_turn = training_text[training_text.index("<|im_start|>user") :]
    for view in sample.views:
        after = user_turn[user_turn.index(view.label) :].removeprefix(view.label)
        assert after.startswith("<|vision_start|><|image_pad|><|vision_end|>")
    assert user_turn.count("<|image_pad|>") == 6


def test_l2b_input_ids_are_identical_and_l2c_the_pixel_budget_is_144(
    processor: Any, ben_row: dict[str, Any], rendered_views: list[Path]
) -> None:
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

    sample = _sample_with_views(ben_row, rendered_views)
    record = qlora.sample_to_chat(sample, max_views=6)
    served = hf_backend.build_messages(_request(sample, rendered_views))

    serving = processor.apply_chat_template(
        served, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    serving_ids = serving["input_ids"][0].tolist()

    collator = DataCollatorForVisionLanguageModeling(
        processor=processor, max_length=4096, completion_only_loss=True
    )
    example = qlora._unify_placeholders(record)
    example["images"] = qlora._open_images(record["images"])
    batch = collator([example])
    labels = batch["labels"][0]
    training_ids = batch["input_ids"][0].tolist()
    prompt_len = int((labels == -100).sum().item())  # no padding in a batch of one
    assert training_ids[:prompt_len] == serving_ids

    pad = layout.image_token_id(processor)
    assert serving_ids.count(pad) == 144 * 6
    assert layout.fixture_render(processor)[1] == 144


def test_g2_mask_audit_passes_on_real_records(
    processor: Any, ben_row: dict[str, Any], rendered_views: list[Path]
) -> None:
    """The four assertions of plan §1.5 G2, with the real collator."""
    samples = cb.from_bigearthnet(ben_row, rng=random.Random(0), augment=False)
    records = [
        qlora.sample_to_chat(_sample_with_views(ben_row, rendered_views), max_views=6)
        for _ in samples
    ]
    report = qlora.audit_masks(records, processor, max_length=4096, tokens_per_view=144, n=3)
    assert report.ok, [r.problems for r in report.records if not r.ok]
    assert report.image_min == report.image_max == 144 * 6
    assert 0 < report.completion_min <= report.completion_max <= qlora.MAX_COMPLETION_TOKENS
    assert report.truncated == 0
    qlora.guard_masks(report)


def test_g2_mask_audit_catches_a_messages_shaped_record(
    processor: Any, ben_row: dict[str, Any], rendered_views: list[Path]
) -> None:
    """A regression to the old shape is refused by the collator itself."""
    sample = _sample_with_views(ben_row, rendered_views)
    record = qlora.sample_to_chat(sample, max_views=6)
    regressed = {
        "messages": record["prompt"] + record["completion"],
        "images": record["images"],
    }
    with pytest.raises((qlora.ProfileError, ValueError, KeyError)):
        qlora.audit_masks([regressed], processor, max_length=4096, tokens_per_view=144, n=1)


def test_g3_token_budget_agrees_with_the_collator(
    processor: Any, ben_row: dict[str, Any], rendered_views: list[Path]
) -> None:
    """The text-only count (images by arithmetic) matches what the collator builds."""
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

    sample = _sample_with_views(ben_row, rendered_views)
    record = qlora.sample_to_chat(sample, max_views=6)
    collator = DataCollatorForVisionLanguageModeling(
        processor=processor, max_length=4096, completion_only_loss=True
    )
    example = qlora._unify_placeholders(record)
    example["images"] = qlora._open_images(record["images"])
    batch = collator([example])
    total = int(batch["attention_mask"][0].sum().item())

    tokenizer = processor.tokenizer
    prompt = qlora.prompt_token_count(record, tokenizer, tokens_per_view=144)
    completion = qlora.completion_token_count(record, tokenizer)
    assert prompt + completion == total
    report = qlora.guard_token_budget([record], tokenizer, 4096, 144)
    assert report.ok and report.prompt_max == prompt
