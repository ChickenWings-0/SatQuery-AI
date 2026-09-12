"""Train/serve prompt alignment (ML_PIPELINE_RECOVERY_PLAN §2), level L1.

The previous adapter was fitted to a user turn laid out as ``image_1 … image_n,
question`` with labels under a key the chat template never printed, and served
on ``label_1, image_1, … question``. Every test here is hermetic — no processor,
no torch — and asserts the one thing L1 can: that the training record's prompt
and the serving backend's message list are the same *structure*, built by the
same function, and that nobody builds one by hand again.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from satquery.models import hf_backend, llamacpp_client
from satquery.models.loader import BackendConfig, GenerationRequest, ModelLoadError, PromptImage
from satquery.models.prompts import layout
from satquery.training import corpus_builder as cb
from satquery.training.vlm import qlora
from tests.unit.fake_processor import FakeProcessor

SRC = Path(__file__).resolve().parents[2] / "src"
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


# ------------------------------------------------------------------ fixtures


def _sample(ben_row: dict[str, Any]) -> cb.CorpusSample:
    return cb.from_bigearthnet(ben_row, rng=random.Random(0), augment=False)[0]


def _request_for(sample: cb.CorpusSample, max_views: int = 6) -> GenerationRequest:
    """The serving request carrying the *same* text and labels as a corpus sample."""
    return GenerationRequest(
        system=sample.system,
        user=sample.user,
        images=tuple(
            PromptImage(view.label, np.zeros((8, 8, 3), np.uint8))
            for view in sample.views[:max_views]
        ),
    )


# ------------------------------------------------------------ the layout fn


def test_user_turn_interleaves_label_then_image_then_question_last() -> None:
    content = layout.user_turn_content(["L1", "L2"], "q?")
    assert content == [
        {"type": "text", "text": "L1"},
        {"type": "image"},
        {"type": "text", "text": "L2"},
        {"type": "image"},
        {"type": "text", "text": "q?"},
    ]


def test_payloads_ride_inside_the_image_block_when_given() -> None:
    content = layout.user_turn_content(["L1"], "q?", images=["PIXELS"])
    assert content[1] == {"type": "image", "image": "PIXELS"}
    with pytest.raises(ValueError, match="pair up"):
        layout.user_turn_content(["L1", "L2"], "q?", images=["only one"])


def test_a_custom_block_mapper_keeps_the_ordering() -> None:
    """llama.cpp's block type is different; its ordering must not be."""
    content = layout.user_turn_content(
        ["L1"], "q?", images=["data:..."], image_block=lambda u: {"type": "image_url", "u": u}
    )
    assert content == [
        {"type": "text", "text": "L1"},
        {"type": "image_url", "u": "data:..."},
        {"type": "text", "text": "q?"},
    ]


def test_chat_prompt_is_system_then_user() -> None:
    prompt = layout.chat_prompt("sys", ["L1"], "q?")
    assert [turn["role"] for turn in prompt] == ["system", "user"]
    assert prompt[0]["content"] == [{"type": "text", "text": "sys"}]
    assert layout.image_placeholder_count(prompt) == 1
    assert layout.image_placeholder_count(layout.chat_prompt("s", ["a"], "q", images=[1])) == 0


# --------------------------------------------------------- L1: structure


def test_l1_serving_messages_equal_the_training_prompt(ben_row: dict[str, Any]) -> None:
    """``strip_payload(hf_backend.build_messages(req)) == sample_to_chat(sample)["prompt"]``.

    Same roles, same order, same labels, same user text last — the serving path
    with its PIL payloads deleted is byte-for-byte the training record's prompt.
    """
    sample = _sample(ben_row)
    request = _request_for(sample)
    served = layout.strip_image_payloads(hf_backend.build_messages(request))
    trained = qlora.sample_to_chat(sample, max_views=6)["prompt"]
    assert served == trained
    assert json.dumps(served, sort_keys=True) == json.dumps(trained, sort_keys=True)


def test_l1_the_offline_path_orders_its_blocks_identically(ben_row: dict[str, Any]) -> None:
    """llama.cpp carries a data URI instead of PIL, in exactly the same slots."""
    sample = _sample(ben_row)
    request = _request_for(sample)
    gguf = llamacpp_client.build_messages(request)[1]["content"]
    trained = qlora.sample_to_chat(sample, max_views=6)["prompt"][1]["content"]
    assert len(gguf) == len(trained)
    for served_block, trained_block in zip(gguf, trained, strict=True):
        if trained_block["type"] == "image":
            assert served_block["type"] == "image_url"
            assert served_block["image_url"]["url"].startswith("data:image/jpeg;base64,")
        else:
            assert served_block == trained_block


def test_no_hand_built_user_turns() -> None:
    """``"type": "image"`` literals live in layout.py and nowhere else.

    The skew happened because two callers each assembled a content list by
    hand and drifted. One of them is now the only place allowed to.
    """
    pattern = re.compile(r'["\']type["\']\s*:\s*["\']image["\']')
    offenders: list[str] = []
    for root in (SRC, SCRIPTS):
        for path in root.rglob("*.py"):
            if path.name == "layout.py":
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(SRC.parent)}:{number}")
    assert offenders == [], f"hand-built image blocks: {offenders}"


# ---------------------------------------------------------- the fingerprint


def test_fingerprint_is_deterministic_and_moves_with_the_pixel_budget() -> None:
    a = layout.layout_fingerprint(FakeProcessor(tokens_per_view=144))
    b = layout.layout_fingerprint(FakeProcessor(tokens_per_view=144))
    assert a == b
    assert len(a) == 64 and all(c in "0123456789abcdef" for c in a)
    # 196 tokens per view is the base repo's default — fact 8 — and must differ.
    assert layout.layout_fingerprint(FakeProcessor(tokens_per_view=196)) != a
    # A changed chat template changes the rendered text, and therefore the hash.
    assert layout.layout_fingerprint(FakeProcessor(template_tag="v2")) != a
    text, per_view = layout.fixture_render(FakeProcessor(tokens_per_view=144))
    assert per_view == 144
    assert text.count("<|image_pad|>") == len(layout.FIXTURE_LABELS)
    assert text.index(layout.FIXTURE_LABELS[0]) < text.index("<|image_pad|>")


def test_fingerprint_covers_the_layout_version() -> None:
    text, per_view = layout.fixture_render(FakeProcessor())
    digest = hashlib.sha256()
    digest.update(layout.LAYOUT_VERSION.encode())
    digest.update(b"\0" + text.encode() + b"\0" + str(per_view).encode())
    assert layout.layout_fingerprint(FakeProcessor()) == digest.hexdigest()


# -------------------------------------------------------- L3: at model load


def _adapter(tmp_path: Path, record: dict[str, Any] | None) -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    if record is not None:
        (adapter / hf_backend.LAYOUT_FILE).write_text(json.dumps(record), encoding="utf-8")
    return adapter


def test_l3_a_matching_layout_record_passes(tmp_path: Path) -> None:
    processor = FakeProcessor(tokens_per_view=144)
    record = qlora.layout_record(processor, corpus_path=None, base_model="m")
    adapter = _adapter(tmp_path, record)
    assert hf_backend.check_layout(adapter, processor)["image_tokens_per_view"] == 144


def test_l3_a_different_pixel_budget_is_refused(tmp_path: Path) -> None:
    """The base repo's processor (196 tokens/view) must not serve a 144-token adapter."""
    adapter = _adapter(tmp_path, qlora.layout_record(FakeProcessor(144)))
    with pytest.raises(ModelLoadError, match="fingerprint"):
        hf_backend.check_layout(adapter, FakeProcessor(196))


def test_l3_a_different_layout_version_is_refused(tmp_path: Path) -> None:
    record = qlora.layout_record(FakeProcessor())
    record["layout_version"] = "images-then-question/v0"
    adapter = _adapter(tmp_path, record)
    with pytest.raises(ModelLoadError, match="layout"):
        hf_backend.check_layout(adapter, FakeProcessor())


def test_l3_an_adapter_without_a_layout_record_is_refused(tmp_path: Path) -> None:
    """runs/full-epoch-v1 has no layout.json — and was trained on the wrong layout."""
    adapter = _adapter(tmp_path, None)
    with pytest.raises(ModelLoadError, match="layout.json"):
        hf_backend.check_layout(adapter, FakeProcessor())


def test_the_processor_must_come_from_the_adapter_directory(tmp_path: Path) -> None:
    """No fallback to the base repo's processor: 196 vs 144 tokens, silently (plan §2.4)."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    backend = hf_backend.HuggingFaceBackend(BackendConfig(adapter_path=adapter))
    with pytest.raises(ModelLoadError, match="processor_config.json"):
        _ = backend.processor_source

    (adapter / "processor_config.json").write_text("{}", encoding="utf-8")
    assert backend.processor_source == str(adapter)
    plain = hf_backend.HuggingFaceBackend(BackendConfig())
    assert plain.processor_source == BackendConfig().model_id


def test_layout_record_carries_what_the_manifest_needs(tmp_path: Path) -> None:
    corpus = tmp_path / "train.jsonl"
    corpus.write_bytes(b"{}\n")
    record = qlora.layout_record(FakeProcessor(144), corpus_path=corpus, base_model="Qwen/x")
    assert record["layout_version"] == layout.LAYOUT_VERSION
    assert record["image_tokens_per_view"] == 144
    assert record["corpus_sha256"] == hashlib.sha256(b"{}\n").hexdigest()
    assert record["base_model"] == "Qwen/x"
    written = qlora.write_layout_record(tmp_path / "out", record)
    assert written.name == hf_backend.LAYOUT_FILE
    assert json.loads(written.read_text())["layout_fingerprint"] == record["layout_fingerprint"]
