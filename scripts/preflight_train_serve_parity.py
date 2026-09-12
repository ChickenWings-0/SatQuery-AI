#!/usr/bin/env python
"""Training-box pre-flight: L2 parity, G2 mask audit, G3 token budget (plan §6 step 5).

Run before every training run, on the box that will train::

    uv run python scripts/preflight_train_serve_parity.py --out runs/full-epoch-v2/preflight
        --corpus data/processed/corpus/v2/train.jsonl --val data/processed/corpus/v2/val.jsonl

No GPU and no weights: the processor is a 2 MB config, the collator is trl's,
and everything runs on CPU in a minute. It writes ``parity_report.json`` and
``mask_audit.json`` to ``--out`` and exits non-zero on any assertion, which is
the STOP condition of ML_PIPELINE_RECOVERY_PLAN §6 step 5.

What is proven, in order:

* **L2 (a)** the templated prompt string is byte-identical between the training
  record and the serving request built from the same sample;
* **L2 (b)** the ``input_ids`` are identical: the training side through trl's
  ``_collate_prompt_completion`` (the prompt width before the completion is
  appended), the serving side through ``hf_backend``'s ``apply_chat_template``;
* **L2 (c)** a 448 px view is ``max_pixels / 16² / 2²`` image tokens (144 under
  the training profile) under the training processor — and, when
  ``--adapter-dir`` is given, under ``AutoProcessor.from_pretrained(adapter_dir)``
  too, with the same layout fingerprint;
* **G3** every record of the whole corpus fits ``max_seq_len`` with its
  completion intact (text only, ~1 ms per sample);
* **G2** the real collator, on ``--audit-samples`` real records with their
  pixels, trains on exactly the tokenised completion and nothing else.

Exit codes: 0 ok · 4 corpus/token-budget · 5 parity or mask audit.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.models import hf_backend  # noqa: E402
from satquery.models.loader import GenerationRequest, PromptImage  # noqa: E402
from satquery.models.prompts import layout  # noqa: E402
from satquery.training.corpus_builder import CorpusSample, noncanonical_grounding  # noqa: E402
from satquery.training.vlm import qlora  # noqa: E402

EXIT_CORPUS = 4
EXIT_PARITY = 5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=qlora.DEFAULT_PROFILE_PATH)
    parser.add_argument("--corpus", type=Path, default=None, help="Override data.corpus.")
    parser.add_argument("--val", type=Path, default=None, help="Override data.val.")
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/preflight"))
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Also load AutoProcessor from this directory and require the same "
        "fingerprint and image-token count (L2c, second half).",
    )
    parser.add_argument("--audit-samples", type=int, default=qlora.MASK_AUDIT_SAMPLES)
    parser.add_argument(
        "--parity-samples",
        type=int,
        default=8,
        help="How many stratified samples get the full L2 string + input_ids comparison.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Defaults to the profile's.")
    return parser.parse_args(argv)


def _request_for(sample: CorpusSample, record: dict[str, Any]) -> GenerationRequest:
    """The serving request for a corpus sample, pixels read from its view files."""
    from PIL import Image

    images: list[PromptImage] = []
    for view, path in zip(sample.views[: len(record["images"])], record["images"], strict=True):
        with Image.open(path) as opened:
            images.append(PromptImage(view.label, np.asarray(opened.convert("RGB"))))
    return GenerationRequest(system=sample.system, user=sample.user, images=tuple(images))


def check_parity(
    samples: Sequence[CorpusSample],
    records: Sequence[dict[str, Any]],
    processor: Any,
    max_length: int,
) -> list[dict[str, Any]]:
    """L2 (a) and (b) on each sample; one dict per sample with any mismatch named."""
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

    collator = DataCollatorForVisionLanguageModeling(
        processor=processor, max_length=max_length, completion_only_loss=True
    )
    results: list[dict[str, Any]] = []
    for sample, record in zip(samples, records, strict=True):
        served = hf_backend.build_messages(_request_for(sample, record))
        training_text = processor.apply_chat_template(
            record["prompt"], tokenize=False, add_generation_prompt=True
        )
        serving_text = processor.apply_chat_template(
            served, tokenize=False, add_generation_prompt=True
        )
        serving_ids = processor.apply_chat_template(
            served, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )["input_ids"][0].tolist()

        example = qlora._unify_placeholders(record)
        example["images"] = qlora._open_images(record["images"])
        batch = collator([example])
        labels = batch["labels"][0]
        training_ids = batch["input_ids"][0].tolist()
        prompt_len = int((labels == -100).sum().item())

        problems: list[str] = []
        if training_text != serving_text:
            problems.append("templated prompt strings differ")
        if training_ids[:prompt_len] != serving_ids:
            problems.append(
                f"prompt input_ids differ (training {prompt_len} vs serving {len(serving_ids)})"
            )
        results.append(
            {
                "id": sample.id,
                "prompt_tokens": prompt_len,
                "views": len(record["images"]),
                "ok": not problems,
                "problems": problems,
            }
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pre-flight. Returns a process exit code."""
    args = parse_args(argv)
    profile = qlora.load_profile(args.config)
    if args.corpus is not None:
        profile.data.corpus = args.corpus
    if args.val is not None:
        profile.data.val = args.val
    seed = args.seed if args.seed is not None else profile.train.seed
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"profile      : {profile.run_name} ({profile.base_model})")
    print(f"corpus       : {profile.data.corpus}")

    # The corpus itself: composition floors and the grounding format.
    try:
        composition = qlora.guard_composition(profile.data.corpus, profile.data.expected_sources)
    except qlora.ProfileError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return EXIT_CORPUS
    print(f"composition  : {composition}")
    samples = qlora.read_jsonl(profile.data.corpus)
    bare = noncanonical_grounding(samples)
    if bare:
        print(
            f"REFUSED: {len(bare)} non-canonical GROUNDING answers, e.g. {bare[0]}",
            file=sys.stderr,
        )
        return EXIT_CORPUS
    print("grounding    : every GROUNDING answer canonical")

    max_views = profile.data.max_views_per_sample
    records = [
        qlora.sample_to_chat(s, image_root=args.image_root, max_views=max_views) for s in samples
    ]
    val_records = (
        qlora.load_corpus(
            profile.data.val,
            max_views=profile.data.max_views_per_sample,
            image_root=args.image_root,
        )
        if profile.data.val.is_file()
        else []
    )
    try:
        qlora.guard_images_present(records, str(profile.data.corpus))
        if val_records:
            qlora.guard_images_present(val_records, str(profile.data.val))
    except qlora.ProfileError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return EXIT_CORPUS

    processor = qlora.load_processor(profile)
    tokens_per_view = qlora.image_tokens_per_view(profile.data.max_pixels)
    tokenizer = getattr(processor, "tokenizer", processor)

    # L2 (c): the pixel budget under the training processor.
    text, measured = layout.fixture_render(processor)
    fingerprint = layout.layout_fingerprint(processor)
    print(
        f"pixel budget : {measured} tokens per {layout.FIXTURE_VIEW_PX} px view "
        f"(profile implies {tokens_per_view})"
    )
    if measured != tokens_per_view:
        print("REFUSED: processor and profile disagree on the image-token count", file=sys.stderr)
        return EXIT_PARITY
    adapter_check: dict[str, Any] | None = None
    if args.adapter_dir is not None:
        from transformers import AutoProcessor

        adapter_processor = AutoProcessor.from_pretrained(str(args.adapter_dir))
        _, adapter_tokens = layout.fixture_render(adapter_processor)
        adapter_fp = layout.layout_fingerprint(adapter_processor)
        adapter_check = {
            "adapter_dir": str(args.adapter_dir),
            "image_tokens_per_view": adapter_tokens,
            "fingerprint": adapter_fp,
            "ok": adapter_tokens == measured and adapter_fp == fingerprint,
        }
        print(f"adapter proc : {adapter_tokens} tokens/view · fingerprint {adapter_fp[:12]}")
        if not adapter_check["ok"]:
            print(
                "REFUSED: the adapter directory's processor differs from training",
                file=sys.stderr,
            )
            return EXIT_PARITY

    # G3: the whole corpus (train + val), text only.
    try:
        budget = qlora.guard_token_budget(
            [*records, *val_records], tokenizer, profile.data.max_seq_len, tokens_per_view,
            label=str(profile.data.corpus),
        )
    except qlora.ProfileError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return EXIT_CORPUS
    print(
        f"token budget : {budget.checked} samples · prompt {budget.prompt_min}–"
        f"{budget.prompt_max} · completion {budget.completion_min}–{budget.completion_max} "
        f"· 0 over {budget.max_seq_len}"
    )

    # L2 (a)+(b) on a stratified handful.
    chosen = qlora.stratified_subset(samples, args.parity_samples, seed=seed)
    chosen_records = [
        qlora.sample_to_chat(s, image_root=args.image_root, max_views=max_views) for s in chosen
    ]
    parity = check_parity(chosen, chosen_records, processor, profile.data.max_seq_len)
    parity_ok = all(row["ok"] for row in parity)
    print(f"L2 parity    : {sum(r['ok'] for r in parity)}/{len(parity)} byte-identical")

    # G2: the real collator on real records.
    audit = qlora.audit_masks(
        records, processor, max_length=profile.data.max_seq_len,
        tokens_per_view=tokens_per_view, n=args.audit_samples, seed=seed,
    )
    qlora.write_mask_audit(args.out / "mask_audit.json", audit)
    print(audit.summary())

    report = {
        "checked_at": datetime.now(UTC).isoformat(),
        "git_sha": qlora.git_sha(),
        "corpus": str(profile.data.corpus),
        "corpus_sha256": qlora.file_sha256(profile.data.corpus),
        "composition": composition,
        "layout_version": layout.LAYOUT_VERSION,
        "layout_fingerprint": fingerprint,
        "image_tokens_per_view": measured,
        "fixture_prompt": text,
        "adapter_processor": adapter_check,
        "token_budget": budget.model_dump(mode="json"),
        "parity": parity,
        "mask_audit": audit.model_dump(mode="json", exclude={"records"}),
        "ok": parity_ok and audit.ok,
    }
    (args.out / "parity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report       : {args.out / 'parity_report.json'}")

    if not parity_ok:
        failed = [r for r in parity if not r["ok"]]
        print(f"REFUSED: L2 parity failed on {len(failed)} sample(s): {failed[0]}", file=sys.stderr)
        return EXIT_PARITY
    try:
        qlora.guard_masks(audit)
    except qlora.ProfileError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return EXIT_PARITY
    print("preflight    : PASS")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    random.seed(0)
    raise SystemExit(main())
