# ML Pipeline Recovery Plan — Training Run v2

> **Status (2026-09-29): executed — design record.** Steps 0–5 landed in `0bada6e`; the
> run it describes produced `runs/sq-lora-v2-full` (answer-token accuracy 26.4 % →
> 81.8 %, see `DOCS/AI_HANDOFF/06_DATA_AND_TRAINING.md`). `project_audit.md`, cited
> below, was deleted on 2026-09-29 (git history). Read this before any retrain.

**Original status:** blueprint, awaiting review · **Date:** 2026-09-12 · **Scope:** the Qwen3-VL-8B QLoRA path (`src/satquery/training/vlm/qlora.py`, `scripts/train_vlm.py`, `src/satquery/models/hf_backend.py`, `scripts/build_corpus.py`) · **Stack this plan was verified against:** trl 1.12.0 · transformers 5.16.1 · peft 0.20.0 · bitsandbytes 0.50.2 · torch 2.9.1+rocm6.4 · datasets 5.0.1 (the versions in `.venv` today; every line number below refers to them).

The previous run (`runs/full-epoch-v1`, 19.2 h, 1,159 steps) produced an adapter that cannot be shown to have learned anything: eval loss 3.935 → 3.930, eval token accuracy 0.4867 → 0.4879, flat from step ~90. This document is the plan for a run whose result can be defended. Nothing in it is a script yet; it is the blueprint the scripts will be written from, and every mechanism it relies on was checked against the installed library source rather than the documentation.

---

## 0. What this plan is built on (verified, not assumed)

Before the four fixes, the facts the fixes depend on. Each was checked in this repository or in `.venv` today.

| # | The brief / docs say | What is actually true on this box | Evidence |
|---|---|---|---|
| 1 | "Implement `DataCollatorForCompletionOnlyLM`" | **It does not exist in trl 1.12.0.** Removed in 0.20; zero hits in `site-packages/trl`. | `grep -rn DataCollatorForCompletionOnlyLM .venv/…/trl` → nothing |
| 2 | "or `assistant_only_loss=True` with `{% generation %}` markers" (also `project_audit.md §2`, `AI_HANDOFF/08 §1`) | **Rejected for vision datasets.** Any dataset with an `images` key → `ValueError("Assistant-only loss is not yet supported for vision datasets")`. | `trl/trainer/sft_trainer.py:1052-1056` |
| 3 | — | **The mechanism that exists:** a *prompt-completion* dataset (`prompt`, `completion`, `images` keys) routed through `DataCollatorForVisionLanguageModeling._collate_prompt_completion`, which tokenises prompt and completion separately, concatenates, and sets `labels[completion_mask == 0] = -100` when `completion_only_loss=True`. | `sft_trainer.py:692-785`; the mask at `771-774`; collator selection at `1221-1227` |
| 4 | — | The completion string is `apply_chat_template(prompt + completion)[len(prompt):]` = `"{answer}<|im_end|>\n"`. So the loss covers the answer **and the end-of-turn token**. The prompt string is `apply_chat_template(prompt, add_generation_prompt=True)` — the exact call `hf_backend._generate` makes. | `trl/data_utils.py::apply_chat_template` |
| 5 | "loss flatlined at ~3.8" | Train 13.01 (step 10) → 3.84 (110) → 3.76 (1150). Eval 3.935/3.932/3.931/3.930/3.930 at 250/500/750/1000/1159. grad_norm ≈ 0.05 throughout. | `runs/full-epoch-v1/checkpoint-1159/trainer_state.json` |
| 6 | "1,000-token system prompts" | System turn mean **2,583 chars** (4,186 for `evidence_qa`); assistant turn mean **27 chars**. 7,589 of 18,530 answers (41 %) are ≤ 3 characters (`d`, `Yes`). Roughly 1 % of supervised tokens were the target. | corpus scan of `full.train.jsonl` |
| 7 | "prompt skew" | Training emits `{"type": "image", "label": …}`; the Qwen3-VL template renders that block as `<|vision_start|><|image_pad|><|vision_end|>` and **never prints `label`**. Serving emits `{"type":"text","text": label}` immediately before each image. Training never saw a label adjacent to pixels. | `chat_template.json` (user branch); `qlora.py:654-656` vs `hf_backend.py:109-113` |
| 8 | (not in brief) pixel budget | The `max_pixels=147456` override **did** persist to the adapter's `processor_config.json` (`size.longest_edge=147456`): a 448 px view is **144** image tokens under both the training processor and the adapter-dir processor; the base repo default is **196**. No skew today — but only because `processor_source` prefers the adapter dir. | measured with `AutoProcessor` on CPU |
| 9 | "silently dropped VRSBench, CDVQA, RSVQA-HR" | `full.train.jsonl` = 15,530 `bigearthnet_v2` + 3,000 `evidence_qa`. VRSBench: 29,615 views rendered, 0 in corpus. CDVQA: 271 pairs rendered (all that were fetched), 0 in corpus. RSVQA-HR: `Images.tar` fetched, **not extracted**, 0 views rendered. The Phase-8 builder now fails fast on an un-fetched source; nothing asserts the *composition* of what it wrote. | corpus scan; `data/processed/views/*`; `data/raw/rsvqa_hr/` |
| 10 | — | trl's `mean_token_accuracy` is computed over `shift_labels != -100`. Once the prompt is masked it *is* assistant-token accuracy, for free. | `sft_trainer.py:1830-1846` |
| 11 | — | A sample whose completion is truncated away entirely yields a finite **zero** loss (denominator clamped), not NaN. It is silently wasted. | `sft_trainer.py:225` |
| 12 | — | Throughput 0.265 samples/s (three manifests agree) → one 16-sample optimiser step ≈ 60 s → 200 steps ≈ 3.4 h. Masking shrinks only the `lm_head` matmul (`chunked_nll` runs it on unmasked positions only); expect ≤ 10 % faster and budget on 0 %. | `runs/*/run_manifest.json` |
| 13 | (not in brief) | 2,655 of 5,294 GROUNDING targets are bare `<|box_start|>…<|box_end|>`; 2,639 are `<|object_ref_start|>NAME<|object_ref_end|><|box_start|>…`. The system prompt's TASK block specifies only the second. Half the grounding supervision contradicts the instruction it is trained under. | corpus scan; `corpus_builder.py:854` |

**Consequence for the brief:** items 1 and 2 of the brief's masking request are dead ends on this stack. The plan below uses item 3, which is mathematically the same objective for our single-turn `system → user → assistant` corpus (the set of unmasked tokens is identical: the assistant's text plus its end-of-turn token). The audit and handoff documents that recommend `assistant_only_loss` must be updated to point here, or the next engineer will lose a day rediscovering the `ValueError`.

---

## 1. The masking fix

### 1.1 Why the run flatlined

With `messages`-shaped records and no masking flag, `_collate_language_modeling` sets `labels = input_ids` and masks only padding (`sft_trainer.py:684-686`). Every one of ~1,400–2,200 prompt tokens per sample was a training target: the rules block (deterministic, learned by step ~90 — that is the 13.0 → 3.8 drop), the FactSheet floats (`0.751899`, unpredictable → a permanent loss floor), 720–864 `<|image_pad|>` positions, and the labels. The 3–100 tokens we cared about contributed ~1 % of the gradient. A grad_norm of 0.05 at lr 1e-4 is what "nothing left to learn from the digits, nothing loud enough from the answers" looks like.

### 1.2 The record shape (exact)

`sample_to_chat()` (`qlora.py:626`) changes from `messages` to `prompt` / `completion`. The user turn's layout comes from the shared function of §2; shown inline here for completeness.

```python
{
    "prompt": [
        {"role": "system", "content": [{"type": "text", "text": sample.system}]},
        {"role": "user", "content": [
            {"type": "text", "text": views[0].label}, {"type": "image"},
            {"type": "text", "text": views[1].label}, {"type": "image"},
            # … one (label, placeholder) pair per view, slot order …
            {"type": "text", "text": sample.user},
        ]},
    ],
    "completion": [
        {"role": "assistant", "content": [{"type": "text", "text": sample.assistant}]},
    ],
    "images": [str(image_root / v.path) …],   # cast to datasets.Image() → PIL at collate, as today
}
```

Two contracts carried over from the current code, both still load-bearing:

- **`images` is a top-level key.** It is the only thing that makes `SFTTrainer` treat the dataset as vision (`sft_trainer.py:1036`) and select the vision collator.
- **Image blocks are bare placeholders** (`{"type": "image"}`, no `image` key). `prepare_multimodal_messages` counts exactly those and refuses the example unless the count equals `len(images)` (`data_utils.py:99-106`). The `label` key is dropped: it did nothing.

One new subtlety: `datasets.Dataset.from_list` unifies the content structs, so an image placeholder round-trips as `{"type": "image", "text": None}`. That is safe — the template tests `content.type == 'image'` before `'text' in content`, and `prepare_multimodal_messages` tests `"image" not in part` — but the placeholder-count helper (`image_placeholder_count`, `qlora.py:673`) and its test (`test_chat_records_satisfy_trls_placeholder_contract`) must be re-pointed at `record["prompt"]`.

### 1.3 What is masked, what is trained (token-level)

For one 5-view BigEarthNet sample the collated sequence is:

```
<|im_start|>system\n{system ≈ 700–1,100 tok}<|im_end|>\n
<|im_start|>user\n
  Image 1 (optical true colour, Sentinel-2)<|vision_start|>{<|image_pad|> × 144}<|vision_end|>
  Image 2 (…)<|vision_start|>{× 144}<|vision_end|>  … ×5 …
  {user question}<|im_end|>\n
<|im_start|>assistant\n                                   ← end of PROMPT: every token above → label −100
{answer}<|im_end|>\n                                     ← COMPLETION: every token → trained
{right padding}                                          ← −100
```

- Trained: the answer tokens, `<|im_end|>` (id 151645, the EOS `generate()` stops on), and the trailing `\n`. Training the EOS is deliberate — the runaway-decode failure in `loader.py` (`CHAT_STOP_STRINGS` docstring) is exactly what an untrained stop token produces.
- Masked: system, labels, all image tokens, the question, the `<|im_start|>assistant\n` header.
- The box sentinels `<|object_ref_start|>` / `<|box_start|>` … are special tokens (ids 151646–151649) and live *inside* the completion, so they are trained. Nothing in the collator strips them.

### 1.4 Configuration changes (`sft_config_kwargs`, `qlora.py:917`)

```python
"completion_only_loss": True,      # EXPLICIT True, never None. None auto-detects from the record
                                   # shape; a regression to `messages` would silently revert to
                                   # full-sequence loss — the exact failure being fixed.
"assistant_only_loss": False,      # explicit + commented: trl 1.12.0 raises for vision datasets
                                   # (sft_trainer.py:1052). Prevents a well-meant "fix".
"pad_to_multiple_of": None,        # the vision prompt-completion collator raises
                                   # NotImplementedError if set (sft_trainer.py:693-697).
"truncation_mode": "keep_start",   # the default, stated: keep_end is rejected for vision datasets.
"max_length": profile.data.max_seq_len,   # unchanged (4096); see the token-budget audit below.
# packing / padding_free stay unset: both are rejected for vision datasets (1043-1051).
```

`loss_type` stays at its default `chunked_nll`: the `lm_head` projection runs only on unmasked positions, which is now 3–100 tokens instead of ~2,000.

### 1.5 Guards that make silent regression impossible

**G1 — trainer self-check** (`build_trainer`, after `SFTTrainer(...)` returns). Assert `trainer._is_vision_dataset is True`, `trainer.completion_only_loss is True`, and `isinstance(trainer.data_collator, DataCollatorForVisionLanguageModeling)`; otherwise raise `ProfileError`. These are private attributes on a pinned version; the pin (`trl==1.12.0` in `pyproject.toml`'s `vlm-train` extra) is part of the fix.

**G2 — mask audit, before the weights load** (new `audit_masks(records, processor, n=32)` in `qlora.py`; `load_model_and_processor` is split so the processor can be built first — it is a 2 MB config, not the 16 GB checkpoint). For each of 32 stratified records, run the real collator (`DataCollatorForVisionLanguageModeling(processor, max_length, completion_only_loss=True)`) on the record with its PIL images and assert:

1. `input_ids[labels != -100].tolist() == tokenizer(sample.assistant + "<|im_end|>\n", add_special_tokens=False).input_ids` — the unmasked tokens are exactly the completion, tokenised the way the collator tokenises it (same call, same `add_special_tokens=False`).
2. Every `<|image_pad|>` position has label −100, and the first unmasked position is after the last `<|image_pad|>`.
3. `count(<|image_pad|>) == 144 × len(views)` — the pixel budget the adapter will be served under (§2.5).
4. `0 < unmasked tokens ≤ 512` and no truncation (`len(input_ids) < max_length`).

Output: one log line (`mask audit: 32/32 ok · prompt 1,412–2,208 tok · completion 3–96 tok · image 720–864 tok · 0 truncated`) and `mask_audit.json` in the run directory. Any failure → exit 5 before `from_pretrained` is called. This is the check that would have caught the 19-hour run in 30 seconds.

**G3 — token-budget audit, whole corpus** (`guard_token_budget` in `load_corpus`). Truncation is applied *after* concatenation and from the right (`sft_trainer.py:761-768`), so an over-long sample loses its completion first — 100 % masked, finite zero loss (fact 11), one wasted forward/backward. Tokenise every prompt's text (no pixels: image tokens are `144 × n_views` by arithmetic) and refuse to start if any `prompt_tokens + completion_tokens + 8 > max_seq_len`, listing the offenders. Cost: ~1 ms per sample, text only. The `evidence_qa` samples are the longest today (~2,100 tokens) and fit with room; the audit exists for the corpus we have not built yet.

**G4 — unit tests** (hermetic, no torch):
- `test_records_are_prompt_completion_shaped` — keys are exactly `{prompt, completion, images}`; prompt roles `[system, user]`; completion `[assistant]`.
- `test_completion_holds_only_the_assistant_text` — the completion carries no image placeholders, no system text.
- `test_placeholder_count_matches_images` — updated helper over `record["prompt"]`.
- `test_sft_config_pins_completion_only_loss` — `sft_config_kwargs()["completion_only_loss"] is True` and `["assistant_only_loss"] is False`; a future edit that drops either fails CI.

### 1.6 What the logged metrics mean after the fix

- `loss` / `eval_loss` — mean NLL over completion tokens only. Not comparable to the 3.8 of the old run; a zero-shot baseline must be *measured* (§3.4).
- `mean_token_accuracy` / `eval_mean_token_accuracy` — assistant-token accuracy (fact 10). Inflated on short answers by the two structural tokens (§3.6); a curve to watch, not a verdict.
- `grad_norm` — expected to be an order of magnitude larger than 0.05 early on; the gradient now comes from tokens the model is wrong about.

---

## 2. Prompt alignment — train and serve byte-identical

### 2.1 The skew, precisely

Same system text (both paths call `build_system_prompt` — `corpus_builder.assemble` docstring and `prompts/builder.py`), same user text (`build_user_prompt`), same labels (`render/view_labels`). The divergence is only in the **layout of the user turn**: serving interleaves `label_i, image_i`; training emitted `image_1 … image_n, user` with the labels attempted via a key the template ignores. The adapter was fitted to an unlabelled-image layout and is served on a labelled one; both `hf_backend.build_messages`' docstring and DATA_ADAPTATION_PLAN §2.4 say the unlabelled layout is the unreliable one.

### 2.2 One layout function, two callers

New module `src/satquery/models/prompts/layout.py` — pure Python, no torch, importable by the training package and the serving package:

```python
LAYOUT_VERSION: Final[str] = "label-before-image/v1"

def user_turn_content(labels: Sequence[str], user_text: str,
                      images: Sequence[Any] | None = None) -> list[dict[str, Any]]:
    """[label_1, image_1, …, label_n, image_n, user_text].
    images=None  → bare {"type": "image"} placeholders  (trl's contract, training)
    images given → {"type": "image", "image": payload}   (transformers' contract, serving)
    len(images) must equal len(labels) when given."""

def chat_prompt(system: str, labels, user_text, images=None) -> list[dict[str, Any]]:
    """[system turn, user turn] — the prompt half of a prompt-completion record and,
    identically, the message list hf_backend hands to apply_chat_template."""

def layout_fingerprint(processor) -> str:
    """sha256 over apply_chat_template(chat_prompt(FIXTURE), tokenize=False,
    add_generation_prompt=True) + the image-token count of a 448² fixture image.
    A fixed fixture, so it is deterministic and costs nothing."""
```

- `hf_backend.build_messages(request)` becomes one line: `chat_prompt(request.system, [i.label for i in request.images], request.user, images=[_to_pil(i) for i in request.images])`.
- `qlora.sample_to_chat(sample)` builds `"prompt": chat_prompt(sample.system, [v.label for v in views], sample.user)`.
- `llamacpp_client.build_messages` keeps its `image_url` data-URI block type but derives **ordering** from `user_turn_content` through a payload mapper, so the existing test `test_every_view_is_labelled_immediately_before_its_own_pixels` (`tests/unit/test_models.py:363`) continues to cover both serving paths.

Neither caller is allowed to construct a content list by hand again; a grep-based test (`test_no_hand_built_user_turns`) asserts `"type": "image"` literals appear only in `layout.py`.

### 2.3 Three-level parity proof

| Level | Where it runs | What it asserts |
|---|---|---|
| **L1 — structure** (hermetic unit test, CI) | `tests/unit/test_prompt_layout.py` | For a fixture request/sample with identical text: `strip_payload(hf_backend.build_messages(req)) == qlora.sample_to_chat(sample)["prompt"]`, where `strip_payload` deletes `image` keys. Roles, order, labels, user text last — all equal. |
| **L2 — tokens** (integration test, skipped unless the processor is in the HF cache; also `scripts/preflight_train_serve_parity.py`, run on the training box before every run) | `tests/integration/test_train_serve_parity.py` | (a) `processor.apply_chat_template(…, tokenize=False, add_generation_prompt=True)` strings are **byte-identical** for the two paths. (b) `input_ids` are identical: the training side via `DataCollatorForVisionLanguageModeling._collate_prompt_completion` on the record (take `input_ids[:prompt_len]` where `prompt_len` = the collator's prompt width before concatenation), the serving side via `hf_backend`'s `apply_chat_template(tokenize=True, return_dict=True)`. Qwen adds no BOS (`add_bos_token: false`) and both paths pass `add_special_tokens=False`, so this is expected to hold exactly — the test is what turns "expected" into "known". (c) 144 `<|image_pad|>` tokens per 448 px view under the training processor **and** under `AutoProcessor.from_pretrained(adapter_dir)`. |
| **L3 — runtime** (every model load in production) | `hf_backend._load` | The training run writes `layout.json` (`LAYOUT_VERSION`, `layout_fingerprint`, git sha, corpus sha256) beside the adapter and into `run_manifest.json`. `_load` recomputes `layout_fingerprint(processor)` with the processor it actually loaded and raises `ModelLoadError` on mismatch — same philosophy as the existing hard failure on a missing `adapter_config.json` (`hf_backend.py:223-227`): serving a layout the adapter never saw must look like an error, not like a bad answer. |

### 2.4 Pixel-budget parity (144 vs 196)

Verified today (fact 8): `processor_source` prefers the adapter dir and the persisted `size.longest_edge=147456` gives 144 tokens/view, matching training. Two things make this durable rather than lucky:

1. The fingerprint (§2.2) includes the fixture image's token count, so a base-repo processor (196 tokens) fails L3 loudly.
2. `hf_backend.processor_source` gains a warning-to-error: if an adapter is configured and the adapter dir has **no** `processor_config.json`, refuse to load rather than fall back to the base repo's processor.

### 2.5 Answer-side alignment: the grounding target format (fact 13)

Half the GROUNDING targets violate the format the system prompt dictates. Fix in `_ben_txt_answer` (`corpus_builder.py:854`): every box is emitted through the one serialiser in `prompts/box_format.py` in the tagged form `<|object_ref_start|>NAME<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>`. Where the reBEN row carries the class (`<ref>…</ref>` — `_BEN_TXT_REF`) use it; for the `<point>(…)</point>` questions with no class in the row, resolve NAME from the patch's label set (`row["labels"]` is available to `from_bigearthnet_txt`) or exclude the row from GROUNDING — never emit the bare form. Corpus-level test: every GROUNDING assistant turn matches the canonical regex or is exactly `NONE`. At serving this keeps `box_format.parse` on its structural path instead of the bare-digit fallback `hf_backend.py:314-321` describes.

---

## 3. The 200-step probe

### 3.1 What it must prove

Two different questions, two different runs, both cheap next to 19 hours:

- **A. Can the pipeline learn at all?** If gradients reach the answer tokens through the mask, a small fixed subset must be memorised: loss → ~0, accuracy → ~1, greedy generation reproduces the targets. Failure here is a plumbing bug, and the run is short enough to iterate on.
- **B. Does the real recipe descend on held-out data?** The first 200 optimiser steps of the actual run, with the actual schedule, evaluated on data the model never sees, and — crucially — with a zero-shot baseline measured at step 0 by the same code.

Stage B **is** the first 200 steps of the full run: same output dir, same profile, `total_steps = plan.steps` so warm-up and cosine are laid out for the full epoch. If it passes, the full run continues with `--resume auto` from `checkpoint-200`. No compute is thrown away.

### 3.2 Stage A — memorisation probe (≈ 1 h)

| Item | Value |
|---|---|
| Subset | 64 samples, fixed seed, ids recorded: 16 short-answer VQA (≤ 3 chars), 16 prose VQA, 16 GROUNDING, 16 `evidence_qa` CROSS_MODAL_VQA. Written to `data/processed/corpus/probe_memorise.jsonl`; used as **both** train and eval (in-sample by design). |
| Overrides | `--max-steps 200`, `gradient_accumulation_steps 4` (effective batch 4 → 800 sample-passes = 12.5 epochs), `logging_steps 1`, `eval_steps 25`, `save_steps 100`, lr 1e-4. Per-device batch stays 1, so VRAM is unchanged. |
| Cost | 800 / 0.265 s⁻¹ ≈ 50 min + 8 evals × 64 samples (~1.3 min each) ≈ **1 h**. |
| Then | Greedy-decode all 64 with `checkpoint-200` through `HuggingFaceBackend` (the serving path, not the trainer): exact match for VQA/CROSS_MODAL after `apply_stop` + strip; IoU vs the target box for GROUNDING. |

### 3.3 Stage B — descent probe (≈ 4 h)

| Item | Value |
|---|---|
| Data | The rebuilt full-mix corpus (§5), real profile A: 1×16, lr 1e-4, cosine, warm-up 3 % of the **full** planned steps. |
| Held-out set | 128 samples from `val.jsonl`, stratified by source × task, ids recorded, asserted `∉ train` by id and by image sha256. Reused unchanged as the full run's eval set and as the ablation dry-run set. |
| Overrides | `--max-steps 200`, `eval_steps 50` (evals at 50/100/150/200), `--eval-first` (§3.4), `save_steps 50`. |
| Cost | 3,200 / 0.265 ≈ 3.4 h + 5 evals ≈ 15 min + two generation passes over 128 (step 0 and step 200) ≈ 30 min → **≈ 4 h**. |

### 3.4 The zero-shot masked baseline (new `--eval-first` flag)

`trainer.evaluate()` is called **before** `trainer.train()` and its `eval_loss` / `eval_mean_token_accuracy` are recorded in `run_manifest.json` as `zero_shot_eval`. The old run never had this number; without it "the loss went down" has no reference point. It also anchors the ablation: the step-0 generation pass in Stage B is row Z1 of §4 on the 128-item set.

### 3.5 Pass / fail gates

All gates are evaluated by `scripts/probe_report.py` from `trainer_state.json` plus the generation outputs; it writes `probe_report.json` and exits non-zero on any failure. `run_overnight.sh` will not start the full run on a non-zero exit.

| Gate | Stage A (memorise) | Stage B (descend) | Rationale |
|---|---|---|---|
| Masked train loss | < 0.10 at step 200; < 0.5 by step 60 | smoothed (window 5) loss at step 200 < at step 50, and < 0.8 × step-10 value | A proves gradient flow; B proves the recipe moves on unseen data. |
| Eval masked loss | < 0.15 (in-sample) | step-200 ≤ 0.75 × zero-shot (`--eval-first`) | The number the old run could not report. |
| Token accuracy (trl) | > 0.97 | ≥ zero-shot + 0.10 absolute | Curve only; inflation caveat §3.6. |
| Generation EM (VQA + CROSS_MODAL) | ≥ 44 / 48 | step-200 > step-0 by ≥ 5 pp | Cannot be inflated by structural tokens. |
| Generation IoU (GROUNDING) | ≥ 0.9 on ≥ 14 / 16 | mean IoU at 200 > at 0 | The format is learnable and parsed by the real parser. |
| Format compliance | 64 / 64 parse (letter / prose / canonical box / NONE) | ≥ 95 % at step 200 | The §2.5 fix took. |
| Stability | no NaN/inf; grad_norm > 0.2 at step 10 | no NaN/inf; peak VRAM ≤ 21.5 GiB; throughput ≥ 0.22 samples/s | Old run's 0.05 grad_norm was the symptom. |

Diagnosis map when a gate fails: **A-loss flat** → mask or adapter wiring (re-run G2; check `trainable_parameter_report`); **A-loss falls, EM low** → decode/stop/parse path or a parity break (L2 would have caught the latter); **B-eval flat while A passed** → data problem (leakage-free but uninformative labels, format inconsistency) or lr; **token acc up, EM flat** → §3.6.

### 3.6 Reading token accuracy honestly

41 % of samples have ≤ 3-character answers, so their completion is 1–2 answer tokens + `<|im_end|>` + `\n`. Once the model has learned the two structural tokens (it will, within an epoch of the subset), those samples sit at 50–67 % accuracy with the *answer* still wrong. Therefore: (1) `probe_report.py` also reports `answer_token_accuracy` — accuracy over completion positions excluding the final two — computed from a `compute_loss` override in a thin `SatQuerySFTTrainer` subclass that re-uses trl's own masked-accuracy code with an extra exclusion; (2) the verdict metric is generation exact-match, which no structural token can inflate.

### 3.7 The probe report artefact

`runs/probe-A-memorise/probe_report.json` and `runs/probe-B-descent/probe_report.json`:

```
stage, git_sha, corpus_sha256, subset_ids_sha256, steps,
zero_shot_eval{loss, token_acc, answer_token_acc},
curve[{step, loss, token_acc, answer_token_acc, grad_norm, lr}],
final_eval{…}, generation{em_vqa, em_cross_modal, iou_grounding_mean, format_compliance, truncation_rate},
gates[{name, threshold, value, pass}], verdict: PASS | FAIL, wall_clock_s, peak_vram_gib
```

Stage A doubles as the regression test the handoff asked for: any change to `qlora.py`, `layout.py` or the collator path re-runs it before the next long run.

---

## 4. The ablation

### 4.1 What "clean" means here

Everything held equal except the adapter: same base weights (bf16, sdpa — the production path; NF4 generation is broken on gfx1100 per `hf_backend.py:18-27`, so "adapter served under NF4" is not a row), same processor (from the adapter dir, 144 tokens/view), same prompts (built by `build_system_prompt` / `build_user_prompt` / `layout.chat_prompt` with the FactSheet from the same deterministic tools), same decoding (greedy, `seed=0`, `max_new_tokens=384`, `CHAT_STOP_STRINGS`), same parsers (`box_format.parse`, `CitationValidator` strict), same items in the same order, one harness, one command. The harness drives `HuggingFaceBackend` — the code that serves — so the table measures what production does, not a test-only path.

### 4.2 Rows

| Row | Configuration | Purpose |
|---|---|---|
| **Z0** | Qwen3-VL-8B-Instruct, no adapter, base-repo processor (196 tok/view) | Zero-shot at native resolution — proves the baseline was not handicapped by our pixel budget. |
| **Z1** | Same weights, adapter-dir processor (144 tok/view) | Zero-shot with *input-identical* tokens to row A. **The reference row.** |
| **P200** | Adapter at `checkpoint-200` (Stage B) | The trajectory: did 200 steps already move it. |
| **A** | Adapter, final checkpoint | The claim. |
| A-noFS *(optional)* | Row A with the FactSheet block emptied | Does the model cite evidence or recite the corpus. |
| Spec rows 1–2 (generic VLM, EarthDial-4B) | DATA_ADAPTATION_PLAN §8.5 | Later; not required to prove Z1 → A. |

### 4.3 Evaluation sets (held-out, leakage-guarded)

Every set is loaded with the train corpus's ids and image sha256 set in memory; an item present in either is a hard error, and the `Deduplicator` quarantine (`QUARANTINED_SPLITS`) is consulted the same way the builder does.

| Set | Source · split · n | Metric(s) |
|---|---|---|
| **E1** BEN-v2 19-class | BEN val, 200 patches × 19 binary questions ("Does the image contain *class*?") | 19-class multi-label **micro-F1** (§8.1) and 6-class merge micro-F1 (§8.2). Binary questions keep both rows in-distribution; a "list every class" prompt would test a task the corpus never taught. |
| **E2** VRSBench | val, 500 VQA + 200 referring-expression grounding | VQA accuracy (normalised EM); grounding **Acc@0.5** IoU. |
| **E3** CDVQA | val, 300 | change-VQA accuracy. |
| **E4** RSVQA-HR | val, 400 stratified over presence / comparison / count / area | per-type + aggregate accuracy (after the §5 render pass). |
| **E5** evidence_qa | a **val** split that must first be generated (today `evidence_qa.val.jsonl` is 0 bytes), 300 | citation strict-mode pass rate (§8.3); cited-value exactness against the sheet. |
| **E6** in-domain BEN VQA | the corpus's own `val.jsonl`, 300 mcq/binary | the most direct "did it learn the corpus" number; also the Stage-B dry-run set. |

Per row and set, four secondary numbers always accompany the headline: format-compliance rate, truncation rate (hit `max_new_tokens`), empty/refusal rate, mean answer length. A row that "improves accuracy" by emitting `NONE` everywhere is visible immediately.

### 4.4 Harness design

`scripts/eval_ablation.py --rows Z1,A --sets E1,E2,E5,E6 --n-per-set … --out runs/ablation/<stamp>/`:

- Loads one model configuration at a time (VRAM), runs every requested set, writes `items.jsonl` (id, row, set, prompt sha256, raw answer, parsed answer, correct/score, latency, truncated), and is resumable per `(row, set)`.
- Emits `summary.json` and regenerates **`DOCS/ABLATION_RESULTS.md`**: the table, the exact configuration block (git sha, adapter sha256, processor fingerprint, decoding parameters, n per cell), and a caption stating "single seed, single run, greedy decoding".
- Row Z1 on E6 with n = 64 is the smoke test run *before* Stage B, so harness bugs are found on a 5-minute job, not after 19 hours.

### 4.5 Statistical reporting

Per cell: n, point estimate, Wilson 95 % CI. Per Z1 → A delta: paired bootstrap (10,000 resamples, items paired by id) → `Δ = +x.x pp [lo, hi]`; McNemar exact p for accuracy-type metrics. No seed variance is claimed; the caption says so. Bad numbers are published in the same table — the handoff (§08 item 2) already requires this.

### 4.6 Table template

| Set / metric | Z0 (base, 196 tok) | Z1 (base, 144 tok) | P200 | A (adapted) | Δ Z1→A [95 % CI] |
|---|---|---|---|---|---|
| E1 BEN-19 micro-F1 | | | | | |
| E1 BEN-6 micro-F1 | | | | | |
| E2 VRSBench VQA acc | | | | | |
| E2 VRSBench grounding Acc@0.5 | | | | | |
| E3 CDVQA acc | | | | | |
| E4 RSVQA-HR acc (presence/comp/count/area/all) | | | | | |
| E5 citation strict pass rate | | | | | |
| E6 in-domain VQA acc | | | | | |
| format compliance / truncation / empty (all sets) | | | | | |

---

## 5. Corpus rebuild — the missing sources

### 5.1 State on disk today (fact 9)

| Source | Raw | Rendered views | In `full.train.jsonl` |
|---|---|---|---|
| bigearthnet_v2 | 273 GB | 24,995 patches | 15,530 (+ 3,000 evidence_qa) |
| vrsbench | 24 GB, zips unpacked | 29,615 | **0** |
| cdvqa | 538 images = 271 pairs; 9,000 / 2,000 question rows | 271 | **0** |
| rsvqa_hr | `Images.tar` present, `Data/` **not extracted** | **0** | **0** |
| dior_rsvg | 7.4 GB present | — | excluded: `UNRESOLVED_SOURCES` (gated release) — a provenance decision, keep it out unless resolved |
| evidence_qa | derived | — | 3,000 train / **0 val** |

### 5.2 Steps

1. `scripts/fetch_sources.py --source rsvqa_hr` (extracts `Data/`), then `scripts/render_vhr_views.py --source rsvqa_hr --split all --size 448 --workers 16`.
2. Build with **every** source named and the strict view policy: `scripts/build_corpus.py --sources bigearthnet_v2 vrsbench rsvqa_hr cdvqa evidence_qa --on-missing-views fail`. `fail`, not `skip`: `skip` is how a source quietly loses 90 % of itself.
3. Generate the `evidence_qa` **val** split from validation-patch FactSheets (E5 depends on it).
4. Apply the §2.5 grounding-format fix before the build, so the corpus is rebuilt once.

### 5.3 Composition assertion — no silent drops, ever again

- `build_corpus.py` gains `--require-sources` (default on): after `build_corpus_streaming`, any requested source with `train == 0`, or `train < 0.5 × min(target, built)`, exits 5 with the per-source table. It also writes `composition.json` next to the JSONL.
- The profile gains `data.expected_sources: {bigearthnet_v2: 8000, vrsbench: …}` (minimum train counts) and `train_vlm.py` counts the corpus it is *about to train on* — from the JSONL lines, not from the builder's report — and refuses if any expectation is unmet. The trainer verifies its own diet; a corpus file swapped under it is caught at start-up, and `run_manifest.json` records the counts.

### 5.4 Budget arithmetic

`hours = samples × epochs / 0.265 / 3600`. The §5 target of 65 k samples is 68 h at one epoch — not an overnight. Two honest presets, chosen by the GPU window actually available, both with every Track-B source present:

| Preset | Composition (train) | Steps @ 16 | Wall |
|---|---|---|---|
| **sprint** (one long night + a day) | BEN 8,000 · VRSBench 6,000 · RSVQA-HR 2,500 · CDVQA 1,500 · evidence_qa 2,000 = 20,000 | 1,250 | ≈ 21 h |
| **full §5 minus DIOR** | 18,000 · 20,000 · 10,000 · 8,000 · 3,000 = 59,000 | 3,688 | ≈ 62 h |

Masking's throughput gain is not budgeted. Checkpoints stay at every 50 steps (≈ 54 min of work at risk), `save_total_limit 3`.

---

## 6. Execution runbook — ordered, gated

Each step has a STOP condition. Nothing later starts while an earlier STOP is open.

| # | Step | Command / artefact | STOP if |
|---|---|---|---|
| 0 | Branch `train-v2`; pin `trl==1.12.0 transformers==5.16.1 peft==0.20.0` in the `vlm-train` extra; update `project_audit.md §2` and `AI_HANDOFF/08 §1` to point at this plan (fact 2). | `uv lock` | lockfile drift |
| 1 | Implement `prompts/layout.py`; re-point `hf_backend.build_messages`, `llamacpp_client.build_messages`; L1 tests. | `uv run pytest tests/unit` | any red |
| 2 | Implement the prompt-completion `sample_to_chat`, `sft_config_kwargs` changes, G1–G4, `audit_masks`, `guard_token_budget`, `--eval-first`, `SatQuerySFTTrainer`. | `uv run pytest tests/unit` | any red |
| 3 | Grounding-format fix (§2.5) + corpus test. | `uv run pytest tests/unit -k grounding` | any bare box survives |
| 4 | Corpus rebuild (§5.2–5.3); regenerate `evidence_qa` val. | `composition.json`; every source ≥ floor | exit 5 |
| 5 | Training-box pre-flight: L2 parity (`scripts/preflight_train_serve_parity.py`), G2 on 32 samples, G3 on the full corpus. | `mask_audit.json`, parity report | any assertion |
| 6 | **Stage A** memorisation probe (§3.2) → `probe_report.py`. | `runs/probe-A-memorise/probe_report.json` | verdict FAIL |
| 7 | Ablation harness smoke: row Z1 on E6, n = 64 (§4.4). | `runs/ablation/smoke/` | parse/leakage error |
| 8 | **Stage B** descent probe = first 200 steps of the real run (§3.3), `--eval-first`. | `runs/full-epoch-v2/probe_report.json` | verdict FAIL |
| 9 | Full run: same command, `--resume auto` picks up `checkpoint-200`; `eval_steps 250` on the 128-item held-out set; Ctrl+C-safe as today. | `runs/full-epoch-v2/adapter`, `run_manifest.json` incl. `zero_shot_eval`, `layout`, `composition` | eval loss rising for 2 consecutive evals (early stop, keep best) |
| 10 | Ablation (§4) for rows Z0, Z1, P200, A on E1–E6. | `DOCS/ABLATION_RESULTS.md`, `runs/ablation/<stamp>/` | — publish whatever it says |
| 11 | Serve: `SATQUERY_VLM_ADAPTER_PATH=runs/full-epoch-v2/adapter`; L3 fingerprint check passes on load; re-run `inference_probe*.json`; re-export GGUF (`scripts/merge_export.py`, handoff Tier-1 item 4). | probes, GGUF | `ModelLoadError` on fingerprint |

Wall-clock, sprint preset: steps 5–8 ≈ 5.5 h · step 9 ≈ 18 h remaining · step 10 ≈ 6–8 h across four rows. Everything before step 6 is CPU/CI work.

---

## 7. Other defects found during this read (fix in the same pass)

1. **Grounding target format split 50/50** (fact 13) → §2.5.
2. **`evidence_qa.val.jsonl` is empty** → the citation metric has no held-out set until it is generated (§5.2).
3. **Over-long samples become zero-loss silently** (fact 11) → G3.
4. **`image_placeholder_count` and its test target the `messages` shape** → re-point at `prompt` (§1.2).
5. **`run_manifest.json` records neither the corpus composition, nor a zero-shot eval, nor the prompt layout** → all three added (§3.4, §2.3-L3, §5.3).
6. **`--sanity-check` takes the first 100 lines**, which in a source-ordered corpus is one source and one task → stratify, reusing the Stage-A sampler.
7. **`hf_backend.processor_source` falls back to the base repo** when the adapter dir lacks `processor_config.json` → 196 vs 144 tokens/view, silently. Becomes a hard error (§2.4).
8. **Two documents recommend the dead-end mechanism** (`project_audit.md §2 "What good looks like"`, `AI_HANDOFF/08 §1`) → updated in step 0.
9. **NF4-trained, bf16-served** (audit §2 item 3) is not a bug we can fix on this hardware; the ablation measures the adapter *as served*, which is the only number that matters. Recorded, not resolved.

---

## 8. Definition of done

- [ ] `sft_config_kwargs()["completion_only_loss"] is True`, pinned by a unit test; `assistant_only_loss` explicitly `False` with the trl-1.12 reason in a comment.
- [ ] Records are `{prompt, completion, images}`; G1–G4 in place; `mask_audit.json` shows 32/32 with unmasked tokens == completion tokens.
- [ ] `layout.py` is the only place a user turn is laid out; L1 green in CI, L2 green on the training box, L3 enforced in `hf_backend._load`.
- [ ] Every GROUNDING target is canonical or `NONE`.
- [ ] Corpus contains every requested source above its floor; `composition.json` and the profile's `expected_sources` agree; `evidence_qa` has a val split.
- [ ] Stage A `PASS`; Stage B `PASS` with a recorded zero-shot baseline and a descending held-out curve.
- [ ] `DOCS/ABLATION_RESULTS.md` exists with rows Z1 and A on E1–E6, CIs and paired deltas, generated by one command — numbers published regardless of sign.
- [ ] The served adapter passes the layout fingerprint on load; `inference_probe*.json` re-run; GGUF re-exported from the same adapter.
