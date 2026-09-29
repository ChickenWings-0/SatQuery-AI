# 06 — Data Engine and Training

Spec: `DOCS/DATA_ADAPTATION_PLAN.md` (frozen). The v1 → v2 fix record:
`DOCS/ML_PIPELINE_RECOVERY_PLAN.md`. This file records what was built, what was
trained, and how to redo it.

> **2026-09-29:** `data/raw/` and `data/processed/` were deleted to free ~536 GiB,
> along with every run except the final adapter. Everything below that needs views
> or a corpus must be rebuilt first (last section).

## Sources

| Source | Track | Access | In v2 corpus |
|---|---|---|---|
| `bigearthnet_v2` (S1+S2, 10 m, 12 bands) | A | `hf download torchgeo/bigearthnet` + tarballs (`scripts/render_views.py` docstring) | ✅ |
| `vrsbench` (VHR) | B | annotation zips → `local_sources` (HF loader crashes 20,264 rows in) | ✅ |
| `rsvqa_hr` | B | off-Hub JSON; filter `active: true` or leak 330,324 test rows → `local_sources` | ✅ |
| `cdvqa` (bi-temporal) | B | WebDataset, `fetch_sources.py --source cdvqa` unpacks | ✅ |
| `evidence_qa` (synthetic) | A | `training/data/builders/evidence_qa.py` from measured BEN scalars, split-stratified | ✅ |
| `dior_rsvg` | B | HF, **gated** — accept terms, `HF_TOKEN`, re-run | ❌ excluded |
| LEVIR-CD, OSCD | — | torchgeo datamodules | CD training only |

Splits: official train → train, official val → val, official test **quarantined**
(a leak is a `LeakageError` build failure).

## The v2 corpus (`data/processed/corpus/v2-full/`, deleted)

Built by `PRESET=full scripts/rebuild_corpus_v2.sh` ("§5 minus DIOR", 59,000 rows):

| Source | Train | Val | Target |
|---|---:|---:|---:|
| bigearthnet_v2 | 16,200 | 1,800 | 18,000 |
| vrsbench | 18,000 | 2,000 | 20,000 |
| rsvqa_hr | 9,000 | 1,000 | 10,000 |
| cdvqa | 7,198 | 802 | 8,000 |
| evidence_qa | 2,700 | 300 | 3,000 |
| **Total** | **53,098** | **5,902** | 59,000 |

Every source is image-hashed (`dedup_image_path` + `HashTally` — the v1 seam where
BEN rows went unhashed is fixed and tested), deduplicated (SHA-256 → DCT pHash
Hamming ≤ 5, `image_key` so multi-annotation images survive), audited by the
CitationValidator at build time, and written with `composition.json` and `MANIFEST.md`.
Val tasks: BEN (SCENE_CLASSIFY, GROUNDING, VQA, CROSS_MODAL_VQA, CROSS_MODAL_COMPARE),
VRSBench (VQA, GROUNDING, CAPTION, COUNT), RSVQA-HR (VQA, COUNT), CDVQA (CHANGE_VQA),
evidence_qa (VQA, CROSS_MODAL_VQA).

## Unified instruction sample

Each row: rendered view paths + exact label strings (`render/view_labels`), the
FactSheet, and `messages` whose last assistant turn is the reference. Boxes via
`box_format.serialise` (0–1000); numbers via `citation_validator.format_number`;
citations as `[tool.scalar]`. At training time `sample_to_chat()` turns a row into a
**prompt-completion record** `{prompt, completion, images}` laid out by
`models/prompts/layout.py` — label as a text part before each image, identical to serving.

## History: why v1 failed (kept short — `ML_PIPELINE_RECOVERY_PLAN.md` has the detail)

`runs/full-epoch-v1` (2026-09-10, deleted): 18,530 BEN + evidence_qa rows only,
~19 h, **flat loss** (3.82 → 3.76; eval 3.935 → 3.930; token accuracy ~0.50). Causes:
1. loss over the whole sequence (1,000-token system prompt + 30-float FactSheet
   dominated); 2. training images carried labels in a non-text `label` key the chat
   template dropped, while serving put labels as text — train/serve skew; 3. BEN rows
   never hashed; Track B absent. A PoC adapter (`poc-v1`) was the only GGUF export.

**The trl foot-gun that matters for any retrain:** on trl 1.12 `assistant_only_loss=True`
raises `ValueError("Assistant-only loss is not yet supported for vision datasets")` for
any dataset with `images`, and `DataCollatorForCompletionOnlyLM` was removed in trl
0.20. The working mechanism is **prompt-completion records + `completion_only_loss=True`**,
with `audit_masks()` asserting collated labels equal the tokenised answer before the
weights load.

## The production run — `runs/sq-lora-v2-full`

Profile A (`configs/train/qlora_qwen3vl8b_rocm24g.yaml`): Qwen3-VL-8B-Instruct, NF4
double-quant + bf16 compute, LoRA r=16 α=32 dropout 0.05 on `q,k,v,o,gate,up,down` +
`qkv,proj,fc1,fc2` of the **last 8 ViT blocks** (by name), `max_pixels` 147,456
(144 image tokens per 448 px view), `max_seq_len` 4096, ≤ 6 views/sample, batch 1 ×
accum 16, 1 epoch, lr 1e-4 cosine, 3 % warmup, `paged_adamw_8bit`, grad checkpointing,
`save_steps` 50 / keep 3, eval every 250 steps on 256 samples, seed 42.

Results (from `run_manifest.json` / `train.log`): 3,318 steps, 40.7 h, 0.363 samples/s,
peak **13.42 GiB**; zero-shot eval loss 8.466 / answer-token accuracy 26.4 % →
epoch-1 eval loss **0.1867** / answer-token accuracy **81.8 %**, train loss 0.3025.
Mask audit 32/32. Note `run_manifest.json` says `run_name: sq-lora-v1` — the profile's
default name was never changed; the directory name is authoritative.

The adapter directory holds `adapter_model.safetensors` (~86 MB), `adapter_config.json`,
processor/tokenizer/chat template, and **`layout.json`** (layout version + fingerprint
the HF backend verifies at load).

## Evaluation (`src/satquery/eval/`, `scripts/eval_benchmark.py`)

`make eval` samples 100 per source from `v2-full/val.jsonl` (round-robin by task,
`sample_ids.json` for reproducibility), runs the served backend, scores with pure
functions, and writes `runs/eval/<name>/results.{md,json,csv}` (tracked) plus
`predictions.jsonl` and `confusion/` (ignored). `make eval-baseline` reruns the same ids
with no adapter; `make eval-self-check` checks scorers against references (must read
100 %). Results for the production adapter are in `02`. Not yet produced: the baseline
column and a `SATQUERY_VLM_BACKEND=llamacpp` (Q4_K_M) column.

## Merge / export (`scripts/merge_export.py`)

bf16 merge **on CPU** (never onto NF4 weights) → safetensors + processor + chat
template → `convert_hf_to_gguf.py --outtype bf16` → `--mmproj --outtype f16` →
`llama-quantize … Q4_K_M` (needs a llama.cpp checkout with Qwen3-VL support). The
mmproj stays f16 — vision-tower quantisation breaks grounding first. Output and
checksums in `models/` (2026-09-22, see `02`). The intermediate bf16 GGUF was not kept.

## Change detection (`training/cd/`, `scripts/train_cd.py`)

Siamese ResNet-18 (SSL4EO-S12 init), per-scale differencing, FPN decoder, Lightning,
threshold calibrated on val. LEVIR-CD: 60 epochs @ 256 px, bf16-mixed (fp16 overflows
on gfx1100). **F1 0.858 / IoU 0.751** at threshold 0.9 — below the 0.88 gate. Identity
normalisation (a recorded-but-unapplied stats bug was corrected; `.bak-prenorm` files
predate it). Self-test baked in (`changed_fraction 0.0499 ± 0.02`). No OSCD run.

## Segmenter and DOFA

`data/checkpoints/seg/segformer-b5-loveda/` — downloaded HF SegFormer-B5 LoveDA, not
trained here. DOFA — `torchgeo.models.dofa_base_patch16_224` pretrained, downloaded
on demand, no fine-tuning.

## Rebuilding after the cleanup (order matters)

1. Disk: ~340 GB free needed for sources + views (available now).
2. Fetch: BigEarthNet via the `hf download` recipe in `scripts/render_views.py`;
   `scripts/fetch_sources.py --source rsvqa_hr` / `--source cdvqa`; VRSBench zips into
   `data/raw/vrsbench`; LEVIR-CD via `train_cd.py --download` only if retraining CD.
3. Render: `scripts/render_views.py` (BEN) and `scripts/render_vhr_views.py` (VHR).
4. Build: `PRESET=full scripts/rebuild_corpus_v2.sh` — same seed and composition
   reproduce the same val split, so `runs/eval/sq-lora-v2-full/sample_ids.json` stays valid.
5. Then `make eval-baseline`, `make eval` with the llama.cpp backend, or a retrain
   (`preflight_train_serve_parity.py` → `train_vlm.py --sanity-check` → full run →
   `merge_export.py` from that adapter).
