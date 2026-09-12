# 06 — Data Engine and Training

Spec: `DOCS/DATA_ADAPTATION_PLAN.md` (frozen). Audit: `DOCS/project_audit.md §1–2`.
This file records the plan, then **what actually happened**, then how to redo it.

## The planned corpus (§5 of the data plan) — 65,000 samples, one adapter

| Source | Planned | Share | Track | Access | Status on disk |
|---|---:|---:|---|---|---|
| `bigearthnet_v2` (S1+S2, 10 m, 12 bands) | 18,000 | 27.7 % | A | HF `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt` (9.6 M-row parquet + tarballs) | rendered; **in corpus** |
| `vrsbench` (VHR 0.1–3 m) | 20,000 | 30.8 % | B | annotation zips (HF loader crashes 20,264 rows in) → `local_sources` | 29,615 tiles rendered; **not in corpus** |
| `rsvqa_hr` | 10,000 | 15.4 % | B | off-Hub JSON; one file has *all* 955,664 questions, split marked by `active: true` (naive read leaks 330,324 test/Philadelphia rows) → `local_sources` | raw fetched (3 GB); **not in corpus** |
| `cdvqa` (bi-temporal pairs) | 8,000 | 12.3 % | B | WebDataset, 1,533 shards → `fetch_sources.py --source cdvqa` unpacks | 271 pairs rendered; **not in corpus** |
| `dior_rsvg` | 6,000 | 9.2 % | B | HF, **gated — unresolved** | absent |
| `evidence_qa` (synthetic) | 3,000 | 4.6 % | A | `training/data/builders/evidence_qa.py` from measured BEN scalars | **in corpus** (train only; val = 0 rows) |

Splits: official train → `train.jsonl`, official val → `val.jsonl`, official test
**quarantined** and never read (a leak is a `LeakageError` build failure).

## What the production adapter actually trained on

`runs/full-epoch-v1` ← `data/processed/corpus/full.train.jsonl` = **18,530 rows =
BigEarthNet-v2 (15,530) + evidence_qa (3,000)**. Track B is absent. The image-level
dedup ran on nothing for BEN because `scripts/build_corpus.py::bind_view_paths()`
deliberately does not set `image_path` for BEN and `hashes_for()` only hashes rows
with `image_path`/`pre_path`, so every BEN row got `{}` and `deduplicate()` passed
unhashed rows through (patch ids are unique, so practical leakage risk is low, but
the "strict dedup" claim is not true of this artifact). No test covers that seam.

Why the corpus was cut: the throughput probe measured **0.265 samples/s** (≈68–73 h
per 65k epoch), RSVQA-HR/CDVQA readers were still being fixed, and the overnight
window was fixed. The decision is recorded at the tail of
`logs/overnight-20260909-181509.log`.

## Unified instruction sample (what one JSONL row is)

Each row carries: the list of rendered views (paths + the exact label string from
`render/view_labels.label_for_view`), the FactSheet the sample is conditioned on,
and the assistant turn. Boxes are serialised by `models/prompts/box_format.serialise`
(Qwen native 0–1000 form). Numbers are formatted by
`evidence/citation_validator.format_number`. Citations in answers are `[key]`
markers. `corpus_builder.assemble()` runs the CitationValidator over every training
answer and raises `CitationError` on any unresolved number.

## Rendering for training (`scripts/render_views.py`, `render_vhr_views.py`)

Same renderer as serving (`render/`), 448 px, fixed domains, PNG for index/SAR views,
JPEG q92 for natural composites. Views land under `data/processed/views/<source>/`;
`rendered.jsonl` / `factsheets*.jsonl` record paths and measured scalars. Six views
per sample max. Never train on `patch_dummy_images.py` placeholders.

## Dedup (`corpus_builder.Deduplicator`)

1. exact: SHA-256 of the source image;
2. near: DCT pHash (grey, 32×32, low 8×8 minus DC, 64 bits), Hamming ≤ 5, 8-band
   bucketing so lookups do not scan 65k hashes;
3. quarantine: any train image matching a test-split hash by either method → fatal.

`check(..., image_key=…)` distinguishes "many annotations per image" (legitimate:
VRSBench ~10/tile, RSVQA-HR ~100/tile, CDVQA ~40/pair) from "same image twice".
Without it VRSBench collapsed 205,121 → 20,264 and CDVQA 9,000 → 220 in an early run.
Streaming build with a bounded hash cache (`HASH_CACHE_SIZE = 65,536`) and per-source
reservoir sampling to the §5 targets.

## VLM training (`training/vlm/qlora.py`, `scripts/train_vlm.py`)

Profile A (`configs/train/qlora_qwen3vl8b_rocm24g.yaml`): Qwen3-VL-8B-Instruct, NF4
double-quant with bf16 compute, LoRA r=16 α=32 dropout 0.05 on
`q,k,v,o,gate,up,down` + `qkv,proj,fc1,fc2` of the **last 8 ViT blocks** (resolved
by name, not index), `max_pixels` 147,456 (384², must be a multiple of 32²),
`max_seq_len` 4096, batch 1 × accum 16, 1 epoch, lr 1e-4 cosine, 3 % warmup,
`paged_adamw_8bit`, grad checkpointing, `save_steps` 50 / `save_total_limit` 3,
`eval_samples` 256, resumable (`resume_from_checkpoint: auto`, optimiser state kept).
VRAM estimator predicted 17.13 GiB; measured peak 16.78 GiB. Full epoch: 1,158 steps,
~19 h.

**Defects in the objective (fix before any re-run):**
1. **No assistant-only loss.** `sft_config_kwargs()` sets neither
   `assistant_only_loss` nor `completion_only_loss`. Loss is over system prompt +
   30-float FactSheet + answer. Train loss 13.0 → 3.8 by 8 % of the epoch, then flat
   to 3.76; token accuracy 0.50 → 0.51; eval loss 3.935 → 3.930. Fix: trl ≥ 0.20
   `assistant_only_loss=True` with `{% generation %}` markers added to Qwen3-VL's
   chat template, or a prompt-completion dataset shape. Prove it on a 200-step probe
   before spending 19 h.
2. **Train/serve prompt skew.** `sample_to_chat()` emits images as
   `{"type": "image", "label": ...}` — `label` is not a text part and the template
   drops it, so training images were unlabelled except via the system prompt.
   Serving (`hf_backend.build_messages`) puts each label as a text part directly
   before its image. Align `sample_to_chat` with `build_messages`.
3. NF4-trained LoRA served on bf16 base activations — unmeasured cost; include in
   the ablation.

`sample_to_chat()` also documents two real trl foot-guns (the `images` key
requirement and bare-placeholder counting) — read its docstring before touching it.

## Merge / export (`scripts/merge_export.py`)

Merges LoRA into base → safetensors → GGUF f16 + Q4_K_M + mmproj via llama.cpp's
converter. **Only `poc-v1` has been exported.** The llama.cpp demo path therefore
serves the PoC adapter, not `full-epoch-v1`.

## Change detection (`training/cd/`, `scripts/train_cd.py`)

Siamese: shared ResNet-18 encoder initialised from SSL4EO-S12 (via torchgeo),
per-scale differencing, FPN decoder, Lightning task with threshold calibration on
val. LEVIR-CD run: 60 epochs @ 256 px, bf16-mixed (fp16 overflows in deep ResNet
stages on gfx1100). Result **F1 0.858 / IoU 0.751** at calibrated threshold 0.9 —
below the 0.88 gate; the checkpoint JSON states this and must not be quoted as
meeting it. Normalisation: identity (a bug where measured stats were recorded but
never applied was corrected; the `.bak-prenorm` files predate the fix). Self-test
baked into the bundle (`changed_fraction 0.0499 ± 0.02` on a fixed crop). **OSCD
(10 m) run not done** — data is downloaded; the plan wants both resolutions to
record cross-resolution degradation as Cartosat-transfer evidence.

## Segmenter

`data/checkpoints/seg/segformer-b5-loveda/` is a downloaded HF SegFormer-B5 LoveDA
snapshot (not trained here). Classes map onto the shared vocabulary (building, road,
water, barren, forest, agriculture). No OpenEarthMap training happened.

## DOFA

`tools/crossmodal_dofa.py` uses `torchgeo.models.dofa_base_patch16_224` pretrained
weights (downloaded on demand). No fine-tuning.

## Evaluation status

Nothing reproducible exists. `scripts/eval_vrsbench_zeroshot.py` runs a 5-item
synthetic mock by default. `scripts/test_inference.py` is a single-patch grounding
smoke test. The data plan's §8 protocol (BEN-v2 19-class micro-F1 zero-shot vs
adapted; 6-class replication target ≈0.59 → ≈0.83; evidence-audit fraction, reference
0.9746; VRSBench/RSVQA-HR/CDVQA downstream; the four-row ablation) is all still to
do — that is Phase 8.

## How to redo the whole thing correctly (order matters)

1. Fix the loss masking and the label skew in `qlora.py`; add a 10-step ML
   regression test on a tiny corpus asserting assistant-token loss decreases.
2. Fix `hashes_for`/`bind_view_paths` so every source is hashed; add the test.
3. Include VRSBench + CDVQA (already rendered) and RSVQA-HR in `build_corpus.py`;
   re-budget `COMPOSITION` for the measured 0.265 samples/s and the available
   window; generate an `evidence_qa` val split.
4. `train_vlm.py --sanity-check`, then a 200-step probe; confirm assistant-token
   accuracy climbs well past 0.5.
5. Full run. Then `merge_export.py` from *that* adapter so the GGUF matches.
6. Ablation: base Qwen3-VL bf16 vs adapter, same prompt, same 200 VRSBench + BEN-val
   items, same validator. Publish the numbers in `DOCS/` even if bad.
