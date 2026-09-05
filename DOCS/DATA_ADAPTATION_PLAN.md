# SatQuery AI — Rendering, Corpus & Domain-Adaptation Plan (FROZEN)

**Status:** FROZEN as of Phase 0 (view catalogue + sample schema) / executed in Phase 7.
**Authority:** Derived strictly from `DOCS/Master.md` §1, §2, §6.2, §6.3, §8-Phase-2, §8-Phase-7, §9.
**Satisfies:** mandatory requirement 1 — "at least one visual or vision-language model must be fine-tuned or domain-adapted using BigEarthNet or open-source remote-sensing data."

> **The load-bearing decision (Master.md §6.2):** we do **not** modify patch embeddings to accept 12-band input. We render every raster into a small set of *named, interpretable 3-channel views* and feed them through Qwen3-VL's existing multi-image interface with explicit naming in the prompt. One renderer serves three consumers: the VLM at training time, the VLM at inference time, and the frontend evidence gallery.

---

## 1. Why This Shape

| Constraint | How the rendering approach answers it |
|---|---|
| BigEarthNet is 12-band S2 + 2-band S1; hidden set is 4-band Cartosat-2S + 1-band RISAT | Views are computed per sensor from a band-alias map. **Zero architecture change** across all four sensors. |
| Hidden set is ~15x finer GSD than the mandated training data | Resolution augmentation (§7.3) plus a two-track corpus (§5) spanning 0.3-20 m. |
| Answers must be evidence-grounded and citable | Index views are rendered on **fixed** value domains, so "NDVI 0.72" means the same thing in every image the model has ever seen. |
| Frontend needs human-viewable evidence | The exact images the model saw are emitted as `RENDERED_VIEW` artifacts. |
| 24 GB VRAM | Views per sample are capped at 6 — the dominant memory knob (Master.md §6.2). |

---

## 2. View Catalogue (FROZEN)

### 2.1 Views

| `view_id` | Name | Channels | Value domain | Colormap | File |
|---|---|---|---|---|---|
| `TC` | True colour | `red, green, blue` | per-image p2-p98 | none | JPEG q92 |
| `FCIR` | False-colour infrared | `nir, red, green` | per-image p2-p98 | none | JPEG q92 |
| `SWIR` | SWIR composite | `swir2, nir, red` | per-image p2-p98 | none | JPEG q92 |
| `NDVI` | Vegetation index | `(nir-red)/(nir+red)` | **fixed [-1, +1]** | `RdYlGn` | **PNG** |
| `NDWI` | Water index | `(green-nir)/(green+nir)` | **fixed [-1, +1]** | `BrBG_r` | **PNG** |
| `NDBI` | Built-up index | `(swir1-nir)/(swir1+nir)` | **fixed [-1, +1]** | `RdBu_r` | **PNG** |
| `SARFC` | SAR false colour | `VV_dB, VH_dB, (VV/VH)_dB` | **fixed clips** (§2.3) | none | JPEG q92 |
| `SARDB` | SAR backscatter | `VV_dB` (or sole pol), replicated | **fixed [-25, 0] dB** | `gray` | **PNG** |
| `PAN` | Panchromatic | `pan` replicated | per-image p2-p98 | `gray` | JPEG q92 |
| `CHANGE` | Change overlay | mask over post `TC` | — | red @ 45 % alpha | PNG |

**Two rules that are not negotiable:**

1. **Index and SAR views use FIXED domains, never per-image normalisation.** Per-image stretch would make NDVI 0.7 render identically to NDVI 0.2 in a low-contrast scene, and the model could never learn absolute physical semantics. This is the single most important detail in the recipe.
2. **Index and SAR views are PNG, not JPEG.** JPEG chroma subsampling on a colormapped field shifts the apparent value by a visible amount. Natural-looking composites (`TC`, `FCIR`, `SWIR`, `SARFC`, `PAN`) tolerate JPEG q92; measurement views do not.

`CHANGE` is runtime-only — it is an evidence artifact, never a training input.

### 2.2 Band alias map (`configs/band_aliases.yaml`)

Shared verbatim with capability matching (`AGENT_POLICY_DAG.md` §5). 1-indexed band numbers.

| Sensor | Detection heuristic | blue | green | red | nir | swir1 | swir2 | pan | vv | vh |
|---|---|---|---|---|---|---|---|---|---|---|
| Sentinel-2 (BigEarthNet-v2, 12 band) | `band_count==12`, uint16 | 2 | 3 | 4 | 8 | 11 | 12 | – | – | – |
| Sentinel-2 L2A (13 band) | `band_count==13`, uint16 | 2 | 3 | 4 | 8 | 12 | 13 | – | – | – |
| Sentinel-1 GRD | `band_count==2`, names `VV`/`VH` | – | – | – | – | – | – | – | 1 | 2 |
| Cartosat-2S MX | `band_count==4`, `gsd_m < 3` | 1 | 2 | 3 | 4 | – | – | – | – | – |
| Cartosat-2S PAN | `band_count==1`, `gsd_m < 1` | – | – | – | – | – | – | 1 | – | – |
| RISAT-1 | `band_count==1`, dB-like histogram | – | – | – | – | – | – | – | 1 | – |
| Generic RGB (benchmark PNG) | `band_count==3`, uint8 | 3 | 2 | 1 | – | – | – | – | – | – |

BigEarthNet-v2 S2 patches carry 12 bands (B10 excluded): `B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12`.

> **Unavailability is explicit, never substituted.** Cartosat-2S has no SWIR, so `SWIR` and `NDBI` are **unavailable** for it. The renderer returns `unavailable` with a reason; the FactSheet records `ndbi_mean: null`; the prompt omits the view. Silently substituting another band to keep the view count up would produce a confidently wrong built-up answer on the hidden evaluation set. This gap is precisely why the cross-modal SAR path exists — on Cartosat/RISAT pairs, built-up evidence comes from radar backscatter, not NDBI.

### 2.3 Normalisation (FROZEN)

```
reflectance composites (TC, FCIR, SWIR, PAN)
    per-band percentile stretch p2-p98 over valid (non-nodata) pixels -> uint8

indices (NDVI, NDWI, NDBI)
    clip to [-1, +1] -> colormap -> uint8 RGB          # fixed, image-independent

SAR
    sigma0_dB = 10 * log10(max(DN, eps))               # if input is not already dB
    VV_dB      clip [-25,  0]
    VH_dB      clip [-30, -5]
    ratio_dB = VV_dB - VH_dB, clip [0, 15]
    each channel linearly mapped to [0, 255]           # fixed, image-independent

geometry
    resample to view_size_px (default 448) with bilinear (reflectance) /
    nearest (masks); pad with zeros and record pad extent
```

`eps = 1e-6`. Nodata pixels render as pure black `(0,0,0)` in every view and are excluded from every statistic.

### 2.4 View selection policy (cap: 6)

| Pair type | Modality | Views, in slot order | n |
|---|---|---|---|
| `SINGLE` | optical, SWIR present | `TC, FCIR, SWIR, NDVI, NDBI, NDWI` | 6 |
| `SINGLE` | optical, no SWIR (Cartosat) | `TC, FCIR, NDVI, NDWI` | 4 |
| `SINGLE` | SAR, dual-pol | `SARFC, SARDB` | 2 |
| `SINGLE` | SAR, single-pol (RISAT) | `SARDB` | 1 |
| `SINGLE` | panchromatic | `PAN` | 1 |
| `SINGLE` | generic RGB | `TC` | 1 |
| `CROSS_MODAL` | mixed | `TC, FCIR, NDVI, NDBI, SARFC, SARDB` | 6 |
| `BI_TEMPORAL` | optical | `TC(pre), TC(post), FCIR(pre), FCIR(post), NDBI(pre), NDBI(post)` | 6 |
| `BI_TEMPORAL` | SAR | `SARFC(pre), SARFC(post), SARDB(pre), SARDB(post)` | 4 |

Slot order is fixed and deterministic. If a view is unavailable it is dropped and later slots renumber — the *label* carries the meaning, not the slot index.

### 2.5 View labels — the shared constant

```
Image {slot} ({modality} {view_name}{, temporal_role}{, sensor}{, scale_note})
```

Examples, verbatim:

```
Image 1 (optical true colour, Sentinel-2)
Image 4 (optical NDVI heatmap, fixed scale -1 to +1)
Image 5 (SAR false colour VV/VH/ratio, Sentinel-1, fixed scale)
Image 1 (optical true colour, pre-change, 2019-03-14)
Image 2 (optical true colour, post-change, 2021-07-02)
```

> **Training and inference must emit byte-identical labels.** The entire recipe rests on the language model binding a named view to its physical meaning; a mismatch between corpus labels and runtime labels silently destroys the adaptation while every test still passes. Therefore the label builder lives in exactly one place — `src/satquery/render/view_labels.py` — and is imported by both `training/data/build_corpus.py` and the runtime prompt builder. **Duplicating this string anywhere is a defect.**

---

## 3. Unified Instruction Sample Schema

One JSONL line per sample. `data/processed/corpus/{train,val}.jsonl`.

```jsonc
{
  "id": "ben2:S2A_MSIL2A_20180413T95029_37_58",
  "source": "bigearthnet_v2",          // bigearthnet_v2|vrsbench|rsvqa_hr|cdvqa|dior_rsvg|evidence_qa
  "task": "SCENE_CLASSIFY",            // a TaskType from API_CONTRACT.md 2.1
  "pair_type": "CROSS_MODAL",
  "views": [
    { "slot": 1, "view_id": "TC",    "path": "views/ben2/…/TC.jpg",
      "label": "Image 1 (optical true colour, Sentinel-2)" },
    { "slot": 2, "view_id": "SARFC", "path": "views/ben2/…/SARFC.jpg",
      "label": "Image 2 (SAR false colour VV/VH/ratio, Sentinel-1, fixed scale)" }
  ],
  "messages": [
    { "role": "system",    "content": "…" },
    { "role": "user",      "content": "…" },
    { "role": "assistant", "content": "…" }
  ],
  "fact_sheet": { "ndvi_mean": 0.62, "sigma0_vv_mean_db": -8.4 },
  "meta": { "sensor": "Sentinel-2 L2A + Sentinel-1 GRD", "gsd_m": 10.0,
            "labels": ["Broad-leaved forest","Pastures"],
            "split": "train", "source_split": "train",
            "augmentations": ["rescale_2.0x"] }
}
```

`fact_sheet` is carried for two reasons: it supervises `evidence_qa` (§4.6), and it lets Phase 8 run the CitationValidator over generated answers to compute the evidence-audit metric (§8.3).

---

## 4. Source Corpora & Builders

`training/data/builders/`. Every builder emits the §3 schema and nothing else.

### 4.1 `bigearthnet.py` — **the mandated source**

BigEarthNet-v2.0 / reBEN: 549,488 co-located Sentinel-1 GRD + Sentinel-2 L2A patch pairs, 1200 m x 1200 m, 19-class multi-label nomenclature.

**Subset selector runs on metadata, before any patch download** (Master.md §8-Phase-7: *"Write the subset selector before downloading"*):

1. Fetch the reBEN metadata table only (parquet, < 1 GB).
2. Keep the **official** train/val/test split. Never re-split — the published baselines depend on it.
3. Iterative stratification over the 19-class label-combination vector, balanced additionally on season where the metadata provides it.
4. Target **20,000 train / 3,234 val**. Emit a manifest of patch ids.
5. Download only the manifested patches (~15 GB, versus ~400 GB for the archive).

**Generated tasks:** `SCENE_CLASSIFY` (multi-label), plus balanced yes/no presence questions, plus cross-modal summaries. Question templates (`configs/templates/bigearthnet.yaml`):

- *"Which land-cover classes are present in this scene?"* -> comma-separated class list
- *"Is there {class} in this image?"* -> yes/no, negatives sampled to 50 %
- *"Does the radar imagery indicate built-up structures here?"* -> answer grounded in σ⁰ and VV/VH ratio
- *"Summarise the land cover, citing both the optical and radar evidence."* -> 2-3 sentences naming both

**Two taxonomies are produced:**

- `bigearthnet19` — the official 19-class nomenclature. **This is our primary mandate metric.**
- `bigearthnet6` — a 6-class merge (`configs/ben6_merge.yaml`) mirroring the reference study's protocol so the headline 0.59 -> 0.83 comparison is like-for-like.

> **Honesty note on the reference numbers.** Master.md §6.2 cites the published BigEarthNet-v2 result of micro-F1 **0.5921 zero-shot -> 0.8275 adapted** under a 6-class merge with 12,900 train / 3,234 val. We replicate that protocol as closely as the published description permits. If the exact class merge cannot be recovered from the paper, we report it as *comparable but not identical*, and **our own measured zero-shot baseline — not the published 0.5921 — is the reference point for our claimed delta.** Claiming parity with a protocol we could not reproduce would not survive a judge's question.

### 4.2 `vrsbench.py`

VRSBench (`xiang709/VRSBench` on Hugging Face): 29,614 VHR images, 123,221 VQA pairs, 52,472 object references, plus detailed captions. Imagery is 3-channel VHR aerial (~0.1-3 m).

Tasks: `VQA`, `CAPTION`, `GROUNDING`. Views: `TC` only (already 3-channel). Sample **20,000** from the official **train** split.

> The official VRSBench **test** split is quarantined for Phase 8 and must never enter the corpus. `build_corpus.py` asserts this by id.

### 4.3 `rsvqa.py`

RSVQA-HR: 10,659 images, 1,066,316 QA pairs across presence / comparison / count / area. Sample **10,000**, balanced across the four question types (they are wildly imbalanced in the raw data, and unbalanced sampling teaches the model to guess the majority type). Tasks: `VQA`, `COUNT`. Views: `TC`.

### 4.4 `cdvqa.py`

CDVQA — change-detection VQA built over the SECOND bi-temporal dataset. Sample **8,000**. Task: `CHANGE_VQA`. `pair_type: BI_TEMPORAL`. Views: `TC(pre), TC(post)`.

### 4.5 `dior_rsvg.py`

DIOR-RSVG referring-expression grounding (~17k images, ~38k expressions). Sample **6,000**. Task: `GROUNDING`. Views: `TC`.

**The assistant's answer must use Qwen3-VL's native normalised-box format, byte-identical to what `tools/text_grounding.py` parses at inference.** Both sides import one serialiser, `src/satquery/models/prompts/box_format.py`. A training/inference format drift here produces a model that grounds perfectly and a parser that returns nothing.

### 4.6 `evidence_qa.py` — **synthetic, and load-bearing**

Programmatically generated from the FactSheet the Phase-2 renderer already computes for each BigEarthNet patch and each LEVIR-CD pair. **3,000 samples.**

*Why this exists:* none of the five public corpora teach citation behaviour. They teach remote-sensing vocabulary and task formats, but every answer in them is unsourced prose. Without explicit supervision, the LoRA will learn to *sound* like a remote-sensing analyst while still inventing numbers — which defeats the FactSheet-constrained generation the whole architecture is built around (Master.md §1, §4.4). Because these samples are generated *from* the FactSheet, every number in every answer is correct and attributable by construction — perfect supervision at near-zero cost.

Templates:

- Q: *"What is the mean NDVI, and what does it indicate?"*
  A: *"The mean NDVI is 0.62, indicating dense healthy vegetation across most of the scene."*
- Q: *"Do the optical and radar signals agree about built-up area?"*
  A: *"Partially. NDBI averages -0.08, a low built-up signal, while σ⁰_VV averages -8.4 dB with a VV/VH ratio of 6.1 dB. Both are consistent with rough vegetated terrain rather than urban structures, so the sensors agree."*
- Q: *"How much of this scene changed, and in which direction?"*
  A: *"7.4 % of the scene changed. NDBI rose from -0.12 to 0.08, consistent with new built-up surface."*

### 4.7 Deduplication and leakage control

**VRSBench is constructed from DOTA and DIOR imagery; DIOR-RSVG is constructed from DIOR. Overlap between §4.2 and §4.5 is near-certain, and cross-source leakage into the Phase 8 test split is the realistic failure mode.**

`build_corpus.py` therefore runs, in order:

1. Exact dedup on source-image SHA-256, across all sources.
2. Near-dup dedup on perceptual hash (pHash, Hamming `<= 5`), across all sources.
3. A hard assertion that no image in `train.jsonl` appears — by either hash — in any quarantined test split (VRSBench test, RSVQA test, CDVQA test, DIOR-RSVG test).

Step 3 failing is a **build error**, not a warning. A leaked benchmark is worse than no benchmark.

---

## 5. Corpus Composition

| Source | Samples | Share | Purpose | Track |
|---|---|---|---|---|
| `bigearthnet_v2` | 18,000 | 27.7 % | **mandate**; SAR + spectral physics grounding | A (sensor) |
| `vrsbench` | 20,000 | 30.8 % | VQA + caption + grounding at VHR | B (resolution) |
| `rsvqa_hr` | 10,000 | 15.4 % | counting / presence / comparison | B |
| `cdvqa` | 8,000 | 12.3 % | bi-temporal change VQA | B |
| `dior_rsvg` | 6,000 | 9.2 % | referring-expression grounding | B |
| `evidence_qa` | 3,000 | 4.6 % | citation behaviour | A |
| **Total** | **65,000** | | | |

This is Master.md §8-Phase-7's target table (59-64 k) plus the 3 k `evidence_qa` addition.

**Two-track rationale (Master.md §1):** Track A grounds the model in sensor physics using the mandated dataset at 10 m. Track B aligns it to the task formats and ~0.3-3 m resolution of the hidden Cartosat-2S evaluation set. Both are merged into one corpus and one adapter — swapping adapters at runtime would add latency and a failure mode for no measured benefit.

**Splits:** each source's official train split feeds `train.jsonl`; official val feeds `val.jsonl`; official test is quarantined for Phase 8 and never read by the corpus builder.

---

## 6. Training Profiles

### 6.1 Profile A (default) — `configs/train/qlora_qwen3vl8b_rocm24g.yaml`

```yaml
run_name: sq-lora-v1
base_model: Qwen/Qwen3-VL-8B-Instruct
attn_implementation: sdpa          # flash-attn CK is unreliable on gfx1100 (Master.md 6.2)
torch_dtype: bfloat16

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: bfloat16

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules_llm:    [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  target_modules_vision: [qkv, proj, fc1, fc2]
  vision_blocks: last_8            # ViT adaptation is what teaches SAR/index semantics

data:
  corpus: data/processed/corpus/train.jsonl
  val:    data/processed/corpus/val.jsonl
  max_views_per_sample: 6
  view_size_px: 448
  max_pixels: 200704               # 448*448 per view - the dominant VRAM knob
  max_seq_len: 4096

train:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16  # effective batch 16
  gradient_checkpointing: true
  num_train_epochs: 1
  learning_rate: 1.0e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  optim: paged_adamw_8bit
  max_grad_norm: 1.0
  bf16: true
  seed: 42
  dataloader_num_workers: 8        # 16 logical cores, leave headroom
  save_steps: 250
  eval_steps: 250
  logging_steps: 10
  resume_from_checkpoint: auto

env:
  HIP_VISIBLE_DEVICES: "0"         # hide the 7800X3D iGPU - critical (Master.md 2)
  ROCR_VISIBLE_DEVICES: "0"
  PYTORCH_HIP_ALLOC_CONF: expandable_segments:True
```

Expected: ~5.5 GB quantised weights, ~50 M trainable params, ~200 MB adapter, peak **18-21 GB** of 24 GB.

### 6.2 Profile B (fallback) — `configs/train/lora_qwen3vl4b_bf16.yaml`

Switch here if bitsandbytes NF4 misbehaves on gfx1100 (Master.md §9, risk row 3). Deltas only:

```yaml
base_model: Qwen/Qwen3-VL-4B-Instruct
quantization: { load_in_4bit: false }
train:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  optim: adamw_torch_fused
```

~8 GB bf16 weights, comfortable in 24 GB. **This is a config switch, not a rewrite** — `train_lora.py` is size- and quantisation-agnostic.

### 6.3 OOM playbook

Turn these knobs in order, one at a time:

1. `max_views_per_sample` 6 -> 4
2. `max_pixels` 448² -> 392² (153 664)
3. `max_seq_len` 4096 -> 3072
4. switch to Profile B

---

## 7. Run Schedule

### 7.1 The step-count arithmetic

Master.md §6.2 quotes *"~2.5 k steps ≈ 4-7 h"* alongside *"2 epochs"*. At effective batch 16, 2,500 steps is 40,000 sample-passes — which is 2 epochs of a **20 k** corpus, not of the 65 k corpus §5 specifies. Rather than silently pick one, we run both, which is the better engineering sequence anyway:

| Run | Corpus | Epochs | Steps | Est. wall time | Purpose |
|---|---|---|---|---|---|
| **1 — pilot** | 20,000 stratified subsample | 2 | ~2,500 | **4-7 h** | Validates pipeline, memory ceiling, and label plumbing. Produces an early ablation number. Matches Master.md's stated budget exactly. |
| **2 — full** | 65,000 | 1 | ~4,060 | **7-11 h** | The submitted adapter. |

Two overnight runs, comfortably inside Phase 7's 6-8 day allocation. Run 1 exists to catch a corpus bug before spending a full night on it — the failure mode we are buying insurance against is discovering a label mismatch (§2.5) after an 11-hour run.

### 7.2 Pre-rendering

Views are **pre-rendered to disk**, not rendered in the dataloader. Three reasons: reproducibility (the exact bytes trained on are on disk and hashable), throughput (the renderer is numpy-heavy and would starve the GPU), and reuse (the same rendered views are Phase 8 eval fixtures). Rendering is a one-time ~6 h CPU pass across 16 threads.

### 7.3 Augmentation for the Cartosat-2S / RISAT gap

Applied at corpus-build time, recorded in `meta.augmentations` so any result is reproducible:

| Augmentation | Applied to | Rate | Simulates |
|---|---|---|---|
| **Rescale** — resample to `U(1, 4)x` coarser (VHR) or `2x` finer (BEN) | vrsbench, dior_rsvg, bigearthnet_v2 | 30 % | the 0.3-20 m GSD span, bridging BEN's 10 m to Cartosat's 0.65 m |
| **SWIR dropout** — drop `SWIR` and `NDBI` views | bigearthnet_v2 | 20 % | Cartosat-2S's 4-band set (no SWIR) |
| **Single-pol** — drop VH, render `SARDB` only | bigearthnet_v2 | 20 % | RISAT single-polarisation acquisition |

The SWIR-dropout samples matter more than they look: they are what teach the model to reach for radar evidence when NDBI is unavailable — exactly the situation the hidden evaluation set puts it in.

---

## 8. Evaluation & Ablation Protocol

### 8.1 Primary mandate metric

BigEarthNet-v2 validation, **19-class** multi-label micro-F1, zero-shot vs adapted. This is the defensible evidence that requirement 1 was satisfied.

### 8.2 Reference-replication metric

BigEarthNet-v2 validation, **6-class merge**, micro-F1. Target trajectory ≈0.59 -> ≈0.83 (Master.md §6.2), subject to the honesty note in §4.1.

### 8.3 Evidence-audit metric

Fraction of generated answers in which **every** numeric span resolves to the FactSheet, measured by running `CitationValidator` (`AGENT_POLICY_DAG.md` §7.1) in `strict` mode over the val set. The reference study reports 0.9746. This metric is the direct measurement of whether `evidence_qa` (§4.6) did its job.

### 8.4 Downstream benchmarks (Phase 8)

| Benchmark | Metrics |
|---|---|
| VRSBench | VQA accuracy · caption BLEU-4 / ROUGE-L / CIDEr · grounding [email protected] |
| RSVQA-HR | per-type accuracy (presence, comparison, count, area) + aggregate |
| CDVQA | change-VQA accuracy |

### 8.5 The ablation table (Master.md §8-Phase-8)

| Row | Configuration |
|---|---|
| 1 | Generic VLM, zero-shot (no RS adaptation, no tools) |
| 2 | EarthDial-4B, zero-shot (published RS-VLM baseline) |
| 3 | Ours, zero-shot (Qwen3-VL-8B, no adapter, no tools) |
| 4 | **Ours, adapted + agentic tools** |

Row 3 -> row 4 isolates our contribution. Row 1 is the "monolithic generic VLM" the problem statement disqualifies — measuring it is how we demonstrate the disqualification is warranted rather than assumed. Every number reproducible from one `make eval`.

---

## 9. Budgets

### 9.1 Disk (377 GB available)

| Item | Size |
|---|---|
| BigEarthNet-v2 metadata (parquet) | < 1 GB |
| BigEarthNet-v2, 20 k selected S1+S2 patches | ~15 GB |
| VRSBench images | ~12 GB |
| RSVQA-HR images | ~15 GB |
| CDVQA / SECOND | ~5 GB |
| DIOR-RSVG images | ~8 GB |
| LEVIR-CD + OSCD (Phase 5) | ~4 GB |
| Pre-rendered views (65 k samples, mixed JPEG/PNG) | ~18 GB |
| Base weights (Qwen3-VL-8B bf16 + 4B fallback) | ~25 GB |
| Checkpoints (~200 MB adapter x ~20 saves) | ~4 GB |
| GGUF exports for the offline demo path | ~8 GB |
| **Total** | **~115 GB** |

Roughly 260 GB headroom. The archive-scale download Master.md §9 warns about is avoided entirely by the metadata-first selector (§4.1).

### 9.2 Time

| Stage | Wall time |
|---|---|
| Dataset downloads | ~4 h (network-bound) |
| Corpus build + pre-render | ~6 h (16 threads) |
| Run 1 — pilot | 4-7 h |
| Run 2 — full | 7-11 h |
| Merge + GGUF export | ~1 h |
| Evaluation sweep | ~4 h |

---

## 10. Resolved Ambiguities & Deltas from `Master.md`

| # | Item | Master.md state | Resolution | Rationale |
|---|---|---|---|---|
| 1 | Epoch/step arithmetic | §6.2: "2 epochs" and "~2.5 k steps"; §8-Phase-7: 59-64 k corpus | Two runs — 20 k x 2 epochs (pilot), 65 k x 1 epoch (full) | The two figures are only jointly consistent with a ~20 k corpus. Pilot-then-full is the better sequence regardless, and Run 1 matches the quoted budget exactly. |
| 2 | `evidence_qa` source | Not present | **Added, 3 k samples** | No public RS corpus teaches citation behaviour. Without it the adapter learns RS vocabulary but still invents numbers, defeating the FactSheet-constrained architecture. Generated from the FactSheet, so supervision is free and exact. |
| 3 | Index normalisation | Not specified | **Fixed domains**, never per-image | Per-image stretch destroys absolute semantics; the model could never learn what NDVI 0.7 means. Highest-consequence detail in the recipe. |
| 4 | View file formats | Not specified | PNG for index/SAR, JPEG q92 for composites | JPEG chroma subsampling visibly shifts colormapped index values. |
| 5 | Missing SWIR on Cartosat | Not addressed | `NDBI`/`SWIR` marked **unavailable**; never substituted; SAR carries built-up evidence instead | Silent substitution yields confidently wrong built-up answers on the hidden set. Motivates the cross-modal path. |
| 6 | Label-string ownership | Not specified | Single source of truth, `render/view_labels.py`, imported by corpus builder and runtime | A train/inference label drift silently destroys the adaptation while all tests still pass. |
| 7 | Box format ownership | Not specified | Single serialiser, `models/prompts/box_format.py` | Same failure class as #6, for grounding. |
| 8 | VRSBench / DIOR-RSVG overlap | Not addressed | pHash dedup + hard assertion against quarantined test splits | Both derive from DIOR/DOTA imagery; benchmark leakage is the realistic failure mode, and a leaked benchmark is worse than none. |
| 9 | Published 0.59 -> 0.83 target | §6.2 quotes it as the target | Replicate the protocol; if the exact 6-class merge is unrecoverable, report as comparable-not-identical and anchor the claimed delta to **our own** measured zero-shot baseline | Claiming parity with an unreproduced protocol would not survive a judge's question. |
| 10 | Pre-render vs on-the-fly | Not specified | **Pre-render** to disk | Reproducibility, dataloader throughput, and reuse as Phase 8 eval fixtures. |
| 11 | Resolution/band augmentation | §1 describes two-track intent | Concrete rates in §7.3 | Makes the Cartosat/RISAT mitigation (Master.md §9, highest-likelihood risk) executable and measurable. |
