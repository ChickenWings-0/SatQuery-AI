# 01 — Project Context

## What this is

**SatQuery AI** — an agentic vision-language assistant for remote sensing. Tagline:
*"From Space to Answers — geospatial intelligence that shows its work."*

- **Origin:** built as an entry for Smart India Hackathon (SIH) 2026, problem statement **26167**,
  sponsor **ISRO / Space Applications Centre (SAC)**: "Agentic Vision-Language
  Assistant for Remote Sensing".
- **Team shape:** one owner builds everything — backend, ML, data engine, frontend.
  Early plans mention "a teammate builds the frontend"; ignore that.
- **Timeline:** work began 2026-09-05 (`ca22b20`). Feature work closed 2026-09-18
  (`1782163`); the GGUF export was produced 2026-09-22; on 2026-09-29 the training
  data was deleted to free disk and the docs were consolidated. On 2026-09-29 the
  owner stopped treating it as an SIH entry: it is now a **personal project** under
  Apache-2.0. The rubric below still explains why the system is shaped as it is.
- **Deployment target:** fully local, offline-capable. Two machines:
  1. the **24 GB ROCm dev box** (bf16 base + LoRA adapter via transformers), and
  2. an **8 GB RTX 4070 Windows laptop** (merged Q4_K_M GGUF via `llama-server`,
     no Python ML stack) — `DOCS/DEMO_LAPTOP_RUNBOOK.md`.
  Must survive "kill the network at judging"; `?mock=1` is the last-resort fallback.

## What the problem statement rewards (and disqualifies)

From `DOCS/Master.md §1`:

- The rubric **explicitly disqualifies monolithic generic VLMs.** The scored novelty
  is the *agentic, query-driven framework*: a controller that parses the query,
  validates geospatial input compatibility, selects and sequences specialist tools
  from a registry, fuses their outputs, and returns an **evidence-grounded answer
  plus an auditable execution trace**.
- **Internal chain-of-thought is not evaluated** — only the observable trace is.
- Mandatory requirements (numbering used throughout the docs and code comments):
  1. **Domain adaptation** of a VLM on the mandated dataset (BigEarthNet).
  2. **VQA + captioning**, with **grounding** (boxes) or **segmentation** as an option.
  3. **Bi-temporal change** — change VQA *and* spatial change maps.
  4. **Cross-modal optical + SAR reasoning.**
  5. **Input validation / compatibility checking** of geospatial inputs.

How each is met today:

| Req | Where | Evidence |
|---|---|---|
| 1 | `runs/sq-lora-v2-full/adapter` (BigEarthNet-v2 + VHR sources) | zero-shot vs adapted answer-token accuracy 26.4 % → 81.8 %; `runs/eval/sq-lora-v2-full/results.md` |
| 2 | `vlm_vqa`, `vlm_caption`, `text_grounding`, `semantic_segmenter` | VQA 83.4 %, grounding R@0.5 48.9 %, box format 100 % |
| 3 | `siamese_change_detector` / `image_diff_change` + `change_statistics` + `vlm_change_vqa` | CDVQA 77 %; LEVIR-CD F1 0.858 |
| 4 | `spectral_index_analyzer` ∥ `sar_backscatter_analyzer` → `physics_agreement`, `crossmodal_consistency` (DOFA) | cross-modal VQA 81.4 %, fact recall 100 % |
| 5 | `ingest/compatibility.py`, 10 named checks, pre-flight UI | e2e negative scenario (`INSUFFICIENT_OVERLAP`) |

Two consequences drive every decision:
1. **The trace is a first-class product, not logging.**
2. **The VLM is one tool among many, not the system.**

## The hidden-evaluation gap (drives model and data choices)

The ISRO/SAC hidden test set is **Cartosat-2S optical (~0.65 m pan / ~1.6 m MX,
4 bands, no SWIR) + RISAT SAR (single-pol)**. The mandated training dataset is
**BigEarthNet (Sentinel-1/2, 10 m, 12 bands)** — a 10–15× GSD gap and a different
band set. The answer is a **two-track corpus merged into one adapter**:

- **Track A (sensor physics, mandate):** BigEarthNet-v2 at 10 m, plus synthetic
  `evidence_qa` samples that teach fact-sheet citation.
- **Track B (resolution/task alignment):** VRSBench, RSVQA-HR, CDVQA at VHR.
  DIOR-RSVG stayed out (gated on HF); LEVIR-CD trains the change detector instead.

The v2 corpus contains both tracks (`06`). No Cartosat/RISAT augmentation was done.

## The design thesis, in five decisions

1. **Spectral rendering + LoRA instead of modifying patch embeddings.** Each N-band
   raster is rendered into a catalogue of named 3-channel views (true colour, FCIR,
   SWIR, NDVI/NDWI/NDBI on a fixed −1…+1 domain, SAR false colour, SAR dB) fed
   through the model's multi-image interface, each preceded by its label as text.
   Zero architecture change across sensors; the same renderer produces the evidence
   PNGs the UI shows.
2. **Deterministic policy-table planner, not free-form ReAct.** Reproducible,
   sub-second planning, unit-testable routing, no invented tools. The LLM does
   *slot filling* and *answer synthesis* only.
3. **FactSheet + CitationValidator.** Every tool scalar is keyed `<tool>.<scalar>`
   with a unit. Every numeric span the VLM emits must resolve to a sheet entry
   (the `[key]` marker if present, else by value) with matching units; unresolved
   spans land in `uncited_numeric_spans` and cap confidence. Default policy `flag`.
4. **Honest degradation everywhere.** Missing checkpoint → tool drops out and plans
   use the declared fallback, recorded `DEGRADED`. VLM unavailable → deterministic
   template answer, `template_fallback: true`, capped at 0.65. Never a 500 for a
   recoverable step.
5. **One machine, one box.** In-process job store, SQLite trace store, filesystem
   artifact store, no broker. Docker images exist for packaging, but the demo runs
   natively.

## Confirmed decisions (owner, recorded in Master.md §2 and later)

- Base VLM: **Qwen3-VL-8B-Instruct + own LoRA**.
- Training: **100 % local** on the RX 7900 XTX via ROCm; no cloud.
- Serving: local-first; bf16 + adapter on the dev box, merged Q4_K_M GGUF on the laptop.
- Python pinned to **3.11**; package manager **uv**; Node ≥ 22.12.
- **No OpenMMLab / mmcv** (ROCm build trap). Pure PyTorch + torchgeo + Lightning for CD.
- Attention: **sdpa** (flash-attn CK unreliable on RDNA3).
- Maps: **MapLibre**, not Cesium; imagery from Microsoft Planetary Computer STAC,
  fetched server-side (`POST /v1/imagery/fetch`), online features off by default.
- Repository policy: **one branch, `main`**.

## Vocabulary you will see everywhere

| Term | Meaning |
|---|---|
| `TaskType` | VQA, CAPTION, GROUNDING, SEGMENTATION, COUNT, SCENE_CLASSIFY, CHANGE_VQA, CHANGE_CAPTION, CHANGE_MAP, CROSS_MODAL_VQA, CROSS_MODAL_COMPARE, UNSUPPORTED |
| `PairType` | SINGLE, CROSS_MODAL, BI_TEMPORAL, INCOMPATIBLE |
| `Modality` | optical, sar, panchromatic, unknown |
| `InputManifest` | everything read from one upload (driver, CRS, GSD, bands, dtype, nodata, time, warnings) |
| `CompatibilityReport` | the 10-check battery over a pair + `pair_type` + `overall` + `actions_taken` + `common_grid` |
| policy key | `"{TaskType}\|{PairType}\|{ModalityKey}"`, e.g. `CHANGE_VQA\|BI_TEMPORAL\|optical` |
| FactSheet | union of all tool scalars, keyed `<tool>.<scalar>` — the *only* numbers allowed in an answer |
| view label | the exact string the model sees for a view, e.g. `Image 3 (optical false colour infrared, pre-change, Sentinel-2, 2019-04-12)` |
| layout | `label-before-image/v1` — label as a text part before each image; fingerprinted in the adapter's `layout.json` |
| `AuditTrace` | the full run record; schema version `1.0` |
| caps | named confidence ceilings (`uncited_claims → 0.60` etc.) applied in a frozen order |
| Phase −1 … 9 | roadmap stages in `Master.md §8` (all closed) |
| Track 1–4 | the pre-final tracks in `ROADMAP_REMAINING_FIXES.md`: eval suite, GGUF export, e2e parity, UI (SITREP / GeoJSON / STAC) — all landed |
| Tier A / B / C | demo network tiers: internet + API / local API only / nothing (`?mock=1`) |
| SITREP | the one-page PDF brief generated client-side from a trace |
| v1 / v2 | the failed first adapter (`full-epoch-v1`, deleted) / the production adapter (`sq-lora-v2-full`) |
