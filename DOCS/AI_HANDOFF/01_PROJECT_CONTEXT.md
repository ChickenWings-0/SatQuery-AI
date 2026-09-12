# 01 — Project Context

## What this is

**SatQuery AI** — an agentic vision-language assistant for remote sensing. Tagline used
in the pitch: *"From Space to Answers — geospatial intelligence that shows its work."*

- **Competition:** Smart India Hackathon (SIH), problem statement **26167**, sponsor
  **ISRO / Space Applications Centre (SAC)**: "Agentic Vision-Language Assistant for
  Remote Sensing".
- **Team shape:** one person (the repo owner) builds the backend, ML, data engine and
  — as of Phase 8 — also the frontend. Early plans refer to "a teammate builds the
  frontend"; that is no longer how the work is split. Treat the whole repo as one
  owner's.
- **Timeline stated in the master plan:** 1–2 months from 2026-09-05. Work began
  2026-09-05 (`ca22b20 Initial commit`); today is 2026-09-11.
- **Deployment target:** fully local, offline-capable, one 24 GB consumer GPU. Must
  survive "kill the network at judging".

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

Two consequences drive every decision:
1. **The trace is a first-class product, not logging.** Versioned, schema-validated,
   built by the executor, not reconstructed from logs.
2. **The VLM is one tool among many, not the system.** Every claim in the answer
   must be traceable to a deterministic tool output — the anti-hallucination story
   and the differentiator.

## The hidden-evaluation gap (drives model and data choices)

The ISRO/SAC hidden test set is **Cartosat-2S optical (~0.65 m pan / ~1.6 m MX,
4 bands, no SWIR) + RISAT SAR (single-pol)**. The *mandated* training dataset is
**BigEarthNet (Sentinel-1/2, 10 m, 12 bands)** — a 10–15× GSD gap and a different
band set. The plan's answer is a **two-track corpus merged into one adapter**:

- **Track A (sensor physics, mandate):** BigEarthNet-v2 at 10 m, plus synthetic
  `evidence_qa` samples that teach fact-sheet citation.
- **Track B (resolution/task alignment):** VHR corpora — VRSBench, RSVQA-HR, CDVQA,
  DIOR-RSVG, LEVIR-CD at 0.5 m.

**Reality check (see `02`, `06`):** the adapter that exists (`runs/full-epoch-v1`)
was trained on Track A only (BigEarthNet + evidence_qa, 18,530 samples). Track B
never entered the corpus. This is the single biggest gap between plan and artifact.

## The design thesis, in five decisions

1. **Spectral rendering + LoRA instead of modifying patch embeddings.** Each
   N-band raster is rendered into a small catalogue of named, interpretable
   3-channel views (true colour, false-colour IR, SWIR, NDVI/NDWI/NDBI heatmaps on a
   fixed −1…+1 domain, SAR false colour, SAR dB) and fed through the model's
   existing multi-image interface with explicit names in the prompt. One decision
   pays three times: zero architecture change across Sentinel-2/Sentinel-1/
   Cartosat/RISAT; the same renderer produces the evidence PNGs the UI shows; and a
   published recipe on exactly this idea reports BigEarthNet-v2 micro-F1
   0.59 → 0.83 (the target "we adapted a VLM" evidence).
2. **Deterministic policy-table planner, not free-form ReAct.** Byte-identical
   reruns, sub-second planning, unit-testable routing, and no failure mode where an
   LLM invents a tool. The LLM does *slot filling* (structured JSON) and *answer
   synthesis* — never control flow.
3. **FactSheet + CitationValidator.** Every scalar a tool produces is keyed
   `<tool>.<scalar>` with a unit. The VLM is shown the sheet; every numeric span it
   emits must resolve to a sheet entry with matching units; unresolved spans land in
   `uncited_numeric_spans` and cap confidence. Default policy is `flag`, not `strip`.
4. **Honest degradation everywhere.** Missing checkpoint → tool drops out of the
   registry and plans degrade to the declared fallback, recorded `DEGRADED`. VLM
   unavailable → deterministic template answer, `template_fallback: true`, capped
   at 0.65. Never a silent substitution, never a 500 for a recoverable step.
5. **One machine, one box.** In-process job store, SQLite trace store, filesystem
   artifact store, no broker, no Docker (yet). Chosen for a hackathon demo, and
   listed as debt in `08`.

## Confirmed decisions (from the owner, recorded in Master.md §2)

- Base VLM: **Qwen3-VL-8B-Instruct + own LoRA** (native normalised-bbox grounding,
  real multi-image interface, llama.cpp support).
- Training: **100 % local** on the RX 7900 XTX via ROCm; no cloud.
- Serving: **local-first**; cloud optional; must survive a dead network.
- Python pinned to **3.11** (3.14 wheels unreliable). Package manager: **uv**.
- **Do not use OpenMMLab / mmcv** (ROCm build trap). Pure PyTorch + torchgeo +
  Lightning for change detection.
- Attention: **sdpa** (flash-attn CK unreliable on RDNA3).

## Vocabulary you will see everywhere

| Term | Meaning |
|---|---|
| `TaskType` | resolved analysis task: VQA, CAPTION, GROUNDING, SEGMENTATION, COUNT, SCENE_CLASSIFY, CHANGE_VQA, CHANGE_CAPTION, CHANGE_MAP, CROSS_MODAL_VQA, CROSS_MODAL_COMPARE, UNSUPPORTED |
| `PairType` | SINGLE, CROSS_MODAL, BI_TEMPORAL, INCOMPATIBLE |
| `Modality` | optical, sar, panchromatic, unknown |
| `InputManifest` | everything read from one upload (driver, CRS, GSD, bands, dtype, nodata, acquisition time, warnings) |
| `CompatibilityReport` | the 10-check battery over a pair + `pair_type` + `overall` + `actions_taken` + `common_grid` |
| policy key | `"{TaskType}\|{PairType}\|{ModalityKey}"`, e.g. `CHANGE_VQA\|BI_TEMPORAL\|optical` |
| FactSheet | union of all tool scalars, keyed `<tool>.<scalar>` — the *only* numbers allowed in an answer |
| view / view label | one rendered 3-channel image and the exact string the model sees for it, e.g. `Image 3 (optical false colour infrared, pre-change, Sentinel-2, 2019-04-12)` |
| `AuditTrace` | the full run record; schema version `1.0` |
| caps | named confidence ceilings (`uncited_claims → 0.60` etc.) applied in a frozen order |
| Phase −1 … 9 | the roadmap stages in `Master.md §8` |
| Profile A / B | training configs: A = QLoRA NF4 on 8B (default); B = bf16 LoRA on 4B (OOM fallback) |
| Track A / B | corpus tracks: sensor-physics (BigEarthNet) / VHR alignment |
| "the seam" | shorthand from the audit for script-level glue that the library tests do not cover (e.g. `hashes_for` in `build_corpus.py`) |
