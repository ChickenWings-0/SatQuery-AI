# 02 — Current State (as of 2026-09-11, HEAD `2cb0348`)

Everything here was either measured on the dev box or read from the repo. Where a
number comes from the independent audit (`DOCS/project_audit.md`, dated 2026-09-11,
run against `77bea25`) it is marked *(audit)*.

## Git history (complete)

| Commit | Date | Subject |
|---|---|---|
| `ca22b20` | 2026-09-05 | Initial commit (only `.gitattributes`) |
| `290d93a` | 2026-09-05 | doc files (Master.md, API_CONTRACT, AGENT_POLICY_DAG, DATA_ADAPTATION_PLAN, OPEN_SOURCE_ASSETS) |
| `5b20b70` | 2026-09-06 | Phases 0–6 backend: API, ingestion, rendering, DAG controller, Siamese CD, DOFA fusion |
| `44e6947` | 2026-09-06 | Phase 7: VRAM estimator recalibrated, hermetic tests, `evidence_qa` builder, overnight script |
| `77bea25` | 2026-09-11 | "Phase 8": corpus dedup + local sources, bf16 serving backend, async jobs API, mission-control frontend |
| `2cb0348` | 2026-09-11 | documentation update (`repo documentation/` + `DOCS/project_audit.md`) |

Working tree is clean. Only `main` exists. No tags, no CI, no remote configured
in the snapshot (check `git remote -v`).

## Phase status vs `Master.md §8`

| Phase | Objective | Status | Evidence |
|---|---|---|---|
| −1 | Fedora + ROCm bring-up | ✅ | gfx1100 detected; torch 2.9.1+rocm6.4 in `.venv` |
| 0 | Frozen contract, mocked API | ✅ | `openapi.json` and `schema.d.ts` regenerate byte-identical *(audit)* |
| 1 | Ingestion + compatibility | ✅ | 56 tests in `test_ingest.py`; 10 named checks in `schemas/enums.CheckName` |
| 2 | Rendering + artifact store | ✅ | 77 tests in `test_render.py`; content-addressed blob store under `data/artifacts/` |
| 3 | Registry / planner / executor / trace | ✅ | 15 tools in `configs/registry.yaml`; 22 policy entries; SQLite trace store |
| 4 | VLM serving (HF bf16 + llama.cpp) | ✅ | both backends; but the GGUF was exported from `poc-v1`, not `full-epoch-v1` |
| 5 | Bi-temporal change detection | ⚠️ partial | LEVIR-CD ResNet-18 Siamese: **F1 0.858 vs 0.88 gate**; no OSCD run (data downloaded) |
| 6 | Cross-modal + grounding | ✅ code | DOFA tool, physics agreement, text grounding, segmenter, counter; 62 tests. DIOR-RSVG training data unresolved (gated) |
| 7 | Corpus + QLoRA | ⚠️ infra done, **result not** | 19 h run completed; loss flat from step ~90; BEN-only corpus; no ablation |
| 8 | Benchmark evaluation | ❌ not started | no `src/satquery/eval/`; `scripts/eval_vrsbench_zeroshot.py` defaults to a 5-item mock |
| 9 | Hardening / demo / Docker / CI | ❌ not started | no Dockerfile, Makefile, `.github/`, `make demo` |
| FE | Frontend (originally a teammate track) | ✅ ahead of plan | mission-control console, SSE streaming, DAG dialog, swipe viewer, bbox overlay, offline MSW mock |

## Measured numbers you can quote

**Tests** *(audit, re-verified counts by grep)*:
- Backend: **503 pytest cases**, all green, ~4 min wall (the `test_models.py` case
  exercising real SDPA on ROCm is the long pole). Files: `test_vlm_training` 103,
  `test_render` 77, `test_crossmodal_grounding` 62, `test_ingest` 56,
  `test_change_detection` 44, `test_models` 41, `test_agent` 30, contract 39.
- Frontend: **194 vitest cases** in 15 files, ~4.6 s. `tsc -b` clean. `vite build`
  clean (~1.8 s). `npm run lint` (oxlint) crashes on this checkout — missing native
  binding, not a code problem.
- Hygiene *(audit)*: `mypy --strict` 1 error (`training/local_sources.py:132`,
  `no-any-return`); `ruff` 1 error (import sort in `scripts/test_inference.py`).

**Training runs** (all under gitignored `runs/`, all Qwen3-VL-8B-Instruct, NF4 QLoRA
r=16 α=32, last-8 ViT blocks + all LLM projections, effective batch 16, lr 1e-4
cosine, seed 42):

| Run | Corpus | Samples | Steps | Peak VRAM | Outcome |
|---|---|---|---|---|---|
| `sq-lora-v1-sanity` (2026-09-06) | `train.jsonl` truncated | 100 | 6 | 21.49 GiB | sanity check; `max_pixels` 200704 |
| `throughput-probe-20260906-221333` | `probe.train.jsonl` | 2,853 | 178 | 18.42 GiB | first throughput probe |
| `poc-v1` (2026-09-08) | `poc/train.jsonl` | 12,202 | 762 | 16.97 GiB | PoC; **this** adapter was merged and exported to GGUF (`runs/poc-v1/export*.gguf`) |
| `throughput-probe` (2026-09-09) | `probe.train.jsonl` | 18,530 planned, stopped at epoch 0.17 | 200 | 16.77 GiB | **0.265 samples/s** → ~68–73 h per 65k epoch |
| **`full-epoch-v1`** (finished 2026-09-10 11:46 UTC) | `full.train.jsonl` | **18,530** | **1,158** | **16.78 GiB** | the "production" adapter; ~19 h; flat loss (see below) |

**The flat loss** *(audit, from `full_training_run.log`)*:

| epoch | train loss | grad_norm | mean_token_accuracy |
|---:|---:|---:|---:|
| 0.008 | 13.01 | 29.9 | 0.26 |
| 0.078 | 3.82 | 0.08 | 0.50 |
| 0.49 | 3.82 | 0.06 | 0.50 |
| 0.99 | 3.76 | 0.05 | 0.51 |

Eval loss 3.935 → 3.930 over the epoch. Root cause identified: **loss is computed over
the whole sequence** — no `assistant_only_loss` / `completion_only_loss` in
`training/vlm/qlora.py::sft_config_kwargs()`. The ~1,000-token system prompt plus a
FactSheet of up to 30 unpredictable floats dominates the token count. See `08` #1.

**Inference probe** (`runs/full-epoch-v1/inference_probe*.json`, one held-out BEN
patch, bf16 + adapter, 18.67 GiB peak): grounding returned a single full-frame box
`(0,0),(1000,1000)`; a discrete probe reports IoU 0.66. Smoke test, not a result.

**Change detector** (`data/checkpoints/cd/levircd_resnet18.ckpt.json`): ResNet-18
Siamese, SSL4EO-S12 encoder init, 60 epochs @ 256 px, bf16-mixed. Test metrics:
**F1 0.858, IoU 0.751, P 0.864, R 0.852**, F1@0.5 0.839; calibrated threshold 0.9.
Below the 0.88 gate — the JSON says so in plain text. A normalisation bug (measured
stats recorded but never applied) was found and corrected to identity. `.bak-*`
files are the pre-fix checkpoints.

**Corpus on disk** (`data/processed/corpus/`, gitignored):

| File | Lines | Sources | Notes |
|---|---:|---|---|
| `train.jsonl` | 15,530 | bigearthnet_v2 only | 0 rows carry `image_sha256` *(audit)* |
| `val.jsonl` | 2,470 | bigearthnet_v2 only | |
| `evidence_qa.train.jsonl` | 3,000 | synthetic | |
| `evidence_qa.val.jsonl` | **0** | — | no held-out citation supervision |
| `full.train.jsonl` | 18,530 | BEN + evidence_qa | **what `full-epoch-v1` trained on** |
| `probe.train.jsonl` | 18,530 | same | throughput probe copy |
| `poc/train.jsonl`, `poc/val.jsonl` | 12,202 / 3,915 | | PoC subset |

Rendered views exist on disk for BEN (`data/processed/views/bigearthnet_v2/`),
VRSBench (29,615 tiles, `views/vrsbench/rendered.jsonl`) and CDVQA (271 pairs,
`views/cdvqa/`). VRSBench and CDVQA were rendered but **never entered any corpus**.

**Serving** *(measured)*: bf16 base ~16.4 GB weights; ~18.7 GiB peak with six views
and the adapter; VRAM guard budget 22 GB. llama.cpp Q4_K_M path ~6–9 GB.

## What the current `.env` on the dev box says

```
SATQUERY_VLM_BACKEND=llamacpp
SATQUERY_VLM_SERVER_URL=http://127.0.0.1:8080
SATQUERY_SEG_CHECKPOINT=data/checkpoints/seg/segformer-b5-loveda
```

i.e. the box is currently configured for the **llama.cpp path (stock/PoC GGUF,
no full-epoch adapter)** and the LoveDA SegFormer-B5. `SATQUERY_CD_CHECKPOINT` is not
set in `.env`, so `siamese_change_detector` is *not servable* from `.env` alone and
bi-temporal plans degrade to `image_diff_change` unless the env var is exported.
(The CD checkpoint exists at `data/checkpoints/cd/levircd_resnet18.ckpt.pt`.)

## Known defects, ranked (full detail in `08`)

1. **Training objective** — whole-sequence loss; flat curve; adapter effect unproven.
2. **Train/serve prompt skew** — training put image labels in a non-text `label`
   key (ignored by the chat template); serving inserts labels as text parts before
   each image. The adapter was trained on a layout the serving code calls unreliable.
3. **No ablation / no eval harness** — the rubric's headline claim is unmeasured.
4. **Corpus is BEN-only and unhashed** — dedup ran on nothing for the trained
   source; VRSBench/CDVQA/RSVQA-HR/DIOR-RSVG absent.
5. **Citation match is by value, not key** — `0.75 [wrong.key]` passes.
6. **`ingest()` blocks the event loop** in `/v1/analyze` and `/v1/jobs`.
7. **GPU semaphore is per-request**, not per-process.
8. **Jobs are in-memory; cancel is client-only** (no `DELETE /v1/jobs/{id}`).
9. **Frontend re-runs the job on SSE drop** instead of reattaching.
10. **CD below gate, single resolution** (no OSCD run).
11. **No CI, no Docker, no Makefile; root `README.md` is 0 bytes;
    `src/satquery_ai/` is a leftover scaffold.**

## What is genuinely strong (do not regress these)

- The evidence/citation chain: FactSheet → CitationValidator (unit-aware, segment
  matching, tolerance pivot at 10, year/index exclusions) → named confidence caps
  → trace. Build-time citation audit of the corpus.
- Contract chain: `API_CONTRACT.md` → `openapi.json` → `schema.d.ts`, byte-identical
  on regeneration, locked by contract tests on both sides.
- Deterministic planner + async executor with per-step timeouts, CPU/GPU
  semaphores, content-hash cache, one-level fallbacks, honest `DEGRADED`.
- bf16 serving with a VRAM guard that re-checks after load; box sentinels survive
  decoding; stop sequences applied post-generation.
- Training harness: pydantic profiles, name-based ViT block resolution, VRAM
  estimator within 0.4 GiB of measured, resumable checkpoints with optimiser
  state, graceful SIGINT, complete `run_manifest.json`.
- Frontend: Tailwind v4 done idiomatically, contrast-tested tokens, letterbox-correct
  SVG bbox overlay in Qwen's 0–1000 frame, pure SSE reducer tested on a recorded run,
  MSW offline mode, self-hosted metric-matched fonts.
