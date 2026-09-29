# 02 — Current State (as of 2026-09-29, HEAD `1782163`)

Everything here was measured on the dev box or read from the repo on the snapshot
date.

## Git history

| Commit | Date | Subject |
|---|---|---|
| `ca22b20` | 2026-09-05 | Initial commit |
| `290d93a` | 2026-09-05 | doc files (Master, API_CONTRACT, AGENT_POLICY_DAG, DATA_ADAPTATION_PLAN, OPEN_SOURCE_ASSETS) |
| `5b20b70` | 2026-09-06 | Phases 0–6 backend: API, ingestion, rendering, DAG controller, Siamese CD, DOFA fusion |
| `44e6947` | 2026-09-06 | Phase 7: VRAM estimator, hermetic tests, `evidence_qa` builder, overnight script |
| `77bea25` | 2026-09-11 | corpus dedup + local sources, bf16 serving, async jobs API, mission-control frontend |
| `2cb0348` | 2026-09-11 | documentation update + independent audit |
| `72a0da3` | 2026-09-12 | remediation plan: concurrency, key-aware citations, `DELETE /v1/jobs/{id}`, CI, Docker, Makefile |
| `0bada6e` | 2026-09-12 | ML recovery plan steps 0–5 (loss masking, shared prompt layout, formatting) |
| `662740a` | 2026-09-15 | frontend revamp (landing, router, sections), v2 corpus rebuild, full QLoRA profile |
| `d10ca92` | 2026-09-15 | roadmap for the four pre-final tracks, README refresh |
| `91d8292` | 2026-09-16 | Tracks 1–4: eval suite, GGUF export tooling, API parity, SITREP / GeoJSON / STAC UI |
| `b2b87e9` | 2026-09-16 | NaN-weight scanner, degenerate-output tripwire, final QA checklist, first eval results |
| `be14970` | 2026-09-16 | re-run eval (`results.*` updated) |
| `bf0fbbe` | 2026-09-18 | user guide, sidebar history, Maps fixes, recorded S1 `.tif` fixtures |
| `1782163` | 2026-09-18 | Playwright overlay spec fix |

**Uncommitted on 2026-09-29:** `.claude/` untracked + ignored; all Markdown moved into
`DOCS/` (`repo documentation/` removed); this bundle rewritten; obsolete docs deleted
(`09`). `frontend/public/sitemap.xml` is dirty only because a local build rewrote its
`lastmod` — do not commit it.

## Phase / track status

| Stage | Status | Evidence |
|---|---|---|
| Phases −1 … 7 | ✅ | see `Master.md §8`; Phase 7 re-done as the v2 run |
| Phase 8 — benchmark eval | ✅ | `src/satquery/eval/`, `scripts/eval_benchmark.py`, `make eval`; results committed under `runs/eval/sq-lora-v2-full/` |
| Phase 9 — hardening | ✅ | `.github/workflows/ci.yml` (backend · frontend · e2e · docker), `Makefile` (`make ci`, `demo`, `eval`, `e2e`…), `Dockerfile` + `frontend/Dockerfile` + `docker-compose.yml`, pre-commit |
| Remediation plan (non-ML audit findings) | ✅ | concurrency module, key-aware validator, `to_thread` ingest, cancel end-to-end, SSE reattach, `BBOX_SET` consumption |
| ML recovery plan | ✅ | prompt-completion records + `completion_only_loss`, shared `layout.py`, mask audit, layout fingerprint check at load |
| Track 1 — eval suite | ✅ | 500-sample run 2026-09-16 (numbers below) |
| Track 2 — merge + GGUF | ✅ | `models/` produced 2026-09-22 (below); laptop runbook written |
| Track 3 — e2e parity | ✅ | `scripts/e2e_parity.py`, `make e2e`; last run 2026-09-15 (below) |
| Track 4 — UI | ✅ | SITREP PDF, RFC 7946 GeoJSON export, Maps STAC discovery HUD; Playwright `track4.spec.ts` offline |
| Manual QA (`FINAL_QA_CHECKLIST.md`) | ❌ not recorded | sign-off sheet is empty |

## Measured numbers you can quote

**Tests (2026-09-29):** backend **623** pytest cases in 17 files (collected);
frontend **356** vitest cases in 37 files, all passing (4.6 s); Playwright
`frontend/e2e/{overlay,track4}.spec.ts`. Gates are `make ci` (ruff check, mypy
--strict, pytest, oxlint, tsc, vitest, vite build, contract both halves). Entry
bundle 173.7 KB against a 180 KB budget (`npm run check:bundle`).

**Production adapter — `runs/sq-lora-v2-full/adapter`** (run finished
2026-09-15 05:38 UTC):

| | |
|---|---|
| Base | Qwen3-VL-8B-Instruct, NF4 double-quant backbone during training, bf16 compute |
| LoRA | r=16, α=32, dropout 0.05; 7 LLM projections × 36 layers + `qkv,proj,fc1,fc2` of the last 8 ViT blocks; 44.5 M trainable params |
| Corpus | 53,098 train / 5,902 val, `data/processed/corpus/v2-full/` (**deleted 2026-09-29**) |
| Schedule | 1 epoch, 3,318 steps, batch 1 × 16, lr 1e-4 cosine, 3 % warmup, `paged_adamw_8bit`, seed 42 |
| Wall / VRAM | 146,348 s (≈ 40.7 h), 0.363 samples/s; **peak 13.42 GiB** (estimate 17.1 GiB) |
| Mask audit | 32/32 ok; prompt 694–2,192 tok, completion 3–97 tok, 0 truncated |
| Layout | `label-before-image/v1`, 144 image tokens per 448 px view, fingerprint in `adapter/layout.json` |
| Zero-shot (same val subset, 256 samples) | eval loss **8.466**, answer-token accuracy **26.4 %** |
| Adapted, epoch 1.0 | train loss **0.3025**, eval loss **0.1867**, answer-token accuracy **81.8 %** |

The zero-shot row is the base model scored by the trainer before step 1 on the same
eval subset — that *is* the domain-adaptation ablation. The benchmark-level baseline
(`make eval-baseline`) was never run.

**Benchmark — `runs/eval/sq-lora-v2-full/results.md`** (2026-09-16, 500 held-out val
samples, 100 per source, backend `hf`, bf16 + adapter, git `b2b87e9`, seed 42):

| Source | Accuracy | Grounding R@0.5 | Mean IoU | Citation precision | Uncited-number rate |
|---|---|---|---|---|---|
| BigEarthNet-v2 | 53.3 % | 40.0 % | 41.6 % | 100 % | 0 % |
| CDVQA | 77.0 % | — | — | — | — |
| Evidence QA | 87.0 % | — | — | 100 % | 0 % |
| RSVQA-HR | 91.0 % | — | — | 100 % | 0 % |
| VRSBench | 88.0 % | 56.0 % | 52.6 % | 100 % | 0 % |
| **All** | **80.7 %** | **48.9 %** | **47.7 %** | **100 %** | **0 %** |

Per task: COUNT 100 %, VQA 83.4 %, CROSS_MODAL_VQA 81.4 % (fact recall 100 %),
CHANGE_VQA 77 %, box format valid 100 %, captions BLEU-4 13.7 / ROUGE-L 36.7,
**SCENE_CLASSIFY 5 % exact / 50.2 % label-F1** (multi-label BigEarthNet classes are
the weak spot and drag the BEN row down). Mean latency 5.8 s/sample, 0 generation
errors, 0 truncations. Worst cases rendered under `runs/eval/sq-lora-v2-full/confusion/`
(gitignored).

**Export — `models/`** (2026-09-22, `merge_manifest.json` records adapter SHA-256
`7785f9eb…`):

| File | Size |
|---|---|
| `sq-lora-v2-full-merged/` | bf16 merged safetensors, 17 GB, 8.77 B params |
| `sq-lora-v2-full-merged-Q4_K_M.gguf` | 5.0 GB |
| `sq-lora-v2-full-merged-mmproj-f16.gguf` | 1.16 GB (vision tower not quantised on purpose) |
| `SHA256SUMS` | checksums for the three above |

Laptop serving settings baked into the manifest: `SATQUERY_VLM_MAX_VIEWS=3`,
`--image-min-tokens 1024`, context 8192. The Q4_K_M build has **not** been scored
with `make eval` (Track 2 verification step 3 is open).

**E2E parity** (`runs/e2e/latest.json`, 2026-09-15, gitignored): 5 scenarios against
a live API — grounding (VHR PNG), bi-temporal change (S2 pair), cross-modal (S2 + S1),
negative no-overlap pair (rejected `INSUFFICIENT_OVERLAP`), negative disabled tool
(degrades, no error). Sync vs jobs+SSE traces identical except one run where the
jobs path did not emit `LATENCY_BUDGET_EXCEEDED` (40.7 s DAG vs 30 s budget).
The cross-modal run answered from the template (`vlm_vqa did not run`). Re-run
`make e2e` before the final.

**Change detector** (`data/checkpoints/cd/levircd_resnet18.ckpt.json`): ResNet-18
Siamese, SSL4EO-S12 init, 60 epochs @ 256 px, bf16-mixed. Test **F1 0.858, IoU 0.751,
P 0.864, R 0.852**, calibrated threshold 0.9. Below the 0.88 gate — say so if asked.
No OSCD (10 m) run.

**Segmenter:** `data/checkpoints/seg/segformer-b5-loveda/` (downloaded HF snapshot,
not trained here).

**Serving:** bf16 base ~16.4 GB weights; ~18.7 GiB peak with six views + adapter;
VRAM guard budget 22 GB. Laptop: Q4_K_M + mmproj ≈ 6.2 GB + KV cache, target
< 7.4 GB with one 3-view request.

## What is on disk after the 2026-09-29 cleanup

Deleted (≈ 595 GiB; full list in the untracked, ignored `cleanup_report_2026.md`):
`data/raw/` (BigEarthNet, VRSBench, RSVQA-HR, DIOR-RSVG, CDVQA, LEVIR-CD, OSCD),
`data/processed/` (rendered views + corpus), every non-final run
(`poc-v1`, `full-epoch-v1`, `throughput-probe*`, `sq-lora-v1-sanity`,
`sq-lora-v2-sanity`), the v2 intermediate checkpoints, and the HF cache for
`Qwen/Qwen3-VL-8B-Instruct`.

Kept: `runs/sq-lora-v2-full/{adapter,train.log,run_manifest.json,mask_audit.json,preflight}`,
`runs/eval/`, `runs/e2e/`, `models/`, `data/checkpoints/`, `data/artifacts/`,
`data/traces.sqlite3`.

Consequences:
- **`make eval` / `eval-baseline` cannot run** — the runner resolves views from
  `data/processed/views/`. Re-fetch + re-render + rebuild first (`06`).
- **Retraining** needs the whole data pipeline re-run (hundreds of GB, hours of CPU).
- **Serving the adapter via `hf`** re-downloads the 17 GB base model on first load.
  Alternative (untested): `SATQUERY_VLM_MODEL_PATH=models/sq-lora-v2-full-merged`
  with `SATQUERY_VLM_ADAPTER_PATH` unset — the merged weights *are* base + adapter,
  but `Answer.generator` will not carry the `+adapter:` suffix.
- The llama.cpp / laptop path is unaffected.

## Current `.env` on the dev box

```
SATQUERY_VLM_BACKEND=hf
SATQUERY_VLM_ADAPTER_PATH=runs/sq-lora-v2-full/adapter
SATQUERY_VLM_DTYPE=bfloat16
SATQUERY_VLM_VRAM_BUDGET_GB=22
SATQUERY_SEG_CHECKPOINT=data/checkpoints/seg/segformer-b5-loveda
SATQUERY_CD_CHECKPOINT=data/checkpoints/cd/levircd_resnet18.ckpt.pt
```

## Known gaps (full detail in `08`)

1. Manual QA sign-off and the Wi-Fi-off laptop rehearsal are not recorded.
2. No benchmark-level baseline column and no Q4_K_M column.
3. SCENE_CLASSIFY (multi-label BigEarthNet) is weak: 5 % exact / 50 % F1.
4. CD below its 0.88 gate; no OSCD run.
5. Jobs are in-memory (lost on restart); SQLite/filesystem stores are single-process.

## What is genuinely strong (do not regress these)

- The evidence chain: FactSheet → key- and unit-aware CitationValidator → named
  confidence caps → trace; 100 % citation precision on the benchmark.
- Contract chain `API_CONTRACT.md` → `openapi.json` → `schema.d.ts`, locked in CI.
- Train/serve parity: one `layout.py`, a mask audit before weights load, and a layout
  fingerprint that refuses a mismatched adapter at load.
- Deterministic planner + async executor with process-wide CPU/GPU semaphores,
  content-hash cache, one-level fallbacks, honest `DEGRADED`.
- bf16 serving with a VRAM guard, NaN-weight scanner and degenerate-output tripwire.
- Frontend: contrast-tested tokens, letterbox-correct bbox overlay, pure SSE reducer
  with reattach, offline MSW mode that covers the whole Maps/STAC flow.
