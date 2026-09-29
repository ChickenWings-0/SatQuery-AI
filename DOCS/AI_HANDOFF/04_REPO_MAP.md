# 04 — Repository Map

Root: `/home/chickenwings/SatQuery-AI`. Python package root is `src/` (`pyproject`
sets `pythonpath = ["src"]`). Snapshot 2026-09-29.

## Top level

| Path | Tracked? | Role |
|---|---|---|
| `README.md` | yes | public README: current model, quick start, gates, layout, contributing |
| `PRODUCT.md` | yes | product brief for the impeccable design plugin (must stay at root) |
| `pyproject.toml` / `uv.lock` | yes | metadata, base deps, extras `vlm`, `vlm-train`, `cd`, dev group; ruff/mypy/pytest config. **The lock pins CUDA torch — the dev box does not use it (`05`)** |
| `.python-version` | yes | `3.11` |
| `openapi.json` | yes | frozen machine contract — `make contract` |
| `Makefile` | yes | `make help`; `ci`, `test`, `contract`, `demo`, `demo-cpu`, `eval`, `eval-baseline`, `eval-self-check`, `e2e`, `e2e-cpu`, `frontend-e2e`, `docker` |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` | yes | backend image (ROCm runtime is large; CI builds the `deps` target) + frontend nginx image |
| `.github/workflows/ci.yml` | yes | backend · frontend · e2e (Playwright) · docker jobs |
| `.pre-commit-config.yaml` | yes | ruff, mypy, the main-only branch hook |
| `.env.example` | yes | documented `SATQUERY_*` settings; `.env` is local and ignored |
| `run_overnight.sh` | yes | v1-era overnight playbook (datasets → render → GPU probe); superseded for corpus builds by `scripts/rebuild_corpus_v2.sh` |
| `DOCS/` | yes | all documentation (`09`) |
| `.claude/` | **no** | per-user Claude Code settings (ignored since 2026-09-29) |
| `cleanup_report_2026.md` | no | record of the 2026-09-29 disk cleanup (ignored) |
| `full_training_run.log`, `training_run.log`, `train_levircd.log`, `.levircd.pid`, `logs/` | no | v1-era logs; safe to delete |
| `.venv/` | no | Python 3.11 with the **hand-installed ROCm torch stack** |

## `src/satquery/`

```
core/        config.py (Settings, load_env_file)  logging.py (structlog)
schemas/     FROZEN contract: enums, manifest, compatibility, tool, trace, api, version (SCHEMA_VERSION="1.0")
ingest/      reader, manifest_builder, modality, bands, coregistration, compatibility, capabilities, errors,
             pipeline, stac_fetch (Planetary Computer COG windows)
render/      views, view_labels (THE label builder), composites, indices, colormaps, tiling (ViewGeometry),
             overlays, renderer, artifact_store
registry/    registry (YAML → ToolSpec ∩ runnable tools), capability_match
tools/       base, catalog, one module per tool, tiled_inference, change_common, vlm_runtime
agent/       task_classifier, query_parser, planner, executor, concurrency (process-wide DeviceGates),
             aggregator, pipeline, events
evidence/    fact_sheet, citation_validator (key- and unit-aware), confidence
models/      loader, hf_backend, llamacpp_client, prompts/{templates, layout, builder, box_format}
trace/       builder, store (SQLite)
api/         app, dependencies, uploads, jobs (JobStore), fixtures (Phase 0 mocks),
             routers/{health, registry, validate, analyze, jobs, traces, artifacts, imagery}
training/    corpus_builder, local_sources, vlm/qlora, cd/{data, encoder, model, module, checkpoint}
eval/        sampler, runner, scorers, report
```

`training/data/builders/evidence_qa.py` (top level, not under `src/`) is the synthetic
evidence-QA builder.

## `scripts/`

| Script | Purpose |
|---|---|
| `rocm_env.sh` | `source` in every GPU shell (`HIP_VISIBLE_DEVICES=0`, allocator config) |
| `serve_vlm.sh` / `serve_vlm.ps1` | `llama-server` with GGUF + mmproj (Linux / Windows; KV-cache and flash-attn flags via env) |
| `demo_laptop/` | `.env.laptop` (copied to `.env` on the laptop) and `serve_frontend.py` (serves `frontend/dist`, proxies `/v1`) |
| `export_openapi.py` | regenerate `openapi.json` (argparse) |
| `make_synthetic_fixtures.py` | synthetic GeoTIFFs with known CRS/GSD/shift for tests and e2e |
| `fetch_sources.py` | fetch RSVQA-HR, unpack CDVQA WebDataset |
| `render_views.py` / `render_vhr_views.py` | pre-render views for BigEarthNet / VHR sources into `data/processed/views/` |
| `build_corpus.py` | six sources → deduplicated corpus; `--composition SOURCE=N`; hard-fails on leakage |
| `rebuild_corpus_v2.sh` | end-to-end v2 corpus rebuild (`PRESET=sprint\|full`) with composition floors |
| `patch_dummy_images.py` | black placeholders for `--sanity-check` only — never train on them |
| `preflight_train_serve_parity.py` | proves the serving prompt is byte-identical to training before an adapter is trusted |
| `train_vlm.py` | QLoRA driver (`--sanity-check`, then the run) |
| `merge_export.py` | bf16 CPU merge → safetensors → GGUF (`--gguf --llama-cpp <dir> --quant Q4_K_M`), writes `SHA256SUMS` + `merge_manifest.json` |
| `test_inference.py` | grounding / cross-modal probe of an adapter on one held-out patch (needs `data/processed`) |
| `eval_benchmark.py` | Track 1 benchmark CLI (`make eval`, `--baseline`, `--self-check`, `--resume`) |
| `eval_vrsbench_zeroshot.py` | early zero-shot VRSBench baseline (superseded by `eval_benchmark.py`) |
| `e2e_parity.py` | live-API parity: 3 scenarios + 2 negatives, sync vs jobs/SSE (`make e2e`) |
| `train_cd.py` | Siamese CD on LEVIR-CD / OSCD |
| `git/main_only.sh` | pre-commit / pre-push hook refusing non-`main` branches (`SATQUERY_ALLOW_BRANCH=1` bypass) |

## `configs/`

| File | Content |
|---|---|
| `registry.yaml` | 15 `ToolSpec`s |
| `policy_table.yaml` | 29 entries (`policy_table_v1`) incl. the generic `*\|*\|*` |
| `band_aliases.yaml` | sensor → logical bands: S2 BEN, S2 L2A, S1 GRD, Cartosat-2S MX/PAN, RISAT-1, Generic RGB |
| `class_vocabulary.yaml` | canonical classes + synonyms |
| `agreement_pairs.yaml` | cross-tool agreement pairs |
| `query_seed_set.jsonl` | labelled queries for the kNN classifier (tracked despite `*.jsonl` ignore) |
| `train/qlora_qwen3vl8b_rocm24g.yaml` | **Profile A** — the v2 production profile |
| `train/lora_qwen3vl4b_bf16.yaml` | Profile B — 4B bf16 OOM fallback (never needed) |

## `data/` (gitignored except `.keep`)

```
data/
  checkpoints/  cd/levircd_resnet18.ckpt.{pt,json} (+ .bak-prenorm / .bak-preselftest, runs/ Lightning dir)
                seg/segformer-b5-loveda/  (HF snapshot)
  artifacts/    blobs/<sha>/…  imagery/ (STAC fetches)  traces/<trace_id>/  (~150 traces)
  traces.sqlite3
  raw/, processed/   DELETED 2026-09-29 — re-created by the fetch/render/build scripts
```

## `runs/` (gitignored except `runs/eval/*/results.*` and `sample_ids.json`)

```
runs/
  sq-lora-v2-full/   adapter/ (LoRA + processor + tokenizer + layout.json), train.log, run_manifest.json,
                     mask_audit.json, preflight/, README.md (trl model card)
  eval/sq-lora-v2-full/   results.{md,json,csv}, sample_ids.json (tracked); predictions.jsonl, confusion/ (ignored)
  e2e/latest.json         last parity run
  sq-lora-v2/preflight/, Sep13_*_fedora/   tiny leftovers
```

## `models/` (gitignored)

`sq-lora-v2-full-merged/` (bf16 safetensors + `merge_manifest.json`),
`sq-lora-v2-full-merged-Q4_K_M.gguf`, `sq-lora-v2-full-merged-mmproj-f16.gguf`,
`SHA256SUMS`. This is also the owner's backup of the exported weights — do not delete.

## `frontend/`

```
package.json    dev, build (prebuild rewrites public/sitemap.xml), typecheck, test, test:e2e, lint (oxlint),
                check:bundle, gen:api, fonts:sync, preview
.nvmrc / engines   Node ≥ 22.12      .npmrc   legacy-peer-deps=true
vite.config.ts  proxy /v1 → http://127.0.0.1:8000 (IPv4 on purpose)
playwright.config.ts   vite preview + ?mock=1, browser offline
e2e/            overlay.spec.ts, track4.spec.ts
public/         fonts/ (woff2 + SITREP ttf subsets), mock-artifacts/, samples/{globe,stac}/, og/, sitemap, llms.txt
scripts/        build-land-mask.py, check-bundle.mjs, gen-public-meta.mjs, gen-sitrep-fonts.py
src/
  App.tsx, main.tsx, format.ts
  api/          client, types, events, sse, schema.d.ts (generated)
  shell/        router, shortcuts, storage, meta, quotes, useHotkeys, useReducedMotion
  state/        job, ui, focus, map, stac, library, exports, settings, theme, notifications, account, shortcuts, landing
  thread/       useRun, resume, newQuery, annotate, bbox, boxes
  evidence/     scene, views, georef
  export/       sitrep/, geojson/, download.ts
  geo/          nominatim, stac, planetary, collections, deadline
  kpi/          registry
  pages/        Landing (+landing/globe), Maps (+maps/hud), Projects, Saved (+saved/), UseCases, Report, NotFound
  components/   shell, stage, thread, pipeline, panels, export, ui
  mocks/        MSW browser, handlers, fixtures, scenarios, captured/*.json
  styles/       theme.css (token spec)
  __tests__ alongside each area (37 files, 356 tests)
```

## `tests/`

`conftest.py` (synthetic fixtures), `contract/` (`test_api.py`, `test_jobs_sse.py`),
`integration/test_train_serve_parity.py`, `unit/` (agent, change detection, citation
keys, concurrency, corpus script, crossmodal/grounding, eval scorers, evidence_qa,
ingest, models, prompt layout, render, STAC fetch, VLM training). 623 cases, hermetic,
no network, torch only where under test. `tests/fixtures/` is an empty placeholder
(e2e scenes are generated at run time into a temp dir).
