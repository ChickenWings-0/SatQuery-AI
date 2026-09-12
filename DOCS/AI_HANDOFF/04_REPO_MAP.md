# 04 — Repository Map

Root: `/home/chickenwings/SatQuery-AI`. Python package root is `src/` (`pyproject`
sets `pythonpath = ["src"]`). Line counts are from the 2026-09-11 snapshot.

## Top level

| Path | Tracked? | Role |
|---|---|---|
| `pyproject.toml` | yes | project metadata; base deps (FastAPI, rasterio, geopandas, shapely, scikit-image, OpenCV headless, pydantic, structlog, sse-starlette); extras `vlm`, `vlm-train`, `cd`; dev group (pytest, pytest-asyncio, ruff, mypy); ruff/mypy/pytest config |
| `uv.lock` | yes | authoritative lock (uv 0.12.10) |
| `.python-version` | yes | `3.11` |
| `openapi.json` | yes | **frozen contract export** — regenerate with `scripts/export_openapi.py`; must diff clean |
| `README.md` | yes | **0 bytes** (defect; see `08`) — the real README is `repo documentation/README.md` |
| `.env` | **no** | local settings, `SATQUERY_*` prefix (see `05`) |
| `.gitignore` | yes | ignores `data/*`, `/runs/`, `*.log`, `.env*`, `frontend/node_modules`, `frontend/dist`, caches |
| `.gitattributes` | yes | `* text=auto` |
| `.claude/settings.json` | yes | enables the `impeccable` plugin (frontend design skill) |
| `.vscode/settings.json` | yes | trivial |
| `run_overnight.sh` | yes | 3-stage overnight playbook: datasets → render pass → GPU probe; logs to `logs/overnight-*.log` |
| `full_training_run.log`, `training_run.log`, `train_levircd.log` | no | raw logs of the full-epoch VLM run, an earlier run, and the LEVIR-CD run |
| `.levircd.pid` | no | leftover pid file from the CD training |
| `.venv/` | no | uv-managed venv, Python 3.11, torch 2.9.1+rocm6.4, transformers 5.16.1, trl 1.12.0 |

## `src/satquery/` — the package (see `03` for the per-module table)

```
satquery/
  __init__.py
  core/        config.py (Settings, load_env_file)  logging.py (structlog)
  schemas/     FROZEN contract: enums, manifest, compatibility, tool, trace, api, version (SCHEMA_VERSION="1.0")
  ingest/      reader, manifest_builder, modality, bands, coregistration, compatibility, capabilities, errors, pipeline
  render/      views (catalogue+selection), view_labels (THE label builder), composites, indices, colormaps,
               tiling (ViewGeometry), overlays, renderer, artifact_store
  registry/    registry (YAML → ToolSpec, intersected with runnable tools), capability_match
  tools/       base (protocol), catalog (name→impl, runnable_tools), one module per tool, tiled_inference,
               change_common, vlm_runtime (shared VLM tool machinery)
  agent/       task_classifier, query_parser, planner, executor (DagExecutor, ExecutionCache), aggregator,
               pipeline (orchestrator), events (progress sink)
  evidence/    fact_sheet, citation_validator, confidence
  models/      loader (device, VRAM guard, LazyBackend, GenerationRequest/Result), hf_backend, llamacpp_client,
               prompts/{templates (grounded_v1), builder, box_format}
  trace/       builder, store (SQLite)
  api/         app (create_app), dependencies, uploads, jobs (JobStore), fixtures (Phase 0 mocks),
               routers/{health, registry, validate, analyze, jobs, traces, artifacts}
  training/    corpus_builder (2164 lines), local_sources, vlm/qlora (1278), cd/{data, encoder, model, module, checkpoint}
src/satquery_ai/__init__.py     leftover "Hello from satquery-ai!" scaffold — delete (pyproject still points a script at it)
```

## `training/` (top level, not under `src/`)

`training/data/builders/evidence_qa.py` — the synthetic evidence-QA source builder.
(Master.md planned all builders here; in practice they live in
`src/satquery/training/corpus_builder.py` and this is the one that stayed.)

## `scripts/`

| Script | Purpose |
|---|---|
| `rocm_env.sh` | `source` it in every GPU shell: `HIP_VISIBLE_DEVICES=0`, `ROCR_VISIBLE_DEVICES=0`, `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`, `HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| `serve_vlm.sh` | start `llama-server` with a GGUF + mmproj (the offline path) |
| `export_openapi.py` | regenerate `openapi.json` from `create_app()` (note: treats *any* argv as output path — `--help` writes a file named `--help`) |
| `make_synthetic_fixtures.py` | synthetic GeoTIFFs with known CRS/GSD/shift/overlap for the ingest tests |
| `fetch_sources.py` | fetch RSVQA-HR (off-Hub JSON) and CDVQA (unpack 1,533 WebDataset shards into pre/post PNG pairs) |
| `render_views.py` | Phase 2 pre-render pass for BigEarthNet patches → `data/processed/views/bigearthnet_v2/`, plus measured scalars for `evidence_qa` |
| `render_vhr_views.py` | pre-render pass for the 3-channel VHR sources (VRSBench, RSVQA-HR, CDVQA) |
| `build_corpus.py` (797 lines) | wires the six sources through `corpus_builder`; `--limit`, `--require-views`; hard-fails on test-split leakage. **Contains the `hashes_for` / `bind_view_paths` seam that left BEN rows unhashed** |
| `patch_dummy_images.py` | black 448² placeholders so `--sanity-check` can run before the render pass exists (never train on these) |
| `train_vlm.py` | QLoRA driver: `--sanity-check` first, then the run; `--config configs/train/*.yaml --output-dir runs/<name>` |
| `merge_export.py` | merge LoRA into base, export safetensors and GGUF via llama.cpp's converter (`--dry-run` prints commands) |
| `test_inference.py` | serve the adapter on one held-out BEN patch, four probes, audit citations → `runs/<run>/inference_probe*.json` |
| `eval_vrsbench_zeroshot.py` | zero-shot VRSBench VQA baseline; defaults to a 5-item synthetic mock unless `--data-dir` |
| `train_cd.py` | Lightning training of the Siamese CD on LEVIR-CD or OSCD (`--download` required to fetch) |

## `configs/`

| File | Content |
|---|---|
| `registry.yaml` | the 15 `ToolSpec`s (accepts/produces/scalars_schema/device/est_ms/fallback/available) |
| `policy_table.yaml` | 22 policy entries (`policy_table_v1`), keyed `Task\|Pair\|Modality`, with the generic `*\|*\|*` fallback |
| `band_aliases.yaml` | sensor → logical band numbers (1-indexed): S2 BEN 12-band, S2 L2A 13-band, S1 GRD, Cartosat-2S MX/PAN, RISAT-1, Generic RGB; `fallback_by_band_count` |
| `class_vocabulary.yaml` | canonical classes (built_up, water, vegetation, bare_soil, road, aircraft, ship, vehicle, airport, cloud) + synonyms |
| `agreement_pairs.yaml` | frozen cross-tool agreement pairs with disagreement scales |
| `query_seed_set.jsonl` | ~130 labelled queries for the Stage C kNN classifier |
| `train/qlora_qwen3vl8b_rocm24g.yaml` | **Profile A**: 8B, NF4 double-quant, r=16 α=32, `max_pixels` 147456 (384²), seq 4096, batch 1×16, 1 epoch, lr 1e-4, `paged_adamw_8bit`, `save_steps` 50 |
| `train/lora_qwen3vl4b_bf16.yaml` | **Profile B**: 4B bf16 LoRA, seq 3072, `adamw_torch_fused` |

## `configs`-adjacent data files (gitignored, under `data/`)

```
data/
  raw/            ben/V2/*.tar.gz* (BigEarthNet-v2 S1/S2 + reference maps, partial downloads present)
                  vrsbench/ (Annotations_train/*.json …)   rsvqa_hr/ (USGS_split_* JSON + Images.tar)
                  cdvqa/ (train/val.jsonl after unpack)     dior_rsvg/ (HF cache; gated — unresolved)
                  Onera Satellite Change Detection dataset - {Images,Train Labels,Test Labels}(.zip)  (OSCD)
                  train.zip / val.zip / test.zip  (LEVIR-CD)
  processed/      views/{bigearthnet_v2,vrsbench,cdvqa}/   corpus/*.jsonl (see 02)
  checkpoints/    cd/levircd_resnet18.ckpt.{pt,json} (+ .bak-prenorm, .bak-preselftest; runs/…/version_0 Lightning ckpt)
                  seg/segformer-b5-loveda/ (HF snapshot: config.json, model.safetensors, preprocessor_config.json)
  artifacts/      blobs/<sha>/…  traces/<trace_id>/index.json   (content-addressed store; ~115 traces on disk)
  traces.sqlite3  (created on first run)
```

## `runs/` (gitignored) — VLM adapters

Each run dir: `adapter/` (LoRA safetensors + tokenizer/processor config),
`checkpoint-N/` (resumable, with optimiser state), `run_manifest.json` (profile,
plan, VRAM estimate, peak VRAM, resolved target modules), `runs/<date>/events.out.tfevents*`
(TensorBoard), `README.md` (trl auto model card).

- `sq-lora-v1-sanity/`, `throughput-probe/`, `throughput-probe-20260906-221333/`
- `poc-v1/` — plus `export/` (merged safetensors, 5 shards), `export-f16.gguf`,
  `export-Q4_K_M.gguf`, `mmproj-export-f16.gguf`
- `full-epoch-v1/` — the production adapter (`checkpoint-1100/1150/1159`, `adapter/`,
  `inference_probe.json`, `inference_probe_discrete.json`)

## `frontend/`

```
package.json         scripts: dev, build (tsc -b && vite build), typecheck, test (vitest run), lint (oxlint — broken binding),
                     preview, gen:api (openapi-typescript ../openapi.json → src/api/schema.d.ts), fonts:sync
vite.config.ts       @tailwindcss/vite + react; alias @ → src; proxy /v1 → http://127.0.0.1:8000 (IPv4 on purpose)
vitest.config.ts     happy-dom
index.html           font preloads, root
public/              favicon.svg, fonts/*.woff2 (+ licences), mock-artifacts/ (art_0..13), mockServiceWorker.js
.vercel/             a Vercel project link exists (project.json) — deployment of the static build was at least set up
dist/                built bundle (gitignored; present on disk)
src/
  main.tsx, App.tsx, format.ts
  api/         client.ts, types.ts, events.ts, sse.ts, schema.d.ts (generated, 1,475 lines)
  state/       job.ts (SSE reducer), ui.ts, focus.ts
  thread/      useRun.ts, annotate.ts, bbox.ts
  evidence/    scene.ts, views.ts (groupForScalar etc.)
  kpi/         registry.ts (fact-sheet key → KPI card)
  shell/       useHotkeys.ts, useReducedMotion.ts
  styles/      theme.css (730 lines, the token spec)
  mocks/       browser.ts, handlers.ts, fixtures.ts, scenarios.ts, captured/*.json (recorded real responses)
  components/
    shell/     AppShell, Sidebar, HealthStrip, ModeSelector
    stage/     DataStage, Dropzone, PreflightPanel, ManifestCard, CompatibilityDetails, SceneHeader,
               ImageViewer (swipe/zoom, 397 lines), BboxOverlay, EvidenceTray, KpiCards
    thread/    ThreadPanel, QueryComposer, GroundedAnswer, PipelinePulse, PreviousQueries, SidebarTabs, SuggestionChips
    pipeline/  PipelineDialog, DagCanvas (lazy)
    panels/    DatasetsPanel, HistoryPanel, ToolsPanel
    ui/        dialog, Disclosure, icons, StatusDot
  __tests__ dirs alongside each area (15 files, 194 tests)
```

## `tests/`

`conftest.py` (synthetic fixtures via `make_synthetic_fixtures`), `contract/`
(`test_api.py` 26, `test_jobs_sse.py` 13), `unit/` (`test_agent` 30,
`test_change_detection` 44, `test_crossmodal_grounding` 62, `test_ingest` 56,
`test_models` 41, `test_render` 77, `test_vlm_training` 103). Hermetic; no network;
torch only where under test. `asyncio_mode = auto`.

## `DOCS/` and `repo documentation/` — see `09_DOC_INDEX.md`.

## `logs/`

`overnight-2026090{6,7,9}-*.log` — six overnight playbook logs. The last
(`overnight-20260909-181509.log`) ends with the throughput number (0.265 samples/s)
and the three open items that led to the BEN-only corpus decision.
