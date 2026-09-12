# 03 — Architecture

The authoritative long-form version is `repo documentation/ARCHITECTURE.md` (395
lines, written 2026-09-11 against the current code) and the three frozen specs in
`DOCS/` (`API_CONTRACT.md`, `AGENT_POLICY_DAG.md`, `DATA_ADAPTATION_PLAN.md`). This
file is the compressed, code-anchored map. File paths are relative to
`src/satquery/` unless stated.

## 1. Request flow

```
upload(s) ─▶ POST /v1/validate ─▶ InputManifest[] + CompatibilityReport + supported_tasks   (CPU only)
question ─▶ POST /v1/jobs (202 {job_id})  ── or ── POST /v1/analyze (sync)
            GET /v1/jobs/{id}/events  (SSE)
              queued → stage(ingesting, validating, rendering, planning, executing, aggregating, done)
              → plan → step_started / step_completed / artifact (interleaved) → done | error
            AnalyzeResponse { answer{text, citations[], uncited_numeric_spans[]},
                              confidence{overall, method, components, caps_applied},
                              resolved_task, artifacts[], trace: AuditTrace }
GET /v1/traces/{trace_id}      the stored AuditTrace (SQLite)
GET /v1/artifacts/{trace}/{id}.{ext}   raw PNG / GeoTIFF / GeoJSON bytes
GET /v1/registry               live tool catalogue     GET /v1/health   device, VRAM, igpu_masked, tools
```

Stage sequence, module by module:

| Stage | Module(s) | What happens |
|---|---|---|
| Ingest | `ingest/reader.py`, `manifest_builder.py`, `modality.py`, `bands.py` | rasterio open (GeoTIFF/PNG/JPEG); every field that cannot be read is `null` with a reason in `warnings`; modality + sensor fingerprint from band count/dtype/dynamic range; logical band resolution via `configs/band_aliases.yaml` |
| Validate | `ingest/coregistration.py`, `compatibility.py`, `capabilities.py`, `errors.py` | pair-type inference; 10 checks (`crs_match, bounds_overlap_iou, gsd_ratio, coregistration_offset_px, band_sufficiency, modality_distinct, temporal_ordering, georeference_present, size_ratio, nodata_extent`); reproject/resample/crop to a common grid, recorded in `actions_taken`; `supported_tasks` from the registry matcher; failures map onto the frozen error taxonomy (`API_CONTRACT §6`) |
| Classify + parse | `agent/task_classifier.py`, `query_parser.py` | 4-stage cascade: A PairType gate → B ordered regex rules (`rules_v1`) → C embedding kNN over `configs/query_seed_set.jsonl` (only if B < 0.75) → D LLM slot-fill (always, for slots; may *propose* a task only if B+C < 0.55, which applies the `llm_task_proposal` cap). Slots normalised onto `configs/class_vocabulary.yaml`. Final conf < 0.40 → `UNSUPPORTED` + generic DAG (or 422 if `allow_generic_fallback=false`) |
| Plan | `agent/planner.py`, `registry/capability_match.py`, `configs/policy_table.yaml` | policy key `TaskType\|PairType\|ModalityKey`; lookup order exact → `T\|P\|*` → `*\|*\|*` (flagged `GENERIC_PLAN_USED`, capped). Capability matching substitutes the declared one-level `fallback` for unavailable tools, visibly `DEGRADED`. Input tokens: `pre, post, optical, sar, single, all, @N, @N:ARTIFACT_TYPE` |
| Render | `render/renderer.py`, `views.py`, `composites.py`, `indices.py`, `colormaps.py`, `tiling.py`, `view_labels.py`, `artifact_store.py` | the frozen view catalogue (TC, FCIR, SWIR, NDVI, NDWI, NDBI, SARFC, SARDB, PAN, CHANGE) rendered at 448 px, ≤ 6 per request in a fixed slot order; index/SAR views on fixed domains as PNG; each view labelled by `view_labels.label_for_view` — the exact string the model reads; content-addressed blob store under `data/artifacts/` |
| Execute | `agent/executor.py`, `tools/*`, `tools/base.py`, `tools/catalog.py` | async DAG; tools run under `asyncio.to_thread` with `wait_for` timeouts; separate CPU and GPU semaphores (per executor instance — see `08`); content-hash `ExecutionCache` whose key excludes wall-clock; per-step `OK / DEGRADED / FAILED / SKIPPED`; only `spectral_renderer` is `optional: false` (its failure is a 500). Every artifact is emitted as an SSE event as soon as written |
| Aggregate | `agent/aggregator.py`, `evidence/fact_sheet.py`, `citation_validator.py`, `confidence.py` | FactSheet = union of tool scalars keyed `<tool>.<scalar>`; VLM step (if any) writes prose; validator resolves numeric spans; template fallback if no VLM; confidence formula + caps |
| Trace | `trace/builder.py`, `trace/store.py` | `AuditTrace` (schema 1.0) assembled by the executor, persisted to SQLite (`data/traces.sqlite3`) |
| API | `api/app.py`, `api/routers/*`, `api/jobs.py`, `api/uploads.py`, `api/fixtures.py` | FastAPI app factory; in-process `JobStore` (`MAX_JOBS = 64`, oldest-first eviction); ingestion runs *before* the 202 so bad uploads are 4xx not error events; SSE with 15 s heartbeat and buffered replay to late subscribers; Phase 0 fixtures still served for contract tests |

`agent/pipeline.py` is the orchestrator (images + query → AuditTrace) used by both
routers; `agent/events.py` is the optional progress sink the jobs router plugs in.

## 2. The tool registry (15 tools, `configs/registry.yaml`)

| Tool | Category | Device | Needs | Fallback | Produces | Key scalars |
|---|---|---|---|---|---|---|
| `spectral_renderer` | geo | cpu | — | — | RENDERED_VIEW | views_rendered |
| `raster_statistics` | analysis | cpu | — | — | SCALARS | brightness_mean, dynamic_range, valid_pixel_pct |
| `spectral_index_analyzer` | analysis | cpu | red, nir (+green, swir1) | — | SCALARS, HEATMAP | nd{vi,wi,bi}_{mean,std,…}, {vegetation,water,built_up}_fraction_pct |
| `sar_backscatter_analyzer` | analysis | cpu | vv (+vh) | — | SCALARS, HEATMAP | sigma0_{vv,vh}_db_*, vv_vh_ratio_db_*, speckle_cv, low_backscatter_fraction_pct |
| `change_statistics` | analysis | cpu | a CHANGE_MASK | — | SCALARS, GEOJSON | changed_area_{pct,m2,km2}, component_count, changed_<class>_{pct,m2} |
| `object_counter` | analysis | cpu | BBOX_SET or SEGMENTATION | — | SCALARS | count, <class>_count |
| `image_diff_change` | cv | cpu | 2 images | — | CHANGE_MASK, OVERLAY_PNG, SCALARS | CVA (optical) / log-ratio (SAR), Otsu threshold |
| `siamese_change_detector` | cv | rocm:0 | `SATQUERY_CD_CHECKPOINT` | image_diff_change | CHANGE_MASK, OVERLAY_PNG, SCALARS | tiled inference (`tools/tiled_inference.py`), calibrated threshold |
| `semantic_segmenter` | cv | rocm:0 | `SATQUERY_SEG_CHECKPOINT` (SegFormer-B5 LoveDA) | spectral_index_analyzer | SEGMENTATION, BBOX_SET, SCALARS | <class>_fraction_pct, <class>_area_m2 |
| `text_grounding` | cv | rocm:0 | VLM | semantic_segmenter | BBOX_SET, GEOJSON, SCALARS | n_boxes, mean_box_score; pixel → WGS84 via the padded-canvas `ViewGeometry` affine |
| `physics_agreement` | fusion | cpu | optical + SAR | — | SCALARS, HEATMAP, TEXT | agreement_pct, ndbi_ambiguous_pct, built_up_confirmed_pct, bare_soil_reassigned_pct (the cross-modal demo moment) |
| `crossmodal_consistency` | fusion | rocm:0 | torchgeo DOFA (`dofa_base_patch16_224`) | physics_agreement | SCALARS, HEATMAP | agreement_mean, divergent_area_pct |
| `vlm_vqa` / `vlm_caption` / `vlm_change_vqa` | vlm | rocm:0 | a servable backend | — | TEXT | none (VLM contributes no scalars by design) |

`available: true` in YAML is intent; `registry.default_registry` intersects it with
`tools.catalog.runnable_tools()` (checkpoint present? backend servable?) so the live
`/v1/registry` is honest per machine. Every tool module imports torch lazily so the
registry builds on a CPU-only box.

## 3. The evidence mechanism (the part everything else exists for)

**FactSheet** (`evidence/fact_sheet.py`): every scalar with unit and producing step.
The VLM is shown it and cannot add to it.

**Prompt** (`models/prompts/templates.py`, version `grounded_v1`, selected by
`SATQUERY_VLM_PROMPT_VERSION`; `builder.py` assembles views + sheet + question;
`box_format.py` is the one serialiser/parser for Qwen's
`<|object_ref_start|>label<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>`
form on a 0–1000 frame, with fallbacks for JSON `bbox_2d` and bare
`label(x1,y1),(x2,y2)`). Citation markers in trained answers are `[key]`.

**CitationValidator** (`evidence/citation_validator.py`):
- regex finds numeric spans, understands thousands separators/decimals and the units
  `% km² m² dB px m`; excludes years, image indices, list markers.
- **units must match, not just magnitude**, matched on `_`-delimited segments of the
  scalar name (`7.4 %` may only cite a `*_pct`; `sigma0_vv_db_mean` is a dB mean).
- tolerance: relative 1 % above 10, absolute 0.05 below.
- resolved → `Citation{claim, source: "step:N/scalars.<key>", value}`; unresolved →
  `uncited_numeric_spans`. Policy `flag` (default) keeps the text; `strip` is for
  benchmark scoring only (`SATQUERY_CITATION_POLICY`).
- grounding answers: `box_format.strip_boxes` first — coordinates are positions,
  not claims.
- **Known weakness:** resolution is by *value*, not by the `[key]` the model emitted
  (`08` #5).

**Confidence** (`evidence/confidence.py`, `weighted_tool_agreement_v1`):
`overall = 0.20·task_classification + 0.20·input_quality + 0.35·tool_mean + 0.25·cross_tool_agreement`,
VLM steps excluded from `tool_mean`, a single uncorroborated tool scores 0.75, agreement
pairs from `configs/agreement_pairs.yaml`. Then caps, frozen order, each recorded by name:
`compatibility_warnings 0.80 · llm_task_proposal 0.75 · degraded_execution 0.70 ·
template_fallback 0.65 · uncited_claims 0.60 · failed_execution 0.50 · generic_plan 0.45`.
Caps clamp (worst wins). A policy entry may waive a cap (`CHANGE_MAP` waives
`template_fallback` because it is VLM-free by design).

**Template fallback** (`agent/aggregator.compose`): deterministic per-task prose from
the sheet when the VLM is disabled/unavailable/timed out; `template_fallback: true`.

## 4. VLM backends (`models/`)

`loader.py`: `get_backend()` loads on first call; `current_backend()` reports
without loading (so `/v1/health` never triggers a 16 GB load); `available_backend()`
probes the machine offline; `guard_vram()` estimates before load and **re-checks
after**; idle unload after `SATQUERY_VLM_IDLE_UNLOAD_S` (default 900 s);
`LazyBackend` serialises generation behind an `RLock`; greedy decoding seeds torch
from `request.seed`; stop sequences are applied *after* generation.

- **`hf`** (`hf_backend.py`): transformers in-process, **bf16 only**, LoRA applied at
  load from `SATQUERY_VLM_ADAPTER_PATH` (hard-fails if `adapter_config.json` missing
  rather than silently serving stock Qwen). `_strip_chat_tokens` removes scaffolding by
  name instead of `skip_special_tokens=True`, which would delete the box sentinels.
  Labels are inserted as **text parts immediately before each image**.
- **`llamacpp`** (`llamacpp_client.py`): HTTP client to `llama-server`
  (`scripts/serve_vlm.sh`) holding a Q4_K_M GGUF + mmproj. ~6–9 GB, seconds to start,
  no adapter. Do not run both on one card — the VRAM guard refuses the bf16 load.

## 5. Frontend (`frontend/`, React 19 + Vite 6 + TS 5.9 + Tailwind v4 + Zustand 5)

- No router. Three-column shell (`components/shell/AppShell.tsx`): nav rail ·
  stage (scene header, A/B swipe viewer with zoom/pan and SVG bbox layer, evidence
  strip, KPI cards) · thread panel (chat, results, citations, history). Two-dimensional
  breakpoints `wide` (≥768 px **and** ≥500 px tall) and `desk` (≥1280 px and ≥500 px).
- Stores: `state/job.ts` (pure reducer over the SSE union, tested against a recorded
  run in `mocks/captured/events.bitemporal.json`), `state/ui.ts` (nav, uploads,
  pre-flight, session history), `state/focus.ts` (cross-column focus: view, swipe,
  lit KPI, armed pipeline step, composer draft).
- `api/client.ts` typed against generated `api/schema.d.ts` (`npm run gen:api` from
  `../openapi.json`); `api/sse.ts` parser with last-event-id; `thread/useRun.ts`
  owns submit → stream → result/cancel/retry.
- `thread/annotate.ts` maps `citations[].claim` substrings to spans (longest first,
  digit-boundary rule); `thread/bbox.ts` is a TS port of `box_format.py`'s parser;
  `components/stage/BboxOverlay.tsx` draws in `viewBox="0 0 1000 1000"` sized to the
  raster's `object-fit: contain` rectangle (natural size + ResizeObserver), inside the
  zoom transform, `vector-effect: non-scaling-stroke`.
- `styles/theme.css` is a spec: every hue has fill / text / on-fill tokens with
  computed WCAG ratios; `styles/__tests__/contrast.test.ts` parses the CSS and fails
  the suite if a pair breaks. `--color-warn` is the only hue allowed on an uncited span.
- `?mock=1` starts MSW (`mocks/`) with captured fixtures — the whole UI runs with
  the backend down. Fonts self-hosted in `public/fonts/` with metric-matched fallbacks.
- Lazy-loaded DAG dialog (`components/pipeline/DagCanvas.tsx`, @xyflow + dagre).

## 6. Training subsystems (detail in `06`)

- `training/corpus_builder.py` (2,164 lines): unified instruction sample schema,
  per-source builders, streaming reservoir sampling, `Deduplicator` (SHA-256 exact →
  DCT pHash Hamming ≤ 5 with 8-band bucketing → quarantine `LeakageError`), build-time
  CitationValidator audit (`CitationError`). Imports the *same* label builder, box
  serialiser and number formatter the server uses.
- `training/local_sources.py`: correct readers for VRSBench (zips), RSVQA-HR
  (`active` flag), CDVQA (unpacked WebDataset).
- `training/vlm/qlora.py` (1,278 lines): pydantic profile, NF4 + LoRA over LLM
  projections and last-N ViT blocks resolved by name, VRAM estimator + `guard_budget`,
  trl `SFTTrainer`, resumable, SIGINT-safe, writes `run_manifest.json`.
- `training/cd/*`: Lightning Siamese CD (shared ResNet encoder from SSL4EO-S12,
  per-scale differencing, FPN decoder), torchgeo datamodules for LEVIR-CD/OSCD,
  threshold calibration, checkpoint bundle with provenance + self-test.
- `training/data/builders/evidence_qa.py` (top-level `training/`, not `src/`): the
  synthetic citation-behaviour source.
