# 03 — Architecture

The long-form version is `DOCS/ARCHITECTURE.md`; the frozen specs are
`DOCS/API_CONTRACT.md`, `DOCS/AGENT_POLICY_DAG.md` and `DOCS/DATA_ADAPTATION_PLAN.md`.
This file is the compressed, code-anchored map. Paths are relative to
`src/satquery/` unless stated.

## 1. Request flow

```
upload(s) ─▶ POST /v1/validate ─▶ InputManifest[] + CompatibilityReport + supported_tasks   (CPU only)
   or  Maps page ─▶ POST /v1/imagery/fetch (STAC COG window → GeoTIFFs) ─▶ GET /v1/imagery/{fetch_id}/{name}
question ─▶ POST /v1/jobs (202 {job_id})  ── or ── POST /v1/analyze (sync)
            GET /v1/jobs/{id}/events  (SSE, id: + Last-Event-ID replay)    DELETE /v1/jobs/{id}  (real cancel)
              queued → stage(ingesting, validating, rendering, planning, executing, aggregating, done)
              → plan → step_started / step_completed / artifact (interleaved) → done | error
            AnalyzeResponse { answer{text, citations[], uncited_numeric_spans[]},
                              confidence{overall, method, components, caps_applied},
                              resolved_task, artifacts[], trace: AuditTrace }
GET /v1/jobs/{id}                 job status (used for reattach)
GET /v1/traces/{trace_id}         the stored AuditTrace (SQLite)
GET /v1/artifacts/{trace}/{id}.{ext}   raw PNG / GeoTIFF / GeoJSON bytes
GET /v1/registry                  live tool catalogue     GET /v1/health   device, VRAM, igpu_masked, tools, jobs, leaked permits
```

| Stage | Module(s) | What happens |
|---|---|---|
| Ingest | `ingest/reader.py`, `manifest_builder.py`, `modality.py`, `bands.py` | rasterio open (GeoTIFF/PNG/JPEG) under `asyncio.to_thread`; unreadable fields are `null` with a reason in `warnings`; modality + sensor fingerprint; logical bands via `configs/band_aliases.yaml` |
| Fetch (optional) | `ingest/stac_fetch.py`, `api/routers/imagery.py` | server-side COG window read from Planetary Computer (signed URLs), clipped to bbox, written as GeoTIFFs the upload path accepts |
| Validate | `ingest/coregistration.py`, `compatibility.py`, `capabilities.py`, `errors.py` | pair-type inference; 10 checks (`crs_match, bounds_overlap_iou, gsd_ratio, coregistration_offset_px, band_sufficiency, modality_distinct, temporal_ordering, georeference_present, size_ratio, nodata_extent`); reproject/resample/crop to a common grid, recorded in `actions_taken`; failures map onto the frozen error taxonomy (`API_CONTRACT §6`) |
| Classify + parse | `agent/task_classifier.py`, `query_parser.py` | cascade: A PairType gate → B regex rules → C embedding kNN over `configs/query_seed_set.jsonl` (if B < 0.75) → D LLM slot-fill (may *propose* a task only if B+C < 0.55, applying the `llm_task_proposal` cap). Final conf < 0.40 → `UNSUPPORTED` + generic DAG (or 422) |
| Plan | `agent/planner.py`, `registry/capability_match.py`, `configs/policy_table.yaml` | 29 entries keyed `Task\|Pair\|Modality`; lookup exact → `T\|P\|*` → `*\|*\|*` (flagged, capped). Unavailable tools replaced by their one-level `fallback`, visibly `DEGRADED` |
| Render | `render/*` | frozen view catalogue (TC, FCIR, SWIR, NDVI, NDWI, NDBI, SARFC, SARDB, PAN, CHANGE) at 448 px, ≤ `SATQUERY_VLM_MAX_VIEWS` (default 6) in a fixed slot order; index/SAR views on fixed domains as PNG; labels from `render/view_labels.label_for_view`; content-addressed blobs under `data/artifacts/` |
| Execute | `agent/executor.py`, `agent/concurrency.py`, `tools/*` | async DAG; tools under `asyncio.to_thread` with `wait_for` timeouts; **process-wide** `DeviceGates` (CPU/GPU semaphores created in the app lifespan, leaked-permit accounting when a timed-out thread keeps running); content-hash `ExecutionCache`; per-step `OK / DEGRADED / FAILED / SKIPPED`; artifacts streamed as SSE events as soon as written |
| Aggregate | `agent/aggregator.py`, `evidence/*` | FactSheet; VLM prose; CitationValidator; template fallback; confidence + caps |
| Trace | `trace/builder.py`, `trace/store.py` | `AuditTrace` (schema 1.0) persisted to SQLite (`data/traces.sqlite3`), write off the loop |
| API | `api/app.py`, `api/routers/*`, `api/jobs.py`, `api/uploads.py` | FastAPI app factory; in-process `JobStore` (finished-only eviction, trace fallback, clean shutdown); ingestion runs *before* the 202 so bad uploads are 4xx; SSE 15 s heartbeat + buffered replay |

`agent/pipeline.py` orchestrates (images + query → AuditTrace) for both routers;
`agent/events.py` is the progress sink the jobs router plugs in.

## 2. The tool registry (15 tools, `configs/registry.yaml`)

| Tool | Device | Needs | Fallback | Key scalars |
|---|---|---|---|---|
| `spectral_renderer` | cpu | — | — (only non-optional tool) | views_rendered |
| `raster_statistics` | cpu | — | — | brightness_mean, dynamic_range, valid_pixel_pct |
| `spectral_index_analyzer` | cpu | red, nir (+green, swir1) | — | nd{vi,wi,bi}_{mean,std,…}, {vegetation,water,built_up}_fraction_pct |
| `sar_backscatter_analyzer` | cpu | vv (+vh) | — | sigma0_{vv,vh}_db_*, vv_vh_ratio_db_*, speckle_cv, low_backscatter_fraction_pct |
| `change_statistics` | cpu | CHANGE_MASK | — | changed_area_{pct,m2,km2}, component_count |
| `object_counter` | cpu | BBOX_SET or SEGMENTATION | — | count, <class>_count |
| `image_diff_change` | cpu | 2 images | — | CVA (optical) / log-ratio (SAR), Otsu |
| `siamese_change_detector` | rocm:0 | `SATQUERY_CD_CHECKPOINT` | image_diff_change | tiled inference, calibrated threshold |
| `semantic_segmenter` | rocm:0 | `SATQUERY_SEG_CHECKPOINT` | spectral_index_analyzer | <class>_fraction_pct, <class>_area_m2 |
| `text_grounding` | rocm:0 | VLM | semantic_segmenter | n_boxes, mean_box_score; pixel → WGS84 via `ViewGeometry` (pixel-only when no CRS) |
| `physics_agreement` | cpu | optical + SAR | — | agreement_pct, built_up_confirmed_pct, … |
| `crossmodal_consistency` | rocm:0 | torchgeo DOFA | physics_agreement | agreement_mean, divergent_area_pct |
| `vlm_vqa` / `vlm_caption` / `vlm_change_vqa` | rocm:0 | servable backend | — | none by design |

`available: true` in YAML is intent; `registry.default_registry` intersects it with
`tools.catalog.runnable_tools()`, so `/v1/registry` is honest per machine. Tool
modules import torch lazily.

## 3. The evidence mechanism

**FactSheet** (`evidence/fact_sheet.py`): every scalar with unit and producing step.

**Prompt** (`models/prompts/`): `templates.py` (`grounded_v1`, selected by
`SATQUERY_VLM_PROMPT_VERSION`), `layout.py` (**the one user-turn layout**,
`label-before-image/v1`, used by training *and* serving, fingerprinted),
`builder.py` (views + sheet + question), `box_format.py` (Qwen
`<|object_ref_start|>label<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>`
on 0–1000, with fallbacks for JSON `bbox_2d` and bare `label(x1,y1),(x2,y2)`).
Citation markers are `[tool.scalar]`.

**CitationValidator** (`evidence/citation_validator.py`):
- finds numeric spans (thousands separators, decimals, units `% km² m² dB px m`);
  excludes years, image indices, list markers.
- **key-aware:** a `[key]` marker adjacent to a span must name a sheet key whose value
  matches (`UNKNOWN_KEY` / `KEY_VALUE_MISMATCH` otherwise); bare numbers fall back to
  value search, ties broken by closeness then key.
- **units must match** on `_`-delimited segments of the scalar name.
- tolerance: relative 1 % above 10, absolute 0.05 below.
- resolved → `Citation{claim, source: "step:N/scalars.<key>", value}`; unresolved →
  `uncited_numeric_spans`. `flag` (default) keeps text; `strip` only for scoring.
- grounding answers: `box_format.strip_boxes` first — coordinates are not claims.

**Confidence** (`evidence/confidence.py`, `weighted_tool_agreement_v1`):
`0.20·task_classification + 0.20·input_quality + 0.35·tool_mean + 0.25·cross_tool_agreement`;
VLM steps excluded from `tool_mean`; single uncorroborated tool = 0.75; pairs from
`configs/agreement_pairs.yaml`. Caps in frozen order, recorded by name:
`compatibility_warnings 0.80 · llm_task_proposal 0.75 · degraded_execution 0.70 ·
template_fallback 0.65 · uncited_claims 0.60 · failed_execution 0.50 · generic_plan 0.45`.
A policy entry may waive a cap (`CHANGE_MAP` waives `template_fallback`).

**Template fallback** (`agent/aggregator.compose`): deterministic prose from the sheet
when the VLM is disabled/unavailable/timed out; `template_fallback: true`.

## 4. VLM backends (`models/`)

`loader.py`: `get_backend()` loads on first call; `current_backend()` reports without
loading (health never triggers a load); `guard_vram()` estimates before load and
re-checks after; idle unload after `SATQUERY_VLM_IDLE_UNLOAD_S`; `LazyBackend`
serialises generation behind an `RLock`; greedy decoding seeded from `request.seed`;
stop sequences applied after generation.

- **`hf`** (`hf_backend.py`): transformers in-process, **bf16 only**. Base from
  `SATQUERY_VLM_MODEL_PATH` (local dir) or `SATQUERY_VLM_MODEL_ID`; LoRA from
  `SATQUERY_VLM_ADAPTER_PATH`. Refuses an adapter without `adapter_config.json` or
  without a matching `layout.json` fingerprint (the v1 adapter would be refused).
  Scans loaded weights for NaN/Inf and trips on degenerate output (runs of one token).
  `_strip_chat_tokens` removes scaffolding by name — never `skip_special_tokens=True`.
- **`llamacpp`** (`llamacpp_client.py`): OpenAI-shaped HTTP client to `llama-server`
  (`scripts/serve_vlm.sh` / `serve_vlm.ps1`) holding the merged Q4_K_M GGUF + f16
  mmproj. Same interface; no torch needed. Don't run both on one card.

## 5. Frontend (`frontend/`, React 19.2 + Vite 6 + TS 5.9 + Tailwind v4 + Zustand 5)

- **Sections, not a router library:** `shell/router.ts` syncs `state/ui.section` with
  the URL; `App.tsx` switches on it. Sections: **Landing** (`/`, three.js globe via
  @react-three/fiber), **Console** (the three-column analysis shell), **Maps**
  (MapLibre stage + STAC discovery HUD), **Projects**, **Saved** (IndexedDB library),
  **Use Cases** gallery, and **Report** (`/report/<trace>`, printable audit record).
  Heavy pages are lazy chunks with error boundaries; the entry bundle budget is
  180 KB, enforced by `npm run check:bundle` and `__tests__/bundle-boundaries.test.ts`.
- **Console:** nav rail · stage (scene header, A/B swipe viewer with zoom/pan and SVG
  bbox layer, evidence tray, KPI cards) · thread panel (chat, grounded answer with
  citation pills, pipeline pulse, history). Lazy DAG dialog (@xyflow + dagre).
- **State** (`state/`): `job` (pure SSE reducer, tested on a recorded run), `ui`,
  `focus`, `map`, `stac`, `library`, `exports`, `settings` (online features off by
  default), `theme` (light/dark), `notifications`, `account`, `shortcuts`, `landing`.
- **API:** `api/client.ts` typed against generated `api/schema.d.ts`; `api/sse.ts`
  with last-event-id; `thread/useRun.ts` submit → stream → result / cancel
  (`DELETE /v1/jobs/{id}`) / bounded reattach (`thread/resume.ts`).
- **Evidence:** `thread/annotate.ts` maps citation claims to spans (longest first,
  digit boundary); `thread/boxes.ts` consumes the `BBOX_SET` artifact, `thread/bbox.ts`
  (TS port of `box_format.py`) is the text fallback; `BboxOverlay.tsx` draws in
  `viewBox="0 0 1000 1000"` sized to the raster's `object-fit: contain` rectangle.
- **Exports** (`export/`): `sitrep/` (one-page PDF via pdf-lib, lazy chunk, embeds
  Geist subsets), `geojson/` (RFC 7946, WGS84 when bounds exist else explicitly pixel
  space, optional vectorised masks), `download.ts`.
- **Geo** (`geo/`): `nominatim.ts` (1 req/s), `stac.ts`, `planetary.ts` (SAS tokens),
  `collections.ts`, `deadline.ts` (5 s network deadline).
- **Offline:** `?mock=1` starts MSW with recorded fixtures, including Nominatim/STAC
  replies and two real S1 RTC GeoTIFFs under `public/samples/stac/`
  (`DOCS/STAC_FIXTURES.md`); Playwright proves the whole flow with the network off.
- `styles/theme.css` is a spec: every hue has fill/text/on-fill tokens with computed
  WCAG ratios, checked by `contrast.test.ts`.

## 6. Training and evaluation subsystems (detail in `06`)

- `training/corpus_builder.py`: unified sample schema, per-source builders, streaming
  reservoir sampling, `Deduplicator` (SHA-256 → DCT pHash ≤ 5 → quarantine
  `LeakageError`), build-time citation audit, `HashTally`, `composition.json` / `MANIFEST.md`.
- `training/local_sources.py`: correct readers for VRSBench, RSVQA-HR (`active` flag), CDVQA.
- `training/vlm/qlora.py`: pydantic profile, NF4 + LoRA (LLM projections + last-N ViT
  blocks by name), VRAM estimator, **prompt-completion records with
  `completion_only_loss`**, `audit_masks()` before weights load, resumable, SIGINT-safe,
  writes `run_manifest.json`, `mask_audit.json`, `adapter/layout.json`.
- `training/cd/*`: Lightning Siamese CD (SSL4EO-S12 ResNet encoder, FPN decoder).
- `training/data/builders/evidence_qa.py` (top-level `training/`): synthetic citation source.
- `eval/`: `sampler.py` (stratified, round-robin by task, writes `sample_ids.json`),
  `runner.py` (through `LazyBackend`, resume-safe `predictions.jsonl`), `scorers.py`
  (pure; exact match, set-F1, Hungarian IoU, citation precision, BLEU/ROUGE),
  `report.py` (`results.{md,json,csv}` + `confusion/`).
