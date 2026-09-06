# SatQuery AI — API Contract (FROZEN)

**Status:** FROZEN as of Phase 0. **Schema version:** `1.0`
**Authority:** Derived strictly from `DOCS/Master.md` §3, §4, §5, §8-Phase-0.
**Audience:** Frontend engineer (primary consumer), backend implementers.

> **Frozen means:** fields may be *added* within `1.0` (additive, optional, defaulted). Renaming, removing, retyping, or changing the meaning of any field requires a bump to `2.0` and a written migration note in this file. The frontend may assume every `1.0` field below exists for the life of the project.
>
> Generated artifact of record: `openapi.json` (produced by `scripts/export_openapi.py`). **This document is the prose contract; `openapi.json` is the machine contract. If they disagree, this document wins and `openapi.json` is regenerated.**

---

## 1. Conventions

| Concern | Rule |
|---|---|
| Base URL | `http://localhost:8000` (dev). All paths below are relative to it. |
| API prefix | `/v1` on every functional route. |
| Auth | **None.** Hackathon scope. No `Authorization` header is read. Do not add one. |
| CORS | `*` origins, methods `GET, POST, OPTIONS`, headers `*`, `expose_headers: ["X-Trace-Id"]`. |
| JSON casing | `snake_case` everywhere. No exceptions. |
| Timestamps | ISO-8601 UTC with explicit `Z` suffix, e.g. `2026-09-05T12:00:00Z`. |
| Durations | Integer **milliseconds**, field suffix `_ms`. |
| Lengths / areas | Metres (`_m`), square metres (`_m2`), square kilometres (`_km2`). Never mixed. |
| Percentages | Float `0.0-100.0`, field suffix `_pct`. **Not** a 0-1 fraction. |
| Confidences | Float `0.0-1.0`, rounded to 4 decimal places. Field name `confidence`. |
| Native bounds | `[minx, miny, maxx, maxy]` in the raster's own CRS. |
| WGS84 bounds | `[min_lon, min_lat, max_lon, max_lat]` — **lon first**, EPSG:4326 axis-order-normalised. |
| Affine transform | rasterio/GDAL 6-tuple `[a, b, c, d, e, f]` where `x = a*col + b*row + c`, `y = d*col + e*row + f`. |
| Pixel boxes | `[x_min, y_min, x_max, y_max]`, integer, top-left origin, `x_max`/`y_max` exclusive. |
| Nulls | A field that is *inapplicable* is `null`. A field that is *unknown* is `null` **and** carries an entry in the nearest `warnings` array. Never `0`, never `""`, never omitted. |
| Trace id | 32-char lowercase hex. Echoed on every response in the `X-Trace-Id` header. |
| Ids | `img_{n}` for inputs, `art_{n}` for artifacts — both zero-indexed, stable within a trace. |

---

## 2. Enumerations

All enums are **closed**. A client receiving an unknown value must treat it as a contract violation and surface it, not silently coerce.

### 2.1 `TaskType`

Maps 1:1 to the mandatory capabilities in the problem statement.

| Value | Requirement served | Min images | Pair types |
|---|---|---|---|
| `VQA` | req 2 (**mandatory**) | 1 | SINGLE, CROSS_MODAL |
| `CAPTION` | req 2 (option A) | 1 | SINGLE, CROSS_MODAL |
| `GROUNDING` | req 2 (option B) | 1 | SINGLE, CROSS_MODAL |
| `SEGMENTATION` | req 2 (option B) | 1 | SINGLE |
| `COUNT` | RSVQA counting | 1 | SINGLE |
| `SCENE_CLASSIFY` | BigEarthNet-style multi-label | 1 | SINGLE, CROSS_MODAL |
| `CHANGE_VQA` | req 3 (**mandatory**) | 2 | BI_TEMPORAL |
| `CHANGE_CAPTION` | req 3 (**mandatory** alt.) | 2 | BI_TEMPORAL |
| `CHANGE_MAP` | req 3 (spatial change map) | 2 | BI_TEMPORAL |
| `CROSS_MODAL_VQA` | req 4 (**mandatory**) | 2 | CROSS_MODAL |
| `CROSS_MODAL_COMPARE` | req 4 | 2 | CROSS_MODAL |
| `UNSUPPORTED` | classifier fallback | 1 | any |

### 2.2 `PairType`

`SINGLE` · `CROSS_MODAL` · `BI_TEMPORAL` · `INCOMPATIBLE`

### 2.3 `Modality`

`optical` · `sar` · `panchromatic` · `unknown`

### 2.4 `ImageRole`

`single` · `pre` · `post` · `optical` · `sar` · `null`

`pre`/`post` are used for `BI_TEMPORAL`; `optical`/`sar` for `CROSS_MODAL`; `single` for `SINGLE`. Roles are assigned by the server unless overridden via `options.roles`.

### 2.5 `ArtifactType`

| Value | Payload | Also emitted as |
|---|---|---|
| `RENDERED_VIEW` | named 3-channel view (see `DATA_ADAPTATION_PLAN.md` §2) | PNG |
| `CHANGE_MASK` | binary/semantic change raster | PNG + GeoTIFF |
| `SEGMENTATION` | class-indexed mask | PNG + GeoTIFF |
| `BBOX_SET` | detection/grounding boxes | JSON + GeoJSON |
| `HEATMAP` | continuous field (index, agreement, confidence) | PNG + GeoTIFF |
| `OVERLAY_PNG` | evidence composited over a basemap view | PNG |
| `GEOJSON` | vector features in EPSG:4326 | GeoJSON |
| `SCALARS` | numeric-only result, no raster | JSON (inline) |
| `TEXT` | free text produced by a tool | JSON (inline) |

### 2.6 `CheckName`

`crs_match` · `bounds_overlap_iou` · `gsd_ratio` · `coregistration_offset_px` · `band_sufficiency` · `modality_distinct` · `temporal_ordering` · `georeference_present` · `size_ratio` · `nodata_extent`

### 2.7 `CheckStatus` / `Overall`

`CheckStatus`: `PASS` · `WARN` · `FAIL` · `SKIP`
`CompatibilityReport.overall`: `PASS` · `PASS_WITH_WARNINGS` · `FAIL`

Derivation rule (deterministic): any `FAIL` → `FAIL`; else any `WARN` → `PASS_WITH_WARNINGS`; else `PASS`.

### 2.8 `ToolStatus`

`OK` · `DEGRADED` · `FAILED` · `SKIPPED`

- `DEGRADED` — the tool's declared `fallback` ran in its place, or it ran with reduced settings. Result is usable; confidence is penalised (see `AGENT_POLICY_DAG.md` §7).
- `FAILED` — no usable output. Only reachable when no fallback exists.
- `SKIPPED` — an upstream dependency failed, or capability matching excluded it.

### 2.9 `ToolCategory`

`analysis` · `vlm` · `cv` · `geo` · `fusion`

### 2.10 `Device`

`rocm:0` · `cpu` · `auto`

---

## 3. Core Object Schemas

### 3.1 `InputManifest`

One per uploaded image. Produced by Phase 1.

| Field | Type | Null? | Notes |
|---|---|---|---|
| `id` | string | no | `img_0`, `img_1`, … in upload order |
| `role` | `ImageRole` | yes | server-assigned unless overridden |
| `filename` | string | no | original client filename, sanitised |
| `sha256` | string | no | 64-char lowercase hex of raw bytes |
| `size_bytes` | int | no | |
| `driver` | string | no | rasterio driver: `GTiff`, `PNG`, `JPEG` |
| `modality` | `Modality` | no | |
| `modality_confidence` | float | no | 0-1 |
| `sensor_guess` | string | yes | `Sentinel-2 L2A`, `Sentinel-1 GRD`, `Cartosat-2S`, `RISAT-1`, … |
| `crs` | string | yes | `EPSG:32643`; `null` for non-georeferenced |
| `transform` | float[6] | yes | |
| `bounds_native` | float[4] | yes | |
| `bounds_wgs84` | float[4] | yes | |
| `gsd_m` | float | yes | ground sample distance, metres |
| `width` / `height` | int | no | pixels |
| `band_count` | int | no | |
| `dtype` | string | no | `uint8`, `uint16`, `int16`, `float32` |
| `band_names` | string[] | yes | `["B02","B03",…]` or `["VV","VH"]` |
| `nodata_pct` | float | no | 0-100 |
| `acquisition_time` | datetime | yes | from TIFF/GDAL metadata if present |
| `is_georeferenced` | bool | no | `crs != null && transform != null` |
| `warnings` | string[] | no | may be empty |

### 3.2 `CheckResult`

| Field | Type | Null? | Notes |
|---|---|---|---|
| `name` | `CheckName` | no | |
| `status` | `CheckStatus` | no | |
| `value` | float \| string | yes | measured value |
| `threshold` | string | yes | human-readable rule, e.g. `"iou >= 0.80"` |
| `detail` | string | no | one sentence, safe to display verbatim |

**Thresholds (frozen):**

| Check | PASS | WARN | FAIL |
|---|---|---|---|
| `crs_match` | identical CRS | differ, reprojected successfully | reprojection impossible |
| `bounds_overlap_iou` | `>= 0.80` | `0.30 <= iou < 0.80` | `< 0.30` |
| `gsd_ratio` | `<= 1.5` | `1.5 < r <= 4.0` | `> 4.0` |
| `coregistration_offset_px` | `<= 1.0` | `1.0 < d <= 5.0` | `> 5.0` |
| `band_sufficiency` | required bands present | substitute used | required band absent, no substitute |
| `modality_distinct` | modalities differ (CROSS_MODAL) | one `unknown` | both identical for CROSS_MODAL |
| `temporal_ordering` | timestamps present and ordered | timestamps absent, order assumed from upload | timestamps present and equal |
| `georeference_present` | both georeferenced | neither (benchmark PNG path) | exactly one |
| `size_ratio` | `<= 1.5` | `<= 4.0` | `> 4.0` |
| `nodata_extent` | `< 5 %` | `5-40 %` | `> 40 %` |

`gsd_ratio` and `size_ratio` are always computed as `max/min`, so they are `>= 1.0`.

### 3.3 `CompatibilityReport`

| Field | Type | Null? | Notes |
|---|---|---|---|
| `pair_type` | `PairType` | no | |
| `pair_type_source` | `metadata` \| `heuristic` \| `user_declared` | no | `user_declared` when `options.pair_type_hint` was supplied |
| `checks` | `CheckResult[]` | no | every applicable check, including `SKIP` |
| `overall` | `Overall` | no | derived per §2.7 |
| `actions_taken` | string[] | no | e.g. `"reprojected img_1 EPSG:4326->EPSG:32643 (bilinear)"` |
| `common_grid` | `CommonGrid` | yes | `null` for `SINGLE` and for `FAIL` |

`CommonGrid`: `{ crs: string, transform: float[6], width: int, height: int, gsd_m: float, resampling: "nearest"|"bilinear"|"cubic" }`

### 3.4 `ArtifactRef`

| Field | Type | Null? | Notes |
|---|---|---|---|
| `id` | string | no | `art_0`, … |
| `type` | `ArtifactType` | no | |
| `mime` | string | no | `image/png`, `image/jpeg`, `image/tiff`, `application/geo+json`, `application/json` |
| `label` | string | no | display name, e.g. `"Change mask (binary)"`, `"Image 3 (optical NDVI)"` |
| `url` | string | yes | primary render; `null` for `SCALARS`/`TEXT` |
| `geotiff_url` | string | yes | georeferenced companion, when applicable |
| `geojson_url` | string | yes | vector companion, when applicable |
| `geo` | `{crs, transform}` | yes | `null` when the artifact is not georeferenced |
| `width` / `height` | int | yes | pixels, for raster artifacts |
| `stats` | object | yes | free-form, artifact-type specific |
| `inline` | object | yes | payload for `SCALARS`/`BBOX_SET`/`TEXT` — avoids a second round trip |
| `produced_by_step` | int | no | index into `plan.steps` |

**`inline` shape for `BBOX_SET`:**

```json
{ "boxes": [ { "label": "aircraft", "score": 0.91,
               "bbox_px": [412, 88, 501, 160],
               "bbox_wgs84": [77.1021, 28.5544, 77.1039, 28.5559] } ] }
```

### 3.5 `InputContract` (inside `ToolSpec`)

```jsonc
{ "pair_types": ["BI_TEMPORAL"], "modalities": ["optical","sar"],
  "min_images": 2, "max_images": 2,
  "required_bands": ["red","nir"],        // logical band names, resolved per-sensor
  "optional_bands": ["swir1"],
  "gsd_range_m": [0.3, 30.0],
  "min_size_px": 256, "max_size_px": 20000,
  "requires_georeference": true }
```

### 3.6 `ToolSpec` (served by `GET /v1/registry`)

| Field | Type | Notes |
|---|---|---|
| `name` | string | registry key, snake_case, stable |
| `version` | string | `"1.0.0"` or `"1.0.0+ckpt:9a3f"` |
| `category` | `ToolCategory` | |
| `description` | string | one line, shown in the frontend capabilities panel |
| `accepts` | `InputContract` | |
| `produces` | `ArtifactType[]` | |
| `scalars_schema` | object | JSON Schema of this tool's FactSheet contribution |
| `device` | `Device` | |
| `est_ms` | int | typical wall time, for the progress bar |
| `fallback` | string \| null | name of the tool used when this one fails |
| `available` | bool | **runtime** — false when weights/backend are missing |
| `unavailable_reason` | string \| null | |

### 3.7 `PlanStep`

```jsonc
{ "step": 2, "tool": "siamese_change_detector", "depends_on": [1],
  "input_refs": ["img_0","img_1"], "reason": "policy_table:CHANGE_VQA|BI_TEMPORAL|optical" }
```

`step` is 1-indexed and equals the step's position in topological order.

### 3.8 `Execution`

| Field | Type | Notes |
|---|---|---|
| `step` | int | matches `PlanStep.step` |
| `tool` / `version` | string | resolved at execution time |
| `status` | `ToolStatus` | |
| `device_used` | `Device` | |
| `duration_ms` | int | |
| `params` | object | **exact** effective parameters — the auditability payload |
| `input_refs` | string[] | `img_*` and/or `art_*` |
| `output_refs` | string[] | `art_*` |
| `scalars` | object | merged into `fact_sheet` under this tool's namespace |
| `confidence` | float | |
| `cache_hit` | bool | |
| `fallback_of` | string \| null | populated when this execution replaced a failed tool |
| `error` | string \| null | |

### 3.9 `Answer`

```jsonc
{ "text": "Approximately 7.4% of the scene (1.07 km2) transitioned to built-up...",
  "citations": [ { "claim": "7.4% of the scene",
                   "source": "step:2/scalars.changed_area_pct",
                   "value": 7.4 } ],
  "uncited_numeric_spans": [],
  "generator": "vlm_change_vqa@1.0.0+adapter:sq-lora-v3",
  "template_fallback": false }
```

`source` grammar: `step:{n}/scalars.{dotted.path}`. For a claim spanning two scalars, join with `|`:
`"step:3/scalars.ndbi_mean_pre|ndbi_mean_post"`.

`uncited_numeric_spans` lists any number the model emitted that the `CitationValidator` could not resolve. **A non-empty array is a visible honesty signal — render it.** See `AGENT_POLICY_DAG.md` §6.

### 3.10 `Confidence`

```jsonc
{ "overall": 0.86, "method": "weighted_tool_agreement_v1",
  "components": { "task_classification": 0.94, "input_quality": 0.97,
                  "tool_mean": 0.88, "cross_tool_agreement": 0.81 },
  "caps_applied": [] }
```

Formula and caps: `AGENT_POLICY_DAG.md` §7. `caps_applied` names any rule that clamped `overall`.

### 3.11 `AuditTrace`

The mandatory deliverable of requirement 5. Top-level keys, in order:

`trace_id` · `schema_version` · `created_at` · `duration_ms` · `query` · `resolved_task` · `inputs` · `compatibility` · `plan` · `executions` · `artifacts` · `fact_sheet` · `answer` · `confidence` · `warnings` · `errors`

```jsonc
"query":         { "raw": "...", "normalized": "...", "language": "en" },
"resolved_task": { "primary": "CHANGE_VQA", "secondary": ["CHANGE_MAP"],
                   "slots": { ... }, "confidence": 0.94,
                   "classifier": "rules_v1+llm_slotfill_v1" },
"plan":          { "planner": "policy_table_v1",
                   "policy_key": "CHANGE_VQA|BI_TEMPORAL|optical",
                   "steps": [ PlanStep, ... ] },
"fact_sheet":    { "<tool>.<scalar>": <number|string>, ... },
"warnings":      [ { "code": "...", "message": "...", "ref": "img_1" } ],
"errors":        [ { "code": "...", "message": "...", "step": 4 } ]
```

`fact_sheet` keys are **namespaced by tool name** to prevent collisions (`change_statistics.changed_area_pct`). The un-namespaced alias in Master.md §4.4's illustrative JSON is a display convenience only; the wire format is namespaced. Citation `source` strings always use the `step:N/scalars.*` form, which is unambiguous regardless.

---

## 4. Endpoints

### 4.1 `POST /v1/analyze`

Synchronous analysis. Use for anything expected under 30 s; otherwise use `/v1/jobs`.

**Request** — `multipart/form-data`

| Part | Type | Required | Notes |
|---|---|---|---|
| `images` | file[] | yes | 1-2 files. GeoTIFF/TIFF preferred; PNG/JPEG accepted (benchmark path). |
| `query` | text | yes | 1-1000 chars, UTF-8. |
| `options` | text (JSON) | no | see below |

**`options`**

```jsonc
{ "pair_type_hint": "BI_TEMPORAL",      // forces PairType; sets pair_type_source="user_declared"
  "roles": { "before.tif": "pre", "after.tif": "post" },
  "include_trace": true,                 // default true
  "include_rendered_views": true,        // default true - emit RENDERED_VIEW artifacts
  "artifact_format": "png+geotiff",      // "png" | "png+geotiff" (default "png+geotiff")
  "vlm_backend": "auto",                 // "auto" | "hf" | "llamacpp" | "none"
  "adapter_version": null,               // pin a LoRA adapter; null = server default
  "enable_tools": null,                  // string[] allow-list, null = all
  "disable_tools": [],                   // string[] deny-list
  "max_latency_ms": 30000,
  "seed": 0 }                            // determinism; 0 = server default
```

**Response** `200` — `AnalyzeResponse`

```jsonc
{ "trace_id": "b3f1...",
  "answer": Answer,
  "artifacts": [ ArtifactRef, ... ],
  "confidence": Confidence,
  "compatibility": CompatibilityReport,   // duplicated at top level for cheap UI access
  "resolved_task": { ... },               // duplicated at top level
  "trace": AuditTrace | null }            // null when options.include_trace=false
```

**Status codes:** `200` success (including `DEGRADED` executions) · `400` malformed request · `413` payload too large · `415` unsupported format · `422` inputs unusable for any task · `503` model/device unavailable · `500` unhandled.

> **Contract guarantee:** a tool failure never produces a 5xx. It produces `200` with `status: "DEGRADED"` or `"FAILED"` on that execution, a populated `errors[]`, and a reduced `confidence.overall`. The frontend must render partial results.

### 4.2 `POST /v1/jobs`

Identical multipart body to `/v1/analyze`. Returns `202`:

```json
{ "job_id": "b3f1...", "status": "queued",
  "poll_url": "/v1/jobs/b3f1...", "events_url": "/v1/jobs/b3f1.../events" }
```

`job_id` **is** the eventual `trace_id`.

### 4.3 `GET /v1/jobs/{job_id}`

```jsonc
{ "job_id": "b3f1...", "status": "queued"|"running"|"succeeded"|"failed",
  "stage": "executing", "step": 2, "total_steps": 4, "pct": 45,
  "created_at": "...", "updated_at": "...",
  "result": AnalyzeResponse | null, "error": ApiError | null }
```

### 4.4 `GET /v1/jobs/{job_id}/events` — SSE

`Content-Type: text/event-stream`. See §5.

### 4.5 `POST /v1/validate`

Ingestion + compatibility only. No tools run, no GPU touched. Cheap pre-flight for the upload UI.

**Request:** `multipart/form-data` with `images` (required) and `options` (optional; only `pair_type_hint` and `roles` are honoured). `query` is **not** required.

**Response** `200`:

```jsonc
{ "inputs": [ InputManifest, ... ],
  "compatibility": CompatibilityReport,
  "supported_tasks": ["CHANGE_VQA","CHANGE_CAPTION","CHANGE_MAP"],
  "warnings": [ ... ] }
```

`supported_tasks` is computed by running capability matching over the registry with these inputs — it lets the UI grey out impossible questions before the user types one.

### 4.6 `GET /v1/artifacts/{trace_id}/{artifact_id}.{ext}`

`ext` in `png` · `jpg` · `tif` · `geojson` · `json`. Returns raw bytes with the correct `Content-Type`. (`jpg` was added in Phase 2 — see §9 row 11.)
Headers: `Cache-Control: public, max-age=31536000, immutable` (artifacts are content-addressed and never mutate), `ETag`.
`404` if unknown; `410` if the trace has been evicted.

### 4.7 `GET /v1/traces/{trace_id}`

Returns the full `AuditTrace`. `404` if unknown.

### 4.8 `GET /v1/registry`

```jsonc
{ "registry_version": "1.0.0", "tools": [ ToolSpec, ... ],
  "task_types": [ ... ], "adapter_version": "sq-lora-v3" }
```

Drives the frontend "capabilities" panel. `available: false` entries must be shown as disabled, not hidden — the judges should see the registry is real.

### 4.9 `GET /v1/health`

```jsonc
{ "status": "ok"|"degraded", "version": "0.1.0", "schema_version": "1.0",
  "device": { "backend": "rocm", "name": "AMD Radeon RX 7900 XTX",
              "hip_visible_devices": "0", "vram_total_mb": 24576,
              "vram_used_mb": 9210, "igpu_masked": true },
  "models": [ { "name": "qwen3-vl-8b", "backend": "hf", "loaded": true,
                "adapter": "sq-lora-v3" } ],
  "tools_available": 11, "tools_total": 13 }
```

`igpu_masked` asserts `HIP_VISIBLE_DEVICES=0` took effect (Master.md §9, risk row 2). A `false` here is a red flag before any demo.

---

## 5. SSE Event Protocol

Each message: `event: <type>` + `data: <json>`. Heartbeat comment `: ping` every 15 s.

| `event` | `data` |
|---|---|
| `queued` | `{ "job_id": "..." }` |
| `stage` | `{ "stage": "ingesting"\|"validating"\|"rendering"\|"planning"\|"executing"\|"aggregating"\|"done", "pct": 20 }` |
| `plan` | `{ "steps": [ PlanStep, ... ] }` — emitted once, lets the UI draw the DAG before execution |
| `step_started` | `{ "step": 2, "tool": "siamese_change_detector", "est_ms": 1800 }` |
| `step_completed` | `{ "step": 2, "status": "OK", "duration_ms": 1830, "confidence": 0.88, "output_refs": ["art_3"] }` |
| `artifact` | `ArtifactRef` — emitted as soon as each artifact is written, so evidence streams in |
| `answer_delta` | `{ "text": "..." }` — token chunks during VLM synthesis |
| `done` | `AnalyzeResponse` |
| `error` | `ApiError` |

Ordering guarantee: `queued` → `stage(ingesting…planning)` → `plan` → interleaved `step_*`/`artifact` → `answer_delta`* → `done` \| `error`. `done` and `error` are terminal and mutually exclusive.

---

## 6. Error Taxonomy

```jsonc
{ "error": { "code": "INSUFFICIENT_OVERLAP", "http_status": 422,
             "message": "The two images overlap by only 12% of their combined extent. Bi-temporal analysis needs at least 30%.",
             "hint": "Upload images covering the same footprint, or crop them to their shared area.",
             "ref": "img_1", "trace_id": "b3f1..." } }
```

| Code | HTTP | Meaning |
|---|---|---|
| `INVALID_RASTER` | 400 | rasterio cannot open the file |
| `TOO_MANY_IMAGES` | 400 | more than 2 images |
| `EMPTY_QUERY` | 400 | `query` missing or blank on `/v1/analyze` |
| `IMAGE_TOO_LARGE` | 413 | exceeds `max_upload_mb` (default 512) or `max_size_px` |
| `UNSUPPORTED_FORMAT` | 415 | driver not in the allow-list |
| `MISSING_GEOREFERENCE` | 422 | exactly one image is georeferenced (mixed pair is unresolvable) |
| `CRS_MISMATCH_UNRESOLVABLE` | 422 | reprojection failed |
| `INSUFFICIENT_OVERLAP` | 422 | `bounds_overlap_iou < 0.30` |
| `GSD_MISMATCH` | 422 | `gsd_ratio > 4.0` |
| `MODALITY_AMBIGUOUS` | 422 | CROSS_MODAL requested but both images classify identically |
| `QUERY_UNCLASSIFIABLE` | 422 | classifier below threshold **and** generic fallback disabled |
| `NO_CAPABLE_TOOL` | 422 | capability matching yielded an empty DAG |
| `TOOL_FAILED_NO_FALLBACK` | 500 | required tool failed with no `fallback` declared |
| `VRAM_EXHAUSTED` | 503 | HIP OOM; retryable |
| `MODEL_UNAVAILABLE` | 503 | weights or backend missing |

`message` is user-facing and safe to display verbatim. `hint` is an actionable next step. Never leak file paths or stack traces into either.

---

## 7. Worked Examples

These four are the `make demo` scenarios (Master.md §10) and are the fixtures behind Phase 0's mocked responses.

### 7.1 Single optical — *"Describe this scene and identify water bodies."*

Input: one 12-band Sentinel-2 GeoTIFF, EPSG:32643, 10 m.
`pair_type: SINGLE` · `resolved_task.primary: CAPTION`, `secondary: ["SEGMENTATION"]`
Plan: `spectral_renderer` → `spectral_index_analyzer` → `vlm_caption`
Artifacts: `RENDERED_VIEW` x3 (TC, FCIR, NDWI), `HEATMAP` (NDWI), `SEGMENTATION` (water mask, PNG+GeoTIFF), `OVERLAY_PNG`
Citations resolve to `spectral_index_analyzer.ndwi_mean`, `.water_fraction_pct`.

### 7.2 Single SAR — *"What kind of terrain is this?"*

Input: one 2-band VV/VH GeoTIFF.
`resolved_task.primary: VQA` · Plan: `spectral_renderer` → `raster_statistics` → `sar_backscatter_analyzer` → `vlm_vqa`
Answer must cite `sar_backscatter_analyzer.sigma0_vv_mean_db` and `.vv_vh_ratio_mean`.

### 7.3 Cross-modal — *"What does the SAR show that the optical misses?"*

Input: co-registered Sentinel-2 + Sentinel-1 pair.
`pair_type: CROSS_MODAL` · `resolved_task.primary: CROSS_MODAL_COMPARE`
Plan: `spectral_renderer` → (`spectral_index_analyzer` || `sar_backscatter_analyzer`) → `crossmodal_consistency` → `physics_agreement` → `vlm_vqa`
Artifacts include a `HEATMAP` agreement map. `compatibility.checks` must contain a `PASS` `modality_distinct`.

### 7.4 Bi-temporal — *"How much built-up area appeared?"*

The canonical example. Full trace in Master.md §4.4, with these contract-level corrections applied:

- Tool name is `siamese_change_detector` (registry key); `changeformer_levircd` in the Master.md illustration is a checkpoint id and belongs in `version` as `"1.0.0+ckpt:9a3f"` — see §9.
- `fact_sheet` keys are namespaced: `change_statistics.changed_area_pct`, `spectral_index_analyzer.ndbi_mean_pre`.

---

## 8. Frontend Integration Notes

1. **Start against the mock.** Phase 0 ships every endpoint returning schema-valid fixtures. `openapi.json` is committed — generate a typed client from it (`openapi-typescript`, `orval`).
2. **Call `/v1/validate` on file select, before the user types.** It is fast and GPU-free, and `supported_tasks` tells you which questions to offer.
3. **Prefer `/v1/jobs` + SSE for anything bi-temporal.** Change detection alone is ~1.8 s; a full VLM synthesis pushes past a comfortable sync wait.
4. **Draw the DAG from the `plan` SSE event.** The execution trace *is* the product — a visible tool graph filling in step by step is the single most persuasive thing on screen for the judges.
5. **Render `uncited_numeric_spans` when non-empty.** It demonstrates the anti-hallucination guard is live rather than claimed.
6. **Never hide `DEGRADED`.** Show the fallback badge and the reduced confidence.
7. **Artifacts are immutable and cacheable forever.** Safe to preload every `ArtifactRef.url`.
8. **`RENDERED_VIEW` artifacts are the evidence gallery.** They are literally the images the VLM saw — showing them beside the answer is the grounding story.

---

## 9. Resolved Ambiguities & Deltas from `Master.md`

Recorded so the delta is reviewable rather than silent. Nothing here contradicts Master.md; each item fills a gap it left open.

| # | Item | Master.md state | Resolution | Rationale |
|---|---|---|---|---|
| 1 | Change-detection tool name | §5 says `siamese_change_detector`; §4.4 trace shows `changeformer_levircd` | Registry key is **`siamese_change_detector`**; checkpoint identity lives in `version` | §5 is the registry table; §4.4 is an illustration. One name per tool, or capability matching breaks. |
| 2 | `RENDERED_VIEW` artifact type | Not in §4.3's list | **Added** to `ArtifactType` | §3② mandates the renderer feed "BOTH the VLM AND the frontend as PNG evidence" — that requires an artifact type. |
| 3 | `physics_agreement` tool | In §8 Phase 6 build list, absent from §5 registry | **Added** as a registry entry distinct from `crossmodal_consistency` | Phase 6 names two modules; the deterministic rule layer has different `accepts` (no georeference needed, any band count) and must be independently selectable. |
| 4 | `fact_sheet` key collisions | §4.4 shows flat keys | **Namespaced** `<tool>.<scalar>` | Two tools both emit `mean` — flat keys silently overwrite. Citations use `step:N/...` and are unaffected. |
| 5 | `TaskType` enum members | Referenced, never enumerated | Enumerated in §2.1 | A frozen contract requires a closed set. Every member maps to a stated requirement. |
| 6 | Check thresholds | §8 Phase 1 gives "30 % overlap → WARN" and "±0.5 px recovery" | Full threshold table in §3.2, consistent with those two anchors | Testable, and the frontend needs them to explain a WARN. |
| 7 | Error codes | §8 Phase 9 says "error taxonomy" | Defined now in §6 | The taxonomy is part of the frozen contract; deferring it to Phase 9 would break the frontend late. |
| 8 | `ArtifactRef.inline` | Not specified | **Added** | Boxes and scalars are small; a second HTTP round trip per box set is wasteful. |
| 9 | `available` on `ToolSpec` | Not specified | **Added** | The registry is served live; a tool whose weights are absent must be visibly disabled, not silently missing. |
| 10 | Percent convention | Mixed in examples | Fixed: `_pct` is always 0-100 | `changed_area_pct: 7.4` in §4.4 confirms this reading. |
| 11 | Artifact extension `jpg` | §4.6 listed `png` · `tif` · `geojson` · `json`; §3.4's `mime` list omitted `image/jpeg` | **Added** `jpg` / `image/jpeg` to both lists (additive within `1.0`) | `DATA_ADAPTATION_PLAN.md` §2.1 mandates JPEG q92 for the reflectance composites (`TC`, `FCIR`, `SWIR`, `SARFC`, `PAN`) and PNG only for measurement views. Serving those composites requires the extension. Index, SAR and change views remain PNG — that half of the rule is load-bearing and unchanged. |
