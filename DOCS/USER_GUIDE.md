# SatQuery AI — User Guide

*From Space to Answers — geospatial intelligence that shows its work.*

This guide is for three readers: a **judge** who has minutes and wants to see the
machine think; a **tester** who needs to reproduce a run and check its claims; and a
**developer** who has to keep it running on a laptop that may have no internet. Every
path below was taken from the code as it stands (`frontend/src`, `src/satquery`,
`scripts/`), not from intentions. Where a feature has a limit, the limit is stated.

---

## Contents

1. [System overview & local-first philosophy](#1-system-overview--local-first-philosophy)
2. [Satellite imagery sourcing](#2-satellite-imagery-sourcing-step-by-step)
3. [Core workflow & UI features](#3-core-workflow--ui-features)
4. [Keyboard shortcuts & pro tips](#4-keyboard-shortcuts--pro-tips)
5. [Troubleshooting & common pitfalls](#5-troubleshooting--common-pitfalls)
6. [Appendix: quick reference](#appendix-quick-reference)

---

## 1. System overview & local-first philosophy

### 1.1 What SatQuery AI is

SatQuery AI is an **agentic vision-language assistant for satellite imagery** built for
Smart India Hackathon problem statement 26167 (ISRO / Space Applications Centre). The
one-line promise: *every number in an answer is bound to a measurement a tool made, or
it is flagged as uncited.*

A query goes through a fixed, deterministic pipeline:

```
 upload ──► ingest ──► validate (11 checks) ──► render views ──► plan (policy table)
                                                                      │
                     ┌────────────────────────────────────────────────┘
                     ▼
              execute DAG (specialist tools, in parallel where independent)
                     │
                     ▼
              aggregate FactSheet ──► VLM synthesises prose ──► CitationValidator
                     │
                     ▼
              Answer + citations + artifacts + versioned AuditTrace (SQLite)
```

The VLM is **one tool among many, never the system**. Control flow is chosen by
`configs/policy_table.yaml`, not by the model; re-running the same inputs produces a
byte-identical trace once timestamps are dropped (`scripts/e2e_parity.py` tests this).

### 1.2 Architecture summary

| Layer | What it is | Where |
|---|---|---|
| **Backend** | FastAPI, Python 3.11, pydantic-settings (`SATQUERY_*` env vars). Routers: `analyze`, `jobs` (SSE), `validate`, `imagery`, `artifacts`, `traces`, `registry`, `health`. | `src/satquery/api/` |
| **Ingest & validation** | rasterio readers, modality detection (optical / SAR), 11 named compatibility checks, phase-correlation co-registration. | `src/satquery/ingest/` |
| **Tool registry** | `spectral_renderer`, `raster_statistics`, `spectral_index_analyzer`, `sar_backscatter_analyzer`, `change_statistics`, `object_counter`, `image_diff_change`, `siamese_change_detector`, `semantic_segmenter`, `text_grounding`, `physics_agreement`, `crossmodal_consistency`, `vlm_vqa`, `vlm_caption`, `vlm_change_vqa`. | `configs/registry.yaml`, `src/satquery/tools/` |
| **VLM** | `Qwen/Qwen3-VL-8B-Instruct` + QLoRA adapter `runs/sq-lora-v2-full/adapter` (LoRA r=16, α=32). Served either as bf16 base + adapter through transformers (`SATQUERY_VLM_BACKEND=hf`) or merged → Q4_K_M GGUF behind `llama-server` (`SATQUERY_VLM_BACKEND=llamacpp`). Same interface either way. | `src/satquery/models/`, `scripts/serve_vlm.sh`, `scripts/merge_export.py` |
| **Audit trace** | `AuditTrace` built by the executor, schema-validated, persisted to `data/traces.sqlite3`, returned verbatim and served at `GET /v1/traces/{id}`. | `src/satquery/trace/` |
| **Frontend** | React 19 + Vite 6 + Tailwind v4, Zustand 5 state, TanStack Query 5, MapLibre GL for the Maps page, `@xyflow/react` for the live DAG, `pdf-lib` for SITREP. Fonts are self-hosted; nothing depends on a CDN. | `frontend/src/` |
| **Wire contract** | Frozen at schema `1.0`: `openapi.json` ↔ `frontend/src/api/schema.d.ts`, enforced by `make contract`. | `DOCS/API_CONTRACT.md` |

The frontend's sections are `landing`, `explore` (the console), `usecases`, `maps`,
`saved`, `projects`, `report/<trace>`. Navigation is a Zustand `section` string with a
thin URL router (`frontend/src/shell/router.ts`).

### 1.3 The three-tier offline model

The demo is designed to survive the network being pulled at any point. Each tier is a
strict subset of the one above it and needs no code change to switch.

| Tier | Network | VLM | What you get | How to start |
|---|---|---|---|---|
| **A — Cloud + Local VLM** | Internet available | Local (bf16 adapter or GGUF) | Everything: Maps-page STAC discovery over Planetary Computer, place search, optional basemap tiles, plus the full local pipeline. | `make demo` with `.env` pointing at the adapter/GGUF; enable **Settings → Imagery search on the Maps page**. |
| **B — Air-gapped Local VLM** | None | Local (bf16 adapter or GGUF) | The full pipeline on your own files (dropzone or fixtures). Maps page discovery is gated off and says so; scene stage still renders your imagery in geo or pixel space. | `make demo` (or the demo-laptop runbook in `scripts/demo_laptop/README.md`). Leave *Imagery search* off. |
| **C — Zero-network rehearsal** | None | None (recorded fixtures) | The whole UI — plan, live DAG, evidence tray, citations, SITREP, GeoJSON, Maps shelf — driven by MSW from `frontend/src/mocks/captured/`. Backend can be down. | Open `http://localhost:5173/?mock=1`. |

A fourth, partial mode exists for CPU-only machines: `make demo-cpu`
(`SATQUERY_VLM_BACKEND=none`) runs the real backend with templated answers so the DAG,
SSE stream, and citations can be checked without weights.

**Rule of thumb for a judge's table:** rehearse Tier C first (it always works), run
Tier B for the demo, and only turn on Tier A if the venue Wi-Fi has been tested from that
laptop that day.

### 1.4 Getting the stack up

```bash
# Python 3.11 via uv; Node >= 22.12 (frontend/.nvmrc)
uv sync --group dev                 # API + tools + test suite (no torch)
cd frontend && npm ci && cd ..

cp .env.example .env                # SATQUERY_* settings
make demo                           # API :8000 + Vite :5173 (Ctrl-C stops both)
```

To serve the fine-tuned model on the 24 GB GPU box:

```bash
uv sync --extra vlm                 # first time only — torch/transformers/accelerate
source scripts/rocm_env.sh          # every shell that touches the GPU (ROCm)
export SATQUERY_VLM_BACKEND=hf
export SATQUERY_VLM_ADAPTER_PATH=runs/sq-lora-v2-full/adapter
```

> **ROCm boxes:** the lockfile's CUDA torch has been replaced by hand. **Never run a bare
> `uv sync` or a plain `uv run`** there — use `uv run --no-sync …` or `.venv/bin/…`.
> See `DOCS/AI_HANDOFF/05_ENVIRONMENT_AND_SETUP.md`.

The HealthStrip in the console header reports the API, the VLM backend (`ready`), and
the tool count (`tools 6/6` when both checkpoints are configured). Nothing should be
demonstrated until it is green.

---

## 2. Satellite imagery sourcing (step-by-step)

SatQuery accepts **GeoTIFF** (preferred — carries CRS, transform, band metadata) and
plain **PNG/JPEG** (treated as non-georeferenced VHR; grounding and counting still work,
geometric checks are skipped). You need:

- **Single-image tasks:** one file.
- **Bi-temporal change:** two scenes of the **same sensor**, same footprint, same CRS,
  ideally same orbit (SAR) — T1 earlier than T2.
- **Cross-modal:** one optical scene + one SAR scene over the same footprint.

Three ways to get there, from easiest to most manual.

### Method A — Zero-download cloud fetch (Maps page HUD)

The Maps page searches Microsoft Planetary Computer's STAC catalogue, and the **backend**
clips the chosen scene(s) to your window and hands them to the console exactly as if you
had dropped the files. No key is needed for search; the SAS token for asset reads is
handled server-side (`src/satquery/ingest/stac_fetch.py`).

**Prerequisites**

1. Internet reachable from the machine running the *backend* (it does the clipping).
2. **Settings (`⌘,`) → Imagery search on the Maps page → on.** This is off by default so
   an air-gapped judging hall never sees a hanging request. In `?mock=1` it is always on
   and answers from `frontend/public/samples/stac/`.

**Steps**

1. Press `G` then `M` (or the Maps entry in the sidebar) to open the Maps page. Press
   `B` to toggle globe / flat projection if you like.
2. **Place search** (top of the discovery column): type at least 3 characters of a
   city or region — *"Ahmedabad"*, *"Chennai port"*. Results come from OSM Nominatim
   (1 request/second, cached for the session). Pick one; the map flies to its bounding
   box. You can also type coordinates into the **Coordinate locator**.
3. **Dates:** pick a window with the two date inputs or a preset — **30 d**, **6 mo**,
   **1 yr**. For change detection, a year apart is the demo's default vocabulary.
4. **Sensor (Collection picker):**
   - **Optical** → `sentinel-2-l2a` (bands B02 B03 B04 B08 B11 B12, 10 m).
   - **SAR** → `sentinel-1-rtc` (VV/VH, terrain-corrected, 10 m). RTC is preferred over
     GRD because it is what the adapter's BigEarthNet-v2 SAR training data looks like.
   - **Both** → one of each, for a cross-modal query.
   - When optical is in play a **cloud ceiling** slider appears; the STAC query filters
     `eo:cloud_cover < N`. 20 % is a sensible default; drop to 10 % for coastal cities in
     monsoon months.
5. **Scene shelf:** thumbnails appear sorted by date, each with its cloud percentage
   (optical) or orbit (SAR). Click a card — or click its footprint on the map — to select
   **T1**. For a change pair, tick *pair* and select **T2**. The **Time slider** lets you
   scrub the shelf by acquisition date; the **Swipe handle** compares two footprints on
   the map.
6. Read the **Fetch bar** at the bottom. It refuses a bad pair before any bytes move:
   - *"T1 and T2 are different sensors"* — pick two optical or two SAR.
   - *"These two scenes are on different orbits; the co-registration check will fail"* —
     SAR pairs must share a relative orbit.
7. Click **Analyse** (single) or **Analyse change** (pair). The label reports progress
   honestly: *"Clipping two scenes to the pipeline's window…"* then *"Fetching 1 of 2 —
   4.1 MB of 9.6 MB"*. The backend clips to at most `SATQUERY_IMAGERY_MAX_PX` pixels on
   the long side, so a fetched scene is a few MB, never a product zip.
8. The console opens with a suggested question already in the composer
   (*"What changed between these two images?"*, *"Is this area built-up?"*, *"What does
   the backscatter say about the surface here?"*) and **pre-flight is already running**.
   Edit the question or press Enter.

If a band the pipeline wanted was missing from the STAC item, the response carries an
`IMAGERY_BAND_MISSING` warning and it shows in the notification popover — the run
continues with the bands it has.

### Method B — Copernicus Browser export (manual, small files)

Use this when you want a specific scene the Maps page does not surface, or you are
preparing fixtures ahead of an offline demo. The goal is a **clipped, analysis-ready
16-bit GeoTIFF under ~25 MB**, not the 1 GB SAFE product zip.

1. Open <https://browser.dataspace.copernicus.eu/> and sign in (free Copernicus Data
   Space account).
2. **Search** tab → choose the data collection:
   - **Sentinel-2 → L2A** (surface reflectance, atmospherically corrected). Set
     **Max. cloud coverage** to 10–20 %. Avoid L1C unless you specifically want
     top-of-atmosphere values.
   - **Sentinel-1 → GRD → IW** (Interferometric Wide swath, the land-imaging mode), dual
     polarisation **VV+VH**. Note the **relative orbit** number of the scene you pick — a
     change pair needs the same one.
3. Set the **time range** and press *Search*. Click a result to load it into the
   **Visualize** tab.
4. **Area of interest:** use the **AOI box tool** (the pentagon/square icon on the
   right-hand rail) and draw a rectangle over your target. Keep it modest — roughly
   5 × 5 km to 15 × 15 km at 10 m works well; the pipeline renders 448 px views, so
   a 100 km scene gains nothing and costs VRAM.
5. **Export:** click the download icon → choose the **Analytical** tab (not *Basic*,
   which is a screenshot):
   - **Image format:** `TIFF (16-bit)`.
   - **Image resolution:** `Medium` or `High` — check the reported pixel dimensions;
     ~1000–2500 px on a side is the sweet spot.
   - **Coordinate system:** `UTM` (the native grid, e.g. EPSG:32643 for Gujarat) or
     `WGS 84` — both work; the pipeline reprojects the second raster of a pair to the
     first when CRS differs.
   - **Layers:** for Sentinel-2 tick the raw bands you need — **B02, B03, B04, B08, B11,
     B12** covers TC, FCIR, NDVI, NDBI and NDWI. Untick visualised layers such as
     *True color* (8-bit, already stretched). For Sentinel-1 tick **VV** and **VH** in
     linear (or dB — the SAR analyser reports which it detected).
   - **Clip extra data**: on (crops to the AOI).
   - Press **Download**. The result is one multi-band GeoTIFF (or one per band — the
     dropzone accepts either; if you exported single bands, stack them first in QGIS
     via *Raster → Miscellaneous → Build Virtual Raster* and export as GeoTIFF).
6. **For a change pair:** repeat steps 3–5 with the *same AOI* (Copernicus Browser keeps
   the drawn box between scenes) and a later date. Same sensor, same orbit for SAR.
7. **For a cross-modal pair:** export the S2 L2A bands and an S1 GRD IW VV+VH scene over
   the same AOI within a few weeks of each other.

Sanity-check before dropping: `gdalinfo file.tif` should show a CRS, a non-identity
transform, and 2–6 bands of `UInt16` or `Float32`.

### Method C — Built-in fixtures

Two sets exist; use whichever is on the machine.

| Location | What | Made by |
|---|---|---|
| `data/e2e/` (optional, ~20 MB, shipped on the demo USB stick) | Real clipped GeoTIFFs with the parity script's file names: `s2_pre.tif`, `s2_post.tif`, `s1_vvvh.tif`, `s2_disjoint.tif`, `benchmark_rgb.png`. | Clipped from CDVQA / BigEarthNet-v2 scenes with rasterio. |
| `data/fixtures/synthetic/` | Synthetic 256 px GeoTIFFs in EPSG:32643 (Delhi NCR origin) with *known* GSD, injected 3-px shifts and controlled footprint overlaps. | `uv run --no-sync python scripts/make_synthetic_fixtures.py` |

`scripts/e2e_parity.py` renders the synthetic set into a temp directory automatically;
point it at real scenes with `--scenes-dir data/e2e`. The five scenarios and their
fixtures are the demo scripts:

| Scenario | Drop | Ask |
|---|---|---|
| Single-image grounding | `benchmark_rgb.png` | *Where is the airplane? Highlight it with a bounding box.* |
| Bi-temporal change | `s2_pre.tif` + `s2_post.tif` | *What changed between these two images?* |
| Cross-modal optical + SAR | `s2_pre.tif` + `s1_vvvh.tif` | *Is this area built-up? Use both sensors together.* |
| Negative: no overlap | `s2_pre.tif` + `s2_disjoint.tif` | *What changed between these two images?* → **FAIL/DEGRADED** on `bounds_overlap_iou`, never a 500 |
| Negative: degraded plan | `s2_pre.tif` | *Describe this scene and identify water bodies.* with a tool disabled → PASS with the missing tool flagged |

If `data/e2e/` is absent on a clean clone, generate synthetic fixtures with the script
above; the pipeline behaviour is identical, only the pictures are duller.

---

## 3. Core workflow & UI features

### 3.1 The console: thread & dropzone

Press `G` `E` or `⌘K` to open **New Query**. The console has three regions: the
**thread** (composer, chat, results, citations, history tabs) on the left, the **data
stage** (image viewer, evidence tray, KPI cards) in the centre, and the **sidebar** with
Datasets / Tools / History panels.

**Ingestion**

1. Drag one or two files onto the **Dropzone**, or click it to browse. Accepted:
   `.tif/.tiff`, `.png`, `.jpg`. Order matters for pairs: the first file is **T1**.
2. The moment files land, the frontend calls `POST /v1/validate` and the **Preflight
   panel** opens. Nothing is analysed yet; this is free.
3. The **Manifest card** shows what the reader extracted per file: CRS (e.g.
   `EPSG:32643`), GSD in metres, size in pixels, band count, detected modality
   (`optical` / `sar`), nodata percentage, and any reader warnings.

**The 11 compatibility checks** (API_CONTRACT §2.6) each report `PASS`, `WARN`, `FAIL`
or `SKIP` with the measured value and the threshold:

| Check | What it measures | Typical failure |
|---|---|---|
| `georeference_present` | CRS and non-identity transform on every input | PNG screenshots → `WARN`, geometric checks `SKIP` |
| `crs_match` | Both inputs in one CRS (or reprojectable) | Mixed UTM zones → auto-reproject, noted in `actions` |
| `bounds_overlap_iou` | Footprint IoU of the pair | Disjoint scenes → `FAIL` |
| `gsd_ratio` | Ratio of ground sample distances | 10 m vs 0.65 m → `FAIL` (not comparable) |
| `coregistration_offset_px` | Phase-correlation shift between T1 and T2 | > threshold → `WARN`, shift is corrected at ingest |
| `band_sufficiency` | Bands needed for the planned indices | 3-band RGB asked for NDBI → `WARN`, NDBI skipped |
| `modality_distinct` | Cross-modal pair really is optical + SAR | Two optical files for a cross-modal question |
| `temporal_ordering` | T1 acquisition before T2 | Files dropped in the wrong order |
| `size_ratio` | Pixel dimensions comparable | Wildly different rasters |
| `nodata_extent` | Fraction of nodata pixels | Mostly-black clip → `WARN` |
| *(pair type)* | Classification into `single`, `bitemporal`, `crossmodal` | Drives which policy entries are eligible |

Expand **Compatibility details** to see every value. `FAIL` blocks the run with a
named check and a hint; `WARN` lets it proceed and is carried into the trace.

**Asking**

Type in the composer (press `/` to focus it) or pick a **Suggestion chip** — the chips are
generated from the detected pair type, so a cross-modal pair offers *"Is this area
built-up? Use both sensors together."* Press Enter. The frontend submits
`POST /v1/jobs` and opens the SSE stream.

### 3.2 Agentic pipeline & the live DAG

Press `P` on the stage (or click the **Pipeline pulse** in the thread) to open the
**Pipeline dialog**. It is drawn from the `plan` SSE event, which arrives **before any
step runs** — the whole graph appears at once, then lights up.

**Reading the graph**

- Each node is a registry tool; the edge means "needs the output of". Nodes with no
  edge between them are independent and the executor runs them **concurrently** — this
  is a DAG walk, not a list.
- Node states: *queued* (dim) → *running* (accent ring, elapsed timer) → *done* (emerald,
  duration) or *failed* (red, error code). Hover for the step's inputs and scalars.
- A node labelled with **`fallback_of`** means a substitution happened and the plan says
  so: e.g. `image_diff_change (fallback_of siamese_change_detector)` means
  `SATQUERY_CD_CHECKPOINT` was not set. That is honest, but for the demo you want the
  Siamese model — see §5.

**The three canonical shapes**

*Cross-modal optical + SAR* — six nodes, two parallel analysers:

```
spectral_renderer ─┬─► spectral_index_analyzer ─┐
                   │                            ├─► crossmodal_consistency ─► physics_agreement ─► vlm_vqa
                   └─► sar_backscatter_analyzer ─┘
```

Watch `spectral_index_analyzer` (NDVI / NDBI / NDWI means and histograms) and
`sar_backscatter_analyzer` (σ⁰ VV / VH in dB, VV/VH ratio) both in the *running* state
in the same frame. `physics_agreement` then checks that the optical and radar stories
agree (built-up ⇒ high NDBI **and** high VV backscatter) and emits a `verdict` the VLM
must respect.

*Bi-temporal change* — Siamese branch parallel to the index analyser:

```
spectral_renderer ─┬─► siamese_change_detector ─► change_statistics ─┐
                   │                                                  ├─► vlm_change_vqa
                   └─► spectral_index_analyzer ──────────────────────┘
```

`siamese_change_detector` produces the `CHANGE_MASK` artifact; `change_statistics`
turns it into `changed_area_pct`, `changed_area_ha`, and per-index deltas.

*Single-image grounding* — a chain: `spectral_renderer → text_grounding → vlm_vqa`, with
`object_counter` added when the question counts.

**The event stream** (`frontend/src/api/events.ts`): `queued → stage(ingesting …
planning) → plan → step_started / artifact / step_completed (interleaved) → stage
(aggregating) → answer_delta* → done`. Open DevTools → Network → the
`/v1/jobs/{id}/events` request → *EventStream* to watch it raw; `id:` increases
monotonically. Reloading mid-run recovers from `GET /v1/jobs/{id}`.

### 3.3 Evidence tray & citations

Artifacts appear in the **Evidence tray** *as they arrive*, before the answer. Use `←` /
`→` to step through views, `[` / `]` to nudge the compare-slider by 5 %.

**View groups** (`frontend/src/evidence/views.ts`): **TC** true colour, **FCIR** false
colour IR, **NDVI**, **NDBI**, **NDWI**, **SAR VV**, **SAR VH**, **CHANGE**. Pairs show T1
and T2 behind a **swipe** slider; change runs add the mask as an overlay with an opacity
control.

**Bounding boxes** from grounding come back in the `[0, 1000]` normalised frame and are
drawn by `BboxOverlay` over the exact view the VLM saw. A box is a *position claim* and
is validated as such — out-of-range or zero-area boxes are dropped and counted.

**Citations** — the part a judge should click:

- Every number in the answer that came from a tool is a **citation pill** whose
  `source` reads `step:N/scalars.<path>`, e.g. `step:2/scalars.changed_area_pct`.
- **Hover** a pill → the matching measurement card highlights in the tray.
  **Click** → the stage scrolls to that card and shows the step, the scalar, its unit,
  and the tool that measured it.
- The **Citations tab** in the thread lists every bound span; the **Results tab** lists
  the KPI cards (`frontend/src/kpi/registry.ts` picks which scalars become cards).
- A number the model produced that **no tool measured** is an **uncited numeric span**.
  The `CitationValidator` is `flag`-not-`strip` by default: the span stays in the text
  but is underlined in amber with an *uncited* marker, and `answer.uncited_numeric_spans`
  is non-empty. This is the honesty signal — an empty list means every figure is
  grounded.

To verify a citation independently: open **Sidebar → History → the trace**, find step N,
and read `scalars.<path>` in the raw trace. The value in the answer text must be the
value in the trace, verbatim.

### 3.4 Tactical SITREP (one-page PDF)

`⌘⇧S` (or the **SITREP** button in the thread's result header) builds a one-page PDF
brief **entirely in the browser** with `pdf-lib`; nothing is sent to a server, and it
works in Tier C. The code is lazy-loaded on first use (`sitrep-*` chunk).

**Layout of the page**

1. **Header:** scene id, sensor(s), acquisition date(s), CRS, generated-at timestamp,
   trace id.
2. **Scene panel:** the primary view (and the change mask / boxes drawn over it),
   composited from the same artifacts the tray shows.
3. **KPI table:** the rows the KPI cards show, through the same `selectKpis` — up to
   `MAX_CARDS`, each with value, unit, and the `step:N/scalars.…` it came from. If a
   number is in the PDF, it is in the trace.
4. **Answer:** the prose with citation spans marked, and uncited spans marked
   differently.
5. **Footer:** `citations: bound N · uncited M`, generator string
   (`…+adapter:sq-lora-v2-full`), schema version.

**The uncited warning badge:** when `uncited > 0` the PDF carries an amber badge in the
header. It does not mean the answer is wrong; it means *M* figures in the prose were not
produced by a tool and should be read as the model's opinion. A judge reading a SITREP
without the badge can trust every number to a measurement.

The file is named `sitrep-<sceneId>-<timestamp>.pdf` and saved through the browser's
download path. **Settings → Export** controls whether the download prompts for a
location.

### 3.5 GeoJSON exporter

`⌘⇧G` on the stage (or the **Export GeoJSON** button) writes an **RFC 7946**
FeatureCollection. If the run produced only boxes, the file downloads on the click; if it
also rendered a `CHANGE_MASK` or `SEGMENTATION` raster, a small dialog asks one question.

**What is in the file**

- **CRS:** WGS 84 / CRS84 (lon, lat) when the manifest carried a georeference — the
  dialog says *"Opens directly in QGIS and ArcGIS Pro."* Boxes and masks are mapped from
  the 0–1000 view frame through the scene's bounds.
- **Pixel space fallback:** with no CRS the dialog says so in amber and the file carries
  a `satquery:crs = "pixel"` note; coordinates are the 0–1000 image frame. Honest, but not
  a map layer.
- **Box layers:** one Feature per bounding box, properties `label`, `score`, `view`,
  `produced_by_step`.
- **Provenance:** `trace_id`, generator, schema version, step list — so the file is
  auditable on its own.

**Include masks (the toggle):** vectorises the mask raster **on this device**. The
implementation (`frontend/src/export/geojson/masks.ts`) traces pixel boundaries — every
inside pixel contributes the sides it does not share with another inside pixel, sides
are chained into closed rings, holes are wound correctly, and each ring is simplified
with Douglas–Peucker at **half a pixel**. The output is one `MultiPolygon` per class with
the class name in the attribute table. (It is a marching-squares-equivalent written from
scratch because the published package is AGPL; large rasters are resampled nearest to
`maxSide` first.) Leave it on for GIS hand-off; turn it off for a quick box-only file.

**Loading in QGIS**

1. *Layer → Add Layer → Add Vector Layer*, pick `satquery-<trace8>.geojson`.
2. QGIS reads the CRS as EPSG:4326 automatically. Drop your source GeoTIFF alongside — the
   boxes and mask polygons should sit exactly on the features they describe.
3. Open the attribute table (`F6`) to see `class`, `produced_by_step`, `source_artifact`.
4. Style by `class` (*Symbology → Categorized*) to colour changed / built-up / water.

For a saved run (Saved page), the exporter re-fetches the trace; if the server has pruned
it, the file is still written but marked *"Exported without tool provenance"*.

### 3.6 History & session management

- **Thread → History tab / Sidebar → History panel:** every run this browser session
  submitted, each rehydrated live from `GET /v1/traces/{id}`. The backend has no
  "list traces" route on purpose — the panel shows what *this client has seen*, and each
  entry is the real persisted audit trace, not a cached copy. A 404 means the server
  pruned it; a transport error means the server is gone (and offers *Retry*).
- **Previous queries** (thread): quick re-ask of past questions on the current scene.
- **Saved (`G` `S`):** runs you starred, laid out as a library (layout in Settings).
  `/report/<trace>` renders a shareable read-only page of one run — this is what the
  terminal `done` payload equals.
- **Projects (`G` `P`):** client-side grouping of saved runs. No server accounts exist;
  the profile card is one local identity.
- **Storage:** the **Storage quota** meter in the sidebar shows local usage.
  **Settings → Session history / Saved runs & projects / Everything** clear the
  corresponding local state. The trace database on the server (`data/traces.sqlite3`)
  is untouched by any of these.
- **Notifications** (bell): warnings such as `IMAGERY_BAND_MISSING`, transport drops,
  and export completions.

---

## 4. Keyboard shortcuts & pro tips

The single source of truth is `frontend/src/shell/shortcuts.ts`; the in-app guide
(`⌘/` or `?`) renders the same table and has a filter box. On Windows/Linux `⌘` is
`Ctrl`.

**Chords (always active, never gated)**

| Keys | Action | Scope |
|---|---|---|
| `⌘K` | New query (opens the console, focuses the composer) | global |
| `⌘/` | Keyboard shortcuts guide | global |
| `⌘J` | Toggle dark / light theme | global |
| `⌘,` | Settings | global |
| `⌘⇧S` | SITREP — one-page PDF brief | thread |
| `⌘⇧G` | Export GeoJSON | stage |

**Single keys** (can be turned off in *Settings → Single-key shortcuts* for speech input
or screen readers — WCAG 2.1.4)

| Keys | Action | Scope |
|---|---|---|
| `?` | Keyboard shortcuts guide | global |
| `/` | Focus the composer (Ask) | thread |
| `←` / `→` | Previous / next evidence view | stage |
| `[` / `]` | Swipe slider left / right by 5 % | stage |
| `P` | Toggle the processing pipeline (DAG) | stage |
| `B` | Toggle globe / flat projection | maps |

**Go-to sequences** (press `G`, then the letter within a second)

| Keys | Destination |
|---|---|
| `G` `H` | Landing |
| `G` `E` | New Query (the console) |
| `G` `U` | Use cases |
| `G` `M` | Maps |
| `G` `S` | Saved |
| `G` `P` | Projects |

**Pro tips**

- **Prime the demo:** open `?mock=1` in one tab and the live console in another. If the
  GPU stalls mid-pitch, `⌘Tab` to the rehearsal tab — same UI, same flow.
- **Show concurrency deliberately:** run the cross-modal scenario and open the pipeline
  (`P`) *before* pressing Enter. The two analysers lighting up together is the one frame
  that proves the executor walks a DAG.
- **Click a number, not a sentence.** The most convincing 10 seconds is: hover a
  citation, watch the card highlight, click, read the same value in the trace.
- **Run it twice.** Same inputs, same question: identical plan, identical scalars,
  identical answer text. Only timestamps differ.
- **Dark theme for imagery.** The default keeps the raster the brightest thing on
  screen; `⌘J` toggles if the projector washes out.
- **Reduced motion** (Settings) disables the DAG pulse and swipe easing for a
  screen-recorded demo.
- **Windows keys:** `Ctrl+Shift+S` for SITREP, `Ctrl+Shift+G` for GeoJSON — the buttons
  carry `aria-keyshortcuts` for both.

---

## 5. Troubleshooting & common pitfalls

### 5.1 Non-georeferenced images (pixel-space warnings)

**Symptom:** the manifest shows `CRS —`, the reader warns *"The image is not
georeferenced; geometric checks were skipped."*, the Maps stage falls back to
**PixelStage** (pan/zoom, cursor reads pixel x/y), and the GeoJSON dialog says
*"Pixel space — not georeferenced."*

**Cause:** a PNG/JPEG, or a TIFF whose transform is GDAL's identity placeholder
(`0,1,0,0,0,1`) or that lacks a CRS. `src/satquery/ingest/reader.py` sets
`is_georeferenced=False` and `crs_match`, `bounds_overlap_iou`, `gsd_ratio`,
`coregistration_offset_px` report `SKIP`.

**What still works:** VQA, captioning, grounding boxes, counting, segmentation on a
single image. **What does not:** anything needing two scenes aligned in the world —
change detection and cross-modal pairs need a georeference on both.

**Fix:** export from Copernicus Browser's *Analytical* tab (§2 Method B), or fetch from
the Maps page. If you only have a PNG of a known area, `gdal_translate -a_srs EPSG:4326
-a_ullr W N E S in.png out.tif` attaches a georeference — but the co-registration check
will still measure real misalignment, honestly.

### 5.2 VRAM allocation conflicts

| Symptom | Cause | Fix |
|---|---|---|
| HF backend refuses to load, `OutOfMemory` at startup on the 24 GB box | `llama-server` is still running (holds ~9 GB); the bf16 load needs ~18.7 GiB peak with six views | Stop llama-server. **Never run both backends on one 24 GB card.** |
| llama-server dies on model load on the 8 GB laptop, `nvidia-smi` > 7.4 GB | All 36 layers offloaded plus projector plus KV cache exceeds the usable ~7.2 GB | `SATQUERY_VLM_GPU_LAYERS=28` then restart `serve_vlm.ps1` — ~8 layers on CPU, half speed, but it answers. |
| Answers truncate or the server 500s on the laptop | Six 448 px views ≈ 6 × 1024 image tokens overflow the 8192 context | `SATQUERY_VLM_MAX_VIEWS=3` (already in `.env.laptop`). The trace records `views_ceiling: 3` and why. |
| Generation is garbage on ROCm gfx1100 with NF4 | NF4 generation is broken on that GPU | Serve bf16 base + adapter (`SATQUERY_VLM_BACKEND=hf`), or the Q4_K_M GGUF. Never NF4 for inference. |
| `image_diff_change` appears instead of `siamese_change_detector` | `SATQUERY_CD_CHECKPOINT` unset or wrong path; the registry only counts configured checkpoints | Set it in `.env` to the Siamese checkpoint under `data/checkpoints/cd/`; HealthStrip should read `tools 6/6`. |
| First request takes > 60 s | Cold weight load / kernel compile | Warm it: run `make e2e` (or one `?mock=1`-free query) before the judges sit down. |
| Torch disappeared after a `uv` command on the ROCm box | A bare `uv sync` / `uv run` re-synced the lockfile's CUDA torch | Reinstall per `DOCS/AI_HANDOFF/05_ENVIRONMENT_AND_SETUP.md`; from then on `uv run --no-sync`. |

Check the actual memory with `rocm-smi --showmeminfo vram` (ROCm) or `nvidia-smi`
(CUDA) while a request is in flight, not idle.

### 5.3 Full offline demonstration with `?mock=1`

Open `http://localhost:5173/?mock=1` (or the built `dist/` served by
`scripts/demo_laptop/serve_frontend.py`). Mock Service Worker intercepts every
`/v1/*` call and answers from recorded fixtures:

- `frontend/src/mocks/captured/health.json`, `registry.json`, `validate.single.json`,
  `validate.bitemporal.json`, `events.bitemporal.json` — real responses captured from a
  live server, replayed with realistic timing.
- `frontend/public/samples/stac/` — a recorded Planetary Computer search so the Maps
  page shelf, T1/T2 selection and fetch all work with the network switched off.
- Scenarios in `frontend/src/mocks/scenarios.ts` choose the flow from what you drop
  (one file → single; two → bi-temporal) and the question you ask.

Everything client-side is real in this mode: the DAG animation, citation hover/click,
SITREP PDF, GeoJSON with mask vectorisation, history, saved runs. What is *not* real: the
answer text and the scalars are fixtures, and the HealthStrip shows a **mock** badge so
nobody mistakes it for inference. `make frontend-e2e` runs Playwright against exactly this
mode with the browser offline; `DOCS/FINAL_QA_CHECKLIST.md` §1 lists what must pass.

### 5.4 Other pitfalls

- **Maps page says "Imagery search is off":** enable *Settings → Imagery search on the
  Maps page*. It is off by default so an air-gapped hall never waits on DNS.
- **Place search returns nothing:** type ≥ 3 characters; Nominatim is rate-limited to
  one request per second and refuses generic `User-Agent`s — the app sets a `Referer`.
- **Fetch bar refuses a SAR pair:** different relative orbits. Pick two scenes from the
  same orbit track; otherwise `coregistration_offset_px` will fail, correctly but
  confusingly.
- **`bounds_overlap_iou` FAIL on files you know overlap:** one is in a different UTM zone
  and the reprojection action was declined, or one is not georeferenced. Check the
  manifest's CRS line for both.
- **`band_sufficiency` WARN on a 3-band S2 export:** you exported *True color* instead of
  raw bands. NDBI/NDWI need B11/B12/B08. Re-export with the band list in §2 Method B.
- **Uncited spans on every answer with `demo-cpu`:** expected — templated answers have no
  VLM to bind prose to; the KPI cards and citations from tools are still real.
- **`make contract` fails after editing the API:** regenerate `openapi.json` with
  `scripts/export_openapi.py` and `schema.d.ts` with the frontend script; the contract
  is additive-only at 1.0.
- **Stream stops, HealthStrip red:** the API died. Partial artifacts stay visible; restart
  the API and the strip recovers without a reload. The job can be re-submitted.

---

## Appendix: quick reference

**Environment variables** (`.env`, read by pydantic-settings)

| Variable | Purpose |
|---|---|
| `SATQUERY_VLM_BACKEND` | `hf` (bf16 + adapter), `llamacpp` (GGUF via llama-server), `none` (templated) |
| `SATQUERY_VLM_ADAPTER_PATH` | `runs/sq-lora-v2-full/adapter` |
| `SATQUERY_VLM_SERVER_URL` | `http://127.0.0.1:8080` for llamacpp |
| `SATQUERY_VLM_MAX_VIEWS` | 6 on 24 GB, 3 on 8 GB |
| `SATQUERY_VLM_GPU_LAYERS` | llama-server layer offload (laptop fallback: 28) |
| `SATQUERY_SEG_CHECKPOINT` / `SATQUERY_CD_CHECKPOINT` | SegFormer and Siamese checkpoints; both required for `tools 6/6` |
| `SATQUERY_TRACE_DB` | SQLite path for audit traces (`data/traces.sqlite3`) |
| `SATQUERY_ARTIFACT_ROOT` | Rendered views, masks, clipped imagery (`data/artifacts`) |
| `SATQUERY_IMAGERY_MAX_PX` | Long-side cap for Maps-page clips |

**Make targets**

| Target | Does |
|---|---|
| `make demo` / `make demo-cpu` | API :8000 + frontend :5173, with / without the VLM |
| `make e2e` / `make e2e-cpu` | Start API from `.env`, run `scripts/e2e_parity.py` (5 scenarios), stop |
| `make frontend-e2e` | Playwright against `?mock=1`, browser offline |
| `make ci` | ruff, mypy --strict, pytest, oxlint, tsc, vitest, vite build, contract chain |
| `make contract` | `openapi.json` ↔ `schema.d.ts` in sync |

**API endpoints** (prefix `/v1`)

| Route | Use |
|---|---|
| `POST /validate` | Pre-flight: manifests + 11 checks, no analysis |
| `POST /analyze` | Synchronous full run |
| `POST /jobs` → `GET /jobs/{id}/events` | Streaming run (SSE); `GET /jobs/{id}` snapshot |
| `POST /imagery/fetch` → `GET /imagery/{fetch_id}/{name}` | Clip Planetary Computer scenes to a bbox |
| `GET /artifacts/{id}` | Raw bytes of a rendered view or mask |
| `GET /traces/{id}` | The persisted `AuditTrace` |
| `GET /registry`, `GET /health` | Tool availability, backend readiness |

**Where to read next:** `DOCS/AI_HANDOFF/00_START_HERE.md` (orientation),
`DOCS/API_CONTRACT.md` (wire contract), `DOCS/AGENT_POLICY_DAG.md` (why each tool is
planned), `DOCS/FINAL_QA_CHECKLIST.md` (the demo-day checklist),
`scripts/demo_laptop/README.md` (the air-gapped runbook).
