# SatQuery AI — Final Manual QA Checklist (SIH 2026 Final)

*Manual runbook for the last pass before the presentation. Written 2026-09-16
against `main` (Tracks 1–4 complete, entry bundle 173.7 KB / 180 KB budget).
Every step names the file or fixture it exercises so a failure can be traced
to code, not to memory. Nothing here changes source; this is a reading of it.*

**How to use this document.** Work top to bottom. Each item is a `[ ]` box with
an **Expect** line — the pass criterion — and, where it matters, a
**If it fails** pointer. Record the result in the *Sign-off sheet* at the
end. A ❗ marks a step that would embarrass us on stage if skipped.

Conventions:

| Symbol | Meaning |
|---|---|
| **A / B / C** | network tier (see §3): A = internet + local API, B = local API only, C = nothing, `?mock=1` |
| ⌘ | `Ctrl` on Windows / Linux, `Cmd` on macOS |
| `:5173` / `:8000` / `:8080` | frontend · FastAPI · llama-server |

---

## 0. Pre-flight (do once per machine, before any section below)

### 0.1 Automated gates are green — do not start manual QA on a red tree

- [ ] `make ci` on the 24 GB box passes (backend lint/typecheck/test/contract + frontend lint/typecheck/test/contract/build).
  **Expect:** exit 0. ❗
- [ ] `cd frontend && npm run build && npm run check:bundle`.
  **Expect:** entry ≤ 180 KB (last known 173.7 KB); neither `sitrep-*.js`, `geojson-masks-*.js`/`marching-squares`, nor the maplibre chunk appears in the entry's `imports`. Reference: `frontend/src/__tests__/bundle-boundaries.test.ts`.
- [ ] `make frontend-e2e` (Playwright, `?mock=1`, browser offline) passes all three specs in `frontend/e2e/track4.spec.ts` plus `overlay.spec.ts`.
  **Expect:** 4/4 green. This is the automated twin of §1 and §3.4; if it is red, the manual version will be too.
- [ ] `make contract` green — `openapi.json` ↔ `frontend/src/api/schema.d.ts` unchanged.

### 0.2 Environment sanity

- [ ] `.env` on the GPU box reads `SATQUERY_VLM_BACKEND=hf`, `SATQUERY_VLM_ADAPTER_PATH=runs/sq-lora-v2-full/adapter`, and both `SATQUERY_SEG_CHECKPOINT` / `SATQUERY_CD_CHECKPOINT` point at files that exist.
  **If it fails:** with `SATQUERY_CD_CHECKPOINT` unset the planner silently substitutes `image_diff_change` for `siamese_change_detector` and §4.2 will fail on the tool *name*.
- [ ] `llama-server` is **not** running on the 24 GB box while the HF backend loads (it holds ~9 GB; the bf16 load is refused).
- [ ] Browser: a clean profile or cleared site data for `localhost:5173` so remembered settings (`satquery.settings`, `satquery.maps.basemap`, `satquery.maps.projection`) do not mask a default-state bug. Note the defaults you expect: `onlineFeatures=false`, basemap `none`, projection `mercator`.
- [ ] Have QGIS 3.34 (or later) installed on the Windows laptop for §1.2.
- [ ] Have an A4 **and** a Letter printer preset (or a PDF viewer that shows page boundaries for both) for §1.1.

---

## 1. Track 4 UI features

Run this section first in **tier A** on the GPU box (real backend, `make demo`),
then repeat the ❗ rows in **tier C** (`?mock=1`) on the laptop. Both must
pass; Track 4's design rule is "every feature works under `?mock=1`".

### 1.1 Tactical SITREP (client-side one-page PDF)

Code under test: `frontend/src/export/sitrep/{useSitrep,compose,scene,render,build}.ts`,
`export/download.ts`. pdf-lib is lazy-loaded; fonts embed from `/fonts/*.woff2`.

**Entry points — all three must produce the same file**

- [ ] Run the bi-temporal change query (drop the two CDVQA/Sentinel-2 GeoTIFFs, "What changed between these two images?"). Wait for the answer card.
- [ ] Response card action row → the fourth action **SITREP** (tooltip `One-page brief as PDF (⌘⇧S)`).
  **Expect:** button state runs `composing → rendering → done` (no frozen UI), then a download of `SITREP-<sceneId>-<yyyymmdd-hhmm>.pdf` and the toast `SITREP saved.` `sceneId` is the manifest scene id, else the first 8 chars of the trace id (`compose.ts:78`).
- [ ] Press **⌘⇧S** with the thread focused.
  **Expect:** identical filename pattern, new timestamp. With **no run loaded**, ⌘⇧S shows the info toast `Nothing to brief yet — run a query first.` and downloads nothing (`shell/useHotkeys.ts:89`).
- [ ] Saved page → any saved run → **Export menu → SITREP**.
  **Expect:** same PDF; this path fetches `GET /v1/traces/{id}` for the tool chain.
- [ ] **Saved-run degrade:** restart the API (traces are in-memory) or pick a saved run whose trace no longer exists, then Export → SITREP.
  **Expect:** PDF still renders from the `SavedRun` summary; TOOL CHAIN line reads *not available — trace no longer on the server*; a toast explains; no console error. ❗

**Content correctness — open the PDF**

- [ ] Header band: `SATQUERY AI · SITREP`, scene id, generated-at timestamp, `Query: "…"` verbatim, `Status: OK | DEGRADED | FAILED` matching the console's status pill.
- [ ] Scene panel (left, 92 mm): for a pair, **T1 | T2** side by side; for cross-modal, **optical | SAR**; labels correct; boxes burned in at the same positions as `BboxOverlay` on screen; north arrow; scale bar with a plausible length for the scene's GSD; bounds strip in DMS. For a non-georeferenced input (benchmark PNG) the bounds strip and scale bar are **absent**, not zero.
- [ ] ANSWER (right, 88 mm): answer text with citation superscripts ¹ ² ³ numbered in first-appearance order, matching the numbers the thread shows on hover.
- [ ] KEY MEASUREMENTS: the rows are **exactly** the rows the on-screen KpiCards show, same values, same units, tabular figures (Geist Mono aligned on the decimal). The PDF must never show a number the UI did not (`selectKpis` is the single source; `compose.test.ts` asserts it — you are confirming visually).
- [ ] TOOL CHAIN line: `parse → checks N/11 → <tool>@<version> → …` in plan order, matching the DAG panel; wraps at most once.
- [ ] CITATIONS line: `N bound · 0 uncited`, `trace <id> · v1.0`. Trace id matches `/report/<trace>`.
- [ ] **Uncited badge:** open `/?mock=1&scenario=ungrounded`, run, click SITREP.
  **Expect:** CITATIONS reads `4 bound · 2 uncited` and a warn-tone **`2 UNCITED`** badge is visible (`render.ts:361`); the generator string in the console shows `(SYNTHETIC FIXTURE)`. This scenario is synthetic and labelled — never demo it as a real run. ❗
- [ ] **Overflow policy:** run a query with a long answer (or the `ungrounded` scenario, which is the longest fixture).
  **Expect:** answer clipped with an ellipsis and a footnote `see full report at /report/<trace>`; the PDF is **always exactly 1 page** (check the viewer's page count).
- [ ] Fonts: text is Geist / Geist Mono, not Helvetica fallback — zoom to 400 % on a superscript and on a `%` sign.

**Print**

- [ ] Print (or print-preview) on **A4** and on **US Letter**.
  **Expect:** nothing clipped at 18 mm margins on either; the scene panel and the right column both fit; footer line fully visible. ❗

**Non-functional**

- [ ] Network tab during SITREP in tier B (Wi-Fi off): only same-origin requests (`/fonts/*.woff2`, `/v1/artifacts/...` already cached). No third-party host.
- [ ] First SITREP click loads a new chunk (pdf-lib + fontkit); second click does not re-download it.
- [ ] Time from click to download on the laptop < 3 s for a pair scene.

### 1.2 GeoJSON exporter (RFC 7946, optional `marching-squares` masks)

Code under test: `frontend/src/export/geojson/{build,winding,masks,plan,provenance,useGeoJsonExport}.ts`,
`components/export/GeoJsonExportDialog.tsx`, `components/stage/EvidenceTray.tsx`.

**Entry points**

- [ ] After a georeferenced run: Evidence tray button reads **`Export GeoJSON`**.
- [ ] After a **non-georeferenced** run (the benchmark PNG grounding query): button reads **`Export GeoJSON (pixel space)`**. The label is decided solely by `bounds !== null`. ❗ A judge must never load a pixel-space file thinking it is georeferenced.
- [ ] **⌘⇧G** with a run loaded exports; with no run shows `Nothing to export yet — run a query first.`
- [ ] Saved → Export menu → **GeoJSON** works; on a run whose trace is gone, the file still downloads with provenance columns `null` and the toast `Exported without tool provenance — the server no longer has this trace.`

**Dialog behaviour**

- [ ] Run with **no class raster** (grounding / change-only): clicking Export downloads immediately — **no dialog**.
- [ ] Run **with** a `semantic_segmenter` class raster (a SEGMENTATION or "identify water bodies" query): a Radix dialog opens with checkbox **Include segmentation masks**, a CRS line reading `WGS84 (CRS84)` or `Pixel space — not georeferenced`, and an **Export** button. Esc closes it without exporting.
- [ ] Toasts: `GeoJSON exported in WGS84.` / `GeoJSON exported in pixel space (no georeference).`
- [ ] Filename: `satquery-<first 8 of trace id>.geojson`, MIME `application/geo+json`.

**File correctness — open the file in a text editor**

- [ ] Top-level `type: "FeatureCollection"`, a `bbox` member `[w, s, e, n]` covering all features, **no `crs` member anywhere** (RFC 7946 §4 forbids it). ❗
- [ ] Each box → one `Polygon` feature, ring **closed** (first == last coordinate), 5 positions, exterior ring **counter-clockwise** (right-hand rule), coordinates rounded to **7 decimals**, longitude before latitude.
- [ ] Per-feature `properties`: `label`, `confidence`, `view_id`, `source_step`, `tool`, `tool_version`, `citation` (`step:n/scalars.<path>` or `null`), `trace_id`, `scene_id`, `acquired_at`, `sensor`.
- [ ] **Pair run:** two distinct `view_id` values (T1 / T2), one feature per box per view.
- [ ] **Cross-modal run:** `view_id` values `optical` / `SAR`.
- [ ] **Pixel space:** a top-level `note` explaining the frame, and per-feature `crs: "pixel"`, `frame: 1000`; coordinates in `[0, 1000]`; still no `crs` member on the collection.
- [ ] **Masks on:** additional `MultiPolygon` features, one per class, with a class `label`; rings closed and CCW; visually simplified (0.5 px Douglas–Peucker — vertex count is well below the pixel perimeter). Export **with** masks is larger than **without**; both parse.
- [ ] Paste the file into a strict validator (e.g. geojsonlint or `npx geojson-validation`) — zero errors.

**GIS round-trip on the Windows laptop** ❗

- [ ] QGIS 3.34 → *Layer → Add Layer → Add Vector Layer* → the WGS84 file.
  **Expect:** layer CRS auto-detected as EPSG:4326 (no CRS prompt); boxes land on the correct place over an OSM/XYZ basemap (tier A) or over the fetched GeoTIFF from §1.3 (tier B).
- [ ] Right-click → *Open Attribute Table*: the `citation`, `tool`, `tool_version`, `view_id` columns are populated; filter `view_id = 'T2'` hides the T1 layer.
- [ ] Load the masks file: MultiPolygons align with the class raster edges when overlaid on the scene GeoTIFF.
- [ ] Load the **pixel-space** file: QGIS shows it in an unknown/flat frame with a 0–1000 extent — it does **not** appear anywhere on the globe. This is correct.
- [ ] (If available) ArcGIS Pro → *Add Data*: same file opens; attribute table shows the same columns.

### 1.3 STAC / MapLibre discovery HUD

Code under test: `frontend/src/pages/Maps.tsx`, `pages/maps/MapStage.tsx`,
`pages/maps/hud/{Discover,PlaceSearch,TimeSlider,CollectionPicker,SceneShelf,FetchBar,ZoomHud,useImageryFetch}.tsx`,
`geo/{nominatim,stac,planetary,collections,deadline}.ts`, `state/{stac,map,settings}.ts`,
backend `src/satquery/api/routers/imagery.py` + `ingest/stac_fetch.py`.

**Online gate (tier A, clean profile)**

- [ ] Open `/maps` with `onlineFeatures` off (default).
  **Expect:** the Discover HUD renders exactly one line — `Imagery search is off. Turn on online features in Settings to search Sentinel scenes.` — and an **[Open settings]** button. Network tab shows **no** request to `nominatim.openstreetmap.org` or `planetarycomputer.microsoft.com`, not even a preflight. ❗
- [ ] [Open settings] lands on Settings → **Data controls** → **Online features** toggle. Turn it on; close; the HUD is live. Reload: the setting is remembered (`satquery.settings`).
- [ ] Empty-state Maps page (no run loaded) shows a **Find imagery** button that opens Discover; the world view renders with `georef = null`.

**Place search (`PlaceSearch.tsx`, `geo/nominatim.ts`)**

- [ ] Type `Ah` — nothing happens (min 3 chars). Type `Ahm` — one request after ~400 ms debounce, not one per keystroke.
- [ ] Results list: ↑ / ↓ move, ⏎ selects, Esc closes; the selected place writes `place` + `bbox` to the store and the map **flies** to it once (not on every re-render).
- [ ] Type quickly through several queries: at most 1 request/s reaches Nominatim (rate-limit rule); stale responses do not overwrite a newer query's list.
- [ ] Unknown place (`zzqx`) → an empty list, no error, input stays editable.
- [ ] Nominatim 429/5xx (simulate by DevTools request blocking with a 429 override, or throttle to Offline) → inline `Place search is rate-limited — try again in a moment.` (429) or `Search needs a network — the loaded scene is still here.` **[Retry]** (offline). No toast.

**Sensor selection (`CollectionPicker.tsx`, `geo/collections.ts`)**

- [ ] Segmented control: **Sentinel-2 · Sentinel-1 · Both**; exactly one active; `aria-pressed`/`aria-checked` correct; keyboard arrows move the selection.
- [ ] Sentinel-2 → request body `collections: ["sentinel-2-l2a"]` with a cloud filter (`eo:cloud_cover < 20` default; the cloud slider changes the filter value).
- [ ] Sentinel-1 → `sentinel-1-rtc` preferred, `sentinel-1-grd` fallback; cards show **orbit state** (ascending/descending) and polarisation instead of cloud %.
- [ ] Both → two collections in one request; shelf mixes cards with a sensor badge each.

**Time slider (`TimeSlider.tsx`)**

- [ ] Single mode: one handle; label shows the window as ISO dates; preset **last 30 d** sets `datetime` accordingly (verify in the request body).
- [ ] Pair mode (after picking "Use as T1"): two handles; preset **6 mo apart** places them ~183 days apart; handles cannot cross; each move re-queries once after debounce, not continuously while dragging.
- [ ] Keyboard: focus a handle, arrow keys step by one day, PgUp/PgDn by a month; the value is announced (aria-valuetext).
- [ ] Narrow window with no scenes → shelf reads `No scenes in this window. Widen the dates or raise the cloud limit.`

**Scene shelf and footprints (`SceneShelf.tsx`, `MapStage.tsx`)**

- [ ] Results appear as **footprint polygons** (`stac-footprints-fill/-line`) on the map and as cards in the shelf; count in the header matches the number of footprints.
- [ ] Hover a card → its footprint highlights; hover a footprint → its card highlights (two-way `hover`).
- [ ] Click a footprint → card scrolls into view and becomes selected (`-selected` layer style).
- [ ] Thumbnails load (SAS-signed for PC assets; fallback to `rendered_preview` if the token call fails — verify by blocking `/api/sas/v1/token/*`: thumbs still show).
- [ ] Shelf virtualises: with 24 results, scrolling is smooth and the DOM does not contain all 24 cards at once.
- [ ] **Use as T1 / Use as T2** buttons: T1 then T2 enables pair mode; picking two S1 scenes on **different orbits** shows `These two scenes are on different orbits; the co-registration check will fail. Pick a same-orbit pair.` and the Analyse-change button is disabled or warns. ❗
- [ ] Reload the page: the shelf is **empty** (`state/stac.ts` is session-only, deliberately not persisted — yesterday's rehearsal must not appear on stage).

**Fetch → console (`FetchBar.tsx`, `useImageryFetch.ts`, `POST /v1/imagery/fetch`)**

- [ ] Single scene → button reads **Analyse**; pair → **Analyse change**.
- [ ] Click: progress line `Fetching 1 of 2 — clipping to the pipeline's window…`; the backend returns `202 { upload_ids, scene_ids, crs, gsd_m }`; the console opens with the fetched views loaded and the compatibility panel green (**11/11** for a same-orbit S1 pair). ❗
- [ ] Fetched GeoTIFFs (retrieve via the artifact URL or from the ArtifactStore dir) open in QGIS with the correct UTM CRS and a ≤ 2048 px window — not a full 1 GB tile.
- [ ] S2 fetch carries bands B02/B03/B04/B08/B11/B12 (what `spectral_index_analyzer` needs); S1 carries `vv`, `vh`; manifest records `sat:orbit_state`.
- [ ] Error paths (tier A, then pull the cable mid-flow): PC unreachable → FetchBar shows the API's `502` hint `Planetary Computer unreachable`, **selection preserved**, Retry works. Local API down → the composer's transport toast (`Start the API…`) and the HealthStrip goes red.
- [ ] Oversized window (draw/choose a bbox far larger than 2048 px at 10 m) → `413` shown inline, no crash.

**Projection / globe (`ZoomHud.tsx`, `state/map.ts`)**

- [ ] Globe toggle in the zoom HUD has `aria-pressed`; toggling switches `map.setProjection` between mercator and globe with the scene image, footprint line, and STAC footprints still drawn on the sphere.
- [ ] Choice persists across reload (`satquery.maps.projection`).
- [ ] Works with **no run loaded** (Playwright covers this; confirm visually the sphere is not a black ball — the `land-110m.json` / `land.png` overlay is drawn).

**Basemap**

- [ ] Satellite basemap (Esri World Imagery) is **off by default**; turning it on loads tiles in tier A and is remembered.
- [ ] Block `server.arcgisonline.com` in DevTools, pan: basemap flips to `none` (not remembered), toast `Basemap unreachable — showing scene only.` with **Retry**; the scene image stays. ❗

---

## 2. Live DAG parity & SSE streaming (tier A, GPU box)

Code under test: `src/satquery/api/routers/jobs.py`, `frontend/src/api/events.ts`,
`state/job.ts`, `components/pipeline/{DagCanvas,PipelineDialog}.tsx`, `scripts/e2e_parity.py`.

### 2.1 Scripted parity first

- [ ] `make e2e` (starts the API from `.env`, runs `scripts/e2e_parity.py`, stops it).
  **Expect:** **5/5** — `grounding`, `change`, `crossmodal` positive; `disjoint`, `uncited` negative — and for every positive scenario the SSE trace is byte-identical (timestamps and `trace_id` dropped) to the synchronous `POST /v1/analyze` trace. Total wall time < 3 min after model load. ❗
- [ ] Confirm the script's concurrency assertion passed for `change` **and** `crossmodal` (`expect_concurrency=True`): overlapping `step_started`/`step_completed` frames for the independent branches.
- [ ] `scripts/preflight_train_serve_parity.py` passes with the adapter path from `.env` (view budget 448 px / `max_pixels=147456`).
- [ ] `Answer.generator` in any response reads `…+adapter:sq-lora-v2-full` (built from the adapter directory name).

### 2.2 Concurrent execution through the UI — cross-modal optical + SAR

- [ ] Drop `s2_pre.tif` + `s1_vvvh.tif` (from `tests/fixtures/e2e/` or `data/e2e/`), query *"Is this area built-up? Use both sensors together."*, submit.
- [ ] Preflight panel: 11 checks listed, all pass (cross-modal pair type); manifest shows two sensors.
- [ ] **`plan` event** arrives before any step: the DAG canvas draws six nodes at once — `spectral_renderer → {spectral_index_analyzer ∥ sar_backscatter_analyzer} → crossmodal_consistency → physics_agreement → vlm_vqa` — with the two analysers drawn as **parallel siblings**, not a chain. ❗
- [ ] Watch the nodes: `spectral_index_analyzer` and `sar_backscatter_analyzer` are both in the *running* state **at the same time** for at least one visible frame; each finishes independently (one may finish first — the other must stay running, not flicker to done).
- [ ] Open DevTools → Network → the `GET /v1/jobs/{id}/events` stream → EventStream tab. Verify the order grammar from `frontend/src/api/events.ts`:
  `queued → stage(ingesting…planning) → plan → interleaved step_started / step_completed / artifact → answer_delta* → done`. `id:` fields increase monotonically; `event:` names are exactly those strings; no unknown event names.
- [ ] Between the two analysers' `step_started` frames and the first `step_completed`, `artifact` frames may interleave — the evidence tray shows NDBI and SAR VV/VH views appearing **as they arrive**, before the answer.
- [ ] Answer card: NDBI mean **and** σ⁰ VV are both cited; hovering each citation highlights the matching measurement; clicking lands on the card in the evidence tray. `physics_agreement.verdict` is present in the fact sheet.
- [ ] Terminal `done` payload equals what `/report/<trace>` renders (query, answer, citations, steps).
- [ ] Repeat the identical query. **Expect:** identical plan, identical scalars, identical answer text (deterministic control flow; only timestamps differ).

### 2.3 Concurrent execution — bi-temporal change

- [ ] Drop `s2_pre.tif` + `s2_post.tif`, "What changed between these two images?".
- [ ] DAG shows `spectral_renderer → {siamese_change_detector → change_statistics} ∥ spectral_index_analyzer → vlm_change_vqa`; `spectral_index_analyzer` runs concurrently with the Siamese branch (it only needs the renderer).
- [ ] The tool is **`siamese_change_detector`**, not `image_diff_change` (the latter means the CD checkpoint was not picked up — see §0.2).
- [ ] `CHANGE_MASK` artifact appears; `changed_area_pct` is cited as `step:N/scalars.changed_area_pct`; the answer text contains the cited value verbatim; clicking it lands on the `change_statistics` measurement card. ❗

### 2.4 Streaming robustness

- [ ] Submit, then reload the page mid-run: the job store recovers via `GET /v1/jobs/{id}` (snapshot) and the DAG shows the current state; no duplicate steps.
- [ ] Submit two jobs back-to-back from two tabs: each tab's stream shows only its own `job_id`; both complete.
- [ ] Kill the API mid-stream (Ctrl-C `make demo` backend): the console shows an `error`/transport state within seconds, the HealthStrip turns red, partial artifacts already received stay visible, and nothing spins forever. Restart the API → HealthStrip recovers without a reload.
- [ ] Negative — mismatched pair (`s2_pre.tif` + `s2_disjoint.tif`): status **FAIL/DEGRADED**, the failing check named in the compatibility panel, HTTP 200-family, **never a 500**; the SSE stream terminates with a proper terminal event. ❗
- [ ] Negative — `uncited` scenario ("Describe this scene and identify water bodies." with a tool disabled per the script): degrades to PASS with the missing tool flagged; the honesty flag renders if any numeric span is uncited.
- [ ] Check the API log for the run: no stack traces, no `WARNING` about unknown SSE events, one `plan` per job.

---

## 3. Offline resilience — the 3-tier strategy

Definition (`DOCS/UI_TRACK_4_ARCHITECTURE.md §4`):

| Tier | Reachable | Must work |
|---|---|---|
| **A** | internet + local API | everything |
| **B** | local API only | console, scene map, globe, SITREP, GeoJSON; discovery degrades gracefully |
| **C** | nothing → `?mock=1` | all three Track 4 features from fixtures via MSW |

Invariants to hold in every row below: (1) every online call is behind
`onlineFeatures`; (2) every failure surfaces within **5 s** (`geo/deadline.ts`);
(3) failing never removes anything already on screen.

### 3.1 Tier B — pull the network, keep the API

Set up in tier A: a run loaded, Discover open with results on the shelf, satellite basemap on, globe on. Then disable Wi-Fi / unplug.

- [ ] Nothing already on screen disappears: answer card, evidence tray, DAG, the map's scene image, the STAC footprints and cards all remain. ❗
- [ ] Pan the map: basemap auto-falls back to `none` with the `Basemap unreachable — showing scene only.` toast + Retry; the `none` choice is **not** written to localStorage (reconnect + reload → satellite comes back on its own).
- [ ] Globe toggle still works (client-side geometry).
- [ ] Type a new place: within 5 s the list is replaced by `Search needs a network — the loaded scene is still here.` **[Retry]**; input stays editable; no toast.
- [ ] Move the time slider (new STAC search): `status: 'offline'`; **existing results stay clickable**, footprints stay on the map, shelf header shows an offline pill.
- [ ] Click **Analyse** on a selected scene: `POST /v1/imagery/fetch` reaches the local API, which returns the `502` envelope with hint `Planetary Computer unreachable`; FetchBar shows the hint; selection preserved.
- [ ] SITREP and GeoJSON export work exactly as in §1 (zero third-party requests — verify in the Network tab). ❗
- [ ] Submit a new query on already-uploaded files: the whole DAG runs (local GPU); HealthStrip stays green.
- [ ] Reconnect: the `online` event flips `offline → idle` and the last discovery query re-runs **once** automatically; a second failure needs a manual Retry.

### 3.2 Tier C — nothing reachable, `?mock=1` (MSW intercepts)

Set up: Wi-Fi **off**, API **stopped**, frontend served from `dist` (`scripts/demo_laptop/serve_frontend.py` or `npm run preview`). Open `http://localhost:5173/?mock=1`. Also do this once with DevTools → Network → **Offline** checked, which is stricter than Wi-Fi off (it blocks loopback too).

- [ ] The MSW service worker registers (console: no `[MSW]` errors; `quiet: true` so no banner) before first render; HealthStrip reads healthy from `healthFixture`; Tools panel shows the registry from `registryFixture`.
- [ ] Every third-party host has an explicit handler — confirm **zero** requests leave the page during the whole rehearsal below (Network tab filter: exclude `localhost`). `main.tsx` starts MSW with `onUnhandledRequest: 'bypass'`, so an unhandled host would go to the network silently; the only proof is an empty external request list. ❗
- [ ] The query string survives navigation: click Maps, Saved, Landing → the URL keeps `?mock=1` (`shell/router.ts:88`). Losing it mid-demo would silently switch to a dead backend.
- [ ] Follow the **rehearsal script** (`UI_TRACK_4_ARCHITECTURE.md §4.4`) end to end:
  1. `/maps?mock=1` → **Find imagery** → type `Ahm` → Ahmedabad (from `public/samples/stac/places/ahmedabad.json`) → map flies.
  2. **Sentinel-1** → preset **6 mo apart** → shelf shows 16 S1 RTC fixtures (`search/s1-ahmedabad.json`), thumbnails from `thumbs/`.
  3. **Use as T1 / Use as T2** on the two *descending* scenes of 2026-03-03 and 2026-09-06 (same orbit) → **Analyse change** → `fetch/pair-ahmedabad.json` → console with two views loaded, compatibility **11/11**.
  4. Run → recorded SSE replay (`GET */v1/jobs/:jobId/events` handler) → DAG animates → answer card.
  5. **SITREP** → PDF downloads. **Export GeoJSON** → file downloads; open in QGIS on the laptop. ❗
- [ ] Other recorded fixtures behave: `Ben` → Bengaluru + Sentinel-1 → `s1-bengaluru.json` (12 scenes); `Che` → Chennai + Sentinel-2 → `s2-chennai.json`; `Ahm` + Sentinel-2 → `s2-ahmedabad.json`. An unrecorded pair (e.g. Chennai + S1) falls back to the nearest same-sensor recording — acceptable for rehearsal; note it is not a real result.
- [ ] Known limitation to remember on stage: the mock search handler **ignores dates** — moving the slider does not change the shelf. Do not narrate the slider as filtering in tier C.
- [ ] The online gate is bypassed in mock (`Discover.tsx:23` → `onlineFeatures || mockRequested()`) — Discover is live even with the setting off. Confirm; and confirm that **without** `?mock=1` the gate holds (Playwright test 2).
- [ ] Scenarios: `?mock=1` (canonical, templated cited answer, `uncited = 0`), `?mock=1&scenario=ungrounded` (two uncited spans, honesty flag rendered, generator says SYNTHETIC), `?mock=1&scenario=grounded` (one box on a 2:1 letterboxed image, overlay geometry correct at three window widths). An unknown `scenario=` falls back to canonical.
- [ ] Saved page in mock: save the run, reload, it is still there (IndexedDB), Export → SITREP/GeoJSON both work via `GET */v1/traces/:traceId` handler.
- [ ] Landing page in mock/offline: the three.js globe renders from `/samples/globe/land.png`; on a machine with no WebGL (or `prefers-reduced-motion`) the `GlobePoster` static frame renders instead — nothing blank. Test reduced motion via OS setting or DevTools rendering emulation.
- [ ] Hard refresh (Ctrl-Shift-R) with `?mock=1` still boots (the service worker re-registers; no cached-worker mismatch).

### 3.3 Tier transitions

- [ ] A → B → A during one session (toggle Wi-Fi twice): no reload needed; discovery recovers; basemap recovers; no duplicated toasts.
- [ ] B → C: stop the API while the console is idle. HealthStrip goes red with a clear message; the recovery instruction the team will use is *reload with `?mock=1`* — confirm the URL edit is the only step needed.
- [ ] From C back to A: remove `?mock=1`, reload — MSW is not started, real API is used (check that `/v1/health` hits `:8000`).

### 3.4 Tier C on the laptop specifically

- [ ] Repeat §3.2 rehearsal script on the Windows laptop from `frontend/dist` served by `serve_frontend.py`, Wi-Fi off, from a **cold boot**. Time it; note the total. This is the "rehearsed twice with Wi-Fi off" item on the cross-track checklist — do it twice. ❗

---

## 4. Laptop demo — GGUF `serve_vlm.ps1` on the air-gapped RTX 4070

Reference: `scripts/serve_vlm.ps1`, `DOCS/DEMO_LAPTOP_RUNBOOK.md`, `.env.laptop`.
Machine: Windows 11, RTX 4070 Laptop (8 GB VRAM, ~7.2 GB usable). **Wi-Fi off for everything below.**

### 4.1 The stick

- [ ] `models/SHA256SUMS` check (README step 1) prints nothing — no `BAD <file>` line — for `sq-lora-v2-full-merged-Q4_K_M.gguf` (~5.2 GB) and `sq-lora-v2-full-merged-mmproj-f16.gguf` (~1.1 GB). ❗
- [ ] `models/sq-lora-v2-full-merged/merge_manifest.json` present; it names the adapter SHA-256, base `Qwen/Qwen3-VL-8B-Instruct`, `SATQUERY_VLM_MAX_VIEWS=3`, context 8192.
- [ ] llama.cpp zips are a CUDA 12.4 x64 build **≥ b6600** (Qwen3-VL support); both `llama-*-bin-win-cuda-12.4-x64.zip` and `cudart-*.zip` unzipped into the **same** folder `..\llama.cpp\` (sibling of the repo) so `serve_vlm.ps1` auto-resolves `llama-server.exe`.
- [ ] `frontend/dist/` was built with `VITE_API_BASE=http://127.0.0.1:8000` (grep the built JS for `127.0.0.1:8000`).
- [ ] `data/checkpoints/seg/segformer-b5-loveda` and `data/checkpoints/cd/levircd_resnet18.ckpt.pt` exist (the registry plans the real tools only if both are present).
- [ ] `data/e2e/` scenes present if you intend `e2e_parity.py --scenes-dir data\e2e`.

### 4.2 Python side (no torch, no CUDA Python)

- [ ] `uv sync --group dev` completes offline from the wheel cache (pre-warm before the day). **This machine may run a bare `uv sync`; the ROCm box may not.**
- [ ] `Copy-Item scripts\demo_laptop\.env.laptop .env` → `.env` reads `SATQUERY_VLM_BACKEND=llamacpp`, `SATQUERY_VLM_SERVER_URL=http://127.0.0.1:8080`, `SATQUERY_VLM_MAX_VIEWS=3`, both checkpoint paths.

### 4.3 `serve_vlm.ps1`

- [ ] Run `.\scripts\serve_vlm.ps1` with **no** GGUFs in place → exits **66** with the "does not exist… mmproj required" message. Restore the files. (Verifies the guard; a vision model without its mmproj answers about no image at all.)
- [ ] Rename `..\llama.cpp` temporarily → exits **127** with the "was not found" message. Restore.
- [ ] Run normally. **Expect:** `server listening` on `127.0.0.1:8080` in **< 30 s**; the argument line shows `--ctx-size 8192 --n-gpu-layers 99 --image-min-tokens 1024 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --temp 0 --seed 0 --no-warmup`. ❗
- [ ] `curl http://127.0.0.1:8080/v1/models` answers with the model id, network adapter disabled.
- [ ] `nvidia-smi` idle after load: ≈ 6.3–6.8 GB used (weights + mmproj + KV).
- [ ] Env overrides are honoured: `$env:SATQUERY_VLM_PORT=8081; .\scripts\serve_vlm.ps1` binds 8081 (then revert). `-Model` / `-MmProj` parameters accept explicit paths.
- [ ] **Fallback path:** `$env:SATQUERY_VLM_GPU_LAYERS = 28; .\scripts\serve_vlm.ps1` starts, answers a request (slower), `nvidia-smi` shows less VRAM. Note the tokens/s difference for the speaker notes. Revert the variable.

### 4.4 API + frontend against the GGUF

- [ ] Window 2: `uv run --no-sync uvicorn satquery.api.app:app --host 127.0.0.1 --port 8000`; `curl http://127.0.0.1:8000/v1/health` reports the VLM `ready` and the tool count, offline.
- [ ] Window 3: `uv run --no-sync python scripts\demo_laptop\serve_frontend.py`; `http://localhost:5173` opens; HealthStrip shows VLM **`ready`** and **`tools 6/6`**. ❗
- [ ] `uv run --no-sync python scripts\e2e_parity.py --base-url http://127.0.0.1:8000` → **5/5**, Wi-Fi off. The cross-modal scenario needs both checkpoints. ❗
- [ ] During a **3-view** request (the cross-modal query), watch `nvidia-smi -l 1`: peak **< 7.4 GB**; no `CUDA out of memory` / server crash. If it spills, that is the `GPU_LAYERS=28` trigger. ❗
- [ ] Trace of a pair/cross-modal run records `views_ceiling: 3` with the explanatory note (the policy table still plans six; the cap is honest, not silent). Show this in the report page — a judge who asks sees the cap.
- [ ] Grounding quality at Q4_K_M: run the `grounding` scenario through the UI; the airplane box is on the airplane (this is what `--image-min-tokens 1024` protects). Compare visually with the bf16 result screenshot from the GPU box.
- [ ] Latency: first-token and total time for the change query and the cross-modal query; write them in the sign-off sheet. Anything > 60 s total needs a speaker-notes plan (talk through the DAG while it runs).
- [ ] Run the full §1 Track 4 pass (SITREP, GeoJSON → QGIS on this laptop, discovery in tier C) against this live GGUF stack.

### 4.5 Failure drills (rehearse the recoveries, not just the happy path)

- [ ] Kill `llama-server` mid-query → API returns a proper error envelope, console shows it, HealthStrip shows VLM down. Restart `serve_vlm.ps1` → HealthStrip recovers without restarting uvicorn.
- [ ] Laptop sleep/wake with all three windows open → everything still answers (CUDA context survives). If not, the recovery is: restart window 1 only.
- [ ] Reboot → cold start to first answer, timed. Target < 3 min including model load. Write it down.
- [ ] `?mock=1` fallback from this same `dist` (§3.4) — the last-resort path if the GPU dies on stage. ❗

### 4.6 Numbers that go on the slides

- [ ] `runs/eval/<name>/results.md` exists for **both** the bf16 run and the Q4_K_M run; the Q4_K_M column is within ≤ 2 pt VQA accuracy and ≤ 3 pt grounding recall of bf16 — or, if not, the gap is on the slide as a finding. The laptop serves Q4_K_M, so that is the honest number for the demo.
- [ ] The deck's numbers match those files exactly (no hand-transcription drift).

---

## 5. Cross-cutting checks (30 minutes, once)

- [ ] **Keyboard shortcuts** modal (⌘/) lists ⌘⇧S and ⌘⇧G; ⌘K, ⌘J, ⌘, all work; no shortcut fires while typing in an input except the chords documented as such.
- [ ] **Theme** (⌘J): dark and light both render the DAG, KPI cards, map HUD, and dialogs legibly; the SITREP PDF is theme-independent (print-safe tones).
- [ ] **Resize:** console at 1280 × 720 (projector) — DAG, evidence tray, and answer all visible without horizontal scroll; Maps HUD column collapses sensibly at ≤ 1024 px.
- [ ] **Projector rehearsal:** at 1280 × 720 mirrored, font sizes in the answer, KPI values, and DAG labels are readable from 5 m.
- [ ] **Report page** `/report/<trace>` renders for a live trace, a saved trace, and a mock trace; browser Print → multi-page audit record intact (this is separate from the one-page SITREP).
- [ ] **Storage:** Settings → Data controls shows the storage estimate; *Export data* / *Clear data* work; after Clear, Saved is empty and `onlineFeatures` resets to off.
- [ ] **Console hygiene:** one full A-tier run and one full C-tier run with DevTools open — **zero** uncaught errors, zero React key warnings, zero `[maps]` warnings other than the expected basemap one in tier B.
- [ ] **No CDN dependence:** in tier C, view the page source and the Network tab — every font, script, style, and image is same-origin.
- [ ] **Clock:** the demo laptop's clock is correct (SITREP timestamps and the STAC "last 30 d" preset both derive from it).

---

## 6. Sign-off sheet

| Section | Machine | Tier | Result | Tester | Date | Notes / defects |
|---|---|---|---|---|---|---|
| 0 Pre-flight | GPU box | A | | | | |
| 1.1 SITREP | GPU box | A | | | | |
| 1.1 SITREP | Laptop | C | | | | |
| 1.2 GeoJSON + QGIS | Laptop | A/B | | | | |
| 1.2 GeoJSON | Laptop | C | | | | |
| 1.3 Discovery HUD | GPU box | A | | | | |
| 1.3 Discovery HUD | Laptop | C | | | | |
| 2 DAG parity / SSE | GPU box | A | | | | |
| 3.1 Tier B | GPU box | B | | | | |
| 3.2 Tier C (run 1) | Laptop | C | | | | |
| 3.4 Tier C (run 2, cold boot) | Laptop | C | | | | |
| 4 GGUF laptop stack | Laptop | B | | | | |
| 4.5 Failure drills | Laptop | B/C | | | | |
| 5 Cross-cutting | both | all | | | | |

Timings to record: cold boot → first answer (laptop) ______ · change query total (laptop) ______ · cross-modal total (laptop) ______ · SITREP click → download ______ · peak VRAM during 3-view request ______ GB.

**Release rule:** every ❗ row passes on the laptop with Wi-Fi off, or the
presentation opens in `?mock=1` and says so. No code changes after sign-off;
if a defect forces one, `make ci` + the affected section re-run.
