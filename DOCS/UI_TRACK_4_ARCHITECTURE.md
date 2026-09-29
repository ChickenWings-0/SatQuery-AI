# Track 4 — High-Impact UI Features: Architectural Blueprint

> **Status (2026-09-29): implemented — design record.** SITREP, GeoJSON export and the
> STAC discovery HUD shipped in `91d8292`; Playwright `frontend/e2e/track4.spec.ts`
> covers them offline. Read this for the *why*; the code is the current truth.

*Originally a planning document. This expands
`ROADMAP_REMAINING_FIXES.md` §Track 4 into the component tree, the state and
data flow, the bundle boundaries, and the offline behaviour that code must
follow. Where this document and the roadmap disagree, this document wins.*

**Status as of 2026-09-15.** All three features live in `frontend/`. Mode is
**Operate** (PRODUCT.md): scanability, native expectations, brand in the
details. Every feature must work under `?mock=1` because that is the fallback
if the demo GPU or the venue network dies on stage.

## Corrections to the roadmap that this blueprint supersedes

| Roadmap says | Reality in the repo | What this document does |
|---|---|---|
| "MapLibre ≥ 4.6 supports globe natively" | `frontend/package.json` pins `maplibre-gl ^6.9.0`; globe is `map.setProjection({ type: 'globe' })` | No new dependency; a toggle in `ZoomHud`. |
| "behind `state/settings.ts → online features`" | `state/settings.ts` has only `customInstructions` and `seed` | Adds `onlineFeatures: boolean` (default `false`) to `Preferences`, persisted under the existing `satquery.settings` key. |
| GeoJSON "remove the optional `crs` field" | `export/geojson.ts` emits `crs` on every georeferenced collection **and** `footprintToGeoJson` | Both drop it; RFC 7946 §4 forbids the member. Consumers assume CRS84. |
| `?mock=1` "shows the whole flow without internet" | `main.tsx` starts MSW with `onUnhandledRequest: 'bypass'` — an unhandled third-party call **goes to the network** | Every third-party host gets an explicit handler (§2.4); nothing relies on bypass. |
| SITREP entry in "`components/thread/ThreadPanel.tsx` header" | The response card already has a `CardAction` row (bookmark, copy, share) | SITREP is a fourth `CardAction`, not a header button — it belongs to the answer, not the panel. |

---

## 1. Component hierarchy

Conventions shared by all three trees:

- A box marked **(existing)** is touched only to mount one child or add one
  menu row. A box marked **lazy** is reached only through `import()` and is
  the sole static importer of its heavy dependency (§3).
- Pure modules (`compose.ts`, `build.ts`, `geo/*.ts`) have no React and no
  DOM; they are what vitest covers. Hooks are the seam between stores and
  the pure modules. Components render and dispatch, nothing else.
- Every new user-facing string is listed in §1.4 so copy is reviewed once.

### 1.1 Tactical SITREP generator

```
components/thread/ThreadPanel.tsx (existing)         pages/saved/ExportMenu.tsx (existing)      shell/useHotkeys.ts (existing)
  └─ response card action row                          └─ <PopoverContent>                            └─ binding 'sitrep' ⌘⇧S
       └─ <CardAction label="SITREP">                       └─ <button class="menu-row">                   (scope: thread, chord: true)
              │                                                    │                                                │
              └────────────────────────┬───────────────────────────┴────────────────────────────────────────────────┘
                                       ▼
                     export/sitrep/useSitrep.ts                      ← the ONE seam: reads stores, drives the pure modules
                       state: 'idle' | 'composing' | 'rendering' | 'done' | 'failed'
                       run(source: 'live' | SavedRun)
                         │
                         ├─ 1. source → SitrepInput
                         │      live : useJobStore.result (AnalyzeResponse) + useUiStore.validation + job.artifacts
                         │      saved: SavedRun + fetchTrace(traceId)  (GET /v1/traces/{id}; on 404 → 'failed', toast)
                         │
                         ├─ 2. export/sitrep/compose.ts              pure, sync, unit-tested
                         │      composeSitrep(input): SitrepModel
                         │        ├─ kpi/registry.ts        selectKpis(fact_sheet)  → KEY MEASUREMENTS (same rows KpiCards show)
                         │        ├─ thread/annotate.ts     parseSource(citation)   → superscript numbering
                         │        ├─ thread/boxes.ts        boxesForResult()        → boxes to burn in
                         │        ├─ evidence/views.ts      groupViews() / primaryOf() → which views are T1|T2 / optical|SAR
                         │        └─ evidence/georef.ts     sceneGeoref()           → bounds strip, gsd for the scale bar
                         │
                         ├─ 3. export/sitrep/scene.ts                DOM (canvas), no pdf-lib
                         │      drawScenePanel(model, images): Promise<Uint8Array /* PNG */>
                         │        ├─ same-origin <img> decode (pattern of pages/saved/capture.ts thumbnail())
                         │        ├─ box overlay: components/stage/BboxOverlay.tsx geometry, re-drawn with 2D canvas
                         │        └─ north arrow · scale bar (evidence/georef.ts metersPerPixel) · bounds text (formatDms)
                         │
                         ├─ 4. export/sitrep/render.ts     ***lazy*** — sole static importer of pdf-lib + @pdf-lib/fontkit
                         │      renderSitrep(model, scenePng): Promise<Uint8Array>
                         │        ├─ PDFDocument.create(); registerFontkit; embed Geist + Geist Mono from /fonts/*.woff2 (fetch)
                         │        ├─ one A4 page; letter-safe margins (18 mm) so the same page prints on both
                         │        └─ layout table §1.1.1
                         │
                         └─ 5. export/download.ts                   shared with GeoJSON (moved out of export/geojson.ts)
                                downloadBlob(name, blob)
                                name = SITREP-<sceneId>-<yyyymmdd-hhmm>.pdf   (sceneId = manifest scene id, else trace id[0:8])
```

#### 1.1.1 `SitrepModel` (what `compose.ts` returns, what `render.ts` draws)

```ts
interface SitrepModel {
  header:   { sceneId: string; generatedAt: string; traceId: string; version: string; status: 'OK' | 'DEGRADED' | 'FAILED' }
  query:    string
  answer:   { text: string; citations: { index: number; claim: string; source: string }[] }   // superscripts = index
  scene:    { mode: 'single' | 'pair' | 'cross-modal'; labels: [string] | [string, string]; bounds: Bounds | null; gsdM: number | null }
  measurements: { label: string; value: string; unit?: string; source: string }[]             // ≤ MAX_CARDS, from selectKpis
  toolChain: { step: number; tool: string; version: string; state: NodeState }[]              // from result.trace.steps
  citations: { bound: number; uncited: number }                                                // uncited > 0 ⇒ visible badge
}
```

No number reaches the PDF that did not come through `selectKpis` or a
citation's `value`. This is the property the unit test asserts, and the
reason `compose.ts` exists as a separate pure module.

#### 1.1.2 Page layout (A4 portrait, also valid on US Letter)

```
┌──────────────────────────────────────────────────────────┐
│ SATQUERY AI · SITREP            <scene id> · <timestamp>  │  header band, Geist 600, 9 pt meta
│ Query: "…"                                 Status: OK     │  status uses the console's ok/warn/fail tones (print-safe)
├───────────────────────────┬──────────────────────────────┤
│  scene panel (PNG from    │  ANSWER — citation tags as   │  left column 92 mm, right column 88 mm
│  scene.ts): T1 | T2 or    │  superscripts ¹ ² ³          │
│  optical | SAR, boxes     │                              │
│  burned in, north arrow,  │  KEY MEASUREMENTS            │  tabular figures (Geist Mono)
│  scale bar, bounds strip  │  changed_area_pct   12.4 %   │
│                           │  ndbi_mean          0.31     │
├───────────────────────────┴──────────────────────────────┤
│ TOOL CHAIN  parse → checks 11/11 → change_detect@1.2 →   │  one line, wraps once at most
│             change_statistics@1.0 → vlm_change_vqa@1.0    │
│ CITATIONS   3 bound · 0 uncited        trace <id> · v1.0  │  uncited > 0 ⇒ "1 UNCITED" badge in warn tone
└──────────────────────────────────────────────────────────┘
```

Overflow policy: answer text is clipped to the column with an ellipsis and a
"see full report at /report/<trace>" footnote; the page count is **always
1**. The Playwright test parses the output back and asserts that.

### 1.2 GeoJSON exporter

```
components/stage/EvidenceTray.tsx (existing)     pages/saved/ExportMenu.tsx (existing)     shell/useHotkeys.ts (existing)
  └─ <ExportGeoJsonButton/>                        └─ menu row "GeoJSON"                      └─ binding 'geojson' ⌘⇧G
       label: "Export GeoJSON" | "Export GeoJSON (pixel space)"   (decided by `bounds !== null`, nowhere else)
              │                                           │                                        │
              └───────────────────────┬───────────────────┴────────────────────────────────────────┘
                                      ▼
                     export/geojson/useGeoJsonExport.ts
                       ├─ if run has no class raster ──────────────► build + download immediately (no dialog)
                       └─ else open <GeoJsonExportDialog/>            components/export/GeoJsonExportDialog.tsx
                                                                        ├─ Radix Dialog (components/ui/dialog.tsx)
                                                                        ├─ checkbox "Include segmentation masks"
                                                                        ├─ CRS line: "WGS84 (CRS84)" | "Pixel space — not georeferenced"
                                                                        └─ [Export]
                       ▼
             export/geojson/build.ts                    pure; replaces the current export/geojson.ts
               buildCollection(input): FeatureCollection
                 ├─ boxes  → one Polygon feature per box **per view** (view_id), corner() through bounds
                 ├─ winding.ts   rewind(ring) → exterior CCW, closed, 7-decimal rounding
                 ├─ bbox   of all features on the collection
                 ├─ properties per feature (§1.2.1)
                 └─ pixel space ⇒ top-level `note` + per-feature `crs: 'pixel', frame: 1000`; **no `crs` member ever**
             export/geojson/masks.ts                    ***lazy*** — sole static importer of marching-squares
               vectoriseMasks(raster, classes): MultiPolygon features, simplified 0.5 px (Douglas–Peucker, own 40-line impl)
             export/download.ts                         downloadBlob(`satquery-<trace8>.geojson`, blob 'application/geo+json')
```

#### 1.2.1 Feature properties (the QGIS attribute table *is* the audit trace)

| property | source |
|---|---|
| `label`, `confidence` | `NormalisedBox.label` / `.score` |
| `view_id` | `ViewGroup.key` (T1/T2 or optical/SAR become two toggleable layers) |
| `source_step`, `tool`, `tool_version` | the `AuditTrace.steps[n]` that produced the artifact holding the box |
| `citation` | `step:n/scalars.<path>` when the box is the subject of a cited scalar, else `null` |
| `trace_id`, `scene_id`, `acquired_at`, `sensor` | `AnalyzeResponse.trace_id`, `InputManifest` fields |
| `crs`, `frame` | **only in pixel space**: `'pixel'`, `1000` |

`SavedRun` today stores only `boxes` and `bounds`; the saved-run path fetches
the trace (`GET /v1/traces/{id}`) to fill the step/tool columns, and degrades
to `null`s with a toast ("Exported without tool provenance — the server no
longer has this trace.") when it cannot.

### 1.3 Interactive map / STAC imagery fetcher

```
pages/Maps.tsx (existing)
 ├─ (existing) empty state — gains one button: "Find imagery" → opens Discover even with no run loaded
 ├─ (existing) <MapStage/> | <PixelStage/>
 │     MapStage.tsx (existing, lazy, owner of maplibre-gl) — additions:
 │       ├─ source 'stac-footprints' (geojson) + layers 'stac-footprints-fill' / '-line' / '-selected'
 │       ├─ click on a footprint → useStacStore.hover/select; hover ⇄ SceneShelf highlight
 │       ├─ projection: useMapStore.projection ('mercator' | 'globe') → map.setProjection()
 │       └─ when no run is loaded, MapStage now mounts with georef = null and the world view (§2.2)
 │
 └─ pages/maps/hud/Discover.tsx                 the right-hand HUD column; renders nothing when gated off (§4)
      ├─ <OnlineGate/>                          settings.onlineFeatures || mockRequested() — else a one-line invitation
      ├─ hud/PlaceSearch.tsx                    input, 400 ms debounce, result list, ↑↓⏎, → useStacStore.setPlace
      ├─ hud/TimeSlider.tsx                     one handle (single) or two (pair); presets "last 30 d", "6 mo apart"
      ├─ hud/CollectionPicker.tsx               Segmented (components/ui/Segmented.tsx): Sentinel-2 · Sentinel-1 · Both
      ├─ hud/SceneShelf.tsx                     virtualised list of StacCard
      │     └─ StacCard: thumb, date, cloud % (S2) / orbit state + polarisation (S1), [Use as T1] [Use as T2]
      └─ hud/FetchBar.tsx                       "Analyse" | "Analyse change" (pair) · progress · error line
                                                  → useImageryFetch (§2.3) → console

pages/maps/hud/ZoomHud.tsx (existing) — gains a Globe toggle button (aria-pressed)

geo/                                            plain TS, plain fetch, zero React
 ├─ nominatim.ts     search(q, signal): Promise<Place[]>       q ≥ 3 chars; 1 req/s; UA via `Referer` only (browser cannot set User-Agent)
 ├─ stac.ts          searchItems(q, signal): Promise<StacItem[]>   POST /api/stac/v1/search, limit 24, sortby datetime desc
 ├─ planetary.ts     signedHref(href, collection): Promise<string>  SAS token cache per collection (expiry from response)
 ├─ collections.ts   COLLECTIONS table: id, label, assets the pipeline wants, filters (cloud, orbit)
 └─ deadline.ts      withDeadline(promise, ms, signal) — the 5 s rule in one place
```

### 1.4 New UI copy (reviewed once, here)

| Where | String |
|---|---|
| SITREP action | `SITREP` · tooltip `One-page brief as PDF (⌘⇧S)` |
| SITREP toasts | `SITREP saved.` · `Could not build the SITREP — <reason>.` |
| GeoJSON button | `Export GeoJSON` · `Export GeoJSON (pixel space)` |
| GeoJSON toasts | `GeoJSON exported in WGS84.` · `GeoJSON exported in pixel space (no georeference).` (both existing) |
| Online gate (off) | `Imagery search is off. Turn on online features in Settings to search Sentinel scenes.` [Open settings] |
| Offline state | `Search needs a network — the loaded scene is still here.` [Retry] |
| Empty shelf | `No scenes in this window. Widen the dates or raise the cloud limit.` |
| Fetch progress | `Fetching <n> of <m> — clipping to the pipeline's window…` |
| Pair orbit mismatch | `These two scenes are on different orbits; the co-registration check will fail. Pick a same-orbit pair.` |

---

## 2. State management and data flow

### 2.1 Stores

| Store | Persistence | Owns | Touched by Track 4 |
|---|---|---|---|
| `state/job.ts` | session | live run (`result`, `artifacts`, `nodes`) | read-only (SITREP, GeoJSON live path) |
| `state/ui.ts` | session | `section`, `files`, `validation`, `selectFiles`, `recentRuns` | read (georef); `selectFiles`-equivalent for fetched scenes (§2.3) |
| `state/map.ts` | `basemap` in localStorage | Maps HUD: layers, split, basemap, zoom, cursor | + `projection: 'mercator' \| 'globe'` (remembered, key `satquery.maps.projection`) |
| `state/settings.ts` | localStorage `satquery.settings` | `customInstructions`, `seed` | + `onlineFeatures: boolean` (default `false`) and `setOnlineFeatures` |
| `state/library.ts` | IndexedDB | `SavedRun[]`, projects | read-only |
| **`state/stac.ts`** (new) | **session only, never persisted** | discovery flow (below) | new |

`state/stac.ts` is deliberately not persisted: a shelf of stale scenes from
yesterday's rehearsal appearing on stage is worse than an empty one.

```ts
interface StacQuery {
  placeText: string
  place: Place | null                       // { name, bbox: Bounds, center: [lon, lat] }
  bbox: Bounds | null                       // place.bbox, or the map viewport if the user drew none
  datetime: [from: string, to: string]      // ISO dates; TimeSlider writes these
  collections: ('sentinel-2-l2a' | 'sentinel-1-rtc' | 'sentinel-1-grd')[]
  cloudMax: number                          // S2 only, default 20
}

interface StacState {
  query: StacQuery
  status: 'idle' | 'searching' | 'ok' | 'empty' | 'error' | 'offline'
  error: string | null
  results: StacItem[]                       // normalised: id, collection, datetime, bbox, geometry, cloud, orbitState, thumbHref, assetHrefs
  selection: { t1: string | null; t2: string | null }
  hover: string | null
  fetch: { status: 'idle' | 'running' | 'done' | 'failed'; progress: [done: number, total: number]; uploadIds: string[]; error: string | null }

  setPlace(place: Place | null): void       // also sets bbox and asks the map to fly there (via a one-shot `flyTarget`)
  setDatetime(range): void; setCollections(ids): void; setCloudMax(n): void
  search(signal?: AbortSignal): Promise<void>
  select(slot: 't1' | 't2', id: string | null): void
  setHover(id: string | null): void
  reset(): void
}
```

### 2.2 How `stac.ts` and `MapStage.tsx` interact

The rule: **`MapStage` owns every maplibre object; `stac.ts` owns every fact.**
Nothing outside `MapStage.tsx` imports `maplibre-gl` (enforced by
`bundle-boundaries.test.ts`), and `MapStage` never calls Nominatim or STAC.

```
PlaceSearch ──setPlace()──► stac.ts ──(subscribe: query.place)──► MapStage: map.flyTo(place.center) / fitBounds(place.bbox)
TimeSlider  ──setDatetime──► stac.ts ──search()──► geo/stac.ts ──► results[]
                                                        │
                                          ┌─────────────┴──────────────┐
                                          ▼                            ▼
                     MapStage (subscribe: results, selection, hover)   SceneShelf (subscribe: results, selection, hover)
                       source 'stac-footprints'.setData(toFeatureCollection(results))
                       paint: fill-opacity by [hover, selected]; 'stac-footprints-selected' line for t1/t2
                       map.on('click', 'stac-footprints-fill') → select(nextFreeSlot, id)
                       map.on('mousemove', …)                   → setHover(id)      (throttled, 60 Hz max)
```

Selectors, not whole-store subscriptions: `MapStage` subscribes to
`results`, `selection`, `hover`, and `query.place` individually so a keystroke
in the search box never re-renders the map.

**Map lifecycle change.** Today `Maps.tsx` mounts `MapStage` only when a run
has a georef, and `MapStage` rebuilds the map when `georef.bounds` changes.
Discovery needs a map with no run. So:

- `MapStage` accepts `georef: Georef | null`. With `null` it initialises at a
  world view (`center: [78, 22], zoom: 3` — India, the demo's AOI) with the
  same `EMPTY_STYLE`; the footprint/scene effects early-return.
- The mount effect keys on `georef?.bounds.join(',') ?? 'world'`, so loading
  a run after searching rebuilds the map exactly as it does today.
- `Maps.tsx` mounts `MapStage` whenever `groups.length > 0 && georef` **or**
  Discover is open; the empty-state card gains a "Find imagery" button that
  opens Discover.

**Globe projection.** `useMapStore.projection` → `map.setProjection({ type })`
in an effect. Image sources and geojson sources render unchanged on the
globe; the swipe-clip effect (which rebuilds the B layer's coordinates from
unprojected screen x) is disabled while `projection === 'globe'` and the
`SwipeHandle` hides, because `unproject` on a globe near the horizon is not
a longitude line. This is stated in the HUD ("Swipe is available in flat
view").

### 2.3 Fetch → console handoff

```
FetchBar "Analyse [change]"
  └─ export nothing; useImageryFetch():
       1. validate selection: pair ⇒ both S1 or both S2; S1 pair ⇒ same sat:orbit_state, else inline error (no request)
       2. POST /v1/imagery/fetch  { items: [{ collection, id, assets: [...] }], bbox, max_px: 2048 }
            ← 202 { upload_ids: [...], scene_ids: [...] }            (backend does the COG window read, §5)
       3. useUiStore.selectUploads(uploadIds, meta)   ← new action, sibling of selectFiles: same pre-flight (POST /v1/validate
            by upload_id), same `validation` result, same stage — the fetched scene is not a special path
       4. setSection('explore'); composer pre-filled from the collection ("What changed between these two acquisitions?")
```

The frontend never downloads a COG. A Sentinel-2 tile is ~1 GB; the
pipeline wants a ≤ 2048 px window; `rasterio` reads that window in one HTTP
range request from the backend. The browser sees only `upload_id`s that
`POST /v1/jobs` and `/v1/analyze` already accept, so nothing downstream
changes.

### 2.4 How `?mock=1` intercepts the network

`main.tsx` starts MSW *before the first render* when `?mock=1` is present,
with `onUnhandledRequest: 'bypass'`. Bypass means **an unhandled request
goes to the real network**. So every host Track 4 talks to gets an explicit
handler in `mocks/handlers.ts`, and the fixtures live as static files under
`public/samples/stac/`, served by redirect — the same pattern the artifact
handler uses for `public/mock-artifacts/`.

```
public/samples/stac/
  places/ahmedabad.json  places/bengaluru.json  places/chennai.json     ← 3 recorded Nominatim responses
  search/s2-ahmedabad.json  search/s1-ahmedabad.json  search/s1-bengaluru.json  search/s2-chennai.json
  thumbs/<item-id>.jpg                                                   ← ≤ 40 KB each, 24 max
  fetch/pair-ahmedabad.json                                              ← recorded /v1/imagery/fetch response
  README.md                                                               ← provenance + re-record command
```

| Request | Handler behaviour |
|---|---|
| `GET https://nominatim.openstreetmap.org/search?q=…` | pick the fixture whose name the query starts with (case-insensitive); unknown → `[]` after `delay(250)` |
| `POST https://planetarycomputer.microsoft.com/api/stac/v1/search` | read `collections` + `bbox` from the body, serve the matching `search/*.json`; the fixture's thumbnail hrefs are rewritten to `/samples/stac/thumbs/…` |
| `GET https://planetarycomputer.microsoft.com/api/sas/v1/token/:collection` | `{ token: 'mock', 'msft:expiry': <now+1h> }` |
| `POST */v1/imagery/fetch` | `delay(900)`, serve `fetch/pair-ahmedabad.json` (its `upload_id`s are the ones the recorded `/v1/validate` fixture already answers for) |

Guarantees this needs, and how they are kept:

- `geo/*` use plain `fetch` (no XHR wrappers, no `new Image()` for data) so
  the service worker sees every call. Thumbnails are `<img src>` — also
  intercepted by the worker, hence the href rewrite.
- `mockRequested()` (exists in `mocks/browser.ts`) is re-exported from a
  tiny non-mock module so `Discover.tsx` can read it without importing MSW
  into the production bundle. Under mock the online gate is treated as on.
- Playwright runs the whole flow with the network stubbed *off* at the
  browser level (`context.setOffline(true)`) after page load, proving no
  bypass leak.

---

## 3. Dependency rules and Vite bundle boundaries

The console entry budget is **180 KB gz** (`frontend/scripts/check-bundle.mjs`)
and every heavy dependency has exactly one static owner reached only through
`import()` (`src/__tests__/bundle-boundaries.test.ts`). Track 4 adds two
dependencies and zero to the entry.

| Dependency | Size (gz, approx.) | Sole static importer | Load trigger | Chunk name |
|---|---|---|---|---|
| `pdf-lib` + `@pdf-lib/fontkit` | ~250 KB + ~120 KB | `export/sitrep/render.ts` | `await import('@/export/sitrep/render')` inside `useSitrep.run()` — on click or ⌘⇧S, **never at mount** | `assets/sitrep-[hash].js` |
| `marching-squares` | ~6 KB | `export/geojson/masks.ts` | `import()` when "Include masks" is ticked and Export pressed | (default naming) |
| `maplibre-gl` | (already lazy) | `pages/maps/MapStage.tsx` | unchanged | `assets/map-[hash].js` |
| STAC / Nominatim | none | `geo/*` are plain `fetch` | — | entry-safe, but only imported by `pages/maps/**` (itself lazy) |

Rules, each with its guard:

1. **`pdf-lib` is dynamically imported at the click.** `useSitrep` holds the
   module promise so a second click does not re-fetch; nothing pre-warms it.
   Guard: `OWNERS['pdf-lib'] = /^export\/sitrep\/render\.ts$/` and
   `OWNERS['@pdf-lib/fontkit']` in `bundle-boundaries.test.ts`; a new case in
   "reaches every lazy owner only through import()" for
   `from '@/export/sitrep/render'`; `check-bundle.mjs` leak list gains
   `'sitrep'`; `vite.config.ts chunkFileNames` names the chunk when
   `has('node_modules/pdf-lib/')`.
2. **Fonts are fetched, not bundled.** Geist woff2 already ships in
   `public/fonts/` (`fonts:sync`); `render.ts` fetches them at render time
   and caches the bytes in module scope. Offline-safe: same origin.
3. **`compose.ts`, `scene.ts`, `build.ts`, `winding.ts` have no heavy
   imports** and may be statically imported by the entry-adjacent
   `ThreadPanel`; they are a few KB. Guard: `check-bundle` budget.
4. **`geo/*` never import React, stores, or maplibre.** They take arguments
   and return promises. Guard: a `bundle-boundaries` case asserting no
   `from '@/state/` or `from 'react'` inside `geo/`.
5. **Fixtures are static assets**, never `import`ed (`public/samples/stac/`
   is outside `src/`). `mocks/**` stays reachable only via the dynamic
   import in `main.tsx`.
6. **No Cesium, no deck.gl, no turf.** Winding and simplification are
   ~60 lines and are written in-repo (`winding.ts`); pulling `@turf/*` for
   them would cost more than the feature.
7. **`state/stac.ts` is imported only by `pages/maps/**`**, so it rides in
   the Maps chunk. Guard: a `bundle-boundaries` case, same shape as the
   library rule.

`npm run build && npm run check:bundle` after each PR; `sitrep-*.js` must
not appear in `entry.imports`.

---

## 4. Offline resilience — the venue Wi-Fi drops on stage

Three network tiers exist on the demo laptop, and the behaviour is defined
per tier:

| Tier | What is reachable | Which features work |
|---|---|---|
| **A** internet + local API | everything | all |
| **B** local API only (venue Wi-Fi gone) | `:8000`, same-origin assets | scene map, globe, SITREP, GeoJSON, the console; discovery degrades (below) |
| **C** nothing (GPU dead ⇒ `?mock=1`) | static files via MSW | all three, from fixtures |

Every online call obeys three invariants: (1) it is behind
`settings.onlineFeatures` (off by default, remembered), (2) it fails to a
toast or inline line within **5 s** (`geo/deadline.ts`, one constant), and
(3) failing never removes anything already on screen.

### 4.1 MapLibre stage and globe

| Condition | Behaviour |
|---|---|
| Basemap tiles fail (`map.on('error')`, `sourceId === 'basemap'`) | **Existing path, unchanged:** basemap → `none` (not remembered), toast "Basemap unreachable — showing scene only." with Retry. |
| Globe toggled with no network | Works: projection is client-side geometry. The scene image source, footprint line, and STAC footprints render on the globe with `EMPTY_STYLE`; there is no globe tile layer to fail. Optional: a low-res `public/samples/globe/land.png` (already present for the landing globe) as an `image` source over `[-180,-90,180,90]` so the globe is not a black sphere — same-origin, offline-safe. |
| Network drops mid-flyTo | Nothing to do; no tiles in flight unless basemap is on, which is the row above. |
| Scene images (`/v1/artifacts/...`) fail | Local API is down, which is tier C: the user reloads with `?mock=1`; the HealthStrip already says the API is unreachable. |

### 4.2 STAC fetcher and place search

| Step | Failure | Exact behaviour |
|---|---|---|
| Nominatim search | timeout 5 s / DNS / TLS | `status: 'offline'`; result list replaced by "Search needs a network — the loaded scene is still here." [Retry]; input stays editable; no toast (inline is enough, it is where the user is looking). |
| Nominatim search | HTTP 429 / 5xx | `status: 'error'`, inline "Place search is rate-limited — try again in a moment." ; `Retry-After` honoured if present. |
| STAC search | timeout / transport | `status: 'offline'`; **existing `results` are kept** and stay clickable; footprints stay on the map; shelf header shows an offline pill. |
| STAC search | 4xx (bad bbox etc.) | `status: 'error'` with the API's message; a contract bug, so also `console.error`. |
| SAS token | any failure | Thumbnails fall back to the item's `rendered_preview` (no token needed) or a neutral placeholder tile; search results are unaffected (search needs no token). |
| `POST /v1/imagery/fetch` | local API up, PC unreachable | API answers the §6 envelope (502, `hint: "Planetary Computer unreachable"`); FetchBar shows the hint; **selection is preserved** for retry. |
| `POST /v1/imagery/fetch` | local API down | `SatQueryError.transport` → same toast the composer uses ("Start the API…"); HealthStrip already red. |
| `navigator.onLine` becomes true / `online` event | — | `status: 'offline'` → `'idle'` and the last query re-runs once automatically (at most one auto-retry; after that, manual). |

The gate itself: with `onlineFeatures === false` and not in mock, Discover
renders a single line and a button to Settings → Data tab; no request is
ever constructed, not even a DNS prefetch.

### 4.3 SITREP and GeoJSON

Both are zero-network in tiers A and B: inputs are the stores, artifacts are
same-origin `/v1/artifacts/...` (already loaded and decoded for the stage),
fonts are same-origin. The canvas is not tainted because every image is
same-origin (the invariant `pages/saved/capture.ts` already relies on). Only
the saved-run path fetches (`GET /v1/traces/{id}`), and it degrades:

- trace 404/transport → SITREP still renders from the `SavedRun` summary
  (headline KPI, boxes, thumb) with TOOL CHAIN reading "not available —
  trace no longer on the server"; GeoJSON exports with provenance columns
  `null` and the toast in §1.2.1.

### 4.4 Tier C — `?mock=1` rehearsal script

Wi-Fi off, API stopped, `npm run preview` (or the packaged `dist/`):

1. `/maps?mock=1` → "Find imagery" → type `Ahm` → Ahmedabad → map flies.
2. Sentinel-1 → date presets "6 mo apart" → shelf shows S1 fixtures.
3. Use as T1 / Use as T2 (same orbit) → "Analyse change" → console, two
   views, compatibility 11/11 (the recorded bi-temporal validate fixture).
4. Run → recorded SSE replay → answer card → SITREP → PDF downloads.
5. Evidence tray → Export GeoJSON → file downloads; open in QGIS on the
   laptop (QGIS is offline-capable).

This is the "rehearsed twice with Wi-Fi off" item on the cross-track
checklist.

---

## 5. Backend boundary (additive only)

The only server work in Track 4. Kept to one page here; the frontend
contract is what matters to this document.

```
src/satquery/api/routers/imagery.py      POST /v1/imagery/fetch
src/satquery/ingest/stac_fetch.py        window_read(item, assets, bbox, max_px) -> Path  (rasterio COG range reads)
src/satquery/schemas/api.py              ImageryFetchRequest / ImageryFetchResponse
tests/unit/test_stac_fetch.py            256 px window from a local synthetic COG; no network
```

| | |
|---|---|
| Request | `{ items: [{ collection, id, assets: string[] }], bbox: [w,s,e,n], max_px: 2048 }` |
| Response | `202 { upload_ids: string[], scene_ids: string[], crs: string, gsd_m: number }` — each `upload_id` is a GeoTIFF in the ArtifactStore that `/v1/validate`, `/v1/analyze`, `/v1/jobs` accept today |
| Errors | §6 envelope; `502` with `hint` when Planetary Computer is unreachable; `413` when the window exceeds `max_px` after clipping |
| Contract | additive: new router + schema; regenerate `openapi.json` and `frontend/src/api/schema.d.ts` (`npm run gen:api`) in the same change; `DOCS/API_CONTRACT.md` gains §4.11 |
| Assets | S2: B02/B03/B04/B08/B11/B12 (what `spectral_index_analyzer` reads); S1 RTC preferred over GRD: `vv`, `vh`; record `sat:orbit_state` in the manifest |

---

## 6. Verification

| Feature | vitest | Playwright (`?mock=1`, browser offline) | Manual |
|---|---|---|---|
| SITREP | `compose.test.ts`: recorded CDVQA fixture → 3 citations ⇒ 3 measurement rows, `uncited === 0`; `?scenario=ungrounded` fixture ⇒ `uncited === 2` and model carries the badge flag; no measurement value absent from `selectKpis` output | click SITREP → download `application/pdf` > 20 KB; parse back with pdf-lib in the test ⇒ exactly 1 page; ⌘⇧S produces the same file name pattern | print on A4 and Letter, nothing clipped |
| GeoJSON | `geojson.test.ts` +: exterior ring CCW, ring closed, `bbox` present, no `crs` member, 7-decimal rounding, one feature per box per view, pixel-space `note`; validate against the `geojson` npm validator | click Export GeoJSON in the tray → file; pair run ⇒ two `view_id` values | QGIS 3.34 *Add Vector Layer*: boxes on the S2 tile, attribute table shows `citation`; ArcGIS Pro *Add Data* |
| STAC | `geo/__tests__/`: recorded Nominatim + STAC fixtures replayed; `deadline.ts` rejects at 5 s; orbit-mismatch rule; `stac.ts` reducer: offline keeps `results` | search "Ahmedabad" → two S1 scenes 6 months apart → "Analyse change" → console with two views and 11/11 | online: real API; fetched GeoTIFFs open in QGIS with correct CRS |
| Bundles | `bundle-boundaries.test.ts` new rows (§3) | — | `npm run build && npm run check:bundle`: entry ≤ 180 KB, no `sitrep`/`map` in entry imports |

---

## 7. Implementation order and file list

Order: **4.2 → 4.1 → 4.3.** GeoJSON is a refactor of working code with new
tests; SITREP reuses its download helper and the capture pattern; STAC is
the largest surface and the only one with backend work.

**New files**

```
frontend/src/export/download.ts
frontend/src/export/geojson/{build,winding,masks,useGeoJsonExport}.ts        (build.ts replaces export/geojson.ts; tests move with it)
frontend/src/components/export/GeoJsonExportDialog.tsx
frontend/src/export/sitrep/{compose,scene,render,useSitrep}.ts
frontend/src/export/sitrep/__tests__/compose.test.ts
frontend/src/geo/{nominatim,stac,planetary,collections,deadline}.ts
frontend/src/geo/__tests__/{nominatim,stac,deadline}.test.ts
frontend/src/state/stac.ts
frontend/src/pages/maps/hud/{Discover,PlaceSearch,TimeSlider,CollectionPicker,SceneShelf,FetchBar}.tsx
frontend/public/samples/stac/**                                               (fixtures; re-record command in DOCS/STAC_FIXTURES.md)
frontend/e2e/{sitrep,geojson,stac}.spec.ts
src/satquery/api/routers/imagery.py · src/satquery/ingest/stac_fetch.py · tests/unit/test_stac_fetch.py
```

**Modified files**

```
frontend/src/components/thread/ThreadPanel.tsx        one CardAction
frontend/src/pages/saved/ExportMenu.tsx               SITREP row; GeoJSON row → useGeoJsonExport
frontend/src/components/stage/EvidenceTray.tsx        ExportGeoJsonButton
frontend/src/shell/shortcuts.ts · shell/useHotkeys.ts ⌘⇧S, ⌘⇧G (chord: true), 'globe' G in maps scope
frontend/src/state/settings.ts                        onlineFeatures
frontend/src/components/shell/SettingsDialog.tsx      Data tab toggle
frontend/src/state/map.ts                             projection
frontend/src/state/ui.ts                              selectUploads
frontend/src/pages/Maps.tsx · pages/maps/MapStage.tsx · pages/maps/hud/ZoomHud.tsx
frontend/src/mocks/handlers.ts · mocks/browser.ts     Track 4 handlers; mockRequested re-export
frontend/src/__tests__/bundle-boundaries.test.ts · vite.config.ts · scripts/check-bundle.mjs
frontend/package.json                                 pdf-lib, @pdf-lib/fontkit, marching-squares (deps); geojson-validation (dev)
src/satquery/schemas/api.py · openapi.json · frontend/src/api/schema.d.ts · DOCS/API_CONTRACT.md
```

Nothing in this list is written until this document is reviewed.
