# SatQuery AI — Frontend Blueprint (SIH 2026 presentation build)

**Status:** design brief, produced by `/impeccable shape`. No code in this document is final; every
snippet is an interface contract for the build that follows. Companion to `PRODUCT.md` (product
truth) and `frontend/src/styles/theme.css` (incumbent visual authority).

**Scope in one line:** four Operate-mode sections that are currently dead buttons, an account
popover, a notification centre, a light theme, a shortcuts guide, a daily quote, one header
deletion, and a production-polish checklist (§6) — all inside the *existing* SatQuery world,
elevated rather than replaced. The Persuade-mode landing page has its own document:
[`landing_page_blueprint.md`](landing_page_blueprint.md).

**Documents:** `PRODUCT.md` (product truth) · this file (app shell, sections, tokens, production
checklist) · `landing_page_blueprint.md` (the `/` page, Odida, the 3D globe).

---

## 0. Direction contract

### 0.1 What is already true (and stays)

The incumbent world in `theme.css` is coherent, product-specific and enforced by test: warm
near-black ground, terracotta for edges/icons/active, sand as the accent's small-text voice and the
citation-pill fill, an emerald/amber/red status vocabulary, Inter Variable + Geist Mono Variable
self-hosted for the offline lab. The honesty guards (citation pills, wavy `uncited` underline,
nulls-stay-null) are the brand. **This is refinement plus expansion, not redesign.** Nothing below
introduces a second accent hue or a gradient that is not derived from the terracotta/sand pair.
The one sanctioned addition to the type system is **Odida**, a display face confined to the landing
page's `.t-display` / `.t-display-sub` (declared in §0.5, specified in `landing_page_blueprint.md`
§3); it never appears in the console.

### 0.2 What changes

The world gains a *night-side* material layer it never had: the atmosphere of a ground station.
Three new materials, used everywhere the brief asks for "geospatial aesthetics" so they read as one
system rather than a landing-page costume:

| Material | What it is | Where it appears |
|---|---|---|
| **Graticule** | a 1px sand grid at 6–8 % alpha on a 48px module, with a heavier every-fourth line, drawn by CSS `background-image` (two `linear-gradient`s), never an image | landing hero ground, empty states, Maps HUD ground, popover headers |
| **Reticle** | a crosshair (two hairlines + 3px tick marks + a 10px open square) in sand, 40 % alpha | hero focal lock, bounding-box previews, coordinate locator, notification empty state |
| **Glass** | `oklch(0.22 0.018 55 / 0.62)` fill · `backdrop-filter: blur(16px) saturate(1.2)` · 1px edge at `oklch(0.93 0.015 65 / 0.08)` · inner top highlight `inset 0 1px 0 oklch(1 0 0 / 0.04)` | floating HUD controls on Maps, account popover, notification popover, shortcuts modal, landing nav |

Glass is *bounded*: it appears only on elements that float over imagery or the graticule. Cards on
the plain ground stay the opaque `.card`. Blur regions are always `contain: paint` and never larger
than a viewport quarter, so a 24 GB GPU driving a VLM is not also compositing a full-screen blur.

### 0.3 Visitor modes per surface

| Surface | Mode | Consequence |
|---|---|---|
| `/` Landing | **Persuade** | one rehearsed focal sequence, real product truth as proof, single CTA |
| Use Cases | Operate (gallery) | scanable grid, load-in-one-click, no marketing copy |
| Maps | Operate (HUD) | controls recede over imagery; keyboard-complete |
| Saved / Projects | Operate (workspace) | density first; empty state is the one delight moment |
| Popovers, modal, notifications | Operate | fast (≤200 ms), predictable, dismiss-anywhere |

### 0.4 Token migration: hex → OKLCH, dark → dual theme

`@theme` currently hard-codes dark hexes. Two changes, done once, before any feature work:

1. **Every `--color-*` becomes OKLCH**, with the resolved sRGB hex kept in a trailing comment so the
   existing contrast test keeps passing until it is taught OKLCH (see §0.6).
2. **Theme scoping.** Tailwind v4 `@theme` variables are global; the light theme is applied by
   redeclaring the same custom properties under `:root:not(.dark)` in `@layer base`, and `.dark` on
   `<html>` (set by `useThemeStore`, §4.3) wins. Tailwind utilities (`bg-bg-main`, `text-text-lo`)
   resolve to `var(--color-…)` at runtime, so *no component changes* for theming — that is the whole
   reason to use `@theme` rather than `@theme inline`.

```css
/* theme.css — additions and conversions (values are the incumbent hexes, converted) */
@theme {
  /* ground & surfaces */
  --color-bg-main:          oklch(0.150 0.012 60);   /* #100c0a */
  --color-surface-card:     oklch(0.220 0.018 55);   /* #211814 */
  --color-surface-elevated: oklch(0.262 0.024 55);   /* #2d2018 */
  --color-line:             oklch(0.245 0.020 50);   /* #2a1f1a */
  --color-line-soft:        oklch(0.205 0.016 50);   /* #1e1612 */
  /* text */
  --color-text-hi:          oklch(0.930 0.015 65);   /* #f0e6df — 15.8:1 on ground */
  --color-text-lo:          oklch(0.620 0.030 55);   /* #9a8579 — 5.57:1 ground · 4.98:1 card */
  /* accent (unchanged roles) */
  --color-accent-warm:        oklch(0.600 0.110 45); /* #ba704f — edges, icons, ≥18px text */
  --color-accent-warm-strong: oklch(0.560 0.110 40); /* #ab6242 — button fill, white on it 4.61:1 */
  --color-accent-warm-text:   oklch(0.770 0.090 65); /* #dfa878 — small-text accent voice */
  --color-surface-sand:       oklch(0.770 0.090 65);
  --color-evidence:           oklch(0.770 0.090 65);
  --color-on-evidence:        oklch(0.150 0.012 60);
  /* status — unchanged */
  --color-ok:   oklch(0.723 0.190 150);  /* #22c55e */
  --color-warn: oklch(0.705 0.190 45);   /* #f97316 */
  --color-fail: oklch(0.640 0.200 25);   /* #f25555 */
  --color-skip: oklch(0.520 0.025 55);   /* #75685e */

  /* NEW — night-side materials */
  --color-grid:        oklch(0.770 0.090 65 / 0.07);  /* graticule hairline            */
  --color-grid-major:  oklch(0.770 0.090 65 / 0.14);  /* every fourth line             */
  --color-reticle:     oklch(0.770 0.090 65 / 0.40);
  --color-glass:       oklch(0.220 0.018 55 / 0.62);
  --color-glass-edge:  oklch(0.930 0.015 65 / 0.08);
  --color-glass-hi:    oklch(1 0 0 / 0.04);            /* inset top highlight           */
  --color-glow:        oklch(0.600 0.110 45 / 0.22);   /* atmospheric radial            */
  --color-scrim:       oklch(0.150 0.012 60 / 0.72);   /* over imagery behind popovers  */

  /* NEW — non-colour tokens */
  --blur-glass: 16px;
  --shadow-float: 0 1px 0 var(--color-glass-hi) inset, 0 12px 40px -12px oklch(0 0 0 / 0.6);
  --shadow-lift:  0 8px 24px -10px oklch(0 0 0 / 0.55);
  --radius-hud: 0.875rem;
  --grid-module: 48px;
  --ease-out-quint: cubic-bezier(0.16, 1, 0.3, 1);
  --ease-in-out-soft: cubic-bezier(0.4, 0, 0.2, 1);
  --dur-feedback: 120ms;   /* hover, press                      */
  --dur-state:    220ms;   /* toggles, chips, tab changes       */
  --dur-overlay:  300ms;   /* popovers, modals, view transitions */
  --dur-focal:    720ms;   /* the landing hero, once             */
  --font-display: 'Inter Variable', 'Inter Fallback', ui-sans-serif, system-ui, sans-serif;
}
```

**Light theme** (`:root:not(.dark)`, applied in `@layer base`, same property names). Every pair below
was chosen to meet the same contract as the dark set and *must* be added to `contrast.test.ts`:

| Token | Light value | Contract |
|---|---|---|
| `--color-bg-main` | `oklch(0.975 0.010 80)` warm paper | — |
| `--color-surface-card` | `oklch(0.995 0.004 80)` | — |
| `--color-surface-elevated` | `oklch(1 0 0)` | — |
| `--color-line` | `oklch(0.880 0.014 70)` | 1.4:1 vs card (non-text, decorative) |
| `--color-line-soft` | `oklch(0.930 0.010 75)` | — |
| `--color-text-hi` | `oklch(0.220 0.020 50)` | ≥ 14:1 on paper |
| `--color-text-lo` | `oklch(0.470 0.030 50)` | ≥ 6.0:1 paper · ≥ 6.5:1 card |
| `--color-accent-warm` | `oklch(0.560 0.110 40)` | 4.6:1 on paper (edges + ≥18px text only) |
| `--color-accent-warm-strong` | `oklch(0.500 0.120 38)` | white on it ≥ 5.5:1 |
| `--color-accent-warm-text` | `oklch(0.450 0.110 40)` | ≥ 7:1 on paper — the light-mode small-text voice |
| `--color-surface-sand` / `--color-evidence` | `oklch(0.850 0.080 70)` | `--color-on-evidence` `oklch(0.220 0.020 50)` ≥ 9:1 |
| `--color-ok-text` | `oklch(0.480 0.150 150)` | ≥ 4.5:1 on `ok/12` over paper |
| `--color-warn-text` | `oklch(0.520 0.160 45)` | ≥ 4.5:1 on `warn/12` over paper |
| `--color-fail` | `oklch(0.520 0.200 25)` | ≥ 4.5:1 on paper |
| `--color-glass` | `oklch(0.995 0.004 80 / 0.70)` | — |
| `--color-glass-edge` | `oklch(0.220 0.020 50 / 0.10)` | — |
| `--color-grid` / `-major` | `oklch(0.560 0.110 40 / 0.07)` / `/ 0.14` | — |
| `--color-scrim` | `oklch(0.975 0.010 80 / 0.72)` | — |

`color-scheme` follows the class: `:root { color-scheme: light } .dark { color-scheme: dark }`.
`::selection` and the sidebar focus-ring override stay as they are (they reference tokens).

### 0.5 Type scale (editorial additions)

The console's `.t-page / .t-panel / .t-eyebrow / .t-meta / .t-mono` stay. Additions:

| Class | Face | Size / leading / tracking / weight | Use |
|---|---|---|---|
| `.t-display` | **Odida** (`--font-odida`) | `clamp(3rem, 6.8vw, 6.25rem)` / 0.94 / −0.02em / 400, `text-wrap: balance`, `font-synthesis: none` | landing `h1` only |
| `.t-display-sub` | **Odida** | `clamp(1.25rem, 1.9vw, 1.625rem)` / 1.3 / 0 / 400, `max-width: 30ch` | landing standfirst only |
| `.t-section` | Inter | `clamp(1.75rem, 3vw, 2.5rem)` / 1.08 / −0.025em / 600 | landing section titles, gallery header |
| `.t-lede` | Inter | 1rem / 1.55 / 0 / 400, `max-width: 60ch` | section intros |
| `.t-coord` | Geist Mono | 0.6875rem / 1 / +0.08em / 500, `tabular-nums` | coordinates, tile ids, trace ids, HUD readouts |
| `.t-quote` | Inter | 0.8125rem / 1.5 / 0 / 400, `font-style: normal` (no true italic is loaded — do not fake one) | daily quote |

**Odida token.** `theme.css` gains the `@font-face` for `Odida` (+ a metric-matched `Odida Fallback`)
and the `@theme` entry `--font-odida: 'Odida', 'Odida Fallback', 'Inter Variable', …`, which yields the
`font-odida` utility. The `@font-face` block, the preload rule and the measured fallback metrics are
specified in `landing_page_blueprint.md` §3. **Rule:** `--font-odida` is referenced by exactly two
selectors (`.t-display`, `.t-display-sub`) and the `font-odida` utility is banned in `src/components`
and `src/pages/!(landing)` — enforced by the source test in §6.5.

Pairing rule: Odida carries the landing voice, Inter carries everything else, Geist Mono carries
*measurement*. Any number that came from the machine (coordinates, GSD, pct, ms) is mono; any number
in prose is Inter tabular — the same rule the console already follows with `[data-metric]`.

### 0.6 Contrast test extension

`src/styles/__tests__/contrast.test.ts` parses `#hex` only. Extend it to parse `oklch(L C H[/ A])`
via a 30-line OKLab→linear-sRGB conversion (Björn Ottosson's published matrices), gamut-clip, then
reuse the existing `luminance`/`contrast`/`over` helpers. Add a second `describe` block that reads
the `:root:not(.dark)` overrides and asserts the light pairs in §0.4. Tokens carrying `/ alpha` are
tested via `over(ground)` exactly as `bg-ok/12` is today.

### 0.7 Routing (no React Router — still)

`App.tsx` deliberately has no router. The landing page needs a URL a judge can type, and the sections
deserve back-button behaviour. Add `src/shell/router.ts`: a 40-line `history` sync, not a library.

```ts
// src/shell/router.ts
export const PATHS: Record<Section, string> = {
  home: '/', explore: '/explore', datasets: '/datasets', tools: '/tools',
  usecases: '/use-cases', maps: '/maps', saved: '/saved', projects: '/projects', history: '/history',
}
export function sectionFromPath(path: string): Section        // unknown → 'notFound' (§6.4)
export function bindRouter(): () => void                        // pushState on setSection; popstate → setSection
```

`useUiStore.setSection` calls `history.pushState` (guarded by `typeof window`); `bindRouter()` runs
once in `main.tsx`. The initial `section` becomes `sectionFromPath(location.pathname)`, so `/` opens
the landing page and `/explore` opens the console and any other path opens the 404 page (§6.4). `Section` gains
`'notFound'`. `nginx.conf` already has `try_files $uri $uri/ /index.html`.

---

## 1. The Landing Page — `/` (`section === 'home'`)

**Extracted.** The landing page — hero, Odida display typography, the interactive `three` /
`@react-three/fiber` globe, the pipeline story, capabilities showcase, motion sequence and its own
Suspense/chunk map — now lives in [`landing_page_blueprint.md`](landing_page_blueprint.md). It
depends on this document for tokens (§0.4), the router (§0.7), shared primitives (§5.1) and the
production checklist (§6); it restates none of them.

What this document still owns about `/`:

- `App.tsx` renders `<Landing/>` via `React.lazy` **outside** `AppShell` when `section === 'home'`
  (no sidebar, no header, no thread column).
- The sidebar wordmark becomes a `<button>` that calls `setSection('home')`, and its `<h1>` becomes
  a `<p>` — see §6.2.
- `Landing` is the only surface allowed to use `--font-odida`, `.t-display` and `.t-display-sub`.
- Its dependencies (`three`, `@react-three/fiber`, `@react-three/drei`) join the `globe` manual
  chunk defined in §6.5 and must never be imported by any file outside `src/pages/landing/`.

---

## 2. The four sidebar sections

Shared: `Sidebar.tsx` drops the `LIVE` set; every section routes to itself. `App.tsx` gains four
lazy pages. Each page owns a `SectionHeader` (`src/components/ui/SectionHeader.tsx`, new): `.t-page`
title, `.t-meta` count/status on the right, optional action slot, 1px `rule` below. Page padding
follows `<main>` (`px-4 py-5 wide:px-7 wide:py-6`).

### 2.1 Use Cases — `/use-cases`

**Job.** A presenter or a first-time user picks a *real, runnable* investigation and is in the
console with imagery loaded and a question in the box in one click.

**Files**

| Path | New / Modified |
|---|---|
| `src/pages/UseCases.tsx` | new |
| `src/pages/usecases/UseCaseCard.tsx` | new |
| `src/pages/usecases/catalogue.ts` | new — the curated list, typed |
| `src/pages/usecases/useLoadSample.ts` | new — fetch → `File[]` → `selectFiles` → `proposeQuestion` → `setSection('explore')` |
| `public/samples/<slug>/{manifest.json, pre.tif, post.tif, thumb.webp}` | **assets the team supplies** |
| `src/state/ui.ts` | modified — `loadingSample: string \| null` |

**Data model**

```ts
export interface UseCase {
  slug: string
  title: string                       // "Urban sprawl, Bengaluru"
  region: string                      // "Karnataka, IN"
  centroid: [lat: number, lon: number]
  years: [string, string] | [string]  // ["2019", "2024"]
  sensor: 'Sentinel-2' | 'Sentinel-1' | 'VHR optical' | 'Sentinel-2 + Sentinel-1'
  pairType: 'SINGLE' | 'BI_TEMPORAL' | 'CROSS_MODAL'
  task: TaskType                      // from '@/api/types'
  question: string                    // what lands in the composer
  tools: string[]                     // registry names shown as chips
  files: string[]                     // relative to /samples/<slug>/
  thumb: string
  sizeMb: number                      // shown before download; honest about the wait
}
```

**Curated catalogue (8; each must map to a real `TaskType` and real supplied rasters)**

| Title | Sensor · pair | Task | Question |
|---|---|---|---|
| Urban sprawl, Bengaluru | Sentinel-2 · bi-temporal 2019/2024 | `CHANGE_VQA` | "How much of the scene transitioned to built-up between the two dates?" |
| Brahmaputra flood delineation | Sentinel-1 VV/VH · single | `SEGMENTATION` | "Outline the inundated area and estimate its extent." |
| Airbase runway — aircraft grounding | VHR optical · single | `COUNT` | "Count the aircraft on the apron and locate each one." |
| Sundarbans mangrove NDVI | Sentinel-2 · bi-temporal | `CHANGE_MAP` | "Map where vegetation vigour dropped between the dates." |
| Chennai reservoir water extent | Sentinel-2 · bi-temporal | `CHANGE_VQA` | "Did the reservoir's surface water shrink or grow?" |
| Optical vs SAR — cloud-covered Kerala | Sentinel-2 + Sentinel-1 · cross-modal | `CROSS_MODAL_COMPARE` | "Where do optical and SAR disagree about built-up area?" |
| Land-cover, Rann of Kutch | Sentinel-2 · single | `SCENE_CLASSIFY` | "Classify the land cover in this scene." |
| Port expansion, Mundra | VHR optical · bi-temporal | `CHANGE_CAPTION` | "Summarise what changed at the port." |

Any row without supplied rasters ships **hidden**, not with a placeholder image.

**Layout.** `SectionHeader` (`Use cases` · `8 investigations`) with a filter row of `chip`s by
`task` family (`Change · Cross-modal · Grounding · Classify`, multi-select, `aria-pressed`). Grid:
`grid-cols-1 wide:grid-cols-2 desk:grid-cols-3`, gap 16px. Cards are equal-height.

**Card anatomy (`UseCaseCard`, `.card-flush`, 4:3 thumb):**

```
┌───────────────────────────────┐
│ thumb.webp (object-cover)     │  ← graticule overlay at 6 %, reticle at centroid
│  ┌ 12°58′N 77°35′E ┐          │  ← .t-coord glass pill, top-left
│                        [S-2]  │  ← sensor badge, top-right, glass
├───────────────────────────────┤
│ Urban sprawl, Bengaluru       │  .t-panel
│ Karnataka · 2019 → 2024       │  .t-meta
│ [CHANGE_VQA] [siamese_cd] [+2]│  chips (mono); +N reveals on hover/focus
│                               │
│ ▶ Load into canvas    142 MB  │  footer: .btn-ghost sm · size in .t-coord
└───────────────────────────────┘
```

**States**

| State | Treatment |
|---|---|
| rest | `.card-flush`; thumb saturate 0.85 |
| hover / focus-within | `translateY(−3px)`, `box-shadow: var(--shadow-lift)`, border → `accent-warm/40`, thumb saturate 1, reticle scales 1 → 0.9 (the lock), 220 ms `--ease-out-quint`; `@media (hover: hover)` only for the lift |
| loading (`loadingSample === slug`) | footer button becomes a determinate bar fed by `fetch` progress (`ReadableStream` byte count vs `sizeMb`); label `Fetching 42 / 142 MB`; other cards `pointer-events: none; opacity .6` |
| loaded | navigates; the console's `Dropzone` is already replaced by the manifest list because `selectFiles` ran |
| fetch error | inline `text-fail` line under the footer: `Sample unavailable (HTTP 404).` — button re-enabled |
| filter empty | `No use cases match those filters.` + `Clear filters` ghost button; graticule empty tile |

**Load semantics (`useLoadSample`).** `fetch` each file → `new File([blob], name, { type })` →
`useUiStore.getState().selectFiles(files)` → `useFocusStore.getState().proposeQuestion(question)` →
`setSection('explore')`. The composer is focused with the question in place; pre-flight is already
running when the stage appears. Playing a use case is therefore *exactly* the demo flow, not a
special path.

**Polish.** Card image `<img>` uses `decoding="async"` and a 10px blurred `background-image` of a
32px LQIP embedded in `catalogue.ts` (data URI ≤ 600 B each) so the grid never pops. Keyboard: cards
are `<article>` with one focusable `<button>` (the footer); `Enter` on a card also loads.

### 2.2 Maps — `/maps`

**Job.** See the loaded scene *in the world*: where it is, at what scale, with each rendered view
switchable, and A/B swipe over a real map frame. The map is a **HUD around the console's own
evidence**, not a GIS.

**Truth constraint.** The backend serves no tiles and the lab may have no internet. So the Maps page
has two modes, declared in the HUD: **Scene** (always available — the rendered view artifacts of the
current run placed by the manifest's bounds) and **Basemap** (optional — an online raster tile source
behind the scene, off by default, toggled by the user, remembered in `localStorage`). Cartosat is a
*label the manifest may carry*, not a tile layer we ship; the layer switcher lists whatever sensors
the loaded manifest(s) declare and shows the Sentinel-2 / Cartosat / SAR names only when true.

**Dependency decision.** Add `maplibre-gl` (≈ 250 KB gz, lazy-loaded on this page only). It gives
projection, graticule-correct pan/zoom, image sources placed by corner coordinates, and a swipe
compare via two synchronised map instances — all of which `react-zoom-pan-pinch` cannot do. When
the manifest has no georeference (`manifest.crs === null`), the page falls back to the existing
`ImageViewer` full-bleed with the HUD's pixel-space readouts and says so in the HUD.

**Files**

| Path | New / Modified |
|---|---|
| `src/pages/Maps.tsx` | new — mode branch: georeferenced → `MapStage`, else `PixelStage` |
| `src/pages/maps/MapStage.tsx` | new — `maplibre-gl` wrapper, lazy |
| `src/pages/maps/PixelStage.tsx` | new — `ImageViewer` in full-bleed |
| `src/pages/maps/hud/LayerSwitcher.tsx` | new |
| `src/pages/maps/hud/CoordinateLocator.tsx` | new |
| `src/pages/maps/hud/ZoomHud.tsx` | new |
| `src/pages/maps/hud/SwipeHandle.tsx` | new |
| `src/pages/maps/hud/ScaleBar.tsx` | new |
| `src/state/map.ts` | new — `useMapStore` |
| `src/evidence/georef.ts` | new — bounds/affine helpers from `InputManifest` |

**State**

```ts
interface MapState {
  layerKey: string | null            // ViewGroup.key shown as A (reuses focus.activeViewKey semantics)
  compareKey: string | null          // ViewGroup.key shown as B; null = no split
  split: number                      // 0–100, shares nudge semantics with focus.swipe
  basemap: 'none' | 'satellite'      // persisted 'satquery.maps.basemap'
  view: { center: [lon: number, lat: number]; zoom: number; bearing: 0 }
  cursor: { lon: number; lat: number; x: number; y: number } | null   // pixel + world
  locatorOpen: boolean
  opacity: number                    // scene over basemap, 0–1, default 1
  setLayer; setCompare; setSplit; setBasemap; setView; setCursor; toggleLocator; setOpacity
}
```

**Layout — full bleed.** `Maps.tsx` renders `<main>` without padding (`AppShell` accepts a
`bleed` prop, or the page sets `-m-4 wide:-m-7` on its root). The map fills the centre column; the
thread column stays (the judge can still ask). HUD elements float with 16px inset:

```
┌ centre column ─────────────────────────────────────────────────────┐
│ ┌ glass ───────────────┐                          ┌ glass ───────┐ │
│ │ ▣ Sentinel-2 TC   ▾  │                          │  +  │  −  │⌂ │ │  layer switcher · zoom HUD
│ │ ▢ NDVI               │                          └──────────────┘ │
│ │ ▢ SAR VV/VH          │                                           │
│ │ ▢ CHANGE             │              ╫  ← swipe handle            │
│ │ ── basemap  ○ off    │                                           │
│ └──────────────────────┘                                           │
│                                                                    │
│                       ┼ cursor reticle                             │
│                                                                    │
│ ┌ glass ─────────────────────────────┐   ┌ glass ───────────────┐  │
│ │ 12°58′41″N 77°35′22″E · z 12.4     │   │ ━━━━━ 500 m  · 10 m/px│  │  locator · scale bar
│ └────────────────────────────────────┘   └──────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

- **LayerSwitcher** (top-left, glass, `role="radiogroup"` for A, a second `Compare with…` select
  for B). Rows: colour swatch (TC = terracotta, NDVI = ok, SAR = text-lo, CHANGE = warn — the
  same legend colours `thread/legend.ts` already assigns), name, sensor tag in `.t-coord`. The
  basemap switch sits under a `rule` with the `○ off` state default and a one-line
  `needs internet` note in `.t-meta`.
- **ZoomHud** (top-right): `+` `−` `⌂` (fit to scene). 36px glass buttons, `aria-keyshortcuts`.
- **SwipeHandle**: when `compareKey` is set, a vertical 2px sand line with a 28px glass grip at the
  vertical centre, draggable, `role="slider"` (`[`/`]` nudge it through `useHotkeys` — the existing
  `OWNS_ARROWS` rule already protects it). Labels `A` / `B` in `.t-coord` pinned to the top corners
  of each half.
- **CoordinateLocator** (bottom-left): live cursor lat/lon in DMS + zoom, mono. Click → expands to an
  input (`12.978, 77.589` or DMS) with `Go` → `flyTo` 600 ms. Copy button copies decimal degrees.
  When `crs === null`: shows `px 1024, 768 · no georeference in manifest`.
- **ScaleBar** (bottom-right): computed from the map's `metersPerPixel` at the centre latitude;
  the GSD from the manifest sits beside it as `10 m/px` so a judge sees the sensor resolution and
  the screen scale as two different numbers.
- **Cursor reticle**: the `.reticle` symbol follows the pointer (`transform: translate3d`, no
  layout), hidden on touch.

**States**

| State | Treatment |
|---|---|
| no run loaded | full-bleed graticule with a centred glass card: `Nothing on the map yet` · `Load a use case or drop imagery` · two ghost buttons → `/use-cases`, `/explore` |
| run loaded, georeferenced | as above |
| run loaded, no CRS | `PixelStage`; HUD locator in pixel mode; a `.t-meta` amber line in the layer switcher: `Manifest carries no CRS — showing pixel space` |
| basemap on, offline | tiles fail → maplibre `error` event → basemap auto-switches to `none`, toast (§4.4 notification): `Basemap unreachable — showing scene only` |
| loading scene images | the scene's footprint is drawn as a dashed sand polygon immediately; the raster fades in over it (220 ms) |

**Polish.** Map `flyTo` uses `essential: true` only for user-invoked moves; `prefers-reduced-motion`
→ `jumpTo`. All HUD panes are `contain: paint` and `backdrop-filter` only within their own box.
Pinch on touch is native to maplibre; the HUD collapses to a single bottom glass bar on `wide` and
below (layer name · locator · `⋯` opens the full switcher as a sheet).

### 2.3 Saved — `/saved`

**Job.** Every run this browser has kept: dense, scannable, re-openable, exportable. "Saved" is a
*flat list of runs*; "Projects" (§2.4) groups them.

**Truth constraint.** There is no server-side save. Runs are persisted **locally** in IndexedDB
(`idb-keyval`, 600 B) keyed by `traceId`, holding the `RunSummary` below plus a downscaled thumbnail
(`canvas.toBlob` of the primary view at 320px, stored as Blob). The full trace is re-fetched from
`GET /v1/traces/{trace_id}` on open; if the server no longer has it, the card says so and offers
the cached summary only. The UI says `Saved on this device` in the header — never "cloud".

**Files**

| Path | New / Modified |
|---|---|
| `src/pages/Saved.tsx` | new |
| `src/pages/saved/RunRow.tsx` | new — list row |
| `src/pages/saved/RunCard.tsx` | new — card with bbox preview |
| `src/pages/saved/BboxThumb.tsx` | new — thumb + SVG boxes from `thread/boxes.ts` |
| `src/pages/saved/ExportMenu.tsx` | new |
| `src/export/geojson.ts` | new |
| `src/export/report.tsx` | new — printable report route `/report/:traceId` |
| `src/state/library.ts` | new — `useLibraryStore` (persisted to IndexedDB, not localStorage) |
| `src/components/thread/GroundedAnswer.tsx` | modified — `Save` action calls `library.save(...)` |
| `src/components/thread/PreviousQueries.tsx` | modified — saved runs show a bookmark glyph |

**Data model**

```ts
export interface SavedRun {
  traceId: string
  query: string
  taskType: TaskType
  pairType: 'SINGLE' | 'BI_TEMPORAL' | 'CROSS_MODAL'
  sensors: string[]
  savedAt: number
  ranAt: number
  outcome: RunOutcome
  confidence: number | null
  headline: { label: string; value: string } | null   // first KPI, already formatted
  boxes: Box[]                                        // from thread/boxes.ts, pixel space
  bounds: [w: number, s: number, e: number, n: number] | null
  thumb: Blob | null
  projectId: string | null
  tags: string[]
}
interface LibraryState {
  runs: Record<string, SavedRun>
  hydrated: boolean
  view: 'cards' | 'rows'                 // persisted 'satquery.saved.view'
  sort: 'savedAt' | 'ranAt' | 'confidence'
  filter: { task: TaskType[] ; projectId: string | null; text: string }
  selected: Set<string>                  // bulk actions
  save(run: SavedRun): Promise<void>; remove(id): Promise<void>; assign(ids, projectId): Promise<void>
  setView; setSort; setFilter; toggleSelect; clearSelection
}
```

**Layout.** `SectionHeader` (`Saved` · `14 runs · on this device`) with a right-side control group:
search input (`/` is taken; use `⌘F`-free plain input), sort select, `cards | rows` segmented
toggle. Bulk bar slides up from the bottom of `<main>` when `selected.size > 0`: `3 selected · Add
to project ▾ · Export GeoJSON · Delete`.

**Rows view (dense, default on `wide` and below).** A `<table>`-semantics grid, 40px rows,
sticky header, columns: `☐ · thumb 48×32 · query (truncate) · task chip · sensors · confidence
(mono, 2 dp, colour by `≥0.8 ok / ≥0.6 warn / else fail`) · saved (relative) · ⋯`. Row hover
`bg-sidebar-hi`; `Enter` opens.

**Cards view (`desk` default).** `grid-cols-2 desk:grid-cols-3`, `.card-flush`:

```
┌────────────────────────────────┐
│ BboxThumb 16:10                │  thumb + boxes stroked --color-bbox, labels --color-bbox-label
│  [CHANGE_VQA]     conf 0.86    │  glass chips over the image
├────────────────────────────────┤
│ "How much of the scene…"       │  .t-panel, 2-line clamp
│ 7.4 % built-up expansion       │  headline KPI, mono value
│ S-2 · bi-temporal · 2d ago     │  .t-meta
│ [Project: Bengaluru ▾] [⤓] [⋯] │  footer
└────────────────────────────────┘
```

`BboxThumb` draws the `Box[]` as SVG over the thumb with `vector-effect: non-scaling-stroke`; when
`boxes.length === 0` the reticle sits at the centre instead, so grounding runs and change runs read
differently at a glance.

**Export (`ExportMenu`, Radix Popover, glass)**

| Item | Behaviour | Honesty rule |
|---|---|---|
| `GeoJSON` | `FeatureCollection` of boxes; if `bounds` present, pixel → lon/lat by affine; else `properties.crs: "pixel"` and a top-level `"note"` | never emit lon/lat from a run without bounds |
| `PDF report` | opens `/report/:traceId` in a new tab; `@media print` stylesheet; user prints to PDF | no PDF library; the report re-renders `GroundedAnswer`, KPI cards, the static DAG SVG, and the audit trace table |
| `Copy trace id` | clipboard | — |
| `Open in Maps` | `setSection('maps')` with this run active | disabled without bounds, with a tooltip saying why |

**Empty state (`/impeccable delight`).** Full-height centred composition on a graticule tile with a
reticle that *breathes* (opacity only):

> **Nothing saved yet.**
> Runs you keep will appear here — with their boxes, numbers and trace.
> `[Run a use case]` `[Open the console]`

The graticule under the empty state is the only place on the console pages where it appears at the
major (14 %) alpha. On first save, the page transitions: the empty-state card shrinks to the new
card's position (FLIP with `view-transition-name: run-<id>` where supported, else `sq-arrive`).

**States:** `hydrated === false` → 6 skeleton cards (`.card` with three `bg-line-soft` bars, 1.6 s
shimmer, reduced-motion → static); trace re-fetch 404 → card gets an amber `.t-meta` line `Trace no
longer on the server — showing the saved summary`; delete → row/card collapses 220 ms, `Undo` toast
for 6 s.

### 2.4 Projects — `/projects`

**Job.** Group saved runs into investigations ("Bengaluru sprawl 2019–2024") and see them as one
thing: a footprint, a count, a status.

**Files**

| Path | New / Modified |
|---|---|
| `src/pages/Projects.tsx` | new — index |
| `src/pages/projects/ProjectCard.tsx` | new |
| `src/pages/projects/ProjectDetail.tsx` | new — `/projects/:id`, reuses `Saved` list filtered |
| `src/pages/projects/NewProjectDialog.tsx` | new — Radix Dialog (existing `ui/dialog.tsx`) |
| `src/state/library.ts` | modified — projects live beside runs |

**Data model**

```ts
export interface Project {
  id: string; name: string; description: string
  createdAt: number; updatedAt: number
  colour: 'terracotta' | 'sand' | 'ok' | 'warn'   // tag hue, from the status vocabulary only
  aoi: [w, s, e, n] | null                          // union of member bounds, recomputed on assign
  pinned: boolean
}
// LibraryState gains:
projects: Record<string, Project>
createProject(input): Promise<Project>; renameProject; deleteProject(id, { keepRuns: boolean })
```

**Index layout.** `SectionHeader` (`Projects` · `3 projects · 14 runs`) with `+ New project`
primary. Grid `wide:grid-cols-2 desk:grid-cols-3`:

```
┌────────────────────────────────┐
│ ▤ AOI preview                  │  graticule + union footprint as a dashed sand polygon,
│    ┌─────┐                     │  member thumbs as 4 stacked 40px tiles, top-right
│    │     │        ▣▣▣▣ +3      │
│    └─────┘                     │
├────────────────────────────────┤
│ Bengaluru sprawl 2019–2024     │  .t-panel
│ 7 runs · 5 succeeded · 1 failed│  .t-meta with StatusDots
│ updated 2h ago                 │
│ [Open]  [Export ▾]  [⋯]        │
└────────────────────────────────┘
```

Card hover = the same lift as Use Cases (one lift, one shadow, everywhere). Pinned projects sort
first with a sand bookmark glyph.

**Detail.** Breadcrumb `Projects / Bengaluru sprawl` in the `SectionHeader` title slot; description
inline-editable (`contenteditable=plaintext-only`, saves on blur); tabs `Runs · Map · Report`:
`Runs` is the Saved rows view pre-filtered; `Map` is `MapStage` with all member footprints; `Report`
is the print route for the whole project (each run a page, project summary first).

**Export.** `GeoJSON` merges member boxes with `properties.traceId`; `PDF report` opens
`/report/project/:id`.

**Empty state.** The delight moment of the whole console, because it is the first thing a new user
sees under a promising word:

```
        ┼            ┼            ┼          ← three reticles on graticule intersections,
                                                 breathing out of phase (0 / 1.3 / 2.6 s)
             No projects yet.
   A project is a folder for runs that belong
   to the same question — one place, one
   footprint, one report.

        [ + Create your first project ]
        Saved runs can be filed later.
```

On `Create`, the dialog opens with the name field focused and a placeholder that is a real example
(`Brahmaputra monsoon 2025`), not `Project name`.

**Delete** requires typing nothing — it is undoable for 8 s via toast; `keepRuns` is the default
and the dialog copy says `Runs stay in Saved`.

---

## 3. Account popover

**Files**

| Path | New / Modified |
|---|---|
| `src/components/shell/AccountPopover.tsx` | new |
| `src/components/shell/StorageQuota.tsx` | new — `navigator.storage.estimate()` |
| `src/components/ui/popover.tsx` | new — Radix Popover wrapper with the glass treatment + `sq-pop-in` |
| `src/components/shell/Sidebar.tsx` | modified — user card becomes the trigger; the settings gear becomes the same trigger's secondary icon; the shortcuts checkbox moves into the popover |
| `src/state/account.ts` | new — `useAccountStore` (display name, role, avatar initial; persisted; single local identity) |
| `package.json` | modified — `@radix-ui/react-popover` |

**Truth.** There is no auth. The popover is honest about it: `Sign out` is present but performs
`library.clear()` + `account.reset()` with a confirm, and is labelled `Sign out & clear this device`.

**Anatomy (width 280px, anchored `side="top" align="start"` to the user card, 8px offset)**

```
┌──────────────────────────────────────┐
│  (A)  Aksh                           │  avatar 32px sand; name .t-panel
│       Researcher · Student   [edit]  │  role in .t-meta; edit → inline rename
│  ──────────────────────────────────  │  rule
│  Storage on this device              │  .t-eyebrow
│  ▮▮▮▮▮▮▮▯▯▯▯  212 MB of 2.1 GB       │  quota bar: bg-line track, accent-warm-strong fill,
│                                      │  mono numbers; amber at ≥ 80 %, red at ≥ 95 %
│  ──────────────────────────────────  │
│  ⚙  Settings                    ⌘,   │  each row: 36px, icon 16px text-lo, label, kbd right
│  ▦  Workspaces & projects       G P  │  → /projects
│  ⌨  Single-key shortcuts   [ on ]    │  switch (Radix-less: <button role=switch>), kbd ?
│  ☾  Theme        [ ☾ dark | ☀ light ]│  segmented, §4.3
│  ──────────────────────────────────  │
│  ⇥  Sign out & clear this device     │  text-fail on hover only
└──────────────────────────────────────┘
```

**Visual spec.** `.glass` + `--shadow-float` + `--radius-hud`; a 1px `--color-glass-edge` border with
a *border glow*: `outline: 1px solid oklch(0.600 0.110 45 / 0.18); outline-offset: 0` fading in 220
ms after open — the terracotta edge says "this floats over the console". Header strip carries the
graticule at 6 % (24px module) so the popover shares the ground-station material. Focus rings:
`:focus-visible` → 2px `--color-accent-warm` ring, 2px offset, inside the popover the offset is 0 and
the ring is inset so it never clips at the glass edge.

**Motion.** `sq-pop-in`: opacity 0 → 1, `translateY(4px) scale(0.98)` → identity, 200 ms
`--ease-out-quint`, `transform-origin: bottom left`. Exit 120 ms opacity only. Reduced motion →
opacity only.

**Keyboard.** Trigger is a `<button aria-haspopup="dialog" aria-expanded>`. Inside: Radix roving
focus; `↑↓` move, `Enter` activates, `Esc` closes and returns focus. The kbd badges are live: `⌘,`
opens Settings (a stub `SettingsDialog` listing theme, shortcuts, mock mode, `Clear device data`),
`G P` is a *sequence* chord handled in `useHotkeys` (gated by `shortcutsEnabled`, 600 ms window).

**Storage quota (`StorageQuota`).** `navigator.storage.estimate()` → `{ usage, quota }`. Not
supported → the row reads `Storage estimate unavailable in this browser`, no bar. This is a real
number and the only quota the product can honestly show.

---

## 4. Detail fixes & micro-architecture

### 4.1 Daily space quote — `useDailyQuote`

**Files:** `src/shell/quotes.ts` (data), `src/shell/useDailyQuote.ts` (hook),
`src/components/shell/DailyQuote.tsx` (replaces the mission line in `Sidebar.tsx`).

```ts
export interface Quote { text: string; by: string; role?: string; attributed?: true }
export const QUOTES: readonly Quote[]   // length === 31, asserted by a test
export function useDailyQuote(now = new Date()): Quote {
  return QUOTES[(now.getDate() - 1) % QUOTES.length]!   // 1..31 → 0..30, deterministic
}
```

Rendered in the sidebar's mission slot: `.t-quote` text, then `— Vikram Sarabhai` in `.t-meta`, the
existing 32px terracotta bar under it. Hidden on phones as the mission line is today. A `title`
tooltip on the bar shows `Quote 12 of 31 · changes daily`. No animation on mount (the sidebar is
Operate); the text cross-fades 220 ms when the date rolls past midnight in an open session (a
`setTimeout` to the next local midnight, re-armed on `visibilitychange`).

**The 31 (attribution to be verified line-by-line before shipping; `attributed: true` marks lines
whose exact wording or source is disputed and the UI renders `— attributed to …`):**

| # | Quote | By |
|---|---|---|
| 1 | "There are some who question the relevance of space activities in a developing nation. To us, there is no ambiguity of purpose." | Vikram Sarabhai |
| 2 | "We must be second to none in the application of advanced technologies to the real problems of man and society." | Vikram Sarabhai |
| 3 | "Look again at that dot. That's here. That's home. That's us." | Carl Sagan |
| 4 | "Somewhere, something incredible is waiting to be known." | Carl Sagan (attributed) |
| 5 | "The cosmos is within us. We are made of star-stuff." | Carl Sagan |
| 6 | "Saare jahan se achha." — asked how India looked from orbit | Rakesh Sharma |
| 7 | "Dream is not that which you see while sleeping; it is something that does not let you sleep." | A. P. J. Abdul Kalam (attributed) |
| 8 | "The path from dreams to success does exist. May you have the vision to find it, the courage to get on to it, and the perseverance to follow it." | Kalpana Chawla |
| 9 | "Earth is the cradle of humanity, but one cannot live in a cradle forever." | Konstantin Tsiolkovsky |
| 10 | "We came all this way to explore the Moon, and the most important thing is that we discovered the Earth." | William Anders, Apollo 8 |
| 11 | "The stars don't look bigger, but they do look brighter." | Sally Ride |
| 12 | "Never be limited by other people's limited imaginations." | Mae Jemison |
| 13 | "Magnificent desolation." | Buzz Aldrin |
| 14 | "We do not realize what we have on Earth until we leave it." | Jim Lovell |
| 15 | "You develop an instant global consciousness, a people orientation, an intense dissatisfaction with the state of the world, and a compulsion to do something about it." | Edgar Mitchell |
| 16 | "Orbiting Earth in the spaceship, I saw how beautiful our planet is. People, let us preserve and increase this beauty, not destroy it!" | Yuri Gagarin |
| 17 | "That's one small step for a man, one giant leap for mankind." | Neil Armstrong |
| 18 | "Equipped with his five senses, man explores the universe around him and calls the adventure Science." | Edwin Hubble |
| 19 | "Any sufficiently advanced technology is indistinguishable from magic." | Arthur C. Clarke |
| 20 | "Look up at the stars and not down at your feet. Try to make sense of what you see, and wonder about what makes the universe exist." | Stephen Hawking |
| 21 | "The universe is under no obligation to make sense to you." | Neil deGrasse Tyson |
| 22 | "If I have seen further it is by standing on the shoulders of giants." | Isaac Newton |
| 23 | "Measure what is measurable, and make measurable what is not so." | Galileo Galilei (attributed) |
| 24 | "The Earth is a very small stage in a vast cosmic arena." | Carl Sagan |
| 25 | "It is science alone that can solve the problems of hunger and poverty, of insanitation and illiteracy…" | Jawaharlal Nehru |
| 26 | "Space is for everybody. It's not just for a few people in science or math, or for a select group of astronauts." | Christa McAuliffe |
| 27 | "Once you have tasted flight, you will forever walk the earth with your eyes turned skyward." | attributed to Leonardo da Vinci |
| 28 | "The Earth was small, light blue, and so touchingly alone, our home that must be defended like a holy relic." | Aleksei Leonov |
| 29 | "From out there on the Moon, international politics look so petty." | Edgar Mitchell |
| 30 | "Across the sea of space, the stars are other suns." | Carl Sagan |
| 31 | "Astronomy compels the soul to look upward, and leads us from this world to another." | Plato |

(A remote-sensing-specific line from an ISRO/SAC figure, if the team has a sourced one, should
replace #27, which is the weakest attribution.)

### 4.2 Keyboard shortcuts guide & global toggle

**Files:** `src/state/shortcuts.ts` (new, `useShortcutStore`), `src/shell/shortcuts.ts` (new —
the binding table, single source of truth), `src/components/shell/KeyboardShortcutsModal.tsx` (new),
`src/shell/useHotkeys.ts` (modified — reads the table and the new store), `src/state/ui.ts` (modified
— `shortcutsEnabled` and `setShortcutsEnabled` removed; `readShortcutPreference` moves),
`src/shell/__tests__/useHotkeys.test.tsx` (modified).

```ts
// src/state/shortcuts.ts
import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
interface ShortcutState {
  enabled: boolean            // WCAG 2.1.4 master switch for single-key bindings
  guideOpen: boolean
  setEnabled(v: boolean): void
  openGuide(): void; closeGuide(): void; toggleGuide(): void
}
export const useShortcutStore = create<ShortcutState>()(
  persist((set) => ({ enabled: true, guideOpen: false, /* … */ }),
    { name: 'satquery.shortcuts', storage: createJSONStorage(() => safeLocalStorage),
      partialize: (s) => ({ enabled: s.enabled }) }),
)
// safeLocalStorage: the try/catch wrapper from ui.ts, so a private window still boots
```

```ts
// src/shell/shortcuts.ts
export type Scope = 'global' | 'stage' | 'thread' | 'maps' | 'navigation'
export interface Binding {
  id: string; keys: string[]; label: string; scope: Scope
  chord: boolean               // true = works even when `enabled` is false (⌘K, ⌘,, ⌘/)
  when?: (state: AppSnapshot) => boolean
}
export const BINDINGS: readonly Binding[] = [
  { id: 'new-query', keys: ['⌘', 'K'], label: 'New query', scope: 'global', chord: true },
  { id: 'guide', keys: ['?'], label: 'Keyboard shortcuts', scope: 'global', chord: false },
  { id: 'guide-chord', keys: ['⌘', '/'], label: 'Keyboard shortcuts', scope: 'global', chord: true },
  { id: 'focus-composer', keys: ['/'], label: 'Ask', scope: 'thread', chord: false },
  { id: 'swipe-left', keys: ['['], label: 'Swipe left 5 %', scope: 'stage', chord: false },
  { id: 'swipe-right', keys: [']'], label: 'Swipe right 5 %', scope: 'stage', chord: false },
  { id: 'view-prev', keys: ['←'], label: 'Previous evidence view', scope: 'stage', chord: false },
  { id: 'view-next', keys: ['→'], label: 'Next evidence view', scope: 'stage', chord: false },
  { id: 'pipeline', keys: ['P'], label: 'Toggle processing pipeline', scope: 'stage', chord: false },
  { id: 'go-home', keys: ['G', 'H'], label: 'Go to landing', scope: 'navigation', chord: false },
  { id: 'go-explore', keys: ['G', 'E'], label: 'Go to Explore', scope: 'navigation', chord: false },
  { id: 'go-usecases', keys: ['G', 'U'], label: 'Go to Use cases', scope: 'navigation', chord: false },
  { id: 'go-maps', keys: ['G', 'M'], label: 'Go to Maps', scope: 'navigation', chord: false },
  { id: 'go-saved', keys: ['G', 'S'], label: 'Go to Saved', scope: 'navigation', chord: false },
  { id: 'go-projects', keys: ['G', 'P'], label: 'Go to Projects', scope: 'navigation', chord: false },
  { id: 'theme', keys: ['⌘', 'J'], label: 'Toggle theme', scope: 'global', chord: true },
  { id: 'settings', keys: ['⌘', ','], label: 'Settings', scope: 'global', chord: true },
]
```

`useHotkeys` becomes a dispatcher over `BINDINGS`: chords first (never gated), then `if
(!useShortcutStore.getState().enabled) return`, then single keys with the existing `isTyping` /
`OWNS_ARROWS` guards, plus a 600 ms `G`-prefix sequence state. The `SHORTCUTS` help strip export is
derived from `BINDINGS.filter(b => b.scope === 'stage')` so the strip and the modal cannot drift.

**Modal (`⌘K` command-palette aesthetic, but it is a guide, not a palette — it does not execute):**
Radix Dialog, 560px wide, centred, `.glass` on `--color-scrim`, `--shadow-float`, `sq-dialog-in`.
Header: a search input (`Filter shortcuts…`, filters rows by label/keys as you type — this is the
"palette" feel) with `Esc` kbd on the right. Body: grouped by `Scope` with `.t-eyebrow` group titles,
each row `label … [kbd][kbd]`; kbds use `.kbd` at 11px with the glass fill. Footer: the master
switch row — `Single-key shortcuts  [ on ]` with the WCAG sentence in `.t-meta`: `Turn off if you use
speech input or a tool that types for you. Chords (⌘…) always work.` When `enabled === false`, the
single-key rows render at 50 % opacity with a `disabled` tag; chord rows stay full. Opening with `?`
is itself a single-key shortcut, so `⌘/` exists as the always-on route in.

The sidebar checkbox is deleted from `Sidebar.tsx`; the switch lives in the account popover (§3)
and in the modal footer, both bound to the same store. The e2e/unit test that asserts the checkbox
moves to the popover.

### 4.3 Theme switcher

**Files:** `src/state/theme.ts` (new), `src/shell/applyTheme.ts` (new), `index.html` (modified —
an inline pre-paint script), `src/styles/theme.css` (modified — §0.4), `src/components/shell/
ThemeToggle.tsx` (new — used by the account popover and the landing nav).

```ts
type ThemePref = 'dark' | 'light' | 'system'
interface ThemeState {
  pref: ThemePref                      // persisted 'satquery.theme'
  resolved: 'dark' | 'light'           // pref + matchMedia
  setPref(p: ThemePref): void
  toggle(): void                       // dark ⇄ light (system → the opposite of resolved)
}
```

`applyTheme(resolved)`: `document.documentElement.classList.toggle('dark', resolved === 'dark')`
and `style.colorScheme = resolved`. A `matchMedia('(prefers-color-scheme: dark)')` listener updates
`resolved` while `pref === 'system'`. **No flash:** `index.html` gets a 6-line inline script before
the stylesheet that reads `localStorage['satquery.theme']` (try/catch) and sets the class before
first paint; the store hydrates from the same key and agrees with the DOM.

**Toggle control (`ThemeToggle`)** — a 3-segment `role="radiogroup"` (`☾ dark · ☀ light · ◐ system`)
in the popover; a single icon button on the landing nav. The switch is animated as a *sun/moon
mask*: the icon is one SVG circle with a `clip-path` circle that slides 220 ms `--ease-out-quint`;
reduced motion → instant. Theme change itself uses `document.startViewTransition` when available
(cross-fade 220 ms) — never a `transition: background-color` on `*`, which paints every node.

`color-scheme` on `:root` ensures native controls (the file input, selects, scrollbars) follow.
`theme.css` keeps `scrollbar-color` on tokens so it flips too.

### 4.4 Notification centre

**Files:** `src/state/notifications.ts` (new), `src/components/shell/NotificationPopover.tsx`
(new), `src/components/shell/RadarPulse.tsx` (new — the empty-state SVG), `src/components/shell/
HealthStrip.tsx` (modified — the bell becomes the trigger; the badge count sits on it),
`src/state/ui.ts` (modified — `settleRun` also calls `notifications.push`), `src/thread/useRun.ts`
(modified — on terminal events), `src/components/ui/Toast.tsx` (new — the same store's `transient`
lane, rendered bottom-centre of `<main>`).

```ts
type Kind = 'run' | 'health' | 'device' | 'library' | 'maps'
type Tone = 'ok' | 'warn' | 'fail' | 'info'
export interface Notification {
  id: string; kind: Kind; tone: Tone
  title: string; body?: string
  at: number; read: boolean
  action?: { label: string; run: () => void }   // "Open run", "Undo", "Retry basemap"
  transient?: boolean                            // toast-only, auto-dismiss 6 s, never listed
}
interface NotificationState {
  items: Notification[]                          // newest first, cap 50, session-only
  open: boolean
  unread: number                                 // derived, kept for the badge selector
  push(n: Omit<Notification, 'id' | 'at' | 'read'>): string
  markAllRead(); dismiss(id); clear(); setOpen(v)
}
```

**Producers (all real events — nothing synthetic):**

| Source | Notification |
|---|---|
| `settleRun(traceId, 'succeeded')` while `section !== 'explore'` | `Run finished · "<query…>"` · action `Open run` → `openRun` |
| `settleRun(traceId, 'failed')` | tone `fail`, `Run failed` + first `errors[0].message` |
| health poll transitions `ok → warn / fail` and back | `API degraded` / `API unreachable` / `System ready again` |
| `igpu_masked === false` on first health | tone `warn`, `Integrated GPU not masked — mask it before demoing` (pinned until read) |
| library save / delete / undo | transient toasts |
| basemap tile failure | transient + action `Retry` |

**Popover** (Radix Popover, glass, 320px, `side="bottom" align="end"`): header `Notifications` ·
`Mark all read` ghost; list rows 56px: `StatusDot` (existing) by tone, title `.t-panel` at 13px,
body `.t-meta` one-line clamp, relative time `.t-coord` right. Unread rows carry a 2px sand bar on
the leading edge (the same `nav-active` inset-shadow idiom). Row action button on hover/focus.
Footer: `Clear` when any item exists.

**Bell badge:** an 8px `--color-accent-warm` dot with a 2px `bg-surface-card` ring; `aria-label`
`Notifications, 2 unread`. A new unread *pulses once* (`sq-status-pulse` × 1, 600 ms) — not a loop.

**Empty state — "Zero unread alerts."** `RadarPulse`: a 96px SVG — the reticle at the centre, two
concentric circles at 30 % / 15 % alpha, and a *sweep*: a conic-gradient sector (`--color-reticle` →
transparent, 60°) rotating 4 s linear, masked to the outer circle. Under it: `.t-panel` `Zero unread
alerts.` and `.t-meta` `Run outcomes, device health and library changes will land here.` The
sweep runs only while the popover is open; reduced motion → the sector is static at 45° and the
outer ring breathes (opacity) instead. Hovering the radar does nothing — it is not a control.

### 4.5 Header cleanup — remove "Earth View"

**Files:** delete `src/components/shell/ModeSelector.tsx`; modify `src/components/shell/AppShell.tsx`;
add `src/components/shell/SceneCrumb.tsx`; update any test referencing `Analysis mode`.

The header keeps its grid row (it also spans the thread column and holds the health strip). The left
slot becomes `SceneCrumb`: `.t-eyebrow` section name (`Explore`) and, when a run is loaded, `›` then
the scene name from `SceneHeader`'s source (`manifest.name` or file name) in `.t-panel`, then the
`pairType` chip (`BI_TEMPORAL`) in mono. On the landing page the header does not render at all
(`Landing` sits outside `AppShell`). Nothing else moves; the removal leaves the bar with a reason to
exist rather than a hole.

---

## 5. Cross-cutting

### 5.1 New shared primitives

| File | Purpose |
|---|---|
| `src/components/ui/popover.tsx` | Radix Popover + glass + `sq-pop-in`, `collisionPadding: 12` |
| `src/components/ui/Glass.tsx` | `<div class="glass">` with `contain: paint`; accepts `inset` for HUD placement |
| `src/components/ui/Graticule.tsx` | `aria-hidden` ground layer; props `module`, `major`, `mask` |
| `src/components/ui/Reticle.tsx` | the SVG symbol; props `size`, `breathe`, `lock` |
| `src/components/ui/SectionHeader.tsx` | page title row |
| `src/components/ui/Segmented.tsx` | `role="radiogroup"` pill switch (theme, cards/rows) |
| `src/components/ui/Skeleton.tsx` | shimmer bars; honours reduced motion |
| `src/components/ui/Toast.tsx` | transient notifications |
| `src/components/ui/ErrorBoundary.tsx` | pairs with every lazy route (§6.5) |
| `src/shell/DocumentMeta.tsx` | per-route head tags (§6.1) |
| `src/components/ui/icons.tsx` | + `MoonIcon SunIcon ContrastIcon KeyboardIcon LayersIcon LocateIcon SplitIcon FolderIcon PinIcon TrashIcon UndoIcon ArrowRightIcon RadarIcon SignOutIcon UsersIcon` — same 1.5px stroke grammar as the existing set |

CSS classes added to `theme.css` `@layer components`: `.glass`, `.graticule`, `.reticle`,
`.t-display`, `.t-display-sub`, `.t-section`, `.t-lede`, `.t-coord`, `.t-quote`, `.lift` (the one
hover-lift rule, `@media (hover: hover)` only). Keyframes: `sq-pop-in`, `sq-scan`, `sq-wipe`,
`sq-drift`, `sq-breathe`, `sq-sweep`, `sq-shimmer`, each with a `prefers-reduced-motion` override in
the same block as the existing ones.

### 5.2 Sound & haptics

The product runs in a judging hall. **No sound.** `navigator.vibrate(8)` on touch devices only, for
exactly two moments: a run reaching a terminal state while the tab is hidden, and a successful
sample load. Both are wrapped in `try` and gated by `matchMedia('(pointer: coarse)')`.

### 5.3 Loading skeletons

Every page with async content ships a skeleton that matches its final layout's *dimensions*
(cards 16:10, rows 40px, HUD panes at their final size) so nothing shifts when data lands. Shimmer
is a `linear-gradient` `background-position` loop, 1.6 s, `--color-line-soft` → `--color-line`;
reduced motion → static `--color-line-soft`.

### 5.4 Responsiveness

Everything above is specified for `desk` (≥1280 × ≥500) and degrades to `wide` (≥768) and phone
exactly as `AppShell` already does: HUD panes collapse to a bottom bar, grids go to one column,
the account popover becomes a bottom sheet under `wide` (Radix Popover with `side="top"` and full
width), the shortcuts modal is `inset-3`. The landing hero is single-column with the artifact
below the CTA, `.t-display` at its clamp minimum, ambience reduced to graticule + one glow.

### 5.5 Accessibility contract

- All new token pairs in `contrast.test.ts` (both themes).
- Every popover/modal: focus trap, `Esc`, focus return, `aria-labelledby`.
- Every animation has the reduced-motion path written in the same file as the keyframe.
- Single-key shortcuts gated by `useShortcutStore.enabled`; chords never gated; the guide reachable
  by `⌘/` and from the popover without a shortcut.
- Landing sections are `<section aria-labelledby>`; the pipeline story exposes a text fallback
  (`<ol>` of the four stages) that is always in the DOM.
- Maps: every HUD control is keyboard-operable; the map canvas has `aria-label` describing the
  scene and `role="application"`.

### 5.6 Tests to add or move

| Test | Asserts |
|---|---|
| `styles/__tests__/contrast.test.ts` | OKLCH parsing; light-theme pairs |
| `shell/__tests__/useDailyQuote.test.ts` | 31 entries, day 1 → index 0, day 31 → index 30, every entry has `by` |
| `shell/__tests__/useHotkeys.test.tsx` | chords work with `enabled=false`; `G`-sequences time out at 600 ms; `?` opens the guide |
| `state/__tests__/theme.test.ts` | `.dark` toggles; `system` follows `matchMedia` |
| `state/__tests__/notifications.test.ts` | `settleRun` pushes; cap 50; transient never listed |
| `state/__tests__/library.test.ts` | save/assign/delete; GeoJSON with and without bounds |
| `shell/__tests__/router.test.ts` | path ↔ section round trip; unknown → `notFound`; trailing slash normalised |
| `e2e/landing.spec.ts` | `/` renders hero; `Launch Mission` lands on `/explore` with composer focused; reduced-motion snapshot |
| `src/__tests__/headings.test.tsx` | every routed page renders exactly one `h1` (§6.2) |
| `src/__tests__/meta.test.tsx` | every route sets `<title>`, description, canonical (§6.1) |
| `src/__tests__/bundle-boundaries.test.ts` | no static import of `three`, `@react-three/*`, `maplibre-gl` outside their lazy owners; no `font-odida` outside `src/pages/landing` (§6.5) |
| `src/__tests__/alt-text.test.ts` | every `<img` in `src/` has a non-empty `alt` or an explicit `alt=""` + `aria-hidden` (§6.4) |

### 5.7 Build order (dependency-respecting, not a schedule)

1. §0.4–0.7: tokens → OKLCH, light theme, contrast test, router, primitives (§5.1).
2. §4.2, §4.3, §4.5: shortcut store + guide, theme, header cleanup (small, unblock everything).
3. §3 + §4.4 + §4.1: popover primitive → account popover → notifications → quote.
4. §2.1 Use Cases (needs sample assets — request them now).
5. §2.3 / §2.4 Saved + Projects (library store, export).
6. §2.2 Maps (maplibre, lazy).
7. §6 production checklist items that touch the shell (`DocumentMeta`, `h1` policy, 404, chunks).
8. The landing page per `landing_page_blueprint.md` (last: it showcases everything above and its
   proof strip must show real numbers).
9. One batched inspection round (desktop + phone, both themes, reduced motion), one fix batch, stop.

### 5.8 Assets the team must supply

- `public/samples/<slug>/` rasters + `thumb.webp` for each Use Case row that should be visible.
- `public/og/default.png` 1200×630 (§6.1) — the console's share image.
- A sourced ISRO/SAC remote-sensing quote for #27 (optional).
- Landing assets (Odida font, land mask, capability previews, `og/landing.png`) are listed in
  `landing_page_blueprint.md` §10.

### 5.9 Open decisions (not for the builder to invent)

1. Headline copy: `Ask the satellite. / Audit the answer.` vs the tagline variant
   (`landing_page_blueprint.md` §5.1).
2. Whether an online basemap is allowed in the judging hall (§2.2 — default off either way).
3. Stage 3 naming in the pipeline story (`landing_page_blueprint.md` §6.1) — the backend does not
   orthorectify.
4. Whether `Sign out & clear this device` should exist at all without accounts (§3).
5. Quote attributions marked *(attributed)* — keep, replace, or drop.
6. The public canonical origin for `sitemap.xml` / `robots.txt` / `canonical` (§6.3) — the app also
   runs on a lab LAN where none of them matter; they must not break there.

---

## 6. Production polish — the anti-vibecoding checklist

The giveaways of a generated frontend are not visual; they are the defaults nobody changed: the Vite
title, a wordmark `h1` on every page, `alt=""` on a chart, a 3 MB entry chunk, source maps in
production, and a `/robots.txt` that 404s. Each item below names the mechanism, the file, and the
test that keeps it true. Everything here applies to every route, landing included.

### 6.1 SEO & document metadata

**No `react-helmet`.** React 19 hoists `<title>`, `<meta>` and `<link>` rendered anywhere in the tree
into `<head>` natively and de-duplicates them by `name`/`rel`. A dependency that re-implements that
is the vibecoding tell in itself.

| Path | New / Modified |
|---|---|
| `src/shell/DocumentMeta.tsx` | new — the one component that renders head tags |
| `src/shell/meta.ts` | new — the per-route table below + `SITE` config |
| `src/App.tsx` | modified — every page renders `<DocumentMeta page=…/>` first |
| `index.html` | modified — static fallbacks (§6.1.3) |
| `public/og/default.png`, `public/og/landing.png` | **assets** — 1200×630 |

```ts
// src/shell/meta.ts
export const SITE = {
  name: 'SatQuery AI',
  origin: import.meta.env.VITE_PUBLIC_ORIGIN ?? '',   // '' on the lab LAN → canonical/og:url omitted
  defaultDescription: 'Mission control for grounded Earth-observation analysis. Ask a question of satellite imagery and get an answer where every number is traced to the tool that measured it.',
  ogImage: '/og/default.png',
  locale: 'en_IN',
} as const

export interface PageMeta { title: string; description: string; path: string; ogImage?: string; noindex?: boolean }
export const META: Record<Section | 'notFound' | 'report', PageMeta> = {
  home:      { title: 'SatQuery AI — Ask the satellite. Audit the answer.', description: '…', path: '/', ogImage: '/og/landing.png' },
  explore:   { title: 'Explore · SatQuery AI', description: 'Drop imagery, run pre-flight, ask a grounded question.', path: '/explore' },
  usecases:  { title: 'Use cases · SatQuery AI', description: 'Eight runnable geospatial investigations over Indian scenes.', path: '/use-cases' },
  maps:      { title: 'Maps · SatQuery AI', description: 'The loaded scene in the world, with every rendered view switchable.', path: '/maps' },
  saved:     { title: 'Saved · SatQuery AI', description: 'Runs kept on this device.', path: '/saved', noindex: true },
  projects:  { title: 'Projects · SatQuery AI', description: 'Investigations grouped by question.', path: '/projects', noindex: true },
  datasets:  { title: 'Datasets · SatQuery AI', description: '…', path: '/datasets' },
  tools:     { title: 'Tools · SatQuery AI', description: 'The tool registry, live from /v1/registry.', path: '/tools' },
  history:   { title: 'History · SatQuery AI', description: '…', path: '/history', noindex: true },
  report:    { title: 'Report · SatQuery AI', description: '…', path: '/report', noindex: true },
  notFound:  { title: 'Not found · SatQuery AI', description: '…', path: '', noindex: true },
}
```

```tsx
// src/shell/DocumentMeta.tsx — rendered once per page; React 19 hoists these
export function DocumentMeta({ page, title }: { page: keyof typeof META; title?: string }) {
  const m = META[page]
  const t = title ?? m.title                       // pages with a subject override: "Bengaluru sprawl · Projects · SatQuery AI"
  const url = SITE.origin ? SITE.origin + m.path : null
  return (
    <>
      <title>{t}</title>
      <meta name="description" content={m.description} />
      {m.noindex && <meta name="robots" content="noindex" />}
      {url && <link rel="canonical" href={url} />}
      <meta property="og:type" content="website" />
      <meta property="og:site_name" content={SITE.name} />
      <meta property="og:title" content={t} />
      <meta property="og:description" content={m.description} />
      <meta property="og:image" content={(SITE.origin || '') + (m.ogImage ?? SITE.ogImage)} />
      <meta property="og:image:width" content="1200" />
      <meta property="og:image:height" content="630" />
      <meta property="og:image:alt" content={`${SITE.name} — ${m.description}`} />
      {url && <meta property="og:url" content={url} />}
      <meta property="og:locale" content={SITE.locale} />
      <meta name="twitter:card" content="summary_large_image" />
    </>
  )
}
```

Rules:

1. **Title grammar** is `Subject · Section · SatQuery AI` — most specific first, the brand last, `·`
   as the separator. The landing page is the one exception (brand first, then the claim).
2. **`<html lang="en">`** is already in `index.html`. Keep it; add `dir="ltr"`. Content in another
   language (a Hindi quote, if one is added) carries its own `lang` attribute on the element.
3. **Static fallbacks in `index.html`** stay for crawlers that do not execute JS: the existing
   `<title>` and `description`, plus `og:title`, `og:description`, `og:image` (`/og/default.png`)
   and `<link rel="canonical">` only when `VITE_PUBLIC_ORIGIN` is set at build time (Vite `%VITE_…%`
   substitution). The React tree overrides them per page.
4. **`theme-color`** currently hard-codes `#100C0A` and `color-scheme` says `dark`. Both become
   theme-aware: two `<meta name="theme-color" media="(prefers-color-scheme: …)">` tags in
   `index.html`, and `applyTheme()` (§4.3) updates the live one on toggle.
5. A route that shows user data (`saved`, `projects`, `history`, `report`) is `noindex`.

### 6.2 Semantic HTML — exactly one `h1` per page

**Current defect.** `Sidebar.tsx:104` renders the wordmark as `<h1>` on every console page, and each
page title (`HistoryPanel`, `DatasetsPanel`, `ToolsPanel`, `SceneHeader`) is an `<h2>`. So every
page has exactly one `h1`, and it is the wrong one: the product name, not the page. Fix:

| Element | Now | Becomes |
|---|---|---|
| Sidebar wordmark | `<h1>` | `<p class="…">` inside the `<button>` that routes home |
| `SectionHeader` title (new primitive, §5.1) | — | `<h1 class="t-page">` |
| `HistoryPanel`, `DatasetsPanel`, `ToolsPanel` titles | `<h2 class="t-page">` | `<h1>` via `SectionHeader` |
| `SceneHeader` scene title (Explore) | `<h2 class="t-page">` | `<h1>` — on Explore the scene *is* the page; before a scene loads, the `Dropzone` heading `Drop imagery` is the `h1` |
| Landing headline | — | the only `h1` on `/` |
| 404 page | — | `<h1>` `We can't find that page.` |
| Popovers / modals | — | `h2` via `aria-labelledby`; a dialog never carries an `h1` |

Heading levels never skip: `h1` → section `h2` → card `h3`. Landmarks: one `<main id="main">` per
page, `<nav aria-label="Primary">` (exists), `<header>` (exists), `<aside>` for the thread column,
`<footer>` on the landing page. A visually-hidden `Skip to main content` link is the first focusable
element in `AppShell` and `Landing`.

**Test.** `headings.test.tsx` renders every routed page with mocks and asserts
`document.querySelectorAll('h1').length === 1`, and that no heading level increases by more than
one between consecutive headings in DOM order.

### 6.3 Crawlers — `sitemap.xml`, `robots.txt`, `llms.txt`, canonical

The app is a Vite SPA with no SSR, so these are **static files in `public/`**, generated at build time
by a small script (`scripts/gen-public-meta.ts`, run from `npm run build` via `prebuild`) from the
same `META` table, so a new route cannot be forgotten.

```
public/robots.txt
  User-agent: *
  Allow: /
  Disallow: /saved
  Disallow: /projects
  Disallow: /history
  Disallow: /report/
  Disallow: /v1/
  Sitemap: ${VITE_PUBLIC_ORIGIN}/sitemap.xml
```

No AI-crawler block. The file is explicit about it with a comment (`# AI crawlers welcome — see
/llms.txt`) so a reviewer does not assume an omission. `GPTBot`, `ClaudeBot`, `PerplexityBot`,
`Google-Extended` are *not* listed, which is the same as `Allow`.

```
public/sitemap.xml   — one <url> per META entry without noindex: /, /explore, /use-cases, /maps,
                       /datasets, /tools; <lastmod> = build date; no priorities (they are ignored)
```

```
public/llms.txt      — Markdown, ≤ 2 KB, in the llms.txt convention:
  # SatQuery AI
  > Evidence-bound geospatial VLM assistant … (one paragraph from PRODUCT.md)
  ## What it does          (the capability table from README, as bullets)
  ## How answers are built (the four pipeline stages, with the deterministic-controller sentence)
  ## Pages                 (each public route with its description)
  ## Docs                  (links to DOCS/API_CONTRACT.md, AGENT_POLICY_DAG.md if the repo is public)
  ## Not for               (no accounts, no cloud storage, runs offline)
```

`llms-full.txt` is not generated — there is nothing long-form to add beyond the docs links.

**Canonical.** `<link rel="canonical">` is rendered per page by `DocumentMeta` (§6.1) and only when
`VITE_PUBLIC_ORIGIN` is set; on the lab LAN there is no public origin and a canonical pointing at
`http://192.168…` would be wrong. Trailing slashes: none (`/use-cases`, not `/use-cases/`); the
router normalises on load.

`nginx.conf` already serves `try_files $uri … /index.html`, so the three static files are served
as files and everything else falls through to the app.

### 6.4 UX & performance floor

**404 page — `src/pages/NotFound.tsx`.** The router (§0.7) maps an unknown path to `notFound`, not
silently to Explore. Rendered inside `AppShell` (the sidebar still works — the user is lost, not
locked out), `DocumentMeta page="notFound"`, and the server still returns `index.html` with 200 (an
SPA cannot send a 404 status; a `<meta name="robots" content="noindex">` covers the crawler case).
Composition: graticule ground, one reticle drifting slowly off-centre, `<h1>` `We can't find that
page.`, `.t-meta` `The address may be wrong, or the run it pointed at was never saved on this
device.` with the attempted path in `.t-coord`, and two actions: `Open the console` (primary) ·
`Back` (`history.back()`, ghost, shown only when `history.length > 1`). No illustration, no joke.

**Source maps.** `vite.config.ts` → `build: { sourcemap: false }` explicitly (it is the default, but
an explicit `false` survives a future `sourcemap: true` copy-paste from a debugging session). The
dev server keeps maps. Error reporting, if ever added, uploads maps out-of-band rather than serving
them.

**`alt` text.** Policy, enforced by `alt-text.test.ts` (a source scan) and an oxlint rule
(`jsx-a11y/alt-text` via the `oxlint` a11y plugin, `--deny-warnings` is already on):

| Image | `alt` |
|---|---|
| Evidence view (`ImageViewer`, `EvidenceTray`) | the view label + scene: `"NDVI, post-change, Bengaluru 2024"` — never `''`. `primary?.label ?? ''` becomes `primary?.label ?? group.name` |
| Manifest thumbnail (`ManifestCard`) | `""` + `aria-hidden` — decorative beside the filename it repeats (already correct; keep) |
| Use Case / Saved thumbs | the title: `"Urban sprawl, Bengaluru — 2019 to 2024"` |
| Capability previews | scene + overlay, e.g. `"Sentinel-2 true-colour pair over Bengaluru with the change mask on the right half"` |
| Avatar initial | not an image (it is text in a `<span>`) |
| SVG icons | `aria-hidden` (already the case in `icons.tsx`); icon-only buttons carry `aria-label` |
| The globe canvas | `aria-hidden`; its wrapper is `role="img"` with a label |

**Other floor items, one line each:**

- Every `fetch`/`useQuery` surface has a loading, empty, and error state (§5.3 skeletons; every
  section in §2 lists its error copy).
- No `console.log` in `src/` (oxlint `no-console: error`, `console.error` allowed in `catch`).
- `index.html` gets `<link rel="manifest">` + `/manifest.webmanifest` (name, short_name,
  theme/background colours, the SVG icon and a 512px PNG) so "Add to Home Screen" on a judge's
  phone produces a real icon, not a screenshot.
- `apple-touch-icon` 180px PNG.
- External links (`GitHub`, docs) carry `rel="noopener"`; none carry `target="_blank"` without an
  `↗` glyph and `aria-label` that says it opens in a new tab.
- Forms: every input has a `<label>`; the composer already does.
- `user-select` is never disabled on text; `outline: none` never appears without a `:focus-visible`
  replacement (the base already provides one).
- The favicon is the existing SVG plus an `.ico` fallback for the one browser that still asks.

### 6.5 Bundle splitting & Suspense boundaries

Every heavy dependency has exactly one owner file that imports it, and that owner is reached only
through `React.lazy`. The console entry must stay under **180 KB gz** JS (today's console without
the DAG); the landing entry under **60 KB gz** before the globe chunk.

| Chunk (`manualChunks`) | Contents | Owner (only importer) | Suspense fallback |
|---|---|---|---|
| `globe` | `three`, `@react-three/fiber`, `@react-three/drei` (subpath imports only) | `src/pages/landing/globe/Globe.tsx` | `<GlobePoster/>` — a finished SVG still |
| `map` | `maplibre-gl` (+ its CSS, imported as `?inline` and injected by the owner) | `src/pages/maps/MapStage.tsx` | the scene footprint drawn as a dashed sand polygon on the graticule + HUD skeleton at final size |
| `dag` | `@xyflow/react`, `dagre` | `src/components/pipeline/DagCanvas.tsx` (already lazy) | existing `PipelineDialog` skeleton |
| `landing` | the page + `PipelineStory` + `Capabilities` | `src/App.tsx` via `lazy(() => import('@/pages/Landing'))` | ground + graticule + nav skeleton, no spinner |
| `usecases`, `saved`, `projects`, `maps`, `notfound` | one route each | `src/App.tsx` | `SectionHeader` at final height + 6 skeleton cards |
| `library` | `idb-keyval` + `src/state/library.ts` | `src/pages/Saved.tsx` / `Projects.tsx` + the `Save` action (dynamic `import()` at click time) | — |
| `report` | `src/export/report.tsx` | its own route | — |
| vendor (Vite default) | `react`, `react-dom`, `zustand`, `@tanstack/react-query`, `@radix-ui/*` | entry | — |

```tsx
// src/App.tsx — shape
const Landing  = lazy(() => import('@/pages/Landing'))
const UseCases = lazy(() => import('@/pages/UseCases'))
const Maps     = lazy(() => import('@/pages/Maps'))
const Saved    = lazy(() => import('@/pages/Saved'))
const Projects = lazy(() => import('@/pages/Projects'))
const NotFound = lazy(() => import('@/pages/NotFound'))

if (section === 'home') return <Suspense fallback={<LandingSkeleton/>}><Landing/></Suspense>
return (
  <AppShell stage={<Suspense fallback={<SectionSkeleton/>}>{stage}</Suspense>} thread={<ThreadPanel/>} />
)
```

Rules:

1. **Boundaries sit at the route and at the heavy widget — never around a whole layout.** The
   sidebar, header and thread column render synchronously; only the centre column suspends. A
   judge clicking `Maps` sees the shell instantly and the map arrive.
2. **Fallbacks are final-size.** A Suspense fallback that is a spinner shifts the layout when the
   chunk lands; every fallback above reserves the exact box.
3. **Preload on intent, not on load.** `Sidebar` calls the route's `import()` on `pointerenter` /
   `focus` of its nav button (a 150 ms head start that removes the fallback in practice). The
   `globe` chunk preloads after the landing page's first paint (`requestIdleCallback`, fallback
   `setTimeout 1`), and only when WebGL is supported and motion is not reduced.
4. **Barrel imports are banned for the heavy three.** `import { OrbitControls } from
   '@react-three/drei'` pulls the whole library through Rollup's tree-shaking heuristics unevenly;
   use `@react-three/drei/core/OrbitControls`. Likewise `three/examples/jsm/...` never appears
   outside the `globe` owner.
5. **`vite.config.ts`** additions: `build.sourcemap: false`, `build.rollupOptions.output.manualChunks`
   as above, `build.chunkSizeWarningLimit: 250` (so the warning still fires for anything that is
   not one of the named chunks), `assetsInclude: ['**/*.glsl']`.
6. **Budget test.** `bundle-boundaries.test.ts` greps `src/` for static imports of `three`,
   `@react-three/`, `maplibre-gl`, `@xyflow/react`, `idb-keyval` and asserts each appears only in
   its owner; CI additionally runs `vite build` and fails if `dist/assets/index-*.js` exceeds the
   entry budget (a 12-line script reading `dist/.vite/manifest.json` with `build.manifest: true`).
7. **Error boundaries pair with Suspense boundaries.** Each lazy route sits inside an
   `ErrorBoundary` (`src/components/ui/ErrorBoundary.tsx`, new) whose fallback is the section's
   header plus `This view failed to load. [Retry]` — a chunk that 404s after a deploy (stale
   `index.html`, new hashes) is the most common production-only crash in an SPA, and `Retry`
   calls `location.reload()`.

### 6.6 Checklist (paste into the PR template)

- [ ] `<title>` is `Subject · Section · SatQuery AI` and changes per route
- [ ] `description`, `canonical` (when public), `og:image` 1200×630 with `og:image:alt`
- [ ] `<html lang="en" dir="ltr">`; foreign-language spans carry `lang`
- [ ] exactly one `h1` per page, and it is the page — not the brand
- [ ] `robots.txt` allows all agents and lists the sitemap; `sitemap.xml` and `llms.txt` generated from `META`
- [ ] custom 404 with `noindex`, reachable from the router
- [ ] `build.sourcemap: false`
- [ ] every `<img>` has descriptive `alt`, or `alt=""` + `aria-hidden` when decorative
- [ ] heavy libs (`three`, `maplibre-gl`, `@xyflow/react`) each have one lazy owner; entry ≤ 180 KB gz
- [ ] every Suspense boundary has a final-size fallback and an ErrorBoundary
- [ ] `font-odida` appears nowhere outside `src/pages/landing`
- [ ] no `console.log`, no `outline: none` without `:focus-visible`, no `target="_blank"` without `rel="noopener"` and a label
- [ ] `manifest.webmanifest`, `apple-touch-icon`, theme-aware `theme-color`
