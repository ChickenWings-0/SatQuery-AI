# SatQuery AI — Landing Page Blueprint (`/`)

**Status:** design brief, produced by `/impeccable shape`. Extracted from `frontend_blueprint.md` §1
and extended with the Odida display face and the interactive 3D globe. No code here is final; every
snippet is an interface contract. The token system, router, primitives, motion durations and the
anti-vibecoding checklist it depends on live in [`frontend_blueprint.md`](frontend_blueprint.md)
§0 and §6 — this document does not restate them.

**Visitor mode:** Persuade. The visitor decides and acts. One rehearsed focal sequence, product truth
as proof, a single door into the console.

---

## 1. Job, audience, outcome

A judge or a first-time visitor arrives with 20 seconds of attention. They must understand *this
system is auditable, and I can try it now.* Success is one of two actions — **Launch Mission**
(opens the console on Explore with the composer focused) or **Drop Imagery** (drop rasters on the
hero and land in the console with pre-flight already running). Both are the same door.

The page is the only Persuade surface in the product. Everything it shows must be either a real
capability, a real number from `/v1/health` / `/v1/registry`, or a recorded fixture value — the
console's honesty premise does not stop at the marketing page.

---

## 2. Component tree

```
<Landing>                                  src/pages/Landing.tsx  (React.lazy from App.tsx)
├─ <DocumentMeta page="home">              §6.1 of frontend_blueprint — title, description, canonical, og
├─ <LandingNav>                            glass bar; wordmark button, links, ThemeToggle, "Open console"
├─ <main id="main">
│  ├─ <Hero>                               section, aria-labelledby=#hero-title
│  │  ├─ <Ambience>                        aria-hidden: Graticule · Glow ×2 · Reticle ×3 · ScanLine
│  │  ├─ <HeroCopy>                        eyebrow (.t-coord + .t-eyebrow), <h1 class="t-display">, .t-display-sub, CTA cluster
│  │  ├─ <DropLaunchOverlay>               shown while a drag is over the window (useDropLaunch)
│  │  └─ <Suspense fallback={<GlobePoster/>}>
│  │       <Globe/>                        React.lazy → src/pages/landing/globe/Globe.tsx  (three + R3F chunk)
│  │     </Suspense>
│  ├─ <ProofStrip>                         real numbers, or the honest offline line
│  ├─ <PipelineStory>                      section, four stages, scroll-played
│  │  └─ <Stage n={1..4}>                  each owns its own mini-animation
│  ├─ <Capabilities>                       section, tabs + preview stage
│  │  ├─ <CapabilityChip role="tab"> ×3
│  │  └─ <CapabilityPreview role="tabpanel">
│  └─ <ClosingCta>                         one line, one button, the graticule fades out beneath it
└─ <LandingFooter>                         wordmark, "SIH 2026 · PS 26167", links: Use cases · Maps · GitHub (if public)
```

Files:

| Path | New / Modified | Role |
|---|---|---|
| `src/pages/Landing.tsx` | new | page shell, `React.lazy` target |
| `src/pages/landing/LandingNav.tsx` | new | floating glass nav |
| `src/pages/landing/Hero.tsx` | new | composition + CTA cluster + drop target |
| `src/pages/landing/Ambience.tsx` | new | graticule / glow / reticle / scan-line (CSS only) |
| `src/pages/landing/DropLaunchOverlay.tsx` | new | full-hero drop affordance |
| `src/pages/landing/useDropLaunch.ts` | new | window drag → `selectFiles` → `setSection('explore')` |
| `src/pages/landing/globe/Globe.tsx` | new | R3F `<Canvas>` wrapper, lazy chunk boundary |
| `src/pages/landing/globe/Earth.tsx` | new | icosphere + `ShaderMaterial` |
| `src/pages/landing/globe/earth.vert.glsl` / `earth.frag.glsl` | new | shader source (imported as `?raw`) |
| `src/pages/landing/globe/Satellite.tsx` | new | low-poly body + trail |
| `src/pages/landing/globe/orbits.ts` | new | orbital elements + `positionAt(t)` |
| `src/pages/landing/globe/Controls.tsx` | new | restricted `OrbitControls` |
| `src/pages/landing/globe/GlobePoster.tsx` | new | static SVG globe: Suspense fallback, no-WebGL fallback, reduced-motion still frame |
| `src/pages/landing/globe/tokens.ts` | new | linear-sRGB triplets derived from the OKLCH tokens (+ test) |
| `src/pages/landing/useGlobeVisibility.ts` | new | IntersectionObserver → frameloop on/off |
| `src/pages/landing/ProofStrip.tsx` | new | `/v1/health` + `/v1/registry` |
| `src/pages/landing/PipelineStory.tsx` | new | four-stage story |
| `src/pages/landing/Capabilities.tsx` | new | tabs + preview |
| `src/pages/landing/ClosingCta.tsx`, `LandingFooter.tsx` | new | |
| `src/pages/landing/useReveal.ts` | new | once-only reveal hook |
| `src/state/landing.ts` | new | `useLandingStore` (§4) |
| `src/App.tsx` | modified | `section === 'home'` renders `<Landing/>` outside `AppShell` |
| `src/components/shell/Sidebar.tsx` | modified | wordmark → `<button>` to `home`; `<h1>` → `<p>` (one `h1` per page rule) |
| `src/styles/theme.css` | modified | Odida `@font-face` + `--font-odida`, `.t-display*` (§3), landing keyframes (§7) |
| `vite.config.ts` | modified | `manualChunks` for `three`/`@react-three/*`; `?raw` glsl imports |
| `package.json` | modified | `three`, `@react-three/fiber`, `@react-three/drei`, `@types/three` |
| `public/fonts/Odida.ttf` → `Odida.woff2` | **asset the team supplies** | see §3 |
| `public/samples/capabilities/{change,crossmodal,grounding}/*.webp` | **asset the team supplies** | |
| `public/og/landing.png` (1200×630) | **asset the team supplies** (or rendered from the poster SVG) | `og:image` |

---

## 3. Typography — Odida on the display tier

Odida is the landing page's voice and nothing else's. It appears in exactly two classes,
`.t-display` and `.t-display-sub`, and both classes are allowed only on this page. Section titles,
ledes, the console, and every UI control stay Inter; measurements stay Geist Mono. A display face on
a KPI card is the fastest way to make a console look like a template.

**Font file.** The brief assumes `public/fonts/Odida.ttf`. Ship it as `Odida.woff2` (30–40 % smaller,
same glyphs) via a `fonts:sync` step — `npx ttf2woff2 < Odida.ttf > Odida.woff2` — and keep the TTF
only as a source. If licensing forbids conversion, load the TTF; the `@font-face` below lists both.

```css
/* theme.css — beside the Inter / Geist Mono declarations */
@font-face {
  font-family: 'Odida';
  font-style: normal;
  font-weight: 400;                 /* Odida ships one weight; never synthesise bold */
  font-display: swap;               /* the headline must paint in the fallback, then swap */
  src:
    url('/fonts/Odida.woff2') format('woff2'),
    url('/fonts/Odida.ttf') format('truetype');
  unicode-range: U+0000-00FF, U+2013-2014, U+2018-2019, U+201C-201D, U+2026;  /* Latin + the quotes/dashes the copy uses */
}

/* Metric-matched fallback so the pre-swap headline occupies the same box.
   The four override values MUST be measured from the actual Odida file
   (fontkit / `npx fontpie public/fonts/Odida.woff2`) — the numbers below are
   placeholders and a wrong ascent-override causes a visible headline jump. */
@font-face {
  font-family: 'Odida Fallback';
  src: local('Impact'), local('Arial Narrow'), local('Liberation Sans Narrow'), local('Arial');
  size-adjust: 000%;        /* measure */
  ascent-override: 000%;    /* measure */
  descent-override: 000%;   /* measure */
  line-gap-override: 0%;
}

@theme {
  --font-odida: 'Odida', 'Odida Fallback', 'Inter Variable', 'Inter Fallback', ui-sans-serif, sans-serif;
}
```

`--font-odida` in `@theme` yields the `font-odida` utility; the classes below use the variable
directly so the pairing cannot be applied by accident elsewhere.

```css
@layer components {
  .t-display {
    font-family: var(--font-odida);
    font-size: clamp(3rem, 6.8vw, 6.25rem);
    line-height: 0.94;
    letter-spacing: -0.02em;          /* re-tune once Odida's sidebearings are seen; display faces often need 0 */
    font-weight: 400;
    text-wrap: balance;
    font-synthesis: none;             /* no faux bold / italic, ever */
  }
  .t-display-sub {
    font-family: var(--font-odida);
    font-size: clamp(1.25rem, 1.9vw, 1.625rem);
    line-height: 1.3;
    letter-spacing: 0;
    font-weight: 400;
    max-width: 30ch;
    font-synthesis: none;
  }
}
```

**Preload.** `index.html` preloads Odida *only when the path is `/`* — a preload hint on the console
routes is wasted bandwidth for a face they never render. Since `index.html` is static, do this with a
`<link rel="preload" … media="…">`-free approach: `Landing.tsx` injects
`<link rel="preload" as="font" href="/fonts/Odida.woff2" type="font/woff2" crossOrigin="">` via React
19's native hoisting, so the hint exists exactly when the page does.

**Pairing rule.** Odida (voice, two classes) · Inter (everything else on the page) · Geist Mono
(coordinates, tool names, policy keys, readouts). Three faces is the ceiling; the eyebrow stays
Inter `.t-eyebrow` so the coordinate prefix in mono has one neighbour, not two.

---

## 4. State — `useLandingStore`

Landing state is small and mostly local; the store exists because the globe, the hero copy and the
drop overlay live in different subtrees and share three facts.

```ts
// src/state/landing.ts
import { create } from 'zustand'

export type HeroPhase = 'poster' | 'acquiring' | 'live'
export type GlobeSupport = 'unknown' | 'webgl' | 'none'

interface LandingState {
  phase: HeroPhase                 // poster → acquiring (focal sequence running) → live (loops on)
  globe: GlobeSupport              // detected once on mount; 'none' keeps <GlobePoster/> forever
  globeVisible: boolean            // IntersectionObserver; drives R3F frameloop
  dragOver: boolean                // a file drag is over the window
  activeCapability: 'change' | 'crossmodal' | 'grounding'
  hoveredCapability: LandingState['activeCapability'] | null
  storyPlayed: ReadonlySet<1 | 2 | 3 | 4>
  storyReplaying: 1 | 2 | 3 | 4 | null

  setPhase(p: HeroPhase): void
  setGlobe(s: GlobeSupport): void
  setGlobeVisible(v: boolean): void
  setDragOver(v: boolean): void
  setCapability(id: LandingState['activeCapability']): void
  hoverCapability(id: LandingState['hoveredCapability']): void
  markStagePlayed(n: 1 | 2 | 3 | 4): void
  replayStory(): Promise<void>     // sequences 1→4 with 400 ms gaps, sets storyReplaying
  reset(): void                    // on unmount, so a return visit replays the focal moment
}
```

Nothing here is persisted. Cross-store reads (`useReducedMotion`, `useThemeStore.resolved`,
`useUiStore.selectFiles`, `useFocusStore.proposeQuestion`) are called via `getState()` from actions,
never subscribed inside the R3F tree (a Zustand subscription inside `useFrame` is a re-render per
frame).

---

## 5. Hero

### 5.1 Composition

Full viewport (`min-height: 100svh`), 12-column grid from `desk`, single column below `wide`.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ [◉ SatQuery AI]                            Use cases · Maps  [☾] [Open console ↗] │  glass nav, 56px
│                                                                                  │
│      12°58′N 77°35′E  EVIDENCE-BOUND GEOSPATIAL INTELLIGENCE                     │  cols 1–7
│      ┼                                                   ╭────────────────╮      │
│      Ask the satellite.                                 ╱   ·  ·  ·  ·     ╲     │  cols 7–12
│      Audit the answer.                                 │  ·   ╱───╲  ·  ·  │     │  the globe, bleeding
│                                                        │ ·   │  ⊕  │   · · │     │  past the right edge
│      Bi-temporal, cross-modal SAR/optical analysis     │  ·   ╲───╱ ·  ·   │     │  by ~12 %
│      where every number is bound to a measurement       ╲  ·  ·  ·  ·  ·  ╱      │
│      and the tool graph is part of the answer.           ╰────────────────╯       │
│                                                            ▪ ─ ─ ─ satellite     │
│      [▶ Launch Mission]   [⬚ Drop imagery]   ⌘K                                 │
│                                                                                  │
│  ● System Ready · 13/15 tools · schema 1.0 · Qwen3-VL-8B + QLoRA · runs offline  │  proof strip
└──────────────────────────────────────────────────────────────────────────────────┘
```

- **Eyebrow:** `.t-coord` coordinate + `.t-eyebrow` label on one line. The coordinate is *real*: the
  sub-satellite point the globe is initially centred on (Bengaluru, `12°58′N 77°35′E`), which is also
  the first Use Case's centroid. If the globe's initial target changes, the coordinate changes.
- **Headline (`<h1 class="t-display">`):** `Ask the satellite.` / `Audit the answer.` — two lines,
  the second in `text-accent-warm-text`. Alternative: `From space to answers —` / `with the
  receipts.` One of the two; the choice is open decision #1 in `frontend_blueprint.md` §5.9.
  This `h1` is the page's only `h1`.
- **Standfirst (`.t-display-sub`):** the three-line sentence above, `max-width: 30ch`.
- **CTA cluster:** `Launch Mission` (`.btn-primary`, 44px, arrow translates 2px on hover;
  `setSection('explore')` → `focusComposer()` after the view transition) and `Drop imagery`
  (`.btn-ghost` + hidden `<input type=file multiple accept={ACCEPT}>`, `ACCEPT` imported from
  `Dropzone.tsx`). `⌘K` in `.kbd` beside them. The *entire hero* is a drop target
  (`useDropLaunch`): while `dragOver`, `DropLaunchOverlay` draws a dashed sand rectangle over the
  hero, the copy dims to 40 %, and the globe's satellites brighten — the page says "here".
- **Globe (cols 7–12):** §5.3. It bleeds past the right edge by ~12 % so the sphere reads as a
  horizon, not a sticker. Below `wide` it sits under the CTA cluster at 72 vw, centred, not bleeding.

### 5.2 Ambience layers (`Ambience.tsx`, `aria-hidden`, `position: absolute; inset: 0`)

1. **Graticule** — `.graticule` (48px module, major every 4th), radial `mask-image` fading to
   nothing at the viewport edges. It is the same graticule the globe's shader draws on the sphere
   (§5.3), so the flat grid and the spherical grid read as one coordinate system.
2. **Glow** — a 900px radial of `--color-glow` centred behind the globe (78 % / 40 %); a second 420px
   glow at 12 % / 82 % at half alpha for a horizon. Drift per §7.
3. **Reticle** — one at the headline's leading cap-height (the focal lock), two at graticule
   intersections near the globe at 30 % alpha. `wide` and below: headline reticle only.
4. **Scan-line** — 1px sand at 25 % alpha, sweeps once on entry (§7), then removed from the DOM.

### 5.3 The globe — `three` + `@react-three/fiber`

**Concept.** Not a photoreal Earth. A *ground-station plot*: a dark sphere carrying the same sand
graticule as the page, continents as a sparse terracotta point field, a fresnel rim in sand, and
two–three satellites whose orbits leave fading trails. It is the product's coordinate system made
tangible, and the satellites are the *sensors the product listens to* (Sentinel-2 at ~786 km
sun-synchronous, Sentinel-1 at ~693 km, and one VHR at ~500 km — the inclinations below are the
real ones, scaled to the sphere).

**Dependencies.** `three` (≈ 150 KB gz), `@react-three/fiber`, `@react-three/drei` (import
`OrbitControls` and `Line` only — never the barrel). All three land in one chunk (`globe`) via
`build.rollupOptions.output.manualChunks`, loaded by `React.lazy(() => import('./globe/Globe'))`
inside `<Suspense fallback={<GlobePoster/>}>`. The console never imports the chunk.

**Support detection (once, in `Globe.tsx` before `<Canvas>` mounts):**

```ts
function detectWebGL(): GlobeSupport {
  try { const c = document.createElement('canvas'); return c.getContext('webgl2') || c.getContext('webgl') ? 'webgl' : 'none' }
  catch { return 'none' }
}
```

`'none'`, `prefers-reduced-motion`, `navigator.deviceMemory < 2`, or `saveData` → `GlobePoster`
stays. The poster is a *finished* still, not a spinner: an SVG of the same sphere (graticule
ellipses, rim, one satellite with a static trail) at the same size, so a machine without WebGL gets
the composition, not an apology.

**Scene graph.**

```tsx
<Canvas
  dpr={[1, 1.75]}                       // never 3× on a laptop retina beside a VLM
  frameloop={globeVisible ? 'always' : 'never'}
  gl={{ antialias: true, alpha: true, powerPreference: 'low-power' }}
  camera={{ position: [0, 0.35, 3.1], fov: 34 }}
  onCreated={({ gl }) => gl.setClearColor(0x000000, 0)}   // the page ground shows through
  aria-hidden                            // the h1 and standfirst carry the meaning
>
  <Suspense fallback={null}>
    <Earth />                            // radius 1
    <Satellite orbit={ORBITS.s2} />      // inclination 98.6°, r 1.123, period 44 s (scaled)
    <Satellite orbit={ORBITS.s1} />      // inclination 98.2°, r 1.109, period 40 s, phase +2.1 rad
    <Satellite orbit={ORBITS.vhr} />     // inclination 97.4°, r 1.078, period 36 s, phase +4.3 rad
    <Controls />
  </Suspense>
</Canvas>
```

**Earth (`Earth.tsx`).** `IcosahedronGeometry(1, 5)` (10 242 verts — enough for a smooth rim, cheap
enough for integrated GPUs). One `ShaderMaterial`, `transparent`, `depthWrite: true`:

| Uniform | Source | Meaning |
|---|---|---|
| `uGround` | `tokens.bgMain` (linear) | sphere base — darker than the page by L −0.03 so it reads as a body |
| `uGrid` / `uGridMajor` | `tokens.sand` at 0.07 / 0.14 | graticule lines every 10° / 30°, drawn in the fragment shader from spherical UV, `fwidth`-antialiased |
| `uRim` | `tokens.sand` | fresnel `pow(1 - dot(N, V), 3.0)` × 0.55 — the atmosphere |
| `uLand` | `tokens.terracotta` | continents: a 1024×512 single-channel land mask (`public/samples/globe/land.png`, ≤ 40 KB, the team renders it from Natural Earth) sampled and stippled — `step(0.5, land) * dotgrid(uv, 180)` — so land is a field of points, not a fill |
| `uTerminator` | vec3 sun dir, rotates 1 rev / 180 s | night side dims land points to 35 %; nothing is black — the page is already dark |
| `uTheme` | 0 dark / 1 light | on light theme `uGround` → `tokens.paper` L −0.04 and grid alpha ×1.6 |

No lights, no PBR, no shadow maps. The whole material is one fragment shader; the vertex shader
passes normal + world position.

**Satellites (`Satellite.tsx`).** Low-poly: body `BoxGeometry(0.018, 0.012, 0.026)` in
`MeshBasicMaterial({ color: tokens.sand })`, two panels `PlaneGeometry(0.06, 0.016)` in
`tokens.terracotta` at 55 % opacity, `DoubleSide`. Position from `orbits.ts`:

```ts
export interface Orbit { r: number; inclDeg: number; raanDeg: number; periodS: number; phase: number }
export function positionAt(o: Orbit, t: number, out: Vector3): Vector3 {
  const a = o.phase + (2 * Math.PI * t) / o.periodS
  out.set(Math.cos(a) * o.r, 0, Math.sin(a) * o.r)             // circular orbit in its plane
  out.applyAxisAngle(X_AXIS, (o.inclDeg * Math.PI) / 180)      // tilt by inclination
  out.applyAxisAngle(Y_AXIS, (o.raanDeg * Math.PI) / 180)      // rotate the ascending node
  return out
}
```

`useFrame((_, dt) => …)` advances a per-satellite clock (`clock += Math.min(dt, 1/30)` — a tab that
was hidden must not teleport the satellite), sets `mesh.position` from `positionAt`, and
`mesh.lookAt(0,0,0)` so the panels face nadir. No allocations in the loop: the `Vector3` is reused.

**Trails.** A ring buffer of 96 positions per satellite, written every other frame, rendered with
drei `<Line>` (`Line2`, `linewidth: 1.2` px, `vertexColors`) whose colour array fades from `sand`
at α 0.55 (head) to α 0 (tail). Additive blending, `depthTest: true` so the trail passes behind the
sphere. Total: 3 × 96 verts updated per frame — negligible. While `dragOver` is true, head alpha
rises to 0.9 and trail length doubles for 600 ms — the satellites acknowledge the drop.

**Controls (`Controls.tsx`).** drei `OrbitControls` with:

| Prop | Value | Why |
|---|---|---|
| `enablePan` | `false` | the globe stays centred |
| `enableZoom` | `false` | **wheel must scroll the page.** A hero that eats the wheel is the classic 3D-landing failure |
| custom pinch / `+` `−` | `minDistance 2.7 · maxDistance 3.5` | limited zoom on touch and via two 28px glass buttons at the sphere's lower-right; keyboard `+`/`−` when the canvas is focused |
| `rotateSpeed` | `0.35` | gentle |
| `minPolarAngle / maxPolarAngle` | `π/2 − 0.55 · π/2 + 0.35` | you can tilt to see the poles' graticule convergence, not flip under it |
| `autoRotate` / `autoRotateSpeed` | `true` / `0.28` | ~ 1 rev / 3.5 min; pauses while dragging, resumes after 2.5 s |
| `enableDamping` / `dampingFactor` | `true` / `0.06` | inertia, so a flick spins and settles |
| `touches.ONE` | `ROTATE`; `touches.TWO` → `DOLLY` only | a one-finger vertical swipe over the canvas on a phone still needs to scroll: the canvas wrapper sets `touch-action: pan-y` and rotation engages only on horizontal intent (`|dx| > |dy|` over the first 8 px) |

The wrapper `<div>` is `tabIndex=0` with `aria-label="Interactive globe. Drag to rotate; press plus
or minus to zoom."` and `role="img"`; arrow keys rotate 12° per press when focused. The cursor is
`grab` / `grabbing`.

**Visibility + budget (`useGlobeVisibility`).** `IntersectionObserver` on the wrapper at 0.1
threshold sets `globeVisible`; `frameloop` flips to `'never'` when it leaves — the hero's loops
stop the moment the pipeline story is on screen. `document.visibilitychange` does the same. Target:
≤ 4 ms/frame on an Intel Iris Xe at 1.75 dpr; measured once in the inspection round, and if it
misses, `dpr` drops to `[1, 1.25]` before any material is simplified.

**Theme.** `useThemeStore.resolved` is read once per change via `useEffect` → sets `uTheme` and the
satellite material colours. The canvas has `alpha: true`, so the page ground is the sphere's
background in both themes.

**Tokens → linear sRGB (`tokens.ts`).** three.js wants linear RGB. The file exports the four
colours as `[r, g, b]` triplets computed *once at build time* from the OKLCH values in `theme.css`
(the OKLab → linear-sRGB matrices are the same ones `contrast.test.ts` gains in `frontend_blueprint`
§0.6). A test asserts each triplet round-trips to the token hex within 1/255, so a token change that
forgets the globe fails CI.

```ts
export const tokens = {
  bgMain:     [0.0046, 0.0032, 0.0028],  // oklch(0.150 0.012 60)  #100c0a
  sand:       [0.7379, 0.3925, 0.1878],  // oklch(0.770 0.090 65)  #dfa878
  terracotta: [0.4910, 0.1620, 0.0800],  // oklch(0.600 0.110 45)  #ba704f
  paper:      [0.9216, 0.8879, 0.8310],  // light-theme ground
} as const satisfies Record<string, readonly [number, number, number]>
```

(Values are illustrative; the build derives them.)

### 5.4 Proof strip (`ProofStrip.tsx`)

Reads `/v1/health` and `/v1/registry` with the console's `useQuery` keys (shared cache). Renders
only what is known: `● System Ready · 13/15 tools · schema 1.0 · <device.name>`. API down →
`API offline — the console still opens with recorded fixtures` and the CTA appends `?mock=1`. No
placeholder numbers, ever.

---

## 6. Below the fold

### 6.1 Agentic pipeline story (`PipelineStory.tsx`)

`<section aria-labelledby="story-title">`, `<h2 class="t-section">` **What happens between the
question and the number.** Lede, verbatim from `AGENT_POLICY_DAG.md` §0: *"The LLM performs slot
filling and answer synthesis only. It never chooses a tool, never orders steps, and never invents a
capability."*

The brief's four stages mapped to what the backend does. *Orthorectification* is not performed, so
stage 3 is named for what runs (open decision #3).

| # | Stage title | Sub-label (`.t-coord`) | What animates | Truth source |
|---|---|---|---|---|
| 1 | **Raster ingest & pre-flight** | `POST /v1/validate · 11 checks` | eleven check pills fill left→right; one goes amber — the system reports, it does not hide | `API_CONTRACT.md` §4.5, `PreflightPanel.tsx` |
| 2 | **Spatial policy DAG** | `policy_key = TaskType\|PairType\|Modality` | a policy key types itself in mono, resolves, and a 5-node DAG draws edge-by-edge (dagre layout, static SVG — not `@xyflow`) | `AGENT_POLICY_DAG.md` §1, the `plan` SSE event |
| 3 | **Co-register · render · mask** | `spectral_renderer → semantic_segmenter` | an A/B pair slides into registration (2px → 0), TC → NDVI → CHANGE tabs light in turn, a mask fills at 12 % | `render/view_labels.py`, `evidence/views.ts` |
| 4 | **Evidence-bound synthesis** | `vlm_change_vqa → CitationValidator` | a sentence streams; two numbers become sand citation pills that "pull" toward the KPI above; one gets the amber wavy `uncited` underline and keeps it | `GroundedAnswer.tsx`, contract §2.10 |

Horizontal rail on `desk` (1px `--color-line` connector with a travelling 2px sand dash), vertical
below. Each stage plays once on entering the viewport (`useReveal`, 25 %); a `Replay` ghost button
at the rail's end runs `replayStory()`. An always-present `<ol class="sr-only">` lists the four
stages in prose for screen readers and for a failed script.

### 6.2 Capabilities showcase (`Capabilities.tsx`)

`<h2 class="t-section">` **Three things a chat box cannot do.** Three chips (`role="tab"`, roving
tabindex, hover-preview after 120 ms on pointer devices, click commits) and one preview stage
(`role="tabpanel"`, `.card-flush`, 16:10):

| Chip | Task types | Preview shows | Copy (`.t-lede`) |
|---|---|---|---|
| **Multimodal change detection** | `CHANGE_VQA`, `CHANGE_MAP` | A/B swipe with CHANGE as B, the changed-area KPI, `siamese_change_detector` + `change_statistics` chips | "Two dates, one co-registered pair, and a change mask whose area is a measured scalar, not a guess." |
| **SAR / optical cross-modal consistency** | `CROSS_MODAL_COMPARE` | optical TC beside SAR VV/VH, a `physics_agreement` verdict pill, the disagreement mask | "Backscatter and reflectance answer separately; `physics_agreement` says where they disagree, and the answer tells you." |
| **Sub-pixel grounding** | `GROUNDING`, `COUNT` | a VHR frame with three `BboxOverlay` boxes, a reticle locked on one, the count KPI | "Boxes are validated as positions against the raster grid before the VLM is allowed to cite them." |

Previews are `public/samples/capabilities/<chip>/*.webp` with `alt` text that names the scene and
the overlay (`"Sentinel-2 true-colour pair over Bengaluru with the change mask on the right half"`).
Chip change: outgoing 120 ms fade, incoming `sq-arrive` 260 ms. `Try this in the console →` calls
`proposeQuestion(chip.sampleQuestion)` then navigates.

### 6.3 Closing CTA + footer

One `.t-section` line — **Bring your own scene.** — one `.btn-primary` `Open the console`, the
graticule fading out beneath it so the page ends on plain ground. Footer: wordmark (a `<p>`, not a
heading), `Smart India Hackathon 2026 · PS 26167 · ISRO / SAC`, links `Use cases · Maps · Docs`.

---

## 7. Motion & pacing

**Thesis.** One focal moment — *acquisition*: the ground station locks on and the globe comes up.
It runs once per visit in ≈ 1.2 s; everything after is feedback, continuity, or the globe's own
quiet physics.

**Focal sequence (entry):**

| t (ms) | Layer | Animation |
|---|---|---|
| 0 | graticule | opacity 0 → 1, 600 ms `--ease-out-quint` |
| 0 | scan-line | translateY(−2 %) → 102 %, 720 ms linear; removed on `animationend` |
| 120 | headline reticle | scale 1.6 → 1, opacity 0 → 1, 480 ms — the lock |
| 180 | eyebrow | opacity + translateY(6px → 0), 320 ms |
| 240 / 320 | headline lines 1 / 2 | `clip-path: inset(0 100% 0 0)` → `inset(0)`, 420 ms — a wipe, not a fade-up |
| 460 | standfirst, CTA cluster | opacity + translateY(8px → 0), 320 ms, 40 ms stagger |
| 520 | **globe** | `GlobePoster` is already painted. When the `globe` chunk resolves, `<Globe>` mounts with the wrapper at opacity 0 and cross-fades over the poster in 400 ms; the sphere's `uRim` ramps 0 → 0.55 over 900 ms and the satellites' trails grow from 0 to full length over the same 900 ms — the plot "comes online". If the chunk resolves before 520 ms, the mount waits for it; if after, it plays whenever it lands. |
| 700 | proof strip | opacity, 300 ms |
| 800 | glow | drift loop starts; `phase` → `'live'` |

**Ambient loops (only while `globeVisible`; `animation-play-state: paused` otherwise):**

- Glow drift: `translate` waypoints `(0,0) → (3%,−2%) → (−2%,1%) → (0,0)`, 18 s, ease-in-out.
- Reticle breath: opacity 0.40 ↔ 0.55, 4 s.
- Globe: auto-rotate 0.28, satellites on their periods, terminator 180 s. Nothing else.
- The flat graticule never moves.

**Scroll reveals (`useReveal`):** sections below the hero animate once at 25 % visibility: opacity
0 → 1, translateY(12px → 0), 320 ms `--ease-out-quint`; `data-reveal-child` staggers at 40 ms, capped
at 240 ms total. Pipeline stages use their own sequence instead.

**Feedback:** buttons 120 ms; tab change 220 ms; nav underline grows from the left in 160 ms; the
theme toggle's sun/moon mask 220 ms.

**Reduced motion:** no scan-line; graticule, reticle, reveals and headline wipes become 160 ms opacity
fades; the globe stays as `GlobePoster` (still frame, satellites static) with a small `Play globe`
ghost button that opts in; pipeline stages render final frames with `Replay` available.

**Budget:** one `backdrop-filter` region (the nav), one WebGL canvas at ≤ 1.75 dpr running only while
visible, zero other canvases. LCP must be the `h1` text: the poster SVG is inline (no fetch), the
`globe` chunk is `modulepreload`ed from `Landing.tsx` after first paint, never from `index.html`.

---

## 8. Suspense & chunk map for this page

| Boundary | Chunk | Fallback | Preload |
|---|---|---|---|
| `App.tsx` → `<Landing>` | `landing` (page + copy + Ambience + story + capabilities, ≈ 25 KB gz) | full-page: ground colour + graticule + the glass nav skeleton (no spinner) | none — it *is* the entry for `/` |
| `Hero` → `<Globe>` | `globe` (`three` + R3F + drei subset + shaders, ≈ 190 KB gz) | `<GlobePoster/>` — a finished still | `import()` kicked off in `useEffect` after first paint, and only if `detectWebGL() === 'webgl'` and not reduced-motion |
| `PipelineStory` stage 2 | inline (dagre is already in the console bundle; the story imports only the layout function, tree-shaken) | final-frame SVG | — |

`vite.config.ts`:

```ts
build: {
  sourcemap: false,
  rollupOptions: { output: { manualChunks: {
    globe: ['three', '@react-three/fiber', '@react-three/drei'],
    map:   ['maplibre-gl'],
    dag:   ['@xyflow/react', 'dagre'],
  } } },
},
assetsInclude: ['**/*.glsl'],
```

A `vitest` test asserts that `src/App.tsx`, `src/components/**` and `src/pages/!(landing)/**` never
statically import from `three`, `@react-three/*` or `maplibre-gl` (a `grep`-style source test — the
cheapest bundle guard that exists).

---

## 9. Accessibility on this page

- Exactly one `h1` (the headline); sections are `h2`; nothing skips a level.
- The canvas is `aria-hidden`; its wrapper is a focusable `role="img"` with a label and keyboard
  rotation. The meaning of the hero is in text.
- All drop-zone affordances have a button equivalent (`Drop imagery` is also a file input).
- Every image has a descriptive `alt`; decorative ambience is `aria-hidden` and CSS-only.
- Contrast: Odida at display size is "large text" — `text-accent-warm-text` on `bg-main` (≥ 9:1)
  and, on the light theme, the light `--color-accent-warm-text` (≥ 7:1); the standfirst at
  1.25 rem is *not* large text and stays `text-text-hi`.
- The proof strip is `role="status"` so a screen reader hears the health line once.

---

## 10. Assets the team must supply

- `public/fonts/Odida.ttf` (and the derived `Odida.woff2`) + measured fallback metrics (§3).
- `public/samples/globe/land.png` — 1024×512 equirectangular land mask, 1-bit, ≤ 40 KB.
- `public/samples/capabilities/{change,crossmodal,grounding}/*.webp`.
- `public/og/landing.png` 1200×630 — render the poster SVG over the graticule with the headline.

## 11. Open decisions (not for the builder to invent)

1. Headline copy variant (§5.1).
2. Odida letter-spacing and fallback metrics — set after the file is in hand.
3. Two or three satellites (three specified; drop the VHR one if the frame budget misses).
4. Whether the land mask is acceptable on the demo network (it is a local asset, so yes by default).
