# Landing page revamp — implementation blueprint

Scope: items 1–5 of the master polish list, for `frontend/src/pages/Landing.tsx`
and its children.

> **Status (2026-09-13): implemented.** §1–§5 are in the tree. The §7 decisions
> were taken as: (1) Geist across the whole app — `--font-sans` is Geist and
> `--font-display` aliases it; Inter is gone. (2) The JS fallback is built:
> `useScrollProgress` writes `--p` on the hero and drives the same properties
> the `@supports (animation-timeline)` block hands to the compositor, so
> Firefox gets the hero parallax, the beam loop and the globe reactivity.
> (3) The recording carries no aircraft boxes and no `physics_agreement`
> verdict, so `mocks/grounding.ts` is a **declared synthetic** fixture
> (self-consistent geometry, `grounding.test.ts`) and the cross-modal count is
> derived from the drawn regions; both HUDs say `synthetic` on the readout.
> `pages/landing/evidence.ts` pins the recorded change figures to the JSON
> (`evidence.test.ts`). Deviations from the spec below, each for a reason
> seen in the render: the headline runs three lines at 1280 (23ch will not
> fit 7 columns at 88 px — the two-line claim in §2.5 was wrong); the
> cross-modal seam runs (520,0)→(360,400) so the disagreeing region actually
> straddles it; the closing graticule drifts by `transform` rather than
> `mask-position` (the mask must stay put for the fade to hold); on a phone
> the stages read as a timeline with the text beside the badge, and the HUD's
> bottom-right readout is hidden below `wide`.
It supersedes §3 (typography), §6.1–6.2 and §7 of `landing_page_blueprint.md`
where the two disagree, and leaves the rest of that document standing.

Non-negotiables carried over from `PRODUCT.md` and the codebase:

- **Offline.** Every font is self-hosted from `public/fonts/`; nothing loads from a CDN.
- **Budget.** The console entry stays ≤ 180 KB gz (`npm run check:bundle`). Everything
  here lands in the `Landing` chunk (currently 9.6 KB gz) or the `globe` chunk. No new
  dependency: no GSAP, no Framer Motion, no Lenis. Scroll-driven work is CSS
  `animation-timeline` with a small `IntersectionObserver`/`rAF` fallback.
- **Reduced motion is a first-class path**, not a kill switch: every effect below names its
  still alternative, and `src/styles/__tests__/motion.test.ts` keeps holding.
- **Nothing invented.** All numbers on the page still come from
  `mocks/captured/events.bitemporal.json`; the new HUDs draw evidence, they do not fabricate it.
- **Tokens only.** Colour via `--color-*` OKLCH tokens from `theme.css`; the contrast suite
  (`contrast.test.ts`) parses that file, so any new colour token goes through `@theme`.

Sequence the work as: §2 typography → §1 pipeline layout → §4 tabs → §5 HUDs → §3 motion.
Type and layout first because every motion value below is measured against the new rhythm.

---

## 1. "How it works" pipeline — un-claustrophobing `PipelineStory.tsx`

### 1.1 Diagnosis (measured against the current render)

| Symptom | Cause in code |
|---|---|
| Four columns feel jammed at 1280 | `desk:grid-cols-4 desk:gap-6` → four 292 px tracks with 24 px gutters inside a 1280 px max; the number badge, title, sub, body and diagram all stack with `gap-4` (16 px), so the column reads as one dense slab. |
| Diagrams collide with the text above them | `<div className="mt-auto">` pins the diagram to the column bottom; when body copy is short the gap is large, when long it is 16 px. Rhythm varies per column, which is worse than being uniformly tight. |
| The "beam" line sits on top of the badges | `top-5` (20 px) is the badge's centre only at exactly 40 px badge height; the dashed run is `12 240` on a `100%` line, so its dash count depends on viewport width. |
| Heading → grid gap too small relative to the grid's own gaps | `mt-14` (56 px) heading-to-grid vs `gap-10` (40 px) between rows on `wide`. The rule is *more space above a group than inside it*; 56 vs 40 barely says so. |

### 1.2 Target layout

**Section frame** (unchanged container, new vertical rhythm):

```
py-24            →  py-28 desk:py-36          (112 / 144 px)
heading block    →  max-w-[42ch] for the h2, lede stays 60ch
heading → grid   →  mt-16 desk:mt-20          (64 / 80 px)
grid → replay    →  mt-12                     (48 px)
```

**Grid** (`<ol>`):

```
grid-cols-1 gap-14                 phone   (56 px between stages)
wide:grid-cols-2 wide:gap-x-10 wide:gap-y-16    (40 / 64 px)
desk:grid-cols-4 desk:gap-x-12     (48 px gutters → 4 × 268 px tracks at 1280)
```

48 px gutters are the minimum at which four columns of 13.5 px body copy read as four
columns rather than one paragraph with rivers. The tracks shrink from 292 → 268 px, which
is fine: body copy is capped at `max-w-[30ch]` below, so the track never has to be wide.

**Inside a stage** (`<li>`), replace the single `gap-4` stack with three explicitly spaced
groups — *marker*, *text*, *diagram* — because they are three different things:

```
<li class="relative flex flex-col">
  <span class="badge …" />                 <!-- 40 px, unchanged -->
  <div class="mt-6">                       <!-- 24 px: marker → title -->
    <h3 class="t-panel text-[15px] …" />
    <p  class="t-coord mt-2 …" />          <!-- 8 px: title → sub (was 6) -->
    <p  class="mt-4 max-w-[30ch] …" />     <!-- 16 px: sub → body (was 12); measure capped -->
  </div>
  <div class="mt-8 min-h-[128px]">         <!-- 32 px: text → diagram, fixed slot -->
    <Part played={on} />
  </div>
</li>
```

- Drop `mt-auto`. Give the diagram a **fixed slot** (`min-h-[128px]`, the tallest Part is
  Stage3 at 96 + 8 + 22 = 126 px) so every column's diagram starts on the same baseline
  and the 32 px text→diagram gap is constant across all four. Stage4's paragraph is the
  one Part taller than the slot at narrow widths; cap it at `max-w-[28ch]` and let the
  slot grow — the *others* still align, and the tallest defines the row.
- Stage title: `text-[15px]` → `text-[16px] leading-[1.25]` (`t-panel` is 14 px; the
  landing page may run one size larger than the console).
- Body: `text-[13.5px] leading-relaxed` → `text-[14px] leading-[1.6]`, `max-w-[30ch]`.
- Sub (`t-coord`): unchanged size, but `opacity` no longer varies; colour `text-text-lo`.

**The beam** (the `<li aria-hidden>` line):

- Position: `top-5` → `top-[20px]` is the same value; the real fix is to make the badge
  40 px *exactly* (`size-10` is already 40 px) and give the line `h-px` at `top-[19.5px]`
  so it passes through the badge centre. Under each badge, paint a 48 px wide
  `bg-bg-main` gap so the beam appears to pass *behind* the badge, not through it: add
  `before:` on the badge — `before:absolute before:inset-y-1/2 before:-inset-x-1 before:h-px before:bg-bg-main`
  — or simpler, give the badge `outline outline-[6px] outline-bg-main`.
- Dash: `strokeDasharray="12 240"` → use `pathLength={1000}` on the line and
  `strokeDasharray="14 986"`; the animated dash then travels once per 6 s regardless of
  viewport width, and the `sq-dash` keyframe animates `stroke-dashoffset` from 1000 → 0.
- The beam is the *only* infinitely-looping animation in the section. Keep it; it is the
  pipeline's "the machine is on" signal. Under reduced motion it becomes a static dashed
  `4 8` line.

**Phone (`< wide`)**: the stages stack; the beam is hidden (`desk:block` already). Add a
2 px vertical rule on the left, from badge to badge, so the sequence still reads as one
line: `wide:hidden absolute left-5 top-10 bottom-0 w-px bg-line`, per `<li>` except the
last. Stage `gap-14` gives that rule room to be seen.

### 1.3 Type inside the diagrams

The Parts use `chip` (11 px mono) for the eleven checks and four tabs; at 268 px track
width the eleven check chips wrap to four lines. Fine — but give the `<ul>` `gap-x-1.5
gap-y-2` so the rows have air, and switch the chips to `whitespace-nowrap` (already the
default in `.chip`).

### 1.4 Acceptance

- At 1280 px: four tracks of 268 px, 48 px gutters, all four diagrams' top edges within
  1 px of each other (computed `offsetTop`).
- Heading-to-grid (80) > row gap (64) > text-to-diagram (32) > marker-to-title (24) >
  title-to-sub (8): spacing reads as a hierarchy.
- Body measure ≤ 30ch, no orphan lines under 4 words in the four `body` strings
  (rewrite copy if one appears; the strings are yours).

---

## 2. Typography overhaul — Odida out, Geist in

### 2.1 What goes

- `@font-face 'Odida'` and `'Odida Fallback'` blocks in `theme.css` (lines ~170–195).
- `--font-odida` from `@theme`.
- `public/fonts/Odida.ttf`, `public/fonts/Odida.woff2`.
- `"fonts:odida"` script in `package.json`.
- The `<link rel="preload" … Odida.woff2>` in `Landing.tsx`.
- The test `keeps the Odida face on the landing page` in `bundle-boundaries.test.ts` —
  replace it with `keeps the display face on the landing page` asserting the same for
  `t-display|--font-display` (see 2.4).
- `landing_page_blueprint.md §3` becomes historical; point it at this document.

### 2.2 What comes in

Two variable faces, both already resolvable offline:

| Role | Family | Source | File in `public/fonts/` |
|---|---|---|---|
| Display + UI sans | **Geist Variable** (wght 100–900) | `@fontsource-variable/geist` (add as devDependency, same publisher as the existing `geist-mono`) | `geist-latin-wght-normal.woff2`, `geist-latin-ext-wght-normal.woff2` |
| Mono | **Geist Mono Variable** | already installed | already shipped |

Extend `"fonts:sync"` in `package.json` to copy the two Geist files alongside the Inter
and Geist Mono ones. Inter stays as the **console** body face — this brief is the landing
page, and PRODUCT.md pins Inter + Geist Mono as the incumbent identity. Geist is the
*display voice* and the landing page's body voice; the console is untouched.

> Decision to confirm with the team (one line, not a blocker): whether to also move the
> console's `--font-sans` to Geist. It is a one-token change and Inter/Geist are
> metrically close enough that the fallback `size-adjust` needs re-measuring only once.
> Default: **no** — keep Inter in the console, Geist on the landing page.

### 2.3 Tailwind v4 tokens (`@theme` in `theme.css`)

```css
@font-face {
  font-family: 'Geist Variable';
  font-style: normal;
  font-weight: 100 900;
  font-display: swap;
  src: url('/fonts/geist-latin-wght-normal.woff2') format('woff2-variations');
  unicode-range: /* same Latin range as the Inter block */;
}
/* + the latin-ext block, same shape */

@font-face {
  font-family: 'Geist Fallback';
  src: local('Arial'), local('Liberation Sans'), local('Helvetica Neue'), local('Segoe UI');
  /* Measure with `npx fontpie public/fonts/geist-latin-wght-normal.woff2 -f Arial`.
     Expected neighbourhood: size-adjust ≈ 104%, ascent ≈ 91%, descent ≈ 22%.
     Do not ship guessed values: a wrong ascent-override is a visible headline jump. */
  size-adjust: …; ascent-override: …; descent-override: …; line-gap-override: 0%;
}

@theme {
  --font-display: 'Geist Variable', 'Geist Fallback', 'Inter Variable', ui-sans-serif, sans-serif;
  --font-landing: var(--font-display);   /* body voice on `/` only */
  /* --font-sans and --font-mono unchanged */
}
```

`Landing.tsx` root gets `font-[var(--font-landing)]` (or a `.landing` class setting
`font-family`), so every unstyled paragraph on the page picks up Geist without touching
the console.

### 2.4 The display scale — replacing `.t-display`, `.t-display-sub`, `.t-section`, `.t-lede`

Current problems, named: Odida at `line-height: 0.94` with `-0.02em` tracking gave the
headline a compressed, slightly crooked kerning at large sizes; `t-section` at `-0.025em`
on Inter 600 crowds the counters; `tracking-tight` is used ad hoc in three places
(`Hero`, `ProofStrip`, `LandingNav`) with no scale behind it.

New scale, all `font-family: var(--font-display)`, `font-synthesis: none`,
`text-wrap: balance` on headings, `font-feature-settings: 'ss01', 'cv11'` (Geist's
single-storey a / straight-leg y set — use them; it is what makes Geist look like Geist
rather than Inter):

| Class | Size | Line-height | Tracking | Weight | Notes |
|---|---|---|---|---|---|
| `.t-display` | `clamp(2.75rem, 1.2rem + 5.6vw, 5.5rem)` (44 → 88 px) | `1.02` | `-0.035em` | 600 | Two lines max at 1280; the second line keeps `text-accent-warm-text`. `max-width: 14ch`. |
| `.t-display-sub` | `clamp(1.125rem, 0.9rem + 0.9vw, 1.375rem)` (18 → 22 px) | `1.4` | `-0.01em` | 400 | `max-width: 34ch`, `color: text-lo` (was text-hi — the sub should not compete with the h1). |
| `.t-section` | `clamp(1.875rem, 1.2rem + 2.4vw, 3rem)` (30 → 48 px) | `1.06` | `-0.03em` | 600 | `max-width: 24ch`. |
| `.t-lede` | `1.0625rem` (17 px) | `1.55` | `0` | 400 | `max-width: 58ch`. |
| `.t-stage` (new) | `1rem` | `1.25` | `-0.015em` | 600 | The four stage titles and three capability labels. |
| `.t-coord` | unchanged (mono) | | `0.08em` | 500 | Readouts; stays Geist Mono. |

Tracking floor is `-0.04em` (craft floor); `-0.035em` at 88 px is the tightest value here
and is applied *only* above 44 px. Below `wide`, `.t-display` steps its tracking back to
`-0.025em` via a media query — negative tracking that is right at 88 px is wrong at 44.

Kerning: set `font-kerning: normal` and `font-optical-sizing: auto` on the landing root.
Geist has no optical-size axis, so the second line is a no-op safety; the first fixes the
"awkward kerning" that came from Odida's sparse kern table, not from Tailwind.

Remove every stray `tracking-tight` / `tracking-[-0.01em]` on the page in favour of the
classes above; `oxlint` will not catch these, so grep for `tracking-` under
`src/pages/landing/` and expect zero hits when done.

### 2.5 Hero-specific

- Headline copy is two lines by design. At 1280 px, "From space to answers —" / "with the
  receipts." should fit in `7/12` columns at 88 px: 14ch max-width holds it. If the em
  dash orphans on a narrow desk width, drop it from line 1 and let line 2 carry it
  ("— with the receipts.").
- The `data-hero="line-1|line-2"` wipe animation stays; at `line-height: 1.02` the wipe's
  `clip-path` inset must include 4 px of overscan top and bottom so descenders (p, y) are
  not clipped during the reveal: `clip-path: inset(-4px 100% -4px 0)` → `inset(-4px 0 -4px 0)`.
- Sub copy: 34ch at 22 px is ~ 470 px, which sits under the h1 without touching the globe.

### 2.6 Preload and swap

`Landing.tsx` preloads `geist-latin-wght-normal.woff2` (replacing the Odida preload).
Inter stays preloaded from `index.html` for the console. Both are `font-display: swap`
with metric-matched fallbacks, so first paint occupies the right box.

### 2.7 Acceptance

- `grep -r Odida frontend/ --exclude-dir=node_modules` → 0 hits.
- `npm run fonts:sync` places four Geist files; `ls public/fonts` shows no `.ttf`.
- Lighthouse "font-display" audit passes; no CLS from the swap at 1280 and 390 px.
- Contrast test still passes (no colour changed).

---

## 3. Top-tier motion

One authored moment already exists — the hero's acquisition timeline (`[data-hero]`).
This plan adds **three systems** that extend it down the page rather than three new
"effects": a scroll-driven timeline, globe scroll-reactivity, and cursor-reactive
lighting. Each is transform/opacity/filter/mask only; nothing touches layout.

### 3.1 Scroll-driven system (CSS `animation-timeline`, with fallback)

**Mechanism.** Replace the once-only `useReveal` entrance with *scroll-linked* motion where
it earns it, and keep the once-only reveal where scrubbing would be noise.

| Element | Behaviour | Implementation |
|---|---|---|
| Hero copy block | Parallax up and fade as the hero leaves: `translateY(0 → -60px)`, `opacity 1 → 0` over the range *hero top → hero 70 % scrolled*. | `animation: sq-hero-exit linear both; animation-timeline: view(); animation-range: exit 0% exit 70%;` on the copy column. |
| Ambience glow (`data-hero="glow"`) | Drifts with scroll on top of its 18 s idle drift: `translateY(0 → 120px)` across the hero's exit. | Second animation on the same element; the two compose because `sq-drift` animates `transform` via `translate` and the scroll one uses `--glow-scroll` written into a `translate` custom property — use CSS `translate:` (individual transform property) for one and `transform:` for the other so they do not fight. |
| Section headings (`data-reveal`) | Keep the once-only rise. Scrubbing a heading is noise. | Unchanged. |
| Pipeline beam | Dash offset **also** advances with scroll: the beam travels as the reader travels. | `animation-timeline: view(block)` on the SVG line with `animation-range: cover 0% cover 100%`, keyframes `stroke-dashoffset 1000 → 0`. The 6 s idle loop is removed in favour of scroll-linkage; under `(animation-timeline)` unsupported, the loop stays. |
| Stage diagrams | Trigger stays `IntersectionObserver` (they are *play-once* sequences with internal timing). Add a scroll-linked `clip-path: inset(0 0 100% 0 → 0)` on the diagram frame so each diagram "develops" as it enters. | `animation-range: entry 0% entry 60%`. |
| Capability preview frame | `scale(0.96 → 1)`, `filter: blur(6px → 0)` over entry. Blur is allowed here because the element is one card, `contain: paint`, and stops at 0. | `animation-range: entry 0% entry 50%`. |
| Closing CTA graticule | Mask moves: `mask-position` scrubbed so the grid appears to scroll under the copy at half speed. | `animation-timeline: view()` on the mask wrapper. |

**Fallback.** `@supports not (animation-timeline: view())` → the scrolled properties fall
back to the existing `data-reveal` once-only behaviour; a 30-line
`useScrollProgress(ref)` hook (rAF-throttled, writes `--p` 0…1 on the element) drives the
hero parallax only, because that is the one place a static fallback is visibly worse.
Safari ≥ 26 and Chromium support `animation-timeline`; Firefox behind a flag — the
fallback matters for the judge's laptop if it runs Firefox.

**Reduced motion.** Every `animation-timeline` rule sits inside
`@media (prefers-reduced-motion: no-preference)`. Under `reduce`, the page is the current
page minus the beam loop. The motion test's assertions (`RUNNING` dot stilled, transitions
kept) are unaffected.

**Performance guard.** Everything scroll-linked is on the compositor (`transform`,
`opacity`, `filter`, `mask-position`, `stroke-dashoffset` on an SVG with
`will-change: stroke-dashoffset` is *not* compositable — accept it, it is one line
element). Budget: 0 layout invalidations during scroll, checked with the Performance
panel's "Layout Shift" and "Recalc" rows over a full-page scroll.

### 3.2 Globe scroll-reactivity (`globe/Controls.tsx`, `globe/Earth.tsx`, `state/landing.ts`)

The globe currently auto-rotates at `0.28` and drifts its terminator once per 180 s. Make
it *answer* the scroll:

- **Store.** Add `scroll: number` (0…1, hero exit progress) to `useLandingStore`, written
  by the `useScrollProgress` hook from 3.1 attached to the hero section. R3F components
  read it via `useLandingStore.getState().scroll` inside `useFrame` — never subscribe (a
  Zustand subscription inside `useFrame` re-renders per frame; the file already says so).
- **Rotation.** In `Earth.tsx`'s `useFrame`, target rotation `y += dt * 0.02 + (scrollDelta * 0.9)`
  where `scrollDelta` is the change in `scroll` since last frame. Scrolling 100 % of the
  hero turns the globe ~ 52° (0.9 rad). It is *additive* to auto-rotation and to
  `OrbitControls` drag, and damped through the existing `enableDamping` so it never snaps.
- **Parallax.** The globe wrapper (`data-hero="globe"`) gets a scroll-linked
  `translateY(0 → 90px) scale(1 → 1.08)` over hero exit — it lags the copy (which moves
  −60 px), so the two layers separate as the reader scrolls. CSS only, 3.1 mechanism.
- **Tilt toward the reader.** Camera `position.y` eases from `0.3 → 0.55` as `scroll`
  goes 0 → 1 (`camera.position.y += (target - y) * 0.06` per frame): the globe rolls
  slightly "under" the page as it leaves. Clamp `minPolarAngle` accordingly so a drag
  cannot fight it.
- **Rim.** `uRimStrength` target `0.55 → 0.85` across the exit: the atmosphere brightens as
  the globe is left behind — the last thing the eye sees of it is the limb.
- **Satellites.** Trail `length` uniform `1 → 1.6` with scroll: orbits stretch as the
  camera "pulls away". Under `dragOver` the existing `1` override still wins.
- **Frameloop.** `frameloop` already follows `globeVisible`; the hero-exit range ends
  before the globe leaves the viewport, so no frames are spent on an invisible canvas.
- **Reduced motion.** `scroll` is still written (it is cheap) but the globe ignores it;
  the poster (`GlobePoster`) never reacts to scroll at all.

### 3.3 Cursor-reactive lighting

Three surfaces respond to the pointer; nothing else does. Pointer-driven effects are
gated on `(hover: hover) and (pointer: fine)` — on touch the page is the scroll version.

1. **Hero glow follows the cursor, lazily.** The large `data-hero="glow"` disc gets
   `--mx`/`--my` (viewport fractions) written by a single `pointermove` listener on the
   hero, rAF-throttled, and eased in CSS with `transition: translate 900ms var(--ease-out-quint)`
   on the `translate` property: `translate: calc((var(--mx) - 0.78) * 120px) calc((var(--my) - 0.4) * 80px)`.
   Max travel ± 120 × 80 px, so it is a light that *leans*, not a spotlight that follows.
2. **Globe rim light.** In `Earth.tsx`, `uSun` is currently a 180 s clock. Blend it:
   `sun = normalize(mix(clockSun, cursorSun, 0.35))` where `cursorSun` is
   `(mx − 0.5, −(my − 0.5), 0.6)` in view space. The terminator swings ~ 20° toward the
   cursor, damped at `0.05`/frame. Visible, subtle, and it makes the globe read as *lit
   by the room the reader is in*.
3. **Capability preview HUD specular** (§5). Each HUD card gets a `::before` radial
   highlight, `oklch(from var(--color-evidence) l c h / 0.10)` at `var(--mx) var(--my)`
   (card-local), `mix-blend-mode: plus-lighter`, 0 → 1 opacity on `:hover`. The same
   listener writes card-local fractions; the highlight is a single gradient repaint,
   `contain: paint` on the card.

Not cursor-reactive, by decision: headings, the pipeline stages, the nav, buttons beyond
their existing 120 ms feedback. Three lights, one hand.

### 3.4 Timeline audit

After 3.1–3.3 the page has: one entrance sequence (hero, ≤ 1.2 s), one scroll-linked
system, three cursor lights, one per-stage play-once sequence, and **zero** infinite
loops on desktop other than the globe's own rotation and the `data-breathe` reticles.
Remove the `sq-drift` 18 s glow loop once the cursor lean exists — two idle motions on one
element is one too many.

### 3.5 Acceptance

- Full-page scroll at 1280 × 800 on the demo laptop: ≥ 55 fps in the Performance panel
  with the globe live; 0 layout shifts.
- `prefers-reduced-motion: reduce`: no scroll-linked motion, no cursor light, globe
  static poster unless "Play globe" is pressed. `motion.test.ts` green.
- Keyboard-only walkthrough: nothing here changes focus order or hides a control.

---

## 4. Capability tabs — click-to-switch (`Capabilities.tsx`, `state/landing.ts`)

### 4.1 Why hover-to-preview fails

`hoveredCapability` overrides `activeCapability` for the preview. Moving the mouse across
the list to reach "Try this in the console" flips the preview three times; the selected
tab (`aria-selected`) and the shown panel disagree for as long as the pointer rests on a
different row; on touch the two states are indistinguishable. A preview that changes
under the cursor is a slideshow, not a choice.

### 4.2 Target interaction

- **Selection is a click (or Enter/Space, or ↑/↓).** The tab list is already a real
  `role="tablist"`; keep `aria-selected`, `tabIndex` roving and the arrow handler.
  Add `Home`/`End`. Selection *follows focus* on arrow keys (current behaviour) — that is
  the WAI-ARIA "automatic activation" pattern and it is right for three tabs whose panels
  are cheap.
- **Hover does one thing only: it previews the *row*, never the panel.** On hover the row
  shows its collapsed `copy` line at `opacity 0.7` (peek), `border-line` edge, and a
  4 px leading bar in `--color-line` — the same `nav-active` grammar as the console at
  low intensity. The panel does not move.
- **Switch motion.** On selection the preview cross-fades and slides: outgoing
  `opacity 1 → 0, translateX(0 → -12px)` over 160 ms, incoming
  `opacity 0 → 1, translateX(12px → 0)` over 260 ms `var(--ease-out-quint)`, 60 ms
  overlap. Implemented with the two panels stacked in a grid cell (`grid-area: 1/1`) and
  `data-state="in|out"` attributes, not `key`-remounting (`sq-arrive` on a new `key` is
  the current approach; it cannot animate the outgoing side). The tools chip row below
  the preview cross-fades only (120 ms) — it is metadata.
- **Selected row.** `border-accent-warm/50 bg-accent-glow` stays; its `copy` expands with
  the existing `max-height` transition but with `grid-template-rows: 0fr → 1fr` instead,
  which animates to the true height with no magic `max-h-24`.
- **"Try this in the console"** reads `activeCapability`, never `hovered`. Today it can
  seed the wrong question if the pointer is on another row when it is clicked.

### 4.3 State

`state/landing.ts`: remove `hoveredCapability` and `hoverCapability`. Hover is CSS-only
now (`:hover` on the row), so nothing about it belongs in a store. Add
`capabilityDirection: 1 | -1` (set in `setCapability` from the index delta) so the slide
direction follows the list order — selecting a lower row slides up, a higher one slides
down. Keep `activeCapability` persisted only for the page's lifetime (`reset()` clears it).

### 4.4 Deep link

`/#cap-grounding` selects that capability on load (read `location.hash` once in an
effect). Cheap, and it lets a judge be sent straight to the one that matters to them.

### 4.5 Acceptance

- Moving the pointer across all three rows changes nothing in the panel.
- Click, Enter, Space, ↑/↓, Home/End all switch and keep `aria-selected` ⇔ panel in sync.
- `tryIt` seeds the question of the *selected* capability under any pointer position.
- Touch: identical to pointer, minus the row peek (no hover on touch).

---

## 5. Capability visuals — the fallback HUDs (`Capabilities.tsx` → new `landing/hud/*.tsx`)

The three previews are drawn with absolutely-positioned `div`s at percentage offsets and
`outline` rectangles: at 16:10 on a 720 px panel they read as wireframe boxes on a grid,
and nothing about the "change" preview says *change* except a label. These are replaced
by three authored SVG HUDs, one per capability, sharing one visual grammar with the Use
Cases plates just shipped (`pages/usecases/FamilyPlate.tsx`) so the site and the console
draw evidence the same way. They live under the real `.webp` when one exists and *are*
the visual when it does not.

### 5.1 Shared grammar (`landing/hud/Hud.tsx`)

- `viewBox 0 0 640 400`, `preserveAspectRatio="xMidYMid slice"`, `aria-hidden`.
- Layers, bottom to top: ground (`--color-bg-main`), *scene* (the drawn imagery
  stand-in), *graticule* (`Graticule module={40}`, existing), *evidence* (masks, boxes,
  seams), *readouts* (glass chips, existing `.glass` + `.t-coord`), *frame* (corner
  ticks in `--color-reticle`, a 1 px inner border `--color-glass-edge`).
- **Glow accents.** One `<filter id="hud-glow">` — `feGaussianBlur stdDeviation="3"` +
  `feMerge` of blur and source — applied to the *evidence* layer only. Glow is the
  evidence colour at `0.55` alpha behind a crisp 1 px line of the same colour at full
  alpha. Rule: a glowing element is always also drawn sharp; glow alone is decoration.
- **Colours**, by role, from tokens only: change/warn areas `--color-warn`; agreement
  `--color-ok`; disagreement `--color-fail`; boxes `--color-bbox` / `--color-bbox-fill` /
  `--color-bbox-label`; imagery stand-in tones `--color-evidence` (sand) and
  `--color-skip` (SAR grey); seams and readouts `--color-accent-warm-text`.
- **Scanline sweep.** A single 1 px `--color-evidence` line at 25 % alpha sweeps
  top→bottom once, 900 ms, when the HUD becomes active (the `sq-scan` keyframe already
  exists). Once, not looping.
- **Seeded detail.** The same mulberry32 `rng(seed)` as `FamilyPlate.tsx` — extract it to
  `src/lib/rng.ts` and import from both — so the drawn parcels are stable across renders
  and across the two surfaces.
- **Readouts carry real numbers or none.** `5.34 %`, `0.35 km²`, `1 region`, `DISAGREE ·
  2 regions`, `3 validated positions` all come from the captured fixture, as today.
  Nothing else is numeric.

### 5.2 HUD A — Multimodal change detection (`ChangeHud.tsx`)

Composition (640 × 400):

1. **Scene.** Two half-frames on a vertical seam at x = 320. Left: ~ 14 seeded parcels in
   sand at 0.14 fill / 0.35 stroke (fields), two thin `--color-skip` polylines (roads).
   Right: the same parcels and roads (identical seed — this is the *same place*), plus 6
   built-up parcels in terracotta hatch (the `FamilyPlate` hatch pattern, 5 px, 45°).
2. **Evidence.** The change mask: a `<path>` union approximating the 6 new parcels,
   filled `--color-warn` at 0.16, stroked 1 px `--color-warn`, with `#hud-glow`. It
   *develops*: `stroke-dasharray` draws the outline over 700 ms on activation, then the
   fill fades in over 300 ms.
3. **Seam.** 1.5 px `--color-accent-warm-text` at 0.8, `strokeDasharray 6 6`; a 14 px
   glass handle circle at mid-height. A `≡` glyph inside the handle (three 1 px lines) —
   it is the console's own swipe handle.
4. **Readouts.** Top-left `A · 2019 · TC`, top-right `B · 2024 · CHANGE`; bottom-left
   glass chip `5.34 % changed · 0.35 km²` in `.t-mono`; bottom-right a mini legend:
   warn swatch `change mask`, hatch swatch `built-up (B)`.
5. **Frame ticks** and the scanline.

Motion on activation (all ≤ 1.2 s total, transform/opacity/dasharray only): seam handle
slides from x = 400 to 320 (260 ms) → mask outline draws (700 ms) → fill and readout chip
fade (300 ms). Reduced motion: final frame only.

### 5.3 HUD B — SAR / optical consistency (`CrossModalHud.tsx`)

1. **Scene.** Diagonal seam from (400, 0) to (240, 400). Optical side (left): 7 seeded
   soft ellipses, sand and `--color-ok` radial gradients (explicit `<stop>` colours — see
   the bug fixed in `FamilyPlate.tsx`, `currentColor` does not resolve from the shape),
   plus two cloud blobs in `--color-text-hi` at 0.10 with `feGaussianBlur 8` — *this*
   is the "under cloud" the copy talks about. SAR side (right): the 24 px speckle pattern
   in `--color-skip`, 14 bright backscatter streaks in sand at 0.7.
2. **Evidence.** Two regions crossing the seam. Region 1 (agree): a rounded rect
   straddling the seam, `--color-ok` at 0.12 fill, 1 px stroke, glow. Region 2
   (disagree): same shape, `--color-fail` 0.12 fill, 1 px **dashed** stroke `4 3`, glow;
   inside it, on the optical side, a cloud blob — the disagreement *is* the cloud. A thin
   leader line from region 2 to the readout chip.
3. **Seam.** 1.5 px, solid, `--color-accent-warm-text`; three `--color-ok` dots and one
   `--color-warn` dot spaced along it (agreement samples), each with glow.
4. **Readouts.** Top-left `optical · TC`, top-right `SAR · VV/VH`; bottom-left glass chip
   `physics_agreement` + `.chip bg-fail/12 text-fail` `DISAGREE · 2 regions`; bottom-right
   legend: ok `agree`, fail-dashed `disagree`.

Activation: the SAR side "acquires" — its speckle pattern fades in from 0 over 500 ms
while the streaks scale from `scaleX(0)` at their left edge (200 ms, staggered 30 ms);
then the two regions draw (dasharray, 500 ms); then the fail region's glow pulses **once**
(`sq-pulse-once`). Reduced motion: final frame.

### 5.4 HUD C — Sub-pixel grounding (`GroundingHud.tsx`)

1. **Scene.** An apron: a large light polygon (`--color-skip` at 0.18) with 1 px
   `--color-skip` taxiway lines at 0.4 and a dashed centreline; three seeded aircraft
   silhouettes — each a 5-point `<path>` (fuselage + swept wings, ~ 48 × 40), sand at
   0.45, rotated ±20°.
2. **Evidence.** Three bounding boxes in `--color-bbox` stroke 1 px, `--color-bbox-fill`,
   corner ticks 8 px in `--color-bbox-label` (the reticle grammar), glow on the ticks only.
   Each box carries a `.t-coord` label above it: `aircraft · 0.91 / 0.92 / 0.93` — the
   fixture's confidences. Under each box, a 1 px **sub-pixel grid**: the 40 px graticule
   subdivided 4 × inside the box only (`--color-grid-major`), which is what "validated
   against the raster grid" looks like.
3. **Reticle.** The existing `Reticle` (64) centred on the middle aircraft, `data-breathe`
   → replaced by a one-shot `sq-lock` on activation.
4. **Readouts.** Top-left `VHR · 0.5 m GSD`, top-right `GROUNDING · COUNT`; bottom-left
   glass chip `3 validated positions`; bottom-right the pixel coordinate of the locked
   box in `.t-coord`: `px 412,218 → 12°58′N 77°35′E` (the fixture's centroid; the pixel
   pair is that box's own origin in the 640 × 400 space, so it is honest by
   construction).

Activation: reticle `sq-lock` (480 ms) → boxes draw one after another (dasharray, 300 ms,
120 ms stagger) → labels rise (`sq-rise`, 200 ms each) → readout chip fades. Reduced
motion: final frame.

### 5.5 Image overlay contract

When `public/samples/capabilities/<id>.webp` loads, the *scene* layer is hidden
(`data-has-image` on the HUD root → `.hud-scene { opacity: 0 }`) and every other layer
stays: graticule, evidence, readouts, frame. The HUDs are therefore not a "fallback" that
disappears; they are the annotation layer the real imagery gets. The `<img>` sits under
the SVG with `object-cover`, `saturate(0.85)` → `1` on hover as the Use Cases cards do.

### 5.6 Size and code shape

Three files of ~ 120 lines of JSX each plus `Hud.tsx` (~ 80) and `lib/rng.ts` (~ 20).
Estimated +5 KB gz on the `Landing` chunk; the entry is untouched. Pure SVG + CSS, no
canvas, no library. Each HUD is a function of `(active: boolean, seed: string)` and owns
no state.

### 5.7 Acceptance

- With the sample images absent, the three previews are distinguishable at a glance from
  across a room (the judge test): a split with a warm mask; a diagonal seam with a red
  dashed region; three ticked boxes on an apron.
- With images present, the annotation layer aligns with the same percentages the current
  `div` boxes use (so the supplied `.webp`s, if already cropped to today's overlay, still
  line up): change mask around 62–92 % × 38–66 %; cross-modal regions at 18 % and 66 %;
  boxes at (22, 28), (48, 44), (66, 24) × (14, 10). Keep these numbers.
- No colour outside `--color-*`; contrast suite green; `oxlint` clean; the alt-text test
  (`__tests__/alt-text.test.ts`) still finds the `<img>` alts.
- Reduced motion: every HUD renders its final frame with no animation.

---

## 6. Order of work, estimates, and tests to touch

| Step | Files | Est. | Tests to update |
|---|---|---|---|
| 2 Typography | `theme.css`, `package.json`, `Landing.tsx`, `Hero.tsx`, `ProofStrip.tsx`, `LandingNav.tsx`, `public/fonts/` | 0.5 d | `bundle-boundaries.test.ts` (Odida assertion → display face) |
| 1 Pipeline layout | `PipelineStory.tsx`, `theme.css` (`sq-dash`) | 0.5 d | none (visual); add a Playwright screenshot to `e2e/` at 1280 and 390 |
| 4 Tabs | `Capabilities.tsx`, `state/landing.ts` | 0.5 d | new `Capabilities.test.tsx`: hover does not change panel; keys do; `tryIt` uses the selected id |
| 5 HUDs | `landing/hud/{Hud,ChangeHud,CrossModalHud,GroundingHud}.tsx`, `lib/rng.ts`, `usecases/FamilyPlate.tsx` (import rng) | 1.5 d | `alt-text.test.ts` unchanged; add a render test that each HUD mounts with and without an image |
| 3 Motion | `theme.css`, `useScrollProgress.ts`, `Ambience.tsx`, `globe/Earth.tsx`, `globe/Controls.tsx`, `Hero.tsx`, `state/landing.ts` | 1.5 d | `motion.test.ts` (add: scroll-linked rules are inside `no-preference`); manual perf pass |

Total ≈ 4.5 days for one engineer. Ship §2 + §1 together (a visual-only PR), then §4 + §5,
then §3 last — motion is the layer most likely to be cut for time, and the page is already
whole without it.

## 7. Open decisions for the team

1. Move the console's `--font-sans` to Geist as well, or keep Inter? (Default: keep Inter.)
2. Is the demo laptop's browser Chromium? If Firefox, §3.1's fallback path is the primary
   path and the scroll-linked beam/HUD development should be dropped rather than
   emulated in JS.
3. Confidences `0.91 / 0.92 / 0.93` for the three aircraft boxes are what the current
   markup shows; confirm they match `events.bitemporal.json` (or the grounding fixture, if
   one is added) before they go on the HUD — the plan's rule is no invented numbers.
