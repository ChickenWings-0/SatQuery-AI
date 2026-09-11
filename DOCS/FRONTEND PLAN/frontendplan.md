# Mission-Control Dashboard Overhaul — SatQuery AI

Transform the existing warm-cream light-theme dashboard into a dark-mode, earth-toned mission-control interface matching the [reference screenshot](file:///home/chickenwings/SatQuery-AI/dashboard_reference.png).

## Current State Assessment

The existing frontend is a **well-architected React 19 + Vite + Tailwind v4 + TypeScript** application. The codebase is production-quality in structure:

| Layer | What exists | Status |
|-------|------------|--------|
| **Framework** | React 19, Vite 6, TypeScript 5.9, Tailwind v4 | ✅ Solid |
| **State** | Zustand stores: [`ui.ts`](file:///home/chickenwings/SatQuery-AI/frontend/src/state/ui.ts), [`job.ts`](file:///home/chickenwings/SatQuery-AI/frontend/src/state/job.ts), [`focus.ts`](file:///home/chickenwings/SatQuery-AI/frontend/src/state/focus.ts) | ✅ Solid |
| **API** | Typed client with SSE streaming, health polling, artifact URLs | ✅ Solid |
| **Compare Slider** | `react-compare-slider` v4 with `react-zoom-pan-pinch` | ✅ Already exists |
| **Citation System** | `annotate.ts` + `GroundedAnswer` + `parseSource` | ✅ Already exists |
| **Evidence Tray** | `EvidenceTray` with `ViewGroup` model | ✅ Already exists |
| **KPI Cards** | `KpiCards` with degraded-step detection | ✅ Already exists |
| **Layout Shell** | 3-column CSS grid with responsive breakpoints | ✅ Exists (needs restyle) |
| **Design Tokens** | Warm cream light theme in [`theme.css`](file:///home/chickenwings/SatQuery-AI/frontend/src/styles/theme.css) | 🔄 Must be replaced |
| **Visual Design** | Utilitarian/functional, light mode only | 🔄 Needs full overhaul |

> [!IMPORTANT]
> **Key insight:** ~80% of the component logic, state management, API integration, and streaming architecture is already correct. This overhaul is primarily a **design system swap** (light → dark tokens) + **layout restructuring** (3-column rebalance) + **new UI components** (nav rail, mode selector, scene header, suggestion chips, tabbed sidebar, history list). We are *not* rewriting the data layer.

---

## User Review Required

> [!WARNING]
> **Tailwind v4 will be preserved.** The existing codebase uses Tailwind v4's `@theme` directive and `@custom-variant` — migrating away would break every component. The new tokens will be authored in the same `@theme` block, just with dark-mode values.

> [!IMPORTANT]
> **No new routing library.** The existing section-switching via Zustand (`useUiStore.section`) works. The reference dashboard's navigation rail (Home, Explore, Datasets, Tools, Use Cases, Maps, Saved, Projects) will expand the existing 4-item `NAV_SECTIONS` to the full set. This is a state-only change, not a router migration.

> [!IMPORTANT]
> **`react-compare-slider` stays.** It already handles the A/B swipe with zoom-pan-pinch. The visual overhaul wraps it in the new dark card treatment with overlay controls, timestamp labels, and a feature legend — no library change.

---

## Open Questions

> [!IMPORTANT]
> **1. SVG Bounding-Box Overlay.** You mention a spatial grounding layer for `<|box_start|>(ymin,xmin),(ymax,xmax)<|box_end|>` coordinates. The existing API contract and answer format use `Citation` objects with `step:N/scalars.path` sources — **do the bounding-box tokens already appear in the answer text**, or is this a future API surface? If it's future, I'll scaffold the SVG overlay layer with a placeholder parser.

> [!IMPORTANT]
> **2. Multi-Spectral Strip Image Sources.** The reference shows 4 fixed thumbnails (True Color, NDVI Heatmap, SAR Backscatter, Change Mask). Currently, the `EvidenceTray` dynamically renders whatever `ViewGroup`s the run produces. Should we: **(a)** keep the dynamic behavior but restyle it to match the reference, or **(b)** hard-code a 4-slot strip with placeholder images when no run has been executed?

> [!IMPORTANT]
> **3. User Profile Card.** The reference shows "Aksh · Student · Researcher" with an avatar and settings cog in the sidebar footer. Is there an authentication/user system, or should this be a static mock for now?

---

## Proposed Changes

### Phase 1: Design System — Token Swap

#### [MODIFY] [`theme.css`](file:///home/chickenwings/SatQuery-AI/frontend/src/styles/theme.css)

Complete replacement of the `@theme` block. Every token gets a dark-mode equivalent:

```
Current Token                  → New Value                    Purpose
──────────────────────────────────────────────────────────────────────
--color-bg-main: #faf6f0       → #100C0A                     Base canvas
--color-surface-card: #ffffff  → #211814                     Cards, sidebars
--color-surface-sand: #dfa878  → #DFA878                     Secondary accent (kept)
--color-line: #ecdcc9          → #2A1F1A                     Borders, hairlines
--color-line-soft: #f2e7da     → #1E1612                     Internal rules
--color-sidebar: #6c3428       → #211814                     Nav column (now same as surface)
--color-sidebar-hi: #7d4033    → #2D2018                     Sidebar hover
--color-sidebar-text: #f5e9e1  → #E8D5C4                     Sidebar foreground
--color-sidebar-text-lo        → #9A8579                     Dimmed sidebar text
--color-text-hi: #2b1d17       → #F0E6DF                     Primary text (inverted)
--color-text-lo: #6b5b52       → #9A8579                     Secondary text
--color-accent-warm: #ba704f   → #BA704F                     Primary accent (kept)
--color-accent-cool: #cee6f3   → #BA704F (merged)            Selection = accent now
--color-ok: #4a7c59            → #22C55E                     System Ready green
--color-warn: #c67e00          → #F97316                     Warning amber/orange
```

Additional new tokens:
- `--color-surface-elevated: #2D2018` — modal/dropdown backgrounds
- `--color-accent-warm-glow: rgba(186,112,79,0.15)` — hover/active tints
- Card border: `1px solid #2A1F1A` with `border-radius: 0.75rem`

Update `color-scheme` from `light` to `dark`. Update scrollbar colors. Update `::selection`. Update all component classes (`.card`, `.btn-primary`, `.btn-ghost`, `.kbd`, `.chip`) to work on dark grounds. Recompute all contrast ratios for dark-on-dark to maintain WCAG AA.

---

### Phase 2: Layout Shell Restructuring

#### [MODIFY] [`AppShell.tsx`](file:///home/chickenwings/SatQuery-AI/frontend/src/components/shell/AppShell.tsx)

Adjust the 3-column grid proportions:
- Left: `240px` → `200px` (compact nav rail)
- Center: `1fr` (unchanged)  
- Right: `420px` → `380px` (intelligence sidebar)
- Add a thin top header bar spanning the center+right columns for the mode selector and system status badge

#### [MODIFY] [`Sidebar.tsx`](file:///home/chickenwings/SatQuery-AI/frontend/src/components/shell/Sidebar.tsx)

Major restructure to match the reference nav rail:

1. **Header:** SatQuery AI logo + "From Space to Answers" subtitle
2. **CTA:** "+ New Query" button with keyboard shortcut badge (`⌘ K`)
3. **Nav Items:** Expand from 4 to 8 sections — add proper SVG icons instead of Unicode glyphs
4. **Footer:** Mission statement badge + User profile card
5. **Recent Queries:** Move from sidebar body to right-sidebar History tab

#### [MODIFY] [`state/ui.ts`](file:///home/chickenwings/SatQuery-AI/frontend/src/state/ui.ts)

Expand `NAV_SECTIONS` from `['explore', 'datasets', 'tools', 'history']` to `['home', 'explore', 'datasets', 'tools', 'usecases', 'maps', 'saved', 'projects']`.

---

### Phase 3: Center Analysis Workspace

#### [MODIFY] [`DataStage.tsx`](file:///home/chickenwings/SatQuery-AI/frontend/src/components/stage/DataStage.tsx)

Add the top scene header with:
- Mode selector dropdown (Earth View)
- Scene title ("Urban Expansion Analysis")
- Metadata strip: sensor, region, coordinates, timestamps
- Action buttons: Share, Download, Fullscreen

#### [MODIFY] [`ImageViewer.tsx`](file:///home/chickenwings/SatQuery-AI/frontend/src/components/stage/ImageViewer.tsx)

Add overlay controls:
- **Timestamp labels:** "T1 2023-01-01" / "T2 2024-01-01" pill badges in top corners
- **Zoom controls:** +/- buttons, layer reset, compass/orientation indicator (positioned top-right)
- **Scale bar:** "10 km" indicator at bottom-left
- **Feature legend:** "New Built-up Area" pills at bottom-right
- **Swipe handle:** Restyle the center divider with the `<>` icon
- **Dark card treatment:** Dark background with terracotta-tinted border

#### [NEW] `BboxOverlay.tsx` (`frontend/src/components/stage/BboxOverlay.tsx`)

SVG overlay component for rendering parsed `<|box_start|>(ymin,xmin),(ymax,xmax)<|box_end|>` coordinates. Scaffold with a parser and absolutely-positioned SVG that scales with the zoom transform.

#### [MODIFY] [`EvidenceTray.tsx`](file:///home/chickenwings/SatQuery-AI/frontend/src/components/stage/EvidenceTray.tsx)

Restyle as the multi-spectral strip:
- Larger thumbnails (wider cards with satellite imagery)
- Dark card backgrounds with terracotta active border
- Descriptive labels: "True Color (RGB) · Sentinel-2 · 10m"
- Keep dynamic artifact-driven rendering

#### [MODIFY] [`KpiCards.tsx`](file:///home/chickenwings/SatQuery-AI/frontend/src/components/stage/KpiCards.tsx)

Restyle as "Key Insights" strip:
- Dark surface cards with sand-colored metric icons
- Large stat numbers in light text
- Warning badges (amber triangle for model estimates)
- Descriptive sub-labels

---

### Phase 4: Right Intelligence Sidebar

#### [MODIFY] [`ThreadPanel.tsx`](file:///home/chickenwings/SatQuery-AI/frontend/src/components/thread/ThreadPanel.tsx)

Major restructure:

1. **Tabbed header:** Chat | Results | Citations | History — using a new local tab state
2. **Chat tab:**
   - Move `QueryComposer` into chat interface styling
   - Add quick-suggestion chips row below the input
   - Style as a chat bubble interface
3. **Response card:**
   - Restyle `GroundedAnswer` as an "Audited Response Card"
   - Add SatQuery AI avatar + timestamp header
   - Inline action buttons (copy, bookmark, share)
   - Citation pills restyled with terracotta/sand tones
   - Verification disclaimer box (amber warning for model estimates)
   - "View Processing Pipeline" / "View Sources" action buttons
4. **History section:**
   - Previous queries list with relative timestamps
   - "View all →" link

#### [MODIFY] [`QueryComposer.tsx`](file:///home/chickenwings/SatQuery-AI/frontend/src/components/thread/QueryComposer.tsx)

Restyle with dark theme:
- Dark input field with terracotta focus border
- Image attachment and "Auto" mode selector controls
- Terracotta send button (arrow icon)
- Quick-suggestion chips below: "Compare vegetation before and after 2020", "Detect new constructions", "Show NDVI trend", "Find floods in my area"

#### [MODIFY] [`GroundedAnswer.tsx`](file:///home/chickenwings/SatQuery-AI/frontend/src/components/thread/GroundedAnswer.tsx)

Restyle citation pills and uncited spans for dark mode. Add the verification disclaimer box.

#### [MODIFY] [`HealthStrip.tsx`](file:///home/chickenwings/SatQuery-AI/frontend/src/components/shell/HealthStrip.tsx)

Move system status from sidebar to top header bar. Restyle as "● System Ready" badge with emerald green.

---

### Phase 5: New UI Components & Assets

#### [NEW] `ModeSelector.tsx` (`frontend/src/components/shell/ModeSelector.tsx`)

Dropdown/chip for "Earth View" mode in the top header. Initially a static label; wired to be expandable.

#### [NEW] `SceneHeader.tsx` (`frontend/src/components/stage/SceneHeader.tsx`)

The title + metadata + action buttons bar above the viewer. Extracts scene info from the active result/validation state.

#### [NEW] `SuggestionChips.tsx` (`frontend/src/components/thread/SuggestionChips.tsx`)

Row of clickable suggestion pills that call `useFocusStore.proposeQuestion()`. Connects to the existing `supported_tasks` from validation.

#### [NEW] `SidebarTabs.tsx` (`frontend/src/components/thread/SidebarTabs.tsx`)

Tab bar component for Chat | Results | Citations | History in the right sidebar.

#### [NEW] `PreviousQueries.tsx` (`frontend/src/components/thread/PreviousQueries.tsx`)

History list component consuming `useUiStore.recentRuns`, moved from sidebar to right panel.

#### [NEW] Nav Icons (`frontend/src/components/ui/icons.tsx`)

SVG icon components for: Home, Explore, Datasets, Tools, Use Cases, Maps, Saved, Projects, Plus, Share, Download, Fullscreen, ZoomIn, ZoomOut, Reset, Settings, Bell.

---

### Phase 6: Polish & Micro-Animations

#### [MODIFY] [`theme.css`](file:///home/chickenwings/SatQuery-AI/frontend/src/styles/theme.css)

- Add subtle hover glow transitions on cards and nav items (120ms ease, using `background-color` and `border-color` only — no layout properties)
- Card hover: border shifts from `#2A1F1A` → `#BA704F` at 40% opacity
- Active nav item: terracotta left border accent + subtle background tint
- Add `sq-glow` keyframe for the "System Ready" dot pulse
- Ensure `prefers-reduced-motion` overrides are maintained for all new animations

---

## File Change Summary

| Action | File | Scope |
|--------|------|-------|
| MODIFY | `theme.css` | Complete token swap + new component styles |
| MODIFY | `AppShell.tsx` | Grid proportions + top header bar |
| MODIFY | `Sidebar.tsx` | Full nav rail redesign |
| MODIFY | `state/ui.ts` | Expand nav sections |
| MODIFY | `DataStage.tsx` | Scene header integration |
| MODIFY | `ImageViewer.tsx` | Overlay controls, dark styling |
| MODIFY | `EvidenceTray.tsx` | Multi-spectral strip restyle |
| MODIFY | `KpiCards.tsx` | Dark key-insights cards |
| MODIFY | `ThreadPanel.tsx` | Tabbed sidebar restructure |
| MODIFY | `QueryComposer.tsx` | Dark chat interface |
| MODIFY | `GroundedAnswer.tsx` | Dark citations, disclaimer box |
| MODIFY | `HealthStrip.tsx` | Move to header, emerald status |
| NEW | `BboxOverlay.tsx` | SVG grounding overlay (scaffold) |
| NEW | `ModeSelector.tsx` | Earth View dropdown |
| NEW | `SceneHeader.tsx` | Scene title + metadata bar |
| NEW | `SuggestionChips.tsx` | Quick-action chips |
| NEW | `SidebarTabs.tsx` | Chat/Results/Citations/History tabs |
| NEW | `PreviousQueries.tsx` | History list in right panel |
| NEW | `icons.tsx` | SVG icon set |
| MODIFY | `index.html` | Update meta description, theme-color |

**Estimated file count:** 12 modified, 7 new — **19 files total**

---

## Verification Plan

### Automated Tests

```bash
cd frontend && npm run typecheck    # TypeScript compilation must pass
cd frontend && npm run test         # Existing vitest suite must pass
cd frontend && npm run lint         # Oxlint must pass
cd frontend && npm run build        # Production bundle must build cleanly
```

### Manual Verification

1. **Visual match**: Launch dev server (`npm run dev`) and compare side-by-side with the reference screenshot
2. **Dark mode contrast**: Verify all text elements meet WCAG AA (4.5:1 for body text, 3:1 for UI components) using browser DevTools contrast checker
3. **A/B slider**: Confirm `react-compare-slider` still functions with dark styling, keyboard shortcuts `[`/`]` still work
4. **Citation flow**: Click a citation pill → evidence tray scrolls → KPI card highlights → pipeline modal arms on correct step
5. **Responsive**: Test at phone (390px), tablet (768px), and desktop (1440px+) — shell breakpoints (`wide:` / `desk:`) must still trigger correctly
6. **MSW mock mode**: `?mock=1` flag must still work for offline development
7. **SSE streaming**: Verify events still render correctly with new dark card styling
8. **Reduced motion**: `prefers-reduced-motion: reduce` must suppress all new animations while keeping state indicators
