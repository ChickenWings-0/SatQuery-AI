# Frontend Technical Audit — SatQuery AI

`/impeccable audit` on `frontend/` (React 19 + Tailwind v4 + Vite, three-column
Operate-mode analysis console). Audited 2026-09-07 against commit `44e6947`
plus the uncommitted `frontend/` tree.

No `PRODUCT.md` / `DESIGN.md` exists; the visual authority is
`frontend/src/styles/theme.css` plus `DOCS/FRONTEND_ROADMAP.md §F0`, and the code
is unusually explicit about its own rules — which is what makes the violations
findable.

This is a code-level pass: every finding was verified by reading the source, and
every contrast number was computed from the token hexes in `theme.css` (WCAG 2.x
relative luminance), not from a tool. The bundled Impeccable detector was run
over `frontend/src` and returned **zero** findings, so the Implementation
Integrity dimension rests on manual verification only. No browser session was
run, so nothing here depends on a screenshot.

## Audit Health Score

| # | Dimension | Score | Key finding |
|---|-----------|-------|-------------|
| 1 | Accessibility | 2 | `theme.css` bans orange-on-cream, then `text-warn` (1.99:1) and `text-accent-warm` (3.53:1) carry small text in 8+ places |
| 2 | Performance | 3 | Three components subscribe to whole Zustand stores; `ThreadPanel` re-renders on every swipe-drag frame |
| 3 | Responsive Design | 1 | Fixed 240px sidebar with no breakpoint below `xl`; at 375px the stage gets ~135px |
| 4 | Theming | 3 | Strong, enforced token system — but Inter/Geist Mono are declared and never loaded |
| 5 | Implementation Integrity | 2 | "Questions these images can answer" chips are `<button>`s with no `onClick` |
| **Total** | | **11/20** | **Acceptable — significant work needed** |

## Implementation Integrity verdict — **Pass, with two honesty defects**

The system is coherent and unmistakably product-specific. Status colour is one
vocabulary (`StatusDot` serves both `ToolStatus` and `CheckStatus`), empty states
refuse to invent data (`DatasetsPanel` says the route does not exist), null
handling follows the API contract's rule rather than defaulting to `0`, and the
progressive-disclosure bet is structurally enforced — `@xyflow/react` is imported
in exactly one module behind `React.lazy`. Nothing here is interchangeable with a
generic dashboard.

Two things break the product's own honesty premise, which is why this is not a 4:

- `DagCanvas.tsx:66` hard-codes `progress = 65` for a RUNNING step and the
  comment above it claims `est_ms` drives the bar. `est_ms` is not referenced.
  A fabricated determinate progress bar in the one surface built to prove the
  pipeline is real.
- `PreflightPanel.tsx:102` renders the task suggestions as buttons that do
  nothing. The most inviting affordance on the first screen is inert.

## Findings

### P0 — blocking

**1. Task-suggestion chips are inert buttons** — `stage/PreflightPanel.tsx:101-111`
· Integrity / UX. The `<button>` has `title` and hover styling but no `onClick`.
On the first screen, the primary call to action silently fails. Fix: on click,
write the `SUGGESTIONS[task]` string into the composer and focus it (the composer
is already registered in `useFocusStore.composer`; add a `setComposerText` action
or lift the textarea value into the store).

### P1 — fix before release

**2. `--color-warn` used as text on cream — 1.99:1** — `CompatibilityDetails.tsx:16`
(`PASS_WITH_WARNINGS` headline), `Dropzone.tsx:85`, `PreflightPanel.tsx:123`,
`ManifestCard.tsx:80` · WCAG 1.4.3 (needs 4.5:1). `#f59e0b` on `#faf6f0` is
effectively invisible. `theme.css:20-24` already states the rule — amber is a
fill or an underline, never text on cream. Fix: add `--color-warn-text` (a darker
amber, ~`#8a5a00`, ≈5.4:1) and swap these four sites; keep `--color-warn` for
fills, dots and the `.uncited` underline.

**3. `--color-accent-warm` as small interactive text — 3.53:1** — `.btn-ghost`
(`theme.css:200`), `Dropzone.tsx:70`, `PreflightPanel.tsx:54`,
`PipelineDialog.tsx:172`, `Disclosure.tsx:24` · WCAG 1.4.3. The same self-declared
rule ("rust is for borders, icons and large text") violated at 14px on four
controls. Fix: darken the token used for text to ~`#8f4f33` (≈5.2:1), or give
these controls the `.btn-primary` fill.

**4. Sidebar text at low opacity — 2.6:1 to 4.05:1** — `Sidebar.tsx:36,68,73,91`
(`opacity-55/45/40`), `HealthStrip.tsx:21,49,54,57,61` (`opacity-60`) · WCAG
1.4.3. The empty-state line "Questions you ask will collect here." computes to
**2.64:1**; the "Recent queries" eyebrow to 2.94:1; the VRAM and tool-count
readouts to 4.05:1. Fix: replace opacity with explicit on-sidebar tokens
(`--color-sidebar-text-lo`, ≥4.5:1 on `#6c3428`) and delete the opacity
modifiers — opacity over a dark ground is what made these hard to see coming.

**5. Focus ring invisible in the sidebar — 2.56:1** — `theme.css:87-91`. The one
global focus treatment is `--color-accent-warm`, which fails 3:1 (WCAG 1.4.11)
against the brown column that holds the primary navigation. Fix: `:focus-visible`
inside the sidebar switches to `--color-sidebar-text`.

**6. Global arrow-key capture breaks the A/B slider and the DAG** —
`shell/useHotkeys.ts:55-62`. `isTyping()` only excuses `INPUT/TEXTAREA/SELECT`, so
`ArrowLeft`/`ArrowRight` are `preventDefault`ed even when the
`ReactCompareSlider` handle or a React Flow node has focus. `ImageViewer.tsx:117`
claims "The handle is focusable, so arrow keys move it too" — it cannot. Also
WCAG 2.1.4 (single-character shortcuts with no off switch). Fix: bail out when
`event.target.closest('[role="slider"], .react-flow, [role="dialog"]')` matches,
and gate the listener on a preference the user can turn off.

**7. No `prefers-reduced-motion` anywhere** — `animate-pulse` on RUNNING pills
(`PipelinePulse.tsx:41`), animated DAG edges (`DagCanvas.tsx:188`), all the 120ms
colour transitions · WCAG 2.3.3. Fix: one `@media (prefers-reduced-motion:
reduce)` block in `theme.css` that stills the pulse and the edge dash **while
keeping the colour state change** — do not blanket-kill to `0.01ms`; the pulse is
the only "this step is running" signal.

**8. No live regions on a streaming UI** — `DataStage.tsx:70-74` (stage name),
`ThreadPanel.tsx:132,137` (errors), `PreflightPanel.tsx:62` (pre-flight running)
· WCAG 4.1.3. A screen-reader user gets no notice that a job started, degraded, or
failed. Fix: `role="status"` on the stage/phase line and the pre-flight message,
`role="alert"` on the two error blocks.

**9. Layout has no breakpoint below `xl`** — `shell/AppShell.tsx:27-31`. The
sidebar is a hard `240px` track at every width; at 375px that leaves ~135px for
the image stage, which also carries `min-h-[clamp(340px,48vh,620px)]`. The
`PipelineDialog` inspector is a fixed `w-80` beside the graph
(`PipelineDialog.tsx:184`), leaving ~35px of canvas on a phone. Fix: below `md`,
collapse the sidebar to an icon rail or a top bar, and stack the dialog's
inspector under the graph.

**10. Declared fonts are never loaded** — `theme.css:58-59` names
`'Inter Variable'` and `'Geist Mono'`; nothing in `index.html`, `theme.css` or
`src/` loads either, and there is no `@font-face`. The whole product renders in
`system-ui` / `ui-monospace`, so the typographic identity is nominal — and it
changes shape per OS. Fix: self-host both (woff2 in `public/fonts`, `@font-face`
with `font-display: swap`, preload the sans), or delete the names and commit to
the system stack deliberately.

### P2 — next pass

**11. `StatusDot`'s `aria-label` sits on a bare `<span>`** — `ui/StatusDot.tsx:26`.
`aria-label` on a role-less generic element is ignored by most AT, so the
compatibility table and the pipeline pulse communicate status by colour alone
(WCAG 1.4.1). Fix: `role="img"` on the span, or render visually-hidden text.

**12. Query textarea has no accessible name** — `thread/QueryComposer.tsx:38`.
Placeholder only (WCAG 3.3.2). Fix: `aria-label="Question about this imagery"`.

**13. `title` is the only tooltip mechanism** — citation pills
(`GroundedAnswer.tsx:41`), the KPI degraded dot (`KpiCards.tsx:49`), `StatusDot`,
tray thumbnails. Unavailable to touch, unreliable for keyboard. Fix: keep `title`
as a fallback, add visible or `aria-describedby` text for the degraded-KPI and
uncited-number cases, which carry real meaning.

**14. Whole-store subscriptions cause avoidable re-renders** —
`ThreadPanel.tsx:102-103` (`useFocusStore()`), `PreflightPanel.tsx:30` and
`Dropzone.tsx:22` (`useUiStore()`). `setSwipe` fires on every pointermove during
an A/B drag, so the entire right column — answer, pulse, confidence bars —
re-renders per frame. Fix: per-field selectors, matching the pattern already used
in `DataStage.tsx:23-32`.

**15. Touch targets under 44px** — nav rows ~33px (`Sidebar.tsx:50`), recent-run
rows ~30px (`Sidebar.tsx:84`), submit button ~30px (`QueryComposer.tsx:69`),
citation pills ~20px (`GroundedAnswer.tsx:45`) · WCAG 2.5.8 (24px minimum, 44px
target). Fix alongside finding 9.

**16. Dialog `selected` step goes stale** — `PipelineDialog.tsx:76`. `useState`
seeds from `focusedStep` once; the component never unmounts (only its content is
gated by `open &&`), so after the first node click every later citation opens the
dialog on the wrong node. Fix: reset `selected` to `null` when `open` flips true,
or derive from `focusedStep` with a key.

**17. Chip contrast in the tools registry** — `ToolsPanel.tsx:32`. `text-skip` on
`bg-skip/15` ≈3.15:1 and `text-ok` on `bg-ok/12` ≈4.17:1 at 11px; both fail AA.
Fix: darker on-chip foregrounds, or larger text.

**18. `HistoryPanel` and `DatasetsPanel` bypass the type scale** —
`HistoryPanel.tsx:51`, `DatasetsPanel.tsx:10` use raw `text-2xl font-semibold`
where `.t-page` exists for exactly this. Two of six sections drift off the scale
the CSS was written to enforce.

### P3 — polish

**19. Six hard-coded hexes outside the token system** — `DagCanvas.tsx:82,189,221`,
`ImageViewer.tsx:44,93,95`. Several duplicate token values exactly (`#dfa878` =
`--color-surface-sand`, `#f59e0b` = `--color-warn`). Fix: read them off
`getComputedStyle`, or expose them as CSS vars the JS can reference.

**20. Dead `animate-in fade-in-0` classes** — `ui/dialog.tsx:25`. Those are
`tailwindcss-animate` utilities; the plugin is not installed and `theme.css`
imports only `tailwindcss`, so the dialog has no entrance animation at all. Fix:
install the plugin or write the keyframes (respecting finding 7).

**21. Dead code** — empty `useEffect` in `DagCanvas.tsx:204-207`; unused export
`warningsFor` in `state/ui.ts:81`; `.chip-truncate` duplicates every `.chip` rule
rather than composing it (`theme.css:249-261`).

**22. Recent-query rows do not open their run** — `Sidebar.tsx:82` navigates to
History but does not select the clicked trace, so the user re-finds it by hand.

## Patterns and systemic issues

- **The palette rules are written down and then broken.** `theme.css:20-24` is a
  precise, correct warning about which colours may carry text; findings 2, 3, 5
  and 17 are all that rule violated in different components. The tokens are not
  the problem — the missing `on-*` text variants are.
- **Opacity substitutes for a colour token** on the dark sidebar (finding 4): nine
  call sites, all invisible failures, because nobody can eyeball 2.6:1.
- **Assistive-technology parity was never a pass.** No live region, no accessible
  name on the primary input, status conveyed by colour to AT — three independent
  gaps in a product whose entire pitch is auditability.
- **Everything below 1280px is unattended.** One `sm:` and one `xl:` breakpoint in
  3,000 lines of component code.

## Positive findings — keep these

- The lazy boundary around `@xyflow/react` is real and structurally enforced by a
  single import site; `DagCanvas.tsx:1-16` explains why. MSW is dynamically
  imported so fixtures cannot reach the production bundle.
- `StatusDot` as the single status vocabulary across `ToolStatus` and
  `CheckStatus` — one component, one meaning, product-wide.
- Honest nulls: `ManifestCard` renders `—` and surfaces the reason from
  `warnings[]`; `HealthStrip` refuses to draw a 0% VRAM bar rather than asserting
  something the server did not say; `DatasetsPanel` says the route does not exist
  instead of inventing a catalogue.
- `min-w-0` / `min-h-0` discipline through the grid and flex chains, with comments
  explaining the failure each one prevents.
- The comment layer generally records *why*, including past failures — rare, and
  worth protecting through any refactor.

## Recommended actions, in order

1. **[P0/P1] `/impeccable harden`** — findings 1 (inert suggestion chips), 6
   (arrow-key capture breaking the slider and DAG), 16 (stale dialog step).
2. **[P1] `/impeccable colorize`** — findings 2, 3, 4, 5, 17: add
   `--color-warn-text`, a darker rust for text, on-sidebar foreground tokens and a
   sidebar focus-ring variant, then sweep the call sites.
3. **[P1] `/impeccable adapt`** — findings 9 and 15: breakpoints below `xl`,
   sidebar rail, dialog stacking, 44px targets.
4. **[P1] `/impeccable animate`** — findings 7 and 20: a reduced-motion block that
   preserves the running-state signal, plus a real dialog entrance.
5. **[P1] `/impeccable typeset`** — finding 10: self-host Inter and Geist Mono, or
   commit to the system stack.
6. **[P2] `/impeccable optimize`** — finding 14: selector-scoped store reads.
7. **[P2/P3] `/impeccable polish`** — findings 11, 12, 13, 18, 19, 21, 22.

## How to verify fixes

- `cd frontend && npm run typecheck && npm run lint && npm test` after each pass.
- `npm run dev`, then `http://localhost:5173/?mock=1` — MSW drives the whole run
  with the backend down, so streaming, degraded steps and citations are all
  reachable without a GPU.
- Contrast: re-derive each changed pair from the hexes in `theme.css` (all the
  numbers above came from that), or check with the DevTools contrast picker at
  `?mock=1`.
- Responsive: 375 / 768 / 1280 / 1920, plus the `PipelineDialog` open at each.
- Keyboard: `/`, `←`/`→`, `[`/`]`, `P`, then Tab into the A/B handle and confirm
  the arrows move the swipe rather than the evidence tray.
- Re-run `impeccable detect --json frontend/src` and `/impeccable audit` to
  re-score.
