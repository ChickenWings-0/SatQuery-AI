# SatQuery AI — Frontend Implementation Roadmap

## Context

The backend pipeline is real and working: `POST /v1/analyze` is wired end-to-end to
`satquery.agent.pipeline.analyze`, and the wire format is frozen at schema `1.0`
(`DOCS/API_CONTRACT.md`, machine mirror `openapi.json`). There is **no frontend in the repo** —
`find -name package.json` returns nothing. This plan builds one.

**Chosen direction** (revised): the *Geospatial Dashboard* — a warm, earth-toned light theme in a
three-column layout, built around **progressive disclosure**. The imagery is the hero; the answer
sits beside it; the `@xyflow/react` execution DAG is **not on the main screen** and opens only
when the user asks for it via a `View Processing Pipeline` control. The honesty guards from the
previous direction are retained verbatim: clickable orange citation pills and amber wavy
underlines for `uncited_numeric_spans`.

```
┌──────────┬──────────────────────────────────────┬─────────────────────────┐
│ #6C3428  │  THE DATA STAGE                      │ INTERROGATION THREAD    │
│          │                                      │                         │
│ Explore  │  ┌────────────────────────────────┐  │ ┌─────────────────────┐ │
│ Datasets │  │   A ◀────────╫────────▶ B      │  │ │ ask a question…     │ │
│ Tools    │  │   pre/post swipe viewer        │  │ └─────────────────────┘ │
│ History  │  └────────────────────────────────┘  │                         │
│          │  [TC][FCIR][NDVI*][SAR][CHANGE]      │ …7.4% of the scene ▔▔▔  │
│  ── runs │        * active = #CEE6F3            │  transitioned to        │
│  b3f1…   │                                      │  built-up …             │
│  9a02…   │  KEY INSIGHTS                        │ …roughly 3 km² ~~~~     │
│          │  ┌────────┐┌────────┐┌────────┐      │                         │
│          │  │  7.4%  ││ 1.07   ││  0.86  │      │ ┌─────────────────────┐ │
│          │  │ built- ││  km²   ││ conf.  │      │ │ View Processing     │ │
│          │  │ up exp.││ changed││        │      │ │ Pipeline        ▸   │ │
│          │  └────────┘└────────┘└────────┘      │ └─────────────────────┘ │
└──────────┴──────────────────────────────────────┴─────────────────────────┘
                    ↑ clicking the button opens the DAG full-screen
```

**The tension this creates, stated plainly.** Contract §8.4 says the filling-in tool graph is
"the single most persuasive thing on screen for the judges". Hiding it by default trades that
away. The mitigation is in F5: a compact **live pipeline pulse** — one small pill per step,
streaming its status in real time directly under the answer — so the machine is still visibly
working during the wait, and the button expands that same row into the full graph. Progressive
disclosure without going silent.

### The gap this plan closes first

The contract §4.2–4.4 and §4.7 define `POST /v1/jobs`, `GET /v1/jobs/{id}`,
`GET /v1/jobs/{id}/events` (SSE) and `GET /v1/traces/{trace_id}`. **None of them are
implemented.** `src/satquery/api/routers/` contains only `analyze`, `validate`, `artifacts`,
`registry`, `health`, and `openapi.json` confirms five paths. The entire streaming design depends
on SSE, so Phase B below adds it. The user chose to build it as part of this work.

---

## Phase B — Backend: jobs + SSE + traces (prerequisite)

Small, additive, touches no tool code.

**B1. Thread an optional event sink through the pipeline.**
`src/satquery/agent/pipeline.py::analyze` and `src/satquery/agent/executor.py::DagExecutor`
already have the exact seams needed — the plan is built at `pipeline.py:222` before
`executor.run`, and `DagExecutor.run` appends each `Execution` at `executor.py:279`/`:348`.
Add an optional `emit: Callable[[str, dict], None] | None = None` parameter to
`DagExecutor.__init__` and to `analyze(...)`, called at:

| Call site | Event |
|---|---|
| after `task_classifier.classify` / before planning | `stage{stage:"planning", pct}` |
| after `planner.plan_for` | `plan{steps:[PlanStep,…]}` |
| `DagExecutor._run_step` entry | `step_started{step, tool, est_ms}` |
| each artifact materialised (`_materialize`) | `artifact{ArtifactRef}` |
| after each `report.executions.append` | `step_completed{step, status, duration_ms, confidence, output_refs}` |
| aggregation | `stage{stage:"aggregating"}` |

Default `None` keeps `/v1/analyze` and every existing test byte-identical. Do **not** change any
tool signature — the whole point of the registry is that adding a tool never touches the planner.

**B2. `src/satquery/api/jobs.py` — in-process job store.**
`dict[str, Job]` guarded by an `asyncio.Lock`, plus one `asyncio.Queue` per job as the event fan-out.
`Job` holds `status`, `stage`, `step`, `total_steps`, `pct`, `created_at`, `updated_at`,
`result: AnalyzeResponse | None`, `error: ApiError | None`, and the buffered event list (so a
client that connects late replays from the start — a judge who opens the tab a second after
submitting must still see the DAG fill in). Register it in `dependencies.py` alongside
`get_artifact_store` / `get_trace_store`.

**B3. `src/satquery/api/routers/jobs.py`.**
- `POST /v1/jobs` — same multipart body as `analyze.py:29` (reuse `parse_options` and
  `spooled_uploads` from `api/uploads.py` verbatim). Spawns an `asyncio.Task` running the same
  `ingest` + `run_analysis` sequence with `emit` bound to the job's queue. Returns `202`
  `{job_id, status:"queued", poll_url, events_url}`. `job_id` **is** the trace id — pass it in as
  `AnalysisRequest.trace_id`, which already exists as a field (`pipeline.py:52`).
- `GET /v1/jobs/{job_id}` — the poll shape of §4.3.
- `GET /v1/jobs/{job_id}/events` — `StreamingResponse(media_type="text/event-stream")`. Replay
  buffered events, then drain the queue. `: ping` comment every 15 s. Emit in the guaranteed §5
  order; `done`/`error` are terminal and must close the stream.
- Map the §6 error taxonomy the same way `analyze.py` does, but into an `error` event rather than
  an HTTP status — a job that fails after `202` has no status code left to use.

**B4. `GET /v1/traces/{trace_id}`** — a five-line router over the existing
`TraceStore` (`src/satquery/trace/store.py`), `404` when absent. The store already persists the
full `AuditTrace` as JSON with `trace_id` as primary key, so this is a lookup, not a feature.

**B5. Regenerate `openapi.json`** via `scripts/export_openapi.py`. This is the frontend's type
source, so it must be regenerated *before* Phase F1.

**B6. Tests** in `tests/contract/`, matching the existing style: assert the SSE event ordering
invariant of §5, that `job_id == trace_id`, that a late subscriber gets the full replay, and that
a `DEGRADED` execution still terminates in `done` (never `error`) per the §4.1 guarantee.

---

## Phase F — Frontend

Location: `frontend/` at the repo root. Add it to `.gitignore`'s `node_modules` and nothing else.

### Stack

- **Vite 6 + React 19 + TypeScript strict.** No Next.js: no SSR value, the API is a local
  FastAPI, and Vite's HMR is the faster loop.
- **`openapi-typescript`** over the committed `openapi.json`, as `npm run gen:api` →
  `src/api/schema.d.ts`. Hand-writing `1.0` types is a defect; §8.1 explicitly recommends this.
- **Tailwind CSS v4** (CSS-first `@theme`) + **shadcn/ui** + **lucide-react**.
- **`@xyflow/react` v12** + **`dagre`** for layout — **lazy-loaded**. It must sit behind a
  `React.lazy()` boundary inside the pipeline modal so it is not in the initial bundle; the DAG is
  no longer on the critical path and should not cost the first paint.
- **`react-compare-slider`** for the A/B pre-post swipe, and **`react-zoom-pan-pinch`** for the
  viewer's pan/zoom. Both are small and beat hand-rolling pointer maths.
- **`@tanstack/react-query`** for `/v1/registry`, `/v1/health`, `/v1/traces/{id}`.
- **`@microsoft/fetch-event-source`** for SSE. **Not `EventSource`** — the job is created by
  `POST /v1/jobs` with a multipart body and `EventSource` is GET-only; also needed for the
  buffered-replay reconnect.
- **Zustand** for the live job store.
- **`motion`** (Framer Motion v11+) for the pipeline drawer transition, KPI card entry and the
  evidence-tray thumbnail streaming.
- **shadcn `Dialog`/`Drawer`** for the progressive-disclosure pipeline surface.
- **MSW** for a mock server driven by `src/satquery/api/fixtures.py`'s canonical §7.4 bi-temporal
  scenario, including a scripted mock SSE stream. This is both the offline dev path and the
  demo-day fallback.

### F0 — Scaffold and design tokens

`npm create vite`, Tailwind v4, shadcn init, path aliases, ESLint/Prettier. Write
`src/styles/theme.css` as the single token file — **one warm earth-toned light theme, no dark
mode**:

```css
@theme {
  /* ── surfaces ─────────────────────────────────────────────────────────── */
  --color-bg-main:      #FAF6F0;  /* warm cream — the application ground     */
  --color-surface-card: #FFFFFF;  /* raised cards: viewer, KPI, answer       */
  --color-surface-sand: #DFA878;  /* tan — borders, muted fills, dividers    */
  --color-sidebar:      #6C3428;  /* dark brown — left navigation column     */
  --color-sidebar-text: #F5E9E1;  /* on-sidebar foreground                   */

  /* ── text ─────────────────────────────────────────────────────────────── */
  --color-text-hi:      #2B1D17;  /* deep warm near-black                    */
  --color-text-lo:      #6B5B52;  /* warm muted grey-brown                   */

  /* ── accents ──────────────────────────────────────────────────────────── */
  --color-accent-warm:  #BA704F;  /* rust — secondary accents, icons, rules  */
  --color-accent-cool:  #CEE6F3;  /* light blue — ACTIVE SELECTION only      */
  --color-evidence:     #F97316;  /* citation pills ONLY — never a status    */

  /* ── status (ToolStatus ∪ CheckStatus) ────────────────────────────────── */
  --color-ok:   #4A7C59;          /* forest green, tuned for the cream bg    */
  --color-warn: #F59E0B;          /* amber — WARN + the uncited underline    */
  --color-fail: #B3261E;
  --color-skip: #9C8F84;          /* warm grey                               */
}
```

**Three rules, enforced by review:**

1. The four status colours map 1:1 onto **both** `ToolStatus` (`OK/DEGRADED/FAILED/SKIPPED`) and
   `CheckStatus` (`PASS/WARN/FAIL/SKIP`) — the same colour means the same thing in the
   compatibility table, on the KPI cards and in the pipeline modal. Status greens/reds are
   deliberately desaturated toward the earth palette; `--color-warn` stays at the specified
   `#F59E0B` because it does double duty as the hallucination underline and must stay loud.
2. `--color-accent-cool` (`#CEE6F3`) means **"this is currently selected"** and nothing else —
   active evidence thumbnail, active sidebar row, focused KPI card. Using it decoratively
   destroys the only selection signal in the UI.
3. `--color-evidence` is reserved for citation affordances so grounding never reads as a status,
   and `--color-warn` is the only colour allowed on an uncited span.

**Contrast constraints (these palette values are not all text-safe — plan around it):**

| Pair | Ratio | Verdict |
|---|---|---|
| `#2B1D17` on `#FAF6F0` | ~15:1 | body text — fine |
| `#F5E9E1` on `#6C3428` | ~8:1 | sidebar text — fine |
| `#BA704F` on `#FAF6F0` | ~3.6:1 | **large text, icons, borders only** — never small body text |
| `#F97316` on `#FAF6F0` | ~2.6:1 | **fails as text.** Citation pills must be an orange *fill* with `#2B1D17` text (~5.8:1), or an orange underline under dark text — never orange text on cream |
| `#2B1D17` on `#CEE6F3` | ~13:1 | selected-state text — fine |
| `#2B1D17` on `#DFA878` | ~8:1 | fine, but reserve sand for chrome rather than text panels |

Fonts: `Inter Variable` with `font-variant-numeric: tabular-nums` on every metric; `Geist Mono`
for ids, tool names, CRS strings and scalar values. KPI numerals get a display size (36–44px)
with tabular figures so the cards do not reflow as digits stream in.

### F1 — Typed API client and the SSE reducer

`src/api/client.ts` — thin `fetch` wrappers over the generated types. Base URL from
`VITE_API_BASE` (default `http://localhost:8000`).

`src/state/job.ts` — the heart of the app. A Zustand store whose only mutation path is a
**reducer over the SSE event union**. The §5 ordering guarantee makes this a clean state machine,
not defensive parsing:

```ts
type JobEvent =
  | { type: 'queued';         data: { job_id: string } }
  | { type: 'stage';          data: { stage: Stage; pct: number } }
  | { type: 'plan';           data: { steps: PlanStep[] } }
  | { type: 'step_started';   data: { step: number; tool: string; est_ms: number } }
  | { type: 'step_completed'; data: { step: number; status: ToolStatus;
                                      duration_ms: number; confidence: number;
                                      output_refs: string[] } }
  | { type: 'artifact';       data: ArtifactRef }
  | { type: 'answer_delta';   data: { text: string } }
  | { type: 'done';           data: AnalyzeResponse }
  | { type: 'error';          data: ApiError };
```

Note the client-side status widening: the contract's `ToolStatus` has no `PENDING`/`RUNNING`. The
store holds `NodeState = 'PENDING' | 'RUNNING' | ToolStatus`, where `PENDING` is set for every
step the moment `plan` lands and `RUNNING` on `step_started`. An unknown enum value is a contract
violation and must be surfaced, never coerced (§2).

**Definition of done for F1:** an integration test drives the reducer with a recorded event
sequence and asserts the terminal state equals the `done` payload.

### F2 — Upload and pre-flight validation

`POST /v1/validate` fires on file *select*, before the user types (§8.2). It is GPU-free, so
there is no reason to make the user wait for it later. It renders **in the centre column**, which
is the Data Stage's empty state: drop target → manifest cards → compatibility summary. The full
ten-row check table lives behind a `Compatibility details` disclosure, consistent with the
progressive-disclosure pattern; the collapsed summary shows only `overall` and the count of
non-`PASS` rows.

- One card per `InputManifest`: thumbnail, `modality` + `modality_confidence`, `sensor_guess`,
  CRS, `gsd_m`, `width×height`, `band_count`, `band_names`, `nodata_pct`, `acquisition_time`.
  A `null` field renders as `—` with the matching `warnings[]` entry beside it — never `0` or
  blank, per §1's nulls rule.
- The **compatibility battery** as a real table: all ten `CheckName`s including `SKIP` rows, each
  with `status` dot, measured `value`, the human-readable `threshold` string, and `detail`
  verbatim (§3.2 states it is safe to display as-is).
- `actions_taken[]` as a short list ("reprojected img_1 EPSG:4326→EPSG:32643 (bilinear)") —
  cheap, and it shows the server did real geo work.
- `supported_tasks[]` becomes suggested-question chips seeded into the right column's query input;
  every other task is rendered **disabled with a reason**, not hidden.
- `overall: FAIL` blocks submission and surfaces the §6 error `message` + `hint`.

### F3 — App shell and the left navigation column

`src/components/shell/AppShell.tsx` — a CSS grid, `240px 1fr 420px`, full viewport height, no
page scroll (each column scrolls independently). Below 1280px the right column collapses to a
bottom sheet; below 900px the left column collapses to an icon rail. Mobile is out of scope.

**Left column** — solid `--color-sidebar` (`#6C3428`), `--color-sidebar-text` foreground:

- Wordmark, then primary navigation: **Explore · Datasets · Tools · History**.
  - *Explore* is the main dashboard (everything below).
  - *Datasets* lists the local fixture scenes under `data/processed/views/`.
  - *Tools* is the capabilities panel: all 13 tools from `GET /v1/registry`, grouped by
    `ToolCategory`, with `available: false` shown **disabled and carrying
    `unavailable_reason`** — §4.8 requires them visible, not hidden.
  - *History* lists prior runs by `trace_id`, rehydrating via `GET /v1/traces/{id}` (the Phase-B4
    endpoint). This is where the audit-trail deliverable becomes navigable.
- The active row is filled `--color-accent-cool` with `--color-text-hi` text — the one selection
  colour, per F0 rule 2.
- Pinned to the bottom: the health strip from `GET /v1/health` (see F6), compact enough to live
  in the sidebar rather than eat a top bar.

### F4 — The Data Stage (centre column)

Three stacked regions in one scroll container. This column is the hero.

**F4a — The image viewer (top, ~55% height).**
A large `--color-surface-card` panel with a `--color-surface-sand` hairline border.

- For `BI_TEMPORAL`: an **A/B swipe slider** (`react-compare-slider`) over the pre and post views
  of whichever view type is active, with `pre`/`post` corner labels taken from `ImageRole`, plus a
  keyboard-draggable handle. This is the money shot of the change-detection demo.
- For `SINGLE` / `CROSS_MODAL`: a single pane; cross-modal offers an optical↔SAR toggle that
  reuses the same slider component, since the pairing is the same shape.
- Pan/zoom via `react-zoom-pan-pinch`, with zoom state **shared across view switches** so
  toggling TC → NDVI keeps the user's framing.
- Overlay controls: opacity slider for `CHANGE_MASK` / `OVERLAY_PNG` composited over the `TC`
  base; `BBOX_SET` boxes drawn from `inline.boxes[].bbox_px` with label + score.
- A **static** legend for index and SAR views — correct precisely because the renderer fixes the
  domains (NDVI/NDWI/NDBI −1…+1, SARDB −25…0 dB). Read those constants from `SCALE_NOTES` in
  `src/satquery/render/view_labels.py`; do not re-derive them client-side.

**F4b — The evidence tray (middle, fixed ~120px).**
A horizontally scrolling strip of thumbnails, one per `RENDERED_VIEW`, appended live on each
`artifact` SSE event with a short fade-in — evidence visibly accumulating is the streaming signal
the centre column contributes.

- The **active** thumbnail gets a 2px `--color-accent-cool` (`#CEE6F3`) ring plus a tinted
  backplate; inactive thumbnails get a `--color-surface-sand` hairline. Clicking one drives F4a.
- Each caption uses the exact label grammar from `view_labels.py` — "Image 1 (optical NDVI
  heatmap, pre-change, Sentinel-2, fixed scale −1 to +1)" — truncated with the full string on
  hover. The view vocabulary is fixed: `TC, FCIR, SWIR, NDVI, NDWI, NDBI, SARFC, SARDB, PAN,
  CHANGE`.
- Group by image slot (`img_0` / `img_1`) with a slot divider, so a bi-temporal run reads as two
  banks rather than one undifferentiated row.
- Artifacts are `Cache-Control: immutable` (§4.6) — preload every `ArtifactRef.url` on arrival.
  Note the mixed extensions: reflectance composites are JPEG, measurement views are PNG (§9 row 11).
- Non-`RENDERED_VIEW` artifacts (`GEOJSON`, `SCALARS`, `TEXT`) do not belong in the tray; they
  surface as KPI cards (F4c) or in the pipeline modal (F5b).

**F4c — Key Insights (bottom, KPI cards).**
Three to five cards derived from `fact_sheet`, e.g. "**7.4 %** Built-up expansion", "**1.07 km²**
Area changed", "**0.86** Confidence".

The contract note that makes this non-trivial: `fact_sheet` is a **free-form, tool-namespaced
map** (`<tool>.<scalar>`, §9 row 4) — there is no "headline metric" field to read. So add
`src/kpi/registry.ts`: a hand-curated lookup keyed by namespaced scalar name giving
`{ label, unit, format, priority, goodDirection? }`, e.g.
`'change_statistics.changed_area_pct' → { label: 'Built-up expansion', unit: '%', priority: 100 }`.
Selection rule: take the highest-priority known keys present, and if fewer than three match, fall
back to auto-formatting the first unmatched numeric scalars (de-namespaced, humanised). The
curated path makes the demo scenarios read beautifully; the fallback means an unanticipated tool
never produces an empty row. Seed the registry from the §7 worked examples and
`configs/registry.yaml`'s `scalars_schema` blocks.

Cards animate their numerals in as `step_completed` scalars land, use `--color-accent-warm` for
the unit/label rule, and are click-linked to the citation that references them (F5a). A card whose
producing step was `DEGRADED` carries a small status dot in `--color-warn` — never hide `DEGRADED`
(§8.6).

### F5 — The Interrogation Thread (right column) and progressive disclosure

**F5a — Query and grounded answer.**
Query input pinned top with the `supported_tasks` chips beneath it. The answer streams from
`answer_delta`; the terminal `done` payload then **replaces** the accumulated text, so citation
offsets are computed once against authoritative text rather than a partially streamed string.

- **Citations.** `citations[].source` parses as `step:{n}/scalars.{dotted.path}`, with `|` joining
  multi-scalar claims. Match each `claim` substring in `text` and wrap it as an **orange citation
  pill** — `--color-evidence` fill with `--color-text-hi` text (per F0's contrast table, orange
  text on cream fails; the fill does not). Hover → popover naming the resolved tool
  (`plan.steps[n].tool`), the dotted scalar path and the `value`.
- **Clicking a citation** — the interaction that replaces the old canvas fly-to, since the DAG is
  no longer on screen — does three things at once: highlights the matching **KPI card** in F4c,
  switches the evidence tray to the view that step produced (via `Execution.output_refs`), and
  sets a pending focus so that *if* the user then opens the pipeline modal, it opens centred on
  that node. Grounding stays one click away without the graph being present.
- **`uncited_numeric_spans`.** Amber (`--color-warn`) wavy underline on each span
  (`text-decoration: wavy underline`), tooltip "not grounded in any tool output", plus a count
  badge on the answer header. Non-empty is a *feature being displayed*, not an error state — §8.5.
- **`template_fallback: true`** gets its own visible badge: the answer was written by the
  template, not the VLM.
- **Confidence** below the answer: `overall` as a ring, the four `components` as a small bar set,
  and `caps_applied[]` listed explicitly when non-empty — a clamped confidence should say which
  rule clamped it.

**F5b — `View Processing Pipeline` (the progressive-disclosure boundary).**

Directly under the answer sit two things:

1. **The live pipeline pulse** — a single compact row of pills, one per `plan.steps[]`, rendered
   the instant the `plan` event lands and updating live: `PENDING` (sand outline) → `RUNNING`
   (pulsing `--color-accent-warm`) → the `ToolStatus` colour, with `duration_ms` under each on
   completion. Roughly 32px tall. This is what keeps the machine visible during the wait without
   putting a graph on the main screen, and it is the answer to the §8.4 tension flagged in the
   Context section.
2. **A secondary button, `View Processing Pipeline`** — outlined in `--color-accent-warm`, with a
   step count and a degraded-count badge when any step is not `OK`.

Clicking it opens a **full-screen shadcn `Dialog`** (a right-side `Drawer` at narrower widths)
containing the `@xyflow/react` canvas. The whole DAG module is behind `React.lazy()` and is
**never mounted on the main screen** — that is the hard requirement of this pattern, and it should
be asserted by a test (`expect(document.querySelector('.react-flow')).toBeNull()` before the
button is clicked).

Inside the modal, unchanged from the previous direction because none of it depends on the shell:

- Left-to-right `dagre` layout, drawn with every node `PENDING` the moment `plan` lands, so
  reopening mid-run shows live state.
- Node card: tool name (mono), `ToolCategory` glyph, device chip (`rocm:0` / `cpu`), and an
  `est_ms`-driven determinate progress bar started by `step_started` that snaps to real
  `duration_ms` on `step_completed`. Cache hits get a bolt glyph; `confidence` renders as a ring.
- `DEGRADED` nodes get a diagonal hazard stripe and a `fallback_of: <tool>` badge (§8.6).
- Edges from `PlanStep.depends_on`, dash-flow animated while the downstream node is RUNNING; edge
  hover shows `PlanStep.reason` (`policy_table:CHANGE_VQA|BI_TEMPORAL|optical`) — that string is
  the proof the planner is declarative rather than hardcoded.
- Selecting a node opens an inspector rail *inside the modal*: its `params` (the auditability
  payload of §3.8), its `scalars`, its `output_refs` previewed inline.
- A tab in the modal holds the **FactSheet** table (namespaced `<tool>.<scalar>`, grouped by tool,
  monospace values) and another holds the raw `AuditTrace` JSON viewer with copy + download.
  Requirement 5's deliverable is one click from the answer, not on the default screen.

The modal is also the destination for the F2 `Compatibility details` disclosure, so there is
exactly **one** "show me the machine" surface in the product rather than several.

### F6 — Chrome, resilience, demo mode

- Health strip pinned to the **sidebar footer** (not a top bar — the three columns should reach
  the top edge) from `GET /v1/health`: `status` dot, device backend + name,
  `vram_used_mb/vram_total_mb`, `tools_available/tools_total`, and **`igpu_masked`** as a
  pass/fail chip — §4.9 calls a `false` there a red flag before any demo, so it must be visible
  without clicking. Per-model `loaded` + `adapter` on hover.
- Error surfaces render `message` and `hint` verbatim from the §6 taxonomy; `VRAM_EXHAUSTED` gets
  a retry button since the contract marks it retryable.
- **Partial results are the normal path**, not an error path: a `FAILED` step must still render
  the answer, the evidence tray entries that did get produced, the KPI cards that resolved, and
  the reduced confidence (§4.1 guarantee).
- `?mock=1` forces the MSW path so the whole UI demos with the GPU off.
- Keyboard: `/` focuses the query input, `←`/`→` step the evidence tray, `[`/`]` nudge the A/B
  slider, `P` toggles the pipeline modal, `Esc` closes it.

---

## Build order

| # | Phase | Blocks |
|---|---|---|
| 1 | B1–B5 backend jobs/SSE/traces + regenerate `openapi.json` | everything typed |
| 2 | F0 scaffold + tokens | — |
| 3 | F1 typed client + SSE reducer + MSW mock stream | F4, F5 |
| 4 | F3 app shell + left navigation | F4, F5 |
| 5 | F2 upload + validate (centre-column empty state) | first user-visible milestone |
| 6 | F4a/F4b viewer + evidence tray | the hero |
| 7 | F5a answer, citation pills, uncited underlines | the second hero |
| 8 | F4c KPI cards + `src/kpi/registry.ts` | needs F5a's citation links |
| 9 | F5b pipeline pulse + lazy DAG modal | last, by design — it is the disclosed layer |
| 10 | F6 chrome, error states, demo mode | |
| 11 | B6 contract tests for SSE ordering | can run parallel to 6–10 |

Milestone 3 is the real checkpoint: once the reducer runs green against the MSW stream, 6–9 are
largely independent. Note the inversion from the previous plan — the DAG is now built **last**
rather than first, because nothing in the main screen depends on it. If the schedule slips, F5b's
pipeline pulse alone (step 9, first half) still carries the transparency story and the modal can
land after.

---

## Verification

**Backend (Phase B).**
```bash
uv run pytest tests/ -q                       # existing suite must stay green — B1 is additive
uv run python scripts/export_openapi.py       # then: git diff openapi.json  → 4 new paths only
uv run uvicorn satquery.api.app:app --reload
curl -N -X POST localhost:8000/v1/jobs -F images=@a.tif -F images=@b.tif -F query='How much built-up area appeared?'
curl -N localhost:8000/v1/jobs/<id>/events    # assert order: queued → stage* → plan → step_*/artifact* → answer_delta* → done
```
Assert by eye that `job_id == trace_id`, then `curl localhost:8000/v1/traces/<id>` returns the
full `AuditTrace`.

**Frontend.**
```bash
cd frontend && npm run gen:api && npm run typecheck && npm run test && npm run dev
```
- `npm run test` covers the SSE reducer against a recorded event sequence, the
  `step:{n}/scalars.{path}` citation parser (including the `|` multi-scalar form), and the KPI
  selection rule in `src/kpi/registry.ts` (curated hit, and the fallback when fewer than three
  keys match).
- **Progressive-disclosure assertion:** a test must confirm no `@xyflow/react` instance is mounted
  on first paint (`document.querySelector('.react-flow')` is `null`), that it appears only after
  `View Processing Pipeline` is clicked, and that the DAG chunk is code-split out of the entry
  bundle (`npm run build` → the xyflow chunk is separate from `index-*.js`).
- Open `http://localhost:5173/?mock=1` — the full flow must run against MSW with the backend down.
- Then run live against `uvicorn` with the four §7 worked-example scenarios: single optical,
  single SAR, cross-modal, bi-temporal. Local fixtures exist under
  `data/processed/views/bigearthnet_v2/` and `data/processed/views/vrsbench/`.
  Bi-temporal specifically must be checked for the A/B slider, since it is the only scenario that
  exercises it.
- Deliberately break one tool (`options.disable_tools`) and confirm the UI renders a `DEGRADED`
  pulse pill, its `fallback_of` badge inside the modal, the `--color-warn` dot on the affected KPI
  card, a reduced confidence, *and* a complete answer — the partial-render guarantee is the single
  most important behaviour to verify, because it is the one the demo will actually hit.
- Contrast audit: run axe (or Lighthouse) over the Explore screen and confirm no citation pill,
  KPI label or sand-bordered control falls below AA — the F0 table flags `#BA704F` and `#F97316`
  as the two values that will fail if used as small text.

## Out of scope

Auth (§1: none, deliberately), multi-turn conversation (the backend is stateless and single-shot —
any thread UI would imply a memory that does not exist), a slippy map / basemap tiles, a dark
theme, and mobile layouts.
