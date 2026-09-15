# Product

<!-- impeccable:product-schema 1 -->

> Written from the explicit brief plus repository evidence (README, DOCS/API_CONTRACT.md,
> DOCS/AGENT_POLICY_DAG.md, frontend/src). The user asked for no interview round; every
> line marked **[inferred]** is a hypothesis to confirm, not a confirmed fact.

## Platform

web

## Users

- **Primary:** the SatQuery AI team presenting at Smart India Hackathon (problem statement
  26167, ISRO / Space Applications Centre) and the **judges** watching a live demo on a
  single 24 GB consumer-GPU laptop, possibly with no internet. State of mind: evaluating
  dozens of teams, minutes per team, looking for evidence that the system is real.
- **Secondary:** a student or researcher running their own bi-temporal / cross-modal
  scene through the console after the demo. Role label in the UI: `Student · Researcher`.
- **[inferred]** No multi-user accounts exist; the profile card is a single local identity.

## Product Purpose

An agentic vision-language assistant for satellite imagery that **shows its work**: a
query is parsed, inputs pass 11 named compatibility checks, a deterministic policy table
selects and sequences specialist tools from a registry, and the answer is
evidence-grounded — every number bound to a measured scalar or flagged as uncited — with a
versioned, schema-validated audit trace. Success = a judge can see the machine think,
click any number, and land on the measurement that produced it.

## Positioning

"The VLM is one tool among many, never the system." Control flow is deterministic
(byte-identical reruns), the LLM only slot-fills and synthesises, and the
`CitationValidator` flags rather than strips. Neighbouring "chat with your satellite"
products cannot truthfully claim an auditable, tool-bound answer.

## Operating Context

- Runs fully local beside a GPU on a lab network that may have no internet route: fonts
  are self-hosted, nothing may depend on a CDN, basemap tiles are optional.
- Wire contract frozen at schema `1.0` (`openapi.json` ↔ `src/api/schema.d.ts`,
  enforced by `make contract`). Frontend may add UI-only state freely; it may not invent
  API fields.
- Streaming: `POST /v1/jobs` → SSE `plan` / `step_started` / `artifact` /
  `step_completed` / `stage` events. The DAG is drawn from the `plan` event.
- Rendered evidence views: TC, FCIR, NDVI, NDBI, NDWI, SAR VV/VH, CHANGE (from
  `render/view_labels.py`); the frontend groups them (`src/evidence/views.ts`).
- `?mock=1` runs the whole UI on recorded fixtures with the backend down.

## Capabilities and Constraints

- Task types: VQA, CAPTION, SCENE_CLASSIFY, COUNT, GROUNDING, SEGMENTATION, CHANGE_VQA,
  CHANGE_CAPTION, CHANGE_MAP, CROSS_MODAL_VQA, CROSS_MODAL_COMPARE.
- Registry tools: spectral_renderer, raster_statistics, spectral_index_analyzer,
  sar_backscatter_analyzer, change_statistics, object_counter, image_diff_change,
  siamese_change_detector, semantic_segmenter, text_grounding, physics_agreement,
  crossmodal_consistency, vlm_vqa, vlm_caption, vlm_change_vqa.
- **Not performed by the backend:** orthorectification, tiling/basemap serving, user
  accounts, saved projects, notifications. UI features in these areas are client-side
  or clearly labelled as such.
- Stack: React 19, Vite 6, Tailwind v4 (`@theme` in `src/styles/theme.css`), Zustand 5,
  TanStack Query 5, Radix Dialog, @xyflow/react (lazy), react-compare-slider,
  react-zoom-pan-pinch. Tests: vitest + happy-dom, Playwright e2e, a contrast test that
  parses every `--color-*` token out of `theme.css`.
- Sections are a Zustand `section` string, not a router (`src/App.tsx`).
- **[inferred]** Undecided: whether an online basemap (tiles) is permitted during the
  demo. Treat as optional and degrade to scene-only.

## Brand Commitments

- Name: **SatQuery AI**. Tagline: *From Space to Answers*.
- Voice: plain, honest, specific; the code comments set the register ("a 0% bar would
  assert something the server did not say").
- Incumbent visual identity (frontend/src/styles/theme.css): warm near-black ground,
  terracotta accent (borders/icons/active), sand as the accent's text voice and the
  citation-pill fill, emerald / amber / red status vocabulary, Inter Variable + Geist
  Mono Variable self-hosted. Honesty guards are binding: clickable citation pills,
  amber wavy underline for uncited numeric spans.
- **[inferred]** The team wants the identity kept and elevated, not replaced.

## Evidence on Hand

- Recorded fixtures: `frontend/src/mocks/captured/*.json`, `frontend/public/mock-artifacts/`.
- Trained adapter `runs/full-epoch-v1`, limitations documented in `DOCS/project_audit.md` §2.
- Corpus sources: BigEarthNet-v2 (Sentinel-2 / Sentinel-1), VRSBench, CDVQA, RSVQA-HR.
- **Absent, must not be fabricated:** customer logos, testimonials, benchmark leaderboard
  claims, Cartosat scenes, sample rasters for the Use Cases gallery (to be supplied).

## Product Principles

1. Never render a claim the server did not make (nulls stay null, uncited stays underlined).
2. Progressive disclosure: imagery is the hero, the DAG opens on request, but the machine
   is never silent while working.
3. Works offline and on a phone; the keyboard story is part of the product (WCAG 2.1.4
   single-key shortcuts must be switchable off).
4. Everything visible must be reachable: no dead controls, no scaffolding that looks live.
5. Contrast is enforced by test, not by comment.
