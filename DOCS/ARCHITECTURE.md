# SatQuery AI — Architecture

This is the technical deep-dive. It follows one request from upload to
rendered answer, then goes back over the three subsystems that make the
result trustworthy: the **Data Engine** that produced the adapter, the
**Inference layer** that serves it safely, and the **Frontend** that makes the
audit trail visible.

Wherever this document states a rule, the rule is enforced in code and usually
by a test; file paths are given so the claim can be checked.

---

## 1. End-to-end request flow

```
                 ┌──────────────┐
  upload(s) ───▶ │ /v1/validate │ ──▶ InputManifest[] + CompatibilityReport + supported_tasks
                 └──────────────┘        (no GPU touched)
                        │
  question ──────▶ POST /v1/jobs (202) ──▶ job_id
                        │
                 GET /v1/jobs/{id}/events   ← Server-Sent Events
                        │
     ┌──────────────────┴───────────────────────────────────────┐
     │  ingesting → validating → rendering → planning →         │
     │  executing (one `step_*` + `artifact` event per tool) →  │
     │  aggregating → done                                      │
     └──────────────────────────────────────────────────────────┘
                        │
                 AnalyzeResponse: answer{text, citations, uncited_numeric_spans}
                                  confidence{overall, components, caps_applied}
                                  resolved_task, artifacts, trace (AuditTrace)
```

**Pre-flight** (`src/satquery/ingest/`, `api/routers/validate.py`). Each upload
becomes an `InputManifest`: driver, modality (optical/SAR) with a confidence,
sensor guess, CRS, affine transform, native bounds, GSD, band count and names,
dtype, no-data fraction, acquisition time. Every field that cannot be read is
`null` with a reason in `warnings` — never a default that looks like a
measurement. A `CompatibilityReport` runs ten checks over the pair (CRS
agreement, footprint overlap, resolution ratio, temporal order, …) and the
registry's capability matcher turns the result into `supported_tasks`, which
the UI uses to offer only questions this pair can honestly answer.

**Classification and planning** (`agent/task_classifier.py`, `agent/planner.py`,
`configs/policy_table.yaml`). The question is resolved to a `TaskType`
(`CHANGE_VQA`, `GROUNDING`, `COUNT`, `CROSS_MODAL_COMPARE`, …) by rules first
and an LLM slot-filler second; an LLM-proposed task caps the final confidence
at 0.75. The policy table maps (task, pair type, modality) to a plan — a DAG of
registered tools with declared inputs and outputs.

**Rendering** (`render/`). Before any tool runs, the frozen view catalogue
(`DATA_ADAPTATION_PLAN.md §2`) renders up to six evidence views per scene —
true colour, false-colour IR, SWIR, NDVI/NDWI/NDBI on a fixed −1…1 domain, SAR
false colour and backscatter in dB — each with a label like
`Image 3 (optical false colour infrared, pre-change, Sentinel-2, 2019-04-12)`.
That label string is shared between training and serving (`view_labels.py`);
it is what the model was trained to read, and what the frontend displays
verbatim under the viewer.

**Execution** (`agent/executor.py`, `tools/`). The DAG runs with per-step
status (`OK`, `DEGRADED`, `FAILED`, `SKIPPED`). Every deterministic tool writes
its scalars into the **FactSheet** under namespaced keys —
`change_statistics.changed_area_pct`, `spectral_index_analyzer.ndbi_mean`,
`sar_backscatter_analyzer.sigma0_vv_db_mean` — and its rasters into the
artifact store. Each artifact is emitted as an SSE event the moment it exists.

**Synthesis and validation** (`agent/aggregator.py`, `evidence/`). The VLM step
receives the views and the fact sheet and writes prose. The CitationValidator
then audits that prose (§3.3 below). The confidence scorer combines tool
agreement with a ladder of caps. The whole thing — inputs, plan, executions,
fact sheet, answer, citations, confidence — is persisted as an `AuditTrace` in
SQLite and is fetchable at `/v1/traces/{trace_id}` for as long as the server
runs.

---

## 2. The Data Engine

The adapter is only as honest as the corpus it was trained on, and a
remote-sensing corpus assembled from public benchmarks has two specific ways
to go wrong: the same tile appearing under two names, and a benchmark's test
split leaking into training. `src/satquery/training/corpus_builder.py` and
`scripts/build_corpus.py` exist to make both impossible rather than unlikely.

### 2.1 Sources, and reading them correctly

| Source | Train / val (v2) | Role |
|---|---|---|
| BigEarthNet-v2 (Sentinel-1 + Sentinel-2, 10 m) | 16,200 / 1,800 | sensor-physics grounding — the mandated dataset |
| VRSBench (VHR, 0.1–3 m) | 18,000 / 2,000 | VQA, captioning, grounding at the evaluation set's resolution |
| RSVQA-HR | 9,000 / 1,000 | counting / presence / comparison |
| CDVQA (bi-temporal pairs) | 7,198 / 802 | change VQA |
| evidence_qa (synthetic) | 2,700 / 300 | citation behaviour: answers that quote a fact sheet |
| DIOR-RSVG | — | referring-expression grounding; gated on Hugging Face, excluded until access is granted |
| **Total** | **53,098 / 5,902** | one merged corpus, one adapter (`runs/sq-lora-v2-full`) |

Three of the sources do not ship as Hub tables, and reading them with
`datasets.load_dataset` was silently wrong in three different ways.
`training/local_sources.py` holds the readers that are right:

- **VRSBench** publishes annotation zips, not a table. The reader opens the
  archives and joins on image id.
- **RSVQA-HR** publishes one JSON per split that contains *all* 955,664
  questions, with the split's members marked `active: true`. Reading the train
  file at face value pulls 330,324 test and Philadelphia rows into training.
  The reader filters on the flag, and the quarantine check (§2.2) turns a
  regression here into a build error.
- **CDVQA** is a WebDataset of 1,533 tar shards. `scripts/fetch_sources.py
  --source cdvqa` unpacks them into pre/post PNG pairs and flat rows, because
  the render pass has to address images by path.

Every builder emits one **unified instruction sample** (`§3` of the data
plan): the list of views, their labels, the fact sheet the sample is
conditioned on, and the assistant turn. Boxes in assistant turns are serialised
by one function, `models/prompts/box_format.serialise`, into Qwen's native
normalised form — the same function whose parser the serving path and the
frontend use. Training and inference cannot drift apart because there is
nothing to drift.

### 2.2 Deduplication and leakage control

`build_corpus.py` runs, in order, across *all* sources:

1. **Exact dedup** on the source image's SHA-256.
2. **Near-duplicate dedup** on a DCT perceptual hash (greyscale, 32×32, low
   8×8 block minus DC, 64 bits), Hamming distance ≤ 5.
3. **A hard assertion** that no image in `train.jsonl` appears, by either
   hash, in any quarantined test split (VRSBench test, RSVQA test, CDVQA test,
   DIOR-RSVG test). This is a build error, not a warning: a leaked benchmark is
   worse than no benchmark.

One subtlety made the naive version useless. Four sources publish many samples
per image — VRSBench ~10 annotations per tile, RSVQA-HR ~100 questions,
CDVQA ~40 per pair. Deduplicating *samples* collapses VRSBench from 205,121
candidates to 20,264 and CDVQA from 9,000 to 220. `Deduplicator.check` therefore
takes an `image_key`: a repeat of an already-admitted image by the *same* key
is legitimate re-use; a collision between two *different* images is exactly as
fatal as before.

The builder streams: rows are hashed and admitted as they pass, a bounded hash
cache (`HASH_CACHE_SIZE = 65,536`) keeps memory flat, and each source is
reservoir-sampled to its §5 target so composition is set by the plan rather
than by whichever source happens to be largest. `--limit` stops each split
early for smoke runs.

### 2.3 Training profile

`configs/train/qlora_qwen3vl8b_rocm24g.yaml`: Qwen3-VL-8B-Instruct, base
weights loaded in NF4 with double quantisation and bf16 compute, LoRA r=16
α=32 on every LLM projection and the last 8 vision blocks, max sequence 4,096,
per-device batch 1 × 16 accumulation, one epoch, lr 1e-4 cosine, bf16 autocast.
Loss is computed on the answer only: each sample becomes a prompt-completion
record with `completion_only_loss`, and `audit_masks()` checks the collated labels
before the weights load. The user turn is laid out by `models/prompts/layout.py` —
the same function serving uses — and its fingerprint is written beside the adapter
and checked at load. On the 24 GB ROCm card the v2 run took 40.7 hours
(3,318 steps, 13.4 GiB peak); answer-token accuracy on the eval split went from
26.4 % zero-shot to 81.8 %. A bf16 LoRA profile for the 4B model exists as the OOM
fallback. Why the first run (v1) learned nothing is recorded in
`ML_PIPELINE_RECOVERY_PLAN.md`.

---

## 3. The Inference layer

### 3.1 Backends

`models/loader.py` exposes `get_backend()` (loads on first call) and
`current_backend()` (reports without loading, so `/v1/health` polling never
triggers a 16 GB load). Two backends implement the same interface:

- **`hf`** — `transformers` in-process. The production path with the adapter.
- **`llamacpp`** — an HTTP client to a `llama-server` holding the merged
  (base + adapter) Q4_K_M GGUF and an f16 mmproj, exported by
  `scripts/merge_export.py`. Starts in seconds, ~6 GB, survives a broken ROCm
  install, and is what the 8 GB demo laptop runs.

Both apply stop sequences *after* generation rather than trusting the engine;
a fine-tuned checkpoint does not reliably emit EOS.

### 3.2 bf16 memory safety

The adapter was trained under NF4, so the obvious serving choice is NF4 —
~5 GB of weights. It is the wrong choice on this hardware, and the reason is
recorded in `models/hf_backend.py` because it cost a full training run to
discover: on gfx1100 / ROCm 6.4 / torch 2.9.1, **bitsandbytes 4-bit
generation is broken**. The base weights alone, adapter detached, answer
nonsense under NF4 and answer correctly under bf16. 4-bit *training* is
unaffected, which is why a full training run completed without anyone noticing.

So the serving path is deliberately unquantised: base weights in bf16
(~16.4 GB), LoRA applied over them at load, ~18.7 GiB peak with six views.
That fits a 24 GB card with the guardrails below and does not fit without
them:

- **A VRAM guard in front of the load** (`vlm_vram_budget_gb`). If another
  process — typically a warm `llama-server` — already holds enough of the
  card that the load would OOM, the load is refused up front with a clear
  message rather than failing halfway through and leaving a half-initialised
  model resident.
- **Idle unload** (`vlm_idle_unload_s`) releases the weights after a quiet
  period so the change detector and segmenter can have the card back.
- **View cap of six** per request, set in the render policy, bounds the
  vision-token count and therefore activation memory.
- **`SATQUERY_VLM_DTYPE=bfloat16` is the only value that fits**: fp32 cannot
  load at all, fp16 loses the exponent range the vision tower needs.

### 3.3 The citation and guardrail interceptor

This is the part of the system the rest exists to support.

**FactSheet** (`evidence/fact_sheet.py`). Every scalar a deterministic tool
produces, keyed `<tool>.<scalar>`, with its unit and the step that produced
it. The VLM is shown the sheet; it is *not* allowed to add to it.

**CitationValidator** (`evidence/citation_validator.py`). After generation,
every numeric span in the answer is located by a regex that understands
thousands separators, decimals and the units `% km² m² dB px m` — and refuses
to read "2019" as three digits or "5 meters" as "5 m". Each span is resolved
against the sheet under three rules:

- **A `[key]` marker is binding.** When the model writes `0.82
  [spectral_index_analyzer.ndvi_mean]`, the number must match *that* key's value;
  an unknown key or a mismatched value is flagged (`UNKNOWN_KEY` /
  `KEY_VALUE_MISMATCH`) rather than rescued by a coincidental match elsewhere in
  the sheet. Bare numbers fall back to value search.
- **Units must match, not just magnitude.** `7.4 %` may only cite a `*_pct`
  scalar. `7.4` matching `changed_area_km2` because the digits agree would
  manufacture a citation for something nobody measured. Matching is on
  `_`-delimited *segments* of the scalar name, so `sigma0_vv_db_mean` is
  correctly a mean of decibels.
- **Tolerance is relative above 10, absolute below** (1 % / 0.05), so
  rounding in prose does not break a citation and a genuinely different
  number does not pass as one.

A resolved span becomes a `Citation {claim, source: "step:N/scalars.path",
value}`. An unresolved span goes into `uncited_numeric_spans`. The default
policy is **`flag`, not `strip`**: deleting the sentence would hide the
problem and produce incoherent text. `strip` exists for benchmark scoring only.

Grounding answers are a special case handled by `box_format.strip_boxes`: a
box coordinate is a position, not a claim, and there is no fact-sheet key it
could ever resolve against. Before this, every grounding run was capped for
"uncited" numbers it had been *instructed* to emit.

**Confidence** (`evidence/confidence.py`, method `weighted_tool_agreement_v1`):

```
overall = 0.20·task_classification + 0.20·input_quality
        + 0.35·tool_mean            + 0.25·cross_tool_agreement
```

then a ladder of caps, applied in a frozen order and each recorded by name:

| cap | ceiling |
|---|---|
| `compatibility_warnings` | 0.80 |
| `llm_task_proposal` | 0.75 |
| `degraded_execution` | 0.70 |
| `template_fallback` | 0.65 |
| `uncited_claims` | 0.60 |
| `failed_execution` | 0.50 |
| `generic_plan` | 0.45 |

Three deliberate choices: VLM steps are excluded from `tool_mean` (a fluent
paragraph must not raise confidence in a weak measurement); a single
uncorroborated tool scores 0.75, not 1.0; and caps *clamp* rather than
subtract, so the worst applicable ceiling wins and the trace names it.

**Template fallback** (`agent/aggregator.compose`). If the VLM is disabled,
unavailable or times out, a deterministic writer composes the answer from the
fact sheet by task type, marks `template_fallback: true`, and the confidence
is capped at 0.65. The measurements are still real; only the prose is
templated, and the UI says so.

### 3.4 The API surface

Frozen as OpenAPI schema 1.0 (`openapi.json`, regenerated by
`scripts/export_openapi.py`; the frontend's `schema.d.ts` is generated from
it, so a wire change fails `tsc` before it fails a demo).

| Route | Purpose |
|---|---|
| `GET /v1/health` | device name, VRAM used/total, `igpu_masked`, tools available, schema version |
| `GET /v1/registry` | every registered tool and its capabilities |
| `POST /v1/validate` | pre-flight: manifests, compatibility, supported tasks |
| `POST /v1/analyze` | synchronous analysis (contract tests, scripting) |
| `POST /v1/jobs` → 202 | queue an analysis |
| `GET /v1/jobs/{id}` | poll / reattach |
| `DELETE /v1/jobs/{id}` | cancel — stops the job on the server, not just in the UI |
| `GET /v1/jobs/{id}/events` | SSE stream: `stage`, `plan`, `step_started/completed`, `artifact`, `result`, `error`; `id:` + `Last-Event-ID` replay |
| `POST /v1/imagery/fetch` | clip one or two Planetary Computer STAC scenes to a bbox as GeoTIFFs (Maps page) |
| `GET /v1/imagery/{fetch_id}/{name}` | download one fetched GeoTIFF |
| `GET /v1/traces/{trace_id}` | the stored AuditTrace |
| `GET /v1/artifacts/{trace}/{id}.{ext}` | raw bytes of one rendered view or mask |

The job store (`api/jobs.py`) is deliberately in-process — one box, one demo,
no broker — evicting only finished jobs past `MAX_JOBS` so memory is bounded,
with a trace-store fallback for evicted results. Tools share process-wide CPU/GPU
semaphores (`agent/concurrency.py`) so concurrent jobs cannot stack GPU models.
Errors use one §6 envelope (`code`, `message`, `hint`) everywhere.

---

## 4. The Frontend

`frontend/` is a React 19 + Vite 6 + TypeScript 5.9 application styled with
Tailwind v4 design tokens. It has no router library: `shell/router.ts` syncs a
`section` in the UI store with the URL, and `App.tsx` switches between Landing
(`/`, a three.js globe), the Console (below), Maps (MapLibre + STAC discovery),
Projects, Saved (IndexedDB library), Use Cases and the printable Report
(`/report/<trace>`). Heavy sections are lazy chunks; the console entry is held
under 180 KB by `npm run check:bundle`. Exports — the one-page SITREP PDF
(pdf-lib) and RFC 7946 GeoJSON — live in `src/export/`; `USER_GUIDE.md` walks
through all of it. The rest of this section describes the Console.

### 4.1 Layout

A three-column shell (`components/shell/AppShell.tsx`), written mobile-first
with two-dimensional breakpoints — `wide` is `≥768px and ≥500px tall`, `desk`
is `≥1280px and ≥500px tall` — because a landscape phone is wide enough to
trip a width query and far too short for a nav column.

```
┌──────┬────────────────────────────────────────┬──────────────────┐
│ nav  │  top bar: mode chip · system status     │                  │
│ rail ├────────────────────────────────────────┤  thread panel    │
│ 200  │  scene header (task, sensor, CRS, T1/T2)│  Chat · Results  │
│      │  viewer (A/B swipe · zoom · bbox layer) │  Citations ·     │
│      │  evidence strip (one card per view)     │  History         │
│      │  key insights (fact-sheet KPI cards)    │  380             │
└──────┴────────────────────────────────────────┴──────────────────┘
```

On a phone the rail becomes a horizontally scrolling bar and the thread a
bottom sheet capped at `min(55vh, 24rem)`.

### 4.2 Streaming and state

Zustand stores are kept apart by responsibility; the three at the core of the
Console are:

- **`state/job.ts`** — a pure reducer over the SSE event stream. Every event
  type from the contract maps to one state transition; it is tested against
  a recorded real run (`mocks/captured/events.bitemporal.json`) with no DOM.
- **`state/ui.ts`** — client-only: nav section, uploads, pre-flight result,
  session history, the single-key-shortcut preference.
- **`state/focus.ts`** — cross-column focus: which view the viewer shows, the
  swipe position, which KPI is lit, which pipeline step a citation pointed at,
  the composer draft. This is what lets a citation click do three things in
  three columns without prop-drilling.

`api/sse.ts` parses the stream with reconnect and last-event-id; `thread/useRun.ts`
owns the job lifecycle (submit → stream → result / cancel via `DELETE` /
bounded reattach via `thread/resume.ts` if the stream drops). MSW
(`?mock=1`) replays the captured fixtures so the whole UI runs with the
backend down.

### 4.3 Citations, in the interface

`thread/annotate.ts` turns `answer.text` plus `citations[].claim` into
segments. The contract gives claims as substrings, not offsets, and naive
`indexOf` is wrong: with claims `"350,106.12 m2"` and `"1"`, `indexOf("1")`
lands inside the larger number. Two rules fix it — longest claim first, and a
claim beginning or ending in a digit cannot match beside a digit, comma or
period. `uncited_numeric_spans` go through the same machinery and render as
an amber wavy underline with a visually-hidden "(not grounded in any tool
output)" for screen readers.

Clicking a citation calls `focusCitation`: the evidence tray switches to the
view the citing step produced (`evidence/views.groupForScalar`), the KPI card
carrying the same scalar lights its terracotta edge, and the pipeline dialog
is armed to open on that node.

### 4.4 Bounding boxes: parse once, draw in normalised space

`thread/bbox.ts` is a TypeScript port of `box_format.py`'s parser, with the
same three-form cascade — tagged sentinels, JSON `bbox_2d`, and the bare
`label(x1,y1),(x2,y2)` form the fine-tuned checkpoint actually emits — the
same clamp to `[0, 1000]`, the same inversion correction, the same
zero-extent drop and dedupe. It has its own test file mirroring the backend's
rules.

`components/stage/BboxOverlay.tsx` then does no pixel arithmetic at all:

```tsx
<svg viewBox="0 0 1000 1000" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
  <rect x={box.xMin} y={box.yMin} width={box.xMax - box.xMin} height={box.yMax - box.yMin}
        vectorEffect="non-scaling-stroke" … />
</svg>
```

The `viewBox` *is* Qwen's normalised frame, so a box's coordinates are its
SVG coordinates. `preserveAspectRatio="none"` stretches the frame to the
raster's own aspect ratio. Two details make this correct rather than
approximately correct:

- The raster is `object-fit: contain` and is therefore letterboxed inside its
  cell whenever the cell's aspect differs. The overlay is sized to the
  raster's *rendered rectangle* — computed from the image's natural size and
  a `ResizeObserver` on the cell — not to the cell, or every box would miss
  on a wide window.
- The overlay lives inside the zoom/pan transform, so it follows the image;
  `vector-effect: non-scaling-stroke` keeps the stroke at 1.5 screen pixels at
  every zoom and aspect ratio, and labels are HTML so type never distorts.
  `pointer-events: none` lets the swipe handle and the pan pass through.

The chat strips the same tokens before annotation (`stripBboxTokens`), so the
prose reads cleanly and the boxes appear where they mean something.

### 4.5 Design system and its tests

`styles/theme.css` declares the dark palette as Tailwind v4 `@theme` tokens
with one rule per hue: `--color-<hue>` is a fill (≥ 3:1 as a mark),
`--color-<hue>-text` is small text (≥ 4.5:1), `--color-on-<hue>` is the
foreground on that fill. `styles/__tests__/contrast.test.ts` parses the CSS
at test time and computes WCAG luminance for every pair the app actually
renders — including chip washes like `bg-warn/20`, which is where naive
checks miss. Changing a hex that breaks a pair fails the suite rather than
shipping. `motion.test.ts` pins the reduced-motion contract and allows
exactly two infinite animations: the RUNNING pulse and the health glow.
