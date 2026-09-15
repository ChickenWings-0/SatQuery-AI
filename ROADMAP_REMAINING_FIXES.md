# SatQuery AI — Remaining Work Before the SIH 2026 Final

*Planning document. Nothing in here is implemented yet; each track names the
architecture, the files to touch, and the verification that makes it "done".*

**Status as of 2026-09-15.** The full QLoRA run is complete and lives on `main`:

| Run | `runs/sq-lora-v2-full/adapter` |
|---|---|
| Base | `Qwen/Qwen3-VL-8B-Instruct`, NF4 backbone, LoRA r=16 / α=32 on all LLM projections + last 8 vision blocks |
| Corpus | 53,098 train / 5,902 val (`data/processed/corpus/v2-full/`), seed 42, prompt `grounded_v1` |
| Epoch 1.0 | train loss **0.3025**, eval loss **0.1867**, answer-token accuracy **81.76 %** |
| Probes | 100 % resolution on grounding and cross-modal fact-checking probes (`scripts/test_inference.py`) |
| Adapter size | ~86 MB `adapter_model.safetensors` (96 MB directory with processor + tokenizer) |

Repository policy from today: **one branch, `main`.** `scripts/git/main_only.sh`
refuses commits and pushes from anywhere else (see README → Contributing).

The four tracks below are independent enough to run in parallel. Recommended
order if one person does everything: **3 → 1 → 2 → 4**, because Track 3 makes
the adapter the thing the API actually serves, Track 1 turns that into slide
numbers, Track 2 is the insurance policy for the demo machine, and Track 4 is
what the judges see.

---

## Track 1 — Validation Benchmark Suite (numbers for the slides)

### Goal

One command that runs the served model over a stratified held-out slice of
`val.jsonl` and prints a table we can paste into the deck:

| Source | n | Accuracy | Grounding recall@0.5 / mean IoU | Citation precision | Uncited-number rate |
|---|---|---|---|---|---|
| BigEarthNet-v2 | 100 | | | | |
| VRSBench | 100 | | | | |
| RSVQA-HR | 100 | | | | |
| CDVQA | 100 | | | | |
| Evidence QA | 100 | | | | |
| **All** | 500 | | | | |

Plus a "before" column from the un-adapted base model, since
`scripts/eval_vrsbench_zeroshot.py` already establishes that baseline shape.

### What the val corpus gives us (no new data needed)

`data/processed/corpus/v2-full/val.jsonl` — 5,902 rows, already deduplicated
against the quarantined test splits. Per-source / per-task counts:

| Source | rows | tasks present |
|---|---|---|
| `bigearthnet_v2` | 1,800 | SCENE_CLASSIFY, GROUNDING, VQA, CROSS_MODAL_VQA, CROSS_MODAL_COMPARE |
| `vrsbench` | 2,000 | VQA, GROUNDING, CAPTION, COUNT |
| `rsvqa_hr` | 1,000 | VQA, COUNT |
| `cdvqa` | 802 | CHANGE_VQA (2-view, `pair_type=BI_TEMPORAL`) |
| `evidence_qa` | 300 | VQA, CROSS_MODAL_VQA |

Each row carries `messages` (the last assistant turn is the reference answer),
`views` (rendered JPEGs on disk), `fact_sheet`, and `meta`. Reference formats
that the scorer has to parse:

- **Grounding:** `<|object_ref_start|>label<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>`,
  coordinates normalised to `BOX_SCALE = 1000`
  (`src/satquery/models/prompts/box_format.py` already parses and validates this).
- **Tool-cited scalars:** `... 0.82 [spectral_index_analyzer.ndvi_mean]` — the
  `[tool.scalar]` tag is the citation; the number before it is the claim.
- **Closed VQA:** `Yes.` / `No.` / `Top-left` / class lists — exact match after
  normalisation (lower-case, strip trailing period, split class lists on `, ` /
  ` and `).
- **Captions:** free text — report BLEU-4 + ROUGE-L only, no accuracy claim.

### Architecture

```
scripts/eval_benchmark.py                     CLI entry point
src/satquery/eval/__init__.py                 (package exists, currently empty)
src/satquery/eval/sampler.py                  stratified sample of val.jsonl
src/satquery/eval/scorers.py                  one scorer per task family
src/satquery/eval/runner.py                   batch inference through the LazyBackend
src/satquery/eval/report.py                   markdown + JSON + CSV writers
tests/unit/test_eval_scorers.py               scorers are pure functions → unit-testable
```

**`sampler.py`** — `stratified_sample(path, per_source: int, seed: int) -> list[CorpusSample]`.
Reuse `satquery.training.corpus_builder.CorpusSample` so the eval rows are
byte-identical to what training saw (the L2 parity tests in
`tests/integration/test_train_serve_parity.py` already prove that path). Cap at
`per_source` per source, round-robin across tasks inside a source so 100
BigEarthNet rows are not 90 SCENE_CLASSIFY. Write the chosen ids to
`<out>/sample_ids.json` so the run is reproducible and the "before" column
runs on the identical slice.

**`runner.py`** — builds one `GenerationRequest` per sample from `sample.views`
+ the user turn of `sample.messages` (never the assistant turn), sends it
through `satquery.models.loader.LazyBackend` so `SATQUERY_VLM_BACKEND=hf|llamacpp`
picks the engine with zero code change, and records `{id, source, task,
reference, prediction, latency_ms, prompt_tokens, completion_tokens}` to
`<out>/predictions.jsonl` as it goes (resume-safe: skip ids already present).
Greedy decoding, `max_new_tokens=256`, six-view budget identical to training
(`max_pixels=147456`).

**`scorers.py`** — pure functions, one per family:

| Task family | Metric | Notes |
|---|---|---|
| VQA / CHANGE_VQA / SCENE_CLASSIFY | exact-match accuracy; SCENE_CLASSIFY also set-F1 over CLC labels | normalise before comparing |
| COUNT | exact match on the integer + MAE | parse `count is N` |
| GROUNDING | recall@IoU≥0.5, mean IoU, and *format validity* (did the model emit a parseable box at all) | parse via `box_format.py`; multi-box references → Hungarian match on IoU |
| CROSS_MODAL_VQA / evidence_qa VQA | (a) numeric tolerance: predicted scalar within ±5 % or ±0.02 abs of reference; (b) citation tag matches | this is the "fact-checking" number |
| Every task | **Citation precision** = cited tags that exist in `fact_sheet` ÷ all cited tags; **uncited-number rate** = numeric spans with no `[...]` tag ÷ numeric spans | run `satquery.evidence.citation_validator` on the prediction against the row's `fact_sheet` — same code path the API uses |
| CAPTION | BLEU-4, ROUGE-L | `evaluate`/`rouge-score` are dev-only deps; guard the import |

**`report.py`** — writes `results.md` (the table above, ready for the slide),
`results.json` (every metric, plus git SHA, adapter path, backend, seed,
timestamp, `run_manifest.json` digest), and `results.csv`. Also emits a
`confusion/` folder with the 20 worst grounding IoUs and 20 wrong VQA answers
rendered as `view + reference + prediction` PNG strips — those are the "honest
failure" slide.

### Files to touch

- `scripts/eval_benchmark.py` — new. Args: `--val`, `--per-source 100`,
  `--seed 42`, `--backend`, `--adapter`, `--out runs/eval/<name>`, `--resume`,
  `--baseline` (forces `SATQUERY_VLM_ADAPTER_PATH=` empty for the "before" run).
- `src/satquery/eval/{sampler,runner,scorers,report}.py` — new.
- `tests/unit/test_eval_scorers.py` — new; fixture rows per task family with
  known scores (IoU of two hand-drawn boxes, a citation that exists / does not).
- `Makefile` — `make eval` (100/source, adapter) and `make eval-baseline`.
- `pyproject.toml` — `rouge-score` under `[dependency-groups] dev` only.
- `DOCS/project_audit.md §2` — replace the hand-written limitations paragraph
  with a link to `runs/eval/<name>/results.md`.

### Verification

1. `uv run --no-sync pytest tests/unit/test_eval_scorers.py` green
   (never a bare `uv sync` on the ROCm box).
2. `make eval` on the 24 GB box: 500 samples end-to-end in < 45 min at bf16
   (≈ 4–5 s/sample with six views). `results.md` produced; `sample_ids.json`
   has exactly 100 ids per source.
3. `make eval-baseline` on the same `sample_ids.json`; the adapter column must
   beat the baseline on grounding recall and citation precision — if it does
   not, that is a finding for the slide, not a reason to hide the column.
4. Sanity anchors: the eval-loss row of `runs/sq-lora-v2-full/train.log` and
   the 100 % probe resolution should be consistent with citation precision
   ≥ 0.95 on `evidence_qa`. A large gap means the runner is not reproducing
   the training prompt — re-check against `scripts/preflight_train_serve_parity.py`.
5. Commit `runs/eval/<name>/results.{md,json,csv}` **explicitly** — `/runs/` is
   git-ignored, so the eval outputs get a `!runs/eval/**/results.*` exception
   (already added to `.gitignore`).

---

## Track 2 — LoRA Merge & GGUF Quantisation (air-gapped laptop demo)

### Goal

A single `sq-lora-v2-full-Q4_K_M.gguf` (~5.2 GB) + `mmproj-f16.gguf` (~1.1 GB)
that `llama-server` runs on a **Windows RTX 4070 Laptop (8 GB VRAM)** with the
network cable unplugged, and that the FastAPI backend talks to through the
existing `SATQUERY_VLM_BACKEND=llamacpp` path — no torch, no ROCm, no CUDA
Python stack on the demo machine.

### What already exists

- `scripts/merge_export.py` — merges in **bf16 on the CPU** (deliberately never
  onto NF4 weights), copies the processor + chat template alongside the weights,
  and with `--gguf --llama-cpp <dir>` shells out to llama.cpp's
  `convert_hf_to_gguf.py` and `llama-quantize`. `DEFAULT_QUANT = "Q4_K_M"`.
  `--dry-run` prints the commands.
- `scripts/serve_vlm.sh` — starts `llama-server` with `--image-min-tokens 1024`
  (Qwen3-VL degrades on grounding below that; ggml-org/llama.cpp#16842),
  context 16384, `-ngl 99`.
- `src/satquery/models/llamacpp_client.py` — the OpenAI-shaped client; already
  interchangeable with the HF backend.

### Steps

**2.1 Merge (on the ROCm box, CPU, ~16 GB system RAM):**

```bash
source scripts/rocm_env.sh
uv run --no-sync python scripts/merge_export.py \
    --adapter runs/sq-lora-v2-full/adapter \
    --out models/sq-lora-v2-full-merged
```

Expected: `models/sq-lora-v2-full-merged/` with `model-0000x-of-0000y.safetensors`
(~16.4 GB bf16), `config.json`, `preprocessor_config.json`, `chat_template.json`,
tokenizer files, and a `run_manifest.json` recording the adapter SHA it came from.

**2.2 Convert + quantise (needs a llama.cpp checkout with Qwen3-VL support,
≥ b6600):**

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp ../llama.cpp
cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_HIP=ON && cmake --build ../llama.cpp/build -j   # quantiser only needs CPU
uv run --no-sync python scripts/merge_export.py \
    --adapter runs/sq-lora-v2-full/adapter --out models/sq-lora-v2-full-merged \
    --gguf --llama-cpp ../llama.cpp --quant Q4_K_M
```

Which runs, in order: `convert_hf_to_gguf.py --outtype bf16` (LLM, ~16 GB
intermediate), `convert_hf_to_gguf.py --mmproj --outtype f16` (vision tower +
merger), `llama-quantize <bf16> <Q4_K_M> Q4_K_M`. The mmproj is **not**
quantised — vision-tower quantisation is what breaks grounding first.

**2.3 VRAM budget on the 4070 Laptop (8 GB, of which ~7.2 GB is usable):**

| Component | Size |
|---|---|
| Q4_K_M LLM weights | ~5.0–5.2 GB |
| mmproj f16 | ~1.1 GB |
| KV cache, 8192 ctx, q8_0 K/V | ~0.45 GB |
| Compute buffers / image encode | ~0.4 GB |

That fits **only** with `--ctx-size 8192 --cache-type-k q8_0 --cache-type-v q8_0
--flash-attn on`, and only with ≤ 3 views per request. So: add
`SATQUERY_VLM_MAX_VIEWS` (default 6, set 3 on the laptop) to
`src/satquery/core/config.py` and honour it in `vlm_runtime.py` view selection.
If it still does not fit, the fallback is `-ngl 28` (offload ~24 of 36 layers
and let the rest run on CPU) — slower, but demo-safe. Document both in
`scripts/serve_vlm.ps1`.

**2.4 Windows packaging:**

- `scripts/serve_vlm.ps1` — PowerShell twin of `serve_vlm.sh`, same env-var
  names, uses the official `llama-<build>-bin-win-cuda-12.4-x64.zip` +
  `cudart-llama-bin-win-cuda-12.4-x64.zip` (no CUDA toolkit install needed).
- `scripts/demo_laptop/README.md` — the 10-line runbook: unzip llama.cpp, put
  the two GGUFs in `models/`, run `serve_vlm.ps1`, then `uv run --no-sync
  uvicorn satquery.api.app:app` with a `.env` that says `SATQUERY_VLM_BACKEND=llamacpp`.
  The Python side needs no `--extra vlm`; `uv sync --group dev` on Windows is
  fine because that machine has no hand-installed torch to lose.
- Bundle: `sq-demo-<date>.7z` = the two GGUFs + `frontend/dist/` + the
  segmentation and CD checkpoints from `data/checkpoints/` + sample scenes.
  ~7 GB; copy on a USB stick, checksum in `SHA256SUMS`.

### Files to touch

- `scripts/merge_export.py` — add `--quant`, `--mmproj-outtype`, `--max-views`
  note in the manifest; write `SHA256SUMS` next to the outputs.
- `src/satquery/core/config.py`, `src/satquery/tools/vlm_runtime.py` —
  `SATQUERY_VLM_MAX_VIEWS`.
- `scripts/serve_vlm.sh` — add `--cache-type-k/v`, `--flash-attn` flags via env.
- `scripts/serve_vlm.ps1`, `scripts/demo_laptop/README.md` — new.
- `.gitignore` already ignores `*.gguf`, `/models/`; nothing to change.

### Verification

1. `scripts/merge_export.py --dry-run` prints three commands and exits 0.
2. Merged bf16 model through the HF backend answers the 5 probe queries in
   `scripts/test_inference.py` **identically** to adapter-over-base (same
   greedy decode → same tokens). Any drift means the merge touched quantised
   weights.
3. Run Track 1's `make eval` with `SATQUERY_VLM_BACKEND=llamacpp` against the
   Q4_K_M server: accept ≤ 2 pt drop in VQA accuracy and ≤ 3 pt drop in
   grounding recall vs. bf16. Put both columns on the slide.
4. On the laptop, with Wi-Fi off: `serve_vlm.ps1` reaches "server listening"
   in < 30 s, `nvidia-smi` shows < 7.4 GB used with one 3-view request in
   flight, and the frontend's HealthStrip reports the VLM as `ready`.
5. `curl http://127.0.0.1:8080/v1/models` and the backend's `/v1/health`
   both answer with the network adapter disabled.

---

## Track 3 — End-to-End API & DAG Parity Check

### Goal

The FastAPI service serves **the new adapter in bf16** by default, and one
scripted session proves the whole DAG — parser → 11 compatibility checks →
policy table → tool executor → CitationValidator → AuditTrace — over the three
demo families, streamed to the browser.

> Note on endpoint names: the streaming route is **not** `/api/query`. The
> API prefix is `settings.api_prefix = "/v1"`, and the routes are
> `POST /v1/analyze` (synchronous multipart) and `POST /v1/jobs` (202) +
> `GET /v1/jobs/{id}/events` (SSE), see `src/satquery/api/routers/{analyze,jobs}.py`
> and `DOCS/API_CONTRACT.md` (frozen 1.0, additive changes only).

### 3.1 Configuration

`.env` is git-ignored and read by pydantic-settings with the `SATQUERY_` prefix.
Today it points at llama.cpp. Change to:

```dotenv
SATQUERY_VLM_BACKEND=hf
SATQUERY_VLM_ADAPTER_PATH=runs/sq-lora-v2-full/adapter
SATQUERY_SEG_CHECKPOINT=data/checkpoints/seg/segformer-b5-loveda
SATQUERY_CD_CHECKPOINT=data/checkpoints/cd/<current siamese checkpoint>
```

The HF backend applies the adapter over bf16 base weights at load (~18.7 GiB
peak with six views; NF4 generation is broken on gfx1100, see
`src/satquery/models/hf_backend.py`). **`llama-server` must not be running at
the same time** on the 24 GB box — it holds ~9 GB and the bf16 load is refused.

A root `.env.example` with these lines (and the llama.cpp alternative
commented out) is committed alongside this plan; `cp .env.example .env` is the
documented first step in the README.

### 3.2 Parity script

`scripts/e2e_parity.py` — runs against a live server, no pytest, prints a
checklist. Three scenarios, each from a committed fixture under
`tests/fixtures/e2e/` (small GeoTIFFs, < 2 MB each, made by
`scripts/make_synthetic_fixtures.py` or clipped from real CDVQA / BigEarthNet
scenes with `rasterio` — real is better for the demo):

| Scenario | Inputs | Expected plan (from `configs/policy_table.yaml`) | Assertions |
|---|---|---|---|
| **Single-image grounding** | one VRSBench-style VHR PNG, "Where is the airplane?" | `vlm_grounding` (+ `object_counter` if the query counts) | answer has ≥ 1 box in `[0,1000]`, `uncited_numeric_spans == []`, trace schema-valid |
| **Bi-temporal change (CDVQA pair)** | two co-registered GeoTIFFs, "What changed between these two images?" | `change_detect` → `change_statistics` → `vlm_change_vqa` | 11 checks pass (CRS, GSD, overlap, co-registration …); `changed_area_pct` cited as `step:2/scalars.changed_area_pct`; answer text mentions the cited value verbatim |
| **Cross-modal optical + SAR** | S2 12-band + S1 GRD, "Is this area built-up?" | `spectral_index_analyzer` ∥ `sar_backscatter_analyzer` → `physics_agreement` → `vlm_vqa` | NDBI and σ⁰_VV both cited; `physics_agreement.verdict` present; DAG shows the two analysers ran concurrently (overlapping `started_at`/`finished_at`) |

Plus two negative cases that must **degrade, not 500**: a pair with mismatched
CRS (check fails → `DEGRADED` status with the named check in
`compatibility.failed`), and a query whose numbers the model invents (inject
via `options.debug_force_uncited=true` → `uncited_numeric_spans` non-empty and
the frontend renders the honesty flag).

For every scenario, run it **twice**: once via `POST /v1/analyze`, once via
`POST /v1/jobs` + SSE. Assert the SSE event sequence is
`queued → planned → step:N started/finished (per tool) → validated → done`
and that the final `AuditTrace` from the SSE path is byte-identical (after
dropping timestamps and `trace_id`) to the synchronous one — the "byte-identical
reruns" claim from PRODUCT.md, tested rather than asserted.

### 3.3 Things the run will probably surface

- The `Answer.generator` string must read `...+adapter:sq-lora-v2-full`; it is
  built from the adapter directory name — check `vlm_runtime.py`.
- Registry `checkpoint_available()` only counts explicitly configured
  checkpoints; if `SATQUERY_CD_CHECKPOINT` is unset the change scenario will
  silently plan `image_diff_change` instead of the Siamese model. The parity
  script should assert on the tool *name* in the plan, not just on success.
- `max_pixels`/view budget parity: the trained contract is 448 px views,
  `max_pixels=147456`; `scripts/preflight_train_serve_parity.py` is the gate —
  run it once with the new adapter before anything else.

### Files to touch

- `.env` (local); `.env.example` is already committed.
- `scripts/e2e_parity.py` — new. `--base-url`, `--scenario`, `--json out`.
- `tests/fixtures/e2e/` — new fixtures + `README.md` with provenance/licence.
- `tests/contract/test_jobs_sse.py` — extend with the event-sequence assertion
  against the recorded fixture so CI keeps it honest without a GPU.
- `Makefile` — `make e2e` (starts API with the `.env`, runs the script, stops it).
- `DOCS/AI_HANDOFF/08_OPEN_ISSUES_AND_NEXT_STEPS.md` — close the "adapter not
  wired to serving" item.

### Verification

1. `scripts/preflight_train_serve_parity.py` passes with
   `SATQUERY_VLM_ADAPTER_PATH=runs/sq-lora-v2-full/adapter`.
2. `make e2e` prints 5/5 scenarios green on the 24 GB box, total wall time
   < 3 min after model load.
3. `make contract` still green — no wire-contract change was needed (if one
   was, it is additive and `openapi.json` + `schema.d.ts` were regenerated in
   the same commit).
4. In the browser at `:5173`: submit the CDVQA pair, watch the DAG panel light
   up step by step, click the `changed_area_pct` citation in the answer and
   land on the `change_statistics` measurement card.

---

## Track 4 — High-Impact UI Features

All three live in `frontend/`. Mode for these surfaces is **Operate**
(PRODUCT.md): scanability and native expectations first, brand in the details.
Each feature must work with `?mock=1` (recorded fixtures, no backend) because
that is the fallback if the demo GPU dies on stage.

### 4.1 Tactical SITREP Generator (client-side 1-page PDF)

**Today:** `frontend/src/pages/Report.tsx` renders `/report/<trace>` and hands
the browser's *Save as PDF* the job. That is a multi-page audit record, not a
one-page brief, and it depends on the judge's print dialog.

**Target:** a "SITREP" button in the thread header and in `saved/ExportMenu.tsx`
that produces `SITREP-<scene>-<yyyymmdd-hhmm>.pdf` in one click:

```
┌──────────────────────────────────────────────────────────┐
│ SATQUERY AI · SITREP            <scene id> · <timestamp>  │
│ Query: "…"                                 Status: OK     │
├───────────────────────────┬──────────────────────────────┤
│  scene view (T1 | T2 or   │  ANSWER (citation tags as    │
│  optical | SAR) with      │  superscripts)               │
│  boxes burned in, north   │                              │
│  arrow, scale bar, bounds │  KEY MEASUREMENTS            │
│                           │  changed_area_pct   12.4 %   │
│                           │  ndbi_mean          0.31     │
├───────────────────────────┴──────────────────────────────┤
│ TOOL CHAIN  parse → checks 11/11 → change_detect@1.2 →   │
│             change_statistics@1.0 → vlm_change_vqa@1.0    │
│ CITATIONS   3 bound · 0 uncited        trace <id> · v1.0  │
└──────────────────────────────────────────────────────────┘
```

**Architecture:**

- `frontend/src/export/sitrep/compose.ts` — pure function
  `composeSitrep(trace: AuditTrace, run: SavedRun, images: ImageBitmap[]) -> SitrepModel`.
  No DOM; unit-testable. Picks the KPI rows from `kpi/registry.ts` (same
  selection the KpiCards use, so the PDF never shows a number the UI did not).
- `frontend/src/export/sitrep/render.ts` — draws the scene panel on an
  offscreen `<canvas>` (reuse `pages/saved/capture.ts` and `thread/bbox.ts`
  for the box overlay; `evidence/georef.ts` for the bounds strip) and lays the
  page out with **pdf-lib** (pure TS, no workers, ~250 KB, works offline;
  jsPDF's font handling is the reason not to pick it). Embed Geist from
  `public/fonts/` so the PDF matches the console.
- `frontend/src/export/sitrep/download.ts` — `Blob` → `<a download>`.
- Entry points: `components/thread/ThreadPanel.tsx` header button,
  `pages/saved/ExportMenu.tsx` item, keyboard `⌘⇧S` in `shell/shortcuts.ts`.
- Bundle boundary: pdf-lib loads through `import()` behind the button so the
  landing and console bundles do not grow — `src/__tests__/bundle-boundaries.test.ts`
  already enforces boundaries; add the rule.

**Verification:** vitest on `compose.ts` with the recorded CDVQA fixture (3
citations → 3 rows, uncited count 0; a fixture with `uncited_numeric_spans`
→ the PDF carries a visible "1 UNCITED" badge). Playwright: click SITREP in
`?mock=1`, assert a download of `application/pdf` > 20 KB, and that
`pdf-lib` parses it back to exactly 1 page. Manual: print it on A4 and
letter — nothing clipped.

### 4.2 QGIS / ArcGIS — "Export .GeoJSON"

**Today:** `frontend/src/export/geojson.ts` exists and is correct: boxes map
through the manifest's WGS84 bounds when present, otherwise the collection is
emitted in pixel space **and says so** — never fake lon/lat. Tests in
`export/__tests__/geojson.test.ts`. It is wired into `saved/ExportMenu.tsx`
but not into the live thread, and the properties are thin.

**Remaining work:**

- RFC 7946 strictness: no `crs` member (7946 forbids it — remove the optional
  field; put `"crs": "pixel"` in a top-level `properties`-free `note` as now),
  right-hand-rule winding (exterior ring counter-clockwise), closed rings,
  `bbox` member on the collection, coordinates rounded to 7 decimals.
- Per-feature `properties`: `label`, `confidence`, `source_step`,
  `tool`, `tool_version`, `citation` (`step:n/scalars.path`), `trace_id`,
  `scene_id`, `acquired_at`, `sensor`. QGIS's attribute table becomes the
  audit trace.
- Multi-view runs: one feature per box per view, with `view_id` — a T1/T2 pair
  produces two layers a user can toggle.
- Also emit segmentation masks when the run has them (`semantic_segmenter`
  outputs a class raster): vectorise on the client with `marching-squares`
  → `MultiPolygon` per class, simplified with a 0.5 px tolerance. Optional,
  behind `Include masks` in the export dialog.
- Entry points: button in `components/stage/EvidenceTray.tsx` next to the
  boxes, `ExportMenu.tsx`, keyboard `⌘⇧G`.
- Copy: the button reads "Export GeoJSON" when bounds exist and
  "Export GeoJSON (pixel space)" when they do not — a judge must not be able
  to load a pixel-space file into QGIS and think it is georeferenced.

**Verification:** extend `geojson.test.ts` with a winding-order test and a
`bbox` test; validate a sample against the `geojson` npm validator in the
test. Manual: `Layer → Add Layer → Add Vector Layer` in QGIS 3.34 on the
Windows laptop, boxes land on the Sentinel-2 tile; open the attribute table
and see the citation column. Same file in ArcGIS Pro via *Add Data*.

### 4.3 Interactive Map / STAC Imagery Fetcher

**Today:** `frontend/src/pages/Maps.tsx` + `pages/maps/MapStage.tsx` is a
MapLibre stage that places the current run's views by their WGS84 bounds, with
an optional online satellite basemap that auto-disables when tiles cannot be
reached (`state/map.ts`). The landing page has a three.js globe
(`pages/landing/globe/`). No search, no STAC.

**Target:** from the Maps page, type a place, pick a date window, choose
Sentinel-2 (optical) / Sentinel-1 (SAR) / both, see the matching scenes as
footprints, pick one or a T1/T2 pair, and **send it to the console as a new
query** — imagery fetched at the resolution the pipeline wants, not full COGs.

**Decision: stay on MapLibre.** Cesium is 3 MB gzip, needs an ion token for
the default assets, and gives us nothing the demo needs; the landing globe
already covers the "wow" moment. The Maps page gets a `Globe` projection
toggle (MapLibre ≥ 4.6 supports it natively).

**Architecture:**

```
frontend/src/geo/nominatim.ts        search(q) -> Place[]   (OSM Nominatim, 1 req/s, UA header, debounce 400 ms)
frontend/src/geo/stac.ts             searchItems({bbox, datetime, collections, cloud}) -> StacItem[]
frontend/src/geo/planetary.ts        signed-URL wrapper: GET https://planetarycomputer.microsoft.com/api/sas/v1/token/<collection>
frontend/src/geo/__tests__/          fixtures recorded from the real API, replayed in vitest
frontend/src/pages/maps/hud/PlaceSearch.tsx      search box + result list
frontend/src/pages/maps/hud/TimeSlider.tsx       date window; two handles for a T1/T2 pair
frontend/src/pages/maps/hud/SceneShelf.tsx       thumbnails of matches, cloud %, orbit direction (S1), "Use as T1 / T2"
frontend/src/state/stac.ts                       query, results, selection, loading/error
src/satquery/api/routers/imagery.py              POST /v1/imagery/fetch  (backend does the raster work)
src/satquery/ingest/stac_fetch.py                COG window read via rasterio, clip to bbox, write GeoTIFF into the ArtifactStore
```

**Data sources (Microsoft Planetary Computer, no key required for search;
SAS token for asset reads):**

- `sentinel-2-l2a` — optical; filter `eo:cloud_cover < 20`; assets B02/B03/B04/B08/B11/B12 (what `spectral_index_analyzer` needs) at 10–20 m.
- `sentinel-1-grd` — SAR; assets `vv`, `vh`; record `sat:orbit_state` so a T1/T2 pair is same-orbit (co-registration check will fail otherwise, which is correct but confusing on stage).
- `sentinel-1-rtc` — preferred over GRD when available: terrain-corrected, matches what BigEarthNet-v2 S1 looks like.

**Why the fetch is server-side:** a Sentinel-2 tile is ~1 GB; the pipeline
wants a ≤ 2048 px window. `rasterio` reads a COG window over HTTP in one
range request; the browser cannot. The router returns an `upload_id` that
`POST /v1/analyze` / `/v1/jobs` already accept, so nothing downstream
changes. This is an **additive** contract change: new router, new schema
in `src/satquery/schemas/api.py`, regenerate `openapi.json` + `schema.d.ts`.

**Offline behaviour (non-negotiable for the demo):** every network call is
behind `state/settings.ts → online features` (off by default, remembered),
fails to a toast within 5 s, and the Maps page still shows the current run's
scene. Pre-record 3 place searches + STAC results as fixtures under
`public/samples/stac/` so `?mock=1` shows the whole flow without internet.

**Verification:** vitest replays recorded Nominatim/STAC fixtures
(`geo/__tests__/`); backend unit test reads a 256 px window from a local
synthetic COG (`tests/unit/test_stac_fetch.py`, no network). Playwright in
`?mock=1`: search "Ahmedabad", pick two S1 scenes 6 months apart, click
"Analyse change", land in the console with two views loaded and the
compatibility panel showing 11/11. Manual, online: the same with the real
API; the fetched GeoTIFFs open in QGIS with correct CRS.

---

## Cross-track checklist for the presentation week

- [ ] `make ci` green on `main` after each track lands.
- [ ] Slide numbers come only from `runs/eval/<name>/results.md` (Track 1),
      both bf16 and Q4_K_M columns (Track 2).
- [ ] Demo laptop rehearsed **with Wi-Fi off** twice, including the `?mock=1`
      fallback (Track 2 + 4).
- [ ] `DOCS/AI_HANDOFF/00_START_HERE.md` and `README.md` say which adapter is
      current (`sq-lora-v2-full`) and how it is served.
- [ ] One committed `SHA256SUMS` for the bundle on the USB stick.
- [ ] Nothing on any branch but `main`.
