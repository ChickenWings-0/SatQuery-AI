# SatQuery AI — Independent Codebase Audit

**Role:** Principal AI Architect / Lead Code Reviewer
**Date:** 2026-09-11 · **HEAD:** `77bea25` (Phase 8 commit) · **Branch:** `main` (clean)
**Method:** Every rating below is backed by something I ran or read, not by the docs. Commands executed: full backend `pytest` (503 tests), `ruff`, `mypy --strict`, frontend `tsc -b`, `vitest` (194 tests), `vite build`, `openapi.json` regeneration diff, `schema.d.ts` regeneration diff, corpus JSONL statistics, and a parse of the 19-hour `full_training_run.log`.

> **Note on location.** The brief asked for `docs/project_audit.md`. The repository has no lowercase `docs/`; its documentation folder is `DOCS/`. This file lives there. No other file was touched.

---

## 0. Scorecard

| Section | Score | One-line verdict |
|---|:---:|---|
| Data Pipeline | **7 / 10** | Architecturally excellent, operationally under-delivered: the corpus that was actually trained on is 18.5k BigEarthNet-only samples with **zero image hashes**, so the dedup never ran on it. |
| ML Training | **4 / 10** | Infrastructure is first-rate; the *result* is not. Loss is computed over the entire sequence (no assistant-only masking), train loss sat at ~3.8 from 8% of the epoch to the end, and there is no zero-shot-vs-adapted ablation. The headline "domain-adapted VLM" claim is currently unsupported. |
| Inference Backend | **8 / 10** | The strongest part of the repo. Deterministic planner, DAG executor, honest degradation, a citation validator that is genuinely hard to fool, and a bf16 serving path with a VRAM guard that reconciles against reality. Real gaps: value-only citation matching, in-memory jobs, no Docker. |
| Frontend UI/UX | **8 / 10** | Idiomatic Tailwind v4, contract-typed end to end, contrast-tested tokens, letterbox-correct bbox overlay, self-hosted fonts with metric fallbacks. Debt: a dropped SSE stream re-runs the GPU job instead of reattaching, cancel is client-only, box parsing is duplicated in TypeScript. |
| Testing | **8 / 10** | 697 tests, all green, hermetic, contract-locked. But no CI, and the one seam that actually failed in production (script-level hash binding) has no test. |

**Overall: 7 / 10.** This is a serious, well-engineered agentic system with an unusually mature evidence/citation story. It is being let down by its ML result, which is the one thing the SIH rubric's "domain adaptation" requirement scores. Fix the training loss masking and run one honest ablation before anything else.

---

## 1. Data Pipeline — 7 / 10

**Scope read:** `src/satquery/training/corpus_builder.py` (2,164 lines), `local_sources.py`, `scripts/build_corpus.py` (797), `scripts/fetch_sources.py`, `scripts/render_views.py`, `scripts/render_vhr_views.py`, `training/data/builders/evidence_qa.py`, plus the JSONL outputs under `data/processed/corpus/`.

### Massive wins
- **One label writer, one box serialiser, one number formatter.** The corpus imports `label_for_view`, `build_system_prompt`, `box_format.serialise` and `citation_validator.format_number` from the *same modules the server uses* (`corpus_builder.py:41-58`). Train/serve dialect drift is structurally impossible for labels and boxes. This is the single best design decision in the data layer.
- **Citation audit at build time.** `assemble()` runs the CitationValidator over every training answer and raises `CitationError` if a number does not resolve (`corpus_builder.py:104-110`). A template bug fails the build, not the eval.
- **Dedup is real engineering, not a `set()`.** SHA-256 exact → DCT pHash near-dup (Hamming ≤ 5) → quarantine assertion, with 8-band bucketing so lookups do not scan 65k hashes (`corpus_builder.py:1560-1800`). The `image_key` concept (`check()` docstring, line 1736) correctly separates "many annotations per image" from "same image twice" — the exact failure that collapsed CDVQA from 9,000 to 215 samples in an earlier run. `LeakageError` is fatal by design. The math in the band-bucketing comment is correct (5 differing bits touch ≤5 of 8 bands, so ≥3 match).
- **Streaming build with a reservoir.** `build_corpus_streaming` and `iter_source_samples` fixed a documented OOM on the 9.6M-row BigEarthNet.txt parquet. Peak memory is one corpus, not one dataset.
- **Honest source resolution.** `LOCAL_SOURCES` / `UNRESOLVED_SOURCES` in `build_corpus.py` document *why* each HF loader fails (VRSBench's zip-sweep crash 20,264 rows in; RSVQA-HR being off-Hub; CDVQA being WebDataset). This is the kind of hard-won knowledge that usually lives in someone's head.

### Brutal truths
1. **The trained corpus is not the 65k corpus.** Measured from disk:

   | File | Lines | Sources | `image_sha256` present |
   |---|---:|---|---:|
   | `train.jsonl` | 15,530 | bigearthnet_v2 only | **0** |
   | `val.jsonl` | 2,470 | bigearthnet_v2 only | **0** |
   | `full.train.jsonl` (what `full-epoch-v1` trained on) | 18,530 | bigearthnet_v2 + evidence_qa | **0** |
   | `evidence_qa.val.jsonl` | **0** | — | — |

   VRSBench (29,615 rendered tiles on disk), CDVQA (271 pairs) and RSVQA-HR (3 GB raw) never entered the corpus. The overnight log's own closing note says so: *"RSVQA-HR and CDVQA sources — 29% of the §5 composition, still unresolved."* Track B (VHR / Cartosat alignment) — the whole hidden-evaluation-gap strategy from `Master.md §1` — is absent from the trained adapter.

2. **Dedup was silently bypassed for the only source that was trained on.** `bind_view_paths()` deliberately does not set `image_path` for BigEarthNet (`build_corpus.py:363-364`), but `hashes_for()` only hashes when `image_path` or `pre_path` exists (`build_corpus.py:571`). Result: every BEN row gets `{}`, `deduplicate()` passes unhashed samples through by design ("dropping unhashed samples silently would shrink the corpus"), and the "strict image-level dedup" ran on nothing. The library is correct; the script-level seam is broken, and there is no test on `hashes_for` (grep: 0 hits in `tests/`). Practical impact is low for BEN (patch ids are unique), but the *claim* in the brief is not currently true of the artifact.

3. **`evidence_qa` has no validation split** (0 lines). Its training answers are the model's citation-behaviour supervision and nothing measures it on held-out data.

4. **DIOR-RSVG is unresolved** (gated), so referring-expression grounding at VHR — the primary tool for mandatory requirement 2's grounding option — has no training signal beyond BigEarthNet.txt boxes at 10 m.

5. **Root `README.md` is 0 bytes.** A judge or teammate cloning the repo gets nothing.

### Smaller debt
- `COMPOSITION` totals 65,000 but the throughput probe measured 0.265 samples/s → ~68 h per epoch. The plan's "one overnight run" assumption is off by ~10×; the composition should be re-budgeted, not just noted in a log.
- `data/processed/corpus/` contains five overlapping corpora (`train`, `full.train`, `probe.train`, `poc/*`) with no manifest saying which was used for which run. `run_manifest.json` records it per run, which is good, but the directory itself is ambiguous.

---

## 2. ML Training — 4 / 10

**Scope read:** `src/satquery/training/vlm/qlora.py` (1,278), `scripts/train_vlm.py`, `configs/train/*.yaml`, `src/satquery/training/cd/*`, `scripts/train_cd.py`, `runs/full-epoch-v1/run_manifest.json`, `runs/full-epoch-v1/inference_probe*.json`, `full_training_run.log`, `train_levircd.log`, `data/checkpoints/cd/levircd_resnet18.ckpt.json`.

### Massive wins
- **The training harness is production-grade.** Pydantic-validated profiles, `resolve_target_modules` that finds the last-N ViT blocks by name rather than hard-coded indices, a VRAM estimator whose prediction (17.13 GiB) landed within 0.4 GiB of the measured peak (16.77), `guard_budget` before any weights load, resumable checkpoints with optimiser state (`save_only_model: False`, with a comment explaining exactly why), SIGINT/SIGTERM graceful stop at the next step boundary, and a `run_manifest.json` that records everything down to the resolved module names. I rarely see this level of care.
- **`sample_to_chat()` documents trl's foot-guns precisely** (`qlora.py:626-671`): the `images` key requirement and the bare-placeholder counting rule are both real and both undocumented upstream.
- **Change detection is honest.** `levircd_resnet18.ckpt.json` records F1 0.858 against the 0.88 gate and says in plain text: *"no result table may quote it as meeting the gate."* The checkpoint also records a normalisation bug that was found and corrected (identity stats were applied; measured stats were never used). This is exactly the right culture.

### Brutal truths
1. **The loss is computed over the whole sequence.** `sft_config_kwargs()` (`qlora.py:917-975`) sets neither `assistant_only_loss` nor `completion_only_loss`; grep across `src/ scripts/ configs/` finds no occurrence. Each sample's system turn is ~1,000+ tokens of rules plus a FactSheet of up to 30 floats like `0.751899`. Those digits are unpredictable, dominate the token count, and are being *trained on*. The observable consequence:

   | epoch | train loss | grad_norm | mean_token_accuracy |
   |---:|---:|---:|---:|
   | 0.008 | 13.01 | 29.9 | 0.26 |
   | 0.078 | 3.82 | 0.08 | 0.50 |
   | 0.49 | 3.82 | 0.06 | 0.50 |
   | 0.99 | 3.76 | 0.05 | 0.51 |

   Eval loss: 3.935 → 3.930 across the entire epoch. Token accuracy 0.487 → 0.488. **The 19-hour run is flat from step ~90 onward.** Either the adapter learned almost nothing, or whatever it learned is invisible under a loss dominated by fact-sheet digits. Either way the run cannot be used as evidence of adaptation. This is the single highest-priority defect in the repository.

2. **Train/serve prompt skew, in the direction the docs promise cannot happen.** Training (`qlora.py:654-657`) emits image blocks as `{"type": "image", "label": ...}` — the `label` key is not a text part; the chat template ignores it, so the model saw *unlabelled* images with names only in the system prompt. Serving (`hf_backend.py:107-118`) inserts each label as a **text part immediately before its image**, with a docstring explaining that naming images only in the system prompt "silently breaks". The adapter was therefore trained on the layout the serving code says is unreliable, and served on one it never saw.

3. **NF4 training, bf16 serving.** Documented and justified (`hf_backend.py:17-26`: bitsandbytes 4-bit *generation* produces token soup on gfx1100). Fine as a workaround — but a LoRA fitted against NF4-dequantised activations is applied to different base activations in bf16. Nobody has measured what that costs, and with a flat loss curve it may not matter, but it must be in the ablation.

4. **No ablation exists.** `Master.md` Phase 7 requires "zero-shot micro-F1 vs adapted micro-F1" and Phase 8 requires a four-row table. There is no `src/satquery/eval/` module (`ls` → missing), no `make eval`, and `scripts/eval_vrsbench_zeroshot.py` defaults to a five-item synthetic mock. `inference_probe_discrete.json` is a single-patch probe with IoU 0.66 — a smoke test, not a result. The judged claim ("we domain-adapted a VLM") is currently a claim.

5. **One epoch, one run, no seed variance.** Acceptable for a hackathon timeline; not acceptable to present as a converged result.

6. **CD is below gate and single-resolution.** Phase 5 asks for LEVIR-CD (0.5 m) *and* OSCD (10 m) to record cross-resolution degradation. OSCD is downloaded (`data/raw/Onera…`) but the only checkpoint is `levircd_resnet18` at 0.5 m.

### What "good" looks like from here
Set `assistant_only_loss=True` (trl ≥ 0.20 supports it with the `{% generation %}` chat-template markers; Qwen3-VL's template needs the markers added) or move to a prompt-completion dataset shape; re-run the 200-step throughput probe and confirm assistant-token accuracy climbs well past 0.5; align `sample_to_chat` with `build_messages`; *then* run the epoch. Expect the same 19 hours to mean something this time.

---

## 3. Inference Backend — 8 / 10

**Scope read:** `src/satquery/api/*`, `agent/*` (planner 458, executor 970, aggregator 366, task_classifier 467), `evidence/*`, `models/*` (loader 787, hf_backend 339, llamacpp_client, prompts/), `tools/*` (15 tools), `registry/*`, `ingest/*`, `render/*`, `trace/*`, `configs/policy_table.yaml`, `configs/registry.yaml`.

### Massive wins
- **The trace is a product, not a log.** `AuditTrace` is built by the executor, schema-versioned, persisted to SQLite, and returned verbatim. `openapi.json` regenerates byte-identical from `create_app()` (verified), and `frontend/src/api/schema.d.ts` regenerates byte-identical from it (verified). The contract chain is intact end to end.
- **Deterministic control flow.** `policy_table.yaml` (`TaskType|PairType|Modality` → DAG, three-tier fallback to a flagged generic entry) means no LLM ever chooses a tool. The pipeline docstring commits to identical traces for identical inputs and the `greedy → torch.manual_seed(request.seed)` line in `hf_backend.py:264` follows through.
- **Executor is correctly async.** Tools run under `asyncio.to_thread` with `wait_for` timeouts and separate CPU/GPU semaphores (`executor.py:273-274, 715-716`); a failed tool degrades to its declared fallback and is recorded `DEGRADED`, never a 500 (only the renderer is `optional: false`). `LazyBackend` serialises VLM generation behind an `RLock` because transformers generation is not reentrant.
- **The VRAM guard reconciles.** `HuggingFaceBackend._load` estimates before loading and then re-checks *after* (`hf_backend.py:172-177`) — "a guard that is never reconciled against reality is decoration." The adapter path hard-fails if `adapter_config.json` is missing rather than silently serving stock Qwen under the fine-tuned name.
- **Box sentinels survive decoding.** `_strip_chat_tokens` removes chat scaffolding by name instead of `skip_special_tokens=True`, which would delete `<|box_start|>` and reduce grounding to a bare-digit heuristic (`hf_backend.py:283-292`). Correct, subtle, and documented.
- **CitationValidator is genuinely hard to fool.** Unit-aware matching by `_`-delimited name segment (so `7.4 %` cannot cite `changed_area_km2`, and a unitless `0.03` cannot cite `changed_area_pct`), tolerance pivot at 10, deterministic tie-break, year/image-index/list-marker exclusions, and a `flag`-not-`strip` default with the reasoning written down. `vlm_runtime.synthesise` runs it on the tool's own output and drops the step to `DEGRADED` on any uncited span.
- **Jobs API is thoughtfully done.** Ingestion runs synchronously *before* the 202 so a bad upload is a 4xx, not an error event; the stage sequence is buffered and replayed to late subscribers; SSE has a 15 s heartbeat and cancels its pending `anext` on disconnect.

### Brutal truths
1. **Citation resolution is by value, not by key.** `validate()` resolves each numeric span to *any* fact within tolerance (`citation_validator.py:191-206`), and `strip_citation_markers()` only checks that a `[key]` *exists* in the sheet. An answer reading `0.75 [sar_backscatter_analyzer.sigma0_vv_db_mean]` — right number, wrong key — passes both checks and lands in the trace as grounded. The `[key]` the model was trained to emit is the stronger signal and it is being thrown away.
2. **`ingest()` blocks the event loop.** Both `/v1/analyze` and `/v1/jobs` call synchronous rasterio + phase-correlation ingestion inside `async def` handlers (`analyze.py:53`, `jobs.py:157`). One large GeoTIFF pair stalls every SSE heartbeat in the process.
3. **GPU concurrency is per-request, not per-process.** The executor's `_gpu` semaphore is created per `DagExecutor` (`pipeline.py:214`), i.e. per request. The VLM is protected by the backend lock, but two concurrent jobs can both load SegFormer-B5, the CD model and DOFA on top of an 18.7 GiB VLM. Phase 9's "10 concurrent requests without VRAM exhaustion" is not met.
4. **`JobStore` is in-memory** (`api/jobs.py`) with oldest-first eviction. A restart drops every live job (traces survive in SQLite, which is the right split, but the client has no way to learn the job died). There is no `DELETE /v1/jobs/{id}`; "cancel" in the UI is purely client-side and the GPU keeps working.
5. **Phase 9 is unstarted.** No `Dockerfile`, no `docker-compose.yml`, no `Makefile`, no `.github/`. The "kill the network at judging" story rests on the llama.cpp path, and `runs/poc-v1/export-Q4_K_M.gguf` was exported from the *PoC* adapter, not `full-epoch-v1`.
6. **Hygiene:** `mypy --strict` has 1 error (`local_sources.py:132`, `no-any-return`); `ruff` has 1 (import sort in `scripts/test_inference.py`); `src/satquery_ai/__init__.py` is a leftover "Hello from satquery-ai!" scaffold; `scripts/export_openapi.py` treats *any* argv as an output path (I discovered this by running `--help`, which wrote a file named `--help`; removed).

### Smaller observations
- `text_grounding.py` converts boxes through the padded-canvas `ViewGeometry` affine rather than the raster's, which is correct and the kind of detail that is usually wrong. Unreferenced PNGs get pixels and no GeoJSON rather than invented lon/lat — good.
- Every tool module imports torch lazily so the registry can build the capabilities panel on a CPU-only box. Good.

---

## 4. Frontend UI/UX — 8 / 10

**Scope read:** all of `frontend/src` (11.6k lines incl. the 1,475-line generated schema), `theme.css` (730), `vite.config.ts`, `package.json`, `index.html`. Ran `tsc -b` (clean), `vitest` (194/194), `vite build` (clean, 1.8 s).

### Massive wins
- **Tailwind v4 is done properly.** No `tailwind.config.*`, no `postcss.config.*`, no `@tailwind base` or `@apply` anywhere in `src/` (grep: 0 hits). Tokens live in `@theme`, shell breakpoints are two-dimensional `@custom-variant wide/desk` (width *and* height, so a landscape phone does not get a 240 px sidebar), component classes sit in `@layer components`, and the plugin is `@tailwindcss/vite`. This is the v4 idiom, not v3 with the version bumped.
- **The token file is a spec, not a palette.** Every hex carries its computed WCAG ratio against the ground it actually lands on *including its own 8/12/15/20% tint* — the chip case that naive checks miss — and `styles/__tests__/contrast.test.ts` pins them. Four rules are stated and enforceable: status hues map 1:1 onto `ToolStatus ∪ CheckStatus`; `--color-accent-cool` means "selected" and nothing else; `--color-evidence` is for citations only; `--color-warn` is the only hue allowed on an uncited span. That last one is the honesty signal made visual.
- **Fonts are self-hosted with metric-matched fallbacks** (`size-adjust` / `ascent-override` computed from the shipped woff2 `hhea` tables), so an offline judging machine gets the same face and the pre-swap paint does not jolt a content-sized column.
- **The bbox overlay is geometrically correct.** `viewBox="0 0 1000 1000"` + `preserveAspectRatio="none"` maps Qwen's normalised frame with zero arithmetic; `vector-effect: non-scaling-stroke` keeps strokes at 1.5 px under zoom and non-square stretch; labels are HTML so type never distorts and flip below the box at the top edge. Crucially `ImageViewer` computes the `object-fit: contain` rectangle from `naturalWidth/Height` and a `ResizeObserver` (`ImageViewer.tsx:25-83`) and sizes the overlay to *that*, not the cell — without which every box lands off-target on a wide window. The overlay sits inside `TransformComponent` so zoom/pan carry it.
- **State model is testable by construction.** `job.ts` is a pure `reduce(state, event)` over the SSE union, widened with client-only `PENDING/RUNNING`; `job.test.ts` drives it with recorded fixtures and asserts the terminal state equals the `done` payload.
- **Accessibility was thought about, not sprinkled.** Citation pills use `aria-describedby` (name stays the visible number, WCAG 2.5.3); the uncited underline gets an `sr-only` spoken equivalent; touch targets key on `pointer: coarse` not width; the WCAG 2.5.8 inline-target exemption is claimed explicitly via `data-inline-target`; `useReducedMotion` exists and is tested; one looping animation in the whole product.
- **Offline path is real and lazy.** `?mock=1` starts MSW from a dynamic import; the production bundle contains a separate 360 KB `browser-*.js` chunk that only loads on request. `vite.config.ts` proxies to `127.0.0.1` with the IPv6 `localhost` gotcha written down.

### Brutal truths
1. **A dropped SSE stream re-runs the job.** `useRun.submit` wraps `createJob` + `streamJob` in one try; any transport error after the 202 calls `markFailed()` and offers `retry`, which **re-submits the same files and query** — a fresh 20-60 s GPU run. `sse.ts:44` says "this stream never reconnects", yet `client.ts:235` already has `getJson<JobStatusResponse>('/v1/jobs/${jobId}')`. Reattaching (poll `/v1/jobs/{id}`, or re-open `/events` which replays the buffered stage sequence) is a few lines away and would turn a Wi-Fi blip at the judges' table from a re-run into a non-event.
2. **Cancel is a lie to the server.** `cancel()` aborts the fetch and marks failed; the backend keeps the GPU busy. Not the frontend's fault (no endpoint exists), but the button implies otherwise.
3. **Box parsing is duplicated across languages.** `thread/bbox.ts` (288 lines) is a hand-port of `box_format.py`, tested for parity. But the backend already emits a `BBOX_SET` artifact with pixel *and* WGS84 coordinates from `text_grounding`. The frontend re-parses raw answer text instead of consuming the structured artifact — so any parser drift shows up as "the map disagrees with the trace". Consume the artifact; keep the parser only as a fallback for text-only answers.
4. **`oxlint` is broken in this checkout** — the native binding is missing (`@oxlint/binding-linux-x64-gnu`), so `npm run lint` crashes. Environment/npm issue, not code, but it means lint has not been run on whatever was committed after it broke.
5. **The change legend is hard-coded** — `ImageViewer.tsx:361`: `New built-up area` on every change task regardless of `target_class`.
6. **`react-compare-slider` logs `CSS.registerProperty is not a function` under happy-dom** on every test file that mounts `ImageViewer`. Harmless, but noise hides signal; stub it in `vitest.config.ts`.
7. **Bundle:** 433 KB main + 280 KB DagCanvas (xyflow + dagre) + 360 KB MSW chunk. Fine for a LAN demo; DagCanvas is already code-split, which is the right call.

---

## 5. Testing — 8 / 10

**Measured:** backend 503 tests collected, 503 passed, ~4 min wall (the `test_models.py` case that exercises real SDPA on ROCm is the long pole); frontend 15 files / 194 tests passed in 4.6 s.

| Backend file | Tests | Frontend file | Tests |
|---|---:|---|---:|
| `test_render.py` | 105 | `job.test.ts` | reducer over recorded SSE |
| `test_vlm_training.py` | 103 | `contrast.test.ts` | pins every token pair |
| `test_crossmodal_grounding.py` | 78 | `bbox.test.ts` / `annotate.test.ts` | parser parity |
| `test_ingest.py` | 63 | `run-render`, `resilience`, `progressive-disclosure` | component |
| `test_change_detection.py` | 44 | `useHotkeys`, `useReducedMotion` | hooks |
| `test_models.py` | 41 | `stream.test.ts` | MSW end-to-end offline |
| `test_agent.py` | 30 | `fixtures.test.ts`, `registry.test.ts`, `format.test.ts` | |
| `contract/test_api.py` + `test_jobs_sse.py` | 39 | | |

### Massive wins
- **Hermetic and fast enough to run on every save.** Synthetic GeoTIFFs from `make_synthetic_fixtures.py`; no network; torch only where it is the thing under test.
- **Contract tests lock the frozen schema** and the SSE event union, on both sides of the wire.
- **The tests read like specifications.** `test_an_invented_number_survives_into_the_trace_as_a_flagged_span`, `test_quarantined_test_images_are_a_build_error`, `test_evidence_qa_never_narrates_a_view_it_did_not_attach`. Someone will be able to maintain this.
- **The analytic golden test** (`test_composite_matches_an_analytically_computed_golden_image`) checks the stretch against a closed-form reference rather than a committed PNG that rots.
- **Contrast is a test, not a promise.** Retuning a fill cannot silently take its text with it.

### Brutal truths
1. **No CI.** No `.github/workflows`, no pre-commit. 697 green tests are only green on the machine that remembers to run them. Add one workflow that runs `ruff`, `mypy`, `pytest -m "not gpu"`, `tsc -b`, `vitest run` and `vite build`; today's 1 ruff + 1 mypy error would have been caught.
2. **The seam that broke is untested.** `Deduplicator` has excellent tests (exact, near, quarantine, streaming leak refusal). `scripts/build_corpus.py::hashes_for` / `bind_view_paths` — the code that decides *whether the Deduplicator sees a hash at all* — has zero coverage, and that is exactly where the BigEarthNet corpus lost its hashes. A test that builds one BEN row through `_iter_source_samples` and asserts `meta.image_sha256 is not None` would have failed the build.
3. **No ML regression test.** Nothing asserts that a training step on a fixed micro-corpus *reduces assistant-token loss*. A single 10-step test on a tiny model (or even a linear head over the same collator) would have exposed the missing loss mask before the 19-hour run.
4. **No end-to-end test with the real adapter.** Understandable (GPU, 16 GB weights), but `scripts/test_inference.py` is a script, not a test, and its results are not archived beyond `inference_probe*.json`.
5. **Frontend has no true browser test.** Playwright is installed (`devDependencies`) but unused; there is no test of the actual bbox overlay geometry under a non-square image, which is the highest-risk visual code path.

---

## 6. Cross-cutting: plan vs reality

| Master.md phase | Status | Evidence |
|---|---|---|
| 0 Contract | ✅ Done, still in sync | openapi/schema.d.ts diff = 0 |
| 1 Ingest/compat | ✅ Done | 63 tests; 11 named checks in `compatibility.py` |
| 2 Render/artifacts | ✅ Done | 105 tests; content-addressed blob store |
| 3 Registry/planner/executor/trace | ✅ Done | 15 tools in `registry.yaml`; policy table with fallback tiers |
| 4 VLM serving | ✅ Done (bf16 HF + llama.cpp) | GGUF exported from PoC, not from `full-epoch-v1` |
| 5 Change detection | ⚠️ Partial | LEVIR-CD F1 0.858 < 0.88 gate; no OSCD run |
| 6 Cross-modal + grounding | ✅ Code done, 78 tests | DIOR-RSVG training data unresolved |
| 7 Corpus + QLoRA | ⚠️ **Infrastructure done, result not** | BEN-only 18.5k; flat loss; no ablation |
| 8 Benchmark eval | ❌ Not started | no `src/satquery/eval/`, no table |
| 9 Hardening/demo | ❌ Not started | no Docker/Makefile/CI; no `make demo` |
| Frontend (teammate track) | ✅ Ahead of plan | mission control, DAG modal, swipe, overlay, offline mock |

---

## 7. Prioritised remediation (highest leverage first)

1. **Fix the training objective** — assistant-only loss, align `sample_to_chat` with `build_messages`, prove on a 200-step probe that assistant-token accuracy moves. *Everything the rubric scores for "domain adaptation" depends on this.*
2. **Run one honest ablation** — base Qwen3-VL bf16 vs adapter, same prompt, same 200 VRSBench + BEN-val items, same validator. Put the numbers in `DOCS/` even if they are bad. A bad honest number beats a missing one at SIH.
3. **Bind hashes for every source** in `build_corpus.py` and add the test that would have caught it. Then rebuild the corpus with VRSBench + CDVQA included (they are already rendered) and re-budget `COMPOSITION` against the measured 0.265 samples/s.
4. **Key-aware citation check** — when a `[key]` is present, require the adjacent number to match *that* key's value; only fall back to value-search for bare numbers.
5. **Frontend reattach instead of re-run** on SSE drop; add `DELETE /v1/jobs/{id}` so cancel means cancel.
6. **Process-wide GPU semaphore** in the executor; `asyncio.to_thread(ingest, …)` in both routers.
7. **CI + Dockerfile + Makefile** — Phase 9's first day of work; it is also what makes points 1-6 stick.
8. Small: fill `README.md`; delete `src/satquery_ai`; fix the 2 lint/type errors; make `export_openapi.py` use argparse; re-export the GGUF from whichever adapter is actually served.

---

*Nothing outside this file was created or modified. A stray `--help` file produced while probing `scripts/export_openapi.py` was deleted; `git status` is clean apart from the pre-existing untracked `repo documentation/` folder.*
