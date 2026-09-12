# SatQuery AI — Remediation Plan

**Source:** `DOCS/project_audit.md` (2026-09-11, HEAD `77bea25`) · **Written against:** HEAD `2cb0348`
**Scope:** every non-ML finding in the audit, plus the additional defects found while re-reading the code for this plan. ML training / loss-masking / ablation work is deliberately excluded and tracked separately.
**Format:** each item names the files to touch, the logic of the fix, and the test that proves it. No code is written here.

Items marked **★ NEW** were not in the audit. They came from reading the code paths the audit's findings pass through and noting what else fails silently on those paths.

---

## Execution status (2026-09-12)

Every item below is implemented except where marked. Gates at completion: backend 553 tests (was 503), frontend 215 (was 194) + 3 Playwright, ruff / mypy --strict / oxlint --deny-warnings / tsc clean in both the ROCm venv and a torch-free CI-shaped venv, `make ci` green, both container images build (podman).

| Item | Status | Notes |
|---|---|---|
| 1.1 – 1.5, 1.7 | ✅ | `hashes_for` → `dedup_image_path` + `HashTally`; factsheet binding; split-stratified `evidence_qa`; split-aware reservoirs |
| 1.6 | ◐ code done | `--composition SOURCE=N`, `MANIFEST.md` writer, throughput arithmetic in the `COMPOSITION` docstring. **The corpus rebuild itself was not run** (hours of CPU, owner's call on the mix). |
| 1.8 | ✅ | mypy fixed; README written; DIOR-RSVG note |
| 2.1 – 2.9 | ✅ | `agent/concurrency.py`; leaked-permit rule; `to_thread` on ingest/uploads/trace write/health probe; `DELETE /v1/jobs/{id}`; key-aware validator; `INVALID_OPTIONS`; query bounds; finished-only eviction + trace fallback + shutdown; SSE `id:` / `Last-Event-ID` |
| 2.10 | ◐ | argparse exporter, scaffold removed, ruff fixed, byte-bounded cache. **GGUF re-export not done** (needs the post-training adapter). |
| extra | ✅ | Cold-start timeout now covers the whole wave (fixed a pre-existing order-dependent test failure); `/v1/health` device probe off the loop |
| 3.1 – 3.6 | ✅ | `thread/resume.ts` bounded reattach; cancel wired end to end; `thread/boxes.ts` consumes `BBOX_SET`; legend follows `target_class`; run outcomes in history |
| 4.1 – 4.8 | ✅ | oxlint (needed Node ≥ 22.12 — local Node upgraded to 22.23.2, `.nvmrc` + `engines` pinned); vitest `setupFiles` stub; CI (backend · frontend · e2e · docker jobs); Makefile; pre-commit; Dockerfiles + compose + nginx; Playwright overlay spec |

Deferred to the ML track: everything in §6.

## 0. How to read this plan

| Phase | Theme | Items | Blocking dependency |
|---|---|---|---|
| 1 | Data pipeline integrity | 1.1 – 1.8 | none |
| 2 | Inference & concurrency | 2.1 – 2.9 | none (2.4 before 3.2) |
| 3 | Frontend resilience | 3.1 – 3.6 | 2.4 (cancel endpoint), 2.9 (SSE `id:` field) |
| 4 | Tooling & CI | 4.1 – 4.8 | should land **first** in practice — it is what stops phases 1-3 regressing |

Suggested execution order: **4.1 → 4.2 → 4.3** (so lint/tests are green and enforced), then Phase 1, Phase 2, Phase 3 in numeric order, then the rest of Phase 4. Every item is small enough to be one commit; the phase boundaries are review boundaries, not release boundaries.

Conventions that apply to every item:

- The API schema is frozen at `1.0` (API_CONTRACT.md §0). Anything that changes `openapi.json` must be **additive** (new endpoint, new optional field, new enum value on a string field). After any such change: `uv run python scripts/export_openapi.py`, then `cd frontend && npm run gen:api`, then commit both regenerated files. `tests/contract/test_api.py` locks the schema and must be updated in the same commit.
- Tests are hermetic. No network, no GPU, no real weights. A new test that needs an image builds it from `tests/fixtures/make_synthetic_fixtures.py` or writes a tiny PNG with PIL into `tmp_path`.
- Every fix to a silent failure gets a test whose name states the behaviour (the repo already does this: `test_an_invented_number_survives_into_the_trace_as_a_flagged_span`). Match that style.

---

## Phase 1 — Data Pipeline Integrity

### 1.1 `hashes_for` never sees a BigEarthNet image

**Audit ref:** §1 "Brutal truths" 2 · §5 "Brutal truths" 2
**Files:** `scripts/build_corpus.py` (`bind_view_paths` L331-365, `_hashes_of` L550, `hashes_for` L557-571, `_iter_source_samples` L634)

**What is wrong.** `bind_view_paths()` deliberately does not set `image_path` for `CorpusSource.BIGEARTHNET_V2` (L361-362: "carries its own band paths; only the view map is added"). `hashes_for()` reads only `image_path` or `pre_path` (L568). Every BEN row therefore hashes to `{}`; `_meta_for()` in `corpus_builder.py` copies `None` into `SampleMeta.image_sha256 / phash`; `Deduplicator.check()` is called with `sha256=None, image_hash=None` and accepts everything. The audit measured 0 hashes across 18,530 trained lines.

**Fix logic.**

1. Make `hashes_for()` resolve the *dedup image* from the same place the corpus line points at, in this order:
   - `row["image_path"]` (single-image sources),
   - `row["pre_path"]` (CDVQA — hash the pre image; the pair shares an id),
   - **new:** the first entry of `row["view_paths"]` in `VIEW_IDS_BY_SOURCE` order for that source. For BEN that is `TC` (`view_paths["TC"]`), which is the file `image_key_of()` will also return (`sample.views[0].path`), so the sha256 owner key and the image key agree.
2. `hashes_for` needs the `source` to know the canonical first view. Change its signature to `hashes_for(source: CorpusSource, row: Mapping[str, Any])` and pass `source` at L634. Keep the memoised `_hashes_of(path)` unchanged.
3. Keep the "not on disk → `{}`" behaviour for `--on-missing-views keep`, but make it **loud** (see 1.3).

**Regression test (new, `tests/unit/test_corpus_script.py` — see 1.5 for why a new file):**
`test_bigearthnet_rows_are_hashed_from_their_true_colour_view`: write one 8×8 PNG to `tmp_path/bigearthnet_v2/P1/TC.png` (plus the other six view names, any content), call `build_corpus.bind_view_paths(BIGEARTHNET_V2, row, "P1", tmp_path)` then `hashes_for(BIGEARTHNET_V2, row)`, assert both `image_sha256` and `phash` are present and `image_sha256 == sha256_of(tmp_path/.../TC.png)`. A second test asserts the same for VRSBench (`image_path`) and CDVQA (`pre_path`), so the three code paths are each pinned.

### 1.2 Bind the hashes end-to-end through `_iter_source_samples`

**Audit ref:** §5 "Brutal truths" 2 ("a test that builds one BEN row through `_iter_source_samples` and asserts `meta.image_sha256 is not None` would have failed the build")
**Files:** `scripts/build_corpus.py` (`_iter_source_samples` L618-642), `tests/unit/test_corpus_script.py`

**Fix logic.** No code change beyond 1.1 is needed for the happy path, but the seam must be tested at the level that broke: the *script* stream, not the library.

**Regression test:** `test_iter_source_samples_binds_a_hash_to_every_bigearthnet_sample`. Monkeypatch `build_corpus.load_rows_by_split` to yield one `("train", ben_row)` (reuse the `ben_row` fixture in `test_vlm_training.py` — move it to `tests/conftest.py` so both files see it), render seven tiny PNGs under `tmp_path`, build an `argparse.Namespace(sources=["bigearthnet_v2"], views_root=tmp_path, limit=None, download=False, require_views=True, on_missing_views=None, seed=0)`, drain `_iter_source_samples`, and assert every yielded `CorpusSample.meta.image_sha256` and `.phash` is non-`None`. Then feed the same stream into `cb.build_corpus_streaming(..., dedup=Deduplicator())` twice (same rows yielded twice) and assert `report.dropped_exact > 0` — i.e. the dedup now *fires* on BEN.

### 1.3 ★ NEW — An unrenderable image hashes to `{}` silently

**Files:** `scripts/build_corpus.py` (`_hashes_of` L550-555, `hashes_for`), `src/satquery/training/corpus_builder.py` (`_greyscale` L1616-1631)

**What is wrong.**
- `_hashes_of()` returns `None` for a missing file, and `hashes_for()` turns that into `{}` — the same value as "this source has no pixels". With `--on-missing-views keep` (the default) the corpus line is written *and* silently exempted from dedup. Two silent failures compound.
- `phash(path)` → `_greyscale()` → `PIL.Image.open()` raises `UnidentifiedImageError` / `OSError` on a truncated JPEG. Neither `_hashes_of` nor `hashes_for` catches it, so one corrupt file aborts a 20-hour build with a PIL traceback rather than a `CorpusError` naming the item.

**Fix logic.**
- `hashes_for` returns a small result type (or the dict plus a `reason`) distinguishing `HASHED`, `MISSING`, `UNREADABLE`. `_iter_source_samples` counts each and prints the counts in the per-source summary line (`kept … unhashed_missing … unhashed_unreadable …`).
- Under `--on-missing-views fail` / `--require-views`, `MISSING` raises `CorpusError(f"{item_id}: dedup image missing: {path}")` — today `check_views` only checks the *sample's* views after conversion; hashing happens before and never checks.
- `UNREADABLE` always raises `CorpusError` with the path and the PIL message. A corrupt render is a render-pass bug and must not enter training under any policy.
- `main()` prints a final `WARNING — N samples entered the corpus without an image hash; dedup did not check them` when the count is non-zero, and exits non-zero if `--require-views` was given.

**Tests:** `test_a_corrupt_view_is_a_build_error_not_a_traceback` (write 10 bytes of garbage as `TC.png`, expect `CorpusError` mentioning the path); `test_missing_views_are_counted_and_reported_under_keep`.

### 1.4 ★ NEW — The streaming build never attaches FactSheets, so `evidence_qa` cannot be generated in-band

**Files:** `scripts/build_corpus.py` (`bind_view_paths`, `_iter_source_samples`, `main`), `scripts/render_views.py` (L545-558, factsheet output)

**What is wrong.** `from_bigearthnet()` reads `row.get("fact_sheet")` and `iter_evidence_qa()` returns immediately when `sample.fact_sheet` is empty. Nothing in `build_corpus.py` ever loads `data/processed/views/bigearthnet_v2/factsheets.jsonl` (written by `render_views.py`) into the row. So the in-band `evidence_qa` path prints its "note: … no BigEarthNet sample carried one" message on every real build, and the evidence corpus that was actually trained on came from the standalone `training/data/builders/evidence_qa.py` — a second code path with its own split logic (see 1.5) and no image hashes at all (its `SampleMeta` sets neither `image_sha256` nor `phash`).

**Fix logic.**
- Add `--factsheets PATH` to `parse_args` (default: `<views_root>/bigearthnet_v2/factsheets.jsonl` if it exists).
- Add a `FactSheetIndex` loader: read the JSONL once, build `dict[patch_id -> {"fact_sheet": …, "views": …, "labels": …}]`. ~25k entries × a few hundred bytes is well within memory and does not violate the streaming design (the *rows* stream; the *index* is a bounded side table).
- In `bind_view_paths` for BEN: if the patch is in the index, set `row.setdefault("fact_sheet", entry["fact_sheet"])` and prefer `entry["views"]` over the conventional `view_paths` (the renderer knows which views it actually wrote; a patch missing S1 gets no SAR views rather than a black placeholder path).
- Because `iter_evidence_qa` copies `sample.meta` with `model_copy(deep=True)`, the generated evidence samples inherit the BEN hashes from 1.1 for free, and `image_key_of()` returns the same `TC` path, so `Deduplicator.check` treats them as further annotations of an admitted image (the "seen is not None → accepted, not re-indexed" branch) rather than as duplicates.
- Deprecate the mock path in `training/data/builders/evidence_qa.py` (`generate()` over placeholder views) behind an explicit `--allow-mock` flag; a training corpus must never be built from it by accident.

**Tests:** `test_factsheet_index_attaches_measurements_to_bigearthnet_rows`; `test_evidence_qa_is_generated_in_band_when_factsheets_are_bound` (drive `_iter_source_samples` with `sources=["bigearthnet_v2","evidence_qa"]` and assert at least one `EVIDENCE_QA` sample is yielded with `meta.image_sha256` set).

### 1.5 `evidence_qa` has no validation split — root cause and fix

**Audit ref:** §1 "Brutal truths" 3
**Files:** `training/data/builders/evidence_qa.py` (`generate_from_factsheets` L330-389), `scripts/render_views.py` (L551-558)

**Root cause (verified by reading, not guessed).** `render_views.py` merges `factsheets.*.jsonl` in `sorted()` order into `factsheets.jsonl`: `factsheets.train.jsonl` (~20,000 rows) precedes `factsheets.validation.jsonl`. `generate_from_factsheets` walks records sequentially and stops at `len(samples) >= count`. With `--count 3000` it never reaches a validation record, so every generated sample carries `split="train"` and `evidence_qa.val.jsonl` is empty.

**Fix logic.**
- Stratify: partition `records` by `split` first, allocate `count` proportionally (`round(count * n_split / n_total)`, minimum 1 for any non-empty split, remainder to train), and draw from each partition with the seeded rng. Keep `PASSES_OVER_RECORDS` semantics per partition.
- Refuse silently-empty splits: if the factsheet file contains validation records but the output has zero val samples, raise `SystemExit` with a message. If the file has *no* validation records at all, print a warning naming the fact.
- Once 1.4 lands, the streaming build path becomes the primary route and this script becomes the standalone/debug route; keep both correct.

**Tests:** `test_evidence_qa_val_split_is_populated_when_factsheets_carry_validation_rows` (12 fake records, 9 train / 3 validation, `count=4` → at least one val sample). Add `tests/unit/test_evidence_qa_builder.py`; the builder currently has no test file of its own.

### 1.6 Bind hashes for every source, then rebuild with VRSBench + CDVQA

**Audit ref:** §1 "Brutal truths" 1, §7 item 3
**Files:** `scripts/build_corpus.py` (`COMPOSITION` L~100-135, `LOCAL_SOURCES`), `run_overnight.sh`, `DOCS/DATA_ADAPTATION_PLAN.md`

**Fix logic.** This is an *operational* item that 1.1-1.4 unblock, not a code change of its own:
1. Re-budget `COMPOSITION` against the measured 0.265 samples/s: one 12-hour epoch is ~11,500 samples. Record the arithmetic in the `COMPOSITION` docstring, not only in a log. Proposed mix for the next run (to be confirmed by whoever owns training): BEN-txt 4,000 · BEN-labels 1,500 · evidence_qa 2,500 · VRSBench 2,500 · CDVQA 1,000; DIOR-RSVG and RSVQA-HR remain 0 until resolved.
2. Run `build_corpus.py --sources bigearthnet_v2,evidence_qa,vrsbench,cdvqa --on-missing-views skip --factsheets …` and assert in the log that `dropped_exact + dropped_near > 0` **and** the unhashed count from 1.3 is 0.
3. Write `data/processed/corpus/MANIFEST.md` (audit §1 "Smaller debt"): one row per JSONL — sources, line count, hashed count, build command, seed, which `runs/*` consumed it. `build_corpus.main()` appends a row automatically at the end of every successful build.

### 1.7 ★ NEW — `Reservoir` subsampling ignores the split, so val can silently shrink to zero

**Files:** `src/satquery/training/corpus_builder.py` (`build_corpus_streaming` L1960-2040, `Reservoir` L1905)

**What is wrong.** One reservoir per *source* samples uniformly from the stream and the train/val split is applied *after* selection. With BEN's ~6% validation share and a small per-source target (e.g. 200 in a probe), the expected val count is ~12 and the variance is high; a target of 50 can plausibly yield 0 val samples with no error. The overnight probe corpora under `data/processed/corpus/poc/` show exactly this shape.

**Fix logic.** Key reservoirs by `(source, split)`. Split the per-source target with the same proportional rule as 1.5 (val target = `max(1, round(target × val_share))` where `val_share` is a constant per source, documented next to `COMPOSITION`). After collection, if a source produced any val candidates but zero val samples, raise `CorpusError`. Keep `SourceReport.train/val` so the printed table shows the outcome.

**Test:** `test_streaming_build_never_drops_a_split_it_was_offered` — 95 train + 5 val samples, target 20, assert `report.sources[..].val >= 1` for every seed 0-9.

### 1.8 Remaining audit hygiene in this phase

| Item | File | Logic |
|---|---|---|
| `mypy --strict` error | `src/satquery/training/local_sources.py:132` | Annotate the parsed JSON (`cast(list[dict[str, Any]], …)` or a `TypeGuard`) so the function does not return `Any`. |
| Root `README.md` is 0 bytes | `README.md` | Fill from `DOCS/AI_HANDOFF/00_START_HERE.md` + `05_ENVIRONMENT_AND_SETUP.md`: one-paragraph purpose, `uv sync` / `npm i` / `make demo`, link to the contract and the audit. |
| DIOR-RSVG unresolved (gated) | `scripts/build_corpus.py` `UNRESOLVED_SOURCES` | No code fix. Record the licence-acceptance step in `README.md` and keep the source excluded with its existing message. |

---

## Phase 2 — Inference & Concurrency

### 2.1 Move the GPU semaphore to the process level

**Audit ref:** §3 "Brutal truths" 3, §7 item 6
**Files:** `src/satquery/agent/executor.py` (L258-274 constructor, `_guarded` L700-720), `src/satquery/agent/pipeline.py` (L214 `DagExecutor(...)`), `src/satquery/api/app.py` (lifespan L52-62), `src/satquery/api/dependencies.py`, `src/satquery/api/routers/analyze.py`, `src/satquery/api/routers/jobs.py`, **new** `src/satquery/agent/concurrency.py`

**What is wrong.** `DagExecutor.__init__` creates `self._gpu = asyncio.Semaphore(max_parallel_gpu)` per instance, and `pipeline.analyze()` creates one executor per request. Two concurrent jobs each hold their own permit and both load SegFormer-B5, the CD model and DOFA on top of the resident VLM.

**Fix logic.**
1. New module `agent/concurrency.py` with a `DeviceGates` class holding one `asyncio.Semaphore` per `Device` (`ROCM_0` → `MAX_PARALLEL_GPU_TOOLS`, `CPU` → `MAX_PARALLEL_TOOLS`) and a `for_device(device) -> Semaphore` accessor.
2. **Do not make it a module-level global.** Since Python 3.10 an `asyncio.Semaphore` binds to the event loop on first *contended* `acquire()`; a global bound to one loop raises `RuntimeError: … is bound to a different event loop` when contended from another — which pytest-asyncio's per-test loops will do. Instead:
   - create the `DeviceGates` in `create_app()`'s `lifespan` and store it on `app.state.gates`;
   - expose `get_device_gates(request: Request) -> DeviceGates` in `dependencies.py`;
   - thread it `router → pipeline.analyze(gates=…) → DagExecutor(gates=…)`. `DagExecutor` keeps its constructor defaults (a private `DeviceGates()` when none is passed) so the 30 executor tests are untouched.
3. `_guarded()` picks `self._gates.for_device(spec.device)` instead of `self._gpu / self._cpu`.
4. The VLM's `LazyBackend` `RLock` stays: it serialises *generation*; the gate serialises *VRAM residency* across tools. Both are needed.

**Tests:** `test_two_executors_share_one_gpu_permit` — build two `DagExecutor`s over one `DeviceGates(gpu=1)`, register a fake ROCm tool whose `run` records `max_concurrent` with a threading counter and sleeps 50 ms, run both plans with `asyncio.gather`, assert `max_concurrent == 1`. A sibling asserts two CPU tools *do* overlap, so the fix is not "everything serialised".

### 2.2 ★ NEW — A timed-out GPU tool releases its permit while its thread still holds VRAM

**Files:** `src/satquery/agent/executor.py` (`_guarded` L700-720)

**What is wrong.** `async with semaphore: await asyncio.wait_for(asyncio.to_thread(...))` — on timeout `wait_for` cancels the *awaiter*, the `async with` exits and releases the permit, but the worker thread (the docstring admits this) keeps running and keeps its model on the GPU. The next GPU step acquires the permit immediately and can OOM against a tool the executor believes has finished.

**Fix logic.** Acquire the permit manually. Wrap the thread future in `asyncio.shield()` before `wait_for`. On `TimeoutError`, do **not** release; instead attach `future.add_done_callback(lambda _: gate.release())` so the permit returns when the thread actually ends, and record a `WarningItem(code="TOOL_TIMEOUT_LEAKED_PERMIT", …)` so the trace says the box is degraded. Add a hard ceiling (`MAX_LEAKED_PERMITS`) beyond which `/v1/health` reports `degraded` (see 2.8).

**Test:** fake tool that sleeps 300 ms with `timeout_ms=50`; assert the second GPU step does not start until the first thread finishes (measure with `time.perf_counter`), and that the warning is present.

### 2.3 Unblock the event loop: `ingest()` in all three routers

**Audit ref:** §3 "Brutal truths" 2, §7 item 6
**Files:** `src/satquery/api/routers/analyze.py` (L49), `src/satquery/api/routers/jobs.py` (L157), `src/satquery/api/routers/validate.py` (L39 — **missed by the audit**, same defect), `src/satquery/api/uploads.py` (`persist_uploads` L36-88)

**Fix logic.**
- Replace each `ingest(...)` call with `await asyncio.to_thread(ingest, sources, pair_type_hint=…, roles=…, max_images=…)`. `ingest` is pure CPU/IO (rasterio + phase correlation) and touches no asyncio state; `IngestError` propagates through `to_thread` unchanged so the `@app.exception_handler(IngestError)` path is unaffected.
- `persist_uploads` also reads the spooled upload synchronously in the handler (`upload.file.read(CHUNK_BYTES)` loop). Move the copy loop to a thread as well (`await asyncio.to_thread(_copy_uploads, …)`), or use `await upload.read(CHUNK_BYTES)` (Starlette's async API). Prefer the former: it keeps the byte-budget check in one synchronous function that is already tested.
- Also in `pipeline.analyze()`: `traces.put(trace)` (L308) is a synchronous SQLite write on the loop. Wrap in `to_thread`; `TraceStore` already opens connections with `check_same_thread=False`.

**Test:** `tests/contract/test_jobs_sse.py::test_ingestion_does_not_stall_the_heartbeat` — monkeypatch `ingest` with a function that `time.sleep(0.5)`s, open an SSE stream on an *earlier* job in the same app, POST a new job concurrently (via `anyio` task group with `httpx.AsyncClient`), and assert a `: ping` or event frame arrives on the first stream within 0.2 s. Set `HEARTBEAT_SECONDS` low via monkeypatch for the test.

### 2.4 `DELETE /v1/jobs/{id}` — a cancel that cancels

**Audit ref:** §3 "Brutal truths" 4, §4 "Brutal truths" 2, §7 item 5
**Files:** `src/satquery/api/jobs.py` (`Job` dataclass, `JobStore`), `src/satquery/api/routers/jobs.py` (`_execute`, new route), `src/satquery/agent/executor.py` (`run()` loop L282-300), `src/satquery/agent/pipeline.py` (`AnalysisRequest`), `src/satquery/schemas/api.py` (`JobStatusResponse`), `src/satquery/api/app.py` (L79 CORS), `openapi.json`, `frontend/src/api/schema.d.ts`, `DOCS/API_CONTRACT.md` (§4), `tests/contract/test_api.py`, `tests/contract/test_jobs_sse.py`

**Fix logic.**
1. **Hold the handle.** Add `task: asyncio.Task[None] | None` and `cancel_requested: asyncio.Event` to `Job`. `create_job` assigns `job.task = task` after `create_task`.
2. **Cooperative cancellation in the executor.** `DagExecutor.run()` accepts an optional `cancelled: asyncio.Event`. At the top of each `while pending:` iteration, if set, mark every remaining step `SKIPPED` with `error="cancelled by client"` and return the partial `ExecutionReport`. This is the deterministic boundary: the in-flight tool finishes (its thread cannot be killed), the next one never starts. Thread `AnalysisRequest.cancelled` → `executor.run(cancelled=…)`.
3. **Route.** `DELETE /v1/jobs/{job_id}` → 404 if unknown; **409** if already terminal (`succeeded`/`failed`); otherwise set `job.cancel_requested`, `job.task.cancel()`, return **202** with the `JobStatusResponse` snapshot. Document in API_CONTRACT §4.9 that cancellation is best-effort and the current tool step completes.
4. **Terminal event on cancel.** `_execute()` currently catches `Exception` only, so a cancelled task ends with **no terminal event and every subscriber hangs**. Add `except asyncio.CancelledError:` that publishes `error` with `ApiError(code="JOB_CANCELLED", http_status=499, message="The run was cancelled by the client.")` and re-raises. `_absorb` maps that to `status="failed"`; no new `JobStatus` literal is needed, so the frozen schema is untouched. Add `"JOB_CANCELLED"` to the documented error codes.
5. **CORS.** `allow_methods` in `app.py` is `["GET","POST","OPTIONS"]`; add `"DELETE"` or the browser preflight fails whenever the frontend is served from a different origin than the API (the Vite proxy hides this in dev; a Docker deploy will not).
6. **Contract.** Regenerate `openapi.json` and `schema.d.ts`; add the route to `tests/contract/test_api.py`'s path list.

**Tests (`test_jobs_sse.py`):** `test_delete_cancels_a_running_job_and_terminates_the_stream` (fake slow tool; DELETE mid-run; assert the SSE stream ends with `error.code == "JOB_CANCELLED"` and every unstarted step is `SKIPPED` in the poll snapshot), `test_delete_on_a_finished_job_is_409`, `test_delete_unknown_job_is_404`.

### 2.5 Key-aware citation matching

**Audit ref:** §3 "Brutal truths" 1, §7 item 4
**Files:** `src/satquery/evidence/citation_validator.py` (`resolve` L191-206, `validate` L209-249, `NumericSpan`), `src/satquery/models/prompts/builder.py` (`strip_citation_markers` L267-292, `MarkedAnswer` L104-117, `CITATION_MARKER` L34), `src/satquery/tools/vlm_runtime.py` (L245-256), `src/satquery/agent/aggregator.py` (`vlm_answer` L272-307, `aggregate` L325-365)

**What is wrong.** `strip_citation_markers` removes `[key]` before `validate` runs, so the validator never knows which key the model attached to which number. `resolve()` then matches by value alone: `0.75 [sar_backscatter_analyzer.sigma0_vv_db_mean]` passes if *any* unitless fact is within 0.05 of 0.75. The key the adapter was trained to emit — the strongest signal — is discarded, and the `unknown_keys` list only catches keys absent from the sheet, not keys that name the wrong fact.

**Fix logic.**
1. **Keep marker positions.** `strip_citation_markers` records, for each marker, the character offset in the *cleaned* text at which it was removed: `MarkedAnswer.markers: list[tuple[int, str]]` (offset, key). Because markers are removed left-to-right and whitespace is collapsed afterwards, compute offsets on the intermediate string *before* the `[ \t]{2,}` collapse and adjust, or perform the collapse inside the same substitution pass so offsets stay exact. Pin this with a test on a string containing two adjacent markers.
2. **Bind spans to keys.** In `validate(text, sheet, policy, markers=None)`: for each `NumericSpan`, find the nearest marker whose offset is ≥ `span.end` and ≤ `span.end + MAX_MARKER_GAP` (≈ 3 characters — a closing paren or a comma is allowed, a whole clause is not), and not already claimed by an earlier span. If found:
   - if `key not in sheet` → uncited, reason `UNKNOWN_KEY`;
   - else if `not unit_compatible(fact.scalar, span.unit)` or `not within_tolerance(span.value, fact.value)` → uncited, reason `KEY_VALUE_MISMATCH` (this is the case the audit describes);
   - else → citation to **that** fact; do not run the value search at all.
   A span with no adjacent marker falls back to today's `resolve()` by value.
3. **Report the reason.** Extend `ValidationResult.uncited_numeric_spans` from `list[str]` to keep the wire type (frozen) but add a parallel `uncited_reasons: list[str]` on the dataclass (not the schema). `vlm_runtime.synthesise` puts reasons into the TEXT artifact's `inline` dict (additive) and into the step note text so the trace explains *why* a span failed.
4. **Do it once, in one place.** `vlm_runtime` validates the tool's own output with markers; `aggregator.aggregate()` validates the final answer text without markers (the text artifact is already stripped). Add `markers` to the TEXT artifact's inline payload (`"citation_markers": [[offset, key], …]`) so the aggregator can pass them through to `validate` and both passes agree. Templated answers (`compose()`) carry no markers and keep the value path.

**Tests (`tests/unit/test_agent.py` or a new `test_citation_validator.py`):**
`test_a_right_number_with_the_wrong_key_is_uncited`, `test_a_bare_number_still_resolves_by_value`, `test_a_marker_binds_to_the_span_immediately_before_it_not_an_earlier_one`, `test_marker_offsets_survive_whitespace_collapse`. Also extend the corpus-side audit: `corpus_builder.assemble()` runs the same validator, so a template that emits a wrong key now fails the build — add `test_a_template_citing_the_wrong_key_is_a_build_error`.

### 2.6 ★ NEW — `parse_options` swallows malformed options and unknown fields

**Files:** `src/satquery/api/uploads.py` (`parse_options` L26-33), `src/satquery/schemas/api.py` (`AnalyzeOptions` L25-44), `src/satquery/api/routers/{analyze,jobs,validate}.py`

**What is wrong.**
- `parse_options` catches `json.JSONDecodeError` and `ValueError` (which includes pydantic's `ValidationError`) and returns default `AnalyzeOptions()`. A client sending `{"seed": "abc"}` or `{"disable_tools": "vlm"}` gets a silently different run than it asked for, with a 200.
- `pair_type_hint: str | None` is not an enum; `PairType(parsed.pair_type_hint)` in each router raises `ValueError` → an unhandled **500** without the §6 envelope.
- `vlm_backend`, `adapter_version`, `max_latency_ms`, `artifact_format` are accepted and never read by the pipeline (grep confirms only `settings.vlm_backend` is consumed). The contract advertises them.

**Fix logic.**
- `parse_options` raises a new `IngestError` subclass `InvalidOptionsError(code="INVALID_OPTIONS", http_status=400)` carrying pydantic's error list in `hint`. The existing exception handler renders it as the envelope.
- Type `pair_type_hint` as `PairType | None` on the model so validation happens in one place; delete the three `PairType(...)` conversions.
- For the four unread options: honour `max_latency_ms` by passing it into `DagExecutor` as an overall deadline (a `WarningItem(code="LATENCY_BUDGET_EXCEEDED")` when the DAG overruns, no cancellation), and for the other three either wire them or mark them `deprecated=True` in the `Field` and add a warning `OPTION_IGNORED` to the response. Additive either way.

**Tests:** `test_malformed_options_are_a_400_not_a_silent_default`, `test_an_invalid_pair_type_hint_is_a_400_with_the_error_envelope`.

### 2.7 ★ NEW — `query` length is documented but not enforced

**Files:** `src/satquery/api/routers/analyze.py` (L34), `src/satquery/api/routers/jobs.py` (L137)

`Form(description="1-1000 chars")` has no `min_length`/`max_length`. An empty query reaches the classifier. Add `min_length=1, max_length=1000` on both `Form()` declarations (FastAPI turns violations into 422, matching the contract's "malformed request" row). Strip leading/trailing whitespace before the check, or an all-spaces query still passes. Test: `test_an_empty_query_is_rejected_before_ingestion`.

### 2.8 `JobStore` lifecycle: running jobs can be evicted, restarts are invisible, shutdown leaks

**Audit ref:** §3 "Brutal truths" 4
**Files:** `src/satquery/api/jobs.py` (`JobStore.create` L96-102), `src/satquery/api/routers/jobs.py` (`_RUNNING`, `_execute`), `src/satquery/api/app.py` (lifespan), `src/satquery/api/routers/health.py`

**What is wrong (beyond the audit's "in-memory" point).**
- `create()` evicts oldest-first regardless of status. Under load the 65th job evicts a *running* job: its task keeps publishing into an orphaned `Job`, and the client's poll/SSE gets 404 mid-run.
- `Job.subscribers` queues are unbounded; a subscriber that never reads (a stuck proxy) grows without limit for the life of the job.
- `lifespan` does not cancel `_RUNNING` on shutdown, so uvicorn reload/exit leaves `satquery-upload-*` temp directories behind and GPU threads running.
- After a restart every job id is unknown; a client mid-stream gets 404 with no way to distinguish "never existed" from "the server restarted".

**Fix logic.**
- Evict only terminal jobs; if every retained job is running, refuse the new one with **429** (`code="TOO_MANY_JOBS"`), which is honest back-pressure.
- Bound each subscriber queue (`maxsize=1024`); on `QueueFull`, drop the subscriber and log — the buffered `events` list is still replayed on reconnect.
- In `lifespan` shutdown: set every running job's `cancel_requested`, cancel `_RUNNING` tasks, `await asyncio.gather(*_RUNNING, return_exceptions=True)`, so `_execute`'s `finally` removes the upload directories.
- On `GET /v1/jobs/{id}` 404, if `traces.get(job_id)` succeeds, return the trace-derived terminal snapshot (`status="succeeded"`, `result` from the trace) instead — the trace store is the durable half by design, so use it. Otherwise 404 with `hint="The job may predate a server restart; check /v1/traces/{id}."`.
- Expose `jobs.running` / `jobs.retained` / leaked GPU permits (2.2) in `/v1/health` (additive fields).

**Tests:** `test_a_running_job_is_never_evicted`, `test_a_finished_job_is_recoverable_from_its_trace_after_eviction`, `test_shutdown_cancels_running_jobs_and_removes_their_uploads`.

### 2.9 ★ NEW — SSE frames carry no `id:`, so reattachment must replay everything

**Files:** `src/satquery/api/routers/jobs.py` (`_sse`, `_stream`, `job_events`), `src/satquery/api/jobs.py` (`subscribe`)

**Fix logic (prerequisite for 3.1).** Number events at publish time (`JobEvent.seq`, the index in `job.events`) and emit `id: {seq}` on every frame. `job_events` reads the standard `Last-Event-ID` request header (and a `?after=` query fallback for clients that cannot set headers) and `subscribe(job, after=seq)` skips the backlog up to and including that id. Full replay stays the default so the existing late-subscriber contract test is unchanged. Test: `test_last_event_id_resumes_without_replaying_delivered_events`.

### 2.10 Remaining audit hygiene in this phase

| Item | File | Logic |
|---|---|---|
| `export_openapi.py` treats any argv as a path | `scripts/export_openapi.py` L35-37 | Replace `sys.argv[1]` with `argparse` (`--out`, default `openapi.json`); `--help` must not write a file. |
| Leftover scaffold | `src/satquery_ai/__init__.py`, `pyproject.toml` `[project.scripts]` | Delete the package and the `satquery-ai = "satquery_ai:main"` entry (or repoint it at a real CLI). |
| `ruff` import-sort error | `scripts/test_inference.py` | `uv run ruff check --fix`. |
| `ExecutionCache` bounded by count, not bytes | `src/satquery/agent/executor.py` L184-220 | 256 entries of encoded PNG/GeoTIFF bytes can be several GB. Track `sum(len(blob))` and evict on a byte budget (`SATQUERY_CACHE_MB`, default 512). |
| GGUF exported from the PoC adapter | `runs/poc-v1/export-Q4_K_M.gguf` | Re-export from whichever adapter `SATQUERY_VLM_ADAPTER_PATH` points at, and record the source adapter in the GGUF's sidecar JSON. Operational; do after the training rerun. |

---

## Phase 3 — Frontend Resilience

### 3.1 Reattach on a dropped SSE stream instead of re-running the job

**Audit ref:** §4 "Brutal truths" 1, §7 item 5
**Files:** `frontend/src/thread/useRun.ts`, `frontend/src/api/client.ts` (`streamJob` L300-371, `jobStatus` L234, `STREAM_STALL_MS` L43), `frontend/src/api/sse.ts` (`parseFrame` — currently discards `id`), `frontend/src/state/job.ts` (`reduce`, store actions), `frontend/src/components/thread/ThreadPanel.tsx` (L200-235 failure banner), `frontend/src/mocks/handlers.ts` (jobs handlers L79-130), `frontend/src/mocks/__tests__/stream.test.ts`, `frontend/src/state/__tests__/job.test.ts`

**What is wrong.** `useRun.submit` wraps `createJob` + `streamJob` in one `try`; any failure after the 202 calls `markFailed()` and the only recovery offered is `retry()`, which calls `submit(lastQuery)` — a new upload and a new GPU run. `streamJob`'s docstring explicitly chooses not to reconnect.

**Fix logic.**
1. **Split the two failures.** In `useRun`, keep `accepted` and the `job_id` in a ref. A failure *before* the 202 stays as today (retry = resubmit). A failure *after* the 202 enters a new phase `'reconnecting'` (add to `JobPhase`; client-only, never on the wire) and calls a new `resume(jobId)`.
2. **`resume` is a bounded loop, not an infinite one.** Up to `MAX_REATTACH = 5` attempts with backoff `1s, 2s, 4s, 8s, 8s`:
   - first `jobStatus(jobId)` (cheap, tells us whether the job still exists and whether it already finished);
   - if `status` is `succeeded`/`failed`, synthesise the terminal `done`/`error` event from the snapshot's `result`/`error` and apply it — no stream needed;
   - if 404, stop: the server restarted or evicted the job; show "The server no longer knows this run" with the retry (resubmit) button — this is the *only* case that should resubmit;
   - otherwise re-open `streamJob(jobId, …, { lastEventId })`.
3. **Do not double-apply events.** `sse.ts::parseFrame` keeps the `id` field (`SseFrame.id?: string`); `streamJob` tracks the last id it delivered and sends it as `Last-Event-ID` on reattach (server side: 2.9). Until 2.9 ships, the reducer is already safe against a full replay because `queued` resets state to `initialJobState` and `artifact` de-duplicates by id — the only visible cost is the DAG flashing back to `PENDING` for a few frames. Once 2.9 ships, that flash disappears.
4. **Make the stall watchdog reconnect-aware.** Today a stall throws a `SatQueryError` with `retryable: true`. Route it to `resume`, not `retry`. Keep the stall timer at 60 s (4× the 15 s heartbeat).
5. **UI.** `ThreadPanel` shows a distinct banner for `'reconnecting'` ("Connection lost — reattaching to the run (attempt 2 of 5)…") with a cancel button; the retry-resubmit button appears only for the 404 case or after the attempts are exhausted.
6. **Test in MSW.** Add a scenario to `mocks/scenarios.ts` where the first `/events` response closes after the `plan` event with no terminal frame and the second returns the full sequence; `stream.test.ts` asserts the store ends in `succeeded`, `createJob` was called **once**, and `jobStatus` was called before the second `/events`. A second scenario returns 404 from `jobStatus` and asserts phase `failed` with `retryable: true`.

### 3.2 Wire the cancel button to `DELETE /v1/jobs/{id}`

**Audit ref:** §4 "Brutal truths" 2
**Depends on:** 2.4
**Files:** `frontend/src/api/client.ts` (new `cancelJob(jobId, signal)` → `send(path, {method:'DELETE'})`), `frontend/src/thread/useRun.ts` (`cancel` L96-101), `frontend/src/state/job.ts` (`markFailed` → accept an optional `ApiError` so the store can carry `JOB_CANCELLED`), `frontend/src/api/events.ts` (no change — `error` payload is already `ApiError`), `frontend/src/mocks/handlers.ts` (add `http.delete('*/v1/jobs/:jobId')`), `frontend/src/components/thread/ThreadPanel.tsx`

**Fix logic.** `cancel()` becomes: fire `cancelJob(jobId)` (fire-and-forget with its own short timeout; the UI must not wait on the server to feel responsive), keep the SSE stream **open** so the `error{JOB_CANCELLED}` event arrives and the reducer closes the phase naturally, and only if the DELETE fails (404/network) fall back to today's local `markFailed()`. The banner copy changes from "Run cancelled. The question is still in the box." to distinguish "cancelled on the server" from "stopped watching; the server may still be working". Also call `cancelJob` from the `submit()` path when a previous run is aborted by a new question (today it only aborts the fetch), and from the unmount cleanup.

**Test:** `resilience.test.tsx`: click cancel → DELETE handler called with the job id → store phase `failed` with `error.code === 'JOB_CANCELLED'`; and a second test where the DELETE handler returns 404 → the local fallback banner appears.

### 3.3 Consume the `BBOX_SET` artifact; keep the text parser as fallback only

**Audit ref:** §4 "Brutal truths" 3
**Files:** `frontend/src/components/stage/ImageViewer.tsx` (L205-208 `parseBboxTokens(result.answer.text)`), `frontend/src/components/thread/GroundedAnswer.tsx` (L26, L92-93), `frontend/src/components/thread/ThreadPanel.tsx` (L38 `stripBboxTokens`), `frontend/src/components/stage/BboxOverlay.tsx` (`NormalisedBox` consumer), `frontend/src/thread/bbox.ts`, **new** `frontend/src/thread/boxes.ts`, `frontend/src/api/types.ts`, `frontend/src/thread/__tests__/bbox.test.ts`

**What the backend already emits** (`src/satquery/tools/text_grounding.py` L177-230): an `ArtifactRef` with `type: "BBOX_SET"` and `inline: { boxes: [{ id, label, score, bbox_px: [x1,y1,x2,y2], bbox_normalised: [x1,y1,x2,y2] /* 0-1000 */, bbox_wgs84: [minLon,minLat,maxLon,maxLat] | null }], frame: { width, height }, source_image: "img_0" }`.

**Fix logic.**
1. New `thread/boxes.ts` exporting `boxesForResult(result: AnalyzeResponse | null, artifacts: ArtifactRef[]): { boxes: NormalisedBox[]; source: 'artifact' | 'text' | 'none'; sourceImage: string | null }`:
   - find the **last** `BBOX_SET` artifact in the run (same "last synthesiser wins" rule the aggregator uses);
   - map `bbox_normalised` → `NormalisedBox` directly (the overlay already draws in the 0-1000 frame with `viewBox="0 0 1000 1000"`, so no arithmetic); carry `label`, `score`, `id`;
   - only when no `BBOX_SET` exists, fall back to `parseBboxTokens(result.answer.text)` and mark `source: 'text'`.
2. Add a typed `BboxSetInline` interface in `api/types.ts` and a runtime guard (`isBboxSetInline`) — `inline` is `Record<string, unknown>` on the wire, so a malformed payload must degrade to the text fallback, not throw.
3. `ImageViewer` uses `boxesForResult`; `sourceImage` selects which cell's overlay draws the boxes (today the boxes go on the primary regardless — with a bi-temporal grounding step this is a latent wrong-image bug).
4. `GroundedAnswer` and `ThreadPanel` keep `stripBboxTokens` for display (the *text* still contains the tokens and must be cleaned for reading) but take `boxCount` from `boxesForResult`, so the count and the drawing can never disagree.
5. Show a small "boxes from answer text" hint in the overlay when `source === 'text'`, so a judge can see when the frontend fell back — consistent with the honesty-signal design.
6. Keep `bbox.ts` and its parity tests: the fallback must still match `box_format.py`.

**Tests:** `boxes.test.ts` — artifact present → boxes come from the artifact even if the text parses to a different set; artifact absent → text fallback; malformed inline → text fallback; `source_image` propagated. Add a captured fixture with a real `BBOX_SET` to `mocks/captured/` (capture it from a grounding run with `scripts/test_inference.py`, or hand-write it against the schema).

### 3.4 Change legend is hard-coded

**Audit ref:** §4 "Brutal truths" 5
**Files:** `frontend/src/components/stage/ImageViewer.tsx` (L363-374)

**Fix logic.** `result.resolved_task.slots` carries the classifier's `target_class` when the query named one (`task_classifier.py` L77). Legend text becomes `slots.target_class ? `New ${humanise(slots.target_class)}` : 'Detected change'`; `humanise` maps the registry class vocabulary (`built_up` → "built-up area", `water` → "water", …) with the raw slot as fallback. Test in `run-render.test.tsx` with two fixtures (one with `target_class: "water"`, one without).

### 3.5 ★ NEW — A new question or an unmount abandons the server job without cancelling it

**Files:** `frontend/src/thread/useRun.ts` (`submit` L66-69, unmount effect L58)

Covered by 3.2's "also call `cancelJob`" clause; listed separately so it is not forgotten. Every path that calls `inFlight.current?.abort()` while a job id is known must also send the DELETE. Test: submitting a second question while the first streams sends exactly one DELETE for the first job id.

### 3.6 ★ NEW — `rememberRun` history references jobs that may no longer exist

**Files:** `frontend/src/state/ui.ts` (`rememberRun`), `frontend/src/components/panels/HistoryPanel.tsx`

`rememberRun` stores `{traceId, query, at}` on the 202, before the run finishes. A run that fails, is cancelled, or dies with the server appears in history and clicking it fetches `/v1/traces/{id}` → 404. `HistoryPanel` already handles the 404 (L19-26), but the entry stays forever. Fix: `rememberRun` on `done` (not on 202), or store a `status` and grey out non-succeeded entries. Test: a run ending in `error` does not appear in history.

---

## Phase 4 — Tooling & CI

### 4.1 Fix the `oxlint` crash

**Audit ref:** §4 "Brutal truths" 4
**Files:** `frontend/package.json`, `frontend/package-lock.json`, `frontend/node_modules/` (local only)

**Diagnosis (reproduced).** `npm run lint` fails with `Cannot find native binding … Cannot find module '@oxlint/binding-wasm32-wasi'`. `package-lock.json` *does* list `@oxlint/binding-linux-x64-gnu@1.81.0` (L1405) but `node_modules/@oxlint/` does not contain it — this is npm/cli#4828: optional platform dependencies are skipped when the lockfile is updated on a machine where the resolution differs. Not a code bug; an install-state bug.

**Fix logic.**
1. Locally: `rm -rf node_modules package-lock.json && npm install` (exactly what the error message says), verify `ls node_modules/@oxlint/` shows `binding-linux-x64-gnu`, run `npm run lint` and fix whatever it reports (lint has not run since the binding broke, so expect findings). Commit the regenerated lockfile.
2. Pin the platform binding as an explicit `optionalDependencies` entry (`"@oxlint/binding-linux-x64-gnu": "1.81.0"`) so a future lockfile refresh cannot drop it, and add `"lint": "oxlint --deny-warnings"` so warnings fail CI.
3. In CI (4.3) use `npm ci`, which installs from the lockfile and does not exhibit the bug on a clean checkout.

### 4.2 Silence the `react-compare-slider` `CSS.registerProperty` log

**Audit ref:** §4 "Brutal truths" 6
**Files:** `frontend/vitest.config.ts`, **new** `frontend/src/test/setup.ts`

**Diagnosis.** `react-compare-slider/dist/hooks-*.mjs` runs an IIFE at *module load* that calls `CSS.registerProperty(...)` inside `try/catch` and `console.debug`s the failure. happy-dom defines `CSS` but not `registerProperty`, so every test file that imports `ImageViewer` logs once.

**Fix logic.** Add `test.setupFiles: ['./src/test/setup.ts']` to `vitest.config.ts`. The setup file, guarded by `typeof CSS !== 'undefined'`, defines `CSS.registerProperty = () => undefined` (and `CSS.supports ??= () => false`) *before* any component module is imported — `setupFiles` run after the environment is created and before the test file, which is the right moment. Do **not** filter `console.debug` globally; that would hide real debug output. Verify with `npx vitest run src/components/__tests__/run-render.test.tsx` producing no `[react-compare-slider]` line.

### 4.3 GitHub Actions CI

**Audit ref:** §5 "Brutal truths" 1, §7 item 7
**Files:** **new** `.github/workflows/ci.yml`, `pyproject.toml` (`[tool.pytest.ini_options]`), `tests/unit/test_models.py` (marker), **new** `Makefile`

**Design.** Two jobs, both on `ubuntu-latest`, triggered on `push` to `main` and `pull_request`. Nothing needs a GPU: the audit confirms 503 backend tests are hermetic and torch is only imported lazily. Keep the CI runtime under ~6 minutes.

**`backend` job**
1. `actions/checkout@v4`, `astral-sh/setup-uv@v5` with cache, Python `3.11` (from `.python-version`).
2. `uv sync --group dev` (no `--extra vlm`; the serving/training extras are optional by design and pulling ROCm torch into CI is wrong). Because `test_change_detection.py` and `test_vlm_training.py` use `pytest.importorskip`, they skip cleanly.
3. `uv run ruff check .` and `uv run ruff format --check .`
4. `uv run mypy` (strict, per `pyproject.toml`).
5. `uv run pytest -m "not gpu"` — **`--strict-markers` is on, so register the marker first**: add `markers = ["gpu: needs a ROCm/CUDA device and real weights"]` to `[tool.pytest.ini_options]`, and put `@pytest.mark.gpu` on the one `test_models.py` case the audit identified as the "long pole exercising real SDPA on ROCm" (find it with `-k sdpa --durations=5`). Everything else already runs on CPU.
6. Contract drift check: `uv run python scripts/export_openapi.py --out /tmp/openapi.json && diff -q openapi.json /tmp/openapi.json` — fails the build if someone changed a schema without regenerating (the audit verified this diff is currently 0; keep it that way).

**`frontend` job**
1. `actions/setup-node@v4` with `node-version: 22` and `cache: npm`, `cache-dependency-path: frontend/package-lock.json`.
2. `npm ci` (working directory `frontend`).
3. `npm run gen:api && git diff --exit-code src/api/schema.d.ts` — the second half of the contract chain.
4. `npm run lint`, `npm run typecheck`, `npm test`, `npm run build`.

**`Makefile`** (root) mirrors the CI steps so they are one command locally: `make lint`, `make typecheck`, `make test`, `make contract`, `make ci` (all of the above), `make demo` (backend + frontend dev servers — Phase 9's first deliverable). CI calls the `make` targets so the two cannot drift.

**Branch protection.** After the first green run, require both jobs on `main`.

### 4.4 Pre-commit hooks

**Files:** **new** `.pre-commit-config.yaml`, `README.md`

`ruff` (check + format), `mypy` (as a local hook via `uv run mypy`), and a local hook that runs `oxlint` and `tsc -b --noEmit` when any `frontend/src/**` file is staged. Document `uv run pre-commit install` in the README. This is what catches the ruff/mypy one-liners the audit found before they reach CI.

### 4.5 Containerisation (Phase 9, first day)

**Audit ref:** §3 "Brutal truths" 5
**Files:** **new** `Dockerfile`, **new** `docker-compose.yml`, **new** `frontend/Dockerfile`, `.dockerignore`, `src/satquery/core/config.py`

**Fix logic.**
- Backend image from a ROCm PyTorch base (`rocm/pytorch` matching `scripts/rocm_env.sh`), `uv sync --extra vlm --frozen`, `uvicorn satquery.api.app:app`. Weights, `data/`, `runs/` and the SQLite trace DB are **volumes**, never baked in. `.dockerignore` excludes `data/`, `runs/`, `logs/`, `*.log`, `node_modules/`.
- Frontend image: `npm ci && npm run build` then `nginx` serving `dist/` with `/v1` proxied to the backend service — which is why 2.4's CORS `DELETE` addition matters even though dev never hits it.
- `docker-compose.yml` with the `/dev/kfd` and `/dev/dri` device mappings ROCm needs, plus a `cpu` profile that sets `SATQUERY_VLM_BACKEND=none` so `make demo-cpu` works on a laptop with templated answers.
- CI does **not** build the ROCm image (too large); it builds the frontend image and runs `docker build --target deps` for the backend to catch Dockerfile rot.

### 4.6 ★ NEW — `openapi.json` drift is checked by tests only if someone runs them

Covered by 4.3 steps 6 and frontend 3; listed so the reason is recorded: the audit *verified* zero drift by running the exporters manually. That verification is now a CI gate, and after 2.4 adds a route the gate is what proves both regenerated files were committed together.

### 4.7 ★ NEW — Test-fixture ownership

**Files:** `tests/conftest.py`, `tests/unit/test_vlm_training.py`

The `ben_row` fixture and the `_FakeDatasets` stand-in live inside `test_vlm_training.py` (a 100+ test file that also needs `importorskip("torch")` for part of its content). Phase 1's new `test_corpus_script.py` needs both. Move them to `tests/conftest.py` so the corpus-script tests never import the torch-gated module and always run in CI.

### 4.8 Frontend browser test for the overlay geometry

**Audit ref:** §5 "Brutal truths" 5
**Files:** **new** `frontend/e2e/overlay.spec.ts`, **new** `frontend/playwright.config.ts`, `frontend/package.json`, `.github/workflows/ci.yml`

Playwright is installed and unused. One spec, run against `vite preview` with `?mock=1` (MSW, no backend): load a grounding scenario whose primary image is 2:1, resize the viewport to 1400×700 and 500×900, and assert that the overlay `<svg>`'s bounding box equals the `<img>`'s rendered content box (`object-fit: contain` rectangle) within 1 px in both cases, and that a known box's rendered centre lands where `bbox_normalised` says it should. This is the one visual code path the audit calls highest-risk and the one nothing currently tests in a real layout engine. Run it in CI as a third job with `npx playwright install --with-deps chromium`.

---

## 5. Verification checklist (what "done" looks like per phase)

**Phase 1**
- `uv run pytest tests/unit/test_corpus_script.py` green, including the `_iter_source_samples` hash-binding test.
- A `--limit 50 --dry-run` build prints `unhashed: 0` for BEN.
- `data/processed/corpus/MANIFEST.md` exists and names every JSONL.
- `evidence_qa.val.jsonl` has > 0 lines on the next real build.

**Phase 2**
- `test_two_executors_share_one_gpu_permit`, `test_delete_cancels_a_running_job_and_terminates_the_stream`, `test_a_right_number_with_the_wrong_key_is_uncited`, `test_malformed_options_are_a_400_not_a_silent_default` all green.
- `openapi.json` and `schema.d.ts` regenerated in the same commit as the `DELETE` route; contract tests updated.
- `API_CONTRACT.md` gains §4.9 (`DELETE /v1/jobs/{id}`), the `JOB_CANCELLED` code, and the `id:`/`Last-Event-ID` note in §5.

**Phase 3**
- `stream.test.ts` reattach scenarios green; `createJob` called once across a dropped stream.
- Cancel button produces a `DELETE` in the MSW log and the phase closes via the server's `error{JOB_CANCELLED}` event.
- `boxes.test.ts` green; ImageViewer draws from `BBOX_SET` when present.

**Phase 4**
- `.github/workflows/ci.yml` green on `main`; both jobs required.
- `npm run lint` runs and passes; no `[react-compare-slider]` line in `npm test` output.
- `make ci` reproduces CI locally.

---

## 6. Explicitly out of scope (tracked elsewhere)

- Assistant-only loss masking, `sample_to_chat` ↔ `build_messages` alignment, the NF4→bf16 ablation, the zero-shot vs adapted eval table, OSCD cross-resolution CD run, seed-variance runs. (Audit §2 and §7 items 1-2.)
- Anything that changes the meaning of an existing `1.0` contract field.
