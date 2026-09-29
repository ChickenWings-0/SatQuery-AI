# 07 — Conventions and Hard-Won Gotchas

## Code conventions (match the surrounding code)

**Python**
- Python 3.11, `from __future__ import annotations`, full type hints; **`mypy --strict`
  must pass** (`packages=["satquery"]`, pydantic plugin). Untyped third-party libs
  are `ignore_missing_imports`.
- `ruff check` with `E F I UP B SIM N D`, line length 100, Google docstrings; `D`
  enforced on `src/` and `scripts/` (and `frontend/scripts/*.py`, which ruff also
  scans). **`ruff format` is not a gate** — ~58 files drift from it; do not
  bulk-reformat unless asked.
- Every module opens with a docstring saying *what* it is and citing the spec section
  it implements (`(AGENT_POLICY_DAG.md §6)`). Keep doing this.
- Comments are **explanatory prose about why**, often recording a measurement or a
  failure. Match the density; do not strip them.
- Pydantic v2; `StrEnum` for closed enums; frozen models for the contract; `Final`
  for module constants.
- **Heavy imports are lazy** (torch, transformers, torchgeo, lightning, peft, trl,
  datasets, bitsandbytes inside function bodies) so the API and tests run torch-free.
- structlog with dotted event names.
- Tests are hermetic and read like specifications
  (`test_an_invented_number_survives_into_the_trace_as_a_flagged_span`). Synthetic
  GeoTIFFs from `scripts/make_synthetic_fixtures.py`; golden checks against
  closed-form references, not committed PNGs.
- Optional deps are extras (`vlm`, `vlm-train`, `cd`); the base install stays torch-free.

**TypeScript / frontend**
- React 19.2 function components, TS 5.9 strict, Tailwind v4 idiom only: tokens in
  `@theme` in `styles/theme.css`, `@custom-variant`, component classes in
  `@layer components`; no `tailwind.config.*`, no `@apply`, no `postcss.config`.
- Wire types come only from generated `src/api/schema.d.ts` (re-exported by `api/types.ts`).
- Zustand stores split by responsibility; reducers pure and DOM-free.
- Every new hue needs fill/text/on-fill tokens with computed ratios (`contrast.test.ts`).
  `--color-warn` is reserved for uncited spans; `--color-evidence` for citations.
- Heavy dependencies (three, maplibre-gl, pdf-lib, marching-squares, IndexedDB) load
  behind `import()`; `bundle-boundaries.test.ts` and `check:bundle` enforce the 180 KB entry.
- Every network feature sits behind `state/settings` "online features" (off by
  default), fails to a toast within 5 s (`geo/deadline.ts`), and has a recorded
  fixture for `?mock=1`.
- Gates: `npm run lint` (oxlint --deny-warnings), `typecheck`, `test`, `build`.

**Contract discipline**
- `DOCS/API_CONTRACT.md` is the prose contract (it wins over `openapi.json`). Change it
  → schemas → `scripts/export_openapi.py` → `npm run gen:api`; `make contract` must be
  green. Only additive, optional, defaulted changes within `1.0`.
- Every enum is closed. Error envelope is the §6 taxonomy; `message` is user-safe.

**Git**
- One branch, `main`, enforced by `scripts/git/main_only.sh` hooks. Small commits.
  Never commit `data/`, `runs/` (except `runs/eval/*/results.*`), `models/`, logs,
  `.env`, `.claude/`, or a `public/sitemap.xml` that only changed its `lastmod`.

## One-source-of-truth rules (duplicating any of these is a defect)

| Thing | The one place | Consumers |
|---|---|---|
| view label string | `render/view_labels.py::label_for_view` | corpus, prompt builder, frontend displays verbatim |
| user-turn layout | `models/prompts/layout.py` (`label-before-image/v1`) | `training/vlm/qlora.sample_to_chat`, `hf_backend`, `llamacpp_client`; fingerprint in `adapter/layout.json` |
| box serialisation/parsing | `models/prompts/box_format.py` | corpus, backends, `text_grounding`, eval scorers; `frontend/src/thread/bbox.ts` is a tested port |
| number formatting | `evidence/citation_validator.format_number` | corpus answers, templates |
| band aliases | `configs/band_aliases.yaml` | `ingest/bands.py`, capability matching, renderer |
| class vocabulary | `configs/class_vocabulary.yaml` | slot normalisation, segmenter mapping |
| index maths | `render/indices.py` | renderer, analysers, `render_views.py` |
| routing | `configs/policy_table.yaml` | planner only; new tools via `registry.yaml` + `tools/catalog.py` |
| system prompt | `models/prompts/templates.py` (`grounded_v1`) | builder; corpus |
| KPI selection | `frontend/src/kpi/registry.ts` | KPI cards and the SITREP PDF (never a number the UI did not show) |

## Gotchas — things that silently produce wrong results (or delete things)

1. **A bare `uv sync` or plain `uv run` deletes the ROCm torch stack** (`05`). Use
   `uv run --no-sync` / `.venv/bin/…`.
2. **iGPU.** Without `HIP_VISIBLE_DEVICES=0` ROCm sees two agents; `rocm_env.sh`
   sets it, `/v1/health.igpu_masked` reports it.
3. **4-bit generation is broken** on gfx1100 / ROCm 6.4 / torch 2.9.1 / bitsandbytes.
   4-bit *training* is fine. Serve bf16.
4. **`skip_special_tokens=True` deletes `<|box_start|>`** — `_strip_chat_tokens`
   removes scaffolding by name.
5. **Fine-tuned checkpoints do not reliably emit EOS** — stop sequences post-generation.
6. **Per-image stretch on index views destroys the physics.** Fixed domains, PNG.
   Cartosat has no SWIR → NDBI *unavailable*, never substituted.
7. **Labels must be text parts before each image, identically in training and
   serving.** v1 was trained otherwise and learned nothing useful from its labels;
   `layout.py` + the fingerprint check now make drift a load error.
8. **trl vision datasets:** `assistant_only_loss` raises; use prompt-completion +
   `completion_only_loss` (`06`).
9. **RSVQA-HR train JSON contains every question** — filter `active: true`.
   VRSBench HF loader crashes; CDVQA must be unpacked.
10. **Dedup by sample vs by image** — pass `image_key` or multi-annotation sources collapse.
11. **`SATQUERY_CD_CHECKPOINT` / `SATQUERY_SEG_CHECKPOINT` are not Settings fields**;
    `create_app()` copies `.env` into `os.environ`, so **the local `.env` leaks into
    pytest** in test-order-dependent ways. Tests touching `find_checkpoint()` must
    `monkeypatch.delenv`. CI has no `.env`.
12. **`/v1/health` must never load the model** — `current_backend()`, not `get_backend()`.
13. **Never run `llama-server` and the `hf` backend on one card.**
14. **Vite proxy targets `127.0.0.1`, not `localhost`** (Node resolves `::1` first).
15. **Citation `claim` strings are substrings, not offsets** — longest-first,
    digit-boundary matching in `annotate.ts`.
16. **Box overlay is sized to the raster's `object-fit: contain` rectangle.**
17. **Strip boxes before citation validation** — coordinates are not claims.
18. **A VLM step never raises tool confidence.**
19. **`max_pixels` side must be a multiple of 32** (384/416/448); 144 tokens per
    448 px view is part of the layout fingerprint.
20. **fp16 overflows in deep ResNet stages on gfx1100** — CD trains `bf16-mixed`.
21. **`Master.md §2` environment facts are historical** (Windows, Python 3.14). Do not
    re-run Phase −1.
22. **`asyncio.Semaphore` binds to the first loop that contends it** — `DeviceGates`
    are created in the app lifespan, not at module scope (pytest runs a loop per test).
23. **`*.tif` and `*.jsonl` are gitignored globally** — the two S1 RTC fixtures under
    `frontend/public/samples/stac/fetch/` and `configs/*.jsonl` are negated
    explicitly; new fixtures need their own negation *and* an explicit `git add`.
24. **`vite preview` SPA-falls-back unknown paths to `index.html` with 200** — asset
    presence checks must also reject `text/html`.
25. **Playwright wipes `frontend/test-results/` on every run** — never park files
    there. Never `pkill -f vite` (it matches the agent's own shell).
26. **npm peer modes differ:** `frontend/.npmrc` sets `legacy-peer-deps=true` for
    local and CI installs, but `frontend/Dockerfile` copies no `.npmrc`, so Docker's
    `npm ci` is strict. Reproduce with `npm ci --dry-run` on a copy without `.npmrc`;
    don't rewrite the lock in strict mode; pin versions under legacy instead.
27. **React must stay on 19.2.x** — `@react-three/fiber@9.7.0` peer-requires `<19.3`.
28. **`npm run build` rewrites tracked `public/sitemap.xml`** with today's UTC
    `lastmod`; restore it after local builds.
29. **The bf16 HF backend needs the base weights** — after the 2026-09-29 cache wipe
    the first load re-downloads 17 GB unless `SATQUERY_VLM_MODEL_PATH` points at
    `models/sq-lora-v2-full-merged` (then leave the adapter unset).
