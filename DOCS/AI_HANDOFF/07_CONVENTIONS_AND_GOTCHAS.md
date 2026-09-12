# 07 — Conventions and Hard-Won Gotchas

## Code conventions (match the surrounding code)

**Python**
- Python 3.11, `from __future__ import annotations`, full type hints; **`mypy --strict`
  must pass** (`packages=["satquery"]`, pydantic plugin, `warn_unused_ignores`).
  Untyped third-party libs (rasterio, skimage, torch, transformers, torchgeo,
  lightning, peft, trl, datasets, bitsandbytes) are declared `ignore_missing_imports`.
- `ruff` with `E F I UP B SIM N D`, line length 100, **Google docstring convention**;
  `D` is enforced on `src/` and `scripts/` (tests exempt from D100/D103).
- Every module opens with a docstring that says *what* it is and cites the spec
  section it implements (`(AGENT_POLICY_DAG.md §6)`, `(API_CONTRACT §4.6)`). Keep
  doing this.
- Comment style is **explanatory prose about why**, often long, often recording a
  measurement or a failure that motivated the code (`# 384*384 per view. The dominant
  VRAM knob… fell from 7.0 GiB to 4.4 GiB`). Match the density; do not strip these.
- Pydantic v2 everywhere for data; `StrEnum` for closed enums; frozen models for the
  contract. `Final` for module constants.
- **Heavy imports are lazy** (`torch`, `transformers`, `torchgeo`, `lightning`,
  `peft`, `trl`, `datasets`, `bitsandbytes` are imported inside function bodies) so
  the registry, the API and the test suite run on a CPU-only checkout. Never
  hoist them to module scope.
- structlog with dotted event names (`events.sink_failed`).
- Tests are hermetic and read like specifications
  (`test_an_invented_number_survives_into_the_trace_as_a_flagged_span`). Name new
  tests that way. Synthetic GeoTIFFs come from `scripts/make_synthetic_fixtures.py`.
  Golden checks compare against closed-form references, not committed PNGs.
- Optional deps are extras (`vlm`, `vlm-train`, `cd`); the base install must stay
  torch-free.

**TypeScript / frontend**
- React 19 function components, TS 5.9 strict, Tailwind v4 idiom only: tokens in
  `@theme` in `styles/theme.css`, `@custom-variant wide/desk`, component classes in
  `@layer components`, **no** `tailwind.config.*`, no `@apply`, no `postcss.config`.
- Types come from generated `src/api/schema.d.ts`; `src/api/types.ts` re-exports the
  named ones. Never hand-write a wire type.
- Zustand stores are kept apart by responsibility (`job` = pure projection of the
  server stream, `ui` = client-only, `focus` = cross-column focus). Reducers stay
  pure and DOM-free so they can be tested on recorded fixtures.
- Every new hue needs fill / text / on-fill tokens with computed ratios and will be
  checked by `contrast.test.ts`. `--color-warn` is reserved for uncited spans;
  `--color-evidence` for citations; `--color-accent-cool` means "selected".
- Exactly two infinite animations are allowed (RUNNING pulse, health glow) —
  `motion.test.ts` pins this.
- Gates: `npm run typecheck`, `npm run test`, `npm run build`. Lint is broken on
  this box (oxlint binding).

**Contract discipline**
- `DOCS/API_CONTRACT.md` is the prose contract. Change it → change the schemas →
  `scripts/export_openapi.py` → `npm run gen:api`. Both generated files must be
  byte-identical to what is committed; contract tests on both sides lock the enums
  and the SSE union. Bumping `SCHEMA_VERSION` is a big deal.
- Every enum is **closed**; a client receiving an unknown value treats it as a
  contract violation.
- Error envelope is the §6 taxonomy (`code`, `http_status`, `message`, `hint`,
  `ref`, `trace_id`). `message` is user-safe; never leak paths or stack traces.

**Git**
- Single `main` branch; large phase commits with descriptive subjects. Attribution
  trailers for AI-assisted commits are given in the session (see system reminder
  in the session, if any). Do not commit `data/`, `runs/`, logs or `.env`.

## One-source-of-truth rules (duplicating any of these is a defect)

| Thing | The one place | Consumers |
|---|---|---|
| view label string | `render/view_labels.py::label_for_view` | corpus builder, prompt builder, frontend displays it verbatim |
| box serialisation/parsing | `models/prompts/box_format.py` | corpus, hf_backend, text_grounding; `frontend/src/thread/bbox.ts` is a *tested port* |
| number formatting | `evidence/citation_validator.format_number` | corpus answers, templates |
| band aliases | `configs/band_aliases.yaml` | `ingest/bands.py`, capability matching, renderer |
| class vocabulary | `configs/class_vocabulary.yaml` | slot normalisation, segmenter mapping, scalar name patterns |
| index maths | `render/indices.py` | renderer, `spectral_index_analyzer`, `sar_backscatter_analyzer`, `render_views.py` |
| routing | `configs/policy_table.yaml` | planner only; tools are added via `registry.yaml` + `tools/catalog.py`, never by touching the planner |
| the system prompt / FactSheet constraint | `models/prompts/templates.py` (`grounded_v1`) | builder; corpus (imports `build_system_prompt`) |

## Gotchas — things that silently produce wrong results

1. **iGPU.** Without `HIP_VISIBLE_DEVICES=0` (and `ROCR_VISIBLE_DEVICES=0`) ROCm
   sees two agents and torch misbehaves. `scripts/rocm_env.sh` sets both; the training
   YAMLs also carry them; `/v1/health.igpu_masked` reports it.
2. **4-bit generation is broken on gfx1100 / ROCm 6.4 / torch 2.9.1 / bitsandbytes.**
   Base weights answer "Name three primary colours" with a run of close-parens under
   NF4 and correctly under bf16. 4-bit *training* is fine. Serve bf16. Revisit only
   against a bitsandbytes build that demonstrably generates.
3. **`skip_special_tokens=True` deletes `<|box_start|>`.** `hf_backend._strip_chat_tokens`
   removes chat scaffolding by name instead. Keep it that way or grounding
   silently degrades to a bare-digit heuristic.
4. **Fine-tuned checkpoints do not reliably emit EOS.** Stop sequences are applied
   after generation in both backends.
5. **Per-image stretch on index views destroys the physics.** NDVI/NDWI/NDBI on a
   fixed [−1, +1]; SAR on fixed dB clips; PNG not JPEG for those. Cartosat has no SWIR
   → NDBI/SWIR are *unavailable*, `ndbi_mean: null`; never substitute a band.
6. **View labels must be byte-identical between training and serving** (see rules
   above). All tests pass if they drift; only the adaptation dies.
7. **Naming images only in the system prompt "silently breaks"** the binding
   between label and image (per `hf_backend.build_messages`' docstring). Labels go
   as text parts directly before each image. The current adapter was trained the
   other way (`06`).
8. **RSVQA-HR train JSON contains every question**; filter on `active: true` or leak
   330,324 test rows into training. VRSBench's HF loader crashes 20,264 rows in;
   use `local_sources`. CDVQA is WebDataset; unpack first.
9. **Dedup by sample vs by image.** Pass `image_key` to `Deduplicator.check` or
   multi-annotation sources collapse to nothing.
10. **`hashes_for` needs `image_path`/`pre_path`** — BEN rows currently get none, so
    they are never hashed. Untested seam.
11. **`SATQUERY_CD_CHECKPOINT` / `SATQUERY_SEG_CHECKPOINT` are not Settings fields.**
    They work from `.env` only because `load_env_file()` copies `SATQUERY_*` into
    `os.environ` at startup. A real env var always wins over the file.
12. **`/v1/health` must never load the model** — use `current_backend()`, not
    `get_backend()`, in anything that polls.
13. **Do not run `llama-server` and the `hf` backend on the same card.** The VRAM
    guard refuses the bf16 load and every `vlm_*` step degrades.
14. **Vite proxy targets `127.0.0.1`, not `localhost`.** Node ≥ 18 resolves
    `localhost` to `::1` first; uvicorn binds IPv4; every proxied call would
    ECONNREFUSED.
15. **`scripts/export_openapi.py` treats any argv as an output path** — `--help`
    writes a file named `--help`. Delete it if you trip this.
16. **Citation `claim` strings are substrings, not offsets.** `annotate.ts` must match
    longest-first with digit-boundary rules or `"1"` lands inside `"350,106.12 m2"`.
17. **Box overlay must be sized to the raster's `object-fit: contain` rectangle**, not
    its container, or every box misses on a wide window.
18. **Grounding answers must have boxes stripped before citation validation**
    (`box_format.strip_boxes`) — coordinates are not claims. Before this fix every
    grounding run was capped for "uncited" numbers it was told to emit.
19. **A VLM step never raises tool confidence.** VLM steps are excluded from
    `tool_mean`; a single uncorroborated tool scores 0.75.
20. **`ingest()` is synchronous rasterio + phase correlation** called inside async
    handlers; a big GeoTIFF pair stalls SSE heartbeats. Known, unfixed (`08`).
21. **The executor's GPU semaphore is per `DagExecutor` (per request)**, not
    per process. Two concurrent jobs can both load SegFormer/CD/DOFA on top of the VLM.
22. **Master.md §2 environment facts are historical** (Windows 11, Python 3.14 only,
    "repo is empty"). The box is Fedora + uv + 3.11 now. Do not re-run Phase −1.
23. **`repo documentation/README.md` claims "65k samples" and "19-hour single-epoch
    run" over the full corpus.** The run was 18.5k BEN-only. Fix the claim, not the
    handoff.
24. **fp16 overflows in the deep ResNet stages on gfx1100** — CD training uses
    `bf16-mixed`; the VLM uses bf16 for the same reason in the vision tower.
25. **`max_pixels` side must be a multiple of 32** (Qwen3-VL merges patches 2×2):
    384 / 416 / 448 are the neighbours; 416 lands exactly on the VRAM line.
