# SatQuery AI — Setup Guide

Everything here has been run on the development box (Fedora, Python 3.11,
Node 22.23, AMD RX 7900 XTX on ROCm 6.4). CUDA works the same way with the
ROCm-specific steps skipped. For the air-gapped Windows laptop, follow
`DEMO_LAPTOP_RUNBOOK.md` instead; for what to do once it is running, see
`USER_GUIDE.md`.

There are three levels of setup. Start at the lowest that does what you need:

| Level | What you get | GPU needed |
|---|---|---|
| **A. Frontend only** | The full console, Maps and exports on recorded fixtures (`?mock=1`) | no |
| **B. API + frontend, no VLM** | Real ingestion, pre-flight, deterministic tools, templated answers | no |
| **C. Full stack with the fine-tuned model** | Everything, including grounded VLM answers and bounding boxes | 24 GB (bf16) — or 8 GB with the GGUF |

---

## Prerequisites

- **Python 3.11** and [`uv`](https://docs.astral.sh/uv/) (`uv.lock` is authoritative)
- **Node ≥ 22.12** (`frontend/.nvmrc`) and npm — oxlint's native binding needs it
- **GDAL/PROJ** come in as wheels with `rasterio`/`pyproj`; no system install on Linux
- For level C: a 24 GB GPU with a working PyTorch build (ROCm ≥ 6.4 or CUDA), or an
  8 GB GPU and `llama-server` for the GGUF path

> **On a ROCm box whose torch was installed by hand, never run a bare `uv sync` or a
> plain `uv run`** — both re-sync to the lockfile (which pins CUDA torch) and delete
> the ROCm stack. Use `uv run --no-sync …` or `.venv/bin/…`; the `make` targets
> already do. See `AI_HANDOFF/05_ENVIRONMENT_AND_SETUP.md` for the restore commands.
> On a fresh machine (or in CI), `uv sync --group dev` is the right first step.

```bash
git clone git@github.com:ChickenWings-0/SatQuery-AI.git
cd SatQuery-AI
```

---

## Level A — Frontend on mock data (5 minutes, no backend)

```bash
cd frontend
npm ci
npm run dev
```

Open **http://localhost:5173/?mock=1**. The flag starts a Mock Service Worker before
first render and replays recorded runs — pre-flight, streaming pipeline, evidence
views, citations, KPIs — plus recorded Nominatim and STAC replies, so the Maps page's
*Find imagery* flow works with the network off. Drop any two files onto the stage to
trigger a run.

Useful commands in `frontend/`:

```bash
npm run lint          # oxlint --deny-warnings
npm run typecheck     # tsc -b
npm run test          # vitest (356 tests)
npm run build         # production bundle into dist/ (also rewrites public/sitemap.xml — don't commit that)
npm run check:bundle  # entry chunk ≤ 180 KB
npm run test:e2e      # Playwright, offline, ?mock=1 (first: npx playwright install chromium)
npm run gen:api       # regenerate src/api/schema.d.ts from ../openapi.json
```

---

## Level B — Backend API without the VLM

### 1. Create the environment

```bash
uv sync --group dev      # fresh machine only — see the ROCm warning above
```

This installs FastAPI, rasterio, the geospatial stack and the test tooling — no torch.
The API, every deterministic tool without a checkpoint, and the whole test suite run
without it.

### 2. Check the install

```bash
make ci                  # ruff, mypy --strict, pytest (623), oxlint, tsc, vitest, build, contract
```

### 3. Configure

Settings are read by `pydantic-settings` with the `SATQUERY_` prefix, from the
environment or a `.env` in the repo root (gitignored). Start from the template:

```bash
cp .env.example .env
```

For a VLM-free run set `SATQUERY_VLM_BACKEND=none` (or use `make demo-cpu`, which does
it for you). Optional deterministic checkpoints, if you have them under
`data/checkpoints/` (gitignored):

```bash
SATQUERY_SEG_CHECKPOINT=data/checkpoints/seg/segformer-b5-loveda      # semantic_segmenter
SATQUERY_CD_CHECKPOINT=data/checkpoints/cd/levircd_resnet18.ckpt.pt   # siamese_change_detector
```

Without a checkpoint the tool reports itself not servable, drops out of the registry,
and plans degrade to spectral methods — by design, and visibly in the trace.

### 4. Run it

```bash
make demo-cpu            # API on :8000 + frontend on :5173, VLM off; Ctrl-C stops both
```

or by hand:

```bash
uv run --no-sync uvicorn satquery.api.app:app --host 127.0.0.1 --port 8000 --reload
cd frontend && npm run dev
```

- Interactive docs: http://127.0.0.1:8000/docs
- Health: `curl http://127.0.0.1:8000/v1/health`

Open **http://localhost:5173/** (no `?mock=1`). Vite proxies `/v1/*` to
`http://127.0.0.1:8000`. If you serve the built `dist/` elsewhere, set `VITE_API_BASE`
at build time. Answers are templated from the measured evidence (`templated answer`
chip, confidence capped at 0.65) because the VLM is off.

---

## Level C — Full stack with the fine-tuned model

### Option 1: bf16 base + adapter (24 GB card)

```bash
uv sync --extra vlm          # fresh CUDA machine only; a ROCm box installs torch by hand
source scripts/rocm_env.sh   # every shell that touches the GPU (masks the iGPU)
```

`.env`:

```bash
SATQUERY_VLM_BACKEND=hf
SATQUERY_VLM_ADAPTER_PATH=runs/sq-lora-v2-full/adapter
SATQUERY_VLM_DTYPE=bfloat16
SATQUERY_VLM_VRAM_BUDGET_GB=22
SATQUERY_SEG_CHECKPOINT=data/checkpoints/seg/segformer-b5-loveda
SATQUERY_CD_CHECKPOINT=data/checkpoints/cd/levircd_resnet18.ckpt.pt
SATQUERY_LOG_JSON=false
```

The adapter directory is gitignored — copy it in. The base weights download from
Hugging Face on first load (~17 GB). If you have the merged export instead, point
`SATQUERY_VLM_MODEL_PATH=models/sq-lora-v2-full-merged` at it and leave
`SATQUERY_VLM_ADAPTER_PATH` unset. **Leave `DTYPE` at `bfloat16`** — it is the only
setting that fits in 24 GB and generates correctly on this hardware
(`src/satquery/models/hf_backend.py` explains why NF4 is refused at serving time).
The backend also refuses an adapter whose `layout.json` fingerprint does not match
the loaded processor.

```bash
make demo
```

The first request loads the model (~30–60 s once cached); `/v1/health` keeps
answering during the load. Expect ~18.7 GiB peak with six views. The model unloads
after `SATQUERY_VLM_IDLE_UNLOAD_S` seconds idle so the change detector can have the card.

End-to-end check against the running stack:

```bash
make e2e                 # 3 scenarios + 2 negatives, sync vs jobs/SSE → runs/e2e/latest.json
```

### Option 2: merged Q4_K_M GGUF via llama.cpp (8 GB+, no Python ML stack)

```bash
# needs llama-server on PATH (llama.cpp with Qwen3-VL support)
scripts/serve_vlm.sh models/sq-lora-v2-full-merged-Q4_K_M.gguf \
                     models/sq-lora-v2-full-merged-mmproj-f16.gguf
```

`.env`:

```bash
SATQUERY_VLM_BACKEND=llamacpp
SATQUERY_VLM_SERVER_URL=http://127.0.0.1:8080
SATQUERY_VLM_MAX_VIEWS=3          # on an 8 GB card; 6 on 24 GB
```

Do **not** run this and the `hf` backend on the same card: the bf16 load is refused
by the VRAM guard and every `vlm_*` step degrades. On Windows use
`scripts/serve_vlm.ps1` and follow `DEMO_LAPTOP_RUNBOOK.md`.

To produce the GGUF pair from an adapter:

```bash
uv run --no-sync python scripts/merge_export.py --adapter runs/sq-lora-v2-full/adapter \
    --out models/sq-lora-v2-full-merged --gguf --llama-cpp ../llama.cpp --quant Q4_K_M
```

---

## Rebuilding the data, evaluating, retraining (optional)

The dev box's raw datasets, rendered views and corpus were deleted on 2026-09-29.
Rebuilding needs ~340 GB of disk and hours of CPU; retraining needs ~41 h of GPU.

```bash
source scripts/rocm_env.sh

# 1. Fetch sources into data/raw/ (BigEarthNet: see the hf download recipe in scripts/render_views.py)
uv run --no-sync python scripts/fetch_sources.py --source rsvqa_hr --out data/raw
uv run --no-sync python scripts/fetch_sources.py --source cdvqa    --out data/raw
#    VRSBench: unpack the published zips into data/raw/vrsbench

# 2. Render views and build the deduplicated v2 corpus (hard-fails on test-split leakage)
uv run --no-sync python scripts/render_views.py --input data/raw/ben --out data/processed/views
uv run --no-sync python scripts/render_vhr_views.py --source vrsbench
uv run --no-sync python scripts/render_vhr_views.py --source cdvqa
PRESET=full ./scripts/rebuild_corpus_v2.sh      # also renders RSVQA-HR views, then builds and gates

# 3. Benchmark the served model (100 per source; same seed → same sample_ids.json)
make eval                 # adapter
make eval-baseline        # base model, same samples

# 4. Retrain (QLoRA, NF4 base, r=16)
uv run --no-sync python scripts/preflight_train_serve_parity.py
uv run --no-sync python scripts/train_vlm.py --config configs/train/qlora_qwen3vl8b_rocm24g.yaml \
    --output-dir runs/<name> --sanity-check
uv run --no-sync python scripts/train_vlm.py --config configs/train/qlora_qwen3vl8b_rocm24g.yaml \
    --output-dir runs/<name>
```

Read `ML_PIPELINE_RECOVERY_PLAN.md` before changing anything in the training path.
DIOR-RSVG is gated: accept its terms on Hugging Face and `export HF_TOKEN=…` to include it.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| torch became a CUDA build / disappeared | a bare `uv sync` or `uv run` re-synced the lock — reinstall the ROCm wheels (`AI_HANDOFF/05`) |
| UI badge says **Offline** | API not on :8000. `curl 127.0.0.1:8000/v1/health` |
| UI badge says **Degraded** | `igpu_masked: false` or a checkpoint failed to load — `source scripts/rocm_env.sh` |
| every `vlm_*` step degrades with a VRAM message | `llama-server` holds the card; stop it or switch to `llamacpp` |
| model load refused: `layout.json` / fingerprint | wrong adapter, or a processor from elsewhere — use the adapter directory as shipped |
| answers are nonsense or one repeated token | not in bf16, or NaN weights — the backend's scanner and tripwire name which |
| `Pre-flight failed: cannot be analysed together` | a compatibility check failed — expand *Compatibility details* |
| `npm run lint` crashes loading oxlint | Node < 22.12 on `PATH` |
| fonts look like Arial | `npm run fonts:sync` |
| `make eval` can't find views | `data/processed/` is missing — rebuild it (above) |
