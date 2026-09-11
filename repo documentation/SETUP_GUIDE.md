# SatQuery AI — Setup Guide

Everything here has been run on the development box (Fedora, Python 3.11,
Node 22, AMD RX 7900-class GPU on ROCm 6.4). CUDA works the same way with the
ROCm-specific step skipped.

There are three levels of setup. Start at the lowest that does what you need:

| Level | What you get | GPU needed |
|---|---|---|
| **A. Frontend only** | The full console on recorded fixtures (`?mock=1`) | no |
| **B. API + frontend, no VLM** | Real ingestion, pre-flight, deterministic tools, templated answers | no |
| **C. Full stack with the fine-tuned model** | Everything, including grounded VLM answers and bounding boxes | 24 GB |

---

## Prerequisites

- **Python 3.11+** and [`uv`](https://docs.astral.sh/uv/) (the project uses
  `uv.lock`; plain `pip` works but the lock is authoritative)
- **Node 18+** (22 recommended) and npm
- **GDAL/PROJ** are pulled in as wheels by `rasterio`/`pyproj`; no system
  install is needed on Linux
- For level C: a 24 GB GPU with a working PyTorch build for it
  (ROCm ≥ 6.4 or CUDA), and the adapter directory from the training run

Clone and enter the repo:

```bash
git clone <your-fork-url> SatQuery-AI
cd SatQuery-AI
```

---

## Level A — Frontend on mock data (5 minutes, no backend)

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173/?mock=1**. The `?mock=1` flag starts a Mock
Service Worker before first render and replays a recorded bi-temporal run —
pre-flight, streaming pipeline, evidence views, citations, KPIs — with no
server at all. Drop any two files onto the stage to trigger it.

Useful commands in `frontend/`:

```bash
npm run typecheck   # tsc -b --noEmit
npm run test        # vitest, 194 tests
npm run build       # production bundle into dist/
npm run gen:api     # regenerate src/api/schema.d.ts from ../openapi.json
```

---

## Level B — Backend API without the VLM

### 1. Create the environment

```bash
uv sync            # base deps + dev group (pytest, ruff, mypy)
```

This installs FastAPI, rasterio, the geospatial stack and the test tooling.
It does **not** install torch or transformers; the API, every deterministic
tool that has no checkpoint, and the whole test suite run without them.

### 2. Check the install

```bash
uv run pytest -q                 # ~450 tests, no GPU
uv run ruff check . && uv run mypy
```

### 3. Configure

Settings are read by `pydantic-settings` with the `SATQUERY_` prefix, from
the environment or from a `.env` file in the repo root (gitignored). For a
VLM-free run:

```bash
cat > .env <<'EOF'
SATQUERY_VLM_DISABLED=true
SATQUERY_LOG_JSON=false
EOF
```

Optional deterministic model checkpoints, if you have them under
`data/checkpoints/` (also gitignored):

```bash
SATQUERY_SEG_CHECKPOINT=data/checkpoints/segformer-loveda      # semantic_segmenter
SATQUERY_CD_CHECKPOINT=data/checkpoints/siamese-cd.ckpt         # change_detect
```

Without a configured checkpoint the corresponding tool reports itself as not
servable, drops out of the registry, and plans degrade to spectral methods —
by design, and visibly in the trace.

### 4. Run the API

```bash
uv run uvicorn satquery.api.app:app --host 127.0.0.1 --port 8000 --reload
```

- Interactive docs: http://127.0.0.1:8000/docs
- Health: `curl http://127.0.0.1:8000/v1/health`
- Registry: `curl http://127.0.0.1:8000/v1/registry`

### 5. Run the frontend against it

In a second terminal:

```bash
cd frontend && npm run dev
```

Open **http://localhost:5173/** (no `?mock=1`). Vite's dev server proxies
`/v1/*` to `http://127.0.0.1:8000`, so no `VITE_API_BASE` is needed. If you
serve the built `dist/` from elsewhere, set `VITE_API_BASE` at build time to
the API origin.

The top-bar badge should read **System Ready** within a few seconds. Drop one
or two GeoTIFF/PNG/JPEG scenes, ask a question, and watch the pipeline.
Answers will be templated from the measured evidence (`templated answer`
chip, confidence capped at 0.65) because the VLM is disabled.

---

## Level C — Full stack with the fine-tuned adapter

### 1. Install the serving extra

```bash
uv sync --extra vlm          # torch, transformers, accelerate
```

On ROCm, source the device environment first in every shell that touches the
GPU (it pins the visible device and the allocator behaviour):

```bash
source scripts/rocm_env.sh
```

Make sure the integrated GPU is masked (`HIP_VISIBLE_DEVICES=0`, set by that
script). `/v1/health` reports `igpu_masked`; the UI shows **Degraded** if it is
false, because the 8B model would otherwise share VRAM with the desktop
compositor.

### 2. Point the backend at the adapter

The adapter from the training run lives under `runs/<run-name>/adapter/`
(gitignored — copy it in, or mount it). Then:

```bash
cat > .env <<'EOF'
SATQUERY_VLM_BACKEND=hf
SATQUERY_VLM_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
SATQUERY_VLM_ADAPTER_PATH=runs/full-epoch-v1/adapter
SATQUERY_VLM_DTYPE=bfloat16
SATQUERY_VLM_VRAM_BUDGET_GB=22
SATQUERY_LOG_JSON=false
EOF
```

The base weights download from Hugging Face on first load (~16 GB) into the
standard HF cache; set `SATQUERY_VLM_MODEL_PATH` instead of `MODEL_ID` to use
a local copy. **Leave `DTYPE` at `bfloat16`** — it is the only setting that
both fits in 24 GB and generates correctly on this hardware; see
`src/satquery/models/hf_backend.py` for why NF4 is refused at serving time.

### 3. Run it

```bash
uv run uvicorn satquery.api.app:app --host 127.0.0.1 --port 8000
cd frontend && npm run dev
```

The first request loads the model (~30–60 s); `/v1/health` continues to answer
during the load because it never triggers one. Expect ~18.7 GiB peak with six
evidence views. The model unloads after `SATQUERY_VLM_IDLE_UNLOAD_S` seconds
idle so the change detector can have the card back.

Quick smoke test of the model path without the UI:

```bash
uv run python scripts/test_inference.py --adapter runs/full-epoch-v1/adapter
```

### Alternative: the llama.cpp path (lighter, no adapter)

If ROCm is broken or memory is tight, serve a Q4 GGUF instead:

```bash
# needs llama-server on PATH (build llama.cpp with -DGGML_HIP=ON or -DGGML_VULKAN=ON)
scripts/serve_vlm.sh models/Qwen3-VL-8B-Instruct-Q4_K_M.gguf models/Qwen3-VL-8B-Instruct-mmproj-f16.gguf
```

and in `.env`:

```bash
SATQUERY_VLM_BACKEND=llamacpp
SATQUERY_VLM_SERVER_URL=http://127.0.0.1:8080
```

~6 GB of VRAM, starts in seconds, base model only. Do **not** also set
`SATQUERY_VLM_BACKEND=hf` on the same card: the bf16 load is refused by the
VRAM guard when llama-server already holds the budget, and every `vlm_*` step
would degrade.

---

## Reproducing the training run (optional)

Requires the training extra, ~340 GB of disk for sources and rendered views,
and a day of GPU time.

```bash
uv sync --extra vlm-train
source scripts/rocm_env.sh

# 1. Fetch the corpus sources into data/raw/
uv run python scripts/fetch_sources.py --source rsvqa_hr --out data/raw
uv run python scripts/fetch_sources.py --source cdvqa    --out data/raw
# (BigEarthNet-v2 and VRSBench stream from the Hub)

# 2. Pre-render the frozen view catalogue at 448 px
uv run python scripts/render_views.py --input data/raw/ben --out data/processed/views --split train

# 3. Build the deduplicated 65k corpus (hard-fails on test-split leakage)
uv run python scripts/build_corpus.py --out data/processed/corpus

# 4. Train (QLoRA, NF4 base, r=16) — ~19 h on a 24 GB card
uv run python scripts/train_vlm.py --config configs/train/qlora_qwen3vl8b_rocm24g.yaml \
    --output-dir runs/full-epoch-v1
```

`run_overnight.sh` chains all of this with a throughput probe and disk-space
checks; `tail -f logs/overnight-*.log` to watch it. Everything it writes goes
under `data/`, `runs/` and `logs/`, all gitignored.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| UI badge says **Offline** | API not on :8000, or Vite proxy target unreachable. Check `curl 127.0.0.1:8000/v1/health`. |
| UI badge says **Degraded** | `igpu_masked: false` or a tool checkpoint failed to load. Hover the badge for details; `source scripts/rocm_env.sh`. |
| Every `vlm_*` step degrades with a VRAM message | Another process (usually `llama-server`) holds the card. Stop it, or switch `SATQUERY_VLM_BACKEND=llamacpp`. |
| Answers are nonsense with the `hf` backend | You are not in bf16. `SATQUERY_VLM_DTYPE` must be `bfloat16`; 4-bit generation is broken on ROCm 6.4. |
| `Pre-flight failed: cannot be analysed together` | Compatibility check refused the pair. Expand *Compatibility details* on the stage — usually CRS or footprint. |
| `npm run lint` crashes loading oxlint | Known native-binding issue on this platform; `typecheck` and `test` are the gates. |
| Fonts look like Arial | `public/fonts/*.woff2` missing — `npm run fonts:sync`. |
