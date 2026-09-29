# 05 — Environment, Configuration and Commands

Human-oriented guides: `DOCS/SETUP_GUIDE.md` (levels A/B/C), `DOCS/USER_GUIDE.md`
(using the product), `DOCS/DEMO_LAPTOP_RUNBOOK.md` (the air-gapped Windows laptop).
This file is the operational cheat sheet plus everything specific to the dev box.

## The development box

| Fact | Value | Consequence |
|---|---|---|
| OS | Fedora Linux (kernel 7.x), migrated from Windows 11 | native ROCm PyTorch |
| GPU | **AMD Radeon RX 7900 XTX, 24 GB, gfx1100** | budgets assume 24 GB, guard at 22 GB |
| iGPU | Ryzen 7 7800X3D integrated graphics | ROCm enumerates it and breaks PyTorch → `HIP_VISIBLE_DEVICES=0` always; `/v1/health` reports `igpu_masked` |
| CPU / RAM | 7800X3D, 16 threads / 31 GB | dataloader workers capped at 8; the bf16 merge needs ~16 GB RAM |
| Disk | 929 GB volume, ~698 GB free after the 2026-09-29 cleanup | a full data rebuild fits again |
| Python | 3.11 via uv (`.venv`); system Python is 3.14 — unusable | always `uv run --no-sync` or `.venv/bin/…` |
| Node | **`~/.local/node-v22` = v22.23.2** (oxlint needs ≥ 22.12; old 22.11 kept as `~/.local/node-v22.11.0-backup`) | put it first on `PATH` |
| ML stack in `.venv` | torch **2.9.1+rocm6.4**, torchvision 0.24.1+rocm6.4, transformers 5.16.1, trl 1.12.0, peft 0.20.0, datasets 5.0.1, bitsandbytes, accelerate, lightning, torchgeo, tensorboard | **hand-installed, not in the lock** (below) |
| bitsandbytes | NF4 **training** works; **4-bit generation is broken** | serve bf16 |
| attention | `sdpa` only | flash-attn CK unreliable on RDNA3 |

### The hand-installed ROCm stack — read before running any `uv` command

`uv.lock` pins **CUDA** torch and none of the ROCm wheels. The `.venv` stack was
installed with `uv pip install --index-url https://download.pytorch.org/whl/rocm6.4 …`.
**A bare `uv sync` (with or without `--group dev`) or a plain `uv run` re-syncs to the
lock and deletes the whole ROCm stack** — this happened on 2026-09-12.

- Run things with `uv run --no-sync …` or `.venv/bin/python …`.
- Add a dependency with `uv add --no-sync <pkg>` then `uv pip install <pkg>`.
- For a lock-faithful throwaway env: `UV_PROJECT_ENVIRONMENT=/tmp/sq-venv uv sync --group dev`.
- To restore after an accident (wheels are usually still in `~/.cache/uv`):
  ```bash
  uv pip install --index-url https://download.pytorch.org/whl/rocm6.4 \
      "torch==2.9.1+rocm6.4" "torchvision==0.24.1+rocm6.4"
  uv pip install "transformers==5.16.1" "trl==1.12.0" "datasets==5.0.1" \
      accelerate peft bitsandbytes tensorboard lightning torchgeo
  ```
- The Makefile sets `UV ?= uv run --no-sync`, so `make` targets are safe on this box;
  CI (no ROCm) does a clean `uv sync --group dev`, which is fine there.

The Windows demo laptop has no hand-installed torch, so `uv sync --group dev` is safe
there (and it needs no ML extras at all).

## Environment variables (`SATQUERY_*`, `core/config.py::Settings`, env or `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `SATQUERY_ENVIRONMENT` | `dev` | dev / staging / prod |
| `SATQUERY_LOG_LEVEL` / `SATQUERY_LOG_JSON` | INFO / true | `LOG_JSON=false` for a human console |
| `SATQUERY_API_PREFIX` | `/v1` | |
| `SATQUERY_CORS_ALLOW_ORIGINS` | `["*"]` | |
| `SATQUERY_ARTIFACT_ROOT` / `SATQUERY_TRACE_DB` | `data/artifacts` / `data/traces.sqlite3` | |
| `SATQUERY_CACHE_MB` | 2048 | byte-bounded execution cache |
| `SATQUERY_MAX_UPLOAD_MB` / `SATQUERY_MAX_IMAGES` | 512 / 2 | |
| `SATQUERY_CITATION_POLICY` | `flag` | `strip` only for benchmark scoring |
| `SATQUERY_ALLOW_GENERIC_FALLBACK` | true | false → 422 `QUERY_UNCLASSIFIABLE` |
| `SATQUERY_VLM_BACKEND` | `auto` | `auto` / `hf` / `llamacpp` / `none` |
| `SATQUERY_VLM_DISABLED` | false | true = deterministic-only, templated answers (`make e2e-cpu` uses it; `make demo-cpu` sets `VLM_BACKEND=none`) |
| `SATQUERY_VLM_MODEL_ID` | `Qwen/Qwen3-VL-8B-Instruct` | HF id |
| `SATQUERY_VLM_MODEL_PATH` | — | local weights dir (e.g. `models/sq-lora-v2-full-merged`), overrides the HF lookup |
| `SATQUERY_VLM_ADAPTER_PATH` | — | **`runs/sq-lora-v2-full/adapter`**; unset = base model |
| `SATQUERY_VLM_SERVER_URL` | — | llama.cpp path (`http://127.0.0.1:8080`) |
| `SATQUERY_VLM_DEVICE` / `SATQUERY_VLM_DTYPE` | `rocm:0` / `bfloat16` | bf16 is the only value that fits and generates correctly |
| `SATQUERY_VLM_VRAM_BUDGET_GB` | 22.0 | guard ceiling |
| `SATQUERY_VLM_MAX_VIEWS` | 6 | **3 on the 8 GB laptop** |
| `SATQUERY_VLM_MAX_NEW_TOKENS` / `…_REQUEST_TIMEOUT_S` / `…_IDLE_UNLOAD_S` | 384 / 180 / 900 | |
| `SATQUERY_VLM_ATTN_IMPLEMENTATION` | None | `sdpa` on ROCm |
| `SATQUERY_VLM_PROMPT_VERSION` | `grounded_v1` | |
| `SATQUERY_CD_CHECKPOINT` / `SATQUERY_SEG_CHECKPOINT` | — | **not Settings fields**; read by the tools from `os.environ`, which `load_env_file()` populates from `.env` at startup |

The dev box `.env` (see `02`) serves the adapter via `hf` with both checkpoints set.
`scripts/demo_laptop/.env.laptop` is the laptop's `.env` (llama.cpp, 3 views).

## Commands

```bash
export PATH=$HOME/.local/node-v22/bin:$PATH
source scripts/rocm_env.sh                     # EVERY shell that touches the GPU

# --- Quality gates (what CI runs)
make ci                                        # ruff, mypy --strict, pytest, oxlint, tsc, vitest, build, contract
make test | make contract | make frontend-e2e  # subsets; frontend-e2e needs `npx playwright install chromium`
cd frontend && npm run build && npm run check:bundle   # entry ≤ 180 KB (restore public/sitemap.xml after)

# --- Run
make demo                                      # API :8000 + frontend :5173, from .env
make demo-cpu                                  # same with SATQUERY_VLM_BACKEND=none
uv run --no-sync uvicorn satquery.api.app:app --host 127.0.0.1 --port 8000
cd frontend && npm run dev                     # http://localhost:5173  (or /?mock=1 with no backend)
scripts/serve_vlm.sh models/sq-lora-v2-full-merged-Q4_K_M.gguf models/sq-lora-v2-full-merged-mmproj-f16.gguf

# --- Verification against a live model
make e2e                                       # scripts/e2e_parity.py → runs/e2e/latest.json
make e2e-cpu                                   # DAG/SSE/citations without weights
make eval-self-check                           # scorers vs their own references = 100 % (needs the val corpus)
make eval / make eval-baseline                 # 500-sample benchmark (needs data/processed — see 06)

# --- Export
uv run --no-sync python scripts/merge_export.py --adapter runs/sq-lora-v2-full/adapter \
    --out models/sq-lora-v2-full-merged --gguf --llama-cpp ../llama.cpp --quant Q4_K_M

# --- Data / training (all need the data pipeline re-run first)
PRESET=full ./scripts/rebuild_corpus_v2.sh
uv run --no-sync python scripts/preflight_train_serve_parity.py
uv run --no-sync python scripts/train_vlm.py --config configs/train/qlora_qwen3vl8b_rocm24g.yaml --output-dir runs/<name> --sanity-check
uv run --no-sync python scripts/train_vlm.py --config configs/train/qlora_qwen3vl8b_rocm24g.yaml --output-dir runs/<name>
uv run --no-sync python scripts/train_cd.py --dataset levircd --root data/raw --download
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `torch` suddenly CUDA / missing | someone ran `uv sync` or plain `uv run` — restore (above) |
| UI badge **Offline** | API not on :8000; Vite proxies to `127.0.0.1`, not `localhost` |
| UI badge **Degraded** | `igpu_masked: false` or a checkpoint failed to load; `source scripts/rocm_env.sh` |
| every `vlm_*` step degrades with a VRAM message | `llama-server` holds the card; stop it or switch backend |
| `ModelLoadError … layout.json` / fingerprint | adapter trained under a different layout or processor — use `runs/sq-lora-v2-full/adapter` with its own processor files |
| first `hf` request takes minutes | base model re-downloading (HF cache was cleared 2026-09-29) — or use `SATQUERY_VLM_MODEL_PATH=models/sq-lora-v2-full-merged` without an adapter |
| nonsense / repeated-token answers | not bf16, or NaN weights — the tripwire and NaN scanner will say which |
| `Pre-flight failed: cannot be analysed together` | a compatibility check failed — expand *Compatibility details* |
| `npm run lint` binding error | Node < 22.12 on `PATH` |
| Docker `npm ci` fails but local install works | the image has no `.npmrc` (strict peers); see `07` |
| fonts look like Arial | `npm run fonts:sync` |
| `make eval` / `test_inference.py` file-not-found on views | `data/processed/` was deleted — rebuild it (`06`) |
