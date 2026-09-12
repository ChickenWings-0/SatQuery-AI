# 05 — Environment, Configuration and Commands

The long-form guide is `repo documentation/SETUP_GUIDE.md` (three levels: A frontend
mock only, B API without VLM, C full stack). This file is the operational cheat
sheet plus everything that is *specific to this machine*.

## The development box (measured, `Master.md §2`)

| Fact | Value | Consequence |
|---|---|---|
| OS | Fedora Linux (kernel 7.1.x), migrated from Windows 11 | native ROCm PyTorch works |
| GPU | **AMD Radeon RX 7900 XTX, 24 GB, gfx1100** | ROCm 6.4 target; every VRAM budget assumes 24 GB, guard at 22 GB |
| iGPU | Ryzen 7 7800X3D integrated graphics | **ROCm enumerates it as a second agent and breaks PyTorch** → `HIP_VISIBLE_DEVICES=0` always; `/v1/health` reports `igpu_masked` |
| CPU / RAM | 7800X3D, 16 threads / 31 GB | dataloader workers capped at 8 |
| Disk | ~377 GB free at project start | full BigEarthNet does not fit; stratified subsets |
| Python | 3.11 via uv (`.venv`) — system Python is 3.14, unusable for torch/rasterio wheels | never `pip` into system Python |
| Node | v22.11.0 | frontend |
| torch | 2.9.1+rocm6.4; transformers 5.16.1; trl 1.12.0; datasets 5.0.1 | (from `runs/*/README.md` model cards) |
| bitsandbytes | works for NF4 **training**, **broken for 4-bit generation** on this stack | serve bf16 |
| attention | `sdpa` only | flash-attn CK unreliable on RDNA3 |

## Environment variables (`SATQUERY_*`, read by `core/config.py::Settings` from env or `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `SATQUERY_ENVIRONMENT` | `dev` | dev / staging / prod |
| `SATQUERY_LOG_LEVEL` / `SATQUERY_LOG_JSON` | INFO / true | structlog; `LOG_JSON=false` for a human console |
| `SATQUERY_API_PREFIX` | `/v1` | |
| `SATQUERY_CORS_ALLOW_ORIGINS` | `["*"]` | |
| `SATQUERY_ARTIFACT_ROOT` | `data/artifacts` | content-addressed store |
| `SATQUERY_TRACE_DB` | `data/traces.sqlite3` | |
| `SATQUERY_MAX_UPLOAD_MB` / `SATQUERY_MAX_IMAGES` | 512 / 2 | |
| `SATQUERY_CITATION_POLICY` | `flag` | `strip` only for benchmark scoring |
| `SATQUERY_ALLOW_GENERIC_FALLBACK` | true | false → 422 `QUERY_UNCLASSIFIABLE` |
| `SATQUERY_VLM_BACKEND` | `auto` | `auto` / `hf` / `llamacpp` / `none` — intent; `available_backend()` decides |
| `SATQUERY_VLM_DISABLED` | false | true = deterministic-only baseline, templated answers |
| `SATQUERY_VLM_MODEL_ID` | `Qwen/Qwen3-VL-8B-Instruct` | HF id |
| `SATQUERY_VLM_MODEL_PATH` | — | local weights dir, overrides the HF cache lookup |
| `SATQUERY_VLM_ADAPTER_PATH` | — | **set to `runs/full-epoch-v1/adapter` to serve the fine-tune**; unset = stock Qwen |
| `SATQUERY_VLM_GGUF_PATH` / `SATQUERY_VLM_SERVER_URL` | — | llama.cpp path |
| `SATQUERY_VLM_DEVICE` | `rocm:0` | contract device string (maps to torch `cuda:0`) |
| `SATQUERY_VLM_DTYPE` | `bfloat16` | **the only value that fits and generates correctly** |
| `SATQUERY_VLM_VRAM_BUDGET_GB` | 22.0 | hard ceiling for the guard |
| `SATQUERY_VLM_MAX_NEW_TOKENS` | 384 | |
| `SATQUERY_VLM_IDLE_UNLOAD_S` | 900 | 0 keeps weights resident |
| `SATQUERY_VLM_REQUEST_TIMEOUT_S` | 180 | |
| `SATQUERY_VLM_ATTN_IMPLEMENTATION` | None | `sdpa` recommended on ROCm |
| `SATQUERY_VLM_PROMPT_VERSION` | `grounded_v1` | key into `models/prompts/templates.TEMPLATES` |
| `SATQUERY_CD_CHECKPOINT` | — | **not a Settings field**; read directly by `tools/change_detect.py`. `.env` is still honoured because `load_env_file()` copies `SATQUERY_*` into `os.environ` at startup |
| `SATQUERY_SEG_CHECKPOINT` | — | same, for `tools/semantic_segmenter.py` |

Without a configured checkpoint the tool reports itself not servable, drops out of
the registry, and plans degrade to the declared fallback — by design.

**Current `.env` on the box:** `VLM_BACKEND=llamacpp`, `VLM_SERVER_URL=http://127.0.0.1:8080`,
`SEG_CHECKPOINT=data/checkpoints/seg/segformer-b5-loveda`. No `CD_CHECKPOINT`, no
adapter path. To serve the fine-tuned adapter instead:

```bash
cat > .env <<'EOF2'
SATQUERY_VLM_BACKEND=hf
SATQUERY_VLM_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
SATQUERY_VLM_ADAPTER_PATH=runs/full-epoch-v1/adapter
SATQUERY_VLM_DTYPE=bfloat16
SATQUERY_VLM_VRAM_BUDGET_GB=22
SATQUERY_SEG_CHECKPOINT=data/checkpoints/seg/segformer-b5-loveda
SATQUERY_CD_CHECKPOINT=data/checkpoints/cd/levircd_resnet18.ckpt.pt
SATQUERY_LOG_JSON=false
EOF2
```
and make sure no `llama-server` is holding the card.

## Commands

```bash
# --- Python env
uv sync                       # base + dev (no torch) — enough for API, tools without checkpoints, all tests
uv sync --extra vlm           # + torch/transformers/accelerate (serving)
uv sync --extra vlm-train     # + peft/trl/bitsandbytes/datasets/tensorboard (training)
uv sync --extra cd            # + lightning/torchgeo/torchvision (CD training, DOFA)
source scripts/rocm_env.sh    # EVERY shell that touches the GPU

# --- Quality gates (all must pass; CI does not exist yet)
uv run pytest -q              # 503 tests, ~4 min
uv run ruff check .
uv run mypy                   # strict; packages=["satquery"]
uv run python scripts/export_openapi.py && git diff --exit-code openapi.json
cd frontend && npm run typecheck && npm run test && npm run build && npm run gen:api && git diff --exit-code src/api/schema.d.ts

# --- Run
uv run uvicorn satquery.api.app:app --host 127.0.0.1 --port 8000 --reload   # API; /docs, /v1/health
cd frontend && npm install && npm run dev                                   # http://localhost:5173 (proxied) or /?mock=1 (no backend)
scripts/serve_vlm.sh <model.gguf> <mmproj.gguf>                             # llama.cpp path on :8080
uv run python scripts/test_inference.py --adapter runs/full-epoch-v1/adapter  # adapter smoke test

# --- Data / training
uv run python scripts/fetch_sources.py --source rsvqa_hr --out data/raw
uv run python scripts/fetch_sources.py --source cdvqa    --out data/raw
uv run python scripts/render_views.py --input data/raw/ben --out data/processed/views --split train
uv run python scripts/render_vhr_views.py ...
uv run python scripts/build_corpus.py --out data/processed/corpus [--limit N] [--require-views]
uv run python scripts/train_vlm.py --config configs/train/qlora_qwen3vl8b_rocm24g.yaml --output-dir runs/<name> --sanity-check
uv run python scripts/train_vlm.py --config configs/train/qlora_qwen3vl8b_rocm24g.yaml --output-dir runs/<name>
uv run python scripts/merge_export.py --adapter runs/<name>/adapter ... [--dry-run]
uv run python scripts/train_cd.py --dataset levircd [--download] ...
./run_overnight.sh [--stage N]   ;  tail -f logs/overnight-*.log
```

## Troubleshooting (from `SETUP_GUIDE.md`, verified)

| Symptom | Cause / fix |
|---|---|
| UI badge **Offline** | API not on :8000; Vite proxies to `127.0.0.1` (not `localhost` — Node resolves `::1` first, uvicorn binds IPv4) |
| UI badge **Degraded** | `igpu_masked: false` or a checkpoint failed to load; `source scripts/rocm_env.sh` |
| every `vlm_*` step degrades with a VRAM message | another process (usually `llama-server`) holds the card; stop it or switch backend |
| nonsense answers on the `hf` backend | not in bf16; 4-bit generation is broken on ROCm 6.4 |
| `Pre-flight failed: cannot be analysed together` | compatibility FAIL — usually CRS or footprint; expand *Compatibility details* |
| `npm run lint` crashes | oxlint native binding missing on this platform; `typecheck` + `test` are the gates |
| fonts look like Arial | `npm run fonts:sync` |
| training dies in the processor before step 1 | view paths in the corpus point at nothing; run the render pass, or `patch_dummy_images.py` for a *sanity check only* |
| Python 3.14 picked up | use `uv run`; `.python-version` pins 3.11 |
| OOM during training | knobs in order: views/sample → `max_pixels` → `max_seq_len` → Profile B (`DATA_ADAPTATION_PLAN §6.3`) |
