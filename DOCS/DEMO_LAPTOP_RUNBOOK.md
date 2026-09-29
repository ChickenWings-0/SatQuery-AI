# SatQuery AI — air-gapped demo laptop runbook

*Windows 11 · RTX 4070 Laptop (8 GB VRAM) · no network required after step 0.*

This is the path the judges see if the training box is not in the room. It
serves the **merged, Q4_K_M-quantised** `sq-lora-v2-full` model through
`llama-server` (CUDA, no Python ML stack), the FastAPI backend against it,
and the built frontend in front of both. Everything comes off one USB stick.

## The 10 lines

```powershell
# 1  Verify the copy (from the stick's root; compare against SHA256SUMS)
Get-Content models\SHA256SUMS | ForEach-Object { $h,$f = $_ -split '\s+'; if ((Get-FileHash "models\$f").Hash -ne $h.ToUpper()) { "BAD $f" } }
# 2  Unzip llama.cpp (CUDA build + cudart) next to the repo -> ..\llama.cpp\llama-server.exe
Expand-Archive llama-*-bin-win-cuda-12.4-x64.zip ..\llama.cpp; Expand-Archive cudart-llama-bin-win-cuda-12.4-x64.zip ..\llama.cpp
# 3  Python deps (no --extra vlm: this machine has no torch to protect and needs none)
uv sync --group dev
# 4  Point the API at the server (edit .env: backend, url, 3 views)
Copy-Item scripts\demo_laptop\.env.laptop .env
# 5  Start the model server (window 1)  — "server listening" in < 30 s
.\scripts\serve_vlm.ps1
# 6  Start the API (window 2)
uv run --no-sync uvicorn satquery.api.app:app --host 127.0.0.1 --port 8000
# 7  Serve the built frontend (window 3)  — frontend\dist came on the stick; no Node needed
uv run --no-sync python scripts\demo_laptop\serve_frontend.py
# 8  Smoke it — 5/5 must pass with Wi-Fi OFF
uv run --no-sync python scripts\e2e_parity.py --base-url http://127.0.0.1:8000
# 9  Open the console
Start-Process http://localhost:5173
# 10 If the GPU spills (nvidia-smi > 7.4 GB, or the server dies on load): offload fewer layers and retry 5
$env:SATQUERY_VLM_GPU_LAYERS = 28; .\scripts\serve_vlm.ps1
```

If nothing serves at all, the console still runs against recorded fixtures:
open <http://localhost:5173/?mock=1>. It shows the full flow — plan, DAG,
citations, evidence — without a model. Rehearse that path too.

## What is on the stick

Built on the 24 GB box by `scripts/merge_export.py` (ROADMAP Track 2):

| File | Size | Made by |
|---|---|---|
| `models/sq-lora-v2-full-merged-Q4_K_M.gguf` | ~5.2 GB | `llama-quantize … Q4_K_M` |
| `models/sq-lora-v2-full-merged-mmproj-f16.gguf` | ~1.1 GB | `convert_hf_to_gguf.py --mmproj --outtype f16` (never quantised — the vision side is what breaks grounding first) |
| `models/sq-lora-v2-full-merged/merge_manifest.json` | 1 KB | adapter SHA-256, base model, `SATQUERY_VLM_MAX_VIEWS=3`, context 8192 |
| `models/SHA256SUMS` | 1 KB | `sha256sum` format; step 1 checks it |
| `llama-b<build>-bin-win-cuda-12.4-x64.zip` + `cudart-llama-bin-win-cuda-12.4-x64.zip` | ~90 MB | github.com/ggml-org/llama.cpp/releases — a build ≥ b6600 (Qwen3-VL support) |
| `frontend/dist/` | ~4 MB | `npm run build` with `VITE_API_BASE=http://127.0.0.1:8000` |
| `data/checkpoints/seg/`, `data/checkpoints/cd/` | ~0.4 GB | the SegFormer and Siamese checkpoints the registry needs to plan the real tools |
| `data/e2e/` (optional) | ~20 MB | real clipped GeoTIFFs for `e2e_parity.py --scenes-dir data\e2e` |
| the repository itself | ~50 MB | `git archive` of `main`, or a plain copy without `.venv` |

`scripts/demo_laptop/.env.laptop` is the `.env` for this machine:

```dotenv
SATQUERY_VLM_BACKEND=llamacpp
SATQUERY_VLM_SERVER_URL=http://127.0.0.1:8080
SATQUERY_VLM_MAX_VIEWS=3
SATQUERY_SEG_CHECKPOINT=data/checkpoints/seg/segformer-b5-loveda
SATQUERY_CD_CHECKPOINT=data/checkpoints/cd/levircd_resnet18.ckpt.pt
```

## Why these numbers

The card has 8 GB, of which ~7.2 GB is usable once the desktop has its share.

| Component | VRAM |
|---|---|
| Q4_K_M language model | ~5.0–5.2 GB |
| f16 projector | ~1.1 GB |
| KV cache, 8192 ctx, q8_0 keys and values | ~0.45 GB |
| compute buffers, image encode | ~0.4 GB |

So `serve_vlm.ps1` defaults to `--ctx-size 8192 --cache-type-k q8_0
--cache-type-v q8_0 --flash-attn on`, and the API caps prompts at **three
views** (`SATQUERY_VLM_MAX_VIEWS=3`): each 448 px view is ~1024 image tokens
(`--image-min-tokens 1024` stays — Qwen3-VL's grounding degrades below it,
and citation fidelity is the one thing this system cannot trade), and six of
them would not leave room for the answer. The policy table still plans six;
the trace records `views_ceiling: 3` and a note saying why, so a judge who
asks sees the cap rather than a silent difference from the 24 GB run.

Line 10 (`SATQUERY_VLM_GPU_LAYERS=28`) leaves ~8 of 36 layers on the CPU.
Roughly half the speed, but it starts and it answers.

## Before the day

- [ ] Steps 1–9 run twice with Wi-Fi **off**, from a cold boot.
- [ ] `nvidia-smi` during a 3-view request reads < 7.4 GB.
- [ ] `scripts/e2e_parity.py` 5/5 (the cross-modal scenario needs both checkpoints in `data/checkpoints/`).
- [ ] The HealthStrip in the console shows the VLM `ready` and `tools 6/6`.
- [ ] `runs/eval/<name>/results.md` for **both** the bf16 and the Q4_K_M runs is in the deck — the quantised column is the one this laptop serves, so it is the honest number for the demo.
- [ ] `?mock=1` rehearsed once as the fallback.

## Making the stick (on the 24 GB box)

```bash
source scripts/rocm_env.sh
uv run --no-sync python scripts/merge_export.py \
    --adapter runs/sq-lora-v2-full/adapter --out models/sq-lora-v2-full-merged \
    --gguf --llama-cpp ../llama.cpp --quant Q4_K_M --mmproj-outtype f16
cd frontend && VITE_API_BASE=http://127.0.0.1:8000 npm run build && cd ..
# copy: models/*.gguf models/SHA256SUMS models/sq-lora-v2-full-merged/merge_manifest.json
#       frontend/dist data/checkpoints the two llama.cpp zips, and the repo
```

The merge runs on the CPU in bf16 (~16 GB RAM, ~10 min); the conversion and
quantisation take another ~15 min and ~20 GB of scratch for the bf16
intermediate, which the script deletes once the Q4_K_M file exists.
