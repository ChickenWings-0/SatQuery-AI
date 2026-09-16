#!/usr/bin/env bash
# Start the offline demo path: llama.cpp's server holding a GGUF Qwen3-VL.
#
# The transformers backend needs no server — it loads in-process. This script
# exists for the other path (Master.md §8 Phase 4): a quantised model that starts
# in seconds, fits in ~6 GB instead of ~16, and survives both an unplugged
# network and a ROCm install broken by a kernel update.
#
# For the transformers path with the Phase 7 adapter, no server is involved —
# point the process at the adapter and it is applied over bf16 base weights at
# load (~18.7 GiB peak, six views):
#     export SATQUERY_VLM_BACKEND=hf
#     export SATQUERY_VLM_ADAPTER_PATH=runs/full-epoch-v1/adapter
# That path is deliberately unquantised: NF4 generation is broken on gfx1100,
# see satquery/models/hf_backend.py.
#
# Point SatQuery at it with:
#     export SATQUERY_VLM_BACKEND=llamacpp
#     export SATQUERY_VLM_SERVER_URL=http://127.0.0.1:8080
#
# Usage: scripts/serve_vlm.sh [path/to/model.gguf] [path/to/mmproj.gguf]
set -euo pipefail

MODEL="${1:-${SATQUERY_VLM_GGUF_PATH:-models/Qwen3-VL-8B-Instruct-Q4_K_M.gguf}}"
MMPROJ="${2:-${SATQUERY_VLM_MMPROJ_PATH:-models/Qwen3-VL-8B-Instruct-mmproj-f16.gguf}}"
HOST="${SATQUERY_VLM_HOST:-127.0.0.1}"
PORT="${SATQUERY_VLM_PORT:-8080}"
CONTEXT="${SATQUERY_VLM_CONTEXT:-16384}"
GPU_LAYERS="${SATQUERY_VLM_GPU_LAYERS:-99}"
# Qwen3-VL degrades on grounding tasks below 1024 image tokens: llama.cpp
# warns about it at load, and the failure mode is a confident answer about a
# region the model could not actually resolve. Citation fidelity is the one
# thing this system cannot trade, so the floor is set here rather than left
# to llama.cpp's smaller default. See ggml-org/llama.cpp#16842.
IMAGE_MIN_TOKENS="${SATQUERY_VLM_IMAGE_MIN_TOKENS:-1024}"
# KV-cache precision and flash attention. Defaults are the 24 GB box's (f16
# cache, auto flash attention); the 8 GB laptop sets q8_0/q8_0/on to fit the
# Q4_K_M model, its f16 projector and an 8192 context — see serve_vlm.ps1 and
# scripts/demo_laptop/README.md, which are the same flags for Windows.
CACHE_TYPE_K="${SATQUERY_VLM_CACHE_TYPE_K:-f16}"
CACHE_TYPE_V="${SATQUERY_VLM_CACHE_TYPE_V:-f16}"
FLASH_ATTN="${SATQUERY_VLM_FLASH_ATTN:-auto}"

SERVER="${LLAMA_SERVER_BIN:-llama-server}"

if ! command -v "$SERVER" >/dev/null 2>&1; then
  echo "error: '$SERVER' is not on PATH." >&2
  echo "  Build llama.cpp with HIP (-DGGML_HIP=ON) or Vulkan (-DGGML_VULKAN=ON)," >&2
  echo "  or set LLAMA_SERVER_BIN to the binary." >&2
  exit 127
fi

for file in "$MODEL" "$MMPROJ"; do
  if [[ ! -f "$file" ]]; then
    echo "error: $file does not exist." >&2
    echo "  Both the weights and the multimodal projector are required; a" >&2
    echo "  vision model served without its mmproj answers about no image at all." >&2
    exit 66
  fi
done

echo "serving $MODEL on http://${HOST}:${PORT}"
exec "$SERVER" \
  --model "$MODEL" \
  --mmproj "$MMPROJ" \
  --host "$HOST" \
  --port "$PORT" \
  --ctx-size "$CONTEXT" \
  --n-gpu-layers "$GPU_LAYERS" \
  --image-min-tokens "$IMAGE_MIN_TOKENS" \
  --cache-type-k "$CACHE_TYPE_K" \
  --cache-type-v "$CACHE_TYPE_V" \
  --flash-attn "$FLASH_ATTN" \
  --temp 0 \
  --seed 0 \
  --no-warmup
