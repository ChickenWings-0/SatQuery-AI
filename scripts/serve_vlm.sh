#!/usr/bin/env bash
# Start the offline demo path: llama.cpp's server holding a GGUF Qwen3-VL.
#
# The transformers backend needs no server — it loads in-process. This script
# exists for the other path (Master.md §8 Phase 4): a quantised model that starts
# in seconds, fits in ~6 GB instead of ~16, and survives both an unplugged
# network and a ROCm install broken by a kernel update.
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
  --temp 0 \
  --seed 0 \
  --no-warmup
