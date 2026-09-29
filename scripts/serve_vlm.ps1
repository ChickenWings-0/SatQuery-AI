<#
.SYNOPSIS
    Start the offline demo path on Windows: llama-server holding the Q4_K_M
    SatQuery GGUF on an RTX 4070 Laptop (8 GB VRAM).

.DESCRIPTION
    The PowerShell twin of scripts/serve_vlm.sh — same environment variable
    names, same flags, one machine class in mind. The defaults are the 8 GB
    budget worked out in ROADMAP_REMAINING_FIXES.md Track 2:

        Q4_K_M language model      ~5.0-5.2 GB
        f16 multimodal projector   ~1.1 GB
        KV cache, 8192 ctx, q8_0   ~0.45 GB
        compute / image encode     ~0.4 GB

    which fits only with --ctx-size 8192, --cache-type-k/v q8_0 and
    --flash-attn on, and only with SATQUERY_VLM_MAX_VIEWS=3 on the API side
    (every 448 px view is ~1024 image tokens of that context). If nvidia-smi
    still shows the card spilling, set SATQUERY_VLM_GPU_LAYERS=28 to leave the
    last layers on the CPU: slower, but it answers.

    Needs the official CUDA build of llama.cpp — llama-<build>-bin-win-cuda-12.4-x64.zip
    plus cudart-llama-bin-win-cuda-12.4-x64.zip from the same release, unzipped
    into one folder — no CUDA toolkit install. See DOCS/DEMO_LAPTOP_RUNBOOK.md.

    Point SatQuery at it with (in .env):
        SATQUERY_VLM_BACKEND=llamacpp
        SATQUERY_VLM_SERVER_URL=http://127.0.0.1:8080
        SATQUERY_VLM_MAX_VIEWS=3

.PARAMETER Model
    Path to the Q4_K_M GGUF. Default: $env:SATQUERY_VLM_GGUF_PATH, else
    models\sq-lora-v2-full-merged-Q4_K_M.gguf.

.PARAMETER MmProj
    Path to the f16 projector GGUF. Default: $env:SATQUERY_VLM_MMPROJ_PATH, else
    models\sq-lora-v2-full-merged-mmproj-f16.gguf.

.EXAMPLE
    .\scripts\serve_vlm.ps1
    .\scripts\serve_vlm.ps1 -Model D:\sq\model.gguf -MmProj D:\sq\mmproj.gguf
    $env:SATQUERY_VLM_GPU_LAYERS = 28; .\scripts\serve_vlm.ps1
#>
[CmdletBinding()]
param(
    [string]$Model = $(if ($env:SATQUERY_VLM_GGUF_PATH) { $env:SATQUERY_VLM_GGUF_PATH } else { "models\sq-lora-v2-full-merged-Q4_K_M.gguf" }),
    [string]$MmProj = $(if ($env:SATQUERY_VLM_MMPROJ_PATH) { $env:SATQUERY_VLM_MMPROJ_PATH } else { "models\sq-lora-v2-full-merged-mmproj-f16.gguf" })
)

$ErrorActionPreference = "Stop"

function Get-Setting([string]$Name, [string]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
    return $value
}

$Bind          = Get-Setting "SATQUERY_VLM_HOST"             "127.0.0.1"
$Port          = Get-Setting "SATQUERY_VLM_PORT"             "8080"
# 8192, not the 16384 the 24 GB box uses: the difference is the KV cache the
# laptop does not have room for, and three views fit comfortably inside it.
$Context       = Get-Setting "SATQUERY_VLM_CONTEXT"          "8192"
$GpuLayers     = Get-Setting "SATQUERY_VLM_GPU_LAYERS"       "99"
# Qwen3-VL degrades on grounding below 1024 image tokens (ggml-org/llama.cpp#16842);
# citation fidelity is the one thing this system cannot trade, so the floor stays
# even on the small card — the view *count* is what gets reduced, not the resolution.
$ImageMinTok   = Get-Setting "SATQUERY_VLM_IMAGE_MIN_TOKENS" "1024"
$CacheTypeK    = Get-Setting "SATQUERY_VLM_CACHE_TYPE_K"     "q8_0"
$CacheTypeV    = Get-Setting "SATQUERY_VLM_CACHE_TYPE_V"     "q8_0"
$FlashAttn     = Get-Setting "SATQUERY_VLM_FLASH_ATTN"       "on"
$ServerBin     = Get-Setting "LLAMA_SERVER_BIN"              "llama-server.exe"

# Resolve the binary: an explicit path, then PATH, then a llama.cpp folder
# sitting next to the repository (where the runbook says to unzip it).
$server = $null
if (Test-Path $ServerBin) {
    $server = (Resolve-Path $ServerBin).Path
} elseif (Get-Command $ServerBin -ErrorAction SilentlyContinue) {
    $server = (Get-Command $ServerBin).Source
} else {
    $sibling = Join-Path (Split-Path $PSScriptRoot -Parent) "..\llama.cpp\llama-server.exe"
    if (Test-Path $sibling) { $server = (Resolve-Path $sibling).Path }
}
if (-not $server) {
    Write-Error @"
'$ServerBin' was not found.
  Unzip llama-<build>-bin-win-cuda-12.4-x64.zip and cudart-llama-bin-win-cuda-12.4-x64.zip
  into ..\llama.cpp next to this repository, or set LLAMA_SERVER_BIN to llama-server.exe.
"@
    exit 127
}

foreach ($file in @($Model, $MmProj)) {
    if (-not (Test-Path $file)) {
        Write-Error @"
$file does not exist.
  Both the weights and the multimodal projector are required; a vision model
  served without its mmproj answers about no image at all. Verify the copy with
    Get-FileHash -Algorithm SHA256 (Get-Content models\SHA256SUMS | ForEach-Object { ($_ -split '\s+')[1] })
"@
        exit 66
    }
}

$arguments = @(
    "--model", $Model,
    "--mmproj", $MmProj,
    "--host", $Bind,
    "--port", $Port,
    "--ctx-size", $Context,
    "--n-gpu-layers", $GpuLayers,
    "--image-min-tokens", $ImageMinTok,
    "--cache-type-k", $CacheTypeK,
    "--cache-type-v", $CacheTypeV,
    "--flash-attn", $FlashAttn,
    "--temp", "0",
    "--seed", "0",
    "--no-warmup"
)

Write-Host "serving $Model on http://${Bind}:${Port}"
Write-Host "  ctx $Context · kv $CacheTypeK/$CacheTypeV · flash-attn $FlashAttn · gpu layers $GpuLayers · image-min-tokens $ImageMinTok"
Write-Host "  API side: SATQUERY_VLM_BACKEND=llamacpp SATQUERY_VLM_SERVER_URL=http://${Bind}:${Port} SATQUERY_VLM_MAX_VIEWS=3"

& $server @arguments
exit $LASTEXITCODE
