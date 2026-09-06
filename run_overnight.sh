#!/usr/bin/env bash
#
# run_overnight.sh — the three-stage overnight playbook.
#
#   ./run_overnight.sh              # run every stage in order
#   ./run_overnight.sh --stage 3    # run one stage only
#   tail -f logs/overnight-*.log    # watch from another terminal
#
# Stage 1  datasets      network-bound, ~4 h
# Stage 2  render pass   CPU-bound, ~6 h, 16 threads
# Stage 3  GPU probe     ~30 min on the 7900 XTX
#
# READ THIS BEFORE RUNNING. Two of the three stages are not fully implemented,
# and this script stops at each of those with instructions rather than guessing:
#
#   * BigEarthNet imagery. The repo id in DOCS is the annotation export
#     (BigEarthNet.txt), not the S1/S2 patches. `BIFOLD-BigEarthNetv2-0/
#     BigEarthNet-V2` returns 401. Candidates exist on the Hub
#     (earthnets/BigEarthNetV2, GFM-Bench/BigEarthNet, torchgeo/bigearthnet)
#     but none is verified to key patches by the `patch_id` our corpus records,
#     and picking the wrong mirror is how a quarantined test split gets in.
#   * The Phase 2 render pass. There is no CLI driving `render.renderer`, so
#     stage 2 has nothing to execute. `src/satquery/render/` has the renderer;
#     what is missing is the script that walks patches, writes the views, and
#     records each patch's FactSheet.
#
# What DOES run unattended tonight: the VRSBench download (verified — the repo
# publishes Images_train.zip / Images_val.zip) and the GPU throughput probe.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="logs"
LOG="${LOG_DIR}/overnight-${STAMP}.log"
RAW_DIR="data/raw"
CORPUS_DIR="data/processed/corpus"
VIEWS_DIR="data/processed/views"
PROBE_DIR="runs/throughput-probe-${STAMP}"
PROFILE="configs/train/qlora_qwen3vl8b_rocm24g.yaml"
PROBE_STEPS=200

ONLY_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) ONLY_STAGE="$2"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOG_DIR" "$RAW_DIR"
exec > >(tee -a "$LOG") 2>&1

banner() { echo; echo "==================== $* ===================="; echo "[$(date '+%F %T')]"; }
note()   { echo "  -- $*"; }
skip()   { echo "  ** SKIPPED: $*"; }
wants()  { [[ -z "$ONLY_STAGE" || "$ONLY_STAGE" == "$1" ]]; }

echo "SatQuery AI overnight run — ${STAMP}"
echo "log: ${LOG}"

# --------------------------------------------------------------- stage 1
if wants 1; then
banner "STAGE 1/3 — dataset downloads (network-bound, ~4 h)"

note "VRSBench: images + annotations, ~12 GB, into ${RAW_DIR}/vrsbench"
uv run hf download xiang709/VRSBench \
    --repo-type dataset \
    --local-dir "${RAW_DIR}/vrsbench" \
    --include "*.zip"
note "VRSBench archives downloaded. Unzipping:"
for archive in "${RAW_DIR}"/vrsbench/*.zip; do
  [[ -e "$archive" ]] || continue
  note "  unzip $(basename "$archive")"
  unzip -q -o "$archive" -d "${RAW_DIR}/vrsbench"
done

note "BigEarthNet.txt annotations: already streamed by build_corpus.py; no bulk download needed."

echo
note "DIOR-RSVG (gated — needs an accepted licence and HF_TOKEN):"
if [[ -n "${HF_TOKEN:-}" ]]; then
  uv run hf download danielz01/DIOR-RSVG \
      --repo-type dataset --local-dir "${RAW_DIR}/dior_rsvg" \
  || skip "DIOR-RSVG download failed — has this account accepted the terms at
           https://huggingface.co/datasets/danielz01/DIOR-RSVG ?"
else
  skip "HF_TOKEN is not set. Accept the terms on the dataset page, then
        export HF_TOKEN=hf_... and re-run with --stage 1."
fi

echo
note "BigEarthNet S1/S2 PATCH IMAGERY — not attempted."
cat <<'BEN'
        The id in DOCS/OPEN_SOURCE_ASSETS.md is the text export, and
        BIFOLD-BigEarthNetv2-0/BigEarthNet-V2 returns 401. Unverified
        candidates:
            earthnets/BigEarthNetV2
            GFM-Bench/BigEarthNet
            torchgeo/bigearthnet
        Before scripting one, confirm it keys patches by the same patch_id the
        corpus records, e.g. S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57,
        and that its splits match reBEN's. A mirror that renames or re-splits
        patches puts held-out data into training (§4.7).
BEN
fi

# --------------------------------------------------------------- stage 2
if wants 2; then
banner "STAGE 2/3 — Phase 2 render pass (CPU-bound, ~6 h)"
if [[ -f scripts/render_views.py ]]; then
  note "rendering views + FactSheets into ${VIEWS_DIR}"
  uv run python scripts/render_views.py \
      --input "${RAW_DIR}" \
      --out "${VIEWS_DIR}" \
      --size 448 \
      --workers 16
  note "regenerating evidence_qa from the MEASURED FactSheets"
  uv run python training/data/builders/evidence_qa.py --count 3000
else
  skip "scripts/render_views.py does not exist — nothing drives the renderer yet."
  cat <<'RENDER'
        The renderer itself is in src/satquery/render/ (render_views(),
        RenderSource, the frozen view catalogue). What is missing is the driver:
        walk the downloaded patches, select views per §2.4, write JPEG/PNG to
        data/processed/views/<source>/<patch_id>/<VIEW>.{jpg,png}, and record
        each patch's measured scalars so evidence_qa can be regenerated from
        real numbers instead of the mock sheets it currently uses.

        Until this exists the corpus points at solid-black placeholders, and any
        adapter trained on it has learned nothing about imagery.
RENDER
fi
fi

# --------------------------------------------------------------- stage 3
if wants 3; then
banner "STAGE 3/3 — GPU throughput probe (${PROBE_STEPS} steps, ~30 min)"

# shellcheck source=scripts/rocm_env.sh
source scripts/rocm_env.sh
note "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} (the iGPU must stay hidden)"

PROBE_CORPUS="${CORPUS_DIR}/probe.train.jsonl"
: > "$PROBE_CORPUS"
for part in "${CORPUS_DIR}/train.jsonl" "${CORPUS_DIR}/evidence_qa.train.jsonl"; do
  [[ -f "$part" ]] && cat "$part" >> "$PROBE_CORPUS"
done
if [[ ! -s "$PROBE_CORPUS" ]]; then
  echo "  !! no corpus to probe with. Run scripts/build_corpus.py first." >&2
  exit 1
fi
note "probe corpus: $(wc -l < "$PROBE_CORPUS") rows -> ${PROBE_CORPUS}"

note "checking every referenced view resolves to a real file"
uv run python scripts/patch_dummy_images.py --corpus "$PROBE_CORPUS" || true
uv run python scripts/patch_dummy_images.py --corpus "$PROBE_CORPUS" --verify || true

note "profile: ${PROFILE} (384 px, 6 views, effective batch 16)"
uv run python scripts/train_vlm.py \
    --config "$PROFILE" \
    --corpus "$PROBE_CORPUS" \
    --output-dir "$PROBE_DIR" \
    --max-steps "$PROBE_STEPS" \
    --resume auto

banner "PROBE COMPLETE"
note "peak VRAM and metrics: ${PROBE_DIR}/run_manifest.json"
note "read peak_vram_gib against the profile's 17.13 GiB estimate, and"
note "metrics.train_samples_per_second to turn '7-11 h' into a real number:"
note "  65000 samples / (samples_per_second * 3600) = hours for one epoch"
if [[ -f "${PROBE_DIR}/run_manifest.json" ]]; then
  uv run python -c "
import json,sys
m=json.load(open('${PROBE_DIR}/run_manifest.json'))
print('  peak_vram_gib     :', m.get('peak_vram_gib'))
print('  estimate_gib      :', round(m.get('vram_estimate',{}).get('total_bytes',0)/1024**3, 2))
print('  metrics           :', json.dumps(m.get('metrics',{}), indent=2)[:600])
"
fi
fi

banner "OVERNIGHT RUN FINISHED"
echo "[$(date '+%F %T')]  full log: ${LOG}"
echo
echo "Blocked on a decision from you, in priority order:"
echo "  1. BigEarthNet patch imagery — which mirror, verified against patch_id."
echo "  2. scripts/render_views.py — the Phase 2 driver does not exist yet."
echo "  3. RSVQA-HR and CDVQA sources (29% of the §5 composition, still unresolved)."
