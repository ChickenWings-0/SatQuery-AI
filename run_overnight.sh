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
# BigEarthNet mirror: torchgeo/bigearthnet V2, verified 2026-09-07. Its patch_id
# and s1_name match our annotation export byte for byte, and its split agrees on
# 1,764 of 1,770 patches checked — the six differences are patches the export
# calls "bench" and the mirror calls "test", held out either way. No patch the
# export calls train or validation is called test by the mirror.
#
# Size: 118.6 GB of tarballs (S1 54.8 + S2 63.5), which cannot be subsetted —
# they are monolithic archives, so the metadata-first selector §9.1 describes
# cannot avoid the bulk download. Budget ~270 GB with extraction. This box has
# 862 GB free, so it fits, but §9.1's "15 GB" line is stale.
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
# Stable, deliberately not stamped. --resume auto looks for checkpoints in
# this directory, and a name carrying the launch time is a different
# directory on every launch — so an interrupted probe could never find its
# own checkpoints and silently started again from step zero. The log keeps
# the timestamp; the run state has to stay put to be resumable.
PROBE_DIR="${PROBE_DIR:-runs/throughput-probe}"
PROFILE="configs/train/qlora_qwen3vl8b_rocm24g.yaml"
PROBE_STEPS=200
BEN_MIRROR="torchgeo/bigearthnet"
RENDER_PATCHES=20000
# CDVQA is a WebDataset that repeats each pair's image bytes once per question,
# so the whole train split is 52 GB for 2,968 distinct scenes. 300 shards is
# ~24 GB and ~1,300 scenes behind ~30,000 candidate samples — comfortably above
# the 8,000 §5 target, and the scene count is what actually bounds the variety.
CDVQA_SHARDS="${CDVQA_SHARDS:-300}"
CDVQA_VAL_SHARDS="${CDVQA_VAL_SHARDS:-40}"

ONLY_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) ONLY_STAGE="$2"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOG_DIR" "$RAW_DIR" "$CORPUS_DIR" "$VIEWS_DIR"
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
note "RSVQA-HR: Zenodo record 6344367 (off-Hub), ~13.5 GB into ${RAW_DIR}/rsvqa_hr"
note "  the Hub has no usable RSVQA-HR release; see scripts/fetch_sources.py"
uv run python scripts/fetch_sources.py --source rsvqa_hr --out "${RAW_DIR}"

echo
note "CDVQA: ${CDVQA_SHARDS} WebDataset shards from ljx620/CDVQA into ${RAW_DIR}/cdvqa"
note "  ~79 MB a shard, 100 samples each; the §5 target is 8,000"
uv run python scripts/fetch_sources.py --source cdvqa --out "${RAW_DIR}" \
    --shards "${CDVQA_SHARDS}" --val-shards "${CDVQA_VAL_SHARDS}"

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
note "BigEarthNet S1/S2 patches from ${BEN_MIRROR} — 118.6 GB, the long pole"
AVAIL_GB="$(df -BG --output=avail . | tail -1 | tr -dc '0-9')"
if (( AVAIL_GB < 300 )); then
  echo "  !! only ${AVAIL_GB} GB free; the archives plus extraction need ~270 GB." >&2
  exit 1
fi
note "  ${AVAIL_GB} GB free — proceeding"
uv run hf download "$BEN_MIRROR" \
    --repo-type dataset \
    --local-dir "${RAW_DIR}/ben" \
    --include "V2/*"

note "reassembling the split archives and extracting"
for part in S1 S2; do
  if [[ ! -d "${RAW_DIR}/ben/BigEarthNet-${part}" ]]; then
    cat "${RAW_DIR}"/ben/V2/BigEarthNet-${part}.tar.gza* > "${RAW_DIR}/ben/${part}.tar.gz"
    tar -xzf "${RAW_DIR}/ben/${part}.tar.gz" -C "${RAW_DIR}/ben"
    rm -f "${RAW_DIR}/ben/${part}.tar.gz"
    note "  BigEarthNet-${part} extracted"
  else
    note "  BigEarthNet-${part} already extracted — skipping"
  fi
done

fi

# --------------------------------------------------------------- stage 2
if wants 2; then
banner "STAGE 2/3 — Phase 2 render pass (CPU-bound, ~6 h)"
note "rendering ${RENDER_PATCHES} train patches into ${VIEWS_DIR}/bigearthnet_v2"
uv run python scripts/render_views.py \
    --input "${RAW_DIR}/ben" \
    --out "${VIEWS_DIR}" \
    --split train \
    --limit "${RENDER_PATCHES}" \
    --size 448 \
    --workers 16

note "rendering the validation patches"
uv run python scripts/render_views.py \
    --input "${RAW_DIR}/ben" \
    --out "${VIEWS_DIR}" \
    --split validation \
    --limit 3234 \
    --size 448 \
    --workers 16

note "rendering the 3-channel VHR sources — TC only, no FactSheets to measure"
for vhr in vrsbench rsvqa_hr cdvqa; do
  note "  ${vhr}"
  uv run python scripts/render_vhr_views.py \
      --source "$vhr" \
      --input "${RAW_DIR}" \
      --out "${VIEWS_DIR}" \
      --split all \
      --size 448 \
      --workers 16
done

FACTS="${VIEWS_DIR}/bigearthnet_v2/factsheets.jsonl"
if [[ ! -s "$FACTS" ]]; then
  echo "  !! ${FACTS} is empty after both render passes. Stopping — regenerating" >&2
  echo "     evidence_qa from an empty sheet would silently fall back to nothing." >&2
  exit 1
fi
note "regenerating evidence_qa from the MEASURED FactSheets in ${FACTS} ($(wc -l < "$FACTS") sheets)"
uv run python training/data/builders/evidence_qa.py --count 3000 --factsheets "$FACTS"
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
  # An `[[ -f ]] && cat` here ends the loop body with a false test whenever a
  # part is absent, and under `set -e` that exits the whole script silently.
  if [[ -f "$part" ]]; then
    cat "$part" >> "$PROBE_CORPUS"
  else
    note "  ${part} not present — omitted from the probe corpus"
  fi
done
if [[ ! -s "$PROBE_CORPUS" ]]; then
  echo "  !! no corpus to probe with. Run scripts/build_corpus.py first." >&2
  exit 1
fi
note "probe corpus: $(wc -l < "$PROBE_CORPUS") rows -> ${PROBE_CORPUS}"

note "checking every referenced view resolves to a real file"
uv run python scripts/patch_dummy_images.py --corpus "$PROBE_CORPUS"
# --verify exits 1 whenever the corpus still leans on placeholders, which is the
# expected state for a throughput probe. Informational here, so its code is not
# allowed to end the run. The patching call above no longer carries "|| true",
# so a genuine failure to write the placeholders now stops the stage.
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
echo "Still open, in priority order:"
echo "  1. Throughput: 0.248 samples/s puts one epoch of 65k at ~73 h. Decide the"
echo "     corpus size and view count before committing the GPU (see the notes)."
echo "  2. DIOR-RSVG — 9.2% of the §5 composition, still unresolved: the canonical"
echo "     release is gated. Accept the terms and set HF_TOKEN. (RSVQA-HR and CDVQA"
echo "     are resolved; they are fetched in stage 1 above.)"
echo "  3. Re-run scripts/build_corpus.py once the real views exist, so the corpus"
echo "     points at rendered pixels rather than placeholders."
