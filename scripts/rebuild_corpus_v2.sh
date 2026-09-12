#!/usr/bin/env bash
#
# rebuild_corpus_v2.sh — ML_PIPELINE_RECOVERY_PLAN §5.2–5.3, in order, gated.
#
#   ./scripts/rebuild_corpus_v2.sh                 # sprint preset (§5.4), ~20 k train
#   PRESET=full ./scripts/rebuild_corpus_v2.sh     # "full §5 minus DIOR", ~59 k train
#   OUT=data/processed/corpus/v2 ./scripts/rebuild_corpus_v2.sh
#
# 1. RSVQA-HR: extract Data/ from the already-fetched Images.tar, render TC views.
# 2. Build with EVERY Track-B source named and --on-missing-views fail — never
#    skip: skip is how a source quietly loses 90 % of itself.
# 3. build_corpus.py's own post-build gates: every requested source above its
#    floor (composition.json), every GROUNDING answer canonical. Exit 5 stops
#    here; nothing downstream reads a corpus that failed them.
# 4. evidence_qa's val split is generated in-band from the validation-patch
#    FactSheets (it used to be 0 bytes); asserted non-empty below.
#
# The §2.5 grounding-format fix is in the builder, so the corpus is built once.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PRESET="${PRESET:-sprint}"
OUT="${OUT:-data/processed/corpus/v2}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-16}"
SOURCES="bigearthnet_v2 vrsbench rsvqa_hr cdvqa evidence_qa"

case "$PRESET" in
  sprint)  COMPOSITION="bigearthnet_v2=8000 vrsbench=6000 rsvqa_hr=2500 cdvqa=1500 evidence_qa=2000" ;;
  full)    COMPOSITION="bigearthnet_v2=18000 vrsbench=20000 rsvqa_hr=10000 cdvqa=8000 evidence_qa=3000" ;;
  *) echo "unknown PRESET=$PRESET (sprint|full)" >&2; exit 2 ;;
esac

banner() { echo; echo "==================== $* ===================="; echo "[$(date '+%F %T')]"; }

banner "1/3 RSVQA-HR: extract + render (§5.2 step 1)"
if [[ -d data/raw/rsvqa_hr/Data ]] && [[ "$(find data/raw/rsvqa_hr/Data -name '*.png' | wc -l)" -ge 10000 ]]; then
  echo "  have data/raw/rsvqa_hr/Data"
else
  uv run python scripts/fetch_sources.py --source rsvqa_hr
fi
uv run python scripts/render_vhr_views.py --source rsvqa_hr --split all --size 448 --workers "$WORKERS"

banner "2/3 build: $PRESET preset → $OUT (§5.2 step 2, §5.3 gates)"
mkdir -p "$OUT"
# shellcheck disable=SC2086
uv run python scripts/build_corpus.py \
    --sources $SOURCES \
    --on-missing-views fail \
    --composition $COMPOSITION \
    --seed "$SEED" \
    --out "$OUT"

banner "3/3 verify (§5.2 step 3, §5.3)"
test -s "$OUT/composition.json" || { echo "composition.json missing" >&2; exit 5; }
uv run python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
comp = json.loads((out / "composition.json").read_text())
val = [json.loads(l) for l in (out / "val.jsonl").read_text().splitlines() if l.strip()]
evidence_val = sum(1 for s in val if s["source"] == "evidence_qa")
for name, row in comp["sources"].items():
    print(f"  {name:16s} train {row['train']:>6d}  val {row['val']:>5d}")
if evidence_val == 0:
    print("evidence_qa has no val split (E5 needs one)", file=sys.stderr)
    sys.exit(5)
print(f"  evidence_qa val split: {evidence_val} samples")
PY
echo
echo "corpus at $OUT — next: scripts/preflight_train_serve_parity.py --corpus $OUT/train.jsonl"
