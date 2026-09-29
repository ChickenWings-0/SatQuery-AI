# 08 — Open Issues and Next Steps

The build is feature-complete. Everything the 2026-09-11 audit, the remediation plan,
the ML recovery plan and the four pre-final tracks asked for has landed (`02`). What
is left is ordered by what could embarrass the demo first.

## Tier 1 — before the SIH final (no code, just doing it)

1. **Run the manual QA and fill in the sign-off sheet** in
   `DOCS/FINAL_QA_CHECKLIST.md §6`. It is empty. Every ❗ row must pass on the laptop
   with Wi-Fi off, or the presentation opens in `?mock=1` and says so.
2. **Rehearse the laptop stack twice with Wi-Fi off** (`DOCS/DEMO_LAPTOP_RUNBOOK.md`):
   copy `models/*.gguf` + `SHA256SUMS`, `frontend/dist/`, `data/checkpoints/`;
   verify checksums; `serve_vlm.ps1` listening < 30 s; < 7.4 GB VRAM with a 3-view
   request; HealthStrip shows the VLM `ready`. Record the timings the checklist asks for.
3. **Re-run `make ci` and `make e2e`** on the final tree (last e2e was 2026-09-15, before
   the 09-16/09-18 commits). Note `make e2e` with `hf` will re-download the base model.
4. **Commit the 2026-09-29 docs/gitignore reorganisation** (leave out
   `frontend/public/sitemap.xml`).
5. **Slide numbers come only from `runs/eval/sq-lora-v2-full/results.md`** and the
   zero-shot → adapted answer-token accuracy in `run_manifest.json`. Do not quote the
   CD model as meeting its 0.88 gate.

## Tier 2 — extra evidence (needs the data pipeline rebuilt, `06` last section)

6. **Benchmark baseline column:** `make eval-baseline` on the same `sample_ids.json`.
   The trainer's zero-shot eval already shows the gap (26.4 % → 81.8 % answer-token
   accuracy), but a per-source table is stronger on a slide.
7. **Q4_K_M column:** `make eval` with `SATQUERY_VLM_BACKEND=llamacpp` against the
   exported GGUF; accept ≤ 2 pt VQA drop, ≤ 3 pt grounding drop (Track 2 step 3).
8. **Merged-model parity check:** the merged bf16 model through `hf`
   (`SATQUERY_VLM_MODEL_PATH=models/sq-lora-v2-full-merged`) should decode the probe
   queries identically to adapter-over-base (Track 2 step 2). Not recorded.

## Tier 3 — model quality (after the final, if the project continues)

9. **SCENE_CLASSIFY** (multi-label BigEarthNet) is 5 % exact / 50 % label-F1 and drags
   BEN to 53 %. Options: more SCENE_CLASSIFY share in the corpus, a label-set answer
   format that scores partial credit, or route scene classification to a
   deterministic classifier and let the VLM phrase it.
10. **Grounding** R@0.5 48.9 % (BEN 40 %, VRSBench 56 %). DIOR-RSVG (gated) is the
    obvious extra source.
11. **Change detection:** close LEVIR-CD F1 0.858 → ≥ 0.88; run OSCD (10 m) to record
    cross-resolution degradation.
12. **Cartosat-2S / RISAT augmentation track** (`DATA_ADAPTATION_PLAN §7.3`) — the
    hidden test set's sensors; nothing done yet.

## Tier 4 — engineering debt

13. **Jobs are in-memory** (lost on restart); traces are SQLite and artifacts a local
    filesystem — fine for one box, not for multi-process serving.
14. **"11 compatibility checks"** in README / `USER_GUIDE.md` / the roadmap means the 10
    `schemas/enums.CheckName` checks plus the pair-type classification. Harmless, but
    say "10 checks + pair-type" if a judge asks to count them.
15. **`run_manifest.json` records `run_name: sq-lora-v1`** for the v2 run — set
    `run_name` in the profile before the next training run.
16. **v1-era leftovers:** root `*.log`, `.levircd.pid`, `logs/`, `run_overnight.sh`
    (superseded by `rebuild_corpus_v2.sh`), `scripts/eval_vrsbench_zeroshot.py`
    (superseded by `eval_benchmark.py`), `runs/Sep13_*_fedora/`, `runs/sq-lora-v2/`.
    All harmless; delete when convenient.
17. `ruff format` drift (~58 files) — only if the owner wants formatting enforced.

## Open questions

- Which machine presents: the 24 GB box (bf16 + adapter) or the 8 GB laptop (Q4_K_M)?
  The runbook assumes the laptop is the fallback "if the training box is not in the room".
- Is the hosted static build (`frontend/.vercel/` link) part of the submission?

## What *not* to do

- Do not run a bare `uv sync` on the dev box.
- Do not quantise the `hf` serving path to NF4.
- Do not serve an adapter without its `layout.json`, or with a processor from elsewhere.
- Do not add a second implementation of labels, layout, box format, number format or
  index maths.
- Do not let an LLM choose tools; extend `policy_table.yaml`.
- Do not delete `models/` — it is the owner's backup of the exported weights.
- Do not commit `data/`, `runs/` (except eval results), `models/`, logs, `.env`, `.claude/`.
