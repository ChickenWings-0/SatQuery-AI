# 08 — Open Issues and Next Steps

Ordered by leverage. Items 1–3 are what the SIH rubric's "domain adaptation" score
depends on; everything else is polish by comparison. Source for most items:
`DOCS/project_audit.md §7` (2026-09-11), cross-checked against the code.

## Tier 1 — the ML result (do these first, in this order)

1. **Fix the training objective — follow `DOCS/ML_PIPELINE_RECOVERY_PLAN.md`.**
   Do **not** reach for `assistant_only_loss=True` + `{% generation %}` markers, and do
   not look for `DataCollatorForCompletionOnlyLM`: on the pinned stack (trl 1.12.0,
   `pyproject.toml` `vlm-train`) the first raises
   `ValueError("Assistant-only loss is not yet supported for vision datasets")` for any
   dataset with an `images` key, and the second was removed in trl 0.20. The working
   mechanism is the **prompt-completion record shape** with `completion_only_loss=True`
   (plan §1): `sample_to_chat()` emits `{prompt, completion, images}`, the vision
   collator masks everything up to and including `<|im_start|>assistant\n`, and
   `audit_masks()` checks the collated labels equal the tokenised answer before the
   weights load. The user turn is laid out by `satquery.models.prompts.layout` on
   both the training and serving paths (plan §2) — label as a text part before each
   image — with a fingerprint checked at model load. Then the Stage A memorisation
   probe and Stage B descent probe (plan §3) **before** the epoch.
2. **Run one honest ablation.** Base Qwen3-VL-8B bf16 vs adapter, same prompt, same
   200 VRSBench VQA + BEN-val items, same CitationValidator. Metrics per
   `DATA_ADAPTATION_PLAN §8`: BEN-v2 19-class micro-F1 (zero-shot vs adapted),
   6-class replication, evidence-audit fraction (`strict` validator mode), VRSBench
   VQA accuracy. Put the table in `DOCS/` even if the numbers are bad.
3. **Rebuild the corpus properly.** Bind hashes for every source in
   `scripts/build_corpus.py` (`hashes_for` / `bind_view_paths`) and add the test
   that asserts a BEN row through `_iter_source_samples` has `image_sha256`. Include
   VRSBench (29,615 tiles rendered) and CDVQA (271 pairs rendered) and RSVQA-HR;
   resolve or drop DIOR-RSVG (gated). Generate an `evidence_qa` **val** split.
   Re-budget `COMPOSITION` (65k at 0.265 samples/s ≈ 68–73 h) to the GPU window
   actually available.
4. **Re-export the GGUF from the served adapter** (`scripts/merge_export.py`) so the
   offline llama.cpp path is not the PoC adapter.

## Tier 2 — backend correctness and robustness

5. **Key-aware citation check.** When the model emits `[key]`, require the adjacent
   number to match *that* key's value; fall back to value search only for bare
   numbers (`evidence/citation_validator.py::validate`, `strip_citation_markers`).
6. **Unblock the event loop.** `asyncio.to_thread(ingest, …)` in
   `api/routers/analyze.py` and `api/routers/jobs.py`.
7. **Process-wide GPU semaphore** in the executor (currently created per
   `DagExecutor` in `agent/pipeline.py`). Phase 9's "10 concurrent requests without
   VRAM exhaustion" is not met.
8. **`DELETE /v1/jobs/{id}`** so the UI's cancel actually stops the GPU; contract
   bump + `openapi.json` + `schema.d.ts` regeneration.
9. **Change detection:** train the OSCD (10 m) run and record cross-resolution
   degradation; try to close the LEVIR-CD gap (F1 0.858 → ≥ 0.88).
10. **Hygiene:** fill root `README.md` (copy `repo documentation/README.md` and fix the
    65k/19-h claims); delete `src/satquery_ai/` and the `[project.scripts]` entry
    pointing at it; fix the 1 mypy (`local_sources.py:132`) and 1 ruff
    (`scripts/test_inference.py` import order) error; make `export_openapi.py` use
    argparse; remove `.levircd.pid`; fix the hard-coded change legend
    (`ImageViewer.tsx` ~line 361 always says "New built-up area").

## Tier 3 — frontend

11. **Reattach instead of re-run on SSE drop.** `thread/useRun.ts` currently marks
    failed and `retry` re-submits the whole job (a fresh 20–60 s GPU run). Poll
    `GET /v1/jobs/{id}` (already in `client.ts`) or reopen `/events` (server replays
    the buffered stage sequence).
12. **Consume the `BBOX_SET` artifact** from `text_grounding` (pixel + WGS84) instead
    of re-parsing answer text; keep `bbox.ts` as the fallback for text-only answers.
13. Stub `CSS.registerProperty` in `vitest.config.ts` to silence
    `react-compare-slider` noise; add one Playwright test of the bbox overlay on a
    non-square image (the highest-risk visual path); fix or replace oxlint.

## Tier 4 — Phase 8 and Phase 9 as planned

14. **Phase 8 eval harness:** `src/satquery/eval/` with VRSBench (VQA acc, caption
    BLEU-4/ROUGE-L/CIDEr, grounding mAP@0.5), RSVQA-HR (per-type + aggregate), CDVQA
    accuracy, `metrics.py`, `report.py` (markdown + CSV), a synthetic Cartosat-2S /
    RISAT proxy dry-run (resample VHR to 0.65 m, simulate single-pol SAR), and the
    four-row ablation table from one `make eval`.
15. **Phase 9 hardening:** CI workflow (`ruff`, `mypy`, `pytest -m "not gpu"`,
    `tsc -b`, `vitest run`, `vite build`), `Dockerfile`/`docker-compose.yml` on a
    `rocm/pytorch` base, `Makefile` with `make demo` running the four Master.md §10
    scenarios (single optical, single SAR, cross-modal, bi-temporal), warm-start
    preloading, request caching, cold start → first answer < 60 s, the
    "unplug the network" test.
16. Saved analyses and shareable trace links (UI is scaffolded for both).
17. Cartosat-2S / RISAT augmentation track (`DATA_ADAPTATION_PLAN §7.3`).

## Open questions nobody has answered yet

- How much GPU time is actually available before submission? This sets the corpus
  size (item 3) — 65k is ~3 days at measured throughput.
- Is DIOR-RSVG access obtainable, or is referring-expression grounding trained from
  VRSBench boxes alone?
- Will the demo run the `hf` bf16 path (adapter, 18.7 GiB) or the llama.cpp path
  (no adapter, fast start)? The current `.env` says llama.cpp. Judging story depends on it.
- The frontend has a `.vercel/` project link — is a hosted static build (mock mode)
  part of the submission?
- Is `openapi.json` schema `1.0` allowed to bump for `DELETE /v1/jobs/{id}`, or is
  it additive without a version bump?

## What *not* to do

- Do not re-run the 19-hour training with the current objective.
- Do not quantise the serving path to NF4 to "match training".
- Do not add a second implementation of labels, box format, number format, or
  index maths anywhere (including the frontend beyond the existing tested port).
- Do not let an LLM choose tools; extend `policy_table.yaml` instead.
- Do not quote the CD checkpoint as meeting the 0.88 gate, or the corpus as 65k.
- Do not commit `data/`, `runs/`, logs, or `.env`.
