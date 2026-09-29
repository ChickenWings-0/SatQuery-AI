# Session Bootstrap Prompt

Paste the block below as the first message of a new LLM / agent session working on
this repository. Adjust the last line to the task at hand.

---

You are working on **SatQuery AI** at `/home/chickenwings/SatQuery-AI` — an agentic
vision-language assistant for satellite imagery built for Smart India Hackathon
problem 26167 (ISRO/SAC). Python 3.11 via `uv`, FastAPI backend under
`src/satquery/`, React 19 + Vite + Tailwind v4 frontend under `frontend/`, a QLoRA
fine-tune of Qwen3-VL-8B (`runs/sq-lora-v2-full/adapter`) trained on an AMD RX 7900 XTX
(ROCm 6.4, 24 GB), and a merged Q4_K_M GGUF in `models/` for an 8 GB demo laptop.

Before doing anything, read `DOCS/AI_HANDOFF/` in this order: `00_START_HERE.md`,
`01_PROJECT_CONTEXT.md`, `02_CURRENT_STATE.md`, then `03_ARCHITECTURE.md`,
`04_REPO_MAP.md`, `05_ENVIRONMENT_AND_SETUP.md`, `07_CONVENTIONS_AND_GOTCHAS.md` and
`08_OPEN_ISSUES_AND_NEXT_STEPS.md`. Read `06_DATA_AND_TRAINING.md` if the task touches
data, training or evaluation. `09_DOC_INDEX.md` lists every other document.

Hard rules you must not break:
- **Never run a bare `uv sync` or plain `uv run`** — it deletes the hand-installed ROCm
  torch stack. Use `uv run --no-sync …`, `.venv/bin/…`, or `make` targets.
- No number in an answer may originate from the model; deterministic tools own all
  quantities; uncited numbers are flagged, never hidden.
- No LLM chooses tools; routing lives in `configs/policy_table.yaml`.
- One source of truth for view labels (`render/view_labels.py`), prompt layout
  (`models/prompts/layout.py`), box format (`models/prompts/box_format.py`), number
  formatting (`evidence/citation_validator.format_number`), band aliases
  (`configs/band_aliases.yaml`), index maths (`render/indices.py`).
- Index/SAR views use fixed value domains and PNG; unavailable bands are reported,
  never substituted.
- Serve the VLM in bf16 only; `source scripts/rocm_env.sh` (`HIP_VISIBLE_DEVICES=0`) in
  every GPU shell; never run `llama-server` and the `hf` backend on one card.
- Heavy imports (torch, transformers, torchgeo, lightning, peft, trl) stay inside
  function bodies.
- Wire-format changes are additive only: update `DOCS/API_CONTRACT.md`, the pydantic
  schemas, regenerate `openapi.json` and `frontend/src/api/schema.d.ts`; `make contract`
  must be green.
- Before claiming done: `make ci` (ruff check, mypy --strict, 623 pytest, oxlint, tsc,
  356 vitest, vite build, contract). Put `~/.local/node-v22/bin` first on `PATH`.
- Match the existing style: strict typing, Google docstrings citing the spec section,
  long explanatory *why* comments, specification-style test names.
- One branch, `main`. Never commit `data/`, `runs/` (except `runs/eval/*/results.*`),
  `models/`, `*.log`, `.env`, `.claude/`, or a `sitemap.xml` that only changed its date.
- Do not delete `models/` — it is the owner's backup of the exported weights.

Known state you should not rediscover: the project is feature-complete for the SIH
final (all phases, the remediation plan, the ML recovery plan and Tracks 1–4 landed).
The v2 adapter scores 80.7 % accuracy / 48.9 % grounding R@0.5 / 100 % citation
precision on a 500-sample held-out benchmark (`runs/eval/sq-lora-v2-full/results.md`);
zero-shot → adapted answer-token accuracy is 26.4 % → 81.8 %. The raw data, rendered
views, corpus, old runs and the HF base-model cache were **deleted on 2026-09-29**, so
evaluation and training need the data pipeline rebuilt first. Open work is mainly
manual QA sign-off and laptop rehearsal (`08`).

Task: <describe the task here>.
