# Session Bootstrap Prompt

Paste the block below as the first message of a new LLM / agent session working on
this repository. Adjust the last line to the task at hand.

---

You are working on **SatQuery AI** at `/home/chickenwings/SatQuery-AI` — an agentic
vision-language assistant for satellite imagery built for Smart India Hackathon
problem 26167 (ISRO/SAC). Python 3.11 via `uv`, FastAPI backend under
`src/satquery/`, React 19 + Vite + Tailwind v4 frontend under `frontend/`, QLoRA
fine-tune of Qwen3-VL-8B on an AMD RX 7900 XTX (ROCm 6.4, 24 GB).

Before doing anything, read the handoff bundle in `DOCS/AI_HANDOFF/` in this order:
`00_START_HERE.md`, `01_PROJECT_CONTEXT.md`, `02_CURRENT_STATE.md`, then
`03_ARCHITECTURE.md`, `04_REPO_MAP.md`, `05_ENVIRONMENT_AND_SETUP.md`,
`07_CONVENTIONS_AND_GOTCHAS.md`, and `08_OPEN_ISSUES_AND_NEXT_STEPS.md`. Read
`06_DATA_AND_TRAINING.md` if the task touches data or training. `09_DOC_INDEX.md`
tells you which other documents are frozen contracts (`DOCS/API_CONTRACT.md`,
`DOCS/AGENT_POLICY_DAG.md`, `DOCS/DATA_ADAPTATION_PLAN.md`) versus plans versus stale.

Hard rules you must not break:
- No number in an answer may originate from the model; deterministic tools own all
  quantities; uncited numbers are flagged, never hidden.
- No LLM chooses tools; routing lives in `configs/policy_table.yaml`.
- One source of truth for view labels (`render/view_labels.py`), box format
  (`models/prompts/box_format.py`), number formatting
  (`evidence/citation_validator.format_number`), band aliases
  (`configs/band_aliases.yaml`), index maths (`render/indices.py`).
- Index/SAR views use fixed value domains and PNG; unavailable bands are reported,
  never substituted.
- Serve the VLM in bf16 only (4-bit generation is broken on this ROCm stack);
  `source scripts/rocm_env.sh` (`HIP_VISIBLE_DEVICES=0`) in every GPU shell.
- Heavy imports (torch, transformers, torchgeo, lightning, peft, trl) stay inside
  function bodies.
- If you change the wire format: update `DOCS/API_CONTRACT.md`, the pydantic
  schemas, regenerate `openapi.json` (`scripts/export_openapi.py`) and
  `frontend/src/api/schema.d.ts` (`npm run gen:api`); both must be byte-identical
  on regeneration.
- Quality gates before claiming done: `uv run pytest -q` (503 tests),
  `uv run ruff check .`, `uv run mypy`; in `frontend/`: `npm run typecheck`,
  `npm run test` (194), `npm run build`. There is no CI — you are the CI.
- Match the existing code style: strict typing, Google docstrings citing the spec
  section, long explanatory *why* comments, specification-style test names.
- Never commit `data/`, `runs/`, `*.log`, or `.env`.

Known state you should not rediscover: Phases 0–6 and the frontend are complete and
green; the one full QLoRA run (`runs/full-epoch-v1`, 18.5k BigEarthNet-only samples)
has a flat loss curve because loss was computed over the whole sequence and there is
no zero-shot-vs-adapted ablation; Phase 8 (eval) and Phase 9 (Docker/CI/demo) are not
started; the CD checkpoint is F1 0.858 (below the 0.88 gate). The prioritised
backlog is in `08_OPEN_ISSUES_AND_NEXT_STEPS.md`.

Task: <describe the task here>.
