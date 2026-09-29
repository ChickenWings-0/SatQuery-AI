# SatQuery AI — AI Handoff Bundle

**Purpose.** These files give a fresh LLM session (or a new human contributor) every
piece of context needed to work on this repository without re-deriving it: what the
project is, why it is shaped the way it is, what is built and measured, what is still
open, and the conventions and machine gotchas that cost days to learn the first time.

**Snapshot date:** 2026-09-29 · **HEAD:** `1782163` (`check fix`) on `main`, plus an
uncommitted docs/gitignore reorganisation (all Markdown moved into `DOCS/`, `.claude/`
ignored). **Remote:** `git@github.com:ChickenWings-0/SatQuery-AI.git`.
**Repo root:** `/home/chickenwings/SatQuery-AI` · **Owner git identity:** `ChickenWings-0`.

---

## How to use this bundle

1. Read this file, then `01` → `02` in order. That is enough to hold a conversation
   about the project and to triage a request.
2. Before touching code, read `03` (architecture), `04` (repo map) and `07`
   (conventions & gotchas). Several gotchas are silent failures (wrong answers or a
   deleted torch stack, not crashes).
3. Before touching training/data, read `06` — and note that **the raw data, rendered
   views and corpus were deleted on 2026-09-29** to free disk (see `02`).
4. Before proposing work, read `08` — what is actually open.
5. `09` indexes every other document and says which are **frozen contracts**,
   **runbooks**, **design records** and **history**.
6. `SESSION_BOOTSTRAP_PROMPT.md` is a paste-ready prompt that points a new session
   at this bundle.

## Reading order

| File | What it gives you | Read when |
|---|---|---|
| `00_START_HERE.md` | this page; 60-second summary | always |
| `01_PROJECT_CONTEXT.md` | the problem, the rubric, the design thesis, decisions | always |
| `02_CURRENT_STATE.md` | what is done, measured numbers, what is on disk, known gaps | always |
| `03_ARCHITECTURE.md` | request flow, every subsystem, evidence/citation mechanism, API, frontend | before code changes |
| `04_REPO_MAP.md` | every directory with its role; what is gitignored and where artifacts live | before code changes |
| `05_ENVIRONMENT_AND_SETUP.md` | hardware, the hand-installed ROCm stack, env vars, commands | before running anything |
| `06_DATA_AND_TRAINING.md` | corpus, the v1 → v2 story, the production run, CD, export | before ML work |
| `07_CONVENTIONS_AND_GOTCHAS.md` | code style, one-source-of-truth rules, hard-won gotchas | before code changes |
| `08_OPEN_ISSUES_AND_NEXT_STEPS.md` | what is left, in priority order | before proposing work |
| `09_DOC_INDEX.md` | index of all other docs | when you need a spec |
| `SESSION_BOOTSTRAP_PROMPT.md` | paste-ready prompt for a new session | when starting a session |

## 60-second summary

SatQuery AI is an **agentic vision-language assistant for satellite imagery**, built
for **Smart India Hackathon problem 26167 (ISRO / Space Applications Centre)**. A user
uploads one or two rasters (optical, SAR, single-date or bi-temporal) — or finds them
on the Maps page via STAC — asks a question in natural language, and receives an
answer in which **every number is traced to a deterministic tool that measured it**
and every located object is a bounding box drawn on the raster. It runs offline on
one 24 GB consumer GPU (AMD RX 7900 XTX, ROCm), and a quantised build runs on an 8 GB
Windows laptop with no network.

The thesis: **the VLM is one tool among many, never the system, and it is never
allowed to invent a measurement.** A deterministic policy-table planner turns the
question into a DAG of registered tools; their scalars form a *FactSheet*; a
fine-tuned Qwen3-VL-8B writes prose from the views and the sheet; a
*CitationValidator* resolves every numeric span back to a sheet entry (key- and
unit-aware) and flags what it cannot. The whole run is an `AuditTrace`, persisted
and served as a first-class API object. A React console makes the audit visible.

**State in one line:** feature-complete; no longer an SIH entry — a personal project since 2026-09-29. Backend (623 pytest) and
frontend (356 vitest + 2 Playwright specs) green; CI, Docker, Makefile done; the
production adapter **`runs/sq-lora-v2-full/adapter`** (53,098-sample multi-source
corpus, loss-masked, answer-token accuracy 26.4 % zero-shot → 81.8 %) scores **80.7 %
accuracy / 100 % citation precision** on a 500-sample held-out benchmark; merged +
Q4_K_M GGUF exported to `models/` for the demo laptop. What remains is optional:
a recorded manual QA pass (`DOCS/FINAL_QA_CHECKLIST.md`), extra benchmark columns,
and model-quality work. See `02` and `08`.

## Non-negotiables (memorise these)

- **No number in an answer may come from the model.** Deterministic tools own all
  quantities. Uncited numbers are *flagged*, never hidden.
- **The trace is a product, not a log.** Built by the executor, schema-versioned,
  returned verbatim. Identical reruns for identical inputs.
- **No LLM ever chooses a tool.** Routing is `configs/policy_table.yaml`.
- **One source of truth for anything the model reads:** view labels
  (`render/view_labels.py`), prompt layout (`models/prompts/layout.py`), box format
  (`models/prompts/box_format.py`), number formatting
  (`evidence/citation_validator.format_number`), band aliases
  (`configs/band_aliases.yaml`). Duplicating any of these is a defect.
- **Index and SAR views use fixed value domains and PNG**, never per-image stretch
  or JPEG. Unavailable bands are reported unavailable, never substituted.
- **Serve in bf16, never NF4** — 4-bit generation is broken on this ROCm stack.
- **`HIP_VISIBLE_DEVICES=0`** in every GPU shell — the iGPU breaks ROCm enumeration.
- **Never run a bare `uv sync` / plain `uv run` on the dev box** — it deletes the
  hand-installed ROCm torch stack. Use `uv run --no-sync` or `.venv/bin/…` (`05`).
- **The frozen contract** is `DOCS/API_CONTRACT.md` → `openapi.json` →
  `frontend/src/api/schema.d.ts`. Additive changes only within schema `1.0`;
  `make contract` must stay green.
- **One branch: `main`.** `scripts/git/main_only.sh` hooks refuse anything else.
