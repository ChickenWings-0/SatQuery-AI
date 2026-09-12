# SatQuery AI — AI Handoff Bundle

**Purpose.** These files give a fresh LLM session (or a new human contributor) every
piece of context needed to work on this repository without re-deriving it: what the
project is, why it is shaped the way it is, what is actually built versus what the
docs claim, what is broken, what to do next, and the conventions and hardware gotchas
that cost days to learn the first time.

**Snapshot date:** 2026-09-11 · **HEAD:** `2cb0348` (`documentation update`) on `main`, clean.
**Repo root:** `/home/chickenwings/SatQuery-AI` · **Owner git identity:** `ChickenWings-0`.

---

## How to use this bundle

1. Read this file, then `01` → `02` in order. That is enough to hold a conversation
   about the project and to triage a request.
2. Before touching code, read `03` (architecture), `04` (repo map) and `07`
   (conventions & gotchas). The gotchas are not optional — several of them are
   silent failures (wrong answers, not crashes).
3. Before touching training/data, read `06`.
4. Before proposing work, read `08` — the prioritised list of what is actually
   wrong and what is next. Do not re-discover the flat-loss training defect; it is
   already documented there.
5. `09` indexes every other document in the repo and says which are **frozen
   contracts**, which are **plans**, and which are **stale**.
6. `SESSION_BOOTSTRAP_PROMPT.md` is a paste-ready system/user prompt that points a
   new session at this bundle.

## Reading order

| File | What it gives you | Read when |
|---|---|---|
| `00_START_HERE.md` | this page; 60-second summary | always |
| `01_PROJECT_CONTEXT.md` | the problem, the rubric, the design thesis, decisions log | always |
| `02_CURRENT_STATE.md` | what is done / partial / missing, measured numbers, known defects | always |
| `03_ARCHITECTURE.md` | request flow, every subsystem, the evidence/citation mechanism, API, frontend | before code changes |
| `04_REPO_MAP.md` | every directory and file with its role; what is gitignored and where the big artifacts live | before code changes |
| `05_ENVIRONMENT_AND_SETUP.md` | hardware, ROCm, env vars, commands, run levels, troubleshooting | before running anything |
| `06_DATA_AND_TRAINING.md` | corpus sources, what was *actually* trained on, every run, CD checkpoint, throughput | before ML work |
| `07_CONVENTIONS_AND_GOTCHAS.md` | code style, one-source-of-truth rules, the hard-won gotchas | before code changes |
| `08_OPEN_ISSUES_AND_NEXT_STEPS.md` | prioritised remediation, phase 8/9 backlog, open questions | before proposing work |
| `09_DOC_INDEX.md` | index of all other docs; frozen vs plan vs stale | when you need a spec |
| `SESSION_BOOTSTRAP_PROMPT.md` | paste-ready prompt for a new session | when starting a session |

## 60-second summary

SatQuery AI is an **agentic vision-language assistant for satellite imagery**, built
for **Smart India Hackathon problem 26167 (ISRO / Space Applications Centre)**. A user
uploads one or two rasters (optical, SAR, single-date or bi-temporal), asks a
question in natural language, and receives an answer in which **every number is
traced to a deterministic tool that measured it** and every located object is a
bounding box drawn on the raster. It runs offline on one 24 GB consumer GPU
(AMD RX 7900 XTX, ROCm).

The thesis that drives everything: **the VLM is one tool among many, never the
system, and it is never allowed to invent a measurement.** A deterministic
policy-table planner turns the question into a DAG of registered tools; their
scalars form a *FactSheet*; a fine-tuned Qwen3-VL-8B writes prose from the views and
the sheet; a *CitationValidator* then resolves every numeric span in that prose back
to a sheet entry (units must match) and flags what it cannot. The entire run —
inputs, compatibility checks, plan, executions, artifacts, fact sheet, answer,
citations, confidence with named caps — is an `AuditTrace`, persisted and served as a
first-class API object. A React mission-control frontend makes the audit visible.

**State in one line:** backend (Phases 0–6) and frontend are complete and well
tested (503 backend + 194 frontend tests, all green); the QLoRA training
infrastructure is complete but the **one full training run produced a flat loss
curve and there is no zero-shot-vs-adapted ablation**, so the "domain-adapted VLM"
claim is currently unsupported; Phase 8 (benchmark eval) and Phase 9 (Docker/CI/
demo hardening) are not started. Highest-priority fix: assistant-only loss masking,
then one honest ablation. See `02` and `08`.

## Non-negotiables (memorise these)

- **No number in an answer may come from the model.** Deterministic tools own all
  quantities. Uncited numbers are *flagged*, never hidden.
- **The trace is a product, not a log.** Built by the executor, schema-versioned,
  returned verbatim. Byte-identical reruns for identical inputs.
- **No LLM ever chooses a tool.** Routing is `configs/policy_table.yaml`.
- **One source of truth for anything the model reads:** view labels
  (`render/view_labels.py`), box serialisation (`models/prompts/box_format.py`),
  number formatting (`evidence/citation_validator.format_number`), band aliases
  (`configs/band_aliases.yaml`). Duplicating any of these strings is a defect.
- **Index and SAR views use fixed value domains and PNG**, never per-image stretch
  or JPEG. Unavailable bands are reported unavailable, never substituted.
- **Serve in bf16, never NF4** — 4-bit generation is broken on this ROCm stack.
- **`HIP_VISIBLE_DEVICES=0`** in every GPU shell — the iGPU breaks ROCm enumeration.
- **The frozen contract** is `DOCS/API_CONTRACT.md` → `openapi.json` →
  `frontend/src/api/schema.d.ts`. Changing the wire format means regenerating both
  and it must stay byte-identical on regeneration.
