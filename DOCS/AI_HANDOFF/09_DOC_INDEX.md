# 09 — Index of All Documentation in the Repo

Status legend: **FROZEN** = a contract; code must match it. **CURRENT** = describes the
tree as of 2026-09-18 or later. **RUNBOOK** = steps to follow. **DESIGN RECORD** = a
plan that has been executed; read for the *why*, not for status. Everything lives in
`DOCS/` except `README.md` and `PRODUCT.md`, which must stay at the root.

## Root

| File | Status | What it contains |
|---|---|---|
| `README.md` | CURRENT | public README: current model, capabilities, quick start, gates, layout, one-branch policy |
| `PRODUCT.md` | CURRENT | product brief (users, modes, what "done" means) read by the impeccable design plugin |

## Contracts

| File | Status | What it contains |
|---|---|---|
| `API_CONTRACT.md` | **FROZEN** 1.0 (additive only) | wire contract: enums, core objects, endpoints (incl. imagery fetch, job cancel), SSE protocol, error taxonomy, examples |
| `AGENT_POLICY_DAG.md` | **FROZEN** | routing key, classification cascade, slot schemas, policy table, capability matching, executor semantics, validator and confidence rules |
| `DATA_ADAPTATION_PLAN.md` | **FROZEN** spec / executed | view catalogue, band aliases, label format, unified sample schema, source builders, composition, training profiles, eval protocol |

## Using and running it

| File | Status | What it contains |
|---|---|---|
| `USER_GUIDE.md` | CURRENT (2026-09-18) | for judges, testers, developers: local-first philosophy, sourcing imagery, every UI feature, shortcuts, troubleshooting |
| `SETUP_GUIDE.md` | CURRENT | levels A (frontend mock) / B (API, no VLM) / C (full stack), llama.cpp path, rebuilding data and training |
| `DEMO_LAPTOP_RUNBOOK.md` | RUNBOOK | air-gapped Windows RTX 4070 laptop: Q4_K_M GGUF via `llama-server`, API, built frontend, off one USB stick |
| `FINAL_QA_CHECKLIST.md` | RUNBOOK (sign-off empty) | manual QA before the final, per network tier, with a sign-off sheet |
| `HACKATHON_PITCH.md` | CURRENT | the SIH narrative, value props, demo script, measured numbers |

## Architecture and design

| File | Status | What it contains |
|---|---|---|
| `ARCHITECTURE.md` | CURRENT | long-form technical deep-dive: request flow, data engine, inference layer, frontend |
| `Master.md` | DESIGN RECORD (foundational) | original master plan and 10-phase roadmap; SIH framing; hidden-eval gap; risk register. **§2 environment facts are pre-migration** |
| `ML_PIPELINE_RECOVERY_PLAN.md` | DESIGN RECORD | why v1 learned nothing and exactly what v2 changed (loss masking, shared layout, fingerprint, probes); read before any retrain |
| `ROADMAP_REMAINING_FIXES.md` | DESIGN RECORD | the four pre-final tracks (eval suite, merge/GGUF, e2e parity, UI) with their verification steps — some steps still open (`08`) |
| `UI_TRACK_4_ARCHITECTURE.md` | DESIGN RECORD | SITREP / GeoJSON / STAC component tree, state, bundle boundaries, offline behaviour |
| `frontend_blueprint.md` | DESIGN RECORD | app shell, sections, tokens, light theme, production-polish checklist |
| `landing_page_blueprint.md` | DESIGN RECORD | the `/` page, display face, 3D globe |
| `OPEN_SOURCE_ASSETS.md` | reference | external models/datasets/libraries with their Hub/GitHub ids |

## Asset notes (moved from the folders they describe)

| File | Describes |
|---|---|
| `FONTS.md` | `frontend/public/fonts/` — provenance, licences, `npm run fonts:sync` |
| `SAMPLE_SCENES.md` | `frontend/public/samples/` — Use Cases gallery layout |
| `STAC_FIXTURES.md` | `frontend/public/samples/stac/` — recorded `?mock=1` discovery fixtures and the re-record command |

## `DOCS/AI_HANDOFF/` — this bundle

`00`–`09` plus `SESSION_BOOTSTRAP_PROMPT.md`, rewritten 2026-09-29. Amend whenever the
state, the measured numbers or the open-issue list changes.

## Deleted on 2026-09-29 (in git history if ever needed)

`project_audit.md` (2026-09-11 audit of `77bea25`; every finding since fixed),
`remediation_plan.md` (executed 2026-09-12), `FRONTEND_ROADMAP.md`,
`FRONTEND_AUDIT.md`, `FRONTEND PLAN/` (superseded frontend plans),
`landing_page_revamp_plan.md` (implemented 2026-09-13), `PROJECT_OVERVIEW.md` (the
old README, replaced by the root `README.md`), `FRONTEND_VITE_TEMPLATE.md` (Vite
boilerplate).

## Other in-repo documentation

| Location | Content |
|---|---|
| module docstrings across `src/satquery/` | each cites its spec section; executor, hf_backend, corpus_builder, qlora, catalog carry the real design rationale |
| `configs/*.yaml` header comments | why each table exists and what it protects |
| `scripts/*.py` docstrings | usage and the failure each script guards against |
| `runs/sq-lora-v2-full/run_manifest.json`, `mask_audit.json`, `train.log` | exact profile, plan, VRAM, zero-shot and final metrics |
| `runs/eval/sq-lora-v2-full/results.md` | the benchmark table for the slides |
| `models/sq-lora-v2-full-merged/merge_manifest.json` | adapter SHA, GGUF files, laptop serving settings |
| `data/checkpoints/cd/levircd_resnet18.ckpt.json` | CD metrics, threshold, provenance, normalisation note, self-test |
| `openapi.json` | the machine-readable contract |
