# 09 — Index of All Documentation in the Repo

Status legend: **FROZEN** = a contract; code must match it. **PLAN** = intended
design; partly implemented; read with `02` in hand. **CURRENT** = written 2026-09-11
against the current code. **STALE** = superseded; useful for history only.

## `DOCS/`

| File | Size | Status | What it contains |
|---|---|---|---|
| `Master.md` | 41 KB | PLAN (foundational) | the master architecture & 10-phase roadmap; SIH problem framing; hidden-eval gap; hardware constraints (**§2 environment facts are pre-migration and stale**); schema sketches; tool registry target; model/library selection; risk register; end-to-end verification (`make demo` four scenarios). Each phase ends with the "next prompt" that was used to drive the work |
| `API_CONTRACT.md` | 28 KB | **FROZEN** | the wire contract, schema 1.0: conventions, all enums (§2), core objects (§3: InputManifest, CheckResult, CompatibilityReport, ArtifactRef, InputContract, ToolSpec, PlanStep, Execution, Answer, Confidence, AuditTrace), endpoints (§4), **SSE protocol (§5)**, **error taxonomy (§6)**, four worked examples (§7), frontend integration notes (§8), deltas from Master (§9) |
| `AGENT_POLICY_DAG.md` | 26 KB | **FROZEN** | routing key; 4-stage classification cascade with the exact regex table and priorities; slot schemas; the full policy table; capability matching; executor semantics (scheduling, cache key, status propagation, FactSheet merge); CitationValidator and `weighted_tool_agreement_v1` rules; determinism guarantees; Phase 3 golden tests |
| `DATA_ADAPTATION_PLAN.md` | 29 KB | **FROZEN** (spec) / PLAN (execution) | the frozen view catalogue, band alias map, normalisation, selection policy and label format (§2); unified sample schema (§3); per-source builders incl. the known loader failures (§4); 65k composition (§5); training Profiles A/B and OOM playbook (§6); run schedule and pre-rendering (§7); evaluation & ablation protocol (§8); disk/time budgets (§9) |
| `OPEN_SOURCE_ASSETS.md` | 2 KB | reference | the external models/datasets/libs with their Hub/GitHub ids: Qwen3-VL-8B, GeoChat (reference), BigEarthNet.txt, VRSBench, CDVQA, LEVIR-CD, OSCD, DOFA, Open-CD/UCD, TorchGeo, rasterio |
| `project_audit.md` | 28 KB | **CURRENT** (2026-09-11, HEAD 77bea25) | independent scorecard (Data 7, ML 4, Backend 8, Frontend 8, Testing 8 → overall 7/10) with measured evidence, per-area wins and "brutal truths", plan-vs-reality table, prioritised remediation. **The single most useful document for triage.** |
| `FRONTEND_ROADMAP.md` | 31 KB | PLAN (executed) | the frontend build plan: Phase B backend prerequisites (jobs + SSE + traces) and F0–F6 frontend phases; stack; verification |
| `FRONTEND_AUDIT.md` | 14 KB | CURRENT-ish (2026-09-07) | frontend technical audit with P0–P3 findings and health score; predates the mission-control overhaul |
| `FRONTEND PLAN/frontendplan.md` | — | STALE | first mission-control overhaul plan |
| `FRONTEND PLAN/updatedfrontendplan.md` | — | PLAN (executed) | v2 of the overhaul: token swap, layout shell, centre workspace, right sidebar, assets, polish; execution order and verification |

## `repo documentation/` (all CURRENT, 2026-09-11)

| File | What it contains |
|---|---|
| `README.md` | the public-facing README: problem, how it answers, features, tech stack table, repo map, status. **Overstates the corpus (65k) and training run** — see `02` |
| `ARCHITECTURE.md` | the technical deep-dive: request flow; Data Engine (sources, dedup, profile); Inference layer (backends, bf16 memory safety, citation interceptor, confidence, API); Frontend (layout, state, citations, bbox overlay, design system tests). Accurate to code; file paths given |
| `SETUP_GUIDE.md` | levels A/B/C setup, commands, env, training reproduction, troubleshooting table |
| `HACKATHON_PITCH.md` | the SIH narrative: three value props (deterministic facts vs estimates, spatial grounding, single-GPU offline), demo script in order, "why more than a wrapper", what's next |

## `DOCS/AI_HANDOFF/` — this bundle (CURRENT)

`00`–`09` plus `SESSION_BOOTSTRAP_PROMPT.md`. Regenerate or amend whenever the phase
status, the measured numbers, or the open-issue list changes.

## Other in-repo documentation

| Location | Content |
|---|---|
| module docstrings across `src/satquery/` | each cites its spec section; several (executor, hf_backend, corpus_builder, qlora, catalog, registry.yaml, policy_table.yaml, train YAMLs) carry long rationale comments that are the real design record |
| `configs/*.yaml` header comments | why each table exists and the invariants it protects |
| `scripts/*.py` module docstrings | usage and the failure each script guards against |
| `runs/*/run_manifest.json` | exact profile, plan, VRAM estimate, peak, resolved target modules per run |
| `runs/*/README.md` | trl auto-generated model cards (framework versions are the useful part) |
| `data/checkpoints/cd/levircd_resnet18.ckpt.json` | CD metrics, threshold, provenance, the normalisation bug note, self-test |
| `logs/overnight-*.log` | the overnight playbook runs; the last one ends with the open items that shaped the corpus decision |
| `frontend/README.md` | Vite template boilerplate — not project docs |
| `frontend/public/fonts/README.md` | font provenance and `fonts:sync` |
| `openapi.json` | the machine-readable contract (68 KB) |
