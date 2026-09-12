# SatQuery AI

*From Space to Answers — geospatial intelligence that shows its work.*

An agentic vision-language assistant for satellite imagery, built for Smart India
Hackathon problem statement 26167 (ISRO / Space Applications Centre). A query is
parsed, the inputs are validated for compatibility, a deterministic policy table
selects and sequences specialist tools from a registry, and the answer that comes
back is **evidence-grounded** — every number is bound to a measured scalar or
flagged as uncited — with a **versioned, schema-validated audit trace** of what ran.

The VLM (Qwen3-VL-8B with a QLoRA adapter) is one tool among many, never the system.
Runs fully local on one 24 GB consumer GPU and survives the network being pulled.

## What it does

| Capability | How |
|---|---|
| VQA, captioning, grounding (boxes) | `vlm_*` tools over pre-rendered views, boxes validated as positions |
| Bi-temporal change (VQA + change maps) | Siamese CD model + change statistics; views co-registered at ingest |
| Cross-modal optical + SAR | Spectral-index and backscatter analysers feed one FactSheet |
| Input validation / compatibility | 11 named checks (CRS, GSD, overlap, co-registration …) before any tool runs |
| Citation honesty | `CitationValidator`: unit-aware, key-aware, `flag`-not-`strip` by default |
| Audit trace | `AuditTrace` built by the executor, persisted to SQLite, returned verbatim |

## Quick start

```bash
# Python 3.11 via uv; Node >= 22.12 (see frontend/.nvmrc)
uv sync --group dev                       # API, tools without checkpoints, the whole test suite
cd frontend && npm ci && cd ..

make demo                                 # API on :8000, frontend on :5173
make demo-cpu                             # same, VLM disabled (templated answers, no weights)
```

Open <http://localhost:5173>. Without a backend, <http://localhost:5173/?mock=1> runs
the UI against recorded fixtures.

Serving the fine-tuned model needs the optional extras and a ROCm/CUDA torch:

```bash
uv sync --extra vlm                       # + torch/transformers/accelerate
source scripts/rocm_env.sh                # every shell that touches the GPU
export SATQUERY_VLM_ADAPTER_PATH=runs/full-epoch-v1/adapter
```

> On a ROCm box the lockfile's CUDA torch is replaced by hand — see
> `DOCS/AI_HANDOFF/05_ENVIRONMENT_AND_SETUP.md`. **Do not run a bare `uv sync`
> in that environment**: it removes the hand-installed stack.

## Quality gates

```bash
make ci          # everything CI runs: ruff, mypy --strict, pytest, oxlint, tsc, vitest, vite build,
                 # and the two halves of the contract chain (openapi.json <-> schema.d.ts)
make test        # both test suites only
make contract    # openapi.json and frontend/src/api/schema.d.ts are in sync
```

`.github/workflows/ci.yml` runs the same targets on every push and PR. Install the
pre-commit hooks with `uv run pre-commit install`.

## Layout

```
src/satquery/        the package: api/ agent/ evidence/ ingest/ render/ tools/ models/ trace/ training/
scripts/             render passes, corpus build, training, export, smoke tests
training/            data builders and configs for the QLoRA and CD runs
frontend/            React 19 + Vite + Tailwind v4 mission-control UI, typed from openapi.json
tests/               hermetic: synthetic rasters, no network, torch only where it is under test
DOCS/                Master.md (plan) · API_CONTRACT.md (frozen 1.0) · AGENT_POLICY_DAG.md ·
                     project_audit.md · remediation_plan.md · AI_HANDOFF/ (start here)
configs/             registry.yaml, policy_table.yaml, training profiles
```

## Data and training

The corpus is built from BigEarthNet-v2 (mandated, 10 m) plus VHR sources
(VRSBench, CDVQA, RSVQA-HR) — see `DOCS/DATA_ADAPTATION_PLAN.md` and
`scripts/build_corpus.py`. Every source is image-hashed and deduplicated against the
quarantined benchmark test splits; a leak fails the build.

**DIOR-RSVG is gated.** To include it: accept the terms at
<https://huggingface.co/datasets/danielz01/DIOR-RSVG>, `export HF_TOKEN=hf_...`, then
re-run `./run_overnight.sh --stage 1`. Until then the build excludes it and says so.

The current adapter (`runs/full-epoch-v1`) and its known limitations are documented
honestly in `DOCS/project_audit.md` §2 and `DOCS/AI_HANDOFF/06_DATA_AND_TRAINING.md`.

## Where to read next

1. `DOCS/AI_HANDOFF/00_START_HERE.md` — the 60-second summary and reading order.
2. `DOCS/API_CONTRACT.md` — the frozen wire contract (additive changes only).
3. `DOCS/remediation_plan.md` — what is being fixed and in which order.
