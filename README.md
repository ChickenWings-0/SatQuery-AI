# SatQuery AI

*From Space to Answers — geospatial intelligence that shows its work.*

An agentic vision-language assistant for satellite imagery. It started as an entry
for Smart India Hackathon problem statement 26167 (ISRO / Space Applications Centre)
and is now a personal project. A query is
parsed, the inputs are validated for compatibility, a deterministic policy table
selects and sequences specialist tools from a registry, and the answer that comes
back is **evidence-grounded** — every number is bound to a measured scalar or
flagged as uncited — with a **versioned, schema-validated audit trace** of what ran.

The VLM (Qwen3-VL-8B with a QLoRA adapter) is one tool among many, never the system.
Runs fully local on one 24 GB consumer GPU and survives the network being pulled.

## Current model

| | `runs/sq-lora-v2-full/adapter` (2026-09-15) |
|---|---|
| Base | `Qwen/Qwen3-VL-8B-Instruct` · NF4 backbone during training · LoRA r=16, α=32 on all LLM projections + last 8 vision blocks |
| Corpus | 53,098 train / 5,902 val from BigEarthNet-v2, VRSBench, RSVQA-HR, CDVQA and Evidence QA (`data/processed/corpus/v2-full/`) |
| Epoch 1.0 | train loss 0.3025 · eval loss 0.1867 · answer-token accuracy 81.76 % (zero-shot base: 26.4 %) |
| Benchmark | 500 held-out samples: 80.7 % accuracy · 48.9 % grounding R@0.5 · 100 % citation precision · 0 % uncited numbers ([`runs/eval/sq-lora-v2-full/results.md`](runs/eval/sq-lora-v2-full/results.md)) |
| Served as | bf16 base + adapter through the transformers backend, or merged → Q4_K_M GGUF (`models/`, 5.0 GB + 1.2 GB mmproj) through `llama-server` |

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
uv sync --extra vlm                       # + torch/transformers/accelerate (first time only)
source scripts/rocm_env.sh                # every shell that touches the GPU
cp .env.example .env                      # SATQUERY_* settings, read by pydantic-settings
export SATQUERY_VLM_BACKEND=hf
export SATQUERY_VLM_ADAPTER_PATH=runs/sq-lora-v2-full/adapter
```

The offline path — a Q4_K_M GGUF behind `scripts/serve_vlm.sh` (Linux) or
`scripts/serve_vlm.ps1` (Windows) with `SATQUERY_VLM_BACKEND=llamacpp` — is the
same interface and needs no Python ML stack at all; do not run both at once on a
24 GB card. `scripts/merge_export.py` produces the GGUF pair from the adapter,
and `DOCS/DEMO_LAPTOP_RUNBOOK.md` is the runbook for the air-gapped 8 GB
demo laptop (`SATQUERY_VLM_MAX_VIEWS=3`, 8192 context, q8_0 KV cache).

> On a ROCm box the lockfile's CUDA torch is replaced by hand — see
> `DOCS/AI_HANDOFF/05_ENVIRONMENT_AND_SETUP.md`. **Do not run a bare `uv sync`
> or a plain `uv run` in that environment**: both re-sync the lockfile and
> remove the hand-installed stack. Use `uv run --no-sync ...` or `.venv/bin/...`.

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
src/satquery/        the package: api/ agent/ eval/ evidence/ ingest/ render/ tools/ models/ trace/ training/
scripts/             render passes, corpus build, training, merge/export, serving, smoke tests, git/ policy
training/            data builders and configs for the QLoRA and CD runs
runs/                training and eval outputs (git-ignored except runs/eval/*/results.*)
frontend/            React 19 + Vite + Tailwind v4 console: landing, console, maps, projects, saved, report
tests/               hermetic: synthetic rasters, no network, torch only where it is under test
DOCS/                all documentation — AI_HANDOFF/ (start here), API_CONTRACT.md (frozen 1.0),
                     USER_GUIDE.md, SETUP_GUIDE.md, DEMO_LAPTOP_RUNBOOK.md, FINAL_QA_CHECKLIST.md
configs/             registry.yaml, policy_table.yaml, training profiles
models/              merged + GGUF exports of the current adapter (git-ignored)
PRODUCT.md           who this is for and what "done" means for the demo
```

## Data and training

The corpus is built from BigEarthNet-v2 (mandated, 10 m) plus VHR sources
(VRSBench, CDVQA, RSVQA-HR) — see `DOCS/DATA_ADAPTATION_PLAN.md` and
`scripts/build_corpus.py`. Every source is image-hashed and deduplicated against the
quarantined benchmark test splits; a leak fails the build.

**DIOR-RSVG is gated.** To include it: accept the terms at
<https://huggingface.co/datasets/danielz01/DIOR-RSVG>, `export HF_TOKEN=hf_...`, then
re-run `./run_overnight.sh --stage 1`. Until then the build excludes it and says so.

The v2 corpus is rebuilt end-to-end by `scripts/rebuild_corpus_v2.sh`; the
training profile is `configs/train/qlora_qwen3vl8b_rocm24g.yaml` and the run is
launched with `scripts/train_vlm.py`. `scripts/preflight_train_serve_parity.py`
proves the serving prompt is byte-identical to the training prompt before any
adapter is trusted. Why the first adapter (`full-epoch-v1`) learned nothing and
what changed for v2 is recorded in `DOCS/ML_PIPELINE_RECOVERY_PLAN.md` and
`DOCS/AI_HANDOFF/06_DATA_AND_TRAINING.md`.

> The raw datasets, rendered views and corpus were deleted from the dev box on
> 2026-09-29 to free disk. Rebuild them (`06_DATA_AND_TRAINING.md`, last section)
> before running `make eval` or training again.

## Contributing

**This repository has one branch: `main`.** No feature, backup or experiment
branches — work lands on `main` in small, reviewed commits, and every clone is
expected to be fast-forwardable to `origin/main`. `scripts/git/main_only.sh` is
installed as a `pre-commit` and `pre-push` hook (also declared in
`.pre-commit-config.yaml`) and refuses to commit or push from any other branch;
`SATQUERY_ALLOW_BRANCH=1` bypasses it for a deliberate detached-HEAD bisect.

```bash
git switch main && git pull --ff-only
# ... work ...
make ci && git add -A && git commit && git push origin main
```

## Where to read next

1. `DOCS/AI_HANDOFF/00_START_HERE.md` — the 60-second summary and reading order.
2. `DOCS/USER_GUIDE.md` — every feature of the console, Maps and exports, and troubleshooting.
3. `DOCS/FINAL_QA_CHECKLIST.md` and `DOCS/DEMO_LAPTOP_RUNBOOK.md` — the pre-final QA pass
   and the air-gapped laptop setup.
4. `DOCS/API_CONTRACT.md` — the frozen wire contract (additive changes only).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). Third-party fonts, Copernicus Sentinel
imagery and OpenStreetMap fixtures keep their own terms, listed in [`NOTICE`](NOTICE).
Model weights are not in this repository.
