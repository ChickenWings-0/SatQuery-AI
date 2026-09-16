# SatQuery AI — the quality gates and the demo, as one-word commands.
#
# CI (.github/workflows/ci.yml) calls these same targets, so what passes here
# passes there. Nothing below needs a GPU: the serving and training stacks are
# optional extras and every test that would touch them skips without them.
#
#   make ci        every gate, backend and frontend
#   make demo      API on :8000 and the Vite dev server on :5173, side by side
#   make demo-cpu  the same with the VLM disabled — templated answers, no weights
#   make e2e       start the API from .env, run scripts/e2e_parity.py, stop it
#   make eval      the validation benchmark: 100 held-out samples per source through the adapter
#   make eval-baseline  the same sample ids through the un-adapted base model
#
# UV defaults to `uv run --no-sync`: on the ROCm box the lockfile's CUDA torch
# is replaced by hand, and a plain `uv run` re-syncs and removes it.

SHELL := bash
.DEFAULT_GOAL := help

UV      ?= uv run --no-sync
NPM     ?= npm
FRONTEND := frontend
PYTEST_ARGS ?= -m "not gpu"
E2E_PORT ?= 8000
E2E_ARGS ?=
E2E_ENV  ?=
EVAL_NAME ?= sq-lora-v2-full
EVAL_ADAPTER ?= runs/$(EVAL_NAME)/adapter
EVAL_PER_SOURCE ?= 100
EVAL_ARGS ?=

.PHONY: help lint typecheck test contract build ci \
        backend-lint backend-typecheck backend-test backend-contract \
        frontend-install frontend-lint frontend-typecheck frontend-test frontend-contract frontend-build frontend-e2e \
        demo demo-cpu docker e2e e2e-cpu eval eval-baseline eval-self-check

help:
	@grep -E '^[a-z][a-z0-9-]*:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | column -t -s $$'\t'

# ------------------------------------------------------------------ backend

backend-lint:  ## ruff over src, scripts and tests
	$(UV) ruff check .

backend-typecheck:  ## mypy --strict over the satquery package
	$(UV) mypy

backend-test:  ## pytest, GPU-marked cases excluded (override with PYTEST_ARGS=)
	$(UV) pytest $(PYTEST_ARGS)

backend-contract:  ## openapi.json must match what create_app() produces
	$(UV) python scripts/export_openapi.py --out /tmp/satquery-openapi.json
	@diff -q openapi.json /tmp/satquery-openapi.json \
	  || (echo "openapi.json is stale: run 'uv run python scripts/export_openapi.py' and commit" && exit 1)

# ----------------------------------------------------------------- frontend

frontend-install:  ## npm ci (lockfile-exact; installs the oxlint native binding)
	cd $(FRONTEND) && $(NPM) ci

frontend-lint:  ## oxlint, warnings denied
	cd $(FRONTEND) && $(NPM) run lint

frontend-typecheck:  ## tsc -b --noEmit
	cd $(FRONTEND) && $(NPM) run typecheck

frontend-test:  ## vitest
	cd $(FRONTEND) && $(NPM) test

frontend-contract:  ## schema.d.ts must match openapi.json
	cd $(FRONTEND) && npx openapi-typescript ../openapi.json -o /tmp/satquery-schema.d.ts >/dev/null \
	  && (diff -q src/api/schema.d.ts /tmp/satquery-schema.d.ts \
	      || (echo "frontend/src/api/schema.d.ts is stale: run 'npm run gen:api' and commit" && exit 1))

frontend-build:  ## vite build
	cd $(FRONTEND) && $(NPM) run build

frontend-e2e:  ## Playwright against vite preview with the MSW mock (needs 'npx playwright install chromium')
	cd $(FRONTEND) && $(NPM) run test:e2e

# --------------------------------------------------------------- aggregates

lint: backend-lint frontend-lint  ## both linters
typecheck: backend-typecheck frontend-typecheck  ## both type checkers
test: backend-test frontend-test  ## both test suites
contract: backend-contract frontend-contract  ## both halves of the schema chain
build: frontend-build  ## production frontend bundle

ci: backend-lint backend-typecheck backend-test backend-contract \
    frontend-lint frontend-typecheck frontend-test frontend-contract frontend-build  ## everything CI runs

# --------------------------------------------------------------------- demo

demo:  ## API (:8000) + frontend dev server (:5173); Ctrl-C stops both
	@trap 'kill 0' EXIT; \
	$(UV) uvicorn satquery.api.app:app --host 127.0.0.1 --port 8000 & \
	(cd $(FRONTEND) && $(NPM) run dev) & \
	wait

demo-cpu:  ## the demo with the VLM off — no weights, templated answers
	SATQUERY_VLM_BACKEND=none $(MAKE) demo

e2e:  ## API from .env on :$(E2E_PORT), scripts/e2e_parity.py against it, then stop (E2E_ARGS= to pass flags)
	@set -euo pipefail; \
	$(E2E_ENV) $(UV) uvicorn satquery.api.app:app --host 127.0.0.1 --port $(E2E_PORT) --log-level warning & \
	server=$$!; trap 'kill $$server 2>/dev/null; wait $$server 2>/dev/null || true' EXIT; \
	$(UV) python scripts/e2e_parity.py --base-url http://127.0.0.1:$(E2E_PORT) \
	  --json runs/e2e/latest.json $(E2E_ARGS)

e2e-cpu:  ## the parity check with the VLM disabled — DAG, SSE and citations without weights
	$(MAKE) e2e E2E_ENV="SATQUERY_VLM_DISABLED=true"

# --------------------------------------------------------------------- eval

eval:  ## validation benchmark through the adapter -> runs/eval/$(EVAL_NAME)/results.md
	$(UV) python scripts/eval_benchmark.py --name $(EVAL_NAME) --adapter $(EVAL_ADAPTER) \
	  --per-source $(EVAL_PER_SOURCE) --resume $(EVAL_ARGS)

eval-baseline:  ## the base model on the same sample_ids.json -> runs/eval/base-*/results.md
	$(UV) python scripts/eval_benchmark.py --baseline --baseline-of $(EVAL_NAME) \
	  --ids runs/eval/$(EVAL_NAME)/sample_ids.json --resume $(EVAL_ARGS)

eval-self-check:  ## no model: the scorers against their own references must read 100 %
	$(UV) python scripts/eval_benchmark.py --self-check --per-source $(EVAL_PER_SOURCE) --quiet $(EVAL_ARGS)

docker:  ## build both images (backend target 'deps' only; the ROCm runtime is large)
	docker build --target deps -t satquery-api:deps .
	docker build -t satquery-web:latest $(FRONTEND)
