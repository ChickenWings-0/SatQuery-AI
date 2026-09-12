# SatQuery AI — the quality gates and the demo, as one-word commands.
#
# CI (.github/workflows/ci.yml) calls these same targets, so what passes here
# passes there. Nothing below needs a GPU: the serving and training stacks are
# optional extras and every test that would touch them skips without them.
#
#   make ci        every gate, backend and frontend
#   make demo      API on :8000 and the Vite dev server on :5173, side by side
#   make demo-cpu  the same with the VLM disabled — templated answers, no weights

SHELL := bash
.DEFAULT_GOAL := help

UV      ?= uv
NPM     ?= npm
FRONTEND := frontend
PYTEST_ARGS ?= -m "not gpu"

.PHONY: help lint typecheck test contract build ci \
        backend-lint backend-typecheck backend-test backend-contract \
        frontend-install frontend-lint frontend-typecheck frontend-test frontend-contract frontend-build frontend-e2e \
        demo demo-cpu docker

help:
	@grep -E '^[a-z][a-z0-9-]*:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | column -t -s $$'\t'

# ------------------------------------------------------------------ backend

backend-lint:  ## ruff over src, scripts and tests
	$(UV) run ruff check .

backend-typecheck:  ## mypy --strict over the satquery package
	$(UV) run mypy

backend-test:  ## pytest, GPU-marked cases excluded (override with PYTEST_ARGS=)
	$(UV) run pytest $(PYTEST_ARGS)

backend-contract:  ## openapi.json must match what create_app() produces
	$(UV) run python scripts/export_openapi.py --out /tmp/satquery-openapi.json
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
	$(UV) run uvicorn satquery.api.app:app --host 127.0.0.1 --port 8000 & \
	(cd $(FRONTEND) && $(NPM) run dev) & \
	wait

demo-cpu:  ## the demo with the VLM off — no weights, templated answers
	SATQUERY_VLM_BACKEND=none $(MAKE) demo

docker:  ## build both images (backend target 'deps' only; the ROCm runtime is large)
	docker build --target deps -t satquery-api:deps .
	docker build -t satquery-web:latest $(FRONTEND)
