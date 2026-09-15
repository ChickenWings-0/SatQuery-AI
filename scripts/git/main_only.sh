#!/usr/bin/env bash
# Refuse to commit or push from any branch other than main.
#
# Installed as a pre-commit and pre-push hook via .pre-commit-config.yaml
# (`.venv/bin/pre-commit install --hook-type pre-commit --hook-type pre-push`).
# The team decision from 2026-09-15 is a single trunk: no feature branches,
# no backup branches — history lives on main and nowhere else. Set
# SATQUERY_ALLOW_BRANCH=1 to bypass deliberately (e.g. a detached HEAD bisect).
set -euo pipefail

if [[ "${SATQUERY_ALLOW_BRANCH:-0}" == "1" ]]; then
  exit 0
fi

branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
if [[ "$branch" != "main" ]]; then
  cat >&2 <<MSG
error: this repository works on 'main' only (current branch: '$branch').

  git switch main && git merge --ff-only "$branch" && git branch -d "$branch"

Set SATQUERY_ALLOW_BRANCH=1 to bypass on purpose.
MSG
  exit 1
fi
