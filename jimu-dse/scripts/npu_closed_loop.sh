#!/usr/bin/env bash
# Compatibility entry point for the configurable Python closed-loop driver.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=python3
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi
exec "${PYTHON_BIN}" "${REPO_ROOT}/jimu-dse/scripts/closed_loop.py" run "$@"
