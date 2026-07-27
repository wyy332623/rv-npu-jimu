#!/usr/bin/env bash
# Compatibility entry point for the configurable Python closed-loop driver.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec python3 "${REPO_ROOT}/jimu-dse/scripts/closed_loop.py" run "$@"
