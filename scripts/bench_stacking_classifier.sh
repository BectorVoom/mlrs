#!/usr/bin/env bash
# Run `bench_stacking_classifier.py` against THIS checkout's python package and
# the extension currently installed in `crates/mlrs-py/python/mlrs/`.
#
# The repo venv's `.pth` points at the MAIN checkout, so `PYTHONPATH` is set
# here to make a worktree's own package win — without it a worktree run
# silently benchmarks the main checkout's copy.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/crates/mlrs-py/python"
export ROCM_PATH="${ROCM_PATH:-/home/user/rocm/opt/rocm}"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
VENV_PY="${VENV_PY:-$PWD/../../../.venv/bin/python}"
exec "$VENV_PY" scripts/bench_stacking_classifier.py "$@"
