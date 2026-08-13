#!/usr/bin/env bash
# Run the StackingRegressor python suites against the extension currently
# installed in `crates/mlrs-py/python/mlrs/`, from THIS checkout.
#
# The repo venv's `.pth` points at the MAIN checkout, so `PYTHONPATH` is set
# here to make a worktree's own python package win — without it a worktree run
# silently exercises the main checkout's copy.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/crates/mlrs-py/python"
export ROCM_PATH="${ROCM_PATH:-/home/user/rocm/opt/rocm}"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
VENV_PY="${VENV_PY:-$PWD/../../../.venv/bin/python}"
exec "$VENV_PY" -m pytest \
    crates/mlrs-py/python/tests/test_oracle_stacking.py \
    crates/mlrs-py/python/tests/test_stacking_meta_engine.py \
    "$@"
