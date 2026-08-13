#!/usr/bin/env bash
# Run the meta-assembly ladder against the extension currently installed in
# `crates/mlrs-py/python/mlrs/`, from THIS checkout (see run_stacking_tests.sh
# for why PYTHONPATH is set here).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/crates/mlrs-py/python"
export ROCM_PATH="${ROCM_PATH:-/home/user/rocm/opt/rocm}"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
VENV_PY="${VENV_PY:-$PWD/../../../.venv/bin/python}"
exec "$VENV_PY" scripts/bench_stacking_meta.py "$@"
