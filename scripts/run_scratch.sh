#!/usr/bin/env bash
# Run a python script against THIS checkout's mlrs package and the installed
# extension (see run_stacking_tests.sh for why PYTHONPATH is set here).
#
#   scripts/run_scratch.sh /path/to/script.py [args...]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/crates/mlrs-py/python"
export ROCM_PATH="${ROCM_PATH:-/home/user/rocm/opt/rocm}"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
VENV_PY="${VENV_PY:-$PWD/../../../.venv/bin/python}"
exec "$VENV_PY" "$@"
