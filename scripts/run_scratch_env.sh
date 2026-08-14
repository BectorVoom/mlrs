#!/usr/bin/env bash
# `run_scratch.sh` with the OLD (pre-RIDGE-ARM-CAL) dispatch forced, so a
# before/after A/B runs on ONE build — the only way the two columns are
# comparable (mlrs-bench-verify-knob-is-live).
set -euo pipefail
cd "$(dirname "$0")/.."
export MLRS_RIDGE_GRAM_HOST=0
exec ./scripts/run_scratch.sh "$@"
