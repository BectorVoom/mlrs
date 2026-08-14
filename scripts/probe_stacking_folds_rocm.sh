#!/usr/bin/env bash
# The stacking fold-sweep diagnosis probe (STACK-FOLD-01) on rocm.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/rocm_env.sh
source scripts/rocm_env.sh
exec cargo test -p mlrs-algos --release --features rocm \
    --test stacking_folds_perf_test -- --ignored --nocapture "$@"
