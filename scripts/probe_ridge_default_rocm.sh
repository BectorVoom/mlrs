#!/usr/bin/env bash
# `Ridge()` default-arm ladder + the forced host/device A/B, on rocm.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/rocm_env.sh
source scripts/rocm_env.sh
exec cargo test -p mlrs-algos --release --features rocm \
    --test ridge_default_perf_test -- --ignored --nocapture "$@"
