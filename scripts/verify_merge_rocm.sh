#!/usr/bin/env bash
# Post-merge Rust verification on rocm (STACK-GPU-ENGINE branch).
#
# rocm rather than cpu on purpose: `linear_regression_test` calls the typestate
# `Fit::fit` directly to exercise `fit_gram_eig`, and that device path costs
# ~228 s per large case on the cpu backend (cubecl-cpu's thread-per-unit + -O0
# JIT). That is pre-existing and unrelated to any change here — the PyO3 host
# arm bypasses it for real callers — but it makes the cpu Rust suite unusable as
# a fast gate. The Python suites cover cpu.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/rocm_env.sh
source scripts/rocm_env.sh
exec cargo test -p mlrs-algos -p mlrs-backend --release --features rocm \
    --test stacking_test \
    --test linear_regression_test \
    --test ridge_test \
    --test ridge_host_fit_test \
    --test ridge_classifier_test \
    --test linear_persist_test \
    --test gram_host_test \
    --test stacking_meta_test "$@"
