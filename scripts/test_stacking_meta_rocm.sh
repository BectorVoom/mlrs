#!/usr/bin/env bash
# The stacking meta-assembly kernel test on the rocm backend (STACK-META-01).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/rocm_env.sh
source scripts/rocm_env.sh
exec cargo test -p mlrs-backend --features rocm --test stacking_meta_test "$@"
