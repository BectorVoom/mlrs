#!/usr/bin/env bash
# Rebuild the rocm-backend `_mlrs` extension in place — the rocm twin of
# `build_cpu_ext.sh`.
#
# ROCm lives at a NONSTANDARD prefix on this development machine
# (`/home/user/rocm/opt/rocm`, not `/opt/rocm`), so the paths are set here
# rather than assumed; override `ROCM_PATH` in the environment for a normal
# install.
set -euo pipefail
cd "$(dirname "$0")/.."
export ROCM_PATH="${ROCM_PATH:-/home/user/rocm/opt/rocm}"
export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
cargo build -p mlrs-py --release --features rocm,extension-module "$@"
cp -f target/release/libmlrs_py.so crates/mlrs-py/python/mlrs/_mlrs.abi3.so
echo "installed -> crates/mlrs-py/python/mlrs/_mlrs.abi3.so (rocm)"
