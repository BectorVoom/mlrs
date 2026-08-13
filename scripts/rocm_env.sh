#!/usr/bin/env bash
# Source-able ROCm environment for this development machine, whose ROCm 7.x
# install lives at a NONSTANDARD prefix (`/home/user/rocm/opt/rocm`, not
# `/opt/rocm`). Override `ROCM_PATH` for a normal install.
#
#   source scripts/rocm_env.sh && cargo test --features rocm ...
export ROCM_PATH="${ROCM_PATH:-/home/user/rocm/opt/rocm}"
export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
