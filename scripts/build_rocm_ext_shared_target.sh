#!/usr/bin/env bash
# `build_rocm_ext.sh`, but building into a CARGO_TARGET_DIR that may be shared
# with another checkout — and copying the freshly-built `.so` out of THAT
# directory in the same command.
#
# Sharing a target dir with a parallel worktree reuses the compiled dependency
# graph (minutes per build), and cargo's own lock makes concurrent invocations
# safe. What is NOT safe is building here and copying later: `libmlrs_py.so`
# under a shared target dir is a single file, so another session's build
# between the two steps would install ITS binary into this checkout. Build and
# copy are therefore one step, here.
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/home/user/Documents/workspace/mlrs/target}"
export ROCM_PATH="${ROCM_PATH:-/home/user/rocm/opt/rocm}"
export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
cargo build -p mlrs-py --release --features rocm,extension-module "$@"
cp -f "$CARGO_TARGET_DIR/release/libmlrs_py.so" crates/mlrs-py/python/mlrs/_mlrs.abi3.so
echo "installed -> crates/mlrs-py/python/mlrs/_mlrs.abi3.so (rocm, from $CARGO_TARGET_DIR)"
