#!/usr/bin/env bash
# Rebuild the cpu-backend `_mlrs` extension in place (maturin's `-m` wants a
# Cargo.toml, so the pyproject template is not directly usable here).
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build -p mlrs-py --release --features cpu,extension-module "$@"
cp -f target/release/libmlrs_py.so crates/mlrs-py/python/mlrs/_mlrs.abi3.so
echo "installed -> crates/mlrs-py/python/mlrs/_mlrs.abi3.so"
