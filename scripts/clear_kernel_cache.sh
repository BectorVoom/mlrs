#!/usr/bin/env bash
# Clear the CubeCL on-disk compiled-kernel cache.
#
# `cubecl.toml` sets `[compilation] cache = "global"`, which lands in
# `${XDG_CONFIG_HOME:-$HOME/.config}/<backend>/<cubecl-ver>/`. A workspace
# `cargo clean` does NOT touch that, and the cache key
# (`KernelId::stable_hash()`) does not cover the kernel body — so a stale entry
# survives both a rebuild and a clean.
#
# `crates/mlrs-kernels/build.rs` clears it automatically whenever a `#[cube]`
# source changes. Run this by hand for the cases the build script cannot see:
#   * you changed a device compiler flag (HIPRTC_COMPILE_OPTIONS_APPEND, NVRTC),
#     which the cache key also ignores;
#   * you changed cubecl itself (a vendored fork, a `[patch]`, a local path dep);
#   * you are not sure, and want to prove a result against a cold cache.
set -euo pipefail
config_root="${XDG_CONFIG_HOME:-$HOME/.config}"
removed=0
for backend in hip cuda vulkan; do
    for dir in "$config_root/$backend" "target/$backend" "crates"/*/target/"$backend"; do
        if [ -d "$dir" ]; then
            rm -rf -- "$dir"
            echo "cleared $dir"
            removed=1
        fi
    done
done
[ "$removed" -eq 1 ] || echo "no CubeCL kernel cache found (already cold)"
