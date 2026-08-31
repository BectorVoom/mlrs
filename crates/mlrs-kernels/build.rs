//! Invalidate the CubeCL on-disk kernel cache whenever a `#[cube]` source changes.
//!
//! # Why this exists
//!
//! `cubecl.toml` at the workspace root turns on CubeCL's on-disk compiled-kernel
//! cache (4.5x on a warm ROCm test run). That cache is keyed by
//! `KernelId::stable_hash()`, which hashes the kernel's `type_name`,
//! `address_type`, `cube_dim`, `mode` and `#[comptime]` `info` — and **not** the
//! kernel body or the generated IR.
//!
//! So editing a `#[cube]` function without renaming it, changing its generics,
//! or changing a comptime argument produces the SAME cache key, and the warm
//! cache serves the PREVIOUS binary. Measured: sabotaging `rbf_map` to
//! `F::exp(-gamma*x) + 1000` left `kernel_matrix_all_kernels_f32` PASSING
//! against a warm cache, and failing at `max_abs = 1.0000000269e3` once the
//! cache was cleared. In a repo whose daily work is editing CubeCL kernels that
//! is a silent-wrong-binary on every kernel edit, and it is what makes the cache
//! unsafe to enable on its own.
//!
//! This script closes that gap the only way the cache key allows: by making the
//! CACHE a build output of the kernel sources. Cargo re-runs it exactly when a
//! file under `src/` changes (`cargo::rerun-if-changed`), and when it does, the
//! stale cache is deleted. Kernel edit => rebuild => cache cleared => next run
//! recompiles. No edit => no re-run => the cache is kept and the win stands.
//!
//! # Why it is keyed on a CONTENT hash, not just cargo's timestamps
//!
//! Cargo re-runs a build script for reasons that are not "the kernels changed"
//! (a clean checkout, a `cargo clean`, a dependency bump, a feature change).
//! Clearing a *global* cache on each of those would throw away kernels this
//! working tree never touched — and with `cache = "global"` that directory is
//! shared with every other worktree and project on the machine. So the script
//! records a digest of the kernel sources in `OUT_DIR` and clears the cache only
//! when the digest actually MOVES. A fresh `OUT_DIR` (no digest recorded) also
//! clears, because "we cannot prove the cache matches these sources" is the same
//! situation as "it does not".
//!
//! # What it deliberately does NOT do
//!
//! It does not touch `target/autotune` (a separate cache, keyed on measured
//! timings, invalidated by its own logic) and it does not clear on a change to
//! the *host* prims in `mlrs-backend` — those are ordinary Rust, recompiled by
//! cargo, and never enter a device binary.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo::rerun-if-changed=src");
    println!("cargo::rerun-if-changed=build.rs");
    // The cache root is chosen in `cubecl.toml`; a change there can move it.
    println!("cargo::rerun-if-changed=../../cubecl.toml");
    // Escape hatch, for a bisect or a deliberate experiment against a warm cache.
    println!("cargo::rerun-if-env-changed=MLRS_KEEP_KERNEL_CACHE");

    if std::env::var_os("MLRS_KEEP_KERNEL_CACHE").is_some() {
        println!("cargo::warning=MLRS_KEEP_KERNEL_CACHE set: NOT clearing the CubeCL kernel cache. A kernel edit will be served its OLD binary.");
        return;
    }

    let digest = kernel_source_digest(Path::new("src"));
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is set for a build script"));
    let stamp = out_dir.join("kernel-sources.digest");

    // Unchanged sources: the cache on disk was built from exactly these kernels,
    // so keep it. This is the common path and the whole point of the cache.
    if fs::read_to_string(&stamp).is_ok_and(|prev| prev.trim() == digest) {
        return;
    }

    let dirs = cache_roots();
    if dirs.is_empty() {
        // Not an error — a cold tree has nothing to clear. Said out loud because
        // "found nothing" and "looked in the wrong place" are the same silence,
        // and the wrong place is exactly how this script was broken once.
        println!("cargo::warning=kernel sources changed; no CubeCL cache dir found to clear (cwd {})", std::env::current_dir().unwrap_or_default().display());
    }
    for dir in dirs {
        // `remove_dir_all` on a missing path is an error, not a no-op; a missing
        // cache is the desired end state either way.
        if dir.exists() {
            match fs::remove_dir_all(&dir) {
                Ok(()) => println!("cargo::warning=kernel sources changed: cleared CubeCL cache {}", dir.display()),
                // Never fail the build over this. A cache we could not clear is
                // reported loudly, because the consequence is a stale binary.
                Err(e) => println!("cargo::warning=COULD NOT clear CubeCL kernel cache {} ({e}). Device kernels may be STALE — delete it by hand before trusting results.", dir.display()),
            }
        }
    }

    let _ = fs::write(&stamp, &digest);
}

/// A content digest over every `.rs` under `dir`, path-qualified.
///
/// Paths are included and the map is ordered, so a RENAME (same bytes, new file)
/// and a reordering of `read_dir`'s output are both accounted for — the former
/// changes a kernel's `type_name`, which is part of the cache key, and the
/// latter would otherwise make the digest depend on filesystem iteration order.
fn kernel_source_digest(dir: &Path) -> String {
    let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    collect(dir, &mut files);

    // FNV-1a, 64-bit. A build script may not take a dependency for this, and the
    // requirement here is change-detection against an accidental edit, not
    // resistance to a crafted collision.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for (path, bytes) in &files {
        eat(path.as_bytes());
        eat(b"\0");
        eat(bytes);
        eat(b"\0");
    }
    format!("{h:016x}")
}

fn collect(dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            if let Ok(bytes) = fs::read(&path) {
                out.insert(path.to_string_lossy().into_owned(), bytes);
            }
        }
    }
}

/// Every directory a cubecl backend may have written compiled kernels into.
///
/// The subtlety that made the first version of this script a no-op: a build
/// script runs with cwd = its OWN package (`crates/mlrs-kernels/`), but the
/// cache root is resolved from the cwd of whatever process ran the kernels — for
/// `cargo test -p mlrs-backend` that is `crates/mlrs-backend/`, so
/// `CacheConfig::Target` lands in `crates/mlrs-backend/target/`. Relative paths
/// from here never reach it. Every package's `target/` has to be swept, plus the
/// workspace's own, because any of them can be a test binary's cwd.
///
/// All backend subdirectories are cleared regardless of which `CacheConfig` is
/// currently configured: the setting can change (`cubecl.toml` is a
/// `rerun-if-changed` above), and a cache left by a previous setting is exactly
/// as stale as the current one. Only the `hip`/`cuda`/`vulkan` subdirectories
/// are touched — never a `target/` or a config root itself.
fn cache_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();

    // Walk up to the workspace root (the `Cargo.toml` carrying `[workspace]`),
    // then sweep it and every member's `target/`.
    let mut dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut workspace_root = None;
    loop {
        let manifest = dir.join("Cargo.toml");
        if fs::read_to_string(&manifest).is_ok_and(|t| t.contains("[workspace]")) {
            workspace_root = Some(dir.clone());
            break;
        }
        if !dir.pop() {
            break;
        }
    }

    if let Some(ws) = workspace_root {
        roots.push(ws.join("target"));
        if let Ok(entries) = fs::read_dir(ws.join("crates")) {
            for e in entries.flatten() {
                if e.path().is_dir() {
                    roots.push(e.path().join("target"));
                }
            }
        }
    }
    // This package's own, and the cwd itself (`CacheConfig::Local`).
    roots.push(PathBuf::from("target"));
    roots.push(PathBuf::from("."));

    let mut dirs = Vec::new();
    for root in roots {
        for backend in ["hip", "cuda", "vulkan"] {
            let d = root.join(backend);
            if d.is_dir() && !dirs.contains(&d) {
                dirs.push(d);
            }
        }
    }
    dirs
}
