//! Shared spectral-family host recovery math (WR-06).
//!
//! `SpectralEmbedding` (SPECTRAL-01) and `SpectralClustering` (SPECTRAL-02) run
//! the IDENTICAL post-eig host recovery — slice the smallest eigenvectors →
//! `/dd` (`D^-1/2` diffusion-map recovery, D-07) → `_deterministic_vector_sign_flip`
//! → transpose — differing ONLY in `drop_first`:
//!
//! - `SpectralEmbedding` (`drop_first = true`, D-08): compute the smallest
//!   `n_components + 1` eigenvectors and DROP the trivial ≈0 one (row 0), keeping
//!   rows `1..=n_components`.
//! - `SpectralClustering` (`drop_first = false`, D-11): keep ALL `n_components`
//!   smallest eigenvectors (the trivial ≈0 one is KEPT as a KMeans feature).
//!
//! The exact operation ORDER is load-bearing (RESEARCH §D-07/D-08, Pitfall 2):
//! slice smallest (ascending) → `/dd` → sign-flip → drop-first/transpose. A wrong
//! order fails the sklearn value/label match. Factoring the two former verbatim
//! copies into [`recover`] keeps the embedding and clustering paths bit-identical
//! by construction (the 09-04 file-disjointness rationale expired once both files
//! landed — they must not silently desynchronize on a future fix).
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

use std::mem::size_of;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

/// Reproduce the pinned sklearn `_spectral_embedding` recovery on the host
/// (RESEARCH §D-07/D-08). `v_host` is the DESCENDING eig `V` column-major
/// (`v_host[col*n + r] = V[r, col]`); `dd` is the length-`n` degree vector the
/// Laplacian returned. Returns the row-major `n × n_components` recovered matrix
/// (the `embedding_` for SE, the KMeans `maps` for SC).
///
/// `drop_first` selects the per-estimator slice (the ONLY real difference, WR-06):
///   - `true`  → slice the smallest `n_components + 1` eigenvectors, drop the
///     trivial ≈0 row 0, keep rows `1..=n_components` (SpectralEmbedding, D-08).
///   - `false` → slice the smallest `n_components` eigenvectors, keep ALL rows
///     `0..n_components` (SpectralClustering, D-11).
///
/// ORDER (load-bearing — a wrong order fails the value/label match):
///   1. slice the smallest `m` eigenvectors (ascending; the `r`-th smallest is
///      descending column `n - 1 - r`) into an `m × n` array;
///   2. `emb[r][i] /= dd[i]` — the `D^-1/2` recovery, BEFORE the sign flip (D-07);
///   3. `_deterministic_vector_sign_flip` per ROW (argmax|row| → sign → multiply);
///   4. keep rows per `drop_first`; transpose → row-major `n × n_components`.
pub(crate) fn recover<F>(
    v_host: &[f64],
    dd: &[f64],
    n: usize,
    n_components: usize,
    drop_first: bool,
) -> Vec<F>
where
    F: Float + CubeElement + Pod,
{
    // drop_first = TRUE (SE, D-08) needs one EXTRA eigenvector (the trivial ≈0 one
    // that is then dropped); drop_first = FALSE (SC, D-11) keeps all n_components.
    let m = if drop_first {
        n_components + 1
    } else {
        n_components
    };

    // 1. Slice the m smallest eigenvectors into an m × n row-major array. v1 eig
    //    sorts w DESCENDING, so the r-th SMALLEST eigenvector is descending column
    //    (n - 1 - r) (RESEARCH Pitfall 3 / Eig column snippet).
    let mut emb = vec![0.0f64; m * n];
    for r in 0..m {
        let col = n - 1 - r;
        for i in 0..n {
            emb[r * n + i] = v_host[col * n + i];
        }
    }

    // 2. /dd recovery (D-07) — BEFORE the sign flip. dd is the Laplacian's
    //    returned degree vector (NOT a fresh sqrt).
    for r in 0..m {
        for i in 0..n {
            emb[r * n + i] /= dd[i];
        }
    }

    // 3. _deterministic_vector_sign_flip on the m × n array (per ROW): the
    //    largest-magnitude element of each eigenvector is made positive
    //    (sklearn extmath, exact). Lowest-index tie-break on equal magnitudes.
    for r in 0..m {
        let row = &emb[r * n..(r + 1) * n];
        let mut max_idx = 0usize;
        let mut max_abs = row[0].abs();
        for (i, &val) in row.iter().enumerate().skip(1) {
            if val.abs() > max_abs {
                max_abs = val.abs();
                max_idx = i;
            }
        }
        let sign = if emb[r * n + max_idx] < 0.0 { -1.0 } else { 1.0 };
        if sign < 0.0 {
            for i in 0..n {
                emb[r * n + i] = -emb[r * n + i];
            }
        }
    }

    // 4. Keep the per-estimator rows and transpose into a row-major
    //    n × n_components matrix. drop_first = TRUE drops row 0 (kept rows are
    //    1..=n_components); drop_first = FALSE keeps rows 0..n_components.
    let row_offset = usize::from(drop_first);
    let mut out = vec![F::from_int(0i64); n * n_components];
    for c in 0..n_components {
        let r = c + row_offset;
        for i in 0..n {
            out[i * n_components + c] = f64_to_host::<F>(emb[r * n + i]);
        }
    }
    out
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side combine. Shared by the
/// two spectral estimators (WR-06 — formerly triplicated, mirrors the `eig.rs` /
/// `kernel_ridge.rs` bytemuck pair).
pub(crate) fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("spectral family is f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
pub(crate) fn f64_to_host<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("spectral family is f32/f64 only"),
    }
}
