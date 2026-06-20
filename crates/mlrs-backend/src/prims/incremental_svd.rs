//! `incremental_svd` — host-side incremental (batched) SVD merge (PRIM-07).
//!
//! Implements scikit-learn 1.7.1's `IncrementalPCA.partial_fit` merge EXACTLY
//! (RESEARCH Pattern 1 "Stacked-matrix merge") — a dense re-SVD of a small
//! stacked matrix per batch, NOT a streaming rank-update. The whole thing is
//! host glue over the Phase-3 thin [`crate::prims::svd::svd`] primitive plus the
//! Phase-2 [`column_reduce`]; ZERO new device kernel (the `[v2-P1]` decision is
//! settled).
//!
//! ## The merge (per batch, RESEARCH Pattern 1)
//! Given the running state `(components_ [k×p, Vᵀ rows], singular_values_ [k],
//! mean_ [p], var_ [p], n_samples_seen_)` and a new batch `X_batch [b×p]`:
//!   1. `col_batch_mean = column_reduce(X_batch, Mean)` — this batch's OWN mean,
//!      read BEFORE the running-stats update.
//!   2. Chan-Golub-LeVeque running update ([`incremental_mean_var`], f64) →
//!      `(col_mean, col_var, n_total)`.
//!   3. Center the batch by the UPDATED running `col_mean`.
//!   4. FIRST batch (k == 0): the stack is the centered batch alone (b×p).
//!      SUBSEQUENT: the `(k+b+1) × p` host stack
//!        rows `0..k`     = `singular_values_[i] * components_[i, :]`  (Σ·Vᵀ),
//!        rows `k..k+b`   = the centered batch,
//!        row  `k+b`      = `sqrt((n_seen·b)/n_total) · (prev_mean − col_batch_mean)`.
//!   5. Validate `(k+b+1) ≤ MAX_ROWS`, `p ≤ MAX_COLS` BEFORE the svd (ASVS V5).
//!   6. Upload the stack ONCE, run the v1 `svd` → `(U, S, Vᵀ)` (S descending).
//!   7. `align_rows` on the `Vᵀ` rows (== sklearn `svd_flip(u_based_decision=
//!      False)`) after EVERY batch (Pitfall 5).
//!   8. `explained_variance = S²/(n_total−1)` (ddof=1, Pitfall 1);
//!      `explained_variance_ratio = S²/(sum(col_var)·n_total)` (Pitfall 6).
//!   9. Keep the top `n_components`; store device-resident.
//!
//! All combine math accumulates in `f64` via the bit-cast helpers regardless of
//! `F` (Pitfall 4). The per-batch SVD scratch is released back to the pool to
//! keep the D-10 memory gate green.
//!
//! Tests live in `crates/mlrs-backend/tests/incremental_svd_test.rs` (AGENTS.md
//! §2 — no in-source `#[cfg(test)] mod tests`).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::sign_flip::align_rows;
use mlrs_core::PrimError;
use mlrs_kernels::{MAX_COLS, MAX_ROWS};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::reduce::{column_reduce, ReducePath, ScalarOp};
use crate::prims::svd::svd;
use crate::runtime::ActiveRuntime;

/// The running thin decomposition carried across `partial_fit` batches
/// (RESEARCH Pattern 1). `components_` is device-resident (`k × p`, `Vᵀ` rows,
/// row-major) per D-03; the small running statistics (`singular_values_`,
/// `mean_`, `var_`, `explained_variance_*`) are kept on the host in `f64` since
/// every batch re-reads them for the stack build and the ddof=1 / ratio finalize
/// (the same host-side discipline `pca.rs` uses for its length-`k` S pass).
pub struct IncrementalSvdState {
    /// `Vᵀ` rows (`k × n_features`, row-major), sign-aligned after every batch.
    pub components_: DeviceArray<ActiveRuntime, f64>,
    /// Top-`k` singular values, descending (length `k`).
    pub singular_values_: Vec<f64>,
    /// `S²/(n_total−1)` per retained component (ddof=1, length `k`).
    pub explained_variance_: Vec<f64>,
    /// `explained_variance_ / (sum(var_)·… )` per retained component (length `k`).
    pub explained_variance_ratio_: Vec<f64>,
    /// Running per-feature mean (length `n_features`).
    pub mean_: Vec<f64>,
    /// Running per-feature variance (length `n_features`).
    pub var_: Vec<f64>,
    /// Total samples merged so far.
    pub n_samples_seen_: usize,
    /// Number of features (columns).
    pub n_features: usize,
    /// Number of components retained.
    pub n_components: usize,
}

/// Chan-Golub-LeVeque running per-feature mean/variance/count update, host-side
/// in `f64` — the exact `sklearn.utils.extmath._incremental_mean_and_var`
/// algorithm (RESEARCH Pattern 1 step 1 / A5).
///
/// Inputs: the prior running `(last_mean, last_var, last_count)` and the new
/// batch `x_batch` (row-major `b × p`). Returns `(col_mean, col_var, n_total)`,
/// the UPDATED running statistics. The first batch passes `last_count == 0`
/// (then `last_mean`/`last_var` are ignored and the result is the batch's own
/// per-feature mean/variance).
///
/// Variance is the per-feature POPULATION variance (ddof=0, dividing the
/// unnormalized sum of squares by `n_total`) — this matches sklearn's `var_`
/// attribute and feeds the `explained_variance_ratio_` denominator.
pub fn incremental_mean_var(
    last_mean: &[f64],
    last_var: &[f64],
    last_count: usize,
    x_batch: &[f64],
    b: usize,
    p: usize,
) -> (Vec<f64>, Vec<f64>, usize) {
    let n_total = last_count + b;

    // Per-feature batch sum and sum-of-squares (single host pass).
    let mut new_sum = vec![0.0f64; p];
    for r in 0..b {
        for (c, ns) in new_sum.iter_mut().enumerate() {
            *ns += x_batch[r * p + c];
        }
    }

    if n_total == 0 {
        return (vec![0.0; p], vec![0.0; p], 0);
    }

    let mut col_mean = vec![0.0f64; p];
    let mut col_var = vec![0.0f64; p];

    for c in 0..p {
        let last_sum = last_mean[c] * last_count as f64;
        let updated_mean = (last_sum + new_sum[c]) / n_total as f64;
        col_mean[c] = updated_mean;

        // Unnormalized variances (Welford/CGL combine).
        let last_unnorm_var = last_var[c] * last_count as f64;
        // New batch unnormalized variance: Σ (x − batch_mean)².
        let batch_mean = new_sum[c] / b as f64;
        let mut new_unnorm_var = 0.0f64;
        for r in 0..b {
            let d = x_batch[r * p + c] - batch_mean;
            new_unnorm_var += d * d;
        }

        let updated_unnorm_var = if last_count == 0 {
            new_unnorm_var
        } else {
            // Correction term (CGL): (last_count/n_total · b) · (last_mean − batch_mean)².
            let last_over_new = last_count as f64 / n_total as f64;
            let mean_diff = last_mean[c] - batch_mean;
            let correction = last_over_new * b as f64 * mean_diff * mean_diff;
            last_unnorm_var + new_unnorm_var + correction
        };
        col_var[c] = updated_unnorm_var / n_total as f64;
    }

    (col_mean, col_var, n_total)
}

/// Merge one batch `x_batch` (row-major `b × p`) into the running incremental
/// decomposition `state` (RESEARCH Pattern 1), returning the updated state.
///
/// Pass `state = None` for the FIRST batch (it SVDs the centered batch alone);
/// pass `Some(prev)` to stack `[Σ·Vᵀ ; X_centered ; mean_correction]` and re-SVD.
/// `n_components` is the number of components to retain (must be `≤ min(stacked
/// rows, p)`). The stacked shape is validated against the Phase-3 SVD caps
/// (`(k+b+1) ≤ MAX_ROWS`, `p ≤ MAX_COLS`) BEFORE the svd call (ASVS V5 /
/// T-07-05); a violation returns [`PrimError::ShapeMismatch`] attributable to the
/// merge.
///
/// Generic over `F` (`f32`/`f64`) for the device upload precision; the running
/// statistics and all combine math are `f64` (Pitfall 4).
pub fn merge<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    state: Option<IncrementalSvdState>,
    x_batch: &DeviceArray<ActiveRuntime, F>,
    (b, p): (usize, usize),
    n_components: usize,
) -> Result<IncrementalSvdState, PrimError>
where
    F: Float + CubeElement + Pod,
{
    // --- Geometry guard (ASVS V5): the batch must be a well-formed b×p. ---
    if b == 0 || p == 0 || b.checked_mul(p).map(|v| v != x_batch.len()).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x_batch",
            rows: b,
            cols: p,
            len: x_batch.len(),
        });
    }

    // --- 1. col_batch_mean = column_reduce(x_batch, Mean) — this batch's OWN
    //        mean, read BEFORE the running-stats update (Pitfall 2). ---
    let col_batch_mean_dev = column_reduce::<F>(pool, x_batch, b, p, ScalarOp::Mean, ReducePath::Shared)?
        .expect("shared path is never plane-gated to None");
    let col_batch_mean: Vec<f64> = col_batch_mean_dev
        .to_host(pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    col_batch_mean_dev.release_into(pool);

    // Host copy of the batch in f64 (the combine + center + stack all run in f64).
    let x_host = x_batch.to_host(pool);
    let x64: Vec<f64> = x_host.iter().map(|&v| host_to_f64(v)).collect();

    // --- 2. Chan-Golub-LeVeque running update → (col_mean, col_var, n_total). ---
    let (prev_mean, prev_var, prev_count, prev_k, prev_components, prev_sv) = match &state {
        Some(s) => (
            s.mean_.clone(),
            s.var_.clone(),
            s.n_samples_seen_,
            s.singular_values_.len(),
            Some(s.components_.to_host(pool)),
            s.singular_values_.clone(),
        ),
        None => (vec![0.0; p], vec![0.0; p], 0, 0, None, Vec::new()),
    };

    let (col_mean, col_var, n_total) =
        incremental_mean_var(&prev_mean, &prev_var, prev_count, &x64, b, p);

    // --- 3. Center the batch by the UPDATED running col_mean. ---
    let mut x_centered = vec![0.0f64; b * p];
    for r in 0..b {
        for c in 0..p {
            x_centered[r * p + c] = x64[r * p + c] - col_mean[c];
        }
    }

    // --- 4. BRANCH: build the stacked host matrix (Pitfall 3). ---
    let k = prev_k;
    let stacked_rows = if state.is_some() { k + b + 1 } else { b };

    // --- 5. VALIDATE the stacked shape against the SVD caps BEFORE the svd
    //        call (ASVS V5 / T-07-05). svd.rs validates too, but validate here
    //        so the error is attributable to the merge. ---
    if stacked_rows > MAX_ROWS as usize || p > MAX_COLS as usize {
        return Err(PrimError::ShapeMismatch {
            operand: "incremental_svd_stack",
            rows: stacked_rows,
            cols: p,
            len: stacked_rows * p,
        });
    }

    let mut stacked = vec![0.0f64; stacked_rows * p];
    if let (Some(components_host), true) = (prev_components.as_ref(), state.is_some()) {
        // rows 0..k = singular_values_[i] * components_[i, :]  (= Σ·Vᵀ).
        for i in 0..k {
            let sv = prev_sv[i];
            for c in 0..p {
                let comp = host_to_f64(components_host[i * p + c]);
                stacked[i * p + c] = sv * comp;
            }
        }
        // rows k..k+b = the centered batch.
        for r in 0..b {
            for c in 0..p {
                stacked[(k + r) * p + c] = x_centered[r * p + c];
            }
        }
        // row k+b = sqrt((n_seen·b)/n_total) · (prev_mean − col_batch_mean).
        let scale = ((prev_count as f64 * b as f64) / n_total as f64).sqrt();
        for c in 0..p {
            stacked[(k + b) * p + c] = scale * (prev_mean[c] - col_batch_mean[c]);
        }
    } else {
        // FIRST batch: stacked = X_centered alone (b × p).
        stacked[..b * p].copy_from_slice(&x_centered[..b * p]);
    }

    // --- 6. Upload the stack ONCE, run the v1 svd (S descending). ---
    let stacked_f: Vec<F> = stacked.iter().map(|&v| f64_to_host::<F>(v)).collect();
    let stacked_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &stacked_f);
    let (u, s, vt) = svd::<F>(pool, &stacked_dev, (stacked_rows, p))?;

    let s_host = s.to_host(pool);
    let s64: Vec<f64> = s_host.iter().map(|&v| host_to_f64(v)).collect();
    let vt_host = vt.to_host(pool);
    let svd_k = s64.len(); // = min(stacked_rows, p)

    // --- 7. align_rows on the Vᵀ rows (== svd_flip u_based_decision=False),
    //        applied after EVERY batch (Pitfall 5). ---
    let vt_rows: Vec<Vec<f64>> = (0..svd_k)
        .map(|j| (0..p).map(|c| host_to_f64(vt_host[j * p + c])).collect())
        .collect();
    let vt_flipped = align_rows(&vt_rows);

    // --- 8. explained_variance = S²/(n_total−1) (ddof=1, Pitfall 1);
    //        ratio = S²/(sum(col_var)·n_total) (Pitfall 6). ---
    let denom = (n_total.saturating_sub(1)).max(1) as f64;
    let ev_all: Vec<f64> = s64.iter().map(|&sigma| (sigma * sigma) / denom).collect();
    let var_sum: f64 = col_var.iter().sum();
    let ratio_denom = if (var_sum * n_total as f64).abs() > 0.0 {
        var_sum * n_total as f64
    } else {
        1.0
    };
    let ratio_all: Vec<f64> = s64.iter().map(|&sigma| (sigma * sigma) / ratio_denom).collect();

    // --- 9. Keep the top n_components (clamped to what the SVD produced). ---
    let nc = n_components.min(svd_k);
    let mut components_host = vec![0.0f64; nc * p];
    for j in 0..nc {
        components_host[j * p..(j + 1) * p].copy_from_slice(&vt_flipped[j][..p]);
    }
    let components_dev: DeviceArray<ActiveRuntime, f64> =
        DeviceArray::from_host(pool, &components_host);

    let singular_values_ = s64[..nc].to_vec();
    let explained_variance_ = ev_all[..nc].to_vec();
    let explained_variance_ratio_ = ratio_all[..nc].to_vec();

    // --- 10. Release the SVD scratch + the uploaded stack (memory gate). ---
    u.release_into(pool);
    s.release_into(pool);
    vt.release_into(pool);
    stacked_dev.release_into(pool);
    if let Some(prev) = state {
        prev.components_.release_into(pool);
    }

    Ok(IncrementalSvdState {
        components_: components_dev,
        singular_values_,
        explained_variance_,
        explained_variance_ratio_,
        mean_: col_mean,
        var_: col_var,
        n_samples_seen_: n_total,
        n_features: p,
        n_components: nc,
    })
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side combine / finalize.
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("incremental_svd is f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
fn f64_to_host<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("incremental_svd is f32/f64 only"),
    }
}
