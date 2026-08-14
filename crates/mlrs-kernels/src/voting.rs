//! Prediction-voting kernels (VOTE-01) — the device arm of
//! `mlrs.VotingRegressor`'s `_predict` and `transform`.
//!
//! A voting regressor asks each of its `k` fitted members for a length-`n`
//! prediction column and then does two things with them:
//!
//! ```text
//!   transform(X) -> mat (n × k)   mat[r, j] = predⱼ[r]
//!   predict(X)   -> avg (n)       avg[r]    = (Σⱼ predⱼ[r]·wⱼ) / (Σⱼ wⱼ)
//! ```
//!
//! Unlike stacking's meta-matrix assembly (`crate::stacking`), which is a pure
//! strided copy, `predict` **reduces**: it reads `n·k` and writes `n`. That is
//! the one structural reason a device arm is worth having here at all — the
//! download shrinks by a factor of `k` — and it is why this module exists as its
//! own kernel set rather than reusing the scatter.
//!
//! ## Why `predict` never materialises the `n × k` matrix
//!
//! sklearn builds the matrix first (`np.asarray([est.predict(X) …]).T`) and
//! averages it. Doing that on the device would cost an `n · k` intermediate
//! buffer AND make every unit read with stride `k` — the columns of a row-major
//! `n × k` matrix are exactly the wrong layout for a per-row reduction of
//! adjacent units. The members' predictions arrive as `k` SEPARATE `n`-long
//! columns, so the accumulation runs over those instead:
//! [`vote_init_weighted`] once, then [`vote_add_weighted`] `k − 1` times, then
//! [`vote_divide`]. Every access is contiguous, and the only device buffer is
//! the `n`-long accumulator.
//!
//! One launch per member rather than one launch total is forced by CubeCL:
//! `Array` arguments are fixed at kernel-definition time, so a kernel cannot
//! take a caller-chosen `k` of them (the same constraint `stack_meta_block`
//! documents). The launches are read-after-write on `acc` and therefore rely on
//! stream ordering — they are submitted to the one client stream in order, which
//! is the same guarantee every iterative mlrs prim (`sgd`, `kmeans`) already
//! depends on.
//!
//! ## Summation order is part of the contract, not an implementation detail
//!
//! `np.average(a, axis=1, weights=w)` evaluates `np.multiply(a, w).sum(axis=1)`
//! and then DIVIDES by `w.sum()`. Both halves are reproduced exactly:
//!
//! * the product is formed per element (`predⱼ[r] · wⱼ`) and accumulated
//!   left-to-right in member order, which is numpy's own order for this
//!   reduction — the `k` axis of an `(n, k)` array numpy built by transposing
//!   `(k, n)` is strided, so its pairwise blocking above 8 elements does not
//!   apply. Measured exact at `k = 1, 2, 5, 9, 16` rather than assumed
//!   (`test_many_members_still_match_numpys_reduction_exactly`);
//! * [`vote_divide`] divides by the weight sum rather than multiplying by its
//!   reciprocal, because `x / s` and `x * (1/s)` are different floating-point
//!   numbers and the oracle compares against numpy bit for bit where it can.
//!
//! `vote_init_weighted` exists so the accumulator starts from `pred₀ · w₀`
//! rather than from a zeroed buffer: `0 + (−0.0)` is `+0.0`, so seeding with a
//! zero would silently change the sign of an all-negative-zero row, and a pool
//! handle is not zero-initialised in the first place.
//!
//! ## The one thing the device CANNOT be held to: FMA contraction
//!
//! Reproducing numpy's order gets the device arm to within **one ULP**, not to
//! bit equality, and the gap is not a bug in this file. `acc + pred·w` is the
//! canonical fused-multiply-add shape, and a GPU backend contracts it into a
//! single FMA instruction — one rounding where numpy performs two. Measured on
//! rocm (gfx1151, f32): every value within 1 ULP of numpy's, some of them
//! exactly equal and some one step away. cubecl exposes no per-kernel
//! `fp-contract` control, and the cpu backend (LLVM at `-O0`) does not contract
//! at all, so the same source is bit-exact there and 1 ULP off on a real GPU.
//!
//! The consequence is a documented, tested asymmetry rather than a hidden one:
//! the `numpy` and `host` arms are bit-identical to `np.average`, the `device`
//! arm is within a few ULP of it — comfortably inside mlrs's 1e-5 contract, and
//! *more* accurate than the reference rather than less. `transform` has no
//! arithmetic to contract and stays bit-exact on every arm.
//!
//! Per AGENTS.md §2 this file carries no in-file test module; the live launch
//! tests are in `crates/mlrs-backend/tests/voting_test.rs` (this crate is
//! backend-feature-free and cannot launch anything itself).

use cubecl::prelude::*;

/// Seed the accumulator with the FIRST member's weighted prediction,
/// `acc[i] = pred[i] · weight`.
///
/// Separate from [`vote_add_weighted`] so neither kernel carries a runtime
/// branch (D-07 prefers a statement over a conditional expression, and the
/// cheapest conditional is the one the host resolves). Bounds are checked
/// against `pred.len()` so the standard ceiling-division launch may
/// over-provision units safely (T-0203-01).
#[cube(launch)]
pub fn vote_init_weighted<F: Float + CubeElement>(pred: &Array<F>, acc: &mut Array<F>, weight: F) {
    let tid = ABSOLUTE_POS;
    if tid < pred.len() {
        acc[tid] = pred[tid] * weight;
    }
}

/// Accumulate one more member, `acc[i] += pred[i] · weight`.
///
/// Read-after-write on `acc` against the preceding launch; see the module docs
/// for why stream ordering is what makes that sound. Written as an explicit
/// `acc[tid] = acc[tid] + …` rather than `+=` to keep the load and the store in
/// the IR in the order the comment above describes.
#[cube(launch)]
pub fn vote_add_weighted<F: Float + CubeElement>(pred: &Array<F>, acc: &mut Array<F>, weight: F) {
    let tid = ABSOLUTE_POS;
    if tid < pred.len() {
        acc[tid] = acc[tid] + pred[tid] * weight;
    }
}

/// Normalise the accumulator in place, `acc[i] /= denom`.
///
/// A DIVISION, deliberately — see the module docs. `denom` is the host-computed
/// weight sum; it is validated non-zero before the launch, because a kernel
/// cannot return an error and numpy answers a zero weight sum with a
/// `ZeroDivisionError` rather than an infinity.
#[cube(launch)]
pub fn vote_divide<F: Float + CubeElement>(acc: &mut Array<F>, denom: F) {
    let tid = ABSOLUTE_POS;
    if tid < acc.len() {
        acc[tid] = acc[tid] / denom;
    }
}

/// Write one member's prediction column into the `n × k` transform matrix,
/// `mat[r, col] = pred[r]`.
///
/// This is the `transform` half, and it is a pure scatter with no arithmetic —
/// the write is strided by `n_cols` while the read is contiguous, which is the
/// transpose sklearn performs with `np.asarray([…]).T`. The caller guarantees
/// `col < n_cols` and that `mat` is `pred.len() * n_cols` long; both are
/// validated host-side in `mlrs_backend::prims::voting` before any launch.
#[cube(launch)]
pub fn vote_write_col<F: Float + CubeElement>(
    pred: &Array<F>,
    mat: &mut Array<F>,
    n_cols: u32,
    col: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < pred.len() {
        mat[tid * n_cols as usize + col as usize] = pred[tid];
    }
}
