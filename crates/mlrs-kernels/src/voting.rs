//! Prediction-voting kernels (VOTE-01 / VOTE-CLF-01) — the device arm of
//! `mlrs.VotingRegressor`'s `_predict` / `transform` and of
//! `mlrs.VotingClassifier`'s `predict` / `predict_proba` / `transform`.
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
//! ## The classifier half (VOTE-CLF-01)
//!
//! `VotingClassifier` runs one of two entirely different aggregations, chosen by
//! its `voting` parameter, and they have nothing in common but the members:
//!
//! ```text
//!   voting='hard'  predict(X)[r] = argmax_c Σⱼ wⱼ·[predⱼ[r] == c]
//!   voting='soft'  proba(X)[r,c] = (Σⱼ probaⱼ[r,c]·wⱼ) / (Σⱼ wⱼ)
//!                  predict(X)[r] = argmax_c proba(X)[r,c]
//! ```
//!
//! **Soft voting needs no new arithmetic.** `np.average(probas, axis=0,
//! weights=w)` over a `(k, n, C)` stack is *exactly* the regressor's row mean
//! with `n·C` in place of `n` — the reduced axis is still the member axis and
//! each member still contributes one contiguous block. So [`vote_init_weighted`]
//! / [`vote_add_weighted`] / [`vote_divide`] are reused unchanged, and the only
//! genuinely new kernel on that route is [`vote_argmax_rows`], which turns the
//! averaged `(n, C)` block into labels **without downloading it** — the whole
//! reason the soft route has a device arm worth having.
//!
//! **Hard voting is a scatter-accumulate, not a reduction over a fixed axis.**
//! sklearn writes it as `np.apply_along_axis(lambda x: argmax(bincount(x,
//! weights=w)), axis=1, arr=predictions)` — a Python-level loop over `n` rows.
//! On the device it becomes an `n × n_bins` tally: [`vote_bincount_add`] runs
//! once per member, each unit owning one ROW (so the read-modify-write of
//! `counts[r·n_bins + label]` needs no atomic — no two units address the same
//! row), and [`vote_argmax_bounded`] reduces each row.
//!
//! ### `np.bincount`'s length is per row, and it is observable
//!
//! `np.bincount(x, weights=w)` returns `x.max() + 1` entries, not `n_classes` —
//! so `argmax` never looks at a class above the row's own largest prediction.
//! With non-negative weights that is invisible (any class present has a count
//! ≥ the absent classes' implicit 0, and `argmax` takes the FIRST maximum). With
//! **negative** weights it is not: `w = [-1, -2]` on a row of `[0, 0]` gives
//! `[-3]` → class 0, where a full-width tally would give `[-3, 0, …]` → class 1.
//! sklearn's `weights` constraint is `array-like`, which admits negatives, so
//! [`vote_bincount_add`] tracks each row's ceiling in `hi` and
//! [`vote_argmax_bounded`] scans `0..=hi[r]` rather than `0..n_bins`.
//!
//! ### Counting happens in `F`, and that is exact
//!
//! `np.bincount` accumulates weights in `float64` regardless of the weights'
//! own dtype, and returns `int64` counts when `weights is None`. The tally here
//! is an `F` accumulation in member order, which reproduces the weighted case
//! bit for bit when `F = f64`; the uniform case is a sum of `1.0`s and is exact
//! in either width for any `k` a real ensemble has (`k < 2^24` in `f32`). The
//! host arm always uses `f64`, matching numpy; the device arm's width is the
//! caller's, and `mlrs_backend::prims::voting` refuses the `f64` device route on
//! a backend without `f64` kernels rather than silently narrowing it.
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

// ------------------------------------------------------------------------- //
// VotingClassifier (VOTE-CLF-01)
// ------------------------------------------------------------------------- //

/// Clear the `n × n_bins` tally before the first member accumulates into it.
///
/// A pool handle is NOT zero-initialised — it is whatever the last owner left —
/// and [`vote_bincount_add`] only ever touches the ONE bin each member voted
/// for, so every other bin has to be zeroed here or it would contribute a stale
/// count to the argmax.
#[cube(launch)]
pub fn vote_counts_zero<F: Float + CubeElement>(counts: &mut Array<F>) {
    let tid = ABSOLUTE_POS;
    if tid < counts.len() {
        counts[tid] = F::new(0.0_f32);
    }
}

/// Clear the per-row label ceiling `hi` that bounds [`vote_argmax_bounded`].
///
/// Separate from [`vote_counts_zero`] because the two buffers have different
/// element types AND different lengths (`n` versus `n · n_bins`), so they cannot
/// share a launch. Starting at `0` is what lets [`vote_bincount_add`] use an
/// unconditional `max` for every member including the first: a label is a
/// `u32`, so `max(0, label) == label`.
#[cube(launch)]
pub fn vote_hi_zero(hi: &mut Array<u32>) {
    let tid = ABSOLUTE_POS;
    if tid < hi.len() {
        hi[tid] = 0u32;
    }
}

/// Add one member's votes to the per-row tally — `counts[r, labels[r]] += weight`
/// — and raise that row's label ceiling.
///
/// **One unit per ROW, which is what makes the atomic unnecessary.** Unit `r`
/// is the only unit in the launch that addresses row `r`, so the
/// read-modify-write of `counts[r · n_bins + label]` cannot race another unit
/// even though two rows may vote for the same class. Between members the
/// launches are read-after-write on `counts` and are ordered by the single
/// client stream, exactly as [`vote_add_weighted`] is.
///
/// `hi[r]` accumulates `max` over the members, giving each row the
/// `x.max()` that `np.bincount` derives its length from — see the module docs
/// for why that bound is observable rather than cosmetic.
///
/// The caller guarantees `labels[r] < n_bins` (host-side, from the same scan
/// that chose `n_bins`); a kernel cannot report otherwise.
#[cube(launch)]
pub fn vote_bincount_add<F: Float + CubeElement>(
    labels: &Array<u32>,
    counts: &mut Array<F>,
    hi: &mut Array<u32>,
    n_bins: u32,
    weight: F,
) {
    let tid = ABSOLUTE_POS;
    if tid < labels.len() {
        let label = labels[tid];
        let slot = tid * n_bins as usize + label as usize;
        counts[slot] = counts[slot] + weight;
        if label > hi[tid] {
            hi[tid] = label;
        }
    }
}

/// `out[r] = argmax over counts[r, 0..=hi[r]]`, lowest index on a tie.
///
/// The scan is a plain serial sweep by ONE unit per row rather than a
/// shared-memory tree. `n_bins` is the class count — three, ten, rarely more —
/// so a tree would spend more on its barriers than on the comparisons, and the
/// per-row variant of a shared reduction is the exact shape that has been
/// measured pathological on this project's `PyO3` paths
/// (`mlrs-row-reduce-shared-landmine`).
///
/// `> best` and not `>= best`: numpy's `argmax` returns the FIRST maximum, and
/// so does `np.argmax(np.bincount(...))`, which is the operation this
/// reproduces.
#[cube(launch)]
pub fn vote_argmax_bounded<F: Float + CubeElement>(
    counts: &Array<F>,
    hi: &Array<u32>,
    out: &mut Array<u32>,
    n_bins: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < hi.len() {
        let base = tid * n_bins as usize;
        let last = hi[tid];
        let mut best = counts[base];
        let mut best_idx = 0u32;
        let mut c = 1u32;
        while c <= last {
            let v = counts[base + c as usize];
            if v > best {
                best = v;
                best_idx = c;
            }
            c += 1u32;
        }
        out[tid] = best_idx;
    }
}

/// `out[r] = argmax over mat[r, 0..n_cols]`, lowest index on a tie.
///
/// The soft-voting counterpart of [`vote_argmax_bounded`], and deliberately a
/// separate kernel rather than the same one with a saturated `hi`: soft voting
/// averages PROBABILITIES, over which every class is always in range, so
/// carrying an `n`-long ceiling buffer would be an allocation and a launch spent
/// to encode the constant `n_cols - 1`.
///
/// This kernel is the reason the soft route's device arm is worth having. It
/// consumes the `n × C` average IN PLACE on the device and emits `n` labels, so
/// `predict` never downloads the probability block at all.
#[cube(launch)]
pub fn vote_argmax_rows<F: Float + CubeElement>(mat: &Array<F>, out: &mut Array<u32>, n_cols: u32) {
    let tid = ABSOLUTE_POS;
    if tid < out.len() {
        let base = tid * n_cols as usize;
        let mut best = mat[base];
        let mut best_idx = 0u32;
        let mut c = 1u32;
        while c < n_cols {
            let v = mat[base + c as usize];
            if v > best {
                best = v;
                best_idx = c;
            }
            c += 1u32;
        }
        out[tid] = best_idx;
    }
}

/// Write one member's `n × width` probability block into the horizontally
/// stacked `n × (k · width)` transform matrix.
///
/// This is `np.hstack(probas)`, the `voting='soft', flatten_transform=True`
/// transform. Unlike [`vote_write_col`] the source is 2-D, so the unit index
/// walks the BLOCK (contiguous read) and the destination stride is the full
/// output width — one unit per element rather than per row, because a block is
/// `n · width` elements and a per-row unit would serialise `width` of them.
///
/// The caller guarantees `mat` is `block.len() / width · out_stride` long and
/// that `col_offset + width <= out_stride`; both are checked host-side in
/// `mlrs_backend::prims::voting` before any launch.
#[cube(launch)]
pub fn vote_write_block<F: Float + CubeElement>(
    block: &Array<F>,
    mat: &mut Array<F>,
    width: u32,
    out_stride: u32,
    col_offset: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < block.len() {
        let row = tid / width as usize;
        let col = tid % width as usize;
        mat[row * out_stride as usize + col_offset as usize + col] = block[tid];
    }
}
