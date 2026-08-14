//! `RidgeCV`'s generalized-CV streaming sweep, on the device (RIDGECV-02).
//!
//! One kernel, [`gcv_cov_sweep`], carries the whole `O(n·d² + n·n_alphas·d·
//! (n_y+2))` part of the `cv=None`, `n > d` fit. Everything else in that fit is
//! `O(d³)` or smaller and stays on the host: the `d × d` eigendecomposition, the
//! per-alpha spectral weights, and the `n_alphas × d × n_y` coefficient block.
//!
//! ## What the sweep computes
//! With `X̃` the preprocessed design (weighted-centered, then `√w`-rescaled),
//! `λ, V` the eigendecomposition of `X̃ᵀX̃`, and `W = X̃·V`:
//!
//! ```text
//! qᵢₐ    = Σₖ Wᵢₖ² / (λₖ + αₐ)                  = diag(X̃·Hinv·X̃ᵀ)ᵢ
//! rswᵢₐ  = Σₖ Wᵢₖ · (Vᵀ X̃ᵀ√w)ₖ / (λₖ + αₐ)
//! ttᵢₐₜ  = Σₖ Wᵢₖ · (Vᵀ X̃ᵀỹ)ₖₜ / (λₖ + αₐ)
//! looe   = (ỹᵢₜ − ttᵢₐₜ) / (1 − qᵢₐ − (√wᵢ − rswᵢₐ)·√wᵢ / Σw)
//! ```
//!
//! The three divisions by `λₖ + αₐ` are folded into the host-side operands
//! `g`/`gzsw`/`gz` before launch, so the kernel never divides inside a `k` loop
//! — exactly as `ridge_cv.rs::gcv_cov`'s host sweep does not.
//!
//! ## Why the row TILE, and why `W` is not materialized
//! The naive shape — one unit per `(row, k)` — issues one global `V` load per
//! multiply-add, and the projection `W = X̃·V` is `n·d²` of them. That is the
//! same 0.33-FMA-per-load pathology `bayes_predict_std` measured at ~1% of peak
//! (`prims::normal_eq` module docs). Here a cube stages [`GCV_ROW_TILE`] rows of
//! `X̃` into shared memory and holds [`GCV_ROW_TILE`] accumulators per unit, so
//! each `v[j·d + k]` load feeds four multiply-adds instead of one.
//!
//! `W` itself never reaches global memory: the tile's `GCV_ROW_TILE × d` block
//! is consumed for EVERY alpha before the next tile overwrites it, which is the
//! device transcription of the host sweep's `ROW_BLOCK` streaming (and the
//! reason neither arm allocates the `n × d` `W` the textbook form implies).
//!
//! ## Shared-memory budget, and the `d` cap it implies
//! Two `GCV_ROW_TILE × GCV_MAX_D` tiles at `f64` is 16 KiB — inside the 32 KiB
//! floor every backend this repo targets provides, and checked against
//! `capability::active_max_shared_memory` at the launch site anyway. `d` above
//! [`GCV_MAX_D`] has no device arm and falls back to the host sweep; that is a
//! shape cap, not a correctness one.
//!
//! ## cpu-MLIR contract
//! `SharedMemory` + `sync_cube` (the `reduce.rs::argmin_shared` idiom), `F`/`u32`
//! accumulators only, STATEMENT-form `if` guards, no mutable `bool`. Every
//! `sync_cube` sits under the cube-uniform `b < nblocks` guard and inside loops
//! whose trip counts are cube-uniform (`rows_per_block`, `d`), so all units of a
//! cube execute the same barrier count.
//!
//! Tests live in `crates/mlrs-backend/tests/ridge_gcv_test.rs` and
//! `crates/mlrs-algos/tests/ridge_cv_device_test.rs` (AGENTS.md §2).

use cubecl::prelude::*;

/// Rows one cube stages and projects together.
///
/// Four is the register-tile width that turns one `V` load into four
/// multiply-adds; the same 4-wide choice `gram_xty_blocked`'s 1×8 tile and
/// `bayes_predict_std`'s 4×4 tile were measured into. Raising it raises the
/// shared-memory footprint linearly and the `d` cap falls with it.
pub const GCV_ROW_TILE: u32 = 4;

/// Largest `n_features` the device sweep accepts.
///
/// `GCV_ROW_TILE × GCV_MAX_D` `f64` per shared tile, two tiles = 16 KiB. The
/// caller gates on this and on the adapter's reported shared-memory size, and
/// routes a wider design to the host sweep.
pub const GCV_MAX_D: u32 = 256;

/// Units per cube. Matches `prims::gram`'s `BLOCKED_CUBE_DIM`, so the two
/// row-blocked passes of a `RidgeCV` device fit have the same occupancy shape.
pub const GCV_CUBE_DIM: u32 = 64;

/// The tail of the LOO identity for ONE `(row, alpha, target)`: the residual,
/// then either its square (folded into the block's score accumulator) or the
/// rescaled prediction.
///
/// A helper rather than an inlined block because the contraction unrolls
/// [`GCV_ROW_TILE`] rows into registers, so this tail appears four times per
/// target and four copies of it would be four places for the `weighted` rescale
/// or the `(i·n_alphas + a)·n_y + t` index to drift apart.
///
/// The `partials` accumulation is a plain read-modify-write and needs no atomic:
/// the caller's unit owns `(block, a)` for the whole cube, so no other unit ever
/// touches this slot.
#[cube]
#[allow(clippy::too_many_arguments)]
fn gcv_emit_row<F: Float + CubeElement>(
    y: &Array<F>,
    ymean: &Array<F>,
    partials: &mut Array<F>,
    cv_out: &mut Array<F>,
    i: u32,
    a: u32,
    tgt: u32,
    n_y: u32,
    n_alphas: u32,
    pbase: u32,
    s_i: F,
    denom: F,
    tt: F,
    weighted: u32,
    want_predictions: u32,
    emit_values: u32,
) {
    let yt = s_i * (y[(i * n_y + tgt) as usize] - ymean[tgt as usize]);
    let looe = (yt - tt) / denom;

    if want_predictions == 1u32 {
        // sklearn: `predictions = y − looe`, un-rescaled by `√w` ONLY when
        // weights were given, then re-offset by the target mean.
        let mut p = yt - looe;
        if weighted == 1u32 {
            p /= s_i;
        }
        p += ymean[tgt as usize];
        if emit_values == 1u32 {
            cv_out[((i * n_alphas + a) * n_y + tgt) as usize] = p;
        }
    }
    if want_predictions == 0u32 {
        let sq = looe * looe;
        partials[(pbase + a * n_y + tgt) as usize] += sq;
        if emit_values == 1u32 {
            cv_out[((i * n_alphas + a) * n_y + tgt) as usize] = sq;
        }
    }
}

/// One cube per row block: project each `GCV_ROW_TILE`-row tile into the
/// eigenbasis and contract it against every alpha.
///
/// Operands (all `f64` in practice — the caller widens an `f32` design first,
/// because the LOO denominator `1 − q` is a cancellation and `q → 1` is the
/// interesting end of the alpha grid):
///
/// - `x` — `n × d` row-major RAW design, `y` — `n × n_y` row-major RAW target.
///   The preprocessing (`√wᵢ · (xᵢⱼ − x̄ⱼ)`) is applied AS THE ROW IS STAGED, so
///   no preprocessed copy exists on either arm.
/// - `xmean`/`ymean` — the weighted column means (`d` / `n_y`), zeros when
///   `fit_intercept` is off.
/// - `sqrt_sw` — `√wᵢ` (`n`), all ones when unweighted. It is a real buffer in
///   both cases: the intercept correction reads `√wᵢ` on its own, so a
///   switched-off branch would still have to bind something (`row_scale_center`
///   precedent), and ones cost `n` elements against the `n · d` design.
/// - `v` — `d × d` eigenvectors, `v[j·d + k]` = component `j` of eigenvector
///   `k`.
/// - `g` — `n_alphas × d`, `1/(λₖ + αₐ)`.
/// - `gz` — `n_alphas × d × n_y`, `g ⊙ (Vᵀ X̃ᵀỹ)`.
/// - `gzsw` — `n_alphas × d`, `g ⊙ (Vᵀ X̃ᵀ√w)`.
/// - `partials` — `nblocks × n_alphas × n_y`, the per-block sums of `looe²`.
///   Each unit OWNS the `(block, a)` slots it writes (it zeroes them itself, so
///   the caller may hand over an unzeroed pool buffer), so the accumulation
///   needs no atomics.
/// - `cv_out` — `n × n_alphas × n_y` row-major (`n`-major, matching the host
///   arm's layout), squared errors or rescaled LOO predictions. Bind a
///   one-element dummy when `emit_values` is 0.
///
/// `want_predictions` switches the per-row output from `looe²` to the rescaled
/// prediction; `emit_values` is whether `cv_out` is written at all. `weighted`
/// selects sklearn's un-rescale of that prediction by `√wᵢ`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gcv_cov_sweep<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    xmean: &Array<F>,
    ymean: &Array<F>,
    sqrt_sw: &Array<F>,
    v: &Array<F>,
    g: &Array<F>,
    gz: &Array<F>,
    gzsw: &Array<F>,
    partials: &mut Array<F>,
    cv_out: &mut Array<F>,
    n: u32,
    d: u32,
    n_y: u32,
    n_alphas: u32,
    nblocks: u32,
    rows_per_block: u32,
    sw_sum: F,
    fit_intercept: u32,
    weighted: u32,
    want_predictions: u32,
    emit_values: u32,
) {
    let mut xs = SharedMemory::<F>::new((GCV_ROW_TILE * GCV_MAX_D) as usize);
    let mut ws = SharedMemory::<F>::new((GCV_ROW_TILE * GCV_MAX_D) as usize);

    // Linearized cube id over the (possibly Y-folded) grid — UNIFORM per cube,
    // so every barrier below is reached by all units or by none.
    let b = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if b < nblocks {
        let t = UNIT_POS;
        let stride = CUBE_DIM_X;
        let start = b * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }
        let zero = F::from_int(0i64);
        let one = F::from_int(1i64);

        // Each unit zeroes exactly the `(block, alpha)` slots it will later
        // accumulate into, which is what makes the accumulation atomic-free AND
        // lets the caller pass a recycled pool buffer.
        let pbase = b * n_alphas * n_y;
        let mut az = t;
        while az < n_alphas {
            let mut tz = 0u32;
            while tz < n_y {
                partials[(pbase + az * n_y + tz) as usize] = zero;
                tz += 1u32;
            }
            az += stride;
        }

        let mut r0 = start;
        while r0 < end {
            let mut live = end - r0;
            if live > GCV_ROW_TILE {
                live = GCV_ROW_TILE;
            }

            // --- Stage: X̃ = √w ⊙ (X − x̄) for the tile's rows. Out-of-range
            //     lanes of the last tile stage 0, so the projection loop below
            //     is uniform; their `W` is discarded by the `rr < live` guard
            //     in the contraction. ---
            let mut rs = 0u32;
            while rs < GCV_ROW_TILE {
                let mut j = t;
                while j < d {
                    let mut val = zero;
                    if rs < live {
                        let i = r0 + rs;
                        val = sqrt_sw[i as usize] * (x[(i * d + j) as usize] - xmean[j as usize]);
                    }
                    xs[(rs * GCV_MAX_D + j) as usize] = val;
                    j += stride;
                }
                rs += 1u32;
            }
            sync_cube();

            // --- Project: W = X̃·V, four rows per `V` load. Adjacent units own
            //     adjacent `k`, so `v[j·d + k]` is a coalesced read. ---
            let mut k = t;
            while k < d {
                let mut a0 = zero;
                let mut a1 = zero;
                let mut a2 = zero;
                let mut a3 = zero;
                let mut j = 0u32;
                while j < d {
                    let vjk = v[(j * d + k) as usize];
                    a0 += xs[j as usize] * vjk;
                    a1 += xs[(GCV_MAX_D + j) as usize] * vjk;
                    a2 += xs[(2u32 * GCV_MAX_D + j) as usize] * vjk;
                    a3 += xs[(3u32 * GCV_MAX_D + j) as usize] * vjk;
                    j += 1u32;
                }
                ws[k as usize] = a0;
                ws[(GCV_MAX_D + k) as usize] = a1;
                ws[(2u32 * GCV_MAX_D + k) as usize] = a2;
                ws[(3u32 * GCV_MAX_D + k) as usize] = a3;
                k += stride;
            }
            sync_cube();

            // `√wᵢ` for the tile's rows — alpha-INDEPENDENT, so it is hoisted
            // out of the alpha loop. A dead lane keeps `1`, which only ever
            // feeds arithmetic the `rc < live` guards below discard.
            let mut si0 = one;
            let mut si1 = one;
            let mut si2 = one;
            let mut si3 = one;
            if live > 0u32 {
                si0 = sqrt_sw[r0 as usize];
            }
            if live > 1u32 {
                si1 = sqrt_sw[(r0 + 1u32) as usize];
            }
            if live > 2u32 {
                si2 = sqrt_sw[(r0 + 2u32) as usize];
            }
            if live > 3u32 {
                si3 = sqrt_sw[(r0 + 3u32) as usize];
            }

            // --- Contract: one unit per ALPHA (strided), so a unit owns the
            //     whole `(block, alpha)` accumulator and the scores need no
            //     cross-unit reduction and no atomics.
            //
            //     The TILE'S FOUR ROWS ARE THE INNER AXIS HERE TOO, for the same
            //     reason they are in the projection: each `g`/`gzsw`/`gz` load
            //     is loop-invariant across them, so holding four accumulators
            //     turns one global load into four multiply-adds instead of one.
            //     Measured on gfx1151 at `n = 100 000, d = 64`, 30 alphas: the
            //     sweep went from 110 ms to the figure `ridge_cv_device_perf_
            //     test::device_phase_attribution` now prints. ---
            let mut a = t;
            while a < n_alphas {
                let gbase = a * d;
                let mut q0 = zero;
                let mut q1 = zero;
                let mut q2 = zero;
                let mut q3 = zero;
                let mut r_0 = zero;
                let mut r_1 = zero;
                let mut r_2 = zero;
                let mut r_3 = zero;
                let mut kk = 0u32;
                while kk < d {
                    let gk = g[(gbase + kk) as usize];
                    let sk = gzsw[(gbase + kk) as usize];
                    let w0 = ws[kk as usize];
                    let w1 = ws[(GCV_MAX_D + kk) as usize];
                    let w2 = ws[(2u32 * GCV_MAX_D + kk) as usize];
                    let w3 = ws[(3u32 * GCV_MAX_D + kk) as usize];
                    q0 += gk * w0 * w0;
                    q1 += gk * w1 * w1;
                    q2 += gk * w2 * w2;
                    q3 += gk * w3 * w3;
                    r_0 += w0 * sk;
                    r_1 += w1 * sk;
                    r_2 += w2 * sk;
                    r_3 += w3 * sk;
                    kk += 1u32;
                }

                let mut den0 = one - q0;
                let mut den1 = one - q1;
                let mut den2 = one - q2;
                let mut den3 = one - q3;
                if fit_intercept == 1u32 {
                    den0 -= (si0 - r_0) * si0 / sw_sum;
                    den1 -= (si1 - r_1) * si1 / sw_sum;
                    den2 -= (si2 - r_2) * si2 / sw_sum;
                    den3 -= (si3 - r_3) * si3 / sw_sum;
                }

                let mut tgt = 0u32;
                while tgt < n_y {
                    let mut t0 = zero;
                    let mut t1 = zero;
                    let mut t2 = zero;
                    let mut t3 = zero;
                    let mut k2 = 0u32;
                    while k2 < d {
                        let gzk = gz[((gbase + k2) * n_y + tgt) as usize];
                        t0 += ws[k2 as usize] * gzk;
                        t1 += ws[(GCV_MAX_D + k2) as usize] * gzk;
                        t2 += ws[(2u32 * GCV_MAX_D + k2) as usize] * gzk;
                        t3 += ws[(3u32 * GCV_MAX_D + k2) as usize] * gzk;
                        k2 += 1u32;
                    }
                    if live > 0u32 {
                        gcv_emit_row::<F>(
                            y,
                            ymean,
                            partials,
                            cv_out,
                            r0,
                            a,
                            tgt,
                            n_y,
                            n_alphas,
                            pbase,
                            si0,
                            den0,
                            t0,
                            weighted,
                            want_predictions,
                            emit_values,
                        );
                    }
                    if live > 1u32 {
                        gcv_emit_row::<F>(
                            y,
                            ymean,
                            partials,
                            cv_out,
                            r0 + 1u32,
                            a,
                            tgt,
                            n_y,
                            n_alphas,
                            pbase,
                            si1,
                            den1,
                            t1,
                            weighted,
                            want_predictions,
                            emit_values,
                        );
                    }
                    if live > 2u32 {
                        gcv_emit_row::<F>(
                            y,
                            ymean,
                            partials,
                            cv_out,
                            r0 + 2u32,
                            a,
                            tgt,
                            n_y,
                            n_alphas,
                            pbase,
                            si2,
                            den2,
                            t2,
                            weighted,
                            want_predictions,
                            emit_values,
                        );
                    }
                    if live > 3u32 {
                        gcv_emit_row::<F>(
                            y,
                            ymean,
                            partials,
                            cv_out,
                            r0 + 3u32,
                            a,
                            tgt,
                            n_y,
                            n_alphas,
                            pbase,
                            si3,
                            den3,
                            t3,
                            weighted,
                            want_predictions,
                            emit_values,
                        );
                    }
                    tgt += 1u32;
                }
                a += stride;
            }

            // Barrier before the next tile overwrites the staged rows.
            sync_cube();
            r0 += GCV_ROW_TILE;
        }
    }
}
