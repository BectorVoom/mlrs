//! `huber` — the device kernels behind `HuberRegressor`'s GPU objective engine
//! (HUBER-02).
//!
//! ## What these exist to remove
//! The first device arm of `prims::huber_objective` evaluated the Huber
//! objective as two `prims::gemm` launches with the CLASSIFICATION step on the
//! host between them. Per objective evaluation — and an L-BFGS solve spends
//! dozens of them — that shape paid:
//!
//! | # | step | cost |
//! |---|---|---|
//! | 1 | upload `w` | `d_aug` floats up |
//! | 2 | `gemm` → margins | the design pass |
//! | 3 | `margins.to_host` | **a device SYNC + `n` floats down** |
//! | 4 | host classify loop | `O(n)` serial host work |
//! | 5 | upload `g` | **`n` floats up** |
//! | 6 | `gemm` transposed → `x̃ᵀg` | the design pass |
//! | 7 | `xtg.to_host` | **a second device SYNC** |
//!
//! Steps 3–5 are the whole problem: two `n`-length transfers and two full
//! pipeline stalls per evaluation, on a quantity (`g`) that is produced and
//! consumed entirely on the device. That is the launch-latency-bound shape this
//! repo has hit repeatedly in iterative prims ([[mlrs-gpu-perf-root-cause]]) —
//! and the fix is always the same: keep the intermediate resident and read back
//! only the `O(d)` result.
//!
//! With the kernels here one evaluation is `row pass → reduce → fold →
//! transposed pass → fold`: five launches and **one** readback of `d_aug + 5`
//! floats. Nothing of size `n` crosses the bus at all, ever.
//!
//! ## The synthetic intercept column is never materialized
//! sklearn's Huber intercept is a constant-`1.0` column appended to the design.
//! Materializing it costs an `n × (d+1)` device buffer AND — when the caller's
//! design was already device-resident — a `to_host` + host augment +
//! `from_host` round-trip of the WHOLE design at construction, which is three
//! passes over `n·d` and a sync to write a column of ones.
//!
//! Both design passes read the caller's UNAUGMENTED `n × d` slab instead, and
//! the two places the extra column would have contributed are folded in for
//! free:
//!
//! - its contribution to every margin is the CONSTANT `w[d]`, so it is passed
//!   to [`huber_row_pass`] as the `bias` scalar and added there;
//! - its entry of `x̃ᵀ·g` is `Σᵢ gᵢ`, which is just one more scalar reduction,
//!   so it rides along as the 5th quantity of the same blocked fold the loss
//!   terms already use.
//!
//! The design is therefore consumed exactly as the caller supplied it, which is
//! what lets a device-resident `fit` upload NOTHING.
//!
//! ## The two design passes are shaped DIFFERENTLY, on purpose
//! The margin (`X·w`) and the gradient (`Xᵀ·g`) are the only `O(n·d)` steps.
//! Neither goes through `prims::gemm`: both are GEMV, and a tiled matmul
//! degenerated to a single output column measured **three orders of magnitude**
//! off the hardware here (see [`huber_margin_rows`] for the number).
//!
//! They are not the same kernel either, because the transpose flips which axis
//! is contiguous:
//!
//! - the FORWARD pass ([`huber_row_pass`]) gives each unit a whole row. A single
//!   load is not coalesced, but each unit reads its row once and in order, so a
//!   warp's working set is contiguous and stays in L1.
//! - the TRANSPOSED pass ([`huber_xtg_blocked`]) gives each unit a column within
//!   a row-block, so neighbouring units read neighbouring addresses and every
//!   load IS coalesced.
//!
//! Everything else here touches only the `O(n)` and `O(d)` arrays, where a bare
//! `ABSOLUTE_POS` map is coalesced by construction.
//!
//! ## The accumulation shape (and why it is blocked, not flat)
//! [`huber_quad_reduce_blocked`] + [`huber_fold_partials`] is a two-level
//! reduction: `nblocks` blocks each summing `rows_per_block` rows, then one
//! fold of `nblocks` partials. Balanced (`nblocks ≈ rows_per_block ≈ √n`) the
//! round-off of a random-walk error model is `O(n^¼·ε)` instead of the flat
//! sum's `O(√n·ε)` — at `n = 100 000`, ~35·ε against ~316·ε. It matters here
//! and not in a plain mean because `∂L/∂σ` is the DIFFERENCE of two `O(n)` sums
//! that nearly cancel at the optimum, so it is the first gradient entry a
//! narrow accumulation destroys (`prims::huber_objective`'s precision note),
//! and the device arm accumulates in `F` where the host arm has `f64` free.
//!
//! ## cubecl-cpu MLIR safety
//! Kept to the `sgd`/`gmm` house rules even though the cpu backend routes
//! `HuberRegressor` to the host fused pass and never launches these: only
//! `F`/`u32` accumulators, `if`-guarded forward `while` loops, statement-form
//! `if` (never an `if`-expression in value position), no `SharedMemory`, no
//! `bool`, no infinity sentinel, no scatter. That is what lets a direct
//! cubecl-cpu execution test verify kernels whose production backend is a GPU
//! ([[mlrs-gaussian-mixture-cuda-device]]'s technique).
//!
//! Negations are passed IN as scalars (`neg_outlier_scale`) rather than formed
//! with a unary `-` in the kernel — the same defensive habit the rest of this
//! crate uses for `F::new(0.0) - x`.
//!
//! All kernels are generic over `<F: Float + CubeElement>` and carry NO backend
//! feature (D-13).
//!
//! Tests live in `crates/mlrs-backend/tests/huber_device_test.rs` (AGENTS.md §2
//! — never an in-source `#[cfg(test)] mod tests`).

use cubecl::prelude::*;

pub use self::huber_classify_rows as huber_classify_rows_kernel;
pub use self::huber_copy_into as huber_copy_into_kernel;
pub use self::huber_fold_partials as huber_fold_partials_kernel;
pub use self::huber_margin_rows as huber_margin_rows_kernel;
pub use self::huber_outlier_mask_rows as huber_outlier_mask_rows_kernel;
pub use self::huber_quad_reduce_blocked as huber_quad_reduce_blocked_kernel;
pub use self::huber_row_pass as huber_row_pass_kernel;
pub use self::huber_xtg_blocked as huber_xtg_blocked_kernel;

/// `margins[i] = Σⱼ x[i·d + j]·w[j]` — the forward design pass, as a plain
/// one-unit-per-row dot.
///
/// ## Why this is NOT `prims::gemm`
/// Both design passes are GEMV (`N = 1`), and the tuned `cubek-matmul`
/// substrate is built for `M × K × N` tiles. MEASURED on a gfx1151 iGPU
/// (rocm, `f32`, `n = 100 000`, `d = 16`): routing these two products through
/// `prims::gemm` cost **142 ms per L-BFGS iteration** — for 1.6 M multiply-adds,
/// i.e. three orders of magnitude off the hardware. Degenerating a tiled matmul
/// to a single output column is the pathology, not the arithmetic;
/// `huber_device_perf_test.rs` keeps the `gemm` route reachable
/// (`MLRS_HUBER_DEVICE=gemm`) so that ratio stays measurable rather than
/// remembered.
///
/// ## The access pattern, and where it stops working
/// Unit `i` streams row `i` and nothing else. Neighbouring units are `d`
/// elements apart, so a single load is NOT coalesced — the bet is that across
/// the whole `j` loop each unit reads its row once and in order, so a
/// wavefront's working set is `wave_width · d` contiguous elements and stays in
/// L1, making the total DRAM traffic `n·d`, which is the floor.
///
/// That bet is `d`-dependent and it FAILS at large `d`, which the crossover
/// ladder shows plainly: at 64 lanes and `f32` the working set is 4 KB at
/// `d = 16` but 32 KB at `d = 128`, past this iGPU's L1 — and the `50 000 × 128`
/// rung is the one where the device arm is furthest behind the host pass (118×,
/// against 9× at `10 000 × 64`), despite having the SAME `n·d` as a rung that
/// does far better. See `HUBER_DEVICE_MIN_WORK`'s table.
///
/// The fix, if this kernel is ever the thing worth fixing, is the alternative
/// this shape rejected for small `d`: one unit per `(row, j)` tile with a plane
/// reduction, which buys coalescing per load and pays a reduction per row.
/// Right now it is not worth fixing, because on this hardware the host pass
/// wins at every `d` anyway; on a discrete card it would be the first thing to
/// revisit.
///
/// The intercept is deliberately absent: it is a CONSTANT contribution
/// `w[d]` that [`huber_classify_rows`] adds as its `bias` scalar, so the
/// synthetic column is never materialized (module docs) and this kernel reads
/// the caller's own `n × d` slab.
#[cube(launch)]
pub fn huber_margin_rows<F: Float + CubeElement>(
    x: &Array<F>,
    w: &Array<F>,
    margins: &mut Array<F>,
    n: u32,
    d: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let base = (i as u32) * d;
        let mut acc = F::new(0.0_f32);
        let mut j = 0u32;
        while j < d {
            acc += x[(base + j) as usize] * w[j as usize];
            j += 1u32;
        }
        margins[i] = acc;
    }
}

/// ROW-BLOCKED transposed design pass: `psums[b·d + j] = Σ_{i ∈ block b}
/// g[i]·x[i·d + j]`, folded to `Xᵀ·g` by [`huber_fold_partials`] with
/// `len = d`.
///
/// One unit per `(block b, column j)` at `ABSOLUTE_POS = b·d + j`. Consecutive
/// units hold consecutive `j` at the same row, so every load `x[i·d + j]` is
/// perfectly COALESCED — the mirror image of [`huber_margin_rows`], and the
/// reason the two passes are shaped differently despite contracting the same
/// matrix.
///
/// Blocking over rows is what provides the parallelism (`d` alone is far too
/// few units) and, as a side effect, the two-level summation the module docs
/// want for round-off. The `b·d + j` layout is exactly the `nblocks × len`
/// shape [`huber_fold_partials`] consumes, so no bespoke reducer is needed.
///
/// The intercept column is absent here too: its entry of `X̃ᵀ·g` is `Σᵢ gᵢ`,
/// which rides the scalar fold as quantity 4 of [`huber_classify_rows`].
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn huber_xtg_blocked<F: Float + CubeElement>(
    x: &Array<F>,
    g: &Array<F>,
    psums: &mut Array<F>,
    n: u32,
    d: u32,
    nblocks: u32,
    rows_per_block: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = nblocks * d;
    if tid < total as usize {
        let b = (tid as u32) / d;
        let j = (tid as u32) % d;
        let start = b * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }
        let mut acc = F::new(0.0_f32);
        let mut i = start;
        while i < end {
            acc += g[i as usize] * x[(i * d + j) as usize];
            i += 1u32;
        }
        psums[tid] = acc;
    }
}

/// Quantities [`huber_classify_rows`] emits per sample and
/// [`huber_quad_reduce_blocked`] folds — `sq`, `out_abs`, `out_sw`, `count`,
/// `g` (module docs). Exported so the launching prim and the kernel cannot
/// disagree about the stride.
pub const HUBER_QUANTITIES: u32 = 5;

/// Classify every sample against the outlier threshold and emit BOTH the
/// per-sample gradient factor `g` and the five per-sample reduction terms.
///
/// `g[i] = ∂L/∂mᵢ` — the quantity the transposed GEMM contracts against the
/// design to form `Xᵀ·g`. It is produced and consumed entirely on the device;
/// this kernel exists so that it never round-trips.
///
/// `margins` holds `X·w[..d]` WITHOUT the synthetic intercept column, which the
/// `bias` scalar (`w[d]`, or `0` when no intercept is fitted) supplies here —
/// see the module docs for why that column is never materialized.
///
/// `quad` is `5·n` long, laid out as five CONTIGUOUS length-`n` segments
/// (`quad[q·n + i]`) rather than interleaved quintuples, so each write here is
/// coalesced across neighbouring units and
/// [`huber_quad_reduce_blocked`]'s scan of one segment is a contiguous read:
///
/// | q | term | folds into |
/// |---|---|---|
/// | 0 | `swᵢ·rᵢ²` on inliers, else 0 | `sq_sum` |
/// | 1 | `swᵢ·\|rᵢ\|` on outliers, else 0 | `out_abs_sum` |
/// | 2 | `swᵢ` on outliers, else 0 | `out_sw_sum` |
/// | 3 | `1` on outliers, else 0 | `n_outliers` |
/// | 4 | `gᵢ` (every sample) | the intercept's `x̃ᵀg` entry |
///
/// Scalars, all formed once per evaluation on the host: `thr = ε·σ`,
/// `inlier_scale = −2/σ`, `outlier_scale = 2·ε` and its negation
/// `neg_outlier_scale`. `weighted` is `1` when `sw` is the caller's real
/// length-`n` weight vector and `0` when it is a length-1 placeholder that is
/// never indexed — the device twin of the host pass's `WEIGHTED` const generic,
/// as a runtime flag because a kernel cannot be monomorphized on one.
///
/// The outlier test is STRICTLY greater (`|rᵢ| > ε·σ`), matching sklearn's
/// `abs_linear_loss > epsilon * sigma`: a residual exactly ON the threshold is
/// an inlier. The sign convention at a zero residual is `+1` (sklearn's
/// `np.ones_like` + `< 0` mask); a zero residual can never be an outlier, so
/// the tie only has to be well defined.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn huber_classify_rows<F: Float + CubeElement>(
    margins: &Array<F>,
    y: &Array<F>,
    sw: &Array<F>,
    g: &mut Array<F>,
    quad: &mut Array<F>,
    n: u32,
    weighted: u32,
    bias: F,
    thr: F,
    inlier_scale: F,
    outlier_scale: F,
    neg_outlier_scale: F,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let zero = F::new(0.0_f32);
        let res = y[i] - (margins[i] + bias);
        let a = res.abs();
        let mut s = F::new(1.0_f32);
        if weighted == 1u32 {
            s = sw[i];
        }
        // Seeded with the INLIER form, which is two multiplies and is what the
        // overwhelming majority of samples take; the outlier branch overrides
        // it. Written this way rather than as a zero-seeded two-armed `if`
        // because the latter leaves the seed provably dead, which the expansion
        // reports as an unused assignment.
        let mut gi = inlier_scale * s * res;
        let mut q0 = zero;
        let mut q1 = zero;
        let mut q2 = zero;
        let mut q3 = zero;
        if a > thr {
            q1 = s * a;
            q2 = s;
            q3 = F::new(1.0_f32);
            gi = neg_outlier_scale * s;
            if res < zero {
                gi = outlier_scale * s;
            }
        } else {
            q0 = s * res * res;
        }
        g[i] = gi;
        let nn = n as usize;
        quad[i] = q0;
        quad[nn + i] = q1;
        quad[2 * nn + i] = q2;
        quad[3 * nn + i] = q3;
        quad[4 * nn + i] = gi;
    }
}

/// [`huber_margin_rows`] and [`huber_classify_rows`] FUSED — the product
/// path's first launch.
///
/// Both are one-unit-per-row maps over the same `i`, so running them separately
/// costs an extra launch plus a full `n`-element store and reload of a value
/// that is consumed immediately. Fusing them keeps the margin in a register.
/// That matters more than the traffic: an L-BFGS solve spends dozens of
/// evaluations, each of which must synchronize once for the driver to choose
/// its next step, so the FIXED per-evaluation cost — launches and that one
/// stall — is the floor the whole device arm sits on, and every launch removed
/// lowers it.
///
/// The split pair is kept alive for the `MLRS_HUBER_DEVICE=gemm` A/B route,
/// which cannot use this kernel because its margins come out of the matmul
/// substrate rather than a row scan. The classification body below is therefore
/// deliberately a duplicate of [`huber_classify_rows`]'s and the two must be
/// changed together; `huber_device_test.rs::device_engine_matches_roundtrip_arm`
/// is what fails if they drift, since the round-trip arm exercises the split
/// pair and the resident arm exercises this one.
///
/// Arguments and semantics are otherwise exactly [`huber_classify_rows`]'s,
/// with `x`/`w`/`d` replacing its `margins` input.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn huber_row_pass<F: Float + CubeElement>(
    x: &Array<F>,
    w: &Array<F>,
    y: &Array<F>,
    sw: &Array<F>,
    g: &mut Array<F>,
    quad: &mut Array<F>,
    n: u32,
    d: u32,
    weighted: u32,
    bias: F,
    thr: F,
    inlier_scale: F,
    outlier_scale: F,
    neg_outlier_scale: F,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let zero = F::new(0.0_f32);
        let base = (i as u32) * d;
        let mut m = bias;
        let mut j = 0u32;
        while j < d {
            m += x[(base + j) as usize] * w[j as usize];
            j += 1u32;
        }
        let res = y[i] - m;
        let a = res.abs();
        let mut s = F::new(1.0_f32);
        if weighted == 1u32 {
            s = sw[i];
        }
        let mut gi = inlier_scale * s * res;
        let mut q0 = zero;
        let mut q1 = zero;
        let mut q2 = zero;
        let mut q3 = zero;
        if a > thr {
            q1 = s * a;
            q2 = s;
            q3 = F::new(1.0_f32);
            gi = neg_outlier_scale * s;
            if res < zero {
                gi = outlier_scale * s;
            }
        } else {
            q0 = s * res * res;
        }
        g[i] = gi;
        let nn = n as usize;
        quad[i] = q0;
        quad[nn + i] = q1;
        quad[2 * nn + i] = q2;
        quad[3 * nn + i] = q3;
        quad[4 * nn + i] = gi;
    }
}

/// Stage-1 blocked reduction of the [`HUBER_QUANTITIES`] segments
/// [`huber_row_pass`] / [`huber_classify_rows`] wrote.
///
/// One unit per `(block b, quantity q)` at `ABSOLUTE_POS = b·nq + q`: scan only
/// block `b`'s row range of segment `q` and write `psums[b·nq + q]`. The
/// `b·nq + q` layout (rather than `q·nblocks + b`) is deliberate — it is
/// exactly the `nblocks × len` shape [`huber_fold_partials`] folds with
/// `len = nq`, so no second bespoke reducer is needed.
///
/// Two levels rather than one flat sum for the round-off reason in the module
/// docs; the caller picks `nblocks ≈ rows_per_block ≈ √n`.
#[cube(launch)]
pub fn huber_quad_reduce_blocked<F: Float + CubeElement>(
    quad: &Array<F>,
    psums: &mut Array<F>,
    n: u32,
    nq: u32,
    nblocks: u32,
    rows_per_block: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = nblocks * nq;
    if tid < total as usize {
        let b = (tid as u32) / nq;
        let q = (tid as u32) % nq;
        let start = b * rows_per_block;
        let mut end = start + rows_per_block;
        if end > n {
            end = n;
        }
        let base = q * n;
        let mut acc = F::new(0.0_f32);
        let mut i = start;
        while i < end {
            acc += quad[(base + i) as usize];
            i += 1u32;
        }
        psums[tid] = acc;
    }
}

/// Stage-2 fold: `out[out_offset + t] = Σ_b partials[b·len + t]`.
///
/// The `out_offset` is what lets a whole evaluation come back in ONE readback:
/// the transposed GEMM's `d` gradient entries are copied into `out[0..d]` by
/// [`huber_copy_into`] and the [`HUBER_QUANTITIES`] scalars are folded into
/// `out[d_aug..]` here, so the estimator synchronizes once instead of twice.
///
/// Generalizes `gmm::gmm_fold_partials`, which writes at `out[t]` only.
#[cube(launch)]
pub fn huber_fold_partials<F: Float + CubeElement>(
    partials: &Array<F>,
    out: &mut Array<F>,
    len: u32,
    nblocks: u32,
    out_offset: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < len as usize {
        let mut acc = F::new(0.0_f32);
        let mut b = 0u32;
        while b < nblocks {
            acc += partials[(b * len + tid as u32) as usize];
            b += 1u32;
        }
        out[(out_offset + tid as u32) as usize] = acc;
    }
}

/// `out[out_offset + t] = src[t]` for `t < len` — the device-to-device gather
/// that puts the transposed GEMM's result into the shared readback buffer.
///
/// A launch rather than a second `to_host` because the readback, not the copy,
/// is what costs: a `d`-element device copy is a few microseconds of an
/// already-running pipeline, while a second `to_host` is a full stall.
#[cube(launch)]
pub fn huber_copy_into<F: Float + CubeElement>(
    src: &Array<F>,
    out: &mut Array<F>,
    len: u32,
    out_offset: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < len as usize {
        out[(out_offset + tid as u32) as usize] = src[tid];
    }
}

/// sklearn's `outliers_` mask as `1.0`/`0.0` floats: `|yᵢ − mᵢ − bias| > σ·ε`
/// at the FITTED parameters, given the margins a single GEMM already produced.
///
/// Emitted as `F` rather than a `bool`/`u32` because the cubecl-cpu MLIR
/// lowering rejects mutable `bool` and this crate keeps ONE element type per
/// kernel; the host thresholds at `!= 0` on read-back. Runs exactly ONCE per
/// fit, which is why it is a separate unfused pass — folding an `n`-length mask
/// write into the per-iteration evaluation would cost a store on every one of
/// the dozens of evaluations to save one pass at the end.
#[cube(launch)]
pub fn huber_outlier_mask_rows<F: Float + CubeElement>(
    margins: &Array<F>,
    y: &Array<F>,
    mask: &mut Array<F>,
    n: u32,
    bias: F,
    thr: F,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let a = (y[i] - (margins[i] + bias)).abs();
        let mut m = F::new(0.0_f32);
        if a > thr {
            m = F::new(1.0_f32);
        }
        mask[i] = m;
    }
}
