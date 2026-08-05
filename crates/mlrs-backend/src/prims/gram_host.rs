//! `gram_host` — the **host** arm of the normal-equations formation
//! (`RIDGE-POS-PERF-CPU`): column means, the centered Gram `XᵀX` and the
//! centered right-hand side `Xᵀy`, computed straight from host memory.
//!
//! ## Why a host arm exists
//! The device composition `center_columns` → `gram_xty` is the right shape on a
//! GPU and a pathology on the cpu backend, for the reason every other cpu
//! campaign in this crate found (`sgd_host`, `hgb_host`, KNN, HDBSCAN, UMAP,
//! ARIMA): `cubecl-cpu` maps ONE OS THREAD PER UNIT and JITs at LLVM `-O0`, so
//! a launch is worth tens of microseconds before it computes anything. Worse,
//! `center_columns`' cpu fallback is `prims::reduce::column_reduce`, which
//! walks the `d` columns ONE AT A TIME with a fresh upload + launch + blocking
//! readback per column (the pathology
//! `mlrs-ridge-optimization`'s colmean kernel fixed for every backend *except*
//! cpu, which is gated off that kernel because MLIR rejects `SharedMemory`).
//!
//! Measured before this module, `Ridge::fit` on the cpu backend at a mere
//! `1 000 × 8`: **60.1 s**, of which `preprocess` was 59.6 s — against
//! scikit-learn's 0.0004 s.
//!
//! ## What it computes
//! [`centered_gram_xty`] returns exactly the operands the `positive` /
//! `solver='lbfgs'` arm needs:
//!
//! ```text
//! x_mean[j] = Σ_r w_r·x[r,j] / Σ_r w_r      (unweighted: the plain mean)
//! G[i,j]    = Σ_r w_r·(x[r,i] − x̄_i)·(x[r,j] − x̄_j)
//! b[i]      = Σ_r w_r·(x[r,i] − x̄_i)·(y_r − ȳ)
//! ```
//!
//! which is the Gram of the `√w`-rescaled, centered design that
//! `ridge.rs::preprocess` would have built — without ever materializing that
//! `n × d` design. Only the `d² + d` outputs are allocated per worker.
//!
//! Centering is applied to the DATA before squaring, not undone afterwards via
//! `XᵀX − n·x̄x̄ᵀ`: the identity form loses catastrophic amounts of precision on
//! a design whose column means dominate its spread, and this codebase's oracle
//! gate is a strict abs-AND-rel `1e-5` (`mlrs_core::compare::is_close`).
//!
//! ## Shape of the inner loop
//! Rows are blocked ([`ROW_BLOCK`]) and TRANSPOSED into a `d × ROW_BLOCK`
//! feature-major tile as they are centered, so the `(i, j)` accumulation reads
//! two CONTIGUOUS runs and LLVM turns it into packed FMAs. The `(i, j)` pairs
//! are then walked in `4 × 4` register blocks, which cuts tile loads per
//! multiply-add from 2 to ½ — without it the loop is L1-bandwidth-bound rather
//! than FLOP-bound. Only the lower triangle is accumulated; the upper is
//! mirrored once at the end.
//!
//! Row blocks are split across [`crate::capability::cpu_launch_units`] scoped
//! threads, sized by WORK (the `linear_predict::host_units` precedent) so a
//! small fit never pays a thread spawn it cannot amortize. Each worker owns a
//! private `d² + d` accumulator and they are summed once at the join.
//!
//! Tests live in `crates/mlrs-backend/tests/gram_host_test.rs` (AGENTS.md §2).

use bytemuck::Pod;

use super::host_simd::avx2_available;

/// Rows per transposed tile.
///
/// The tile is `d × ROW_BLOCK` `f64`, so it must stay small enough to live in
/// L2 across the whole `(i, j)` sweep while still amortizing the transpose:
/// 64 rows is 128 KiB at `d = 256` and 32 KiB at `d = 64`, and the `4 × 4`
/// register block working set (8 lanes × 64 rows = 4 KiB) fits L1 with room to
/// spare at every `d`.
const ROW_BLOCK: usize = 64;

/// Multiply-adds one worker thread must be given before spawning it pays.
///
/// The same work-proportional sizing as `linear_predict::HOST_ELEMS_PER_UNIT`,
/// and for the same reason: spawning and joining a `std::thread` costs tens of
/// microseconds, which is the ENTIRE budget of a small fit (scikit-learn fits
/// `1 000 × 8` in 0.4 ms). Unlike the predict paths this loop is FLOP-bound
/// rather than bandwidth-bound, so the threshold counts multiply-adds
/// (`n · d²/2`), not bytes moved.
const HOST_MACS_PER_UNIT: usize = 1 << 19;

/// Multiply-adds below which the DEVICE arm cannot win on any backend.
///
/// The Gram is `n·d²/2` multiply-adds; at 8.4 M of them the host pass costs
/// ~0.2 ms on a 16-core desktop. The device arm has to upload the design and
/// issue four launches before it computes anything, and that fixed cost is
/// larger than 0.2 ms on every adapter this codebase has measured — ~50 µs per
/// launch on a T4 ([[mlrs-rf-fit-optimization]]) and a 0.53 ms floor on the
/// local wgpu adapter at an `8 000`-element design. So this bound is about
/// DISPATCH, not about how fast the adapter computes, which is why it is
/// applied on every backend rather than tuned per backend.
///
/// Deliberately conservative: the next rung up the local ladder
/// (`n=10 000, d=64`, 41 M MACs) measured at parity between the two arms, so
/// the threshold sits an order of magnitude below the first shape where the
/// answer is in any doubt.
const HOST_FIT_MAX_MACS: usize = 1 << 23;

/// Design elements above which the shape stops being "small" regardless of the
/// multiply-add count — a very wide-`n`, `d = 1` design reaches
/// [`HOST_FIT_MAX_MACS`] while still being a large operand, and there the
/// device's `O(n·d)` streaming is competitive again.
const HOST_FIT_MAX_ELEMS: usize = 1 << 20;

/// Whether the host arm should carry the normal-equations formation for an
/// `n × d` design.
///
/// `true` on the cpu backend — where the device composition is the pathology
/// the module docs describe — and, on ANY backend, below the fixed-cost floor
/// ([`HOST_FIT_MAX_MACS`] / [`HOST_FIT_MAX_ELEMS`]).
///
/// `MLRS_RIDGE_GRAM_HOST=0` forces the device arm back on for A/B, and `=1`
/// forces the host arm at any size on any backend — worth reaching for on an
/// INTEGRATED adapter, where the design upload alone can cost more than the
/// whole host fit (measured locally: 122 ms to upload a 102 MiB design against
/// an 88 ms end-to-end host fit). Read through [`crate::abflag`] rather than
/// `std::env` so a test can scope the override without an environment data
/// race.
pub fn gram_host_applicable(n: usize, d: usize) -> bool {
    if let Some(v) = crate::abflag::var("MLRS_RIDGE_GRAM_HOST") {
        return v != "0";
    }
    if crate::capability::active_backend_name() == "cpu" {
        return true;
    }
    let macs = n.saturating_mul(d).saturating_mul(d) / 2;
    macs <= HOST_FIT_MAX_MACS && n.saturating_mul(d) <= HOST_FIT_MAX_ELEMS
}

/// Column means, target mean, centered Gram and centered `Xᵀy` from host slices.
///
/// - `x` is the `n × d` row-major design, `y` the length-`n` target. Both are
///   read in the caller's element type (`f32` / `f64`) and accumulated in `f64`.
/// - `sample_weight`, when present, is the length-`n` non-negative weight
///   vector; it produces the WEIGHTED mean and the `√w` row rescale in one
///   pass, matching `ridge.rs::preprocess`'s weighted arm.
/// - `fit_intercept = false` leaves both means at zero, so the raw (uncentered)
///   Gram is returned — sklearn's `_preprocess_data` contract.
///
/// Returns `(x_mean, y_mean, gram, xty)` with `gram` the full symmetric `d × d`
/// row-major matrix (the lower triangle is accumulated and mirrored).
///
/// Panics if the slice lengths disagree with `(n, d)`; callers validate
/// geometry first (ASVS V5 — `ridge.rs::fit_from_host_slice`).
pub fn centered_gram_xty<F: Pod>(
    x: &[F],
    y: &[F],
    n: usize,
    d: usize,
    sample_weight: Option<&[f64]>,
    fit_intercept: bool,
) -> (Vec<f64>, f64, Vec<f64>, Vec<f64>) {
    match size_of::<F>() {
        4 => centered_gram_xty_t::<f32>(
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(y),
            n,
            d,
            sample_weight,
            fit_intercept,
        ),
        8 => centered_gram_xty_t::<f64>(
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(y),
            n,
            d,
            sample_weight,
            fit_intercept,
        ),
        other => unreachable!("gram_host is f32/f64 only, got a {other}-byte element"),
    }
}

/// Host element types [`centered_gram_xty`] dispatches to. Deliberately minimal
/// — the arithmetic all happens in `f64`, so the only thing the element type
/// has to do is widen.
trait GramElem: Copy + Send + Sync {
    /// Widen one element to the `f64` the accumulators use.
    fn wide(self) -> f64;
}

impl GramElem for f32 {
    #[inline]
    fn wide(self) -> f64 {
        self as f64
    }
}

impl GramElem for f64 {
    #[inline]
    fn wide(self) -> f64 {
        self
    }
}

/// Monomorphized body of [`centered_gram_xty`].
fn centered_gram_xty_t<T: GramElem>(
    x: &[T],
    y: &[T],
    n: usize,
    d: usize,
    sw: Option<&[f64]>,
    fit_intercept: bool,
) -> (Vec<f64>, f64, Vec<f64>, Vec<f64>) {
    assert_eq!(x.len(), n * d, "gram_host: x length must be n*d");
    assert_eq!(y.len(), n, "gram_host: y length must be n");

    let units = host_units(n, d);
    let (x_mean, y_mean) = if fit_intercept {
        column_means(x, y, n, d, sw, units)
    } else {
        (vec![0.0f64; d], 0.0f64)
    };

    let (gram, xty) = accumulate(x, y, n, d, &x_mean, y_mean, sw, units);
    (x_mean, y_mean, gram, xty)
}

/// Column means, target means, centered Gram and centered `Xᵀy` for MULTIPLE
/// target columns in one pass — the `RidgeClassifier` (multiclass one-hot ±1
/// targets) twin of [`centered_gram_xty`].
///
/// The Gram `G = XᵀX` depends only on `x`, never on `y`, so forming it once
/// and reusing it for every target column (rather than calling
/// [`centered_gram_xty`] once per column, which would re-walk the whole
/// `O(n·d²)` design `k` times) is the entire point of this function: only the
/// `O(n·d·k)` `Xᵀy` term actually needs a per-column pass, and even that shares
/// the SAME centered-and-transposed tile the Gram sweep already built.
///
/// - `x` is the `n × d` row-major design, `y` the `n × k` row-major target
///   matrix (`k` target columns per row, e.g. sklearn's one-hot `±1`
///   `LabelBinarizer` output). Both are read in the caller's element type and
///   accumulated in `f64`.
/// - `sample_weight` / `fit_intercept` behave exactly as in
///   [`centered_gram_xty`].
///
/// Returns `(x_mean[d], y_means[k], gram[d·d], xty[d·k])`, `xty` row-major
/// (`xty[i·k + c]` is feature `i`'s dot with target column `c`). `k = 1`
/// reproduces [`centered_gram_xty`] bit-for-bit (same accumulation order).
///
/// Panics if the slice lengths disagree with `(n, d)` / `(n, k)`; callers
/// validate geometry first, as `centered_gram_xty` requires.
pub fn centered_gram_multi_xty<F: Pod>(
    x: &[F],
    y: &[F],
    n: usize,
    d: usize,
    k: usize,
    sample_weight: Option<&[f64]>,
    fit_intercept: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    match size_of::<F>() {
        4 => centered_gram_multi_xty_t::<f32>(
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(y),
            n,
            d,
            k,
            sample_weight,
            fit_intercept,
        ),
        8 => centered_gram_multi_xty_t::<f64>(
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(y),
            n,
            d,
            k,
            sample_weight,
            fit_intercept,
        ),
        other => unreachable!("gram_host is f32/f64 only, got a {other}-byte element"),
    }
}

/// Monomorphized body of [`centered_gram_multi_xty`].
fn centered_gram_multi_xty_t<T: GramElem>(
    x: &[T],
    y: &[T],
    n: usize,
    d: usize,
    k: usize,
    sw: Option<&[f64]>,
    fit_intercept: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    assert_eq!(x.len(), n * d, "gram_host: x length must be n*d");
    assert_eq!(y.len(), n * k, "gram_host: y length must be n*k");

    let units = host_units(n, d);
    let (x_mean, y_means) = if fit_intercept {
        column_means_multi(x, y, n, d, k, sw, units)
    } else {
        (vec![0.0f64; d], vec![0.0f64; k])
    };

    let (gram, xty) = accumulate_multi(x, y, n, d, k, &x_mean, &y_means, sw, units);
    (x_mean, y_means, gram, xty)
}

/// Multi-column twin of [`column_means`].
fn column_means_multi<T: GramElem>(
    x: &[T],
    y: &[T],
    n: usize,
    d: usize,
    k: usize,
    sw: Option<&[f64]>,
    units: usize,
) -> (Vec<f64>, Vec<f64>) {
    let (mut sums, mut y_sums, w_sum) = if units <= 1 {
        column_sums_multi(x, y, d, k, sw.map(|w| (w, 0usize)))
    } else {
        let rows = n.div_ceil(units);
        let partials: Vec<(Vec<f64>, Vec<f64>, f64)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..units)
                .filter_map(|u| {
                    let r0 = u * rows;
                    if r0 >= n {
                        return None;
                    }
                    let r1 = (r0 + rows).min(n);
                    let xs = &x[r0 * d..r1 * d];
                    let ys = &y[r0 * k..r1 * k];
                    Some(scope.spawn(move || column_sums_multi(xs, ys, d, k, sw.map(|w| (w, r0)))))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("gram_host_multi column-mean worker panicked"))
                .collect()
        });
        let mut s = vec![0.0f64; d];
        let mut ys = vec![0.0f64; k];
        let mut ws = 0.0f64;
        for (ps, py, pw) in partials {
            for (a, b) in s.iter_mut().zip(ps.iter()) {
                *a += *b;
            }
            for (a, b) in ys.iter_mut().zip(py.iter()) {
                *a += *b;
            }
            ws += pw;
        }
        (s, ys, ws)
    };

    if w_sum > 0.0 {
        for m in sums.iter_mut() {
            *m /= w_sum;
        }
        for m in y_sums.iter_mut() {
            *m /= w_sum;
        }
    }
    (sums, y_sums)
}

/// Multi-column twin of [`column_sums`].
fn column_sums_multi<T: GramElem>(
    x: &[T],
    y: &[T],
    d: usize,
    k: usize,
    sw: Option<(&[f64], usize)>,
) -> (Vec<f64>, Vec<f64>, f64) {
    let rows = y.len() / k;
    let mut sums = vec![0.0f64; d];
    let mut y_sums = vec![0.0f64; k];
    match sw {
        None => {
            for r in 0..rows {
                let row = &x[r * d..(r + 1) * d];
                for (s, v) in sums.iter_mut().zip(row.iter()) {
                    *s += v.wide();
                }
                let yrow = &y[r * k..(r + 1) * k];
                for (s, v) in y_sums.iter_mut().zip(yrow.iter()) {
                    *s += v.wide();
                }
            }
            (sums, y_sums, rows as f64)
        }
        Some((w, r0)) => {
            let mut w_sum = 0.0f64;
            for r in 0..rows {
                let wr = w[r0 + r];
                if wr == 0.0 {
                    continue;
                }
                let row = &x[r * d..(r + 1) * d];
                for (s, v) in sums.iter_mut().zip(row.iter()) {
                    *s += wr * v.wide();
                }
                let yrow = &y[r * k..(r + 1) * k];
                for (s, v) in y_sums.iter_mut().zip(yrow.iter()) {
                    *s += wr * v.wide();
                }
                w_sum += wr;
            }
            (sums, y_sums, w_sum)
        }
    }
}

/// Multi-column twin of [`accumulate`]: the Gram accumulation is IDENTICAL to
/// the single-column sweep (it never reads `y`); only `xty` grows from a
/// length-`d` vector to a length-`d·k` row-major matrix.
#[allow(clippy::too_many_arguments)]
fn accumulate_multi<T: GramElem>(
    x: &[T],
    y: &[T],
    n: usize,
    d: usize,
    k: usize,
    x_mean: &[f64],
    y_means: &[f64],
    sw: Option<&[f64]>,
    units: usize,
) -> (Vec<f64>, Vec<f64>) {
    let (mut gram, xty) = if units <= 1 {
        chunk_gram_multi(x, y, d, k, x_mean, y_means, sw.map(|w| (w, 0usize)))
    } else {
        let rows = n.div_ceil(units);
        let partials: Vec<(Vec<f64>, Vec<f64>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..units)
                .filter_map(|u| {
                    let r0 = u * rows;
                    if r0 >= n {
                        return None;
                    }
                    let r1 = (r0 + rows).min(n);
                    let xs = &x[r0 * d..r1 * d];
                    let ys = &y[r0 * k..r1 * k];
                    Some(scope.spawn(move || {
                        chunk_gram_multi(xs, ys, d, k, x_mean, y_means, sw.map(|w| (w, r0)))
                    }))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("gram_host_multi accumulate worker panicked"))
                .collect()
        });
        let mut g = vec![0.0f64; d * d];
        let mut b = vec![0.0f64; d * k];
        for (pg, pb) in partials {
            for (a, v) in g.iter_mut().zip(pg.iter()) {
                *a += *v;
            }
            for (a, v) in b.iter_mut().zip(pb.iter()) {
                *a += *v;
            }
        }
        (g, b)
    };

    for i in 0..d {
        for j in 0..i {
            gram[j * d + i] = gram[i * d + j];
        }
    }
    (gram, xty)
}

/// Multi-column twin of [`chunk_gram`]: centers (+ `√w`-scales) both `x` and
/// EVERY column of `y` into transposed tiles, then reuses [`block4x4`]/[`dot`]
/// for the Gram and one [`dot`] per `(feature, target-column)` pair for `xty`.
#[allow(clippy::too_many_arguments)]
fn chunk_gram_multi<T: GramElem>(
    x: &[T],
    y: &[T],
    d: usize,
    k: usize,
    x_mean: &[f64],
    y_means: &[f64],
    sw: Option<(&[f64], usize)>,
) -> (Vec<f64>, Vec<f64>) {
    let mut gram = vec![0.0f64; d * d];
    let mut xty = vec![0.0f64; d * k];
    let mut tile = vec![0.0f64; d * ROW_BLOCK];
    let mut ycol = vec![0.0f64; k * ROW_BLOCK];

    let rows = y.len() / k;
    let mut r0 = 0usize;
    while r0 < rows {
        let rb = ROW_BLOCK.min(rows - r0);

        for r in 0..rb {
            let scale = match sw {
                None => 1.0,
                Some((w, base)) => w[base + r0 + r].sqrt(),
            };
            let row = &x[(r0 + r) * d..(r0 + r + 1) * d];
            for j in 0..d {
                tile[j * ROW_BLOCK + r] = (row[j].wide() - x_mean[j]) * scale;
            }
            let yrow = &y[(r0 + r) * k..(r0 + r + 1) * k];
            for c in 0..k {
                ycol[c * ROW_BLOCK + r] = (yrow[c].wide() - y_means[c]) * scale;
            }
        }

        dispatch_sweep_block_multi(&tile, &ycol, rb, d, k, &mut gram, &mut xty);
        r0 += rb;
    }

    (gram, xty)
}

/// Multi-column twin of [`sweep_block`]: the Gram half is copy-pasted verbatim
/// (same `4 × 4` register-blocked sweep over the `x` tile alone); the `xty`
/// half loops over the `k` target columns instead of accumulating one.
#[inline(always)]
fn sweep_block_multi(
    tile: &[f64],
    ycol: &[f64],
    rb: usize,
    d: usize,
    k: usize,
    gram: &mut [f64],
    xty: &mut [f64],
) {
    let full = d - d % 4;

    let mut i0 = 0usize;
    while i0 < full {
        let mut j0 = 0usize;
        while j0 < i0 {
            let mut acc = [0.0f64; 16];
            block4x4(tile, i0, j0, rb, &mut acc);
            for a in 0..4 {
                for b in 0..4 {
                    gram[(i0 + a) * d + (j0 + b)] += acc[a * 4 + b];
                }
            }
            j0 += 4;
        }
        let mut acc = [0.0f64; 16];
        block4x4(tile, i0, i0, rb, &mut acc);
        for a in 0..4 {
            for b in 0..=a {
                gram[(i0 + a) * d + (i0 + b)] += acc[a * 4 + b];
            }
        }
        i0 += 4;
    }

    for i in full..d {
        let ti = &tile[i * ROW_BLOCK..i * ROW_BLOCK + rb];
        for j in 0..=i {
            let tj = &tile[j * ROW_BLOCK..j * ROW_BLOCK + rb];
            gram[i * d + j] += dot(ti, tj);
        }
    }
    for i in 0..full {
        let ti = &tile[i * ROW_BLOCK..i * ROW_BLOCK + rb];
        for j in full..=i {
            let tj = &tile[j * ROW_BLOCK..j * ROW_BLOCK + rb];
            gram[i * d + j] += dot(ti, tj);
        }
    }

    for i in 0..d {
        let ti = &tile[i * ROW_BLOCK..i * ROW_BLOCK + rb];
        for c in 0..k {
            let yc = &ycol[c * ROW_BLOCK..c * ROW_BLOCK + rb];
            xty[i * k + c] += dot(ti, yc);
        }
    }
}

/// Worker threads to split the fit across — see [`HOST_MACS_PER_UNIT`]. Never
/// more than the machine offers ([`crate::capability::cpu_launch_units`], which
/// `MLRS_CPU_UNITS` overrides for A/B), never fewer than one, and never more
/// than there are row blocks to hand out.
fn host_units(n: usize, d: usize) -> usize {
    let macs = (n.saturating_mul(d).saturating_mul(d) / 2).max(1);
    (macs / HOST_MACS_PER_UNIT)
        .clamp(1, crate::capability::cpu_launch_units().max(1) as usize)
        .min(n.div_ceil(ROW_BLOCK).max(1))
}

/// Pass 1: `x_mean[j] = Σ_r w_r·x[r,j] / Σ_r w_r` and the matching `y_mean`,
/// accumulated in `f64` over contiguous row chunks in parallel.
///
/// With no `sample_weight` the weight sum is exactly `n`, so this is the plain
/// column mean and the multiply is skipped entirely.
fn column_means<T: GramElem>(
    x: &[T],
    y: &[T],
    n: usize,
    d: usize,
    sw: Option<&[f64]>,
    units: usize,
) -> (Vec<f64>, f64) {
    let (mut sums, mut y_sum, w_sum) = if units <= 1 {
        column_sums(x, y, d, sw.map(|w| (w, 0usize)))
    } else {
        let rows = n.div_ceil(units);
        let partials: Vec<(Vec<f64>, f64, f64)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..units)
                .filter_map(|u| {
                    let r0 = u * rows;
                    if r0 >= n {
                        return None;
                    }
                    let r1 = (r0 + rows).min(n);
                    let xs = &x[r0 * d..r1 * d];
                    let ys = &y[r0..r1];
                    Some(scope.spawn(move || column_sums(xs, ys, d, sw.map(|w| (w, r0)))))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("gram_host column-mean worker panicked"))
                .collect()
        });
        let mut s = vec![0.0f64; d];
        let mut ys = 0.0f64;
        let mut ws = 0.0f64;
        for (ps, py, pw) in partials {
            for (a, b) in s.iter_mut().zip(ps.iter()) {
                *a += *b;
            }
            ys += py;
            ws += pw;
        }
        (s, ys, ws)
    };

    // A zero weight sum means every weight is zero; the caller rejects that
    // before reaching here (`AlgoError::ZeroSampleWeightSum`), but guarding the
    // divide keeps a NaN out of the means if a new caller ever forgets.
    if w_sum > 0.0 {
        for m in sums.iter_mut() {
            *m /= w_sum;
        }
        y_sum /= w_sum;
    }
    (sums, y_sum)
}

/// Column sums (and the target sum + weight sum) over one contiguous row chunk.
/// `sw` carries the FULL weight vector plus this chunk's first global row index,
/// because the weights are indexed by absolute row.
fn column_sums<T: GramElem>(
    x: &[T],
    y: &[T],
    d: usize,
    sw: Option<(&[f64], usize)>,
) -> (Vec<f64>, f64, f64) {
    let mut sums = vec![0.0f64; d];
    let mut y_sum = 0.0f64;
    match sw {
        None => {
            for (r, yv) in y.iter().enumerate() {
                let row = &x[r * d..(r + 1) * d];
                for (s, v) in sums.iter_mut().zip(row.iter()) {
                    *s += v.wide();
                }
                y_sum += yv.wide();
            }
            (sums, y_sum, y.len() as f64)
        }
        Some((w, r0)) => {
            let mut w_sum = 0.0f64;
            for (r, yv) in y.iter().enumerate() {
                let wr = w[r0 + r];
                if wr == 0.0 {
                    continue;
                }
                let row = &x[r * d..(r + 1) * d];
                for (s, v) in sums.iter_mut().zip(row.iter()) {
                    *s += wr * v.wide();
                }
                y_sum += wr * yv.wide();
                w_sum += wr;
            }
            (sums, y_sum, w_sum)
        }
    }
}

/// Pass 2: the centered Gram and `Xᵀy`, split across contiguous row chunks.
/// Each worker owns a private `d² + d` `f64` accumulator; the chunks are summed
/// once at the join.
#[allow(clippy::too_many_arguments)]
fn accumulate<T: GramElem>(
    x: &[T],
    y: &[T],
    n: usize,
    d: usize,
    x_mean: &[f64],
    y_mean: f64,
    sw: Option<&[f64]>,
    units: usize,
) -> (Vec<f64>, Vec<f64>) {
    let (mut gram, xty) = if units <= 1 {
        chunk_gram(x, y, d, x_mean, y_mean, sw.map(|w| (w, 0usize)))
    } else {
        let rows = n.div_ceil(units);
        let partials: Vec<(Vec<f64>, Vec<f64>)> = std::thread::scope(|scope| {
            let handles: Vec<_> =
                (0..units)
                    .filter_map(|u| {
                        let r0 = u * rows;
                        if r0 >= n {
                            return None;
                        }
                        let r1 = (r0 + rows).min(n);
                        let xs = &x[r0 * d..r1 * d];
                        let ys = &y[r0..r1];
                        Some(scope.spawn(move || {
                            chunk_gram(xs, ys, d, x_mean, y_mean, sw.map(|w| (w, r0)))
                        }))
                    })
                    .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("gram_host accumulate worker panicked"))
                .collect()
        });
        let mut g = vec![0.0f64; d * d];
        let mut b = vec![0.0f64; d];
        for (pg, pb) in partials {
            for (a, v) in g.iter_mut().zip(pg.iter()) {
                *a += *v;
            }
            for (a, v) in b.iter_mut().zip(pb.iter()) {
                *a += *v;
            }
        }
        (g, b)
    };

    // Only the lower triangle was accumulated (the `j <= i` sweep); mirror it.
    for i in 0..d {
        for j in 0..i {
            gram[j * d + i] = gram[i * d + j];
        }
    }
    (gram, xty)
}

/// The centered Gram lower triangle + `Xᵀy` over one contiguous row chunk.
///
/// Rows are consumed [`ROW_BLOCK`] at a time: each block is centered, optionally
/// `√w`-scaled, and transposed into a `d × ROW_BLOCK` feature-major tile, after
/// which the `(i, j)` sweep is pure contiguous-vs-contiguous dot products.
fn chunk_gram<T: GramElem>(
    x: &[T],
    y: &[T],
    d: usize,
    x_mean: &[f64],
    y_mean: f64,
    sw: Option<(&[f64], usize)>,
) -> (Vec<f64>, Vec<f64>) {
    let mut gram = vec![0.0f64; d * d];
    let mut xty = vec![0.0f64; d];
    let mut tile = vec![0.0f64; d * ROW_BLOCK];
    let mut ycol = vec![0.0f64; ROW_BLOCK];

    let rows = y.len();
    let mut r0 = 0usize;
    while r0 < rows {
        let rb = ROW_BLOCK.min(rows - r0);

        // --- Center (+ √w scale) and transpose the block into the tile. The
        //     write is strided and the later reads are contiguous, which is the
        //     trade that makes the O(n·d²) sweep vectorize. ---
        for r in 0..rb {
            let scale = match sw {
                None => 1.0,
                Some((w, base)) => w[base + r0 + r].sqrt(),
            };
            let row = &x[(r0 + r) * d..(r0 + r + 1) * d];
            for j in 0..d {
                tile[j * ROW_BLOCK + r] = (row[j].wide() - x_mean[j]) * scale;
            }
            ycol[r] = (y[r0 + r].wide() - y_mean) * scale;
        }

        dispatch_sweep_block(&tile, &ycol, rb, d, &mut gram, &mut xty);
        r0 += rb;
    }

    (gram, xty)
}

/// `gram[i,j] += Σ_r tile[i,r]·tile[j,r]` for `j <= i`, plus
/// `xty[i] += Σ_r tile[i,r]·ycol[r]`, over one transposed tile.
///
/// Walked in `4 × 4` register blocks: 8 tile loads feed 16 multiply-adds, so
/// the loop is FLOP-bound instead of L1-bound (a plain pair-at-a-time dot
/// sweep loads 2 values per multiply-add). The `i`/`j` tails past the last full
/// group of four fall back to the scalar pair sweep.
#[inline(always)]
fn sweep_block(tile: &[f64], ycol: &[f64], rb: usize, d: usize, gram: &mut [f64], xty: &mut [f64]) {
    let full = d - d % 4;

    let mut i0 = 0usize;
    while i0 < full {
        // Square j-blocks strictly below the diagonal: the whole 4 × 4 result
        // is inside the lower triangle, so every lane is stored.
        let mut j0 = 0usize;
        while j0 < i0 {
            let mut acc = [0.0f64; 16];
            block4x4(tile, i0, j0, rb, &mut acc);
            for a in 0..4 {
                for b in 0..4 {
                    gram[(i0 + a) * d + (j0 + b)] += acc[a * 4 + b];
                }
            }
            j0 += 4;
        }
        // Diagonal block: same kernel, but only the `j <= i` lanes are stored.
        let mut acc = [0.0f64; 16];
        block4x4(tile, i0, i0, rb, &mut acc);
        for a in 0..4 {
            for b in 0..=a {
                gram[(i0 + a) * d + (i0 + b)] += acc[a * 4 + b];
            }
        }
        i0 += 4;
    }

    // `i` tail (fewer than four remaining features): scalar pair sweep over the
    // full row, and the `j` tail of every earlier row.
    for i in full..d {
        let ti = &tile[i * ROW_BLOCK..i * ROW_BLOCK + rb];
        for j in 0..=i {
            let tj = &tile[j * ROW_BLOCK..j * ROW_BLOCK + rb];
            gram[i * d + j] += dot(ti, tj);
        }
    }
    for i in 0..full {
        let ti = &tile[i * ROW_BLOCK..i * ROW_BLOCK + rb];
        for j in full..=i {
            let tj = &tile[j * ROW_BLOCK..j * ROW_BLOCK + rb];
            gram[i * d + j] += dot(ti, tj);
        }
    }

    for i in 0..d {
        let ti = &tile[i * ROW_BLOCK..i * ROW_BLOCK + rb];
        xty[i] += dot(ti, &ycol[..rb]);
    }
}

/// The `4 × 4` micro-kernel: sixteen independent reduction chains over `rb`
/// contiguous tile elements, which LLVM keeps in vector accumulators and feeds
/// with eight loads per iteration.
#[inline]
fn block4x4(tile: &[f64], i0: usize, j0: usize, rb: usize, acc: &mut [f64; 16]) {
    let ia = &tile[i0 * ROW_BLOCK..i0 * ROW_BLOCK + rb];
    let ib = &tile[(i0 + 1) * ROW_BLOCK..(i0 + 1) * ROW_BLOCK + rb];
    let ic = &tile[(i0 + 2) * ROW_BLOCK..(i0 + 2) * ROW_BLOCK + rb];
    let id = &tile[(i0 + 3) * ROW_BLOCK..(i0 + 3) * ROW_BLOCK + rb];
    let ja = &tile[j0 * ROW_BLOCK..j0 * ROW_BLOCK + rb];
    let jb = &tile[(j0 + 1) * ROW_BLOCK..(j0 + 1) * ROW_BLOCK + rb];
    let jc = &tile[(j0 + 2) * ROW_BLOCK..(j0 + 2) * ROW_BLOCK + rb];
    let jd = &tile[(j0 + 3) * ROW_BLOCK..(j0 + 3) * ROW_BLOCK + rb];

    let mut a = [0.0f64; 16];
    for r in 0..rb {
        let (u0, u1, u2, u3) = (ia[r], ib[r], ic[r], id[r]);
        let (v0, v1, v2, v3) = (ja[r], jb[r], jc[r], jd[r]);
        a[0] += u0 * v0;
        a[1] += u0 * v1;
        a[2] += u0 * v2;
        a[3] += u0 * v3;
        a[4] += u1 * v0;
        a[5] += u1 * v1;
        a[6] += u1 * v2;
        a[7] += u1 * v3;
        a[8] += u2 * v0;
        a[9] += u2 * v1;
        a[10] += u2 * v2;
        a[11] += u2 * v3;
        a[12] += u3 * v0;
        a[13] += u3 * v1;
        a[14] += u3 * v2;
        a[15] += u3 * v3;
    }
    for (s, v) in acc.iter_mut().zip(a.iter()) {
        *s += *v;
    }
}

/// `a·b` over two equal-length contiguous `f64` runs, split across eight
/// independent accumulators so the FP-add latency is hidden and LLVM can keep
/// the chains in vector registers (the `linear_predict::host_dot` shape). No
/// `mul_add`: without `target-feature=+fma` it lowers to a `fma` LIBRARY CALL.
/// Run the `O(n·d²)` sweep on the machine's REAL vector unit.
///
/// [`block4x4`]'s sixteen accumulator chains are `f64`, so the baseline SSE2 the
/// crate is compiled for gives them TWO lanes where this machine has four (AVX2)
/// or eight (AVX-512). The chains are independent, so widening them reassociates
/// nothing and the Gram is bit-for-bit what it was — see
/// [`host_simd`](super::host_simd) for the full argument, the measurement, and
/// why this is written as an explicit twin rather than a closure helper.
#[inline]
fn dispatch_sweep_block(
    tile: &[f64],
    ycol: &[f64],
    rb: usize,
    d: usize,
    gram: &mut [f64],
    xty: &mut [f64],
) {
    #[cfg(target_arch = "x86_64")]
    if avx2_available() {
        // SAFETY: guarded by the runtime detection this branch tests; the body is
        // the ordinary `sweep_block`, which contains nothing unsafe.
        unsafe {
            sweep_block_avx2(tile, ycol, rb, d, gram, xty);
        }
        return;
    }
    sweep_block(tile, ycol, rb, d, gram, xty);
}

/// [`sweep_block`] compiled for AVX2 + FMA — see [`dispatch_sweep_block`].
///
/// # Safety
/// The caller must have established that the CPU supports `avx2` and `fma`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sweep_block_avx2(
    tile: &[f64],
    ycol: &[f64],
    rb: usize,
    d: usize,
    gram: &mut [f64],
    xty: &mut [f64],
) {
    sweep_block(tile, ycol, rb, d, gram, xty);
}

/// Multi-target twin of [`dispatch_sweep_block`].
#[inline]
fn dispatch_sweep_block_multi(
    tile: &[f64],
    ycol: &[f64],
    rb: usize,
    d: usize,
    k: usize,
    gram: &mut [f64],
    xty: &mut [f64],
) {
    #[cfg(target_arch = "x86_64")]
    if avx2_available() {
        // SAFETY: as `dispatch_sweep_block`.
        unsafe {
            sweep_block_multi_avx2(tile, ycol, rb, d, k, gram, xty);
        }
        return;
    }
    sweep_block_multi(tile, ycol, rb, d, k, gram, xty);
}

/// [`sweep_block_multi`] compiled for AVX2 + FMA.
///
/// # Safety
/// The caller must have established that the CPU supports `avx2` and `fma`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sweep_block_multi_avx2(
    tile: &[f64],
    ycol: &[f64],
    rb: usize,
    d: usize,
    k: usize,
    gram: &mut [f64],
    xty: &mut [f64],
) {
    sweep_block_multi(tile, ycol, rb, d, k, gram, xty);
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    const LANES: usize = 8;
    let mut acc = [0.0f64; LANES];
    let n = a.len().min(b.len());
    let chunks = n / LANES;
    for c in 0..chunks {
        let base = c * LANES;
        for (l, s) in acc.iter_mut().enumerate() {
            *s += a[base + l] * b[base + l];
        }
    }
    let mut tail = 0.0f64;
    for k in chunks * LANES..n {
        tail += a[k] * b[k];
    }
    acc.iter().sum::<f64>() + tail
}
