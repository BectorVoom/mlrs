//! Linear-SVM primal objective evaluator (SVM-FIT-CPU) — the one function the
//! `LinearSVC` / `LinearSVR` L-BFGS solve calls per iteration and line-search
//! step.
//!
//! Both linear SVMs minimize `½‖w‖² + C·Σᵢ ℓ(mᵢ, tᵢ)` over the
//! synthetic-feature-augmented design `x̃ = [x | intercept_scaling]`, where
//! `mᵢ = x̃ᵢ·w`. Every evaluation therefore needs exactly two things from the
//! design matrix:
//!
//! 1. the **margins** `m = x̃·w` (an `n`-length matvec), and
//! 2. the **data-term gradient** `x̃ᵀ·g` where `gᵢ = ∂ℓ/∂mᵢ` (a `d_aug`-length
//!    transposed matvec).
//!
//! [`SvmObjective`] owns whatever operand form the active backend wants and
//! evaluates both in one call, so the estimators never hand-roll the launch (or
//! the host loop) themselves.
//!
//! ## Why the cpu backend does NOT go through the device GEMM
//! The original evaluator ran both matvecs as `prims::gemm` launches with a
//! host round-trip on each side: upload `w`, GEMM, read `m` back, host loss
//! loop, upload `g`, GEMM (transa), read `x̃ᵀg` back. On a discrete GPU that is
//! the right shape — the design is already device-resident and the two
//! crossings are `n` and `d_aug` floats against an `n·d` matvec.
//!
//! On the **cpu** backend it is a catastrophe, and not by a small factor.
//! Instrumented `LinearSVC.fit` on a 16-core Zen5, `n = 5000`, `d_aug = 17`,
//! f64, 31 objective evaluations (`MLRS_SVM_PROBE=1`):
//!
//! | stage | total | per eval |
//! |---|---|---|
//! | margin GEMM launch + its blocking read-back | 302 ms | 9.7 ms |
//! | `x̃ᵀg` GEMM launch + its blocking read-back | 89 ms | 2.9 ms |
//! | the host loss/gradient loop over all `n` samples | 0.22 ms | 7 µs |
//! | `w` / `g` uploads | 0.15 ms | 5 µs |
//!
//! The arithmetic is 85 000 multiply-adds per matvec. 9.7 ms for that is over
//! three orders of magnitude off the machine's roofline, and the whole fit came
//! to 392 ms against scikit-learn's 2.7 ms. The reason is the
//! `cubecl-cpu` execution model (see the `mlrs-cubecl-cpu-execution-model`
//! notes): kernels are JIT-compiled at LLVM **`-O0`** — no vectorizer, no
//! unroller — and a launch spawns one OS **thread per unit**, so a GEMM shaped
//! for a GPU pays thousands of thread spawns to do work a single core finishes
//! in tens of microseconds. No amount of kernel tuning pays that back; the
//! launch itself is the cost.
//!
//! [`SvmObjective::eval`]'s cpu arm therefore does the whole evaluation on the
//! host, compiled into this crate at the release profile's `-O3` where it
//! auto-vectorizes, split across [`host_units`] scoped threads (the
//! `linear_predict_host` precedent, with its own knee — see
//! [`SVM_ELEMS_PER_UNIT`]).
//!
//! ## One fused pass, and no augmented copy
//! The host arm also removes two things the device arm cannot:
//!
//! - **Two passes over the design become one.** Row `i` needs its dot product
//!   with `w` and then contributes `gᵢ·x̃ᵢ` to the gradient. Both touch the SAME
//!   row, so the fused loop computes `mᵢ`, evaluates `ℓ`, and accumulates the
//!   gradient while the row is still in L1 — the design is streamed from memory
//!   ONCE per evaluation instead of twice. These matvecs are bandwidth-bound at
//!   the feature counts a linear SVM is fitted at, so that is close to a 2×.
//! - **The augmented design is never materialized.** The synthetic column is a
//!   constant, so `mᵢ = xᵢ·w[..d] + intercept_scaling·w[d]` and
//!   `x̃ᵀg[d] = intercept_scaling·Σgᵢ` reproduce it exactly. The device arm has
//!   to build and upload an `n × (d+1)` copy of the design because a GEMM reads
//!   a contiguous operand; the host arm reads the caller's `n × d` slab in
//!   place.
//!
//! ## Accumulation precision (host arm)
//! The host arm accumulates margins and gradients in **`f64` regardless of
//! `F`**, where the device GEMM accumulates in `F`. This is deliberate and is a
//! correctness fix, not just a fidelity nicety: with `F = f32` the gradient sum
//! `Σᵢ gᵢ·xᵢⱼ` over `n` samples carries a round-off floor around
//! `√n·ε_f32`, which at `n ≥ 1000` sits ABOVE the default `tol = 1e-4`
//! stopping gradient. The f32 solve could then never satisfy its own tolerance
//! and `fit` returned `NotConverged` on inputs it had actually solved (see
//! `svm_objective_test.rs::f32_fit_converges_at_default_tol`). Widening `f32`
//! lanes to `f64` costs one conversion per element on a pass that is
//! memory-bound on the `f32` load anyway.
//!
//! Tests live in `crates/mlrs-backend/tests/svm_objective_test.rs` (AGENTS.md
//! §2), never an in-source `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_core::PrimError;
// Both conversions are device-arm-only: the cpu arm's fused pass is
// monomorphized on the concrete `f32`/`f64` element and never round-trips a
// value through the generic `F` bit-cast.
#[cfg(not(feature = "cpu"))]
use mlrs_core::{f64_to_host, host_to_f64};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Per-sample margin loss: `(ℓ(margin, target), ∂ℓ/∂margin)`.
///
/// `LinearSVC` passes squared hinge, `LinearSVR` squared epsilon-insensitive.
/// `Sync` because the cpu arm evaluates it from several worker threads at once;
/// both estimators' closures capture only `Copy` scalars, so this is free.
pub trait MarginLoss: Sync {
    /// `(loss_i, dloss/dmargin)` at this sample's margin and target.
    fn eval(&self, margin: f64, target: f64) -> (f64, f64);
}

impl<T> MarginLoss for T
where
    T: Fn(f64, f64) -> (f64, f64) + Sync,
{
    #[inline]
    fn eval(&self, margin: f64, target: f64) -> (f64, f64) {
        self(margin, target)
    }
}

/// What one objective evaluation produces: the summed data-term loss
/// `Σᵢ ℓ(mᵢ, tᵢ)` and the data-term gradient `x̃ᵀ·g` (length `d_aug`).
///
/// The caller adds the regularizer (`½‖w‖²` / `w`) and the `C` weight — those
/// are scalar host arithmetic over `d_aug` values and belong with the estimator,
/// not the operand.
#[derive(Debug, Clone)]
pub struct SvmEval {
    /// `Σᵢ ℓ(mᵢ, tᵢ)` over every sample.
    pub data_loss: f64,
    /// `x̃ᵀ·g`, length `d_aug` (`n_features + 1` when fitting an intercept).
    pub xtg: Vec<f64>,
}

/// The design matrix in whatever form the active backend evaluates against,
/// prepared ONCE per `fit` and reused by every L-BFGS iteration and line-search
/// step (the bounded-allocation iterative-solver shape, 05-11).
///
/// - **cpu**: the host `n × d` slab, read in place. No augmented copy, no
///   device operand, no launch (module docs).
/// - **wgpu / cuda / rocm**: the device-resident `n × d_aug` augmented design
///   the two GEMMs read.
pub struct SvmObjective<F> {
    /// Sample count.
    n: usize,
    /// UNaugmented feature count. cpu arm only: the device arms materialize the
    /// synthetic column into the operand at construction, so nothing downstream
    /// of `new` needs to know where the unaugmented design ends.
    #[cfg(feature = "cpu")]
    d: usize,
    /// Augmented weight length: `d + 1` when fitting an intercept, else `d`.
    d_aug: usize,
    /// The synthetic column's constant value (Pitfall 5); inert when
    /// `d_aug == d`. cpu arm only, for the same reason as `d`.
    #[cfg(feature = "cpu")]
    intercept_scaling: f64,
    /// Per-sample targets (±1 labels for SVC, regression targets for SVR),
    /// length `n`, host-resident because the loss is evaluated on the host on
    /// every backend.
    targets: Vec<f64>,
    /// cpu arm: the design read in place, unaugmented.
    #[cfg(feature = "cpu")]
    x_host: Vec<F>,
    /// device arms: the augmented `n × d_aug` design both GEMMs read.
    #[cfg(not(feature = "cpu"))]
    x_aug: DeviceArray<ActiveRuntime, F>,
}

impl<F> SvmObjective<F>
where
    F: Float + CubeElement + Pod,
{
    /// Prepare the evaluator for an `n × d` row-major device-resident design.
    ///
    /// `targets` is length `n`. `fit_intercept` appends the synthetic
    /// `intercept_scaling` column (Pitfall 5) — materialized into the device
    /// operand on the device arms, folded into the arithmetic on the cpu arm.
    ///
    /// Geometry is validated before anything is allocated: `n·d == x.len()`,
    /// `targets.len() == n`, both dims non-zero.
    pub fn new(
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        (n, d): (usize, usize),
        targets: Vec<f64>,
        intercept_scaling: f64,
        fit_intercept: bool,
    ) -> Result<Self, PrimError> {
        if n == 0 || d == 0 || n.checked_mul(d).map(|v| v != x.len()).unwrap_or(true) {
            return Err(PrimError::ShapeMismatch {
                operand: "x",
                rows: n,
                cols: d,
                len: x.len(),
            });
        }
        if targets.len() != n {
            return Err(PrimError::ShapeMismatch {
                operand: "targets",
                rows: n,
                cols: 1,
                len: targets.len(),
            });
        }
        let d_aug = if fit_intercept { d + 1 } else { d };

        #[cfg(feature = "cpu")]
        {
            Ok(Self {
                n,
                d,
                d_aug,
                intercept_scaling,
                targets,
                x_host: x.to_host(pool),
            })
        }
        #[cfg(not(feature = "cpu"))]
        {
            // The device arms feed `prims::gemm`, which reads one contiguous
            // operand — so the synthetic column has to be materialized here.
            let x_host = x.to_host(pool);
            let _ = intercept_scaling;
            let mut aug: Vec<F> = vec![f64_to_host::<F>(0.0); n * d_aug];
            for r in 0..n {
                aug[r * d_aug..r * d_aug + d].copy_from_slice(&x_host[r * d..(r + 1) * d]);
                if fit_intercept {
                    aug[r * d_aug + d] = f64_to_host::<F>(intercept_scaling);
                }
            }
            Ok(Self {
                n,
                d_aug,
                targets,
                x_aug: DeviceArray::from_host(pool, &aug),
            })
        }
    }

    /// The augmented weight length the caller's `w` must have.
    pub fn d_aug(&self) -> usize {
        self.d_aug
    }

    /// Evaluate `Σᵢ ℓ(x̃ᵢ·w, tᵢ)` and `x̃ᵀ·g` at the augmented weights `w`.
    ///
    /// `w.len()` must equal [`d_aug`](Self::d_aug).
    ///
    /// Generic over the loss rather than taking `&dyn MarginLoss`: the cpu arm
    /// calls it ONCE PER SAMPLE inside the fused row loop, where an indirect
    /// call would both cost more than the row's own arithmetic at the feature
    /// counts a linear SVM is fitted at AND block the loop from vectorizing.
    /// Monomorphizing inlines the (small, branchy) loss body into the loop.
    pub fn eval<L: MarginLoss>(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        w: &[f64],
        loss: &L,
    ) -> Result<SvmEval, PrimError> {
        if w.len() != self.d_aug {
            return Err(PrimError::DimMismatch {
                dim: "d_aug",
                lhs: w.len(),
                rhs: self.d_aug,
            });
        }
        #[cfg(feature = "cpu")]
        {
            let _ = pool;
            Ok(self.eval_host(w, loss))
        }
        #[cfg(not(feature = "cpu"))]
        {
            self.eval_device(pool, w, loss)
        }
    }

    /// Release the operand back to the pool, consuming the evaluator.
    pub fn release_into(self, pool: &mut BufferPool<ActiveRuntime>) {
        #[cfg(feature = "cpu")]
        {
            let _ = pool;
        }
        #[cfg(not(feature = "cpu"))]
        {
            self.x_aug.release_into(pool);
        }
    }
}

// ---------------------------------------------------------------------------
// cpu arm — the fused host pass
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu")]
impl<F> SvmObjective<F>
where
    F: Float + CubeElement + Pod,
{
    /// One fused pass over the design (module docs): per row, the margin dot
    /// product, the loss/gradient scalar, and the gradient accumulation, split
    /// across scoped threads on contiguous row chunks.
    ///
    /// The `F` → `f32`/`f64` dispatch mirrors `linear_predict_host`: the row
    /// loop is monomorphized on the CONCRETE element type so it vectorizes,
    /// rather than going through `host_to_f64` element by element.
    fn eval_host<L: MarginLoss>(&self, w: &[f64], loss: &L) -> SvmEval {
        match size_of::<F>() {
            4 => self.eval_host_typed::<f32, L>(w, loss),
            8 => self.eval_host_typed::<f64, L>(w, loss),
            other => unreachable!("linear SVM is f32/f64 only, got a {other}-byte element"),
        }
    }

    fn eval_host_typed<T: HostElem, L: MarginLoss>(&self, w: &[f64], loss: &L) -> SvmEval {
        let x: &[T] = bytemuck::cast_slice(&self.x_host);
        let (n, d, d_aug) = (self.n, self.d, self.d_aug);
        // The synthetic column contributes a CONSTANT `intercept_scaling·w[d]`
        // to every margin, so it is hoisted out of the row loop entirely.
        let (bias, scale) = if d_aug > d {
            (self.intercept_scaling * w[d], self.intercept_scaling)
        } else {
            (0.0, 0.0)
        };
        let wd = &w[..d];

        let units = host_units(n * d).min(n.max(1));
        if units <= 1 {
            let mut acc = Accum::new(d, d_aug, scale);
            acc.rows(x, wd, bias, &self.targets, loss);
            return acc.finish();
        }

        // Contiguous row chunks: thread `i` owns rows `[i·rows, i·rows + k)`,
        // so its design slab and its target run are both unbroken ranges.
        let rows = n.div_ceil(units);
        let partials: Vec<Accum> = std::thread::scope(|scope| {
            let handles: Vec<_> = self
                .targets
                .chunks(rows)
                .enumerate()
                .map(|(i, tchunk)| {
                    let slab = &x[i * rows * d..(i * rows + tchunk.len()) * d];
                    scope.spawn(move || {
                        let mut acc = Accum::new(d, d_aug, scale);
                        acc.rows(slab, wd, bias, tchunk, loss);
                        acc
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("svm objective row worker panicked"))
                .collect()
        });

        let mut total = Accum::new(d, d_aug, scale);
        for p in partials {
            total.data_loss += p.data_loss;
            total.gsum += p.gsum;
            for (t, v) in total.xtg.iter_mut().zip(p.xtg.iter()) {
                *t += *v;
            }
        }
        total.finish()
    }
}

/// One worker's partial `(Σℓ, x̃ᵀg)`, plus the separate `Σgᵢ` the synthetic
/// column's gradient entry is derived from (it is `intercept_scaling·Σgᵢ`, so
/// tracking the plain sum keeps the constant out of the inner loop).
#[cfg(feature = "cpu")]
struct Accum {
    data_loss: f64,
    gsum: f64,
    /// Length `d_aug`; entry `d` (when present) is filled at [`Accum::finish`].
    xtg: Vec<f64>,
    /// The synthetic column's `intercept_scaling`, or `0.0` when unaugmented.
    scale: f64,
    /// Unaugmented feature count — where the synthetic entry lands.
    d: usize,
}

#[cfg(feature = "cpu")]
impl Accum {
    /// `d` is the UNaugmented feature count and `scale` the synthetic column's
    /// `intercept_scaling` (`0.0` when unaugmented) — both carried on the
    /// accumulator so a REDUCED partial still knows where the synthetic entry
    /// lands and what constant it takes.
    fn new(d: usize, d_aug: usize, scale: f64) -> Self {
        Self {
            data_loss: 0.0,
            gsum: 0.0,
            xtg: vec![0.0; d_aug],
            scale,
            d,
        }
    }

    /// The fused row loop: `mᵢ = xᵢ·w + bias`, `(ℓᵢ, gᵢ) = loss(mᵢ, tᵢ)`,
    /// `xtg += gᵢ·xᵢ`. Both uses of row `i` happen while it is in L1, so the
    /// design is streamed once (module docs).
    ///
    /// Accumulates in `f64` whatever `T` is — the round-off floor that widening
    /// removes is what let an f32 solve miss its own `tol` (module docs).
    fn rows<T: HostElem, L: MarginLoss>(
        &mut self,
        x: &[T],
        w: &[f64],
        bias: f64,
        targets: &[f64],
        loss: &L,
    ) {
        let d = self.d;
        let g_acc = &mut self.xtg[..d];
        for (r, &t) in targets.iter().enumerate() {
            let row = &x[r * d..(r + 1) * d];
            let margin = T::dot(row, w) + bias;
            let (li, gi) = loss.eval(margin, t);
            self.data_loss += li;
            // A zero subgradient is the COMMON case for both SVM losses (every
            // sample outside the margin / inside the epsilon tube), and skipping
            // its axpy skips the whole `d`-element accumulate.
            if gi != 0.0 {
                self.gsum += gi;
                T::axpy(g_acc, row, gi);
            }
        }
    }

    /// Fold the hoisted synthetic-column entry in and hand back the result.
    fn finish(mut self) -> SvmEval {
        if self.xtg.len() > self.d {
            self.xtg[self.d] = self.scale * self.gsum;
        }
        SvmEval {
            data_loss: self.data_loss,
            xtg: self.xtg,
        }
    }
}

/// The concrete host element types the fused pass is monomorphized over, with
/// the two kernels it needs written against `f64` accumulators.
///
/// Like `linear_predict`'s `HostFloat` this deliberately avoids `mul_add`:
/// without `target-feature=+fma` (which the default `x86-64` baseline lacks)
/// `mul_add` lowers to a LIBRARY CALL, an order of magnitude slower than the
/// `mul`+`add` pair LLVM vectorizes here.
#[cfg(feature = "cpu")]
trait HostElem: Pod + Copy + Send + Sync {
    /// `Σⱼ row[j]·w[j]` in `f64`, over [`DOT_LANES`] independent accumulators.
    /// `w` is `f64` whatever `Self` is, so an `f32` design still accumulates
    /// wide (module docs).
    fn dot(row: &[Self], w: &[f64]) -> f64;

    /// `acc[j] += g·row[j]` in `f64`.
    fn axpy(acc: &mut [f64], row: &[Self], g: f64);
}

/// Design elements one worker thread must be given before spawning it pays,
/// and the reason it is EIGHT TIMES smaller than `linear_predict`'s otherwise
/// identical knee (`1 << 18`).
///
/// A `predict` is ONE streaming pass, so its sizing is a pure spawn-cost
/// amortization: give a thread enough bytes that `std::thread` setup disappears
/// against DRAM bandwidth. A `fit` re-reads the SAME design 20–100 times (once
/// per L-BFGS iteration and line-search step), so the dominant effect is
/// instead whether each worker's slab stays resident in its core's private L2
/// between evaluations. Splitting more finely than the spawn cost alone would
/// justify buys cache residency that a coarser split gives up.
///
/// That shows up as a large, superlinear cliff rather than a gentle slope.
/// `LinearSVR.fit` wall-clock on a 16-core Zen5, best of 9, by forced knee:
///
/// | n × d (dtype) | `1<<18` | `1<<17` | `1<<16` | `1<<15` | `1<<14` |
/// |---|---|---|---|---|---|
/// | 10 000 × 64 (f32) | 10.8 ms | 3.90 ms | **3.56 ms** | 3.89 ms | 3.84 ms |
/// | 10 000 × 64 (f64) | 8.71 ms | — | **4.11 ms** | 4.48 ms | 4.65 ms |
/// | 50 000 × 16 (f32) | 6.79 ms | 4.61 ms | 5.25 ms | 5.58 ms | 5.39 ms |
/// | 100 000 × 16 (f32) | 8.61 ms | 9.05 ms | 8.35 ms | 8.53 ms | 8.38 ms |
/// | 1 000 × 16 (f32) | 0.36 ms | 0.40 ms | 0.23 ms | 0.34 ms | 0.39 ms |
///
/// At `10 000 × 64` the coarse knee leaves each of 2 threads a 1.3 MiB slab
/// that spills L2 on every one of the 27 evaluations; at `1<<16` the same work
/// splits 9 ways into ~285 KiB slabs that stay resident, and the fit is 3×
/// faster. Below `1<<16` the curve is flat — the split is already fine enough —
/// while the small end (`1 000 × 16`, and the `1 000`-row rungs generally) sits
/// under one unit at every candidate, so its spread is run-to-run noise on an
/// identical single-threaded path, not a knee effect. `1 << 16` is the smallest
/// value that captures the cliff without fanning small fits out needlessly.
#[cfg(feature = "cpu")]
const SVM_ELEMS_PER_UNIT: usize = 1 << 16;

/// Worker threads to split `elems` design elements across — see
/// [`SVM_ELEMS_PER_UNIT`]. Never more than the machine offers
/// ([`crate::capability::cpu_launch_units`], which `MLRS_CPU_UNITS` overrides
/// for A/B), never fewer than one.
#[cfg(feature = "cpu")]
fn host_units(elems: usize) -> usize {
    (elems / SVM_ELEMS_PER_UNIT)
        .clamp(1, crate::capability::cpu_launch_units().max(1) as usize)
}

/// Independent accumulators the dot product is split across — the natural SIMD
/// group. At `-O3` LLVM keeps the fixed-size `[f64; 8]` in AVX registers and
/// turns the body into multiply-add pairs, while the 8 independent chains hide
/// FP-add latency. Also divides the feature counts linear SVMs are actually
/// fitted at (16, 64), so the scalar remainder is usually empty.
#[cfg(feature = "cpu")]
const DOT_LANES: usize = 8;

#[cfg(feature = "cpu")]
impl HostElem for f64 {
    #[inline]
    fn dot(row: &[f64], w: &[f64]) -> f64 {
        let mut acc = [0.0f64; DOT_LANES];
        let mut rows = row.chunks_exact(DOT_LANES);
        let mut ws = w.chunks_exact(DOT_LANES);
        for (xc, wc) in rows.by_ref().zip(ws.by_ref()) {
            for i in 0..DOT_LANES {
                acc[i] += xc[i] * wc[i];
            }
        }
        let mut sum = 0.0;
        for a in acc {
            sum += a;
        }
        for (xv, wv) in rows.remainder().iter().zip(ws.remainder()) {
            sum += *xv * *wv;
        }
        sum
    }

    #[inline]
    fn axpy(acc: &mut [f64], row: &[f64], g: f64) {
        for (a, &xv) in acc.iter_mut().zip(row) {
            *a += g * xv;
        }
    }
}

#[cfg(feature = "cpu")]
impl HostElem for f32 {
    #[inline]
    fn dot(row: &[f32], w: &[f64]) -> f64 {
        let mut acc = [0.0f64; DOT_LANES];
        let mut rows = row.chunks_exact(DOT_LANES);
        let mut ws = w.chunks_exact(DOT_LANES);
        for (xc, wc) in rows.by_ref().zip(ws.by_ref()) {
            for i in 0..DOT_LANES {
                acc[i] += xc[i] as f64 * wc[i];
            }
        }
        let mut sum = 0.0;
        for a in acc {
            sum += a;
        }
        for (xv, wv) in rows.remainder().iter().zip(ws.remainder()) {
            sum += *xv as f64 * *wv;
        }
        sum
    }

    #[inline]
    fn axpy(acc: &mut [f64], row: &[f32], g: f64) {
        for (a, &xv) in acc.iter_mut().zip(row) {
            *a += g * xv as f64;
        }
    }
}

// ---------------------------------------------------------------------------
// device arms — the original two-GEMM evaluator
// ---------------------------------------------------------------------------

#[cfg(not(feature = "cpu"))]
impl<F> SvmObjective<F>
where
    F: Float + CubeElement + Pod,
{
    /// `m = X̃·w` then `X̃ᵀ·g`, both as `prims::gemm` launches with the loss
    /// evaluated on the host between them (the shape a discrete GPU wants: the
    /// design never leaves the device and only `n + d_aug` floats cross).
    fn eval_device<L: MarginLoss>(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        w: &[f64],
        loss: &L,
    ) -> Result<SvmEval, PrimError> {
        use crate::prims::gemm::gemm;

        let (n, d_aug) = (self.n, self.d_aug);
        let w_host: Vec<F> = w.iter().map(|&v| f64_to_host::<F>(v)).collect();
        let w_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &w_host);
        let margins = match gemm::<F>(
            pool,
            &self.x_aug,
            (n, d_aug),
            &w_dev,
            (d_aug, 1),
            false,
            false,
            None,
        ) {
            Ok(m) => m,
            Err(e) => {
                w_dev.release_into(pool);
                return Err(e);
            }
        };
        let margins_host = margins.to_host(pool);
        margins.release_into(pool);
        w_dev.release_into(pool);

        let mut data_loss = 0.0f64;
        let mut g: Vec<F> = vec![f64_to_host::<F>(0.0); n];
        for i in 0..n {
            let (li, gi) = loss.eval(host_to_f64(margins_host[i]), self.targets[i]);
            data_loss += li;
            g[i] = f64_to_host::<F>(gi);
        }
        let g_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &g);

        // The LOGICAL op is (d_aug × n)·(n × 1); the stored X̃ is (n × d_aug) so
        // `transa` presents the transposed view (gemm.rs:78).
        let xtg = match gemm::<F>(
            pool,
            &self.x_aug,
            (d_aug, n),
            &g_dev,
            (n, 1),
            true,
            false,
            None,
        ) {
            Ok(v) => v,
            Err(e) => {
                g_dev.release_into(pool);
                return Err(e);
            }
        };
        let xtg_host = xtg.to_host(pool);
        xtg.release_into(pool);
        g_dev.release_into(pool);

        Ok(SvmEval {
            data_loss,
            xtg: xtg_host.iter().map(|&v| host_to_f64(v)).collect(),
        })
    }
}
