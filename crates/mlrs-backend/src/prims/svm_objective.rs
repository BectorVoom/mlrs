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
//! auto-vectorizes, split across [`host_units`] workers.
//!
//! ## Why the workers are a PERSISTENT pool, not per-evaluation threads
//! The first version of that host arm fanned out with `std::thread::scope`
//! INSIDE `eval`. That is the right shape for a `predict`, which makes exactly
//! one pass — and the wrong one here, because a fit calls `eval` 25-40 times
//! (once per L-BFGS iteration and per line-search step) and every one of those
//! calls was re-spawning the whole pool. The spawn is not a fixed small cost
//! either: it is tens of microseconds per worker when a core is free and
//! unbounded when one is not, so a wider split made it worse rather than
//! better and the whole arm collapsed on a machine with other work on it.
//!
//! The evaluator now owns a [`WorkerPool`] for the LIFETIME OF THE SOLVE:
//! threads are spawned once in [`SvmObjective::new`] and each evaluation costs
//! two barrier crossings instead of `units` thread spawns. Measured on a
//! 16-core Zen5, f32, `LinearSVC.fit` wall-clock (min over 27 fits in separate
//! processes, quiet machine):
//!
//! | n × d | per-eval spawn | persistent pool |
//! |---|---|---|
//! | 1 000 × 16 | 0.318 ms | **0.181 ms** |
//! | 10 000 × 16 | 1.558 ms | **0.836 ms** |
//! | 10 000 × 64 | 3.688 ms | **1.622 ms** |
//! | 100 000 × 16 | 9.595 ms | **4.337 ms** |
//! | 50 000 × 64 | 11.141 ms | **4.977 ms** |
//! | 200 000 × 64 | 62.587 ms | **43.061 ms** |
//!
//! scikit-learn's liblinear on the same rungs: 0.617 / 4.593 / 14.445 / 119.9 /
//! 205.4 / 993.8 ms.
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

#[cfg(not(feature = "cpu"))]
use std::marker::PhantomData;

use mlrs_core::PrimError;
// Both conversions are device-arm-only: the cpu arm's fused pass is
// monomorphized on the concrete `f32`/`f64` element and never round-trips a
// value through the generic `F` bit-cast.
#[cfg(not(feature = "cpu"))]
use mlrs_core::{f64_to_host, host_to_f64};

use crate::device_array::DeviceArray;
#[cfg(feature = "cpu")]
use crate::prims::host_pool::{Shared, WorkerPool};
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

/// Whether [`SvmDesign::Host`] is the CHEAPER operand form on this backend.
///
/// True exactly where the objective evaluates from host memory (cpu), which is
/// where handing the caller's slab over directly removes three full passes over
/// it. On the device backends the design has to reach the GEMM operand anyway,
/// so a host slab would only be copied an extra time on its way there — those
/// callers should keep uploading and pass [`SvmDesign::Device`]. The
/// [`sgd_host_available`](crate::prims::sgd::sgd_host_available) precedent.
///
/// Both forms are always ACCEPTED; this only says which one to prefer.
pub fn svm_host_ingress_preferred() -> bool {
    cfg!(feature = "cpu")
}

/// Where the design the evaluator reads comes from.
///
/// The solve is identical either way; what differs is how many times the
/// `n × d` slab is copied before the first evaluation. A caller that already
/// HAS the design in host memory — the Arrow buffer at the PyO3 boundary — can
/// hand it over directly instead of uploading it to a [`DeviceArray`] the cpu
/// arm would immediately read straight back (the no-upload host-slice ingress,
/// `mbsgd_classifier::fit_from_host_slice` precedent). On the cpu backend that
/// removes THREE full passes over the design (`from_host` copies twice,
/// `to_host` once) from every fit.
pub enum SvmDesign<'a, F> {
    /// Device-resident, `n × d` row-major — the [`Fit`](crate) trait's operand.
    Device(&'a DeviceArray<ActiveRuntime, F>),
    /// Host-resident, `n × d` row-major — the caller's own buffer.
    Host(&'a [F]),
}

impl<F: Float + CubeElement + Pod> SvmDesign<'_, F> {
    /// Element count, whichever form this is.
    fn len(&self) -> usize {
        match self {
            SvmDesign::Device(x) => x.len(),
            SvmDesign::Host(x) => x.len(),
        }
    }
}

/// The cpu arm's design: the caller's own buffer when it was already host
/// resident, an owned copy only when it had to be pulled off the device.
///
/// A `Cow` would say the same thing but demands `[F]: ToOwned` — a `Clone`
/// bound this evaluator otherwise never needs — on the struct and on every
/// signature that names it.
#[cfg(feature = "cpu")]
enum HostDesign<'a, F> {
    /// The caller's slab, read in place (the no-upload ingress).
    Borrowed(&'a [F]),
    /// Pulled off the device because that is where the caller had it.
    Owned(Vec<F>),
}

#[cfg(feature = "cpu")]
impl<F> HostDesign<'_, F> {
    #[inline]
    fn as_slice(&self) -> &[F] {
        match self {
            HostDesign::Borrowed(s) => s,
            HostDesign::Owned(v) => v,
        }
    }
}

/// The design matrix in whatever form the active backend evaluates against,
/// prepared ONCE per `fit` and reused by every L-BFGS iteration and line-search
/// step (the bounded-allocation iterative-solver shape, 05-11).
///
/// - **cpu**: the host `n × d` slab, read in place (borrowed outright when the
///   caller supplied [`SvmDesign::Host`]), plus the [`WorkerPool`] the fused
///   pass runs on.
/// - **wgpu / cuda / rocm**: the device-resident `n × d_aug` augmented design
///   the two GEMMs read.
pub struct SvmObjective<'a, F> {
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
    /// cpu arm: the design read in place, unaugmented — BORROWED from the
    /// caller when it was already host-resident, owned only when it had to be
    /// pulled off the device.
    #[cfg(feature = "cpu")]
    x_host: HostDesign<'a, F>,
    /// cpu arm: the threads the fused pass is split across, spawned ONCE for
    /// the whole solve. `None` below the parallel knee ([`SVM_ELEMS_PER_UNIT`]),
    /// where the pass runs inline on the calling thread.
    #[cfg(feature = "cpu")]
    workers: Option<WorkerPool>,
    /// device arms: the augmented `n × d_aug` design both GEMMs read.
    #[cfg(not(feature = "cpu"))]
    x_aug: DeviceArray<ActiveRuntime, F>,
    /// Binds `'a` on the device arms, which borrow nothing.
    #[cfg(not(feature = "cpu"))]
    _borrow: PhantomData<&'a [F]>,
}

impl<'a, F> SvmObjective<'a, F>
where
    F: Float + CubeElement + Pod,
{
    /// Prepare the evaluator for an `n × d` row-major design.
    ///
    /// `targets` is length `n`. `fit_intercept` appends the synthetic
    /// `intercept_scaling` column (Pitfall 5) — materialized into the device
    /// operand on the device arms, folded into the arithmetic on the cpu arm.
    ///
    /// Geometry is validated before anything is allocated: `n·d == x.len()`,
    /// `targets.len() == n`, both dims non-zero.
    pub fn new(
        pool: &mut BufferPool<ActiveRuntime>,
        x: SvmDesign<'a, F>,
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
            let x_host = match x {
                SvmDesign::Host(slab) => HostDesign::Borrowed(slab),
                SvmDesign::Device(dev) => HostDesign::Owned(dev.to_host(pool)),
            };
            // Spawned ONCE for the whole solve, not once per evaluation: the
            // L-BFGS driver calls `eval` 25-40 times and `std::thread` setup
            // dominated every one of them (see [`SVM_ELEMS_PER_UNIT`]).
            let units = host_units(n * d).min(n.max(1));
            Ok(Self {
                n,
                d,
                d_aug,
                intercept_scaling,
                targets,
                x_host,
                workers: (units > 1).then(|| WorkerPool::new(units)),
            })
        }
        #[cfg(not(feature = "cpu"))]
        {
            // The device arms feed `prims::gemm`, which reads one contiguous
            // operand — so the synthetic column has to be materialized here.
            let x_host: Vec<F> = match x {
                SvmDesign::Device(dev) => dev.to_host(pool),
                SvmDesign::Host(slab) => slab.to_vec(),
            };
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
                _borrow: PhantomData,
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
impl<F> SvmObjective<'_, F>
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
        let x: &[T] = bytemuck::cast_slice(self.x_host.as_slice());
        let (n, d, d_aug) = (self.n, self.d, self.d_aug);
        // The synthetic column contributes a CONSTANT `intercept_scaling·w[d]`
        // to every margin, so it is hoisted out of the row loop entirely.
        let (bias, scale) = if d_aug > d {
            (self.intercept_scaling * w[d], self.intercept_scaling)
        } else {
            (0.0, 0.0)
        };
        let wd = &w[..d];

        let Some(workers) = self.workers.as_ref() else {
            let mut acc = Accum::new(d, d_aug, scale);
            acc.rows(x, wd, bias, &self.targets, loss);
            return acc.finish();
        };

        // Contiguous row chunks: unit `u` owns rows `[u·rows, u·rows + k)`, so
        // its design slab and its target run are both unbroken ranges.
        let units = workers.units();
        let rows = n.div_ceil(units);
        let mut partials: Vec<Accum> = (0..units)
            .map(|_| Accum::new(d, d_aug, scale))
            .collect();
        {
            // SAFETY (`Shared`'s contract): unit `u` is the only writer of
            // `partials[u]` within the pass, and `run` does not return until
            // every unit has finished — the barrier release is what publishes
            // the writes to the reducing thread below.
            let slots = Shared::new(&mut partials);
            // Bound out of `self` rather than captured through it: the pool is
            // deliberately `!Sync` (one driver per pool), so a closure holding
            // `&self` — which owns the pool — could not itself be `Sync` and so
            // could not be dispatched. The pass needs only the targets.
            let targets = &self.targets;
            workers.run(&|u: usize| {
                let lo = (u * rows).min(n);
                let hi = (lo + rows).min(n);
                if lo == hi {
                    return;
                }
                let acc = unsafe { &mut slots.get_mut()[u] };
                acc.rows(&x[lo * d..hi * d], wd, bias, &targets[lo..hi], loss);
            });
        }

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

/// Design elements one worker must be given before splitting the pass pays.
///
/// This constant was originally `1 << 16`, and that value was a SPAWN-cost
/// amortization: with a fresh `std::thread` per worker per evaluation, a
/// worker had to be handed enough bytes for its setup to disappear against
/// DRAM bandwidth, and splitting more finely than that lost outright.
///
/// The persistent [`WorkerPool`] (module docs) removes that cost, so the
/// break-even moved down with it — what a worker must now cover is only two
/// barrier crossings, which are microseconds. Re-measured with the pool in
/// place, `LinearSVC.fit` wall-clock on a 16-core Zen5, f32, min over 21 fits
/// in separate processes on a quiet machine:
///
/// | n × d | `1<<11` | `1<<12` | `1<<13` | `1<<14` | `1<<15` | `1<<16` |
/// |---|---|---|---|---|---|---|
/// | 1 000 × 16 | 0.194 | 0.242 | 0.180 | 0.193 | 0.182 | 0.184 |
/// | 5 000 × 16 | 0.538 | 0.573 | 0.449 | 0.465 | 0.695 | — |
/// | 10 000 × 16 | 0.718 | 0.723 | 0.837 | 0.640 | 0.714 | 1.159 |
/// | 10 000 × 64 | 1.642 | 1.214 | 1.903 | 1.450 | 1.271 | 1.328 |
/// | 50 000 × 16 | 2.066 | 2.000 | 2.084 | 2.029 | 2.769 | — |
/// | 100 000 × 16 | 4.263 | 4.211 | 4.299 | 4.314 | 4.309 | 4.330 |
/// | 50 000 × 64 | 4.826 | 4.914 | 4.932 | 4.991 | 5.033 | 4.845 |
///
/// The curve is FLAT from `1<<11` to `1<<15` — the spread within a row there is
/// run-to-run noise with no trend — and only the old `1<<16` is distinguishable,
/// costing 31 % at `10 000 × 16` because it holds that fit to two workers when
/// the machine has sixteen. `1 << 14` sits in the middle of the flat region with
/// margin on both sides, so it is neither at the noisy small edge nor near the
/// value that is measurably wrong.
///
/// Note this only bounds units from BELOW-the-work side. The upper bound is
/// [`crate::capability::cpu_launch_units`], and on the largest rungs (where the
/// pass is DRAM-bound) the two bounds meet: at `200 000 × 128` the fit is 157 ms
/// on one worker, 88 ms on two, 86 ms on four and 96 ms on sixteen — bandwidth
/// saturates well before the core count does, so extra workers past ~4 buy
/// nothing and eventually cost a little. That residual is under 10 % and is left
/// alone rather than fitted with a second constant.
#[cfg(feature = "cpu")]
const SVM_ELEMS_PER_UNIT: usize = 1 << 14;

/// Worker threads to split `elems` design elements across — see
/// [`SVM_ELEMS_PER_UNIT`]. Never more than the machine offers
/// ([`crate::capability::cpu_launch_units`], which `MLRS_CPU_UNITS` overrides
/// for A/B), never fewer than one.
///
/// `MLRS_SVM_ELEMS_PER_UNIT` overrides the knee itself for on-target A/B
/// (through [`crate::abflag`], never a raw `std::env::var` — see its docs).
#[cfg(feature = "cpu")]
fn host_units(elems: usize) -> usize {
    let knee = crate::abflag::var("MLRS_SVM_ELEMS_PER_UNIT")
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(SVM_ELEMS_PER_UNIT);
    (elems / knee).clamp(1, crate::capability::cpu_launch_units().max(1) as usize)
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
impl<F> SvmObjective<'_, F>
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
