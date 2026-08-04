//! `sgd` — minibatch-SGD device kernels (PRIM-10, SGDSVM-01..02).
//!
//! Seven feature-free `#[cube]` kernels carry the FULL device work of ONE
//! minibatch SGD step, so the host epoch loop is launch-only (no per-batch
//! device↔host synchronization — the cuML-parity requirement; a host readback
//! per batch made CUDA training latency-bound, not compute-bound). The host
//! still owns the learning-rate schedule (`optimal` t0 / `invscaling`) and the
//! scalar penalty factors (all f64), feeding them as by-value scalars:
//!
//! 1. [`sgd_margin`] (pass 1, per-sample) computes the linear margin
//!    `p[i] = Σ_j x[(row_offset+i)*d + j]·w[j] + bias[0]` over the `b`
//!    minibatch rows — the over-provisioned per-element map (`ABSOLUTE_POS`,
//!    bounds-checked). `x` is the FULL `n × d` design; `row_offset` selects the
//!    batch rows IN PLACE (no per-batch host slice/re-upload). The intercept is
//!    device-resident (`bias`, length 1).
//! 2. [`sgd_grad`] (per-sample) applies the `dloss` subgradient table to the
//!    margin: `g[i] = clamp(dloss(p[i], y[row_offset+i]), ±1e12)`. The loss
//!    family is selected by the by-value `loss_id` scalar (see
//!    `prims::sgd::loss_id` for the table). Replaces the host f64 `dloss` pass
//!    (which forced a per-batch `p[]` readback + `g[]` upload).
//! 3. [`sgd_weight_update`] (pass 2, per-coordinate) applies the SGD coordinate
//!    update `w[j] = (w[j] - eta·inv_b·Σ_i g[i]·x[(row_offset+i)*d + j]) ·
//!    l2_factor` — one unit per coordinate `j` GATHERS the minibatch gradient
//!    (single-owner reduction) and writes its own `w[j]`. `l2_factor` fuses the
//!    lazy-L2 shrink that previously required a host round-trip (`1.0` when no
//!    L2 penalty applies — an exact multiplicative identity).
//! 4. [`sgd_l1_shrink`] (per-coordinate) applies sklearn's cumulative-L1
//!    soft-shrink (`l1penalty`): per coordinate, replays the `b` per-sample
//!    budget steps `u += du` with the running `q[j]` correction — the exact
//!    per-sample sequence the old host loop ran, now single-owner on device.
//! 5. [`sgd_bias_update`] (single-unit) folds `bias[0] -= eta·inv_b·Σ_i g[i]`
//!    — the intercept step, kept device-resident.
//! 6. [`sgd_loss`] (per-sample) computes the sklearn loss VALUE `loss(p[i],
//!    y[row_offset+i])` — as opposed to [`sgd_grad`]'s derivative — selected by
//!    the same by-value `loss_id` (MBSGD-PERF-WGPU: replaces a coefficient-delta
//!    convergence check with sklearn's own training-loss-plateau one).
//! 7. [`sgd_sumloss`] (single-unit) folds this batch's `Σ_i loss[i]` into the
//!    running epoch `sumloss[0]` accumulator — the host reads the 1-element
//!    `sumloss` ONCE per epoch for the `tol`/`n_iter_no_change` gate.
//!
//! ## cubecl-cpu MLIR safety (the primary correctness gate)
//! The cpu(f64) backend's MLIR lowering rejects shared-memory tiles + mutable
//! `bool` flags + the floating infinity constant + descending-shift loops (plan
//! 05-02 + spike 001/002). All kernels here use ONLY `F`/`u32` accumulators,
//! `if`-guarded forward loops, and statement-form `if` for conditional
//! assignment — no shared-memory tile, no `bool`, no infinity sentinel, no
//! `if`-expression in value position, single-owner reductions/no scatter. The
//! per-sample/per-element kernels use the proven bare-`ABSOLUTE_POS` map shape;
//! the single-unit kernels use the `CUBE_POS_X`/`UNIT_POS_X == 0` `top_k` shape
//! (spike finding 002-A: never a bare-`ABSOLUTE_POS` per-row loop launch with
//! `CubeDim {x:1}`).
//!
//! All kernels are generic over `<F: Float + CubeElement>` and carry NO backend
//! feature (D-13).
//!
//! Tests live in `crates/mlrs-backend/tests/sgd_test.rs` (AGENTS.md §2 — never an
//! in-source `#[cfg(test)] mod tests`).

use cubecl::prelude::*;

pub use self::sgd_bias_update as sgd_bias_update_kernel;
pub use self::sgd_grad as sgd_grad_kernel;
pub use self::sgd_l1_shrink as sgd_l1_shrink_kernel;
pub use self::sgd_loss as sgd_loss_kernel;
pub use self::sgd_margin as sgd_margin_kernel;
pub use self::sgd_sumloss as sgd_sumloss_kernel;
pub use self::sgd_weight_update as sgd_weight_update_kernel;

/// Pass 1 (per-sample margin): `p[i] = Σ_j x[(row_offset+i)*d + j]·w[j] +
/// bias[0]` over the `b` minibatch rows of the FULL row-major `n × d` design
/// `x` and the length-`d` weight vector `w`.
///
/// - `x` is the FULL row-major `n × d` design matrix; `row_offset` selects the
///   batch's first row (no per-batch host slice — the batch is addressed in
///   place).
/// - `w` is the length-`d` weight vector.
/// - `bias` is the length-1 DEVICE-RESIDENT intercept (read at `[0]`).
/// - `p` is the length-`b` output margin vector (`p[i]`).
/// - `row_offset`, `b`, `d` are scalar args passed BY VALUE (cubecl 0.10 — no
///   `ScalarArg`).
///
/// One unit handles one row at `ABSOLUTE_POS`; bounds-checked on `i < b` so the
/// ceiling-division launch may over-provision threads safely. The dot is a
/// forward `while` scan seeded `F::from_int(0i64)` — no shared-memory tile, no
/// `bool`, no infinity sentinel (cubecl-cpu MLIR-safe).
#[cube(launch)]
pub fn sgd_margin<F: Float + CubeElement>(
    x: &Array<F>,
    w: &Array<F>,
    bias: &Array<F>,
    p: &mut Array<F>,
    row_offset: u32,
    b: u32,
    d: u32,
) {
    let i = ABSOLUTE_POS;
    if i < b as usize {
        let mut acc = F::from_int(0i64);
        let mut j = 0u32;
        // Forward accumulation (no descending shift, no bool flag, no infinity).
        while j < d {
            let idx = (row_offset as usize + i) * d as usize + j as usize;
            acc += x[idx] * w[j as usize];
            j += 1u32;
        }
        p[i] = acc + bias[0];
    }
}

/// Per-sample subgradient: `g[i] = clamp(dloss(p[i], y[row_offset+i]), ±1e12)`
/// — the device form of the host `dloss` table (`prims::sgd::dloss`), selected
/// by the by-value `loss_id` scalar:
///
/// | `loss_id` | loss | `dloss(p, y)` |
/// |---|---|---|
/// | 0 | hinge | `-y` if `p·y ≤ 1` else `0` |
/// | 1 | log | `-y / (1 + exp(y·p))` |
/// | 2 | squared hinge | `-2·y·(1 − p·y)` if `1 − p·y > 0` else `0` |
/// | 3 | squared error | `p − y` |
/// | 4 | ε-insensitive | `∓1` outside the ε-tube else `0` |
/// | 5 | squared ε-insensitive | `∓2·(|y−p| − ε)` outside the tube else `0` |
///
/// `y` is the FULL length-`n` target; `row_offset` aligns it with the batch's
/// `p[]`. One unit per sample at `ABSOLUTE_POS` (the proven per-element map
/// shape); statement-form `if` chains only (no `if`-expression in value
/// position, no `bool` accumulator), `F::exp` in the launch-proven static form.
#[cube(launch)]
pub fn sgd_grad<F: Float + CubeElement>(
    p: &Array<F>,
    y: &Array<F>,
    g: &mut Array<F>,
    row_offset: u32,
    b: u32,
    loss_id: u32,
    epsilon: F,
) {
    let i = ABSOLUTE_POS;
    if i < b as usize {
        let pi = p[i];
        let yi = y[row_offset as usize + i];
        let zero = F::from_int(0i64);
        let one = F::from_int(1i64);
        let two = F::from_int(2i64);
        let mut gi = zero;
        if loss_id == 0u32 {
            // Hinge: z = p·y; z <= 1 → -y.
            let z = pi * yi;
            if z <= one {
                gi = zero - yi;
            }
        }
        if loss_id == 1u32 {
            // Log: -y / (1 + exp(y·p)).
            gi = (zero - yi) / (one + F::exp(yi * pi));
        }
        if loss_id == 2u32 {
            // Squared hinge: z = 1 - p·y; z > 0 → -2·y·z.
            let z = one - pi * yi;
            if z > zero {
                gi = (zero - two) * yi * z;
            }
        }
        if loss_id == 3u32 {
            // Squared error: p - y.
            gi = pi - yi;
        }
        if loss_id == 4u32 {
            // Epsilon-insensitive: y-p > ε → -1; p-y > ε → 1; else 0.
            if yi - pi > epsilon {
                gi = zero - one;
            }
            if pi - yi > epsilon {
                gi = one;
            }
        }
        if loss_id == 5u32 {
            // Squared epsilon-insensitive: z = y-p; z > ε → -2(z-ε);
            // z < -ε → 2(-z-ε); else 0.
            let z = yi - pi;
            if z > epsilon {
                gi = (zero - two) * (z - epsilon);
            }
            if zero - z > epsilon {
                gi = two * ((zero - z) - epsilon);
            }
        }
        // Clip ±1e12 (the host dloss clamp, on device — statement-form ifs).
        let cap = F::new(1e12_f32);
        if gi > cap {
            gi = cap;
        }
        if gi < zero - cap {
            gi = zero - cap;
        }
        g[i] = gi;
    }
}

/// Pass 2 (per-coordinate weight update + fused lazy-L2 shrink):
/// `w[j] = (w[j] − eta·inv_b·Σ_i g[i]·x[(row_offset+i)*d + j]) · l2_factor`
/// over the `b` minibatch rows for coordinate `j` of the FULL row-major `n × d`
/// design `x`.
///
/// - `x` is the FULL row-major `n × d` design matrix; `row_offset` selects the
///   batch rows in place.
/// - `g` is the length-`b` per-sample gradient (device-resident, from
///   [`sgd_grad`]).
/// - `w` is the length-`d` weight vector, updated IN PLACE (single-owner write
///   per coordinate `j`).
/// - `eta` is the scheduled learning rate passed BY VALUE.
/// - `inv_b` is the `1/batch_size` averaging factor passed BY VALUE.
/// - `l2_factor` is the compounded lazy-L2 shrink
///   `max(0, 1 − (1−l1_ratio)·eta·alpha)^bsz` (host f64 arithmetic, lowered to
///   `F`), applied AFTER the gradient step — the order sklearn `_plain_sgd`
///   uses. Callers pass exactly `1.0` when no L2 applies (an exact
///   multiplicative identity, so the no-penalty path is unchanged).
/// - `row_offset`, `d`, `b` are scalar args passed BY VALUE.
///
/// One unit handles one coordinate at `ABSOLUTE_POS`; bounds-checked on `j < d`.
/// The minibatch gradient is a forward `while` GATHER seeded `F::from_int(0i64)`
/// — single-owner (each unit reduces its own column and writes its own `w[j]`),
/// no cross-unit reduction/scatter, no shared-memory tile, no `bool`, no
/// infinity sentinel (cubecl-cpu MLIR-safe).
#[cube(launch)]
pub fn sgd_weight_update<F: Float + CubeElement>(
    x: &Array<F>,
    g: &Array<F>,
    w: &mut Array<F>,
    eta: F,
    inv_b: F,
    l2_factor: F,
    row_offset: u32,
    d: u32,
    b: u32,
) {
    let j = ABSOLUTE_POS;
    if j < d as usize {
        let mut grad = F::from_int(0i64);
        let mut i = 0u32;
        // Forward GATHER over the minibatch rows (single-owner, no scatter).
        while i < b {
            let idx = (row_offset as usize + i as usize) * d as usize + j;
            grad += g[i as usize] * x[idx];
            i += 1u32;
        }
        w[j] = (w[j] - eta * inv_b * grad) * l2_factor;
    }
}

/// Cumulative-L1 soft-shrink (sklearn `l1penalty`), per coordinate: replays the
/// `b` per-sample budget steps (starting from the host-tracked `u_start`) with
/// the running per-coordinate correction `q[j]`, pulling `w[j]` toward zero by
/// the accumulated budget:
///
/// ```text
/// for s in 0..b:
///     u = u_start + (s+1)·du        // du = l1_ratio·eta·alpha per sample
///     z = w[j]
///     if w[j] > 0: w[j] = max(0, w[j] − (u + q[j]))
///     if w[j] < 0: w[j] = min(0, w[j] + (u − q[j]))
///     q[j] += w[j] − z
/// ```
///
/// This is the per-sample sequence the previous HOST loop ran (samples outer,
/// coordinates inner; `u` derived multiplicatively rather than repeat-added —
/// equal up to one rounding per sample, a cpu-MLIR constraint, see the body) —
/// for one coordinate the two orders are the same operation sequence, so
/// per-coordinate parallelism is sound (each unit owns its `w[j]`/`q[j]` pair,
/// single-owner). The host advances its `u` mirror by the same `b·du` after the
/// launch, so `u_start` stays in lock-step across batches. Statement-form
/// `if`s only; forward `while` (cpu-MLIR-safe, the `CUBE_POS_X` selecting-unit
/// launch shape — `CubeCount::Static(d,1,1)` / `CubeDim {x:1,y:1,z:1}`).
#[cube(launch)]
pub fn sgd_l1_shrink<F: Float + CubeElement>(
    w: &mut Array<F>,
    q: &mut Array<F>,
    u_start: F,
    du: F,
    d: u32,
    b: u32,
) {
    // One CUBE per coordinate, single selecting unit (the `top_k` shape —
    // spike finding 002-A: a bare-`ABSOLUTE_POS` launch over this loop-carried
    // multi-accumulator body fails the cpu-MLIR pass pipeline). Launched
    // `CubeCount::Static(d,1,1)` / `CubeDim {x:1,y:1,z:1}`.
    let j = CUBE_POS_X;
    if j < d {
        if UNIT_POS_X == 0u32 {
            let jj = j as usize;
            let zero = F::from_int(0i64);
            let mut wj = w[jj];
            let mut qj = q[jj];
            let mut s = 0u32;
            while s < b {
                // u is DERIVED from the sample counter (u_start + (s+1)·du),
                // not loop-carried: a third chained F accumulator (`u += du`
                // feeding `wj`) fails the cpu-MLIR pass pipeline (bisected in
                // plan 19; two coupled accumulators are the proven ceiling).
                // Equal to the repeated-add form up to one rounding per sample.
                let u = u_start + F::cast_from(s + 1u32) * du;
                let z = wj;
                if z > zero {
                    let mut cand = z - (u + qj);
                    if cand < zero {
                        cand = zero;
                    }
                    wj = cand;
                }
                if z < zero {
                    let mut cand = z + (u - qj);
                    if cand > zero {
                        cand = zero;
                    }
                    wj = cand;
                }
                qj += wj - z;
                s += 1u32;
            }
            w[jj] = wj;
            q[jj] = qj;
        }
    }
}

/// Intercept step (single-unit): `bias[0] -= eta·inv_b·Σ_i g[i]` over the
/// length-`b` batch gradient — the device-resident form of the host intercept
/// update (intercept_decay = 1.0 dense, A3).
///
/// Launched `CubeCount::Static(1,1,1)` / `CubeDim {x:1,y:1,z:1}` with the
/// `CUBE_POS_X`/`UNIT_POS_X == 0` selecting-unit shape (spike finding 002-A:
/// a bare-`ABSOLUTE_POS` per-row loop launch is a cpu-MLIR landmine). The sum
/// is a forward `while` in the same order the host f64 sum ran (bitwise-equal
/// on the f64 path).
#[cube(launch)]
pub fn sgd_bias_update<F: Float + CubeElement>(
    g: &Array<F>,
    bias: &mut Array<F>,
    eta: F,
    inv_b: F,
    b: u32,
) {
    let row = CUBE_POS_X;
    if row < 1u32 {
        if UNIT_POS_X == 0u32 {
            let mut s = F::from_int(0i64);
            let mut i = 0u32;
            while i < b {
                s += g[i as usize];
                i += 1u32;
            }
            bias[0] = bias[0] - eta * inv_b * s;
        }
    }
}

/// Per-sample sklearn loss VALUE (as opposed to [`sgd_grad`]'s derivative):
/// `loss[i] = loss(p[i], y[row_offset+i])`, selected by the by-value
/// `loss_id` scalar (same table index as [`sgd_grad`]):
///
/// | `loss_id` | loss | `loss(p, y)` |
/// |---|---|---|
/// | 0 | hinge | `max(0, 1 − p·y)` |
/// | 1 | log | `log(1 + exp(−p·y))`, clipped like sklearn's `Log.loss` |
/// | 2 | squared hinge | `max(0, 1 − p·y)²` |
/// | 3 | squared error | `½·(p − y)²` |
/// | 4 | ε-insensitive | `max(0, |y−p| − ε)` |
/// | 5 | squared ε-insensitive | `max(0, |y−p| − ε)²` |
///
/// `y` is the FULL length-`n` target; `row_offset` aligns it with the batch's
/// `p[]`. One unit per sample at `ABSOLUTE_POS` (the proven per-element map
/// shape); statement-form `if` chains only. The `log` branch computes the
/// UNCLIPPED `log1p(exp(−z))` value first and OVERWRITES it in the two extreme
/// regions (`|z| > 18`, sklearn's own clip point) — a transient overflow to
/// `+inf` in the discarded default is harmless IEEE arithmetic, and this stays
/// in statement-if form (no `if`-expression in value position, cpu-MLIR-safe).
#[cube(launch)]
pub fn sgd_loss<F: Float + CubeElement>(
    p: &Array<F>,
    y: &Array<F>,
    loss_out: &mut Array<F>,
    row_offset: u32,
    b: u32,
    loss_id: u32,
    epsilon: F,
) {
    let i = ABSOLUTE_POS;
    if i < b as usize {
        let pi = p[i];
        let yi = y[row_offset as usize + i];
        let zero = F::from_int(0i64);
        let one = F::from_int(1i64);
        let mut li = zero;
        if loss_id == 0u32 {
            // Hinge: max(0, 1 - p*y).
            let z = one - pi * yi;
            if z > zero {
                li = z;
            }
        }
        if loss_id == 1u32 {
            // Log: log(1 + exp(-p*y)), clipped at |z| = 18 like sklearn.
            let z = pi * yi;
            let eighteen = F::new(18.0f32);
            li = F::exp(zero - z).log1p();
            if z > eighteen {
                li = F::exp(zero - z);
            }
            if z < zero - eighteen {
                li = zero - z;
            }
        }
        if loss_id == 2u32 {
            // Squared hinge: max(0, 1 - p*y)^2.
            let z = one - pi * yi;
            if z > zero {
                li = z * z;
            }
        }
        if loss_id == 3u32 {
            // Squared error: 0.5*(p-y)^2.
            let d = pi - yi;
            li = F::new(0.5f32) * d * d;
        }
        if loss_id == 4u32 {
            // Epsilon-insensitive: max(0, |y-p|-eps).
            let z = (yi - pi).abs() - epsilon;
            if z > zero {
                li = z;
            }
        }
        if loss_id == 5u32 {
            // Squared epsilon-insensitive: max(0, |y-p|-eps)^2.
            let z = (yi - pi).abs() - epsilon;
            if z > zero {
                li = z * z;
            }
        }
        loss_out[i] = li;
    }
}

/// Loss-plateau fold (single-unit): folds this batch's `Σ_i loss[i]` into the
/// running epoch `sumloss[0]` accumulator (the host zeroes `sumloss` at epoch
/// start and reads it ONCE per epoch for the sklearn `tol`/`n_iter_no_change`
/// gate — MBSGD-PERF-WGPU).
///
/// Single selecting unit (`CUBE_POS_X`/`UNIT_POS_X == 0` shape, launched
/// `(1,1,1)/(1,1,1)`); the sum is a forward `while` seeded `F::from_int(0i64)`,
/// mirroring [`sgd_bias_update`]'s reduction exactly.
#[cube(launch)]
pub fn sgd_sumloss<F: Float + CubeElement>(loss: &Array<F>, sumloss: &mut Array<F>, b: u32) {
    let row = CUBE_POS_X;
    if row < 1u32 {
        if UNIT_POS_X == 0u32 {
            let mut s = F::from_int(0i64);
            let mut i = 0u32;
            while i < b {
                s += loss[i as usize];
                i += 1u32;
            }
            sumloss[0] = sumloss[0] + s;
        }
    }
}
