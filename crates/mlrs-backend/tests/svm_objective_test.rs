//! Linear-SVM primal objective evaluator (`prims::svm_objective::SvmObjective`)
//! validation — the SVM-FIT-CPU perf lever.
//!
//! `SvmObjective` is what `LinearSVC`/`LinearSVR`'s L-BFGS solve calls per
//! iteration and line-search step: it returns `Σᵢ ℓ(x̃ᵢ·w, tᵢ)` and the data
//! gradient `x̃ᵀ·g` over the synthetic-feature-augmented design. Its cpu arm
//! replaces the two `prims::gemm` launches (three orders of magnitude off the
//! machine's roofline there — see the prim's module docs) with ONE fused `-O3`
//! host pass that never materializes the augmented design.
//!
//! Because that arm is an INDEPENDENT implementation of the same maths rather
//! than a tuned kernel, every test here checks it against a DIRECT, deliberately
//! naive host reference that builds `x̃` explicitly and runs two separate
//! textbook loops. The cases cover both intercept modes, both dtypes, the
//! `d_aug`-boundary geometry, and the multi-worker split (the fan-out only
//! engages above `SVM_ELEMS_PER_UNIT`, so one case is sized to force it).
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::capability::{self, FloatKind};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::svm_objective::{SvmDesign, SvmEval, SvmObjective};
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{f64_to_host, PrimError};

/// Deterministic `[-1, 1)` stream (splitmix64), so a failure is reproducible
/// and the two dtypes see the SAME values.
fn uniform_pm1(seed: u64, n: usize) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

/// The squared-hinge margin loss `LinearSVC` fits with: `z = 1 − t·m`,
/// `ℓ = max(0, z)²`, `∂ℓ/∂m = −2·t·max(0, z)`. Its subgradient is ZERO for every
/// sample outside the margin, which is exactly the branch the fused pass skips.
fn squared_hinge(margin: f64, target: f64) -> (f64, f64) {
    let z = 1.0 - target * margin;
    if z > 0.0 {
        (z * z, -2.0 * target * z)
    } else {
        (0.0, 0.0)
    }
}

/// The squared-epsilon-insensitive loss `LinearSVR` fits with, at `ε = 0.1`.
/// Unlike squared hinge this is non-zero for MOST samples, so it exercises the
/// always-accumulate path.
fn squared_eps(margin: f64, target: f64) -> (f64, f64) {
    const EPS: f64 = 0.1;
    let r = target - margin;
    let viol = r.abs() - EPS;
    if viol > 0.0 {
        let s = if r >= 0.0 { 1.0 } else { -1.0 };
        (viol * viol, -2.0 * s * viol)
    } else {
        (0.0, 0.0)
    }
}

/// The DIRECT reference: materialize `x̃ = [x | intercept_scaling]`, then two
/// separate textbook loops (margins, then `x̃ᵀg`) — the shape `SvmObjective`'s
/// cpu arm fuses and its device arm hands to two GEMMs. Deliberately naive so it
/// shares no code with either.
///
/// `x` is the UNaugmented `n × d` design in f64.
fn objective_ref(
    x: &[f64],
    n: usize,
    d: usize,
    w: &[f64],
    targets: &[f64],
    intercept_scaling: f64,
    fit_intercept: bool,
    loss: impl Fn(f64, f64) -> (f64, f64),
) -> (f64, Vec<f64>) {
    let d_aug = if fit_intercept { d + 1 } else { d };
    let mut x_aug = vec![0.0f64; n * d_aug];
    for r in 0..n {
        x_aug[r * d_aug..r * d_aug + d].copy_from_slice(&x[r * d..(r + 1) * d]);
        if fit_intercept {
            x_aug[r * d_aug + d] = intercept_scaling;
        }
    }

    let mut margins = vec![0.0f64; n];
    for r in 0..n {
        let mut acc = 0.0f64;
        for j in 0..d_aug {
            acc += x_aug[r * d_aug + j] * w[j];
        }
        margins[r] = acc;
    }

    let mut data_loss = 0.0f64;
    let mut g = vec![0.0f64; n];
    for i in 0..n {
        let (li, gi) = loss(margins[i], targets[i]);
        data_loss += li;
        g[i] = gi;
    }

    let mut xtg = vec![0.0f64; d_aug];
    for (j, out) in xtg.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for i in 0..n {
            acc += x_aug[i * d_aug + j] * g[i];
        }
        *out = acc;
    }
    (data_loss, xtg)
}

/// Build the prim over an `n × d` design in `F` and evaluate it at `w`.
fn run_case<F>(
    x_f64: &[f64],
    n: usize,
    d: usize,
    w: &[f64],
    targets: &[f64],
    intercept_scaling: f64,
    fit_intercept: bool,
    loss: &(impl Fn(f64, f64) -> (f64, f64) + Sync + Copy),
) -> SvmEval
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_host: Vec<F> = x_f64.iter().map(|&v| f64_to_host::<F>(v)).collect();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);

    let obj = SvmObjective::<F>::new(
        &mut pool,
        SvmDesign::Device(&x_dev),
        (n, d),
        targets.to_vec(),
        intercept_scaling,
        fit_intercept,
    )
    .expect("SvmObjective::new accepts a valid geometry");
    assert_eq!(
        obj.d_aug(),
        if fit_intercept { d + 1 } else { d },
        "d_aug reflects the synthetic intercept column"
    );
    let ev = obj
        .eval(&mut pool, w, &loss)
        .expect("eval accepts a d_aug-length w");
    obj.release_into(&mut pool);
    ev
}

/// Assert `got` matches `want` to a RELATIVE band — the sums here run over
/// thousands of terms, so an absolute band would be meaningless at the scales
/// `data_loss` reaches.
fn assert_rel(got: f64, want: f64, band: f64, what: &str) {
    let denom = want.abs().max(1.0);
    let rel = (got - want).abs() / denom;
    assert!(
        rel <= band,
        "{what}: got={got:e} want={want:e} rel_err={rel:e} (band={band:e})"
    );
}

fn assert_rel_slice(got: &[f64], want: &[f64], band: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(want).enumerate() {
        assert_rel(g, e, band, &format!("{what}[{i}]"));
    }
}

/// One (dtype, loss, intercept-mode, geometry) case against the direct
/// reference. `band` accounts for the dtype of the DESIGN — the accumulators are
/// f64 on the cpu arm and `F` on the device arms, so f32 gets the wider band.
fn check_case<F>(
    n: usize,
    d: usize,
    seed: u64,
    fit_intercept: bool,
    intercept_scaling: f64,
    loss: &(impl Fn(f64, f64) -> (f64, f64) + Sync + Copy),
    targets: Vec<f64>,
    band: f64,
    what: &str,
) where
    F: Float + CubeElement + Pod,
{
    let x = uniform_pm1(seed, n * d);
    let d_aug = if fit_intercept { d + 1 } else { d };
    let w = uniform_pm1(seed + 7, d_aug);

    let (want_loss, want_xtg) = objective_ref(
        &x,
        n,
        d,
        &w,
        &targets,
        intercept_scaling,
        fit_intercept,
        loss,
    );
    let got = run_case::<F>(
        &x,
        n,
        d,
        &w,
        &targets,
        intercept_scaling,
        fit_intercept,
        loss,
    );

    assert_rel(got.data_loss, want_loss, band, &format!("{what} data_loss"));
    assert_rel_slice(&got.xtg, &want_xtg, band, &format!("{what} xtg"));
}

/// ±1 labels derived from the design's own first column, so roughly half the
/// samples land inside the squared-hinge margin (a non-zero subgradient) and
/// half outside (the skipped-axpy branch).
fn pm1_targets(seed: u64, n: usize) -> Vec<f64> {
    uniform_pm1(seed, n)
        .iter()
        .map(|&v| if v >= 0.0 { 1.0 } else { -1.0 })
        .collect()
}

/// Element counts. `MULTI_N` is sized so `n·d` clears `SVM_ELEMS_PER_UNIT`
/// (`1 << 16`) several times over and the cpu arm actually FANS OUT — the
/// single-threaded path would otherwise be the only one under test, and the
/// per-worker partial reduction (including the hoisted synthetic-column entry,
/// which is folded in at reduce time) would go unchecked.
const MULTI_N: usize = 20_000;
const MULTI_D: usize = 16;

#[test]
fn squared_hinge_with_intercept_f64() {
    capability::log_oracle_dtype(FloatKind::F64, capability::active_backend_name(), "default");
    if capability::skip_f64_with_log() {
        return;
    }
    check_case::<f64>(
        257,
        8,
        11,
        true,
        1.0,
        &squared_hinge,
        pm1_targets(99, 257),
        1e-12,
        "squared_hinge f64 +intercept",
    );
}

#[test]
fn squared_hinge_with_intercept_f32() {
    capability::log_oracle_dtype(FloatKind::F32, capability::active_backend_name(), "default");
    check_case::<f32>(
        257,
        8,
        11,
        true,
        1.0,
        &squared_hinge,
        pm1_targets(99, 257),
        1e-5,
        "squared_hinge f32 +intercept",
    );
}

/// `fit_intercept = false` — the design is NOT augmented, so `d_aug == d` and
/// the hoisted synthetic term must contribute nothing at all.
#[test]
fn squared_hinge_no_intercept_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_case::<f64>(
        129,
        5,
        23,
        false,
        1.0,
        &squared_hinge,
        pm1_targets(31, 129),
        1e-12,
        "squared_hinge f64 -intercept",
    );
}

/// A non-unit `intercept_scaling` (Pitfall 5): the synthetic column's value
/// enters BOTH the margin (`scaling·w[d]`) and its gradient entry
/// (`scaling·Σgᵢ`), so a case at `1.0` would not catch dropping either factor.
#[test]
fn intercept_scaling_is_applied_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_case::<f64>(
        193,
        6,
        5,
        true,
        2.75,
        &squared_hinge,
        pm1_targets(77, 193),
        1e-12,
        "squared_hinge f64 intercept_scaling=2.75",
    );
}

/// The regression loss, whose subgradient is non-zero for MOST samples — the
/// complement of squared hinge's mostly-skipped accumulate.
#[test]
fn squared_eps_insensitive_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let n = 311;
    check_case::<f64>(
        n,
        7,
        41,
        true,
        1.0,
        &squared_eps,
        uniform_pm1(1234, n),
        1e-12,
        "squared_eps f64",
    );
}

/// `d = 1`: the dot product is pure remainder (below one SIMD lane group), and
/// with an intercept `d_aug = 2`.
#[test]
fn single_feature_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_case::<f64>(
        64,
        1,
        3,
        true,
        1.0,
        &squared_hinge,
        pm1_targets(17, 64),
        1e-12,
        "squared_hinge f64 d=1",
    );
}

/// `d = 8` exactly — the lane group divides the row with an EMPTY remainder,
/// the other side of the `d = 1` case.
#[test]
fn lane_aligned_features_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_case::<f64>(
        96,
        8,
        13,
        false,
        1.0,
        &squared_eps,
        uniform_pm1(64, 96),
        1e-12,
        "squared_eps f64 d=8 aligned",
    );
}

/// Large enough to FAN OUT across worker threads (see [`MULTI_N`]): checks the
/// per-worker partial reduction, including that the hoisted synthetic-column
/// entry survives it.
#[test]
fn multi_worker_split_matches_reference_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_case::<f64>(
        MULTI_N,
        MULTI_D,
        1009,
        true,
        1.5,
        &squared_hinge,
        pm1_targets(2011, MULTI_N),
        1e-11,
        "squared_hinge f64 multi-worker",
    );
}

#[test]
fn multi_worker_split_matches_reference_f32() {
    check_case::<f32>(
        MULTI_N,
        MULTI_D,
        1009,
        true,
        1.5,
        &squared_eps,
        uniform_pm1(2011, MULTI_N),
        1e-4,
        "squared_eps f32 multi-worker",
    );
}

/// The evaluator is built ONCE per fit and evaluated 25-40 times — and on the
/// cpu arm those evaluations now run on a PERSISTENT worker pool, so every
/// dispatch after the first reuses threads that are parked on a barrier rather
/// than freshly spawned.
///
/// That reuse is exactly what the single-evaluation cases above cannot see: a
/// stale published task, a barrier epoch that fails to advance, or a partial
/// accumulator that is not reset would all produce a CORRECT first evaluation
/// and a wrong second one. This drives one objective through a sequence of
/// distinct `w` values — sized to fan out ([`MULTI_N`]) — and checks EVERY
/// evaluation against the direct reference, so a stale-dispatch bug fails here
/// instead of silently corrupting an L-BFGS line search.
///
/// The `w` sequence is deliberately varied in sign and scale so the fraction of
/// samples with a non-zero subgradient (the branch the fused pass skips) is
/// different on each pass.
#[test]
fn repeated_evaluations_reuse_the_pool_correctly() {
    let (n, d) = (MULTI_N, MULTI_D);
    let d_aug = d + 1;
    let x = uniform_pm1(4243, n * d);
    let targets = pm1_targets(4457, n);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_host: Vec<f32> = x.iter().map(|&v| f64_to_host::<f32>(v)).collect();
    let obj = SvmObjective::<f32>::new(
        &mut pool,
        SvmDesign::Host(&x_host),
        (n, d),
        targets.clone(),
        1.5,
        true,
    )
    .expect("valid geometry");

    for (pass, scale) in [0.0f64, 1.0, -0.25, 8.0, 1e-3, -3.0].iter().enumerate() {
        let w: Vec<f64> = uniform_pm1(97 + pass as u64, d_aug)
            .iter()
            .map(|v| v * scale)
            .collect();
        let (want_loss, want_xtg) =
            objective_ref(&x, n, d, &w, &targets, 1.5, true, squared_hinge);
        let got = obj
            .eval(&mut pool, &w, &(squared_hinge as fn(f64, f64) -> (f64, f64)))
            .expect("eval accepts a d_aug-length w");
        assert_rel(
            got.data_loss,
            want_loss,
            1e-4,
            &format!("pass {pass}: data_loss"),
        );
        assert_rel_slice(&got.xtg, &want_xtg, 1e-4, &format!("pass {pass}: xtg"));
    }
    obj.release_into(&mut pool);
}

/// The two design forms are the SAME operand: [`SvmDesign::Host`] exists only
/// to skip copies the caller has already paid for, so an evaluation over a
/// borrowed host slab must agree with one over the uploaded `DeviceArray`
/// EXACTLY — not merely within a band. A divergence here would mean the
/// no-upload ingress had changed the arithmetic rather than just the plumbing.
#[test]
fn host_and_device_designs_evaluate_identically() {
    let (n, d) = (MULTI_N, MULTI_D);
    let x = uniform_pm1(5051, n * d);
    let targets = pm1_targets(5077, n);
    let w = uniform_pm1(5099, d + 1);
    let loss = squared_hinge as fn(f64, f64) -> (f64, f64);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_host: Vec<f32> = x.iter().map(|&v| f64_to_host::<f32>(v)).collect();
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x_host);

    let from_dev = {
        let o = SvmObjective::<f32>::new(
            &mut pool,
            SvmDesign::Device(&x_dev),
            (n, d),
            targets.clone(),
            1.5,
            true,
        )
        .expect("valid geometry");
        let ev = o.eval(&mut pool, &w, &loss).expect("valid w");
        o.release_into(&mut pool);
        ev
    };
    let from_host = {
        let o = SvmObjective::<f32>::new(
            &mut pool,
            SvmDesign::Host(&x_host),
            (n, d),
            targets,
            1.5,
            true,
        )
        .expect("valid geometry");
        let ev = o.eval(&mut pool, &w, &loss).expect("valid w");
        o.release_into(&mut pool);
        ev
    };
    x_dev.release_into(&mut pool);

    assert_eq!(
        from_host.data_loss, from_dev.data_loss,
        "the host-slice ingress must not change the summed loss by even one ULP"
    );
    assert_eq!(
        from_host.xtg, from_dev.xtg,
        "the host-slice ingress must not change the gradient by even one ULP"
    );
}

// ---------------------------------------------------------------------------
// Geometry rejection (ASVS V5): a malformed shape is a typed error BEFORE any
// allocation or launch, never an out-of-bounds read.
// ---------------------------------------------------------------------------

fn new_with<F>(
    x_len: usize,
    n: usize,
    d: usize,
    targets: usize,
) -> Result<usize, PrimError>
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_host: Vec<F> = vec![f64_to_host::<F>(1.0); x_len.max(1)];
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
    // The evaluator borrows `x_dev`, so it cannot leave this frame — the
    // rejection cases only care about the ERROR, so a successful construction
    // is reduced to its `d_aug` and released here.
    let obj = SvmObjective::<F>::new(
        &mut pool,
        SvmDesign::Device(&x_dev),
        (n, d),
        vec![0.0; targets],
        1.0,
        true,
    )?;
    let d_aug = obj.d_aug();
    obj.release_into(&mut pool);
    Ok(d_aug)
}

#[test]
fn rejects_shape_mismatch() {
    // x.len() != n*d
    assert!(
        matches!(
            new_with::<f32>(40, 7, 8, 7),
            Err(PrimError::ShapeMismatch { operand: "x", .. })
        ),
        "a design whose length disagrees with (n, d) is rejected"
    );
    // zero dims
    assert!(
        matches!(
            new_with::<f32>(8, 0, 8, 0),
            Err(PrimError::ShapeMismatch { operand: "x", .. })
        ),
        "a zero-row design is rejected"
    );
    assert!(
        matches!(
            new_with::<f32>(8, 8, 0, 8),
            Err(PrimError::ShapeMismatch { operand: "x", .. })
        ),
        "a zero-feature design is rejected"
    );
    // targets length != n
    assert!(
        matches!(
            new_with::<f32>(64, 8, 8, 7),
            Err(PrimError::ShapeMismatch {
                operand: "targets",
                ..
            })
        ),
        "a target vector that is not one-per-sample is rejected"
    );
}

#[test]
fn rejects_wrong_w_length() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_host: Vec<f32> = vec![1.0; 64];
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x_host);
    let obj = SvmObjective::<f32>::new(
        &mut pool,
        SvmDesign::Device(&x_dev),
        (8, 8),
        vec![1.0; 8],
        1.0,
        true,
    )
        .expect("valid geometry");
    // d_aug is 9 (8 features + the synthetic column); 8 is the classic off-by-one.
    let err = obj.eval(&mut pool, &[0.0; 8], &(squared_hinge as fn(f64, f64) -> (f64, f64)));
    assert!(
        matches!(err, Err(PrimError::DimMismatch { dim: "d_aug", .. })),
        "an unaugmented-length w is rejected rather than read out of bounds"
    );
    obj.release_into(&mut pool);
}
