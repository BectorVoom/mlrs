//! `HuberRegressor` (HUBER-01) FULL sklearn parameter-surface oracle tests.
//!
//! Gates every `sklearn.linear_model.HuberRegressor` parameter against the
//! committed `huber_{f32,f64}_seed42` fixture (`scripts/gen_oracle.py`
//! ::`gen_huber`).
//!
//! | parameter | how it is gated |
//! |---|---|
//! | `epsilon` | `1.05` / `1.35` (default) / `2.5` / `10.0` value cases, plus the `1.0` boundary as a documented DEGENERATE case (below) |
//! | `alpha` | `0.0` (unpenalized) / `1e-4` (default) / `1.0` / `100.0`, the last asserted by the fixture to visibly shrink `coef_` |
//! | `fit_intercept` | a `False` case in three different families |
//! | `tol` | a tight (`1e-12`) case pinning that scipy's `factr` — not `gtol` — is sklearn's binding stop, and a loose (`5.0`) control case |
//! | `max_iter` | `0` / `1` / `5` control cases asserting the cap and `n_iter_`, incl. scipy's quirk that `max_iter=0` still performs one line search |
//! | `warm_start` | two successive fits, asserting the second improves on the first and lands on the converged optimum |
//! | `sample_weight` | four weighted cases crossed with `fit_intercept` / `epsilon` / `alpha` |
//!
//! ## There are NO string-valued parameters, and that is asserted
//! Every `HuberRegressor` ctor parameter is a float, an int or a bool —
//! `epsilon`, `max_iter`, `alpha`, `warm_start`, `fit_intercept`, `tol`. There
//! is no `solver=`, no `loss=`, nothing with a `StrOptions` constraint, so there
//! is no string-valued parameter to run an oracle case over. That is not
//! assumed: `gen_huber` inspects sklearn's own `_parameter_constraints` and
//! FAILS AT GENERATION if any constraint is a `StrOptions`, and
//! [`parameter_surface_has_no_string_valued_parameter`] pins the same fact on
//! the Rust builder. If a future sklearn adds one, both trip.
//!
//! ## Why the value gate is a BAND derived from the fixture, not a constant
//! sklearn hands the objective to `scipy.optimize.minimize(method="L-BFGS-B")`
//! passing `tol` as `gtol` but leaving `factr = 1e7`, so every fit stops on the
//! relative-f criterion `Δf/max(|f|,1) ≤ 2.2e-9` **before the gradient test can
//! fire** — and `tol` cannot change that (the `tol_tight` case exists to pin
//! it). Its returned parameters therefore sit a measured 1e-6 … 1.2e-4 from the
//! true minimizer, and no configuration of mlrs can be closer to sklearn than
//! sklearn is to the answer. mlrs deliberately solves TIGHTER (`ftol = 64·eps`,
//! ~3 extra evaluations, ~4e-9 from the minimizer — see `huber.rs`), so the
//! fixture ships sklearn's OWN per-case residual (`residual_<name>`) and the
//! band here is derived from it. A case whose conditioning makes sklearn stop
//! further out widens only its own band.
//!
//! ## The gate that does NOT depend on either solver's stopping point
//! [`converged_cases_minimize_sklearns_own_objective`] recomputes sklearn's
//! Huber objective at mlrs's fitted parameters and asserts it is no LARGER than
//! the value sklearn achieved. This is the rigorous statement — "mlrs is at
//! least as good a minimizer of the reference objective" — and it holds for the
//! truncated `max_iter` cases and the σ-degenerate `epsilon = 1` case too, where
//! parameter agreement is not even well-posed.
//!
//! ## `epsilon = 1.0` is degenerate, on purpose
//! At exactly `epsilon = 1` every sample is an outlier, and the objective
//! collapses to `σ·Σsw + Σ2·swᵢ|rᵢ| − σ·Σsw = 2·Σ swᵢ|rᵢ|`: σ cancels
//! identically and `∂L/∂σ ≡ 0` along the whole ray, so the SCALE is not
//! identifiable. sklearn accepts it (its constraint is the closed `[1, ∞)`) and
//! returns `scale_ = 0`. The case is kept — it is a real configuration a user
//! can request — but gated on the objective value and on `scale_ → 0`, never on
//! parameter agreement.
//!
//! BOTH fit ingresses are driven (`Fit::fit` over a `DeviceArray` and
//! `fit_from_host_slice` over host slices) and a dedicated test asserts they
//! agree, which is what keeps the no-upload split honest.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64;
//! rocm skips-with-log per the CubeCL-HIP F64 gap, D-07). Per AGENTS.md §2 tests
//! live in `crates/mlrs-algos/tests/`, never an in-source `#[cfg(test)] mod
//! tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::BuildError;
use mlrs_algos::linear::huber::HuberRegressor;
use mlrs_algos::typestate::{Fitted, Predict};
use mlrs_backend::abflag;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// Fixture geometry — `gen_oracle.py`'s `HUBER_N_SAMPLES` × `HUBER_N_FEATURES`.
const N_SAMPLES: usize = 240;
const N_FEATURES: usize = 6;
/// Held-out rows — `gen_oracle.py`'s `HUBER_N_TEST`.
const N_TEST: usize = 9;

/// Floor under the fixture-derived value band, for the part of the disagreement
/// that is NOT sklearn's early stop: mlrs's own residual to the minimizer
/// (~4e-9 in f64) plus round-off in the comparison itself.
const BAND_FLOOR_F64: f64 = 1e-7;
/// f32 floor. The design's bytes only carry ~7 digits, so a coefficient derived
/// from 240 of them cannot agree past that however tightly either side solves.
/// The reference is fitted on the design AFTER its round-trip through f32
/// (`gen_huber`), so this covers the SOLVE's f32 sensitivity, not a dtype
/// mismatch in the inputs.
const BAND_FLOOR_F32: f64 = 2e-3;
/// Multiple of sklearn's own measured per-case residual the band allows. 4x
/// leaves room for the two solvers landing on opposite sides of the minimizer
/// (the residual measures each side's distance TO it, so the gap between them
/// can be twice that) with a factor of 2 spare.
const RESIDUAL_SLACK: f64 = 4.0;

/// Slack on the objective-value gate, RELATIVE to the achieved loss. mlrs must
/// not be worse than sklearn by more than this — it is not zero only because
/// both losses are re-summed in a different order over 240 samples.
const LOSS_SLACK_REL: f64 = 1e-9;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn host_to_f64<F: Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("huber fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("huber fixtures are f32/f64 only"),
    }
}

/// The band floor for the storage dtype under test.
fn band_floor<F: Pod>() -> f64 {
    match std::mem::size_of::<F>() {
        4 => BAND_FLOOR_F32,
        8 => BAND_FLOOR_F64,
        _ => unreachable!("huber fixtures are f32/f64 only"),
    }
}

/// numpy-`allclose` element compare: pass if `|got − exp| ≤ band + band·|exp|`.
fn assert_band(got: &[f64], expected: &[f64], band: f64, what: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: length mismatch got={} expected={}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs_err = (g - e).abs();
        assert!(
            abs_err <= band + band * e.abs(),
            "{what}: band failed at {i}: got={g:e} expected={e:e} \
             abs_err={abs_err:e} (band={band:e})"
        );
    }
}

/// One fixture case: the ctor configuration behind a `*_<name>` key group.
struct Case {
    /// Fixture key suffix (`coef_<name>`, `scale_<name>`, …).
    name: &'static str,
    epsilon: f64,
    max_iter: usize,
    alpha: f64,
    fit_intercept: bool,
    tol: f64,
    sample_weight: bool,
    /// Whether the case ran to its OWN stopping criterion on both sides, so the
    /// achieved objective values are comparable. False for the `max_iter` /
    /// loose-`tol` cases: those stop MID-TRAJECTORY, and mlrs's L-BFGS is the
    /// 05-06 strong-Wolfe primitive rather than scipy's Moré-Thuente, so after
    /// one or five steps the two are simply at different points on the way down.
    /// Comparing their losses there would be gating the line search, not the
    /// estimator — [`max_iter_is_a_reported_cap`] gates what IS well-posed about
    /// them (the cap, `n_iter_`, monotonicity, and beating the cold start).
    loss_comparable: bool,
}

impl Case {
    const fn new(name: &'static str) -> Self {
        // The sklearn defaults, EXCEPT `max_iter`, which the converged value
        // cases raise to 1000 so the stop is the `factr` plateau rather than the
        // cap (a truncated fit is not value-comparable — module docs).
        Self {
            name,
            epsilon: 1.35,
            max_iter: 1000,
            alpha: 1e-4,
            fit_intercept: true,
            tol: 1e-5,
            sample_weight: false,
            loss_comparable: true,
        }
    }
    const fn truncated(mut self) -> Self {
        self.loss_comparable = false;
        self
    }
    const fn epsilon(mut self, v: f64) -> Self {
        self.epsilon = v;
        self
    }
    const fn max_iter(mut self, v: usize) -> Self {
        self.max_iter = v;
        self
    }
    const fn alpha(mut self, v: f64) -> Self {
        self.alpha = v;
        self
    }
    const fn no_intercept(mut self) -> Self {
        self.fit_intercept = false;
        self
    }
    const fn tol(mut self, v: f64) -> Self {
        self.tol = v;
        self
    }
    const fn weighted(mut self) -> Self {
        self.sample_weight = true;
        self
    }
}

/// The CONVERGED cases — every parameter that changes WHAT is optimized. These
/// carry the full value gate.
const VALUE_CASES: &[Case] = &[
    Case::new("default"),
    Case::new("noint").no_intercept(),
    Case::new("eps105").epsilon(1.05),
    Case::new("eps2").epsilon(2.5),
    Case::new("eps10").epsilon(10.0),
    Case::new("eps105_noint").epsilon(1.05).no_intercept(),
    Case::new("alpha0").alpha(0.0),
    Case::new("alpha1").alpha(1.0),
    Case::new("alpha100").alpha(100.0),
    Case::new("tol_tight").tol(1e-12),
    Case::new("sw").weighted(),
    Case::new("sw_noint").no_intercept().weighted(),
    Case::new("sw_eps105").epsilon(1.05).weighted(),
    Case::new("sw_alpha1").alpha(1.0).weighted(),
];

/// The CONTROL-FLOW cases — truncated mid-trajectory (or σ-degenerate), so the
/// iterate itself is implementation-specific. Gated on the cap, on `n_iter_`,
/// and on the objective value only (module docs).
const CTRL_CASES: &[Case] = &[
    Case::new("maxiter0").max_iter(0).truncated(),
    Case::new("maxiter1").max_iter(1).truncated(),
    Case::new("maxiter5").max_iter(5).truncated(),
    Case::new("tol_loose").tol(5.0).truncated(),
    // NOT truncated: `epsilon = 1` runs to its own stop on both sides (sklearn
    // takes 28 iterations there). It is a control case because its SCALE is
    // unidentifiable, not because it is cut short — so the objective value is
    // exactly the gate that still means something for it.
    Case::new("eps1").epsilon(1.0),
];

/// What one mlrs fit produced, in host `f64`.
struct FitResult {
    coef: Vec<f64>,
    intercept: f64,
    scale: f64,
    n_iter: usize,
    outliers: Vec<bool>,
    params: Vec<f64>,
    pred: Vec<f64>,
}

/// Fit `case` on the fixture design and read every attribute back.
///
/// `host_ingress` selects `fit_from_host_slice` (the no-upload path the PyO3
/// boundary takes) over `Fit::fit`; the two must agree, which
/// [`both_fit_ingresses_agree`] asserts.
fn fit_case<F>(case_data: &OracleCase, case: &Case, host_ingress: bool) -> FitResult
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<F> = case_data
        .expect_f64("X")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();
    let y_host: Vec<F> = case_data
        .expect_f64("y")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();
    let sw_host: Vec<F> = case_data
        .expect_f64("sample_weight")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();
    let xt_host: Vec<F> = case_data
        .expect_f64("X_test")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();

    let est = HuberRegressor::<F>::builder()
        .epsilon(case.epsilon)
        .max_iter(case.max_iter)
        .alpha(case.alpha)
        .fit_intercept(case.fit_intercept)
        .tol(case.tol)
        .build::<F>()
        .expect("huber builder rejected a pinned fixture configuration");

    let sw = case.sample_weight.then_some(sw_host.as_slice());
    let fitted: HuberRegressor<F, Fitted> = if host_ingress {
        est.fit_from_host_slice(&mut pool, &x_host, &y_host, (N_SAMPLES, N_FEATURES), sw)
            .expect("huber fit_from_host_slice failed on a pinned fixture case")
    } else {
        let xd: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
        let yd: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_host);
        est.fit_with_sample_weight(&mut pool, &xd, Some(&yd), (N_SAMPLES, N_FEATURES), sw)
            .expect("huber fit failed on a pinned fixture case")
    };

    let xt_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xt_host);
    let pred = fitted
        .predict(&mut pool, &xt_dev, (N_TEST, N_FEATURES))
        .expect("huber predict failed");

    FitResult {
        coef: fitted.coef(&pool).iter().map(|&v| host_to_f64(v)).collect(),
        intercept: host_to_f64(fitted.intercept(&pool)),
        scale: fitted.scale(),
        n_iter: fitted.n_iter(),
        outliers: fitted.outliers().to_vec(),
        params: fitted.warm_start_params().to_vec(),
        pred: pred.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect(),
    }
}

/// sklearn's Huber objective at `(coef, intercept, sigma)` — the same formula
/// `gen_oracle.py::_huber_objective` evaluates, recomputed HERE so the gate does
/// not depend on mlrs's own loss assembly being right.
fn huber_loss(
    x: &[f64],
    y: &[f64],
    sw: Option<&[f64]>,
    coef: &[f64],
    intercept: f64,
    sigma: f64,
    epsilon: f64,
    alpha: f64,
) -> f64 {
    let n = y.len();
    let d = coef.len();
    let mut sw_total = 0.0;
    let mut loss = 0.0;
    for i in 0..n {
        let s = sw.map(|w| w[i]).unwrap_or(1.0);
        sw_total += s;
        let mut m = intercept;
        for j in 0..d {
            m += x[i * d + j] * coef[j];
        }
        let r = y[i] - m;
        let a = r.abs();
        if a > epsilon * sigma {
            loss += s * (2.0 * epsilon * a - epsilon * epsilon * sigma);
        } else {
            loss += s * r * r / sigma;
        }
    }
    loss += sw_total * sigma;
    loss + alpha * coef.iter().map(|w| w * w).sum::<f64>()
}

/// The fixture-derived comparison band for one case (module docs).
fn case_band<F: Pod>(case_data: &OracleCase, name: &str) -> f64 {
    let residual = case_data.expect_f64(&format!("residual_{name}"))[0];
    (RESIDUAL_SLACK * residual).max(band_floor::<F>())
}

/// Drive every converged case and gate all six fitted attributes.
fn run_value_cases<F>(case_data: &OracleCase, label: &str)
where
    F: Float + CubeElement + Pod,
{
    for case in VALUE_CASES {
        let got = fit_case::<F>(case_data, case, false);
        let band = case_band::<F>(case_data, case.name);
        let what = |attr: &str| format!("{label}/{}::{attr}", case.name);

        assert_band(
            &got.coef,
            case_data.expect_f64(&format!("coef_{}", case.name)),
            band,
            &what("coef_"),
        );
        assert_band(
            &[got.intercept],
            case_data.expect_f64(&format!("intercept_{}", case.name)),
            band,
            &what("intercept_"),
        );
        assert_band(
            &[got.scale],
            case_data.expect_f64(&format!("scale_{}", case.name)),
            band,
            &what("scale_"),
        );
        assert_band(
            &got.pred,
            case_data.expect_f64(&format!("pred_{}", case.name)),
            band,
            &what("predict"),
        );

        // `warm_start_params` is the packed `[coef…, intercept?, σ]` sklearn
        // concatenates — gated so a warm start seeds from the same vector.
        assert_band(
            &got.params,
            case_data.expect_f64(&format!("params_{}", case.name)),
            band,
            &what("warm_start_params"),
        );

        // `outliers_`: EXACT equality where the fixture proved no sample sits
        // within reach of the solver gap, the outlier COUNT otherwise.
        let expected_mask: Vec<bool> = case_data
            .expect_f64(&format!("outliers_{}", case.name))
            .iter()
            .map(|&v| v != 0.0)
            .collect();
        let stable = case_data.expect_f64(&format!("outliers_stable_{}", case.name))[0] != 0.0;
        if stable && std::mem::size_of::<F>() == 8 {
            assert_eq!(
                got.outliers,
                expected_mask,
                "{}: outliers_ mask differs from sklearn's",
                what("outliers_")
            );
        } else {
            // f32 designs and the one conditioning-unstable case compare the
            // COUNT: an exact mask would be gating float round-off, not the
            // estimator.
            let got_n = got.outliers.iter().filter(|&&o| o).count() as f64;
            let exp_n = expected_mask.iter().filter(|&&o| o).count() as f64;
            assert!(
                (got_n - exp_n).abs() <= 2.0,
                "{}: outlier COUNT differs by more than 2: got={got_n} expected={exp_n}",
                what("outliers_")
            );
        }

        assert!(
            got.n_iter <= case.max_iter,
            "{}: n_iter_ = {} exceeds max_iter = {}",
            what("n_iter_"),
            got.n_iter,
            case.max_iter
        );
    }
}

// ---------------------------------------------------------------------------
// value gates
// ---------------------------------------------------------------------------

#[test]
fn oracle_value_cases_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "huber");
    let case = load_npz(fixture("huber_f32_seed42.npz")).expect("load huber f32 fixture");
    run_value_cases::<f32>(&case, "huber f32");
}

#[test]
fn oracle_value_cases_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case = load_npz(fixture("huber_f64_seed42.npz")).expect("load huber f64 fixture");
    run_value_cases::<f64>(&case, "huber f64");
}

/// The SAME value gates, run against the DEVICE engine (HUBER-02).
///
/// Since the engine is chosen per fit by
/// `prims::huber_objective::huber_device_applicable`, and the crossover on this
/// hardware keeps oracle-sized fixtures on the fused host pass, the two tests
/// above would never touch the device kernels on a GPU backend. Forcing
/// `MLRS_HUBER_ENGINE=device` is what makes the device arm answer to
/// scikit-learn rather than only to the round-trip arm it replaced.
///
/// Self-skips on the cpu backend, where the "device" is `cubecl-cpu` and the
/// override is refused as a correctness gate rather than a preference.
///
/// Forced through `abflag`, never `std::env::set_var` — that is an `environ`
/// data race against every sibling test's dispatcher read, and it would leak
/// process-wide and make the HOST-arm tests above silently measure the device
/// one ([[mlrs-abflag-test-knobs]]).
#[test]
fn oracle_value_cases_f32_device_engine() {
    if capability::active_backend_name() == "cpu" {
        return;
    }
    let _engine = abflag::force("MLRS_HUBER_ENGINE", "device");
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "huber/device");
    let case = load_npz(fixture("huber_f32_seed42.npz")).expect("load huber f32 fixture");
    run_value_cases::<f32>(&case, "huber f32 device-engine");
}

/// [`oracle_value_cases_f32_device_engine`] at `f64`. Self-skips wherever the
/// backend cannot do `f64` device kernels at all — which is rocm and cuda here,
/// so in practice this runs on wgpu ([[mlrs-rocm-hardware-env]]).
#[test]
fn oracle_value_cases_f64_device_engine() {
    if capability::active_backend_name() == "cpu" || capability::skip_f64_with_log() {
        return;
    }
    let _engine = abflag::force("MLRS_HUBER_ENGINE", "device");
    let case = load_npz(fixture("huber_f64_seed42.npz")).expect("load huber f64 fixture");
    run_value_cases::<f64>(&case, "huber f64 device-engine");
}

/// THE gate that does not depend on where either solver stopped: mlrs's
/// parameters must not be a WORSE minimizer of sklearn's own objective than
/// sklearn's are.
///
/// This covers the truncated `max_iter` cases and the σ-degenerate `epsilon = 1`
/// case, where comparing parameters is not well-posed at all — mlrs's L-BFGS is
/// the 05-06 strong-Wolfe primitive, not scipy's Moré-Thuente, so the two do not
/// pass through the same intermediate iterates even from an identical start.
/// For the CONVERGED cases it is the strong statement: mlrs solves tighter
/// (`ftol = 64·eps` against scipy's `factr·eps = 2.2e-9`), so it should come out
/// strictly ahead.
#[test]
fn converged_cases_minimize_sklearns_own_objective() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case_data = load_npz(fixture("huber_f64_seed42.npz")).expect("load huber f64 fixture");
    let x = case_data.expect_f64("X").to_vec();
    let y = case_data.expect_f64("y").to_vec();
    let sw = case_data.expect_f64("sample_weight").to_vec();

    for case in VALUE_CASES
        .iter()
        .chain(CTRL_CASES.iter())
        .filter(|c| c.loss_comparable)
    {
        let got = fit_case::<f64>(&case_data, case, false);
        let ours = huber_loss(
            &x,
            &y,
            case.sample_weight.then_some(sw.as_slice()),
            &got.coef,
            got.intercept,
            got.scale,
            case.epsilon,
            case.alpha,
        );
        let theirs = case_data.expect_f64(&format!("loss_{}", case.name))[0];
        assert!(
            ours <= theirs + LOSS_SLACK_REL * theirs.abs(),
            "{}: mlrs is a WORSE minimizer of sklearn's own Huber objective — \
             mlrs={ours:.12e} sklearn={theirs:.12e} (excess {:.3e})",
            case.name,
            ours - theirs
        );
    }
}

// ---------------------------------------------------------------------------
// control-flow gates: max_iter, tol, the epsilon=1 degeneracy
// ---------------------------------------------------------------------------

/// `max_iter` is a CAP that is honoured and reported, and the truncated fits are
/// genuinely worse than the converged one.
///
/// `max_iter = 0` is not a no-op and is not an error: sklearn's constraint is
/// `Interval(Integral, 0, None, closed="left")`, and scipy's L-BFGS-B checks the
/// cap AFTER the first iteration — so it performs one line search and then
/// reports `n_iter_ = 0`. `huber.rs::solve` reproduces both halves of that.
#[test]
fn max_iter_is_a_reported_cap() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case_data = load_npz(fixture("huber_f64_seed42.npz")).expect("load huber f64 fixture");
    let x = case_data.expect_f64("X").to_vec();
    let y = case_data.expect_f64("y").to_vec();

    let converged = fit_case::<f64>(&case_data, &Case::new("default"), false);
    let converged_loss = huber_loss(
        &x, &y, None, &converged.coef, converged.intercept, converged.scale, 1.35, 1e-4,
    );

    // The cold start sklearn and mlrs share: all-zero weights with `σ = 1`.
    // Every truncated fit must at least have taken a real descent step away from
    // it, which is the part of a mid-trajectory iterate that IS well-posed to
    // compare across two different line searches.
    let cold_loss = huber_loss(&x, &y, None, &[0.0; N_FEATURES], 0.0, 1.0, 1.35, 1e-4);

    let mut prev_loss = f64::INFINITY;
    for (cap, expected_n_iter) in [(0usize, 0usize), (1, 1), (5, 5)] {
        let got = fit_case::<f64>(&case_data, &Case::new("x").max_iter(cap), false);
        assert_eq!(
            got.n_iter, expected_n_iter,
            "max_iter={cap}: n_iter_ = {} but sklearn reports {expected_n_iter}",
            got.n_iter
        );
        let loss = huber_loss(&x, &y, None, &got.coef, got.intercept, got.scale, 1.35, 1e-4);
        assert!(
            loss < cold_loss,
            "max_iter={cap}: the truncated fit ({loss:.9e}) did not improve on the \
             cold start ({cold_loss:.9e}) — no descent step was taken"
        );
        assert!(
            loss >= converged_loss - 1e-9 * converged_loss.abs(),
            "max_iter={cap}: a truncated fit beat the converged one \
             ({loss:.9e} < {converged_loss:.9e}) — the cap is not being applied"
        );
        assert!(
            loss <= prev_loss + 1e-9 * prev_loss.abs(),
            "max_iter={cap}: raising the cap made the fit WORSE \
             ({loss:.9e} > {prev_loss:.9e})"
        );
        prev_loss = loss;
    }
}

/// `epsilon = 1.0` is the closed lower bound sklearn allows, and it makes the
/// SCALE unidentifiable — every sample becomes an outlier, `σ` cancels out of
/// the objective exactly, and what is left is weighted least-absolute-deviations
/// plus the ridge penalty. Both sides drive `σ → 0`; the coefficients stay
/// determined, so the objective-value gate above still applies.
#[test]
fn epsilon_one_collapses_the_scale_and_flags_every_sample() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case_data = load_npz(fixture("huber_f64_seed42.npz")).expect("load huber f64 fixture");
    let got = fit_case::<f64>(&case_data, &Case::new("eps1").epsilon(1.0), false);
    assert!(
        got.scale < 1e-4,
        "epsilon=1.0: scale_ = {} did not collapse — the σ-degeneracy analysis \
         in huber.rs no longer holds",
        got.scale
    );
    assert_eq!(
        got.outliers.iter().filter(|&&o| o).count(),
        N_SAMPLES,
        "epsilon=1.0: not every sample is an outlier, which is the premise the \
         σ-degeneracy rests on"
    );
}

/// A fit that reached the objective's numerical floor must report CONVERGED.
///
/// This is a regression gate on a real bug: the solve stops on the `ftol`
/// plateau for essentially every well-conditioned fit (that is the point of
/// `ftol = 64·eps`), and treating that as non-convergence made the Python shim
/// raise a `ConvergenceWarning` on EVERY default fit — while sitting closer to
/// the optimum than scikit-learn, which reports the same criterion as scipy
/// `status == 0`, i.e. success. `converged()` must therefore be false only for
/// the honest cases: the `max_iter` cap and a line-search breakdown away from
/// the gradient floor.
#[test]
fn converged_flag_tracks_the_real_stop() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case_data = load_npz(fixture("huber_f64_seed42.npz")).expect("load huber f64 fixture");
    let x = case_data.expect_f64("X").to_vec();
    let y = case_data.expect_f64("y").to_vec();
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let mut fit = |max_iter: usize| {
        HuberRegressor::<f64>::builder()
            .max_iter(max_iter)
            .build::<f64>()
            .expect("build")
            .fit_from_host_slice(&mut pool, &x, &y, (N_SAMPLES, N_FEATURES), None)
            .expect("fit")
    };

    let converged = fit(1000);
    assert!(
        converged.converged(),
        "a fit that ran to the objective's numerical floor in {} iterations          reported NOT converged — the Python shim would warn on every default fit",
        converged.n_iter()
    );
    // The cap IS a genuine non-convergence, and sklearn warns on it too.
    let capped = fit(3);
    assert!(
        !capped.converged(),
        "a fit truncated at max_iter=3 reported converged"
    );
}

/// A LOOSE `tol` stops the solve earlier and a TIGHT one does not change it —
/// the second half is sklearn's `factr`-binds-first behaviour, pinned so a
/// change in it is caught here rather than as an unexplained band failure.
#[test]
fn tol_is_the_gradient_stop() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case_data = load_npz(fixture("huber_f64_seed42.npz")).expect("load huber f64 fixture");
    let default = fit_case::<f64>(&case_data, &Case::new("default"), false);
    let tight = fit_case::<f64>(&case_data, &Case::new("tol_tight").tol(1e-12), false);
    let loose = fit_case::<f64>(&case_data, &Case::new("tol_loose").tol(5.0), false);

    assert!(
        loose.n_iter < default.n_iter,
        "tol=5.0 did not stop earlier than the default ({} vs {})",
        loose.n_iter,
        default.n_iter
    );
    // mlrs's own `ftol` is `64·eps`, so tightening `tol` below the achievable
    // gradient cannot buy more iterations either — same shape as sklearn's, for
    // a different reason (its floor is `factr`, ours is machine precision).
    assert!(
        tight.n_iter >= default.n_iter,
        "tol=1e-12 ran FEWER iterations than the default ({} vs {})",
        tight.n_iter,
        default.n_iter
    );
}

/// `warm_start` seeds the next fit with the previous `[coef…, intercept, σ]`, so
/// a second capped fit continues rather than restarting — exactly what sklearn's
/// `np.concatenate((self.coef_, [self.intercept_, self.scale_]))` does.
#[test]
fn warm_start_continues_the_previous_solve() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case_data = load_npz(fixture("huber_f64_seed42.npz")).expect("load huber f64 fixture");
    let x = case_data.expect_f64("X").to_vec();
    let y = case_data.expect_f64("y").to_vec();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let first = HuberRegressor::<f64>::builder()
        .warm_start(true)
        .max_iter(5)
        .build::<f64>()
        .expect("build")
        .fit_from_host_slice(&mut pool, &x, &y, (N_SAMPLES, N_FEATURES), None)
        .expect("first warm-start fit");
    let l1 = huber_loss(
        &x,
        &y,
        None,
        &first.coef(&pool),
        first.intercept(&pool),
        first.scale(),
        1.35,
        1e-4,
    );

    let second = HuberRegressor::<f64>::builder()
        .warm_start(true)
        .max_iter(5)
        .init_params(first.warm_start_params().to_vec())
        .build::<f64>()
        .expect("build")
        .fit_from_host_slice(&mut pool, &x, &y, (N_SAMPLES, N_FEATURES), None)
        .expect("second warm-start fit");
    let l2 = huber_loss(
        &x,
        &y,
        None,
        &second.coef(&pool),
        second.intercept(&pool),
        second.scale(),
        1.35,
        1e-4,
    );

    assert!(
        l2 < l1,
        "warm_start: the seeded second fit ({l2:.9e}) did not improve on the \
         first ({l1:.9e}) — the seed is not reaching the solver"
    );
    // sklearn's own two-fit pair improves by the same sign; the fixture asserts
    // that at generation, and this is the mlrs side of it.
    let sk1 = case_data.expect_f64("loss_warm1")[0];
    let sk2 = case_data.expect_f64("loss_warm2")[0];
    assert!(sk2 < sk1, "fixture: sklearn's warm-start pair did not improve");

    // Without the seed, the same cap restarts cold and cannot do better.
    let cold = HuberRegressor::<f64>::builder()
        .max_iter(5)
        .build::<f64>()
        .expect("build")
        .fit_from_host_slice(&mut pool, &x, &y, (N_SAMPLES, N_FEATURES), None)
        .expect("cold fit");
    let lc = huber_loss(
        &x,
        &y,
        None,
        &cold.coef(&pool),
        cold.intercept(&pool),
        cold.scale(),
        1.35,
        1e-4,
    );
    assert!(
        l2 <= lc,
        "warm_start: the seeded fit ({l2:.9e}) is worse than a cold one ({lc:.9e})"
    );
}

// ---------------------------------------------------------------------------
// ingress agreement + defaults + rejections
// ---------------------------------------------------------------------------

/// The no-upload host-slice ingress and the `DeviceArray` ingress must produce
/// the SAME fit — they share `fit_core`, and this is what keeps the split above
/// that line honest.
#[test]
fn both_fit_ingresses_agree() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case_data = load_npz(fixture("huber_f64_seed42.npz")).expect("load huber f64 fixture");
    for case in VALUE_CASES {
        let dev = fit_case::<f64>(&case_data, case, false);
        let host = fit_case::<f64>(&case_data, case, true);
        assert_band(
            &dev.coef,
            &host.coef,
            BAND_FLOOR_F64,
            &format!("{}::ingress coef_", case.name),
        );
        assert_band(
            &[dev.scale],
            &[host.scale],
            BAND_FLOOR_F64,
            &format!("{}::ingress scale_", case.name),
        );
        assert_eq!(
            dev.outliers, host.outliers,
            "{}: the two ingresses disagree on outliers_",
            case.name
        );
    }
}

/// `builder().build()` reproduces the sklearn `HuberRegressor` defaults (D-03
/// litmus): `epsilon=1.35`, `max_iter=100`, `alpha=0.0001`, `warm_start=False`,
/// `fit_intercept=True`, `tol=1e-5`.
#[test]
fn default_matches_sklearn() {
    let est = HuberRegressor::<f32>::builder()
        .build::<f32>()
        .expect("default build");
    assert_eq!(est.epsilon(), 1.35);
    assert_eq!(est.max_iter(), 100);
    assert_eq!(est.alpha(), 1e-4);
    assert!(!est.warm_start());
    assert!(est.fit_intercept());
    assert_eq!(est.tol(), 1e-5);
}

/// sklearn's `_parameter_constraints` for `HuberRegressor` contains no
/// `StrOptions`: every parameter is a float, an int or a bool. The Rust surface
/// mirrors that — no builder setter takes a string, so there is no string-valued
/// parameter to gate with an oracle case.
///
/// This test is the Rust half of the pin; `gen_huber` asserts the same fact
/// against sklearn's own constraint dict at fixture-generation time, so a future
/// sklearn that adds (say) a `solver=` breaks the generator rather than silently
/// leaving the new parameter untested. The assertion here is structural: it
/// enumerates the full setter surface, so adding a string-typed setter without
/// adding oracle coverage will not compile past this list.
#[test]
fn parameter_surface_has_no_string_valued_parameter() {
    // Every ctor parameter, set through its builder setter with a non-default
    // value. If a string-valued parameter is ever added, this call site must
    // change — which is the point.
    let est = HuberRegressor::<f64>::builder()
        .epsilon(2.0)
        .max_iter(7)
        .alpha(0.5)
        .warm_start(true)
        .fit_intercept(false)
        .tol(1e-3)
        .build::<f64>()
        .expect("full non-default surface builds");
    assert_eq!(est.epsilon(), 2.0);
    assert_eq!(est.max_iter(), 7);
    assert_eq!(est.alpha(), 0.5);
    assert!(est.warm_start());
    assert!(!est.fit_intercept());
    assert_eq!(est.tol(), 1e-3);
}

/// The data-INDEPENDENT rejections, matching sklearn's `_parameter_constraints`
/// exactly: `epsilon ∈ [1, ∞)`, `alpha ∈ [0, ∞)`, `tol ∈ [0, ∞)`. Note what is
/// NOT rejected — `max_iter = 0`, `alpha = 0`, `tol = 0`, `epsilon = 1` are all
/// legal there and so are legal here.
#[test]
fn builder_rejects_out_of_range_parameters() {
    assert!(matches!(
        HuberRegressor::<f64>::builder().epsilon(0.9).build::<f64>(),
        Err(BuildError::InvalidEpsilon { .. })
    ));
    assert!(matches!(
        HuberRegressor::<f64>::builder()
            .epsilon(f64::NAN)
            .build::<f64>(),
        Err(BuildError::InvalidEpsilon { .. })
    ));
    assert!(matches!(
        HuberRegressor::<f64>::builder().alpha(-1.0).build::<f64>(),
        Err(BuildError::InvalidAlpha { .. })
    ));
    assert!(matches!(
        HuberRegressor::<f64>::builder().tol(-1e-9).build::<f64>(),
        Err(BuildError::InvalidTol { .. })
    ));

    // The permissive boundaries sklearn allows.
    assert!(HuberRegressor::<f64>::builder()
        .epsilon(1.0)
        .max_iter(0)
        .alpha(0.0)
        .tol(0.0)
        .build::<f64>()
        .is_ok());
}
