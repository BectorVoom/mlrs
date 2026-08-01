//! Non-negative ridge primitive (`prims::nnls::ridge_nnls`) validation — the
//! device arm of `Ridge(positive=True)` / `solver='lbfgs'`.
//!
//! The kernel runs the WHOLE projected-coordinate-descent sweep loop, including
//! its convergence test, inside one cube. That makes it the kind of prim where
//! "it returned something plausible" is easy to mistake for correctness: a
//! barrier placed wrong, a stale broadcast, or a silent pipeline-creation
//! failure all yield a smooth-looking vector. So the primary gate here is NOT a
//! comparison against a reimplementation of the same algorithm — it is the
//! KKT optimality certificate for the bound-constrained problem
//! (`nnls_satisfies_kkt_*`), which any correct solver must satisfy and which a
//! subtly-broken sweep cannot. The host-CD comparison is the secondary check.
//!
//! `min_{w ≥ 0} ½‖Xw − y‖² + ½α‖w‖²` is strictly convex for `α > 0` over a box,
//! so its constrained minimiser is UNIQUE — which is what makes both checks
//! sharp rather than merely consistent.
//!
//! Fixtures deliberately include cases where the bound BINDS (the unconstrained
//! ridge solution has negative components). Without those the non-negativity
//! projection is never exercised and the suite would pass on a solver that
//! ignored the constraint entirely.
//!
//! Runs under `--features wgpu` (and cuda/rocm); on cpu the prim's dispatch
//! predicate sends `Ridge` to the algos-crate host twin instead, so these tests
//! skip there — `crates/mlrs-algos/tests/ridge_params_test.rs` covers that arm.
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::capability::{self, FloatKind};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::nnls::{device_nnls_applicable, ridge_nnls};
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::PrimError;

/// Sweep cap for the fixtures — far above what these well-conditioned systems
/// need, so a case that stops early stopped on the tolerance (which is what the
/// KKT check then holds to account), not on the cap.
const MAX_ITER: usize = 2000;

/// Stop tolerance handed to both arms.
const TOL: f64 = 1e-10;

/// A fixture: `n` samples, `d` features, ridge penalty, and whether the
/// unconstrained solution is expected to have negative components (i.e. whether
/// the non-negativity bound BINDS — asserted, so a fixture that silently stops
/// exercising the projection fails loudly instead of going quiet).
struct Case {
    name: &'static str,
    n: usize,
    d: usize,
    alpha: f64,
    binds: bool,
}

const CASES: &[Case] = &[
    // Bound inactive: every unconstrained coefficient is already positive, so
    // the solve must reproduce the plain ridge solution.
    Case { name: "interior", n: 40, d: 5, alpha: 1.0, binds: false },
    // Bound binding: sign-alternating targets drive several coefficients
    // negative unconstrained, so the projection is load-bearing.
    Case { name: "binding_small", n: 30, d: 6, alpha: 1.0, binds: true },
    Case { name: "binding_wide", n: 120, d: 24, alpha: 0.5, binds: true },
    // d = 1 degenerate cube (a single unit; the axpy and the scalar update run
    // on the same unit, so the broadcast barriers must still be correct).
    Case { name: "single_feature", n: 25, d: 1, alpha: 2.0, binds: true },
    // d = 64: a full wavefront-plus cube, and the widest shape the Ridge
    // oracle fixtures reach.
    Case { name: "d64", n: 300, d: 64, alpha: 1.0, binds: true },
    // Larger alpha: the Hessian diagonal is dominated by the penalty, which is
    // the regime where a wrong `alpha` placement (row vs diagonal) would show.
    Case { name: "heavy_penalty", n: 50, d: 8, alpha: 50.0, binds: true },
    // `d == NNLS_MAX_DIM`: the cube-dim cap exactly. This is the shape where an
    // adapter that could not honour the workgroup size would fail pipeline
    // creation SILENTLY, leaving `coef` all-zero — which the KKT dual-feasibility
    // check rejects (at `w = 0` the gradient is `−Xᵀy`, negative wherever `Xᵀy`
    // is positive), so the cap is proven reachable rather than assumed.
    Case { name: "cap_d256", n: 600, d: 256, alpha: 1.0, binds: true },
];

/// Deterministic LCG in `[-1, 1)` — fixtures must be reproducible across
/// backends and runs (no `rand` seeding drift).
fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// Build a fixture's design `X` (`n × d` row-major) and target `y` (`n`).
///
/// `binds` shapes `y` rather than `X`: an alternating-sign response makes the
/// unconstrained ridge solution mix signs (so the bound binds), while a
/// positive-combination response leaves it interior.
fn make_case(case: &Case) -> (Vec<f64>, Vec<f64>) {
    let mut st = 0x5eed_1234_u64 ^ (case.n as u64) << 17 ^ (case.d as u64);
    let x: Vec<f64> = (0..case.n * case.d).map(|_| lcg(&mut st)).collect();
    let y: Vec<f64> = (0..case.n)
        .map(|i| {
            let row = &x[i * case.d..(i + 1) * case.d];
            let mut acc = 0.0;
            for (j, &v) in row.iter().enumerate() {
                // Interior: all-positive weights. Binding: alternating weights,
                // which pushes roughly half the coefficients negative — except
                // at `d == 1`, where there is no odd index to alternate onto, so
                // the single coefficient itself carries the negative weight.
                let negate = case.binds && (j % 2 == 1 || case.d == 1);
                let w = if negate { -1.5 } else { 1.0 };
                acc += w * v;
            }
            acc + 0.05 * lcg(&mut st)
        })
        .collect();
    (x, y)
}

/// Host `gram = XᵀX` (`d×d` row-major) and `xty = Xᵀy` (`d`), in f64.
fn host_gram_xty(x: &[f64], y: &[f64], n: usize, d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut gram = vec![0.0f64; d * d];
    let mut xty = vec![0.0f64; d];
    for i in 0..n {
        for a in 0..d {
            xty[a] += x[i * d + a] * y[i];
            for b in 0..d {
                gram[a * d + b] += x[i * d + a] * x[i * d + b];
            }
        }
    }
    (gram, xty)
}

/// The gradient of the ridge objective at `w`: `∇f = G·w − c + α·w`.
fn gradient(gram: &[f64], xty: &[f64], w: &[f64], d: usize, alpha: f64) -> Vec<f64> {
    (0..d)
        .map(|j| {
            let mut acc = 0.0;
            for k in 0..d {
                acc += gram[j * d + k] * w[k];
            }
            acc - xty[j] + alpha * w[j]
        })
        .collect()
}

/// Host projected cyclic coordinate descent — the reference solve (the twin of
/// `mlrs_algos::linear::ridge_solvers::nonnegative_cd`, restated here because
/// `mlrs-backend` cannot depend on `mlrs-algos`).
fn host_nnls(gram: &[f64], xty: &[f64], d: usize, alpha: f64, tol: f64, max_iter: usize) -> Vec<f64> {
    let mut w = vec![0.0f64; d];
    for _ in 0..max_iter {
        let mut max_change = 0.0f64;
        let mut max_weight = 0.0f64;
        for j in 0..d {
            let hess = gram[j * d + j] + alpha;
            if hess <= 0.0 {
                continue;
            }
            let mut grad = -xty[j] + alpha * w[j];
            for k in 0..d {
                grad += gram[j * d + k] * w[k];
            }
            let next = (w[j] - grad / hess).max(0.0);
            max_change = max_change.max((next - w[j]).abs());
            w[j] = next;
            max_weight = max_weight.max(next.abs());
        }
        if max_change <= tol * max_weight.max(1.0) {
            break;
        }
    }
    w
}

/// Unconstrained ridge solve `(G + αI)·w = c` by Gaussian elimination with
/// partial pivoting — used ONLY to assert that a fixture's bound really binds
/// (or really does not), so the suite cannot go quietly vacuous.
fn unconstrained_ridge(gram: &[f64], xty: &[f64], d: usize, alpha: f64) -> Vec<f64> {
    let mut a = vec![0.0f64; d * (d + 1)];
    for r in 0..d {
        for c in 0..d {
            a[r * (d + 1) + c] = gram[r * d + c];
        }
        a[r * (d + 1) + r] += alpha;
        a[r * (d + 1) + d] = xty[r];
    }
    for col in 0..d {
        let piv = (col..d)
            .max_by(|&p, &q| {
                a[p * (d + 1) + col]
                    .abs()
                    .partial_cmp(&a[q * (d + 1) + col].abs())
                    .unwrap()
            })
            .unwrap();
        if piv != col {
            for c in 0..=d {
                a.swap(col * (d + 1) + c, piv * (d + 1) + c);
            }
        }
        let p = a[col * (d + 1) + col];
        for r in 0..d {
            if r == col {
                continue;
            }
            let f = a[r * (d + 1) + col] / p;
            for c in col..=d {
                a[r * (d + 1) + c] -= f * a[col * (d + 1) + c];
            }
        }
    }
    (0..d).map(|r| a[r * (d + 1) + d] / a[r * (d + 1) + r]).collect()
}

/// Run `ridge_nnls` end-to-end at element width `F` over a host-formed Gram,
/// returning `coef` promoted to f64.
fn run_nnls<F>(gram64: &[f64], xty64: &[f64], d: usize, alpha: f64) -> Vec<f64>
where
    F: Float + CubeElement + Pod,
{
    let to_f = |v: f64| -> F {
        match std::mem::size_of::<F>() {
            4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
            8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
            _ => unreachable!("nnls_test is f32/f64 only"),
        }
    };
    let to_f64 = |v: &F| -> f64 {
        match std::mem::size_of::<F>() {
            4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(v)) as f64,
            8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(v)),
            _ => unreachable!("nnls_test is f32/f64 only"),
        }
    };

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let gram_h: Vec<F> = gram64.iter().map(|&v| to_f(v)).collect();
    let xty_h: Vec<F> = xty64.iter().map(|&v| to_f(v)).collect();
    let gram_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &gram_h);
    let xty_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xty_h);

    let coef = ridge_nnls::<F>(&mut pool, &gram_dev, &xty_dev, d, alpha, TOL, Some(MAX_ITER))
        .expect("ridge_nnls rejects nothing for a valid in-cap shape");
    coef.to_host_metered(&mut pool).iter().map(to_f64).collect()
}

/// Skip guard: on cpu (and any adapter the kernel cannot be staged on) `Ridge`
/// takes the algos-crate host twin, so the device prim is not the arm under
/// test. Reported rather than silently passing.
fn skip_non_device<F>(d: usize, label: &str) -> bool {
    if device_nnls_applicable::<F>(d) {
        return false;
    }
    println!(
        "nnls {label} backend={}: SKIPPED (host twin carries this arm here)",
        capability::active_backend_name()
    );
    true
}

/// **The primary gate.** The returned `w` must satisfy the KKT conditions of
/// the bound-constrained problem — feasibility (`w ≥ 0`), stationarity on the
/// free set (`∇f_j ≈ 0` where `w_j > 0`), and dual feasibility on the active
/// set (`∇f_j ≥ 0` where `w_j = 0`).
///
/// This is an INDEPENDENT certificate: it re-derives nothing from the solver's
/// own iteration and would reject a swept-but-not-converged `w`, a `w` from a
/// solver that dropped the constraint, and a `w` from one that mis-placed
/// `alpha` — none of which a comparison against a same-shaped reimplementation
/// reliably catches.
fn assert_kkt(w: &[f64], gram: &[f64], xty: &[f64], d: usize, alpha: f64, tol: f64, label: &str) {
    let g = gradient(gram, xty, w, d, alpha);
    // Scale the stationarity bound by the problem's magnitude: `∇f` carries the
    // units of `Xᵀy`, so an absolute bound alone would be meaningless across
    // fixtures of different `n`.
    let scale = xty.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    for j in 0..d {
        assert!(
            w[j] >= 0.0,
            "{label}: coefficient {j} violates the non-negativity bound: {}",
            w[j]
        );
        if w[j] > 0.0 {
            assert!(
                g[j].abs() <= tol * scale,
                "{label}: KKT stationarity failed on the FREE set at {j}: w={}, grad={:e} \
                 (bound {:e})",
                w[j],
                g[j],
                tol * scale
            );
        } else {
            assert!(
                g[j] >= -tol * scale,
                "{label}: KKT dual feasibility failed on the ACTIVE set at {j}: grad={:e} \
                 (bound {:e}) — the solve stopped at a non-optimal vertex",
                g[j],
                -tol * scale
            );
        }
    }
}

/// KKT optimality of the device solve, f64.
#[test]
fn nnls_satisfies_kkt_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F64, backend, "default");

    if capability::skip_f64_with_log() {
        println!("nnls f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }

    for case in CASES {
        if skip_non_device::<f64>(case.d, case.name) {
            continue;
        }
        let (x, y) = make_case(case);
        let (gram, xty) = host_gram_xty(&x, &y, case.n, case.d);

        // The fixture must exercise what it claims to (see the module docs).
        let unc = unconstrained_ridge(&gram, &xty, case.d, case.alpha);
        let any_neg = unc.iter().any(|&v| v < 0.0);
        assert_eq!(
            any_neg, case.binds,
            "fixture '{}' is stale: unconstrained solution {unc:?} has binds={any_neg}, \
             fixture declares binds={}",
            case.name, case.binds
        );

        let w = run_nnls::<f64>(&gram, &xty, case.d, case.alpha);
        assert_kkt(&w, &gram, &xty, case.d, case.alpha, 1e-6, case.name);
    }

    println!("nnls f64 backend={backend}: KKT-optimal on every fixture");
}

/// KKT optimality of the device solve, f32. The certificate is the same; only
/// the bound loosens to the f32 working precision of a Gram-based solve (the
/// Gram squares `X`'s condition number, so this is the accuracy the whole
/// `positive` arm has at f32 — not a concession specific to the device kernel).
#[test]
fn nnls_satisfies_kkt_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F32, backend, "default");

    for case in CASES {
        if skip_non_device::<f32>(case.d, case.name) {
            continue;
        }
        let (x, y) = make_case(case);
        let (gram, xty) = host_gram_xty(&x, &y, case.n, case.d);
        let w = run_nnls::<f32>(&gram, &xty, case.d, case.alpha);
        assert_kkt(&w, &gram, &xty, case.d, case.alpha, 1e-2, case.name);
    }

    println!("nnls f32 backend={backend}: KKT-optimal on every fixture");
}

/// The device solve agrees with the host projected-CD twin — the arm-equivalence
/// check that backs `Ridge`'s dispatch between them.
///
/// The constrained minimiser is unique, so this is a real agreement bound and
/// not just "two runs of the same code".
#[test]
fn nnls_matches_host_cd_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F64, backend, "default");

    if capability::skip_f64_with_log() {
        println!("nnls f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }

    for case in CASES {
        if skip_non_device::<f64>(case.d, case.name) {
            continue;
        }
        let (x, y) = make_case(case);
        let (gram, xty) = host_gram_xty(&x, &y, case.n, case.d);
        let got = run_nnls::<f64>(&gram, &xty, case.d, case.alpha);
        let want = host_nnls(&gram, &xty, case.d, case.alpha, TOL, MAX_ITER);

        let scale = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        for j in 0..case.d {
            let err = (got[j] - want[j]).abs();
            assert!(
                err <= 1e-6 * scale,
                "{}: device/host arm disagreement at coefficient {j}: device={}, host={}, \
                 abs_err={err:e} (bound {:e})",
                case.name,
                got[j],
                want[j],
                1e-6 * scale
            );
            // The two arms must also agree on the SUPPORT (which coefficients
            // the bound pins to zero) — a shared minimum with a different active
            // set would mean one of them is stopping early.
            assert_eq!(
                got[j] == 0.0,
                want[j] == 0.0,
                "{}: device/host active sets differ at coefficient {j}: device={}, host={}",
                case.name,
                got[j],
                want[j]
            );
        }
    }

    println!("nnls f64 backend={backend}: device arm matches the host projected-CD twin");
}

/// With the bound inactive the solve must reproduce the plain (unconstrained)
/// ridge solution — the sanity anchor that the projection is not perturbing an
/// already-feasible optimum.
#[test]
fn nnls_interior_case_matches_unconstrained_ridge_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    if capability::skip_f64_with_log() {
        println!("nnls f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = &CASES[0];
    assert!(!case.binds, "CASES[0] must be the interior fixture");
    if skip_non_device::<f64>(case.d, case.name) {
        return;
    }

    let (x, y) = make_case(case);
    let (gram, xty) = host_gram_xty(&x, &y, case.n, case.d);
    let got = run_nnls::<f64>(&gram, &xty, case.d, case.alpha);
    let want = unconstrained_ridge(&gram, &xty, case.d, case.alpha);

    for j in 0..case.d {
        assert!(
            (got[j] - want[j]).abs() <= 1e-8 * want[j].abs().max(1.0),
            "interior fixture: coefficient {j} deviates from the unconstrained ridge \
             solution: got={}, want={}",
            got[j],
            want[j]
        );
    }
    println!("nnls f64 backend={backend}: interior case reproduces unconstrained ridge");
}

/// Geometry is validated BEFORE the unsafe launch (ASVS V5): a non-square Gram,
/// a mismatched `xty`, `d = 0`, and an over-cap `d` are all rejected.
#[test]
fn nnls_rejects_bad_geometry() {
    let _ = env_logger::builder().is_test(true).try_init();
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let gram: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &[1.0f32; 9]);
    let xty: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &[1.0f32; 3]);

    // d = 4 against a 3×3 Gram.
    assert!(matches!(
        ridge_nnls::<f32>(&mut pool, &gram, &xty, 4, 1.0, TOL, None),
        Err(PrimError::ShapeMismatch { operand: "nnls.gram", .. })
    ));
    // d = 0.
    assert!(matches!(
        ridge_nnls::<f32>(&mut pool, &gram, &xty, 0, 1.0, TOL, None),
        Err(PrimError::ShapeMismatch { operand: "nnls.gram", .. })
    ));
    // Correct Gram, wrong-length xty.
    let short: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &[1.0f32; 2]);
    assert!(matches!(
        ridge_nnls::<f32>(&mut pool, &gram, &short, 3, 1.0, TOL, None),
        Err(PrimError::ShapeMismatch { operand: "nnls.xty", .. })
    ));
    // Over the single-cube cap — must be REJECTED, not launched with a cube dim
    // the adapter cannot honour (Ridge routes these to the host twin).
    assert!(matches!(
        ridge_nnls::<f32>(&mut pool, &gram, &xty, 4096, 1.0, TOL, None),
        Err(PrimError::ShapeMismatch { operand: "nnls.gram", .. })
    ));
}

/// The dispatch predicate must refuse the cpu backend and any over-cap `d` —
/// the two conditions `Ridge` relies on to keep the host twin's shapes working.
#[test]
fn device_gate_refuses_cpu_and_over_cap() {
    let over_cap = 4096usize;
    assert!(
        !device_nnls_applicable::<f32>(over_cap),
        "d={over_cap} exceeds the single-cube cap and must route to the host twin"
    );
    assert!(!device_nnls_applicable::<f32>(0), "d=0 must route to the host twin");

    let is_cpu = capability::active_backend_name() == "cpu";
    assert_eq!(
        device_nnls_applicable::<f32>(8),
        !is_cpu,
        "the device arm must be taken on every backend EXCEPT cpu (backend={})",
        capability::active_backend_name()
    );
}
