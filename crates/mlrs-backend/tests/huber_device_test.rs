//! `HuberObjective`'s DEVICE engine (HUBER-02) — the resident-`g` arm.
//!
//! `huber_objective_test.rs` already checks whatever arm the active backend
//! compiles against a naive host reference, so on wgpu/cuda/rocm it validates
//! the maths of this engine. What it cannot check is the thing HUBER-02 is
//! actually about: that the engine which keeps the per-sample gradient factor
//! `g` DEVICE-resident agrees with the round-trip arm it replaced, and that the
//! `MLRS_HUBER_DEVICE` A/B knob selecting between them is LIVE.
//!
//! That second half matters more than it sounds. A knob that silently does
//! nothing turns every A/B sweep built on it into a comparison of a variant
//! against ITSELF — flat, plausible, and worthless ([[mlrs-bench-verify-knob-is-live]]).
//! [`ab_knob_selects_a_different_arm`] pins it by construction rather than by
//! timing: the two arms compute `Σᵢ gᵢ` (the intercept's gradient entry) by
//! genuinely different routes — a blocked device fold in `F` versus a serial
//! host sum in `f64` — so on a fixture large enough for the two orders to
//! differ, bit-equality of that entry would mean one arm never ran.
//!
//! The blocked reduction also has a shape the reference test does not probe: a
//! RAGGED last block. `quad_blocks` picks `nblocks ≈ rows_per_block ≈ √n`, so
//! any `n` that is not a perfect square leaves the final block short, and a
//! kernel that clamped its row range wrongly would drop or double-count exactly
//! those rows. [`ragged_blocks_cover_every_row`] walks a run of such `n`.
//!
//! Every test here forces `MLRS_HUBER_ENGINE=device`. The arm is chosen per fit
//! rather than per build, and the measured crossover keeps fixtures this size on
//! the fused host pass, so WITHOUT that force each assertion below would be
//! comparing the host arm against itself.
//!
//! The whole file is skipped on the cpu backend, where `cubecl-cpu` maps one OS
//! thread per unit and the device arm is refused as a correctness gate rather
//! than a preference — the override cannot reach it there, by design.
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`.

#![cfg(not(feature = "cpu"))]

use mlrs_backend::device::Device;
use mlrs_backend::abflag;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::huber_objective::{HuberDesign, HuberEval, HuberObjective};
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Deterministic `[-1, 1)` stream (splitmix64), so a failure is reproducible.
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

/// One fixture: an `n × d` design, targets with gross outliers in them (without
/// those the outlier branch never runs and half of each arm is untested), and a
/// weight vector.
struct Fixture {
    x: Vec<f32>,
    targets: Vec<f64>,
    weights: Vec<f64>,
    w: Vec<f64>,
}

/// Everything here runs at `f32`: it is the ONE dtype every device backend in
/// this repo supports. rocm's `cubek-matmul` rejects `f64` operands outright and
/// cuda does not advertise `f64` at all ([[mlrs-rocm-hardware-env]],
/// [[mlrs-cubecl-cuda-f64-not-advertised]]), so an `f64` fixture here would
/// self-skip on exactly the hardware this engine exists for.
fn fixture(seed: u64, n: usize, d: usize, d_aug: usize) -> Fixture {
    let x64 = uniform_pm1(seed, n * d);
    let shock = uniform_pm1(seed ^ 0xC0DE, n);
    let targets: Vec<f64> = uniform_pm1(seed ^ 0xABCD, n)
        .iter()
        .enumerate()
        // ~8 % of rows take a large additive shock, so they land outside `ε·σ`.
        .map(|(i, v)| {
            v * 3.0
                + if shock[i] > 0.84 {
                    25.0 * shock[i]
                } else {
                    0.0
                }
        })
        .collect();
    Fixture {
        x: x64.iter().map(|&v| v as f32).collect(),
        targets,
        weights: uniform_pm1(seed ^ 0x1234, n)
            .iter()
            .map(|v| 1.5 + v)
            .collect(),
        w: uniform_pm1(seed ^ 0x55AA, d_aug),
    }
}

/// Evaluate the fixture through `HuberObjective`, with `MLRS_HUBER_DEVICE`
/// forced to `knob` for the duration (`None` = the product default).
///
/// Forced through [`abflag`], never `std::env::set_var`: the latter is an
/// `environ` data race against every sibling test's dispatcher read, and it
/// leaks process-wide so the OTHER arm's assertions become vacuous
/// ([[mlrs-abflag-test-knobs]]).
fn eval_with(
    knob: Option<&str>,
    f: &Fixture,
    (n, d): (usize, usize),
    fit_intercept: bool,
    weighted: bool,
    sigma: f64,
    epsilon: f64,
) -> HuberEval {
    // The ENGINE choice and the WITHIN-engine A/B are separate knobs. The
    // engine gate keeps oracle-sized fits on the fused host pass by default, so
    // without this force every assertion in this file would be comparing the
    // host arm against itself — the vacuous-A/B failure mode
    // ([[mlrs-bench-verify-knob-is-live]]) arrived at from a third direction.
    let _engine = abflag::force("MLRS_HUBER_ENGINE", "device");
    let _guard = match knob {
        Some(v) => abflag::force("MLRS_HUBER_DEVICE", v),
        None => abflag::clear("MLRS_HUBER_DEVICE"),
    };
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let obj = HuberObjective::<f32>::new(
        &mut pool,
        HuberDesign::Host(&f.x),
        (n, d),
        f.targets.clone(),
        weighted.then(|| f.weights.clone()),
        fit_intercept,
        Device::Auto,
    )
    .expect("HuberObjective::new rejected a valid geometry");
    let out = obj
        .eval(&mut pool, &f.w[..obj.d_aug()], sigma, epsilon)
        .expect("eval");
    obj.release_into(&mut pool);
    out
}

/// The band two arms of the SAME maths may differ by at `f32`, over `n` terms.
///
/// `k·√n·|term|·ε_f32` — the random-walk round-off of an `n`-term `f32` sum. The
/// `√n` and the `ε_f32` are the model; `k = 4` covers the two sides being on
/// opposite ends of it. The `|term|` matters and is easy to forget: this file's
/// largest reduction is `xᵀg`, whose summands are `gᵢ·xᵢⱼ` with
/// `|gᵢ| ≤ 2·ε_huber·swᵢ ≈ 6.75` on an outlier at the fixture's weights — so the
/// absolute error floor is ~7× what a unit-magnitude model predicts, on a
/// RESULT of order 1. That is why the band is absolute-ish (`assert_close`
/// floors the scale at 1) rather than purely relative.
///
/// NOT the flat `1e-5` this file first used: that is the repo's
/// *sklearn-agreement* bar, a statement about the ANSWER, and it sits below the
/// floor two `f32` reductions can meet once `n` is a few thousand.
///
/// The outlier COUNT is deliberately still compared for exact equality: it is
/// an integer, and a width difference flipping a sample across `ε·σ` is a real
/// event worth looking at rather than a tolerance to widen.
fn cross_arm_band(n: usize) -> f64 {
    4.0 * (n as f64).sqrt() * HUBER_MAX_TERM * f32::EPSILON as f64
}

/// Bound on `|gᵢ·xᵢⱼ|`, the largest summand any reduction here accumulates:
/// `2·ε_huber·max(swᵢ)·max|xᵢⱼ|` = `2 · 1.35 · 2.5 · 1.0`, from
/// [`fixture`]'s ranges. Written out rather than folded into a single fudge
/// constant so that a fixture change which widens the weights makes the band
/// move for a reason.
const HUBER_MAX_TERM: f64 = 2.0 * 1.35 * 2.5;

/// [`cross_arm_band`], widened for the `MLRS_HUBER_DEVICE=gemm` route.
///
/// The matmul substrate is free to reassociate and to tile the contraction
/// however it likes, so its summation order is neither the host's nor the
/// blocked kernels', and its error is a multiple of the random-walk model
/// rather than a match for it. Kept SEPARATE from the tight band deliberately:
/// the claim that matters — that the resident engine agrees with the round-trip
/// arm — is still asserted at [`cross_arm_band`], and loosening that to
/// accommodate a substrate neither arm uses would have thrown away the
/// sensitivity where it counts.
fn gemm_band(n: usize) -> f64 {
    8.0 * cross_arm_band(n)
}

/// `|a − b| ≤ tol·max(1, |b|)` — the mixed abs/rel form, because these
/// quantities range from an `O(1)` outlier count to an `O(n)` squared-loss sum.
fn assert_close(got: f64, expected: f64, tol: f64, what: &str) {
    let scale = expected.abs().max(1.0);
    assert!(
        (got - expected).abs() <= tol * scale,
        "{what}: got {got:.12e}, expected {expected:.12e} \
         (|Δ| = {:.3e}, allowed {:.3e})",
        (got - expected).abs(),
        tol * scale
    );
}

/// The resident-`g` engine and the round-trip arm it replaced agree, across
/// both intercept modes and both weighting modes.
///
/// The tolerance is [`cross_arm_band`] rather than bit-equality, and
/// deliberately so: the arms reduce in DIFFERENT widths and orders — the device
/// engine sums `F` in a two-level blocked fold, the round-trip arm sums `f64`
/// serially on the host, the `gemm` route lets the matmul substrate reassociate
/// as it likes. Requiring them to match exactly would be requiring the device
/// engine not to exist.
#[test]
fn device_engine_matches_roundtrip_arm() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    // `d = 12` takes the FUSED row pass; `d = 96` is above `HUBER_FUSE_MAX_D`
    // and takes the SPLIT `margin → classify` pair, which the default path
    // reaches only on a wide design. Both widths are walked because the split
    // is a measured perf choice, and a perf choice that silently changed the
    // answer would be the worst kind.
    for (n, d) in [(4_096usize, 12usize), (2_048, 96)] {
        check_agreement(n, d);
    }
}

/// The agreement checks for one `(n, d)` geometry.
fn check_agreement(n: usize, d: usize) {
    let band = cross_arm_band(n);
    let gband = gemm_band(n);
    for fit_intercept in [true, false] {
        for weighted in [false, true] {
            let d_aug = if fit_intercept { d + 1 } else { d };
            let f = fixture(0xBEEF, n, d, d_aug);
            let label = format!("n={n}/d={d}/intercept={fit_intercept}/weighted={weighted}");
            let dev = eval_with(None, &f, (n, d), fit_intercept, weighted, 0.9, 1.35);
            let rt = eval_with(Some("0"), &f, (n, d), fit_intercept, weighted, 0.9, 1.35);

            // The `gemm` A/B route is the ONLY caller of the split
            // `huber_margin_rows` + `huber_classify_rows` pair — the product
            // path runs the fused `huber_row_pass`, whose classification body
            // is a deliberate duplicate of the split kernel's. This is the
            // check that fails if the two ever drift.
            let gm = eval_with(Some("gemm"), &f, (n, d), fit_intercept, weighted, 0.9, 1.35);
            assert_close(gm.sq_sum, rt.sq_sum, gband, &format!("{label}::gemm/sq_sum"));
            assert_eq!(
                gm.n_outliers, rt.n_outliers,
                "{label}: the split classify kernel (MLRS_HUBER_DEVICE=gemm) \
                 and the host loop disagree on the outlier count — the fused \
                 and split classification bodies have drifted"
            );
            for (j, (&a, &b)) in gm.xtg.iter().zip(rt.xtg.iter()).enumerate() {
                assert_close(a, b, gband, &format!("{label}::gemm/xtg[{j}]"));
            }

            assert_close(dev.sq_sum, rt.sq_sum, band, &format!("{label}::sq_sum"));
            assert_close(
                dev.out_abs_sum,
                rt.out_abs_sum,
                band,
                &format!("{label}::out_abs_sum"),
            );
            assert_close(
                dev.out_sw_sum,
                rt.out_sw_sum,
                band,
                &format!("{label}::out_sw_sum"),
            );
            // The classification is a THRESHOLD test, so the count is an
            // integer both arms must reach identically — a residual near `ε·σ`
            // is the one place a width difference could flip a sample, and if
            // it ever does, that is a real divergence to look at rather than a
            // tolerance to widen.
            assert_eq!(
                dev.n_outliers, rt.n_outliers,
                "{label}: outlier COUNT differs between the device and \
                 round-trip arms"
            );
            assert!(
                dev.n_outliers > 0 && dev.n_outliers < n,
                "{label}: {} of {n} rows are outliers — the fixture stopped \
                 exercising BOTH branches of the classification",
                dev.n_outliers
            );
            assert_eq!(dev.xtg.len(), d_aug, "{label}: xtg length");
            assert_eq!(rt.xtg.len(), d_aug, "{label}: xtg length (round-trip)");
            for (j, (&a, &b)) in dev.xtg.iter().zip(rt.xtg.iter()).enumerate() {
                assert_close(a, b, band, &format!("{label}::xtg[{j}]"));
            }
        }
    }
}

/// The `MLRS_HUBER_DEVICE` knob really does select a different implementation.
///
/// Asserted STRUCTURALLY, not by timing. The intercept's gradient entry is
/// `Σᵢ gᵢ`; the device arm forms it as a two-level blocked fold in `F` and the
/// round-trip arm as a serial `f64` host sum. Over 65 536 terms those two
/// orders cannot coincide to the last bit unless one of them never ran — which
/// is exactly the failure mode ([[mlrs-bench-verify-knob-is-live]]) that makes
/// a whole A/B sweep vacuous, so it is pinned here rather than assumed.
///
/// The same assertion also proves the two are not accidentally the same code
/// path via some `cfg` — the reason it is a separate test from the agreement
/// one above, which would PASS if the knob were dead.
#[test]
fn ab_knob_selects_a_different_arm() {
    let (n, d) = (65_536, 8);
    let f = fixture(0x51DE, n, d, d + 1);
    let dev = eval_with(None, &f, (n, d), true, false, 0.9, 1.35);
    let rt = eval_with(Some("0"), &f, (n, d), true, false, 0.9, 1.35);

    let a = dev.xtg[d];
    let b = rt.xtg[d];
    assert_ne!(
        a.to_bits(),
        b.to_bits(),
        "the device and round-trip arms produced a BIT-IDENTICAL Σgᵢ over \
         {n} terms ({a:.17e}); they reduce in different widths and orders, so \
         this means MLRS_HUBER_DEVICE is dead and every A/B built on it is \
         comparing one arm against itself"
    );
    // …and still the same number, to the accuracy that matters.
    assert_close(a, b, cross_arm_band(n), "Σgᵢ");
}

/// A device-resident ingress and a host ingress produce the same evaluation.
///
/// This is the HUBER-02 zero-copy claim's correctness half: because the
/// synthetic intercept column is never materialized, a `HuberDesign::Device`
/// operand is BORROWED and read in place rather than pulled to host, augmented
/// and pushed back. The two ingresses must therefore be indistinguishable in
/// their output — and the borrowed buffer must survive, which the caller's
/// `release_into` after the objective's would fault on if the objective had
/// wrongly taken ownership.
#[test]
fn device_ingress_matches_host_ingress() {
    let (n, d) = (2_048, 10);
    let f = fixture(0x0DDBA11, n, d, d + 1);
    let _engine = abflag::force("MLRS_HUBER_ENGINE", "device");
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let xd: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &f.x);
    let mut results = Vec::new();
    for host_ingress in [true, false] {
        let design = if host_ingress {
            HuberDesign::Host(f.x.as_slice())
        } else {
            HuberDesign::Device(&xd)
        };
        let obj =
            HuberObjective::<f32>::new(&mut pool, design, (n, d), f.targets.clone(), None, true, Device::Auto)
                .expect("HuberObjective::new rejected a valid geometry");
        let ev = obj.eval(&mut pool, &f.w, 0.9, 1.35).expect("eval");
        let mask = obj
            .outlier_mask(&mut pool, &f.w, 0.9, 1.35)
            .expect("outlier_mask");
        assert_eq!(
            mask.iter().filter(|&&m| m).count(),
            ev.n_outliers,
            "host_ingress={host_ingress}: the mask kernel disagrees with the \
             reduction's own outlier count"
        );
        obj.release_into(&mut pool);
        results.push(ev);
    }
    // The borrowed operand outlived the objective that read it.
    xd.release_into(&mut pool);

    let (h, dv) = (&results[0], &results[1]);
    assert_eq!(
        h.n_outliers, dv.n_outliers,
        "the two ingresses classified a different number of samples"
    );
    assert_close(dv.sq_sum, h.sq_sum, 1e-6, "sq_sum");
    assert_close(dv.out_abs_sum, h.out_abs_sum, 1e-6, "out_abs_sum");
    assert_close(dv.out_sw_sum, h.out_sw_sum, 1e-6, "out_sw_sum");
    for (j, (&a, &b)) in dv.xtg.iter().zip(h.xtg.iter()).enumerate() {
        assert_close(a, b, 1e-6, &format!("xtg[{j}]"));
    }
}

/// The blocked fold covers every row when the last block is RAGGED.
///
/// `quad_blocks` targets `nblocks ≈ rows_per_block ≈ √n`, so every `n` that is
/// not a perfect square leaves a short final block. A kernel that clamped its
/// row range wrongly would drop those rows (or, with the clamp missing
/// entirely, read past `n`); either way the outlier COUNT — an exact integer
/// over every row — is what catches it, so that is what this asserts, against
/// the round-trip arm's independent host loop.
///
/// The run brackets a perfect square (`64² = 4 096`) on both sides and includes
/// a prime, which is the worst case for the layout.
#[test]
fn ragged_blocks_cover_every_row() {
    let d = 6;
    for &n in &[4_093usize, 4_095, 4_096, 4_097, 5_000, 7_919] {
        let band = cross_arm_band(n);
        let f = fixture(0xF00D ^ n as u64, n, d, d + 1);
        let dev = eval_with(None, &f, (n, d), true, true, 0.75, 1.35);
        let rt = eval_with(Some("0"), &f, (n, d), true, true, 0.75, 1.35);
        assert_eq!(
            dev.n_outliers, rt.n_outliers,
            "n={n}: the blocked fold counted {} outliers where the serial host \
             loop counted {} — the ragged last block is mis-covered",
            dev.n_outliers, rt.n_outliers
        );
        assert_close(dev.sq_sum, rt.sq_sum, band, &format!("n={n}::sq_sum"));
        assert_close(
            dev.out_sw_sum,
            rt.out_sw_sum,
            band,
            &format!("n={n}::out_sw_sum"),
        );
    }
}
