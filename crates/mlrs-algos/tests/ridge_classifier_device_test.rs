//! `RidgeClassifier` ON-DEVICE fit and predict (RIDGECLF-CUDA) — arm-equivalence
//! gates.
//!
//! The sklearn-oracle gates live in `ridge_classifier_test.rs` and already cover
//! the fused device fit end to end (its `fit_case_device` helper calls
//! `fit_with_sample_weight`, which now routes the two normal-equations solvers
//! through `fit_device_normal_equations`). What THIS file adds is the pairwise
//! agreement the oracle cannot see:
//!
//! | gate | what it proves |
//! |---|---|
//! | fused device fit ≡ shared-Gram HOST fit | the two fit arms agree on `coef_`/`intercept_`, across `fit_intercept`, `class_weight`, `sample_weight` and `positive` |
//! | device `predict` ≡ host `predict` | the fused classify kernel's `argmax`/sign/`classes_` lookup matches the host decision logic EXACTLY (labels are integers — no tolerance) |
//! | device `decision_function` ≡ host | the multi-target predict kernel's scores match the host matvec |
//! | binary (`n_targets == 1`) and multiclass | the two decision rules are different code paths in both arms |
//! | non-contiguous `classes_` | the label table is the training labels, never a fabricated `0..k` range |
//!
//! Every case runs on whatever backend the crate was built with, so the same
//! file is the cuda/rocm/wgpu gate and the cpu one.
//!
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::linear::ridge::RidgeSolver;
use mlrs_algos::linear::ridge_classifier::{ClassWeight, RidgeClassifier};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Deterministic operands without a fixture: a 64-bit LCG, so both arms see
/// byte-identical inputs on every run and a failure is reproducible from the
/// seed alone.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Top 53 bits → [0, 1), then centered to [-1, 1).
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("ridge_classifier is f32/f64 only"),
    }
}

fn to_f64<F: Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("ridge_classifier is f32/f64 only"),
    }
}

/// One synthetic problem: an `n × d` design, `n` labels drawn from `labels`
/// (round-robin so every class is populated), an `n_query × d` query, and a
/// non-uniform `sample_weight`.
struct Data<F> {
    x: Vec<F>,
    y: Vec<F>,
    xq: Vec<F>,
    sw: Vec<F>,
    n: usize,
    d: usize,
    n_query: usize,
}

fn make_data<F: Pod>(n: usize, d: usize, n_query: usize, labels: &[i64], seed: u64) -> Data<F> {
    let mut rng = Lcg(seed);
    // A label-correlated design: each class shifts one feature, so the fit is a
    // real (non-degenerate) separation rather than noise, and a coefficient
    // sign error would change `predict` rather than being absorbed.
    let mut x = Vec::with_capacity(n * d);
    let mut y = Vec::with_capacity(n);
    for r in 0..n {
        let ci = r % labels.len();
        for c in 0..d {
            let shift = if c == ci % d { 1.5 } else { 0.0 };
            x.push(f64_to::<F>(rng.next_f64() + shift));
        }
        y.push(f64_to::<F>(labels[ci] as f64));
    }
    let xq = (0..n_query * d)
        .map(|_| f64_to::<F>(rng.next_f64()))
        .collect();
    // Strictly positive, non-uniform, and NOT all equal — an all-ones vector
    // would make the weighted and unweighted arms coincide and the gate vacuous.
    let sw = (0..n)
        .map(|r| f64_to::<F>(0.25 + 1.5 * ((r % 7) as f64) / 7.0))
        .collect();
    Data {
        x,
        y,
        xq,
        sw,
        n,
        d,
        n_query,
    }
}

/// One `(solver, positive, fit_intercept, class_weight, sample_weight)` combo,
/// exercised through BOTH fit arms.
struct Spec {
    name: &'static str,
    fit_intercept: bool,
    positive: bool,
    class_weight: ClassWeight,
    sample_weight: bool,
}

fn specs() -> Vec<Spec> {
    vec![
        Spec {
            name: "default",
            fit_intercept: true,
            positive: false,
            class_weight: ClassWeight::Uniform,
            sample_weight: false,
        },
        Spec {
            name: "no_intercept",
            fit_intercept: false,
            positive: false,
            class_weight: ClassWeight::Uniform,
            sample_weight: false,
        },
        Spec {
            name: "balanced",
            fit_intercept: true,
            positive: false,
            class_weight: ClassWeight::Balanced,
            sample_weight: false,
        },
        Spec {
            name: "sample_weight",
            fit_intercept: true,
            positive: false,
            class_weight: ClassWeight::Uniform,
            sample_weight: true,
        },
        Spec {
            name: "balanced_and_sample_weight",
            fit_intercept: true,
            positive: false,
            class_weight: ClassWeight::Balanced,
            sample_weight: true,
        },
        Spec {
            name: "no_intercept_weighted",
            fit_intercept: false,
            positive: false,
            class_weight: ClassWeight::Balanced,
            sample_weight: true,
        },
        Spec {
            name: "positive",
            fit_intercept: true,
            positive: true,
            class_weight: ClassWeight::Uniform,
            sample_weight: false,
        },
    ]
}

fn build<F>(spec: &Spec) -> RidgeClassifier<F>
where
    F: Float + CubeElement + Pod,
{
    RidgeClassifier::<F>::builder()
        .alpha(0.7)
        .fit_intercept(spec.fit_intercept)
        .positive(spec.positive)
        .solver(RidgeSolver::Auto)
        .class_weight(spec.class_weight.clone())
        .tol(1e-10)
        .max_iter(Some(5000))
        .build::<F>()
        .unwrap_or_else(|e| panic!("spec '{}' must build: {e}", spec.name))
}

/// abs-OR-rel compare against a tolerance derived from the float width. The two
/// arms are NOT bit-identical by construction — the device arm accumulates the
/// Gram in `F` where the host arm accumulates in `f64` — so this is a numeric
/// agreement gate, not an equality one.
fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= tol * (1.0 + w.abs()),
            "{what}: element {i} differs: device={g:e} host={w:e} (tol={tol:e})"
        );
    }
}

/// The whole gate for one `(dtype, label set)` pair.
fn run<F>(labels: &[i64], tol: f64, label: &str)
where
    F: Float + CubeElement + Pod,
{
    let data = make_data::<F>(240, 6, 37, labels, 0x5eed_1234);
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &data.x);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &data.y);
    let xq_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &data.xq);
    let shape = (data.n, data.d);
    let qshape = (data.n_query, data.d);

    for spec in specs() {
        let sw = spec.sample_weight.then_some(data.sw.as_slice());

        // --- The FUSED device arm. ---
        let dev = build::<F>(&spec)
            .fit_with_sample_weight(&mut pool, &x_dev, Some(&y_dev), shape, sw)
            .unwrap_or_else(|e| panic!("{label} [{}]: device fit: {e}", spec.name));

        // --- The shared-Gram HOST arm, on the same operands. `fit_from_host_slice`
        //     refuses when `host_fit_applicable` is false, which is a real
        //     possibility on a device backend above the dispatch-cost floor —
        //     so the gate skips rather than fails there, and says so. ---
        let host_est = build::<F>(&spec);
        if !host_est.host_fit_applicable(shape) {
            eprintln!(
                "{label} [{}]: host arm not applicable on this backend/shape — \
                 fit comparison skipped (predict comparison still runs)",
                spec.name
            );
        } else {
            let host = host_est
                .fit_from_host_slice(&mut pool, &data.x, &data.y, shape, sw)
                .unwrap_or_else(|e| panic!("{label} [{}]: host fit: {e}", spec.name));

            assert_eq!(
                dev.classes(),
                host.classes(),
                "{label} [{}]: classes_ must agree",
                spec.name
            );
            assert_eq!(
                dev.solver(),
                host.solver(),
                "{label} [{}]: solver_ must agree",
                spec.name
            );
            let dc: Vec<f64> = dev.coef(&pool).iter().map(|&v| to_f64(v)).collect();
            let hc: Vec<f64> = host.coef(&pool).iter().map(|&v| to_f64(v)).collect();
            assert_close(&dc, &hc, tol, &format!("{label} [{}] coef_", spec.name));
            let di: Vec<f64> = dev.intercept(&pool).iter().map(|&v| to_f64(v)).collect();
            let hi: Vec<f64> = host.intercept(&pool).iter().map(|&v| to_f64(v)).collect();
            assert_close(&di, &hi, tol, &format!("{label} [{}] intercept_", spec.name));

            if spec.positive {
                assert!(
                    dc.iter().all(|&c| c >= -tol),
                    "{label} [{}]: positive=true produced a negative coef_",
                    spec.name
                );
            }
        }

        // --- DEVICE predict vs HOST predict, on the SAME fitted estimator, so
        //     any difference is the kernel's and not the fit's. ---
        let host_pred = dev
            .predict_labels_from_host(&pool, &data.xq, qshape)
            .unwrap_or_else(|e| panic!("{label} [{}]: host predict: {e}", spec.name));
        assert!(
            host_pred.operand_finite,
            "{label} [{}]: the synthetic query is finite by construction",
            spec.name
        );
        let dev_labels = dev
            .predict_labels_device(&mut pool, &xq_dev, qshape)
            .unwrap_or_else(|e| panic!("{label} [{}]: device predict: {e}", spec.name));
        let dev_labels = dev_labels.to_host(&pool);
        assert_eq!(
            dev_labels, host_pred.labels,
            "{label} [{}]: device predict must reproduce the host labels EXACTLY \
             (both take the strict `>` tie-break)",
            spec.name
        );
        // A gate that only ever sees one class would pass on a broken argmax.
        let distinct: std::collections::BTreeSet<i32> = dev_labels.iter().copied().collect();
        assert!(
            distinct.len() > 1,
            "{label} [{}]: every query row predicted the same class ({distinct:?}) — \
             the argmax/sign gate would be vacuous",
            spec.name
        );
        // …and they must be TRAINING labels, not dense `0..k` indices.
        for l in &distinct {
            assert!(
                dev.classes().contains(&(*l as i64)),
                "{label} [{}]: predicted label {l} is not one of classes_ {:?}",
                spec.name,
                dev.classes()
            );
        }

        // --- DEVICE decision_function vs HOST. ---
        let host_scores = dev
            .decision_function_from_host(&pool, &data.xq, qshape)
            .unwrap_or_else(|e| panic!("{label} [{}]: host decision: {e}", spec.name));
        let dev_scores = dev
            .decision_function_device(&mut pool, &xq_dev, qshape)
            .unwrap_or_else(|e| panic!("{label} [{}]: device decision: {e}", spec.name));
        let ds: Vec<f64> = dev_scores.to_host(&pool).iter().map(|&v| to_f64(v)).collect();
        assert_eq!(
            host_scores.n_targets,
            dev.n_targets(),
            "{label} [{}]: decision_function row width",
            spec.name
        );
        assert_close(
            &ds,
            &host_scores.values,
            tol,
            &format!("{label} [{}] decision_function", spec.name),
        );
    }
}

#[test]
fn ridge_classifier_device_arms_agree_binary_f32() {
    // `{0, 3}` — non-contiguous, so a fabricated `0..n_classes` label table
    // would fail the `classes_` membership assert rather than pass silently.
    run::<f32>(&[0, 3], 2e-3, "binary f32");
}

#[test]
fn ridge_classifier_device_arms_agree_multiclass_f32() {
    run::<f32>(&[0, 2, 5, 9], 2e-3, "multiclass f32");
}

#[test]
fn ridge_classifier_device_arms_agree_binary_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    run::<f64>(&[0, 3], 1e-9, "binary f64");
}

#[test]
fn ridge_classifier_device_arms_agree_multiclass_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    run::<f64>(&[0, 2, 5, 9], 1e-9, "multiclass f64");
}

#[test]
fn ridge_classifier_device_predict_rejects_wrong_geometry() {
    let data = make_data::<f32>(80, 4, 10, &[0, 1, 2], 7);
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &data.x);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &data.y);
    let fitted = RidgeClassifier::<f32>::new()
        .fit_with_sample_weight(&mut pool, &x_dev, Some(&y_dev), (data.n, data.d), None)
        .expect("fit must succeed");

    // A query whose feature count disagrees with the fitted one is a typed
    // error on the DEVICE ingress too, not an out-of-bounds kernel read.
    // `expect_err` is unavailable here — `DeviceArray` is deliberately not
    // `Debug` (it would have to read the device buffer back to print) — so the
    // Ok arm is rejected by hand.
    let bad: DeviceArray<ActiveRuntime, f32> =
        DeviceArray::from_host(&mut pool, &vec![0.0f32; 10 * (data.d + 1)]);
    match fitted.predict_labels_device(&mut pool, &bad, (10, data.d + 1)) {
        Ok(_) => panic!("a wrong n_features query must be rejected by predict"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("n_features"),
                "expected an n_features mismatch, got: {msg}"
            );
        }
    }

    // Same for decision_function.
    match fitted.decision_function_device(&mut pool, &bad, (10, data.d + 1)) {
        Ok(_) => panic!("a wrong n_features query must be rejected by decision_function"),
        Err(e) => assert!(format!("{e}").contains("n_features")),
    }
}
