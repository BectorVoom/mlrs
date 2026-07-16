//! HistGradientBoosting prim (GBT-01) standalone hand-oracle tests.
//!
//! Exercises `mlrs_backend::prims::hist_gradient_boosting::{hgb_fit_reg,
//! hgb_fit_class, hgb_predict_reg, hgb_predict_proba}` on the tiny 8-point,
//! 2-feature dataset of `random_forest_test.rs`, whose ONE-iteration boosted
//! tree is fully HAND-COMPUTED (gains, splits, shrunk leaf values,
//! predictions), so the whole kernel pipeline (grad → blocked hist → reduce →
//! cum → gain → best-split → partition → raw update → traverse → sum) is
//! validated against exact expected VALUES — never just non-panic (the spike
//! verification discipline; a silent cpu-MLIR miscompile reads back as wrong
//! values here).
//!
//! Hand derivation (squared error, `lr = 0.5`, `l2 = 0`, `min_samples_leaf =
//! 1`, one iteration): baseline = mean(y) = 1.75; gradients `g = 1.75 − y`.
//! Root best gain = f0 @ 0.5 (`G_l = 3, H_l = 4, G_r = −3, H_r = 4` → gain
//! `9/4 + 9/4 = 4.5`, root loss 0). Left child has CONSTANT gradients → every
//! split gain is exactly 0 → leaf (the sklearn `gain <= 0` rule), value
//! `−0.5·3/4 = −0.375`. Right child best = f0 @ 0.75 (gain `3.125 + 0.125 −
//! 2.25 = 1.0`), TIED with f1 @ 0.35 — the flat-(feature, bin) tie-break
//! picks f0 (k = 5 < 8). Depth-2 leaves: values `+0.625` (rows y = 3) and
//! `+0.125` (rows y = 2). Train predictions: `1.375 / 2.375 / 1.875`.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64).
//! The CLASSIFIER f64 functions additionally skip on wgpu: the log-loss
//! kernels use `F::exp`, and 64-bit `exp` is unimplemented in this
//! environment's RADV shader compiler (ACO "Unimplemented NIR instr bit size:
//! div 64 fexp2" → driver SIGSEGV — the same landmine that fails the shipped
//! `kernel_matrix_test` f64 RBF case on wgpu). cpu remains the f64 gate for
//! the exp paths; the regressor f64 functions have no `exp` and run on wgpu.
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)]` module.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::hist_gradient_boosting::{
    hgb_fit_class, hgb_fit_reg, hgb_predict_proba, hgb_predict_reg, HgbParams,
};
use mlrs_backend::runtime::{self, ActiveRuntime};

const TOL: f64 = 1e-5;

/// Skip f64 log-loss (exp-using) cases on wgpu: 64-bit `exp` is unimplemented
/// in the RADV ACO shader compiler here and SIGSEGVs the driver (see the
/// module header). cpu/cuda/rocm run them.
fn skip_f64_exp_on_wgpu() -> bool {
    if cfg!(feature = "wgpu") {
        eprintln!("skipping f64 exp kernel on wgpu: RADV ACO lacks 64-bit fexp2");
        return true;
    }
    false
}

fn host_to_f64<F: Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("hgb tests are f32/f64 only"),
    }
}

fn from_f64<F: Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("hgb tests are f32/f64 only"),
    }
}

/// The `random_forest_test.rs` hand-oracle dataset: 8 rows × 2 features on a
/// 0.1 grid (8 distinct f0 values → with `n_bins = 8` the edges are the EXACT
/// consecutive midpoints, 0.5 included).
fn xdata() -> Vec<f64> {
    vec![
        0.1, 0.1, // y=1
        0.2, 0.9, // y=1
        0.3, 0.4, // y=1
        0.4, 0.6, // y=1
        0.9, 0.1, // y=2
        0.8, 0.3, // y=2
        0.7, 0.9, // y=3
        0.6, 0.8, // y=3
    ]
}

fn ydata() -> Vec<f64> {
    vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
}

fn params(max_iter: usize, max_depth: usize, lr: f64) -> HgbParams {
    HgbParams {
        max_iter,
        max_depth,
        n_bins: 8,
        learning_rate: lr,
        l2_regularization: 0.0,
        min_samples_leaf: 1,
    }
}

fn upload<F>(pool: &mut BufferPool<ActiveRuntime>, v: &[f64]) -> DeviceArray<ActiveRuntime, F>
where
    F: Float + CubeElement + Pod,
{
    let host: Vec<F> = v.iter().map(|&x| from_f64::<F>(x)).collect();
    DeviceArray::from_host(pool, &host)
}

/// ONE-iteration regression: model structure, shrunk leaf values and train
/// predictions all hand-computed (see the module header derivation).
fn check_regressor_hand_oracle<F>()
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev = upload::<F>(&mut pool, &xdata());
    let y_dev = upload::<F>(&mut pool, &ydata());

    let model = hgb_fit_reg::<F>(&mut pool, &x_dev, (8, 2), &y_dev, &params(1, 2, 0.5))
        .expect("fit hand-oracle regressor");

    assert_eq!(model.n_classes(), 1);
    assert_eq!(model.k(), 1);
    let base = model.baseline_host(&pool);
    assert!(
        (host_to_f64(base[0]) - 1.75).abs() <= TOL,
        "baseline: got {}, want mean(y) = 1.75",
        host_to_f64(base[0])
    );

    let feats = model.split_feature_host(&pool);
    let thrs = model.threshold_host(&pool);
    let leaves = model.is_leaf_host(&pool);
    let values = model.leaf_value_host(&pool);

    // Root: interior, f0 @ 0.5 (gain 4.5, unique maximum).
    assert_eq!(leaves[0], 0, "root must be interior");
    assert_eq!(feats[0], 0, "root split feature");
    assert!(
        (host_to_f64(thrs[0]) - 0.5).abs() <= TOL,
        "root threshold: got {}, want 0.5",
        host_to_f64(thrs[0])
    );

    // Node 1 (left): constant gradients → every gain is exactly 0 → leaf
    // (the sklearn `gain <= 0` finalize rule), value −lr·G/H = −0.375.
    assert_eq!(leaves[1], 1, "left child must leaf on zero gain");
    assert!(
        (host_to_f64(values[1]) + 0.375).abs() <= TOL,
        "left leaf value: got {}, want -0.375",
        host_to_f64(values[1])
    );

    // Node 2 (right): interior, f0 @ 0.75 — gain 1.0, TIED with f1 @ 0.35;
    // the flat-(feature, bin) strict-> tie-break picks the lower flat index.
    assert_eq!(leaves[2], 0, "right child is interior");
    assert_eq!(feats[2], 0, "right child split feature (tie-break)");
    assert!(
        (host_to_f64(thrs[2]) - 0.75).abs() <= TOL,
        "right threshold: got {}, want 0.75",
        host_to_f64(thrs[2])
    );

    // Depth-2 leaves under node 2: [0.6, 0.7] → +0.625, [0.8, 0.9] → +0.125.
    assert_eq!(leaves[5], 1);
    assert_eq!(leaves[6], 1);
    assert!(
        (host_to_f64(values[5]) - 0.625).abs() <= TOL,
        "right-left leaf value: got {}, want 0.625",
        host_to_f64(values[5])
    );
    assert!(
        (host_to_f64(values[6]) - 0.125).abs() <= TOL,
        "right-right leaf value: got {}, want 0.125",
        host_to_f64(values[6])
    );

    // Train predictions: baseline + one shrunk tree.
    let pred = hgb_predict_reg::<F>(&mut pool, &model, &x_dev, (8, 2))
        .expect("predict")
        .to_host(&pool);
    let want = [1.375, 1.375, 1.375, 1.375, 1.875, 1.875, 2.375, 2.375];
    for (i, (&got, &w)) in pred.iter().zip(want.iter()).enumerate() {
        let g = host_to_f64(got);
        assert!(
            (g - w).abs() <= TOL,
            "train prediction {i}: got {g}, want {w}"
        );
    }
}

/// Many-iteration regression convergence: with `lr = 0.5` the residual halves
/// per iteration on this separable target, so 60 iterations reach `y` to
/// well below 1e-4 (validates the sequential raw-update path end to end).
fn check_regressor_convergence<F>()
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev = upload::<F>(&mut pool, &xdata());
    let y_dev = upload::<F>(&mut pool, &ydata());

    let model = hgb_fit_reg::<F>(&mut pool, &x_dev, (8, 2), &y_dev, &params(60, 3, 0.5))
        .expect("fit convergence regressor");
    let pred = hgb_predict_reg::<F>(&mut pool, &model, &x_dev, (8, 2))
        .expect("predict")
        .to_host(&pool);
    for (i, (&got, &w)) in pred.iter().zip(ydata().iter()).enumerate() {
        let g = host_to_f64(got);
        assert!(
            (g - w).abs() <= 1e-3,
            "converged prediction {i}: got {g}, want {w}"
        );
    }
}

/// Binary log-loss: separable labels by `x0 < 0.5`; the baseline is the
/// log-odds (0 for a balanced target) and the boosted probabilities converge
/// to the correct side (argmax = label, confident within 30 iterations).
fn check_binary_classifier<F>()
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev = upload::<F>(&mut pool, &xdata());
    let y_idx: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 1, 1];

    let model = hgb_fit_class::<F>(&mut pool, &x_dev, (8, 2), &y_idx, 2, &params(30, 2, 0.3))
        .expect("fit binary classifier");
    assert_eq!(model.k(), 1, "binary uses ONE raw column");
    assert_eq!(model.n_classes(), 2);
    let base = model.baseline_host(&pool);
    assert!(
        host_to_f64(base[0]).abs() <= TOL,
        "balanced binary baseline must be logit(0.5) = 0, got {}",
        host_to_f64(base[0])
    );

    let proba = hgb_predict_proba::<F>(&mut pool, &model, &x_dev, (8, 2))
        .expect("predict_proba")
        .to_host(&pool);
    for (i, &want) in y_idx.iter().enumerate() {
        let p0 = host_to_f64(proba[i * 2]);
        let p1 = host_to_f64(proba[i * 2 + 1]);
        assert!(
            (p0 + p1 - 1.0).abs() <= TOL,
            "row {i}: probabilities must sum to 1 (got {p0} + {p1})"
        );
        let got = if p1 > p0 { 1 } else { 0 };
        assert_eq!(got, want, "row {i}: argmax label");
        let pw = if want == 1 { p1 } else { p0 };
        assert!(pw > 0.9, "row {i}: converged proba {pw} not confident");
    }
}

/// Multiclass log-loss (3 classes → K = 3 batched trees per iteration):
/// baseline = mean-centered log priors; train argmax recovers the labels and
/// probability rows sum to 1.
fn check_multiclass_classifier<F>()
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev = upload::<F>(&mut pool, &xdata());
    let y_idx: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 2, 2];

    let model = hgb_fit_class::<F>(&mut pool, &x_dev, (8, 2), &y_idx, 3, &params(30, 2, 0.3))
        .expect("fit multiclass classifier");
    assert_eq!(model.k(), 3, "multiclass uses n_classes raw columns");

    // Mean-centered log priors: props (1/2, 1/4, 1/4) → logs mean-centered.
    let logs = [0.5f64.ln(), 0.25f64.ln(), 0.25f64.ln()];
    let mean = logs.iter().sum::<f64>() / 3.0;
    let base = model.baseline_host(&pool);
    for (c, &l) in logs.iter().enumerate() {
        assert!(
            (host_to_f64(base[c]) - (l - mean)).abs() <= TOL,
            "baseline[{c}]: got {}, want {}",
            host_to_f64(base[c]),
            l - mean
        );
    }

    let proba = hgb_predict_proba::<F>(&mut pool, &model, &x_dev, (8, 2))
        .expect("predict_proba")
        .to_host(&pool);
    for (i, &want) in y_idx.iter().enumerate() {
        let row: Vec<f64> = (0..3).map(|c| host_to_f64(proba[i * 3 + c])).collect();
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() <= TOL, "row {i}: proba sum {sum} != 1");
        let mut best = 0usize;
        for c in 1..3 {
            if row[c] > row[best] {
                best = c;
            }
        }
        assert_eq!(best as u32, want, "row {i}: argmax label (proba {row:?})");
    }
}

/// Determinism: two identical fits produce byte-identical predictions (no
/// RNG anywhere in the HGB pipeline).
fn check_deterministic<F>()
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev = upload::<F>(&mut pool, &xdata());
    let y_dev = upload::<F>(&mut pool, &ydata());

    let p = params(10, 3, 0.1);
    let m1 = hgb_fit_reg::<F>(&mut pool, &x_dev, (8, 2), &y_dev, &p).expect("fit 1");
    let m2 = hgb_fit_reg::<F>(&mut pool, &x_dev, (8, 2), &y_dev, &p).expect("fit 2");
    let p1 = hgb_predict_reg::<F>(&mut pool, &m1, &x_dev, (8, 2))
        .expect("predict 1")
        .to_host(&pool);
    let p2 = hgb_predict_reg::<F>(&mut pool, &m2, &x_dev, (8, 2))
        .expect("predict 2")
        .to_host(&pool);
    for (i, (&a, &b)) in p1.iter().zip(p2.iter()).enumerate() {
        assert_eq!(
            host_to_f64(a).to_bits(),
            host_to_f64(b).to_bits(),
            "prediction {i} must be bit-identical across fits"
        );
    }
}

#[test]
fn regressor_hand_oracle_f32() {
    check_regressor_hand_oracle::<f32>();
}

#[test]
fn regressor_hand_oracle_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_regressor_hand_oracle::<f64>();
}

#[test]
fn regressor_convergence_f32() {
    check_regressor_convergence::<f32>();
}

#[test]
fn regressor_convergence_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_regressor_convergence::<f64>();
}

#[test]
fn binary_classifier_f32() {
    check_binary_classifier::<f32>();
}

#[test]
fn binary_classifier_f64() {
    if capability::skip_f64_with_log() || skip_f64_exp_on_wgpu() {
        return;
    }
    check_binary_classifier::<f64>();
}

#[test]
fn multiclass_classifier_f32() {
    check_multiclass_classifier::<f32>();
}

#[test]
fn multiclass_classifier_f64() {
    if capability::skip_f64_with_log() || skip_f64_exp_on_wgpu() {
        return;
    }
    check_multiclass_classifier::<f64>();
}

#[test]
fn deterministic_across_fits_f32() {
    check_deterministic::<f32>();
}

/// Geometry / hyperparameter validation surfaces typed errors BEFORE any
/// launch (T-05-03-01): wrong y length, bad class indices, zero iterations,
/// out-of-range depth/bins, mismatched predict width.
#[test]
fn validation_rejects_bad_inputs() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev = upload::<f32>(&mut pool, &xdata());
    let y_dev = upload::<f32>(&mut pool, &[1.0, 2.0, 3.0]);
    let ok = params(1, 2, 0.5);

    assert!(hgb_fit_reg::<f32>(&mut pool, &x_dev, (8, 2), &y_dev, &ok).is_err());

    let y_full = upload::<f32>(&mut pool, &ydata());
    let mut bad = ok;
    bad.max_iter = 0;
    assert!(hgb_fit_reg::<f32>(&mut pool, &x_dev, (8, 2), &y_full, &bad).is_err());
    let mut bad = ok;
    bad.max_depth = 17;
    assert!(hgb_fit_reg::<f32>(&mut pool, &x_dev, (8, 2), &y_full, &bad).is_err());
    let mut bad = ok;
    bad.n_bins = 1;
    assert!(hgb_fit_reg::<f32>(&mut pool, &x_dev, (8, 2), &y_full, &bad).is_err());
    let mut bad = ok;
    bad.learning_rate = 0.0;
    assert!(hgb_fit_reg::<f32>(&mut pool, &x_dev, (8, 2), &y_full, &bad).is_err());
    let mut bad = ok;
    bad.l2_regularization = -1.0;
    assert!(hgb_fit_reg::<f32>(&mut pool, &x_dev, (8, 2), &y_full, &bad).is_err());
    let mut bad = ok;
    bad.min_samples_leaf = 0;
    assert!(hgb_fit_reg::<f32>(&mut pool, &x_dev, (8, 2), &y_full, &bad).is_err());

    // Class index out of range + a class absent from the dense index space.
    assert!(hgb_fit_class::<f32>(&mut pool, &x_dev, (8, 2), &[0, 0, 0, 0, 1, 1, 3, 1], 3, &ok)
        .is_err());
    assert!(hgb_fit_class::<f32>(&mut pool, &x_dev, (8, 2), &[0, 0, 0, 0, 2, 2, 2, 2], 3, &ok)
        .is_err());

    // Predict geometry: wrong feature width.
    let model = hgb_fit_reg::<f32>(&mut pool, &x_dev, (8, 2), &y_full, &ok).expect("fit");
    let xq3 = upload::<f32>(&mut pool, &[0.1, 0.2, 0.3]);
    assert!(hgb_predict_reg::<f32>(&mut pool, &model, &xq3, (1, 3)).is_err());
}
