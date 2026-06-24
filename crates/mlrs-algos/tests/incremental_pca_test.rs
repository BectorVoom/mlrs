//! Plan 07-05 — IncrementalPCA (DECOMP-03) sklearn oracle tests.
//!
//! Each test loads its committed `IncrementalPCA(n_components, whiten,
//! batch_size).fit(X)` fixture, fits the device estimator BOTH via the explicit
//! `partial_fit` stream over `gen_batches(n, batch_size)` AND via the one-shot
//! `fit()` (which internally loops `partial_fit` — D-02), sign-aligns
//! `components_` rows with `align_rows` (= sklearn `svd_flip(u_based_decision=
//! False)`, D-03), and asserts every attribute (`components_`,
//! `explained_variance_`, `explained_variance_ratio_`, `singular_values_`,
//! `mean_`, `var_`, `n_samples_seen_`) + `transform`/`inverse_transform` against
//! the sklearn reference within the 1e-5 abs+rel contract (f64) / `IPCA_F32_BAND`
//! (f32).
//!
//! Two fixtures per dtype: `incremental_pca_nowhiten_*` and
//! `incremental_pca_whiten_*` (30×6, n_components=3, batch_size=10 — the stacked
//! per-batch SVD matrix `nc+bs+1=14 ≤ 256`, `n_features=6 ≤ 64`).
//!
//! f64 cases carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips-with-log, D-07). f32 stays a documented per-family band
//! (`IPCA_F32_BAND`, pinned from the standalone PRIM-07 f32 measurement in plan
//! 07-03). Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::decomposition::IncrementalPCA;
use mlrs_algos::typestate::{Fit, Fitted, PartialFit, Transform};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::sign_flip::align_rows;
use mlrs_core::{load_npz, OracleCase, Tolerance, F64_TOL};

/// IncrementalPCA fixture geometry (gen_oracle.py `IPCA_SHAPE` = 30×6).
const IPCA_N: usize = 30;
const IPCA_P: usize = 6;
const IPCA_N_COMPONENTS: usize = 3;
const IPCA_BATCH_SIZE: usize = 10;

/// f32-on-rocm per-family tolerance band for IncrementalPCA. Pinned from the
/// standalone PRIM-07 incremental-SVD merge f32 measurement (Plan 07-03 SUMMARY:
/// observed components_ max_abs 3.6e-7 / max_rel 2.0e-6 — the streaming SVD merge
/// re-expands `Σ·Vᵀ` so per-batch error does not compound). 1e-4 carries margin
/// over the measured error and matches the v1 PCA f32 family band. f64 stays
/// strict `F64_TOL` (1e-5). The IncrementalPCA estimator adds only the small p×nc
/// transform GEMM round-off on top of the merge, so this band holds for the
/// estimator-level attrs + transform/inverse_transform too (Claude's-discretion,
/// A4 — to be re-measured on rocm at the phase gate and re-pinned with margin).
const IPCA_F32_BAND: Tolerance = Tolerance::new(1e-4, 1e-4);

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
        _ => unreachable!("incremental_pca fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("incremental_pca fixtures are f32/f64 only"),
    }
}

/// numpy-`allclose` element compare: pass if `|got − exp| ≤ atol + rtol·|exp|`
/// (abs-OR-rel), the strict 1e-5 ABSOLUTE arm never loosened (the svd_test.rs
/// precedent).
fn assert_close(got: &[f64], expected: &[f64], tol: &Tolerance, what: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: length mismatch got={} expected={}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs_err = (g - e).abs();
        let allclose = abs_err <= tol.abs + tol.rel * e.abs();
        assert!(
            allclose,
            "{what}: allclose failed at {i}: got={g:e} expected={e:e} \
             abs_err={abs_err:e} (atol={:e}, rtol={:e})",
            tol.abs, tol.rel
        );
    }
}

/// Sign-align a row-major `(rows × cols)` matrix by its ROWS (each row a
/// component / singular vector) — the sklearn `svd_flip` canonicalization.
fn align_matrix_rows(mat: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let row_vecs: Vec<Vec<f64>> = (0..rows)
        .map(|r| (0..cols).map(|c| mat[r * cols + c]).collect())
        .collect();
    let flipped = align_rows(&row_vecs);
    let mut out = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = flipped[r][c];
        }
    }
    out
}

/// Sign-align a row-major `(rows × cols)` matrix by its COLUMNS (each column the
/// projection onto one component) — the transform-side `svd_flip`.
fn align_matrix_cols(mat: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let col_vecs: Vec<Vec<f64>> = (0..cols)
        .map(|c| (0..rows).map(|r| mat[r * cols + c]).collect())
        .collect();
    let flipped = align_rows(&col_vecs);
    let mut out = vec![0.0f64; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            out[r * cols + c] = flipped[c][r];
        }
    }
    out
}

/// Fitted IncrementalPCA host attributes for an oracle compare.
struct IpcaFit {
    components: Vec<f64>,
    explained_variance: Vec<f64>,
    explained_variance_ratio: Vec<f64>,
    singular_values: Vec<f64>,
    mean: Vec<f64>,
    var: Vec<f64>,
    n_samples_seen: usize,
    transform: Vec<f64>,
    inverse: Vec<f64>,
}

/// Promote a fitted `IncrementalPCA<F>` plus `transform(X)` /
/// `inverse_transform(transform(X))` to host f64 attributes.
fn collect_fit<F>(
    pca: &IncrementalPCA<F, Fitted>,
    pool: &mut BufferPool<ActiveRuntime>,
    x_dev: &DeviceArray<ActiveRuntime, F>,
    n_samples: usize,
    n_features: usize,
    nc: usize,
) -> IpcaFit
where
    F: Float + CubeElement + Pod,
{
    let promote = |v: Vec<F>| v.iter().map(|&x| host_to_f64(x)).collect::<Vec<f64>>();

    let components = promote(pca.components(pool));
    let explained_variance = promote(pca.explained_variance(pool));
    let explained_variance_ratio = promote(pca.explained_variance_ratio(pool));
    let singular_values = promote(pca.singular_values(pool));
    let mean = promote(pca.mean(pool));
    let var = promote(pca.var(pool));
    let n_samples_seen = pca.n_samples_seen();

    let z = pca
        .transform(pool, x_dev, (n_samples, n_features))
        .expect("transform(X)");
    let transform = promote(z.to_host(pool));

    let inv = pca
        .inverse_transform(pool, &z, (n_samples, nc))
        .expect("inverse_transform(transform(X))");
    let inverse = promote(inv.to_host(pool));
    z.release_into(pool);
    inv.release_into(pool);

    IpcaFit {
        components,
        explained_variance,
        explained_variance_ratio,
        singular_values,
        mean,
        var,
        n_samples_seen,
        transform,
        inverse,
    }
}

/// Fit via the EXPLICIT `partial_fit` stream over `gen_batches(n, batch_size)`
/// (the sklearn `fit()` loop made explicit) and collect the host attributes.
fn fit_via_partial_fit<F>(case: &OracleCase, whiten: bool) -> IpcaFit
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_f64 = case.expect_f64("X").to_vec();
    let x_host: Vec<F> = x_f64.iter().map(|&v| f64_to::<F>(v)).collect();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);

    let unfit = IncrementalPCA::<F>::builder()
        .n_components(IPCA_N_COMPONENTS)
        .whiten(whiten)
        .batch_size(Some(IPCA_BATCH_SIZE))
        .build::<F>()
        .expect("IncrementalPcaBuilder::build is infallible");

    // Naive equal chunking (== sklearn gen_batches ONLY when
    // IPCA_N % IPCA_BATCH_SIZE == 0, which holds here: 30 % 10 == 0). For a
    // non-divisible geometry this would emit a SHORT trailing batch, whereas
    // the real gen_batches(min_batch=n_components) FOLDS the remainder into the
    // prior batch — a different stream and a different merged state (WR-04).
    //
    // The consuming-self typestate `partial_fit` (Pitfall 5): the FIRST batch
    // consumes the `Unfit` value (`Unfit → Fitted`); every SUBSEQUENT batch
    // consumes the `Fitted` value (`Fitted → Fitted`), accumulating the stream.
    let first_b = IPCA_BATCH_SIZE.min(IPCA_N);
    let first_host: Vec<F> = x_host[0..first_b * IPCA_P].to_vec();
    let first_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &first_host);
    let mut pca = unfit
        .partial_fit(&mut pool, &first_dev, None, (first_b, IPCA_P))
        .expect("partial_fit first batch (Unfit -> Fitted)");
    first_dev.release_into(&mut pool);

    let mut start = first_b;
    while start < IPCA_N {
        let b = IPCA_BATCH_SIZE.min(IPCA_N - start);
        let batch_host: Vec<F> = x_host[start * IPCA_P..(start + b) * IPCA_P].to_vec();
        let batch_dev: DeviceArray<ActiveRuntime, F> =
            DeviceArray::from_host(&mut pool, &batch_host);
        pca = pca
            .partial_fit(&mut pool, &batch_dev, None, (b, IPCA_P))
            .expect("partial_fit batch (Fitted -> Fitted)");
        batch_dev.release_into(&mut pool);
        start += b;
    }

    let fit = collect_fit(&pca, &mut pool, &x_dev, IPCA_N, IPCA_P, IPCA_N_COMPONENTS);
    x_dev.release_into(&mut pool);
    fit
}

/// Fit via the ONE-SHOT `fit()` (which loops `partial_fit` over `gen_batches`
/// with the explicit `batch_size`, D-02) and collect the host attributes.
fn fit_via_fit<F>(case: &OracleCase, whiten: bool) -> IpcaFit
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_f64 = case.expect_f64("X").to_vec();
    let x_host: Vec<F> = x_f64.iter().map(|&v| f64_to::<F>(v)).collect();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);

    let pca = IncrementalPCA::<F>::builder()
        .n_components(IPCA_N_COMPONENTS)
        .whiten(whiten)
        .batch_size(Some(IPCA_BATCH_SIZE))
        .build::<F>()
        .expect("IncrementalPcaBuilder::build is infallible")
        .fit(&mut pool, &x_dev, None, (IPCA_N, IPCA_P))
        .expect("IncrementalPCA::fit on a valid shape");

    let fit = collect_fit(&pca, &mut pool, &x_dev, IPCA_N, IPCA_P, IPCA_N_COMPONENTS);
    x_dev.release_into(&mut pool);
    fit
}

/// Assert every fitted attribute of `fit` matches the sklearn `case` within
/// `tol` (components compared AFTER `align_rows`).
fn assert_attrs(fit: &IpcaFit, case: &OracleCase, tol: &Tolerance, label: &str) {
    assert_eq!(
        fit.n_samples_seen, IPCA_N,
        "{label}: n_samples_seen_ accumulates to n_total"
    );
    assert_eq!(
        case.expect_f64("n_samples_seen_")[0] as usize,
        IPCA_N,
        "{label}: oracle n_samples_seen_ == n"
    );
    assert_close(&fit.mean, case.expect_f64("mean_"), tol, &format!("{label} mean_"));
    assert_close(&fit.var, case.expect_f64("var_"), tol, &format!("{label} var_"));
    assert_close(
        &fit.singular_values,
        case.expect_f64("singular_values_"),
        tol,
        &format!("{label} singular_values_"),
    );
    assert_close(
        &fit.explained_variance,
        case.expect_f64("explained_variance_"),
        tol,
        &format!("{label} explained_variance_"),
    );
    assert_close(
        &fit.explained_variance_ratio,
        case.expect_f64("explained_variance_ratio_"),
        tol,
        &format!("{label} explained_variance_ratio_"),
    );

    let got_c = align_matrix_rows(&fit.components, IPCA_N_COMPONENTS, IPCA_P);
    let exp_c = align_matrix_rows(case.expect_f64("components_"), IPCA_N_COMPONENTS, IPCA_P);
    assert_close(&got_c, &exp_c, tol, &format!("{label} components_"));
}

/// Assert `transform` / `inverse_transform` match the sklearn `case` within `tol`
/// (transform compared AFTER column `align_rows` — the projection sign follows
/// the component sign).
fn assert_transforms(fit: &IpcaFit, case: &OracleCase, tol: &Tolerance, label: &str) {
    let got_t = align_matrix_cols(&fit.transform, IPCA_N, IPCA_N_COMPONENTS);
    let exp_t = align_matrix_cols(case.expect_f64("transform"), IPCA_N, IPCA_N_COMPONENTS);
    assert_close(&got_t, &exp_t, tol, &format!("{label} transform"));

    // inverse_transform(transform(X)) is sign- and basis-invariant (the sign
    // cancels in Z·components_); compare directly against sklearn's reference.
    assert_close(
        &fit.inverse,
        case.expect_f64("inverse_transform"),
        tol,
        &format!("{label} inverse_transform"),
    );
}

// ===========================================================================
// defaults-equality (BLDR-01)
// ===========================================================================

/// BLDR-01: the zero-arg `new()` defaults equal the builder's `build()` defaults
/// (`IncrementalPCA::new().hyperparams_eq(&IncrementalPCA::builder().build()?)`).
#[test]
fn incremental_pca_defaults_equal() {
    let from_new = IncrementalPCA::<f64>::new();
    let from_builder = IncrementalPCA::<f64>::builder()
        .build::<f64>()
        .expect("IncrementalPcaBuilder::build is infallible");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "IncrementalPCA::new() defaults must equal builder().build() defaults"
    );
}

// ===========================================================================
// partial_fit: all attrs via the explicit batch stream (whiten on/off)
// ===========================================================================

/// All attrs via the explicit `partial_fit` stream vs sklearn, whiten=False, f32.
#[test]
fn incremental_pca_partial_fit_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("incremental_pca_nowhiten_f32_seed42.npz"))
        .expect("load incremental_pca_nowhiten_f32");
    let fit = fit_via_partial_fit::<f32>(&case, false);
    assert_attrs(&fit, &case, &IPCA_F32_BAND, "partial_fit f32");
}

/// All attrs via the explicit `partial_fit` stream vs sklearn, whiten=False, f64.
#[test]
fn incremental_pca_partial_fit_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("incremental_pca partial_fit f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("incremental_pca_nowhiten_f64_seed42.npz"))
        .expect("load incremental_pca_nowhiten_f64");
    let fit = fit_via_partial_fit::<f64>(&case, false);
    assert_attrs(&fit, &case, &F64_TOL, "partial_fit f64");
}

// ===========================================================================
// fit(): all attrs via the one-shot fit (loops partial_fit, D-02)
// ===========================================================================

/// All attrs via the one-shot `fit()` vs sklearn, whiten=False, f32.
#[test]
fn incremental_pca_fit_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("incremental_pca_nowhiten_f32_seed42.npz"))
        .expect("load incremental_pca_nowhiten_f32");
    let fit = fit_via_fit::<f32>(&case, false);
    assert_attrs(&fit, &case, &IPCA_F32_BAND, "fit f32");
}

/// All attrs via the one-shot `fit()` vs sklearn, whiten=False, f64.
#[test]
fn incremental_pca_fit_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("incremental_pca fit f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("incremental_pca_nowhiten_f64_seed42.npz"))
        .expect("load incremental_pca_nowhiten_f64");
    let fit = fit_via_fit::<f64>(&case, false);
    assert_attrs(&fit, &case, &F64_TOL, "fit f64");
}

// ===========================================================================
// explained_variance_ratio_ : denom = sum(col_var)·n_total (Pitfall 6)
// ===========================================================================

/// `explained_variance_ratio_` vs sklearn, f32 — the denominator is
/// `sum(col_var)·n_total` (Pitfall 6, NOT the truncated S² sum).
#[test]
fn incremental_pca_explained_variance_ratio_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("incremental_pca_nowhiten_f32_seed42.npz"))
        .expect("load incremental_pca_nowhiten_f32");
    let fit = fit_via_fit::<f32>(&case, false);
    assert_close(
        &fit.explained_variance_ratio,
        case.expect_f64("explained_variance_ratio_"),
        &IPCA_F32_BAND,
        "explained_variance_ratio_ f32",
    );
    let ratio_sum: f64 = fit.explained_variance_ratio.iter().sum();
    assert!(
        ratio_sum <= 1.0 + 1e-4,
        "explained_variance_ratio_ must sum to <= 1, got {ratio_sum}"
    );
}

/// `explained_variance_ratio_` vs sklearn, f64 — denom = `sum(col_var)·n_total`.
#[test]
fn incremental_pca_explained_variance_ratio_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("incremental_pca ev_ratio f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("incremental_pca_nowhiten_f64_seed42.npz"))
        .expect("load incremental_pca_nowhiten_f64");
    let fit = fit_via_fit::<f64>(&case, false);
    assert_close(
        &fit.explained_variance_ratio,
        case.expect_f64("explained_variance_ratio_"),
        &F64_TOL,
        "explained_variance_ratio_ f64",
    );
    let ratio_sum: f64 = fit.explained_variance_ratio.iter().sum();
    assert!(
        ratio_sum <= 1.0 + 1e-5,
        "explained_variance_ratio_ must sum to <= 1, got {ratio_sum}"
    );
}

// ===========================================================================
// n_samples_seen_ accumulation across partial_fit calls (D-03)
// ===========================================================================

/// `n_samples_seen_` accumulates across successive `partial_fit` calls (D-03):
/// 0 before any batch, then the running total after each batch.
#[test]
fn incremental_pca_n_samples_seen_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("incremental_pca n_samples_seen f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("incremental_pca_nowhiten_f64_seed42.npz"))
        .expect("load incremental_pca_nowhiten_f64");
    let x_f64 = case.expect_f64("X").to_vec();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let unfit = IncrementalPCA::<f64>::builder()
        .n_components(IPCA_N_COMPONENTS)
        .whiten(false)
        .batch_size(Some(IPCA_BATCH_SIZE))
        .build::<f64>()
        .expect("IncrementalPcaBuilder::build is infallible");

    // Before any partial_fit, n_samples_seen_ == 0 on the Unfit value (D-03).
    assert_eq!(unfit.n_samples_seen(), 0, "n_samples_seen_ starts at 0");

    // FIRST batch: Unfit -> Fitted (Pitfall 5).
    let first_b = IPCA_BATCH_SIZE.min(IPCA_N);
    let first_host: Vec<f64> = x_f64[0..first_b * IPCA_P].to_vec();
    let first_dev: DeviceArray<ActiveRuntime, f64> =
        DeviceArray::from_host(&mut pool, &first_host);
    let mut pca = unfit
        .partial_fit(&mut pool, &first_dev, None, (first_b, IPCA_P))
        .expect("partial_fit first batch (Unfit -> Fitted)");
    first_dev.release_into(&mut pool);
    let mut expected = first_b;
    assert_eq!(
        pca.n_samples_seen(),
        expected,
        "n_samples_seen_ after the first batch"
    );

    // SUBSEQUENT batches: Fitted -> Fitted, accumulating n_samples_seen_.
    let mut start = first_b;
    while start < IPCA_N {
        let b = IPCA_BATCH_SIZE.min(IPCA_N - start);
        let batch_host: Vec<f64> = x_f64[start * IPCA_P..(start + b) * IPCA_P].to_vec();
        let batch_dev: DeviceArray<ActiveRuntime, f64> =
            DeviceArray::from_host(&mut pool, &batch_host);
        pca = pca
            .partial_fit(&mut pool, &batch_dev, None, (b, IPCA_P))
            .expect("partial_fit batch (Fitted -> Fitted)");
        batch_dev.release_into(&mut pool);
        expected += b;
        assert_eq!(
            pca.n_samples_seen(),
            expected,
            "n_samples_seen_ accumulates after each batch"
        );
        start += b;
    }
    assert_eq!(pca.n_samples_seen(), IPCA_N, "final n_samples_seen_ == n");
}

// ===========================================================================
// transform: (X − mean_)·componentsᵀ, whiten on/off (D-06)
// ===========================================================================

/// `transform(X)` vs sklearn, whiten=False, f32 (after column `align_rows`).
#[test]
fn incremental_pca_transform_nowhiten_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("incremental_pca_nowhiten_f32_seed42.npz"))
        .expect("load incremental_pca_nowhiten_f32");
    let fit = fit_via_fit::<f32>(&case, false);
    assert_transforms(&fit, &case, &IPCA_F32_BAND, "transform nowhiten f32");
}

/// `transform(X)` vs sklearn, whiten=False, f64.
#[test]
fn incremental_pca_transform_nowhiten_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("incremental_pca transform nowhiten f64 backend={backend}: SKIPPED");
        return;
    }
    let case = load_npz(fixture("incremental_pca_nowhiten_f64_seed42.npz"))
        .expect("load incremental_pca_nowhiten_f64");
    let fit = fit_via_fit::<f64>(&case, false);
    assert_transforms(&fit, &case, &F64_TOL, "transform nowhiten f64");
}

/// `transform(X)` vs sklearn WITH whiten=True (components scaled by
/// `1/sqrt(explained_variance_)`, D-06), f32.
#[test]
fn incremental_pca_transform_whiten_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("incremental_pca_whiten_f32_seed42.npz"))
        .expect("load incremental_pca_whiten_f32");
    let fit = fit_via_fit::<f32>(&case, true);
    // The whiten fixture's `transform`/`inverse_transform` are the whitened
    // round-trip; the fitted attrs (components_/ev/sv/mean/var) are identical to
    // the unwhitened fit (whiten only affects transform). Assert both.
    assert_attrs(&fit, &case, &IPCA_F32_BAND, "whiten attrs f32");
    assert_transforms(&fit, &case, &IPCA_F32_BAND, "transform whiten f32");
}

/// `transform(X)` vs sklearn WITH whiten=True, f64.
#[test]
fn incremental_pca_transform_whiten_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("incremental_pca transform whiten f64 backend={backend}: SKIPPED");
        return;
    }
    let case = load_npz(fixture("incremental_pca_whiten_f64_seed42.npz"))
        .expect("load incremental_pca_whiten_f64");
    let fit = fit_via_fit::<f64>(&case, true);
    assert_attrs(&fit, &case, &F64_TOL, "whiten attrs f64");
    assert_transforms(&fit, &case, &F64_TOL, "transform whiten f64");
}

// ===========================================================================
// inverse_transform: un-whiten + reconstruct, whiten on/off (D-06)
// ===========================================================================

/// `inverse_transform(transform(X))` vs sklearn, whiten=False, f64.
#[test]
fn incremental_pca_inverse_transform_nowhiten_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("incremental_pca inverse nowhiten f64 backend={backend}: SKIPPED");
        return;
    }
    let case = load_npz(fixture("incremental_pca_nowhiten_f64_seed42.npz"))
        .expect("load incremental_pca_nowhiten_f64");
    let fit = fit_via_fit::<f64>(&case, false);
    assert_close(
        &fit.inverse,
        case.expect_f64("inverse_transform"),
        &F64_TOL,
        "inverse_transform nowhiten f64",
    );
}

/// `inverse_transform(transform(X))` vs sklearn WITH whiten=True (the inverse
/// un-whitens by multiplying back `sqrt(explained_variance_)` before the
/// reconstruction GEMM, D-06), f64.
#[test]
fn incremental_pca_inverse_transform_whiten_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("incremental_pca inverse whiten f64 backend={backend}: SKIPPED");
        return;
    }
    let case = load_npz(fixture("incremental_pca_whiten_f64_seed42.npz"))
        .expect("load incremental_pca_whiten_f64");
    let fit = fit_via_fit::<f64>(&case, true);
    assert_close(
        &fit.inverse,
        case.expect_f64("inverse_transform"),
        &F64_TOL,
        "inverse_transform whiten f64",
    );
}

/// `inverse_transform(transform(X))` vs sklearn WITH whiten=True, f32.
#[test]
fn incremental_pca_inverse_transform_whiten_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("incremental_pca_whiten_f32_seed42.npz"))
        .expect("load incremental_pca_whiten_f32");
    let fit = fit_via_fit::<f32>(&case, true);
    assert_close(
        &fit.inverse,
        case.expect_f64("inverse_transform"),
        &IPCA_F32_BAND,
        "inverse_transform whiten f32",
    );
}
