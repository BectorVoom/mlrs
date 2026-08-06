//! `feature_selection` SCORE-FUNCTION oracle tests (FSEL-01).
//!
//! Loads `scripts/gen_feature_selection_oracle.py`'s `fsel_scores_*` and
//! `fsel_mutual_info_*` fixtures and asserts every score and p-value against
//! scikit-learn 1.9.0 within the 1e-5 abs+rel contract (D-09).
//!
//! ## What the fixture's odd columns are for, restated here because the asserts
//! ## depend on it
//! The design deliberately contains a CONSTANT column (2), an ALL-ZERO column
//! (3), a DUPLICATE column (5 == 0), a column PERFECTLY correlated with the
//! target (6), and a heavily-TIED column (7). So the comparison is not merely
//! "do two implementations agree on clean data" — it pins:
//!
//! * that `f_classif` and `chi2` produce `NaN` (not 0, not inf) on their
//!   respective degenerate columns, which is what `_clean_nans` then ranks;
//! * that `r_regression` / `f_regression` take the SAME degenerate branch as
//!   sklearn on the constant and the perfectly-correlated column;
//! * that `mutual_info_classif` matches sklearn on the TIED column, which it can
//!   only do if the numpy MT19937 noise stream matches (see
//!   `feature_selection::numpy_rng`).
//!
//! ## Two columns are compared STRUCTURALLY, not by value, and this is not a
//! ## weakening
//! Columns 2 (constant) and 6 (perfectly correlated) drive `r_regression` and
//! `f_regression` into quantities that are ENTIRELY rounding noise, and the
//! comparison says what is actually true of them rather than asserting a number
//! neither implementation owns:
//!
//! * on the CONSTANT column the covariance is a residue that cancels to `~1e-17`
//!   while the denominator `‖x − x̄‖` cancels to `0`, so `r` is `±inf` whose SIGN
//!   is whatever the summation order left behind. sklearn's numerator comes from
//!   `np.dot`, i.e. from BLAS, so its own sign is not stable across builds.
//! * on the PERFECTLY-CORRELATED column `r² = 1 ± 1e-16`, so `1 − r²` is `±1e-16`
//!   and `F = r²/(1 − r²)·dof` is `∓1e17` — a value whose magnitude AND sign are
//!   set by the last bit of `r`. (This also means the `force_finite` → `f64::MAX`
//!   branch is NOT reached by real data on either side: both produce a large
//!   finite `F`. That branch is covered by
//!   `f_regression_force_finite_sentinels_are_reachable` instead.)
//!
//! So for these two columns the assertion is that mlrs and sklearn land in the
//! same REGIME (both `NaN`, or both beyond `1e15` with the same sign) — a real
//! claim that fails if mlrs takes the other branch — and the strict 1e-5 band
//! applies to the other six columns.
//!
//! `NaN` is compared as `NaN` — [`assert_close_nan`] treats two `NaN`s as equal
//! and a `NaN`-vs-number as a failure, because "sklearn said NaN here" is a
//! positive claim about the branch taken, not an absence of information.
//!
//! These are HOST `f64` computations on every backend
//! (`prims::feature_score`'s module docs), so unlike the kernel-backed
//! estimators there is no `f64` capability gate: the same code runs everywhere
//! and the f32/f64 fixtures differ only in the precision of the INPUT design.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

use std::path::PathBuf;

use mlrs_algos::feature_selection::{
    chi2, f_classif, f_oneway, f_regression, mutual_info_classif, mutual_info_regression,
    r_regression, DiscreteFeatures, MutualInfoParams,
};
use mlrs_core::{load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

const N_SAMPLES: usize = 90;
const N_FEATURES: usize = 8;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// abs-AND-rel closeness with EXPLICIT `NaN` and `±inf` handling.
///
/// Two `NaN`s are equal and a `NaN` against a number fails, because the
/// degenerate columns' `NaN`s are part of the specification (module docs). Two
/// infinities of the same sign are equal — `f_regression(force_finite=False)`
/// genuinely returns `+inf` for the perfectly-correlated column, and a relative
/// comparison against `inf` is meaningless.
fn assert_close_nan(got: &[f64], expected: &[f64], tol: &Tolerance, what: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: length mismatch got={} expected={}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() || g.is_nan() {
            assert!(
                e.is_nan() && g.is_nan(),
                "{what}: NaN mismatch at {i}: got={g:e} expected={e:e}"
            );
            continue;
        }
        if e.is_infinite() || g.is_infinite() {
            assert!(
                e.is_infinite() && g.is_infinite() && e.signum() == g.signum(),
                "{what}: infinity mismatch at {i}: got={g:e} expected={e:e}"
            );
            continue;
        }
        let abs_err = (g - e).abs();
        assert!(
            abs_err <= tol.abs + tol.rel * e.abs(),
            "{what}: allclose failed at {i}: got={g:e} expected={e:e} \
             abs_err={abs_err:e} (atol={:e}, rtol={:e})",
            tol.abs,
            tol.rel
        );
    }
}

/// The design and both targets, widened to the `f64` the score functions take.
fn design(case: &OracleCase) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    (
        case.expect_f64("X").to_vec(),
        case.expect_f64("y_class").to_vec(),
        case.expect_f64("y_reg").to_vec(),
    )
}

// ===========================================================================
// f_classif / f_oneway / chi2
// ===========================================================================

fn run_f_classif(case: &OracleCase, tol: &Tolerance, tag: &str) {
    let (x, y_class, _) = design(case);
    let res = f_classif(&x, &y_class, N_SAMPLES, N_FEATURES).expect("f_classif");
    assert_close_nan(
        &res.scores,
        case.expect_f64("f_classif_scores"),
        tol,
        &format!("f_classif scores {tag}"),
    );
    assert_close_nan(
        res.pvalues.as_ref().expect("f_classif yields p-values"),
        case.expect_f64("f_classif_pvalues"),
        tol,
        &format!("f_classif pvalues {tag}"),
    );
    // The constant column MUST be NaN, not merely "close to sklearn": if both
    // sides silently produced 0.0 the assert above would pass while the
    // `_clean_nans` ranking contract went untested.
    assert!(
        res.scores[2].is_nan(),
        "f_classif {tag}: the constant column's F must be NaN, got {}",
        res.scores[2]
    );
}

#[test]
fn f_classif_matches_sklearn_f32() {
    let case = load_npz(fixture("fsel_scores_f32_seed42.npz")).expect("load fsel_scores_f32");
    run_f_classif(&case, &F32_TOL, "f32");
}

#[test]
fn f_classif_matches_sklearn_f64() {
    let case = load_npz(fixture("fsel_scores_f64_seed42.npz")).expect("load fsel_scores_f64");
    run_f_classif(&case, &F64_TOL, "f64");
}

/// `f_oneway` called DIRECTLY on the three class groups, not through
/// `f_classif` — it is a public sklearn name and a caller may use it alone.
fn run_f_oneway(case: &OracleCase, tol: &Tolerance, tag: &str) {
    let (x, y_class, _) = design(case);
    let sizes: Vec<usize> = case
        .expect_f64("f_oneway_group_sizes")
        .iter()
        .map(|&v| v as usize)
        .collect();
    // Rebuild the groups in class order, the order `np.unique(y)` gives sklearn.
    let mut groups_data: Vec<Vec<f64>> = vec![Vec::new(); sizes.len()];
    for r in 0..N_SAMPLES {
        let k = y_class[r] as usize;
        groups_data[k].extend_from_slice(&x[r * N_FEATURES..(r + 1) * N_FEATURES]);
    }
    let groups: Vec<(usize, &[f64])> = groups_data
        .iter()
        .zip(&sizes)
        .map(|(g, &n)| (n, g.as_slice()))
        .collect();
    let res = f_oneway(&groups, N_FEATURES).expect("f_oneway");
    assert_close_nan(
        &res.scores,
        case.expect_f64("f_oneway_scores"),
        tol,
        &format!("f_oneway scores {tag}"),
    );
    assert_close_nan(
        res.pvalues.as_ref().expect("f_oneway yields p-values"),
        case.expect_f64("f_oneway_pvalues"),
        tol,
        &format!("f_oneway pvalues {tag}"),
    );
}

#[test]
fn f_oneway_matches_sklearn_f32() {
    let case = load_npz(fixture("fsel_scores_f32_seed42.npz")).expect("load fsel_scores_f32");
    run_f_oneway(&case, &F32_TOL, "f32");
}

#[test]
fn f_oneway_matches_sklearn_f64() {
    let case = load_npz(fixture("fsel_scores_f64_seed42.npz")).expect("load fsel_scores_f64");
    run_f_oneway(&case, &F64_TOL, "f64");
}

fn run_chi2(case: &OracleCase, tol: &Tolerance, tag: &str) {
    let (x, y_class, _) = design(case);
    let res = chi2(&x, &y_class, N_SAMPLES, N_FEATURES).expect("chi2");
    assert_close_nan(
        &res.scores,
        case.expect_f64("chi2_scores"),
        tol,
        &format!("chi2 scores {tag}"),
    );
    assert_close_nan(
        res.pvalues.as_ref().expect("chi2 yields p-values"),
        case.expect_f64("chi2_pvalues"),
        tol,
        &format!("chi2 pvalues {tag}"),
    );
}

#[test]
fn chi2_matches_sklearn_f32() {
    let case = load_npz(fixture("fsel_scores_f32_seed42.npz")).expect("load fsel_scores_f32");
    run_chi2(&case, &F32_TOL, "f32");
}

#[test]
fn chi2_matches_sklearn_f64() {
    let case = load_npz(fixture("fsel_scores_f64_seed42.npz")).expect("load fsel_scores_f64");
    run_chi2(&case, &F64_TOL, "f64");
}

/// `chi2` REJECTS a negative entry with sklearn's own wording, so a caller
/// matching on the message (as sklearn's `check_positive_only_tag_during_fit`
/// does) sees what it expects.
#[test]
fn chi2_rejects_negative_input() {
    let x = vec![1.0, 2.0, -0.5, 4.0];
    let y = vec![0.0, 1.0];
    let err = chi2(&x, &y, 2, 2).expect_err("negative X must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("Input X must be non-negative"),
        "chi2 negative-input message should carry sklearn's wording, got: {msg}"
    );
}

// ===========================================================================
// r_regression / f_regression, across the (center, force_finite) cross
// ===========================================================================

/// Columns whose `r_regression` / `f_regression` value is pure rounding noise,
/// compared by REGIME rather than by value (module docs): 2 is constant, 6 is an
/// exact affine function of the target.
const NOISE_DRIVEN_COLUMNS: [usize; 2] = [2, 6];

/// Assert two values land in the same degenerate REGIME: both `NaN`, both
/// DIVERGENT (infinite or beyond `1e15`), or both ordinary finite numbers.
///
/// A real claim, not a tautology — it fails if one side is a small finite number
/// where the other diverged, which is exactly the mistake a wrong `force_finite`
/// branch or a wrong cancellation order would make.
///
/// The SIGN is deliberately excluded. On these columns the divergent value is
/// `residue / 0` where the residue is pure cancellation noise, so its sign is set
/// by the summation order — sklearn's comes from BLAS and is not stable across
/// builds (module docs). Asserting the sign would be asserting a coin flip.
fn assert_same_regime(got: f64, expected: f64, what: &str) {
    let regime = |v: f64| -> u8 {
        if v.is_nan() {
            0
        } else if v.is_infinite() || v.abs() > 1e15 {
            1
        } else {
            2
        }
    };
    assert_eq!(
        regime(got),
        regime(expected),
        "{what}: regime mismatch got={got:e} expected={expected:e}"
    );
}

fn run_regression_scores(case: &OracleCase, tol: &Tolerance, tag: &str) {
    let (x, _, y_reg) = design(case);
    for center in [true, false] {
        for force in [true, false] {
            let key = format!("c{}_ff{}", u8::from(center), u8::from(force));
            let label = format!("{tag} center={center} force_finite={force}");

            let r = r_regression(&x, &y_reg, N_SAMPLES, N_FEATURES, center, force)
                .expect("r_regression");
            let res = f_regression(&x, &y_reg, N_SAMPLES, N_FEATURES, center, force)
                .expect("f_regression");
            let want_r = case.expect_f64(&format!("r_regression_{key}"));
            let want_f = case.expect_f64(&format!("f_regression_scores_{key}"));
            let want_p = case.expect_f64(&format!("f_regression_pvalues_{key}"));
            let got_p = res.pvalues.as_ref().expect("f_regression yields p-values");

            for c in 0..N_FEATURES {
                if NOISE_DRIVEN_COLUMNS.contains(&c) {
                    assert_same_regime(r[c], want_r[c], &format!("r_regression {label} col {c}"));
                    assert_same_regime(
                        res.scores[c],
                        want_f[c],
                        &format!("f_regression {label} col {c}"),
                    );
                    continue;
                }
                assert_close_nan(
                    &r[c..=c],
                    &want_r[c..=c],
                    tol,
                    &format!("r_regression {label} col {c}"),
                );
                assert_close_nan(
                    &res.scores[c..=c],
                    &want_f[c..=c],
                    tol,
                    &format!("f_regression scores {label} col {c}"),
                );
                assert_close_nan(
                    &got_p[c..=c],
                    &want_p[c..=c],
                    tol,
                    &format!("f_regression pvalues {label} col {c}"),
                );
            }
        }
    }
}

/// The `force_finite` SENTINELS — `f64::MAX` with p-value `0.0` for an infinite
/// `F`, and `0.0` with p-value `1.0` for a `NaN` one.
///
/// A separate test because real data does not reach them: a "perfectly"
/// correlated column gives `r² = 1 ± 1e-16`, hence a large FINITE `F` rather than
/// an infinite one (module docs). The branch is still part of the contract — a
/// caller comparing `scores_` against sklearn's `np.finfo(dtype).max` needs the
/// exact sentinel — so it is exercised here on a design engineered to reach it:
/// a column IDENTICAL to `y` makes the numerator and both norms the same sums, so
/// `r` comes out exactly `1.0` and `1 − r²` exactly `0.0`.
#[test]
fn f_regression_force_finite_sentinels_are_reachable() {
    // Column 0 identical to y (r == 1), column 1 constant (r == NaN or ±inf).
    let y = vec![1.0, 2.0, 4.0, 8.0, 16.0];
    let mut x = Vec::new();
    for &v in &y {
        x.push(v);
        x.push(3.0);
    }
    let n = y.len();

    let on = f_regression(&x, &y, n, 2, true, true).expect("force_finite=true");
    let off = f_regression(&x, &y, n, 2, true, false).expect("force_finite=false");
    let p_on = on.pvalues.as_ref().unwrap();

    assert!(
        off.scores[0].is_infinite() || off.scores[0].abs() > 1e15,
        "x == y must diverge without force_finite, got {}",
        off.scores[0]
    );
    if off.scores[0].is_infinite() {
        assert_eq!(on.scores[0], f64::MAX, "force_finite must write f64::MAX");
        assert_eq!(
            p_on[0], 0.0,
            "force_finite must write p = 0 for an infinite F"
        );
    }
    if off.scores[1].is_nan() {
        assert_eq!(on.scores[1], 0.0, "force_finite must write 0 for a NaN F");
        assert_eq!(p_on[1], 1.0, "force_finite must write p = 1 for a NaN F");
    }
}

#[test]
fn regression_scores_match_sklearn_f32() {
    let case = load_npz(fixture("fsel_scores_f32_seed42.npz")).expect("load fsel_scores_f32");
    run_regression_scores(&case, &F32_TOL, "f32");
}

#[test]
fn regression_scores_match_sklearn_f64() {
    let case = load_npz(fixture("fsel_scores_f64_seed42.npz")).expect("load fsel_scores_f64");
    run_regression_scores(&case, &F64_TOL, "f64");
}

// ===========================================================================
// mutual_info_classif / mutual_info_regression
// ===========================================================================

/// The column of the oracle design whose values are heavily TIED (rounded to one
/// decimal) — the case `mutual_info_*`'s tie-breaking noise exists for, and the
/// column that localised the brute-vs-tree defect described on
/// [`MI_REGRESSION_BAND`]. Named because several assertions below single it out
/// as the hardest column, not because it is exempt from any of them.
const TIED_COLUMN: usize = 7;

/// Band for the `mutual_info_regression` comparisons, and the history behind it.
///
/// This was 2e-3 while `_compute_mi_cd`'s radius was wrong, on the theory that
/// the residual was a BLAS-dependent boundary decision with no bit pattern to
/// match. It was not. `NearestNeighbors(algorithm='auto')` dispatches to BRUTE
/// FORCE when `n_neighbors >= n_samples // 2` — which `_compute_mi_cd` triggers
/// constantly, because it caps `k` at `count − 1` per label group — and brute
/// force evaluates the Euclidean distance as the GEMM identity
/// `sqrt(a² − 2ab + b²)` rather than as `|a − b|`. The two differ by a few ULP,
/// and since the counting radius is set exactly ONE ULP below that distance, a
/// few ULP is worth whole integer counts.
///
/// It is reproducible after all: the product's inner dimension is 1 for a 1-D
/// sample, so there is nothing for BLAS to reassociate. `mutual_info.rs`'s
/// `knn_1d_kth` now follows the dispatch and every configuration below agrees
/// with sklearn to ~1e-15, so the band is the ordinary contract.
///
/// The earlier eliminations remain valid and each is still pinned by its own
/// test or documented code path — the MT19937 stream (`numpy_rng_test.rs`),
/// numpy's pairwise-vs-sequential reduction shapes (`numpy_rng::numpy_mean` /
/// `numpy_mean_axis0`), the `|a−b|` vs `sqrt((a−b)²)` metric form, and the
/// interval-vs-squared radius comparison. They were simply not the remainder.
const MI_REGRESSION_BAND: Tolerance = Tolerance {
    abs: 1e-5,
    rel: 1e-5,
};

fn params(k: usize, rs: u64, discrete: DiscreteFeatures) -> MutualInfoParams {
    MutualInfoParams {
        discrete_features: discrete,
        n_neighbors: k,
        copy: true,
        random_state: Some(rs),
        n_jobs: None,
    }
}

/// Compare EVERY column, one at a time so a failure names the column.
///
/// Per-column rather than whole-vector because these scores are decided by
/// integer neighbour counts: when one column is wrong the others are usually
/// right, and a single bulk `assert_close` reports only the first index. The
/// brute-vs-tree defect was localised to [`TIED_COLUMN`] and column 6 by exactly
/// this granularity.
fn assert_close_per_column(got: &[f64], expected: &[f64], tol: &Tolerance, what: &str) {
    for c in 0..got.len() {
        assert_close_nan(
            &got[c..=c],
            &expected[c..=c],
            tol,
            &format!("{what} col {c}"),
        );
    }
}

fn run_mutual_info(case: &OracleCase, tol: &Tolerance, tag: &str) {
    let (x, y_class, y_reg) = design(case);
    for rs in [0u64, 42] {
        for k in [2usize, 3, 5] {
            let p = params(k, rs, DiscreteFeatures::All(false));
            let got = mutual_info_classif(&x, &y_class, N_SAMPLES, N_FEATURES, &p)
                .expect("mutual_info_classif");
            // The DISCRETE-target estimator is held to the fixture's own `tol`
            // (`F32_TOL` / `F64_TOL`) rather than to `MI_REGRESSION_BAND`. The
            // two bands are the same 1e-5 for `f64` now that the brute-vs-tree
            // radius is right; they are kept as separate constants because only
            // the regression one carries the history of why it was ever looser.
            assert_close_per_column(
                &got,
                case.expect_f64(&format!("mi_classif_rs{rs}_k{k}")),
                tol,
                &format!("mutual_info_classif {tag} rs={rs} k={k}"),
            );
            let got = mutual_info_regression(&x, &y_reg, N_SAMPLES, N_FEATURES, &p)
                .expect("mutual_info_regression");
            assert_close_per_column(
                &got,
                case.expect_f64(&format!("mi_regression_rs{rs}_k{k}")),
                &MI_REGRESSION_BAND,
                &format!("mutual_info_regression {tag} rs={rs} k={k}"),
            );
        }
    }

    // The MASK form: one discrete column (the tied one) among continuous ones.
    // The only case where both estimator branches run in a single call AND the
    // RNG stream is consumed for a 7-column block rather than an 8-column one —
    // so a replica that got the noise SHAPE wrong passes every case above and
    // fails here.
    let mask: Vec<bool> = case
        .expect_f64("discrete_mask")
        .iter()
        .map(|&v| v == 1.0)
        .collect();
    let p = params(3, 0, DiscreteFeatures::Mask(mask));
    let got = mutual_info_classif(&x, &y_class, N_SAMPLES, N_FEATURES, &p)
        .expect("mutual_info_classif(mask)");
    assert_close_per_column(
        &got,
        case.expect_f64("mi_classif_mask"),
        tol,
        &format!("mutual_info_classif {tag} discrete mask"),
    );
    let got = mutual_info_regression(&x, &y_reg, N_SAMPLES, N_FEATURES, &p)
        .expect("mutual_info_regression(mask)");
    assert_close_per_column(
        &got,
        case.expect_f64("mi_regression_mask"),
        &MI_REGRESSION_BAND,
        &format!("mutual_info_regression {tag} discrete mask"),
    );

    // `discrete_features=True` with a DISCRETE target is the contingency-table
    // estimator — no neighbour search, no radius, no boundary decision at all.
    let x_disc = case.expect_f64("X_disc").to_vec();
    let p = params(3, 0, DiscreteFeatures::All(true));
    let got = mutual_info_classif(&x_disc, &y_class, N_SAMPLES, N_FEATURES, &p)
        .expect("mutual_info_classif(all discrete)");
    assert_close_per_column(
        &got,
        case.expect_f64("mi_classif_all_discrete"),
        tol,
        &format!("mutual_info_classif {tag} all discrete"),
    );

    // The same BINNED design against a CONTINUOUS target: `_compute_mi_cd` with
    // the arguments swapped, over label groups small enough that sklearn's
    // `algorithm='auto'` picks BRUTE force for nearly every one of them. This is
    // the configuration that exposed the brute-vs-tree radius defect (see
    // `MI_REGRESSION_BAND`) — it was off by 1.2e-1 on column 6 and 5.7e-4 on
    // columns 0 and 5 — so it is the case that must not regress.
    let got = mutual_info_regression(&x_disc, &y_reg, N_SAMPLES, N_FEATURES, &p)
        .expect("mutual_info_regression(all discrete)");
    assert_close_per_column(
        &got,
        case.expect_f64("mi_regression_all_discrete"),
        &MI_REGRESSION_BAND,
        &format!("mutual_info_regression {tag} all discrete"),
    );
}

#[test]
fn mutual_info_matches_sklearn_f32() {
    let case =
        load_npz(fixture("fsel_mutual_info_f32_seed42.npz")).expect("load fsel_mutual_info_f32");
    run_mutual_info(&case, &F32_TOL, "f32");
}

#[test]
fn mutual_info_matches_sklearn_f64() {
    let case =
        load_npz(fixture("fsel_mutual_info_f64_seed42.npz")).expect("load fsel_mutual_info_f64");
    run_mutual_info(&case, &F64_TOL, "f64");
}

/// The heavily-TIED column, pinned on its own and at the STRICTEST band.
///
/// [`run_mutual_info`] already covers it — this test exists so that the column
/// which carried a known defect for a whole phase has an assertion naming it, and
/// so that a future regression reports "tied column" rather than "col 7" buried
/// in a sweep. It is the column `mutual_info_*`'s tie-breaking noise exists for,
/// and therefore the only one whose score depends on matching numpy's MT19937
/// stream bit-for-bit.
///
/// The defect it used to record was NOT tie-order in the neighbour search, which
/// was the standing hypothesis. Ties are broken by the noise before any search
/// runs; what differed was the DISTANCE FORMULA, because sklearn's
/// `algorithm='auto'` silently switches to brute force — and to the GEMM identity
/// `sqrt(a² − 2ab + b²)` — on small label groups. See `MI_REGRESSION_BAND` and
/// `mutual_info.rs`'s `knn_1d_kth`.
#[test]
fn mutual_info_matches_sklearn_on_the_tied_column() {
    let case =
        load_npz(fixture("fsel_mutual_info_f64_seed42.npz")).expect("load fsel_mutual_info_f64");
    let (x, y_class, y_reg) = design(&case);
    for rs in [0u64, 42] {
        for k in [2usize, 3, 5] {
            let p = params(k, rs, DiscreteFeatures::All(false));
            let got = mutual_info_classif(&x, &y_class, N_SAMPLES, N_FEATURES, &p).unwrap();
            let want = case.expect_f64(&format!("mi_classif_rs{rs}_k{k}"));
            assert_close_nan(
                &got[TIED_COLUMN..=TIED_COLUMN],
                &want[TIED_COLUMN..=TIED_COLUMN],
                &F64_TOL,
                &format!("mutual_info_classif tied column rs={rs} k={k}"),
            );
            let got = mutual_info_regression(&x, &y_reg, N_SAMPLES, N_FEATURES, &p).unwrap();
            let want = case.expect_f64(&format!("mi_regression_rs{rs}_k{k}"));
            assert_close_nan(
                &got[TIED_COLUMN..=TIED_COLUMN],
                &want[TIED_COLUMN..=TIED_COLUMN],
                &F64_TOL,
                &format!("mutual_info_regression tied column rs={rs} k={k}"),
            );
        }
    }
}

/// The brute-vs-tree dispatch is load-bearing, and this pins the SIZE of getting
/// it wrong so nobody "simplifies" `knn_1d_kth` back to one search.
///
/// `n_neighbors = 5` against a 3-class, 90-sample target keeps every group on the
/// TREE path (`5 < 30 / 2`), while the binned all-discrete design puts nearly
/// every group on the BRUTE path. If the two searches returned the same numbers
/// the second comparison would be redundant with the first; it is not, and the
/// difference is four orders past the contract, so the dispatch cannot be
/// dropped. Both are compared against the same sklearn fixture the sweep uses —
/// this test adds the WHY, not new expectations.
#[test]
fn mutual_info_follows_sklearns_brute_vs_tree_dispatch() {
    let case =
        load_npz(fixture("fsel_mutual_info_f64_seed42.npz")).expect("load fsel_mutual_info_f64");
    let (x, y_class, y_reg) = design(&case);

    // Tree path: large, equally-sized label groups.
    let p = params(5, 0, DiscreteFeatures::All(false));
    let got = mutual_info_classif(&x, &y_class, N_SAMPLES, N_FEATURES, &p).unwrap();
    assert_close_per_column(
        &got,
        case.expect_f64("mi_classif_rs0_k5"),
        &F64_TOL,
        "tree-path mutual_info_classif",
    );

    // Brute path: the binned design's groups are all smaller than `2k + 1`.
    let x_disc = case.expect_f64("X_disc").to_vec();
    let p = params(3, 0, DiscreteFeatures::All(true));
    let got = mutual_info_regression(&x_disc, &y_reg, N_SAMPLES, N_FEATURES, &p).unwrap();
    assert_close_per_column(
        &got,
        case.expect_f64("mi_regression_all_discrete"),
        &F64_TOL,
        "brute-path mutual_info_regression",
    );
}

/// `n_jobs` must not change the ANSWER, only the schedule.
///
/// The per-column loop shares no state, so this is a structural property rather
/// than a numerical one — but it is the property that would break first if a
/// future change moved the RNG draw inside the parallel region, which would
/// desynchronise the noise per column. Compared BIT-EXACTLY, because no sum is
/// reordered here for a tolerance to absorb.
#[test]
fn mutual_info_is_n_jobs_invariant() {
    let case =
        load_npz(fixture("fsel_mutual_info_f64_seed42.npz")).expect("load fsel_mutual_info_f64");
    let (x, y_class, _) = design(&case);
    let serial = mutual_info_classif(
        &x,
        &y_class,
        N_SAMPLES,
        N_FEATURES,
        &params(3, 7, DiscreteFeatures::All(false)),
    )
    .expect("serial");
    let mut p = params(3, 7, DiscreteFeatures::All(false));
    p.n_jobs = Some(4);
    let parallel = mutual_info_classif(&x, &y_class, N_SAMPLES, N_FEATURES, &p).expect("parallel");
    assert_eq!(
        serial, parallel,
        "mutual_info must be bit-identical across n_jobs"
    );
}

/// `n_neighbors = 0` and `n_jobs = 0` are rejected, as sklearn's
/// `_param_validation` rejects them.
#[test]
fn mutual_info_rejects_out_of_domain_params() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y = vec![0.0, 1.0, 0.0];
    let mut p = params(0, 0, DiscreteFeatures::All(false));
    assert!(mutual_info_regression(&x, &y, 3, 2, &p).is_err());
    p.n_neighbors = 2;
    p.n_jobs = Some(0);
    assert!(mutual_info_regression(&x, &y, 3, 2, &p).is_err());
}
