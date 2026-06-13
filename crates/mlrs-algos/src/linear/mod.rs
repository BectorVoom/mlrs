//! `linear` — linear models (LINEAR-01 .. LINEAR-04).
//!
//! Module index for the Phase-4/5 linear estimators. They deliberately use
//! DIFFERENT solvers and must not be unified (RESEARCH Anti-Patterns):
//!
//! - `LinearRegression` (LINEAR-01) — **SVD pseudo-inverse**
//!   `coef = V·diag(σ⁺)·Uᵀ·y` with sklearn's small-singular-value cutoff,
//!   matching sklearn's default `lstsq` (D-02). Added by plan **04-03**.
//! - `Ridge` (LINEAR-02) — **Cholesky normal-equations**
//!   `(XᵀX + αI)·coef = Xᵀy` via the new Cholesky/solve primitive (D-02). α
//!   never penalizes the intercept (center-then-solve, D-05). Added by plan
//!   **04-05**.
//! - `Lasso` (LINEAR-03) + `ElasticNet` (LINEAR-04) — **coordinate descent**
//!   (the iterative-solver family). Both share ONE coordinate-descent helper
//!   ([`coordinate_descent::cd_fit`]) built on the 05-05 `cd_solve` primitive:
//!   Lasso is ElasticNet with `l1_ratio == 1` (→ `l2_reg = 0`, pure L1, D-03).
//!   They map the user-facing `(alpha, l1_ratio)` to sklearn's un-normalized
//!   `(l1_reg = α·l1_ratio·n, l2_reg = α·(1−l1_ratio)·n)` and recover the
//!   unpenalized `intercept_ = ȳ − x̄·coef_` by center-then-solve (D-13). Added
//!   by plan **05-09**. This CD path is NOT unified with the L-BFGS
//!   `LogisticRegression` solver (05-10) — a different optimizer for a different
//!   objective.
//! - `LogisticRegression` (LINEAR-05) — **L-BFGS** over the symmetric
//!   over-parameterized multinomial softmax objective (`l2_reg = 1/(C·n)`,
//!   intercept unpenalized — Pitfall 3; K full weight vectors so binary is the
//!   K=2 case, D-12) on the validated 05-06 `lbfgs_minimize` primitive. The
//!   oracle gates on the gauge-invariant `predict`/`predict_proba` (PRIMARY,
//!   1e-5; `coef_` looser secondary — Pitfall 5 gauge freedom). Added by plan
//!   **05-10**. Deliberately NOT the coordinate-descent solver above (D-03).
//!
//! The estimator plans UNCOMMENT/add their own `pub mod <estimator>;` line here
//! and create the matching file; they do NOT edit `lib.rs` (owned by 04-01),
//! keeping the estimator plans file-disjoint and parallel-safe.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

pub mod coordinate_descent;
pub mod elastic_net;
pub mod lasso;
pub mod linear_regression;
pub mod logistic;
pub mod ridge;
