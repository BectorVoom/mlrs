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
//! - `BayesianRidge` (LINEAR-06) — **evidence maximization** (MacKay 1992) over
//!   the ONE-TIME symmetric eigendecomposition of the Gram
//!   ([`sym_eig`], a Householder+QL reduction, NOT the cold-path Jacobi sweep
//!   `ridge_solvers` carries). Deliberately not unified with `Ridge`: the
//!   penalty here is not a hyperparameter but a fitted quantity, re-estimated
//!   each iteration alongside the noise precision. The whole iteration runs in
//!   the eigenbasis at `O(d)` per step — including the residual `‖y − Xw‖²`,
//!   which sklearn recomputes with an `O(n·d)` pass — so `n_samples` leaves the
//!   loop entirely. Added by plan **06-01**.
//!
//! - `HuberRegressor` (HUBER-01) — **L-BFGS over `[w, c, σ]`** with a `σ > 0`
//!   barrier, minimizing the jointly-convex perspective form of the Huber loss.
//!   Deliberately NOT unified with `LinearSVR`, whose squared-epsilon-insensitive
//!   primal has a FIXED tube width: here the scale is a fitted parameter, so the
//!   per-sample loss is not a function of the margin alone and the objective
//!   needs three extra `O(n)` reductions per evaluation for `∂L/∂σ` (its own
//!   [`huber_objective`](mlrs_backend::prims::huber_objective) prim, sharing the
//!   `svm_objective` cpu-host/device-GEMM split but not its evaluator). Added by
//!   plan **HUBER-01**.
//!
//! - `RidgeCV` (RIDGECV-01) — **generalized (leave-one-out) cross-validation**
//!   in closed form off ONE symmetric eigendecomposition
//!   ([`ridge_cv::ridge_gcv`]), plus an explicit `GridSearchCV` arm for a
//!   user-supplied `cv` ([`ridge_cv::ridge_cv_grid`]). Deliberately NOT a loop
//!   over [`ridge::Ridge`]: the whole point is that the Gram, its
//!   eigendecomposition and the eigenbasis projection of the design are formed
//!   ONCE and shared by every `alpha`, which drops sklearn's `O(n_alphas·n·d²)`
//!   default route to `O(n·d²) + O(n_alphas·n·d)`.
//!
//! The estimator plans UNCOMMENT/add their own `pub mod <estimator>;` line here
//! and create the matching file; they do NOT edit `lib.rs` (owned by 04-01),
//! keeping the estimator plans file-disjoint and parallel-safe.
//!
//! ## Persistence (LINEAR-PERSIST, prototype)
//!
//! [`linear_persist`] pins the `mlrs-linear` discriminator on the shared
//! [`persist`](crate::persist) safetensors container and owns the dense-linear
//! core — one row-major `[n_targets, n_features]` `coef_` tensor plus an
//! `[n_targets]` `intercept_`, with every constructor scalar in `__metadata__`
//! and nothing derivable written twice. Each estimator implements
//! [`SaveModel`](crate::persist::SaveModel) /
//! [`LoadModel`](crate::persist::LoadModel) in its OWN file against that core,
//! which is what keeps the fitted fields private — the same shape
//! [`naive_bayes`](crate::naive_bayes) uses, and the reason neither family
//! needed a `Serialize` derive across every estimator struct.
//!
//! Four estimators are wired, and between them they cover every shape the
//! format has to handle:
//!
//! | estimator | what it adds to the core |
//! |---|---|
//! | [`LinearRegression`](linear_regression::LinearRegression) | nothing — its whole fitted state IS the core |
//! | [`Ridge`](ridge::Ridge) | eight hyperparameters (two `Option`s, two enums), three fitted diagnostics (`n_iter_`/`solver_`/`device_`), multi-target `coef_` |
//! | [`Lasso`](lasso::Lasso) | the shared [`CdScalars`](linear_persist::CdScalars) |
//! | [`ElasticNet`](elastic_net::ElasticNet) | those plus `param:l1_ratio` |
//!
//! Two of those are worth knowing about before adding the fifth. Ridge holds
//! `coef_` FEATURES-major for its fused predict GEMM while the file stores
//! sklearn's TARGETS-major orientation, which
//! [`linear_persist::to_targets_major`] reconciles at the boundary — a borrow,
//! not a copy, for every single-target model. And `Lasso`/`ElasticNet` are near
//! duplicates on disk (`Lasso` IS `ElasticNet` at `l1_ratio == 1`), which is
//! what makes the `estimator` discriminator load bearing: it is the only thing
//! stopping an `ElasticNet` file loading as a `Lasso` that quietly drops the L2
//! half of the penalty. Whatever the remaining members need is a subset of
//! these four.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

pub mod linear_persist;

pub mod bayesian_ridge;
pub mod coordinate_descent;
pub mod huber;
pub mod elastic_net;
pub mod lasso;
pub mod linear_regression;
pub mod logistic;
// `RANSACRegressor` (RANSAC-01) — the outlier-EXCLUDING robust regressor, the
// counterpart to `HuberRegressor`'s outlier-DOWNWEIGHTING one. Deliberately not
// unified with it: there is no objective to minimize here, only a consensus
// search over random sub-samples, and the base model it refits is an ordinary
// least-squares solve. The compute engine is
// `mlrs_backend::prims::ransac_host` — host-resident on EVERY backend, because
// a trial is a launch-bound `n × d` pass whose result the NEXT draw's stopping
// rule must read back (see that module's docs).
pub mod ransac;
pub mod ridge;
pub mod ridge_classifier;
pub mod ridge_cv;
pub mod ridge_solvers;
pub mod sym_eig;

// Phase-10 SGD / linear-SVM (SGDSVM-01..04). This index lands the shared
// `sgd_config` (typed Loss/Penalty/LearningRate enums + `SgdConfig` lowering
// target, D-04/D-06) and the four builder-fronted estimator homes.
// `MBSGDClassifier`/`MBSGDRegressor` are minibatch-SGD models (the `sgd_solve`
// prim, PRIM-10); `LinearSVC`/`LinearSVR` solve the L2-regularized squared-hinge /
// squared-epsilon-insensitive PRIMAL via the validated L-BFGS primitive
// (Open-Q1 resolution — NOT the coordinate-descent solver; the SVM objective is
// smooth+convex but not the Lasso/ElasticNet soft-threshold CD objective). Each
// estimator is constructed via its `*Builder` + `build() -> Result<_, BuildError>`
// (D-01 — Phase-10 INTRODUCES the builder pattern; existing low-arity estimators
// are NOT retrofitted, D-02).
pub mod sgd_config;
pub mod mbsgd_classifier;
pub mod mbsgd_regressor;
pub mod linear_svc;
pub mod linear_svr;
