#!/usr/bin/env python3
"""Seeded NumPy oracle-fixture generator for mlrs (build-time only, D-03).

This script regenerates the committed ``.npz`` oracle blobs that the Rust test
suite reads with **no Python in the loop** (D-03). It is the *canonical*
regeneration tool: ``numpy.random.default_rng(seed)`` is the authoritative
seeded RNG (avoid Rust-side RNG, RESEARCH Pitfall 7), and the committed blobs
are checked in so CI never runs this script.

Phase 1 emits the saxpy smoke case only. Phase 4 extends this module with the
estimator/primitive fixtures: ``gen_cholesky`` (scipy SPD solve + L factor),
``gen_linear_regression`` / ``gen_ridge`` (sklearn ``coef_``/``intercept_``),
``gen_pca`` / ``gen_truncated_svd`` (sklearn fitted decomposition attributes),
all under the ``case_dtype_seed`` naming convention (D-01/D-02/D-07). These need
``scipy`` + ``scikit-learn`` in addition to ``numpy`` — regen in a /tmp venv
(PEP 668): ``python3 -m venv /tmp/oracle-venv &&
/tmp/oracle-venv/bin/pip install numpy scipy scikit-learn``. The committed blobs
are checked in; CI never runs this script.

Fixture contract (consumed by ``mlrs_core::oracle::load_npz``):
  - named arrays ``a`` / ``x`` / ``y`` / ``expected``
  - ``a`` is the scalar multiplier, ``x`` / ``y`` the input vectors,
    ``expected = a * x + y`` — every array cast to the fixture's dtype.
  - file name encodes ``case_dtype_seed`` (e.g. ``saxpy_f32_seed42.npz``).

Run:
    python3 scripts/gen_oracle.py
Requires only ``numpy`` (sklearn is NOT needed for the saxpy fixture; it
arrives with the Phase-4 estimator fixtures).
"""

from __future__ import annotations

import os

import numpy as np

# Resolve the repo root from this file's location so the script is runnable
# from any working directory and always writes to ``<repo>/tests/fixtures``.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIXTURE_DIR = os.path.join(_REPO_ROOT, "tests", "fixtures")

# Seed and problem size are fixed so the committed blobs are byte-reproducible.
SEED = 42
N = 1024
# Scalar multiplier for the saxpy case. Chosen non-integer so the f32/f64 paths
# exercise real rounding rather than exact integer arithmetic.
A = 2.5

# GEMM convention-fixture shape (D-12). Small NON-square so the fixture also
# exercises rectangular geometry: A is m×k, B is k×n, C = A @ B is m×n.
GEMM_M, GEMM_K, GEMM_N = 5, 4, 3

# Distance convention-fixture shape (D-12, PRIM-03). X is rows_x×cols, Y is
# rows_y×cols; the pairwise squared distance D is rows_x×rows_y. Non-square so
# the fixture exercises rectangular geometry and rows_x != rows_y.
DIST_ROWS_X, DIST_ROWS_Y, DIST_COLS = 5, 4, 3

# Covariance convention-fixture shape (D-12, PRIM-04). A is
# n_samples×n_features (observations in rows, features in columns — the
# ``rowvar=False`` convention); the covariance C is n_features×n_features.
# n_samples > n_features and non-square so the fixture exercises a realistic
# rectangular data matrix and ddof actually changes the normalisation.
COV_N_SAMPLES, COV_N_FEATURES = 7, 4

# SVD convention-fixture shapes (D-05, PRIM-05). SVD_TALL is m≥n (the standard
# thin-SVD orientation); SVD_WIDE is m<n so the fixture exercises the Aᵀ-swap
# path (run Jacobi on Aᵀ then swap U↔V, D-05). Small + non-square so geometry is
# realistic without being a stress test.
SVD_TALL = (8, 4)
SVD_WIDE = (4, 8)
# SVD_TALL_ODD has an ODD thin dimension (k = min(m, n) = 5) to pin the
# circle-method round-robin schedule for odd `cols` (CR-01 — the even-only
# schedule silently omitted ~half the column pairs for odd k, returning a
# wrong/non-orthonormal factorization). 9×5 keeps the fixture tiny while
# exercising the ghost-padded odd-parity pairing.
SVD_TALL_ODD = (9, 5)

# Symmetric-eig convention-fixture size (D-06, PRIM-05). EIG_N is the order of
# the square symmetric matrix the eig primitive decomposes; small so the
# committed fixture stays tiny while still exercising sort/sign handling.
EIG_N = 4

# ---- Phase-7 covariance & projection fixture sizes ----
# EmpiricalCovariance (COV-01). Two cases: a well-conditioned full-rank case
# (n > p) and a RANK-DEFICIENT case (n <= p) so the `precision_ = pinvh(cov)`
# floor (eig-based pseudo-inverse, NOT Cholesky — must tolerate a singular
# covariance, D-05) is actually exercised. p <= 64 keeps the symmetric-eig
# `precision_` path inside the Phase-3 MAX_DIM cap.
EMPCOV_FULLRANK = (16, 5)   # n=16 > p=5
EMPCOV_RANKDEF = (4, 6)     # n=4 <= p=6 → covariance is singular (rank <= 4)

# LedoitWolf (COV-02). TWO sample counts per ROADMAP criterion 3 so the
# shrinkage_ closed form is pinned across n; p <= 64.
LW_N_SMALL, LW_N_LARGE, LW_P = 12, 40, 5

# IncrementalPCA (DECOMP-03). Sized so the per-batch STACKED matrix clears the
# Phase-3 SVD caps: the merge stacks `n_components` running-basis rows + a
# `batch_size` batch + 1 mean-correction row, so `n_components + batch_size + 1`
# must be <= MAX_ROWS (256) and `n_features` <= MAX_COLS (64) (RESEARCH A2 /
# Open Q3). 30 samples, 6 features, n_components=3, batch_size=10 →
# 3 + 10 + 1 = 14 <= 256 and 6 <= 64.
IPCA_SHAPE = (30, 6)
IPCA_N_COMPONENTS = 3
IPCA_BATCH_SIZE = 10

# johnson_lindenstrauss_min_dim (PROJ-01/02, D-12 — the ONE RandomProjection
# value oracle). A small (n_samples, eps) grid; eps strictly in (0, 1).
JL_N_SAMPLES = (100, 1000, 10000)
JL_EPS = (0.1, 0.2, 0.5)

# ---- Phase-8 kernel-family fixture sizes ----
# kernel_matrix (PRIM-08, D-01/D-02). Small NON-square X/Y sharing a feature
# dimension so the fixture pins the general K(X, Y) (rows_x × rows_y) for all
# four kernels (linear/rbf/poly/sigmoid).
KM_ROWS_X, KM_ROWS_Y, KM_COLS = 5, 4, 3
# KernelRidge (KERNEL-01, D-04/D-05). n_samples <= 64 (A2 — the n×n training
# Gram clears the Phase-3/4 MAX_DIM cap so the dual Cholesky solve stays in
# range). A handful of test rows + a 2-target multi-RHS case (D-04).
KR_N_SAMPLES, KR_N_FEATURES, KR_N_TEST = 12, 4, 5
# KernelDensity (KERNEL-02, D-10). Tiny n so the brute-force density matches
# sklearn's exact-forced (atol=0, rtol=0) tree; a small query set Q.
KD_N_SAMPLES, KD_N_FEATURES, KD_N_QUERY = 10, 3, 6
# Spectral family (PRIM-09 / SPECTRAL-01/02). n_samples <= 64 (D-05 — the n×n
# Laplacian clears the v1 eig MAX_DIM=64 cap). SE_N_FEATURES is chosen so the
# `gamma=None -> 1/n_features` default (D-04) is a non-trivial value the oracle
# exercises. SE_N_COMPONENTS=2 is the sklearn default (D-08). SC clusters are
# WELL-SEPARATED (D-10) so the partition is unique up to permutation.
LAP_N = 8
SE_N_SAMPLES, SE_N_FEATURES, SE_N_COMPONENTS = 12, 5, 2
SC_N_SAMPLES, SC_N_FEATURES, SC_N_CLUSTERS = 12, 2, 3


def gen_saxpy(seed: int = SEED, n: int = N, dtype=np.float32) -> str:
    """Generate one seeded saxpy fixture and write it to ``tests/fixtures``.

    Returns the absolute path of the written ``.npz``.
    """
    rng = np.random.default_rng(seed)
    # ``a`` as a 1-element array (not a 0-d scalar) so the named-array reader
    # decodes it to a single-element slice unambiguously.
    a = np.asarray([A], dtype=dtype)
    x = rng.standard_normal(n).astype(dtype)
    y = rng.standard_normal(n).astype(dtype)
    expected = (a[0] * x + y).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"saxpy_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(out_path, a=a, x=x, y=y, expected=expected)
    return out_path


def gen_gemm(seed: int = SEED, dtype=np.float32) -> str:
    """Generate one seeded GEMM convention fixture (D-12, PRIM-01).

    Stores named arrays ``A`` (m×k), ``B`` (k×n) and the NumPy reference product
    ``C = A @ B`` (m×n), every array cast to the fixture's dtype. The shape is
    small and non-square (``GEMM_M``×``GEMM_K``×``GEMM_N``) so the fixture also
    exercises rectangular geometry. Returns the absolute path written.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((GEMM_M, GEMM_K)).astype(dtype)
    b = rng.standard_normal((GEMM_K, GEMM_N)).astype(dtype)
    # Reference product. Compute in the fixture dtype so the committed C matches
    # what a same-dtype device GEMM should produce (the loader exposes both an
    # f32 and an f64 view, so the Rust test compares at the dtype under test).
    c = (a @ b).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"gemm_{dtype_tag}_seed{seed}.npz")
    np.savez(out_path, A=a, B=b, C=c)
    return out_path


def gen_distance(seed: int = SEED, dtype=np.float32, sqrt: bool = False) -> str:
    """Generate one seeded pairwise-distance convention fixture (D-12, PRIM-03).

    Stores named arrays ``X`` (rows_x×cols), ``Y`` (rows_y×cols) and the NumPy
    reference pairwise distance ``D`` (rows_x×rows_y), every array cast to the
    fixture's dtype. ``D[i,j] = sum_k (X[i,k] - Y[j,k])**2`` (the SQUARED
    Euclidean distance); when ``sqrt`` is set, ``D = sqrt(squared)`` (the
    optional Euclidean boundary, D-08).

    The reference is computed the direct way (``(X[:,None,:] - Y[None,:,:])**2``
    summed over the feature axis) rather than the GEMM-expansion the device
    uses, so the fixture is an INDEPENDENT oracle of the expansion identity, not
    a tautology. Returns the absolute path written.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((DIST_ROWS_X, DIST_COLS)).astype(dtype)
    y = rng.standard_normal((DIST_ROWS_Y, DIST_COLS)).astype(dtype)
    # Direct squared pairwise distance (compute in fixture dtype to match a
    # same-dtype device result): broadcast over the feature axis.
    diff = x[:, None, :].astype(dtype) - y[None, :, :].astype(dtype)
    sq = (diff * diff).sum(axis=2).astype(dtype)
    d = np.sqrt(sq).astype(dtype) if sqrt else sq

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    kind = "sqrt" if sqrt else "sq"
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"dist_{kind}_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(out_path, X=x, Y=y, D=d)
    return out_path


def gen_covariance(seed: int = SEED, dtype=np.float32, ddof: int = 1) -> str:
    """Generate one seeded covariance convention fixture (D-12, PRIM-04).

    Stores named arrays ``A`` (n_samples×n_features) and the NumPy reference
    covariance ``C`` (n_features×n_features), every array cast to the fixture's
    dtype. The reference is ``np.cov(A, rowvar=False, ddof=ddof)``:

      - ``rowvar=False`` so the FEATURES are the columns of ``A`` (matching the
        host API's ``(n_samples, n_features)`` row-major contract — observations
        in rows). This pins exactly the convention PCA + the linear closed-form
        solvers inherit.
      - ``ddof=0`` is the population normalisation (divide by ``n``); ``ddof=1``
        is the sample normalisation (divide by ``n − 1``). Both are emitted so
        the device covariance is pinned for BOTH conventions (D-12).

    ``np.cov`` centres each column by its mean before forming ``AᵀA`` and then
    divides by ``n − ddof`` — exactly the device pipeline (column-mean centring →
    ``AᵀA`` via GEMM(transa) → ``1/(n−ddof)`` scale). The fixture is therefore
    the authoritative normalisation oracle, not a tautology of the device
    algebra. Returns the absolute path written.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((COV_N_SAMPLES, COV_N_FEATURES)).astype(dtype)
    # rowvar=False: variables (features) are the COLUMNS of A. Compute in the
    # fixture dtype so the committed C matches a same-dtype device covariance.
    c = np.cov(a, rowvar=False, ddof=ddof).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"cov_ddof{ddof}_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(out_path, A=a, C=c)
    return out_path


def gen_argmin_tie(seed: int = SEED) -> str:
    """Generate the deliberate-tie argmin convention fixture (D-02, PRIM-02).

    Emits a small 2D ``int32`` matrix that contains, by construction, at least
    one TIED minimum per row AND a tied global minimum, so the device argmin's
    lowest-index tie-break can be pinned against numpy's ``argmin`` (which also
    returns the lowest index on ties). Named arrays:

      - ``X``            the ``rows × cols`` int32 input matrix.
      - ``argmin_full``  scalar (length-1) numpy ``X.argmin()`` over the flat
                         row-major buffer — the lowest flat index of the global
                         minimum.
      - ``argmin_rows``  length-``rows`` numpy ``X.argmin(axis=1)`` — the lowest
                         column index of each row's minimum.

    The matrix is integer-VALUED but stored as ``float64`` so the existing
    oracle loader (``mlrs_core::oracle::load_npz``, which decodes only 4-/8-byte
    FLOAT dtypes) reads it directly; the integer index references are likewise
    stored as ``float64`` (every index is exactly representable). The ``i32`` in
    the file name records the integer-valued nature of the source data, not its
    on-disk dtype. Returns the absolute path written.
    """
    rng = np.random.default_rng(seed)
    rows, cols = 4, 6
    # Random small integers, then deliberately PLANT ties on the minimum so the
    # tie-break is actually exercised (not just incidentally hit).
    x = rng.integers(low=0, high=9, size=(rows, cols)).astype(np.float64)
    # Row 0: tie the minimum at columns 1 and 4 (lowest index 1 must win).
    x[0, :] = np.array([5, 1, 7, 3, 1, 8], dtype=np.float64)
    # Row 1: tie the minimum at columns 0 and 2.
    x[1, :] = np.array([2, 6, 2, 9, 4, 7], dtype=np.float64)
    # Row 2: a clear single minimum at column 3 (control row).
    x[2, :] = np.array([6, 5, 8, 0, 7, 9], dtype=np.float64)
    # Row 3: tie the minimum at columns 2 and 5.
    x[3, :] = np.array([4, 7, 1, 6, 8, 1], dtype=np.float64)

    flat = x.reshape(-1)
    argmin_full = np.asarray([float(flat.argmin())], dtype=np.float64)
    argmin_rows = x.argmin(axis=1).astype(np.float64)

    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"argmin_tie_i32_seed{seed}.npz"
    )
    np.savez(
        out_path,
        X=x,
        argmin_full=argmin_full,
        argmin_rows=argmin_rows,
    )
    return out_path


def gen_svd(
    seed: int = SEED,
    dtype=np.float32,
    shape: tuple[int, int] = SVD_TALL,
    kind: str = "tall",
) -> str:
    """Generate one seeded thin-SVD convention fixture (D-05/D-09, PRIM-05).

    Stores named arrays ``A`` (``shape``), ``U``, ``S``, ``Vt`` — the NumPy
    reference thin SVD ``U, S, Vt = np.linalg.svd(A, full_matrices=False)``
    (D-02: ``full_matrices=False`` so ``U`` is ``m×k`` and ``Vt`` is ``k×n`` with
    ``k = min(m, n)``). ``np.linalg.svd`` ALWAYS returns the singular values in
    DESCENDING order (D-04), so the fixture stores them as-is; the Rust test
    sign-aligns ``U``/``Vt`` rows with ``align_rows`` before comparing (D-03 —
    singular vectors are only defined up to a sign). Every array is cast to the
    fixture's dtype so the committed reference matches a same-dtype device SVD.

    The file name encodes ``svd_{kind}_{dtype}_seed{seed}``; ``kind`` is ``tall``
    (m≥n, the thin orientation) or ``wide`` (m<n, the Aᵀ-swap path, D-05).
    Returns the absolute path written.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape).astype(dtype)
    # Thin SVD (full_matrices=False, D-02): U is m×k, S is length-k descending,
    # Vt is k×n with k = min(m, n). Compute in the fixture dtype.
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    u = u.astype(dtype)
    s = s.astype(dtype)
    vt = vt.astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"svd_{kind}_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(out_path, A=a, U=u, S=s, Vt=vt)
    return out_path


def gen_eigh(
    seed: int = SEED,
    dtype=np.float32,
    n: int = EIG_N,
    kind: str = "symmetric",
) -> str:
    """Generate one seeded symmetric-eig convention fixture (D-04/D-06, PRIM-05).

    Builds a SYMMETRIC ``n×n`` matrix ``A`` (the eig primitive's only v1 feeder
    is the symmetric-by-construction covariance Gram, D-06) by symmetrising a
    random matrix as ``A = (M + Mᵀ) / 2``, then decomposes it with
    ``w, V = np.linalg.eigh(A)``. ``np.linalg.eigh`` returns eigenvalues in
    ASCENDING order; the device eig primitive sorts DESCENDING (D-04) so
    estimators inherit the right order — therefore the fixture stores ``w`` and
    the eigenvector columns ``V`` REVERSED to descending here, matching what the
    primitive emits (the test then compares directly, no re-sort). Eigenvectors
    are only defined up to a sign, so the Rust test sign-aligns columns with
    ``align_rows`` before comparing (D-03).

    Stores named arrays ``A`` (``n×n`` symmetric), ``w`` (length-``n`` descending
    eigenvalues), ``V`` (``n×n`` eigenvectors as COLUMNS, descending). The file
    name encodes ``eigh_{dtype}_seed{seed}``. Returns the absolute path written.
    """
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, n)).astype(dtype)
    # Symmetrise (D-06: the eig primitive trusts symmetry; the oracle must feed a
    # genuinely symmetric matrix). Compute in the fixture dtype.
    a = ((m + m.T) * 0.5).astype(dtype)
    w_asc, v_asc = np.linalg.eigh(a)
    # eigh returns ASCENDING; reverse to DESCENDING (D-04) so the fixture matches
    # the primitive's output order. Reverse eigenvalues and the eigenvector
    # COLUMNS together so each column stays paired with its eigenvalue.
    w = w_asc[::-1].astype(dtype)
    v = v_asc[:, ::-1].astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    # ``kind`` distinguishes the well-conditioned case from degenerate variants
    # (e.g. clustered eigenvalues, D-08); the default symmetric case omits the
    # kind tag for a stable, canonical file name.
    suffix = "" if kind == "symmetric" else f"_{kind}"
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"eigh{suffix}_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(out_path, A=a, w=w, V=v)
    return out_path


# ---------------------------------------------------------------------------
# Phase-4 estimator / primitive fixtures (D-01/D-02/D-07).
# ---------------------------------------------------------------------------

# Cholesky/solve convention-fixture order (D-02, the new SPD-solve primitive).
# A is n×n SPD (= MᵀM + λI, well-conditioned); b is n×rhs; the test compares the
# device solve x against scipy's reference AND checks the ‖L·Lᵀ−A‖ invariant.
CHOL_N, CHOL_RHS = 6, 2
# Ridge that the primitive backs uses a single RHS, but the standalone Cholesky
# fixture carries rhs>1 to exercise the multi-column triangular solve.

# Linear-model convention-fixture shapes (LINEAR-01/02). FULL-RANK case (tall,
# well-conditioned) + a NEAR-COLLINEAR case (a duplicated-then-perturbed column
# so the small-σ cutoff is genuinely exercised — RESEARCH Pitfall 1 / Open Q3).
LIN_N_SAMPLES, LIN_N_FEATURES = 12, 4
LIN_TEST_SAMPLES = 3

# PCA/TruncatedSVD convention-fixture shapes (DECOMP-01/02). TALL (m>n) is the
# standard case; WIDE (n_features>n_samples) exercises the k=min(m,n) truncation
# and the wide SVD path. n_components < min(m,n) so truncation is real.
PCA_TALL = (10, 4)
PCA_WIDE = (4, 6)
PCA_N_COMPONENTS_TALL = 3
PCA_N_COMPONENTS_WIDE = 2
TSVD_SHAPE = (10, 5)
TSVD_N_COMPONENTS = 3

# ---------------------------------------------------------------------------
# Phase-5 distance-based / iterative-solver fixtures (CLUSTER/NEIGH/LINEAR).
# ---------------------------------------------------------------------------

# KMeans convention-fixture (CLUSTER-01, D-09 injected init). A small,
# well-separated 3-blob design (30 samples × 4 features, K=3) so Lloyd converges
# to the SAME partition from the injected init in both mlrs and sklearn — the
# oracle compares centers/labels/inertia up to a label permutation.
KM_N_SAMPLES, KM_N_FEATURES, KM_K = 30, 4, 3

# DBSCAN convention-fixture (CLUSTER-02). eps/min_samples chosen on a 2-blob +
# scattered-noise design so the result has ≥1 cluster, ≥1 noise point (-1), and
# ≥1 border point (Pitfall 7 determinism).
DB_N_SAMPLES, DB_N_FEATURES = 40, 2
DB_EPS, DB_MIN_SAMPLES = 0.7, 4

# KNN convention-fixture (NEIGH-01/02/03 — one fixture serves all three). A
# train set + held-out query set, k neighbors, with DISTINCT distances (Pitfall 8
# — avoid tie ambiguity). Carries both classification targets (y_class) and
# regression targets (y_reg) so the single blob serves classifier + regressor.
KNN_N_TRAIN, KNN_N_QUERY, KNN_N_FEATURES = 30, 8, 3
KNN_K, KNN_N_CLASSES = 5, 3

# Lasso / ElasticNet convention-fixture (LINEAR-03/04). A design with a genuinely
# SPARSE solution (some exact-zero coefficients, Pitfall 1) — more features than
# are truly active.
CD_N_SAMPLES, CD_N_FEATURES = 50, 8
LASSO_ALPHA = 0.5
EN_ALPHA, EN_L1_RATIO = 0.5, 0.5

# LogisticRegression convention-fixture (LINEAR-05). Binary (2-class) + multiclass
# (3-class); predict/predict_proba is the PRIMARY gauge-invariant gate (Pitfall 5).
LOG_N_SAMPLES, LOG_N_QUERY, LOG_N_FEATURES = 40, 8, 4
LOG_C, LOG_MAX_ITER = 1.0, 100

# Phase-10 SGD / linear-SVM convention-fixtures (SGDSVM-01..04). The fixtures are
# PINNED-DETERMINISTIC: shuffle=False (natural row order, no MT19937 to match),
# tol=0 + fixed max_iter (both solvers run the SAME number of full epochs to the
# SAME iterate), explicit eta0/schedule (Pitfall 2/7). The Rust oracle test
# constructs the estimator with EXPLICIT pinned setters, NOT the bare
# builder().build() default (a SEPARATE D-03 litmus checks the default equals
# sklearn's default). n_samples >= n_features so LinearSVC dual='auto' resolves to
# primal (RESEARCH §dual='auto').
SGD_N_SAMPLES, SGD_N_QUERY, SGD_N_FEATURES = 40, 8, 4
# Pinned SGD schedule overrides (deterministic, non-default).
SGD_MAX_ITER = 50
SGD_ETA0 = 0.01
SGD_ALPHA = 1e-4
# LinearSVC / LinearSVR pins.
SVM_C = 1.0
SVR_EPSILON = 0.1
SVM_MAX_ITER = 1000

# Naive Bayes fixture geometry (Phase 11, NB-01..05). Small, well-separated,
# DEFAULT-constructor fits so the default-matches-sklearn test is meaningful.
# GaussianNB uses continuous blobs (reuses _sgd_blobs); the three count-based
# variants use small non-negative integer counts; CategoricalNB uses small
# integer-encoded categorical features with no unseen categories at predict (A3).
NB_N_SAMPLES, NB_N_QUERY, NB_N_FEATURES = 40, 8, 4
NB_N_CLASSES = 3
# Per-feature category count for the CategoricalNB integer-encoded generator.
NB_N_CATEGORIES = 4


def _nb_count_blobs(seed: int, n_classes: int = NB_N_CLASSES):
    """Small NON-NEGATIVE integer-count `X`/`Xq`/`y` for the count-based NB
    variants (Multinomial / Bernoulli / Complement). Each class draws Poisson
    counts from a class-specific per-feature rate so the classes are
    well-separated (a meaningful default fit). Returns `(x, y, xq)` integer arrays.
    """
    rng = np.random.default_rng(seed)
    # Class-specific Poisson rates: class k emphasizes feature block k.
    rates = np.full((n_classes, NB_N_FEATURES), 1.0)
    for k in range(n_classes):
        rates[k, k % NB_N_FEATURES] += 6.0
    per = NB_N_SAMPLES // n_classes
    x = np.vstack(
        [rng.poisson(rates[k], size=(per, NB_N_FEATURES)) for k in range(n_classes)]
    ).astype(np.int64)
    y = np.concatenate([np.full(per, k) for k in range(n_classes)]).astype(np.int64)
    qper = NB_N_QUERY // n_classes
    xq = np.vstack(
        [rng.poisson(rates[k], size=(qper, NB_N_FEATURES)) for k in range(n_classes)]
    ).astype(np.int64)
    return x, y, xq


def _nb_categorical_blobs(seed: int, n_classes: int = NB_N_CLASSES):
    """Small integer-ENCODED categorical `X`/`Xq`/`y` for CategoricalNB. Each
    feature has `NB_N_CATEGORIES` levels; class k biases each feature toward a
    class-specific modal category so the classes separate. NO unseen categories
    at predict (A3): `Xq` is drawn from the SAME per-class modal distribution and
    every category index stays in `[0, NB_N_CATEGORIES)`. Returns `(x, y, xq)`.
    """
    rng = np.random.default_rng(seed)
    per = NB_N_SAMPLES // n_classes
    qper = NB_N_QUERY // n_classes

    def draw(n_rows: int) -> np.ndarray:
        blocks = []
        for k in range(n_classes):
            # Per-class categorical probabilities biased toward category (k+j) % C.
            rows = np.empty((n_rows, NB_N_FEATURES), dtype=np.int64)
            for j in range(NB_N_FEATURES):
                probs = np.full(NB_N_CATEGORIES, 1.0)
                probs[(k + j) % NB_N_CATEGORIES] += 6.0
                probs = probs / probs.sum()
                rows[:, j] = rng.choice(NB_N_CATEGORIES, size=n_rows, p=probs)
            blocks.append(rows)
        return np.vstack(blocks)

    x = draw(per)
    xq = draw(qper)
    y = np.concatenate([np.full(per, k) for k in range(n_classes)]).astype(np.int64)
    return x, y, xq


def _save_nb(out_path: str, x, xq, y, predict, predict_proba, dtype, **extra):
    """Common savez for an NB fixture: cast every array to the fixture dtype and
    store `X`/`Xq`/`y`/`predict`/`predict_proba` (the exact-label hard gate +
    the proba band gate)."""

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    payload = dict(
        X=c(x),
        Xq=c(xq),
        y=c(y),
        predict=c(predict),
        predict_proba=c(predict_proba),
    )
    payload.update({k: c(v) for k, v in extra.items()})
    np.savez(out_path, **payload)
    return out_path


def gen_gaussian_nb(seed: int = SEED, dtype=np.float32) -> str:
    """GaussianNB (NB-01) fixture — DEFAULT-constructor fit on continuous blobs.

    Reuses ``_sgd_blobs`` (well-separated Gaussian class blobs). Stores
    ``X``/``Xq``/``y``/``predict``/``predict_proba`` in the fixture dtype. The
    default ``GaussianNB()`` (var_smoothing=1e-9, priors=None) is fit so the
    default-matches-sklearn test is meaningful.
    """
    from sklearn.naive_bayes import GaussianNB

    _, x, y, xq = _sgd_blobs(seed, n_classes=NB_N_CLASSES)
    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    clf = GaussianNB().fit(x, y)
    out_path = os.path.join(_FIXTURE_DIR, f"gaussian_nb_{dtype_tag}_seed{seed}.npz")
    return _save_nb(
        out_path, x, xq, y, clf.predict(xq), clf.predict_proba(xq), dtype
    )


def gen_multinomial_nb(seed: int = SEED, dtype=np.float32) -> str:
    """MultinomialNB (NB-02) fixture — DEFAULT-constructor fit on integer counts."""
    from sklearn.naive_bayes import MultinomialNB

    x, y, xq = _nb_count_blobs(seed)
    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    clf = MultinomialNB().fit(x, y)
    out_path = os.path.join(_FIXTURE_DIR, f"multinomial_nb_{dtype_tag}_seed{seed}.npz")
    return _save_nb(
        out_path, x, xq, y, clf.predict(xq), clf.predict_proba(xq), dtype
    )


def gen_bernoulli_nb(seed: int = SEED, dtype=np.float32) -> str:
    """BernoulliNB (NB-03) fixture — DEFAULT-constructor fit (binarize=0.0) on
    integer counts (binarized internally by the default threshold)."""
    from sklearn.naive_bayes import BernoulliNB

    x, y, xq = _nb_count_blobs(seed)
    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    clf = BernoulliNB().fit(x, y)
    out_path = os.path.join(_FIXTURE_DIR, f"bernoulli_nb_{dtype_tag}_seed{seed}.npz")
    return _save_nb(
        out_path, x, xq, y, clf.predict(xq), clf.predict_proba(xq), dtype
    )


def gen_complement_nb(seed: int = SEED, dtype=np.float32) -> str:
    """ComplementNB (NB-04) fixture — DEFAULT-constructor fit (norm=False) on
    integer counts."""
    from sklearn.naive_bayes import ComplementNB

    x, y, xq = _nb_count_blobs(seed)
    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    clf = ComplementNB().fit(x, y)
    out_path = os.path.join(_FIXTURE_DIR, f"complement_nb_{dtype_tag}_seed{seed}.npz")
    return _save_nb(
        out_path, x, xq, y, clf.predict(xq), clf.predict_proba(xq), dtype
    )


def gen_categorical_nb(seed: int = SEED, dtype=np.float32) -> str:
    """CategoricalNB (NB-05) fixture — DEFAULT-constructor fit (min_categories=None)
    on integer-encoded categorical features (no unseen categories at predict, A3)."""
    from sklearn.naive_bayes import CategoricalNB

    x, y, xq = _nb_categorical_blobs(seed)
    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    clf = CategoricalNB().fit(x, y)
    out_path = os.path.join(_FIXTURE_DIR, f"categorical_nb_{dtype_tag}_seed{seed}.npz")
    return _save_nb(
        out_path, x, xq, y, clf.predict(xq), clf.predict_proba(xq), dtype
    )


def gen_kmeans(seed: int = SEED, dtype=np.float32) -> str:
    """Generate one seeded KMeans fixture (CLUSTER-01, D-09 injected init).

    Fits ``sklearn.cluster.KMeans`` with an INJECTED fixed ``init`` array (D-09 —
    k-means++ RNG cannot be reproduced bit-for-bit across numpy/Rust, so the
    oracle supplies the initial centers and both mlrs and sklearn run Lloyd from
    the SAME init), ``n_init=1`` (D-09b), ``max_iter=300``, ``tol=1e-4``. A small
    well-separated 3-blob design (``KM_N_SAMPLES``×``KM_N_FEATURES``, K=``KM_K``)
    so Lloyd converges identically up to a label permutation. Stores ``X``,
    ``init`` (the injected centers), ``centers`` (``cluster_centers_``),
    ``labels`` (``labels_``, int-valued), ``inertia`` (``inertia_``). Every array
    passes through the ``c()`` C-contiguous wrapper. Returns the path written.
    """
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    # Three well-separated blobs so the partition is unambiguous.
    centers_true = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [8.0, 8.0, 8.0, 8.0],
            [-8.0, 8.0, -8.0, 8.0],
        ]
    )
    per = KM_N_SAMPLES // KM_K
    x = np.vstack(
        [
            centers_true[k] + 0.4 * rng.standard_normal((per, KM_N_FEATURES))
            for k in range(KM_K)
        ]
    )
    # Injected init (D-09): one actual sample drawn from each blob region so the
    # init is sensible but FIXED (not k-means++ RNG). Both mlrs + sklearn start
    # Lloyd here.
    init = np.vstack([x[k * per] for k in range(KM_K)]).astype(np.float64)

    km = KMeans(
        n_clusters=KM_K, init=init, n_init=1, max_iter=300, tol=1e-4
    ).fit(x)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"kmeans_{dtype_tag}_seed{seed}.npz")
    np.savez(
        out_path,
        X=c(x),
        init=c(init),
        centers=c(km.cluster_centers_),
        labels=c(km.labels_),
        inertia=c([km.inertia_]),
    )
    return out_path


def gen_dbscan(seed: int = SEED, dtype=np.float32) -> str:
    """Generate one seeded DBSCAN fixture (CLUSTER-02).

    Fits ``sklearn.cluster.DBSCAN(eps=DB_EPS, min_samples=DB_MIN_SAMPLES,
    metric='euclidean', algorithm='brute')`` on a 2-blob + scattered-noise design
    chosen so the result has ≥1 cluster, ≥1 noise point (label ``-1``), and ≥1
    border point (Pitfall 7 determinism — core = eps-neighbor-count incl. self ≥
    min_samples). Stores ``X``, ``eps``, ``min_samples``, ``labels`` (``labels_``,
    noise=-1, int-valued), ``core_sample_indices`` (``core_sample_indices_``,
    int-valued). Every array passes through ``c()``. Returns the path written.
    """
    from sklearn.cluster import DBSCAN

    rng = np.random.default_rng(seed)
    # Two tight blobs (clusterable) + a handful of scattered points (noise).
    blob_a = np.array([0.0, 0.0]) + 0.2 * rng.standard_normal((16, DB_N_FEATURES))
    blob_b = np.array([3.0, 3.0]) + 0.2 * rng.standard_normal((16, DB_N_FEATURES))
    noise = rng.uniform(low=-2.0, high=5.0, size=(8, DB_N_FEATURES))
    x = np.vstack([blob_a, blob_b, noise])

    db = DBSCAN(
        eps=DB_EPS,
        min_samples=DB_MIN_SAMPLES,
        metric="euclidean",
        algorithm="brute",
    ).fit(x)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"dbscan_{dtype_tag}_seed{seed}.npz")
    np.savez(
        out_path,
        X=c(x),
        eps=c([DB_EPS]),
        min_samples=c([DB_MIN_SAMPLES]),
        labels=c(db.labels_),
        core_sample_indices=c(db.core_sample_indices_),
    )
    return out_path


def gen_knn(seed: int = SEED, dtype=np.float32) -> str:
    """Generate one seeded KNN fixture (NEIGH-01/02/03 — one fixture, all three).

    Fits ``sklearn.neighbors.NearestNeighbors(n_neighbors=KNN_K,
    algorithm='brute', metric='euclidean')`` on a train set and queries a held-out
    set; ALSO fits a ``KNeighborsClassifier`` and ``KNeighborsRegressor`` (default
    ``weights='uniform'``) so the single blob serves all three neighbor
    estimators. Distances are DISTINCT by construction (Pitfall 8 — avoid tie
    ambiguity). Stores:

      - ``X`` (train), ``Xq`` (query), ``k``,
      - ``distances`` (sqrt-Euclidean k-NN distances of Xq), ``indices``
        (int-valued neighbor indices into X),
      - ``y_class`` (int classification targets), ``y_reg`` (float regression
        targets),
      - ``predict_class`` (classifier ``predict(Xq)``, int), ``predict_proba``
        (classifier ``predict_proba(Xq)``), ``predict_reg`` (regressor
        ``predict(Xq)``).

    Every array passes through ``c()``. Returns the path written.
    """
    from sklearn.neighbors import (
        KNeighborsClassifier,
        KNeighborsRegressor,
        NearestNeighbors,
    )

    rng = np.random.default_rng(seed)
    # Spread the train points widely so pairwise distances are distinct (Pitfall
    # 8): random + a per-row unique offset.
    x = rng.standard_normal((KNN_N_TRAIN, KNN_N_FEATURES)) * 3.0
    x += np.arange(KNN_N_TRAIN)[:, None] * 0.01
    xq = rng.standard_normal((KNN_N_QUERY, KNN_N_FEATURES)) * 3.0

    nn = NearestNeighbors(
        n_neighbors=KNN_K, algorithm="brute", metric="euclidean"
    ).fit(x)
    distances, indices = nn.kneighbors(xq)  # sqrt-Euclidean, ascending

    # Classification + regression targets over the SAME train set.
    y_class = rng.integers(low=0, high=KNN_N_CLASSES, size=KNN_N_TRAIN)
    y_reg = x @ rng.standard_normal(KNN_N_FEATURES) + 0.5

    clf = KNeighborsClassifier(
        n_neighbors=KNN_K, algorithm="brute", metric="euclidean"
    ).fit(x, y_class)
    reg = KNeighborsRegressor(
        n_neighbors=KNN_K, algorithm="brute", metric="euclidean"
    ).fit(x, y_reg)
    predict_class = clf.predict(xq)
    predict_proba = clf.predict_proba(xq)
    predict_reg = reg.predict(xq)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"knn_{dtype_tag}_seed{seed}.npz")
    np.savez(
        out_path,
        X=c(x),
        Xq=c(xq),
        k=c([KNN_K]),
        distances=c(distances),
        indices=c(indices),
        y_class=c(y_class),
        y_reg=c(y_reg),
        predict_class=c(predict_class),
        predict_proba=c(predict_proba),
        predict_reg=c(predict_reg),
    )
    return out_path


# Phase-13 multi-metric KNN-graph oracle (PRIM-11, D-05). The fixed Minkowski-p
# test exponent (p != 1, 2 so it is a genuine non-degenerate Minkowski case that
# the general direct kernel — not a special-cased L1/L2 fast path — must satisfy).
KNN_METRIC_P = 3.0
# The duplicate-point design row index pair (R-9): two TRAIN rows are made
# identical so the include_self=false self-drop must drop the SELF index by
# IDENTITY (D-02), keeping the GENUINE duplicate as a distance-0 neighbour. This
# is the only catch for the cpu-MLIR SILENT self-drop miscompile (FINDING 002-B).
KNN_DUP_ROW_A, KNN_DUP_ROW_B = 0, 4


def gen_knn_metric(
    seed: int = SEED, dtype=np.float32, metric: str = "euclidean", p=None
) -> str:
    """Generate one per-metric KNN-graph oracle fixture (PRIM-11, D-05).

    Fits ``sklearn.neighbors.NearestNeighbors(n_neighbors=K_query,
    algorithm='brute', metric=metric, p=p)`` on a self-referential train set
    (X-vs-X — the KNN graph queries the train points against themselves) and
    stores BOTH the ``k+1`` self-inclusive neighbours (so the prim test can drop
    the self column for ``include_self=false``) and is consumable as the
    ``include_self=true`` ``k`` set (column 0 = self at distance 0).

    ``metric`` is one of ``{"euclidean","manhattan","cosine","chebyshev",
    "minkowski"}``; ``p`` is passed to ``NearestNeighbors`` only for
    ``"minkowski"`` (a fixed non-degenerate exponent, ``KNN_METRIC_P``).

    DUPLICATE-POINT design (R-9): train rows ``KNN_DUP_ROW_A`` and
    ``KNN_DUP_ROW_B`` are made IDENTICAL, so for those query rows a genuine
    neighbour sits at distance 0 alongside self. The ``include_self=false``
    self-drop MUST drop the self index by IDENTITY (D-02), NOT "first
    zero-distance", or it diverges from this oracle. For ``"cosine"`` no row is
    zero-norm (A4) — the standard_normal design plus per-row offset keeps every
    row well away from the origin.

    Stores (mirrors ``gen_knn`` structure, ``c()`` dtype-cast, ``np.savez``):

      - ``X`` (train, self-queried), ``k`` (the requested k true neighbours),
      - ``distances`` / ``indices`` — the sklearn ``k+1`` self-inclusive
        neighbours of X-vs-X (ascending; column 0 = self, distance 0),
      - ``p`` (the Minkowski exponent, or NaN for non-Minkowski metrics),
      - ``dup_row_a`` / ``dup_row_b`` (the identical-row index pair, for the R-9
        VALUE assert).

    The metric tag is carried in the FILENAME only (never an in-blob string
    array — ``mlrs_core::load_npz`` decodes only 4/8-byte float arrays).

    Returns the path written. Filename:
    ``knn_{metric}_{dtype_tag}_seed{seed}.npz``.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    # Spread the train points widely so pairwise distances are distinct (Pitfall
    # 8) EXCEPT the deliberate duplicate pair below: random + a per-row unique
    # offset. X is queried against ITSELF (the KNN graph is X-vs-X).
    x = rng.standard_normal((KNN_N_TRAIN, KNN_N_FEATURES)) * 3.0
    x += np.arange(KNN_N_TRAIN)[:, None] * 0.01
    # DUPLICATE-POINT design (R-9): make row B an EXACT copy of row A.
    x[KNN_DUP_ROW_B, :] = x[KNN_DUP_ROW_A, :]

    # Request k+1 neighbours so the prim test can drop the self column per row for
    # include_self=false AND read column 0 = self for include_self=true.
    k_query = KNN_K + 1
    p_arg = KNN_METRIC_P if metric == "minkowski" else (p if p is not None else 2)
    nn = NearestNeighbors(
        n_neighbors=k_query, algorithm="brute", metric=metric, p=p_arg
    ).fit(x)
    # Enforce the mlrs lowest-index tie-break as the CANONICAL oracle rule so the
    # committed fixtures are derivable from THIS generator (not hand-patched) and
    # the index gate stays INDEPENDENT of the prim's own selection (CR-01/CR-02).
    #
    # A plain lexsort of sklearn's k+1 result is NOT enough: at a BOUNDARY tie
    # (two points equidistant at the (k+1)-th slot, e.g. chebyshev row 25 where
    # indices 0 and 4 are both at the cutoff distance) sklearn arbitrarily returns
    # ONE of them, so reordering the already-returned set cannot recover the
    # lowest-index member. We therefore over-fetch ALL neighbours, then per row
    # select the first k+1 by a global lexicographic key (primary: distance,
    # secondary: neighbour index). This deterministically resolves every tie —
    # including boundary membership — to the lowest index, reproducing the prim's
    # documented convention from an independent rule.
    nn_all = NearestNeighbors(
        n_neighbors=x.shape[0], algorithm="brute", metric=metric, p=p_arg
    ).fit(x)
    dist_all, idx_all = nn_all.kneighbors(x)
    distances = np.empty((x.shape[0], k_query), dtype=dist_all.dtype)
    indices = np.empty((x.shape[0], k_query), dtype=idx_all.dtype)
    for r in range(x.shape[0]):
        order = np.lexsort((idx_all[r], dist_all[r]))  # primary=distance, secondary=index
        sel = order[:k_query]
        distances[r] = dist_all[r][sel]
        indices[r] = idx_all[r][sel]

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    p_store = float(KNN_METRIC_P) if metric == "minkowski" else float("nan")
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"knn_{metric}_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        k=c([KNN_K]),
        distances=c(distances),
        indices=c(indices),
        p=c([p_store]),
        dup_row_a=c([KNN_DUP_ROW_A]),
        dup_row_b=c([KNN_DUP_ROW_B]),
        # NOTE: the metric tag lives in the FILENAME, not as an in-blob array —
        # mlrs_core::load_npz only decodes 4/8-byte float arrays (a numpy
        # unicode `metric` array would make load_npz return InvalidData and
        # break the consuming Rust test). The float `p` carries the only
        # metric-dependent scalar the test needs.
    )
    return out_path


# ---------------------------------------------------------------------------
# Phase-15 HDBSCAN oracle fixtures (HDBS-01..04, D-03/D-04/D-06/D-07). Per-metric
# × per-dtype committed blobs dumping sklearn.cluster.HDBSCAN's OWN labels /
# probabilities / centroids / medoids (the PRIMARY, zero-new-dep oracle) PLUS
# hdbscan 0.8.44's labels / outlier_scores (the GLOSH oracle + cross-check, D-07).
# All arrays are 4/8-byte floats (load_npz constraint, oracle.rs:115-135): labels
# are stored float-valued and cast `as i64` in the Rust test; the metric + dtype
# tag rides the FILENAME only (never an in-blob unicode string). Regenerate ONLY
# in a /tmp venv with `numpy>=1.26 scikit-learn==1.9.0 hdbscan==0.8.44` (PEP-668);
# the resulting blobs are committed, CI never runs this script.
#
# HDBS-02 D-04 exactness (Pitfall 1): the per-metric GATE fixtures spread points so
# the MST edge weights are DISTINCT (the sort is tie-free → exactness holds under
# any stable tie rule). A SEPARATE tie-heavy + duplicate-point fixture
# (`hdbscan_tieheavy_*`) deliberately collides distances (grid + an exact duplicate
# row, R-9) so the spike/back-end can characterise whether ties flip labels — the
# D-04 TRUE GATE. Nested-density fixtures (`hdbscan_nested_*`, two sub-blobs inside
# each of two super-clusters) make the non-default eom/leaf/ε/max_cluster_size/alpha
# knobs DEMONSTRABLY diverge from defaults (Pitfall 5) — asserted in-script before
# writing. Edge cases (`hdbscan_allnoise_*`, `hdbscan_single_*`, `hdbscan_tiny_*`)
# pin the all-noise / single-cluster / n<min_cluster_size degenerate paths.
# ---------------------------------------------------------------------------

# HDBSCAN gate-fixture geometry. The per-metric blob design: 3 well-separated
# blobs (so the partition is unambiguous up to permutation) + a per-row 1e-3
# offset that makes every pairwise — hence every MST edge — weight DISTINCT
# (Pitfall 1 option 2: tie-free sort → exact under any stable rule). A handful of
# uniform-scatter noise points exercises the `-1` sentinel.
HDB_BLOB_N_PER = 12
HDB_BLOB_K = 3
HDB_BLOB_N_FEATURES = 4
HDB_BLOB_NOISE = 5
HDB_MIN_CLUSTER_SIZE = 5
HDB_MINKOWSKI_P = 3.0
# Tie-heavy + duplicate-point design (R-9 / D-04 TRUE GATE): TWO well-separated
# integer-lattice clusters (so the partition is real, not all-noise) whose
# INTERNAL pairwise — hence MST — distances COLLIDE heavily (a unit grid yields
# many equal 1 / √2 / 2 edges), plus one row that is an EXACT copy of another in
# the same cluster (a genuine distance-0 duplicate). The MST tie handling must
# reproduce the oracle partition on this adversarial design.
HDB_TIE_DUP_A, HDB_TIE_DUP_B = 0, 7
# Tie-heavy uses a smaller min_cluster_size: each lattice cluster has 9 points, so
# mcs=3 lets both form while keeping the runt-fallout behaviour exercised.
HDB_TIE_MCS = 3
# Nested-density design (Pitfall 5 / D-09): two sub-blobs (gap 1.5) inside each of
# two well-separated super-clusters (gap 30). eom MERGES each pair → 2 clusters;
# leaf SPLITS → 4. min_cluster_size 20 sits between sub-blob (30) and the runts.
HDB_NESTED_SUB_GAP = 1.5
HDB_NESTED_SUPER_GAP = 30.0
HDB_NESTED_SPREAD = 0.25
HDB_NESTED_N_SUB = 30
HDB_NESTED_MCS = 20


def _hdbscan_blob_design(rng) -> np.ndarray:
    """3 well-separated blobs + scatter noise, per-row offset → distinct MST edges."""
    centers = np.array(
        [[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0], [-10.0, 10.0, -10.0, 10.0]]
    )[: HDB_BLOB_K]
    x = np.vstack(
        [
            centers[c] + 0.35 * rng.standard_normal((HDB_BLOB_N_PER, HDB_BLOB_N_FEATURES))
            for c in range(HDB_BLOB_K)
        ]
    )
    # Per-row 1e-3 offset: pushes every pairwise distance apart so the MST sort is
    # tie-free (Pitfall 1 option 2 — exactness holds under any stable tie rule).
    x = x + np.arange(x.shape[0])[:, None] * 1e-3
    noise = rng.uniform(low=-6.0, high=6.0, size=(HDB_BLOB_NOISE, HDB_BLOB_N_FEATURES))
    return np.vstack([x, noise])


def _hdbscan_tieheavy_design(rng) -> np.ndarray:
    """Two integer-lattice clusters (tie-heavy MST) + one EXACT duplicate row (R-9).

    Each cluster is a 3×3 unit grid → many INTERNAL pairwise distances are equal
    (1, √2, 2, …), so the MST sort is deliberately TIE-HEAVY (the D-04 stress). The
    two grids are well separated (gap 20) so a genuine 2-cluster partition forms
    (not all-noise). No per-row offset — we WANT the collisions here.
    """
    ax, ay = np.meshgrid(np.arange(3.0), np.arange(3.0))
    cluster_a = np.column_stack([ax.ravel(), ay.ravel()])  # 9 points around origin
    cluster_b = cluster_a + np.array([20.0, 20.0])  # 9 points far away
    x = np.vstack([cluster_a, cluster_b])  # 18 points
    # R-9: make row B an EXACT copy of row A (both inside cluster A) — a genuine
    # distance-0 duplicate. The MST/labelling must keep both in the same cluster,
    # identically to the oracle.
    x[HDB_TIE_DUP_B, :] = x[HDB_TIE_DUP_A, :]
    return x


def _hdbscan_nested_design(rng) -> np.ndarray:
    """Two sub-blobs inside each of two super-clusters (eom merges, leaf splits)."""
    pts = []
    for super_c in ([0.0, 0.0], [HDB_NESTED_SUPER_GAP, HDB_NESTED_SUPER_GAP]):
        for s in (0.0, HDB_NESTED_SUB_GAP):
            c = np.array([super_c[0] + s, super_c[1]])
            pts.append(c + HDB_NESTED_SPREAD * rng.standard_normal((HDB_NESTED_N_SUB, 2)))
    x = np.vstack(pts)
    # Tiny per-row offset → distinct MST edges so eom/leaf divergence is the only
    # source of label difference (not tie flips).
    return x + np.arange(x.shape[0])[:, None] * 1e-4


def gen_hdbscan(
    seed: int = SEED,
    dtype=np.float32,
    metric: str = "euclidean",
    structure: str = "blobs",
) -> str:
    """Generate one HDBSCAN oracle fixture (HDBS-01..04, D-03/D-04/D-06/D-07).

    Fits ``sklearn.cluster.HDBSCAN`` (PRIMARY oracle — ``copy=True`` pins the
    sklearn-1.10 ``FutureWarning``) AND ``hdbscan.HDBSCAN`` 0.8.44 (for GLOSH
    ``outlier_scores_`` and the labels cross-check, D-07) on a per-``structure``
    design, then ``np.savez`` the float-cast arrays. ``metric`` is one of
    ``{euclidean, manhattan, cosine, chebyshev, minkowski, precomputed}``; for
    ``minkowski`` the sklearn ``metric_params={'p': HDB_MINKOWSKI_P}`` is passed;
    for ``precomputed`` the design is converted to a square Euclidean distance
    matrix via ``pairwise_distances`` and stored as ``X`` (sklearn refuses
    ``store_centers`` with a precomputed matrix, so the centre arrays are empty
    there).

    ``structure`` is one of ``{blobs, tieheavy, nested, allnoise, single, tiny}``.
    The ``blobs`` design (default) uses distinct-MST-edge-weight spreading
    (Pitfall 1 option 2) so the labels gate is tie-free; ``tieheavy`` is the D-04
    TRUE GATE (integer grid + an exact duplicate row, R-9); ``nested`` carries the
    hierarchical density that makes the non-default knobs diverge (Pitfall 5).

    Stores (all 4/8-byte float, ``c()``-cast — labels are float-valued, cast
    ``as i64`` in the Rust test): ``X``; sklearn ``labels`` / ``probabilities`` /
    ``centroids`` / ``medoids``; hdbscan-0.8.44 ``hdb_labels`` / ``outlier_scores``;
    and for the ``nested`` structure the per-knob label vectors
    ``labels_eom`` / ``labels_leaf`` / ``labels_maxcluster`` / ``labels_alpha``
    (sklearn) and ``labels_epsilon`` (hdbscan 0.8.44 — sklearn 1.9.0's
    ``epsilon_search`` crashes on any merging-epsilon tree, so the epsilon knob is
    cross-oracled against the hdbscan library per D-07). The metric + dtype tag
    rides the FILENAME ONLY.

    Returns the path written. Filename: ``hdbscan_{tag}_{dtype}_seed{seed}.npz``
    where ``tag`` is the ``metric`` for the per-metric gate or the ``structure``
    name for the metric-agnostic specials.
    """
    from sklearn.cluster import HDBSCAN as SkHDBSCAN
    from sklearn.metrics import pairwise_distances

    import hdbscan as hdb  # /tmp venv, pinned 0.8.44 — GLOSH + cross-check oracle.

    rng = np.random.default_rng(seed)
    if structure == "blobs":
        x_design = _hdbscan_blob_design(rng)
    elif structure == "tieheavy":
        x_design = _hdbscan_tieheavy_design(rng)
    elif structure == "nested":
        x_design = _hdbscan_nested_design(rng)
    elif structure == "allnoise":
        # Pure uniform scatter, no density structure → every point is noise (-1).
        x_design = rng.uniform(low=-20.0, high=20.0, size=(20, 3))
    elif structure == "single":
        # One tight homogeneous blob. A single Gaussian has NO density split, so
        # eom would reject the root (all-noise) UNLESS allow_single_cluster=True
        # (set below) — which makes the whole blob the one selected cluster.
        x_design = np.array([2.0, -1.0, 3.0]) + 0.4 * rng.standard_normal((40, 3))
        x_design = x_design + np.arange(x_design.shape[0])[:, None] * 1e-3
    elif structure == "tiny":
        # n < min_cluster_size → sklearn yields all-noise (no cluster can form).
        x_design = rng.standard_normal((HDB_MIN_CLUSTER_SIZE - 2, 3))
    else:
        raise ValueError(f"unknown hdbscan structure {structure!r}")

    # The per-structure min_cluster_size: nested needs the larger mcs that sits
    # between the sub-blob size and the runt threshold for eom/leaf to diverge;
    # tieheavy uses the smaller lattice-cluster mcs so both 9-point grids form.
    if structure == "nested":
        mcs = HDB_NESTED_MCS
    elif structure == "tieheavy":
        mcs = HDB_TIE_MCS
    else:
        mcs = HDB_MIN_CLUSTER_SIZE

    # precomputed (D-02): square Euclidean distance matrix; sklearn refuses
    # store_centers on it, so centres come out empty.
    is_precomputed = metric == "precomputed"
    if is_precomputed:
        x_in = pairwise_distances(x_design, metric="euclidean")
        sk_metric = "precomputed"
        store = None
    else:
        x_in = x_design
        sk_metric = metric
        store = "both"

    sk_kw = dict(
        min_cluster_size=mcs,
        metric=sk_metric,
        cluster_selection_method="eom",
        copy=True,  # pin the sklearn-1.10 FutureWarning (copy default flips False→True).
    )
    # The `tiny` edge case has n < min_cluster_size; min_samples defaults to
    # min_cluster_size and would exceed n. Pin min_samples=1 so sklearn (and
    # hdbscan) run and yield the expected all-noise labelling instead of erroring.
    if structure == "tiny":
        sk_kw["min_samples"] = 1
    # The `single` edge case: a homogeneous blob needs allow_single_cluster=True
    # for eom to select the (split-free) root as the one cluster (else all-noise),
    # plus a small min_samples so the blob's body is dense-reachable (the default
    # min_samples=min_cluster_size over-flags a loose single blob as noise).
    if structure == "single":
        sk_kw["allow_single_cluster"] = True
        sk_kw["min_samples"] = 2
    if store is not None:
        sk_kw["store_centers"] = store
    if metric == "minkowski":
        sk_kw["metric_params"] = {"p": HDB_MINKOWSKI_P}
    h = SkHDBSCAN(**sk_kw).fit(x_in)

    centroids = getattr(h, "centroids_", None)
    medoids = getattr(h, "medoids_", None)
    if centroids is None:
        centroids = np.empty((0, 0))
    if medoids is None:
        medoids = np.empty((0, 0))

    # hdbscan 0.8.44 cross-check + GLOSH outlier_scores (D-07). Force
    # ``algorithm='generic'``: the default ``'best'`` routes to a BallTree that
    # rejects ``cosine`` (and is an APPROXIMATION for the others); ``'generic'``
    # is the exact brute-force path supporting every metric uniformly, matching
    # sklearn's dense ``algorithm='brute'``/'auto' computation (D-07 cross-check).
    hdb_kw = dict(
        min_cluster_size=mcs,
        metric=metric,
        cluster_selection_method="eom",
        algorithm="generic",
    )
    if metric == "minkowski":
        hdb_kw["p"] = HDB_MINKOWSKI_P
    if structure == "tiny":
        hdb_kw["min_samples"] = 1
    if structure == "single":
        hdb_kw["allow_single_cluster"] = True
        hdb_kw["min_samples"] = 2
    hl = hdb.HDBSCAN(**hdb_kw).fit(x_in)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    save_kw = dict(
        X=c(x_in),
        labels=c(h.labels_),
        probabilities=c(h.probabilities_),
        centroids=c(centroids),
        medoids=c(medoids),
        hdb_labels=c(hl.labels_),
        outlier_scores=c(hl.outlier_scores_),
    )

    # Nested-density knob fixtures (Pitfall 5 / D-09): produce the non-default
    # eom/leaf/max_cluster_size/alpha label vectors (sklearn) + epsilon (hdbscan),
    # and ASSERT each genuinely differs from the eom default BEFORE writing.
    if structure == "nested":
        def sk_labels(**over):
            kw = dict(
                min_cluster_size=mcs, metric=sk_metric, copy=True,
                cluster_selection_method="eom",
            )
            kw.update(over)
            return SkHDBSCAN(**kw).fit(x_in).labels_

        labels_eom = h.labels_
        labels_leaf = sk_labels(cluster_selection_method="leaf")
        labels_maxcluster = sk_labels(max_cluster_size=35)
        labels_alpha = sk_labels(alpha=0.5)
        # epsilon: sklearn 1.9.0 epsilon_search crashes on merging trees; oracle the
        # epsilon knob against hdbscan 0.8.44 (D-07 cross-oracle), leaf+eps merges.
        labels_leaf_hdb = hdb.HDBSCAN(
            min_cluster_size=mcs, metric=metric, cluster_selection_method="leaf",
            algorithm="generic",
        ).fit(x_in).labels_
        labels_epsilon = hdb.HDBSCAN(
            min_cluster_size=mcs, metric=metric,
            cluster_selection_method="leaf", cluster_selection_epsilon=1.0,
            algorithm="generic",
        ).fit(x_in).labels_

        # Pitfall 5: each non-default knob MUST demonstrably diverge from default.
        assert not np.array_equal(labels_eom, labels_leaf), (
            "nested eom/leaf must differ (Pitfall 5)"
        )
        assert not np.array_equal(labels_eom, labels_maxcluster), (
            "nested max_cluster_size must change eom labels (Pitfall 5)"
        )
        assert not np.array_equal(labels_eom, labels_alpha), (
            "nested alpha!=1.0 must change eom labels (Pitfall 5)"
        )
        assert not np.array_equal(labels_leaf_hdb, labels_epsilon), (
            "nested cluster_selection_epsilon>0 must merge leaf labels (Pitfall 5)"
        )
        save_kw.update(
            labels_eom=c(labels_eom),
            labels_leaf=c(labels_leaf),
            labels_maxcluster=c(labels_maxcluster),
            labels_alpha=c(labels_alpha),
            labels_leaf_default=c(labels_leaf_hdb),
            labels_epsilon=c(labels_epsilon),
        )

    # Tie-heavy fixture (R-9): record the duplicate-row index pair for the VALUE
    # assert (the duplicate must share its partner's label).
    if structure == "tieheavy":
        save_kw.update(
            dup_row_a=c([HDB_TIE_DUP_A]),
            dup_row_b=c([HDB_TIE_DUP_B]),
        )

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    tag = metric if structure == "blobs" else structure
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"hdbscan_{tag}_{dtype_tag}_seed{seed}.npz")
    np.savez(out_path, **save_kw)
    return out_path


# ---------------------------------------------------------------------------
# Phase-14 UMAP oracle fixtures (UMAP-01..04, D-02). Per-stage × per-metric
# committed blobs dumping umap-learn 0.5.12's OWN internals (NEVER recomputed in
# numpy — RESEARCH Pitfall 6). All arrays are 4/8-byte floats (load_npz
# constraint): KNN indices/COO row-col indices are encoded as float, the metric
# tag lives in the FILENAME (the gen_knn_metric precedent). Regenerate ONLY in a
# /tmp venv with `numpy scipy scikit-learn umap-learn==0.5.12` (PEP 668); the
# resulting blobs are committed, CI never runs this script.
# ---------------------------------------------------------------------------

# Fixed UMAP oracle design (small, CONNECTED at n_neighbors so the single-
# component spectral_layout path matches — RESEARCH Q1). n<=64 keeps spectral on
# the dense-Jacobi path the mlrs `eig` prim reproduces.
UMAP_N = 60
UMAP_N_FEATURES = 8
UMAP_N_NEIGHBORS = 10
UMAP_MINKOWSKI_P = 3.0
# Layout/transform property-gate design: well-separated blobs so trustworthiness
# / kNN-overlap / downstream-ARI are meaningful (3 clusters, deterministic).
UMAP_LAYOUT_N = 60
UMAP_LAYOUT_CLUSTERS = 3
UMAP_TRANSFORM_N_NEW = 15
UMAP_RANDOM_STATE = 42
UMAP_N_EPOCHS = 200
# a/b curve-fit grid (metric-independent, one fixture): (min_dist, spread) pairs.
UMAP_AB_GRID = (
    (0.1, 1.0),
    (0.0, 1.0),
    (0.5, 1.0),
    (0.1, 2.0),
    (0.25, 0.5),
)

# Metric tag → sklearn NearestNeighbors (metric, p) AND umap-learn metric string.
# The umap `metric=` strings match sklearn's for all five (umap dispatches the
# same names to its numba distance fns).
_UMAP_METRICS = {
    "euclidean": ("euclidean", 2),
    "manhattan": ("manhattan", 1),
    "cosine": ("cosine", 2),
    "chebyshev": ("chebyshev", 2),
    "minkowski": ("minkowski", UMAP_MINKOWSKI_P),
}


def _umap_design(seed: int):
    """The shared (n, d) UMAP fixture design — random, well-spread so pairwise
    distances are distinct, no zero-norm row (cosine-safe, A4)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((UMAP_N, UMAP_N_FEATURES)) * 3.0
    x += np.arange(UMAP_N)[:, None] * 0.01
    # Keep every row well away from the origin so cosine is well-defined.
    x += 5.0
    return x


def _umap_knn(x, metric_tag: str):
    """sklearn brute KNN matching the mlrs knn_graph prim (X-vs-X, self-dropped,
    lowest-index tie-break) — the umap membership stage consumes these."""
    from sklearn.neighbors import NearestNeighbors

    sk_metric, p_arg = _UMAP_METRICS[metric_tag]
    k = UMAP_N_NEIGHBORS
    # Over-fetch ALL then per-row lexsort (distance, index) for the documented
    # lowest-index tie-break, then drop self (column 0, distance 0) → (n, k).
    nn = NearestNeighbors(
        n_neighbors=x.shape[0], algorithm="brute", metric=sk_metric, p=p_arg
    ).fit(x)
    dist_all, idx_all = nn.kneighbors(x)
    knn_dist = np.empty((x.shape[0], k), dtype=np.float64)
    knn_idx = np.empty((x.shape[0], k), dtype=np.int64)
    for r in range(x.shape[0]):
        order = np.lexsort((idx_all[r], dist_all[r]))
        sel = [j for j in order if idx_all[r][j] != r][:k]
        knn_dist[r] = dist_all[r][sel]
        knn_idx[r] = idx_all[r][sel]
    return knn_dist, knn_idx


def _umap_cast(dtype):
    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    return c


_UMAP_DTYPE_TAG = {np.float32: "f32", np.float64: "f64"}


def gen_umap_fuzzy(
    seed: int = SEED, dtype=np.float64, metric: str = "euclidean"
) -> str:
    """Smooth-kNN ρ/σ + membership + t-conorm union oracle (UMAP-02, D-02).

    Dumps umap-learn 0.5.12's OWN ``smooth_knn_dist`` (``sigmas``, ``rhos``) and
    ``fuzzy_simplicial_set`` graph (COO ``rows``/``cols``/``vals``) for one
    metric on the fixed UMAP design. The KNN (``knn_idx``/``knn_dist``) the umap
    internals consume are also stored so the mlrs host stages run on the SAME
    neighbours. Stores scalar params ``set_op_mix_ratio``/``local_connectivity``/
    ``n_neighbors``. Indices are float-encoded; metric tag in the filename.
    """
    import numpy as _np
    from umap.umap_ import fuzzy_simplicial_set, smooth_knn_dist

    c = _umap_cast(dtype)
    x = _umap_design(seed)
    knn_dist, knn_idx = _umap_knn(x, metric)

    set_op_mix_ratio = 1.0
    local_connectivity = 1.0
    sigmas, rhos = smooth_knn_dist(
        knn_dist.astype(_np.float64),
        float(UMAP_N_NEIGHBORS),
        local_connectivity=local_connectivity,
    )
    sk_metric, _ = _UMAP_METRICS[metric]
    graph, _s, _r, _d = fuzzy_simplicial_set(
        x,
        UMAP_N_NEIGHBORS,
        _np.random.RandomState(seed),
        sk_metric,
        knn_indices=knn_idx,
        knn_dists=knn_dist,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
        return_dists=True,
    )
    coo = graph.tocoo()

    dtype_tag = _UMAP_DTYPE_TAG[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"umap_fuzzy_{metric}_{dtype_tag}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        knn_idx=c(knn_idx),  # float-encoded indices (load_npz: floats only)
        knn_dist=c(knn_dist),
        sigmas=c(sigmas),
        rhos=c(rhos),
        rows=c(coo.row),  # float-encoded COO row index
        cols=c(coo.col),  # float-encoded COO col index
        vals=c(coo.data),
        n_neighbors=c([UMAP_N_NEIGHBORS]),
        set_op_mix_ratio=c([set_op_mix_ratio]),
        local_connectivity=c([local_connectivity]),
    )
    return out_path


def gen_umap_spectral(
    seed: int = SEED, dtype=np.float64, metric: str = "euclidean"
) -> str:
    """Spectral-init oracle (UMAP-02, D-02). Dumps umap-learn's OWN
    ``spectral_layout`` coords on the symmetric fuzzy graph (n<=64 CONNECTED
    design so the single-component laplacian+eig path matches — RESEARCH Q1).

    Stores the symmetric graph COO (``rows``/``cols``/``vals``) and the spectral
    coordinates ``coords`` (n, n_components). The value-gate compares up-to-sign
    per column (umap applies NO sign-flip; mlrs `recover` does — RESEARCH Q3).
    """
    import numpy as _np
    from umap.spectral import spectral_layout
    from umap.umap_ import fuzzy_simplicial_set

    c = _umap_cast(dtype)
    x = _umap_design(seed)
    knn_dist, knn_idx = _umap_knn(x, metric)
    sk_metric, _ = _UMAP_METRICS[metric]
    graph, _s, _r, _d = fuzzy_simplicial_set(
        x,
        UMAP_N_NEIGHBORS,
        _np.random.RandomState(seed),
        sk_metric,
        knn_indices=knn_idx,
        knn_dists=knn_dist,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        return_dists=True,
    )
    # Symmetrize (t-conorm union is already symmetric, but spectral_layout takes
    # the symmetric affinity — mirror umap's own simplicial_set_embedding which
    # uses graph + graph.T - graph.multiply(graph.T); here the union graph IS the
    # symmetric affinity, so use it directly as umap's spectral_layout input).
    g = graph.maximum(graph.transpose()).tocoo()
    n_components = 2
    # umap's spectral_layout defaults its ARPACK eigsh solver to `tol=1e-4`
    # (`tol or 1e-4` inside `_spectral_layout`), so its eigenvectors carry up to
    # ~4e-5 iterative error vs the EXACT eigenvectors of the same Laplacian. mlrs
    # uses an EXACT dense Jacobi `eig`, so the ≤1e-5 value-gate is only meaningful
    # against near-exact umap coords. Pass a machine-tight `tol` (and a generous
    # `maxiter`) so umap's OWN spectral_layout converges to the exact eigenvectors
    # — still umap's own internal, just at full precision (RESEARCH Q4 / borderline
    # value-gate boundary). `0.0` would re-trigger the 1e-4 default via `tol or`.
    sym = graph.maximum(graph.transpose())
    coords = spectral_layout(
        x,
        sym,
        n_components,
        _np.random.RandomState(seed),
        tol=1e-12,
        maxiter=20000,
    )

    dtype_tag = _UMAP_DTYPE_TAG[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"umap_spectral_{metric}_{dtype_tag}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        rows=c(g.row),
        cols=c(g.col),
        vals=c(g.data),
        coords=c(coords),
        n_components=c([n_components]),
    )
    return out_path


def gen_umap_ab(seed: int = SEED, dtype=np.float64) -> str:
    """a/b curve-fit oracle (UMAP-01/02, D-06). Metric-independent — ONE fixture.

    Dumps umap-learn's OWN ``find_ab_params`` outputs over the
    ``UMAP_AB_GRID`` of ``(min_dist, spread)`` pairs. Stores ``min_dist`` /
    ``spread`` / ``a`` / ``b`` parallel arrays (one row per grid point). The mlrs
    host LM port value-gates ``a``/``b`` to <=1e-5 against these.
    """
    from umap.umap_ import find_ab_params

    c = _umap_cast(dtype)
    min_dists = []
    spreads = []
    a_vals = []
    b_vals = []
    for (min_dist, spread) in UMAP_AB_GRID:
        a, b = find_ab_params(spread, min_dist)
        min_dists.append(min_dist)
        spreads.append(spread)
        a_vals.append(a)
        b_vals.append(b)

    dtype_tag = _UMAP_DTYPE_TAG[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"umap_ab_{dtype_tag}.npz")
    np.savez(
        out_path,
        min_dist=c(min_dists),
        spread=c(spreads),
        a=c(a_vals),
        b=c(b_vals),
    )
    return out_path


def _umap_layout_design(seed: int):
    """Well-separated blobs + true labels for the property-gate (UMAP-03)."""
    from sklearn.datasets import make_blobs

    x, y = make_blobs(
        n_samples=UMAP_LAYOUT_N,
        n_features=UMAP_N_FEATURES,
        centers=UMAP_LAYOUT_CLUSTERS,
        cluster_std=1.0,
        random_state=seed,
    )
    return x.astype(np.float64), y.astype(np.int64)


def gen_umap_layout(
    seed: int = SEED, dtype=np.float64, metric: str = "euclidean"
) -> str:
    """SGD-layout property-gate reference (UMAP-03, D-02). Dumps umap-learn's
    fitted ``embedding_`` + true ``labels`` (for downstream-ARI) on a fixed
    ``random_state``/``n_epochs``. NOT an element-wise oracle — mlrs SplitMix64 !=
    umap Tausworthe, so the gate is trustworthiness/kNN-overlap/ARI (UMAP-03).
    """
    import umap as _umap

    c = _umap_cast(dtype)
    x, y = _umap_layout_design(seed)
    sk_metric, p_arg = _UMAP_METRICS[metric]
    kwds = {"p": p_arg} if metric == "minkowski" else {}
    reducer = _umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=2,
        metric=sk_metric,
        metric_kwds=kwds,
        random_state=UMAP_RANDOM_STATE,
        n_epochs=UMAP_N_EPOCHS,
    )
    embedding = reducer.fit_transform(x)

    dtype_tag = _UMAP_DTYPE_TAG[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"umap_layout_{metric}_{dtype_tag}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        embedding=c(embedding),
        labels=c(y),  # float-encoded integer labels (ARI)
        n_neighbors=c([UMAP_N_NEIGHBORS]),
        n_epochs=c([UMAP_N_EPOCHS]),
        random_state=c([UMAP_RANDOM_STATE]),
    )
    return out_path


def gen_umap_transform(
    seed: int = SEED, dtype=np.float64, metric: str = "euclidean"
) -> str:
    """Transform-new-points property sub-gate reference (UMAP-04, D-02). Dumps
    ``X_train``, ``X_new``, the fitted ``embedding`` (train), and umap's
    ``transform`` output ``embedding_new``. Gate: trustworthiness of new points
    >= umap - eps (NOT element-wise).
    """
    import umap as _umap

    c = _umap_cast(dtype)
    x, y = _umap_layout_design(seed)
    rng = np.random.default_rng(seed + 1)
    # New points drawn from the SAME generating distribution region.
    x_new = x[:UMAP_TRANSFORM_N_NEW] + rng.standard_normal(
        (UMAP_TRANSFORM_N_NEW, UMAP_N_FEATURES)
    ) * 0.1
    sk_metric, p_arg = _UMAP_METRICS[metric]
    kwds = {"p": p_arg} if metric == "minkowski" else {}
    reducer = _umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=2,
        metric=sk_metric,
        metric_kwds=kwds,
        random_state=UMAP_RANDOM_STATE,
        n_epochs=UMAP_N_EPOCHS,
    )
    embedding = reducer.fit_transform(x)
    embedding_new = reducer.transform(x_new)

    dtype_tag = _UMAP_DTYPE_TAG[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"umap_transform_{metric}_{dtype_tag}.npz"
    )
    np.savez(
        out_path,
        X_train=c(x),
        X_new=c(x_new),
        embedding=c(embedding),
        embedding_new=c(embedding_new),
        labels=c(y),
        n_neighbors=c([UMAP_N_NEIGHBORS]),
        n_epochs=c([UMAP_N_EPOCHS]),
        random_state=c([UMAP_RANDOM_STATE]),
    )
    return out_path


def gen_lasso(seed: int = SEED, dtype=np.float32) -> str:
    """Generate one seeded Lasso fixture (LINEAR-03, sklearn coordinate descent).

    Fits ``sklearn.linear_model.Lasso(alpha=LASSO_ALPHA, fit_intercept=True,
    tol=1e-4, max_iter=1000)`` on a design whose true coefficient vector is SPARSE
    (only some features active) so the fitted ``coef_`` has genuine exact zeros
    (Pitfall 1 — the soft-threshold zeroing must be reproduced). Stores ``X``,
    ``y``, ``alpha``, ``coef`` (``coef_``, incl. exact zeros), ``intercept``
    (``intercept_``). Every array passes through ``c()``. Returns the path.
    """
    from sklearn.linear_model import Lasso

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((CD_N_SAMPLES, CD_N_FEATURES))
    # SPARSE ground truth: only 3 of CD_N_FEATURES coefficients are non-zero.
    true_coef = np.zeros(CD_N_FEATURES)
    true_coef[[0, 3, 5]] = [2.5, -1.8, 3.1]
    y = x @ true_coef + 0.5 + 0.05 * rng.standard_normal(CD_N_SAMPLES)

    reg = Lasso(
        alpha=LASSO_ALPHA, fit_intercept=True, tol=1e-4, max_iter=1000
    ).fit(x, y)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"lasso_{dtype_tag}_seed{seed}.npz")
    np.savez(
        out_path,
        X=c(x),
        y=c(y),
        alpha=c([LASSO_ALPHA]),
        coef=c(reg.coef_),
        intercept=c([reg.intercept_]),
    )
    return out_path


def gen_elastic_net(seed: int = SEED, dtype=np.float32) -> str:
    """Generate one seeded ElasticNet fixture (LINEAR-04, sklearn CD).

    Fits ``sklearn.linear_model.ElasticNet(alpha=EN_ALPHA, l1_ratio=EN_L1_RATIO,
    fit_intercept=True, tol=1e-4, max_iter=1000)`` on the same sparse-ground-truth
    design as ``gen_lasso`` (the shared CD kernel serves both, D-03). Stores ``X``,
    ``y``, ``alpha``, ``l1_ratio``, ``coef`` (``coef_``), ``intercept``
    (``intercept_``). Every array passes through ``c()``. Returns the path.
    """
    from sklearn.linear_model import ElasticNet

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((CD_N_SAMPLES, CD_N_FEATURES))
    true_coef = np.zeros(CD_N_FEATURES)
    true_coef[[0, 3, 5]] = [2.5, -1.8, 3.1]
    y = x @ true_coef + 0.5 + 0.05 * rng.standard_normal(CD_N_SAMPLES)

    reg = ElasticNet(
        alpha=EN_ALPHA,
        l1_ratio=EN_L1_RATIO,
        fit_intercept=True,
        tol=1e-4,
        max_iter=1000,
    ).fit(x, y)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"elastic_net_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        y=c(y),
        alpha=c([EN_ALPHA]),
        l1_ratio=c([EN_L1_RATIO]),
        coef=c(reg.coef_),
        intercept=c([reg.intercept_]),
    )
    return out_path


def _symmetric_multinomial_reference(x, y, xq, n_classes, c_inv, l2_reg):
    """Hand-rolled SYMMETRIC-multinomial L-BFGS reference (D-12) via scipy.

    This is the EXACT objective ``crates/mlrs-algos/src/linear/logistic.rs`` and
    the 05-06 ``softmax_loss_grad`` kernel minimize — K full weight vectors
    (symmetric over-parameterization), NOT sklearn's binomial-sigmoid binary loss.
    For ``raw[i,k] = x_i·W_k + b_k``:

        loss(W,b) = (1/n)·Σ_i [ logsumexp_k(raw[i]) − raw[i, y_i] ]
                    + ½·l2_reg·‖W‖²          (intercept b UNPENALIZED, Pitfall 3)

    with ``l2_reg = 1/(C·n)`` (Pitfall 3). The parameter vector is
    ``[W (k×d) | b (k)]`` flattened — exactly the Rust closure's layout. We
    minimize with ``scipy.optimize.minimize(method="L-BFGS-B")`` from a zero start
    (matching the Rust ``x0 = 0`` warm-start) at a TIGHT tolerance so the reference
    is the true minimizer of OUR objective, and return ``(coef (k×d), intercept
    (k), predict_proba(Xq) (nq×k), predict(Xq) (nq,))``.

    sklearn 1.9 has NO symmetric-multinomial binary API (its K=2 path is the
    binomial sigmoid, which differs from this objective by ~3.6e-3 under L2), so
    the binary fixture is a deliberate, user-approved SELF-REFERENCE against this
    hand-rolled trusted oracle — see the 05-10 SUMMARY / STATE decisions.
    """
    from scipy.optimize import minimize

    n, d = x.shape
    k = n_classes

    def unpack(theta):
        w = theta[: k * d].reshape(k, d)
        b = theta[k * d :]
        return w, b

    def loss_and_grad(theta):
        w, b = unpack(theta)
        raw = x @ w.T + b  # (n, k)
        row_max = raw.max(axis=1, keepdims=True)  # logsumexp stability (Pitfall 4)
        ex = np.exp(raw - row_max)
        lse = row_max[:, 0] + np.log(ex.sum(axis=1))  # (n,)
        raw_y = raw[np.arange(n), y]  # (n,)
        data_loss = (lse - raw_y).mean()
        reg_loss = 0.5 * l2_reg * (w * w).sum()  # intercept UNPENALIZED
        loss = data_loss + reg_loss

        p = ex / ex.sum(axis=1, keepdims=True)  # softmax (n, k)
        ind = np.zeros((n, k))
        ind[np.arange(n), y] = 1.0
        diff = (p - ind) / n  # (n, k)
        grad_w = diff.T @ x + l2_reg * w  # (k, d)
        grad_b = diff.sum(axis=0)  # (k,)
        return loss, np.concatenate([grad_w.ravel(), grad_b])

    theta0 = np.zeros(k * d + k)
    res = minimize(
        loss_and_grad,
        theta0,
        jac=True,
        method="L-BFGS-B",
        options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": 2000},
    )
    w, b = unpack(res.x)

    raw_q = xq @ w.T + b  # (nq, k)
    raw_q -= raw_q.max(axis=1, keepdims=True)
    ex_q = np.exp(raw_q)
    proba = ex_q / ex_q.sum(axis=1, keepdims=True)
    predict = proba.argmax(axis=1)
    return w, b, proba, predict


def gen_logistic(seed: int = SEED, dtype=np.float32, multiclass: bool = False) -> str:
    """Generate one seeded LogisticRegression fixture (LINEAR-05).

    Two fixture families per dtype with DIFFERENT trusted references (a deliberate,
    user-approved split — see the 05-10 SUMMARY / STATE decisions):

      - ``multi`` (3-class): sklearn ``LogisticRegression(solver='lbfgs', C=LOG_C,
        max_iter=LOG_MAX_ITER, tol=1e-4, fit_intercept=True)``. sklearn ≥1.5 is
        multinomial-by-default (no deprecated ``multi_class`` arg) and its K≥3
        multinomial loss IS the symmetric multinomial the Rust estimator minimizes
        — so multiclass STAYS SKLEARN-FAITHFUL.
      - ``binary`` (2-class): a hand-rolled SYMMETRIC-multinomial SELF-REFERENCE
        (``_symmetric_multinomial_reference`` via ``scipy.optimize.minimize`` on the
        EXACT D-12 objective the Rust kernel minimizes), NOT sklearn. sklearn's K=2
        path is the BINOMIAL SIGMOID loss, which differs from the symmetric 2-class
        multinomial under L2 by ~3.6e-3; the estimator deliberately keeps D-12
        (symmetric multinomial for ALL K), so its binary ``predict_proba`` is
        validated against OUR trusted reference at the strict 1e-5 gate, NOT against
        sklearn's binomial fit. This is a user-approved correctness tradeoff
        documented LOUDLY in the SUMMARY / STATE / REQUIREMENTS LINEAR-05 note.

    ``predict_proba``/``predict`` are the PRIMARY gauge-invariant gate (Pitfall 5
    — the symmetric over-parameterized softmax has gauge freedom in ``coef_``);
    ``coef_`` is the looser secondary reference. Stores ``X``, ``Xq``, ``y``,
    ``C``, ``coef`` (``coef_``), ``intercept`` (``intercept_``), ``predict``
    (``predict(Xq)``, int), ``predict_proba`` (``predict_proba(Xq)``). Every array
    passes through ``c()``. Returns the path written.
    """
    rng = np.random.default_rng(seed)
    n_classes = 3 if multiclass else 2
    # Well-separated class blobs so the fit converges cleanly and predict is
    # unambiguous.
    centers = rng.standard_normal((n_classes, LOG_N_FEATURES)) * 4.0
    per = LOG_N_SAMPLES // n_classes
    x = np.vstack(
        [
            centers[k] + rng.standard_normal((per, LOG_N_FEATURES))
            for k in range(n_classes)
        ]
    )
    y = np.concatenate([np.full(per, k) for k in range(n_classes)])
    xq = np.vstack(
        [
            centers[k] + rng.standard_normal((LOG_N_QUERY // n_classes, LOG_N_FEATURES))
            for k in range(n_classes)
        ]
    )

    if multiclass:
        # K≥3: sklearn multinomial == symmetric multinomial → SKLEARN-FAITHFUL.
        # Fit at a TIGHT tolerance (tol=1e-10, generous max_iter) so the fixture is
        # the TRUE MINIMUM of the (shared) multinomial objective, NOT sklearn's
        # default early stop. At its default tol=1e-4 sklearn halts ~3.2e-5 short of
        # the minimum, which would put predict_proba borderline OVER the strict 1e-5
        # gate against our (more deeply converged) solver. At the true minimum our
        # symmetric-multinomial solver and sklearn's multinomial agree to ~5e-8 —
        # this stays fully sklearn-faithful (it IS sklearn, just fully converged).
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            solver="lbfgs",
            C=LOG_C,
            max_iter=10000,
            tol=1e-10,
            fit_intercept=True,
        ).fit(x, y)
        coef = clf.coef_
        intercept = clf.intercept_
        predict = clf.predict(xq)
        predict_proba = clf.predict_proba(xq)
    else:
        # K=2: hand-rolled symmetric-multinomial SELF-REFERENCE (NOT sklearn's
        # binomial sigmoid). l2_reg = 1/(C·n) — the Rust estimator's exact scaling.
        n_samples = x.shape[0]
        l2_reg = 1.0 / (LOG_C * n_samples)
        coef, intercept, predict_proba, predict = _symmetric_multinomial_reference(
            x.astype(np.float64),
            y.astype(np.int64),
            xq.astype(np.float64),
            n_classes,
            LOG_C,
            l2_reg,
        )

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    kind = "multi" if multiclass else "binary"
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"logistic_{kind}_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        Xq=c(xq),
        y=c(y),
        C=c([LOG_C]),
        coef=c(coef),
        intercept=c(intercept),
        predict=c(predict),
        predict_proba=c(predict_proba),
    )
    return out_path


def gen_cholesky(seed: int = SEED, dtype=np.float32, n: int = CHOL_N,
                 rhs: int = CHOL_RHS) -> str:
    """Generate one seeded Cholesky/SPD-solve fixture (D-02, the new primitive).

    Builds a WELL-CONDITIONED symmetric positive-definite ``A = MᵀM + λI`` (λ
    keeps the smallest eigenvalue comfortably away from 0 so the f32 Cholesky is
    stable) and a random RHS ``b`` (n×rhs). Stores:

      - ``A`` (n×n SPD), ``b`` (n×rhs),
      - ``x`` = ``scipy.linalg.solve(A, b, assume_a="pos")`` — the reference
        solution the device solve is compared against (``‖A·x − b‖`` invariant),
      - ``L`` = ``scipy.linalg.cholesky(A, lower=True)`` — the lower factor for
        the ``‖L·Lᵀ − A‖`` reconstruction invariant.

    Every array is cast to the fixture dtype so the committed reference matches a
    same-dtype device solve. Returns the absolute path written.
    """
    import scipy.linalg as sla

    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, n))
    # MᵀM is SPD up to rank; + λI guarantees strict positive-definiteness and a
    # benign condition number for the f32 gate.
    a = (m.T @ m + (n * 1.0) * np.eye(n)).astype(dtype)
    b = rng.standard_normal((n, rhs)).astype(dtype)
    # Reference solve (assume_a="pos" routes scipy to its Cholesky path) and the
    # lower factor, both computed in the fixture dtype.
    x = sla.solve(a.astype(dtype), b.astype(dtype), assume_a="pos").astype(dtype)
    lower = sla.cholesky(a.astype(np.float64), lower=True).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"cholesky_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(out_path, A=a, b=b, x=x, L=lower)
    return out_path


def gen_linear_regression(seed: int = SEED, dtype=np.float32) -> str:
    """Generate one seeded LinearRegression fixture (LINEAR-01, sklearn).

    Stores BOTH a full-rank case and a near-collinear case so the SVD-pseudo-
    inverse small-σ cutoff (RESEARCH Pitfall 1) is exercised:

      - ``X`` (full-rank, n_samples×n_features), ``y``,
      - ``coef``/``intercept`` = sklearn ``LinearRegression(fit_intercept=True)``
        ``coef_``/``intercept_`` on ``X``,
      - ``X_test`` (held-out) and ``y_pred`` = ``predict(X_test)``,
      - ``X_coll`` (near-collinear: feature 2 = feature 0 + tiny noise), ``y_coll``,
        and the sklearn ``coef_col``/``intercept_col`` on that collinear system —
        the case that breaks a no-cutoff pseudo-inverse.

    sklearn's ``LinearRegression`` is ``scipy.linalg.lstsq`` (gelsd / SVD), the
    exact contract LINEAR-01 pins. Every array cast to the fixture dtype.
    Returns the absolute path written.
    """
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((LIN_N_SAMPLES, LIN_N_FEATURES))
    true_coef = rng.standard_normal(LIN_N_FEATURES)
    y = x @ true_coef + 0.5 + 0.01 * rng.standard_normal(LIN_N_SAMPLES)

    reg = LinearRegression(fit_intercept=True).fit(x, y)
    x_test = rng.standard_normal((LIN_TEST_SAMPLES, LIN_N_FEATURES))
    y_pred = reg.predict(x_test)

    # NEAR-COLLINEAR case: duplicate column 0 into column 2 with a tiny
    # perturbation → a near-zero singular value the cutoff must drop. A no-cutoff
    # pseudo-inverse blows up the coefficients here (Pitfall 1).
    x_coll = x.copy()
    x_coll[:, 2] = x_coll[:, 0] + 1e-7 * rng.standard_normal(LIN_N_SAMPLES)
    y_coll = x_coll @ true_coef + 0.5 + 0.01 * rng.standard_normal(LIN_N_SAMPLES)
    reg_coll = LinearRegression(fit_intercept=True).fit(x_coll, y_coll)

    def c(arr):
        return np.asarray(arr).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"linear_regression_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        y=c(y),
        coef=c(reg.coef_),
        intercept=c([reg.intercept_]),
        X_test=c(x_test),
        y_pred=c(y_pred),
        X_coll=c(x_coll),
        y_coll=c(y_coll),
        coef_col=c(reg_coll.coef_),
        intercept_col=c([reg_coll.intercept_]),
    )
    return out_path


def gen_ridge(seed: int = SEED, dtype=np.float32,
              alphas=(0.1, 1.0, 10.0)) -> str:
    """Generate one seeded Ridge fixture (LINEAR-02, sklearn cholesky solver).

    Stores ``X``, ``y``, the ``alpha`` sweep, and the stacked sklearn
    ``Ridge(alpha, fit_intercept=True, solver="cholesky")`` ``coef_``/
    ``intercept_`` for each alpha (rows = alphas). The sweep includes
    ``alpha=1.0`` (well-conditioned, the strict-1e-5 case) plus a smaller and a
    larger alpha so the device Cholesky normal-equations path is pinned across
    regularisation strengths. The intercept is NOT penalized (centering, D-05) —
    matching sklearn. Every array cast to the fixture dtype. Returns the path.
    """
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((LIN_N_SAMPLES, LIN_N_FEATURES))
    true_coef = rng.standard_normal(LIN_N_FEATURES)
    y = x @ true_coef + 0.5 + 0.01 * rng.standard_normal(LIN_N_SAMPLES)

    coefs = []
    intercepts = []
    for a in alphas:
        reg = Ridge(alpha=a, fit_intercept=True, solver="cholesky").fit(x, y)
        coefs.append(reg.coef_)
        intercepts.append(reg.intercept_)

    def c(arr):
        return np.asarray(arr).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"ridge_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        y=c(y),
        alpha=c(list(alphas)),
        coef=c(np.vstack(coefs)),
        intercept=c(np.asarray(intercepts)),
    )
    return out_path


def gen_pca(seed: int = SEED, dtype=np.float32, shape=PCA_TALL,
            n_components: int = PCA_N_COMPONENTS_TALL, kind: str = "tall") -> str:
    """Generate one seeded PCA fixture (DECOMP-01, sklearn svd_solver='full').

    Stores ``X`` (``shape``), ``n_components``, and the sklearn
    ``PCA(n_components, svd_solver="full")`` fitted attributes — ``components_``,
    ``explained_variance_``, ``explained_variance_ratio_``, ``singular_values_``,
    ``mean_`` — plus ``transform(X)``. This is sklearn's verified ``_fit_full``
    arithmetic: center by column means → ``svd(full_matrices=False)`` →
    ``svd_flip(u_based_decision=False)`` → ``explained_variance_ = S²/(n−1)``
    (RESEARCH-verified). ``kind`` is ``tall`` (m>n) or ``wide``
    (n_features>n_samples). Every array cast to the fixture dtype. The Rust test
    sign-aligns ``components_`` rows with ``align_rows`` before comparing (D-03).
    Returns the absolute path written.
    """
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    pca = PCA(n_components=n_components, svd_solver="full").fit(x)
    transformed = pca.transform(x)

    def c(arr):
        # Force C-contiguous (row-major) so the committed flat buffer matches the
        # row-major `n_components x n_features` convention every Rust consumer
        # assumes. sklearn PCA's `components_` is FORTRAN-contiguous (it comes
        # from scipy's column-major `Vt`); without this the npz stores the
        # column-major ravel and `load_npz(..).expect_f64("components_")` yields a
        # transposed flat buffer, silently breaking the row-major contract
        # (04-04 Rule-1 fix).
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    # The TALL case is also written under the canonical (kind-less)
    # ``pca_{dtype}_seed{seed}.npz`` name so a consumer can load the default PCA
    # fixture without knowing the tall/wide split; the wide case keeps its kind.
    arrays = dict(
        X=c(x),
        n_components=c([n_components]),
        components_=c(pca.components_),
        explained_variance_=c(pca.explained_variance_),
        explained_variance_ratio_=c(pca.explained_variance_ratio_),
        singular_values_=c(pca.singular_values_),
        mean_=c(pca.mean_),
        transform=c(transformed),
    )
    out_path = os.path.join(
        _FIXTURE_DIR, f"pca_{kind}_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(out_path, **arrays)
    if kind == "tall":
        canonical = os.path.join(
            _FIXTURE_DIR, f"pca_{dtype_tag}_seed{seed}.npz"
        )
        np.savez(canonical, **arrays)
    return out_path


def gen_truncated_svd(seed: int = SEED, dtype=np.float32, shape=TSVD_SHAPE,
                      n_components: int = TSVD_N_COMPONENTS) -> str:
    """Generate one seeded TruncatedSVD fixture (DECOMP-02, sklearn arpack).

    Uses ``algorithm="arpack"`` (DETERMINISTIC, D-07) — NOT the sklearn default
    ``"randomized"`` — with ``random_state=42`` so the committed blob is
    reproducible. Stores ``X`` (``shape``), ``n_components``, and the sklearn
    ``TruncatedSVD`` fitted attributes ``components_``, ``explained_variance_``,
    ``singular_values_`` plus ``transform(X)``. TruncatedSVD does NOT center X
    (thin SVD of uncentered X) and ``explained_variance_`` is the variance of the
    transformed columns, NOT ``S²/(n−1)`` (RESEARCH Pitfall 2). Every array cast
    to the fixture dtype; the Rust test sign-aligns ``components_`` rows with
    ``align_rows`` (D-03). Returns the absolute path written.
    """
    from sklearn.decomposition import TruncatedSVD

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    # algorithm="arpack" → deterministic (D-07); random_state pins the arpack v0.
    tsvd = TruncatedSVD(
        n_components=n_components, algorithm="arpack", random_state=42
    ).fit(x)
    transformed = tsvd.transform(x)

    def c(arr):
        return np.asarray(arr).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"truncated_svd_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        n_components=c([n_components]),
        components_=c(tsvd.components_),
        explained_variance_=c(tsvd.explained_variance_),
        singular_values_=c(tsvd.singular_values_),
        transform=c(transformed),
    )
    return out_path


def gen_empirical_covariance(seed: int = SEED, dtype=np.float32,
                             shape=EMPCOV_FULLRANK, kind: str = "fullrank",
                             assume_centered: bool = False) -> str:
    """Generate one seeded EmpiricalCovariance fixture (COV-01).

    Stores ``X`` (``shape = (n, p)``), ``covariance_``, ``location_`` and
    ``precision_`` from ``sklearn.covariance.EmpiricalCovariance(
    assume_centered).fit(X)``. ``covariance_`` is the biased (``ddof=0``)
    empirical covariance of the (optionally centered) data; ``location_`` is the
    column-mean vector (all-zero when ``assume_centered``); ``precision_`` is the
    pseudo-inverse ``pinvh(covariance_)`` — which for the RANK-DEFICIENT
    (``n <= p``) ``kind="rankdef"`` case exercises the eig-based pinvh floor (the
    covariance is singular, so a Cholesky inverse would fail — D-05). ``p <= 64``
    keeps the symmetric-eig ``precision_`` path inside the MAX_DIM cap.
    VALUE-matched 1e-5. Returns the absolute path written.
    """
    from sklearn.covariance import EmpiricalCovariance

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    est = EmpiricalCovariance(assume_centered=assume_centered).fit(x)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR,
        f"empirical_covariance_{kind}_{dtype_tag}_seed{seed}.npz",
    )
    np.savez(
        out_path,
        X=c(x),
        covariance_=c(est.covariance_),
        location_=c(est.location_),
        precision_=c(est.precision_),
        assume_centered=c([1 if assume_centered else 0]),
    )
    return out_path


def gen_ledoit_wolf(seed: int = SEED, dtype=np.float32,
                    n: int = LW_N_SMALL, p: int = LW_P) -> str:
    """Generate one seeded LedoitWolf fixture (COV-02).

    Stores ``X`` (``shape = (n, p)``), ``covariance_`` and ``shrinkage_`` (as a
    length-1 array) from ``sklearn.covariance.LedoitWolf().fit(X)``. The
    Ledoit–Wolf estimator shrinks the empirical covariance toward a
    scaled-identity target by the closed-form optimal ``shrinkage_ ∈ [0, 1]``
    (RESEARCH Pattern 3). Emitted at TWO sample counts ``n`` (ROADMAP criterion 3)
    so the shrinkage closed form is pinned across n. ``p <= 64``. VALUE-matched
    1e-5. Returns the absolute path written.

    The design is a low-rank-plus-noise CORRELATED matrix (2 latent factors +
    small isotropic noise), NOT pure ``standard_normal`` — an identity-covariance
    Gaussian drives ``shrinkage_`` to the degenerate ``1.0`` (full shrink to the
    identity target), which makes a weak oracle; correlated data lands
    ``shrinkage_`` strictly inside ``(0, 1)`` so the closed-form β/δ arithmetic is
    actually exercised.
    """
    from sklearn.covariance import LedoitWolf

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    loadings = rng.standard_normal((2, p))
    x = z @ loadings + 0.3 * rng.standard_normal((n, p))
    est = LedoitWolf().fit(x)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"ledoit_wolf_n{n}_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        covariance_=c(est.covariance_),
        shrinkage_=c([est.shrinkage_]),
        location_=c(est.location_),
    )
    return out_path


def gen_incremental_pca(seed: int = SEED, dtype=np.float32,
                        shape=IPCA_SHAPE,
                        n_components: int = IPCA_N_COMPONENTS,
                        batch_size: int = IPCA_BATCH_SIZE,
                        whiten: bool = False) -> str:
    """Generate one seeded IncrementalPCA fixture (DECOMP-03).

    Stores ``X`` (``shape``), the hyperparameters ``n_components`` / ``batch_size``
    / ``whiten``, and the sklearn ``IncrementalPCA(n_components, whiten,
    batch_size).fit(X)`` fitted attributes — ``components_``,
    ``explained_variance_``, ``explained_variance_ratio_``, ``singular_values_``,
    ``mean_``, ``var_``, ``n_samples_seen_`` — plus ``transform(X)`` and
    ``inverse_transform(transform(X))``.

    ``components_`` is forced C-contiguous (sklearn's is Fortran-order; without
    this the committed flat blob would be the column-major ravel and silently
    transpose — the 04-04 Rule-1 pitfall). The Rust test sign-aligns
    ``components_`` rows with ``align_rows`` before comparing (DECOMP-03). Sized
    so the per-batch stacked SVD matrix clears the Phase-3 caps
    (``n_components + batch_size + 1 <= 256`` and ``n_features <= 64``).
    Emitted with ``whiten=False`` AND ``whiten=True``. VALUE-matched 1e-5 after
    align_rows. Returns the absolute path written.
    """
    from sklearn.decomposition import IncrementalPCA

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    ipca = IncrementalPCA(
        n_components=n_components, whiten=whiten, batch_size=batch_size
    ).fit(x)
    transformed = ipca.transform(x)
    reconstructed = ipca.inverse_transform(transformed)

    def c(arr):
        # Force C-contiguous (row-major) so the committed flat buffer matches the
        # row-major `n_components x n_features` convention every Rust consumer
        # assumes (sklearn `components_` is Fortran-order — 04-04 Rule-1 fix).
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    whiten_tag = "whiten" if whiten else "nowhiten"
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR,
        f"incremental_pca_{whiten_tag}_{dtype_tag}_seed{seed}.npz",
    )
    np.savez(
        out_path,
        X=c(x),
        n_components=c([n_components]),
        batch_size=c([batch_size]),
        whiten=c([1 if whiten else 0]),
        components_=c(ipca.components_),
        explained_variance_=c(ipca.explained_variance_),
        explained_variance_ratio_=c(ipca.explained_variance_ratio_),
        singular_values_=c(ipca.singular_values_),
        mean_=c(ipca.mean_),
        var_=c(ipca.var_),
        n_samples_seen_=c([ipca.n_samples_seen_]),
        transform=c(transformed),
        inverse_transform=c(reconstructed),
    )
    return out_path


def gen_jl_min_dim(seed: int = SEED, dtype=np.float32) -> str:
    """Generate the johnson_lindenstrauss_min_dim value oracle (PROJ-01/02, D-12).

    Emits ``sklearn.random_projection.johnson_lindenstrauss_min_dim(n_samples,
    eps)`` over the small ``(n_samples, eps)`` grid (eps strictly in ``(0, 1)``)
    as a value oracle: stores the ``n_samples`` grid, the ``eps`` grid, and the
    resulting INTEGER ``min_dim`` matrix (row i / col j = min_dim(n_samples[i],
    eps[j])). This is the ONLY RandomProjection value oracle (D-12 — the RNG is
    SplitMix64, not MT19937, so NO matrix/transform oracle is value-matched; only
    this closed-form JL bound is). VALUE-matched 1e-5 (the values are integers).
    The ``seed`` is unused (the bound is deterministic) but kept for the uniform
    generator signature / file-name convention. Returns the absolute path.
    """
    from sklearn.random_projection import johnson_lindenstrauss_min_dim

    n_samples = np.asarray(JL_N_SAMPLES, dtype=np.int64)
    eps = np.asarray(JL_EPS, dtype=np.float64)
    min_dim = np.empty((len(n_samples), len(eps)), dtype=np.int64)
    for i, ns in enumerate(n_samples):
        for j, ep in enumerate(eps):
            min_dim[i, j] = int(
                johnson_lindenstrauss_min_dim(int(ns), eps=float(ep))
            )

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"jl_min_dim_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(
        out_path,
        n_samples=np.ascontiguousarray(n_samples).astype(dtype),
        eps=np.ascontiguousarray(eps).astype(dtype),
        min_dim=np.ascontiguousarray(min_dim).astype(dtype),
    )
    return out_path


def gen_kernel_matrix(seed: int = SEED, dtype=np.float32) -> str:
    """Generate one seeded kernel-matrix fixture (PRIM-08, D-01/D-02).

    Emits row-major ``X`` (``KM_ROWS_X × KM_COLS``) and ``Y`` (``KM_ROWS_Y ×
    KM_COLS``) plus the host-reference kernel matrix ``K`` for each of the four
    kernels, computed with ``sklearn.metrics.pairwise.pairwise_kernels``:
      - ``K_linear``  = ``X·Yᵀ``.
      - ``K_rbf``     = ``exp(-γ·‖xᵢ − yⱼ‖²)`` with γ resolved to the sklearn
        ``None`` default ``1/n_features`` PLUS a second explicit-γ matrix
        ``K_rbf_gamma`` (γ = 0.5) so both the resolved-default and explicit paths
        are pinned (D-05).
      - ``K_poly``    = ``(γ·⟨xᵢ, yⱼ⟩ + coef0)^degree`` with γ = 1/n_features,
        degree = 3, coef0 = 1 (the sklearn defaults).
      - ``K_sigmoid`` = ``tanh(γ·⟨xᵢ, yⱼ⟩ + coef0)`` with γ = 1/n_features,
        coef0 = 1.

    All arrays ``np.ascontiguousarray(...).astype(dtype)`` (row-major — the PCA
    fix). The resolved γ is stored as ``gamma_default`` for the Rust side to
    reconstruct the default-γ case. Returns the path.
    """
    from sklearn.metrics.pairwise import pairwise_kernels

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((KM_ROWS_X, KM_COLS))
    y = rng.standard_normal((KM_ROWS_Y, KM_COLS))
    gamma_default = 1.0 / KM_COLS
    degree = 3
    coef0 = 1.0
    gamma_explicit = 0.5

    k_linear = pairwise_kernels(x, y, metric="linear")
    k_rbf = pairwise_kernels(x, y, metric="rbf", gamma=gamma_default)
    k_rbf_gamma = pairwise_kernels(x, y, metric="rbf", gamma=gamma_explicit)
    k_poly = pairwise_kernels(
        x, y, metric="poly", gamma=gamma_default, degree=degree, coef0=coef0
    )
    k_sigmoid = pairwise_kernels(
        x, y, metric="sigmoid", gamma=gamma_default, coef0=coef0
    )

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"kernel_matrix_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        Y=c(y),
        gamma_default=c([gamma_default]),
        gamma_explicit=c([gamma_explicit]),
        degree=c([degree]),
        coef0=c([coef0]),
        K_linear=c(k_linear),
        K_rbf=c(k_rbf),
        K_rbf_gamma=c(k_rbf_gamma),
        K_poly=c(k_poly),
        K_sigmoid=c(k_sigmoid),
    )
    return out_path


def gen_kernel_ridge(seed: int = SEED, dtype=np.float32) -> str:
    """Generate one seeded KernelRidge fixture (KERNEL-01, D-04/D-05).

    Emits row-major ``X`` / ``y`` / ``X_test`` plus ``reg.predict(X_test)`` for
    each case (one per kernel + a multi-target + a gamma=None + an explicit-gamma
    case). sklearn ``KernelRidge`` fits RAW data with NO intercept (D-06):
      - ``y_linear`` / ``y_rbf`` / ``y_poly`` / ``y_sigmoid``: one case per kernel
        (alpha=1.0, gamma=1/n_features default, degree=3, coef0=1).
      - ``y_multi``: a 2-target (multi-RHS, D-04) rbf case → predictions are
        ``KR_N_TEST × 2``.
      - ``y_rbf_gamma``: an EXPLICIT gamma (0.5) rbf case (D-05) so the
        resolved-default (``y_rbf``) and explicit paths are both pinned.

    ``n_samples ≤ 64`` (A2 — the n×n Gram clears the MAX_DIM cap). All arrays
    ``np.ascontiguousarray(...).astype(dtype)`` (row-major). Returns the path.
    """
    from sklearn.kernel_ridge import KernelRidge

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((KR_N_SAMPLES, KR_N_FEATURES))
    true_coef = rng.standard_normal(KR_N_FEATURES)
    y = x @ true_coef + 0.1 * rng.standard_normal(KR_N_SAMPLES)
    y2 = np.column_stack(
        [y, x @ rng.standard_normal(KR_N_FEATURES) + 0.1 * rng.standard_normal(KR_N_SAMPLES)]
    )
    x_test = rng.standard_normal((KR_N_TEST, KR_N_FEATURES))

    alpha = 1.0
    gamma_default = 1.0 / KR_N_FEATURES
    gamma_explicit = 0.5
    degree = 3
    coef0 = 1.0

    def fit_predict(kernel, target, **kw):
        reg = KernelRidge(alpha=alpha, kernel=kernel, **kw).fit(x, target)
        return reg.predict(x_test)

    y_linear = fit_predict("linear", y)
    y_rbf = fit_predict("rbf", y, gamma=gamma_default)
    y_poly = fit_predict("poly", y, gamma=gamma_default, degree=degree, coef0=coef0)
    y_sigmoid = fit_predict("sigmoid", y, gamma=gamma_default, coef0=coef0)
    y_multi = fit_predict("rbf", y2, gamma=gamma_default)        # KR_N_TEST × 2
    y_rbf_gamma = fit_predict("rbf", y, gamma=gamma_explicit)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"kernel_ridge_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        y=c(y),
        y2=c(y2),
        X_test=c(x_test),
        alpha=c([alpha]),
        gamma_default=c([gamma_default]),
        gamma_explicit=c([gamma_explicit]),
        degree=c([degree]),
        coef0=c([coef0]),
        y_linear=c(y_linear),
        y_rbf=c(y_rbf),
        y_poly=c(y_poly),
        y_sigmoid=c(y_sigmoid),
        y_multi=c(y_multi),
        y_rbf_gamma=c(y_rbf_gamma),
    )
    return out_path


def gen_kernel_density(seed: int = SEED, dtype=np.float32) -> str:
    """Generate one seeded KernelDensity fixture (KERNEL-02, D-10).

    Emits row-major ``X`` (training) / ``Q`` (queries) plus
    ``kde.score_samples(Q)`` (length-``KD_N_QUERY`` log-densities) for each of
    sklearn's six kernels, all fit with ``atol=0, rtol=0`` (D-10 forced-exact so
    the brute-force tree matches a direct sum), plus two bandwidth-rule cases:
      - ``ld_gaussian`` / ``ld_tophat`` / ``ld_epanechnikov`` / ``ld_exponential``
        / ``ld_linear`` / ``ld_cosine``: per-kernel at a fixed numeric bandwidth.
      - ``ld_scott`` / ``ld_silverman``: gaussian kernel with the ``'scott'`` /
        ``'silverman'`` bandwidth rules (D-09) so the host bandwidth-resolution
        closed form is pinned. The resolved bandwidths are stored as
        ``bw_scott`` / ``bw_silverman`` for the Rust side.

    Tiny ``n`` so brute force matches the exact-forced tree. All arrays
    ``np.ascontiguousarray(...).astype(dtype)`` (row-major). Returns the path.
    """
    from sklearn.neighbors import KernelDensity

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((KD_N_SAMPLES, KD_N_FEATURES))
    q = rng.standard_normal((KD_N_QUERY, KD_N_FEATURES))
    bandwidth = 1.0

    kernels = (
        "gaussian",
        "tophat",
        "epanechnikov",
        "exponential",
        "linear",
        "cosine",
    )

    def score(bw):
        kde = KernelDensity(
            bandwidth=bw, kernel="gaussian", atol=0, rtol=0
        ).fit(x)
        return kde, kde.score_samples(q)

    arrays = {}
    for k in kernels:
        kde = KernelDensity(
            bandwidth=bandwidth, kernel=k, atol=0, rtol=0
        ).fit(x)
        arrays[f"ld_{k}"] = kde.score_samples(q)

    # Bandwidth-rule cases (D-09): sklearn resolves the string rule into the
    # numeric `bandwidth_` attribute at fit; store both the log-density and the
    # resolved bandwidth so the Rust host closed form can be pinned directly.
    kde_scott, ld_scott = score("scott")
    kde_silverman, ld_silverman = score("silverman")

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"kernel_density_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        Q=c(q),
        bandwidth=c([bandwidth]),
        bw_scott=c([kde_scott.bandwidth_]),
        bw_silverman=c([kde_silverman.bandwidth_]),
        ld_scott=c(ld_scott),
        ld_silverman=c(ld_silverman),
        **{k: c(v) for k, v in arrays.items()},
    )
    return out_path


def gen_laplacian(seed: int = SEED, dtype=np.float32, isolated: bool = False) -> str:
    """Generate one normalized-graph-Laplacian fixture (PRIM-09).

    Emits a ready ``n×n`` affinity ``A`` plus the host-reference symmetric
    normalized Laplacian ``L = I − D^-1/2 A D^-1/2`` and the degree-normalization
    vector ``dd[i] = sqrt(degree_i)`` (or ``1`` for an isolated/zero-degree node —
    the typed-zero guard, so ``L`` is finite everywhere and ``L[i,i] = 0`` for an
    isolated node). The Laplacian reproduces scipy's ``_laplacian_dense``
    (``normed=True``) form: the affinity diagonal is zeroed BEFORE the degree
    reduction.

    ``isolated=True`` forces one node's row/column to zero (a zero-degree node) so
    the no-NaN / no-infinite-value guard is exercised. All arrays
    ``np.ascontiguousarray(...).astype(dtype)`` (row-major). Returns the path.
    """
    rng = np.random.default_rng(seed)
    n = LAP_N
    # Symmetric non-negative affinity with a zero diagonal (an rbf-style graph).
    raw = rng.random((n, n))
    a = 0.5 * (raw + raw.T)
    np.fill_diagonal(a, 0.0)
    if isolated:
        # Force the last node to be isolated (zero degree): zero its row + column.
        a[n - 1, :] = 0.0
        a[:, n - 1] = 0.0

    # scipy _laplacian_dense (normed=True): degree on the diagonal-zeroed affinity,
    # dd = sqrt(degree) with a typed-zero guard (dd=1 where degree==0).
    degree = a.sum(axis=1)
    dd = np.sqrt(degree)
    dd_guard = np.where(degree == 0.0, 1.0, dd)
    # L = I − D^-1/2 A D^-1/2; the isolated-node diagonal is 0 (1 - isolated).
    inv = 1.0 / dd_guard
    lap = -a * np.outer(inv, inv)
    diag = np.where(degree == 0.0, 0.0, 1.0)
    np.fill_diagonal(lap, diag)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    tag = "laplacian_isolated" if isolated else "laplacian"
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"{tag}_{dtype_tag}_seed{seed}.npz")
    np.savez(out_path, A=c(a), L=c(lap), dd=c(dd_guard))
    return out_path


# SpectralEmbedding kNN-affinity neighbor count (SPECTRAL-01). sklearn's
# ``SpectralEmbedding`` default ``n_neighbors=None`` resolves to
# ``max(n_samples // 10, 1)`` which, at ``SE_N_SAMPLES=12``, is ``1`` — a
# DISCONNECTED kNN graph whose normalized-Laplacian has a high-multiplicity zero
# eigenvalue. A dense full-spectrum Jacobi ``eig`` (the v2 D-05 path) cannot
# reproduce ARPACK's pick within that degenerate zero subspace, so the
# committed kNN oracle pins an EXPLICIT ``n_neighbors`` that yields a CONNECTED,
# well-separated spectrum the dense pipeline matches to machine precision. The
# rbf oracle (the RESEARCH-validated 8.3e-7 path) is the strict primary gate;
# the kNN oracle is the secondary D-03 connectivity-affinity gate.
SE_N_NEIGHBORS = 5


def gen_spectral_embedding(
    seed: int = SEED, dtype=np.float32, degenerate: bool = False
) -> str:
    """Generate one SpectralEmbedding fixture (SPECTRAL-01, D-01/D-04/D-05/D-09).

    Stores row-major ``X`` plus two committed ``embedding_`` oracles so the
    Wave-2 estimator can value-match BOTH affinity paths against a real sklearn
    reference produced by the dense-eig-faithful configuration:

    - ``embedding``      — ``affinity='rbf'``, ``gamma=1/n_features`` (D-02/D-04).
      This is the RESEARCH-validated dense full-spectrum path (reproduces sklearn
      ARPACK to ~1e-15 here); the STRICT 1e-5 primary gate.
    - ``embedding_knn``  — ``affinity='nearest_neighbors'`` with an EXPLICIT
      ``n_neighbors=SE_N_NEIGHBORS`` (D-03) chosen so the kNN graph is connected
      and the spectrum well-separated, so the dense pipeline matches sklearn
      exactly (the default ``n_neighbors→1`` is disconnected/degenerate and
      cannot be value-matched by a dense eigensolver — see ``SE_N_NEIGHBORS``).

    ``n_components=2`` (D-08), ``n_samples ≤ 64`` (D-05).

    ``degenerate=True`` places the samples on a circle so the rbf
    normalized-Laplacian has a DEGENERATE Fiedler pair (the first non-zero
    eigenvalue has multiplicity 2). The kept embedding then spans a genuinely
    degenerate 2-D eigenspace: a per-element value match is impossible (the
    eigenvectors are defined only up to rotation), but the COLUMN SPACE matches
    sklearn — so the Wave-2 ``subspace`` test (D-09, principal angles) is the
    correct gate. Only ``embedding`` (rbf) is stored for the degenerate fixture.
    Returns the path.
    """
    from sklearn.manifold import SpectralEmbedding

    n, d = SE_N_SAMPLES, SE_N_FEATURES
    if degenerate:
        # Points on a circle → an rbf affinity that approximates a cycle graph,
        # whose normalized Laplacian has a degenerate Fiedler pair (multiplicity
        # 2). The trivial eigenvalue stays simple (connected graph), so the
        # AMBIGUITY is in the kept eigenspace — exactly the D-09 subspace case.
        # IN-01: this geometry is deterministic (linspace/cos/sin), so no `rng`
        # is needed here; it is created only on the non-degenerate path below.
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x = np.zeros((n, d))
        x[:, 0] = np.cos(theta)
        x[:, 1] = np.sin(theta)
    else:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, d))

    gamma = 1.0 / d  # D-04: gamma=None → 1/n_features (resolved at fit).

    # rbf oracle (D-02/D-04): the strict primary gate. random_state fixes the
    # internal sign/RNG so the committed embedding_ is reproducible.
    se_rbf = SpectralEmbedding(
        n_components=SE_N_COMPONENTS,
        affinity="rbf",
        gamma=gamma,
        random_state=seed,
    )
    embedding = se_rbf.fit_transform(x)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    tag = "spectral_embedding_degenerate" if degenerate else "spectral_embedding"
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"{tag}_{dtype_tag}_seed{seed}.npz")

    payload = dict(
        X=c(x),
        embedding=c(embedding),
        gamma_default=c([gamma]),
        n_neighbors=c([SE_N_NEIGHBORS]),
    )
    if not degenerate:
        # kNN-connectivity oracle (D-03): explicit connected n_neighbors.
        se_knn = SpectralEmbedding(
            n_components=SE_N_COMPONENTS,
            affinity="nearest_neighbors",
            n_neighbors=SE_N_NEIGHBORS,
            random_state=seed,
        )
        payload["embedding_knn"] = c(se_knn.fit_transform(x))

    np.savez(out_path, **payload)
    return out_path


def gen_spectral_clustering(seed: int = SEED, dtype=np.float32) -> str:
    """Generate one SpectralClustering fixture (SPECTRAL-02, D-01/D-10).

    CRITICAL (D-01): fit sklearn ``SpectralClustering`` with its OWN DEFAULT
    constructor for affinity/gamma — the default is ``affinity='rbf'``,
    ``gamma=1.0`` (literal, D-04). Only ``n_clusters`` (and ``random_state`` for
    reproducibility) is set; affinity/gamma are NOT overridden.

    The fixture data is WELL-SEPARATED (D-10) so the partition is UNIQUE up to a
    permutation → any KMeans converges to the same labels (the exact-labels gate
    is sign-/init-immune). Stores row-major ``X`` + the fitted ``labels_``.
    ``n_samples ≤ 64`` (D-05). Returns the path.
    """
    from sklearn.cluster import SpectralClustering

    rng = np.random.default_rng(seed)
    n, d, k = SC_N_SAMPLES, SC_N_FEATURES, SC_N_CLUSTERS
    # k well-separated blobs (centers 12 units apart) so the embedding partition
    # is unambiguous (D-10) — the v2 spectral analogue of the tuned DBSCAN fixture.
    per = n // k
    centers = np.array([[12.0 * i, 12.0 * i] for i in range(k)])
    blocks = []
    for i in range(k):
        cnt = per if i < k - 1 else n - per * (k - 1)
        blocks.append(rng.standard_normal((cnt, d)) * 0.2 + centers[i])
    x = np.vstack(blocks)

    # D-01: own default affinity ('rbf', gamma=1.0); D-10: well-separated so the
    # inner KMeans (default kmeans++) lands on the unique partition.
    sc = SpectralClustering(n_clusters=k, random_state=seed)
    labels = sc.fit_predict(x)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"spectral_clustering_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        labels=c(labels),
    )
    return out_path


def _sgd_blobs(seed: int, n_classes: int = 2):
    """Build well-separated class/regression blobs `X`/`Xq`/`y` for the SGD/SVM
    fixtures (shared shape; classifier uses class blobs, regressor a linear map).
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, SGD_N_FEATURES)) * 4.0
    per = SGD_N_SAMPLES // n_classes
    x = np.vstack(
        [
            centers[k] + rng.standard_normal((per, SGD_N_FEATURES))
            for k in range(n_classes)
        ]
    )
    y = np.concatenate([np.full(per, k) for k in range(n_classes)])
    xq = np.vstack(
        [
            centers[k] + rng.standard_normal((SGD_N_QUERY // n_classes, SGD_N_FEATURES))
            for k in range(n_classes)
        ]
    )
    return rng, x, y, xq


def gen_mbsgd_classifier(
    seed: int = SEED, dtype=np.float32, loss: str = "hinge"
) -> str:
    """Generate one PINNED-DETERMINISTIC MBSGDClassifier fixture (SGDSVM-01).

    Fits ``sklearn.linear_model.SGDClassifier`` with the deterministic pins
    ``shuffle=False, tol=0, max_iter=SGD_MAX_ITER`` and an explicit schedule so
    the Rust solver can reproduce the EXACT iterate (Pitfall 2/7). Two variants:

      - ``loss="hinge"`` (default): emit BOTH a ``constant``-schedule fixture AND
        an ``optimal``-schedule fixture so the t0/Bottou math (A1/Pitfall 3) is
        isolated from the gradient math — a constant-schedule match with an
        optimal-schedule mismatch localizes the bug to ``t0``.
      - ``loss="log_loss"``: a SECOND variant for the ``predict_proba`` gate.

    Stores ``X``/``Xq``/``y``/``coef``/``intercept``/``predict`` (and
    ``predict_proba`` for the log-loss variant). Returns the path written.
    """
    from sklearn.linear_model import SGDClassifier

    _, x, y, xq = _sgd_blobs(seed, n_classes=2)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]

    if loss == "log_loss":
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=SGD_ALPHA,
            learning_rate="constant",
            eta0=SGD_ETA0,
            shuffle=False,
            tol=0.0,
            max_iter=SGD_MAX_ITER,
            fit_intercept=True,
            random_state=seed,
        ).fit(x, y)
        os.makedirs(_FIXTURE_DIR, exist_ok=True)
        out_path = os.path.join(
            _FIXTURE_DIR, f"mbsgd_classifier_log_{dtype_tag}_seed{seed}.npz"
        )
        np.savez(
            out_path,
            X=c(x),
            Xq=c(xq),
            y=c(y),
            coef=c(clf.coef_),
            intercept=c(clf.intercept_),
            predict=c(clf.predict(xq)),
            predict_proba=c(clf.predict_proba(xq)),
        )
        return out_path

    # hinge default — emit constant-schedule (primary) AND optimal-schedule.
    # The default file name is the constant-schedule fixture; the optimal-schedule
    # variant carries an `_optimal` infix so the Wave-1 t0 test can load it.
    paths = []
    for schedule, infix in (("constant", ""), ("optimal", "_optimal")):
        kwargs = dict(
            loss="hinge",
            penalty="l2",
            alpha=SGD_ALPHA,
            learning_rate=schedule,
            shuffle=False,
            tol=0.0,
            max_iter=SGD_MAX_ITER,
            fit_intercept=True,
            random_state=seed,
        )
        if schedule == "constant":
            kwargs["eta0"] = SGD_ETA0
        clf = SGDClassifier(**kwargs).fit(x, y)
        os.makedirs(_FIXTURE_DIR, exist_ok=True)
        out_path = os.path.join(
            _FIXTURE_DIR, f"mbsgd_classifier{infix}_{dtype_tag}_seed{seed}.npz"
        )
        np.savez(
            out_path,
            X=c(x),
            Xq=c(xq),
            y=c(y),
            coef=c(clf.coef_),
            intercept=c(clf.intercept_),
            predict=c(clf.predict(xq)),
        )
        paths.append(out_path)
    return paths[0]


def gen_mbsgd_regressor(seed: int = SEED, dtype=np.float32) -> str:
    """Generate one PINNED-DETERMINISTIC MBSGDRegressor fixture (SGDSVM-02).

    Fits ``sklearn.linear_model.SGDRegressor`` (``squared_error`` + ``invscaling``)
    with ``shuffle=False, tol=0, max_iter=SGD_MAX_ITER`` and explicit
    ``eta0``/``power_t`` (Pitfall 2/7). Stores ``X``/``Xq``/``y``/``coef``/
    ``intercept``/``predict``. Returns the path written.
    """
    from sklearn.linear_model import SGDRegressor

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((SGD_N_SAMPLES, SGD_N_FEATURES))
    true_coef = rng.standard_normal(SGD_N_FEATURES)
    y = x @ true_coef + 0.5 + 0.05 * rng.standard_normal(SGD_N_SAMPLES)
    xq = rng.standard_normal((SGD_N_QUERY, SGD_N_FEATURES))

    reg = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=SGD_ALPHA,
        learning_rate="invscaling",
        eta0=SGD_ETA0,
        power_t=0.25,
        shuffle=False,
        tol=0.0,
        max_iter=SGD_MAX_ITER,
        fit_intercept=True,
        random_state=seed,
    ).fit(x, y)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"mbsgd_regressor_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        Xq=c(xq),
        y=c(y),
        coef=c(reg.coef_),
        intercept=c(reg.intercept_),
        predict=c(reg.predict(xq)),
    )
    return out_path


def gen_linear_svc(seed: int = SEED, dtype=np.float32) -> str:
    """Generate one LinearSVC fixture (SGDSVM-03).

    Fits ``sklearn.svm.LinearSVC`` (``squared_hinge`` default, ``dual='auto'``,
    ``intercept_scaling=1.0``). With n_samples >= n_features, ``dual='auto'``
    resolves to primal (RESEARCH §dual='auto'). LinearSVC is liblinear CD —
    converged (no SGD pins needed). Stores ``X``/``Xq``/``y``/``coef``/
    ``intercept``/``predict`` (labels). Returns the path written.
    """
    from sklearn.svm import LinearSVC

    _, x, y, xq = _sgd_blobs(seed, n_classes=2)

    clf = LinearSVC(
        loss="squared_hinge",
        penalty="l2",
        C=SVM_C,
        dual="auto",
        intercept_scaling=1.0,
        fit_intercept=True,
        max_iter=SVM_MAX_ITER,
        tol=1e-4,
        random_state=seed,
    ).fit(x, y)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"linear_svc_{dtype_tag}_seed{seed}.npz")
    np.savez(
        out_path,
        X=c(x),
        Xq=c(xq),
        y=c(y),
        coef=c(clf.coef_),
        intercept=c(clf.intercept_),
        predict=c(clf.predict(xq)),
    )
    return out_path


def gen_linear_svr(seed: int = SEED, dtype=np.float32) -> str:
    """Generate one LinearSVR fixture (SGDSVM-04).

    Fits ``sklearn.svm.LinearSVR`` (``squared_epsilon_insensitive`` default +
    ``epsilon``, ``dual='auto'``, ``intercept_scaling=1.0``). Liblinear CD —
    converged. Stores ``X``/``Xq``/``y``/``coef``/``intercept``/``predict``.
    Returns the path written.
    """
    from sklearn.svm import LinearSVR

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((SGD_N_SAMPLES, SGD_N_FEATURES))
    true_coef = rng.standard_normal(SGD_N_FEATURES)
    y = x @ true_coef + 0.5 + 0.05 * rng.standard_normal(SGD_N_SAMPLES)
    xq = rng.standard_normal((SGD_N_QUERY, SGD_N_FEATURES))

    reg = LinearSVR(
        loss="squared_epsilon_insensitive",
        epsilon=SVR_EPSILON,
        C=SVM_C,
        dual="auto",
        intercept_scaling=1.0,
        fit_intercept=True,
        max_iter=SVM_MAX_ITER,
        tol=1e-4,
        random_state=seed,
    ).fit(x, y)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"linear_svr_{dtype_tag}_seed{seed}.npz")
    np.savez(
        out_path,
        X=c(x),
        Xq=c(xq),
        y=c(y),
        coef=c(reg.coef_),
        intercept=c(reg.intercept_),
        predict=c(reg.predict(xq)),
    )
    return out_path


def main() -> None:
    for dtype in (np.float32, np.float64):
        path = gen_saxpy(dtype=dtype)
        print(f"wrote {path}")
    for dtype in (np.float32, np.float64):
        path = gen_gemm(dtype=dtype)
        print(f"wrote {path}")
    # Distance (PRIM-03): squared f32/f64 + the sqrt f64 variant (D-12).
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_distance(dtype=dtype, sqrt=False)}")
    print(f"wrote {gen_distance(dtype=np.float64, sqrt=True)}")
    # Covariance (PRIM-04): population (ddof=0) f64, sample (ddof=1) f64 + f32
    # so BOTH ddof conventions are pinned and the f32 sample case is covered.
    print(f"wrote {gen_covariance(dtype=np.float64, ddof=0)}")
    print(f"wrote {gen_covariance(dtype=np.float64, ddof=1)}")
    print(f"wrote {gen_covariance(dtype=np.float32, ddof=1)}")
    print(f"wrote {gen_argmin_tie()}")
    # SVD (PRIM-05, D-05/D-09): tall (m≥n) f32+f64 to exercise the f64 cpu gate,
    # plus a wide (m<n) f32 case for the Aᵀ-swap path. np.linalg.svd is the
    # numpy reference (full_matrices=False, descending S — D-02/D-04).
    print(f"wrote {gen_svd(dtype=np.float32, shape=SVD_TALL, kind='tall')}")
    print(f"wrote {gen_svd(dtype=np.float64, shape=SVD_TALL, kind='tall')}")
    print(f"wrote {gen_svd(dtype=np.float32, shape=SVD_WIDE, kind='wide')}")
    # Odd thin-dim (k=5) tall case (CR-01): f32 (cpu+rocm) + f64 (cpu gate) so
    # the committed numpy oracle pins the odd-parity pairing the primitive must
    # now hold. 9×5 → U is 9×5, S length 5, Vt is 5×5.
    print(f"wrote {gen_svd(dtype=np.float32, shape=SVD_TALL_ODD, kind='tall_odd')}")
    print(f"wrote {gen_svd(dtype=np.float64, shape=SVD_TALL_ODD, kind='tall_odd')}")
    # Symmetric eig (PRIM-05, D-04/D-06): f32+f64 so the f64 cpu path is pinned.
    # np.linalg.eigh is the numpy reference, REVERSED to descending (D-04).
    print(f"wrote {gen_eigh(dtype=np.float32)}")
    print(f"wrote {gen_eigh(dtype=np.float64)}")

    # ---- Phase-4 estimator/primitive fixtures (D-01/D-02/D-07) ----
    # Each generator writes BOTH f32 (rocm gate) and f64 (cpu gate) blobs.
    # Cholesky/SPD-solve primitive (D-02): scipy reference + L factor.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_cholesky(dtype=dtype)}")
    # LinearRegression (LINEAR-01): full-rank + near-collinear (small-σ cutoff).
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_linear_regression(dtype=dtype)}")
    # Ridge (LINEAR-02): cholesky solver, alpha sweep incl. the strict 1.0 case.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_ridge(dtype=dtype)}")
    # PCA (DECOMP-01): tall (m>n) + wide (n_features>n_samples); svd_solver=full.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_pca(dtype=dtype, shape=PCA_TALL, n_components=PCA_N_COMPONENTS_TALL, kind='tall')}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_pca(dtype=dtype, shape=PCA_WIDE, n_components=PCA_N_COMPONENTS_WIDE, kind='wide')}")
    # TruncatedSVD (DECOMP-02): DETERMINISTIC algorithm='arpack' (NOT randomized).
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_truncated_svd(dtype=dtype)}")

    # ---- Phase-5 distance-based / iterative-solver fixtures ----
    # Each generator writes BOTH f32 (rocm gate) and f64 (cpu gate) blobs.
    # KMeans (CLUSTER-01): injected init (D-09) so Lloyd is deterministic.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_kmeans(dtype=dtype)}")
    # DBSCAN (CLUSTER-02): eps/min_samples giving cluster + noise(-1) + border.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_dbscan(dtype=dtype)}")
    # KNN (NEIGH-01/02/03): one fixture serves NearestNeighbors + classifier +
    # regressor; distinct distances (Pitfall 8).
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_knn(dtype=dtype)}")
    # Phase-13 multi-metric KNN-graph oracle (PRIM-11, D-05): the full fixed
    # metric set (euclidean, manhattan, cosine, chebyshev, minkowski-p) × {f32
    # (rocm gate), f64 (cpu gate)}, each carrying a DUPLICATE-POINT train row
    # (R-9) so the include_self=false self-drop VALUE assert catches the cpu-MLIR
    # silent miscompile (FINDING 002-B). X is queried against itself (X-vs-X).
    for metric in ("euclidean", "manhattan", "cosine", "chebyshev", "minkowski"):
        for dtype in (np.float32, np.float64):
            print(f"wrote {gen_knn_metric(dtype=dtype, metric=metric)}")
    # ---- Phase-14 UMAP oracle fixtures (UMAP-01..04, D-02) ----
    # Per-stage × per-metric committed blobs dumping umap-learn 0.5.12 internals
    # (NEVER recomputed — RESEARCH Pitfall 6). f64 only (the cpu value gate; the
    # deterministic stages value-gate to <=1e-5 in host f64 — RESEARCH §host-f64
    # readback). Regen in a /tmp venv with `umap-learn==0.5.12` (PEP 668).
    for metric in ("euclidean", "manhattan", "cosine", "chebyshev", "minkowski"):
        print(f"wrote {gen_umap_fuzzy(dtype=np.float64, metric=metric)}")
        print(f"wrote {gen_umap_spectral(dtype=np.float64, metric=metric)}")
        print(f"wrote {gen_umap_layout(dtype=np.float64, metric=metric)}")
        print(f"wrote {gen_umap_transform(dtype=np.float64, metric=metric)}")
    # a/b curve fit is metric-independent — one fixture.
    print(f"wrote {gen_umap_ab(dtype=np.float64)}")
    # ---- Phase-15 HDBSCAN oracle fixtures (HDBS-01..04, D-03/D-04/D-06/D-07) ----
    # Per-metric GATE blobs (distinct-MST-edge-weight, Pitfall 1 opt 2) over the
    # full metric set × {f32 (rocm gate), f64 (cpu gate)}; each carries sklearn
    # labels/probabilities/centroids/medoids (PRIMARY oracle) + hdbscan 0.8.44
    # hdb_labels/outlier_scores (GLOSH + cross-check, D-07). Regen in a /tmp venv
    # with `numpy>=1.26 scikit-learn==1.9.0 hdbscan==0.8.44` (PEP-668).
    for metric in (
        "euclidean", "manhattan", "cosine", "chebyshev", "minkowski", "precomputed"
    ):
        for dtype in (np.float32, np.float64):
            print(f"wrote {gen_hdbscan(dtype=dtype, metric=metric, structure='blobs')}")
    # Metric-agnostic specials (euclidean): the D-04 TRUE GATE tie-heavy +
    # duplicate-point fixture (R-9), the nested-density knob fixture (eom/leaf/
    # epsilon/max_cluster_size/alpha diverge — Pitfall 5, asserted in-script), and
    # the all-noise / single-cluster / n<min_cluster_size edge cases.
    for structure in ("tieheavy", "nested", "allnoise", "single", "tiny"):
        for dtype in (np.float32, np.float64):
            print(f"wrote {gen_hdbscan(dtype=dtype, metric='euclidean', structure=structure)}")
    # Lasso (LINEAR-03): sparse coef_ with exact zeros (Pitfall 1).
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_lasso(dtype=dtype)}")
    # ElasticNet (LINEAR-04): shared CD design, l1_ratio mixing.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_elastic_net(dtype=dtype)}")
    # LogReg (LINEAR-05): binary + multiclass; predict/predict_proba primary gate.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_logistic(dtype=dtype, multiclass=False)}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_logistic(dtype=dtype, multiclass=True)}")

    # ---- Phase-7 covariance & projection fixtures ----
    # Each VALUE-matched generator writes BOTH f32 (rocm gate) and f64 (cpu gate)
    # blobs. EmpiricalCovariance (COV-01): full-rank (n>p) + RANK-DEFICIENT (n<=p)
    # so the eig-based pinvh `precision_` floor is exercised.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_empirical_covariance(dtype=dtype, shape=EMPCOV_FULLRANK, kind='fullrank')}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_empirical_covariance(dtype=dtype, shape=EMPCOV_RANKDEF, kind='rankdef')}")
    # WR-02: assume_centered=True drives the SEPARATE uncentered host-Gram
    # branch (Xᵀ·X/n, location_ all-zero) that the centered fixtures never reach.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_empirical_covariance(dtype=dtype, shape=EMPCOV_FULLRANK, kind='centered', assume_centered=True)}")
    # LedoitWolf (COV-02): TWO sample counts n (ROADMAP criterion 3).
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_ledoit_wolf(dtype=dtype, n=LW_N_SMALL)}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_ledoit_wolf(dtype=dtype, n=LW_N_LARGE)}")
    # IncrementalPCA (DECOMP-03): whiten=False AND whiten=True; stacked SVD
    # matrix sized under MAX_ROWS/MAX_COLS.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_incremental_pca(dtype=dtype, whiten=False)}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_incremental_pca(dtype=dtype, whiten=True)}")
    # johnson_lindenstrauss_min_dim (PROJ-01/02, D-12): the ONE value oracle.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_jl_min_dim(dtype=dtype)}")

    # ---- Phase-8 kernel-family fixtures ----
    # Each generator writes BOTH f32 (rocm gate) and f64 (cpu gate) blobs.
    # kernel_matrix (PRIM-08): the 4 kernels (linear/rbf/poly/sigmoid) + a
    # default-gamma and explicit-gamma RBF case (D-01/D-02/D-05).
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_kernel_matrix(dtype=dtype)}")
    # KernelRidge (KERNEL-01): one case per kernel + 2-target multi-RHS (D-04) +
    # gamma None/explicit (D-05) + degree=3/coef0=1 poly/sigmoid defaults.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_kernel_ridge(dtype=dtype)}")
    # KernelDensity (KERNEL-02): all 6 kernels forced-exact (atol=0, rtol=0) +
    # scott/silverman bandwidth rules (D-09/D-10).
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_kernel_density(dtype=dtype)}")

    # ---- Phase-9 spectral-family fixtures ----
    # laplacian (PRIM-09): the normalized-graph-Laplacian value fixture + an
    # isolated-node (zero-degree) fixture for the no-NaN/no-infinite-value guard.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_laplacian(dtype=dtype, isolated=False)}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_laplacian(dtype=dtype, isolated=True)}")
    # SpectralEmbedding (SPECTRAL-01): the default-constructor embedding (D-01) +
    # a degenerate-spectrum fixture for the subspace test (D-09).
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_spectral_embedding(dtype=dtype, degenerate=False)}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_spectral_embedding(dtype=dtype, degenerate=True)}")
    # SpectralClustering (SPECTRAL-02): default-constructor labels on a
    # well-separated fixture (D-01/D-10) — exact labels up to permutation.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_spectral_clustering(dtype=dtype)}")

    # ---- Phase-10 SGD / linear-SVM fixtures (SGDSVM-01..04) ----
    # Each generator writes BOTH f32 (rocm gate) and f64 (cpu gate) blobs, PINNED
    # deterministic (shuffle=False, tol=0, fixed max_iter, explicit schedule).
    # MBSGDClassifier (SGDSVM-01): hinge default emits constant + optimal schedule
    # variants (A1/Pitfall 3 t0 isolation); a SECOND log-loss variant feeds the
    # predict_proba gate.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_mbsgd_classifier(dtype=dtype, loss='hinge')}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_mbsgd_classifier(dtype=dtype, loss='log_loss')}")
    # MBSGDRegressor (SGDSVM-02): squared_error + invscaling pinned.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_mbsgd_regressor(dtype=dtype)}")
    # LinearSVC (SGDSVM-03): squared_hinge, dual='auto'→primal, intercept_scaling.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_linear_svc(dtype=dtype)}")
    # LinearSVR (SGDSVM-04): squared_epsilon_insensitive + epsilon.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_linear_svr(dtype=dtype)}")

    # ---- Phase-11 Naive Bayes fixtures (NB-01..05) ----
    # Each generator writes BOTH f32 (rocm gate) and f64 (cpu gate) blobs from the
    # estimator's OWN DEFAULT constructor (D-02 — so the default-matches-sklearn
    # test is meaningful). predict = exact-label HARD gate; predict_proba = band.
    # GaussianNB (NB-01): continuous blobs.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_gaussian_nb(dtype=dtype)}")
    # MultinomialNB (NB-02): integer counts.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_multinomial_nb(dtype=dtype)}")
    # BernoulliNB (NB-03): integer counts (binarize=0.0 default).
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_bernoulli_nb(dtype=dtype)}")
    # ComplementNB (NB-04): integer counts (norm=False default, argmin decode).
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_complement_nb(dtype=dtype)}")
    # CategoricalNB (NB-05): integer-encoded categorical features (no unseen, A3).
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_categorical_nb(dtype=dtype)}")


if __name__ == "__main__":
    main()
