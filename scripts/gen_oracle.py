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
import sys
import warnings

import numpy as np

# Resolve the repo root from this file's location so the script is runnable
# from any working directory and always writes to ``<repo>/tests/fixtures``.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Make the sibling generator modules importable no matter the working directory
# (`main()` imports `gen_feature_selection_oracle` at the bottom of this file).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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

# LINEAR-01 large-`n_samples` fixture (the `fit_gram_eig` Gram+eig path, above
# the direct-SVD single-cube kernel's `MAX_ROWS = 256` row cap). Deliberately
# `n_samples > 256` (crosses the cap) with `n_features` comfortably under the
# eig path's `MAX_DIM = 64` (D-06).
LIN_LARGE_N_SAMPLES, LIN_LARGE_N_FEATURES = 2000, 20

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


# KNeighborsRegressor full-parameter oracle (KNN-REG-PARAMS). Larger than
# `gen_knn`'s 30x3 blob because the parameter surface needs shapes that blob
# cannot express: a second target column (multi-output) and a query that
# COINCIDES with a training point (the `weights='distance'` 1/0 branch).
KNN_REG_N_TRAIN, KNN_REG_N_QUERY, KNN_REG_N_FEATURES = 40, 12, 4
KNN_REG_K = 5
KNN_REG_N_OUTPUTS = 2
# Non-degenerate Minkowski exponent — p != 1, 2, inf, so it exercises the
# general `minkowski_dist` kernel rather than one of the collapsed fast paths.
KNN_REG_P = 3.0
# Query rows overwritten with an exact copy of a TRAIN row. `weights='distance'`
# then divides by zero for those rows and must take sklearn's indicator branch
# (coincident neighbours weight 1, all others 0) instead of producing NaN. Two
# rows so the case is not a single-row accident, and a train row that is itself
# duplicated (see `_knn_reg_data`) so a row can have TWO zero-distance
# neighbours — the case where "use the nearest" and "use all the zeros" differ.
KNN_REG_COINCIDENT_QUERIES = (3, 9)
# Every STRING the `metric` parameter accepts, aliases included. Five distance
# FUNCTIONS, nine spellings: `l2` is `euclidean`, `l1`/`cityblock` are
# `manhattan`, `infinity` is `chebyshev`, and `minkowski` at the default `p = 2`
# collapses onto `euclidean`. Each is generated separately so the oracle proves
# the fold rather than assuming it.
KNN_REG_METRIC_STRINGS = (
    "minkowski",
    "euclidean",
    "l2",
    "manhattan",
    "l1",
    "cityblock",
    "chebyshev",
    "infinity",
    "cosine",
)
# Every STRING the `algorithm` parameter accepts. mlrs resolves all four to
# brute force; sklearn genuinely builds a tree for two of them.
KNN_REG_ALGORITHMS = ("auto", "brute", "kd_tree", "ball_tree")


def _knn_reg_data(seed: int):
    """The shared `(X, Xq, y, y_multi)` design for the KNN-regressor fixtures.

    Train points are spread widely with a per-row offset so pairwise distances
    are distinct (Pitfall 8 — a tie would make the oracle's neighbour choice
    ambiguous), EXCEPT for one deliberately duplicated pair, and two query rows
    are exact copies of train rows so the zero-distance weighting branch is
    reachable. All rows are shifted positive before the cosine cases so no row
    is near-zero-norm, where cosine distance is numerically meaningless and
    sklearn and mlrs would legitimately disagree.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((KNN_REG_N_TRAIN, KNN_REG_N_FEATURES)) * 3.0
    x += np.arange(KNN_REG_N_TRAIN)[:, None] * 0.01
    # Push every row off the origin: cosine distance is undefined at zero norm
    # and unstable near it.
    x += 5.0
    # One duplicated TRAIN pair, so a coincident query has two zero-distance
    # neighbours rather than one.
    x[7] = x[2]

    xq = rng.standard_normal((KNN_REG_N_QUERY, KNN_REG_N_FEATURES)) * 3.0 + 5.0
    for i, q in enumerate(KNN_REG_COINCIDENT_QUERIES):
        xq[q] = x[2 if i == 0 else 15]

    y = x @ rng.standard_normal(KNN_REG_N_FEATURES) + 0.5
    y_multi = np.column_stack(
        [y, x @ rng.standard_normal(KNN_REG_N_FEATURES) - 1.25]
    )
    return x, xq, y, y_multi


def gen_knn_regressor_params(seed: int = SEED, dtype=np.float32) -> str:
    """Generate the KNeighborsRegressor full-parameter oracle (KNN-REG-PARAMS).

    One fixture, every hyperparameter combination the DEVICE path serves, all
    from `sklearn.neighbors.KNeighborsRegressor(algorithm='brute', ...)`:

      * `weights` in {uniform, distance}
      * `metric` in {euclidean, manhattan, chebyshev, minkowski(p=3), cosine}
      * a multi-output target under both weightings

    Array names are `predict_<metric>_<weights>`, plus
    `predict_multi_<weights>` for the 2-column target and
    `distances_manhattan` / `indices_manhattan` for the `kneighbors` surface
    under a NON-default metric (the one place a metric bug could hide behind a
    correct `predict`).

    The callable-`weights` and callable-`metric` paths are deliberately NOT
    stored here: they are host-side reimplementations of sklearn's own host
    code, so a committed oracle would only pin numpy against numpy. The Python
    tests exercise them against a live sklearn instead, which is the comparison
    that can actually fail.
    """
    from sklearn.neighbors import KNeighborsRegressor

    x, xq, y, y_multi = _knn_reg_data(seed)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    out = {
        "X": c(x),
        "Xq": c(xq),
        "y": c(y),
        "y_multi": c(y_multi),
        "k": c([KNN_REG_K]),
        "p": c([KNN_REG_P]),
    }

    metrics = {
        "euclidean": dict(metric="euclidean"),
        "manhattan": dict(metric="manhattan"),
        "chebyshev": dict(metric="chebyshev"),
        "minkowski": dict(metric="minkowski", p=KNN_REG_P),
        "cosine": dict(metric="cosine"),
    }
    for name, kw in metrics.items():
        for weights in ("uniform", "distance"):
            reg = KNeighborsRegressor(
                n_neighbors=KNN_REG_K,
                algorithm="brute",
                weights=weights,
                **kw,
            ).fit(x, y)
            out[f"predict_{name}_{weights}"] = c(reg.predict(xq))

    for weights in ("uniform", "distance"):
        reg = KNeighborsRegressor(
            n_neighbors=KNN_REG_K,
            algorithm="brute",
            weights=weights,
            metric="euclidean",
        ).fit(x, y_multi)
        out[f"predict_multi_{weights}"] = c(reg.predict(xq))

    nn = KNeighborsRegressor(
        n_neighbors=KNN_REG_K, algorithm="brute", metric="manhattan"
    ).fit(x, y)
    dist, idx = nn.kneighbors(xq)
    out["distances_manhattan"] = c(dist)
    out["indices_manhattan"] = c(idx)

    # --- Every STRING value of `metric`, including the aliases (KNN-REG-PARAMS
    #     oracle completion). The block above covers the five distinct distance
    #     FUNCTIONS; this one covers the nine strings a user can type, so
    #     `metric='l1'` is gated by sklearn-under-`'l1'` rather than by the
    #     assumption that mlrs folds it onto `manhattan` correctly.
    #
    #     Generated under `algorithm='auto'` — the default, and the only value
    #     that accepts all nine (`'infinity'` is tree-only, `'cosine'` is
    #     brute-only; see `_ALGORITHM_VALID_METRICS` in the shim).
    for metric in KNN_REG_METRIC_STRINGS:
        for weights in ("uniform", "distance"):
            reg = KNeighborsRegressor(
                n_neighbors=KNN_REG_K,
                algorithm="auto",
                weights=weights,
                metric=metric,
            ).fit(x, y)
            out[f"alias_{metric}_{weights}"] = c(reg.predict(xq))

    # --- Every STRING value of `algorithm`, at the default metric.
    #
    #     mlrs runs brute force for all four, so these arrays gate that claim:
    #     `alg_kd_tree_*` is sklearn's K-D TREE answer, and mlrs's brute-force
    #     predict has to reproduce it. The design has one duplicated train pair,
    #     but both copies carry the same target (`y` is derived from `x` after
    #     the duplication), so the tie the two search strategies may break
    #     differently cannot move a prediction.
    for algorithm in KNN_REG_ALGORITHMS:
        for weights in ("uniform", "distance"):
            reg = KNeighborsRegressor(
                n_neighbors=KNN_REG_K,
                algorithm=algorithm,
                weights=weights,
            ).fit(x, y)
            out[f"alg_{algorithm}_{weights}"] = c(reg.predict(xq))

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"knn_reg_params_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(out_path, **out)
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
# HDBSCAN string-valued-parameter oracle (HDBS-PARAMS). One fixture carrying a
# sklearn label vector for EVERY string a user can pass to `metric=`,
# `algorithm=`, `cluster_selection_method=` and `store_centers=`.
#
# Why this is a SEPARATE fixture from `gen_hdbscan`:
#   * It is the PYTHON-boundary gate. The eleven `metric` strings collapse onto
#     six `Metric` enum values in Rust (`l2` IS `euclidean`, `cityblock`/`l1` ARE
#     `manhattan`, `infinity` IS `chebyshev`, `p` IS `minkowski`), so testing all
#     eleven only means something at the layer that does the resolving — the
#     shim. Replaying it there catches an alias wired to the wrong enum, which no
#     Rust-side test can see.
#   * It needs only sklearn, not the pinned `hdbscan` 0.8.44 GLOSH oracle, so it
#     regenerates in a plain sklearn-1.9.0 environment.
#   * n = 600 > `kdtree::KD_MIN_ROWS` (512) DELIBERATELY: below that threshold
#     `algorithm='auto'` never builds a tree and the four algorithm values would
#     agree vacuously. At 600 rows `auto` genuinely builds and calibrates one, so
#     "all four agree" is a real statement about the tree route.
HDB_PARAMS_N = 600
HDB_PARAMS_K = 4
HDB_PARAMS_D = 4
HDB_PARAMS_MCS = 15

# Every string sklearn accepts for `metric=` that mlrs's distance core supports.
# Grouped by the enum each resolves to, which is what the shim gate checks.
HDB_PARAM_METRICS = [
    "euclidean", "l2",
    "manhattan", "cityblock", "l1",
    "chebyshev", "infinity",
    "minkowski", "p",
    "cosine",
    "precomputed",
]
HDB_PARAM_ALGORITHMS = ["auto", "brute", "kd_tree", "ball_tree"]
HDB_PARAM_CSMS = ["eom", "leaf"]
HDB_PARAM_STORE = ["centroid", "medoid", "both"]


def gen_hdbscan_params(seed: int = SEED, dtype=np.float32) -> str:
    """Generate the HDBSCAN string-parameter oracle fixture (HDBS-PARAMS).

    Design: ``HDB_PARAMS_K`` well-separated isotropic blobs in
    ``HDB_PARAMS_D`` dimensions, plus the per-row 1e-3 offset the other HDBSCAN
    fixtures use so every MST edge weight is distinct and the labelling is
    tie-free under any stable rule (Pitfall 1 option 2). Separation is generous
    (centres on a scaled simplex) so the SAME partition survives every metric —
    otherwise a metric gate would be testing the fixture's fragility rather than
    the metric.

    Stores ``X`` (the ``n × d`` design), ``X_precomputed`` (its square Euclidean
    distance matrix, for the ``precomputed`` metric string) and one float-valued
    label vector per string value:

      * ``labels_metric_<name>``    — for each of ``HDB_PARAM_METRICS``
      * ``labels_algorithm_<name>`` — for each of ``HDB_PARAM_ALGORITHMS``
      * ``labels_csm_<name>``       — for each of ``HDB_PARAM_CSMS``
      * ``centroids_store`` / ``medoids_store`` — from ``store_centers='both'``

    plus ``minkowski_p`` and ``min_cluster_size`` as 1-element arrays so the
    replaying test needs no constant duplicated from this file.

    ASSERTS before writing that the design actually clusters (``k`` clusters
    found, not all-noise) — a fixture that degenerated to all-noise would make
    every downstream gate pass vacuously.

    Returns the path written.
    Filename: ``hdbscan_params_{dtype}_seed{seed}.npz``.
    """
    from sklearn.cluster import HDBSCAN as SkHDBSCAN
    from sklearn.metrics import pairwise_distances

    rng = np.random.default_rng(seed)
    # Centres far enough apart that the partition is metric-independent: a
    # scaled identity basis keeps every inter-centre distance equal under L1,
    # L2 and L-inf alike, and the 40x scale dwarfs the 0.5 within-blob spread.
    centers = 40.0 * np.eye(HDB_PARAMS_K, HDB_PARAMS_D)
    per = HDB_PARAMS_N // HDB_PARAMS_K
    x = np.vstack(
        [c + 0.5 * rng.standard_normal((per, HDB_PARAMS_D)) for c in centers]
    )
    # Tie-free MST edges (the `_hdbscan_blob_design` idiom).
    x = x + np.arange(x.shape[0])[:, None] * 1e-3
    dist = pairwise_distances(x, metric="euclidean")

    def labels(**over):
        x_in = over.pop("_x", x)
        kw = dict(min_cluster_size=HDB_PARAMS_MCS, copy=True)
        kw.update(over)
        return SkHDBSCAN(**kw).fit(x_in).labels_

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    save_kw = {
        "X": c(x),
        "X_precomputed": c(dist),
        "minkowski_p": c([HDB_MINKOWSKI_P]),
        "min_cluster_size": c([HDB_PARAMS_MCS]),
    }

    for met in HDB_PARAM_METRICS:
        over = {"metric": met}
        if met in ("minkowski", "p"):
            over["metric_params"] = {"p": HDB_MINKOWSKI_P}
        if met == "precomputed":
            over["_x"] = dist
        save_kw[f"labels_metric_{met}"] = c(labels(**over))

    for alg in HDB_PARAM_ALGORITHMS:
        save_kw[f"labels_algorithm_{alg}"] = c(labels(algorithm=alg))

    for csm in HDB_PARAM_CSMS:
        save_kw[f"labels_csm_{csm}"] = c(labels(cluster_selection_method=csm))

    fitted = SkHDBSCAN(
        min_cluster_size=HDB_PARAMS_MCS, copy=True, store_centers="both"
    ).fit(x)
    save_kw["centroids_store"] = c(fitted.centroids_)
    save_kw["medoids_store"] = c(fitted.medoids_)

    # The fixture must actually cluster — an all-noise design would let every
    # gate below pass without testing anything.
    base = save_kw["labels_metric_euclidean"].astype(np.int64)
    found = len(set(base.tolist()) - {-1})
    assert found == HDB_PARAMS_K, (
        f"hdbscan params design must yield {HDB_PARAMS_K} clusters, got {found}"
    )
    assert save_kw["centroids_store"].shape == (HDB_PARAMS_K, HDB_PARAMS_D), (
        f"store_centers='both' must yield one centroid row per cluster, got "
        f"{save_kw['centroids_store'].shape}"
    )
    # Every metric string must reproduce that same partition — the separation
    # above is chosen to guarantee it, and a fixture where it stopped holding
    # would silently weaken the per-metric gate into "whatever this metric did".
    for met in HDB_PARAM_METRICS:
        got = save_kw[f"labels_metric_{met}"].astype(np.int64)
        assert len(set(got.tolist()) - {-1}) == HDB_PARAMS_K, (
            f"metric {met!r} did not recover {HDB_PARAMS_K} clusters — the design "
            f"is no longer metric-independent"
        )

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"hdbscan_params_{dtype_tag}_seed{seed}.npz"
    )
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


def gen_linear_regression_large(seed: int = SEED, dtype=np.float32) -> str:
    """Generate the large-`n_samples` LinearRegression fixture (LINEAR-01),
    exercising the `fit_gram_eig` Gram+eig path (`n_samples > 256`, the
    direct-SVD single-cube kernel's row cap).

    Same shape as [`gen_linear_regression`] (full-rank + near-collinear,
    ``X``/``y``/``coef``/``intercept``/``X_test``/``y_pred``/``X_coll``/
    ``y_coll``/``coef_col``/``intercept_col``) at
    ``LIN_LARGE_N_SAMPLES × LIN_LARGE_N_FEATURES``, so the Rust test can reuse
    the exact same fixture-consumption code as the small-fixture oracle test
    with only the geometry constants swapped. sklearn's ``LinearRegression``
    is still ``scipy.linalg.lstsq`` (gelsd / SVD) — the reference the
    Gram+eig path must match to 1e-5 despite forming normal equations
    internally (D-02 numerical tradeoff, see `linear_regression.rs`
    `fit_gram_eig` docs). Returns the absolute path written.
    """
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(seed)
    n, p = LIN_LARGE_N_SAMPLES, LIN_LARGE_N_FEATURES
    x = rng.standard_normal((n, p))
    true_coef = rng.standard_normal(p)
    y = x @ true_coef + 0.5 + 0.01 * rng.standard_normal(n)

    reg = LinearRegression(fit_intercept=True).fit(x, y)
    x_test = rng.standard_normal((LIN_TEST_SAMPLES, p))
    y_pred = reg.predict(x_test)

    # NEAR-COLLINEAR case: duplicate column 0 into column 2 with a tiny
    # perturbation, mirroring `gen_linear_regression`'s small-fixture cutoff
    # case, but at the large-N geometry this fixture targets.
    x_coll = x.copy()
    x_coll[:, 2] = x_coll[:, 0] + 1e-7 * rng.standard_normal(n)
    y_coll = x_coll @ true_coef + 0.5 + 0.01 * rng.standard_normal(n)
    reg_coll = LinearRegression(fit_intercept=True).fit(x_coll, y_coll)

    def c(arr):
        return np.asarray(arr).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"linear_regression_large_{dtype_tag}_seed{seed}.npz"
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


# --- Ridge FULL parameter surface (LINEAR-02) ------------------------------ #
# Geometry for the `gen_ridge_params` fixture. Deliberately larger and
# better-conditioned than the LIN_* alpha-sweep fixture: the stochastic solvers
# (`sag`/`saga`) need enough samples for their averaged-gradient table to be
# meaningful, and every solver must land on the SAME minimizer for the
# cross-solver agreement assert to be a real gate rather than a tautology.
RIDGE_PARAMS_N_SAMPLES, RIDGE_PARAMS_N_FEATURES = 40, 5
# alpha > 0 makes the objective STRICTLY convex, hence a unique minimizer that
# every solver must reach — the premise of comparing eight solvers to one
# reference. (alpha = 0 is covered by the LinearRegression fixtures.)
RIDGE_PARAMS_ALPHA = 1.0
# Both sides are run to a TIGHT tolerance so the comparison is against the
# CONVERGED optimum, not against scipy's/sklearn's particular early-stop point:
# at sklearn's own default tol=1e-4 the iterative solvers stop while still
# ~1e-4 away, which would make a 1e-5 oracle gate meaningless.
RIDGE_PARAMS_TOL = 1e-10
RIDGE_PARAMS_MAX_ITER = 100000


def gen_ridge_params(seed: int = SEED, dtype=np.float32) -> str:
    """Generate the Ridge full-parameter-surface fixture (LINEAR-02).

    Covers every ``sklearn.linear_model.Ridge`` ctor parameter that changes the
    fit — all eight ``solver`` values, ``fit_intercept``, ``positive``, and
    ``fit(..., sample_weight=...)`` — plus the ``solver_`` attribute each case
    resolves to. ``copy_X`` is excluded on purpose: it cannot change the result
    (it only controls whether sklearn centers in place), and ``max_iter``/``tol``
    are exercised as the CONVERGENCE knobs they are rather than as separate
    cases (see ``RIDGE_PARAMS_TOL``).

    Each case is stored as ``coef_<case>`` / ``intercept_<case>`` /
    ``solver_<case>``. The reference is fitted on the design AFTER the
    round-trip through the fixture dtype, so an f32 fixture's reference is the
    answer for the exact bytes the test feeds back in (the f32/f64 split is
    then purely about the solver's working precision).

    Requires ``scikit-learn==1.9.0``.
    """
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Ridge

    n, d = RIDGE_PARAMS_N_SAMPLES, RIDGE_PARAMS_N_FEATURES
    rng = np.random.default_rng(seed + 77)
    # Round-trip through the fixture dtype BEFORE fitting (see the docstring).
    x = rng.standard_normal((n, d)).astype(dtype).astype(np.float64)
    true_coef = rng.standard_normal(d)
    y = (x @ true_coef + 0.5 + 0.01 * rng.standard_normal(n)).astype(dtype).astype(np.float64)
    # Strictly-positive, non-uniform weights so the weighted cases genuinely
    # differ from the unweighted ones.
    sw = rng.uniform(0.25, 3.0, size=n).astype(dtype).astype(np.float64)

    common = dict(
        alpha=RIDGE_PARAMS_ALPHA,
        tol=RIDGE_PARAMS_TOL,
        max_iter=RIDGE_PARAMS_MAX_ITER,
    )
    # (case name, ctor kwargs, use_sample_weight)
    cases = [
        # --- the eight solver values, fit_intercept=True, unweighted --------- #
        ("auto", dict(solver="auto"), False),
        ("cholesky", dict(solver="cholesky"), False),
        ("svd", dict(solver="svd"), False),
        ("lsqr", dict(solver="lsqr"), False),
        ("sparse_cg", dict(solver="sparse_cg"), False),
        ("sag", dict(solver="sag", random_state=0), False),
        ("saga", dict(solver="saga", random_state=0), False),
        # --- positive=True: 'lbfgs' explicitly, and via 'auto' -------------- #
        ("lbfgs_pos", dict(solver="lbfgs", positive=True), False),
        ("auto_pos", dict(solver="auto", positive=True), False),
        # --- fit_intercept=False -------------------------------------------- #
        ("cholesky_noint", dict(solver="cholesky", fit_intercept=False), False),
        ("svd_noint", dict(solver="svd", fit_intercept=False), False),
        ("lsqr_noint", dict(solver="lsqr", fit_intercept=False), False),
        ("sag_noint", dict(solver="sag", fit_intercept=False, random_state=0), False),
        ("lbfgs_pos_noint", dict(solver="lbfgs", positive=True, fit_intercept=False), False),
        # --- sample_weight (the `_rescale_data` regime AND the sag/saga
        #     direct-weight regime, which sklearn splits on) ------------------ #
        ("cholesky_sw", dict(solver="cholesky"), True),
        ("svd_sw", dict(solver="svd"), True),
        ("lsqr_sw", dict(solver="lsqr"), True),
        ("sparse_cg_sw", dict(solver="sparse_cg"), True),
        ("sag_sw", dict(solver="sag", random_state=0), True),
        ("saga_sw", dict(solver="saga", random_state=0), True),
        ("lbfgs_pos_sw", dict(solver="lbfgs", positive=True), True),
        ("cholesky_noint_sw", dict(solver="cholesky", fit_intercept=False), True),
    ]

    def c(arr):
        return np.asarray(arr).astype(dtype)

    out = {"X": c(x), "y": c(y), "sample_weight": c(sw)}
    solver_names = []
    for name, kwargs, use_sw in cases:
        est = Ridge(**{**common, **kwargs})
        with warnings.catch_warnings():
            # A ConvergenceWarning here would mean the reference itself is not
            # at the optimum — turn it into an error so a bad fixture cannot be
            # committed silently.
            warnings.simplefilter("error", ConvergenceWarning)
            est.fit(x, y, sample_weight=sw if use_sw else None)
        out[f"coef_{name}"] = c(est.coef_)
        out[f"intercept_{name}"] = c([est.intercept_])
        solver_names.append(f"{name}={est.solver_}")

    # Every unweighted, intercept-fitting, unconstrained solver must agree with
    # `cholesky` — the premise of the whole fixture. Asserted HERE (at
    # generation) so a sklearn upgrade that breaks it is caught in the script
    # rather than showing up as an unexplained Rust test failure.
    ref = out["coef_cholesky"].astype(np.float64)
    for name in ("auto", "svd", "lsqr", "sparse_cg", "sag", "saga"):
        got = out[f"coef_{name}"].astype(np.float64)
        assert np.allclose(got, ref, atol=1e-7, rtol=1e-7), (
            f"gen_ridge_params: solver '{name}' disagrees with 'cholesky' "
            f"({got} vs {ref}) — the converged-optimum premise does not hold"
        )
    # The non-negativity constraint must actually BIND (otherwise the positive
    # cases would silently be the same test as the unconstrained ones).
    assert (out["coef_lbfgs_pos"] >= 0).all(), "positive fit produced a negative coef_"
    assert (ref < 0).any(), (
        "gen_ridge_params: the unconstrained solution is already all-positive, "
        "so `positive=True` is not exercised — reseed the fixture"
    )

    # sklearn's `solver_` per case, asserted HERE rather than shipped in the
    # archive: `mlrs_core::load_npz` rejects any array that is not a 4- or
    # 8-byte float, so a string array would break the whole fixture load. The
    # Rust/Python tests carry the same expectations as literals.
    expected_resolution = {
        "auto": "cholesky",
        "cholesky": "cholesky",
        "svd": "svd",
        "lsqr": "lsqr",
        "sparse_cg": "sparse_cg",
        "sag": "sag",
        "saga": "saga",
        "lbfgs_pos": "lbfgs",
        "auto_pos": "lbfgs",
    }
    resolved = dict(pair.split("=", 1) for pair in solver_names)
    for case, want in expected_resolution.items():
        assert resolved[case] == want, (
            f"gen_ridge_params: sklearn resolved solver_ for '{case}' to "
            f"'{resolved[case]}', expected '{want}' — the auto-dispatch "
            f"assumption in ridge.rs::RidgeSolver::resolve has drifted"
        )

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"ridge_params_{dtype_tag}_seed{seed}.npz")
    out["alpha"] = c([RIDGE_PARAMS_ALPHA])
    # Every array in an oracle .npz must be float32/float64 (the loader's
    # contract), so max_iter ships as a float and the reader casts it back.
    out["tol"] = np.asarray([RIDGE_PARAMS_TOL], dtype=np.float64)
    out["max_iter"] = np.asarray([float(RIDGE_PARAMS_MAX_ITER)], dtype=np.float64)
    np.savez(out_path, **out)
    return out_path


# --- RidgeClassifier FULL parameter surface (LINEAR-07) -------------------- #


def gen_ridge_classifier(seed: int = SEED, dtype=np.float32, multiclass: bool = False) -> str:
    """Generate the RidgeClassifier full-parameter-surface fixture (LINEAR-07).

    Covers every ``sklearn.linear_model.RidgeClassifier`` ctor parameter that
    changes the fit — the ``solver`` family (via the shared ``Ridge`` normal
    equations), ``fit_intercept``, ``positive``, ``class_weight`` (``None`` /
    ``'balanced'`` / a PARTIAL dict, to exercise the "class absent from the
    dict keeps weight 1.0" default-fill) and ``fit(..., sample_weight=...)`` —
    for both a binary (2-class) and multiclass (3-class) target, reusing the
    tight ``tol``/``max_iter`` from ``gen_ridge_params`` so every iterative
    solver lands on the SAME converged optimum as ``cholesky``.

    Stores ``coef_<case>`` (``atleast_2d``, so binary is `(1, d)`),
    ``intercept_<case>``, ``predict_<case>`` and ``decision_<case>``
    (reshaped to `(n_test, -1)`, so binary is `(n_test, 1)`). Requires
    ``scikit-learn==1.9.0``.
    """
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import RidgeClassifier

    n_classes = 3 if multiclass else 2
    n, d = (90, 6) if multiclass else (60, 5)
    n_test_per_class = 4
    rng = np.random.default_rng(seed + 133)
    centers = rng.standard_normal((n_classes, d)) * 4.0
    per = n // n_classes
    x = np.vstack(
        [centers[k] + rng.standard_normal((per, d)) for k in range(n_classes)]
    ).astype(dtype).astype(np.float64)
    y = np.concatenate([np.full(per, k) for k in range(n_classes)])
    n = x.shape[0]
    xq = np.vstack(
        [centers[k] + rng.standard_normal((n_test_per_class, d)) for k in range(n_classes)]
    ).astype(dtype).astype(np.float64)

    # Strictly-positive, non-uniform weights (the `gen_ridge_params` precedent)
    # so the weighted cases genuinely differ from the unweighted ones.
    sw = rng.uniform(0.25, 3.0, size=n).astype(dtype).astype(np.float64)
    # A PARTIAL dict — only class 0 is named — to exercise sklearn's
    # "classes absent from class_weight keep weight 1.0" fill rule.
    cw_partial = {0: 2.5}

    common = dict(alpha=RIDGE_PARAMS_ALPHA, tol=RIDGE_PARAMS_TOL, max_iter=RIDGE_PARAMS_MAX_ITER)
    # (case name, ctor kwargs, use_sample_weight)
    cases = [
        ("auto", dict(solver="auto"), False),
        ("cholesky", dict(solver="cholesky"), False),
        ("svd", dict(solver="svd"), False),
        ("lsqr", dict(solver="lsqr"), False),
        ("sparse_cg", dict(solver="sparse_cg"), False),
        ("sag", dict(solver="sag", random_state=0), False),
        ("saga", dict(solver="saga", random_state=0), False),
        ("lbfgs_pos", dict(solver="lbfgs", positive=True), False),
        ("cholesky_noint", dict(solver="cholesky", fit_intercept=False), False),
        ("cholesky_balanced", dict(solver="cholesky", class_weight="balanced"), False),
        ("cholesky_dict_partial", dict(solver="cholesky", class_weight=cw_partial), False),
        ("cholesky_sw", dict(solver="cholesky"), True),
        ("cholesky_sw_balanced", dict(solver="cholesky", class_weight="balanced"), True),
    ]

    def c(arr):
        # `ascontiguousarray` BEFORE the dtype cast: sklearn's multi-target
        # `RidgeClassifier.coef_` is `Ridge`'s `linalg.solve(...).T`, an
        # F-CONTIGUOUS view (`.astype`'s default `order='K'` preserves that
        # layout), and `np.savez` records the array's OWN memory order in the
        # `.npy` header. The Rust loader (`npyz`) reads the flat byte buffer
        # assuming row-major, so a Fortran-ordered array round-trips
        # TRANSPOSED — this bites `coef_`, not the 1-D arrays, which is why no
        # earlier generator needed it.
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    out = {"X": c(x), "Xq": c(xq), "y": c(y), "sample_weight": c(sw)}
    solver_names = []
    coefs = {}
    for name, kwargs, use_sw in cases:
        est = RidgeClassifier(**{**common, **kwargs})
        with warnings.catch_warnings():
            # A ConvergenceWarning here would mean the reference itself is not
            # at the optimum — turn it into an error so a bad fixture cannot be
            # committed silently.
            warnings.simplefilter("error", ConvergenceWarning)
            est.fit(x, y, sample_weight=sw if use_sw else None)
        coef2d = np.atleast_2d(est.coef_)
        out[f"coef_{name}"] = c(coef2d)
        out[f"intercept_{name}"] = c(np.atleast_1d(est.intercept_))
        out[f"predict_{name}"] = c(est.predict(xq))
        out[f"decision_{name}"] = c(est.decision_function(xq).reshape(len(xq), -1))
        solver_names.append(f"{name}={est.solver_}")
        coefs[name] = coef2d.astype(np.float64)

    # Every unweighted, intercept-fitting, unconstrained solver must agree with
    # `cholesky` — the premise of the whole fixture (the `gen_ridge_params`
    # precedent, now over the multi-output `{-1,+1}` target).
    ref = coefs["cholesky"]
    for name in ("auto", "svd", "lsqr", "sparse_cg", "sag", "saga"):
        assert np.allclose(coefs[name], ref, atol=1e-7, rtol=1e-7), (
            f"gen_ridge_classifier: solver '{name}' disagrees with 'cholesky' "
            f"({coefs[name]} vs {ref})"
        )
    assert (coefs["lbfgs_pos"] >= 0).all(), "positive fit produced a negative coef_"
    assert (ref < 0).any(), (
        "gen_ridge_classifier: the unconstrained solution is already all-positive, "
        "so `positive=True` is not exercised — reseed the fixture"
    )

    expected_resolution = {
        "auto": "cholesky",
        "cholesky": "cholesky",
        "svd": "svd",
        "lsqr": "lsqr",
        "sparse_cg": "sparse_cg",
        "sag": "sag",
        "saga": "saga",
        "lbfgs_pos": "lbfgs",
    }
    resolved = dict(pair.split("=", 1) for pair in solver_names)
    for case, want in expected_resolution.items():
        assert resolved[case] == want, (
            f"gen_ridge_classifier: sklearn resolved solver_ for '{case}' to "
            f"'{resolved[case]}', expected '{want}'"
        )

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    kind = "multi" if multiclass else "binary"
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"ridge_classifier_{kind}_{dtype_tag}_seed{seed}.npz")
    out["alpha"] = c([RIDGE_PARAMS_ALPHA])
    # Every array in an oracle .npz must be float32/float64 (the loader's
    # contract), so max_iter/n_classes ship as floats and the reader casts back.
    out["tol"] = np.asarray([RIDGE_PARAMS_TOL], dtype=np.float64)
    out["max_iter"] = np.asarray([float(RIDGE_PARAMS_MAX_ITER)], dtype=np.float64)
    out["n_classes"] = np.asarray([float(n_classes)], dtype=np.float64)
    out["cw_partial_label"] = np.asarray([float(next(iter(cw_partial)))], dtype=np.float64)
    out["cw_partial_weight"] = np.asarray([float(next(iter(cw_partial.values())))], dtype=np.float64)
    np.savez(out_path, **out)
    return out_path


# --- BayesianRidge FULL parameter surface (LINEAR-06) ---------------------- #
# Two geometries, because sklearn's `_update_coef_` and `_log_marginal_likelihood`
# each branch on `n_samples > n_features` and its `sigma_` uses a FULL-matrix SVD
# only in the wide case. TALL is the ordinary regime; WIDE is rank-deficient by
# construction (rank == n_samples), which is what exercises the zero-padded
# `eigen_vals_full` and the null-space directions that survive only in `sigma_`.
BAYES_TALL_N_SAMPLES, BAYES_TALL_N_FEATURES = 60, 8
BAYES_WIDE_N_SAMPLES, BAYES_WIDE_N_FEATURES = 6, 10
# Rows held out of the fit, for `predict` and `predict(return_std=True)`.
BAYES_N_TEST = 7


def gen_bayesian_ridge(seed: int = SEED, dtype=np.float32) -> str:
    """Generate the BayesianRidge full-parameter-surface fixture (LINEAR-06).

    Covers every ``sklearn.linear_model.BayesianRidge`` ctor parameter that
    changes the fit — ``max_iter``, ``tol``, the four Gamma hyperpriors
    ``alpha_1``/``alpha_2``/``lambda_1``/``lambda_2``, ``alpha_init``,
    ``lambda_init``, ``compute_score``, ``fit_intercept`` — plus
    ``fit(..., sample_weight=...)`` and both `n_samples ≶ n_features`
    branches. ``copy_X`` and ``verbose`` are excluded on purpose: neither can
    change the result (``copy_X`` only controls whether sklearn centers in place,
    ``verbose`` only prints).

    Each case ships every fitted attribute the estimator exposes:
    ``coef_`` / ``intercept_`` / ``alpha_`` / ``lambda_`` / ``sigma_`` /
    ``n_iter_``, plus ``scores_`` for the ``compute_score`` cases and the
    held-out ``predict`` mean and ``return_std=True`` deviation for the default
    case. Gating ``alpha_``/``lambda_``/``n_iter_`` — not just ``coef_`` — is
    what makes this a test of the EVIDENCE ITERATION rather than of a ridge
    solve at some penalty: a wrong update rule that happens to converge to a
    similar penalty would still miss the iteration count and the precisions.

    The reference is fitted on the design AFTER the round-trip through the
    fixture dtype, so an f32 fixture's reference is the answer for the exact
    bytes the test feeds back in.

    Requires ``scikit-learn==1.9.0``.
    """
    from sklearn.linear_model import BayesianRidge

    rng = np.random.default_rng(seed + 106)

    def design(n, d):
        # Round-trip through the fixture dtype BEFORE fitting (see docstring).
        x = rng.standard_normal((n, d)).astype(dtype).astype(np.float64)
        w = rng.standard_normal(d)
        # Noise at 0.5 (not 0.1): with a near-perfect fit the evidence ratio
        # `lambda_/alpha_` lands orders of magnitude below the Gram's spectrum,
        # every case collapses onto the OLS solution, and the hyperprior cases
        # stop being distinguishable from the default (the premise assert below
        # catches exactly that).
        y = (x @ w + 0.5 + 0.5 * rng.standard_normal(n)).astype(dtype).astype(np.float64)
        return x, y

    nt, dt_ = BAYES_TALL_N_SAMPLES, BAYES_TALL_N_FEATURES
    nw, dw = BAYES_WIDE_N_SAMPLES, BAYES_WIDE_N_FEATURES
    x, y = design(nt, dt_)
    xw, yw = design(nw, dw)
    xtest = rng.standard_normal((BAYES_N_TEST, dt_)).astype(dtype).astype(np.float64)
    # Strictly-positive, non-uniform weights so the weighted cases genuinely
    # differ from the unweighted ones.
    sw = rng.uniform(0.25, 3.0, size=nt).astype(dtype).astype(np.float64)

    # (case name, ctor kwargs, use_sample_weight, wide?)
    cases = [
        # --- defaults, and the intercept switch ------------------------------ #
        ("default", dict(), False, False),
        ("noint", dict(fit_intercept=False), False, False),
        # --- the convergence knobs. `max_iter=1` cannot reach the `iter_ != 0`
        #     convergence test at all, so it pins the non-converged path; the
        #     tight/loose tol pair pins `n_iter_` from both sides. ----------- #
        ("maxiter1", dict(max_iter=1), False, False),
        ("maxiter5", dict(max_iter=5), False, False),
        ("tol_tight", dict(tol=1e-8, max_iter=1000), False, False),
        ("tol_loose", dict(tol=1e-1), False, False),
        # --- the four Gamma hyperpriors, moved far off 1e-6 so they BITE ---- #
        ("priors", dict(alpha_1=1.0, alpha_2=5.0, lambda_1=50.0, lambda_2=1.0), False, False),
        ("priors_zero", dict(alpha_1=0.0, alpha_2=0.0, lambda_1=0.0, lambda_2=0.0), False, False),
        # --- explicit initial precisions (skips sklearn's 1/(var(y)+eps)) --- #
        ("init", dict(alpha_init=2.5, lambda_init=0.1), False, False),
        ("init_alpha_only", dict(alpha_init=10.0), False, False),
        # --- compute_score: the log-marginal-likelihood trace ---------------- #
        ("score", dict(compute_score=True), False, False),
        ("score_maxiter3", dict(compute_score=True, max_iter=3), False, False),
        ("score_noint", dict(compute_score=True, fit_intercept=False), False, False),
        # --- sample_weight (sklearn's `sw_sum` enters BOTH the alpha update
        #     and the log marginal likelihood, so it is scored too) ---------- #
        ("sw", dict(), True, False),
        ("sw_noint", dict(fit_intercept=False), True, False),
        ("sw_score", dict(compute_score=True), True, False),
        # --- n_samples < n_features: the `U`-branch posterior mean, the padded
        #     `logdet_sigma`, and the full-basis `sigma_` ------------------- #
        ("wide", dict(), False, True),
        ("wide_noint", dict(fit_intercept=False), False, True),
        ("wide_score", dict(compute_score=True), False, True),
    ]

    def c(arr):
        return np.asarray(arr).astype(dtype)

    out = {
        "X": c(x),
        "y": c(y),
        "sample_weight": c(sw),
        "X_test": c(xtest),
        "X_wide": c(xw),
        "y_wide": c(yw),
    }
    for name, kwargs, use_sw, wide in cases:
        xx, yy = (xw, yw) if wide else (x, y)
        est = BayesianRidge(**kwargs)
        est.fit(xx, yy, sample_weight=sw if use_sw else None)
        out[f"coef_{name}"] = c(est.coef_)
        out[f"intercept_{name}"] = c([est.intercept_])
        # The precisions and the score trace are compared in f64 whatever the
        # design's dtype: they are scalars derived from an f64 accumulation on
        # both sides, and rounding them to f32 would hide a real drift in the
        # evidence iteration behind the storage format.
        out[f"alpha_{name}"] = np.asarray([est.alpha_], dtype=np.float64)
        out[f"lambda_{name}"] = np.asarray([est.lambda_], dtype=np.float64)
        out[f"sigma_{name}"] = np.asarray(est.sigma_, dtype=np.float64).ravel()
        # Every array in an oracle .npz must be float32/float64 (the loader's
        # contract), so the iteration count ships as a float and the reader
        # casts it back.
        out[f"n_iter_{name}"] = np.asarray([float(est.n_iter_)], dtype=np.float64)
        if kwargs.get("compute_score"):
            out[f"scores_{name}"] = np.asarray(est.scores_, dtype=np.float64)

    # Held-out prediction for the default case, mean AND predictive std, so the
    # `sigma_` gate is not purely an attribute compare — `predict(return_std)` is
    # the only consumer that turns `sigma_` back into an observable.
    est = BayesianRidge().fit(x, y)
    mean, std = est.predict(xtest, return_std=True)
    out["pred_default"] = c(mean)
    out["predstd_default"] = np.asarray(std, dtype=np.float64)

    # --- Premises asserted HERE (at generation) so a sklearn upgrade that
    #     breaks one is caught in this script rather than showing up as an
    #     unexplained Rust failure. ---
    assert out["n_iter_maxiter1"][0] == 1.0, "max_iter=1 must report n_iter_ == 1"
    assert out["n_iter_tol_loose"][0] < out["n_iter_tol_tight"][0], (
        "gen_bayesian_ridge: the loose-tol case did not stop earlier than the "
        "tight-tol one, so `tol` is not being exercised — reseed the fixture"
    )
    assert out["n_iter_default"][0] > 1, (
        "gen_bayesian_ridge: the default case converged in ONE iteration, so the "
        "evidence loop is untested — reseed the fixture"
    )
    assert len(out["scores_score"]) == int(out["n_iter_score"][0]) + 1, (
        "gen_bayesian_ridge: scores_ is not n_iter_ + 1 long — sklearn's "
        "trailing post-loop score append has changed shape"
    )
    assert not np.allclose(out["coef_default"], out["coef_priors"], atol=1e-6), (
        "gen_bayesian_ridge: the Gamma hyperpriors did not change the fit, so "
        "alpha_1/alpha_2/lambda_1/lambda_2 are not exercised"
    )
    assert not np.allclose(out["coef_default"], out["coef_sw"], atol=1e-6), (
        "gen_bayesian_ridge: sample_weight did not change the fit"
    )
    assert np.linalg.matrix_rank(xw) == nw, (
        "gen_bayesian_ridge: the wide design is not rank-deficient, so the "
        "n_samples < n_features branch is not exercised"
    )

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"bayesian_ridge_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(out_path, **out)
    return out_path


# --- HuberRegressor FULL parameter surface (HUBER-01) ---------------------- #
# One geometry is enough: unlike BayesianRidge, nothing in the Huber solve
# branches on `n_samples <=> n_features`. What DOES have to be engineered into
# the design is the outlier structure — a fixture with no gross outliers makes
# every `epsilon` collapse onto the same least-squares answer and stops testing
# the estimator at all (the premise asserts below catch exactly that).
HUBER_N_SAMPLES, HUBER_N_FEATURES = 240, 6
HUBER_N_TEST = 9
# Fraction of rows given a large additive shock in `y`. 8 % is enough that the
# default epsilon=1.35 classifies a double-digit number of samples as outliers
# while the quadratic core still has most of the data.
HUBER_OUTLIER_FRAC = 0.08


def _huber_objective(params, x, y, epsilon, alpha, sample_weight):
    """sklearn's Huber objective at `params = [coef…, intercept, sigma]`.

    The layout here ALWAYS carries an intercept slot, even for a
    ``fit_intercept=False`` fit where sklearn pins ``intercept_ = 0.0``: a zero
    intercept contributes nothing to either the residuals or the penalty, so one
    layout serves both and the caller never has to branch.

    Duplicated here (rather than imported from ``sklearn.linear_model._huber``)
    because the fixture stores the loss as a GATE the Rust test compares against,
    and that gate has to keep meaning the same thing if sklearn refactors its
    private helper. It is the formula from the class docstring, and the premise
    assert below pins it against ``_huber_loss_and_gradient`` at generation time.
    """
    d = x.shape[1]
    w = params[:d]
    intercept = params[d]
    sigma = params[-1]
    r = y - x @ w - intercept
    a = np.abs(r)
    outlier = a > epsilon * sigma
    sw = sample_weight
    loss = sigma * np.sum(sw)
    loss += np.sum(sw[~outlier] * r[~outlier] ** 2) / sigma
    loss += 2.0 * epsilon * np.sum(sw[outlier] * a[outlier])
    loss -= sigma * epsilon**2 * np.sum(sw[outlier])
    loss += alpha * float(w @ w)
    return loss


def gen_huber(seed: int = SEED, dtype=np.float32) -> str:
    """Generate the HuberRegressor full-parameter-surface fixture (HUBER-01).

    Covers every ``sklearn.linear_model.HuberRegressor`` ctor parameter —
    ``epsilon``, ``max_iter``, ``alpha``, ``warm_start``, ``fit_intercept``,
    ``tol`` — plus ``fit(..., sample_weight=...)``. sklearn's Huber has NO
    string-valued parameter (every one is a float, an int or a bool), which the
    premise assert below PINS: if a future sklearn adds one (a ``solver=``, say),
    the generator fails here rather than the parameter silently going untested.

    ## The three case families, and why the split is forced
    scikit-learn hands the objective to ``scipy.optimize.minimize(method=
    "L-BFGS-B")`` passing ``tol`` as ``gtol`` but leaving ``factr`` at its
    ``1e7`` default, so every fit actually stops on the RELATIVE-f criterion
    ``Δf/max(|f|,1) ≤ 1e7·eps ≈ 2.2e-9`` — before the gradient test can fire.
    Two measured consequences drive this fixture's design:

    1. sklearn's returned parameters sit ~1e-6 … 6e-6 (ABSOLUTE) from the true
       minimizer, and ``tol`` cannot move that: sweeping it from 1e-5 to 1e-12
       leaves ``n_iter_`` and every attribute bit-identical. mlrs deliberately
       solves tighter (``ftol = 64·eps``), so the parameter agreement is bounded
       below by sklearn's own residual. ``huber_max_param_residual`` ships that
       measured residual so the Rust band is justified by a number rather than
       chosen, and the assert below fails if a scipy/sklearn upgrade grows it.
    2. A fit truncated by ``max_iter`` stops MID-TRAJECTORY, and two different
       L-BFGS implementations (scipy's Moré-Thuente line search vs the 05-06
       strong-Wolfe primitive) do not pass through the same intermediate points.
       Those cases are therefore CONTROL-FLOW cases: the fixture ships their
       ``n_iter_`` and their achieved loss, and the Rust test gates the cap and
       the loss ordering, NOT the coefficients.

    So: ``value`` cases (converged — every parameter that changes WHAT is
    optimized), ``ctrl`` cases (``max_iter``/``tol`` truncation — properties
    only), and the ``warm`` pair (two successive fits of one warm-started
    estimator).

    Every case ships ``coef_``/``intercept_``/``scale_``/``n_iter_``/
    ``outliers_`` plus ``loss_*``, the objective value sklearn ACHIEVED — the
    gate that is well-posed regardless of where either solver stopped.

    The reference is fitted on the design AFTER the round-trip through the
    fixture dtype, so an f32 fixture's reference is the answer for the exact
    bytes the test feeds back in.

    Requires ``scikit-learn==1.9.0``.
    """
    from sklearn.linear_model import HuberRegressor
    from sklearn.linear_model._huber import _huber_loss_and_gradient
    from sklearn.utils._param_validation import StrOptions

    rng = np.random.default_rng(seed + 135)
    n, d = HUBER_N_SAMPLES, HUBER_N_FEATURES

    # Round-trip through the fixture dtype BEFORE fitting (see docstring).
    x = rng.standard_normal((n, d)).astype(dtype).astype(np.float64)
    true_coef = rng.standard_normal(d)
    y = x @ true_coef + 1.5 + 0.4 * rng.standard_normal(n)
    # Gross outliers: a heavy one-sided-ish shock on a minority of rows. This is
    # the whole point of the estimator, and without it `epsilon` is inert.
    n_out = int(round(HUBER_OUTLIER_FRAC * n))
    out_idx = rng.choice(n, size=n_out, replace=False)
    y[out_idx] += 25.0 * rng.standard_normal(n_out) + 15.0
    y = y.astype(dtype).astype(np.float64)
    x_test = rng.standard_normal((HUBER_N_TEST, d)).astype(dtype).astype(np.float64)
    # Strictly-positive, non-uniform weights so the weighted cases genuinely
    # differ from the unweighted ones.
    sw = rng.uniform(0.25, 3.0, size=n).astype(dtype).astype(np.float64)
    ones = np.ones(n)

    # (case name, ctor kwargs, use_sample_weight)
    #
    # CONVERGED value cases — max_iter is raised to 1000 everywhere so the stop
    # is sklearn's `factr` plateau (i.e. the optimum) rather than the cap, which
    # is what makes a coefficient comparison well-posed at all.
    value_cases = [
        ("default", dict(), False),
        ("noint", dict(fit_intercept=False), False),
        # --- epsilon: the outlier cut-off on the SCALED residual. 1.05 sits just
        #     inside sklearn's `[1, inf)` bound (most robust NON-degenerate
        #     setting — see the `eps1` control case for why exactly 1.0 is not
        #     one); 10.0 pushes every sample into the quadratic core, i.e. onto
        #     plain least squares with a fitted scale. --------------------- #
        ("eps105", dict(epsilon=1.05), False),
        ("eps2", dict(epsilon=2.5), False),
        ("eps10", dict(epsilon=10.0), False),
        ("eps105_noint", dict(epsilon=1.05, fit_intercept=False), False),
        # --- alpha: the `alpha·‖w‖²` ridge penalty (NOT ½α‖w‖²). 0 is the
        #     unpenalized objective; 100 shrinks the coefficients visibly. ---- #
        ("alpha0", dict(alpha=0.0), False),
        ("alpha1", dict(alpha=1.0), False),
        ("alpha100", dict(alpha=100.0), False),
        # --- tol: sklearn's `factr` binds first, so a TIGHT tol is inert. It is
        #     included precisely to pin that (the assert below fails if a
        #     sklearn upgrade ever makes gtol the binding stop again). -------- #
        ("tol_tight", dict(tol=1e-12), False),
        # --- sample_weight: enters every term, including `σ·Σswᵢ`. ---------- #
        ("sw", dict(), True),
        ("sw_noint", dict(fit_intercept=False), True),
        ("sw_eps105", dict(epsilon=1.05), True),
        ("sw_alpha1", dict(alpha=1.0), True),
    ]
    # CONTROL-FLOW cases — truncated mid-trajectory, so properties only.
    #
    # `eps1` is here rather than in `value_cases` for a MATHEMATICAL reason, not
    # a numerical one. At exactly `epsilon = 1` the scale stops being
    # identifiable: once every sample is an outlier the objective reduces to
    #   σ·Σsw + Σ 2·|rᵢ|·swᵢ − σ·Σsw  =  2·Σ swᵢ|rᵢ|,
    # i.e. weighted least-ABSOLUTE-deviations with σ cancelling out exactly, and
    # `∂L/∂σ ≡ 0` along the whole ray. sklearn accepts `epsilon = 1` (its
    # constraint is the closed `[1, ∞)`) and duly returns `scale_ = 0` after a
    # long badly-scaled descent, ~4.6e-4 from its own optimum. The COEFFICIENTS
    # are still uniquely determined (LAD + ridge), so this case is gated on the
    # objective value and on `scale_ → 0`, never on parameter agreement.
    ctrl_cases = [
        ("maxiter0", dict(max_iter=0), False),
        ("maxiter1", dict(max_iter=1), False),
        ("maxiter5", dict(max_iter=5), False),
        ("tol_loose", dict(tol=5.0), False),
        ("eps1", dict(epsilon=1.0, max_iter=1000), False),
    ]

    def c(arr):
        return np.asarray(arr).astype(dtype)

    out = {
        "X": c(x),
        "y": c(y),
        "sample_weight": c(sw),
        "X_test": c(x_test),
    }

    def record(name, est, kwargs, use_sw):
        weights = sw if use_sw else ones
        params = np.concatenate([est.coef_, [est.intercept_, est.scale_]])
        eps_k = kwargs.get("epsilon", 1.35)
        alpha_k = kwargs.get("alpha", 1e-4)
        fi_k = kwargs.get("fit_intercept", True)
        out[f"coef_{name}"] = c(est.coef_)
        out[f"intercept_{name}"] = c([est.intercept_])
        # The scale, the loss and the iteration count are compared in f64
        # whatever the design's dtype: the solve accumulates in f64 on both
        # sides, and rounding them to f32 would hide real drift behind the
        # storage format (the `bayesian_ridge` precision precedent).
        out[f"scale_{name}"] = np.asarray([est.scale_], dtype=np.float64)
        out[f"n_iter_{name}"] = np.asarray([float(est.n_iter_)], dtype=np.float64)
        out[f"outliers_{name}"] = est.outliers_.astype(np.float64)
        # `params` is `[coef…, intercept, sigma]` — sklearn's own packing, and
        # what a warm start seeds from.
        packed = np.concatenate([est.coef_, [est.intercept_]]) if fi_k else est.coef_
        out[f"params_{name}"] = np.concatenate([packed, [est.scale_]]).astype(np.float64)
        out[f"loss_{name}"] = np.asarray(
            [_huber_objective(params, x, y, eps_k, alpha_k, weights)],
            dtype=np.float64,
        )
        return params, eps_k, alpha_k, fi_k

    residuals = []
    for name, kwargs, use_sw in value_cases:
        kw = dict(max_iter=1000)
        kw.update(kwargs)
        est = HuberRegressor(**kw)
        est.fit(x, y, sample_weight=sw if use_sw else None)
        params, eps_k, alpha_k, fi_k = record(name, est, kw, use_sw)
        out[f"pred_{name}"] = c(est.predict(x_test))
        # How far sklearn's own stop leaves it from the true minimizer: refit the
        # SAME objective with scipy driven to machine precision and measure. This
        # is the number the Rust band has to cover.
        tight = _huber_tight_optimum(x, y, eps_k, alpha_k, sw if use_sw else ones, fi_k)
        if not fi_k:
            # Pad the missing intercept slot with the 0.0 sklearn pins, so the
            # comparison uses the one `[coef…, intercept, sigma]` layout.
            tight = np.concatenate([tight[:d], [0.0], tight[d:]])
        residual = float(np.abs(params - tight).max())
        # Shipped PER CASE, not just as a maximum: the Rust test DERIVES its
        # comparison band from this number instead of carrying a hand-chosen
        # constant, so a case whose conditioning makes sklearn stop further out
        # (`eps105_noint` is 20x the median here — no intercept plus a tight
        # epsilon leaves the residuals large and the scale poorly determined)
        # widens only its own band, and every other case stays tight.
        out[f"residual_{name}"] = np.asarray([residual], dtype=np.float64)
        residuals.append(residual)

    for name, kwargs, use_sw in ctrl_cases:
        with warnings.catch_warnings():
            # A truncated fit warns ConvergenceWarning by construction — that is
            # the case being generated, not a problem to surface.
            warnings.simplefilter("ignore")
            est = HuberRegressor(**kwargs)
            est.fit(x, y, sample_weight=sw if use_sw else None)
        record(name, est, kwargs, use_sw)

    # --- warm_start: two successive fits of ONE estimator. The second starts
    #     from the first's `[coef_, intercept_, scale_]` instead of the cold
    #     `[0…, 0, 1]`, so it lands closer to the optimum for the same cap. --- #
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ws = HuberRegressor(warm_start=True, max_iter=5)
        ws.fit(x, y)
        out["params_warm1"] = np.concatenate(
            [ws.coef_, [ws.intercept_, ws.scale_]]
        ).astype(np.float64)
        out["loss_warm1"] = np.asarray(
            [_huber_objective(out["params_warm1"], x, y, 1.35, 1e-4, ones)],
            dtype=np.float64,
        )
        ws.fit(x, y)
        out["params_warm2"] = np.concatenate(
            [ws.coef_, [ws.intercept_, ws.scale_]]
        ).astype(np.float64)
        out["loss_warm2"] = np.asarray(
            [_huber_objective(out["params_warm2"], x, y, 1.35, 1e-4, ones)],
            dtype=np.float64,
        )

    out["huber_max_param_residual"] = np.asarray([max(residuals)], dtype=np.float64)

    # --- Premises asserted HERE (at generation) so a sklearn upgrade that breaks
    #     one is caught in this script rather than as an unexplained Rust
    #     failure. ---
    assert not any(
        isinstance(cons, StrOptions)
        for conss in HuberRegressor._parameter_constraints.values()
        for cons in conss
    ), (
        "gen_huber: sklearn's HuberRegressor grew a STRING-valued parameter. "
        "The Rust parameter-surface test asserts there are none — add the new "
        "parameter to `value_cases` and drop it from that assertion"
    )
    ref_loss, _ = _huber_loss_and_gradient(
        np.concatenate([np.zeros(d), [0.0, 1.0]]), x, y, 1.35, 1e-4, ones
    )
    assert np.isclose(
        ref_loss,
        _huber_objective(np.concatenate([np.zeros(d), [0.0, 1.0]]), x, y, 1.35, 1e-4, ones),
        rtol=1e-12,
    ), "gen_huber: the local objective no longer matches sklearn's _huber_loss_and_gradient"
    n_out_default = int(out["outliers_default"].sum())
    flagged = out["outliers_default"].astype(bool)
    # The injected shocks must ALL land in the outlier set (otherwise the design
    # is not testing the robust branch), and the majority of the clean rows must
    # NOT (otherwise `epsilon` is effectively zero and every sample is linear).
    # Note the outlier COUNT is legitimately much larger than the injected
    # fraction: Huber's fitted `σ` is a robust scale WELL below the residual
    # standard deviation, so a healthy share of clean Gaussian rows sit beyond
    # `1.35·σ` too. That is the estimator working, not the fixture misbehaving.
    assert flagged[out_idx].all(), (
        "gen_huber: not every injected gross outlier was classified as one — the "
        "shock magnitude is too small relative to the noise"
    )
    clean = np.setdiff1d(np.arange(n), out_idx)
    assert flagged[clean].mean() < 0.5, (
        f"gen_huber: {flagged[clean].mean():.0%} of the CLEAN rows are classified "
        "as outliers, so the quadratic core holds a minority of the data — "
        "retune the noise level"
    )
    assert n_out_default >= 5, (
        f"gen_huber: only {n_out_default} outliers of {n} — the robust branch is "
        "barely exercised; retune HUBER_OUTLIER_FRAC"
    )
    assert not np.allclose(out["coef_default"], out["coef_eps105"], atol=1e-6), (
        "gen_huber: epsilon=1.05 did not change the fit, so `epsilon` is not exercised"
    )
    assert float(out["scale_eps1"][0]) < 1e-6, (
        "gen_huber: epsilon=1.0 no longer collapses the scale, so the σ-degeneracy\n"
        "        documented on the `eps1` control case has changed — re-derive it"
    )
    assert int(out["outliers_eps1"].sum()) == n, (
        "gen_huber: epsilon=1.0 no longer classifies EVERY sample as an outlier, "
        "which is the premise the σ-degeneracy argument rests on"
    )
    assert not np.allclose(out["coef_default"], out["coef_eps10"], atol=1e-6), (
        "gen_huber: epsilon=10 did not change the fit, so `epsilon` is not exercised"
    )
    assert not np.allclose(out["coef_default"], out["coef_alpha100"], atol=1e-6), (
        "gen_huber: alpha=100 did not shrink the fit, so `alpha` is not exercised"
    )
    assert not np.allclose(out["coef_default"], out["coef_sw"], atol=1e-6), (
        "gen_huber: sample_weight did not change the fit"
    )
    assert np.allclose(out["coef_default"], out["coef_tol_tight"], atol=0, rtol=0), (
        "gen_huber: a 1e-12 tol changed the fit, so scipy's `factr` is no longer "
        "the binding stop — re-derive the module docs' stopping analysis in "
        "`huber.rs` and the Rust band below"
    )
    assert out["n_iter_maxiter0"][0] == 0.0 and out["n_iter_maxiter1"][0] == 1.0, (
        "gen_huber: max_iter=0/1 no longer report n_iter_ 0/1"
    )
    assert out["loss_maxiter5"][0] > out["loss_default"][0], (
        "gen_huber: the max_iter=5 fit is not worse than the converged one, so "
        "the truncation is not being exercised"
    )
    assert out["loss_warm2"][0] < out["loss_warm1"][0], (
        "gen_huber: the warm-started second fit did not improve on the first, so "
        "`warm_start` is not being exercised"
    )
    # A CEILING, not the band: the per-case `residual_*` entries are what the
    # Rust bands are derived from. This assert only catches a wholesale
    # regression in scipy/sklearn's stopping — at which point the module-doc
    # analysis in `huber.rs` needs rewriting, not just a wider tolerance.
    assert out["huber_max_param_residual"][0] < 1e-3, (
        "gen_huber: sklearn's own distance from the true optimum grew past 1e-3 "
        f"({out['huber_max_param_residual'][0]:.3e}) — the Rust band in "
        "huber_test.rs was sized against this and must be re-derived"
    )
    # The outlier mask is compared for EXACT equality in Rust, so no sample may
    # sit within a hair of the `ε·σ` boundary where a 1e-6 parameter wobble could
    # flip it.
    stability = []
    for name, kwargs, _use_sw in value_cases:
        eps_k = kwargs.get("epsilon", 1.35)
        resid = np.abs(
            y
            - x @ out[f"coef_{name}"].astype(np.float64)
            - float(out[f"intercept_{name}"][0])
        )
        margin = np.abs(resid - float(out[f"scale_{name}"][0]) * eps_k)
        # How far a sample's `|rᵢ| − ε·σ` can move when the parameters move by
        # the measured solver gap `R`: the residual shifts by at most
        # `R·‖xᵢ‖₁ + R` (coefficients + intercept) and the threshold by `ε·R`.
        # Four times that is the safety factor; a fixture whose closest sample
        # sits inside it cannot support an EXACT `outliers_` comparison.
        gap = float(out[f"residual_{name}"][0])
        flip_bound = 4.0 * gap * (np.abs(x).sum(axis=1).max() + 1.0 + eps_k)
        # Rather than assert this everywhere and reseed until it holds, the
        # verdict is SHIPPED: the Rust test compares `outliers_` for exact
        # equality only where the fixture can prove no sample is within reach of
        # the solver gap. `eps105_noint` is the one case that fails it — with no
        # intercept to absorb the offset its residuals are large, its scale is
        # poorly determined, and sklearn's own stop is 20x further out than
        # elsewhere — so its mask is gated on the outlier COUNT instead, and its
        # coefficients still carry the full value gate.
        stable = bool(margin.min() > flip_bound)
        out[f"outliers_stable_{name}"] = np.asarray([float(stable)], dtype=np.float64)
        stability.append((name, stable, float(margin.min()), float(flip_bound)))

    n_stable = sum(1 for _n, ok, _m, _b in stability if ok)
    assert n_stable >= len(value_cases) - 1, (
        "gen_huber: more than one value case has an unstable outlier mask "
        f"({[n for n, ok, _m, _b in stability if not ok]}) — the fixture's "
        "conditioning has drifted and the exact `outliers_` gate has lost most "
        "of its coverage; reseed"
    )
    assert bool(out["outliers_stable_default"][0]), (
        "gen_huber: even the DEFAULT case cannot support an exact outliers_ gate"
    )

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"huber_{dtype_tag}_seed{seed}.npz")
    np.savez(out_path, **out)
    return out_path


def _huber_tight_optimum(x, y, epsilon, alpha, sample_weight, fit_intercept):
    """The Huber minimizer driven to machine precision, for the residual probe.

    sklearn cannot produce this itself — its `factr` is not reachable through
    the public API — so the probe calls scipy directly with the same objective
    and a tolerance six orders tighter.
    """
    from scipy import optimize
    from sklearn.linear_model._huber import _huber_loss_and_gradient

    d = x.shape[1]
    n_params = d + 2 if fit_intercept else d + 1
    p0 = np.zeros(n_params)
    p0[-1] = 1.0
    bounds = np.tile([-np.inf, np.inf], (n_params, 1))
    bounds[-1][0] = np.finfo(np.float64).eps * 10
    res = optimize.minimize(
        _huber_loss_and_gradient,
        p0,
        method="L-BFGS-B",
        jac=True,
        args=(x, y, epsilon, alpha, sample_weight),
        bounds=bounds,
        options={"maxiter": 100000, "gtol": 1e-14, "ftol": 1e-18},
    )
    return res.x


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


# ---------------------------------------------------------------------------
# LARGE SpectralEmbedding fixtures (SPECTRAL-PERF-CPU) — the Lanczos branch
# ---------------------------------------------------------------------------
# The host pipeline routes `n_samples <= DENSE_N (512)` to a dense `sym_eig` and
# everything above it to a thick-restart Lanczos. Every fixture above is n=12,
# so the Lanczos arm had NO sklearn oracle at all. These two fixtures sit well
# above the threshold (800 / 700) and pin BOTH affinity families on it.
#
# Two properties are VERIFIED before a fixture is written (a violation of either
# makes a per-element value comparison meaningless, so both are hard asserts):
#
#   1. the affinity graph is CONNECTED — `_graph_is_connected`, the same
#      predicate whose failure makes sklearn warn. A disconnected graph gives
#      the normalized Laplacian a zero eigenvalue of multiplicity = #components,
#      and the kept eigenvectors are then an arbitrary basis of that null space.
#   2. the kept part of the spectrum is NON-DEGENERATE — every consecutive gap
#      among `λ_0 … λ_nev` (INCLUDING the boundary gap `λ_{nev-1} → λ_nev`, which
#      is what makes the retained subspace itself well defined) exceeds
#      `SE_LARGE_MIN_GAP`. Two eigenvalues within ~1e-6 would leave the pair
#      defined only up to a rotation and ARPACK's choice inside it would be
#      unreproducible by any other solver.
#
# The gaps that hold for the committed parameters are printed by
# `_spectral_spectrum_report` at generation time and recorded in the Rust test.

# kNN case: 800 samples in 8 features. `centers=3, cluster_std=8.0` deliberately
# OVERLAPS the blobs — three tight, well-separated blobs would give the kNN graph
# three components (verified: cluster_std=2.5 is disconnected at every k tried),
# and this is the spectral analogue of the well-separated-clusters requirement
# running the OTHER way.
SE_LARGE_N, SE_LARGE_D, SE_LARGE_COMPONENTS = 800, 8, 3
SE_LARGE_CENTERS, SE_LARGE_STD = 3, 8.0
SE_LARGE_N_NEIGHBORS = 15
# rbf case: 700 samples in 6 features, `gamma=None → 1/n_features`. A dense
# Gaussian kernel is strictly positive, so the graph is connected by
# construction; the assert still runs, since that is a property of the DATA.
SE_LARGE_RBF_N, SE_LARGE_RBF_D, SE_LARGE_RBF_COMPONENTS = 700, 6, 2
# Minimum admissible consecutive eigenvalue gap over the kept spectrum. Six
# orders of magnitude above the ~1e-6 degeneracy floor the brief names, so a
# small perturbation of the data cannot silently turn the fixture degenerate.
SE_LARGE_MIN_GAP = 1e-3


def _spectral_spectrum_report(affinity, nev, label):
    """Connectivity + eigenvalue-gap gate for a large spectral fixture.

    ``affinity`` is the sklearn-side affinity (sparse or dense, EXACTLY what
    ``SpectralEmbedding`` builds internally). Forms the same normalized
    Laplacian ``_spectral_embedding`` decomposes — ``csgraph.laplacian(A,
    normed=True)`` followed by sklearn's ``_set_diag(L, 1)`` — takes its dense
    spectrum with scipy, prints the smallest eigenvalues and their gaps, and
    ASSERTS the fixture is usable for a per-element value comparison. Returns
    the smallest ``nev + 2`` eigenvalues so the caller can commit them.
    """
    from scipy.sparse.csgraph import laplacian as csgraph_laplacian
    from scipy.linalg import eigvalsh
    from sklearn.manifold._spectral_embedding import _graph_is_connected

    connected = _graph_is_connected(affinity)
    dense = affinity.toarray() if hasattr(affinity, "toarray") else np.asarray(affinity)
    lap = csgraph_laplacian(dense, normed=True)
    # sklearn's `_set_diag(laplacian, 1, norm_laplacian)` — unconditional, so it
    # overrides scipy's `1 - isolated` on a zero-degree node.
    np.fill_diagonal(lap, 1.0)
    w = eigvalsh(lap)
    gaps = np.diff(w[: nev + 1])
    print(f"  {label}: connected={connected}")
    print(f"  {label}: smallest {nev + 2} eigs = "
          f"{np.array2string(w[: nev + 2], precision=8)}")
    print(f"  {label}: gaps over the kept spectrum = "
          f"{np.array2string(gaps, precision=8)}")
    assert connected, (
        f"{label}: affinity graph is DISCONNECTED — the zero eigenvalue is "
        "degenerate and the embedding is not reproducible; raise n_neighbors "
        "or spread the data"
    )
    assert gaps.min() > SE_LARGE_MIN_GAP, (
        f"{label}: smallest kept eigenvalue gap {gaps.min():.3e} <= "
        f"{SE_LARGE_MIN_GAP:.0e} — the retained eigenspace is (near-)degenerate "
        "and a per-element oracle comparison would be meaningless"
    )
    return w[: nev + 2]


def gen_spectral_embedding_large(seed: int = SEED, dtype=np.float64) -> str:
    """Large-`n` kNN SpectralEmbedding fixture (SPECTRAL-PERF-CPU).

    ``n_samples=800 > DENSE_N=512``, so the host pipeline solves this one with
    the THICK-RESTART LANCZOS arm rather than the dense ``sym_eig`` every other
    spectral fixture exercises. ``affinity='nearest_neighbors'`` with an
    EXPLICIT ``n_neighbors=SE_LARGE_N_NEIGHBORS`` chosen so the graph is
    connected (the ``None`` default would resolve to ``80`` here, which is a far
    denser graph and defeats the point of a sparse oracle).

    Stores row-major ``X``, sklearn's ``embedding_`` (``n × 3``), the neighbor
    count, the resolved shape, and the smallest Laplacian eigenvalues (the
    verified non-degenerate spectrum, for provenance). Returns the path.
    """
    from sklearn.datasets import make_blobs
    from sklearn.manifold import SpectralEmbedding
    from sklearn.neighbors import kneighbors_graph

    n, d, k = SE_LARGE_N, SE_LARGE_D, SE_LARGE_N_NEIGHBORS
    x, _ = make_blobs(
        n_samples=n,
        n_features=d,
        centers=SE_LARGE_CENTERS,
        cluster_std=SE_LARGE_STD,
        random_state=seed,
    )

    # The affinity sklearn builds internally for `nearest_neighbors`:
    # `kneighbors_graph(X, k, include_self=True)` symmetrized as `0.5*(A + Aᵀ)`.
    aff = kneighbors_graph(x, k, include_self=True)
    aff = 0.5 * (aff + aff.T)
    # drop_first=True → the estimator asks for n_components + 1 eigenvectors.
    nev = SE_LARGE_COMPONENTS + 1
    eigs = _spectral_spectrum_report(aff, nev, "spectral_embedding_large (knn)")

    se = SpectralEmbedding(
        n_components=SE_LARGE_COMPONENTS,
        affinity="nearest_neighbors",
        n_neighbors=k,
        random_state=seed,
    )
    embedding = se.fit_transform(x)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"spectral_embedding_large_{dtype_tag}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        embedding=c(embedding),
        n_neighbors=c([k]),
        shape=c([n, d]),
        n_components=c([SE_LARGE_COMPONENTS]),
        eigs=c(eigs),
    )
    return out_path


def gen_spectral_embedding_large_rbf(seed: int = SEED, dtype=np.float64) -> str:
    """Large-`n` rbf SpectralEmbedding fixture (SPECTRAL-PERF-CPU).

    The DENSE-affinity twin of :func:`gen_spectral_embedding_large`:
    ``n_samples=700 > DENSE_N=512`` so the Lanczos arm runs, but here it drives a
    dense `n × n` operator instead of a CSR one, which is a different matvec
    path. ``gamma=None`` is left UNSET on the estimator so sklearn resolves it to
    ``1/n_features`` itself (D-04) — the committed ``gamma`` array records the
    value that resolution must produce.

    Plain Gaussian data (not blobs): three tight blobs would put the smallest
    non-trivial eigenvalues within ~1e-3 of each other, which is exactly the
    near-degeneracy this fixture must avoid. Returns the path.
    """
    from sklearn.manifold import SpectralEmbedding
    from sklearn.metrics.pairwise import rbf_kernel

    n, d = SE_LARGE_RBF_N, SE_LARGE_RBF_D
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    gamma = 1.0 / d  # D-04: gamma=None → 1/n_features, resolved at fit.

    aff = rbf_kernel(x, gamma=gamma)
    nev = SE_LARGE_RBF_COMPONENTS + 1
    eigs = _spectral_spectrum_report(aff, nev, "spectral_embedding_large (rbf)")

    se = SpectralEmbedding(
        n_components=SE_LARGE_RBF_COMPONENTS,
        affinity="rbf",
        random_state=seed,
    )
    embedding = se.fit_transform(x)
    assert se.gamma_ == gamma, "sklearn must resolve gamma=None to 1/n_features"

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"spectral_embedding_large_rbf_{dtype_tag}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        embedding=c(embedding),
        gamma=c([gamma]),
        shape=c([n, d]),
        n_components=c([SE_LARGE_RBF_COMPONENTS]),
        eigs=c(eigs),
    )
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


def gen_linear_svc_multiclass(seed: int = SEED, dtype=np.float32) -> str:
    """Generate one MULTICLASS LinearSVC fixture (one-vs-rest).

    The 3-class twin of :func:`gen_linear_svc`. sklearn's ``LinearSVC`` fits
    ``n_classes`` independent one-vs-rest sub-problems through liblinear and
    stacks them, so ``coef_`` is ``(3, n_features)`` and ``intercept_`` is
    ``(3,)`` — the asymmetry against the binary fixture (``(1, d)`` / ``(1,)``)
    is exactly what the estimator has to reproduce, which is why this is a
    SEPARATE fixture rather than a wider ``n_classes`` on the existing one.

    Also stores ``decision`` (the ``(n_query, 3)`` decision function), because
    the argmax that turns it into a label is where a transposed or mis-strided
    ``coef_`` would still produce plausible-looking labels.
    """
    from sklearn.svm import LinearSVC

    _, x, y, xq = _sgd_blobs(seed, n_classes=3)

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
    out_path = os.path.join(
        _FIXTURE_DIR, f"linear_svc_multiclass_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        Xq=c(xq),
        y=c(y),
        classes=c(clf.classes_),
        coef=c(clf.coef_),
        intercept=c(clf.intercept_),
        decision=c(clf.decision_function(xq)),
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


# ---- Phase-17 DecisionTree oracle fixtures (TREE-01, D-07/D-09) ----
# The Tier-1 correctness witness (Plan 03) value-asserts the spike's
# histogram/gain/partition MATH against these committed sklearn reference trees.
# Determinism recipe (RESEARCH §Tier-1, D-07): inject FIXED bootstrap-row and
# feature-column index arrays (plain integers, NEVER RNG-drawn) and fit sklearn
# on exactly X[bootstrap_idx][:, feature_idx]. With max_features=None over that
# fixed column subset, sklearn considers precisely the injected columns and no
# RNG enters the split selection, so a single tree reproduces element-wise.
#
# Standard fixtures fit on a ~60x8 synthetic set restricted to a 5-column subset;
# the emitted sklearn ``tree_`` attributes (children_left/right, feature,
# threshold, value) let Plan 03 compare split STRUCTURE (exact) and leaf VALUES
# (<=1e-5 f64). Note ``feature`` indexes the INJECTED feature_idx subset (the
# columns actually handed to sklearn), NOT the original X columns; Plan 03 maps
# back through feature_idx when needed.
#
# Adversarial fixtures (the 002-B silent-miscompile backstop, T-17-01) are the
# histogram analogue of Phase 13's duplicate-point row: two IDENTICAL feature
# columns force an exact gain TIE, and a perfectly-separable target forces both
# children to become PURE leaves (zero impurity / zero variance). The generator
# INDEPENDENTLY verifies the tie exists (pure-numpy impurity over each tied
# column) and documents sklearn's canonical resolution — lowest feature index,
# then lowest threshold — WITHOUT ever hand-patching the blob to match any mlrs
# kernel pick (Phase-13 CR-01/CR-02; committed blobs stay reproducible from this
# generator alone). Because the tied columns are identical, the resulting
# partition/children/leaf-values are invariant to which tied column is recorded.
DT_N_SAMPLES = 60
DT_N_FEATURES = 8
DT_MAX_DEPTH = 4
# FIXED injected indices (D-07) — plain integer arrays, never RNG-drawn. The
# bootstrap sample draws WITH replacement (repeats present, as a real RF bag
# would) and the feature subset selects 5 of the 8 columns.
DT_BOOTSTRAP_IDX = np.array(
    [
        0, 3, 3, 7, 11, 14, 14, 18, 21, 22, 25, 29, 31, 33, 36, 38,
        41, 41, 44, 47, 49, 50, 52, 55, 57, 59, 2, 6, 9, 13, 16, 19,
        23, 26, 28, 30, 34, 37, 39, 42, 45, 48, 51, 54, 56, 58, 1, 5,
    ],
    dtype=np.int64,
)
DT_FEATURE_IDX = np.array([0, 2, 3, 5, 6], dtype=np.int64)


def _dt_gini_best_impurity(col, y):
    """Independent (pure-numpy) best weighted-Gini child impurity for one column.

    Scans midpoint thresholds between sorted-unique values and returns the
    minimum n-weighted child Gini impurity achievable on ``col`` (lower is a
    better split). Used to PROVE the adversarial gain tie is genuine WITHOUT
    consulting sklearn's choice — the tie-break verification stays independent
    of the reference estimator (Phase-13 CR-01/CR-02).
    """
    n = len(y)
    classes = np.unique(y)
    uniq = np.unique(col)
    best = np.inf
    for a, b in zip(uniq[:-1], uniq[1:]):
        thr = (a + b) / 2.0
        left = col <= thr
        right = ~left
        nl, nr = int(left.sum()), int(right.sum())
        if nl == 0 or nr == 0:
            continue

        def gini(mask):
            m = int(mask.sum())
            if m == 0:
                return 0.0
            p = np.array([(y[mask] == c).sum() / m for c in classes])
            return 1.0 - float((p * p).sum())

        weighted = (nl * gini(left) + nr * gini(right)) / n
        best = min(best, weighted)
    return best


def _dt_var_best_impurity(col, y):
    """Independent best n-weighted child VARIANCE for one column (regression tie)."""
    n = len(y)
    uniq = np.unique(col)
    best = np.inf
    for a, b in zip(uniq[:-1], uniq[1:]):
        thr = (a + b) / 2.0
        left = col <= thr
        right = ~left
        nl, nr = int(left.sum()), int(right.sum())
        if nl == 0 or nr == 0:
            continue
        vl = float(np.var(y[left])) if nl else 0.0
        vr = float(np.var(y[right])) if nr else 0.0
        weighted = (nl * vl + nr * vr) / n
        best = min(best, weighted)
    return best


def gen_decision_tree_clf(
    seed: int = SEED, dtype=np.float32, structure: str = "standard"
) -> str:
    """Generate one DecisionTreeClassifier(gini) reference fixture (TREE-01, D-09).

    ``structure="standard"`` fits on a ~60x8 synthetic binary-class set restricted
    to the fixed ``DT_BOOTSTRAP_IDX`` rows and ``DT_FEATURE_IDX`` columns (D-07).
    ``structure="adversarial"`` builds the forced-pure-leaf + gain-tie backstop
    (two identical columns, perfectly separable target) and INDEPENDENTLY asserts
    the tie is genuine. Emits ``X``, ``y``, ``bootstrap_idx``, ``feature_idx`` and
    the sklearn ``tree_`` attributes (``children_left/right``, ``feature``,
    ``threshold``, ``value``). Returns the path written.
    """
    from sklearn.tree import DecisionTreeClassifier

    if structure == "adversarial":
        # Two IDENTICAL columns => exact gain tie; perfectly-separable y => both
        # children become PURE leaves. bootstrap/feature injection is the trivial
        # identity here so the engineered tie/leaf survive verbatim.
        base = np.array([0] * 8 + [1] * 8, dtype=np.float64)
        x = np.column_stack([base, base])
        y = base.astype(np.int64)
        boot = np.arange(len(y), dtype=np.int64)
        feat = np.arange(x.shape[1], dtype=np.int64)
        x_fit = x[boot][:, feat]
        # Data-construction sanity check (NOT a tie-break proof): the two
        # byte-identical columns trivially achieve the same best child impurity.
        # On identical columns this can never fail, so it proves nothing about
        # sklearn's pick (IN-03) — it only confirms the fixture was built tied.
        imp0 = _dt_gini_best_impurity(x_fit[:, 0], y[boot])
        imp1 = _dt_gini_best_impurity(x_fit[:, 1], y[boot])
        assert abs(imp0 - imp1) < 1e-12, (
            f"adversarial clf data malformed (columns not tied): {imp0} vs {imp1}"
        )
        # Pass max_depth=DT_MAX_DEPTH to match the standard branch AND the
        # witness builder (tree_witness.rs MAX_DEPTH=4), so both sides share one
        # depth cap (WR-03). The adversarial design is depth-1 by construction
        # (two identical columns, perfectly-separable target → one split, two
        # pure leaves), so the cap is never reached and the emitted tree is
        # unchanged — but the cap is now explicit instead of an undocumented
        # coupling to that data property.
        clf = DecisionTreeClassifier(
            criterion="gini", max_depth=DT_MAX_DEPTH, random_state=seed
        )
        clf.fit(x_fit, y[boot])
        # Load-bearing guard (IN-03 / WR-01): pin sklearn's CANONICAL lowest-index
        # pick at the gain-tie root. The witness gates the adversarial clf as a
        # FUNCTION (feature-index-independent), so THIS generator-side assert is
        # what verifies the committed blob against the documented tie-break rule
        # (lowest feature index). A sklearn shuffle/RNG change that recorded
        # feature 1 here fails LOUDLY at regeneration instead of silently shipping
        # a blob whose recorded root feature no longer matches the documented rule.
        assert int(clf.tree_.feature[0]) == 0, (
            "adversarial clf: sklearn recorded root split feature "
            f"{int(clf.tree_.feature[0])}, expected canonical lowest index 0"
        )
        suffix = "clf_adv"
    else:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((DT_N_SAMPLES, DT_N_FEATURES))
        # Binary target with signal on a couple of the SELECTED columns so the
        # injected feature subset is genuinely informative.
        logits = 1.3 * x[:, 0] - 0.9 * x[:, 3] + 0.6 * x[:, 6]
        y = (logits > np.median(logits)).astype(np.int64)
        boot = DT_BOOTSTRAP_IDX
        feat = DT_FEATURE_IDX
        x_fit = x[boot][:, feat]
        clf = DecisionTreeClassifier(
            criterion="gini", max_depth=DT_MAX_DEPTH, random_state=seed
        )
        clf.fit(x_fit, y[boot])
        suffix = "clf"

    t = clf.tree_

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"tree_dt_{suffix}_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        y=c(y),
        bootstrap_idx=c(boot),
        feature_idx=c(feat),
        children_left=c(t.children_left),
        children_right=c(t.children_right),
        feature=c(t.feature),
        threshold=c(t.threshold),
        value=c(t.value),
    )
    return out_path


def gen_decision_tree_reg(
    seed: int = SEED, dtype=np.float32, structure: str = "standard"
) -> str:
    """Generate one DecisionTreeRegressor(squared_error) reference fixture (D-09).

    Mirror of ``gen_decision_tree_clf`` for the regression leaf path. Standard fits
    a continuous target on the fixed injected rows/columns; ``adversarial`` forces
    a zero-variance (pure) leaf plus an exact split-variance tie between two
    identical columns, independently verified. Emits the same array set (``value``
    here carries the regression-mean leaves). Returns the path written.
    """
    from sklearn.tree import DecisionTreeRegressor

    if structure == "adversarial":
        # Identical columns => variance-reduction tie; two-level constant target
        # => each child has ZERO variance (pure regression leaf).
        base = np.array([0] * 8 + [1] * 8, dtype=np.float64)
        x = np.column_stack([base, base])
        y = base * 4.0 + 1.0  # -> {1.0, 5.0}, each region constant
        boot = np.arange(len(y), dtype=np.int64)
        feat = np.arange(x.shape[1], dtype=np.int64)
        x_fit = x[boot][:, feat]
        # Data-construction sanity check (NOT a tie-break proof, IN-03): identical
        # columns trivially achieve the same best child variance — always true.
        v0 = _dt_var_best_impurity(x_fit[:, 0], y[boot])
        v1 = _dt_var_best_impurity(x_fit[:, 1], y[boot])
        assert abs(v0 - v1) < 1e-12, (
            f"adversarial reg data malformed (columns not tied): {v0} vs {v1}"
        )
        # Pass max_depth=DT_MAX_DEPTH to match the standard branch AND the
        # witness builder (tree_witness.rs MAX_DEPTH=4), so both sides share one
        # depth cap (WR-03). The adversarial design is depth-1 by construction
        # (identical columns, two-level constant target → one split, two
        # zero-variance leaves), so the cap is never reached and the emitted tree
        # is unchanged — but the cap is now explicit instead of an undocumented
        # coupling to that data property.
        reg = DecisionTreeRegressor(
            criterion="squared_error", max_depth=DT_MAX_DEPTH, random_state=seed
        )
        reg.fit(x_fit, y[boot])
        # Load-bearing guard (IN-03 / WR-01): pin sklearn's canonical lowest-index
        # pick at the variance-tie root so the committed blob is verified against
        # the documented rule, not an always-true identity.
        assert int(reg.tree_.feature[0]) == 0, (
            "adversarial reg: sklearn recorded root split feature "
            f"{int(reg.tree_.feature[0])}, expected canonical lowest index 0"
        )
        suffix = "reg_adv"
    else:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((DT_N_SAMPLES, DT_N_FEATURES))
        # Continuous target with signal on selected columns + mild noise.
        y = (
            2.0 * x[:, 0] - 1.5 * x[:, 3] + 0.8 * x[:, 6]
            + 0.05 * rng.standard_normal(DT_N_SAMPLES)
        )
        boot = DT_BOOTSTRAP_IDX
        feat = DT_FEATURE_IDX
        x_fit = x[boot][:, feat]
        reg = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=DT_MAX_DEPTH,
            random_state=seed,
        )
        reg.fit(x_fit, y[boot])
        suffix = "reg"

    t = reg.tree_

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"tree_dt_{suffix}_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(
        out_path,
        X=c(x),
        y=c(y),
        bootstrap_idx=c(boot),
        feature_idx=c(feat),
        children_left=c(t.children_left),
        children_right=c(t.children_right),
        feature=c(t.feature),
        threshold=c(t.threshold),
        value=c(t.value),
    )
    return out_path


# Phase-19 RandomForest FOREST-level oracle fixtures (ENSEMBLE-01). Unlike the
# Phase-17 single-tree injected-index witness fixtures (tree_dt_*), these gate
# the full mlrs forest ESTIMATOR surface. Two tiers per task:
#   - DETERMINISTIC tier (bootstrap=False, max_features=None): all sklearn
#     trees are identical and RNG-free; with grid-valued features (every
#     feature has << n_bins distinct values) the mlrs binned candidate set
#     equals sklearn's exact midpoint set, and both growers reach PURE leaves,
#     so TRAIN-set predictions/probas match sklearn EXACTLY (asserted here at
#     generation: sklearn train accuracy / R² == 1, probas one-hot).
#     Held-out predictions are NOT exact-gated (equal-quality splits may pick
#     different-but-decision-equivalent thresholds — the Phase-17 witness Open
#     Question 1 resolution); they are gated statistically instead.
#   - STATISTICAL tier (sklearn defaults: bootstrap + sqrt features): the
#     held-out sklearn accuracy / R² is stored and the mlrs forest must land
#     within a small margin.
RF_N_TRAIN = 96
RF_N_TEST = 48
RF_N_FEATURES = 5
RF_GRID = 16  # distinct values per feature (<< n_bins-1 = 31 candidates)
RF_DET_MAX_DEPTH = 12
RF_STAT_N_ESTIMATORS = 64
RF_STAT_MAX_DEPTH = 8


def _rf_grid_data(rng, n_rows: int):
    """Grid-valued features in [0, 1]: RF_GRID distinct values per feature."""
    raw = rng.integers(0, RF_GRID, size=(n_rows, RF_N_FEATURES))
    return raw.astype(np.float64) / (RF_GRID - 1)


def gen_random_forest_classifier(seed: int = SEED, dtype=np.float32) -> str:
    """Generate one RandomForest CLASSIFIER fixture (ENSEMBLE-01).

    Stores ``X``/``y`` (train), ``Xq``/``yq`` (held-out), the deterministic
    tier's train-set ``det_pred_train``/``det_proba_train`` (asserted == y /
    one-hot at generation), and the statistical tier's held-out sklearn
    accuracy ``stat_acc_test``. Returns the path written.
    """
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(seed)
    x = _rf_grid_data(rng, RF_N_TRAIN)
    xq = _rf_grid_data(rng, RF_N_TEST)
    # No duplicate train rows: a duplicated row with a noise-flipped label
    # would make pure leaves impossible and break the exact tier.
    assert np.unique(x, axis=0).shape[0] == RF_N_TRAIN, "duplicate train rows"

    def rule(a):
        return np.where(a[:, 0] < 0.5, 0, np.where(a[:, 1] < 0.5, 1, 2))

    y = rule(x)
    yq = rule(xq)
    # ~10% label noise so trees must genuinely isolate noisy points.
    flip = rng.random(RF_N_TRAIN) < 0.10
    y = np.where(flip, (y + rng.integers(1, 3, size=RF_N_TRAIN)) % 3, y)

    det = RandomForestClassifier(
        n_estimators=2,
        bootstrap=False,
        max_features=None,
        max_depth=RF_DET_MAX_DEPTH,
        random_state=0,
    ).fit(x, y)
    det_pred_train = det.predict(x)
    det_proba_train = det.predict_proba(x)
    # Load-bearing generation-time guards: the deterministic tier only gates
    # EXACT parity if sklearn itself reaches purity on the train set.
    assert (det_pred_train == y).all(), "det clf tier: sklearn not pure on train"
    assert np.allclose(det_proba_train.max(axis=1), 1.0), "det clf tier: proba not one-hot"
    # RF-IMP-01: feature_importances_ on the SAME deterministic-tier fitted
    # forest, where mlrs and sklearn trees are proven structurally identical
    # (exact-tier oracle assertion, TASK-02).
    ref_feature_importances = det.feature_importances_

    stat = RandomForestClassifier(
        n_estimators=RF_STAT_N_ESTIMATORS,
        max_depth=RF_STAT_MAX_DEPTH,
        random_state=0,
    ).fit(x, y)
    stat_acc_test = float((stat.predict(xq) == yq).mean())

    # RF-OOB-01 (TASK-06): a SECOND sklearn construction, same seed/data/
    # statistical-tier hyperparameters as `stat` above, plus
    # `oob_score=True, bootstrap=True` (bootstrap is already sklearn's
    # default, made explicit here since `oob_score=True` requires it).
    # `oob_score_` is a fixed, non-dtype-dependent sklearn float, computed
    # ONCE and cast into both the f32 and f64 fixture files below.
    stat_oob = RandomForestClassifier(
        n_estimators=RF_STAT_N_ESTIMATORS,
        max_depth=RF_STAT_MAX_DEPTH,
        bootstrap=True,
        oob_score=True,
        random_state=0,
    ).fit(x, y)
    ref_oob_score = float(stat_oob.oob_score_)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"rf_cls_{dtype_tag}_seed{seed}.npz")
    np.savez(
        out_path,
        X=c(x),
        y=c(y),
        Xq=c(xq),
        yq=c(yq),
        det_pred_train=c(det_pred_train),
        det_proba_train=c(det_proba_train),
        stat_acc_test=c([stat_acc_test]),
        ref_feature_importances=c(ref_feature_importances),
        ref_oob_score=c([ref_oob_score]),
    )
    return out_path


def gen_random_forest_regressor(seed: int = SEED, dtype=np.float32) -> str:
    """Generate one RandomForest REGRESSOR fixture (ENSEMBLE-01).

    The deterministic tier uses a PIECEWISE-CONSTANT target (finite level set)
    so both growers reach zero-variance leaves and train predictions match
    ``y`` exactly (asserted at generation). The statistical tier stores the
    held-out sklearn R² for the margin gate. Returns the path written.
    """
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(seed + 1)
    x = _rf_grid_data(rng, RF_N_TRAIN)
    xq = _rf_grid_data(rng, RF_N_TEST)
    assert np.unique(x, axis=0).shape[0] == RF_N_TRAIN, "duplicate train rows"

    def levels(a):
        # Piecewise-constant on a 2-feature grid of cells → finite level set.
        return (
            1.0 * (a[:, 0] >= 0.5)
            + 2.5 * (a[:, 1] >= 0.5)
            + 0.75 * (a[:, 2] >= 0.25)
        )

    y = levels(x)
    yq = levels(xq)

    det = RandomForestRegressor(
        n_estimators=2,
        bootstrap=False,
        max_features=1.0,
        max_depth=RF_DET_MAX_DEPTH,
        random_state=0,
    ).fit(x, y)
    det_pred_train = det.predict(x)
    assert np.allclose(det_pred_train, y, atol=1e-12), "det reg tier: sklearn not pure on train"
    # RF-IMP-01: feature_importances_ on the SAME deterministic-tier fitted
    # forest, where mlrs and sklearn trees are proven structurally identical
    # (exact-tier oracle assertion, TASK-03, mirrors the classifier's TASK-02).
    ref_feature_importances = det.feature_importances_

    stat = RandomForestRegressor(
        n_estimators=RF_STAT_N_ESTIMATORS,
        max_depth=RF_STAT_MAX_DEPTH,
        random_state=0,
    ).fit(x, y)
    pred_q = stat.predict(xq)
    ss_res = float(((yq - pred_q) ** 2).sum())
    ss_tot = float(((yq - yq.mean()) ** 2).sum())
    stat_r2_test = 1.0 - ss_res / ss_tot

    # RF-OOB-01 (TASK-07): a SECOND sklearn construction, same seed/data/
    # statistical-tier hyperparameters as `stat` above, plus
    # `oob_score=True, bootstrap=True` (bootstrap is already sklearn's
    # default, made explicit here since `oob_score=True` requires it).
    # `oob_score_` (R²-based for RandomForestRegressor) is a fixed,
    # non-dtype-dependent sklearn float, computed ONCE and cast into both
    # the f32 and f64 fixture files below (mirrors the classifier's TASK-06).
    stat_oob = RandomForestRegressor(
        n_estimators=RF_STAT_N_ESTIMATORS,
        max_depth=RF_STAT_MAX_DEPTH,
        bootstrap=True,
        oob_score=True,
        random_state=0,
    ).fit(x, y)
    ref_oob_score = float(stat_oob.oob_score_)

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"rf_reg_{dtype_tag}_seed{seed}.npz")
    np.savez(
        out_path,
        X=c(x),
        y=c(y),
        Xq=c(xq),
        yq=c(yq),
        det_pred_train=c(det_pred_train),
        stat_r2_test=c([stat_r2_test]),
        ref_feature_importances=c(ref_feature_importances),
        ref_oob_score=c([ref_oob_score]),
    )
    return out_path


# HistGradientBoosting oracle fixtures (GBT-01). The mlrs grower is
# LEVEL-WISE with a depth bound; sklearn's grower is leaf-wise (best-first)
# with a leaf budget. With ``max_leaf_nodes=None`` and ``max_depth=D`` the
# leaf budget is gone, growth ORDER is irrelevant (each node's split is
# independent), and sklearn's tree equals the mlrs level-wise tree — so the
# deterministic tier pins EXACT train parity without needing purity:
#   - grid-valued features (16 distinct values << max_bins) make sklearn's
#     ``_BinMapper`` midpoints identical to the mlrs candidate edges;
#   - HGB has NO RNG (no bootstrap, no feature subsampling), so identical
#     candidate sets + identical gain rule => identical trees + identical
#     shrunk leaf values => train predictions match to float error.
#   Held-out predictions stay margin-gated (near-tie gains may resolve to
#   decision-equivalent-on-train but different thresholds — the RF lesson).
#   - the STATISTICAL tier uses sklearn DEFAULTS (leaf-wise, 31 leaves,
#     max_iter=100, early_stopping off) vs mlrs defaults (depth 6) and gates
#     the held-out accuracy/R² within a margin.
HGB_N_TRAIN = 96
HGB_N_TEST = 48
HGB_N_FEATURES = 5
HGB_DET_MAX_ITER = 20
HGB_DET_MAX_DEPTH = 6
HGB_DET_MIN_SAMPLES_LEAF = 5
HGB_DET_LEARNING_RATE = 0.1


def _hgb_det_kwargs():
    """The deterministic-tier sklearn kwargs shared by both HGB generators."""
    return dict(
        max_iter=HGB_DET_MAX_ITER,
        learning_rate=HGB_DET_LEARNING_RATE,
        max_depth=HGB_DET_MAX_DEPTH,
        max_leaf_nodes=None,
        min_samples_leaf=HGB_DET_MIN_SAMPLES_LEAF,
        l2_regularization=0.0,
        max_bins=255,
        early_stopping=False,
        random_state=0,
    )


def gen_hgb_regressor(seed: int = SEED, dtype=np.float32, rng_offset: int = 31) -> str:
    """Generate one HistGradientBoosting REGRESSOR fixture (GBT-01).

    Stores ``X``/``y`` (train), ``Xq``/``yq`` (held-out), the deterministic
    tier's train predictions ``det_pred_train`` (exact-gated) and held-out
    R² ``det_r2_test``, plus the sklearn-DEFAULTS statistical tier's held-out
    R² ``stat_r2_test``. Returns the path written.

    `rng_offset` selects the `_rf_grid_data` draw — tunable for the same
    reason as `gen_hgb_classifier`'s: mlrs's GBT-01 sibling-subtraction
    optimization makes its histogram reduction float-noise-sensitive like
    sklearn's own (independent reduction tree, not bit-identical), so an
    offset must be probed against the ACTUAL mlrs fit (both backends — the
    committed value passed cpu but a stale one failed on wgpu at f64, since
    block/reduction order differs by backend even before subtraction).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(seed + rng_offset)
    x = _rf_grid_data(rng, HGB_N_TRAIN)
    xq = _rf_grid_data(rng, HGB_N_TEST)
    assert np.unique(x, axis=0).shape[0] == HGB_N_TRAIN, "duplicate train rows"

    def levels(a):
        # Piecewise-constant target (well-separated split gains).
        return (
            1.0 * (a[:, 0] >= 0.5)
            + 2.5 * (a[:, 1] >= 0.5)
            + 0.75 * (a[:, 2] >= 0.25)
        )

    y = levels(x)
    yq = levels(xq)

    det = HistGradientBoostingRegressor(**_hgb_det_kwargs()).fit(x, y)
    det_pred_train = det.predict(x)
    det_pred_test = det.predict(xq)
    ss_res = float(((yq - det_pred_test) ** 2).sum())
    ss_tot = float(((yq - yq.mean()) ** 2).sum())
    det_r2_test = 1.0 - ss_res / ss_tot
    # Generation-time sanity: 20 boosted iterations must fit this clean
    # target well, or the fixture gates nothing.
    train_r2 = det.score(x, y)
    assert train_r2 > 0.95, f"det hgb reg tier: sklearn train R² {train_r2} too low"

    stat = HistGradientBoostingRegressor(early_stopping=False, random_state=0).fit(x, y)
    pred_q = stat.predict(xq)
    ss_res = float(((yq - pred_q) ** 2).sum())
    stat_r2_test = 1.0 - ss_res / ss_tot

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"hgb_reg_{dtype_tag}_seed{seed}.npz")
    np.savez(
        out_path,
        X=c(x),
        y=c(y),
        Xq=c(xq),
        yq=c(yq),
        det_pred_train=c(det_pred_train),
        det_r2_test=c([det_r2_test]),
        stat_r2_test=c([stat_r2_test]),
    )
    return out_path


def gen_hgb_classifier(seed: int = SEED, dtype=np.float32, rng_offset: int = 40) -> str:
    """Generate one HistGradientBoosting CLASSIFIER fixture (GBT-01).

    Carries BOTH class-count paths in one file: the 3-class rule target
    (softmax, K = 3 trees/iteration) and its binarized sibling ``y == 0``
    (sigmoid, K = 1). Stores the deterministic tier's train
    probabilities/labels for each (exact-gated) and a NOISY-label statistical
    tier's held-out accuracy (margin-gated).

    CRITICAL (deterministic-tier design): the det-tier labels are the CLEAN
    rule — no noise. Noisy labels create EXACT-TIE split gains whose float
    resolution differs between sklearn's and mlrs's histogram-SUBTRACTION
    sibling reduction (both subtract now — mlrs's GBT-01 sibling-subtraction
    optimization matches sklearn's own approach — but the two are independent
    implementations with different reduction trees, so they are NOT
    bit-identical); a tie that resolves into a different row partition
    diverges the ensembles from that iteration on. On the clean target every
    informative split's gain is well-separated (remaining ties occur only
    inside PURE nodes, where all candidate children share one value, so
    predictions are unaffected) — verified at generation by the
    train-accuracy assertion. `rng_offset` selects the `_rf_grid_data` draw;
    it is tunable because "well-separated" is empirical, not provable in
    closed form — see `scripts/gen_oracle.py`'s git history for how the
    committed offset was chosen (probed against the actual mlrs subtraction
    path, not just sklearn's `train_acc == 1.0` check, which both offsets
    satisfy despite one producing a near-tie ~2.4e-5 off at f64 tolerance).
    Label noise is exercised by the statistical tier instead.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(seed + rng_offset)
    x = _rf_grid_data(rng, HGB_N_TRAIN)
    xq = _rf_grid_data(rng, HGB_N_TEST)
    assert np.unique(x, axis=0).shape[0] == HGB_N_TRAIN, "duplicate train rows"

    def rule(a):
        return np.where(a[:, 0] < 0.5, 0, np.where(a[:, 1] < 0.5, 1, 2))

    y = rule(x)
    yq = rule(xq)
    y_bin = (y == 0).astype(np.int64)
    # Noisy sibling for the statistical tier (~10% flips).
    flip = rng.random(HGB_N_TRAIN) < 0.10
    y_noisy = np.where(flip, (y + rng.integers(1, 3, size=HGB_N_TRAIN)) % 3, y)

    det = HistGradientBoostingClassifier(**_hgb_det_kwargs()).fit(x, y)
    det_pred_train = det.predict(x)
    det_proba_train = det.predict_proba(x)
    acc = float((det_pred_train == y).mean())
    assert acc == 1.0, f"det hgb clf tier: sklearn train accuracy {acc} != 1 on clean rule"

    det_bin = HistGradientBoostingClassifier(**_hgb_det_kwargs()).fit(x, y_bin)
    det_pred_bin_train = det_bin.predict(x)
    det_proba_bin_train = det_bin.predict_proba(x)
    assert det_bin.n_iter_ == HGB_DET_MAX_ITER

    stat = HistGradientBoostingClassifier(early_stopping=False, random_state=0).fit(
        x, y_noisy
    )
    stat_acc_test = float((stat.predict(xq) == yq).mean())

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"hgb_cls_{dtype_tag}_seed{seed}.npz")
    np.savez(
        out_path,
        X=c(x),
        y=c(y),
        y_noisy=c(y_noisy),
        Xq=c(xq),
        yq=c(yq),
        y_bin=c(y_bin),
        det_pred_train=c(det_pred_train),
        det_proba_train=c(det_proba_train),
        det_pred_bin_train=c(det_pred_bin_train),
        det_proba_bin_train=c(det_proba_bin_train),
        stat_acc_test=c([stat_acc_test]),
    )
    return out_path


# ---- Metrics surface fixtures (METR-01/02/03, .planning/plans/metrics-surface) ----
#
# All four generators below require ``scikit-learn==1.9.0`` (the version this
# fixture was pinned against — TASK-02 Q6 resolution, PLAN.md, confirmed by
# `/tmp/oracle-venv/bin/python -c "import sklearn; print(sklearn.__version__)"`
# printing `1.9.0` at fixture-generation time).
#
# OvO + `sample_weight` probe (Q10, Plan-Check Issue 2), run BEFORE writing
# `gen_metrics_classification_multiclass`'s weighted-OvO branch, against this
# module's own `y_true`/`y_proba`/`sample_weight`:
#
#     roc_auc_score(y_true, y_proba, multi_class="ovo", average="macro", sample_weight=w)
#
# Outcome under scikit-learn==1.9.0: RAISES
#     ValueError("sample_weight is not supported for multiclass one-vs-one
#     ROC AUC score, 'sample_weight' must be None in this case.")
# => Branch A. No `ref_roc_auc_ovo_{macro,weighted}_sw` array is generated;
# TASK-10/TASK-21 implement/test the `MetricError::WeightedOvoUnsupported` /
# `ValueError` gate instead of a value.
_METRICS_FIXTURE_SEED = SEED


def _c_metrics(arr, dtype):
    """Float-cast helper for the metrics fixtures (Plan-Check Issue 5): EVERY
    array saved into a `metrics_*.npz` file — including label arrays,
    `labels*` arrays, and `ref_confusion*` count matrices — MUST go through
    this cast. `mlrs_core::oracle::load_npz` only decodes 4-/8-byte float
    dtypes and fails the WHOLE file on any int32/int64 array.
    """
    return np.ascontiguousarray(np.asarray(arr)).astype(dtype)


def gen_metrics_classification_binary(seed: int = _METRICS_FIXTURE_SEED, dtype=np.float32) -> str:
    """Generate the binary classification metrics oracle fixture (METR-CLS-01
    ..07, METR-CLS-09). Requires ``scikit-learn==1.9.0`` (the version this
    fixture was pinned against — see the Q6 resolution in
    `.planning/plans/metrics-surface/PLAN.md`).

    Builds a small binary `y_true`/`y_pred` set, a tie-heavy `y_score`
    (quantized to 5 distinct levels so `roc_auc_score`/`precision_recall_curve`
    exercise average-rank tie handling), a non-uniform `sample_weight`, and a
    2-column row-major `y_prob_binary` proba array (Plan-Check Issue 7,
    explicit binary `log_loss` reference). Every `ref_*` value is computed via
    the corresponding `sklearn.metrics` function, INCLUDING the weighted
    `precision_recall_curve` triple (Plan-Check Issue 1, always generated).
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    rng = np.random.default_rng(seed + 101)
    n = 40
    y_true = rng.integers(0, 2, size=n)
    flip = rng.random(n) < 0.2
    y_pred = np.where(flip, 1 - y_true, y_true).astype(np.int64)
    # Tie-heavy score: quantize to 5 distinct levels so several samples share
    # an exact score (average-rank tie handling, SPEC METR-CLS-07/09).
    y_score = np.round(rng.random(n) * 4) / 4.0
    sample_weight = rng.uniform(0.5, 2.5, size=n)
    proba_pos = np.clip(y_score, 0.0, 1.0)
    y_prob_binary = np.column_stack([1.0 - proba_pos, proba_pos])

    ref_accuracy = accuracy_score(y_true, y_pred)
    ref_accuracy_sw = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    ref_confusion = confusion_matrix(y_true, y_pred)
    ref_confusion_sw = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    ref_precision_binary = precision_score(y_true, y_pred, pos_label=1, average="binary")
    ref_recall_binary = recall_score(y_true, y_pred, pos_label=1, average="binary")
    ref_f1_binary = f1_score(y_true, y_pred, pos_label=1, average="binary")
    ref_precision_binary_sw = precision_score(
        y_true, y_pred, pos_label=1, average="binary", sample_weight=sample_weight
    )
    ref_recall_binary_sw = recall_score(
        y_true, y_pred, pos_label=1, average="binary", sample_weight=sample_weight
    )
    ref_f1_binary_sw = f1_score(
        y_true, y_pred, pos_label=1, average="binary", sample_weight=sample_weight
    )
    ref_roc_auc = roc_auc_score(y_true, y_score)
    ref_roc_auc_sw = roc_auc_score(y_true, y_score, sample_weight=sample_weight)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    pr_precision_sw, pr_recall_sw, pr_thresholds_sw = precision_recall_curve(
        y_true, y_score, sample_weight=sample_weight
    )
    ref_log_loss_binary = log_loss(y_true, y_prob_binary)

    def c(arr):
        return _c_metrics(arr, dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"metrics_cls_binary_{dtype_tag}_seed{seed}.npz")
    np.savez(
        out_path,
        y_true=c(y_true),
        y_pred=c(y_pred),
        y_score=c(y_score),
        sample_weight=c(sample_weight),
        y_prob_binary=c(y_prob_binary),
        ref_accuracy=c([ref_accuracy]),
        ref_accuracy_sw=c([ref_accuracy_sw]),
        ref_confusion=c(ref_confusion),
        ref_confusion_sw=c(ref_confusion_sw),
        ref_precision_binary=c([ref_precision_binary]),
        ref_recall_binary=c([ref_recall_binary]),
        ref_f1_binary=c([ref_f1_binary]),
        ref_precision_binary_sw=c([ref_precision_binary_sw]),
        ref_recall_binary_sw=c([ref_recall_binary_sw]),
        ref_f1_binary_sw=c([ref_f1_binary_sw]),
        ref_roc_auc=c([ref_roc_auc]),
        ref_roc_auc_sw=c([ref_roc_auc_sw]),
        ref_pr_precision=c(pr_precision),
        ref_pr_recall=c(pr_recall),
        ref_pr_thresholds=c(pr_thresholds),
        ref_pr_precision_sw=c(pr_precision_sw),
        ref_pr_recall_sw=c(pr_recall_sw),
        ref_pr_thresholds_sw=c(pr_thresholds_sw),
        ref_log_loss_binary=c([ref_log_loss_binary]),
    )
    return out_path


def gen_metrics_classification_multiclass(seed: int = _METRICS_FIXTURE_SEED, dtype=np.float32) -> str:
    """Generate the 3-class classification metrics oracle fixture
    (METR-CLS-01..08). Requires ``scikit-learn==1.9.0`` (Q6 resolution).

    OvO+`sample_weight` probed under scikit-learn==1.9.0: RAISES
    (`ValueError: sample_weight is not supported for multiclass one-vs-one
    ROC AUC score, ...`) — Branch A (Plan-Check Issue 2, Q10). No
    `ref_roc_auc_ovo_{macro,weighted}_sw` array is generated here; the
    weighted OvR pair (`ref_roc_auc_ovr_{macro,weighted}_sw`) IS always
    generated (Plan-Check Issue 1 — OvR carries no carve-out).

    Also carries a `labels`-reorder triple (`y_true_labelreorder`/
    `y_pred_labelreorder`/`labels_reorder=[2,0,1]`, Plan-Check Issue 6) with
    `ref_{precision,recall,f1}_labelreorder` computed via
    `average='macro'` — proves `labels` is ACCEPTED in a non-lexicographic
    order without erroring (macro's mean-over-classes is order-invariant when
    `labels` is a full permutation of the observed class set, so this is a
    "labels order is accepted, value stays correct" acceptance test rather
    than a "column semantics change" test; see the `average=None` per-class
    tests for order-sensitive coverage).
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    rng = np.random.default_rng(seed + 202)
    n = 60
    y_true = rng.integers(0, 3, size=n)
    y_true[:3] = [0, 1, 2]  # ensure every class present at least once
    flip = rng.random(n) < 0.25
    y_pred = np.where(flip, (y_true + rng.integers(1, 3, size=n)) % 3, y_true).astype(np.int64)
    y_true = y_true.astype(np.int64)
    raw_proba = rng.random((n, 3))
    y_proba = raw_proba / raw_proba.sum(axis=1, keepdims=True)
    sample_weight = rng.uniform(0.5, 2.5, size=n)

    ref_accuracy = accuracy_score(y_true, y_pred)
    ref_accuracy_sw = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    ref_confusion = confusion_matrix(y_true, y_pred)

    averages = {"macro": "macro", "micro": "micro", "weighted": "weighted", "none": None}
    ref_precision = {
        k: precision_score(y_true, y_pred, average=v) for k, v in averages.items()
    }
    ref_recall = {k: recall_score(y_true, y_pred, average=v) for k, v in averages.items()}
    ref_f1 = {k: f1_score(y_true, y_pred, average=v) for k, v in averages.items()}

    ref_precision_macro_sw = precision_score(
        y_true, y_pred, average="macro", sample_weight=sample_weight
    )
    ref_recall_macro_sw = recall_score(
        y_true, y_pred, average="macro", sample_weight=sample_weight
    )
    ref_f1_macro_sw = f1_score(y_true, y_pred, average="macro", sample_weight=sample_weight)

    ref_log_loss = log_loss(y_true, y_proba)
    ref_log_loss_sw = log_loss(y_true, y_proba, sample_weight=sample_weight)

    ref_roc_auc_ovr_macro = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    ref_roc_auc_ovr_weighted = roc_auc_score(
        y_true, y_proba, multi_class="ovr", average="weighted"
    )
    ref_roc_auc_ovo_macro = roc_auc_score(y_true, y_proba, multi_class="ovo", average="macro")
    ref_roc_auc_ovo_weighted = roc_auc_score(
        y_true, y_proba, multi_class="ovo", average="weighted"
    )
    ref_roc_auc_ovr_macro_sw = roc_auc_score(
        y_true, y_proba, multi_class="ovr", average="macro", sample_weight=sample_weight
    )
    ref_roc_auc_ovr_weighted_sw = roc_auc_score(
        y_true, y_proba, multi_class="ovr", average="weighted", sample_weight=sample_weight
    )
    # Branch A (see module docstring above): OvO + sample_weight RAISES under
    # scikit-learn==1.9.0 — no ref_roc_auc_ovo_*_sw is computed/saved.

    labels_reorder = np.array([2, 0, 1], dtype=np.int64)
    y_true_labelreorder = y_true.copy()
    y_pred_labelreorder = y_pred.copy()
    ref_precision_labelreorder = precision_score(
        y_true_labelreorder, y_pred_labelreorder, labels=labels_reorder, average="macro"
    )
    ref_recall_labelreorder = recall_score(
        y_true_labelreorder, y_pred_labelreorder, labels=labels_reorder, average="macro"
    )
    ref_f1_labelreorder = f1_score(
        y_true_labelreorder, y_pred_labelreorder, labels=labels_reorder, average="macro"
    )

    def c(arr):
        return _c_metrics(arr, dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"metrics_cls_multiclass_{dtype_tag}_seed{seed}.npz")
    np.savez(
        out_path,
        y_true=c(y_true),
        y_pred=c(y_pred),
        y_proba=c(y_proba),
        sample_weight=c(sample_weight),
        ref_accuracy=c([ref_accuracy]),
        ref_accuracy_sw=c([ref_accuracy_sw]),
        ref_confusion=c(ref_confusion),
        ref_precision_macro=c([ref_precision["macro"]]),
        ref_precision_micro=c([ref_precision["micro"]]),
        ref_precision_weighted=c([ref_precision["weighted"]]),
        ref_precision_none=c(ref_precision["none"]),
        ref_recall_macro=c([ref_recall["macro"]]),
        ref_recall_micro=c([ref_recall["micro"]]),
        ref_recall_weighted=c([ref_recall["weighted"]]),
        ref_recall_none=c(ref_recall["none"]),
        ref_f1_macro=c([ref_f1["macro"]]),
        ref_f1_micro=c([ref_f1["micro"]]),
        ref_f1_weighted=c([ref_f1["weighted"]]),
        ref_f1_none=c(ref_f1["none"]),
        ref_precision_macro_sw=c([ref_precision_macro_sw]),
        ref_recall_macro_sw=c([ref_recall_macro_sw]),
        ref_f1_macro_sw=c([ref_f1_macro_sw]),
        ref_log_loss=c([ref_log_loss]),
        ref_log_loss_sw=c([ref_log_loss_sw]),
        ref_roc_auc_ovr_macro=c([ref_roc_auc_ovr_macro]),
        ref_roc_auc_ovr_weighted=c([ref_roc_auc_ovr_weighted]),
        ref_roc_auc_ovo_macro=c([ref_roc_auc_ovo_macro]),
        ref_roc_auc_ovo_weighted=c([ref_roc_auc_ovo_weighted]),
        ref_roc_auc_ovr_macro_sw=c([ref_roc_auc_ovr_macro_sw]),
        ref_roc_auc_ovr_weighted_sw=c([ref_roc_auc_ovr_weighted_sw]),
        y_true_labelreorder=c(y_true_labelreorder),
        y_pred_labelreorder=c(y_pred_labelreorder),
        labels_reorder=c(labels_reorder),
        ref_precision_labelreorder=c([ref_precision_labelreorder]),
        ref_recall_labelreorder=c([ref_recall_labelreorder]),
        ref_f1_labelreorder=c([ref_f1_labelreorder]),
    )
    return out_path


def gen_metrics_classification_degenerate(seed: int = _METRICS_FIXTURE_SEED) -> str:
    """Generate the hand-built classification-metrics degenerate fixture
    (empty-class confusion, all-one-class confusion, zero-division P/R/F1,
    single-sample accuracy, single-class roc_auc error-gate inputs, log_loss
    0/1-probability clipping, log_loss `labels`-reorder). f64 only (SPEC §6:
    exact/integer-valued tier, hand-built tiny arrays — no f32 variant
    needed). Requires ``scikit-learn==1.9.0`` (Q6 resolution).

    `log_loss` clip-vs-renormalize probed under scikit-learn==1.9.0 against a
    row that does NOT sum to 1 (`[0.3, 0.3]`, sum 0.6): sklearn's value
    (`0.857399...`) matches the CLIP-ONLY formula `-mean(log(p[true]))`
    exactly, NOT the row-renormalized value (`0.490414...`) — sklearn clips
    (and warns) but does not renormalize. `log_loss`'s Rust implementation
    (TASK-08) must match clip-only.

    `log_loss`'s `labels` parameter probed for reorder semantics: passing
    `labels=[1, 0]` (non-lexicographic) produces the IDENTICAL value to
    `labels=[0, 1]` (sorted) and to omitting `labels` entirely — sklearn
    warns ("assumes labels are ordered lexicographically") but does NOT
    remap which probability COLUMN is treated as which class; column `j`
    is always the `j`-th smallest label in the resolved class set,
    regardless of the order `labels` is passed in. TASK-08's `log_loss`
    must mirror this: `labels` (when given) defines the accepted class SET
    (sorted internally for column indexing), not a column permutation.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
    )

    y_true_empty = np.array([0, 1, 0, 1], dtype=np.int64)
    y_pred_empty = np.array([0, 0, 1, 1], dtype=np.int64)
    labels_empty = np.array([0, 1, 2], dtype=np.int64)
    ref_confusion_empty = confusion_matrix(y_true_empty, y_pred_empty, labels=labels_empty)

    y_true_one = np.array([3, 3, 3], dtype=np.int64)
    y_pred_one = np.array([3, 3, 3], dtype=np.int64)
    ref_confusion_one = confusion_matrix(y_true_one, y_pred_one)

    # precision zero-division: positive class (1) never predicted.
    y_true_zp = np.array([1, 1, 0, 0], dtype=np.int64)
    y_pred_zp = np.array([0, 0, 0, 0], dtype=np.int64)
    ref_precision_zerodiv = precision_score(y_true_zp, y_pred_zp, pos_label=1, zero_division=0)

    # recall zero-division: positive class (1) never appears in y_true (no
    # true positives are even possible).
    y_true_zr = np.array([0, 0, 0, 0], dtype=np.int64)
    y_pred_zr = np.array([1, 0, 1, 0], dtype=np.int64)
    ref_recall_zerodiv = recall_score(y_true_zr, y_pred_zr, pos_label=1, zero_division=0)

    # f1 zero-division: positive class (1) absent from both y_true and y_pred
    # (both precision's and recall's denominators are zero).
    y_true_zf = np.array([0, 0], dtype=np.int64)
    y_pred_zf = np.array([0, 0], dtype=np.int64)
    ref_f1_zerodiv = f1_score(y_true_zf, y_pred_zf, pos_label=1, zero_division=0)

    y_true_single_match = np.array([1], dtype=np.int64)
    y_pred_single_match = np.array([1], dtype=np.int64)
    ref_acc_single_match = accuracy_score(y_true_single_match, y_pred_single_match)
    y_true_single_mismatch = np.array([1], dtype=np.int64)
    y_pred_single_mismatch = np.array([0], dtype=np.int64)
    ref_acc_single_mismatch = accuracy_score(y_true_single_mismatch, y_pred_single_mismatch)

    # Single-class roc_auc input: NO ref value is stored (an error gate, SPEC
    # §6). Empirically (scikit-learn==1.9.0), roc_auc_score on this input
    # does NOT raise — it emits UndefinedMetricWarning and returns `nan`.
    # mlrs intentionally diverges here: `roc_auc_score_binary` returns
    # `Err(MetricError::SingleClassRocAuc)` (mapped to `PyValueError` at the
    # PyO3 boundary) rather than a silent NaN, matching the plan's explicit
    # "gate the error, not a value" design intent (SPEC §9 risk 6) and the
    # project's broader fail-closed philosophy. This is a DOCUMENTED,
    # deliberate deviation from raw sklearn's own (NaN + warning) behavior on
    # this specific degenerate input, not an oracle-fidelity violation (no
    # numeric value is being compared either way).
    y_true_singleclass = np.array([1, 1, 1, 1], dtype=np.int64)
    y_score_singleclass = np.array([0.1, 0.4, 0.6, 0.9])

    # log_loss 0.0/1.0-probability clipping degenerate.
    y_true_clip = np.array([0, 1, 0, 1], dtype=np.int64)
    y_prob_clip = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    ref_log_loss_clip = log_loss(y_true_clip, y_prob_clip, labels=[0, 1])

    # log_loss `labels`-reorder acceptance (Plan-Check Issue 6): non-sorted
    # `labels=[1, 0]` must be ACCEPTED (no error) and produce the SAME value
    # as the sorted/no-labels case (see the probe note in this docstring).
    y_true_logloss_labelreorder = np.array([0, 1, 0, 1], dtype=np.int64)
    y_prob_logloss_labelreorder = np.array(
        [[0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.9, 0.1]]
    )
    labels_logloss_reorder = np.array([1, 0], dtype=np.int64)
    ref_log_loss_labelreorder = log_loss(
        y_true_logloss_labelreorder,
        y_prob_logloss_labelreorder,
        labels=labels_logloss_reorder,
    )

    def c(arr):
        return _c_metrics(arr, np.float64)

    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"metrics_cls_degenerate_seed{seed}.npz")
    np.savez(
        out_path,
        y_true_empty=c(y_true_empty),
        y_pred_empty=c(y_pred_empty),
        labels_empty=c(labels_empty),
        ref_confusion_empty=c(ref_confusion_empty),
        y_true_one=c(y_true_one),
        y_pred_one=c(y_pred_one),
        ref_confusion_one=c(ref_confusion_one),
        y_true_zp=c(y_true_zp),
        y_pred_zp=c(y_pred_zp),
        ref_precision_zerodiv=c([ref_precision_zerodiv]),
        y_true_zr=c(y_true_zr),
        y_pred_zr=c(y_pred_zr),
        ref_recall_zerodiv=c([ref_recall_zerodiv]),
        y_true_zf=c(y_true_zf),
        y_pred_zf=c(y_pred_zf),
        ref_f1_zerodiv=c([ref_f1_zerodiv]),
        y_true_single_match=c(y_true_single_match),
        y_pred_single_match=c(y_pred_single_match),
        ref_acc_single_match=c([ref_acc_single_match]),
        y_true_single_mismatch=c(y_true_single_mismatch),
        y_pred_single_mismatch=c(y_pred_single_mismatch),
        ref_acc_single_mismatch=c([ref_acc_single_mismatch]),
        y_true_singleclass=c(y_true_singleclass),
        y_score_singleclass=c(y_score_singleclass),
        y_true_clip=c(y_true_clip),
        y_prob_clip=c(y_prob_clip),
        ref_log_loss_clip=c([ref_log_loss_clip]),
        y_true_logloss_labelreorder=c(y_true_logloss_labelreorder),
        y_prob_logloss_labelreorder=c(y_prob_logloss_labelreorder),
        labels_logloss_reorder=c(labels_logloss_reorder),
        ref_log_loss_labelreorder=c([ref_log_loss_labelreorder]),
    )
    return out_path


def gen_metrics_regression(seed: int = _METRICS_FIXTURE_SEED, dtype=np.float32) -> str:
    """Generate the regression metrics oracle fixture (METR-REG-01..03).
    Requires ``scikit-learn==1.9.0`` (Q6 resolution). Single-output only (1-D
    `y_true`/`y_pred`) — no 2-D array anywhere in this generator, per SPEC §2's
    multioutput non-goal.

    `ref_r2_const` is read off the ACTUAL sklearn-computed value (SPEC §5 REG
    note, SPEC §9 risk 5), not hand-derived: constant `y_true_const` with a
    NON-exact-matching `y_pred_const` yields sklearn's documented `0.0`
    (verified against the installed scikit-learn==1.9.0, not assumed).
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rng = np.random.default_rng(seed + 303)
    n = 30
    y_true = rng.uniform(-5.0, 5.0, size=n)
    y_pred = y_true + rng.normal(0.0, 0.5, size=n)
    sample_weight = rng.uniform(0.5, 2.5, size=n)

    ref_r2 = r2_score(y_true, y_pred)
    ref_r2_sw = r2_score(y_true, y_pred, sample_weight=sample_weight)
    ref_mse = mean_squared_error(y_true, y_pred)
    ref_mse_sw = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    ref_mae = mean_absolute_error(y_true, y_pred)
    ref_mae_sw = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)

    # Constant-target degenerate: y_true_const is all-equal (ss_tot == 0);
    # y_pred_const does NOT exactly match it.
    y_true_const = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    y_pred_const = np.array([5.0, 5.1, 4.9, 5.0, 5.2])
    ref_r2_const = r2_score(y_true_const, y_pred_const)

    # Perfect-prediction degenerate.
    y_perfect = rng.uniform(-3.0, 3.0, size=10)
    ref_r2_perfect = r2_score(y_perfect, y_perfect)
    ref_mse_perfect = mean_squared_error(y_perfect, y_perfect)
    ref_mae_perfect = mean_absolute_error(y_perfect, y_perfect)

    def c(arr):
        return _c_metrics(arr, dtype)

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"metrics_reg_{dtype_tag}_seed{seed}.npz")
    np.savez(
        out_path,
        y_true=c(y_true),
        y_pred=c(y_pred),
        sample_weight=c(sample_weight),
        ref_r2=c([ref_r2]),
        ref_r2_sw=c([ref_r2_sw]),
        ref_mse=c([ref_mse]),
        ref_mse_sw=c([ref_mse_sw]),
        ref_mae=c([ref_mae]),
        ref_mae_sw=c([ref_mae_sw]),
        y_true_const=c(y_true_const),
        y_pred_const=c(y_pred_const),
        ref_r2_const=c([ref_r2_const]),
        y_perfect=c(y_perfect),
        ref_r2_perfect=c([ref_r2_perfect]),
        ref_mse_perfect=c([ref_mse_perfect]),
        ref_mae_perfect=c([ref_mae_perfect]),
    )
    return out_path


def _tree_shap_arrays(model, is_classifier):
    """Flatten a fitted sklearn forest's per-tree arrays (children/feature/
    threshold/value/node_sample_weight) plus per-tree node counts — the SAME
    layout `ForestInference.load_from_sklearn` / the Rust `TreeSpec` slicer
    consume (SHAP-01)."""
    cl, cr, fe, th, va, nsw, counts = [], [], [], [], [], [], []
    for est in model.estimators_:
        t = est.tree_
        counts.append(int(t.node_count))
        cl.append(np.asarray(t.children_left, dtype=np.int64))
        cr.append(np.asarray(t.children_right, dtype=np.int64))
        fe.append(np.asarray(t.feature, dtype=np.int64))
        th.append(np.asarray(t.threshold, dtype=np.float64))
        nsw.append(np.asarray(t.weighted_n_node_samples, dtype=np.float64))
        v = np.asarray(t.value, dtype=np.float64)  # (n_nodes, 1, n_values)
        va.append(v[:, 0, :].reshape(-1) if is_classifier else v[:, 0, 0].reshape(-1))
    return (
        np.concatenate(cl),
        np.concatenate(cr),
        np.concatenate(fe),
        np.concatenate(th),
        np.concatenate(va),
        np.concatenate(nsw),
        np.array(counts, dtype=np.int64),
    )


def gen_arima(seed: int = SEED) -> str:
    """Generate the ARIMA oracle fixture (TSA-01, Phase 22).

    Simulates a stationary AR(2)/MA(1) zero-mean process, then fits
    ``statsmodels.tsa.statespace.sarimax.SARIMAX(order=(2,0,1), trend='n',
    concentrate_scale=True, enforce_stationarity=False,
    enforce_invertibility=False)`` — the EXACT state-space convention mlrs's
    ``Arima`` reproduces (verified digit-for-digit at design time; see
    `crates/mlrs-algos/src/timeseries/arima.rs` module docs).

    Two gate tiers (the TreeSHAP/t-SNE convention):
    - DETERMINISTIC: `loglik_at_true_params` — the concentrated Kalman
      log-likelihood evaluated at FIXED known parameters, no optimizer
      involved. Gated ≤1e-6.
    - BAND: the statsmodels MLE fit's `loglik`/`aicc`/`forecast` — mlrs's own
      L-BFGS (zero-start, finite-difference gradient) may converge to a
      different point on this non-convex surface, so the gate is
      "at least as good a fit" (loglik) plus a forecast band, not exact
      parameter equality.

    Stores (all float64): ``y`` (length 120), ``true_params`` (phi1,phi2,
    theta1), ``loglik_at_true_params``, ``sm_loglik``/``sm_aicc``
    (statsmodels' MLE fit), ``sm_forecast`` (5-step-ahead).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    rng = np.random.default_rng(seed)
    n = 120
    phi_true = [0.5, -0.2]
    theta_true = [0.3]
    burn = 50
    e = rng.normal(size=n + burn)
    y = np.zeros(n + burn)
    for t in range(2, n + burn):
        y[t] = phi_true[0] * y[t - 1] + phi_true[1] * y[t - 2] + e[t] + theta_true[0] * e[t - 1]
    y = y[burn:]

    # enforce_stationarity/invertibility LEFT AT THEIR DEFAULT (True): with
    # them off, statsmodels switches from exact-stationary to
    # approximate_diffuse initialization (verified empirically at design
    # time) — a DIFFERENT filter from the one `Arima` implements and is
    # gated against (module docs: exact stationary Lyapunov P1). The true
    # simulated params lie well inside the stationary/invertible region, so
    # the default constrained fit lands at essentially the same optimum an
    # unconstrained search would.
    mod = SARIMAX(y, order=(2, 0, 1), trend="n", concentrate_scale=True)
    ll_true = mod.loglike(phi_true + theta_true)
    res = mod.fit(disp=False, method="lbfgs")
    fc = res.forecast(steps=5)

    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"arima_seed{seed}.npz")
    np.savez(
        out_path,
        y=y.astype(np.float64),
        true_params=np.array(phi_true + theta_true, dtype=np.float64),
        loglik_at_true_params=np.array([ll_true], dtype=np.float64),
        sm_loglik=np.array([res.llf], dtype=np.float64),
        sm_aicc=np.array([res.aicc], dtype=np.float64),
        sm_forecast=fc.astype(np.float64),
    )
    return out_path


def gen_tree_shap(seed: int = SEED) -> list:
    """Generate the TreeSHAP oracle fixtures (SHAP-01, Phase 21): a fitted
    sklearn RandomForestClassifier + RandomForestRegressor, their per-tree
    node arrays (the SAME layout `ForestInference` imports), and
    `shap.TreeExplainer` SHAP values + expected_value on a query set.

    mlrs's Rust `ForestInference::from_trees` imports THESE arrays and
    computes `shap_values` using the EXACT `node_sample_weight` cover carried
    here — so the Rust gate is a direct ≤1e-5 replay of `shap.TreeExplainer`
    on the identical tree structure, no Python needed at test time.

    Two fixtures: ``tree_shap_classifier_seed{seed}.npz`` /
    ``tree_shap_regressor_seed{seed}.npz``. All arrays float64-cast
    (``load_npz`` rejects int arrays — ``children_left``/``children_right``/
    ``feature``/``node_counts`` are int-valued but float-typed).
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    import shap

    rng = np.random.default_rng(seed)
    paths = []

    # --- classifier ---
    x = rng.normal(size=(60, 3))
    y = ((x[:, 0] + x[:, 1] > 0).astype(int) + (x[:, 2] > 0.5).astype(int))
    m = RandomForestClassifier(n_estimators=5, max_depth=4, random_state=seed).fit(x, y)
    xq = rng.normal(size=(6, 3))
    cl, cr, fe, th, va, nsw, counts = _tree_shap_arrays(m, is_classifier=True)
    expl = shap.TreeExplainer(m)
    sv = np.asarray(expl.shap_values(xq))  # (q, f, n_classes)
    ev = np.asarray(expl.expected_value, dtype=np.float64)
    n_classes = int(m.n_classes_)

    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"tree_shap_classifier_seed{seed}.npz")
    np.savez(
        out_path,
        X=x.astype(np.float64),
        Xq=xq.astype(np.float64),
        children_left=cl.astype(np.float64),
        children_right=cr.astype(np.float64),
        feature=fe.astype(np.float64),
        threshold=th,
        value=va,
        node_sample_weight=nsw,
        node_counts=counts.astype(np.float64),
        n_values=np.array([n_classes], dtype=np.float64),
        n_features=np.array([x.shape[1]], dtype=np.float64),
        shap_values=sv.reshape(-1),
        expected_value=ev,
        predict_proba=m.predict_proba(xq).astype(np.float64).reshape(-1),
    )
    paths.append(out_path)

    # --- regressor ---
    x = rng.normal(size=(50, 3))
    y = x @ rng.normal(size=3)
    m = RandomForestRegressor(n_estimators=4, max_depth=4, random_state=seed).fit(x, y)
    xq = rng.normal(size=(6, 3))
    cl, cr, fe, th, va, nsw, counts = _tree_shap_arrays(m, is_classifier=False)
    expl = shap.TreeExplainer(m)
    sv = np.asarray(expl.shap_values(xq))  # (q, f)
    ev = np.asarray(np.atleast_1d(expl.expected_value), dtype=np.float64)

    out_path = os.path.join(_FIXTURE_DIR, f"tree_shap_regressor_seed{seed}.npz")
    np.savez(
        out_path,
        X=x.astype(np.float64),
        Xq=xq.astype(np.float64),
        children_left=cl.astype(np.float64),
        children_right=cr.astype(np.float64),
        feature=fe.astype(np.float64),
        threshold=th,
        value=va,
        node_sample_weight=nsw,
        node_counts=counts.astype(np.float64),
        n_values=np.array([1.0]),
        n_features=np.array([x.shape[1]], dtype=np.float64),
        shap_values=sv.reshape(-1),
        expected_value=ev,
        predict=m.predict(xq).astype(np.float64),
    )
    paths.append(out_path)
    return paths


def gen_agglomerative(seed: int = SEED, dtype=np.float32, metric: str = "euclidean") -> str:
    """Generate one AgglomerativeClustering oracle fixture (AGGLO-01).

    Fits ``sklearn.cluster.AgglomerativeClustering(linkage='single')`` (1.9.0)
    on a three-blob design at ``n_clusters ∈ {2, 3, 5}`` and stores the EXACT
    ``labels_`` per cut plus the shared ``children_`` dendrogram. mlrs ports the
    unstructured single-linkage pipeline line-for-line (`mst_linkage_core` /
    scipy `label` / `_hc_cut` incl. Python-heapq array order), so the Rust gate
    is EXACT equality — no permutation matching.

    The design spreads the blobs and adds a per-row unique offset so all MST
    edge weights are distinct (the HDBSCAN Pitfall-1 option-2 convention): the
    merge order — and hence children/labels — is then stable under the f32
    cast and the device GEMM-expansion Euclidean roundoff.

    Stores (ALL arrays float-cast — ``load_npz`` rejects int arrays): ``X``
    (``c()``-cast to the fixture dtype), ``children`` ((n-1)×2, float64),
    ``labels_k2`` / ``labels_k3`` / ``labels_k5`` (float64). The metric rides
    the filename: ``agglomerative_{metric}_{dtype}_seed{seed}.npz``.
    """
    from sklearn.cluster import AgglomerativeClustering as SkAgglo

    rng = np.random.default_rng(seed)
    # Small-magnitude centers: the device Euclidean path is a GEMM expansion
    # (‖x‖² + ‖y‖² − 2x·y), whose f32 cancellation error scales with ‖x‖² — at
    # center magnitude ~5 the absolute distance noise is ~1e-5, far under the
    # asserted 2e-3 MST-edge-weight gap below.
    centers = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [5.0, 5.0, -5.0, 2.5, -2.5],
            [-5.5, 3.5, 4.5, -4.5, 2.0],
        ]
    )
    # Retry jitter draws until ALL sorted single-linkage MST edge weights are
    # pairwise separated by > 2e-3 (deterministic loop; the gap makes the merge
    # order — and hence children/labels — exact under the f32 cast AND the
    # device GEMM-expansion roundoff). ~57 within-blob edges over an O(1) weight
    # range: a few draws suffice.
    for _attempt in range(200):
        blocks = [
            centers[b] + 0.8 * rng.standard_normal((20, 5)) for b in range(3)
        ]
        x_design = np.vstack(blocks)
        from scipy.cluster.hierarchy import linkage as _scipy_linkage
        from scipy.spatial.distance import pdist as _pdist

        weights = np.sort(_scipy_linkage(_pdist(x_design), method="single")[:, 2])
        if np.diff(weights).min() > 2e-3:
            break
    else:
        raise RuntimeError("no tie-free agglomerative design found in 200 draws")

    labels_by_k = {}
    children = None
    for k in (2, 3, 5):
        model = SkAgglo(n_clusters=k, metric=metric, linkage="single").fit(x_design)
        labels_by_k[k] = model.labels_
        children = model.children_  # cut-independent (same dendrogram)

    def c(arr):
        return np.asarray(arr, dtype=dtype)

    dtype_tag = "f32" if dtype == np.float32 else "f64"
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"agglomerative_{metric}_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(
        out_path,
        X=c(x_design),
        children=children.astype(np.float64),
        labels_k2=labels_by_k[2].astype(np.float64),
        labels_k3=labels_by_k[3].astype(np.float64),
        labels_k5=labels_by_k[5].astype(np.float64),
    )
    return out_path


def gen_tsne(seed: int = SEED, dtype=np.float32) -> str:
    """Generate the TSNE oracle fixture (TSNE-01).

    Two gate tiers (the UMAP convention):
    - DETERMINISTIC: the dense joint-probability matrix ``P`` from sklearn's
      ``_joint_probabilities`` (f32-rounded distances, f64 search — mlrs ports
      it line-for-line, gated ≤1e-5).
    - BAND: ``sklearn.manifold.TSNE(method='exact', init='pca')`` embedding's
      ``kl_divergence_`` and ``trustworthiness`` — mlrs's own embedding must
      reach the same neighborhood-preservation band (chaotic dynamics + a
      different PCA solver make exact equality meaningless).

    Stores (ALL float — ``load_npz`` rejects ints): ``X`` (``c()``-cast),
    ``P`` (dense n×n, f64), ``embedding`` (n×2, f64), ``kl`` (len-1),
    ``trust`` (len-1, n_neighbors=5), ``perplexity`` (len-1).
    """
    from scipy.spatial.distance import squareform
    from sklearn.manifold import TSNE as SkTSNE
    from sklearn.manifold import trustworthiness
    from sklearn.manifold._t_sne import _joint_probabilities
    from sklearn.metrics import pairwise_distances

    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [6.0, 6.0, -6.0, 3.0, -3.0],
            [-7.0, 4.0, 5.0, -5.0, 2.5],
        ]
    )
    x_design = np.vstack(
        [centers[b] + 0.7 * rng.standard_normal((16, 5)) for b in range(3)]
    )  # 48 × 5, three well-separated blobs

    perplexity = 10.0
    dsq = pairwise_distances(x_design, metric="euclidean", squared=True)
    p_cond = _joint_probabilities(dsq, perplexity, 0)  # condensed
    p_dense = squareform(p_cond)  # dense n×n, diagonal 0

    model = SkTSNE(
        n_components=2,
        perplexity=perplexity,
        method="exact",
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    )
    emb = model.fit_transform(x_design)
    trust = trustworthiness(x_design, emb, n_neighbors=5)

    def c(arr):
        return np.asarray(arr, dtype=dtype)

    dtype_tag = "f32" if dtype == np.float32 else "f64"
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"tsne_{dtype_tag}_seed{seed}.npz")
    np.savez(
        out_path,
        X=c(x_design),
        P=p_dense.astype(np.float64),
        embedding=np.asarray(emb, dtype=np.float64),
        kl=np.array([model.kl_divergence_], dtype=np.float64),
        trust=np.array([trust], dtype=np.float64),
        perplexity=np.array([perplexity], dtype=np.float64),
    )
    return out_path


# --- Preprocessing scalers (PREP-01, Phase 24) ----------------------------- #
# Every design matrix pins one CONSTANT (or all-zero, for MaxAbsScaler) column
# so the fixture also exercises the degenerate zero-scale gate
# (`_handle_zeros_in_scale` — a constant column's scale/range/IQR must become
# `1`, never divide by `0`). `n=60, d=5` mirrors the other small closed-form
# estimator fixtures (PCA tall etc).


def gen_standard_scaler(seed: int = SEED, dtype=np.float32) -> str:
    """`StandardScaler` fixture (PREP-01): mean_/var_/scale_ + transform +
    inverse_transform, column 2 held constant (zero-variance gate)."""
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((60, 5))
    x[:, 2] = 3.0
    est = StandardScaler().fit(x)
    transformed = est.transform(x)
    inv = est.inverse_transform(transformed)

    def c(arr):
        return np.asarray(arr, dtype=dtype)

    dtype_tag = "f32" if dtype == np.float32 else "f64"
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"standard_scaler_{dtype_tag}_seed{seed}.npz")
    np.savez(
        out_path,
        X=c(x),
        mean_=c(est.mean_),
        var_=c(est.var_),
        scale_=c(est.scale_),
        transform=c(transformed),
        inverse=c(inv),
    )
    return out_path


def gen_min_max_scaler(seed: int = SEED, dtype=np.float32) -> str:
    """`MinMaxScaler` fixture (PREP-01, `feature_range=(-2, 3)`): data_min_/
    data_max_/scale_/min_ + transform + inverse_transform, column 2 constant."""
    from sklearn.preprocessing import MinMaxScaler

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((60, 5))
    x[:, 2] = -1.5
    feature_range = (-2.0, 3.0)
    est = MinMaxScaler(feature_range=feature_range).fit(x)
    transformed = est.transform(x)
    inv = est.inverse_transform(transformed)

    def c(arr):
        return np.asarray(arr, dtype=dtype)

    dtype_tag = "f32" if dtype == np.float32 else "f64"
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"min_max_scaler_{dtype_tag}_seed{seed}.npz")
    np.savez(
        out_path,
        X=c(x),
        feature_range=c(feature_range),
        data_min_=c(est.data_min_),
        data_max_=c(est.data_max_),
        scale_=c(est.scale_),
        min_=c(est.min_),
        transform=c(transformed),
        inverse=c(inv),
    )
    return out_path


def gen_max_abs_scaler(seed: int = SEED, dtype=np.float32) -> str:
    """`MaxAbsScaler` fixture (PREP-01): max_abs_/scale_ + transform +
    inverse_transform, column 2 all-zero (zero-scale gate)."""
    from sklearn.preprocessing import MaxAbsScaler

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((60, 5))
    x[:, 2] = 0.0
    est = MaxAbsScaler().fit(x)
    transformed = est.transform(x)
    inv = est.inverse_transform(transformed)

    def c(arr):
        return np.asarray(arr, dtype=dtype)

    dtype_tag = "f32" if dtype == np.float32 else "f64"
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"max_abs_scaler_{dtype_tag}_seed{seed}.npz")
    np.savez(
        out_path,
        X=c(x),
        max_abs_=c(est.max_abs_),
        scale_=c(est.scale_),
        transform=c(transformed),
        inverse=c(inv),
    )
    return out_path


def gen_robust_scaler(seed: int = SEED, dtype=np.float32) -> str:
    """`RobustScaler` fixture (PREP-01, default `quantile_range=(25, 75)`):
    center_/scale_ + transform + inverse_transform, column 2 constant."""
    from sklearn.preprocessing import RobustScaler

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((60, 5))
    x[:, 2] = 4.0
    est = RobustScaler().fit(x)
    transformed = est.transform(x)
    inv = est.inverse_transform(transformed)

    def c(arr):
        return np.asarray(arr, dtype=dtype)

    dtype_tag = "f32" if dtype == np.float32 else "f64"
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"robust_scaler_{dtype_tag}_seed{seed}.npz")
    np.savez(
        out_path,
        X=c(x),
        center_=c(est.center_),
        scale_=c(est.scale_),
        transform=c(transformed),
        inverse=c(inv),
    )
    return out_path


def gen_normalizer(seed: int = SEED, dtype=np.float32, norm: str = "l2") -> str:
    """`Normalizer` fixture (PREP-01, `norm` in `{'l1', 'l2', 'max'}`): row 0
    forced all-zero (zero-norm gate — the row must transform unchanged)."""
    from sklearn.preprocessing import Normalizer

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((60, 5))
    x[0, :] = 0.0
    transformed = Normalizer(norm=norm).fit_transform(x)

    def c(arr):
        return np.asarray(arr, dtype=dtype)

    dtype_tag = "f32" if dtype == np.float32 else "f64"
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"normalizer_{norm}_{dtype_tag}_seed{seed}.npz")
    np.savez(out_path, X=c(x), transform=c(transformed))
    return out_path


def gen_binarizer(seed: int = SEED, dtype=np.float32) -> str:
    """`Binarizer` fixture (PREP-01, `threshold=0.5`), including exact-threshold
    entries (sklearn's `>` is STRICT — a tie must binarize to 0)."""
    from sklearn.preprocessing import Binarizer

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((60, 5))
    x[1, :] = 0.5  # exactly the threshold: must binarize to 0 (strict '>')
    threshold = 0.5
    transformed = Binarizer(threshold=threshold).fit_transform(x)

    def c(arr):
        return np.asarray(arr, dtype=dtype)

    dtype_tag = "f32" if dtype == np.float32 else "f64"
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"binarizer_{dtype_tag}_seed{seed}.npz")
    np.savez(out_path, X=c(x), threshold=c([threshold]), transform=c(transformed))
    return out_path



# GaussianMixture oracle geometry (MIX-01). Three WELL-SEPARATED blobs so the
# likelihood surface has one dominant optimum and every `init_params` route
# converges to the SAME fixed point — which is what makes an init-by-init
# comparison meaningful across two different RNGs (D-09: numpy's `Generator`
# stream is not reproducible from Rust, so the oracle pins the ANSWER, not the
# path to it).
GMM_N_SAMPLES, GMM_N_FEATURES, GMM_K = 300, 4, 3
# A query block, disjoint from the training design, for the scoring surface.
GMM_N_QUERY = 40
# The convergence cases run to machine precision so BOTH engines sit on the
# same stationary point rather than stopping at different places inside
# sklearn's default `tol = 1e-3` band (which alone permits ~1e-3 parameter
# disagreement and would make a 1e-5 comparison meaningless).
GMM_TOL_TIGHT = 1e-12
GMM_MAX_ITER_TIGHT = 2000

GMM_COV_TYPES = ("full", "tied", "diag", "spherical")
GMM_INIT_PARAMS = ("kmeans", "k-means++", "random", "random_from_data")


def _gmm_design(seed: int, dtype):
    """The shared GaussianMixture design + query block, dtype-round-tripped."""
    rng = np.random.default_rng(seed + 909)
    # The separation is TUNED, not arbitrary. Measured on this design across
    # `scale in (9, 5, 3.5, 3)`: at 9 the blobs are so distinct that `kmeans`
    # init converges in 2 iterations and the EM loop is barely exercised; at 3.5
    # the `full` optimum stops being init-independent (`kmeans` and `random`
    # land on different local maxima) and the fixture's premise breaks. 5 is the
    # widest separation that still costs 3-321 iterations depending on the
    # init — real EM work, one optimum.
    centers = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [5.0, 5.0, 0.0, 0.0],
            [-5.0, 10.0 / 3.0, 5.0, -10.0 / 3.0],
        ]
    )
    per = GMM_N_SAMPLES // GMM_K
    x = np.vstack(
        [c + 0.9 * rng.standard_normal((per, GMM_N_FEATURES)) for c in centers]
    )
    # Round-trip through the fixture dtype BEFORE fitting, so the reference is
    # the answer for the exact bytes the Rust test reads back.
    x = x.astype(dtype).astype(np.float64)
    xq = (
        centers[rng.integers(0, GMM_K, GMM_N_QUERY)]
        + 1.2 * rng.standard_normal((GMM_N_QUERY, GMM_N_FEATURES))
    )
    xq = xq.astype(dtype).astype(np.float64)
    return x, xq


def _gmm_injected(cov_type: str, seed: int):
    """A FIXED (weights, means, precisions) init for the exact-parity cases.

    With all three injected there is no RNG anywhere in the fit, so mlrs and
    sklearn run byte-comparable EM from the same starting point and every fitted
    attribute — including ``lower_bound_`` and ``n_iter_`` — must agree to 1e-5.
    These cases are the ones that pin the ALGORITHM; the `init_params` cases pin
    the initializations.
    """
    rng = np.random.default_rng(seed + 4242)
    k, d = GMM_K, GMM_N_FEATURES
    weights = np.full(k, 1.0 / k)
    # Offset from the true centers so EM has real work to do, but close enough
    # that it lands in the global optimum.
    means = np.array(
        [
            [1.5, -1.0, 0.5, 0.5],
            [7.0, 8.0, 1.0, -1.0],
            [-7.5, 5.0, 8.0, -5.0],
        ]
    )
    base = np.eye(d) + 0.15 * rng.standard_normal((d, d))
    spd = base @ base.T + d * np.eye(d)
    if cov_type == "full":
        prec = np.stack([spd * (1.0 + 0.1 * i) for i in range(k)])
    elif cov_type == "tied":
        prec = spd
    elif cov_type == "diag":
        prec = np.stack([np.diag(spd) * (1.0 + 0.1 * i) for i in range(k)])
    else:
        prec = np.array([1.0 + 0.1 * i for i in range(k)])
    return weights, means, prec


def gen_gaussian_mixture(seed: int = SEED, dtype=np.float32) -> str:
    """Generate the GaussianMixture full-parameter-surface fixture (MIX-01).

    Two families of cases, because the estimator has two independently testable
    halves:

    1. ``{cov}_{init}`` — the 4x4 cross of ``covariance_type`` x
       ``init_params``, each fitted to machine precision. These pin that mlrs's
       four covariance parameterizations and four initializations all reach
       sklearn's optimum, up to the component permutation the init chooses.
    2. ``inj_{cov}`` — the same four covariance types with ``weights_init`` /
       ``means_init`` / ``precisions_init`` all supplied, which removes the RNG
       entirely. These pin the EM ARITHMETIC exactly: ``lower_bound_``,
       ``n_iter_``, ``converged_``, the scoring surface (``predict`` /
       ``predict_proba`` / ``score_samples``), and ``bic`` / ``aic``.
    3. ``reg{n}_{cov}`` — a ``reg_covar`` sweep on the injected init, since
       ``reg_covar`` is the one numeric hyperparameter that changes the fitted
       covariance directly rather than through convergence.

    Requires ``scikit-learn==1.9.0``.
    """
    from sklearn.mixture import GaussianMixture

    x, xq = _gmm_design(seed, dtype)
    n, d, k = GMM_N_SAMPLES, GMM_N_FEATURES, GMM_K

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    out = {"X": c(x), "Xq": c(xq)}

    def record(name, est, with_scoring: bool):
        out[f"weights_{name}"] = c(est.weights_)
        out[f"means_{name}"] = c(est.means_)
        out[f"cov_{name}"] = c(np.ravel(est.covariances_))
        out[f"prec_chol_{name}"] = c(np.ravel(est.precisions_cholesky_))
        out[f"lower_bound_{name}"] = c([est.lower_bound_])
        # sklearn's per-iteration bound trace for the WINNING restart (length
        # `n_iter_`). Recorded for every case: it is the only fitted attribute
        # that pins the SHAPE of the ascent rather than just its endpoint, so a
        # convergence rule that lands on the right answer by a different route
        # fails here and nowhere else.
        out[f"lower_bounds_{name}"] = c(np.asarray(est.lower_bounds_))
        out[f"n_iter_{name}"] = c([est.n_iter_])
        out[f"converged_{name}"] = c([1.0 if est.converged_ else 0.0])
        out[f"labels_{name}"] = c(est.predict(x))
        if with_scoring:
            out[f"predict_{name}"] = c(est.predict(xq))
            out[f"proba_{name}"] = c(np.ravel(est.predict_proba(xq)))
            out[f"score_samples_{name}"] = c(est.score_samples(xq))
            out[f"bic_{name}"] = c([est.bic(xq)])
            out[f"aic_{name}"] = c([est.aic(xq)])
            out[f"n_parameters_{name}"] = c([est._n_parameters()])

    # --- family 1: covariance_type x init_params, converged ----------------- #
    for cov in GMM_COV_TYPES:
        for init in GMM_INIT_PARAMS:
            est = GaussianMixture(
                n_components=k,
                covariance_type=cov,
                tol=GMM_TOL_TIGHT,
                max_iter=GMM_MAX_ITER_TIGHT,
                n_init=1,
                init_params=init,
                random_state=0,
            ).fit(x)
            record(f"{cov}_{init.replace('-', '')}", est, with_scoring=False)

    # --- family 2: fully injected init (no RNG anywhere) -------------------- #
    for cov in GMM_COV_TYPES:
        w0, m0, p0 = _gmm_injected(cov, seed)
        out[f"winit_{cov}"] = c(w0)
        out[f"minit_{cov}"] = c(m0)
        out[f"pinit_{cov}"] = c(np.ravel(p0))
        est = GaussianMixture(
            n_components=k,
            covariance_type=cov,
            tol=1e-8,
            max_iter=200,
            n_init=1,
            weights_init=w0,
            means_init=m0,
            precisions_init=p0,
        ).fit(x)
        record(f"inj_{cov}", est, with_scoring=True)

        # A hard single-iteration case: `max_iter=1` leaves no room for two
        # engines to converge to the same place by different routes, so it is
        # the strictest possible test of ONE E-step plus ONE M-step.
        est1 = GaussianMixture(
            n_components=k,
            covariance_type=cov,
            tol=0.0,
            max_iter=1,
            n_init=1,
            weights_init=w0,
            means_init=m0,
            precisions_init=p0,
        ).fit(x)
        record(f"iter1_{cov}", est1, with_scoring=False)

    # --- family 3: the reg_covar sweep on the injected init ----------------- #
    for i, reg in enumerate((1e-6, 1e-2, 1.0)):
        for cov in GMM_COV_TYPES:
            w0, m0, p0 = _gmm_injected(cov, seed)
            est = GaussianMixture(
                n_components=k,
                covariance_type=cov,
                tol=1e-8,
                max_iter=200,
                n_init=1,
                reg_covar=reg,
                weights_init=w0,
                means_init=m0,
                precisions_init=p0,
            ).fit(x)
            record(f"reg{i}_{cov}", est, with_scoring=False)

    # Every `init_params` route must land on the SAME optimum (up to a component
    # permutation) for a given `covariance_type` — that is the premise of family
    # 1, and it is asserted HERE so a bad fixture cannot be committed silently.
    for cov in GMM_COV_TYPES:
        ref = np.sort(out[f"lower_bound_{cov}_kmeans"].astype(np.float64))
        for init in GMM_INIT_PARAMS:
            got = out[f"lower_bound_{cov}_{init.replace('-', '')}"].astype(np.float64)
            assert np.allclose(got, ref, atol=1e-6, rtol=1e-6), (
                f"gen_gaussian_mixture: covariance_type='{cov}' init_params="
                f"'{init}' converged to lower_bound {got} but 'kmeans' reached "
                f"{ref} — the design is not separable enough for an "
                "init-independent oracle"
            )

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"gaussian_mixture_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(out_path, **out)
    return out_path


# ---------------------------------------------------------------------------
# BayesianGaussianMixture full-parameter-surface fixtures (MIX-02)
# ---------------------------------------------------------------------------

# The variational sibling shares GaussianMixture's design (`_gmm_design`) — the
# separation there was already tuned so every `init_params` route lands on ONE
# optimum, and that premise is what makes an init-by-init comparison possible
# across two irreproducible RNGs (D-09). Only the case matrix differs.
BGM_PRIOR_TYPES = ("dirichlet_process", "dirichlet_distribution")
# The case-name spelling: the Rust side cannot carry `-` in an npz key it builds
# by format!, so `k-means++` becomes `kmeans++` exactly as in the MIX-01
# fixture, and the two prior types collapse to `dp` / `dd`.
BGM_PRIOR_TAGS = {"dirichlet_process": "dp", "dirichlet_distribution": "dd"}
# The stick-breaking pin (family D) needs MORE components than the fit families,
# because the whole point is the `Σ_{j>c} nk_j` recursion — at k <= 2 it is
# nearly vacuous. Deliberately UNEQUAL counts, so a transposed or reversed
# cumulative sum cannot pass by symmetry.
# Family 1 runs `n_init` restarts, not one. That is not padding: the two
# SPARSE initializations ('k-means++' / 'random_from_data') put all their mass
# on `k` rows, which makes their first M-step a lottery over WHICH rows — and
# mlrs draws from a different stream (D-09). A single restart therefore lets the
# two engines land in different basins even where sklearn's own four routes
# agree with each other, which is exactly what `n_init` exists to defeat. Five
# restarts was the smallest count at which every compared case agrees.
BGM_N_INIT = 5
BGM_STICK_K = 5
BGM_STICK_NK = (137.0, 61.5, 12.25, 3.0, 0.5)
BGM_STICK_PRIOR = 0.37


def _bgm_scoring(out, name, est, xq):
    """Record the scoring surface of one fitted BayesianGaussianMixture."""
    out[f"predict_{name}"] = est.predict(xq)
    out[f"proba_{name}"] = np.ravel(est.predict_proba(xq))
    # NO `predict_log_proba`: sklearn's mixture estimators do not define one.
    # mlrs exposes it (the `PredictLogProba` typestate the whole crate shares),
    # so the Rust side pins it against `ln(predict_proba)` instead of against a
    # reference that does not exist.
    out[f"score_samples_{name}"] = est.score_samples(xq)
    out[f"score_{name}"] = np.array([est.score(xq)])


def _bgm_record(out, name, est, x, cast, with_scoring=False, xq=None):
    """Record every fitted attribute of one BayesianGaussianMixture.

    Includes the four variational posteriors sklearn exposes on top of
    `GaussianMixture`'s attribute set (`weight_concentration_`,
    `mean_precision_`, `degrees_of_freedom_`) and the five resolved priors,
    because those are precisely what distinguishes this estimator: a
    transcription that got the Wishart update right but the prior resolution
    wrong would produce plausible means and covariances and fail only here.
    """
    wc = est.weight_concentration_
    if est.weight_concentration_prior_type == "dirichlet_process":
        out[f"wca_{name}"] = cast(wc[0])
        out[f"wcb_{name}"] = cast(wc[1])
    else:
        out[f"wca_{name}"] = cast(wc)
    out[f"weights_{name}"] = cast(est.weights_)
    out[f"means_{name}"] = cast(est.means_)
    out[f"cov_{name}"] = cast(np.ravel(est.covariances_))
    out[f"prec_chol_{name}"] = cast(np.ravel(est.precisions_cholesky_))
    out[f"beta_{name}"] = cast(est.mean_precision_)
    out[f"dof_{name}"] = cast(np.ravel(np.atleast_1d(est.degrees_of_freedom_)))
    # The resolved priors (sklearn's `*_prior_` fitted attributes).
    out[f"pwc_{name}"] = cast([est.weight_concentration_prior_])
    out[f"pbeta_{name}"] = cast([est.mean_precision_prior_])
    out[f"pmean_{name}"] = cast(np.ravel(est.mean_prior_))
    out[f"pdof_{name}"] = cast([est.degrees_of_freedom_prior_])
    out[f"pcov_{name}"] = cast(np.ravel(np.atleast_1d(est.covariance_prior_)))
    out[f"lower_bound_{name}"] = cast([float(np.ravel(est.lower_bound_)[0])])
    out[f"lower_bounds_{name}"] = cast(np.ravel(est.lower_bounds_))
    out[f"n_iter_{name}"] = cast([est.n_iter_])
    out[f"converged_{name}"] = cast([1.0 if est.converged_ else 0.0])
    out[f"labels_{name}"] = cast(est.predict(x))
    if with_scoring:
        scored = {}
        _bgm_scoring(scored, name, est, xq)
        for key, value in scored.items():
            out[key] = cast(value)


def gen_bayesian_mixture(seed: int = SEED, dtype=np.float32) -> str:
    """Generate the BayesianGaussianMixture fixture (MIX-02).

    Four families, because the estimator has four independently testable parts
    and only two of them can be pinned by a converged end-to-end fit:

    1. ``{cov}_{init}_{ptype}`` — the 4 x 4 x 2 cross of ALL THREE string-valued
       hyperparameters, each fitted to machine precision. Pins that every
       covariance parameterization, every initialization route and both weight
       priors reach sklearn's optimum. Compared up to a component PERMUTATION,
       because the initializations use two different RNGs (D-09).
    2. ``k1{cov}_{ptype}`` / ``k1i{cov}_{ptype}`` — ``n_components=1`` with
       ``init_params='random'``. At k=1 that initialization is RNG-FREE in both
       engines (a one-column responsibility matrix normalizes to exactly 1.0
       whatever was drawn), so these are compared EXACTLY, in order, including
       every posterior, every prior, ``lower_bound_``, ``lower_bounds_``,
       ``n_iter_``, ``converged_``, and the whole scoring surface. The ``i``
       variant runs ``max_iter=1, tol=0``, leaving no room for two engines to
       reach the same place by different routes.
    3. ``pr{i}{cov}_{ptype}`` — the five prior hyperparameters swept off their
       defaults, at k=1 so the comparison stays exact. These are the parameters
       with NO analogue in ``GaussianMixture``, so nothing else in the suite
       would notice if one were ignored.
    4. ``stick_{ptype}`` — the weight-posterior arithmetic evaluated on a FIXED
       ``nk`` vector at ``k=5`` via sklearn's own ``_estimate_weights`` /
       ``_estimate_log_weights``. This family exists because family 1 CANNOT
       pin it: under ``dirichlet_process`` the stick-breaking recursion is
       order-dependent (component ``c``'s second Beta parameter sums the ``nk``
       of everything after it), so two engines that find the same clustering in
       a different order legitimately disagree on ``weight_concentration_`` and
       ``weights_``. Evaluating the recursion at a fixed, unequal ``nk`` removes
       the ordering question entirely.

    Requires ``scikit-learn==1.9.0``.
    """
    from sklearn.mixture import BayesianGaussianMixture

    x, xq = _gmm_design(seed, dtype)
    n, d, k = GMM_N_SAMPLES, GMM_N_FEATURES, GMM_K

    def c(arr):
        return np.ascontiguousarray(np.asarray(arr)).astype(dtype)

    out = {"X": c(x), "Xq": c(xq)}

    # --- family 1: covariance_type x init_params x prior_type, converged ---- #
    for cov in GMM_COV_TYPES:
        for init in GMM_INIT_PARAMS:
            for ptype in BGM_PRIOR_TYPES:
                name = f"{cov}_{init.replace('-', '')}_{BGM_PRIOR_TAGS[ptype]}"
                est = BayesianGaussianMixture(
                    n_components=k,
                    covariance_type=cov,
                    init_params=init,
                    weight_concentration_prior_type=ptype,
                    tol=GMM_TOL_TIGHT,
                    max_iter=GMM_MAX_ITER_TIGHT,
                    n_init=BGM_N_INIT,
                    random_state=0,
                ).fit(x)
                _bgm_record(out, name, est, x, c)

    # --- family 2: k=1, RNG-free, compared exactly --------------------------- #
    for cov in GMM_COV_TYPES:
        for ptype in BGM_PRIOR_TYPES:
            tag = BGM_PRIOR_TAGS[ptype]
            est = BayesianGaussianMixture(
                n_components=1,
                covariance_type=cov,
                init_params="random",
                weight_concentration_prior_type=ptype,
                tol=1e-8,
                max_iter=200,
                random_state=0,
            ).fit(x)
            _bgm_record(out, f"k1{cov}_{tag}", est, x, c, with_scoring=True, xq=xq)

            est1 = BayesianGaussianMixture(
                n_components=1,
                covariance_type=cov,
                init_params="random",
                weight_concentration_prior_type=ptype,
                tol=0.0,
                max_iter=1,
                random_state=0,
            ).fit(x)
            _bgm_record(out, f"k1i{cov}_{tag}", est1, x, c)

    # --- family 3: the prior sweep, k=1, exact ------------------------------ #
    # One non-default value per prior, each chosen far enough from the default
    # that ignoring the parameter cannot pass: gamma 100x smaller, beta0 5x
    # larger, nu0 well above `n_features`, and an m0/W0 pair unrelated to the
    # design's own moments.
    mean_prior = np.array([1.0, -2.0, 0.5, 3.0])[:d]
    cov_prior_full = np.eye(d) * 2.5 + 0.25
    prior_sweep = [
        {"weight_concentration_prior": 0.01},
        {"mean_precision_prior": 5.0},
        {"degrees_of_freedom_prior": float(d) + 3.5},
        {"mean_prior": mean_prior},
    ]
    for i, kwargs in enumerate(prior_sweep):
        for cov in GMM_COV_TYPES:
            for ptype in BGM_PRIOR_TYPES:
                tag = BGM_PRIOR_TAGS[ptype]
                est = BayesianGaussianMixture(
                    n_components=1,
                    covariance_type=cov,
                    init_params="random",
                    weight_concentration_prior_type=ptype,
                    tol=0.0,
                    max_iter=1,
                    random_state=0,
                    **kwargs,
                ).fit(x)
                _bgm_record(out, f"pr{i}{cov}_{tag}", est, x, c)
    # `covariance_prior` is swept separately: its SHAPE depends on
    # covariance_type, so it cannot ride the shared kwargs dict above.
    cov_prior_by_type = {
        "full": cov_prior_full,
        "tied": cov_prior_full,
        "diag": np.linspace(0.7, 2.2, d),
        "spherical": 1.75,
    }
    for cov in GMM_COV_TYPES:
        for ptype in BGM_PRIOR_TYPES:
            tag = BGM_PRIOR_TAGS[ptype]
            cp = cov_prior_by_type[cov]
            out[f"cpin_{cov}"] = c(np.ravel(np.atleast_1d(cp)))
            est = BayesianGaussianMixture(
                n_components=1,
                covariance_type=cov,
                init_params="random",
                weight_concentration_prior_type=ptype,
                tol=0.0,
                max_iter=1,
                random_state=0,
                covariance_prior=cp,
            ).fit(x)
            _bgm_record(out, f"pr4{cov}_{tag}", est, x, c)

    # --- family 4: the weight posterior on a FIXED nk ----------------------- #
    nk = np.asarray(BGM_STICK_NK, dtype=np.float64)
    out["stick_nk"] = c(nk)
    out["stick_prior"] = c([BGM_STICK_PRIOR])
    for ptype in BGM_PRIOR_TYPES:
        tag = BGM_PRIOR_TAGS[ptype]
        est = BayesianGaussianMixture(
            n_components=BGM_STICK_K,
            covariance_type="full",
            weight_concentration_prior_type=ptype,
            weight_concentration_prior=BGM_STICK_PRIOR,
        )
        # sklearn's `_check_weights_parameters` normally runs inside `fit`; the
        # prior is set directly here because there is no design to fit — the
        # point of this family is to evaluate the weight update ALONE.
        est.weight_concentration_prior_ = BGM_STICK_PRIOR
        est._estimate_weights(nk)
        wc = est.weight_concentration_
        if ptype == "dirichlet_process":
            out[f"stick_wca_{tag}"] = c(wc[0])
            out[f"stick_wcb_{tag}"] = c(wc[1])
        else:
            out[f"stick_wca_{tag}"] = c(wc)
        out[f"stick_logw_{tag}"] = c(est._estimate_log_weights())
        # `weights_` is derived inside `_set_parameters`, which needs the full
        # posterior tuple; the Gaussian half is dummy 1-D data because only the
        # weight half is read here.
        one = np.ones(BGM_STICK_K)
        est._set_parameters(
            (
                wc,
                one,
                np.zeros((BGM_STICK_K, 1)),
                one,
                np.ones((BGM_STICK_K, 1, 1)),
                np.ones((BGM_STICK_K, 1, 1)),
            )
        )
        out[f"stick_weights_{tag}"] = c(est.weights_)

    # --- family 1's stability flags ----------------------------------------- #
    # A `{cov}_{init}_{ptype}` case can only be compared VALUE-for-value if the
    # variational objective has one attracting basin for that combination —
    # otherwise mlrs's initialization RNG (D-09) legitimately lands in a
    # different one and no tolerance can bridge it. So instead of asserting
    # basin-uniqueness (which is FALSE for this estimator, unlike for
    # `GaussianMixture`), the generator MEASURES it and records the verdict.
    #
    # The measured exception, stable across every design shape tried
    # (separation 3-12, sigma 0.6-1.2, simplex and asymmetric centers, k=2/3/4):
    # `covariance_type='tied'` with `weight_concentration_prior_type=
    # 'dirichlet_process'` and a SPARSE initialization ('k-means++' /
    # 'random_from_data', both of which put all their mass on `k` rows). With
    # `nk ~= 1` per component the shared tied covariance is still essentially
    # the prior — i.e. the whole design's covariance — so the first E-step is
    # near-uniform, and the stick-breaking prior's built-in order asymmetry then
    # pushes the mass onto the low indices and prunes a component before the
    # covariance can shrink. That is the Dirichlet PROCESS doing exactly what it
    # exists to do; the symmetric Dirichlet, having no order asymmetry, recovers
    # from the same start. Both engines exhibit it; only WHICH collapsed
    # solution they reach is RNG-dependent.
    for cov in GMM_COV_TYPES:
        for ptype in BGM_PRIOR_TYPES:
            tag = BGM_PRIOR_TAGS[ptype]
            ref = float(out[f"lower_bound_{cov}_kmeans_{tag}"][0])
            agree = 0
            for init in GMM_INIT_PARAMS:
                name = f"{cov}_{init.replace('-', '')}_{tag}"
                got = float(out[f"lower_bound_{name}"][0])
                stable = abs(got - ref) <= 1e-2 + 1e-6 * abs(ref)
                out[f"stable_{name}"] = c([1.0 if stable else 0.0])
                agree += int(stable)
            # The design must still be separable enough that the basin the
            # data-driven routes find is a MAJORITY verdict, not a coin flip —
            # otherwise "stable" would be a statement about `kmeans` alone.
            assert agree >= 2, (
                f"gen_bayesian_mixture: covariance_type='{cov}' "
                f"weight_concentration_prior_type='{ptype}' has no agreeing "
                f"pair of init_params routes (only {agree} of "
                f"{len(GMM_INIT_PARAMS)} reached lower_bound {ref}) — the "
                "design is not separable enough to pin any optimum"
            )

    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(
        _FIXTURE_DIR, f"bayesian_mixture_{dtype_tag}_seed{seed}.npz"
    )
    np.savez(out_path, **out)
    return out_path


# ---------------------------------------------------------------------------
# TSNE full-parameter-surface fixtures (TSNE-PARAMS)
# ---------------------------------------------------------------------------

# Every `metric=` string sklearn 1.9.0's TSNE accepts that can be evaluated on
# a generic float design. Ordered so the aliases sit next to their canonical
# name; the Rust side collapses them, and the fixture proves the collapse is
# value-preserving rather than assuming it.
TSNE_GENERIC_METRICS = (
    "euclidean",
    "l2",
    "sqeuclidean",
    "l1",
    "manhattan",
    "cityblock",
    "chebyshev",
    "minkowski",
    "cosine",
    "correlation",
    "canberra",
    "braycurtis",
    "seuclidean",
    "mahalanobis",
    "hamming",
    "matching",
    "jaccard",
    "dice",
    "rogerstanimoto",
    "russellrao",
    "sokalsneath",
    "yule",
)


def _tsne_metric_design(seed: int):
    """The design the metric fixtures are measured on.

    Three properties are deliberate, and each one exists to stop a whole class
    of metric from degenerating into a constant:

    * **Genuine zeros (~35% of entries).** The six metrics sklearn evaluates on
      a BOOLEAN cast (`dice`/`jaccard`/`rogerstanimoto`/`russellrao`/
      `sokalsneath`/`yule`) reduce to the contingency counts of `x != 0`. On an
      all-nonzero Gaussian design every row casts to all-true and every one of
      those metrics returns exactly 0 for every pair — a fixture that would
      pass against almost any implementation. The zeros are what give them
      information.
    * **No all-zero row.** `sokalsneath` RAISES and `dice` returns NaN for a
      pair of all-zero rows. Those degeneracies are mirrored by the Rust port
      and tested separately; they must not contaminate the value fixture.
    * **Full-rank, well-conditioned covariance.** `mahalanobis` inverts it and
      `seuclidean` divides by its diagonal.
    """
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [6.0, 6.0, -6.0, 3.0, -3.0],
            [-7.0, 4.0, 5.0, -5.0, 2.5],
        ]
    )
    x = np.vstack([centers[b] + 0.7 * rng.standard_normal((16, 5)) for b in range(3)])
    # Punch in zeros for the boolean family, then repair any all-zero row.
    mask = rng.random(x.shape) < 0.35
    x = np.where(mask, 0.0, x)
    for i in range(x.shape[0]):
        if not np.any(x[i]):
            x[i, i % x.shape[1]] = 1.0 + 0.1 * i
    return x


def gen_tsne_metrics(seed: int = SEED, dtype=np.float32) -> str:
    """`TSNE(metric=...)` value fixture (TSNE-PARAMS) — one archive carrying
    sklearn's pairwise distances AND the dense joint-probability matrix for
    every metric string.

    Why both tiers: the distance matrix pins the metric formula itself, and the
    `P` matrix pins that t-SNE consumes it the way sklearn does — including the
    `distances **= 2` that applies to every metric EXCEPT `'euclidean'` (which
    sklearn requests pre-squared) and the f32 rounding inside the perplexity
    search. A port can get the first right and the second wrong.

    Stores ``X`` plus ``D_<metric>`` / ``P_<metric>`` per metric, the
    2-feature ``Xh`` + ``D_haversine`` / ``P_haversine`` pair (haversine is
    only defined on 2 dimensions), the NaN-carrying ``Xnan`` +
    ``D_nan_euclidean`` / ``P_nan_euclidean`` pair, and the ``Xpre`` square
    matrix for ``metric='precomputed'``.
    """
    import warnings

    from scipy.spatial.distance import squareform
    from sklearn.manifold._t_sne import _joint_probabilities
    from sklearn.metrics.pairwise import pairwise_distances

    x = _tsne_metric_design(seed)
    n = x.shape[0]
    perplexity = 10.0

    def c(arr):
        return np.asarray(arr, dtype=dtype)

    def dense_p(dist, metric):
        """sklearn `_fit`'s exact-method distance stage: square unless the
        metric is `'euclidean'` (already squared), then the perplexity search."""
        d = np.array(dist, dtype=np.float64, copy=True)
        if metric != "euclidean":
            d = d**2
        return squareform(_joint_probabilities(d, perplexity, 0))

    out = {"X": c(x), "perplexity": c([perplexity])}

    for m in TSNE_GENERIC_METRICS:
        with warnings.catch_warnings():
            # The bool-cast metrics emit DataConversionWarning by design.
            warnings.simplefilter("ignore")
            if m == "euclidean":
                dist = pairwise_distances(x, metric=m, squared=True)
            else:
                dist = pairwise_distances(x, metric=m)
        out[f"D_{m}"] = np.asarray(dist, dtype=np.float64)
        out[f"P_{m}"] = dense_p(dist, m)

    # haversine: radian (latitude, longitude), 2 features only.
    rng = np.random.default_rng(seed + 11)
    xh = rng.uniform(-1.0, 1.0, size=(n, 2))
    dist = pairwise_distances(xh, metric="haversine")
    out["Xh"] = c(xh)
    out["D_haversine"] = np.asarray(dist, dtype=np.float64)
    out["P_haversine"] = dense_p(dist, "haversine")

    # nan_euclidean: a design with genuine missing entries, no all-NaN row.
    xn = np.array(x, copy=True)
    nan_mask = rng.random(xn.shape) < 0.12
    xn = np.where(nan_mask, np.nan, xn)
    for i in range(n):
        if np.all(np.isnan(xn[i])):
            xn[i, 0] = float(i)
    dist = pairwise_distances(xn, metric="nan_euclidean")
    out["Xnan"] = c(xn)
    out["D_nan_euclidean"] = np.asarray(dist, dtype=np.float64)
    out["P_nan_euclidean"] = dense_p(dist, "nan_euclidean")

    # precomputed: X IS the distance matrix. Built from a DIFFERENT metric than
    # euclidean so a port that quietly recomputes euclidean distances instead of
    # reading the matrix fails the gate.
    xpre = pairwise_distances(x, metric="cityblock")
    dist = pairwise_distances(xpre, metric="precomputed")
    out["Xpre"] = c(xpre)
    out["D_precomputed"] = np.asarray(dist, dtype=np.float64)
    out["P_precomputed"] = dense_p(dist, "precomputed")

    dtype_tag = "f32" if dtype == np.float32 else "f64"
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"tsne_metrics_{dtype_tag}_seed{seed}.npz")
    np.savez(out_path, **out)
    return out_path


def gen_tsne_params(seed: int = SEED, dtype=np.float32) -> str:
    """`TSNE(method=..., init=..., learning_rate=...)` fixture (TSNE-PARAMS).

    The `method` and `init` gates share ONE injected starting embedding
    (``init_array``). That is the whole point of the design: t-SNE's descent is
    1000 chaotic iterations, so two runs from different inits are incomparable
    at the value level, and a band gate on a stochastic init proves very little.
    Feeding sklearn and mlrs the SAME init removes the only source of
    divergence that is not arithmetic, which is what makes a tight
    neighbourhood-preservation band meaningful.

    Stores, for each ``method`` in {``barnes_hut``, ``exact``} and each ``init``
    in {``pca``, ``random``, ``array``}: ``emb_<tag>``, ``kl_<tag>``,
    ``trust_<tag>``. Also stores ``lr_auto`` — the value sklearn's
    ``learning_rate='auto'`` resolves to, ``max(n / early_exaggeration / 4,
    50)`` — so the Rust side can gate the ``'auto'`` arm against an EXPLICIT
    learning rate for exact equality rather than by band.
    """
    from sklearn.manifold import TSNE as SkTSNE
    from sklearn.manifold import trustworthiness

    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [6.0, 6.0, -6.0, 3.0, -3.0],
            [-7.0, 4.0, 5.0, -5.0, 2.5],
        ]
    )
    x = np.vstack([centers[b] + 0.7 * rng.standard_normal((16, 5)) for b in range(3)])
    n = x.shape[0]
    perplexity = 10.0
    early_exaggeration = 12.0
    init_array = 1e-4 * rng.standard_normal((n, 2))

    def c(arr):
        return np.asarray(arr, dtype=dtype)

    out = {
        "X": c(x),
        "perplexity": c([perplexity]),
        "init_array": np.asarray(init_array, dtype=np.float64),
        "lr_auto": c([max(n / early_exaggeration / 4.0, 50.0)]),
    }

    def record(tag, model):
        emb = model.fit_transform(x)
        out[f"emb_{tag}"] = np.asarray(emb, dtype=np.float64)
        out[f"kl_{tag}"] = c([model.kl_divergence_])
        out[f"trust_{tag}"] = c([trustworthiness(x, emb, n_neighbors=5)])
        out[f"niter_{tag}"] = c([model.n_iter_])

    # `method`: both objectives, from the SAME injected init.
    for method in ("barnes_hut", "exact"):
        record(
            method,
            SkTSNE(
                n_components=2,
                perplexity=perplexity,
                method=method,
                init=np.array(init_array, copy=True),
                learning_rate="auto",
                max_iter=1000,
                random_state=seed,
            ),
        )

    # `init`: the three accepted forms, on the exact objective so the only
    # difference between the runs is where the descent started.
    for tag, init in (
        ("init_pca", "pca"),
        ("init_random", "random"),
        ("init_array", np.array(init_array, copy=True)),
    ):
        record(
            tag,
            SkTSNE(
                n_components=2,
                perplexity=perplexity,
                method="exact",
                init=init,
                learning_rate="auto",
                max_iter=1000,
                random_state=seed,
            ),
        )

    dtype_tag = "f32" if dtype == np.float32 else "f64"
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    out_path = os.path.join(_FIXTURE_DIR, f"tsne_params_{dtype_tag}_seed{seed}.npz")
    np.savez(out_path, **out)
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
    # LinearRegression large-N (LINEAR-01 `fit_gram_eig` path, n_samples > 256).
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_linear_regression_large(dtype=dtype)}")
    # Ridge (LINEAR-02): cholesky solver, alpha sweep incl. the strict 1.0 case.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_ridge(dtype=dtype)}")
    # Ridge FULL parameter surface (LINEAR-02): every sklearn `solver`, with and
    # without `fit_intercept` / `positive` / `sample_weight`.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_ridge_params(dtype=dtype)}")
    # RidgeClassifier FULL parameter surface (LINEAR-07): binary + multiclass,
    # every solver, class_weight, positive, sample_weight.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_ridge_classifier(dtype=dtype, multiclass=False)}")
        print(f"wrote {gen_ridge_classifier(dtype=dtype, multiclass=True)}")
    # BayesianRidge FULL parameter surface (LINEAR-06): the evidence iteration,
    # both n_samples <=> n_features branches, hyperpriors, inits, scores, weights.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_bayesian_ridge(dtype=dtype)}")
    # HuberRegressor FULL parameter surface (HUBER-01): epsilon/alpha/tol/
    # fit_intercept/sample_weight converged value cases, the max_iter & loose-tol
    # control-flow cases, and the warm_start pair.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_huber(dtype=dtype)}")
    # GaussianMixture FULL parameter surface (MIX-01): the covariance_type x
    # init_params cross fitted to machine precision, plus fully-injected-init
    # cases that remove the RNG entirely, plus a reg_covar sweep.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_gaussian_mixture(dtype=dtype)}")
    # BayesianGaussianMixture FULL parameter surface (MIX-02): the
    # covariance_type x init_params x weight_concentration_prior_type cross, the
    # RNG-free k=1 exact families, the five-prior sweep, and the stick-breaking
    # pin on a fixed nk.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_bayesian_mixture(dtype=dtype)}")
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
    # KNeighborsRegressor full parameter surface (KNN-REG-PARAMS): every
    # weights x metric combination the device path serves, plus a multi-output
    # target and a coincident query (the weights='distance' 1/0 branch).
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_knn_regressor_params(dtype=dtype)}")
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
    # HDBSCAN string-valued-parameter surface (HDBS-PARAMS): one sklearn label
    # vector per accepted string of `metric` / `algorithm` /
    # `cluster_selection_method`, plus the `store_centers` blocks. sklearn-only —
    # no `hdbscan` 0.8.44 needed, so this one regenerates in a plain env.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_hdbscan_params(dtype=dtype)}")
    # ARIMA (TSA-01): AR(2)/MA(1) zero-mean process, statsmodels SARIMAX
    # oracle (fixed-param loglik + MLE fit + forecast).
    print(f"wrote {gen_arima()}")
    # TreeSHAP (SHAP-01): sklearn RF classifier+regressor node arrays +
    # shap.TreeExplainer values (the ForestInference import path oracle).
    for path in gen_tree_shap():
        print(f"wrote {path}")
    # AgglomerativeClustering (AGGLO-01): single-linkage, EXACT labels+children
    # per metric × {f32, f64} at n_clusters ∈ {2, 3, 5}. Regen in the same /tmp
    # venv as the hdbscan fixtures.
    for metric in ("euclidean", "manhattan", "cosine"):
        for dtype in (np.float32, np.float64):
            print(f"wrote {gen_agglomerative(dtype=dtype, metric=metric)}")
    # TSNE (TSNE-01): deterministic P-matrix gate + KL/trustworthiness band.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_tsne(dtype=dtype)}")
    # TSNE full parameter surface (TSNE-PARAMS): one archive per dtype carrying
    # sklearn's distances AND joint probabilities for every `metric` string, and
    # a second carrying the `method` / `init` / `learning_rate='auto'` gates.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_tsne_metrics(dtype=dtype)}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_tsne_params(dtype=dtype)}")
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
    # SpectralEmbedding LANCZOS arm (SPECTRAL-PERF-CPU): n_samples above the
    # host pipeline's DENSE_N=512 dense/iterative threshold, one sparse (kNN)
    # and one dense (rbf) affinity. f64 only — the Lanczos arm is a host f64
    # pipeline and the fixtures gate its sklearn parity, not a dtype band.
    print(f"wrote {gen_spectral_embedding_large(dtype=np.float64)}")
    print(f"wrote {gen_spectral_embedding_large_rbf(dtype=np.float64)}")
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
    # LinearSVC one-vs-rest multiclass: (n_classes, d) coef_ + decision_function.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_linear_svc_multiclass(dtype=dtype)}")
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

    # ---- Phase-17 DecisionTree oracle fixtures (TREE-01, D-07/D-09) ----
    # Injected fixed-index (bootstrap rows + feature subset) sklearn reference
    # trees the Plan-03 Tier-1 witness value-asserts against. gini-classifier +
    # squared-error-regressor, f32 (rocm gate) + f64 (cpu gate). The adversarial
    # variant carries a forced-pure-leaf + exact gain TIE (the 002-B silent
    # histogram/argmax-miscompile backstop, T-17-01).
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_decision_tree_clf(dtype=dtype)}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_decision_tree_reg(dtype=dtype)}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_decision_tree_clf(dtype=dtype, structure='adversarial')}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_decision_tree_reg(dtype=dtype, structure='adversarial')}")

    # ---- Phase-19 RandomForest forest-level fixtures (ENSEMBLE-01) ----
    # Deterministic (bootstrap=False, exact train parity, asserted pure) +
    # statistical (sklearn-defaults held-out accuracy/R² margin) tiers.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_random_forest_classifier(dtype=dtype)}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_random_forest_regressor(dtype=dtype)}")

    # ---- HistGradientBoosting fixtures (GBT-01) ----
    # Deterministic (max_leaf_nodes=None + depth bound => level-wise
    # equivalence, exact train parity) + statistical (sklearn defaults,
    # held-out margin) tiers; the classifier fixture carries the 3-class
    # softmax AND binarized sigmoid paths.
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_hgb_regressor(dtype=dtype)}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_hgb_classifier(dtype=dtype)}")

    # ---- Metrics surface fixtures (METR-01/02/03) ----
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_metrics_classification_binary(dtype=dtype)}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_metrics_classification_multiclass(dtype=dtype)}")
    print(f"wrote {gen_metrics_classification_degenerate()}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_metrics_regression(dtype=dtype)}")

    # ---- Preprocessing scaler fixtures (PREP-01, Phase 24) ----
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_standard_scaler(dtype=dtype)}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_min_max_scaler(dtype=dtype)}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_max_abs_scaler(dtype=dtype)}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_robust_scaler(dtype=dtype)}")
    for dtype in (np.float32, np.float64):
        for norm in ("l1", "l2", "max"):
            print(f"wrote {gen_normalizer(dtype=dtype, norm=norm)}")
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_binarizer(dtype=dtype)}")

    # ---- feature_selection fixtures (FSEL-01) ----
    # Delegated to their own module: the 18-name feature-selection surface shares
    # one design matrix and one recording helper, so its parameter cross is
    # readable there and would not be here. Still regenerated by this entry
    # point, so `python3 scripts/gen_oracle.py` remains the single command.
    from gen_feature_selection_oracle import main as gen_feature_selection_all

    gen_feature_selection_all()


if __name__ == "__main__":
    main()
