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


if __name__ == "__main__":
    main()
