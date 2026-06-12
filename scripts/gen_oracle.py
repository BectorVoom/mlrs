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


if __name__ == "__main__":
    main()
