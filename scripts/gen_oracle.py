#!/usr/bin/env python3
"""Seeded NumPy oracle-fixture generator for mlrs (build-time only, D-03).

This script regenerates the committed ``.npz`` oracle blobs that the Rust test
suite reads with **no Python in the loop** (D-03). It is the *canonical*
regeneration tool: ``numpy.random.default_rng(seed)`` is the authoritative
seeded RNG (avoid Rust-side RNG, RESEARCH Pitfall 7), and the committed blobs
are checked in so CI never runs this script.

Phase 1 emits the saxpy smoke case only. Phase 4+ extends this module to
``import sklearn``, fit estimators, and ``np.savez`` their fitted attributes
(``coef_`` / ``intercept_`` / ...) under the same ``case_dtype_seed`` naming
convention (D-01/D-02).

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
    # Symmetric eig (PRIM-05, D-04/D-06): f32+f64 so the f64 cpu path is pinned.
    # np.linalg.eigh is the numpy reference, REVERSED to descending (D-04).
    print(f"wrote {gen_eigh(dtype=np.float32)}")
    print(f"wrote {gen_eigh(dtype=np.float64)}")


if __name__ == "__main__":
    main()
