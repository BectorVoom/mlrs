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


def main() -> None:
    for dtype in (np.float32, np.float64):
        path = gen_saxpy(dtype=dtype)
        print(f"wrote {path}")
    for dtype in (np.float32, np.float64):
        path = gen_gemm(dtype=dtype)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
