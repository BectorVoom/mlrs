//! `sym_eig` — a HOST symmetric eigendecomposition (Householder tridiagonal
//! reduction + implicit-shift QL with accumulated eigenvectors).
//!
//! ## Why this exists next to the Jacobi sweep already in `ridge_solvers`
//! [`ridge_solvers::gram_eig_ridge`](crate::linear::ridge_solvers::gram_eig_ridge)
//! carries a cyclic-Jacobi decomposition, and that is the right choice THERE:
//! it is a COLD path, reached only after a Cholesky factorization has already
//! failed, so the simplest correct rotation sweep wins.
//!
//! `BayesianRidge` is the opposite case. Its eigendecomposition is on the
//! critical path of EVERY fit — the whole reason the estimator can iterate in
//! `O(d)` instead of sklearn's `O(n·d)` is that the spectrum is computed once up
//! front (see `bayesian_ridge.rs`) — so the decomposition itself becomes the
//! second-largest cost of the fit after the Gram formation, and the constant
//! factor matters:
//!
//! | | work | `d = 256` |
//! |---|---|---|
//! | cyclic Jacobi | `~3·d³` per sweep × ~8–10 sweeps | ~400 MFLOP |
//! | this (tred2 + tql2) | `(4/3)·d³` reduce + `~6·d³` accumulate | ~150 MFLOP |
//!
//! Jacobi also has no fixed iteration bound — it sweeps until the off-diagonal
//! mass falls below a threshold — whereas the QL iteration converges cubically
//! and terminates per-eigenvalue, which is what makes the fit's cost
//! predictable.
//!
//! ## What it computes
//! [`sym_eig`] takes a row-major `d × d` SYMMETRIC matrix and returns
//! `(lambda, v)` with the eigenvalues in DESCENDING order and `v[r·d + c]` the
//! `r`-th component of the `c`-th eigenvector — i.e. eigenvectors in COLUMNS,
//! the same convention `prims::eig` uses so the two are drop-in comparable.
//! `v` is orthonormal, so `A = V·diag(λ)·Vᵀ` and `Vᵀ = V⁻¹`.
//!
//! Descending order is not cosmetic: `bayesian_ridge.rs` truncates to the
//! leading `k = min(n_samples, n_features)` directions to reproduce the shape of
//! sklearn's thin `linalg.svd(X, full_matrices=False)`, and "leading" means
//! largest-`λ` first.
//!
//! ## Provenance
//! The reduction and the QL iteration are the standard EISPACK `tred2`/`tql2`
//! pair (via the public-domain JAMA transcription), which is the same algorithm
//! LAPACK's `dsyev` runs and numpy's `eigh` reaches. Transcribing a
//! numerically-vetted recurrence is deliberate: a hand-rolled reduction that is
//! merely *algebraically* right loses orthogonality on clustered spectra, and
//! this codebase's oracle gate is a strict `1e-5`.
//!
//! Only the LOWER triangle of the input is read, so a caller holding a Gram
//! whose upper triangle was mirrored (or not) gets the same answer.
//!
//! Tests live in `crates/mlrs-algos/tests/sym_eig_test.rs` (AGENTS.md §2).

/// `2⁻⁵²` — the `f64` unit roundoff the QL convergence test scales by, verbatim
/// from the reference implementation.
const EPS: f64 = 2.220446049250313e-16;

/// QL iterations allowed per eigenvalue before the sweep gives up.
///
/// The shifted QL iteration converges cubically for symmetric matrices, so a
/// well-formed problem needs a handful; the cap only exists so a pathological
/// (e.g. NaN-poisoned) input cannot spin forever. Reaching it leaves the
/// remaining eigenvalues at their last iterate rather than panicking — a
/// degraded answer is recoverable by the caller, an abort in a `fit` is not.
const MAX_QL_ITER: usize = 50;

/// Symmetric eigendecomposition of a row-major `d × d` matrix.
///
/// Returns `(lambda, v)`: `lambda` holds the `d` eigenvalues in DESCENDING
/// order, and `v[r·d + c]` is component `r` of the eigenvector for `lambda[c]`
/// (eigenvectors in COLUMNS — the `prims::eig` convention).
///
/// Only the lower triangle of `a` is read. `d == 0` returns two empty vectors.
///
/// Panics if `a.len() != d * d`; callers validate geometry first.
pub fn sym_eig(a: &[f64], d: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), d * d, "sym_eig: matrix must be d*d");
    if d == 0 {
        return (Vec::new(), Vec::new());
    }
    if d == 1 {
        return (vec![a[0]], vec![1.0]);
    }

    // `w` starts as a symmetric copy built from the LOWER triangle, so an input
    // whose upper triangle is stale (or absent) decomposes identically. Because
    // the matrix is symmetric, that copy is simultaneously `V` and `Vᵀ` — which
    // is why the transposed working layout below needs no extra setup.
    let mut w = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..=i {
            let x = a[i * d + j];
            w[i * d + j] = x;
            w[j * d + i] = x;
        }
    }
    let mut diag = vec![0.0f64; d];
    let mut sub = vec![0.0f64; d];

    tred2(&mut w, &mut diag, &mut sub, d);
    tql2(&mut w, &mut diag, &mut sub, d);
    sort_descending(&mut diag, &mut w, d);

    // `w` holds eigenvectors in ROWS; the published contract is COLUMNS. One
    // `O(d²)` transpose against the `O(d³)` decomposition that produced it.
    let mut v = vec![0.0f64; d * d];
    for c in 0..d {
        let row = &w[c * d..(c + 1) * d];
        for (r, &val) in row.iter().enumerate() {
            v[r * d + c] = val;
        }
    }
    (diag, v)
}

/// Householder reduction of the symmetric `w` to tridiagonal form (EISPACK
/// `tred2`), operating on the TRANSPOSED layout — `w[c·d + r]` is `V[r][c]`.
///
/// On return `diag`/`sub` hold the tridiagonal's diagonal and subdiagonal, and
/// `w` holds the accumulated orthogonal transformation (in rows).
///
/// ## Why the layout is transposed (BAYES-GPU)
/// This recurrence is written against a column-major FORTRAN original: every
/// inner loop of both the reduction and the back-accumulation walks `k` over
/// `V[k][j]` for a FIXED `j`. Held row-major, that is a stride of `d` elements
/// — at `d = 256, f64` a 2 KiB stride, so each iteration touches a fresh
/// 64-byte cache line and uses 8 of its bytes. The matrix is 512 KiB there, far
/// past L1 and past a shared L2 slice, so the loop runs at line-fetch rate and
/// the vectorizer cannot form a single packed load.
///
/// Storing the TRANSPOSE makes every one of those loops contiguous — the two
/// dot-product-and-axpy pairs in the back-accumulation become plain slice
/// operations over `w`'s rows — at the cost of one `O(d²)` transpose in
/// [`sym_eig`]. The arithmetic, the operand order and hence the rounding are
/// UNCHANGED: this is purely a re-indexing, which is what
/// `sym_eig_test.rs`'s reconstruction and orthogonality gates hold it to.
fn tred2(w: &mut [f64], diag: &mut [f64], sub: &mut [f64], d: usize) {
    for (j, dj) in diag.iter_mut().enumerate().take(d) {
        *dj = w[j * d + (d - 1)];
    }

    // --- Reduce row by row from the bottom. ---
    for i in (1..d).rev() {
        // Scale the row to keep the Householder norm out of overflow/underflow.
        let mut scale = 0.0f64;
        for k in 0..i {
            scale += diag[k].abs();
        }
        let mut h = 0.0f64;

        if scale == 0.0 {
            // The row is already zero off the subdiagonal — nothing to reflect.
            sub[i] = diag[i - 1];
            for j in 0..i {
                diag[j] = w[j * d + (i - 1)];
                w[j * d + i] = 0.0;
                w[i * d + j] = 0.0;
            }
        } else {
            for k in 0..i {
                diag[k] /= scale;
                h += diag[k] * diag[k];
            }
            let mut f = diag[i - 1];
            let mut g = h.sqrt();
            if f > 0.0 {
                g = -g;
            }
            sub[i] = scale * g;
            h -= f * g;
            diag[i - 1] = f - g;
            for s in sub.iter_mut().take(i) {
                *s = 0.0;
            }

            // Apply the similarity transformation to the remaining columns.
            for j in 0..i {
                f = diag[j];
                w[i * d + j] = f;
                g = sub[j] + w[j * d + j] * f;
                for k in (j + 1)..i {
                    let wkj = w[j * d + k];
                    g += wkj * diag[k];
                    sub[k] += wkj * f;
                }
                sub[j] = g;
            }
            f = 0.0;
            for j in 0..i {
                sub[j] /= h;
                f += sub[j] * diag[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                sub[j] -= hh * diag[j];
            }
            for j in 0..i {
                let fj = diag[j];
                let gj = sub[j];
                for k in j..i {
                    w[j * d + k] -= fj * sub[k] + gj * diag[k];
                }
                diag[j] = w[j * d + (i - 1)];
                w[j * d + i] = 0.0;
            }
        }
        diag[i] = h;
    }

    // --- Accumulate the transformations into `w`. ---
    //
    // Both inner loops now run along `w`'s rows: `g` is a dot product of row
    // `i+1` with row `j`, and the update is an axpy of row `i+1` into row `j`.
    for i in 0..(d - 1) {
        w[i * d + (d - 1)] = w[i * d + i];
        w[i * d + i] = 1.0;
        let h = diag[i + 1];
        if h != 0.0 {
            for (k, dk) in diag.iter_mut().enumerate().take(i + 1) {
                *dk = w[(i + 1) * d + k] / h;
            }
            for j in 0..=i {
                let mut g = 0.0f64;
                for k in 0..=i {
                    g += w[(i + 1) * d + k] * w[j * d + k];
                }
                for k in 0..=i {
                    w[j * d + k] -= g * diag[k];
                }
            }
        }
        for k in 0..=i {
            w[(i + 1) * d + k] = 0.0;
        }
    }
    for (j, dj) in diag.iter_mut().enumerate().take(d) {
        *dj = w[j * d + (d - 1)];
        w[j * d + (d - 1)] = 0.0;
    }
    w[(d - 1) * d + (d - 1)] = 1.0;
    sub[0] = 0.0;
}

/// Implicit-shift QL iteration on the tridiagonal `(diag, sub)`, accumulating
/// the rotations into `w` (EISPACK `tql2`), in the transposed layout
/// [`tred2`] leaves behind.
///
/// On return `diag` holds the eigenvalues (unordered) and the ROWS of `w` the
/// matching eigenvectors.
///
/// The rotation accumulation is this routine's `O(d³)` term and the whole
/// reason for the layout: a plane rotation mixes eigenvector columns `i` and
/// `i + 1` across ALL rows, which is two strided walks of the row-major form
/// and two contiguous ROW updates here. The pair is taken through
/// `split_at_mut` so both rows are borrowed mutably at once without indexing
/// through bounds checks per element.
fn tql2(w: &mut [f64], diag: &mut [f64], sub: &mut [f64], d: usize) {
    for i in 1..d {
        sub[i - 1] = sub[i];
    }
    sub[d - 1] = 0.0;

    let mut f = 0.0f64;
    let mut tst1 = 0.0f64;

    for l in 0..d {
        // Track the largest row magnitude seen so far — the QL convergence test
        // is RELATIVE to it, which is what keeps a badly scaled spectrum from
        // either spinning or terminating early.
        tst1 = tst1.max(diag[l].abs() + sub[l].abs());

        // Find the first negligible subdiagonal at or after `l`.
        let mut m = l;
        while m < d {
            if sub[m].abs() <= EPS * tst1 {
                break;
            }
            m += 1;
        }

        if m > l {
            let mut iter = 0usize;
            loop {
                iter += 1;

                // Implicit shift from the trailing 2×2.
                let mut g = diag[l];
                let mut p = (diag[l + 1] - g) / (2.0 * sub[l]);
                let mut r = p.hypot(1.0);
                if p < 0.0 {
                    r = -r;
                }
                diag[l] = sub[l] / (p + r);
                diag[l + 1] = sub[l] * (p + r);
                let dl1 = diag[l + 1];
                let mut h = g - diag[l];
                for dv in diag.iter_mut().take(d).skip(l + 2) {
                    *dv -= h;
                }
                f += h;

                // The QL sweep itself, bottom-up from `m`.
                p = diag[m];
                let mut c = 1.0f64;
                let mut c2 = c;
                let mut c3 = c;
                let el1 = sub[l + 1];
                let mut s = 0.0f64;
                let mut s2 = 0.0f64;
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * sub[i];
                    h = c * p;
                    r = p.hypot(sub[i]);
                    sub[i + 1] = s * r;
                    s = sub[i] / r;
                    c = p / r;
                    p = c * diag[i] - s * g;
                    diag[i + 1] = h + s * (c * g + s * diag[i]);

                    // Accumulate the plane rotation into the eigenvectors —
                    // rows `i` and `i + 1` of `w`, both contiguous.
                    let (lo, hi) = w.split_at_mut((i + 1) * d);
                    let row_i = &mut lo[i * d..i * d + d];
                    let row_j = &mut hi[..d];
                    for (a, b) in row_i.iter_mut().zip(row_j.iter_mut()) {
                        let hk = *b;
                        *b = s * *a + c * hk;
                        *a = c * *a - s * hk;
                    }
                }
                p = -s * s2 * c3 * el1 * sub[l] / dl1;
                sub[l] = s * p;
                diag[l] = c * p;

                if sub[l].abs() <= EPS * tst1 || iter >= MAX_QL_ITER {
                    break;
                }
            }
        }
        diag[l] += f;
        sub[l] = 0.0;
    }
}

/// Reorder `(lambda, w)` into DESCENDING eigenvalue order by selection sort,
/// swapping whole eigenvector ROWS of the transposed layout alongside.
///
/// A selection sort is `O(d²)` comparisons against the `O(d³)` decomposition
/// that precedes it, and it moves each eigenvector at most once — as one
/// contiguous `swap_with_slice` rather than the `d` strided element swaps the
/// column layout would need.
fn sort_descending(diag: &mut [f64], w: &mut [f64], d: usize) {
    for i in 0..d.saturating_sub(1) {
        let mut best = i;
        for (j, &val) in diag.iter().enumerate().take(d).skip(i + 1) {
            if val > diag[best] {
                best = j;
            }
        }
        if best != i {
            diag.swap(i, best);
            let (lo, hi) = w.split_at_mut(best * d);
            lo[i * d..i * d + d].swap_with_slice(&mut hi[..d]);
        }
    }
}
