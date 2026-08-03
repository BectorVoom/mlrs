//! `sym_eig` (LINEAR-06) — the host symmetric eigendecomposition.
//!
//! Gates the three properties `bayesian_ridge.rs` relies on, on matrices whose
//! answers are known independently of any reference implementation:
//!
//! 1. **Reconstruction** — `A == V·diag(λ)·Vᵀ`.
//! 2. **Orthonormality** — `Vᵀ·V == I`. This is the property a hand-rolled
//!    reduction loses first, and losing it is silent: the reconstruction can
//!    still look plausible while the change of basis is no longer a rotation, at
//!    which point `Σcoef² == Σc²` (which the evidence loop assumes) stops
//!    holding.
//! 3. **Descending order** — `bayesian_ridge.rs` truncates to the LEADING `k`
//!    directions to mirror a thin SVD, so "leading" has to mean largest-λ.
//!
//! The RANK-DEFICIENT case is the one that matters most: a `BayesianRidge` fit
//! on a wide design centers it first, which costs a degree of freedom, so the
//! Gram it decomposes routinely has a null space several dimensions wide. A
//! degenerate (repeated) eigenvalue is exactly where a QL sweep that stops early
//! stops producing orthogonal vectors.
//!
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use mlrs_algos::linear::sym_eig::sym_eig;

/// `‖A − V·diag(λ)·Vᵀ‖_max`.
fn reconstruction_error(a: &[f64], lambda: &[f64], v: &[f64], d: usize) -> f64 {
    let mut worst: f64 = 0.0;
    for i in 0..d {
        for j in 0..d {
            let got: f64 = (0..d)
                .map(|c| v[i * d + c] * lambda[c] * v[j * d + c])
                .sum();
            worst = worst.max((got - a[i * d + j]).abs());
        }
    }
    worst
}

/// `‖VᵀV − I‖_max`.
fn orthonormality_error(v: &[f64], d: usize) -> f64 {
    let mut worst: f64 = 0.0;
    for p in 0..d {
        for q in 0..d {
            let dot: f64 = (0..d).map(|r| v[r * d + p] * v[r * d + q]).sum();
            let want = if p == q { 1.0 } else { 0.0 };
            worst = worst.max((dot - want).abs());
        }
    }
    worst
}

/// Assert all three invariants at once, with `scale` the matrix's magnitude (the
/// reconstruction error is relative to it; orthonormality is absolute).
fn assert_decomposition(a: &[f64], d: usize, scale: f64, what: &str) -> (Vec<f64>, Vec<f64>) {
    let (lambda, v) = sym_eig(a, d);
    assert_eq!(lambda.len(), d, "{what}: eigenvalue count");
    assert_eq!(v.len(), d * d, "{what}: eigenvector matrix size");

    let rec = reconstruction_error(a, &lambda, &v, d);
    assert!(
        rec <= 1e-11 * scale.max(1.0),
        "{what}: reconstruction error {rec:e} exceeds 1e-11 * {scale:e}"
    );
    let orth = orthonormality_error(&v, d);
    assert!(
        orth <= 1e-11,
        "{what}: orthonormality error {orth:e} exceeds 1e-11 (V is not a rotation)"
    );
    for w in lambda.windows(2) {
        assert!(
            w[0] >= w[1],
            "{what}: eigenvalues are not descending ({} then {})",
            w[0],
            w[1]
        );
    }
    (lambda, v)
}

/// A deterministic PRNG so the random cases are reproducible without pulling in
/// an RNG crate (the splitmix64 precedent used across this workspace's tests).
fn splitmix(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Uniform on [-1, 1).
    ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// `XᵀX` for a row-major `n × d` `X`, optionally column-centered first.
fn gram(x: &[f64], n: usize, d: usize, center: bool) -> Vec<f64> {
    let mut xc = x.to_vec();
    if center {
        for j in 0..d {
            let mean: f64 = (0..n).map(|i| x[i * d + j]).sum::<f64>() / n as f64;
            for i in 0..n {
                xc[i * d + j] -= mean;
            }
        }
    }
    let mut g = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            g[i * d + j] = (0..n).map(|r| xc[r * d + i] * xc[r * d + j]).sum();
        }
    }
    g
}

/// A diagonal matrix decomposes to its own diagonal, in descending order.
#[test]
fn sym_eig_diagonal() {
    let d = 5;
    let want = [7.0, 1.0, 5.0, -2.0, 3.0];
    let mut a = vec![0.0f64; d * d];
    for (i, &w) in want.iter().enumerate() {
        a[i * d + i] = w;
    }
    let (lambda, _) = assert_decomposition(&a, d, 7.0, "diagonal");
    let mut sorted = want;
    sorted.sort_by(|x, y| y.partial_cmp(x).unwrap());
    for (got, exp) in lambda.iter().zip(sorted.iter()) {
        assert!(
            (got - exp).abs() < 1e-12,
            "diagonal: got {got} expected {exp}"
        );
    }
}

/// The `d == 1` and `d == 2` shapes the reduction special-cases.
#[test]
fn sym_eig_tiny() {
    let (lambda, v) = sym_eig(&[3.5], 1);
    assert_eq!(lambda, vec![3.5]);
    assert_eq!(v, vec![1.0]);

    // [[2, 1], [1, 2]] has eigenvalues 3 and 1.
    let (lambda, _) = assert_decomposition(&[2.0, 1.0, 1.0, 2.0], 2, 2.0, "2x2");
    assert!((lambda[0] - 3.0).abs() < 1e-12 && (lambda[1] - 1.0).abs() < 1e-12);
}

/// A dense full-rank Gram — the ordinary `n_samples > n_features` regime.
#[test]
fn sym_eig_full_rank_gram() {
    let (n, d) = (60, 8);
    let mut st = 42u64;
    let x: Vec<f64> = (0..n * d).map(|_| splitmix(&mut st)).collect();
    let g = gram(&x, n, d, true);
    let (lambda, _) = assert_decomposition(&g, d, 60.0, "full-rank gram");
    // A Gram is positive semi-definite: no eigenvalue may be meaningfully
    // negative, or `bayesian_ridge`'s `1/(λ + r)` would flip sign.
    assert!(
        lambda.iter().all(|&l| l > -1e-10),
        "full-rank gram: negative eigenvalue in a PSD matrix: {lambda:?}"
    );
}

/// The `bayesian_ridge` wide case: a CENTERED `6 × 10` design, whose Gram has
/// rank `n − 1 = 5` and therefore a FIVE-fold degenerate zero eigenvalue.
///
/// This is the shape that broke the first implementation. A repeated eigenvalue
/// is where an eigenvector sweep loses orthogonality, and here the degenerate
/// block is half the matrix.
#[test]
fn sym_eig_rank_deficient_centered_gram() {
    let (n, d) = (6, 10);
    let mut st = 7u64;
    let x: Vec<f64> = (0..n * d).map(|_| splitmix(&mut st)).collect();
    let g = gram(&x, n, d, true);
    let (lambda, _) = assert_decomposition(&g, d, 6.0, "rank-deficient gram");

    // Exactly `n - 1 = 5` directions carry signal; the rest are numerically zero.
    let significant = lambda.iter().filter(|&&l| l > 1e-8).count();
    assert_eq!(
        significant, 5,
        "rank-deficient gram: expected rank 5 after centering, got {significant} \
         (eigenvalues {lambda:?})"
    );
    assert!(
        lambda[5..].iter().all(|&l| l.abs() < 1e-10),
        "rank-deficient gram: the null block is not numerically zero: {:?}",
        &lambda[5..]
    );
}

/// A matrix with CLUSTERED but distinct eigenvalues — the hardest case for the
/// shifted QL iteration's deflation test, and the one where an eigenvector can
/// silently come back non-orthogonal to its neighbour.
#[test]
fn sym_eig_clustered_spectrum() {
    let d = 12;
    // Build V·diag(λ)·Vᵀ from a known rotation so the answer is exact by
    // construction: a product of Givens rotations is orthogonal to machine
    // precision, which a random matrix would not be.
    let mut v = vec![0.0f64; d * d];
    for i in 0..d {
        v[i * d + i] = 1.0;
    }
    let mut st = 99u64;
    for p in 0..d {
        for q in (p + 1)..d {
            let theta = splitmix(&mut st);
            let (c, s) = (theta.cos(), theta.sin());
            for r in 0..d {
                let (a, b) = (v[r * d + p], v[r * d + q]);
                v[r * d + p] = c * a - s * b;
                v[r * d + q] = s * a + c * b;
            }
        }
    }
    // Eigenvalues 1.0, 1.0 + 1e-9, 1.0 + 2e-9, … plus one far-separated value.
    let lambda: Vec<f64> = (0..d)
        .map(|i| if i == 0 { 100.0 } else { 1.0 + i as f64 * 1e-9 })
        .collect();
    let mut a = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            a[i * d + j] = (0..d)
                .map(|c| v[i * d + c] * lambda[c] * v[j * d + c])
                .sum();
        }
    }
    assert_decomposition(&a, d, 100.0, "clustered spectrum");
}

/// A LARGE Gram (`d = 193`), which the small cases above cannot stand in for.
///
/// [`sym_eig`] works on the TRANSPOSE of the eigenvector matrix so that every
/// inner loop of `tred2`/`tql2` runs along a contiguous row instead of striding
/// by `d` (the `172 ms → 18 ms` measurement at `d = 256` that motivated it).
/// That rewrite re-indexed the whole EISPACK recurrence and replaced the
/// eigenvector column swap with a `swap_with_slice` of rows, so it is exactly
/// the kind of change a `d = 12` fixture cannot gate: the small cases run few
/// Householder steps, few QL sweeps and almost no reordering, and a
/// transposition slip in a rarely-taken branch survives them.
///
/// `d = 193` is deliberately odd and not a power of two, so the row bands, the
/// `i - 1` / `i + 1` neighbour indices and the trailing `d - 1` fixups are all
/// exercised off any convenient alignment. `n = 260` keeps the Gram full-rank
/// and well conditioned, so the `1e-11` reconstruction and orthonormality
/// bounds are the same ones every other case is held to — the point is that the
/// SIZE changes, not the tolerance.
#[test]
fn sym_eig_large_full_rank_gram() {
    let (n, d) = (260usize, 193usize);
    let mut s = 0x51D_5EED_u64;
    let x: Vec<f64> = (0..n * d).map(|_| splitmix(&mut s)).collect();
    let a = gram(&x, n, d, true);
    let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let (lambda, _) = assert_decomposition(&a, d, scale, "large full-rank gram");

    // A centered `260 × 193` Gaussian Gram has rank `d` (one DOF goes to the
    // centering, and `n − 1 > d`), so every eigenvalue is strictly positive —
    // a sign flip from a mis-transposed rotation would show up here even if the
    // reconstruction stayed inside tolerance.
    assert!(
        lambda.iter().all(|&l| l > 0.0),
        "large full-rank gram: a PSD full-rank Gram must have no non-positive \
         eigenvalue, got min {:e}",
        lambda.iter().cloned().fold(f64::INFINITY, f64::min)
    );
}

/// A large RANK-DEFICIENT Gram (`d = 129`, `n = 40`), the shape
/// `BayesianRidge`'s wide fixture generalizes to.
///
/// Rank deficiency is where a transposition slip does the most damage without
/// tripping a reconstruction bound: the null directions carry no signal, so `A`
/// reconstructs fine while the eigenvectors spanning the null space can still be
/// wrong. `clamp_numerical_rank` then keys off exactly these eigenvalues, so
/// they have to come back at the accumulation's rounding floor rather than
/// merely "small".
#[test]
fn sym_eig_large_rank_deficient_gram() {
    let (n, d) = (40usize, 129usize);
    let mut s = 0xDEFEC7_u64;
    let x: Vec<f64> = (0..n * d).map(|_| splitmix(&mut s)).collect();
    let a = gram(&x, n, d, true);
    let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let (lambda, _) = assert_decomposition(&a, d, scale, "large rank-deficient gram");

    // Centering costs one DOF, so the rank is `n − 1 = 39`; the remaining
    // `d − 39` eigenvalues are structurally zero and must land at the
    // accumulation floor, not merely near it.
    let rank = n - 1;
    let floor = 1e-9 * lambda[0];
    assert!(
        lambda[rank - 1] > floor,
        "large rank-deficient gram: eigenvalue {} of the {rank}-dim range \
         collapsed to {:e}",
        rank - 1,
        lambda[rank - 1]
    );
    for (i, &l) in lambda.iter().enumerate().skip(rank) {
        assert!(
            l.abs() <= floor,
            "large rank-deficient gram: null direction {i} came back at {l:e}, \
             above the {floor:e} floor `clamp_numerical_rank` keys off"
        );
    }
}
