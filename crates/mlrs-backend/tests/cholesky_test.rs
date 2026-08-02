//! Plan 04-02 — Cholesky/SPD-solve primitive (D-02) standalone validation.
//!
//! Exercises the NEW single-cube Cholesky factor + triangular-solve primitive
//! (`mlrs_backend::prims::cholesky::cholesky_solve`) on cpu (f32 + f64) and rocm
//! (f32; f64 skip-with-log per the CubeCL-HIP F64 gap, D-07). Three checks
//! validate the primitive STANDALONE before Ridge (04-05) consumes it, mirroring
//! the Phase-2/3 primitive-first discipline:
//!
//!   - **`‖A·x − b‖` solve invariant** — solve `A·x = b` on the device for the
//!     committed `scipy.linalg.solve(A, b, assume_a="pos")` fixture and assert the
//!     RESIDUAL `‖A·x − b‖` is within 1e-5 (the scale-invariant form of the 1e-5
//!     contract). Also compares `x` directly against the stored scipy reference.
//!   - **`‖L·Lᵀ − A‖` factor invariant** — read back the KERNEL-EMITTED lower
//!     factor `L` (via `cholesky_solve_with_factor`, NOT re-derived on the host),
//!     reconstruct `L·Lᵀ`, and assert it matches the fixture `A` within tolerance.
//!   - **Non-SPD guard** — feed a synthetically indefinite matrix and assert the
//!     host returns `PrimError::NotPositiveDefinite` (negative-pivot flag), never
//!     a NaN-poisoned factor (RESEARCH Pitfall 4).
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64;
//! rocm skips-with-log — EXPECTED, not a defect, D-07). Per AGENTS.md §2 tests
//! live here, never as an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::abflag;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::cholesky::{
    cholesky_solve, cholesky_solve_reg, cholesky_solve_with_factor,
};
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, PrimError};

/// Cholesky fixture geometry (gen_oracle.py `CHOL_N` × `CHOL_RHS`): A is n×n,
/// b/x are n×rhs, L is n×n.
const CHOL_N: usize = 6;
const CHOL_RHS: usize = 2;

/// Residual / reconstruction tolerance for the well-conditioned SPD fixture. The
/// fixture is `A = MᵀM + nI` (benign condition number), so the f32 device solve
/// reaches ~1e-6; 1e-5 is the project contract.
const SOLVE_TOL: f64 = 1e-5;
/// f64 cpu path is far tighter; keep the same 1e-5 contract bound.
const RECON_TOL: f64 = 1e-5;

/// Resolve a workspace-root-relative fixture path (matches `svd_test.rs`).
fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Assert the named array exists with exactly `len` elements (flat).
fn assert_len(case: &OracleCase, name: &str, len: usize) {
    let got = case.expect_f64(name).len();
    assert_eq!(
        got, len,
        "fixture array '{name}' should have {len} elements, got {got}"
    );
}

fn host_to_f64<F: bytemuck::Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("cholesky tests are f32/f64 only"),
    }
}

fn from_f64<F: bytemuck::Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("cholesky tests are f32/f64 only"),
    }
}

/// Frobenius norm of a flat matrix.
fn fro(a: &[f64]) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Host triple-loop matrix product `C (m×n) = A (m×k) · B (k×n)`, all row-major.
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// `fixture-dtype` host vector from the f64 fixture array.
fn fixture_vec<F: bytemuck::Pod>(case: &OracleCase, name: &str) -> Vec<F> {
    case.expect_f64(name)
        .iter()
        .map(|&x| from_f64::<F>(x))
        .collect()
}

/// Shared solve body: load the fixture, run `cholesky_solve` on the device, and
/// assert both `‖A·x − b‖ ≤ 1e-5` and `x` vs the stored scipy reference.
fn check_solve<F>(fixture_name: &str)
where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load cholesky fixture");
    let a: Vec<F> = fixture_vec::<F>(&case, "A");
    let b: Vec<F> = fixture_vec::<F>(&case, "b");
    let x_ref: Vec<f64> = case.expect_f64("x").to_vec();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let a_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &a);
    let b_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &b);

    let x_dev = cholesky_solve::<F>(&mut pool, &a_dev, &b_dev, CHOL_N, CHOL_RHS, None)
        .expect("cholesky solve on a valid SPD system");
    let x: Vec<f64> = x_dev.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect();
    x_dev.release_into(&mut pool);

    // (a) ‖A·x − b‖ residual invariant (scale-invariant 1e-5 contract).
    let a64: Vec<f64> = a.iter().map(|&v| host_to_f64(v)).collect();
    let b64: Vec<f64> = b.iter().map(|&v| host_to_f64(v)).collect();
    let ax = matmul(&a64, &x, CHOL_N, CHOL_N, CHOL_RHS);
    let resid: Vec<f64> = ax.iter().zip(b64.iter()).map(|(&p, &q)| p - q).collect();
    let b_fro = fro(&b64).max(1.0);
    let rel = fro(&resid) / b_fro;
    assert!(
        rel <= SOLVE_TOL,
        "‖A·x−b‖/‖b‖={rel:e} exceeds the {SOLVE_TOL:e} solve contract"
    );

    // (b) x vs the scipy reference (the fixture is well-conditioned so a direct
    //     compare holds to 1e-5).
    for (i, (&g, &e)) in x.iter().zip(x_ref.iter()).enumerate() {
        let abs_err = (g - e).abs();
        assert!(
            abs_err <= SOLVE_TOL + SOLVE_TOL * e.abs(),
            "x[{i}] mismatch vs scipy: got={g:e} expected={e:e} abs_err={abs_err:e}"
        );
    }
}

/// Shared factor body: read back the KERNEL-EMITTED lower factor `L` and assert
/// `‖L·Lᵀ − A‖` matches the fixture `A` within tolerance (L is NOT re-derived on
/// the host — it is the kernel's `l_out` buffer, the unambiguous L source).
fn check_factor<F>(fixture_name: &str)
where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load cholesky fixture");
    let a: Vec<F> = fixture_vec::<F>(&case, "A");
    let b: Vec<F> = fixture_vec::<F>(&case, "b");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let a_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &a);
    let b_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &b);

    let (x_dev, l_dev) =
        cholesky_solve_with_factor::<F>(&mut pool, &a_dev, &b_dev, CHOL_N, CHOL_RHS, None)
            .expect("cholesky factor on a valid SPD system");
    // Read back the kernel-written L (row-major n×n, strictly-upper = 0).
    let l: Vec<f64> = l_dev.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect();
    x_dev.release_into(&mut pool);
    l_dev.release_into(&mut pool);

    // Reconstruct L·Lᵀ (n×n) and compare against the fixture A. Lᵀ is L read with
    // transposed indices; build it explicitly then matmul.
    let mut lt = vec![0.0f64; CHOL_N * CHOL_N];
    for i in 0..CHOL_N {
        for j in 0..CHOL_N {
            lt[i * CHOL_N + j] = l[j * CHOL_N + i];
        }
    }
    let llt = matmul(&l, &lt, CHOL_N, CHOL_N, CHOL_N);
    let a64: Vec<f64> = a.iter().map(|&v| host_to_f64(v)).collect();
    let diff: Vec<f64> = llt.iter().zip(a64.iter()).map(|(&p, &q)| p - q).collect();
    let a_fro = fro(&a64).max(1.0);
    let rel = fro(&diff) / a_fro;
    assert!(
        rel <= RECON_TOL,
        "‖L·Lᵀ−A‖/‖A‖={rel:e} exceeds the {RECON_TOL:e} factor contract \
         (L read back from the kernel l_out buffer, not re-derived)"
    );
}

/// LOAD-NOT-JUST-PRESENT check: load the committed `cholesky_f64_seed42.npz` via
/// `mlrs_core::load_npz` and assert the `A`/`b`/`x`/`L` keys exist with the
/// expected n×n / n×rhs shapes. Proves the committed blob is well-formed.
#[test]
fn fixture_loads() {
    let case = load_npz(fixture("cholesky_f64_seed42.npz")).expect("load cholesky_f64");
    assert_len(&case, "A", CHOL_N * CHOL_N);
    assert_len(&case, "b", CHOL_N * CHOL_RHS);
    assert_len(&case, "x", CHOL_N * CHOL_RHS);
    assert_len(&case, "L", CHOL_N * CHOL_N);
    assert_eq!(case.shape("A"), Some([CHOL_N as u64, CHOL_N as u64].as_slice()));
    assert_eq!(case.shape("b"), Some([CHOL_N as u64, CHOL_RHS as u64].as_slice()));
}

/// `‖A·x − b‖` solve invariant, f32 (runs on cpu AND rocm).
#[test]
fn cholesky_solves_spd_system_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    check_solve::<f32>("cholesky_f32_seed42.npz");
}

/// `‖A·x − b‖` solve invariant, f64 (cpu runs; rocm skips-with-log).
#[test]
fn cholesky_solves_spd_system_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("cholesky f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    check_solve::<f64>("cholesky_f64_seed42.npz");
}

/// `‖L·Lᵀ − A‖` reconstruction invariant, f32 (runs on cpu AND rocm). Reads the
/// KERNEL-EMITTED L factor (l_out buffer), never re-derives it on the host.
#[test]
fn cholesky_factor_reconstructs_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    check_factor::<f32>("cholesky_f32_seed42.npz");
}

/// `‖L·Lᵀ − A‖` reconstruction invariant, f64 (cpu runs; rocm skips-with-log).
#[test]
fn cholesky_factor_reconstructs_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("cholesky f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    check_factor::<f64>("cholesky_f64_seed42.npz");
}

// ---------------------------------------------------------------------------
// RIDGE-DEFAULT-CUDA: the `alpha` diagonal parameter and the wide (`n > MAX_DIM`)
// factorization arm.
// ---------------------------------------------------------------------------

/// A well-conditioned SPD `n × n` system built on the host: `A = MᵀM + n·I` with
/// `M` from a counter-based splitmix64 stream (the `gram_test.rs` generator), plus
/// a length-`n` right-hand side. Returned in `f64`; callers narrow.
fn spd_system(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let m: Vec<f64> = (0..n * n).map(|_| next()).collect();
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += m[k * n + i] * m[k * n + j];
            }
            a[i * n + j] = acc;
        }
        a[i * n + i] += n as f64;
    }
    let b: Vec<f64> = (0..n).map(|_| next()).collect();
    (a, b)
}

/// Reference `f64` host Cholesky solve of `(A + αI)·x = b` — the oracle the
/// device arms are checked against. Deliberately the textbook row-oriented
/// recurrence, i.e. NOT the schedule either kernel uses, so an error in the
/// kernels' shared work would not be reproduced here.
fn host_solve_reg(a: &[f64], b: &[f64], n: usize, alpha: f64) -> Vec<f64> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j] + if i == j { alpha } else { 0.0 };
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = if i == j { sum.sqrt() } else { sum / l[j * n + j] };
        }
    }
    let mut z = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * z[k];
        }
        z[i] = s / l[i * n + i];
    }
    let mut x = vec![0.0f64; n];
    for step in 0..n {
        let i = n - 1 - step;
        let mut s = z[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    x
}

/// Solve `(A + αI)·x = b` on the device (one rhs) and return `x` in `f64`.
fn device_solve_reg<F>(a64: &[f64], b64: &[f64], n: usize, alpha: f64) -> Vec<f64>
where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let a: Vec<F> = a64.iter().map(|&v| from_f64::<F>(v)).collect();
    let b: Vec<F> = b64.iter().map(|&v| from_f64::<F>(v)).collect();
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let a_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &a);
    let b_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &b);
    let x_dev = cholesky_solve_reg::<F>(&mut pool, &a_dev, &b_dev, n, 1, alpha, None)
        .expect("regularized cholesky solve on an SPD system");
    let x = x_dev.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect();
    x_dev.release_into(&mut pool);
    a_dev.release_into(&mut pool);
    b_dev.release_into(&mut pool);
    x
}

/// Relative `‖got − want‖ / max(‖want‖, 1)` of two host vectors.
fn rel_err(got: &[f64], want: &[f64]) -> f64 {
    let diff: Vec<f64> = got.iter().zip(want).map(|(&g, &w)| g - w).collect();
    fro(&diff) / fro(want).max(1.0)
}

/// The `alpha` parameter must factor `(A + αI)` — i.e. adding the penalty inside
/// the kernel must equal adding it on the host and solving the plain system.
///
/// This is the check that makes the `d²` Gram round-trip removable: `Ridge`
/// stopped reading its Gram back to write `α` on the diagonal, so the ONLY thing
/// standing between the two formations is this equality.
#[test]
fn cholesky_alpha_matches_a_host_regularized_matrix() {
    let _ = env_logger::builder().is_test(true).try_init();
    const N: usize = 32;
    const ALPHA: f64 = 3.5;
    let (a, b) = spd_system(N, 42);

    // The device arm with `alpha` folded in.
    let folded = device_solve_reg::<f32>(&a, &b, N, ALPHA);

    // The composition it replaces: add α on the host, solve the plain system.
    let mut a_reg = a.clone();
    for i in 0..N {
        a_reg[i * N + i] += ALPHA;
    }
    let unfolded = device_solve_reg::<f32>(&a_reg, &b, N, 0.0);

    let rel = rel_err(&folded, &unfolded);
    assert!(
        rel <= SOLVE_TOL,
        "alpha-in-kernel vs alpha-on-host disagree by {rel:e} (> {SOLVE_TOL:e}) — \
         the Ridge round-trip removal is not equivalence-preserving"
    );

    // ...and both agree with an f64 host reference, so this is not two arms
    // being wrong the same way.
    let want = host_solve_reg(&a, &b, N, ALPHA);
    let rel_ref = rel_err(&folded, &want);
    assert!(
        rel_ref <= SOLVE_TOL,
        "alpha-in-kernel vs the f64 host oracle disagree by {rel_ref:e} (> {SOLVE_TOL:e})"
    );
}

/// `alpha = 0` must leave the unregularized callers BIT-IDENTICAL, which is what
/// lets `cholesky_solve` be a thin forward to `cholesky_solve_reg`.
#[test]
fn cholesky_alpha_zero_is_bitwise_the_plain_solve() {
    let _ = env_logger::builder().is_test(true).try_init();
    const N: usize = 16;
    let (a, b) = spd_system(N, 7);
    let a32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
    let b32: Vec<f32> = b.iter().map(|&v| v as f32).collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let a_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &a32);
    let b_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &b32);

    let plain = cholesky_solve::<f32>(&mut pool, &a_dev, &b_dev, N, 1, None).expect("plain solve");
    let reg =
        cholesky_solve_reg::<f32>(&mut pool, &a_dev, &b_dev, N, 1, 0.0, None).expect("alpha=0");
    assert_eq!(
        plain.to_host(&pool),
        reg.to_host(&pool),
        "alpha=0 must be the plain solve BIT for BIT — anything else is a silent \
         behaviour change for KernelRidge and the memory-gate callers"
    );
}

/// The wide arm must solve an order the narrow (shared-memory) kernel cannot
/// hold — the cap that used to reject `Ridge` at `d > 64` and `KernelRidge`
/// above 64 samples with `NotSquare`.
#[test]
fn cholesky_wide_arm_solves_above_max_dim() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    for &n in &[100usize, 256] {
        let (a, b) = spd_system(n, 1234 + n as u64);
        let got = device_solve_reg::<f32>(&a, &b, n, 1.0);
        let want = host_solve_reg(&a, &b, n, 1.0);
        let rel = rel_err(&got, &want);
        assert!(
            rel <= SOLVE_TOL,
            "wide cholesky at n={n} backend={backend}: rel={rel:e} > {SOLVE_TOL:e}"
        );
        println!("cholesky wide n={n} backend={backend}: rel={rel:e}");
    }
}

/// The two arms must AGREE on an order both support.
///
/// They cover disjoint `n` ranges in production, so without `MLRS_CHOLESKY_WIDE`
/// no test could compare them and a broken wide kernel would only ever be
/// checked against a host oracle it might match for the wrong reason. The knob
/// is read BEFORE the size gate ([`use_wide_kernel`]) precisely so this can force
/// each arm at the same `n`; see the vacuity trap in the Gram dispatcher.
#[test]
fn cholesky_wide_and_narrow_arms_agree() {
    let _ = env_logger::builder().is_test(true).try_init();
    const N: usize = 48; // below MAX_DIM, so the DEFAULT here is the narrow arm
    const ALPHA: f64 = 0.75;
    let (a, b) = spd_system(N, 99);

    let narrow = {
        let _g = abflag::force("MLRS_CHOLESKY_WIDE", "0");
        device_solve_reg::<f32>(&a, &b, N, ALPHA)
    };
    let wide = {
        let _g = abflag::force("MLRS_CHOLESKY_WIDE", "1");
        device_solve_reg::<f32>(&a, &b, N, ALPHA)
    };

    // Mutation guard: if the force were ignored the two would be bit-identical,
    // which would make the tolerance assertion below vacuous. The arms
    // re-associate differently, so on a well-conditioned f32 system they agree
    // closely but essentially never exactly — assert they are CLOSE, and report
    // the exact-equality case so a silently-disarmed knob is visible.
    let rel = rel_err(&wide, &narrow);
    assert!(
        rel <= SOLVE_TOL,
        "wide and narrow cholesky arms disagree at n={N}: rel={rel:e} > {SOLVE_TOL:e}"
    );
    if wide == narrow {
        println!(
            "NOTE cholesky wide/narrow agreed BIT for BIT at n={N} — check that \
             MLRS_CHOLESKY_WIDE actually reaches the dispatcher"
        );
    }
    let want = host_solve_reg(&a, &b, N, ALPHA);
    assert!(
        rel_err(&wide, &want) <= SOLVE_TOL && rel_err(&narrow, &want) <= SOLVE_TOL,
        "both arms must also match the f64 host oracle at n={N}"
    );
}

/// The wide arm's non-SPD guard: it factors past the failing pivot with a finite
/// placeholder (so every barrier stays unconditional), but the host must STILL
/// see `NotPositiveDefinite` and never hand back the garbage solution.
#[test]
fn cholesky_wide_arm_rejects_non_spd() {
    let _ = env_logger::builder().is_test(true).try_init();
    const N: usize = 96;
    let mut a = vec![0.0f32; N * N];
    for i in 0..N {
        a[i * N + i] = if i == N / 2 { -4.0 } else { 2.0 };
    }
    let b = vec![1.0f32; N];

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let a_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &a);
    let b_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &b);

    match cholesky_solve::<f32>(&mut pool, &a_dev, &b_dev, N, 1, None) {
        Err(PrimError::NotPositiveDefinite { pivot_index, .. }) => {
            assert_eq!(
                pivot_index,
                N / 2,
                "the wide arm must report the FIRST failing pivot, not a later one"
            );
        }
        Ok(_) => panic!("the wide arm accepted an indefinite matrix — the SPD guard is broken"),
        Err(other) => panic!("expected NotPositiveDefinite, got {other:?}"),
    }
}

/// Non-SPD guard: feed a synthetically INDEFINITE matrix (a negative diagonal
/// entry makes the leading pivot non-positive) and assert the host returns
/// `PrimError::NotPositiveDefinite` (the negative-pivot flag) rather than a
/// NaN-poisoned factor (RESEARCH Pitfall 4). f32, runs on cpu AND rocm.
#[test]
fn cholesky_rejects_non_spd() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    // A clearly indefinite matrix: a negative diagonal forces a non-positive
    // pivot at index 0 (the very first sqrt argument is negative). Symmetric so
    // it is a legitimate "looks square + symmetric but not SPD" input.
    let n = CHOL_N;
    let rhs = 1usize;
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        a[i * n + i] = if i == 0 { -4.0 } else { 2.0 };
    }
    let b = vec![1.0f32; n * rhs];

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let a_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &a);
    let b_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &b);

    let res = cholesky_solve::<f32>(&mut pool, &a_dev, &b_dev, n, rhs, None);
    match res {
        Err(PrimError::NotPositiveDefinite {
            operand,
            pivot_index,
            pivot_value,
        }) => {
            assert_eq!(operand, "cholesky", "NotPositiveDefinite names the operand");
            assert_eq!(pivot_index, 0, "the negative diagonal is at index 0");
            assert!(
                pivot_value.is_finite() && pivot_value <= 0.0,
                "pivot_value should be the non-positive √ argument, got {pivot_value:e}"
            );
            println!(
                "cholesky non-SPD backend={backend}: rejected at pivot {pivot_index} \
                 (value={pivot_value:e}) — typed error, not a NaN factor"
            );
        }
        Ok(_) => panic!(
            "an indefinite matrix (negative pivot) must return NotPositiveDefinite, \
             not Ok — the SPD guard is broken"
        ),
        Err(other) => panic!("expected NotPositiveDefinite, got a different error: {other:?}"),
    }
}
