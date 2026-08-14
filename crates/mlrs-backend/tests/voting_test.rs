//! Prediction voting, device arm (`prims::voting`) — VOTE-01.
//!
//! Two claims are gated, and they need different assertions:
//!
//! * **`vote_transform_device` is a pure transpose.** No arithmetic, so the
//!   device result must equal the host layout BYTE for byte. A tolerance would
//!   hide exactly the bug a scatter can have — a column written at the wrong
//!   index, a row stride off by one, the whole matrix transposed.
//! * **`vote_average_device` computes**, and is held to a FEW ULP rather than to
//!   equality. The kernel accumulates in the same member order the host loop
//!   uses and divides by the same host-computed weight sum, so a reassociated
//!   accumulation or a reciprocal-multiply would still fail the bound below —
//!   but exact equality is unattainable on a real GPU, because `acc + pred·w`
//!   contracts into a fused multiply-add that rounds ONCE where the host rounds
//!   twice. Measured on rocm gfx1151 (f32): a maximum of one ULP. The cpu
//!   backend does not contract and passes the same bound at zero. The bound is
//!   deliberately tight (`4 · eps` relative) rather than the project's 1e-5:
//!   1e-5 is ~80 f32 ULP and would let a genuine accumulation bug through.
//!
//! Both references are computed here rather than imported so this test does not
//! depend on `mlrs-algos` (which depends on THIS crate); the two implementations
//! being independently written is also what makes the comparison worth making.
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::capability::{self, FloatKind};
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::voting::{
    vote_average_device, vote_engine, vote_transform_device, VoteEngine, ENGINE_KNOB,
};
use mlrs_backend::runtime::{self, ActiveRuntime};

/// One case: `n_rows` tall, `k` members, and the weights `predict` uses.
struct Case {
    n_rows: usize,
    k: usize,
    weights: Vec<f64>,
}

/// Member `j`'s prediction column, with values distinguishable ACROSS members
/// AND across rows — a scatter that swaps two columns, or that transposes the
/// matrix, has to change at least one value.
fn column_values(j: usize, n_rows: usize) -> Vec<f64> {
    (0..n_rows)
        .map(|r| 1000.0 * (j as f64 + 1.0) + r as f64)
        .collect()
}

/// The transform matrix the device arm must reproduce: column `j` is member
/// `j`'s prediction.
fn host_transform_ref(cols: &[Vec<f64>], n_rows: usize) -> Vec<f64> {
    let k = cols.len();
    let mut out = vec![0.0f64; n_rows * k];
    for (j, col) in cols.iter().enumerate() {
        for r in 0..n_rows {
            out[r * k + j] = col[r];
        }
    }
    out
}

/// The weighted mean the device arm must reproduce, in numpy's order.
fn host_average_ref(cols: &[Vec<f64>], weights: &[f64], n_rows: usize) -> Vec<f64> {
    let mut denom = weights[0];
    for &w in &weights[1..] {
        denom += w;
    }
    (0..n_rows)
        .map(|r| {
            let mut acc = cols[0][r] * weights[0];
            for j in 1..cols.len() {
                acc += cols[j][r] * weights[j];
            }
            acc / denom
        })
        .collect()
}

fn to_f<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("voting_test is f32/f64 only"),
    }
}

fn to_f64<F: Pod>(v: &F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(v)),
        _ => unreachable!("voting_test is f32/f64 only"),
    }
}

/// Run one case's transform through the device arm at element type `F`.
fn run_transform<F>(case: &Case, raw: &[Vec<f64>]) -> Vec<f64>
where
    F: Float + CubeElement + Pod,
{
    let typed: Vec<Vec<F>> = raw
        .iter()
        .map(|c| c.iter().copied().map(to_f::<F>).collect())
        .collect();
    let cols: Vec<&[F]> = typed.iter().map(|v| v.as_slice()).collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let got = vote_transform_device::<F>(&mut pool, &cols, case.n_rows)
        .expect("vote_transform_device rejects nothing for equal-length columns");
    got.iter().map(to_f64).collect()
}

/// Run one case's weighted average through the device arm at element type `F`.
///
/// The weight sum is formed in `F` and left to right, which is what the host arm
/// hands the kernel — computing it in f64 here and rounding would make the two
/// arms divide by different bits and turn the equality below into a near-miss.
fn run_average<F>(case: &Case, raw: &[Vec<f64>]) -> Vec<f64>
where
    F: Float + CubeElement + Pod + std::ops::Add<Output = F>,
{
    let typed: Vec<Vec<F>> = raw
        .iter()
        .map(|c| c.iter().copied().map(to_f::<F>).collect())
        .collect();
    let cols: Vec<&[F]> = typed.iter().map(|v| v.as_slice()).collect();
    let weights: Vec<F> = case.weights.iter().copied().map(to_f::<F>).collect();
    let mut denom = weights[0];
    for &w in &weights[1..] {
        denom = denom + w;
    }

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let got = vote_average_device::<F>(&mut pool, &cols, &weights, denom, case.n_rows)
        .expect("vote_average_device rejects nothing for equal-length columns");
    got.iter().map(to_f64).collect()
}

/// Every shape the shim can produce: one member, several members, a single row,
/// and a row count well past one cube so the launch's over-provisioned units are
/// exercised. The weights cover uniform, non-uniform, and a NEGATIVE entry
/// (numpy permits those; only a zero SUM is an error, and that never reaches the
/// device arm).
fn cases() -> Vec<Case> {
    vec![
        Case {
            n_rows: 8,
            k: 1,
            weights: vec![1.0],
        },
        Case {
            n_rows: 8,
            k: 2,
            weights: vec![1.0, 1.0],
        },
        Case {
            n_rows: 8,
            k: 3,
            weights: vec![2.0, 1.0, 3.0],
        },
        Case {
            n_rows: 1,
            k: 4,
            weights: vec![0.25, 0.25, 0.25, 0.25],
        },
        Case {
            n_rows: 8,
            k: 2,
            weights: vec![3.0, -1.0],
        },
        Case {
            n_rows: 5000,
            k: 3,
            weights: vec![1.5, 0.5, 2.0],
        },
    ]
}

#[test]
fn device_transform_matches_the_host_transpose_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F32, backend, "default");

    for case in cases() {
        let raw: Vec<Vec<f64>> = (0..case.k).map(|j| column_values(j, case.n_rows)).collect();
        let expected = host_transform_ref(&raw, case.n_rows);
        let got = run_transform::<f32>(&case, &raw);
        assert_eq!(got.len(), case.n_rows * case.k);
        // The values are small integers exactly representable in f32, so the
        // dtype round-trip is exact and this stays an EQUALITY.
        assert_eq!(got, expected, "f32 transform mismatch at k={}", case.k);
    }
    println!("voting f32 backend={backend}: device transform matches the host transpose");
}

#[test]
fn device_transform_matches_the_host_transpose_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F64, backend, "default");

    // `f64_device_kernels_available`, NOT `skip_f64_with_log` (the ADVERTISED
    // flag): rocm and cuda decline to advertise f64 because their MATMUL rejects
    // f64 operands, yet they run these kernels — a multiply-add and a copy —
    // fine. Gating on the narrow flag would skip the exact cells the Python
    // `device` arm goes on to exercise (STACK-META-01's landmine).
    if !capability::f64_device_kernels_available() {
        println!("voting f64 backend={backend}: SKIPPED (no f64 device kernels on this adapter)");
        return;
    }

    for case in cases() {
        let raw: Vec<Vec<f64>> = (0..case.k).map(|j| column_values(j, case.n_rows)).collect();
        let expected = host_transform_ref(&raw, case.n_rows);
        let got = run_transform::<f64>(&case, &raw);
        assert_eq!(got, expected, "f64 transform mismatch at k={}", case.k);
    }
    println!("voting f64 backend={backend}: device transform matches the host transpose");
}

/// Assert the device reduction against the host one to within `n_ulp` relative
/// ULP of the element type.
///
/// Written as a relative bound rather than an absolute one because the fixtures
/// span several orders of magnitude; `eps` is the element type's, so the f32 and
/// f64 cells are held to the same NUMBER of ULP rather than to the same
/// distance.
fn assert_within_ulp(got: &[f64], expected: &[f64], eps: f64, n_ulp: f64, what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: length");
    for (r, (&g, &e)) in got.iter().zip(expected).enumerate() {
        let tol = n_ulp * eps * e.abs().max(f64::MIN_POSITIVE);
        assert!(
            (g - e).abs() <= tol,
            "{what}: row {r} was {g} against {e} (tolerance {tol})"
        );
    }
}

#[test]
fn device_average_matches_the_host_reduction_to_within_a_few_ulp_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F32, backend, "default");

    for case in cases() {
        // Fractional values, so the sum is NOT exactly representable and a
        // reassociated accumulation or a fused multiply-add would show up.
        let raw: Vec<Vec<f64>> = (0..case.k)
            .map(|j| {
                (0..case.n_rows)
                    .map(|r| 0.1 + j as f64 * 0.37 + r as f64 * 0.013)
                    .collect()
            })
            .collect();
        let got = run_average::<f32>(&case, &raw);

        // The reference is computed in f32 too — the whole claim is that the
        // device reduces in the ELEMENT type, not in something wider.
        let f32_cols: Vec<Vec<f64>> = raw
            .iter()
            .map(|c| c.iter().map(|&v| v as f32 as f64).collect())
            .collect();
        let expected = host_average_ref_f32(&f32_cols, &case.weights, case.n_rows);
        assert_within_ulp(
            &got,
            &expected,
            f64::from(f32::EPSILON),
            4.0,
            &format!("f32 average at k={}", case.k),
        );
    }
    println!("voting f32 backend={backend}: device average matches the host reduction to <=4 ULP");
}

/// The f32 twin of [`host_average_ref`] — every operation rounded to f32, in
/// numpy's order, so the comparison is against the arithmetic the kernel
/// actually performs rather than against a more accurate answer.
fn host_average_ref_f32(cols: &[Vec<f64>], weights: &[f64], n_rows: usize) -> Vec<f64> {
    let w: Vec<f32> = weights.iter().map(|&v| v as f32).collect();
    let mut denom = w[0];
    for &v in &w[1..] {
        denom += v;
    }
    (0..n_rows)
        .map(|r| {
            let mut acc = cols[0][r] as f32 * w[0];
            for j in 1..cols.len() {
                acc += cols[j][r] as f32 * w[j];
            }
            (acc / denom) as f64
        })
        .collect()
}

#[test]
fn device_average_matches_the_host_reduction_to_within_a_few_ulp_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F64, backend, "default");

    // `f64_device_kernels_available`, NOT `skip_f64_with_log` (the ADVERTISED
    // flag): rocm and cuda decline to advertise f64 because their MATMUL rejects
    // f64 operands, yet they run these kernels — a multiply-add and a copy —
    // fine. Gating on the narrow flag would skip the exact cells the Python
    // `device` arm goes on to exercise (STACK-META-01's landmine).
    if !capability::f64_device_kernels_available() {
        println!("voting f64 backend={backend}: SKIPPED (no f64 device kernels on this adapter)");
        return;
    }

    for case in cases() {
        let raw: Vec<Vec<f64>> = (0..case.k)
            .map(|j| {
                (0..case.n_rows)
                    .map(|r| 0.1 + j as f64 * 0.37 + r as f64 * 0.013)
                    .collect()
            })
            .collect();
        let expected = host_average_ref(&raw, &case.weights, case.n_rows);
        let got = run_average::<f64>(&case, &raw);
        assert_within_ulp(
            &got,
            &expected,
            f64::EPSILON,
            4.0,
            &format!("f64 average at k={}", case.k),
        );
    }
    println!("voting f64 backend={backend}: device average matches the host reduction to <=4 ULP");
}

#[test]
fn mismatched_column_lengths_are_rejected_before_any_launch() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [1.0f32, 2.0];
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    assert!(vote_transform_device::<f32>(&mut pool, &[&a[..], &b[..]], 3).is_err());
    assert!(vote_average_device::<f32>(&mut pool, &[&a[..], &b[..]], &[1.0, 1.0], 2.0, 3).is_err());
    // A weight vector that does not match the column count is the other way the
    // kernel could read out of range, and it is rejected on the host too.
    assert!(vote_average_device::<f32>(&mut pool, &[&a[..]], &[1.0, 1.0], 2.0, 3).is_err());
}

#[test]
fn the_engine_knob_defaults_to_numpy_and_names_the_arm_it_resolves() {
    // The knob is read through `abflag`, which is the test-visible indirection
    // (`mlrs-abflag-test-knobs`): never `std::env::set_var` from a test.
    assert_eq!(VoteEngine::Numpy.as_str(), "numpy");
    assert_eq!(VoteEngine::Host.as_str(), "host");
    assert_eq!(VoteEngine::Device.as_str(), "device");
    assert_eq!(ENGINE_KNOB, "MLRS_VOTING_ENGINE");
    // Unset in the test process, so the default arm is what a user gets.
    if mlrs_backend::abflag::var(ENGINE_KNOB).is_none() {
        assert_eq!(vote_engine(), VoteEngine::Numpy);
    }
}

// --------------------------------------------------------------------------- //
// VotingClassifier (VOTE-CLF-01)
// --------------------------------------------------------------------------- //
//
// The classifier's device arm adds three claims to the two above:
//
// * `vote_hard_predict_device` is a weighted bincount argmax, and — unlike the
//   average — it is held to EXACT equality. There is no `acc + a·b` for a GPU to
//   contract: the kernel adds a scalar weight to a bin, which is one rounding
//   everywhere. If this ever drifts, the cause is the tally or the tie-break,
//   not the hardware.
// * `vote_soft_predict_device` FUSES the argmax into the reduction, so its
//   labels must equal the argmax of what `vote_soft_proba_device` returns
//   separately. That is the assertion that catches a fused path reading the
//   accumulator before the divide, or with the wrong stride.
// * `vote_hstack_device` is a pure scatter and is held byte-for-byte, for the
//   reason `vote_transform_device` is.

use mlrs_backend::prims::voting::{
    vote_hard_predict_device, vote_hstack_device, vote_soft_predict_device, vote_soft_proba_device,
};

/// A deterministic label matrix with the properties the tally has to survive:
/// unanimous rows, split rows, three-way ties, and a row whose members all vote
/// for a class ABOVE 0 (so a leaked tally from the previous row is visible).
fn label_columns(k: usize, n_rows: usize, n_classes: u32) -> Vec<Vec<u32>> {
    (0..k)
        .map(|j| {
            (0..n_rows)
                .map(|r| ((r * 7 + j * 3 + r / 5) as u32) % n_classes)
                .collect()
        })
        .collect()
}

/// The host reference for a weighted bincount argmax, written independently of
/// `mlrs_algos` (which depends on this crate) — including `np.bincount`'s
/// per-row length, which is the subtle half.
fn host_hard_ref(cols: &[Vec<u32>], weights: &[f64], n_rows: usize, n_bins: usize) -> Vec<u32> {
    (0..n_rows)
        .map(|r| {
            let mut tally = vec![0.0f64; n_bins];
            let mut hi = 0u32;
            for (j, col) in cols.iter().enumerate() {
                tally[col[r] as usize] += weights[j];
                hi = hi.max(col[r]);
            }
            let mut best = tally[0];
            let mut best_idx = 0u32;
            for c in 1..=hi {
                if tally[c as usize] > best {
                    best = tally[c as usize];
                    best_idx = c;
                }
            }
            best_idx
        })
        .collect()
}

fn hard_device<F>(cols: &[Vec<u32>], weights: &[f64], n_rows: usize, n_bins: u32) -> Vec<u32>
where
    F: Float + CubeElement + Pod,
{
    let refs: Vec<&[u32]> = cols.iter().map(|v| v.as_slice()).collect();
    let typed: Vec<F> = weights.iter().copied().map(to_f::<F>).collect();
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    vote_hard_predict_device::<F>(&mut pool, &refs, &typed, n_rows, n_bins)
        .expect("hard predict device arm")
}

#[test]
fn hard_voting_on_the_device_equals_the_host_bincount_argmax_exactly() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F32, backend, "default");

    for (k, n_rows, n_classes) in [
        (1usize, 17usize, 3u32),
        (3, 257, 3),
        (5, 1024, 7),
        (8, 33, 2),
    ] {
        let cols = label_columns(k, n_rows, n_classes);
        for weights in [vec![1.0f64; k], (1..=k).map(|j| j as f64 * 1.5).collect()] {
            let expected = host_hard_ref(&cols, &weights, n_rows, n_classes as usize);
            let got = hard_device::<f32>(&cols, &weights, n_rows, n_classes);
            // EXACT: the tally is an add of a scalar into a bin, not a
            // multiply-accumulate, so there is nothing for a GPU to contract.
            assert_eq!(got, expected, "hard vote at k={k}, n={n_rows}");
        }
    }
    println!("voting-clf backend={backend}: device hard vote matches the host tally exactly");
}

#[test]
fn a_hard_vote_on_the_device_respects_each_rows_own_label_ceiling() {
    // The negative-weight case from the algos suite, on the device: `np.bincount`
    // never sizes past `x.max() + 1`, so class 2 is not a candidate here even
    // though its implicit 0.0 beats both members' negative weights. A kernel
    // that scanned the full `n_bins` would answer 1.
    let a = vec![0u32];
    let b = vec![0u32];
    let got = hard_device::<f32>(&[a, b], &[-1.0, -2.0], 1, 3);
    assert_eq!(got, vec![0]);
}

#[test]
fn the_hard_vote_tally_is_zeroed_before_the_first_member_accumulates() {
    // A pool handle is whatever its last owner left. Running twice through the
    // SAME pool with different data is what makes a missing zero-fill visible:
    // the second call would see the first call's counts.
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let all_two = vec![vec![2u32; 64], vec![2u32; 64], vec![2u32; 64]];
    let all_zero = vec![vec![0u32; 64], vec![0u32; 64], vec![0u32; 64]];
    let w = [1.0f32; 3];
    for (cols, expected) in [(&all_two, 2u32), (&all_zero, 0u32)] {
        let refs: Vec<&[u32]> = cols.iter().map(|v| v.as_slice()).collect();
        let got = vote_hard_predict_device::<f32>(&mut pool, &refs, &w, 64, 3).unwrap();
        assert_eq!(got, vec![expected; 64]);
    }
}

#[test]
fn soft_voting_fuses_the_argmax_without_changing_which_class_wins() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F32, backend, "default");

    let (n_rows, n_cols, k) = (200usize, 4usize, 3usize);
    // Well-separated probabilities: the two arms are the same kernel chain, but
    // holding a FUSED argmax to a near-tie would be gating on the contraction
    // gap rather than on the fusion.
    let blocks: Vec<Vec<f32>> = (0..k)
        .map(|j| {
            (0..n_rows * n_cols)
                .map(|i| {
                    let r = i / n_cols;
                    let c = i % n_cols;
                    if c == (r + j) % n_cols {
                        0.7f32
                    } else {
                        0.1f32
                    }
                })
                .collect()
        })
        .collect();
    let refs: Vec<&[f32]> = blocks.iter().map(|v| v.as_slice()).collect();
    let w = [2.0f32, 1.0, 1.0];
    let denom = w[0] + w[1] + w[2];

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let proba = vote_soft_proba_device::<f32>(&mut pool, &refs, &w, denom, n_rows, n_cols).unwrap();
    let fused =
        vote_soft_predict_device::<f32>(&mut pool, &refs, &w, denom, n_rows, n_cols).unwrap();

    assert_eq!(proba.len(), n_rows * n_cols);
    let separate: Vec<u32> = (0..n_rows)
        .map(|r| {
            let row = &proba[r * n_cols..(r + 1) * n_cols];
            let mut best = row[0];
            let mut best_idx = 0u32;
            for (c, &v) in row.iter().enumerate().skip(1) {
                if v > best {
                    best = v;
                    best_idx = c as u32;
                }
            }
            best_idx
        })
        .collect();
    assert_eq!(
        fused, separate,
        "fused argmax disagreed with the downloaded one"
    );
    // Member 0 carries twice the weight, so its peak class wins every row.
    let expected: Vec<u32> = (0..n_rows).map(|r| (r % n_cols) as u32).collect();
    assert_eq!(fused, expected);
    println!("voting-clf backend={backend}: fused soft predict matches the two-step route");
}

#[test]
fn hstack_on_the_device_is_byte_identical_to_the_host_layout() {
    for (k, n_rows, width) in [(1usize, 5usize, 3usize), (2, 128, 4), (4, 37, 1)] {
        let blocks: Vec<Vec<f32>> = (0..k)
            .map(|j| {
                (0..n_rows * width)
                    .map(|i| 1000.0 * (j as f32 + 1.0) + i as f32)
                    .collect()
            })
            .collect();
        let refs: Vec<&[f32]> = blocks.iter().map(|v| v.as_slice()).collect();
        let client = runtime::active_client();
        let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
        let got = vote_hstack_device::<f32>(&mut pool, &refs, n_rows, width).unwrap();

        let stride = k * width;
        let mut expected = vec![0.0f32; n_rows * stride];
        for (j, block) in blocks.iter().enumerate() {
            for r in 0..n_rows {
                for c in 0..width {
                    expected[r * stride + j * width + c] = block[r * width + c];
                }
            }
        }
        // No arithmetic, so equality — a column written at the wrong offset or a
        // stride off by one is precisely what a tolerance would hide.
        assert_eq!(got, expected, "hstack at k={k}, width={width}");
    }
}

#[test]
fn the_classifier_arms_reject_a_malformed_call_before_any_launch() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let a = [0u32, 1, 2];
    let b = [0u32, 1];
    // Mismatched column lengths, and a weight vector that does not match `k`:
    // both would make a kernel read past its upload rather than fail.
    assert!(
        vote_hard_predict_device::<f32>(&mut pool, &[&a[..], &b[..]], &[1.0, 1.0], 3, 3).is_err()
    );
    assert!(vote_hard_predict_device::<f32>(&mut pool, &[&a[..]], &[1.0, 1.0], 3, 3).is_err());
    // Zero classes has no argmax to take.
    assert!(vote_hard_predict_device::<f32>(&mut pool, &[&a[..]], &[1.0], 3, 0).is_err());

    let p = [0.5f32, 0.5, 0.25, 0.75];
    let q = [0.5f32, 0.5];
    assert!(
        vote_soft_proba_device::<f32>(&mut pool, &[&p[..], &q[..]], &[1.0, 1.0], 2.0, 2, 2)
            .is_err()
    );
    assert!(vote_soft_predict_device::<f32>(&mut pool, &[&p[..]], &[1.0], 1.0, 2, 0).is_err());
    assert!(vote_hstack_device::<f32>(&mut pool, &[&p[..], &q[..]], 2, 2).is_err());
}

#[test]
fn the_classifier_arms_run_at_f64_where_the_backend_has_f64_kernels() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F64, backend, "default");
    if !capability::f64_device_kernels_available() {
        println!("voting-clf f64 backend={backend}: SKIPPED (no f64 device kernels)");
        return;
    }
    // f64 is the width `np.bincount(x, weights=w)` accumulates in, so this is
    // the configuration the Python `host` arm is compared against.
    let cols = label_columns(4, 301, 5);
    let weights: Vec<f64> = vec![0.25, 1.5, 3.0, 0.75];
    let expected = host_hard_ref(&cols, &weights, 301, 5);
    let got = hard_device::<f64>(&cols, &weights, 301, 5);
    assert_eq!(got, expected);
    println!("voting-clf f64 backend={backend}: device hard vote matches the host tally exactly");
}
