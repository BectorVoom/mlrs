//! Meta-matrix assembly, device arm (`prims::stacking_meta`) — STACK-META-01.
//!
//! The device scatter and the host copy
//! (`mlrs_algos::ensemble::stacking::concatenate_predictions`) must produce the
//! SAME BYTES, not merely close ones: neither arm performs arithmetic, so a
//! tolerance here would hide exactly the class of bug this kernel can have — a
//! block written at the wrong offset, a row stride off by one, a column of a
//! multi-column block transposed. Every assertion below is therefore an
//! equality on the raw values.
//!
//! The reference is computed here rather than imported so this test does not
//! depend on `mlrs-algos` (which depends on THIS crate); the two implementations
//! being independently written is also what makes the comparison worth making.
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::capability::{self, FloatKind};
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::stacking_meta::{concat_meta_device, meta_engine, MetaEngine};
use mlrs_backend::runtime::{self, ActiveRuntime};

/// One test case's blocks: `(cols, values)` per block, row-major and
/// `n_rows`-tall, plus the optional passthrough design.
struct Case {
    n_rows: usize,
    block_cols: Vec<usize>,
    x_cols: Option<usize>,
}

/// Fill a block with values distinguishable ACROSS blocks and across
/// (row, column) within a block — a scatter that swaps two blocks, or that
/// transposes one, has to change at least one value.
fn block_values(tag: usize, n_rows: usize, cols: usize) -> Vec<f64> {
    (0..n_rows * cols)
        .map(|i| {
            let r = i / cols;
            let c = i % cols;
            1000.0 * tag as f64 + 10.0 * r as f64 + c as f64
        })
        .collect()
}

/// The meta matrix the device arm must reproduce: blocks side by side in order,
/// then `X`.
fn host_meta_ref(case: &Case, blocks: &[Vec<f64>], x: Option<&Vec<f64>>) -> (Vec<f64>, usize) {
    let n_meta: usize = case.block_cols.iter().sum();
    let width = n_meta + case.x_cols.unwrap_or(0);
    let mut out = vec![0.0f64; case.n_rows * width];
    for r in 0..case.n_rows {
        let mut col = 0usize;
        for (b, &cols) in case.block_cols.iter().enumerate() {
            for c in 0..cols {
                out[r * width + col + c] = blocks[b][r * cols + c];
            }
            col += cols;
        }
        if let (Some(xc), Some(xv)) = (case.x_cols, x) {
            for c in 0..xc {
                out[r * width + n_meta + c] = xv[r * xc + c];
            }
        }
    }
    (out, width)
}

/// Run one case through the device arm at element type `F`, returning the
/// assembled matrix promoted to f64.
fn run_device<F>(case: &Case) -> Vec<f64>
where
    F: Float + CubeElement + Pod,
{
    let to_f = |v: f64| -> F {
        match std::mem::size_of::<F>() {
            4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
            8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
            _ => unreachable!("stacking_meta_test is f32/f64 only"),
        }
    };
    let to_f64 = |v: &F| -> f64 {
        match std::mem::size_of::<F>() {
            4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(v)) as f64,
            8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(v)),
            _ => unreachable!("stacking_meta_test is f32/f64 only"),
        }
    };

    let raw: Vec<Vec<f64>> = case
        .block_cols
        .iter()
        .enumerate()
        .map(|(b, &cols)| block_values(b + 1, case.n_rows, cols))
        .collect();
    let x_raw = case.x_cols.map(|cols| block_values(99, case.n_rows, cols));

    let typed: Vec<Vec<F>> = raw
        .iter()
        .map(|b| b.iter().copied().map(to_f).collect())
        .collect();
    let x_typed: Option<Vec<F>> = x_raw
        .as_ref()
        .map(|v| v.iter().copied().map(to_f).collect());

    let blocks: Vec<(&[F], usize)> = typed
        .iter()
        .zip(&case.block_cols)
        .map(|(v, &c)| (v.as_slice(), c))
        .collect();
    let mut offsets = Vec::with_capacity(blocks.len());
    let mut acc = 0usize;
    for &c in &case.block_cols {
        offsets.push(acc);
        acc += c;
    }
    let n_meta = acc;
    let width = n_meta + case.x_cols.unwrap_or(0);
    let x_arg = match (&x_typed, case.x_cols) {
        (Some(v), Some(c)) => Some((v.as_slice(), c)),
        _ => None,
    };

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let got = concat_meta_device::<F>(
        &mut pool,
        &blocks,
        &offsets,
        case.n_rows,
        n_meta,
        width,
        x_arg,
    )
    .expect("concat_meta_device rejects nothing for a valid layout");
    got.iter().map(to_f64).collect()
}

/// Every shape the shim can produce: one block, several blocks, a
/// multi-column block (a multi-output regressor), passthrough on and off, and a
/// row count well past one cube so the launch's over-provisioned units are
/// exercised.
fn cases() -> Vec<Case> {
    vec![
        Case {
            n_rows: 8,
            block_cols: vec![1],
            x_cols: None,
        },
        Case {
            n_rows: 8,
            block_cols: vec![1, 1],
            x_cols: None,
        },
        Case {
            n_rows: 8,
            block_cols: vec![1, 1],
            x_cols: Some(3),
        },
        Case {
            n_rows: 8,
            block_cols: vec![3],
            x_cols: None,
        },
        Case {
            n_rows: 8,
            block_cols: vec![1, 4, 2],
            x_cols: Some(5),
        },
        Case {
            n_rows: 1,
            block_cols: vec![1, 2],
            x_cols: Some(1),
        },
        Case {
            n_rows: 5000,
            block_cols: vec![1, 1, 1],
            x_cols: Some(4),
        },
    ]
}

#[test]
fn device_meta_matches_host_layout_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F32, backend, "default");

    for case in cases() {
        let raw: Vec<Vec<f64>> = case
            .block_cols
            .iter()
            .enumerate()
            .map(|(b, &cols)| block_values(b + 1, case.n_rows, cols))
            .collect();
        let x_raw = case.x_cols.map(|cols| block_values(99, case.n_rows, cols));
        let (expected, width) = host_meta_ref(&case, &raw, x_raw.as_ref());

        let got = run_device::<f32>(&case);
        assert_eq!(got.len(), case.n_rows * width);
        // The values are small integers exactly representable in f32, so the
        // dtype round-trip is exact and this stays an EQUALITY.
        assert_eq!(got, expected, "f32 meta mismatch for {:?}", case.block_cols);
    }
    println!("stacking_meta f32 backend={backend}: device scatter matches the host layout");
}

#[test]
fn device_meta_matches_host_layout_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(FloatKind::F64, backend, "default");

    if capability::skip_f64_with_log() {
        println!("stacking_meta f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }

    for case in cases() {
        let raw: Vec<Vec<f64>> = case
            .block_cols
            .iter()
            .enumerate()
            .map(|(b, &cols)| block_values(b + 1, case.n_rows, cols))
            .collect();
        let x_raw = case.x_cols.map(|cols| block_values(99, case.n_rows, cols));
        let (expected, _width) = host_meta_ref(&case, &raw, x_raw.as_ref());

        let got = run_device::<f64>(&case);
        assert_eq!(got, expected, "f64 meta mismatch for {:?}", case.block_cols);
    }
    println!("stacking_meta f64 backend={backend}: device scatter matches the host layout");
}

/// A mis-shaped block is rejected BEFORE any launch — a kernel cannot fail, so
/// an unvalidated bad shape would corrupt a neighbouring block's columns
/// instead of raising.
#[test]
fn device_meta_rejects_a_mis_shaped_block() {
    let _ = env_logger::builder().is_test(true).try_init();
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let good = vec![1.0f32, 2.0, 3.0, 4.0];
    let short = vec![1.0f32, 2.0];
    let blocks: Vec<(&[f32], usize)> = vec![(good.as_slice(), 1), (short.as_slice(), 1)];
    let err = concat_meta_device::<f32>(&mut pool, &blocks, &[0, 1], 4, 2, 2, None);
    assert!(
        err.is_err(),
        "a 2-element block for 4 rows must be rejected"
    );

    // A passthrough layout with no X is the other unlaunched rejection.
    let blocks: Vec<(&[f32], usize)> = vec![(good.as_slice(), 1)];
    let err = concat_meta_device::<f32>(&mut pool, &blocks, &[0], 4, 1, 4, None);
    assert!(
        err.is_err(),
        "a passthrough width with no X must be rejected"
    );
}

/// The `MLRS_STACK_META_ENGINE` knob is LIVE: forcing each value resolves to the
/// arm it names, and the default is `numpy`.
///
/// Without this a benchmark sweep over the knob could be comparing an arm
/// against itself and report a flat ladder as "no difference"
/// (`mlrs-bench-verify-knob-is-live`).
#[test]
fn engine_knob_resolves_each_arm() {
    let _guard = mlrs_backend::abflag::clear("MLRS_STACK_META_ENGINE");
    assert_eq!(meta_engine(), MetaEngine::Numpy, "unset must mean numpy");

    for (value, expected) in [
        ("numpy", MetaEngine::Numpy),
        ("host", MetaEngine::Host),
        ("device", MetaEngine::Device),
    ] {
        let _forced = mlrs_backend::abflag::force("MLRS_STACK_META_ENGINE", value);
        assert_eq!(meta_engine(), expected, "knob value {value:?}");
        assert_eq!(expected.as_str(), value, "arm name must round-trip");
    }

    // An unrecognized value falls back to the default rather than raising — a
    // typo in a sweep script must not surface as an exception from `fit`.
    let _typo = mlrs_backend::abflag::force("MLRS_STACK_META_ENGINE", "gpu");
    assert_eq!(meta_engine(), MetaEngine::Numpy);
}
