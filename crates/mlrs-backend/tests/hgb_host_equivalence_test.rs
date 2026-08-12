//! GBT-PERF-CPU — the cpu HOST arm of the HistGradientBoosting fit must be a
//! BITWISE replay of the device kernel pipeline, not merely a close one.
//!
//! Gradient boosting is sequential: iteration `t + 1`'s trees are grown from
//! iteration `t`'s raw predictions, so a single last-ULP difference compounds
//! and can flip a split — a macroscopic change in the fitted model that a
//! tolerance test would only catch by luck. These tests therefore compare the
//! two arms' complete-tree model arrays BIT FOR BIT (`to_bits`), across all
//! three losses, both float types, and both histogram paths (the single-shot
//! sibling-subtraction path and the deep-level node-CHUNKED fallback).
//!
//! They also pin the pool's THREAD-COUNT INDEPENDENCE: a 1-worker replay and
//! the default multi-worker replay must agree bitwise, which is what makes the
//! fit reproducible on any machine.
//!
//! The arms are selected through the public prim with the `abflag`
//! thread-local override (never `std::env::set_var` — see the `abflag` module
//! doc on the `environ` data race and silently-vacuous kernel-agreement
//! assertions).
//!
//! cpu-only: `hgb_host` is gated to the cpu backend, so on wgpu/CUDA/ROCm both
//! arms would be the same device path and the comparison would be vacuous.
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)]` module.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device::Device;
use mlrs_backend::abflag;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::hist_gradient_boosting::{
    hgb_fit_class, hgb_fit_reg, HgbModel, HgbParams,
};
use mlrs_backend::runtime::{self, ActiveRuntime};

/// The host arm only exists on cpu; elsewhere this comparison is vacuous.
fn skip_off_cpu() -> bool {
    if capability::active_backend_name() != "cpu" {
        eprintln!("hgb host-arm equivalence is cpu-only; skipping");
        return true;
    }
    false
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform01(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("hgb tests are f32/f64 only"),
    }
}

/// Deterministic features plus a 3-class rule (and its regression twin).
fn make_data<F: Pod>(n: usize, d: usize, seed: u64) -> (Vec<F>, Vec<F>, Vec<u32>) {
    let mut s = seed;
    let mut x = Vec::with_capacity(n * d);
    let mut y_reg = Vec::with_capacity(n);
    let mut y_idx = Vec::with_capacity(n);
    for _ in 0..n {
        let mut row = Vec::with_capacity(d);
        for _ in 0..d {
            row.push(uniform01(&mut s));
        }
        let (a, b) = (row[0], row[1]);
        // A non-additive target so the trees genuinely keep splitting.
        y_reg.push(f64_to::<F>(
            3.0 * a - 2.0 * b * b + 0.5 * (a * b) + uniform01(&mut s) * 0.1,
        ));
        y_idx.push(if a < 0.4 {
            0u32
        } else if b < 0.5 {
            1
        } else {
            2
        });
        x.extend(row.into_iter().map(f64_to::<F>));
    }
    (x, y_reg, y_idx)
}

/// Every model array of a fitted ensemble, as raw bits (float arrays go
/// through `to_bits` so `-0.0` and NaN payloads compare exactly too).
struct ModelBits {
    split_feature: Vec<u32>,
    is_leaf: Vec<u32>,
    threshold: Vec<u64>,
    leaf_value: Vec<u64>,
    baseline: Vec<u64>,
}

fn bits_of<F: Pod>(v: &[F]) -> Vec<u64> {
    v.iter()
        .map(|x| match std::mem::size_of::<F>() {
            4 => bytemuck::from_bytes::<f32>(bytemuck::bytes_of(x)).to_bits() as u64,
            8 => bytemuck::from_bytes::<f64>(bytemuck::bytes_of(x)).to_bits(),
            _ => unreachable!(),
        })
        .collect()
}

fn snapshot<F>(model: &HgbModel<F>, pool: &BufferPool<ActiveRuntime>) -> ModelBits
where
    F: Float + CubeElement + Pod,
{
    ModelBits {
        split_feature: model.split_feature_host(pool),
        is_leaf: model.is_leaf_host(pool),
        threshold: bits_of(&model.threshold_host(pool)),
        leaf_value: bits_of(&model.leaf_value_host(pool)),
        baseline: bits_of(&model.baseline_host(pool)),
    }
}

fn assert_same(a: &ModelBits, b: &ModelBits, what: &str) {
    assert_eq!(a.split_feature, b.split_feature, "{what}: split_feature");
    assert_eq!(a.is_leaf, b.is_leaf, "{what}: is_leaf");
    assert_eq!(a.threshold, b.threshold, "{what}: threshold bits");
    assert_eq!(a.leaf_value, b.leaf_value, "{what}: leaf_value bits");
    assert_eq!(a.baseline, b.baseline, "{what}: baseline bits");
    // A model whose every node is a leaf would make the comparison vacuous.
    assert!(
        a.is_leaf.iter().any(|&l| l == 0),
        "{what}: the fit produced no interior node — the comparison is vacuous"
    );
}

/// What the fit is fitting.
#[derive(Clone, Copy)]
enum Kind {
    Reg,
    Binary,
    Multi,
}

/// Fit one configuration under the given A/B knobs and return its model bits.
fn fit_bits<F>(
    kind: Kind,
    n: usize,
    d: usize,
    params: &HgbParams,
    host: bool,
    workers: &str,
) -> ModelBits
where
    F: Float + CubeElement + Pod,
{
    let _g_host = abflag::force("MLRS_HGB_HOST", if host { "1" } else { "0" });
    let _g_workers = abflag::force("MLRS_HGB_WORKERS", workers);
    // Pin the DEFAULT histogram stripe policy: `MLRS_HGB_EXACT=0` deliberately
    // regroups the histogram sums, so an ambient setting of it would make this
    // bitwise assertion fail for a legitimate reason. `clear` forces "unset"
    // on this thread regardless of the process environment.
    let _g_exact = abflag::clear("MLRS_HGB_EXACT");

    let (x, y_reg, y_idx) = make_data::<F>(n, d, 7);
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);

    let bits = match kind {
        Kind::Reg => {
            let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_reg);
            let m = hgb_fit_reg::<F>(&mut pool, &x_dev, (n, d), &y_dev, params, Device::Auto).expect("fit reg");
            snapshot(&m, &pool)
        }
        Kind::Binary => {
            let y2: Vec<u32> = y_idx.iter().map(|&c| u32::from(c >= 1)).collect();
            let m = hgb_fit_class::<F>(&mut pool, &x_dev, (n, d), &y2, 2, params, Device::Auto).expect("fit bin");
            snapshot(&m, &pool)
        }
        Kind::Multi => {
            let m = hgb_fit_class::<F>(&mut pool, &x_dev, (n, d), &y_idx, 3, params, Device::Auto)
                .expect("fit multi");
            snapshot(&m, &pool)
        }
    };
    bits
}

/// The default geometry: `max_depth = 4` with `n_bins = 32` keeps every level
/// on the single-shot sibling-SUBTRACTION path (`nodes <= subtract_cap`),
/// which is the path a real fit takes.
fn default_params() -> HgbParams {
    HgbParams {
        max_iter: 12,
        max_depth: 4,
        n_bins: 32,
        learning_rate: 0.1,
        l2_regularization: 0.0,
        min_samples_leaf: 5,
    }
}

/// A geometry whose per-node histogram is fat enough that `subtract_cap` falls
/// BELOW the deepest level's node count, forcing the node-CHUNKED fallback
/// (`hgb_fit_impl`'s deep-level branch) — the other half of the host replay.
fn chunked_params() -> HgbParams {
    HgbParams {
        max_iter: 3,
        max_depth: 5,
        n_bins: 256,
        learning_rate: 0.2,
        l2_regularization: 0.5,
        min_samples_leaf: 2,
    }
}

fn run_kind<F>(kind: Kind, label: &str, params: &HgbParams, n: usize, d: usize)
where
    F: Float + CubeElement + Pod,
{
    let device = fit_bits::<F>(kind, n, d, params, false, "1");
    let host_par = fit_bits::<F>(kind, n, d, params, true, "4");
    let host_serial = fit_bits::<F>(kind, n, d, params, true, "1");
    assert_same(
        &device,
        &host_par,
        &format!("{label} host(4 workers) vs device"),
    );
    assert_same(
        &host_serial,
        &host_par,
        &format!("{label} host worker-count independence"),
    );
}

#[test]
fn host_arm_matches_device_bitwise_f32() {
    if skip_off_cpu() {
        return;
    }
    let p = default_params();
    run_kind::<f32>(Kind::Reg, "reg f32", &p, 1500, 6);
    run_kind::<f32>(Kind::Binary, "binary f32", &p, 1500, 6);
    run_kind::<f32>(Kind::Multi, "multi f32", &p, 1500, 6);
}

#[test]
fn host_arm_matches_device_bitwise_f64() {
    if skip_off_cpu() {
        return;
    }
    let p = default_params();
    run_kind::<f64>(Kind::Reg, "reg f64", &p, 1500, 6);
    run_kind::<f64>(Kind::Binary, "binary f64", &p, 1500, 6);
    run_kind::<f64>(Kind::Multi, "multi f64", &p, 1500, 6);
}

/// Assert this geometry really lands on the deep-level node-CHUNKED branch —
/// otherwise the "chunked path" test would silently re-test the single-shot
/// one. Mirrors `hgb_fit_impl`'s `subtract_cap` arithmetic (the 64 MiB
/// `HGB_HIST_BUDGET_BYTES` transient budget divided by the per-node
/// histogram + scores footprint).
fn assert_chunks<F>(p: &HgbParams, d: usize, k: usize) {
    let (nb, sz) = (p.n_bins, std::mem::size_of::<F>());
    let per_node_bytes = k * d * nb * 3 * sz * 2 + k * d * (nb - 1) * sz;
    let cap = (64usize << 20) / per_node_bytes.max(1);
    assert!(
        (1usize << p.max_depth) > cap,
        "geometry does not chunk (deepest level {} <= cap {cap}) — the test would be vacuous",
        1usize << p.max_depth
    );
}

#[test]
fn host_arm_matches_device_on_the_chunked_path() {
    if skip_off_cpu() {
        return;
    }
    // A wide, finely-binned multiclass fit: one node's histogram budget puts
    // `subtract_cap` below the deepest level's node count, so the fit takes
    // the node-chunked branch (and never the sibling subtraction there).
    let p = chunked_params();
    assert_chunks::<f64>(&p, 64, 3);
    run_kind::<f64>(Kind::Multi, "multi f64 chunked", &p, 600, 64);
    assert_chunks::<f32>(&p, 128, 3);
    run_kind::<f32>(Kind::Multi, "multi f32 chunked", &p, 400, 128);
}
