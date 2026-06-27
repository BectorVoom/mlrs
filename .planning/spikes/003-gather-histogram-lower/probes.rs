//! Standalone-launch VALUE-asserting probes for the three Phase-17 tree spike
//! kernels (SC-1 / A1 lowering / A4 argmax-safety evidence for TREE-01).
//!
//! Each probe hand-builds a tiny input whose exact result is computable by an
//! in-test host oracle, launches ONE kernel through its `tree_spike` wrapper, and
//! asserts the read-back equals the oracle — never a bare non-panic (a 002-B
//! silent miscompile compiles, launches, and returns plausible-wrong data). The
//! histogram probe additionally guards the 002-A all-zeros symptom (a kernel that
//! never launched reads back zeros).
//!
//! Every probe is a generic `fn ..<F>()` run for f32 (always) and f64 (the cpu
//! correctness gate; SKIPS-with-log on an adapter lacking f64 — e.g. rocm). The
//! kernel source lives in `tree_spike/mod.rs`; this file is host-side launch +
//! assert only, never fenced kernel code.
//!
//! Per AGENTS.md, tests live in `tests/`, never as `#[cfg(test)] mod tests`.

mod tree_spike;

use cubecl::prelude::{CubeElement, Float};
use mlrs_backend::capability;
use tree_spike::{
    build_tree, from_f64, host_to_f64, launch_histogram, launch_relabel, launch_split_find,
};

/// f64 skip-with-log gate + a backend/dtype log line. Returns `true` when the
/// probe should early-return (f64 unsupported on this adapter). f32 always runs.
fn gate_and_log<F: bytemuck::Pod>(label: &str) -> bool {
    if std::mem::size_of::<F>() == 8 && capability::skip_f64_with_log() {
        println!(
            "{label} f64 backend={}: SKIPPED (no f64 support on this adapter)",
            capability::active_backend_name()
        );
        return true;
    }
    println!(
        "{label} backend={} dtype={}: running",
        capability::active_backend_name(),
        std::any::type_name::<F>()
    );
    false
}

// ─────────────────────────────────────────────────────────────────────────────
// Probe 1 — GATHER histogram: per-cell count + value-sum vs an in-test oracle.
// ─────────────────────────────────────────────────────────────────────────────

fn check_histogram<F>()
where
    F: Float + CubeElement + bytemuck::Pod,
{
    if gate_and_log::<F>("tree_histogram") {
        return;
    }
    let n_samples = 6usize;
    let n_feat = 2usize;
    let n_nodes = 2usize;
    let n_bins = 3usize;

    let node_id = vec![0u32, 0, 0, 1, 1, 1];
    // binned[s * n_feat + f], all values in 0..n_bins.
    let binned = vec![
        0u32, 1, // s0: f0=0 f1=1
        1, 2, // s1: f0=1 f1=2
        0, 0, // s2: f0=0 f1=0
        2, 1, // s3: f0=2 f1=1
        1, 1, // s4: f0=1 f1=1
        2, 2, // s5: f0=2 f1=2
    ];
    let y_f64 = [1.0f64, 0.0, 1.0, 0.0, 1.0, 1.0];
    let y: Vec<F> = y_f64.iter().map(|&v| from_f64::<F>(v)).collect();

    // In-test host oracle: independent recompute of every cell.
    let n_cells = n_nodes * n_feat * n_bins;
    let mut want_c = vec![0.0f64; n_cells];
    let mut want_v = vec![0.0f64; n_cells];
    for s in 0..n_samples {
        let nid = node_id[s] as usize;
        for f in 0..n_feat {
            let b = binned[s * n_feat + f] as usize;
            let cell = (nid * n_feat + f) * n_bins + b;
            want_c[cell] += 1.0;
            want_v[cell] += y_f64[s];
        }
    }

    let (counts, vsums) =
        launch_histogram::<F>(&node_id, &binned, &y, n_samples, n_feat, n_nodes, n_bins);

    // 002-A guard: a kernel that never launched reads back all zeros.
    assert!(
        counts.iter().any(|&v| host_to_f64(v) != 0.0),
        "histogram read back all zeros — kernel did not launch (002-A loud failure)"
    );

    for c in 0..n_cells {
        let gc = host_to_f64(counts[c]);
        let gv = host_to_f64(vsums[c]);
        assert!(
            (gc - want_c[c]).abs() <= 1e-6,
            "count mismatch at cell {c}: got {gc} want {}",
            want_c[c]
        );
        assert!(
            (gv - want_v[c]).abs() <= 1e-6,
            "value-sum mismatch at cell {c}: got {gv} want {}",
            want_v[c]
        );
    }
    println!(
        "tree_histogram [{}]: launched + per-cell count/value-sum match host oracle ✓",
        std::any::type_name::<F>()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Probe 2 — split-find argmax with a deliberate gain TIE (A4 VALUE assertion).
// ─────────────────────────────────────────────────────────────────────────────

fn check_split_find<F>()
where
    F: Float + CubeElement + bytemuck::Pod,
{
    if gate_and_log::<F>("tree_splitfind") {
        return;
    }
    let n_nodes = 1usize;
    let n_cand = 4usize;
    // Candidates: (col0,bin0)=0.1, (col0,bin1)=0.5, (col1,bin0)=0.5, (col1,bin1)=0.2.
    // Max gain 0.5 TIES between c1 (col0,bin1) and c2 (col1,bin0). The tie rule
    // (lowest feature index, then lowest bin) must pick c1 → (col0, bin1).
    let gain_f64 = [0.1f64, 0.5, 0.5, 0.2];
    let gain: Vec<F> = gain_f64.iter().map(|&v| from_f64::<F>(v)).collect();
    let col_of = vec![0u32, 0, 1, 1];
    let bin_of = vec![0u32, 1, 0, 1];

    let (bg, bc, bb) = launch_split_find::<F>(&gain, &col_of, &bin_of, n_nodes, n_cand);

    // 002-A guard: a non-launch reads back 0 gain (every candidate here is > 0).
    assert!(
        host_to_f64(bg[0]) != 0.0,
        "split-find read back 0 gain — kernel did not launch (002-A)"
    );
    assert!(
        (host_to_f64(bg[0]) - 0.5).abs() <= 1e-6,
        "best gain: got {} want 0.5",
        host_to_f64(bg[0])
    );
    // A4: the tie resolves to the lowest feature index, then lowest bin.
    assert_eq!(
        bc[0], 0u32,
        "argmax tie must resolve to the lowest feature index (col 0), got col {}",
        bc[0]
    );
    assert_eq!(
        bb[0], 1u32,
        "argmax tie winner bin must be 1 (col 0's winning bin), got bin {}",
        bb[0]
    );
    println!(
        "tree_splitfind [{}]: argmax + tie→lowest(feature,bin) value-correct ✓",
        std::any::type_name::<F>()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Probe 3 — relabel-partition: exact left/right child labels per D-02.
// ─────────────────────────────────────────────────────────────────────────────

fn check_relabel<F>()
where
    F: Float + CubeElement + bytemuck::Pod,
{
    if gate_and_log::<F>("tree_relabel") {
        return;
    }
    let n_samples = 5usize;
    let n_feat = 1usize;
    let node_id = vec![0u32; n_samples];
    // Feature-0 bins per sample. Node 0 splits feature 0 after bin 1: bv > 1 → go
    // right (left_child + 1 = 2), else left (left_child = 1).
    let binned = vec![0u32, 1, 2, 1, 3];
    let split_active = vec![1u32];
    let split_col = vec![0u32];
    let split_bin = vec![1u32];
    let left_child = vec![1u32];

    let got = launch_relabel(
        &node_id,
        &binned,
        &split_active,
        &split_col,
        &split_bin,
        &left_child,
        n_samples,
        n_feat,
    );

    // 002-A guard: a non-launch leaves every label at the root (0).
    assert!(
        got.iter().any(|&v| v != 0),
        "relabel left every node_id at 0 — kernel did not launch (002-A)"
    );

    // Exact left/right child label per sample (D-02): left=1, right=left+1=2.
    for (s, &g) in got.iter().enumerate() {
        let expect = if binned[s] > 1 { 2u32 } else { 1u32 };
        assert_eq!(
            g, expect,
            "sample {s} bin={} → got child {g} want {expect} (D-02 left/right)",
            binned[s]
        );
    }
    println!(
        "tree_relabel [{}]: exact left/right child labels per D-02 ✓",
        std::any::type_name::<F>()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Probe 4 — end-to-end build_tree composing all three kernels (SC-1 backstop +
// SparseTreeNode D-02/D-03/D-04 validation).
// ─────────────────────────────────────────────────────────────────────────────

fn check_build_tree<F>()
where
    F: Float + CubeElement + bytemuck::Pod,
{
    if gate_and_log::<F>("tree_build") {
        return;
    }
    // 8 samples, 1 feature, 4 bins, cleanly separable: bins {0,1} → y=0, bins
    // {2,3} → y=1. The root split after bin 1 is perfect → two pure leaves.
    let n_samples = 8usize;
    let n_feat = 1usize;
    let n_bins = 4usize;
    let binned = vec![0u32, 1, 0, 1, 2, 3, 2, 3];
    let y_f64 = [0.0f64, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let y: Vec<F> = y_f64.iter().map(|&v| from_f64::<F>(v)).collect();
    let bin_edges = vec![vec![0.5f64, 1.5, 2.5]]; // n_bins-1 thresholds for feature 0

    let (nodes, leaves) = build_tree::<F>(
        &binned, &y, &bin_edges, n_samples, n_feat, n_bins, 3, 2,
    );

    // Root must be an internal node splitting feature 0.
    assert_eq!(
        nodes[0].colid, 0,
        "root should split on feature 0, got colid {}",
        nodes[0].colid
    );
    // D-02/D-03/D-04 invariants across every node.
    for n in &nodes {
        if n.colid >= 0 {
            assert!(
                n.left_child >= 0,
                "internal node must have a non-negative left_child (right = left+1, D-02)"
            );
        } else {
            assert_eq!(
                n.left_child, -1,
                "leaf node must have left_child == -1 (D-03)"
            );
            assert!(
                (n.value as usize) < leaves.len(),
                "leaf value must be a valid offset into the leaf buffer (D-04), got {}",
                n.value
            );
        }
    }
    assert!(!leaves.is_empty(), "leaf-value buffer must be populated");
    // Separable data → at least one pure-0 leaf and one pure-1 leaf.
    assert!(
        leaves.iter().any(|&p| p == 0.0) && leaves.iter().any(|&p| p == 1.0),
        "separable data should yield pure leaves (probs 0 and 1): {leaves:?}"
    );
    println!(
        "tree_build [{}]: {} nodes, {} leaves, D-02/D-03/D-04 hold ✓",
        std::any::type_name::<F>(),
        nodes.len(),
        leaves.len()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test entry points: f32 always runs; f64 is the cpu correctness gate.
// ─────────────────────────────────────────────────────────────────────────────

fn init() {
    let _ = env_logger::builder().is_test(true).try_init();
}

#[test]
fn tree_histogram_f32_value_correct() {
    init();
    check_histogram::<f32>();
}

#[test]
fn tree_histogram_f64_value_correct() {
    init();
    check_histogram::<f64>();
}

#[test]
fn tree_splitfind_f32_argmax_and_tie() {
    init();
    check_split_find::<f32>();
}

#[test]
fn tree_splitfind_f64_argmax_and_tie() {
    init();
    check_split_find::<f64>();
}

#[test]
fn tree_relabel_f32_child_labels() {
    init();
    check_relabel::<f32>();
}

#[test]
fn tree_relabel_f64_child_labels() {
    init();
    check_relabel::<f64>();
}

#[test]
fn tree_build_tree_f32_end_to_end() {
    init();
    check_build_tree::<f32>();
}

#[test]
fn tree_build_tree_f64_end_to_end() {
    init();
    check_build_tree::<f64>();
}
