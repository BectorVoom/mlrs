//! Tier-1 correctness WITNESS for the Phase-17 RandomForest feasibility spike
//! (TREE-01, D-07/D-09) — the real correctness gate for SC-2/SC-3 and abort
//! signal A5. Composes the Plan-02 kernels (histogram / split-find / relabel)
//! through the host `build_tree` loop on the Plan-01 injected-fixed-index
//! fixtures and proves a single tree reproduces `sklearn.tree.DecisionTree*`:
//! exact split STRUCTURE + `<=1e-5` (f64) leaf VALUES vs BOTH
//! `DecisionTreeClassifier(gini)` AND `DecisionTreeRegressor(squared_error)`.
//!
//! ## Why this is the correctness gate (not just "the kernels launched")
//!
//! Plan-02's probes proved each kernel value-correctly under cpu-MLIR on a toy
//! input. This witness proves the histogram/gain/partition MATH is correct by
//! reproducing sklearn EXACTLY. Indices are INJECTED (fixed bootstrap rows +
//! fixed feature subset, D-07) so there is NO RNG: if the tree diverges, the
//! MATH is wrong (not the seed). A `<=1e-5` match is therefore a real proof.
//!
//! ## Structural comparison is a LOCKSTEP traversal, not array-index equality
//!
//! sklearn lays nodes out depth-first (a parent's right child is NOT
//! `left_child + 1`; e.g. the clf fixture has root.left=1, root.right=8). The
//! mlrs `SparseTreeNode` contract lays children ADJACENT (`right = left + 1`,
//! D-02). So the witness CANNOT `assert_eq!` my node array against sklearn's
//! `children_left`. Instead it walks BOTH trees from the root in lockstep,
//! asserting at every node: same leaf/internal kind, same split FEATURE
//! (`colid` == sklearn `feature`, both indexing the injected subset), the same
//! samples routed left/right (DECISION-equivalence, Open Question 1), and on
//! leaves the dereferenced `value` offset matches sklearn `<=1e-5` (D-04).
//!
//! ## Threshold = DECISION-equivalence, never raw-float equality (Open Q 1 / A2)
//!
//! The witness bins each feature on host quantile edges (D-10) — here the
//! decision-exact midpoints between sorted-unique values, so every sklearn
//! split point is representable. A node's binned threshold equals the midpoint
//! of the GLOBALLY-consecutive uniques, which can differ from sklearn's
//! NODE-local midpoint while routing the node's samples IDENTICALLY. So the
//! witness gates the decision boundary (which samples go left/right), NOT the
//! raw `threshold` float. This is the resolution of Open Question 1.
//!
//! ## The f32 path is a COMPANION smoke check, NOT a bit-exact sklearn match
//!
//! f64 is the real correctness gate (CLAUDE.md: abs/rel `<=1e-5`). The generator
//! fits sklearn in float64 but commits `X` / `threshold` / `value` cast to the
//! fixture dtype, so the f32 witness reconstructs `x_fit` from the f32-ROUNDED
//! `X`, re-derives its unique values / bin layout, rebuilds the tree, and
//! decision-routes against the f32-rounded `threshold`. If f32 rounding ever
//! collapsed two previously-distinct feature values into one unique (changing
//! the bin layout) or flipped a sample sitting within ~1e-7 of a threshold, the
//! node-count or decision-equivalence assertions could fail on a tree that is
//! functionally correct. It passes on this well-separated random-normal data,
//! so the f32 run is a companion SMOKE check of the kernel plumbing — NOT an
//! independent bit-exact reproduction of sklearn's float64 fit (IN-01).
//!
//! ## Regressor split-feature ties are RNG-determined — gate the FUNCTION, not
//! the recorded feature (Pitfall 4 non-circularity)
//!
//! At minimal (2-sample) regression nodes EVERY feature achieves the identical
//! maximum variance reduction (any feature perfectly separates two points).
//! sklearn's `BestSplitter` breaks such ties with its internal `random_state`
//! feature shuffle, so the recorded split FEATURE at a tie node is RNG, NOT a
//! deterministic correctness signal. The injected-index recipe (D-07) removes
//! the BAGGING rng but NOT the splitter's internal tie-break rng. Conforming our
//! kernel's pick to sklearn's shuffled choice would be a CIRCULAR oracle
//! (Pitfall 4 — explicitly forbidden by the research). So the regressor witness
//! gates the tree as a FUNCTION: identical node/leaf counts, an identical
//! induced PARTITION of the training points into leaves, and per-point
//! predictions (dereferenced `value` offsets) `<=1e-5` vs sklearn. The
//! classifier has no such ties and keeps the strict per-node feature lockstep.
//!
//! ## Regression path — variance criterion, SAME three kernels
//!
//! Plan-02's `build_tree` computes binary-Gini gain (it has the histogram's
//! count + value-sum). `DecisionTreeRegressor(squared_error)` needs VARIANCE
//! reduction, which also needs the per-cell sum of SQUARES. Rather than mutate
//! the shared Plan-02 `build_tree` (Plan 04 depends on its signature), this
//! witness composes the SAME three public kernel wrappers in a local
//! `build_tree_variance` that launches the histogram a second time on `y^2` to
//! get the per-cell sum-of-squares and computes variance gain on the host. The
//! kernels under test (histogram / split-find / relabel) are IDENTICAL to the
//! classifier path; only the host gain formula differs. The leaf `value` field
//! carries the regression MEAN through the SAME offset semantics that the
//! classifier path uses for class probability — the D-09 multiclass-uniform
//! proof. (Recorded as a Rule-3 deviation in the plan SUMMARY.)
//!
//! Per AGENTS.md, tests live in `tests/`, never as `#[cfg(test)] mod tests`.

mod tree_spike;

use cubecl::prelude::{CubeElement, Float};
use mlrs_backend::capability;
use mlrs_core::{assert_slice_close, load_npz, OracleCase, F64_TOL};
use std::path::PathBuf;
use tree_spike::{
    build_tree, build_tree_with, from_f64, host_to_f64, launch_histogram, SparseTreeNode,
};

// sklearn hyperparameters baked into the Plan-01 generator (gen_oracle.py:
// DT_MAX_DEPTH = 4, sklearn default min_samples_split = 2).
const MAX_DEPTH: usize = 4;
const MIN_SPLIT: usize = 2;

/// Which sklearn estimator the fixture mirrors (selects gain criterion + the
/// sklearn `value` leaf-shape decode).
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// `DecisionTreeClassifier(gini)` — leaf `value` is the class distribution
    /// `[P(0), P(1)]`; the positive-class probability is `value[node*2 + 1]`.
    Clf,
    /// `DecisionTreeRegressor(squared_error)` — leaf `value` is the mean,
    /// `value[node]`.
    Reg,
}

/// Resolve a workspace-root-relative fixture path (matches `covariance_test.rs`).
fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// f64 skip-with-log gate (clone of the `self_drop_gather_test.rs` idiom).
/// Returns `true` when the witness should early-return (f64 unsupported on this
/// adapter, e.g. rocm). f32 always runs.
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

fn dtype_tag<F>() -> &'static str {
    match std::mem::size_of::<F>() {
        4 => "f32",
        8 => "f64",
        _ => unreachable!("tree witness is f32/f64 only"),
    }
}

/// Reconstruct the exact matrix sklearn was fit on: `X[bootstrap_idx][:,
/// feature_idx]` plus `y[bootstrap_idx]` (D-07). Returns `(x_fit, y_fit, n,
/// nf)` with `x_fit` flat row-major `[r*nf + j]`.
fn reconstruct(case: &OracleCase) -> (Vec<f64>, Vec<f64>, usize, usize) {
    let xshape = case.shape("X").expect("fixture has X");
    let ncol = xshape[1] as usize;
    let x = case.expect_f64("X");
    let yfull = case.expect_f64("y");
    let boot = case.expect_f64("bootstrap_idx"); // integer-valued f64
    let feat = case.expect_f64("feature_idx");
    let n = boot.len();
    let nf = feat.len();
    let mut x_fit = vec![0.0f64; n * nf];
    let mut y_fit = vec![0.0f64; n];
    for r in 0..n {
        let br = boot[r] as usize;
        for j in 0..nf {
            let fc = feat[j] as usize;
            x_fit[r * nf + j] = x[br * ncol + fc];
        }
        y_fit[r] = yfull[br];
    }
    (x_fit, y_fit, n, nf)
}

/// Decision-exact host binning (D-10): per feature, bin edges are the midpoints
/// between sorted-unique values, so every sklearn midpoint split point is
/// representable as a bin boundary. Returns `(binned, bin_edges, n_bins)` where
/// `binned[r*nf + j]` is the rank of the value among feature `j`'s uniques and
/// `bin_edges[j]` has `n_bins-1` entries (padded with `+inf` for unused bins, so
/// a split there has an empty child → zero gain → never chosen).
fn make_bins(x_fit: &[f64], n: usize, nf: usize) -> (Vec<u32>, Vec<Vec<f64>>, usize) {
    let mut uniq: Vec<Vec<f64>> = Vec::with_capacity(nf);
    for j in 0..nf {
        let mut vals: Vec<f64> = (0..n).map(|r| x_fit[r * nf + j]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in fixture features"));
        vals.dedup(); // exact dedup: bootstrap repeats are byte-identical floats
        uniq.push(vals);
    }
    let n_bins = uniq.iter().map(|u| u.len()).max().unwrap_or(2).max(2);
    let mut bin_edges = vec![vec![f64::INFINITY; n_bins - 1]; nf];
    for j in 0..nf {
        let u = &uniq[j];
        for b in 0..u.len().saturating_sub(1) {
            bin_edges[j][b] = (u[b] + u[b + 1]) / 2.0;
        }
    }
    let mut binned = vec![0u32; n * nf];
    for r in 0..n {
        for j in 0..nf {
            let v = x_fit[r * nf + j];
            let rank = uniq[j]
                .iter()
                .position(|&uv| uv == v)
                .expect("value is one of its own uniques");
            binned[r * nf + j] = rank as u32;
        }
    }
    (binned, bin_edges, n_bins)
}

/// Local REGRESSION builder: drives the SAME shared `build_tree_with` frontier
/// skeleton as the classifier (histogram → split-find → relabel, adjacency D-02,
/// leaf sentinel D-03/D-04), supplying a VARIANCE-reduction gain (squared_error)
/// and a regression-MEAN leaf. The gain closure launches the histogram a second
/// time on `y^2` (per-cell sum of squares) so it can form `var = E[y^2] - E[y]^2`
/// per child. Factoring the skeleton into `build_tree_with` removes the ~100-line
/// near-verbatim copy of `build_tree`'s loop (WR-02) while leaving the Plan-04
/// `build_tree` signature intact.
fn build_tree_variance<F>(
    binned: &[u32],
    y: &[F],
    bin_edges: &[Vec<f64>],
    n_samples: usize,
    n_feat: usize,
    n_bins: usize,
    max_depth: usize,
    min_samples: usize,
) -> (Vec<SparseTreeNode<F>>, Vec<f64>)
where
    F: Float + CubeElement + bytemuck::Pod,
{
    assert!(n_bins >= 2, "variance build needs n_bins >= 2");
    let n_cand = n_feat * (n_bins - 1);
    let cell = move |nid: usize, f: usize, b: usize| (nid * n_feat + f) * n_bins + b;
    let var = |sq: f64, sm: f64, c: f64| -> f64 {
        if c <= 0.0 {
            0.0
        } else {
            let m = sm / c;
            (sq / c - m * m).max(0.0)
        }
    };

    // Criterion-specific level math: variance-reduction gain + per-node purity
    // (a node is pure when its target variance is ~0). Needs the per-cell sum of
    // SQUARES, so this closure launches the histogram a SECOND time on y^2 (the
    // count + value-sum on y are provided by the shared driver).
    let variance_gain = move |node_id: &[u32],
                              counts: &[F],
                              vsums: &[F],
                              frontier: &[u32],
                              n_nodes_total: usize|
          -> (Vec<f64>, Vec<bool>) {
        let ysq: Vec<F> = y
            .iter()
            .map(|&v| {
                let h = host_to_f64(v);
                from_f64::<F>(h * h)
            })
            .collect();
        let (_c2, vsqs) =
            launch_histogram::<F>(node_id, binned, &ysq, n_samples, n_feat, n_nodes_total, n_bins);

        let mut gain_h = vec![0.0f64; n_nodes_total * n_cand];
        let mut pure = vec![false; n_nodes_total];
        for &nid_u in frontier {
            let nid = nid_u as usize;
            // Purity from feature 0: a zero-variance node cannot reduce variance.
            let (mut t0, mut s0, mut q0) = (0.0f64, 0.0f64, 0.0f64);
            for b in 0..n_bins {
                t0 += host_to_f64(counts[cell(nid, 0, b)]);
                s0 += host_to_f64(vsums[cell(nid, 0, b)]);
                q0 += host_to_f64(vsqs[cell(nid, 0, b)]);
            }
            pure[nid] = var(q0, s0, t0) <= 1e-12;
            for f in 0..n_feat {
                let (mut tot, mut sm, mut sq) = (0.0f64, 0.0f64, 0.0f64);
                for b in 0..n_bins {
                    tot += host_to_f64(counts[cell(nid, f, b)]);
                    sm += host_to_f64(vsums[cell(nid, f, b)]);
                    sq += host_to_f64(vsqs[cell(nid, f, b)]);
                }
                let parent = var(sq, sm, tot);
                let (mut lc, mut lp, mut lsq) = (0.0f64, 0.0f64, 0.0f64);
                for b in 0..(n_bins - 1) {
                    lc += host_to_f64(counts[cell(nid, f, b)]);
                    lp += host_to_f64(vsums[cell(nid, f, b)]);
                    lsq += host_to_f64(vsqs[cell(nid, f, b)]);
                    let rc = tot - lc;
                    let rp = sm - lp;
                    let rsq = sq - lsq;
                    let g = if lc > 0.0 && rc > 0.0 {
                        parent - (lc / tot) * var(lsq, lp, lc) - (rc / tot) * var(rsq, rp, rc)
                    } else {
                        0.0
                    };
                    gain_h[nid * n_cand + (f * (n_bins - 1) + b)] = g;
                }
            }
        }
        (gain_h, pure)
    };

    // Regression leaf value = mean of y over the node (regression-mean leaf, D-09).
    let leaf_mean = |sum_y: f64, tot: f64| if tot > 0.0 { sum_y / tot } else { 0.0 };

    build_tree_with::<F, _, _>(
        binned, y, bin_edges, n_samples, n_feat, n_bins, max_depth, min_samples, variance_gain,
        leaf_mean,
    )
}

/// Bundle of the sklearn `tree_` reference arrays for the lockstep walk.
struct SkTree<'a> {
    feature: &'a [f64],
    threshold: &'a [f64],
    children_left: &'a [f64],
    children_right: &'a [f64],
    value: &'a [f64],
    kind: Kind,
}

impl SkTree<'_> {
    fn is_leaf(&self, i: usize) -> bool {
        self.feature[i] < 0.0
    }
    /// sklearn's leaf value, decoded by leaf shape (D-09): classifier =>
    /// P(class 1) = `value[i*2+1]`; regressor => mean = `value[i]`.
    fn leaf_value(&self, i: usize) -> f64 {
        match self.kind {
            Kind::Clf => self.value[i * 2 + 1],
            Kind::Reg => self.value[i],
        }
    }
}

/// Lockstep structural + decision + leaf-value comparison of my tree (rooted at
/// `my`) against sklearn's (rooted at `sk`) over the sample `rows` that reach
/// this node. Robust to the differing node layouts.
#[allow(clippy::too_many_arguments)]
fn compare_rec<F>(
    my: i32,
    sk: usize,
    rows: &[usize],
    nodes: &[SparseTreeNode<F>],
    leaf_buf: &[f64],
    sk_tree: &SkTree,
    x_fit: &[f64],
    nf: usize,
) where
    F: bytemuck::Pod + Copy,
{
    let node = nodes[my as usize];
    if sk_tree.is_leaf(sk) {
        assert_eq!(
            node.colid, -1,
            "sklearn node {sk} is a leaf but my node {my} is internal (colid {})",
            node.colid
        );
        // D-04: dereference `value` as an offset into the shared leaf buffer
        // (NOT a scalar) and compare THAT to sklearn's leaf output (<=1e-5).
        let mine = leaf_buf[node.value as usize];
        let theirs = sk_tree.leaf_value(sk);
        assert_slice_close(&[mine], &[theirs], &F64_TOL);
        return;
    }

    // Internal node: same split feature (D-03 colid>=0 + subset-index match).
    assert!(
        node.colid >= 0,
        "sklearn node {sk} is internal but my node {my} is a leaf"
    );
    let sk_feat = sk_tree.feature[sk] as i32;
    assert_eq!(
        node.colid, sk_feat,
        "split feature mismatch at sklearn node {sk}: mine {} sklearn {sk_feat}",
        node.colid
    );

    // DECISION-equivalence (Open Question 1): route this node's samples by my
    // binned midpoint AND by sklearn's exact threshold; assert identical
    // partitions (NOT raw-threshold equality).
    let f = node.colid as usize;
    let my_thr = host_to_f64(node.threshold);
    let sk_thr = sk_tree.threshold[sk];
    let (mut my_l, mut my_r) = (Vec::new(), Vec::new());
    let (mut sk_l, mut sk_r) = (Vec::new(), Vec::new());
    for &r in rows {
        let v = x_fit[r * nf + f];
        if v <= my_thr {
            my_l.push(r);
        } else {
            my_r.push(r);
        }
        if v <= sk_thr {
            sk_l.push(r);
        } else {
            sk_r.push(r);
        }
    }
    assert_eq!(
        my_l, sk_l,
        "decision-equivalence (left set) mismatch at sklearn node {sk}"
    );
    assert_eq!(
        my_r, sk_r,
        "decision-equivalence (right set) mismatch at sklearn node {sk}"
    );

    // Recurse: my right child is implicit left+1 (D-02); sklearn's is explicit.
    compare_rec(
        node.left_child,
        sk_tree.children_left[sk] as usize,
        &sk_l,
        nodes,
        leaf_buf,
        sk_tree,
        x_fit,
        nf,
    );
    compare_rec(
        node.left_child + 1,
        sk_tree.children_right[sk] as usize,
        &sk_r,
        nodes,
        leaf_buf,
        sk_tree,
        x_fit,
        nf,
    );
}

/// Route a single training row to the leaf node index it reaches in MY tree
/// (follow `colid`/`threshold`; right child is implicit `left_child + 1`, D-02).
fn my_leaf<F>(nodes: &[SparseTreeNode<F>], x_fit: &[f64], nf: usize, r: usize) -> usize
where
    F: bytemuck::Pod + Copy,
{
    let mut i = 0usize;
    while nodes[i].colid >= 0 {
        let f = nodes[i].colid as usize;
        let thr = host_to_f64(nodes[i].threshold);
        i = if x_fit[r * nf + f] <= thr {
            nodes[i].left_child as usize
        } else {
            nodes[i].left_child as usize + 1
        };
    }
    i
}

/// Route a single training row to the leaf node index it reaches in sklearn's
/// tree (follow `feature`/`threshold`/`children_left`/`children_right`).
fn sk_leaf(sk_tree: &SkTree, x_fit: &[f64], nf: usize, r: usize) -> usize {
    let mut i = 0usize;
    while !sk_tree.is_leaf(i) {
        let f = sk_tree.feature[i] as usize;
        let thr = sk_tree.threshold[i];
        i = if x_fit[r * nf + f] <= thr {
            sk_tree.children_left[i] as usize
        } else {
            sk_tree.children_right[i] as usize
        };
    }
    i
}

/// Canonicalize a per-row leaf-id labelling into the induced PARTITION: a sorted
/// list of sorted row-groups, independent of leaf numbering / orientation.
fn partition(leaf_of: &[usize]) -> Vec<Vec<usize>> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (r, &lid) in leaf_of.iter().enumerate() {
        groups.entry(lid).or_default().push(r);
    }
    let mut parts: Vec<Vec<usize>> = groups.into_values().collect();
    parts.sort();
    parts
}

/// Orientation/feature-INDEPENDENT equivalence used for the regressor (where
/// split-feature ties are sklearn-RNG-determined, see module doc): prove the
/// trees are the SAME FUNCTION. Asserts (1) identical induced partition of the
/// training rows into leaves, and (2) per-row predictions match `<=1e-5` after
/// dereferencing MY `value` offset into the leaf buffer (the D-04/D-09 proof —
/// the one `value` field carries the regression mean).
fn assert_function_equiv<F>(
    nodes: &[SparseTreeNode<F>],
    leaf_buf: &[f64],
    sk_tree: &SkTree,
    x_fit: &[f64],
    n: usize,
    nf: usize,
) where
    F: bytemuck::Pod + Copy,
{
    let my_leaf_of: Vec<usize> = (0..n).map(|r| my_leaf(nodes, x_fit, nf, r)).collect();
    let sk_leaf_of: Vec<usize> = (0..n).map(|r| sk_leaf(sk_tree, x_fit, nf, r)).collect();

    // (1) Identical induced partition (same grouping of points into leaves) —
    // catches a genuine different-partition divergence; tolerates only the
    // RNG-determined feature/orientation relabelling at gain-tie nodes.
    assert_eq!(
        partition(&my_leaf_of),
        partition(&sk_leaf_of),
        "induced leaf partition mismatch (a real structural divergence, not a tie relabel)"
    );

    // (2) Per-row predictions match (dereference my `value` offset → mean).
    let mine: Vec<f64> = my_leaf_of
        .iter()
        .map(|&i| leaf_buf[nodes[i].value as usize])
        .collect();
    let theirs: Vec<f64> = sk_leaf_of.iter().map(|&i| sk_tree.leaf_value(i)).collect();
    assert_slice_close(&mine, &theirs, &F64_TOL);
}

/// Validate the SparseTreeNode contract over every node (D-02/D-03/D-04).
fn assert_contract<F>(nodes: &[SparseTreeNode<F>], leaf_buf: &[f64]) {
    for (i, n) in nodes.iter().enumerate() {
        if n.colid >= 0 {
            // Internal: left_child valid and right = left+1 is in range (D-02).
            assert!(
                n.left_child >= 0,
                "internal node {i} must have left_child >= 0 (D-02)"
            );
            let right = n.left_child as usize + 1;
            assert!(
                right < nodes.len(),
                "internal node {i} right child {right} out of range (D-02 adjacency)"
            );
        } else {
            // Leaf: sentinel colid==-1, left_child==-1, value is a valid offset.
            assert_eq!(n.colid, -1, "leaf node {i} sentinel must be colid==-1 (D-03)");
            assert_eq!(
                n.left_child, -1,
                "leaf node {i} must have left_child == -1 (D-03)"
            );
            assert!(
                n.value >= 0 && (n.value as usize) < leaf_buf.len(),
                "leaf node {i} value {} must be a valid offset into the leaf buffer (D-04)",
                n.value
            );
        }
    }
}

/// Build the tree for `kind`/`adversarial` and run the full witness assertions
/// (contract + node/leaf counts + lockstep structure/decision/leaf-values).
/// Returns the built `(nodes, leaf_buffer)` for any extra per-test asserts, or
/// `None` when the f64 gate skipped this adapter.
fn run_witness<F>(kind: Kind, adversarial: bool) -> Option<(Vec<SparseTreeNode<F>>, Vec<f64>)>
where
    F: Float + CubeElement + bytemuck::Pod,
{
    let label = match (kind, adversarial) {
        (Kind::Clf, false) => "tree_witness_clf",
        (Kind::Reg, false) => "tree_witness_reg",
        (Kind::Clf, true) => "tree_witness_clf_adv",
        (Kind::Reg, true) => "tree_witness_reg_adv",
    };
    if gate_and_log::<F>(label) {
        return None;
    }

    let suffix = match (kind, adversarial) {
        (Kind::Clf, false) => "clf",
        (Kind::Reg, false) => "reg",
        (Kind::Clf, true) => "clf_adv",
        (Kind::Reg, true) => "reg_adv",
    };
    let name = format!("tree_dt_{suffix}_{}_seed42.npz", dtype_tag::<F>());
    let case = load_npz(fixture(&name)).unwrap_or_else(|e| panic!("load {name}: {e}"));

    let (x_fit, y_fit, n, nf) = reconstruct(&case);
    let (binned, bin_edges, n_bins) = make_bins(&x_fit, n, nf);
    let y_f: Vec<F> = y_fit.iter().map(|&v| from_f64::<F>(v)).collect();

    // Build via the SAME three kernels: classifier through the shared Plan-02
    // `build_tree` (gini); regressor through the local variance composition.
    let (nodes, leaf_buf) = match kind {
        Kind::Clf => build_tree::<F>(
            &binned, &y_f, &bin_edges, n, nf, n_bins, MAX_DEPTH, MIN_SPLIT,
        ),
        Kind::Reg => build_tree_variance::<F>(
            &binned, &y_f, &bin_edges, n, nf, n_bins, MAX_DEPTH, MIN_SPLIT,
        ),
    };

    // Contract (D-02/D-03/D-04).
    assert_contract(&nodes, &leaf_buf);

    // Exact node + leaf counts vs sklearn — no missing / extra nodes.
    let sk_feature = case.expect_f64("feature");
    let sk_nodes = sk_feature.len();
    let sk_leaves = sk_feature.iter().filter(|&&f| f < 0.0).count();
    let my_leaves = nodes.iter().filter(|n| n.colid < 0).count();
    assert_eq!(
        nodes.len(),
        sk_nodes,
        "node count mismatch: mine {} sklearn {sk_nodes}",
        nodes.len()
    );
    assert_eq!(
        my_leaves, sk_leaves,
        "leaf count mismatch: mine {my_leaves} sklearn {sk_leaves}"
    );

    // Lockstep structure + decision-equivalence + leaf values (<=1e-5).
    let sk_tree = SkTree {
        feature: sk_feature,
        threshold: case.expect_f64("threshold"),
        children_left: case.expect_f64("children_left"),
        children_right: case.expect_f64("children_right"),
        value: case.expect_f64("value"),
        kind,
    };
    match (kind, adversarial) {
        // Standard classifier: no gain-ties → strict per-node lockstep (exact
        // split FEATURE + decision-equivalence + leaf values), the "structure
        // EXACT" gate the plan asks for.
        (Kind::Clf, false) => {
            let all_rows: Vec<usize> = (0..n).collect();
            compare_rec(0, 0, &all_rows, &nodes, &leaf_buf, &sk_tree, &x_fit, nf);
            println!(
                "{label} [{}]: {} nodes / {my_leaves} leaves, n_bins={n_bins} — \
                 structure EXACT (per-node feature), decision-equivalent, leaf \
                 values <=1e-5 vs sklearn ✓",
                dtype_tag::<F>(),
                nodes.len()
            );
        }
        // Regressor (all) + ADVERSARIAL classifier: the split FEATURE at a
        // gain-tie node is sklearn-RNG-determined. The adversarial clf root is,
        // by construction, an EXACT gain tie between two identical columns, so
        // which column sklearn records there is an RNG outcome of its internal
        // feature shuffle. Asserting `colid == sklearn.feature` at that node
        // would be the CIRCULAR oracle the module doc forbids (WR-01). Gate the
        // tree as a FUNCTION instead (induced partition + per-row predictions),
        // which is feature-index-independent. The kernel's OWN lowest-index
        // tie-break is proven separately by `check_adversarial`
        // (`nodes[0].colid == 0`), so dropping the sklearn-feature equality at
        // the tie node loses no real coverage.
        (Kind::Clf, true) | (Kind::Reg, _) => {
            assert_function_equiv(&nodes, &leaf_buf, &sk_tree, &x_fit, n, nf);
            println!(
                "{label} [{}]: {} nodes / {my_leaves} leaves, n_bins={n_bins} — \
                 counts EXACT, induced partition identical, leaf predictions \
                 <=1e-5 vs sklearn (split-feature ties RNG-gated, \
                 function-equivalence) ✓",
                dtype_tag::<F>(),
                nodes.len()
            );
        }
    }
    Some((nodes, leaf_buf))
}

fn init() {
    let _ = env_logger::builder().is_test(true).try_init();
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 1 — Tier-1 clf(gini) + reg(squared_error) VALUE witness (SC-2/SC-3/A5).
// f64 is the cpu correctness gate (skips-with-log on rocm); f32 companion runs.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tree_witness_clf_f64_matches_sklearn() {
    init();
    run_witness::<f64>(Kind::Clf, false);
}

#[test]
fn tree_witness_clf_f32_matches_sklearn() {
    init();
    run_witness::<f32>(Kind::Clf, false);
}

#[test]
fn tree_witness_reg_f64_matches_sklearn() {
    init();
    run_witness::<f64>(Kind::Reg, false);
}

#[test]
fn tree_witness_reg_f32_matches_sklearn() {
    init();
    run_witness::<f32>(Kind::Reg, false);
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 2 — Adversarial fixture witness: forced-pure-leaf + gain TIE. This is the
// explicit 002-B silent-cross-loop-miscompile backstop. A boundary miscompile
// that the happy-path clf/reg witness above would ship must FAIL here. The tie
// is resolved against the INDEPENDENT generator-encoded rule (lowest feature
// index — verified in gen_oracle.py via pure-numpy impurity, never conformed to
// the kernel's pick — Phase-13 CR-01/CR-02), so the gate is NON-CIRCULAR.
// ─────────────────────────────────────────────────────────────────────────────

/// Shared adversarial assertions: the lockstep witness already proves the tree
/// reproduces sklearn; here we additionally assert the two boundary properties
/// explicitly — (1) the gain-TIE root resolves to the lowest feature index
/// (feature 0, sklearn's independently-verified canonical pick), and (2) both
/// children are forced-PURE leaves whose dereferenced values match sklearn.
fn check_adversarial<F>(kind: Kind)
where
    F: Float + CubeElement + bytemuck::Pod,
{
    let Some((nodes, leaf_buf)) = run_witness::<F>(kind, true) else {
        return; // f64 skipped on this adapter
    };

    // (1) Gain-TIE backstop: two identical feature columns tie exactly; the
    // argmax tie-break (lowest feature index) must pick feature 0 — the pick the
    // generator verified INDEPENDENTLY (not conformed to this kernel).
    assert_eq!(
        nodes[0].colid, 0,
        "adversarial gain TIE must resolve to the lowest feature index (0), got colid {}",
        nodes[0].colid
    );

    // (2) Forced-PURE-leaf backstop: the root's two children are adjacent
    // (D-02) and BOTH are leaves (colid == -1) with dereferenced values.
    let left = nodes[0].left_child as usize;
    let right = left + 1;
    assert_eq!(
        nodes[left].colid, -1,
        "adversarial left child must be a forced-pure leaf (colid == -1)"
    );
    assert_eq!(
        nodes[right].colid, -1,
        "adversarial right child must be a forced-pure leaf (colid == -1)"
    );

    // The pure-leaf values: classifier => P(class1) in {0, 1}; regressor =>
    // constant region means {1.0, 5.0}. Exact-match the dereferenced offsets.
    let lv = leaf_buf[nodes[left].value as usize];
    let rv = leaf_buf[nodes[right].value as usize];
    let (want_l, want_r) = match kind {
        Kind::Clf => (0.0, 1.0), // left region x=0 => class 0; right x=1 => class 1
        Kind::Reg => (1.0, 5.0), // left region y==1.0; right region y==5.0
    };
    assert_slice_close(&[lv, rv], &[want_l, want_r], &F64_TOL);

    println!(
        "adversarial [{}/{}]: gain-TIE → feature 0 (independent rule), forced-pure \
         leaves value-match sklearn — 002-B silent-miscompile backstop GREEN ✓",
        match kind {
            Kind::Clf => "clf",
            Kind::Reg => "reg",
        },
        dtype_tag::<F>()
    );
}

#[test]
fn tree_witness_adversarial_clf_f64_backstop() {
    init();
    check_adversarial::<f64>(Kind::Clf);
}

#[test]
fn tree_witness_adversarial_reg_f64_backstop() {
    init();
    check_adversarial::<f64>(Kind::Reg);
}

#[test]
fn tree_witness_adversarial_clf_f32_backstop() {
    init();
    check_adversarial::<f32>(Kind::Clf);
}

#[test]
fn tree_witness_adversarial_reg_f32_backstop() {
    init();
    check_adversarial::<f32>(Kind::Reg);
}
