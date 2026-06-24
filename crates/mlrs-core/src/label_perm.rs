//! Clustering label best-permutation matching helper (FOUND-08).
//!
//! Cluster labels are permutation-invariant: a clustering that assigns
//! `[1,1,0,0]` is identical to one that assigns `[0,0,1,1]` — only the *names*
//! of the clusters differ. Comparing predicted labels to an oracle directly
//! would therefore spuriously fail on a relabeling. This module finds the
//! best mapping from predicted labels onto reference labels (greedy over a
//! confusion matrix) so a permuted-but-equal labeling matches, while a
//! genuinely different partition does not.
//!
//! Greedy assignment (repeatedly take the largest confusion-matrix cell) is the
//! standard, dependency-free clustering-evaluation approach and is exact for
//! the small label cardinalities used in oracle fixtures. A full Hungarian
//! solver can replace the greedy core later without changing this API.

use std::collections::HashMap;

/// Builds a confusion matrix `conf[pred][ref]` = count of points with predicted
/// label `pred` and reference label `ref`, plus the dense label vocabularies.
///
/// Labels may be any non-contiguous `i64` values; they are densified to
/// `0..k` indices internally.
fn confusion(pred: &[i64], reference: &[i64]) -> (Vec<Vec<u64>>, Vec<i64>, Vec<i64>) {
    let pred_labels = sorted_unique(pred);
    let ref_labels = sorted_unique(reference);
    let pred_idx: HashMap<i64, usize> = pred_labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i))
        .collect();
    let ref_idx: HashMap<i64, usize> = ref_labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i))
        .collect();

    let mut conf = vec![vec![0u64; ref_labels.len()]; pred_labels.len()];
    for (&p, &r) in pred.iter().zip(reference.iter()) {
        conf[pred_idx[&p]][ref_idx[&r]] += 1;
    }
    (conf, pred_labels, ref_labels)
}

fn sorted_unique(labels: &[i64]) -> Vec<i64> {
    let mut v: Vec<i64> = labels.to_vec();
    v.sort_unstable();
    v.dedup();
    v
}

/// Greedily maps each predicted label to a reference label by repeatedly
/// selecting the largest remaining confusion-matrix cell.
///
/// Returns `map[pred_label] = ref_label`. Predicted labels left unmatched
/// (when there are more predicted than reference labels) are absent from the
/// map.
pub fn best_mapping(pred: &[i64], reference: &[i64]) -> HashMap<i64, i64> {
    let (conf, pred_labels, ref_labels) = confusion(pred, reference);

    // Collect (count, pred_i, ref_j) and process in descending count order.
    let mut cells: Vec<(u64, usize, usize)> = Vec::new();
    for (i, row) in conf.iter().enumerate() {
        for (j, &c) in row.iter().enumerate() {
            if c > 0 {
                cells.push((c, i, j));
            }
        }
    }
    // Descending count; ties broken deterministically by (pred_i, ref_j).
    cells.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

    let mut map = HashMap::new();
    let mut used_ref = vec![false; ref_labels.len()];
    let mut used_pred = vec![false; pred_labels.len()];
    for (_c, i, j) in cells {
        if used_pred[i] || used_ref[j] {
            continue;
        }
        used_pred[i] = true;
        used_ref[j] = true;
        map.insert(pred_labels[i], ref_labels[j]);
    }
    map
}

/// Remaps `pred` through the best mapping onto the reference label space.
///
/// Unmapped predicted labels are passed through unchanged (they cannot match a
/// reference label, so they will register as mismatches downstream).
pub fn remap(pred: &[i64], reference: &[i64]) -> Vec<i64> {
    let map = best_mapping(pred, reference);
    pred.iter().map(|p| *map.get(p).unwrap_or(p)).collect()
}

/// Fraction of points whose best-permutation-remapped predicted label equals
/// the reference label. `1.0` means a perfect permutation match.
///
/// Panics if the slices differ in length.
pub fn best_match_accuracy(pred: &[i64], reference: &[i64]) -> f64 {
    assert_eq!(
        pred.len(),
        reference.len(),
        "best_match_accuracy: length mismatch pred={} reference={}",
        pred.len(),
        reference.len()
    );
    if pred.is_empty() {
        return 1.0;
    }
    let remapped = remap(pred, reference);
    let correct = remapped
        .iter()
        .zip(reference.iter())
        .filter(|(a, b)| a == b)
        .count();
    correct as f64 / pred.len() as f64
}

/// Like [`best_match_accuracy`], but with the noise sentinel `-1` pinned as a
/// fixed point: `-1` maps ONLY to `-1`, and `-1` is reachable ONLY from `-1`.
///
/// HDBSCAN labels are a permutation of sklearn's EXCEPT the `-1` noise class,
/// which is *not* a relabelable cluster id. A point sklearn calls noise that
/// mlrs calls a cluster member (or vice-versa) is a genuine failure, never a
/// relabeling — so [`best_match_accuracy`] (which would happily permute `-1`
/// into a cluster id to maximize the match) is wrong for the HDBSCAN gate.
///
/// The non-noise labels match up to permutation via the existing greedy
/// matcher: `-1` is filtered out of *both* label vocabularies before the
/// confusion matrix / greedy assignment is built, then `(-1 → -1)` is force-
/// inserted. A remapped `pred[i]` of `-1` therefore stays `-1`, scoring correct
/// only where `reference[i]` is also `-1`.
///
/// Returns `1.0` for a perfect (noise-pinned) permutation match; empty input is
/// a vacuous `1.0`. Panics if the slices differ in length.
pub fn best_match_accuracy_pinned_noise(pred: &[i64], reference: &[i64]) -> f64 {
    assert_eq!(
        pred.len(),
        reference.len(),
        "best_match_accuracy_pinned_noise: length mismatch pred={} reference={}",
        pred.len(),
        reference.len()
    );
    if pred.is_empty() {
        return 1.0;
    }

    // Filter -1 out of BOTH vocabularies so noise never enters the confusion
    // matrix or the greedy assignment — only genuine cluster ids are permuted.
    let pred_clusters: Vec<i64> = pred.iter().copied().filter(|&l| l != -1).collect();
    let ref_clusters: Vec<i64> = reference.iter().copied().filter(|&l| l != -1).collect();

    // Greedy cluster-id mapping over the noise-free slices; then pin -1 -> -1.
    let mut map = best_mapping(&pred_clusters, &ref_clusters);
    map.insert(-1, -1);

    // Remap each pred label (a -1 stays -1 via the forced pin; an unmapped
    // cluster id passes through and registers as a mismatch downstream).
    let correct = pred
        .iter()
        .zip(reference.iter())
        .filter(|(p, r)| map.get(p).unwrap_or(p) == *r)
        .count();
    correct as f64 / pred.len() as f64
}

/// Returns `true` when `pred` is a perfect permutation of `reference`
/// (best-match accuracy == 1.0).
pub fn is_perfect_match(pred: &[i64], reference: &[i64]) -> bool {
    (best_match_accuracy(pred, reference) - 1.0).abs() < f64::EPSILON
}
