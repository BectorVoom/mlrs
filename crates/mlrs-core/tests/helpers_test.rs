//! Tests for the comparison helpers (sign-flip, label-permutation) and the
//! named-`.npz` oracle loader (Task 2, FOUND-08 / D-01-03).
//!
//! Per AGENTS.md these live in `tests/`, never as a `#[cfg(test)] mod tests`
//! inside `src/`.

use std::path::PathBuf;

use mlrs_core::compare::assert_slice_close;
use mlrs_core::label_perm::{
    best_mapping, best_match_accuracy, best_match_accuracy_pinned_noise, is_perfect_match, remap,
};
use mlrs_core::oracle::load_npz;
use mlrs_core::sign_flip::{align_rows, align_sign, canonical_sign};
use mlrs_core::F32_TOL;

// --- sign_flip --------------------------------------------------------------

#[test]
fn sign_flipped_vector_passes_after_alignment() {
    // A PCA component and its negation must align to the same canonical sign,
    // so assert_close passes after alignment (FOUND-08).
    let component = [0.1, -0.9, 0.4];
    let flipped: Vec<f64> = component.iter().map(|x| -x).collect();

    let a = align_sign(&component);
    let b = align_sign(&flipped);
    assert_slice_close(&a, &b, &F32_TOL);
}

#[test]
fn genuinely_different_vector_still_fails_after_alignment() {
    let component = [0.1, -0.9, 0.4];
    let different = [0.1, -0.9, 0.5]; // last element genuinely differs

    let a = align_sign(&component);
    let b = align_sign(&different);
    // Aligned, but they are NOT equal -> assert_slice_close must panic.
    let result = std::panic::catch_unwind(|| assert_slice_close(&a, &b, &F32_TOL));
    assert!(
        result.is_err(),
        "genuinely different vectors must still fail"
    );
}

#[test]
fn canonical_sign_makes_largest_magnitude_positive() {
    // Largest magnitude is -0.9 (index 1) -> sign should be -1.0 to flip it +.
    assert_eq!(canonical_sign(&[0.1, -0.9, 0.4]), -1.0);
    // Already canonical (largest +0.9) -> +1.0 (no-op).
    assert_eq!(canonical_sign(&[0.1, 0.9, -0.4]), 1.0);
    // All-zero / empty -> +1.0 no-op.
    assert_eq!(canonical_sign(&[0.0, 0.0]), 1.0);
    assert_eq!(canonical_sign(&[]), 1.0);
}

#[test]
fn align_rows_canonicalizes_each_component_independently() {
    let rows = vec![vec![0.1, -0.9, 0.4], vec![-0.8, 0.2, 0.1]];
    let aligned = align_rows(&rows);
    // Row 0: flip (largest -0.9) -> [-0.1, 0.9, -0.4]; row 1: flip (largest
    // -0.8) -> [0.8, -0.2, -0.1].
    assert_slice_close(&aligned[0], &[-0.1, 0.9, -0.4], &F32_TOL);
    assert_slice_close(&aligned[1], &[0.8, -0.2, -0.1], &F32_TOL);
}

// --- label_perm -------------------------------------------------------------

#[test]
fn permuted_labeling_matches_perfectly() {
    // [1,1,0,0] is the same partition as [0,0,1,1] under a relabel.
    let pred = [1, 1, 0, 0];
    let reference = [0, 0, 1, 1];
    assert!(is_perfect_match(&pred, &reference));
    assert_eq!(best_match_accuracy(&pred, &reference), 1.0);

    // Mapping should swap the labels.
    let map = best_mapping(&pred, &reference);
    assert_eq!(map.get(&1), Some(&0));
    assert_eq!(map.get(&0), Some(&1));

    let remapped = remap(&pred, &reference);
    assert_eq!(remapped, reference.to_vec());
}

#[test]
fn different_labeling_does_not_match() {
    // [0,1,0,1] is a genuinely different partition from [0,0,1,1].
    let pred = [0, 1, 0, 1];
    let reference = [0, 0, 1, 1];
    assert!(!is_perfect_match(&pred, &reference));
    assert!(best_match_accuracy(&pred, &reference) < 1.0);
}

#[test]
fn identity_labeling_matches() {
    let pred = [0, 1, 2, 0, 1, 2];
    let reference = [0, 1, 2, 0, 1, 2];
    assert!(is_perfect_match(&pred, &reference));
}

#[test]
fn three_cluster_permutation_matches() {
    // Relabel 0->2, 1->0, 2->1.
    let pred = [2, 0, 1, 2, 0, 1];
    let reference = [0, 1, 2, 0, 1, 2];
    assert!(is_perfect_match(&pred, &reference));
}

// --- label_perm: -1-pinned noise (HDBS-02) ----------------------------------

#[test]
fn pinned_noise_permutes_clusters_but_fixes_noise() {
    // Cluster ids 0/1 are swapped (a valid relabeling); -1 stays -1.
    let pred = [0, 0, 1, 1, -1];
    let reference = [1, 1, 0, 0, -1];
    assert_eq!(best_match_accuracy_pinned_noise(&pred, &reference), 1.0);
}

#[test]
fn pinned_noise_counts_noise_vs_cluster_confusion_as_mismatch() {
    // pred index 3 is -1 (noise) where reference says cluster 1 — a genuine
    // confusion that must NOT be permuted away. Exactly one of five wrong.
    let pred = [0, 0, 1, -1, -1];
    let reference = [0, 0, 1, 1, -1];
    let acc = best_match_accuracy_pinned_noise(&pred, &reference);
    assert!(
        acc < 1.0,
        "noise/cluster confusion must score < 1.0, got {acc}"
    );
    assert!((acc - 0.8).abs() < 1e-12, "expected 4/5 correct, got {acc}");
}

#[test]
fn pinned_noise_all_noise_matches() {
    let pred = [-1, -1, -1];
    let reference = [-1, -1, -1];
    assert_eq!(best_match_accuracy_pinned_noise(&pred, &reference), 1.0);
}

#[test]
fn pinned_noise_pure_permutation_no_noise_present() {
    // No -1 anywhere: behaves like a plain permutation match.
    let pred = [0, 1, 2];
    let reference = [2, 0, 1];
    assert_eq!(best_match_accuracy_pinned_noise(&pred, &reference), 1.0);
}

#[test]
fn pinned_noise_empty_is_vacuous_match() {
    let pred: [i64; 0] = [];
    let reference: [i64; 0] = [];
    assert_eq!(best_match_accuracy_pinned_noise(&pred, &reference), 1.0);
}

// --- oracle::load_npz -------------------------------------------------------

fn fixture_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/oracle_case.npz");
    p
}

#[test]
fn load_npz_reads_named_arrays_for_f32_and_f64() {
    let case = load_npz(fixture_path()).expect("load oracle fixture");

    let mut names = case.names();
    names.sort();
    assert_eq!(names, vec!["a", "expected", "x", "y"]);

    // f32 views (x and y were stored as f32).
    let x = case.expect_f32("x");
    let y = case.expect_f32("y");
    assert_eq!(x, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(y, &[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
    assert_eq!(case.shape("x"), Some(&[6u64][..]));

    // f64 views (a and expected were stored as f64).
    let a = case.expect_f64("a");
    let expected = case.expect_f64("expected");
    assert_eq!(a, &[3.0]);
    // expected = a*x + y.
    let computed: Vec<f64> = (0..6).map(|i| 3.0 * x[i] as f64 + y[i] as f64).collect();
    assert_slice_close(expected, &computed, &F32_TOL);
}

#[test]
fn load_npz_exposes_both_precisions_per_array() {
    let case = load_npz(fixture_path()).expect("load oracle fixture");
    // Every array is decodable at both f32 and f64.
    let x_f64 = case.expect_f64("x");
    let expected_f32 = case.expect_f32("expected");
    assert_slice_close(x_f64, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &F32_TOL);
    assert_eq!(expected_f32.len(), 6);
}

#[test]
fn load_npz_missing_array_returns_none() {
    let case = load_npz(fixture_path()).expect("load oracle fixture");
    assert!(case.f64("does_not_exist").is_none());
    assert!(case.f32("does_not_exist").is_none());
}
