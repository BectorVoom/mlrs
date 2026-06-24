# Deferred items — Phase 15 (HDBSCAN)

Out-of-scope discoveries logged during execution (not fixed; not caused by the
current plan's changes).

## 15-04

- **`cargo clippy --features cpu` fails in `mlrs-kernels`** (pre-existing). The
  workspace-level `--features cpu` flag propagates to `mlrs-kernels`, which has no
  `cpu` feature, and the crate already emits 27 warnings + 1 error independent of
  any HDBSCAN code. `cargo build` / `cargo test --features cpu -p mlrs-algos` are
  clean (zero warnings on the new condense/stability/select code). Not fixed —
  unrelated crate, out of the 15-04 file set.

- **`labels_alpha` selection knob (Pitfall 2) deferred to 15-05.** The nested
  fixture's `labels_alpha` oracle was generated on the FEATURE path, whose alpha
  placement (`pair_distance /= alpha`, RAW core — Variant B) differs from the
  precomputed path's whole-matrix scaling (Variant A). sklearn itself partitions
  differently for the two paths (precomputed+α=0.5 → 2 clusters, feature+α=0.5 →
  3), so the knob is intrinsically a feature-metric (15-05) gate. Marked by the
  `#[ignore]`d `selection_knob_alpha_feature_path` test with an `un-ignore in
  15-05` marker; the other four knobs (eom/leaf/maxcluster/epsilon) pass via the
  precomputed path now.
