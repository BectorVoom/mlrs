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

- **`labels_alpha` selection knob (Pitfall 2) deferred to 15-05.** RESOLVED in
  15-05. The `selection_knob_alpha_feature_path` test is un-ignored and green: the
  feature Euclidean (Variant-B) path with `alpha=0.5` (`pair_distance /= alpha`,
  RAW core) reproduces the feature-path `labels_alpha` oracle exactly under the
  pinned-noise gate.

## 15-05

- **`cargo test --features cpu -p mlrs-kernels mutual_reachability` (the plan's
  Task-1 verify command) cannot run as written** — `mlrs-kernels` carries NO
  backend feature (Criterion 1: it stays runtime-free), so `--features cpu` errors
  with "the package 'mlrs-kernels' does not contain this feature: cpu", and the
  crate has no runtime to LAUNCH a kernel anyway. The MR kernel's VALUE gate (incl.
  the R-9 duplicate-point row) therefore lives in `mlrs-backend`
  (`tests/mutual_reachability_test.rs`, run via
  `cargo test --features cpu -p mlrs-backend --test mutual_reachability_test`),
  where the concrete `ActiveRuntime` exists — mirroring how the `distance.rs`
  kernels are value-tested through the `knn_graph` prim, not in `mlrs-kernels`
  itself. The kernel still compiles under `cargo build -p mlrs-kernels`.

- **`cargo clippy --features cpu` still fails in `mlrs-kernels`** (pre-existing,
  unchanged from 15-04): `elementwise.rs:282` trips `approx_constant`
  (`FRAC_PI_2`) + 28 warnings, independent of HDBSCAN. The new MR kernel itself is
  clean save the INTENTIONAL `collapsible_if` warning (the nested
  `if i<rows_x { if j<rows_y {…} }` guard is the mandated cpu-MLIR shape, matching
  `distance.rs`). `cargo build --features cpu -p mlrs-algos`/`-p mlrs-backend` are
  warning-free on all new code. Not fixed — unrelated crate, out of the file set.
