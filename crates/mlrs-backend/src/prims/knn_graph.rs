//! KNN-graph primitive (PRIM-11, Phase 13) — the empty-but-registered shell
//! plan 13-03 fills with the `Metric` enum + `knn_graph` host orchestrator.
//!
//! The shared multi-metric DIRECTED k-nearest-neighbour graph primitive:
//! `knn_graph<F>(pool, x, (n, d), k, metric, include_self, p) -> (indices,
//! distances)` `(n, k)`. It composes the launch-proven `distance` + `top_k`
//! prims (and the new direct-distance + self-drop kernels in
//! `mlrs-kernels::distance`) into one device-resident, query-axis-tiled
//! primitive; UMAP (Phase 14) and HDBSCAN (Phase 15) consume it. No estimator
//! wrapper this phase (D-03).
//!
//! This file is the Wave-1 scaffold so `pub mod knn_graph;` compiles today
//! (mirrors the Phase-8/9 Wave-0 stub-registration precedent). Plan 13-03 fully
//! owns and rewrites the body: the `Metric` enum, the validate-before-launch
//! geometry guard (`n*d`, `k <= n-1` when `include_self=false`, `p >= 1`,
//! u32-overflow), the metric routing (Euclidean/Cosine via GEMM; the direct
//! kernels otherwise), `top_k(k+1)`, and the index-identity `self_drop` GATHER.
//! The `crates/mlrs-backend/tests/knn_graph_test.rs` oracle harness (plan 13-01)
//! is RED-by-design until `Metric` + `knn_graph` land here.
