//! `mlrs-py` — PyO3 binding layer for mlrs (cdylib).
//!
//! This crate owns the process-wide `#[global_allocator]` (FOUND-09): mimalloc
//! is wired exactly once in [`allocator`], the single cdylib artifact, and never
//! in any library crate. The allocator activation proof lives in the separate
//! test file `crates/mlrs-py/tests/allocator_test.rs` (AGENTS.md §2 — no
//! in-source test module).

// The `#[global_allocator]` definition. Source-only; its activation test is in
// `tests/allocator_test.rs` (FOUND-09: source/test separation).
mod allocator;

/// Boundary errors use `anyhow` (D-10); this alias exercises the dependency
/// until the full PyO3 surface lands in a later phase.
pub type BoundaryResult<T> = anyhow::Result<T>;
