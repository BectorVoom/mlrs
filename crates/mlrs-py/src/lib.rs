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

// The shared binding primitives the `#[pyclass]` wrappers (Plan 03) consume:
// boundary error mapping, Arrow PyCapsule ingress, device→host egress, and the
// f64-on-incapable-backend capability guard. The `#[pymodule] _mlrs` + global
// pool + dtype-dispatch macro are added in this plan's Task 2.
pub mod capability;
pub mod egress;
pub mod errors;
pub mod ingress;

/// Boundary errors use `anyhow` (D-10); this alias exercises the dependency
/// until the full PyO3 surface lands in a later phase.
pub type BoundaryResult<T> = anyhow::Result<T>;
