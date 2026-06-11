//! `mlrs-py` — PyO3 binding layer for mlrs (cdylib).
//!
//! This crate owns the process-wide `#[global_allocator]` (FOUND-09); it must
//! be defined exactly once and only here (never in a library crate). The
//! mimalloc wiring and the PyO3 surface are completed in Plan 05; Wave 0 keeps
//! a minimal compiling stub.
//!
//! `mimalloc` is referenced here so the dependency is exercised; the actual
//! `#[global_allocator]` activation + its `tests/allocator_test.rs` proof land
//! in Plan 05.

/// Placeholder re-export so the `mimalloc` dependency is wired (not yet the
/// global allocator — that activation is Plan 05).
pub use mimalloc::MiMalloc as _MiMalloc;

/// Boundary errors use `anyhow` (D-10); this alias exercises the dependency
/// until the real PyO3 surface lands in Plan 05.
pub type BoundaryResult<T> = anyhow::Result<T>;
