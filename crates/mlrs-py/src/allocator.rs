//! Process-wide global allocator wiring for mlrs (FOUND-09).
//!
//! mlrs uses [`mimalloc`](https://github.com/microsoft/mimalloc) as its
//! `#[global_allocator]`. The allocator MUST be defined **exactly once** for the
//! whole process and ONLY in this final cdylib artifact — never in a library
//! crate (`mlrs-core` / `mlrs-kernels` / `mlrs-backend` / `mlrs-algos`). A
//! library crate setting `#[global_allocator]` would force that choice on every
//! downstream consumer and break build integrity if two such crates were linked
//! (T-05-02). Keeping the definition here, in the binding cdylib, is the single
//! source of truth (RESEARCH Pattern 7 / A8).
//!
//! Per AGENTS.md §2 this file is SOURCE only — its activation proof lives in the
//! separate test file `crates/mlrs-py/tests/allocator_test.rs`, never as an
//! in-source `#[cfg(test)]` test module.

use mimalloc::MiMalloc;

/// The process global allocator. Defined exactly once, here in the cdylib.
///
/// Every heap allocation in the `mlrs-py` extension module (and anything linked
/// into it) is served by mimalloc's sharded, thread-local free lists.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
