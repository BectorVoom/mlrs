//! `rng` — host-side random projection-matrix generation (PRIM-06).
//!
//! Empty Wave-0 stub. Plan **07-02** fills this with the host-side RNG matrix
//! generator that backs `GaussianRandomProjection` / `SparseRandomProjection`
//! (PROJ-01/02), promoting the `SplitMix64` PRNG out of `prims::kmeans`
//! VERBATIM (RESEARCH Pitfall 7 — do NOT alter the mix, or the stream changes
//! and `kmeanspp_test.rs` breaks) and adding:
//! - a Box–Muller Gaussian generator scaled `N(0, 1/n_components)`,
//! - the Achlioptas sparse generator (value `±sqrt((1/density)/n_components)`),
//! - an UNBIASED Fisher–Yates permutation via `SplitMix64::next_below`
//!   (rejection sampling, NOT `next_u64() % n` — RESEARCH Anti-Pattern "biased
//!   modulo").
//!
//! All hyperparameter guards (`density ∈ (0, 1]`, `n_components ≥ 1`) reject
//! BEFORE any allocation (ASVS V5). The seed is a caller-supplied `u64`
//! (documented, NEVER `OsRng`/`rand` — ASVS V6) so same-seed → identical matrix
//! (T-07-02). The generated matrix is host-built then single-uploaded via
//! `DeviceArray::from_host` (the D-10 memory-gate contract).
//!
//! Tests live in `crates/mlrs-backend/tests/rng_test.rs` (AGENTS.md §2 — no
//! in-source `#[cfg(test)] mod tests`).
