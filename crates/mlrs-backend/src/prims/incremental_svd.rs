//! `incremental_svd` — host-side incremental (batched) SVD merge (PRIM-07).
//!
//! Empty Wave-0 stub. Plan **07-03** fills this with the stacked-matrix SVD
//! merge that backs `IncrementalPCA` (DECOMP-03). The merge is host glue over
//! the Phase-3 thin `svd` primitive (RESEARCH Pattern 1 "Stacked-matrix merge"):
//! given the running basis `(singular_values_, components_, mean_,
//! n_samples_seen_)` and a new centered batch, it builds the stacked host
//! matrix —
//!   rows `0..k`   = `singular_values_[i] * components_[i, :]`,
//!   rows `k..k+b` = the centered batch,
//!   row  `k+b`    = the mean-correction term —
//! uploads it ONCE via `DeviceArray::from_host`, re-runs `svd`, then applies
//! `sign_flip::align_rows` to the resulting `Vᵀ` rows (== sklearn
//! `svd_flip(u_based_decision=False)`) after EVERY batch. The first batch is the
//! plain SVD of the centered batch alone (RESEARCH Pitfall 3).
//!
//! All combine math accumulates in `f64` via the `host_to_f64`/`f64_to_host`
//! bit-cast helpers regardless of `F` (RESEARCH Pitfall 4). The stacked matrix
//! MUST satisfy the Phase-3 SVD caps `(k + batch_size + 1) ≤ MAX_ROWS` and
//! `n_features ≤ MAX_COLS`, validated as a typed `PrimError::ShapeMismatch`
//! BEFORE any launch (ASVS V5). SVD scratch is released back to the pool to keep
//! the D-10 memory gate green.
//!
//! Tests live in `crates/mlrs-backend/tests/incremental_svd_test.rs` (AGENTS.md
//! §2 — no in-source `#[cfg(test)] mod tests`).
