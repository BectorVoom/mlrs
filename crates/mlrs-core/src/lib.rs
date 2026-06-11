//! `mlrs-core` — host-side foundation for mlrs.
//!
//! This crate carries NO backend feature flags. It holds the numerical
//! comparison harness (`assert_close`), tolerance policy, sign-flip and
//! label-permutation helpers, the scikit-learn oracle (`.npz`) loader, and
//! the typed error enums shared across the workspace.
//!
//! Most module bodies are filled in Plan 02; Wave 0 (Plan 01) only stands up
//! the compiling skeleton so downstream plans edit their own module file.

pub mod compare;
pub mod error;
pub mod label_perm;
pub mod oracle;
pub mod sign_flip;
pub mod tolerance;

// Re-export the most-used symbols so downstream crates/tests can write
// `use mlrs_core::{assert_close, F32_TOL, BridgeError};` directly.
pub use compare::{assert_close, assert_slice_close, is_close, NEAR_ZERO_FLOOR};
pub use error::{BridgeError, PrimError};
pub use label_perm::{best_match_accuracy, best_mapping, is_perfect_match, remap};
pub use oracle::{load_npz, load_npz_reader, OracleCase};
pub use sign_flip::{align_rows, align_sign, align_sign_in_place, canonical_sign};
pub use tolerance::{Tolerance, F32_TOL, F64_TOL};
