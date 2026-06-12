//! `lbfgs` — stable softmax loss + gradient kernel (LINEAR-05). **Wave-0 stub.**
//!
//! Filled by plan **05-06**: `#[cube]` kernels that emit the multinomial
//! logistic loss + gradient for the L-BFGS solver — the numerically stable
//! log-sum-exp (`m = max_k raw_k; lse = m + log(Σ exp(raw_k − m))`, analog
//! `reduce.rs`'s `reduce_max_shared` for the max and `dist_combine_clamp` for the
//! 2D logits map). The L-BFGS two-loop recursion + line search live HOST-side
//! (D-10), reading back one scalar (max projected gradient) per outer iteration;
//! the kernel only computes loss + grad.
//!
//! Until 05-06 fills it this file is an empty (compiling) module body; the Wave-0
//! scaffold owns the `pub mod lbfgs;` registration in `lib.rs`, so 05-06 only
//! touches THIS file (file-disjoint, parallel-safe). No `#[cube]` kernel yet.
//!
//! Tests live in `crates/mlrs-backend/tests/lbfgs_test.rs` (AGENTS.md §2).
