# Deferred Items — Phase 10

Out-of-scope discoveries logged during execution (NOT fixed in-plan).

- [10-02] `clippy::approx_constant` error in `crates/mlrs-kernels/src/elementwise.rs:282` (FRAC_PI_2 literal `1.570_796_326_794_896_6`). Pre-existing, unrelated to SGD; `cargo clippy -p mlrs-kernels` fails on it. Fix: use `F::new(core::f64::consts::FRAC_PI_2)` or allow the lint. (Out of scope: SCOPE BOUNDARY — pre-existing, different file.)
