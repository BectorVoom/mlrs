//! `host_simd` — run a host prim's hot region on the machine's REAL vector unit.
//!
//! Every `*_host` prim in this crate is compiled for the **x86-64 baseline**:
//! `mlrs` ships one binary per backend and sets no `target-cpu`, so `rustc`
//! targets SSE2 — 128-bit vectors, four `f32` lanes, and no FMA. That is a 2001
//! instruction set.
//!
//! `cubecl-cpu` has no such constraint. It JITs through LLVM's `ExecutionEngine`,
//! which targets the HOST, so a device "cpu kernel" is compiled for whatever the
//! machine actually has (AVX-512 on the Zen5 everything here was measured on).
//! This is why the first version of [`knn_host`](super::knn_host)'s scan —
//! ordinary Rust at `-O3` — measured **2× SLOWER** than the `-O0` JIT kernel it
//! was replacing. The Rust was not worse; it was being compiled for a narrower
//! machine.
//!
//! [`avx2_available`] detects AVX2 + FMA once, and each prim pairs it with a
//! `#[target_feature]` TWIN of its hot function, so that function's body is
//! optimized for the wider target while the baseline body stays the one that runs
//! on a machine without it. The binary's floor is unchanged.
//!
//! ## It cannot change a result
//! Two properties make this safe to apply to a numerical kernel without
//! re-validating its output:
//!
//! 1. **Widening a vector reassociates nothing.** Vectorization across
//!    INDEPENDENT accumulators (the only kind LLVM may perform without
//!    fast-math) computes each lane's sum in exactly the lane's own order,
//!    whatever the register width. A loop LLVM would not vectorize at SSE2 is
//!    not vectorized at AVX2 either.
//! 2. **`fma` does not silently contract.** Rust emits no `contract` fast-math
//!    flag, so `a + b * c` stays a multiply and an add even where an FMA
//!    instruction exists. Only an explicit `mul_add` becomes one — and that call
//!    already produces the same value on both paths (without hardware FMA it
//!    lowers to an `fmaf` LIBRARY call, which is slow, not different; several
//!    prims here avoid `mul_add` for exactly that reason and are unaffected).
//!
//! `knn_host_test::avx2_and_baseline_agree_bitwise` asserts (1) and (2) together
//! on real data, over five metrics and both float widths.
//!
//! ## The shape to write, and the one that does NOT work
//! Write an explicit twin:
//!
//! ```ignore
//! #[inline]
//! fn dispatch_hot(args: ...) {
//!     #[cfg(target_arch = "x86_64")]
//!     if avx2_available() {
//!         // SAFETY: guarded by the detection this branch tests.
//!         unsafe { hot_avx2(args) }
//!         return;
//!     }
//!     hot(args)
//! }
//!
//! #[cfg(target_arch = "x86_64")]
//! #[target_feature(enable = "avx2", enable = "fma")]
//! unsafe fn hot_avx2(args: ...) { hot(args) }
//!
//! #[inline(always)]            // <- REQUIRED: this is what gets widened
//! fn hot(args: ...) { ... }
//! ```
//!
//! The ergonomic alternative — a `with_avx2(|| ...)` helper taking a closure —
//! was written, measured, and DELETED. It does not propagate: on the same
//! `knn_host` scan, the twin above gives 1.35-1.59× while routing the identical
//! body through a generic `#[target_feature]` shim that calls a closure gives
//! 1.00-1.19× (euclidean 0.041 s twin vs 0.049 s closure vs 0.056 s baseline,
//! and manhattan/cosine were a dead heat with the baseline). The feature set is a
//! property of how a function was COMPILED, and the extra closure hop is enough
//! to stop the hot body being inlined INTO the widened frame.
//!
//! The same reasoning rules out one line in
//! [`WorkerPool::run`](super::host_pool::WorkerPool::run): its workers reach the
//! pass through a FUNCTION POINTER, so nothing on the call side can affect how
//! the pass was compiled. That is why this is a per-prim change.
//!
//! ## What it bought, per prim (16-core Zen5, `MLRS_HOST_AVX2` A/B)
//!
//! | prim | hot function | config | speedup |
//! |------|--------------|--------|---------|
//! | [`knn_host`](super::knn_host) | `scan_block` | 50k×5k×16, k=5, per metric | 1.15-1.59× |
//! | [`huber_objective`](super::huber_objective) | `Accum::rows` | 50k×32 / 100k×64 fit | 2.50× / 1.87× |
//! | [`svm_objective`](super::svm_objective) | `Accum::rows` | 50k×32 / 100k×64 fit | 1.90× / 1.51× |
//! | [`linear_predict`](super::linear_predict) | `matvec_bias_rows` | 200k×32 / ×64 predict | 2.00× / 1.25× |
//! | [`gram_host`](super::gram_host) | `sweep_block` | Ridge fit, d=16 | 1.55× |
//! | [`sgd_host`](super::sgd_host) | `solve_loss` | 20k×256 / 10k×512 fit | 1.06× / 1.09× |
//!
//! Two results that are NOT in the table, because they are the useful negatives:
//!
//! - **`gram_host` above d≈64 shows nothing** end to end (1.0× at d=64/128). The
//!   sweep itself still gets faster; what happens is that the `O(d³)` eigen/
//!   Cholesky solve after it takes over the fit. Widening a hot loop only shows
//!   up where that loop is the cost.
//! - **`hgb_host` gains nothing at all** (0.106 s vs 0.106 s at 50k×16; 0.167 vs
//!   0.165 at 50k×64 — i.e. a hair SLOWER, inside noise). Its dominant pass is
//!   the histogram gather, whose inner statement is
//!   `slice[bin*3 + j] += ...` with a DATA-DEPENDENT bin — a scatter, which no
//!   register width helps. The twins were written, measured, and removed rather
//!   than left in as inert boilerplate. Do not re-add them without a different
//!   histogram shape to justify it.
//!
//! [`gmm_host`](super::gmm_host) and `tsne_host` are not done: their hot regions
//! are 150-line pass CLOSURES rather than callable leaf functions, so applying
//! this means extracting a body first, and both files were being rewritten
//! concurrently when this went in. `gmm_host` in particular is `f64`-dense
//! (SSE2 gives its Mahalanobis loops two lanes where this machine has four) and
//! is the best remaining candidate.
//!
//! Tests live in `crates/mlrs-backend/tests/` (AGENTS.md §2).

/// Does this machine have AVX2 + FMA?
///
/// `MLRS_HOST_AVX2=0` forces the baseline body for every host prim at once —
/// the A/B knob every measurement in this module's table was taken with, and the
/// escape hatch if a machine ever mis-detects.
///
/// ## Why the environment half is cached and the override half is not
/// CPUID and `getenv` are both process-constant, so they are resolved once: this
/// is called from dispatchers that run per row block, and `abflag::var`'s
/// fallback allocates a `String` out of `std::env::var` on every miss.
///
/// The THREAD-LOCAL override cannot be cached with them. A test flips this knob
/// mid-process to compare the two bodies, and a fully-cached answer would ignore
/// the second flip — leaving `avx2_and_baseline_agree_bitwise` comparing the
/// AVX2 body against ITSELF and passing no matter what. (It did, until this
/// split.) [`abflag::local_override`](crate::abflag::local_override) is the cheap
/// half: a thread-local borrow and a hash lookup, no `getenv`, no allocation
/// unless an override is actually set.
#[cfg(target_arch = "x86_64")]
pub fn avx2_available() -> bool {
    use std::sync::OnceLock;
    if let Some(forced) = crate::abflag::local_override("MLRS_HOST_AVX2") {
        return forced.as_deref() != Some("0") && cpu_has_avx2();
    }
    static ENV_AND_CPU: OnceLock<bool> = OnceLock::new();
    *ENV_AND_CPU.get_or_init(|| {
        if std::env::var("MLRS_HOST_AVX2").as_deref() == Ok("0") {
            return false;
        }
        cpu_has_avx2()
    })
}

/// The CPUID half alone, cached — a machine's feature set cannot change.
#[cfg(target_arch = "x86_64")]
fn cpu_has_avx2() -> bool {
    use std::sync::OnceLock;
    static CPU: OnceLock<bool> = OnceLock::new();
    *CPU.get_or_init(|| {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    })
}

/// Always false off x86-64 — there is no second body to dispatch to.
#[cfg(not(target_arch = "x86_64"))]
pub fn avx2_available() -> bool {
    false
}
