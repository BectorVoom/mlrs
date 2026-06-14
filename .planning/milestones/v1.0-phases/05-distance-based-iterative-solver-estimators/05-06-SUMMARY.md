---
phase: 05-distance-based-iterative-solver-estimators
plan: 06
subsystem: prims
tags: [lbfgs, softmax, multinomial, logistic, two-loop-recursion, strong-wolfe, convex-quadratic, d10, d12, cubecl-cpu, host-loop, oracle, primitive-first]

# Dependency graph
requires:
  - phase: 05-distance-based-iterative-solver-estimators
    plan: 01
    provides: "lbfgs.rs/prims/lbfgs.rs/lbfgs_test.rs stubs + lib.rs/prims/mod.rs registrations; logistic_binary_{f32,f64}.npz + logistic_multi_{f32,f64}.npz fixtures (X/y/coef/intercept); AlgoError::NotConverged/InvalidC"
  - phase: 05-distance-based-iterative-solver-estimators
    plan: 02
    provides: "cubecl-cpu MLIR-safe #[cube] idiom (no SharedMemory/bool/F::INFINITY/atomics; single-unit GATHER, F/u32 accumulators, if-guards) — reused for the softmax loss/grad kernel"
  - phase: 05-distance-based-iterative-solver-estimators
    plan: 05
    provides: "host-loop + single-scalar-readback (to_host_metered) prim pattern (D-10); validate-before-launch + host_to_f64/from_f64 helper shape"
  - phase: 04-closed-form-estimators
    provides: "prims::cholesky validate→launch→scalar-readback wrapper; DeviceArray from_host/to_host/to_host_metered/release_into; capability::skip_f64_with_log"
provides:
  - "mlrs_kernels::lbfgs::softmax_loss_grad — feature-free #[cube] stable symmetric-multinomial softmax loss + gradient kernel (K full weight vectors, D-12; logsumexp row-max-before-exp Pitfall 4; intercept unpenalized Pitfall 3)"
  - "mlrs_backend::prims::lbfgs::lbfgs_minimize — generic host L-BFGS (two-loop recursion m=10 + strong-Wolfe line search; gtol=1e-4/ftol=64·eps/maxls=50/maxiter=100; (s,y) history + grad reused; ONE scalar max|grad| per iteration, D-10) returning LbfgsResult{x,loss,max_grad,iters,converged}"
  - "mlrs_backend::prims::lbfgs::softmax_loss_grad — host launcher for the Task-1 kernel (validate-before-launch ASVS V5; one metered scalar loss readback)"
  - "lbfgs_test.rs GREEN on cpu(f64): convex-quadratic invariant (x*=A⁻¹b within 1e-5, Pitfall 5) + softmax loss/grad oracle (binary+multi, f32/f64, finite-difference gradient check) + L-BFGS+softmax no-NaN convergence smoke"
affects: [05-10, 05-11]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Host iterative-optimizer with a closure objective: lbfgs_minimize is GENERIC over `FnMut(&[f64]) -> (loss, grad)`, so the solver core is validated standalone on a convex quadratic (pure host closure, no device) BEFORE the device-backed softmax closure consumes it (Pitfall 5 — isolates solver correctness from sklearn path-matching)"
    - "cubecl-cpu-safe softmax/logsumexp: the whole loss+grad runs on unit 0 of a single cube (modest LogReg n/K/d) with F/u32 accumulators + if-guarded row-max scan (no SharedMemory/bool/F::INFINITY/atomics); the natural log is the `.ln()` method and exp is `.exp()` (cube Float method forms — NOT F::log which the macro rejects)"
    - "Symmetric over-parameterized multinomial (D-12): K full weight vectors so binary is genuinely the K=2 case of the SAME kernel + same host loop; the parameter vector is [W(k×d) | b(k)] flattened, the closure splits/relaunches/re-flattens (gradW | gradb)"
    - "Strong-Wolfe (bracket+zoom, Nocedal & Wright Alg. 3.5/3.6) line search capped at maxls=50; the (s,y) pair is skipped when sᵀy ≤ 1e-10 to keep the implicit Hessian positive definite"

key-files:
  created: []
  modified:
    - "crates/mlrs-kernels/src/lbfgs.rs (filled the 05-01 stub: softmax_loss_grad #[cube] kernel; pub use inside file)"
    - "crates/mlrs-backend/src/prims/lbfgs.rs (filled the 05-01 stub: lbfgs_minimize host loop + line_search_wolfe/zoom + softmax_loss_grad launcher + validate_softmax_geometry; constants LBFGS_M/MAXLS/MAXITER/GTOL/FTOL)"
    - "crates/mlrs-backend/tests/lbfgs_test.rs (de-#[ignore]d: convex-quadratic invariant + softmax oracle + finite-difference grad check + no-NaN smoke)"

key-decisions:
  - "lbfgs_minimize is parameterized by a host closure returning (loss, grad), NOT hard-wired to the softmax kernel — this is what makes the Pitfall-5 convex-quadratic standalone validation possible (a pure-host closure proves the solver before any device path) and lets plan 05-10's LogReg estimator wrap softmax_loss_grad in a closure"
  - "Natural log inside the #[cube] kernel uses the `.ln()` method (and `.exp()`), not `F::log(x)` / `Log::log(x)` — the cube macro maps the associated-function `Log::log` to `__expand_log` which fails type-parameter resolution, while the method forms resolve to the cube logsumexp ops (mirrors the existing `.sqrt()` usage in cholesky.rs/jacobi). Captured as a pattern."
  - "Test geometry n is derived from y.len() (39 for the K=3 multi fixture — `LOG_N_SAMPLES//n_classes*n_classes`, 40 for K=2 binary), not hardcoded LOG_N_SAMPLES — the multiclass blob keeps `per` rows per class"
  - "Tasks 2 and 3 committed together (69abaae): both modify prims/lbfgs.rs + lbfgs_test.rs in inseparable ways (the convex-quadratic invariant and the softmax oracle share the same two files and the same lbfgs_minimize entry); a clean per-task file split is not possible without artificial intermediate reverts. Task 1 (the kernel) is its own commit (ed0627a)."

patterns-established:
  - "Closure-objective host optimizer (Pitfall 5 standalone gate): a generic `lbfgs_minimize(x0, FnMut(&[f64])->(f64,Vec<f64>), …)` is provable on a convex quadratic with a known minimizer x*=A⁻¹b BEFORE any device-backed objective consumes it — the iterative-solver analogue of the algebraic-invariant pattern (cholesky ‖A·x−b‖)"

requirements-completed: [LINEAR-05]

# Metrics
duration: 18min
completed: 2026-06-13
---

# Phase 5 Plan 06: L-BFGS Solver Primitive (D-03, highest risk) Summary

**The NEW L-BFGS solver core for LogisticRegression — THE highest correctness risk in the project: a feature-free `#[cube]` stable symmetric-multinomial softmax loss+gradient kernel (K full weight vectors D-12; row-max-before-exp logsumexp Pitfall 4; intercept unpenalized + `l2_reg=1/(C·n)` Pitfall 3) driven by a GENERIC host `lbfgs_minimize` (two-loop recursion `m=10` + strong-Wolfe line search; scipy constants `gtol=1e-4`/`ftol=64·eps`/`maxls=50`/`maxiter=100`; `(s,y)` history + gradient reused; exactly ONE scalar `max|grad|` per iteration, D-10). Proven CORRECT STANDALONE on a convex quadratic `½xᵀAx−bᵀx → x*=A⁻¹b` within 1e-5 (Pitfall 5) BEFORE the softmax path; the softmax loss/grad matches a host numpy-equivalent reference within 1e-5 for binary AND multiclass (gradient cross-checked by central finite difference), and L-BFGS+softmax converges without NaN — GREEN on cpu(f64) before the LogReg estimator (05-10) consumes it (D-01 primitive-first).**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-06-13T03:30:00Z
- **Completed:** 2026-06-13T03:48:00Z
- **Tasks:** 3 (Task 1 TDD-shaped; Tasks 2-3 co-located)
- **Files modified:** 3 (kernel + prim + test — all 05-01 stubs filled; zero shared-file edits)

## Accomplishments
- Filled `mlrs_kernels::lbfgs::softmax_loss_grad`: one feature-free `#[cube]` kernel generic over `<F: Float + CubeElement>` that emits the SYMMETRIC over-parameterized multinomial loss + gradient on unit 0 of a single cube — `raw[i,k]=x[i]·w[k]+b[k]`, the STABLE logsumexp (`row_max` via an `if`-guarded forward scan, then `row_max + (Σ exp(raw−row_max)).ln()` — Pitfall 4), per-row softmax `p`, the loss `(1/n)Σ(lse−raw_y)+½·l2_reg·‖W‖²` (intercept UNPENALIZED — Pitfall 3), and `gradW=(1/n)(P−Y)ᵀX+l2_reg·W`, `gradb=(1/n)Σ(p−Y)` (no penalty). K full weight vectors (D-12), so binary is the K=2 case. cubecl-cpu MLIR-safe (no SharedMemory/bool/F::INFINITY/atomics).
- Filled `mlrs_backend::prims::lbfgs::lbfgs_minimize`: a GENERIC host L-BFGS parameterized by a closure `FnMut(&[f64]) -> (loss, grad)`. Owns the standard two-loop recursion with an `m=10` ring of `(s,y)` pairs (γ-scaled initial Hessian from the most recent pair), a strong-Wolfe bracket+zoom line search (Nocedal & Wright Alg. 3.5/3.6, `c1=1e-4`/`c2=0.9`, `maxls=50`), the scipy constants (`gtol=1e-4` on `max|grad|`, `ftol=64·eps` relative-f, `maxiter=100`), reuses the gradient + history every iteration, and reads exactly ONE scalar (`max|grad|`) per iteration (D-10). Returns `LbfgsResult{x,loss,max_grad,iters,converged}` — the last iterate even at `maxiter` (the estimator surfaces `NotConverged`).
- Filled `mlrs_backend::prims::lbfgs::softmax_loss_grad`: the host launcher for the Task-1 kernel; validates `x.len()==n*d` / `y.len()==n` / `w.len()==k*d` / `b.len()==k` → `PrimError::ShapeMismatch` BEFORE any unsafe launch (T-05-06-01 / ASVS V5), launches the single-cube kernel, reads back the loss via `to_host_metered` (the metered D-10 scalar) plus the `(gradW, gradb)` vectors.
- De-`#[ignore]`d `lbfgs_test.rs` with the two-stage standalone oracle: (1) the convex-quadratic invariant — a fixed SPD `A`, `f(x)=½xᵀAx−bᵀx`, gradient `Ax−b`, asserting the L-BFGS iterate equals `x*=A⁻¹b` (host Gaussian-elimination solve) within 1e-5 (Pitfall 5, f32+f64); (2) the softmax oracle — for `logistic_binary` (K=2) and `logistic_multi` (K=3, f32+f64), the device loss matches a host numpy-equivalent reference within 1e-5 AND the device `(gradW,gradb)` match a central finite difference of that reference; plus a smoke test that `lbfgs_minimize` driven by `softmax_loss_grad` converges on the binary fixture without NaN.
- Verified the full gate: `cargo build -p mlrs-kernels` green; `cargo test --features cpu -p mlrs-backend --test lbfgs_test` 9/9 green (convex-quadratic + softmax oracle + smoke, incl. f64); `cargo build -p mlrs-backend --features rocm --tests` green; `lib.rs`/`prims/mod.rs` untouched.

## Task Commits

1. **Task 1: stable symmetric-multinomial softmax loss+grad `#[cube]` kernel** — `ed0627a` (feat)
2. **Tasks 2+3: lbfgs_minimize host loop + softmax launcher + standalone oracle** — `69abaae` (feat) — co-located in `prims/lbfgs.rs` + `lbfgs_test.rs` (inseparable; see Decisions)

## Files Created/Modified
- `crates/mlrs-kernels/src/lbfgs.rs` — `softmax_loss_grad` `#[cube]` kernel; `pub use self::softmax_loss_grad as lbfgs_softmax_loss_grad` inside the file. (lib.rs untouched.)
- `crates/mlrs-backend/src/prims/lbfgs.rs` — `lbfgs_minimize` + `line_search_wolfe`/`zoom` + `softmax_loss_grad` launcher + `validate_softmax_geometry` + `max_abs`/`dot`/`axpy`/`host_to_f64`/`from_f64`; constants `LBFGS_M/MAXLS/MAXITER/GTOL/FTOL`; `LbfgsResult`. (prims/mod.rs untouched.)
- `crates/mlrs-backend/tests/lbfgs_test.rs` — `check_convex_quadratic` + `solve_spd` + `ref_loss` + `check_softmax_oracle` + `check_lbfgs_softmax_smoke` + 9 tests.

## Decisions Made
- **lbfgs_minimize is closure-parameterized, not softmax-hardwired:** this is what makes the Pitfall-5 convex-quadratic standalone validation a pure-host test (no device), proving the solver before any device path consumes it, and lets plan 05-10 wrap `softmax_loss_grad` in a closure. The parameter vector for LogReg is `[W(k×d) | b(k)]` flattened.
- **`.ln()`/`.exp()` method forms in the kernel, not `F::log`/`Log::log`:** the cube macro maps the associated-function `Log::log(x)` to `__expand_log`, which fails type-parameter resolution; the `.ln()`/`.exp()` method forms resolve to the cube logsumexp ops (mirroring the existing `.sqrt()` in `cholesky.rs`/`jacobi_*`). Captured as a pattern for downstream kernel plans.
- **Test `n` derived from `y.len()`:** the multiclass blob keeps `LOG_N_SAMPLES//n_classes*n_classes = 39` rows (K=3) vs 40 (K=2 binary), so the oracle reads `n = y.len()` instead of hardcoding `LOG_N_SAMPLES`.
- **Tasks 2+3 in one commit:** both modify `prims/lbfgs.rs` + `lbfgs_test.rs` in interleaved ways (the convex-quadratic invariant and the softmax oracle share the same two files and the `lbfgs_minimize` entry); a clean per-task file split was not possible without artificial intermediate reverts. Task 1 (the kernel) is its own commit.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking issue] `F::log` / `Log::log` rejected by the cube macro — switched to `.ln()`/`.exp()` method forms**
- **Found during:** Task 1 (building the kernel)
- **Issue:** The natural log written as `F::log(x)` resolved to a non-existent `F::__expand_log` associated function (`E0599`), and the trait-qualified `Log::log(x)` failed with `E0782` (the macro suggested `__expand_ln`). The cubecl `Log`/`Exp` traits define `log`/`exp` as associated functions, but the `#[cube]` macro only lowers the `.ln()`/`.exp()` METHOD forms (the same pattern the codebase already uses for `.sqrt()`).
- **Fix:** Used `(raw−row_max).exp()` and `sum_exp.ln()` (and `(raw−lse).exp()` for the softmax). Within scope (this plan's own kernel), no architectural change — the kernel's public signature and math are exactly as planned.
- **Files modified:** `crates/mlrs-kernels/src/lbfgs.rs`
- **Commit:** `ed0627a`

**2. [Rule 1 - Bug] Test hardcoded `n = LOG_N_SAMPLES` but the multiclass fixture has 39 rows**
- **Found during:** Task 3 (running the multiclass softmax oracle)
- **Issue:** `gen_logistic` builds the multiclass blob with `per = LOG_N_SAMPLES // n_classes` rows per class → `13*3 = 39` samples (not 40); the binary blob is `20*2 = 40`. Hardcoding `n = LOG_N_SAMPLES = 40` made the `X geometry` assert fail (156 vs 160).
- **Fix:** Derive `n = y_raw.len()` in both the oracle and smoke helpers.
- **Files modified:** `crates/mlrs-backend/tests/lbfgs_test.rs`
- **Commit:** `69abaae`

## Known Stubs

None. The kernel + `lbfgs_minimize` + `softmax_loss_grad` are fully implemented; the oracle exercises real device output (the asserted loss/gradient flow from the kernel, not hardcoded values), and the convex-quadratic minimizer is checked against a genuine host `A⁻¹b` solve.

## Issues Encountered
- The kernel re-export lives at `mlrs_kernels::lbfgs::lbfgs_softmax_loss_grad` (module path), NOT the crate root, because `lib.rs` (owned by the 05-01 scaffold) re-exports a fixed symbol set and this plan must not edit it. The prim imports via the module path — no shared-file edit needed (same as 05-05).

## Next Phase Readiness
- **Plan 05-10 (LogisticRegression estimator) unblocked:** `lbfgs_minimize(x0, closure, …) -> LbfgsResult` + `softmax_loss_grad(pool, x, y, w, b, n, d, k, l2_reg) -> (loss, gradW, gradb)` are the D-01 primitive-first core. The estimator flattens `[W|b]`, sets `l2_reg = 1/(C·n)`, wraps `softmax_loss_grad` in the closure, runs `lbfgs_minimize`, and gates on `predict`/`predict_proba` (the primary gauge-invariant gate, Pitfall 5; `coef_` is the looser secondary). The convex-quadratic invariant proves the solver is correct, so any `coef_` drift is a gauge-freedom artifact, not a solver bug.
- **Plan 05-11 (memory gate) ready:** `softmax_loss_grad` reads back exactly ONE scalar (the loss) via `to_host_metered`, and `lbfgs_minimize` reuses the gradient + `(s,y)` history across iterations — the D-10 bounded-allocation contract.
- No blockers. cpu(f64) full + rocm(f32) test-target build both green; `lib.rs`/`prims/mod.rs` untouched so the sibling Wave-2 plans stay file-disjoint.

## Threat Flags

None — no new network/auth/file surface. The trust boundary is the validated `softmax_loss_grad(n,d,k)` / `lbfgs_minimize(x0)` geometry, mitigated as the register specified: T-05-06-01 (validate parameter-vector + `n*d`/`y`/`W`/`b` lengths → `ShapeMismatch` before unsafe launch), T-05-06-02 (stable logsumexp → no NaN on well-separated classes; smoke test pins no-NaN convergence), T-05-06-03 (`maxiter=100` cap → bounded iterations, never a silent hang), T-05-06-04 (exactly ONE metered scalar readback per iteration; gradient + `(s,y)` history reused).

## Self-Check: PASSED

- All modified files verified present (lbfgs.rs kernel, prims/lbfgs.rs, lbfgs_test.rs, this SUMMARY).
- Both task commits verified in git history (`ed0627a`, `69abaae`).
- `cargo test --features cpu -p mlrs-backend --test lbfgs_test` 9/9 green (convex-quadratic invariant + softmax oracle binary+multi f32/f64 + no-NaN smoke); `cargo build -p mlrs-kernels` + `-p mlrs-backend --features rocm --tests` green; `lib.rs`/`prims/mod.rs` untouched.

---
*Phase: 05-distance-based-iterative-solver-estimators*
*Completed: 2026-06-13*
