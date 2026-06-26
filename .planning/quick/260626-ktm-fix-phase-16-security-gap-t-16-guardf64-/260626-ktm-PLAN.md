---
phase: quick-260626-ktm
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/mlrs-py/src/estimators/covariance.rs
  - crates/mlrs-py/src/lib.rs
autonomous: true
requirements: [T-16-GUARDF64]
must_haves:
  truths:
    - "No live-code site in crates/mlrs-py/src/estimators/ uses the legacy panicking pool lock (only the naive_bayes.rs doc counter-example remains)"
    - "covariance.rs locks the global pool exclusively through the poison-recovering lock_pool() path"
    - "The lib.rs lock_pool doc comment no longer claims any estimator module carries the legacy panicking form"
    - "crate mlrs-py compiles cleanly under --features cpu after the swap"
  artifacts:
    - path: "crates/mlrs-py/src/estimators/covariance.rs"
      provides: "PyEmpiricalCovariance / PyLedoitWolf bindings using lock_pool()"
      contains: "lock_pool()"
    - path: "crates/mlrs-py/src/lib.rs"
      provides: "Refreshed lock_pool doc comment (covariance migrated, zero legacy holders)"
      contains: "lock_pool"
  key_links:
    - from: "crates/mlrs-py/src/estimators/covariance.rs"
      to: "crates/mlrs-py/src/lib.rs"
      via: "crate::lock_pool() — sanctioned poison-recovering pool guard (WR-02/WR-04)"
      pattern: "crate::lock_pool\\(\\)"
---

<objective>
Close Phase-16 security gap T-16-GUARDF64 (lock_pool half): migrate the SOLE remaining
estimator module still on the legacy panicking pool lock. Replace all 12 legacy
`crate::global_pool().lock().expect("pool mutex")` sites in
`crates/mlrs-py/src/estimators/covariance.rs` with the sanctioned poison-recovering
`crate::lock_pool()` path, and refresh the now-stale doc comment in
`crates/mlrs-py/src/lib.rs` that still names other modules as legacy holders.

Purpose: A single surviving legacy `.lock().expect("pool mutex")` re-panics on a poisoned
mutex and re-bricks the interpreter, defeating the WR-02 poison-recovery on every module
once the pool is poisoned. covariance.rs is the last holdout; converting it makes the
brick-prevention complete across the binding layer.
Output: covariance.rs locks exclusively through `lock_pool()`; lib.rs doc comment reflects
zero legacy holders.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md

# The 12 legacy lock sites to swap
@crates/mlrs-py/src/estimators/covariance.rs

# lock_pool() definition (~line 144) + stale doc comment (~lines 108-118)
@crates/mlrs-py/src/lib.rs

# Reference: already-migrated lock_pool() usage pattern
@crates/mlrs-py/src/estimators/linear.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Swap the 12 legacy pool-lock sites in covariance.rs to lock_pool()</name>
  <files>crates/mlrs-py/src/estimators/covariance.rs</files>
  <action>
At each of the 12 sites (current lines 85, 117, 124, 131, 138, 145, 152, 222, 252,
259, 266, 273), replace the call expression `crate::global_pool().lock().expect("pool mutex")`
with `crate::lock_pool()`. This is a mechanical, identical swap — both expressions return
`MutexGuard<'static, BufferPool<ActiveRuntime>>` and `lock_pool()` takes no arguments.

Keep the existing left-hand bindings exactly as they are: the two `let mut pool = ...`
sites (lines 85, 222) stay `let mut pool = crate::lock_pool();` and the ten `let pool = ...`
sites stay `let pool = crate::lock_pool();`. Do NOT alter surrounding code, indentation,
or the `py.detach`/`Python::detach` closures that hold the guard.

This is the sanctioned poison-recovering path per lib.rs (WR-02/WR-04): on a poisoned
mutex it recovers the guard and re-baselines the BufferPool accounting instead of panicking.
Use a global string replacement of the exact legacy expression — every live occurrence in
this file is one of the 12 target sites.

Do NOT touch crates/mlrs-py/src/estimators/naive_bayes.rs — its line-19 occurrence is an
intentional doc-comment counter-example and is out of scope.
  </action>
  <verify>
    <automated>test "$(grep -cE 'global_pool\(\)\.lock\(\)\.expect' crates/mlrs-py/src/estimators/covariance.rs)" = "0" && test "$(grep -c 'crate::lock_pool()' crates/mlrs-py/src/estimators/covariance.rs)" = "12" && echo OK</automated>
  </verify>
  <done>
covariance.rs has zero `global_pool().lock().expect` occurrences and exactly 12
`crate::lock_pool()` call sites. A repo-wide
`grep -rnE 'global_pool\(\)\.lock\(\)\.expect' crates/mlrs-py/src/estimators/`
returns ONLY `naive_bayes.rs:19` (the doc-comment counter-example).
  </done>
</task>

<task type="auto">
  <name>Task 2: Refresh the stale lock_pool doc comment in lib.rs and confirm clean build</name>
  <files>crates/mlrs-py/src/lib.rs</files>
  <action>
In the `lock_pool` doc comment (the "## This is the SANCTIONED lock path (WR-04)" block,
~lines 108-118), the sentence/parenthetical that lists
`linear`/`cluster`/`decomposition`/`covariance`/`neighbors`/`projection` wrappers as still
carrying the legacy panicking form is now STALE — after Task 1 NO estimator module uses the
legacy panicking lock. Update that text so it states that every estimator wrapper now uses
`lock_pool` exclusively, that covariance was the last remaining legacy holder and was
migrated under T-16-GUARDF64, and that new estimators MUST continue to use `lock_pool`.
Keep the rest of the doc comment (the WR-02 rationale, the ACCOUNTING RE-BASELINE / WR-06
sections, and the `#[allow(dead_code)]` attribute and fn body) unchanged.

Do NOT remove the legitimate `global_pool().lock().expect("pool mutex")` reference earlier
in the same paragraph that explains WHY one surviving legacy site re-bricks the interpreter
— that reference is the rationale, not a stale module list. (lib.rs is outside the
estimators/ grep scope, so this reference does not affect Task 1's verification.)
  </action>
  <verify>
    <automated>test "$(grep -c 'neighbors`/`projection' crates/mlrs-py/src/lib.rs)" = "0" && test "$(grep -c 'still carry the legacy panicking form' crates/mlrs-py/src/lib.rs)" = "0" && cargo build -p mlrs-py --features cpu 2>&1 | tail -3</automated>
  </verify>
  <done>
The stale module list ("`neighbors`/`projection`" and "still carry the legacy panicking
form") is gone from lib.rs; the doc comment now reflects covariance as the migrated last
holder with zero legacy holders remaining. `cargo build -p mlrs-py --features cpu`
completes cleanly (no errors).
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Python `py.detach` closure → global BufferPool guard | A device fault / OOM / unsupported-op panic inside a detach closure that holds the pool guard poisons the mutex |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-16-GUARDF64 | Denial of Service | `crates/mlrs-py/src/estimators/covariance.rs` 12 pool-lock sites | mitigate | Replace legacy `global_pool().lock().expect("pool mutex")` with poison-recovering `crate::lock_pool()`, so one recoverable device error no longer permanently bricks the interpreter via mutex poisoning (WR-02/WR-04). |
</threat_model>

<verification>
- `grep -rnE 'global_pool\(\)\.lock\(\)\.expect' crates/mlrs-py/src/estimators/` → ONLY `naive_bayes.rs:19`
- `grep -c 'crate::lock_pool()' crates/mlrs-py/src/estimators/covariance.rs` → `12`
- lib.rs doc comment no longer lists any module as a legacy holder
- `cargo build -p mlrs-py --features cpu` → clean
</verification>

<success_criteria>
Zero live-code legacy pool-lock sites remain in any estimator module; covariance.rs uses
`lock_pool()` at all 12 sites; the lib.rs doc comment accurately states no estimator module
carries the legacy panicking form (covariance migrated under T-16-GUARDF64); mlrs-py builds
cleanly under the cpu feature. Behavior-preserving — existing oracle suites cover numerical
correctness; no new tests required.
</success_criteria>

<output>
Create `.planning/quick/260626-ktm-fix-phase-16-security-gap-t-16-guardf64-/260626-ktm-SUMMARY.md` when done
</output>
