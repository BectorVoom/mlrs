---
phase: 07-covariance-projection
plan: 02
subsystem: backend
tags: [prim, rng, splitmix64, gaussian, achlioptas, fisher-yates, host-glue, memory-gate, reproducibility]

# Dependency graph
requires:
  - phase: 07-covariance-projection
    plan: 01
    provides: "prims::rng empty stub FILE + #[ignore] rng_test.rs scaffold + prims/mod.rs registration"
  - phase: 05-distance-iterative-estimators
    provides: "SplitMix64 (private) in prims/kmeans.rs + kmeanspp_sample callers + host_to_f64/f64_to_host bit-cast idiom"
  - phase: 02-foundations
    provides: "BufferPool/PoolStats counters + DeviceArray::from_host single-upload + memory_gate_test.rs idiom"
  - phase: 03-svd-eig
    provides: "skip_f64_with_log f64 capability gate (gemm_test.rs pattern)"
provides:
  - "mlrs_backend::prims::rng::SplitMix64 (pub, promoted verbatim — backend-reproducible host PRNG, ASVS V6)"
  - "prims::rng::gaussian_matrix::<F> — N(0,1/n_components) Box-Muller, host-generate + single upload"
  - "prims::rng::sparse_achlioptas_matrix::<F> — v=sqrt((1/density)/n_components), dense storage (D-12)"
  - "prims::rng::permutation — unbiased Fisher-Yates via next_below"
  - "5 live rng_ tests (distribution + byte-identical seed-repro + Achlioptas + bijection + memory gate)"
affects: [07-06-random-projection]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Host-side RNG glue (generate on host → ONE DeviceArray::from_host upload) — NO device kernel, dodges the cpu-MLIR SharedMemory/atomics landmines entirely (RESEARCH Anti-Pattern device-side RNG)"
    - "Verbatim PRNG promotion: SplitMix64 moved kmeans.rs → rng.rs byte-frozen (mix constants unchanged), kmeans.rs now `use crate::prims::rng::SplitMix64` so the k-means++ stream is identical (Pitfall 7)"
    - "Box-Muller with a cached second sample so the stream is fully seed-determined (T-07-02)"
    - "Achlioptas branch from a single next_f64 per entry: [0,d/2)→+v, [d/2,d)→−v, [d,1)→0 — exactly `density` nonzero mass"

key-files:
  created: []
  modified:
    - crates/mlrs-backend/src/prims/rng.rs
    - crates/mlrs-backend/src/prims/kmeans.rs
    - crates/mlrs-backend/tests/rng_test.rs

key-decisions:
  - "next_u64()%-grep acceptance criterion (literal-0) is unsatisfiable under the verbatim-promotion mandate: the promoted SplitMix64's OWN doc-comments warn 'A plain next_u64() % bound is biased' + the two module doc lines cite 'NEVER next_u64() % n'. No actual biased modulo exists in any generator (permutation uses next_below); T-07-03 is met. Literal grep == 3, all in doc-comments."
  - "Box-Muller (not inverse-CDF/ziggurat) per RESEARCH §Box-Muller: exact, branch-light, trivially reproducible from two uniforms; speed irrelevant at v2 matrix sizes. u1 floored to f64::MIN_POSITIVE so ln(u1) is finite on the vanishingly-rare exact-zero draw."
  - "host_to_f64 NOT copied (unused here) — only f64_to_host is needed because every generator accumulates in f64 and writes back to F; avoids a dead-code helper."

requirements-completed: [PRIM-06]

# Metrics
duration: 9min
completed: 2026-06-20
---

# Phase 7 Plan 02: PRIM-06 RNG-Matrix Primitive Summary

**Landed `prims/rng.rs` as pure host-side glue: promoted `SplitMix64` verbatim out of `kmeans.rs` (now `pub`, byte-frozen stream so kmeans tests stay green) and added a Box-Muller `N(0,1/n_components)` Gaussian generator, an Achlioptas sparse generator (`v=sqrt((1/density)/n_components)`, dense), and an unbiased Fisher-Yates permutation — all generate-on-host → single `DeviceArray::from_host` upload, validated for distribution + BYTE-IDENTICAL seed-reproducibility (the cpu==rocm PRIM-06 gate) + a bounded-allocation PoolStats memory gate.**

## Performance

- **Duration:** ~9 min
- **Completed:** 2026-06-20
- **Tasks:** 2 of 2
- **Files modified:** 3 (0 created, 3 modified — the rng.rs/rng_test.rs stub FILES were created by 07-01)

## Accomplishments

### Task 1 — Promote SplitMix64 + add Gaussian/Achlioptas/permutation generators (commit 1a1c19d)
- Moved the `SplitMix64` struct + all four methods (`new`/`next_u64`/`next_f64`/`next_below`) out of `kmeans.rs` into `prims/rng.rs` VERBATIM — the mix constants (`0x9E37…`, `0xBF58…`, `0x94D0…`) and method bodies are byte-identical; made the struct + methods `pub`. `kmeans.rs` now `use crate::prims::rng::SplitMix64` so `kmeanspp_sample` (callers unchanged) compiles against the promoted struct. The k-means++ stream is therefore identical (Pitfall 7).
- `gaussian_matrix::<F>(pool, seed, n_components, n_features)`: Box-Muller from pairs of `next_f64()` uniforms (cached second sample so the stream is fully seed-determined), each scaled `× 1/sqrt(n_components)` → `N(0, 1/n_components)`. f64 accumulate, cast to F via `f64_to_host`, ONE `DeviceArray::from_host` upload. Row-major flat.
- `sparse_achlioptas_matrix::<F>(pool, seed, n_components, n_features, density)`: per-entry branch from one `next_f64()` — `[0, density/2)→+v`, `[density/2, density)→−v`, `[density,1)→0`, with `v = sqrt((1/density)/n_components)`. Stored DENSE (D-12). Single upload.
- `permutation(seed, n) -> Vec<usize>`: Fisher-Yates over `0..n` using `SplitMix64::next_below(i+1)` (UNBIASED rejection sampling) for each swap index — never `next_u64() % n`.
- Validate-before-allocate (ASVS V5): `validate_shape` rejects `n_components==0`/`n_features==0` as a typed `PrimError::ShapeMismatch`; Achlioptas additionally rejects `density ∉ (0,1]` as a synthetic `"density"` ShapeMismatch (the PrimError layer has no `InvalidDensity` — that's the estimator-side AlgoError, Plan 06) — all BEFORE any allocation.
- Gate: `cargo build -p mlrs-backend --features cpu` exits 0; `kmeanspp_test` (3/3) + `lloyd_test` (4/4) pass — the SplitMix64 source move preserved the exact stream.

### Task 2 — rng_test.rs: live distribution + seed-repro + Achlioptas + permutation + memory gate (commit 6ee33e9)
- Removed all `#[ignore]` and wired the real assertions over the live `prims::rng` calls:
  - `rng_gaussian_distribution`: 64×512=32768-sample Gaussian; sample mean `< 5e-3` (≈0) and variance within 20% of `1/n_components` (generous statistical band, not 1e-5).
  - `rng_seed_reproducible`: two same-seed `gaussian_matrix::<f64>` draws are asserted BYTE-IDENTICAL (`Vec` equality — the PRIM-06 hard gate guaranteeing cpu==rocm), a different seed differs, and Achlioptas + permutation are likewise byte-reproducible. f64-gated by `skip_f64_with_log`.
  - `rng_achlioptas_density`: 128×256 sparse matrix; observed nonzero fraction within 0.03 of `density=0.25`, every nonzero entry exactly `±v`.
  - `rng_permutation_bijection`: over `RNG_TRIALS=50` seeds × `n ∈ {0,1,2,7,64,257}`, the sorted output equals `0..n` (bijection); seed-reproducible, different seed differs.
  - `rng_memory_gate`: drives `gaussian_matrix` N=6× at a fixed shape, releasing each matrix; asserts `allocations`/`live_bytes`/`peak_bytes` all CONSERVE from iteration 1 (the free-list serves the released matrix buffer — host-generate-then-single-upload bounded footprint, the D-10 idiom from `memory_gate_test.rs`).
- Gate: `cargo test -p mlrs-backend --features cpu --test rng_test` → 5/5 pass, 0 ignored, no warnings.

## Deviations from Plan

### Acceptance-criterion reconciliation (not a code change)

**1. [Rule 1 - Acceptance criterion] `grep -c "next_u64() % " rng.rs == 0` is unsatisfiable under verbatim promotion**
- **Found during:** Task 1 acceptance-grep verification.
- **Issue:** The criterion requires zero literal `next_u64() % ` matches. But the plan ALSO mandates promoting `SplitMix64` VERBATIM (Pitfall 7), and the struct's own `next_below` doc-comment reads "A plain `next_u64() % bound` is biased…", plus the module doc cites "NEVER `next_u64() % n`" twice. The literal grep therefore returns 3 — all inside warning doc-comments, none an actual biased-modulo call site.
- **Resolution:** No code change. The criterion's INTENT (no biased modulo in real code) is satisfied — `permutation` draws via `next_below` (unbiased rejection sampling), and T-07-03 (the threat-register Tampering mitigation) is met. Verbatim promotion takes precedence over a literal-0 line count that the promoted text makes impossible.
- **Files:** crates/mlrs-backend/src/prims/rng.rs

### Other notes
- `host_to_f64` was NOT copied from kmeans.rs (the plan said "copy host_to_f64/f64_to_host verbatim"): every generator accumulates in f64 and only writes BACK to F, so only `f64_to_host` is used. Copying the unused `host_to_f64` would be dead code — omitted to keep the module warning-clean. `f64_to_host` is byte-verbatim from kmeans.rs.
- Box-Muller floors `u1` to `f64::MIN_POSITIVE` before `ln(u1)` to avoid `-inf` on the vanishingly-rare exact-zero `next_f64()` draw; deterministic and seed-stable.

## Known Stubs

None. Both `prims/rng.rs` and `rng_test.rs` are now fully implemented — no `#[ignore]`, no placeholder data, no empty stub bodies remain from the 07-01 scaffold. The four PRIM-06 generators are live and the five tests assert real distribution/reproducibility/bijection/memory properties.

## Threat Flags

None. No new network endpoint, auth path, file access, or schema change. The RNG is host-only, seed-driven (never OsRng / never the `rand` crate — ASVS V6), and introduces no new device kernel. The plan's threat register (T-07-02 reproducibility, T-07-03 unbiased draw, T-07-04 hyperparameter guard) is fully mitigated as specified.

## Verification

- `cargo build -p mlrs-backend --features cpu` → exit 0.
- `cargo test -p mlrs-backend --features cpu --test kmeanspp_test` → 3/3 pass (RNG stream identity preserved after the SplitMix64 move — Pitfall 7).
- `cargo test -p mlrs-backend --features cpu --test lloyd_test` → 4/4 pass.
- `cargo test -p mlrs-backend --features cpu --test rng_test` → 5/5 pass (distribution + byte-identical seed-repro + Achlioptas + permutation + memory gate); 0 ignored.
- `cargo test -p mlrs-backend --features rocm --test rng_test --no-run` → builds (the rocm cross-backend seed-repro check is an opportunistic phase-level gate; f64 cases skip-with-log per D-07).
- Acceptance greps: `pub struct SplitMix64`==1, `use crate::prims::rng`(kmeans)==2, generator-fn count==3, `struct SplitMix64`(kmeans)==0, `#[ignore]`(rng_test)==0. The `next_u64() %`==0 criterion is reconciled above (Rule 1).

## Self-Check: PASSED

All 3 modified source/test files + the SUMMARY exist on disk; both task commits (1a1c19d, 6ee33e9) are present in git history.
