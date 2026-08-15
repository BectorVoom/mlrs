//! Direct pairwise GATHER distance kernels for the KNN-graph primitive
//! (PRIM-11, Phase 13) — the empty-but-registered home plan 13-02 fills.
//!
//! The v1 GEMM-expansion `distance` prim only covers (squared) Euclidean; the
//! L1 / L-infinity / general-exponent metrics cannot be expressed as a GEMM and
//! need direct per-output-element feature-loop kernels. Plan 13-02 lands those
//! `#[cube(launch)]` kernels here (one unit per output pair `(i,j)`, a runtime
//! `while kk < cols` loop over the feature dim) plus the per-row `self_drop`
//! GATHER kernel; this file is the Wave-1 scaffold so `pub mod distance;`
//! compiles today (mirrors the Phase-8/9 Wave-0 stub-registration precedent).
//!
//! ## cpu-MLIR authoring contract (the kernels plan 13-02 adds MUST follow)
//!
//! These kernels are validated under `cubecl-cpu` (the MLIR backend, the f64
//! correctness gate). cpu-MLIR fails LOUDLY outside its proven op-set OR — worse
//! — SILENTLY miscompiles. The contract, distilled from the validated spikes:
//!
//! - **STATIC transcendentals only.** Use the associated form `F::powf(diff, p)`
//!   / `F::powf(acc, inv_p)` for the general-exponent metric — a bounded
//!   feature-loop accumulator plus `F::powf` lowers fine. NEVER the instance
//!   `x.powf()` form (it can mis-lower in the `#[cube]` IR). `.abs()` is the one
//!   instance form that is allowed (jacobi-proven).
//! - **STATEMENT-form running comparison.** The L-infinity running maximum is a
//!   mutable-variable `if` guard (`let mut acc = …; if diff > acc { acc = diff; }`),
//!   NEVER an `if`-expression in value position. Diffs are non-negative so the
//!   `F::from_int(0i64)` seed is correct.
//! - **Per-element 2D launch** for the pairwise kernels: `ABSOLUTE_POS_X` /
//!   `ABSOLUTE_POS_Y` (`u32`) with `CubeDim {x:16, y:16}` and ceiling-div counts,
//!   guarded `if i < rows_x { if j < rows_y { … } }`.
//! - **Per-row GATHER launch** for the self-drop kernel: `CUBE_POS_X` /
//!   `UNIT_POS_X == 0u32` with `CubeCount::Static(n, 1, 1)`, `CubeDim {x:1,y:1,z:1}`
//!   — NEVER a bare 1D `ABSOLUTE_POS` launch (that is a loud MLIR pass failure;
//!   the kernel never runs and reads back zeros).
//! - **No cross-sibling-loop accumulator.** A flag/counter written in one `while`
//!   and read in a SEPARATE sibling `while` SILENTLY miscompiles. Recompute any
//!   per-row positional value with a self-contained nested count inside the
//!   consuming loop (the self-shift `src = s + #self-cols-at-cols-<=-s` idiom).
//! - **`F` / `u32` accumulators only** — no mutable-bool scans. **Banned
//!   entirely** (panic at launch): `SharedMemory`, `Atomic`, the infinity
//!   constant, and descending-shift loops.
//! - Scalar kernel params (dims, the general-exponent value) pass **by value**
//!   in cubecl 0.10 (no `ScalarArg` wrapper).
//!
//! Plan 13-02 adds the kernel bodies AND their `pub use distance::{…}` re-export
//! line in lib.rs as part of that plan's edit (file-disjoint, single-owner).

use cubecl::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Direct pairwise distance kernels (cpu-MLIR-safe; VALIDATED spike-001 shapes).
//
// One unit per output element `(i, j)`; a runtime `while kk < cols` loop over the
// feature dim; only `F`/`u32` accumulators + `if` guards. No SharedMemory, no
// Atomic, no infinity constant, no mutable-bool scan, no descending-shift loop.
// `.abs()` is the jacobi-proven instance form (the one allowed instance form);
// the general-exponent power MUST be the STATIC `F::powf` associated form (the
// instance `x.powf()` can mis-lower in the `#[cube]` IR). Output is row-major
// (`rows_x × rows_y`): `out[i * rows_y + j]`.
// ─────────────────────────────────────────────────────────────────────────────

/// Manhattan (L1) pairwise distance: `out[i*rows_y+j] = sum_k |x_ik - y_jk|`.
///
/// cpu-MLIR contract: per-element 2D launch (`ABSOLUTE_POS_{X,Y}`), bounded
/// feature loop, `F`/`u32` accumulators only; the per-term absolute difference
/// uses the allowed instance `.abs()` form, no root applied.
#[cube(launch)]
pub fn manhattan_dist<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    out: &mut Array<F>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    if i < rows_x {
        if j < rows_y {
            let xb = i * cols;
            let yb = j * cols;
            let mut acc = F::from_int(0i64);
            let mut kk = 0u32;
            while kk < cols {
                let diff = (x[(xb + kk) as usize] - y[(yb + kk) as usize]).abs();
                acc += diff;
                kk += 1u32;
            }
            out[(i * rows_y + j) as usize] = acc;
        }
    }
}

/// Additive chi-squared kernel value (KERNEL-01 full parameter surface):
/// `out[i*rows_y+j] = -sum_k (x_ik - y_jk)^2 / (x_ik + y_jk)`.
///
/// This is sklearn's `additive_chi2_kernel` verbatim — including the sign. The
/// kernel VALUE is the negated chi² statistic, so the sum is accumulated
/// positive and negated once at the store rather than accumulated negative;
/// that keeps the accumulator's magnitude monotone, which is what the f32 path
/// needs when the per-term ratios are wildly different in scale.
///
/// The `nom > 0` guard is sklearn's `if nom != 0` from `_chi2_kernel_fast`, and
/// it is load-bearing rather than defensive: a feature that is zero in BOTH rows
/// contributes `0/0`, and skipping it is the difference between agreeing with
/// sklearn and returning NaN for every pair of rows that share a zero column —
/// which is most pairs, since chi² is a histogram kernel and histograms are
/// sparse. `x >= 0` is a caller obligation (checked host-side, as sklearn's
/// `check_non_negative` does), so `nom != 0` and `nom > 0` coincide and the
/// comparison is written in the form that lowers most predictably.
///
/// cpu-MLIR contract: per-element 2D launch (`ABSOLUTE_POS_{X,Y}`), bounded
/// feature loop, `F`/`u32` accumulators only, STATEMENT-form `if` guard, no
/// transcendental, no SharedMemory, no infinity constant.
#[cube(launch)]
pub fn additive_chi2_dist<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    out: &mut Array<F>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    if i < rows_x {
        if j < rows_y {
            let xb = i * cols;
            let yb = j * cols;
            let zero = F::from_int(0i64);
            let mut acc = F::from_int(0i64);
            let mut kk = 0u32;
            while kk < cols {
                let a = x[(xb + kk) as usize];
                let b = y[(yb + kk) as usize];
                let nom = a + b;
                if nom > zero {
                    let diff = a - b;
                    acc += diff * diff / nom;
                }
                kk += 1u32;
            }
            out[(i * rows_y + j) as usize] = -acc;
        }
    }
}

/// Chebyshev (L-infinity) pairwise distance: `out[i*rows_y+j] = max_k |x_ik - y_jk|`.
///
/// cpu-MLIR contract: the running maximum is a mutable-variable STATEMENT-form
/// `if` guard (`if diff > acc { acc = diff; }`), NEVER an `if`-expression in value
/// position. Per-term differences are non-negative so the `F::from_int(0i64)` seed
/// is correct.
#[cube(launch)]
pub fn chebyshev_dist<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    out: &mut Array<F>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    if i < rows_x {
        if j < rows_y {
            let xb = i * cols;
            let yb = j * cols;
            let mut acc = F::from_int(0i64);
            let mut kk = 0u32;
            while kk < cols {
                let diff = (x[(xb + kk) as usize] - y[(yb + kk) as usize]).abs();
                if diff > acc {
                    acc = diff;
                }
                kk += 1u32;
            }
            out[(i * rows_y + j) as usize] = acc;
        }
    }
}

/// Minkowski-`p` pairwise distance: `out[i*rows_y+j] = (sum_k |x_ik - y_jk|^p)^(1/p)`.
///
/// # Precondition (caller obligation, WR-02)
/// `p >= 1` is a HARD caller precondition: this kernel computes `inv_p = 1/p` with
/// NO in-kernel positive-`p` guard (an in-kernel branch would risk a cpu-MLIR
/// mis-lower and the host already validates `p` typed). A `p == 0` launch divides
/// by zero (→ inf) and then `F::powf(acc, inv_p)` yields inf/NaN distances rather
/// than a typed error. The ONLY supported launch path is through the validated
/// `knn_graph` entry (`validate_geometry` rejects `p < 1` BEFORE any launch); do
/// not launch this kernel directly with unchecked `p`.
///
/// The named cpu-MLIR feasibility unknown for this phase (VALIDATED spike 001):
/// an in-kernel general-exponent power inside the feature-loop accumulator, then a
/// final `^(1/p)` root. cpu-MLIR contract: BOTH powers use the STATIC associated
/// `F::powf(base, exp)` form (the instance form can mis-lower); `p` passes by value
/// (cubecl 0.10 has no `ScalarArg` wrapper). Subsumes L1 (`p=1`) and L2 (`p=2`) per
/// the spike depth probe; fast-path special-casing is an optimization, not a
/// correctness need.
#[cube(launch)]
pub fn minkowski_dist<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    out: &mut Array<F>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
    p: F,
) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    if i < rows_x {
        if j < rows_y {
            let xb = i * cols;
            let yb = j * cols;
            let mut acc = F::from_int(0i64);
            let mut kk = 0u32;
            while kk < cols {
                let diff = (x[(xb + kk) as usize] - y[(yb + kk) as usize]).abs();
                acc += F::powf(diff, p);
                kk += 1u32;
            }
            let inv_p = F::new(1.0_f32) / p;
            out[(i * rows_y + j) as usize] = F::powf(acc, inv_p);
        }
    }
}

/// Cosine pairwise distance: `out[i*rows_y+j] = 1 − (x_i·y_j) / (‖x_i‖·‖y_j‖)`,
/// clamped to `[0, 2]` — sklearn's `cosine_distances` (KNN-REG-PARAMS).
///
/// # Why the norms are arguments, not recomputed
/// `‖x_i‖` depends only on `i` and `‖y_j‖` only on `j`, so recomputing them
/// inside the `(i, j)` loop would triple the feature reads for no reason. The
/// caller precomputes both with the one-launch
/// [`row_sumsq`](crate::knn::row_sumsq) GATHER kernel and passes the SUM OF
/// SQUARES (not the root) — the root is taken once per output element here, so
/// the feeders stay a plain arithmetic reduction.
///
/// # Zero rows (sklearn parity)
/// sklearn normalises with `preprocessing.normalize`, which leaves an all-zero
/// row at zero, so its cosine similarity against anything is `0` and the
/// distance is `1`. A zero norm here therefore yields similarity `0` — NOT a
/// division by zero — via the statement-form `if denom > 0` guard.
///
/// The `[0, 2]` clamp mirrors sklearn's own `np.clip` on the same quantity: the
/// similarity of two unit vectors is analytically in `[-1, 1]`, but the
/// floating-point dot/root can leave it a few ulp outside, and a negative
/// distance would break the `top_k` ordering contract and any `1/d` weighting.
///
/// cpu-MLIR contract: per-element 2D launch (`ABSOLUTE_POS_{X,Y}`), bounded
/// feature loop, `F` accumulators only, STATEMENT-form `if` guards, no
/// `SharedMemory`, no mutable `bool`, no infinity constant. `F::sqrt` is the
/// STATIC associated form (the `sqrt_elem` precedent), not the instance one.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn cosine_dist<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    xnorm: &Array<F>,
    ynorm: &Array<F>,
    out: &mut Array<F>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    if i < rows_x {
        if j < rows_y {
            let xb = i * cols;
            let yb = j * cols;
            let mut dot = F::from_int(0i64);
            let mut kk = 0u32;
            while kk < cols {
                dot += x[(xb + kk) as usize] * y[(yb + kk) as usize];
                kk += 1u32;
            }
            let zero = F::from_int(0i64);
            let one = F::from_int(1i64);
            let two = F::from_int(2i64);
            // `‖x‖·‖y‖ = sqrt(‖x‖²·‖y‖²)` — one root instead of two.
            let denom = F::sqrt(xnorm[i as usize] * ynorm[j as usize]);
            let mut sim = zero;
            if denom > zero {
                sim = dot / denom;
            }
            let mut d = one - sim;
            if d < zero {
                d = zero;
            }
            if d > two {
                d = two;
            }
            out[(i * rows_y + j) as usize] = d;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Self-drop-by-index-identity GATHER kernel (cpu-MLIR-safe; VALIDATED spike-002).
//
// Input is the `top_k(k+1)` result (ascending `(val, idx)` per row); output is the
// `k` true neighbours with the self column (the slot whose index == the query row)
// removed — the `include_self=false` UMAP path (D-02: drop by INDEX IDENTITY, not
// first-zero-distance, so a duplicate point at distance 0 is handled correctly).
// ─────────────────────────────────────────────────────────────────────────────

/// Per-row self-drop GATHER: removes the index-identity self column from a
/// `top_k(k+1)` result, emitting the `k` true neighbours per row.
///
/// cpu-MLIR contract (two VALIDATED landmines this kernel must NOT trip):
/// - **002-A (loud):** launch via `CUBE_POS_X` / `UNIT_POS_X == 0u32` (one cube per
///   query row, one selecting unit) — NEVER a bare 1D `ABSOLUTE_POS` launch, which
///   is a loud MLIR pass failure (the kernel never runs and reads back zeros).
/// - **002-B (silent):** the per-output-slot shift is recomputed LOCALLY via a
///   nested count inside the consuming `while` (`src = s + #self-cols-at-cols-<=-s`)
///   — NEVER a flag/counter written in one `while` and read in a separate sibling
///   `while` (that silently miscompiles under the cube macro).
///
/// Fallback (R-3): if self is absent from the top-`(k+1)` (shouldn't happen for
/// X-vs-X), `bump` stays 0 for every `s` so `src = s`, dropping the last column `k`.
/// Uses only `u32`/`F` accumulators and STATEMENT-form `if`; no mutable bool, no
/// SharedMemory, no infinity constant.
#[cube(launch)]
pub fn self_drop_gather<F: Float + CubeElement>(
    in_val: &Array<F>,
    in_idx: &Array<u32>,
    out_val: &mut Array<F>,
    out_idx: &mut Array<u32>,
    rows: u32,
    k: u32,
    k1: u32, // k + 1
) {
    let row = CUBE_POS_X;
    if row < rows {
        if UNIT_POS_X == 0u32 {
            let ibase = row * k1;
            let obase = row * k;
            let mut s = 0u32;
            while s < k {
                let mut bump = 0u32;
                let mut c = 0u32;
                while c < s + 1u32 {
                    if in_idx[(ibase + c) as usize] == row {
                        bump += 1u32;
                    }
                    c += 1u32;
                }
                // WR-03: clamp the source column into the (k+1)-wide row so a
                // self index that (unexpectedly) appears more than once cannot push
                // `src = s + bump` past the row end (`in_idx[ibase + k + 1]`) — an
                // OOB device read for the last row. For the single-self-occurrence
                // X-vs-X invariant `src < k1` always holds, so this clamp is inert
                // there; it is defense-in-depth, not a behavior change. STATEMENT-
                // form mutable-`if` guard (cpu-MLIR-safe, no if-expr in value pos).
                let mut src = s + bump;
                if src >= k1 {
                    src = k1 - 1u32;
                }
                out_val[(obase + s) as usize] = in_val[(ibase + src) as usize];
                out_idx[(obase + s) as usize] = in_idx[(ibase + src) as usize];
                s += 1u32;
            }
        }
    }
}

/// Squared-Euclidean (L2²) pairwise distance computed DIRECTLY:
/// `out[i*rows_y+j] = sum_k (x_ik - y_jk)²`.
///
/// ## Why a direct kernel when `prims::distance` already computes L2 (KNN-01)
/// The shared L2 path is the GEMM-expansion `‖x‖² + ‖y‖² − 2·XYᵀ`, which routes
/// the cross term through `cubek-matmul`. That is the right shape when the
/// contraction dimension is large, but the pairwise-distance shape has
/// `k = n_features` — 16 or 32 for a typical KNN problem — against an enormous
/// `m × n` output. Measured on a Tesla T4, that GEMM sustained roughly
/// 0.1 GFLOP/s on a device capable of thousands: 2.28 s for a 2_000 × 10_000
/// distance matrix at `d = 16`, which was ~99% of the whole KNN predict after the
/// selection kernel was fixed. It is the same tiny-`K`/huge-output pathology that
/// `gram.rs` and `colmean.rs` were written to escape elsewhere in this crate.
///
/// This kernel does the obvious thing instead: one unit per OUTPUT element, a
/// bounded loop over the `cols` features. Total arithmetic is identical, but it
/// is perfectly parallel over `rows_x × rows_y` with no matmul machinery, no
/// `XYᵀ` intermediate, and no separate norm reductions.
///
/// ## Accuracy: better where it matters, comparable elsewhere
/// `‖x‖² + ‖y‖² − 2·x·y` is a catastrophic cancellation for near-identical rows
/// — that is precisely why the expansion path needs its unconditional
/// `max(d², 0)` clamp (RESEARCH Pitfall 5). The direct form sums squares of
/// differences, so every term is non-negative: the result can NEVER go negative
/// (no clamp needed) and the near-zero distances are recovered without
/// cancellation. Those are exactly the pairs a nearest-neighbor search selects.
///
/// This is NOT a claim of uniformly higher accuracy: over well-separated pairs
/// the two routes are comparable and neither dominates (measured worst-case
/// absolute error over a random 37×91 block at `d = 16`: direct 3.2e-6, expansion
/// 2.3e-6). The guarantees the direct form does provide are non-negativity and
/// the cancellation case; both are pinned by
/// `knn_test.rs::distance_direct_matches_gemm_expansion`.
///
/// Returns the SQUARED distance (the order-preserving form top-k selects on); the
/// caller applies the optional sqrt at the boundary, exactly as with the
/// expansion path.
///
/// cpu-MLIR contract: per-element 2D launch (`ABSOLUTE_POS_{X,Y}`), bounded
/// feature loop, `F` accumulator only, no `SharedMemory`, no mutable `bool` —
/// the `manhattan_dist` shape with a squared term instead of `.abs()`.
#[cube(launch)]
pub fn euclidean_sq_dist<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    out: &mut Array<F>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    if i < rows_x {
        if j < rows_y {
            let xb = i * cols;
            let yb = j * cols;
            let mut acc = F::from_int(0i64);
            let mut kk = 0u32;
            while kk < cols {
                let diff = x[(xb + kk) as usize] - y[(yb + kk) as usize];
                acc += diff * diff;
                kk += 1u32;
            }
            out[(i * rows_y + j) as usize] = acc;
        }
    }
}

/// Squared-Euclidean pairwise distance with SHARED-MEMORY TILING — the data-reuse
/// form of [`euclidean_sq_dist`] (KNN-01 root-cause fix).
///
/// ## The defect this fixes
/// [`euclidean_sq_dist`] gives every OUTPUT element its own unit, and that unit
/// walks `x[i, ..]` and `y[j, ..]` in full out of GLOBAL memory. Global traffic is
/// therefore `rows_x × rows_y × cols × 2 × 4` bytes — it scales with `cols`, and
/// nothing is ever reused. Measured on a Tesla T4, the kernel lands within
/// **3–8% of exactly that traffic model** at three different problem sizes
/// (0.215s / 0.821s / 3.293s against a predicted 0.200 / 0.800 / 3.200), i.e. it
/// is perfectly memory-saturated for a pathologically redundant access pattern.
/// The matrix write everyone assumes is the problem accounts for only ~3% of it.
/// A `cols` sweep at fixed output size confirms the same thing directly: runtime
/// grows ~linearly with `cols` (6.9× from `cols=8` to `cols=128`) where a
/// matrix-bound kernel would be flat.
///
/// ## The fix
/// One CUBE computes a `TILE × TILE` block of the output. For each `TILE`-wide
/// slice of the feature dimension, the cube cooperatively stages `TILE` rows of
/// `x` and `TILE` rows of `y` into SharedMemory ONCE, then all `TILE × TILE` units
/// accumulate out of shared memory. Global reads per output tile drop from
/// `2 × TILE² × cols` to `2 × TILE × cols` — a **`TILE`-fold (16×) traffic
/// reduction**, which is the whole gap to cuML's `fusedL2NN`.
///
/// ## Coalescing
/// Both the staging loads and the final store are arranged so the FASTEST-varying
/// unit index walks CONTIGUOUS global addresses:
/// - loads index the feature (`k`) axis with `UNIT_POS_X`, so a warp reads one
///   contiguous run of a row rather than striding by `cols`;
/// - the store maps `UNIT_POS_X` to `j` (the minor axis of the row-major output),
///   so a warp writes one contiguous run of `out` rather than striding by
///   `rows_y`. This is why `CUBE_POS_X` blocks the `rows_y` axis and `CUBE_POS_Y`
///   the `rows_x` axis — the host launch config must match.
///
/// ## Shared-memory layout is TRANSPOSED to avoid bank conflicts
/// The tiles are staged as `[k][row]`, not `[row][k]`. With the natural
/// `[row][k]` layout the inner-loop read `ys[UNIT_POS_X * 16 + kk]` has the 16
/// consecutive lanes of a warp addressing 16 floats apart, which collapses onto
/// two banks (~8-way conflict) and serializes every shared read. Transposed, that
/// read becomes `ys[kk * 16 + UNIT_POS_X]` — 16 CONSECUTIVE addresses, so it is
/// conflict-free — while `xs[kk * 16 + UNIT_POS_Y]` resolves to two addresses per
/// warp and broadcasts. The staging writes take the strided pattern instead, but
/// they run once per 16 accumulate steps, so the trade is heavily favourable.
///
/// Numerically identical accumulation ORDER to [`euclidean_sq_dist`] (ascending
/// `k`, one `F` accumulator), so results agree bit for bit.
///
/// cpu-MLIR contract: `SharedMemory` + `sync_cube` (the `reduce.rs::argmin_shared`
/// idiom), `F`/`u32` accumulators, STATEMENT-form `if` guards only, no mutable
/// `bool`. Every `sync_cube` is outside any non-uniform branch: the `k` loop is
/// driven by the scalar `cols`, so all units execute the same barrier count.
#[cube(launch)]
pub fn euclidean_sq_dist_tiled<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    out: &mut Array<F>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
) {
    // TILE = 16, matching the launch config's CubeDim {x: 16, y: 16}.
    let mut xs = SharedMemory::<F>::new(256usize);
    let mut ys = SharedMemory::<F>::new(256usize);

    // UNIT_POS_X is the fastest-varying index and is mapped to the OUTPUT's minor
    // axis (j) so the final store is coalesced.
    let j = CUBE_POS_X * 16u32 + UNIT_POS_X;
    let i = CUBE_POS_Y * 16u32 + UNIT_POS_Y;

    // Staging roles: UNIT_POS_Y picks which row of the tile this unit loads,
    // UNIT_POS_X picks the feature offset — so a warp reads contiguous memory.
    let load_row = UNIT_POS_Y;
    let load_k = UNIT_POS_X;
    let x_row = CUBE_POS_Y * 16u32 + load_row;
    let y_row = CUBE_POS_X * 16u32 + load_row;

    let mut acc = F::from_int(0i64);
    let mut k0 = 0u32;
    while k0 < cols {
        let kc = k0 + load_k;

        // Stage x rows. Out-of-range lanes stage 0; those contributions are
        // discarded either by the `kc < cols` compute guard or by the final
        // bounds-checked store, so the pad value never reaches an output.
        let mut xv = F::from_int(0i64);
        if x_row < rows_x {
            if kc < cols {
                xv = x[(x_row * cols + kc) as usize];
            }
        }
        // TRANSPOSED staging layout `[k][row]`, not `[row][k]` — see the bank
        // -conflict note in the docs.
        xs[(load_k * 16u32 + load_row) as usize] = xv;

        let mut yv = F::from_int(0i64);
        if y_row < rows_y {
            if kc < cols {
                yv = y[(y_row * cols + kc) as usize];
            }
        }
        ys[(load_k * 16u32 + load_row) as usize] = yv;

        sync_cube();

        // Accumulate this feature slice entirely out of shared memory. Unit
        // (UNIT_POS_X, UNIT_POS_Y) owns output (i, j), so it reads the x tile row
        // for `i` and the y tile row for `j`.
        let mut kk = 0u32;
        while kk < 16u32 {
            if k0 + kk < cols {
                let diff =
                    xs[(kk * 16u32 + UNIT_POS_Y) as usize] - ys[(kk * 16u32 + UNIT_POS_X) as usize];
                acc += diff * diff;
            }
            kk += 1u32;
        }

        // Barrier before the next slice overwrites the staged tiles.
        sync_cube();
        k0 += 16u32;
    }

    if i < rows_x {
        if j < rows_y {
            out[(i * rows_y + j) as usize] = acc;
        }
    }
}

/// Squared-Euclidean pairwise distance with shared-memory tiling AND 2×2 REGISTER
/// BLOCKING — the arithmetic-intensity form of [`euclidean_sq_dist_tiled`]
/// (KNN-01, third iteration).
///
/// ## Why, after tiling already landed
/// [`euclidean_sq_dist_tiled`] cut global traffic and won 1.4–2.4× on a T4, but it
/// still sat ~4.6× above its OWN roofline. Tiling fixes global traffic; it does
/// not fix the fact that each unit performs one FMA per **two** shared-memory
/// loads (`xs[..]` and `ys[..]`), so the inner loop is bound by shared-memory
/// throughput rather than by arithmetic.
///
/// Register blocking raises that ratio: each unit owns a 2×2 block of outputs, so
/// the 4 values it loads per feature step (2 from `xs`, 2 from `ys`) feed **4**
/// FMAs instead of 1 — a 2× better load-to-FMA ratio, with the 4 accumulators
/// living in registers. It also doubles the output block per cube from 16×16 to
/// 32×32, which halves global traffic again (reads per output element fall from
/// `d/8` to `d/16`).
///
/// ## Index mapping (coalescing + bank conflicts, both preserved)
/// The two outputs a unit owns along each axis are **16 apart**, not adjacent:
/// `j ∈ {base_j + ux, base_j + 16 + ux}`. That is deliberate — it keeps the
/// fastest-varying unit index `ux` mapped to consecutive `j`, so each store is
/// still a contiguous run, and it keeps the shared read `ys[k*32 + jj*16 + ux]`
/// on 16 consecutive addresses (conflict-free) while `xs[k*32 + ii*16 + uy]`
/// broadcasts across the warp. Tiles stay staged `[k][row]` for the same reason
/// as [`euclidean_sq_dist_tiled`].
///
/// Accumulation is still ascending `k` into one accumulator per output, so results
/// are BITWISE identical to both [`euclidean_sq_dist`] and
/// [`euclidean_sq_dist_tiled`].
///
/// cpu-MLIR contract: `SharedMemory` + `sync_cube`, `F`/`u32` accumulators,
/// STATEMENT-form `if` guards, no mutable `bool`. Barriers sit outside every
/// non-uniform branch (the `k` loop is driven by the scalar `cols`).
#[cube(launch)]
pub fn euclidean_sq_dist_rb<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    out: &mut Array<F>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
) {
    // 32 rows staged per operand per 16-wide feature slice, `[k][row]` layout.
    let mut xs = SharedMemory::<F>::new(512usize);
    let mut ys = SharedMemory::<F>::new(512usize);

    let ux = UNIT_POS_X;
    let uy = UNIT_POS_Y;
    let base_i = CUBE_POS_Y * 32u32;
    let base_j = CUBE_POS_X * 32u32;

    // Staging roles: `ux` indexes the feature axis so a warp reads contiguously.
    let load_k = ux;
    let load_r0 = uy;
    let load_r1 = uy + 16u32;

    let mut acc00 = F::from_int(0i64);
    let mut acc01 = F::from_int(0i64);
    let mut acc10 = F::from_int(0i64);
    let mut acc11 = F::from_int(0i64);

    let mut k0 = 0u32;
    while k0 < cols {
        let kc = k0 + load_k;

        // Stage both 32-row halves of each operand. Out-of-range lanes stage 0;
        // those values are discarded by the `k0 + kk < cols` compute guard or by
        // the bounds-checked stores, so a pad never reaches an output.
        let mut xv0 = F::from_int(0i64);
        if base_i + load_r0 < rows_x {
            if kc < cols {
                xv0 = x[((base_i + load_r0) * cols + kc) as usize];
            }
        }
        xs[(load_k * 32u32 + load_r0) as usize] = xv0;

        let mut xv1 = F::from_int(0i64);
        if base_i + load_r1 < rows_x {
            if kc < cols {
                xv1 = x[((base_i + load_r1) * cols + kc) as usize];
            }
        }
        xs[(load_k * 32u32 + load_r1) as usize] = xv1;

        let mut yv0 = F::from_int(0i64);
        if base_j + load_r0 < rows_y {
            if kc < cols {
                yv0 = y[((base_j + load_r0) * cols + kc) as usize];
            }
        }
        ys[(load_k * 32u32 + load_r0) as usize] = yv0;

        let mut yv1 = F::from_int(0i64);
        if base_j + load_r1 < rows_y {
            if kc < cols {
                yv1 = y[((base_j + load_r1) * cols + kc) as usize];
            }
        }
        ys[(load_k * 32u32 + load_r1) as usize] = yv1;

        sync_cube();

        // 4 shared loads feed 4 FMAs (the 1×1 kernel needs 2 loads per FMA).
        let mut kk = 0u32;
        while kk < 16u32 {
            if k0 + kk < cols {
                let xa = xs[(kk * 32u32 + uy) as usize];
                let xb = xs[(kk * 32u32 + 16u32 + uy) as usize];
                let ya = ys[(kk * 32u32 + ux) as usize];
                let yb = ys[(kk * 32u32 + 16u32 + ux) as usize];

                let d00 = xa - ya;
                acc00 += d00 * d00;
                let d01 = xa - yb;
                acc01 += d01 * d01;
                let d10 = xb - ya;
                acc10 += d10 * d10;
                let d11 = xb - yb;
                acc11 += d11 * d11;
            }
            kk += 1u32;
        }

        sync_cube();
        k0 += 16u32;
    }

    let i0 = base_i + uy;
    let i1 = base_i + 16u32 + uy;
    let j0 = base_j + ux;
    let j1 = base_j + 16u32 + ux;

    if i0 < rows_x {
        if j0 < rows_y {
            out[(i0 * rows_y + j0) as usize] = acc00;
        }
        if j1 < rows_y {
            out[(i0 * rows_y + j1) as usize] = acc01;
        }
    }
    if i1 < rows_x {
        if j0 < rows_y {
            out[(i1 * rows_y + j0) as usize] = acc10;
        }
        if j1 < rows_y {
            out[(i1 * rows_y + j1) as usize] = acc11;
        }
    }
}

/// Squared-Euclidean pairwise distance with 4×4 REGISTER BLOCKING over 32-wide
/// feature slices — the highest-intensity variant (KNN-01, fourth iteration).
///
/// ## Why, after 2×2 blocking already landed
/// [`euclidean_sq_dist_rb`] reached 4 FMAs per 4 shared loads and still measured
/// ~5.8× above its own roofline on a T4. Two things are still left on the table:
///
/// 1. **Arithmetic intensity.** A 4×4 block turns 8 shared loads into **16**
///    FMAs — 2 FMA/load, double the 2×2 kernel's ratio.
/// 2. **Global traffic and load width.** A 16×16 cube now covers a **64×64**
///    output block, halving reads per output again (`d/16` → `d/32`), and the
///    feature slice widens from 16 to **32**, so each staged row is a 128-byte
///    contiguous run — a full coalesced transaction rather than a 64-byte half.
///
/// Shared use is 2 × 32 × 64 × 4 B = 16 KB per cube, which still allows 4 cubes
/// (1024 units) resident per T4 SM.
///
/// ## Staging map
/// The 256 units stage 64 rows × 32 features per operand in 8 passes. Pass `L`
/// has unit `(ux, uy)` load row `L*8 + uy/2` at feature `ux + 16*(uy%2)`, so the
/// 32 consecutive units of a warp read 32 consecutive floats of ONE row. Tiles are
/// staged `[k][row]` so the inner-loop `ys[k*64 + jj*16 + ux]` read hits 16
/// consecutive addresses (conflict-free) while `xs[..+uy]` broadcasts — the same
/// discipline as [`euclidean_sq_dist_tiled`], and the outputs a unit owns are
/// again **16 apart** so the store stays coalesced.
///
/// Accumulation is ascending `k` into one accumulator per output, so results are
/// BITWISE identical to all three other squared-Euclidean kernels.
///
/// cpu-MLIR contract: `SharedMemory` + `sync_cube`, `F`/`u32` accumulators,
/// STATEMENT-form `if` guards, no mutable `bool`; barriers outside every
/// non-uniform branch.
#[cube(launch)]
pub fn euclidean_sq_dist_rb4<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    out: &mut Array<F>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
) {
    // [k][row] layout: 32 feature slots × 64 rows.
    let mut xs = SharedMemory::<F>::new(2048usize);
    let mut ys = SharedMemory::<F>::new(2048usize);

    let ux = UNIT_POS_X;
    let uy = UNIT_POS_Y;
    let base_i = CUBE_POS_Y * 64u32;
    let base_j = CUBE_POS_X * 64u32;

    // Warp-contiguous staging coordinates (see the staging map in the docs).
    let load_k = ux + 16u32 * (uy % 2u32);
    let load_r_base = uy / 2u32;

    let mut acc00 = F::from_int(0i64);
    let mut acc01 = F::from_int(0i64);
    let mut acc02 = F::from_int(0i64);
    let mut acc03 = F::from_int(0i64);
    let mut acc10 = F::from_int(0i64);
    let mut acc11 = F::from_int(0i64);
    let mut acc12 = F::from_int(0i64);
    let mut acc13 = F::from_int(0i64);
    let mut acc20 = F::from_int(0i64);
    let mut acc21 = F::from_int(0i64);
    let mut acc22 = F::from_int(0i64);
    let mut acc23 = F::from_int(0i64);
    let mut acc30 = F::from_int(0i64);
    let mut acc31 = F::from_int(0i64);
    let mut acc32 = F::from_int(0i64);
    let mut acc33 = F::from_int(0i64);

    let mut k0 = 0u32;
    while k0 < cols {
        let kc = k0 + load_k;

        let mut l = 0u32;
        while l < 8u32 {
            let r = l * 8u32 + load_r_base;

            let mut xv = F::from_int(0i64);
            if base_i + r < rows_x {
                if kc < cols {
                    xv = x[((base_i + r) * cols + kc) as usize];
                }
            }
            xs[(load_k * 64u32 + r) as usize] = xv;

            let mut yv = F::from_int(0i64);
            if base_j + r < rows_y {
                if kc < cols {
                    yv = y[((base_j + r) * cols + kc) as usize];
                }
            }
            ys[(load_k * 64u32 + r) as usize] = yv;

            l += 1u32;
        }

        sync_cube();

        // 8 shared loads feed 16 FMAs.
        let mut kk = 0u32;
        while kk < 32u32 {
            if k0 + kk < cols {
                let xa0 = xs[(kk * 64u32 + 0u32 + uy) as usize];
                let xa1 = xs[(kk * 64u32 + 16u32 + uy) as usize];
                let xa2 = xs[(kk * 64u32 + 32u32 + uy) as usize];
                let xa3 = xs[(kk * 64u32 + 48u32 + uy) as usize];
                let yb0 = ys[(kk * 64u32 + 0u32 + ux) as usize];
                let yb1 = ys[(kk * 64u32 + 16u32 + ux) as usize];
                let yb2 = ys[(kk * 64u32 + 32u32 + ux) as usize];
                let yb3 = ys[(kk * 64u32 + 48u32 + ux) as usize];
                let d00 = xa0 - yb0;
                acc00 += d00 * d00;
                let d01 = xa0 - yb1;
                acc01 += d01 * d01;
                let d02 = xa0 - yb2;
                acc02 += d02 * d02;
                let d03 = xa0 - yb3;
                acc03 += d03 * d03;
                let d10 = xa1 - yb0;
                acc10 += d10 * d10;
                let d11 = xa1 - yb1;
                acc11 += d11 * d11;
                let d12 = xa1 - yb2;
                acc12 += d12 * d12;
                let d13 = xa1 - yb3;
                acc13 += d13 * d13;
                let d20 = xa2 - yb0;
                acc20 += d20 * d20;
                let d21 = xa2 - yb1;
                acc21 += d21 * d21;
                let d22 = xa2 - yb2;
                acc22 += d22 * d22;
                let d23 = xa2 - yb3;
                acc23 += d23 * d23;
                let d30 = xa3 - yb0;
                acc30 += d30 * d30;
                let d31 = xa3 - yb1;
                acc31 += d31 * d31;
                let d32 = xa3 - yb2;
                acc32 += d32 * d32;
                let d33 = xa3 - yb3;
                acc33 += d33 * d33;
            }
            kk += 1u32;
        }

        sync_cube();
        k0 += 32u32;
    }

    let i0 = base_i + 0u32 + uy;
    let i1 = base_i + 16u32 + uy;
    let i2 = base_i + 32u32 + uy;
    let i3 = base_i + 48u32 + uy;
    let j0 = base_j + 0u32 + ux;
    let j1 = base_j + 16u32 + ux;
    let j2 = base_j + 32u32 + ux;
    let j3 = base_j + 48u32 + ux;
    if i0 < rows_x {
        if j0 < rows_y {
            out[(i0 * rows_y + j0) as usize] = acc00;
        }
        if j1 < rows_y {
            out[(i0 * rows_y + j1) as usize] = acc01;
        }
        if j2 < rows_y {
            out[(i0 * rows_y + j2) as usize] = acc02;
        }
        if j3 < rows_y {
            out[(i0 * rows_y + j3) as usize] = acc03;
        }
    }
    if i1 < rows_x {
        if j0 < rows_y {
            out[(i1 * rows_y + j0) as usize] = acc10;
        }
        if j1 < rows_y {
            out[(i1 * rows_y + j1) as usize] = acc11;
        }
        if j2 < rows_y {
            out[(i1 * rows_y + j2) as usize] = acc12;
        }
        if j3 < rows_y {
            out[(i1 * rows_y + j3) as usize] = acc13;
        }
    }
    if i2 < rows_x {
        if j0 < rows_y {
            out[(i2 * rows_y + j0) as usize] = acc20;
        }
        if j1 < rows_y {
            out[(i2 * rows_y + j1) as usize] = acc21;
        }
        if j2 < rows_y {
            out[(i2 * rows_y + j2) as usize] = acc22;
        }
        if j3 < rows_y {
            out[(i2 * rows_y + j3) as usize] = acc23;
        }
    }
    if i3 < rows_x {
        if j0 < rows_y {
            out[(i3 * rows_y + j0) as usize] = acc30;
        }
        if j1 < rows_y {
            out[(i3 * rows_y + j1) as usize] = acc31;
        }
        if j2 < rows_y {
            out[(i3 * rows_y + j2) as usize] = acc32;
        }
        if j3 < rows_y {
            out[(i3 * rows_y + j3) as usize] = acc33;
        }
    }
}
