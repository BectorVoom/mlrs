//! `linear_predict` — fused on-device linear-model inference kernel
//! (LINEAR-01/02 predict perf lever).
//!
//! Feature-free `#[cube]` kernel generic over `<F: Float + CubeElement>`,
//! launched by `mlrs_backend::prims::linear_predict`.
//!
//! ## Why this exists (the predict-side host-sync pathology)
//! Every dense linear regressor (`LinearRegression`, `Ridge`, `Lasso`,
//! `ElasticNet`) shared ONE predict body: form `raw = X·coef` via the generic
//! tiled `gemm` (a skinny `m×1` output), then broadcast-add the scalar
//! intercept. That broadcast was done on the HOST — `intercept.to_host()`
//! (a blocking scalar readback), then `raw.to_host()` (an `m`-length device→host
//! copy), then an element-wise host loop, then `DeviceArray::from_host()` (an
//! `m`-length host→device copy BACK). The PyO3 boundary then reads the result
//! to host ONE more time. On a discrete GPU across PCIe those round-trips — not
//! the arithmetic — dominate `predict`, exactly like `center`'s per-column
//! readback pathology (see `mlrs_kernels::colmean` module docs) and `gram`'s
//! skinny-GEMM starvation (see `mlrs_kernels::gram`).
//!
//! [`linear_predict_bias`] collapses the whole predict into a SINGLE launch
//! that stays device-resident end-to-end: one unit per output row computes
//! `y[r] = Σ_c X[r,c]·coef[c] + bias`, reading the intercept straight from its
//! length-1 device buffer (no scalar readback) and writing the length-`m`
//! result the caller materializes with its one unavoidable readback. The
//! feature axis these models fit is small and capped
//! (`GRAM_EIG_MAX_FEATURES = 64`), so the per-row dot loop is short and the
//! row-major (mildly uncoalesced) column stride is absorbed by L2 — the win is
//! the eliminated PCIe round-trips and the fused bias, not the FLOPs.
//!
//! ## cubecl-cpu MLIR safety
//! GATHER-only: no `SharedMemory`, no atomics, no mutable `bool` — an ascending
//! `while` scan over `F` accumulators. Safe on EVERY backend (cpu included), so
//! `prims::linear_predict` needs no cpu fallback (unlike the `SharedMemory`
//! `gram`/`colmean` perf kernels).

use cubecl::prelude::*;

/// Fused linear-model inference: `out[r] = Σ_c x[r,c]·coef[c] + bias[0]`.
///
/// - `x` is the `m × n` row-major test matrix, `coef` the length-`n` fitted
///   coefficients, `bias` a length-1 device buffer holding the intercept
///   (`0` for the fit-intercept-`false` case — the caller always supplies a
///   real length-1 buffer, so there is no branch here).
/// - One unit per output row (`r < m`); the slack lanes of the final block are
///   masked by the `r < m` guard. The dot product accumulates in `F`, matching
///   the precision of the `gemm` path it replaces (the fitted feature count is
///   small and capped, so a sequential `F` sum stays within the 1e-5 oracle
///   contract).
#[cube(launch)]
pub fn linear_predict_bias<F: Float + CubeElement>(
    x: &Array<F>,
    coef: &Array<F>,
    bias: &Array<F>,
    out: &mut Array<F>,
    m: u32,
    n: u32,
) {
    let r = ABSOLUTE_POS;
    if r < m as usize {
        let base = r * n as usize;
        let mut acc = F::new(0.0_f32);
        let mut c = 0u32;
        while c < n {
            acc += x[base + c as usize] * coef[c as usize];
            c += 1u32;
        }
        out[r] = acc + bias[0];
    }
}

/// Multi-target fused linear-model inference (RIDGE-MULTI-TARGET):
/// `out[r,t] = Σ_c x[r,c]·coef[c,t] + bias[t]`, `coef` row-major `n × k` and
/// `bias` length `k` (one intercept per target).
///
/// One unit per output ROW (`r < m`), same as [`linear_predict_bias`]; the unit
/// loops over its `k` targets and, for each, re-walks the row's `n` features.
/// This re-reads `x[r,·]` from global memory `k` times instead of caching it in
/// registers across targets — `k` (fitted target count) is small in every
/// realistic multi-output regression (a handful to a few dozen columns) and `n`
/// is already capped at `GRAM_EIG_MAX_FEATURES = 64`, so the row a warp
/// re-reads is short and stays hot in L1/L2 across the `k` passes. Caching the
/// row in a per-unit scratch buffer (mirroring [`linear_predict_bias_shared`]'s
/// staged tile) is a real lever for large `k` and is deliberately NOT taken here
/// — CubeCL local arrays need a comptime bound and `k` is a runtime value, so
/// caching would need either a `SharedMemory` staging scheme (barriers, adapter
/// SLM budget checks — the `linear_predict_bias_shared` precedent) or a capped
/// fixed-size local array; both are follow-up work, not required for
/// correctness or for beating a host matvec, which is what this kernel exists
/// to do.
///
/// ## cubecl-cpu MLIR safety
/// GATHER-only, same as [`linear_predict_bias`]: no `SharedMemory`, no atomics,
/// nested ascending `while` scans over `F` accumulators. Safe on every backend.
#[cube(launch)]
pub fn linear_predict_bias_multi<F: Float + CubeElement>(
    x: &Array<F>,
    coef: &Array<F>,
    bias: &Array<F>,
    out: &mut Array<F>,
    m: u32,
    n: u32,
    k: u32,
) {
    let r = ABSOLUTE_POS;
    if r < m as usize {
        let base = r * n as usize;
        let mut t = 0u32;
        while t < k {
            let mut acc = F::new(0.0_f32);
            let mut c = 0u32;
            while c < n {
                acc += x[base + c as usize] * coef[(c * k + t) as usize];
                c += 1u32;
            }
            out[r * k as usize + t as usize] = acc + bias[t as usize];
            t += 1u32;
        }
    }
}

/// Rows per cube for the coalesced [`linear_predict_bias_shared`] tile. Also the
/// cube's thread count (`CubeDim.x`), so each thread finalizes exactly one row.
/// Kept `64` (2 warps) to match the `gram_xty_shared` 64-thread cube that wins
/// on the T4, and to bound the padded f64 tile below the 48 KiB shared budget:
/// `PREDICT_ROWS_PER_BLOCK * (PREDICT_MAX_FEATURES + 1) * 8 = 64·65·8 = 33.3 KiB`.
pub const PREDICT_ROWS_PER_BLOCK: u32 = 64;

/// Feature-count floor below which the coalesced shared-tile kernel is NOT
/// worth its fixed per-cube cost (the 64-thread cube + the comptime `64·65`
/// padded shared tile lower occupancy relative to the 256-thread GATHER cube).
/// Below this `n`, the GATHER kernel's row stride (`n·4 ≤ 64` bytes for f32,
/// at most half a 128-byte cache line) still coalesces two-plus rows per line,
/// so it stays ahead. The value is the measured wgpu crossover: on the perf
/// ladder the shared kernel LOSES at `n = 16` (~2.5×) but WINS from `n = 24`
/// up (2.5–3.5×). A discrete GPU (the T4 target) has a harsher uncoalesced
/// penalty than this env's unified-memory iGPU, so its crossover is no higher —
/// gating here is conservative (it never regresses the measured backend and
/// still captures the full `n = 24..64` win, including the `n = 64`
/// worst-case that motivated the kernel).
pub const PREDICT_SHARED_MIN_FEATURES: u32 = 24;

/// The fitted feature-count ceiling this shared kernel is sized for — the same
/// `GRAM_EIG_MAX_FEATURES = 64` cap the dense linear `fit` path enforces, so a
/// fitted model never exceeds it. The host (`prims::linear_predict`) routes any
/// `n > PREDICT_MAX_FEATURES` back to the GATHER kernel (defensive — the padded
/// shared tile is a comptime `64·65` allocation).
pub const PREDICT_MAX_FEATURES: u32 = 64;

/// Padded per-row stride of the shared X tile (`n + 1`, evaluated at the
/// `PREDICT_MAX_FEATURES` ceiling). The `+ 1` makes the compute-phase stride
/// `65` — coprime to the 32 shared-memory banks — so the `d²`-free per-row dot
/// reads `shm_x[row·65 + c]` conflict-free across the 32 lanes of a warp (for
/// a fixed `c`, lanes `t = 0..31` hit banks `(t·65) mod 32 = t`, a perfect
/// permutation). Without the pad, an even `n` (worst case `n = 64`, stride a
/// multiple of 32) collapses all 32 lanes onto ONE bank — a 32-way conflict.
const PREDICT_TILE_STRIDE: u32 = PREDICT_MAX_FEATURES + 1;

/// Element count of the padded shared X tile (`64 · 65 = 4160`).
const PREDICT_TILE_ELEMS: usize =
    (PREDICT_ROWS_PER_BLOCK * PREDICT_TILE_STRIDE) as usize;

/// Total per-cube `SharedMemory` element count [`linear_predict_bias_shared`]
/// allocates: the padded X tile ([`PREDICT_TILE_ELEMS`] = `64·65`) plus the
/// length-64 coef stage. Multiply by `size_of::<F>()` for the byte footprint.
/// Exported so the host dispatcher (`prims::linear_predict`) can check this
/// against the ADAPTER's `max_shared_memory_size` before launching on wgpu —
/// the CUDA 48 KiB budget the tile was sized for does NOT bound a wgpu adapter
/// (WebGPU downlevel default is 16 KiB), and this kernel now runs only on wgpu.
/// Derived from the SAME named constants the two `SharedMemory::new` calls below
/// use (`PREDICT_TILE_ELEMS` + the length-`PREDICT_MAX_FEATURES` coef stage), so
/// it stays in sync by construction as long as those calls stay symbol-driven
/// (never a hardcoded literal).
pub const PREDICT_SHARED_ELEMS: usize = PREDICT_TILE_ELEMS + PREDICT_MAX_FEATURES as usize;

/// Coalesced, shared-staged fused inference: the CUDA/wgpu perf variant of
/// [`linear_predict_bias`] — same `out[r] = Σ_c x[r,c]·coef[c] + bias[0]`, but
/// laid out to hit peak memory bandwidth on a discrete GPU.
///
/// ## Why this exists (the predict-side UNCOALESCED-read pathology)
/// [`linear_predict_bias`] assigns ONE thread per output row over the `m × n`
/// ROW-MAJOR `x`. At loop step `c`, the 32 threads of a warp (rows `r..r+31`)
/// read `x[(r+k)·n + c]` — addresses `n` elements apart. For the fitted
/// feature counts these models produce (`n` up to `64`), that stride spans a
/// full cache line PER THREAD, so each 32-byte memory sector delivers ONE
/// useful float instead of eight: the warp issues ~8–32× the memory
/// transactions a coalesced read would, and the kernel — which is purely
/// bandwidth-bound (every `x` element is read exactly once, no reuse) — runs at
/// a fraction of peak. The wgpu perf probe shows the signature directly:
/// `100000×64` predict costs ~10× more per element than `1000000×16` despite
/// less data, precisely because `n = 64` is the worst-case stride.
///
/// ## The fix — coalesced load + conflict-free compute
/// Each cube owns a block of [`PREDICT_ROWS_PER_BLOCK`] consecutive rows and
/// stages that `B × n` slab of `x` into `SharedMemory` with a FULLY COALESCED
/// streaming load (thread `t` reads the contiguous global element `row0·n + t`,
/// then `+ CUBE_DIM_X`, …), so the global reads now hit peak bandwidth. The
/// slab is written at the padded stride [`PREDICT_TILE_STRIDE`] so the per-row
/// dot in the compute phase is shared-bank-conflict-free (see that constant).
/// `coef` (length `n ≤ 64`, reused by every row) is staged once into a second
/// shared buffer. Thread `t` then finalizes row `row0 + t` from shared. The net
/// effect is the SAME useful bytes moved once, but coalesced — the arithmetic
/// is unchanged, so the result is bit-for-bit the GATHER kernel's within the
/// `F`-order sum (validated against the same host reference).
///
/// ## cubecl-cpu MLIR safety
/// Uses `SharedMemory` (like `gram_xty_shared` / `reduce_*_shared`) — the host
/// (`prims::linear_predict`) gates the cpu backend back to the GATHER kernel
/// (the `use_shared_gram` `#[cfg(feature = "cpu")]` precedent), so this variant
/// only ever runs on wgpu / cuda / rocm.
#[cube(launch)]
pub fn linear_predict_bias_shared<F: Float + CubeElement>(
    x: &Array<F>,
    coef: &Array<F>,
    bias: &Array<F>,
    out: &mut Array<F>,
    m: u32,
    n: u32,
    nblocks: u32,
) {
    let mut shm_x = SharedMemory::<F>::new(PREDICT_TILE_ELEMS);
    let mut shm_coef = SharedMemory::<F>::new(PREDICT_MAX_FEATURES as usize);

    // Linearized cube id over the (possibly Y-folded) grid — UNIFORM per cube,
    // so the guard below is a safe barrier scope (the `gram_xty_shared` idiom).
    let b = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let t = UNIT_POS_X;
    if b < nblocks {
        // Stage coef (length n <= PREDICT_MAX_FEATURES) once; every row reuses it.
        if t < n {
            shm_coef[t as usize] = coef[t as usize];
        }

        // Stage the block's B×n X slab, COALESCED. `idx` walks the contiguous
        // global range `[row0·n, row0·n + B·n)`; consecutive threads read
        // consecutive addresses (the coalesced load), and write the padded
        // shared slot `shm_x[lr·STRIDE + c]`.
        let row0 = b * PREDICT_ROWS_PER_BLOCK;
        let tile = PREDICT_ROWS_PER_BLOCK * n;
        let mut idx = t;
        while idx < tile {
            let lr = idx / n;
            let c = idx % n;
            let grow = row0 + lr;
            if grow < m {
                shm_x[(lr * PREDICT_TILE_STRIDE + c) as usize] = x[(grow * n + c) as usize];
            }
            idx += CUBE_DIM_X;
        }
        sync_cube();

        // Compute: thread t finalizes local row t (= global row row0 + t). The
        // per-row dot reads the padded shared tile (conflict-free) and the
        // shared coef (broadcast).
        if t < PREDICT_ROWS_PER_BLOCK {
            let grow = row0 + t;
            if grow < m {
                let rowbase = t * PREDICT_TILE_STRIDE;
                let mut acc = F::new(0.0_f32);
                let mut c = 0u32;
                while c < n {
                    acc += shm_x[(rowbase + c) as usize] * shm_coef[c as usize];
                    c += 1u32;
                }
                out[grow as usize] = acc + bias[0];
            }
        }
    }
}
