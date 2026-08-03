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

/// `BayesianRidge::predict(X, return_std=True)`'s second return value, fused
/// into one device launch:
///
/// ```text
/// x̃ᵢ     = xᵢ − X_offset_
/// std[i] = √( x̃ᵢ·Σ·x̃ᵢᵀ + 1/α )
/// ```
///
/// ## The quadratic form is evaluated as a SUM OF SQUARES, not as `x̃ᵀΣx̃`
/// `Σ = V·diag(1/(α·λⱼ + λ_prec))·Vᵀ` is symmetric positive definite (every
/// `α·λⱼ + λ_prec` is strictly positive: `λⱼ ≥ 0` for a Gram and `λ_prec > 0`),
/// so it factors as `Σ = M·Mᵀ` with `M = V·diag(1/√(α·λⱼ + λ_prec))` and
///
/// ```text
/// x̃·Σ·x̃ᵀ = ‖Mᵀ·x̃‖² = Σⱼ (mⱼ·x̃)²
/// ```
///
/// The host passes `mt`, which is `Mᵀ` stored ROW-MAJOR, so `mⱼ` is the
/// contiguous run `mt[j·d .. j·d + d]`. That form cannot go negative — the
/// direct `x̃ᵀ(Σx̃)` differences two quantities of mixed sign and needs a
/// `.max(0.0)` before the `sqrt`, which on the device would be a silent NaN
/// instead of a host-side clamp.
///
/// ## A 4×4 REGISTER TILE, because the scalar form is load-bound
/// One `(row, j)` pair at a time issues two loads (`x[i·d+k]`, `mt[j·d+k]`) per
/// multiply-add — 0.33 FMA per load counting the `offset` read. Measured on a
/// Tesla P100 that ran at **9.2 G multiply-adds/s**, about 1% of the card's
/// `f64` peak, and lost to scikit-learn's OpenBLAS GEMM on the CPU beside it
/// (712 ms vs 284 ms at `d = 256`, 100 000 rows). Nothing about a kernel that
/// starved for operands is fixed by launch shape.
///
/// Each unit therefore owns **4 rows × 4 `j` directions**. Inside the `k` loop
/// it loads 4 `x` values, 4 `mt` values and one `offset`, and does 16
/// multiply-adds with them — `1.8` FMA per load, a ~5× improvement in arithmetic
/// intensity. The reuse is two-way and that is the point: blocking only over `j`
/// would cut the `x` traffic but leave `mt` re-read once per row, and blocking
/// only over rows would do the reverse. This is the same `4 × 4` register tile
/// [`crate::gram::gram_xty_blocked`] uses, and for the same reason.
///
/// No `SharedMemory` and no `sync_cube`: the tile lives entirely in registers
/// (~28 `f64` accumulators and operands per unit), so the kernel stays free of
/// barriers and of the shared-memory budget that caps the Gram kernels' `d`.
///
/// ## Shape, tails and precision
/// One unit per ROW TILE, `ntiles = ⌈n/4⌉`, bounds-checked so the
/// ceiling-division launch may over-provision safely (T-0203-01). The final
/// tile's out-of-range rows are CLAMPED to `n − 1` rather than branched on: they
/// then compute a duplicate of the last row's value and simply are not written
/// back, which keeps the inner loop free of divergence. `d` not divisible by 4
/// is handled by a scalar `j` tail after the tiled loop.
///
/// Accumulation is in `F`, matching [`linear_predict_bias`]: the summands of
/// `q` are all non-negative (no cancellation) and the inner `mⱼ·x̃` runs over the
/// fitted feature count, so an `f32` sum stays inside the 1e-5 oracle contract.
/// `noise = 1/α` is a scalar `F` by value (A6, like [`scale`]'s `factor`).
///
/// ## `tile0`: why this kernel is launched in slices
/// The unit handles GLOBAL tile `tile0 + ABSOLUTE_POS`, so the caller can cover
/// `n` rows with several bounded launches instead of one unbounded one. That is
/// a WATCHDOG guard, not a tiling optimization: this is the crate's first kernel
/// whose single-launch cost grows as `n·d²`, and a display-driving GPU cancels
/// any submission that overruns the compositor's timeout. See
/// `crate::prims::linear_predict::STD_MACS_PER_LAUNCH` for the budget, and
/// `use_host_std` for why wgpu does not take this kernel at all.
///
/// [`scale`]: crate::elementwise::scale
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn bayes_predict_std<F: Float + CubeElement>(
    x: &Array<F>,
    offset: &Array<F>,
    mt: &Array<F>,
    out: &mut Array<F>,
    tile0: u32,
    ntiles: u32,
    n: u32,
    d: u32,
    noise: F,
) {
    let t = ABSOLUTE_POS + tile0 as usize;
    if t < ntiles as usize {
        let i0 = (t as u32) * 4u32;

        // Clamp the tile's four row ids into range. The clamped lanes recompute
        // the last row and are not written back below.
        let r0 = i0;
        let mut r1 = i0 + 1u32;
        let mut r2 = i0 + 2u32;
        let mut r3 = i0 + 3u32;
        if r1 >= n {
            r1 = n - 1u32;
        }
        if r2 >= n {
            r2 = n - 1u32;
        }
        if r3 >= n {
            r3 = n - 1u32;
        }
        let b0 = (r0 * d) as usize;
        let b1 = (r1 * d) as usize;
        let b2 = (r2 * d) as usize;
        let b3 = (r3 * d) as usize;

        let zero = F::new(0.0_f32);
        let mut q0 = zero;
        let mut q1 = zero;
        let mut q2 = zero;
        let mut q3 = zero;

        // --- The 4 (rows) × 4 (j) tiled body. ---
        let mut j = 0u32;
        while j + 4u32 <= d {
            let m0 = (j * d) as usize;
            let m1 = m0 + d as usize;
            let m2 = m1 + d as usize;
            let m3 = m2 + d as usize;

            let mut a00 = zero;
            let mut a01 = zero;
            let mut a02 = zero;
            let mut a03 = zero;
            let mut a10 = zero;
            let mut a11 = zero;
            let mut a12 = zero;
            let mut a13 = zero;
            let mut a20 = zero;
            let mut a21 = zero;
            let mut a22 = zero;
            let mut a23 = zero;
            let mut a30 = zero;
            let mut a31 = zero;
            let mut a32 = zero;
            let mut a33 = zero;

            let mut k = 0u32;
            while k < d {
                let kk = k as usize;
                let o = offset[kk];
                let x0 = x[b0 + kk] - o;
                let x1 = x[b1 + kk] - o;
                let x2 = x[b2 + kk] - o;
                let x3 = x[b3 + kk] - o;
                let y0 = mt[m0 + kk];
                let y1 = mt[m1 + kk];
                let y2 = mt[m2 + kk];
                let y3 = mt[m3 + kk];
                a00 += x0 * y0;
                a01 += x0 * y1;
                a02 += x0 * y2;
                a03 += x0 * y3;
                a10 += x1 * y0;
                a11 += x1 * y1;
                a12 += x1 * y2;
                a13 += x1 * y3;
                a20 += x2 * y0;
                a21 += x2 * y1;
                a22 += x2 * y2;
                a23 += x2 * y3;
                a30 += x3 * y0;
                a31 += x3 * y1;
                a32 += x3 * y2;
                a33 += x3 * y3;
                k += 1u32;
            }

            q0 += a00 * a00 + a01 * a01 + a02 * a02 + a03 * a03;
            q1 += a10 * a10 + a11 * a11 + a12 * a12 + a13 * a13;
            q2 += a20 * a20 + a21 * a21 + a22 * a22 + a23 * a23;
            q3 += a30 * a30 + a31 * a31 + a32 * a32 + a33 * a33;
            j += 4u32;
        }

        // --- Scalar `j` tail (`d % 4 != 0`), at most three directions. ---
        while j < d {
            let m = (j * d) as usize;
            let mut a0 = zero;
            let mut a1 = zero;
            let mut a2 = zero;
            let mut a3 = zero;
            let mut k = 0u32;
            while k < d {
                let kk = k as usize;
                let o = offset[kk];
                let y = mt[m + kk];
                a0 += (x[b0 + kk] - o) * y;
                a1 += (x[b1 + kk] - o) * y;
                a2 += (x[b2 + kk] - o) * y;
                a3 += (x[b3 + kk] - o) * y;
                k += 1u32;
            }
            q0 += a0 * a0;
            q1 += a1 * a1;
            q2 += a2 * a2;
            q3 += a3 * a3;
            j += 1u32;
        }

        out[r0 as usize] = F::sqrt(q0 + noise);
        if i0 + 1u32 < n {
            out[r1 as usize] = F::sqrt(q1 + noise);
        }
        if i0 + 2u32 < n {
            out[r2 as usize] = F::sqrt(q2 + noise);
        }
        if i0 + 3u32 < n {
            out[r3 as usize] = F::sqrt(q3 + noise);
        }
    }
}
