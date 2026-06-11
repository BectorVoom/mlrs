//! Dual-path reduction kernels (PRIM-02) — a plane/subgroup path AND a
//! shared-memory tree fallback for sum / min / max / sum-of-squares (L2-norm),
//! plus full-array and per-row argmin / argmax with a lowest-index tie-break.
//!
//! ## Two SEPARATE functions per op (D-03)
//! Each reduction op ships as TWO distinct `#[cube(launch)]` functions —
//! `reduce_*_plane` and `reduce_*_shared` — rather than one kernel with a
//! `#[comptime]` branch. RESEARCH (§"Dual-Path Reduction Mechanics") recommends
//! this so each path is a named launch the host's `ReducePath` selector and the
//! reduce tests can exercise independently. The plane path uses `PLANE_DIM`
//! (NO hardcoded 32 — D-03); the shared path uses a `SharedMemory` log₂ tree
//! (pairwise-stable, Pitfall 3).
//!
//! ## Per-cube partials → host finalize
//! Every kernel reduces ONE cube's worth of input and writes a single partial
//! per cube (shared path) or per plane (plane path). The host (`mlrs-backend`)
//! finalizes: a single launch covering the whole array yields one cube ⇒ one
//! partial = the full-array result; row reductions launch one cube per row ⇒
//! one partial per row. mean is NOT a kernel — the host runs the sum reduction
//! then scales by `1/n` (two-pass for stability).
//!
//! ## Plane-path partial layout
//! The plane kernels write `output[CUBE_POS_X * planes_per_cube + PLANE_POS]`,
//! i.e. one partial per (cube, plane). The host sizes `output` to
//! `num_cubes * planes_per_cube` and folds the per-plane partials. With the
//! launch configs the host uses (one cube per reduced segment, `CubeDim.x` a
//! multiple of the plane width) this is exact.
//!
//! ## argmin / argmax (D-02, Pitfall 4)
//! The index kernels carry a `(value, index)` pair through every combine and,
//! on EQUAL values, keep the LOWER index — in both the plane shuffle and the
//! shared-memory tree. This matches numpy / sklearn `argmin`/`argmax` tie-break
//! (lowest index wins). Values live in an `F` array; indices in a `u32` array.
//!
//! All kernels are generic over `<F: Float + CubeElement>` and carry NO backend
//! feature (D-13). Tests live in `crates/mlrs-backend/tests/reduce_test.rs`
//! (AGENTS.md §2 — never an in-source `mod tests`).

use cubecl::prelude::*;

// ===========================================================================
// SUM
// ===========================================================================

/// Plane-path sum reduction: fold each plane with `plane_shuffle_xor` over
/// `PLANE_DIM` (NO hardcoded width — D-03), then combine the per-plane partials
/// in shared memory so the kernel writes exactly ONE partial per cube at
/// `output[CUBE_POS_X]` — the SAME output layout as the shared-memory path.
///
/// ## Why combine in-cube (not one partial per plane)
/// `PLANE_DIM` is runtime-variable on some adapters (this env's wgpu reports
/// `plane_size_min=32, plane_size_max=64`), so a host that pre-sizes a
/// `num_cubes * planes_per_cube` output cannot know the true `planes_per_cube`.
/// Folding the plane partials inside the cube (each plane-leader writes its
/// partial to `shared[PLANE_POS]`, then unit 0 sums them) removes ALL host
/// dependence on the plane width: the host treats this kernel exactly like the
/// shared kernel (one partial per cube), and `plane_shuffle_xor` is still the
/// genuinely-exercised subgroup primitive (D-03).
#[cube(launch)]
pub fn reduce_sum_plane<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>) {
    // Per-plane partials (one slot per plane in the cube). 256 is the max
    // CubeDim.x, so at the smallest plane width (1) this still fits.
    let mut shared = SharedMemory::<F>::new(256usize);
    let mut acc = if ABSOLUTE_POS < input.len() {
        input[ABSOLUTE_POS]
    } else {
        F::from_int(0i64)
    };
    let mut i = 1u32;
    while i < PLANE_DIM {
        acc += plane_shuffle_xor(acc, i);
        i *= 2u32;
    }
    // Each plane-leader publishes its plane's partial.
    if UNIT_POS_PLANE == 0u32 {
        shared[PLANE_POS as usize] = acc;
    }
    sync_cube();
    // Unit 0 sums the per-plane partials into one cube partial.
    if UNIT_POS_X == 0u32 {
        let planes_per_cube = CUBE_DIM_X / PLANE_DIM;
        let mut total = F::from_int(0i64);
        let mut p = 0u32;
        while p < planes_per_cube {
            total += shared[p as usize];
            p += 1u32;
        }
        output[CUBE_POS_X as usize] = total;
    }
}

/// Shared-memory sum reduction: `log₂` tree over a single cube (pairwise-stable,
/// Pitfall 3), writing one partial per cube at `output[CUBE_POS_X]`.
#[cube(launch)]
pub fn reduce_sum_shared<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>) {
    let mut shared = SharedMemory::<F>::new(256usize);
    let tid = UNIT_POS_X;
    let gid = ABSOLUTE_POS_X;

    shared[tid as usize] = if (gid as usize) < input.len() {
        input[gid as usize]
    } else {
        F::from_int(0i64)
    };
    sync_cube();

    let mut s = CUBE_DIM_X / 2u32;
    while s > 0u32 {
        if tid < s {
            let v = shared[(tid + s) as usize];
            shared[tid as usize] += v;
        }
        sync_cube();
        s /= 2u32;
    }
    if tid == 0u32 {
        output[CUBE_POS_X as usize] = shared[0usize];
    }
}

// ===========================================================================
// SUM OF SQUARES (L2-norm = sqrt of this; the host applies the sqrt)
// ===========================================================================

/// Plane-path sum-of-squares reduction (basis for the L2 norm). Each unit
/// squares its element first, then the plane folds the squares and the per-plane
/// partials are combined in shared memory to one cube partial (see
/// [`reduce_sum_plane`] for the in-cube-combine rationale).
#[cube(launch)]
pub fn reduce_sumsq_plane<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>) {
    let mut shared = SharedMemory::<F>::new(256usize);
    let v = if ABSOLUTE_POS < input.len() {
        input[ABSOLUTE_POS]
    } else {
        F::from_int(0i64)
    };
    let mut acc = v * v;
    let mut i = 1u32;
    while i < PLANE_DIM {
        acc += plane_shuffle_xor(acc, i);
        i *= 2u32;
    }
    if UNIT_POS_PLANE == 0u32 {
        shared[PLANE_POS as usize] = acc;
    }
    sync_cube();
    if UNIT_POS_X == 0u32 {
        let planes_per_cube = CUBE_DIM_X / PLANE_DIM;
        let mut total = F::from_int(0i64);
        let mut p = 0u32;
        while p < planes_per_cube {
            total += shared[p as usize];
            p += 1u32;
        }
        output[CUBE_POS_X as usize] = total;
    }
}

/// Shared-memory sum-of-squares reduction (basis for the L2 norm).
#[cube(launch)]
pub fn reduce_sumsq_shared<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>) {
    let mut shared = SharedMemory::<F>::new(256usize);
    let tid = UNIT_POS_X;
    let gid = ABSOLUTE_POS_X;

    let v = if (gid as usize) < input.len() {
        input[gid as usize]
    } else {
        F::from_int(0i64)
    };
    shared[tid as usize] = v * v;
    sync_cube();

    let mut s = CUBE_DIM_X / 2u32;
    while s > 0u32 {
        if tid < s {
            let val = shared[(tid + s) as usize];
            shared[tid as usize] += val;
        }
        sync_cube();
        s /= 2u32;
    }
    if tid == 0u32 {
        output[CUBE_POS_X as usize] = shared[0usize];
    }
}

// ===========================================================================
// MIN
// ===========================================================================

/// Plane-path min reduction. OOB lanes seed with the cube's first element (a
/// real value that never beats a true min); each plane folds via shuffle, then
/// the per-plane minima are combined in shared memory to one cube partial.
#[cube(launch)]
pub fn reduce_min_plane<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>) {
    let mut shared = SharedMemory::<F>::new(256usize);
    let mut acc = if ABSOLUTE_POS < input.len() {
        input[ABSOLUTE_POS]
    } else {
        input[0usize]
    };
    let mut i = 1u32;
    while i < PLANE_DIM {
        let other = plane_shuffle_xor(acc, i);
        if other < acc {
            acc = other;
        }
        i *= 2u32;
    }
    if UNIT_POS_PLANE == 0u32 {
        shared[PLANE_POS as usize] = acc;
    }
    sync_cube();
    if UNIT_POS_X == 0u32 {
        let planes_per_cube = CUBE_DIM_X / PLANE_DIM;
        let mut best = shared[0usize];
        let mut p = 1u32;
        while p < planes_per_cube {
            let v = shared[p as usize];
            if v < best {
                best = v;
            }
            p += 1u32;
        }
        output[CUBE_POS_X as usize] = best;
    }
}

/// Shared-memory min reduction (`log₂` tree).
#[cube(launch)]
pub fn reduce_min_shared<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>) {
    let mut shared = SharedMemory::<F>::new(256usize);
    let tid = UNIT_POS_X;
    let gid = ABSOLUTE_POS_X;

    // OOB lanes seed with the first cube element so they never beat a real
    // value (the host pads segments to the cube width).
    shared[tid as usize] = if (gid as usize) < input.len() {
        input[gid as usize]
    } else {
        input[0usize]
    };
    sync_cube();

    let mut s = CUBE_DIM_X / 2u32;
    while s > 0u32 {
        if tid < s {
            let other = shared[(tid + s) as usize];
            if other < shared[tid as usize] {
                shared[tid as usize] = other;
            }
        }
        sync_cube();
        s /= 2u32;
    }
    if tid == 0u32 {
        output[CUBE_POS_X as usize] = shared[0usize];
    }
}

// ===========================================================================
// MAX
// ===========================================================================

/// Plane-path max reduction (mirror of [`reduce_min_plane`]).
#[cube(launch)]
pub fn reduce_max_plane<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>) {
    let mut shared = SharedMemory::<F>::new(256usize);
    let mut acc = if ABSOLUTE_POS < input.len() {
        input[ABSOLUTE_POS]
    } else {
        input[0usize]
    };
    let mut i = 1u32;
    while i < PLANE_DIM {
        let other = plane_shuffle_xor(acc, i);
        if other > acc {
            acc = other;
        }
        i *= 2u32;
    }
    if UNIT_POS_PLANE == 0u32 {
        shared[PLANE_POS as usize] = acc;
    }
    sync_cube();
    if UNIT_POS_X == 0u32 {
        let planes_per_cube = CUBE_DIM_X / PLANE_DIM;
        let mut best = shared[0usize];
        let mut p = 1u32;
        while p < planes_per_cube {
            let v = shared[p as usize];
            if v > best {
                best = v;
            }
            p += 1u32;
        }
        output[CUBE_POS_X as usize] = best;
    }
}

/// Shared-memory max reduction (`log₂` tree).
#[cube(launch)]
pub fn reduce_max_shared<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>) {
    let mut shared = SharedMemory::<F>::new(256usize);
    let tid = UNIT_POS_X;
    let gid = ABSOLUTE_POS_X;

    shared[tid as usize] = if (gid as usize) < input.len() {
        input[gid as usize]
    } else {
        input[0usize]
    };
    sync_cube();

    let mut s = CUBE_DIM_X / 2u32;
    while s > 0u32 {
        if tid < s {
            let other = shared[(tid + s) as usize];
            if other > shared[tid as usize] {
                shared[tid as usize] = other;
            }
        }
        sync_cube();
        s /= 2u32;
    }
    if tid == 0u32 {
        output[CUBE_POS_X as usize] = shared[0usize];
    }
}

// ===========================================================================
// ARGMIN / ARGMAX (value + index, lowest-index tie-break — D-02, Pitfall 4)
// ===========================================================================

/// Shared-memory argmin: carries `(value, index)` through a `log₂` tree and, on
/// EQUAL values, keeps the LOWER index (D-02). Writes the winning value at
/// `out_val[CUBE_POS_X]` and its global index at `out_idx[CUBE_POS_X]`.
///
/// The carried index is the GLOBAL element index (`ABSOLUTE_POS_X`), so the
/// per-cube partial is the true index within the reduced segment. The host
/// launches one cube per reduced segment (full array = one cube; per-row = one
/// cube per row with row-relative indices, see the host API).
#[cube(launch)]
pub fn argmin_shared<F: Float + CubeElement>(
    input: &Array<F>,
    out_val: &mut Array<F>,
    out_idx: &mut Array<u32>,
) {
    let mut sval = SharedMemory::<F>::new(256usize);
    let mut sidx = SharedMemory::<u32>::new(256usize);
    let tid = UNIT_POS_X;
    let gid = ABSOLUTE_POS_X;

    if (gid as usize) < input.len() {
        sval[tid as usize] = input[gid as usize];
        sidx[tid as usize] = tid;
    } else {
        // OOB lane: seed with the cube's first element and a SENTINEL-high index
        // so a tie never lets the pad index win.
        sval[tid as usize] = input[0usize];
        sidx[tid as usize] = CUBE_DIM_X;
    }
    sync_cube();

    let mut s = CUBE_DIM_X / 2u32;
    while s > 0u32 {
        if tid < s {
            let ov = sval[(tid + s) as usize];
            let oi = sidx[(tid + s) as usize];
            let cv = sval[tid as usize];
            let ci = sidx[tid as usize];
            // Strictly smaller value wins; on a tie the LOWER index wins.
            if ov < cv {
                sval[tid as usize] = ov;
                sidx[tid as usize] = oi;
            } else if ov == cv {
                if oi < ci {
                    sidx[tid as usize] = oi;
                }
            }
        }
        sync_cube();
        s /= 2u32;
    }
    if tid == 0u32 {
        out_val[CUBE_POS_X as usize] = sval[0usize];
        out_idx[CUBE_POS_X as usize] = sidx[0usize];
    }
}

/// Shared-memory argmax: carries `(value, index)` and, on EQUAL values, keeps
/// the LOWER index (D-02).
#[cube(launch)]
pub fn argmax_shared<F: Float + CubeElement>(
    input: &Array<F>,
    out_val: &mut Array<F>,
    out_idx: &mut Array<u32>,
) {
    let mut sval = SharedMemory::<F>::new(256usize);
    let mut sidx = SharedMemory::<u32>::new(256usize);
    let tid = UNIT_POS_X;
    let gid = ABSOLUTE_POS_X;

    if (gid as usize) < input.len() {
        sval[tid as usize] = input[gid as usize];
        sidx[tid as usize] = tid;
    } else {
        sval[tid as usize] = input[0usize];
        sidx[tid as usize] = CUBE_DIM_X;
    }
    sync_cube();

    let mut s = CUBE_DIM_X / 2u32;
    while s > 0u32 {
        if tid < s {
            let ov = sval[(tid + s) as usize];
            let oi = sidx[(tid + s) as usize];
            let cv = sval[tid as usize];
            let ci = sidx[tid as usize];
            // Strictly larger value wins; on a tie the LOWER index wins.
            if ov > cv {
                sval[tid as usize] = ov;
                sidx[tid as usize] = oi;
            } else if ov == cv {
                if oi < ci {
                    sidx[tid as usize] = oi;
                }
            }
        }
        sync_cube();
        s /= 2u32;
    }
    if tid == 0u32 {
        out_val[CUBE_POS_X as usize] = sval[0usize];
        out_idx[CUBE_POS_X as usize] = sidx[0usize];
    }
}
