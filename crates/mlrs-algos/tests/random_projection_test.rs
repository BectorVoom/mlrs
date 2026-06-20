//! Plan 07-06 — RandomProjection (PROJ-01/02) property + value-oracle tests.
//!
//! WAVE-0 SCAFFOLD (this file is created by plan 07-01). Every test function is
//! `#[ignore]` and asserts ONLY fixture-load + shape well-formedness (and, for
//! the JL case, the value-fixture load) — it makes NO reference to the
//! not-yet-existent `mlrs_algos::projection::{gaussian,sparse}` estimators (the
//! module is an empty stub until plan 07-06). This is the 04-01 / 05-01 Wave-0
//! pattern: the test crate must COMPILE today; plan 07-06 removes the `#[ignore]`,
//! wires the real Gaussian/Sparse projection + `johnson_lindenstrauss_min_dim`,
//! and turns each stub into the live property / value assertion.
//!
//! IMPORTANT — the gate here is a STRUCTURAL PROPERTY SET, **NOT** the 1e-5
//! oracle (D-12). The RNG is host SplitMix64, not numpy's MT19937, so the
//! projection matrix CANNOT match sklearn element-wise; only
//! `johnson_lindenstrauss_min_dim` is value-matched. The property gate (plan
//! 07-06 wires) covers: the JL distortion bound (`(1−eps)·‖u−v‖² ≤
//! ‖proj(u)−proj(v)‖² ≤ (1+eps)·‖u−v‖²`) AVERAGED over many trials, the matrix
//! moment stats (Gaussian mean≈0/var≈1/n_components; Achlioptas density), seed
//! reproducibility (same `u64` seed → identical matrix), and the
//! `transform == X·componentsᵀ` self-consistency.
//!
//! D-11 mitigates strict-band JL flakiness with a FIXED SplitMix64 seed +
//! averaging over a PINNED trial count `JL_TRIALS` — set here so plan 07-06
//! inherits the exact count.
//!
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::load_npz;

/// PINNED averaging trial count for the JL distortion / moment property gates
/// (D-11 — a single unlucky draw never flips the strict band; plan 07-06 uses
/// this exact constant so the gate is reproducible across runs and backends).
const JL_TRIALS: usize = 50;

/// johnson_lindenstrauss_min_dim grid sizes (gen_oracle.py `JL_N_SAMPLES` /
/// `JL_EPS` are length-3 each → a 3×3 `min_dim` matrix).
const JL_GRID: usize = 3;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// JL distortion bound averaged over `JL_TRIALS` (PROPERTY, NOT 1e-5 — D-12):
/// the pairwise squared distances are preserved within `(1 ± eps)` on average.
///
/// WAVE-0 STUB. Plan 07-06 removes `#[ignore]` and wires the Gaussian projection
/// over `JL_TRIALS` fixed-seed draws → averaged distortion within the eps band.
#[test]
#[ignore = "wave-0 scaffold: projection::{gaussian,sparse} land in plan 07-06"]
fn random_projection_jl_distortion() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    assert!(JL_TRIALS >= 1, "JL_TRIALS must be positive (D-11 averaging)");
}

/// Projection-matrix moment stats (PROPERTY): Gaussian mean ≈ 0 / var ≈
/// 1/n_components; Achlioptas non-zero density ≈ `density`. Averaged over
/// `JL_TRIALS`.
///
/// WAVE-0 STUB. Plan 07-06 wires the moment checks over the projection matrices.
#[test]
#[ignore = "wave-0 scaffold: projection::{gaussian,sparse} land in plan 07-06"]
fn random_projection_matrix_moments() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    assert!(JL_TRIALS >= 1, "JL_TRIALS must be positive");
}

/// Seed reproducibility (PROPERTY / T-07-02): the SAME `u64` seed → an identical
/// projection matrix (host SplitMix64 is deterministic, never OsRng).
///
/// WAVE-0 STUB. Plan 07-06 wires two same-seed projections + element equality.
#[test]
#[ignore = "wave-0 scaffold: projection::{gaussian,sparse} land in plan 07-06"]
fn random_projection_seed_reproducible() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    assert!(JL_TRIALS >= 1, "JL_TRIALS must be positive");
}

/// `transform == X · componentsᵀ` self-consistency (PROPERTY): the transform is
/// the single GEMM of X against the stored projection matrix (no centering —
/// RandomProjection does not center, D-12).
///
/// WAVE-0 STUB. Plan 07-06 wires the device `transform` vs a host `X·componentsᵀ`.
#[test]
#[ignore = "wave-0 scaffold: projection::{gaussian,sparse} land in plan 07-06"]
fn random_projection_transform_self_consistent() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    assert!(JL_TRIALS >= 1, "JL_TRIALS must be positive");
}

/// `johnson_lindenstrauss_min_dim` VALUE oracle (the ONE value-matched RP check,
/// D-12): the integer min-dim grid matches sklearn over the `(n_samples, eps)`
/// grid.
///
/// WAVE-0 STUB: loads the committed `jl_min_dim` blob and asserts the grid shapes
/// are well-formed. Plan 07-06 removes `#[ignore]` and wires
/// `johnson_lindenstrauss_min_dim(n_samples, eps)` → integer-exact compare vs
/// `min_dim`.
#[test]
#[ignore = "wave-0 scaffold: projection JL value oracle lands in plan 07-06"]
fn random_projection_jl_min_dim() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("jl_min_dim_f32_seed42.npz"))
        .expect("load jl_min_dim_f32");
    assert_eq!(case.expect_f64("n_samples").len(), JL_GRID, "n_samples grid len");
    assert_eq!(case.expect_f64("eps").len(), JL_GRID, "eps grid len");
    assert_eq!(
        case.shape("min_dim").expect("min_dim").to_vec(),
        vec![JL_GRID as u64, JL_GRID as u64],
        "min_dim is a JL_GRID×JL_GRID matrix"
    );
    // eps grid is strictly in (0, 1) (the JL bound's valid range).
    for &e in case.expect_f64("eps") {
        assert!((0.0..1.0).contains(&e), "eps {e} must be in (0, 1)");
    }
}
