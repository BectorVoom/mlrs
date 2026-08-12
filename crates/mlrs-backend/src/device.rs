//! `device` — the execution-placement hyperparameter (DEVICE-PARAM-01).
//!
//! Almost every estimator in this workspace has TWO implementations of its
//! heavy phase: a HOST arm written in plain Rust over host slices, and a DEVICE
//! arm built from `cubecl` kernels. Which one runs has, until now, been decided
//! entirely by an internal heuristic — the backend name plus a size floor — with
//! `MLRS_*` environment flags as the only override, and those exist for
//! benchmarking rather than for callers.
//!
//! That was the wrong shape for two reasons a user runs into immediately:
//!
//! 1. **The heuristic is a guess about hardware it cannot see.** The floors were
//!    tuned on the machines this repo was developed on. On an integrated
//!    adapter the design upload alone can cost more than the entire host fit
//!    (measured: 122 ms to upload a 102 MiB design against an 88 ms host fit),
//!    and on a discrete card with a fast link the opposite holds. The person
//!    running the code knows which they have; the library does not.
//! 2. **An environment variable is not an API.** It is process-global, so it
//!    cannot differ between two estimators in one pipeline, it does not survive
//!    into a `clone()`d estimator, and it is invisible to `get_params`.
//!
//! [`Device`] is the hyperparameter that fixes both. It is a *preference*, not
//! an assertion: it overrides the heuristic wherever the estimator genuinely
//! has both arms for the requested configuration, and where it does not — a
//! solver with only one implementation, say — the fit still runs, and the
//! estimator reports which arm it actually took through its `device_` fitted
//! attribute. That is the [`RidgeSolver`](crate) `solver_` precedent: sklearn's
//! `Ridge` accepts `solver='auto'` and then tells you what ran, and a caller
//! who needs to know can always ask rather than infer.
//!
//! ## The one place `device` DOES change the answer: UMAP
//! Everywhere else the two arms are the same computation and agree to ~1e-15,
//! which is the property `test_device_param.py::test_arms_agree_on_the_fit`
//! exists to hold. UMAP's layout is the exception, and it is not a defect in
//! this parameter: the host driver and the device kernel are two different
//! implementations of a STOCHASTIC SGD, and they consume the negative-sampling
//! RNG in different orders. Same seed, different embedding — measured at ~1.5e1
//! apart on a 300x6 design, i.e. a completely different (equally valid) layout.
//!
//! That was already true before this parameter existed — it is what you got by
//! moving between a cpu and a gpu backend — but `device` makes it selectable
//! within one process, so it has to be said out loud. A caller who needs a
//! reproducible embedding must hold `device` fixed, exactly as they must hold
//! `random_state` fixed.
//!
//! ## Why `"gpu"` and not `"device"`
//! Internally this codebase says "host" and "device", where "device" means
//! *whatever `cubecl` runtime is active* — which on a `--features cpu` build is
//! `cubecl-cpu`, running the same kernels on the CPU. The user-facing spelling
//! is `"cpu"`/`"gpu"` because that is what the choice means on the backends
//! people actually deploy (cuda / rocm / wgpu), and because `device="device"`
//! is not a sentence.
//!
//! The honest consequence, documented rather than hidden: on a cpu-backend
//! build `device="gpu"` is legal and selects the `cubecl-cpu` kernel path. It
//! is not an error — it is the device arm, and it is genuinely what was asked
//! for — but it is normally much SLOWER than `"cpu"` there, because
//! `cubecl-cpu` maps one OS thread per unit and JITs at `-O0`. Nothing in this
//! module tries to be clever about that; a caller who sets it gets it.
//!
//! ## Precedence
//! ```text
//! device="cpu"/"gpu"   >   MLRS_* abflag   >   the shape heuristic
//!                          \___ consulted only under `Auto` ___/
//! ```
//! An estimator built with an EXPLICIT `device` never consults an abflag at
//! all — that is the property that makes the parameter reproducible, and
//! `device_param_test::an_explicit_device_ignores_the_abflag` pins it. A stray
//! `MLRS_*` in someone's environment must not move a fit the caller pinned.
//!
//! Under `Auto` the abflags keep their existing authority over the heuristic,
//! and that half matters just as much: every perf probe and A/B sweep in this
//! repo drives a DEFAULT-constructed estimator and forces the arm through
//! `abflag`, so demoting them there would silently break that whole apparatus.
//!
//! (An earlier draft of this table had the first two terms the wrong way round.
//! The code was always as described here; the table was not. Precedence stated
//! backwards in a doc is worse than no doc, because it is exactly what someone
//! reaches for instead of reading `prefers_host`.)
//!
//! Tests live in `crates/mlrs-backend/tests/device_test.rs` (AGENTS.md §2).

/// Where an estimator should run its heavy phase.
///
/// The Python spelling is the lower-case name (`"auto"` / `"cpu"` / `"gpu"`);
/// see [`Device::from_name`] for the parse and [`Device::name`] for the
/// round-trip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    /// Let the estimator's own heuristic decide (the default, and the ONLY
    /// value that consults the `MLRS_*` A/B flags).
    #[default]
    Auto,
    /// Force the HOST arm — plain Rust over host slices, no kernel launch and,
    /// on the ingress paths that support it, no upload of the design at all.
    Cpu,
    /// Force the DEVICE arm — `cubecl` kernels on the active runtime. On a
    /// cpu-backend build this is the `cubecl-cpu` path (see the module docs).
    Gpu,
}

impl Device {
    /// The user-facing string, for `device_` and for diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            Device::Auto => "auto",
            Device::Cpu => "cpu",
            Device::Gpu => "gpu",
        }
    }

    /// Parse the sklearn-style string, or `None` for an unrecognised one.
    ///
    /// Returns `Option` rather than a `Result` so the *caller's* error type
    /// carries the rejection: `mlrs-algos` turns this into
    /// `BuildError::UnknownDevice` alongside its other `StrOptions`-shaped
    /// rejections, and `mlrs-backend` stays free of that dependency.
    pub fn from_name(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(Device::Auto),
            "cpu" => Some(Device::Cpu),
            "gpu" => Some(Device::Gpu),
            _ => None,
        }
    }

    /// Should the HOST arm run?
    ///
    /// `heuristic` is the estimator's existing size/backend gate — the thing
    /// that decided this before there was a hyperparameter. It is a closure
    /// because it is not free (several gates query backend capabilities) and
    /// an explicit `device` makes it dead work.
    ///
    /// This is the ONE place the precedence rule in the module docs is
    /// implemented, so every estimator that routes through it agrees on it.
    pub fn prefers_host(self, heuristic: impl FnOnce() -> bool) -> bool {
        match self {
            Device::Cpu => true,
            Device::Gpu => false,
            Device::Auto => heuristic(),
        }
    }

    /// Should the DEVICE arm run?
    ///
    /// The mirror of [`Device::prefers_host`], for the gates that are phrased
    /// the other way round — `GaussianMixture::device_fit_applicable`, say,
    /// where the host arm is always available and the question is whether the
    /// kernels are worth it. Spelled out rather than left to callers to write
    /// `!prefers_host(|| !heuristic())`, because that double negation is
    /// exactly the kind of thing that gets it backwards once.
    pub fn prefers_device(self, heuristic: impl FnOnce() -> bool) -> bool {
        match self {
            Device::Cpu => false,
            Device::Gpu => true,
            Device::Auto => heuristic(),
        }
    }

    /// The arm this preference resolves to, as the string `device_` reports.
    ///
    /// Takes the RESOLVED boolean rather than recomputing, so an estimator
    /// cannot report one arm and run another — the caller passes the same
    /// value it branched on.
    pub fn resolved_name(host: bool) -> &'static str {
        if host {
            "cpu"
        } else {
            "gpu"
        }
    }
}
