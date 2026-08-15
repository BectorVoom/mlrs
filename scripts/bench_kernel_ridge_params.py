#!/usr/bin/env python3
"""Per-PARAMETER `KernelRidge` comparison against scikit-learn (KERNEL-PARAMS).

This harness benches the PARAMETER SURFACE: it walks the values of one parameter
at a time with everything else at its sklearn default, so a regression is
attributable to the parameter that caused it rather than to "KernelRidge got
slower".

Only the parameters that can move the wall clock get a sweep, and each one gets
it for a stated reason:

``kernel``        the one string-valued parameter, and the only one that changes
                  which BASE OP runs. Four families (`linear`, `poly`,
                  `sigmoid`, `cosine`) reach `K` through GEMM; `rbf` goes
                  through the squared-euclidean distance path; `laplacian` and
                  the chi² pair go through direct `O(n²·d)` per-element kernels
                  with no data reuse at all; `precomputed` runs no base op. That
                  is a genuine spread and this sweep is what measures it.
``alpha``         scalar versus a per-target VECTOR. One penalty means one
                  `(K + αI)` and therefore ONE Cholesky shared across all `t`
                  targets; distinct penalties mean `t` different matrices and
                  `t` factorisations. The cost is in the solve, so the sweep
                  scales `n_targets` to make it visible.
``sample_weight`` present versus absent. Weighting is an extra `n²` host pass
                  folded into the pass that injects `α`, so the prediction is
                  "nearly free"; the sweep exists to check that it IS.
``kernel``        as a CALLABLE rather than a name — the same parameter, but the
  (callable)      route is completely different (sklearn's Python-level pairwise
                  loop on both sides, then the precomputed solve). Benched
                  separately at a smaller `n` because an `O(n²)` Python loop at
                  the main size takes minutes on both engines.

`gamma`, `degree`, `coef0` and `kernel_params` get NO sweep. They change what
the kernel computes, not how much work it is: `gamma` is one multiply per
element, `coef0` one add, `kernel_params` is a dict lookup. `degree` is the only
one with any claim — `powf` is not free — and it is included as a sub-sweep of
the poly kernel to check that an integer degree is not accidentally taking a
slower path than a fractional one.

    python3 scripts/bench_kernel_ridge_params.py [sweep ...]

with ``sweep`` one of ``kernel`` / ``alpha`` / ``weight`` / ``callable`` /
``degree`` (default: all but ``callable``, which is slow). ``KR_PARAM_N`` /
``KR_PARAM_D`` / ``KR_PARAM_NT`` override the design size.

## Methodology
Each cell is the MINIMUM of ``KR_PARAM_REPS`` (default 9) alternating
mlrs/sklearn calls, taken after one untimed warm-up call per engine (mlrs pays a
one-time device pipeline compile on the first launch of each kernel shape, and
sklearn pays its own import-time lazy work). Interleaving is what makes the pair
comparable on a machine that is not idle: a competing job that inflates one
engine's minimum inflates the other's too, where two separately-timed blocks
would attribute the whole difference to whichever ran during the spike.

Each cell reports BOTH clocks: wall time, and (in brackets) `process_time` —
CPU time actually spent on a core. Wall time is what a user feels and is the
headline; CPU time is what survives a shared machine, because being descheduled
by a co-tenant does not inflate it. Read CPU time as a lower bound for
comparisons ACROSS parameter values, never as "how long the fit takes": it
excludes device execution, which is most of the point of a GPU backend.

`fit` is the timed call throughout. `predict` for this estimator is a
cross-kernel plus one GEMM and is not where any of these parameters live — the
solve is, and the solve is in `fit`.

## On a shared machine, set ``KR_PARAM_SPACING``
A cell's reps take under a second in total, so on a busy box they all land
inside the same burst of someone else's job and the minimum is a minimum of
contended samples. ``KR_PARAM_SPACING=8`` sleeps between reps so they straddle a
co-tenant's duty cycle and at least one lands in a gap — which, since the cell
reports a minimum, is the one that counts. It stretches a full run to roughly
half an hour and is the difference between a number and a guess.

## Read the occupancy line before trusting anything here
The run prints `procs_running` before and after the sweeps, and labels itself
CONTENDED if either end is above 2. That is not decoration: past sessions in
this repo have had a single co-tenant job INVERT an mlrs-vs-sklearn verdict, and
two runs of THIS harness were contaminated by unrelated jobs landing on the box
mid-sweep. A CONTENDED run's numbers are noise — the giveaway is that the same
engine's time for the same work varies several-fold between sweeps. Gate on
`procs_running`, NOT on loadavg: a cubecl device arm inflates loadavg, so a
loadavg gate in front of a device sweep never fires.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

N = int(os.environ.get("KR_PARAM_N", 700))
D = int(os.environ.get("KR_PARAM_D", 24))
NT = int(os.environ.get("KR_PARAM_NT", 8))
# 9, not 5: the cell is a MINIMUM, so extra reps only help — each one is another
# chance to land in a gap between a co-tenant's bursts. On a quiet box the extra
# four cost seconds and change nothing.
REPS = int(os.environ.get("KR_PARAM_REPS", 9))

# Seconds to sleep BETWEEN reps. 0 (the default) is right on a machine you own.
#
# On a SHARED one it is the only thing that works. A cell's nine reps take well
# under a second, so with no spacing they all land inside the same burst of
# whatever else is running and the minimum is the minimum of nine contended
# samples — which is not a measurement of this code. Spacing spreads the reps
# across a co-tenant's duty cycle so at least one lands in a gap, and since the
# cell reports a MINIMUM, that one is the one that counts. `KR_PARAM_SPACING=8`
# with the default nine reps stretches a cell to ~70s, long enough to straddle
# the burst period this box exhibits.
SPACING = float(os.environ.get("KR_PARAM_SPACING", 0.0))

# `n` for the callable sweep. An O(n²) Python-level pairwise loop runs on BOTH
# engines there, so the main size would spend minutes measuring the interpreter.
CALLABLE_N = int(os.environ.get("KR_PARAM_CALLABLE_N", 150))

# Every kernel name, with the `gamma` each needs to be well-defined. `chi2` is
# the one family with no gamma default (see `kernel_ridge.rs`).
KERNELS = [
    ("linear", {}),
    ("rbf", {}),
    ("poly", {}),
    ("sigmoid", {}),
    ("laplacian", {}),
    ("cosine", {}),
    ("chi2", {"gamma": 0.7}),
    ("additive_chi2", {}),
]

# Target counts for the alpha sweep. 1 is the degenerate case where scalar and
# per-target are the same amount of work; the spread only opens up above it.
TARGET_SWEEP = [1, 2, 4, 8, 16]

DEGREES = [1.0, 2.0, 2.5, 3.0, 7.0]


def bench_dtype():
    """The float type BOTH engines are benched at.

    `float64` where the backend has it, `float32` where it does not (rocm /
    cuda). Not a tolerance concession — an f64 design on an f64-incapable
    backend does not run slower, it RAISES at ingress, so hardcoding f64 would
    make this harness unusable on exactly the hardware worth measuring. sklearn
    is given the same dtype, so the comparison stays like-for-like.
    """
    import mlrs

    return np.float64 if mlrs.backend_supports_f64() else np.float32


def make_data(n: int = N, d: int = D, n_targets: int = 1, seed: int = 42):
    """A NON-NEGATIVE design (the chi² families require it) with `n_targets`
    smooth targets, so ONE design serves every cell — a per-kernel design would
    let a timing difference hide behind a different condition number."""
    dtype = bench_dtype()
    rng = np.random.default_rng(seed)
    x = (rng.random((n, d)) + 0.1).astype(dtype)
    coef = rng.random((d, n_targets)) + 0.5
    y = x @ coef + 0.05 * rng.standard_normal((n, n_targets))
    if n_targets == 1:
        y = y[:, 0]
    return x, np.ascontiguousarray(y, dtype=dtype)


def procs_running() -> int:
    """How many processes are on a core right now.

    `procs_running`, not loadavg: a cubecl device arm inflates loadavg on a
    busy-wait backend, so a loadavg gate in front of a device sweep never fires
    and a loadavg reading says more about mlrs than about the machine. This
    counts what is actually competing for the measurement. Returns -1 if
    /proc/stat is unreadable (non-Linux), which reads as "unknown", not "quiet".
    """
    try:
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("procs_running"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


def _contention_note(running: int) -> str:
    if running < 0:
        return "  <-- unknown occupancy"
    if running <= 2:
        return ""
    return "  <-- CONTENDED, these numbers are noise"


def interleaved(call_a, call_b, reps: int = REPS):
    """Best-of-`reps` for two calls, ALTERNATING so both see the same machine.

    Returns `(wall_a, wall_b, cpu_a, cpu_b)` — BOTH clocks, minimum over reps.

    Wall clock is the headline — it is the number a user feels. CPU time is
    reported beside it because being descheduled by a co-tenant does not inflate
    it, which makes it the more stable of the two on a shared box.

    Neither is a clean reading of mlrs's device work, and the CPU column in
    particular must not be read as one. `process_time` sums CPU across ALL
    THREADS of the process, so a backend whose worker threads spin while the
    device runs bills that spinning here; and it excludes device execution
    itself, which is most of the point of a GPU backend. Both distortions push
    in opposite directions and neither is bounded. Use the pair as a
    consistency check — a cell whose two clocks disagree about the ORDERING of
    two parameter values is a cell that measured the machine, not the code.

    The real defence against a busy box is `spacing`: see `SPACING`.
    """
    wa = wb = ca = cb = float("inf")
    for rep in range(reps):
        # Spread the reps out (see `SPACING`). Skipped before the first one so a
        # quiet-box run pays nothing extra for the option being available.
        if rep and SPACING:
            time.sleep(SPACING)
        t0, c0 = time.perf_counter(), time.process_time()
        call_a()
        wa = min(wa, time.perf_counter() - t0)
        ca = min(ca, time.process_time() - c0)
        t0, c0 = time.perf_counter(), time.process_time()
        call_b()
        wb = min(wb, time.perf_counter() - t0)
        cb = min(cb, time.process_time() - c0)
    return wa, wb, ca, cb


def row(label: str, tm: float, ts: float, cm: float = 0.0, cs: float = 0.0) -> None:
    verdict = "WIN " if ts > tm else "LOSS"
    cpu = f"   [cpu {cm:>7.4f} / {cs:>7.4f}]" if (cm or cs) else ""
    print(
        f"  {label:<24} mlrs {tm:>9.4f}s  sklearn {ts:>9.4f}s  "
        f"{ts / tm:>7.2f}x  {verdict}{cpu}"
    )


def bench_fit(label, kwargs, x, y, sample_weight=None) -> None:
    """One cell: time `fit` on both engines, interleaved.

    Warnings are suppressed inside the timed region rather than around it:
    `additive_chi2` raises the indefinite-Gram warning on EVERY fit in both
    engines, and `warnings.warn`'s registry bookkeeping is not what this is
    trying to measure.
    """
    import mlrs
    from sklearn.kernel_ridge import KernelRidge as Sk

    def m_call():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlrs.KernelRidge(**kwargs).fit(x, y, sample_weight=sample_weight)

    def s_call():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Sk(**kwargs).fit(x, y, sample_weight=sample_weight)

    m_call()  # warm-up: mlrs compiles its device pipeline on first launch
    s_call()
    tm, ts, cm, cs = interleaved(m_call, s_call)
    row(label, tm, ts, cm, cs)


def sweep_kernel() -> None:
    """The string-valued parameter — the base-op sweep."""
    x, y = make_data()
    print(f"kernel sweep (n={N} d={D}, single target, alpha=1.0)")
    for name, kw in KERNELS:
        bench_fit(name, {"kernel": name, **kw}, x, y)

    # `precomputed` cannot share the design — its X IS the Gram — so it gets its
    # own cell against a matrix the other rows' `rbf` produced. It is the floor:
    # whatever it costs is the solve alone, with no kernel evaluation at all.
    from sklearn.metrics.pairwise import rbf_kernel

    k = np.ascontiguousarray(rbf_kernel(x, gamma=1.0 / D), dtype=bench_dtype())
    print(f"\nprecomputed (n={N}) — the solve with NO kernel evaluation")
    bench_fit("precomputed", {"kernel": "precomputed"}, k, y)


def sweep_alpha() -> None:
    """Scalar versus per-target `alpha`, as the target count grows.

    The scalar row is the shared factorisation; the vector row is `t` of them.
    Both are benched at every `t` so the sweep shows the two curves diverging
    rather than one number in isolation.
    """
    print(f"alpha sweep (n={N} d={D}, kernel='rbf') — scalar vs per-target vector")
    for t in TARGET_SWEEP:
        x, y = make_data(n_targets=t)
        bench_fit(f"t={t:<3} scalar", {"kernel": "rbf", "alpha": 1.0}, x, y)
        alphas = list(np.logspace(-2, 2, t))
        bench_fit(f"t={t:<3} per-target", {"kernel": "rbf", "alpha": alphas}, x, y)
        # A uniform VECTOR must take the scalar's shared-factorisation path. If
        # this row tracks the per-target row instead of the scalar row, the
        # `one_alpha` fast path has stopped firing.
        if t > 1:
            bench_fit(
                f"t={t:<3} uniform vec",
                {"kernel": "rbf", "alpha": [1.0] * t},
                x,
                y,
            )


def sweep_weight() -> None:
    """`sample_weight` present versus absent — the extra `n²` host pass."""
    print(f"sample_weight sweep (n={N} d={D}, kernel='rbf')")
    rng = np.random.default_rng(7)
    for t in (1, NT):
        x, y = make_data(n_targets=t)
        sw = rng.random(x.shape[0]) + 0.1
        bench_fit(f"t={t:<3} unweighted", {"kernel": "rbf"}, x, y)
        bench_fit(f"t={t:<3} weighted", {"kernel": "rbf"}, x, y, sample_weight=sw)


def sweep_degree() -> None:
    """`degree` on the poly kernel — integer versus fractional exponents.

    Expected to be FLAT: `powf` does not care whether its exponent is an
    integer. A non-flat row would mean one of the two engines is special-casing
    an integer degree, which is worth knowing either way.
    """
    x, y = make_data()
    print(f"degree sweep (n={N} d={D}, kernel='poly')")
    for deg in DEGREES:
        bench_fit(f"degree={deg}", {"kernel": "poly", "degree": deg}, x, y)


def sweep_callable() -> None:
    """A CALLABLE `kernel` — the same parameter, an entirely different route.

    Both engines evaluate the callable through sklearn's `pairwise_kernels`, so
    this cell is measuring the Python loop plus whichever solve follows it. The
    comparison is still meaningful — the solve is the part that differs — but
    the ratio will be much closer to 1 than any named kernel's, because most of
    the time is spent in code both engines share.
    """
    x, y = make_data(n=CALLABLE_N)

    def k(a, b):
        return float(np.dot(a, b)) ** 2 + 1.0

    print(f"callable kernel (n={CALLABLE_N} d={D}) — Python pairwise on both sides")
    bench_fit("callable", {"kernel": k}, x, y)
    bench_fit("named poly (reference)", {"kernel": "poly", "degree": 2.0}, x, y)


SWEEPS = {
    "kernel": sweep_kernel,
    "alpha": sweep_alpha,
    "weight": sweep_weight,
    "degree": sweep_degree,
    "callable": sweep_callable,
}

# `callable` is excluded from the default set: it is dominated by an O(n²)
# Python loop that neither engine owns, so it answers a different question than
# the rest and costs the most to ask.
DEFAULT_SWEEPS = ["kernel", "alpha", "weight", "degree"]


def main(argv: list[str]) -> int:
    wanted = argv[1:] or DEFAULT_SWEEPS
    unknown = [w for w in wanted if w not in SWEEPS]
    if unknown:
        print(f"unknown sweep(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(SWEEPS)}", file=sys.stderr)
        return 2

    before = procs_running()
    print(f"procs_running {before}{_contention_note(before)}")
    print(f"dtype {np.dtype(bench_dtype()).name}\n")

    for i, name in enumerate(wanted):
        if i:
            print()
        SWEEPS[name]()

    # Read the occupancy again at the END. A run that started quiet and finished
    # contended is exactly as unusable as one that started contended, and the
    # only way to tell them apart afterwards is to have recorded both. This
    # machine has had three unrelated jobs land on it mid-session.
    after = procs_running()
    print(f"\nprocs_running {before} -> {after}{_contention_note(max(before, after))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
