#!/usr/bin/env python3
"""HuberRegressor **fit** wall-clock: mlrs (cpu backend) vs scikit-learn.

The HUBER-01 cpu probe. Both engines minimize the SAME objective with the same
family of solver (L-BFGS over `[coef, intercept, sigma]`), so the comparison is
like-for-like at library defaults. What differs is how one objective evaluation
is performed, and that is the whole story:

    sklearn   `_huber_loss_and_gradient` in NumPy — FIVE passes over the design
              per evaluation, two of them `axis0_safe_slice` fancy-index COPIES
              of the inlier / outlier row blocks (so another whole `n x d` is
              allocated and written on every single evaluation)
    mlrs      ONE fused pass: the margin dot product, the `|r| > eps*sigma`
              classification, the three scalar reductions `d/dsigma` needs and
              the gradient accumulate all happen while the row is still in L1 —
              no allocation, no augmented copy, split across a worker pool that
              is spawned once for the whole solve

So the expected shape is that mlrs' margin is roughly flat in `n_features` (both
engines are bandwidth-bound on the same matrix) and GROWS with `n_samples` up to
the point where the pass saturates memory bandwidth.

    .venv/bin/python scripts/bench_huber.py [--reps 5] [--check]
                     [--engine mlrs|sklearn|both] [--sweep]

Two caveats carried from the other cpu probes, both load-bearing:

  * OpenBLAS keeps its workers SPINNING after a call, so interleaving both
    engines in one process taxes whichever runs second. The default schedule is
    INTERLEAVED (engines alternate rep by rep, so a load burst hits both); re-run
    a marginal rung with `--engine mlrs` / `--engine sklearn` in separate
    processes before believing it (`mlrs-cpu-bench-separate-processes`).
  * On a busy box read the `cpu (s)` column, not `fit (s)`: `time.process_time`
    excludes time spent descheduled. Note it also CHARGES mlrs for its worker
    threads, so on wide rungs it is a deliberately pessimistic reading of the
    mlrs column rather than a like-for-like one.

`--check` prints max|Δcoef| against sklearn plus `scale_`, `n_iter_` and the
outlier count. A speed number means nothing without it: mlrs deliberately solves
TIGHTER than sklearn (sklearn leaves scipy's `factr` at its `1e7` default and so
stops on the relative-f criterion ~1e-6 from the minimizer, which `tol` cannot
change), so mlrs is expected to run a FEW MORE iterations and land closer to the
optimum. `--check` is what shows that the extra iterations bought accuracy
rather than being wasted.

`--sweep` replaces the ladder with a per-parameter cost sweep at one geometry —
the parameters that change the iteration count (`epsilon`, `alpha`, `tol`,
`max_iter`, `warm_start`) and the ones that change the per-pass cost
(`sample_weight`, `fit_intercept`).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

# (rows, features) of the timed fit. Walks both axes independently: `n` at fixed
# `d` isolates the streaming cost of the design (what the fused pass and the
# worker split address), `d` at fixed `n` isolates the per-row dot product and
# the `d`-length gradient accumulate.
CONFIGS = [
    (1_000, 8),
    (10_000, 8),
    (10_000, 64),
    (100_000, 16),
    (100_000, 64),
    (50_000, 128),
    (200_000, 32),
]

# Fraction of rows given a large additive shock in `y`. Without gross outliers
# every sample sits in the quadratic core, the fit degenerates to least squares,
# and the benchmark measures a problem nobody would reach for this estimator to
# solve.
OUTLIER_FRAC = 0.08


def make_design(n: int, d: int, dtype, seed: int = 42):
    """A linear design with `OUTLIER_FRAC` of the targets grossly shocked."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = x @ w + 1.5 + 0.4 * rng.standard_normal(n)
    n_out = max(1, int(round(OUTLIER_FRAC * n)))
    idx = rng.choice(n, size=n_out, replace=False)
    y[idx] += 25.0 * rng.standard_normal(n_out) + 15.0
    return (
        np.ascontiguousarray(x.astype(dtype)),
        np.ascontiguousarray(y.astype(dtype)),
    )


def timed_call(fn):
    """(wall, cpu, result) seconds for ONE call."""
    w0, c0 = time.perf_counter(), time.process_time()
    out = fn()
    return time.perf_counter() - w0, time.process_time() - c0, out


class Samples:
    """Per-engine timing accumulator: min wall, min cpu, first wall, last model."""

    def __init__(self):
        self.wall = []
        self.cpu = []
        self.model = None

    def add(self, wall, cpu, model):
        self.wall.append(wall)
        self.cpu.append(cpu)
        self.model = model

    @property
    def best(self):
        return min(self.wall)

    @property
    def best_cpu(self):
        return min(self.cpu)

    @property
    def first(self):
        return self.wall[0]


def _ctor_kwargs(args):
    return dict(
        epsilon=args.epsilon,
        max_iter=args.max_iter,
        alpha=args.alpha,
        fit_intercept=not args.no_intercept,
        tol=args.tol,
    )


def run_ladder(args, MlrsEst, SkEst, dt, configs):
    header = (
        f"{'n':>8} {'d':>5} | {'engine':>8} "
        f"{'fit (s)':>10} {'cpu (s)':>10} {'first (s)':>10} {'n_iter':>7}"
    )
    print(header)
    print("-" * len(header))
    engines = [e for e in ("mlrs", "sklearn") if args.engine in ("both", e)]
    common = _ctor_kwargs(args)

    for n, d in configs:
        x, y = make_design(n, d, dt)
        sw = None
        if args.sample_weight:
            sw = np.abs(np.random.default_rng(7).standard_normal(n)).astype(dt) + 0.25

        def fit_mlrs():
            m = MlrsEst(**common)
            m.fit(x, y, sample_weight=sw)
            return m

        def fit_sk():
            m = SkEst(**common)
            m.fit(x, y, sample_weight=sw)
            return m

        fits = {"mlrs": fit_mlrs, "sklearn": fit_sk}
        samples = {e: Samples() for e in engines}
        failed = {}

        order = (
            [e for _ in range(args.reps) for e in engines]
            if args.schedule == "interleaved"
            else [e for e in engines for _ in range(args.reps)]
        )
        for eng in order:
            if eng in failed:
                continue
            try:
                samples[eng].add(*timed_call(fits[eng]))
            except Exception as exc:  # noqa: BLE001
                failed[eng] = f"{type(exc).__name__}: {exc}"

        for eng, msg in failed.items():
            print(f"{n:>8} {d:>5} | {eng:>8}  FAILED: {msg}")

        ok = [e for e in engines if e not in failed]
        for eng in ok:
            s = samples[eng]
            print(
                f"{n:>8} {d:>5} | {eng:>8} "
                f"{s.best:>10.4f} {s.best_cpu:>10.4f} {s.first:>10.4f} "
                f"{int(s.model.n_iter_):>7}"
            )
        if len(ok) == 2:
            wall_x = samples["sklearn"].best / samples["mlrs"].best
            cpu_x = samples["sklearn"].best_cpu / samples["mlrs"].best_cpu
            note = f"{wall_x:.2f}x wall / {cpu_x:.2f}x cpu vs sklearn"
            mm, sm = samples["mlrs"].model, samples["sklearn"].model
            if args.check:
                a = np.asarray(mm.coef_, dtype=np.float64).ravel()
                b = np.asarray(sm.coef_, dtype=np.float64).ravel()
                dev = float(np.max(np.abs(a - b)))
                # The iteration counts are EXPECTED to differ — mlrs solves to
                # `ftol = 64*eps` where sklearn stops at scipy's `factr = 1e7`.
                # Printed rather than flagged, with the accuracy that bought it
                # next to it.
                note += (
                    f"  | dcoef={dev:.2e}"
                    f" scale={mm.scale_:.6g}/{sm.scale_:.6g}"
                    f" nout={int(np.sum(mm.outliers_))}/{int(np.sum(sm.outliers_))}"
                    f" niter={mm.n_iter_}/{sm.n_iter_}"
                )
            print(f"{'':>8} {'':>5} | {note}")


def run_sweep(args, MlrsEst, SkEst, dt):
    """Per-parameter cost, at one geometry, for both engines.

    Split deliberately into the parameters that move the ITERATION COUNT and the
    ones that move the COST PER PASS — the two call for different reading. A row
    whose `n_iter` moved is a conditioning/stopping effect; a row whose `n_iter`
    held but whose time moved is a per-row cost effect.
    """
    n, d = args.sweep_n, args.sweep_d
    x, y = make_design(n, d, dt)
    rng = np.random.default_rng(7)
    sw = (np.abs(rng.standard_normal(n)) + 0.25).astype(dt)

    base = _ctor_kwargs(args)
    cases = [
        ("default", {}, False, False),
        # --- iteration-count knobs -------------------------------------- #
        ("epsilon=1.05", dict(epsilon=1.05), False, False),
        ("epsilon=2.5", dict(epsilon=2.5), False, False),
        ("epsilon=10.0", dict(epsilon=10.0), False, False),
        ("alpha=0", dict(alpha=0.0), False, False),
        ("alpha=100", dict(alpha=100.0), False, False),
        ("tol=1e-2", dict(tol=1e-2), False, False),
        ("tol=5.0", dict(tol=5.0), False, False),
        ("max_iter=5", dict(max_iter=5), False, False),
        ("max_iter=1000", dict(max_iter=1000), False, False),
        # --- per-pass cost knobs ---------------------------------------- #
        ("fit_intercept=False", dict(fit_intercept=False), False, False),
        ("sample_weight", {}, True, False),
        # --- warm_start: the refit case --------------------------------- #
        ("warm_start (2nd fit)", dict(warm_start=True), False, True),
    ]

    engines = [e for e in ("mlrs", "sklearn") if args.engine in ("both", e)]
    header = (
        f"{'configuration':>22} | {'engine':>8} {'fit (s)':>10} "
        f"{'cpu (s)':>10} {'n_iter':>7} {'ms/iter':>9}"
    )
    print(f"\n[parameter cost sweep] n={n} d={d} dtype={dt.__name__}")
    print(header)
    print("-" * len(header))

    for label, over, weighted, refit in cases:
        kw = dict(base)
        kw.update(over)
        w = sw if weighted else None
        for eng in engines:
            Est = MlrsEst if eng == "mlrs" else SkEst
            best_w = best_c = float("inf")
            model = None
            for _ in range(args.reps):
                m = Est(**kw)
                if refit:
                    # Warm start only means anything on the SECOND fit; the
                    # first is charged separately and not timed.
                    m.fit(x, y, sample_weight=w)
                wall, cpu, model = timed_call(
                    lambda m=m: (m.fit(x, y, sample_weight=w), m)[1]
                )
                best_w = min(best_w, wall)
                best_c = min(best_c, cpu)
            it = int(model.n_iter_)
            print(
                f"{label:>22} | {eng:>8} {best_w:>10.4f} {best_c:>10.4f} "
                f"{it:>7} {best_w * 1e3 / max(it, 1):>9.2f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--engine", default="both", choices=["both", "mlrs", "sklearn"])
    ap.add_argument("--epsilon", type=float, default=1.35)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--no-intercept", action="store_true")
    ap.add_argument("--sample-weight", action="store_true")
    ap.add_argument(
        "--check",
        action="store_true",
        help="print max|Δcoef|, scale_, the outlier counts and both n_iter_",
    )
    ap.add_argument("--configs", default="", help="comma-separated n:d")
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="per-parameter cost sweep at one geometry instead of the ladder",
    )
    ap.add_argument("--sweep-n", type=int, default=100_000)
    ap.add_argument("--sweep-d", type=int, default=16)
    ap.add_argument(
        "--schedule",
        default="interleaved",
        choices=["interleaved", "blocked"],
        help="alternate engines rep by rep (default) or run each engine's reps "
        "back to back",
    )
    args = ap.parse_args()

    import mlrs
    from sklearn.linear_model import HuberRegressor as SkEst

    from mlrs import HuberRegressor as MlrsEst

    dt = np.float32 if args.dtype == "float32" else np.float64
    print(
        f"mlrs {mlrs.__file__} | epsilon={args.epsilon} alpha={args.alpha} "
        f"tol={args.tol} max_iter={args.max_iter} dtype={args.dtype} "
        f"sample_weight={args.sample_weight}"
    )

    if args.sweep:
        run_sweep(args, MlrsEst, SkEst, dt)
        return

    configs = CONFIGS
    if args.configs:
        configs = [tuple(int(v) for v in c.split(":")) for c in args.configs.split(",")]
    run_ladder(args, MlrsEst, SkEst, dt, configs)


if __name__ == "__main__":
    main()
