#!/usr/bin/env python3
"""RANSACRegressor **fit** wall-clock: mlrs (cpu backend) vs scikit-learn.

The RANSAC-01 cpu probe. Both engines run the SAME algorithm on the SAME draw
sequence — mlrs reproduces numpy's MT19937 exactly, so for a given
`random_state` the two visit the same sub-samples in the same order and stop on
the same trial. That makes this an unusually clean comparison: identical work,
different execution, and `--check` asserts the identity by comparing
`n_trials_` and `inlier_mask_` and not merely the coefficients.

What differs per trial:

    sklearn   `clone`-free but `validate_data`-heavy: `LinearRegression.fit` on
              the sub-sample re-validates and re-centers, `estimator.predict(X)`
              is one BLAS `gemv` over the whole design, and then
              `estimator.score(X[inliers], y[inliers])` runs a SECOND `gemv`
              over a fancy-indexed COPY of the consensus set — an `n_in x d`
              allocation and memcpy on every trial that improves on the
              incumbent.
    mlrs      ONE fused pass: the row dot product, the loss, the
              `<= residual_threshold` classification and the R2 numerator all
              happen while the row is still in L1, over a worker pool spawned
              once for the whole fit. The consensus score reuses those residuals
              instead of predicting again, so the second `gemv` and its copy do
              not exist. The sub-sample solve is a `min_samples x d` one-sided
              Jacobi SVD, which at `d + 1` rows is smaller than the LAPACK call
              overhead sklearn pays.

So the expected shape is a margin that GROWS with `n_samples` (the streaming
pass and the copy sklearn avoids paying) and is roughly flat in `max_trials`
(both engines pay it per trial).

    .venv/bin/python scripts/bench_ransac_cpu.py [--reps 5] [--check]
                     [--engine mlrs|sklearn|both] [--sweep]
                     [--base ols|ridge|lasso|tree|knn|svr] [--device auto|cpu|gpu]

`--base` swaps the BASE ESTIMATOR (RANSAC-02). Anything but `ols` is a base mlrs
does not fit natively, so its sub-model goes back through Python once per trial
(`RansacBase::Foreign`) while everything else — the loss, the mask, the
consensus, the R2, the stop rules — still runs in Rust. That is the arm to point
this at when the question is "what did moving the LOOP buy", separately from
"what did moving the SUB-MODEL SOLVE buy", because with a foreign base both
engines call the identical `estimator.fit`/`predict` and the difference is the
loop around them and nothing else.

Expect PARITY there, not a win, and read a deviation from it as noise unless it
repeats: measured 0.86x / 1.00x / 0.82x (ridge, `20k x 16` / `50k x 32` /
`200k x 32`) and 0.82-1.07x for tree and knn bases. Both engines are
bandwidth-bound at the top rung and mlrs makes one extra streaming pass over the
predictions there. Run-to-run spread on this box is ~15%, so a single rung means
nothing on its own — which is why `--engine` exists and why the two engines
should be run in SEPARATE processes.

`--device gpu` moves the per-trial scan to the batched device kernels. Read
`batch_width` in the header line to confirm the arm engaged before believing a
number ([[mlrs-bench-verify-knob-is-live]]).

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

`--sweep` replaces the ladder with a per-parameter cost sweep at one geometry.
The parameters split into two kinds, and the two call for different reading:
those that change HOW MANY TRIALS run (`max_trials`, `stop_probability`,
`stop_n_inliers`, `stop_score`, `residual_threshold` — through the consensus
size that feeds `_dynamic_max_trials`) and those that change the COST OF ONE
TRIAL (`min_samples`, `loss`, `sample_weight`, the base `fit_intercept`). A row
whose `n_trials` moved is the former; a row whose `n_trials` held but whose time
moved is the latter. The `ms/trial` column is what makes them separable.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

# (rows, features) of the timed fit. Walks both axes independently: `n` at fixed
# `d` isolates the streaming cost of the per-trial scan (what the fused pass and
# the worker split address), `d` at fixed `n` isolates the per-row dot product
# AND the sub-sample solve, whose default `min_samples = d + 1` makes it the one
# part of the fit that grows as `d^3`.
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
# every model wins the whole design on the first trial, `_dynamic_max_trials`
# collapses `max_trials` to one, and the benchmark measures a problem nobody
# would reach for this estimator to solve.
OUTLIER_FRAC = 0.25


def make_design(n: int, d: int, dtype, seed: int = 42):
    """A linear design with `OUTLIER_FRAC` of the targets grossly shocked."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = x @ w + 1.5 + 0.1 * rng.standard_normal(n)
    n_out = max(1, int(round(OUTLIER_FRAC * n)))
    idx = rng.choice(n, size=n_out, replace=False)
    y[idx] += 40.0 * np.sign(rng.standard_normal(n_out)) + 20.0
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


def make_base(name, seed):
    """The `estimator=` object, fresh per construction (sklearn clones it)."""
    if name == "ols":
        return None
    if name == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0)
    if name == "lasso":
        from sklearn.linear_model import Lasso

        return Lasso(alpha=0.1, max_iter=200)
    if name == "tree":
        from sklearn.tree import DecisionTreeRegressor

        return DecisionTreeRegressor(max_depth=4, random_state=seed)
    if name == "knn":
        from sklearn.neighbors import KNeighborsRegressor

        return KNeighborsRegressor(n_neighbors=5)
    if name == "svr":
        from sklearn.svm import SVR

        return SVR(kernel="rbf", C=10.0)
    raise ValueError(f"unknown base estimator {name!r}")


def _ctor_kwargs(args, d=None):
    kw = dict(
        min_samples=args.min_samples,
        residual_threshold=args.residual_threshold,
        max_trials=args.max_trials,
        stop_probability=args.stop_probability,
        loss=args.loss,
        random_state=args.seed,
    )
    if args.base != "ols":
        kw["estimator"] = make_base(args.base, args.seed)
        # sklearn REQUIRES an explicit `min_samples` for any base that is not a
        # `LinearRegression` — `n_features + 1` is only the right default
        # sub-sample size for a linear model.
        if kw["min_samples"] is None and d is not None:
            kw["min_samples"] = d + 1
    return kw


def _agreement(mm, sm):
    """The identity claim this benchmark rests on, as a printable verdict."""
    a = np.asarray(mm.estimator_.coef_, dtype=np.float64).ravel()
    b = np.asarray(sm.estimator_.coef_, dtype=np.float64).ravel()
    return (
        f"dcoef={float(np.max(np.abs(a - b))):.2e}"
        f" trials={mm.n_trials_}/{sm.n_trials_}"
        f" mask={'=' if np.array_equal(np.asarray(mm.inlier_mask_), sm.inlier_mask_) else 'DIFFER'}"
        f" inliers={int(np.sum(np.asarray(mm.inlier_mask_)))}"
    )


def run_ladder(args, MlrsEst, SkEst, dt, configs):
    header = (
        f"{'n':>8} {'d':>5} | {'engine':>8} "
        f"{'fit (s)':>10} {'cpu (s)':>10} {'first (s)':>10} {'trials':>7} "
        f"{'ms/trial':>9}"
    )
    print(header)
    print("-" * len(header))
    engines = [e for e in ("mlrs", "sklearn") if args.engine in ("both", e)]
    print(f"[base={args.base} device={args.device}]")

    for n, d in configs:
        x, y = make_design(n, d, dt)
        sw = None
        if args.sample_weight:
            sw = np.abs(np.random.default_rng(7).standard_normal(n)) + 0.25

        common = _ctor_kwargs(args, d)

        def fit_of(Est, is_mlrs):
            def go():
                kw = dict(common)
                if is_mlrs and args.device != "auto":
                    kw["device"] = args.device
                if kw.get("estimator") is not None:
                    # A fresh base per fit: sklearn `clone`s it, and a
                    # rep that reused a fitted one would not be timing a fit.
                    kw["estimator"] = make_base(args.base, args.seed)
                m = Est(**kw)
                m.fit(x, y, sample_weight=sw)
                return m

            return go

        fits = {"mlrs": fit_of(MlrsEst, True), "sklearn": fit_of(SkEst, False)}
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
            tr = int(s.model.n_trials_)
            if eng == "mlrs":
                # The arm that actually ran, printed rather than assumed.
                arm = getattr(s.model._mlrs_obj, "device_used", lambda: "?")()
                width = getattr(s.model._mlrs_obj, "batch_width", lambda: 1)()
                print(f"{'':>8} {'':>5} | arm={arm} batch_width={width}")
            print(
                f"{n:>8} {d:>5} | {eng:>8} "
                f"{s.best:>10.4f} {s.best_cpu:>10.4f} {s.first:>10.4f} "
                f"{tr:>7} {s.best * 1e3 / max(tr, 1):>9.3f}"
            )
        if len(ok) == 2:
            wall_x = samples["sklearn"].best / samples["mlrs"].best
            cpu_x = samples["sklearn"].best_cpu / samples["mlrs"].best_cpu
            note = f"{wall_x:.2f}x wall / {cpu_x:.2f}x cpu vs sklearn"
            if args.check:
                note += "  | " + _agreement(
                    samples["mlrs"].model, samples["sklearn"].model
                )
            print(f"{'':>8} {'':>5} | {note}")


def run_sweep(args, MlrsEst, SkEst, dt):
    """Per-parameter cost, at one geometry, for both engines.

    Split deliberately into the parameters that move the TRIAL COUNT and the
    ones that move the COST PER TRIAL (module docs) — `ms/trial` is what tells
    them apart, and `trials` is what says which kind a row is.
    """
    n, d = args.sweep_n, args.sweep_d
    x, y = make_design(n, d, dt)
    sw = (np.abs(np.random.default_rng(7).standard_normal(n)) + 0.25)
    # A threshold near the MAD default, so the `residual_threshold` rows below
    # are a real perturbation of the consensus size rather than a degenerate
    # all-in / all-out.
    mad = float(np.median(np.abs(y - np.median(y))))

    base = _ctor_kwargs(args, d)
    cases = [
        ("default", {}, False),
        # --- trial-count knobs ------------------------------------------- #
        ("max_trials=20", dict(max_trials=20), False),
        ("max_trials=500", dict(max_trials=500), False),
        ("stop_probability=0.5", dict(stop_probability=0.5), False),
        ("stop_probability=0.999", dict(stop_probability=0.999), False),
        ("stop_probability=1.0", dict(stop_probability=1.0), False),
        ("stop_n_inliers=n/2", dict(stop_n_inliers=n // 2), False),
        ("stop_score=0.5", dict(stop_score=0.5), False),
        (f"residual_threshold={mad:.3g}", dict(residual_threshold=mad), False),
        ("residual_threshold=0.25", dict(residual_threshold=0.25), False),
        # --- per-trial cost knobs ---------------------------------------- #
        ("min_samples=d+1 (default)", {}, False),
        ("min_samples=4*(d+1)", dict(min_samples=4 * (d + 1)), False),
        ("min_samples=0.5", dict(min_samples=0.5), False),
        ("loss=squared_error", dict(loss="squared_error"), False),
        ("sample_weight", {}, True),
    ]

    engines = [e for e in ("mlrs", "sklearn") if args.engine in ("both", e)]
    header = (
        f"{'configuration':>26} | {'engine':>8} {'fit (s)':>10} "
        f"{'cpu (s)':>10} {'trials':>7} {'ms/trial':>9} {'inliers':>8}"
    )
    print(f"\n[parameter cost sweep] n={n} d={d} dtype={dt.__name__}")
    print(header)
    print("-" * len(header))

    for label, over, weighted in cases:
        kw = dict(base)
        kw.update(over)
        w = sw if weighted else None
        best = {}
        for eng in engines:
            Est = MlrsEst if eng == "mlrs" else SkEst
            best_w = best_c = float("inf")
            model = None
            for _ in range(args.reps):
                wall, cpu, model = timed_call(
                    lambda: (lambda m: (m.fit(x, y, sample_weight=w), m)[1])(Est(**kw))
                )
                best_w = min(best_w, wall)
                best_c = min(best_c, cpu)
            tr = int(model.n_trials_)
            best[eng] = best_w
            print(
                f"{label:>26} | {eng:>8} {best_w:>10.4f} {best_c:>10.4f} "
                f"{tr:>7} {best_w * 1e3 / max(tr, 1):>9.3f} "
                f"{int(np.sum(np.asarray(model.inlier_mask_))):>8}"
            )
        if len(best) == 2:
            print(
                f"{'':>26} | {best['sklearn'] / best['mlrs']:.2f}x wall vs sklearn"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--engine", default="both", choices=["both", "mlrs", "sklearn"])
    ap.add_argument("--min-samples", type=float, default=None)
    ap.add_argument("--residual-threshold", type=float, default=None)
    ap.add_argument("--max-trials", type=int, default=100)
    ap.add_argument("--stop-probability", type=float, default=0.99)
    ap.add_argument(
        "--loss", default="absolute_error",
        choices=["absolute_error", "squared_error"],
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--base",
        default="ols",
        choices=["ols", "ridge", "lasso", "tree", "knn", "svr"],
        help="the RANSAC base estimator; anything but `ols` exercises the "
        "one-call-per-trial bridge arm on the mlrs side",
    )
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="mlrs execution placement for the per-trial scan",
    )
    ap.add_argument("--sample-weight", action="store_true")
    ap.add_argument(
        "--check",
        action="store_true",
        help="print max|dcoef|, both n_trials_ and whether inlier_mask_ agrees",
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

    from sklearn.linear_model import RANSACRegressor as SkEst

    from mlrs import RANSACRegressor as MlrsEst

    dt = np.float32 if args.dtype == "float32" else np.float64
    # `min_samples=None` is sklearn's default; an int-valued float from the CLI
    # has to go back to an int or sklearn's own branch reads it as a fraction.
    if args.min_samples is not None and args.min_samples >= 1:
        args.min_samples = int(args.min_samples)

    if args.sweep:
        run_sweep(args, MlrsEst, SkEst, dt)
        return

    configs = CONFIGS
    if args.configs:
        configs = [
            tuple(int(v) for v in spec.split(":")) for spec in args.configs.split(",")
        ]
    run_ladder(args, MlrsEst, SkEst, dt, configs)


if __name__ == "__main__":
    main()
