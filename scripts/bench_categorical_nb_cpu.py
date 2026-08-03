#!/usr/bin/env python3
"""CategoricalNB **fit** wall-clock: mlrs (cpu backend) vs scikit-learn.

The CATNB-FIT-CPU probe. Both engines compute the same thing — per-``(feature,
class, category)`` counts smoothed into ``feature_log_prob_`` — so this is a
like-for-like comparison with no iteration budget to pin: a CategoricalNB fit is
one tabulation pass, and `--check` asserts the fitted tables agree.

The ladder deliberately WIDENS as well as lengthens. The pre-optimization fit
ran ``n_features`` COLUMN-strided passes for the per-feature max and
``n_features`` more for the tabulation, so its cost grew as ``O(n · d²)`` and its
margin over sklearn collapsed from ~2x at ``d = 8`` to ~1x at ``d = 512`` — a
regression only a wide rung exposes. Keep the ``d >= 128`` rungs when trimming
the ladder.

    .venv/bin/python scripts/bench_categorical_nb_cpu.py [--reps 3] [--check]
                     [--engine mlrs|sklearn|both] [--dtype int64|float64|float32]

The ``--engine`` caveat from ``bench_linear_predict_cpu.py`` applies verbatim:
BLAS/threading state left behind by one engine can tax whichever runs second.
Re-run a suspicious rung with ``--engine mlrs`` / ``--engine sklearn`` in
separate processes.

On a machine with unrelated load, prefer the default INTERLEAVED schedule
(engines alternate rep by rep, so a load burst hits both) and read the
``cpu (s)`` column — ``time.process_time`` excludes time the process spent
descheduled. NOTE that mlrs' fit passes ARE multi-threaded above ~32k elements
while sklearn's is single-threaded numpy, so on this benchmark CPU time
OVERSTATES mlrs' cost and wall clock is the user-facing number; the two columns
bracket the answer rather than agreeing.

``--dtype`` matters more here than on the float benchmarks: sklearn validates
CategoricalNB's `X` as ``dtype="int"``, so an integer input costs it no
conversion, while mlrs' shim uploads floats and pays an ``astype`` in
``check_array``. ``int64`` (the default) is therefore the rung that is HARDEST
for mlrs and the honest one to quote.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

# (rows, features, n_classes, categories-per-feature) of the timed fit.
CONFIGS = [
    (1_000, 8, 3, 5),
    (10_000, 16, 4, 10),
    (50_000, 32, 5, 8),
    (100_000, 64, 10, 12),
    (500_000, 8, 2, 4),
    (100_000, 128, 4, 10),
    (50_000, 256, 4, 10),
    (20_000, 512, 4, 10),
    (100_000, 32, 20, 50),
]


def make_categorical(n: int, d: int, n_classes: int, n_cat: int, seed: int = 42):
    """Integer-coded categories with a RAGGED per-feature category count.

    Feature ``j`` spans ``2 + (j % n_cat)`` categories, so ``n_categories_``
    varies across features and the flat offset-indexed count table is exercised
    the way real one-hot-free categorical data exercises it — a uniform
    ``n_cat`` for every feature would let a square-table implementation pass.
    """
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, 2 + (j % n_cat), size=n) for j in range(d)]
    x = np.column_stack(cols)
    y = rng.integers(0, n_classes, size=n)
    return x, y


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


def max_table_deviation(mlrs_model, sk_model, x, sample=512):
    """max |Δ log P(c | x)| between the two fitted models on `sample` rows.

    The fitted ``feature_log_prob_`` tables are the thing this benchmark's
    optimization actually touched, but the mlrs shim does not surface them (only
    the predict surface is public), so the check reads them THROUGH
    ``predict_log_proba``: the joint log-likelihood is a plain sum of looked-up
    table entries plus the class log-prior, so any divergence in a table shows up
    here undamped. A row subset keeps ``--check`` from dominating the timing run
    on the large rungs.
    """
    xs = x[:sample]
    a = np.asarray(mlrs_model.predict_log_proba(xs), dtype=np.float64)
    b = np.asarray(sk_model.predict_log_proba(xs), dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--dtype", default="int64", choices=["int64", "float64", "float32"]
    )
    ap.add_argument("--engine", default="both", choices=["both", "mlrs", "sklearn"])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument(
        "--check",
        action="store_true",
        help="print max|Δlog P(c|x)| vs sklearn (reads the fitted tables through "
        "predict_log_proba — the shim does not surface them directly)",
    )
    ap.add_argument(
        "--configs", default="", help="comma-separated n:d:n_classes:n_categories"
    )
    ap.add_argument(
        "--schedule",
        default="interleaved",
        choices=["interleaved", "blocked"],
        help="alternate engines rep by rep (default) or run each engine's "
        "reps back to back",
    )
    args = ap.parse_args()

    import mlrs
    from mlrs.naive_bayes import CategoricalNB as MlrsEst
    from sklearn.naive_bayes import CategoricalNB as SkEst

    configs = CONFIGS
    if args.configs:
        configs = [tuple(int(v) for v in c.split(":")) for c in args.configs.split(",")]

    dt = {"int64": np.int64, "float64": np.float64, "float32": np.float32}[args.dtype]
    print(f"mlrs {mlrs.__name__} | alpha={args.alpha} dtype={args.dtype}")
    header = (
        f"{'n':>7} {'d':>4} {'C':>3} {'K':>4} | {'engine':>8} "
        f"{'fit (s)':>10} {'cpu (s)':>10} {'first (s)':>10}"
    )
    print(header)
    print("-" * len(header))

    engines = [e for e in ("mlrs", "sklearn") if args.engine in ("both", e)]

    for n, d, n_classes, n_cat in configs:
        x, y = make_categorical(n, d, n_classes, n_cat)
        x = np.ascontiguousarray(x.astype(dt))

        def fit_mlrs():
            return MlrsEst(alpha=args.alpha).fit(x, y)

        def fit_sk():
            return SkEst(alpha=args.alpha).fit(x, y)

        fits = {"mlrs": fit_mlrs, "sklearn": fit_sk}
        samples = {e: Samples() for e in engines}
        failed = {}

        # Interleaved by default: a load burst from an unrelated process hits
        # both engines rather than taxing whichever happens to run second.
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
            print(f"{n:>7} {d:>4} {n_classes:>3} {n_cat:>4} | {eng:>8}  FAILED: {msg}")

        ok = [e for e in engines if e not in failed]
        for eng in ok:
            s = samples[eng]
            print(
                f"{n:>7} {d:>4} {n_classes:>3} {n_cat:>4} | {eng:>8} "
                f"{s.best:>10.4f} {s.best_cpu:>10.4f} {s.first:>10.4f}"
            )
        if len(ok) == 2:
            wall_x = samples["sklearn"].best / samples["mlrs"].best
            cpu_x = samples["sklearn"].best_cpu / samples["mlrs"].best_cpu
            note = f"{wall_x:.2f}x wall / {cpu_x:.2f}x cpu vs sklearn"
            if args.check:
                dev = max_table_deviation(
                    samples["mlrs"].model, samples["sklearn"].model, x
                )
                note += f" | max|Δlog P(c|x)| = {dev:.3e}"
            print(f"{'':>7} {'':>4} {'':>3} {'':>4} | {note}")
        print()


if __name__ == "__main__":
    main()
