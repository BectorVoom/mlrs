#!/usr/bin/env python3
"""WEIGHTED ``fit(X, y, sample_weight)`` wall-clock vs scikit-learn, all five NB.

The NB-SW-PERF probe. The unweighted fits are covered by
``bench_nb_fit_cpu.py`` (Gaussian/Multinomial/Complement),
``bench_bernoulli_nb_cpu.py`` and ``bench_categorical_nb_cpu.py``; this one
times the arm those scripts never touch — the one a non-``None``
``sample_weight`` selects.

That arm is genuinely different code in all five, not a multiply bolted onto the
same loop:

* Bernoulli / Categorical widen their EXACT ``u32`` occurrence tables to ``f64``
  (a fractional weight is not an integer count), i.e. double the accumulator
  traffic of the unweighted sweep.
* Gaussian additionally accumulates the UNWEIGHTED whole-column ``sum``/``sumsq``
  (``epsilon_`` is computed before sklearn looks at the weights, so it is no
  longer reducible out of the weighted per-class totals) — four accumulators per
  element against the unweighted arm's two.

sklearn's weighted arm is likewise not free: the discrete four materialize
``Y *= sample_weight[:, None]`` (an ``n x C`` f64 matrix) before the count GEMM,
and GaussianNB switches to a weighted ``np.average`` per class.

    .venv/bin/python scripts/bench_nb_fit_sw_cpu.py [--reps 3] [--check]
                     [--estimator all|gaussian|multinomial|complement|bernoulli|categorical]
                     [--engine mlrs|sklearn|both] [--dtype float64|float32]
                     [--weights fractional|integer|ones|none]

``--weights none`` times the UNWEIGHTED arm through the same harness, which is
the A/B that says what the weighted path costs over the fit it replaces.

The ``--engine`` caveat from ``bench_linear_predict_cpu.py`` applies verbatim:
BLAS/threading state left behind by one engine can tax whichever runs second.
Re-run a suspicious rung with ``--engine mlrs`` / ``--engine sklearn`` in
SEPARATE processes — on a loaded box that has inverted verdicts before. Read the
``cpu (s)`` column as well as ``fit (s)``: mlrs' sweep is multi-threaded above
~32k elements while sklearn's weighting pass is single-threaded numpy (its GEMM
is not), so the two columns bracket the answer rather than agreeing.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

# (rows, features, n_classes) of the timed fit — the shared ladder of the three
# unweighted NB scripts, so a weighted number is directly comparable to the
# unweighted one for the same rung.
CONFIGS = [
    (1_000, 8, 3),
    (10_000, 16, 4),
    (50_000, 32, 5),
    (100_000, 64, 10),
    (500_000, 8, 2),
    (100_000, 128, 4),
    (50_000, 256, 4),
    (20_000, 512, 4),
    (100_000, 32, 20),
]

# CategoricalNB needs a category count per feature as well; keyed by rung index.
N_CATEGORIES = [5, 10, 8, 12, 4, 10, 10, 10, 50]


def make_data(kind: str, n: int, d: int, n_classes: int, n_cat: int, seed: int = 42):
    """The input each estimator is actually for.

    ``gaussian`` gets real-valued standard-normal features (negatives included —
    only non-finite values are rejected); ``multinomial``/``complement`` get
    non-negative integer counts; ``bernoulli`` exact 0/1; ``categorical``
    integer-coded categories with a RAGGED per-feature category count (feature
    ``j`` draws from ``2 + (j % n_cat)`` categories), which is what the flat
    offset table in the Rust tabulation is for.
    """
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        x = rng.standard_normal((n, d))
    elif kind == "bernoulli":
        x = (rng.random((n, d)) < 0.3).astype(np.float64)
    elif kind == "categorical":
        cols = [rng.integers(0, 2 + (j % n_cat), size=n) for j in range(d)]
        x = np.stack(cols, axis=1).astype(np.float64)
    else:
        x = rng.poisson(2.0, size=(n, d)).astype(np.float64)
    y = rng.integers(0, n_classes, size=n)
    return x, y


def make_weights(kind: str, n: int, seed: int = 7):
    """The ``sample_weight`` vector, or ``None`` for the unweighted arm.

    ``fractional`` is the case that rules out any "repeat the rows"
    implementation and forces the f64 tables; ``integer`` is the case a repeat
    COULD express; ``ones`` is the uniform vector that takes the weighted code
    path while producing the unweighted answer (the agreement fixture).
    """
    if kind == "none":
        return None
    rng = np.random.default_rng(seed)
    if kind == "ones":
        return np.ones(n, dtype=np.float64)
    if kind == "integer":
        return rng.integers(1, 4, size=n).astype(np.float64)
    return rng.uniform(0.1, 3.0, size=n)


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


def max_deviation(mlrs_model, sk_model, x, sample=512):
    """max |delta log P(c | x)| between the two fitted models on `sample` rows.

    The fitted tables are what this benchmark's optimization touches, but the
    mlrs shim does not surface them (only the predict surface is public), so the
    check reads them THROUGH ``predict_log_proba``. A row subset keeps
    ``--check`` from dominating the timing run on the large rungs.
    """
    xs = x[:sample]
    a = np.asarray(mlrs_model.predict_log_proba(xs), dtype=np.float64)
    b = np.asarray(sk_model.predict_log_proba(xs), dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b)))


NAMES = {
    "gaussian": "GaussianNB",
    "multinomial": "MultinomialNB",
    "complement": "ComplementNB",
    "bernoulli": "BernoulliNB",
    "categorical": "CategoricalNB",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--engine", default="both", choices=["both", "mlrs", "sklearn"])
    ap.add_argument(
        "--estimator", default="all", choices=["all", *NAMES.keys()]
    )
    ap.add_argument(
        "--weights",
        default="fractional",
        choices=["fractional", "integer", "ones", "none"],
        help="sample_weight vector to fit with ('none' times the unweighted arm)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="print max|dlog P(c|x)| vs sklearn (reads the fitted tables through "
        "predict_log_proba — the shim does not surface them directly)",
    )
    ap.add_argument("--configs", default="", help="comma-separated n:d:n_classes")
    ap.add_argument(
        "--schedule",
        default="interleaved",
        choices=["interleaved", "blocked"],
        help="alternate engines rep by rep (default) or run each engine's "
        "reps back to back",
    )
    args = ap.parse_args()

    import mlrs.naive_bayes as mlrs_nb
    import sklearn.naive_bayes as sk_nb

    kinds = list(NAMES) if args.estimator == "all" else [args.estimator]

    configs = list(CONFIGS)
    n_cats = list(N_CATEGORIES)
    if args.configs:
        configs = [tuple(int(v) for v in c.split(":")) for c in args.configs.split(",")]
        n_cats = [10] * len(configs)

    dt = {"float64": np.float64, "float32": np.float32}[args.dtype]
    engines = [e for e in ("mlrs", "sklearn") if args.engine in ("both", e)]

    for kind in kinds:
        name = NAMES[kind]
        MlrsEst = getattr(mlrs_nb, name)
        SkEst = getattr(sk_nb, name)
        print(f"\n=== {name} | dtype={args.dtype} weights={args.weights}")
        header = (
            f"{'n':>7} {'d':>4} {'C':>3} | {'engine':>8} "
            f"{'fit (s)':>10} {'cpu (s)':>10} {'first (s)':>10}"
        )
        print(header)
        print("-" * len(header))

        for rung, (n, d, n_classes) in enumerate(configs):
            n_cat = n_cats[rung] if rung < len(n_cats) else 10
            x, y = make_data(kind, n, d, n_classes, n_cat)
            x = np.ascontiguousarray(x.astype(dt))
            w = make_weights(args.weights, n)

            fits = {
                "mlrs": lambda: MlrsEst().fit(x, y, sample_weight=w),
                "sklearn": lambda: SkEst().fit(x, y, sample_weight=w),
            }
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
                    wall, cpu, model = timed_call(fits[eng])
                except Exception as exc:  # noqa: BLE001 — a rung may be invalid
                    failed[eng] = f"{type(exc).__name__}: {exc}"
                    continue
                samples[eng].add(wall, cpu, model)

            for eng in engines:
                if eng in failed:
                    print(f"{n:>7} {d:>4} {n_classes:>3} | {eng:>8} {failed[eng]}")
                    continue
                s = samples[eng]
                print(
                    f"{n:>7} {d:>4} {n_classes:>3} | {eng:>8} "
                    f"{s.best:>10.4f} {s.best_cpu:>10.4f} {s.first:>10.4f}"
                )
            if len(engines) == 2 and not failed:
                m, sk = samples["mlrs"], samples["sklearn"]
                line = (
                    f"{'':>7} {'':>4} {'':>3} | {'speedup':>8} "
                    f"{sk.best / m.best:>9.2f}x {sk.best_cpu / m.best_cpu:>9.2f}x"
                )
                if args.check:
                    line += f"   max|dlogP|={max_deviation(m.model, sk.model, x):.3e}"
                print(line)


if __name__ == "__main__":
    main()
