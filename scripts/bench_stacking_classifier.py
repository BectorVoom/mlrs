#!/usr/bin/env python3
"""StackingClassifier parameter-sweep harness (STACK-CLF-01).

The twin of ``bench_stacking.py``, measuring the ``StackingClassifier``
parameters that can move the clock against
``sklearn.ensemble.StackingClassifier`` on byte-identical data:

    ================  ================================================
    parameter         why it is on the perf ladder
    ================  ================================================
    ``cv``            THE cost driver, exactly as for the regressor: an
                      int ``k`` costs ``k + 1`` base fits per member,
                      ``cv="prefit"`` costs ZERO.
    ``stack_method``  the classifier's own parameter, and the only one
                      that changes the SHAPE of the work: ``predict``
                      is one column per member, ``predict_proba`` is
                      ``n_classes`` of them (one on a binary target,
                      where the collinear column is dropped). It also
                      changes what each fold has to compute.
    ``n_classes``     not a parameter, but what makes ``stack_method``
                      matter — the meta matrix is
                      ``n x (k * n_classes)`` under ``predict_proba``,
                      so the ladder walks it.
    ``passthrough``   widens the meta matrix by ``d`` columns.
    ``n_jobs``        joblib fan-out. A scheduling knob only: a win is
                      required on the host arm, a FLAT line on the
                      device arm (an mlrs member holds an unpicklable
                      device handle, so the stack fits serially and
                      warns — see ``_effective_n_jobs``).
    ================  ================================================

The meta-matrix COPY is deliberately not measured here. It is well under 1% of
a fit, so this ladder cannot resolve it; ``bench_stacking_meta.py --level copy
--cols <n_classes>`` measures that arm directly, in the shape a classifier
produces.

## Two arms, because they answer different questions

* ``host`` — sklearn base members on BOTH sides. Isolates the meta-estimator
  layer: mlrs's Rust ``StratifiedKFold`` + Rust method resolution + Rust
  meta-layout against sklearn's Python equivalents, with every base fit
  identical.
* ``device`` — mlrs base members in the mlrs stack vs sklearn base members in
  the sklearn stack. The end-to-end deployment comparison, dominated by the
  base estimators, which is the point.

## Members, and why they differ per ``stack_method``

No single mlrs classifier implements all three response methods today
(``mlrs.LogisticRegression`` has ``predict_proba`` but no
``decision_function``), so the ``stack_method`` ladder runs on two member
pairs — a probabilistic one (``GaussianNB``) and a margin one
(``RidgeClassifier``) — and each ``stack_method`` value is measured on the pair
that can serve it. The pair is printed with the row. Both are closed-form; see
:func:`members_for` for why that matters more than it looks.

## Measurement discipline

Every cell runs in a FRESH subprocess (``--cell``) and the harness reports the
MINIMUM of ``--repeat`` runs, interleaved across implementations. Both are
load-bearing on this project: in-process interleaving and a busy box have each
inverted a verdict before.

Usage
-----
    python3 scripts/bench_stacking_classifier.py                # full sweep
    python3 scripts/bench_stacking_classifier.py --arm host
    python3 scripts/bench_stacking_classifier.py --n 50000 --repeat 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

N_DEFAULT = 20_000
D_DEFAULT = 32
SEED = 42

CV_LADDER = [2, 3, 5, 10, "prefit"]
#: ``(stack_method, member pair)``. "proba" members implement
#: ``predict_proba``; "margin" members implement ``decision_function``.
STACK_METHOD_LADDER = [
    ("auto", "proba"),
    ("predict", "proba"),
    ("predict_proba", "proba"),
    ("auto", "margin"),
    ("decision_function", "margin"),
]
N_CLASSES_LADDER = [2, 5, 10]
PASSTHROUGH_LADDER = [False, True]
N_JOBS_LADDER = [None, 2, 4]


def design(n, d, n_classes, dtype):
    """A separable multiclass problem, deterministic in SEED."""
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n, d)).astype(dtype)
    w = rng.standard_normal(d).astype(dtype)
    score = X @ w + (0.5 * rng.standard_normal(n)).astype(dtype)
    if n_classes == 2:
        return X, (score > 0).astype(np.int64)
    cuts = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])
    return X, np.digitize(score, cuts).astype(np.int64)


def members_for(impl, arm, pair):
    """``(members, final_estimator)`` for one implementation and member pair.

    Members are deliberately CHEAP and DETERMINISTIC — two `GaussianNB`s for the
    probabilistic pair, two `RidgeClassifier`s for the margin one, and a
    `RidgeClassifier` as the meta learner. Both are closed-form: no iteration
    count that varies with the data, and no convergence path that varies between
    runs.

    That is not a convenience. A first draft used `LogisticRegression` members,
    and the same configuration measured 2.26 s in one sweep and 3.33 s in
    another — a 47% spread that swamped every effect this ladder is trying to
    show. The parameters here move the *composition* (how many base fits, how
    wide the meta matrix, which response is computed), and that signal is only
    visible when the members themselves do not add noise of their own.
    """
    if impl == "sklearn" or arm == "host":
        from sklearn.linear_model import RidgeClassifier
        from sklearn.naive_bayes import GaussianNB

        if pair == "proba":
            members = [("nb", GaussianNB()), ("nb2", GaussianNB(var_smoothing=1e-6))]
        else:
            members = [
                ("rc", RidgeClassifier()),
                ("rc2", RidgeClassifier(alpha=10.0)),
            ]
        return members, RidgeClassifier()

    import mlrs

    if pair == "proba":
        members = [
            ("nb", mlrs.GaussianNB()),
            ("nb2", mlrs.GaussianNB(var_smoothing=1e-6)),
        ]
    else:
        members = [
            ("rc", mlrs.RidgeClassifier()),
            ("rc2", mlrs.RidgeClassifier(alpha=10.0)),
        ]
    return members, mlrs.RidgeClassifier()


def build(impl, arm, pair, stack_method, cv, passthrough, n_jobs, X, y):
    """The estimator under test.

    ``cv="prefit"`` needs its members ALREADY fitted, so that construction
    happens here — and is deliberately NOT timed, because "prefit" exists
    precisely for the case where those fits already happened elsewhere.
    """
    members, final = members_for(impl, arm, pair)
    if cv == "prefit":
        members = [(name, est.fit(X, y)) for name, est in members]

    if impl == "sklearn":
        from sklearn.ensemble import StackingClassifier
    else:
        from mlrs import StackingClassifier

    return StackingClassifier(
        members,
        final_estimator=final,
        cv=cv,
        stack_method=stack_method,
        passthrough=passthrough,
        n_jobs=n_jobs,
    )


def run_cell(impl, arm, pair, stack_method, cv, passthrough, n_jobs, n, d, n_classes):
    """One timed cell: ``(fit_s, predict_s, checksum, startup_s)``.

    The checksum is the predicted-label mean, printed by the harness so a
    configuration that got fast by predicting something else is visible in the
    output rather than only in the oracle suite.
    """
    dtype = np.float64
    if impl == "mlrs":
        import mlrs

        # WARM-UP, and why it is not cheating: loading `_mlrs.abi3.so` runs the
        # driver probe and brings up the CubeCL runtime (~90 ms here), which is
        # a once-per-process cost and would otherwise be charged to whichever
        # cell ran first. It is reported separately instead.
        t0 = time.perf_counter()
        mlrs._load_ext()
        startup_s = time.perf_counter() - t0
        if arm == "device":
            dtype = np.float64 if mlrs.backend_supports_f64() else np.float32
    else:
        startup_s = 0.0

    X, y = design(n, d, n_classes, dtype)

    # A full warm-up fit AT THE REAL SHAPE, so device-pipeline compilation and
    # any first-call import inside the composed estimators land here rather than
    # in the timed fit below.
    #
    # The shape matters, and this is the trap the harness is written around: a
    # tiny warm-up (n=64, n=256) warms a DIFFERENT pipeline, because which arm
    # and which kernel an mlrs estimator selects depends on the shape. A 546 ms
    # rocm pipeline compile then lands inside the timed region of every cell —
    # once per cell, since each cell is a fresh subprocess, so `min` over
    # repeats does not remove it either. Its signature is a cost that is FLAT in
    # the parameter being swept (fit time barely moving from `cv=3` to `cv=10`
    # while `cv="prefit"` is 100x cheaper); fold work cannot do that, fixed
    # overhead can. Warming at the real shape costs one extra fit per cell and
    # is what makes the ladder mean what it says.
    build(impl, arm, pair, stack_method, cv, passthrough, n_jobs, X, y).fit(X, y)

    est = build(impl, arm, pair, stack_method, cv, passthrough, n_jobs, X, y)

    t0 = time.perf_counter()
    est.fit(X, y)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = est.predict(X)
    predict_s = time.perf_counter() - t0

    return fit_s, predict_s, float(np.asarray(pred, dtype=np.float64).mean()), startup_s


def spawn(impl, arm, pair, stack_method, cv, passthrough, n_jobs, n, d, n_classes):
    """Run one cell in a FRESH interpreter and parse its JSON line."""
    argv = [
        sys.executable, __file__, "--cell",
        "--impl", impl,
        "--arm", arm,
        "--pair", pair,
        "--stack-method", stack_method,
        "--cv", str(cv),
        "--n", str(n),
        "--d", str(d),
        "--n-classes", str(n_classes),
        "--n-jobs", "none" if n_jobs is None else str(n_jobs),
    ]
    if passthrough:
        argv.append("--passthrough")
    out = subprocess.run(argv, capture_output=True, text=True)
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    return None


def measure(impl, arm, pair, stack_method, cv, passthrough, n_jobs, n, d,
            n_classes, repeat):
    """MIN over ``repeat`` fresh-process runs — the noise-resistant statistic."""
    best = None
    checksum = None
    startup = 0.0
    for _ in range(repeat):
        got = spawn(impl, arm, pair, stack_method, cv, passthrough, n_jobs, n, d,
                    n_classes)
        if got is None:
            return None, None, None, None
        checksum = got["checksum"]
        startup = got["startup_s"]
        if best is None or got["fit_s"] < best[0]:
            best = (got["fit_s"], got["predict_s"])
    return best[0], best[1], checksum, startup


_STARTUP_REPORTED = []


def _report_startup(startup_s):
    if startup_s and not _STARTUP_REPORTED:
        _STARTUP_REPORTED.append(startup_s)
        print(
            f"  [mlrs one-time `_mlrs` extension load: {startup_s * 1e3:.0f} ms "
            "per process — excluded from the timings below]"
        )


def sweep(title, cells, arm, n, d, repeat):
    print(f"\n=== {title} (arm={arm}, n={n}, d={d}, min of {repeat}) ===")
    print(
        f"{'config':<34}{'sklearn fit':>13}{'mlrs fit':>12}{'speedup':>9}"
        f"{'sklearn pred':>14}{'mlrs pred':>12}{'speedup':>9}"
    )
    for label, pair, stack_method, cv, passthrough, n_jobs, n_classes in cells:
        sk = measure("sklearn", arm, pair, stack_method, cv, passthrough, n_jobs,
                     n, d, n_classes, repeat)
        ml = measure("mlrs", arm, pair, stack_method, cv, passthrough, n_jobs,
                     n, d, n_classes, repeat)
        sk_fit, sk_pred, sk_sum, _ = sk
        ml_fit, ml_pred, ml_sum, ml_startup = ml
        _report_startup(ml_startup)
        if sk_fit is None:
            print(f"{label:<34}{'sklearn cell failed':>13}")
            continue
        if ml_fit is None:
            print(f"{label:<34}{sk_fit:>12.4f}s{'n/a':>12}")
            continue
        drift = "" if abs(sk_sum - ml_sum) < 1e-3 * (1 + abs(sk_sum)) else "  !checksum"
        print(
            f"{label:<34}{sk_fit:>12.4f}s{ml_fit:>11.4f}s{sk_fit / ml_fit:>8.2f}x"
            f"{sk_pred:>13.4f}s{ml_pred:>11.4f}s{sk_pred / ml_pred:>8.2f}x{drift}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--d", type=int, default=D_DEFAULT)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--arm", choices=["host", "device", "both"], default="both")
    ap.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--impl", choices=["sklearn", "mlrs"], help=argparse.SUPPRESS)
    ap.add_argument("--pair", choices=["proba", "margin"], default="proba",
                    help=argparse.SUPPRESS)
    ap.add_argument("--stack-method", default="auto", help=argparse.SUPPRESS)
    ap.add_argument("--cv", help=argparse.SUPPRESS)
    ap.add_argument("--n-classes", type=int, default=2, help=argparse.SUPPRESS)
    ap.add_argument("--passthrough", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--n-jobs", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.cell:
        cv = args.cv if args.cv == "prefit" else int(args.cv)
        n_jobs = None if args.n_jobs == "none" else int(args.n_jobs)
        fit_s, predict_s, checksum, startup_s = run_cell(
            args.impl, args.arm, args.pair, args.stack_method, cv,
            args.passthrough, n_jobs, args.n, args.d, args.n_classes,
        )
        print(json.dumps({
            "fit_s": fit_s,
            "predict_s": predict_s,
            "checksum": checksum,
            "startup_s": startup_s,
        }))
        return

    print(f"# loadavg={os.getloadavg()[0]:.1f}")
    if os.getloadavg()[0] > 4:
        print("# WARNING: loadavg > 4 — wall-clock cells are contended")

    arms = ["host", "device"] if args.arm == "both" else [args.arm]
    for arm in arms:
        sweep(
            "cv ladder — the base-fit multiplier",
            [(f"cv={cv!r} (binary, auto)", "proba", "auto", cv, False, None, 2)
             for cv in CV_LADDER],
            arm, args.n, args.d, args.repeat,
        )
        sweep(
            "stack_method — the response each member computes",
            [(f"{sm} [{pair}] (5-class, cv=5)", pair, sm, 5, False, None, 5)
             for sm, pair in STACK_METHOD_LADDER],
            arm, args.n, args.d, args.repeat,
        )
        sweep(
            "n_classes — meta width under predict_proba (k * n_classes)",
            [(f"n_classes={c} predict_proba (cv=5)", "proba", "predict_proba", 5,
              False, None, c) for c in N_CLASSES_LADDER],
            arm, args.n, args.d, args.repeat,
        )
        sweep(
            "passthrough — the wider meta matrix",
            [(f"passthrough={p} (5-class, cv=5)", "proba", "predict_proba", 5, p,
              None, 5) for p in PASSTHROUGH_LADDER],
            arm, args.n, args.d, args.repeat,
        )
        sweep(
            "n_jobs — joblib fan-out",
            [(f"n_jobs={j!r} (binary, cv=5)", "proba", "auto", 5, False, j, 2)
             for j in N_JOBS_LADDER],
            arm, args.n, args.d, args.repeat,
        )


if __name__ == "__main__":
    main()
