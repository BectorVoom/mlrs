#!/usr/bin/env python3
"""``mlrs.metrics`` full-parameter performance (METR-PARAM-01) — which of the
new parameters actually moves the clock, and where mlrs stands against sklearn.

Most of the eleven metrics are one O(n) reduction, and most of their parameters
only pick which reduction runs at the end. Those cannot move the clock and are
not swept here. The ones that CAN are the ones that change the amount of work,
and each gets a ``--level``:

    ==================  =========================  ===============================
    ``--level``         parameter under test       why it can matter
    ==================  =========================  ===============================
    ``roc-multiclass``  ``multi_class``,           **the big one.** ``'ovr'`` runs
                        ``average``                ``K`` binary sweeps over ``n``
                                                   samples; ``'ovo'`` runs
                                                   ``K*(K-1)`` sweeps over the
                                                   class-PAIR subsets, so the cost
                                                   grows quadratically in ``K``.
                                                   ``average='micro'`` is a third
                                                   shape again: ONE sweep over
                                                   ``n*K`` raveled pairs
    ``roc-binary``      ``max_fpr``                the partial-AUC path
                                                   materializes the ROC polyline
                                                   instead of streaming the
                                                   integral
    ``prcurve``         ``drop_intermediate``      one extra O(m) mask pass, but a
                                                   much shorter output to egress
    ``prf``             ``average``,               ``average=None`` egresses a
                        ``zero_division``          vector; ``zero_division='warn'``
                                                   must not cost a second pass
    ``confusion``       ``normalize``              O(K^2) after an O(n) tabulation
                                                   — expected to be free
    ``logloss``         ``labels``, ``normalize``  neither should cost anything;
                                                   the level exists to watch the
                                                   per-sample CLASS LOOKUP scale
                                                   with ``--classes``
    ``regression``      ``multioutput``            the 2-D path walks ``n*k``
                                                   elements and reduces ``k``
                                                   columns; ``'variance_weighted'``
                                                   needs the per-output variance
                                                   the others do not
    ==================  =========================  ===============================

Method: every cell is the MINIMUM of ``--repeat`` runs, and the mlrs/sklearn
arms are interleaved inside the repeat loop rather than run in separate blocks,
so a drifting machine perturbs both arms equally. ``--cpu-time`` switches the
clock from wall to ``time.process_time`` (use it when the box is busy — a
loaded machine has inverted a wall-clock verdict in this repo before).

Run:
    python3 scripts/bench_metrics_params.py --level all --repeat 7
    python3 scripts/bench_metrics_params.py --level roc-multiclass --classes 3,5,10,20
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import mlrs.metrics as mm
from sklearn import metrics as sk


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #


def _best(fn, repeat, clock):
    """Minimum of ``repeat`` timed calls of ``fn`` (seconds)."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter() if clock == "wall" else time.process_time()
        fn()
        t1 = time.perf_counter() if clock == "wall" else time.process_time()
        best = min(best, t1 - t0)
    return best


def _pair(mlrs_fn, sk_fn, repeat, clock):
    """Interleaved min-of-``repeat`` for the two arms."""
    best_m = float("inf")
    best_s = float("inf")
    for _ in range(repeat):
        best_m = min(best_m, _best(mlrs_fn, 1, clock))
        best_s = min(best_s, _best(sk_fn, 1, clock))
    return best_m, best_s


def _row(label, t_mlrs, t_sk):
    ratio = t_sk / t_mlrs if t_mlrs > 0 else float("inf")
    verdict = "mlrs" if ratio > 1.0 else "sklearn"
    print(
        f"  {label:<38} mlrs {t_mlrs * 1e3:9.3f} ms   sklearn {t_sk * 1e3:9.3f} ms"
        f"   {ratio:6.2f}x ({verdict})"
    )


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def make_multiclass(n, k, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, k, size=n).astype(np.int64)
    y_pred = np.where(rng.random(n) < 0.3, (y_true + 1) % k, y_true).astype(np.int64)
    proba = rng.random((n, k)) + 0.05
    proba /= proba.sum(axis=1, keepdims=True)
    sw = rng.uniform(0.5, 2.5, size=n)
    return y_true, y_pred, proba, sw


def make_binary(n, seed=0, tied=False):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n).astype(np.int64)
    score = rng.random(n)
    if tied:
        score = np.round(score * 32) / 32.0
    return y_true, score, rng.uniform(0.5, 2.5, size=n)


def make_regression(n, k, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.normal(size=(n, k))
    y_pred = y_true + 0.3 * rng.normal(size=(n, k))
    return y_true, y_pred, rng.uniform(0.5, 2.5, size=n)


# --------------------------------------------------------------------------- #
# levels
# --------------------------------------------------------------------------- #


def level_roc_multiclass(args):
    print(f"\n== roc_auc_score: multi_class x average (n={args.n}) ==")
    for k in args.classes:
        y_true, _, proba, _ = make_multiclass(args.n, k)
        print(f"\n n_classes={k}")
        cells = [
            ("ovr average='macro'", dict(multi_class="ovr", average="macro")),
            ("ovr average='weighted'", dict(multi_class="ovr", average="weighted")),
            ("ovr average='micro'", dict(multi_class="ovr", average="micro")),
            ("ovr average=None", dict(multi_class="ovr", average=None)),
            ("ovo average='macro'", dict(multi_class="ovo", average="macro")),
            ("ovo average='weighted'", dict(multi_class="ovo", average="weighted")),
        ]
        for label, kw in cells:
            t_m, t_s = _pair(
                lambda kw=kw: mm.roc_auc_score(y_true, proba, **kw),
                lambda kw=kw: sk.roc_auc_score(y_true, proba, **kw),
                args.repeat,
                args.clock,
            )
            _row(label, t_m, t_s)


def level_roc_binary(args):
    print(f"\n== roc_auc_score: max_fpr (binary, n={args.n}) ==")
    y_true, score, sw = make_binary(args.n)
    for label, kw in [
        ("max_fpr=None (full AUC)", {}),
        ("max_fpr=1.0 (short-circuits)", dict(max_fpr=1.0)),
        ("max_fpr=0.5 (partial+McClish)", dict(max_fpr=0.5)),
        ("max_fpr=0.1 (partial+McClish)", dict(max_fpr=0.1)),
        ("max_fpr=0.5, sample_weight", dict(max_fpr=0.5, sample_weight=sw)),
    ]:
        t_m, t_s = _pair(
            lambda kw=kw: mm.roc_auc_score(y_true, score, **kw),
            lambda kw=kw: sk.roc_auc_score(y_true, score, **kw),
            args.repeat,
            args.clock,
        )
        _row(label, t_m, t_s)


def level_prcurve(args):
    print(f"\n== precision_recall_curve: drop_intermediate (n={args.n}) ==")
    for tied in (False, True):
        y_true, score, sw = make_binary(args.n, tied=tied)
        tag = "tied scores (33 levels)" if tied else "continuous scores"
        n_full = len(np.unique(score))
        n_drop = len(sk.precision_recall_curve(y_true, score, drop_intermediate=True)[2])
        print(f"\n {tag}: {n_full} distinct thresholds -> {n_drop} kept when dropping")
        for label, kw in [
            ("drop_intermediate=False", dict(drop_intermediate=False)),
            ("drop_intermediate=True", dict(drop_intermediate=True)),
            ("drop_intermediate=True, sw", dict(drop_intermediate=True, sample_weight=sw)),
        ]:
            t_m, t_s = _pair(
                lambda kw=kw: mm.precision_recall_curve(y_true, score, **kw),
                lambda kw=kw: sk.precision_recall_curve(y_true, score, **kw),
                args.repeat,
                args.clock,
            )
            _row(label, t_m, t_s)


def level_prf(args):
    print(f"\n== precision/recall/f1: average, zero_division (n={args.n}) ==")
    for k in args.classes:
        y_true, y_pred, _, sw = make_multiclass(args.n, k)
        print(f"\n n_classes={k}")
        for label, kw in [
            ("average='micro'", dict(average="micro")),
            ("average='macro'", dict(average="macro")),
            ("average='weighted'", dict(average="weighted")),
            ("average=None", dict(average=None)),
            ("average='macro', sample_weight", dict(average="macro", sample_weight=sw)),
            ("average='macro', zero_division=0", dict(average="macro", zero_division=0)),
        ]:
            t_m, t_s = _pair(
                lambda kw=kw: mm.f1_score(y_true, y_pred, **kw),
                lambda kw=kw: sk.f1_score(y_true, y_pred, **kw),
                args.repeat,
                args.clock,
            )
            _row("f1_score " + label, t_m, t_s)


def level_confusion(args):
    print(f"\n== confusion_matrix: normalize (n={args.n}) ==")
    for k in args.classes:
        y_true, y_pred, _, sw = make_multiclass(args.n, k)
        print(f"\n n_classes={k}")
        for label, kw in [
            ("normalize=None", {}),
            ("normalize='true'", dict(normalize="true")),
            ("normalize='pred'", dict(normalize="pred")),
            ("normalize='all'", dict(normalize="all")),
            ("normalize='true', sample_weight", dict(normalize="true", sample_weight=sw)),
        ]:
            t_m, t_s = _pair(
                lambda kw=kw: mm.confusion_matrix(y_true, y_pred, **kw),
                lambda kw=kw: sk.confusion_matrix(y_true, y_pred, **kw),
                args.repeat,
                args.clock,
            )
            _row(label, t_m, t_s)


def level_logloss(args):
    print(f"\n== log_loss: labels / normalize (n={args.n}) ==")
    for k in args.classes:
        y_true, _, proba, sw = make_multiclass(args.n, k)
        classes = list(range(k))
        print(f"\n n_classes={k}")
        for label, kw in [
            ("labels=None", {}),
            ("labels=[0..K-1]", dict(labels=classes)),
            ("labels=[0..K-1], sample_weight", dict(labels=classes, sample_weight=sw)),
            ("normalize=False", dict(normalize=False)),
        ]:
            t_m, t_s = _pair(
                lambda kw=kw: mm.log_loss(y_true, proba, **kw),
                lambda kw=kw: sk.log_loss(y_true, proba, **kw),
                args.repeat,
                args.clock,
            )
            _row(label, t_m, t_s)


def level_regression(args):
    print(f"\n== r2/mse/mae: multioutput (n={args.n}) ==")
    for k in args.outputs:
        y_true, y_pred, sw = make_regression(args.n, k)
        print(f"\n n_outputs={k}")
        weights = np.linspace(1.0, 2.0, k)
        cells = [
            ("mse multioutput='uniform_average'", mm.mean_squared_error, sk.mean_squared_error, {}),
            (
                "mse multioutput='raw_values'",
                mm.mean_squared_error,
                sk.mean_squared_error,
                dict(multioutput="raw_values"),
            ),
            (
                "mse multioutput=[w...]",
                mm.mean_squared_error,
                sk.mean_squared_error,
                dict(multioutput=weights),
            ),
            (
                "mse uniform_average, sample_weight",
                mm.mean_squared_error,
                sk.mean_squared_error,
                dict(sample_weight=sw),
            ),
            ("mae multioutput='uniform_average'", mm.mean_absolute_error, sk.mean_absolute_error, {}),
            ("r2  multioutput='uniform_average'", mm.r2_score, sk.r2_score, {}),
            (
                "r2  multioutput='raw_values'",
                mm.r2_score,
                sk.r2_score,
                dict(multioutput="raw_values"),
            ),
            (
                "r2  multioutput='variance_weighted'",
                mm.r2_score,
                sk.r2_score,
                dict(multioutput="variance_weighted"),
            ),
        ]
        for label, m_fn, s_fn, kw in cells:
            if k == 1 and not isinstance(kw.get("multioutput", ""), str):
                # sklearn: "Custom weights are useful only in multi-output
                # cases." — there is no single-output cell to time.
                continue
            t_m, t_s = _pair(
                lambda m_fn=m_fn, kw=kw: m_fn(y_true, y_pred, **kw),
                lambda s_fn=s_fn, kw=kw: s_fn(y_true, y_pred, **kw),
                args.repeat,
                args.clock,
            )
            _row(label, t_m, t_s)


LEVELS = {
    "roc-multiclass": level_roc_multiclass,
    "roc-binary": level_roc_binary,
    "prcurve": level_prcurve,
    "prf": level_prf,
    "confusion": level_confusion,
    "logloss": level_logloss,
    "regression": level_regression,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", default="all", choices=["all", *LEVELS])
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--classes", default="3,5,10")
    ap.add_argument("--outputs", default="1,4,16")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument(
        "--cpu-time",
        dest="clock",
        action="store_const",
        const="cpu",
        default="wall",
        help="time with process_time instead of the wall clock (busy box)",
    )
    args = ap.parse_args()
    args.classes = [int(x) for x in args.classes.split(",")]
    args.outputs = [int(x) for x in args.outputs.split(",")]

    print(
        f"# bench_metrics_params n={args.n} repeat={args.repeat} clock={args.clock} "
        f"(min of {args.repeat}, arms interleaved)"
    )
    levels = LEVELS.values() if args.level == "all" else [LEVELS[args.level]]
    for fn in levels:
        fn(args)


if __name__ == "__main__":
    main()
