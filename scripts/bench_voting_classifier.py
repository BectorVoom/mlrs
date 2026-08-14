#!/usr/bin/env python3
"""VotingClassifier performance (VOTE-CLF-01) — the parameters that move the clock.

``mlrs.VotingClassifier(estimators, *, voting='hard', weights=None, n_jobs=None,
flatten_transform=True, verbose=False)``. Four of those five can change how long
a call takes, and this harness measures each at the level where it acts:

    ====================  ============  ==========================================
    parameter             ``--level``   what it decides
    ====================  ============  ==========================================
    ``voting``            ``agg``,      **the big one.** ``'hard'`` and ``'soft'``
                          ``call``      run entirely different aggregations over
                                        entirely different data (``n x k``
                                        integer labels versus ``k`` blocks of
                                        ``n x n_classes`` floats), so they are
                                        not two settings of one code path — they
                                        are two code paths
    (the arm)             ``agg``       ``MLRS_VOTING_ENGINE`` — which of numpy /
                                        Rust-host / CubeCL performs it
    ``weights``           ``agg``       ``np.bincount(weights=...)`` /
                                        ``np.average(weights=...)`` versus the
                                        uniform fast paths
    ``flatten_transform`` ``agg``       under soft voting, whether ``transform``
                                        copies (``np.hstack``) or returns the
                                        stack untouched. ``--mode
                                        soft-transform`` times the copy; the
                                        ``False`` case has nothing to time
    ``n_jobs``            ``fit``       the joblib fan-out over the member fits.
                                        Run it BOTH ways (see ``--balanced``): a
                                        voting ensemble fits each member once, so
                                        the ceiling is Amdahl's `total / slowest`
                                        and an imbalanced pool reports "n_jobs
                                        does nothing" when nothing is in fact
                                        being left on the table
    ``verbose``           —             one ``print`` per member; not measured,
                                        it cannot be a measurable share of a fit
    ====================  ============  ==========================================

``n_classes`` is not a parameter but it scales the soft route linearly (a member
contributes ``n_classes`` columns instead of one) and leaves the hard route
almost untouched, so it is swept with ``--classes``.

## Why hard voting is the interesting cell

sklearn's hard route is

    np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=w)), 1, preds)

— a PYTHON-LEVEL loop over `n` rows, allocating a fresh ``bincount`` array per
row. It is the one place in either voting estimator where sklearn is not already
running vectorised numpy. ``docs/voting.md`` concluded that the regressor's
``np.average`` is hard to beat precisely because it IS vectorised; this is the
opposite situation, and the ladder is here to say by how much.

The soft route, by contrast, is `np.average` over a 3-D stack — the same
reduction ``bench_voting.py`` already measured, with ``n * n_classes`` elements
per member instead of ``n``. Its one structurally new opportunity is
``soft-predict``, where the device arm FUSES the argmax into the reduction and
never downloads the ``(n, C)`` average at all.

## Two levels, because they answer different questions

* ``--level agg`` (default) — the aggregation ALONE, on synthetic member
  responses. This is the arm comparison and the ``voting`` comparison,
  undiluted, and where ``weights`` and ``flatten_transform`` live.
* ``--level call`` — a whole ``predict`` (members included) per ``voting`` value,
  which says how much of a real call the aggregation actually is.
* ``--level fit`` — a whole ``VotingClassifier.fit`` per ``n_jobs``.

## Measurement discipline

Every cell runs in a FRESH subprocess (``--cell``) and the harness reports the
MINIMUM of ``--repeat`` runs; cells are interleaved across arms rather than run
in blocks, so a drifting machine penalizes all of them equally. Both rules are
load-bearing on this project's cpu backend, where in-process interleaving and a
busy box have each inverted a verdict before.

The one-time ``_mlrs`` extension load is warmed OUTSIDE every timed region and
reported separately: it is ~35-95 ms, which is larger than the entire
aggregation at most sizes.

Usage
-----
    python3 scripts/bench_voting_classifier.py                      # hard predict
    python3 scripts/bench_voting_classifier.py --mode soft-proba
    python3 scripts/bench_voting_classifier.py --mode soft-predict
    python3 scripts/bench_voting_classifier.py --mode soft-transform
    python3 scripts/bench_voting_classifier.py --weights            # ...weighted
    python3 scripts/bench_voting_classifier.py --classes 2,3,10,50
    python3 scripts/bench_voting_classifier.py --level call         # hard vs soft
    python3 scripts/bench_voting_classifier.py --level fit --balanced
    python3 scripts/bench_voting_classifier.py --repeat 7 --cpu-time

Requires numpy + sklearn + mlrs (built for the backend under test).
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

SEED = 42
ARMS = ["numpy", "host", "device"]

#: The four aggregations `voting` selects between, by the name `--mode` uses.
MODES = ["hard-predict", "soft-proba", "soft-predict", "soft-transform"]

#: ``(n_rows, k_members)`` for the aggregation level. Walks from "smaller than
#: the FFI crossing" to "big enough for bandwidth to matter", and separately
#: walks ``k`` at fixed ``n`` — ``k`` is what sets the upload/download ratio the
#: device arm's whole case rests on.
AGG_LADDER = [
    (1_000, 3),
    (10_000, 3),
    (100_000, 3),
    (1_000_000, 3),
    (100_000, 2),
    (100_000, 8),
    (1_000_000, 8),
]

#: ``(n_rows, n_features, k_members)`` for the end-to-end levels.
FIT_LADDER = [
    (10_000, 32, 4),
    (100_000, 32, 4),
    (100_000, 128, 4),
]

#: ``n_jobs`` values swept at ``--level fit``. Members are sklearn estimators,
#: because a joblib fan-out over an mlrs member is reduced to serial BY DESIGN
#: (`_effective_n_jobs`) and would measure the warning rather than the fan-out.
N_JOBS = [None, 2, 4]


# --------------------------------------------------------------------------- #
# synthetic member responses
# --------------------------------------------------------------------------- #


def label_columns(n, k, n_classes):
    """``k`` encoded label columns of ``n`` rows — what hard voting aggregates.

    Each member agrees with a hidden truth ~70% of the time and errs in its own
    direction otherwise. Two shapes are deliberately avoided:

    * identical columns — the tally is unanimous, the argmax exits on the first
      bin, and the ladder would time a best case no real ensemble has;
    * a fixed per-member OFFSET — every row is then a k-way tie, the argmax
      always answers 0, and the checksum degenerates to zero, which also
      destroys the cross-arm agreement check.
    """
    rng = np.random.default_rng(SEED)
    truth = rng.integers(0, n_classes, size=n, dtype=np.int64)
    cols = []
    for j in range(k):
        wrong = (truth + 1 + (j % max(n_classes - 1, 1))) % n_classes
        cols.append(np.where(rng.random(n) < 0.7, truth, wrong).astype(np.int64))
    return cols


def proba_blocks(n, k, n_classes, dtype):
    """``k`` row-stochastic ``n x n_classes`` blocks — what soft voting
    aggregates."""
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(k):
        block = rng.random((n, n_classes)).astype(dtype)
        out.append(block / block.sum(axis=1, keepdims=True))
    return out


# --------------------------------------------------------------------------- #
# the four aggregations, per arm
# --------------------------------------------------------------------------- #


def time_agg(arm, n, k, n_classes, dtype, reps, weighted, mode):
    """Seconds for one aggregation on `arm`, min over `reps` in-cell.

    The in-cell minimum is on top of the across-process minimum, not instead of
    it: the aggregation is short enough at small ``n`` that a single sample is
    mostly scheduler noise.
    """
    import mlrs

    weights = [1.0 + j * 0.5 for j in range(k)] if weighted else None

    if mode == "hard-predict":
        cols = label_columns(n, k, n_classes)
        if arm == "numpy":
            def once():
                preds = np.asarray(cols).T
                return np.apply_along_axis(
                    lambda row: np.argmax(np.bincount(row, weights=weights)), 1, preds
                )
        else:
            def once():
                return mlrs.ensemble._vote_labels_via_rust(cols, weights, arm, n_classes)
    else:
        blocks = proba_blocks(n, k, n_classes, dtype)
        if mode == "soft-proba":
            if arm == "numpy":
                def once():
                    return np.average(np.asarray(blocks), axis=0, weights=weights)
            else:
                def once():
                    return mlrs.ensemble._vote_proba_via_rust(
                        blocks, "proba", weights, arm
                    )
        elif mode == "soft-predict":
            if arm == "numpy":
                def once():
                    return np.argmax(
                        np.average(np.asarray(blocks), axis=0, weights=weights), axis=1
                    )
            else:
                def once():
                    return mlrs.ensemble._vote_proba_via_rust(
                        blocks, "predict", weights, arm
                    )
        else:  # soft-transform — `flatten_transform=True`'s copy
            if arm == "numpy":
                def once():
                    return np.hstack(blocks)
            else:
                def once():
                    return mlrs.ensemble._vote_proba_via_rust(blocks, "hstack", None, arm)

    warm = once()
    if warm is None:
        raise RuntimeError(f"arm {arm!r} declined these columns")

    best, best_cpu = float("inf"), float("inf")
    for _ in range(reps):
        w0, c0 = time.perf_counter(), time.process_time()
        out = once()
        best = min(best, time.perf_counter() - w0)
        best_cpu = min(best_cpu, time.process_time() - c0)
    return best, best_cpu, float(np.asarray(out, dtype=np.float64).sum())


# --------------------------------------------------------------------------- #
# whole calls
# --------------------------------------------------------------------------- #


def build_problem(n, d, n_classes, dtype):
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n, d)).astype(dtype)
    w = rng.standard_normal(d).astype(dtype)
    score = X @ w
    edges = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])
    return X, np.searchsorted(edges, score).astype(np.int64)


def member_pool(k, balanced):
    """The composed members for the `call` / `fit` levels.

    **Every member here has a LINEAR-time `predict`, deliberately.** The obvious
    fourth pick is `KNeighborsClassifier`, and it makes the ladder unusable: its
    `predict` is O(n_query x n_train), so at n = 10^5 one cell runs for minutes
    and the members drown the aggregation by four orders of magnitude rather than
    the two this level exists to quantify. Meta-estimator ladders amplify member
    costs — time one member at the ladder's shape before adding it.

    `ExtraTreesClassifier` stands in as the fourth: a different algorithm from
    the other three (so the members genuinely disagree) with a predict that is
    linear in `n`.
    """
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier

    if balanced:
        return [
            (f"tree{i}", DecisionTreeClassifier(max_depth=8, random_state=SEED + i))
            for i in range(k)
        ]
    return [
        ("nb", GaussianNB()),
        ("lr", LogisticRegression(max_iter=200)),
        ("tree", DecisionTreeClassifier(max_depth=8, random_state=SEED)),
        ("extra", ExtraTreesClassifier(n_estimators=10, max_depth=8, random_state=SEED)),
    ][:k]


def time_call(voting, n, d, k, n_classes, dtype, reps):
    """Seconds for a whole ``predict`` at one ``voting`` value, members included.

    The FIT is excluded (done once, outside the timed region): what is being
    compared is the cost of the two routes' response collection AND their
    aggregation, which is what a caller choosing ``voting`` actually pays.
    """
    import mlrs

    X, y = build_problem(n, d, n_classes, dtype)
    est = mlrs.VotingClassifier(member_pool(k, balanced=False), voting=voting)
    est.fit(X, y)
    est.predict(X[:64])

    best, best_cpu = float("inf"), float("inf")
    for _ in range(reps):
        w0, c0 = time.perf_counter(), time.process_time()
        out = est.predict(X)
        best = min(best, time.perf_counter() - w0)
        best_cpu = min(best_cpu, time.process_time() - c0)
    return best, best_cpu, float(np.asarray(out, dtype=np.float64).sum())


def time_fit(n, d, k, n_classes, n_jobs, dtype, reps, balanced, voting):
    """Seconds for a whole ``VotingClassifier.fit`` at one ``n_jobs``.

    Two member sets, because they answer different questions:

    * the MIXED pool (default) is what an ensemble usually looks like — members
      of wildly different cost. Its speedup ceiling is Amdahl's, `total /
      slowest`. A ladder run only on this pool would report "n_jobs does
      nothing" and hide WHY.
    * the BALANCED pool (``--balanced``) is `k` copies of the same tree with
      different seeds, so the ceiling is `k`.

    ``voting`` does NOT affect a fit — every member is fitted the same way
    either way — so this level is run at the default and the claim is asserted
    by the ``--level call`` ladder instead.
    """
    import mlrs

    X, y = build_problem(n, d, n_classes, dtype)
    pool = member_pool(k, balanced)

    def build():
        return mlrs.VotingClassifier(pool, n_jobs=n_jobs, voting=voting)

    build().fit(X[:256], y[:256])

    best, best_cpu = float("inf"), float("inf")
    for _ in range(reps):
        est = build()
        w0, c0 = time.perf_counter(), time.process_time()
        est.fit(X, y)
        best = min(best, time.perf_counter() - w0)
        best_cpu = min(best_cpu, time.process_time() - c0)
    return best, best_cpu, float(np.asarray(est.predict(X[:64]), dtype=np.float64).mean())


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #


def cell(args):
    """One measurement, in this fresh interpreter, as a JSON line."""
    os.environ["MLRS_VOTING_ENGINE"] = args.arm

    t0 = time.perf_counter()
    import mlrs

    mlrs._load_ext()
    startup_s = time.perf_counter() - t0

    resolved = mlrs._load_ext().voting_engine()
    # A knob that did not take would make the whole sweep a comparison of numpy
    # against numpy and report it as "no difference". Fail loudly instead
    # (`mlrs-bench-verify-knob-is-live`).
    if resolved != args.arm:
        print(json.dumps({"error": f"knob resolved to {resolved!r}, wanted {args.arm!r}"}))
        return 1

    dtype = np.float64 if mlrs.backend_supports_f64() else np.float32
    if args.level == "agg":
        seconds, cpu_seconds, checksum = time_agg(
            args.arm, args.n, args.k, args.n_classes, dtype,
            args.inner, args.weights, args.mode,
        )
    elif args.level == "call":
        seconds, cpu_seconds, checksum = time_call(
            args.voting, args.n, args.d, args.k, args.n_classes, dtype, args.inner
        )
    else:
        n_jobs = None if args.n_jobs == 0 else args.n_jobs
        seconds, cpu_seconds, checksum = time_fit(
            args.n, args.d, args.k, args.n_classes, n_jobs, dtype,
            args.inner, args.balanced, args.voting,
        )

    print(
        json.dumps(
            {
                "arm": args.arm,
                "seconds": seconds,
                "cpu_seconds": cpu_seconds,
                "checksum": checksum,
                "startup_s": startup_s,
                "dtype": np.dtype(dtype).name,
                "loadavg": os.getloadavg()[0],
            }
        )
    )
    return 0


def spawn(args, *, arm, n, k, n_classes, d=0, n_jobs=0, voting="hard"):
    """Run one cell in a FRESH interpreter and parse its JSON line."""
    argv = [
        sys.executable, __file__, "--cell",
        "--arm", arm, "--level", args.level, "--mode", args.mode,
        "--voting", voting,
        "--n", str(n), "--k", str(k), "--d", str(d),
        "--n-classes", str(n_classes),
        "--n-jobs", str(n_jobs), "--inner", str(args.inner),
    ]
    if args.weights:
        argv.append("--weights")
    if args.balanced:
        argv.append("--balanced")
    out = subprocess.run(argv, capture_output=True, text=True)
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            payload = json.loads(line)
            if "seconds" not in payload:
                raise RuntimeError(payload.get("error", "cell produced no timing"))
            return payload
    return None


def warn_if_contended(args):
    if not args.cpu_time and os.getloadavg()[0] > 4:
        # A co-tenanted box has INVERTED a verdict on this project before
        # (`mlrs-cpu-bench-separate-processes`). Say so on the output itself, so
        # a ladder read months later carries its own caveat.
        print("# WARNING: loadavg > 4 — wall-clock cells are contended; re-run with --cpu-time")


def sweep_agg(args):
    clock = "cpu_seconds" if args.cpu_time else "seconds"
    for n_classes in args.classes:
        print(
            f"# level=agg mode={args.mode} n_classes={n_classes} "
            f"weights={'yes' if args.weights else 'no'} repeat={args.repeat} "
            f"inner={args.inner} clock={'CPU time' if args.cpu_time else 'wall'} "
            f"loadavg={os.getloadavg()[0]:.1f}"
        )
        warn_if_contended(args)
        header = (
            f"{'config':<24}{'numpy':>12}{'host':>12}{'device':>12}"
            f"{'host/np':>10}{'dev/np':>10}"
        )
        print(header)
        print("-" * len(header))

        startups, dtypes = [], set()
        for n, k in AGG_LADDER:
            best = {arm: float("inf") for arm in ARMS}
            checks = {}
            for _ in range(args.repeat):
                for arm in ARMS:  # interleaved, not blocked
                    got = spawn(args, arm=arm, n=n, k=k, n_classes=n_classes)
                    if got is None:
                        continue
                    best[arm] = min(best[arm], got[clock])
                    checks[arm] = got["checksum"]
                    startups.append(got["startup_s"])
                    dtypes.add(got["dtype"])

            print(
                f"{f'n={n:,} k={k}':<24}"
                f"{fmt(best['numpy'])}{fmt(best['host'])}{fmt(best['device'])}"
                f"{ratio(best['numpy'], best['host'])}{ratio(best['numpy'], best['device'])}"
            )
            # A checksum split means an arm computed something else; the ladder
            # is meaningless then, so say so on the line itself.
            if len({round(v, 6) for v in checks.values()}) > 1:
                print(f"    !! checksums disagree: {checks}")

        report_startup(startups)
        print(f"# dtype: {', '.join(sorted(dtypes)) or 'n/a'}")
        print("# ratios are numpy/arm — above 1.00x means the arm BEAT numpy\n")


def sweep_call(args):
    """``predict`` end to end, hard against soft, on one arm.

    This is the ladder that says what ``voting`` costs a CALLER — the
    aggregation ladder above is the same question with the members' own
    ``predict``/``predict_proba`` removed.
    """
    clock = "cpu_seconds" if args.cpu_time else "seconds"
    for n_classes in args.classes:
        print(
            f"# level=call arm={args.arm} n_classes={n_classes} repeat={args.repeat} "
            f"inner={args.inner} clock={'CPU time' if args.cpu_time else 'wall'} "
            f"loadavg={os.getloadavg()[0]:.1f}"
        )
        warn_if_contended(args)
        header = f"{'config':<28}{'hard':>14}{'soft':>14}{'soft/hard':>12}"
        print(header)
        print("-" * len(header))

        startups = []
        for n, d, k in FIT_LADDER:
            best = {}
            for voting in ("hard", "soft"):
                b = float("inf")
                for _ in range(args.repeat):
                    got = spawn(
                        args, arm=args.arm, n=n, k=k, d=d,
                        n_classes=n_classes, voting=voting,
                    )
                    if got is None:
                        continue
                    b = min(b, got[clock])
                    startups.append(got["startup_s"])
                best[voting] = b
            rel = (
                f"{best['soft'] / best['hard']:.2f}x"
                if best["hard"] not in (0, float("inf"))
                else "n/a"
            )
            print(
                f"{f'n={n:,} d={d} k={k}':<28}"
                f"{fmt_ms(best['hard'])}{fmt_ms(best['soft'])}{rel:>12}"
            )
        report_startup(startups)
        print("# soft/hard above 1.00x means soft voting's whole predict is SLOWER\n")


def sweep_fit(args):
    clock = "cpu_seconds" if args.cpu_time else "seconds"
    print(
        f"# level=fit arm={args.arm} voting={args.voting} "
        f"members={'balanced' if args.balanced else 'mixed'} "
        f"repeat={args.repeat} inner={args.inner} "
        f"clock={'CPU time' if args.cpu_time else 'wall'} loadavg={os.getloadavg()[0]:.1f}"
    )
    warn_if_contended(args)
    cols = "".join(f"{('n_jobs=' + str(j)):>14}" for j in N_JOBS)
    header = f"{'config':<28}{cols}{'best/serial':>13}"
    print(header)
    print("-" * len(header))

    startups = []
    for n, d, k in FIT_LADDER:
        best = {}
        for j in N_JOBS:
            b = float("inf")
            for _ in range(args.repeat):
                got = spawn(
                    args, arm=args.arm, n=n, k=k, d=d,
                    n_classes=args.classes[0],
                    n_jobs=0 if j is None else j, voting=args.voting,
                )
                if got is None:
                    continue
                b = min(b, got[clock])
                startups.append(got["startup_s"])
            best[j] = b
        serial = best[None]
        fastest = min(v for v in best.values() if v != float("inf"))
        row = "".join(
            f"{(f'{best[j] * 1000:.1f}m' if best[j] != float('inf') else 'n/a'):>14}"
            for j in N_JOBS
        )
        speedup = f"{serial / fastest:.2f}x" if fastest else "n/a"
        print(f"{f'n={n:,} d={d} k={k}':<28}{row}{speedup:>13}")

    report_startup(startups)
    print("# best/serial above 1.00x means SOME n_jobs beat the serial fit")


def fmt(v):
    return f"{v * 1000:>11.3f}m" if v != float("inf") else f"{'n/a':>12}"


def fmt_ms(v):
    return f"{v * 1000:>13.2f}m" if v != float("inf") else f"{'n/a':>14}"


def ratio(base, v):
    if v == float("inf") or base == float("inf") or v == 0:
        return f"{'n/a':>10}"
    return f"{base / v:>9.2f}x"


def report_startup(startups):
    if startups:
        print(
            f"\n# _mlrs load (excluded from every cell): "
            f"min {min(startups) * 1000:.1f} ms, max {max(startups) * 1000:.1f} ms"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--arm", choices=ARMS, default="numpy")
    p.add_argument("--level", choices=["agg", "call", "fit"], default="agg")
    p.add_argument(
        "--mode",
        choices=MODES,
        default="hard-predict",
        help="which aggregation to time; the four `voting` x method combinations",
    )
    p.add_argument(
        "--voting",
        choices=["hard", "soft"],
        default="hard",
        help="the `voting` value for the `call` / `fit` levels",
    )
    p.add_argument(
        "--weights",
        action="store_true",
        help="use a non-uniform weight vector (the weighted bincount / average path)",
    )
    p.add_argument(
        "--balanced",
        action="store_true",
        help=(
            "fit level: use k equal-cost members instead of the mixed pool, so "
            "the n_jobs ceiling is k rather than Amdahl's"
        ),
    )
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--k", type=int, default=3, help="member count")
    p.add_argument("--d", type=int, default=32, help="feature count (call/fit levels)")
    p.add_argument("--n-classes", type=int, default=3)
    p.add_argument(
        "--classes",
        default="3",
        help="comma-separated class counts to sweep (soft voting scales with it)",
    )
    p.add_argument("--n-jobs", type=int, default=0, help="0 means None")
    p.add_argument("--repeat", type=int, default=3, help="fresh processes per cell")
    p.add_argument("--inner", type=int, default=5, help="in-process reps per cell")
    p.add_argument(
        "--cpu-time",
        action="store_true",
        help="report process CPU time instead of wall clock (use on a loaded box)",
    )
    args = p.parse_args()
    args.classes = [int(c) for c in str(args.classes).split(",") if c]

    if args.cell:
        return cell(args)
    if args.level == "agg":
        sweep_agg(args)
    elif args.level == "call":
        sweep_call(args)
    else:
        sweep_fit(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
