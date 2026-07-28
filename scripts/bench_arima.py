#!/usr/bin/env python3
"""ARIMA / AutoARIMA `fit` wall-clock harness — mlrs (cpu backend) vs statsmodels.

sklearn ships no ARIMA, so the reference engine is `statsmodels.tsa.statespace.
sarimax.SARIMAX` — the same library the committed oracle fixture under
`tests/fixtures/arima_seed42.npz` was generated with, and the reference the Rust
`timeseries::arima` module docs verify exact loglikelihood agreement against
(`trend='n'`, `concentrate_scale=True`, default `enforce_stationarity`/
`enforce_invertibility` — see the module docs / `mlrs-arima-landmines` for why
those flags matter: passing `False` silently switches statsmodels to a
different, non-comparable Kalman initialization).

Both engines fit the SAME series with the SAME `(p, d, q)` order, so the number
compares the whole fit path directly: differencing -> Kalman-filter state-space
MLE -> concentrated loglikelihood. `AutoARIMA` additionally times the exhaustive
`(p, q)` grid search (mlrs runs it host-parallel across `std::thread::scope`
workers; statsmodels/pmdarima has no directly equivalent bounded-grid mode, so
the AutoARIMA row instead replays the same grid serially in Python for a
matched comparison).

Log-likelihood is printed alongside the timings as the correctness cross-check
— a speedup that reaches a worse optimum is not a speedup (both MLEs maximize
the SAME concentrated likelihood, so mlrs should be competitive with or better
than statsmodels' llf, per the module's own `fit_band` oracle gate).

    python3 scripts/bench_arima.py                  # default ladder
    python3 scripts/bench_arima.py --n 500 2000      # custom length ladder
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

# (n_obs, order)
CONFIGS = [
    (120, (2, 0, 1)),
    (500, (2, 0, 1)),
    (1_000, (3, 0, 2)),
    (2_000, (5, 0, 5)),
]

AUTO_CONFIGS = [
    (120, 3, 3),
    (500, 3, 3),
    (1_000, 5, 5),
]


def splitmix64(state: int) -> int:
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def make_series(n: int, seed: int = 42) -> np.ndarray:
    """A stationary AR(2)/MA(1)-flavored series (same recursion the Rust
    `arima_perf_test.rs` probe uses), deterministic across engines."""
    s = seed
    y = np.zeros(n)
    e_prev = 0.0
    for t in range(n):
        s = splitmix64(s)
        u = (s >> 11) / float(1 << 53)
        e = (u - 0.5) * 2.0
        prev1 = y[t - 1] if t >= 1 else 0.0
        prev2 = y[t - 2] if t >= 2 else 0.0
        y[t] = 0.5 * prev1 - 0.2 * prev2 + e + 0.3 * e_prev
        e_prev = e
    return y


def sm_fit(y: np.ndarray, order: tuple[int, int, int]):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    p, d, q = order
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(y, order=(p, d, q), trend="n", concentrate_scale=True)
        res = mod.fit(disp=False)
    return res.llf


def sm_auto(y: np.ndarray, d: int, max_p: int, max_q: int):
    best_llf = -np.inf
    best_aicc = np.inf
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                llf = sm_fit(y, (p, d, q))
            except Exception:
                continue
            k = p + q + 1
            n = len(y) - d
            if n - k - 1 <= 0:
                continue
            aic = -2 * llf + 2 * k
            aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
            if aicc < best_aicc:
                best_aicc, best_llf = aicc, llf
    return best_llf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="*", default=None, help="n_obs ladder override")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    import mlrs

    configs = CONFIGS
    if args.n:
        configs = [(n, (2, 0, 1)) for n in args.n]

    # One-time backend init (cubecl-cpu client + first read-back) — process-wide,
    # shared by every mlrs estimator, excluded from the timed rows below.
    warm = make_series(64)
    t0 = time.perf_counter()
    mlrs.ARIMA(order=(1, 0, 0)).fit(warm)
    cold_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    mlrs.ARIMA(order=(1, 0, 0)).fit(warm)
    warm_s = time.perf_counter() - t0
    print(
        f"mlrs one-time backend init: {cold_s - warm_s:.4f} s "
        f"(first call {cold_s:.4f} s, warm {warm_s:.4f} s) — excluded below"
    )

    def timed(fn):
        best = float("inf")
        out = None
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            out = fn()
            best = min(best, time.perf_counter() - t0)
        return out, best

    print("\n== ARIMA.fit (fixed order) ==")
    header = f"{'n':>7} {'order':>10} | {'engine':>10} {'fit(s)':>10} {'speedup':>8} {'llf':>12}"
    print(header)
    print("-" * len(header))
    for n, order in configs:
        y = make_series(n)

        sm_llf, sm_s = timed(lambda: sm_fit(y, order))
        print(f"{n:>7} {str(order):>10} | {'statsmodels':>10} {sm_s:>10.4f} {1.0:>8.2f} {sm_llf:>12.3f}")

        ml_est, ml_s = timed(lambda: mlrs.ARIMA(order=order).fit(y))
        print(
            f"{n:>7} {str(order):>10} | {'mlrs':>10} {ml_s:>10.4f} "
            f"{sm_s / ml_s:>8.2f} {ml_est.llf:>12.3f}"
        )

    print("\n== AutoARIMA (p, q) grid search ==")
    header = f"{'n':>7} {'grid':>8} | {'engine':>10} {'fit(s)':>10} {'speedup':>8} {'llf':>12}"
    print(header)
    print("-" * len(header))
    for n, max_p, max_q in AUTO_CONFIGS:
        y = make_series(n)
        grid = f"{max_p}x{max_q}"

        sm_llf, sm_s = timed(lambda: sm_auto(y, 0, max_p, max_q))
        print(f"{n:>7} {grid:>8} | {'statsmodels':>10} {sm_s:>10.4f} {1.0:>8.2f} {sm_llf:>12.3f}")

        ml_est, ml_s = timed(lambda: mlrs.AutoARIMA(d=0, max_p=max_p, max_q=max_q).fit(y))
        print(
            f"{n:>7} {grid:>8} | {'mlrs':>10} {ml_s:>10.4f} "
            f"{sm_s / ml_s:>8.2f} {ml_est.llf:>12.3f}"
        )


if __name__ == "__main__":
    main()
