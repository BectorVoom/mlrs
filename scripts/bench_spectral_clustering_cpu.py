#!/usr/bin/env python3
"""SpectralClustering **fit** wall-clock: mlrs (cpu backend) vs scikit-learn.

The SPECTRAL-02 cpu probe, and the labels-side companion to
`bench_spectral_embedding_cpu.py`. Both engines run the same four stages —
build the affinity, form the symmetric-normalized Laplacian, extract its
smallest `n_components` eigenpairs (`drop_first=False` here, unlike the
embedding), then discretize that embedding into labels with k-means — so this
is a like-for-like comparison at library defaults. What differs is HOW the
middle two stages are evaluated:

    sklearn   `kneighbors_graph` (or `rbf_kernel`), then ARPACK in SHIFT-INVERT
              mode (`eigsh(L, k, sigma=-1e-5, which="LM", tol=0)`), which
              factorizes `L - sigma*I` before it can iterate
    mlrs      a KD-tree-routed parallel host kNN scan, the affinity kept SPARSE
              in CSR, and a thick-restart Lanczos whose only contact with the
              matrix is a matvec — so there is no factorization at all

The expected shape is therefore the same as the embedding probe: mlrs' margin
GROWS with `n_samples` (sklearn pays a sparse factorization that scales worse
than a matvec) and the `rbf` rungs favor mlrs further still, because a dense
affinity makes sklearn's factorization dense while mlrs' matvec stays a plain
`O(n^2)` pass. The k-means tail is shared work at the same `n_init=10`, so it
dilutes the ratio without biasing it.

    .venv/bin/python scripts/bench_spectral_clustering_cpu.py [--reps 5] [--check]
                     [--engine mlrs|sklearn|both] [--dtype float64]

The `--engine` caveat from `bench_linear_predict_cpu.py` applies verbatim:
OpenBLAS keeps its workers SPINNING after a call, so interleaving both engines
in one process taxes whichever runs second. Re-run a suspicious rung with
`--engine mlrs` / `--engine sklearn` in separate processes.

On a machine with unrelated load, prefer the default INTERLEAVED schedule
(engines alternate rep by rep, so a load burst hits both) and read the
`cpu (s)` column -- `time.process_time` excludes time the process spent
descheduled, and it is the number to trust on a busy box (the caveat recorded
in `mlrs-hgb-cpu-bench-caveat`). Note that CPU time also CHARGES mlrs for its
worker threads while sklearn's ARPACK stage is single-threaded, so `cpu (s)` is
a deliberately pessimistic reading of the mlrs column rather than a
like-for-like one.

**Correctness here is ARI, never elementwise, and that is not a shortcut.**
Two things make it the only defensible metric:

1. A cluster label is an arbitrary NAME. Two correct engines routinely return
   the same partition under a different permutation of `0..k-1`, so comparing
   label ids directly reports a difference that does not exist.
   `adjusted_rand_score` scores the PARTITION and is invariant to that
   relabeling; `--check` prints it, and flags anything below 0.99.
2. A DISCONNECTED affinity graph makes the embedding degenerate: the Laplacian
   has one zero eigenvalue per connected component, so the kept eigenvectors
   span a fully degenerate null space and are defined only up to an arbitrary
   rotation WITHIN it. sklearn and mlrs then return different — both correct —
   bases, and any elementwise `max|Δembedding|` comparison is meaningless.
   For CLUSTERING this degeneracy is benign, and in fact it is the regime
   spectral clustering is designed for: a rotation preserves the pairwise
   distances between embedding rows, so k-means recovers the same partition
   from either basis and ARI stays 1.0 while the embeddings look nothing alike.
   That is exactly why this script scores partitions rather than values, and
   why the embedding probe (which cannot do that) has to steer AROUND
   disconnection instead.

The corollary drove the choice of fixture, and it inverts the usual intuition:
**`make_blobs`-style well-separated clusters DISCONNECT a kNN graph.** The
tidiest, most obviously-correct-looking data is precisely the data that
disconnects — each blob becomes its own component once every point's `k`
nearest neighbors are its own blob-mates. `bench_spectral_embedding_cpu.py`
therefore has to use uniform-random data to stay connected; this script does
the opposite and generates blobs on purpose, because clustering NEEDS real
structure for ARI to mean anything, and it can tolerate the degeneracy that
the embedding probe cannot. `--check` reports the component count on the kNN
rungs so a rung with MORE components than `n_clusters` is visible: that case
is the one that can genuinely split the two engines, because the retained
`k`-dimensional subspace is then an arbitrary slice of a wider null space and
the rotation-invariance argument above no longer applies.

**Shim parameter coverage.** `mlrs.cluster.SpectralClustering` carries
sklearn's full surface except `kernel_params`, which is provably a no-op for a
string affinity (sklearn overwrites `gamma`/`degree`/`coef0` from the
estimator's own attributes and then filters every other key out in
`pairwise_kernels`). `--assign-labels` therefore selects a real third axis:
`kmeans` is sklearn's default, while `discretize` and `cluster_qr` skip the
k-means tail entirely and so shift where the time goes.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

# (rows, features, n_clusters, affinity) of the timed fit. The ladder walks `n`
# at fixed `d` (isolating the eigensolver, which is where sklearn's shift-invert
# factorization has to pay) and covers both affinities (sparse CSR matvec vs a
# dense one). `--assign-labels` is the third axis, run separately rather than
# crossed into the ladder so a rung line stays comparable to the embedding
# probe's.
CONFIGS = [
    (1_000, 8, 3, "nearest_neighbors"),
    (3_000, 16, 5, "nearest_neighbors"),
    (5_000, 16, 5, "nearest_neighbors"),
    (1_000, 8, 3, "rbf"),
    (2_000, 16, 4, "rbf"),
]


def make_design(n, d, k, seed=0, sep=1.5, spread=0.7):
    """`k` isotropic Gaussian blobs, plus their ground-truth labels.

    Deliberately blobs, and deliberately NOT the uniform-random design that
    `bench_spectral_embedding_cpu.py` uses: clustering needs real structure for
    ARI to mean anything, and it tolerates the kNN-graph disconnection that
    separated blobs cause (module docstring).

    The scale is chosen so that `gamma=1.0` — sklearn's literal default, which
    the shim mirrors — is a sensible bandwidth: centers are random unit
    directions scaled to a pairwise separation of `sep`, and the per-coordinate
    sigma is set so the mean SQUARED within-cluster distance is `spread**2`,
    independent of `d`. At the defaults that puts the rbf weights at roughly
    `exp(-0.49)` within a cluster against `exp(-2.7)` across, i.e. a graph with
    clear block structure and no underflow — an unscaled blob generator would
    drive every off-diagonal rbf entry to a denormal and leave the Laplacian
    numerically equal to the identity.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    # k random directions on a sphere of radius r are ~r*sqrt(2) apart in high
    # d, so scale by sep/sqrt(2) to hit the requested pairwise separation.
    centers *= sep / np.sqrt(2.0)
    sigma = spread / np.sqrt(2.0 * d)
    y = np.repeat(np.arange(k), n // k)
    y = np.concatenate([y, np.full(n - y.size, k - 1)])
    x = centers[y] + sigma * rng.standard_normal((n, d))
    perm = rng.permutation(n)
    return np.ascontiguousarray(x[perm]), np.ascontiguousarray(y[perm])


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


def graph_components(model):
    """Connected components of a fitted sklearn model's `affinity_matrix_`.

    Returns `None` when the count is not cheaply available (a dense affinity is
    kept off this path — an rbf kernel is strictly positive, so it is connected
    by construction unless it underflows, and densifying it for a component
    count would cost more than the rung being timed).
    """
    from scipy.sparse import issparse
    from scipy.sparse.csgraph import connected_components

    a = getattr(model, "affinity_matrix_", None)
    if a is None or not issparse(a):
        return None
    return int(connected_components(a, directed=False, return_labels=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--engine", default="both", choices=["both", "mlrs", "sklearn"])
    ap.add_argument("--n-neighbors", type=int, default=10,
                    help="kNN affinity degree; sklearn's SpectralClustering "
                         "default is a literal 10 (it does NOT resolve None to "
                         "n_samples // 10 the way SpectralEmbedding does)")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="rbf coefficient; sklearn's literal default is 1.0 and "
                         "the shim mirrors it (NOT 1/n_features)")
    ap.add_argument("--assign-labels", default="kmeans",
                    choices=["kmeans", "discretize", "cluster_qr"],
                    help="label-assignment strategy; both engines use the same one")
    ap.add_argument("--check", action="store_true",
                    help="print the adjusted Rand index between the two label "
                         "vectors (labels agree only up to a permutation, so "
                         "ARI is the metric — never compare label ids)")
    ap.add_argument("--configs", default="",
                    help="comma-separated n:d:k:affinity")
    ap.add_argument("--schedule", default="interleaved",
                    choices=["interleaved", "blocked"],
                    help="alternate engines rep by rep (default) or run each "
                         "engine's reps back to back")
    args = ap.parse_args()

    import mlrs
    from mlrs.cluster import SpectralClustering as MlrsEst
    from sklearn.cluster import SpectralClustering as SkEst
    from sklearn.metrics import adjusted_rand_score

    configs = CONFIGS
    if args.configs:
        configs = []
        for c in args.configs.split(","):
            n, d, k, aff = c.split(":")
            configs.append((int(n), int(d), int(k), aff))

    dt = np.float32 if args.dtype == "float32" else np.float64
    print(f"mlrs {mlrs.__name__} | n_neighbors={args.n_neighbors} "
          f"gamma={args.gamma} dtype={args.dtype} reps={args.reps} "
          f"schedule={args.schedule}")
    header = (f"{'n':>7} {'d':>4} {'k':>3} {'affinity':>18} | {'engine':>8} "
              f"{'fit (s)':>10} {'cpu (s)':>10} {'first (s)':>10}")
    print(header)
    print("-" * len(header))

    engines = [e for e in ("mlrs", "sklearn") if args.engine in ("both", e)]

    for n, d, k, aff in configs:
        x_raw, y_true = make_design(n, d, k)
        x = np.ascontiguousarray(x_raw.astype(dt))
        # Everything not named here sits at the SAME default on both sides.
        common = dict(
            n_clusters=k,
            affinity=aff,
            gamma=args.gamma,
            n_neighbors=args.n_neighbors,
            assign_labels=args.assign_labels,
            random_state=0,
        )

        def fit_mlrs():
            m = MlrsEst(**common)
            m.fit(x)
            return m

        def fit_sk():
            with warnings.catch_warnings():
                # The "not fully connected" warning is advisory in sklearn and
                # changes nothing; the rung line reports the component count
                # explicitly instead of letting a warning scroll past.
                warnings.simplefilter("ignore")
                m = SkEst(**common)
                m.fit(x)
                return m

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
            print(f"{n:>7} {d:>4} {k:>3} {aff:>18} | {eng:>8}  FAILED: {msg}")

        ok = [e for e in engines if e not in failed]
        for eng in ok:
            s = samples[eng]
            print(f"{n:>7} {d:>4} {k:>3} {aff:>18} | {eng:>8} "
                  f"{s.best:>10.4f} {s.best_cpu:>10.4f} {s.first:>10.4f}")
        if len(ok) == 2:
            wall_x = samples["sklearn"].best / samples["mlrs"].best
            cpu_x = samples["sklearn"].best_cpu / samples["mlrs"].best_cpu
            note = f"{wall_x:.2f}x wall / {cpu_x:.2f}x cpu vs sklearn"
            mm, sm = samples["mlrs"].model, samples["sklearn"].model
            if args.check:
                a = np.asarray(mm.labels_).ravel()
                b = np.asarray(sm.labels_).ravel()
                # Labels match only up to a permutation of the cluster ids, so
                # the agreement metric MUST be permutation-invariant.
                ari = float(adjusted_rand_score(a, b))
                note += (
                    f" | ARI(mlrs,sklearn) = {ari:.4f}"
                    f" | vs truth {adjusted_rand_score(y_true, a):.4f}"
                    f"/{adjusted_rand_score(y_true, b):.4f}"
                )
                ncc = graph_components(sm)
                if ncc is not None:
                    note += f" | components {ncc}"
                if ari < 0.99:
                    note += (
                        f"  [!] ARI {ari:.4f} < 0.99 — the two engines found "
                        f"DIFFERENT partitions, so the timing above is not a "
                        f"like-for-like number"
                    )
                    if ncc is not None and ncc > k:
                        note += (
                            f"; the graph has {ncc} components > n_clusters={k}, "
                            f"so the kept eigenspace is an arbitrary slice of a "
                            f"wider null space and the two bases need not agree"
                        )
            print(f"{'':>7} {'':>4} {'':>3} {'':>18} | {note}")
        print()


if __name__ == "__main__":
    main()
