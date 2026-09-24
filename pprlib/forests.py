"""Spanning-forest (SF) Monte Carlo estimators (paper Sections 2.1 and 4).

A rooted spanning forest F is sampled by the loop-erased alpha-random walk
(Algorithm 1, a variant of Wilson's algorithm). Nodes are processed in
order, and from each node not yet in the forest a walk runs until it either
stops (the node becomes a root, with probability 1 - alpha per step) or hits
the forest. Loops are erased by overwriting ``next[u]``. For every node u,
``root(u)`` has the same distribution as the stopping node of an
alpha-random walk from u. Two unbiased estimators of ``Pi x`` follow:

* root estimator (x~, Lemma 4.1): ``sum_u x(u) e_root(u)``. Works on any graph.
* degree estimator (x-dot, Lemma 4.1, undirected only): inside each tree,
  spread the tree's total mass proportionally to degree,
  ``x-dot(u) = d(u) * sum_{v in T(u)} x(v) / sum_{v in T(u)} d(v)``.
  Its variance is strictly smaller (Lemma 4.3).

Algorithms (proposal alpha convention):

    MCF / MCFV : average over T' forests                      (Algorithm 6)
    PF  / PFV  : then K power iterations                      (Algorithm 7)
    PPF / PPFV : progressive, residual-driven batches         (Algorithm 8)

The "V" versions use the degree estimator. Here they are selected with
``variant="degree"``.
"""

from __future__ import annotations

import time

import numpy as np

from ._accel import njit
from .graph import Graph
from .power import propagate
from .result import PPRResult
from .sampling import make_rng
from .sources import check_distribution
from .variance_reduced import residual
from .walks import check_alpha

VARIANTS = ("root", "degree")


@njit(cache=True)
def _loop_erased_forest(indptr, indices, local_cum, weighted, uniform_dangling, alpha, seed):
    np.random.seed(seed)
    n = indptr.size - 1
    in_forest = np.zeros(n, dtype=np.bool_)
    nxt = np.full(n, -1, dtype=np.int64)
    root = np.full(n, -1, dtype=np.int64)
    steps = 0
    for i in range(n):
        u = i
        while not in_forest[u]:
            if np.random.random() >= alpha:  # the walk stops, so u becomes a root
                in_forest[u] = True
                root[u] = u
            else:
                lo = indptr[u]
                hi = indptr[u + 1]
                if hi == lo:  # dangling node
                    v = np.random.randint(0, n) if uniform_dangling else u
                elif not weighted:
                    v = indices[lo + int(np.random.random() * (hi - lo))]
                else:
                    x = np.random.random()
                    a, b = lo, hi - 1
                    while a < b:  # first position whose cumulative probability exceeds x
                        mid = (a + b) // 2
                        if local_cum[mid] > x:
                            b = mid
                        else:
                            a = mid + 1
                    v = indices[a]
                nxt[u] = v
                u = v
                steps += 1
        r = root[u]
        u = i
        while not in_forest[u]:
            root[u] = r
            in_forest[u] = True
            u = nxt[u]
    return root, steps


def sample_forest(graph: Graph, alpha: float, rng=None):
    """Sample a rooted spanning forest and return ``(root, n_steps)``, where ``root[u]`` is u's root."""
    alpha = check_alpha(alpha)
    rng = make_rng(rng)
    seed = int(rng.integers(0, 2**31 - 1))
    root, steps = _loop_erased_forest(
        graph.indptr, graph.indices, graph.local_cum, graph.weighted,
        graph.dangling == "uniform", alpha, seed,
    )
    return root, int(steps)


def _check_variant(graph: Graph, variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if variant == "degree":
        if not graph.is_undirected():
            raise ValueError("the degree (V) estimator requires an undirected graph")
        if graph.dangling == "uniform" and graph.dangling_nodes.size:
            raise ValueError("the degree (V) estimator needs dangling='self_loop' (reversibility)")


def forest_estimate(graph: Graph, root: np.ndarray, x: np.ndarray, variant: str = "root") -> np.ndarray:
    """Estimate ``Pi x`` from one forest, with the root estimator or the degree estimator."""
    n = graph.n
    if variant == "root":
        return np.bincount(root, weights=x, minlength=n)
    d = graph.forest_degree
    mass = np.bincount(root, weights=x, minlength=n)
    dsum = np.bincount(root, weights=d, minlength=n)
    return d * mass[root] / dsum[root]


def _forest_average(graph, x, alpha, n_forests, rng, variant):
    est = np.zeros(graph.n)
    steps = 0
    for _ in range(n_forests):
        root, s = sample_forest(graph, alpha, rng)
        est += forest_estimate(graph, root, x, variant)
        steps += s
    return est / max(n_forests, 1), steps


def mcf(graph: Graph, sigma, alpha: float, n_forests: int, rng=None, *, variant: str = "root") -> PPRResult:
    """MCF (``variant="root"``) or MCFV (``variant="degree"``): the plain forest average."""
    alpha = check_alpha(alpha)
    sigma = check_distribution(sigma, graph.n)
    _check_variant(graph, variant)
    rng = make_rng(rng)
    t0 = time.perf_counter()
    est, steps = _forest_average(graph, sigma, alpha, int(n_forests), rng, variant)
    return PPRResult(
        estimate=est, method="mcf" + ("v" if variant == "degree" else ""), alpha=alpha,
        time=time.perf_counter() - t0, n_forests=int(n_forests), n_walk_steps=steps,
    )


def pf(graph: Graph, sigma, alpha: float, n_forests: int, K: int, rng=None, *, variant: str = "root") -> PPRResult:
    """PF / PFV: forest average followed by K power iterations."""
    alpha = check_alpha(alpha)
    sigma = check_distribution(sigma, graph.n)
    _check_variant(graph, variant)
    rng = make_rng(rng)
    t0 = time.perf_counter()
    est, steps = _forest_average(graph, sigma, alpha, int(n_forests), rng, variant)
    est = propagate(graph, sigma, alpha, est, K)
    return PPRResult(
        estimate=est, method="pf" + ("v" if variant == "degree" else ""), alpha=alpha,
        time=time.perf_counter() - t0, n_forests=int(n_forests), n_walk_steps=steps,
        n_power_iterations=int(K),
    )


def ppf(
    graph: Graph, sigma, alpha: float, n_forests: int, K: int, n_batches: int = 3,
    rng=None, *, variant: str = "root",
) -> PPRResult:
    """PPF / PPFV: residual-driven batches of forests, with K power iterations after each batch."""
    alpha = check_alpha(alpha)
    sigma = check_distribution(sigma, graph.n)
    _check_variant(graph, variant)
    rng = make_rng(rng)
    t0 = time.perf_counter()
    sizes = [len(c) for c in np.array_split(np.arange(int(n_forests)), n_batches)]
    x = np.zeros(graph.n)
    res_norms, steps = [], 0
    for b in range(n_batches):
        r = sigma if b == 0 else residual(graph, sigma, alpha, x)
        res_norms.append(float(np.abs(r).sum()))
        if sizes[b]:
            corr, s = _forest_average(graph, r, alpha, sizes[b], rng, variant)
            steps += s
            x = x + corr
        x = propagate(graph, sigma, alpha, x, K)
    res_norms.append(float(np.abs(residual(graph, sigma, alpha, x)).sum()))
    return PPRResult(
        estimate=x, method="ppf" + ("v" if variant == "degree" else ""), alpha=alpha,
        time=time.perf_counter() - t0, n_forests=int(n_forests), n_walk_steps=steps,
        n_power_iterations=int(K) * n_batches + n_batches,
        history={"residual_l1": np.array(res_norms), "batch_sizes": sizes},
    )
