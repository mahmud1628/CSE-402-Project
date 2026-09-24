"""Vectorised alpha-random walks.

A walk starts at some node. At each step it stops with probability
``1 - alpha``; otherwise it moves to a random out-neighbour chosen by
``Graph.step``. All walks advance together, one NumPy operation per step, so
the Python overhead is paid once per step and not once per walk.
"""

from __future__ import annotations

import numpy as np

from .graph import Graph
from .sampling import AliasTable


def check_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha (damping factor) must satisfy 0 <= alpha < 1")
    return alpha


def walk_terminals(graph: Graph, starts, alpha: float, rng: np.random.Generator):
    """Run one alpha-random walk from each start node.

    Returns ``(terminals, n_steps)``. ``terminals[i]`` is where walk ``i``
    stopped, and ``n_steps`` is the total number of edges traversed.
    """
    cur = np.array(starts, dtype=np.int64)
    active = np.arange(cur.size)
    n_steps = 0
    while active.size:
        active = active[rng.random(active.size) < alpha]
        if not active.size:
            break
        cur[active] = graph.step(cur[active], rng)
        n_steps += active.size
    return cur, n_steps


def walk_visits(graph: Graph, starts, alpha: float, rng: np.random.Generator, weights=None):
    """Weighted visit counts ``sum_i weights[i] * (#visits of walk i to v)``.

    The start node counts as a visit. Since each visit ends the walk with
    probability ``1 - alpha``, ``(1 - alpha) * visits`` is also an unbiased
    PPR estimator. This is the "every-visit" estimator used by the paper's
    released code.
    """
    n = graph.n
    cur = np.array(starts, dtype=np.int64)
    w = np.ones(cur.size) if weights is None else np.asarray(weights, dtype=np.float64)
    counts = np.bincount(cur, weights=w, minlength=n).astype(np.float64)
    active = np.arange(cur.size)
    n_steps = 0
    buf_nodes, buf_w, buffered = [], [], 0
    while active.size:
        active = active[rng.random(active.size) < alpha]
        if not active.size:
            break
        cur[active] = graph.step(cur[active], rng)
        n_steps += active.size
        buf_nodes.append(cur[active])
        buf_w.append(w[active])
        buffered += active.size
        if buffered > 4_000_000:
            counts += np.bincount(np.concatenate(buf_nodes), np.concatenate(buf_w), minlength=n)
            buf_nodes, buf_w, buffered = [], [], 0
    if buf_nodes:
        counts += np.bincount(np.concatenate(buf_nodes), np.concatenate(buf_w), minlength=n)
    return counts, n_steps


ESTIMATORS = ("terminal", "visits")


def walk_estimate(
    graph: Graph,
    x: np.ndarray,
    alpha: float,
    n_walks: int,
    rng: np.random.Generator,
    estimator: str = "terminal",
    chunk_size: int = 2_000_000,
):
    """Unbiased Monte Carlo estimate of ``Pi x`` for a (possibly signed) vector x.

    ``Pi = (1 - alpha)(I - alpha P)^{-1}`` is the PPR matrix, whose columns are
    single-source PPR vectors, so ``Pi sigma = pi_sigma``.

    Start nodes are drawn from ``|x| / ||x||_1``. With the terminal estimator,
    walk i adds ``sgn(x(s_i)) * ||x||_1 / n_walks`` at the node where it stops
    (paper Lemma 3.13). With ``x = sigma`` this is plain MCW (Algorithm 3);
    with ``x`` equal to a residual, it is the correction term used by PPW.

    Returns ``(estimate, n_steps)``.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}")
    n_walks = int(n_walks)
    est = np.zeros(graph.n)
    norm = float(np.abs(x).sum())
    if n_walks <= 0 or norm == 0.0:
        return est, 0
    table = AliasTable(np.abs(x))
    signed = bool(np.any(x < 0))
    n_steps = 0
    done = 0
    while done < n_walks:
        b = min(chunk_size, n_walks - done)
        starts = table.sample(b, rng)
        sgn = np.sign(x[starts]) if signed else None
        if estimator == "terminal":
            ends, steps = walk_terminals(graph, starts, alpha, rng)
            est += np.bincount(ends, weights=sgn, minlength=graph.n)
        else:
            counts, steps = walk_visits(graph, starts, alpha, rng, weights=sgn)
            est += (1.0 - alpha) * counts
        n_steps += steps
        done += b
    est *= norm / n_walks
    return est, n_steps
