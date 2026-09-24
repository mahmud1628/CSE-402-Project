"""Deterministic power iteration.

Proposal convention::

    r^(k+1) = alpha * P r^(k) + (1 - alpha) * s,    stop when ||r^(k+1) - r^(k)|| < eps.

Convergence facts in the L1 norm, using ``||P||_1 = 1`` for a
column-stochastic P:

* a priori:     ||r^(k) - pi||_1 <= alpha^k ||r^(0) - pi||_1 <= 2 alpha^k
* a posteriori: ||r^(k+1) - pi||_1 <= alpha / (1 - alpha) * ||r^(k+1) - r^(k)||_1
"""

from __future__ import annotations

import time

import numpy as np

from .graph import Graph
from .result import PPRResult
from .sources import check_distribution
from .walks import check_alpha

NORMS = {
    "l1": lambda v: float(np.abs(v).sum()),
    "l2": lambda v: float(np.sqrt(np.dot(v, v))),
    "linf": lambda v: float(np.abs(v).max()),
}


def propagate(graph: Graph, sigma: np.ndarray, alpha: float, x: np.ndarray, K: int) -> np.ndarray:
    """Apply K power-iteration steps ``x <- (1 - alpha) sigma + alpha P x`` to x.

    Expanding gives
    ``sum_{k<K} (1-alpha) alpha^k P^k sigma + alpha^K P^K x``.
    This is the estimator x^(K) of paper Lemma 3.6, written in the
    proposal's alpha convention. ``x`` may also be an (n, q) matrix, with one
    column of ``sigma`` per column of ``x``.
    """
    for _ in range(int(K)):
        x = (1.0 - alpha) * sigma + alpha * graph.apply_P(x)
    return x


def power_iteration(
    graph: Graph,
    sigma: np.ndarray,
    alpha: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 100_000,
    x0: np.ndarray | None = None,
    norm: str = "l1",
    exact: np.ndarray | None = None,
) -> PPRResult:
    """Power method with the stopping rule ``||r^(k+1) - r^(k)|| < tol``.

    ``history`` records ``diff`` (the step norm at every iteration), the
    L1 ``a_posteriori_bound``, and, if ``exact`` is given, the true L1
    ``error`` at every iteration, which is used in convergence plots.
    """
    alpha = check_alpha(alpha)
    sigma = check_distribution(sigma, graph.n)
    nrm = NORMS[norm]
    t0 = time.perf_counter()
    x = sigma.copy() if x0 is None else np.array(x0, dtype=np.float64)
    diffs, errors = [], []
    if exact is not None:
        errors.append(NORMS["l1"](x - exact))
    k = 0
    for k in range(1, max_iter + 1):
        x_new = (1.0 - alpha) * sigma + alpha * graph.apply_P(x)
        d = nrm(x_new - x)
        x = x_new
        diffs.append(d)
        if exact is not None:
            errors.append(NORMS["l1"](x - exact))
        if d < tol:
            break
    elapsed = time.perf_counter() - t0
    diffs = np.array(diffs)
    history = {"diff": diffs, "converged": bool(diffs.size and diffs[-1] < tol)}
    if norm == "l1":
        history["a_posteriori_bound"] = alpha / (1.0 - alpha) * diffs if alpha < 1 else diffs
    if exact is not None:
        history["error"] = np.array(errors)  # errors[k] = ||r^(k) - pi||_1, with k = 0 the start
    return PPRResult(
        estimate=x, method="power", alpha=alpha, time=elapsed,
        n_power_iterations=k, history=history,
    )
