"""Variance-reduced Monte Carlo on alpha-random walks: PW and PPW (paper Sections 3.2-3.3).

The formulas below use the proposal's convention, where alpha is the
*damping* factor (paper alpha = 1 - our alpha).

* PW (Algorithm 4), "first walk, then propagate": run MCW with T walks,
  then apply K power iterations to the estimate. It remains unbiased and
  the variance shrinks by at least ``alpha^(2K)`` (Lemmas 3.4, 3.7).
* PPW (Algorithm 5), progressive sampling: split the T walks into B
  batches. Before each batch, compute the residual
  ``r = sigma - (pi_hat - alpha P pi_hat) / (1 - alpha)``, which satisfies
  ``pi = pi_hat + Pi r`` (Lemma 3.12). Estimate ``Pi r`` with walks started
  from ``|r| / ||r||_1``, add it to ``pi_hat``, and propagate K times. The
  variance of each batch is ``||r||_1^2 - ||Pi r||_2^2`` (Lemma 3.14), so it
  falls as the estimate improves.
"""

from __future__ import annotations

import time

import numpy as np

from .graph import Graph
from .power import propagate
from .result import PPRResult
from .sampling import make_rng
from .sources import check_distribution
from .walks import check_alpha, walk_estimate


def residual(graph: Graph, sigma: np.ndarray, alpha: float, x: np.ndarray) -> np.ndarray:
    """Residual ``r = sigma + alpha/(1-alpha) P x - x/(1-alpha)``, so that ``pi = x + Pi r``."""
    return sigma - (x - alpha * graph.apply_P(x)) / (1.0 - alpha)


def pw(
    graph: Graph,
    sigma,
    alpha: float,
    n_walks: int,
    K: int,
    rng=None,
    *,
    estimator: str = "terminal",
) -> PPRResult:
    """PW: MCW with ``n_walks`` walks followed by ``K`` power iterations.

    Cost: O(T / (1 - alpha) + K m).
    """
    alpha = check_alpha(alpha)
    sigma = check_distribution(sigma, graph.n)
    rng = make_rng(rng)
    t0 = time.perf_counter()
    x, steps = walk_estimate(graph, sigma, alpha, n_walks, rng, estimator)
    t_walk = time.perf_counter() - t0
    x = propagate(graph, sigma, alpha, x, K)
    elapsed = time.perf_counter() - t0
    return PPRResult(
        estimate=x, method="pw", alpha=alpha, time=elapsed,
        n_walks=int(n_walks), n_walk_steps=int(steps), n_power_iterations=int(K),
        history={"walk_time": t_walk, "power_time": elapsed - t_walk},
    )


def ppw(
    graph: Graph,
    sigma,
    alpha: float,
    n_walks: int,
    K: int,
    n_batches: int = 3,
    rng=None,
    *,
    estimator: str = "terminal",
    record_residuals: bool = False,
) -> PPRResult:
    """PPW: progressive PW with ``n_batches`` batches and K power iterations per batch.

    ``history["residual_l1"][b]`` is ``||r||_1`` just before batch b. The last
    entry is the residual of the final estimate, a free a-posteriori quality
    indicator: ``||pi - pi_hat||_1 <= ||Pi||_1 ||r||_1 = ||r||_1``.

    ``record_residuals=True`` also stores each batch's residual vector in
    ``history["residuals"]``, which variance analyses use.

    Cost: O(T / (1 - alpha) + K B m + B m).
    """
    alpha = check_alpha(alpha)
    sigma = check_distribution(sigma, graph.n)
    rng = make_rng(rng)
    if n_batches < 1:
        raise ValueError("n_batches must be >= 1")
    t0 = time.perf_counter()
    sizes = [len(c) for c in np.array_split(np.arange(int(n_walks)), n_batches)]
    x = np.zeros(graph.n)
    res_norms, total_steps, residuals = [], 0, []
    for b in range(n_batches):
        r = sigma if b == 0 else residual(graph, sigma, alpha, x)
        res_norms.append(float(np.abs(r).sum()))
        if record_residuals:
            residuals.append(r.copy())
        corr, steps = walk_estimate(graph, r, alpha, sizes[b], rng, estimator)
        total_steps += steps
        x = propagate(graph, sigma, alpha, x + corr, K)
    res_norms.append(float(np.abs(residual(graph, sigma, alpha, x)).sum()))
    return PPRResult(
        estimate=x, method="ppw", alpha=alpha, time=time.perf_counter() - t0,
        n_walks=int(n_walks), n_walk_steps=int(total_steps),
        n_power_iterations=int(K) * n_batches + n_batches,  # K per batch plus one per residual
        history={"residual_l1": np.array(res_norms), "batch_sizes": sizes,
                 **({"residuals": residuals} if record_residuals else {})},
    )
