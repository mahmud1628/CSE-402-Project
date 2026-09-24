"""Standard Monte Carlo PPR estimation: MCW (paper Algorithm 3)."""

from __future__ import annotations

import time

from .graph import Graph
from .result import PPRResult
from .sampling import make_rng
from .sources import check_distribution
from .walks import check_alpha, walk_estimate


def mcw(
    graph: Graph,
    sigma,
    alpha: float,
    n_walks: int,
    rng=None,
    *,
    estimator: str = "terminal",
) -> PPRResult:
    """Monte Carlo with alpha-random walks.

    Each of ``n_walks`` walks starts at s ~ sigma, and the estimate is the
    empirical distribution of the stopping nodes (Lemma 3.1). The
    single-walk variance is ``1 - ||pi||_2^2`` (Lemma 3.2), so the averaged
    estimate has ``E||pi_hat - pi||_2^2 = (1 - ||pi||_2^2) / n_walks``.

    ``estimator="visits"`` uses the every-visit variant instead: each visit
    to v contributes ``(1 - alpha) / n_walks``.
    """
    alpha = check_alpha(alpha)
    sigma = check_distribution(sigma, graph.n)
    rng = make_rng(rng)
    t0 = time.perf_counter()
    est, steps = walk_estimate(graph, sigma, alpha, n_walks, rng, estimator)
    return PPRResult(
        estimate=est, method="mcw", alpha=alpha, time=time.perf_counter() - t0,
        n_walks=int(n_walks), n_walk_steps=int(steps),
    )
