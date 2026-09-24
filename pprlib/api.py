"""One entry point for every PPR method: ``compute_ppr(graph, source, alpha, method, ...)``.

Stage 2 code can switch algorithms by changing a string, for example::

    res = compute_ppr(g, "Messi", alpha=0.85, method="ppw", n_walks=10_000, K=5)
    res.top_k(5, g)
"""

from __future__ import annotations

import math
import time

import numpy as np

from . import forests, monte_carlo, variance_reduced
from .exact import exact_ppr
from .power import power_iteration
from .result import PPRResult
from .sources import as_distribution
from .theory import theoretical_K

METHODS = ("exact", "power", "mcw", "pw", "ppw", "mcf", "mcfv", "pf", "pfv", "ppf", "ppfv")


def default_params(graph, alpha: float, eps: float = 0.5, n_batches: int = 3) -> dict:
    """The paper's default parameter choices (Section 5.1), in the proposal's alpha convention.

    T = n ln n walks, K = log_alpha(eps^2) (divided by B for the progressive
    methods), and B = 3. The number of forests is chosen to cost roughly as
    much as T walks: one forest touches every node, so T' = T / n = ln n.
    """
    n = graph.n
    T = max(1, math.ceil(n * math.log(max(n, 2))))
    return {
        "n_walks": T,
        "K": theoretical_K(alpha, eps),
        "K_progressive": theoretical_K(alpha, eps / math.sqrt(n_batches)),
        "n_batches": n_batches,
        "n_forests": max(1, math.ceil(math.log(max(n, 2)))),
    }


def compute_ppr(graph, source=None, alpha: float = 0.85, method: str = "power", *, rng=None, **kw) -> PPRResult:
    """Compute PPR with the named method.

    Parameters
    ----------
    source : node label/index, dict of weights, length-n array, or None (uniform)
    alpha : damping factor, the probability that the surfer continues (proposal convention)
    method : one of ``METHODS``
    kw : method parameters: ``n_walks``, ``K``, ``n_batches``, ``n_forests``,
        ``estimator`` ("terminal"/"visits"), ``tol`` (power). Anything left
        unset is filled in from ``default_params``.
    """
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    sigma = as_distribution(graph, source)
    d = default_params(graph, alpha, kw.pop("eps", 0.5), kw.get("n_batches", 3))
    progressive = method in ("ppw", "ppf", "ppfv")
    T = kw.pop("n_walks", d["n_walks"])
    K = kw.pop("K", d["K_progressive"] if progressive else d["K"])
    B = kw.pop("n_batches", d["n_batches"])
    F = kw.pop("n_forests", d["n_forests"])
    estimator = kw.pop("estimator", "terminal")

    if method == "exact":
        t0 = time.perf_counter()
        est = exact_ppr(graph, sigma, alpha, kw.pop("solver", "auto"))
        res = PPRResult(estimate=np.asarray(est), method="exact", alpha=alpha, time=time.perf_counter() - t0)
    elif method == "power":
        res = power_iteration(graph, sigma, alpha, **kw)
        kw = {}
    elif method == "mcw":
        res = monte_carlo.mcw(graph, sigma, alpha, T, rng, estimator=estimator)
    elif method == "pw":
        res = variance_reduced.pw(graph, sigma, alpha, T, K, rng, estimator=estimator)
    elif method == "ppw":
        res = variance_reduced.ppw(graph, sigma, alpha, T, K, B, rng, estimator=estimator)
    else:
        variant = "degree" if method.endswith("v") else "root"
        base = method.rstrip("v")
        if base == "mcf":
            res = forests.mcf(graph, sigma, alpha, F, rng, variant=variant)
        elif base == "pf":
            res = forests.pf(graph, sigma, alpha, F, K, rng, variant=variant)
        else:
            res = forests.ppf(graph, sigma, alpha, F, K, B, rng, variant=variant)
    if kw:
        raise TypeError(f"unused parameters for {method}: {sorted(kw)}")
    return res
