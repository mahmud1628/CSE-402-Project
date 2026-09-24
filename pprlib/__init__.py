"""pprlib: Personalized PageRank by power iteration, Monte Carlo, and variance-reduced Monte Carlo.

Written from scratch for the CSE 402 project, following Liao et al.,
"Efficient Personalized PageRank Computation: The Power of Variance-Reduced
Monte Carlo Approaches" (SIGMOD 2023).

alpha convention (the one used in our proposal)
-----------------------------------------------
``alpha`` is the **damping factor**, the probability that the random surfer
*continues*::

    pi = alpha * P pi + (1 - alpha) * sigma

The paper uses the opposite convention, ``paper_alpha = 1 - alpha``. So the
paper's alpha = 0.2 is our alpha = 0.8, and its alpha = 0.01 is our 0.99.
Every function in this package takes the proposal's alpha; use
``theory.to_paper_alpha`` to convert.

Modules
-------
graph           Graph (CSR, weighted/unweighted, directed/undirected, dangling rules)
sources         source distributions sigma (one-hot, uniform, weights, random)
sampling        RNG helper, Walker alias tables
walks           vectorised alpha-random walks, unbiased walk estimator of Pi x
power           power iteration (with convergence history), K-step propagation
exact           ground truth by direct linear solve
monte_carlo     MCW (standard Monte Carlo)
variance_reduced PW, PPW, residual
forests         spanning-forest estimators: MCF(V), PF(V), PPF(V)
theory          the paper's variance formulas and sample-size bounds
metrics         L1/L2/Linf/relative errors, top-k precision, Kendall tau
evaluation      repeated trials, MSE and variance, ranking stability
datasets        SNAP loader, edge-list reader, graph generators, paper toy graph
api             compute_ppr(graph, source, alpha, method, ...)
"""

from . import (
    datasets, evaluation, exact, forests, metrics, monte_carlo, power,
    sampling, sources, theory, variance_reduced, walks,
)
from .api import METHODS, compute_ppr, default_params
from .exact import exact_ppr, ppr_matrix
from .forests import mcf, pf, ppf, sample_forest
from .graph import Graph
from .monte_carlo import mcw
from .power import power_iteration, propagate
from .result import PPRResult
from .sources import as_distribution, one_hot, uniform
from .variance_reduced import ppw, pw, residual

__all__ = [
    "Graph", "PPRResult", "compute_ppr", "default_params", "METHODS",
    "exact_ppr", "ppr_matrix", "power_iteration", "propagate",
    "mcw", "pw", "ppw", "residual", "mcf", "pf", "ppf", "sample_forest",
    "one_hot", "uniform", "as_distribution",
    "datasets", "evaluation", "exact", "forests", "metrics", "monte_carlo", "power",
    "sampling", "sources", "theory", "variance_reduced", "walks",
]
