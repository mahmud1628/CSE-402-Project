"""Exact football PPR: pi = alpha P pi + (1-alpha) sigma.

Alpha is always the continuation/damping probability. Exact LU results are
ground truth on these small networks. Monte Carlo runs here are smoke tests,
not a variance or runtime comparison study.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import pprlib as pl
from .networks import PassingNetwork

ALPHAS = (0.5, 0.8, 0.85, 0.9, 0.99)


def checked_ppr(graph: pl.Graph, source=None, alpha: float = 0.85) -> tuple[pl.PPRResult, dict]:
    """Verify exact against power to an absolute L1 tolerance of 1e-10."""
    exact = pl.compute_ppr(graph, source, alpha, method="exact")
    power = pl.compute_ppr(graph, source, alpha, method="power", tol=1e-13, exact=exact.estimate)
    error = pl.metrics.l1_error(power.estimate, exact.estimate)
    if not power.history["converged"] or error > 1e-10:
        raise ArithmeticError(f"power disagrees with exact: L1={error:.3g}")
    if not np.isclose(exact.estimate.sum(), 1.0, atol=1e-12) or exact.estimate.min() < -1e-12:
        raise ArithmeticError("exact PPR is not a probability vector")
    return exact, dict(alpha=alpha, power_l1_error=error, power_iterations=power.n_power_iterations,
                       exact_mass=float(exact.estimate.sum()))


def group_sources(network: PassingNetwork) -> dict[str, np.ndarray]:
    """Uniform mass on goalkeepers/defenders by their first recorded match role.

    Aggregated networks include a player if any first-recorded match role
    matches. These describe source groups, not claims about playing quality.
    """
    groups = {}
    roles = network.players.position.fillna("")
    for name, mask in [("goalkeepers", roles.str.contains("Goalkeeper", regex=False)),
                       ("defenders", roles.str.contains("Back", regex=False))]:
        labels = network.players.loc[mask, "label"]
        if len(labels):
            groups[name] = pl.sources.from_weights(network.graph, {label: 1.0 for label in labels})
    return groups


def analyze(network: PassingNetwork, alphas=ALPHAS) -> dict:
    """Centrality, positional PPR, all-source matrices and alpha sensitivity.

    Matrix rows are destinations and columns are source players. All saved
    source and estimate arrays use ``network.players`` / graph node order.
    """
    g = network.graph
    sources = {"uniform": pl.uniform(g), **group_sources(network)}
    ranking_rows, check_rows, sensitivity = [], [], []
    matrices, vectors = {}, {}
    baseline = pl.compute_ppr(g, None, 0.85, method="exact").estimate
    for alpha in alphas:
        matrix = pl.ppr_matrix(g, alpha)
        np.testing.assert_allclose(matrix.sum(axis=0), 1.0, atol=1e-12)
        matrices[alpha] = matrix
        for source_name, sigma in sources.items():
            result, check = checked_ppr(g, sigma, alpha)
            np.testing.assert_allclose(matrix @ sigma, result.estimate, atol=1e-12)
            vectors[alpha, source_name] = result.estimate
            check_rows.append(dict(source=source_name, **check))
            for rank, (label, score) in enumerate(result.top_k(g.n, g), 1):
                player = network.players.iloc[g.index(label)]
                ranking_rows.append(dict(alpha=alpha, source=source_name, rank=rank,
                                         player_id=int(player.player_id), player=label, score=score))
        # Check single-source power solutions too, not just their uniform average.
        for i, label in enumerate(g.labels):
            result = pl.compute_ppr(g, label, alpha, "power", tol=1e-13)
            error = pl.metrics.l1_error(result.estimate, matrix[:, i])
            if not result.history["converged"] or error > 1e-10:
                raise ArithmeticError(f"single-source power check failed for {label}")
            check_rows.append(dict(source=f"player:{int(network.players.iloc[i].player_id)}", alpha=alpha,
                                   power_l1_error=error, power_iterations=result.n_power_iterations,
                                   exact_mass=float(matrix[:, i].sum())))
        sensitivity.append(dict(alpha=alpha, reference_alpha=0.85,
                                kendall_tau=pl.metrics.kendall_tau(vectors[alpha, "uniform"], baseline)))
    return dict(rankings=pd.DataFrame(ranking_rows), checks=pd.DataFrame(check_rows),
                sensitivity=pd.DataFrame(sensitivity), matrices=matrices, vectors=vectors, sources=sources)


def smoke_test(network: PassingNetwork, *, seed: int = 0, quick: bool = False) -> pd.DataFrame:
    """One fixed-seed run per directed MC method at alpha 0.8, 0.85 and 0.99.

    Budgets are explicit: 30,000/90,000 walks and 300/900 forests (quick/full),
    three progressive batches. A conservative K makes propagation stable at
    high damping. These deliberately unequal budgets are not a performance
    benchmark. Undirected degree-weighted variants are outside this workflow.
    """
    g = network.graph
    rows = []
    for alpha in (0.8, 0.85, 0.99):
        exact = pl.compute_ppr(g, None, alpha, "exact").estimate
        K = math.ceil(math.log(1e-4) / math.log(alpha))
        for method in ("mcw", "pw", "ppw", "mcf", "pf", "ppf"):
            params = dict(K=K, n_batches=3)
            if method in {"mcw", "pw", "ppw"}:
                params["n_walks"] = 30_000 if quick else 90_000
            else:
                params["n_forests"] = 300 if quick else 900
            result = pl.compute_ppr(g, None, alpha, method, rng=seed, **params)
            if not np.all(np.isfinite(result.estimate)):
                raise ArithmeticError(f"nonfinite {method} estimate")
            rows.append(dict(method=method, alpha=alpha, seed=seed, K=K if method not in {"mcw", "mcf"} else 0,
                             n_batches=3 if method in {"ppw", "ppf"} else 1,
                             n_walks=result.n_walks, n_forests=result.n_forests,
                             n_power_iterations=result.n_power_iterations, seconds=result.time,
                             l1_error=pl.metrics.l1_error(result.estimate, exact),
                             estimate_mass=float(result.estimate.sum()), min_estimate=float(result.estimate.min())))
    return pd.DataFrame(rows)
