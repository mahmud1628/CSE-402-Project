"""Repeated-trial evaluation helpers for numerical comparisons (used in stages 1 and 3).

Monte Carlo estimators are random, so a single run says little. These
helpers run a method many times with independent seeds and measure its
error distribution, its variance, and how stable the rankings are.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import metrics
from .result import PPRResult


def run_trials(fn: Callable[[np.random.Generator], PPRResult], n_trials: int, seed: int = 0) -> list[PPRResult]:
    """Call ``fn(rng)`` ``n_trials`` times, each with an independent child RNG."""
    children = np.random.SeedSequence(seed).spawn(n_trials)
    return [fn(np.random.default_rng(c)) for c in children]


def stack(results: list[PPRResult]) -> np.ndarray:
    """Estimates as an array of shape (n_trials, n)."""
    return np.stack([r.estimate for r in results])


def mean_squared_error(results, exact) -> float:
    """Mean over trials of ``||pi_hat - pi||_2^2``.

    For an unbiased estimator this equals its total variance. For an average
    of T i.i.d. samples it is the single-sample variance divided by T.
    """
    E = stack(results) - exact
    return float((E * E).sum(axis=1).mean())


def total_variance(results) -> float:
    """``sum_u Var[pi_hat(u)]`` across trials (unbiased sample variance, ddof=1)."""
    X = stack(results)
    return float(X.var(axis=0, ddof=1).sum())


def bias_l1(results, exact) -> float:
    """``||mean(pi_hat) - pi||_1``. It should shrink like 1/sqrt(n_trials) for an unbiased method."""
    return float(np.abs(stack(results).mean(axis=0) - exact).sum())


def summarize(results: list[PPRResult], exact: np.ndarray, *, k: int = 10, mu=None, eps=None) -> dict:
    """Mean and standard deviation of every error metric and of the cost, plus the MSE."""
    rows = [metrics.all_errors(r.estimate, exact, mu=mu, eps=eps, k=k) for r in results]
    out = {}
    for key in rows[0]:
        vals = np.array([row[key] for row in rows])
        out[key] = float(vals.mean())
        out[key + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    out["mse"] = mean_squared_error(results, exact)
    out["time"] = float(np.mean([r.time for r in results]))
    out["n_walks"] = results[0].n_walks
    out["n_forests"] = results[0].n_forests
    out["n_power_iterations"] = results[0].n_power_iterations
    out["n_walk_steps"] = float(np.mean([r.n_walk_steps for r in results]))
    out["n_trials"] = len(results)
    return out


def ranking_stability(results: list[PPRResult], k: int = 10) -> dict:
    """How consistent rankings are *between* independent runs, with no ground truth needed.

    Returns the mean pairwise top-k overlap and the mean pairwise Kendall tau
    on the union of the top-k sets.
    """
    X = stack(results)
    overlaps, taus = [], []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            a, b = metrics.top_k(X[i], k), metrics.top_k(X[j], k)
            overlaps.append(len(set(a) & set(b)) / k)
            idx = np.union1d(a, b)
            taus.append(metrics.kendall_tau(X[i][idx], X[j][idx]))
    return {"pairwise_topk_overlap": float(np.mean(overlaps)), "pairwise_tau": float(np.mean(taus))}
