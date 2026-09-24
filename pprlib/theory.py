"""Theoretical quantities from the paper, in the proposal's alpha convention.

Convention mapping: ``paper_alpha = 1 - alpha``. The paper's stop
probability is our restart probability ``1 - alpha``, and the paper's
``(1 - alpha_p)^K`` becomes our ``alpha^K``.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .graph import Graph
from .sampling import make_rng


def to_paper_alpha(alpha: float) -> float:
    """Proposal damping factor -> the paper's stop probability."""
    return 1.0 - alpha


def from_paper_alpha(alpha_paper: float) -> float:
    return 1.0 - alpha_paper


# ---------------------------------------------------------------- variances
def walk_variance(pi: np.ndarray) -> float:
    """Single-walk variance of the MCW estimator (Lemma 3.2): ``1 - ||pi||_2^2``."""
    return float(1.0 - np.dot(pi, pi))


def pw_variance_bound(pi: np.ndarray, alpha: float, K: int) -> float:
    """Upper bound on the single-walk variance of x^(K) (Lemma 3.7): ``alpha^(2K) (1 - ||pi||^2)``."""
    return float(alpha ** (2 * K) * walk_variance(pi))


def exact_pw_variance(graph: Graph, pi: np.ndarray, alpha: float, K: int) -> float:
    """Exact single-walk variance of x^(K) = const + alpha^K P^K e_t, where t ~ pi.

    ``Var = alpha^(2K) (sum_t pi(t) ||P^K e_t||^2 - ||P^K pi||^2)``.
    """
    return pw_variance_curve(graph, pi, alpha, [K])[0]


def pw_variance_curve(graph: Graph, pi: np.ndarray, alpha: float, Ks) -> np.ndarray:
    """Exact single-walk PW variance for every K in ``Ks``, computed in one sweep.

    It uses the centred form ``alpha^(2K) sum_t pi(t) ||P^K (e_t - pi)||_2^2``,
    which is algebraically equal to the formula above. The uncentred version
    subtracts two O(1) numbers and loses all precision once the variance falls
    below about 1e-16.

    P^k is applied to the columns e_t - pi with pi(t) > 0, i.e. to a dense
    block of size n x |supp(pi)|. Only use this on small or medium graphs.
    """
    Ks = sorted(int(k) for k in Ks)
    supp = np.flatnonzero(pi > 0)
    E = np.repeat(-pi[:, None], supp.size, axis=1)
    E[supp, np.arange(supp.size)] += 1.0
    out, k = {}, 0
    for K in Ks:
        while k < K:
            E = graph.apply_P(E)
            k += 1
        col_sq = (E * E).sum(axis=0)
        out[K] = float(alpha ** (2 * K) * np.dot(pi[supp], col_sq))
    return np.array([out[K] for K in Ks])


def residual_walk_variance(Pi: np.ndarray, r: np.ndarray) -> float:
    """Single-walk variance of the residual estimator (Lemma 3.14): ``||r||_1^2 - ||Pi r||_2^2``."""
    Pr = Pi @ r
    return float(np.abs(r).sum() ** 2 - np.dot(Pr, Pr))


# ------------------------------------------------------ sample-size formulas
def theoretical_K(alpha: float, eps: float) -> int:
    """K = log_alpha(eps^2), the paper's default K = log_{1-alpha_p} eps^2 (Section 5.1)."""
    return max(1, math.ceil(math.log(eps * eps) / math.log(alpha)))


def optimal_K(alpha: float, eps: float) -> int:
    """Minimiser of the cost T/(1-alpha) + K m under Corollary 1.

    K = log_alpha((1 - alpha) eps^2 / ln(1/alpha)).
    """
    return max(1, math.ceil(math.log((1 - alpha) * eps * eps / math.log(1 / alpha)) / math.log(alpha)))


def chernoff_W(eps: float, mu: float, p_fail: float) -> float:
    """W = (2 eps/3 + 2) log(2/p_f) / (eps^2 mu) (Lemma 3.11)."""
    return (2 * eps / 3 + 2) * math.log(2 / p_fail) / (eps * eps * mu)


def pw_walks_for_guarantee(n: int, alpha: float, K: int, eps: float, mu=None, p_fail=None) -> int:
    """Walks T > alpha^K W that give an (eps, mu, p_f) relative-error guarantee (Lemma 3.11).

    ``mu`` and ``p_fail`` default to 1/n.
    """
    mu = 1.0 / n if mu is None else mu
    p_fail = 1.0 / n if p_fail is None else p_fail
    return math.ceil(alpha ** K * chernoff_W(eps, mu, p_fail))


def mcw_walks_for_guarantee(n: int, eps: float, mu=None, p_fail=None) -> int:
    """Plain MC, the K = 0 case: T > W."""
    mu = 1.0 / n if mu is None else mu
    p_fail = 1.0 / n if p_fail is None else p_fail
    return math.ceil(chernoff_W(eps, mu, p_fail))


def power_iterations_for_tol(alpha: float, l1_tol: float) -> int:
    """Iterations k with 2 alpha^k <= l1_tol (the a-priori L1 bound, starting from r^(0) = s)."""
    if alpha == 0:
        return 1
    return max(1, math.ceil(math.log(l1_tol / 2) / math.log(alpha)))


# ----------------------------------------------------------- forest analysis
def forest_statistics(graph: Graph, alpha: float, n_forests: int = 50, rng=None) -> dict:
    """Empirical quantities behind Lemma 4.4 and Table 2.

    * ``n_r = (1/n) 1^T Q 1``: expected number of nodes sharing a node's root.
    * ``n_rd = (1/n) 1^T Q_d 1``: the degree-weighted analogue (undirected graphs).
    * ``tau_forest``: mean seconds per forest; ``forest_steps``: mean walk steps per forest.
    """
    from .forests import sample_forest

    rng = make_rng(rng)
    n = graph.n
    d = graph.forest_degree
    nr, nrd, t, steps = 0.0, 0.0, 0.0, 0
    sample_forest(graph, alpha, rng)  # untimed warm-up (JIT compilation)
    for _ in range(n_forests):
        t0 = time.perf_counter()
        root, s = sample_forest(graph, alpha, rng)
        t += time.perf_counter() - t0
        steps += s
        size = np.bincount(root, minlength=n)
        nr += size[root].sum() / n
        dsum = np.bincount(root, weights=d, minlength=n)
        d2 = np.bincount(root, weights=d * d, minlength=n)
        # 1^T S~^T S~ 1 = sum over trees C of |C|^2 * sum_{u in C} d(u)^2 / D(C)^2
        roots = np.flatnonzero(size)
        nrd += (size[roots] ** 2 * d2[roots] / dsum[roots] ** 2).sum() / n
    return {
        "n_r": nr / n_forests,
        "n_rd": nrd / n_forests,
        "tau_forest": t / n_forests,
        "forest_steps": steps / n_forests,
    }


def walk_time(graph: Graph, alpha: float, n_walks: int = 200_000, rng=None) -> dict:
    """Measure tau_walk, the mean seconds per alpha-random walk from a uniform start (vectorised)."""
    from .walks import walk_terminals

    rng = make_rng(rng)
    walk_terminals(graph, rng.integers(0, graph.n, size=100), alpha, rng)  # untimed warm-up
    starts = rng.integers(0, graph.n, size=n_walks)
    t0 = time.perf_counter()
    _, steps = walk_terminals(graph, starts, alpha, rng)
    t = time.perf_counter() - t0
    return {"tau_walk": t / n_walks, "walk_steps": steps / n_walks}
