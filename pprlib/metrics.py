"""Accuracy and ranking metrics for comparing an estimate with the exact PPR vector."""

from __future__ import annotations

import numpy as np
from scipy import stats


def l1_error(est, exact) -> float:
    return float(np.abs(est - exact).sum())


def l2_error(est, exact) -> float:
    d = est - exact
    return float(np.sqrt(np.dot(d, d)))


def sq_l2_error(est, exact) -> float:
    d = est - exact
    return float(np.dot(d, d))


def linf_error(est, exact) -> float:
    return float(np.abs(est - exact).max())


def relative_error_stats(est, exact, mu: float, eps: float | None = None) -> dict:
    """Relative error over the nodes with ``exact >= mu`` (paper Definition 2.1).

    Returns the maximum and mean relative error. If ``eps`` is given, also
    returns the fraction of relevant nodes with ``|est - exact| <= eps * exact``.
    """
    mask = exact >= mu
    if not mask.any():
        return {"n_relevant": 0, "max_rel": 0.0, "mean_rel": 0.0, "frac_ok": 1.0}
    rel = np.abs(est[mask] - exact[mask]) / exact[mask]
    out = {"n_relevant": int(mask.sum()), "max_rel": float(rel.max()), "mean_rel": float(rel.mean())}
    if eps is not None:
        out["frac_ok"] = float((rel <= eps).mean())
    return out


def top_k(scores, k: int) -> np.ndarray:
    """Indices of the k largest scores. Ties are broken by index, so the result is deterministic."""
    return np.argsort(-np.asarray(scores), kind="stable")[:k]


def top_k_precision(est, exact, k: int) -> float:
    """|top-k(est) intersect top-k(exact)| / k."""
    k = min(k, len(exact))
    return len(set(top_k(est, k)) & set(top_k(exact, k))) / k


def kendall_tau(est, exact, k: int | None = None) -> float:
    """Kendall's tau-b between the two score vectors.

    If k is given, only the exact top-k nodes are compared.
    """
    if k is not None:
        idx = top_k(exact, k)
        est, exact = np.asarray(est)[idx], np.asarray(exact)[idx]
    tau = stats.kendalltau(est, exact).statistic
    return float(tau) if np.isfinite(tau) else 1.0


def spearman_rho(est, exact, k: int | None = None) -> float:
    if k is not None:
        idx = top_k(exact, k)
        est, exact = np.asarray(est)[idx], np.asarray(exact)[idx]
    rho = stats.spearmanr(est, exact).statistic
    return float(rho) if np.isfinite(rho) else 1.0


def all_errors(est, exact, *, mu: float | None = None, eps: float | None = None, k: int = 10) -> dict:
    """One-stop dictionary of every error metric (handy for DataFrame rows)."""
    out = {
        "l1": l1_error(est, exact),
        "l2": l2_error(est, exact),
        "linf": linf_error(est, exact),
        f"prec@{k}": top_k_precision(est, exact, k),
        f"tau@{k}": kendall_tau(est, exact, k),
    }
    if mu is not None:
        rel = relative_error_stats(est, exact, mu, eps)
        out["max_rel"] = rel["max_rel"]
        if eps is not None:
            out["frac_ok"] = rel["frac_ok"]
    return out
