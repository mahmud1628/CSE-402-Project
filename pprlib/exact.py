"""Ground-truth PPR by solving the linear system directly.

    (I - alpha P) pi = (1 - alpha) sigma

* ``dense``: LU factorisation of the dense matrix (small graphs).
* ``direct``: sparse LU (SuperLU), factorised once and reused for many right-hand sides.
* ``power``: power iteration run to a tiny a-posteriori error bound (large graphs).

With ``dangling="uniform"``, P = P0 + (1/n) 1 d^T is a rank-1 update of a
sparse matrix, where d is the dangling-node indicator. The ``direct``
solver handles this with the Sherman-Morrison formula.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .graph import Graph
from .walks import check_alpha


def _as_matrix(sigma, n):
    s = np.asarray(sigma, dtype=np.float64)
    if s.shape[0] != n:
        raise ValueError("sigma must have n rows")
    return s


def exact_ppr(graph: Graph, sigma, alpha: float, method: str = "auto", tol: float = 1e-15) -> np.ndarray:
    """Exact PPR for sigma of shape (n,) or (n, q), where each column is a source distribution."""
    alpha = check_alpha(alpha)
    n = graph.n
    s = _as_matrix(sigma, n)
    b = (1.0 - alpha) * s
    if method == "auto":
        method = "dense" if n <= 2000 else ("direct" if n <= 30000 else "power")

    if method == "dense":
        A = np.eye(n) - alpha * graph.dense_P()
        return np.linalg.solve(A, b)

    if method == "direct":
        A0 = (sp.identity(n, format="csc") - alpha * graph.P).tocsc()
        lu = spla.splu(A0)
        y = lu.solve(b)
        if graph.dangling == "uniform" and graph.dangling_nodes.size:
            # A = A0 - alpha u v^T  with u = 1/n, v = dangling indicator
            u = np.full(n, 1.0 / n)
            z = lu.solve(u)
            vz = z[graph.dangling_nodes].sum()
            vy = y[graph.dangling_nodes].sum(axis=0)
            y = y + alpha * np.multiply.outer(z, vy).reshape(y.shape) / (1.0 - alpha * vz)
        return y

    if method == "power":
        # The L1 error is at most alpha/(1-alpha) * step, so make the step small enough.
        step_tol = tol * (1.0 - alpha) / max(alpha, 1e-300)
        x = s.copy()
        for _ in range(1_000_000):
            x_new = b + alpha * graph.apply_P(x)
            d = np.abs(x_new - x).sum(axis=0)
            x = x_new
            if np.all(d < step_tol):
                break
        return x

    raise ValueError(f"unknown method {method!r}")


def ppr_matrix(graph: Graph, alpha: float) -> np.ndarray:
    """Dense PPR matrix ``Pi = (1 - alpha)(I - alpha P)^{-1}``, whose column u is pi_{e_u}.

    It costs O(n^3), so only use it on small graphs, e.g. when evaluating
    the theoretical variance formulas.
    """
    return exact_ppr(graph, np.eye(graph.n), alpha, method="dense" if graph.n <= 5000 else "direct")
