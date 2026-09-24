"""Graph representation shared by every PPR algorithm in the package.

Conventions
-----------
* Nodes are stored internally as integers ``0..n-1``. Optional ``labels``
  map them to arbitrary hashable names, such as player names in stage 2.
* An edge ``u -> v`` has weight ``w(u, v) > 0``. A surfer at ``u`` moves to
  ``v`` with probability ``w(u, v) / s_out(u)``, where ``s_out(u)`` is the
  sum of ``w(u, v)`` over all ``v``. An unweighted graph has every weight
  equal to 1, which gives the paper's uniform choice among out-neighbours.
* ``P`` is the **column-stochastic** transition matrix,
  ``P[v, u] = w(u, v) / s_out(u)``. Hence
  ``(P x)(v) = sum over u -> v of x(u) * w(u, v) / s_out(u)``. This matches
  the paper's ``P = A D_out^{-1}``.
* Dangling nodes (``s_out = 0``) are handled in one of two ways:

  ``"self_loop"``
      The surfer stays where it is. This is the convention of the paper's
      released code.
  ``"uniform"``
      The surfer jumps to a uniformly random node. This is the classic
      PageRank fix.

  Every component (walks, power iteration, exact solver, forests) uses the
  same rule, so all methods estimate the same vector.
"""

from __future__ import annotations

from typing import Hashable, Iterable, Sequence

import numpy as np
import scipy.sparse as sp

DANGLING_MODES = ("self_loop", "uniform")


class Graph:
    """Directed, optionally weighted graph stored in CSR form.

    Parameters
    ----------
    n : int
        Number of nodes.
    src, dst : array-like of int
        Edge endpoints (``src[i] -> dst[i]``) given as node indices in ``[0, n)``.
    weights : array-like of float, optional
        Positive edge weights. If omitted, the graph is unweighted and
        duplicate edges are collapsed. If given, duplicate edges have their
        weights summed, which is what we want for repeated passes.
    directed : bool
        If False, each edge is treated as undirected and its reverse is
        added automatically. Self-loops are not doubled.
    labels : sequence, optional
        Names of the nodes, where ``labels[i]`` names node ``i``.
    dangling : {"self_loop", "uniform"}
        How to treat nodes with no out-edges (see the module docstring).
    """

    def __init__(
        self,
        n: int,
        src,
        dst,
        weights=None,
        *,
        directed: bool = True,
        labels: Sequence[Hashable] | None = None,
        dangling: str = "self_loop",
    ):
        if n <= 0:
            raise ValueError("a graph needs at least one node")
        if dangling not in DANGLING_MODES:
            raise ValueError(f"dangling must be one of {DANGLING_MODES}")
        src = np.asarray(src, dtype=np.int64).ravel()
        dst = np.asarray(dst, dtype=np.int64).ravel()
        if src.shape != dst.shape:
            raise ValueError("src and dst must have the same length")
        if src.size and (min(src.min(), dst.min()) < 0 or max(src.max(), dst.max()) >= n):
            raise ValueError("edge endpoint out of range [0, n)")

        self.weighted = weights is not None
        if self.weighted:
            w = np.asarray(weights, dtype=np.float64).ravel()
            if w.shape != src.shape:
                raise ValueError("weights must have one entry per edge")
            if not np.all(np.isfinite(w)) or np.any(w < 0):
                raise ValueError("edge weights must be finite and non-negative")
            keep = w > 0
            src, dst, w = src[keep], dst[keep], w[keep]
        else:
            w = np.ones(src.size, dtype=np.float64)

        if not directed:
            nl = src != dst
            src, dst, w = (
                np.concatenate([src, dst[nl]]),
                np.concatenate([dst, src[nl]]),
                np.concatenate([w, w[nl]]),
            )

        A = sp.csr_matrix((w, (src, dst)), shape=(n, n))  # sums duplicates
        A.sum_duplicates()
        A.sort_indices()
        if not self.weighted:
            A.data[:] = 1.0

        self.n = int(n)
        self.directed = bool(directed)
        self.dangling = dangling
        self.A = A  # A[u, v] = w(u -> v)
        self.indptr = A.indptr.astype(np.int64)
        self.indices = A.indices.astype(np.int64)
        self.edge_weights = A.data.astype(np.float64)
        self.m = int(A.nnz)
        self.out_degree = np.diff(self.indptr)
        self.out_strength = np.asarray(A.sum(axis=1)).ravel()
        self.dangling_nodes = np.flatnonzero(self.out_degree == 0)

        if labels is not None:
            labels = list(labels)
            if len(labels) != n:
                raise ValueError("need exactly one label per node")
            self._index = {lab: i for i, lab in enumerate(labels)}
            if len(self._index) != n:
                raise ValueError("labels must be unique")
        else:
            self._index = None
        self.labels = labels

        self._build_transition()
        self._is_undirected = None if directed else True

    # ------------------------------------------------------------------ build
    def _build_transition(self):
        n = self.n
        rows = np.repeat(np.arange(n), self.out_degree)
        s = self.out_strength
        prob = self.edge_weights / s[rows] if self.m else self.edge_weights
        self.edge_prob = prob  # row-stochastic transition probabilities per edge

        # Column-stochastic P, stored as CSR so that P @ x is fast.
        P = sp.csr_matrix((prob, (self.indices, rows)), shape=(n, n))
        if self.dangling == "self_loop" and self.dangling_nodes.size:
            d = self.dangling_nodes
            P = P + sp.csr_matrix((np.ones(d.size), (d, d)), shape=(n, n))
        self.P = P.tocsr()

        # Cumulative probabilities within each row, for weighted sampling of
        # the next node. The key is (row index + cumulative probability), so a
        # single global searchsorted finds the sampled edge. The last entry of
        # every row is forced to exactly 1.
        if self.m:
            cum = np.cumsum(prob)
            start = self.indptr[:-1]
            before = np.where(start > 0, cum[np.maximum(start - 1, 0)], 0.0)
            local = cum - np.repeat(before, self.out_degree)
            ends = self.indptr[1:][self.out_degree > 0] - 1
            local[ends] = 1.0
        else:
            local = np.zeros(0)
        self.local_cum = local
        self._cum_key = rows + local

    # ----------------------------------------------------------- constructors
    @classmethod
    def from_edges(
        cls,
        edges: Iterable,
        *,
        directed: bool = True,
        weighted: bool | None = None,
        nodes: Sequence[Hashable] | None = None,
        dangling: str = "self_loop",
    ) -> "Graph":
        """Build a graph from ``(u, v)`` or ``(u, v, w)`` tuples of *labels*.

        ``nodes`` fixes the node set and its order, which is useful for
        including isolated nodes. Otherwise nodes are numbered in order of
        first appearance.
        """
        edges = list(edges)
        if weighted is None:
            weighted = bool(edges) and len(edges[0]) == 3
        index: dict = {}
        labels: list = []
        if nodes is not None:
            for lab in nodes:
                if lab in index:
                    raise ValueError(f"duplicate node {lab!r}")
                index[lab] = len(labels)
                labels.append(lab)

        def idx(lab):
            if lab not in index:
                if nodes is not None:
                    raise KeyError(f"edge endpoint {lab!r} not in `nodes`")
                index[lab] = len(labels)
                labels.append(lab)
            return index[lab]

        src, dst, w = [], [], []
        for e in edges:
            src.append(idx(e[0]))
            dst.append(idx(e[1]))
            if weighted:
                w.append(float(e[2]))
        return cls(
            len(labels), src, dst, w if weighted else None,
            directed=directed, labels=labels, dangling=dangling,
        )

    @classmethod
    def from_adjacency(
        cls,
        W,
        *,
        labels: Sequence[Hashable] | None = None,
        directed: bool = True,
        weighted: bool = True,
        dangling: str = "self_loop",
    ) -> "Graph":
        """Build from a (dense or sparse) matrix with ``W[u, v]`` = weight of ``u -> v``.

        With ``directed=False``, ``W`` must already be symmetric. It is used
        as is and edges are not added twice.
        """
        W = sp.coo_matrix(W)
        if W.shape[0] != W.shape[1]:
            raise ValueError("adjacency matrix must be square")
        g = cls(
            W.shape[0], W.row, W.col, W.data if weighted else None,
            directed=True, labels=labels, dangling=dangling,
        )
        if not directed:
            if not g.is_undirected():
                raise ValueError("directed=False requires a symmetric matrix")
            g.directed = False
        return g

    # -------------------------------------------------------------- utilities
    def index(self, node) -> int:
        """Internal index of a node given by label, or by index when the graph has no labels."""
        if self._index is not None:
            try:
                return self._index[node]
            except KeyError:
                raise KeyError(f"unknown node label {node!r}") from None
        i = int(node)
        if not 0 <= i < self.n:
            raise IndexError(f"node index {i} out of range")
        return i

    def label(self, i: int):
        return self.labels[i] if self.labels is not None else int(i)

    def is_undirected(self) -> bool:
        """True if the weighted adjacency matrix is symmetric."""
        if self._is_undirected is None:
            diff = self.A - self.A.T
            self._is_undirected = bool(diff.nnz == 0 or np.allclose(diff.data, 0.0))
        return self._is_undirected

    @property
    def forest_degree(self) -> np.ndarray:
        """Degree/strength d(u) used by the degree-weighted forest estimator.

        Isolated nodes get 1, which makes the estimator exact for them.
        """
        d = self.out_strength.copy()
        d[d == 0] = 1.0
        return d

    # ---------------------------------------------------------- linear algebra
    def apply_P(self, x: np.ndarray) -> np.ndarray:
        """Return ``P @ x`` for a vector or an (n, k) matrix, including dangling handling."""
        y = self.P @ x
        if self.dangling == "uniform" and self.dangling_nodes.size:
            y = y + x[self.dangling_nodes].sum(axis=0) / self.n
        return y

    def dense_P(self) -> np.ndarray:
        """Dense column-stochastic P. Only use this on small graphs."""
        P = self.P.toarray()
        if self.dangling == "uniform" and self.dangling_nodes.size:
            P[:, self.dangling_nodes] += 1.0 / self.n
        return P

    # ------------------------------------------------------------- random walk
    def step(self, nodes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Move each surfer in ``nodes`` one step (vectorised)."""
        nodes = np.asarray(nodes, dtype=np.int64)
        deg = self.out_degree[nodes]
        out = np.empty_like(nodes)
        has = deg > 0
        if not has.all():
            dn = ~has
            if self.dangling == "self_loop":
                out[dn] = nodes[dn]
            else:
                out[dn] = rng.integers(0, self.n, size=int(dn.sum()))
        u = nodes[has]
        U = rng.random(u.size)
        lo = self.indptr[u]
        hi = self.indptr[u + 1] - 1
        if self.weighted:
            pos = np.searchsorted(self._cum_key, u + U, side="right")
            pos = np.clip(pos, lo, hi)
        else:
            pos = np.minimum(lo + (U * deg[has]).astype(np.int64), hi)
        out[has] = self.indices[pos]
        return out

    # ------------------------------------------------------------------ repr
    def __repr__(self) -> str:
        kind = "directed" if self.directed else "undirected"
        wt = "weighted" if self.weighted else "unweighted"
        return (
            f"Graph(n={self.n}, m={self.m}, {kind}, {wt}, "
            f"dangling={self.dangling!r} [{self.dangling_nodes.size} nodes])"
        )
