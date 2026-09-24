"""Common result container returned by every PPR method."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PPRResult:
    """An estimated PPR vector plus its cost accounting.

    Attributes
    ----------
    estimate : ndarray, shape (n,)
        The estimated PPR vector pi_hat.
    method : str
        Name of the algorithm that produced it.
    alpha : float
        Damping factor (proposal convention: probability of *continuing* the walk).
    time : float
        Wall-clock seconds spent inside the algorithm.
    n_walks, n_walk_steps : int
        Random walks simulated and edges traversed in total.
    n_forests : int
        Rooted spanning forests sampled.
    n_power_iterations : int
        Sparse matrix-vector products with P.
    history : dict
        Per-iteration or per-batch diagnostics, e.g. residual norms.
    """

    estimate: np.ndarray
    method: str
    alpha: float
    time: float = 0.0
    n_walks: int = 0
    n_walk_steps: int = 0
    n_forests: int = 0
    n_power_iterations: int = 0
    history: dict = field(default_factory=dict)

    def top_k(self, k: int, graph=None):
        """The k highest-scoring nodes as ``[(node, score), ...]``, using labels if ``graph`` is given."""
        order = np.argsort(-self.estimate, kind="stable")[:k]
        name = graph.label if graph is not None else int
        return [(name(i), float(self.estimate[i])) for i in order]

    def as_dict(self, graph) -> dict:
        """Map each node label to its score."""
        return {graph.label(i): float(v) for i, v in enumerate(self.estimate)}
