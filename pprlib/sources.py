"""Constructors and validation for the source / personalization distribution sigma.

This is ``s`` in the proposal: ``r = alpha P r + (1 - alpha) s``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .graph import Graph
from .sampling import make_rng


def one_hot(graph: Graph, node) -> np.ndarray:
    """sigma = e_s, which gives single-source PPR. ``node`` is a label or an index."""
    s = np.zeros(graph.n)
    s[graph.index(node)] = 1.0
    return s


def uniform(graph: Graph) -> np.ndarray:
    """sigma = 1/n, which gives PageRank centrality."""
    return np.full(graph.n, 1.0 / graph.n)


def from_weights(graph: Graph, weights) -> np.ndarray:
    """Normalise non-negative weights, given as an array of length n or a {node: weight} mapping."""
    if isinstance(weights, Mapping):
        s = np.zeros(graph.n)
        for node, w in weights.items():
            s[graph.index(node)] += float(w)
    else:
        s = np.array(weights, dtype=np.float64).ravel()
    return check_distribution(s / s.sum() if s.sum() > 0 else s, graph.n)


def random_distribution(graph: Graph, rng=None) -> np.ndarray:
    """n i.i.d. U[0,1] numbers, normalised (the setting of paper Section 5.7)."""
    s = make_rng(rng).random(graph.n)
    return s / s.sum()


def check_distribution(sigma, n: int) -> np.ndarray:
    """Validate that sigma is a probability vector of length n and return it as float64."""
    s = np.asarray(sigma, dtype=np.float64).ravel()
    if s.size != n:
        raise ValueError(f"sigma has length {s.size}, expected {n}")
    if np.any(s < 0) or not np.all(np.isfinite(s)):
        raise ValueError("sigma must be finite and non-negative")
    if not np.isclose(s.sum(), 1.0, atol=1e-9):
        raise ValueError(f"sigma must sum to 1 (sums to {s.sum():.6g})")
    return s


def as_distribution(graph: Graph, source=None) -> np.ndarray:
    """Convenience coercion.

    ``None`` means uniform, an array means a distribution, a mapping means
    weights, and anything else is a single node.
    """
    if source is None:
        return uniform(graph)
    if isinstance(source, Mapping):
        return from_weights(graph, source)
    if isinstance(source, np.ndarray) and source.ndim == 1 and source.size == graph.n:
        return check_distribution(source, graph.n)
    return one_hot(graph, source)
