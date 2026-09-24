"""Random sampling utilities: RNG handling and Walker's alias method."""

from __future__ import annotations

import numpy as np

from ._accel import njit


def make_rng(rng=None) -> np.random.Generator:
    """Accept ``None``, an int seed, or a Generator, and return a Generator."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


@njit(cache=True)
def _build_alias(scaled):
    """Vose's O(n) alias-table construction.

    ``scaled`` holds the probabilities multiplied by n, so they average 1.
    The array is modified in place.
    """
    n = scaled.size
    prob = np.ones(n)
    alias = np.arange(n)
    small = np.empty(n, dtype=np.int64)
    large = np.empty(n, dtype=np.int64)
    ns = 0
    nl = 0
    for i in range(n):
        if scaled[i] < 1.0:
            small[ns] = i
            ns += 1
        else:
            large[nl] = i
            nl += 1
    while ns > 0 and nl > 0:
        ns -= 1
        s = small[ns]
        l = large[nl - 1]
        prob[s] = scaled[s]
        alias[s] = l
        scaled[l] = (scaled[l] + scaled[s]) - 1.0
        if scaled[l] < 1.0:
            nl -= 1
            small[ns] = l
            ns += 1
    # Anything left over has probability 1 up to rounding error (prob is already 1).
    return prob, alias


class AliasTable:
    """O(1)-per-draw sampling from a discrete distribution (Walker, 1974).

    Only the non-zero entries of ``weights`` are kept, so sampling from a
    sparse vector, such as a residual with few non-zeros, stays cheap.
    ``sample`` returns indices into the original ``weights`` array.
    """

    def __init__(self, weights):
        w = np.asarray(weights, dtype=np.float64).ravel()
        if np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite and non-negative")
        self.support = np.flatnonzero(w)
        if self.support.size == 0:
            raise ValueError("weights must have a positive entry")
        p = w[self.support]
        self.total = float(p.sum())
        k = self.support.size
        self.prob, self.alias = _build_alias(p * (k / self.total))

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        k = self.support.size
        if k == 1:
            return np.full(size, self.support[0], dtype=np.int64)
        i = rng.integers(0, k, size=size)
        coin = rng.random(size)
        picked = np.where(coin < self.prob[i], i, self.alias[i])
        return self.support[picked]
