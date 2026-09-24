"""Statistical tests: unbiasedness and variance of every Monte Carlo estimator.

The seeds are fixed, so the tests are deterministic. Tolerances are several
standard errors wide.
"""

import numpy as np
import pytest

import pprlib as pl
from pprlib import datasets, evaluation, theory

ALPHA = 0.8


@pytest.fixture(scope="module")
def directed():
    g = datasets.random_weighted_digraph(30, 0.12, rng=7)  # weighted, has dangling nodes
    s = pl.one_hot(g, 0)
    return g, s, pl.exact_ppr(g, s, ALPHA)


@pytest.fixture(scope="module")
def undirected():
    g = datasets.barabasi_albert(40, 2, rng=4)
    s = pl.uniform(g)
    return g, s, pl.exact_ppr(g, s, ALPHA)


def _mean_close(results, pi, tol):
    mean = evaluation.stack(results).mean(axis=0)
    assert np.abs(mean - pi).sum() < tol, np.abs(mean - pi).sum()


@pytest.mark.parametrize("estimator", ["terminal", "visits"])
def test_mcw_unbiased(directed, estimator):
    g, s, pi = directed
    res = evaluation.run_trials(lambda r: pl.mcw(g, s, ALPHA, 2000, r, estimator=estimator), 200, seed=1)
    _mean_close(res, pi, 0.03)


def test_mcw_variance_matches_lemma_3_2(directed):
    g, s, pi = directed
    T = 500
    res = evaluation.run_trials(lambda r: pl.mcw(g, s, ALPHA, T, r), 400, seed=2)
    single = evaluation.mean_squared_error(res, pi) * T
    assert single == pytest.approx(theory.walk_variance(pi), rel=0.12)


@pytest.mark.parametrize("K", [1, 3])
def test_pw_unbiased_and_variance(directed, K):
    g, s, pi = directed
    T = 500
    res = evaluation.run_trials(lambda r: pl.pw(g, s, ALPHA, T, K, r), 400, seed=3)
    _mean_close(res, pi, 0.02)
    single = evaluation.mean_squared_error(res, pi) * T
    exact_var = theory.exact_pw_variance(g, pi, ALPHA, K)
    assert single == pytest.approx(exact_var, rel=0.15)
    assert exact_var <= theory.pw_variance_bound(pi, ALPHA, K) * (1 + 1e-9)


def test_ppw_unbiased_and_reduces_residual(directed):
    g, s, pi = directed
    res = evaluation.run_trials(lambda r: pl.ppw(g, s, ALPHA, 3000, 2, 3, r), 200, seed=4)
    _mean_close(res, pi, 0.01)
    r_hist = np.mean([x.history["residual_l1"] for x in res], axis=0)
    assert np.all(np.diff(r_hist) < 0)  # ||r||_1 shrinks batch by batch
    # PPW must beat PW for the same number of walks and total power iterations
    pw_res = evaluation.run_trials(lambda r: pl.pw(g, s, ALPHA, 3000, 6, r), 200, seed=5)
    assert evaluation.mean_squared_error(res, pi) < evaluation.mean_squared_error(pw_res, pi)


def test_residual_walk_variance_lemma_3_14(directed):
    g, s, pi = directed
    Pi = pl.ppr_matrix(g, ALPHA)
    x = pi + np.random.default_rng(0).normal(0, 0.01, g.n)
    r = pl.residual(g, s, ALPHA, x)
    T = 400
    rng_trials = evaluation.run_trials(
        lambda rng: pl.PPRResult(pl.walks.walk_estimate(g, r, ALPHA, T, rng)[0], "x", ALPHA), 400, seed=6
    )
    single = evaluation.mean_squared_error(rng_trials, Pi @ r) * T
    assert single == pytest.approx(theory.residual_walk_variance(Pi, r), rel=0.15)


def test_forest_root_is_walk_endpoint(directed):
    """root(u) should have the same distribution as the stopping node of a walk from u."""
    g, _, _ = directed
    Pi = pl.ppr_matrix(g, ALPHA)
    rng = np.random.default_rng(8)
    F = 4000
    counts = np.zeros((g.n, g.n))
    for _ in range(F):
        root, _ = pl.sample_forest(g, ALPHA, rng)
        counts[root, np.arange(g.n)] += 1
    np.testing.assert_allclose(counts / F, Pi, atol=0.03)


@pytest.mark.parametrize("method", ["mcf", "pf", "ppf"])
def test_forest_methods_unbiased_directed(directed, method):
    g, s, pi = directed
    fn = {
        "mcf": lambda r: pl.mcf(g, s, ALPHA, 20, r),
        "pf": lambda r: pl.pf(g, s, ALPHA, 20, 2, r),
        "ppf": lambda r: pl.ppf(g, s, ALPHA, 21, 2, 3, r),
    }[method]
    # Measured: L1 error of the plain forest average is about 4.5 / sqrt(#forests) here
    # (3000 forests in total gives about 0.08), so the tolerance is roughly 2 standard errors.
    _mean_close(evaluation.run_trials(fn, 150, seed=9), pi, 0.15)


def test_degree_estimator_unbiased_and_better(undirected):
    g, s, pi = undirected
    root_res = evaluation.run_trials(lambda r: pl.mcf(g, s, ALPHA, 5, r, variant="root"), 300, seed=10)
    deg_res = evaluation.run_trials(lambda r: pl.mcf(g, s, ALPHA, 5, r, variant="degree"), 300, seed=10)
    _mean_close(deg_res, pi, 0.03)
    _mean_close(root_res, pi, 0.12)  # L1 error is about 2.3 / sqrt(1500) = 0.06 for this estimator
    assert evaluation.mean_squared_error(deg_res, pi) < evaluation.mean_squared_error(root_res, pi)  # Lemma 4.3


def test_degree_variant_rejects_directed(directed):
    g, s, _ = directed
    with pytest.raises(ValueError):
        pl.mcf(g, s, ALPHA, 1, 0, variant="degree")


def test_compute_ppr_all_methods_with_labels():
    edges = [("A", "B", 3), ("B", "C", 1), ("C", "A", 2), ("B", "A", 5), ("C", "B", 1)]
    g = pl.Graph.from_edges(edges, directed=True)
    pi = pl.compute_ppr(g, "A", 0.85, "exact").estimate
    for m in pl.METHODS:
        if m.endswith("v") and m != "mcw":
            continue  # the degree (V) variants need an undirected graph
        res = pl.compute_ppr(g, "A", 0.85, m, rng=0, n_walks=20000, n_forests=2000, K=10)
        assert np.abs(res.estimate - pi).sum() < 0.05, m
        assert res.top_k(1, g)[0][0] in {"A", "B"}
