"""Deterministic tests: graph construction, transition matrix, exact/power solvers, paper example."""

import numpy as np
import pytest

import pprlib as pl
from pprlib import datasets, theory
from pprlib.sampling import AliasTable


# --------------------------------------------------------------------- graph
def test_duplicates_unweighted_collapse_weighted_sum():
    g = pl.Graph(3, [0, 0, 1], [1, 1, 2])
    assert g.m == 2 and g.out_degree.tolist() == [1, 1, 0]
    gw = pl.Graph(3, [0, 0, 1], [1, 1, 2], [2.0, 3.0, 1.0])
    assert gw.A[0, 1] == 5.0


def test_undirected_symmetrises_and_is_idempotent():
    g1 = pl.Graph(3, [0, 1], [1, 2], directed=False)
    g2 = pl.Graph(3, [0, 1, 1, 2], [1, 0, 2, 1], directed=False)
    assert g1.m == g2.m == 4 and g1.is_undirected()
    assert (g1.A != g2.A).nnz == 0


@pytest.mark.parametrize("dangling", ["self_loop", "uniform"])
def test_P_is_column_stochastic(dangling):
    g = datasets.random_weighted_digraph(40, 0.05, rng=0, dangling=dangling)
    assert g.dangling_nodes.size > 0
    P = g.dense_P()
    np.testing.assert_allclose(P.sum(axis=0), 1.0)
    x = np.random.default_rng(1).random(g.n)
    np.testing.assert_allclose(g.apply_P(x), P @ x)


def test_weighted_step_distribution():
    g = pl.Graph(3, [0, 0, 0], [0, 1, 2], [1.0, 2.0, 7.0])
    rng = np.random.default_rng(0)
    nxt = g.step(np.zeros(200_000, dtype=np.int64), rng)
    freq = np.bincount(nxt, minlength=3) / nxt.size
    np.testing.assert_allclose(freq, [0.1, 0.2, 0.7], atol=0.005)


def test_labels_and_from_adjacency():
    W = np.array([[0, 3, 0], [1, 0, 1], [0, 0, 0]])
    g = pl.Graph.from_adjacency(W, labels=["a", "b", "c"])
    assert g.index("b") == 1 and g.label(2) == "c"
    assert g.dangling_nodes.tolist() == [2]
    np.testing.assert_allclose(g.dense_P()[:, 1], [0.5, 0, 0.5])


def test_alias_table_frequencies():
    w = np.array([0.0, 1.0, 0.0, 3.0, 6.0])
    t = AliasTable(w)
    s = t.sample(300_000, np.random.default_rng(0))
    np.testing.assert_allclose(np.bincount(s, minlength=5) / s.size, w / w.sum(), atol=0.004)


# ------------------------------------------------------- exact / power / toy
def test_paper_running_example():
    g = datasets.paper_toy_graph()
    s = pl.one_hot(g, "v1")
    alpha = theory.from_paper_alpha(0.2)
    np.testing.assert_allclose(pl.exact_ppr(g, s, alpha), [0.2, 12 / 35, 8 / 35, 8 / 35])
    # Fig. 2(c): a walk from v1 stops at v2, giving x^(1) and x^(4)
    e2 = pl.one_hot(g, "v2")
    np.testing.assert_allclose(pl.propagate(g, s, alpha, e2, 1), [0.2, 0, 0.4, 0.4])
    np.testing.assert_allclose(pl.propagate(g, s, alpha, e2, 4), [0.2, 0.3648, 0.2176, 0.2176])
    # Fig. 3: residual of pi_hat = (0.2, 0.34, 0.23, 0.23)
    r = pl.residual(g, s, alpha, np.array([0.2, 0.34, 0.23, 0.23]))
    np.testing.assert_allclose(r, [0, 0.02, -0.01, -0.01], atol=1e-12)


@pytest.mark.parametrize("dangling", ["self_loop", "uniform"])
def test_exact_solvers_agree(dangling):
    g = datasets.random_weighted_digraph(60, 0.04, rng=3, dangling=dangling)
    S = np.eye(g.n)[:, :5]
    dense = pl.exact_ppr(g, S, 0.85, "dense")
    np.testing.assert_allclose(pl.exact_ppr(g, S, 0.85, "direct"), dense, atol=1e-12)
    np.testing.assert_allclose(pl.exact_ppr(g, S, 0.85, "power"), dense, atol=1e-12)
    np.testing.assert_allclose(dense.sum(axis=0), 1.0)


def test_power_iteration_convergence_and_bounds():
    g = datasets.erdos_renyi(200, 0.03, rng=1)
    s = pl.one_hot(g, 0)
    pi = pl.exact_ppr(g, s, 0.9)
    res = pl.power_iteration(g, s, 0.9, tol=1e-12, exact=pi)
    assert res.history["converged"]
    err = res.history["error"]
    k = np.arange(err.size)
    assert np.all(err <= 2 * 0.9 ** k + 1e-12)  # a-priori bound
    assert np.all(err[1:] <= res.history["a_posteriori_bound"] + 1e-12)  # a-posteriori bound
    assert err[-1] < 1e-10


def test_residual_invariant():
    g = datasets.random_weighted_digraph(50, 0.08, rng=2)
    s = pl.sources.random_distribution(g, 0)
    Pi = pl.ppr_matrix(g, 0.8)
    x = np.random.default_rng(5).random(g.n)  # any vector at all
    r = pl.residual(g, s, 0.8, x)
    np.testing.assert_allclose(x + Pi @ r, Pi @ s, atol=1e-12)  # Lemma 3.12


def test_pw_linearity_lemma_3_8():
    """Propagating the average equals averaging the propagated per-walk estimators."""
    g = datasets.paper_toy_graph()
    s = pl.one_hot(g, "v1")
    ends = np.array([1, 1, 2, 3, 0])
    avg_then_prop = pl.propagate(g, s, 0.8, np.bincount(ends, minlength=4) / 5, 3)
    prop_then_avg = np.mean([pl.propagate(g, s, 0.8, np.eye(4)[t], 3) for t in ends], axis=0)
    np.testing.assert_allclose(avg_then_prop, prop_then_avg)


def test_theory_helpers():
    assert theory.to_paper_alpha(0.8) == pytest.approx(0.2)
    assert theory.theoretical_K(0.8, 0.5) == 7  # log_0.8(0.25) = 6.2
    assert theory.power_iterations_for_tol(0.5, 1e-3) == 11
