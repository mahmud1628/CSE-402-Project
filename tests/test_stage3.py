"""Small, offline checks for Stage 3 recording and reproducibility."""
from __future__ import annotations

from argparse import Namespace
import numpy as np
import pandas as pd
import pytest

from experiments.stage3 import common


def _args(tmp_path, *, limit=1):
    stage2 = tmp_path / "stage2"
    common.write_synthetic_stage2(stage2)
    return Namespace(stage2_results=stage2, output=tmp_path / "stage3", seed=17,
                     trials=2, quick=True, network_limit=limit, resume=True)


def test_alignment_rejects_missing_truth_label(tmp_path):
    args = _args(tmp_path)
    net = common.load_networks(args.stage2_results)[0]
    path = args.stage2_results / "ground_truth" / f"{net.network_id}_a0.8_influence.csv"
    pd.read_csv(path, index_col="destination").iloc[:-1].to_csv(path)
    with pytest.raises(ValueError, match="labels"):
        common.truth_for(args.stage2_results, net, .8, "goalkeeper:10", common.pl.one_hot(net.graph, net.graph.labels[0]))


def test_seed_is_order_independent():
    assert common.stable_seed(7, "E", "network", 1) == common.stable_seed(7, "E", "network", 1)
    assert common.stable_seed(7, "E", "network", 1) != common.stable_seed(7, "E", "network", 2)


def test_trials_summary_and_progressive_budget(tmp_path):
    args = _args(tmp_path); cfg = common.config(args); nets = common.load_networks(args.stage2_results); net = nets[0]
    common.run_manifest(args, cfg, nets); sigma = common.pl.uniform(net.graph)
    rows = common.execute_trials(args, cfg, experiment="unit", cohort="secondary", network=net,
                                 source_id="uniform", sigma=sigma, alpha=.8, method="ppw", T=60, K=2, B=3, residuals=True)
    assert len(rows) == 2
    assert set(rows.n_power_iterations) == {9}
    summary = common.summarize(args.output, args.stage2_results, {n.network_id: n for n in nets}, common.read_trials(args.output))
    row = summary.iloc[0]; X = np.stack([np.load(args.output / p) for p in rows.estimate_path])
    truth = common.truth_for(args.stage2_results, net, .8, "uniform", sigma)
    lhs = ((X - truth) ** 2).sum(axis=1).mean()
    rhs = (len(X)-1)/len(X) * row.total_sample_variance + ((X.mean(0)-truth) ** 2).sum()
    assert lhs == pytest.approx(rhs)


def test_small_end_to_end_e1_e2(tmp_path):
    from experiments.stage3 import e1_power_convergence, e2_error_vs_samples
    args = _args(tmp_path, limit=1)
    e1_power_convergence.run(args)
    summary = e2_error_vs_samples.run(args)
    assert (args.output / "figures" / "e1_power_convergence.png").exists()
    assert (args.output / "tables" / "trials.csv").exists()
    assert not summary.empty
