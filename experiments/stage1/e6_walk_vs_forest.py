"""E6: alpha-random-walk vs spanning-forest estimators for PageRank centrality (paper Table 2, Lemma 4.4).

With sigma = 1/n, the single-sample variances are:

    walk estimator x-hat:        1 - ||pi||^2
    forest root estimator x~:    n_r / n - ||pi||^2,   where n_r = (1/n) 1^T Q 1
    forest degree estimator x.:  n_rd / n - ||pi||^2   (undirected graphs only)

At equal running time, the predicted MSE is therefore

    walks:   (1 - ||pi||^2) / T
    forests: (n_r / n - ||pi||^2) / F,   with F = T * tau_walk / tau_forest

We measure n_r, n_rd, tau_walk and tau_forest, predict which estimator wins,
and compare the prediction with the measured L1 error and MSE (T = n ln n walks).
"""

import math

import numpy as np
import pandas as pd

from common import (
    budget, graph, ground_truth, log, nlogn, parse_args, save_csv, sources, too_expensive,
)
import pprlib as pl
from pprlib import evaluation, theory


def run(args):
    rows = []
    for name in args.datasets:
        g = graph(name)
        b = budget(name, args.quick)
        sigma = sources(g, "prc", None)[0][1]
        trials = max(2, b["trials"] // 2)
        for alpha in args.alphas:
            pi = ground_truth(name, g, sigma, alpha)
            T = nlogn(g.n)
            while too_expensive(T, alpha):
                T //= 4
            fs = theory.forest_statistics(g, alpha, n_forests=5 if b["large"] else 30, rng=args.seed)
            tw = theory.walk_time(g, alpha, n_walks=min(500_000, 20 * g.n), rng=args.seed)["tau_walk"]
            F = max(1, math.ceil(T * tw / fs["tau_forest"]))
            pi2 = float(np.dot(pi, pi))
            row = dict(dataset=name, alpha=alpha, paper_alpha=round(1 - alpha, 3), n=g.n, T=T, F=F,
                       n_pi2=g.n * pi2, n_tau_walk_over_tau_forest=g.n * tw / fs["tau_forest"],
                       n_r=fs["n_r"], n_rd=fs["n_rd"] if g.is_undirected() else np.nan)
            row["pred_mse_walk"] = (1 - pi2) / T
            row["pred_mse_forest"] = (fs["n_r"] / g.n - pi2) / F
            if g.is_undirected():
                row["pred_mse_forest_deg"] = (fs["n_rd"] / g.n - pi2) / F
            est = {
                "walk": lambda r: pl.mcw(g, sigma, alpha, T, r),
                "forest": lambda r: pl.mcf(g, sigma, alpha, F, r),
            }
            if g.is_undirected():
                est["forest_deg"] = lambda r: pl.mcf(g, sigma, alpha, F, r, variant="degree")
            for key, fn in est.items():
                res = evaluation.run_trials(fn, trials, args.seed)
                row[f"l1_{key}"] = float(np.mean([pl.metrics.l1_error(x.estimate, pi) for x in res]))
                row[f"mse_{key}"] = evaluation.mean_squared_error(res, pi)
                row[f"time_{key}"] = float(np.mean([x.time for x in res]))
            row["lemma44_predicts_forest_better"] = bool(row["pred_mse_forest"] < row["pred_mse_walk"])
            row["observed_forest_better"] = bool(row["mse_forest"] < row["mse_walk"])
            rows.append(row)
            log(f"{name} alpha={alpha}: n||pi||^2={row['n_pi2']:.2f} n_r={row['n_r']:.1f} "
                f"L1 walk={row['l1_walk']:.3f} forest={row['l1_forest']:.3f}"
                + (f" forestV={row['l1_forest_deg']:.4f}" if g.is_undirected() else ""))
    df = pd.DataFrame(rows)
    save_csv(df, "e6_walk_vs_forest")
    cols = ["dataset", "paper_alpha", "n_pi2", "n_tau_walk_over_tau_forest", "n_r", "l1_walk", "l1_forest"]
    cols += [c for c in ("l1_forest_deg",) if c in df]
    log("Table 2 reproduction:\n" + df[cols].to_string(index=False, float_format="%.3g"))
    agree = (df.lemma44_predicts_forest_better == df.observed_forest_better).mean()
    log(f"Lemma 4.4 prediction agrees with the measurement in {agree:.0%} of cases")
    return df


if __name__ == "__main__":
    run(parse_args(__doc__))
