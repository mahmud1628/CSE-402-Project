"""E1: deterministic power convergence and exact-solver timing baselines."""
from __future__ import annotations
import numpy as np
import pandas as pd
try:  # package import for tests; direct-script import for the documented command
    from .common import config, load_networks, match_networks, output_paths, parse_args, run_manifest, save_figure, truth_for, track, pl, mpl
except ImportError:  # pragma: no cover
    from common import config, load_networks, match_networks, output_paths, parse_args, run_manifest, save_figure, truth_for, track, pl, mpl

def run(args):
    cfg = config(args); all_networks = load_networks(args.stage2_results); nets = match_networks(all_networks)
    if args.quick: nets = nets[:min(4, len(nets))]
    if cfg["network_limit"]: nets = nets[:cfg["network_limit"]]
    run_manifest(args, cfg, all_networks); paths = output_paths(args.output); rows = []
    jobs = [dict(network=net, alpha=alpha) for net in nets for alpha in cfg["alphas"]]
    for job in track("E1 power", jobs, args.output, lambda j: f"a={j['alpha']:g}", resumable=False):
        net, alpha = job["network"], job["alpha"]; sigma = pl.uniform(net.graph)
        truth = truth_for(args.stage2_results, net, alpha, "uniform", sigma)
        exact = pl.compute_ppr(net.graph, sigma, alpha, "exact")
        power = pl.power_iteration(net.graph, sigma, alpha, tol=1e-13, exact=truth)
        err, diff = power.history["error"], power.history["diff"]
        for k in range(len(err)):
            rows.append(dict(network_id=net.network_id, match_id=net.match_id, competition=net.competition, alpha=alpha,
                             iteration=k, l1_error=err[k], prior_bound=2 * alpha ** k,
                             posterior_bound=(alpha / (1-alpha) * diff[k-1]) if k else np.nan,
                             step_norm=diff[k-1] if k else np.nan, exact_seconds=exact.time,
                             power_seconds=power.time, power_iterations=power.n_power_iterations))
    frame = pd.DataFrame(rows); frame.to_csv(paths["tables"] / "power_convergence.csv", index=False)
    targets = []
    for (nid, alpha), d in frame.groupby(["network_id", "alpha"]):
        for tol in (1e-4, 1e-6, 1e-8):
            hit = d.loc[d.l1_error <= tol, "iteration"]
            targets.append(dict(network_id=nid, alpha=alpha, tolerance=tol, iterations=int(hit.iloc[0]) if len(hit) else np.nan))
    pd.DataFrame(targets).to_csv(paths["tables"] / "power_targets.csv", index=False)
    plt = mpl(); fig, ax = plt.subplots(figsize=(6, 4))
    for (alpha, nid), d in frame.groupby(["alpha", "network_id"]): ax.semilogy(d.iteration, d.l1_error, alpha=.45, label=f"a={alpha:g}" if nid == frame.network_id.iloc[0] else None)
    ax.set(xlabel="power iterations", ylabel="true L1 error", title="Stage 3 E1 · power convergence")
    ax.legend(); fig.tight_layout(); save_figure(fig, args.output, "e1_power_convergence")
    return frame
if __name__ == "__main__": run(parse_args(__doc__))
