"""E1: Convergence of the deterministic power method.

We check three things:
* the error decays geometrically at rate alpha (a priori: ||r^(k) - pi||_1 <= 2 alpha^k);
* the a-posteriori bound alpha/(1-alpha) * ||r^(k) - r^(k-1)||_1 holds and is tight;
* the number of iterations needed for a tolerance grows like log(tol)/log(alpha),
  which is why small paper-alpha (alpha -> 1 here) is the hard case.
"""

import numpy as np
import pandas as pd
import scipy.sparse.linalg as spla

from common import (
    ROOT, alpha_colors, budget, graph, ground_truth, log, mpl, paper_alpha_label,
    parse_args, query_nodes, save_csv, save_fig, sources,
)
import pprlib as pl

ALPHAS = (0.5, 0.8, 0.9, 0.95, 0.99)


def second_eigenvalue_modulus(g) -> float:
    """|lambda_2(P)|. The power method's asymptotic rate is alpha * |lambda_2|, which is at most alpha."""
    vals = spla.eigs(g.P.astype(float), k=3, which="LM", return_eigenvectors=False, maxiter=5000, tol=1e-8)
    mods = np.sort(np.abs(vals))[::-1]
    return float(mods[1])


def run(args):
    rows, summary = [], []
    for name in args.datasets:
        g = graph(name)
        lam2 = second_eigenvalue_modulus(g)
        log(f"{name}: |lambda_2(P)| = {lam2:.4f}")
        q = query_nodes(g, 1, args.seed)
        for kind in ("ssq", "prc"):
            tag, sigma = sources(g, kind, q)[0]
            for alpha in args.alphas:
                pi = ground_truth(name, g, sigma, alpha)
                res = pl.power_iteration(g, sigma, alpha, tol=1e-13, max_iter=20000, exact=pi)
                err = res.history["error"]
                k = np.arange(err.size)
                apost = np.r_[np.nan, res.history["a_posteriori_bound"]]
                for kk in range(err.size):
                    rows.append(dict(dataset=name, source=kind, alpha=alpha, k=kk, error=err[kk],
                                     apriori_bound=2 * alpha ** kk, aposteriori_bound=apost[kk]))
                # Observed rate: geometric-mean decay factor over the range where the
                # error is above round-off level.
                ok = (err > 1e-12) & (k >= 1)
                slope = np.polyfit(k[ok], np.log(err[ok]), 1)[0] if ok.sum() > 2 else np.nan
                need = {tol: int(np.argmax(err < tol)) if (err < tol).any() else np.nan for tol in (1e-4, 1e-8)}
                summary.append(dict(
                    dataset=name, source=kind, alpha=alpha, paper_alpha=round(1 - alpha, 4),
                    observed_rate=float(np.exp(slope)), theory_rate=alpha, alpha_lambda2=alpha * lam2,
                    iters_to_1e4=need[1e-4], theory_iters_1e4=pl.theory.power_iterations_for_tol(alpha, 1e-4),
                    iters_to_1e8=need[1e-8], theory_iters_1e8=pl.theory.power_iterations_for_tol(alpha, 1e-8),
                    time_per_iter_ms=1e3 * res.time / max(res.n_power_iterations, 1),
                    apost_bound_holds=bool(np.all(err[1:] <= res.history["a_posteriori_bound"] * (1 + 1e-9) + 1e-15)),
                ))
                log(f"{name} {kind} alpha={alpha}: rate {np.exp(slope):.4f} (theory {alpha}), "
                    f"{res.n_power_iterations} iters, {res.time:.2f}s")
    df, sm = pd.DataFrame(rows), pd.DataFrame(summary)
    save_csv(df, "e1_power_convergence_curves")
    save_csv(sm, "e1_power_convergence_summary")
    plot(df, sm, args)
    return sm


def plot(df, sm, args):
    plt = mpl()
    names = list(dict.fromkeys(df.dataset))
    fig, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 7), squeeze=False, sharey=True)
    cols = alpha_colors(args.alphas)
    for j, name in enumerate(names):
        for i, kind in enumerate(("ssq", "prc")):
            ax = axes[i, j]
            for a, c in zip(args.alphas, cols):
                d = df[(df.dataset == name) & (df.source == kind) & (df.alpha == a)]
                ax.semilogy(d.k, d.error, color=c, label=paper_alpha_label(a))
                ax.semilogy(d.k, d.apriori_bound, color=c, linewidth=1, linestyle=":")
            ax.set_ylim(1e-14, 3)
            ax.set_xlim(0, df[(df.dataset == name) & (df.source == kind)].k.max())
            ax.set_title(f"{name} · {'single-source' if kind == 'ssq' else 'PageRank centrality'}")
            ax.set_xlabel("iteration k")
            if j == 0:
                ax.set_ylabel("L1 error ‖r⁽ᵏ⁾ − π‖₁")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("E1 · Power-method L1 error vs iteration: straight lines = geometric decay at rate α\n"
                 "solid = measured error, dotted = a-priori bound 2αᵏ", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_fig(fig, "e1_power_convergence")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0.45, 1], [0.45, 1], color="#aaa9a4", linewidth=1, label="rate = α (theory)")
    for kind, mk in (("ssq", "o"), ("prc", "s")):
        d = sm[sm.source == kind]
        ax.plot(d.theory_rate, d.observed_rate, linestyle="none", marker=mk, markersize=8,
                color="#2a78d6", mfc="#fcfcfb" if kind == "prc" else "#2a78d6",
                label="single-source" if kind == "ssq" else "PageRank centrality")
    ax.set_xlabel("damping factor α")
    ax.set_ylabel("observed decay factor per iteration")
    ax.set_title("E1 · Observed convergence rate vs theory (all datasets)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "e1_power_rate")


if __name__ == "__main__":
    run(parse_args(__doc__, default_alphas=ALPHAS))
