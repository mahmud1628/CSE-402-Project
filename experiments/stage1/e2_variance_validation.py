"""E2: Empirical variance vs the paper's variance formulas.

For an unbiased estimator averaged over T i.i.d. samples, E||pi_hat - pi||_2^2
equals the single-sample variance divided by T. We therefore estimate the
single-sample variance as ``T * MSE`` over many independent trials and
compare it with:

(a) MCW:  Var = 1 - ||pi||_2^2                                         (Lemma 3.2)
(b) PW:   exact Var[x^(K)] and the bound alpha^(2K) (1 - ||pi||_2^2)    (Lemma 3.7)
(c) PPW:  per-walk variance of batch b, ||r_b||_1^2 - ||Pi r_b||_2^2    (Lemma 3.14),
          which should fall batch after batch as the residual shrinks.
"""

import numpy as np
import pandas as pd

from common import (
    ALPHAS, graph, ground_truth, nlogn, log, mpl, paper_alpha_label, parse_args,
    query_nodes, save_csv, save_fig, sources, METHOD_STYLE,
)
import pprlib as pl
from pprlib import evaluation, theory

DATASETS = ["toy", "email-Eu-core", "ca-GrQc"]


def mse_single(results, pi, T):
    """T * MSE and its standard error (from the spread of per-trial squared errors)."""
    sq = np.array([np.sum((r.estimate - pi) ** 2) for r in results])
    return T * sq.mean(), T * sq.std(ddof=1) / np.sqrt(sq.size)


def run(args):
    trials = 100 if args.quick else 300
    rows_mcw, rows_pw, rows_ppw = [], [], []
    for name in args.datasets:
        g = graph(name)
        qs = query_nodes(g, 1 if name == "toy" else 3, args.seed)
        srcs = sources(g, "ssq", qs) + ([] if name == "toy" else sources(g, "prc", qs))
        for alpha in args.alphas:
            T = 2000 if alpha < 0.95 else 500
            Ks = [0, 1, 2, 3, 5, 8, 12] if alpha < 0.95 else [0, 1, 5, 10, 25, 50, 100, 200]
            for tag, sigma in srcs:
                pi = ground_truth(name, g, sigma, alpha)
                # (a) MCW
                res = evaluation.run_trials(lambda r: pl.mcw(g, sigma, alpha, T, r), trials, args.seed)
                emp, se = mse_single(res, pi, T)
                rows_mcw.append(dict(dataset=name, alpha=alpha, source=tag, T=T, trials=trials,
                                     empirical=emp, se=se, theory=theory.walk_variance(pi)))
                # (b) PW for several K
                exact = theory.pw_variance_curve(g, pi, alpha, Ks)
                for K, ex in zip(Ks, exact):
                    res = evaluation.run_trials(lambda r: pl.pw(g, sigma, alpha, T, K, r), trials, args.seed + 1)
                    emp, se = mse_single(res, pi, T)
                    rows_pw.append(dict(dataset=name, alpha=alpha, source=tag, K=K, T=T, empirical=emp, se=se,
                                        exact=ex, bound=theory.pw_variance_bound(pi, alpha, K)))
                log(f"{name} {tag} alpha={alpha}: MCW & PW done")
            # (c) PPW residual-driven variance decay, on the first single-source query
            tag, sigma = srcs[0]
            pi = ground_truth(name, g, sigma, alpha)
            Pi = pl.ppr_matrix(g, alpha)
            B, K = 6, 1 if alpha < 0.95 else 10
            # Two per-batch budgets: a small one, and the paper's regime (about n ln n walks in total).
            # PPW only contracts the residual when the per-batch walk count is large enough.
            for T_b in sorted({500, max(500, nlogn(g.n) // 2)}):
                res = evaluation.run_trials(
                    lambda r: pl.ppw(g, sigma, alpha, B * T_b, K, B, r, record_residuals=True),
                    max(20, trials // 5), args.seed + 2)
                for b in range(B):
                    v = [theory.residual_walk_variance(Pi, x.history["residuals"][b]) for x in res]
                    rl1 = [x.history["residual_l1"][b] for x in res]
                    rows_ppw.append(dict(dataset=name, alpha=alpha, source=tag, batch=b + 1, K=K, T_batch=T_b,
                                         residual_l1=np.mean(rl1), per_walk_variance=np.mean(v),
                                         mcw_variance=theory.walk_variance(pi),
                                         pw_bound_same_K=theory.pw_variance_bound(pi, alpha, K)))
            log(f"{name} alpha={alpha}: PPW done")
    a, b, c = pd.DataFrame(rows_mcw), pd.DataFrame(rows_pw), pd.DataFrame(rows_ppw)
    save_csv(a, "e2a_mcw_variance")
    save_csv(b, "e2b_pw_variance")
    save_csv(c, "e2c_ppw_variance")
    plot(a, b, c, args)
    a["rel_err"] = (a.empirical - a.theory) / a.theory
    # Rows with exact variance below 1e-20 are at the float64 round-off floor and cannot be compared.
    b["rel_err_vs_exact"] = (b.empirical - b.exact) / b.exact.where(b.exact > 1e-20)
    log(f"MCW: max |empirical/theory - 1| = {a.rel_err.abs().max():.3f}")
    log(f"PW (exact var > 1e-20): max |empirical/exact - 1| = {b.rel_err_vs_exact.abs().max():.3f}; "
        f"bound violated: {int((b.exact > b.bound * (1 + 1e-9)).sum())} times")
    return a, b, c


def plot(a, b, c, args):
    plt = mpl()
    # (a) identity plot
    fig, ax = plt.subplots(figsize=(5, 4.4))
    lo, hi = a.theory.min() * 0.8, 1.05
    ax.plot([lo, hi], [lo, hi], color="#aaa9a4", linewidth=1, label="empirical = theory")
    for alpha, mk in zip(args.alphas, ("o", "s")):
        d = a[a.alpha == alpha]
        ax.errorbar(d.theory, d.empirical, yerr=2 * d.se, linestyle="none", marker=mk, markersize=7,
                    color=METHOD_STYLE["mcw"]["color"], mfc="#fcfcfb" if mk == "s" else None,
                    label=paper_alpha_label(alpha), elinewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("theory: 1 − ‖π‖₂²  (Lemma 3.2)")
    ax.set_ylabel("empirical single-walk variance  T·MSE  (±2 s.e.)")
    ax.set_title("E2a · MCW variance matches Lemma 3.2")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "e2a_mcw_variance")

    # (b) PW variance vs K, one panel per (dataset, alpha), first single-source query
    panels = b.groupby(["dataset", "alpha"], sort=False)
    fig, axes = plt.subplots(len(args.alphas), b.dataset.nunique(), figsize=(4.2 * b.dataset.nunique(), 3.6 * len(args.alphas)),
                             squeeze=False)
    names = list(dict.fromkeys(b.dataset))
    for (name, alpha), d in panels:
        ax = axes[list(args.alphas).index(alpha), names.index(name)]
        d = d[d.source == d.source.iloc[0]]
        st = METHOD_STYLE["pw"]
        ax.plot(d.K, d.bound, color=st["color"], linestyle="--", linewidth=1.2, label="bound α²ᴷ(1−‖π‖²)")
        ax.plot(d.K, d.exact, color=st["color"], label="exact Var[x̂⁽ᴷ⁾]")
        ax.errorbar(d.K, d.empirical, yerr=2 * d.se, linestyle="none", marker="o", color=INK_POINTS,
                    markersize=5, label="empirical T·MSE", elinewidth=1)
        ax.set_yscale("log")
        if d.exact.min() < 1e-22:
            ax.axhspan(1e-40, 1e-24, color="#f0efec", zorder=0)
            ax.text(d.K.max(), 3e-25, "float64 round-off floor", ha="right", va="top", fontsize=8, color="#52514e")
            ax.set_ylim(bottom=max(d.exact.min() / 10, 1e-34))
        ax.set_title(f"{name} · {d.source.iloc[0]} · {paper_alpha_label(alpha)}", fontsize=9)
        ax.set_xlabel("power iterations K")
        ax.set_ylabel("single-walk variance")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("E2b · PW: each power iteration shrinks the variance by at least α² (Lemma 3.7)", x=0.01, ha="left")
    fig.tight_layout()
    save_fig(fig, "e2b_pw_variance")

    # (c) PPW per-walk variance by batch
    names = list(dict.fromkeys(c.dataset))
    fig, axes = plt.subplots(len(args.alphas), len(names), figsize=(4.2 * len(names), 3.6 * len(args.alphas)),
                             squeeze=False, sharex=True)
    col = METHOD_STYLE["ppw"]["color"]
    for i, alpha in enumerate(args.alphas):
        for j, name in enumerate(names):
            ax = axes[i, j]
            d0 = c[(c.dataset == name) & (c.alpha == alpha)]
            budgets = sorted(d0.T_batch.unique())
            for T_b in budgets:
                d = d0[d0.T_batch == T_b]
                small = T_b == budgets[0] and len(budgets) > 1
                ax.semilogy(d.batch, d.per_walk_variance, color=col, marker="o",
                            linestyle="--" if small else "-", mfc="#fcfcfb" if small else col,
                            markeredgecolor=col, label=f"{T_b:,} walks / batch")
            ax.axhline(d0.mcw_variance.iloc[0], color=METHOD_STYLE["mcw"]["color"], linewidth=1,
                       label="MCW per-walk variance")
            ax.set_title(f"{name} · {paper_alpha_label(alpha)} · K={int(d0.K.iloc[0])}", fontsize=9)
            ax.set_xlabel("batch b")
            if j == 0:
                ax.set_ylabel("per-walk variance ‖r_b‖₁² − ‖Π r_b‖₂²")
            ax.legend(fontsize=7)
    fig.suptitle("E2c · PPW (Lemma 3.14): with enough walks per batch the residual and variance shrink geometrically;\n"
                 "with too few (dashed), Monte Carlo noise amplified by 1/(1−α) makes the residual grow", x=0.01, ha="left")
    fig.tight_layout()
    save_fig(fig, "e2c_ppw_variance")


INK_POINTS = "#0b0b0b"

if __name__ == "__main__":
    args = parse_args(__doc__)
    if args.datasets in (["email-Eu-core", "ca-GrQc"], None) or set(args.datasets) - set(DATASETS):
        args.datasets = [d for d in DATASETS if d in args.datasets or d == "toy"] or DATASETS
    run(args)
