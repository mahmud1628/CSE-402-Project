"""E3: Accuracy and ranking stability vs the number of random-walk samples T.

MCW, PW and PPW all average T unbiased samples, so the error falls like
T^(-1/2), a slope of -1/2 on a log-log plot. Variance reduction does not
change the slope; it lowers the constant, i.e. the line shifts down. The
vertical gap between the lines is the sample saving: how many times fewer
walks reach the same accuracy. ``T * MSE`` should be constant in T and equal
the per-sample variance.

K follows the paper's defaults, K = log_alpha(eps^2) for PW and
log_alpha(eps^2 / B) per batch for PPW, with eps = 0.5 and B = 3.
"""

import numpy as np
import pandas as pd

from common import (
    METHOD_STYLE, budget, evaluate, graph, log, method_line, mpl, nlogn,
    paper_alpha_label, parse_args, query_nodes, save_csv, save_fig, too_expensive,
)

METHODS = ("mcw", "pw", "ppw")
FACTORS = (1 / 256, 1 / 64, 1 / 16, 1 / 4, 1)


def run(args):
    rows = []
    for name in args.datasets:
        g = graph(name)
        b = budget(name, args.quick)
        qs = query_nodes(g, b["queries"], args.seed)
        for alpha in args.alphas:
            for kind in ("ssq", "prc"):
                for f in FACTORS:
                    T = max(10, int(f * nlogn(g.n)))
                    if too_expensive(T, alpha):
                        log(f"skip {name} alpha={alpha} T={T:,} (budget guard)")
                        continue
                    for m in METHODS:
                        s = evaluate(name, g, alpha, kind, qs, m, b["trials"], args.seed, n_walks=T)
                        rows.append(dict(dataset=name, alpha=alpha, source=kind, method=m, T=T,
                                         T_over_nlogn=f, T_times_mse=T * s["mse"], **s))
                    log(f"{name} alpha={alpha} {kind} T={T:,}: " + ", ".join(
                        f"{r['method']} L1={r['l1']:.2e}" for r in rows[-3:]))
    df = pd.DataFrame(rows)
    save_csv(df, "e3_error_vs_samples")
    plot(df, args)
    savings(df)
    return df


def savings(df):
    """Measured error ratio L1(MCW) / L1(method) at the same number of walks T = n ln n.

    This is compared directly, with no extrapolation of MCW's T^(-1/2) law.
    """
    out = []
    for (name, alpha, kind), d in df.groupby(["dataset", "alpha", "source"]):
        Tmax = d["T"].max()
        mc = d[(d.method == "mcw") & (d["T"] == Tmax)]
        for m in ("pw", "ppw"):
            dm = d[(d.method == m) & (d["T"] == Tmax)]
            if dm.empty or mc.empty:
                continue
            out.append(dict(dataset=name, alpha=alpha, source=kind, method=m, T=int(Tmax),
                            l1_mcw=mc.l1.iloc[0], l1=dm.l1.iloc[0], error_reduction=mc.l1.iloc[0] / dm.l1.iloc[0],
                            mse_reduction=mc.mse.iloc[0] / dm.mse.iloc[0]))
    s = pd.DataFrame(out)
    save_csv(s, "e3_error_reduction")
    log("error reduction vs MCW at equal T:\n" + s.to_string(index=False, float_format="%.3g"))


def plot(df, args):
    plt = mpl()
    for alpha in sorted(df.alpha.unique()):
        d_a = df[df.alpha == alpha]
        names = list(dict.fromkeys(d_a.dataset))
        for metric, ylabel, fname, logy in (
            ("l1", "mean L1 error ‖π̂ − π‖₁", "e3_l1_vs_T", True),
            ("prec@10", "top-10 precision vs exact ranking", "e3_top10_vs_T", False),
        ):
            fig, axes = plt.subplots(2, len(names), figsize=(4.1 * len(names), 7), squeeze=False)
            for j, name in enumerate(names):
                for i, kind in enumerate(("ssq", "prc")):
                    ax = axes[i, j]
                    d = d_a[(d_a.dataset == name) & (d_a.source == kind)]
                    for m in METHODS:
                        dm = d[d.method == m].sort_values("T")
                        method_line(ax, dm["T"], dm[metric], m)
                        if metric == "l1":
                            ax.fill_between(dm["T"], np.maximum(dm.l1 - dm.l1_std, dm.l1 * 0.2), dm.l1 + dm.l1_std,
                                            color=METHOD_STYLE[m]["color"], alpha=0.1, linewidth=0)
                    if metric == "l1" and not d.empty:
                        mc = d[d.method == "mcw"].sort_values("T")
                        ref = mc.l1.iloc[0] * np.sqrt(mc["T"].iloc[0] / mc["T"])
                        ax.plot(mc["T"], ref * 1.6, color="#aaa9a4", linewidth=1, linestyle=":", label="slope −½")
                    ax.set_xscale("log")
                    if logy:
                        ax.set_yscale("log")
                    else:
                        ax.set_ylim(-0.02, 1.02)
                    ax.set_title(f"{name} · {'single-source' if kind == 'ssq' else 'PageRank centrality'}", fontsize=10)
                    ax.set_xlabel("random walks T")
                    if j == 0:
                        ax.set_ylabel(ylabel)
            h, l = axes[0, 0].get_legend_handles_labels()
            fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.01))
            what = "L1 error" if metric == "l1" else "Top-10 ranking accuracy"
            fig.suptitle(f"E3 · {what} vs number of walks · {paper_alpha_label(alpha)}", x=0.01, ha="left")
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            save_fig(fig, f"{fname}_a{alpha}")


if __name__ == "__main__":
    run(parse_args(__doc__))
