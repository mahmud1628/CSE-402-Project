"""E7: Relative-error guarantee (paper Definition 2.1, Lemma 3.11, Figs. 7-8).

Goal: |pi_hat(u) - pi(u)| <= eps * pi(u) for every u with pi(u) >= mu = 1/n,
with failure probability at most p_f = 1/n. Parameters come from theory:

* MCW: T = W = (2 eps/3 + 2) ln(2/p_f) / (eps^2 mu), the Chernoff bound with K = 0.
* PW:  T = n ln n and K = log_alpha(eps^2), the paper's setting (Section 5.6).
* PPW: T = n ln n, K = log_alpha(eps^2 / B) per batch, B = 3.

For each eps we report the time, the L1 error, the worst relative error over
the relevant nodes, and whether the guarantee held (max_rel <= eps).
"""

import numpy as np
import pandas as pd

from common import (
    budget, evaluate, graph, log, method_line, mpl, nlogn, paper_alpha_label,
    parse_args, query_nodes, save_csv, save_fig, too_expensive,
)
from pprlib import theory

EPS = (0.5, 0.4, 0.3, 0.2, 0.1)


def run(args):
    rows = []
    for name in args.datasets:
        g = graph(name)
        b = budget(name, args.quick)
        qs = query_nodes(g, b["queries"], args.seed)
        trials = max(2, b["trials"] // 4)
        for alpha in args.alphas:
            for kind in ("ssq", "prc"):
                for eps in EPS:
                    configs = {
                        "mcw": dict(n_walks=theory.mcw_walks_for_guarantee(g.n, eps)),
                        "pw": dict(n_walks=nlogn(g.n), K=theory.theoretical_K(alpha, eps)),
                        "ppw": dict(n_walks=nlogn(g.n), K=theory.theoretical_K(alpha, eps / np.sqrt(3)), n_batches=3),
                    }
                    for m, kw in configs.items():
                        if too_expensive(kw["n_walks"], alpha):
                            log(f"skip {name} alpha={alpha} {m} eps={eps}: T={kw['n_walks']:,} (budget guard)")
                            continue
                        s = evaluate(name, g, alpha, kind, qs, m, trials, args.seed, eps=eps, **kw)
                        rows.append(dict(dataset=name, alpha=alpha, source=kind, method=m, eps=eps,
                                         K=kw.get("K", 0), guarantee_met=bool(s["max_rel_worst"] <= eps), **s))
                log(f"{name} alpha={alpha} {kind}: done")
    df = pd.DataFrame(rows)
    save_csv(df, "e7_relative_error")
    log("guarantee met (fraction of configurations):\n"
        + df.groupby("method").guarantee_met.mean().to_string(float_format="%.2f"))
    plot(df)
    return df


def plot(df):
    plt = mpl()
    combos = list(dict.fromkeys(zip(df.dataset, df.alpha)))
    for kind in ("ssq", "prc"):
        fig, axes = plt.subplots(3, len(combos), figsize=(3.9 * len(combos), 9), squeeze=False)
        seen = {}
        for j, (name, alpha) in enumerate(combos):
            d = df[(df.dataset == name) & (df.alpha == alpha) & (df.source == kind)]
            for m in dict.fromkeys(d.method):
                dm = d[d.method == m].sort_values("eps", ascending=False)
                (ln,) = method_line(axes[0, j], dm.eps, dm.time, m)
                method_line(axes[1, j], dm.eps, dm.l1.clip(lower=1e-16), m)
                method_line(axes[2, j], dm.eps, dm.max_rel_worst, m)
                seen.setdefault(m, ln)
            e = np.array(sorted(df.eps.unique()))
            axes[2, j].plot(e, e, color="#aaa9a4", linewidth=1, linestyle=":", label="max_rel = ε (guarantee)")
            for i in range(3):
                axes[i, j].set_yscale("log")
                axes[i, j].invert_xaxis() if not axes[i, j].xaxis_inverted() else None
            axes[0, j].set_title(f"{name} · {paper_alpha_label(alpha)}", fontsize=9)
            axes[2, j].set_xlabel("target relative error ε")
        axes[0, 0].set_ylabel("time per query (s)")
        axes[1, 0].set_ylabel("mean L1 error")
        axes[2, 0].set_ylabel("worst relative error (π ≥ 1/n)")
        order = [m for m in ("mcw", "pw", "ppw") if m in seen]
        h = [seen[m] for m in order] + [axes[2, 0].lines[-1]]
        fig.legend(h, [x.get_label() for x in h], loc="lower center", ncol=len(h), bbox_to_anchor=(0.5, -0.01))
        what = "single-source" if kind == "ssq" else "PageRank centrality"
        fig.suptitle(f"E7 · Relative-error guarantee, {what}: points below the dotted line meet the guarantee",
                     x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        save_fig(fig, f"e7_relative_error_{kind}")


if __name__ == "__main__":
    run(parse_args(__doc__))
