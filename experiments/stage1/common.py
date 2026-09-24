"""Shared configuration and helpers for the stage-1 experiments.

Every experiment writes a CSV (the table view of each figure) and a PNG to
``results/stage1/``. Ground-truth vectors are cached in ``results/cache/``.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pprlib as pl  # noqa: E402
from pprlib import datasets  # noqa: E402

RESULTS = ROOT / "results" / "stage1"
CACHE = ROOT / "results" / "cache"
RESULTS.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

# Proposal convention: alpha is the damping factor. The paper's 0.2 and 0.01 become 0.8 and 0.99.
ALPHA_LARGE_PAPER = 0.8   # paper alpha = 0.2
ALPHA_SMALL_PAPER = 0.99  # paper alpha = 0.01
ALPHAS = (ALPHA_LARGE_PAPER, ALPHA_SMALL_PAPER)

# Per-dataset budgets. Python is slower than the paper's C++, so the largest
# graph gets fewer queries and trials.
DATASETS = {
    "email-Eu-core": dict(queries=5, trials=20, large=False),
    "wiki-Vote": dict(queries=5, trials=10, large=False),
    "ca-GrQc": dict(queries=5, trials=10, large=False),
    "com-youtube": dict(queries=3, trials=2, large=True),
}
QUICK = ["email-Eu-core", "ca-GrQc"]


def parse_args(description: str, default_alphas=ALPHAS):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--datasets", nargs="+", default=None, help="subset of: " + ", ".join(DATASETS))
    ap.add_argument("--alphas", nargs="+", type=float, default=list(default_alphas))
    ap.add_argument("--quick", action="store_true", help="small datasets only, fewer trials")
    ap.add_argument("--no-large", action="store_true", help="skip com-youtube")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.datasets is None:
        args.datasets = QUICK if args.quick else [
            d for d in DATASETS if not (args.no_large and DATASETS[d]["large"])
        ]
    return args


def budget(name: str, quick: bool) -> dict:
    b = dict(DATASETS.get(name, dict(queries=3, trials=5, large=False)))
    if quick:
        b["queries"] = min(b["queries"], 3)
        b["trials"] = max(2, b["trials"] // 4)
    return b


@lru_cache(maxsize=None)
def graph(name: str) -> pl.Graph:
    if name == "toy":
        return datasets.paper_toy_graph()
    return datasets.load_snap(name)


def query_nodes(g: pl.Graph, k: int, seed: int = 0) -> np.ndarray:
    """k distinct random query sources with out-degree > 0 (the paper's protocol)."""
    rng = np.random.default_rng(seed)
    cand = np.flatnonzero(g.out_degree > 0)
    return np.sort(rng.choice(cand, size=min(k, cand.size), replace=False))


def sources(g: pl.Graph, kind: str, queries: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """[(tag, sigma)] for single-source queries ("ssq") or PageRank centrality ("prc")."""
    if kind == "ssq":
        return [(f"s{int(q)}", pl.one_hot(g, g.label(q))) for q in queries]
    if kind == "prc":
        return [("uniform", pl.uniform(g))]
    raise ValueError(kind)


def ground_truth(name: str, g: pl.Graph, sigma: np.ndarray, alpha: float) -> np.ndarray:
    """Exact PPR (direct solve or 1e-15 power iteration), cached on disk."""
    key = hashlib.sha1(sigma.tobytes()).hexdigest()[:12]
    path = CACHE / f"{name}_a{alpha}_{key}.npy"
    if path.exists():
        return np.load(path)
    pi = pl.exact_ppr(g, sigma, alpha, method="auto", tol=1e-15)
    np.save(path, pi)
    return pi


def nlogn(n: int) -> int:
    return math.ceil(n * math.log(n))


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.s = time.perf_counter() - self.t0


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- plotting
# Categorical palette (validated: adjacent CVD dE >= 9.1, normal-vision dE >= 19.6).
# Colour follows the method, never its rank, and is identical in every figure.
# Spanning-forest "V" variants share their base colour and use a dashed line
# plus a hollow marker as secondary encoding.
PALETTE = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
    "yellow": "#eda100", "magenta": "#e87ba4", "violet": "#4a3aa7",
}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

METHOD_STYLE = {
    "power": dict(color=INK_2, marker="s", label="Power method"),
    "mcw": dict(color=PALETTE["blue"], marker="o", label="MCW (standard MC)"),
    "pw": dict(color=PALETTE["orange"], marker="^", label="PW"),
    "ppw": dict(color=PALETTE["aqua"], marker="D", label="PPW"),
    "mcf": dict(color=PALETTE["yellow"], marker="v", label="MCF"),
    "pf": dict(color=PALETTE["magenta"], marker="P", label="PF"),
    "ppf": dict(color=PALETTE["violet"], marker="X", label="PPF"),
}
for base in ("mcf", "pf", "ppf"):
    s = dict(METHOD_STYLE[base])
    s["label"] = s["label"] + "V"
    s["linestyle"] = "--"
    s["mfc"] = SURFACE
    METHOD_STYLE[base + "v"] = s


def mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 6, "lines.markeredgewidth": 1.2,
        "font.size": 10, "axes.titlesize": 11, "legend.frameon": False, "legend.fontsize": 9,
        "figure.dpi": 110, "savefig.dpi": 160,
    })
    return plt


def method_line(ax, x, y, method, **kw):
    st = dict(METHOD_STYLE[method])
    st.update(kw)
    st.setdefault("markeredgecolor", st["color"])
    return ax.plot(x, y, **st)


def alpha_colors(alphas):
    """Ordinal blue ramp (validated steps 250..650) for ordered alpha values."""
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
    idx = np.linspace(0, len(ramp) - 1, len(alphas)).round().astype(int)
    return [ramp[i] for i in idx]


def save_fig(fig, name: str):
    path = RESULTS / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    log(f"wrote {path.relative_to(ROOT)}")


def save_csv(df, name: str):
    path = RESULTS / f"{name}.csv"
    df.to_csv(path, index=False)
    log(f"wrote {path.relative_to(ROOT)}")


def paper_alpha_label(alpha: float) -> str:
    return f"α={alpha:g} (paper α={1 - alpha:.2g})"


# ------------------------------------------------------------ evaluation
# Rough budget guard: expected walk steps allowed per single run. Python simulates about
# 2e7 steps/s here, so 4e8 steps is about 20 s. Configurations above it are skipped
# and logged, never silently truncated.
MAX_STEPS = 4e8


def too_expensive(n_walks: float, alpha: float, limit: float = MAX_STEPS) -> bool:
    return n_walks / (1.0 - alpha) > limit


def evaluate(name, g, alpha, kind, queries, method, trials, seed=0, *, eps=None, **params) -> dict:
    """Run ``method`` ``trials`` times on every source of ``kind`` and average the summaries over sources."""
    from pprlib import evaluation

    rows = []
    for tag, sigma in sources(g, kind, queries):
        pi = ground_truth(name, g, sigma, alpha)
        res = evaluation.run_trials(
            lambda r: pl.compute_ppr(g, sigma, alpha, method, rng=r, **params), trials, seed)
        rows.append(evaluation.summarize(res, pi, k=10, mu=1.0 / g.n, eps=eps))
    keys = rows[0].keys()
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["max_rel_worst"] = float(np.max([r.get("max_rel", np.nan) for r in rows]))
    out["n_sources"] = len(rows)
    return out
