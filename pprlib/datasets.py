"""Benchmark graphs: SNAP downloads, edge-list files, and self-written generators."""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .graph import Graph
from .sampling import make_rng

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# name -> (url, directed). SNAP undirected files list each edge in one or both
# directions; Graph(directed=False) symmetrises and de-duplicates either way.
SNAP = {
    "email-Eu-core": ("https://snap.stanford.edu/data/email-Eu-core.txt.gz", True),
    "wiki-Vote": ("https://snap.stanford.edu/data/wiki-Vote.txt.gz", True),
    "ca-GrQc": ("https://snap.stanford.edu/data/ca-GrQc.txt.gz", False),
    "ca-HepTh": ("https://snap.stanford.edu/data/ca-HepTh.txt.gz", False),
    "soc-Epinions1": ("https://snap.stanford.edu/data/soc-Epinions1.txt.gz", True),
    "com-youtube": ("https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz", False),
}


def load_edge_file(path, *, directed: bool, weighted: bool = False, dangling: str = "self_loop") -> Graph:
    """Read a whitespace-separated edge list: ``u v`` or ``u v w`` per line, with ``#`` comments.

    If the first data line holds a single integer n, the file uses the
    "pvr" format: the ids are already ``0..n-1``, so they are used directly.
    Otherwise the node ids are relabelled to ``0..n-1`` in sorted order, and
    ``graph.labels`` keeps the original ids.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        first = ""
        for line in fh:
            if line.strip() and not line.lstrip().startswith("#"):
                first = line.split()
                break
    header_n = int(first[0]) if len(first) == 1 else None
    cols = 3 if weighted else 2
    df = pd.read_csv(
        path, sep=r"\s+", comment="#", header=None, usecols=range(cols),
        skiprows=1 if header_n is not None else 0, engine="c",
    )
    u = df[0].to_numpy(np.int64)
    v = df[1].to_numpy(np.int64)
    w = df[2].to_numpy(np.float64) if weighted else None
    if header_n is not None:
        return Graph(header_n, u, v, w, directed=directed, dangling=dangling)
    ids, inv = np.unique(np.concatenate([u, v]), return_inverse=True)
    return Graph(
        ids.size, inv[: u.size], inv[u.size:], w, directed=directed,
        labels=ids.tolist(), dangling=dangling,
    )


def load_snap(name: str, data_dir: Path | str = DATA_DIR, dangling: str = "self_loop") -> Graph:
    """Load a SNAP graph listed in ``SNAP``, downloading it into ``data/`` on first use."""
    if name not in SNAP:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(SNAP)}")
    url, directed = SNAP[name]
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / Path(url).name
    if not path.exists():
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        tmp.rename(path)
    return load_edge_file(path, directed=directed, dangling=dangling)


# ---------------------------------------------------------------- generators
def paper_toy_graph() -> Graph:
    """The 4-node running example of the paper (Fig. 2).

    Edges: v1->v2, v2->{v3,v4}, v3->{v2,v4}, v4->{v2,v3}. With alpha = 0.8
    (paper alpha = 0.2), pi_{v1} = (0.2, 0.342857, 0.228571, 0.228571).
    """
    edges = [("v1", "v2"), ("v2", "v3"), ("v2", "v4"), ("v3", "v2"),
             ("v3", "v4"), ("v4", "v2"), ("v4", "v3")]
    return Graph.from_edges(edges, directed=True, nodes=["v1", "v2", "v3", "v4"])


def erdos_renyi(n: int, p: float, *, directed: bool = True, rng=None, dangling: str = "self_loop") -> Graph:
    """G(n, p) random graph without self-loops. The edge count is drawn first, then the endpoints."""
    rng = make_rng(rng)
    if directed:
        pairs = n * (n - 1)
        codes = rng.choice(pairs, size=rng.binomial(pairs, p), replace=False)
        u = codes // (n - 1)
        v = codes % (n - 1)
        v = v + (v >= u)  # skip the diagonal
    else:
        # Each unordered pair {u < v} is coded as u * n + v, and m distinct codes are drawn.
        m = rng.binomial(n * (n - 1) // 2, p)
        chosen = np.empty(0, dtype=np.int64)
        while chosen.size < m:
            a = rng.integers(0, n, size=2 * (m - chosen.size) + 16)
            b = rng.integers(0, n, size=a.size)
            keep = a != b
            code = np.minimum(a, b)[keep] * n + np.maximum(a, b)[keep]
            chosen = np.unique(np.concatenate([chosen, code]))
        chosen = rng.permutation(chosen)[:m]
        u, v = chosen // n, chosen % n
    return Graph(n, u, v, directed=directed, dangling=dangling)


def barabasi_albert(n: int, m: int, *, rng=None) -> Graph:
    """Undirected preferential-attachment graph with a scale-free degree distribution.

    Each new node attaches to m distinct existing nodes, chosen with
    probability proportional to their current degree.
    """
    rng = make_rng(rng)
    if not 1 <= m < n:
        raise ValueError("need 1 <= m < n")
    src, dst = [], []
    ends = list(range(m))  # every node appears here once per unit of degree (the seed nodes once each)
    for new in range(m, n):
        targets = set()
        while len(targets) < m:
            targets.add(ends[rng.integers(len(ends))])
        for t in targets:
            src.append(new)
            dst.append(t)
            ends.extend((new, t))
    return Graph(n, src, dst, directed=False)


def random_weighted_digraph(n: int, p: float, *, max_weight: int = 10, rng=None, dangling="self_loop") -> Graph:
    """Directed G(n, p) with integer weights in [1, max_weight].

    It stands in for the weighted passing networks of stage 2 (in tests).
    """
    rng = make_rng(rng)
    g = erdos_renyi(n, p, directed=True, rng=rng)
    rows = np.repeat(np.arange(n), g.out_degree)
    w = rng.integers(1, max_weight + 1, size=g.m)
    return Graph(n, rows, g.indices, w, directed=True, dangling=dangling)
