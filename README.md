# PPR via Variance-Reduced Monte Carlo — CSE 402 Project

A from-scratch Python reimplementation of Liao et al., *"Efficient Personalized PageRank Computation: The Power of Variance-Reduced Monte Carlo Approaches"* (SIGMOD 2023). It is written to be reused on football passing networks in stage 2.

No code from the authors' `pvr_code/` is used. Every algorithm (walks, alias tables, Wilson-style forest sampling, estimators, solvers) is implemented in `pprlib/`, using only NumPy/SciPy primitives: arrays, sparse matrix-vector products, and LU. numba is optional and only JIT-compiles two sequential loops.

## α convention (the proposal's)

`alpha` is the **damping factor**, the probability that the surfer *continues*:

```
r^(k+1) = α P r^(k) + (1 − α) s          π = α P π + (1 − α) σ
```

The paper uses the opposite convention, **paper α = 1 − our α**. So the paper's α = 0.2 is our α = 0.8, and its "hard" α = 0.01 is our α = 0.99. Every formula in `pprlib` is rewritten in our convention. For example, the paper's variance factor `(1−α)^{2K}` is our `α^{2K}`. Use `pprlib.theory.to_paper_alpha` to convert.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests          # Stage 1 and football correctness tests
```

## Library layout (`pprlib/`)

| module | contents |
|---|---|
| `graph.py` | `Graph`: CSR storage; directed or undirected; weighted or unweighted; labels. Column-stochastic `P` with `P[v,u] = w(u,v)/s_out(u)`. Dangling nodes use `"self_loop"` or `"uniform"`. Vectorised `step()` for weighted sampling. Constructors `from_edges`, `from_adjacency`. |
| `sources.py` | σ helpers: `one_hot`, `uniform`, `from_weights` (dict or array), `random_distribution` |
| `sampling.py` | Walker/Vose alias table with O(1) draws; RNG handling |
| `walks.py` | vectorised α-random walks; `walk_estimate(x)`, an unbiased estimate of `Π x` for signed x (Lemma 3.13) |
| `power.py` | `power_iteration` (history: step norms, true error, a-posteriori bound); `propagate` (K steps) |
| `exact.py` | ground truth: dense LU, sparse LU (with Sherman–Morrison for uniform dangling), or 1e-15 power iteration; `ppr_matrix` |
| `monte_carlo.py` | **MCW**, standard Monte Carlo (Alg. 3) |
| `variance_reduced.py` | **PW** (Alg. 4), **PPW** (Alg. 5), `residual` (Lemma 3.12) |
| `forests.py` | loop-erased α-walk forest sampler (Alg. 1); **MCF/MCFV** (Alg. 6), **PF/PFV** (Alg. 7), **PPF/PPFV** (Alg. 8) |
| `theory.py` | variance formulas (Lemmas 3.2, 3.7, 3.14, 4.2–4.4); K and T from the Chernoff bound (Lemma 3.11, Cor. 1); forest statistics |
| `metrics.py` | L1/L2/L∞, relative error (Def. 2.1), top-k precision, Kendall τ, Spearman ρ |
| `evaluation.py` | repeated independent trials, MSE (the variance, for unbiased methods), bias, pairwise ranking stability |
| `datasets.py` | SNAP downloader and loader, edge-list reader, generators (ER, BA, weighted digraph), the paper's toy graph |
| `api.py` | `compute_ppr(graph, source, alpha, method, **params)`, one entry point for all 11 methods |

## Using it (stage-2 style)

```python
import pprlib as pl

# weighted, directed passing network: (passer, receiver, number of passes)
passes = [("Messi", "Xavi", 12), ("Xavi", "Iniesta", 9), ("Iniesta", "Messi", 7), ("Xavi", "Messi", 5)]
g = pl.Graph.from_edges(passes, directed=True)            # weights are summed over duplicate passes

exact = pl.compute_ppr(g, "Xavi", alpha=0.85, method="exact")
ppw   = pl.compute_ppr(g, "Xavi", alpha=0.85, method="ppw", n_walks=5000, K=5, n_batches=3, rng=0)
print(ppw.top_k(3, g), pl.metrics.l1_error(ppw.estimate, exact.estimate))

# PageRank centrality (uniform σ) with the power method, including its convergence history
pr = pl.compute_ppr(g, None, alpha=0.85, method="power", tol=1e-10)
pr.history["diff"]            # ||r^(k+1) − r^(k)||_1 at each iteration
```

Methods: `exact, power, mcw, pw, ppw, mcf, mcfv, pf, pfv, ppf, ppfv`. The `*v` variants need an undirected graph. `pprlib.evaluation.run_trials` and `summarize` give error, variance, and ranking-stability statistics over many seeds. This is the machinery that stage 3 needs.

## Stage 1: reproducing the base paper

The experiments are in `experiments/stage1/`, and results (a CSV plus a PNG per figure) go to `results/stage1/`.

```bash
cd experiments/stage1
../../.venv/bin/python run_all.py --quick    # about 10 min, email-Eu-core + ca-GrQc
../../.venv/bin/python run_all.py            # full: + wiki-Vote + com-youtube (1.1M nodes, the paper's Youtube)
```

| script | what it checks | paper |
|---|---|---|
| `e1_power_convergence.py` | power method: error decays at rate α; a-priori and a-posteriori bounds; iterations needed vs α | §3.2, proposal |
| `e2_variance_validation.py` | empirical variance (T·MSE) vs Lemma 3.2 (MCW), Lemma 3.7 (PW, exact and bound), Lemma 3.14 (PPW per batch) | §3.1–3.3 |
| `e3_error_vs_samples.py` | L1 error and top-10 ranking accuracy vs walks T for MCW/PW/PPW | sample efficiency |
| `e4_error_vs_time.py` | accuracy vs time for all 10 methods, single-source and PageRank centrality | Figs. 4–5 |
| `e5_parameters.py` | effect of K and of B | Figs. 10–11 |
| `e6_walk_vs_forest.py` | α-walk vs spanning-forest estimators, Lemma 4.4 prediction | Table 2 |
| `e7_relative_error.py` | (ε, 1/n, 1/n) relative-error guarantee with theory-derived T and K | Figs. 7–8 |

Datasets: email-Eu-core (directed, 1,005 nodes), wiki-Vote (directed, 7,115 nodes, 1,005 dangling), ca-GrQc (undirected, 5,242 nodes), and com-youtube (undirected, 1.13M nodes, 2.99M edges, the paper's Youtube). They are downloaded from SNAP into `data/` on first use. We run the paper's two regimes, α = 0.8 and α = 0.99.

## Stage 2: football passing networks

The `football/` package loads local StatsBomb data, builds weighted directed
player networks, and computes exact centrality, single-source and positional
PPR using the unchanged `pprlib` API. The selected 50 matches are bundled in
`experiments/stage2/selected_matches.json`; event/lineup files remain in the
project's `data/` directory, alongside the Stage 1 benchmark archives.

```bash
.venv/bin/python -m pytest -q tests
cd experiments/stage2
../../.venv/bin/python run_all.py --quick  # 5 matches, all four competitions
../../.venv/bin/python run_all.py          # all 50 selected matches
```

Results go to `results/stage2/`, including reloadable networks, exact ground
truth for Stage 3, rankings and PNG figures. See
[the Stage 2 guide](experiments/stage2/README.md) for modelling choices,
filter options, output schemas and reproducible usage.

## Data and Git

Local football data lives in `data/events/`, `data/lineups/`, `data/matches/`
and `data/competitions.json`. Keep the code, tests, requirements, documentation
and `experiments/stage2/selected_matches.json` in Git. That selection manifest
records the 50 matches needed to reproduce Stage 2 without uploading all raw data.

Raw football JSON, benchmark archives, generated `results/`, `.venv/` and caches
are ignored. Existing tracked benchmark archives remain tracked: `.gitignore`
only prevents new untracked files from being added. To stop tracking those
archives while keeping local copies, run `git rm --cached -- data/*.txt.gz`
and commit the resulting index changes. This does not remove old Git history.

See [data setup and attribution](data/README.md) for the directory layout and
the source of the raw files. Selected plots or result summaries can be shared
separately when needed; the entire generated results directory need not be committed.

## Stage 3: controlled numerical comparison

Stage 3 compares power iteration, MCW, PW, PPW and directed forest estimators
with repeated trials, explicit sample/propagation budgets, and saved per-trial
estimates. It reloads Stage 2 networks and truth, so raw StatsBomb data is not
needed after Stage 2 output has been produced.

```bash
# Smoke validation only: all available competitions, three trials
.venv/bin/python experiments/stage3/run_all.py --quick \
  --stage2-results results/stage2 --output results/stage3_quick

# Full study: all 100 team-match networks from a full Stage 2 manifest
.venv/bin/python experiments/stage3/run_all.py \
  --stage2-results results/stage2 --output results/stage3
```

The full command keeps 30 stochastic repetitions and the configured budgets;
use a separate output path for smaller pilots. See the
[Stage 3 guide](experiments/stage3/README.md) and the generated manifest for
coverage and resolved settings.
