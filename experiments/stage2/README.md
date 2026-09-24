# Stage 2 — football passing networks

Stage 2 builds on the unchanged `pprlib` library. It computes structural passing
influence, not player quality or match predictions. Alpha is the **continuation
probability**: `pi = alpha P pi + (1-alpha) sigma`. Thus our 0.8 and 0.99 are the
paper's 0.2 and 0.01; the standard football value here is 0.85.

## Setup and run

From `CSE-402-Project/`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
cd experiments/stage2
../../.venv/bin/python run_all.py --quick
../../.venv/bin/python run_all.py
```

In the supplied workspace the environment was created with
`python3 -m venv --system-site-packages .venv`, reusing the already installed
NumPy, SciPy, pandas, matplotlib and pytest without downloads. No new dependencies
were needed. A normal isolated venv with the requirements above also works.

The default data root is the project's `data/` directory, and the default selection
is the bundled `selected_matches.json`. Both can be overridden:

```bash
../../.venv/bin/python run_all.py --data /path/to/CSE-402-Project/data/events \
  --selection /path/to/selected_matches.json --output /path/to/results --seed 0
```

An events directory, data directory or a repository containing `data/` is accepted.
All data access is local. Missing or malformed selected files fail explicitly;
the runner never downloads or silently substitutes matches. Metadata uses the
standard `matches/<competition_id>/<season_id>.json` layout.

The full run uses 50 matches: 20 La Liga 2015/16, 20 Serie A 2015/16 (matchweeks
5 and 25), five Indian Super League 2021/22 (week 5), and five Serie A Women
2023/24 (week 1). This is a deliberately varied convenience subset, not a random
sample. The quick run takes one match from each competition plus a second match
sharing a team with the first La Liga match, so it also exercises aggregation.
The full run builds 100 team-match networks and 40 team-season aggregates;
the quick run builds 10 plus one. Both use all five alpha values.

Each script can run separately with the same arguments, in this order:

1. `e1_build_networks.py`: build, save and reload-verify networks and W/P matrices.
2. `e2_rankings.py`: exact centrality, positional PPR, all-source matrices and
   power verification; save sensitivity against alpha 0.85.
3. `e3_figures.py`: two team passing maps and two influence heatmaps from the
   first selected match, plus an alpha-sensitivity figure for all match networks.
4. `e4_mc_smoke.py`: six directed MC methods on both teams of the first match.

Outputs default to `results/stage2/`. The current `run_manifest.json` is the
authoritative network list. A run overwrites its named outputs, but does not
delete unrelated/older files; downstream scripts use the manifest, not a glob.
Use separate `--output` directories to keep quick and full runs side by side.

## Modelling choices

- A directed edge is passer to recipient, weighted by the count of completed
  passes. Completion means **absence** of the `pass.outcome` key; even a present
  null outcome is excluded. Missing passer/recipient IDs are dropped and counted
  in diagnostics. Own-goal event types do not generate passing edges.
- Player IDs determine identity, including across matches. Labels use full names;
  equal names belonging to different IDs receive an ID suffix. All matrices and
  vectors use the saved player order, sorted by ID.
- Nodes include starters and incoming substitutes who participate in the chosen
  event interval, even with no completed passes or no observed locations. Unused
  bench players are excluded. Starting XI/substitution events establish entry and
  exit; recorded red/second-yellow cards also establish exit. Lineup metadata
  supplies names and first positions, with event information as fallback.
  Disciplinary events alone do not establish participation: the selected data
  includes a yellow card for an unused bench player (Astrid Gilardi, Como W).
- Full-match networks include periods 1–4 (extra time if present), excluding
  shootouts. Set pieces are included by default. `exclude_set_pieces=True`
  excludes Throw-in, Corner, Free Kick, Goal Kick and Kick Off.
- The default dangling rule is `self_loop`, consistent with the existing library
  default. It can increase influence for a player with no outgoing passes.
  `dangling="uniform"` is available for a different explicit modelling choice.
  The added transition is in P, not a fabricated pass in W.
- `time_window=(start,end)` uses half-open StatsBomb minute+second/60 values.
  These clocks overlap around halftime stoppage time; for a stable starting XI,
  prefer `before_first_substitution=True`, which cuts by the team's first
  substitution **event index**. This includes earlier first-half stoppage events
  even if their minute exceeds the second-half substitution minute. It does not
  independently truncate at dismissals or tactical changes.
- `include_substitutes=False` keeps only starters, dropping passes involving
  incoming players. It does not automatically shorten the match.
- `min_pass_count` removes edges below the threshold after counting duplicates,
  retaining player nodes. Aggregation requires unthresholded component networks
  (`min_pass_count=1`), pools counts by ID, then applies its own threshold.
- Aggregates are confined to the same team, competition and season by the runner.
  The API checks compatible team/filter settings and rejects duplicate matches.
  Counts are pooled, not normalized per match or by minutes played.
- Plot coordinates are means of all recorded player event locations in the
  window, not tracking positions. They retain StatsBomb's 120×80 coordinates.
  A player with no locations is displayed in a labelled off-pitch strip. Node
  area is proportional to uniform-source PageRank; directed arrow width is
  proportional to pass count. Colours come from Stage 1's plotting helpers.
- Positional sources spread mass equally over players whose initial recorded
  role is Goalkeeper or contains Back. Aggregates use the union of initial match
  roles. Positional changes within a match are not modelled dynamically.

## Outputs and Stage 3 hand-off

| Output | Contents |
|---|---|
| `run_manifest.json`, `selected_matches.csv` | Current run's data root, match selection, seed and saved network list |
| `network_summary.csv` | Players, edges, pass counts, dangling nodes, missing endpoints and locations |
| `networks/<id>.csv` | Directed `passer,recipient,passes` edge list |
| `networks/<id>.json` | Ordered players/IDs/roles/locations; match IDs, team, filters, time window and dangling rule; aggregate component metadata |
| `matrices/<id>_W.csv` | Row passer, column recipient; raw pass weights |
| `matrices/<id>_P.csv` | Row destination, column source; column-stochastic transitions |
| `rankings.csv` | Exact uniform/goalkeeper/defender rankings at 0.5, 0.8, 0.85, 0.9, 0.99 |
| `centrality_rankings.csv` | Uniform-personalization subset of the ranking table |
| `power_checks.csv` | Exact/power L1 agreement, mass and iteration count, including every single-player source |
| `alpha_sensitivity.csv` | Kendall tau-b of exact centrality scores vs alpha 0.85, in aligned player order |
| `ground_truth/<id>_sources.csv` | Explicit uniform and positional source vectors |
| `ground_truth/<id>_a<alpha>_vectors.csv` | Exact centrality and positional PPR vectors |
| `ground_truth/<id>_a<alpha>_influence.csv` | Full exact PPR matrix: column u is PPR **from** player u |
| `figures/*.png` | Passing maps, influence heatmaps and alpha sensitivity |
| `mc_smoke.csv` | One run per method/team/alpha, budgets, seed, error, mass and runtime |

Exact results solve `(I-alpha P) pi = (1-alpha) sigma`. Power uses a step tolerance
of 1e-13 and must converge with L1 disagreement <=1e-10. Ground truth is computed
at all five alphas, including every one-hot source, for each saved network.

For example, from the project root:

```python
import pandas as pd
import pprlib as pl
from football.networks import load_saved

g = load_saved("results/stage2/networks/m3825598_t212.csv")
truth = pd.read_csv(
    "results/stage2/ground_truth/m3825598_t212_a0.85_vectors.csv",
    index_col="player",
).loc[g.labels, "uniform"].to_numpy()
# Ready for Stage 3: pl.evaluation.run_trials / summarize / ranking_stability.
```

Library use with filters:

```python
from football.statsbomb import load_events, load_lineups
from football.networks import build_network, save_network

events = load_events("data", 3825598)
lineups = load_lineups("data", 3825598)
network = build_network(events, "Atlético Madrid", lineups,
                        before_first_substitution=True, dangling="uniform")
save_network(network, "results/stage2/custom/atletico_before_sub.csv")
```

The smoke test runs MCW, PW, PPW, MCF, PF and PPF on directed weighted networks
at alpha 0.8, 0.85 and 0.99, with fixed `rng=seed`. Walk budgets are explicitly
30,000 (quick) or 90,000 (full); forest budgets are 300 or 900; progressive methods
use three batches. Propagated methods use `ceil(log(1e-4)/log(alpha))` steps per
propagation phase. This conservative K prevents high-alpha residual growth in a
smoke test; it is not a tuned fair-budget comparison. Estimates are recorded
without clipping/renormalization. Degree-weighted `*v` variants require
undirected graphs and are intentionally outside these directed analyses.
No Stage 3 repeated-trial variance/convergence study is performed here.

## Tests and attribution

`tests/test_football.py` uses only synthetic in-file events/lineups. It covers
completion semantics, missing recipients, duplicate weights, no-pass players,
unused substitutes, windows/stoppage time, filters, shootout exclusion, both
dangling rules, exact/power agreement, a hand-derived three-player solution,
aggregation, name collisions, group sources and CSV/JSON round trips.

Data source: [StatsBomb Open Data](https://github.com/hudl/open-data). Follow
the dataset's attribution requirements (including its logo for published
research) in the supplied `data/STATSBOMB_README.md`. The dataset-selection files
in the workspace record lineup download provenance and checksums.
