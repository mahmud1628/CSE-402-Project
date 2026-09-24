# Stage 3 — repeated numerical comparison

Stage 3 measures deterministic power iteration and the MCW, PW, PPW, MCF,
PF and PPF estimators on saved directed Stage 2 passing networks. It is an
experiment layer; it does not reimplement PPR algorithms.

## Run

From the repository root, create Stage 2 artifacts if absent, then run:

```bash
.venv/bin/python experiments/stage2/run_all.py --output results/stage2
.venv/bin/python experiments/stage3/run_all.py --quick --stage2-results results/stage2 --output results/stage3_quick
.venv/bin/python experiments/stage3/run_all.py --stage2-results results/stage2 --output results/stage3
```

Quick mode is smoke validation only (three trials, 300/1,000 walks). Full mode
uses 30 trials and 1,000/3,000/10,000/30,000 walk budgets. Use `--trials` and
`--network-limit` only for a documented pilot, with a separate output folder.
Compatible trial rows resume automatically; a changed resolved configuration
requires a new output directory.

## What is measured

- E1 saves exact/power convergence histories and 1e-4/1e-6/1e-8 targets.
- E2 saves repeated MCW/PW/PPW errors and rankings. The primary cohort is
  match networks only with uniform sources; the metadata-only secondary cohort
  uses min/median/max size networks per competition plus deterministic role
  sources where available.
- E3 separates MSE, finite-sample total variance, bias and MCW/PW theory.
- E4 varies PW K and PPW B, recording K per batch, actual costs and residuals.
- E5 compares directed forest methods and reports only tested target crossings.

`PPRResult.time` is the algorithm-only timing boundary: loading, exact truth,
scoring, output and plotting are excluded. Progressive estimates are never
clipped or renormalized; nonfinite estimates remain recorded failures.

## Outputs

`run_manifest.json` records config, platform and node order.
`tables/trials.csv` has one row per independent trial and points to raw arrays
under `estimates/`. `tables/summaries.csv` is the per-network/source/config
summary. Stage 2 truth is explicitly reindexed to graph labels; influence
columns are sources and rows are destinations. If retained truth is absent,
exact PPR is recomputed from the saved network, so raw event files are not
needed after Stage 2.
