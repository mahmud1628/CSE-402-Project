"""Shared loading, reproducibility, recording and plotting helpers for Stage 3.

This module deliberately keeps experiments outside :mod:`pprlib`.  It can use
saved Stage 2 networks and ground truth after a checkout is relocated, and it
also exposes a tiny synthetic Stage 2 fixture for tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import pprlib as pl  # noqa: E402
from football.networks import load_saved, save_network  # noqa: E402
from football.networks import PassingNetwork  # noqa: E402
from experiments.stage1.common import METHOD_STYLE, mpl  # noqa: E402

ALPHAS = (0.8, 0.85, 0.99)
WALK_METHODS = ("mcw", "pw", "ppw")
FOREST_METHODS = ("mcf", "pf", "ppf")


@dataclass
class Network:
    network_id: str
    graph: pl.Graph
    metadata: dict
    kind: str
    path: Path

    @property
    def match_id(self):
        return self.metadata.get("match_id")

    @property
    def competition(self):
        return str(self.metadata.get("competition", self.metadata.get("competition_id", "unknown")))

    @property
    def players(self) -> pd.DataFrame:
        return pd.DataFrame(self.metadata["players"])


def parse_args(description: str):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--stage2-results", type=Path, default=ROOT / "results" / "stage2")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "stage3")
    p.add_argument("--seed", type=int, default=202603)
    p.add_argument("--trials", type=int, default=None, help="stochastic repetitions (default: 30 full, 3 quick)")
    p.add_argument("--quick", action="store_true", help="smoke validation only; never publication evidence")
    p.add_argument("--network-limit", type=int, default=None, help="deterministic cap, useful for pilots")
    p.add_argument("--resume", action="store_true", help="reuse compatible trial records already in --output")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)
    return p.parse_args()


def config(args) -> dict:
    """The frozen, editable grids.  Full-mode defaults follow the Stage 3 brief."""
    quick = bool(args.quick)
    return {
        "schema_version": 1,
        "quick": quick,
        "root_seed": int(args.seed),
        "trials": int(args.trials if args.trials is not None else (3 if quick else 30)),
        "alphas": list(ALPHAS),
        "walk_budgets": [300, 1000] if quick else [1000, 3000, 10000, 30000],
        "forest_budgets": [20, 50, 100] if quick else [30, 100, 300, 1000],
        "pw_K": 3 if quick else 6,
        "ppw_K": 1 if quick else 2,
        "ppw_B": 3,
        "K_grid": [0, 1, 2, 5, 10] if quick else [0, 1, 2, 5, 10, 20, 50],
        "B_grid": [1, 2, 3, 5] if quick else [1, 2, 3, 5, 10],
        "network_limit": args.network_limit,
        "timing": "algorithm-only PPRResult.time; excludes loading, truth, metrics, I/O and plotting",
    }


def stable_seed(root_seed: int, *parts: object) -> int:
    """A process/order-independent child seed; never use Python's random hash."""
    raw = "\x1f".join(map(str, (root_seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_networks(stage2: Path) -> list[Network]:
    """Load only manifest-listed Stage 2 networks, validating sidecars and labels."""
    stage2 = Path(stage2).resolve()
    manifest_path = stage2 / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {manifest_path}; run Stage 2 e1/e2 first")
    manifest = _json(manifest_path)
    out = []
    for item in manifest.get("networks", []):
        path = stage2 / item["path"]
        sidecar = path.with_suffix(".json")
        if not path.exists() or not sidecar.exists():
            raise FileNotFoundError(f"manifest network is missing: {path}")
        meta = _json(sidecar)
        labels = [p["label"] for p in meta.get("players", [])]
        if len(labels) != len(set(labels)) or not labels:
            raise ValueError(f"invalid player labels in {sidecar}")
        graph = load_saved(path)
        if graph.labels != labels:
            raise ValueError(f"saved graph label order differs from its sidecar: {path}")
        out.append(Network(item["network_id"], graph, meta, item.get("kind", "unknown"), path))
    if not out:
        raise ValueError("Stage 2 manifest contains no networks")
    return out


def match_networks(networks: Iterable[Network]) -> list[Network]:
    return sorted((n for n in networks if n.kind == "match"), key=lambda n: n.network_id)


def choose_secondary(networks: list[Network], quick: bool = False) -> list[Network]:
    """Metadata-only min/median/max size selection, three match networks per competition."""
    chosen = []
    for competition in sorted({n.competition for n in networks}):
        group = sorted((n for n in networks if n.competition == competition),
                       key=lambda n: (n.graph.n, n.graph.m, n.network_id))
        if not group:
            continue
        count = 1 if quick else min(3, len(group))
        indices = np.linspace(0, len(group) - 1, count).round().astype(int)
        picked = [group[i] for i in indices]
        # In quick mode choose a dangling example within its own competition
        # when available; never trade away coverage of another competition.
        dangling = [n for n in group if n.graph.dangling_nodes.size]
        if dangling and not any(n.graph.dangling_nodes.size for n in picked):
            picked[-1] = dangling[0]
        chosen.extend(picked)
    # Preferably include a dangling case without looking at any estimator error.
    if chosen and not any(n.graph.dangling_nodes.size for n in chosen):
        candidates = [n for n in networks if n.graph.dangling_nodes.size and n not in chosen]
        if candidates:
            candidate = sorted(candidates, key=lambda n: (n.competition, n.network_id))[0]
            # Replace only a network from the same competition, preserving the
            # required four-competition coverage in quick mode.
            for i, old in enumerate(chosen):
                if old.competition == candidate.competition:
                    chosen[i] = candidate
                    break
    return sorted({n.network_id: n for n in chosen}.values(), key=lambda n: n.network_id)


def source_specs(network: Network, secondary: bool) -> list[tuple[str, np.ndarray]]:
    g = network.graph
    specs = [("uniform", pl.uniform(g))]
    if not secondary:
        return specs
    people = network.players.copy()
    people["position"] = people.position.fillna("")
    for name, mask in (("goalkeeper", people.position.str.contains("Goalkeeper", regex=False)),
                       ("defender", people.position.str.contains("Back", regex=False))):
        choices = people.loc[mask].sort_values("player_id")
        if len(choices):
            specs.append((f"{name}:{int(choices.iloc[0].player_id)}", pl.one_hot(g, choices.iloc[0].label)))
    defenders = people.loc[people.position.str.contains("Back", regex=False)]
    if len(defenders):
        specs.append(("defenders", pl.sources.from_weights(g, {x: 1.0 for x in defenders.label})))
    return specs


def truth_for(stage2: Path, network: Network, alpha: float, source_id: str, sigma: np.ndarray) -> np.ndarray:
    """Read and align Stage 2 truth if present; otherwise fail only after checking it."""
    folder = Path(stage2) / "ground_truth"
    vector_path = folder / f"{network.network_id}_a{alpha:g}_vectors.csv"
    influence_path = folder / f"{network.network_id}_a{alpha:g}_influence.csv"
    labels = network.graph.labels
    try:
        if source_id in {"uniform", "defenders"} and vector_path.exists():
            frame = pd.read_csv(vector_path, index_col="player")
            column = "defenders" if source_id == "defenders" else "uniform"
            if column in frame:
                if set(frame.index) != set(labels):
                    raise ValueError(f"truth labels do not match {network.network_id}")
                return frame.loc[labels, column].to_numpy(float)
        if influence_path.exists():
            frame = pd.read_csv(influence_path, index_col="destination")
            if set(frame.index) != set(labels) or set(frame.columns) != set(labels):
                raise ValueError(f"influence labels/orientation do not match {network.network_id}")
            return frame.loc[labels, labels].to_numpy(float) @ sigma
    except KeyError as exc:
        raise ValueError(f"unreadable/alignment-invalid truth for {network.network_id}") from exc
    # Saved networks remain useful even if Stage 2's truth subset was not retained.
    return pl.exact_ppr(network.graph, sigma, alpha)


def output_paths(output: Path) -> dict[str, Path]:
    output = Path(output).resolve()
    paths = {"root": output, "estimates": output / "estimates", "figures": output / "figures",
             "tables": output / "tables", "histories": output / "histories"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def run_manifest(args, cfg: dict, networks: list[Network]) -> None:
    paths = output_paths(args.output)
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    path = paths["root"] / "run_manifest.json"
    if path.exists():
        previous = _json(path)
        if previous.get("config_fingerprint") != fingerprint:
            raise ValueError("output contains an incompatible Stage 3 configuration; use another --output")
        return
    payload = dict(schema_version=1, config=cfg, config_fingerprint=fingerprint,
                   stage2_results=str(Path(args.stage2_results).resolve()),
                   platform=platform.platform(), python=sys.version, numpy=np.__version__,
                   networks=[dict(network_id=n.network_id, kind=n.kind, labels=n.graph.labels,
                                  n_players=n.graph.n, n_edges=n.graph.m, match_id=n.match_id,
                                  competition=n.competition, dangling_players=int(n.graph.dangling_nodes.size)) for n in networks])
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


TRIAL_COLUMNS = ["experiment", "cohort", "network_id", "match_id", "competition", "source_id", "alpha", "method",
                 "trial", "seed", "T", "F", "K", "B", "n_walks", "n_forests", "n_walk_steps", "n_power_iterations",
                 "seconds", "l1", "l2", "sq_l2", "tau", "topk_overlap", "mass", "minimum", "status",
                 "residual_l1", "batch_sizes", "estimate_path"]


def _trial_file(output: Path) -> Path:
    return Path(output) / "tables" / "trials.csv"


def read_trials(output: Path) -> pd.DataFrame:
    p = _trial_file(output)
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=TRIAL_COLUMNS)


def _save_trials(output: Path, frame: pd.DataFrame):
    frame = frame.reindex(columns=TRIAL_COLUMNS)
    frame.sort_values(["experiment", "network_id", "source_id", "alpha", "method", "trial"], inplace=True)
    frame.to_csv(_trial_file(output), index=False)


def _run_method(g, sigma, alpha, method, seed, *, T=0, F=0, K=0, B=1, residuals=False):
    rng = np.random.default_rng(seed)
    if method == "ppw" and residuals:
        return pl.ppw(g, sigma, alpha, T, K, B, rng, record_residuals=True)
    kwargs = {"rng": rng, "K": K, "n_batches": B}
    if method in WALK_METHODS:
        kwargs["n_walks"] = T
    else:
        kwargs["n_forests"] = F
    return pl.compute_ppr(g, sigma, alpha, method, **kwargs)


def trial_key(experiment, network, source_id, alpha, method, trial, T, F, K, B):
    return (experiment, network.network_id, source_id, float(alpha), method, int(trial), int(T), int(F), int(K), int(B))


def execute_trials(args, cfg, *, experiment: str, cohort: str, network: Network, source_id: str,
                   sigma: np.ndarray, alpha: float, method: str, T: int = 0, F: int = 0,
                   K: int = 0, B: int = 1, residuals: bool = False) -> pd.DataFrame:
    """Run/cache deterministic child trials, including every raw estimate as an .npy array."""
    paths = output_paths(args.output)
    current = read_trials(args.output)
    truth = truth_for(args.stage2_results, network, alpha, source_id, sigma)
    rows = []
    existing = {tuple(x) for x in current[["experiment", "network_id", "source_id", "alpha", "method", "trial", "T", "F", "K", "B"]].itertuples(index=False, name=None)} if len(current) else set()
    for trial in range(cfg["trials"]):
        key = trial_key(experiment, network, source_id, alpha, method, trial, T, F, K, B)
        if key in existing:
            continue
        seed = stable_seed(cfg["root_seed"], experiment, network.network_id, source_id, alpha, method, trial, T, F, K, B)
        result = _run_method(network.graph, sigma, alpha, method, seed, T=T, F=F, K=K, B=B, residuals=residuals)
        estimate_rel = f"estimates/{hashlib.sha256(repr(key).encode()).hexdigest()}.npy"
        np.save(paths["root"] / estimate_rel, result.estimate)
        finite = bool(np.all(np.isfinite(result.estimate)))
        hist = result.history
        rows.append(dict(experiment=experiment, cohort=cohort, network_id=network.network_id, match_id=network.match_id,
                         competition=network.competition, source_id=source_id, alpha=alpha, method=method, trial=trial,
                         seed=seed, T=T, F=F, K=K, B=B, n_walks=result.n_walks, n_forests=result.n_forests,
                         n_walk_steps=result.n_walk_steps, n_power_iterations=result.n_power_iterations, seconds=result.time,
                         l1=pl.metrics.l1_error(result.estimate, truth) if finite else np.nan,
                         l2=pl.metrics.l2_error(result.estimate, truth) if finite else np.nan,
                         sq_l2=pl.metrics.sq_l2_error(result.estimate, truth) if finite else np.nan,
                         tau=pl.metrics.kendall_tau(result.estimate, truth) if finite else np.nan,
                         topk_overlap=pl.metrics.top_k_precision(result.estimate, truth, min(5, network.graph.n)) if finite else np.nan,
                         mass=float(result.estimate.sum()) if finite else np.nan,
                         minimum=float(result.estimate.min()) if finite else np.nan,
                         status="ok" if finite else "nonfinite",
                         residual_l1=json.dumps(np.asarray(hist.get("residual_l1", [])).tolist()),
                         batch_sizes=json.dumps(hist.get("batch_sizes", [])), estimate_path=estimate_rel))
    if rows:
        combined = pd.DataFrame(rows) if current.empty else pd.concat([current, pd.DataFrame(rows)], ignore_index=True)
        _save_trials(args.output, combined)
    all_rows = read_trials(args.output)
    # Bracket access is intentional: DataFrame attributes such as ``T`` and
    # ``method`` are not reliably the columns of the same name.
    mask = ((all_rows["experiment"] == experiment) & (all_rows["network_id"] == network.network_id) &
            (all_rows["source_id"] == source_id) & (all_rows["alpha"] == alpha) & (all_rows["method"] == method) &
            (all_rows["T"] == T) & (all_rows["F"] == F) & (all_rows["K"] == K) & (all_rows["B"] == B))
    return all_rows.loc[mask].copy()


def summarize(output: Path, stage2: Path, networks: dict[str, Network], trials: pd.DataFrame) -> pd.DataFrame:
    """Per-configuration summaries; finite-sample variance, MSE and bias stay distinct."""
    rows = []
    group_cols = ["experiment", "cohort", "network_id", "match_id", "competition", "source_id", "alpha", "method", "T", "F", "K", "B"]
    for key, d in trials.groupby(group_cols, dropna=False):
        d = d[d.status == "ok"]
        out = dict(zip(group_cols, key), n_trials=int(len(d)), failures=int((trials.loc[d.index, "status"] != "ok").sum()))
        for metric in ("l1", "l2", "sq_l2", "tau", "topk_overlap", "seconds", "mass", "minimum", "n_walk_steps", "n_power_iterations"):
            out[metric] = float(d[metric].mean()) if len(d) else np.nan
            out[metric + "_sd"] = float(d[metric].std(ddof=1)) if len(d) > 1 else np.nan
        if len(d):
            network = networks[out["network_id"]]
            sigma = dict(source_specs(network, True)).get(out["source_id"], pl.uniform(network.graph))
            truth = truth_for(stage2, network, out["alpha"], out["source_id"], sigma)
            X = np.stack([np.load(Path(output) / p) for p in d.estimate_path])
            out["total_sample_variance"] = float(X.var(axis=0, ddof=1).sum()) if len(X) > 1 else np.nan
            out["mse"] = float(((X - truth) ** 2).sum(axis=1).mean())
            out["bias_l1"] = float(np.abs(X.mean(axis=0) - truth).sum())
            if len(X) > 1:
                pairs = [pl.metrics.top_k_precision(X[i], X[j], min(5, network.graph.n)) for i in range(len(X)) for j in range(i)]
                out["pairwise_topk_overlap"] = float(np.mean(pairs))
            else:
                out["pairwise_topk_overlap"] = np.nan
        rows.append(out)
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(output) / "tables" / "summaries.csv", index=False)
    return frame


def save_figure(fig, output: Path, name: str):
    path = Path(output) / "figures" / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    mpl().close(fig)
    return path


def write_report(output: Path, summary: pd.DataFrame) -> Path:
    """A deliberately conservative report: every number is read from saved trials."""
    path = Path(output) / "RESULTS.md"
    d = summary[summary["status"].eq("ok")] if "status" in summary else summary
    lines = ["# Stage 3 results", "", "This report is generated from the recorded trial CSV; quick and pilot runs are not full-study evidence.", ""]
    if d.empty:
        lines.append("No successful stochastic trial records are available.")
    else:
        lines += [f"Coverage: {d.network_id.nunique()} networks, {d.source_id.nunique()} source types, {len(d)} configurations.", "",
                  "## Recorded aggregate extrema", ""]
        for metric, label in (("l1", "mean L1 error"), ("seconds", "algorithm-only seconds"), ("topk_overlap", "top-k overlap")):
            if metric in d:
                best = d.loc[d[metric].idxmin() if metric != "topk_overlap" else d[metric].idxmax()]
                lines.append(f"- Best observed {label}: `{best[metric]:.4g}` ({best['method']}, {best['network_id']}, alpha={best['alpha']:g}, source={best['source_id']}).")
        lines += ["", "Interpret these as configuration-specific observations, not universal claims. See `tables/summaries.csv`, `tables/variance_validation.csv`, and `tables/sample_to_target.csv` for uncertainty, theory checks and unreached targets."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_synthetic_stage2(path: Path):
    """Create four tiny, fully aligned Stage 2 artifacts for an offline end-to-end test."""
    path = Path(path); (path / "networks").mkdir(parents=True, exist_ok=True); (path / "ground_truth").mkdir(exist_ok=True)
    inventory = []
    for i, comp in enumerate(("La Liga", "Serie A", "Indian Super League", "Serie A Women"), 1):
        labels = [f"GK {i}", f"Back {i}", f"Mid {i}", f"Dangling {i}"]
        g = pl.Graph.from_edges([(labels[0], labels[1], 2), (labels[1], labels[2], 3), (labels[2], labels[0], 1)], nodes=labels, weighted=True, directed=True)
        people = pd.DataFrame(dict(player_id=[i * 10 + j for j in range(4)], player=labels,
                                   label=labels, position=["Goalkeeper", "Center Back", "Midfielder", "Forward"],
                                   starter=[True] * 4, x=[1.0] * 4, y=[1.0] * 4, location_count=[1] * 4))
        meta = dict(schema_version=1, match_id=i, match_ids=[i], team=f"Team {i}", team_id=i,
                    competition=comp, competition_id=i, season_id=1, dangling="self_loop", players=people.to_dict("records"))
        net = PassingNetwork(g, people, pd.DataFrame([(labels[0], labels[1], 2), (labels[1], labels[2], 3), (labels[2], labels[0], 1)], columns=["passer", "recipient", "passes"]), meta)
        saved = save_network(net, path / "networks" / f"synthetic_{i}.csv")
        inventory.append(dict(network_id=f"synthetic_{i}", kind="match", path=str(saved.relative_to(path))))
        for alpha in ALPHAS:
            M = pl.ppr_matrix(g, alpha)
            pd.DataFrame(M, index=pd.Index(labels, name="destination"), columns=labels).to_csv(path / "ground_truth" / f"synthetic_{i}_a{alpha:g}_influence.csv")
            pd.DataFrame({"uniform": M @ pl.uniform(g)}, index=pd.Index(labels, name="player")).to_csv(path / "ground_truth" / f"synthetic_{i}_a{alpha:g}_vectors.csv")
    (path / "run_manifest.json").write_text(json.dumps(dict(schema_version=1, networks=inventory), indent=2) + "\n")
