"""Shared Stage 2 paths, selected local data, and the Stage 1 plotting palette."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Matplotlib must be able to cache fonts in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

from experiments.stage1.common import PALETTE, SURFACE, INK, GRID, mpl, alpha_colors  # noqa: E402
from football import statsbomb  # noqa: E402
from football.networks import PassingNetwork, build_network, load_saved  # noqa: E402


def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--selection", type=Path, default=Path(__file__).with_name("selected_matches.json"))
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "stage2")
    parser.add_argument("--quick", action="store_true", help="five matches spanning all four selected competitions")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.data = statsbomb.data_root(args.data)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    return args


def selected_matches(args) -> list[dict]:
    selection = json.loads(args.selection.read_text(encoding="utf-8"))["matches"]
    if args.quick:
        chosen = {}
        for row in selection:
            chosen.setdefault((row["competition_id"], row["season_id"]), row)
        rows = list(chosen.values())
        first = rows[0]
        extra = next((m for m in selection if m["match_id"] != first["match_id"]
                      and (m["competition_id"], m["season_id"]) == (first["competition_id"], first["season_id"])
                      and set(m["team_ids"]) & set(first["team_ids"])), None)
        if extra is not None:
            rows.append(extra)
        selection = rows
    if not selection or len({m["match_id"] for m in selection}) != len(selection):
        raise ValueError("selection must contain distinct matches")
    return selection


def match_networks(args, selected: list[dict]):
    """Yield two validated full-match networks for each selected match."""
    for match in selected:
        mid = int(match["match_id"])
        events = statsbomb.load_events(args.data, mid)
        lineups = statsbomb.load_lineups(args.data, mid)
        metadata = statsbomb.load_matches(args.data, match["competition_id"], match["season_id"])
        row = metadata.loc[metadata.match_id.eq(mid)]
        if len(row) != 1:
            raise ValueError(f"missing/ambiguous match metadata: {mid}")
        team_ids = {int(row.iloc[0].home_team_home_team_id), int(row.iloc[0].away_team_away_team_id)}
        if team_ids != set(match["team_ids"]) or team_ids != set(lineups.team_id):
            raise ValueError(f"team IDs disagree for match {mid}")
        for team in sorted(team_ids):
            network = build_network(events, team, lineups)
            network.metadata.update(competition_id=match["competition_id"], competition=match["competition"],
                                    season_id=match["season_id"], season=match["season"], match_date=match["date"])
            yield f"m{mid}_t{team}", network


def read_network(path: Path) -> PassingNetwork:
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    players = pd.DataFrame(metadata.pop("players"))
    return PassingNetwork(load_saved(path), players,
                          pd.read_csv(path, keep_default_na=False), metadata)


def prepared_networks(args):
    manifest = args.output / "run_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError("run e1_build_networks.py first (or run_all.py)")
    run = json.loads(manifest.read_text())
    expected = [m["match_id"] for m in selected_matches(args)]
    if run["match_ids"] != expected or run["data_root"] != str(args.data):
        raise ValueError("prepared networks use a different selection/data path; rerun e1_build_networks.py")
    for item in run["networks"]:
        yield item["network_id"], read_network(args.output / item["path"])


def save_csv(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_matrix(matrix: np.ndarray, labels: list, path: Path, row_name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=pd.Index(labels, name=row_name), columns=labels).to_csv(path)


def save_figure(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    mpl().close(fig)
