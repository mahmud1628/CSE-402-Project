"""Completed passes define W[u,v]; P[v,u] = W[u,v] / out_strength(u).

The default self-loop rule keeps a walk at a player with no outgoing passes,
matching the existing library default and keeping every solver consistent.
It can increase that player's centrality; ``dangling='uniform'`` is available
for a different explicit modelling choice. No artificial passes are added to W.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pprlib import Graph
from .statsbomb import pass_table

SET_PIECES = {"Throw-in", "Corner", "Free Kick", "Goal Kick", "Kick Off"}


@dataclass
class PassingNetwork:
    """A graph plus the player identities, plotting positions and provenance."""

    graph: Graph
    players: pd.DataFrame
    edges: pd.DataFrame
    metadata: dict

    @property
    def W(self) -> np.ndarray:
        return self.graph.A.toarray()

    @property
    def P(self) -> np.ndarray:
        return self.graph.dense_P()


def _team_events(events: pd.DataFrame, team) -> pd.DataFrame:
    mask = events["team"].eq(team) if isinstance(team, str) else events["team_id"].eq(team)
    rows = events.loc[mask].copy()
    if rows.empty or rows["team_id"].nunique() != 1:
        raise ValueError(f"unknown or ambiguous team: {team!r}")
    return rows


def _roster(events: pd.DataFrame, lineups: pd.DataFrame | None) -> dict:
    """Participation intervals in event order, with IDs as the identity key."""
    people = {}

    def add(pid, name, position=None, *, entry=None, starter=False):
        if pd.isna(pid):
            return
        pid = int(pid)
        person = people.setdefault(pid, dict(player_id=pid, player=name or str(pid), position=None,
                                            starter=False, entry=np.inf, exit=np.inf))
        if position and not person["position"]:
            person["position"] = position
        person["starter"] |= starter
        if entry is not None:
            person["entry"] = min(person["entry"], entry)

    if lineups is not None and not lineups.empty:
        mid, tid = int(events.iloc[0].match_id), int(events.iloc[0].team_id)
        squad = lineups.loc[lineups.match_id.eq(mid) & lineups.team_id.eq(tid)]
        for row in squad.itertuples():
            add(row.player_id, row.player, row.position, entry=0 if row.starter else None, starter=row.starter)
    for row in events.itertuples():
        for player in row.starting_lineup:
            add(player["player"]["id"], player["player"]["name"], player.get("position", {}).get("name"), entry=0, starter=True)
        # Bench players can receive cards. A disciplinary event alone is not
        # evidence of entering the pitch, even though it has a player ID.
        on_field_event = row.type != "Bad Behaviour" and row.position != "Substitute"
        add(row.player_id, row.player, row.position,
            entry=int(row.index) if on_field_event else None)
        if row.type == "Substitution":
            add(row.replacement_id, row.replacement, row.position, entry=int(row.index))
            if pd.notna(row.player_id):
                people[int(row.player_id)]["exit"] = int(row.index)
        if row.card in {"Red Card", "Second Yellow"} and pd.notna(row.player_id):
            people[int(row.player_id)]["exit"] = int(row.index)
        if row.type == "Pass" and row.completed:
            add(row.recipient_id, row.recipient, entry=int(row.index))
    return people


def _assemble(players: pd.DataFrame, counts: pd.DataFrame, metadata: dict) -> PassingNetwork:
    players = players.sort_values("player_id").reset_index(drop=True).copy()
    # Names remain readable, but two different IDs must never merge by name.
    duplicate = players["player"].duplicated(keep=False)
    players["label"] = [f"{r.player} [{r.player_id}]" if duplicate.iloc[i] else r.player
                        for i, r in enumerate(players.itertuples())]
    labels = dict(zip(players.player_id, players.label))
    edges = pd.DataFrame({
        "passer": counts.passer_id.map(labels), "recipient": counts.recipient_id.map(labels),
        "passes": counts.passes.astype(int),
    }).reset_index(drop=True)
    graph = Graph.from_edges(edges.itertuples(index=False, name=None), nodes=players.label.tolist(),
                             weighted=True, directed=True, dangling=metadata["dangling"])
    return PassingNetwork(graph, players, edges, metadata)


def build_network(
    events: pd.DataFrame, team, lineups: pd.DataFrame | None = None, *,
    time_window: tuple[float | None, float | None] | None = None,
    before_first_substitution: bool = False, include_substitutes: bool = True,
    min_pass_count: int = 1, exclude_set_pieces: bool = False, dangling: str = "self_loop",
) -> PassingNetwork:
    """Build one team's directed, weighted network for one match.

    ``time_window=(start,end)`` uses half-open StatsBomb match minutes, including
    seconds. Periods 1--4 are eligible; shootouts are excluded. First-substitution
    filtering uses the team's event index (also excluding later same-time events).
    Nodes are players on the field during the selected event interval, even if
    they make no completed passes. Unused bench players never become nodes.
    With ``include_substitutes=False`` only starting players are retained; this
    does not restrict the time window automatically. Positions are the first
    recorded lineup role, falling back to event roles, not inferred from x/y.
    """
    if events.empty or events.match_id.nunique() != 1:
        raise ValueError("provide events for exactly one match")
    if not isinstance(min_pass_count, int) or min_pass_count < 1:
        raise ValueError("min_pass_count must be a positive integer")
    start, end = time_window if time_window is not None else (None, None)
    if any(v is not None and (not np.isfinite(v) or v < 0) for v in [start, end]):
        raise ValueError("time bounds must be finite non-negative minutes")
    if start is not None and end is not None and start >= end:
        raise ValueError("time window must have start < end")
    rows = _team_events(events, team)
    people = _roster(rows, lineups)
    clock = rows.minute + rows.second / 60.0
    keep = rows.period.between(1, 4)
    if start is not None:
        keep &= clock >= start
    if end is not None:
        keep &= clock < end
    substitutions = rows.loc[rows.type.eq("Substitution") & rows.period.between(1, 4)]
    cutoff = int(substitutions["index"].min()) if before_first_substitution and not substitutions.empty else None
    if cutoff is not None:
        keep &= rows["index"] < cutoff
    window = rows.loc[keep]
    if window.empty:
        raise ValueError("no team events in the requested window")
    first, last = int(window["index"].min()), int(window["index"].max())
    active = [p for p in people.values() if p["entry"] <= last and p["exit"] > first
              and (include_substitutes or p["starter"])]
    if not active:
        raise ValueError("no participating players in the requested window")
    players = pd.DataFrame(active).drop(columns=["entry", "exit"])
    locations = window.loc[window.player_id.notna() & window.x.notna() & window.y.notna()]
    positions = locations.groupby("player_id").agg(x=("x", "mean"), y=("y", "mean"), location_count=("x", "count"))
    players = players.join(positions, on="player_id")
    players["location_count"] = players.location_count.fillna(0).astype(int)
    passes = pass_table(window)
    complete = passes.loc[passes.completed].copy()
    missing = int((complete.passer_id.isna() | complete.recipient_id.isna()).sum())
    complete = complete.dropna(subset=["passer_id", "recipient_id"])
    if exclude_set_pieces:
        complete = complete.loc[~complete.pass_type.isin(SET_PIECES)]
    complete = complete.loc[complete.passer_id.isin(players.player_id) & complete.recipient_id.isin(players.player_id)]
    counts = complete.groupby(["passer_id", "recipient_id"]).size().reset_index(name="passes")
    counts = counts.loc[counts.passes >= min_pass_count]
    metadata = dict(
        schema_version=1, match_id=int(rows.iloc[0].match_id), match_ids=[int(rows.iloc[0].match_id)],
        team=str(rows.iloc[0].team), team_id=int(rows.iloc[0].team_id), dangling=dangling,
        time_window=[start, end], before_first_substitution=before_first_substitution,
        cutoff_event_index=cutoff, include_substitutes=include_substitutes,
        min_pass_count=min_pass_count, exclude_set_pieces=exclude_set_pieces,
        excluded_set_piece_types=sorted(SET_PIECES) if exclude_set_pieces else [], periods=[1, 2, 3, 4],
        diagnostics=dict(passes_in_window=len(passes), completed_in_window=int(passes.completed.sum()),
                         completed_missing_endpoints=missing, retained_passes=int(counts.passes.sum()),
                         players_without_locations=int(players.location_count.eq(0).sum())),
    )
    return _assemble(players, counts, metadata)


def aggregate(networks: list[PassingNetwork], *, min_pass_count: int = 1) -> PassingNetwork:
    """Pool raw pass counts by player ID, then apply a pooled edge threshold.

    Inputs must be single-match networks for one team, without prior thresholding.
    Coordinates are weighted by the number of observed event locations; roles
    are combined as a sorted list of first-recorded match roles.
    """
    if not networks:
        raise ValueError("at least one network is required")
    if not isinstance(min_pass_count, int) or min_pass_count < 1:
        raise ValueError("min_pass_count must be a positive integer")
    reference = networks[0].metadata
    keys = ["team_id", "dangling", "time_window", "before_first_substitution", "include_substitutes", "exclude_set_pieces"]
    if any(n.metadata.get("match_id") is None or n.metadata["min_pass_count"] != 1 for n in networks):
        raise ValueError("aggregate unthresholded single-match networks")
    if any(n.metadata[k] != reference[k] for n in networks for k in keys):
        raise ValueError("aggregation requires the same team and filter options")
    mids = [n.metadata["match_id"] for n in networks]
    if len(set(mids)) != len(mids):
        raise ValueError("duplicate matches would double-count passes")
    rows, edge_rows = [], []
    for network in networks:
        rows.append(network.players)
        ids = dict(zip(network.players.label, network.players.player_id))
        for edge in network.edges.itertuples():
            edge_rows.append((ids[edge.passer], ids[edge.recipient], edge.passes))
    combined = pd.concat(rows, ignore_index=True)
    players = []
    for pid, group in combined.groupby("player_id", sort=True):
        n = int(group.location_count.sum())
        players.append(dict(player_id=int(pid), player=group.iloc[0].player,
                            position="; ".join(sorted(set(group.position.dropna()))), starter=bool(group.starter.any()),
                            x=float((group.x.fillna(0) * group.location_count).sum() / n) if n else np.nan,
                            y=float((group.y.fillna(0) * group.location_count).sum() / n) if n else np.nan,
                            location_count=n))
    counts = pd.DataFrame(edge_rows, columns=["passer_id", "recipient_id", "passes"])
    counts = counts.groupby(["passer_id", "recipient_id"], as_index=False).passes.sum()
    counts = counts.loc[counts.passes >= min_pass_count]
    metadata = {k: v for k, v in reference.items() if k not in {"diagnostics", "cutoff_event_index"}}
    metadata.update(match_id=None, match_ids=sorted(mids), min_pass_count=min_pass_count,
                    aggregation="sum of completed passes by player ID", component_metadata=[n.metadata for n in networks])
    return _assemble(pd.DataFrame(players), counts, metadata)


def save_network(network: PassingNetwork, path: str | Path) -> Path:
    """Write edge CSV and a sidecar preserving all nodes and their exact order."""
    path = Path(path).with_suffix(".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    network.edges.to_csv(path, index=False)
    # pandas converts missing coordinates/roles to JSON null, not invalid NaN.
    metadata = dict(network.metadata, players=json.loads(network.players.to_json(orient="records")))
    path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def load_saved(path: str | Path) -> Graph:
    """Reload an edge CSV (or its JSON sidecar) with labels/order/dangling intact."""
    path = Path(path).with_suffix(".csv")
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    edges = pd.read_csv(path, dtype={"passer": str, "recipient": str}, keep_default_na=False)
    return Graph.from_edges(edges.itertuples(index=False, name=None),
                            nodes=[p["label"] for p in metadata["players"]], weighted=True,
                            directed=True, dangling=metadata["dangling"])
