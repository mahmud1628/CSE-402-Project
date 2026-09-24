"""Read local StatsBomb JSON into tables without network access.

A pass is completed exactly when its ``pass`` object has no ``outcome`` key.
Coordinates use the original StatsBomb 120 by 80 pitch. Match minutes include
stoppage time; event indices, rather than minute alone, preserve chronology.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EVENT_COLUMNS = [
    "match_id", "event_id", "index", "type", "team_id", "team", "period",
    "minute", "second", "timestamp", "player_id", "player", "position",
    "x", "y", "recipient_id", "recipient", "completed", "pass_type",
    "end_x", "end_y", "replacement_id", "replacement", "starting_lineup", "card",
]
PASS_COLUMNS = [
    "match_id", "team_id", "team", "index", "period", "minute", "second",
    "passer_id", "passer", "recipient_id", "recipient", "completed",
    "pass_type", "x", "y", "end_x", "end_y",
]
LINEUP_COLUMNS = [
    "match_id", "team_id", "team", "player_id", "player", "jersey_number",
    "position", "positions", "starter", "participated",
]


def data_root(path: str | Path) -> Path:
    """Accept a data directory, its events directory, or a repo containing data/."""
    root = Path(path).expanduser().resolve()
    if root.name in {"events", "matches", "lineups"}:
        root = root.parent
    elif (root / "data").is_dir():
        root = root / "data"
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def read_json(path: str | Path) -> list:
    """Read a JSON list, failing explicitly for empty or malformed files."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError(f"expected a JSON list: {path}")
    return rows


def load_competitions(path: str | Path) -> pd.DataFrame:
    return pd.json_normalize(read_json(data_root(path) / "competitions.json"))


def load_matches(path: str | Path, competition_id=None, season_id=None) -> pd.DataFrame:
    """One row per match; nested metadata columns use underscores."""
    root = data_root(path) / "matches"
    if not root.is_dir():
        raise FileNotFoundError(root)
    pattern = f"{competition_id if competition_id is not None else '*'}/{season_id if season_id is not None else '*'}.json"
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no match metadata matching {root / pattern}")
    rows = [row for file in files for row in read_json(file)]
    return pd.json_normalize(rows, sep="_")


def _xy(value) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return np.nan, np.nan


def events_frame(events: list[dict], match_id: int) -> pd.DataFrame:
    """Flatten events, keeping starting lineups for zero-pass player inclusion."""
    rows = []
    for i, event in enumerate(events):
        passing = event.get("pass", {})
        replacement = event.get("substitution", {}).get("replacement", {})
        x, y = _xy(event.get("location"))
        end_x, end_y = _xy(passing.get("end_location"))
        rows.append(dict(
            match_id=int(match_id), event_id=event.get("id"), index=event.get("index", i + 1),
            type=event.get("type", {}).get("name"), team_id=event.get("team", {}).get("id"),
            team=event.get("team", {}).get("name"), period=event.get("period", 1),
            minute=event.get("minute", 0), second=event.get("second", 0), timestamp=event.get("timestamp"),
            player_id=event.get("player", {}).get("id"), player=event.get("player", {}).get("name"),
            position=event.get("position", {}).get("name"), x=x, y=y,
            recipient_id=passing.get("recipient", {}).get("id"), recipient=passing.get("recipient", {}).get("name"),
            completed=event.get("type", {}).get("name") == "Pass" and "outcome" not in passing,
            pass_type=passing.get("type", {}).get("name"), end_x=end_x, end_y=end_y,
            replacement_id=replacement.get("id"), replacement=replacement.get("name"),
            starting_lineup=event.get("tactics", {}).get("lineup", []) if event.get("type", {}).get("name") == "Starting XI" else [],
            card=event.get("bad_behaviour", {}).get("card", event.get("foul_committed", {}).get("card", {})).get("name"),
        ))
    frame = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    for column in ["match_id", "index", "team_id", "player_id", "recipient_id", "replacement_id"]:
        frame[column] = frame[column].astype("Int64")
    return frame.sort_values("index", kind="stable").reset_index(drop=True)


def load_events(path: str | Path, match_id: int) -> pd.DataFrame:
    return events_frame(read_json(data_root(path) / "events" / f"{match_id}.json"), match_id)


def pass_table(events: pd.DataFrame) -> pd.DataFrame:
    """Include incomplete passes; callers can audit or filter ``completed``."""
    return events.loc[events["type"].eq("Pass")].rename(
        columns={"player_id": "passer_id", "player": "passer"}
    )[PASS_COLUMNS].reset_index(drop=True)


def lineups_frame(lineups: list[dict], match_id: int) -> pd.DataFrame:
    """One row per squad player; ``positions`` retains all position intervals.

    Empty positions normally describe unused bench players. Starting XI and
    substitution events are authoritative for actual network participation.
    """
    rows = []
    for team in lineups:
        for player in team.get("lineup", []):
            positions = player.get("positions", [])
            rows.append(dict(
                match_id=int(match_id), team_id=team["team_id"], team=team["team_name"],
                player_id=player["player_id"], player=player["player_name"],
                jersey_number=player.get("jersey_number"), positions=positions,
                position=positions[0].get("position") if positions else None,
                starter=any(p.get("start_reason") == "Starting XI" for p in positions),
                participated=bool(positions),
            ))
    return pd.DataFrame(rows, columns=LINEUP_COLUMNS)


def load_lineups(path: str | Path, match_id: int) -> pd.DataFrame:
    return lineups_frame(read_json(data_root(path) / "lineups" / f"{match_id}.json"), match_id)
