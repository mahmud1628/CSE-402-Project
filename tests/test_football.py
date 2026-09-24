"""Football modelling tests use synthetic StatsBomb records, never local data."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

import pprlib as pl
from football import statsbomb
from football.analysis import analyze, checked_ppr, group_sources
from football.networks import aggregate, build_network, load_saved, save_network

TEAM = {"id": 10, "name": "Test FC"}
PEOPLE = {1: "A", 2: "B", 3: "C", 4: "D", 5: "Unused"}
ROLES = {1: "Goalkeeper", 2: "Left Center Back", 3: "Center Forward", 4: "Center Forward", 5: "Right Back"}


def event(index, kind, player=None, minute=0, **extra):
    row = dict(id=str(index), index=index, type={"name": kind}, team=TEAM,
               minute=minute, second=0, period=1, timestamp="00:00:00.000")
    if player is not None:
        row.update(player={"id": player, "name": PEOPLE[player]}, position={"name": ROLES[player]}, location=[index * 2, 40])
    row.update(extra)
    return row


def passing(index, player, recipient, minute=1, **extra):
    fields = {"end_location": [40, 30], **extra}
    if recipient is not None:
        fields["recipient"] = {"id": recipient, "name": PEOPLE[recipient]}
    return event(index, "Pass", player, minute, **{"pass": fields})


@pytest.fixture
def raw_events():
    return [
        event(1, "Starting XI", tactics={"lineup": [
            {"player": {"id": i, "name": PEOPLE[i]}, "position": {"name": ROLES[i]}}
            for i in (1, 2, 3)]}),
        passing(2, 1, 2), passing(3, 1, 2), passing(4, 2, 3, minute=2),
        passing(5, 2, 1, minute=3, outcome={"name": "Incomplete"}),
        passing(6, 3, 1, minute=4, outcome=None),  # key presence means incomplete
        passing(7, 2, 1, minute=5, type={"name": "Corner"}),
        event(8, "Substitution", 3, 10, substitution={"replacement": {"id": 4, "name": "D"}}),
        passing(9, 4, 1, minute=10), passing(10, 1, 4, minute=11),
        passing(11, 2, None, minute=12),  # completed, but unusable endpoint
        event(12, "Half End", minute=45),
    ]


@pytest.fixture
def raw_lineups():
    return [{"team_id": 10, "team_name": "Test FC", "lineup": [
        dict(player_id=i, player_name=name, jersey_number=i,
             positions=[dict(position=ROLES[i], start_reason="Starting XI" if i<=3 else "Substitution",
                             **{"from": "00:00" if i<=3 else "10:00", "to": None,
                                "from_period": 1, "to_period": None})] if i<=4 else [])
        for i, name in PEOPLE.items()]}]


@pytest.fixture
def frames(raw_events, raw_lineups):
    return statsbomb.events_frame(raw_events, 100), statsbomb.lineups_frame(raw_lineups, 100)


def test_completed_duplicate_and_missing_passes(frames):
    events, lineups = frames
    network = build_network(events, "Test FC", lineups)
    g = network.graph
    assert network.W[g.index("A"), g.index("B")] == 2
    assert network.W[g.index("B"), g.index("A")] == 1  # incomplete not counted
    assert network.W[g.index("C")].sum() == 0  # null outcome still incomplete
    assert network.W.sum() == 6
    assert network.metadata["diagnostics"]["completed_missing_endpoints"] == 1
    assert network.players.player_id.tolist() == [1, 2, 3, 4]
    assert "Unused" not in g.labels


@pytest.mark.parametrize("dangling", ["self_loop", "uniform"])
def test_dangling_stochasticity_and_ppr(frames, dangling):
    network = build_network(frames[0], 10, frames[1], dangling=dangling)
    g = network.graph
    assert g.index("C") in g.dangling_nodes
    np.testing.assert_allclose(network.P.sum(axis=0), 1.0)
    for alpha in (0.5, 0.8, 0.85, 0.9, 0.99):
        result, checks = checked_ppr(g, alpha=alpha)
        assert checks["power_l1_error"] < 1e-10
        assert result.estimate.sum() == pytest.approx(1.0)


def test_hand_computed_three_player_ppr(raw_events):
    # A -> B -> C, with C absorbing. At alpha=1/2 and sigma=e_A,
    # pi_A=1/2, pi_B=1/4, pi_C=(1/2*pi_B)/(1-1/2)=1/4.
    rows = [raw_events[0], passing(2, 1, 2), passing(3, 2, 3)]
    network = build_network(statsbomb.events_frame(rows, 100), 10)
    result = pl.compute_ppr(network.graph, "A", alpha=0.5, method="exact")
    np.testing.assert_allclose(result.estimate, [0.5, 0.25, 0.25], atol=1e-14)


def test_no_pass_player_kept_and_no_edges_allowed(raw_events, raw_lineups):
    rows = [raw_events[0], passing(2, 1, 2, outcome={"name": "Out"})]
    network = build_network(statsbomb.events_frame(rows, 100), 10,
                            statsbomb.lineups_frame(raw_lineups, 100))
    assert network.graph.labels == ["A", "B", "C"]
    assert network.graph.m == 0
    np.testing.assert_array_equal(network.P, np.eye(3))
    assert network.players.loc[network.players.player.eq("C"), "location_count"].iloc[0] == 0


def test_filter_options_and_substitution_boundary(frames):
    events, lineups = frames
    before = build_network(events, 10, lineups, before_first_substitution=True)
    assert before.graph.labels == ["A", "B", "C"]
    assert before.metadata["cutoff_event_index"] == 8
    assert before.W.sum() == 4
    no_subs = build_network(events, 10, lineups, include_substitutes=False)
    assert no_subs.graph.labels == ["A", "B", "C"]
    no_restarts = build_network(events, 10, lineups, exclude_set_pieces=True)
    assert no_restarts.W.sum() == 5
    threshold = build_network(events, 10, lineups, min_pass_count=2)
    assert threshold.graph.m == 1
    assert threshold.W.sum() == 2
    assert threshold.graph.n == 4


def test_time_window_excludes_departed_and_future_players(frames):
    early = build_network(frames[0], 10, frames[1], time_window=(0, 10))
    assert early.graph.labels == ["A", "B", "C"]
    late = build_network(frames[0], 10, frames[1], time_window=(10, 12))
    assert late.graph.labels == ["A", "B", "D"]
    assert late.W.sum() == 2
    with pytest.raises(ValueError, match="start < end"):
        build_network(frames[0], 10, time_window=(10, 5))
    with pytest.raises(ValueError, match="no team events"):
        build_network(frames[0], 10, time_window=(100, 110))


def test_first_sub_uses_event_order_across_stoppage_time(raw_events):
    # First-half 46th minute happens before second-half substitution at 45:00.
    rows = [raw_events[0], passing(2, 1, 2, minute=46),
            event(3, "Substitution", 3, 45, period=2,
                  substitution={"replacement": {"id": 4, "name": "D"}}),
            passing(4, 4, 1, minute=45)]
    rows[-1]["period"] = 2
    network = build_network(statsbomb.events_frame(rows, 100), 10, before_first_substitution=True)
    assert network.W.sum() == 1
    assert network.graph.labels == ["A", "B", "C"]


def test_shootouts_do_not_add_edges_or_players(raw_events):
    rows = [raw_events[0], passing(2, 1, 2)]
    shootout = passing(3, 4, 1, minute=120)
    shootout["period"] = 5
    network = build_network(statsbomb.events_frame(rows + [shootout], 100), 10)
    assert network.W.sum() == 1 and "D" not in network.graph.labels


def test_bench_card_does_not_count_as_participation(raw_events, raw_lineups):
    rows = raw_events + [event(13, "Bad Behaviour", 5, minute=50,
                              position={"name": "Substitute"},
                              bad_behaviour={"card": {"name": "Yellow Card"}})]
    network = build_network(statsbomb.events_frame(rows, 100), 10,
                            statsbomb.lineups_frame(raw_lineups, 100))
    assert "Unused" not in network.graph.labels


def test_red_card_removes_player_from_later_window(raw_events):
    rows = raw_events[:4] + [event(5, "Bad Behaviour", 3, minute=5,
                                 bad_behaviour={"card": {"name": "Red Card"}}),
                            passing(6, 1, 2, minute=10)]
    network = build_network(statsbomb.events_frame(rows, 100), 10, time_window=(10, 11))
    assert network.graph.labels == ["A", "B"]


@pytest.mark.parametrize("dangling", ["self_loop", "uniform"])
def test_roundtrip_preserves_nodes_weights_order_and_metadata(frames, tmp_path, dangling):
    network = build_network(frames[0], 10, frames[1], min_pass_count=2, dangling=dangling)
    path = save_network(network, tmp_path / "match.csv")
    restored = load_saved(path.with_suffix(".json"))
    assert restored.labels == network.graph.labels
    assert restored.dangling == dangling
    np.testing.assert_array_equal(restored.A.toarray(), network.W)
    np.testing.assert_allclose(restored.dense_P(), network.P)
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["match_id"] == 100 and metadata["min_pass_count"] == 2
    assert [p["player_id"] for p in metadata["players"]] == [1, 2, 3, 4]


def test_empty_edge_roundtrip(raw_events, tmp_path):
    network = build_network(statsbomb.events_frame(raw_events[:1], 100), 10)
    graph = load_saved(save_network(network, tmp_path / "empty.csv"))
    assert graph.n == 3 and graph.m == 0


def test_aggregate_counts_and_location_weighting(frames):
    events, lineups = frames
    first = build_network(events, 10, lineups)
    second = build_network(events.assign(match_id=101), 10, lineups.assign(match_id=101))
    combined = aggregate([first, second])
    np.testing.assert_array_equal(combined.W, first.W * 2)
    np.testing.assert_allclose(combined.players.x, first.players.x, equal_nan=True)
    assert combined.metadata["match_ids"] == [100, 101]
    assert aggregate([first, second], min_pass_count=3).graph.m == 1
    with pytest.raises(ValueError, match="duplicate matches"):
        aggregate([first, first])
    with pytest.raises(ValueError, match="unthresholded"):
        aggregate([build_network(events, 10, min_pass_count=2)])


def test_name_collisions_do_not_merge_players(raw_events):
    rows = deepcopy(raw_events[:4])
    for item in rows[0]["tactics"]["lineup"]:
        if item["player"]["id"] in (1, 2):
            item["player"]["name"] = "Same Name"
    graph = build_network(statsbomb.events_frame(rows, 100), 10).graph
    assert graph.labels == ["Same Name [1]", "Same Name [2]", "C"]
    assert graph.A[0, 1] == 2


def test_group_personalization_and_influence_orientation(frames):
    network = build_network(frames[0], 10, frames[1])
    sources = group_sources(network)
    np.testing.assert_array_equal(sources["goalkeepers"], [1, 0, 0, 0])
    np.testing.assert_array_equal(sources["defenders"], [0, 1, 0, 0])
    result = analyze(network, alphas=(0.8, 0.85, 0.99))
    np.testing.assert_allclose(result["matrices"][0.85][:, 0], result["vectors"][0.85, "goalkeepers"])
    assert result["checks"].power_l1_error.max() < 1e-10
    assert result["sensitivity"].loc[result["sensitivity"].alpha.eq(0.85), "kendall_tau"].iloc[0] == pytest.approx(1)


def test_local_loaders(tmp_path, raw_events, raw_lineups):
    for directory in ["events", "lineups", "matches/1"]:
        (tmp_path / directory).mkdir(parents=True)
    for file, data in [("competitions.json", [{"competition_id": 1, "season_id": 2}]),
                       ("matches/1/2.json", [{"match_id": 100, "home_team": {"home_team_id": 10}}]),
                       ("events/100.json", raw_events), ("lineups/100.json", raw_lineups)]:
        (tmp_path / file).write_text(json.dumps(data))
    assert len(statsbomb.load_competitions(tmp_path / "events")) == 1
    assert statsbomb.load_matches(tmp_path, 1, 2).iloc[0].match_id == 100
    events = statsbomb.load_events(tmp_path, 100)
    assert statsbomb.pass_table(events).completed.sum() == 7
    lineups = statsbomb.load_lineups(tmp_path, 100)
    assert lineups.participated.sum() == 4
    with pytest.raises(FileNotFoundError):
        statsbomb.load_lineups(tmp_path, 999)


def test_empty_input_schema_and_invalid_team(frames):
    assert statsbomb.pass_table(statsbomb.events_frame([], 100)).empty
    with pytest.raises(ValueError, match="exactly one match"):
        build_network(statsbomb.events_frame([], 100), 10)
    with pytest.raises(ValueError, match="unknown or ambiguous"):
        build_network(frames[0], "Missing")
    with pytest.raises(ValueError, match="positive integer"):
        build_network(frames[0], 10, min_pass_count=0)
