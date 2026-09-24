"""Build selected match networks and per-team/per-season aggregated networks."""

from collections import defaultdict
import json

import numpy as np
import pandas as pd

from common import parse_args, selected_matches, match_networks, save_csv, save_matrix
from football.networks import aggregate, save_network, load_saved


def main():
    args = parse_args(__doc__)
    selected = selected_matches(args)
    groups = defaultdict(list)
    inventory, summaries = [], []

    def save(key, network, kind):
        path = save_network(network, args.output / "networks" / f"{key}.csv")
        g = load_saved(path)
        np.testing.assert_array_equal(g.A.toarray(), network.W)
        np.testing.assert_allclose(g.dense_P(), network.P, atol=1e-15)
        assert g.labels == network.graph.labels
        np.testing.assert_allclose(network.P.sum(axis=0), 1.0, atol=1e-12)
        save_matrix(network.W, g.labels, args.output / "matrices" / f"{key}_W.csv", "passer")
        save_matrix(network.P, g.labels, args.output / "matrices" / f"{key}_P.csv", "destination")
        inventory.append(dict(network_id=key, kind=kind, path=str(path.relative_to(args.output))))
        summaries.append(dict(network_id=key, kind=kind, team=network.metadata["team"],
                              match_id=network.metadata["match_id"], n_matches=len(network.metadata["match_ids"]),
                              n_players=g.n, n_edges=g.m, n_passes=int(network.W.sum()),
                              dangling_players=len(g.dangling_nodes),
                              players_without_locations=int(network.players.location_count.eq(0).sum()),
                              completed_missing_endpoints=network.metadata.get("diagnostics", {}).get("completed_missing_endpoints", 0)))

    for key, network in match_networks(args, selected):
        save(key, network, "match")
        m = network.metadata
        groups[m["competition_id"], m["season_id"], m["team_id"]].append(network)
    for (competition, season, team), networks in sorted(groups.items()):
        if len(networks) > 1:
            save(f"c{competition}_s{season}_t{team}_aggregate", aggregate(networks), "aggregate")
    save_csv(pd.DataFrame(summaries), args.output / "network_summary.csv")
    save_csv(pd.DataFrame(selected), args.output / "selected_matches.csv")
    manifest = dict(schema_version=1, data_root=str(args.data), selection=str(args.selection.resolve()),
                    quick=args.quick, seed=args.seed, match_ids=[m["match_id"] for m in selected], networks=inventory)
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built and verified {len(inventory)} networks from {len(selected)} matches.", flush=True)


if __name__ == "__main__":
    main()
