"""Save exact centrality, group vectors and all single-source PPR ground truth."""

import pandas as pd

from common import parse_args, prepared_networks, save_csv, save_matrix
from football.analysis import analyze


def main():
    args = parse_args(__doc__)
    rankings, checks, sensitivities = [], [], []
    for key, network in prepared_networks(args):
        result = analyze(network)
        context = dict(network_id=key, match_id=network.metadata["match_id"], team=network.metadata["team"],
                       team_id=network.metadata["team_id"])
        rankings.append(result["rankings"].assign(**context))
        checks.append(result["checks"].assign(**context))
        sensitivities.append(result["sensitivity"].assign(**context))
        sources = pd.DataFrame(result["sources"], index=network.players.label)
        sources.index.name = "player"
        folder = args.output / "ground_truth"
        folder.mkdir(exist_ok=True)
        sources.to_csv(folder / f"{key}_sources.csv")
        for alpha, matrix in result["matrices"].items():
            save_matrix(matrix, network.graph.labels, folder / f"{key}_a{alpha:g}_influence.csv", "destination")
            vectors = pd.DataFrame({name: result["vectors"][alpha, name] for name in result["sources"]},
                                   index=pd.Index(network.graph.labels, name="player"))
            vectors.to_csv(folder / f"{key}_a{alpha:g}_vectors.csv")
        print(f"Exact/power verified: {key}", flush=True)
    table = pd.concat(rankings, ignore_index=True)
    save_csv(table, args.output / "rankings.csv")
    save_csv(table.loc[table.source.eq("uniform")], args.output / "centrality_rankings.csv")
    save_csv(pd.concat(checks, ignore_index=True), args.output / "power_checks.csv")
    save_csv(pd.concat(sensitivities, ignore_index=True), args.output / "alpha_sensitivity.csv")


if __name__ == "__main__":
    main()
