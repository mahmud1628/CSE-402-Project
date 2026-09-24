"""Exercise six directed MC estimators on both teams of one real match."""

import pandas as pd

from common import parse_args, prepared_networks, save_csv
from football.analysis import smoke_test


def main():
    args = parse_args(__doc__)
    rows = []
    first_match = None
    for key, network in prepared_networks(args):
        if first_match is None:
            first_match = network.metadata["match_id"]
        if network.metadata["match_id"] != first_match:
            continue
        rows.append(smoke_test(network, seed=args.seed, quick=args.quick).assign(network_id=key, match_id=first_match))
    save_csv(pd.concat(rows, ignore_index=True), args.output / "mc_smoke.csv")
    print("Six directed MC methods completed for both teams at three alpha values.", flush=True)


if __name__ == "__main__":
    main()
