"""Passing networks, source-to-destination PPR heatmaps and alpha sensitivity."""

import numpy as np
import pandas as pd

from common import parse_args, prepared_networks, mpl, PALETTE, GRID, INK, save_figure


def passing_figure(network, scores):
    """Plot observed average positions; missing positions are clearly separated."""
    plt = mpl()
    from matplotlib.patches import Arc, Circle, Rectangle

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.add_patch(Rectangle((0, 0), 120, 80, fill=False, edgecolor=INK))
    ax.plot([60, 60], [0, 80], color=GRID)
    ax.add_patch(Circle((60, 40), 10, fill=False, edgecolor=GRID))
    for x in [0, 102]:
        ax.add_patch(Rectangle((x, 18), 18, 44, fill=False, edgecolor=GRID))
    ax.add_patch(Arc((12, 40), 20, 20, theta1=-53, theta2=53, color=GRID))
    ax.add_patch(Arc((108, 40), 20, 20, theta1=127, theta2=233, color=GRID))
    points = network.players[["x", "y"]].to_numpy(dtype=float).copy()
    missing = ~np.isfinite(points).all(axis=1)
    if missing.any():
        points[missing, 0] = np.linspace(12, 108, missing.sum())
        points[missing, 1] = 91
        ax.text(0, 87, "No observed location (off-pitch display)", fontsize=9)
    max_passes = max(1, network.edges.passes.max()) if len(network.edges) else 1
    for edge in network.edges.itertuples():
        i, j = network.graph.index(edge.passer), network.graph.index(edge.recipient)
        ax.annotate("", xy=points[j], xytext=points[i],
                    arrowprops=dict(arrowstyle="->", color=PALETTE["blue"], alpha=0.35,
                                    linewidth=4 * edge.passes / max_passes,
                                    connectionstyle="arc3,rad=0.12", shrinkA=9, shrinkB=9), zorder=1)
    ax.scatter(points[:, 0], points[:, 1], s=9000 * scores, c=PALETTE["orange"], edgecolors=INK, zorder=2)
    for i, (x, y) in enumerate(points):
        ax.annotate(str(i + 1), (x, y), ha="center", va="center", fontsize=9, zorder=3)
    names = "\n".join(f"{i+1}. {name}" for i, name in enumerate(network.graph.labels))
    ax.text(1.02, 0.98, names, transform=ax.transAxes, va="top", fontsize=8)
    ax.set(xlim=(-3, 123), ylim=(98 if missing.any() else 83, -3), aspect="equal",
           title=f"{network.metadata['team']} — match {network.metadata['match_id']}\n"
                 "Node area ∝ PageRank (α=0.85); arrow width ∝ completed passes",
           xlabel="StatsBomb x (120 units)", ylabel="StatsBomb y (80 units)")
    ax.grid(False)
    fig.text(0.12, 0.01, "Source: StatsBomb Open Data. Positions: mean recorded player event locations.", fontsize=8)
    return fig


def heatmap_figure(network, matrix):
    plt = mpl()
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = plt.subplots(figsize=(11, 10))
    cmap = LinearSegmentedColormap.from_list("stage1_blue", ["#fcfcfb", PALETTE["blue"]])
    im = ax.imshow(matrix, cmap=cmap, vmin=0)
    ax.set_xticks(range(network.graph.n), network.graph.labels, rotation=90, fontsize=7)
    ax.set_yticks(range(network.graph.n), network.graph.labels, fontsize=7)
    ax.set(xlabel="Source player (column)", ylabel="Destination player (row)",
           title=f"{network.metadata['team']} — exact single-source PPR, α=0.85")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="PPR probability")
    fig.tight_layout()
    return fig


def main():
    args = parse_args(__doc__)
    plt = mpl()
    networks = list(prepared_networks(args))
    first_match = networks[0][1].metadata["match_id"]
    for key, network in networks:
        if network.metadata["match_id"] != first_match:
            continue
        ground = args.output / "ground_truth"
        scores = pd.read_csv(ground / f"{key}_a0.85_vectors.csv").uniform.to_numpy()
        matrix = pd.read_csv(ground / f"{key}_a0.85_influence.csv", index_col=0).to_numpy()
        save_figure(passing_figure(network, scores), args.output / "figures" / f"{key}_passing.png")
        save_figure(heatmap_figure(network, matrix), args.output / "figures" / f"{key}_influence.png")
    sensitivity = pd.read_csv(args.output / "alpha_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    # Only individual match networks here; aggregate results remain in the CSV.
    match_keys = [key for key, net in networks if net.metadata["match_id"] is not None]
    values = sensitivity.loc[sensitivity.network_id.isin(match_keys)]
    for _, rows in values.groupby("network_id"):
        ax.plot(rows.alpha, rows.kendall_tau, color=PALETTE["blue"], alpha=0.12, linewidth=1)
    mean = values.groupby("alpha").kendall_tau.mean()
    ax.plot(mean.index, mean, color=PALETTE["orange"], marker="o", label="Mean across team-match networks")
    ax.set(xlabel="α (continuation probability)", ylabel="Kendall τ-b vs α=0.85",
           title="Sensitivity of exact PageRank rankings", ylim=(-1.05, 1.05))
    ax.legend()
    save_figure(fig, args.output / "figures" / "alpha_sensitivity.png")


if __name__ == "__main__":
    main()
