# Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric as pyg


def plot_graph(edge_index, pos_lat_lon, title=None):
    """
    Plot flattened global graph

    edge_index: (2, N_edges) tensor
    pos_lat_lon: (N_nodes, 2) tensor containing longitudes and latitudes
    """
    fig, axis = plt.subplots(figsize=(8, 8), dpi=200)  # W,H

    # Fix for re-indexed edge indices only containing mesh nodes at
    # higher levels in hierarchy
    edge_index = edge_index - edge_index.min()

    if pyg.utils.is_undirected(edge_index):
        # Keep only 1 direction of edge_index
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]  # (2, M/2)

    # Move all to cpu and numpy, compute (in)-degrees
    degrees = (
        pyg.utils.degree(edge_index[1], num_nodes=pos_lat_lon.shape[0])
        .cpu()
        .numpy()
    )
    edge_index = edge_index.cpu().numpy()
    # Make lon x-axis
    pos = torch.stack((pos_lat_lon[:, 1], pos_lat_lon[:, 0]), dim=1)
    pos = pos.cpu().numpy()

    # Plot edges
    from_pos = pos[edge_index[0]]  # (M/2, 2)
    to_pos = pos[edge_index[1]]  # (M/2, 2)
    edge_lines = np.stack((from_pos, to_pos), axis=1)
    axis.add_collection(
        matplotlib.collections.LineCollection(
            edge_lines, lw=0.4, colors="black", zorder=1
        )
    )

    # Plot nodes
    node_scatter = axis.scatter(
        pos[:, 0],
        pos[:, 1],
        c=degrees,
        s=3,
        marker="o",
        zorder=2,
        cmap="viridis",
        clim=None,
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")

    plt.colorbar(node_scatter, aspect=50)

    if title is not None:
        axis.set_title(title)

    return fig, axis
