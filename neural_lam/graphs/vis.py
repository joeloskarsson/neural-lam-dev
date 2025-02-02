# Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric as pyg


def plot_graph(edge_index, from_node_pos, to_node_pos=None, title=None):
    """
    Plot flattened global graph

    edge_index: (2, N_edges) tensor
    from_node_pos: (N_nodes, 2) tensor containing longitudes and latitudes
    to_node_pos: (N_nodes, 2) tensor containing longitudes and latitudes,
        or None (assumed same as from_node_pos)
    """
    if to_node_pos is None:
        # If to_node_pos is None it is same as from_node_pos
        to_node_pos = from_node_pos

    fig, axis = plt.subplots()

    # Fix for re-indexed edge indices only containing mesh nodes at
    # higher levels in hierarchy
    edge_index = edge_index - edge_index.min()

    if pyg.utils.is_undirected(edge_index):
        # Keep only 1 direction of edge_index
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]  # (2, M/2)

    # Compute (in)-degrees
    degrees = (
        pyg.utils.degree(edge_index[1], num_nodes=to_node_pos.shape[0])
        .cpu()
        .numpy()
    )

    # Move tensors to cpu and make numpy
    edge_index = edge_index.cpu().numpy()
    from_node_pos = from_node_pos.cpu().numpy()
    to_node_pos = to_node_pos.cpu().numpy()

    # Plot edges
    from_pos = from_node_pos[edge_index[0]]  # (M/2, 2)
    to_pos = to_node_pos[edge_index[1]]  # (M/2, 2)
    edge_lines = np.stack((from_pos, to_pos), axis=1)
    axis.add_collection(
        matplotlib.collections.LineCollection(
            edge_lines, lw=0.4, colors="black", zorder=1
        )
    )

    # Plot (receiver) nodes
    node_scatter = axis.scatter(
        to_node_pos[:, 0],
        to_node_pos[:, 1],
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
