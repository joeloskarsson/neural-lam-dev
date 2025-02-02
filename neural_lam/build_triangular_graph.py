# Standard library
import argparse
import os

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
from graphcast import graphcast as gc_gc
from graphcast import grid_mesh_connectivity as gc_gm

# Local
from . import utils
from .config import load_config_and_datastores
from .graphs import create as gcreate
from .graphs import vis as gvis


def main():
    parser = argparse.ArgumentParser(
        description="Triangular graph generation using weather-models-graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Inputs and outputs
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the configuration for neural-lam",
    )
    parser.add_argument(
        "--graph_name",
        type=str,
        default="multiscale",
        help="Name to save graph as (default: multiscale)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If graphs should be plotted during generation ",
    )

    # Graph structure
    parser.add_argument(
        "--splits",
        default=3,
        type=int,
        help="Number of splits to triangular mesh",
    )
    parser.add_argument(
        "--levels",
        type=int,
        help="Number of levels to keep, from finest upwards ",
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Generate hierarchical mesh graph",
    )
    args = parser.parse_args()

    _, datastore, datastore_boundary = load_config_and_datastores(
        config_path=args.config_path
    )

    # Set up dir for saving graph
    save_dir_path = os.path.join(datastore.root_path, "graphs", args.graph_name)
    os.makedirs(save_dir_path, exist_ok=True)

    # Load grid positions
    grid_lat_lon = utils.get_stacked_lat_lons(
        datastore, datastore_boundary
    ).astype(
        np.float32
    )  # Must be float32 for interoperability with gc code
    grid_lat_lon = grid_lat_lon[::1000]  # TODO Remove
    # (num_nodes_full, 2)
    num_grid_nodes = grid_lat_lon.shape[0]

    # Make all longitudes be in [0, 360]
    grid_lat_lon[:, 0] = (grid_lat_lon[:, 0] + 360.0) % 360.0

    grid_lat_lon_torch = torch.tensor(grid_lat_lon, dtype=torch.float32)
    torch.save(
        grid_lat_lon_torch, os.path.join(save_dir_path, "grid_lat_lon.pt")
    )

    # === Create mesh graph ===
    if args.hierarchical:
        # Save up+down edge index + features to disk
        #  torch.save(
        #  mesh_up_ei_list,
        #  os.path.join(save_dir_path, "mesh_up_edge_index.pt"),
        #  )
        #  torch.save(
        #  mesh_down_ei_list,
        #  os.path.join(save_dir_path, "mesh_down_edge_index.pt"),
        #  )
        #  torch.save(
        #  mesh_up_features_list,
        #  os.path.join(save_dir_path, "mesh_up_features.pt"),
        #  )
        #  torch.save(
        #  mesh_down_features_list,
        #  os.path.join(save_dir_path, "mesh_down_features.pt"),
        #  )
        # max_mesh_edge_len = ?
        pass
        # TODO Hierarchical graph
    else:
        merged_mesh, mesh_list = gcreate.create_multiscale_mesh(
            args.splits, args.levels
        )
        max_mesh_edge_len = gc_gc._get_max_edge_distance(mesh_list[-1])
        m2m_graphs = [merged_mesh]

    mesh_graph_features = [
        gcreate.create_mesh_graph_features(mesh_graph)
        for mesh_graph in m2m_graphs
    ]
    # Ordering: edge_index, node_features, edge_features, lat_lon

    # Save to static dir
    for feat_index, file_name in enumerate(
        (
            "m2m_edge_index.pt",
            "mesh_features.pt",
            "m2m_features.pt",
            "mesh_lat_lon.pt",
        )
    ):
        torch.save(
            # Save as list
            [feats[feat_index] for feats in mesh_graph_features],
            os.path.join(save_dir_path, file_name),
        )

    if args.plot:
        # Plot each mesh graph level
        for level_i, (level_edge_index, _, _, level_lat_lon) in enumerate(
            mesh_graph_features
        ):
            gvis.plot_graph(
                level_edge_index, level_lat_lon, title=f"Mesh level {level_i}"
            )
            plt.show()

    # === Grid2Mesh ===
    # Grid2Mesh: Radius-based
    grid_con_mesh = m2m_graphs[-1]  # Mesh that should be connected to grid
    grid_con_lat_lon = mesh_graph_features[-1][-1]

    # Compute maximum edge distance in finest mesh
    # pylint: disable-next=protected-access
    g2m_connect_radius = 0.6 * max_mesh_edge_len

    g2m_edge_index = gcreate.connect_to_mesh_radius(
        grid_lat_lon, grid_con_mesh, g2m_connect_radius
    )

    if args.plot:
        gvis.plot_graph(
            g2m_edge_index,
            from_node_pos=grid_lat_lon_torch,
            to_node_pos=grid_con_lat_lon,
            title="Grid2Mesh",
        )
        plt.show()

    # Get edge features for g2m
    g2m_edge_features = gcreate.create_edge_features(
        g2m_edge_index, sender_coords=grid_lat_lon, receiver_mesh=grid_con_mesh
    )

    # Save g2m
    torch.save(
        g2m_edge_index,
        os.path.join(save_dir_path, "g2m_edge_index.pt"),
    )
    torch.save(
        g2m_edge_features,
        os.path.join(save_dir_path, "g2m_features.pt"),
    )

    # === Mesh2Grid ===
    # Mesh2Grid: Connect from containing mesh triangle

    m2g_edge_index = gcreate.connect_to_grid_containing_tri(
        grid_lat_lon, grid_con_mesh
    )

    # Get edge features for m2g
    m2g_edge_features = gcreate.create_edge_features(
        m2g_edge_index, receiver_coords=grid_lat_lon, sender_mesh=grid_con_mesh
    )

    if args.plot:
        gvis.plot_graph(
            m2g_edge_index,
            from_node_pos=grid_con_lat_lon,
            to_node_pos=grid_lat_lon_torch,
            title="Grid2Mesh",
        )
        plt.show()

    # Save m2g
    torch.save(
        m2g_edge_index,
        os.path.join(save_dir_path, "m2g_edge_index.pt"),
    )
    torch.save(
        m2g_edge_features,
        os.path.join(save_dir_path, "m2g_features.pt"),
    )

    num_mesh_nodes = grid_con_lat_lon.shape[0]
    print(
        f"Created graph with {num_grid_nodes} grid nodes "
        f"connected to {num_mesh_nodes}"
    )
    print(f"#grid / #mesh = {num_grid_nodes/num_mesh_nodes :.2f}")


if __name__ == "__main__":
    main()
