# Standard library
import argparse
import os

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
from graphcast import graphcast as gc_gc
from graphcast import grid_mesh_connectivity as gc_gm

# First-party
import neural_lam.graphs.create as gcreate
import neural_lam.graphs.vis as gvis


def main():
    parser = argparse.ArgumentParser(
        description="Triangular graph generation using weather-models-graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Inputs and outputs
    parser.add_argument(
        "--data_config",
        type=str,
        default="neural_lam/data_config.yaml",
        help="Path to data config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="graphs",
        help="Directory to save graph to",
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

    assert args.output_dir, "Must specify an --output_dir"
    os.makedirs(args.output_dir, exist_ok=True)

    # TODO Get lat_lon from somewhere, use args.data_config
    example_dir = "example_lam_latlons"
    interior_lat_lon_raw = np.load(
        os.path.join(example_dir, "nwp_latlon.npy")
    ).astype(np.float32)
    boundary_lat_lon_raw = np.load(
        os.path.join(example_dir, "o80_latlon.npy")
    ).astype(np.float32)

    interior_lat_lon = np.stack(
        (
            interior_lat_lon_raw[1].flatten(),  # Lon
            interior_lat_lon_raw[0].flatten(),  # Lat
        ),
        axis=1,
    )
    boundary_lat_lon = np.stack(
        (
            boundary_lat_lon_raw[:, 1],  # Lon
            boundary_lat_lon_raw[:, 0],  # Lat
        ),
        axis=1,
    )

    # Concatenate interior and boundary coordinates
    grid_lat_lon = np.concatenate((interior_lat_lon, boundary_lat_lon), axis=0)
    grid_lat_lon = grid_lat_lon[::1000]  # TODO Remove
    # flattened, (num_grid_nodes, 2)
    num_grid_nodes = grid_lat_lon.shape[0]

    # Make all longitudes be in [0, 360]
    grid_lat_lon[:, 0] = (grid_lat_lon[:, 0] + 360.0) % 360.0

    grid_lat_lon_torch = torch.tensor(grid_lat_lon, dtype=torch.float32)
    # TODO: Save in graph dir?
    torch.save(
        grid_lat_lon_torch, os.path.join(args.output_dir, "grid_lat_lon.pt")
    )

    # === Create mesh graph ===
    if args.hierarchical:
        # Save up+down edge index + features to disk
        #  torch.save(
        #  mesh_up_ei_list,
        #  os.path.join(args.output_dir, "mesh_up_edge_index.pt"),
        #  )
        #  torch.save(
        #  mesh_down_ei_list,
        #  os.path.join(args.output_dir, "mesh_down_edge_index.pt"),
        #  )
        #  torch.save(
        #  mesh_up_features_list,
        #  os.path.join(args.output_dir, "mesh_up_features.pt"),
        #  )
        #  torch.save(
        #  mesh_down_features_list,
        #  os.path.join(args.output_dir, "mesh_down_features.pt"),
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
            os.path.join(args.output_dir, file_name),
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
        os.path.join(args.output_dir, "g2m_edge_index.pt"),
    )
    torch.save(
        g2m_edge_features,
        os.path.join(args.output_dir, "g2m_features.pt"),
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
        os.path.join(args.output_dir, "m2g_edge_index.pt"),
    )
    torch.save(
        m2g_edge_features,
        os.path.join(args.output_dir, "m2g_features.pt"),
    )

    num_mesh_nodes = grid_con_lat_lon.shape[0]
    print(
        f"Created graph with {num_grid_nodes} grid nodes "
        f"connected to {num_mesh_nodes}"
    )
    print(f"#grid / #mesh = {num_grid_nodes/num_mesh_nodes :.2f}")


if __name__ == "__main__":
    main()
