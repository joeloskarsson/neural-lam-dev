# Standard library
import argparse
import os

# Third-party
import numpy as np
import scipy
import torch
import zarr
from graphcast import graphcast as gc_gc
from graphcast import grid_mesh_connectivity as gc_gm
from graphcast import icosahedral_mesh as gc_im

# First-party
import neural_lam.graphs.create as gcreate


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
        help="If graphs should be plotted during generation "
        "(default: false)",
    )

    # Graph structure
    parser.add_argument(
        "--splits",
        default=3,
        type=int,
        help="Number of splits to triangular mesh (default: 3)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        help="Number of levels to keep, from finest upwards "
        "(default: None (keep all))",
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Generate hierarchical mesh graph (default: false)",
    )
    args = parser.parse_args()

    assert args.output_dir, "Must specify an --output_dir"
    os.makedirs(args.output_dir, exist_ok=True)

    # TODO Get lat_lon from somewhere, use args.data_config
    example_dir = "example_lam_latlons"
    interior_lat_lon_raw = np.load(os.path.join(example_dir, "nwp_latlon.npy"))
    boundary_lat_lon_raw = np.load(os.path.join(example_dir, "o80_latlon.npy"))

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
    num_grid_nodes = grid_lat_lon.shape[0]
    # flattened, (num_grid_nodes, 2)

    grid_lat_lon_torch = torch.tensor(grid_lat_lon, dtype=torch.float32)
    # TODO: Save in graph dir?
    torch.save(
        grid_lat_lon_torch, os.path.join(args.output_dir, "grid_lat_lon.pt")
    )

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
        pass
        # TODO Hierarchical graph
    else:
        merged_mesh = gcreate.create_multiscale_mesh(args.splits, args.levels)
        m2m_graphs = [merged_mesh]

    mesh_graph_features = [
        gcreate.extract_mesh_graph_features(mesh_graph)
        for mesh_graph in m2m_graphs
    ]
    # Ordering: edge_index, node_features, edge_features, lat_lon

    # Save to static dir
    for feat_index, file_name in enumerate(
        (
            "m2m_edge_index.pt",
            "m2m_features.pt",
            "mesh_features.pt",
            "mesh_lat_lon.pt",
        )
    ):
        torch.save(
            [feats[feat_index] for feats in mesh_graph_features],
            os.path.join(args.output_dir, file_name),
        )

    if args.plot:
        for level_i, (m2m_edge_index, mesh_lat_lon) in enumerate(
            zip(m2m_edge_index_torch, mesh_lat_lon_torch)
        ):
            plot_graph(
                m2m_edge_index, mesh_lat_lon, title=f"Mesh level {level_i}"
            )
            plt.show()

    # Because GC code returns indexes into flattened lat-lon matrix, we have to
    # re-map grid indices. We always work with lon-lat order, to be consistent
    # with WB2 data.
    # This creates the correct mapping for the grid indices
    grid_index_map = (
        torch.arange(num_grid_nodes).reshape(num_lon, num_lat).T.flatten()
    )

    # Grid2Mesh: Radius-based
    grid_con_mesh = m2m_graphs[0]  # Mesh graph that should be connected to grid
    grid_con_mesh_lat_lon = mesh_lat_lon_list[0]

    # Compute maximum edge distance in finest mesh
    # pylint: disable-next=protected-access
    max_mesh_edge_len = gc_gc._get_max_edge_distance(mesh_list[-1])
    g2m_connect_radius = 0.6 * max_mesh_edge_len
    g2m_grid_mesh_indices = gc_gm.radius_query_indices(
        grid_latitude=grid_lat,
        grid_longitude=grid_lon,
        mesh=grid_con_mesh,
        radius=g2m_connect_radius,
    )
    # Returns two arrays of node indices, each [num_edges]

    g2m_edge_index = np.stack(g2m_grid_mesh_indices, axis=0)
    g2m_edge_index_torch = torch.tensor(g2m_edge_index, dtype=torch.long)
    # Grid index fix
    g2m_edge_index_torch[0] = grid_index_map[g2m_edge_index_torch[0]]

    # Only care about edge features here
    _, _, g2m_features = gc_mu.get_bipartite_graph_spatial_features(
        senders_node_lat=grid_lat_lon[:, 0],
        senders_node_lon=grid_lat_lon[:, 1],
        senders=g2m_edge_index[0, :],
        receivers_node_lat=grid_con_mesh_lat_lon[:, 0],
        receivers_node_lon=grid_con_mesh_lat_lon[:, 1],
        receivers=g2m_edge_index[1, :],
        **GC_SPATIAL_FEATURES_KWARGS,
    )
    g2m_features_torch = torch.tensor(g2m_features, dtype=torch.float32)

    torch.save(
        g2m_edge_index_torch,
        os.path.join(args.output_dir, "g2m_edge_index.pt"),
    )
    torch.save(
        g2m_features_torch,
        os.path.join(args.output_dir, "g2m_features.pt"),
    )

    # Mesh2Grid: Connect to containing mesh triangle
    m2g_grid_mesh_indices = gc_gm.in_mesh_triangle_indices(
        grid_latitude=grid_lat,
        grid_longitude=grid_lon,
        mesh=mesh_list[-1],
    )  # Note: Still returned in order (grid, mesh), need to inverse
    m2g_edge_index = np.stack(m2g_grid_mesh_indices[::-1], axis=0)
    m2g_edge_index_torch = torch.tensor(m2g_edge_index, dtype=torch.long)
    # Grid index fix
    m2g_edge_index_torch[1] = grid_index_map[m2g_edge_index_torch[1]]

    # Only care about edge features here
    _, _, m2g_features = gc_mu.get_bipartite_graph_spatial_features(
        senders_node_lat=grid_con_mesh_lat_lon[:, 0],
        senders_node_lon=grid_con_mesh_lat_lon[:, 1],
        senders=m2g_edge_index[0, :],
        receivers_node_lat=grid_lat_lon[:, 0],
        receivers_node_lon=grid_lat_lon[:, 1],
        receivers=m2g_edge_index[1, :],
        **GC_SPATIAL_FEATURES_KWARGS,
    )
    m2g_features_torch = torch.tensor(m2g_features, dtype=torch.float32)

    torch.save(
        m2g_edge_index_torch,
        os.path.join(args.output_dir, "m2g_edge_index.pt"),
    )
    torch.save(
        m2g_features_torch,
        os.path.join(args.output_dir, "m2g_features.pt"),
    )

    num_mesh_nodes = grid_con_mesh_lat_lon.shape[0]
    print(
        f"Created graph with {num_grid_nodes} grid nodes "
        f"connected to {num_mesh_nodes}"
    )
    print(f"#grid / #mesh = {num_grid_nodes/num_mesh_nodes :.2f}")


if __name__ == "__main__":
    main()
