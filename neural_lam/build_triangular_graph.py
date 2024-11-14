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

    # TODO Get latlons from somewhere, use args.data_config
    example_dir = "example_lam_latlons"
    interior_latlons = np.load(os.path.join(example_dir, "nwp_latlon.npy"))
    boundary_latlons = np.load(os.path.join(example_dir, "o80_latlon.npy"))

    # Load grid positions
    fields_group = zarr.open(fields_group_path, mode="r")
    grid_lat = np.array(
        fields_group["latitude"], dtype=np.float32
    )  # (num_lat,)
    grid_lon = np.array(
        fields_group["longitude"], dtype=np.float32
    )  # (num_long,)

    # Create lat-lon grid
    grid_lat_lon = np.stack(
        (
            np.expand_dims(grid_lat, 0).repeat(
                len(grid_lon), 0
            ),  # (num_lat, num_long)
            np.expand_dims(grid_lon, 1).repeat(
                len(grid_lat), 1
            ),  # (num_lon, num_lat)
        ),
        axis=2,
    )  # (num_lon, num_lat, 2)
    num_lon, num_lat, _ = grid_lat_lon.shape
    grid_lat_lon_flat = grid_lat_lon.reshape(-1, 2)
    num_grid_nodes = grid_lat_lon_flat.shape[0]
    # flattened, (num_grid_nodes, 2)

    grid_lat_lon_torch = torch.tensor(grid_lat_lon_flat, dtype=torch.float32)
    # Save in graph dir?
    torch.save(
        grid_lat_lon_torch, os.path.join(graph_dir_path, "grid_lat_lon.pt")
    )

    if args.hierarchical:
        # Save up+down edge index + features to disk
        torch.save(
            mesh_up_ei_list,
            os.path.join(graph_dir_path, "mesh_up_edge_index.pt"),
        )
        torch.save(
            mesh_down_ei_list,
            os.path.join(graph_dir_path, "mesh_down_edge_index.pt"),
        )
        torch.save(
            mesh_up_features_list,
            os.path.join(graph_dir_path, "mesh_up_features.pt"),
        )
        torch.save(
            mesh_down_features_list,
            os.path.join(graph_dir_path, "mesh_down_features.pt"),
        )
    else:
        gcreate.create_multiscale_mesh(args.splits, args.levels)

    m2m_edge_index_list = []
    m2m_features_list = []
    mesh_features_list = []
    mesh_lat_lon_list = []
    for mesh_graph in m2m_graphs:
        mesh_edge_index = np.stack(
            gc_im.faces_to_edges(mesh_graph.faces), axis=0
        )
        m2m_edge_index_list.append(mesh_edge_index)

        # Compute features
        mesh_lat_lon = vertice_cart_to_lat_lon(mesh_graph.vertices)  # (N, 2)
        mesh_features, m2m_features = gc_mu.get_graph_spatial_features(
            node_lat=mesh_lat_lon[:, 0],
            node_lon=mesh_lat_lon[:, 1],
            senders=mesh_edge_index[0, :],
            receivers=mesh_edge_index[1, :],
            **GC_SPATIAL_FEATURES_KWARGS,
        )
        mesh_features_list.append(mesh_features)
        m2m_features_list.append(m2m_features)
        mesh_lat_lon_list.append(mesh_lat_lon)

        # Check that indexing is correct
        _, mesh_theta = gc_mu.lat_lon_deg_to_spherical(
            mesh_lat_lon[:, 0],
            mesh_lat_lon[:, 1],
        )
        assert np.sum(np.abs(mesh_features[:, 0] - np.cos(mesh_theta))) <= 1e-10

    # Convert to torch
    m2m_edge_index_torch = [
        torch.tensor(mesh_ei, dtype=torch.long)
        for mesh_ei in m2m_edge_index_list
    ]
    m2m_features_torch = [
        torch.tensor(m2m_features, dtype=torch.float32)
        for m2m_features in m2m_features_list
    ]
    mesh_features_torch = [
        torch.tensor(mesh_features, dtype=torch.float32)
        for mesh_features in mesh_features_list
    ]
    mesh_lat_lon_torch = [
        torch.tensor(mesh_lat_lon, dtype=torch.float32)
        for mesh_lat_lon in mesh_lat_lon_list
    ]
    # Save to static dir
    torch.save(
        m2m_edge_index_torch,
        os.path.join(graph_dir_path, "m2m_edge_index.pt"),
    )
    torch.save(
        m2m_features_torch,
        os.path.join(graph_dir_path, "m2m_features.pt"),
    )
    torch.save(
        mesh_features_torch,
        os.path.join(graph_dir_path, "mesh_features.pt"),
    )
    torch.save(
        mesh_lat_lon_torch,
        os.path.join(graph_dir_path, "mesh_lat_lon.pt"),
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
        senders_node_lat=grid_lat_lon_flat[:, 0],
        senders_node_lon=grid_lat_lon_flat[:, 1],
        senders=g2m_edge_index[0, :],
        receivers_node_lat=grid_con_mesh_lat_lon[:, 0],
        receivers_node_lon=grid_con_mesh_lat_lon[:, 1],
        receivers=g2m_edge_index[1, :],
        **GC_SPATIAL_FEATURES_KWARGS,
    )
    g2m_features_torch = torch.tensor(g2m_features, dtype=torch.float32)

    torch.save(
        g2m_edge_index_torch,
        os.path.join(graph_dir_path, "g2m_edge_index.pt"),
    )
    torch.save(
        g2m_features_torch,
        os.path.join(graph_dir_path, "g2m_features.pt"),
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
        receivers_node_lat=grid_lat_lon_flat[:, 0],
        receivers_node_lon=grid_lat_lon_flat[:, 1],
        receivers=m2g_edge_index[1, :],
        **GC_SPATIAL_FEATURES_KWARGS,
    )
    m2g_features_torch = torch.tensor(m2g_features, dtype=torch.float32)

    torch.save(
        m2g_edge_index_torch,
        os.path.join(graph_dir_path, "m2g_edge_index.pt"),
    )
    torch.save(
        m2g_features_torch,
        os.path.join(graph_dir_path, "m2g_features.pt"),
    )

    num_mesh_nodes = grid_con_mesh_lat_lon.shape[0]
    print(
        f"Created graph with {num_grid_nodes} grid nodes "
        f"connected to {num_mesh_nodes}"
    )
    print(f"#grid / #mesh = {num_grid_nodes/num_mesh_nodes :.2f}")


if __name__ == "__main__":
    main()
