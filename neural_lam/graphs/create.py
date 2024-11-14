# Third-party
import numpy as np
import scipy
import torch
import zarr
from graphcast import graphcast as gc_gc
from graphcast import grid_mesh_connectivity as gc_gm
from graphcast import icosahedral_mesh as gc_im
from graphcast import model_utils as gc_mu

# First-party
import neural_lam.graphs.graph_utils as gutils

# Keyword arguments to use when calling graphcast functions
# for creating graph features
GC_SPATIAL_FEATURES_KWARGS = {
    "add_node_positions": False,
    "add_node_latitude": True,
    "add_node_longitude": True,
    "add_relative_positions": True,
    "relative_longitude_local_coordinates": True,
    "relative_latitude_local_coordinates": True,
}


def inter_mesh_connection(from_mesh, to_mesh):
    """
    Connect from_mesh to to_mesh
    """
    kd_tree = scipy.spatial.cKDTree(to_mesh.vertices)

    # Each node on lower (from) mesh will connect to 1 or 2 on level above
    # pylint: disable-next=protected-access
    radius = 1.1 * gc_gc._get_max_edge_distance(from_mesh)
    query_indices = kd_tree.query_ball_point(x=from_mesh.vertices, r=radius)

    from_edge_indices = []
    to_edge_indices = []
    for from_index, to_neighbors in enumerate(query_indices):
        from_edge_indices.append(np.repeat(from_index, len(to_neighbors)))
        to_edge_indices.append(to_neighbors)

    from_edge_indices = np.concatenate(from_edge_indices, axis=0).astype(int)
    to_edge_indices = np.concatenate(to_edge_indices, axis=0).astype(int)

    edge_index = np.stack(
        (from_edge_indices, to_edge_indices), axis=0
    )  # (2, M)
    return edge_index


def _create_mesh_levels(splits, levels=None):
    """
    Create a sequence of mesh graph levels by splitting a global icosahedron

    splits: int, number of times to split icosahedron
    levels: int, number of levels to keep (from finest resolution and up)
        if None, keep all levels
    """
    # Mesh, index 0 is initial graph, with longest edges
    mesh_list = gc_im.get_hierarchy_of_triangular_meshes_for_sphere(splits)
    if levels is not None:
        assert (
            levels <= splits + 1
        ), f"Can not keep {levels} levels when doing {splits} splits"
        mesh_list = mesh_list[-levels:]

    return mesh_list


def create_multiscale_mesh(splits, levels):
    """
    Create a multiscale triangular mesh graph

    splits: int, number of times to split icosahedron
    levels: int, number of levels to keep (from finest resolution and up)

    Returns: graphcast.icosahedral_mesh.TriangularMesh, the merged mesh
    """
    mesh_list = _create_mesh_levels(splits, levels)

    # Merge meshes
    # Modify gc code, as this uses some python 3.10 things
    for mesh_i, mesh_ip1 in zip(mesh_list[:-1], mesh_list[1:]):
        # itertools.pairwise(mesh_list):
        num_nodes_mesh_i = mesh_i.vertices.shape[0]
        assert np.allclose(
            mesh_i.vertices, mesh_ip1.vertices[:num_nodes_mesh_i]
        )

    merged_mesh = gc_im.TriangularMesh(
        vertices=mesh_list[-1].vertices,
        faces=np.concatenate([mesh.faces for mesh in mesh_list], axis=0),
    )

    return merged_mesh


def create_hierarchical_mesh(splits, levels):
    """
    Create a hierarchical triangular mesh graph

    splits: int, number of times to split icosahedron
    levels: int, number of levels to keep (from finest resolution and up)

    Returns: list of graphcast.icosahedral_mesh.TriangularMesh, the merged mesh
    """
    mesh_list = _create_mesh_levels(splits, levels)

    mesh_list_rev = list(reversed(mesh_list))  # 0 is finest graph now
    m2m_graphs = mesh_list_rev  # list of num_splitgraphs

    # Up and down edges for hierarchy
    # Reuse code for connecting grid to mesh?
    mesh_up_ei_list = []
    mesh_down_ei_list = []
    mesh_up_features_list = []
    mesh_down_features_list = []
    for from_mesh, to_mesh in zip(mesh_list_rev[:-1], mesh_list_rev[1:]):
        mesh_up_ei = inter_mesh_connection(from_mesh, to_mesh)
        # Down is opposite direction of up
        mesh_down_ei = np.stack((mesh_up_ei[1, :], mesh_up_ei[0, :]), axis=0)
        mesh_up_ei_list.append(torch.tensor(mesh_up_ei, dtype=torch.long))
        mesh_down_ei_list.append(torch.tensor(mesh_down_ei, dtype=torch.long))

        from_mesh_lat_lon = gutils.vertice_cart_to_lat_lon(
            from_mesh.vertices
        )  # (N, 2)
        to_mesh_lat_lon = gutils.vertice_cart_to_lat_lon(
            to_mesh.vertices
        )  # (N, 2)

        # Extract features for hierarchical edges
        _, _, mesh_up_features = gc_mu.get_bipartite_graph_spatial_features(
            senders_node_lat=from_mesh_lat_lon[:, 0],
            senders_node_lon=from_mesh_lat_lon[:, 1],
            senders=mesh_up_ei[0, :],
            receivers_node_lat=to_mesh_lat_lon[:, 0],
            receivers_node_lon=to_mesh_lat_lon[:, 1],
            receivers=mesh_up_ei[1, :],
            **GC_SPATIAL_FEATURES_KWARGS,
        )
        _, _, mesh_down_features = gc_mu.get_bipartite_graph_spatial_features(
            senders_node_lat=to_mesh_lat_lon[:, 0],
            senders_node_lon=to_mesh_lat_lon[:, 1],
            senders=mesh_down_ei[0, :],
            receivers_node_lat=from_mesh_lat_lon[:, 0],
            receivers_node_lon=from_mesh_lat_lon[:, 1],
            receivers=mesh_down_ei[1, :],
            **GC_SPATIAL_FEATURES_KWARGS,
        )
        mesh_up_features_list.append(
            torch.tensor(mesh_up_features, dtype=torch.float32)
        )
        mesh_down_features_list.append(
            torch.tensor(mesh_down_features, dtype=torch.float32)
        )


def extract_mesh_graph_features(mesh_graph: gc_im.TriangularMesh):
    """
    Extract torch tensors for edge_index and features from single TriangularMesh
    """
    mesh_edge_index = np.stack(gc_im.faces_to_edges(mesh_graph.faces), axis=0)

    # Compute features
    mesh_lat_lon = gutils.vertice_cart_to_lat_lon(mesh_graph.vertices)  # (N, 2)
    mesh_node_features, mesh_edge_features = gc_mu.get_graph_spatial_features(
        node_lat=mesh_lat_lon[:, 0],
        node_lon=mesh_lat_lon[:, 1],
        senders=mesh_edge_index[0, :],
        receivers=mesh_edge_index[1, :],
        **GC_SPATIAL_FEATURES_KWARGS,
    )

    return (
        torch.tensor(mesh_edge_index, dtype=torch.float32),
        torch.tensor(mesh_node_features, dtype=torch.float32),
        torch.tensor(mesh_edge_features, dtype=torch.float32),
        torch.tensor(mesh_lat_lon, dtype=torch.float32),
    )

    # TODO Move below to a test?
    # Check that indexing is correct
    #  _, mesh_theta = gc_mu.lat_lon_deg_to_spherical(
    #  mesh_lat_lon[:, 0],
    #  mesh_lat_lon[:, 1],
    #  )
    #  assert np.sum(np.abs(mesh_features[:, 0] - np.cos(mesh_theta))) <= 1e-1
