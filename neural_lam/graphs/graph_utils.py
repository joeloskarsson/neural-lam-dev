# Third-party
import numpy as np
import scipy
from graphcast import graphcast as gc_gc
from graphcast import model_utils as gc_mu


def node_cart_to_lat_lon(node_pos_cart):
    """
    Convert node positions to lat-lon

    node_pos_cart: (N_nodes, 3), cartesian coordinates
    Returns: (N_nodes, 2), lat-lon coordinates
    """
    phi, theta = gc_mu.cartesian_to_spherical(
        node_pos_cart[:, 0], node_pos_cart[:, 1], node_pos_cart[:, 2]
    )
    (
        nodes_lat,
        nodes_lon,
    ) = gc_mu.spherical_to_lat_lon(phi=phi, theta=theta)
    return np.stack((nodes_lon, nodes_lat), axis=1)  # (N, 2)


def node_lat_lon_to_cart(node_lat_lon):
    """
    Convert node positions from lat-lon to cartesian

    NOTE: Based on graphcast.grid_mesh_connectivity._grid_lat_lon_to_coordinates

    node_pos_lat_lon: (N_nodes, 3), cartesian coordinates
    Returns: (N_nodes, 2), lat-lon coordinates
    """
    phi_grid = np.deg2rad(node_lat_lon[:, 0])
    theta_grid = np.deg2rad(90 - node_lat_lon[:, 1])

    return np.stack(
        [
            np.cos(phi_grid) * np.sin(theta_grid),
            np.sin(phi_grid) * np.sin(theta_grid),
            np.cos(theta_grid),
        ],
        axis=-1,
    )


def radius_query_indices_irregular(
    *,
    grid_lat_lon: np.ndarray,
    mesh: gc_gc.icosahedral_mesh.TriangularMesh,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns mesh-grid edge indices for radius query.

    NOTE: This is a modified version of graphcast.radius_query_indices that does
    not assume grid coordinates to be on a regular lat-lon grid. It thus
    directly takes a the lat-lon position of all grid nodes as input

    Args:
        grid_lat_lon: Lat-lon positions for the grid [num_grid_points, 2]
        mesh: Mesh object.
        radius: Radius of connectivity in R3. for a sphere of unit radius.

    Returns:
        tuple with `grid_indices` and `mesh_indices` indicating edges between
        the grid and the mesh such that the distances in a straight line (not
        geodesic) are smaller than or equal to `radius`.
        * grid_indices: Indices of shape [num_grid_points], that index into a
          [num_grid_points, ...] array of grid positions.
        * mesh_indices: Indices of shape [num_edges], that index into
          mesh.vertices.
    """

    # [num_grid_points=num_lat_points * num_lon_points, 3]
    grid_positions = node_lat_lon_to_cart(grid_lat_lon)

    # [num_mesh_points, 3]
    mesh_positions = mesh.vertices
    kd_tree = scipy.spatial.cKDTree(mesh_positions)

    # [num_grid_points, num_mesh_points_per_grid_point]
    # Note `num_mesh_points_per_grid_point` is not constant, so this is a list
    # of arrays, rather than a 2d array.
    query_indices = kd_tree.query_ball_point(x=grid_positions, r=radius)

    grid_edge_indices = []
    mesh_edge_indices = []
    for grid_index, mesh_neighbors in enumerate(query_indices):
        grid_edge_indices.append(np.repeat(grid_index, len(mesh_neighbors)))
        mesh_edge_indices.append(mesh_neighbors)

    # [num_edges]
    grid_edge_indices = np.concatenate(grid_edge_indices, axis=0).astype(int)
    mesh_edge_indices = np.concatenate(mesh_edge_indices, axis=0).astype(int)

    return grid_edge_indices, mesh_edge_indices
