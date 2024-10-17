# Third-party
import numpy as np
from graphcast import model_utils as gc_mu


def vertice_cart_to_lat_lon(vertices):
    """
    Convert vertice positions to lat-lon

    vertices: (N_vert, 3), cartesian coordinates
    Returns: (N_vert, 2), lat-lon coordinates
    """
    phi, theta = gc_mu.cartesian_to_spherical(
        vertices[:, 0], vertices[:, 1], vertices[:, 2]
    )
    (
        nodes_lat,
        nodes_lon,
    ) = gc_mu.spherical_to_lat_lon(phi=phi, theta=theta)
    return np.stack((nodes_lat, nodes_lon), axis=1)  # (N, 2)
