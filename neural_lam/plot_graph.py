# Standard library
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

# Third-party
import cartopy.crs as ccrs
import numpy as np
import plotly.graph_objects as go

# Local
from . import utils
from .config import load_config_and_datastores
from .graphs import graph_utils as gutils

GRID_RADIUS = 1


def create_edge_plot(
    edge_index,
    from_node_lat_lon,
    to_node_lat_lon,
    label,
    color="blue",
    width=1,
    from_radius=1,
    to_radius=1,
):
    """
    Create a plotly object showing edges

    edge_index: (2, M)
    from_node_lat_lon: (N, 2), positions of sender nodes
    to_node_lat_lon: (N, 2), positions of receiver nodes
    label: str, label of plot object
    """
    from_node_cart = (
        gutils.node_lat_lon_to_cart(from_node_lat_lon) * from_radius
    )
    to_node_cart = gutils.node_lat_lon_to_cart(to_node_lat_lon) * to_radius

    edge_start = from_node_cart[edge_index[0]]  # (M, 2)
    edge_end = to_node_cart[edge_index[1]]  # (M, 2)
    n_edges = edge_start.shape[0]

    x_edges = np.stack(
        (edge_start[:, 0], edge_end[:, 0], np.full(n_edges, None)), axis=1
    ).flatten()
    y_edges = np.stack(
        (edge_start[:, 1], edge_end[:, 1], np.full(n_edges, None)), axis=1
    ).flatten()
    z_edges = np.stack(
        (edge_start[:, 2], edge_end[:, 2], np.full(n_edges, None)), axis=1
    ).flatten()

    return go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode="lines",
        line={"color": color, "width": width},
        name=label,
    )


def create_node_plot(node_lat_lon, label, color="blue", size=1, radius=1):
    """
    Create a plotly object showing nodes

    node_lat_lon: (N, 2)
    label: str, label of plot object
    """
    node_pos = gutils.node_lat_lon_to_cart(node_lat_lon) * radius
    return go.Scatter3d(
        x=node_pos[:, 0],
        y=node_pos[:, 1],
        z=node_pos[:, 2],
        mode="markers",
        marker={"color": color, "size": size},
        name=label,
    )


def main():
    """Plot graph structure in 3D using plotly."""
    parser = ArgumentParser(
        description="Plot graph",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the configuration for neural-lam",
    )
    parser.add_argument(
        "--graph_name",
        type=str,
        default="multiscale",
        help="Name of saved graph to plot",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Name of .html file to save interactive plot to",
    )
    parser.add_argument(
        "--show_axis",
        action="store_true",
        help="If the axis should be displayed",
    )
    # Geometry
    parser.add_argument(
        "--mesh_height",
        type=float,
        default=0.1,
        help="Height of mesh over grid",
    )
    parser.add_argument(
        "--mesh_level_dist",
        type=float,
        default=0.3,
        help="Distance between mesh levels",
    )
    parser.add_argument(
        "--edge_width",
        type=float,
        default=0.4,
        help="Width of edges",
    )
    parser.add_argument(
        "--grid_node_size",
        type=float,
        default=1.0,
        help="Size of grid nodes",
    )
    parser.add_argument(
        "--mesh_node_size",
        type=float,
        default=2.5,
        help="Size of mesh nodes",
    )
    # Colors
    parser.add_argument(
        "--g2m_color",
        type=str,
        default="black",
        help="Color of g2m edges",
    )
    parser.add_argument(
        "--m2g_color",
        type=str,
        default="black",
        help="Color of m2g edges",
    )
    parser.add_argument(
        "--grid_color",
        type=str,
        default="black",
        help="Color of grid nodes",
    )
    parser.add_argument(
        "--mesh_color",
        type=str,
        default="blue",
        help="Color of mesh nodes and edges",
    )

    args = parser.parse_args()

    assert (
        args.config_path is not None
    ), "Specify your config with --config_path"

    _, datastore, datastore_boundary = load_config_and_datastores(
        config_path=args.config_path
    )

    # Load graph data
    graph_dir_path = os.path.join(
        datastore.root_path, "graphs", args.graph_name
    )
    hierarchical, graph_ldict = utils.load_graph(graph_dir_path=graph_dir_path)
    # Turn all to numpy
    (g2m_edge_index, m2g_edge_index) = (
        graph_ldict["g2m_edge_index"].numpy(),
        graph_ldict["m2g_edge_index"].numpy(),
    )

    # Plotting is in 3d, with lat-lons
    grid_lat_lon = utils.get_stacked_lat_lons(datastore, datastore_boundary)
    # (num_nodes_full, 3)

    # Add plotting objects to this list
    data_objs = []

    # Plot grid nodes
    data_objs.append(
        create_node_plot(
            grid_lat_lon,
            "Grid Nodes",
            color=args.grid_color,
            radius=GRID_RADIUS,
            size=args.grid_node_size,
        )
    )

    # Radius
    mesh_radius = GRID_RADIUS + args.mesh_height

    # Mesh positioning and edges to plot differ if we have a hierarchical graph
    if hierarchical:
        # TODO Should this really be done here?
        # TODO Now these will either be in-proj coords or lat-lons depending
        # on what kinds of graph was made
        mesh_lat_lon_level = [
            ccrs.PlateCarree().transform_points(
                datastore.coords_projection,
                mesh_coords[:, 0].numpy(),
                mesh_coords[:, 1].numpy(),
            )
            for mesh_coords in graph_ldict["mesh_static_features"]
        ]

        # Make edge_index to numpy
        m2m_edge_index = [ei.numpy() for ei in graph_ldict["m2m_edge_index"]]
        mesh_up_edge_index = [
            ei.numpy() for ei in graph_ldict["mesh_up_edge_index"]
        ]
        mesh_down_edge_index = [
            ei.numpy() for ei in graph_ldict["mesh_down_edge_index"]
        ]

        # Iterate over levels, adding all nodes and edges
        for bot_level_i, intra_ei in enumerate(
            m2m_edge_index,
        ):
            # Extract position and radius
            top_level_i = bot_level_i + 1
            bot_pos = mesh_lat_lon_level[bot_level_i]
            bot_radius = mesh_radius + bot_level_i * args.mesh_level_dist

            # Mesh nodes at bottom level
            data_objs.append(
                create_node_plot(
                    bot_pos,
                    f"Mesh level {bot_level_i} nodes",
                    color=args.mesh_color,
                    radius=bot_radius,
                    size=args.mesh_node_size,
                )
            )
            # Intra-level edges at bottom level
            data_objs.append(
                create_edge_plot(
                    intra_ei,
                    bot_pos,
                    bot_pos,
                    f"Mesh level {bot_level_i} edges",
                    color=args.mesh_color,
                    width=args.edge_width,
                    from_radius=bot_radius,
                    to_radius=bot_radius,
                )
            )

            # Do add include up/down edges for top level
            if top_level_i < len(m2m_edge_index):
                up_ei = mesh_up_edge_index[bot_level_i]
                down_ei = mesh_down_edge_index[bot_level_i]
                top_pos = mesh_lat_lon_level[top_level_i]
                top_radius = mesh_radius + (top_level_i) * args.mesh_level_dist

                # Up edges
                data_objs.append(
                    create_edge_plot(
                        up_ei,
                        bot_pos,
                        top_pos,
                        f"Mesh up {bot_level_i}->{top_level_i} edges",
                        color=args.mesh_color,
                        width=args.edge_width,
                        from_radius=bot_radius,
                        to_radius=top_radius,
                    )
                )
                # Down edges
                data_objs.append(
                    create_edge_plot(
                        down_ei,
                        top_pos,
                        bot_pos,
                        f"Mesh up {top_level_i}->{bot_level_i} edges",
                        color=args.mesh_color,
                        width=args.edge_width,
                        from_radius=top_radius,
                        to_radius=bot_radius,
                    )
                )

        # Connect g2m and m2g only to bottom level
        grid_con_lat_lon = mesh_lat_lon_level[0]
    else:
        # TODO Should this really be done here?
        # TODO Now these will either be in-proj coords or lat-lons depending
        # on what kinds of graph was made
        mesh_proj_pos = graph_ldict["mesh_static_features"].numpy()
        mesh_lat_lon = ccrs.PlateCarree().transform_points(
            datastore.coords_projection,
            mesh_proj_pos[:, 0],
            mesh_proj_pos[:, 1],
        )

        # Non-hierarchical
        m2m_edge_index = graph_ldict["m2m_edge_index"].numpy()
        # TODO Degree-dependent node size?
        #  mesh_degrees = pyg.utils.degree(m2m_edge_index[1]).numpy()
        #  mesh_node_size = mesh_degrees / 2

        data_objs.append(
            create_node_plot(
                mesh_lat_lon,
                "Mesh Nodes",
                radius=mesh_radius,
                color=args.mesh_color,
                size=args.mesh_node_size,
            )
        )
        data_objs.append(
            create_edge_plot(
                m2m_edge_index,
                mesh_lat_lon,
                mesh_lat_lon,
                "Mesh Edges",
                from_radius=mesh_radius,
                to_radius=mesh_radius,
                color=args.mesh_color,
                width=args.edge_width,
            )
        )

        grid_con_lat_lon = mesh_lat_lon

    # Plot G2M
    data_objs.append(
        create_edge_plot(
            g2m_edge_index,
            grid_lat_lon,
            grid_con_lat_lon,
            "G2M Edges",
            color=args.g2m_color,
            width=args.edge_width,
            from_radius=GRID_RADIUS,
            to_radius=mesh_radius,
        )
    )

    # Plot M2G
    data_objs.append(
        create_edge_plot(
            m2g_edge_index,
            grid_con_lat_lon,
            grid_lat_lon,
            "M2G Edges",
            color=args.m2g_color,
            width=args.edge_width,
            from_radius=mesh_radius,
            to_radius=GRID_RADIUS,
        )
    )

    fig = go.Figure(data=data_objs)

    fig.update_layout(scene_aspectmode="data")
    fig.update_traces(connectgaps=False)

    if not args.show_axis:
        # Hide axis
        fig.update_layout(
            scene={
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "zaxis": {"visible": False},
            }
        )

    if args.save:
        fig.write_html(args.save, include_plotlyjs="cdn")
    else:
        fig.show()


if __name__ == "__main__":
    main()
