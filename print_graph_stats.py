from argparse import ArgumentParser

from neural_lam import utils
from neural_lam .config import load_config_and_datastores

def main():
    parser = ArgumentParser(
        description="Print graph stats"
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
        help="Graph to load and use in graph-based model (default: multiscale)",
    )
    args = parser.parse_args()

    # Load neural-lam configuration and datastore to use
    config, datastore, datastore_boundary = load_config_and_datastores(
        config_path=args.config_path
    )

    graph_dir_path = datastore.root_path / "graphs" / args.graph_name
    hierarchical, graph_ldict = utils.load_graph(
        graph_dir_path=graph_dir_path,
        datastore=datastore,
    )

    print(
        f"Edges in subgraphs: g2m={graph_ldict['g2m_features'].shape[0]}, "
        f"m2g={graph_ldict['m2g_features'].shape[0]}"
    )

    num_levels = len(graph_ldict['mesh_static_features'])

    # Number of mesh nodes at each level
    level_mesh_sizes = [
        mesh_feat.shape[0] for mesh_feat in graph_ldict['mesh_static_features']
    ]  # Needs as python list for later


    if hierarchical:
        m2m_features = graph_ldict['m2m_features']
        mesh_up_features = graph_ldict['mesh_up_features']
        mesh_down_features = graph_ldict['mesh_down_features']

        print("Loaded hierarchical graph with structure:")
        for level_index, level_mesh_size in enumerate(level_mesh_sizes):
            same_level_edges = m2m_features[level_index].shape[0]
            print(
                f"level {level_index} - {level_mesh_size} nodes, "
                f"{same_level_edges} same-level edges"
            )

            if level_index < (num_levels - 1):
                up_edges = mesh_up_features[level_index].shape[0]
                down_edges = mesh_down_features[level_index].shape[0]
                print(f"  {level_index}<->{level_index + 1}")
                print(f" - {up_edges} up edges, {down_edges} down edges")


if __name__ == "__main__":
    main()
