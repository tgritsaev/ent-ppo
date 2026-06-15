EXCLUDED_NODES = {
    "alpha": [],
    "capella": [],
}


def excluded_nodes_csv(cluster: str) -> str:
    return ",".join(EXCLUDED_NODES.get(cluster, []))
