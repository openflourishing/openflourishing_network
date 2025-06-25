"""Main openflourishing.org network analysis."""

from __future__ import annotations

import itertools
import json
import json.tool
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from . import convert


def as_int(value: float) -> int:
    """Deal with nan values for integers.

    Args:
        value (float): The value, typically a float.

    Returns:
        int: The integer
    """
    if np.isnan(value):
        return 99999
    return int(value)


def default_link(submission_id: int, parent: int | None) -> dict:
    """Create a default link dictionary.

    Args:
        submission_id (int): The submission id.
        parent (int | None): The parent term in the link.

    Returns:
        dict: The link dictionary.
    """
    link = {
        "submission_id": submission_id,
        "parent": parent,
        "relationship": "includes",
        "linked": set(),
    }
    return link


def not_na(value: float) -> float | None:
    """Check for NA values."""
    if pd.isna(value):
        return None
    return value


def get_stopwords():
    root = Path(__file__).parent.parent
    fname = root / "data" / "stopwords.json"
    with open(fname) as f:
        stopwords = json.load(f)
    return stopwords


def remove_stopwords(terms: set, links: list[dict]) -> None:
    """Remove stopwords in terms and links.

    Args:
        terms (set): The set of terms.
        links (list[dict]): The list of link dictionaries.
    """
    stopwords = get_stopwords()
    terms.difference_update(stopwords)
    for dct in links:
        if dct["parent"] in stopwords:
            dct["parent"] = None
        dct["linked"].difference_update(stopwords)


def _add_edge(
    edges: dict[tuple[int, int], float],
    source: int,
    target: int,
    weight: float,
) -> None:
    """Add an edge.

    Args:
        edges (dict): The edges dictionary, where keys are (source, target)
            tuples, and values are weights.
        source (int): Source node id.
        target (int): Target node id.
        weight (float): Additional edge weight to add.
    """
    if source > target:
        source, target = target, source
    if (source, target) not in edges:
        edges[(source, target)] = weight
    edges[(source, target)] += weight


def create_network_data(terms: set, links: list[dict]) -> tuple[dict, dict]:
    """Create the network data from terms and links.

    Args:
        terms (set): All terms.
        links (list[dict]): The list of link dictionaries.

    Returns:
        tuple:
            - nodes (dict): Dictionary of nodes with ids.
            - edges (dict): Dictionary of edges keyed by (source, target)
                tuples, with weights as values.
    """
    nodes = {
        t: {"term": t, "index": i + 1, "submissions": set()}
        for i, t in enumerate(terms)
    }
    edges = {}
    stopwords = get_stopwords()
    for dct in links:
        linked = dct["linked"]
        parent = dct["parent"]
        submission_id = str(dct["submission_id"])
        if parent is not None and parent not in stopwords:
            for term in linked:
                _add_edge(
                    edges,
                    nodes[parent]["index"],
                    nodes[term]["index"],
                    20.0 / len(linked),
                )
                nodes[term]["submissions"].add(submission_id)
        combinations = list(itertools.combinations(linked, 2))
        for source, target in combinations:
            _add_edge(
                edges,
                nodes[source]["index"],
                nodes[target]["index"],
                10 / len(combinations),
            )
            nodes[source]["submissions"].add(submission_id)
            nodes[target]["submissions"].add(submission_id)

    # combinations = itertools.combinations(terms, 2)
    # for source, target in combinations:
    #     _add_edge(edges, nodes[source], nodes[target], 0.01)
    for node_dct in nodes.values():
        node_dct["submissions"] = ";".join(node_dct["submissions"])
    return nodes, edges


def make_nodes_df(G: nx.Graph) -> pd.DataFrame:
    """Make the nodes dataframe.

    Args:
        G (nx.Graph): The graph.

    Returns:
        pd.DataFrame: Nodes dataframe.
    """
    node_data = []
    for _id, node_dct in G.nodes.items():
        dct = {"ID": _id, "Label": node_dct["term"]}
        dct.update(node_dct)
        node_data.append(dct)
    df = pd.DataFrame(node_data)
    cols = df.columns
    df.columns = [col.title() for col in cols]
    return df


def make_edges_df(G: nx.Graph) -> pd.DataFrame:
    """Make the edges dataframe.

    Args:
        G (nx.Graph): The graph.

    Returns:
        pd.DataFrame: Edges dataframe.
    """
    edge_data = []
    for i, (source, target, edge_dct) in enumerate(G.edges(data=True)):
        dct = {"ID": i, "Source": source, "Target": target}
        dct.update(edge_dct)
        edge_data.append(dct)
    df = pd.DataFrame(edge_data)
    cols = df.columns
    df.columns = [col.title() for col in cols]
    return df


def create_csvs(output_dir: Path, G: nx.Graph, suffix: str = "") -> None:
    """Output csvs of the network.

    Args:
        G (nx.Graph): The graph.
    """
    nodes_df = make_nodes_df(G)
    edges_df = make_edges_df(G)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    fname = output_dir / (
        f"openflourishing_{timestamp}" + suffix + "_csv_nodes.csv"
    )
    nodes_df.to_csv(fname, index=None)
    fname = output_dir / (
        f"openflourishing_{timestamp}" + suffix + "_csv_edges.csv"
    )
    edges_df.to_csv(fname, index=None)


def create_networkx_graph(nodes: dict, edges: dict) -> nx.Graph:
    """Create networkx graph from nodes and edges.

    Args:
        nodes (dict): Dictionary of nodes, with keys of terms and values
            are unique ids.
        edges (dict): Dictionary of (source, target) keys and weights as
            values.

    Returns:
        nx.Graph: The graph.
    """
    G = nx.Graph()
    node_data = []
    for term, node_dct in nodes.items():
        ndct = {
            "label": term,
            "term": term,
            "submissions": node_dct["submissions"],
        }
        node_data.append((node_dct["index"], ndct))
    G.add_nodes_from(node_data)
    edge_data = []
    for (source, target), weight in edges.items():
        edge_data.append((source, target, {"weight": weight}))
    G.add_edges_from(edge_data)
    degree_view = G.degree(weight="weight")
    for node, weighted_degree in degree_view:
        G.nodes[node]["weighted_degree"] = weighted_degree
    return G


def detect_communities(G, seed, resolution):
    community_sets = nx.algorithms.community.louvain_communities(
        G, weight="weight", seed=seed, resolution=resolution
    )
    for i, community_set in enumerate(community_sets):
        for node in community_set:
            G.nodes[node]["community"] = i


def write_graphml(output_dir: Path, G: nx.Graph) -> None:
    """Write the graph to a GraphML file.

    Args:
        G (nx.Graph): The graph.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    fname = output_dir / f"openflourishing_{timestamp}_network.graphml"
    nx.readwrite.write_graphml(G, fname)


def write_gexf(output_dir: Path, G: nx.Graph, suffix="") -> None:
    """Write the graph to a GraphML file.

    Args:
        G (nx.Graph): The graph.
    """
    output_dir = Path.cwd() / "output"
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    fname = output_dir / (
        f"openflourishing_{timestamp}_network" + suffix + ".gexf"
    )
    nx.readwrite.write_gexf(G, fname)


def G_to_dict(G: nx.Graph) -> dict:
    """Return the network in json format.

    Args:
        G (nx.Graph): The network.
    """
    return nx.readwrite.json_graph.node_link_data(G, edges="links")


def write_json(output_dir: Path, G: nx.Graph, suffix: str = "") -> None:
    """Write the network to json.

    Args:
        G (nx.Graph): The network.
    """
    dct = G_to_dict(G)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    fname = output_dir / (
        f"openflourishing_{timestamp}_network" + suffix + ".json"
    )
    with open(fname, "w") as f:
        json.dump(dct, f, indent=4)


def reorient(pos, G):
    term_mapping = nx.get_node_attributes(G, "term")
    inv_mapping = {v: k for k, v in term_mapping.items()}
    centre = pos[inv_mapping["Happiness"]]
    down = pos[inv_mapping["Purpose"]] - pos[inv_mapping["Happiness"]]
    theta = -np.arctan2(down[1], down[0]) + np.radians(270)
    R = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    reoriented = {node_id: R @ (p - centre) for node_id, p in pos.items()}
    right = reoriented[inv_mapping["Physical health"]]
    if right[0] < 0:
        reoriented = {
            node_id: np.array([-p[0], p[1]])
            for node_id, p in reoriented.items()
        }
    return reoriented


def layout(G: nx.Graph) -> None:
    degrees = nx.get_node_attributes(G, "weighted_degree", default=5.0)
    vals = np.array(list(degrees.values()))
    size_min = 1.0
    size_max = 15.0
    size_range = size_max - size_min
    min_deg = np.min(vals)
    deg_range = np.max(vals) - min_deg
    scale = size_range / deg_range
    sizes = {
        node_id: size_min + scale * (degree - min_deg)
        for node_id, degree in degrees.items()
    }
    # use stepwise strategy to improve convergence speed
    print("Laying out Graph...")
    print("100x...")
    pos = nx.forceatlas2_layout(
        G,
        scaling_ratio=5.0,
        node_size=sizes,
        weight="weight",
        max_iter=300,
        jitter_tolerance=100.0,
        seed=0,
    )
    print("10x...")
    pos = nx.forceatlas2_layout(
        G,
        pos=pos,
        scaling_ratio=5.0,
        node_size=sizes,
        weight="weight",
        max_iter=300,
        jitter_tolerance=10.0,
    )
    print("1x...")
    pos = nx.forceatlas2_layout(
        G,
        pos=pos,
        scaling_ratio=5.0,
        node_size=sizes,
        weight="weight",
        max_iter=500,
        jitter_tolerance=1.0,
    )
    print("reorienting...")
    reoriented = reorient(pos, G)
    viz = {
        node_id: {
            "position": {"x": float(p[0]), "y": float(p[1]), "z": 0.0},
            "size": float(sizes[node_id]),
            "color": {"r": 80, "g": 80, "b": 80},
        }
        for node_id, p in reoriented.items()
    }
    nx.set_node_attributes(G, viz, name="viz")


def filtered_edges(G) -> nx.Graph:
    edge_weights = np.array(list(nx.get_edge_attributes(G, "weight").values()))
    cutoff = np.quantile(edge_weights, 0.5)

    def filter_edge(source: str, target: str) -> bool:
        return G[source][target]["weight"] > cutoff

    view = nx.subgraph_view(G, filter_edge=filter_edge)
    graph = nx.Graph(view)
    return graph


def remove_isolated(G):
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)


def run() -> None:
    """Run the analysis."""
    root = Path(__file__).parent.parent
    output_dir = Path.cwd() / "output"
    fname = root / "data" / "links.csv"
    links = convert.csv_to_links(fname)
    links = convert.clean_link_terms(links)
    terms = convert.links_to_terms(links)
    remove_stopwords(terms, links)
    nodes, edges = create_network_data(terms, links)
    G = create_networkx_graph(nodes, edges)
    detect_communities(G, seed=0, resolution=0.75)
    # layout(G)
    # write_graphml(output_dir, G, "")
    # write_gexf(output_dir, G, "")
    # write_json(output_dir, G, "")
    # create_csvs(output_dir, G, "")

    G_filt = filtered_edges(G)
    remove_isolated(G_filt)
    detect_communities(G_filt, seed=0, resolution=0.75)
    layout(G_filt)
    write_gexf(output_dir, G_filt, "_filtered")
    write_json(output_dir, G_filt, "_filtered")
    create_csvs(output_dir, G_filt, "_filtered")
