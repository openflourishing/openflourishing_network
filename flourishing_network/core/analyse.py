"""Main openflourishing.org network analysis."""

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


def not_na(value: float) -> bool:
    """Check for NA values."""
    if pd.isna(value):
        return None
    return value


def remove_stopwords(terms: set, links: list[dict]) -> None:
    """Remove stopwords in terms and links.

    Args:
        terms (set): The set of terms.
        links (list[dict]): The list of link dictionaries.
    """
    root = Path(__file__).parent.parent
    fname = root / "data" / "stopwords.json"
    with open(fname) as f:
        stopwords = json.load(f)
    terms.difference_update(stopwords)
    for dct in links:
        if dct["parent"] in stopwords:
            dct["parent"] = None
        dct["linked"].difference_update(stopwords)


def _add_edge(edges: dict[tuple[int, int]: float],
              source: int, target: int, weight: float) -> None:
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
    nodes = {t: i + 1 for i, t in enumerate(terms)}
    edges = {}
    for dct in links:
        linked = dct["linked"]
        parent = dct["parent"]
        if parent is not None:
            for term in linked:
                _add_edge(
                    edges, nodes[parent], nodes[term], 20.0 / len(linked)
                )
        combinations = list(itertools.combinations(linked, 2))
        for source, target in combinations:
            _add_edge(
                edges, nodes[source], nodes[target], 10 / len(combinations)
            )
    combinations = itertools.combinations(terms, 2)
    for source, target in combinations:
        _add_edge(edges, nodes[source], nodes[target], 0.05)
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


def create_csvs(G: nx.Graph) -> None:
    """Output csvs of the network.

    Args:
        G (nx.Graph): The graph.
    """
    nodes_df = make_nodes_df(G)
    edges_df = make_edges_df(G)
    output_dir = Path.cwd()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    fname = output_dir / f"openflourishing_{timestamp}_csv_nodes.csv"
    nodes_df.to_csv(fname, index=None)
    fname = output_dir / f"openflourishing_{timestamp}_csv_edges.csv"
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
    for term, i in nodes.items():
        node_data.append((i, {"term": term}))
    G.add_nodes_from(node_data)
    edge_data = []
    for i, ((source, target), weight) in enumerate(edges.items()):
        edge_data.append((source, target, {"weight": weight}))
    G.add_edges_from(edge_data)
    degree_view = G.degree(weight="weight")
    for node, weighted_degree in degree_view:
        G.nodes[node]["weighted_degree"] = weighted_degree
    community_sets = nx.algorithms.community.louvain_communities(
        G, weight="weight", seed=0
    )
    for i, community_set in enumerate(community_sets):
        for node in community_set:
            G.nodes[node]["community"] = i
    return G


def write_graphml(G: nx.Graph) -> None:
    """Write the graph to a GraphML file.

    Args:
        G (nx.Graph): The graph.
    """
    output_dir = Path.cwd()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    fname = output_dir / f"openflourishing_{timestamp}_network.graphml"
    nx.readwrite.write_graphml(G, fname)


def G_to_dict(G: nx.Graph) -> dict:
    """Return the network in json format.
    
    Args:
        G (nx.Graph): The network.
    """
    return nx.readwrite.json_graph.node_link_data(G)


def write_json(G: nx.Graph) -> None:
    """Write the network to json.

    Args:
        G (nx.Graph): The network.
    """
    dct = G_to_dict(G)
    output_dir = Path.cwd()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    fname = output_dir / f"openflourishing_{timestamp}_network.json"
    with open(fname, "w") as f:
        json.dump(dct, f, indent=4)


def run() -> None:
    """Run the analysis."""
    root = Path(__file__).parent.parent
    fname = root / "data" / "links.csv"
    links = convert.csv_to_links(fname)
    terms = convert.links_to_terms(links)
    remove_stopwords(terms, links)
    nodes, edges = create_network_data(terms, links)
    G = create_networkx_graph(nodes, edges)
    write_graphml(G)
    write_json(G)
    create_csvs(G)
