"""Main openflourishing.org network analysis."""

from __future__ import annotations

import itertools
import json
import json.tool
import random
from datetime import datetime
from pathlib import Path

import distinctipy
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sknetwork.clustering import Leiden

from . import convert, datasets

random.seed(0)
np.random.seed(0)

COMMUNITY_COLOR_PREFERENCES = {
    "Relationships": "#5B57D8",
    "Physical health": "#74E45E",
    "Happiness": "#F06059",
    "Self-efficacy": "#5CD2FC",
    "Coping": "#E98AFD",
    "Spirituality": "#FCD26D",
    "Meaning": "#717864",
    "Optimism": "#ADB9B2",
    "Work": "#908FFB",
    "Autonomy": "#B2B055",
    "Security": "#AAF4EB",
    "Economy": "#57A2B1",
}


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
    """Check for NA values.

    Args:
        value: The value to check for NA.

    Returns:
        The original value if not NA, None otherwise.
    """
    if pd.isna(value):
        return None
    return value


def get_stopwords() -> list[str]:
    """Get the list of stopwords from the data file.

    Returns:
        List of stopwords to filter out from terms.
    """
    root = Path(__file__).parent.parent
    fname = root / "data" / "stopwords.json"
    with open(fname) as f:
        stopwords = json.load(f)
    return stopwords


def remove_stopwords(terms: dict, links: list[dict]) -> None:
    """Remove stopwords in terms and links.

    Args:
        terms (set): The set of terms.
        links (list[dict]): The list of link dictionaries.
    """
    stopwords = get_stopwords()
    for stopword in stopwords:
        if stopword in terms:
            del terms[stopword]
    for dct in links:
        if dct["parent"] in stopwords:
            dct["parent"] = None
        for term in dct["linked"].copy():
            if term in stopwords:
                dct["linked"].remove(term)


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


def create_network_data(terms: dict, links: list[dict]) -> tuple[dict, dict]:
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
    sorted_terms = list(terms.keys())
    nodes = {
        t: {"term": t, "index": i + 1, "submissions": set()}
        for i, t in enumerate(sorted_terms)
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
                if dct["relationship"] == "includes":
                    nodes[term]["submissions"].add(submission_id)
        combinations = list(itertools.combinations(linked, 2))
        for source, target in combinations:
            _add_edge(
                edges,
                nodes[source]["index"],
                nodes[target]["index"],
                10 / len(combinations),
            )
            if dct["relationship"] == "includes":
                nodes[source]["submissions"].add(submission_id)
                nodes[target]["submissions"].add(submission_id)
    # combinations = itertools.combinations(terms, 2)
    # for source, target in combinations:
    #     _add_edge(edges, nodes[source], nodes[target], 0.01)
    for node_dct in nodes.values():
        subs_str = ";".join(sorted(list(node_dct["submissions"])))
        node_dct["submissions"] = subs_str
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


def write_csvs(
    output_dir: Path, timestamp: str, G: nx.Graph, suffix: str = ""
) -> None:
    """Output CSV files of the network.

    Args:
        output_dir: Directory to write the CSV files to.
        timestamp: Timestamp string to include in filenames.
        G: The networkx graph to export.
        suffix: Optional suffix to add to filenames.
    """
    nodes_df = make_nodes_df(G)
    edges_df = make_edges_df(G)
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
    for node_id, attrs in G.nodes.items():
        attrs["viz"] = {}
    return G


def add_weighted_degree(G: nx.Graph) -> None:
    """Add weighted degree attribute to all nodes in the graph.

    Args:
        G: The networkx graph to modify in-place.
    """
    degree_view = G.degree(weight="weight")
    for node, weighted_degree in degree_view:
        G.nodes[node]["weighted_degree"] = weighted_degree


def add_community_labels(G: nx.Graph, communities: list[dict]) -> list[dict]:
    """Add descriptive labels to communities based on highest-degree nodes.

    Args:
        G: The networkx graph containing node data.
        communities: List of community dictionaries with 'nodes' key.

    Returns:
        List of community dictionaries with added 'label' keys.
    """
    degree_view = G.degree(weight="weight")
    communities_out = [dct.copy() for dct in communities]
    for community in communities_out:
        degrees = [(node, degree_view[node]) for node in community["nodes"]]
        degrees.sort(reverse=True, key=lambda t: t[1])
        name = G.nodes[degrees[0][0]]["term"].strip("*")
        if len(degrees) >= 2:
            name += ", " + G.nodes[degrees[1][0]]["term"].strip("*")
        if len(degrees) >= 3:
            name += ", " + G.nodes[degrees[2][0]]["term"].strip("*")
        if len(degrees) >= 4:
            name += " and " + G.nodes[degrees[3][0]]["term"].strip("*")
        community["label"] = name
    return communities_out


def hex_to_normalized_rgb(hex_color: str) -> tuple[float, float, float]:
    """
    Convert a hex color string to a normalized RGB tuple (values between 0.0 and 1.0).

    Args:
        hex_color (str): Hex color string, e.g., '#FF5733' or 'FF5733'

    Returns:
        tuple: Normalized RGB tuple, e.g., (1.0, 0.341, 0.2)
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError("Hex color must be 6 characters long.")

    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0

    return (r, g, b)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """
    Convert a hex color string to an RGB tuple with integer values (0–255).

    Args:
        hex_color (str): Hex color string, e.g., '#FF5733' or 'FF5733'

    Returns:
        tuple: RGB tuple, e.g., (255, 87, 51)
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError("Hex color must be 6 characters long.")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return (r, g, b)


def to_color_rgb(
    color: tuple[float, float, float], luminance: float = 255 * 0.7
) -> tuple[int, int, int]:
    """Convert a normalized color tuple to RGB integers with luminance scaling.

    Args:
        color: Tuple of (r, g, b) values in range [0, 1].
        luminance: Scaling factor for brightness.

    Returns:
        Tuple of (r, g, b) integer values.
    """
    r, g, b = color
    return int(r * luminance), int(g * luminance), int(b * luminance)


def rgb_color_to_hex(color: tuple[int, int, int]) -> str:
    """Convert a color tuple to hexadecimal color string.

    Args:
        color: Tuple of (r, g, b) values in range [0, 255].

    Returns:
        Hexadecimal color string in format '#RRGGBB'.
    """
    r, g, b = color
    return f"#{r:02X}{g:02X}{b:02X}"


def normalised_rgb_color_to_hex(color: tuple[float, float, float]) -> str:
    """Convert a normalized color tuple to hexadecimal color string.

    Args:
        color: Tuple of (r, g, b) values in range [0, 1].

    Returns:
        Hexadecimal color string in format '#RRGGBB'.
    """
    return rgb_color_to_hex(to_color_rgb(color))


def add_community_colors(communities: list[dict]) -> list[dict]:
    """Add color information to communities using distinct colors.

    Args:
        communities: List of community dictionaries to add colors to.

    Returns:
        List of community dictionaries with added color fields.
    """
    N = len(communities)
    communities_out = [dct.copy() for dct in communities]
    colors = distinctipy.get_colors(N, pastel_factor=0.5, rng=0)
    rgbs = [to_color_rgb(color) for color in colors]
    preferences = COMMUNITY_COLOR_PREFERENCES.copy()
    for community in communities_out:
        found = False
        for pref in preferences:
            if pref in community["label"]:
                color_rgb = hex_to_rgb(preferences[pref])
                if color_rgb in rgbs:
                    rgbs.remove(color_rgb)
                found = True
                break
        if not found:
            color_rgb = rgbs.pop(0)
        community["color_rgb"] = color_rgb
        community["color"] = rgb_color_to_hex(color_rgb)
    return communities_out


def detect_communities(
    G: nx.Graph, seed: int, resolution: float
) -> list[dict]:
    """Detect communities in the graph using Leiden algorithm.

    Args:
        G: The networkx graph to analyze.
        seed: Random seed for reproducible results.
        resolution: Resolution parameter for community detection.

    Returns:
        List of community dictionaries with nodes, keys, labels, and colors.
    """
    node_list = sorted(G.nodes())
    adjacency = csr_matrix(nx.to_scipy_sparse_array(G, node_list))
    leiden = Leiden(
        resolution=resolution, random_state=seed, shuffle_nodes=True
    )
    labels = leiden.fit_predict(adjacency)
    communities = {}
    for node_id, label in zip(node_list, labels, strict=True):
        if label not in communities:
            communities[label] = set([node_id])
        else:
            communities[label].add(node_id)
    communities = list(communities.values())
    communities.sort(reverse=True, key=lambda s: len(s))
    communities = [{"key": i, "nodes": s} for i, s in enumerate(communities)]
    communities = add_community_labels(G, communities)
    communities = add_community_colors(communities)
    return communities


def assign_communities(G: nx.Graph, communities: list[dict]) -> None:
    """Assign community information to graph nodes.

    Args:
        G: The networkx graph to modify in-place.
        communities: List of community dictionaries with node assignments.
    """
    for community in communities:
        for node_id in community["nodes"]:
            G.nodes[node_id]["community"] = community["key"]
            r, g, b = community["color_rgb"]
            color = {"r": r, "g": g, "b": b}
            G.nodes[node_id]["viz"]["color"] = color


def detect_and_assign_communities(G: nx.Graph) -> list[dict]:
    """Detect communities and assign them to graph nodes.

    Args:
        G: The networkx graph to analyze and modify.

    Returns:
        List of community dictionaries with assignments.
    """
    communities = detect_communities(G, seed=1, resolution=1.0)
    assign_communities(G, communities)
    return communities


def write_graphml(output_dir: Path, timestamp: str, G: nx.Graph) -> None:
    """Write the graph to a GraphML file.

    Args:
        output_dir: Directory to write the GraphML file to.
        timestamp: Timestamp string to include in filename.
        G: The networkx graph to export.

    Warning:
        Currently not compatable with dictionary node data! Do not use.
    """
    fname = output_dir / f"openflourishing_{timestamp}_network.graphml"
    nx.readwrite.write_graphml(G, fname)


def write_gexf(
    output_dir: Path, timestamp: str, G: nx.Graph, suffix: str = ""
) -> None:
    """Write the graph to a GEXF file.

    Args:
        output_dir: Directory to write the GEXF file to.
        timestamp: Timestamp string to include in filename.
        G: The networkx graph to export.
        suffix: Optional suffix to add to filename.
    """
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


def write_json(
    output_dir: Path, timestamp: str, G: nx.Graph, suffix: str = ""
) -> None:
    """Write the network to JSON format.

    Args:
        output_dir: Directory to write the JSON file to.
        timestamp: Timestamp string to include in filename.
        G: The networkx graph to export.
        suffix: Optional suffix to add to filename.
    """
    dct = G_to_dict(G)
    fname = output_dir / (
        f"openflourishing_{timestamp}_network" + suffix + ".json"
    )
    with open(fname, "w") as f:
        json.dump(dct, f, indent=2)


def write_dataset(
    output_dir: Path,
    timestamp: str,
    G: nx.Graph,
    communities: list[dict],
    suffix: str = "",
) -> None:
    """Write the dataset JSON file for the network.

    Args:
        output_dir: Directory to write the dataset file to.
        timestamp: Timestamp string to include in filename.
        G: The networkx graph.
        communities: List of community dictionaries.
        suffix: Optional suffix to add to filename.
    """
    fname = output_dir / (
        f"openflourishing_{timestamp}_network_dataset" + suffix + ".json"
    )
    dataset = datasets.get_dataset(G, communities)
    with open(fname, "w") as f:
        json.dump(dataset, f, indent=2)


def reorient(pos: dict[int, np.ndarray], G: nx.Graph) -> dict[int, np.ndarray]:
    """Reorient the graph layout for better visualization.

    Args:
        pos (dict[int, np.ndarray]): Dictionary of node positions.
        G (nx.Graph): The graph.

    Returns:
        dict[int, np.ndarray]: Reoriented node positions.
    """
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
    """Apply a layout algorithm to the graph and update node attributes.

    Args:
        G (nx.Graph): The graph to layout.
    """
    degrees = nx.get_node_attributes(G, "weighted_degree")
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
    print(f"Laying out Graph with {len(vals)} nodes...")
    print("100x...")
    pos = nx.forceatlas2_layout(
        G,
        scaling_ratio=5.0,
        node_size=sizes,
        weight="weight",
        max_iter=400,
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
        max_iter=800,
        jitter_tolerance=10.0,
    )
    print("1x...")
    pos = nx.forceatlas2_layout(
        G,
        pos=pos,
        scaling_ratio=5.0,
        node_size=sizes,
        weight="weight",
        max_iter=1200,
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


def remove_isolated(G: nx.Graph) -> nx.Graph:
    """Remove isolated nodes from the graph.

    Args:
        G (nx.Graph): The graph from which isolated nodes will be removed.
    """
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()


def filter_edges(G: nx.Graph) -> nx.Graph:
    """Filter edges in the graph based on weight and remove isolated nodes.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        nx.Graph: A new graph with filtered edges and isolated nodes removed.
    """
    edge_weights = np.array(list(nx.get_edge_attributes(G, "weight").values()))
    cutoff = np.quantile(edge_weights, 0.5)

    def filter_edge(source: str, target: str) -> bool:
        return G[source][target]["weight"] > cutoff

    view = nx.subgraph_view(G, filter_edge=filter_edge)
    graph = nx.Graph(view)
    largest_connected_graph = remove_isolated(graph)
    return largest_connected_graph


def create_outputs(
    output_dir: Path,
    timestamp: str,
    G: nx.Graph,
    communities: list[dict],
    prefix: str = "",
) -> None:
    """Create output files for the network analysis.

    Args:
        output_dir (Path): Directory to write the output files to.
        timestamp (str): Timestamp string to include in filenames.
        G (nx.Graph): The networkx graph to export.
        communities (list[dict]): List of community dictionaries.
        prefix (str, optional): Optional prefix to add to filenames.
    """
    write_gexf(output_dir, timestamp, G, prefix)
    write_json(output_dir, timestamp, G, prefix)
    write_csvs(output_dir, timestamp, G, prefix)
    write_dataset(output_dir, timestamp, G, communities, prefix)


def analyse_submissions(G: nx.Graph) -> dict:
    """Calculate which communities each submission includes.

    Args:
        G (nx.Graph): The networkx graph containing node data.

    Returns:
        dict: Dictionary mapping submission IDs to community counts.

    """
    submission_communities = {}
    communities = set()
    node_data = G.nodes(data=True)
    for _, node_dct in node_data:
        communities.add(node_dct.get("community", -1))
    communities = {c: 0 for c in communities if c != -1}
    for _, node_dct in node_data:
        submissions = node_dct["submissions"]
        community = node_dct["community"]
        if len(submissions) == 0:
            continue
        submission_ids = submissions.split(";")
        for submission_id in submission_ids:
            submission_id = int(submission_id)
            if submission_id not in submission_communities:
                submission_communities[submission_id] = communities.copy()
            submission_communities[submission_id][community] += 1
    return submission_communities


def output_submission_communities(
    output_dir: Path, timestamp: str, submission_communities: dict
) -> None:
    """Output the submission communities to a JSON file.


    Args:
        output_dir (Path): Directory to write the JSON file to.
        timestamp (str): Timestamp string to include in filename.
        submission_communities (dict): Dictionary of submission communities.
    """
    df = pd.DataFrame.from_records(submission_communities).T
    fname = (
        output_dir / f"openflourishing_{timestamp}_submission_communities.csv"
    )
    df.to_csv(fname, index=False)


def process_network(G: nx.Graph) -> None:
    """Process the network by filtering edges, detecting communities, and
    creating outputs.

    Args:
        G (nx.Graph): The input graph to process.
    """
    root = Path(__file__).parent.parent.parent
    output_dir = root / "output"
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    add_weighted_degree(G)
    # layout(G)
    # communities = detect_and_assign_communities(G)
    # create_outputs(output_dir, timestamp, G, communities, "")
    G_filt = filter_edges(G)
    layout(G_filt)
    communities = detect_and_assign_communities(G_filt)
    create_outputs(output_dir, timestamp, G_filt, communities, "_filtered")
    submission_communities = analyse_submissions(G_filt)
    output_submission_communities(
        output_dir, timestamp, submission_communities
    )


def run() -> None:
    """Run the analysis."""
    root = Path(__file__).parent.parent
    fname = root / "data" / "links.csv"
    links = convert.csv_to_links(fname)
    links = convert.clean_link_terms(links)
    terms = convert.links_to_terms(links)
    remove_stopwords(terms, links)
    nodes, edges = create_network_data(terms, links)
    G = create_networkx_graph(nodes, edges)
    process_network(G)
