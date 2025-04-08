import json.tool
import pandas as pd
import itertools
import numpy as np
import networkx as nx
import json

from pathlib import Path
from datetime import datetime


def get_df():
    root = Path(__file__).parent
    fname = root / "data" / "taxonomies.csv"
    df = pd.read_csv(fname)
    return df


def get_df_rel():
    root = Path(__file__).parent
    fname = root / "data" / "relationships.csv"
    df = pd.read_csv(fname)
    return df

def analyse(df):
    terms = set()
    links = []
    subset = set()
    category = None
    submission_id = None
    for index, row in df.iterrows():
        if row["Submission ID"] != submission_id:
            links.append({"subset": subset, "category": category})
            subset = set()
            category = None
            submission_id = row["Submission ID"]
        term = row["Term"]
        if term not in terms:
            terms.add(term)
        if row["Category"] is not None and not pd.isna(row["Category"]):
            if row["Category"] != category:
                links.append({"subset": subset, "category": category})
                category = row["Category"]
                subset = set()
                if category not in terms:
                    terms.add(category)
        subset.add(term)
    links.append({"subset": subset, "category": category})
    return terms, links

def analyse_relationships(df):
    """These need to be added with different logic."""
    terms = set()
    links = []
    for index, row in df.iterrows():
        if row['Relationship'] == 'related to':
            subset = set([row['Term']])
            others = row['Other terms'].split(',')
            others = set([o.strip(' ') for o in others])
            subset.update(others)
            terms.update(others)
            links.append({"subset": subset, "category": None})
    return terms, links

def remove_stopwords(terms, links):
    root = Path(__file__).parent
    fname = root / "data" / "stopwords.json"
    with open(fname, 'r') as f:
        stopwords = json.load(f)
    terms.difference_update(stopwords)
    for dct in links:
        if dct['category'] in stopwords:
            dct['category'] = None
        dct['subset'].difference_update(stopwords)

def _add_edge(edges, source, target, weight):
    if source > target:
        source, target = target, source
    if (source, target) not in edges:
        edges[(source, target)] = weight
    edges[(source, target)] += weight


def create_network_data(terms, links):
    nodes = {t: i + 1 for i, t in enumerate(terms)}
    edges = {}
    for dct in links:
        subset = dct["subset"]
        category = dct["category"]
        if category is not None:
            for term in subset:
                _add_edge(edges, nodes[category], nodes[term], 20.0 / len(subset))
        combinations = list(itertools.combinations(subset, 2))
        for source, target in combinations:
            _add_edge(edges, nodes[source], nodes[target], 10 / len(combinations))
    combinations = itertools.combinations(terms, 2)
    for source, target in combinations:
        _add_edge(edges, nodes[source], nodes[target], 0.05)
    return nodes, edges


def make_nodes_df(G):
    node_data = []
    for _id, node_dct in G.nodes.items():
        dct = {"ID": _id, "Label": node_dct['term']}
        dct.update(node_dct)
        node_data.append(dct)
    df = pd.DataFrame(node_data)
    cols = df.columns
    df.columns = [col.title() for col in cols]
    return df


def make_edges_df(G):
    edge_data = []
    for i, (source, target, edge_dct) in enumerate(G.edges(data=True)):
        dct = {"ID": i, "Source": source, "Target": target}
        dct.update(edge_dct)
        edge_data.append(dct)
    df = pd.DataFrame(edge_data)
    cols = df.columns
    df.columns = [col.title() for col in cols]
    return df


def create_csvs(G):
    nodes_df = make_nodes_df(G)
    edges_df = make_edges_df(G)
    output_dir = Path.cwd()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    fname = output_dir / f"openflourishing_{timestamp}_csv_nodes.csv"
    nodes_df.to_csv(fname, index=None)
    fname = output_dir / f"openflourishing_{timestamp}_csv_edges.csv"
    edges_df.to_csv(fname, index=None)


def create_networkx_graph(nodes, edges):
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


def write_graphml(G):
    output_dir = Path.cwd()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    fname = output_dir / f"openflourishing_{timestamp}_network.graphml"
    nx.readwrite.write_graphml(G, fname)

def G_to_dict(G):
    return nx.readwrite.json_graph.node_link_data(G)

def write_json(G):
    dct = G_to_dict(G)
    output_dir = Path.cwd()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    fname = output_dir / f"openflourishing_{timestamp}_network.json"
    with open(fname, 'w') as f:
        json.dump(dct, f, indent=4)


def run():
    df = get_df()
    df_rel = get_df_rel()
    terms, links = analyse(df)
    terms_rel, links_rel = analyse_relationships(df_rel)
    terms.update(terms_rel)  # No add with different logic
    links.extend(links_rel)  # No add with different logic
    remove_stopwords(terms, links)
    nodes, edges = create_network_data(terms, links)
    G = create_networkx_graph(nodes, edges)
    write_graphml(G)
    write_json(G)
    create_csvs(G)


if __name__ == "__main__":
    run()
