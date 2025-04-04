import pandas as pd
import itertools
import numpy as np
from pathlib import Path
from datetime import datetime

def get_df():
    root = Path(__file__).parent
    fname = root / 'data' / 'taxonomies.csv'
    df = pd.read_csv(fname)
    return df

def analyse(df):
    terms = set()
    links = []
    subset = set()
    category = None
    submission_id = None
    for index, row in df.iterrows():
        if row['Submission ID'] != submission_id:
            links.append({'subset': subset, 'category': category})
            subset = set()
            category = None
            submission_id = row['Submission ID']
        term = row['Term']
        if term not in terms:
            terms.add(term)
        if row['Category'] is not None and not pd.isna(row['Category']):
            if row['Category'] != category:
                links.append({'subset': subset, 'category': category})
                category = row['Category']
                subset = set()
                if category not in terms:
                    terms.add(category)
        subset.add(term)
    links.append({'subset': subset, 'category': category})
    return terms, links

def _add_edge(edges, source, target, weight):
    if source > target:
        source, target = target, source
    if (source, target) not in edges:
        edges[(source, target)] = weight
    edges[(source, target)] += weight

def create_network_data(terms, links):
    nodes = {t: i+1 for i, t in enumerate(terms)}
    edges = {}
    for dct in links:
        subset = dct['subset']
        category = dct['category']
        if category is not None:
            for term in subset:
                _add_edge(edges, nodes[category], nodes[term], 20.0/len(subset))
        combinations = list(itertools.combinations(subset, 2))
        for source, target in combinations:
            _add_edge(edges, nodes[source], nodes[target], 10/len(combinations))
    combinations = itertools.combinations(terms, 2)
    for source, target in combinations:
        _add_edge(edges, nodes[source], nodes[target], 0.05)
    return nodes, edges


def make_gephi_nodes(nodes):
    rows = []
    for term, i in nodes.items():
        rows.append({'ID': i, "Label": term})
    df = pd.DataFrame(rows)
    return df


def make_gephi_edges(edges):
    rows = []
    for i, ((source, target), weight) in enumerate(edges.items()):
        rows.append({'ID': i,
                     "Source": source,
                     "Target": target,
                     "Weight": weight}
                     )
    df = pd.DataFrame(rows)
    return df


def create_gephi(nodes, edges):
    nodes_df = make_gephi_nodes(nodes)
    edges_df = make_gephi_edges(edges)
    output_dir = Path.cwd()
    timestamp = datetime.now().strftime('%Y-%m-%d-%H%M')
    fname = output_dir / f'openflourishing_{timestamp}_Gephi_nodes.csv'
    nodes_df.to_csv(fname, index=None)
    fname = output_dir / f'openflourishing_{timestamp}_Gephi_edges.csv'
    edges_df.to_csv(fname, index=None)

def run():
    df = get_df()
    terms, links = analyse(df)
    nodes, edges = create_network_data(terms, links)
    create_gephi(nodes, edges)


if __name__ == '__main__':
    run()