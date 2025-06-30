
import networkx as nx
from pathlib import Path
from . import convert


def parse_nodes(G: nx.Graph):
    nodes = []
    for node_id, attrs in G.nodes.items():
        nums = attrs['submissions'].split(";")
        submissions = [int(s) for s in nums if s.isnumeric()]
        node_dct = {
            "key": str(node_id),
            "label": attrs['term'],
            "tag": "Concept",
            "URL": "",
            "cluster": str(attrs['community']),
            "x": attrs['viz']['position']['x'],
            "y": attrs['viz']['position']['y'],
            "size": attrs['viz']['size'],
            "submissions": submissions,
        }
        nodes.append(node_dct)
    return nodes

def parse_edges(G: nx.Graph):
    # Parse edges
    edges = []
    for source, target, attrs in G.edges(data=True):
        weight = float(attrs.get("weight", 1.0))
        edges.append([str(source), str(target)]) #  , weight])
    return edges

def parse_communities(communities):
    clusters = []
    for community in communities:
        cluster = {
            'key': str(community['key']),
            'color': community['color'],
            'clusterLabel': community['label'],
        }
        clusters.append(cluster)
    return clusters

def get_tags():
    tags = [
        { "key": "Chart type", "image": "charttype.svg" },
        { "key": "Company", "image": "company.svg" },
        { "key": "Concept", "image": "concept.svg" },
        { "key": "Field", "image": "field.svg" },
        { "key": "List", "image": "list.svg" },
        { "key": "Method", "image": "method.svg" },
        { "key": "Organization", "image": "organization.svg" },
        { "key": "Person", "image": "person.svg" },
        { "key": "Technology", "image": "technology.svg" },
        { "key": "Tool", "image": "tool.svg" },
        { "key": "unknown", "image": "unknown.svg" }
    ]
    return tags


def get_submissions():
    root = Path(__file__).parent.parent
    fname = root / "data" / "submissions.csv"
    submissions = convert.csv_to_submissions(fname)
    return submissions



def get_dataset(G, communities):
    dataset = {
        'nodes': parse_nodes(G),
        'edges': parse_edges(G),
        'clusters': parse_communities(communities),
        'tags': get_tags(),
        "submissions": get_submissions(),
    }
    return dataset

