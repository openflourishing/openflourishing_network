import xml.etree.ElementTree as ET
import json
from random import random
from flourishing_network.core import convert
from pathlib import Path

# Load and parse the XML file
tree = ET.parse("./scripts/network.gexf")
root = tree.getroot()

# Define the correct namespaces
ns = {
    "gexf": "http://gexf.net/1.3",
    "viz": "http://gexf.net/1.3/viz"
}

# Extract attribute ID to title mapping for nodes
attribute_titles = {}
for attr in root.findall(".//gexf:attributes[@class='node']/gexf:attribute", ns):
    attr_id = attr.attrib["id"]
    title = attr.attrib.get("title", attr_id)
    attribute_titles[attr_id] = title

def get_dummy_scales():
    scale_options = ['ABC', 'DEF', 'GHI']
    scales = []
    for option in scale_options:
        if random() > 0.5:
            scales.append(option)
    if len(scales) == 0:
        scales.append('none')
    return scales

# Parse nodes
nodes = []
for node in root.findall(".//gexf:node", ns):
    node_id = node.attrib["id"]
    label = node.attrib.get("label", node_id)

    # Parse <attvalue> elements
    attr_values = {
        attribute_titles.get(att.attrib["for"], att.attrib["for"]): att.attrib["value"]
        for att in node.findall(".//gexf:attvalue", ns)
    }

    # Extract position (if available)
    pos = node.find("viz:position", ns)
    x = float(pos.attrib["x"]) if pos is not None else 0.0
    y = float(pos.attrib["y"]) if pos is not None else 0.0

        # Build node object
    nodes.append({
        "key": node_id,
        "label": label,
        "tag": "Tool",
        "URL": "",  # Placeholder — add logic if needed
        "cluster": str(attr_values.get("Modularity Class", "")),
        "x": x,
        "y": y,
        "score": float(attr_values.get("weighted_degree", 0.0)),
        "submissions": attr_values.get("submissions", "").split(';'),
    })

# Parse edges
edges = []
for edge in root.findall(".//gexf:edge", ns):
    source = edge.attrib["source"]
    target = edge.attrib["target"]
    weight = float(edge.attrib.get("weight", 1.0))
    edges.append([source, target]) #  , weight])

root = Path(__file__).parent.parent / "flourishing_network"
fname = root / "data" / "submissions.csv"
submissions = convert.csv_to_submissions(fname)

# Combine into final dataset
dataset = {
    "nodes": nodes,
    "edges": edges,
    "clusters": [
        { "key": "7", "color": "#0f8ad7", "clusterLabel": "Relationships" },
        { "key": "12", "color": "#D95818", "clusterLabel": "Emotional experience" },
        { "key": "11", "color": "#c618d6", "clusterLabel": "Vitality" },
        { "key": "0", "color": "#22b955", "clusterLabel": "Adaptability" },
        { "key": "3", "color": "#ae905c", "clusterLabel": "Meaning" },
        { "key": "10", "color": "#579f88", "clusterLabel": "Optimism" },
        { "key": "8", "color": "#2177A6", "clusterLabel": "Autonomy" },
        { "key": "6", "color": "#a59832", "clusterLabel": "Competence" },
        { "key": "2", "color": "#234729", "clusterLabel": "Life satisfaction" },
        { "key": "5", "color": "#48345e", "clusterLabel": "Community and environment" },
        { "key": "13", "color": "#2D7BAF", "clusterLabel": "Security" },
        { "key": "9", "color": "#7e3c4a", "clusterLabel": "Work and leisure" },
        { "key": "4", "color": "#C84FC2", "clusterLabel": "Equanimity" },
        { "key": "1", "color": "#666666", "clusterLabel": "Other" },
    ],
    "tags": [
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
    ],
    "submissions": submissions,
}

# Write to dataset.json
with open("./scripts/dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)