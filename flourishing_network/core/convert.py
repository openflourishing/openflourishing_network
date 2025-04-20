"""Data conversions."""

from __future__ import annotations

import json
import json.tool

import pandas as pd


def json_to_links(fname: str) -> list[dict]:
    """Read a JSON file to link data.

    Args:
        fname (str): The file path.

    Returns:
        list[dict]: A list of link dictionaries.
    """
    with open(fname) as f:
        links = json.load(f)
    return links


def csv_to_links(fname: str) -> list[dict]:
    """Read a CSV file to link data.

    Args:
        fname (str): The file path.

    Returns:
        list[dict]: A list of link dictionaries.
    """
    links = pd.read_csv(fname, index_col=None).to_dict("records")
    for link in links:
        linked = link["linked"].split(",")
        linked = [term.strip() for term in linked]
        link["linked"] = set(linked)
        link["submission_id"] = int(link["submission_id"])
        if pd.isna(link["parent"]):
            link["parent"] = None
    return links


def links_to_terms(links: list[dict]) -> set:
    """Return a set of all terms in the links.

    Args:
        links (list[dict]): The list of link dictionaries.

    Returns:
        set: All terms in the link dictionaries.
    """
    terms = set()
    for link in links:
        if link["parent"] is not None:
            terms.add(link["parent"])
        terms.update(link["linked"])
    return terms


def links_to_text(links: list[dict], fname: str) -> None:
    """Convert links to a string in the openflourishing.org link text format.

    Args:
        links (list[dict]): The list of link dictionaries.
        fname (str): The file path.
    """
    links = [dct.copy() for dct in links]
    rows = []
    submission_id = None
    for link in links:
        if link['submission_id'] != submission_id:
            rows.append(f"<id={link['submission_id']}>")
        lhs = link['parent']
        rel = " --" + link['relationship'].lower() + "-- "
        rhs = ", ".join(link["linked"])
        row = lhs + rel + rhs
        rows.append(row)
    text = rows.join('\n')
    return text

def links_to_text_file(links: list[dict], fname: str) -> None:
    """Convert links to the openflourishing.org link text file.

    Args:
        links (list[dict]): The list of link dictionaries.
        fname (str): The file path.
    """
    string = links_to_text(links, fname)
    with open(fname, "w") as f:
        f.writelines(string)



def links_to_csv(links: list[dict], fname: str) -> None:
    """Save links to a CSV file.

    Args:
        links (list[dict]): The list of link dictionaries.
        fname (str): The file path.
    """
    links = [dct.copy() for dct in links]
    for link in links:
        link["linked"] = ", ".join(link["linked"])
    df = pd.DataFrame(links)
    df.to_csv(fname, index=False)


class SetEncoder(json.JSONEncoder):
    """A custom JSON encoder than converts sets to lists."""

    def default(self: object, obj: object) -> object:
        """The default encoding method."""
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def links_to_json(links: list[dict], fname: str) -> None:
    """Save links to a JSON file.

    Args:
        links (list[dict]): The list of link dictionaries.
        fname (str): The file path.
    """
    with open(fname, "w") as f:
        json.dump(links, f, cls=SetEncoder)


def clean_term(term: str) -> str:
    term = term.lower()
    term = term.replace('well-being', 'wellbeing')
    term = term.capitalize()
    return term


def clean_link_terms(links: list[dict]) -> list[dict]:
    """Make all terms in links title case."""
    out_links = []
    for link in links:
        dct = {}
        for key, val in link.items():
            if key == 'linked':
                dct[key] = set([clean_term(t) for t in val])
            elif key == 'parent':
                dct[key] = None if val is None else clean_term(val)
            else:
                dct[key] = val
        out_links.append(dct)
    return out_links
