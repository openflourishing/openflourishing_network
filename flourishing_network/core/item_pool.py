from pathlib import Path

import json
import numpy as np
import pandas as pd


def get_data_frame():
    path = Path(__file__).parent.parent / "data" / "item_pool.csv"
    df = pd.read_csv(path, index_col=None, encoding="utf-8")
    df = df.drop_duplicates(subset=["Item"])
    df = df.reset_index(drop=True)
    cols = [
        "item",
        "terms",
        "level",
        "tense",
        "response_categories",
        "direction",
        "context",
        "contributor_id",
        "date",
        "ai_drafted",
        "edited",
    ]
    df.columns = cols
    return df


def df_to_dict(df):
    """Convert DataFrame to a list of dictionaries."""
    dct = {}
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        date = row_dict.pop("date", None)
        terms = row_dict.pop("terms").split(",")
        response_cats = row_dict["response_categories"].split(",")
        response_cats = [cat.strip() for cat in response_cats]
        row_dict["response_categories"] = response_cats
        row_dict["ai_drafted"] = row_dict["ai_drafted"] == "y"
        row_dict["edited"] = row_dict["edited"] == "y"
        row_dict["direction"] = (
            0
            if np.isnan(row_dict["direction"])
            else int(row_dict["direction"])
        )
        for term in terms:
            term = term.strip(" *")
            if term not in dct:
                dct[term] = []
            dct[term].append(row_dict)
    return dct


def to_json(dct, filename):
    """Save dictionary to a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dct, f, ensure_ascii=False, indent=2)


def make_item_pool_json(filename):
    """Create item pool JSON file from the DataFrame."""
    df = get_data_frame()
    dct = df_to_dict(df)
    to_json(dct, filename)
    print(f"Item pool JSON saved to {filename}")
