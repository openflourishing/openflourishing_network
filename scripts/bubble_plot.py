"""Create a bubble plot."""

import pandas as pd
import numpy as np
from pathlib import Path

root = Path(__file__).parent.parent
submissions_fname = root / "flourishing_network/data/submissions.csv"
com_counts_fname = root / "output/openflourishing_2025-07-20-1919_submission_communities.csv"


def get_n_eff(x):
    """Effective number of communities."""
    x = np.array(x)
    if np.sum(x) == 0:
        return 0.0
    p = x / np.sum(x)
    return 1.0 / np.sum(p ** 2)


def get_coms(n: int = 12) -> pd.DataFrame:
    """Read community data and calculate effective number of communities.

    Args:
        n (int): Number of top communities to include. Defaults to 12.

    Returns:
        pd.DataFrame: Dataframe with community data and effective community size.
    """
    try:
        df = pd.read_csv(
            com_counts_fname, index_col=None, encoding="utf-8"
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            com_counts_fname, index_col=None, encoding="ISO-8859-1"
        )

    df = df.iloc[:, 0:n]
    print(df.head())
    df['N_eff'] = [get_n_eff(row) for index, row in df.iterrows()]
    df['Submission ID'] = df.index + 1
    df.columns = (
        ['Community ' + str(i + 1) for i in range(n)]
        + ['N_eff', 'Submission ID']
    )
    return df


def get_submissions() -> pd.DataFrame:
    """Read submissions data from the CSV file.

    Returns:
        pd.DataFrame: Dataframe containing submissions data.
    """
    df = pd.read_csv(submissions_fname, index_col=None, encoding="utf-8")
    return df


def merge_submissions_and_coms(
    submissions: pd.DataFrame, coms: pd.DataFrame
) -> pd.DataFrame:
    """Merge submissions with community data.

    Args:
        submissions (pd.DataFrame): Dataframe containing submissions data.
        coms (pd.DataFrame): Dataframe containing community data.

    Returns:
        pd.DataFrame: Merged dataframe.
    """
    merged = pd.merge(submissions, coms, on="Submission ID", how="left")
    merged.sort_values(by='N_eff', ascending=False, inplace=True)
    merged = merged[merged['Type'] == "Scale"]
    return merged


def plot_bubble_chart(merged: pd.DataFrame, n=20) -> None:
    """Plot a bubble chart of submissions by effective community size.

    Args:
        merged (pd.DataFrame): Merged dataframe containing submissions and
            community data.
    """
    import matplotlib.pyplot as plt  # Ensure matplotlib is installed

    plt.figure(figsize=(6   , 12))
    merged_topn = merged.iloc[0:n, :]
    print(merged_topn[["Submission ID", "ScaleAbbr", "N_eff"]])
    cols = merged_topn.columns[
        merged_topn.columns.get_loc('Community 1'):
        merged_topn.columns.get_loc('Community 12') + 1
    ]
    i = 0
    for _, row in merged_topn.iterrows():
        x = np.arange(12)
        y = np.ones_like(x).astype(float) * i
        s = (row[cols] * 10).astype(float)
        plt.scatter(x=x, y=y, s=s, alpha=0.6)
        i += 1
    # plt.plot(merged_topn['N_eff'], np.arange(n), color='red', linestyle='-', linewidth=2)
    plt.title('Bubble Plot of Submissions by Effective Community Size')
    plt.xlabel('Community | Effective Number of Communities')
    plt.ylabel('Scale Abbreviation')
    plt.xticks(ticks=np.arange(12), labels=np.arange(12) + 1)
    plt.yticks(ticks=np.arange(n).astype(float), labels=merged_topn['ScaleAbbr'])
    plt.grid(True)
    plt.tight_layout()
    plt.gca().invert_yaxis()  # Invert y-axis to have the first submission at the top
    plt.show()


if __name__ == "__main__":
    coms = get_coms()
    submissions = get_submissions()
    merged = merge_submissions_and_coms(submissions, coms)
    plot_bubble_chart(merged)
    print("Bubble plot generated successfully.")
