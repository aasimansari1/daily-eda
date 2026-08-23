"""
Day 01 — sklearn wine dataset (178 wines, 13 chemical features, 3 cultivars)
Question: Which chemical features separate the three cultivars most?
Run: python analysis.py   (writes chart.png next to this file)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine

HERE = Path(__file__).parent


def load() -> pd.DataFrame:
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["cultivar"] = pd.Categorical.from_codes(data.target, data.target_names)
    return df


def main() -> None:
    df = load()
    print(f"shape: {df.shape}")
    print(df["cultivar"].value_counts(), "\n")

    # Rank features by how far apart the class means are, relative to overall spread.
    feats = df.columns.drop("cultivar")
    class_means = df.groupby("cultivar", observed=True)[feats].mean()
    separation = (class_means.max() - class_means.min()) / df[feats].std()
    separation = separation.sort_values(ascending=False)
    print("Feature separation score (range of class means / overall std):")
    print(separation.round(2).to_string(), "\n")

    top2 = separation.index[:2].tolist()
    print(f"Top-2 separating features: {top2}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    separation.plot.barh(ax=axes[0], color="#4C72B0")
    axes[0].invert_yaxis()
    axes[0].set_title("How well each feature separates the 3 cultivars")
    axes[0].set_xlabel("range of class means / std")

    for name, grp in df.groupby("cultivar", observed=True):
        axes[1].scatter(grp[top2[0]], grp[top2[1]], label=name, alpha=0.75, s=28)
    axes[1].set_xlabel(top2[0])
    axes[1].set_ylabel(top2[1])
    axes[1].set_title(f"{top2[0]} vs {top2[1]} by cultivar")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=120)
    print("saved chart.png")


if __name__ == "__main__":
    main()
