"""
Day 02 — sklearn wine dataset (178 wines, 13 chemical features, 3 cultivars)
Question: How much accuracy does a classifier lose using only the top-k features vs. all 13?
Run: python analysis.py   (writes chart.png next to this file)

Follows on from day-01, which ranked the features by a separation score.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
SEED = 0


def load() -> pd.DataFrame:
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["cultivar"] = pd.Categorical.from_codes(data.target, data.target_names)
    return df


def separation_ranking(df: pd.DataFrame) -> pd.Series:
    """Same score as day-01: range of per-cultivar means / overall std."""
    feats = df.columns.drop("cultivar")
    class_means = df.groupby("cultivar", observed=True)[feats].mean()
    return ((class_means.max() - class_means.min()) / df[feats].std()).sort_values(ascending=False)


def main() -> None:
    df = load()
    ranking = separation_ranking(df)
    y = df["cultivar"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))

    rows = []
    for k in range(1, len(ranking) + 1):
        cols = ranking.index[:k].tolist()
        scores = cross_val_score(model, df[cols], y, cv=cv, scoring="accuracy")
        rows.append({"k": k, "added": cols[-1], "mean": scores.mean(), "std": scores.std()})
    curve = pd.DataFrame(rows).set_index("k")

    print("5-fold CV accuracy as features are added in separation-score order:")
    print(curve.assign(mean=curve["mean"].round(3), std=curve["std"].round(3)).to_string(), "\n")

    full = curve.loc[len(ranking), "mean"]
    TOL = 0.02
    cheapest = curve[curve["mean"] >= full - TOL].index.min()
    print(f"all 13 features: {full:.3f}")
    print(f"top-2 features:  {curve.loc[2, 'mean']:.3f}  (gap {full - curve.loc[2, 'mean']:.3f})")
    print(f"smallest k within {TOL:.0%} of the full model: k={cheapest} ({curve.loc[cheapest, 'mean']:.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].errorbar(curve.index, curve["mean"], yerr=curve["std"], marker="o",
                     capsize=3, color="#4C72B0")
    axes[0].axhline(full, ls="--", lw=1, color="#888", label=f"all 13 features ({full:.3f})")
    axes[0].axvline(cheapest, ls=":", lw=1, color="#C44E52", label=f"k={cheapest} is within {TOL:.0%}")
    axes[0].set_xlabel("k = number of top-ranked features used")
    axes[0].set_ylabel("5-fold CV accuracy")
    axes[0].set_title("Accuracy vs. how many features you keep")
    axes[0].set_xticks(curve.index)
    axes[0].legend(loc="lower right")

    gain = curve["mean"].diff().fillna(curve["mean"])
    colors = ["#55A868" if g >= 0 else "#C44E52" for g in gain]
    axes[1].bar(curve.index, gain, color=colors)
    axes[1].axhline(0, lw=0.8, color="#333")
    axes[1].set_xlabel("feature added (in rank order)")
    axes[1].set_ylabel("change in CV accuracy")
    axes[1].set_title("What each extra feature actually buys")
    axes[1].set_xticks(curve.index)
    axes[1].set_xticklabels([f"{k}\n{curve.loc[k, 'added'][:11]}" for k in curve.index],
                            fontsize=7, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=120)
    print("saved chart.png")


if __name__ == "__main__":
    main()
