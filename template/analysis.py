"""
Day NN — <dataset>
Question: <one question this analysis answers>
Run: python analysis.py   (writes chart.png next to this file)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent


def load() -> pd.DataFrame:
    # TODO: load your dataset
    raise NotImplementedError


def main() -> None:
    df = load()
    print(df.head())
    print(df.describe())

    fig, ax = plt.subplots(figsize=(8, 5))
    # TODO: one chart that answers the question
    fig.tight_layout()
    fig.savefig(HERE / "chart.png", dpi=120)
    print("saved chart.png")


if __name__ == "__main__":
    main()
