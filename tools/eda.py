"""Shared dataset loaders and analyses for daily-eda.

Every day folder is a thin wrapper that calls run(dataset=..., analysis=...)
from here, so any day can be re-run with `python day-NN/analysis.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 0
BLUE, GREEN, RED, GREY = "#4C72B0", "#55A868", "#C44E52", "#8C8C8C"


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------
@dataclass
class Dataset:
    key: str
    title: str          # shown in headings, e.g. "sklearn `wine`"
    df: pd.DataFrame
    target: str | None
    kind: str           # "classification" | "regression"

    @property
    def numeric(self) -> list[str]:
        cols = self.df.select_dtypes("number").columns.tolist()
        return [c for c in cols if c != self.target]

    @property
    def categorical(self) -> list[str]:
        cols = self.df.columns.drop(self.numeric)
        return [c for c in cols if c != self.target and self.df[c].nunique() <= 20]

    def shape_line(self) -> str:
        n, p = self.df.shape
        return f"{n:,} rows · {p - 1} features"


def _sklearn(name: str):
    from sklearn import datasets as skd

    loader = {
        "wine": skd.load_wine,
        "iris": skd.load_iris,
        "breast_cancer": skd.load_breast_cancer,
        "diabetes": skd.load_diabetes,
    }[name]
    data = loader()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    if name == "diabetes":
        df["target"] = data.target
        return Dataset(name, f"sklearn `{name}`", df, "target", "regression")
    df["target"] = pd.Categorical.from_codes(data.target, data.target_names)
    return Dataset(name, f"sklearn `{name}`", df, "target", "classification")


_SEABORN = {
    "penguins": ("species", "classification"),
    "tips": ("tip", "regression"),
    "titanic": ("survived", "classification"),
    "mpg": ("mpg", "regression"),
    "diamonds": ("price", "regression"),
}


def _seaborn(name: str):
    import seaborn as sns

    target, kind = _SEABORN[name]
    df = sns.load_dataset(name)
    if name == "titanic":  # drop the columns that leak the target outright
        df = df.drop(columns=["alive", "adult_male", "who", "deck"], errors="ignore")
        df["survived"] = df["survived"].astype("category")
    if name == "diamonds":
        df = df.sample(6000, random_state=SEED).reset_index(drop=True)
    return Dataset(name, f"seaborn `{name}`", df, target, kind)


DATASETS = ["wine", "iris", "breast_cancer", "diabetes",
            "penguins", "tips", "titanic", "mpg", "diamonds"]


def load(name: str) -> Dataset:
    if name in _SEABORN:
        return _seaborn(name)
    return _sklearn(name)


def available(name: str) -> bool:
    """seaborn sets are fetched over the network — skip them if that fails."""
    try:
        load(name)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# analyses — each returns (question, insights, next_question) and draws a figure
# --------------------------------------------------------------------------
@dataclass
class Result:
    question: str
    method: str
    insights: list[str]
    next_question: str
    fig: plt.Figure


def _model(kind: str):
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    est = (LogisticRegression(max_iter=2000, random_state=SEED)
           if kind == "classification" else LinearRegression())
    return make_pipeline(StandardScaler(), est)


def _cv(kind: str):
    from sklearn.model_selection import KFold, StratifiedKFold

    if kind == "classification":
        return StratifiedKFold(5, shuffle=True, random_state=SEED)
    return KFold(5, shuffle=True, random_state=SEED)


def _xy(ds: Dataset):
    df = ds.df[ds.numeric + [ds.target]].dropna()
    return df[ds.numeric], df[ds.target]


def _score_name(kind: str) -> str:
    return "accuracy" if kind == "classification" else "R²"


def _ranked_features(ds: Dataset) -> pd.Series:
    """Classification: spread of class means / overall std. Regression: |correlation|."""
    X, y = _xy(ds)
    if ds.kind == "classification":
        means = X.groupby(y.values, observed=True).mean()
        return ((means.max() - means.min()) / X.std()).sort_values(ascending=False)
    return X.corrwith(y).abs().sort_values(ascending=False)


# ---- 1. how well does each feature separate / track the target ------------
def a_signal(ds: Dataset) -> Result:
    rank = _ranked_features(ds)
    metric = ("range of class means / std" if ds.kind == "classification"
              else "|correlation with target|")
    X, y = _xy(ds)
    top2 = rank.index[:2].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rank.head(15).plot.barh(ax=axes[0], color=BLUE)
    axes[0].invert_yaxis()
    axes[0].set_title(f"Feature signal ({metric})")
    axes[0].set_xlabel(metric)

    if ds.kind == "classification":
        for name, grp in X.groupby(y.values, observed=True):
            axes[1].scatter(grp[top2[0]], grp[top2[1]], label=str(name), alpha=0.7, s=24)
        axes[1].legend()
    else:
        axes[1].scatter(X[top2[0]], y, alpha=0.5, s=18, color=BLUE)
        axes[1].set_ylabel(ds.target)
    axes[1].set_xlabel(top2[0])
    axes[1].set_ylabel(top2[1] if ds.kind == "classification" else ds.target)
    axes[1].set_title(f"{top2[0]} vs {top2[1] if ds.kind == 'classification' else ds.target}")
    fig.tight_layout()

    lo = rank.index[-1]
    ratio = rank.iloc[0] / max(rank.iloc[-1], 1e-9)
    weak = int((rank < rank.iloc[0] / 4).sum())
    insights = [
        f"**{rank.index[0]}** carries the most signal ({rank.iloc[0]:.2f} on {metric}), "
        f"followed by **{rank.index[1]}** ({rank.iloc[1]:.2f}).",
        f"The top feature is **{ratio:.0f}× stronger** than the weakest (**{lo}**, {rank.iloc[-1]:.2f}) — "
        f"the signal is far from evenly spread.",
        f"{weak} of {len(rank)} features ({weak / len(rank):.0%}) score below a quarter of the leader — "
        f"{'most of the table is close to dead weight on its own' if weak > len(rank) / 2 else 'the useful signal is spread across a wide middle tier, not just the top few'}.",
    ]
    return Result(
        "Which features carry the most signal about the target?",
        f"Scored every numeric feature by {metric}, ranked them, and plotted the top two against each other.",
        insights,
        "Do the top-ranked features stay on top once a model sees them together?",
        fig,
    )


# ---- 2. how many features do you actually need ---------------------------
def a_topk(ds: Dataset) -> Result:
    from sklearn.model_selection import cross_val_score

    rank = _ranked_features(ds)
    X, y = _xy(ds)
    model, cv, metric = _model(ds.kind), _cv(ds.kind), _score_name(ds.kind)
    scoring = "accuracy" if ds.kind == "classification" else "r2"

    rows = []
    for k in range(1, len(rank) + 1):
        cols = rank.index[:k].tolist()
        s = cross_val_score(model, X[cols], y, cv=cv, scoring=scoring)
        rows.append({"k": k, "added": cols[-1], "mean": s.mean(), "std": s.std()})
    curve = pd.DataFrame(rows).set_index("k")

    full = curve["mean"].iloc[-1]
    tol = 0.02
    cheap = int(curve[curve["mean"] >= full - tol].index.min())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].errorbar(curve.index, curve["mean"], yerr=curve["std"], marker="o",
                     capsize=3, color=BLUE)
    axes[0].axhline(full, ls="--", lw=1, color=GREY, label=f"all {len(rank)} ({full:.3f})")
    axes[0].axvline(cheap, ls=":", lw=1, color=RED, label=f"k={cheap} within {tol:.0%}")
    axes[0].set_xlabel("k = top-ranked features kept")
    axes[0].set_ylabel(f"5-fold CV {metric}")
    axes[0].set_title(f"{metric} vs. how many features you keep")
    axes[0].legend(loc="lower right")

    gain = curve["mean"].diff().fillna(curve["mean"])
    axes[1].bar(curve.index, gain, color=[GREEN if g >= 0 else RED for g in gain])
    axes[1].axhline(0, lw=0.8, color="#333")
    axes[1].set_title("What each extra feature actually buys")
    axes[1].set_xlabel("feature added (rank order)")
    axes[1].set_ylabel(f"change in CV {metric}")
    fig.tight_layout()

    negatives = int((gain.iloc[1:] < 0).sum())
    insights = [
        f"The single best feature already reaches {curve['mean'].iloc[0]:.3f} {metric}; "
        f"all {len(rank)} together reach {full:.3f}.",
        f"**{cheap} features get within {tol:.0%} of the full model** ({curve.loc[cheap, 'mean']:.3f}) — "
        f"the other {len(rank) - cheap} buy almost nothing.",
        f"{negatives} of the {len(rank) - 1} additions *lowered* CV {metric}, "
        f"so rank-order-and-add is a rough heuristic, not a feature selector.",
    ]
    return Result(
        f"How much {metric} do you lose keeping only the top-k features?",
        f"Ranked features by signal, then added them one at a time to a scaled linear model "
        f"and scored each subset with 5-fold CV.",
        insights,
        "Would recursive feature elimination find a better subset of the same size?",
        fig,
    )


# ---- 3. redundancy between features --------------------------------------
def a_redundancy(ds: Dataset) -> Result:
    X = ds.df[ds.numeric].dropna()
    corr = X.corr()
    tri = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
    pairs = tri.abs().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    im = axes[0].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    axes[0].set_yticks(range(len(corr)), corr.columns, fontsize=7)
    axes[0].set_title("Correlation between features")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    top = pairs.head(10)[::-1]
    axes[1].barh([f"{a} ↔ {b}" for a, b in top.index], top.values,
                 color=[RED if tri[i] < 0 else BLUE for i in top.index])
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("|correlation|")
    axes[1].set_title("Most redundant pairs")
    axes[1].tick_params(labelsize=7)
    fig.tight_layout()

    (a, b), v = pairs.index[0], pairs.iloc[0]
    strong = int((pairs > 0.8).sum())
    mean_abs = pairs.mean()
    insights = [
        f"**{a}** and **{b}** are the most redundant pair at |r| = {v:.2f} "
        f"({'positive' if tri[(a, b)] > 0 else 'negative'}) — near-duplicates for a linear model.",
        f"{strong} of {len(pairs)} feature pairs exceed |r| = 0.8, "
        f"so the {len(corr)} columns hold noticeably fewer than {len(corr)} independent dimensions.",
        f"Typical pair correlation is only |r| = {mean_abs:.2f}, meaning the redundancy is "
        f"concentrated in a few clusters rather than spread across the table.",
    ]
    return Result(
        "Which features are redundant with each other?",
        "Computed the full correlation matrix over numeric features and ranked every pair by |r|.",
        insights,
        "How many principal components does it take to keep 95% of the variance?",
        fig,
    )


# ---- 4. distribution shape ------------------------------------------------
def a_skew(ds: Dataset) -> Result:
    X = ds.df[ds.numeric].dropna()
    skew = X.skew().sort_values(key=abs, ascending=False)
    worst = skew.index[0]
    shifted = X[worst] - X[worst].min()
    logged = np.log1p(shifted)
    after = float(pd.Series(logged).skew())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    skew.head(15)[::-1].plot.barh(ax=axes[0], color=[RED if v > 0 else BLUE for v in skew.head(15)[::-1]])
    axes[0].set_title("Skew by feature")
    axes[0].set_xlabel("skew")
    axes[0].tick_params(labelsize=7)
    axes[1].hist(X[worst], bins=40, color=BLUE)
    axes[1].set_title(f"{worst} (skew {skew.iloc[0]:.2f})")
    axes[2].hist(logged, bins=40, color=GREEN)
    axes[2].set_title(f"log1p({worst}) (skew {after:.2f})")
    fig.tight_layout()

    heavy = int((skew.abs() > 1).sum())
    fixed = abs(after) < abs(skew.iloc[0]) / 2
    insights = [
        f"**{worst}** is the most lopsided feature (skew {skew.iloc[0]:.2f}); "
        f"**{skew.index[-1]}** is the most symmetric ({skew.iloc[-1]:.2f}).",
        f"{heavy} of {len(skew)} features have |skew| > 1 — enough that raw means and "
        f"std-based scaling are misleading summaries for them.",
        (f"A log1p transform pulls {worst} from {skew.iloc[0]:.2f} to {after:.2f}, so the tail is "
         f"multiplicative and logging fixes it." if fixed else
         f"log1p only moves {worst} from {skew.iloc[0]:.2f} to {after:.2f} — the shape is not a "
         f"simple heavy right tail, so logging is not the fix here."),
    ]
    return Result(
        "How skewed are the features, and does a log transform fix the worst one?",
        "Measured skew for every numeric feature, then compared the worst offender's histogram "
        "before and after a log1p transform.",
        insights,
        "Does log-transforming the skewed features change which ones a model leans on?",
        fig,
    )


# ---- 5. outliers ----------------------------------------------------------
def a_outliers(ds: Dataset) -> Result:
    X = ds.df[ds.numeric].dropna()
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = q3 - q1
    mask = (X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)
    counts = (mask.sum() / len(X) * 100).sort_values(ascending=False)
    worst = counts.index[0]
    with_out, without = X[worst].mean(), X.loc[~mask[worst], worst].mean()
    shift = abs(with_out - without) / max(X[worst].std(), 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    counts.head(15)[::-1].plot.barh(ax=axes[0], color=RED)
    axes[0].set_title("Share of rows flagged as outliers (1.5×IQR)")
    axes[0].set_xlabel("% of rows")
    axes[0].tick_params(labelsize=7)
    axes[1].boxplot([X[c] / max(X[c].std(), 1e-9) for c in counts.head(8).index],
                    tick_labels=counts.head(8).index, vert=True)
    axes[1].set_title("Top-8 offenders (scaled to unit std)")
    axes[1].tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout()

    any_row = int(mask.any(axis=1).sum())
    insights = [
        f"**{worst}** has the most extreme values — {counts.iloc[0]:.1f}% of rows fall outside 1.5×IQR.",
        f"{any_row} of {len(X)} rows ({any_row / len(X) * 100:.1f}%) are an outlier on at least one "
        f"feature, so dropping outlier rows wholesale would cost a large slice of the data.",
        f"Trimming {worst}'s outliers moves its mean by {shift:.2f} std "
        f"({with_out:.2f} → {without:.2f}) — {'a material shift' if shift > 0.1 else 'barely anything'}.",
    ]
    return Result(
        "Where are the outliers, and how much do they move the summary statistics?",
        "Flagged values outside 1.5×IQR per feature, counted the share of affected rows, and "
        "recomputed the worst feature's mean with them removed.",
        insights,
        "Does a robust scaler beat a standard scaler on this data?",
        fig,
    )


# ---- 6. group gaps --------------------------------------------------------
def a_groups(ds: Dataset) -> Result:
    cats = ds.categorical
    df = ds.df.dropna(subset=[ds.target])
    if ds.kind == "regression":
        by = {c: df.groupby(c, observed=True)[ds.target].mean() for c in cats}
        spread = {c: s.max() - s.min() for c, s in by.items()}
        unit = ds.target
    else:
        pos = df[ds.target].astype(str).unique()[0]
        by = {c: df.groupby(c, observed=True)[ds.target].apply(lambda s: (s.astype(str) == pos).mean())
              for c in cats}
        spread = {c: s.max() - s.min() for c, s in by.items()}
        unit = f"P({ds.target} = {pos})"
    best = max(spread, key=spread.get)
    s = by[best].sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    s.plot.barh(ax=axes[0], color=BLUE)
    axes[0].set_title(f"{unit} by {best}")
    axes[0].set_xlabel(unit)
    pd.Series(spread).sort_values().plot.barh(ax=axes[1], color=GREEN)
    axes[1].set_title("Spread of group means, by grouping column")
    axes[1].set_xlabel(f"max − min of {unit}")
    fig.tight_layout()

    overall = (df[ds.target].mean() if ds.kind == "regression"
               else (df[ds.target].astype(str) == pos).mean())
    insights = [
        f"**{best}** splits the target hardest: {unit} runs from {s.iloc[0]:.3f} "
        f"(**{s.index[0]}**) to {s.iloc[-1]:.3f} (**{s.index[-1]}**).",
        f"That is a {spread[best]:.3f} gap against an overall level of {overall:.3f} — "
        f"{spread[best] / max(abs(overall), 1e-9) * 100:.0f}% of the baseline.",
        f"The weakest grouping (**{min(spread, key=spread.get)}**) spans only "
        f"{min(spread.values()):.3f}, so not every category column is worth encoding.",
    ]
    return Result(
        "Which categorical column splits the target most?",
        f"Grouped by each low-cardinality category column, computed {unit} per group, "
        f"and compared the spread of group means across columns.",
        insights,
        "Do those group gaps survive once the numeric features are controlled for?",
        fig,
    )


# ---- 7. how much data do you need ----------------------------------------
def a_datasize(ds: Dataset) -> Result:
    from sklearn.model_selection import learning_curve

    X, y = _xy(ds)
    metric = _score_name(ds.kind)
    scoring = "accuracy" if ds.kind == "classification" else "r2"
    sizes, train, test = learning_curve(
        _model(ds.kind), X, y, cv=_cv(ds.kind), scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=SEED)
    tr, te = train.mean(1), test.mean(1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, tr, marker="o", color=GREEN, label="train")
    ax.plot(sizes, te, marker="o", color=BLUE, label="cross-validated")
    ax.fill_between(sizes, test.mean(1) - test.std(1), test.mean(1) + test.std(1),
                    color=BLUE, alpha=0.15)
    ax.set_xlabel("training rows")
    ax.set_ylabel(f"{metric}")
    ax.set_title(f"Learning curve — {ds.title}")
    ax.legend()
    fig.tight_layout()

    full = te[-1]
    reach = sizes[np.argmax(te >= full - 0.02)]
    gap = tr[-1] - te[-1]
    insights = [
        f"CV {metric} reaches {full:.3f} at the full {int(sizes[-1])} training rows, "
        f"up from {te[0]:.3f} at {int(sizes[0])} rows.",
        f"**{int(reach)} rows ({reach / sizes[-1] * 100:.0f}% of the data) already get within 2% "
        f"of the final score** — the curve is flat well before the data runs out.",
        (f"Train sits {gap:.3f} above CV at full size, so the model is "
         f"{'overfitting and more data would still help' if gap > 0.05 else 'not memorising — more data would not help much'}."),
    ]
    return Result(
        "How much of the data does the model actually need?",
        "Ran a learning curve for a scaled linear model over 10 training-set sizes with 5-fold CV, "
        "tracking train and cross-validated scores.",
        insights,
        "Does a non-linear model keep improving where the linear one flattens out?",
        fig,
    )


ANALYSES = {
    "signal": (a_signal, "Which features carry the most signal?"),
    "topk": (a_topk, "How much score is lost keeping only the top-k features?"),
    "redundancy": (a_redundancy, "Which features are redundant with each other?"),
    "skew": (a_skew, "How skewed are the features, and does a log fix the worst?"),
    "outliers": (a_outliers, "Where are the outliers and what do they move?"),
    "groups": (a_groups, "Which categorical column splits the target most?"),
    "datasize": (a_datasize, "How much of the data does the model need?"),
}

# analyses that need a usable categorical column
NEEDS_CATEGORICAL = {"groups"}


def valid(dataset: str, analysis: str, ds: Dataset | None = None) -> bool:
    ds = ds or load(dataset)
    if analysis in NEEDS_CATEGORICAL and not ds.categorical:
        return False
    if analysis in {"topk", "signal"} and len(ds.numeric) < 3:
        return False
    return len(ds.numeric) >= 2


def run(dataset: str, analysis: str, out: Path | None = None) -> Result:
    """Run one day's analysis; save chart.png next to `out` and print the findings."""
    ds = load(dataset)
    result = ANALYSES[analysis][0](ds)
    print(f"{ds.title} — {ds.shape_line()}")
    print(f"Question: {result.question}\n")
    for i, line in enumerate(result.insights, 1):
        print(f"{i}. {line.replace('**', '')}")
    if out is not None:
        result.fig.savefig(out / "chart.png", dpi=120)
        print(f"\nsaved {out / 'chart.png'}")
    return result
