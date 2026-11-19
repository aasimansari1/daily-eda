# Day 86 — sklearn `iris` (150 rows · 4 features)

**Question:** Would recursive feature elimination find a better subset of the same size?

**Method:** Scored every budget k two ways — the top-k univariate features, and the k features RFE keeps when it drops the weakest coefficients step by step — under identical 5-fold splits.

**Insights**
1. RFE beats rank-and-add at 0 of the 4 budgets tested, by at most 0.000 accuracy (at k = 1).
2. At k = 4 the two methods agree on 4 of 4 features (**petal length (cm)**, **petal width (cm)**, **sepal length (cm)**, **sepal width (cm)**); RFE swaps in nothing.
3. The full table scores 0.960 accuracy, so ranking univariately is good enough on this data.

**Next question this raises:** Does the winning subset still win with a non-linear model?

![chart](chart.png)
