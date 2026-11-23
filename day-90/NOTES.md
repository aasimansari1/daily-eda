# Day 90 — sklearn `california_housing` (6,000 rows · 8 features)

**Question:** Would recursive feature elimination find a better subset of the same size?

**Method:** Scored every budget k two ways — the top-k univariate features, and the k features RFE keeps when it drops the weakest coefficients step by step — under identical 5-fold splits.

**Insights**
1. RFE beats rank-and-add at 3 of the 8 budgets tested, by at most 0.110 R² (at k = 3).
2. At k = 5 the two methods agree on 4 of 5 features (**AveRooms**, **Latitude**, **Longitude**, **MedInc**); RFE swaps in **AveBedrms**.
3. The full table scores -5.829 R², so the selector earns its compute here.

**Next question this raises:** Does the winning subset still win with a non-linear model?

![chart](chart.png)
