# Day 88 — sklearn `diabetes` (442 rows · 10 features)

**Question:** Would recursive feature elimination find a better subset of the same size?

**Method:** Scored every budget k two ways — the top-k univariate features, and the k features RFE keeps when it drops the weakest coefficients step by step — under identical 5-fold splits.

**Insights**
1. RFE beats rank-and-add at 4 of the 8 budgets tested, by at most 0.020 R² (at k = 8).
2. At k = 5 the two methods agree on 3 of 5 features (**bmi**, **bp**, **s5**); RFE swaps in **s1**, **s2**.
3. The full table scores 0.489 R², so the selector earns its compute here.

**Next question this raises:** Does the winning subset still win with a non-linear model?

![chart](chart.png)
