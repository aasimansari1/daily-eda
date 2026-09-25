# Day 34 — seaborn `diamonds` (6,000 rows · 9 features)

**Question:** How skewed are the features, and does a log transform fix the worst one?

**Method:** Measured skew for every numeric feature, then compared the worst offender's histogram before and after a log1p transform.

**Insights**
1. **table** is the most lopsided feature (skew 1.32); **z** is the most symmetric (0.33).
2. 2 of 6 features have |skew| > 1 — enough that raw means and std-based scaling are misleading summaries for them.
3. A log1p transform pulls table from 1.32 to -0.29, so the tail is multiplicative and logging fixes it.

**Next question this raises:** Does log-transforming the skewed features change which ones a model leans on?

![chart](chart.png)
