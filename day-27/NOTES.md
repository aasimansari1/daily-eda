# Day 27 — sklearn `iris` (150 rows · 4 features)

**Question:** How skewed are the features, and does a log transform fix the worst one?

**Method:** Measured skew for every numeric feature, then compared the worst offender's histogram before and after a log1p transform.

**Insights**
1. **sepal width (cm)** is the most lopsided feature (skew 0.32); **petal width (cm)** is the most symmetric (-0.10).
2. 0 of 4 features have |skew| > 1 — enough that raw means and std-based scaling are misleading summaries for them.
3. log1p only moves sepal width (cm) from 0.32 to -0.36 — the shape is not a simple heavy right tail, so logging is not the fix here.

**Next question this raises:** Does log-transforming the skewed features change which ones a model leans on?

![chart](chart.png)
