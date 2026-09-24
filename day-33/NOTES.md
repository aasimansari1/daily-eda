# Day 33 — seaborn `mpg` (398 rows · 8 features)

**Question:** How skewed are the features, and does a log transform fix the worst one?

**Method:** Measured skew for every numeric feature, then compared the worst offender's histogram before and after a log1p transform.

**Insights**
1. **horsepower** is the most lopsided feature (skew 1.09); **model_year** is the most symmetric (0.02).
2. 1 of 6 features have |skew| > 1 — enough that raw means and std-based scaling are misleading summaries for them.
3. log1p only moves horsepower from 1.09 to -1.19 — the shape is not a simple heavy right tail, so logging is not the fix here.

**Next question this raises:** Does log-transforming the skewed features change which ones a model leans on?

![chart](chart.png)
