# Day 30 — seaborn `penguins` (344 rows · 6 features)

**Question:** How skewed are the features, and does a log transform fix the worst one?

**Method:** Measured skew for every numeric feature, then compared the worst offender's histogram before and after a log1p transform.

**Insights**
1. **body_mass_g** is the most lopsided feature (skew 0.47); **bill_length_mm** is the most symmetric (0.05).
2. 0 of 4 features have |skew| > 1 — enough that raw means and std-based scaling are misleading summaries for them.
3. log1p only moves body_mass_g from 0.47 to -3.12 — the shape is not a simple heavy right tail, so logging is not the fix here.

**Next question this raises:** Does log-transforming the skewed features change which ones a model leans on?

![chart](chart.png)
