# Day 20 — sklearn `diabetes` (442 rows · 10 features)

**Question:** Which features are redundant with each other?

**Method:** Computed the full correlation matrix over numeric features and ranked every pair by |r|.

**Insights**
1. **s1** and **s2** are the most redundant pair at |r| = 0.90 (positive) — near-duplicates for a linear model.
2. 1 of 100 feature pairs exceed |r| = 0.8, so the 10 columns hold noticeably fewer than 10 independent dimensions.
3. Typical pair correlation is only |r| = 0.32, meaning the redundancy is concentrated in a few clusters rather than spread across the table.

**Next question this raises:** How many principal components does it take to keep 95% of the variance?

![chart](chart.png)
