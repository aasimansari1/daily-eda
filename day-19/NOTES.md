# Day 19 — sklearn `breast_cancer` (569 rows · 30 features)

**Question:** Which features are redundant with each other?

**Method:** Computed the full correlation matrix over numeric features and ranked every pair by |r|.

**Insights**
1. **mean radius** and **mean perimeter** are the most redundant pair at |r| = 1.00 (positive) — near-duplicates for a linear model.
2. 44 of 900 feature pairs exceed |r| = 0.8, so the 30 columns hold noticeably fewer than 30 independent dimensions.
3. Typical pair correlation is only |r| = 0.39, meaning the redundancy is concentrated in a few clusters rather than spread across the table.

**Next question this raises:** How many principal components does it take to keep 95% of the variance?

![chart](chart.png)
