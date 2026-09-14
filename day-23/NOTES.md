# Day 23 — seaborn `titanic` (891 rows · 10 features)

**Question:** Which features are redundant with each other?

**Method:** Computed the full correlation matrix over numeric features and ranked every pair by |r|.

**Insights**
1. **pclass** and **fare** are the most redundant pair at |r| = 0.55 (negative) — near-duplicates for a linear model.
2. 0 of 25 feature pairs exceed |r| = 0.8, so the 5 columns hold noticeably fewer than 5 independent dimensions.
3. Typical pair correlation is only |r| = 0.23, meaning the redundancy is concentrated in a few clusters rather than spread across the table.

**Next question this raises:** How many principal components does it take to keep 95% of the variance?

![chart](chart.png)
