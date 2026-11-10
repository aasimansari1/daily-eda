# Day 77 — sklearn `california_housing` (6,000 rows · 8 features)

**Question:** Do the top-ranked features stay on top once a model sees them together?

**Method:** Ranked features one at a time by univariate signal, then fitted a scaled linear model on all of them and compared the two rankings place by place.

**Insights**
1. The fitted model leans hardest on **Latitude**, which the univariate ranking only placed #2 of 8.
2. 4 of the top 5 univariate features are still in the model's top 5 (Spearman rho = 0.67 across all 8) — the two views mostly agree.
3. **Longitude** moves the furthest, 3 places, which is what correlated features do once they have to share the credit.

**Next question this raises:** If the rankings disagree, which one picks the better small subset?

![chart](chart.png)
