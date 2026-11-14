# Day 81 — seaborn `diamonds` (6,000 rows · 9 features)

**Question:** Do the top-ranked features stay on top once a model sees them together?

**Method:** Ranked features one at a time by univariate signal, then fitted a scaled linear model on all of them and compared the two rankings place by place.

**Insights**
1. The fitted model leans hardest on **carat**, which the univariate ranking only placed #1 of 6.
2. 4 of the top 5 univariate features are still in the model's top 5 (Spearman rho = 0.71 across all 6) — the two views mostly agree.
3. **depth** moves the furthest, 2 places, which is what correlated features do once they have to share the credit.

**Next question this raises:** If the rankings disagree, which one picks the better small subset?

![chart](chart.png)
