# Day 72 — sklearn `wine` (178 rows · 13 features)

**Question:** Do the top-ranked features stay on top once a model sees them together?

**Method:** Ranked features one at a time by univariate signal, then fitted a scaled linear model on all of them and compared the two rankings place by place.

**Insights**
1. The fitted model leans hardest on **proline**, which the univariate ranking only placed #3 of 13.
2. 3 of the top 5 univariate features are still in the model's top 5 (Spearman rho = 0.52 across all 13) — the two views disagree more than they agree.
3. **ash** moves the furthest, 8 places, which is what correlated features do once they have to share the credit.

**Next question this raises:** If the rankings disagree, which one picks the better small subset?

![chart](chart.png)
