# Day 78 — seaborn `penguins` (344 rows · 6 features)

**Question:** Do the top-ranked features stay on top once a model sees them together?

**Method:** Ranked features one at a time by univariate signal, then fitted a scaled linear model on all of them and compared the two rankings place by place.

**Insights**
1. The fitted model leans hardest on **bill_length_mm**, which the univariate ranking only placed #2 of 4.
2. 4 of the top 4 univariate features are still in the model's top 4 (Spearman rho = -0.20 across all 4) — the two views disagree more than they agree.
3. **flipper_length_mm** moves the furthest, 3 places, which is what correlated features do once they have to share the credit.

**Next question this raises:** If the rankings disagree, which one picks the better small subset?

![chart](chart.png)
