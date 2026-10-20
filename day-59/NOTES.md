# Day 59 — sklearn `iris` (150 rows · 4 features)

**Question:** Which features carry the most signal about the target?

**Method:** Scored every numeric feature by range of class means / std, ranked them, and plotted the top two against each other.

**Insights**
1. **petal width (cm)** carries the most signal (2.34 on range of class means / std), followed by **petal length (cm)** (2.32).
2. The top feature is **2× stronger** than the weakest (**sepal width (cm)**, 1.51) — the signal is far from evenly spread.
3. 0 of 4 features (0%) score below a quarter of the leader — the useful signal is spread across a wide middle tier, not just the top few.

**Next question this raises:** Do the top-ranked features stay on top once a model sees them together?

![chart](chart.png)
