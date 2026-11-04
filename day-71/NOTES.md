# Day 71 — seaborn `car_crashes` (51 rows · 7 features)

**Question:** How much of the data does the model actually need?

**Method:** Ran a learning curve for a scaled linear model over 10 training-set sizes with 5-fold CV, tracking train and cross-validated scores.

**Insights**
1. CV R² reaches 0.936 at the full 40 training rows, up from 0.591 at 4 rows.
2. **28 rows (70% of the data) already get within 2% of the final score** — the curve is flat well before the data runs out.
3. Train sits 0.024 above CV at full size, so the model is not memorising — more data would not help much.

**Next question this raises:** Does a non-linear model keep improving where the linear one flattens out?

![chart](chart.png)
