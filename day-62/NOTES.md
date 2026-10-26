# Day 62 — seaborn `taxis` (6,433 rows · 12 features)

**Question:** Where are the outliers, and how much do they move the summary statistics?

**Method:** Flagged values outside 1.5×IQR per feature, counted the share of affected rows, and recomputed the worst feature's mean with them removed.

**Insights**
1. **distance** has the most extreme values — 11.4% of rows fall outside 1.5×IQR.
2. 1253 of 6433 rows (19.5%) are an outlier on at least one feature, so dropping outlier rows wholesale would cost a large slice of the data.
3. Trimming distance's outliers moves its mean by 0.30 std (3.02 → 1.87) — a material shift.

**Next question this raises:** Does a robust scaler beat a standard scaler on this data?

![chart](chart.png)
