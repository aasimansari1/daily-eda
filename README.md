# 📊 daily-eda

One small, real exploratory data analysis every day — a dataset, a question, a chart, three insights.

The goal is a habit, not a masterpiece: **10–15 minutes**, one folder per day, everything reproducible.

## Structure

```
daily-eda/
├── template/        # copy this to start a new day
│   ├── analysis.py
│   └── NOTES.md
├── day-01/          # Wine quality — what separates the three cultivars?
│   ├── analysis.py
│   ├── chart.png
│   └── NOTES.md
└── ...
```

## How a day works

1. `cp -r template day-NN`
2. Pick a dataset (built-in sklearn/seaborn sets, Kaggle, a CSV from anywhere).
3. Ask **one** question. Answer it in `analysis.py`, save one chart.
4. Write 3 bullet insights in `NOTES.md`.
5. Commit: `git commit -m "day-NN: <dataset> — <question>"`.

## Run any day

```bash
pip install pandas matplotlib scikit-learn seaborn
python day-01/analysis.py
```

## Log

| Day | Dataset | Question |
|---|---|---|
| 01 | sklearn `wine` | Which chemical features separate the three cultivars most? |

---

Maintained by [Mohd Aasim Ansari](https://github.com/aasimansari1) · MIT
