# 📊 daily-eda

One small, real exploratory data analysis every day — a dataset, a question, a chart, three insights.

The goal is a habit, not a masterpiece: **10–15 minutes**, one folder per day, everything reproducible.

## Structure

```
daily-eda/
├── tools/           # shared dataset loaders + analyses, and the day generator
│   ├── eda.py
│   └── generate_day.py
├── template/        # copy this to start a new day by hand
│   ├── analysis.py
│   └── NOTES.md
├── day-01/          # Wine quality — what separates the three cultivars?
│   ├── analysis.py
│   ├── chart.png
│   └── NOTES.md
├── day-02/          # Wine quality — how many of those features do you need?
│   ├── analysis.py
│   ├── chart.png
│   └── NOTES.md
└── ...
```

## How a day works

A day is one dataset, one question, one chart, three insights.

```bash
python tools/generate_day.py --dry-run   # see what the next day would be
python tools/generate_day.py             # write day-NN/ and update the log
```

The generator picks the next unused `(dataset, question)` pair, runs the analysis for real,
saves `chart.png`, and writes `NOTES.md` with the numbers it actually computed.
To do a day by hand instead, `cp -r template day-NN` and fill it in.

**This runs on a schedule.** `.github/workflows/daily-eda.yml` fires daily at 04:23 UTC,
generates the next day and commits it. Days from 03 onward come from that workflow.

## Run any day

```bash
pip install -r requirements.txt
python day-01/analysis.py     # re-runs that day's analysis and redraws chart.png
```

## Questions in rotation

`signal` · `topk` · `redundancy` · `skew` · `outliers` · `groups` · `datasize`
— crossed with 9 datasets (sklearn `wine`/`iris`/`breast_cancer`/`diabetes`,
seaborn `penguins`/`tips`/`titanic`/`mpg`/`diamonds`) for 57 unique days.

## Log

| Day | Dataset | Question |
|---|---|---|
| 01 | sklearn `wine` | Which chemical features separate the three cultivars most? |
| 02 | sklearn `wine` | How much accuracy is lost using only the top-k features? |

---

Maintained by [Mohd Aasim Ansari](https://github.com/aasimansari1) · MIT
