"""
Day 63 — seaborn `planets` (1,035 rows · 5 features)
Question: Where are the outliers, and how much do they move the summary statistics?
Run: python analysis.py   (writes chart.png next to this file)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tools.eda import run

if __name__ == "__main__":
    run(dataset="planets", analysis="outliers", out=HERE)
