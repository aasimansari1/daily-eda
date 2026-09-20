"""
Day 29 — sklearn `diabetes` (442 rows · 10 features)
Question: How skewed are the features, and does a log transform fix the worst one?
Run: python analysis.py   (writes chart.png next to this file)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tools.eda import run

if __name__ == "__main__":
    run(dataset="diabetes", analysis="skew", out=HERE)
