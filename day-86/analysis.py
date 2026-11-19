"""
Day 86 — sklearn `iris` (150 rows · 4 features)
Question: Would recursive feature elimination find a better subset of the same size?
Run: python analysis.py   (writes chart.png next to this file)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tools.eda import run

if __name__ == "__main__":
    run(dataset="iris", analysis="rfe", out=HERE)
