"""
Day 68 — sklearn `california_housing` (6,000 rows · 8 features)
Question: How much of the data does the model actually need?
Run: python analysis.py   (writes chart.png next to this file)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tools.eda import run

if __name__ == "__main__":
    run(dataset="california_housing", analysis="datasize", out=HERE)
