"""
Day 15 — seaborn `mpg` (398 rows · 8 features)
Question: How much R² do you lose keeping only the top-k features?
Run: python analysis.py   (writes chart.png next to this file)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tools.eda import run

if __name__ == "__main__":
    run(dataset="mpg", analysis="topk", out=HERE)
