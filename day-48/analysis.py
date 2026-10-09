"""
Day 48 — seaborn `diamonds` (6,000 rows · 9 features)
Question: Which categorical column splits the target most?
Run: python analysis.py   (writes chart.png next to this file)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tools.eda import run

if __name__ == "__main__":
    run(dataset="diamonds", analysis="groups", out=HERE)
