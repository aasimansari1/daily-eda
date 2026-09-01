"""
Day 10 — sklearn `iris` (150 rows · 4 features)
Question: How much accuracy do you lose keeping only the top-k features?
Run: python analysis.py   (writes chart.png next to this file)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tools.eda import run

if __name__ == "__main__":
    run(dataset="iris", analysis="topk", out=HERE)
