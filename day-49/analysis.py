"""
Day 49 — sklearn `wine` (178 rows · 13 features)
Question: How much of the data does the model actually need?
Run: python analysis.py   (writes chart.png next to this file)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tools.eda import run

if __name__ == "__main__":
    run(dataset="wine", analysis="datasize", out=HERE)
