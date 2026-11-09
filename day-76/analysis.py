"""
Day 76 — sklearn `digits` (1,797 rows · 61 features)
Question: Do the top-ranked features stay on top once a model sees them together?
Run: python analysis.py   (writes chart.png next to this file)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tools.eda import run

if __name__ == "__main__":
    run(dataset="digits", analysis="weights", out=HERE)
