"""Create the next day-NN folder: run a real analysis, save the chart, write the notes.

Usage:
    python tools/generate_day.py            # create the next day
    python tools/generate_day.py --dry-run  # pick + run it, write nothing
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import eda  # noqa: E402

DAY_RE = re.compile(r"^day-(\d+)$")
USED_RE = re.compile(r'run\(dataset="([^"]+)", analysis="([^"]+)"')

ANALYSIS_FILE = '''"""
Day {day:02d} — {title} ({shape})
Question: {question}
Run: python analysis.py   (writes chart.png next to this file)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tools.eda import run

if __name__ == "__main__":
    run(dataset="{dataset}", analysis="{analysis}", out=HERE)
'''

NOTES_FILE = """# Day {day:02d} — {title} ({shape})

**Question:** {question}

**Method:** {method}

**Insights**
1. {i1}
2. {i2}
3. {i3}

**Next question this raises:** {next_question}

![chart](chart.png)
"""


def existing_days() -> list[int]:
    return sorted(int(m.group(1)) for p in REPO.iterdir()
                  if p.is_dir() and (m := DAY_RE.match(p.name)))


def used_pairs() -> set[tuple[str, str]]:
    pairs = set()
    for p in REPO.glob("day-*/analysis.py"):
        if m := USED_RE.search(p.read_text()):
            pairs.add((m.group(1), m.group(2)))
    return pairs


def pick(day: int) -> tuple[str, str]:
    """Rotate analysis-first so consecutive days change question, not just dataset."""
    used = used_pairs()
    datasets = [d for d in eda.DATASETS if eda.available(d)]
    if not datasets:
        raise SystemExit("no datasets available")

    combos = [(d, a) for a in eda.ANALYSES for d in datasets]
    start = (day - 1) % len(combos)
    order = combos[start:] + combos[:start]

    for dataset, analysis in order:
        if (dataset, analysis) in used:
            continue
        try:
            if eda.valid(dataset, analysis):
                return dataset, analysis
        except Exception as exc:  # a dataset that fails to load or analyse
            print(f"skip {dataset}/{analysis}: {exc}", file=sys.stderr)
    raise SystemExit("every dataset/analysis combination has been used")


def update_readme(day: int, title: str, question: str) -> None:
    readme = REPO / "README.md"
    text = readme.read_text()
    row = f"| {day:02d} | {title} | {question} |"
    if row in text:
        return
    lines = text.rstrip().split("\n")
    last = max(i for i, ln in enumerate(lines) if ln.startswith("| "))
    lines.insert(last + 1, row)
    readme.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    day = (max(existing_days()) + 1) if existing_days() else 1
    dataset, analysis = pick(day)
    print(f"day-{day:02d}: {dataset} / {analysis}\n")

    out = REPO / f"day-{day:02d}"
    if not args.dry_run:
        out.mkdir(exist_ok=True)
    result = eda.run(dataset, analysis, out=None if args.dry_run else out)
    ds = eda.load(dataset)

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return

    fields = dict(day=day, title=ds.title, shape=ds.shape_line(), dataset=dataset,
                  analysis=analysis, question=result.question)
    (out / "analysis.py").write_text(ANALYSIS_FILE.format(**fields))
    (out / "NOTES.md").write_text(NOTES_FILE.format(
        **fields, method=result.method, next_question=result.next_question,
        i1=result.insights[0], i2=result.insights[1], i3=result.insights[2]))
    update_readme(day, ds.title, result.question)
    print(f"\nwrote {out.relative_to(REPO)}/ and updated README.md")


if __name__ == "__main__":
    main()
