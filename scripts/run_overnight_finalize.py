#!/usr/bin/env python3
"""Post-ladder steps: long-tail weights, freeze, holdout, demo slates.

Run only after the research ladder finishes. Do not inspect holdout MAP
until BEST_SYSTEM.json exists from development reports.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

STAGES = [
    "scripts/run_overnight_longtail_weights.py",
    "scripts/freeze_best_system.py",
    "scripts/export_holdout_candidates.py",
    "scripts/run_overnight_holdout_eval.py",
    "scripts/export_demo_recommendations.py",
]


def main() -> None:
    for script in STAGES:
        print(f"\n=== {script} ===", flush=True)
        completed = subprocess.run([PYTHON, str(ROOT / script)], cwd=ROOT)
        if completed.returncode != 0:
            raise SystemExit(f"{script} failed with {completed.returncode}")


if __name__ == "__main__":
    main()
