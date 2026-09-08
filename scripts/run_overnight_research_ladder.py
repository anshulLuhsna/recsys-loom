#!/usr/bin/env python3
"""Run remaining overnight ranking stages in a conservative order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

STAGES = [
    "scripts/analyze_ranking_failures.py",
    "scripts/run_overnight_lgbm_tune.py",
    "scripts/run_overnight_hard_negatives.py",
    "scripts/run_overnight_cross_features.py",
    "scripts/export_overnight_extra_snapshots.py",
    "scripts/run_overnight_scaled_supervision.py",
    "scripts/run_overnight_catboost.py",
    "scripts/run_overnight_listwise.py",
]


def main() -> None:
    for script in STAGES:
        print(f"\n=== {script} ===", flush=True)
        completed = subprocess.run([PYTHON, str(ROOT / script)], cwd=ROOT)
        if completed.returncode != 0:
            print(f"FAILED {script} with {completed.returncode}; continuing", flush=True)


if __name__ == "__main__":
    main()
