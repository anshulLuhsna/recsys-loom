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
    ["scripts/run_overnight_longtail_weights.py"],
    ["scripts/freeze_best_system.py"],
    ["scripts/export_holdout_candidates.py"],
    ["scripts/run_overnight_holdout_eval.py"],
    ["scripts/export_demo_recommendations.py"],
    ["scripts/run_search_ranker.py", "--lexical-only"],
    ["scripts/write_morning_handoff.py"],
]


def main() -> None:
    for command in STAGES:
        print(f"\n=== {' '.join(command)} ===", flush=True)
        completed = subprocess.run(
            [PYTHON, str(ROOT / command[0]), *command[1:]],
            cwd=ROOT,
        )
        if completed.returncode != 0:
            raise SystemExit(f"{command[0]} failed with {completed.returncode}")


if __name__ == "__main__":
    main()
