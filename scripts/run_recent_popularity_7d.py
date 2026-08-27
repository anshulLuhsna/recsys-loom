#!/usr/bin/env python3
"""Run the seven-day recent-popularity experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.popularity import run_popularity_baseline  # noqa: E402


def main() -> None:
    result = run_popularity_baseline(
        transactions_path=ROOT / "transactions_train.csv",
        articles_path=ROOT / "articles.csv",
        output_directory=ROOT / "artifacts" / "recent_popularity_7d",
        training_start="2020-09-09",
        train_end="2020-09-15",
        validation_start="2020-09-16",
        validation_end="2020-09-22",
        k=12,
        threads=2,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
