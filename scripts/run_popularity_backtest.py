#!/usr/bin/env python3
"""Compare all-history and seven-day popularity across eight weekly folds."""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.backtest import build_weekly_windows  # noqa: E402
from recsys_loom.popularity import run_popularity_baseline  # noqa: E402


OUTPUT_DIRECTORY = ROOT / "artifacts" / "popularity_backtest_8w"
METRIC_NAMES = (
    "map_at_12",
    "recall_at_12",
    "micro_recall_at_12",
    "hit_rate_at_12",
)


def summarize(folds: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric_name in METRIC_NAMES:
        all_history = [fold["all_history"]["aggregate"][metric_name] for fold in folds]
        recent = [fold["recent_7d"]["aggregate"][metric_name] for fold in folds]
        deltas = [new - old for old, new in zip(all_history, recent, strict=True)]
        summary[metric_name] = {
            "all_history_mean": statistics.mean(all_history),
            "all_history_median": statistics.median(all_history),
            "recent_7d_mean": statistics.mean(recent),
            "recent_7d_median": statistics.median(recent),
            "mean_absolute_delta": statistics.mean(deltas),
            "recent_7d_wins": sum(new > old for old, new in zip(all_history, recent, strict=True)),
            "all_history_wins": sum(old > new for old, new in zip(all_history, recent, strict=True)),
            "ties": sum(old == new for old, new in zip(all_history, recent, strict=True)),
        }
    return summary


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    windows = build_weekly_windows(final_validation_end=date(2020, 9, 22), weeks=8)
    folds: list[dict[str, Any]] = []
    started = time.monotonic()

    for index, window in enumerate(windows, start=1):
        fold_id = f"{window['validation_start']}_{window['validation_end']}"
        print(f"[{index}/8] {fold_id}: all-history", flush=True)
        all_history = run_popularity_baseline(
            transactions_path=ROOT / "transactions_train.csv",
            articles_path=ROOT / "articles.csv",
            output_directory=OUTPUT_DIRECTORY / fold_id / "all_history",
            train_end=window["train_end"],
            validation_start=window["validation_start"],
            validation_end=window["validation_end"],
            k=12,
            threads=2,
            write_predictions=False,
        )

        print(f"[{index}/8] {fold_id}: recent-7d", flush=True)
        recent = run_popularity_baseline(
            transactions_path=ROOT / "transactions_train.csv",
            articles_path=ROOT / "articles.csv",
            output_directory=OUTPUT_DIRECTORY / fold_id / "recent_7d",
            training_start=window["recent_start"],
            train_end=window["train_end"],
            validation_start=window["validation_start"],
            validation_end=window["validation_end"],
            k=12,
            threads=2,
            write_predictions=False,
        )

        folds.append(
            {
                "fold": fold_id,
                **window,
                "all_history": all_history,
                "recent_7d": recent,
            }
        )
        partial_result = {
            "contract": {
                "folds": 8,
                "fold_length_days": 7,
                "recent_lookback_days": 7,
                "models": ["all_history", "recent_7d"],
                "prediction_count": 12,
                "validation_labels": "unique customer_id and article_id pairs",
            },
            "completed_folds": len(folds),
            "folds": folds,
        }
        (OUTPUT_DIRECTORY / "partial_results.json").write_text(
            json.dumps(partial_result, indent=2), encoding="utf-8"
        )

    result = {
        "contract": {
            "folds": 8,
            "fold_length_days": 7,
            "recent_lookback_days": 7,
            "models": ["all_history", "recent_7d"],
            "prediction_count": 12,
            "validation_labels": "unique customer_id and article_id pairs",
        },
        "runtime_seconds": time.monotonic() - started,
        "summary": summarize(folds),
        "folds": folds,
    }
    (OUTPUT_DIRECTORY / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"Completed in {result['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
