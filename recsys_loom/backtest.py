"""Chronological backtesting utilities for popularity experiments."""

from __future__ import annotations

from datetime import date, timedelta


def build_weekly_windows(
    final_validation_end: date,
    weeks: int,
) -> list[dict[str, str]]:
    """Return oldest-to-newest seven-day folds ending at ``final_validation_end``."""
    if weeks <= 0:
        raise ValueError("weeks must be positive")

    windows: list[dict[str, str]] = []
    for offset in reversed(range(weeks)):
        validation_end = final_validation_end - timedelta(days=7 * offset)
        validation_start = validation_end - timedelta(days=6)
        train_end = validation_start - timedelta(days=1)
        recent_start = train_end - timedelta(days=6)
        windows.append(
            {
                "recent_start": recent_start.isoformat(),
                "train_end": train_end.isoformat(),
                "validation_start": validation_start.isoformat(),
                "validation_end": validation_end.isoformat(),
            }
        )
    return windows
