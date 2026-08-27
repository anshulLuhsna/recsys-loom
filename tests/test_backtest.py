import unittest
from datetime import date

from recsys_loom.backtest import build_weekly_windows


class WeeklyWindowTests(unittest.TestCase):
    def test_builds_nonoverlapping_chronological_windows_without_leakage(self) -> None:
        self.assertEqual(
            build_weekly_windows(final_validation_end=date(2020, 9, 22), weeks=3),
            [
                {
                    "recent_start": "2020-08-26",
                    "train_end": "2020-09-01",
                    "validation_start": "2020-09-02",
                    "validation_end": "2020-09-08",
                },
                {
                    "recent_start": "2020-09-02",
                    "train_end": "2020-09-08",
                    "validation_start": "2020-09-09",
                    "validation_end": "2020-09-15",
                },
                {
                    "recent_start": "2020-09-09",
                    "train_end": "2020-09-15",
                    "validation_start": "2020-09-16",
                    "validation_end": "2020-09-22",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
