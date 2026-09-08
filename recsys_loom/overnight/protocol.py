"""Evaluation-integrity protocol for the overnight ranking program.

The H&M transaction file ends on 2020-09-22. Every previously reported
"final week" metric used 2020-09-16 through 2020-09-22, so that calendar
window is not a pristine temporal holdout.

Development and model selection use earlier weekly snapshots. The official
overnight final number is a customer-holdout evaluation on validation-week
buyers whose 2020-09-16..22 labels were not used for selection.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
OVERNIGHT_DIR = ARTIFACTS / "overnight"
CACHE_DIR = OVERNIGHT_DIR / "cache"
TMP_DIR = ARTIFACTS / "tmp" / "duckdb"

SNAPSHOTS = [
    {
        "cutoff": "2020-08-17",
        "target_start": "2020-08-18",
        "target_end": "2020-08-24",
    },
    {
        "cutoff": "2020-08-24",
        "target_start": "2020-08-25",
        "target_end": "2020-08-31",
    },
    {
        "cutoff": "2020-08-31",
        "target_start": "2020-09-01",
        "target_end": "2020-09-07",
    },
    {
        "cutoff": "2020-09-07",
        "target_start": "2020-09-08",
        "target_end": "2020-09-14",
    },
    {
        "cutoff": "2020-09-15",
        "target_start": "2020-09-16",
        "target_end": "2020-09-22",
    },
]

# Already used to select two-tower K=50. Still the cleanest development
# windows available; do not treat them as unseen final test.
# Extra earlier weeks used only to scale ranking supervision.
# They are never selection or final-evaluation windows.
EXTRA_TRAINING_SNAPSHOTS = [
    {
        "cutoff": "2020-08-03",
        "target_start": "2020-08-04",
        "target_end": "2020-08-10",
    },
    {
        "cutoff": "2020-08-10",
        "target_start": "2020-08-11",
        "target_end": "2020-08-17",
    },
]

SELECTION_FOLDS = [2, 3]
CONTAMINATED_FINAL_FOLD = 4
DEV_CUSTOMERS = 2000


def fold_offset(include_extra_training: bool) -> int:
    return len(EXTRA_TRAINING_SNAPSHOTS) if include_extra_training else 0


def selection_fold_indices(include_extra_training: bool = False) -> list[int]:
    offset = fold_offset(include_extra_training)
    return [offset + index for index in SELECTION_FOLDS]


def contaminated_fold_index(include_extra_training: bool = False) -> int:
    return fold_offset(include_extra_training) + CONTAMINATED_FINAL_FOLD


TT_BUDGET = 50
POPULARITY_BUDGET = 100
EXISTING_SOURCE_BUDGET = 500
MAP_TOLERANCE = 0.0002

BASELINE_SOURCES = [
    "recent_7d_pop",
    "repeat_purchase",
    "cooccurrence",
    "als",
    "content",
    "two_tower",
]

BASELINE_RANKER = {
    "family": "lightgbm_lambdarank",
    "objective": "lambdarank",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42,
    "two_tower_k": TT_BUDGET,
    "sources": BASELINE_SOURCES,
}

PROTOCOL = {
    "name": "customer_holdout_final_evaluation",
    "dataset_end": "2020-09-22",
    "untouched_temporal_window": None,
    "development_folds": [
        SNAPSHOTS[index] for index in SELECTION_FOLDS
    ],
    "contaminated_calendar_window": {
        "cutoff": "2020-09-15",
        "target_start": "2020-09-16",
        "target_end": "2020-09-22",
        "reason": (
            "Metrics for this week were inspected across ranking, "
            "two-tower, image, text, and DCN experiments."
        ),
    },
    "final_evaluation": (
        "Disjoint validation-week customers whose 2020-09-16..22 labels "
        "were not used for architecture or hyperparameter selection."
    ),
    "selection_rule": (
        "Choose the simplest configuration whose mean selection-fold "
        f"MAP@12 is within {MAP_TOLERANCE} of the best mean."
    ),
}


def ensure_directories() -> None:
    OVERNIGHT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
