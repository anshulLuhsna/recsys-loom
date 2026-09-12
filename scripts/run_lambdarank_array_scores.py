#!/usr/bin/env python3
"""Train LambdaRank from numeric arrays in an OpenMP-isolated process."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys_loom.ranking.ranker import train_ranker


def main() -> None:
    input_directory = Path(sys.argv[1])
    output_directory = Path(sys.argv[2])
    metadata = json.loads(
        (input_directory / "metadata.json").read_text(encoding="utf-8")
    )
    train_features = np.load(
        input_directory / "train_features.npy",
        mmap_mode="r",
    )
    validation_features = np.load(
        input_directory / "validation_features.npy",
        mmap_mode="r",
    )
    model = train_ranker(
        train_features,
        np.load(input_directory / "train_labels.npy"),
        np.load(input_directory / "train_groups.npy"),
        metadata["feature_names"],
        validation_features,
        np.load(input_directory / "validation_labels.npy"),
        np.load(input_directory / "validation_groups.npy"),
        verbose=25,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    np.save(
        output_directory / "train_scores.npy",
        model.predict(train_features).astype(np.float32),
    )
    np.save(
        output_directory / "validation_scores.npy",
        model.predict(validation_features).astype(np.float32),
    )


if __name__ == "__main__":
    main()
