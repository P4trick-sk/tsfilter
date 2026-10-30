from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from tsfilter.evaluation.labels import load_labels
from tsfilter.models.features import extract_features


def load_samples(signals_dir: str | Path, labels_json: str | Path):
    labels = load_labels(labels_json, level="filter_class")

    signals, features, filter_classes = [], [], []
    skipped = []

    for sample_id, filter_class in sorted(labels.items()):
        csv_path = Path(signals_dir) / f"{sample_id}.csv"

        if not csv_path.exists():
            continue

        noisy = pd.read_csv(csv_path)["noisy"].to_numpy(dtype=np.float32)
        sample_features = extract_features(noisy)

        # nan alebo inf by pokazili trening (loss = NaN)
        if not np.isfinite(noisy).all() or not np.isfinite(sample_features).all():
            skipped.append(sample_id)
            continue

        signals.append(noisy)
        features.append(sample_features)
        filter_classes.append(filter_class)

    if skipped:
        print(f"Skipped {len(skipped)} samples with nan or inf values")

    return np.stack(signals), np.stack(features), filter_classes


def make_dataset(signals, features, targets, feature_mean, feature_std, normalize = True) -> TensorDataset:
    if normalize:
        signals = (signals - signals.mean(axis=1, keepdims=True)) / (signals.std(axis=1, keepdims=True) + 1e-8)
        features = (features - feature_mean) / feature_std

    return TensorDataset(
        torch.tensor(signals),
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.long),
    )
