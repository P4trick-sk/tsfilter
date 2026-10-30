from __future__ import annotations

import argparse
from pathlib import Path

import itertools
import pandas as pd
import numpy as np


from tsfilter.scripts.train_model import make_loaders, run_experiment

GRID = {
    "channels":             [(16, 32), (16, 32, 64), (32, 64, 128)],
    "kernel_size":          [5, 7, 11],
    "pool_size":            [2, 4],
    "features_hidden_size": [0, 16, 32],
    "combined_hidden_size": [32, 64],
    "learning_rate":        [1e-3, 3e-4],
    # "mode":                 ["signal_only", "features_only", "both"],
    "batch_norm":           [True, False],
}
SEEDS = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grid search for NN parameters.")
    parser.add_argument("--signals-dir", type=Path, required=True, help="Directory with combined per-sample CSVs")
    parser.add_argument("--labels", type=Path, required=True, help="Evaluator json export")
    parser.add_argument("--epochs", type=int, default=40, help="How many passes over the training data")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Share of samples held out for testing")
    parser.add_argument("--samples-per-class", type=int, default=250, help="Randomly pick at most this many samples of each class (smaller classes are kept whole)")
    parser.add_argument("--batch-size", type=int, default=32, help="Samples the network sees before each adjustment")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the results.")
    parser.add_argument("--seed", type=int, default=0, help="Fixes the randomness so a run can be repeated")

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    results = []

    for normalize in [True, False]:
        train_loader, validation_loader, classes, feature_mean, feature_std = make_loaders(
            args.signals_dir,
            args.labels,
            args.validation_fraction,
            args.samples_per_class,
            args.batch_size,
            normalize,
            np.random.default_rng(args.seed)
        )

        for combo in itertools.product(*GRID.values()):
            config = dict(zip(GRID, combo))

            architecture = {
                "use_signal": config.get("mode", "both") in ["signal_only", "both"],
                "use_features": config.get("mode", "both") in ["features_only", "both"],
                "features_hidden_size": config["features_hidden_size"],
                "channels": config["channels"],
                "kernel_size": config["kernel_size"],
                "pool_size": config["pool_size"],
                "combined_hidden_size": config["combined_hidden_size"],
                "batch_norm": config["batch_norm"],
            }

            scores = [run_experiment(
                train_loader,
                validation_loader,
                classes,
                architecture,
                config["learning_rate"],
                args.epochs,
                seed = s
            )[1] for s in SEEDS]

            results.append({"normalize_data": normalize, **config, "mean": np.mean(scores), "std": np.std(scores)})
            pd.DataFrame(results).to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
