from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from tsfilter.models.data import load_samples, make_dataset
from tsfilter.models.features import FEATURE_NAMES
from tsfilter.models.network import FilterPredictor


# ---- data -------------------------------------------------------------------

def sample_per_class(
        targets: np.ndarray,
        samples_per_class: int,
        rng: np.random.Generator
) -> np.ndarray:
    selected_idx = []

    for cls in np.unique(targets):
        members = rng.permutation(np.where(targets == cls)[0])
        selected_idx.extend(members[:samples_per_class]) # ak ma trieda menej vzoriek, zoberieme vsetky

    return np.array(selected_idx)


def split_train_validation(
        targets: np.ndarray,
        validation_fraction: float,
        rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    train_idx, validation_idx = [], []

    for cls in np.unique(targets):
        members = rng.permutation(np.where(targets == cls)[0])

        validation_count = max(1, round(len(members) * validation_fraction))

        validation_idx.extend(members[:validation_count])
        train_idx.extend(members[validation_count:])

    return np.array(train_idx), np.array(validation_idx)


def make_loaders(
        signals_dir: Path,
        labels_path: Path,
        validation_fraction: float,
        samples_per_class: int,
        batch_size: int,
        normalize_data: bool,
        rng: np.random.Generator
) -> tuple[DataLoader, DataLoader, list[str], np.ndarray, np.ndarray]:
    signals, features, labels = load_samples(signals_dir, labels_path)
    classes, targets = np.unique(labels, return_inverse=True)

    selected_idx = sample_per_class(targets, samples_per_class, rng)
    signals, features, targets = signals[selected_idx], features[selected_idx], targets[selected_idx]
    print(f"Samples per class: {dict(zip(classes, np.bincount(targets, minlength=len(classes))))}")

    train_idx, validation_idx = split_train_validation(targets, validation_fraction, rng)

    features_mean = features[train_idx].mean(axis=0)
    features_std = features[train_idx].std(axis=0) + 1e-8

    def loader_of(idx: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = make_dataset(
            signals[idx],
            features[idx],
            targets[idx],
            features_mean,
            features_std,
            normalize=normalize_data
        )

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    print(f"Loaded {len(targets)} samples (train {len(train_idx)}, validation {len(validation_idx)}), classes: {classes}")

    return loader_of(train_idx, True), loader_of(validation_idx, False), classes.tolist(), features_mean, features_std


def predict(
        model: FilterPredictor,
        loader: DataLoader
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    true, predicted = [], []

    with torch.no_grad():
        for signal, features, target in loader:
            true.append(target)
            predicted.append(model(signal, features).argmax(dim=1))

    return torch.cat(true).numpy(), torch.cat(predicted).numpy()

def validation_score(model: FilterPredictor, loader: DataLoader) -> float:
    true, predicted = predict(model, loader)

    return accuracy_score(true, predicted)


# -- trening --

def train_one_epoch(
        model: FilterPredictor,
        loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer
) -> None:
    model.train() # mód pre dropout a batchnorm

    for signal, features, target in loader: # nie iba jeden signál, ale batch
        optimizer.zero_grad() # vynulujeme gradient
        loss = loss_fn(model(signal, features), target)
        loss.backward()
        optimizer.step()

def train(
        model: FilterPredictor,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        loss_fn: nn.Module,
        learning_rate: float, epochs: int
) -> float:
    optimizer = torch.optim.Adam(
        model.parameters(), # ktore parametre optimalizujeme
        lr=learning_rate,
        weight_decay=1e-4
    )
    best_score, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, loss_fn, optimizer)
        score = validation_score(model, validation_loader)

        if score > best_score:
            best_score = score
            best_state = {name: value.clone() for name, value in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  validation accuracy {score:.3f}")

    model.load_state_dict(best_state)
    return best_score


def run_experiment(
        train_loader: DataLoader,
        validation_loader: DataLoader,
        classes: list[str],
        architecture: dict,
        learning_rate: float,
        epochs: int,
        seed: int
) -> tuple[FilterPredictor, float]:
    torch.manual_seed(seed)

    model = FilterPredictor(
        n_features = len(FEATURE_NAMES),
        n_classes = len(classes),
        **architecture
    )

    print(f"Architecture {architecture}, {sum(p.numel() for p in model.parameters())} learnable numbers")

    best_score = train(model, train_loader, validation_loader, nn.CrossEntropyLoss(), learning_rate, epochs)

    return model, best_score

# --- report a ulozenie

def print_report(model: FilterPredictor, validation_loader: DataLoader, classes: list[str], score: float) -> None:
    true, predicted = predict(model, validation_loader)
    matrix = confusion_matrix(true, predicted, labels=range(len(classes)))

    print(f"\nBest validation accuracy: {score:.3f}")
    print("Confusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(matrix, index=classes, columns=classes))


def save_model(model: FilterPredictor, classes: list[str], feature_mean: np.ndarray, feature_std: np.ndarray,
               normalize_data: bool, architecture: dict, path: Path) -> None:
    # normalize_data is stored so whoever loads the model prepares new signals the same way as during training
    torch.save({"state_dict": model.state_dict(), "classes": classes, "feature_mean": feature_mean,
                "feature_std": feature_std, "normalize_data": normalize_data, "n_features": len(FEATURE_NAMES),
                "architecture": architecture}, path)
    print(f"\nSaved model to {path}")


# ---- entry point ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the filter-class selection network.")
    parser.add_argument("--signals-dir", type=Path, required=True, help="Directory with combined per-sample CSVs")
    parser.add_argument("--labels", type=Path, required=True, help="Evaluator json export")
    parser.add_argument("--output", type=Path, default=Path("filter_selector.pt"))
    parser.add_argument("--epochs", type=int, default=40, help="How many passes over the training data")
    parser.add_argument("--batch-size", type=int, default=32, help="Samples the network sees before each adjustment")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate: size of each adjustment")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Share of samples held out for testing")
    parser.add_argument("--samples-per-class", type=int, default=250, help="Randomly pick at most this many samples of each class (smaller classes are kept whole)")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--features-only", action="store_true",help="Leave out the CNN, use only the hand-made features (baseline)")
    mode.add_argument("--signal-only", action="store_true", help="Use only the signal, not the hand-made features")

    parser.add_argument("--features-hidden-size", type=int, default=32, help="Number of hidden neurons in features branch, 0 if none")
    parser.add_argument("--channels", type=int, nargs="+", default=[16, 32, 64], help="Channels of each convolution layer; the number of values = number of layers, the last one = numbers handed to the classifier")
    parser.add_argument("--kernel-size", type=int, default=7, help="Width of each convolution filter (odd number)")
    parser.add_argument("--pool-size", type=int, default=4, help="Each convolution layer shortens the signal this many times")
    parser.add_argument("--combined-hidden-size", type=int, default=64, help="Number of hidden neurons in combined layer")
    parser.add_argument("--no-normalize-data", action="store_true", help="Feed signals and features to the network as they are, without normalization (baseline)")
    parser.add_argument("--no-normalize-batch", action="store_true", help="Leave out BatchNorm after each convolution layer (baseline)")
    parser.add_argument("--seed", type=int, default=0, help="Fixes the randomness so a run can be repeated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_loader, validation_loader, classes, feature_mean, feature_std = make_loaders(
        args.signals_dir,
        args.labels,
        args.validation_fraction,
        args.samples_per_class,
        args.batch_size,
        not args.no_normalize_data,
        np.random.default_rng(args.seed)
    )

    architecture = {
        "use_signal": not args.features_only,
        "use_features": not args.signal_only,
        "features_hidden_size": args.features_hidden_size,
        "channels": tuple(args.channels),
        "kernel_size": args.kernel_size,
        "pool_size": args.pool_size,
        "combined_hidden_size": args.combined_hidden_size,
        "batch_norm": not args.no_normalize_batch,
    }

    model, best_score = run_experiment(
        train_loader,
        validation_loader,
        classes,
        architecture,
        args.learning_rate,
        args.epochs,
        args.seed
    )

    print_report(model, validation_loader, classes, best_score)
    save_model(model, classes, feature_mean, feature_std, not args.no_normalize_data, architecture, args.output)


if __name__ == "__main__":
    main()
