from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def filter_class_of(winner: str) -> str:
    """'OWASmooth-Cut_3' -> 'OWASmooth-Cut', 'Exp_3_0.05' -> 'Exp'."""
    return winner.split("_")[0]


def load_labels(json_path: str | Path, level: str = "filter_class") -> dict[str, str]:
    with open(json_path) as file:
        records = json.load(file)

    convert = filter_class_of if level == "filter_class" else (lambda w: w)

    return { record["datasample_name"]: convert(record["winner"]) for record in records }


def get_distribution(labels: dict[str, str]) -> Counter:
    return Counter(labels.values())
