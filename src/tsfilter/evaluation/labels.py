from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def family_of(winner: str) -> str:
    """'OWASmooth-Cut_3' -> 'OWASmooth-Cut', 'Exp_3_0.05' -> 'Exp'."""
    return winner.split("_")[0]


def load_labels(json_path: str | Path, level: str = "family") -> dict[str, str]:
    """sample_id -> label, level='family' keeps only the method, level='winner' keeps method + parameters."""
    with open(json_path) as file:
        records = json.load(file)
    convert = family_of if level == "family" else (lambda w: w)
    return {record["datasample_name"]: convert(record["winner"]) for record in records}


def get_distribution(labels: dict[str, str]) -> Counter:
    return Counter(labels.values())
