from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from tsfilter.dataset.sample import DatasetSample
from tsfilter.generators.base import GeneratedSeries
from tsfilter.noise.base import NoisySeries


# ---- id assignment -----------------------------------------------------

def _batch_timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _make_sample_id(batch_timestamp: str, index: int) -> str:
    return f"{batch_timestamp}_{index:05d}"


def _ensure_sample_ids(samples: Sequence[DatasetSample]) -> list[DatasetSample]:
    batch_ts = _batch_timestamp()
    result = []
    for index, sample in enumerate(samples):
        if sample.sample_id is None:
            sample = replace(sample, sample_id=_make_sample_id(batch_ts, index))
        result.append(sample)
    return result


# ---- csv -----------------------------------------------------------------

def to_csv(samples: Sequence[DatasetSample], output_dir: str | Path) -> None:
    samples = _ensure_sample_ids(samples)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        with open(f"{output_dir}/{sample.sample_id}.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerow(["t", "clean", "noisy"])
            clean_vals = sample.clean.values
            noisy_vals = sample.noisy.values

            for t in range(len(clean_vals)):
                writer.writerow([t, clean_vals[t], noisy_vals[t]])


# ---- json (de)serialization helpers --------------------------------------

def _generated_series_to_dict(series: GeneratedSeries) -> dict:
    return {"values": series.values.tolist(), "params": series.params}


def _generated_series_from_dict(data: dict) -> GeneratedSeries:
    return GeneratedSeries(values=np.array(data["values"]), params=data["params"])


def _noisy_series_to_dict(series: NoisySeries) -> dict:
    return {
        "values": series.values.tolist(),
        "clean": _generated_series_to_dict(series.clean),
        "params": series.params,
    }


def _noisy_series_from_dict(data: dict) -> NoisySeries:
    return NoisySeries(
        values=np.array(data["values"]),
        clean=_generated_series_from_dict(data["clean"]),
        params=data["params"],
    )


def _sample_to_dict(sample: DatasetSample) -> dict:
    return {
        "sample_id": sample.sample_id,
        "clean": _generated_series_to_dict(sample.clean),
        "noisy": _noisy_series_to_dict(sample.noisy),
        "label": sample.label,
    }


def _sample_from_dict(data: dict) -> DatasetSample:
    return DatasetSample(
        clean=_generated_series_from_dict(data["clean"]),
        noisy=_noisy_series_from_dict(data["noisy"]),
        label=data.get("label"),
        sample_id=data.get("sample_id"),
    )


# ---- json ------------------------------------------------------------------

def to_json(samples: Sequence[DatasetSample], path: str | Path) -> None:
    samples = _ensure_sample_ids(samples)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [_sample_to_dict(sample) for sample in samples]
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


def from_json(path: str | Path) -> list[DatasetSample]:
    with open(path) as file:
        data = json.load(file)
    return [_sample_from_dict(d) for d in data]