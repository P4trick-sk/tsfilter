from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class ParamRange:
    low: float
    high: float

@dataclass
class RelativeParamRange:
    low: float
    high: float
    relative_to: str = "std"
    high_low: float | None = None
    high_high: float | None = None
    floor: float | None = None

def series_stats(series) -> dict:
    values = series.values
    return {
        "std": float(np.std(values)),
        "range": float(np.max(values) - np.min(values)),
        "length": len(values),
    }

def sample_value(rng: np.random.Generator, spec, stats: dict | None = None):
    if isinstance(spec, ParamRange):
        return rng.uniform(spec.low, spec.high)

    if isinstance(spec, RelativeParamRange):
        if stats is None:
            raise ValueError(f"{spec} needs series stats — pass target= to factory.sample()")
        base = stats[spec.relative_to]
        first = rng.uniform(spec.low, spec.high) * base
        if spec.floor is not None:
            first = max(first, spec.floor)
        if spec.high_low is not None:
            second = rng.uniform(spec.high_low, spec.high_high) * base
            if spec.floor is not None:
                second = max(second, spec.floor)
            return (first, second)
        return first

    return spec