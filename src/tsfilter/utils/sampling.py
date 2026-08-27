from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class ParamRange:
    low: float
    high: float

def sample_value(rng: np.random.Generator, spec):
    if isinstance(spec, ParamRange):
        return rng.uniform(spec.low, spec.high)
    return spec