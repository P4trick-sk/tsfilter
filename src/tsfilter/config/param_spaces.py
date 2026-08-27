from __future__ import annotations

from pathlib import Path
import yaml

from tsfilter.generators.series import (
    TrendGenerator, SinusGenerator, RandomWalkGenerator, ConstantGenerator,
)
from tsfilter.noise.noise import GaussianNoise, ImpulsiveNoise
from tsfilter.utils.sampling import ParamRange

GENERATOR_REGISTRY = {
    "TrendGenerator": TrendGenerator,
    "SinusGenerator": SinusGenerator,
    "RandomWalkGenerator": RandomWalkGenerator,
    "ConstantGenerator": ConstantGenerator,
}

NOISE_REGISTRY = {
    "GaussianNoise": GaussianNoise,
    "ImpulsiveNoise": ImpulsiveNoise,
}

def load_param_spaces(path: str | Path, registry: dict) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)

    param_spaces = {}
    for name, params in raw.items():
        if name not in registry:
            raise ValueError(f"Unknown class '{name}' in config — check the registry")
        cls = registry[name]
        param_spaces[cls] = {
            k: ParamRange(v["low"], v["high"]) if isinstance(v, dict) else v
            for k, v in params.items()
        }
    return param_spaces