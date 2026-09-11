from __future__ import annotations

from pathlib import Path
import yaml

from tsfilter.generators.series import (
    TrendGenerator, SinusGenerator, RandomWalkGenerator, ConstantGenerator, ExponentialGenerator,
)
from tsfilter.noise.noise import GaussianNoise, ImpulsiveNoise, PinkNoise, DriftNoise, OutlierNoise
from tsfilter.utils.sampling import ParamRange, RelativeParamRange

GENERATOR_REGISTRY = {
    "TrendGenerator": TrendGenerator,
    "SinusGenerator": SinusGenerator,
    "RandomWalkGenerator": RandomWalkGenerator,
    "ExponentialGenerator": ExponentialGenerator,
    "ConstantGenerator": ConstantGenerator,
}

NOISE_REGISTRY = {
    "GaussianNoise": GaussianNoise,
    "ImpulsiveNoise": ImpulsiveNoise,
    "PinkNoise": PinkNoise,
    "DriftNoise": DriftNoise,
    "OutlierNoise": OutlierNoise,
}

def _parse_spec(spec):
    if not isinstance(spec, dict):
        return spec  # fixed value, no sampling
    if "relative_to" in spec:
        return RelativeParamRange(
            low=spec["low"],
            high=spec["high"],
            relative_to=spec["relative_to"],
            high_low=spec.get("high_low"),
            high_high=spec.get("high_high"),
        )
    return ParamRange(spec["low"], spec["high"])

def load_param_spaces(path: str | Path, registry: dict) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)

    param_spaces = {}
    for name, params in raw.items():
        if name not in registry:
            raise ValueError(f"Unknown class '{name}' in config — check the registry")
        cls = registry[name]
        param_spaces[cls] = {p: _parse_spec(s) for p, s in params.items()}
    return param_spaces