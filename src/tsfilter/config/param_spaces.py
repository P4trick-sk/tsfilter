from __future__ import annotations

from pathlib import Path
import yaml

from tsfilter.generators.composite import MultiGeneratorChoice, MixedGeneratorChoice
from tsfilter.generators.series import (
    TrendGenerator, SinusGenerator, RandomWalkGenerator, ConstantGenerator, ExponentialGenerator,
)
from tsfilter.noise.composite import MultiNoiseChoice, MixedNoiseChoice
from tsfilter.noise.noise import GaussianNoise, ImpulsiveNoise, PinkNoise, DriftNoise
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
            floor=spec.get("floor"),
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

def _load_selection(path: str | Path, registry: dict, multi_cls, mixed_cls) -> list:
    with open(path) as f:
        raw = yaml.safe_load(f)

    selection: list = []

    for name in raw.get("allowed", []):
        if name not in registry:
            raise ValueError(f"Unknown '{name}' in allowed list — check the registry")
        selection.append(registry[name])

    for entry in raw.get("multi", []):
        if entry["base"] not in registry:
            raise ValueError(f"Unknown base '{entry['base']}' in multi entry")
        selection.append(multi_cls(
            base=registry[entry["base"]],
            n_components=ParamRange(entry["n_components"]["low"], entry["n_components"]["high"]),
        ))

    for entry in raw.get("mixed", []):
        unknown = [n for n in entry["candidates"] if n not in registry]
        if unknown:
            raise ValueError(f"Unknown candidate(s) {unknown} in mixed entry")
        selection.append(mixed_cls(
            candidates=[registry[n] for n in entry["candidates"]],
            n_components=ParamRange(entry["n_components"]["low"], entry["n_components"]["high"]),
        ))

    if not selection:
        raise ValueError(f"{path} produced an empty selection — nothing to sample from")

    return selection


def load_generator_selection(path: str | Path) -> list:
    return _load_selection(path, GENERATOR_REGISTRY, MultiGeneratorChoice, MixedGeneratorChoice)


def load_noise_selection(path: str | Path) -> list:
    return _load_selection(path, NOISE_REGISTRY, MultiNoiseChoice, MixedNoiseChoice)