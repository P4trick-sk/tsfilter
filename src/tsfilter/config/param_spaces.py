from __future__ import annotations

from pathlib import Path
import yaml

from tsfilter.generators.series import (
    TrendGenerator, SinusGenerator, RandomWalkGenerator, ConstantGenerator,
)
from tsfilter.generators.factory import ParamRange

GENERATOR_REGISTRY = {
    "TrendGenerator": TrendGenerator,
    "SinusGenerator": SinusGenerator,
    "RandomWalkGenerator": RandomWalkGenerator,
    "ConstantGenerator": ConstantGenerator,
}

def load_param_spaces(path: str | Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)

    param_spaces: dict = {}
    for gen_name, params in raw.items():
        if gen_name not in GENERATOR_REGISTRY:
            raise ValueError(f"Unknown generator '{gen_name}' in config — check GENERATOR_REGISTRY")
        gen_cls = GENERATOR_REGISTRY[gen_name]

        param_spaces[gen_cls] = {
            name: ParamRange(spec["low"], spec["high"]) if isinstance(spec, dict) else spec
            for name, spec in params.items()
        }
    return param_spaces