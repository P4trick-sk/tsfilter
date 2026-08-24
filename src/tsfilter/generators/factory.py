from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from tsfilter.generators.base import BaseGenerator

@dataclass
class ParamRange:
    low: float
    high: float

def _sample(rng: np.random.Generator, spec):
    if isinstance(spec, ParamRange):
        return rng.uniform(spec.low, spec.high)
    return spec


class RandomGeneratorFactory:
    def __init__(
            self,
            length: int,
            param_spaces: dict,
            seed: int | None = None):
        self.length = length
        self.rng = np.random.default_rng(seed)
        self.param_spaces = param_spaces

    def sample(self, generator_cls: type[BaseGenerator]) -> BaseGenerator:
        space = self.param_spaces.get(generator_cls, {})
        kwargs = {name: _sample(self.rng, spec) for name, spec in space.items()} # random value for each arg
        child_seed = int(self.rng.integers(0, 2**32 - 1))
        return generator_cls(length=self.length, seed=child_seed, **kwargs)