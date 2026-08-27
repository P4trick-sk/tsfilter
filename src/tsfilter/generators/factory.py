from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from tsfilter.generators.base import BaseGenerator
from tsfilter.utils.sampling import sample_value

class RandomGeneratorFactory:
    def __init__(self, length: int, param_spaces: dict, seed: int | None = None):
        self.length = length
        self.rng = np.random.default_rng(seed)
        self.param_spaces = param_spaces

    def sample(self, generator_cls: type[BaseGenerator]) -> BaseGenerator:
        space = self.param_spaces.get(generator_cls, {})
        kwargs = {name: sample_value(self.rng, spec) for name, spec in space.items()}
        child_seed = int(self.rng.integers(0, 2**32 - 1))
        return generator_cls(length=self.length, seed=child_seed, **kwargs)