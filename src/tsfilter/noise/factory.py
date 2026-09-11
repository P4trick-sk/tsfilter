from __future__ import annotations

import numpy as np
from .base import BaseNoise
from tsfilter.utils.sampling import sample_value, series_stats
from ..generators.base import GeneratedSeries


class RandomNoiseFactory:
    def __init__(self, param_spaces: dict, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.param_spaces = param_spaces

    def sample(self, noise_cls: type[BaseNoise], target: GeneratedSeries | None = None) -> BaseNoise:
        stats = series_stats(target) if target is not None else None
        space = self.param_spaces.get(noise_cls, {})
        kwargs = {name: sample_value(self.rng, spec, stats) for name, spec in space.items()}
        child_seed = int(self.rng.integers(0, 2 ** 32 - 1))
        return noise_cls(seed=child_seed, **kwargs)