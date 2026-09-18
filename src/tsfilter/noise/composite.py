from __future__ import annotations

from typing import Sequence
from dataclasses import dataclass
from tsfilter.noise.base import BaseNoise
from tsfilter.utils.sampling import ParamRange

import numpy as np

@dataclass
class MultiNoiseChoice:
    base: type[BaseNoise]
    n_components: ParamRange

@dataclass
class MixedNoiseChoice:
    candidates: list[type[BaseNoise]]
    n_components: ParamRange

class CompositeNoise(BaseNoise):
    def __init__(self, noises: Sequence[BaseNoise]):
        super().__init__(seed=None)
        self.noises = noises

    def _sample(self, length: int) -> np.ndarray:
        return np.sum([n._sample(length) for n in self.noises], axis=0)

    def _describe(self) -> dict:
        return {"noise": "composite", "components": [n._describe() for n in self.noises]}