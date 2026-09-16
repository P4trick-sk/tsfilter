from __future__ import annotations

from tsfilter.noise.base import BaseNoise

import numpy as np

class CompositeNoise(BaseNoise):
    def __init__(self, noises: Sequence[BaseNoise]):
        super().__init__(seed=None)
        self.noises = noises

    def _sample(self, length: int) -> np.ndarray:
        return np.sum([n._sample(length) for n in self.noises], axis=0)

    def _describe(self) -> dict:
        return {"noise": "composite", "components": [n._describe() for n in self.noises]}