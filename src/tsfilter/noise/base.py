from __future__ import annotations

import numpy as np

from  tsfilter.generators.base import GeneratedSeries
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class NoisySeries:
    values: np.ndarray                       # clean + noise
    clean: GeneratedSeries
    params: dict = field(default_factory=dict)

class BaseNoise(ABC):
    def __init__(
            self,
            seed: int | None = None,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def _sample(self, length: int) -> np.ndarray:
        raise NotImplementedError

    def apply(self, series: GeneratedSeries) -> NoisySeries:
        noise = self._sample(len(series.values))
        return NoisySeries(
            values=series.values + noise,
            clean=series,
            params=self._describe(),
        )

    def _describe(self) -> dict:
        return {"seed": self.seed}