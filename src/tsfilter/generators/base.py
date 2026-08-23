from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

@dataclass
class GeneratedSeries:
    values: np.ndarray
    params: dict = field(default_factory=dict)

class BaseGenerator(ABC):
    def __init__(self, length: int, seed: int | None = None):
        self.length = length
        self.seed = seed
        self.rng = np.random.default_rng(seed) # we save it in order to continue in the stream for the set seed upon multiple generations

    @abstractmethod
    def generate(self) -> GeneratedSeries:
        raise NotImplementedError