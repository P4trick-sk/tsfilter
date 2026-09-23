from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

class BaseMetric(ABC):
    name: str
    higher_is_better: bool

    @abstractmethod
    def compute(self, clean: np.ndarray, filtered: np.ndarray) -> float:
        raise NotImplementedError