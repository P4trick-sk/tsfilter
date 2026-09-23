from __future__ import annotations

import numpy as np

from tsfilter.evaluation.base import BaseMetric

class MSEMetric(BaseMetric):
    name = "mse"
    higher_is_better = False

    def compute(self, clean: np.ndarray, filtered: np.ndarray) -> float:
        return float(np.mean((clean - filtered) ** 2))


class MAEMetric(BaseMetric):
    name = "mae"
    higher_is_better = False

    def compute(self, clean: np.ndarray, filtered: np.ndarray) -> float:
        return float(np.mean(np.abs(clean - filtered)))

# snr power signalu - power čiste
# power noise - power ciste - vyhladene
# cim vyssie tym lepsie
class SNRMetric(BaseMetric):
    """Signal-to-noise ratio, in decibels: 10*log10(signal power / error power).
    Higher = filtered output is closer to the clean signal."""
    name = "snr"
    higher_is_better = True

    def compute(self, clean: np.ndarray, filtered: np.ndarray) -> float:
        signal_power = np.mean(clean ** 2)
        error_power = np.mean((clean - filtered) ** 2)
        if error_power == 0:
            return float("inf")   # perfect reconstruction
        return float(10 * np.log10(signal_power / error_power))