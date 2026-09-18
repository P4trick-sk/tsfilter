from __future__ import annotations

import numpy as np

from tsfilter.noise.base import BaseNoise

class GaussianNoise(BaseNoise):
    def __init__(
            self,
            mean=0.0,
            sigma=0.01,
            seed: int | None = None
    ):
        super().__init__(seed)
        self.mean = mean
        self.sigma = sigma

    def _sample(self, length: int) -> np.ndarray:
        return self.rng.normal(self.mean, self.sigma, size=length)

    def _describe(self) -> dict:
        return {
            "noise": "gaussian",
            "mean": self.mean,
            "sigma": self.sigma,
            "seed": self.seed}

class ImpulsiveNoise(BaseNoise):
    def __init__(
        self,
        rate: float = 0.05,
        amplitude_range: tuple[float, float] = (1.0, 3.0),
        seed: int | None = None,
    ):
        super().__init__(seed)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        self.rate = rate
        self.amplitude_range = amplitude_range

    def _sample(self, length: int) -> np.ndarray:
        noise = np.zeros(length)
        hit_mask = self.rng.random(length) < self.rate           # which points get hit
        magnitudes = self.rng.uniform(*self.amplitude_range, size=length)
        signs = self.rng.choice([-1.0, 1.0], size=length)
        noise[hit_mask] = (signs * magnitudes)[hit_mask]
        return noise

    def _describe(self) -> dict:
        return {
            "noise": "impulsive",
            "rate": self.rate,
            "amplitude_range": self.amplitude_range,
            "seed": self.seed,
        }

class PinkNoise(BaseNoise):
    def __init__(self, sigma: float = 1.0, seed: int | None = None):
        super().__init__(seed)
        self.sigma = sigma

    def _sample(self, length: int) -> np.ndarray:
        white = self.rng.normal(size=length)
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(length)

        scale = np.ones_like(freqs)
        scale[1:] = 1.0 / np.sqrt(freqs[1:])
        scale[0] = 0.0

        pink = np.fft.irfft(spectrum * scale, n=length)
        pink = pink / np.std(pink) * self.sigma
        return pink

    def _describe(self) -> dict:
        return {"noise": "pink", "sigma": self.sigma, "seed": self.seed}

class DriftNoise(BaseNoise):
    def __init__(self, total_drift: float, seed: int | None = None):
        super().__init__(seed)
        self.total_drift = total_drift

    def _sample(self, length: int) -> np.ndarray:
        if length <= 1:
            return np.zeros(length)
        rate = self.total_drift / (length - 1)
        return np.array(rate * np.arange(length))

    def _describe(self) -> dict:
        return {"noise": "drift", "total_drift": self.total_drift, "seed": self.seed}