from __future__ import annotations

from tsfilter.generators.base import BaseGenerator, GeneratedSeries
import numpy as np

class ConstantGenerator(BaseGenerator):
    def __init__(
            self,
            length: int,
            value: float,
            seed: int | None = None
    ):
        super().__init__(length, seed)
        self.value = value

    def generate(self) -> GeneratedSeries:
        values = np.full(self.length, self.value)
        return GeneratedSeries(
            values=values,
            params={"value": self.value}
        )

class TrendGenerator(BaseGenerator):
    def __init__(
            self,
            length: int,
            slope: float,
            intercept: float= 0.0,
            seed: int | None = None
    ):
        super().__init__(length, seed)
        self.slope = slope
        self.intercept = intercept

    def generate(self) -> GeneratedSeries:
        x = np.arange(self.length)
        values = np.array(self.slope * x + self.intercept)
        return GeneratedSeries(
            values=values,
            params={"slope": self.slope, "intercept": self.intercept}
        )

class RandomWalkGenerator(BaseGenerator):
    def __init__(
            self,
            length: int,
            step_std: float = 0.0,
            start: float = 0.0,
            seed: int | None = None
    ):
        super().__init__(length, seed)
        self.step_std = step_std
        self.start = start

    def generate(self) -> GeneratedSeries:
        steps = self.rng.normal(loc = 0, scale=self.step_std, size=self.length)
        values = np.array(self.start + np.cumsum(steps))
        return GeneratedSeries(
            values=values,
            params={
                "generator": "random_walk",
                "length": self.length,
                "step_std": self.step_std,
                "start": self.start,
                "seed": self.seed,
            },
        )

class SinusGenerator(BaseGenerator):
    def __init__(
        self,
        length: int,
        amplitude: float,
        period: float,
        phase: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__(length, seed)
        self.amplitude = amplitude
        self.period = period
        self.phase = phase

    def generate(self) -> GeneratedSeries:
        t = np.arange(self.length)
        values = self.amplitude * np.sin(2 * np.pi * t / self.period + self.phase)
        return GeneratedSeries(
            values=values,
            params={
                "generator": "sinus",
                "length": self.length,
                "amplitude": self.amplitude,
                "period": self.period,
                "phase": self.phase,
                "seed": self.seed,
            },
        )