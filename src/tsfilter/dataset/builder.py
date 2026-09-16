from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from tsfilter.generators.base import BaseGenerator, GeneratedSeries
from tsfilter.generators.composite import CompositeGenerator
from tsfilter.generators.factory import RandomGeneratorFactory
from tsfilter.noise.base import BaseNoise, NoisySeries
from tsfilter.noise.composite import CompositeNoise
from tsfilter.noise.factory import RandomNoiseFactory
from tsfilter.dataset.sample import DatasetSample

Labeler = Callable[[GeneratedSeries, NoisySeries], dict]

@dataclass
class CompositeConfig:
    n_generators_low: int
    n_generators_high: int
    n_noises_low: int
    n_noises_high: int

class DatasetBuilder:
    def __init__(
        self,
        generator_factory: RandomGeneratorFactory,
        noise_factory: RandomNoiseFactory,
        generator_classes: Sequence[type[BaseGenerator]],
        noise_classes: Sequence[type[BaseNoise]],
        composite_config: CompositeConfig,
        labeler: Labeler | None = None,
        seed: int | None = None,
    ):
        self.generator_factory = generator_factory
        self.noise_factory = noise_factory
        self.generator_classes = generator_classes
        self.noise_classes = noise_classes
        self.labeler = labeler
        self.composite_config = composite_config
        self.rng = np.random.default_rng(seed)

    def _sample_generator(self) -> BaseGenerator:
        n_generators = self.rng.integers(
            self.composite_config.n_generators_low,
            self.composite_config.n_generators_high + 1,
        )

        if n_generators == 1:
            generator_cls = self.generator_classes[self.rng.integers(len(self.generator_classes))]
            return self.generator_factory.sample(generator_cls)

        indices = self.rng.integers(0, len(self.generator_classes), size=n_generators)
        generators = [self.generator_factory.sample(self.generator_classes[int(index)]) for index in indices]
        return CompositeGenerator(generators)

    def _sample_noise(self, target: GeneratedSeries) -> BaseNoise:
        n_noises = self.rng.integers(
            self.composite_config.n_noises_low,
            self.composite_config.n_noises_high + 1,
        )

        if n_noises == 1:
            noise_cls = self.noise_classes[self.rng.integers(len(self.noise_classes))]
            return self.noise_factory.sample(noise_cls, target=target)

        indices = self.rng.integers(0, len(self.noise_classes), size=n_noises)
        noises = [self.noise_factory.sample(self.noise_classes[int(index)], target=target) for index in indices]
        return CompositeNoise(noises)

    def _build_one(self) -> DatasetSample:
        generator = self._sample_generator()
        clean = generator.generate()
        noise = self._sample_noise(clean)
        noisy = noise.apply(clean)

        label = self.labeler(clean, noisy) if self.labeler is not None else None

        return DatasetSample(clean=clean, noisy=noisy, label=label)

    def build(self, n_samples: int) -> list[DatasetSample]:
        return [self._build_one() for _ in range(n_samples)]