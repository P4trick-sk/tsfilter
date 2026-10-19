from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from tsfilter.generators.base import BaseGenerator, GeneratedSeries
from tsfilter.generators.composite import CompositeGenerator, MultiGeneratorChoice, MixedGeneratorChoice
from tsfilter.generators.factory import RandomGeneratorFactory
from tsfilter.noise.base import BaseNoise, NoisySeries
from tsfilter.noise.composite import MultiNoiseChoice, MixedNoiseChoice, CompositeNoise
from tsfilter.noise.factory import RandomNoiseFactory
from tsfilter.dataset.sample import DatasetSample

Labeler = Callable[[GeneratedSeries, NoisySeries], dict]

class DatasetBuilder:
    def __init__(
        self,
        generator_factory: RandomGeneratorFactory,
        noise_factory: RandomNoiseFactory,
        generator_classes: Sequence[type[BaseGenerator]],
        noise_classes: Sequence[type[BaseNoise]],
        labeler: Labeler | None = None,
        seed: int | None = None,
    ):
        self.generator_factory = generator_factory
        self.noise_factory = noise_factory
        self.generator_classes = generator_classes
        self.noise_classes = noise_classes
        self.labeler = labeler
        self.rng = np.random.default_rng(seed)

    def _sample_entry(self, choice: BaseGenerator) -> BaseGenerator:
        if isinstance(choice, MultiGeneratorChoice):
            n = int(self.rng.integers(choice.n_components.low, choice.n_components.high + 1))
            components = [self.generator_factory.sample(choice.base) for _ in range(n)]
            return CompositeGenerator(components)

        if isinstance(choice, MixedGeneratorChoice):
            n = int(self.rng.integers(choice.n_components.low, choice.n_components.high + 1))
            indices = self.rng.integers(0, len(choice.candidates), size=n)
            components = [self._sample_entry(choice.candidates[i]) for i in indices]
            return CompositeGenerator(components)

        return self.generator_factory.sample(choice)

    def _sample_generator(self):
        choice = self.generator_classes[self.rng.integers(len(self.generator_classes))]
        return self._sample_entry(choice)

    def _sample_noise(self, target: GeneratedSeries) -> BaseNoise:
        choice = self.noise_classes[self.rng.integers(len(self.noise_classes))]

        if isinstance(choice, MultiNoiseChoice):
            n = int(self.rng.integers(choice.n_components.low, choice.n_components.high + 1))
            components = [self.noise_factory.sample(choice.base, target=target) for _ in range(n)]
            return CompositeNoise(components)

        if isinstance(choice, MixedNoiseChoice):
            n = int(self.rng.integers(choice.n_components.low, choice.n_components.high + 1))
            indices = self.rng.integers(0, len(choice.candidates), size=n)
            components = [self.noise_factory.sample(choice.candidates[i], target=target) for i in indices]
            return CompositeNoise(components)

        return self.noise_factory.sample(choice, target=target)

    def _build_one(self) -> DatasetSample:
        generator = self._sample_generator()
        clean = generator.generate()
        noise = self._sample_noise(clean)
        noisy = noise.apply(clean)

        label = self.labeler(clean, noisy) if self.labeler is not None else None
        return DatasetSample(clean=clean, noisy=noisy, label=label)

    def build(self, n_samples: int) -> list[DatasetSample]:
        return [self._build_one() for _ in range(n_samples)]