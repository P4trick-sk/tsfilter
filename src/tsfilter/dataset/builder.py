from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from tsfilter.generators.base import BaseGenerator, GeneratedSeries
from tsfilter.generators.factory import RandomGeneratorFactory
from tsfilter.noise.base import BaseNoise, NoisySeries
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

    def _build_one(self) -> DatasetSample:
        gen_cls = self.generator_classes[self.rng.integers(len(self.generator_classes))]
        noise_cls = self.noise_classes[self.rng.integers(len(self.noise_classes))]

        generator = self.generator_factory.sample(gen_cls)
        noise = self.noise_factory.sample(noise_cls)

        clean = generator.generate()
        noisy = noise.apply(clean)

        label = self.labeler(clean, noisy) if self.labeler is not None else None

        return DatasetSample(clean=clean, noisy=noisy, label=label)

    def build(self, n_samples: int) -> list[DatasetSample]:
        return [self._build_one() for _ in range(n_samples)]