from __future__ import annotations

from typing import Sequence
from dataclasses import dataclass

import numpy as np
from tsfilter.generators.base import BaseGenerator, GeneratedSeries
from tsfilter.utils.sampling import ParamRange

@dataclass
class MultiGeneratorChoice:
    base: type[BaseGenerator]
    n_components: ParamRange

@dataclass
class MixedGeneratorChoice:
    candidates: list[type[BaseGenerator]]
    n_components: ParamRange

class CompositeGenerator(BaseGenerator):
    def __init__(self, generators: Sequence[BaseGenerator]):
        lengths = {g.length for g in generators}
        if len(lengths) != 1:
            raise ValueError(f"All generators must share one length, got {lengths}")
        super().__init__(length=lengths.pop())
        self.generators = generators

    def generate(self) -> GeneratedSeries:
        parts = [g.generate() for g in self.generators]
        values = np.sum([p.values for p in parts], axis=0)
        return GeneratedSeries(
            values=values,
            params={"generator": "composite", "components": [p.params for p in parts]},
        )