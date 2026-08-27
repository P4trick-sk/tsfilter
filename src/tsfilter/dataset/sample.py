from __future__ import annotations

from dataclasses import dataclass

from tsfilter.generators.base import GeneratedSeries
from tsfilter.noise.base import NoisySeries

@dataclass
class DatasetSample:
    clean: GeneratedSeries
    noisy: NoisySeries
    label: dict | None = None