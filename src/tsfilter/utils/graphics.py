from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from tsfilter.generators.base import GeneratedSeries

def plot_series(
    series: GeneratedSeries | np.ndarray,
    title: str | None = None,
    label: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    values = series.values if isinstance(series, GeneratedSeries) else series

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(values, label=label)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    if title:
        ax.set_title(title)
    if label:
        ax.legend()

    return ax