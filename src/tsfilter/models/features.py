from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "mean", "std", "min", "max", "skew", "kurtosis", "median",  "mad", "iqr",
    "spec_low", "spec_mid", "spec_high", "spec_flatness",
]


def extract_features(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    mean, std, median = x.mean(), x.std(), np.median(x)
    z_score = (x - mean) / (std + 1e-12) # male cislo, aby sme nedelili 0, keby nahodou
    skew = np.mean(z_score ** 3)
    kurt = np.mean(z_score ** 4) - 3.0

    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1 # medzikvartilove rozmedzie - rozptyl strednych 50 % hodnot

    median_deviation = np.abs(x - median)
    mad = np.median(median_deviation)

    power = np.abs(np.fft.rfft(x )) ** 2
    power = power[1:] # bez nultého prvku - priemer, ten už máme vo features
    total = power.sum() + 1e-12 # male cislo, aby sme nedelili 0, keby nahodou
    low, mid, high = (band.sum() / total for band in np.array_split(power, 3))
    flatness = np.exp(np.mean(np.log(power + 1e-12))) / (power.mean() + 1e-12) # ako je sústredená energia v binoch

    return np.array([mean, std, x.min(), x.max(), skew, kurt, median, mad, iqr, low, mid, high, flatness], dtype=np.float32)
