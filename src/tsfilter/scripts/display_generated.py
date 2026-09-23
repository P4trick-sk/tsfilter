from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tsfilter.utils.graphics import plot_series


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot every CSV file in the data folder.")
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=Path(__file__).resolve().parents[1] / "data_output",
		help="Directory containing CSV files to plot.",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		metavar="N",
		help="Plot only the first N CSV files after sorting by name.",
	)
	return parser.parse_args()


def load_csv_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
	with path.open(newline="") as file:
		reader = csv.DictReader(file)
		clean_values: list[float] = []
		noisy_values: list[float] = []

		for row in reader:
			clean_values.append(float(row["clean"]))
			noisy_values.append(float(row["noisy"]))

	return np.asarray(clean_values), np.asarray(noisy_values)


def plot_csv_file(path: Path) -> None:
	clean, noisy = load_csv_series(path)
	ax = plot_series(clean, title=path.stem, label="clean")
	plot_series(noisy, label="noisy", ax=ax)


def main() -> None:
	args = parse_args()
	csv_files: list[Path] = list(args.data_dir.glob("*.csv"))
	csv_files.sort(key=lambda path: path.name)
	if args.limit is not None:
		csv_files = csv_files[:args.limit]

	if not csv_files:
		raise FileNotFoundError(f"No CSV files found in {args.data_dir}")

	for path in csv_files:
		plot_csv_file(path)

	plt.show()


if __name__ == "__main__":
	main()


