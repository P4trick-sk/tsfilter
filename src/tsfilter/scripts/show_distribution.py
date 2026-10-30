from __future__ import annotations

import argparse
from pathlib import Path

from tsfilter.evaluation.labels import load_labels, get_distribution


def print_distribution(counts, title: str) -> None:
    number_of_samples = sum(counts.values())
    top_count = max(counts.values())

    print(f"\n{title}  ({number_of_samples} samples, {len(counts)} classes)")
    print(f"{'class':<32}{'count':>7}{'share':>8}{'missing to match top':>22}")

    for name, count in counts.most_common():
        print(f"{name:<32}{count:>7}{count / number_of_samples:>8.1%}{top_count - count:>22}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show how evenly filter methods win in a labelled dataset.")
    parser.add_argument("--labels", type=Path, required=True, help="Evaluator json export")
    parser.add_argument("--level", choices=["filter_class", "winner", "both"], default="filter_class",
                        help="filter_class = method only, winner = method + parameters")
    parser.add_argument("--plot", type=Path, help="Optional path to save a bar chart (png)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels = ["filter_class", "winner"] if args.level == "both" else [args.level]

    for level in levels:
        print_distribution(get_distribution(load_labels(args.labels, level)), f"Distribution by {level}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts = get_distribution(load_labels(args.labels, levels[-1])).most_common()
        plt.figure(figsize=(max(6, 0.4 * len(counts)), 4))
        plt.bar([n for n, _ in counts], [c for _, c in counts])
        plt.xticks(rotation=70, ha="right")
        plt.ylabel("samples")
        plt.tight_layout()
        plt.savefig(args.plot)
        print(f"\nPlot saved to {args.plot}")


if __name__ == "__main__":
    main()
