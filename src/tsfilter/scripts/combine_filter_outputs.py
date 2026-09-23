from __future__ import annotations

import argparse
from pathlib import Path

from tsfilter.evaluation.combine import combine_filter_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-filter subdirectory CSVs into one CSV per sample."
    )
    parser.add_argument(
        "--filters-dir", type=Path, required=True,
        help="Root directory containing one subdirectory per filter",
    )
    parser.add_argument(
        "--combined-dir", type=Path, required=True,
        help="Directory to write the combined per-sample CSVs into",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Combining filter outputs from {args.filters_dir} ...")
    combine_filter_outputs(args.filters_dir, args.combined_dir)
    print(f"Done. Combined CSVs written to {args.combined_dir}")


if __name__ == "__main__":
    main()