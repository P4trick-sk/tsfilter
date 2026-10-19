from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsfilter.evaluation.evaluator import Evaluator
from tsfilter.evaluation.metrics import MSEMetric, MAEMetric, SNRMetric

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine per-filter CSVs and evaluate each sample.")
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True, help="Path to export results")
    return parser.parse_args()

def export_to_json(filename: str | Path, data: list[dict]) -> None:
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def main() -> None:
    args = parse_args()

    evaluator = Evaluator(metrics=[MSEMetric(), SNRMetric(), MAEMetric()])
    data = []

    print(f"Evaluating combined samples in {args.dir} ...")
    for csv_path in sorted(args.dir.glob("*.csv")):
        result = evaluator.evaluate_sample(csv_path)

        if not args.export:
            print(f"{csv_path.stem}: winner={result.winner} ")
        else:
            data.append({
                "datasample_name": str(csv_path.stem),
                "winner": str(result.winner)
            })

    if args.export:
        export_to_json(args.export, data)


if __name__ == "__main__":
    main()