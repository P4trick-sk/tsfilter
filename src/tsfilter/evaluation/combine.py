from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RESERVED_COLUMNS = {"t", "clean", "noisy"}
SAMPLE_ID_PATTERN = re.compile(r"(\d{8}T\d{6}_\d+)\.csv$")


def _extract_sample_id(filename: str) -> str | None:
    match = SAMPLE_ID_PATTERN.search(filename)
    return match.group(1) if match else None


def _group_files_by_sample(root_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for filter_dir in root_dir.iterdir():
        if not filter_dir.is_dir():
            continue
        for csv_path in filter_dir.glob("*.csv"):
            sample_id = _extract_sample_id(csv_path.name)
            if sample_id is None:
                continue  # doesn't match expected naming — skipped, not silently merged wrong
            groups.setdefault(sample_id, []).append(csv_path)
    return groups


def combine_sample(files: list[Path]) -> pd.DataFrame:
    base: pd.DataFrame | None = None
    param_frames: list[pd.DataFrame] = []
    seen_columns: set[str] = set(RESERVED_COLUMNS)

    for path in sorted(files):
        df = pd.read_csv(path)

        if base is None:
            base = df[["t", "clean", "noisy"]].copy()
        elif not base["clean"].equals(df["clean"]) or not base["noisy"].equals(df["noisy"]):
            raise ValueError(f"clean/noisy mismatch in {path} — files don't belong to the same sample")

        param_columns = [c for c in df.columns if c not in RESERVED_COLUMNS]
        collisions = seen_columns & set(param_columns)
        if collisions:
            raise ValueError(f"Column name collision from {path}: {collisions}")
        seen_columns.update(param_columns)

        param_frames.append(df[param_columns])

    return pd.concat([base, *param_frames], axis=1)


def combine_filter_outputs(root_dir: str | Path, output_dir: str | Path) -> None:
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_id, files in _group_files_by_sample(root_dir).items():
        combine_sample(files).to_csv(output_dir / f"{sample_id}.csv", index=False)