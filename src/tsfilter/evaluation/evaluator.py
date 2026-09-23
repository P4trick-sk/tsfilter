from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .metrics import BaseMetric

RESERVED_COLUMNS = {"t", "clean", "noisy"}


@dataclass
class EvaluationResult:
    winner: str
    votes: dict[str, int]
    per_metric_winner: dict[str, str]
    scores: dict[str, dict[str, float]]   # filter_column -> metric_name -> value


class Evaluator:
    def __init__(self, metrics: list[BaseMetric]):
        # order matters: earlier metrics have HIGHER tie-break priority
        self.metrics = metrics

    def evaluate_sample(self, csv_path: str | Path) -> EvaluationResult:
        df = pd.read_csv(csv_path)
        filter_columns = [c for c in df.columns if c not in RESERVED_COLUMNS]
        clean = df["clean"].to_numpy()

        scores = {f: {} for f in filter_columns}
        votes = {f: 0 for f in filter_columns}
        per_metric_winner = {}

        for metric in self.metrics:
            metric_scores = {}
            for f in filter_columns:
                value = metric.compute(clean, df[f].to_numpy())
                scores[f][metric.name] = value
                metric_scores[f] = value

            best = max(metric_scores, key=metric_scores.get) if metric.higher_is_better \
                else min(metric_scores, key=metric_scores.get)
            votes[best] += 1
            per_metric_winner[metric.name] = best

        winner = self._resolve_tie(votes, per_metric_winner)
        return EvaluationResult(winner=winner, votes=votes, per_metric_winner=per_metric_winner, scores=scores)

    def _resolve_tie(self, votes: dict[str, int], per_metric_winner: dict[str, str]) -> str:
        top_votes = max(votes.values())
        tied = [f for f, v in votes.items() if v == top_votes]
        if len(tied) == 1:
            return tied[0]

        # walk metrics in priority order; first one whose own winner is
        # among the tied filters breaks the tie
        for metric in self.metrics:
            if per_metric_winner[metric.name] in tied:
                return per_metric_winner[metric.name]
        return tied[0]  # only reached if every metric's winner is somehow outside `tied` — shouldn't happen