"""Metrics for evaluating transparent qualifying baselines."""

from __future__ import annotations

import math

import pandas as pd


def compute_baseline_metrics(
    predictions: pd.DataFrame,
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Compute requested regression, ranking, and Q3 metrics by baseline/checkpoint."""
    metrics: dict[str, dict[str, dict[str, float | None]]] = {}
    for baseline_name, baseline_rows in predictions.groupby("baseline_name", sort=False):
        metrics[str(baseline_name)] = {}
        for checkpoint, checkpoint_rows in baseline_rows.groupby("checkpoint", sort=False):
            metrics[str(baseline_name)][str(checkpoint)] = compute_prediction_metrics(
                checkpoint_rows
            )
    return metrics


def compute_prediction_metrics(predictions: pd.DataFrame) -> dict[str, float | None]:
    """Compute all metrics for one baseline and checkpoint."""
    regression = predictions.dropna(
        subset=["quali_gap_to_pole_sec", "predicted_quali_gap_to_pole_sec"]
    )
    gap_error = (
        regression["predicted_quali_gap_to_pole_sec"] - regression["quali_gap_to_pole_sec"]
    ).abs()
    ranking = predictions.dropna(subset=["quali_position", "predicted_quali_position"])
    position_error = (
        ranking["predicted_quali_position"].astype(float) - ranking["quali_position"].astype(float)
    ).abs()

    return {
        "mae_gap_sec": _finite_or_none(gap_error.mean()),
        "rmse_gap_sec": _finite_or_none(math.sqrt((gap_error.pow(2)).mean())),
        "median_abs_error_gap_sec": _finite_or_none(gap_error.median()),
        "mean_abs_position_error": _finite_or_none(position_error.mean()),
        "spearman_corr": _spearman(ranking),
        "top_3_accuracy": _top_k_recall(ranking, 3),
        "top_5_accuracy": _top_k_recall(ranking, 5),
        "top_10_accuracy": _top_k_recall(ranking, 10),
        "q3_accuracy": _finite_or_none(
            predictions["predicted_reached_q3"].eq(predictions["reached_q3"]).mean()
        ),
    }


def _spearman(ranking: pd.DataFrame) -> float | None:
    event_correlations: list[float] = []
    for _, event_rows in ranking.groupby(["season", "event_slug"], sort=False):
        if (
            len(event_rows) < 2
            or event_rows["predicted_quali_position"].nunique() < 2
            or event_rows["quali_position"].nunique() < 2
        ):
            continue
        correlation = (
            event_rows["predicted_quali_position"]
            .astype(float)
            .corr(
                event_rows["quali_position"].astype(float),
                method="spearman",
            )
        )
        if pd.notna(correlation):
            event_correlations.append(float(correlation))
    if not event_correlations:
        return None
    return float(sum(event_correlations) / len(event_correlations))


def _top_k_recall(ranking: pd.DataFrame, k: int) -> float | None:
    event_scores: list[float] = []
    for _, event_rows in ranking.groupby(["season", "event_slug"], sort=False):
        actual = set(event_rows.loc[event_rows["quali_position"].le(k), "driver"])
        predicted = set(event_rows.loc[event_rows["predicted_quali_position"].le(k), "driver"])
        if actual:
            event_scores.append(len(actual & predicted) / len(actual))
    if not event_scores:
        return None
    return float(sum(event_scores) / len(event_scores))


def _finite_or_none(value: object) -> float | None:
    if pd.isna(value):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None
