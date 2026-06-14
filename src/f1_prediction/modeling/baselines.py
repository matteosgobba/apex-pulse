"""Simple non-ML practice pace baselines."""

from __future__ import annotations

import pandas as pd

from f1_prediction.features.qualifying_targets import TARGET_COLUMNS

BASELINE_FEATURES: dict[str, str] = {
    "best_push_lap": "best_push_lap_time_sec",
    "best_valid_lap": "best_valid_lap_time_sec",
    "theoretical_best_lap": "theoretical_best_lap_time_sec",
}
CHECKPOINT_SESSION_PRIORITY: dict[str, tuple[str, ...]] = {
    "after_fp1": ("fp1",),
    "after_fp2": ("fp2", "fp1"),
    "after_fp3": ("fp3", "fp2", "fp1"),
}
PREDICTION_IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "season",
    "event",
    "event_slug",
    "checkpoint",
    "driver",
    "team",
)


def generate_baseline_predictions(dataset: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions for every transparent baseline."""
    predictions = [predict_baseline(dataset, name) for name in BASELINE_FEATURES]
    return pd.concat(predictions, ignore_index=True)


def predict_baseline(dataset: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    """Rank drivers using the latest available checkpoint-safe practice metric."""
    if baseline_name not in BASELINE_FEATURES:
        raise ValueError(f"Unknown baseline: {baseline_name}")
    required = {*PREDICTION_IDENTIFIER_COLUMNS, *TARGET_COLUMNS}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"Modeling dataset is missing columns: {', '.join(missing)}")

    result = dataset.loc[:, [*PREDICTION_IDENTIFIER_COLUMNS, *TARGET_COLUMNS]].copy()
    selected_metrics: list[pd.Series] = []
    selected_sources: list[pd.Series] = []
    metric_suffix = BASELINE_FEATURES[baseline_name]

    for checkpoint, rows in dataset.groupby("checkpoint", sort=False):
        if checkpoint not in CHECKPOINT_SESSION_PRIORITY:
            raise ValueError(f"Unsupported checkpoint: {checkpoint}")
        metric, source = select_checkpoint_metric(rows, checkpoint, metric_suffix)
        selected_metrics.append(metric)
        selected_sources.append(source)

    result["baseline_name"] = baseline_name
    result["selected_practice_metric_sec"] = pd.concat(selected_metrics).sort_index()
    result["selected_practice_session"] = pd.concat(selected_sources).sort_index()
    result["predicted_quali_gap_to_pole_sec"] = result.groupby(
        ["season", "event_slug", "checkpoint"]
    )["selected_practice_metric_sec"].transform(lambda values: values - values.min())
    result["predicted_quali_position"] = _ordinal_prediction_rank(result)
    result["predicted_reached_q3"] = result["predicted_quali_position"].le(10).astype("int8")
    return result


def select_checkpoint_metric(
    rows: pd.DataFrame,
    checkpoint: str,
    metric_suffix: str,
) -> tuple[pd.Series, pd.Series]:
    """Select the latest non-null metric allowed at one checkpoint."""
    session_priority = CHECKPOINT_SESSION_PRIORITY[checkpoint]
    metric_columns = [f"{session}_{metric_suffix}" for session in session_priority]
    available = pd.DataFrame(
        {
            column: rows[column] if column in rows else pd.Series(pd.NA, index=rows.index)
            for column in metric_columns
        },
        index=rows.index,
    )
    numeric = available.apply(pd.to_numeric, errors="coerce")
    selected = numeric.bfill(axis=1).iloc[:, 0]

    source = pd.Series(pd.NA, index=rows.index, dtype="string")
    for session, column in zip(session_priority, metric_columns, strict=True):
        source = source.mask(source.isna() & numeric[column].notna(), session.upper())
    return selected, source


def _ordinal_prediction_rank(predictions: pd.DataFrame) -> pd.Series:
    positions = pd.Series(index=predictions.index, dtype="Int64")
    group_columns = ["season", "event_slug", "checkpoint"]
    for _, group in predictions.groupby(group_columns, sort=False):
        ordered = group.assign(
            _missing=group["selected_practice_metric_sec"].isna(),
        ).sort_values(
            ["_missing", "selected_practice_metric_sec", "driver"],
            kind="stable",
            na_position="last",
        )
        positions.loc[ordered.index] = range(1, len(ordered) + 1)
    return positions
