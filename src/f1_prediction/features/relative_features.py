"""Leakage-safe relative features derived only from practice aggregates."""

from __future__ import annotations

import pandas as pd

from f1_prediction.data.identity import add_identity_columns

SESSION_GROUP_COLUMNS: tuple[str, ...] = ("season", "event_slug", "session_slug")
METRIC_COLUMNS: dict[str, str] = {
    "best_push": "best_push_lap_time_sec",
    "best_valid": "best_valid_lap_time_sec",
    "theoretical_best": "theoretical_best_lap_time_sec",
}
LEGACY_RANK_COLUMNS: dict[str, str] = {
    "push_lap_rank": "best_push_rank",
    "valid_lap_rank": "best_valid_rank",
}


def add_relative_practice_features(practice_features: pd.DataFrame) -> pd.DataFrame:
    """Add session and team comparisons without reading qualifying columns."""
    required = {
        *SESSION_GROUP_COLUMNS,
        "driver",
        "team",
    }
    missing = sorted(required - set(practice_features.columns))
    if missing:
        raise ValueError(
            f"Practice features are missing relative-feature identifiers: {', '.join(missing)}"
        )

    features = add_identity_columns(practice_features)
    for metric_column in METRIC_COLUMNS.values():
        if metric_column not in features:
            features[metric_column] = float("nan")
    features = features.rename(
        columns={
            old: new
            for old, new in LEGACY_RANK_COLUMNS.items()
            if old in features and new not in features
        }
    )
    for feature_name, metric_column in METRIC_COLUMNS.items():
        metric = pd.to_numeric(features[metric_column], errors="coerce")
        session_best = metric.groupby(
            [features[column] for column in SESSION_GROUP_COLUMNS],
            dropna=False,
        ).transform("min")
        features[f"{feature_name}_gap_to_session_best_sec"] = metric - session_best
        features[f"{feature_name}_rank"] = _rank(
            metric,
            features,
            list(SESSION_GROUP_COLUMNS),
        )
        features[f"{feature_name}_gap_pct_to_session_best"] = _safe_ratio(
            metric - session_best,
            session_best,
        )
        _add_team_features(features, metric, feature_name)
    return features


def _add_team_features(
    features: pd.DataFrame,
    metric: pd.Series,
    feature_name: str,
) -> None:
    team_column = "team_key" if "team_key" in features else "team"
    valid_team = features[team_column].notna()
    team_best = pd.Series(float("nan"), index=features.index, dtype="float64")
    team_rank = pd.Series(pd.NA, index=features.index, dtype="Int64")
    teammate_best = pd.Series(float("nan"), index=features.index, dtype="float64")

    grouped = (
        features.loc[valid_team]
        .assign(_metric=metric.loc[valid_team])
        .groupby(
            [*SESSION_GROUP_COLUMNS, team_column],
            dropna=False,
            sort=False,
        )
    )
    for _, group in grouped:
        values = group["_metric"]
        team_best.loc[group.index] = values.min()
        team_rank.loc[group.index] = values.rank(
            method="min", ascending=True, na_option="keep"
        ).astype("Int64")
        for index in group.index:
            other_values = values.drop(index).dropna()
            if not other_values.empty:
                teammate_best.loc[index] = float(other_values.min())

    features[f"{feature_name}_gap_to_teammate_sec"] = metric - teammate_best
    features[f"{feature_name}_team_rank"] = team_rank
    team_metric_name = {
        "best_push": "team_best_push_lap_time_sec",
        "best_valid": "team_best_valid_lap_time_sec",
        "theoretical_best": "team_theoretical_best_lap_time_sec",
    }[feature_name]
    driver_gap_name = {
        "best_push": "driver_gap_to_team_best_push_sec",
        "best_valid": "driver_gap_to_team_best_valid_sec",
        "theoretical_best": "driver_gap_to_team_theoretical_best_sec",
    }[feature_name]
    features[team_metric_name] = team_best
    features[driver_gap_name] = metric - team_best


def _rank(metric: pd.Series, features: pd.DataFrame, columns: list[str]) -> pd.Series:
    ranked = metric.groupby(
        [features[column] for column in columns],
        dropna=False,
    ).rank(method="min", ascending=True, na_option="keep")
    return ranked.astype("Int64")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid = denominator.notna() & denominator.ne(0)
    result = pd.Series(float("nan"), index=numerator.index, dtype="float64")
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result
