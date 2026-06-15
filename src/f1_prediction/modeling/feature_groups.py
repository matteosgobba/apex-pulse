"""Safe, column-driven feature groups for ablation evaluation."""

from __future__ import annotations

import pandas as pd

from f1_prediction.modeling.splits import get_numeric_feature_columns

DEFAULT_ABLATION_GROUPS: tuple[str, ...] = (
    "base_lap_features",
    "base_plus_relative",
    "base_plus_historical",
    "base_plus_quality",
    "all_features",
)

_RELATIVE_PATTERNS = (
    "teammate",
    "team_",
    "gap_pct",
    "team_rank",
    "driver_gap_to_team",
    "gap_to_session_best",
)
_HISTORICAL_PATTERNS = ("rolling", "expanding", "prev_events")
_QUALITY_PATTERNS = (
    "quality",
    "extreme",
    "latest_available",
    "missing_latest",
    "has_latest",
    "has_any_practice",
    "n_available_sessions",
    "n_total_push_laps_available",
    "n_total_valid_laps_available",
)


def get_feature_groups(dataset: pd.DataFrame) -> dict[str, list[str]]:
    """Return disjoint base/component groups plus documented combinations."""
    all_features = get_numeric_feature_columns(dataset)
    historical = [column for column in all_features if _matches(column, _HISTORICAL_PATTERNS)]
    quality = [column for column in all_features if _matches(column, _QUALITY_PATTERNS)]
    relative = [
        column
        for column in all_features
        if column not in set(historical) | set(quality) and _matches(column, _RELATIVE_PATTERNS)
    ]
    specialized = set(relative) | set(historical) | set(quality)
    base = [column for column in all_features if column not in specialized]

    groups = {
        "base_lap_features": base,
        "relative_features": relative,
        "historical_features": historical,
        "data_quality_features": quality,
        "base_plus_relative": _combine(all_features, base, relative),
        "base_plus_historical": _combine(all_features, base, historical),
        "base_plus_quality": _combine(all_features, base, quality),
        "base_plus_relative_historical": _combine(all_features, base, relative, historical),
        "base_plus_relative_quality": _combine(all_features, base, relative, quality),
        "base_plus_historical_quality": _combine(all_features, base, historical, quality),
        "all_features": all_features,
    }
    return groups


def get_feature_columns_for_group(dataset: pd.DataFrame, group_name: str) -> list[str]:
    """Return one registered safe feature group or raise a clear error."""
    groups = get_feature_groups(dataset)
    if group_name not in groups:
        available = ", ".join(groups)
        raise ValueError(f"Unknown feature group '{group_name}'. Available groups: {available}")
    return groups[group_name]


def _matches(column: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in column for pattern in patterns)


def _combine(all_features: list[str], *groups: list[str]) -> list[str]:
    selected = {column for group in groups for column in group}
    return [column for column in all_features if column in selected]
