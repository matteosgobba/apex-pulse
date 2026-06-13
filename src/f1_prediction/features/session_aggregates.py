"""Driver/session aggregates derived from cleaned practice laps."""

from __future__ import annotations

import math

import pandas as pd

IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "season",
    "event",
    "event_slug",
    "session",
    "session_slug",
    "driver",
)


def aggregate_session_features(clean_laps: pd.DataFrame) -> pd.DataFrame:
    """Create one aggregate row per season, event, session, and driver."""
    identified_laps = clean_laps[clean_laps["driver"].notna()]
    rows = [
        _aggregate_driver(group)
        for _, group in identified_laps.groupby(
            list(IDENTIFIER_COLUMNS),
            dropna=False,
            sort=False,
        )
    ]
    features = pd.DataFrame(rows)
    if features.empty:
        return features

    session_groups = ["season", "event_slug", "session_slug"]
    features["best_valid_gap_to_session_best_sec"] = _gap_to_group_best(
        features,
        "best_valid_lap_time_sec",
        session_groups,
    )
    features["best_push_gap_to_session_best_sec"] = _gap_to_group_best(
        features,
        "best_push_lap_time_sec",
        session_groups,
    )
    features["valid_lap_rank"] = _rank_within_group(
        features,
        "best_valid_lap_time_sec",
        session_groups,
    )
    features["push_lap_rank"] = _rank_within_group(
        features,
        "best_push_lap_time_sec",
        session_groups,
    )
    return features


def _aggregate_driver(group: pd.DataFrame) -> dict[str, object]:
    valid = group[group["is_valid_lap"]]
    push = group[group["is_push_lap"]]
    best_lap = _fastest_row(group)

    best_sector1 = valid["sector1_time_sec"].min()
    best_sector2 = valid["sector2_time_sec"].min()
    best_sector3 = valid["sector3_time_sec"].min()
    theoretical_best = _sum_if_complete(best_sector1, best_sector2, best_sector3)
    best_valid = valid["lap_time_sec"].min()

    row: dict[str, object] = {column: group.iloc[0][column] for column in IDENTIFIER_COLUMNS}
    row.update(
        {
            "team": _first_present(group["team"]),
            "n_laps": len(group),
            "n_valid_laps": int(group["is_valid_lap"].sum()),
            "n_push_laps": int(group["is_push_lap"].sum()),
            "n_soft_laps": int(group["compound"].eq("SOFT").sum()),
            "n_medium_laps": int(group["compound"].eq("MEDIUM").sum()),
            "n_hard_laps": int(group["compound"].eq("HARD").sum()),
            "best_lap_time_sec": group["lap_time_sec"].min(),
            "best_valid_lap_time_sec": best_valid,
            "best_push_lap_time_sec": push["lap_time_sec"].min(),
            "median_valid_lap_time_sec": valid["lap_time_sec"].median(),
            "median_push_lap_time_sec": push["lap_time_sec"].median(),
            "std_valid_lap_time_sec": valid["lap_time_sec"].std(),
            "std_push_lap_time_sec": push["lap_time_sec"].std(),
            "best_sector1_time_sec": best_sector1,
            "best_sector2_time_sec": best_sector2,
            "best_sector3_time_sec": best_sector3,
            "theoretical_best_lap_time_sec": theoretical_best,
            "best_vs_theoretical_gap_sec": _difference(best_valid, theoretical_best),
            "best_lap_compound": _row_value(best_lap, "compound"),
            "best_lap_tyre_life": _row_value(best_lap, "tyre_life"),
            "avg_tyre_life_valid_laps": valid["tyre_life"].mean(),
            "avg_tyre_life_push_laps": push["tyre_life"].mean(),
        }
    )
    return row


def _fastest_row(group: pd.DataFrame) -> pd.Series | None:
    timed = group[group["lap_time_sec"].notna()]
    if timed.empty:
        return None
    return timed.loc[timed["lap_time_sec"].idxmin()]


def _row_value(row: pd.Series | None, column: str) -> object:
    return pd.NA if row is None else row[column]


def _first_present(values: pd.Series) -> object:
    present = values.dropna()
    return pd.NA if present.empty else present.iloc[0]


def _sum_if_complete(*values: object) -> float:
    if any(pd.isna(value) for value in values):
        return math.nan
    return float(sum(values))


def _difference(left: object, right: object) -> float:
    if pd.isna(left) or pd.isna(right):
        return math.nan
    return float(left) - float(right)


def _gap_to_group_best(
    features: pd.DataFrame,
    value_column: str,
    group_columns: list[str],
) -> pd.Series:
    group_best = features.groupby(group_columns)[value_column].transform("min")
    return features[value_column] - group_best


def _rank_within_group(
    features: pd.DataFrame,
    value_column: str,
    group_columns: list[str],
) -> pd.Series:
    return (
        features.groupby(group_columns)[value_column]
        .rank(
            method="min",
            ascending=True,
            na_option="keep",
        )
        .astype("Int64")
    )
