"""Transparent MVP qualifying target construction from raw Q laps."""

from __future__ import annotations

import pandas as pd

from f1_prediction.features.lap_cleaning import clean_session_laps
from f1_prediction.utils.paths import slugify

TARGET_COLUMNS: tuple[str, ...] = (
    "quali_position",
    "quali_best_lap_time_sec",
    "quali_gap_to_pole_sec",
    "reached_q2",
    "reached_q3",
)


def build_qualifying_targets(
    raw_qualifying_laps: pd.DataFrame,
    *,
    season: int,
    event: str,
) -> pd.DataFrame:
    """Build per-driver qualifying targets using valid, accurate lap times."""
    cleaned = clean_session_laps(
        raw_qualifying_laps,
        season=season,
        event=event,
        session="Q",
    )
    identity_columns = [
        "driver",
        "driver_code",
        "driver_name",
        "driver_key",
        "team",
        "team_name",
        "team_key",
    ]
    drivers = cleaned.loc[cleaned["driver_key"].notna(), identity_columns].drop_duplicates(
        "driver_key"
    )
    valid = cleaned[cleaned["is_valid_lap"]]
    best_times = valid.groupby("driver_key", sort=False)["lap_time_sec"].min()

    targets = drivers.reset_index(drop=True)
    targets["season"] = season
    targets["event"] = event
    targets["event_slug"] = slugify(event)
    targets["quali_best_lap_time_sec"] = targets["driver_key"].map(best_times)
    targets = _assign_ordinal_positions(targets)

    pole_time = targets["quali_best_lap_time_sec"].min()
    targets["quali_gap_to_pole_sec"] = targets["quali_best_lap_time_sec"] - pole_time
    has_valid_time = targets["quali_best_lap_time_sec"].notna()
    targets["reached_q2"] = (has_valid_time & targets["quali_position"].le(15)).astype("int8")
    targets["reached_q3"] = (has_valid_time & targets["quali_position"].le(10)).astype("int8")

    columns = [
        "season",
        "event",
        "event_slug",
        *identity_columns,
        *TARGET_COLUMNS,
    ]
    return targets.loc[:, columns]


def _assign_ordinal_positions(targets: pd.DataFrame) -> pd.DataFrame:
    with_time = targets[targets["quali_best_lap_time_sec"].notna()].sort_values(
        "quali_best_lap_time_sec",
        kind="stable",
    )
    without_time = targets[targets["quali_best_lap_time_sec"].isna()]
    ordered = pd.concat([with_time, without_time], ignore_index=True)
    ordered["quali_position"] = pd.Series(range(1, len(ordered) + 1), dtype="Int64")
    return ordered
