"""Stable columns exported from FastF1 lap data."""

from collections.abc import Sequence

import pandas as pd

BASIC_LAP_COLUMNS: tuple[str, ...] = (
    "Time",
    "Driver",
    "DriverNumber",
    "LapTime",
    "LapNumber",
    "Stint",
    "PitOutTime",
    "PitInTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    "IsPersonalBest",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "Team",
    "TrackStatus",
    "Position",
    "Deleted",
    "DeletedReason",
    "FastF1Generated",
    "IsAccurate",
)


def select_basic_lap_data(
    laps: pd.DataFrame,
    columns: Sequence[str] = BASIC_LAP_COLUMNS,
) -> pd.DataFrame:
    """Return a copy containing the available stable lap-level columns."""
    available_columns = [column for column in columns if column in laps.columns]
    if not available_columns:
        raise ValueError("FastF1 returned no recognized lap columns")
    return pd.DataFrame(laps.loc[:, available_columns]).reset_index(drop=True)
