"""Stable columns exported from FastF1 lap data."""

from collections.abc import Sequence

import pandas as pd

REQUIRED_LAP_COLUMNS: tuple[str, ...] = (
    "Driver",
    "LapTime",
    "LapNumber",
    "Stint",
    "Compound",
    "TyreLife",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "IsAccurate",
)

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


def validate_lap_schema(
    laps: pd.DataFrame,
    required_columns: Sequence[str] = REQUIRED_LAP_COLUMNS,
) -> None:
    """Raise a clear error when required exported lap columns are missing."""
    missing_columns = [column for column in required_columns if column not in laps.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Lap data is missing required columns: {missing}")
