"""Normalize raw FastF1 lap data and derive lap validity flags."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from f1_prediction.data.identity import add_identity_columns
from f1_prediction.utils.paths import ensure_directory, slugify

TIME_COLUMN_MAP: dict[str, str] = {
    "LapTime": "lap_time_sec",
    "Sector1Time": "sector1_time_sec",
    "Sector2Time": "sector2_time_sec",
    "Sector3Time": "sector3_time_sec",
}


def clean_session_laps(
    raw_laps: pd.DataFrame,
    *,
    season: int,
    event: str,
    session: str,
) -> pd.DataFrame:
    """Preserve raw columns and add a stable normalized lap schema."""
    cleaned = raw_laps.copy()
    cleaned["season"] = season
    cleaned["event"] = event
    cleaned["event_slug"] = slugify(event)
    cleaned["session"] = session.upper()
    cleaned["session_slug"] = slugify(session)

    cleaned["driver"] = _normalized_string(raw_laps, "Driver", uppercase=True)
    cleaned["team"] = _normalized_string(raw_laps, "Team")
    cleaned = add_identity_columns(cleaned)
    cleaned["lap_number"] = _numeric_column(raw_laps, "LapNumber")
    cleaned["stint"] = _numeric_column(raw_laps, "Stint")
    cleaned["compound"] = _normalized_string(raw_laps, "Compound", uppercase=True)
    cleaned["tyre_life"] = _numeric_column(raw_laps, "TyreLife")

    for raw_column, cleaned_column in TIME_COLUMN_MAP.items():
        cleaned[cleaned_column] = time_to_seconds(_column_or_missing(raw_laps, raw_column))

    cleaned["is_accurate"] = _boolean_column(raw_laps, "IsAccurate", fallback=True)
    cleaned["is_deleted"] = _boolean_column(raw_laps, "Deleted", fallback=False)
    cleaned["is_in_lap"] = _time_flag(raw_laps, "PitInTime")
    cleaned["is_out_lap"] = _time_flag(raw_laps, "PitOutTime")
    cleaned["is_valid_lap"] = (
        cleaned["lap_time_sec"].notna()
        & cleaned["sector1_time_sec"].notna()
        & cleaned["sector2_time_sec"].notna()
        & cleaned["sector3_time_sec"].notna()
        & cleaned["is_accurate"]
        & ~cleaned["is_deleted"]
        & ~cleaned["is_in_lap"]
        & ~cleaned["is_out_lap"]
        & cleaned["driver"].notna()
        & cleaned["lap_number"].notna()
    )
    cleaned["is_push_lap"] = False
    return cleaned


def time_to_seconds(values: pd.Series) -> pd.Series:
    """Convert timedelta-like values to floating-point seconds."""
    if pd.api.types.is_numeric_dtype(values.dtype):
        return pd.to_numeric(values, errors="coerce").astype(float)
    timedeltas = pd.to_timedelta(values, errors="coerce")
    return timedeltas.dt.total_seconds()


def build_clean_lap_output_path(
    output_dir: Path,
    season: int,
    event: str,
    session: str,
) -> Path:
    """Build the deterministic path for cleaned session laps."""
    event_dir = ensure_directory(output_dir / str(season) / slugify(event))
    return event_dir / f"{slugify(session)}_clean_laps.parquet"


def _column_or_missing(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame:
        return frame[column]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(_column_or_missing(frame, column), errors="coerce")


def _normalized_string(
    frame: pd.DataFrame,
    column: str,
    *,
    uppercase: bool = False,
) -> pd.Series:
    values = _column_or_missing(frame, column).astype("string").str.strip()
    values = values.mask(values.eq(""))
    return values.str.upper() if uppercase else values


def _boolean_column(frame: pd.DataFrame, column: str, *, fallback: bool) -> pd.Series:
    if column not in frame:
        return pd.Series(fallback, index=frame.index, dtype=bool)

    values = frame[column]
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype("boolean").fillna(fallback).astype(bool)

    normalized = values.astype("string").str.strip().str.lower()
    true_values = {"true", "1", "yes", "y"}
    false_values = {"false", "0", "no", "n"}
    result = normalized.map(
        lambda value: True if value in true_values else False if value in false_values else fallback
    )
    return result.astype(bool)


def _time_flag(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].notna()
