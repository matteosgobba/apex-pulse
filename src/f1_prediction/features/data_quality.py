"""Checkpoint-safe practice signal quality and outlier flags."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DataQualitySettings:
    """Thresholds for transparent practice signal quality flags."""

    extreme_gap_to_session_best_sec: float = 3.0
    min_push_laps_latest_session: int = 2
    min_valid_laps_latest_session: int = 5


DATA_QUALITY_FEATURE_COLUMNS: tuple[str, ...] = (
    "has_any_practice_time",
    "has_latest_checkpoint_time",
    "latest_available_session",
    "n_available_sessions",
    "n_total_push_laps_available",
    "n_total_valid_laps_available",
    "missing_latest_best_push_lap",
    "missing_latest_theoretical_best_lap",
    "practice_signal_quality_score",
    "latest_best_push_gap_to_session_best_is_extreme",
    "latest_best_valid_gap_to_session_best_is_extreme",
    "latest_theoretical_gap_to_session_best_is_extreme",
)

_CHECKPOINT_SESSIONS: dict[str, tuple[str, ...]] = {
    "after_fp1": ("fp1",),
    "after_fp2": ("fp1", "fp2"),
    "after_fp3": ("fp1", "fp2", "fp3"),
}
_TIME_SUFFIXES = (
    "best_push_lap_time_sec",
    "best_valid_lap_time_sec",
    "theoretical_best_lap_time_sec",
)


def add_data_quality_features(
    dataset: pd.DataFrame,
    settings: DataQualitySettings | None = None,
) -> pd.DataFrame:
    """Add quality and extreme-gap flags using checkpoint-available practice data."""
    if "checkpoint" not in dataset:
        raise ValueError("Data-quality input is missing column: checkpoint")
    config = settings or DataQualitySettings()
    if (
        config.extreme_gap_to_session_best_sec <= 0
        or config.min_push_laps_latest_session < 1
        or config.min_valid_laps_latest_session < 1
    ):
        raise ValueError("Data-quality thresholds must be positive")

    frame = dataset.drop(columns=list(DATA_QUALITY_FEATURE_COLUMNS), errors="ignore").copy()
    for checkpoint, indices in frame.groupby("checkpoint", sort=False).groups.items():
        if checkpoint not in _CHECKPOINT_SESSIONS:
            raise ValueError(f"Unsupported checkpoint: {checkpoint}")
        rows = frame.loc[indices]
        sessions = _CHECKPOINT_SESSIONS[str(checkpoint)]
        latest = sessions[-1]
        availability = pd.DataFrame(
            {session: _session_time_available(rows, session) for session in sessions},
            index=rows.index,
        )
        frame.loc[indices, "has_any_practice_time"] = availability.any(axis=1)
        frame.loc[indices, "has_latest_checkpoint_time"] = availability[latest]
        frame.loc[indices, "latest_available_session"] = _latest_available(availability, sessions)
        frame.loc[indices, "n_available_sessions"] = availability.sum(axis=1)
        frame.loc[indices, "n_total_push_laps_available"] = _sum_session_counts(
            rows, sessions, "n_push_laps"
        )
        frame.loc[indices, "n_total_valid_laps_available"] = _sum_session_counts(
            rows, sessions, "n_valid_laps"
        )

        push = _numeric_column(rows, f"{latest}_best_push_lap_time_sec")
        valid = _numeric_column(rows, f"{latest}_best_valid_lap_time_sec")
        theoretical = _numeric_column(rows, f"{latest}_theoretical_best_lap_time_sec")
        push_count = _numeric_column(rows, f"{latest}_n_push_laps")
        valid_count = _numeric_column(rows, f"{latest}_n_valid_laps")
        team_relative = _team_relative_available(rows, latest)

        frame.loc[indices, "missing_latest_best_push_lap"] = push.isna()
        frame.loc[indices, "missing_latest_theoretical_best_lap"] = theoretical.isna()
        score = (
            push.notna().astype(int)
            + valid.notna().astype(int)
            + theoretical.notna().astype(int)
            + push_count.ge(config.min_push_laps_latest_session).astype(int)
            + valid_count.ge(config.min_valid_laps_latest_session).astype(int)
            + team_relative.astype(int)
        )
        frame.loc[indices, "practice_signal_quality_score"] = score

        gap_columns = {
            "latest_best_push_gap_to_session_best_is_extreme": (
                f"{latest}_best_push_gap_to_session_best_sec"
            ),
            "latest_best_valid_gap_to_session_best_is_extreme": (
                f"{latest}_best_valid_gap_to_session_best_sec"
            ),
            "latest_theoretical_gap_to_session_best_is_extreme": (
                f"{latest}_theoretical_best_gap_to_session_best_sec"
            ),
        }
        for output_column, source_column in gap_columns.items():
            gap = _numeric_column(rows, source_column)
            frame.loc[indices, output_column] = gap.gt(config.extreme_gap_to_session_best_sec)

    boolean_columns = [
        column
        for column in DATA_QUALITY_FEATURE_COLUMNS
        if column.startswith(("has_", "missing_")) or column.endswith("_is_extreme")
    ]
    frame[boolean_columns] = frame[boolean_columns].astype(bool)
    integer_columns = (
        "n_available_sessions",
        "n_total_push_laps_available",
        "n_total_valid_laps_available",
        "practice_signal_quality_score",
    )
    frame[list(integer_columns)] = frame[list(integer_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    return frame


def _session_time_available(rows: pd.DataFrame, session: str) -> pd.Series:
    values = [_numeric_column(rows, f"{session}_{suffix}").notna() for suffix in _TIME_SUFFIXES]
    return pd.concat(values, axis=1).any(axis=1)


def _latest_available(availability: pd.DataFrame, sessions: tuple[str, ...]) -> pd.Series:
    latest = pd.Series(pd.NA, index=availability.index, dtype="string")
    for session in sessions:
        latest = latest.mask(availability[session], session.upper())
    return latest


def _sum_session_counts(rows: pd.DataFrame, sessions: tuple[str, ...], suffix: str) -> pd.Series:
    counts = pd.concat(
        [_numeric_column(rows, f"{session}_{suffix}") for session in sessions], axis=1
    )
    return counts.fillna(0).sum(axis=1)


def _team_relative_available(rows: pd.DataFrame, session: str) -> pd.Series:
    columns = (
        f"{session}_best_push_gap_to_teammate_sec",
        f"{session}_best_valid_gap_to_teammate_sec",
        f"{session}_theoretical_best_gap_to_teammate_sec",
        f"{session}_driver_gap_to_team_best_push_sec",
        f"{session}_driver_gap_to_team_best_valid_sec",
        f"{session}_driver_gap_to_team_theoretical_best_sec",
    )
    available = [_numeric_column(rows, column).notna() for column in columns]
    return pd.concat(available, axis=1).any(axis=1)


def _numeric_column(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows:
        return pd.Series(float("nan"), index=rows.index)
    return pd.to_numeric(rows[column], errors="coerce")
