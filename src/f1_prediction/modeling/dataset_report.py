"""Coverage and missingness reporting for combined modeling datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.data.identity import add_identity_columns
from f1_prediction.data.season_builder import build_combined_dataset_path
from f1_prediction.features.data_quality import DATA_QUALITY_FEATURE_COLUMNS
from f1_prediction.features.historical_features import HISTORICAL_FEATURE_COLUMNS
from f1_prediction.features.modeling_dataset import get_feature_columns
from f1_prediction.features.qualifying_targets import TARGET_COLUMNS
from f1_prediction.utils.paths import ensure_directory


@dataclass(frozen=True)
class DatasetQualitySummary:
    """Key counts and output path for a dataset quality report."""

    dataset_path: Path
    report_path: Path
    n_rows: int
    n_seasons: int
    n_events: int
    n_drivers: int
    checkpoints: tuple[str, ...]
    historical_feature_count: int
    data_quality_feature_count: int


def build_dataset_quality_report(dataset: pd.DataFrame) -> dict[str, object]:
    """Build a JSON-safe coverage and missingness report."""
    required = {"season", "event", "event_slug", "checkpoint", "driver"}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"Modeling dataset is missing columns: {', '.join(missing)}")

    frame = add_identity_columns(dataset)
    frame["_event_key"] = _event_keys(frame)
    feature_columns = get_feature_columns(frame)
    numeric_features = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(frame[column])
    ]
    target_columns = [column for column in TARGET_COLUMNS if column in frame]
    event_order = frame["_event_key"].drop_duplicates().tolist()
    checkpoint_order = frame["checkpoint"].dropna().astype(str).drop_duplicates().tolist()
    all_checkpoints = set(checkpoint_order)

    rows_by_event = frame.groupby("_event_key", sort=False).size()
    checkpoints_per_event = frame.groupby("_event_key", sort=False)["checkpoint"].agg(
        lambda values: list(dict.fromkeys(values.dropna().astype(str)))
    )
    events_with_missing = [
        event
        for event in event_order
        if set(checkpoints_per_event.get(event, [])) != all_checkpoints
    ]
    feature_present = frame[feature_columns].notna().any(axis=1) if feature_columns else False
    target_present = frame[target_columns].notna().any(axis=1) if target_columns else False
    historical_columns = [
        column for column in HISTORICAL_FEATURE_COLUMNS if column in frame.columns
    ]
    quality_columns = [column for column in DATA_QUALITY_FEATURE_COLUMNS if column in frame.columns]
    quality_score = pd.to_numeric(
        frame.get("practice_signal_quality_score", pd.Series(float("nan"), index=frame.index)),
        errors="coerce",
    )
    extreme_columns = [column for column in quality_columns if column.endswith("_is_extreme")]
    extreme_rows = (
        frame[extreme_columns].fillna(False).astype(bool).any(axis=1)
        if extreme_columns
        else pd.Series(False, index=frame.index)
    )
    event_driver_rows = frame.drop_duplicates(["season", "event_slug", "driver_key"])
    events_by_season = event_driver_rows.groupby("season", sort=True)["event_slug"].nunique()
    drivers_by_season = event_driver_rows.groupby("season", sort=True)["driver_key"].nunique()
    teams_by_season = event_driver_rows.groupby("season", sort=True)["team_key"].nunique()
    driver_team_counts = (
        frame.dropna(subset=["driver_key", "team_key"])
        .drop_duplicates(["driver_key", "team_key"])
        .groupby("driver_key", sort=True)["team_key"]
        .nunique()
    )
    drivers_multiple_teams = driver_team_counts[driver_team_counts.gt(1)].index.tolist()
    team_event_counts = (
        event_driver_rows.dropna(subset=["team_key"])
        .groupby(["_event_key", "team_key"], sort=False)["driver_key"]
        .nunique()
    )
    single_driver_teams = [
        f"{event}/{team}" for (event, team), count in team_event_counts.items() if count < 2
    ]
    event_driver_counts = event_driver_rows.groupby("_event_key", sort=False)[
        "driver_key"
    ].nunique()

    return {
        "n_rows": len(frame),
        "n_seasons": int(frame["season"].nunique()),
        "seasons": sorted(_native_list(frame["season"].dropna().unique())),
        "n_events": int(frame["_event_key"].nunique()),
        "events": event_order,
        "n_drivers": int(frame["driver"].nunique()),
        "drivers": sorted(frame["driver"].dropna().astype(str).unique().tolist()),
        "checkpoints": checkpoint_order,
        "rows_by_season": _count_mapping(frame.groupby("season", sort=True).size()),
        "events_by_season": _count_mapping(events_by_season),
        "drivers_by_season": _count_mapping(drivers_by_season),
        "teams_by_season": _count_mapping(teams_by_season),
        "rows_by_event": _count_mapping(rows_by_event),
        "rows_by_checkpoint": _count_mapping(frame.groupby("checkpoint", sort=False).size()),
        "missing_target_counts": {
            column: int(frame[column].isna().sum()) for column in target_columns
        },
        "missing_feature_counts_top_30": _top_missing_counts(frame, feature_columns),
        "numeric_feature_missing_rate_top_30": _top_missing_rates(frame, numeric_features),
        "drivers_per_event": _count_mapping(
            frame.groupby("_event_key", sort=False)["driver"].nunique()
        ),
        "checkpoints_per_event": {str(key): value for key, value in checkpoints_per_event.items()},
        "events_with_missing_checkpoints": events_with_missing,
        "driver_key_count": int(frame["driver_key"].nunique()),
        "team_key_count": int(frame["team_key"].nunique()),
        "driver_key_missing_count": int(frame["driver_key"].isna().sum()),
        "team_key_missing_count": int(frame["team_key"].isna().sum()),
        "team_key_distribution": _count_mapping(
            event_driver_rows["team_key"].value_counts(dropna=True, sort=False)
        ),
        "driver_key_distribution": _count_mapping(
            event_driver_rows["driver_key"].value_counts(dropna=True, sort=False)
        ),
        "drivers_appearing_under_multiple_team_keys": drivers_multiple_teams,
        "teams_with_single_driver_events": single_driver_teams,
        "events_with_less_than_20_drivers": [
            str(event) for event, count in event_driver_counts.items() if count < 20
        ],
        "events_with_failed_or_missing_sessions_if_detectable": events_with_missing,
        "practice_only_driver_rows": int((feature_present & ~target_present).sum()),
        "qualifying_only_driver_rows_if_detectable": int((target_present & ~feature_present).sum()),
        "historical_feature_count": len(historical_columns),
        "data_quality_feature_count": len(quality_columns),
        "rows_with_low_practice_signal_quality": int(quality_score.lt(3).sum()),
        "rows_with_extreme_latest_practice_signal": int(extreme_rows.sum()),
        "created_at_utc": _utc_now(),
    }


def create_dataset_quality_report(
    config: DataConfig,
    dataset_path: Path | None = None,
) -> DatasetQualitySummary:
    """Read a combined dataset and persist its quality report."""
    source_path = _resolve_dataset_path(config, dataset_path)
    dataset = pd.read_parquet(source_path)
    report = build_dataset_quality_report(dataset)
    report_path = config.metrics_output_dir / "dataset_quality_report.json"
    ensure_directory(report_path.parent)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, allow_nan=False)
        report_file.write("\n")
    return DatasetQualitySummary(
        dataset_path=source_path,
        report_path=report_path,
        n_rows=int(report["n_rows"]),
        n_seasons=int(report["n_seasons"]),
        n_events=int(report["n_events"]),
        n_drivers=int(report["n_drivers"]),
        checkpoints=tuple(report["checkpoints"]),
        historical_feature_count=int(report["historical_feature_count"]),
        data_quality_feature_count=int(report["data_quality_feature_count"]),
    )


def _resolve_dataset_path(config: DataConfig, dataset_path: Path | None) -> Path:
    path = dataset_path or build_combined_dataset_path(config.modeling_output_dir)
    if not path.is_absolute():
        path = config.project_root / path
    if not path.is_file():
        raise FileNotFoundError(f"Modeling dataset does not exist: {path}")
    return path


def _event_keys(frame: pd.DataFrame) -> pd.Series:
    return frame["season"].astype(str) + "/" + frame["event_slug"].astype(str)


def _count_mapping(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.items()}


def _top_missing_counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    counts = frame[columns].isna().sum().sort_values(ascending=False, kind="stable").head(30)
    return {str(column): int(value) for column, value in counts.items()}


def _top_missing_rates(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    rates = frame[columns].isna().mean().sort_values(ascending=False, kind="stable").head(30)
    return {str(column): float(value) for column, value in rates.items()}


def _native_list(values: object) -> list[int | float | str]:
    return [value.item() if hasattr(value, "item") else value for value in values]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
