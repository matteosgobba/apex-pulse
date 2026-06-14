"""Coverage and missingness reporting for combined modeling datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.data.season_builder import build_combined_dataset_path
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


def build_dataset_quality_report(dataset: pd.DataFrame) -> dict[str, object]:
    """Build a JSON-safe coverage and missingness report."""
    required = {"season", "event", "event_slug", "checkpoint", "driver"}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"Modeling dataset is missing columns: {', '.join(missing)}")

    frame = dataset.copy()
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
        "practice_only_driver_rows": int((feature_present & ~target_present).sum()),
        "qualifying_only_driver_rows_if_detectable": int((target_present & ~feature_present).sum()),
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
