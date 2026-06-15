"""Event and driver error diagnostics for saved prediction artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from f1_prediction.config import DataConfig, DiagnosticsConfig
from f1_prediction.data.season_builder import build_combined_dataset_path
from f1_prediction.modeling.baselines import BASELINE_FEATURES
from f1_prediction.modeling.metrics import compute_prediction_metrics
from f1_prediction.utils.paths import ensure_directory

EVENT_SUMMARY_COLUMNS: tuple[str, ...] = (
    "prediction_source",
    "season",
    "event",
    "event_slug",
    "checkpoint",
    "model_name",
    "n_rows",
    "mae_gap_sec",
    "rmse_gap_sec",
    "median_abs_error_gap_sec",
    "mean_abs_position_error",
    "spearman_corr",
    "top_3_accuracy",
    "top_5_accuracy",
    "top_10_accuracy",
    "q3_accuracy",
)
DRIVER_SUMMARY_COLUMNS: tuple[str, ...] = (
    "prediction_source",
    "driver",
    "checkpoint",
    "model_name",
    "n_rows",
    "mae_gap_sec",
    "mean_abs_position_error",
    "avg_actual_position",
    "avg_predicted_position",
)


@dataclass(frozen=True)
class DiagnosticsReportSummary:
    """Paths and counts produced by diagnostics generation."""

    preferred_prediction_source: str
    available_prediction_sources: tuple[str, ...]
    n_events: int
    n_drivers: int
    report_path: Path
    event_summary_path: Path
    driver_summary_path: Path


def build_event_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute one metric row per source, event, checkpoint, and model."""
    rows: list[dict[str, object]] = []
    group_columns = [
        "prediction_source",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "model_name",
    ]
    for keys, group in predictions.groupby(group_columns, dropna=False, sort=False):
        metrics = compute_prediction_metrics(group)
        row = dict(zip(group_columns, keys, strict=True))
        row["n_rows"] = len(group)
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows, columns=EVENT_SUMMARY_COLUMNS)


def build_driver_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute driver-level absolute gap and position errors."""
    rows: list[dict[str, object]] = []
    group_columns = ["prediction_source", "driver", "checkpoint", "model_name"]
    for keys, group in predictions.groupby(group_columns, dropna=False, sort=False):
        gap_rows = group.dropna(subset=["quali_gap_to_pole_sec", "predicted_quali_gap_to_pole_sec"])
        position_rows = group.dropna(subset=["quali_position", "predicted_quali_position"])
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "n_rows": len(group),
                "mae_gap_sec": _mean_absolute_error(
                    gap_rows["predicted_quali_gap_to_pole_sec"],
                    gap_rows["quali_gap_to_pole_sec"],
                ),
                "mean_abs_position_error": _mean_absolute_error(
                    position_rows["predicted_quali_position"],
                    position_rows["quali_position"],
                ),
                "avg_actual_position": _finite_or_none(position_rows["quali_position"].mean()),
                "avg_predicted_position": _finite_or_none(
                    position_rows["predicted_quali_position"].mean()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=DRIVER_SUMMARY_COLUMNS)


def build_diagnostics_report_payload(
    event_summary: pd.DataFrame,
    driver_summary: pd.DataFrame,
    *,
    available_sources: list[str],
    preferred_source: str,
    n_events: int,
    n_drivers: int,
    checkpoints: list[str],
    config: DiagnosticsConfig,
) -> dict[str, object]:
    """Build top-error and extreme-error diagnostics for the preferred source."""
    preferred_events = event_summary[event_summary["prediction_source"].eq(preferred_source)]
    preferred_drivers = driver_summary[driver_summary["prediction_source"].eq(preferred_source)]
    baseline_mask = preferred_events["model_name"].isin(BASELINE_FEATURES)
    model_mask = ~baseline_mask
    return {
        "available_prediction_sources": available_sources,
        "preferred_prediction_source": preferred_source,
        "n_events": n_events,
        "n_drivers": n_drivers,
        "checkpoints": checkpoints,
        "worst_events_by_mae_gap": _top_records(preferred_events, "mae_gap_sec", config.top_n),
        "worst_events_by_position_error": _top_records(
            preferred_events, "mean_abs_position_error", config.top_n
        ),
        "worst_drivers_by_mae_gap": _top_records(preferred_drivers, "mae_gap_sec", config.top_n),
        "worst_drivers_by_position_error": _top_records(
            preferred_drivers, "mean_abs_position_error", config.top_n
        ),
        "worst_checkpoint_model_combinations": _checkpoint_model_records(
            preferred_events, config.top_n
        ),
        "baseline_extreme_error_events": _records_above_threshold(
            preferred_events[baseline_mask],
            config.extreme_mae_gap_threshold_sec,
        ),
        "model_extreme_error_events": _records_above_threshold(
            preferred_events[model_mask],
            config.extreme_mae_gap_threshold_sec,
        ),
        "created_at_utc": _utc_now(),
    }


def create_diagnostics_report(
    data_config: DataConfig,
    diagnostics_config: DiagnosticsConfig,
    *,
    walk_forward_path: Path | None = None,
    repeated_path: Path | None = None,
    baseline_path: Path | None = None,
    champion_path: Path | None = None,
    dataset_path: Path | None = None,
) -> DiagnosticsReportSummary:
    """Load available prediction sources and persist diagnostics artifacts."""
    paths = {
        "walk_forward": walk_forward_path
        or data_config.metrics_output_dir / "walk_forward_predictions.parquet",
        "repeated_event_holdout": repeated_path
        or data_config.metrics_output_dir / "repeated_event_holdout_predictions.parquet",
        "baseline_full_dataset": baseline_path
        or data_config.metrics_output_dir / "baseline_predictions.parquet",
        "champion": champion_path
        or data_config.metrics_output_dir / "champion_predictions.parquet",
    }
    frames: list[pd.DataFrame] = []
    available: list[str] = []
    for source, path in paths.items():
        resolved = _resolve_optional_path(path, data_config.project_root)
        if not resolved.is_file():
            continue
        frame = _standardize_predictions(pd.read_parquet(resolved), source)
        frames.append(frame)
        available.append(source)
    if not frames:
        raise FileNotFoundError(
            "No champion, walk-forward, repeated-holdout, or baseline predictions exist"
        )

    predictions = pd.concat(frames, ignore_index=True, sort=False)
    preferred = next(
        source
        for source in (
            "champion",
            "walk_forward",
            "repeated_event_holdout",
            "baseline_full_dataset",
        )
        if source in available
    )
    dataset_source = dataset_path or build_combined_dataset_path(data_config.modeling_output_dir)
    dataset_source = _resolve_optional_path(dataset_source, data_config.project_root)
    dataset = pd.read_parquet(dataset_source) if dataset_source.is_file() else pd.DataFrame()
    preferred_rows = predictions[predictions["prediction_source"].eq(preferred)]
    event_summary = build_event_error_summary(predictions)
    driver_summary = build_driver_error_summary(predictions)
    report = build_diagnostics_report_payload(
        event_summary,
        driver_summary,
        available_sources=available,
        preferred_source=preferred,
        n_events=(
            dataset[["season", "event_slug"]].drop_duplicates().shape[0]
            if not dataset.empty
            else preferred_rows[["season", "event_slug"]].drop_duplicates().shape[0]
        ),
        n_drivers=(
            int(dataset["driver"].nunique())
            if not dataset.empty
            else int(preferred_rows["driver"].nunique())
        ),
        checkpoints=preferred_rows["checkpoint"].dropna().astype(str).drop_duplicates().tolist(),
        config=diagnostics_config,
    )
    ensure_directory(data_config.metrics_output_dir)
    report_path = data_config.metrics_output_dir / "diagnostics_report.json"
    event_path = data_config.metrics_output_dir / "event_error_summary.parquet"
    driver_path = data_config.metrics_output_dir / "driver_error_summary.parquet"
    event_summary.to_parquet(event_path, engine="pyarrow", index=False)
    driver_summary.to_parquet(driver_path, engine="pyarrow", index=False)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, allow_nan=False)
        report_file.write("\n")
    return DiagnosticsReportSummary(
        preferred_prediction_source=preferred,
        available_prediction_sources=tuple(available),
        n_events=int(report["n_events"]),
        n_drivers=int(report["n_drivers"]),
        report_path=report_path,
        event_summary_path=event_path,
        driver_summary_path=driver_path,
    )


def _standardize_predictions(predictions: pd.DataFrame, source: str) -> pd.DataFrame:
    frame = predictions.copy()
    if "model_name" not in frame and "selected_model_name" in frame:
        frame["model_name"] = frame["selected_model_name"]
    if "model_name" not in frame and "baseline_name" in frame:
        frame["model_name"] = frame["baseline_name"]
    if "event" not in frame:
        frame["event"] = frame["event_slug"]
    frame["prediction_source"] = source
    required = {
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "driver",
        "model_name",
        "quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec",
        "quali_position",
        "predicted_quali_position",
        "reached_q3",
        "predicted_reached_q3",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction source {source} is missing columns: {', '.join(missing)}")
    return frame


def _top_records(frame: pd.DataFrame, metric: str, top_n: int) -> list[dict[str, object]]:
    ranked = frame.dropna(subset=[metric]).sort_values(metric, ascending=False).head(top_n)
    return _json_records(ranked)


def _checkpoint_model_records(frame: pd.DataFrame, top_n: int) -> list[dict[str, object]]:
    grouped = (
        frame.groupby(["prediction_source", "checkpoint", "model_name"], sort=False)
        .agg(
            n_rows=("n_rows", "sum"),
            mae_gap_sec=("mae_gap_sec", "mean"),
            mean_abs_position_error=("mean_abs_position_error", "mean"),
        )
        .reset_index()
    )
    return _top_records(grouped, "mae_gap_sec", top_n)


def _records_above_threshold(frame: pd.DataFrame, threshold: float) -> list[dict[str, object]]:
    extreme = frame[frame["mae_gap_sec"].ge(threshold)].sort_values("mae_gap_sec", ascending=False)
    return _json_records(extreme)


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {key: _json_value(value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def _json_value(value: object) -> object:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _mean_absolute_error(predicted: pd.Series, actual: pd.Series) -> float | None:
    if predicted.empty:
        return None
    return _finite_or_none((predicted.astype(float) - actual.astype(float)).abs().mean())


def _finite_or_none(value: object) -> float | None:
    if pd.isna(value):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _resolve_optional_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
