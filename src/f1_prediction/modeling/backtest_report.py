"""Compact model-versus-baseline backtesting report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.data.season_builder import build_combined_dataset_path
from f1_prediction.modeling.dataset_report import build_dataset_quality_report
from f1_prediction.utils.paths import ensure_directory

TABULAR_MODEL_NAMES: tuple[str, ...] = ("ridge", "random_forest")


@dataclass(frozen=True)
class BacktestReportSummary:
    """Key backtesting result counts and output path."""

    dataset_rows: int
    n_events: int
    training_status: str
    tabular_models: tuple[str, ...]
    output_path: Path


def create_backtest_report(
    config: DataConfig,
    *,
    dataset_path: Path | None = None,
    baseline_metrics_path: Path | None = None,
    tabular_metrics_path: Path | None = None,
    quality_report_path: Path | None = None,
) -> BacktestReportSummary:
    """Read available evaluation artifacts and persist a compact summary."""
    source_path = _resolve_path(
        dataset_path or build_combined_dataset_path(config.modeling_output_dir),
        config.project_root,
        required=True,
    )
    baseline_path = _resolve_path(
        baseline_metrics_path or config.metrics_output_dir / "baseline_metrics.json",
        config.project_root,
    )
    tabular_path = _resolve_path(
        tabular_metrics_path or config.metrics_output_dir / "tabular_model_metrics.json",
        config.project_root,
    )
    quality_path = _resolve_path(
        quality_report_path or config.metrics_output_dir / "dataset_quality_report.json",
        config.project_root,
    )
    dataset = pd.read_parquet(source_path)
    quality = (
        _read_json(quality_path)
        if quality_path.is_file()
        else build_dataset_quality_report(dataset)
    )
    baseline_metrics = _read_json(baseline_path) if baseline_path.is_file() else {}
    tabular_metrics = _read_json(tabular_path) if tabular_path.is_file() else None
    payload = build_backtest_report_payload(quality, baseline_metrics, tabular_metrics)

    output_path = config.metrics_output_dir / "backtest_report.json"
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, allow_nan=False)
        output_file.write("\n")
    return BacktestReportSummary(
        dataset_rows=int(payload["dataset_rows"]),
        n_events=int(payload["n_events"]),
        training_status=str(payload["training_status"]),
        tabular_models=tuple(payload["tabular_models_available"]),
        output_path=output_path,
    )


def build_backtest_report_payload(
    quality: dict[str, Any],
    baseline_metrics: dict[str, Any],
    tabular_metrics: dict[str, Any] | None,
) -> dict[str, object]:
    """Compose comparable best-model and best-baseline metrics by checkpoint."""
    training_status = (
        str(tabular_metrics.get("status", "unavailable")) if tabular_metrics else "unavailable"
    )
    model_metrics = tabular_metrics.get("models", {}) if training_status == "trained" else {}
    tabular_models = [name for name in TABULAR_MODEL_NAMES if name in model_metrics]
    checkpoints = [str(value) for value in quality.get("checkpoints", [])]
    if not checkpoints:
        checkpoints = _metric_checkpoints(baseline_metrics, model_metrics)

    holdout_baselines = (
        tabular_metrics.get("best_baseline_by_checkpoint", {})
        if training_status == "trained" and tabular_metrics
        else {}
    )
    best_baselines = _best_baselines(checkpoints, baseline_metrics, holdout_baselines)
    best_models = _best_tabular_models(checkpoints, model_metrics, tabular_models)
    mae_deltas, position_deltas = _comparison_deltas(
        checkpoints,
        best_baselines,
        best_models,
    )
    return {
        "dataset_rows": int(quality.get("n_rows", 0)),
        "n_events": int(quality.get("n_events", 0)),
        "events": list(quality.get("events", [])),
        "n_drivers": int(quality.get("n_drivers", 0)),
        "checkpoints": checkpoints,
        "best_baseline_by_checkpoint": best_baselines,
        "tabular_models_available": tabular_models,
        "best_tabular_model_by_checkpoint": best_models,
        "model_vs_baseline_delta_mae_by_checkpoint": mae_deltas,
        "model_vs_baseline_delta_position_error_by_checkpoint": position_deltas,
        "training_status": training_status,
        "created_at_utc": _utc_now(),
    }


def _best_baselines(
    checkpoints: list[str],
    baseline_metrics: dict[str, Any],
    holdout_baselines: dict[str, Any],
) -> dict[str, dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for checkpoint in checkpoints:
        holdout = holdout_baselines.get(checkpoint)
        if holdout and holdout.get("mae_gap_sec") is not None:
            best[checkpoint] = dict(holdout)
            continue
        candidates = [
            (name, metrics[checkpoint])
            for name, metrics in baseline_metrics.items()
            if checkpoint in metrics and metrics[checkpoint].get("mae_gap_sec") is not None
        ]
        if candidates:
            name, metrics = min(candidates, key=lambda item: float(item[1]["mae_gap_sec"]))
            best[checkpoint] = {
                "baseline_name": name,
                "mae_gap_sec": metrics.get("mae_gap_sec"),
                "mean_abs_position_error": metrics.get("mean_abs_position_error"),
            }
    return best


def _best_tabular_models(
    checkpoints: list[str],
    model_metrics: dict[str, Any],
    tabular_models: list[str],
) -> dict[str, dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for checkpoint in checkpoints:
        candidates = [
            (name, model_metrics[name][checkpoint])
            for name in tabular_models
            if checkpoint in model_metrics[name]
            and model_metrics[name][checkpoint].get("mae_gap_sec") is not None
        ]
        if candidates:
            name, metrics = min(candidates, key=lambda item: float(item[1]["mae_gap_sec"]))
            best[checkpoint] = {
                "model_name": name,
                "mae_gap_sec": metrics.get("mae_gap_sec"),
                "mean_abs_position_error": metrics.get("mean_abs_position_error"),
            }
    return best


def _comparison_deltas(
    checkpoints: list[str],
    baselines: dict[str, dict[str, object]],
    models: dict[str, dict[str, object]],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    mae: dict[str, float | None] = {}
    position: dict[str, float | None] = {}
    for checkpoint in checkpoints:
        baseline = baselines.get(checkpoint, {})
        model = models.get(checkpoint, {})
        mae[checkpoint] = _delta(model.get("mae_gap_sec"), baseline.get("mae_gap_sec"))
        position[checkpoint] = _delta(
            model.get("mean_abs_position_error"),
            baseline.get("mean_abs_position_error"),
        )
    return mae, position


def _delta(model_value: object, baseline_value: object) -> float | None:
    if model_value is None or baseline_value is None:
        return None
    return float(model_value) - float(baseline_value)


def _metric_checkpoints(*metric_groups: dict[str, Any]) -> list[str]:
    checkpoints: list[str] = []
    for group in metric_groups:
        for metrics in group.values():
            for checkpoint in metrics:
                if checkpoint not in checkpoints:
                    checkpoints.append(checkpoint)
    return checkpoints


def _resolve_path(path: Path, project_root: Path, *, required: bool = False) -> Path:
    resolved = path if path.is_absolute() else project_root / path
    if required and not resolved.is_file():
        raise FileNotFoundError(f"Required backtest input does not exist: {resolved}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report root must be an object: {path}")
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
