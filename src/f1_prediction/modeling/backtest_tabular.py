"""Repeated leakage-safe backtesting for simple tabular models."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig, FeatureConfig, ModelConfig
from f1_prediction.data.season_builder import build_combined_dataset_path
from f1_prediction.features.historical_features import (
    HistoricalFeatureSettings,
    add_historical_features,
)
from f1_prediction.modeling.baselines import generate_baseline_predictions
from f1_prediction.modeling.metrics import compute_baseline_metrics, compute_prediction_metrics
from f1_prediction.modeling.splits import (
    SplitStrategy,
    create_dataset_split,
    ordered_event_keys,
)
from f1_prediction.modeling.temporal_weighting import (
    TemporalWeightingPolicy,
    prepare_temporal_training_data,
    supported_weighted_models,
    temporal_artifact_stem,
    temporal_weighting_config_payload,
    unsupported_weighted_models,
)
from f1_prediction.modeling.train_tabular import (
    DEFAULT_MODEL_CONFIG,
    fit_and_predict,
    metrics_by_model_checkpoint,
)
from f1_prediction.utils.paths import ensure_directory

METRIC_NAMES: tuple[str, ...] = (
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
TABULAR_MODELS: tuple[str, ...] = ("ridge", "random_forest")


class BacktestStrategy(str, Enum):
    """Supported repeated tabular backtesting strategies."""

    repeated_event_holdout = "repeated_event_holdout"
    walk_forward = "walk_forward"


@dataclass(frozen=True)
class BacktestFold:
    """One event-safe train/test fold."""

    fold_id: int
    strategy: str
    test_event: str
    train_events: tuple[str, ...]
    train_rows: int
    test_rows: int
    status: str = "pending"
    error_message: str | None = None


@dataclass(frozen=True)
class TabularBacktestSummary:
    """Paths and counts produced by a repeated backtest."""

    status: str
    strategy: str
    n_events: int
    n_folds_total: int
    n_folds_successful: int
    n_folds_failed: int
    prediction_rows: int
    metrics_path: Path
    predictions_path: Path | None
    folds_path: Path


def build_backtest_folds(
    dataset: pd.DataFrame,
    strategy: BacktestStrategy | str,
    *,
    test_events: list[str] | None = None,
    min_train_events: int = 5,
) -> tuple[BacktestFold, ...]:
    """Build repeated holdout or chronological walk-forward fold definitions."""
    frame = dataset.reset_index(drop=True)
    strategy = BacktestStrategy(strategy)
    event_keys = ordered_event_keys(frame)
    row_keys = _event_key_series(frame)
    if strategy is BacktestStrategy.repeated_event_holdout:
        selected = _selected_event_keys(frame, test_events) if test_events else event_keys
        definitions = [
            (event_key, [key for key in event_keys if key != event_key]) for event_key in selected
        ]
    else:
        if min_train_events < 1:
            raise ValueError("min_train_events must be at least 1")
        definitions = [
            (event_keys[index], event_keys[:index])
            for index in range(min_train_events, len(event_keys))
        ]
    return tuple(
        BacktestFold(
            fold_id=index,
            strategy=strategy.value,
            test_event=test_event,
            train_events=tuple(train_events),
            train_rows=int(row_keys.isin(train_events).sum()),
            test_rows=int(row_keys.eq(test_event).sum()),
        )
        for index, (test_event, train_events) in enumerate(definitions, start=1)
    )


def run_tabular_backtest(
    config: DataConfig,
    *,
    strategy: BacktestStrategy | str,
    dataset_path: Path | None = None,
    test_events: list[str] | None = None,
    min_events: int = 5,
    min_train_events: int = 5,
    fail_fast: bool = False,
    model_config: ModelConfig | None = None,
    feature_config: FeatureConfig | None = None,
    temporal_weighting: TemporalWeightingPolicy | str = TemporalWeightingPolicy.uniform,
) -> TabularBacktestSummary:
    """Train and evaluate tabular models plus baselines on repeated event folds."""
    strategy = BacktestStrategy(strategy)
    if min_events < 2:
        raise ValueError("min_events must be at least 2")
    source_path = _resolve_dataset_path(config, dataset_path)
    dataset = pd.read_parquet(source_path).reset_index(drop=True)
    event_keys = ordered_event_keys(dataset)
    paths = _backtest_paths(config.metrics_output_dir, strategy)
    ensure_directory(config.metrics_output_dir)
    settings = model_config or DEFAULT_MODEL_CONFIG
    temporal_policy = TemporalWeightingPolicy(temporal_weighting)
    if len(event_keys) < min_events:
        reason = f"Dataset has {len(event_keys)} unique events; at least {min_events} are required"
        payload = _skipped_payload(strategy, len(event_keys), reason)
        payload["temporal_weighting_policy"] = temporal_policy.value
        payload["temporal_weighting_config"] = temporal_weighting_config_payload(
            settings.temporal_weighting
        )
        _write_json(paths["metrics"], payload)
        _write_json(
            paths["folds"],
            {
                "status": "skipped",
                "strategy": strategy.value,
                "folds": [],
                "temporal_weighting_policy": temporal_policy.value,
            },
        )
        _write_temporal_snapshots(paths, temporal_policy, payload, predictions_path=None)
        paths["predictions"].unlink(missing_ok=True)
        return TabularBacktestSummary(
            status="skipped",
            strategy=strategy.value,
            n_events=len(event_keys),
            n_folds_total=0,
            n_folds_successful=0,
            n_folds_failed=0,
            prediction_rows=0,
            metrics_path=paths["metrics"],
            predictions_path=None,
            folds_path=paths["folds"],
        )

    folds = build_backtest_folds(
        dataset,
        strategy,
        test_events=test_events,
        min_train_events=min_train_events,
    )
    if not folds:
        raise ValueError("No evaluable folds were created for the requested strategy")
    row_keys = _event_key_series(dataset)
    completed_folds: list[BacktestFold] = []
    tabular_frames: list[pd.DataFrame] = []
    baseline_frames: list[pd.DataFrame] = []
    historical_settings = _historical_settings(feature_config)
    robust_threshold = _robust_baseline_threshold(feature_config)
    weight_summaries: list[dict[str, object]] = []

    for fold in folds:
        try:
            fold_scope = dataset[row_keys.isin([*fold.train_events, fold.test_event])].copy()
            fold_scope = add_historical_features(
                fold_scope,
                historical_settings,
                excluded_target_events={fold.test_event},
            )
            fold_keys = _event_key_series(fold_scope)
            train = fold_scope[fold_keys.isin(fold.train_events)].copy()
            test = fold_scope[fold_keys.eq(fold.test_event)].copy()
            if train.empty or test.empty:
                raise ValueError("Fold must contain both training and test rows")
            temporal_result = prepare_temporal_training_data(
                train,
                test_event=fold.test_event,
                event_order=event_keys,
                config=settings.temporal_weighting,
                policy=temporal_policy,
            )
            if temporal_result.train.empty:
                raise ValueError("Temporal weighting left no training rows")
            fold_weight_summary = dict(temporal_result.summary)
            fold_weight_summary["fold_id"] = fold.fold_id
            weight_summaries.append(fold_weight_summary)
            tabular_predictions, _ = fit_and_predict(
                temporal_result.train,
                test,
                model_config=settings,
                sample_weights=temporal_result.sample_weights,
            )
            baseline_predictions = generate_baseline_predictions(
                test,
                robust_extreme_threshold_sec=robust_threshold,
            )
            tabular_frames.append(
                _tag_tabular_predictions(tabular_predictions, fold, temporal_policy)
            )
            baseline_frames.append(
                _tag_baseline_predictions(baseline_predictions, fold, temporal_policy)
            )
            completed_folds.append(_completed_fold(fold, "success", None))
        except Exception as exc:
            completed_folds.append(_completed_fold(fold, "failed", _concise_error(exc)))
            if fail_fast:
                break

    successful = [fold for fold in completed_folds if fold.status == "success"]
    failed = [fold for fold in completed_folds if fold.status == "failed"]
    if tabular_frames:
        tabular_predictions = pd.concat(tabular_frames, ignore_index=True, sort=False)
        baseline_predictions = pd.concat(baseline_frames, ignore_index=True, sort=False)
        combined_predictions = pd.concat(
            [tabular_predictions, baseline_predictions], ignore_index=True, sort=False
        )
        combined_predictions.to_parquet(paths["predictions"], engine="pyarrow", index=False)
        payload = build_backtest_metrics_payload(
            strategy,
            len(event_keys),
            completed_folds,
            tabular_predictions,
            baseline_predictions,
            temporal_weighting_policy=temporal_policy,
            temporal_weighting_config=temporal_weighting_config_payload(
                settings.temporal_weighting
            ),
            training_weight_summary_by_fold=weight_summaries,
        )
        status = "complete" if not failed else "partial"
        payload["status"] = status
        predictions_path: Path | None = paths["predictions"]
    else:
        status = "failed"
        payload = _failed_payload(strategy, len(event_keys), completed_folds)
        paths["predictions"].unlink(missing_ok=True)
        predictions_path = None
    folds_payload = {
        "status": status,
        "strategy": strategy.value,
        "n_folds_total": len(completed_folds),
        "n_folds_successful": len(successful),
        "n_folds_failed": len(failed),
        "folds": [asdict(fold) for fold in completed_folds],
        "temporal_weighting_policy": temporal_policy.value,
        "training_weight_summary_by_fold": weight_summaries,
        "created_at_utc": _utc_now(),
    }
    _write_json(paths["metrics"], payload)
    _write_json(paths["folds"], folds_payload)
    _write_temporal_snapshots(paths, temporal_policy, payload, predictions_path=predictions_path)
    return TabularBacktestSummary(
        status=status,
        strategy=strategy.value,
        n_events=len(event_keys),
        n_folds_total=len(completed_folds),
        n_folds_successful=len(successful),
        n_folds_failed=len(failed),
        prediction_rows=(
            len(tabular_predictions) + len(baseline_predictions) if tabular_frames else 0
        ),
        metrics_path=paths["metrics"],
        predictions_path=predictions_path,
        folds_path=paths["folds"],
    )


def build_backtest_metrics_payload(
    strategy: BacktestStrategy | str,
    n_events: int,
    folds: list[BacktestFold],
    tabular_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    temporal_weighting_policy: TemporalWeightingPolicy | str = TemporalWeightingPolicy.uniform,
    temporal_weighting_config: dict[str, object] | None = None,
    training_weight_summary_by_fold: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Aggregate global and fold-level metrics plus fold-consistent comparisons."""
    strategy = BacktestStrategy(strategy)
    tabular_global = metrics_by_model_checkpoint(tabular_predictions)
    temporal_policy = TemporalWeightingPolicy(temporal_weighting_policy)
    tabular_fold = _fold_metrics(tabular_predictions, "model_name")
    baseline_global = compute_baseline_metrics(baseline_predictions)
    baseline_fold = _fold_metrics(baseline_predictions, "baseline_name")
    tabular_aggregate = _aggregate_metrics(tabular_global, tabular_fold)
    baseline_aggregate = _aggregate_metrics(baseline_global, baseline_fold)
    checkpoints = tabular_predictions["checkpoint"].drop_duplicates().astype(str).tolist()
    best_baselines = _best_by_checkpoint(baseline_global, checkpoints, "baseline_name")
    best_models = _best_by_checkpoint(
        {name: tabular_global[name] for name in TABULAR_MODELS if name in tabular_global},
        checkpoints,
        "model_name",
    )
    mae_deltas, position_deltas = _comparison_deltas(tabular_global, best_baselines)
    return {
        "status": "complete",
        "strategy": strategy.value,
        "n_events": n_events,
        "n_folds_total": len(folds),
        "n_folds_successful": sum(fold.status == "success" for fold in folds),
        "n_folds_failed": sum(fold.status == "failed" for fold in folds),
        "models": list(tabular_global),
        "tabular_models": [name for name in TABULAR_MODELS if name in tabular_global],
        "baseline_models": list(baseline_global),
        "checkpoints": checkpoints,
        "metrics_by_model_checkpoint": tabular_aggregate,
        "metrics_by_fold_model_checkpoint": tabular_fold,
        "baseline_metrics_by_model_checkpoint": baseline_aggregate,
        "baseline_metrics_by_fold_model_checkpoint": baseline_fold,
        "best_model_by_checkpoint": best_models,
        "best_baseline_by_checkpoint": best_baselines,
        "model_vs_best_baseline_delta_mae": mae_deltas,
        "model_vs_best_baseline_delta_position_error": position_deltas,
        "model_vs_best_baseline_delta_mae_by_fold": _fold_mae_deltas(tabular_fold, baseline_fold),
        "temporal_weighting_policy": temporal_policy.value,
        "temporal_weighting_config": temporal_weighting_config or {},
        "weighted_models_supported": list(supported_weighted_models()),
        "weighted_models_unsupported_if_any": list(unsupported_weighted_models()),
        "training_weight_summary_by_fold": training_weight_summary_by_fold or [],
        "created_at_utc": _utc_now(),
    }


def _fold_metrics(predictions: pd.DataFrame, model_column: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for fold_id, fold_rows in predictions.groupby("fold_id", sort=True):
        result[str(int(fold_id))] = {}
        for model_name, model_rows in fold_rows.groupby(model_column, sort=False):
            result[str(int(fold_id))][str(model_name)] = {}
            for checkpoint, checkpoint_rows in model_rows.groupby("checkpoint", sort=False):
                result[str(int(fold_id))][str(model_name)][str(checkpoint)] = (
                    compute_prediction_metrics(checkpoint_rows)
                )
    return result


def _aggregate_metrics(
    global_metrics: dict[str, Any],
    fold_metrics: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model_name, checkpoint_metrics in global_metrics.items():
        result[model_name] = {}
        for checkpoint, metrics in checkpoint_metrics.items():
            fold_values = [
                fold[model_name][checkpoint]
                for fold in fold_metrics.values()
                if model_name in fold and checkpoint in fold[model_name]
            ]
            result[model_name][checkpoint] = {
                "global": metrics,
                "fold_mean": _metric_summary(fold_values, "mean"),
                "fold_std": _metric_summary(fold_values, "std"),
            }
    return result


def _metric_summary(
    values: list[dict[str, float | None]],
    operation: str,
) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for metric in METRIC_NAMES:
        numeric = [float(value[metric]) for value in values if value.get(metric) is not None]
        if not numeric:
            summary[metric] = None
        elif operation == "mean":
            summary[metric] = sum(numeric) / len(numeric)
        else:
            mean = sum(numeric) / len(numeric)
            variance = sum((value - mean) ** 2 for value in numeric) / len(numeric)
            summary[metric] = math.sqrt(variance)
    return summary


def _best_by_checkpoint(
    metrics: dict[str, Any], checkpoints: list[str], name_key: str
) -> dict[str, dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for checkpoint in checkpoints:
        candidates = [
            (name, values[checkpoint])
            for name, values in metrics.items()
            if checkpoint in values and values[checkpoint].get("mae_gap_sec") is not None
        ]
        if candidates:
            name, values = min(candidates, key=lambda item: float(item[1]["mae_gap_sec"]))
            best[checkpoint] = {
                name_key: name,
                "mae_gap_sec": values.get("mae_gap_sec"),
                "mean_abs_position_error": values.get("mean_abs_position_error"),
            }
    return best


def _comparison_deltas(
    tabular_metrics: dict[str, Any], best_baselines: dict[str, dict[str, object]]
) -> tuple[dict[str, dict[str, float | None]], dict[str, dict[str, float | None]]]:
    mae: dict[str, dict[str, float | None]] = {}
    position: dict[str, dict[str, float | None]] = {}
    for model_name in TABULAR_MODELS:
        if model_name not in tabular_metrics:
            continue
        mae[model_name] = {}
        position[model_name] = {}
        for checkpoint, values in tabular_metrics[model_name].items():
            baseline = best_baselines.get(checkpoint, {})
            mae[model_name][checkpoint] = _delta(
                values.get("mae_gap_sec"), baseline.get("mae_gap_sec")
            )
            position[model_name][checkpoint] = _delta(
                values.get("mean_abs_position_error"),
                baseline.get("mean_abs_position_error"),
            )
    return mae, position


def _fold_mae_deltas(tabular_fold: dict[str, Any], baseline_fold: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for fold_id, model_metrics in tabular_fold.items():
        baseline_metrics = baseline_fold.get(fold_id, {})
        checkpoints = {checkpoint for values in model_metrics.values() for checkpoint in values}
        best_baselines = _best_by_checkpoint(baseline_metrics, sorted(checkpoints), "baseline_name")
        fold_deltas, _ = _comparison_deltas(model_metrics, best_baselines)
        result[fold_id] = fold_deltas
    return result


def _selected_event_keys(dataset: pd.DataFrame, requested: list[str]) -> list[str]:
    selected: list[str] = []
    for event in requested:
        split = create_dataset_split(
            dataset,
            SplitStrategy.event_holdout,
            test_events=[event],
        )
        for event_key in split.metadata["test_events"]:
            if event_key not in selected:
                selected.append(event_key)
    return selected


def _tag_tabular_predictions(
    predictions: pd.DataFrame,
    fold: BacktestFold,
    temporal_policy: TemporalWeightingPolicy,
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["fold_id"] = fold.fold_id
    frame["strategy"] = fold.strategy
    frame["test_event"] = fold.test_event
    frame["prediction_type"] = "tabular"
    frame["temporal_weighting_policy"] = temporal_policy.value
    frame["model_training_weighted"] = temporal_policy is not TemporalWeightingPolicy.uniform
    return frame


def _tag_baseline_predictions(
    predictions: pd.DataFrame,
    fold: BacktestFold,
    temporal_policy: TemporalWeightingPolicy,
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["model_name"] = frame["baseline_name"]
    frame["fold_id"] = fold.fold_id
    frame["strategy"] = fold.strategy
    frame["test_event"] = fold.test_event
    frame["prediction_type"] = "baseline"
    frame["temporal_weighting_policy"] = temporal_policy.value
    frame["model_training_weighted"] = False
    return frame


def _completed_fold(fold: BacktestFold, status: str, error: str | None) -> BacktestFold:
    return BacktestFold(
        fold_id=fold.fold_id,
        strategy=fold.strategy,
        test_event=fold.test_event,
        train_events=fold.train_events,
        train_rows=fold.train_rows,
        test_rows=fold.test_rows,
        status=status,
        error_message=error,
    )


def _backtest_paths(metrics_dir: Path, strategy: BacktestStrategy) -> dict[str, Path]:
    prefix = strategy.value
    return {
        "metrics": metrics_dir / f"{prefix}_metrics.json",
        "predictions": metrics_dir / f"{prefix}_predictions.parquet",
        "folds": metrics_dir / f"{prefix}_folds.json",
    }


def _write_temporal_snapshots(
    paths: dict[str, Path],
    policy: TemporalWeightingPolicy,
    payload: dict[str, object],
    *,
    predictions_path: Path | None,
) -> None:
    metrics_stem = temporal_artifact_stem(
        paths["metrics"].stem.removesuffix("_metrics"),
        policy,
    )
    folds_stem = temporal_artifact_stem(
        paths["folds"].stem.removesuffix("_folds"),
        policy,
    )
    predictions_stem = temporal_artifact_stem(
        paths["predictions"].stem.removesuffix("_predictions"),
        policy,
    )
    metrics_snapshot = paths["metrics"].with_name(f"{metrics_stem}_metrics.json")
    folds_snapshot = paths["folds"].with_name(f"{folds_stem}_folds.json")
    predictions_snapshot = paths["predictions"].with_name(f"{predictions_stem}_predictions.parquet")
    _write_json(metrics_snapshot, payload)
    if paths["folds"].is_file():
        folds_snapshot.write_bytes(paths["folds"].read_bytes())
    if predictions_path is not None and predictions_path.is_file():
        predictions_snapshot.write_bytes(predictions_path.read_bytes())


def _event_key_series(dataset: pd.DataFrame) -> pd.Series:
    return dataset["season"].astype(str) + "/" + dataset["event_slug"].astype(str)


def _resolve_dataset_path(config: DataConfig, dataset_path: Path | None) -> Path:
    path = dataset_path or build_combined_dataset_path(config.modeling_output_dir)
    if not path.is_absolute():
        path = config.project_root / path
    if not path.is_file():
        raise FileNotFoundError(f"Modeling dataset does not exist: {path}")
    return path


def _skipped_payload(strategy: BacktestStrategy, n_events: int, reason: str) -> dict[str, object]:
    return {
        "status": "skipped",
        "strategy": strategy.value,
        "reason": reason,
        "n_events": n_events,
        "n_folds_total": 0,
        "n_folds_successful": 0,
        "n_folds_failed": 0,
        "created_at_utc": _utc_now(),
    }


def _failed_payload(
    strategy: BacktestStrategy, n_events: int, folds: list[BacktestFold]
) -> dict[str, object]:
    return {
        "status": "failed",
        "strategy": strategy.value,
        "n_events": n_events,
        "n_folds_total": len(folds),
        "n_folds_successful": 0,
        "n_folds_failed": len(folds),
        "created_at_utc": _utc_now(),
    }


def _delta(model_value: object, baseline_value: object) -> float | None:
    if model_value is None or baseline_value is None:
        return None
    return float(model_value) - float(baseline_value)


def _concise_error(exc: Exception) -> str:
    message = " ".join(str(exc).split()) or "No error details were provided"
    return f"{type(exc).__name__}: {message}"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _historical_settings(config: FeatureConfig | None) -> HistoricalFeatureSettings:
    if config is None or config.historical_features is None:
        return HistoricalFeatureSettings()
    historical = config.historical_features
    return HistoricalFeatureSettings(
        rolling_windows=historical.rolling_windows,
        min_periods=historical.min_periods,
    )


def _robust_baseline_threshold(config: FeatureConfig | None) -> float:
    if config is None or config.baselines is None:
        return 3.0
    return config.baselines.robust_extreme_gap_to_session_best_sec
