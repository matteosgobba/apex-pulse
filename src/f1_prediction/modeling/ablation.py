"""Identical-fold feature-ablation backtesting for simple tabular models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig, FeatureConfig, ModelConfig
from f1_prediction.data.season_builder import build_combined_dataset_path
from f1_prediction.features.historical_features import (
    HistoricalFeatureSettings,
    add_historical_features,
)
from f1_prediction.modeling.backtest_tabular import (
    BacktestFold,
    BacktestStrategy,
    build_backtest_folds,
)
from f1_prediction.modeling.baselines import generate_baseline_predictions
from f1_prediction.modeling.feature_groups import DEFAULT_ABLATION_GROUPS, get_feature_groups
from f1_prediction.modeling.metrics import compute_baseline_metrics
from f1_prediction.modeling.splits import ordered_event_keys
from f1_prediction.modeling.train_tabular import (
    DEFAULT_MODEL_CONFIG,
    fit_and_predict,
    metrics_by_model_checkpoint,
)
from f1_prediction.utils.paths import ensure_directory

ABLATION_MODELS: tuple[str, ...] = ("ridge", "random_forest")


@dataclass(frozen=True)
class AblationBacktestSummary:
    """Paths and counts produced by feature-ablation backtesting."""

    status: str
    strategy: str
    n_events: int
    n_folds_total: int
    n_folds_successful: int
    n_folds_failed: int
    feature_groups: tuple[str, ...]
    prediction_rows: int
    metrics_path: Path
    predictions_path: Path | None
    feature_groups_path: Path


def run_ablation_backtest(
    config: DataConfig,
    *,
    strategy: BacktestStrategy | str = BacktestStrategy.walk_forward,
    dataset_path: Path | None = None,
    feature_group_names: list[str] | None = None,
    min_events: int = 10,
    min_train_events: int = 5,
    fail_fast: bool = False,
    model_config: ModelConfig | None = None,
    feature_config: FeatureConfig | None = None,
) -> AblationBacktestSummary:
    """Evaluate feature groups on one shared event-safe fold collection."""
    strategy = BacktestStrategy(strategy)
    source_path = _resolve_dataset_path(config, dataset_path)
    dataset = pd.read_parquet(source_path).reset_index(drop=True)
    event_keys = ordered_event_keys(dataset)
    selected_names = tuple(feature_group_names or DEFAULT_ABLATION_GROUPS)
    all_groups = get_feature_groups(dataset)
    unknown = [name for name in selected_names if name not in all_groups]
    if unknown:
        raise ValueError(f"Unknown feature groups: {', '.join(unknown)}")
    selected_groups = {name: all_groups[name] for name in selected_names}
    paths = _output_paths(config.metrics_output_dir)
    ensure_directory(config.metrics_output_dir)

    if len(event_keys) < min_events:
        reason = f"Dataset has {len(event_keys)} unique events; at least {min_events} are required"
        payload = _skipped_payload(strategy, len(event_keys), selected_names, reason)
        _write_json(paths["metrics"], payload)
        _write_json(
            paths["groups"],
            _feature_group_payload(strategy, selected_groups, (), status="skipped"),
        )
        paths["predictions"].unlink(missing_ok=True)
        return AblationBacktestSummary(
            status="skipped",
            strategy=strategy.value,
            n_events=len(event_keys),
            n_folds_total=0,
            n_folds_successful=0,
            n_folds_failed=0,
            feature_groups=selected_names,
            prediction_rows=0,
            metrics_path=paths["metrics"],
            predictions_path=None,
            feature_groups_path=paths["groups"],
        )

    folds = build_backtest_folds(
        dataset,
        strategy,
        min_train_events=min_train_events,
    )
    if not folds:
        raise ValueError("No evaluable folds were created for ablation backtesting")

    settings = model_config or DEFAULT_MODEL_CONFIG
    historical_settings = _historical_settings(feature_config)
    robust_threshold = _robust_threshold(feature_config)
    row_keys = _event_key_series(dataset)
    predictions: list[pd.DataFrame] = []
    baseline_frames: list[pd.DataFrame] = []
    completed_folds: list[BacktestFold] = []

    for fold in folds:
        fold_predictions: list[pd.DataFrame] = []
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
            for group_name, columns in selected_groups.items():
                group_predictions, _ = fit_and_predict(
                    train,
                    test,
                    model_config=settings,
                    candidate_features=columns,
                )
                group_predictions = group_predictions[
                    group_predictions["model_name"].isin(ABLATION_MODELS)
                ].copy()
                group_predictions["feature_group"] = group_name
                fold_predictions.append(_tag_predictions(group_predictions, fold, "tabular"))
            baseline = generate_baseline_predictions(
                test,
                robust_extreme_threshold_sec=robust_threshold,
            )
            baseline["model_name"] = baseline["baseline_name"]
            baseline["feature_group"] = "baseline"
            predictions.extend(fold_predictions)
            baseline_frames.append(_tag_predictions(baseline, fold, "baseline"))
            completed_folds.append(_complete_fold(fold, "success", None))
        except Exception as exc:
            completed_folds.append(_complete_fold(fold, "failed", _concise_error(exc)))
            if fail_fast:
                break

    successful = [fold for fold in completed_folds if fold.status == "success"]
    failed = [fold for fold in completed_folds if fold.status == "failed"]
    if predictions:
        tabular_predictions = pd.concat(predictions, ignore_index=True, sort=False)
        baseline_predictions = pd.concat(baseline_frames, ignore_index=True, sort=False)
        combined = pd.concat(
            [tabular_predictions, baseline_predictions], ignore_index=True, sort=False
        )
        combined.to_parquet(paths["predictions"], engine="pyarrow", index=False)
        payload = build_ablation_metrics_payload(
            strategy,
            len(event_keys),
            completed_folds,
            selected_names,
            tabular_predictions,
            baseline_predictions,
        )
        status = "complete" if not failed else "partial"
        payload["status"] = status
        predictions_path: Path | None = paths["predictions"]
    else:
        status = "failed"
        payload = {
            "status": status,
            "strategy": strategy.value,
            "n_events": len(event_keys),
            "n_folds_successful": 0,
            "n_folds_failed": len(failed),
            "feature_groups": list(selected_names),
            "created_at_utc": _utc_now(),
        }
        paths["predictions"].unlink(missing_ok=True)
        predictions_path = None
    _write_json(paths["metrics"], payload)
    _write_json(
        paths["groups"],
        _feature_group_payload(strategy, selected_groups, completed_folds, status=status),
    )
    return AblationBacktestSummary(
        status=status,
        strategy=strategy.value,
        n_events=len(event_keys),
        n_folds_total=len(completed_folds),
        n_folds_successful=len(successful),
        n_folds_failed=len(failed),
        feature_groups=selected_names,
        prediction_rows=(
            len(tabular_predictions) + len(baseline_predictions) if predictions else 0
        ),
        metrics_path=paths["metrics"],
        predictions_path=predictions_path,
        feature_groups_path=paths["groups"],
    )


def build_ablation_metrics_payload(
    strategy: BacktestStrategy | str,
    n_events: int,
    folds: list[BacktestFold],
    feature_groups: tuple[str, ...],
    tabular_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
) -> dict[str, object]:
    """Build metrics and best-group comparisons from shared-fold predictions."""
    strategy = BacktestStrategy(strategy)
    grouped_metrics: dict[str, dict[str, Any]] = {}
    for group_name, rows in tabular_predictions.groupby("feature_group", sort=False):
        grouped_metrics[str(group_name)] = metrics_by_model_checkpoint(rows)
    baseline_metrics = compute_baseline_metrics(baseline_predictions)
    checkpoints = tabular_predictions["checkpoint"].drop_duplicates().astype(str).tolist()
    best_baselines = _best_baselines(baseline_metrics, checkpoints)
    best_by_model = _best_feature_groups(grouped_metrics, checkpoints)
    best_overall = _best_overall(grouped_metrics, checkpoints)
    deltas = _all_deltas(grouped_metrics, best_baselines)
    return {
        "status": "complete",
        "strategy": strategy.value,
        "n_events": n_events,
        "n_folds_total": len(folds),
        "n_folds_successful": sum(fold.status == "success" for fold in folds),
        "n_folds_failed": sum(fold.status == "failed" for fold in folds),
        "feature_groups": list(feature_groups),
        "models": list(ABLATION_MODELS),
        "checkpoints": checkpoints,
        "metrics_by_feature_group_model_checkpoint": grouped_metrics,
        "best_feature_group_by_model_checkpoint": best_by_model,
        "best_overall_by_checkpoint": best_overall,
        "best_baseline_by_checkpoint": best_baselines,
        "model_vs_baseline_delta_mae": deltas,
        "created_at_utc": _utc_now(),
    }


def _best_feature_groups(
    metrics: dict[str, dict[str, Any]], checkpoints: list[str]
) -> dict[str, dict[str, dict[str, object]]]:
    result: dict[str, dict[str, dict[str, object]]] = {model: {} for model in ABLATION_MODELS}
    for model in ABLATION_MODELS:
        for checkpoint in checkpoints:
            candidates = [
                (group, values[model][checkpoint])
                for group, values in metrics.items()
                if model in values
                and checkpoint in values[model]
                and values[model][checkpoint].get("mae_gap_sec") is not None
            ]
            if candidates:
                group, values = min(candidates, key=lambda item: float(item[1]["mae_gap_sec"]))
                result[model][checkpoint] = {
                    "feature_group": group,
                    "mae_gap_sec": values.get("mae_gap_sec"),
                    "mean_abs_position_error": values.get("mean_abs_position_error"),
                }
    return result


def _best_overall(
    metrics: dict[str, dict[str, Any]], checkpoints: list[str]
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for checkpoint in checkpoints:
        candidates = [
            (group, model, model_values[checkpoint])
            for group, group_values in metrics.items()
            for model, model_values in group_values.items()
            if checkpoint in model_values
            and model_values[checkpoint].get("mae_gap_sec") is not None
        ]
        if candidates:
            group, model, values = min(candidates, key=lambda item: float(item[2]["mae_gap_sec"]))
            result[checkpoint] = {
                "feature_group": group,
                "model_name": model,
                "mae_gap_sec": values.get("mae_gap_sec"),
                "mean_abs_position_error": values.get("mean_abs_position_error"),
            }
    return result


def _best_baselines(
    metrics: dict[str, dict[str, Any]], checkpoints: list[str]
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for checkpoint in checkpoints:
        candidates = [
            (name, values[checkpoint])
            for name, values in metrics.items()
            if checkpoint in values and values[checkpoint].get("mae_gap_sec") is not None
        ]
        if candidates:
            name, values = min(candidates, key=lambda item: float(item[1]["mae_gap_sec"]))
            result[checkpoint] = {
                "baseline_name": name,
                "mae_gap_sec": values.get("mae_gap_sec"),
                "mean_abs_position_error": values.get("mean_abs_position_error"),
            }
    return result


def _all_deltas(
    metrics: dict[str, dict[str, Any]], baselines: dict[str, dict[str, object]]
) -> dict[str, dict[str, dict[str, float | None]]]:
    result: dict[str, dict[str, dict[str, float | None]]] = {}
    for group, group_values in metrics.items():
        result[group] = {}
        for model, checkpoint_values in group_values.items():
            result[group][model] = {}
            for checkpoint, values in checkpoint_values.items():
                baseline = baselines.get(checkpoint, {}).get("mae_gap_sec")
                model_mae = values.get("mae_gap_sec")
                result[group][model][checkpoint] = (
                    float(model_mae) - float(baseline)
                    if model_mae is not None and baseline is not None
                    else None
                )
    return result


def _tag_predictions(
    predictions: pd.DataFrame, fold: BacktestFold, prediction_type: str
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["fold_id"] = fold.fold_id
    frame["strategy"] = fold.strategy
    frame["test_event"] = fold.test_event
    frame["prediction_type"] = prediction_type
    return frame


def _complete_fold(fold: BacktestFold, status: str, error: str | None) -> BacktestFold:
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


def _feature_group_payload(
    strategy: BacktestStrategy,
    groups: dict[str, list[str]],
    folds: tuple[BacktestFold, ...] | list[BacktestFold],
    *,
    status: str,
) -> dict[str, object]:
    return {
        "status": status,
        "strategy": strategy.value,
        "feature_groups": {
            name: {"n_features": len(columns), "columns": columns}
            for name, columns in groups.items()
        },
        "folds": [asdict(fold) for fold in folds],
        "created_at_utc": _utc_now(),
    }


def _skipped_payload(
    strategy: BacktestStrategy,
    n_events: int,
    feature_groups: tuple[str, ...],
    reason: str,
) -> dict[str, object]:
    return {
        "status": "skipped",
        "strategy": strategy.value,
        "reason": reason,
        "n_events": n_events,
        "n_folds_total": 0,
        "n_folds_successful": 0,
        "n_folds_failed": 0,
        "feature_groups": list(feature_groups),
        "created_at_utc": _utc_now(),
    }


def _event_key_series(dataset: pd.DataFrame) -> pd.Series:
    return dataset["season"].astype(str) + "/" + dataset["event_slug"].astype(str)


def _historical_settings(config: FeatureConfig | None) -> HistoricalFeatureSettings:
    if config is None or config.historical_features is None:
        return HistoricalFeatureSettings()
    settings = config.historical_features
    return HistoricalFeatureSettings(settings.rolling_windows, settings.min_periods)


def _robust_threshold(config: FeatureConfig | None) -> float:
    if config is None or config.baselines is None:
        return 3.0
    return config.baselines.robust_extreme_gap_to_session_best_sec


def _resolve_dataset_path(config: DataConfig, path: Path | None) -> Path:
    resolved = path or build_combined_dataset_path(config.modeling_output_dir)
    if not resolved.is_absolute():
        resolved = config.project_root / resolved
    if not resolved.is_file():
        raise FileNotFoundError(f"Modeling dataset does not exist: {resolved}")
    return resolved


def _output_paths(metrics_dir: Path) -> dict[str, Path]:
    return {
        "metrics": metrics_dir / "ablation_metrics.json",
        "predictions": metrics_dir / "ablation_predictions.parquet",
        "groups": metrics_dir / "ablation_feature_groups.json",
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


def _concise_error(exc: Exception) -> str:
    message = " ".join(str(exc).split()) or "No error details were provided"
    return f"{type(exc).__name__}: {message}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
