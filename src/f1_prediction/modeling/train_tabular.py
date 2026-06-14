"""Training orchestration for the first simple tabular models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from f1_prediction.config import DataConfig, ModelConfig, RandomForestConfig
from f1_prediction.data.season_builder import build_combined_dataset_path
from f1_prediction.features.qualifying_targets import TARGET_COLUMNS
from f1_prediction.modeling.baselines import generate_baseline_predictions
from f1_prediction.modeling.metrics import compute_baseline_metrics, compute_prediction_metrics
from f1_prediction.modeling.splits import (
    DatasetSplit,
    SplitStrategy,
    create_dataset_split,
    latest_event_holdout,
)
from f1_prediction.modeling.tabular import (
    TARGET_COLUMN,
    build_regressors,
    rank_gap_predictions,
    usable_checkpoint_features,
)
from f1_prediction.utils.paths import ensure_directory

PREDICTION_COLUMNS: tuple[str, ...] = (
    "season",
    "event",
    "event_slug",
    "checkpoint",
    "driver",
    "team",
    *TARGET_COLUMNS,
)
DEFAULT_MODEL_CONFIG = ModelConfig(
    min_events=5,
    random_state=42,
    ridge_alpha=1.0,
    random_forest=RandomForestConfig(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
    ),
)


@dataclass(frozen=True)
class TabularTrainingSummary:
    """Outcome and artifact paths for tabular training."""

    status: str
    reason: str | None
    n_events: int
    train_events: int
    test_events: int
    prediction_rows: int
    models: tuple[str, ...]
    metrics_path: Path
    predictions_path: Path | None
    ridge_model_path: Path | None
    random_forest_model_path: Path | None


def train_tabular_models(
    config: DataConfig,
    dataset_path: Path | None = None,
    *,
    test_season: int | None = None,
    test_events: list[str] | None = None,
    min_events: int | None = None,
    model_config: ModelConfig | None = None,
) -> TabularTrainingSummary:
    """Train checkpoint-specific simple models with an event-safe holdout."""
    settings = model_config or DEFAULT_MODEL_CONFIG
    required_events = min_events if min_events is not None else settings.min_events
    if required_events < 2:
        raise ValueError("min_events must be at least 2")
    if test_season is not None and test_events:
        raise ValueError("Choose either --test-season or --test-events, not both")
    source_path = _resolve_dataset_path(config, dataset_path)
    dataset = pd.read_parquet(source_path).reset_index(drop=True)
    n_events = dataset[["season", "event_slug"]].drop_duplicates().shape[0]
    metrics_path = config.metrics_output_dir / "tabular_model_metrics.json"
    ensure_directory(metrics_path.parent)

    if n_events < required_events:
        reason = f"Dataset has {n_events} unique events; at least {required_events} are required"
        _write_json(
            metrics_path,
            {
                "status": "skipped",
                "reason": reason,
                "n_events": n_events,
                "min_events": required_events,
                "dataset_path": _portable_path(source_path, config.project_root),
                "created_at_utc": _utc_now(),
            },
        )
        return TabularTrainingSummary(
            status="skipped",
            reason=reason,
            n_events=n_events,
            train_events=0,
            test_events=0,
            prediction_rows=0,
            models=(),
            metrics_path=metrics_path,
            predictions_path=None,
            ridge_model_path=None,
            random_forest_model_path=None,
        )

    split = _training_split(dataset, test_season=test_season, test_events=test_events)
    train = dataset.loc[list(split.train_indices)].copy()
    test = dataset.loc[list(split.test_indices)].copy()
    if train.empty:
        raise ValueError("The selected holdout leaves no events available for training")
    predictions, fitted_models = fit_and_predict(train, test, model_config=settings)
    metrics = metrics_by_model_checkpoint(predictions)
    predictions_path = config.metrics_output_dir / "tabular_model_predictions.parquet"
    predictions.to_parquet(predictions_path, engine="pyarrow", index=False)

    models_dir = ensure_directory(config.project_root / "models")
    ridge_model_path = models_dir / "ridge_gap_model.joblib"
    random_forest_model_path = models_dir / "random_forest_gap_model.joblib"
    joblib.dump(fitted_models["ridge"], ridge_model_path)
    joblib.dump(fitted_models["random_forest"], random_forest_model_path)

    payload: dict[str, Any] = {
        "status": "trained",
        "target": TARGET_COLUMN,
        "n_events": n_events,
        "min_events": required_events,
        "split": split.metadata,
        "models": metrics,
        "feature_columns_by_checkpoint": {
            checkpoint: bundle["feature_columns"]
            for checkpoint, bundle in fitted_models["ridge"].items()
        },
        "dataset_path": _portable_path(source_path, config.project_root),
        "predictions_path": _portable_path(predictions_path, config.project_root),
        "created_at_utc": _utc_now(),
    }
    baseline_path = config.metrics_output_dir / "baseline_metrics.json"
    if baseline_path.is_file():
        payload.update(_baseline_comparison(baseline_path, test, metrics))
    _write_json(metrics_path, payload)

    return TabularTrainingSummary(
        status="trained",
        reason=None,
        n_events=n_events,
        train_events=len(split.metadata["train_events"]),
        test_events=len(split.metadata["test_events"]),
        prediction_rows=len(predictions),
        models=tuple(metrics),
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        ridge_model_path=ridge_model_path,
        random_forest_model_path=random_forest_model_path,
    )


def _training_split(
    dataset: pd.DataFrame,
    *,
    test_season: int | None,
    test_events: list[str] | None,
) -> DatasetSplit:
    if test_season is not None:
        return create_dataset_split(
            dataset,
            SplitStrategy.season_holdout,
            test_seasons=[test_season],
        )
    if test_events:
        return create_dataset_split(
            dataset,
            SplitStrategy.event_holdout,
            test_events=test_events,
        )
    return latest_event_holdout(dataset)


def fit_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    model_config: ModelConfig,
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, object]]]]:
    prediction_frames: list[pd.DataFrame] = []
    fitted: dict[str, dict[str, dict[str, object]]] = {"ridge": {}, "random_forest": {}}
    for checkpoint in test["checkpoint"].dropna().astype(str).drop_duplicates():
        train_rows = train[train["checkpoint"].eq(checkpoint)].dropna(subset=[TARGET_COLUMN])
        test_rows = test[test["checkpoint"].eq(checkpoint)].dropna(subset=[TARGET_COLUMN])
        if train_rows.empty or test_rows.empty:
            continue
        features = usable_checkpoint_features(train_rows, checkpoint)
        if not features:
            raise ValueError(f"No usable numeric features for {checkpoint}")

        target_mean = float(train_rows[TARGET_COLUMN].mean())
        target_median = float(train_rows[TARGET_COLUMN].median())
        prediction_frames.append(_constant_predictions(test_rows, "mean_target", target_mean))
        prediction_frames.append(_constant_predictions(test_rows, "median_target", target_median))

        for model_name, estimator in build_regressors(model_config).items():
            estimator.fit(train_rows[features], train_rows[TARGET_COLUMN])
            frame = _prediction_frame(test_rows, model_name)
            frame["predicted_quali_gap_to_pole_sec"] = estimator.predict(test_rows[features])
            prediction_frames.append(_finish_predictions(frame))
            fitted[model_name][checkpoint] = {
                "estimator": estimator,
                "feature_columns": features,
            }
    if not prediction_frames:
        raise ValueError("No checkpoint had enough target data for training and evaluation")
    return pd.concat(prediction_frames, ignore_index=True), fitted


def _constant_predictions(test_rows: pd.DataFrame, model_name: str, value: float) -> pd.DataFrame:
    frame = _prediction_frame(test_rows, model_name)
    frame["predicted_quali_gap_to_pole_sec"] = value
    return _finish_predictions(frame)


def _prediction_frame(test_rows: pd.DataFrame, model_name: str) -> pd.DataFrame:
    columns = [column for column in PREDICTION_COLUMNS if column in test_rows]
    frame = test_rows.loc[:, columns].copy()
    frame["model_name"] = model_name
    return frame


def _finish_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    frame["predicted_quali_position"] = rank_gap_predictions(frame)
    frame["predicted_reached_q3"] = frame["predicted_quali_position"].le(10).astype("int8")
    return frame


def metrics_by_model_checkpoint(
    predictions: pd.DataFrame,
) -> dict[str, dict[str, dict[str, float | None]]]:
    metrics: dict[str, dict[str, dict[str, float | None]]] = {}
    for model_name, model_rows in predictions.groupby("model_name", sort=False):
        metrics[str(model_name)] = {}
        for checkpoint, checkpoint_rows in model_rows.groupby("checkpoint", sort=False):
            metrics[str(model_name)][str(checkpoint)] = compute_prediction_metrics(checkpoint_rows)
    return metrics


def _baseline_comparison(
    baseline_path: Path,
    test: pd.DataFrame,
    model_metrics: dict[str, dict[str, dict[str, float | None]]],
) -> dict[str, object]:
    with baseline_path.open(encoding="utf-8") as baseline_file:
        available_metrics = json.load(baseline_file)
    holdout_metrics = compute_baseline_metrics(generate_baseline_predictions(test))
    best_by_checkpoint: dict[str, dict[str, object]] = {}
    for checkpoint in test["checkpoint"].dropna().astype(str).drop_duplicates():
        candidates = [
            (baseline, values[checkpoint].get("mae_gap_sec"))
            for baseline, values in holdout_metrics.items()
            if checkpoint in values and values[checkpoint].get("mae_gap_sec") is not None
        ]
        if candidates:
            baseline_name, mae = min(candidates, key=lambda item: float(item[1]))
            best_by_checkpoint[checkpoint] = {
                "baseline_name": baseline_name,
                "mae_gap_sec": mae,
                "mean_abs_position_error": holdout_metrics[baseline_name][checkpoint].get(
                    "mean_abs_position_error"
                ),
            }
    deltas: dict[str, dict[str, float | None]] = {}
    for model_name, checkpoint_metrics in model_metrics.items():
        deltas[model_name] = {}
        for checkpoint, values in checkpoint_metrics.items():
            model_mae = values.get("mae_gap_sec")
            baseline = best_by_checkpoint.get(checkpoint, {}).get("mae_gap_sec")
            deltas[model_name][checkpoint] = (
                float(model_mae) - float(baseline)
                if model_mae is not None and baseline is not None
                else None
            )
    return {
        "baseline_comparison_scope": "same_test_holdout",
        "baseline_metrics_file_models": list(available_metrics),
        "best_baseline_by_checkpoint": best_by_checkpoint,
        "model_vs_best_baseline_delta_mae": deltas,
    }


def _resolve_dataset_path(config: DataConfig, dataset_path: Path | None) -> Path:
    path = dataset_path or build_combined_dataset_path(config.modeling_output_dir)
    if not path.is_absolute():
        path = config.project_root / path
    if not path.is_file():
        raise FileNotFoundError(f"Modeling dataset does not exist: {path}")
    return path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
