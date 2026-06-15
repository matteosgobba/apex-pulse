"""Leakage-safe repeated backtesting for gradient-boosted models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from f1_prediction.config import DataConfig, FeatureConfig, ModelConfig
from f1_prediction.data.season_builder import build_combined_dataset_path
from f1_prediction.features.historical_features import add_historical_features
from f1_prediction.modeling.backtest_tabular import (
    BacktestFold,
    BacktestStrategy,
    _aggregate_metrics,
    _best_by_checkpoint,
    _fold_metrics,
    _historical_settings,
    _robust_baseline_threshold,
    build_backtest_folds,
)
from f1_prediction.modeling.baselines import generate_baseline_predictions
from f1_prediction.modeling.feature_groups import get_feature_groups
from f1_prediction.modeling.gradient_boosting import (
    MODEL_NAME,
    build_hist_gradient_boosting_regressor,
)
from f1_prediction.modeling.metrics import compute_baseline_metrics
from f1_prediction.modeling.splits import ordered_event_keys
from f1_prediction.modeling.tabular import TARGET_COLUMN, rank_gap_predictions
from f1_prediction.modeling.train_tabular import PREDICTION_COLUMNS, metrics_by_model_checkpoint
from f1_prediction.utils.paths import ensure_directory

CHECKPOINTS: tuple[str, ...] = ("after_fp1", "after_fp2", "after_fp3")
FEATURE_POLICIES: tuple[str, ...] = ("checkpoint_best", "all_features")


@dataclass(frozen=True)
class BoostedBacktestSummary:
    """Paths and counts produced by boosted-model backtesting."""

    status: str
    strategy: str
    feature_policy: str
    n_events: int
    n_folds_total: int
    n_folds_successful: int
    n_folds_failed: int
    prediction_rows: int
    metrics_path: Path
    predictions_path: Path | None
    folds_path: Path


def resolve_feature_groups_by_checkpoint(
    dataset: pd.DataFrame,
    model_config: ModelConfig,
    *,
    feature_policy: str,
    explicit_feature_group: str | None = None,
) -> dict[str, str]:
    """Resolve one registered safe feature group for each checkpoint."""
    available = get_feature_groups(dataset)
    if explicit_feature_group is not None:
        if explicit_feature_group not in available:
            raise ValueError(f"Unknown feature group: {explicit_feature_group}")
        return {checkpoint: explicit_feature_group for checkpoint in CHECKPOINTS}
    if feature_policy == "all_features":
        return {checkpoint: "all_features" for checkpoint in CHECKPOINTS}
    if feature_policy != "checkpoint_best":
        choices = ", ".join(FEATURE_POLICIES)
        raise ValueError(
            f"Unknown feature policy '{feature_policy}'. Available policies: {choices}"
        )
    policy = model_config.feature_group_policy
    selected = {
        "after_fp1": policy.after_fp1,
        "after_fp2": policy.after_fp2,
        "after_fp3": policy.after_fp3,
    }
    unknown = sorted({name for name in selected.values() if name not in available})
    if unknown:
        raise ValueError(f"Unknown configured feature groups: {', '.join(unknown)}")
    return selected


def fit_and_predict_boosted(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    model_config: ModelConfig,
    feature_groups_by_checkpoint: dict[str, str],
    feature_policy: str,
) -> pd.DataFrame:
    """Fit one boosted regressor per checkpoint and return ranked predictions."""
    if not model_config.hist_gradient_boosting.enabled:
        raise ValueError("hist_gradient_boosting is disabled in model configuration")
    frames: list[pd.DataFrame] = []
    for checkpoint in CHECKPOINTS:
        train_rows = train[train["checkpoint"].eq(checkpoint)].dropna(subset=[TARGET_COLUMN])
        test_rows = test[test["checkpoint"].eq(checkpoint)].dropna(subset=[TARGET_COLUMN])
        if train_rows.empty or test_rows.empty:
            continue
        group_name = feature_groups_by_checkpoint[checkpoint]
        features = get_checkpoint_feature_columns(train_rows, checkpoint, group_name)
        if not features:
            raise ValueError(f"No usable {group_name} features for {checkpoint}")
        estimator = build_hist_gradient_boosting_regressor(model_config.hist_gradient_boosting)
        train_features = _numeric_matrix(train_rows, features)
        test_features = _numeric_matrix(test_rows, features)
        estimator.fit(train_features, train_rows[TARGET_COLUMN])
        columns = [column for column in PREDICTION_COLUMNS if column in test_rows]
        frame = test_rows.loc[:, columns].copy()
        frame["model_name"] = MODEL_NAME
        frame["feature_policy"] = feature_policy
        frame["feature_group"] = group_name
        frame["predicted_quali_gap_to_pole_sec"] = estimator.predict(test_features)
        frame["predicted_quali_position"] = rank_gap_predictions(frame)
        frame["predicted_reached_q3"] = frame["predicted_quali_position"].le(10).astype("int8")
        frames.append(frame)
    if not frames:
        raise ValueError("No checkpoint had enough target data for boosted evaluation")
    return pd.concat(frames, ignore_index=True)


def get_checkpoint_feature_columns(
    dataset: pd.DataFrame,
    checkpoint: str,
    feature_group: str,
) -> list[str]:
    """Return usable group columns after applying checkpoint leakage rules."""
    groups = get_feature_groups(dataset)
    if feature_group not in groups:
        raise ValueError(f"Unknown feature group: {feature_group}")
    allowed = set(_checkpoint_safe_features(dataset, checkpoint))
    return [
        column
        for column in groups[feature_group]
        if column in allowed and dataset[column].notna().any()
    ]


def run_boosted_backtest(
    config: DataConfig,
    *,
    strategy: BacktestStrategy | str = BacktestStrategy.walk_forward,
    feature_policy: str = "checkpoint_best",
    explicit_feature_group: str | None = None,
    dataset_path: Path | None = None,
    test_events: list[str] | None = None,
    min_events: int = 10,
    min_train_events: int = 5,
    fail_fast: bool = False,
    model_config: ModelConfig,
    feature_config: FeatureConfig | None = None,
) -> BoostedBacktestSummary:
    """Backtest boosted models and robust baselines on identical event folds."""
    strategy = BacktestStrategy(strategy)
    source_path = _resolve_dataset_path(config, dataset_path)
    dataset = pd.read_parquet(source_path).reset_index(drop=True)
    event_keys = ordered_event_keys(dataset)
    policy_label = explicit_feature_group or feature_policy
    selected_groups = resolve_feature_groups_by_checkpoint(
        dataset,
        model_config,
        feature_policy=feature_policy,
        explicit_feature_group=explicit_feature_group,
    )
    paths = _output_paths(config.metrics_output_dir)
    ensure_directory(config.metrics_output_dir)
    if len(event_keys) < min_events:
        reason = f"Dataset has {len(event_keys)} unique events; at least {min_events} are required"
        payload = _skipped_payload(strategy, policy_label, len(event_keys), selected_groups, reason)
        _write_json(paths["metrics"], payload)
        _write_json(paths["folds"], {"status": "skipped", "strategy": strategy.value, "folds": []})
        paths["predictions"].unlink(missing_ok=True)
        return BoostedBacktestSummary(
            status="skipped",
            strategy=strategy.value,
            feature_policy=policy_label,
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
        raise ValueError("No evaluable folds were created for boosted backtesting")

    row_keys = _event_key_series(dataset)
    boosted_frames: list[pd.DataFrame] = []
    baseline_frames: list[pd.DataFrame] = []
    completed_folds: list[BacktestFold] = []
    for fold in folds:
        try:
            fold_scope = dataset[row_keys.isin([*fold.train_events, fold.test_event])].copy()
            fold_scope = add_historical_features(
                fold_scope,
                _historical_settings(feature_config),
                excluded_target_events={fold.test_event},
            )
            fold_keys = _event_key_series(fold_scope)
            train = fold_scope[fold_keys.isin(fold.train_events)].copy()
            test = fold_scope[fold_keys.eq(fold.test_event)].copy()
            predictions = fit_and_predict_boosted(
                train,
                test,
                model_config=model_config,
                feature_groups_by_checkpoint=selected_groups,
                feature_policy=policy_label,
            )
            baseline = generate_baseline_predictions(
                test,
                robust_extreme_threshold_sec=_robust_baseline_threshold(feature_config),
            )
            boosted_frames.append(_tag_predictions(predictions, fold, "boosted", policy_label))
            baseline_frames.append(_tag_baselines(baseline, fold))
            completed_folds.append(_complete_fold(fold, "success", None))
        except Exception as exc:
            completed_folds.append(_complete_fold(fold, "failed", _concise_error(exc)))
            if fail_fast:
                break

    successful = [fold for fold in completed_folds if fold.status == "success"]
    failed = [fold for fold in completed_folds if fold.status == "failed"]
    if boosted_frames:
        boosted_predictions = pd.concat(boosted_frames, ignore_index=True, sort=False)
        baseline_predictions = pd.concat(baseline_frames, ignore_index=True, sort=False)
        combined_predictions = pd.concat(
            [boosted_predictions, baseline_predictions], ignore_index=True, sort=False
        )
        combined_predictions.to_parquet(paths["predictions"], engine="pyarrow", index=False)
        payload = build_boosted_metrics_payload(
            strategy,
            policy_label,
            len(event_keys),
            completed_folds,
            selected_groups,
            boosted_predictions,
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
            "feature_policy": policy_label,
            "n_events": len(event_keys),
            "n_folds_total": len(completed_folds),
            "n_folds_successful": 0,
            "n_folds_failed": len(failed),
            "models": [],
            "feature_groups_by_checkpoint": selected_groups,
            "created_at_utc": _utc_now(),
        }
        paths["predictions"].unlink(missing_ok=True)
        predictions_path = None
    _write_json(paths["metrics"], payload)
    _write_json(
        paths["folds"],
        {
            "status": status,
            "strategy": strategy.value,
            "feature_policy": policy_label,
            "n_folds_total": len(completed_folds),
            "n_folds_successful": len(successful),
            "n_folds_failed": len(failed),
            "folds": [asdict(fold) for fold in completed_folds],
            "created_at_utc": _utc_now(),
        },
    )
    return BoostedBacktestSummary(
        status=status,
        strategy=strategy.value,
        feature_policy=policy_label,
        n_events=len(event_keys),
        n_folds_total=len(completed_folds),
        n_folds_successful=len(successful),
        n_folds_failed=len(failed),
        prediction_rows=(
            len(boosted_predictions) + len(baseline_predictions) if boosted_frames else 0
        ),
        metrics_path=paths["metrics"],
        predictions_path=predictions_path,
        folds_path=paths["folds"],
    )


def build_boosted_metrics_payload(
    strategy: BacktestStrategy | str,
    feature_policy: str,
    n_events: int,
    folds: list[BacktestFold],
    feature_groups_by_checkpoint: dict[str, str],
    boosted_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
) -> dict[str, object]:
    """Build fold-consistent boosted and baseline metrics."""
    strategy = BacktestStrategy(strategy)
    model_metrics = metrics_by_model_checkpoint(boosted_predictions)
    baseline_metrics = compute_baseline_metrics(baseline_predictions)
    model_fold = _fold_metrics(boosted_predictions, "model_name")
    baseline_fold = _fold_metrics(baseline_predictions, "baseline_name")
    checkpoints = boosted_predictions["checkpoint"].drop_duplicates().astype(str).tolist()
    best_models = _best_by_checkpoint(model_metrics, checkpoints, "model_name")
    best_baselines = _best_by_checkpoint(baseline_metrics, checkpoints, "baseline_name")
    mae_delta: dict[str, dict[str, float | None]] = {MODEL_NAME: {}}
    position_delta: dict[str, dict[str, float | None]] = {MODEL_NAME: {}}
    for checkpoint in checkpoints:
        model = model_metrics[MODEL_NAME][checkpoint]
        baseline = best_baselines.get(checkpoint, {})
        mae_delta[MODEL_NAME][checkpoint] = _delta(
            model.get("mae_gap_sec"), baseline.get("mae_gap_sec")
        )
        position_delta[MODEL_NAME][checkpoint] = _delta(
            model.get("mean_abs_position_error"),
            baseline.get("mean_abs_position_error"),
        )
    return {
        "status": "complete",
        "strategy": strategy.value,
        "feature_policy": feature_policy,
        "n_events": n_events,
        "n_folds_total": len(folds),
        "n_folds_successful": sum(fold.status == "success" for fold in folds),
        "n_folds_failed": sum(fold.status == "failed" for fold in folds),
        "models": list(model_metrics),
        "checkpoints": checkpoints,
        "feature_groups_by_checkpoint": feature_groups_by_checkpoint,
        "metrics_by_model_checkpoint": _aggregate_metrics(model_metrics, model_fold),
        "metrics_by_fold_model_checkpoint": model_fold,
        "baseline_metrics_by_model_checkpoint": _aggregate_metrics(baseline_metrics, baseline_fold),
        "baseline_metrics_by_fold_model_checkpoint": baseline_fold,
        "best_model_by_checkpoint": best_models,
        "best_baseline_by_checkpoint": best_baselines,
        "model_vs_best_baseline_delta_mae": mae_delta,
        "model_vs_best_baseline_delta_position_error": position_delta,
        "created_at_utc": _utc_now(),
    }


def _checkpoint_safe_features(dataset: pd.DataFrame, checkpoint: str) -> list[str]:
    from f1_prediction.modeling.tabular import usable_checkpoint_features

    return usable_checkpoint_features(dataset, checkpoint)


def _numeric_matrix(dataset: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return dataset.loc[:, columns].astype(float).replace([np.inf, -np.inf], np.nan)


def _tag_predictions(
    predictions: pd.DataFrame,
    fold: BacktestFold,
    prediction_type: str,
    feature_policy: str,
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["strategy"] = fold.strategy
    frame["fold_id"] = fold.fold_id
    frame["test_event"] = fold.test_event
    frame["prediction_type"] = prediction_type
    frame["feature_policy"] = feature_policy
    return frame


def _tag_baselines(predictions: pd.DataFrame, fold: BacktestFold) -> pd.DataFrame:
    frame = predictions.copy()
    frame["model_name"] = frame["baseline_name"]
    frame["feature_policy"] = "baseline"
    frame["feature_group"] = "baseline"
    frame["strategy"] = fold.strategy
    frame["fold_id"] = fold.fold_id
    frame["test_event"] = fold.test_event
    frame["prediction_type"] = "baseline"
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


def _output_paths(metrics_dir: Path) -> dict[str, Path]:
    return {
        "metrics": metrics_dir / "boosted_metrics.json",
        "predictions": metrics_dir / "boosted_predictions.parquet",
        "folds": metrics_dir / "boosted_folds.json",
    }


def _resolve_dataset_path(config: DataConfig, dataset_path: Path | None) -> Path:
    path = dataset_path or build_combined_dataset_path(config.modeling_output_dir)
    if not path.is_absolute():
        path = config.project_root / path
    if not path.is_file():
        raise FileNotFoundError(f"Modeling dataset does not exist: {path}")
    return path


def _event_key_series(dataset: pd.DataFrame) -> pd.Series:
    return dataset["season"].astype(str) + "/" + dataset["event_slug"].astype(str)


def _skipped_payload(
    strategy: BacktestStrategy,
    feature_policy: str,
    n_events: int,
    feature_groups: dict[str, str],
    reason: str,
) -> dict[str, object]:
    return {
        "status": "skipped",
        "strategy": strategy.value,
        "feature_policy": feature_policy,
        "reason": reason,
        "n_events": n_events,
        "n_folds_total": 0,
        "n_folds_successful": 0,
        "n_folds_failed": 0,
        "models": [],
        "feature_groups_by_checkpoint": feature_groups,
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
