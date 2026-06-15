import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from f1_prediction.config import (
    DataConfig,
    FeatureGroupPolicyConfig,
    HistGradientBoostingConfig,
    ModelConfig,
    RandomForestConfig,
)
from f1_prediction.modeling.backtest_tabular import BacktestStrategy, build_backtest_folds
from f1_prediction.modeling.boosted_backtest import (
    build_boosted_metrics_payload,
    get_checkpoint_feature_columns,
    resolve_feature_groups_by_checkpoint,
    run_boosted_backtest,
)
from f1_prediction.modeling.gradient_boosting import (
    MODEL_NAME,
    build_hist_gradient_boosting_regressor,
)


def test_hist_gradient_boosting_factory_builds_estimator() -> None:
    estimator = build_hist_gradient_boosting_regressor(
        HistGradientBoostingConfig(max_iter=25, learning_rate=0.1)
    )

    assert isinstance(estimator, HistGradientBoostingRegressor)
    assert estimator.max_iter == 25
    assert estimator.learning_rate == 0.1


def test_checkpoint_best_policy_uses_configured_groups() -> None:
    selected = resolve_feature_groups_by_checkpoint(
        _dataset(5), _model_config(), feature_policy="checkpoint_best"
    )

    assert selected == {
        "after_fp1": "base_lap_features",
        "after_fp2": "base_plus_quality",
        "after_fp3": "base_plus_relative",
    }


def test_all_features_and_explicit_group_policies() -> None:
    dataset = _dataset(5)
    all_features = resolve_feature_groups_by_checkpoint(
        dataset, _model_config(), feature_policy="all_features"
    )
    explicit = resolve_feature_groups_by_checkpoint(
        dataset,
        _model_config(),
        feature_policy="checkpoint_best",
        explicit_feature_group="base_plus_relative",
    )

    assert set(all_features.values()) == {"all_features"}
    assert set(explicit.values()) == {"base_plus_relative"}


def test_checkpoint_features_exclude_future_identifiers_and_targets() -> None:
    columns = get_checkpoint_feature_columns(
        _dataset(5).query("checkpoint == 'after_fp1'"),
        "after_fp1",
        "all_features",
    )

    assert any(column.startswith("fp1_") for column in columns)
    assert not any(column.startswith(("fp2_", "fp3_", "quali_")) for column in columns)
    assert not {"season", "event_order", "driver", "team"} & set(columns)


def test_boosted_backtest_uses_existing_walk_forward_folds_and_schema(tmp_path: Path) -> None:
    dataset = _dataset(6)
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)

    summary = run_boosted_backtest(
        _config(tmp_path),
        strategy=BacktestStrategy.walk_forward,
        feature_policy="checkpoint_best",
        dataset_path=dataset_path,
        min_events=5,
        min_train_events=3,
        model_config=_model_config(),
    )

    expected_folds = build_backtest_folds(
        dataset, BacktestStrategy.walk_forward, min_train_events=3
    )
    fold_report = json.loads(summary.folds_path.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(summary.predictions_path)
    boosted = predictions[predictions["prediction_type"].eq("boosted")]
    required = {
        "strategy",
        "fold_id",
        "model_name",
        "feature_policy",
        "feature_group",
        "checkpoint",
        "driver",
        "quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec",
        "predicted_quali_position",
    }

    assert summary.n_folds_successful == len(expected_folds)
    assert [fold["test_event"] for fold in fold_report["folds"]] == [
        fold.test_event for fold in expected_folds
    ]
    assert required <= set(boosted.columns)
    assert set(boosted["model_name"]) == {MODEL_NAME}
    assert set(boosted["feature_policy"]) == {"checkpoint_best"}


def test_boosted_metrics_and_baseline_delta_sign() -> None:
    boosted = _predictions(MODEL_NAME, [0.1, 0.2])
    baseline = _predictions("robust_best_push_lap", [0.3, 0.4], baseline=True)
    payload = build_boosted_metrics_payload(
        BacktestStrategy.walk_forward,
        "checkpoint_best",
        2,
        [],
        {"after_fp1": "base_lap_features"},
        boosted,
        baseline,
    )

    metrics = payload["metrics_by_model_checkpoint"][MODEL_NAME]["after_fp1"]
    assert metrics["global"]["mae_gap_sec"] == pytest.approx(0.15)
    assert payload["model_vs_best_baseline_delta_mae"][MODEL_NAME]["after_fp1"] < 0


def test_boosted_backtest_skips_when_dataset_is_too_small(tmp_path: Path) -> None:
    dataset_path = tmp_path / "small.parquet"
    _dataset(3).to_parquet(dataset_path, index=False)

    summary = run_boosted_backtest(
        _config(tmp_path),
        dataset_path=dataset_path,
        min_events=5,
        model_config=_model_config(),
    )
    metrics = json.loads(summary.metrics_path.read_text(encoding="utf-8"))

    assert summary.status == "skipped"
    assert metrics["status"] == "skipped"
    assert metrics["n_events"] == 3


def _predictions(
    model_name: str,
    predicted_gaps: list[float],
    *,
    baseline: bool = False,
) -> pd.DataFrame:
    rows = []
    for fold_id, predicted in enumerate(predicted_gaps, start=1):
        rows.append(
            {
                "fold_id": fold_id,
                "season": 2024,
                "event_slug": f"event-{fold_id}",
                "checkpoint": "after_fp1",
                "driver": "NOR",
                "quali_gap_to_pole_sec": 0.0,
                "predicted_quali_gap_to_pole_sec": predicted,
                "quali_position": 1,
                "predicted_quali_position": 1,
                "reached_q3": 1,
                "predicted_reached_q3": 1,
                "model_name": model_name,
                "baseline_name": model_name if baseline else None,
            }
        )
    return pd.DataFrame(rows)


def _dataset(n_events: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    checkpoints = ("after_fp1", "after_fp2", "after_fp3")
    for event_index in range(1, n_events + 1):
        for checkpoint_index, checkpoint in enumerate(checkpoints, start=1):
            for driver_index, driver in enumerate(("NOR", "VER", "LEC"), start=1):
                gap = (driver_index - 1) * 0.2 + event_index * 0.01
                row: dict[str, object] = {
                    "season": 2024,
                    "event": f"Event {event_index}",
                    "event_slug": f"event-{event_index}",
                    "event_order": event_index,
                    "checkpoint": checkpoint,
                    "driver": driver,
                    "team": f"Team {driver_index}",
                    "quali_position": driver_index,
                    "quali_best_lap_time_sec": 79.0 + gap,
                    "quali_gap_to_pole_sec": gap,
                    "reached_q2": 1,
                    "reached_q3": 1,
                    "practice_signal_quality_score": float(checkpoint_index),
                }
                for session_index, session in enumerate(("fp1", "fp2", "fp3"), start=1):
                    available = session_index <= checkpoint_index
                    pace = 80.0 + gap if available else float("nan")
                    row[f"{session}_best_push_lap_time_sec"] = pace
                    row[f"{session}_best_valid_lap_time_sec"] = pace
                    row[f"{session}_theoretical_best_lap_time_sec"] = pace
                    row[f"{session}_n_push_laps"] = 5.0 if available else float("nan")
                    row[f"{session}_best_push_gap_to_session_best_sec"] = (
                        gap if available else float("nan")
                    )
                    row[f"{session}_best_valid_gap_to_session_best_sec"] = (
                        gap if available else float("nan")
                    )
                    row[f"{session}_theoretical_best_gap_to_session_best_sec"] = (
                        gap if available else float("nan")
                    )
                    row[f"{session}_best_push_gap_to_teammate_sec"] = (
                        gap if available else float("nan")
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def _config(project_root: Path) -> DataConfig:
    return DataConfig(
        project_root=project_root,
        fastf1_cache_dir=project_root / "cache",
        lap_output_dir=project_root / "laps",
        session_metadata_output_dir=project_root / "metadata",
        clean_lap_output_dir=project_root / "clean_laps",
        session_features_output_dir=project_root / "session_features",
        modeling_output_dir=project_root / "modeling",
        metrics_output_dir=project_root / "metrics",
    )


def _model_config() -> ModelConfig:
    return ModelConfig(
        min_events=5,
        random_state=42,
        ridge_alpha=1.0,
        random_forest=RandomForestConfig(
            n_estimators=5,
            max_depth=3,
            min_samples_leaf=1,
        ),
        hist_gradient_boosting=HistGradientBoostingConfig(
            max_iter=20,
            learning_rate=0.1,
            max_leaf_nodes=7,
            l2_regularization=0.1,
            random_state=42,
        ),
        feature_group_policy=FeatureGroupPolicyConfig(),
    )
