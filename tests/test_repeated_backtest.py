import json
from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.config import DataConfig, ModelConfig, RandomForestConfig
from f1_prediction.modeling.backtest_tabular import (
    BacktestStrategy,
    build_backtest_folds,
    build_backtest_metrics_payload,
    run_tabular_backtest,
)


def test_repeated_holdout_creates_one_fold_per_event() -> None:
    folds = build_backtest_folds(_dataset(5), BacktestStrategy.repeated_event_holdout)

    assert len(folds) == 5
    assert {fold.test_event for fold in folds} == {f"2024/event-{index}" for index in range(1, 6)}
    assert all(len(fold.train_events) == 4 for fold in folds)


def test_repeated_holdout_can_restrict_test_events() -> None:
    folds = build_backtest_folds(
        _dataset(5),
        BacktestStrategy.repeated_event_holdout,
        test_events=["Event 2", "Event 4"],
    )

    assert [fold.test_event for fold in folds] == ["2024/event-2", "2024/event-4"]


def test_walk_forward_starts_after_minimum_and_uses_only_past_events() -> None:
    folds = build_backtest_folds(
        _dataset(6),
        BacktestStrategy.walk_forward,
        min_train_events=3,
    )

    assert len(folds) == 3
    assert folds[0].test_event == "2024/event-4"
    ordered = [f"2024/event-{index}" for index in range(1, 7)]
    for fold in folds:
        assert max(ordered.index(event) for event in fold.train_events) < ordered.index(
            fold.test_event
        )


def test_backtest_predictions_and_baselines_use_same_fold_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = tmp_path / "dataset.parquet"
    _dataset(5).to_parquet(dataset_path, index=False)

    summary = run_tabular_backtest(
        config,
        strategy=BacktestStrategy.repeated_event_holdout,
        dataset_path=dataset_path,
        test_events=["Event 4", "Event 5"],
        min_events=5,
        model_config=_model_config(),
    )

    predictions = pd.read_parquet(summary.predictions_path)
    required = {
        "model_name",
        "fold_id",
        "test_event",
        "checkpoint",
        "driver",
        "quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec",
        "predicted_quali_position",
    }
    assert required <= set(predictions.columns)
    baseline_names = set(
        predictions.loc[predictions["prediction_type"].eq("baseline"), "model_name"]
    )
    assert "robust_best_push_lap" in baseline_names
    for fold_id, fold_rows in predictions.groupby("fold_id"):
        tabular_drivers = set(fold_rows.loc[fold_rows["prediction_type"].eq("tabular"), "driver"])
        baseline_drivers = set(fold_rows.loc[fold_rows["prediction_type"].eq("baseline"), "driver"])
        assert tabular_drivers == baseline_drivers, fold_id


def test_fold_metadata_records_success_and_failure(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = tmp_path / "dataset.parquet"
    _dataset(5).to_parquet(dataset_path, index=False)
    from f1_prediction.modeling import backtest_tabular

    original = backtest_tabular.fit_and_predict

    def sometimes_fail(train, test, *, model_config, sample_weights=None):
        if test["event_slug"].eq("event-2").all():
            raise RuntimeError("synthetic fold failure")
        return original(train, test, model_config=model_config, sample_weights=sample_weights)

    monkeypatch.setattr(backtest_tabular, "fit_and_predict", sometimes_fail)
    summary = run_tabular_backtest(
        config,
        strategy=BacktestStrategy.repeated_event_holdout,
        dataset_path=dataset_path,
        test_events=["Event 1", "Event 2"],
        min_events=5,
        model_config=_model_config(),
    )

    folds = json.loads(summary.folds_path.read_text(encoding="utf-8"))["folds"]
    assert summary.n_folds_successful == 1
    assert summary.n_folds_failed == 1
    assert {fold["status"] for fold in folds} == {"success", "failed"}
    failed = next(fold for fold in folds if fold["status"] == "failed")
    assert "synthetic fold failure" in failed["error_message"]


def test_aggregate_metrics_and_delta_sign_from_synthetic_predictions() -> None:
    tabular = _predictions("ridge", [0.1, 0.2])
    forest = _predictions("random_forest", [0.2, 0.4])
    baseline = _predictions("best_push_lap", [0.3, 0.6], baseline=True)
    payload = build_backtest_metrics_payload(
        BacktestStrategy.repeated_event_holdout,
        2,
        [],
        pd.concat([tabular, forest], ignore_index=True),
        baseline,
    )

    ridge = payload["metrics_by_model_checkpoint"]["ridge"]["after_fp1"]
    assert ridge["global"]["mae_gap_sec"] == pytest.approx(0.15)
    assert ridge["fold_mean"]["mae_gap_sec"] == pytest.approx(0.15)
    assert ridge["fold_std"]["mae_gap_sec"] == pytest.approx(0.05)
    assert payload["model_vs_best_baseline_delta_mae"]["ridge"]["after_fp1"] < 0
    assert payload["model_vs_best_baseline_delta_mae"]["random_forest"]["after_fp1"] < 0


def test_min_events_guard_writes_skipped_report(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = tmp_path / "small.parquet"
    _dataset(3).to_parquet(dataset_path, index=False)

    summary = run_tabular_backtest(
        config,
        strategy=BacktestStrategy.walk_forward,
        dataset_path=dataset_path,
        min_events=5,
        model_config=_model_config(),
    )

    metrics = json.loads(summary.metrics_path.read_text(encoding="utf-8"))
    assert summary.status == "skipped"
    assert metrics["status"] == "skipped"
    assert metrics["n_events"] == 3


def test_weighted_backtest_artifacts_coexist_with_uniform_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = tmp_path / "dataset.parquet"
    dataset = pd.concat([_dataset(3), _dataset(3).assign(season=2025)], ignore_index=True)
    dataset.to_parquet(dataset_path, index=False)

    uniform = run_tabular_backtest(
        config,
        strategy=BacktestStrategy.walk_forward,
        dataset_path=dataset_path,
        min_events=5,
        min_train_events=3,
        model_config=_model_config(),
        temporal_weighting="uniform",
    )
    weighted = run_tabular_backtest(
        config,
        strategy=BacktestStrategy.walk_forward,
        dataset_path=dataset_path,
        min_events=5,
        min_train_events=3,
        model_config=_model_config(),
        temporal_weighting="season_priority",
    )

    assert uniform.metrics_path == weighted.metrics_path
    assert (config.metrics_output_dir / "walk_forward_uniform_metrics.json").is_file()
    assert (config.metrics_output_dir / "walk_forward_season_priority_metrics.json").is_file()
    payload = json.loads(
        (config.metrics_output_dir / "walk_forward_season_priority_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["temporal_weighting_policy"] == "season_priority"
    assert payload["training_weight_summary_by_fold"]
    predictions = pd.read_parquet(
        config.metrics_output_dir / "walk_forward_season_priority_predictions.parquet"
    )
    assert set(predictions["temporal_weighting_policy"]) == {"season_priority"}


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
    for event_index in range(1, n_events + 1):
        for checkpoint_index, checkpoint in enumerate(
            ("after_fp1", "after_fp2", "after_fp3"), start=1
        ):
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
                }
                for session_index, session in enumerate(("fp1", "fp2", "fp3"), start=1):
                    available = session_index <= checkpoint_index
                    pace = 80.0 + gap if available else pd.NA
                    row[f"{session}_best_push_lap_time_sec"] = pace
                    row[f"{session}_best_valid_lap_time_sec"] = pace
                    row[f"{session}_theoretical_best_lap_time_sec"] = pace
                    row[f"{session}_n_push_laps"] = 5 if available else pd.NA
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
    )
