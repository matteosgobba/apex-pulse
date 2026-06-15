import json
from pathlib import Path

import pandas as pd

from f1_prediction.config import DataConfig, ModelConfig, RandomForestConfig
from f1_prediction.modeling.ablation import (
    build_ablation_metrics_payload,
    run_ablation_backtest,
)
from f1_prediction.modeling.backtest_tabular import BacktestFold, BacktestStrategy


def test_ablation_uses_identical_successful_folds_for_every_group(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = tmp_path / "dataset.parquet"
    _dataset(5).to_parquet(dataset_path, index=False)

    summary = run_ablation_backtest(
        config,
        strategy=BacktestStrategy.repeated_event_holdout,
        dataset_path=dataset_path,
        feature_group_names=["base_lap_features", "all_features"],
        min_events=5,
        model_config=_model_config(),
    )

    predictions = pd.read_parquet(summary.predictions_path)
    tabular = predictions[predictions["prediction_type"].eq("tabular")]
    fold_sets = {
        group: set(rows["fold_id"]) for group, rows in tabular.groupby("feature_group", sort=False)
    }
    assert summary.n_folds_successful == 5
    assert fold_sets["base_lap_features"] == fold_sets["all_features"] == {1, 2, 3, 4, 5}


def test_ablation_metrics_and_best_group_are_computed_per_model_checkpoint() -> None:
    predictions = pd.concat(
        [
            _predictions("base_lap_features", "ridge", [0.3, 0.3]),
            _predictions("all_features", "ridge", [0.1, 0.1]),
            _predictions("base_lap_features", "random_forest", [0.2, 0.2]),
            _predictions("all_features", "random_forest", [0.4, 0.4]),
        ],
        ignore_index=True,
    )
    baselines = _baseline_predictions([0.25, 0.25])
    folds = [
        BacktestFold(1, "walk_forward", "2024/event-1", (), 0, 1, "success"),
        BacktestFold(2, "walk_forward", "2024/event-2", (), 0, 1, "success"),
    ]

    report = build_ablation_metrics_payload(
        BacktestStrategy.walk_forward,
        2,
        folds,
        ("base_lap_features", "all_features"),
        predictions,
        baselines,
    )

    assert report["metrics_by_feature_group_model_checkpoint"]["all_features"]["ridge"]
    best_ridge = report["best_feature_group_by_model_checkpoint"]["ridge"]["after_fp1"]
    assert best_ridge["feature_group"] == "all_features"
    assert report["best_overall_by_checkpoint"]["after_fp1"]["feature_group"] == "all_features"
    assert report["model_vs_baseline_delta_mae"]["all_features"]["ridge"]["after_fp1"] < 0


def test_ablation_skips_when_dataset_has_too_few_events(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = tmp_path / "small.parquet"
    _dataset(3).to_parquet(dataset_path, index=False)

    summary = run_ablation_backtest(
        config,
        dataset_path=dataset_path,
        min_events=5,
        model_config=_model_config(),
    )

    report = json.loads(summary.metrics_path.read_text())
    assert summary.status == "skipped"
    assert report["status"] == "skipped"
    assert report["n_events"] == 3


def _predictions(group: str, model: str, predicted: list[float]) -> pd.DataFrame:
    rows = []
    for fold_id, value in enumerate(predicted, start=1):
        rows.append(
            {
                "fold_id": fold_id,
                "feature_group": group,
                "model_name": model,
                "season": 2024,
                "event_slug": f"event-{fold_id}",
                "checkpoint": "after_fp1",
                "driver": "NOR",
                "quali_gap_to_pole_sec": 0.0,
                "predicted_quali_gap_to_pole_sec": value,
                "quali_position": 1,
                "predicted_quali_position": 1,
                "reached_q3": 1,
                "predicted_reached_q3": 1,
            }
        )
    return pd.DataFrame(rows)


def _baseline_predictions(predicted: list[float]) -> pd.DataFrame:
    frame = _predictions("baseline", "robust", predicted)
    frame["baseline_name"] = "robust"
    return frame


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
        random_forest=RandomForestConfig(5, 3, 1),
    )
