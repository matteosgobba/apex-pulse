import json
from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.config import DataConfig, ModelConfig, RandomForestConfig
from f1_prediction.modeling.metrics import compute_prediction_metrics
from f1_prediction.modeling.tabular import rank_gap_predictions
from f1_prediction.modeling.train_tabular import fit_and_predict, train_tabular_models


def test_training_skips_when_event_count_is_below_minimum(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = tmp_path / "small.parquet"
    _training_dataset(n_events=2).to_parquet(dataset_path, index=False)

    summary = train_tabular_models(config, dataset_path, min_events=5)

    report = json.loads(summary.metrics_path.read_text(encoding="utf-8"))
    assert summary.status == "skipped"
    assert report["status"] == "skipped"
    assert report["n_events"] == 2
    assert summary.predictions_path is None


def test_simple_tabular_training_runs_on_five_events(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = tmp_path / "training.parquet"
    _training_dataset(n_events=5).to_parquet(dataset_path, index=False)
    config.metrics_output_dir.mkdir(parents=True)
    (config.metrics_output_dir / "baseline_metrics.json").write_text(
        json.dumps({"best_push_lap": {}}), encoding="utf-8"
    )

    summary = train_tabular_models(
        config,
        dataset_path,
        test_events=["Event 5"],
        min_events=5,
    )

    report = json.loads(summary.metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(summary.predictions_path)
    assert summary.status == "trained"
    assert set(summary.models) == {"mean_target", "median_target", "ridge", "random_forest"}
    assert len(predictions) == 48
    assert summary.ridge_model_path.is_file()
    assert summary.random_forest_model_path.is_file()
    assert report["baseline_comparison_scope"] == "same_test_holdout"
    assert all(
        not column.startswith(("fp2_", "fp3_"))
        for column in report["feature_columns_by_checkpoint"]["after_fp1"]
    )


def test_prediction_ranking_is_within_event_and_checkpoint() -> None:
    predictions = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "event_slug": ["monza", "monza", "spa", "spa"],
            "checkpoint": ["after_fp3"] * 4,
            "driver": ["NOR", "VER", "LEC", "SAI"],
            "predicted_quali_gap_to_pole_sec": [0.2, 0.0, 0.4, 0.1],
        }
    )

    positions = rank_gap_predictions(predictions)

    assert positions.tolist() == [2, 1, 2, 1]


def test_tabular_prediction_metrics() -> None:
    predictions = pd.DataFrame(
        {
            "season": [2024, 2024],
            "event_slug": ["monza", "monza"],
            "driver": ["NOR", "VER"],
            "quali_gap_to_pole_sec": [0.0, 0.4],
            "predicted_quali_gap_to_pole_sec": [0.1, 0.2],
            "quali_position": [1, 2],
            "predicted_quali_position": [1, 2],
            "reached_q3": [1, 1],
            "predicted_reached_q3": [1, 1],
        }
    )

    metrics = compute_prediction_metrics(predictions)

    assert metrics["mae_gap_sec"] == pytest.approx(0.15)
    assert metrics["mean_abs_position_error"] == 0.0
    assert metrics["spearman_corr"] == pytest.approx(1.0)


def test_ridge_and_random_forest_receive_sample_weights(monkeypatch) -> None:
    captured: dict[str, list[pd.Series]] = {"ridge": [], "random_forest": []}

    class FakeEstimator:
        def __init__(self, name: str) -> None:
            self.name = name

        def fit(self, _x, _y, **kwargs):
            captured[self.name].append(kwargs["regressor__sample_weight"])
            return self

        def predict(self, x):
            return [0.0] * len(x)

    monkeypatch.setattr(
        "f1_prediction.modeling.train_tabular.build_regressors",
        lambda model_config: {
            "ridge": FakeEstimator("ridge"),
            "random_forest": FakeEstimator("random_forest"),
        },
    )
    dataset = _training_dataset(2)
    train = dataset[dataset["event_slug"].eq("event-1")]
    test = dataset[dataset["event_slug"].eq("event-2")]
    weights = pd.Series(0.25, index=train.index)

    fit_and_predict(
        train,
        test,
        model_config=ModelConfig(
            min_events=2,
            random_state=42,
            ridge_alpha=1.0,
            random_forest=RandomForestConfig(5, 3, 1),
        ),
        sample_weights=weights,
    )

    assert captured["ridge"]
    assert captured["random_forest"]
    assert all(series.eq(0.25).all() for series in captured["ridge"])
    assert all(series.eq(0.25).all() for series in captured["random_forest"])


def _training_dataset(n_events: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    checkpoints = ("after_fp1", "after_fp2", "after_fp3")
    for event_index in range(1, n_events + 1):
        for checkpoint_index, checkpoint in enumerate(checkpoints, start=1):
            for driver_index, driver in enumerate(("NOR", "VER", "LEC", "SAI"), start=1):
                gap = (driver_index - 1) * 0.15 + event_index * 0.01
                row: dict[str, object] = {
                    "season": 2024,
                    "event": f"Event {event_index}",
                    "event_slug": f"event-{event_index}",
                    "event_order": event_index,
                    "checkpoint": checkpoint,
                    "driver": driver,
                    "team": f"Team {driver_index // 2}",
                    "quali_position": driver_index,
                    "quali_best_lap_time_sec": 79.0 + gap,
                    "quali_gap_to_pole_sec": gap,
                    "reached_q2": 1,
                    "reached_q3": 1,
                }
                for session_index, session in enumerate(("fp1", "fp2", "fp3"), start=1):
                    available = session_index <= checkpoint_index
                    pace = 80.0 + gap - session_index * 0.05 if available else pd.NA
                    row[f"{session}_best_push_lap_time_sec"] = pace
                    row[f"{session}_best_valid_lap_time_sec"] = pace
                    row[f"{session}_theoretical_best_lap_time_sec"] = (
                        pace - 0.03 if available else pd.NA
                    )
                    row[f"{session}_n_push_laps"] = 5 + driver_index if available else pd.NA
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
