from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.config import DataConfig, DiagnosticsConfig
from f1_prediction.modeling.diagnostics import (
    build_diagnostics_report_payload,
    build_driver_error_summary,
    build_event_error_summary,
    create_diagnostics_report,
)

DIAGNOSTICS_CONFIG = DiagnosticsConfig(extreme_mae_gap_threshold_sec=1.5, top_n=5)


def test_event_and_driver_error_summaries_compute_expected_metrics() -> None:
    predictions = _standard_predictions("walk_forward")

    event_summary = build_event_error_summary(predictions)
    driver_summary = build_driver_error_summary(predictions)

    event = event_summary.iloc[0]
    assert event["n_rows"] == 2
    assert event["mae_gap_sec"] == pytest.approx(0.2)
    assert event["mean_abs_position_error"] == pytest.approx(0.5)
    nor = driver_summary.set_index("driver").loc["NOR"]
    assert nor["mae_gap_sec"] == pytest.approx(0.2)
    assert nor["avg_actual_position"] == 1.0
    assert nor["avg_predicted_position"] == 1.0


def test_extreme_error_threshold_identifies_baseline_outlier() -> None:
    predictions = _standard_predictions("walk_forward")
    predictions["model_name"] = "best_push_lap"
    predictions["predicted_quali_gap_to_pole_sec"] = [2.0, 2.2]
    event_summary = build_event_error_summary(predictions)
    driver_summary = build_driver_error_summary(predictions)

    report = build_diagnostics_report_payload(
        event_summary,
        driver_summary,
        available_sources=["walk_forward"],
        preferred_source="walk_forward",
        n_events=1,
        n_drivers=2,
        checkpoints=["after_fp1"],
        config=DIAGNOSTICS_CONFIG,
    )

    assert len(report["baseline_extreme_error_events"]) == 1
    assert report["baseline_extreme_error_events"][0]["mae_gap_sec"] > 1.5


def test_diagnostics_prefers_walk_forward_predictions(tmp_path: Path) -> None:
    config = _config(tmp_path)
    walk_path = tmp_path / "walk.parquet"
    repeated_path = tmp_path / "repeated.parquet"
    _artifact_predictions().to_parquet(walk_path, index=False)
    _artifact_predictions().to_parquet(repeated_path, index=False)

    summary = create_diagnostics_report(
        config,
        DIAGNOSTICS_CONFIG,
        walk_forward_path=walk_path,
        repeated_path=repeated_path,
    )

    assert summary.preferred_prediction_source == "walk_forward"
    assert summary.available_prediction_sources == (
        "walk_forward",
        "repeated_event_holdout",
    )
    assert summary.event_summary_path.is_file()
    assert summary.driver_summary_path.is_file()


def test_diagnostics_works_with_only_repeated_predictions(tmp_path: Path) -> None:
    config = _config(tmp_path)
    repeated_path = tmp_path / "repeated.parquet"
    _artifact_predictions().to_parquet(repeated_path, index=False)

    summary = create_diagnostics_report(
        config,
        DIAGNOSTICS_CONFIG,
        walk_forward_path=tmp_path / "missing.parquet",
        repeated_path=repeated_path,
    )

    assert summary.preferred_prediction_source == "repeated_event_holdout"
    assert summary.n_events == 1


def test_diagnostics_can_prefer_champion_predictions(tmp_path: Path) -> None:
    config = _config(tmp_path)
    champion_path = tmp_path / "champion.parquet"
    champion = _artifact_predictions().drop(columns="model_name")
    champion["selected_model_name"] = "nested_champion"
    champion.to_parquet(champion_path, index=False)

    summary = create_diagnostics_report(
        config,
        DIAGNOSTICS_CONFIG,
        walk_forward_path=tmp_path / "missing-walk.parquet",
        repeated_path=tmp_path / "missing-repeated.parquet",
        baseline_path=tmp_path / "missing-baseline.parquet",
        champion_path=champion_path,
    )

    assert summary.preferred_prediction_source == "champion"
    assert summary.available_prediction_sources == ("champion",)


def _standard_predictions(source: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prediction_source": [source, source],
            "season": [2024, 2024],
            "event": ["Monza", "Monza"],
            "event_slug": ["monza", "monza"],
            "checkpoint": ["after_fp1", "after_fp1"],
            "model_name": ["ridge", "ridge"],
            "driver": ["NOR", "VER"],
            "quali_gap_to_pole_sec": [0.0, 0.4],
            "predicted_quali_gap_to_pole_sec": [0.2, 0.2],
            "quali_position": [1, 2],
            "predicted_quali_position": [1, 1],
            "reached_q3": [1, 1],
            "predicted_reached_q3": [1, 1],
        }
    )


def _artifact_predictions() -> pd.DataFrame:
    frame = _standard_predictions("unused").drop(columns="prediction_source")
    frame["baseline_name"] = pd.NA
    return frame


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
