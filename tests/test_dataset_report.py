import json
from pathlib import Path

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.modeling.dataset_report import (
    build_dataset_quality_report,
    create_dataset_quality_report,
)


def test_dataset_quality_report_has_required_structure() -> None:
    report = build_dataset_quality_report(_quality_dataset())

    required = {
        "n_rows",
        "n_seasons",
        "seasons",
        "n_events",
        "events",
        "n_drivers",
        "drivers",
        "checkpoints",
        "rows_by_season",
        "rows_by_event",
        "rows_by_checkpoint",
        "missing_target_counts",
        "missing_feature_counts_top_30",
        "numeric_feature_missing_rate_top_30",
        "drivers_per_event",
        "checkpoints_per_event",
        "events_with_missing_checkpoints",
        "practice_only_driver_rows",
        "qualifying_only_driver_rows_if_detectable",
        "created_at_utc",
    }
    assert required <= report.keys()
    assert report["n_rows"] == 4
    assert report["n_events"] == 2
    assert report["events_with_missing_checkpoints"] == ["2024/spa"]
    assert report["practice_only_driver_rows"] == 1
    assert report["qualifying_only_driver_rows_if_detectable"] == 1


def test_create_dataset_quality_report_writes_json(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = tmp_path / "custom.parquet"
    _quality_dataset().to_parquet(dataset_path, index=False)

    summary = create_dataset_quality_report(config, dataset_path)

    payload = json.loads(summary.report_path.read_text(encoding="utf-8"))
    assert summary.n_events == 2
    assert payload["n_rows"] == 4
    assert summary.report_path.name == "dataset_quality_report.json"


def _quality_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024] * 4,
            "event": ["Monza", "Monza", "Spa", "Spa"],
            "event_slug": ["monza", "monza", "spa", "spa"],
            "checkpoint": ["after_fp1", "after_fp2", "after_fp1", "after_fp1"],
            "driver": ["NOR", "VER", "NOR", "VER"],
            "team": ["A", "B", "A", "B"],
            "fp1_best_valid_lap_time_sec": [80.0, 80.2, 81.0, pd.NA],
            "quali_position": [1, 2, pd.NA, 2],
            "quali_best_lap_time_sec": [79.0, 79.2, pd.NA, 80.0],
            "quali_gap_to_pole_sec": [0.0, 0.2, pd.NA, 1.0],
            "reached_q2": [1, 1, pd.NA, 1],
            "reached_q3": [1, 1, pd.NA, 1],
        }
    )


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
