import json
from pathlib import Path

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.data.season_builder import build_combined_dataset_path
from f1_prediction.modeling.evaluate_baselines import evaluate_baselines


def test_evaluate_baselines_writes_metrics_and_predictions(tmp_path: Path) -> None:
    config = _data_config(tmp_path)
    dataset_path = build_combined_dataset_path(config.modeling_output_dir)
    dataset_path.parent.mkdir(parents=True)
    _modeling_dataset().to_parquet(dataset_path, index=False)

    summary = evaluate_baselines(config)

    metrics = json.loads(summary.metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(summary.predictions_path)
    assert summary.prediction_rows == 18
    assert set(metrics) == {"best_push_lap", "best_valid_lap", "theoretical_best_lap"}
    assert set(metrics["best_push_lap"]) == {"after_fp1", "after_fp2", "after_fp3"}
    assert len(predictions) == 18
    assert summary.metrics_path.is_file()
    assert summary.predictions_path.is_file()


def _data_config(project_root: Path) -> DataConfig:
    return DataConfig(
        project_root=project_root,
        fastf1_cache_dir=project_root / "cache",
        lap_output_dir=project_root / "raw_laps",
        session_metadata_output_dir=project_root / "metadata",
        clean_lap_output_dir=project_root / "clean_laps",
        session_features_output_dir=project_root / "session_features",
        modeling_output_dir=project_root / "modeling",
        metrics_output_dir=project_root / "metrics",
    )


def _modeling_dataset() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for checkpoint in ("after_fp1", "after_fp2", "after_fp3"):
        for index, driver in enumerate(("NOR", "VER"), start=1):
            row: dict[str, object] = {
                "season": 2024,
                "event": "Monza",
                "event_slug": "monza",
                "checkpoint": checkpoint,
                "driver": driver,
                "team": "Team",
                "quali_position": index,
                "quali_best_lap_time_sec": 79.0 + index / 10,
                "quali_gap_to_pole_sec": (index - 1) / 10,
                "reached_q2": 1,
                "reached_q3": 1,
            }
            for session_index, session in enumerate(("fp1", "fp2", "fp3"), start=1):
                value = 80.0 + index / 10 - session_index / 10
                row[f"{session}_best_push_lap_time_sec"] = value
                row[f"{session}_best_valid_lap_time_sec"] = value
                row[f"{session}_theoretical_best_lap_time_sec"] = value - 0.05
            rows.append(row)
    return pd.DataFrame(rows)
