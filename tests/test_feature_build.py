from pathlib import Path

import pandas as pd

from f1_prediction.config import DataConfig, FeatureConfig, PushLapConfig
from f1_prediction.data.fastf1_loader import build_lap_output_path
from f1_prediction.features.build import build_session_features


def test_build_session_features_writes_then_skips_existing_outputs(tmp_path: Path) -> None:
    data_config = _data_config(tmp_path)
    feature_config = FeatureConfig(push_lap=PushLapConfig(1.03, 1.07, ("SOFT", "MEDIUM", "HARD")))
    raw_path = build_lap_output_path(data_config.lap_output_dir, 2024, "Monza", "FP2")
    _raw_laps().to_parquet(raw_path, index=False)

    first = build_session_features(
        2024,
        "Monza",
        data_config,
        feature_config,
        sessions=("FP2",),
    )
    second = build_session_features(
        2024,
        "Monza",
        data_config,
        feature_config,
        sessions=("FP2",),
    )
    forced = build_session_features(
        2024,
        "Monza",
        data_config,
        feature_config,
        sessions=("FP2",),
        force=True,
    )

    assert first.clean_lap_files_written == 1
    assert first.aggregate_rows == 1
    assert first.clean_lap_paths[0].is_file()
    assert first.output_path.is_file()
    assert second.clean_lap_files_written == 0
    assert second.aggregate_rows == 1
    assert forced.clean_lap_files_written == 1
    assert forced.aggregate_rows == 1


def _data_config(project_root: Path) -> DataConfig:
    return DataConfig(
        project_root=project_root,
        fastf1_cache_dir=project_root / "cache",
        lap_output_dir=project_root / "raw_laps",
        session_metadata_output_dir=project_root / "metadata",
        clean_lap_output_dir=project_root / "clean_laps",
        session_features_output_dir=project_root / "session_features",
        modeling_output_dir=project_root / "modeling",
    )


def _raw_laps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Driver": ["NOR"],
            "Team": ["McLaren"],
            "LapNumber": [1.0],
            "Stint": [1.0],
            "Compound": ["SOFT"],
            "TyreLife": [2.0],
            "LapTime": [pd.Timedelta(seconds=80)],
            "Sector1Time": [pd.Timedelta(seconds=25)],
            "Sector2Time": [pd.Timedelta(seconds=30)],
            "Sector3Time": [pd.Timedelta(seconds=25)],
            "IsAccurate": [True],
            "Deleted": [False],
            "PitOutTime": [pd.NaT],
            "PitInTime": [pd.NaT],
        }
    )
