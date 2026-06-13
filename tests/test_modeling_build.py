from pathlib import Path

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.data.fastf1_loader import build_lap_output_path
from f1_prediction.features.build import build_session_features_output_path
from f1_prediction.features.modeling_dataset import build_modeling_dataset_files


def test_modeling_dataset_build_writes_skips_and_forces(tmp_path: Path) -> None:
    config = _data_config(tmp_path)
    practice_path = build_session_features_output_path(
        config.session_features_output_dir,
        2024,
        "Monza",
    )
    qualifying_path = build_lap_output_path(config.lap_output_dir, 2024, "Monza", "Q")
    _practice_features().to_parquet(practice_path, index=False)
    _qualifying_laps().to_parquet(qualifying_path, index=False)

    first = build_modeling_dataset_files(2024, "Monza", config)
    second = build_modeling_dataset_files(2024, "Monza", config)
    forced = build_modeling_dataset_files(2024, "Monza", config, force=True)

    assert first.rows == 6
    assert first.drivers == 2
    assert first.practice_only_drivers == ("ANT",)
    assert first.output_path.is_file()
    assert second.skipped is True
    assert forced.skipped is False
    assert forced.rows == 6


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


def _practice_features() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session in ("FP1", "FP2", "FP3"):
        for driver, team in (("VER", "Red Bull Racing"), ("NOR", "McLaren")):
            rows.append(
                {
                    "season": 2024,
                    "event": "Monza",
                    "event_slug": "monza",
                    "session": session,
                    "session_slug": session.lower(),
                    "driver": driver,
                    "team": team,
                    "n_push_laps": 3,
                    "best_push_lap_time_sec": 81.0,
                }
            )
    rows.append(
        {
            "season": 2024,
            "event": "Monza",
            "event_slug": "monza",
            "session": "FP1",
            "session_slug": "fp1",
            "driver": "ANT",
            "team": "Mercedes",
            "n_push_laps": 1,
            "best_push_lap_time_sec": 83.0,
        }
    )
    return pd.DataFrame(rows)


def _qualifying_laps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Driver": ["VER", "NOR"],
            "Team": ["Red Bull Racing", "McLaren"],
            "LapNumber": [1.0, 1.0],
            "Stint": [1.0, 1.0],
            "Compound": ["SOFT", "SOFT"],
            "TyreLife": [2.0, 2.0],
            "LapTime": pd.to_timedelta([79.0, 79.1], unit="s"),
            "Sector1Time": pd.to_timedelta([25.0, 25.0], unit="s"),
            "Sector2Time": pd.to_timedelta([29.0, 29.0], unit="s"),
            "Sector3Time": pd.to_timedelta([25.0, 25.1], unit="s"),
            "IsAccurate": [True, True],
            "Deleted": [False, False],
            "PitOutTime": [pd.NaT, pd.NaT],
            "PitInTime": [pd.NaT, pd.NaT],
        }
    )
