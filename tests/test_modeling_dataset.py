from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.features.modeling_dataset import (
    CHECKPOINT_SESSIONS,
    build_checkpoint_modeling_dataset,
    build_modeling_output_path,
    get_feature_columns,
)
from f1_prediction.features.qualifying_targets import TARGET_COLUMNS, build_qualifying_targets


def test_qualifying_targets_compute_positions_gaps_and_advancement_flags() -> None:
    targets = build_qualifying_targets(_raw_qualifying_laps(), season=2024, event="Monza")
    indexed = targets.set_index("driver")

    assert len(targets) == 20
    assert indexed.loc["D01", "quali_position"] == 1
    assert indexed.loc["D01", "quali_gap_to_pole_sec"] == pytest.approx(0.0)
    assert indexed.loc["D02", "quali_gap_to_pole_sec"] == pytest.approx(0.1)
    assert indexed.loc["D10", "reached_q3"] == 1
    assert indexed.loc["D11", "reached_q3"] == 0
    assert indexed.loc["D15", "reached_q2"] == 1
    assert indexed.loc["D16", "reached_q2"] == 0


def test_qualifying_targets_keep_driver_without_valid_lap_after_timed_drivers() -> None:
    targets = build_qualifying_targets(_raw_qualifying_laps(), season=2024, event="Monza")
    invalid = targets.set_index("driver").loc["D20"]

    assert invalid["quali_position"] == 20
    assert pd.isna(invalid["quali_best_lap_time_sec"])
    assert pd.isna(invalid["quali_gap_to_pole_sec"])
    assert invalid["reached_q2"] == 0
    assert invalid["reached_q3"] == 0


def test_checkpoint_rows_exclude_future_practice_values_and_qualifying_predictors() -> None:
    dataset = build_checkpoint_modeling_dataset(_practice_features(), _small_targets())

    after_fp1 = dataset[dataset["checkpoint"].eq("after_fp1")]
    after_fp2 = dataset[dataset["checkpoint"].eq("after_fp2")]
    after_fp3 = dataset[dataset["checkpoint"].eq("after_fp3")]

    assert after_fp1["fp1_n_push_laps"].notna().all()
    assert after_fp1.filter(regex=r"^fp[23]_").isna().all().all()
    assert after_fp2["fp2_n_push_laps"].notna().all()
    assert after_fp2.filter(regex=r"^fp3_").isna().all().all()
    assert after_fp3["fp3_n_push_laps"].notna().all()
    assert not any(column.startswith(("q_", "quali_")) for column in get_feature_columns(dataset))
    assert not any("q_leak" in column or "quali_leak" in column for column in dataset.columns)


def test_feature_column_helper_excludes_identifiers_and_targets() -> None:
    dataset = build_checkpoint_modeling_dataset(_practice_features(), _small_targets())
    feature_columns = get_feature_columns(dataset)

    assert feature_columns
    assert not set(TARGET_COLUMNS).intersection(feature_columns)
    assert not {"season", "event", "checkpoint", "driver", "team"}.intersection(feature_columns)


def test_checkpoint_dataset_keeps_target_driver_missing_from_practice() -> None:
    targets = _small_targets()
    practice = _practice_features()
    practice = practice[practice["driver"].ne("VER")]

    dataset = build_checkpoint_modeling_dataset(practice, targets)
    verstappen = dataset[dataset["driver"].eq("VER")]

    assert len(verstappen) == len(CHECKPOINT_SESSIONS)
    practice_columns = [
        column for column in get_feature_columns(dataset) if column.startswith("fp")
    ]
    assert verstappen[practice_columns].isna().all().all()
    assert verstappen["practice_signal_quality_score"].eq(0).all()
    assert ~verstappen["has_any_practice_time"].all()
    assert verstappen["quali_position"].eq(1).all()


def test_modeling_output_path(tmp_path: Path) -> None:
    path = build_modeling_output_path(tmp_path, 2024, "São Paulo")

    assert path == tmp_path / "2024/sao-paulo/modeling_dataset.parquet"


def _raw_qualifying_laps() -> pd.DataFrame:
    drivers = [f"D{index:02d}" for index in range(1, 21)]
    lap_times = [79.0 + index / 10 for index in range(20)]
    accurate = [True] * 19 + [False]
    return pd.DataFrame(
        {
            "Driver": drivers,
            "Team": ["Team"] * 20,
            "LapNumber": [1.0] * 20,
            "Stint": [1.0] * 20,
            "Compound": ["SOFT"] * 20,
            "TyreLife": [2.0] * 20,
            "LapTime": pd.to_timedelta(lap_times, unit="s"),
            "Sector1Time": pd.to_timedelta([25.0] * 20, unit="s"),
            "Sector2Time": pd.to_timedelta([29.0] * 20, unit="s"),
            "Sector3Time": pd.to_timedelta([25.0] * 20, unit="s"),
            "IsAccurate": accurate,
            "Deleted": [False] * 20,
            "PitOutTime": [pd.NaT] * 20,
            "PitInTime": [pd.NaT] * 20,
        }
    )


def _practice_features() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session_index, session in enumerate(("FP1", "FP2", "FP3"), start=1):
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
                    "n_push_laps": session_index,
                    "best_push_lap_time_sec": 80.0 + session_index,
                    "q_leak": 999,
                    "quali_leak": 999,
                }
            )
    return pd.DataFrame(rows)


def _small_targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024, 2024],
            "event": ["Monza", "Monza"],
            "event_slug": ["monza", "monza"],
            "driver": ["VER", "NOR"],
            "quali_position": pd.Series([1, 2], dtype="Int64"),
            "quali_best_lap_time_sec": [79.0, 79.1],
            "quali_gap_to_pole_sec": [0.0, 0.1],
            "reached_q2": [1, 1],
            "reached_q3": [1, 1],
        }
    )
