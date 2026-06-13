from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.config import PushLapConfig
from f1_prediction.features.build import build_session_features_output_path
from f1_prediction.features.lap_cleaning import (
    build_clean_lap_output_path,
    clean_session_laps,
    time_to_seconds,
)
from f1_prediction.features.push_laps import add_push_lap_flags
from f1_prediction.features.session_aggregates import aggregate_session_features

PUSH_CONFIG = PushLapConfig(
    driver_best_pct_threshold=1.03,
    session_best_pct_threshold=1.07,
    allowed_compounds=("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"),
)


def test_time_to_seconds_converts_timedeltas() -> None:
    values = pd.Series(pd.to_timedelta(["00:01:20.500", None]))

    result = time_to_seconds(values)

    assert result.iloc[0] == pytest.approx(80.5)
    assert pd.isna(result.iloc[1])


def test_valid_lap_detection_applies_required_rules() -> None:
    raw = _raw_laps()

    cleaned = clean_session_laps(raw, season=2024, event="Monza", session="FP2")

    assert cleaned["is_valid_lap"].tolist() == [True, False, False, False, False]
    assert cleaned.loc[1, "is_out_lap"]
    assert cleaned.loc[2, "is_in_lap"]
    assert cleaned.loc[3, "is_deleted"]
    assert not cleaned.loc[4, "is_accurate"]


def test_push_lap_detection_uses_driver_and_session_thresholds() -> None:
    laps = _clean_feature_laps()

    flagged = add_push_lap_flags(laps, PUSH_CONFIG)

    assert flagged["is_push_lap"].tolist() == [True, False, True, False]


def test_session_aggregates_compute_driver_features_and_ranks() -> None:
    flagged = add_push_lap_flags(_clean_feature_laps(), PUSH_CONFIG)

    features = aggregate_session_features(flagged).set_index("driver")

    assert len(features) == 3
    assert features.loc["NOR", "n_laps"] == 2
    assert features.loc["NOR", "n_valid_laps"] == 2
    assert features.loc["NOR", "n_push_laps"] == 1
    assert features.loc["NOR", "best_valid_lap_time_sec"] == pytest.approx(80.0)
    assert features.loc["NOR", "theoretical_best_lap_time_sec"] == pytest.approx(79.5)
    assert features.loc["NOR", "best_vs_theoretical_gap_sec"] == pytest.approx(0.5)
    assert features.loc["NOR", "valid_lap_rank"] == 1
    assert features.loc["VER", "valid_lap_rank"] == 2
    assert features.loc["VER", "best_valid_gap_to_session_best_sec"] == pytest.approx(4.0)


def test_session_aggregates_keep_driver_with_no_push_laps() -> None:
    flagged = add_push_lap_flags(_clean_feature_laps(), PUSH_CONFIG)

    features = aggregate_session_features(flagged).set_index("driver")

    assert features.loc["BOT", "n_push_laps"] == 0
    assert pd.isna(features.loc["BOT", "best_push_lap_time_sec"])
    assert pd.isna(features.loc["BOT", "push_lap_rank"])


def test_session_aggregates_keep_driver_with_no_valid_laps() -> None:
    laps = _clean_feature_laps()
    laps.loc[laps["driver"].eq("BOT"), "is_valid_lap"] = False
    flagged = add_push_lap_flags(laps, PUSH_CONFIG)

    features = aggregate_session_features(flagged).set_index("driver")

    assert features.loc["BOT", "n_valid_laps"] == 0
    assert features.loc["BOT", "n_push_laps"] == 0
    assert pd.isna(features.loc["BOT", "best_valid_lap_time_sec"])
    assert pd.isna(features.loc["BOT", "valid_lap_rank"])


def test_feature_output_paths(tmp_path: Path) -> None:
    clean_path = build_clean_lap_output_path(tmp_path / "clean", 2024, "Monza", "FP2")
    features_path = build_session_features_output_path(
        tmp_path / "features",
        2024,
        "Monza",
    )

    assert clean_path == tmp_path / "clean/2024/monza/fp2_clean_laps.parquet"
    assert features_path == tmp_path / "features/2024/monza/practice_session_features.parquet"


def _raw_laps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Driver": ["NOR"] * 5,
            "Team": ["McLaren"] * 5,
            "LapNumber": [1, 2, 3, 4, 5],
            "Stint": [1] * 5,
            "Compound": ["SOFT"] * 5,
            "TyreLife": [1, 2, 3, 4, 5],
            "LapTime": pd.to_timedelta([80, 81, 82, 83, 84], unit="s"),
            "Sector1Time": pd.to_timedelta([25, 25, 25, 25, 25], unit="s"),
            "Sector2Time": pd.to_timedelta([30, 30, 30, 30, 30], unit="s"),
            "Sector3Time": pd.to_timedelta([25, 26, 27, 28, 29], unit="s"),
            "IsAccurate": [True, True, True, True, False],
            "Deleted": [False, False, False, True, False],
            "PitOutTime": [pd.NaT, pd.Timedelta(seconds=1), pd.NaT, pd.NaT, pd.NaT],
            "PitInTime": [pd.NaT, pd.NaT, pd.Timedelta(seconds=1), pd.NaT, pd.NaT],
        }
    )


def _clean_feature_laps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024] * 4,
            "event": ["Monza"] * 4,
            "event_slug": ["monza"] * 4,
            "session": ["FP2"] * 4,
            "session_slug": ["fp2"] * 4,
            "driver": ["NOR", "NOR", "VER", "BOT"],
            "team": ["McLaren", "McLaren", "Red Bull Racing", "Kick Sauber"],
            "lap_time_sec": [80.0, 82.5, 84.0, 85.0],
            "sector1_time_sec": [25.0, 25.5, 26.0, 26.5],
            "sector2_time_sec": [30.0, 30.5, 31.0, 31.5],
            "sector3_time_sec": [25.0, 24.5, 27.0, 27.0],
            "compound": ["SOFT", "SOFT", "MEDIUM", "UNKNOWN"],
            "tyre_life": [2.0, 3.0, 2.0, 4.0],
            "is_valid_lap": [True, True, True, True],
            "is_push_lap": [False] * 4,
        }
    )
