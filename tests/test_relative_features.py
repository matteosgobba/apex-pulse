import pandas as pd
import pytest

from f1_prediction.features.modeling_dataset import build_checkpoint_modeling_dataset
from f1_prediction.features.relative_features import add_relative_practice_features


def test_teammate_gaps_team_ranks_and_aggregates() -> None:
    features = add_relative_practice_features(_practice_features()).set_index("driver")

    assert features.loc["NOR", "best_push_gap_to_teammate_sec"] == pytest.approx(-1.0)
    assert features.loc["PIA", "best_push_gap_to_teammate_sec"] == pytest.approx(1.0)
    assert features.loc["NOR", "best_push_team_rank"] == 1
    assert features.loc["PIA", "best_push_team_rank"] == 2
    assert features.loc["NOR", "team_best_push_lap_time_sec"] == pytest.approx(80.0)
    assert features.loc["PIA", "driver_gap_to_team_best_push_sec"] == pytest.approx(1.0)


def test_teammate_gap_is_null_for_single_driver_team() -> None:
    features = add_relative_practice_features(_practice_features()).set_index("driver")

    assert pd.isna(features.loc["VER", "best_valid_gap_to_teammate_sec"])
    assert features.loc["VER", "best_valid_team_rank"] == 1


def test_session_relative_gap_rank_and_percentage() -> None:
    features = add_relative_practice_features(_practice_features()).set_index("driver")

    assert features.loc["NOR", "theoretical_best_gap_to_session_best_sec"] == 0.0
    assert features.loc["PIA", "theoretical_best_rank"] == 2
    assert features.loc["PIA", "best_push_gap_pct_to_session_best"] == pytest.approx(1 / 80)


def test_relative_features_do_not_use_qualifying_columns() -> None:
    clean = _practice_features()
    poisoned = clean.assign(quali_gap_to_pole_sec=[100.0, -100.0, 999.0])

    clean_result = add_relative_practice_features(clean)
    poisoned_result = add_relative_practice_features(poisoned)

    relative_columns = [
        column
        for column in clean_result
        if "teammate" in column or "team_" in column or "gap_pct" in column
    ]
    pd.testing.assert_frame_equal(
        clean_result[relative_columns],
        poisoned_result[relative_columns],
    )


def test_relative_features_appear_in_checkpoint_modeling_dataset() -> None:
    dataset = build_checkpoint_modeling_dataset(_all_session_features(), _targets())

    assert "fp1_best_push_gap_to_teammate_sec" in dataset
    assert "fp2_team_best_valid_lap_time_sec" in dataset
    assert "fp3_theoretical_best_gap_pct_to_session_best" in dataset
    assert not any("quali_leak" in column for column in dataset.columns)


def _practice_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024] * 3,
            "event": ["Monza"] * 3,
            "event_slug": ["monza"] * 3,
            "session": ["FP1"] * 3,
            "session_slug": ["fp1"] * 3,
            "driver": ["NOR", "PIA", "VER"],
            "team": ["McLaren", "McLaren", "Red Bull Racing"],
            "best_push_lap_time_sec": [80.0, 81.0, 82.0],
            "best_valid_lap_time_sec": [80.0, 81.0, 82.0],
            "theoretical_best_lap_time_sec": [79.5, 80.0, 81.0],
            "push_lap_rank": pd.Series([1, 2, 3], dtype="Int64"),
            "valid_lap_rank": pd.Series([1, 2, 3], dtype="Int64"),
        }
    )


def _all_session_features() -> pd.DataFrame:
    frames = []
    for session in ("FP1", "FP2", "FP3"):
        frame = _practice_features().copy()
        frame["session"] = session
        frame["session_slug"] = session.lower()
        frame["n_push_laps"] = 5
        frame["quali_leak"] = 999
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024] * 3,
            "event": ["Monza"] * 3,
            "event_slug": ["monza"] * 3,
            "driver": ["NOR", "PIA", "VER"],
            "quali_position": pd.Series([1, 2, 3], dtype="Int64"),
            "quali_best_lap_time_sec": [79.0, 79.1, 79.2],
            "quali_gap_to_pole_sec": [0.0, 0.1, 0.2],
            "reached_q2": [1, 1, 1],
            "reached_q3": [1, 1, 1],
        }
    )
