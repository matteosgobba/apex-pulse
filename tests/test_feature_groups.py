import pandas as pd

from f1_prediction.modeling.feature_groups import get_feature_groups
from f1_prediction.modeling.splits import get_numeric_feature_columns


def test_feature_groups_exclude_identifiers_targets_and_qualifying_columns() -> None:
    dataset = _dataset()
    groups = get_feature_groups(dataset)

    excluded = {"season", "event", "driver", "team", "quali_gap_to_pole_sec"}
    assert not excluded.intersection(groups["all_features"])
    assert "quali_current_leak" not in groups["all_features"]


def test_relative_historical_and_quality_patterns_are_classified() -> None:
    groups = get_feature_groups(_dataset())

    assert "fp1_best_push_gap_to_teammate_sec" in groups["relative_features"]
    assert "fp1_team_best_push_lap_time_sec" in groups["relative_features"]
    assert "fp1_best_push_gap_pct_to_session_best" in groups["relative_features"]
    assert "driver_rolling3_quali_gap_mean" in groups["historical_features"]
    assert "team_expanding_q3_rate" in groups["historical_features"]
    assert "driver_prev_events_count" in groups["historical_features"]
    assert "practice_signal_quality_score" in groups["data_quality_features"]
    assert "latest_best_push_gap_to_session_best_is_extreme" in groups["data_quality_features"]


def test_all_features_matches_existing_numeric_safe_helper() -> None:
    dataset = _dataset()

    assert get_feature_groups(dataset)["all_features"] == get_numeric_feature_columns(dataset)


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024],
            "event": ["Monza"],
            "event_slug": ["monza"],
            "checkpoint": ["after_fp1"],
            "driver": ["NOR"],
            "team": ["McLaren"],
            "fp1_best_push_lap_time_sec": [80.0],
            "fp1_n_push_laps": [4],
            "fp1_best_push_gap_to_teammate_sec": [0.1],
            "fp1_team_best_push_lap_time_sec": [79.9],
            "fp1_best_push_gap_pct_to_session_best": [0.002],
            "driver_rolling3_quali_gap_mean": [0.3],
            "team_expanding_q3_rate": [0.5],
            "driver_prev_events_count": [3],
            "practice_signal_quality_score": [6],
            "latest_best_push_gap_to_session_best_is_extreme": [False],
            "quali_current_leak": [999.0],
            "quali_position": [1],
            "quali_best_lap_time_sec": [79.0],
            "quali_gap_to_pole_sec": [0.0],
            "reached_q2": [1],
            "reached_q3": [1],
        }
    )
