import pandas as pd

from f1_prediction.features.data_quality import add_data_quality_features


def test_quality_flags_missing_latest_session_metrics() -> None:
    rows = _quality_rows()
    rows.loc[0, ["fp3_best_push_lap_time_sec", "fp3_theoretical_best_lap_time_sec"]] = pd.NA

    result = add_data_quality_features(rows).iloc[0]

    assert result["has_any_practice_time"]
    assert result["has_latest_checkpoint_time"]
    assert result["latest_available_session"] == "FP3"
    assert result["missing_latest_best_push_lap"]
    assert result["missing_latest_theoretical_best_lap"]


def test_quality_flags_extreme_latest_gaps() -> None:
    rows = _quality_rows()
    rows.loc[0, "fp3_best_push_gap_to_session_best_sec"] = 3.5

    result = add_data_quality_features(rows).iloc[0]

    assert result["latest_best_push_gap_to_session_best_is_extreme"]
    assert not result["latest_best_valid_gap_to_session_best_is_extreme"]


def test_practice_signal_quality_score_counts_six_transparent_signals() -> None:
    result = add_data_quality_features(_quality_rows()).iloc[0]

    assert result["practice_signal_quality_score"] == 6
    assert result["n_available_sessions"] == 3
    assert result["n_total_push_laps_available"] == 9
    assert result["n_total_valid_laps_available"] == 18


def _quality_rows() -> pd.DataFrame:
    row: dict[str, object] = {
        "checkpoint": "after_fp3",
        "driver": "NOR",
    }
    for session in ("fp1", "fp2", "fp3"):
        row.update(
            {
                f"{session}_best_push_lap_time_sec": 80.0,
                f"{session}_best_valid_lap_time_sec": 80.1,
                f"{session}_theoretical_best_lap_time_sec": 79.9,
                f"{session}_n_push_laps": 3,
                f"{session}_n_valid_laps": 6,
                f"{session}_best_push_gap_to_session_best_sec": 0.2,
                f"{session}_best_valid_gap_to_session_best_sec": 0.3,
                f"{session}_theoretical_best_gap_to_session_best_sec": 0.1,
                f"{session}_best_push_gap_to_teammate_sec": 0.1,
            }
        )
    return pd.DataFrame([row])
