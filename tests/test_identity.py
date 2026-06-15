import pandas as pd
import pytest

from f1_prediction.data.identity import (
    add_identity_columns,
    normalize_driver_key,
    normalize_team_key,
)
from f1_prediction.features.historical_features import add_historical_features
from f1_prediction.features.relative_features import add_relative_practice_features


@pytest.mark.parametrize(
    ("team_name", "expected"),
    [
        ("Oracle Red Bull Racing", "red_bull"),
        ("Red Bull Racing", "red_bull"),
        ("Scuderia Ferrari", "ferrari"),
        ("Mercedes-AMG Petronas Formula One Team", "mercedes"),
        ("Stake F1 Team Kick Sauber", "sauber"),
        ("Alfa Romeo", "sauber"),
        ("Visa Cash App RB", "rb"),
        ("AlphaTauri", "rb"),
    ],
)
def test_known_team_aliases_map_to_stable_keys(team_name: str, expected: str) -> None:
    assert normalize_team_key(team_name) == expected


def test_unknown_team_falls_back_to_normalized_slug() -> None:
    assert normalize_team_key("Équipe Example Racing") == "equipe_example_racing"


def test_driver_key_is_stable_and_identity_preserves_raw_columns() -> None:
    raw = pd.DataFrame({"driver": ["VER"], "team": ["Oracle Red Bull Racing"], "value": [1]})

    normalized = add_identity_columns(raw)

    assert normalize_driver_key("ver") == "ver"
    assert normalized.loc[0, "driver"] == "VER"
    assert normalized.loc[0, "team"] == "Oracle Red Bull Racing"
    assert normalized.loc[0, "driver_code"] == "VER"
    assert normalized.loc[0, "driver_key"] == "ver"
    assert normalized.loc[0, "team_key"] == "red_bull"
    assert normalized.loc[0, "value"] == 1


def test_relative_features_group_team_aliases_by_normalized_key() -> None:
    features = add_relative_practice_features(
        pd.DataFrame(
            {
                "season": [2024, 2024],
                "event_slug": ["monza", "monza"],
                "session_slug": ["fp1", "fp1"],
                "driver": ["VER", "PER"],
                "team": ["Oracle Red Bull Racing", "Red Bull Racing"],
                "best_push_lap_time_sec": [80.0, 81.0],
                "best_valid_lap_time_sec": [80.0, 81.0],
                "theoretical_best_lap_time_sec": [79.5, 80.5],
            }
        )
    ).set_index("driver")

    assert features.loc["VER", "best_push_team_rank"] == 1
    assert features.loc["PER", "best_push_team_rank"] == 2
    assert features.loc["PER", "best_push_gap_to_teammate_sec"] == pytest.approx(1.0)


def test_historical_features_use_normalized_driver_and_team_keys() -> None:
    rows = []
    for event_order, (team, gap) in enumerate(
        (("Alfa Romeo", 0.4), ("Stake F1 Team Kick Sauber", 0.6)), start=1
    ):
        rows.append(
            {
                "season": 2023 + event_order - 1,
                "event": f"Event {event_order}",
                "event_slug": f"event-{event_order}",
                "event_order": event_order,
                "checkpoint": "after_fp1",
                "driver": "BOT",
                "team": team,
                "quali_gap_to_pole_sec": gap,
                "quali_position": 10,
                "reached_q3": 1,
            }
        )

    result = add_historical_features(pd.DataFrame(rows)).set_index("event_slug")

    assert result.loc["event-2", "driver_rolling3_quali_gap_mean"] == pytest.approx(0.4)
    assert result.loc["event-2", "team_rolling3_quali_gap_mean"] == pytest.approx(0.4)
    assert result.loc["event-2", "team_key"] == "sauber"
