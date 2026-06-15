import pandas as pd
import pytest

from f1_prediction.features.historical_features import add_historical_features


def test_driver_and_team_rolling_features_use_only_prior_events() -> None:
    features = add_historical_features(_historical_dataset()).set_index(
        ["event_slug", "driver", "checkpoint"]
    )

    first = features.loc[("event-1", "NOR", "after_fp1")]
    second = features.loc[("event-2", "NOR", "after_fp1")]
    assert first["driver_prev_events_count"] == 0
    assert pd.isna(first["driver_rolling3_quali_gap_mean"])
    assert second["driver_prev_events_count"] == 1
    assert second["driver_rolling3_quali_gap_mean"] == pytest.approx(0.2)
    assert second["team_prev_events_count"] == 1
    assert second["team_rolling3_quali_gap_mean"] == pytest.approx(0.1)


def test_current_event_target_does_not_change_its_own_history() -> None:
    dataset = _historical_dataset()
    original = add_historical_features(dataset)
    changed = dataset.copy()
    changed.loc[changed["event_slug"].eq("event-2"), "quali_gap_to_pole_sec"] = 99.0
    rebuilt = add_historical_features(changed)

    columns = ["driver_rolling3_quali_gap_mean", "team_rolling3_quali_gap_mean"]
    original_event = original.loc[original["event_slug"].eq("event-2"), columns].reset_index(
        drop=True
    )
    rebuilt_event = rebuilt.loc[rebuilt["event_slug"].eq("event-2"), columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(original_event, rebuilt_event)


def test_future_event_target_does_not_change_earlier_history() -> None:
    dataset = _historical_dataset()
    original = add_historical_features(dataset)
    changed = dataset.copy()
    changed.loc[changed["event_slug"].eq("event-3"), "quali_gap_to_pole_sec"] = 99.0
    rebuilt = add_historical_features(changed)

    columns = ["driver_rolling3_quali_gap_mean", "team_rolling3_quali_gap_mean"]
    original_event = original.loc[original["event_slug"].eq("event-2"), columns].reset_index(
        drop=True
    )
    rebuilt_event = rebuilt.loc[rebuilt["event_slug"].eq("event-2"), columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(original_event, rebuilt_event)


def test_teammate_historical_gap_uses_prior_event_only() -> None:
    features = add_historical_features(_historical_dataset()).set_index(
        ["event_slug", "driver", "checkpoint"]
    )

    assert features.loc[
        ("event-2", "NOR", "after_fp1"),
        "driver_rolling3_gap_to_teammate_quali_mean",
    ] == pytest.approx(0.2)
    assert features.loc[
        ("event-2", "VER", "after_fp1"),
        "driver_rolling3_gap_to_teammate_quali_mean",
    ] == pytest.approx(-0.2)


def test_excluded_holdout_event_is_not_used_by_later_training_history() -> None:
    features = add_historical_features(
        _historical_dataset(),
        excluded_target_events={"2024/event-2"},
    ).set_index(["event_slug", "driver", "checkpoint"])

    event_three = features.loc[("event-3", "NOR", "after_fp1")]
    assert event_three["driver_prev_events_count"] == 1
    assert event_three["driver_rolling3_quali_gap_mean"] == pytest.approx(0.2)


def _historical_dataset() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    gaps = {
        "event-1": {"NOR": 0.2, "VER": 0.0},
        "event-2": {"NOR": 0.4, "VER": 0.1},
        "event-3": {"NOR": 0.3, "VER": 0.2},
    }
    for event_order, (event, driver_gaps) in enumerate(gaps.items(), start=1):
        for checkpoint in ("after_fp1", "after_fp2", "after_fp3"):
            for position, (driver, gap) in enumerate(driver_gaps.items(), start=1):
                rows.append(
                    {
                        "season": 2024,
                        "event": event.title(),
                        "event_slug": event,
                        "event_order": event_order,
                        "checkpoint": checkpoint,
                        "driver": driver,
                        "team": "Team A",
                        "quali_gap_to_pole_sec": gap,
                        "quali_position": position,
                        "reached_q3": 1,
                    }
                )
    return pd.DataFrame(rows)
