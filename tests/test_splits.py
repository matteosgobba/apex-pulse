import pandas as pd

from f1_prediction.modeling.splits import (
    SplitStrategy,
    create_dataset_split,
    get_numeric_feature_columns,
)


def test_event_holdout_has_no_event_overlap() -> None:
    dataset = _split_dataset()
    split = create_dataset_split(
        dataset,
        SplitStrategy.event_holdout,
        test_events=["Monza"],
    )

    assert set(split.metadata["train_events"]).isdisjoint(split.metadata["test_events"])
    assert split.metadata["test_events"] == ["2024/monza"]


def test_event_holdout_can_report_empty_training_scope() -> None:
    dataset = _split_dataset().query("event_slug == 'monza'").reset_index(drop=True)

    split = create_dataset_split(
        dataset,
        SplitStrategy.event_holdout,
        test_events=["Monza"],
    )

    assert split.metadata["train_rows"] == 0
    assert split.metadata["test_rows"] == 2


def test_season_holdout_has_no_season_overlap() -> None:
    split = create_dataset_split(
        _split_dataset(),
        SplitStrategy.season_holdout,
        test_seasons=[2024],
    )

    assert set(split.metadata["train_seasons"]).isdisjoint(split.metadata["test_seasons"])
    assert split.metadata["test_seasons"] == [2024]


def test_walk_forward_uses_only_prior_events() -> None:
    split = create_dataset_split(
        _split_dataset(),
        SplitStrategy.walk_forward,
        min_train_events=2,
    )

    ordered = ["2023/bahrain", "2023/monaco", "2024/monza"]
    for fold in split.metadata["folds"]:
        assert max(ordered.index(event) for event in fold["train_events"]) < ordered.index(
            fold["test_event"]
        )


def test_numeric_feature_selection_excludes_targets_identifiers_and_future_sessions() -> None:
    dataset = _split_dataset()

    features = get_numeric_feature_columns(dataset, "after_fp1")

    assert features == ["fp1_n_laps"]
    assert "quali_gap_to_pole_sec" not in features
    assert "season" not in features
    assert "fp2_n_laps" not in features


def _split_dataset() -> pd.DataFrame:
    rows = []
    for season, event, event_order in (
        (2023, "Bahrain", 1),
        (2023, "Monaco", 6),
        (2024, "Monza", 16),
    ):
        for driver_index, driver in enumerate(("NOR", "VER"), start=1):
            rows.append(
                {
                    "season": season,
                    "event": event,
                    "event_slug": event.lower(),
                    "event_order": event_order,
                    "checkpoint": "after_fp1",
                    "driver": driver,
                    "team": "Team",
                    "fp1_n_laps": 10 + driver_index,
                    "fp1_best_lap_compound": "SOFT",
                    "fp2_n_laps": 12 + driver_index,
                    "quali_position": driver_index,
                    "quali_best_lap_time_sec": 79.0 + driver_index,
                    "quali_gap_to_pole_sec": float(driver_index - 1),
                    "reached_q2": 1,
                    "reached_q3": 1,
                }
            )
    return pd.DataFrame(rows)
