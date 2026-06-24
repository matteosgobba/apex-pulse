from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.data.season_builder import (
    CONVENTIONAL_2023_EVENTS,
    CONVENTIONAL_2024_EVENTS,
    CONVENTIONAL_2025_EVENTS,
    SPRINT_OR_NONSTANDARD_2025_EXCLUSIONS,
    FailedEventBuild,
    SeasonDatasetBuildSummary,
    SuccessfulEventBuild,
    build_combined_dataset_path,
    build_dataset_report_payload,
    combine_event_dataset_files,
    combine_event_datasets,
    discover_season_events,
    resolve_event_selection,
)


def test_discover_season_events_filters_requested_event(monkeypatch) -> None:
    schedule = pd.DataFrame(
        {
            "RoundNumber": [1, 16],
            "Location": ["Bahrain", "Monza"],
            "EventName": ["Bahrain Grand Prix", "Italian Grand Prix"],
        }
    )
    monkeypatch.setattr(
        "f1_prediction.data.season_builder.fastf1.get_event_schedule",
        lambda season, include_testing=False: schedule,
    )

    events = discover_season_events(2024, requested_events=("Monza",))

    assert len(events) == 1
    assert events[0].season == 2024
    assert events[0].event == "Monza"
    assert events[0].event_name == "Italian Grand Prix"


def test_discover_season_events_accepts_country_event_name_location_and_aliases(
    monkeypatch,
) -> None:
    schedule = pd.DataFrame(
        {
            "RoundNumber": [1, 3, 24],
            "Country": ["Bahrain", "Australia", "United Arab Emirates"],
            "Location": ["Sakhir", "Melbourne", "Yas Island"],
            "EventName": [
                "Bahrain Grand Prix",
                "Australian Grand Prix",
                "Abu Dhabi Grand Prix",
            ],
            "OfficialEventName": ["Official Bahrain", "Official Australia", "Official Abu Dhabi"],
        }
    )
    monkeypatch.setattr(
        "f1_prediction.data.season_builder.fastf1.get_event_schedule",
        lambda season, include_testing=False: schedule,
    )

    events = discover_season_events(
        2024,
        requested_events=("Bahrain", "Australian Grand Prix", "Abu_Dhabi"),
    )

    assert [event.event for event in events] == ["Sakhir", "Melbourne", "Yas Island"]


def test_conventional_preset_resolves_to_non_empty_list() -> None:
    events = resolve_event_selection([2024], preset="conventional_2024")

    assert events == CONVENTIONAL_2024_EVENTS
    assert "Monza" in events


def test_conventional_2023_preset_resolves_to_non_empty_list() -> None:
    events = resolve_event_selection([2023], preset="conventional_2023")

    assert events == CONVENTIONAL_2023_EVENTS
    assert "Bahrain" in events


def test_conventional_2025_preset_resolves_to_documented_non_sprint_events() -> None:
    events = resolve_event_selection([2025], preset="conventional_2025")

    assert events == CONVENTIONAL_2025_EVENTS
    assert "Australia" in events
    assert "Monza" in events
    assert "Abu Dhabi" in events
    assert not set(events) & set(SPRINT_OR_NONSTANDARD_2025_EXCLUSIONS)


def test_multi_season_conventional_preset_resolves_each_season() -> None:
    events = resolve_event_selection([2023, 2024, 2025], preset="conventional")

    assert isinstance(events, dict)
    assert events[2023] == CONVENTIONAL_2023_EVENTS
    assert events[2024] == CONVENTIONAL_2024_EVENTS
    assert events[2025] == CONVENTIONAL_2025_EVENTS


def test_explicit_events_resolve_without_preset() -> None:
    assert resolve_event_selection([2024], events=["Monza", "Japan"]) == ("Monza", "Japan")


def test_invalid_preset_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown event preset"):
        resolve_event_selection([2024], preset="unknown")


def test_combine_event_datasets_concatenates_and_orders_synthetic_events() -> None:
    spa = _event_frame("Spa", 14)
    monza = _event_frame("Monza", 16)

    combined = combine_event_datasets((spa, monza))

    assert len(combined) == 6
    assert combined["event"].tolist()[:3] == ["Spa"] * 3
    assert combined["checkpoint"].tolist()[:3] == [
        "after_fp1",
        "after_fp2",
        "after_fp3",
    ]


def test_combine_event_dataset_files_reads_synthetic_parquets(tmp_path: Path) -> None:
    spa_path = tmp_path / "spa.parquet"
    monza_path = tmp_path / "monza.parquet"
    _event_frame("Spa", 14).to_parquet(spa_path, index=False)
    _event_frame("Monza", 16).to_parquet(monza_path, index=False)

    combined = combine_event_dataset_files((spa_path, monza_path))

    assert len(combined) == 6
    assert set(combined["event"]) == {"Spa", "Monza"}


def test_combined_dataset_adds_historical_features_when_targets_are_available() -> None:
    first = _history_event_frame("Bahrain", 1, 0.2)
    second = _history_event_frame("Monza", 16, 0.4)

    combined = combine_event_datasets((first, second))
    monza = combined[combined["event"].eq("Monza")].iloc[0]

    assert "driver_rolling3_quali_gap_mean" in combined
    assert monza["driver_rolling3_quali_gap_mean"] == pytest.approx(0.2)


def test_dataset_build_report_contains_required_structure(tmp_path: Path) -> None:
    output_path = build_combined_dataset_path(tmp_path / "modeling")
    summary = SeasonDatasetBuildSummary(
        requested_seasons=(2023, 2024),
        n_events_requested=2,
        successful_events=(
            SuccessfulEventBuild(2024, "Monza", "Italian Grand Prix", 60, "event.parquet"),
        ),
        failed_events=(
            FailedEventBuild(2023, "Spa", "Belgian Grand Prix", "RuntimeError: failed"),
        ),
        n_rows=60,
        n_drivers=20,
        n_teams=10,
        checkpoints=("after_fp1", "after_fp2", "after_fp3"),
        output_path=output_path,
        report_path=tmp_path / "metrics/dataset_build_report.json",
        n_columns=90,
        rows_by_checkpoint=(("after_fp1", 20), ("after_fp2", 20), ("after_fp3", 20)),
        events_by_checkpoint=(("after_fp1", 1), ("after_fp2", 1), ("after_fp3", 1)),
        rows_by_season=(("2024", 60),),
        rows_by_event=(("2024/monza", 60),),
        preset="conventional_2024",
    )

    report = build_dataset_report_payload(summary, tmp_path)

    assert report["requested_seasons"] == [2023, 2024]
    assert report["n_events_requested"] == 2
    assert report["n_events_successful"] == 1
    assert report["n_events_failed"] == 1
    assert report["n_rows"] == 60
    assert report["n_drivers"] == 20
    assert report["n_teams"] == 10
    assert report["n_columns"] == 90
    assert report["rows_by_checkpoint"]["after_fp1"] == 20
    assert report["events_by_checkpoint"]["after_fp3"] == 1
    assert report["rows_by_season"]["2024"] == 60
    assert report["rows_by_event"]["2024/monza"] == 60
    assert report["preset"] == "conventional_2024"
    assert report["failed_events"][0]["error_message"] == "RuntimeError: failed"
    assert report["output_path"] == "modeling/combined/modeling_dataset.parquet"
    assert isinstance(report["created_at_utc"], str)


def _event_frame(event: str, round_number: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024] * 3,
            "event": [event] * 3,
            "event_slug": [event.lower()] * 3,
            "event_order": [round_number] * 3,
            "checkpoint": ["after_fp3", "after_fp1", "after_fp2"],
            "driver": ["NOR"] * 3,
            "quali_position": [round_number] * 3,
        }
    )


def _history_event_frame(event: str, round_number: int, gap: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024],
            "event": [event],
            "event_slug": [event.lower()],
            "event_order": [round_number],
            "checkpoint": ["after_fp1"],
            "driver": ["NOR"],
            "team": ["McLaren"],
            "quali_position": [1],
            "quali_gap_to_pole_sec": [gap],
            "reached_q3": [1],
        }
    )
