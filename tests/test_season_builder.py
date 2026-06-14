from pathlib import Path

import pandas as pd

from f1_prediction.data.season_builder import (
    FailedEventBuild,
    SeasonDatasetBuildSummary,
    SuccessfulEventBuild,
    build_combined_dataset_path,
    build_dataset_report_payload,
    combine_event_dataset_files,
    combine_event_datasets,
    discover_season_events,
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


def test_combine_event_datasets_concatenates_and_orders_synthetic_events() -> None:
    spa = _event_frame("Spa", 14)
    monza = _event_frame("Monza", 16)

    combined = combine_event_datasets((spa, monza))

    assert len(combined) == 6
    assert combined["event"].tolist()[:3] == ["Monza"] * 3
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
        checkpoints=("after_fp1", "after_fp2", "after_fp3"),
        output_path=output_path,
        report_path=tmp_path / "metrics/dataset_build_report.json",
    )

    report = build_dataset_report_payload(summary, tmp_path)

    assert report["requested_seasons"] == [2023, 2024]
    assert report["n_events_requested"] == 2
    assert report["n_events_successful"] == 1
    assert report["n_events_failed"] == 1
    assert report["n_rows"] == 60
    assert report["n_drivers"] == 20
    assert report["output_path"] == "modeling/combined/modeling_dataset.parquet"
    assert isinstance(report["created_at_utc"], str)


def _event_frame(event: str, round_number: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024] * 3,
            "event": [event] * 3,
            "event_slug": [event.lower()] * 3,
            "checkpoint": ["after_fp3", "after_fp1", "after_fp2"],
            "driver": ["NOR"] * 3,
            "quali_position": [round_number] * 3,
        }
    )
