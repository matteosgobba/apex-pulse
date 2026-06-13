import json
from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.config import DataConfig
from f1_prediction.data import ingest as ingest_module
from f1_prediction.data.fastf1_loader import FastF1SessionData, build_lap_output_path
from f1_prediction.data.ingest import (
    EventIngestionSummary,
    SessionIngestionResult,
    build_metadata_output_path,
    ingest_event,
    normalize_session_identifiers,
)
from f1_prediction.utils.paths import slugify


def test_event_and_session_slug_generation() -> None:
    assert slugify("Emilia-Romagna") == "emilia-romagna"
    assert slugify("FP2") == "fp2"


def test_output_paths_use_season_event_and_session_slugs(tmp_path: Path) -> None:
    laps_path = build_lap_output_path(tmp_path / "laps", 2024, "São Paulo", "FP1")
    metadata_path = build_metadata_output_path(
        tmp_path / "metadata",
        2024,
        "São Paulo",
        "FP1",
    )

    assert laps_path == tmp_path / "laps/2024/sao-paulo/fp1_laps.parquet"
    assert metadata_path == tmp_path / "metadata/2024/sao-paulo/fp1_metadata.json"
    assert laps_path.parent.is_dir()
    assert metadata_path.parent.is_dir()


def test_ingest_event_skips_complete_existing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _data_config(tmp_path)
    laps_path = build_lap_output_path(config.lap_output_dir, 2024, "Monza", "FP2")
    metadata_path = build_metadata_output_path(
        config.session_metadata_output_dir,
        2024,
        "Monza",
        "FP2",
    )
    laps_path.touch()
    metadata_path.write_text('{"status": "success"}\n', encoding="utf-8")

    def unexpected_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("FastF1 loader should not run for complete existing outputs")

    monkeypatch.setattr(ingest_module, "load_fastf1_session_data", unexpected_load)

    summary = ingest_event(2024, "Monza", config, sessions=("FP2",))

    assert summary.skipped_count == 1
    assert summary.failed_count == 0
    assert summary.results[0].status == "skipped"


def test_event_ingestion_summary_counts_statuses(tmp_path: Path) -> None:
    results = (
        _result(tmp_path, "FP1", "success"),
        _result(tmp_path, "FP2", "skipped"),
        _result(tmp_path, "FP3", "failed"),
    )
    summary = EventIngestionSummary(season=2024, event="Monza", results=results)

    assert summary.success_count == 1
    assert summary.skipped_count == 1
    assert summary.failed_count == 1


def test_ingest_event_force_reloads_complete_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _data_config(tmp_path)
    laps_path = build_lap_output_path(config.lap_output_dir, 2024, "Monza", "FP2")
    metadata_path = build_metadata_output_path(
        config.session_metadata_output_dir,
        2024,
        "Monza",
        "FP2",
    )
    laps_path.touch()
    metadata_path.write_text('{"status": "success"}\n', encoding="utf-8")
    monkeypatch.setattr(
        ingest_module,
        "load_fastf1_session_data",
        lambda **kwargs: _session_data(),
    )

    summary = ingest_event(2024, "Monza", config, sessions=("FP2",), force=True)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert summary.success_count == 1
    assert metadata["status"] == "success"
    assert metadata["n_laps"] == 1
    assert metadata["drivers"] == ["NOR"]


@pytest.mark.parametrize(("fail_fast", "expected_results"), [(False, 2), (True, 1)])
def test_ingest_event_failure_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_fast: bool,
    expected_results: int,
) -> None:
    config = _data_config(tmp_path)

    def failed_load(**kwargs: object) -> None:
        raise RuntimeError("synthetic FastF1 failure")

    monkeypatch.setattr(ingest_module, "load_fastf1_session_data", failed_load)

    summary = ingest_event(
        2024,
        "Monza",
        config,
        sessions=("FP1", "FP2"),
        fail_fast=fail_fast,
    )

    first_metadata = build_metadata_output_path(
        config.session_metadata_output_dir,
        2024,
        "Monza",
        "FP1",
    )
    metadata = json.loads(first_metadata.read_text(encoding="utf-8"))
    assert len(summary.results) == expected_results
    assert summary.failed_count == expected_results
    assert metadata["status"] == "failed"
    assert metadata["error_message"] == "RuntimeError: synthetic FastF1 failure"


def test_normalize_session_identifiers_uppercases_and_deduplicates() -> None:
    assert normalize_session_identifiers(("fp1", "FP2", "fp1")) == ("FP1", "FP2")


def _data_config(project_root: Path) -> DataConfig:
    return DataConfig(
        project_root=project_root,
        fastf1_cache_dir=project_root / "cache",
        lap_output_dir=project_root / "laps",
        session_metadata_output_dir=project_root / "metadata",
    )


def _result(
    tmp_path: Path,
    session: str,
    status: str,
) -> SessionIngestionResult:
    return SessionIngestionResult(
        session=session,
        status=status,  # type: ignore[arg-type]
        laps_path=tmp_path / f"{session}.parquet",
        metadata_path=tmp_path / f"{session}.json",
    )


def _session_data() -> FastF1SessionData:
    laps = pd.DataFrame(
        {
            "Driver": ["NOR"],
            "LapTime": [pd.Timedelta(seconds=80)],
            "LapNumber": [1.0],
            "Stint": [1.0],
            "Compound": ["SOFT"],
            "TyreLife": [1.0],
            "Sector1Time": [pd.Timedelta(seconds=25)],
            "Sector2Time": [pd.Timedelta(seconds=30)],
            "Sector3Time": [pd.Timedelta(seconds=25)],
            "IsAccurate": [True],
        }
    )
    return FastF1SessionData(
        season=2024,
        event_input="Monza",
        event_name="Italian Grand Prix",
        session_input="FP2",
        session_name="Practice 2",
        laps=laps,
        drivers=("NOR",),
    )
