from pathlib import Path

from typer.testing import CliRunner

from f1_prediction.cli import app
from f1_prediction.config import DataConfig
from f1_prediction.data.ingest import EventIngestionSummary


def test_ingest_event_accepts_sessions_after_single_option(monkeypatch, tmp_path: Path) -> None:
    captured_sessions: list[str] = []
    config = DataConfig(
        project_root=tmp_path,
        fastf1_cache_dir=tmp_path / "cache",
        lap_output_dir=tmp_path / "laps",
        session_metadata_output_dir=tmp_path / "metadata",
    )

    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)

    def fake_ingestion(*, season, event, config, sessions, **kwargs):
        captured_sessions.extend(sessions)
        return EventIngestionSummary(season=season, event=event, results=())

    monkeypatch.setattr("f1_prediction.cli.run_event_ingestion", fake_ingestion)

    result = CliRunner().invoke(
        app,
        [
            "ingest-event",
            "--season",
            "2024",
            "--event",
            "Monza",
            "--sessions",
            "FP1",
            "FP2",
            "Q",
        ],
    )

    assert result.exit_code == 0
    assert captured_sessions == ["FP1", "FP2", "Q"]
