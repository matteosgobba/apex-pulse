from pathlib import Path

from typer.testing import CliRunner

from f1_prediction.cli import app
from f1_prediction.config import DataConfig, FeatureConfig, PushLapConfig
from f1_prediction.data.ingest import EventIngestionSummary
from f1_prediction.features.build import SessionFeatureBuildSummary


def test_ingest_event_accepts_sessions_after_single_option(monkeypatch, tmp_path: Path) -> None:
    captured_sessions: list[str] = []
    config = DataConfig(
        project_root=tmp_path,
        fastf1_cache_dir=tmp_path / "cache",
        lap_output_dir=tmp_path / "laps",
        session_metadata_output_dir=tmp_path / "metadata",
        clean_lap_output_dir=tmp_path / "clean_laps",
        session_features_output_dir=tmp_path / "session_features",
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


def test_build_session_features_accepts_session_subset(monkeypatch, tmp_path: Path) -> None:
    captured_sessions: list[str] = []
    config = DataConfig(
        project_root=tmp_path,
        fastf1_cache_dir=tmp_path / "cache",
        lap_output_dir=tmp_path / "laps",
        session_metadata_output_dir=tmp_path / "metadata",
        clean_lap_output_dir=tmp_path / "clean_laps",
        session_features_output_dir=tmp_path / "session_features",
    )
    feature_config = FeatureConfig(push_lap=PushLapConfig(1.03, 1.07, ("SOFT", "MEDIUM", "HARD")))
    output_path = tmp_path / "session_features/2024/monza/practice_session_features.parquet"

    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)
    monkeypatch.setattr(
        "f1_prediction.cli.load_feature_config",
        lambda config_path=None, project_root=None: feature_config,
    )

    def fake_build(*, season, event, sessions, **kwargs):
        captured_sessions.extend(sessions)
        return SessionFeatureBuildSummary(
            season=season,
            event=event,
            sessions=tuple(sessions),
            clean_lap_paths=(),
            clean_lap_files_written=1,
            aggregate_rows=20,
            output_path=output_path,
        )

    monkeypatch.setattr("f1_prediction.cli.run_feature_build", fake_build)

    result = CliRunner().invoke(
        app,
        [
            "build-session-features",
            "--season",
            "2024",
            "--event",
            "Monza",
            "--sessions",
            "FP2",
        ],
    )

    assert result.exit_code == 0
    assert captured_sessions == ["FP2"]
    assert "Aggregate rows: 20" in result.output
