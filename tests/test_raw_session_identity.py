import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from f1_prediction.cli import app
from f1_prediction.config import DataConfig
from f1_prediction.data.raw_session_identity import (
    create_raw_session_identity_validation_report,
    validate_raw_session_identity,
)


def test_matching_raw_path_and_metadata_identity_passes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_raw_q(config, 2026, "Bahrain")

    result = validate_raw_session_identity(config, season=2026, event="Bahrain", session="Q")

    assert result.identity_status == "identity_verified"
    assert result.identity_match is True
    assert result.blocking is False


def test_great_britain_path_with_austrian_metadata_is_known_legacy_mismatch(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_raw_q(config, 2026, "Great Britain", metadata_event_name="Austrian Grand Prix")

    result = validate_raw_session_identity(
        config,
        season=2026,
        event="Great Britain",
        session="Q",
    )

    assert result.identity_status == "legacy_known_mismatch"
    assert result.blocking is True
    assert result.quarantined is True


def test_alias_normalization_allows_sakhir_and_bahrain(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_raw_q(config, 2026, "Sakhir", metadata_event_name="Bahrain Grand Prix")

    result = validate_raw_session_identity(config, season=2026, event="Sakhir", session="Q")

    assert result.identity_status == "identity_verified"
    assert result.identity_match is True


def test_wrong_season_fails(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_raw_q(config, 2026, "Bahrain", metadata_season=2025)

    result = validate_raw_session_identity(config, season=2026, event="Bahrain", session="Q")

    assert result.identity_status == "season_mismatch"
    assert result.blocking is True


def test_wrong_session_fails(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_raw_q(config, 2026, "Bahrain", metadata_session_slug="fp3")

    result = validate_raw_session_identity(config, season=2026, event="Bahrain", session="Q")

    assert result.identity_status == "session_mismatch"
    assert result.blocking is True


def test_australia_remains_legacy_noncanonical_but_identity_verified(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_raw_q(config, 2026, "Australia", metadata_event_name="Australian Grand Prix")

    result = validate_raw_session_identity(config, season=2026, event="Australia", session="Q")

    assert result.identity_status == "identity_verified"
    assert result.legacy_noncanonical is True
    assert result.quarantined_for_prospective_evidence is True


def test_report_writes_quarantine_artifacts_for_australia_and_great_britain(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_raw_q(config, 2026, "Australia", metadata_event_name="Australian Grand Prix")
    _write_raw_q(config, 2026, "Great Britain", metadata_event_name="Austrian Grand Prix")

    create_raw_session_identity_validation_report(
        config,
        season=2026,
        event="Great Britain",
        session="Q",
    )
    quarantine = pd.read_csv(config.metrics_output_dir / "raw_session_identity_quarantine.csv")
    by_slug = {row.event_slug: row for row in quarantine.itertuples(index=False)}

    assert by_slug["great-britain"].identity_status == "legacy_known_mismatch"
    assert bool(by_slug["great-britain"].quarantined) is True
    assert by_slug["australia"].identity_status == "identity_verified"
    assert bool(by_slug["australia"].legacy_noncanonical) is True
    assert bool(by_slug["australia"].quarantined_for_prospective_evidence) is True
    for path in config.metrics_output_dir.glob("raw_session_identity_*"):
        assert str(tmp_path) not in path.read_text(encoding="utf-8", errors="ignore")


def test_cli_raw_session_identity_validate(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_raw_q(config, 2026, "Great Britain", metadata_event_name="Austrian Grand Prix")
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)

    result = CliRunner().invoke(
        app,
        [
            "raw-session-identity-validate",
            "--season",
            "2026",
            "--event",
            "Great Britain",
            "--session",
            "Q",
        ],
    )

    assert result.exit_code == 0
    assert "Status: legacy_known_mismatch" in result.output
    assert "target onboarding allowed: false" in result.output.lower()


def _config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        project_root=tmp_path,
        fastf1_cache_dir=tmp_path / "cache",
        lap_output_dir=tmp_path / "data/raw/laps",
        session_metadata_output_dir=tmp_path / "data/raw/session_metadata",
        clean_lap_output_dir=tmp_path / "data/interim/clean_laps",
        session_features_output_dir=tmp_path / "data/processed/session_features",
        modeling_output_dir=tmp_path / "data/processed/modeling",
        metrics_output_dir=tmp_path / "reports/metrics",
    )


def _write_raw_q(
    config: DataConfig,
    season: int,
    event: str,
    *,
    metadata_event_name: str | None = None,
    metadata_season: int | None = None,
    metadata_session_slug: str = "q",
) -> None:
    slug = event.lower().replace(" ", "-")
    laps_path = config.lap_output_dir / str(season) / slug / "q_laps.parquet"
    laps_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Driver": ["NOR"],
            "Team": ["McLaren"],
            "LapNumber": [1.0],
            "LapTime": [pd.Timedelta(seconds=79)],
        }
    ).to_parquet(laps_path, index=False)
    metadata_path = config.session_metadata_output_dir / str(season) / slug / "q_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "season": metadata_season or season,
                "event_input": event,
                "event_name": metadata_event_name or f"{event} Grand Prix",
                "event_slug": slug,
                "session_input": "Q" if metadata_session_slug == "q" else "FP3",
                "session_name": "Qualifying" if metadata_session_slug == "q" else "Practice 3",
                "session_slug": metadata_session_slug,
                "status": "success",
            }
        ),
        encoding="utf-8",
    )
