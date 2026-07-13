import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from f1_prediction.cli import app
from f1_prediction.config import DataConfig, FeatureConfig, PushLapConfig, load_model_config
from f1_prediction.data.fastf1_loader import build_lap_output_path
from f1_prediction.modeling.prospective_monitoring import create_prospective_monitoring_protocol
from f1_prediction.modeling.prospective_monitoring_rehearsal import (
    create_prospective_monitoring_rehearsal,
)


def test_complete_clean_synthetic_event_reaches_dashboard_published(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_practice_raw(config, 2026, "Synthetic Clean GP")
    _write_q_raw(config, 2026, "Synthetic Clean GP")

    summary = create_prospective_monitoring_rehearsal(
        config,
        load_model_config(),
        _features(),
        protocol_name="season_2026_v1",
        season=2026,
        event="Synthetic Clean GP",
        event_order=3,
    )

    payload = json.loads(summary.summary_path.read_text())
    stages = pd.read_csv(summary.stages_path)
    population = pd.read_csv(summary.driver_population_path)
    current = json.loads(
        (config.metrics_output_dir.parent / "dashboard/current_event.json").read_text()
    )
    historical = json.loads(
        (
            config.metrics_output_dir.parent / "dashboard/historical_monitoring_summary.json"
        ).read_text()
    )

    assert summary.status == "pass"
    assert payload["synthetic_rehearsal"] is True
    assert payload["valid_prospective_evidence"] is False
    assert stages.loc[stages["stage"].eq("dashboard_published"), "status"].iloc[0] == "complete"
    assert current["data"]["lifecycle"]["state"] == "no_event_available"
    assert historical["data"]["valid_prospective_monitoring"]["event_count"] == 0
    assert (
        historical["data"]["synthetic_rehearsal_records"][0]["event_identity"]["event_slug"]
        == "synthetic-clean-gp"
    )
    assert payload["driver_population_counts"]["feature_participant_count"] == 4
    assert payload["driver_population_counts"]["forecast_only_driver_count"] == 0
    assert len(population) == 4


def test_missing_fp3_blocks_before_preparation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_practice_raw(config, 2026, "Synthetic Clean GP", sessions=("FP1", "FP2"))
    _write_q_raw(config, 2026, "Synthetic Clean GP")

    summary = create_prospective_monitoring_rehearsal(
        config,
        load_model_config(),
        _features(),
        protocol_name="season_2026_v1",
        season=2026,
        event="Synthetic Clean GP",
        event_order=3,
    )

    stages = pd.read_csv(summary.stages_path)
    assert summary.status == "blocked"
    assert stages.loc[stages["stage"].eq("practice_artifacts_ready"), "status"].iloc[0] == (
        "blocked"
    )
    assert stages.loc[stages["stage"].eq("event_prepared"), "status"].iloc[0] == "not_started"
    assert not (
        tmp_path
        / "data/processed/monitoring/2026/synthetic-clean-gp/monitoring_fp3_features.parquet"
    ).exists()


def test_raw_q_mismatch_blocks_before_target_creation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_practice_raw(config, 2026, "Synthetic Clean GP")
    _write_q_raw(config, 2026, "Synthetic Clean GP", metadata_event_name="Wrong Grand Prix")

    summary = create_prospective_monitoring_rehearsal(
        config,
        load_model_config(),
        _features(),
        protocol_name="season_2026_v1",
        season=2026,
        event="Synthetic Clean GP",
        event_order=3,
    )

    stages = pd.read_csv(summary.stages_path)
    assert summary.status == "blocked"
    assert (
        stages.loc[
            stages["stage"].eq("raw_q_identity_verified"),
            "status",
        ].iloc[0]
        == "blocked"
    )
    assert stages.loc[stages["stage"].eq("targets_added"), "status"].iloc[0] == "not_started"
    assert not (
        tmp_path
        / "data/processed/monitoring/2026/synthetic-clean-gp/monitoring_qualifying_targets.parquet"
    ).exists()


def test_existing_forecast_blocks_rehearsal_overwrite(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_practice_raw(config, 2026, "Synthetic Clean GP")
    _write_q_raw(config, 2026, "Synthetic Clean GP")
    _write_existing_forecast(config, "Synthetic Clean GP")

    summary = create_prospective_monitoring_rehearsal(
        config,
        load_model_config(),
        _features(),
        protocol_name="season_2026_v1",
        season=2026,
        event="Synthetic Clean GP",
        event_order=3,
    )

    stages = pd.read_csv(summary.stages_path)
    assert summary.status == "blocked"
    assert stages.loc[stages["stage"].eq("preflight_ready"), "status"].iloc[0] == "blocked"
    forecasts = pd.read_parquet(
        config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    )
    assert len(forecasts) == 1


def test_rehearsal_reports_use_project_relative_paths(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_practice_raw(config, 2026, "Synthetic Clean GP", sessions=("FP1", "FP2"))

    summary = create_prospective_monitoring_rehearsal(
        config,
        load_model_config(),
        _features(),
        protocol_name="season_2026_v1",
        season=2026,
        event="Synthetic Clean GP",
        event_order=3,
    )

    for path in (
        summary.summary_path,
        summary.stages_path,
        summary.checks_path,
        summary.failures_path,
        summary.driver_population_path,
        summary.runbook_path,
    ):
        assert str(tmp_path) not in path.read_text(encoding="utf-8", errors="ignore")


def test_rehearsal_cli_registration_and_minimal_execution(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_practice_raw(config, 2026, "Synthetic Clean GP", sessions=("FP1", "FP2"))
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)
    monkeypatch.setattr(
        "f1_prediction.cli.load_model_config",
        lambda config_path=None, project_root=None: load_model_config(),
    )
    monkeypatch.setattr(
        "f1_prediction.cli.load_feature_config",
        lambda config_path=None, project_root=None: _features(),
    )

    result = CliRunner().invoke(
        app,
        [
            "prospective-monitoring-rehearsal",
            "--protocol-name",
            "season_2026_v1",
            "--season",
            "2026",
            "--event",
            "Synthetic Clean GP",
            "--event-order",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert "Prospective monitoring rehearsal complete" in result.output
    assert "Status: blocked" in result.output


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


def _features() -> FeatureConfig:
    return FeatureConfig(push_lap=PushLapConfig(1.03, 1.07, ("SOFT", "MEDIUM", "HARD")))


def _write_protocol(config: DataConfig) -> None:
    create_prospective_monitoring_protocol(
        config,
        load_model_config(),
        protocol_name="season_2026_v1",
        monitor_season=2026,
        train_seasons=(2023,),
        dataset_path=_write_historical_dataset(config),
    )


def _write_historical_dataset(config: DataConfig) -> Path:
    rows = []
    for order, slug in enumerate(("australia", "canada")):
        for driver_index, driver in enumerate(("VER", "NOR", "LEC", "HAM")):
            gap = driver_index * 0.2
            rows.append(
                {
                    "season": 2023,
                    "event": slug.title(),
                    "event_slug": slug,
                    "event_order": order,
                    "checkpoint": "after_fp3",
                    "driver": driver,
                    "driver_key": driver.lower(),
                    "team": f"Team {driver_index}",
                    "team_key": f"team_{driver_index}",
                    "quali_gap_to_pole_sec": gap,
                    "quali_position": driver_index + 1,
                    "quali_best_lap_time_sec": 79.0 + gap,
                    "reached_q2": 1,
                    "reached_q3": 1,
                    "fp3_best_push_lap_time_sec": 80.0 + gap,
                    "fp3_best_valid_lap_time_sec": 80.1 + gap,
                    "fp3_theoretical_best_lap_time_sec": 79.9 + gap,
                    "fp3_best_push_gap_to_session_best_sec": gap,
                    "fp3_best_valid_gap_to_session_best_sec": gap + 0.1,
                    "fp3_theoretical_best_gap_to_session_best_sec": max(gap - 0.1, 0),
                    "practice_signal_quality_score": 6,
                }
            )
    dataset = pd.DataFrame(rows)
    path = config.modeling_output_dir / "combined/modeling_dataset.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(path, index=False)
    return path


def _write_practice_raw(
    config: DataConfig,
    season: int,
    event: str,
    *,
    sessions: tuple[str, ...] = ("FP1", "FP2", "FP3"),
    drivers: tuple[str, ...] = ("VER", "NOR", "LEC", "HAM"),
) -> None:
    for index, session in enumerate(sessions, start=1):
        path = build_lap_output_path(config.lap_output_dir, season, event, session)
        _raw_laps(base_time=80.0 + index, drivers=drivers).to_parquet(path, index=False)
        _write_metadata(config, season, event, session)


def _write_q_raw(
    config: DataConfig,
    season: int,
    event: str,
    *,
    drivers: tuple[str, ...] = ("VER", "NOR", "LEC", "HAM"),
    metadata_event_name: str | None = None,
) -> None:
    path = build_lap_output_path(config.lap_output_dir, season, event, "Q")
    _raw_laps(base_time=79.0, drivers=drivers).to_parquet(path, index=False)
    _write_metadata(config, season, event, "Q", metadata_event_name=metadata_event_name)


def _write_metadata(
    config: DataConfig,
    season: int,
    event: str,
    session: str,
    *,
    metadata_event_name: str | None = None,
) -> None:
    slug = event.strip().lower().replace(" ", "-")
    session_slug = session.lower()
    path = config.session_metadata_output_dir / str(season) / slug / f"{session_slug}_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "season": season,
                "event_input": event,
                "event_name": metadata_event_name or f"{event} Grand Prix",
                "event_slug": slug,
                "session_input": session,
                "session_name": "Qualifying" if session == "Q" else session,
                "session_slug": session_slug,
                "status": "success",
            }
        ),
        encoding="utf-8",
    )


def _raw_laps(
    *,
    base_time: float,
    drivers: tuple[str, ...] = ("VER", "NOR", "LEC", "HAM"),
) -> pd.DataFrame:
    default_teams = ("Red Bull Racing", "McLaren", "Ferrari", "Mercedes")
    teams = tuple(default_teams[index % len(default_teams)] for index, _ in enumerate(drivers))
    path_placeholder = pd.NaT
    return pd.DataFrame(
        {
            "Driver": list(drivers),
            "Team": list(teams),
            "LapNumber": [1.0] * len(drivers),
            "Stint": [1.0] * len(drivers),
            "Compound": ["SOFT"] * len(drivers),
            "TyreLife": [2.0] * len(drivers),
            "LapTime": pd.to_timedelta(
                [base_time + i * 0.2 for i in range(len(drivers))],
                unit="s",
            ),
            "Sector1Time": pd.to_timedelta([25.0] * len(drivers), unit="s"),
            "Sector2Time": pd.to_timedelta([29.0] * len(drivers), unit="s"),
            "Sector3Time": pd.to_timedelta([25.0] * len(drivers), unit="s"),
            "IsAccurate": [True] * len(drivers),
            "Deleted": [False] * len(drivers),
            "PitOutTime": [path_placeholder] * len(drivers),
            "PitInTime": [path_placeholder] * len(drivers),
        }
    )


def _write_existing_forecast(config: DataConfig, event: str) -> None:
    slug = event.strip().lower().replace(" ", "-")
    path = config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "event_slug": slug,
                "driver": "NOR",
                "driver_key": "nor",
                "preflight_status": "ready_to_forecast",
            }
        ]
    ).to_parquet(path, index=False)
