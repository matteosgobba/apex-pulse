import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from f1_prediction.cli import app
from f1_prediction.config import DataConfig, FeatureConfig, PushLapConfig, load_model_config
from f1_prediction.data.fastf1_loader import build_lap_output_path
from f1_prediction.modeling.monitoring_data_integrity_audit import (
    MonitoringDataIntegrityAuditSummary,
    create_monitoring_data_integrity_audit,
)
from f1_prediction.modeling.monitoring_operations import (
    EVENT_ORDER_RESOLUTION_SOURCE,
    resolve_monitoring_event_schedule,
    run_monitoring_after_qualifying,
    run_monitoring_before_qualifying,
)
from f1_prediction.modeling.prospective_monitoring import (
    ProspectiveMonitoringSummary,
    create_prospective_monitoring_protocol,
)


@pytest.fixture(autouse=True)
def _fastf1_schedule_fixture(monkeypatch) -> None:
    monkeypatch.setattr(
        "f1_prediction.modeling.monitoring_operations.fastf1.get_event_schedule",
        lambda season, include_testing=False: _schedule_fixture(season),
    )


def test_belgian_grand_prix_resolves_to_calendar_round() -> None:
    resolution = resolve_monitoring_event_schedule(season=2026, event="Belgian Grand Prix")

    assert resolution.canonical_event == "Belgian Grand Prix"
    assert resolution.event_slug == "belgian-grand-prix"
    assert resolution.event_order == 12
    assert resolution.scheduled_event_date == "2026-07-19"
    assert resolution.event_order_resolution_source == EVENT_ORDER_RESOLUTION_SOURCE


def test_event_order_resolution_ignores_registry_and_legacy_or_synthetic_rows(
    tmp_path: Path,
) -> None:
    config = _configured_workspace(tmp_path, event="Belgian Grand Prix")
    registry_path = config.metrics_output_dir / "prospective_monitoring_event_registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "monitor_season": 2026,
                "event_order": 1,
                "event_slug": "australia",
            },
            {
                "protocol_name": "season_2026_v1",
                "monitor_season": 2026,
                "event_order": 9,
                "event_slug": "great-britain",
            },
            {
                "protocol_name": "season_2026_v1",
                "monitor_season": 2026,
                "event_order": 99,
                "event_slug": "synthetic-clean-gp",
            },
        ]
    ).to_csv(registry_path, index=False)

    summary = run_monitoring_before_qualifying(
        config,
        load_model_config(),
        _features(),
        season=2026,
        event="Belgian Grand Prix",
    )
    payload = json.loads(summary.summary_path.read_text())
    stages = pd.read_csv(summary.stages_path)

    assert summary.status == "pass"
    assert payload["event_order"] == 12
    assert payload["event_order_resolution_source"] == EVENT_ORDER_RESOLUTION_SOURCE
    assert stages["event_order"].dropna().astype(int).unique().tolist() == [12]


def test_only_strictly_earlier_monitoring_events_feed_later_forecast(tmp_path: Path) -> None:
    config = _configured_workspace(tmp_path, event="Monza")
    run_monitoring_before_qualifying(
        config,
        load_model_config(),
        _features(),
        season=2026,
        event="Monza",
    )
    _write_q_raw(config, 2026, "Monza")
    run_monitoring_after_qualifying(config, season=2026, event="Monza")
    _write_practice_raw(config, 2026, "Belgian Grand Prix")
    _write_entry_list(config, 2026, "Belgian Grand Prix")

    summary = run_monitoring_before_qualifying(
        config,
        load_model_config(),
        _features(),
        season=2026,
        event="Belgian Grand Prix",
    )
    forecasts = pd.read_parquet(
        config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    )
    belgium = forecasts[forecasts["event_slug"].astype(str).eq("belgian-grand-prix")]

    assert summary.status == "pass"
    assert belgium["event_order"].eq(12).all()
    assert belgium["prior_monitoring_event_orders"].eq("[3]").all()


def test_ambiguous_and_unknown_event_names_fail_before_ingestion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _configured_workspace(tmp_path)
    ambiguous_schedule = pd.concat(
        [
            _schedule_fixture(2026),
            pd.DataFrame(
                [
                    {
                        "RoundNumber": 14,
                        "EventName": "Belgian Grand Prix",
                        "Location": "Spa Test",
                        "Country": "Belgium",
                        "OfficialEventName": "DUPLICATE BELGIAN GRAND PRIX",
                        "EventDate": "2026-08-02",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    monkeypatch.setattr(
        "f1_prediction.modeling.monitoring_operations.fastf1.get_event_schedule",
        lambda season, include_testing=False: ambiguous_schedule,
    )

    def unexpected_ingest(*args, **kwargs):
        raise AssertionError("ingestion should not be reached")

    monkeypatch.setattr(
        "f1_prediction.modeling.monitoring_operations.ingest_event", unexpected_ingest
    )
    with pytest.raises(ValueError, match="ambiguous"):
        run_monitoring_before_qualifying(
            config,
            load_model_config(),
            _features(),
            season=2026,
            event="Belgian Grand Prix",
        )

    monkeypatch.setattr(
        "f1_prediction.modeling.monitoring_operations.fastf1.get_event_schedule",
        lambda season, include_testing=False: _schedule_fixture(season),
    )
    with pytest.raises(ValueError, match="Closest valid event names"):
        run_monitoring_before_qualifying(
            config,
            load_model_config(),
            _features(),
            season=2026,
            event="Atlantis Grand Prix",
        )


def test_complete_before_qualifying_workflow_succeeds(tmp_path: Path) -> None:
    config = _configured_workspace(tmp_path)

    summary = run_monitoring_before_qualifying(
        config,
        load_model_config(),
        _features(),
        season=2026,
        event="Monza",
    )

    current = _dashboard_current(config)
    assert summary.status == "pass"
    assert current["data"]["event_identity"]["event_slug"] == "monza"
    assert current["data"]["lifecycle"]["state"] in {
        "forecast_available",
        "awaiting_qualifying_targets",
    }
    assert (config.metrics_output_dir / "monitoring_before_qualifying_summary.json").is_file()


def test_complete_after_qualifying_workflow_succeeds(tmp_path: Path) -> None:
    config = _configured_workspace(tmp_path)
    run_monitoring_before_qualifying(
        config,
        load_model_config(),
        _features(),
        season=2026,
        event="Monza",
    )
    _write_q_raw(config, 2026, "Monza")

    summary = run_monitoring_after_qualifying(config, season=2026, event="Monza")

    payload = json.loads(summary.summary_path.read_text())
    current = _dashboard_current(config)
    assert summary.status == "pass"
    assert payload["settlement_denominator"] == 4
    assert current["data"]["event_identity"]["event_slug"] == "monza"
    assert current["data"]["lifecycle"]["state"] == "settled"


def test_before_qualifying_rerun_reuses_existing_forecast(tmp_path: Path) -> None:
    config = _configured_workspace(tmp_path)
    first = run_monitoring_before_qualifying(
        config,
        load_model_config(),
        _features(),
        season=2026,
        event="Monza",
    )
    before = _fingerprint(config.metrics_output_dir / "prospective_monitoring_forecasts.parquet")

    second = run_monitoring_before_qualifying(
        config,
        load_model_config(),
        _features(),
        season=2026,
        event="Monza",
    )

    assert first.status == "pass"
    assert second.status == "pass"
    assert (
        _fingerprint(config.metrics_output_dir / "prospective_monitoring_forecasts.parquet")
        == before
    )
    payload = json.loads(second.summary_path.read_text())
    assert payload["forecast_status"] == "forecast_reused"
    assert payload["forecast_driver_count"] == 4
    assert payload["eligible_driver_count"] == 4
    assert payload["driver_set_parity_status"] == "driver_set_parity_passed"


def test_after_qualifying_rerun_reuses_existing_settlement(tmp_path: Path) -> None:
    config = _configured_workspace(tmp_path)
    run_monitoring_before_qualifying(
        config,
        load_model_config(),
        _features(),
        season=2026,
        event="Monza",
    )
    _write_q_raw(config, 2026, "Monza")
    first = run_monitoring_after_qualifying(config, season=2026, event="Monza")
    before = _fingerprint(config.metrics_output_dir / "prospective_monitoring_settlements.parquet")

    second = run_monitoring_after_qualifying(config, season=2026, event="Monza")

    assert first.status == "pass"
    assert second.status == "pass"
    assert (
        _fingerprint(config.metrics_output_dir / "prospective_monitoring_settlements.parquet")
        == before
    )
    payload = json.loads(second.summary_path.read_text())
    current = _dashboard_current(config)
    assert payload["settlement_status"] == "settlement_reused"
    assert payload["forecast_driver_count"] == 4
    assert payload["eligible_driver_count"] == 4
    assert payload["actual_qualifying_driver_count"] == 4
    assert payload["evaluable_driver_count"] == 4
    assert payload["settlement_denominator"] == 4
    assert payload["forecast_coverage"] == "4/4"
    assert payload["forecast_coverage_ratio"] == 1.0
    assert payload["coverage_status"] == "full_coverage"
    assert payload["forecast_coverage"] == current["data"]["settlement_status"]["forecast_coverage"]


def test_blocked_before_qualifying_preserves_existing_dashboard_state(
    tmp_path: Path,
) -> None:
    config = _configured_workspace(tmp_path)
    run_monitoring_before_qualifying(
        config,
        load_model_config(),
        _features(),
        season=2026,
        event="Monza",
    )
    _write_q_raw(config, 2026, "Monza")
    run_monitoring_after_qualifying(config, season=2026, event="Monza")
    dashboard_path = config.metrics_output_dir.parent / "dashboard/current_event.json"
    dashboard_before = _fingerprint(dashboard_path)
    forecast_before = _fingerprint(
        config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    )
    settlement_before = _fingerprint(
        config.metrics_output_dir / "prospective_monitoring_settlements.parquet"
    )

    summary = run_monitoring_before_qualifying(
        config,
        load_model_config(),
        _features(),
        season=2026,
        event="Monza",
    )

    payload = json.loads(summary.summary_path.read_text())
    stages = pd.read_csv(summary.stages_path)
    current = _dashboard_current(config)
    audit = create_monitoring_data_integrity_audit(config)
    checks = pd.read_csv(audit.checks_path)
    dashboard_actual_check = checks[
        checks["event_slug"].astype(str).eq("monza")
        & checks["check_name"].astype(str).eq("dashboard_actual_values_match_settlement")
    ].iloc[0]

    assert summary.status == "blocked"
    assert payload["forecast_status"] == "forecast_reused"
    assert payload["dashboard_preservation_status"] == "preserved_existing_dashboard"
    assert payload["dashboard_current_event"] == "Monza"
    assert stages.loc[stages["stage"].eq("forecast_created"), "status"].iloc[0] == "blocked"
    assert _fingerprint(dashboard_path) == dashboard_before
    assert (
        _fingerprint(config.metrics_output_dir / "prospective_monitoring_forecasts.parquet")
        == forecast_before
    )
    assert (
        _fingerprint(config.metrics_output_dir / "prospective_monitoring_settlements.parquet")
        == settlement_before
    )
    assert current["data"]["event_identity"]["event_slug"] == "monza"
    assert current["data"]["lifecycle"]["state"] == "settled"
    assert dashboard_actual_check["status"] == "passed"
    assert "forecast_rows=4" in str(dashboard_actual_check["observed_value"])


def test_blocked_preflight_stops_before_forecast(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _configured_workspace(tmp_path)

    def blocked_preflight(*args, **kwargs):
        path = config.metrics_output_dir / "prospective_monitoring_preflight_summary.json"
        path.write_text(
            json.dumps({"status": "blocked", "forecast_allowed": False}),
            encoding="utf-8",
        )
        return ProspectiveMonitoringSummary(
            status="blocked",
            summary_path=path,
            table_paths=(),
        )

    monkeypatch.setattr(
        "f1_prediction.modeling.monitoring_operations.create_prospective_monitoring_preflight",
        blocked_preflight,
    )

    summary = run_monitoring_before_qualifying(
        config,
        load_model_config(),
        _features(),
        season=2026,
        event="Monza",
    )

    stages = pd.read_csv(summary.stages_path)
    assert summary.status == "blocked"
    assert stages.loc[stages["stage"].eq("preflight_ready"), "status"].iloc[0] == "blocked"
    assert stages.loc[stages["stage"].eq("forecast_created"), "status"].iloc[0] == "not_started"
    assert not (config.metrics_output_dir / "prospective_monitoring_forecasts.parquet").exists()


def test_inconsistent_registration_blocks(tmp_path: Path) -> None:
    config = _configured_workspace(tmp_path)
    run_monitoring_before_qualifying(
        config,
        load_model_config(),
        _features(),
        season=2026,
        event="Monza",
    )
    registry_path = config.metrics_output_dir / "prospective_monitoring_event_registry.csv"
    registry = pd.read_csv(registry_path)
    registry.loc[registry["event_slug"].eq("monza"), "event_order"] = 4
    registry.to_csv(registry_path, index=False)

    summary = run_monitoring_before_qualifying(
        config,
        load_model_config(),
        _features(),
        season=2026,
        event="Monza",
    )

    assert summary.status == "blocked"
    assert (
        "event_order=4"
        in json.loads(summary.summary_path.read_text())["recommended_operator_action"]
    )


def test_q_identity_mismatch_blocks_target_ingestion(tmp_path: Path) -> None:
    config = _configured_workspace(tmp_path)
    run_monitoring_before_qualifying(
        config,
        load_model_config(),
        _features(),
        season=2026,
        event="Monza",
    )
    _write_q_raw(config, 2026, "Monza", metadata_event_name="Wrong Grand Prix")

    summary = run_monitoring_after_qualifying(config, season=2026, event="Monza")

    stages = pd.read_csv(summary.stages_path)
    assert summary.status == "blocked"
    assert stages.loc[stages["stage"].eq("raw_q_identity_verified"), "status"].iloc[0] == "blocked"
    assert stages.loc[stages["stage"].eq("targets_added"), "status"].iloc[0] == "not_started"


def test_missing_forecast_blocks_after_qualifying(tmp_path: Path) -> None:
    config = _configured_workspace(tmp_path)
    _write_q_raw(config, 2026, "Monza")

    summary = run_monitoring_after_qualifying(config, season=2026, event="Monza")

    stages = pd.read_csv(summary.stages_path)
    assert summary.status == "blocked"
    assert stages.loc[stages["stage"].eq("targets_added"), "status"].iloc[0] == "blocked"


def test_synthetic_and_legacy_events_are_rejected(tmp_path: Path) -> None:
    config = _configured_workspace(tmp_path, event="Synthetic Clean GP")
    with pytest.raises(ValueError, match="Synthetic rehearsal events"):
        run_monitoring_before_qualifying(
            config,
            load_model_config(),
            _features(),
            season=2026,
            event="Synthetic Clean GP",
        )

    legacy_config = _configured_workspace(tmp_path / "legacy", event="Australia")
    with pytest.raises(ValueError, match="Legacy Australia"):
        run_monitoring_before_qualifying(
            legacy_config,
            load_model_config(),
            _features(),
            season=2026,
            event="Australia",
        )


def test_event_specific_integrity_passes_with_global_great_britain_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _configured_workspace(tmp_path)
    run_monitoring_before_qualifying(
        config,
        load_model_config(),
        _features(),
        season=2026,
        event="Monza",
    )
    _write_q_raw(config, 2026, "Monza")

    def global_great_britain_failure(config_arg):
        summary_path = (
            config_arg.metrics_output_dir / "monitoring_data_integrity_audit_summary.json"
        )
        checks_path = config_arg.metrics_output_dir / "monitoring_data_integrity_audit_checks.csv"
        failures_path = (
            config_arg.metrics_output_dir / "monitoring_data_integrity_audit_failures.csv"
        )
        population_path = (
            config_arg.metrics_output_dir / "monitoring_data_integrity_event_driver_population.csv"
        )
        event_comparison_path = (
            config_arg.metrics_output_dir / "monitoring_data_integrity_event_comparison.csv"
        )
        runbook_path = config_arg.metrics_output_dir / "monitoring_data_integrity_runbook.md"
        checks = pd.DataFrame(
            [
                {
                    "event_slug": "monza",
                    "check_name": "event_specific_clean",
                    "status": "passed",
                    "blocking": False,
                },
                {
                    "event_slug": "great-britain",
                    "check_name": "legacy_raw_q_mismatch_quarantined",
                    "status": "failed",
                    "blocking": True,
                },
            ]
        )
        checks.to_csv(checks_path, index=False)
        checks[checks["blocking"]].to_csv(failures_path, index=False)
        pd.DataFrame().to_csv(population_path, index=False)
        pd.DataFrame().to_csv(event_comparison_path, index=False)
        summary_path.write_text(json.dumps({"status": "fail"}), encoding="utf-8")
        runbook_path.write_text("global legacy failure", encoding="utf-8")
        return MonitoringDataIntegrityAuditSummary(
            status="fail",
            summary_path=summary_path,
            checks_path=checks_path,
            failures_path=failures_path,
            population_path=population_path,
            event_comparison_path=event_comparison_path,
            runbook_path=runbook_path,
            events_audited=2,
            events_with_blocking_integrity_failures=1,
            events_with_warnings=0,
            dashboard_safe_for_public_display=False,
            recommended_operator_action="Keep Great Britain quarantined.",
        )

    monkeypatch.setattr(
        "f1_prediction.modeling.monitoring_operations.create_monitoring_data_integrity_audit",
        global_great_britain_failure,
    )

    summary = run_monitoring_after_qualifying(config, season=2026, event="Monza")

    assert summary.status == "pass"
    assert json.loads(summary.summary_path.read_text())["event_specific_integrity_status"] == "pass"


def test_stage_reports_use_project_relative_paths(tmp_path: Path) -> None:
    config = _configured_workspace(tmp_path)
    summary = run_monitoring_before_qualifying(
        config,
        load_model_config(),
        _features(),
        season=2026,
        event="Monza",
    )

    assert str(tmp_path) not in summary.summary_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in summary.stages_path.read_text(encoding="utf-8")


def test_monitoring_workflow_cli_help_is_registered() -> None:
    runner = CliRunner()

    before = runner.invoke(app, ["monitoring-before-qualifying", "--help"])
    after = runner.invoke(app, ["monitoring-after-qualifying", "--help"])

    assert before.exit_code == 0
    assert after.exit_code == 0
    assert "monitoring-before-qualifying" in before.output
    assert "monitoring-after-qualifying" in after.output


def test_monitoring_workflow_cli_commands_run_against_local_fixture(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _configured_workspace(tmp_path)
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)
    monkeypatch.setattr(
        "f1_prediction.cli.load_model_config",
        lambda config_path=None, project_root=None: load_model_config(),
    )
    monkeypatch.setattr(
        "f1_prediction.cli.load_feature_config",
        lambda config_path=None, project_root=None: _features(),
    )
    runner = CliRunner()

    before = runner.invoke(
        app,
        [
            "monitoring-before-qualifying",
            "--season",
            "2026",
            "--event",
            "Monza",
        ],
    )
    _write_q_raw(config, 2026, "Monza")
    after = runner.invoke(
        app,
        [
            "monitoring-after-qualifying",
            "--season",
            "2026",
            "--event",
            "Monza",
        ],
    )

    assert before.exit_code == 0
    assert after.exit_code == 0
    assert "Status: pass" in before.output
    assert "Status: pass" in after.output


def _configured_workspace(tmp_path: Path, *, event: str = "Monza") -> DataConfig:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_practice_raw(config, 2026, event)
    _write_entry_list(config, 2026, event)
    return config


def _schedule_fixture(season: int) -> pd.DataFrame:
    assert season == 2026
    return pd.DataFrame(
        [
            {
                "RoundNumber": 1,
                "EventName": "Australian Grand Prix",
                "Location": "Melbourne",
                "Country": "Australia",
                "OfficialEventName": "FORMULA 1 AUSTRALIAN GRAND PRIX 2026",
                "EventDate": "2026-03-08",
            },
            {
                "RoundNumber": 3,
                "EventName": "Monza",
                "Location": "Monza",
                "Country": "Italy",
                "OfficialEventName": "FORMULA 1 MONZA GRAND PRIX 2026",
                "EventDate": "2026-03-29",
            },
            {
                "RoundNumber": 9,
                "EventName": "British Grand Prix",
                "Location": "Silverstone",
                "Country": "Great Britain",
                "OfficialEventName": "FORMULA 1 BRITISH GRAND PRIX 2026",
                "EventDate": "2026-07-05",
            },
            {
                "RoundNumber": 12,
                "EventName": "Belgian Grand Prix",
                "Location": "Spa-Francorchamps",
                "Country": "Belgium",
                "OfficialEventName": "FORMULA 1 BELGIAN GRAND PRIX 2026",
                "EventDate": "2026-07-19",
            },
            {
                "RoundNumber": 13,
                "EventName": "Hungarian Grand Prix",
                "Location": "Budapest",
                "Country": "Hungary",
                "OfficialEventName": "FORMULA 1 HUNGARIAN GRAND PRIX 2026",
                "EventDate": "2026-07-26",
            },
        ]
    )


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
) -> None:
    for index, session in enumerate(sessions, start=1):
        path = build_lap_output_path(config.lap_output_dir, season, event, session)
        _raw_laps(base_time=80.0 + index).to_parquet(path, index=False)
        _write_metadata(config, season, event, session)


def _write_q_raw(
    config: DataConfig,
    season: int,
    event: str,
    *,
    metadata_event_name: str | None = None,
) -> None:
    path = build_lap_output_path(config.lap_output_dir, season, event, "Q")
    _raw_laps(base_time=79.0).to_parquet(path, index=False)
    _write_metadata(config, season, event, "Q", metadata_event_name=metadata_event_name)


def _write_entry_list(
    config: DataConfig,
    season: int,
    event: str,
    *,
    drivers: tuple[str, ...] = ("VER", "NOR", "LEC", "HAM"),
    teams: tuple[str, ...] = ("Red Bull", "McLaren", "Ferrari", "Mercedes"),
) -> None:
    slug = event.strip().lower().replace(" ", "-")
    path = (
        config.project_root
        / "data/processed/monitoring"
        / str(season)
        / slug
        / "qualifying_entry_list.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "driver": list(drivers),
            "driver_number": [str(index + 1) for index in range(len(drivers))],
            "full_name": [f"{driver} Driver" for driver in drivers],
            "team": list(teams),
        }
    ).to_csv(path, index=False)


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


def _raw_laps(*, base_time: float) -> pd.DataFrame:
    drivers = ("VER", "NOR", "LEC", "HAM")
    teams = ("Red Bull Racing", "McLaren", "Ferrari", "Mercedes")
    return pd.DataFrame(
        {
            "Driver": list(drivers),
            "Team": list(teams),
            "LapNumber": [1.0] * len(drivers),
            "Stint": [1.0] * len(drivers),
            "Compound": ["SOFT"] * len(drivers),
            "TyreLife": [2.0] * len(drivers),
            "LapTime": pd.to_timedelta(
                [base_time + index * 0.2 for index in range(len(drivers))],
                unit="s",
            ),
            "Sector1Time": pd.to_timedelta([25.0] * len(drivers), unit="s"),
            "Sector2Time": pd.to_timedelta([29.0] * len(drivers), unit="s"),
            "Sector3Time": pd.to_timedelta([25.0] * len(drivers), unit="s"),
            "IsAccurate": [True] * len(drivers),
            "Deleted": [False] * len(drivers),
            "PitOutTime": [pd.NaT] * len(drivers),
            "PitInTime": [pd.NaT] * len(drivers),
        }
    )


def _dashboard_current(config: DataConfig) -> dict:
    return json.loads(
        (config.metrics_output_dir.parent / "dashboard/current_event.json").read_text(
            encoding="utf-8"
        )
    )


def _fingerprint(path: Path) -> bytes:
    return path.read_bytes()
