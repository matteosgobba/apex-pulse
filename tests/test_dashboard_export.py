import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from f1_prediction.cli import app
from f1_prediction.config import DataConfig
from f1_prediction.dashboard.export import export_dashboard_artifacts
from f1_prediction.dashboard.schema import SCHEMA_VERSION, validate_dashboard_artifact_file


def test_empty_workspace_exports_valid_empty_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)

    summary = export_dashboard_artifacts(config)
    current = _read_dashboard(config, "current_event.json")
    manifest = _read_dashboard(config, "dashboard_manifest.json")

    assert summary.status == "empty"
    assert len(summary.artifact_paths) == 7
    assert current["data"]["lifecycle"]["state"] == "no_event_available"
    assert manifest["data"]["event_count"] == 0
    assert all(path["schema_version"] == SCHEMA_VERSION for path in _all_dashboard_payloads(config))


def test_ready_preflight_exports_ready_to_forecast(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Bahrain", 1)])
    _write_preflight(config, "Bahrain", status="ready_to_forecast", forecast_allowed=True)

    export_dashboard_artifacts(config)
    current = _read_dashboard(config, "current_event.json")

    assert current["data"]["lifecycle"]["state"] == "ready_to_forecast"
    assert current["data"]["preflight"]["status"] == "ready_to_forecast"
    assert current["data"]["preflight"]["forecast_allowed"] is True


def test_optional_cached_schedule_is_exported_additively(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Bahrain", 1)])
    _write_cached_session_info(
        config,
        event="Bahrain",
        event_order=1,
        session_name="Practice 1",
        session_number=1,
        start=datetime(2026, 3, 6, 14, 30),
    )
    _write_cached_session_info(
        config,
        event="Bahrain",
        event_order=1,
        session_name="Qualifying",
        session_number=4,
        start=datetime(2026, 3, 7, 18, 0),
    )

    export_dashboard_artifacts(config)
    current = _read_dashboard(config, "current_event.json")
    practice = _read_dashboard(config, "event_practice_status.json")
    manifest = _read_dashboard(config, "dashboard_manifest.json")

    schedule = current["data"]["event_schedule"]
    assert schedule["available"] is True
    assert schedule["source"] == "fastf1_cached_session_info"
    assert schedule["location"] == "Sakhir"
    assert schedule["country"] == "Bahrain"
    assert schedule["circuit"] == "Bahrain International Circuit"
    assert [session["session"] for session in schedule["sessions"]] == ["FP1", "Q"]
    assert schedule["sessions"][0]["scheduled_start_utc"] == "2026-03-06T11:30:00+00:00"
    assert practice["data"]["event_schedule"] == schedule
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["data"]["dashboard_contract_capabilities"]["session_schedule"] is True


def test_missing_cached_schedule_preserves_backward_compatible_unavailable_state(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Bahrain", 1)])

    export_dashboard_artifacts(config)
    current = _read_dashboard(config, "current_event.json")

    assert current["schema_version"] == SCHEMA_VERSION
    assert current["data"]["event_schedule"] == {
        "available": False,
        "reason": "session_schedule_not_available",
        "value": None,
    }


def test_blocked_preflight_exports_blocked_reason(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Bahrain", 1)])
    _write_preflight(
        config,
        "Bahrain",
        status="blocked",
        forecast_allowed=False,
        blocking_check_count=2,
    )

    export_dashboard_artifacts(config)
    current = _read_dashboard(config, "current_event.json")

    assert current["data"]["lifecycle"]["state"] == "blocked"
    assert current["data"]["lifecycle"]["reason"] == "preflight_status_blocked"
    assert current["data"]["preflight"]["blocking_check_count"] == 2


def test_forecast_available_exports_ranked_leaderboard(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Bahrain", 1, target_artifact_present=True)])
    _write_forecasts(config, "Bahrain", [("VER", "Red Bull Racing", 0.3), ("NOR", "McLaren", 0.1)])

    export_dashboard_artifacts(config)
    current = _read_dashboard(config, "current_event.json")
    forecast = _read_dashboard(config, "event_forecast.json")

    assert current["data"]["lifecycle"]["state"] == "forecast_available"
    rows = forecast["data"]["leaderboard"]
    assert [row["driver_code"] for row in rows] == ["NOR", "VER"]
    assert [row["predicted_position"] for row in rows] == [1, 2]
    assert rows[0]["interval_available"] is False


def test_forecast_only_driver_is_exported_separately_from_primary_views(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Bahrain", 1, target_artifact_present=True)])
    _write_target_coverage(
        config,
        "Bahrain",
        [
            ("NOR", True, True, ""),
            ("ARO", False, False, "no_qualifying_lap_rows"),
        ],
    )
    _write_forecasts(
        config,
        "Bahrain",
        [("NOR", "McLaren", 0.1), ("ARO", "Sauber", 0.4)],
    )
    _write_settlements(
        config,
        "Bahrain",
        [
            ("NOR", 0.2, 0.1),
            ("ARO", None, 0.4, False, "no_qualifying_lap_rows"),
        ],
    )

    export_dashboard_artifacts(config)
    forecast = _read_dashboard(config, "event_forecast.json")
    settlement = _read_dashboard(config, "event_settlement.json")

    assert [row["driver_code"] for row in forecast["data"]["leaderboard"]] == ["NOR"]
    eligible_codes = [
        row["driver_code"] for row in forecast["data"]["qualifying_eligible_forecast_rows"]
    ]
    assert eligible_codes == ["NOR"]
    assert [row["driver_code"] for row in forecast["data"]["forecast_only_rows"]] == ["ARO"]
    assert forecast["data"]["forecast_only_rows"][0]["forecast_only_reason"] == (
        "no_qualifying_lap_rows"
    )
    assert [row["driver_code"] for row in settlement["data"]["driver_comparison"]] == ["NOR"]
    assert [row["driver_code"] for row in settlement["data"]["settlement_evaluable_rows"]] == [
        "NOR"
    ]
    assert [row["driver_code"] for row in settlement["data"]["forecast_only_rows"]] == ["ARO"]
    assert settlement["data"]["summary_metrics"]["settlement_evaluable_driver_count"] == 1
    assert settlement["data"]["summary_metrics"]["excluded_driver_count"] == 1


def test_settled_event_exports_driver_comparison_and_metrics(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Bahrain", 1, target_artifact_present=True)])
    _write_forecasts(config, "Bahrain", [("VER", "Red Bull Racing", 0.2), ("NOR", "McLaren", 0.5)])
    _write_settlements(config, "Bahrain", [("VER", 0.1, 0.1), ("NOR", 0.7, 0.2)])

    export_dashboard_artifacts(config)
    current = _read_dashboard(config, "current_event.json")
    settlement = _read_dashboard(config, "event_settlement.json")

    assert current["data"]["lifecycle"]["state"] == "settled"
    assert settlement["data"]["summary_metrics"]["driver_count"] == 2
    assert settlement["data"]["summary_metrics"]["mae_gap_sec"] == 0.24999999999999997
    assert settlement["data"]["summary_metrics"]["top_3_agreement"] == 1.0
    assert settlement["data"]["driver_comparison"][0]["actual_gap_to_pole_sec"] == 0.1


def test_partial_coverage_settled_event_exposes_missing_actual_entrant(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(
        config, [_registry_event("Hungarian Grand Prix", 14, target_artifact_present=True)]
    )
    _write_forecasts(
        config,
        "Hungarian Grand Prix",
        [("VER", "Red Bull Racing", 0.2), ("NOR", "McLaren", 0.5)],
    )
    _write_settlements(
        config,
        "Hungarian Grand Prix",
        [("VER", 0.1, 0.2), ("NOR", 0.7, 0.5)],
    )
    _write_targets(
        config,
        "Hungarian Grand Prix",
        [
            ("VER", "Red Bull Racing", 1, 0.1),
            ("NOR", "McLaren", 2, 0.7),
            ("PER", "Red Bull Racing", 3, 0.9),
        ],
    )

    export_dashboard_artifacts(config)
    current = _read_dashboard(config, "current_event.json")
    settlement = _read_dashboard(config, "event_settlement.json")

    assert current["data"]["lifecycle"]["state"] == "settled_partial_coverage"
    status = current["data"]["settlement_status"]
    assert status["forecast_coverage"] == "2/3"
    assert status["actual_qualifying_driver_count"] == 3
    assert status["coverage_warning"] == "partial qualifying-entry coverage"
    assert status["unforecasted_actual_entrants"] == [
        {"driver": "PER", "driver_code": "PER", "reason": "pre_q_entry_list_resolution_miss"}
    ]
    assert settlement["data"]["summary_metrics"]["scored_driver_count"] == 2
    assert settlement["data"]["summary_metrics"]["forecast_coverage_status"] == "partial_coverage"
    historical = _read_dashboard(config, "historical_monitoring_summary.json")
    history_event = historical["data"]["valid_prospective_monitoring"]["events"][0]
    assert history_event["forecast_checkpoint"] == "after_fp3"
    assert history_event["forecast_coverage"] == "2/3"
    assert len(history_event["forecast_rows"]) == 2
    assert len(history_event["comparison_rows"]) == 2
    assert history_event["unforecasted_actual_entrants"][0]["driver_code"] == "PER"


def test_missing_optional_interval_columns_do_not_break_exports(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Bahrain", 1, target_artifact_present=True)])
    _write_forecasts(config, "Bahrain", [("VER", "Red Bull Racing", 0.2)])
    _write_settlements(config, "Bahrain", [("VER", 0.4, 0.2)])

    export_dashboard_artifacts(config)
    forecast = _read_dashboard(config, "event_forecast.json")
    settlement = _read_dashboard(config, "event_settlement.json")

    assert forecast["data"]["summary"]["interval_availability_rate"] == 0.0
    assert settlement["data"]["interval_diagnostics"]["available"] is False


def test_forecast_without_targets_exports_awaiting_qualifying_targets(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Bahrain", 1, target_artifact_present=False)])
    _write_forecasts(config, "Bahrain", [("VER", "Red Bull Racing", 0.2)])

    export_dashboard_artifacts(config)
    current = _read_dashboard(config, "current_event.json")

    assert current["data"]["lifecycle"]["state"] == "awaiting_qualifying_targets"
    assert current["data"]["forecast_status"]["available"] is True
    assert current["data"]["settlement_status"]["available"] is False


def test_missing_optional_reports_produce_partial_model_summary(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Bahrain", 1)])

    summary = export_dashboard_artifacts(config)
    model = _read_dashboard(config, "model_summary.json")

    assert summary.status == "partial"
    assert model["status"] == "partial"
    assert model["data"]["backtest_summary"]["dataset_rows"] is None


def test_malformed_existing_source_json_produces_invalid_outputs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.metrics_output_dir.mkdir(parents=True)
    (config.metrics_output_dir / "prospective_monitoring_protocol.json").write_text("{bad")

    summary = export_dashboard_artifacts(config)
    manifest = _read_dashboard(config, "dashboard_manifest.json")

    assert summary.status == "invalid"
    assert manifest["status"] == "invalid"
    assert manifest["data"]["source_issues"]


def test_legacy_australia_and_great_britain_are_labeled_descriptive_only(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Australia", 1), _registry_event("Great Britain", 9)])
    _write_forecasts(config, "Australia", [("VER", "Red Bull Racing", 0.2)])
    _write_settlements(config, "Australia", [("VER", 0.3, 0.1)])
    _write_reconciliation(config, [("australia", 44, 1), ("great-britain", 44, 9)])

    export_dashboard_artifacts(config)
    historical = _read_dashboard(config, "historical_monitoring_summary.json")
    legacy = historical["data"]["legacy_descriptive_records"]

    assert {row["event_identity"]["event_slug"] for row in legacy} == {
        "australia",
        "great-britain",
    }
    assert all(row["legacy_noncanonical"] is True for row in legacy)
    assert all(row["eligible_for_valid_prospective_evidence"] is False for row in legacy)


def test_legacy_only_records_do_not_become_default_current_event(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Australia", 1), _registry_event("Great Britain", 9)])
    _write_forecasts(config, "Great Britain", [("NOR", "McLaren", 0.2)])
    _write_settlements(config, "Great Britain", [("NOR", 0.3, 0.2)])
    _write_reconciliation(config, [("australia", 44, 1), ("great-britain", 44, 9)])

    export_dashboard_artifacts(config)
    current = _read_dashboard(config, "current_event.json")
    manifest = _read_dashboard(config, "dashboard_manifest.json")
    historical = _read_dashboard(config, "historical_monitoring_summary.json")

    assert current["data"]["lifecycle"]["state"] == "no_event_available"
    assert manifest["data"]["current_event_reference"]["available"] is False
    assert manifest["data"]["current_event_reference"]["reason"] == "no_event_available"
    legacy_slugs = {
        row["event_identity"]["event_slug"]
        for row in historical["data"]["legacy_descriptive_records"]
    }
    assert legacy_slugs == {
        "australia",
        "great-britain",
    }


def test_legacy_records_are_excluded_from_valid_aggregates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Australia", 1)])
    _write_forecasts(config, "Australia", [("VER", "Red Bull Racing", 0.2)])
    _write_settlements(config, "Australia", [("VER", 0.3, 0.1)])
    _write_event_metrics(config, "Australia", 0.1)
    _write_reconciliation(config, [("australia", 44, 1)])

    export_dashboard_artifacts(config)
    historical = _read_dashboard(config, "historical_monitoring_summary.json")

    assert historical["data"]["valid_prospective_monitoring"]["event_count"] == 0
    assert (
        historical["data"]["valid_prospective_monitoring"]["aggregate_metrics"]["available"]
        is False
    )
    assert len(historical["data"]["legacy_descriptive_records"]) == 1


def test_clean_event_is_selected_ahead_of_legacy_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Australia", 1), _registry_event("Monza", 2)])
    _write_forecasts(config, "Australia", [("VER", "Red Bull Racing", 0.2)])
    _write_settlements(config, "Australia", [("VER", 0.3, 0.1)])
    _write_reconciliation(config, [("australia", 44, 1)])
    _write_preflight(config, "Monza", status="ready_to_forecast", forecast_allowed=True)

    export_dashboard_artifacts(config)
    current = _read_dashboard(config, "current_event.json")

    assert current["data"]["event_identity"]["event_slug"] == "monza"
    assert current["data"]["lifecycle"]["state"] == "ready_to_forecast"


def test_synthetic_rehearsal_current_event_is_excluded_from_valid_history(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(
        config,
        [_registry_event("Synthetic Clean GP", 3, target_artifact_present=True)],
    )
    _write_forecasts(config, "Synthetic Clean GP", [("NOR", "McLaren", 0.1)])
    _write_settlements(config, "Synthetic Clean GP", [("NOR", 0.2, 0.1)])
    _write_event_metrics(config, "Synthetic Clean GP", 0.1)

    export_dashboard_artifacts(config)
    current = _read_dashboard(config, "current_event.json")
    manifest = _read_dashboard(config, "dashboard_manifest.json")
    historical = _read_dashboard(config, "historical_monitoring_summary.json")

    assert current["data"]["lifecycle"]["state"] == "no_event_available"
    assert manifest["data"]["current_event_reference"]["available"] is False
    assert manifest["data"]["current_event_reference"]["reason"] == "no_event_available"
    assert manifest["data"]["eligible_prospective_event_count"] == 0
    assert manifest["data"]["synthetic_rehearsal_event_count"] == 1
    assert historical["data"]["valid_prospective_monitoring"]["event_count"] == 0
    assert (
        historical["data"]["synthetic_rehearsal_records"][0][
            "eligible_for_valid_prospective_evidence"
        ]
        is False
    )


def test_real_event_is_selected_ahead_of_newer_synthetic_rehearsal(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(
        config,
        [
            _registry_event("Monza", 2, target_artifact_present=True),
            _registry_event("Synthetic Clean GP", 9, target_artifact_present=True),
        ],
    )
    _write_forecasts(config, "Monza", [("NOR", "McLaren", 0.2)])
    _write_forecasts(config, "Synthetic Clean GP", [("PIA", "McLaren", 0.1)], append=True)

    export_dashboard_artifacts(config)
    current = _read_dashboard(config, "current_event.json")
    manifest = _read_dashboard(config, "dashboard_manifest.json")

    assert current["data"]["event_identity"]["event_slug"] == "monza"
    assert current["data"]["synthetic_rehearsal"] is False
    assert manifest["data"]["synthetic_rehearsal_event_count"] == 1


def test_event_selection_is_deterministic_and_registry_order_aware(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Bahrain", 2), _registry_event("Monza", 9)])
    _write_forecasts(config, "Bahrain", [("VER", "Red Bull Racing", 0.1)])
    _write_forecasts(config, "Monza", [("NOR", "McLaren", 0.2)], append=True)

    export_dashboard_artifacts(config)
    current = _read_dashboard(config, "current_event.json")

    assert current["data"]["event_identity"]["event_slug"] == "monza"
    assert current["data"]["event_identity"]["event_order"] == 9


def test_preflight_for_one_event_does_not_leak_into_another_event(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Bahrain", 1), _registry_event("Monza", 2)])
    _write_preflight(config, "Bahrain", status="blocked", forecast_allowed=False)

    export_dashboard_artifacts(config, event="Monza", season=2026)
    current = _read_dashboard(config, "current_event.json")

    assert current["data"]["event_identity"]["event_slug"] == "monza"
    assert current["data"]["lifecycle"]["state"] == "practice_in_progress"
    assert current["data"]["preflight"]["available"] is False


def test_source_paths_are_project_relative_and_no_absolute_paths_are_exported(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Bahrain", 1)])

    export_dashboard_artifacts(config)

    for payload in _all_dashboard_payloads(config):
        raw = json.dumps(payload)
        assert str(tmp_path) not in raw
        assert all(not Path(path).is_absolute() for path in payload["source_artifacts"])


def test_dashboard_export_carries_raw_identity_status_for_clean_event(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Bahrain", 1)])
    _write_raw_identity_checks(config, [("Bahrain", "identity_verified", True, False)])

    export_dashboard_artifacts(config)
    current = _read_dashboard(config, "current_event.json")

    assert current["data"]["raw_session_identity"]["raw_session_identity_status"] == (
        "identity_verified"
    )
    assert current["data"]["raw_session_identity"]["raw_session_identity_verified"] is True
    assert current["data"]["raw_session_identity"]["quarantine_status"] == "clear"


def test_dashboard_legacy_history_carries_great_britain_raw_mismatch(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Great Britain", 9)])
    _write_reconciliation(config, [("great-britain", 44, 9)])
    _write_raw_identity_checks(
        config,
        [("Great Britain", "legacy_known_mismatch", False, True)],
    )

    export_dashboard_artifacts(config)
    historical = _read_dashboard(config, "historical_monitoring_summary.json")
    legacy = historical["data"]["legacy_descriptive_records"][0]

    assert legacy["event_identity"]["event_slug"] == "great-britain"
    assert legacy["raw_session_identity_status"] == "legacy_known_mismatch"
    assert legacy["raw_source_mismatch"] is True
    assert legacy["quarantine_status"] == "quarantined"


def test_schema_version_and_envelope_validation_are_enforced(tmp_path: Path) -> None:
    config = _config(tmp_path)
    export_dashboard_artifacts(config)
    path = config.metrics_output_dir.parent / "dashboard/current_event.json"

    payload = validate_dashboard_artifact_file(path)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["artifact_type"] == "current_event"


def test_dashboard_export_cli_registration_and_minimal_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)

    result = CliRunner().invoke(app, ["dashboard-export"])

    assert result.exit_code == 0
    assert "Dashboard export completed" in result.output
    assert "Artifacts written: 7" in result.output


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


def _read_dashboard(config: DataConfig, filename: str) -> dict:
    return json.loads((config.metrics_output_dir.parent / "dashboard" / filename).read_text())


def _all_dashboard_payloads(config: DataConfig) -> list[dict]:
    return [
        json.loads(path.read_text())
        for path in sorted((config.metrics_output_dir.parent / "dashboard").glob("*.json"))
    ]


def _write_protocol(config: DataConfig) -> None:
    _write_json(
        config.metrics_output_dir / "prospective_monitoring_protocol.json",
        {
            "protocol_name": "season_2026_v1",
            "protocol_version": "1.0",
            "protocol_fingerprint": "abc123",
            "monitor_season": 2026,
            "train_seasons": [2023, 2024, 2025],
            "checkpoint": "after_fp3",
            "policy_recommendation": "season_aware_candidate_requires_more_evidence",
            "candidate_identity": {
                "family": "ablation",
                "model_name": "random_forest",
                "feature_group": "base_plus_relative",
                "temporal_weighting_policy": "current_season_only_with_prior",
            },
            "default_identity": {
                "family": "ablation",
                "model_name": "random_forest",
                "feature_group": "base_plus_relative",
                "temporal_weighting_policy": "uniform",
            },
            "uncertainty_configuration": {"method": "conformal_predicted_gap_bucket"},
        },
    )


def _registry_event(
    event: str,
    event_order: int,
    *,
    target_artifact_present: bool = False,
) -> dict:
    slug = event.lower().replace(" ", "-")
    event_dir = f"data/processed/monitoring/2026/{slug}"
    return {
        "protocol_name": "season_2026_v1",
        "monitor_season": 2026,
        "event_order": event_order,
        "event": event,
        "event_slug": slug,
        "checkpoint": "after_fp3",
        "forecast_status": "forecastable",
        "settlement_status": "settleable" if target_artifact_present else "targets_missing",
        "feature_artifact_path": f"{event_dir}/monitoring_fp3_features.parquet",
        "feature_artifact_fingerprint": "feature123",
        "feature_artifact_valid": True,
        "target_artifact_path": f"{event_dir}/monitoring_qualifying_targets.parquet",
        "target_artifact_present": target_artifact_present,
        "target_artifact_valid": target_artifact_present,
        "prequalification_ready": True,
        "forecastable": True,
        "settleable": target_artifact_present,
        "onboarding_status": "registered_not_forecasted",
        "feature_driver_count": 2,
        "target_driver_count": 2 if target_artifact_present else 0,
        "evaluable_driver_count": 2 if target_artifact_present else 0,
        "non_evaluable_driver_count": 0,
        "target_coverage_rate": 1.0 if target_artifact_present else 0.0,
        "partial_target_coverage": False,
        "target_coverage_status": "target_coverage_complete"
        if target_artifact_present
        else "target_missing",
        "settlement_metric_status": "scorable" if target_artifact_present else "not_scorable",
    }


def _write_registry(config: DataConfig, rows: list[dict]) -> None:
    path = config.metrics_output_dir / "prospective_monitoring_event_registry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    for row in rows:
        _write_manifest(config, row)


def _write_cached_session_info(
    config: DataConfig,
    *,
    event: str,
    event_order: int,
    session_name: str,
    session_number: int,
    start: datetime,
) -> None:
    session_dir = (
        config.fastf1_cache_dir
        / "2026"
        / f"2026-03-06_{event.replace(' ', '_')}"
        / session_name.replace(" ", "_")
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": {
            "Meeting": {
                "Number": event_order,
                "Name": event,
                "Location": "Sakhir",
                "Country": {"Name": "Bahrain"},
                "Circuit": {"ShortName": "Bahrain International Circuit"},
            },
            "Name": session_name,
            "Type": "Qualifying" if session_name == "Qualifying" else "Practice",
            "Number": session_number,
            "StartDate": start,
            "EndDate": start + timedelta(hours=1),
            "GmtOffset": timedelta(hours=3),
        }
    }
    with (session_dir / "session_info.ff1pkl").open("wb") as handle:
        pickle.dump(payload, handle)


def _write_manifest(config: DataConfig, registry_row: dict) -> None:
    event_dir = (
        config.project_root / "data/processed/monitoring/2026" / str(registry_row["event_slug"])
    )
    event_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        event_dir / "monitoring_event_manifest.json",
        {
            "season": 2026,
            "event": registry_row["event"],
            "event_slug": registry_row["event_slug"],
            "source_availability": {"FP1": True, "FP2": True, "FP3": True},
            "feature_created_at_utc": "2026-01-01T10:00:00+00:00",
            "target_created_at_utc": "2026-01-01T14:00:00+00:00"
            if registry_row.get("target_artifact_present")
            else None,
            "target_artifact_path": registry_row["target_artifact_path"]
            if registry_row.get("target_artifact_present")
            else None,
            "created_at_utc": "2026-01-01T09:00:00+00:00",
        },
    )


def _write_preflight(
    config: DataConfig,
    event: str,
    *,
    status: str,
    forecast_allowed: bool,
    blocking_check_count: int = 0,
) -> None:
    slug = event.lower().replace(" ", "-")
    _write_json(
        config.metrics_output_dir / "prospective_monitoring_preflight_summary.json",
        {
            "status": status,
            "preflight_run_id": "run123",
            "preflight_summary_path": (
                "reports/metrics/prospective_monitoring_preflight_summary.json"
            ),
            "protocol_name": "season_2026_v1",
            "protocol_fingerprint": "abc123",
            "season": 2026,
            "event": event,
            "event_slug": slug,
            "event_order": 1 if slug == "bahrain" else 2,
            "forecast_allowed": forecast_allowed,
            "blocking_check_count": blocking_check_count,
            "warning_check_count": 0,
            "prospective_monitoring_preflight_runbook_path": (
                "reports/metrics/prospective_monitoring_preflight_runbook.md"
            ),
            "generated_at_utc": "2026-01-01T11:00:00+00:00",
        },
    )


def _write_forecasts(
    config: DataConfig,
    event: str,
    rows: list[tuple[str, str, float]],
    *,
    append: bool = False,
) -> None:
    path = config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    frame = pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "protocol_fingerprint": "abc123",
                "forecast_id": f"{event.lower()}-forecast",
                "forecast_created_at_utc": "2026-01-01T12:00:00+00:00",
                "monitor_season": 2026,
                "event_order": 1 if event == "Bahrain" else 9,
                "season": 2026,
                "event": event,
                "event_slug": event.lower().replace(" ", "-"),
                "checkpoint": "after_fp3",
                "driver": driver,
                "driver_key": driver.lower(),
                "team": team,
                "team_key": team.lower().replace(" ", "-"),
                "prediction_role": "observed_live_policy",
                "diagnostic_only": False,
                "prediction_gap_sec": gap,
                "family": "ablation",
                "model_name": "random_forest",
                "feature_group": "base_plus_relative",
                "temporal_weighting_policy": "uniform",
                "source_lineage_valid": True,
                "live_policy_selected": True,
                "forecast_integrity_status": "valid",
                "qualifying_entry_list_status": "driver_set_parity_passed",
                "qualifying_entry_list_source": "test_fixture",
                "qualifying_entry_list_driver_count": len(rows),
                "qualifying_entry_list_summary_path": "",
                "preflight_status": "ready_to_forecast",
                "preflight_run_id": "run123",
                "preflight_summary_path": (
                    "reports/metrics/prospective_monitoring_preflight_summary.json"
                ),
            }
            for driver, team, gap in rows
        ]
    )
    if append and path.is_file():
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_settlements(
    config: DataConfig,
    event: str,
    rows: list[tuple[str, float | None, float] | tuple[str, float | None, float, bool, str]],
) -> None:
    path = config.metrics_output_dir / "prospective_monitoring_settlements.parquet"
    records = []
    for row in rows:
        driver, actual_gap, predicted_gap = row[:3]
        evaluable = bool(row[3]) if len(row) > 3 else True
        exclusion_reason = str(row[4]) if len(row) > 4 else ""
        absolute_error = (
            abs(predicted_gap - actual_gap) if evaluable and actual_gap is not None else None
        )
        records.append(
            {
                "protocol_name": "season_2026_v1",
                "protocol_fingerprint": "abc123",
                "forecast_id": f"{event.lower()}-forecast",
                "settlement_id": f"{driver}-settlement",
                "settled_at_utc": "2026-01-01T15:00:00+00:00",
                "monitor_season": 2026,
                "event_order": 1 if event == "Bahrain" else 9,
                "season": 2026,
                "event": event,
                "event_slug": event.lower().replace(" ", "-"),
                "checkpoint": "after_fp3",
                "driver": driver,
                "driver_key": driver.lower(),
                "prediction_role": "observed_live_policy",
                "diagnostic_only": False,
                "prediction_gap_sec": predicted_gap,
                "actual_gap_sec": actual_gap,
                "absolute_error_sec": absolute_error,
                "target_evaluable": evaluable,
                "included_in_metrics": evaluable,
                "settlement_evaluable": evaluable,
                "settlement_exclusion_reason": exclusion_reason,
                "settlement_valid": evaluable,
                "forecast_preexisted_settlement": True,
                "forecast_fingerprint_valid": True,
                "forecast_mutation_detected": False,
                "eligible_for_future_prior_evidence": evaluable,
            }
        )
    frame = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_target_coverage(
    config: DataConfig,
    event: str,
    rows: list[tuple[str, bool, bool, str]],
) -> None:
    slug = event.lower().replace(" ", "-")
    path = (
        config.project_root
        / "data/processed/monitoring/2026"
        / slug
        / "monitoring_target_coverage.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "season": 2026,
                "event": event,
                "event_slug": slug,
                "driver": driver,
                "driver_key": driver.lower(),
                "qualifying_target_present": target_present,
                "target_evaluable": target_evaluable,
                "target_missing_reason": reason,
            }
            for driver, target_present, target_evaluable, reason in rows
        ]
    ).to_csv(path, index=False)


def _write_targets(
    config: DataConfig,
    event: str,
    rows: list[tuple[str, str, int, float]],
) -> None:
    slug = event.lower().replace(" ", "-")
    path = (
        config.project_root
        / "data/processed/monitoring/2026"
        / slug
        / "monitoring_qualifying_targets.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "season": 2026,
                "event": event,
                "event_slug": slug,
                "driver": driver,
                "driver_key": driver.lower(),
                "team": team,
                "team_key": team.lower().replace(" ", "-"),
                "quali_position": position,
                "quali_gap_to_pole_sec": gap,
            }
            for driver, team, position, gap in rows
        ]
    ).to_parquet(path, index=False)


def _write_reconciliation(
    config: DataConfig,
    rows: list[tuple[str, int, int]],
) -> None:
    path = config.metrics_output_dir / "prospective_monitoring_event_order_reconciliation.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "monitor_season": 2026,
                "event_slug": slug,
                "forecast_id": f"{slug}-forecast",
                "artifact_event_order": artifact_order,
                "registry_event_order": registry_order,
                "event_order_match": artifact_order == registry_order,
                "event_order_lineage_status": "legacy_noncanonical_event_order",
                "affected_by_prior_evidence_lineage": True,
                "eligible_for_future_prior_evidence_after_reconciliation": False,
                "reconciliation_action": "exclude_from_prior_monitoring_evidence",
                "reconciliation_reason": "event_order_lineage_mismatch_or_unsettled",
            }
            for slug, artifact_order, registry_order in rows
        ]
    ).to_csv(path, index=False)


def _write_raw_identity_checks(
    config: DataConfig,
    rows: list[tuple[str, str, bool, bool]],
) -> None:
    path = config.metrics_output_dir / "raw_session_identity_validation_checks.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    slugs = [
        (event, event.lower().replace(" ", "-"), status, match, blocking)
        for event, status, match, blocking in rows
    ]
    pd.DataFrame(
        [
            {
                "season": 2026,
                "requested_event": event,
                "requested_event_slug": slug,
                "requested_session": "Q",
                "raw_laps_path": f"data/raw/laps/2026/{slug}/q_laps.parquet",
                "raw_metadata_path": (f"data/raw/session_metadata/2026/{slug}/q_metadata.json"),
                "metadata_event_name": f"{event} Grand Prix",
                "expected_event": event,
                "identity_status": status,
                "identity_match": match,
                "blocking": blocking,
                "quarantined": blocking,
                "quarantine_reason": "synthetic quarantine" if blocking else "",
                "reason": "synthetic identity result",
                "recommended_action": "synthetic action",
            }
            for event, slug, status, match, blocking in slugs
        ]
    ).to_csv(path, index=False)


def _write_event_metrics(config: DataConfig, event: str, mae: float) -> None:
    path = config.metrics_output_dir / "prospective_monitoring_event_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "season": 2026,
                "event_slug": event.lower().replace(" ", "-"),
                "checkpoint": "after_fp3",
                "prediction_role": "observed_live_policy",
                "diagnostic_only": False,
                "forecast_rows": 1,
                "rows": 1,
                "scored_rows": 1,
                "excluded_rows": 0,
                "mae_gap_sec": mae,
            }
        ]
    ).to_csv(path, index=False)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
