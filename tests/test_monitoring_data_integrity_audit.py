import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from f1_prediction.cli import app
from f1_prediction.config import DataConfig
from f1_prediction.modeling.monitoring_data_integrity_audit import (
    create_monitoring_data_integrity_audit,
)


def test_fp_only_driver_is_audit_participant_not_public_leaderboard_candidate(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_monitoring_event(config)

    create_monitoring_data_integrity_audit(config)
    population = _read_population(config)

    reserve = population.loc[population["driver_key"].eq("aro")].iloc[0]
    race_driver = population.loc[population["driver_key"].eq("nor")].iloc[0]

    assert bool(reserve["feature_participant"]) is True
    assert bool(reserve["forecast_only_driver"]) is True
    assert reserve["forecast_only_reason"] == "no_qualifying_lap_rows"
    assert bool(reserve["dashboard_primary_leaderboard_eligible"]) is False
    assert bool(race_driver["forecast_eligible_driver"]) is True
    assert bool(race_driver["settlement_evaluable_driver"]) is True


def test_missing_target_does_not_create_zero_actual_or_fake_position(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_monitoring_event(config)

    create_monitoring_data_integrity_audit(config)
    population = _read_population(config)

    reserve = population.loc[population["driver_key"].eq("aro")].iloc[0]
    assert bool(reserve["target_present"]) is False
    assert bool(reserve["target_evaluable"]) is False
    assert bool(reserve["settlement_evaluable_driver"]) is False

    settlements = pd.read_parquet(
        config.metrics_output_dir / "prospective_monitoring_settlements.parquet"
    )
    reserve_settlement = settlements.loc[settlements["driver_key"].eq("aro")].iloc[0]
    assert pd.isna(reserve_settlement["actual_gap_sec"])
    assert pd.isna(reserve_settlement["absolute_error_sec"])


def test_duplicate_actual_qualifying_position_is_detected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_monitoring_event(
        config,
        target_rows=[
            ("NOR", "McLaren", 1, 0.0, True),
            ("VER", "Red Bull Racing", 1, 0.2, True),
        ],
    )

    create_monitoring_data_integrity_audit(config)
    check = _check(config, "qualifying_target_position_unique")

    assert check["status"] == "failed"
    assert bool(check["blocking"]) is True


def test_pole_gap_inconsistency_is_detected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_monitoring_event(
        config,
        target_rows=[
            ("NOR", "McLaren", 1, 0.1, True),
            ("VER", "Red Bull Racing", 2, 0.0, True),
        ],
    )

    create_monitoring_data_integrity_audit(config)
    check = _check(config, "qualifying_target_pole_gap_consistent")

    assert check["status"] == "failed"
    assert bool(check["blocking"]) is True


def test_event_slug_mismatch_between_forecast_and_target_is_detected(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_monitoring_event(config, target_event_slug="wrong-event")

    create_monitoring_data_integrity_audit(config)
    check = _check(config, "event_slug_consistent")

    assert check["status"] == "failed"
    assert bool(check["blocking"]) is True


def test_driver_identity_mismatch_is_detected_as_forecast_only_population(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_monitoring_event(
        config,
        forecast_rows=[("NOR", "McLaren", 0.0), ("VER", "Red Bull Racing", 0.2)],
        target_rows=[
            ("NOR", "McLaren", 1, 0.0, True),
            ("HAM", "Ferrari", 2, 0.2, True),
        ],
        coverage_rows=[
            ("NOR", "McLaren", True, True, ""),
            ("VER", "Red Bull Racing", False, False, "driver_not_in_qualifying_targets"),
            ("HAM", "Ferrari", True, True, ""),
        ],
    )

    create_monitoring_data_integrity_audit(config)
    population = _read_population(config)
    alignment = _check(config, "forecast_to_target_driver_alignment")

    assert alignment["status"] == "warning"
    assert (
        bool(population.loc[population["driver_key"].eq("ver"), "forecast_only_driver"].iloc[0])
        is True
    )
    assert (
        bool(population.loc[population["driver_key"].eq("ham"), "target_present"].iloc[0]) is True
    )


def test_dashboard_actual_value_mismatch_versus_settlement_is_detected(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_monitoring_event(config, dashboard_actual_gap_override=9.999)

    create_monitoring_data_integrity_audit(config)
    check = _check(config, "dashboard_actual_values_match_settlement")

    assert check["status"] == "failed"
    assert bool(check["blocking"]) is True


def test_legacy_australia_and_great_britain_are_quarantined_from_current_event(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_monitoring_event(config, event="Australia", order=1, legacy=True)
    _write_monitoring_event(config, event="Great Britain", order=9, legacy=True, append=True)
    _write_dashboard_current(config, lifecycle_state="no_event_available", event=None)
    _write_dashboard_history(config)

    create_monitoring_data_integrity_audit(config)
    summary = _read_json(config.metrics_output_dir / "monitoring_data_integrity_audit_summary.json")
    population = _read_population(config)

    assert summary["current_event_selection_status"]["safe"] is True
    assert population["legacy_noncanonical"].all()
    assert not population["dashboard_primary_leaderboard_eligible"].any()


def test_legacy_records_remain_available_in_separate_history_section(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_monitoring_event(config, event="Australia", order=1, legacy=True)
    _write_dashboard_current(config, lifecycle_state="no_event_available", event=None)
    _write_dashboard_history(config)

    create_monitoring_data_integrity_audit(config)
    dashboard_history = _read_json(
        config.metrics_output_dir.parent / "dashboard/historical_monitoring_summary.json"
    )

    legacy_events = {
        row["event_identity"]["event_slug"]
        for row in dashboard_history["data"]["legacy_descriptive_records"]
    }
    assert legacy_events == {"australia"}


def test_partial_target_coverage_reports_expected_denominator(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_monitoring_event(config)

    create_monitoring_data_integrity_audit(config)
    comparison = pd.read_csv(
        config.metrics_output_dir / "monitoring_data_integrity_event_comparison.csv"
    )
    row = comparison.iloc[0]

    assert row["forecast_driver_count"] == 3
    assert row["target_driver_count"] == 2
    assert row["forecast_only_driver_count"] == 1


def test_preserved_immutable_snapshot_count_gap_is_classified_not_generic_warning(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_monitoring_event(
        config,
        forecast_rows=[
            ("NOR", "McLaren", 0.0),
            ("VER", "Red Bull Racing", 0.2),
        ],
        target_rows=[
            ("NOR", "McLaren", 1, 0.0, True),
            ("VER", "Red Bull Racing", 2, 0.3, True),
            ("PER", "Red Bull Racing", 3, 0.5, True),
        ],
        coverage_rows=[
            ("NOR", "McLaren", True, True, ""),
            ("VER", "Red Bull Racing", True, True, ""),
            ("PER", "Red Bull Racing", True, False, "pre_q_entry_list_resolution_miss"),
        ],
    )
    _write_features(
        config,
        "Italy",
        [
            ("NOR", "McLaren", 0.0),
            ("VER", "Red Bull Racing", 0.2),
            ("PER", "Red Bull Racing", 0.5),
        ],
    )

    create_monitoring_data_integrity_audit(config)

    forecast_count = _check(config, "forecast_driver_count")
    alignment = _check(config, "feature_to_forecast_driver_alignment")
    assert forecast_count["status"] == "expected_immutable_snapshot"
    assert forecast_count["diagnostic_classification"] == "immutable_snapshot_preserved"
    assert alignment["status"] == "expected_immutable_snapshot"
    assert alignment["diagnostic_classification"] == "immutable_snapshot_preserved"


def test_existing_forecast_artifact_is_not_mutated(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_monitoring_event(config)
    forecast_path = config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    before = forecast_path.read_bytes()

    create_monitoring_data_integrity_audit(config)

    assert forecast_path.read_bytes() == before


def test_clean_future_eligible_event_remains_selectable_as_current(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_monitoring_event(config, event="Australia", order=1, legacy=True)
    _write_monitoring_event(config, event="Italy", order=14, append=True)
    _write_dashboard_current(config, event="Italy")

    create_monitoring_data_integrity_audit(config)
    summary = _read_json(config.metrics_output_dir / "monitoring_data_integrity_audit_summary.json")
    population = _read_population(config)

    assert summary["current_event_selection_status"]["current_event_slug"] == "italy"
    assert summary["current_event_selection_status"]["safe"] is True
    italy = population.loc[population["event_slug"].eq("italy")]
    assert italy["dashboard_primary_leaderboard_eligible"].any()


def test_monitoring_data_integrity_audit_cli_registration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_monitoring_event(config)
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)

    result = CliRunner().invoke(app, ["monitoring-data-integrity-audit"])

    assert result.exit_code == 0
    assert "Monitoring data integrity audit complete" in result.output
    assert "Driver populations:" in result.output


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


def _write_monitoring_event(
    config: DataConfig,
    *,
    event: str = "Italy",
    order: int = 14,
    legacy: bool = False,
    append: bool = False,
    target_event_slug: str | None = None,
    dashboard_actual_gap_override: float | None = None,
    forecast_rows: list[tuple[str, str, float]] | None = None,
    target_rows: list[tuple[str, str, int, float, bool]] | None = None,
    coverage_rows: list[tuple[str, str, bool, bool, str]] | None = None,
) -> None:
    forecast_rows = forecast_rows or [
        ("NOR", "McLaren", 0.0),
        ("VER", "Red Bull Racing", 0.2),
        ("ARO", "Sauber", 0.4),
    ]
    target_rows = target_rows or [
        ("NOR", "McLaren", 1, 0.0, True),
        ("VER", "Red Bull Racing", 2, 0.3, True),
    ]
    coverage_rows = coverage_rows or [
        ("NOR", "McLaren", True, True, ""),
        ("VER", "Red Bull Racing", True, True, ""),
        ("ARO", "Sauber", False, False, "no_qualifying_lap_rows"),
    ]
    _write_protocol(config)
    _append_csv(
        config.metrics_output_dir / "prospective_monitoring_event_registry.csv",
        [_registry_row(event, order, len(forecast_rows), len(target_rows))],
        append=append,
    )
    _append_csv(
        config.metrics_output_dir / "prospective_monitoring_event_order_reconciliation.csv",
        [_reconciliation_row(event, order, legacy)],
        append=append,
    )
    _write_features(config, event, forecast_rows)
    _write_targets(config, event, target_rows, event_slug=target_event_slug)
    _write_coverage(config, event, coverage_rows)
    _append_parquet(
        config.metrics_output_dir / "prospective_monitoring_forecasts.parquet",
        _forecast_frame(event, order, forecast_rows),
        append=append,
    )
    _append_parquet(
        config.metrics_output_dir / "prospective_monitoring_settlements.parquet",
        _settlement_frame(event, order, forecast_rows, target_rows, coverage_rows),
        append=append,
    )
    if not append:
        _write_dashboard_current(config, event=event)
        _write_dashboard_forecast(config, event, forecast_rows)
        _write_dashboard_settlement(
            config,
            event,
            forecast_rows,
            target_rows,
            actual_gap_override=dashboard_actual_gap_override,
        )
        _write_dashboard_history(config)


def _registry_row(event: str, order: int, feature_count: int, target_count: int) -> dict:
    slug = _slug(event)
    event_dir = f"data/processed/monitoring/2026/{slug}"
    return {
        "protocol_name": "season_2026_v1",
        "monitor_season": 2026,
        "event_order": order,
        "event": event,
        "event_slug": slug,
        "checkpoint": "after_fp3",
        "feature_artifact_path": f"{event_dir}/monitoring_fp3_features.parquet",
        "target_artifact_path": f"{event_dir}/monitoring_qualifying_targets.parquet",
        "feature_driver_count": feature_count,
        "target_driver_count": target_count,
        "evaluable_driver_count": target_count,
        "non_evaluable_driver_count": feature_count - target_count,
    }


def _reconciliation_row(event: str, order: int, legacy: bool) -> dict:
    return {
        "protocol_name": "season_2026_v1",
        "monitor_season": 2026,
        "event_slug": _slug(event),
        "forecast_id": f"{_slug(event)}-forecast",
        "artifact_event_order": 44 if legacy else order,
        "registry_event_order": order,
        "event_order_match": not legacy,
        "event_order_lineage_status": "legacy_noncanonical_event_order"
        if legacy
        else "canonical_event_order",
        "eligible_for_future_prior_evidence_after_reconciliation": not legacy,
    }


def _write_protocol(config: DataConfig) -> None:
    _write_json(
        config.metrics_output_dir / "prospective_monitoring_protocol.json",
        {
            "protocol_name": "season_2026_v1",
            "protocol_fingerprint": "abc123",
            "checkpoint": "after_fp3",
        },
    )


def _write_features(
    config: DataConfig,
    event: str,
    rows: list[tuple[str, str, float]],
) -> None:
    slug = _slug(event)
    path = (
        config.project_root
        / "data/processed/monitoring/2026"
        / slug
        / "monitoring_fp3_features.parquet"
    )
    frame = pd.DataFrame(
        [
            {
                "season": 2026,
                "event": event,
                "event_slug": slug,
                "checkpoint": "after_fp3",
                "driver": driver,
                "driver_key": driver.lower(),
                "team": team,
            }
            for driver, team, _gap in rows
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_targets(
    config: DataConfig,
    event: str,
    rows: list[tuple[str, str, int, float, bool]],
    *,
    event_slug: str | None = None,
) -> None:
    slug = _slug(event)
    target_slug = event_slug or slug
    path = (
        config.project_root
        / "data/processed/monitoring/2026"
        / slug
        / "monitoring_qualifying_targets.parquet"
    )
    frame = pd.DataFrame(
        [
            {
                "season": 2026,
                "event": event,
                "event_slug": target_slug,
                "checkpoint": "after_fp3",
                "driver": driver,
                "driver_key": driver.lower(),
                "team": team,
                "quali_position": position,
                "quali_gap_to_pole_sec": gap,
                "target_evaluable": evaluable,
            }
            for driver, team, position, gap, evaluable in rows
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_coverage(
    config: DataConfig,
    event: str,
    rows: list[tuple[str, str, bool, bool, str]],
) -> None:
    slug = _slug(event)
    path = (
        config.project_root
        / "data/processed/monitoring/2026"
        / slug
        / "monitoring_target_coverage.csv"
    )
    frame = pd.DataFrame(
        [
            {
                "season": 2026,
                "event": event,
                "event_slug": slug,
                "driver": driver,
                "driver_key": driver.lower(),
                "team": team,
                "qualifying_target_present": target_present,
                "target_evaluable": target_evaluable,
                "target_missing_reason": reason,
            }
            for driver, team, target_present, target_evaluable, reason in rows
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _forecast_frame(
    event: str,
    order: int,
    rows: list[tuple[str, str, float]],
) -> pd.DataFrame:
    slug = _slug(event)
    return pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "protocol_fingerprint": "abc123",
                "forecast_id": f"{slug}-forecast",
                "forecast_created_at_utc": "2026-01-01T12:00:00+00:00",
                "monitor_season": 2026,
                "event_order": order,
                "season": 2026,
                "event": event,
                "event_slug": slug,
                "checkpoint": "after_fp3",
                "driver": driver,
                "driver_key": driver.lower(),
                "team": team,
                "team_key": team.lower().replace(" ", "-"),
                "prediction_role": "observed_live_policy",
                "prediction_gap_sec": gap,
            }
            for driver, team, gap in rows
        ]
    )


def _settlement_frame(
    event: str,
    order: int,
    forecasts: list[tuple[str, str, float]],
    targets: list[tuple[str, str, int, float, bool]],
    coverage: list[tuple[str, str, bool, bool, str]],
) -> pd.DataFrame:
    slug = _slug(event)
    target_by_driver = {driver: gap for driver, _team, _pos, gap, _ok in targets}
    coverage_by_driver = {
        driver: (target_present, target_evaluable, reason)
        for driver, _team, target_present, target_evaluable, reason in coverage
    }
    records = []
    for driver, team, predicted_gap in forecasts:
        _present, evaluable, reason = coverage_by_driver.get(driver, (False, False, "missing"))
        actual_gap = target_by_driver.get(driver) if evaluable else None
        records.append(
            {
                "protocol_name": "season_2026_v1",
                "protocol_fingerprint": "abc123",
                "forecast_id": f"{slug}-forecast",
                "settlement_id": f"{driver}-settlement",
                "settled_at_utc": "2026-01-01T15:00:00+00:00",
                "monitor_season": 2026,
                "event_order": order,
                "season": 2026,
                "event": event,
                "event_slug": slug,
                "checkpoint": "after_fp3",
                "driver": driver,
                "driver_key": driver.lower(),
                "team": team,
                "prediction_role": "observed_live_policy",
                "prediction_gap_sec": predicted_gap,
                "actual_gap_sec": actual_gap,
                "absolute_error_sec": abs(predicted_gap - actual_gap)
                if actual_gap is not None
                else None,
                "target_evaluable": evaluable,
                "included_in_metrics": evaluable,
                "settlement_evaluable": evaluable,
                "settlement_exclusion_reason": "" if evaluable else reason,
            }
        )
    return pd.DataFrame(records)


def _write_dashboard_current(
    config: DataConfig,
    *,
    event: str | None = "Italy",
    lifecycle_state: str = "settled",
) -> None:
    event_identity = None
    if event is not None:
        event_identity = {
            "season": 2026,
            "event": event,
            "event_slug": _slug(event),
            "event_order": 14,
        }
    _write_json(
        config.metrics_output_dir.parent / "dashboard/current_event.json",
        {
            "schema_version": "1.0",
            "artifact_type": "current_event",
            "generated_at_utc": "2026-01-01T16:00:00+00:00",
            "source_artifacts": [],
            "source_fingerprints": {},
            "status": "empty" if event is None else "complete",
            "data": {
                "event_identity": event_identity,
                "lifecycle": {
                    "state": lifecycle_state,
                    "display_label": lifecycle_state,
                    "reason": lifecycle_state,
                },
            },
        },
    )


def _write_dashboard_forecast(
    config: DataConfig,
    event: str,
    rows: list[tuple[str, str, float]],
) -> None:
    sorted_rows = sorted(rows, key=lambda row: row[2])
    _write_json(
        config.metrics_output_dir.parent / "dashboard/event_forecast.json",
        {
            "schema_version": "1.0",
            "artifact_type": "event_forecast",
            "generated_at_utc": "2026-01-01T16:00:00+00:00",
            "source_artifacts": [],
            "source_fingerprints": {},
            "status": "complete",
            "data": {
                "event_identity": {
                    "season": 2026,
                    "event": event,
                    "event_slug": _slug(event),
                    "event_order": 14,
                },
                "leaderboard": [
                    {
                        "predicted_position": index,
                        "driver": driver,
                        "driver_code": driver,
                        "team": team,
                        "predicted_gap_to_pole_sec": gap,
                    }
                    for index, (driver, team, gap) in enumerate(sorted_rows, start=1)
                ],
            },
        },
    )


def _write_dashboard_settlement(
    config: DataConfig,
    event: str,
    forecasts: list[tuple[str, str, float]],
    targets: list[tuple[str, str, int, float, bool]],
    *,
    actual_gap_override: float | None = None,
) -> None:
    target_by_driver = {driver: (position, gap) for driver, _team, position, gap, _ok in targets}
    rows = []
    for index, (driver, _team, predicted_gap) in enumerate(
        sorted(forecasts, key=lambda row: row[2]),
        start=1,
    ):
        if driver not in target_by_driver:
            continue
        actual_position, actual_gap = target_by_driver[driver]
        rows.append(
            {
                "predicted_position": index,
                "actual_position": actual_position,
                "driver": driver,
                "driver_code": driver,
                "predicted_gap_to_pole_sec": predicted_gap,
                "actual_gap_to_pole_sec": actual_gap_override
                if actual_gap_override is not None
                else actual_gap,
                "absolute_gap_error_sec": abs(predicted_gap - actual_gap),
            }
        )
    _write_json(
        config.metrics_output_dir.parent / "dashboard/event_settlement.json",
        {
            "schema_version": "1.0",
            "artifact_type": "event_settlement",
            "generated_at_utc": "2026-01-01T16:00:00+00:00",
            "source_artifacts": [],
            "source_fingerprints": {},
            "status": "complete",
            "data": {
                "event_identity": {
                    "season": 2026,
                    "event": event,
                    "event_slug": _slug(event),
                    "event_order": 14,
                },
                "driver_comparison": rows,
            },
        },
    )


def _write_dashboard_history(config: DataConfig) -> None:
    registry_path = config.metrics_output_dir / "prospective_monitoring_event_registry.csv"
    registry = pd.read_csv(registry_path) if registry_path.is_file() else pd.DataFrame()
    legacy = [
        {
            "event_identity": {
                "season": int(row["monitor_season"]),
                "event": row["event"],
                "event_slug": row["event_slug"],
                "event_order": int(row["event_order"]),
            },
            "legacy_noncanonical": True,
            "eligible_for_valid_prospective_evidence": False,
            "lifecycle_state": "legacy_descriptive_only",
        }
        for _, row in registry.iterrows()
        if row["event_slug"] in {"australia", "great-britain"}
    ]
    _write_json(
        config.metrics_output_dir.parent / "dashboard/historical_monitoring_summary.json",
        {
            "schema_version": "1.0",
            "artifact_type": "historical_monitoring_summary",
            "generated_at_utc": "2026-01-01T16:00:00+00:00",
            "source_artifacts": [],
            "source_fingerprints": {},
            "status": "complete",
            "data": {
                "valid_prospective_monitoring": {
                    "event_count": 0,
                    "settled_event_count": 0,
                    "forecasted_event_count": 0,
                    "aggregate_metrics": {
                        "available": False,
                        "reason": "no_valid_events",
                        "value": None,
                    },
                    "events": [],
                },
                "legacy_descriptive_records": legacy,
                "backtest_context": {"available": False},
            },
        },
    )


def _append_csv(path: Path, rows: list[dict], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if append and path.is_file():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    frame.to_csv(path, index=False)


def _append_parquet(path: Path, frame: pd.DataFrame, *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.is_file():
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
    frame.to_parquet(path, index=False)


def _check(config: DataConfig, name: str) -> dict:
    checks = pd.read_csv(config.metrics_output_dir / "monitoring_data_integrity_audit_checks.csv")
    rows = checks.loc[checks["check_name"].eq(name)]
    assert not rows.empty
    return rows.iloc[0].to_dict()


def _read_population(config: DataConfig) -> pd.DataFrame:
    return pd.read_csv(
        config.metrics_output_dir / "monitoring_data_integrity_event_driver_population.csv"
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(event: str) -> str:
    return event.lower().replace(" ", "-")
