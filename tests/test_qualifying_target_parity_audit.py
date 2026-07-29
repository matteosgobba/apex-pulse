import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from f1_prediction.cli import app
from f1_prediction.config import DataConfig
from f1_prediction.data.fastf1_loader import build_lap_output_path
from f1_prediction.modeling.qualifying_target_parity_audit import (
    create_qualifying_target_parity_audit,
)


def test_exact_raw_q_target_parity_passes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_event(config)

    summary = create_qualifying_target_parity_audit(config)
    event_summary = _event_summary(config).iloc[0]

    assert summary.status == "pass"
    assert event_summary["event_parity_status"] == "parity_verified"
    assert event_summary["position_match_rate"] == 1.0
    assert event_summary["gap_match_rate"] == 1.0
    assert event_summary["best_lap_match_rate"] == 1.0


def test_stored_target_position_mismatch_is_detected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_event(config, target_position_override={"NOR": 2})

    create_qualifying_target_parity_audit(config)

    assert _check(config, "stored_target_position_matches_raw_q")["status"] == "failed"


def test_stored_target_gap_mismatch_is_detected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_event(config, target_gap_override={"VER": 9.0})

    create_qualifying_target_parity_audit(config)

    assert _check(config, "stored_target_gap_matches_raw_q")["status"] == "failed"


def test_stored_best_lap_mismatch_is_detected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_event(config, target_best_lap_override={"VER": 99.0})

    create_qualifying_target_parity_audit(config)

    assert _check(config, "stored_target_best_lap_matches_raw_q")["status"] == "failed"


def test_raw_q_duplicate_ranking_inconsistency_is_detected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_event(config, raw_laps=[("NOR", "McLaren", 79.0), ("VER", "Red Bull Racing", 79.0)])

    create_qualifying_target_parity_audit(config)

    assert _check(config, "raw_q_position_unique")["status"] == "failed"


def test_raw_q_missing_artifact_is_reported_without_crashing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_event(config, write_raw=False)

    summary = create_qualifying_target_parity_audit(config)

    assert summary.status == "fail"
    assert summary.events_with_missing_raw_q == 1
    assert _check(config, "raw_q_artifact_exists")["status"] == "failed"


def test_driver_identity_normalization_mismatch_is_detected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_event(config, target_driver_override={"VER": "HAM"})

    create_qualifying_target_parity_audit(config)

    assert _check(config, "stored_target_driver_matches_raw_q")["status"] == "failed"


def test_event_slug_mismatch_is_detected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_event(config, target_event_slug="wrong-event")

    create_qualifying_target_parity_audit(config)

    assert (
        _check(config, "event_slug_consistent_across_raw_target_settlement_dashboard")["status"]
        == "failed"
    )


def test_settlement_actual_mismatch_versus_target_is_detected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_event(config, settlement_gap_override={"VER": 8.0})

    create_qualifying_target_parity_audit(config)

    assert _check(config, "settlement_actual_gap_matches_target")["status"] == "failed"


def test_dashboard_actual_mismatch_versus_settlement_is_detected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_event(config, dashboard_gap_override={"VER": 8.0})

    create_qualifying_target_parity_audit(config)

    assert _check(config, "dashboard_actual_gap_matches_settlement")["status"] == "failed"


def test_target_only_driver_is_forecast_coverage_warning_not_settlement_corruption(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_event(
        config,
        raw_laps=[
            ("NOR", "McLaren", 79.0),
            ("VER", "Red Bull Racing", 79.2),
            ("PER", "Red Bull Racing", 79.5),
        ],
        forecast_drivers=("NOR", "VER"),
        settlement_drivers=("NOR", "VER"),
        dashboard_drivers=("NOR", "VER"),
    )

    summary = create_qualifying_target_parity_audit(config)
    event_summary = _event_summary(config).iloc[0]
    driver_comparison = _driver_comparison(config)

    assert summary.status == "warning"
    assert event_summary["event_parity_status"] == "parity_verified"
    assert event_summary["forecast_coverage_status"] == "partial_coverage"
    assert event_summary["forecast_coverage_ratio"] == 2 / 3
    assert event_summary["unforecasted_actual_entrants"] == "per"
    assert event_summary["settlement_projection_status"] == "settlement_projection_verified"
    assert _check(config, "settlement_actual_gap_matches_target")["status"] == "passed"
    per = driver_comparison[driver_comparison["driver"].eq("PER")].iloc[0]
    assert bool(per["unforecasted_actual_entrant"]) is True
    assert per["forecast_coverage_reason"] == "pre_q_entry_list_resolution_miss"
    assert per["parity_status"] == "pre_q_entry_list_resolution_miss"


def test_legacy_event_with_verified_parity_remains_legacy_but_passes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_event(config, event="Australia", legacy=True)

    create_qualifying_target_parity_audit(config)
    event_summary = _event_summary(config).iloc[0]
    driver_comparison = _driver_comparison(config)

    assert bool(event_summary["legacy_noncanonical"]) is True
    assert event_summary["event_parity_status"] == "parity_verified"
    assert driver_comparison["legacy_noncanonical"].all()


def test_legacy_event_with_target_mismatch_remains_legacy_and_fails(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_event(config, event="Australia", legacy=True, target_gap_override={"VER": 9.0})

    create_qualifying_target_parity_audit(config)
    event_summary = _event_summary(config).iloc[0]

    assert bool(event_summary["legacy_noncanonical"]) is True
    assert event_summary["event_parity_status"] == "target_gap_mismatch"


def test_absolute_paths_are_absent_from_reports(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_event(config)

    create_qualifying_target_parity_audit(config)

    for path in config.metrics_output_dir.glob("qualifying_target_parity_*"):
        assert str(tmp_path) not in path.read_text(encoding="utf-8", errors="ignore")


def test_cli_registration_and_minimal_execution(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_event(config)
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)

    result = CliRunner().invoke(app, ["qualifying-target-parity-audit"])

    assert result.exit_code == 0
    assert "Qualifying target parity audit complete" in result.output
    assert "Driver comparison:" in result.output


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


def _write_event(
    config: DataConfig,
    *,
    event: str = "Italy",
    legacy: bool = False,
    write_raw: bool = True,
    raw_laps: list[tuple[str, str, float]] | None = None,
    target_position_override: dict[str, int] | None = None,
    target_gap_override: dict[str, float] | None = None,
    target_best_lap_override: dict[str, float] | None = None,
    target_driver_override: dict[str, str] | None = None,
    target_event_slug: str | None = None,
    settlement_gap_override: dict[str, float] | None = None,
    dashboard_gap_override: dict[str, float] | None = None,
    forecast_drivers: tuple[str, ...] | None = None,
    settlement_drivers: tuple[str, ...] | None = None,
    dashboard_drivers: tuple[str, ...] | None = None,
) -> None:
    raw_laps = raw_laps or [("NOR", "McLaren", 79.0), ("VER", "Red Bull Racing", 79.2)]
    target_position_override = target_position_override or {}
    target_gap_override = target_gap_override or {}
    target_best_lap_override = target_best_lap_override or {}
    target_driver_override = target_driver_override or {}
    settlement_gap_override = settlement_gap_override or {}
    dashboard_gap_override = dashboard_gap_override or {}
    season = 2026
    if write_raw:
        _write_raw_q(config, season, event, raw_laps)
        _write_metadata(config, season, event)
    targets = _target_rows(
        season,
        event,
        raw_laps,
        position_override=target_position_override,
        gap_override=target_gap_override,
        best_lap_override=target_best_lap_override,
        driver_override=target_driver_override,
        event_slug=target_event_slug,
    )
    _write_registry(config, season, event, len(raw_laps), len(targets))
    _write_reconciliation(config, season, event, legacy)
    if forecast_drivers is not None:
        _write_forecasts(config, season, event, targets, forecast_drivers)
    _write_targets(config, season, event, targets)
    _write_settlements(
        config,
        season,
        event,
        targets,
        gap_override=settlement_gap_override,
        settlement_drivers=settlement_drivers,
    )
    _write_dashboard(
        config,
        season,
        event,
        targets,
        gap_override=dashboard_gap_override,
        dashboard_drivers=dashboard_drivers,
    )


def _write_raw_q(
    config: DataConfig,
    season: int,
    event: str,
    rows: list[tuple[str, str, float]],
) -> None:
    path = build_lap_output_path(config.lap_output_dir, season, event, "Q")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Driver": [row[0] for row in rows],
            "Team": [row[1] for row in rows],
            "LapNumber": [1.0] * len(rows),
            "Stint": [1.0] * len(rows),
            "Compound": ["SOFT"] * len(rows),
            "TyreLife": [2.0] * len(rows),
            "LapTime": pd.to_timedelta([row[2] for row in rows], unit="s"),
            "Sector1Time": pd.to_timedelta([25.0] * len(rows), unit="s"),
            "Sector2Time": pd.to_timedelta([29.0] * len(rows), unit="s"),
            "Sector3Time": pd.to_timedelta([25.0] * len(rows), unit="s"),
            "IsAccurate": [True] * len(rows),
            "Deleted": [False] * len(rows),
            "PitOutTime": [pd.NaT] * len(rows),
            "PitInTime": [pd.NaT] * len(rows),
        }
    ).to_parquet(path, index=False)


def _write_metadata(config: DataConfig, season: int, event: str) -> None:
    path = config.session_metadata_output_dir / str(season) / _slug(event) / "q_metadata.json"
    _write_json(
        path,
        {
            "season": season,
            "event_input": event,
            "event_name": f"{event} Grand Prix",
            "event_slug": _slug(event),
            "session_slug": "q",
        },
    )


def _target_rows(
    season: int,
    event: str,
    raw_laps: list[tuple[str, str, float]],
    *,
    position_override: dict[str, int],
    gap_override: dict[str, float],
    best_lap_override: dict[str, float],
    driver_override: dict[str, str],
    event_slug: str | None,
) -> list[dict]:
    pole = min(row[2] for row in raw_laps)
    sorted_rows = sorted(raw_laps, key=lambda row: row[2])
    rows = []
    for index, (driver, team, lap_time) in enumerate(sorted_rows, start=1):
        stored_driver = driver_override.get(driver, driver)
        best_lap = best_lap_override.get(driver, lap_time)
        gap = gap_override.get(driver, lap_time - pole)
        rows.append(
            {
                "season": season,
                "event": event,
                "event_slug": event_slug or _slug(event),
                "driver": stored_driver,
                "driver_key": stored_driver.lower(),
                "team": team,
                "team_key": _slug(team),
                "quali_gap_to_pole_sec": gap,
                "quali_position": position_override.get(driver, index),
                "quali_best_lap_time_sec": best_lap,
            }
        )
    return rows


def _write_registry(
    config: DataConfig,
    season: int,
    event: str,
    feature_count: int,
    target_count: int,
) -> None:
    path = config.metrics_output_dir / "prospective_monitoring_event_registry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "monitor_season": season,
                "season": season,
                "event": event,
                "event_slug": _slug(event),
                "event_order": 1,
                "feature_driver_count": feature_count,
                "target_driver_count": target_count,
            }
        ]
    ).to_csv(path, index=False)


def _write_reconciliation(config: DataConfig, season: int, event: str, legacy: bool) -> None:
    path = config.metrics_output_dir / "prospective_monitoring_event_order_reconciliation.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "monitor_season": season,
                "event_slug": _slug(event),
                "event_order_lineage_status": "legacy_noncanonical_event_order"
                if legacy
                else "canonical_event_order",
            }
        ]
    ).to_csv(path, index=False)


def _write_targets(config: DataConfig, season: int, event: str, rows: list[dict]) -> None:
    path = (
        config.project_root
        / "data/processed/monitoring"
        / str(season)
        / _slug(event)
        / "monitoring_qualifying_targets.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_settlements(
    config: DataConfig,
    season: int,
    event: str,
    targets: list[dict],
    *,
    gap_override: dict[str, float],
    settlement_drivers: tuple[str, ...] | None,
) -> None:
    path = config.metrics_output_dir / "prospective_monitoring_settlements.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "season": season,
                "event": event,
                "event_slug": _slug(event),
                "driver": row["driver"],
                "driver_key": row["driver_key"],
                "prediction_role": "observed_live_policy",
                "actual_gap_sec": gap_override.get(row["driver"], row["quali_gap_to_pole_sec"]),
                "settlement_evaluable": True,
            }
            for row in targets
            if settlement_drivers is None or row["driver"] in settlement_drivers
        ]
    ).to_parquet(path, index=False)


def _write_forecasts(
    config: DataConfig,
    season: int,
    event: str,
    targets: list[dict],
    forecast_drivers: tuple[str, ...],
) -> None:
    path = config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "season": season,
                "event": event,
                "event_slug": _slug(event),
                "driver": row["driver"],
                "driver_key": row["driver_key"],
                "prediction_role": "observed_live_policy",
                "diagnostic_only": False,
            }
            for row in targets
            if row["driver"] in forecast_drivers
        ]
    ).to_parquet(path, index=False)


def _write_dashboard(
    config: DataConfig,
    season: int,
    event: str,
    targets: list[dict],
    *,
    gap_override: dict[str, float],
    dashboard_drivers: tuple[str, ...] | None,
) -> None:
    dashboard_dir = config.metrics_output_dir.parent / "dashboard"
    _write_json(
        dashboard_dir / "current_event.json",
        {
            "data": {
                "event_identity": {
                    "season": season,
                    "event": event,
                    "event_slug": _slug(event),
                }
            }
        },
    )
    _write_json(
        dashboard_dir / "event_settlement.json",
        {
            "data": {
                "driver_comparison": [
                    {
                        "driver": row["driver"],
                        "driver_code": row["driver"],
                        "actual_position": row["quali_position"],
                        "actual_gap_to_pole_sec": gap_override.get(
                            row["driver"], row["quali_gap_to_pole_sec"]
                        ),
                    }
                    for row in targets
                    if dashboard_drivers is None or row["driver"] in dashboard_drivers
                ]
            }
        },
    )


def _event_summary(config: DataConfig) -> pd.DataFrame:
    return pd.read_csv(config.metrics_output_dir / "qualifying_target_parity_event_summary.csv")


def _driver_comparison(config: DataConfig) -> pd.DataFrame:
    return pd.read_csv(config.metrics_output_dir / "qualifying_target_parity_driver_comparison.csv")


def _check(config: DataConfig, name: str) -> dict:
    checks = pd.read_csv(config.metrics_output_dir / "qualifying_target_parity_audit_checks.csv")
    rows = checks.loc[checks["check_name"].eq(name)]
    assert not rows.empty
    return rows.iloc[0].to_dict()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _slug(value: str) -> str:
    return value.lower().replace(" ", "-")
