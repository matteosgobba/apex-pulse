"""Artifact-only integrity audit for monitored forecast and settlement data."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.data.monitoring_onboarding import (
    read_csv,
    read_parquet,
    utc_now,
    write_json,
)
from f1_prediction.data.raw_session_identity import validate_raw_session_identity
from f1_prediction.utils.paths import ensure_directory

LEGACY_LINEAGE_STATUS = "legacy_noncanonical_event_order"
LIVE_POLICY_ROLE = "observed_live_policy"


@dataclass(frozen=True)
class MonitoringDataIntegrityAuditSummary:
    """Output paths and high-level status for the integrity audit."""

    status: str
    summary_path: Path
    checks_path: Path
    failures_path: Path
    population_path: Path
    event_comparison_path: Path
    runbook_path: Path
    events_audited: int
    events_with_blocking_integrity_failures: int
    events_with_warnings: int
    dashboard_safe_for_public_display: bool
    recommended_operator_action: str


def create_monitoring_data_integrity_audit(
    config: DataConfig,
) -> MonitoringDataIntegrityAuditSummary:
    """Audit existing monitoring artifacts without mutating source artifacts."""
    metrics_dir = config.metrics_output_dir
    ensure_directory(metrics_dir)
    sources = _read_sources(config)
    events = _discover_events(sources)
    population = _build_population_table(config, sources, events)
    checks = _build_checks(config, sources, events, population)
    event_comparison = _build_event_comparison(config, sources, events, population)
    failures = checks[
        checks["status"].astype(str).isin(["failed", "warning"]) | checks["blocking"].astype(bool)
    ].copy()
    summary = _build_summary(sources, events, checks, population)
    summary_path = metrics_dir / "monitoring_data_integrity_audit_summary.json"
    checks_path = metrics_dir / "monitoring_data_integrity_audit_checks.csv"
    failures_path = metrics_dir / "monitoring_data_integrity_audit_failures.csv"
    population_path = metrics_dir / "monitoring_data_integrity_event_driver_population.csv"
    event_comparison_path = metrics_dir / "monitoring_data_integrity_event_comparison.csv"
    runbook_path = metrics_dir / "monitoring_data_integrity_runbook.md"
    write_json(summary_path, summary)
    checks.to_csv(checks_path, index=False)
    failures.to_csv(failures_path, index=False)
    population.to_csv(population_path, index=False)
    event_comparison.to_csv(event_comparison_path, index=False)
    runbook_path.write_text(_runbook_markdown(summary, event_comparison), encoding="utf-8")
    return MonitoringDataIntegrityAuditSummary(
        status=str(summary["status"]),
        summary_path=summary_path,
        checks_path=checks_path,
        failures_path=failures_path,
        population_path=population_path,
        event_comparison_path=event_comparison_path,
        runbook_path=runbook_path,
        events_audited=int(summary["events_audited"]),
        events_with_blocking_integrity_failures=int(
            summary["events_with_blocking_integrity_failures"]
        ),
        events_with_warnings=int(summary["events_with_warnings"]),
        dashboard_safe_for_public_display=bool(summary["dashboard_safe_for_public_display"]),
        recommended_operator_action=str(summary["recommended_operator_action"]),
    )


def _read_sources(config: DataConfig) -> dict[str, Any]:
    metrics_dir = config.metrics_output_dir
    dashboard_dir = metrics_dir.parent / "dashboard"
    return {
        "registry": read_csv(metrics_dir / "prospective_monitoring_event_registry.csv"),
        "reconciliation": read_csv(
            metrics_dir / "prospective_monitoring_event_order_reconciliation.csv"
        ),
        "forecasts": read_parquet(metrics_dir / "prospective_monitoring_forecasts.parquet"),
        "settlements": read_parquet(metrics_dir / "prospective_monitoring_settlements.parquet"),
        "forecast_integrity": read_csv(
            metrics_dir / "prospective_monitoring_forecast_integrity_audit.csv"
        ),
        "settlement_integrity": read_csv(
            metrics_dir / "prospective_monitoring_settlement_integrity_audit.csv"
        ),
        "dashboard_manifest": _read_dashboard_json(dashboard_dir / "dashboard_manifest.json"),
        "dashboard_current": _read_dashboard_json(dashboard_dir / "current_event.json"),
        "dashboard_forecast": _read_dashboard_json(dashboard_dir / "event_forecast.json"),
        "dashboard_settlement": _read_dashboard_json(dashboard_dir / "event_settlement.json"),
    }


def _discover_events(sources: dict[str, Any]) -> list[dict[str, Any]]:
    records: dict[tuple[int, str], dict[str, Any]] = {}
    registry = sources["registry"]
    if not registry.empty:
        for _, row in registry.iterrows():
            season = _int_or_none(row.get("monitor_season") or row.get("season"))
            slug = _str_or_none(row.get("event_slug"))
            if season is None or slug is None:
                continue
            records[(season, slug)] = {
                "season": season,
                "event": _str_or_none(row.get("event")) or slug,
                "event_slug": slug,
                "event_order": _int_or_none(row.get("event_order")),
                "registry_row": row.to_dict(),
            }
    for source_name in ("forecasts", "settlements"):
        frame = sources[source_name]
        if frame.empty or "event_slug" not in frame:
            continue
        for _, row in frame.drop_duplicates(["season", "event_slug"]).iterrows():
            season = _int_or_none(row.get("season") or row.get("monitor_season"))
            slug = _str_or_none(row.get("event_slug"))
            if season is None or slug is None:
                continue
            records.setdefault(
                (season, slug),
                {
                    "season": season,
                    "event": _str_or_none(row.get("event")) or slug,
                    "event_slug": slug,
                    "event_order": _int_or_none(row.get("event_order")),
                    "registry_row": {},
                },
            )
    return sorted(
        records.values(),
        key=lambda row: (
            row.get("season") or 0,
            row.get("event_order") if row.get("event_order") is not None else 10_000,
            str(row.get("event_slug")),
        ),
    )


def _build_population_table(
    config: DataConfig,
    sources: dict[str, Any],
    events: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        artifacts = _event_artifacts(config, sources, event)
        legacy = _legacy_noncanonical(sources, event)
        driver_keys = sorted(
            set(_driver_keys(artifacts["features"]))
            | set(_driver_keys(artifacts["forecasts"]))
            | set(_driver_keys(artifacts["targets"]))
            | set(_driver_keys(artifacts["coverage"]))
            | set(_driver_keys(artifacts["settlements"]))
        )
        for driver_key in driver_keys:
            feature = _row_for_driver(artifacts["features"], driver_key)
            forecast = _row_for_driver(artifacts["forecasts"], driver_key)
            target = _row_for_driver(artifacts["targets"], driver_key)
            coverage = _row_for_driver(artifacts["coverage"], driver_key)
            settlement = _row_for_driver(artifacts["settlements"], driver_key)
            target_present = _bool_from_row(
                coverage,
                "qualifying_target_present",
                fallback=target is not None,
            )
            target_evaluable = _bool_from_row(
                coverage,
                "target_evaluable",
                fallback=_bool_from_row(target, "target_evaluable", fallback=target is not None),
            )
            settlement_evaluable = _bool_from_row(
                settlement,
                "settlement_evaluable",
                fallback=False,
            )
            forecast_present = forecast is not None
            forecast_eligible = bool(forecast_present and target_evaluable)
            forecast_only = bool(forecast_present and not forecast_eligible)
            reason = _first_text(
                [
                    _value(coverage, "target_missing_reason"),
                    _value(coverage, "settlement_exclusion_reason"),
                    _value(settlement, "settlement_exclusion_reason"),
                    "target_not_evaluable" if forecast_only else None,
                ]
            )
            identity = feature or forecast or target or coverage or settlement or {}
            rows.append(
                {
                    "season": event["season"],
                    "event": event["event"],
                    "event_slug": event["event_slug"],
                    "driver": _value(identity, "driver"),
                    "driver_key": driver_key,
                    "team": _value(identity, "team"),
                    "feature_participant": feature is not None,
                    "forecast_eligible_driver": forecast_eligible,
                    "forecast_only_driver": forecast_only,
                    "forecast_only_reason": reason if forecast_only else "",
                    "target_present": bool(target_present),
                    "target_evaluable": bool(target_evaluable),
                    "settlement_present": settlement is not None,
                    "settlement_evaluable_driver": bool(settlement_evaluable),
                    "dashboard_primary_leaderboard_eligible": bool(
                        forecast_eligible and not legacy
                    ),
                    "legacy_noncanonical": legacy,
                }
            )
    return pd.DataFrame(rows, columns=_population_columns())


def _build_checks(
    config: DataConfig,
    sources: dict[str, Any],
    events: list[dict[str, Any]],
    population: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dashboard_current = sources["dashboard_current"].get("data", {})
    current_identity = dashboard_current.get("event_identity") or {}
    current_slug = _str_or_none(current_identity.get("event_slug"))
    current_lifecycle = (dashboard_current.get("lifecycle") or {}).get("state")
    current_is_legacy = current_lifecycle == "legacy_descriptive_only"
    for event in events:
        artifacts = _event_artifacts(config, sources, event)
        event_population = population[population["event_slug"].astype(str).eq(event["event_slug"])]
        legacy = _legacy_noncanonical(sources, event)
        rows.extend(_identity_checks(event, artifacts))
        rows.extend(_count_alignment_checks(event, artifacts, event_population))
        rows.extend(_target_quality_checks(event, artifacts))
        rows.extend(_population_checks(event, event_population))
        rows.extend(_settlement_checks(event, artifacts))
        rows.append(_raw_session_identity_check(config, event))
        rows.append(
            _check(
                event,
                "legacy_event_lineage_status",
                "warning" if legacy else "passed",
                False,
                "legacy_noncanonical" if legacy else "canonical",
                "canonical_or_explicitly_quarantined",
                "Legacy event is quarantined from valid prospective evidence."
                if legacy
                else "Event is not marked legacy/noncanonical.",
                "Keep legacy records separate from prospective aggregates." if legacy else "",
            )
        )
        if current_slug == event["event_slug"]:
            rows.extend(_dashboard_checks(sources, event, artifacts))
        else:
            rows.append(
                _check(
                    event,
                    "dashboard_event_identity_consistent",
                    "unavailable",
                    False,
                    current_slug or "no_current_event",
                    event["event_slug"],
                    "Event is not the exported current dashboard event.",
                    "",
                )
            )
            rows.append(
                _check(
                    event,
                    "dashboard_actual_values_match_settlement",
                    "unavailable",
                    False,
                    "not_current_event",
                    "current_event_only",
                    "Dashboard settlement comparison is only exported for the current event.",
                    "",
                )
            )
        rows.append(
            _check(
                event,
                "dashboard_current_event_selection_safe",
                "failed" if current_is_legacy else "passed",
                bool(current_is_legacy),
                current_lifecycle or "no_event_available",
                "no legacy_descriptive_only default current event",
                "Legacy/noncanonical event selected as current."
                if current_is_legacy
                else "Default current event selection does not expose a legacy event.",
                "Export no_event_available when only legacy records exist."
                if current_is_legacy
                else "",
            )
        )
    return pd.DataFrame(rows, columns=_check_columns())


def _raw_session_identity_check(config: DataConfig, event: dict[str, Any]) -> dict[str, Any]:
    result = validate_raw_session_identity(
        config,
        season=int(event["season"]),
        event=str(event["event"]),
        session="Q",
    )
    missing = result.identity_status in {"raw_artifact_missing", "metadata_missing"}
    status = "unavailable" if missing else "passed" if result.identity_match else "failed"
    return _check(
        event,
        "raw_q_session_identity_verified",
        status,
        bool(result.blocking and not missing),
        result.identity_status,
        "identity_verified",
        result.reason,
        result.recommended_action if result.blocking else "",
    )


def _identity_checks(
    event: dict[str, Any],
    artifacts: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for check_name, column, expected in [
        ("event_slug_consistent", "event_slug", event["event_slug"]),
        ("season_consistent", "season", event["season"]),
    ]:
        observed = {
            name: sorted(set(frame[column].dropna().astype(str).tolist()))
            for name, frame in artifacts.items()
            if not frame.empty and column in frame
        }
        bad = any(str(expected) not in values or len(values) > 1 for values in observed.values())
        rows.append(
            _check(
                event,
                check_name,
                "failed" if bad else "passed",
                bad,
                json.dumps(observed, sort_keys=True),
                str(expected),
                "Source artifact identity values diverge."
                if bad
                else "Source artifact identity values are aligned.",
                "Repair event identity mapping before public export." if bad else "",
            )
        )
    rows.append(
        _check(
            event,
            "event_identity_consistent",
            "failed" if any(row["status"] == "failed" for row in rows) else "passed",
            any(row["status"] == "failed" for row in rows),
            "; ".join(f"{row['check_name']}={row['status']}" for row in rows),
            "event_slug_consistent and season_consistent",
            "Event identity consistency follows slug and season checks.",
            "",
        )
    )
    for artifact_name in ("features", "forecasts", "targets", "settlements"):
        frame = artifacts[artifact_name]
        duplicate_count = _duplicate_key_count(frame)
        rows.append(
            _check(
                event,
                "driver_identity_unique",
                "failed" if duplicate_count else "passed",
                bool(duplicate_count),
                f"{artifact_name}:{duplicate_count}",
                "0 duplicate driver keys per artifact",
                "Duplicate driver identities detected."
                if duplicate_count
                else "Driver identities are unique in this artifact.",
                "Resolve duplicate driver_key rows." if duplicate_count else "",
            )
        )
    return rows


def _count_alignment_checks(
    event: dict[str, Any],
    artifacts: dict[str, pd.DataFrame],
    population: pd.DataFrame,
) -> list[dict[str, Any]]:
    feature_count = len(_driver_keys(artifacts["features"]))
    forecast_count = len(_driver_keys(artifacts["forecasts"]))
    target_count = (
        int(population["target_present"].astype(bool).sum()) if not population.empty else 0
    )
    settled_count = (
        int(population["settlement_present"].astype(bool).sum()) if not population.empty else 0
    )
    evaluable_count = (
        int(population["settlement_evaluable_driver"].astype(bool).sum())
        if not population.empty
        else 0
    )
    forecast_only_count = (
        int(population["forecast_only_driver"].astype(bool).sum()) if not population.empty else 0
    )
    immutable_snapshot_preserved = bool(
        forecast_count
        and target_count > forecast_count
        and settled_count == forecast_count
        and evaluable_count == forecast_count
    )
    feature_forecast_aligned = _key_set(artifacts["features"]) == _key_set(artifacts["forecasts"])
    return [
        _check_count(
            event,
            "feature_driver_count",
            feature_count,
            _registry_count(event, "feature_driver_count"),
        ),
        _check(
            event,
            "forecast_driver_count",
            "passed"
            if forecast_count == feature_count
            else "expected_immutable_snapshot"
            if immutable_snapshot_preserved
            else "warning",
            False,
            forecast_count,
            feature_count,
            "Observed forecast count matches feature count."
            if forecast_count == feature_count
            else "Immutable historical forecast snapshot preserved with partial actual coverage."
            if immutable_snapshot_preserved
            else "Observed forecast count differs from feature count.",
            "No retrospective prediction should be generated."
            if immutable_snapshot_preserved
            else "",
            "immutable_snapshot_preserved" if immutable_snapshot_preserved else "",
        ),
        _check_count(
            event,
            "target_driver_count",
            target_count,
            _registry_count(event, "target_driver_count"),
        ),
        _check_count(event, "settled_driver_count", settled_count, forecast_count),
        _check(
            event,
            "feature_to_forecast_driver_alignment",
            "passed"
            if feature_forecast_aligned
            else "expected_immutable_snapshot"
            if immutable_snapshot_preserved
            else "warning",
            False,
            f"feature={feature_count};forecast={forecast_count}",
            "feature drivers equal forecast drivers for immutable snapshot preservation",
            "Forecast snapshots preserve all feature participants."
            if feature_forecast_aligned
            else (
                "Immutable historical forecast snapshot intentionally preserves a narrower "
                "driver set."
            )
            if immutable_snapshot_preserved
            else "Forecast driver set differs from feature participant set.",
            "No retrospective prediction should be generated."
            if immutable_snapshot_preserved
            else "",
            "immutable_snapshot_preserved" if immutable_snapshot_preserved else "",
        ),
        _check(
            event,
            "forecast_to_target_driver_alignment",
            "warning" if forecast_only_count else "passed",
            False,
            f"forecast_only={forecast_only_count}",
            "forecast-only drivers explicitly reasoned",
            "Some forecast rows do not have evaluable qualifying targets."
            if forecast_only_count
            else "Forecast rows align with evaluable target population.",
            "Keep forecast-only rows out of public primary leaderboard."
            if forecast_only_count
            else "",
        ),
        _check(
            event,
            "target_to_settlement_driver_alignment",
            "passed" if evaluable_count <= settled_count else "failed",
            evaluable_count > settled_count,
            f"settlement_evaluable={evaluable_count};settlement_present={settled_count}",
            "all evaluable target drivers have settlement rows",
            "Settlement rows cover evaluable target drivers."
            if evaluable_count <= settled_count
            else "Evaluable target drivers are missing settlement rows.",
            "Repair settlement join coverage." if evaluable_count > settled_count else "",
        ),
    ]


def _target_quality_checks(
    event: dict[str, Any],
    artifacts: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    targets = artifacts["targets"]
    rows: list[dict[str, Any]] = []
    positions = pd.to_numeric(
        targets.get("quali_position", pd.Series(dtype=float)),
        errors="coerce",
    )
    gaps = pd.to_numeric(
        targets.get("quali_gap_to_pole_sec", pd.Series(dtype=float)),
        errors="coerce",
    )
    invalid_position = bool((positions.isna() | positions.le(0) | positions.mod(1).ne(0)).any())
    invalid_gap = bool((gaps.isna() | gaps.lt(0)).any())
    duplicate_positions = bool(positions.duplicated().any()) if not positions.empty else False
    pole_rows = (
        targets.loc[positions.eq(1)]
        if not targets.empty and "quali_position" in targets
        else targets.iloc[0:0]
    )
    pole_consistent = (
        len(pole_rows) == 1
        and not gaps.empty
        and math.isclose(float(pole_rows["quali_gap_to_pole_sec"].iloc[0]), 0.0, abs_tol=1e-9)
        and math.isclose(float(gaps.min()), 0.0, abs_tol=1e-9)
    )
    rows.append(_quality_check(event, "qualifying_target_position_valid", invalid_position))
    rows.append(_quality_check(event, "qualifying_target_gap_valid", invalid_gap))
    rows.append(_quality_check(event, "qualifying_target_position_unique", duplicate_positions))
    rows.append(
        _check(
            event,
            "qualifying_target_pole_gap_consistent",
            "passed" if pole_consistent or targets.empty else "failed",
            bool(not pole_consistent and not targets.empty),
            f"pole_rows={len(pole_rows)};min_gap={_safe_min(gaps)}",
            "exactly one P1 target with 0.000 pole gap",
            "Pole target and minimum gap are internally coherent."
            if pole_consistent or targets.empty
            else "Pole target does not match the zero-gap row.",
            "Repair qualifying target construction."
            if not pole_consistent and not targets.empty
            else "",
        )
    )
    rows.append(
        _check(
            event,
            "target_driver_is_qualifying_eligible",
            "passed",
            False,
            str(len(targets)),
            "target rows are evaluable qualifying rows",
            "Target parquet contains only evaluable target rows.",
            "",
        )
    )
    return rows


def _population_checks(event: dict[str, Any], population: pd.DataFrame) -> list[dict[str, Any]]:
    forecast_only = population[population["forecast_only_driver"].astype(bool)]
    missing_reasons = forecast_only["forecast_only_reason"].astype(str).str.len().eq(0).sum()
    return [
        _check(
            event,
            "feature_driver_is_forecast_eligible",
            "warning" if not forecast_only.empty else "passed",
            False,
            f"forecast_only={len(forecast_only)}",
            "feature participants separated from forecast-eligible drivers",
            "Some feature participants are forecast-only/audit-only."
            if not forecast_only.empty
            else "All forecasted feature participants are forecast eligible.",
            "Keep forecast-only participants out of public primary rows."
            if not forecast_only.empty
            else "",
        ),
        _check(
            event,
            "forecast_only_driver_reason_present",
            "failed" if missing_reasons else "passed",
            bool(missing_reasons),
            str(int(missing_reasons)),
            "0 forecast-only rows without reason",
            "Every forecast-only row has an explicit reason."
            if not missing_reasons
            else "Some forecast-only rows lack reasons.",
            "Populate target coverage/exclusion reason." if missing_reasons else "",
        ),
    ]


def _settlement_checks(
    event: dict[str, Any],
    artifacts: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    settlements = artifacts["settlements"]
    if settlements.empty:
        observed = "no_settlement_rows"
        valid = True
    else:
        included = _bool_series(settlements, "included_in_metrics")
        evaluable = _bool_series(settlements, "settlement_evaluable")
        actual = pd.to_numeric(settlements.get("actual_gap_sec"), errors="coerce")
        error = pd.to_numeric(settlements.get("absolute_error_sec"), errors="coerce")
        valid = bool(
            (included.eq(evaluable)).all()
            and actual.loc[~evaluable].isna().all()
            and error.loc[~evaluable].isna().all()
            and actual.loc[evaluable].notna().all()
            and error.loc[evaluable].notna().all()
        )
        observed = (
            f"rows={len(settlements)};"
            f"evaluable={int(evaluable.sum())};"
            f"non_evaluable={int((~evaluable).sum())}"
        )
    return [
        _check(
            event,
            "settlement_metric_row_eligibility_valid",
            "passed" if valid else "failed",
            not valid,
            observed,
            "only evaluable rows included in metrics with non-null actual/error values",
            "Settlement eligibility flags and metric values are internally coherent."
            if valid
            else "Settlement eligibility flags or metric values are inconsistent.",
            "Repair settlement row eligibility before public display." if not valid else "",
        )
    ]


def _dashboard_checks(
    sources: dict[str, Any],
    event: dict[str, Any],
    artifacts: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    forecast_rows = _dashboard_rows(sources["dashboard_forecast"], "leaderboard")
    settlement_rows = _dashboard_rows(sources["dashboard_settlement"], "driver_comparison")
    dashboard_identity = sources["dashboard_current"].get("data", {}).get("event_identity", {})
    dashboard_slug = _str_or_none(dashboard_identity.get("event_slug"))
    identity_ok = dashboard_slug == event["event_slug"]
    mismatch_count = _dashboard_settlement_mismatches(settlement_rows, artifacts["settlements"])
    return [
        _check(
            event,
            "dashboard_event_identity_consistent",
            "passed" if identity_ok else "failed",
            not identity_ok,
            dashboard_slug or "missing",
            event["event_slug"],
            "Dashboard current event identity matches source event."
            if identity_ok
            else "Dashboard current event identity diverges from source event.",
            "Repair dashboard event mapping." if not identity_ok else "",
        ),
        _check(
            event,
            "dashboard_actual_values_match_settlement",
            "passed" if mismatch_count == 0 else "failed",
            mismatch_count > 0,
            f"mismatches={mismatch_count};forecast_rows={len(forecast_rows)}",
            "dashboard actual position/gap/error values match settlement artifact",
            "Dashboard actual values match settlement rows."
            if mismatch_count == 0
            else "Dashboard actual values differ from settlement rows.",
            "Repair dashboard settlement mapping." if mismatch_count else "",
        ),
    ]


def _build_event_comparison(
    config: DataConfig,
    sources: dict[str, Any],
    events: list[dict[str, Any]],
    population: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        artifacts = _event_artifacts(config, sources, event)
        event_pop = population[population["event_slug"].astype(str).eq(event["event_slug"])]
        targets = artifacts["targets"]
        positions = pd.to_numeric(
            targets.get("quali_position", pd.Series(dtype=float)),
            errors="coerce",
        )
        gaps = pd.to_numeric(
            targets.get("quali_gap_to_pole_sec", pd.Series(dtype=float)),
            errors="coerce",
        )
        pole_rows = targets.loc[positions.eq(1)] if not targets.empty else targets
        forecast_only = event_pop[event_pop["forecast_only_driver"].astype(bool)]
        rows.append(
            {
                "season": event["season"],
                "event": event["event"],
                "event_slug": event["event_slug"],
                "feature_driver_count": int(event_pop["feature_participant"].sum())
                if not event_pop.empty
                else 0,
                "forecast_driver_count": int(_driver_keys(artifacts["forecasts"]).__len__()),
                "target_driver_count": int(event_pop["target_present"].sum())
                if not event_pop.empty
                else 0,
                "settled_driver_count": int(event_pop["settlement_present"].sum())
                if not event_pop.empty
                else 0,
                "forecast_only_driver_count": len(forecast_only),
                "forecast_only_drivers": ",".join(forecast_only["driver"].astype(str).tolist()),
                "forecast_only_reasons": ",".join(
                    sorted(set(forecast_only["forecast_only_reason"].astype(str).tolist()))
                ),
                "qualifying_positions_unique": bool(not positions.duplicated().any()),
                "target_pole_gap_consistent": bool(
                    len(pole_rows) == 1
                    and not gaps.empty
                    and math.isclose(
                        float(pole_rows["quali_gap_to_pole_sec"].iloc[0]), 0.0, abs_tol=1e-9
                    )
                    and math.isclose(float(gaps.min()), 0.0, abs_tol=1e-9)
                )
                if not targets.empty
                else False,
                "event_keys_aligned": _event_keys_aligned(event, artifacts),
                "legacy_noncanonical": _legacy_noncanonical(sources, event),
                "root_cause_assessment": _event_root_cause(event_pop, sources, event),
            }
        )
    return pd.DataFrame(rows)


def _build_summary(
    sources: dict[str, Any],
    events: list[dict[str, Any]],
    checks: pd.DataFrame,
    population: pd.DataFrame,
) -> dict[str, Any]:
    blocking = checks[checks["blocking"].astype(bool)]
    warnings = checks[checks["status"].astype(str).eq("warning")]
    blocking_events = set(blocking["event_slug"].astype(str).tolist())
    warning_events = set(warnings["event_slug"].astype(str).tolist())
    current = sources["dashboard_current"].get("data", {})
    current_lifecycle = (current.get("lifecycle") or {}).get("state", "no_event_available")
    current_slug = (current.get("event_identity") or {}).get("event_slug")
    root_causes = _root_cause_categories(population, current_lifecycle)
    dashboard_blocking = blocking[blocking["check_name"].astype(str).str.startswith("dashboard_")]
    dashboard_safe = not bool(dashboard_blocking.shape[0])
    status = "fail" if not blocking.empty else "warning" if not warnings.empty else "pass"
    if not events:
        status = "empty"
    return {
        "status": status,
        "generated_at_utc": utc_now(),
        "events_audited": len(events),
        "events_with_blocking_integrity_failures": len(blocking_events),
        "events_with_warnings": len(warning_events),
        "root_cause_categories": root_causes,
        "dashboard_safe_for_public_display": dashboard_safe,
        "legacy_records_quarantined": current_lifecycle != "legacy_descriptive_only",
        "current_event_selection_status": {
            "current_event_slug": current_slug,
            "current_lifecycle_state": current_lifecycle,
            "safe": current_lifecycle != "legacy_descriptive_only",
        },
        "recommended_operator_action": _recommended_action(status, root_causes),
    }


def _event_artifacts(
    config: DataConfig,
    sources: dict[str, Any],
    event: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    season = int(event["season"])
    slug = str(event["event_slug"])
    registry_row = event.get("registry_row", {})
    event_dir = config.project_root / "data/processed/monitoring" / str(season) / slug
    feature_path = _resolve_artifact_path(
        config.project_root,
        registry_row.get("feature_artifact_path"),
        event_dir / "monitoring_fp3_features.parquet",
    )
    target_path = _resolve_artifact_path(
        config.project_root,
        registry_row.get("target_artifact_path"),
        event_dir / "monitoring_qualifying_targets.parquet",
    )
    coverage_path = event_dir / "monitoring_target_coverage.csv"
    forecasts = _event_rows(sources["forecasts"], season, slug)
    settlements = _event_rows(sources["settlements"], season, slug)
    return {
        "features": read_parquet(feature_path),
        "forecasts": _live_rows(forecasts),
        "targets": read_parquet(target_path),
        "coverage": read_csv(coverage_path),
        "settlements": _live_rows(settlements),
    }


def _live_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "prediction_role" not in frame:
        return frame
    return frame[frame["prediction_role"].astype(str).eq(LIVE_POLICY_ROLE)].copy()


def _event_rows(frame: pd.DataFrame, season: int, slug: str) -> pd.DataFrame:
    if frame.empty or "event_slug" not in frame:
        return pd.DataFrame()
    mask = frame["event_slug"].astype(str).eq(slug)
    if "season" in frame:
        mask &= pd.to_numeric(frame["season"], errors="coerce").eq(season)
    return frame[mask].copy()


def _legacy_noncanonical(sources: dict[str, Any], event: dict[str, Any]) -> bool:
    reconciliation = sources["reconciliation"]
    if not reconciliation.empty and "event_slug" in reconciliation:
        rows = reconciliation[reconciliation["event_slug"].astype(str).eq(str(event["event_slug"]))]
        if not rows.empty:
            return bool(
                rows["event_order_lineage_status"].astype(str).eq(LEGACY_LINEAGE_STATUS).any()
            )
    return str(event["event_slug"]) in {"australia", "great-britain"}


def _dashboard_settlement_mismatches(
    dashboard_rows: list[dict[str, Any]],
    settlements: pd.DataFrame,
) -> int:
    if not dashboard_rows or settlements.empty:
        return 0
    expected = {}
    comparison = _comparison_from_settlements(settlements)
    for row in comparison:
        expected[_norm_key(row.get("driver_code"))] = row
    mismatches = 0
    for row in dashboard_rows:
        driver_key = _norm_key(row.get("driver_code") or row.get("driver"))
        expected_row = expected.get(driver_key)
        if expected_row is None:
            mismatches += 1
            continue
        for column in (
            "actual_position",
            "actual_gap_to_pole_sec",
            "absolute_gap_error_sec",
        ):
            if not _values_equal(row.get(column), expected_row.get(column)):
                mismatches += 1
                break
    return mismatches


def _comparison_from_settlements(settlements: pd.DataFrame) -> list[dict[str, Any]]:
    frame = settlements.copy()
    frame["_predicted_gap"] = pd.to_numeric(frame.get("prediction_gap_sec"), errors="coerce")
    frame["_actual_gap"] = pd.to_numeric(frame.get("actual_gap_sec"), errors="coerce")
    frame["_predicted_position"] = frame["_predicted_gap"].rank(method="first", na_option="bottom")
    frame["_actual_position"] = frame["_actual_gap"].rank(method="first", na_option="bottom")
    rows = []
    for _, row in frame.iterrows():
        evaluable = bool(row.get("settlement_evaluable", False))
        rows.append(
            {
                "driver_code": _value(row, "driver"),
                "actual_position": _int_or_none(row.get("_actual_position")) if evaluable else None,
                "actual_gap_to_pole_sec": _float_or_none(row.get("actual_gap_sec"))
                if evaluable
                else None,
                "absolute_gap_error_sec": _float_or_none(row.get("absolute_error_sec"))
                if evaluable
                else None,
            }
        )
    return rows


def _dashboard_rows(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = payload.get("data", {}).get(key, [])
    return rows if isinstance(rows, list) else []


def _root_cause_categories(population: pd.DataFrame, current_lifecycle: str) -> list[str]:
    categories = set()
    if not population.empty and population["forecast_only_driver"].astype(bool).any():
        categories.add("monitored_feature_driver_eligibility")
        categories.add("dashboard_population_projection")
    if not population.empty and (~population["settlement_evaluable_driver"].astype(bool)).any():
        categories.add("partial_target_coverage")
    if current_lifecycle == "legacy_descriptive_only":
        categories.add("dashboard_current_event_selection")
    if not population.empty and population["legacy_noncanonical"].astype(bool).any():
        categories.add("legacy_lineage_quarantine")
    return sorted(categories)


def _event_root_cause(
    population: pd.DataFrame,
    sources: dict[str, Any],
    event: dict[str, Any],
) -> str:
    causes = []
    if not population.empty and population["forecast_only_driver"].astype(bool).any():
        causes.append("feature participants included practice-only/non-evaluable drivers")
    if _legacy_noncanonical(sources, event):
        causes.append("legacy noncanonical event-order lineage")
    if not causes:
        causes.append("no blocking integrity issue detected")
    return "; ".join(causes)


def _recommended_action(status: str, root_causes: list[str]) -> str:
    if status == "fail":
        return "repair_blocking_integrity_failures_before_public_dashboard"
    if "dashboard_population_projection" in root_causes:
        return "export_driver_populations_and_keep_forecast_only_rows_out_of_primary_views"
    if "legacy_lineage_quarantine" in root_causes:
        return "keep_legacy_records_in_separate_history_section"
    return "no_operator_action_required"


def _check_count(
    event: dict[str, Any],
    name: str,
    observed: int,
    expected: int | None,
) -> dict[str, Any]:
    if expected is None:
        status = "unavailable"
        reason = "No expected count was available."
    else:
        status = "passed" if observed == expected else "warning"
        reason = (
            "Observed count matches expected count."
            if observed == expected
            else "Observed count differs from expected count."
        )
    return _check(event, name, status, False, observed, expected, reason, "")


def _quality_check(event: dict[str, Any], name: str, failed: bool) -> dict[str, Any]:
    return _check(
        event,
        name,
        "failed" if failed else "passed",
        failed,
        "invalid" if failed else "valid",
        "valid",
        "Target quality check failed." if failed else "Target quality check passed.",
        "Repair qualifying target construction." if failed else "",
    )


def _check(
    event: dict[str, Any],
    name: str,
    status: str,
    blocking: bool,
    observed: Any,
    expected: Any,
    reason: str,
    action: str,
    diagnostic_classification: str = "",
) -> dict[str, Any]:
    return {
        "season": event.get("season"),
        "event": event.get("event"),
        "event_slug": event.get("event_slug"),
        "check_name": name,
        "status": status,
        "blocking": bool(blocking),
        "observed_value": _stringify(observed),
        "expected_value": _stringify(expected),
        "reason": reason,
        "recommended_action": action,
        "diagnostic_classification": diagnostic_classification,
    }


def _registry_count(event: dict[str, Any], column: str) -> int | None:
    value = event.get("registry_row", {}).get(column)
    return _int_or_none(value)


def _driver_keys(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    return sorted(set(_driver_key_series(frame).dropna().astype(str).str.lower().tolist()))


def _key_set(frame: pd.DataFrame) -> set[str]:
    return set(_driver_keys(frame))


def _duplicate_key_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    key_parts = [
        frame[column].astype(str)
        for column in ["season", "event_slug", "checkpoint"]
        if column in frame
    ]
    if not key_parts:
        return 0
    key_parts.append(_driver_key_series(frame).astype(str))
    key_frame = pd.concat(key_parts, axis=1)
    return int(key_frame.duplicated().sum())


def _row_for_driver(frame: pd.DataFrame, driver_key: str) -> dict[str, Any] | None:
    if frame.empty:
        return None
    rows = frame[_driver_key_series(frame).astype(str).str.lower().eq(driver_key)]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _driver_key_series(frame: pd.DataFrame) -> pd.Series:
    if "driver_key" in frame:
        key = frame["driver_key"].where(frame["driver_key"].notna(), frame.get("driver"))
    elif "driver" in frame:
        key = frame["driver"]
    else:
        key = pd.Series([pd.NA] * len(frame), index=frame.index)
    return key.astype("string").str.strip()


def _bool_from_row(row: dict[str, Any] | None, column: str, *, fallback: bool) -> bool:
    if row is None or column not in row or pd.isna(row[column]):
        return fallback
    value = row[column]
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series([False] * len(frame), index=frame.index)
    series = frame[column]
    if series.dtype == object:
        return series.astype(str).str.lower().isin({"true", "1", "yes"})
    return series.fillna(False).astype(bool)


def _event_keys_aligned(event: dict[str, Any], artifacts: dict[str, pd.DataFrame]) -> bool:
    for frame in artifacts.values():
        if frame.empty:
            continue
        if (
            "event_slug" in frame
            and not frame["event_slug"].astype(str).eq(event["event_slug"]).all()
        ):
            return False
        if (
            "season" in frame
            and not pd.to_numeric(frame["season"], errors="coerce").eq(event["season"]).all()
        ):
            return False
    return True


def _read_dashboard_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_artifact_path(project_root: Path, configured: Any, fallback: Path) -> Path:
    if configured is None or pd.isna(configured) or str(configured).strip() == "":
        return fallback
    path = Path(str(configured))
    return path if path.is_absolute() else project_root / path


def _population_columns() -> list[str]:
    return [
        "season",
        "event",
        "event_slug",
        "driver",
        "driver_key",
        "team",
        "feature_participant",
        "forecast_eligible_driver",
        "forecast_only_driver",
        "forecast_only_reason",
        "target_present",
        "target_evaluable",
        "settlement_present",
        "settlement_evaluable_driver",
        "dashboard_primary_leaderboard_eligible",
        "legacy_noncanonical",
    ]


def _check_columns() -> list[str]:
    return [
        "season",
        "event",
        "event_slug",
        "check_name",
        "status",
        "blocking",
        "observed_value",
        "expected_value",
        "reason",
        "recommended_action",
        "diagnostic_classification",
    ]


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _value(row: dict[str, Any] | pd.Series | None, column: str) -> Any:
    if row is None or column not in row:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return value


def _first_text(values: list[Any]) -> str:
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _str_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    return text if text else None


def _int_or_none(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_min(series: pd.Series) -> str:
    if series.empty or series.dropna().empty:
        return ""
    return str(float(series.min()))


def _norm_key(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _values_equal(left: Any, right: Any) -> bool:
    if left is None and right is None:
        return True
    left_num = _float_or_none(left)
    right_num = _float_or_none(right)
    if left_num is not None or right_num is not None:
        return (
            left_num is not None
            and right_num is not None
            and math.isclose(left_num, right_num, abs_tol=1e-9)
        )
    return left == right


def _runbook_markdown(summary: dict[str, Any], event_comparison: pd.DataFrame) -> str:
    lines = [
        "# Monitoring Data Integrity Audit Runbook",
        "",
        "This audit is artifact-only. It does not fetch FastF1 data, retrain models, "
        "create forecasts, add targets, or settle forecasts.",
        "",
        "## Driver Populations",
        "",
        "- `feature_participant`: driver with FP3-safe monitoring feature rows.",
        "- `forecast_eligible_driver`: forecasted driver with an evaluable qualifying target "
        "according to existing coverage artifacts, or pending eligibility when no target coverage "
        "exists.",
        "- `settlement_evaluable_driver`: forecasted driver with a valid matching qualifying "
        "target and evaluable settlement row.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Events audited: `{summary['events_audited']}`",
        f"- Blocking events: `{summary['events_with_blocking_integrity_failures']}`",
        f"- Warning events: `{summary['events_with_warnings']}`",
        f"- Dashboard safe for public display: `{summary['dashboard_safe_for_public_display']}`",
        f"- Recommended action: `{summary['recommended_operator_action']}`",
        "",
        "## Event Findings",
        "",
    ]
    for _, row in event_comparison.iterrows():
        lines.extend(
            [
                f"### {row['season']} {row['event']}",
                "",
                f"- Feature participants: `{row['feature_driver_count']}`",
                f"- Forecasted drivers: `{row['forecast_driver_count']}`",
                f"- Target drivers: `{row['target_driver_count']}`",
                f"- Settled drivers: `{row['settled_driver_count']}`",
                f"- Forecast-only drivers: `{row['forecast_only_drivers'] or 'none'}`",
                f"- Forecast-only reasons: `{row['forecast_only_reasons'] or 'none'}`",
                f"- Legacy noncanonical: `{row['legacy_noncanonical']}`",
                f"- Root cause assessment: {row['root_cause_assessment']}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"
