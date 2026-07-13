"""Artifact-only raw qualifying source parity audit for monitored targets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.data.fastf1_loader import build_lap_output_path
from f1_prediction.data.monitoring_onboarding import (
    read_csv,
    read_json,
    read_parquet,
    target_artifact_path,
    utc_now,
    write_json,
)
from f1_prediction.data.raw_session_identity import validate_raw_session_identity
from f1_prediction.features.lap_cleaning import clean_session_laps
from f1_prediction.utils.paths import ensure_directory, slugify

LIVE_POLICY_ROLE = "observed_live_policy"
LEGACY_LINEAGE_STATUS = "legacy_noncanonical_event_order"
FLOAT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class QualifyingTargetParityAuditSummary:
    """Output paths and high-level status for the raw-Q parity audit."""

    status: str
    summary_path: Path
    checks_path: Path
    failures_path: Path
    event_summary_path: Path
    driver_comparison_path: Path
    runbook_path: Path
    events_audited: int
    events_with_raw_q: int
    events_with_verified_parity: int
    events_with_blocking_failures: int
    events_with_missing_raw_q: int
    recommended_operator_action: str


def create_qualifying_target_parity_audit(
    config: DataConfig,
    *,
    season: int | None = None,
    event: str | None = None,
) -> QualifyingTargetParityAuditSummary:
    """Compare stored monitored targets against independently reconstructed local raw Q data."""
    ensure_directory(config.metrics_output_dir)
    sources = _read_sources(config)
    events = _filter_events(_discover_events(sources), season=season, event=event)
    driver_comparison = _build_driver_comparison(config, sources, events)
    checks = _build_checks(config, sources, events, driver_comparison)
    event_summary = _build_event_summary(config, sources, events, checks, driver_comparison)
    failures = checks[
        checks["status"].astype(str).isin(["failed", "warning"]) | checks["blocking"].astype(bool)
    ].copy()
    summary = _build_summary(events, checks, event_summary)

    summary_path = config.metrics_output_dir / "qualifying_target_parity_audit_summary.json"
    checks_path = config.metrics_output_dir / "qualifying_target_parity_audit_checks.csv"
    failures_path = config.metrics_output_dir / "qualifying_target_parity_audit_failures.csv"
    event_summary_path = config.metrics_output_dir / "qualifying_target_parity_event_summary.csv"
    driver_comparison_path = (
        config.metrics_output_dir / "qualifying_target_parity_driver_comparison.csv"
    )
    runbook_path = config.metrics_output_dir / "qualifying_target_parity_runbook.md"

    write_json(summary_path, summary)
    checks.to_csv(checks_path, index=False)
    failures.to_csv(failures_path, index=False)
    event_summary.to_csv(event_summary_path, index=False)
    driver_comparison.to_csv(driver_comparison_path, index=False)
    runbook_path.write_text(_runbook(summary, event_summary, driver_comparison), encoding="utf-8")

    return QualifyingTargetParityAuditSummary(
        status=str(summary["status"]),
        summary_path=summary_path,
        checks_path=checks_path,
        failures_path=failures_path,
        event_summary_path=event_summary_path,
        driver_comparison_path=driver_comparison_path,
        runbook_path=runbook_path,
        events_audited=int(summary["events_audited"]),
        events_with_raw_q=int(summary["events_with_raw_q"]),
        events_with_verified_parity=int(summary["events_with_verified_parity"]),
        events_with_blocking_failures=int(summary["events_with_blocking_failures"]),
        events_with_missing_raw_q=int(summary["events_with_missing_raw_q"]),
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
        "settlements": read_parquet(metrics_dir / "prospective_monitoring_settlements.parquet"),
        "dashboard_current": _read_json_if_available(dashboard_dir / "current_event.json"),
        "dashboard_settlement": _read_json_if_available(dashboard_dir / "event_settlement.json"),
    }


def _discover_events(sources: dict[str, Any]) -> list[dict[str, Any]]:
    registry = sources["registry"]
    records: dict[tuple[int, str], dict[str, Any]] = {}
    if not registry.empty:
        for _, row in registry.iterrows():
            season = _int_or_none(row.get("monitor_season") or row.get("season"))
            slug = _text_or_none(row.get("event_slug"))
            if season is None or slug is None:
                continue
            records[(season, slug)] = {
                "season": season,
                "event": _text_or_none(row.get("event")) or slug,
                "event_slug": slug,
                "event_order": _int_or_none(row.get("event_order")),
                "registry_row": row.to_dict(),
            }
    settlements = sources["settlements"]
    if not settlements.empty and "event_slug" in settlements:
        for _, row in settlements.drop_duplicates(["season", "event_slug"]).iterrows():
            season = _int_or_none(row.get("season") or row.get("monitor_season"))
            slug = _text_or_none(row.get("event_slug"))
            if season is None or slug is None:
                continue
            records.setdefault(
                (season, slug),
                {
                    "season": season,
                    "event": _text_or_none(row.get("event")) or slug,
                    "event_slug": slug,
                    "event_order": _int_or_none(row.get("event_order")),
                    "registry_row": {},
                },
            )
    return sorted(
        records.values(),
        key=lambda item: (
            item.get("season") or 0,
            item.get("event_order") if item.get("event_order") is not None else 10_000,
            str(item.get("event_slug")),
        ),
    )


def _filter_events(
    events: list[dict[str, Any]],
    *,
    season: int | None,
    event: str | None,
) -> list[dict[str, Any]]:
    if season is None and event is None:
        return events
    event_slug = slugify(event) if event else None
    return [
        item
        for item in events
        if (season is None or item["season"] == season)
        and (event_slug is None or item["event_slug"] == event_slug)
    ]


def _build_driver_comparison(
    config: DataConfig,
    sources: dict[str, Any],
    events: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        raw = _reconstruct_raw_q(config, event)
        targets = _stored_targets(config, event)
        settlements = _settlement_actuals(sources, event)
        dashboard = _dashboard_actuals(sources, event)
        legacy = _legacy_noncanonical(sources, event)
        driver_keys = sorted(
            set(raw["driver_key"].dropna().astype(str).tolist())
            | set(targets["driver_key"].dropna().astype(str).tolist())
            | set(settlements["driver_key"].dropna().astype(str).tolist())
            | set(dashboard["driver_key"].dropna().astype(str).tolist())
        )
        for driver_key in driver_keys:
            raw_row = _row_by_key(raw, driver_key)
            target_row = _row_by_key(targets, driver_key)
            settlement_row = _row_by_key(settlements, driver_key)
            dashboard_row = _row_by_key(dashboard, driver_key)
            identity = raw_row or target_row or settlement_row or dashboard_row or {}
            position_match = _numbers_match(
                _value(target_row, "quali_position"),
                _value(raw_row, "reconstructed_quali_position"),
                integer=True,
            )
            gap_match = _numbers_match(
                _value(target_row, "quali_gap_to_pole_sec"),
                _value(raw_row, "reconstructed_quali_gap_to_pole_sec"),
            )
            best_lap_match = _numbers_match(
                _value(target_row, "quali_best_lap_time_sec"),
                _value(raw_row, "best_valid_q_lap_sec"),
            )
            settlement_match = _numbers_match(
                _value(settlement_row, "settlement_actual_gap_to_pole_sec"),
                _value(target_row, "quali_gap_to_pole_sec"),
            ) and _numbers_match(
                _value(settlement_row, "settlement_actual_position"),
                _value(target_row, "quali_position"),
                integer=True,
            )
            dashboard_match = _numbers_match(
                _value(dashboard_row, "dashboard_actual_gap_to_pole_sec"),
                _value(settlement_row, "settlement_actual_gap_to_pole_sec"),
            ) and _numbers_match(
                _value(dashboard_row, "dashboard_actual_position"),
                _value(settlement_row, "settlement_actual_position"),
                integer=True,
            )
            has_dashboard = dashboard_row is not None
            parity_status = _driver_parity_status(
                raw_row,
                target_row,
                position_match,
                gap_match,
                best_lap_match,
                settlement_match,
                dashboard_match if has_dashboard else True,
            )
            rows.append(
                {
                    "season": event["season"],
                    "event": event["event"],
                    "event_slug": event["event_slug"],
                    "driver": _value(identity, "driver"),
                    "driver_key": driver_key,
                    "raw_q_best_lap_sec": _value(raw_row, "best_valid_q_lap_sec"),
                    "raw_q_position": _value(raw_row, "reconstructed_quali_position"),
                    "raw_q_gap_to_pole_sec": _value(raw_row, "reconstructed_quali_gap_to_pole_sec"),
                    "stored_target_best_lap_sec": _value(target_row, "quali_best_lap_time_sec"),
                    "stored_target_position": _value(target_row, "quali_position"),
                    "stored_target_gap_to_pole_sec": _value(target_row, "quali_gap_to_pole_sec"),
                    "settlement_actual_position": _value(
                        settlement_row, "settlement_actual_position"
                    ),
                    "settlement_actual_gap_to_pole_sec": _value(
                        settlement_row, "settlement_actual_gap_to_pole_sec"
                    ),
                    "dashboard_actual_position": _value(dashboard_row, "dashboard_actual_position"),
                    "dashboard_actual_gap_to_pole_sec": _value(
                        dashboard_row, "dashboard_actual_gap_to_pole_sec"
                    ),
                    "position_match": bool(position_match),
                    "gap_match": bool(gap_match),
                    "best_lap_match": bool(best_lap_match),
                    "settlement_match": bool(settlement_match),
                    "dashboard_match": bool(dashboard_match) if has_dashboard else pd.NA,
                    "parity_status": parity_status,
                    "legacy_noncanonical": legacy,
                }
            )
    return pd.DataFrame(rows, columns=_driver_comparison_columns())


def _reconstruct_raw_q(config: DataConfig, event: dict[str, Any]) -> pd.DataFrame:
    q_path = build_lap_output_path(config.lap_output_dir, event["season"], event["event"], "Q")
    if not q_path.is_file():
        return pd.DataFrame(columns=_raw_reconstruction_columns())
    raw = pd.read_parquet(q_path)
    cleaned = clean_session_laps(raw, season=event["season"], event=event["event"], session="Q")
    identity_columns = ["driver", "driver_key", "team", "team_key"]
    drivers = cleaned.loc[cleaned["driver_key"].notna(), identity_columns].drop_duplicates(
        "driver_key"
    )
    valid = cleaned[cleaned["is_valid_lap"]].copy()
    best = valid.groupby("driver_key", sort=False)["lap_time_sec"].min()
    result = drivers.reset_index(drop=True)
    result["season"] = event["season"]
    result["event"] = event["event"]
    result["event_slug"] = event["event_slug"]
    result["best_valid_q_lap_sec"] = result["driver_key"].map(best)
    result["raw_q_row_count"] = len(raw)
    result["valid_q_lap_count"] = result["driver_key"].map(valid.groupby("driver_key").size())
    result["valid_q_lap_count"] = result["valid_q_lap_count"].fillna(0).astype(int)
    timed = result[result["best_valid_q_lap_sec"].notna()].sort_values(
        "best_valid_q_lap_sec", kind="stable"
    )
    untimed = result[result["best_valid_q_lap_sec"].isna()]
    result = pd.concat([timed, untimed], ignore_index=True)
    result["reconstructed_quali_position"] = pd.Series(range(1, len(result) + 1), dtype="Int64")
    pole = result["best_valid_q_lap_sec"].min()
    result["reconstructed_pole_lap_sec"] = pole
    result["reconstructed_quali_gap_to_pole_sec"] = result["best_valid_q_lap_sec"] - pole
    result["reconstructed_pole_driver"] = (
        result.loc[result["best_valid_q_lap_sec"].eq(pole), "driver"].iloc[0]
        if pd.notna(pole)
        else None
    )
    return result.reindex(columns=_raw_reconstruction_columns())


def _stored_targets(config: DataConfig, event: dict[str, Any]) -> pd.DataFrame:
    path = target_artifact_path(config, event["season"], event["event"])
    frame = read_parquet(path)
    if frame.empty:
        return pd.DataFrame(columns=_target_columns())
    frame = frame.copy()
    frame["driver_key"] = _driver_key_series(frame)
    return frame.reindex(columns=_target_columns())


def _settlement_actuals(sources: dict[str, Any], event: dict[str, Any]) -> pd.DataFrame:
    settlements = _event_rows(sources["settlements"], event)
    settlements = _live_rows(settlements)
    if settlements.empty:
        return pd.DataFrame(columns=_settlement_actual_columns())
    evaluable = settlements[_bool_series(settlements, "settlement_evaluable")].copy()
    if evaluable.empty:
        return pd.DataFrame(columns=_settlement_actual_columns())
    evaluable["driver_key"] = _driver_key_series(evaluable)
    evaluable["_actual_gap"] = pd.to_numeric(evaluable["actual_gap_sec"], errors="coerce")
    evaluable = evaluable.sort_values("_actual_gap", kind="stable").reset_index(drop=True)
    evaluable["settlement_actual_position"] = pd.Series(range(1, len(evaluable) + 1), dtype="Int64")
    evaluable["settlement_actual_gap_to_pole_sec"] = evaluable["_actual_gap"]
    return evaluable.reindex(columns=_settlement_actual_columns())


def _dashboard_actuals(sources: dict[str, Any], event: dict[str, Any]) -> pd.DataFrame:
    current = sources["dashboard_current"].get("data", {})
    identity = current.get("event_identity") or {}
    if (
        _int_or_none(identity.get("season")) != event["season"]
        or _text_or_none(identity.get("event_slug")) != event["event_slug"]
    ):
        return pd.DataFrame(columns=_dashboard_actual_columns())
    rows = sources["dashboard_settlement"].get("data", {}).get("driver_comparison", [])
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(columns=_dashboard_actual_columns())
    frame = pd.DataFrame(rows)
    frame["driver_key"] = _driver_key_series(frame)
    frame = frame.rename(
        columns={
            "actual_position": "dashboard_actual_position",
            "actual_gap_to_pole_sec": "dashboard_actual_gap_to_pole_sec",
        }
    )
    return frame.reindex(columns=_dashboard_actual_columns())


def _build_checks(
    config: DataConfig,
    sources: dict[str, Any],
    events: list[dict[str, Any]],
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        event_rows = comparison[comparison["event_slug"].astype(str).eq(event["event_slug"])]
        raw = _reconstruct_raw_q(config, event)
        targets = _stored_targets(config, event)
        settlements = _settlement_actuals(sources, event)
        dashboard = _dashboard_actuals(sources, event)
        raw_path = build_lap_output_path(
            config.lap_output_dir,
            event["season"],
            event["event"],
            "Q",
        )
        raw_available = raw_path.is_file()
        rows.append(
            _check(
                event,
                "raw_q_artifact_exists",
                "passed" if raw_available else "failed",
                not raw_available,
                "available" if raw_available else "missing",
                "available",
                "Local raw qualifying lap artifact exists."
                if raw_available
                else "Local raw qualifying lap artifact is missing.",
                "Run ingestion locally before parity audit." if not raw_available else "",
            )
        )
        rows.append(_raw_identity_check(config, event))
        rows.extend(_raw_quality_checks(event, raw))
        rows.extend(_target_checks(event, raw, targets, event_rows))
        rows.extend(_settlement_checks(event, targets, settlements, event_rows))
        rows.extend(_dashboard_checks(event, settlements, dashboard, event_rows))
        rows.extend(_cross_identity_checks(event, raw, targets, settlements, dashboard))
    return pd.DataFrame(rows, columns=_check_columns())


def _raw_identity_check(config: DataConfig, event: dict[str, Any]) -> dict[str, Any]:
    result = validate_raw_session_identity(
        config,
        season=int(event["season"]),
        event=str(event["event"]),
        session="Q",
    )
    return _check(
        event,
        "raw_q_session_identity_matches_event",
        "passed" if result.identity_match else "failed",
        bool(result.blocking),
        json.dumps(result.to_record(config.project_root), sort_keys=True),
        f"season={event['season']};event_slug={event['event_slug']};session=q",
        "Raw Q metadata matches the monitored event." if result.identity_match else result.reason,
        result.recommended_action if result.blocking else "",
    )


def _raw_quality_checks(event: dict[str, Any], raw: pd.DataFrame) -> list[dict[str, Any]]:
    if raw.empty:
        return [
            _check(
                event,
                name,
                "unavailable",
                False,
                "raw_q_missing",
                "reconstructable raw Q",
                "Raw Q artifact missing; reconstruction check unavailable.",
                "Run local ingestion or confirm artifact publication.",
            )
            for name in (
                "raw_q_driver_identity_unique",
                "raw_q_best_lap_reconstructable",
                "raw_q_position_unique",
                "raw_q_pole_gap_consistent",
            )
        ]
    duplicate_keys = int(raw["driver_key"].astype(str).duplicated().sum())
    reconstructable = raw["best_valid_q_lap_sec"].notna().any()
    positions = pd.to_numeric(raw["reconstructed_quali_position"], errors="coerce")
    best_laps = pd.to_numeric(raw["best_valid_q_lap_sec"], errors="coerce").dropna()
    duplicate_positions = bool(positions.duplicated().any())
    tied_best_laps = bool(best_laps.duplicated().any())
    pole_gap_ok = (
        raw["reconstructed_quali_gap_to_pole_sec"].dropna().min() == 0
        and raw.loc[positions.eq(1), "reconstructed_quali_gap_to_pole_sec"].iloc[0] == 0
    )
    return [
        _check(
            event,
            "raw_q_driver_identity_unique",
            "passed" if duplicate_keys == 0 else "failed",
            duplicate_keys > 0,
            duplicate_keys,
            0,
            "Raw Q reconstruction has unique driver keys."
            if duplicate_keys == 0
            else "Raw Q reconstruction has duplicate driver keys.",
            "Repair driver identity normalization." if duplicate_keys else "",
        ),
        _check(
            event,
            "raw_q_best_lap_reconstructable",
            "passed" if reconstructable else "failed",
            not reconstructable,
            int(raw["best_valid_q_lap_sec"].notna().sum()),
            ">=1",
            "At least one best valid Q lap was reconstructed."
            if reconstructable
            else "No valid Q lap could be reconstructed.",
            "Inspect raw Q lap validity columns." if not reconstructable else "",
        ),
        _check(
            event,
            "raw_q_position_unique",
            "failed" if duplicate_positions or tied_best_laps else "passed",
            bool(duplicate_positions or tied_best_laps),
            f"duplicate_positions={duplicate_positions};tied_best_laps={tied_best_laps}",
            "unique deterministic raw Q ranking",
            "Raw Q ranking is unique."
            if not duplicate_positions and not tied_best_laps
            else "Raw Q ranking is ambiguous or duplicated.",
            "Inspect tied or duplicated raw Q best laps."
            if duplicate_positions or tied_best_laps
            else "",
        ),
        _check(
            event,
            "raw_q_pole_gap_consistent",
            "passed" if pole_gap_ok else "failed",
            not pole_gap_ok,
            "pole_gap_zero" if pole_gap_ok else "pole_gap_inconsistent",
            "P1 raw gap is 0.000000",
            "Reconstructed pole gap is internally coherent."
            if pole_gap_ok
            else "Reconstructed pole gap is not internally coherent.",
            "Repair raw-Q reconstruction." if not pole_gap_ok else "",
        ),
    ]


def _target_checks(
    event: dict[str, Any],
    raw: pd.DataFrame,
    targets: pd.DataFrame,
    comparison: pd.DataFrame,
) -> list[dict[str, Any]]:
    target_exists = not targets.empty
    raw_keys = set(raw["driver_key"].dropna().astype(str).tolist())
    target_keys = set(targets["driver_key"].dropna().astype(str).tolist())
    compared = comparison[comparison["stored_target_position"].notna()]
    position_bad = compared["position_match"].eq(False).any() if not compared.empty else False
    gap_bad = compared["gap_match"].eq(False).any() if not compared.empty else False
    best_bad = compared["best_lap_match"].eq(False).any() if not compared.empty else False
    return [
        _check(
            event,
            "stored_target_exists",
            "passed" if target_exists else "failed",
            not target_exists,
            len(targets),
            ">=1 target row",
            "Stored monitored target artifact exists."
            if target_exists
            else "Stored monitored target artifact is missing or empty.",
            "Add monitored targets after qualifying." if not target_exists else "",
        ),
        _check(
            event,
            "stored_target_driver_matches_raw_q",
            "passed" if target_keys <= raw_keys else "failed",
            not target_keys <= raw_keys,
            sorted(target_keys - raw_keys),
            "no target-only driver keys",
            "Stored target drivers are present in reconstructed raw Q."
            if target_keys <= raw_keys
            else "Stored target contains drivers not reconstructed from raw Q.",
            "Repair driver identity normalization." if not target_keys <= raw_keys else "",
        ),
        _parity_check(event, "stored_target_position_matches_raw_q", position_bad),
        _parity_check(event, "stored_target_gap_matches_raw_q", gap_bad),
        _parity_check(event, "stored_target_best_lap_matches_raw_q", best_bad),
    ]


def _settlement_checks(
    event: dict[str, Any],
    targets: pd.DataFrame,
    settlements: pd.DataFrame,
    comparison: pd.DataFrame,
) -> list[dict[str, Any]]:
    target_keys = set(targets["driver_key"].dropna().astype(str).tolist())
    settlement_keys = set(settlements["driver_key"].dropna().astype(str).tolist())
    comparable = comparison[
        comparison["stored_target_position"].notna()
        & comparison["settlement_actual_gap_to_pole_sec"].notna()
    ]
    settlement_bad = (
        comparable["settlement_match"].eq(False).any() if not comparable.empty else False
    )
    repair_settlement = (
        "Repair settlement projection."
        if settlement_bad or not target_keys <= settlement_keys
        else ""
    )
    return [
        _check(
            event,
            "settlement_actual_position_matches_target",
            "passed" if not settlement_bad and target_keys <= settlement_keys else "failed",
            bool(settlement_bad or not target_keys <= settlement_keys),
            sorted(target_keys - settlement_keys),
            "settlement actual positions match target positions",
            "Settlement actual positions match stored targets."
            if not settlement_bad and target_keys <= settlement_keys
            else "Settlement actual positions diverge or are missing for target drivers.",
            repair_settlement,
        ),
        _check(
            event,
            "settlement_actual_gap_matches_target",
            "passed" if not settlement_bad and target_keys <= settlement_keys else "failed",
            bool(settlement_bad or not target_keys <= settlement_keys),
            "mismatch" if settlement_bad else "matched",
            "settlement actual gaps match target gaps",
            "Settlement actual gaps match stored targets."
            if not settlement_bad and target_keys <= settlement_keys
            else "Settlement actual gaps diverge or are missing for target drivers.",
            repair_settlement,
        ),
    ]


def _dashboard_checks(
    event: dict[str, Any],
    settlements: pd.DataFrame,
    dashboard: pd.DataFrame,
    comparison: pd.DataFrame,
) -> list[dict[str, Any]]:
    if dashboard.empty:
        return [
            _check(
                event,
                name,
                "unavailable",
                False,
                "dashboard_not_current_event",
                "dashboard current event rows",
                "Dashboard does not currently expose this event's settlement rows.",
                "",
            )
            for name in (
                "dashboard_actual_position_matches_settlement",
                "dashboard_actual_gap_matches_settlement",
            )
        ]
    settlement_keys = set(settlements["driver_key"].dropna().astype(str).tolist())
    dashboard_keys = set(dashboard["driver_key"].dropna().astype(str).tolist())
    comparable = comparison[
        comparison["dashboard_actual_gap_to_pole_sec"].notna()
        & comparison["settlement_actual_gap_to_pole_sec"].notna()
    ]
    dashboard_bad = comparable["dashboard_match"].eq(False).any() if not comparable.empty else False
    missing = settlement_keys - dashboard_keys
    return [
        _check(
            event,
            "dashboard_actual_position_matches_settlement",
            "passed" if not dashboard_bad and not missing else "failed",
            bool(dashboard_bad or missing),
            sorted(missing),
            "dashboard actual positions match settlement rows",
            "Dashboard actual positions match settlement rows."
            if not dashboard_bad and not missing
            else "Dashboard actual positions diverge or are missing.",
            "Repair dashboard projection." if dashboard_bad or missing else "",
        ),
        _check(
            event,
            "dashboard_actual_gap_matches_settlement",
            "passed" if not dashboard_bad and not missing else "failed",
            bool(dashboard_bad or missing),
            "mismatch" if dashboard_bad else "matched",
            "dashboard actual gaps match settlement rows",
            "Dashboard actual gaps match settlement rows."
            if not dashboard_bad and not missing
            else "Dashboard actual gaps diverge or are missing.",
            "Repair dashboard projection." if dashboard_bad or missing else "",
        ),
    ]


def _cross_identity_checks(
    event: dict[str, Any],
    raw: pd.DataFrame,
    targets: pd.DataFrame,
    settlements: pd.DataFrame,
    dashboard: pd.DataFrame,
) -> list[dict[str, Any]]:
    slug_values = _identity_values("event_slug", raw, targets, settlements, dashboard)
    season_values = _identity_values("season", raw, targets, settlements, dashboard)
    return [
        _check(
            event,
            "event_slug_consistent_across_raw_target_settlement_dashboard",
            "passed" if not slug_values or slug_values == {event["event_slug"]} else "failed",
            bool(slug_values and slug_values != {event["event_slug"]}),
            sorted(slug_values),
            event["event_slug"],
            "Event slugs are consistent across available artifacts."
            if not slug_values or slug_values == {event["event_slug"]}
            else "Event slugs diverge across available artifacts.",
            "Repair event mapping." if slug_values and slug_values != {event["event_slug"]} else "",
        ),
        _check(
            event,
            "season_consistent_across_raw_target_settlement_dashboard",
            "passed" if not season_values or season_values == {str(event["season"])} else "failed",
            bool(season_values and season_values != {str(event["season"])}),
            sorted(season_values),
            event["season"],
            "Seasons are consistent across available artifacts."
            if not season_values or season_values == {str(event["season"])}
            else "Seasons diverge across available artifacts.",
            "Repair season mapping."
            if season_values and season_values != {str(event["season"])}
            else "",
        ),
    ]


def _build_event_summary(
    config: DataConfig,
    sources: dict[str, Any],
    events: list[dict[str, Any]],
    checks: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        event_checks = checks[checks["event_slug"].astype(str).eq(event["event_slug"])]
        event_comparison = comparison[comparison["event_slug"].astype(str).eq(event["event_slug"])]
        raw = _reconstruct_raw_q(config, event)
        targets = _stored_targets(config, event)
        settlements = _settlement_actuals(sources, event)
        dashboard = _dashboard_actuals(sources, event)
        blocking = int(event_checks["blocking"].astype(bool).sum())
        warnings = int(event_checks["status"].astype(str).eq("warning").sum())
        rows.append(
            {
                "season": event["season"],
                "event": event["event"],
                "event_slug": event["event_slug"],
                "legacy_noncanonical": _legacy_noncanonical(sources, event),
                "raw_q_available": not raw.empty,
                "raw_q_driver_count": int(raw["driver_key"].notna().sum()) if not raw.empty else 0,
                "stored_target_driver_count": len(targets),
                "settlement_driver_count": len(settlements),
                "dashboard_driver_count": len(dashboard),
                "position_match_rate": _match_rate(
                    event_comparison,
                    "position_match",
                    denominator_column="stored_target_position",
                ),
                "gap_match_rate": _match_rate(
                    event_comparison,
                    "gap_match",
                    denominator_column="stored_target_gap_to_pole_sec",
                ),
                "best_lap_match_rate": _match_rate(
                    event_comparison,
                    "best_lap_match",
                    denominator_column="stored_target_best_lap_sec",
                ),
                "settlement_match_rate": _match_rate(
                    event_comparison,
                    "settlement_match",
                    denominator_column="settlement_actual_gap_to_pole_sec",
                ),
                "dashboard_match_rate": _match_rate(
                    event_comparison,
                    "dashboard_match",
                    denominator_column="dashboard_actual_gap_to_pole_sec",
                ),
                "event_parity_status": _event_parity_status(event_checks),
                "blocking_failure_count": blocking,
                "warning_count": warnings,
            }
        )
    return pd.DataFrame(rows, columns=_event_summary_columns())


def _build_summary(
    events: list[dict[str, Any]],
    checks: pd.DataFrame,
    event_summary: pd.DataFrame,
) -> dict[str, Any]:
    blocking_events = set(
        checks.loc[checks["blocking"].astype(bool), "event_slug"].astype(str).tolist()
    )
    missing_raw = (
        int(event_summary["raw_q_available"].eq(False).sum()) if not event_summary.empty else 0
    )
    verified = (
        int(event_summary["event_parity_status"].astype(str).eq("parity_verified").sum())
        if not event_summary.empty
        else 0
    )
    root_causes = sorted(
        set(
            status
            for status in event_summary.get("event_parity_status", pd.Series(dtype=str)).astype(str)
            if status not in {"parity_verified", "legacy_descriptive_record"}
        )
    )
    status = "empty"
    if events:
        status = "fail" if blocking_events else "warning" if root_causes else "pass"
    return {
        "status": status,
        "generated_at_utc": utc_now(),
        "events_audited": len(events),
        "events_with_raw_q": int(event_summary["raw_q_available"].astype(bool).sum())
        if not event_summary.empty
        else 0,
        "events_with_verified_parity": verified,
        "events_with_blocking_failures": len(blocking_events),
        "events_with_missing_raw_q": missing_raw,
        "legacy_events_audited": int(event_summary["legacy_noncanonical"].astype(bool).sum())
        if not event_summary.empty
        else 0,
        "root_cause_categories": root_causes,
        "dashboard_safe_for_public_display": "dashboard_projection_mismatch" not in root_causes,
        "recommended_operator_action": _recommended_action(status, root_causes),
        "comparison_tolerance": FLOAT_TOLERANCE,
    }


def _event_parity_status(checks: pd.DataFrame) -> str:
    failed = set(checks.loc[checks["blocking"].astype(bool), "check_name"].astype(str).tolist())
    if not failed:
        return "parity_verified"
    categories = set()
    if "raw_q_artifact_exists" in failed:
        categories.add("raw_q_missing")
    if "stored_target_exists" in failed:
        categories.add("stored_target_missing")
    if any("driver" in item for item in failed):
        categories.add("driver_identity_mismatch")
    if any("identity" in item or "slug" in item or "season" in item for item in failed):
        categories.add("event_identity_mismatch")
    if "stored_target_position_matches_raw_q" in failed:
        categories.add("target_position_mismatch")
    if (
        "stored_target_gap_matches_raw_q" in failed
        or "stored_target_best_lap_matches_raw_q" in failed
    ):
        categories.add("target_gap_mismatch")
    if any(item.startswith("settlement_actual") for item in failed):
        categories.add("settlement_projection_mismatch")
    if any(item.startswith("dashboard_actual") for item in failed):
        categories.add("dashboard_projection_mismatch")
    if len(categories) == 1:
        return next(iter(categories))
    return "multiple_integrity_failures"


def _driver_parity_status(
    raw_row: dict[str, Any] | None,
    target_row: dict[str, Any] | None,
    position_match: bool,
    gap_match: bool,
    best_lap_match: bool,
    settlement_match: bool,
    dashboard_match: bool,
) -> str:
    if raw_row is None:
        return "raw_q_missing"
    if target_row is None:
        return "stored_target_missing"
    failures = [
        not position_match,
        not gap_match,
        not best_lap_match,
        not settlement_match,
        not dashboard_match,
    ]
    return "parity_mismatch" if any(failures) else "parity_verified"


def _recommended_action(status: str, root_causes: list[str]) -> str:
    if status == "empty":
        return "create_monitoring_artifacts_before_running_parity_audit"
    if "raw_q_missing" in root_causes:
        return "ingest_local_qualifying_laps_before_target_parity_audit"
    if root_causes:
        return "investigate_raw_target_settlement_dashboard_parity_failures"
    return "no_operator_action_required"


def _parity_check(event: dict[str, Any], name: str, failed: bool) -> dict[str, Any]:
    return _check(
        event,
        name,
        "failed" if failed else "passed",
        bool(failed),
        "mismatch" if failed else "matched",
        "matched",
        "Stored target values match reconstructed raw Q values."
        if not failed
        else "Stored target values differ from reconstructed raw Q values.",
        "Do not rewrite legacy artifacts; investigate target construction for future events."
        if failed
        else "",
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
        "tolerance": FLOAT_TOLERANCE,
        "reason": reason,
        "recommended_action": action,
    }


def _match_rate(
    frame: pd.DataFrame,
    column: str,
    *,
    denominator_column: str | None = None,
) -> float | None:
    if frame.empty or column not in frame:
        return None
    values = frame
    if denominator_column is not None and denominator_column in values:
        values = values[values[denominator_column].notna()]
    values = values[column].dropna()
    if values.empty:
        return None
    return float(values.astype(bool).mean())


def _raw_q_metadata(config: DataConfig, event: dict[str, Any]) -> dict[str, Any]:
    path = (
        config.session_metadata_output_dir
        / str(event["season"])
        / event["event_slug"]
        / "q_metadata.json"
    )
    return read_json(path) if path.is_file() else {}


def _event_rows(frame: pd.DataFrame, event: dict[str, Any]) -> pd.DataFrame:
    if frame.empty or "event_slug" not in frame:
        return pd.DataFrame()
    rows = frame[frame["event_slug"].astype(str).eq(event["event_slug"])].copy()
    if "season" in rows:
        rows = rows[pd.to_numeric(rows["season"], errors="coerce").eq(event["season"])]
    return rows


def _live_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "prediction_role" not in frame:
        return frame
    return frame[frame["prediction_role"].astype(str).eq(LIVE_POLICY_ROLE)].copy()


def _legacy_noncanonical(sources: dict[str, Any], event: dict[str, Any]) -> bool:
    reconciliation = sources["reconciliation"]
    if not reconciliation.empty and "event_slug" in reconciliation:
        rows = reconciliation[reconciliation["event_slug"].astype(str).eq(event["event_slug"])]
        if not rows.empty:
            return bool(
                rows["event_order_lineage_status"].astype(str).eq(LEGACY_LINEAGE_STATUS).any()
            )
    return event["event_slug"] in {"australia", "great-britain"}


def _identity_values(column: str, *frames: pd.DataFrame) -> set[str]:
    values: set[str] = set()
    for frame in frames:
        if frame.empty or column not in frame:
            continue
        values.update(frame[column].dropna().astype(str).tolist())
    return values


def _row_by_key(frame: pd.DataFrame, driver_key: str) -> dict[str, Any] | None:
    if frame.empty or "driver_key" not in frame:
        return None
    rows = frame[frame["driver_key"].astype(str).eq(driver_key)]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _driver_key_series(frame: pd.DataFrame) -> pd.Series:
    if "driver_key" in frame:
        keys = frame["driver_key"].where(frame["driver_key"].notna(), frame.get("driver"))
    elif "driver" in frame:
        keys = frame["driver"]
    else:
        keys = pd.Series([pd.NA] * len(frame), index=frame.index)
    return keys.astype("string").str.strip().str.lower()


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype("string").str.lower().isin({"true", "1", "yes"})


def _numbers_match(
    observed: Any,
    expected: Any,
    *,
    integer: bool = False,
) -> bool:
    if _is_missing(observed) and _is_missing(expected):
        return True
    if _is_missing(observed) or _is_missing(expected):
        return False
    try:
        left = float(observed)
        right = float(expected)
    except (TypeError, ValueError):
        return str(observed) == str(expected)
    if integer:
        return int(left) == int(right)
    return math.isclose(left, right, abs_tol=FLOAT_TOLERANCE)


def _value(row: dict[str, Any] | None, column: str) -> Any:
    if row is None:
        return None
    value = row.get(column)
    return None if _is_missing(value) else value


def _is_missing(value: Any) -> bool:
    return value is None or (not isinstance(value, (dict, list, tuple)) and pd.isna(value))


def _int_or_none(value: Any) -> int | None:
    if _is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _text_or_none(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _event_name_compatible(event_slug: str, event_name: Any) -> bool:
    if _is_missing(event_name):
        return True
    name_slug = slugify(str(event_name))
    if event_slug in name_slug:
        return True
    # Allows Australia/Australian-style naming without accepting unrelated event names.
    return len(event_slug) > 4 and event_slug[:-1] in name_slug


def _read_json_if_available(path: Path) -> dict[str, Any]:
    return read_json(path) if path.is_file() else {}


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if _is_missing(value):
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(list(value) if isinstance(value, set) else value, sort_keys=True)
    return str(value)


def _runbook(
    summary: dict[str, Any],
    event_summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    lines = [
        "# Qualifying Target Parity Audit Runbook",
        "",
        "This audit compares stored monitored qualifying targets against local raw Q lap "
        "Parquet files.",
        "It does not fetch external data, retrain models, mutate forecasts, or rewrite targets.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Events audited: `{summary['events_audited']}`",
        f"- Comparison tolerance: `{summary['comparison_tolerance']}` seconds",
        f"- Recommended action: `{summary['recommended_operator_action']}`",
        "",
        "## Event Results",
    ]
    if event_summary.empty:
        lines.append("No monitored events were discovered.")
    for _, row in event_summary.iterrows():
        event_rows = comparison[comparison["event_slug"].astype(str).eq(str(row["event_slug"]))]
        raw_drivers = event_rows[event_rows["raw_q_position"].notna()]["driver"].astype(str)
        target_drivers = event_rows[event_rows["stored_target_position"].notna()]["driver"].astype(
            str
        )
        lines.extend(
            [
                "",
                f"### {row['season']} {row['event']}",
                "",
                f"- Event slug: `{row['event_slug']}`",
                f"- Legacy noncanonical: `{row['legacy_noncanonical']}`",
                f"- Local raw Q data exists: `{row['raw_q_available']}`",
                f"- Raw Q reconstructed drivers: `{', '.join(raw_drivers.tolist())}`",
                f"- Stored targets compared: `{', '.join(target_drivers.tolist())}`",
                f"- Qualifying positions equal: `{row['position_match_rate']}`",
                f"- Qualifying gaps equal: `{row['gap_match_rate']}`",
                f"- Best Q laps equal: `{row['best_lap_match_rate']}`",
                f"- Settlement actuals match targets: `{row['settlement_match_rate']}`",
                f"- Dashboard values match settlement: `{row['dashboard_match_rate']}`",
                f"- Event parity status: `{row['event_parity_status']}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _raw_reconstruction_columns() -> list[str]:
    return [
        "season",
        "event",
        "event_slug",
        "driver",
        "driver_key",
        "team",
        "team_key",
        "best_valid_q_lap_sec",
        "reconstructed_quali_position",
        "reconstructed_quali_gap_to_pole_sec",
        "reconstructed_pole_driver",
        "reconstructed_pole_lap_sec",
        "raw_q_row_count",
        "valid_q_lap_count",
    ]


def _target_columns() -> list[str]:
    return [
        "season",
        "event",
        "event_slug",
        "driver",
        "driver_key",
        "team",
        "team_key",
        "quali_gap_to_pole_sec",
        "quali_position",
        "quali_best_lap_time_sec",
    ]


def _settlement_actual_columns() -> list[str]:
    return [
        "season",
        "event",
        "event_slug",
        "driver",
        "driver_key",
        "settlement_actual_position",
        "settlement_actual_gap_to_pole_sec",
    ]


def _dashboard_actual_columns() -> list[str]:
    return [
        "driver",
        "driver_key",
        "dashboard_actual_position",
        "dashboard_actual_gap_to_pole_sec",
    ]


def _driver_comparison_columns() -> list[str]:
    return [
        "season",
        "event",
        "event_slug",
        "driver",
        "driver_key",
        "raw_q_best_lap_sec",
        "raw_q_position",
        "raw_q_gap_to_pole_sec",
        "stored_target_best_lap_sec",
        "stored_target_position",
        "stored_target_gap_to_pole_sec",
        "settlement_actual_position",
        "settlement_actual_gap_to_pole_sec",
        "dashboard_actual_position",
        "dashboard_actual_gap_to_pole_sec",
        "position_match",
        "gap_match",
        "best_lap_match",
        "settlement_match",
        "dashboard_match",
        "parity_status",
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
        "tolerance",
        "reason",
        "recommended_action",
    ]


def _event_summary_columns() -> list[str]:
    return [
        "season",
        "event",
        "event_slug",
        "legacy_noncanonical",
        "raw_q_available",
        "raw_q_driver_count",
        "stored_target_driver_count",
        "settlement_driver_count",
        "dashboard_driver_count",
        "position_match_rate",
        "gap_match_rate",
        "best_lap_match_rate",
        "settlement_match_rate",
        "dashboard_match_rate",
        "event_parity_status",
        "blocking_failure_count",
        "warning_count",
    ]
