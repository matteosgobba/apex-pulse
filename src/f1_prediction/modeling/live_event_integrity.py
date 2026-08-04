"""Event-scoped integrity evidence for live prospective monitoring operations.

The production ledgers are append-only, so whole-file bootstrap checksums cannot
distinguish a legitimate new event from a rewrite of historical evidence.  This
module anchors every pre-existing event independently and compares later state
without becoming a forecast or settlement decision source.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig

LIVE_INTEGRITY_SCHEMA_VERSION = "1.0"
LIVE_VALIDATION_SCHEMA_VERSION = "1.0"
LIVE_VALIDATION_STATUS_FILE = "live_validation_status.json"
LIVE_VALIDATION_DIR = "live_validation"
PRE_WEEKEND_BASELINE_FILE = "pre_weekend_baseline.json"

PROTOCOL_FILE = "prospective_monitoring_protocol.json"
REGISTRY_FILE = "prospective_monitoring_event_registry.csv"
FORECAST_FILE = "prospective_monitoring_forecasts.parquet"
SETTLEMENT_FILE = "prospective_monitoring_settlements.parquet"


class IntegrityClassification(str, Enum):
    """Stable comparator outcomes for an append-only live runtime."""

    UNCHANGED = "UNCHANGED"
    VALID_APPEND = "VALID_APPEND"
    MISSING_PREEXISTING_EVENT = "MISSING_PREEXISTING_EVENT"
    PREEXISTING_EVENT_MUTATED = "PREEXISTING_EVENT_MUTATED"
    STATIC_INVARIANT_CHANGED = "STATIC_INVARIANT_CHANGED"
    INVALID_CHRONOLOGY = "INVALID_CHRONOLOGY"
    OTHER_BLOCKING_INTEGRITY_FAILURE = "OTHER_BLOCKING_INTEGRITY_FAILURE"


SUCCESSFUL_CLASSIFICATIONS = {
    IntegrityClassification.UNCHANGED.value,
    IntegrityClassification.VALID_APPEND.value,
}

# Every table below is append-only at event scope.  The registry is handled
# separately because it also defines the chronology contract.
EVENT_TABLES: dict[str, tuple[str, str]] = {
    "forecast": (FORECAST_FILE, "parquet"),
    "shadow_candidates": ("prospective_monitoring_shadow_candidates.parquet", "parquet"),
    "selection_log": ("prospective_monitoring_selection_log.csv", "csv"),
    "training_manifest": ("prospective_monitoring_training_manifest.csv", "csv"),
    "forecast_integrity_audit": (
        "prospective_monitoring_forecast_integrity_audit.csv",
        "csv",
    ),
    "settlement": (SETTLEMENT_FILE, "parquet"),
    "event_metrics": ("prospective_monitoring_event_metrics.csv", "csv"),
    "shadow_evidence_ledger": (
        "prospective_monitoring_shadow_evidence_ledger.csv",
        "csv",
    ),
    "settlement_integrity_audit": (
        "prospective_monitoring_settlement_integrity_audit.csv",
        "csv",
    ),
}

PER_EVENT_EVIDENCE_FILES = (
    "monitoring_event_manifest.json",
    "monitoring_fp3_features.parquet",
    "monitoring_qualifying_targets.parquet",
    "monitoring_target_coverage.csv",
    "qualifying_entry_list.csv",
    "qualifying_entry_list.json",
)
ENTRY_LIST_EVIDENCE_FILES = (
    "qualifying_entry_list_summary.json",
    "qualifying_entry_list_drivers.csv",
    "qualifying_entry_list_exclusions.csv",
    "qualifying_entry_list_failures.csv",
)


def create_live_integrity_baseline(
    config: DataConfig,
    *,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic event-scoped manifest without writing runtime state."""
    metrics = config.metrics_output_dir
    protocol_path = metrics / PROTOCOL_FILE
    registry_path = metrics / REGISTRY_FILE
    if not protocol_path.is_file() or not registry_path.is_file():
        raise FileNotFoundError("Prospective protocol and event registry are required")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    registry = pd.read_csv(registry_path)
    _validate_registry_shape(registry)
    tables = {
        name: _read_table(metrics / filename, kind)
        for name, (filename, kind) in EVENT_TABLES.items()
    }
    dashboard_events = _dashboard_event_records(config.project_root / "reports/dashboard")
    events: list[dict[str, Any]] = []
    ordered_registry = registry.sort_values(
        ["protocol_name", "monitor_season", "event_order", "event_slug"], kind="stable"
    )
    for _, registry_row in ordered_registry.iterrows():
        protocol_name = str(registry_row["protocol_name"])
        season = int(registry_row["monitor_season"])
        event_slug = str(registry_row["event_slug"])
        event_rows = _event_rows(registry, protocol_name, season, event_slug)
        table_fingerprints = {
            name: _frame_fingerprint(_event_rows(frame, protocol_name, season, event_slug))
            for name, frame in tables.items()
        }
        evidence = _filesystem_evidence(config, season, event_slug)
        if event_slug in dashboard_events:
            evidence["historical_dashboard_event"] = {
                "fingerprint": _canonical_fingerprint(dashboard_events[event_slug]),
                "kind": "semantic_json",
            }
        events.append(
            {
                "protocol_name": protocol_name,
                "season": season,
                "event": str(registry_row.get("event") or event_slug),
                "event_slug": event_slug,
                "event_order": int(registry_row["event_order"]),
                "registry_identity_fingerprint": _frame_fingerprint(event_rows),
                "forecast_fingerprint": table_fingerprints.pop("forecast"),
                "settlement_fingerprint": table_fingerprints.pop("settlement"),
                "event_table_fingerprints": table_fingerprints,
                "evidence_fingerprints": evidence,
                "coverage_status": _coverage_status(registry_row),
            }
        )
    static = _static_invariants(config, protocol)
    return {
        "schema_version": LIVE_INTEGRITY_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(),
        "protocol_name": protocol.get("protocol_name"),
        "protocol_fingerprint": protocol.get("protocol_fingerprint"),
        "modeling_dataset_fingerprint": static["modeling_dataset"]["sha256"],
        "event_count": len(events),
        "static_invariants": static,
        "events": events,
    }


def write_live_integrity_baseline(
    config: DataConfig,
    output_path: Path,
    *,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Write one checkpoint atomically and refuse silent replacement."""
    path = output_path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Integrity checkpoint already exists: {path}")
    payload = create_live_integrity_baseline(config, generated_at_utc=generated_at_utc)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(path, payload)
    return payload


def compare_live_integrity(
    config: DataConfig,
    baseline: Mapping[str, Any] | Path,
) -> dict[str, Any]:
    """Compare current artifacts with an earlier event-scoped baseline."""
    expected = _load_baseline(baseline)
    try:
        current = create_live_integrity_baseline(config)
    except ValueError as exc:
        if "registry" not in str(exc).lower() and "chronology" not in str(exc).lower():
            raise
        return {
            "schema_version": LIVE_INTEGRITY_SCHEMA_VERSION,
            "compared_at_utc": datetime.now(UTC).isoformat(),
            "classification": IntegrityClassification.INVALID_CHRONOLOGY.value,
            "success": False,
            "baseline_fingerprint": _canonical_fingerprint(expected),
            "current_fingerprint": None,
            "baseline_event_count": len(expected["events"]),
            "current_event_count": None,
            "new_events": [],
            "missing_preexisting_events": [],
            "mutated_preexisting_events": [],
            "static_invariant_changes": [],
            "chronology_errors": [str(exc)],
            "artifact_errors": [],
        }
    baseline_events = {_event_key(item): item for item in expected["events"]}
    current_events = {_event_key(item): item for item in current["events"]}
    missing = sorted(set(baseline_events) - set(current_events))
    new = sorted(set(current_events) - set(baseline_events))
    mutated: list[dict[str, Any]] = []
    for key in sorted(set(baseline_events) & set(current_events)):
        changed = _event_changes(baseline_events[key], current_events[key])
        if changed:
            mutated.append({"event_key": key, "changed_fields": changed})
    static_changes = sorted(
        key
        for key, value in expected["static_invariants"].items()
        if value != current["static_invariants"].get(key)
    )
    chronology_errors = _chronology_errors(expected["events"], current["events"])
    artifact_errors = _current_artifact_errors(config)
    if static_changes:
        classification = IntegrityClassification.STATIC_INVARIANT_CHANGED
    elif missing:
        classification = IntegrityClassification.MISSING_PREEXISTING_EVENT
    elif mutated:
        classification = IntegrityClassification.PREEXISTING_EVENT_MUTATED
    elif chronology_errors:
        classification = IntegrityClassification.INVALID_CHRONOLOGY
    elif artifact_errors:
        classification = IntegrityClassification.OTHER_BLOCKING_INTEGRITY_FAILURE
    elif new:
        classification = IntegrityClassification.VALID_APPEND
    else:
        classification = IntegrityClassification.UNCHANGED
    return {
        "schema_version": LIVE_INTEGRITY_SCHEMA_VERSION,
        "compared_at_utc": datetime.now(UTC).isoformat(),
        "classification": classification.value,
        "success": classification.value in SUCCESSFUL_CLASSIFICATIONS,
        "baseline_fingerprint": _canonical_fingerprint(expected),
        "current_fingerprint": _canonical_fingerprint(current),
        "baseline_event_count": len(baseline_events),
        "current_event_count": len(current_events),
        "new_events": [_event_identity(current_events[key]) for key in new],
        "missing_preexisting_events": [_event_identity(baseline_events[key]) for key in missing],
        "mutated_preexisting_events": mutated,
        "static_invariant_changes": static_changes,
        "chronology_errors": chronology_errors,
        "artifact_errors": artifact_errors,
    }


def create_live_validation_checkpoint(
    config: DataConfig,
    baseline_path: Path,
    output_path: Path,
    *,
    stage: str,
) -> dict[str, Any]:
    """Persist a small comparison checkpoint, never artifact copies."""
    if output_path.exists():
        raise FileExistsError(f"Live-validation checkpoint already exists: {output_path}")
    comparison = compare_live_integrity(config, baseline_path)
    current_manifest = create_live_integrity_baseline(config)
    payload = {
        "schema_version": LIVE_VALIDATION_SCHEMA_VERSION,
        "stage": stage,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "baseline_path": baseline_path.name,
        "baseline_fingerprint": comparison["baseline_fingerprint"],
        "integrity": comparison,
        "current_state_manifest": current_manifest,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(output_path, payload)
    return payload


def observe_live_validation_tick(
    config: DataConfig,
    tick: Mapping[str, Any],
    *,
    trigger_source: str,
    scheduler_enabled: bool,
    scheduler_running: bool,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Observe one canonical orchestrator result and authoritative artifacts."""
    if trigger_source not in {"scheduler", "manual", "rehearsal"}:
        raise ValueError("trigger_source must be scheduler, manual, or rehearsal")
    instant = (now or datetime.now(UTC)).astimezone(UTC)
    metrics = config.metrics_output_dir
    status_path = metrics / LIVE_VALIDATION_STATUS_FILE
    previous = _read_json(status_path) or {}
    event_slug = tick.get("event_slug")
    season = tick.get("season")
    same_event = bool(event_slug and previous.get("target_event_slug") == event_slug)
    prior = previous if same_event else {}
    baseline_path = metrics / LIVE_VALIDATION_DIR / PRE_WEEKEND_BASELINE_FILE
    comparison: dict[str, Any] | None = None
    if baseline_path.is_file():
        comparison = compare_live_integrity(config, baseline_path)
    facts = _event_facts(config, str(event_slug or ""))
    sessions = {
        item.get("session"): item
        for item in (tick.get("operational_event") or {}).get("sessions", [])
        if isinstance(item, dict)
    }
    fp3_ready = tick.get("fp3_status") == "ready"
    q_ready = tick.get("qualifying_status") == "ready"
    action = str(tick.get("action_taken") or "none")
    completed_at = tick.get("completed_at_utc")
    timeline = {
        "fp3_scheduled_end_utc": _session_value(sessions, "FP3", "scheduled_end_utc"),
        "first_fp3_readiness_success_at_utc": _first_time(
            prior,
            "timeline",
            "first_fp3_readiness_success_at_utc",
            completed_at if fp3_ready else None,
        ),
        "forecast_workflow_started_at_utc": _first_time(
            prior,
            "timeline",
            "forecast_workflow_started_at_utc",
            tick.get("started_at_utc") if action == "run_before_qualifying" else None,
        ),
        "forecast_committed_at_utc": facts["forecast_created_at_utc"],
        "qualifying_scheduled_start_utc": _session_value(sessions, "Q", "scheduled_start_utc"),
        "qualifying_scheduled_end_utc": _session_value(sessions, "Q", "scheduled_end_utc"),
        "first_qualifying_readiness_success_at_utc": _first_time(
            prior,
            "timeline",
            "first_qualifying_readiness_success_at_utc",
            completed_at if q_ready else None,
        ),
        "settlement_workflow_started_at_utc": _first_time(
            prior,
            "timeline",
            "settlement_workflow_started_at_utc",
            tick.get("started_at_utc") if action == "run_after_qualifying" else None,
        ),
        "settlement_committed_at_utc": facts["settlement_created_at_utc"],
    }
    cache_bytes = int(tick.get("fastf1_cache_bytes") or 0)
    runtime_bytes = int(tick.get("runtime_total_known_bytes") or 0)
    operator, reason = _operator_attention(
        tick,
        comparison,
        scheduler_enabled=scheduler_enabled,
        scheduler_running=scheduler_running,
    )
    payload = {
        "schema_version": LIVE_VALIDATION_SCHEMA_VERSION,
        "target_event": tick.get("event"),
        "target_event_slug": event_slug,
        "season": season,
        "round_number": tick.get("round_number"),
        "event_format": tick.get("event_format"),
        "supported_format": bool((tick.get("operational_event") or {}).get("supported")),
        "phase": _validation_phase(str(tick.get("orchestrator_state_after") or "")),
        "run_id": tick.get("run_id"),
        "trigger_source": trigger_source,
        "orchestrator_state": tick.get("orchestrator_state_after"),
        "pre_weekend_baseline_status": (
            "verified"
            if comparison and comparison["success"]
            else "blocking_failure"
            if comparison
            else "not_initialized"
        ),
        "fp1_observed": bool(prior.get("fp1_observed")) or tick.get("fp1_status") == "ready",
        "fp2_observed": bool(prior.get("fp2_observed")) or tick.get("fp2_status") == "ready",
        "fp3_observed": bool(prior.get("fp3_observed")) or fp3_ready,
        "fp3_data_readiness_status": tick.get("fp3_status"),
        "forecast_status": "available" if facts["forecast_row_count"] else "not_created",
        "forecast_created_at_utc": facts["forecast_created_at_utc"],
        "forecast_reused": bool(facts["forecast_row_count"] and action == "none"),
        "forecast_row_count": facts["forecast_row_count"],
        "forecast_coverage_status": facts["forecast_coverage_status"],
        "qualifying_observed": bool(prior.get("qualifying_observed")) or q_ready,
        "qualifying_data_readiness_status": tick.get("qualifying_status"),
        "settlement_status": "available" if facts["settlement_row_count"] else "not_created",
        "settlement_created_at_utc": facts["settlement_created_at_utc"],
        "settlement_row_count": facts["settlement_row_count"],
        "settlement_coverage_status": facts["settlement_coverage_status"],
        "historical_integrity_status": (
            comparison["classification"] if comparison else "not_initialized"
        ),
        "static_invariants_status": (
            "verified"
            if comparison and not comparison["static_invariant_changes"]
            else "changed"
            if comparison
            else "not_initialized"
        ),
        "cache_status": tick.get("cache_warning_status"),
        "fastf1_cache_bytes": cache_bytes,
        "fastf1_cache_high_watermark_bytes": max(
            cache_bytes, int(prior.get("fastf1_cache_high_watermark_bytes") or 0)
        ),
        "runtime_total_known_bytes": runtime_bytes,
        "runtime_total_high_watermark_bytes": max(
            runtime_bytes, int(prior.get("runtime_total_high_watermark_bytes") or 0)
        ),
        "volume_capacity_bytes": tick.get("volume_capacity_bytes"),
        "operator_attention_required": operator != "NONE",
        "operator_attention_category": operator,
        "operator_attention_reason": reason,
        "scheduler_enabled": scheduler_enabled,
        "scheduler_running": scheduler_running,
        "timeline": timeline,
        "last_updated_at_utc": instant.isoformat(),
    }
    validate_live_validation_status(payload)
    metrics.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(status_path, payload)
    return payload


def validate_live_validation_status(payload: Any) -> dict[str, Any]:
    """Validate the independent, read-only live-observation contract."""
    required = {
        "schema_version",
        "target_event",
        "target_event_slug",
        "season",
        "round_number",
        "event_format",
        "supported_format",
        "phase",
        "trigger_source",
        "orchestrator_state",
        "pre_weekend_baseline_status",
        "forecast_status",
        "forecast_row_count",
        "settlement_status",
        "settlement_row_count",
        "historical_integrity_status",
        "static_invariants_status",
        "operator_attention_required",
        "operator_attention_category",
        "timeline",
        "last_updated_at_utc",
    }
    if not isinstance(payload, dict) or required - set(payload):
        raise ValueError("Live-validation status schema is incomplete")
    if payload.get("schema_version") != LIVE_VALIDATION_SCHEMA_VERSION:
        raise ValueError("Unsupported live-validation status schema")
    if payload.get("trigger_source") not in {"scheduler", "manual", "rehearsal"}:
        raise ValueError("Invalid live-validation trigger source")
    if not isinstance(payload.get("operator_attention_required"), bool):
        raise ValueError("Invalid live-validation attention flag")
    if not isinstance(payload.get("timeline"), dict):
        raise ValueError("Invalid live-validation timeline")
    return payload


def _static_invariants(config: DataConfig, protocol: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        "protocol": config.metrics_output_dir / PROTOCOL_FILE,
        "modeling_dataset": config.modeling_output_dir / "combined/modeling_dataset.parquet",
        "bootstrap_manifest": config.metrics_output_dir / "production_state_manifest.json",
        "bootstrap_receipt": config.metrics_output_dir / "production_bootstrap_receipt.json",
    }
    result = {name: _file_identity(path, config.project_root) for name, path in paths.items()}
    result["protocol"]["declared_protocol_fingerprint"] = protocol.get("protocol_fingerprint")
    return result


def _filesystem_evidence(config: DataConfig, season: int, event_slug: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    event_dir = config.project_root / "data/processed/monitoring" / str(season) / event_slug
    for filename in PER_EVENT_EVIDENCE_FILES:
        path = event_dir / filename
        if path.is_file():
            result[f"monitoring/{filename}"] = _file_identity(path, config.project_root)
    entry_dir = config.metrics_output_dir / "qualifying_entry_lists" / str(season) / event_slug
    for filename in ENTRY_LIST_EVIDENCE_FILES:
        path = entry_dir / filename
        if path.is_file():
            result[f"entry_list/{filename}"] = _file_identity(path, config.project_root)
    return result


def _dashboard_event_records(dashboard_dir: Path) -> dict[str, Any]:
    path = dashboard_dir / "historical_monitoring_summary.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload.get("data", {})
    records: list[Any] = []
    records.extend(data.get("legacy_descriptive_records", []))
    records.extend(data.get("synthetic_rehearsal_records", []))
    valid = data.get("valid_prospective_monitoring", {})
    if isinstance(valid, dict):
        records.extend(valid.get("events", []))
    result: dict[str, Any] = {}
    for record in records:
        identity = record.get("event_identity", {}) if isinstance(record, dict) else {}
        slug = str(identity.get("event_slug") or "")
        if slug:
            result[slug] = record
    return result


def _current_artifact_errors(config: DataConfig) -> list[str]:
    metrics = config.metrics_output_dir
    errors: list[str] = []
    try:
        registry = pd.read_csv(metrics / REGISTRY_FILE)
        _validate_registry_shape(registry)
        forecasts = _read_table(metrics / FORECAST_FILE, "parquet")
        shadow = _read_table(
            metrics / "prospective_monitoring_shadow_candidates.parquet", "parquet"
        )
        settlements = _read_table(metrics / SETTLEMENT_FILE, "parquet")
        selection = _read_table(metrics / "prospective_monitoring_selection_log.csv", "csv")
    except (OSError, ValueError, KeyError) as exc:
        return [str(exc)]
    if not forecasts.empty and {"forecast_id", "prediction_role", "driver"} <= set(forecasts):
        if forecasts.duplicated(["forecast_id", "prediction_role", "driver"]).any():
            errors.append("duplicate forecast identity")
        selection_required = {"forecast_id", "forecast_snapshot_hash"}
        if selection_required - set(selection):
            errors.append("forecast selection integrity evidence missing")
        else:
            forecast_ids = set(forecasts["forecast_id"].astype(str))
            selection_ids = set(selection["forecast_id"].astype(str))
            if not forecast_ids.issubset(selection_ids):
                errors.append("forecast selection evidence is incomplete")
            else:
                from f1_prediction.modeling.prospective_monitoring import (
                    forecast_snapshot_hash,
                )

                for _, row in selection.iterrows():
                    forecast_id = str(row.get("forecast_id") or "")
                    expected = str(row.get("forecast_snapshot_hash") or "")
                    forecast_rows = forecasts[forecasts["forecast_id"].astype(str).eq(forecast_id)]
                    shadow_rows = (
                        shadow[shadow["forecast_id"].astype(str).eq(forecast_id)]
                        if "forecast_id" in shadow
                        else pd.DataFrame()
                    )
                    if (
                        forecast_rows.empty
                        or not expected
                        or forecast_snapshot_hash(forecast_rows, shadow_rows) != expected
                    ):
                        errors.append(f"forecast snapshot integrity failed: {forecast_id}")
    if not settlements.empty:
        required = {
            "settlement_id",
            "forecast_id",
            "settlement_valid",
            "forecast_preexisted_settlement",
            "forecast_fingerprint_valid",
            "forecast_mutation_detected",
        }
        if required - set(settlements):
            errors.append("settlement integrity columns missing")
        else:
            if settlements.duplicated(["settlement_id"]).any():
                errors.append("duplicate settlement identity")
            flags = (
                settlements["settlement_valid"].map(_truthy)
                & settlements["forecast_preexisted_settlement"].map(_truthy)
                & settlements["forecast_fingerprint_valid"].map(_truthy)
                & ~settlements["forecast_mutation_detected"].map(_truthy)
            )
            if not flags.all():
                errors.append("invalid settlement integrity flags")
            if not set(settlements["forecast_id"].astype(str)).issubset(
                set(forecasts.get("forecast_id", pd.Series(dtype=str)).astype(str))
            ):
                errors.append("settlement references missing forecast")
    return errors


def _validate_registry_shape(registry: pd.DataFrame) -> None:
    required = {"protocol_name", "monitor_season", "event_slug", "event_order"}
    if required - set(registry):
        raise ValueError("Monitoring registry schema is incomplete")
    identity = ["protocol_name", "monitor_season", "event_slug", "event_order"]
    if registry.duplicated(identity).any():
        raise ValueError("Monitoring registry contains duplicate event identities")
    if registry.duplicated(["protocol_name", "monitor_season", "event_order"]).any():
        raise ValueError("Monitoring registry contains duplicate chronology positions")


def _chronology_errors(baseline: list[dict[str, Any]], current: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    current_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    baseline_max: dict[tuple[str, int], int] = {}
    baseline_keys = {_event_key(item) for item in baseline}
    for item in baseline:
        group = (str(item["protocol_name"]), int(item["season"]))
        baseline_max[group] = max(baseline_max.get(group, 0), int(item["event_order"]))
    for item in current:
        current_groups.setdefault((str(item["protocol_name"]), int(item["season"])), []).append(
            item
        )
    for group, items in current_groups.items():
        orders = [int(item["event_order"]) for item in items]
        if len(orders) != len(set(orders)):
            errors.append(f"duplicate event order for {group[0]} season {group[1]}")
        for item in items:
            if _event_key(item) not in baseline_keys and int(
                item["event_order"]
            ) <= baseline_max.get(group, 0):
                errors.append(
                    f"new event {item['event_slug']} did not append after baseline chronology"
                )
    return errors


def _event_changes(expected: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    fields = (
        "event",
        "event_order",
        "registry_identity_fingerprint",
        "forecast_fingerprint",
        "settlement_fingerprint",
        "event_table_fingerprints",
        "evidence_fingerprints",
        "coverage_status",
    )
    return [field for field in fields if expected.get(field) != current.get(field)]


def _event_rows(
    frame: pd.DataFrame, protocol_name: str, season: int, event_slug: str
) -> pd.DataFrame:
    if frame.empty:
        return frame.iloc[0:0].copy()
    if "event_slug" in frame:
        mask = frame["event_slug"].astype(str).eq(event_slug)
    else:
        identity_column = next(
            (column for column in ("event_key", "test_event") if column in frame), None
        )
        if identity_column is None:
            return frame.iloc[0:0].copy()
        expected = f"{season}/{event_slug}"
        mask = frame[identity_column].astype(str).eq(expected)
    if "protocol_name" in frame:
        mask &= frame["protocol_name"].astype(str).eq(protocol_name)
    season_column = next(
        (column for column in ("monitor_season", "season", "test_season") if column in frame),
        None,
    )
    if season_column:
        numeric = pd.to_numeric(frame[season_column], errors="coerce")
        mask &= numeric.eq(season)
    return frame.loc[mask].copy()


def _event_facts(config: DataConfig, event_slug: str) -> dict[str, Any]:
    metrics = config.metrics_output_dir
    forecasts = _read_table(metrics / FORECAST_FILE, "parquet")
    settlements = _read_table(metrics / SETTLEMENT_FILE, "parquet")
    registry = _read_table(metrics / REGISTRY_FILE, "csv")
    if event_slug:
        forecast_slug = forecasts.get("event_slug", pd.Series(dtype=str)).astype(str)
        forecasts = forecasts[forecast_slug.eq(event_slug)]
        settlements = settlements[
            settlements.get("event_slug", pd.Series(dtype=str)).astype(str).eq(event_slug)
        ]
        registry_slug = registry.get("event_slug", pd.Series(dtype=str)).astype(str)
        registry = registry[registry_slug.eq(event_slug)]
    else:
        forecasts = forecasts.iloc[0:0]
        settlements = settlements.iloc[0:0]
        registry = registry.iloc[0:0]
    if "diagnostic_only" in forecasts:
        forecasts = forecasts[~forecasts["diagnostic_only"].map(_truthy)]
    if "settlement_valid" in settlements:
        settlements = settlements[settlements["settlement_valid"].map(_truthy)]
    row = registry.iloc[-1] if not registry.empty else pd.Series(dtype=object)
    return {
        "forecast_row_count": len(forecasts),
        "settlement_row_count": len(settlements),
        "forecast_created_at_utc": _earliest(forecasts, "forecast_created_at_utc"),
        "settlement_created_at_utc": _earliest(settlements, "settled_at_utc"),
        "forecast_coverage_status": _coverage_status(row),
        "settlement_coverage_status": str(row.get("target_coverage_status") or "unavailable"),
    }


def _operator_attention(
    tick: Mapping[str, Any],
    comparison: Mapping[str, Any] | None,
    *,
    scheduler_enabled: bool,
    scheduler_running: bool,
) -> tuple[str, str | None]:
    if comparison and not comparison.get("success"):
        return "BLOCKING_INTEGRITY_FAILURE", str(comparison.get("classification"))
    if not scheduler_enabled or not scheduler_running:
        return "SCHEDULER_NOT_RUNNING", "The production scheduler is not enabled and running."
    cache = str(tick.get("cache_warning_status") or "")
    if cache in {"capacity_exceeded", "warning_threshold_exceeded"}:
        return "CACHE_CAPACITY_WARNING", cache
    state = str(tick.get("orchestrator_state_after") or "")
    if state == "UNSUPPORTED_WEEKEND_FORMAT":
        return "UNSUPPORTED_FORMAT", str(tick.get("error_message_safe") or "unsupported format")
    if tick.get("retryable"):
        return "RETRYING_DATA_AVAILABILITY", str(tick.get("retry_reason") or "retry pending")
    if state == "BLOCKED":
        action = str(tick.get("action_considered") or "")
        message = str(tick.get("error_message_safe") or "blocking workflow condition")
        if "entry" in message.lower() and "list" in message.lower():
            return "ENTRY_LIST_BLOCKED", message
        if action == "run_before_qualifying":
            return "FORECAST_WORKFLOW_BLOCKED", message
        if action == "run_after_qualifying":
            return "SETTLEMENT_WORKFLOW_BLOCKED", message
        return "BLOCKING_INTEGRITY_FAILURE", message
    return "NONE", None


def _validation_phase(state: str) -> str:
    if state == "UNSUPPORTED_WEEKEND_FORMAT":
        return "waiting_for_supported_weekend"
    if state in {"SETTLED", "SETTLED_PARTIAL_COVERAGE"}:
        return "settled"
    if state in {"READY_FOR_SETTLEMENT", "QUALIFYING_TIME_ELAPSED_DATA_PENDING"}:
        return "qualifying_validation"
    if state in {"FORECAST_AVAILABLE", "WAITING_FOR_QUALIFYING", "QUALIFYING_INITIAL_GRACE"}:
        return "forecast_available"
    if state in {"READY_FOR_FORECAST", "FP3_TIME_ELAPSED_DATA_PENDING", "FP3_INITIAL_GRACE"}:
        return "fp3_validation"
    return "pre_forecast_observation"


def _coverage_status(row: pd.Series) -> str:
    value = row.get("target_coverage_status")
    if value is not None and not pd.isna(value) and str(value).strip():
        return str(value)
    value = row.get("partial_target_coverage")
    if value is not None and not pd.isna(value):
        return "partial" if _truthy(value) else "complete_or_not_applicable"
    return "unavailable"


def _frame_fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"present": False, "row_count": 0, "sha256": None}
    columns = sorted(str(column) for column in frame.columns)
    rows = []
    for record in frame.reindex(columns=columns).to_dict(orient="records"):
        rows.append({key: _normalize(value) for key, value in record.items()})
    rows.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    return {
        "present": True,
        "row_count": len(rows),
        "sha256": _canonical_fingerprint({"columns": columns, "rows": rows}),
    }


def _canonical_fingerprint(value: Any) -> str:
    rendered = json.dumps(
        _normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(rendered).hexdigest()


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None if math.isnan(value) else str(value)
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC").isoformat()
    if hasattr(value, "item"):
        return _normalize(value.item())
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _read_table(path: Path, kind: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path) if kind == "parquet" else pd.read_csv(path)


def _file_identity(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name,
        "present": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": _sha256_file(path) if path.is_file() else None,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_baseline(value: Mapping[str, Any] | Path) -> dict[str, Any]:
    payload = (
        json.loads(value.read_text(encoding="utf-8")) if isinstance(value, Path) else dict(value)
    )
    if "current_state_manifest" in payload:
        nested = payload["current_state_manifest"]
        if not isinstance(nested, dict):
            raise ValueError("Live-validation checkpoint state manifest is invalid")
        payload = nested
    required = {"schema_version", "static_invariants", "events", "event_count"}
    if required - set(payload) or payload.get("schema_version") != LIVE_INTEGRITY_SCHEMA_VERSION:
        raise ValueError("Unsupported or incomplete live-integrity baseline")
    if not isinstance(payload["events"], list):
        raise ValueError("Live-integrity baseline events must be a list")
    return payload


def _event_key(item: Mapping[str, Any]) -> str:
    return f"{item['protocol_name']}:{int(item['season'])}:{item['event_slug']}"


def _event_identity(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item[key] for key in ("protocol_name", "season", "event", "event_slug", "event_order")
    }


def _earliest(frame: pd.DataFrame, column: str) -> str | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_datetime(frame[column], utc=True, errors="coerce").dropna()
    return values.min().isoformat() if not values.empty else None


def _session_value(sessions: Mapping[str, Any], code: str, field: str) -> Any:
    return (sessions.get(code) or {}).get(field)


def _first_time(previous: Mapping[str, Any], section: str, field: str, value: Any) -> Any:
    existing = (previous.get(section) or {}).get(field)
    return existing or value


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}
