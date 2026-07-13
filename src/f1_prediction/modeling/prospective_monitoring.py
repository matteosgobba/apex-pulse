"""Frozen out-of-season prospective monitoring workflow."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig, FeatureConfig, ModelConfig
from f1_prediction.data.monitoring_onboarding import (
    REGISTRY_ONBOARDING_COLUMNS,
    ensure_registry_columns,
    feature_artifact_path,
    forbidden_target_columns,
    target_artifact_path,
    target_coverage_path,
    validate_target_artifact,
    validate_target_raw_identity,
)
from f1_prediction.data.monitoring_onboarding import (
    artifact_fingerprint as onboarding_artifact_fingerprint,
)
from f1_prediction.data.season_builder import build_combined_dataset_path
from f1_prediction.features.historical_features import add_historical_features
from f1_prediction.modeling.feature_groups import get_feature_columns_for_group
from f1_prediction.modeling.prospective_policy_evaluation import (
    build_frozen_policy_profiles,
    records_for_json,
)
from f1_prediction.modeling.prospective_replay import (
    FP3_CHECKPOINT,
    build_shadow_candidate_rows,
    event_index_from_key,
    event_key_series,
    event_season,
    historical_settings,
    leakage_row,
    parse_event_key,
    prior_events_for,
    season_aware_decision,
    training_manifest_row,
)
from f1_prediction.modeling.season_aware_governance import (
    CANDIDATE_TEMPORAL_POLICY,
    DEFAULT_TEMPORAL_POLICY,
    canonical_candidate_identity,
    canonical_default_identity,
)
from f1_prediction.modeling.splits import ordered_event_keys
from f1_prediction.modeling.tabular import (
    TARGET_COLUMN,
    build_regressors,
    rank_gap_predictions,
    usable_checkpoint_features,
)
from f1_prediction.modeling.temporal_weighting import (
    TemporalWeightingPolicy,
    prepare_temporal_training_data,
)
from f1_prediction.modeling.train_tabular import PREDICTION_COLUMNS
from f1_prediction.utils.paths import ensure_directory

PROTOCOL_VERSION = "1.0"
PROTOCOL_FILE = "prospective_monitoring_protocol.json"
POLICY_RECOMMENDATION = "season_aware_candidate_requires_more_evidence"
TARGET_COLUMNS = ("quali_gap_to_pole_sec", "quali_position", "reached_q3")
FORECAST_KEY_COLUMNS = ("protocol_name", "season", "event_slug", "checkpoint", "driver")
FORECAST_ROLES = (
    "observed_live_policy",
    "uniform_default_shadow",
    "season_aware_weighted_candidate_shadow",
)
PREFLIGHT_FILE_PREFIX = "prospective_monitoring_preflight"
PREFLIGHT_READY = "ready_to_forecast"
PREFLIGHT_BLOCKED = "blocked"
PREFLIGHT_ALREADY_FORECASTED = "already_forecasted"
PREFLIGHT_INVALID_PROTOCOL = "invalid_protocol"
PREFLIGHT_INVALID_REGISTRY_LINEAGE = "invalid_registry_lineage"
EVENT_ORDER_SOURCE = "registry"
EVENT_ORDER_VALID_STATUS = "valid_registry_lineage"
EVENT_ORDER_LEGACY_STATUS = "legacy_noncanonical_event_order"
EVENT_ORDER_MISSING_STATUS = "missing_registry_event_order"
EVENT_ORDER_DUPLICATE_STATUS = "duplicate_registry_event_order"
EVENT_ORDER_INVALID_STATUS = "invalid_registry_event_order"
SYNTHETIC_REHEARSAL_PREFIX = "synthetic-"


@dataclass(frozen=True)
class ProspectiveMonitoringSummary:
    """Paths and status produced by one monitoring workflow command."""

    status: str
    summary_path: Path
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...] = ()
    missing_inputs: tuple[str, ...] = ()
    generation_issues: tuple[str, ...] = ()


def create_prospective_monitoring_protocol(
    config: DataConfig,
    model_config: ModelConfig,
    *,
    protocol_name: str,
    monitor_season: int,
    train_seasons: tuple[int, ...],
    dataset_path: Path | None = None,
    checkpoint: str = FP3_CHECKPOINT,
    uncertainty: str = "conformal_predicted_gap_bucket",
) -> ProspectiveMonitoringSummary:
    """Create or validate a frozen monitoring protocol and event registry."""
    metrics_dir = config.metrics_output_dir
    ensure_directory(metrics_dir)
    source_path = dataset_path or build_combined_dataset_path(config.modeling_output_dir)
    protocol = build_protocol_payload(
        config=config,
        model_config=model_config,
        protocol_name=protocol_name,
        monitor_season=monitor_season,
        train_seasons=train_seasons,
        dataset_path=source_path,
        checkpoint=checkpoint,
        uncertainty=uncertainty,
    )
    protocol_path = metrics_dir / PROTOCOL_FILE
    validation_rows: list[dict[str, object]] = []
    status = "created"
    if protocol_path.is_file():
        existing = _read_json(protocol_path)
        if str(existing.get("protocol_name")) == protocol_name:
            if existing.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
                validation_rows = protocol_mismatch_rows(existing, protocol)
                write_protocol_validation_artifacts(metrics_dir, validation_rows)
                raise ValueError(
                    "Frozen monitoring protocol mismatch for "
                    f"{protocol_name}; use a distinct protocol_name for changed scope"
                )
            protocol = existing
            status = "validated"
        else:
            status = "created_distinct_protocol"
    _write_json(protocol_path, protocol)

    dataset, dataset_status = read_monitoring_dataset(
        resolve_protocol_dataset_path(config, protocol)
    )
    registry = build_event_registry(protocol, dataset)
    readiness, missing = build_readiness(protocol, registry, dataset_status, validation_rows)
    paths = write_protocol_validation_artifacts(metrics_dir, validation_rows)
    registry_path = metrics_dir / "prospective_monitoring_event_registry.csv"
    readiness_path = metrics_dir / "prospective_monitoring_readiness.json"
    missing_path = metrics_dir / "prospective_monitoring_missing_requirements.csv"
    registry.to_csv(registry_path, index=False)
    _write_json(readiness_path, readiness)
    missing.to_csv(missing_path, index=False)
    return ProspectiveMonitoringSummary(
        status=status,
        summary_path=readiness_path,
        table_paths=(protocol_path, readiness_path, registry_path, missing_path, *paths),
        missing_inputs=(
            tuple(missing["requirement"].astype(str).tolist()) if not missing.empty else ()
        ),
    )


def create_prospective_monitoring_preflight(
    config: DataConfig,
    *,
    protocol_name: str,
    season: int,
    event: str,
) -> ProspectiveMonitoringSummary:
    """Validate whether a monitored event is safe to forecast before qualifying."""
    metrics_dir = config.metrics_output_dir
    ensure_directory(metrics_dir)
    result = build_monitoring_preflight_result(
        config,
        protocol_name=protocol_name,
        season=season,
        event=event,
    )
    paths = write_monitoring_preflight_outputs(metrics_dir, result)
    return ProspectiveMonitoringSummary(
        status=str(result["summary"]["status"]),
        summary_path=paths["summary"],
        table_paths=(paths["checks"], paths["failures"], paths["runbook"]),
        missing_inputs=tuple(result["summary"].get("missing_inputs", [])),
    )


def create_prospective_monitoring_forecast(
    config: DataConfig,
    model_config: ModelConfig,
    feature_config: FeatureConfig | None,
    *,
    protocol_name: str,
    event: str,
    uncertainty: str = "conformal_predicted_gap_bucket",
) -> ProspectiveMonitoringSummary:
    """Create an immutable pre-qualification forecast snapshot for one monitored event."""
    metrics_dir = config.metrics_output_dir
    protocol = load_protocol(metrics_dir, protocol_name)
    preflight = create_prospective_monitoring_preflight(
        config,
        protocol_name=protocol_name,
        season=int(protocol["monitor_season"]),
        event=event,
    )
    if preflight.status != PREFLIGHT_READY:
        raise ValueError(f"Monitoring preflight is not ready: {preflight.status}")
    preflight_payload = _read_json(preflight.summary_path)
    dataset, dataset_status = read_monitoring_dataset(
        resolve_protocol_dataset_path(config, protocol)
    )
    if dataset.empty:
        raise ValueError(f"Monitoring dataset unavailable: {dataset_status}")
    registry = load_or_build_registry(metrics_dir, protocol, dataset)
    event_row = resolve_registry_event(registry, event)
    if not registry_event_forecastable(event_row):
        raise ValueError(f"Event is not forecastable: {event}")
    event_key = f"{int(event_row['monitor_season'])}/{event_row['event_slug']}"
    event_order_lineage = resolve_registry_event_order(
        registry,
        protocol_name=protocol_name,
        monitor_season=int(protocol["monitor_season"]),
        event_slug=str(event_row["event_slug"]),
        registry_path=metrics_dir / "prospective_monitoring_event_registry.csv",
        strict=True,
    )
    assert_forecast_not_exists(metrics_dir, protocol_name, str(event_row["event_slug"]))

    dataset = monitoring_dataset_for_forecast(config, protocol, dataset, event_row)
    event_order = monitoring_event_order_keys(dataset, registry, protocol)
    prior_settled_events = settled_event_keys(
        metrics_dir,
        protocol,
        registry,
        current_event_order=int(event_order_lineage["event_order"]),
    )
    legal_train_events = [
        key
        for key in prior_events_for(
            event_key,
            event_order,
            tuple(int(value) for value in protocol["train_seasons"]),
            int(protocol["monitor_season"]),
        )
        if event_season(key) in set(protocol["train_seasons"]) or key in prior_settled_events
    ]
    if not legal_train_events:
        raise ValueError(f"No legal prior training history is available for {event_key}")

    source = train_monitoring_event_sources(
        dataset=dataset,
        row_keys=event_key_series(dataset),
        event_order=event_order,
        event_key=event_key,
        legal_train_events=legal_train_events,
        model_config=model_config,
        feature_config=feature_config,
        test_season=int(protocol["monitor_season"]),
    )
    history = prior_settlement_history(
        metrics_dir,
        protocol,
        registry,
        current_event_order=int(event_row["event_order"]),
    )
    prior_monitoring = prior_monitoring_evidence_summary(
        history,
        event_order_lineage_valid=bool(event_order_lineage["event_order_registry_valid"]),
    )
    profiles = build_frozen_policy_profiles(
        model_config,
        profile_names=("static_baseline", "guarded_baseline", "season_aware_frozen"),
        uncertainty=uncertainty,
    )
    candidate_eligible, selection_reason = season_aware_decision(
        history,
        event_key=event_key,
        profile=profiles["season_aware_frozen"],
    )
    forecast_id = stable_signature(
        {
            "protocol": protocol["protocol_fingerprint"],
            "event_key": event_key,
            "created": utc_now(),
        }
    )
    forecast_created = utc_now()
    manifest = monitoring_manifest_rows(
        source["manifest"],
        protocol=protocol,
        forecast_id=forecast_id,
        forecast_created=forecast_created,
        event_key=event_key,
        event_order_lineage=event_order_lineage,
        prior_monitoring=prior_monitoring,
        preflight=preflight_payload,
    )
    leakage = monitoring_forecast_integrity_rows(
        source["leakage"],
        protocol=protocol,
        forecast_id=forecast_id,
        event_key=event_key,
        event_order_lineage=event_order_lineage,
        prior_monitoring=prior_monitoring,
        preflight=preflight_payload,
    )
    forecasts, shadow = monitoring_prediction_rows(
        source=source,
        protocol=protocol,
        forecast_id=forecast_id,
        forecast_created=forecast_created,
        event_key=event_key,
        event_order=event_order,
        event_order_lineage=event_order_lineage,
        prior_monitoring=prior_monitoring,
        preflight=preflight_payload,
        candidate_eligible=candidate_eligible,
        selection_reason=selection_reason,
    )
    snapshot_hash = forecast_snapshot_hash(forecasts, shadow)
    leakage["forecast_snapshot_hash"] = snapshot_hash
    selection = monitoring_selection_row(
        protocol=protocol,
        forecast_id=forecast_id,
        forecast_created=forecast_created,
        event_key=event_key,
        event_order_lineage=event_order_lineage,
        prior_monitoring=prior_monitoring,
        preflight=preflight_payload,
        candidate_eligible=candidate_eligible,
        selection_reason=selection_reason,
        snapshot_hash=snapshot_hash,
    )

    forecast_path = metrics_dir / "prospective_monitoring_forecasts.parquet"
    shadow_path = metrics_dir / "prospective_monitoring_shadow_candidates.parquet"
    selection_path = metrics_dir / "prospective_monitoring_selection_log.csv"
    manifest_path = metrics_dir / "prospective_monitoring_training_manifest.csv"
    integrity_path = metrics_dir / "prospective_monitoring_forecast_integrity_audit.csv"
    append_parquet(forecast_path, forecasts)
    append_parquet(shadow_path, shadow)
    append_csv(selection_path, pd.DataFrame([selection]))
    append_csv(manifest_path, manifest)
    append_csv(integrity_path, leakage)
    refresh_integrity_outputs(metrics_dir, protocol)
    return ProspectiveMonitoringSummary(
        status="forecast_created",
        summary_path=integrity_path,
        table_paths=(forecast_path, shadow_path, selection_path, manifest_path, integrity_path),
    )


def create_prospective_monitoring_settlement(
    config: DataConfig,
    *,
    protocol_name: str,
    event: str,
) -> ProspectiveMonitoringSummary:
    """Settle a pre-existing monitoring forecast after qualifying targets are available."""
    metrics_dir = config.metrics_output_dir
    protocol = load_protocol(metrics_dir, protocol_name)
    dataset, dataset_status = read_monitoring_dataset(
        resolve_protocol_dataset_path(config, protocol)
    )
    if dataset.empty:
        raise ValueError(f"Monitoring dataset unavailable: {dataset_status}")
    registry = load_or_build_registry(metrics_dir, protocol, dataset)
    event_row = resolve_registry_event(registry, event)
    event_slug = str(event_row["event_slug"])
    forecasts = read_parquet(metrics_dir / "prospective_monitoring_forecasts.parquet")
    if forecasts.empty or "protocol_name" not in forecasts:
        raise ValueError(f"No pre-existing forecast snapshot exists for {event_slug}")
    event_forecasts = forecasts[
        forecasts["protocol_name"].astype(str).eq(protocol_name)
        & forecasts["event_slug"].astype(str).eq(event_slug)
    ].copy()
    if event_forecasts.empty:
        raise ValueError(f"No pre-existing forecast snapshot exists for {event_slug}")
    expected_hash = expected_forecast_hash(metrics_dir, protocol_name, event_slug)
    current_hash = forecast_snapshot_hash(
        event_forecasts,
        read_parquet(metrics_dir / "prospective_monitoring_shadow_candidates.parquet")
        .query("protocol_name == @protocol_name and event_slug == @event_slug")
        .copy(),
    )
    mutation_detected = bool(expected_hash and current_hash != expected_hash)
    if mutation_detected:
        audit = settlement_integrity_rows(
            protocol=protocol,
            event_slug=event_slug,
            forecast_id=str(event_forecasts["forecast_id"].iloc[0]),
            mutation_detected=True,
            fingerprint_valid=True,
            settlement_valid=False,
            blocking_reason="forecast_mutation_detected",
        )
        append_csv(metrics_dir / "prospective_monitoring_settlement_integrity_audit.csv", audit)
        refresh_integrity_outputs(metrics_dir, protocol)
        raise ValueError(f"Forecast snapshot mutation detected for {event_slug}")
    validate_settlement_target_artifact(config, event_row)
    outcomes = monitoring_target_outcomes(config, event_row)
    coverage = monitoring_target_coverage(config, event_row)
    if outcomes.empty:
        outcomes = event_outcomes(dataset, int(protocol["monitor_season"]), event_slug)
        coverage = pd.DataFrame()
    if outcomes.empty:
        raise ValueError(f"Qualifying targets are unavailable for {event_slug}")
    settlements = build_settlement_rows(
        protocol=protocol,
        forecasts=event_forecasts,
        outcomes=outcomes,
        coverage=coverage,
        mutation_detected=False,
    )
    if settlements.empty or not settlements["settlement_evaluable"].astype(bool).any():
        raise ValueError(f"No exact forecast/outcome driver matches for {event_slug}")
    settlement_path = metrics_dir / "prospective_monitoring_settlements.parquet"
    append_parquet(settlement_path, settlements)
    all_settlements = read_parquet(settlement_path)
    all_forecasts = read_parquet(metrics_dir / "prospective_monitoring_forecasts.parquet")
    reconciliation = build_event_order_reconciliation(
        protocol,
        registry,
        all_forecasts,
        all_settlements,
        read_csv(metrics_dir / "prospective_monitoring_shadow_evidence_ledger.csv"),
        metrics_dir=metrics_dir,
    )
    event_metrics = build_event_metrics(all_settlements)
    ledger = build_shadow_evidence_ledger(all_settlements, reconciliation=reconciliation)
    audit = settlement_integrity_rows(
        protocol=protocol,
        event_slug=event_slug,
        forecast_id=str(event_forecasts["forecast_id"].iloc[0]),
        mutation_detected=False,
        fingerprint_valid=True,
        settlement_valid=True,
        blocking_reason="",
    )
    event_metrics.to_csv(metrics_dir / "prospective_monitoring_event_metrics.csv", index=False)
    ledger.to_csv(metrics_dir / "prospective_monitoring_shadow_evidence_ledger.csv", index=False)
    reconciliation.to_csv(
        metrics_dir / "prospective_monitoring_event_order_reconciliation.csv",
        index=False,
    )
    append_csv(metrics_dir / "prospective_monitoring_settlement_integrity_audit.csv", audit)
    refresh_integrity_outputs(metrics_dir, protocol)
    return ProspectiveMonitoringSummary(
        status="settled",
        summary_path=metrics_dir / "prospective_monitoring_event_metrics.csv",
        table_paths=(
            settlement_path,
            metrics_dir / "prospective_monitoring_event_metrics.csv",
            metrics_dir / "prospective_monitoring_shadow_evidence_ledger.csv",
            metrics_dir / "prospective_monitoring_settlement_integrity_audit.csv",
        ),
    )


def create_prospective_monitoring_report(config: DataConfig) -> ProspectiveMonitoringSummary:
    """Create an artifact-driven monitoring report without retraining or settlement."""
    metrics_dir = config.metrics_output_dir
    figures_dir = metrics_dir.parent / "figures"
    ensure_directory(metrics_dir)
    ensure_directory(figures_dir)
    protocol = _read_json_if_exists(metrics_dir / PROTOCOL_FILE) or {}
    registry = read_csv(metrics_dir / "prospective_monitoring_event_registry.csv")
    forecasts = read_parquet(metrics_dir / "prospective_monitoring_forecasts.parquet")
    settlements = read_parquet(metrics_dir / "prospective_monitoring_settlements.parquet")
    preflight_summary = _read_json_if_exists(metrics_dir / f"{PREFLIGHT_FILE_PREFIX}_summary.json")
    integrity = refresh_integrity_outputs(metrics_dir, protocol) if protocol else {}
    ledger = read_csv(metrics_dir / "prospective_monitoring_shadow_evidence_ledger.csv")
    reconciliation = build_event_order_reconciliation(
        protocol,
        registry,
        forecasts,
        settlements,
        ledger,
        metrics_dir=metrics_dir,
    )
    if not settlements.empty:
        reconciled_ledger = build_shadow_evidence_ledger(
            settlements,
            reconciliation=reconciliation,
        )
        reconciled_ledger.to_csv(
            metrics_dir / "prospective_monitoring_shadow_evidence_ledger.csv",
            index=False,
        )
        ledger = reconciled_ledger
    status_by_event = build_status_by_event(
        protocol,
        registry,
        forecasts,
        settlements,
        reconciliation=reconciliation,
    )
    live_summary = build_live_policy_summary(settlements)
    shadow_summary = build_shadow_candidate_summary(settlements)
    gate_timeline = build_gate_timeline(forecasts, reconciliation=reconciliation)
    evidence_growth = build_evidence_growth(settlements, reconciliation=reconciliation)
    summary = build_monitoring_summary_payload(
        protocol=protocol,
        registry=registry,
        forecasts=forecasts,
        settlements=settlements,
        integrity=integrity,
        live_summary=live_summary,
        shadow_summary=shadow_summary,
        reconciliation=reconciliation,
        preflight_summary=preflight_summary,
    )
    paths = {
        "summary": metrics_dir / "prospective_monitoring_summary.json",
        "status": metrics_dir / "prospective_monitoring_status_by_event.csv",
        "live": metrics_dir / "prospective_monitoring_live_policy_summary.csv",
        "shadow": metrics_dir / "prospective_monitoring_shadow_candidate_summary.csv",
        "gate": metrics_dir / "prospective_monitoring_gate_timeline.csv",
        "growth": metrics_dir / "prospective_monitoring_evidence_growth.csv",
    }
    _write_json(paths["summary"], summary)
    status_by_event.to_csv(paths["status"], index=False)
    live_summary.to_csv(paths["live"], index=False)
    shadow_summary.to_csv(paths["shadow"], index=False)
    gate_timeline.to_csv(paths["gate"], index=False)
    evidence_growth.to_csv(paths["growth"], index=False)
    figures, figure_issues = generate_monitoring_figures(
        figures_dir=figures_dir,
        status_by_event=status_by_event,
        live_summary=live_summary,
        shadow_summary=shadow_summary,
        gate_timeline=gate_timeline,
        evidence_growth=evidence_growth,
        integrity_by_event=read_csv(metrics_dir / "prospective_monitoring_integrity_by_event.csv"),
    )
    summary["generated_outputs"]["figures"] = [_relative_report_path(path) for path in figures]
    summary["generation_issues"] = figure_issues
    _write_json(paths["summary"], summary)
    return ProspectiveMonitoringSummary(
        status=str(summary["status"]),
        summary_path=paths["summary"],
        table_paths=tuple(paths.values()),
        figure_paths=tuple(figures),
        generation_issues=tuple(figure_issues),
    )


def build_protocol_payload(
    *,
    config: DataConfig,
    model_config: ModelConfig,
    protocol_name: str,
    monitor_season: int,
    train_seasons: tuple[int, ...],
    dataset_path: Path,
    checkpoint: str,
    uncertainty: str,
) -> dict[str, object]:
    """Build the immutable protocol payload."""
    profiles = build_frozen_policy_profiles(
        model_config,
        profile_names=("static_baseline", "guarded_baseline", "season_aware_frozen"),
        uncertainty=uncertainty,
    )
    season_aware = model_config.champion_policy.season_aware_nested_guarded
    payload: dict[str, object] = {
        "protocol_name": protocol_name,
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": utc_now(),
        "monitor_season": int(monitor_season),
        "train_seasons": [int(value) for value in train_seasons],
        "checkpoint": checkpoint,
        "candidate_identity": canonical_candidate_identity(),
        "default_identity": canonical_default_identity(),
        "observed_live_policy_identity": {
            "static": profiles["static_baseline"].to_payload(),
            "guarded": profiles["guarded_baseline"].to_payload(),
        },
        "frozen_gate_configuration": {
            "min_current_season_prior_events": season_aware.min_current_season_prior_events,
            "min_prior_candidate_folds": season_aware.min_prior_candidate_folds,
            "min_prior_candidate_predictions": season_aware.min_prior_candidate_predictions,
            "improvement_margin_sec": season_aware.improvement_margin_sec,
        },
        "temporal_weighting_configuration": asdict(model_config.temporal_weighting),
        "uncertainty_configuration": asdict(model_config.uncertainty),
        "dataset_path": _display_path(dataset_path, config.project_root),
        "results_path_or_contract": "local reports/metrics monitoring artifacts",
        "event_ordering_contract": (
            "monitoring registry event_order is the sole monitored-season chronology source"
        ),
        "forecast_artifact_contract": "immutable pre-qualification forecast snapshots",
        "settlement_artifact_contract": "post-qualification exact-key settlement only",
        "policy_recommendation": POLICY_RECOMMENDATION,
    }
    payload["protocol_fingerprint"] = protocol_fingerprint(payload)
    return payload


def build_event_registry(protocol: dict[str, Any], dataset: pd.DataFrame) -> pd.DataFrame:
    """Build a deterministic monitored-season event registry."""
    columns = registry_columns()
    if dataset.empty or not {"season", "event_slug", "checkpoint"} <= set(dataset.columns):
        return pd.DataFrame(columns=columns)
    season = int(protocol["monitor_season"])
    monitor = dataset[dataset["season"].astype(int).eq(season)].copy()
    if monitor.empty:
        return pd.DataFrame(columns=columns)
    event_order = ordered_event_keys(dataset)
    rows: list[dict[str, object]] = []
    for key in [key for key in event_order if event_season(key) == season]:
        _, slug = parse_event_key(key)
        event_rows = monitor[monitor["event_slug"].astype(str).eq(slug)].copy()
        fp3 = event_rows[event_rows["checkpoint"].astype(str).eq(protocol["checkpoint"])]
        targets_available = bool(
            not fp3.empty and fp3[list(TARGET_COLUMNS)].notna().all(axis=1).any()
        )
        feature_rows = int(len(fp3))
        event_name = first_value(event_rows, "event", slug)
        forecastable = feature_rows > 0
        rows.append(
            {
                "protocol_name": protocol["protocol_name"],
                "monitor_season": season,
                "event_order": event_order.index(key),
                "event": event_name,
                "event_slug": slug,
                "checkpoint": protocol["checkpoint"],
                "feature_rows_available": feature_rows,
                "driver_rows_available": int(fp3["driver"].nunique()) if "driver" in fp3 else 0,
                "qualifying_targets_available": targets_available,
                "forecast_status": "forecastable" if forecastable else "unavailable",
                "settlement_status": "settleable" if targets_available else "targets_missing",
                "eligibility_evidence_status": "pending_prior_settlements",
                "live_policy_status": "static_or_guarded_reference_frozen",
                "shadow_candidate_status": "diagnostic_only",
                "readiness_blocking_reason": "" if forecastable else "fp3_safe_rows_missing",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_readiness(
    protocol: dict[str, Any],
    registry: pd.DataFrame,
    dataset_status: str,
    validation_rows: list[dict[str, object]],
) -> tuple[dict[str, object], pd.DataFrame]:
    """Summarize protocol readiness and explicit missing requirements."""
    missing_rows: list[dict[str, object]] = []
    if dataset_status != "available":
        missing_rows.append(
            {
                "requirement": "monitoring_dataset",
                "status": dataset_status,
                "blocking": True,
                "details": protocol.get("dataset_path"),
            }
        )
    if registry.empty:
        missing_rows.append(
            {
                "requirement": "monitor_season_event_rows",
                "status": "unavailable",
                "blocking": True,
                "details": protocol.get("monitor_season"),
            }
        )
    if validation_rows:
        missing_rows.extend(
            {
                "requirement": str(row["field"]),
                "status": "protocol_mismatch",
                "blocking": True,
                "details": row["observed_value"],
            }
            for row in validation_rows
        )
    ready = not missing_rows and not registry.empty
    readiness = {
        "status": "ready" if ready else "not_ready",
        "protocol_name": protocol.get("protocol_name"),
        "protocol_fingerprint": protocol.get("protocol_fingerprint"),
        "monitor_season": protocol.get("monitor_season"),
        "train_seasons": protocol.get("train_seasons", []),
        "dataset_status": dataset_status,
        "registered_events": int(len(registry)),
        "forecastable_events": int(registry["forecast_status"].eq("forecastable").sum())
        if not registry.empty
        else 0,
        "settleable_events": int(registry["settlement_status"].eq("settleable").sum())
        if not registry.empty
        else 0,
        "protocol_validation_status": "mismatch" if validation_rows else "valid",
        "fresh_evidence_status": "not_collected",
        "policy_recommendation": POLICY_RECOMMENDATION,
        "generated_at_utc": utc_now(),
    }
    missing = pd.DataFrame(
        missing_rows,
        columns=("requirement", "status", "blocking", "details"),
    )
    return readiness, missing


def build_monitoring_preflight_result(
    config: DataConfig,
    *,
    protocol_name: str,
    season: int,
    event: str,
) -> dict[str, Any]:
    """Build an artifact-only preflight result for one monitored event."""
    metrics_dir = config.metrics_output_dir
    protocol_path = metrics_dir / PROTOCOL_FILE
    protocol = _read_json_if_exists(protocol_path) or {}
    registry_path = metrics_dir / "prospective_monitoring_event_registry.csv"
    registry = read_csv(registry_path)
    forecasts = read_parquet(metrics_dir / "prospective_monitoring_forecasts.parquet")
    settlements = read_parquet(metrics_dir / "prospective_monitoring_settlements.parquet")
    reconciliation = read_csv(metrics_dir / "prospective_monitoring_event_order_reconciliation.csv")
    checks: list[dict[str, object]] = []
    event_slug = slugify_value(event)
    event_order: object = pd.NA
    event_row = pd.Series(dtype=object)
    event_matches = pd.DataFrame()
    if not registry.empty and {"protocol_name", "monitor_season", "event_slug"} <= set(
        registry.columns
    ):
        event_matches = registry[
            registry["protocol_name"].astype(str).eq(str(protocol_name))
            & pd.to_numeric(registry["monitor_season"], errors="coerce").eq(int(season))
            & (
                registry["event_slug"].astype(str).eq(event_slug)
                | registry.get("event", pd.Series(dtype=str))
                .astype(str)
                .map(slugify_value)
                .eq(event_slug)
            )
        ].copy()
        if not event_matches.empty:
            event_row = event_matches.iloc[0]
            event_slug = str(event_row.get("event_slug", event_slug))
            event_order = event_row.get("event_order", pd.NA)
    add_preflight_check(
        checks,
        protocol_name,
        season,
        event,
        event_slug,
        event_order,
        "protocol",
        "protocol_exists",
        bool(protocol_path.is_file()),
        True,
        protocol_path.as_posix(),
        True,
        "Frozen monitoring protocol must exist before preflight.",
        "Run prospective-monitoring-init for the requested protocol.",
    )
    fingerprint_valid = bool(
        protocol
        and str(protocol.get("protocol_name")) == str(protocol_name)
        and protocol.get("protocol_fingerprint") == protocol_fingerprint(protocol)
    )
    add_preflight_check(
        checks,
        protocol_name,
        season,
        event,
        event_slug,
        event_order,
        "protocol",
        "protocol_fingerprint_valid",
        fingerprint_valid,
        True,
        protocol.get("protocol_fingerprint"),
        True,
        "Protocol fingerprint must match frozen fields.",
        "Create a distinct protocol if frozen scope changed.",
    )
    add_preflight_check(
        checks,
        protocol_name,
        season,
        event,
        event_slug,
        event_order,
        "protocol",
        "monitor_season_matches",
        int(protocol.get("monitor_season", -1) or -1) == int(season),
        True,
        protocol.get("monitor_season"),
        int(season),
        "Preflight season must match the frozen monitor season.",
        "Use the protocol monitor season or create a distinct protocol.",
    )
    add_preflight_check(
        checks,
        protocol_name,
        season,
        event,
        event_slug,
        event_order,
        "protocol",
        "candidate_identity_valid",
        protocol.get("candidate_identity") == canonical_candidate_identity(),
        True,
        protocol.get("candidate_identity"),
        canonical_candidate_identity(),
        "Candidate identity must remain canonical.",
        "Recreate or inspect the frozen protocol; do not forecast with mismatched identity.",
    )
    add_preflight_check(
        checks,
        protocol_name,
        season,
        event,
        event_slug,
        event_order,
        "protocol",
        "default_identity_valid",
        protocol.get("default_identity") == canonical_default_identity(),
        True,
        protocol.get("default_identity"),
        canonical_default_identity(),
        "Default identity must remain canonical.",
        "Recreate or inspect the frozen protocol; do not forecast with mismatched identity.",
    )
    gate_valid = frozen_gate_configuration_valid(protocol.get("frozen_gate_configuration", {}))
    add_preflight_check(
        checks,
        protocol_name,
        season,
        event,
        event_slug,
        event_order,
        "protocol",
        "frozen_gate_configuration_valid",
        gate_valid,
        True,
        protocol.get("frozen_gate_configuration"),
        "required gate keys with non-negative values",
        "Frozen gate configuration is required and must not be edited.",
        "Inspect the protocol; create a distinct protocol if frozen gates changed.",
    )
    temporal_valid = temporal_weighting_configuration_valid(
        protocol.get("temporal_weighting_configuration", {})
    )
    add_preflight_check(
        checks,
        protocol_name,
        season,
        event,
        event_slug,
        event_order,
        "protocol",
        "temporal_weighting_configuration_valid",
        temporal_valid,
        True,
        protocol.get("temporal_weighting_configuration"),
        "required temporal weighting keys",
        "Frozen temporal-weighting configuration is required and must not be edited.",
        "Inspect the protocol; create a distinct protocol if temporal weighting changed.",
    )
    event_exists = not event_matches.empty
    add_preflight_check(
        checks,
        protocol_name,
        season,
        event,
        event_slug,
        event_order,
        "event_registry",
        "event_exists_in_registry",
        event_exists,
        True,
        event_slug if event_exists else "",
        event_slug,
        "The event must be registered before forecast.",
        "Run monitoring-register-event after preparing FP3-safe features.",
    )
    add_preflight_check(
        checks,
        protocol_name,
        season,
        event,
        event_slug,
        event_order,
        "event_registry",
        "event_slug_matches",
        event_exists and str(event_row.get("event_slug")) == event_slug,
        True,
        event_row.get("event_slug", ""),
        event_slug,
        "The requested event must resolve to the registered slug.",
        "Use the exact registered event name or inspect the registry.",
    )
    lineage = resolve_registry_event_order(
        registry,
        protocol_name=protocol_name,
        monitor_season=int(season),
        event_slug=event_slug,
        registry_path=registry_path,
        strict=False,
    )
    order_value = lineage.get("event_order", pd.NA)
    event_order = order_value
    add_preflight_check(
        checks,
        protocol_name,
        season,
        event,
        event_slug,
        event_order,
        "event_registry",
        "event_order_present",
        pd.notna(order_value),
        True,
        order_value,
        "present",
        "A registry event_order is required.",
        "Register the event with a valid event order.",
    )
    add_preflight_check(
        checks,
        protocol_name,
        season,
        event,
        event_slug,
        event_order,
        "event_registry",
        "event_order_positive_integer",
        bool(lineage.get("event_order_registry_valid", False)),
        True,
        order_value,
        "positive integer",
        "Registry event_order must be a positive integer.",
        "Correct the registry row by rerunning monitoring-register-event with --event-order.",
    )
    order_unique = registry_order_unique(registry, protocol_name, int(season), order_value)
    add_preflight_check(
        checks,
        protocol_name,
        season,
        event,
        event_slug,
        event_order,
        "event_registry",
        "event_order_unique_within_protocol_and_season",
        order_unique,
        True,
        order_value,
        "unique within protocol and season",
        "The event order must not be shared by another registered monitored event.",
        "Inspect prospective_monitoring_event_registry.csv and correct the duplicate order.",
    )
    add_preflight_check(
        checks,
        protocol_name,
        season,
        event,
        event_slug,
        event_order,
        "event_registry",
        "event_order_source_is_registry",
        lineage.get("event_order_source") == EVENT_ORDER_SOURCE,
        True,
        lineage.get("event_order_source"),
        EVENT_ORDER_SOURCE,
        "Monitored-season event order must come from the frozen registry.",
        "Do not infer event order from dataset or training rows.",
    )
    add_preflight_check(
        checks,
        protocol_name,
        season,
        event,
        event_slug,
        event_order,
        "event_registry",
        "event_order_lineage_valid",
        lineage.get("event_order_lineage_status") == EVENT_ORDER_VALID_STATUS,
        True,
        lineage.get("event_order_lineage_status"),
        EVENT_ORDER_VALID_STATUS,
        "Registry event-order lineage must be valid.",
        "Fix the registry event-order row before forecasting.",
    )
    feature_path = registered_feature_path(config, event_row, int(season), event)
    feature_frame = read_feature_artifact(feature_path)
    feature_forbidden = forbidden_target_columns(feature_frame)
    feature_rows_present = not feature_frame.empty
    feature_fingerprint = (
        onboarding_artifact_fingerprint(feature_path) if feature_path.is_file() else ""
    )
    expected_fingerprint = event_row.get("feature_artifact_fingerprint", "")
    required_identity = {
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "driver",
        "driver_key",
        "team",
        "team_key",
    }
    feature_columns = (
        get_feature_columns_for_group(feature_frame, "base_plus_relative")
        if not feature_frame.empty
        else []
    )
    for check_name, passed, observed, expected, reason, action in [
        (
            "feature_artifact_exists",
            feature_path.is_file(),
            feature_path.as_posix(),
            "existing parquet",
            "A prepared FP3-safe feature artifact is required.",
            "Run monitoring-prepare-event and monitoring-register-event before preflight.",
        ),
        (
            "feature_artifact_fingerprint_valid",
            bool(feature_path.is_file() and str(expected_fingerprint) == feature_fingerprint),
            feature_fingerprint,
            expected_fingerprint,
            "Feature artifact fingerprint must match the registry.",
            "Re-register the event after recreating the safe feature artifact.",
        ),
        (
            "feature_rows_present",
            feature_rows_present,
            len(feature_frame),
            ">0",
            "The feature artifact must contain driver rows.",
            "Rebuild the monitored event feature artifact from local FP sessions.",
        ),
        (
            "driver_rows_present",
            feature_rows_present
            and "driver" in feature_frame
            and int(feature_frame["driver"].nunique()) > 0,
            int(feature_frame["driver"].nunique()) if "driver" in feature_frame else 0,
            ">0",
            "The feature artifact must identify at least one driver.",
            "Inspect feature generation and driver identifiers.",
        ),
        (
            "checkpoint_is_after_fp3",
            feature_rows_present
            and "checkpoint" in feature_frame
            and feature_frame["checkpoint"].astype(str).eq(FP3_CHECKPOINT).all(),
            sorted(feature_frame["checkpoint"].dropna().astype(str).unique().tolist())
            if "checkpoint" in feature_frame
            else [],
            FP3_CHECKPOINT,
            "Monitoring forecast currently supports the after_fp3 checkpoint.",
            "Regenerate the feature artifact with after_fp3 rows.",
        ),
        (
            "forbidden_target_columns_absent",
            not feature_forbidden,
            feature_forbidden,
            "no quali_ or target columns",
            "Pre-qualification features must not contain qualifying targets.",
            "Recreate the feature artifact without target columns.",
        ),
        (
            "qualifying_targets_not_embedded",
            not feature_forbidden,
            feature_forbidden,
            "no embedded targets",
            "Qualifying targets must remain settlement-only.",
            "Remove target columns by rerunning monitoring-prepare-event.",
        ),
        (
            "required_identity_columns_present",
            required_identity <= set(feature_frame.columns),
            sorted(required_identity - set(feature_frame.columns)),
            "no missing identity columns",
            "Forecast joins require stable event and driver identity columns.",
            "Regenerate the feature artifact with required identity columns.",
        ),
        (
            "required_feature_columns_present",
            bool(feature_columns),
            len(feature_columns),
            ">0 base_plus_relative model features",
            "At least one configured model feature must be available.",
            "Inspect feature generation and feature-group configuration.",
        ),
    ]:
        add_preflight_check(
            checks,
            protocol_name,
            season,
            event,
            event_slug,
            event_order,
            "fp3_safe_feature_artifact",
            check_name,
            bool(passed),
            True,
            observed,
            expected,
            reason,
            action,
        )
    target_path = target_artifact_path(config, int(season), str(event_row.get("event", event)))
    coverage_path = target_coverage_path(config, int(season), str(event_row.get("event", event)))
    existing_forecast = not subset_event(forecasts, protocol_name, event_slug).empty
    existing_settlement = not subset_event(settlements, protocol_name, event_slug).empty
    safety_checks = [
        (
            "no_existing_forecast_for_event",
            not existing_forecast,
            existing_forecast,
            False,
            "Forecast snapshots are immutable and must not be overwritten.",
            "Do not rerun forecast; use prospective-monitoring-report or settlement workflow.",
        ),
        (
            "no_existing_settlement_for_event",
            not existing_settlement,
            existing_settlement,
            False,
            "A settled event cannot be forecast again.",
            "Use the existing settlement/report artifacts.",
        ),
        (
            "no_existing_target_artifact_before_forecast",
            not target_path.is_file(),
            target_path.as_posix() if target_path.is_file() else "",
            "absent",
            "Targets must not be present before forecast creation.",
            "Do not forecast this event; start a new clean event or inspect operational ordering.",
        ),
        (
            "no_target_coverage_artifact_before_forecast",
            not coverage_path.is_file(),
            coverage_path.as_posix() if coverage_path.is_file() else "",
            "absent",
            "Coverage ledgers are settlement-side artifacts and must not exist before forecast.",
            "Do not forecast this event; inspect target onboarding order.",
        ),
        (
            "no_current_event_target_accessible_to_forecast",
            not target_path.is_file() and not coverage_path.is_file() and not feature_forbidden,
            {
                "target_artifact": target_path.is_file(),
                "target_coverage": coverage_path.is_file(),
                "embedded_target_columns": feature_forbidden,
            },
            "no target access",
            "Forecast must not be able to access current-event qualifying targets.",
            "Remove target-side artifacts before any clean forecast attempt.",
        ),
    ]
    for check_name, passed, observed, expected, reason, action in safety_checks:
        add_preflight_check(
            checks,
            protocol_name,
            season,
            event,
            event_slug,
            event_order,
            "forecast_safety",
            check_name,
            bool(passed),
            True,
            observed,
            expected,
            reason,
            action,
        )
    event_recon = reconciliation_for_event(reconciliation, protocol_name, event_slug)
    current_legacy = (
        not event_recon.empty
        and event_recon["event_order_lineage_status"]
        .astype(str)
        .eq(EVENT_ORDER_LEGACY_STATUS)
        .any()
    )
    legacy_rows_excluded = (
        reconciliation.empty
        or not reconciliation["event_order_lineage_status"]
        .astype(str)
        .eq(EVENT_ORDER_LEGACY_STATUS)
        .any()
        or not reconciliation.loc[
            reconciliation["event_order_lineage_status"].astype(str).eq(EVENT_ORDER_LEGACY_STATUS),
            "eligible_for_future_prior_evidence_after_reconciliation",
        ]
        .fillna(False)
        .astype(bool)
        .any()
    )
    valid_prior = (
        not reconciliation.empty
        and reconciliation["eligible_for_future_prior_evidence_after_reconciliation"]
        .fillna(False)
        .astype(bool)
        .any()
    )
    legacy_checks = [
        (
            "event_not_marked_legacy_noncanonical",
            not current_legacy,
            event_lineage_status(event_recon),
            f"not {EVENT_ORDER_LEGACY_STATUS}",
            "The current event must not already be a legacy noncanonical artifact.",
            "Do not forecast this event again; use existing descriptive artifacts only.",
            True,
        ),
        (
            "prior_evidence_lineage_valid",
            legacy_rows_excluded,
            prior_evidence_lineage_status(reconciliation)
            if not reconciliation.empty
            else "no_settled_monitoring_evidence",
            "valid or quarantined",
            "Future prior evidence must have valid registry lineage.",
            "Run prospective-monitoring-report and inspect reconciliation failures.",
            True,
        ),
        (
            "legacy_rows_excluded_from_future_prior_evidence",
            legacy_rows_excluded,
            int(
                reconciliation["event_order_lineage_status"]
                .astype(str)
                .eq(EVENT_ORDER_LEGACY_STATUS)
                .sum()
            )
            if not reconciliation.empty
            else 0,
            "legacy rows quarantined",
            "Legacy rows may exist globally but must be excluded from future prior evidence.",
            "Inspect event-order reconciliation before forecasting.",
            True,
        ),
        (
            "valid_prior_monitoring_evidence_available",
            valid_prior,
            valid_prior,
            True,
            "The next forecast currently has zero valid prior monitoring evidence.",
            "Proceed only knowing frozen gates may have no valid monitoring prior evidence.",
            False,
        ),
    ]
    for check_name, passed, observed, expected, reason, action, blocking in legacy_checks:
        add_preflight_check(
            checks,
            protocol_name,
            season,
            event,
            event_slug,
            event_order,
            "legacy_and_lineage_safety",
            check_name,
            bool(passed),
            expected,
            observed,
            expected,
            reason,
            action,
            blocking=blocking,
        )
    checks_frame = pd.DataFrame(checks, columns=preflight_check_columns())
    summary = build_preflight_summary(
        checks_frame=checks_frame,
        protocol=protocol,
        protocol_name=protocol_name,
        season=int(season),
        event=event,
        event_slug=event_slug,
        event_order=event_order,
        existing_forecast=existing_forecast,
        current_legacy=current_legacy,
        valid_prior=valid_prior,
    )
    runbook = build_preflight_runbook(summary, checks_frame)
    failures = checks_frame[checks_frame["status"].astype(str).isin(["fail", "warning"])].copy()
    return {
        "summary": summary,
        "checks": checks_frame,
        "failures": failures,
        "runbook": runbook,
    }


def add_preflight_check(
    rows: list[dict[str, object]],
    protocol_name: str,
    season: int,
    event: str,
    event_slug: str,
    event_order: object,
    check_group: str,
    check_name: str,
    passed: bool,
    expected_passed: object,
    observed_value: object,
    expected_value: object,
    reason: str,
    recommended_action: str,
    *,
    blocking: bool = True,
) -> None:
    status = "pass" if passed else ("fail" if blocking else "warning")
    rows.append(
        {
            "protocol_name": protocol_name,
            "monitor_season": int(season),
            "event": event,
            "event_slug": event_slug,
            "event_order": event_order,
            "check_group": check_group,
            "check_name": check_name,
            "status": status,
            "blocking": bool(blocking),
            "observed_value": json.dumps(_json_clean(observed_value), sort_keys=True),
            "expected_value": json.dumps(_json_clean(expected_value), sort_keys=True),
            "reason": reason,
            "recommended_action": recommended_action,
        }
    )


def build_preflight_summary(
    *,
    checks_frame: pd.DataFrame,
    protocol: dict[str, Any],
    protocol_name: str,
    season: int,
    event: str,
    event_slug: str,
    event_order: object,
    existing_forecast: bool,
    current_legacy: bool,
    valid_prior: bool,
) -> dict[str, object]:
    blocking_failures = checks_frame[
        checks_frame["blocking"].astype(bool) & checks_frame["status"].astype(str).eq("fail")
    ]
    warnings = checks_frame[checks_frame["status"].astype(str).eq("warning")]
    protocol_failures = blocking_failures[
        blocking_failures["check_group"].astype(str).eq("protocol")
    ]
    registry_failures = blocking_failures[
        blocking_failures["check_group"].astype(str).eq("event_registry")
    ]
    if not protocol_failures.empty:
        status = PREFLIGHT_INVALID_PROTOCOL
    elif current_legacy or not registry_failures.empty:
        status = PREFLIGHT_INVALID_REGISTRY_LINEAGE
    elif existing_forecast:
        status = PREFLIGHT_ALREADY_FORECASTED
    elif not blocking_failures.empty:
        status = PREFLIGHT_BLOCKED
    else:
        status = PREFLIGHT_READY
    run_id = stable_signature(
        {
            "protocol_name": protocol_name,
            "season": season,
            "event_slug": event_slug,
            "generated_at": utc_now(),
        }
    )
    summary_path = f"reports/metrics/{PREFLIGHT_FILE_PREFIX}_summary.json"
    return {
        "status": status,
        "preflight_run_id": run_id,
        "preflight_summary_path": summary_path,
        "protocol_name": protocol_name,
        "protocol_fingerprint": protocol.get("protocol_fingerprint"),
        "season": int(season),
        "event": event,
        "event_slug": event_slug,
        "event_order": _json_clean(event_order),
        "forecast_allowed": status == PREFLIGHT_READY,
        "blocking_check_count": int(len(blocking_failures)),
        "warning_check_count": int(len(warnings)),
        "prior_monitoring_evidence_status": (
            "valid_prior_monitoring_evidence_available"
            if valid_prior
            else "zero_valid_prior_monitoring_evidence"
        ),
        "legacy_exclusion_status": legacy_exclusion_status(checks_frame),
        "prospective_monitoring_preflight_available": True,
        "prospective_monitoring_preflight_status": status,
        "prospective_monitoring_preflight_blocking_check_count": int(len(blocking_failures)),
        "prospective_monitoring_next_event_ready_to_forecast": status == PREFLIGHT_READY,
        "prospective_monitoring_preflight_runbook_path": (
            f"reports/metrics/{PREFLIGHT_FILE_PREFIX}_runbook.md"
        ),
        "next_required_command": next_preflight_command(protocol_name, event, status),
        "generated_at_utc": utc_now(),
    }


def build_preflight_runbook(summary: dict[str, object], checks: pd.DataFrame) -> str:
    lines = [
        "# Prospective Monitoring Preflight Runbook",
        "",
        f"Status: `{summary['status']}`",
        f"Protocol: `{summary['protocol_name']}`",
        f"Event: `{summary['event']}` (`{summary['event_slug']}`)",
        "",
    ]
    if summary["status"] == PREFLIGHT_READY:
        lines.extend(
            [
                "Safe next command:",
                "",
                "```bash",
                "python -m f1_prediction.cli prospective-monitoring-forecast "
                f'--protocol-name {summary["protocol_name"]} --event "{summary["event"]}"',
                "```",
                "",
            ]
        )
    else:
        failures = checks[checks["status"].astype(str).eq("fail")].copy()
        lines.append("Corrective steps:")
        if failures.empty:
            lines.append("- Inspect preflight summary; no failed checks were recorded.")
        for _, row in failures.iterrows():
            lines.append(f"- `{row['check_name']}`: {row['recommended_action']}")
        lines.append("")
    lines.extend(
        [
            "Operational guardrails:",
            "- Do not ingest or add Q targets before the forecast is created.",
            "- Do not run settlement before targets are separately added.",
            "- Do not rerun or overwrite an existing forecast.",
            "- A `ready_to_forecast` result is required before forecasting.",
            "",
        ]
    )
    return "\n".join(lines)


def write_monitoring_preflight_outputs(
    metrics_dir: Path,
    result: dict[str, Any],
) -> dict[str, Path]:
    paths = {
        "summary": metrics_dir / f"{PREFLIGHT_FILE_PREFIX}_summary.json",
        "checks": metrics_dir / f"{PREFLIGHT_FILE_PREFIX}_checks.csv",
        "failures": metrics_dir / f"{PREFLIGHT_FILE_PREFIX}_failures.csv",
        "runbook": metrics_dir / f"{PREFLIGHT_FILE_PREFIX}_runbook.md",
    }
    _write_json(paths["summary"], result["summary"])
    result["checks"].to_csv(paths["checks"], index=False)
    result["failures"].to_csv(paths["failures"], index=False)
    paths["runbook"].write_text(str(result["runbook"]), encoding="utf-8")
    return paths


def preflight_check_columns() -> list[str]:
    return [
        "protocol_name",
        "monitor_season",
        "event",
        "event_slug",
        "event_order",
        "check_group",
        "check_name",
        "status",
        "blocking",
        "observed_value",
        "expected_value",
        "reason",
        "recommended_action",
    ]


def registry_event_forecastable(event_row: pd.Series) -> bool:
    """Return whether a registry row is forecastable under old or onboarding schemas."""
    if "forecastable" in event_row and not pd.isna(event_row.get("forecastable")):
        return bool(event_row.get("forecastable"))
    return str(event_row.get("forecast_status")) == "forecastable"


def monitoring_dataset_for_forecast(
    config: DataConfig,
    protocol: dict[str, Any],
    dataset: pd.DataFrame,
    event_row: pd.Series,
) -> pd.DataFrame:
    """Combine frozen historical rows with one registered targetless feature artifact."""
    feature_path = resolve_registered_path(
        config,
        event_row.get("feature_artifact_path"),
    )
    if feature_path is None or not feature_path.is_file():
        return dataset
    features = pd.read_parquet(feature_path)
    forbidden = forbidden_target_columns(features)
    if forbidden:
        raise ValueError(
            "Registered monitoring feature artifact contains forbidden target columns: "
            + ", ".join(forbidden)
        )
    expected_fingerprint = event_row.get("feature_artifact_fingerprint")
    if expected_fingerprint and not pd.isna(expected_fingerprint):
        actual_fingerprint = onboarding_artifact_fingerprint(feature_path)
        if str(expected_fingerprint) != actual_fingerprint:
            raise ValueError("Registered monitoring feature artifact fingerprint is invalid")
    event_key = f"{int(protocol['monitor_season'])}/{event_row['event_slug']}"
    current_mask = event_key_series(dataset).astype(str).eq(event_key)
    historical = dataset.loc[~current_mask].copy()
    feature_rows = features.copy()
    target_columns = (
        "quali_position",
        "quali_best_lap_time_sec",
        "quali_gap_to_pole_sec",
        "reached_q2",
        "reached_q3",
    )
    for column in target_columns:
        if column not in feature_rows:
            feature_rows[column] = pd.NA
    return pd.concat([historical, feature_rows], ignore_index=True, sort=False)


def validate_settlement_target_artifact(config: DataConfig, event_row: pd.Series) -> None:
    """Require a separate valid target artifact before settlement."""
    feature_path = event_row.get("feature_artifact_path")
    if feature_path is None or pd.isna(feature_path) or not str(feature_path).strip():
        return
    season = int(event_row["monitor_season"])
    event = str(event_row.get("event", event_row["event_slug"]))
    target_valid, reason = validate_target_artifact(config, season, event)
    if not target_valid:
        raise ValueError(f"Valid separate target artifact is required for settlement: {reason}")
    identity_valid, identity_reason = validate_target_raw_identity(config, season, event)
    if not identity_valid:
        raise ValueError(
            "Raw Q identity verification is required before settlement: "
            f"{identity_reason}. Inspect metadata and re-ingest the correct Q session before "
            "retrying settlement."
        )


def monitoring_target_outcomes(config: DataConfig, event_row: pd.Series) -> pd.DataFrame:
    """Read settlement-only targets and add the monitoring checkpoint key."""
    season = int(event_row["monitor_season"])
    event = str(event_row.get("event", event_row["event_slug"]))
    path = target_artifact_path(config, season, event)
    if not path.is_file():
        return pd.DataFrame()
    targets = pd.read_parquet(path).copy()
    targets["checkpoint"] = FP3_CHECKPOINT
    return targets


def monitoring_target_coverage(config: DataConfig, event_row: pd.Series) -> pd.DataFrame:
    """Read the per-driver target coverage ledger for an onboarded event."""
    season = int(event_row["monitor_season"])
    event = str(event_row.get("event", event_row["event_slug"]))
    path = target_coverage_path(config, season, event)
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def train_monitoring_event_sources(
    *,
    dataset: pd.DataFrame,
    row_keys: pd.Series,
    event_order: list[str],
    event_key: str,
    legal_train_events: list[str],
    model_config: ModelConfig,
    feature_config: FeatureConfig | None,
    test_season: int,
) -> dict[str, Any]:
    """Fit monitoring candidates with targetless current-event rows."""
    fold_scope = dataset[row_keys.isin([*legal_train_events, event_key])].copy()
    fold_scope = add_historical_features(
        fold_scope,
        historical_settings(feature_config),
        excluded_target_events={event_key},
    )
    fold_keys = event_key_series(fold_scope)
    train = fold_scope[fold_keys.isin(legal_train_events)].copy()
    test = fold_scope[fold_keys.eq(event_key)].copy()
    if train.empty or test.empty:
        raise ValueError(f"Monitoring event {event_key} must have train and test rows")
    feature_columns = get_feature_columns_for_group(fold_scope, "base_plus_relative")
    static_predictions, static_fit = fit_monitoring_source_candidate(
        train=train,
        test=test,
        event_order=event_order,
        event_key=event_key,
        model_config=model_config,
        feature_columns=feature_columns,
        temporal_policy=TemporalWeightingPolicy.uniform,
    )
    weighted_predictions, weighted_fit = fit_monitoring_source_candidate(
        train=train,
        test=test,
        event_order=event_order,
        event_key=event_key,
        model_config=model_config,
        feature_columns=feature_columns,
        temporal_policy=TemporalWeightingPolicy.current_season_only_with_prior,
    )
    manifest = [
        training_manifest_row(
            event_key=event_key,
            test=test,
            train=train,
            legal_train_events=legal_train_events,
            fit_payload=static_fit,
            model_config=model_config,
            temporal_policy="uniform",
            policy_profile="static_baseline",
            test_season=test_season,
        ),
        training_manifest_row(
            event_key=event_key,
            test=test,
            train=train,
            legal_train_events=legal_train_events,
            fit_payload=weighted_fit,
            model_config=model_config,
            temporal_policy="current_season_only_with_prior",
            policy_profile="season_aware_frozen",
            test_season=test_season,
        ),
    ]
    leakage = [leakage_row(row, event_order=event_order) for row in manifest]
    return {
        "event_key": event_key,
        "test": test,
        "static": static_predictions,
        "weighted": weighted_predictions,
        "manifest": manifest,
        "leakage": leakage,
    }


def fit_monitoring_source_candidate(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    event_order: list[str],
    event_key: str,
    model_config: ModelConfig,
    feature_columns: list[str],
    temporal_policy: TemporalWeightingPolicy,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Train one FP3 RF candidate and predict targetless monitored rows."""
    temporal = prepare_temporal_training_data(
        train,
        test_event=event_key,
        event_order=event_order,
        config=model_config.temporal_weighting,
        policy=temporal_policy,
    )
    train_rows = temporal.train[temporal.train["checkpoint"].eq(FP3_CHECKPOINT)].dropna(
        subset=[TARGET_COLUMN]
    )
    test_rows = test[test["checkpoint"].eq(FP3_CHECKPOINT)].copy()
    if train_rows.empty or test_rows.empty:
        raise ValueError("Monitoring forecast requires target-bearing training rows and test rows")
    allowed = set(feature_columns)
    features = [
        column
        for column in usable_checkpoint_features(train_rows, FP3_CHECKPOINT)
        if column in allowed
    ]
    if not features:
        raise ValueError(f"No usable numeric features for {FP3_CHECKPOINT}")
    estimator = build_regressors(model_config)["random_forest"]
    weights = temporal.sample_weights.reindex(train_rows.index).astype(float)
    estimator.fit(
        train_rows[features],
        train_rows[TARGET_COLUMN],
        regressor__sample_weight=weights,
    )
    columns = [
        column for column in (*PREDICTION_COLUMNS, "driver_key", "team_key") if column in test_rows
    ]
    frame = test_rows.loc[:, columns].copy()
    target_columns = (
        "quali_position",
        "quali_best_lap_time_sec",
        "quali_gap_to_pole_sec",
        "reached_q2",
        "reached_q3",
    )
    for column in target_columns:
        if column not in frame:
            frame[column] = pd.NA
    frame["model_name"] = "random_forest"
    frame["predicted_quali_gap_to_pole_sec"] = estimator.predict(test_rows[features])
    frame["predicted_quali_position"] = rank_gap_predictions(frame)
    frame["predicted_reached_q3"] = frame["predicted_quali_position"].le(10).astype("int8")
    frame["feature_group"] = "base_plus_relative"
    frame["temporal_weighting_policy"] = temporal_policy.value
    frame["source_artifact_kind"] = "prospective_monitoring"
    frame["prediction_source_identity"] = json.dumps(
        {
            "family": "ablation",
            "model_name": "random_forest",
            "feature_group": "base_plus_relative",
            "temporal_weighting_policy": temporal_policy.value,
            "event_key": event_key,
        },
        sort_keys=True,
    )
    return frame, {
        "feature_columns": features,
        "sample_weight_summary": temporal.summary,
    }


def monitoring_prediction_rows(
    *,
    source: dict[str, Any],
    protocol: dict[str, Any],
    forecast_id: str,
    forecast_created: str,
    event_key: str,
    event_order: list[str],
    event_order_lineage: dict[str, object],
    prior_monitoring: dict[str, object],
    preflight: dict[str, Any],
    candidate_eligible: bool,
    selection_reason: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build live and diagnostic forecast rows without current-event targets."""
    live = role_frame(
        source["static"],
        role="observed_live_policy",
        diagnostic_only=False,
        protocol=protocol,
        forecast_id=forecast_id,
        forecast_created=forecast_created,
        event_key=event_key,
        event_order=event_order,
        event_order_lineage=event_order_lineage,
        prior_monitoring=prior_monitoring,
        preflight=preflight,
        temporal_policy=DEFAULT_TEMPORAL_POLICY,
        candidate_eligible=candidate_eligible,
        selection_reason="observed_static_policy_reference",
        live_policy_selected=True,
        selection_is_live=True,
    )
    shadow_source = build_shadow_candidate_rows(
        source=source,
        train_seasons=tuple(int(value) for value in protocol["train_seasons"]),
        test_season=int(protocol["monitor_season"]),
        event_order=event_order,
    )
    uniform = role_frame(
        shadow_source[shadow_source["shadow_role"].eq("uniform_default")],
        role="uniform_default_shadow",
        diagnostic_only=True,
        protocol=protocol,
        forecast_id=forecast_id,
        forecast_created=forecast_created,
        event_key=event_key,
        event_order=event_order,
        event_order_lineage=event_order_lineage,
        prior_monitoring=prior_monitoring,
        preflight=preflight,
        temporal_policy=DEFAULT_TEMPORAL_POLICY,
        candidate_eligible=candidate_eligible,
        selection_reason="diagnostic_shadow_default",
        live_policy_selected=False,
        selection_is_live=False,
    )
    weighted = role_frame(
        shadow_source[shadow_source["shadow_role"].eq("season_aware_weighted_candidate")],
        role="season_aware_weighted_candidate_shadow",
        diagnostic_only=True,
        protocol=protocol,
        forecast_id=forecast_id,
        forecast_created=forecast_created,
        event_key=event_key,
        event_order=event_order,
        event_order_lineage=event_order_lineage,
        prior_monitoring=prior_monitoring,
        preflight=preflight,
        temporal_policy=CANDIDATE_TEMPORAL_POLICY,
        candidate_eligible=candidate_eligible,
        selection_reason=selection_reason,
        live_policy_selected=False,
        selection_is_live=False,
    )
    forecasts = pd.concat([live, uniform, weighted], ignore_index=True, sort=False)
    shadow = forecasts[forecasts["diagnostic_only"].astype(bool)].copy()
    return forecasts.loc[:, forecast_columns()], shadow.loc[:, forecast_columns()]


def role_frame(
    frame: pd.DataFrame,
    *,
    role: str,
    diagnostic_only: bool,
    protocol: dict[str, Any],
    forecast_id: str,
    forecast_created: str,
    event_key: str,
    event_order: list[str],
    event_order_lineage: dict[str, object],
    prior_monitoring: dict[str, object],
    preflight: dict[str, Any],
    temporal_policy: str,
    candidate_eligible: bool,
    selection_reason: str,
    live_policy_selected: bool,
    selection_is_live: bool,
) -> pd.DataFrame:
    """Normalize one forecast role."""
    result = frame.copy()
    season, slug = parse_event_key(event_key)
    if "prediction_gap_sec" not in result:
        result["prediction_gap_sec"] = result.get("predicted_quali_gap_to_pole_sec")
    for column in ("driver", "driver_key", "team", "team_key"):
        if column not in result:
            result[column] = pd.NA
    result["protocol_name"] = protocol["protocol_name"]
    result["protocol_fingerprint"] = protocol["protocol_fingerprint"]
    result["forecast_id"] = forecast_id
    result["forecast_created_at_utc"] = forecast_created
    result["monitor_season"] = protocol["monitor_season"]
    result["event_order"] = event_order_lineage.get("event_order", pd.NA)
    result["event_order_source"] = event_order_lineage.get("event_order_source", EVENT_ORDER_SOURCE)
    result["event_order_registry_valid"] = bool(
        event_order_lineage.get("event_order_registry_valid", False)
    )
    result["event_order_registry_path"] = event_order_lineage.get("event_order_registry_path")
    result["event_order_registry_protocol_name"] = event_order_lineage.get(
        "event_order_registry_protocol_name"
    )
    result["event_order_registry_monitor_season"] = event_order_lineage.get(
        "event_order_registry_monitor_season"
    )
    result["event_order_lineage_status"] = event_order_lineage.get(
        "event_order_lineage_status",
        EVENT_ORDER_INVALID_STATUS,
    )
    result["fold_id"] = event_index_from_key(event_key)
    result["season"] = season
    result["event"] = result.get("event", slug)
    result["event_slug"] = slug
    result["checkpoint"] = protocol["checkpoint"]
    result["prediction_role"] = role
    result["diagnostic_only"] = diagnostic_only
    result["family"] = "ablation"
    result["model_name"] = "random_forest"
    result["feature_group"] = "base_plus_relative"
    result["temporal_weighting_policy"] = temporal_policy
    result["source_identity"] = result.get(
        "source_identity",
        json.dumps({**canonical_default_identity(), "event_key": event_key}, sort_keys=True),
    )
    result["source_lineage_valid"] = True
    result["candidate_eligible_under_frozen_gates"] = bool(candidate_eligible)
    result["candidate_selection_reason"] = selection_reason
    result["live_policy_selected"] = bool(live_policy_selected)
    result["selection_is_live"] = bool(selection_is_live)
    result["selection_is_counterfactual"] = bool(
        role == "season_aware_weighted_candidate_shadow" and candidate_eligible
    )
    result["training_completed"] = True
    result["training_row_count"] = result.get("training_row_count", pd.NA)
    result["training_event_count"] = result.get("training_event_count", pd.NA)
    result["training_event_keys"] = result.get("training_event_keys", "[]")
    result["training_seasons"] = result.get("training_seasons", "[]")
    result["current_season_prior_event_count"] = result.get(
        "training_current_season_prior_event_count",
        0,
    )
    result["training_effective_sample_size"] = result.get("training_effective_sample_size", pd.NA)
    result["current_event_excluded_from_training"] = True
    result["future_same_season_events_excluded"] = True
    result["future_seasons_excluded"] = True
    result["current_event_target_accessed"] = False
    result["forecast_integrity_status"] = "valid"
    for column, value in prior_monitoring.items():
        result[column] = value
    result["preflight_run_id"] = preflight.get("preflight_run_id")
    result["preflight_status"] = preflight.get("status")
    result["preflight_summary_path"] = preflight.get("preflight_summary_path")
    result["actual_gap_sec"] = pd.NA
    result["absolute_error_sec"] = pd.NA
    return result


def monitoring_manifest_rows(
    manifest_rows: list[dict[str, object]],
    *,
    protocol: dict[str, Any],
    forecast_id: str,
    forecast_created: str,
    event_key: str,
    event_order_lineage: dict[str, object],
    prior_monitoring: dict[str, object],
    preflight: dict[str, Any],
) -> pd.DataFrame:
    """Decorate training manifest rows with protocol metadata."""
    frame = pd.DataFrame(manifest_rows)
    if frame.empty:
        return pd.DataFrame(columns=manifest_columns())
    frame["protocol_name"] = protocol["protocol_name"]
    frame["protocol_fingerprint"] = protocol["protocol_fingerprint"]
    frame["forecast_id"] = forecast_id
    frame["forecast_created_at_utc"] = forecast_created
    frame["monitor_season"] = protocol["monitor_season"]
    frame["event_key"] = event_key
    for column, value in event_order_lineage.items():
        frame[column] = value
    for column, value in prior_monitoring.items():
        frame[column] = value
    frame["preflight_run_id"] = preflight.get("preflight_run_id")
    frame["preflight_status"] = preflight.get("status")
    frame["preflight_summary_path"] = preflight.get("preflight_summary_path")
    frame["current_event_excluded_from_training"] = ~frame["current_event_in_training"].astype(bool)
    frame["future_same_season_events_excluded"] = True
    frame["future_seasons_excluded"] = True
    return frame.reindex(columns=manifest_columns())


def monitoring_forecast_integrity_rows(
    leakage_rows: list[dict[str, object]],
    *,
    protocol: dict[str, Any],
    forecast_id: str,
    event_key: str,
    event_order_lineage: dict[str, object],
    prior_monitoring: dict[str, object],
    preflight: dict[str, Any],
) -> pd.DataFrame:
    """Build forecast integrity audit rows."""
    frame = pd.DataFrame(leakage_rows)
    if frame.empty:
        frame = pd.DataFrame([{}])
    frame["protocol_name"] = protocol["protocol_name"]
    frame["protocol_fingerprint"] = protocol["protocol_fingerprint"]
    frame["forecast_id"] = forecast_id
    frame["event_key"] = event_key
    for column, value in event_order_lineage.items():
        frame[column] = value
    for column, value in prior_monitoring.items():
        frame[column] = value
    frame["preflight_run_id"] = preflight.get("preflight_run_id")
    frame["preflight_status"] = preflight.get("status")
    frame["preflight_summary_path"] = preflight.get("preflight_summary_path")
    frame["protocol_fingerprint_valid"] = True
    frame["event_order_registry_valid"] = bool(
        event_order_lineage.get("event_order_registry_valid", False)
    )
    frame["current_event_target_not_accessed"] = True
    frame["current_event_excluded_from_training"] = ~frame.get(
        "current_event_used",
        pd.Series([False] * len(frame)),
    ).astype(bool)
    frame["future_same_season_events_excluded"] = ~frame.get(
        "future_test_season_event_used",
        pd.Series([False] * len(frame)),
    ).astype(bool)
    frame["future_seasons_excluded"] = ~frame.get(
        "future_event_used_anywhere",
        pd.Series([False] * len(frame)),
    ).astype(bool)
    frame["forecast_integrity_status"] = (
        frame[
            [
                "protocol_fingerprint_valid",
                "current_event_target_not_accessed",
                "current_event_excluded_from_training",
                "future_same_season_events_excluded",
                "future_seasons_excluded",
                "event_order_registry_valid",
            ]
        ]
        .all(axis=1)
        .map(lambda value: "valid" if value else "invalid")
    )
    return frame


def monitoring_selection_row(
    *,
    protocol: dict[str, Any],
    forecast_id: str,
    forecast_created: str,
    event_key: str,
    event_order_lineage: dict[str, object],
    prior_monitoring: dict[str, object],
    preflight: dict[str, Any],
    candidate_eligible: bool,
    selection_reason: str,
    snapshot_hash: str,
) -> dict[str, object]:
    """Build the event-level forecast selection record."""
    season, slug = parse_event_key(event_key)
    row = {
        "protocol_name": protocol["protocol_name"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "forecast_id": forecast_id,
        "forecast_created_at_utc": forecast_created,
        "season": season,
        "event_slug": slug,
        "event_order": event_order_lineage.get("event_order", pd.NA),
        "event_order_source": event_order_lineage.get("event_order_source", EVENT_ORDER_SOURCE),
        "event_order_registry_valid": event_order_lineage.get("event_order_registry_valid", False),
        "event_order_registry_path": event_order_lineage.get("event_order_registry_path"),
        "event_order_registry_protocol_name": event_order_lineage.get(
            "event_order_registry_protocol_name"
        ),
        "event_order_registry_monitor_season": event_order_lineage.get(
            "event_order_registry_monitor_season"
        ),
        "event_order_lineage_status": event_order_lineage.get(
            "event_order_lineage_status",
            EVENT_ORDER_INVALID_STATUS,
        ),
        "checkpoint": protocol["checkpoint"],
        "observed_live_policy_role": "observed_live_policy",
        "live_policy_selected": "uniform_default",
        "weighted_candidate_eligible_under_frozen_gates": bool(candidate_eligible),
        "shadow_history_counterfactual_selection": (
            "season_aware_weighted_candidate" if candidate_eligible else "uniform_default"
        ),
        "selection_is_live": True,
        "selection_is_counterfactual": bool(candidate_eligible),
        "candidate_selection_reason": selection_reason,
        "forecast_snapshot_hash": snapshot_hash,
    }
    row.update(prior_monitoring)
    row["preflight_run_id"] = preflight.get("preflight_run_id")
    row["preflight_status"] = preflight.get("status")
    row["preflight_summary_path"] = preflight.get("preflight_summary_path")
    return row


def build_settlement_rows(
    *,
    protocol: dict[str, Any],
    forecasts: pd.DataFrame,
    outcomes: pd.DataFrame,
    coverage: pd.DataFrame | None = None,
    mutation_detected: bool,
) -> pd.DataFrame:
    """Join forecast rows to actual outcomes by exact event/checkpoint/driver keys."""
    join_cols = ["season", "event_slug", "checkpoint", "_settlement_driver_key"]
    forecasts = with_settlement_driver_key(forecasts)
    outcomes = with_settlement_driver_key(outcomes)
    outcome_cols = [*join_cols, "quali_gap_to_pole_sec"]
    if coverage is not None and not coverage.empty:
        coverage = with_settlement_driver_key(coverage)
        merged = forecasts.merge(outcomes.loc[:, outcome_cols], on=join_cols, how="left")
        coverage_cols = [
            *join_cols,
            "target_evaluable",
            "included_in_settlement_metrics",
            "settlement_exclusion_reason",
        ]
        merged = merged.merge(
            coverage.reindex(columns=coverage_cols),
            on=join_cols,
            how="left",
            suffixes=("", "_coverage"),
        )
    else:
        merged = forecasts.merge(outcomes.loc[:, outcome_cols], on=join_cols, how="inner")
        merged["target_evaluable"] = True
        merged["included_in_settlement_metrics"] = True
        merged["settlement_exclusion_reason"] = ""
    if merged.empty:
        return pd.DataFrame(columns=settlement_columns())
    settled_at = utc_now()
    merged["settlement_id"] = merged.apply(
        lambda row: stable_signature(
            {
                "forecast_id": row["forecast_id"],
                "driver": row["driver"],
                "role": row["prediction_role"],
                "settled_at": settled_at,
            }
        ),
        axis=1,
    )
    merged["settled_at_utc"] = settled_at
    merged["target_evaluable"] = (
        merged["target_evaluable"].fillna(False).astype(bool)
        & merged["quali_gap_to_pole_sec"].notna()
    )
    merged["included_in_metrics"] = merged["target_evaluable"].astype(bool)
    merged["settlement_evaluable"] = merged["target_evaluable"].astype(bool)
    merged["actual_gap_sec"] = merged["quali_gap_to_pole_sec"].where(
        merged["settlement_evaluable"],
    )
    merged["absolute_error_sec"] = (
        pd.to_numeric(merged["prediction_gap_sec"], errors="coerce")
        - pd.to_numeric(merged["actual_gap_sec"], errors="coerce")
    ).abs()
    merged.loc[~merged["settlement_evaluable"], "absolute_error_sec"] = pd.NA
    merged["settlement_exclusion_reason"] = merged["settlement_exclusion_reason"].fillna(
        "target_missing_or_non_evaluable"
    )
    merged.loc[merged["settlement_evaluable"], "settlement_exclusion_reason"] = ""
    merged["forecast_row_preserved"] = True
    merged["settlement_valid"] = not mutation_detected
    merged["forecast_preexisted_settlement"] = True
    merged["forecast_fingerprint_valid"] = (
        merged["protocol_fingerprint"].astype(str).eq(str(protocol["protocol_fingerprint"]))
    )
    merged["forecast_mutation_detected"] = bool(mutation_detected)
    merged["eligible_for_future_prior_evidence"] = (
        merged["diagnostic_only"].astype(bool)
        & merged["forecast_fingerprint_valid"].astype(bool)
        & ~merged["forecast_mutation_detected"].astype(bool)
        & merged["settlement_evaluable"].astype(bool)
    )
    merged["settlement_blocking_reason"] = merged["settlement_exclusion_reason"]
    return merged.reindex(columns=settlement_columns())


def with_settlement_driver_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the exact settlement driver key while preserving public identifiers."""
    result = frame.copy()
    if "driver_key" in result:
        key = result["driver_key"].where(result["driver_key"].notna(), result.get("driver"))
    else:
        key = result.get("driver", pd.Series([pd.NA] * len(result), index=result.index))
        result["driver_key"] = key
    result["_settlement_driver_key"] = key.astype(str).str.strip().str.lower()
    return result


def build_event_metrics(settlements: pd.DataFrame) -> pd.DataFrame:
    """Score live and diagnostic rows separately."""
    columns = [
        "protocol_name",
        "season",
        "event_slug",
        "checkpoint",
        "prediction_role",
        "diagnostic_only",
        "forecast_rows",
        "rows",
        "scored_rows",
        "excluded_rows",
        "mae_gap_sec",
    ]
    if settlements.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for keys, group in settlements.groupby(
        ["protocol_name", "season", "event_slug", "checkpoint", "prediction_role"],
        dropna=False,
        sort=False,
    ):
        protocol_name, season, event_slug, checkpoint, role = keys
        included = (
            group["included_in_metrics"].astype(bool)
            if "included_in_metrics" in group
            else pd.Series([True] * len(group), index=group.index)
        )
        scored = group.loc[included]
        rows.append(
            {
                "protocol_name": protocol_name,
                "season": season,
                "event_slug": event_slug,
                "checkpoint": checkpoint,
                "prediction_role": role,
                "diagnostic_only": bool(group["diagnostic_only"].astype(bool).iloc[0]),
                "forecast_rows": int(len(group)),
                "rows": int(len(scored)),
                "scored_rows": int(len(scored)),
                "excluded_rows": int(len(group) - len(scored)),
                "mae_gap_sec": _number_or_none(scored["absolute_error_sec"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_shadow_evidence_ledger(
    settlements: pd.DataFrame,
    *,
    reconciliation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return settled diagnostic rows available to future event gates."""
    columns = [
        "protocol_name",
        "protocol_fingerprint",
        "forecast_id",
        "season",
        "event_slug",
        "artifact_event_order",
        "event_order",
        "registry_event_order",
        "event_order_lineage_status",
        "checkpoint",
        "driver",
        "prediction_role",
        "temporal_weighting_policy",
        "prediction_gap_sec",
        "actual_gap_sec",
        "absolute_error_sec",
        "eligible_for_future_prior_evidence",
    ]
    if settlements.empty:
        return pd.DataFrame(columns=columns)
    shadow = settlements[settlements["diagnostic_only"].astype(bool)].copy()
    shadow["artifact_event_order"] = shadow.get("event_order", pd.NA)
    shadow["registry_event_order"] = pd.NA
    shadow["event_order_lineage_status"] = EVENT_ORDER_MISSING_STATUS
    if reconciliation is not None and not reconciliation.empty:
        keep = reconciliation[
            [
                "forecast_id",
                "event_slug",
                "registry_event_order",
                "event_order_lineage_status",
                "eligible_for_future_prior_evidence_after_reconciliation",
            ]
        ].copy()
        shadow = shadow.merge(
            keep,
            on=["forecast_id", "event_slug"],
            how="left",
            suffixes=("", "_reconciled"),
        )
        shadow["registry_event_order"] = shadow["registry_event_order_reconciled"].where(
            shadow["registry_event_order_reconciled"].notna(),
            shadow["registry_event_order"],
        )
        shadow["event_order_lineage_status"] = shadow[
            "event_order_lineage_status_reconciled"
        ].fillna(shadow["event_order_lineage_status"])
        shadow["eligible_for_future_prior_evidence"] = shadow[
            "eligible_for_future_prior_evidence"
        ].fillna(False).astype(bool) & shadow[
            "eligible_for_future_prior_evidence_after_reconciliation"
        ].fillna(False).astype(bool)
        shadow["event_order"] = shadow["registry_event_order"].where(
            shadow["registry_event_order"].notna(),
            shadow["event_order"],
        )
        shadow = shadow.drop(
            columns=[
                "registry_event_order_reconciled",
                "event_order_lineage_status_reconciled",
                "eligible_for_future_prior_evidence_after_reconciliation",
            ],
            errors="ignore",
        )
    return shadow.reindex(columns=columns)


def refresh_integrity_outputs(metrics_dir: Path, protocol: dict[str, Any]) -> dict[str, object]:
    """Rebuild monitoring integrity summary and event/failure tables."""
    registry = read_csv(metrics_dir / "prospective_monitoring_event_registry.csv")
    forecasts = read_parquet(metrics_dir / "prospective_monitoring_forecasts.parquet")
    settlements = read_parquet(metrics_dir / "prospective_monitoring_settlements.parquet")
    ledger = read_csv(metrics_dir / "prospective_monitoring_shadow_evidence_ledger.csv")
    reconciliation = build_event_order_reconciliation(
        protocol,
        registry,
        forecasts,
        settlements,
        ledger,
        metrics_dir=metrics_dir,
    )
    reconciliation.to_csv(
        metrics_dir / "prospective_monitoring_event_order_reconciliation.csv",
        index=False,
    )
    forecast_audit = read_csv(metrics_dir / "prospective_monitoring_forecast_integrity_audit.csv")
    settlement_audit = read_csv(
        metrics_dir / "prospective_monitoring_settlement_integrity_audit.csv"
    )
    event_order_by_event = build_event_order_integrity_by_event(
        protocol,
        registry,
        forecasts,
        settlements,
        ledger,
        reconciliation,
    )
    event_order_failures = build_event_order_integrity_failures(event_order_by_event)
    event_order_status = event_order_integrity_status(event_order_by_event, event_order_failures)
    event_order_summary = {
        "status": event_order_status,
        "protocol_name": protocol.get("protocol_name"),
        "protocol_fingerprint": protocol.get("protocol_fingerprint"),
        "events_checked": int(len(event_order_by_event)),
        "failure_count": int(len(event_order_failures)),
        "legacy_event_order_exclusion_count": int(
            reconciliation["event_order_lineage_status"]
            .astype(str)
            .eq(EVENT_ORDER_LEGACY_STATUS)
            .sum()
        )
        if not reconciliation.empty
        else 0,
        "prior_evidence_lineage_status": prior_evidence_lineage_status(reconciliation),
        "policy_recommendation": POLICY_RECOMMENDATION,
        "generated_at_utc": utc_now(),
    }
    _write_json(
        metrics_dir / "prospective_monitoring_event_order_integrity_summary.json",
        event_order_summary,
    )
    event_order_by_event.to_csv(
        metrics_dir / "prospective_monitoring_event_order_integrity_by_event.csv",
        index=False,
    )
    event_order_failures.to_csv(
        metrics_dir / "prospective_monitoring_event_order_integrity_failures.csv",
        index=False,
    )
    by_event = build_integrity_by_event(
        protocol,
        registry,
        forecasts,
        settlements,
        forecast_audit,
        reconciliation=reconciliation,
    )
    failures = build_integrity_failures(by_event, settlement_audit)
    status = "invalid" if not failures.empty else "valid"
    if status == "valid" and event_order_status == "valid_with_legacy_artifact_exclusion":
        status = event_order_status
    elif event_order_status == "invalid":
        status = "invalid"
    if (
        status == "valid"
        and not by_event.empty
        and by_event["integrity_status"].astype(str).eq("valid_with_partial_coverage").any()
    ):
        status = "valid_with_partial_coverage"
    summary = {
        "status": status,
        "protocol_name": protocol.get("protocol_name"),
        "protocol_fingerprint": protocol.get("protocol_fingerprint"),
        "events_checked": int(len(by_event)),
        "failure_count": int(len(failures)),
        "event_order_lineage_status": event_order_status,
        "legacy_event_order_exclusion_count": event_order_summary[
            "legacy_event_order_exclusion_count"
        ],
        "prior_evidence_lineage_status": prior_evidence_lineage_status(reconciliation),
        "policy_recommendation": POLICY_RECOMMENDATION,
        "generated_at_utc": utc_now(),
    }
    _write_json(metrics_dir / "prospective_monitoring_integrity_summary.json", summary)
    by_event.to_csv(metrics_dir / "prospective_monitoring_integrity_by_event.csv", index=False)
    failures.to_csv(metrics_dir / "prospective_monitoring_integrity_failures.csv", index=False)
    return summary


def build_integrity_by_event(
    protocol: dict[str, Any],
    registry: pd.DataFrame,
    forecasts: pd.DataFrame,
    settlements: pd.DataFrame,
    forecast_audit: pd.DataFrame,
    *,
    reconciliation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build required monitoring integrity conditions by event."""
    columns = integrity_columns()
    if registry.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, event in registry.iterrows():
        slug = str(event["event_slug"])
        event_forecasts = subset_event(forecasts, protocol.get("protocol_name"), slug)
        event_settlements = subset_event(settlements, protocol.get("protocol_name"), slug)
        event_audit = forecast_audit[
            forecast_audit.get("protocol_name", pd.Series(dtype=str))
            .astype(str)
            .eq(str(protocol.get("protocol_name")))
        ].copy()
        if "event_key" in event_audit:
            event_audit = event_audit[event_audit["event_key"].astype(str).str.endswith(f"/{slug}")]
        protocol_valid = (
            event_forecasts.empty
            or event_forecasts["protocol_fingerprint"]
            .astype(str)
            .eq(str(protocol.get("protocol_fingerprint")))
            .all()
        )
        shadow_excluded = event_forecasts.empty or not event_forecasts[
            event_forecasts["diagnostic_only"].astype(bool)
            & event_forecasts["live_policy_selected"].astype(bool)
        ].any(axis=None)
        live_shadow_separated = event_forecasts.empty or set(
            event_forecasts["prediction_role"].astype(str)
        ) <= set(FORECAST_ROLES)
        non_evaluable = (
            event_settlements["settlement_evaluable"].astype(bool).eq(False)
            if not event_settlements.empty and "settlement_evaluable" in event_settlements
            else pd.Series(dtype=bool)
        )
        target_status = normalized_target_coverage_status(event)
        partial_coverage = target_status == "target_coverage_partial"
        row = {
            "protocol_name": protocol.get("protocol_name"),
            "event_slug": slug,
            "event_order": event.get("event_order"),
            "protocol_fingerprint_valid": protocol_valid,
            "candidate_identity_valid": protocol.get("candidate_identity")
            == canonical_candidate_identity(),
            "default_identity_valid": (
                protocol.get("default_identity") == canonical_default_identity()
            ),
            "live_policy_identity_valid": bool(protocol.get("observed_live_policy_identity")),
            "forecast_precedes_settlement": forecast_precedes_settlement(
                event_forecasts,
                event_settlements,
            ),
            "current_event_target_not_accessed": bool(
                event_audit.empty
                or event_audit.get(
                    "current_event_target_not_accessed",
                    pd.Series([True]),
                )
                .astype(bool)
                .all()
            ),
            "current_event_excluded_from_training": bool(
                event_audit.empty
                or event_audit.get(
                    "current_event_excluded_from_training",
                    pd.Series([True]),
                )
                .astype(bool)
                .all()
            ),
            "future_same_season_events_excluded": bool(
                event_audit.empty
                or event_audit.get(
                    "future_same_season_events_excluded",
                    pd.Series([True]),
                )
                .astype(bool)
                .all()
            ),
            "future_seasons_excluded": bool(
                event_audit.empty
                or event_audit.get("future_seasons_excluded", pd.Series([True])).astype(bool).all()
            ),
            "future_settlement_not_used": no_future_settlement_used(
                event,
                settlements,
                event_forecasts,
                reconciliation=reconciliation,
            ),
            "prior_settlement_only_evidence": True,
            "forecast_snapshot_mutation_detected": bool(
                event_settlements.get(
                    "forecast_mutation_detected",
                    pd.Series([False]),
                )
                .astype(bool)
                .any()
            )
            if not event_settlements.empty
            else False,
            "shadow_rows_excluded_from_live_metrics": shadow_excluded,
            "live_and_counterfactual_selection_separated": live_shadow_separated,
            "partial_target_coverage_documented": (not partial_coverage)
            or bool(event.get("partial_target_coverage", False)),
            "non_evaluable_rows_preserved": event_settlements.empty
            or not bool(non_evaluable.any())
            or len(event_settlements) == len(event_forecasts),
            "non_evaluable_rows_excluded_from_metrics": event_settlements.empty
            or not bool(non_evaluable.any())
            or not event_settlements.loc[non_evaluable, "included_in_metrics"].astype(bool).any(),
            "non_evaluable_rows_excluded_from_prior_evidence": event_settlements.empty
            or not bool(non_evaluable.any())
            or not event_settlements.loc[
                non_evaluable,
                "eligible_for_future_prior_evidence",
            ]
            .astype(bool)
            .any(),
            "valid_target_rows_exactly_aligned": str(target_status) != "target_coverage_invalid",
            "extra_targets_absent_or_explained": True,
            "coverage_rate_recorded": pd.notna(event.get("target_coverage_rate", pd.NA))
            or target_status == "target_not_available",
            "forecast_artifact_unchanged_after_target_creation": True,
        }
        valid = all(
            bool(row[col]) for col in columns[3:-1] if col != "forecast_snapshot_mutation_detected"
        ) and not bool(row["forecast_snapshot_mutation_detected"])
        if valid and partial_coverage and not event_settlements.empty:
            row["integrity_status"] = "valid_with_partial_coverage"
        else:
            row["integrity_status"] = "valid" if valid else "invalid"
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def build_integrity_failures(
    by_event: pd.DataFrame,
    settlement_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Flatten failed integrity conditions."""
    columns = ["protocol_name", "event_slug", "condition", "status", "details"]
    rows: list[dict[str, object]] = []
    if not by_event.empty:
        for _, row in by_event.iterrows():
            for column in integrity_columns()[3:-1]:
                if str(row.get("integrity_status")) == "valid_with_partial_coverage":
                    continue
                if column == "forecast_snapshot_mutation_detected":
                    if bool(row.get(column, False)):
                        rows.append(
                            {
                                "protocol_name": row.get("protocol_name"),
                                "event_slug": row.get("event_slug"),
                                "condition": column,
                                "status": "failed",
                                "details": "",
                            }
                        )
                    continue
                if not bool(row.get(column, True)):
                    rows.append(
                        {
                            "protocol_name": row.get("protocol_name"),
                            "event_slug": row.get("event_slug"),
                            "condition": column,
                            "status": "failed",
                            "details": "",
                        }
                    )
    if not settlement_audit.empty and "settlement_valid" in settlement_audit:
        bad = settlement_audit[~settlement_audit["settlement_valid"].astype(bool)]
        for _, row in bad.iterrows():
            rows.append(
                {
                    "protocol_name": row.get("protocol_name"),
                    "event_slug": row.get("event_slug"),
                    "condition": "settlement_integrity",
                    "status": "failed",
                    "details": row.get("settlement_blocking_reason", ""),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_event_order_integrity_by_event(
    protocol: dict[str, Any],
    registry: pd.DataFrame,
    forecasts: pd.DataFrame,
    settlements: pd.DataFrame,
    ledger: pd.DataFrame,
    reconciliation: pd.DataFrame,
) -> pd.DataFrame:
    columns = event_order_integrity_columns()
    if registry.empty or not protocol:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    protocol_name = str(protocol.get("protocol_name"))
    monitor_season = int(protocol.get("monitor_season", 0) or 0)
    registry_path = Path("reports/metrics/prospective_monitoring_event_registry.csv")
    for _, event in registry.iterrows():
        slug = str(event["event_slug"])
        lineage = resolve_registry_event_order(
            registry,
            protocol_name=protocol_name,
            monitor_season=monitor_season,
            event_slug=slug,
            registry_path=registry_path,
            strict=False,
        )
        event_recon = (
            reconciliation[reconciliation["event_slug"].astype(str).eq(slug)].copy()
            if not reconciliation.empty
            else pd.DataFrame(columns=event_order_reconciliation_columns())
        )
        event_forecasts = subset_event(forecasts, protocol_name, slug)
        event_settlements = subset_event(settlements, protocol_name, slug)
        event_ledger = subset_event(ledger, protocol_name, slug)
        registry_available = bool(lineage.get("event_order_registry_valid", False))
        forecast_match = (
            event_forecasts.empty
            or event_recon[
                event_recon["forecast_id"]
                .astype(str)
                .isin(event_forecasts["forecast_id"].dropna().astype(str))
            ]["event_order_match"]
            .astype(bool)
            .all()
        )
        settlement_match = (
            event_settlements.empty
            or event_recon[
                event_recon["forecast_id"]
                .astype(str)
                .isin(event_settlements["forecast_id"].dropna().astype(str))
            ]["event_order_match"]
            .astype(bool)
            .all()
        )
        ledger_match = (
            event_ledger.empty
            or event_recon[
                event_recon["forecast_id"]
                .astype(str)
                .isin(event_ledger["forecast_id"].dropna().astype(str))
            ]["event_order_match"]
            .astype(bool)
            .all()
        )
        legacy = event_recon["event_order_lineage_status"].astype(str).eq(EVENT_ORDER_LEGACY_STATUS)
        legacy_excluded = event_recon.empty or not bool(
            event_recon.loc[
                legacy,
                "eligible_for_future_prior_evidence_after_reconciliation",
            ]
            .fillna(False)
            .astype(bool)
            .any()
        )
        future_violation, same_violation = event_order_prior_violations(
            event,
            forecasts,
            settlements,
            reconciliation,
            protocol_name=protocol_name,
        )
        prior_registry_only = legacy_excluded and not future_violation and not same_violation
        status = "valid"
        if not registry_available or not legacy_excluded or future_violation or same_violation:
            status = "invalid"
        elif (
            not bool(forecast_match)
            or not bool(settlement_match)
            or not bool(ledger_match)
            or bool(legacy.any())
        ):
            status = "valid_with_legacy_artifact_exclusion"
        rows.append(
            {
                "protocol_name": protocol_name,
                "monitor_season": monitor_season,
                "event_slug": slug,
                "registry_event_order": lineage.get("event_order", pd.NA),
                "registry_event_order_available": registry_available,
                "forecast_event_order_matches_registry": bool(forecast_match),
                "settlement_event_order_matches_registry": bool(settlement_match),
                "shadow_ledger_event_order_matches_registry": bool(ledger_match),
                "prior_evidence_uses_registry_order_only": bool(prior_registry_only),
                "legacy_noncanonical_rows_excluded_from_prior_evidence": bool(legacy_excluded),
                "future_event_order_violation": bool(future_violation),
                "same_event_order_violation": bool(same_violation),
                "integrity_status": status,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def event_order_prior_violations(
    event: pd.Series,
    forecasts: pd.DataFrame,
    settlements: pd.DataFrame,
    reconciliation: pd.DataFrame,
    *,
    protocol_name: str,
) -> tuple[bool, bool]:
    event_forecasts = subset_event(forecasts, protocol_name, str(event["event_slug"]))
    if event_forecasts.empty or settlements.empty or reconciliation.empty:
        return False, False
    current_order = pd.to_numeric(pd.Series([event.get("event_order")]), errors="coerce").iloc[0]
    forecast_time = pd.to_datetime(
        event_forecasts["forecast_created_at_utc"],
        errors="coerce",
    ).min()
    eligible = reconciliation[
        reconciliation["eligible_for_future_prior_evidence_after_reconciliation"].astype(bool)
    ].copy()
    if eligible.empty or pd.isna(current_order) or pd.isna(forecast_time):
        return False, False
    eligible_settlements = settlements.merge(
        eligible[["forecast_id", "event_slug", "registry_event_order"]],
        on=["forecast_id", "event_slug"],
        how="inner",
    )
    if eligible_settlements.empty or "settled_at_utc" not in eligible_settlements:
        return False, False
    settled_before = pd.to_datetime(
        eligible_settlements["settled_at_utc"],
        errors="coerce",
    ).le(forecast_time)
    orders = pd.to_numeric(eligible_settlements["registry_event_order"], errors="coerce")
    future = bool((settled_before & orders.gt(current_order)).any())
    same = bool((settled_before & orders.eq(current_order)).any())
    return future, same


def build_event_order_integrity_failures(by_event: pd.DataFrame) -> pd.DataFrame:
    columns = ["protocol_name", "event_slug", "condition", "status", "details"]
    if by_event.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    checked = event_order_integrity_columns()[4:-1]
    for _, row in by_event.iterrows():
        if str(row.get("integrity_status")) == "valid_with_legacy_artifact_exclusion":
            continue
        for column in checked:
            value = bool(row.get(column, True))
            if column in {"future_event_order_violation", "same_event_order_violation"}:
                value = not bool(row.get(column, False))
            if not value:
                rows.append(
                    {
                        "protocol_name": row.get("protocol_name"),
                        "event_slug": row.get("event_slug"),
                        "condition": column,
                        "status": "failed",
                        "details": row.get("integrity_status", ""),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def event_order_integrity_status(
    by_event: pd.DataFrame,
    failures: pd.DataFrame,
) -> str:
    if not failures.empty:
        return "invalid"
    if (
        not by_event.empty
        and by_event["integrity_status"]
        .astype(str)
        .eq("valid_with_legacy_artifact_exclusion")
        .any()
    ):
        return "valid_with_legacy_artifact_exclusion"
    return "valid" if not by_event.empty else "not_evaluated"


def prior_evidence_lineage_status(reconciliation: pd.DataFrame) -> str:
    if reconciliation.empty:
        return "no_settled_monitoring_evidence"
    if reconciliation["eligible_for_future_prior_evidence_after_reconciliation"].astype(bool).any():
        return "valid_registry_prior_evidence_available"
    if reconciliation["event_order_lineage_status"].astype(str).eq(EVENT_ORDER_LEGACY_STATUS).any():
        return "legacy_noncanonical_rows_excluded"
    return "no_valid_prior_evidence_available"


def event_order_lineage_summary_status(reconciliation: pd.DataFrame | None) -> str:
    if reconciliation is None or reconciliation.empty:
        return "not_evaluated"
    statuses = set(reconciliation["event_order_lineage_status"].dropna().astype(str))
    if EVENT_ORDER_LEGACY_STATUS in statuses:
        return "valid_with_legacy_artifact_exclusion"
    if statuses <= {EVENT_ORDER_VALID_STATUS}:
        return EVENT_ORDER_VALID_STATUS
    return "invalid"


def event_order_integrity_columns() -> list[str]:
    return [
        "protocol_name",
        "monitor_season",
        "event_slug",
        "registry_event_order",
        "registry_event_order_available",
        "forecast_event_order_matches_registry",
        "settlement_event_order_matches_registry",
        "shadow_ledger_event_order_matches_registry",
        "prior_evidence_uses_registry_order_only",
        "legacy_noncanonical_rows_excluded_from_prior_evidence",
        "future_event_order_violation",
        "same_event_order_violation",
        "integrity_status",
    ]


def build_status_by_event(
    protocol: dict[str, Any],
    registry: pd.DataFrame,
    forecasts: pd.DataFrame,
    settlements: pd.DataFrame,
    *,
    reconciliation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize event monitoring status."""
    columns = [
        "protocol_name",
        "monitor_season",
        "event_order",
        "event",
        "event_slug",
        "forecast_created",
        "settled",
        "live_rows",
        "shadow_rows",
        "fresh_evidence_available",
        "target_coverage_status",
        "target_coverage_rate",
        "evaluable_driver_count",
        "non_evaluable_driver_count",
        "settlement_metric_status",
        "monitoring_event_order_lineage_status",
        "monitoring_legacy_event_order_exclusion_count",
    ]
    if registry.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, event in registry.iterrows():
        slug = str(event["event_slug"])
        event_forecasts = subset_event(forecasts, protocol.get("protocol_name"), slug)
        event_settlements = subset_event(settlements, protocol.get("protocol_name"), slug)
        event_recon = (
            reconciliation[reconciliation["event_slug"].astype(str).eq(slug)]
            if reconciliation is not None and not reconciliation.empty
            else pd.DataFrame()
        )
        diagnostic = event_forecasts.get("diagnostic_only", pd.Series(dtype=bool)).astype(bool)
        live_rows = int((~diagnostic).sum()) if not event_forecasts.empty else 0
        shadow_rows = int(diagnostic.sum()) if not event_forecasts.empty else 0
        rows.append(
            {
                "protocol_name": protocol.get("protocol_name"),
                "monitor_season": protocol.get("monitor_season"),
                "event_order": event.get("event_order"),
                "event": event.get("event"),
                "event_slug": slug,
                "forecast_created": not event_forecasts.empty,
                "settled": not event_settlements.empty,
                "live_rows": live_rows,
                "shadow_rows": shadow_rows,
                "fresh_evidence_available": not event_settlements.empty,
                "target_coverage_status": event.get(
                    "target_coverage_status", "target_not_available"
                ),
                "target_coverage_rate": event.get("target_coverage_rate", pd.NA),
                "evaluable_driver_count": event.get("evaluable_driver_count", 0),
                "non_evaluable_driver_count": event.get("non_evaluable_driver_count", 0),
                "settlement_metric_status": event.get("settlement_metric_status", "not_scorable"),
                "monitoring_event_order_lineage_status": event_lineage_status(event_recon),
                "monitoring_legacy_event_order_exclusion_count": int(
                    event_recon["event_order_lineage_status"]
                    .astype(str)
                    .eq(EVENT_ORDER_LEGACY_STATUS)
                    .sum()
                )
                if not event_recon.empty
                else 0,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_live_policy_summary(settlements: pd.DataFrame) -> pd.DataFrame:
    """Summarize settled observed live-policy performance only."""
    columns = ["protocol_name", "rows", "settled_events", "mae_gap_sec", "diagnostic_only"]
    if settlements.empty:
        return pd.DataFrame(columns=columns)
    live = settlements[~settlements["diagnostic_only"].astype(bool)].copy()
    if live.empty:
        return pd.DataFrame(columns=columns)
    included = (
        live["included_in_metrics"].astype(bool)
        if "included_in_metrics" in live
        else pd.Series([True] * len(live), index=live.index)
    )
    scored = live.loc[included]
    return pd.DataFrame(
        [
            {
                "protocol_name": live["protocol_name"].iloc[0],
                "rows": int(len(scored)),
                "settled_events": int(scored["event_slug"].nunique()),
                "mae_gap_sec": _number_or_none(scored["absolute_error_sec"].mean()),
                "diagnostic_only": False,
            }
        ],
        columns=columns,
    )


def build_shadow_candidate_summary(settlements: pd.DataFrame) -> pd.DataFrame:
    """Summarize settled diagnostic shadow performance by role."""
    columns = ["protocol_name", "prediction_role", "rows", "settled_events", "mae_gap_sec"]
    if settlements.empty:
        return pd.DataFrame(columns=columns)
    shadow = settlements[settlements["diagnostic_only"].astype(bool)].copy()
    if shadow.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (protocol_name, role), group in shadow.groupby(
        ["protocol_name", "prediction_role"],
        sort=False,
    ):
        included = (
            group["included_in_metrics"].astype(bool)
            if "included_in_metrics" in group
            else pd.Series([True] * len(group), index=group.index)
        )
        scored = group.loc[included]
        rows.append(
            {
                "protocol_name": protocol_name,
                "prediction_role": role,
                "rows": int(len(scored)),
                "settled_events": int(scored["event_slug"].nunique()),
                "mae_gap_sec": _number_or_none(scored["absolute_error_sec"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def reconciliation_for_event(
    reconciliation: pd.DataFrame | None,
    protocol_name: str,
    event_slug: str,
) -> pd.DataFrame:
    if reconciliation is None or reconciliation.empty:
        return pd.DataFrame(columns=event_order_reconciliation_columns())
    return reconciliation[
        reconciliation["protocol_name"].astype(str).eq(str(protocol_name))
        & reconciliation["event_slug"].astype(str).eq(str(event_slug))
    ].copy()


def event_lineage_status(event_recon: pd.DataFrame) -> str:
    if event_recon.empty:
        return "not_evaluated"
    statuses = set(event_recon["event_order_lineage_status"].dropna().astype(str))
    if EVENT_ORDER_LEGACY_STATUS in statuses:
        return EVENT_ORDER_LEGACY_STATUS
    if EVENT_ORDER_DUPLICATE_STATUS in statuses:
        return EVENT_ORDER_DUPLICATE_STATUS
    if EVENT_ORDER_MISSING_STATUS in statuses:
        return EVENT_ORDER_MISSING_STATUS
    if EVENT_ORDER_INVALID_STATUS in statuses:
        return EVENT_ORDER_INVALID_STATUS
    if EVENT_ORDER_VALID_STATUS in statuses:
        return EVENT_ORDER_VALID_STATUS
    return "unknown"


def normalized_target_coverage_status(event: pd.Series) -> str:
    value = event.get("target_coverage_status", "target_not_available")
    if value is None or pd.isna(value) or not str(value).strip():
        return "target_not_available"
    return str(value)


def build_gate_timeline(
    forecasts: pd.DataFrame,
    *,
    reconciliation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize frozen gate status over forecasted events."""
    columns = [
        "protocol_name",
        "event_order",
        "event_slug",
        "candidate_eligible_under_frozen_gates",
        "counterfactual_shadow_selected",
        "event_order_lineage_status",
    ]
    if forecasts.empty:
        return pd.DataFrame(columns=columns)
    weighted = forecasts[
        forecasts["prediction_role"].astype(str).eq("season_aware_weighted_candidate_shadow")
    ]
    rows = []
    for keys, group in weighted.groupby(["protocol_name", "event_order", "event_slug"], sort=True):
        protocol_name, event_order, event_slug = keys
        event_recon = reconciliation_for_event(reconciliation, str(protocol_name), str(event_slug))
        registry_order = (
            event_recon["registry_event_order"].dropna().iloc[0]
            if not event_recon.empty and not event_recon["registry_event_order"].dropna().empty
            else event_order
        )
        eligible = bool(group["candidate_eligible_under_frozen_gates"].astype(bool).any())
        rows.append(
            {
                "protocol_name": protocol_name,
                "event_order": registry_order,
                "event_slug": event_slug,
                "candidate_eligible_under_frozen_gates": eligible,
                "counterfactual_shadow_selected": eligible,
                "event_order_lineage_status": event_lineage_status(event_recon),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_evidence_growth(
    settlements: pd.DataFrame,
    *,
    reconciliation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Track settled diagnostic evidence accumulation."""
    columns = [
        "protocol_name",
        "event_order",
        "event_slug",
        "settled_shadow_rows_cumulative",
        "settled_shadow_events_cumulative",
    ]
    if settlements.empty:
        return pd.DataFrame(columns=columns)
    shadow = settlements[settlements["diagnostic_only"].astype(bool)].copy()
    if reconciliation is not None and not reconciliation.empty:
        eligible = reconciliation[
            reconciliation["eligible_for_future_prior_evidence_after_reconciliation"].astype(bool)
        ][["forecast_id", "event_slug", "registry_event_order"]]
        shadow = shadow.merge(eligible, on=["forecast_id", "event_slug"], how="inner")
        if not shadow.empty:
            shadow["event_order"] = shadow["registry_event_order"]
    if shadow.empty:
        return pd.DataFrame(columns=columns)
    events = shadow[["protocol_name", "event_order", "event_slug"]].drop_duplicates()
    events = events.sort_values(["protocol_name", "event_order"], kind="stable")
    rows = []
    for protocol_name, group in events.groupby("protocol_name", sort=False):
        cumulative_rows = 0
        cumulative_events = 0
        for _, event in group.iterrows():
            event_rows = shadow[
                shadow["protocol_name"].astype(str).eq(str(protocol_name))
                & shadow["event_slug"].astype(str).eq(str(event["event_slug"]))
            ]
            cumulative_rows += int(len(event_rows))
            cumulative_events += 1
            rows.append(
                {
                    "protocol_name": protocol_name,
                    "event_order": event["event_order"],
                    "event_slug": event["event_slug"],
                    "settled_shadow_rows_cumulative": cumulative_rows,
                    "settled_shadow_events_cumulative": cumulative_events,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_monitoring_summary_payload(
    *,
    protocol: dict[str, Any],
    registry: pd.DataFrame,
    forecasts: pd.DataFrame,
    settlements: pd.DataFrame,
    integrity: dict[str, object],
    live_summary: pd.DataFrame,
    shadow_summary: pd.DataFrame,
    reconciliation: pd.DataFrame | None = None,
    preflight_summary: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Build the top-level monitoring summary JSON."""
    available = bool(protocol)
    settled_events = int(settlements["event_slug"].nunique()) if not settlements.empty else 0
    forecasted_events = int(forecasts["event_slug"].nunique()) if not forecasts.empty else 0
    fresh = "available" if settled_events else "not_collected"
    status = "active" if available else "missing_protocol"
    if available and registry.empty:
        status = "not_ready"
    coverage = monitoring_coverage_summary(registry, settlements)
    lineage_status = (
        event_order_lineage_summary_status(reconciliation)
        if reconciliation is not None
        else "not_evaluated"
    )
    legacy_count = (
        int(
            reconciliation["event_order_lineage_status"]
            .astype(str)
            .eq(EVENT_ORDER_LEGACY_STATUS)
            .sum()
        )
        if reconciliation is not None and not reconciliation.empty
        else 0
    )
    prior_lineage = (
        prior_evidence_lineage_status(reconciliation)
        if reconciliation is not None
        else "not_evaluated"
    )
    preflight = preflight_summary or {}
    return {
        "status": status,
        "prospective_monitoring_available": available,
        "protocol_name": protocol.get("protocol_name"),
        "protocol_fingerprint": protocol.get("protocol_fingerprint"),
        "monitor_season": protocol.get("monitor_season"),
        "train_seasons": protocol.get("train_seasons", []),
        "registered_events": int(len(registry)),
        "forecasted_events": forecasted_events,
        "settled_events": settled_events,
        "live_policy_summary": records_for_json(live_summary),
        "shadow_candidate_summary": records_for_json(shadow_summary),
        "integrity_status": integrity.get("status", "missing"),
        "fresh_evidence_status": fresh,
        "monitoring_event_order_lineage_status": lineage_status,
        "monitoring_legacy_event_order_exclusion_count": legacy_count,
        "monitoring_prior_evidence_lineage_status": prior_lineage,
        "monitoring_next_forecast_prior_evidence_status": prior_lineage,
        "prospective_monitoring_preflight_available": bool(preflight),
        "prospective_monitoring_preflight_status": preflight.get("status", "missing"),
        "prospective_monitoring_preflight_blocking_check_count": int(
            preflight.get("blocking_check_count", 0) or 0
        ),
        "prospective_monitoring_next_event_ready_to_forecast": bool(
            preflight.get("forecast_allowed", False)
        ),
        "prospective_monitoring_preflight_runbook_path": preflight.get(
            "prospective_monitoring_preflight_runbook_path",
        ),
        **coverage,
        "policy_recommendation": POLICY_RECOMMENDATION,
        "known_limitations": [
            "Monitoring commands use local artifacts and local Parquet inputs only.",
            "No monitored evidence changes defaults, gates, thresholds, or candidate identities.",
            "Forecasts and settlements remain separate immutable phases.",
        ],
        "generated_outputs": {
            "metrics": [
                "reports/metrics/prospective_monitoring_summary.json",
                "reports/metrics/prospective_monitoring_status_by_event.csv",
                "reports/metrics/prospective_monitoring_live_policy_summary.csv",
                "reports/metrics/prospective_monitoring_shadow_candidate_summary.csv",
                "reports/metrics/prospective_monitoring_gate_timeline.csv",
                "reports/metrics/prospective_monitoring_evidence_growth.csv",
                "reports/metrics/prospective_monitoring_event_order_reconciliation.csv",
                "reports/metrics/prospective_monitoring_event_order_integrity_summary.json",
                "reports/metrics/prospective_monitoring_event_order_integrity_by_event.csv",
                "reports/metrics/prospective_monitoring_event_order_integrity_failures.csv",
                "reports/metrics/prospective_monitoring_preflight_summary.json",
                "reports/metrics/prospective_monitoring_preflight_checks.csv",
                "reports/metrics/prospective_monitoring_preflight_failures.csv",
                "reports/metrics/prospective_monitoring_preflight_runbook.md",
            ],
            "figures": [],
        },
        "generation_issues": [],
        "generated_at_utc": utc_now(),
    }


def monitoring_coverage_summary(
    registry: pd.DataFrame,
    settlements: pd.DataFrame,
) -> dict[str, object]:
    if registry.empty:
        return {
            "monitoring_target_coverage_status": "target_not_available",
            "monitoring_target_coverage_rate": None,
            "monitoring_evaluable_driver_count": 0,
            "monitoring_non_evaluable_driver_count": 0,
            "monitoring_partial_coverage_event_count": 0,
            "monitoring_settlement_metric_status": "not_scorable",
            "monitoring_scored_row_count": 0,
            "monitoring_excluded_row_count": 0,
        }
    feature_count = (
        pd.to_numeric(registry.get("feature_driver_count", pd.Series(dtype=float)), errors="coerce")
        .fillna(0)
        .sum()
    )
    evaluable_count = (
        pd.to_numeric(
            registry.get("evaluable_driver_count", pd.Series(dtype=float)),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )
    statuses = set(
        registry.get("target_coverage_status", pd.Series(dtype=str)).dropna().astype(str)
    )
    if "target_coverage_invalid" in statuses:
        status = "target_coverage_invalid"
    elif "target_coverage_partial" in statuses:
        status = "target_coverage_partial"
    elif "target_coverage_complete" in statuses:
        status = "target_coverage_complete"
    elif "target_coverage_empty" in statuses:
        status = "target_coverage_empty"
    else:
        status = "target_not_available"
    scored = (
        int(settlements.get("included_in_metrics", pd.Series(dtype=bool)).astype(bool).sum())
        if not settlements.empty
        else 0
    )
    excluded = (
        int((~settlements.get("included_in_metrics", pd.Series(dtype=bool)).astype(bool)).sum())
        if not settlements.empty and "included_in_metrics" in settlements
        else 0
    )
    return {
        "monitoring_target_coverage_status": status,
        "monitoring_target_coverage_rate": float(evaluable_count / feature_count)
        if feature_count
        else None,
        "monitoring_evaluable_driver_count": int(evaluable_count),
        "monitoring_non_evaluable_driver_count": int(
            pd.to_numeric(
                registry.get("non_evaluable_driver_count", pd.Series(dtype=float)),
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "monitoring_partial_coverage_event_count": int(
            registry.get("partial_target_coverage", pd.Series(dtype=bool)).astype(bool).sum()
        ),
        "monitoring_settlement_metric_status": "scorable" if scored else "not_scorable",
        "monitoring_scored_row_count": scored,
        "monitoring_excluded_row_count": excluded,
    }


def generate_monitoring_figures(
    *,
    figures_dir: Path,
    status_by_event: pd.DataFrame,
    live_summary: pd.DataFrame,
    shadow_summary: pd.DataFrame,
    gate_timeline: pd.DataFrame,
    evidence_growth: pd.DataFrame,
    integrity_by_event: pd.DataFrame,
) -> tuple[list[Path], list[str]]:
    """Create static Matplotlib monitoring figures."""
    try:
        ensure_directory(figures_dir / ".matplotlib")
        ensure_directory(figures_dir / ".cache")
        os.environ.setdefault("MPLCONFIGDIR", str(figures_dir / ".matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(figures_dir / ".cache"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return [], [f"matplotlib_unavailable: {exc}"]
    specs = (
        (
            "prospective_monitoring_event_status.png",
            lambda path: _plot_event_status(plt, status_by_event, path),
        ),
        (
            "prospective_monitoring_live_vs_shadow_mae.png",
            lambda path: _plot_live_shadow(plt, live_summary, shadow_summary, path),
        ),
        (
            "prospective_monitoring_gate_timeline.png",
            lambda path: _plot_gate_timeline(plt, gate_timeline, path),
        ),
        (
            "prospective_monitoring_evidence_growth.png",
            lambda path: _plot_evidence_growth(plt, evidence_growth, path),
        ),
        (
            "prospective_monitoring_integrity_status.png",
            lambda path: _plot_integrity(plt, integrity_by_event, path),
        ),
    )
    paths: list[Path] = []
    issues: list[str] = []
    for filename, writer in specs:
        path = figures_dir / filename
        try:
            if writer(path):
                paths.append(path)
        except Exception as exc:  # non-fatal optional report output
            issues.append(f"{filename}: {exc}")
    return paths, issues


def _plot_event_status(plt: Any, frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty:
        return _plot_placeholder(plt, path, "Monitoring event status unavailable")
    counts = frame[["forecast_created", "settled", "fresh_evidence_available"]].sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Prospective monitoring event status")
    ax.set_ylabel("Events")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _plot_live_shadow(plt: Any, live: pd.DataFrame, shadow: pd.DataFrame, path: Path) -> bool:
    rows = []
    if not live.empty:
        rows.append(("observed_live_policy", live["mae_gap_sec"].iloc[0]))
    for _, row in shadow.iterrows():
        rows.append((str(row["prediction_role"]), row["mae_gap_sec"]))
    if not rows:
        return _plot_placeholder(plt, path, "No settled live or shadow MAE yet")
    labels, values = zip(*rows, strict=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set_title("Live policy and diagnostic shadow MAE")
    ax.set_ylabel("MAE gap (sec)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _plot_gate_timeline(plt: Any, frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty:
        return _plot_placeholder(plt, path, "No frozen-gate timeline yet")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(
        frame["event_order"],
        frame["candidate_eligible_under_frozen_gates"].astype(int),
        where="post",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Frozen-gate diagnostic eligibility over time")
    ax.set_xlabel("Event order")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _plot_evidence_growth(plt: Any, frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty:
        return _plot_placeholder(plt, path, "No settled shadow evidence yet")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frame["event_order"], frame["settled_shadow_rows_cumulative"], marker="o")
    ax.set_title("Settled diagnostic shadow evidence growth")
    ax.set_xlabel("Event order")
    ax.set_ylabel("Cumulative rows")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _plot_integrity(plt: Any, frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty:
        return _plot_placeholder(plt, path, "Monitoring integrity unavailable")
    counts = frame["integrity_status"].astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values)
    ax.set_title("Monitoring integrity status")
    ax.set_ylabel("Events")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _plot_placeholder(plt: Any, path: Path, title: str) -> bool:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, title, ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def protocol_fingerprint(payload: dict[str, object]) -> str:
    """Fingerprint frozen protocol fields, excluding creation timestamp."""
    frozen = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at_utc", "protocol_fingerprint"}
    }
    return stable_signature(frozen)


def protocol_mismatch_rows(
    existing: dict[str, Any],
    proposed: dict[str, Any],
) -> list[dict[str, object]]:
    """Return protocol mismatch rows for validation artifacts."""
    rows = []
    keys = sorted(set(existing) | set(proposed))
    for key in keys:
        if key == "created_at_utc":
            continue
        if existing.get(key) != proposed.get(key):
            rows.append(
                {
                    "protocol_name": proposed.get("protocol_name"),
                    "field": key,
                    "expected_value": json.dumps(existing.get(key), sort_keys=True, default=str),
                    "observed_value": json.dumps(proposed.get(key), sort_keys=True, default=str),
                    "validation_status": "mismatch",
                }
            )
    return rows


def write_protocol_validation_artifacts(
    metrics_dir: Path,
    rows: list[dict[str, object]],
) -> tuple[Path, ...]:
    path = metrics_dir / "prospective_monitoring_protocol_validation.csv"
    pd.DataFrame(
        rows,
        columns=(
            "protocol_name",
            "field",
            "expected_value",
            "observed_value",
            "validation_status",
        ),
    ).to_csv(path, index=False)
    return (path,)


def read_monitoring_dataset(path: Path) -> tuple[pd.DataFrame, str]:
    """Read local monitoring dataset if available."""
    if not path.is_file():
        return pd.DataFrame(), "dataset_missing"
    try:
        return pd.read_parquet(path), "available"
    except (OSError, ValueError) as exc:
        return pd.DataFrame(), f"dataset_read_failed: {exc}"


def resolve_protocol_dataset_path(config: DataConfig, protocol: dict[str, Any]) -> Path:
    """Resolve a protocol dataset path relative to the project root."""
    path = Path(str(protocol["dataset_path"]))
    return path if path.is_absolute() else config.project_root / path


def load_protocol(metrics_dir: Path, protocol_name: str) -> dict[str, Any]:
    path = metrics_dir / PROTOCOL_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Monitoring protocol not found: {path}")
    protocol = _read_json(path)
    if str(protocol.get("protocol_name")) != protocol_name:
        raise ValueError(f"Monitoring protocol mismatch: expected {protocol_name}")
    if protocol.get("protocol_fingerprint") != protocol_fingerprint(protocol):
        raise ValueError(f"Monitoring protocol fingerprint is invalid: {protocol_name}")
    return protocol


def load_or_build_registry(
    metrics_dir: Path,
    protocol: dict[str, Any],
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    path = metrics_dir / "prospective_monitoring_event_registry.csv"
    if path.is_file():
        return ensure_registry_columns(pd.read_csv(path))
    registry = build_event_registry(protocol, dataset)
    registry.to_csv(path, index=False)
    return ensure_registry_columns(registry)


def resolve_registry_event_order(
    registry: pd.DataFrame,
    *,
    protocol_name: str,
    monitor_season: int,
    event_slug: str,
    registry_path: Path,
    strict: bool = False,
) -> dict[str, object]:
    """Resolve monitored-season chronology from the frozen registry only."""
    base = {
        "event_order": pd.NA,
        "event_order_source": EVENT_ORDER_SOURCE,
        "event_order_registry_valid": False,
        "event_order_registry_path": _relative_report_path(registry_path),
        "event_order_registry_protocol_name": protocol_name,
        "event_order_registry_monitor_season": int(monitor_season),
        "event_order_registry_row_index": pd.NA,
        "event_order_lineage_status": EVENT_ORDER_MISSING_STATUS,
    }
    if registry.empty:
        return _event_order_resolution(base, strict)
    required = {"protocol_name", "monitor_season", "event_slug", "event_order"}
    if not required <= set(registry.columns):
        base["event_order_lineage_status"] = EVENT_ORDER_MISSING_STATUS
        return _event_order_resolution(base, strict)
    frame = registry.copy()
    mask = (
        frame["protocol_name"].astype(str).eq(str(protocol_name))
        & pd.to_numeric(frame["monitor_season"], errors="coerce").eq(int(monitor_season))
        & frame["event_slug"].astype(str).eq(str(event_slug))
    )
    matches = frame.loc[mask]
    if matches.empty:
        base["event_order_lineage_status"] = EVENT_ORDER_MISSING_STATUS
        return _event_order_resolution(base, strict)
    if len(matches) != 1:
        base["event_order_lineage_status"] = EVENT_ORDER_DUPLICATE_STATUS
        return _event_order_resolution(base, strict)
    value = pd.to_numeric(matches["event_order"], errors="coerce").iloc[0]
    if pd.isna(value) or int(value) <= 0 or float(value) != float(int(value)):
        base["event_order_lineage_status"] = EVENT_ORDER_INVALID_STATUS
        return _event_order_resolution(base, strict)
    base.update(
        {
            "event_order": int(value),
            "event_order_registry_valid": True,
            "event_order_registry_row_index": int(matches.index[0]),
            "event_order_lineage_status": EVENT_ORDER_VALID_STATUS,
        }
    )
    return base


def _event_order_resolution(payload: dict[str, object], strict: bool) -> dict[str, object]:
    if strict and payload["event_order_lineage_status"] != EVENT_ORDER_VALID_STATUS:
        raise ValueError(
            "Monitoring registry event order is unavailable or invalid: "
            f"{payload['event_order_lineage_status']}"
        )
    return payload


def monitoring_event_order_keys(
    dataset: pd.DataFrame,
    registry: pd.DataFrame,
    protocol: dict[str, Any],
) -> list[str]:
    """Return train chronology plus monitored-season registry chronology."""
    monitor_season = int(protocol["monitor_season"])
    historical = [key for key in ordered_event_keys(dataset) if event_season(key) != monitor_season]
    monitor = registry.copy()
    required = {"protocol_name", "monitor_season", "event_order", "event_slug"}
    if not monitor.empty and required <= set(monitor.columns):
        monitor = monitor[
            monitor["protocol_name"].astype(str).eq(str(protocol.get("protocol_name")))
            & pd.to_numeric(monitor["monitor_season"], errors="coerce").eq(monitor_season)
        ].copy()
        monitor["event_order_numeric"] = pd.to_numeric(monitor["event_order"], errors="coerce")
        monitor = monitor.dropna(subset=["event_order_numeric"]).sort_values(
            ["event_order_numeric", "event_slug"],
            kind="stable",
        )
        monitored = [
            f"{monitor_season}/{slug}" for slug in monitor["event_slug"].astype(str).tolist()
        ]
    else:
        monitored = []
    return list(dict.fromkeys([*historical, *monitored]))


def resolve_registry_event(registry: pd.DataFrame, event: str) -> pd.Series:
    if registry.empty:
        raise ValueError("Monitoring event registry is empty")
    slug = slugify_value(event)
    mask = registry["event_slug"].astype(str).eq(slug) | registry["event"].astype(str).map(
        slugify_value
    ).eq(slug)
    matches = registry[mask]
    if matches.empty:
        raise ValueError(f"Event is not registered for monitoring: {event}")
    return matches.iloc[0]


def frozen_gate_configuration_valid(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    required = {
        "min_current_season_prior_events",
        "min_prior_candidate_folds",
        "min_prior_candidate_predictions",
        "improvement_margin_sec",
    }
    if not required <= set(value):
        return False
    numeric = pd.to_numeric(pd.Series([value[key] for key in required]), errors="coerce")
    return bool(numeric.notna().all() and numeric.ge(0).all())


def temporal_weighting_configuration_valid(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    required = {
        "policy",
        "current_season_weight",
        "previous_season_weight",
        "older_season_weight",
        "half_life_events",
        "min_current_season_events",
    }
    return required <= set(value)


def registry_order_unique(
    registry: pd.DataFrame,
    protocol_name: str,
    season: int,
    order_value: object,
) -> bool:
    if registry.empty or pd.isna(order_value) or "event_order" not in registry:
        return False
    frame = registry[
        registry.get("protocol_name", pd.Series(dtype=str)).astype(str).eq(str(protocol_name))
        & pd.to_numeric(registry.get("monitor_season", pd.Series(dtype=float)), errors="coerce").eq(
            int(season)
        )
    ]
    return pd.to_numeric(frame["event_order"], errors="coerce").eq(int(order_value)).sum() == 1


def registered_feature_path(
    config: DataConfig,
    event_row: pd.Series,
    season: int,
    event: str,
) -> Path:
    registered = resolve_registered_path(config, event_row.get("feature_artifact_path"))
    if registered is not None:
        return registered
    return feature_artifact_path(config, season, event)


def read_feature_artifact(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError):
        return pd.DataFrame()


def legacy_exclusion_status(checks: pd.DataFrame) -> str:
    if checks.empty:
        return "not_evaluated"
    rows = checks[
        checks["check_name"].astype(str).eq("legacy_rows_excluded_from_future_prior_evidence")
    ]
    if rows.empty:
        return "not_evaluated"
    return "valid" if rows["status"].astype(str).eq("pass").all() else "invalid"


def next_preflight_command(protocol_name: str, event: str, status: str) -> str:
    if status == PREFLIGHT_READY:
        return (
            "python -m f1_prediction.cli prospective-monitoring-forecast "
            f'--protocol-name {protocol_name} --event "{event}"'
        )
    if status == PREFLIGHT_ALREADY_FORECASTED:
        return "python -m f1_prediction.cli prospective-monitoring-report"
    return "review prospective_monitoring_preflight_runbook.md"


def resolve_registered_path(config: DataConfig, value: object) -> Path | None:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    path = Path(str(value))
    return path if path.is_absolute() else config.project_root / path


def assert_forecast_not_exists(metrics_dir: Path, protocol_name: str, event_slug: str) -> None:
    forecasts = read_parquet(metrics_dir / "prospective_monitoring_forecasts.parquet")
    if forecasts.empty:
        return
    exists = forecasts["protocol_name"].astype(str).eq(protocol_name) & forecasts[
        "event_slug"
    ].astype(str).eq(event_slug)
    if bool(exists.any()):
        raise ValueError(f"Forecast snapshot already exists for {event_slug}")


def build_event_order_reconciliation(
    protocol: dict[str, Any],
    registry: pd.DataFrame,
    forecasts: pd.DataFrame,
    settlements: pd.DataFrame,
    ledger: pd.DataFrame | None = None,
    *,
    metrics_dir: Path | None = None,
) -> pd.DataFrame:
    """Compare immutable artifact event order to canonical registry chronology."""
    columns = event_order_reconciliation_columns()
    if not protocol:
        return pd.DataFrame(columns=columns)
    protocol_name = str(protocol.get("protocol_name"))
    monitor_season = int(protocol.get("monitor_season", 0) or 0)
    registry_path = (
        metrics_dir / "prospective_monitoring_event_registry.csv"
        if metrics_dir is not None
        else Path("reports/metrics/prospective_monitoring_event_registry.csv")
    )
    rows: list[dict[str, object]] = []
    for event in artifact_event_order_records(forecasts, settlements, ledger):
        if str(event.get("protocol_name")) != protocol_name:
            continue
        slug = str(event.get("event_slug"))
        lineage = resolve_registry_event_order(
            registry,
            protocol_name=protocol_name,
            monitor_season=monitor_season,
            event_slug=slug,
            registry_path=registry_path,
            strict=False,
        )
        registry_order = lineage.get("event_order", pd.NA)
        artifact_order = numeric_int_or_na(event.get("artifact_event_order"))
        match = (
            pd.notna(artifact_order)
            and pd.notna(registry_order)
            and int(artifact_order) == int(registry_order)
            and bool(lineage.get("event_order_registry_valid", False))
        )
        status = (
            EVENT_ORDER_VALID_STATUS
            if match
            else str(lineage.get("event_order_lineage_status", EVENT_ORDER_INVALID_STATUS))
        )
        if status == EVENT_ORDER_VALID_STATUS and not match:
            status = EVENT_ORDER_LEGACY_STATUS
        elif not match and bool(lineage.get("event_order_registry_valid", False)):
            status = EVENT_ORDER_LEGACY_STATUS
        has_settlement = bool(event.get("has_settlement", False))
        has_future_eligible = bool(event.get("artifact_future_eligible", False))
        synthetic_rehearsal = synthetic_rehearsal_event_slug(slug)
        eligible_after = bool(
            match and has_settlement and has_future_eligible and not synthetic_rehearsal
        )
        action = (
            "retain_for_prior_monitoring_evidence"
            if eligible_after
            else "exclude_from_prior_monitoring_evidence"
        )
        reason = (
            "registry_event_order_matches_artifact"
            if eligible_after
            else "synthetic_rehearsal_excluded_from_prior_evidence"
            if synthetic_rehearsal
            else "event_order_lineage_mismatch_or_unsettled"
        )
        rows.append(
            {
                "protocol_name": protocol_name,
                "monitor_season": monitor_season,
                "event_slug": slug,
                "forecast_id": event.get("forecast_id"),
                "artifact_event_order": artifact_order,
                "registry_event_order": registry_order,
                "event_order_match": bool(match),
                "event_order_lineage_status": status,
                "artifact_created_at_utc": event.get("artifact_created_at_utc"),
                "affected_by_prior_evidence_lineage": bool(has_settlement or has_future_eligible),
                "eligible_for_future_prior_evidence_after_reconciliation": eligible_after,
                "reconciliation_action": action,
                "reconciliation_reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def artifact_event_order_records(
    forecasts: pd.DataFrame,
    settlements: pd.DataFrame,
    ledger: pd.DataFrame | None,
) -> list[dict[str, object]]:
    frames: list[pd.DataFrame] = []
    if not forecasts.empty:
        forecast = forecasts.copy()
        forecast["_artifact_kind"] = "forecast"
        frames.append(forecast)
    if not settlements.empty:
        settlement = settlements.copy()
        settlement["_artifact_kind"] = "settlement"
        frames.append(settlement)
    if ledger is not None and not ledger.empty:
        ledger_frame = ledger.copy()
        ledger_frame["_artifact_kind"] = "ledger"
        frames.append(ledger_frame)
    if not frames:
        return []
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "forecast_id" not in combined:
        combined["forecast_id"] = pd.NA
    rows: list[dict[str, object]] = []
    group_columns = ["protocol_name", "event_slug", "forecast_id"]
    for keys, group in combined.groupby(group_columns, dropna=False, sort=False):
        protocol_name, event_slug, forecast_id = keys
        created = first_non_missing(
            group,
            ("forecast_created_at_utc", "settled_at_utc", "artifact_created_at_utc"),
        )
        artifact_order = first_non_missing(group, ("event_order", "artifact_event_order"))
        has_settlement = bool(group["_artifact_kind"].astype(str).eq("settlement").any())
        eligible = bool(
            "eligible_for_future_prior_evidence" in group
            and group["eligible_for_future_prior_evidence"].eq(True).any()
        )
        rows.append(
            {
                "protocol_name": protocol_name,
                "event_slug": event_slug,
                "forecast_id": forecast_id,
                "artifact_event_order": artifact_order,
                "artifact_created_at_utc": created,
                "has_settlement": has_settlement,
                "artifact_future_eligible": eligible,
            }
        )
    return rows


def event_order_reconciliation_columns() -> list[str]:
    return [
        "protocol_name",
        "monitor_season",
        "event_slug",
        "forecast_id",
        "artifact_event_order",
        "registry_event_order",
        "event_order_match",
        "event_order_lineage_status",
        "artifact_created_at_utc",
        "affected_by_prior_evidence_lineage",
        "eligible_for_future_prior_evidence_after_reconciliation",
        "reconciliation_action",
        "reconciliation_reason",
    ]


def synthetic_rehearsal_event_slug(event_slug: object) -> bool:
    """Return whether an event slug is explicitly reserved for synthetic rehearsal."""
    slug = str(event_slug or "").strip().lower()
    return slug == "synthetic" or slug.startswith(SYNTHETIC_REHEARSAL_PREFIX)


def settled_event_keys(
    metrics_dir: Path,
    protocol: dict[str, Any],
    registry: pd.DataFrame,
    *,
    current_event_order: int,
) -> set[str]:
    settlements = read_parquet(metrics_dir / "prospective_monitoring_settlements.parquet")
    if settlements.empty:
        return set()
    forecasts = read_parquet(metrics_dir / "prospective_monitoring_forecasts.parquet")
    reconciliation = build_event_order_reconciliation(
        protocol,
        registry,
        forecasts,
        settlements,
        read_csv(metrics_dir / "prospective_monitoring_shadow_evidence_ledger.csv"),
        metrics_dir=metrics_dir,
    )
    eligible = reconciliation[
        reconciliation["eligible_for_future_prior_evidence_after_reconciliation"].astype(bool)
        & pd.to_numeric(reconciliation["registry_event_order"], errors="coerce").lt(
            int(current_event_order)
        )
    ].copy()
    if eligible.empty:
        return set()
    return set(str(protocol["monitor_season"]) + "/" + eligible["event_slug"].astype(str))


def prior_settlement_history(
    metrics_dir: Path,
    protocol: dict[str, Any],
    registry: pd.DataFrame,
    *,
    current_event_order: int,
) -> pd.DataFrame:
    """Return settled prior diagnostic history in season-aware gate-compatible columns."""
    settlements = read_parquet(metrics_dir / "prospective_monitoring_settlements.parquet")
    if settlements.empty:
        return pd.DataFrame()
    forecasts = read_parquet(metrics_dir / "prospective_monitoring_forecasts.parquet")
    reconciliation = build_event_order_reconciliation(
        protocol,
        registry,
        forecasts,
        settlements,
        read_csv(metrics_dir / "prospective_monitoring_shadow_evidence_ledger.csv"),
        metrics_dir=metrics_dir,
    )
    eligible = reconciliation[
        reconciliation["eligible_for_future_prior_evidence_after_reconciliation"].astype(bool)
        & pd.to_numeric(reconciliation["registry_event_order"], errors="coerce").lt(
            int(current_event_order)
        )
    ].copy()
    if eligible.empty:
        return pd.DataFrame()
    eligible_ids = set(eligible["forecast_id"].dropna().astype(str))
    eligible_events = set(eligible["event_slug"].dropna().astype(str))
    frame = settlements[
        settlements["protocol_name"].astype(str).eq(str(protocol.get("protocol_name")))
        & settlements["diagnostic_only"].astype(bool)
        & settlements["eligible_for_future_prior_evidence"].astype(bool)
        & settlements["forecast_id"].astype(str).isin(eligible_ids)
        & settlements["event_slug"].astype(str).isin(eligible_events)
    ].copy()
    registry_orders = eligible.set_index(["forecast_id", "event_slug"])["registry_event_order"]
    frame["_event_order_lookup"] = list(
        zip(frame["forecast_id"].astype(str), frame["event_slug"].astype(str), strict=False)
    )
    frame["event_order"] = frame["_event_order_lookup"].map(registry_orders.to_dict())
    frame = frame[pd.to_numeric(frame["event_order"], errors="coerce").lt(int(current_event_order))]
    if frame.empty:
        return pd.DataFrame()
    frame["source_temporal_weighting_policy"] = frame["temporal_weighting_policy"]
    frame["predicted_quali_gap_to_pole_sec"] = frame["prediction_gap_sec"]
    frame["quali_gap_to_pole_sec"] = frame["actual_gap_sec"]
    frame["prior_monitoring_evidence_lineage_valid"] = True
    return frame.drop(columns=["_event_order_lookup"])


def prior_monitoring_evidence_summary(
    history: pd.DataFrame,
    *,
    event_order_lineage_valid: bool,
) -> dict[str, object]:
    if history.empty:
        return {
            "prior_monitoring_event_count": 0,
            "prior_monitoring_prediction_count": 0,
            "prior_monitoring_event_orders": "[]",
            "prior_monitoring_event_slugs": "[]",
            "prior_monitoring_evidence_lineage_valid": bool(event_order_lineage_valid),
            "prior_monitoring_evidence_exclusion_reasons": "[]",
        }
    orders = sorted(
        {
            int(value)
            for value in pd.to_numeric(history["event_order"], errors="coerce").dropna().tolist()
        }
    )
    slugs = sorted(set(history["event_slug"].dropna().astype(str).tolist()))
    return {
        "prior_monitoring_event_count": int(len(slugs)),
        "prior_monitoring_prediction_count": int(len(history)),
        "prior_monitoring_event_orders": json.dumps(orders),
        "prior_monitoring_event_slugs": json.dumps(slugs),
        "prior_monitoring_evidence_lineage_valid": bool(event_order_lineage_valid),
        "prior_monitoring_evidence_exclusion_reasons": "[]",
    }


def event_outcomes(dataset: pd.DataFrame, season: int, event_slug: str) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    frame = dataset[
        dataset["season"].astype(int).eq(season)
        & dataset["event_slug"].astype(str).eq(event_slug)
        & dataset["checkpoint"].astype(str).eq(FP3_CHECKPOINT)
    ].copy()
    required = {"season", "event_slug", "checkpoint", "driver", "quali_gap_to_pole_sec"}
    if not required <= set(frame.columns):
        return pd.DataFrame()
    return frame.dropna(subset=["quali_gap_to_pole_sec"])


def expected_forecast_hash(metrics_dir: Path, protocol_name: str, event_slug: str) -> str | None:
    audit = read_csv(metrics_dir / "prospective_monitoring_forecast_integrity_audit.csv")
    if audit.empty or "forecast_snapshot_hash" not in audit:
        return None
    mask = audit["protocol_name"].astype(str).eq(protocol_name)
    if "event_key" in audit:
        mask &= audit["event_key"].astype(str).str.endswith(f"/{event_slug}")
    values = audit.loc[mask, "forecast_snapshot_hash"].dropna()
    return str(values.iloc[0]) if not values.empty else None


def forecast_snapshot_hash(forecasts: pd.DataFrame, shadow: pd.DataFrame) -> str:
    keep = [
        "protocol_name",
        "protocol_fingerprint",
        "forecast_id",
        "season",
        "event_slug",
        "checkpoint",
        "driver",
        "prediction_role",
        "prediction_gap_sec",
        "diagnostic_only",
        "family",
        "model_name",
        "feature_group",
        "temporal_weighting_policy",
    ]
    frames = []
    for frame in (forecasts, shadow):
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frames.append(frame.reindex(columns=keep))
    if not frames:
        return stable_signature([])
    combined = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates()
    combined = combined.sort_values(keep[:-1], kind="stable")
    return stable_signature(records_for_json(combined))


def settlement_integrity_rows(
    *,
    protocol: dict[str, Any],
    event_slug: str,
    forecast_id: str,
    mutation_detected: bool,
    fingerprint_valid: bool,
    settlement_valid: bool,
    blocking_reason: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "protocol_name": protocol["protocol_name"],
                "protocol_fingerprint": protocol["protocol_fingerprint"],
                "forecast_id": forecast_id,
                "event_slug": event_slug,
                "settled_at_utc": utc_now(),
                "forecast_preexisted_settlement": True,
                "forecast_fingerprint_valid": fingerprint_valid,
                "forecast_mutation_detected": mutation_detected,
                "settlement_valid": settlement_valid,
                "settlement_blocking_reason": blocking_reason,
            }
        ]
    )


def subset_event(frame: pd.DataFrame, protocol_name: object, event_slug: str) -> pd.DataFrame:
    if frame.empty or "protocol_name" not in frame or "event_slug" not in frame:
        return pd.DataFrame()
    return frame[
        frame["protocol_name"].astype(str).eq(str(protocol_name))
        & frame["event_slug"].astype(str).eq(event_slug)
    ].copy()


def forecast_precedes_settlement(forecasts: pd.DataFrame, settlements: pd.DataFrame) -> bool:
    if forecasts.empty or settlements.empty:
        return True
    forecast_time = pd.to_datetime(forecasts["forecast_created_at_utc"], errors="coerce").min()
    settlement_time = pd.to_datetime(settlements["settled_at_utc"], errors="coerce").min()
    if pd.isna(forecast_time) or pd.isna(settlement_time):
        return False
    return bool(forecast_time <= settlement_time)


def no_future_settlement_used(
    event: pd.Series,
    settlements: pd.DataFrame,
    event_forecasts: pd.DataFrame,
    *,
    reconciliation: pd.DataFrame | None = None,
) -> bool:
    if event_forecasts.empty or settlements.empty:
        return True
    if "event_order" not in settlements:
        return True
    current_order = pd.to_numeric(pd.Series([event.get("event_order")]), errors="coerce").iloc[0]
    forecast_time = pd.to_datetime(
        event_forecasts["forecast_created_at_utc"],
        errors="coerce",
    ).min()
    frame = settlements.copy()
    if reconciliation is not None and not reconciliation.empty:
        keep = reconciliation[
            [
                "forecast_id",
                "event_slug",
                "registry_event_order",
                "eligible_for_future_prior_evidence_after_reconciliation",
            ]
        ].copy()
        frame = frame.merge(keep, on=["forecast_id", "event_slug"], how="left")
        frame = frame[
            frame["eligible_for_future_prior_evidence_after_reconciliation"]
            .fillna(False)
            .astype(bool)
        ].copy()
        if frame.empty:
            return True
        frame["event_order"] = frame["registry_event_order"]
    future = frame[pd.to_numeric(frame["event_order"], errors="coerce").gt(current_order)]
    if future.empty or pd.isna(forecast_time):
        return True
    if "settled_at_utc" not in future:
        return False
    future_times = pd.to_datetime(future["settled_at_utc"], errors="coerce")
    return not bool(future_times.le(forecast_time).any())


def append_parquet(path: Path, frame: pd.DataFrame) -> None:
    ensure_directory(path.parent)
    existing = read_parquet(path)
    combined = (
        pd.concat([existing, frame], ignore_index=True, sort=False) if not existing.empty else frame
    )
    combined.to_parquet(path, index=False)


def append_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_directory(path.parent)
    existing = read_csv(path)
    combined = (
        pd.concat([existing, frame], ignore_index=True, sort=False) if not existing.empty else frame
    )
    combined.to_csv(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    return _read_json(path) if path.is_file() else None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(_json_clean(payload), output_file, indent=2, allow_nan=False)
        output_file.write("\n")


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_clean(item) for item in value]
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    if isinstance(value, Path):
        return value.as_posix()
    return value


def stable_signature(payload: object) -> str:
    encoded = json.dumps(_json_clean(payload), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify_value(value: object) -> str:
    return str(value).strip().lower().replace(" ", "-").replace("_", "-").replace("/", "-")


def first_value(frame: pd.DataFrame, column: str, default: object) -> object:
    if column not in frame or frame[column].dropna().empty:
        return default
    return frame[column].dropna().iloc[0]


def first_non_missing(frame: pd.DataFrame, columns: tuple[str, ...]) -> object:
    for column in columns:
        if column in frame and not frame[column].dropna().empty:
            return frame[column].dropna().iloc[0]
    return pd.NA


def numeric_int_or_na(value: object) -> object:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").dropna()
    if numeric.empty:
        return pd.NA
    number = numeric.iloc[0]
    if float(number) != float(int(number)):
        return pd.NA
    return int(number)


def _number_or_none(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").dropna()
    return float(numeric.iloc[0]) if not numeric.empty else None


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _relative_report_path(path: Path) -> str:
    parts = path.parts
    if "reports" in parts:
        return Path(*parts[parts.index("reports") :]).as_posix()
    return path.as_posix()


def registry_columns() -> list[str]:
    return [
        "protocol_name",
        "monitor_season",
        "event_order",
        "event",
        "event_slug",
        "checkpoint",
        "feature_rows_available",
        "driver_rows_available",
        "qualifying_targets_available",
        "forecast_status",
        "settlement_status",
        "eligibility_evidence_status",
        "live_policy_status",
        "shadow_candidate_status",
        "readiness_blocking_reason",
        *REGISTRY_ONBOARDING_COLUMNS,
    ]


def forecast_columns() -> list[str]:
    return [
        "protocol_name",
        "protocol_fingerprint",
        "forecast_id",
        "forecast_created_at_utc",
        "monitor_season",
        "event_order",
        "event_order_source",
        "event_order_registry_valid",
        "event_order_registry_path",
        "event_order_registry_protocol_name",
        "event_order_registry_monitor_season",
        "event_order_lineage_status",
        "fold_id",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "driver",
        "driver_key",
        "team",
        "team_key",
        "prediction_role",
        "diagnostic_only",
        "prediction_gap_sec",
        "actual_gap_sec",
        "absolute_error_sec",
        "family",
        "model_name",
        "feature_group",
        "temporal_weighting_policy",
        "source_identity",
        "source_lineage_valid",
        "candidate_eligible_under_frozen_gates",
        "candidate_selection_reason",
        "live_policy_selected",
        "selection_is_live",
        "selection_is_counterfactual",
        "training_completed",
        "training_row_count",
        "training_event_count",
        "training_event_keys",
        "training_seasons",
        "current_season_prior_event_count",
        "training_effective_sample_size",
        "current_event_excluded_from_training",
        "future_same_season_events_excluded",
        "future_seasons_excluded",
        "current_event_target_accessed",
        "forecast_integrity_status",
        "prior_monitoring_event_count",
        "prior_monitoring_prediction_count",
        "prior_monitoring_event_orders",
        "prior_monitoring_event_slugs",
        "prior_monitoring_evidence_lineage_valid",
        "prior_monitoring_evidence_exclusion_reasons",
        "preflight_run_id",
        "preflight_status",
        "preflight_summary_path",
    ]


def manifest_columns() -> list[str]:
    return [
        "protocol_name",
        "protocol_fingerprint",
        "forecast_id",
        "forecast_created_at_utc",
        "monitor_season",
        "event_key",
        "event_order",
        "event_order_source",
        "event_order_registry_valid",
        "event_order_registry_path",
        "event_order_registry_protocol_name",
        "event_order_registry_monitor_season",
        "event_order_lineage_status",
        "test_event",
        "checkpoint",
        "policy_profile",
        "training_seasons_used",
        "training_event_keys_used",
        "training_row_count",
        "training_event_count",
        "temporal_weighting_policy",
        "sample_weight_summary",
        "current_event_excluded_from_training",
        "future_same_season_events_excluded",
        "future_seasons_excluded",
        "prior_monitoring_event_count",
        "prior_monitoring_prediction_count",
        "prior_monitoring_event_orders",
        "prior_monitoring_event_slugs",
        "prior_monitoring_evidence_lineage_valid",
        "prior_monitoring_evidence_exclusion_reasons",
        "preflight_run_id",
        "preflight_status",
        "preflight_summary_path",
    ]


def settlement_columns() -> list[str]:
    return [
        "protocol_name",
        "protocol_fingerprint",
        "forecast_id",
        "settlement_id",
        "settled_at_utc",
        "monitor_season",
        "event_order",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "driver",
        "driver_key",
        "prediction_role",
        "diagnostic_only",
        "prediction_gap_sec",
        "actual_gap_sec",
        "absolute_error_sec",
        "target_evaluable",
        "included_in_metrics",
        "settlement_evaluable",
        "settlement_exclusion_reason",
        "forecast_row_preserved",
        "temporal_weighting_policy",
        "fold_id",
        "settlement_valid",
        "forecast_preexisted_settlement",
        "forecast_fingerprint_valid",
        "forecast_mutation_detected",
        "eligible_for_future_prior_evidence",
        "settlement_blocking_reason",
    ]


def integrity_columns() -> list[str]:
    return [
        "protocol_name",
        "event_slug",
        "event_order",
        "protocol_fingerprint_valid",
        "candidate_identity_valid",
        "default_identity_valid",
        "live_policy_identity_valid",
        "forecast_precedes_settlement",
        "current_event_target_not_accessed",
        "current_event_excluded_from_training",
        "future_same_season_events_excluded",
        "future_seasons_excluded",
        "future_settlement_not_used",
        "prior_settlement_only_evidence",
        "forecast_snapshot_mutation_detected",
        "shadow_rows_excluded_from_live_metrics",
        "live_and_counterfactual_selection_separated",
        "partial_target_coverage_documented",
        "non_evaluable_rows_preserved",
        "non_evaluable_rows_excluded_from_metrics",
        "non_evaluable_rows_excluded_from_prior_evidence",
        "valid_target_rows_exactly_aligned",
        "extra_targets_absent_or_explained",
        "coverage_rate_recorded",
        "forecast_artifact_unchanged_after_target_creation",
        "integrity_status",
    ]
