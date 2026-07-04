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
    forbidden_target_columns,
    target_artifact_path,
    validate_target_artifact,
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
    assert_forecast_not_exists(metrics_dir, protocol_name, str(event_row["event_slug"]))

    dataset = monitoring_dataset_for_forecast(config, protocol, dataset, event_row)
    event_order = ordered_event_keys(dataset)
    prior_settled_events = settled_event_keys(metrics_dir, protocol_name)
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
        protocol_name,
        current_event_order=int(event_row["event_order"]),
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
    )
    leakage = monitoring_forecast_integrity_rows(
        source["leakage"],
        protocol=protocol,
        forecast_id=forecast_id,
        event_key=event_key,
    )
    forecasts, shadow = monitoring_prediction_rows(
        source=source,
        protocol=protocol,
        forecast_id=forecast_id,
        forecast_created=forecast_created,
        event_key=event_key,
        event_order=event_order,
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
    if outcomes.empty:
        outcomes = event_outcomes(dataset, int(protocol["monitor_season"]), event_slug)
    if outcomes.empty:
        raise ValueError(f"Qualifying targets are unavailable for {event_slug}")
    settlements = build_settlement_rows(
        protocol=protocol,
        forecasts=event_forecasts,
        outcomes=outcomes,
        mutation_detected=False,
    )
    if settlements.empty:
        raise ValueError(f"No exact forecast/outcome driver matches for {event_slug}")
    settlement_path = metrics_dir / "prospective_monitoring_settlements.parquet"
    append_parquet(settlement_path, settlements)
    event_metrics = build_event_metrics(read_parquet(settlement_path))
    ledger = build_shadow_evidence_ledger(read_parquet(settlement_path))
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
    integrity = refresh_integrity_outputs(metrics_dir, protocol) if protocol else {}
    status_by_event = build_status_by_event(protocol, registry, forecasts, settlements)
    live_summary = build_live_policy_summary(settlements)
    shadow_summary = build_shadow_candidate_summary(settlements)
    gate_timeline = build_gate_timeline(forecasts)
    evidence_growth = build_evidence_growth(settlements)
    summary = build_monitoring_summary_payload(
        protocol=protocol,
        registry=registry,
        forecasts=forecasts,
        settlements=settlements,
        integrity=integrity,
        live_summary=live_summary,
        shadow_summary=shadow_summary,
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
        "event_ordering_contract": "ordered_event_keys: season then event_order or dataset order",
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
    columns = [column for column in PREDICTION_COLUMNS if column in test_rows]
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
    result["event_order"] = event_order.index(event_key) if event_key in event_order else pd.NA
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
) -> pd.DataFrame:
    """Build forecast integrity audit rows."""
    frame = pd.DataFrame(leakage_rows)
    if frame.empty:
        frame = pd.DataFrame([{}])
    frame["protocol_name"] = protocol["protocol_name"]
    frame["protocol_fingerprint"] = protocol["protocol_fingerprint"]
    frame["forecast_id"] = forecast_id
    frame["event_key"] = event_key
    frame["protocol_fingerprint_valid"] = True
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
    candidate_eligible: bool,
    selection_reason: str,
    snapshot_hash: str,
) -> dict[str, object]:
    """Build the event-level forecast selection record."""
    season, slug = parse_event_key(event_key)
    return {
        "protocol_name": protocol["protocol_name"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "forecast_id": forecast_id,
        "forecast_created_at_utc": forecast_created,
        "season": season,
        "event_slug": slug,
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


def build_settlement_rows(
    *,
    protocol: dict[str, Any],
    forecasts: pd.DataFrame,
    outcomes: pd.DataFrame,
    mutation_detected: bool,
) -> pd.DataFrame:
    """Join forecast rows to actual outcomes by exact event/checkpoint/driver keys."""
    join_cols = ["season", "event_slug", "checkpoint", "driver"]
    outcome_cols = [*join_cols, "quali_gap_to_pole_sec"]
    merged = forecasts.merge(outcomes.loc[:, outcome_cols], on=join_cols, how="inner")
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
    merged["actual_gap_sec"] = merged["quali_gap_to_pole_sec"]
    merged["absolute_error_sec"] = (
        pd.to_numeric(merged["prediction_gap_sec"], errors="coerce")
        - pd.to_numeric(merged["actual_gap_sec"], errors="coerce")
    ).abs()
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
    )
    merged["settlement_blocking_reason"] = ""
    return merged.reindex(columns=settlement_columns())


def build_event_metrics(settlements: pd.DataFrame) -> pd.DataFrame:
    """Score live and diagnostic rows separately."""
    columns = [
        "protocol_name",
        "season",
        "event_slug",
        "checkpoint",
        "prediction_role",
        "diagnostic_only",
        "rows",
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
        rows.append(
            {
                "protocol_name": protocol_name,
                "season": season,
                "event_slug": event_slug,
                "checkpoint": checkpoint,
                "prediction_role": role,
                "diagnostic_only": bool(group["diagnostic_only"].astype(bool).iloc[0]),
                "rows": int(len(group)),
                "mae_gap_sec": _number_or_none(group["absolute_error_sec"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_shadow_evidence_ledger(settlements: pd.DataFrame) -> pd.DataFrame:
    """Return settled diagnostic rows available to future event gates."""
    columns = [
        "protocol_name",
        "protocol_fingerprint",
        "forecast_id",
        "season",
        "event_slug",
        "event_order",
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
    return shadow.reindex(columns=columns)


def refresh_integrity_outputs(metrics_dir: Path, protocol: dict[str, Any]) -> dict[str, object]:
    """Rebuild monitoring integrity summary and event/failure tables."""
    registry = read_csv(metrics_dir / "prospective_monitoring_event_registry.csv")
    forecasts = read_parquet(metrics_dir / "prospective_monitoring_forecasts.parquet")
    settlements = read_parquet(metrics_dir / "prospective_monitoring_settlements.parquet")
    forecast_audit = read_csv(metrics_dir / "prospective_monitoring_forecast_integrity_audit.csv")
    settlement_audit = read_csv(
        metrics_dir / "prospective_monitoring_settlement_integrity_audit.csv"
    )
    by_event = build_integrity_by_event(protocol, registry, forecasts, settlements, forecast_audit)
    failures = build_integrity_failures(by_event, settlement_audit)
    summary = {
        "status": "valid" if failures.empty else "invalid",
        "protocol_name": protocol.get("protocol_name"),
        "protocol_fingerprint": protocol.get("protocol_fingerprint"),
        "events_checked": int(len(by_event)),
        "failure_count": int(len(failures)),
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
        }
        row["integrity_status"] = (
            "valid" if all(bool(row[col]) for col in columns[3:-1]) else "invalid"
        )
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


def build_status_by_event(
    protocol: dict[str, Any],
    registry: pd.DataFrame,
    forecasts: pd.DataFrame,
    settlements: pd.DataFrame,
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
    ]
    if registry.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, event in registry.iterrows():
        slug = str(event["event_slug"])
        event_forecasts = subset_event(forecasts, protocol.get("protocol_name"), slug)
        event_settlements = subset_event(settlements, protocol.get("protocol_name"), slug)
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
    return pd.DataFrame(
        [
            {
                "protocol_name": live["protocol_name"].iloc[0],
                "rows": int(len(live)),
                "settled_events": int(live["event_slug"].nunique()),
                "mae_gap_sec": _number_or_none(live["absolute_error_sec"].mean()),
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
        rows.append(
            {
                "protocol_name": protocol_name,
                "prediction_role": role,
                "rows": int(len(group)),
                "settled_events": int(group["event_slug"].nunique()),
                "mae_gap_sec": _number_or_none(group["absolute_error_sec"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_gate_timeline(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Summarize frozen gate status over forecasted events."""
    columns = [
        "protocol_name",
        "event_order",
        "event_slug",
        "candidate_eligible_under_frozen_gates",
        "counterfactual_shadow_selected",
    ]
    if forecasts.empty:
        return pd.DataFrame(columns=columns)
    weighted = forecasts[
        forecasts["prediction_role"].astype(str).eq("season_aware_weighted_candidate_shadow")
    ]
    rows = []
    for keys, group in weighted.groupby(["protocol_name", "event_order", "event_slug"], sort=True):
        protocol_name, event_order, event_slug = keys
        eligible = bool(group["candidate_eligible_under_frozen_gates"].astype(bool).any())
        rows.append(
            {
                "protocol_name": protocol_name,
                "event_order": event_order,
                "event_slug": event_slug,
                "candidate_eligible_under_frozen_gates": eligible,
                "counterfactual_shadow_selected": eligible,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_evidence_growth(settlements: pd.DataFrame) -> pd.DataFrame:
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
) -> dict[str, object]:
    """Build the top-level monitoring summary JSON."""
    available = bool(protocol)
    settled_events = int(settlements["event_slug"].nunique()) if not settlements.empty else 0
    forecasted_events = int(forecasts["event_slug"].nunique()) if not forecasts.empty else 0
    fresh = "available" if settled_events else "not_collected"
    status = "active" if available else "missing_protocol"
    if available and registry.empty:
        status = "not_ready"
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
            ],
            "figures": [],
        },
        "generation_issues": [],
        "generated_at_utc": utc_now(),
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


def settled_event_keys(metrics_dir: Path, protocol_name: str) -> set[str]:
    settlements = read_parquet(metrics_dir / "prospective_monitoring_settlements.parquet")
    if settlements.empty:
        return set()
    frame = settlements[settlements["protocol_name"].astype(str).eq(protocol_name)]
    return set(frame["season"].astype(str) + "/" + frame["event_slug"].astype(str))


def prior_settlement_history(
    metrics_dir: Path,
    protocol_name: str,
    *,
    current_event_order: int,
) -> pd.DataFrame:
    """Return settled prior diagnostic history in season-aware gate-compatible columns."""
    settlements = read_parquet(metrics_dir / "prospective_monitoring_settlements.parquet")
    if settlements.empty:
        return pd.DataFrame()
    frame = settlements[
        settlements["protocol_name"].astype(str).eq(protocol_name)
        & settlements["diagnostic_only"].astype(bool)
        & settlements["eligible_for_future_prior_evidence"].astype(bool)
    ].copy()
    frame = frame[
        pd.to_numeric(frame["event_order"], errors="coerce")
        .fillna(float("inf"))
        .lt(current_event_order)
    ]
    if frame.empty:
        return pd.DataFrame()
    frame["source_temporal_weighting_policy"] = frame["temporal_weighting_policy"]
    frame["predicted_quali_gap_to_pole_sec"] = frame["prediction_gap_sec"]
    frame["quali_gap_to_pole_sec"] = frame["actual_gap_sec"]
    return frame


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
    future = settlements[
        pd.to_numeric(settlements["event_order"], errors="coerce").gt(current_order)
    ]
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
    ]


def manifest_columns() -> list[str]:
    return [
        "protocol_name",
        "protocol_fingerprint",
        "forecast_id",
        "forecast_created_at_utc",
        "monitor_season",
        "event_key",
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
        "prediction_role",
        "diagnostic_only",
        "prediction_gap_sec",
        "actual_gap_sec",
        "absolute_error_sec",
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
        "integrity_status",
    ]
