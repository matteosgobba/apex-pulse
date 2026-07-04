"""Local monitored-season data onboarding with target isolation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig, FeatureConfig
from f1_prediction.data.fastf1_loader import build_lap_output_path
from f1_prediction.data.identity import add_identity_columns
from f1_prediction.features.build import (
    DEFAULT_PRACTICE_SESSIONS,
    build_session_features,
    build_session_features_output_path,
)
from f1_prediction.features.data_quality import DataQualitySettings, add_data_quality_features
from f1_prediction.features.modeling_dataset import (
    IDENTIFIER_COLUMNS,
    SESSION_IDENTIFIER_COLUMNS,
    get_feature_columns,
)
from f1_prediction.features.qualifying_targets import build_qualifying_targets
from f1_prediction.features.relative_features import add_relative_practice_features
from f1_prediction.utils.paths import ensure_directory, slugify

MONITORING_CHECKPOINT = "after_fp3"
POLICY_RECOMMENDATION = "season_aware_candidate_requires_more_evidence"
FORBIDDEN_TARGET_COLUMNS: tuple[str, ...] = (
    "quali_gap_to_pole_sec",
    "quali_position",
    "quali_best_lap_time_sec",
    "reached_q2",
    "reached_q3",
)
TARGET_ARTIFACT_COLUMNS: tuple[str, ...] = (
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
    "reached_q2",
    "reached_q3",
    "target_evaluable",
    "target_coverage_status",
    "target_source_status",
    "target_created_at_utc",
)
TARGET_COVERAGE_COLUMNS: tuple[str, ...] = (
    "season",
    "event",
    "event_slug",
    "checkpoint",
    "driver",
    "driver_key",
    "team",
    "team_key",
    "feature_row_present",
    "qualifying_target_present",
    "target_evaluable",
    "target_missing_reason",
    "target_source_status",
    "target_validation_status",
    "included_in_settlement_metrics",
    "excluded_from_settlement_metrics",
    "settlement_exclusion_reason",
)
TARGET_COVERAGE_SUMMARY_FIELDS: tuple[str, ...] = (
    "feature_driver_count",
    "target_driver_count",
    "evaluable_driver_count",
    "non_evaluable_driver_count",
    "feature_target_coverage_rate",
    "extra_target_count",
    "identifier_conflict_count",
    "target_coverage_status",
    "partial_target_coverage",
    "settlement_metric_status",
)
REGISTRY_ONBOARDING_COLUMNS: tuple[str, ...] = (
    "feature_artifact_path",
    "feature_artifact_fingerprint",
    "feature_artifact_valid",
    "target_artifact_path",
    "target_artifact_present",
    "target_artifact_valid",
    "prequalification_ready",
    "forecastable",
    "settleable",
    "onboarding_status",
    "onboarding_blocking_reason",
    "feature_driver_count",
    "target_driver_count",
    "evaluable_driver_count",
    "non_evaluable_driver_count",
    "target_coverage_rate",
    "partial_target_coverage",
    "target_coverage_status",
    "settlement_metric_status",
)


@dataclass(frozen=True)
class MonitoringOnboardingSummary:
    """Paths and status produced by one monitoring onboarding command."""

    status: str
    summary_path: Path
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...] = ()
    missing_inputs: tuple[str, ...] = ()
    generation_issues: tuple[str, ...] = ()


def prepare_monitoring_event(
    config: DataConfig,
    feature_config: FeatureConfig,
    *,
    season: int,
    event: str,
    force: bool = False,
) -> MonitoringOnboardingSummary:
    """Build FP3-safe monitoring features from local practice artifacts only."""
    event_dir = monitoring_event_dir(config, season, event)
    ensure_directory(event_dir)
    manifest_path = event_manifest_path(config, season, event)
    source_paths = {
        session: build_lap_output_path(config.lap_output_dir, season, event, session)
        for session in DEFAULT_PRACTICE_SESSIONS
    }
    missing_sessions = [session for session, path in source_paths.items() if not path.is_file()]
    if missing_sessions:
        manifest = base_manifest(config, season, event)
        manifest.update(
            {
                "source_paths": {
                    session: portable_path(path, config.project_root)
                    for session, path in source_paths.items()
                },
                "source_availability": {
                    session: source_paths[session].is_file()
                    for session in DEFAULT_PRACTICE_SESSIONS
                },
                "fp_session_coverage": ",".join(
                    session
                    for session in DEFAULT_PRACTICE_SESSIONS
                    if source_paths[session].is_file()
                ),
                "driver_row_count": 0,
                "safe_feature_count": 0,
                "forbidden_target_column_count": 0,
                "feature_artifact_fingerprint": None,
                "preparation_status": "failed",
                "target_availability_status": "not_checked",
                "chronological_event_order_status": "unknown",
                "readiness_blockers": [
                    f"missing_{session.lower()}_raw_laps" for session in missing_sessions
                ],
            }
        )
        write_json(manifest_path, manifest)
        refresh_onboarding_integrity(config)
        raise FileNotFoundError(
            "Missing local practice lap artifacts: "
            + ", ".join(f"{session}={source_paths[session]}" for session in missing_sessions)
        )

    build_session_features(
        season,
        event,
        config,
        feature_config,
        sessions=DEFAULT_PRACTICE_SESSIONS,
        force=force,
    )
    practice_path = build_session_features_output_path(
        config.session_features_output_dir,
        season,
        event,
    )
    practice = pd.read_parquet(practice_path)
    features = build_fp3_safe_feature_rows(practice, feature_config)
    forbidden = forbidden_target_columns(features)
    if forbidden:
        raise ValueError(
            "Prepared monitoring feature artifact would contain target columns: "
            + ", ".join(forbidden)
        )
    validate_feature_grain(features)
    output_path = feature_artifact_path(config, season, event)
    ensure_directory(output_path.parent)
    features.to_parquet(output_path, index=False)
    fingerprint = artifact_fingerprint(output_path)
    manifest = base_manifest(config, season, event)
    manifest.update(
        {
            "source_paths": {
                session: portable_path(path, config.project_root)
                for session, path in source_paths.items()
            },
            "source_availability": {session: True for session in DEFAULT_PRACTICE_SESSIONS},
            "fp_session_coverage": ",".join(DEFAULT_PRACTICE_SESSIONS),
            "practice_feature_path": portable_path(practice_path, config.project_root),
            "feature_artifact_path": portable_path(output_path, config.project_root),
            "driver_row_count": int(features["driver"].nunique()),
            "safe_feature_count": int(len(get_feature_columns(features))),
            "forbidden_target_column_count": 0,
            "feature_artifact_fingerprint": fingerprint,
            "feature_created_at_utc": utc_now(),
            "preparation_status": "prepared",
            "target_availability_status": target_status(config, season, event),
            "chronological_event_order_status": chronological_status(features),
            "readiness_blockers": []
            if chronological_status(features) == "valid"
            else ["chronological_event_order_missing"],
        }
    )
    write_json(manifest_path, manifest)
    refresh_onboarding_integrity(config)
    return MonitoringOnboardingSummary(
        status="prepared",
        summary_path=manifest_path,
        table_paths=(output_path, manifest_path),
    )


def register_monitoring_event(
    config: DataConfig,
    *,
    protocol_name: str,
    season: int,
    event: str,
    event_order: int | None = None,
) -> MonitoringOnboardingSummary:
    """Register one prepared event in the frozen monitoring event registry."""
    metrics_dir = config.metrics_output_dir
    protocol = load_protocol(metrics_dir, protocol_name)
    if int(protocol.get("monitor_season", -1)) != int(season):
        raise ValueError(
            f"Protocol {protocol_name} monitors {protocol.get('monitor_season')}, not {season}"
        )
    if protocol.get("protocol_fingerprint") != protocol_fingerprint(protocol):
        raise ValueError(f"Monitoring protocol fingerprint is invalid: {protocol_name}")
    manifest = load_event_manifest(config, season, event)
    features, feature_valid, feature_blocker = load_valid_feature_artifact(config, season, event)
    if not feature_valid:
        raise ValueError(feature_blocker or "Feature artifact is invalid")
    if int(manifest.get("season", season)) != int(season):
        raise ValueError("Prepared event manifest season does not match registration request")
    if manifest.get("event_slug") != slugify(event):
        raise ValueError("Prepared event manifest event does not match registration request")
    order = resolve_event_order(
        event_order=event_order,
        manifest=manifest,
        features=features,
        registry=read_registry(metrics_dir),
    )
    target_path = target_artifact_path(config, season, event)
    target_present = target_path.is_file()
    target_valid = validate_target_artifact(config, season, event)[0] if target_present else False
    forecasts = read_parquet(metrics_dir / "prospective_monitoring_forecasts.parquet")
    settlements = read_parquet(metrics_dir / "prospective_monitoring_settlements.parquet")
    registry = upsert_registry_row(
        read_registry(metrics_dir),
        protocol=protocol,
        manifest=manifest,
        features=features,
        event_order=order,
        feature_valid=True,
        feature_blocker="",
        target_present=target_present,
        target_valid=target_valid,
        forecasts=forecasts,
        settlements=settlements,
    )
    registry_path = metrics_dir / "prospective_monitoring_event_registry.csv"
    ensure_directory(registry_path.parent)
    registry.to_csv(registry_path, index=False)
    refresh_onboarding_integrity(config)
    return MonitoringOnboardingSummary(
        status="registered",
        summary_path=registry_path,
        table_paths=(registry_path, event_manifest_path(config, season, event)),
    )


def add_monitoring_targets(
    config: DataConfig,
    *,
    season: int,
    event: str,
) -> MonitoringOnboardingSummary:
    """Create settlement-only qualifying targets for a prepared monitoring event."""
    manifest = load_event_manifest(config, season, event)
    features, feature_valid, feature_blocker = load_valid_feature_artifact(config, season, event)
    if not feature_valid:
        raise ValueError(feature_blocker or "Valid pre-qualification feature artifact is required")
    before_fingerprint = artifact_fingerprint(feature_artifact_path(config, season, event))
    if manifest.get("feature_artifact_fingerprint") != before_fingerprint:
        raise ValueError("Feature artifact fingerprint mismatch before target creation")
    q_path = build_lap_output_path(config.lap_output_dir, season, event, "Q")
    if not q_path.is_file():
        raise FileNotFoundError(f"Local qualifying lap artifact is missing: {q_path}")
    forecast_path = config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    before_forecast_fingerprint = (
        artifact_fingerprint(forecast_path) if forecast_path.is_file() else None
    )
    raw_q = pd.read_parquet(q_path)
    targets = build_qualifying_targets(raw_q, season=season, event=event)
    target_rows, coverage, coverage_summary = build_target_artifacts_with_coverage(
        features,
        targets,
        season=season,
        event=event,
    )
    if target_rows.empty:
        write_target_coverage_artifact(config, season, event, coverage)
        raise ValueError("No evaluable qualifying targets can be constructed for settlement")
    output_path = target_artifact_path(config, season, event)
    coverage_path = target_coverage_path(config, season, event)
    ensure_directory(output_path.parent)
    target_rows.to_parquet(output_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    after_fingerprint = artifact_fingerprint(feature_artifact_path(config, season, event))
    if before_fingerprint != after_fingerprint:
        raise ValueError("Feature artifact mutated during target creation")
    after_forecast_fingerprint = (
        artifact_fingerprint(forecast_path) if forecast_path.is_file() else None
    )
    if before_forecast_fingerprint != after_forecast_fingerprint:
        raise ValueError("Forecast artifact mutated during target creation")
    manifest.update(
        {
            "target_artifact_path": portable_path(output_path, config.project_root),
            "target_artifact_fingerprint": artifact_fingerprint(output_path),
            "target_coverage_artifact_path": portable_path(coverage_path, config.project_root),
            "target_coverage_artifact_fingerprint": artifact_fingerprint(coverage_path),
            "target_source_paths": {"Q": portable_path(q_path, config.project_root)},
            "target_availability_status": "available",
            "target_created_at_utc": utc_now(),
            "feature_unchanged_after_target_creation": True,
            "forecast_artifact_unchanged_after_target_creation": True,
            **coverage_summary,
        }
    )
    write_json(event_manifest_path(config, season, event), manifest)
    update_registered_target_state(config, season, event)
    refresh_onboarding_integrity(config)
    return MonitoringOnboardingSummary(
        status="targets_added",
        summary_path=event_manifest_path(config, season, event),
        table_paths=(output_path, coverage_path, event_manifest_path(config, season, event)),
    )


def create_monitoring_data_readiness_report(config: DataConfig) -> MonitoringOnboardingSummary:
    """Summarize monitored-season onboarding readiness from local artifacts."""
    metrics_dir = config.metrics_output_dir
    figures_dir = metrics_dir.parent / "figures"
    ensure_directory(metrics_dir)
    ensure_directory(figures_dir)
    integrity = refresh_onboarding_integrity(config)
    protocol = read_json_if_exists(metrics_dir / "prospective_monitoring_protocol.json") or {}
    event_order_integrity = (
        read_json_if_exists(
            metrics_dir / "prospective_monitoring_event_order_integrity_summary.json"
        )
        or {}
    )
    monitoring_summary = (
        read_json_if_exists(metrics_dir / "prospective_monitoring_summary.json") or {}
    )
    registry = read_registry(metrics_dir)
    manifests = discover_event_manifests(config)
    by_event = build_readiness_by_event(config, protocol, registry, manifests)
    missing = build_missing_inputs(protocol, registry, by_event)
    summary = {
        "status": readiness_status(protocol, by_event),
        "monitoring_data_onboarding_available": True,
        "protocol_name": protocol.get("protocol_name"),
        "protocol_fingerprint": protocol.get("protocol_fingerprint"),
        "monitor_season": protocol.get("monitor_season"),
        "prepared_event_count": int(by_event["prepared"].sum()) if not by_event.empty else 0,
        "registered_event_count": int(by_event["registered"].sum()) if not by_event.empty else 0,
        "forecastable_event_count": int(by_event["forecastable"].sum())
        if not by_event.empty
        else 0,
        "target_available_event_count": int(by_event["target_artifact_present"].sum())
        if not by_event.empty
        else 0,
        "settleable_event_count": int(by_event["settleable"].sum()) if not by_event.empty else 0,
        "target_isolation_status": integrity.get("target_isolation_status", "unknown"),
        "monitoring_target_coverage_status": aggregate_coverage_status(by_event),
        "monitoring_target_coverage_rate": aggregate_coverage_rate(by_event),
        "monitoring_evaluable_driver_count": int(by_event["evaluable_driver_count"].sum())
        if "evaluable_driver_count" in by_event
        else 0,
        "monitoring_non_evaluable_driver_count": int(by_event["non_evaluable_driver_count"].sum())
        if "non_evaluable_driver_count" in by_event
        else 0,
        "monitoring_partial_coverage_event_count": int(
            by_event["partial_target_coverage"].astype(bool).sum()
        )
        if "partial_target_coverage" in by_event
        else 0,
        "monitoring_settlement_metric_status": aggregate_metric_status(by_event),
        "monitoring_event_order_lineage_status": monitoring_summary.get(
            "monitoring_event_order_lineage_status",
            event_order_integrity.get("status", "not_evaluated"),
        ),
        "monitoring_legacy_event_order_exclusion_count": int(
            monitoring_summary.get(
                "monitoring_legacy_event_order_exclusion_count",
                event_order_integrity.get("legacy_event_order_exclusion_count", 0),
            )
            or 0
        ),
        "monitoring_prior_evidence_lineage_status": monitoring_summary.get(
            "monitoring_prior_evidence_lineage_status",
            event_order_integrity.get("prior_evidence_lineage_status", "not_evaluated"),
        ),
        "monitoring_next_forecast_prior_evidence_status": monitoring_summary.get(
            "monitoring_next_forecast_prior_evidence_status",
            event_order_integrity.get("prior_evidence_lineage_status", "not_evaluated"),
        ),
        "chronological_order_status": integrity.get("chronological_order_status", "unknown"),
        "integrity_status": integrity.get("status", "missing"),
        "available_next_actions": available_next_actions(by_event),
        "policy_recommendation": POLICY_RECOMMENDATION,
        "generated_at_utc": utc_now(),
    }
    summary_path = metrics_dir / "monitoring_data_readiness_summary.json"
    by_event_path = metrics_dir / "monitoring_data_readiness_by_event.csv"
    missing_path = metrics_dir / "monitoring_data_readiness_missing_inputs.csv"
    write_json(summary_path, summary)
    by_event.to_csv(by_event_path, index=False)
    missing.to_csv(missing_path, index=False)
    figures, issues = generate_readiness_figures(
        figures_dir=figures_dir,
        readiness_by_event=by_event,
        integrity_by_event=read_csv(metrics_dir / "monitoring_onboarding_integrity_by_event.csv"),
    )
    summary["generated_outputs"] = {
        "metrics": [
            "reports/metrics/monitoring_data_readiness_summary.json",
            "reports/metrics/monitoring_data_readiness_by_event.csv",
            "reports/metrics/monitoring_data_readiness_missing_inputs.csv",
        ],
        "figures": [relative_report_path(path) for path in figures],
    }
    summary["generation_issues"] = issues
    write_json(summary_path, summary)
    return MonitoringOnboardingSummary(
        status=str(summary["status"]),
        summary_path=summary_path,
        table_paths=(summary_path, by_event_path, missing_path),
        figure_paths=tuple(figures),
        missing_inputs=tuple(missing["requirement"].astype(str).tolist())
        if not missing.empty
        else (),
        generation_issues=tuple(issues),
    )


def build_fp3_safe_feature_rows(
    practice_features: pd.DataFrame,
    feature_config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Build targetless after-FP3 rows from practice aggregates."""
    practice = add_relative_practice_features(add_identity_columns(practice_features))
    source_columns = [
        column
        for column in practice.columns
        if column not in SESSION_IDENTIFIER_COLUMNS
        and not column.lower().startswith(("q_", "quali_"))
        and column not in FORBIDDEN_TARGET_COLUMNS
        and "target" not in column.lower()
    ]
    identity = latest_driver_identity(practice)
    frame = identity.copy()
    frame["checkpoint"] = MONITORING_CHECKPOINT
    for session in DEFAULT_PRACTICE_SESSIONS:
        rows = practice[practice["session"].astype(str).eq(session)]
        renamed = rows.loc[:, ["driver_key", *source_columns]].rename(
            columns={column: f"{session.lower()}_{column}" for column in source_columns}
        )
        frame = frame.merge(renamed, on="driver_key", how="left", validate="one_to_one")
    quality = feature_config.data_quality if feature_config is not None else None
    settings = (
        DataQualitySettings(
            extreme_gap_to_session_best_sec=quality.extreme_gap_to_session_best_sec,
            min_push_laps_latest_session=quality.min_push_laps_latest_session,
            min_valid_laps_latest_session=quality.min_valid_laps_latest_session,
        )
        if quality is not None
        else DataQualitySettings()
    )
    frame = add_data_quality_features(frame, settings)
    columns = [column for column in IDENTIFIER_COLUMNS if column in frame]
    feature_columns = [
        column
        for column in get_feature_columns(frame)
        if column not in columns and column not in forbidden_target_columns(frame)
    ]
    extra = [
        column
        for column in ("event_order",)
        if column in frame and column not in columns and column not in feature_columns
    ]
    return frame.loc[:, [*columns, *extra, *feature_columns]]


def latest_driver_identity(practice: pd.DataFrame) -> pd.DataFrame:
    """Return one targetless identity row per driver from the latest practice session."""
    order = {"FP1": 1, "FP2": 2, "FP3": 3}
    frame = practice.copy()
    frame["_session_order"] = frame["session"].map(order).fillna(0)
    columns = [
        column for column in IDENTIFIER_COLUMNS if column in frame and column != "checkpoint"
    ]
    if "event_order" in frame:
        columns.append("event_order")
    latest = frame.sort_values("_session_order", kind="stable").drop_duplicates(
        "driver_key",
        keep="last",
    )
    return latest.loc[:, columns].reset_index(drop=True)


def validate_feature_grain(features: pd.DataFrame) -> None:
    required = {"season", "event_slug", "checkpoint", "driver"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Monitoring features are missing columns: {', '.join(missing)}")
    duplicates = features.duplicated(["season", "event_slug", "checkpoint", "driver"])
    if duplicates.any():
        raise ValueError("Monitoring features are not unique by season/event/checkpoint/driver")


def build_target_artifact_rows(targets: pd.DataFrame, coverage_status: str) -> pd.DataFrame:
    rows = targets.copy()
    rows["target_evaluable"] = True
    rows["target_coverage_status"] = coverage_status
    rows["target_source_status"] = "available"
    rows["target_created_at_utc"] = utc_now()
    return rows.reindex(columns=TARGET_ARTIFACT_COLUMNS)


def build_target_artifacts_with_coverage(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    season: int,
    event: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build valid target rows plus a full feature-driver coverage ledger."""
    validate_target_key_integrity(targets)
    feature_keys = feature_key_frame(features)
    target_keys = target_key_frame(targets)
    extra = target_keys.merge(
        feature_keys[["season", "event_slug", "checkpoint", "driver_key"]],
        on=["season", "event_slug", "checkpoint", "driver_key"],
        how="left",
        indicator=True,
    )
    extra_count = int(extra["_merge"].eq("left_only").sum()) if not extra.empty else 0
    if extra_count:
        raise ValueError(f"Target rows without matching feature rows: {extra_count}")
    conflicts = identifier_conflicts(feature_keys, target_keys)
    if conflicts:
        raise ValueError(f"Feature/target identifier conflicts: {conflicts}")

    target_by_key = {
        key: row
        for key, row in target_keys.set_index(
            ["season", "event_slug", "checkpoint", "driver_key"],
            drop=False,
        ).iterrows()
    }
    coverage_rows: list[dict[str, object]] = []
    for _, feature in feature_keys.iterrows():
        key = (
            int(feature["season"]),
            str(feature["event_slug"]),
            str(feature["checkpoint"]),
            str(feature["driver_key"]),
        )
        target = target_by_key.get(key)
        target_present = target is not None
        evaluable = bool(target_present and target.get("target_evaluable", False))
        missing_reason = ""
        if not target_present:
            missing_reason = "no_qualifying_lap_rows"
        elif not evaluable:
            missing_reason = "no_valid_qualifying_lap"
        exclusion_reason = "" if evaluable else missing_reason
        coverage_rows.append(
            {
                "season": int(feature["season"]),
                "event": feature.get("event", event),
                "event_slug": feature["event_slug"],
                "checkpoint": feature["checkpoint"],
                "driver": feature["driver"],
                "driver_key": feature["driver_key"],
                "team": feature.get("team"),
                "team_key": feature.get("team_key"),
                "feature_row_present": True,
                "qualifying_target_present": target_present,
                "target_evaluable": evaluable,
                "target_missing_reason": missing_reason,
                "target_source_status": "available" if target_present else "missing",
                "target_validation_status": "valid" if evaluable else "non_evaluable",
                "included_in_settlement_metrics": evaluable,
                "excluded_from_settlement_metrics": not evaluable,
                "settlement_exclusion_reason": exclusion_reason,
            }
        )
    coverage = pd.DataFrame(coverage_rows, columns=TARGET_COVERAGE_COLUMNS)
    evaluable_count = int(coverage["target_evaluable"].astype(bool).sum())
    feature_count = int(len(coverage))
    target_driver_count = int(coverage["qualifying_target_present"].astype(bool).sum())
    non_evaluable_count = feature_count - evaluable_count
    coverage_rate = float(evaluable_count / feature_count) if feature_count else 0.0
    status = target_coverage_status(
        feature_driver_count=feature_count,
        evaluable_driver_count=evaluable_count,
        non_evaluable_driver_count=non_evaluable_count,
        extra_target_count=extra_count,
        identifier_conflict_count=len(conflicts),
    )
    target_rows = build_target_artifact_rows(
        target_keys[target_keys["target_evaluable"].astype(bool)].copy(),
        coverage_status=status,
    )
    summary = {
        "feature_driver_count": feature_count,
        "target_driver_count": target_driver_count,
        "evaluable_driver_count": evaluable_count,
        "non_evaluable_driver_count": non_evaluable_count,
        "feature_target_coverage_rate": coverage_rate,
        "extra_target_count": extra_count,
        "identifier_conflict_count": len(conflicts),
        "target_coverage_status": status,
        "partial_target_coverage": status == "target_coverage_partial",
        "settlement_metric_status": "scorable" if evaluable_count else "not_scorable",
    }
    return target_rows, coverage, summary


def write_target_coverage_artifact(
    config: DataConfig,
    season: int,
    event: str,
    coverage: pd.DataFrame,
) -> None:
    path = target_coverage_path(config, season, event)
    ensure_directory(path.parent)
    coverage.to_csv(path, index=False)


def target_coverage_status(
    *,
    feature_driver_count: int,
    evaluable_driver_count: int,
    non_evaluable_driver_count: int,
    extra_target_count: int,
    identifier_conflict_count: int,
) -> str:
    if extra_target_count or identifier_conflict_count:
        return "target_coverage_invalid"
    if feature_driver_count <= 0 or evaluable_driver_count <= 0:
        return "target_coverage_empty"
    if non_evaluable_driver_count > 0:
        return "target_coverage_partial"
    return "target_coverage_complete"


def validate_target_key_integrity(targets: pd.DataFrame) -> None:
    if targets.empty:
        return
    keys = ["season", "event_slug", "checkpoint", "driver_key"]
    frame = target_key_frame(targets)
    duplicates = frame.duplicated(keys)
    if duplicates.any():
        duplicate_keys = frame.loc[duplicates, keys].to_dict("records")
        raise ValueError(f"Duplicate target rows for exact identifier: {duplicate_keys}")


def feature_key_frame(features: pd.DataFrame) -> pd.DataFrame:
    required = ["season", "event", "event_slug", "checkpoint", "driver", "driver_key"]
    missing = sorted(set(required) - set(features.columns))
    if missing:
        raise ValueError(f"Feature artifact missing identifier columns: {', '.join(missing)}")
    frame = features.copy()
    frame["season"] = frame["season"].astype(int)
    frame["event_slug"] = frame["event_slug"].astype(str)
    frame["checkpoint"] = frame["checkpoint"].astype(str)
    frame["driver_key"] = frame["driver_key"].astype(str)
    frame["driver"] = frame["driver"].astype(str)
    return frame.reindex(
        columns=[
            "season",
            "event",
            "event_slug",
            "checkpoint",
            "driver",
            "driver_key",
            "team",
            "team_key",
        ]
    )


def target_key_frame(targets: pd.DataFrame) -> pd.DataFrame:
    frame = targets.copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "event",
                "event_slug",
                "checkpoint",
                "driver",
                "driver_key",
                "team",
                "team_key",
                "quali_gap_to_pole_sec",
                "quali_position",
                "quali_best_lap_time_sec",
                "reached_q2",
                "reached_q3",
                "target_evaluable",
            ]
        )
    frame["checkpoint"] = MONITORING_CHECKPOINT
    frame["season"] = frame["season"].astype(int)
    frame["event_slug"] = frame["event_slug"].astype(str)
    frame["driver_key"] = frame["driver_key"].astype(str)
    frame["driver"] = frame["driver"].astype(str)
    frame["target_evaluable"] = (
        frame[["quali_gap_to_pole_sec", "quali_best_lap_time_sec"]].notna().all(axis=1)
    )
    return frame.reindex(
        columns=[
            "season",
            "event",
            "event_slug",
            "checkpoint",
            "driver",
            "driver_key",
            "team",
            "team_key",
            "quali_gap_to_pole_sec",
            "quali_position",
            "quali_best_lap_time_sec",
            "reached_q2",
            "reached_q3",
            "target_evaluable",
        ]
    )


def identifier_conflicts(features: pd.DataFrame, targets: pd.DataFrame) -> list[dict[str, object]]:
    if features.empty or targets.empty:
        return []
    merged = features.merge(
        targets,
        on=["season", "event_slug", "checkpoint", "driver_key"],
        suffixes=("_feature", "_target"),
        how="inner",
    )
    conflicts: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        if str(row.get("driver_feature")) != str(row.get("driver_target")):
            conflicts.append(
                {
                    "driver_key": row.get("driver_key"),
                    "feature_driver": row.get("driver_feature"),
                    "target_driver": row.get("driver_target"),
                }
            )
        feature_team = row.get("team_key_feature")
        target_team = row.get("team_key_target")
        if (
            pd.notna(feature_team)
            and pd.notna(target_team)
            and str(feature_team) != str(target_team)
        ):
            conflicts.append(
                {
                    "driver_key": row.get("driver_key"),
                    "feature_team_key": feature_team,
                    "target_team_key": target_team,
                }
            )
    return conflicts


def validate_feature_target_alignment(features: pd.DataFrame, targets: pd.DataFrame) -> None:
    feature_keys = feature_join_keys(features)
    target_keys = target_join_keys(targets)
    extra = sorted(target_keys - feature_keys)
    if extra:
        raise ValueError(f"Feature/target identifier alignment failed; extra_targets={extra}")
    conflicts = identifier_conflicts(feature_key_frame(features), target_key_frame(targets))
    if conflicts:
        raise ValueError(f"Feature/target identifier conflicts: {conflicts}")


def feature_join_keys(features: pd.DataFrame) -> set[tuple[object, ...]]:
    return set(
        features[["season", "event_slug", "checkpoint", "driver_key"]]
        .assign(season=lambda frame: frame["season"].astype(int))
        .itertuples(index=False, name=None)
    )


def target_join_keys(targets: pd.DataFrame) -> set[tuple[object, ...]]:
    frame = targets.copy()
    frame["checkpoint"] = MONITORING_CHECKPOINT
    return set(
        frame[["season", "event_slug", "checkpoint", "driver_key"]]
        .assign(season=lambda data: data["season"].astype(int))
        .itertuples(index=False, name=None)
    )


def load_valid_feature_artifact(
    config: DataConfig,
    season: int,
    event: str,
) -> tuple[pd.DataFrame, bool, str]:
    path = feature_artifact_path(config, season, event)
    if not path.is_file():
        return pd.DataFrame(), False, "feature_artifact_missing"
    frame = pd.read_parquet(path)
    forbidden = forbidden_target_columns(frame)
    if forbidden:
        return frame, False, "forbidden_target_columns_present"
    try:
        validate_feature_grain(frame)
    except ValueError as exc:
        return frame, False, str(exc)
    manifest = read_json_if_exists(event_manifest_path(config, season, event)) or {}
    if manifest.get("feature_artifact_fingerprint") != artifact_fingerprint(path):
        return frame, False, "feature_artifact_fingerprint_mismatch"
    return frame, True, ""


def validate_target_artifact(
    config: DataConfig,
    season: int,
    event: str,
) -> tuple[bool, str]:
    path = target_artifact_path(config, season, event)
    if not path.is_file():
        return False, "target_artifact_missing"
    targets = pd.read_parquet(path)
    missing = sorted(set(TARGET_ARTIFACT_COLUMNS) - set(targets.columns))
    if missing:
        return False, "target_artifact_missing_columns"
    manifest = read_json_if_exists(event_manifest_path(config, season, event)) or {}
    if manifest.get("target_artifact_fingerprint") != artifact_fingerprint(path):
        return False, "target_artifact_fingerprint_mismatch"
    coverage_valid, coverage_reason = validate_target_coverage_artifact(config, season, event)
    if not coverage_valid:
        return False, coverage_reason
    features, feature_valid, reason = load_valid_feature_artifact(config, season, event)
    if not feature_valid:
        return False, reason
    try:
        validate_feature_target_alignment(features, targets)
    except ValueError as exc:
        return False, str(exc)
    if targets.empty:
        return False, "target_coverage_empty"
    return True, ""


def validate_target_coverage_artifact(
    config: DataConfig,
    season: int,
    event: str,
) -> tuple[bool, str]:
    path = target_coverage_path(config, season, event)
    if not path.is_file():
        return False, "target_coverage_artifact_missing"
    coverage = pd.read_csv(path)
    missing = sorted(set(TARGET_COVERAGE_COLUMNS) - set(coverage.columns))
    if missing:
        return False, "target_coverage_missing_columns"
    manifest = read_json_if_exists(event_manifest_path(config, season, event)) or {}
    if manifest.get("target_coverage_artifact_fingerprint") != artifact_fingerprint(path):
        return False, "target_coverage_artifact_fingerprint_mismatch"
    features, feature_valid, reason = load_valid_feature_artifact(config, season, event)
    if not feature_valid:
        return False, reason
    if len(coverage) != len(features):
        return False, "target_coverage_feature_row_count_mismatch"
    if not coverage["feature_row_present"].astype(bool).all():
        return False, "target_coverage_missing_feature_rows"
    status = str(manifest.get("target_coverage_status", ""))
    if status not in {"target_coverage_complete", "target_coverage_partial"}:
        return False, status or "target_coverage_invalid"
    if int(coverage["target_evaluable"].astype(bool).sum()) <= 0:
        return False, "target_coverage_empty"
    return True, ""


def update_registered_target_state(config: DataConfig, season: int, event: str) -> None:
    registry_path = config.metrics_output_dir / "prospective_monitoring_event_registry.csv"
    registry = read_registry(config.metrics_output_dir)
    if registry.empty:
        return
    slug = slugify(event)
    mask = registry["monitor_season"].astype(int).eq(int(season)) & registry["event_slug"].astype(
        str
    ).eq(slug)
    if not mask.any():
        return
    target_path = target_artifact_path(config, season, event)
    target_valid, target_reason = validate_target_artifact(config, season, event)
    coverage_summary = target_coverage_summary(config, season, event)
    registry.loc[mask, "target_artifact_path"] = portable_path(target_path, config.project_root)
    registry.loc[mask, "target_artifact_present"] = target_path.is_file()
    registry.loc[mask, "target_artifact_valid"] = target_valid
    registry.loc[mask, "settleable"] = target_valid
    registry.loc[mask, "settlement_status"] = "settleable" if target_valid else "targets_missing"
    apply_coverage_summary_to_registry(registry, mask, coverage_summary)
    forecasts = read_parquet(config.metrics_output_dir / "prospective_monitoring_forecasts.parquet")
    settlements = read_parquet(
        config.metrics_output_dir / "prospective_monitoring_settlements.parquet"
    )
    registry.loc[mask, "onboarding_status"] = registry.loc[mask].apply(
        lambda row: lifecycle_state(row, forecasts, settlements),
        axis=1,
    )
    if not target_valid:
        registry.loc[mask, "onboarding_blocking_reason"] = target_reason
    registry.to_csv(registry_path, index=False)


def upsert_registry_row(
    registry: pd.DataFrame,
    *,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    features: pd.DataFrame,
    event_order: int,
    feature_valid: bool,
    feature_blocker: str,
    target_present: bool,
    target_valid: bool,
    forecasts: pd.DataFrame,
    settlements: pd.DataFrame,
) -> pd.DataFrame:
    registry = ensure_registry_columns(registry)
    season = int(manifest["season"])
    slug = str(manifest["event_slug"])
    target_path = Path(str(manifest.get("target_artifact_path") or ""))
    coverage_summary = manifest_coverage_summary(manifest)
    row = {
        "protocol_name": protocol["protocol_name"],
        "monitor_season": season,
        "event_order": int(event_order),
        "event": manifest["event"],
        "event_slug": slug,
        "checkpoint": protocol["checkpoint"],
        "feature_rows_available": int(len(features)) if feature_valid else 0,
        "driver_rows_available": int(features["driver"].nunique()) if feature_valid else 0,
        "qualifying_targets_available": bool(target_valid),
        "forecast_status": "forecastable" if feature_valid else "unavailable",
        "settlement_status": "settleable" if target_valid else "targets_missing",
        "eligibility_evidence_status": "pending_prior_settlements",
        "live_policy_status": "static_or_guarded_reference_frozen",
        "shadow_candidate_status": "diagnostic_only",
        "readiness_blocking_reason": "" if feature_valid else feature_blocker,
        "feature_artifact_path": manifest.get("feature_artifact_path"),
        "feature_artifact_fingerprint": manifest.get("feature_artifact_fingerprint"),
        "feature_artifact_valid": bool(feature_valid),
        "target_artifact_path": (
            portable_path(target_path, Path.cwd()) if target_path.as_posix() else ""
        ),
        "target_artifact_present": bool(target_present),
        "target_artifact_valid": bool(target_valid),
        "prequalification_ready": bool(feature_valid),
        "forecastable": bool(feature_valid),
        "settleable": bool(target_valid),
        "onboarding_blocking_reason": "" if feature_valid else feature_blocker,
        "feature_driver_count": coverage_summary.get(
            "feature_driver_count",
            int(features["driver"].nunique()) if feature_valid else 0,
        ),
        "target_driver_count": coverage_summary.get("target_driver_count", 0),
        "evaluable_driver_count": coverage_summary.get("evaluable_driver_count", 0),
        "non_evaluable_driver_count": coverage_summary.get("non_evaluable_driver_count", 0),
        "target_coverage_rate": coverage_summary.get("target_coverage_rate", 0.0),
        "partial_target_coverage": coverage_summary.get("partial_target_coverage", False),
        "target_coverage_status": coverage_summary.get(
            "target_coverage_status",
            "target_not_available",
        ),
        "settlement_metric_status": coverage_summary.get(
            "settlement_metric_status",
            "not_scorable",
        ),
    }
    temp = pd.DataFrame([row])
    row["onboarding_status"] = lifecycle_state(temp.iloc[0], forecasts, settlements)
    mask = (
        registry.get("protocol_name", pd.Series(dtype=str))
        .astype(str)
        .eq(str(protocol["protocol_name"]))
        & registry.get("monitor_season", pd.Series(dtype=int)).astype(str).eq(str(season))
        & registry.get("event_slug", pd.Series(dtype=str)).astype(str).eq(slug)
    )
    registry = registry.loc[~mask].copy() if not registry.empty else registry
    registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True, sort=False)
    registry = registry.sort_values(["monitor_season", "event_order"], kind="stable")
    return registry.reindex(columns=registry_columns())


def lifecycle_state(row: pd.Series, forecasts: pd.DataFrame, settlements: pd.DataFrame) -> str:
    if not bool(row.get("feature_artifact_valid", False)):
        return "invalid_or_blocked"
    protocol = str(row.get("protocol_name"))
    slug = str(row.get("event_slug"))
    forecasted = artifact_has_event(forecasts, protocol, slug)
    settled = artifact_has_event(settlements, protocol, slug)
    target_valid = bool(row.get("target_artifact_valid", False))
    if settled:
        return "settled"
    if forecasted and target_valid:
        return "forecasted_settleable"
    if forecasted:
        return "forecasted_unsettled"
    return "registered_not_forecasted"


def manifest_coverage_summary(manifest: dict[str, Any]) -> dict[str, object]:
    rate = manifest.get("feature_target_coverage_rate", manifest.get("target_coverage_rate", 0.0))
    return {
        "feature_driver_count": int(manifest.get("feature_driver_count", 0) or 0),
        "target_driver_count": int(manifest.get("target_driver_count", 0) or 0),
        "evaluable_driver_count": int(manifest.get("evaluable_driver_count", 0) or 0),
        "non_evaluable_driver_count": int(manifest.get("non_evaluable_driver_count", 0) or 0),
        "target_coverage_rate": float(rate or 0.0),
        "partial_target_coverage": bool(manifest.get("partial_target_coverage", False)),
        "target_coverage_status": str(
            manifest.get("target_coverage_status", "target_not_available")
        ),
        "settlement_metric_status": str(manifest.get("settlement_metric_status", "not_scorable")),
    }


def target_coverage_summary(config: DataConfig, season: int, event: str) -> dict[str, object]:
    manifest = read_json_if_exists(event_manifest_path(config, season, event)) or {}
    summary = manifest_coverage_summary(manifest)
    coverage_path = target_coverage_path(config, season, event)
    if not coverage_path.is_file():
        return summary
    coverage = pd.read_csv(coverage_path)
    if coverage.empty:
        return {**summary, "target_coverage_status": "target_coverage_empty"}
    evaluable = int(coverage["target_evaluable"].astype(bool).sum())
    feature_count = int(len(coverage))
    target_count = int(coverage["qualifying_target_present"].astype(bool).sum())
    non_evaluable = feature_count - evaluable
    return {
        "feature_driver_count": feature_count,
        "target_driver_count": target_count,
        "evaluable_driver_count": evaluable,
        "non_evaluable_driver_count": non_evaluable,
        "target_coverage_rate": float(evaluable / feature_count) if feature_count else 0.0,
        "partial_target_coverage": non_evaluable > 0 and evaluable > 0,
        "target_coverage_status": target_coverage_status(
            feature_driver_count=feature_count,
            evaluable_driver_count=evaluable,
            non_evaluable_driver_count=non_evaluable,
            extra_target_count=int(manifest.get("extra_target_count", 0) or 0),
            identifier_conflict_count=int(manifest.get("identifier_conflict_count", 0) or 0),
        ),
        "settlement_metric_status": "scorable" if evaluable else "not_scorable",
    }


def apply_coverage_summary_to_registry(
    registry: pd.DataFrame,
    mask: pd.Series,
    coverage_summary: dict[str, object],
) -> None:
    for column, value in coverage_summary.items():
        if column in registry.columns:
            registry.loc[mask, column] = value


def artifact_has_event(frame: pd.DataFrame, protocol_name: str, event_slug: str) -> bool:
    return (
        not frame.empty
        and "protocol_name" in frame
        and "event_slug" in frame
        and bool(
            (
                frame["protocol_name"].astype(str).eq(protocol_name)
                & frame["event_slug"].astype(str).eq(event_slug)
            ).any()
        )
    )


def refresh_onboarding_integrity(config: DataConfig) -> dict[str, object]:
    metrics_dir = config.metrics_output_dir
    ensure_directory(metrics_dir)
    protocol = read_json_if_exists(metrics_dir / "prospective_monitoring_protocol.json") or {}
    registry = read_registry(metrics_dir)
    manifests = discover_event_manifests(config)
    by_event = build_onboarding_integrity_by_event(config, protocol, registry, manifests)
    failures = build_onboarding_failures(by_event)
    status = "invalid" if not failures.empty else "valid"
    if (
        failures.empty
        and not by_event.empty
        and by_event["integrity_status"].astype(str).eq("valid_with_partial_coverage").any()
    ):
        status = "valid_with_partial_coverage"
    summary = {
        "status": status,
        "events_checked": int(len(by_event)),
        "failure_count": int(len(failures)),
        "target_isolation_status": target_isolation_status(by_event),
        "chronological_order_status": chronological_order_status(by_event),
        "policy_recommendation": POLICY_RECOMMENDATION,
        "generated_at_utc": utc_now(),
    }
    write_json(metrics_dir / "monitoring_onboarding_integrity_summary.json", summary)
    by_event.to_csv(metrics_dir / "monitoring_onboarding_integrity_by_event.csv", index=False)
    failures.to_csv(metrics_dir / "monitoring_onboarding_integrity_failures.csv", index=False)
    readiness = build_onboarding_readiness(by_event)
    readiness.to_csv(metrics_dir / "monitoring_onboarding_readiness.csv", index=False)
    return summary


def build_onboarding_integrity_by_event(
    config: DataConfig,
    protocol: dict[str, Any],
    registry: pd.DataFrame,
    manifests: list[dict[str, Any]],
) -> pd.DataFrame:
    columns = onboarding_integrity_columns()
    rows: list[dict[str, object]] = []
    for manifest in manifests:
        season = int(manifest.get("season", 0))
        event = str(manifest.get("event", ""))
        slug = str(manifest.get("event_slug", slugify(event)))
        feature_path = feature_artifact_path(config, season, event)
        target_path = target_artifact_path(config, season, event)
        coverage_path = target_coverage_path(config, season, event)
        features = pd.read_parquet(feature_path) if feature_path.is_file() else pd.DataFrame()
        targets = pd.read_parquet(target_path) if target_path.is_file() else pd.DataFrame()
        feature_valid, feature_reason = (
            load_valid_feature_artifact(config, season, event)[1:]
            if feature_path.is_file()
            else (False, "feature_artifact_missing")
        )
        target_valid, _ = (
            validate_target_artifact(config, season, event)
            if target_path.is_file()
            else (False, "target_artifact_missing")
        )
        registered = registry[registry.get("event_slug", pd.Series(dtype=str)).astype(str).eq(slug)]
        protocol_valid = True
        if not registered.empty and protocol:
            protocol_valid = protocol.get("protocol_fingerprint") == protocol_fingerprint(protocol)
        feature_fingerprint_valid = feature_path.is_file() and manifest.get(
            "feature_artifact_fingerprint"
        ) == artifact_fingerprint(feature_path)
        target_fingerprint_valid = not target_path.is_file() or manifest.get(
            "target_artifact_fingerprint"
        ) == artifact_fingerprint(target_path)
        coverage_fingerprint_valid = not coverage_path.is_file() or manifest.get(
            "target_coverage_artifact_fingerprint"
        ) == artifact_fingerprint(coverage_path)
        event_alignment = True
        driver_alignment = True
        extra_targets_absent = True
        if target_path.is_file() and not features.empty and not targets.empty:
            try:
                validate_feature_target_alignment(features, targets)
            except ValueError:
                event_alignment = False
                driver_alignment = False
                extra_targets_absent = False
        coverage_summary = target_coverage_summary(config, season, event)
        partial_documented = (
            not bool(coverage_summary.get("partial_target_coverage", False))
            or coverage_path.is_file()
        )
        coverage_rate_recorded = (
            not target_path.is_file() or coverage_summary.get("target_coverage_rate") is not None
        )
        row = {
            "season": season,
            "event": event,
            "event_slug": slug,
            "fp3_safe_feature_artifact_exists": feature_path.is_file(),
            "forbidden_target_columns_absent": not forbidden_target_columns(features),
            "qualifying_target_artifact_separate": target_path != feature_path,
            "feature_artifact_unchanged_after_target_creation": bool(
                manifest.get("feature_unchanged_after_target_creation", True)
            ),
            "feature_artifact_fingerprint_valid": feature_fingerprint_valid,
            "target_artifact_fingerprint_valid": target_fingerprint_valid,
            "target_coverage_artifact_fingerprint_valid": coverage_fingerprint_valid,
            "event_identifier_alignment_valid": event_alignment,
            "driver_identifier_alignment_valid": driver_alignment,
            "chronological_event_order_valid": (
                manifest.get("chronological_event_order_status") == "valid"
                or pd.notna(row_value(registered, "event_order", pd.NA))
            ),
            "protocol_fingerprint_valid_when_registered": protocol_valid,
            "forecastable_without_target_access": feature_valid
            and not forbidden_target_columns(features),
            "settleable_only_after_target_artifact_exists": (
                not bool(row_value(registered, "settleable", False)) or target_path.is_file()
            ),
            "feature_artifact_valid": feature_valid,
            "target_artifact_present": target_path.is_file(),
            "target_artifact_valid": target_valid,
            "partial_target_coverage_documented": partial_documented,
            "valid_target_rows_exactly_aligned": event_alignment and driver_alignment,
            "extra_targets_absent_or_explained": extra_targets_absent,
            "coverage_rate_recorded": coverage_rate_recorded,
            "forecast_artifact_unchanged_after_target_creation": bool(
                manifest.get("forecast_artifact_unchanged_after_target_creation", True)
            ),
            "target_coverage_status": coverage_summary.get(
                "target_coverage_status",
                "target_not_available",
            ),
            "target_coverage_rate": coverage_summary.get("target_coverage_rate", 0.0),
            "evaluable_driver_count": coverage_summary.get("evaluable_driver_count", 0),
            "non_evaluable_driver_count": coverage_summary.get("non_evaluable_driver_count", 0),
            "blocking_reason": feature_reason,
        }
        checks = [column for column in columns[3:-1] if column not in {"blocking_reason"}]
        valid = all(
            bool(row.get(col))
            for col in checks
            if col
            not in {
                "target_artifact_present",
                "target_artifact_valid",
                "target_coverage_rate",
                "evaluable_driver_count",
                "non_evaluable_driver_count",
            }
        )
        if valid and row["target_coverage_status"] == "target_coverage_partial":
            row["integrity_status"] = "valid_with_partial_coverage"
        else:
            row["integrity_status"] = "valid" if valid else "invalid"
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def build_onboarding_failures(by_event: pd.DataFrame) -> pd.DataFrame:
    columns = ["season", "event_slug", "condition", "status", "details"]
    rows = []
    if by_event.empty:
        return pd.DataFrame(columns=columns)
    checked = [column for column in onboarding_integrity_columns()[3:-2]]
    for _, row in by_event.iterrows():
        if str(row.get("integrity_status")) == "valid_with_partial_coverage":
            continue
        for column in checked:
            if column in {
                "target_artifact_present",
                "target_artifact_valid",
                "target_coverage_rate",
                "evaluable_driver_count",
                "non_evaluable_driver_count",
            }:
                continue
            if not bool(row.get(column, True)):
                rows.append(
                    {
                        "season": row.get("season"),
                        "event_slug": row.get("event_slug"),
                        "condition": column,
                        "status": "failed",
                        "details": row.get("blocking_reason", ""),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def build_onboarding_readiness(by_event: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "season",
        "event",
        "event_slug",
        "prepared",
        "registered",
        "forecastable",
        "target_artifact_present",
        "settleable",
        "integrity_status",
        "next_action",
    ]
    if by_event.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, row in by_event.iterrows():
        prepared = bool(row["feature_artifact_valid"])
        target = bool(row["target_artifact_valid"])
        integrity_status = str(row["integrity_status"])
        rows.append(
            {
                "season": row["season"],
                "event": row["event"],
                "event_slug": row["event_slug"],
                "prepared": prepared,
                "registered": bool(row["protocol_fingerprint_valid_when_registered"]),
                "forecastable": prepared and bool(row["chronological_event_order_valid"]),
                "target_artifact_present": bool(row["target_artifact_present"]),
                "settleable": prepared and target,
                "integrity_status": integrity_status,
                "next_action": next_action(prepared, target, integrity_status),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_readiness_by_event(
    config: DataConfig,
    protocol: dict[str, Any],
    registry: pd.DataFrame,
    manifests: list[dict[str, Any]],
) -> pd.DataFrame:
    columns = [
        "protocol_name",
        "monitor_season",
        "event_order",
        "event",
        "event_slug",
        "prepared",
        "registered",
        "forecastable",
        "target_artifact_present",
        "settleable",
        "onboarding_status",
        "readiness_blocking_reason",
        "feature_driver_count",
        "target_driver_count",
        "evaluable_driver_count",
        "non_evaluable_driver_count",
        "target_coverage_rate",
        "partial_target_coverage",
        "target_coverage_status",
        "settlement_metric_status",
    ]
    rows = []
    for manifest in manifests:
        season = int(manifest["season"])
        event = str(manifest["event"])
        slug = str(manifest["event_slug"])
        registered = registry[registry.get("event_slug", pd.Series(dtype=str)).astype(str).eq(slug)]
        feature_valid = load_valid_feature_artifact(config, season, event)[1]
        target_valid = (
            validate_target_artifact(config, season, event)[0]
            if target_artifact_path(config, season, event).is_file()
            else False
        )
        row = registered.iloc[0].to_dict() if not registered.empty else {}
        rows.append(
            {
                "protocol_name": protocol.get("protocol_name"),
                "monitor_season": season,
                "event_order": row.get("event_order", manifest.get("event_order")),
                "event": event,
                "event_slug": slug,
                "prepared": bool(feature_valid),
                "registered": bool(not registered.empty),
                "forecastable": bool(row.get("forecastable", False)),
                "target_artifact_present": target_artifact_path(config, season, event).is_file(),
                "settleable": bool(row.get("settleable", target_valid)),
                "onboarding_status": row.get(
                    "onboarding_status",
                    "prepared_not_registered" if feature_valid else "invalid_or_blocked",
                ),
                "readiness_blocking_reason": row.get(
                    "onboarding_blocking_reason",
                    ";".join(manifest.get("readiness_blockers", [])),
                ),
                **target_coverage_summary(config, season, event),
            }
        )
    if not rows and protocol:
        rows.append(
            {
                "protocol_name": protocol.get("protocol_name"),
                "monitor_season": protocol.get("monitor_season"),
                "event_order": pd.NA,
                "event": pd.NA,
                "event_slug": pd.NA,
                "prepared": False,
                "registered": False,
                "forecastable": False,
                "target_artifact_present": False,
                "settleable": False,
                "onboarding_status": "no_prepared_events",
                "readiness_blocking_reason": "monitor_season_event_rows",
                "feature_driver_count": 0,
                "target_driver_count": 0,
                "evaluable_driver_count": 0,
                "non_evaluable_driver_count": 0,
                "target_coverage_rate": 0.0,
                "partial_target_coverage": False,
                "target_coverage_status": "target_not_available",
                "settlement_metric_status": "not_scorable",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_missing_inputs(
    protocol: dict[str, Any],
    registry: pd.DataFrame,
    by_event: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["requirement", "status", "blocking", "details"]
    rows = []
    if not protocol:
        rows.append(
            {
                "requirement": "prospective_monitoring_protocol",
                "status": "missing",
                "blocking": True,
                "details": "reports/metrics/prospective_monitoring_protocol.json",
            }
        )
    if by_event.empty or not bool(by_event["prepared"].any()):
        rows.append(
            {
                "requirement": "prepared_monitoring_event",
                "status": "unavailable",
                "blocking": True,
                "details": protocol.get("monitor_season"),
            }
        )
    if registry.empty:
        rows.append(
            {
                "requirement": "registered_monitoring_event",
                "status": "unavailable",
                "blocking": True,
                "details": protocol.get("protocol_name"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def available_next_actions(by_event: pd.DataFrame) -> list[str]:
    if by_event.empty or not bool(by_event["prepared"].any()):
        return ["prepare_event_from_local_fp1_fp2_fp3"]
    if not bool(by_event["registered"].any()):
        return ["register_prepared_event"]
    if bool(by_event["forecastable"].any()) and not bool(by_event["target_artifact_present"].any()):
        return ["run_forecast_before_qualifying", "add_targets_after_qualifying"]
    if bool(by_event["settleable"].any()):
        return ["run_settlement"]
    return ["inspect_readiness_blockers"]


def readiness_status(protocol: dict[str, Any], by_event: pd.DataFrame) -> str:
    if not protocol:
        return "missing_protocol"
    if by_event.empty or not bool(by_event["prepared"].any()):
        return "not_ready"
    if bool(by_event["forecastable"].any()):
        return "ready_for_forecast"
    return "blocked"


def generate_readiness_figures(
    *,
    figures_dir: Path,
    readiness_by_event: pd.DataFrame,
    integrity_by_event: pd.DataFrame,
) -> tuple[list[Path], list[str]]:
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
            "monitoring_data_readiness_event_status.png",
            lambda path: plot_status(plt, readiness_by_event, path),
        ),
        (
            "monitoring_data_readiness_session_coverage.png",
            lambda path: plot_session_coverage(plt, readiness_by_event, path),
        ),
        (
            "monitoring_data_readiness_target_isolation.png",
            lambda path: plot_target_isolation(plt, integrity_by_event, path),
        ),
        (
            "monitoring_data_readiness_forecast_settlement_flow.png",
            lambda path: plot_flow(plt, readiness_by_event, path),
        ),
    )
    paths: list[Path] = []
    issues: list[str] = []
    for filename, writer in specs:
        path = figures_dir / filename
        try:
            if writer(path):
                paths.append(path)
        except Exception as exc:
            issues.append(f"{filename}: {exc}")
    return paths, issues


def plot_status(plt: Any, frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty:
        return plot_placeholder(plt, path, "Monitoring data readiness unavailable")
    counts = frame["onboarding_status"].astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values)
    ax.set_title("Monitoring onboarding event status")
    ax.set_ylabel("Events")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_session_coverage(plt: Any, frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty:
        return plot_placeholder(plt, path, "No prepared FP session coverage yet")
    counts = frame[["prepared", "registered", "forecastable"]].sum()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values)
    ax.set_title("Prepared and registered monitoring events")
    ax.set_ylabel("Events")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_target_isolation(plt: Any, frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty:
        return plot_placeholder(plt, path, "Target isolation not yet evaluated")
    counts = frame["forbidden_target_columns_absent"].astype(bool).value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["valid" if value else "invalid" for value in counts.index], counts.values)
    ax.set_title("FP3-safe target isolation")
    ax.set_ylabel("Events")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_flow(plt: Any, frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty:
        return plot_placeholder(plt, path, "Forecast and settlement flow unavailable")
    counts = frame[["forecastable", "target_artifact_present", "settleable"]].sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values)
    ax.set_title("Monitoring forecast and settlement readiness")
    ax.set_ylabel("Events")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_placeholder(plt: Any, path: Path, title: str) -> bool:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, title, ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def monitoring_event_dir(config: DataConfig, season: int, event: str) -> Path:
    return ensure_directory(
        config.project_root / "data/processed/monitoring" / str(season) / slugify(event)
    )


def feature_artifact_path(config: DataConfig, season: int, event: str) -> Path:
    return monitoring_event_dir(config, season, event) / "monitoring_fp3_features.parquet"


def target_artifact_path(config: DataConfig, season: int, event: str) -> Path:
    return monitoring_event_dir(config, season, event) / "monitoring_qualifying_targets.parquet"


def target_coverage_path(config: DataConfig, season: int, event: str) -> Path:
    return monitoring_event_dir(config, season, event) / "monitoring_target_coverage.csv"


def event_manifest_path(config: DataConfig, season: int, event: str) -> Path:
    return monitoring_event_dir(config, season, event) / "monitoring_event_manifest.json"


def base_manifest(config: DataConfig, season: int, event: str) -> dict[str, object]:
    return {
        "season": int(season),
        "event": event,
        "event_slug": slugify(event),
        "monitoring_event_dir": portable_path(
            monitoring_event_dir(config, season, event),
            config.project_root,
        ),
        "created_at_utc": utc_now(),
    }


def load_event_manifest(config: DataConfig, season: int, event: str) -> dict[str, Any]:
    path = event_manifest_path(config, season, event)
    if not path.is_file():
        raise FileNotFoundError(f"Monitoring event manifest is missing: {path}")
    return read_json(path)


def discover_event_manifests(config: DataConfig) -> list[dict[str, Any]]:
    root = config.project_root / "data/processed/monitoring"
    if not root.is_dir():
        return []
    manifests = []
    for path in sorted(root.glob("*/*/monitoring_event_manifest.json")):
        manifests.append(read_json(path))
    return manifests


def resolve_event_order(
    *,
    event_order: int | None,
    manifest: dict[str, Any],
    features: pd.DataFrame,
    registry: pd.DataFrame,
) -> int:
    if event_order is not None:
        return int(event_order)
    if manifest.get("event_order") is not None:
        return int(manifest["event_order"])
    if "event_order" in features and features["event_order"].notna().any():
        return int(pd.to_numeric(features["event_order"], errors="coerce").dropna().min())
    if not registry.empty and "event_order" in registry:
        return int(pd.to_numeric(registry["event_order"], errors="coerce").max()) + 1
    raise ValueError(
        "Chronological event order is unavailable; supply --event-order or include event_order "
        "in the prepared feature artifact."
    )


def chronological_status(features: pd.DataFrame) -> str:
    if "event_order" not in features:
        return "unknown"
    numeric = pd.to_numeric(features["event_order"], errors="coerce").dropna()
    return "valid" if not numeric.empty else "unknown"


def target_status(config: DataConfig, season: int, event: str) -> str:
    return "available" if target_artifact_path(config, season, event).is_file() else "not_available"


def forbidden_target_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in frame.columns
        if column in FORBIDDEN_TARGET_COLUMNS
        or column.lower().startswith("quali_")
        or "target" in column.lower()
    )


def registry_columns() -> list[str]:
    base = [
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
    ]
    return [*base, *REGISTRY_ONBOARDING_COLUMNS]


def ensure_registry_columns(registry: pd.DataFrame) -> pd.DataFrame:
    result = registry.copy()
    for column in registry_columns():
        if column not in result:
            result[column] = pd.NA
    return result.reindex(columns=registry_columns())


def read_registry(metrics_dir: Path) -> pd.DataFrame:
    path = metrics_dir / "prospective_monitoring_event_registry.csv"
    if not path.is_file():
        return pd.DataFrame(columns=registry_columns())
    return ensure_registry_columns(pd.read_csv(path))


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.is_file() else None


def write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(json_clean(payload), output_file, indent=2, allow_nan=False)
        output_file.write("\n")


def artifact_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def stable_signature(payload: object) -> str:
    encoded = json.dumps(json_clean(payload), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def protocol_fingerprint(payload: dict[str, Any]) -> str:
    frozen = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at_utc", "protocol_fingerprint"}
    }
    return stable_signature(frozen)


def load_protocol(metrics_dir: Path, protocol_name: str) -> dict[str, Any]:
    path = metrics_dir / "prospective_monitoring_protocol.json"
    if not path.is_file():
        raise FileNotFoundError(f"Monitoring protocol not found: {path}")
    protocol = read_json(path)
    if str(protocol.get("protocol_name")) != protocol_name:
        raise ValueError(f"Monitoring protocol mismatch: expected {protocol_name}")
    return protocol


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_clean(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def relative_report_path(path: Path) -> str:
    parts = path.parts
    if "reports" in parts:
        return Path(*parts[parts.index("reports") :]).as_posix()
    return path.as_posix()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def row_value(frame: pd.DataFrame, column: str, default: object) -> object:
    if frame.empty or column not in frame:
        return default
    values = frame[column].dropna()
    return values.iloc[0] if not values.empty else default


def next_action(prepared: bool, target_valid: bool, integrity_status: str) -> str:
    if integrity_status != "valid":
        return "fix_integrity_failure"
    if not prepared:
        return "prepare_event"
    if not target_valid:
        return "forecast_or_add_targets_when_available"
    return "settle_forecast_when_forecast_exists"


def target_isolation_status(by_event: pd.DataFrame) -> str:
    if by_event.empty:
        return "not_evaluated"
    if (
        "target_coverage_status" in by_event
        and by_event["target_coverage_status"].astype(str).eq("target_coverage_partial").any()
    ):
        return "valid_with_partial_coverage"
    ok = (
        by_event["forbidden_target_columns_absent"].astype(bool).all()
        and by_event["qualifying_target_artifact_separate"].astype(bool).all()
    )
    return "valid" if ok else "invalid"


def aggregate_coverage_status(by_event: pd.DataFrame) -> str:
    if by_event.empty or "target_coverage_status" not in by_event:
        return "target_not_available"
    statuses = set(by_event["target_coverage_status"].dropna().astype(str))
    if "target_coverage_invalid" in statuses:
        return "target_coverage_invalid"
    if "target_coverage_partial" in statuses:
        return "target_coverage_partial"
    if "target_coverage_complete" in statuses:
        return "target_coverage_complete"
    if "target_coverage_empty" in statuses:
        return "target_coverage_empty"
    return "target_not_available"


def aggregate_coverage_rate(by_event: pd.DataFrame) -> float | None:
    if by_event.empty or "feature_driver_count" not in by_event:
        return None
    feature_count = pd.to_numeric(by_event["feature_driver_count"], errors="coerce").fillna(0).sum()
    evaluable_count = (
        pd.to_numeric(by_event["evaluable_driver_count"], errors="coerce").fillna(0).sum()
        if "evaluable_driver_count" in by_event
        else 0
    )
    if feature_count <= 0:
        return None
    return float(evaluable_count / feature_count)


def aggregate_metric_status(by_event: pd.DataFrame) -> str:
    if by_event.empty or "settlement_metric_status" not in by_event:
        return "not_scorable"
    statuses = set(by_event["settlement_metric_status"].dropna().astype(str))
    if "scorable" in statuses:
        return "scorable"
    return "not_scorable"


def chronological_order_status(by_event: pd.DataFrame) -> str:
    if by_event.empty:
        return "not_evaluated"
    return (
        "valid"
        if by_event["chronological_event_order_valid"].astype(bool).all()
        else "invalid_or_missing"
    )


def onboarding_integrity_columns() -> list[str]:
    return [
        "season",
        "event",
        "event_slug",
        "fp3_safe_feature_artifact_exists",
        "forbidden_target_columns_absent",
        "qualifying_target_artifact_separate",
        "feature_artifact_unchanged_after_target_creation",
        "feature_artifact_fingerprint_valid",
        "target_artifact_fingerprint_valid",
        "target_coverage_artifact_fingerprint_valid",
        "event_identifier_alignment_valid",
        "driver_identifier_alignment_valid",
        "chronological_event_order_valid",
        "protocol_fingerprint_valid_when_registered",
        "forecastable_without_target_access",
        "settleable_only_after_target_artifact_exists",
        "feature_artifact_valid",
        "target_artifact_present",
        "target_artifact_valid",
        "partial_target_coverage_documented",
        "valid_target_rows_exactly_aligned",
        "extra_targets_absent_or_explained",
        "coverage_rate_recorded",
        "forecast_artifact_unchanged_after_target_creation",
        "target_coverage_status",
        "target_coverage_rate",
        "evaluable_driver_count",
        "non_evaluable_driver_count",
        "blocking_reason",
        "integrity_status",
    ]
