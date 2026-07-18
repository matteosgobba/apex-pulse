"""One-command monitored-event operational workflows."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig, FeatureConfig, ModelConfig
from f1_prediction.dashboard.export import export_dashboard_artifacts
from f1_prediction.data.ingest import ingest_event
from f1_prediction.data.monitoring_onboarding import (
    add_monitoring_targets,
    feature_artifact_path,
    load_valid_feature_artifact,
    prepare_monitoring_event,
    read_csv,
    read_json_if_exists,
    read_parquet,
    register_monitoring_event,
    target_artifact_path,
    utc_now,
    validate_target_artifact,
    validate_target_raw_identity,
    write_json,
)
from f1_prediction.data.raw_session_identity import (
    IDENTITY_VERIFIED,
    create_raw_session_identity_validation_report,
    validate_raw_session_identity,
)
from f1_prediction.modeling.monitoring_data_integrity_audit import (
    create_monitoring_data_integrity_audit,
)
from f1_prediction.modeling.prospective_monitoring import (
    PREFLIGHT_ALREADY_FORECASTED,
    PREFLIGHT_READY,
    create_prospective_monitoring_forecast,
    create_prospective_monitoring_preflight,
    create_prospective_monitoring_report,
    create_prospective_monitoring_settlement,
    expected_forecast_hash,
    forecast_snapshot_hash,
    synthetic_rehearsal_event_slug,
)
from f1_prediction.modeling.qualifying_target_parity_audit import (
    create_qualifying_target_parity_audit,
)
from f1_prediction.utils.paths import ensure_directory, slugify

DEFAULT_PROTOCOL_NAME = "season_2026_v1"
LEGACY_EVENT_SLUGS = {"australia", "great-britain"}
LIVE_POLICY_ROLE = "observed_live_policy"

BEFORE_WORKFLOW = "monitoring_before_qualifying"
AFTER_WORKFLOW = "monitoring_after_qualifying"

BEFORE_STAGES: tuple[str, ...] = (
    "practice_ingested",
    "event_prepared",
    "event_registered",
    "preflight_ready",
    "forecast_created",
    "dashboard_exported",
)

AFTER_STAGES: tuple[str, ...] = (
    "qualifying_ingested",
    "raw_q_identity_verified",
    "targets_added",
    "settled",
    "parity_audit_passed",
    "integrity_audit_passed",
    "dashboard_exported",
)


@dataclass(frozen=True)
class MonitoringWorkflowSummary:
    """Summary paths and status for one monitored-event operational workflow."""

    status: str
    workflow: str
    summary_path: Path
    stages_path: Path
    completed: bool
    blocking_failure_count: int
    warning_count: int
    event: str
    event_slug: str
    dashboard_current_event: str | None


def run_monitoring_before_qualifying(
    config: DataConfig,
    model_config: ModelConfig,
    feature_config: FeatureConfig,
    *,
    season: int,
    event: str,
    event_order: int,
    protocol_name: str = DEFAULT_PROTOCOL_NAME,
    allow_test_event: bool = False,
    progress: Callable[[str], None] | None = None,
) -> MonitoringWorkflowSummary:
    """Run the guarded before-qualifying monitoring workflow."""
    event_slug = slugify(event)
    recorder = _WorkflowRecorder(config, BEFORE_WORKFLOW, protocol_name, season, event, event_slug)
    recorder.run(
        BEFORE_STAGES,
        {
            "practice_ingested": lambda: _ingest_sessions(
                config,
                season=season,
                event=event,
                sessions=("FP1", "FP2", "FP3"),
                progress=progress,
            ),
            "event_prepared": lambda: _prepare_event(
                config,
                feature_config,
                season=season,
                event=event,
                allow_test_event=allow_test_event,
            ),
            "event_registered": lambda: _register_event(
                config,
                protocol_name=protocol_name,
                season=season,
                event=event,
                event_order=event_order,
            ),
            "preflight_ready": lambda: _run_preflight(
                config,
                protocol_name=protocol_name,
                season=season,
                event=event,
                event_slug=event_slug,
            ),
            "forecast_created": lambda: _create_or_reuse_forecast(
                config,
                model_config,
                feature_config,
                protocol_name=protocol_name,
                season=season,
                event=event,
                event_slug=event_slug,
            ),
            "dashboard_exported": lambda: _export_dashboard(
                config,
                event_slug=event_slug,
                expected_lifecycle={"forecast_available", "awaiting_qualifying_targets"},
            ),
        },
    )
    return recorder.write()


def run_monitoring_after_qualifying(
    config: DataConfig,
    *,
    season: int,
    event: str,
    protocol_name: str = DEFAULT_PROTOCOL_NAME,
    allow_test_event: bool = False,
    progress: Callable[[str], None] | None = None,
) -> MonitoringWorkflowSummary:
    """Run the guarded after-qualifying monitoring workflow."""
    event_slug = slugify(event)
    recorder = _WorkflowRecorder(config, AFTER_WORKFLOW, protocol_name, season, event, event_slug)
    recorder.run(
        AFTER_STAGES,
        {
            "qualifying_ingested": lambda: _ingest_sessions(
                config,
                season=season,
                event=event,
                sessions=("Q",),
                progress=progress,
            ),
            "raw_q_identity_verified": lambda: _verify_raw_q_identity(
                config,
                season=season,
                event=event,
                allow_test_event=allow_test_event,
            ),
            "targets_added": lambda: _add_or_reuse_targets(
                config,
                protocol_name=protocol_name,
                season=season,
                event=event,
                event_slug=event_slug,
            ),
            "settled": lambda: _settle_or_reuse_event(
                config,
                protocol_name=protocol_name,
                season=season,
                event=event,
                event_slug=event_slug,
            ),
            "parity_audit_passed": lambda: _run_event_parity_audit(
                config,
                season=season,
                event=event,
                event_slug=event_slug,
            ),
            "integrity_audit_passed": lambda: _run_event_integrity_audit(
                config,
                event_slug=event_slug,
            ),
            "dashboard_exported": lambda: _export_dashboard(
                config,
                event_slug=event_slug,
                expected_lifecycle={"settled"},
            ),
        },
    )
    return recorder.write()


class _WorkflowRecorder:
    def __init__(
        self,
        config: DataConfig,
        workflow: str,
        protocol_name: str,
        season: int,
        event: str,
        event_slug: str,
    ) -> None:
        self.config = config
        self.workflow = workflow
        self.protocol_name = protocol_name
        self.season = int(season)
        self.event = event
        self.event_slug = event_slug
        self.stage_rows: list[dict[str, Any]] = []
        self.stage_status: dict[str, str] = {}
        self.summary_values: dict[str, Any] = {}
        self.blocked = False
        self.blocking_reason = ""
        self.current_stage = "not_started"

    def run(
        self,
        stages: tuple[str, ...],
        actions: dict[str, Callable[[], dict[str, Any]]],
    ) -> None:
        for stage in stages:
            if self.blocked:
                continue
            self._run_stage(stage, actions[stage])
        for stage in stages:
            if stage not in self.stage_status:
                self._stage(
                    stage,
                    "not_started",
                    False,
                    "",
                    "",
                    (),
                    "Stage was not reached because an earlier stage blocked.",
                    _recommended_action(stage),
                )

    def _run_stage(self, stage: str, action: Callable[[], dict[str, Any]]) -> None:
        self.current_stage = stage
        started = utc_now()
        try:
            result = action()
        except (FileNotFoundError, ValueError, OSError) as exc:
            self.blocked = True
            self.blocking_reason = str(exc)
            self._stage(
                stage,
                "blocked",
                True,
                started,
                utc_now(),
                (),
                str(exc),
                _recommended_action(stage),
            )
            return
        for key, value in result.items():
            if key != "artifact_paths":
                self.summary_values[key] = value
        paths = tuple(Path(path) for path in result.get("artifact_paths", ()))
        self._stage(
            stage,
            "complete",
            False,
            started,
            utc_now(),
            paths,
            str(result.get("reason", f"{stage} completed.")),
            str(result.get("recommended_action", "Continue to the next guarded stage.")),
        )

    def _stage(
        self,
        stage: str,
        status: str,
        blocking: bool,
        started: str,
        completed: str,
        artifact_paths: tuple[Path, ...],
        reason: str,
        recommended_action: str,
    ) -> None:
        self.stage_status[stage] = status
        self.stage_rows.append(
            {
                "workflow": self.workflow,
                "protocol_name": self.protocol_name,
                "season": self.season,
                "event": self.event,
                "event_slug": self.event_slug,
                "stage": stage,
                "status": status,
                "blocking": bool(blocking),
                "started_at_utc": started,
                "completed_at_utc": completed,
                "reason": reason,
                "recommended_action": recommended_action,
                "artifact_paths": "|".join(
                    _project_relative(path, self.config.project_root) for path in artifact_paths
                ),
            }
        )

    def write(self) -> MonitoringWorkflowSummary:
        metrics_dir = ensure_directory(self.config.metrics_output_dir)
        stages = pd.DataFrame(self.stage_rows)
        blocking_count = int(
            stages["blocking"]
            .astype(bool)
            .where(stages["status"].astype(str).eq("blocked"), False)
            .sum()
        )
        warning_count = int(stages["status"].astype(str).eq("warning").sum())
        completed = bool(
            self.stage_status.get("dashboard_exported") == "complete" and blocking_count == 0
        )
        status = "pass" if completed else "blocked"
        summary = {
            "status": status,
            "workflow": self.workflow,
            "protocol_name": self.protocol_name,
            "season": self.season,
            "event": self.event,
            "event_slug": self.event_slug,
            "completed": completed,
            "blocking_failure_count": blocking_count,
            "warning_count": warning_count,
            "current_stage": self.current_stage,
            "forecast_status": self.summary_values.get("forecast_status", "not_run"),
            "raw_q_identity_status": self.summary_values.get("raw_q_identity_status", "not_run"),
            "target_status": self.summary_values.get("target_status", "not_run"),
            "settlement_status": self.summary_values.get("settlement_status", "not_run"),
            "event_specific_parity_status": self.summary_values.get(
                "event_specific_parity_status",
                "not_run",
            ),
            "event_specific_integrity_status": self.summary_values.get(
                "event_specific_integrity_status",
                "not_run",
            ),
            "dashboard_export_status": self.summary_values.get(
                "dashboard_export_status",
                "not_run",
            ),
            "dashboard_current_event": self.summary_values.get("dashboard_current_event"),
            "recommended_operator_action": (
                "Workflow completed. Continue with the next monitored-event phase."
                if completed
                else self.blocking_reason or "Resolve the blocking stage and rerun."
            ),
            "settlement_denominator": self.summary_values.get("settlement_denominator"),
            "generated_at_utc": utc_now(),
        }
        prefix = (
            "monitoring_before_qualifying"
            if self.workflow == BEFORE_WORKFLOW
            else "monitoring_after_qualifying"
        )
        summary_path = metrics_dir / f"{prefix}_summary.json"
        stages_path = metrics_dir / f"{prefix}_stages.csv"
        write_json(summary_path, summary)
        stages.to_csv(stages_path, index=False)
        return MonitoringWorkflowSummary(
            status=status,
            workflow=self.workflow,
            summary_path=summary_path,
            stages_path=stages_path,
            completed=completed,
            blocking_failure_count=blocking_count,
            warning_count=warning_count,
            event=self.event,
            event_slug=self.event_slug,
            dashboard_current_event=summary["dashboard_current_event"],
        )


def _ingest_sessions(
    config: DataConfig,
    *,
    season: int,
    event: str,
    sessions: tuple[str, ...],
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    summary = ingest_event(
        season,
        event,
        config,
        sessions=sessions,
        force=False,
        fail_fast=True,
        progress=progress,
    )
    if summary.failed_count:
        failed = next(result for result in summary.results if result.status == "failed")
        raise ValueError(f"{failed.session} ingestion failed: {failed.error_message}")
    paths = []
    for result in summary.results:
        paths.extend([result.laps_path, result.metadata_path])
    return {
        "artifact_paths": paths,
        "reason": (
            f"Ingestion complete for {', '.join(sessions)} "
            f"({summary.success_count} loaded, {summary.skipped_count} reused)."
        ),
    }


def _prepare_event(
    config: DataConfig,
    feature_config: FeatureConfig,
    *,
    season: int,
    event: str,
    allow_test_event: bool,
) -> dict[str, Any]:
    _reject_blocked_event(season, event, allow_test_event=allow_test_event)
    features, valid, reason = load_valid_feature_artifact(config, season, event)
    if valid:
        return {
            "artifact_paths": (feature_artifact_path(config, season, event),),
            "reason": (
                "Existing prepared feature artifact reused for "
                f"{features['driver'].nunique()} drivers."
            ),
        }
    if feature_artifact_path(config, season, event).is_file():
        raise ValueError(f"Existing feature artifact is invalid: {reason}")
    summary = prepare_monitoring_event(config, feature_config, season=season, event=event)
    return {
        "artifact_paths": summary.table_paths,
        "reason": "Prepared monitored FP3 feature artifact.",
    }


def _register_event(
    config: DataConfig,
    *,
    protocol_name: str,
    season: int,
    event: str,
    event_order: int,
) -> dict[str, Any]:
    existing = _registry_row(config, protocol_name, season, event)
    if existing is not None:
        _validate_existing_registry_row(existing, event_order)
        return {
            "artifact_paths": (
                config.metrics_output_dir / "prospective_monitoring_event_registry.csv",
            ),
            "reason": "Existing identical monitoring registry row reused.",
        }
    summary = register_monitoring_event(
        config,
        protocol_name=protocol_name,
        season=season,
        event=event,
        event_order=event_order,
    )
    return {
        "artifact_paths": summary.table_paths,
        "reason": "Event registered in the monitoring registry.",
    }


def _run_preflight(
    config: DataConfig,
    *,
    protocol_name: str,
    season: int,
    event: str,
    event_slug: str,
) -> dict[str, Any]:
    summary = create_prospective_monitoring_preflight(
        config,
        protocol_name=protocol_name,
        season=season,
        event=event,
    )
    payload = read_json_if_exists(summary.summary_path) or {}
    forecast_allowed = bool(payload.get("forecast_allowed", False))
    if summary.status == PREFLIGHT_ALREADY_FORECASTED and _existing_forecast_valid(
        config,
        protocol_name,
        event_slug,
    ):
        return {
            "artifact_paths": (summary.summary_path, *summary.table_paths),
            "forecast_status": "forecast_reused",
            "reason": "Preflight reports an existing valid immutable forecast.",
        }
    if summary.status != PREFLIGHT_READY or not forecast_allowed:
        raise ValueError(
            f"Preflight blocked forecast creation: status={summary.status}, "
            f"forecast_allowed={forecast_allowed}."
        )
    return {
        "artifact_paths": (summary.summary_path, *summary.table_paths),
        "forecast_status": "preflight_ready",
        "reason": "Preflight returned ready_to_forecast and forecast_allowed=true.",
    }


def _create_or_reuse_forecast(
    config: DataConfig,
    model_config: ModelConfig,
    feature_config: FeatureConfig,
    *,
    protocol_name: str,
    season: int,
    event: str,
    event_slug: str,
) -> dict[str, Any]:
    if _event_rows(config, "prospective_monitoring_settlements.parquet", protocol_name, event_slug):
        raise ValueError("Existing settlement rows block before-qualifying forecast workflow.")
    if target_artifact_path(config, season, event).is_file():
        raise ValueError("Existing target artifact blocks before-qualifying forecast workflow.")
    if _forecast_rows(config, protocol_name, event_slug).empty:
        summary = create_prospective_monitoring_forecast(
            config,
            model_config,
            feature_config,
            protocol_name=protocol_name,
            event=event,
        )
        return {
            "artifact_paths": summary.table_paths,
            "forecast_status": summary.status,
            "reason": "Immutable forecast snapshot created.",
        }
    if not _existing_forecast_valid(config, protocol_name, event_slug):
        raise ValueError("Existing forecast snapshot is not immutable or valid for this event.")
    return {
        "artifact_paths": (config.metrics_output_dir / "prospective_monitoring_forecasts.parquet",),
        "forecast_status": "forecast_reused",
        "reason": "Existing valid immutable forecast snapshot reused.",
    }


def _verify_raw_q_identity(
    config: DataConfig,
    *,
    season: int,
    event: str,
    allow_test_event: bool,
) -> dict[str, Any]:
    _reject_blocked_event(season, event, allow_test_event=allow_test_event)
    identity = validate_raw_session_identity(config, season=season, event=event, session="Q")
    _summary, _checks, _failures, _quarantine, runbook = (
        create_raw_session_identity_validation_report(
            config,
            season=season,
            event=event,
            session="Q",
        )
    )
    if (
        identity.identity_status != IDENTITY_VERIFIED
        or identity.blocking
        or identity.quarantined_for_prospective_evidence
    ):
        raise ValueError(
            "Raw Q identity validation blocks after-qualifying workflow: "
            f"{identity.identity_status}."
        )
    return {
        "artifact_paths": (identity.raw_laps_path, identity.raw_metadata_path, runbook),
        "raw_q_identity_status": identity.identity_status,
        "reason": "Raw Q identity is verified.",
    }


def _add_or_reuse_targets(
    config: DataConfig,
    *,
    protocol_name: str,
    season: int,
    event: str,
    event_slug: str,
) -> dict[str, Any]:
    if not _existing_forecast_valid(config, protocol_name, event_slug):
        raise ValueError("A valid immutable forecast is required before target onboarding.")
    target_path = target_artifact_path(config, season, event)
    if target_path.is_file():
        valid, reason = validate_target_artifact(config, season, event)
        identity_valid, identity_reason = validate_target_raw_identity(config, season, event)
        if not valid or not identity_valid:
            raise ValueError(
                "Existing target artifact is invalid or has stale provenance: "
                f"{reason or identity_reason}."
            )
        return {
            "artifact_paths": (target_path,),
            "target_status": "targets_reused",
            "reason": "Existing valid target artifact reused.",
        }
    summary = add_monitoring_targets(config, season=season, event=event)
    return {
        "artifact_paths": summary.table_paths,
        "target_status": summary.status,
        "reason": "Monitoring targets added.",
    }


def _settle_or_reuse_event(
    config: DataConfig,
    *,
    protocol_name: str,
    season: int,
    event: str,
    event_slug: str,
) -> dict[str, Any]:
    existing = _settlement_rows(config, protocol_name, event_slug)
    if not existing.empty:
        _validate_existing_settlement(config, protocol_name, season, event_slug)
        return {
            "artifact_paths": (
                config.metrics_output_dir / "prospective_monitoring_settlements.parquet",
            ),
            "settlement_status": "settlement_reused",
            "settlement_denominator": _settlement_denominator(config, event_slug),
            "reason": "Existing valid immutable settlement reused.",
        }
    valid, reason = validate_target_raw_identity(config, season, event)
    if not valid:
        raise ValueError(f"Settlement requires verified target provenance: {reason}.")
    summary = create_prospective_monitoring_settlement(
        config,
        protocol_name=protocol_name,
        event=event,
    )
    report = create_prospective_monitoring_report(config)
    return {
        "artifact_paths": (*summary.table_paths, report.summary_path, *report.table_paths),
        "settlement_status": summary.status,
        "settlement_denominator": _settlement_denominator(config, event_slug),
        "reason": "Settlement completed and monitoring report refreshed.",
    }


def _run_event_parity_audit(
    config: DataConfig,
    *,
    season: int,
    event: str,
    event_slug: str,
) -> dict[str, Any]:
    summary = create_qualifying_target_parity_audit(config, season=season, event=event)
    events = pd.read_csv(summary.event_summary_path)
    event_rows = events[events["event_slug"].astype(str).eq(event_slug)]
    if (
        event_rows.empty
        or not event_rows["event_parity_status"].astype(str).eq("parity_verified").all()
    ):
        raise ValueError("Event-specific qualifying target parity audit failed.")
    return {
        "artifact_paths": (
            summary.summary_path,
            summary.checks_path,
            summary.failures_path,
            summary.event_summary_path,
            summary.driver_comparison_path,
            summary.runbook_path,
        ),
        "event_specific_parity_status": "pass",
        "reason": "Event-specific qualifying target parity audit passed.",
    }


def _run_event_integrity_audit(
    config: DataConfig,
    *,
    event_slug: str,
) -> dict[str, Any]:
    summary = create_monitoring_data_integrity_audit(config)
    checks = pd.read_csv(summary.checks_path)
    failures = checks[
        checks["event_slug"].astype(str).eq(event_slug)
        & checks["blocking"].astype(bool)
        & checks["status"].astype(str).eq("failed")
    ]
    if not failures.empty:
        raise ValueError("Event-specific monitoring data integrity audit failed.")
    return {
        "artifact_paths": (
            summary.summary_path,
            summary.checks_path,
            summary.failures_path,
            summary.population_path,
            summary.event_comparison_path,
            summary.runbook_path,
        ),
        "event_specific_integrity_status": "pass",
        "reason": "Event-specific monitoring data integrity audit passed.",
    }


def _export_dashboard(
    config: DataConfig,
    *,
    event_slug: str,
    expected_lifecycle: set[str],
) -> dict[str, Any]:
    summary = export_dashboard_artifacts(config)
    current_path = config.metrics_output_dir.parent / "dashboard/current_event.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    current_slug = current.get("data", {}).get("event_identity", {}).get("event_slug")
    lifecycle = current.get("data", {}).get("lifecycle", {}).get("state")
    if current_slug != event_slug:
        raise ValueError(f"Dashboard current event is {current_slug}, expected {event_slug}.")
    if lifecycle not in expected_lifecycle:
        raise ValueError(
            f"Dashboard lifecycle is {lifecycle}, expected one of {expected_lifecycle}."
        )
    return {
        "artifact_paths": summary.artifact_paths,
        "dashboard_export_status": summary.status,
        "dashboard_current_event": summary.current_event,
        "reason": f"Dashboard export refreshed with lifecycle {lifecycle}.",
    }


def _reject_blocked_event(season: int, event: str, *, allow_test_event: bool) -> None:
    slug = slugify(event)
    if (int(season), slug) in {(2026, value) for value in LEGACY_EVENT_SLUGS}:
        raise ValueError("Legacy Australia and Great Britain artifacts are not operational events.")
    if synthetic_rehearsal_event_slug(slug) and not allow_test_event:
        raise ValueError("Synthetic rehearsal events are not valid operational monitoring events.")


def _registry_row(
    config: DataConfig,
    protocol_name: str,
    season: int,
    event: str,
) -> pd.Series | None:
    registry = read_csv(config.metrics_output_dir / "prospective_monitoring_event_registry.csv")
    if registry.empty:
        return None
    slug = slugify(event)
    mask = (
        registry.get("protocol_name", pd.Series(dtype=str)).astype(str).eq(protocol_name)
        & pd.to_numeric(registry.get("monitor_season", pd.Series(dtype=float)), errors="coerce").eq(
            int(season)
        )
        & registry.get("event_slug", pd.Series(dtype=str)).astype(str).eq(slug)
    )
    rows = registry[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def _validate_existing_registry_row(row: pd.Series, event_order: int) -> None:
    observed_order = int(
        pd.to_numeric(pd.Series([row.get("event_order")]), errors="coerce").iloc[0]
    )
    if observed_order != int(event_order):
        raise ValueError(
            f"Existing registry row has event_order={observed_order}, expected {event_order}."
        )
    if not bool(row.get("feature_artifact_valid", False)):
        raise ValueError("Existing registry row does not reference a valid feature artifact.")
    if not bool(row.get("forecastable", False)):
        raise ValueError("Existing registry row is not forecastable.")


def _forecast_rows(config: DataConfig, protocol_name: str, event_slug: str) -> pd.DataFrame:
    forecasts = read_parquet(config.metrics_output_dir / "prospective_monitoring_forecasts.parquet")
    if forecasts.empty:
        return pd.DataFrame()
    return forecasts[
        forecasts["protocol_name"].astype(str).eq(protocol_name)
        & forecasts["event_slug"].astype(str).eq(event_slug)
    ].copy()


def _settlement_rows(config: DataConfig, protocol_name: str, event_slug: str) -> pd.DataFrame:
    settlements = read_parquet(
        config.metrics_output_dir / "prospective_monitoring_settlements.parquet"
    )
    if settlements.empty:
        return pd.DataFrame()
    return settlements[
        settlements["protocol_name"].astype(str).eq(protocol_name)
        & settlements["event_slug"].astype(str).eq(event_slug)
    ].copy()


def _event_rows(
    config: DataConfig,
    filename: str,
    protocol_name: str,
    event_slug: str,
) -> bool:
    frame = read_parquet(config.metrics_output_dir / filename)
    if frame.empty:
        return False
    return bool(
        (
            frame["protocol_name"].astype(str).eq(protocol_name)
            & frame["event_slug"].astype(str).eq(event_slug)
        ).any()
    )


def _existing_forecast_valid(config: DataConfig, protocol_name: str, event_slug: str) -> bool:
    forecasts = _forecast_rows(config, protocol_name, event_slug)
    if forecasts.empty:
        return False
    expected = expected_forecast_hash(config.metrics_output_dir, protocol_name, event_slug)
    if not expected:
        return False
    shadow = read_parquet(
        config.metrics_output_dir / "prospective_monitoring_shadow_candidates.parquet"
    )
    if not shadow.empty:
        shadow = shadow[
            shadow["protocol_name"].astype(str).eq(protocol_name)
            & shadow["event_slug"].astype(str).eq(event_slug)
        ].copy()
    return forecast_snapshot_hash(forecasts, shadow) == expected


def _validate_existing_settlement(
    config: DataConfig,
    protocol_name: str,
    season: int,
    event_slug: str,
) -> None:
    settlements = _settlement_rows(config, protocol_name, event_slug)
    if settlements.empty:
        raise ValueError("Existing settlement rows are missing.")
    audit = read_csv(
        config.metrics_output_dir / "prospective_monitoring_settlement_integrity_audit.csv"
    )
    if audit.empty:
        raise ValueError("Existing settlement integrity audit is missing.")
    event_audit = audit[
        audit["protocol_name"].astype(str).eq(protocol_name)
        & audit["event_slug"].astype(str).eq(event_slug)
    ]
    if event_audit.empty or not event_audit["settlement_valid"].astype(bool).all():
        raise ValueError("Existing settlement is not valid for reuse.")
    valid, reason = validate_target_raw_identity(config, season, event_slug)
    if not valid:
        raise ValueError(f"Existing settlement target provenance is invalid: {reason}.")


def _settlement_denominator(config: DataConfig, event_slug: str) -> int | None:
    metrics = read_csv(config.metrics_output_dir / "prospective_monitoring_event_metrics.csv")
    if metrics.empty:
        return None
    rows = metrics[
        metrics["event_slug"].astype(str).eq(event_slug)
        & metrics["prediction_role"].astype(str).eq(LIVE_POLICY_ROLE)
    ]
    if rows.empty:
        return None
    return int(rows.iloc[0]["scored_rows"])


def _recommended_action(stage: str) -> str:
    actions = {
        "practice_ingested": "Ensure FP1, FP2, and FP3 raw artifacts exist locally.",
        "event_prepared": "Review monitoring event preparation artifacts.",
        "event_registered": "Fix or remove the inconsistent registry row before rerunning.",
        "preflight_ready": "Resolve preflight blockers before creating a forecast.",
        "forecast_created": "Keep the immutable forecast and proceed after qualifying.",
        "qualifying_ingested": "Ensure Q raw artifacts exist locally.",
        "raw_q_identity_verified": "Run raw-session-identity-validate and fix raw Q metadata.",
        "targets_added": "Repair target provenance before settlement.",
        "settled": "Inspect settlement integrity before dashboard publication.",
        "parity_audit_passed": "Repair target/raw-Q parity for this event.",
        "integrity_audit_passed": "Repair event-specific monitoring integrity failures.",
        "dashboard_exported": "Refresh dashboard artifacts after the prior stages pass.",
    }
    return actions.get(stage, "Resolve the blocking condition and rerun.")


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
