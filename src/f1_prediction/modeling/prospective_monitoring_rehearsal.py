"""End-to-end guarded rehearsal for prospective monitoring operations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig, FeatureConfig, ModelConfig
from f1_prediction.dashboard.export import export_dashboard_artifacts
from f1_prediction.data.monitoring_onboarding import (
    add_monitoring_targets,
    event_manifest_path,
    feature_artifact_path,
    prepare_monitoring_event,
    register_monitoring_event,
    target_artifact_path,
    utc_now,
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
    PREFLIGHT_READY,
    create_prospective_monitoring_forecast,
    create_prospective_monitoring_preflight,
    create_prospective_monitoring_report,
    create_prospective_monitoring_settlement,
    synthetic_rehearsal_event_slug,
)
from f1_prediction.modeling.qualifying_target_parity_audit import (
    create_qualifying_target_parity_audit,
)
from f1_prediction.utils.paths import ensure_directory, slugify

REHEARSAL_STAGES: tuple[str, ...] = (
    "not_started",
    "practice_artifacts_ready",
    "event_prepared",
    "event_registered",
    "preflight_ready",
    "forecast_created",
    "raw_q_identity_verified",
    "targets_added",
    "settled",
    "audits_passed",
    "dashboard_published",
)

LEGACY_EVENT_SLUGS = ("australia", "great-britain")


@dataclass(frozen=True)
class ProspectiveMonitoringRehearsalSummary:
    """Output paths and status for a monitored-event rehearsal."""

    status: str
    summary_path: Path
    stages_path: Path
    checks_path: Path
    failures_path: Path
    driver_population_path: Path
    runbook_path: Path
    blocking_failure_count: int
    warning_count: int
    event: str
    event_slug: str
    synthetic_rehearsal: bool
    valid_prospective_evidence: bool


def create_prospective_monitoring_rehearsal(
    config: DataConfig,
    model_config: ModelConfig,
    feature_config: FeatureConfig,
    *,
    protocol_name: str,
    season: int,
    event: str,
    event_order: int | None = None,
    synthetic_rehearsal: bool | None = None,
) -> ProspectiveMonitoringRehearsalSummary:
    """Run the guarded monitoring workflow for one already-ingested event."""
    event_slug = slugify(event)
    synthetic = (
        synthetic_rehearsal
        if synthetic_rehearsal is not None
        else synthetic_rehearsal_event_slug(event_slug)
    )
    output = _Recorder(config, protocol_name, season, event, event_slug, synthetic)
    legacy_before = _legacy_artifact_fingerprints(config, protocol_name)
    output.start()

    for stage, action in (
        ("practice_artifacts_ready", lambda: _validate_practice_artifacts(config, season, event)),
        (
            "event_prepared",
            lambda: _prepare_event(config, feature_config, season=season, event=event),
        ),
        (
            "event_registered",
            lambda: _register_event(
                config,
                protocol_name=protocol_name,
                season=season,
                event=event,
                event_order=event_order,
            ),
        ),
        (
            "preflight_ready",
            lambda: _preflight_event(
                config,
                protocol_name=protocol_name,
                season=season,
                event=event,
            ),
        ),
        (
            "forecast_created",
            lambda: _forecast_event(
                config,
                model_config,
                feature_config,
                protocol_name=protocol_name,
                event=event,
                event_slug=event_slug,
            ),
        ),
        (
            "raw_q_identity_verified",
            lambda: _verify_raw_q_identity(config, season=season, event=event),
        ),
        ("targets_added", lambda: _add_targets(config, season=season, event=event)),
        (
            "settled",
            lambda: _settle_event(
                config,
                protocol_name=protocol_name,
                season=season,
                event=event,
                event_slug=event_slug,
            ),
        ),
        (
            "audits_passed",
            lambda: _run_audits(
                config,
                protocol_name=protocol_name,
                season=season,
                event=event,
                event_slug=event_slug,
            ),
        ),
        (
            "dashboard_published",
            lambda: _export_dashboard(
                config,
                event_slug=event_slug,
                synthetic_rehearsal=synthetic,
            ),
        ),
    ):
        if output.blocked:
            break
        output.run_stage(stage, action)

    legacy_after = _legacy_artifact_fingerprints(config, protocol_name)
    _add_legacy_checks(output, legacy_before, legacy_after)
    _add_population_checks(output)
    _add_dashboard_checks(output)
    return output.write()


class _Recorder:
    def __init__(
        self,
        config: DataConfig,
        protocol_name: str,
        season: int,
        event: str,
        event_slug: str,
        synthetic_rehearsal: bool,
    ) -> None:
        self.config = config
        self.protocol_name = protocol_name
        self.season = season
        self.event = event
        self.event_slug = event_slug
        self.synthetic_rehearsal = synthetic_rehearsal
        self.stage_rows: list[dict[str, Any]] = []
        self.check_rows: list[dict[str, Any]] = []
        self.stage_status: dict[str, str] = {}
        self.artifact_paths: dict[str, list[str]] = {}
        self.blocked = False
        self.blocking_reason = ""
        self.summary_values: dict[str, Any] = {}

    def start(self) -> None:
        now = utc_now()
        self.stage_rows.append(
            {
                "stage": "not_started",
                "status": "complete",
                "blocking": False,
                "started_at_utc": now,
                "completed_at_utc": now,
                "artifact_paths": "",
                "reason": "Rehearsal state machine initialized.",
                "recommended_action": "Validate local artifacts before running workflow stages.",
            }
        )
        self.stage_status["not_started"] = "complete"

    def run_stage(self, stage: str, action: Callable[[], dict[str, Any]]) -> None:
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
            self.check(
                stage,
                f"{stage}_blocking_failure",
                "failed",
                True,
                str(exc),
                "stage completes without blocking failure",
                _recommended_action(stage),
            )
            return
        paths = tuple(str(path) for path in result.get("artifact_paths", ()))
        for key, value in result.items():
            if key != "artifact_paths":
                self.summary_values[key] = value
        self.artifact_paths[stage] = [
            _project_relative(Path(path), self.config.project_root) for path in paths
        ]
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
        artifact_paths: tuple[str, ...],
        reason: str,
        recommended_action: str,
    ) -> None:
        self.stage_status[stage] = status
        self.stage_rows.append(
            {
                "stage": stage,
                "status": status,
                "blocking": bool(blocking),
                "started_at_utc": started,
                "completed_at_utc": completed,
                "artifact_paths": "|".join(
                    _project_relative(Path(path), self.config.project_root)
                    for path in artifact_paths
                ),
                "reason": reason,
                "recommended_action": recommended_action,
            }
        )

    def check(
        self,
        stage: str,
        check_name: str,
        status: str,
        blocking: bool,
        observed: Any,
        expected: Any,
        recommended_action: str,
    ) -> None:
        self.check_rows.append(
            {
                "stage": stage,
                "check_name": check_name,
                "status": status,
                "blocking": bool(blocking),
                "observed_value": json.dumps(_json_clean(observed), sort_keys=True),
                "expected_value": json.dumps(_json_clean(expected), sort_keys=True),
                "recommended_action": recommended_action,
            }
        )

    def write(self) -> ProspectiveMonitoringRehearsalSummary:
        for stage in REHEARSAL_STAGES:
            if stage not in self.stage_status:
                self._stage(
                    stage,
                    "not_started",
                    False,
                    "",
                    "",
                    (),
                    "Stage was not reached because an earlier stage blocked."
                    if self.blocked
                    else "Stage was not reached.",
                    _recommended_action(stage),
                )
        metrics_dir = ensure_directory(self.config.metrics_output_dir)
        stages = pd.DataFrame(self.stage_rows)
        checks = pd.DataFrame(self.check_rows)
        failures = checks[
            checks["status"].astype(str).isin(["failed", "warning"])
            | checks["blocking"].astype(bool)
        ].copy()
        population = _driver_population(self.config, self.event_slug)
        blocking_count = int(
            checks["blocking"]
            .astype(bool)
            .where(checks["status"].astype(str).eq("failed"), False)
            .sum()
        )
        warning_count = int(checks["status"].astype(str).eq("warning").sum())
        completed = bool(
            self.stage_status.get("dashboard_published") == "complete" and blocking_count == 0
        )
        valid_evidence = bool(completed and not self.synthetic_rehearsal)
        summary = {
            "status": "pass" if completed else "blocked",
            "protocol_name": self.protocol_name,
            "season": self.season,
            "event": self.event,
            "event_slug": self.event_slug,
            "synthetic_rehearsal": self.synthetic_rehearsal,
            "rehearsal_completed": completed,
            "blocking_failure_count": blocking_count,
            "warning_count": warning_count,
            "preflight_status": self.summary_values.get("preflight_status", "not_run"),
            "raw_q_identity_status": self.summary_values.get("raw_q_identity_status", "not_run"),
            "forecast_status": self.summary_values.get("forecast_status", "not_run"),
            "target_status": self.summary_values.get("target_status", "not_run"),
            "settlement_status": self.summary_values.get("settlement_status", "not_run"),
            "integrity_audit_status": self.summary_values.get(
                "integrity_audit_status",
                "not_run",
            ),
            "parity_audit_status": self.summary_values.get("parity_audit_status", "not_run"),
            "dashboard_status": self.summary_values.get("dashboard_status", "not_run"),
            "dashboard_current_event": self.summary_values.get("dashboard_current_event"),
            "valid_prospective_evidence": valid_evidence,
            "recommended_operator_action": (
                "Use this rehearsal as workflow validation only; do not treat it as real "
                "prospective evidence."
                if self.synthetic_rehearsal
                else "Retain the clean event as valid prospective evidence."
                if completed
                else self.blocking_reason or "Resolve blocking rehearsal checks and rerun."
            ),
            "driver_population_counts": _population_counts(population),
            "generated_at_utc": utc_now(),
        }
        summary_path = metrics_dir / "prospective_monitoring_rehearsal_summary.json"
        stages_path = metrics_dir / "prospective_monitoring_rehearsal_stages.csv"
        checks_path = metrics_dir / "prospective_monitoring_rehearsal_checks.csv"
        failures_path = metrics_dir / "prospective_monitoring_rehearsal_failures.csv"
        population_path = metrics_dir / "prospective_monitoring_rehearsal_driver_population.csv"
        runbook_path = metrics_dir / "prospective_monitoring_rehearsal_runbook.md"
        write_json(summary_path, summary)
        stages.to_csv(stages_path, index=False)
        checks.to_csv(checks_path, index=False)
        failures.to_csv(failures_path, index=False)
        population.to_csv(population_path, index=False)
        runbook_path.write_text(_runbook(summary, stages, failures), encoding="utf-8")
        return ProspectiveMonitoringRehearsalSummary(
            status=str(summary["status"]),
            summary_path=summary_path,
            stages_path=stages_path,
            checks_path=checks_path,
            failures_path=failures_path,
            driver_population_path=population_path,
            runbook_path=runbook_path,
            blocking_failure_count=blocking_count,
            warning_count=warning_count,
            event=self.event,
            event_slug=self.event_slug,
            synthetic_rehearsal=self.synthetic_rehearsal,
            valid_prospective_evidence=valid_evidence,
        )


def _validate_practice_artifacts(config: DataConfig, season: int, event: str) -> dict[str, Any]:
    paths: list[Path] = []
    for session in ("FP1", "FP2", "FP3"):
        identity = validate_raw_session_identity(
            config,
            season=season,
            event=event,
            session=session,
        )
        if identity.blocking:
            raise ValueError(
                f"{session} raw identity validation failed: {identity.identity_status}"
            )
        paths.extend([identity.raw_laps_path, identity.raw_metadata_path])
    return {
        "artifact_paths": paths,
        "reason": "FP1, FP2, and FP3 raw artifacts and metadata match the requested event.",
    }


def _prepare_event(
    config: DataConfig,
    feature_config: FeatureConfig,
    *,
    season: int,
    event: str,
) -> dict[str, Any]:
    summary = prepare_monitoring_event(config, feature_config, season=season, event=event)
    features_path = feature_artifact_path(config, season, event)
    features = pd.read_parquet(features_path)
    if features.empty:
        raise ValueError("Prepared feature artifact contains no drivers.")
    return {
        "artifact_paths": (features_path, summary.summary_path),
        "reason": f"Prepared {features['driver'].nunique()} FP3-safe feature participants.",
    }


def _register_event(
    config: DataConfig,
    *,
    protocol_name: str,
    season: int,
    event: str,
    event_order: int | None,
) -> dict[str, Any]:
    summary = register_monitoring_event(
        config,
        protocol_name=protocol_name,
        season=season,
        event=event,
        event_order=event_order,
    )
    registry = pd.read_csv(summary.summary_path)
    slug = slugify(event)
    row = registry[registry["event_slug"].astype(str).eq(slug)]
    if row.empty:
        raise ValueError("Registered event is missing from the monitoring registry.")
    if row["event_order"].duplicated(keep=False).any():
        raise ValueError("Registered event order is not unique.")
    return {
        "artifact_paths": summary.table_paths,
        "reason": "Event registered in the frozen monitoring registry.",
    }


def _preflight_event(
    config: DataConfig,
    *,
    protocol_name: str,
    season: int,
    event: str,
) -> dict[str, Any]:
    summary = create_prospective_monitoring_preflight(
        config,
        protocol_name=protocol_name,
        season=season,
        event=event,
    )
    if summary.status != PREFLIGHT_READY:
        raise ValueError(f"Preflight status is {summary.status}, expected {PREFLIGHT_READY}.")
    return {
        "artifact_paths": (summary.summary_path, *summary.table_paths),
        "preflight_status": summary.status,
        "reason": "Preflight returned ready_to_forecast.",
    }


def _forecast_event(
    config: DataConfig,
    model_config: ModelConfig,
    feature_config: FeatureConfig,
    *,
    protocol_name: str,
    event: str,
    event_slug: str,
) -> dict[str, Any]:
    existing = _event_rows(
        config.metrics_output_dir / "prospective_monitoring_forecasts.parquet",
        protocol_name,
        event_slug,
    )
    if not existing.empty:
        raise ValueError("Existing forecast snapshot cannot be overwritten.")
    summary = create_prospective_monitoring_forecast(
        config,
        model_config,
        feature_config,
        protocol_name=protocol_name,
        event=event,
    )
    forecasts = _event_rows(
        config.metrics_output_dir / "prospective_monitoring_forecasts.parquet",
        protocol_name,
        event_slug,
    )
    if forecasts.empty:
        raise ValueError("Forecast rows were not written for the rehearsal event.")
    if not forecasts["preflight_status"].astype(str).eq(PREFLIGHT_READY).all():
        raise ValueError("Forecast rows do not preserve ready preflight provenance.")
    return {
        "artifact_paths": summary.table_paths,
        "forecast_status": summary.status,
        "reason": (
            f"Immutable forecast snapshot created for {forecasts['driver'].nunique()} drivers."
        ),
    }


def _verify_raw_q_identity(config: DataConfig, *, season: int, event: str) -> dict[str, Any]:
    identity = validate_raw_session_identity(config, season=season, event=event, session="Q")
    (
        _summary,
        _checks,
        _failures,
        _quarantine,
        runbook,
    ) = create_raw_session_identity_validation_report(
        config,
        season=season,
        event=event,
        session="Q",
    )
    if identity.identity_status != IDENTITY_VERIFIED or identity.blocking:
        raise ValueError(f"Raw Q identity validation failed: {identity.identity_status}.")
    return {
        "artifact_paths": (identity.raw_laps_path, identity.raw_metadata_path, runbook),
        "raw_q_identity_status": identity.identity_status,
        "reason": "Raw Q path and metadata identity are verified for the rehearsal event.",
    }


def _add_targets(config: DataConfig, *, season: int, event: str) -> dict[str, Any]:
    if target_artifact_path(config, season, event).is_file():
        raise ValueError("Existing target artifact cannot be overwritten by rehearsal.")
    summary = add_monitoring_targets(config, season=season, event=event)
    targets = pd.read_parquet(target_artifact_path(config, season, event))
    if targets["quali_position"].duplicated().any():
        raise ValueError("Target qualifying positions are not unique.")
    pole_gap = pd.to_numeric(
        targets.loc[targets["quali_position"].astype(int).eq(1), "quali_gap_to_pole_sec"],
        errors="coerce",
    )
    if pole_gap.empty or not pole_gap.eq(0).all():
        raise ValueError("Target pole gap must equal zero.")
    manifest = json.loads(event_manifest_path(config, season, event).read_text(encoding="utf-8"))
    if manifest.get("raw_session_identity_status") != IDENTITY_VERIFIED:
        raise ValueError("Target manifest does not preserve verified raw-Q identity provenance.")
    return {
        "artifact_paths": summary.table_paths,
        "target_status": summary.status,
        "reason": f"Targets added for {len(targets)} evaluable qualifying drivers.",
    }


def _settle_event(
    config: DataConfig,
    *,
    protocol_name: str,
    season: int,
    event: str,
    event_slug: str,
) -> dict[str, Any]:
    existing = _event_rows(
        config.metrics_output_dir / "prospective_monitoring_settlements.parquet",
        protocol_name,
        event_slug,
    )
    if not existing.empty:
        raise ValueError("Existing settlement cannot be overwritten by rehearsal.")
    summary = create_prospective_monitoring_settlement(
        config,
        protocol_name=protocol_name,
        event=event,
    )
    _validate_settlement_matches_targets(config, protocol_name, season, event_slug)
    report = create_prospective_monitoring_report(config)
    return {
        "artifact_paths": (*summary.table_paths, report.summary_path, *report.table_paths),
        "settlement_status": summary.status,
        "reason": "Settlement rows match stored targets and monitoring report was refreshed.",
    }


def _run_audits(
    config: DataConfig,
    *,
    protocol_name: str,
    season: int,
    event: str,
    event_slug: str,
) -> dict[str, Any]:
    parity = create_qualifying_target_parity_audit(config, season=season, event=event)
    parity_events = pd.read_csv(parity.event_summary_path)
    parity_row = parity_events[parity_events["event_slug"].astype(str).eq(event_slug)]
    if (
        parity_row.empty
        or not parity_row["event_parity_status"].astype(str).eq("parity_verified").all()
    ):
        raise ValueError("Rehearsal event failed qualifying-target parity audit.")
    integrity = create_monitoring_data_integrity_audit(config)
    integrity_checks = pd.read_csv(integrity.checks_path)
    event_failures = integrity_checks[
        integrity_checks["event_slug"].astype(str).eq(event_slug)
        & integrity_checks["blocking"].astype(bool)
        & integrity_checks["status"].astype(str).eq("failed")
    ]
    if not event_failures.empty:
        raise ValueError("Rehearsal event failed monitoring data integrity audit.")
    return {
        "artifact_paths": (
            parity.summary_path,
            parity.checks_path,
            parity.failures_path,
            parity.event_summary_path,
            parity.driver_comparison_path,
            parity.runbook_path,
            integrity.summary_path,
            integrity.checks_path,
            integrity.failures_path,
            integrity.population_path,
            integrity.event_comparison_path,
            integrity.runbook_path,
        ),
        "parity_audit_status": "pass",
        "integrity_audit_status": "pass",
        "global_parity_audit_status": parity.status,
        "global_integrity_audit_status": integrity.status,
        "reason": "Rehearsal event passed parity and integrity audits.",
    }


def _export_dashboard(
    config: DataConfig,
    *,
    event_slug: str,
    synthetic_rehearsal: bool,
) -> dict[str, Any]:
    summary = export_dashboard_artifacts(config)
    current_path = config.metrics_output_dir.parent / "dashboard/current_event.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    current_slug = current.get("data", {}).get("event_identity", {}).get("event_slug")
    lifecycle = current.get("data", {}).get("lifecycle", {}).get("state")
    if synthetic_rehearsal:
        if current_slug == event_slug:
            raise ValueError("Synthetic rehearsal event was selected as Dashboard Current Event.")
        if not current.get("generated_at_utc"):
            raise ValueError("Dashboard current event freshness timestamp is missing.")
        historical_path = (
            config.metrics_output_dir.parent / "dashboard/historical_monitoring_summary.json"
        )
        historical = json.loads(historical_path.read_text(encoding="utf-8"))
        synthetic_rows = historical.get("data", {}).get("synthetic_rehearsal_records", [])
        if not any(
            row.get("event_identity", {}).get("event_slug") == event_slug for row in synthetic_rows
        ):
            raise ValueError(
                "Synthetic rehearsal event is missing from dashboard internal history."
            )
        return {
            "artifact_paths": summary.artifact_paths,
            "dashboard_status": summary.status,
            "dashboard_current_event": summary.current_event,
            "reason": "Dashboard export kept the synthetic rehearsal out of Current Event.",
        }
    if current_slug != event_slug:
        raise ValueError(f"Dashboard current event is {current_slug}, expected {event_slug}.")
    if lifecycle != "settled":
        raise ValueError(f"Dashboard lifecycle is {lifecycle}, expected settled.")
    if not current.get("generated_at_utc"):
        raise ValueError("Dashboard current event freshness timestamp is missing.")
    return {
        "artifact_paths": summary.artifact_paths,
        "dashboard_status": summary.status,
        "dashboard_current_event": summary.current_event,
        "reason": "Dashboard export selected the clean rehearsal event as Current Event.",
    }


def _validate_settlement_matches_targets(
    config: DataConfig,
    protocol_name: str,
    season: int,
    event_slug: str,
) -> None:
    targets = pd.read_parquet(target_artifact_path(config, season, event_slug))
    settlements = _event_rows(
        config.metrics_output_dir / "prospective_monitoring_settlements.parquet",
        protocol_name,
        event_slug,
    )
    live = settlements[settlements["prediction_role"].astype(str).eq("observed_live_policy")]
    evaluable = live[live["settlement_evaluable"].astype(bool)]
    target_by_driver = targets.set_index("driver_key")
    for _, row in evaluable.iterrows():
        target = target_by_driver.loc[str(row["driver_key"])]
        if float(row["actual_gap_sec"]) != float(target["quali_gap_to_pole_sec"]):
            raise ValueError("Settlement actual gaps do not match stored targets.")
    event_metrics = pd.read_csv(
        config.metrics_output_dir / "prospective_monitoring_event_metrics.csv"
    )
    metric_rows = event_metrics[
        event_metrics["event_slug"].astype(str).eq(event_slug)
        & event_metrics["prediction_role"].astype(str).eq("observed_live_policy")
    ]
    if metric_rows.empty:
        raise ValueError("Settlement metric denominator is missing.")
    if int(metric_rows.iloc[0]["scored_rows"]) != int(len(evaluable)):
        raise ValueError("Settlement metric denominator does not match evaluable rows.")


def _add_legacy_checks(
    output: _Recorder,
    before: dict[str, str],
    after: dict[str, str],
) -> None:
    for name in ("forecasts", "settlements"):
        output.check(
            "audits_passed",
            f"legacy_{name}_fingerprint_unchanged",
            "passed" if before.get(name) == after.get(name) else "failed",
            True,
            {"before": before.get(name), "after": after.get(name)},
            "unchanged",
            "Do not mutate Australia or Great Britain legacy artifacts.",
        )
    reconciliation = _read_csv(
        output.config.metrics_output_dir / "prospective_monitoring_event_order_reconciliation.csv"
    )
    for slug in LEGACY_EVENT_SLUGS:
        rows = (
            reconciliation[reconciliation["event_slug"].astype(str).eq(slug)]
            if "event_slug" in reconciliation
            else pd.DataFrame()
        )
        eligible = (
            rows["eligible_for_future_prior_evidence_after_reconciliation"].astype(bool).any()
            if not rows.empty
            else False
        )
        output.check(
            "audits_passed",
            f"{slug}_not_prior_valid_evidence",
            "passed" if not eligible else "failed",
            True,
            eligible,
            False,
            "Keep legacy Australia and Great Britain excluded from future prior evidence.",
        )
    raw_identity = _read_csv(
        output.config.metrics_output_dir / "raw_session_identity_validation_checks.csv"
    )
    gb = (
        raw_identity[raw_identity["requested_event_slug"].astype(str).eq("great-britain")]
        if "requested_event_slug" in raw_identity
        else pd.DataFrame()
    )
    if not gb.empty:
        status = str(gb.iloc[0].get("identity_status"))
        output.check(
            "audits_passed",
            "great_britain_raw_q_mismatch_quarantined",
            "passed" if status == "legacy_known_mismatch" else "failed",
            True,
            status,
            "legacy_known_mismatch",
            "Do not reinterpret Great Britain raw Q as valid evidence.",
        )


def _add_population_checks(output: _Recorder) -> None:
    population = _driver_population(output.config, output.event_slug)
    if population.empty:
        return
    counts = _population_counts(population)
    output.summary_values["driver_population_counts"] = counts
    forecast_only = population[population["forecast_only_driver"].astype(bool)]
    output.check(
        "audits_passed",
        "forecast_only_drivers_have_reason",
        "passed"
        if forecast_only.empty
        or forecast_only["forecast_only_reason"].fillna("").astype(str).str.len().gt(0).all()
        else "failed",
        True,
        int(len(forecast_only)),
        "every forecast-only row has a reason",
        "Populate coverage reasons for non-evaluable forecast rows.",
    )


def _add_dashboard_checks(output: _Recorder) -> None:
    current_path = output.config.metrics_output_dir.parent / "dashboard/current_event.json"
    if not current_path.is_file():
        return
    current = json.loads(current_path.read_text(encoding="utf-8"))
    forecast = json.loads(
        (output.config.metrics_output_dir.parent / "dashboard/event_forecast.json").read_text(
            encoding="utf-8"
        )
    )
    current_slug = current.get("data", {}).get("event_identity", {}).get("event_slug")
    if output.synthetic_rehearsal:
        output.check(
            "dashboard_published",
            "synthetic_rehearsal_excluded_from_current_event",
            "passed" if current_slug != output.event_slug else "failed",
            True,
            current_slug or "no_event_available",
            "not selected as Current Event",
            "Keep synthetic rehearsals out of the public Current Event selection.",
        )
        historical_path = (
            output.config.metrics_output_dir.parent / "dashboard/historical_monitoring_summary.json"
        )
        historical = json.loads(historical_path.read_text(encoding="utf-8"))
        synthetic_rows = historical.get("data", {}).get("synthetic_rehearsal_records", [])
        synthetic_slugs = {
            row.get("event_identity", {}).get("event_slug") for row in synthetic_rows
        }
        output.check(
            "dashboard_published",
            "synthetic_rehearsal_recorded_as_internal_history",
            "passed" if output.event_slug in synthetic_slugs else "failed",
            True,
            len(synthetic_rows),
            "synthetic rehearsal recorded outside Current Event",
            "Retain synthetic rehearsal records only in internal dashboard history.",
        )
        return
    output.check(
        "dashboard_published",
        "dashboard_current_event_is_clean_rehearsal_event",
        "passed" if current_slug == output.event_slug else "failed",
        True,
        current_slug,
        output.event_slug,
        "Export dashboard after the clean event is settled.",
    )
    leaderboard = forecast.get("data", {}).get("leaderboard", [])
    if isinstance(leaderboard, list):
        output.check(
            "dashboard_published",
            "primary_leaderboard_excludes_forecast_only_rows",
            "passed"
            if all(not row.get("forecast_only_driver", False) for row in leaderboard)
            else "failed",
            True,
            len(leaderboard),
            "only qualifying-eligible drivers",
            "Use qualifying_eligible_forecast_rows for public leaderboard output.",
        )


def _legacy_artifact_fingerprints(config: DataConfig, protocol_name: str) -> dict[str, str]:
    return {
        "forecasts": _subset_fingerprint(
            _event_subset(
                config.metrics_output_dir / "prospective_monitoring_forecasts.parquet",
                protocol_name,
                LEGACY_EVENT_SLUGS,
            )
        ),
        "settlements": _subset_fingerprint(
            _event_subset(
                config.metrics_output_dir / "prospective_monitoring_settlements.parquet",
                protocol_name,
                LEGACY_EVENT_SLUGS,
            )
        ),
    }


def _event_rows(path: Path, protocol_name: str, event_slug: str) -> pd.DataFrame:
    return _event_subset(path, protocol_name, (event_slug,))


def _event_subset(path: Path, protocol_name: str, event_slugs: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if frame.empty or "event_slug" not in frame:
        return pd.DataFrame()
    return frame[
        frame.get("protocol_name", pd.Series(dtype=str)).astype(str).eq(protocol_name)
        & frame["event_slug"].astype(str).isin(event_slugs)
    ].copy()


def _subset_fingerprint(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "empty"
    ordered = frame.sort_index(axis=1).sort_values(
        [
            column
            for column in ("event_slug", "forecast_id", "driver", "prediction_role")
            if column in frame
        ],
        kind="stable",
    )
    payload = ordered.astype(object).where(pd.notna(ordered), None).to_dict("records")
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _driver_population(config: DataConfig, event_slug: str) -> pd.DataFrame:
    path = config.metrics_output_dir / "monitoring_data_integrity_event_driver_population.csv"
    if path.is_file():
        frame = pd.read_csv(path)
        if not frame.empty and "event_slug" in frame:
            return frame[frame["event_slug"].astype(str).eq(event_slug)].copy()
    return pd.DataFrame(
        columns=[
            "season",
            "event",
            "event_slug",
            "driver",
            "driver_key",
            "feature_participant",
            "forecast_eligible_driver",
            "forecast_only_driver",
            "forecast_only_reason",
            "target_present",
            "settlement_evaluable_driver",
        ]
    )


def _population_counts(population: pd.DataFrame) -> dict[str, int]:
    if population.empty:
        return {
            "feature_participant_count": 0,
            "forecast_eligible_driver_count": 0,
            "forecast_only_driver_count": 0,
            "target_present_driver_count": 0,
            "settlement_evaluable_driver_count": 0,
        }
    return {
        "feature_participant_count": _bool_sum(population, "feature_participant"),
        "forecast_eligible_driver_count": _bool_sum(population, "forecast_eligible_driver"),
        "forecast_only_driver_count": _bool_sum(population, "forecast_only_driver"),
        "target_present_driver_count": _bool_sum(population, "target_present"),
        "settlement_evaluable_driver_count": _bool_sum(
            population,
            "settlement_evaluable_driver",
        ),
    }


def _bool_sum(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _runbook(summary: dict[str, Any], stages: pd.DataFrame, failures: pd.DataFrame) -> str:
    lines = [
        "# Prospective Monitoring Rehearsal Runbook",
        "",
        f"Status: `{summary['status']}`",
        f"Event: `{summary['event']}` (`{summary['event_slug']}`)",
        f"Synthetic rehearsal: `{str(summary['synthetic_rehearsal']).lower()}`",
        f"Valid prospective evidence: `{str(summary['valid_prospective_evidence']).lower()}`",
        "",
        "## Stage Results",
        "",
    ]
    for _, row in stages.iterrows():
        lines.append(f"- `{row['stage']}`: `{row['status']}` - {row['reason']}")
    lines.extend(
        [
            "",
            "## Next Real GP Operator Sequence",
            "",
            "Before qualifying:",
            "",
            "```bash",
            ".venv/bin/python -m f1_prediction.cli ingest-event "
            '--season 2026 --event "<EVENT>" --sessions FP1 FP2 FP3',
            ".venv/bin/python -m f1_prediction.cli monitoring-prepare-event "
            '--season 2026 --event "<EVENT>"',
            ".venv/bin/python -m f1_prediction.cli monitoring-register-event "
            "--protocol-name season_2026_v1 --season 2026 "
            '--event "<EVENT>" --event-order <ORDER>',
            ".venv/bin/python -m f1_prediction.cli prospective-monitoring-preflight "
            "--protocol-name season_2026_v1 --season 2026 "
            '--event "<EVENT>"',
            ".venv/bin/python -m f1_prediction.cli prospective-monitoring-forecast "
            "--protocol-name season_2026_v1 "
            '--event "<EVENT>"',
            "```",
            "",
            "After qualifying:",
            "",
            "```bash",
            ".venv/bin/python -m f1_prediction.cli ingest-event "
            '--season 2026 --event "<EVENT>" --sessions Q',
            ".venv/bin/python -m f1_prediction.cli raw-session-identity-validate "
            '--season 2026 --event "<EVENT>" --session Q',
            ".venv/bin/python -m f1_prediction.cli monitoring-add-targets "
            '--season 2026 --event "<EVENT>"',
            ".venv/bin/python -m f1_prediction.cli prospective-monitoring-settle "
            "--protocol-name season_2026_v1 "
            '--event "<EVENT>"',
            ".venv/bin/python -m f1_prediction.cli qualifying-target-parity-audit",
            ".venv/bin/python -m f1_prediction.cli monitoring-data-integrity-audit",
            ".venv/bin/python -m f1_prediction.cli dashboard-export",
            "```",
            "",
            "Expected status: preflight `ready_to_forecast`, raw-Q identity "
            "`identity_verified`, target onboarding `targets_added`, settlement "
            "`settled`, audits with no blocking failure for the clean event, and "
            "dashboard lifecycle `settled`.",
        ]
    )
    if not failures.empty:
        lines.extend(["", "## Blocking Failures", ""])
        for _, row in failures.iterrows():
            lines.append(f"- `{row['stage']}` / `{row['check_name']}`: {row['recommended_action']}")
    return "\n".join(lines) + "\n"


def _recommended_action(stage: str) -> str:
    return {
        "practice_artifacts_ready": "Ingest FP1, FP2, and FP3 locally for the requested event.",
        "event_prepared": "Rerun monitoring-prepare-event after practice artifacts are complete.",
        "event_registered": "Register the prepared event with a unique frozen event order.",
        "preflight_ready": "Resolve preflight failures before creating any forecast.",
        "forecast_created": (
            "Do not overwrite forecasts; use a clean event or inspect existing artifacts."
        ),
        "raw_q_identity_verified": "Ingest and validate the correct local Q session metadata.",
        "targets_added": "Run target onboarding only after raw-Q identity is verified.",
        "settled": "Settle only matching forecast and verified target artifacts.",
        "audits_passed": "Inspect parity and integrity failures before dashboard publication.",
        "dashboard_published": "Refresh dashboard export after clean settlement and audits.",
    }.get(stage, "Inspect the rehearsal reports and rerun from the first blocked stage.")


def _json_clean(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_clean(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_clean(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()
