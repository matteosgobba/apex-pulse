"""Read-only dashboard artifact exporter."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.dashboard.schema import (
    DashboardEnvelope,
    EventIdentity,
    LifecycleState,
    SourceArtifact,
    available,
    unavailable,
    validate_dashboard_document,
    validate_lifecycle_state,
)
from f1_prediction.utils.paths import ensure_directory, slugify

DASHBOARD_FILES: tuple[str, ...] = (
    "dashboard_manifest.json",
    "current_event.json",
    "event_forecast.json",
    "event_settlement.json",
    "event_practice_status.json",
    "historical_monitoring_summary.json",
    "model_summary.json",
)

ARTIFACT_TYPES: dict[str, str] = {
    "dashboard_manifest.json": "dashboard_manifest",
    "current_event.json": "current_event",
    "event_forecast.json": "event_forecast",
    "event_settlement.json": "event_settlement",
    "event_practice_status.json": "event_practice_status",
    "historical_monitoring_summary.json": "historical_monitoring_summary",
    "model_summary.json": "model_summary",
}

LIVE_POLICY_ROLE = "observed_live_policy"
LEGACY_LINEAGE_STATUS = "legacy_noncanonical_event_order"
VALID_LINEAGE_STATUS = "valid_registry_lineage"
SESSION_NAMES = ("FP1", "FP2", "FP3", "Q")


@dataclass(frozen=True)
class DashboardExportSummary:
    """Paths and status produced by dashboard artifact export."""

    status: str
    output_dir: Path
    artifact_paths: tuple[Path, ...]
    current_event: str | None
    lifecycle_state: str
    generation_issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceReadResult:
    """Loaded source artifact frames and payloads."""

    protocol: dict[str, Any]
    registry: pd.DataFrame
    reconciliation: pd.DataFrame
    event_order_integrity: dict[str, Any]
    preflight_summary: dict[str, Any]
    preflight_checks: pd.DataFrame
    preflight_failures: pd.DataFrame
    forecasts: pd.DataFrame
    settlements: pd.DataFrame
    monitoring_summary: dict[str, Any]
    readiness_summary: dict[str, Any]
    backtest_report: dict[str, Any]
    portfolio_summary: dict[str, Any]
    event_metrics: pd.DataFrame
    status_by_event: pd.DataFrame
    live_policy_summary: pd.DataFrame
    shadow_candidate_summary: pd.DataFrame
    manifests: dict[tuple[int, str], dict[str, Any]]
    target_coverages: dict[tuple[int, str], pd.DataFrame]
    issues: tuple[str, ...]


@dataclass(frozen=True)
class EventContext:
    """Normalized event-level source state before dashboard projection."""

    identity: EventIdentity
    registry_row: dict[str, Any]
    manifest: dict[str, Any]
    reconciliation_rows: pd.DataFrame
    forecast_rows: pd.DataFrame
    settlement_rows: pd.DataFrame
    target_coverage_rows: pd.DataFrame
    event_metric_rows: pd.DataFrame
    preflight_summary: dict[str, Any] | None
    lifecycle: LifecycleState
    legacy_noncanonical: bool
    eligible_for_valid_prospective_evidence: bool


def export_dashboard_artifacts(
    config: DataConfig,
    *,
    output_dir: Path | None = None,
    season: int | None = None,
    event: str | None = None,
) -> DashboardExportSummary:
    """Export stable dashboard-facing JSON from existing final artifacts only."""
    destination = output_dir or (config.metrics_output_dir.parent / "dashboard")
    ensure_directory(destination)
    generated_at = _utc_now()
    source_artifacts = _build_source_artifacts(config)
    source_paths = tuple(source.path for source in source_artifacts)
    source_fingerprints = {
        source.path: source.to_fingerprint_entry() for source in source_artifacts
    }
    sources = _read_sources(config)
    contexts = _build_event_contexts(sources, season=season, event=event)
    current_event = _select_current_event(contexts, season=season, event=event)
    documents = _build_documents(
        sources=sources,
        contexts=contexts,
        current_event=current_event,
        generated_at=generated_at,
        source_artifacts=source_paths,
        source_fingerprints=source_fingerprints,
    )
    status = _overall_status(documents.values())
    artifact_paths: list[Path] = []
    for filename in DASHBOARD_FILES:
        payload = documents[filename]
        validate_dashboard_document(payload)
        path = destination / filename
        _write_json(path, payload)
        artifact_paths.append(path)
    current_name = (
        current_event.identity.event
        if current_event and current_event.identity.event is not None
        else None
    )
    lifecycle_state = current_event.lifecycle.state if current_event else "no_event_available"
    return DashboardExportSummary(
        status=status,
        output_dir=destination,
        artifact_paths=tuple(artifact_paths),
        current_event=current_name,
        lifecycle_state=lifecycle_state,
        generation_issues=sources.issues,
    )


def _build_documents(
    *,
    sources: SourceReadResult,
    contexts: list[EventContext],
    current_event: EventContext | None,
    generated_at: str,
    source_artifacts: tuple[str, ...],
    source_fingerprints: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    invalid = bool(sources.issues)
    return {
        "dashboard_manifest.json": _envelope(
            "dashboard_manifest",
            generated_at,
            source_artifacts,
            source_fingerprints,
            _manifest_status(contexts, current_event, invalid),
            _manifest_data(contexts, current_event, sources),
        ),
        "current_event.json": _envelope(
            "current_event",
            generated_at,
            source_artifacts,
            source_fingerprints,
            _event_doc_status(current_event, invalid),
            _current_event_data(current_event, sources),
        ),
        "event_forecast.json": _envelope(
            "event_forecast",
            generated_at,
            source_artifacts,
            source_fingerprints,
            _forecast_status(current_event, invalid),
            _forecast_data(current_event),
        ),
        "event_settlement.json": _envelope(
            "event_settlement",
            generated_at,
            source_artifacts,
            source_fingerprints,
            _settlement_status(current_event, invalid),
            _settlement_data(current_event),
        ),
        "event_practice_status.json": _envelope(
            "event_practice_status",
            generated_at,
            source_artifacts,
            source_fingerprints,
            _event_doc_status(current_event, invalid),
            _practice_status_data(current_event, sources),
        ),
        "historical_monitoring_summary.json": _envelope(
            "historical_monitoring_summary",
            generated_at,
            source_artifacts,
            source_fingerprints,
            "invalid" if invalid else "complete",
            _historical_monitoring_data(contexts, sources),
        ),
        "model_summary.json": _envelope(
            "model_summary",
            generated_at,
            source_artifacts,
            source_fingerprints,
            "invalid" if invalid else _model_summary_status(sources),
            _model_summary_data(sources),
        ),
    }


def _envelope(
    artifact_type: str,
    generated_at: str,
    source_artifacts: tuple[str, ...],
    source_fingerprints: dict[str, dict[str, Any]],
    status: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    return DashboardEnvelope(
        artifact_type=artifact_type,
        generated_at_utc=generated_at,
        source_artifacts=source_artifacts,
        source_fingerprints=source_fingerprints,
        status=status,  # type: ignore[arg-type]
        data=_json_safe(data),
    ).to_dict()


def _build_event_contexts(
    sources: SourceReadResult,
    *,
    season: int | None,
    event: str | None,
) -> list[EventContext]:
    identities = _event_identities(sources)
    contexts = [
        _build_event_context(sources, identity)
        for identity in identities
        if _matches_selection(identity, season=season, event=event)
    ]
    return sorted(
        contexts,
        key=lambda context: (
            context.identity.season or -1,
            context.identity.event_order or -1,
            context.identity.event_slug or "",
        ),
    )


def _event_identities(sources: SourceReadResult) -> list[EventIdentity]:
    rows: dict[tuple[int, str], EventIdentity] = {}
    for _, row in sources.registry.iterrows():
        identity = _identity_from_mapping(row.to_dict())
        if identity.season is not None and identity.event_slug:
            rows[(identity.season, identity.event_slug)] = identity
    for frame in (sources.forecasts, sources.settlements):
        for _, row in frame.iterrows():
            identity = _identity_from_mapping(row.to_dict())
            if identity.season is not None and identity.event_slug:
                rows.setdefault((identity.season, identity.event_slug), identity)
    if not rows and sources.preflight_summary:
        identity = _identity_from_mapping(sources.preflight_summary)
        if identity.season is not None and identity.event_slug:
            rows[(identity.season, identity.event_slug)] = identity
    return list(rows.values())


def _identity_from_mapping(mapping: dict[str, Any]) -> EventIdentity:
    season = _optional_int(
        mapping.get("season", mapping.get("monitor_season")),
    )
    event_slug = _optional_str(mapping.get("event_slug"))
    event = _optional_str(mapping.get("event")) or _title_from_slug(event_slug)
    event_order = _optional_int(mapping.get("event_order", mapping.get("registry_event_order")))
    return EventIdentity(
        season=season,
        event=event,
        event_slug=event_slug,
        event_order=event_order,
    )


def _build_event_context(sources: SourceReadResult, identity: EventIdentity) -> EventContext:
    season = identity.season
    slug = identity.event_slug or ""
    registry_row = _matching_row(
        sources.registry,
        season=season,
        event_slug=slug,
        season_column="monitor_season",
    )
    if registry_row:
        identity = _merge_identity(identity, _identity_from_mapping(registry_row))
    manifest = sources.manifests.get((identity.season or -1, slug), {})
    reconciliation = _matching_rows(
        sources.reconciliation,
        season=identity.season,
        event_slug=slug,
        season_column="monitor_season",
    )
    forecasts = _matching_rows(sources.forecasts, season=identity.season, event_slug=slug)
    settlements = _matching_rows(sources.settlements, season=identity.season, event_slug=slug)
    target_coverage = sources.target_coverages.get((identity.season or -1, slug), pd.DataFrame())
    metrics = _matching_rows(sources.event_metrics, season=identity.season, event_slug=slug)
    preflight = (
        sources.preflight_summary
        if _preflight_matches(sources.preflight_summary, identity)
        else None
    )
    legacy = _is_legacy_noncanonical(identity, reconciliation)
    eligible = not legacy
    lifecycle = _resolve_lifecycle(
        identity=identity,
        registry_row=registry_row,
        manifest=manifest,
        reconciliation=reconciliation,
        forecasts=forecasts,
        settlements=settlements,
        preflight=preflight,
        legacy_noncanonical=legacy,
    )
    return EventContext(
        identity=identity,
        registry_row=registry_row,
        manifest=manifest,
        reconciliation_rows=reconciliation,
        forecast_rows=forecasts,
        settlement_rows=settlements,
        target_coverage_rows=target_coverage,
        event_metric_rows=metrics,
        preflight_summary=preflight,
        lifecycle=lifecycle,
        legacy_noncanonical=legacy,
        eligible_for_valid_prospective_evidence=eligible,
    )


def _resolve_lifecycle(
    *,
    identity: EventIdentity,
    registry_row: dict[str, Any],
    manifest: dict[str, Any],
    reconciliation: pd.DataFrame,
    forecasts: pd.DataFrame,
    settlements: pd.DataFrame,
    preflight: dict[str, Any] | None,
    legacy_noncanonical: bool,
) -> LifecycleState:
    if legacy_noncanonical:
        return _lifecycle(
            "legacy_descriptive_only",
            "Legacy descriptive only",
            "event_order_lineage_is_legacy_noncanonical",
        )
    if _has_live_rows(settlements):
        return _lifecycle("settled", "Settled", "forecast_has_post_qualifying_settlement")
    if _has_live_rows(forecasts) and not _target_artifact_present(registry_row, manifest):
        return _lifecycle(
            "awaiting_qualifying_targets",
            "Awaiting qualifying targets",
            "forecast_exists_without_settlement_targets",
        )
    if _has_live_rows(forecasts):
        return _lifecycle("forecast_available", "Forecast available", "forecast_snapshot_exists")
    if preflight and str(preflight.get("status")) == "ready_to_forecast":
        return _lifecycle(
            "ready_to_forecast", "Ready to forecast", "latest_matching_preflight_ready"
        )
    if preflight and _optional_int(preflight.get("blocking_check_count")):
        return _lifecycle(
            "blocked",
            "Blocked",
            f"preflight_status_{preflight.get('status')}",
        )
    if _registry_blocked(registry_row):
        return _lifecycle("blocked", "Blocked", "registry_or_onboarding_blocker_present")
    if registry_row or manifest:
        return _lifecycle(
            "practice_in_progress", "Practice in progress", "monitoring_event_prepared"
        )
    return _lifecycle("no_event_available", "No event available", "no_monitoring_artifacts")


def _lifecycle(state: str, display_label: str, reason: str) -> LifecycleState:
    validate_lifecycle_state(state)
    return LifecycleState(state=state, display_label=display_label, reason=reason)  # type: ignore[arg-type]


def _select_current_event(
    contexts: list[EventContext],
    *,
    season: int | None,
    event: str | None,
) -> EventContext | None:
    if not contexts:
        return None
    if season is not None or event is not None:
        return contexts[-1]
    clean = [context for context in contexts if not context.legacy_noncanonical]
    if clean:
        active = [
            context
            for context in clean
            if context.lifecycle.state
            in {
                "practice_in_progress",
                "ready_to_forecast",
                "forecast_available",
                "awaiting_qualifying_targets",
                "blocked",
            }
        ]
        return (active or clean)[-1]
    return None


def _manifest_data(
    contexts: list[EventContext],
    current_event: EventContext | None,
    sources: SourceReadResult,
) -> dict[str, Any]:
    eligible = [context for context in contexts if not context.legacy_noncanonical]
    legacy = [context for context in contexts if context.legacy_noncanonical]
    current_ref = (
        {
            "artifact": "current_event.json",
            "event_identity": current_event.identity.to_dict(),
            "lifecycle_state": current_event.lifecycle.state,
        }
        if current_event
        else unavailable("no_event_available")
    )
    return {
        "current_event_reference": current_ref,
        "available_pages": {
            "current_event": available("current_event.json"),
            "forecast": _page_availability(current_event, "forecast"),
            "settlement": _page_availability(current_event, "settlement"),
            "practice_status": available("event_practice_status.json"),
            "historical_monitoring": available("historical_monitoring_summary.json"),
            "model_summary": available("model_summary.json"),
        },
        "event_count": len(contexts),
        "eligible_prospective_event_count": len(eligible),
        "legacy_descriptive_event_count": len(legacy),
        "dashboard_contract_capabilities": {
            "forecast_leaderboard": any(
                _has_live_rows(context.forecast_rows) for context in contexts
            ),
            "forecast_intervals": any(
                _forecast_has_intervals(context.forecast_rows) for context in contexts
            ),
            "practice_checkpoint_status": True,
            "post_qualifying_comparison": any(
                _has_live_rows(context.settlement_rows) for context in contexts
            ),
            "historical_prospective_monitoring": bool(eligible),
            "legacy_descriptive_records": bool(legacy),
            "probability_metrics": False,
            "lap_by_lap_live_updates": False,
        },
        "artifacts": {filename: filename for filename in DASHBOARD_FILES},
        "source_issues": list(sources.issues),
    }


def _current_event_data(
    current_event: EventContext | None,
    sources: SourceReadResult,
) -> dict[str, Any]:
    if current_event is None:
        return _empty_event_data("no_event_available")
    return {
        "event_identity": current_event.identity.to_dict(),
        "lifecycle": current_event.lifecycle.to_dict(),
        "freshness": _freshness(current_event, sources),
        "monitoring_protocol": _monitoring_protocol(sources.protocol),
        "registry_lineage": _registry_lineage(current_event),
        "preflight": _preflight_block(current_event),
        "forecast_status": _forecast_status_block(current_event),
        "settlement_status": _settlement_status_block(current_event),
        "legacy_status": _legacy_status_block(current_event),
        "summary_kpis": _summary_kpis(current_event),
    }


def _forecast_data(current_event: EventContext | None) -> dict[str, Any]:
    if current_event is None:
        return {
            "event_identity": EventIdentity().to_dict(),
            "lifecycle_state": "no_event_available",
            "forecast_metadata": unavailable("no_event_available"),
            "leaderboard": unavailable("forecast_not_available"),
            "qualifying_eligible_forecast_rows": unavailable("forecast_not_available"),
            "forecast_only_rows": unavailable("forecast_not_available"),
            "settlement_evaluable_rows": unavailable("forecast_not_available"),
            "summary": unavailable("forecast_not_available"),
        }
    live = _live_rows(current_event.forecast_rows)
    if live.empty:
        return {
            "event_identity": current_event.identity.to_dict(),
            "lifecycle_state": current_event.lifecycle.state,
            "forecast_metadata": unavailable("forecast_not_available"),
            "leaderboard": unavailable("forecast_not_available"),
            "qualifying_eligible_forecast_rows": unavailable("forecast_not_available"),
            "forecast_only_rows": unavailable("forecast_not_available"),
            "settlement_evaluable_rows": unavailable("forecast_not_available"),
            "summary": unavailable("forecast_not_available"),
        }
    qualifying_rows = _forecast_leaderboard(
        current_event,
        _forecast_population_rows(current_event, live, "qualifying_eligible"),
    )
    forecast_only_rows = _forecast_leaderboard(
        current_event,
        _forecast_population_rows(current_event, live, "forecast_only"),
    )
    settlement_evaluable_rows = _forecast_leaderboard(
        current_event,
        _forecast_population_rows(current_event, live, "settlement_evaluable"),
    )
    return {
        "event_identity": current_event.identity.to_dict(),
        "lifecycle_state": current_event.lifecycle.state,
        "forecast_metadata": _forecast_metadata(current_event, live),
        "leaderboard": qualifying_rows,
        "qualifying_eligible_forecast_rows": qualifying_rows,
        "forecast_only_rows": forecast_only_rows,
        "settlement_evaluable_rows": settlement_evaluable_rows,
        "summary": {
            "forecasted_driver_count": len(qualifying_rows),
            "predicted_pole_driver": qualifying_rows[0]["driver_code"] if qualifying_rows else None,
            "forecast_only_driver_count": len(forecast_only_rows),
            "settlement_evaluable_driver_count": len(settlement_evaluable_rows),
            "checkpoint": _first_non_null(live, "checkpoint"),
            "interval_availability_rate": _interval_availability_rate(live),
        },
    }


def _settlement_data(current_event: EventContext | None) -> dict[str, Any]:
    if current_event is None:
        return {
            "event_identity": EventIdentity().to_dict(),
            "lifecycle_state": "no_event_available",
            "settlement_metadata": unavailable("no_event_available"),
            "summary_metrics": unavailable("settlement_not_available"),
            "driver_comparison": unavailable("settlement_not_available"),
            "interval_diagnostics": unavailable("intervals_not_available"),
        }
    live = _live_rows(current_event.settlement_rows)
    if live.empty:
        return {
            "event_identity": current_event.identity.to_dict(),
            "lifecycle_state": current_event.lifecycle.state,
            "settlement_metadata": unavailable("settlement_not_available"),
            "summary_metrics": unavailable("settlement_not_available"),
            "driver_comparison": unavailable("settlement_not_available"),
            "interval_diagnostics": unavailable("intervals_not_available"),
        }
    comparison = _settlement_comparison(_settlement_evaluable_rows(live))
    audit_rows = _settlement_comparison(live)
    return {
        "event_identity": current_event.identity.to_dict(),
        "lifecycle_state": current_event.lifecycle.state,
        "settlement_metadata": {
            "settled_at_utc": _first_non_null(live, "settled_at_utc"),
            "protocol_name": _first_non_null(live, "protocol_name"),
            "protocol_fingerprint": _first_non_null(live, "protocol_fingerprint"),
            "checkpoint": _first_non_null(live, "checkpoint"),
            "settlement_valid": _all_truthy(live, "settlement_valid"),
        },
        "summary_metrics": _settlement_metrics(comparison, total_rows=len(live)),
        "driver_comparison": comparison,
        "settlement_evaluable_rows": comparison,
        "forecast_only_rows": [
            row for row in audit_rows if not bool(row.get("settlement_evaluable"))
        ],
        "interval_diagnostics": unavailable("intervals_not_available"),
    }


def _practice_status_data(
    current_event: EventContext | None,
    sources: SourceReadResult,
) -> dict[str, Any]:
    if current_event is None:
        return {
            "event_identity": EventIdentity().to_dict(),
            "lifecycle_state": "no_event_available",
            "sessions": _empty_sessions(),
            "monitoring_readiness": unavailable("no_event_available"),
            "preflight": unavailable("no_event_available"),
            "notes": ["No monitored event is available."],
        }
    return {
        "event_identity": current_event.identity.to_dict(),
        "lifecycle_state": current_event.lifecycle.state,
        "sessions": _sessions(current_event),
        "monitoring_readiness": {
            "status": sources.readiness_summary.get("status"),
            "forecastable_event_count": sources.readiness_summary.get("forecastable_event_count"),
            "settleable_event_count": sources.readiness_summary.get("settleable_event_count"),
            "target_isolation_status": sources.readiness_summary.get("target_isolation_status"),
            "chronological_order_status": sources.readiness_summary.get(
                "chronological_order_status"
            ),
        },
        "preflight": _preflight_block(current_event),
        "notes": [
            "Session availability is artifact-based and does not represent live telemetry.",
            "Q availability reflects target artifact availability after qualifying.",
        ],
    }


def _historical_monitoring_data(
    contexts: list[EventContext],
    sources: SourceReadResult,
) -> dict[str, Any]:
    valid = [context for context in contexts if not context.legacy_noncanonical]
    legacy = [context for context in contexts if context.legacy_noncanonical]
    return {
        "valid_prospective_monitoring": {
            "event_count": len(valid),
            "settled_event_count": sum(
                _has_live_rows(context.settlement_rows) for context in valid
            ),
            "forecasted_event_count": sum(
                _has_live_rows(context.forecast_rows) for context in valid
            ),
            "aggregate_metrics": _valid_aggregate_metrics(valid),
            "events": [_historical_event_row(context) for context in valid],
        },
        "legacy_descriptive_records": [_legacy_event_row(context) for context in legacy],
        "backtest_context": {
            "available": bool(sources.backtest_report),
            "preferred_backtest_strategy": sources.backtest_report.get(
                "preferred_backtest_strategy"
            ),
            "n_events": sources.backtest_report.get("n_events"),
            "n_folds_successful": sources.backtest_report.get("n_folds_successful"),
            "champion_selection_mode": sources.backtest_report.get("champion_selection_mode"),
        },
    }


def _model_summary_data(sources: SourceReadResult) -> dict[str, Any]:
    backtest = sources.backtest_report
    portfolio = sources.portfolio_summary
    protocol = sources.protocol
    return {
        "model_status": {
            "training_status": backtest.get("training_status"),
            "champion_available": backtest.get("champion_available"),
            "champion_selection_mode": backtest.get("champion_selection_mode"),
            "portfolio_project_status": portfolio.get("project_status"),
        },
        "current_policy_summary": {
            "protocol_name": protocol.get("protocol_name"),
            "policy_recommendation": protocol.get("policy_recommendation")
            or backtest.get("prospective_monitoring_policy_recommendation"),
            "candidate_identity": protocol.get("candidate_identity"),
            "default_identity": protocol.get("default_identity"),
            "gates_are_dashboard_read_only": True,
        },
        "supported_forecast_outputs": {
            "predicted_ranking": True,
            "predicted_gap_to_pole": True,
            "q3_probability": False,
            "probability_metrics": False,
            "lap_by_lap_live_updates": False,
        },
        "uncertainty_summary": {
            "monitoring_uncertainty_method": protocol.get("uncertainty_configuration", {}).get(
                "method"
            )
            if isinstance(protocol.get("uncertainty_configuration"), dict)
            else None,
            "backtest_interval_coverage_by_checkpoint": backtest.get(
                "champion_interval_coverage_by_checkpoint"
            ),
            "backtest_interval_width_by_checkpoint": backtest.get(
                "champion_interval_width_by_checkpoint"
            ),
            "monitoring_forecast_intervals_available": False,
        },
        "backtest_summary": {
            "dataset_rows": backtest.get("dataset_rows"),
            "n_events": backtest.get("n_events"),
            "n_drivers": backtest.get("n_drivers"),
            "checkpoints": backtest.get("checkpoints"),
            "preferred_backtest_strategy": backtest.get("preferred_backtest_strategy"),
            "best_champion_selection_mode_overall": backtest.get(
                "best_champion_selection_mode_overall"
            ),
        },
        "limitations": portfolio.get("limitations", []),
        "data_source_summary": {
            "primary_source": "FastF1 public historical data",
            "paid_apis_required": False,
            "private_team_data_used": False,
            "model_card_available": (Path("reports/model_card.md").is_file()),
        },
    }


def _manifest_status(
    contexts: list[EventContext],
    current_event: EventContext | None,
    invalid: bool,
) -> str:
    if invalid:
        return "invalid"
    if current_event is None or not contexts:
        return "empty"
    if not any(_has_live_rows(context.forecast_rows) for context in contexts):
        return "partial"
    return "complete"


def _event_doc_status(current_event: EventContext | None, invalid: bool) -> str:
    if invalid:
        return "invalid"
    return "empty" if current_event is None else "complete"


def _forecast_status(current_event: EventContext | None, invalid: bool) -> str:
    if invalid:
        return "invalid"
    if current_event is None:
        return "empty"
    return "complete" if _has_live_rows(current_event.forecast_rows) else "partial"


def _settlement_status(current_event: EventContext | None, invalid: bool) -> str:
    if invalid:
        return "invalid"
    if current_event is None:
        return "empty"
    return "complete" if _has_live_rows(current_event.settlement_rows) else "partial"


def _model_summary_status(sources: SourceReadResult) -> str:
    return "complete" if sources.backtest_report and sources.portfolio_summary else "partial"


def _overall_status(documents: Any) -> str:
    statuses = [document["status"] for document in documents]
    for status in ("invalid", "empty", "partial"):
        if status in statuses:
            return status
    return "complete"


def _build_source_artifacts(config: DataConfig) -> tuple[SourceArtifact, ...]:
    root = config.project_root
    paths = [
        config.metrics_output_dir / "prospective_monitoring_protocol.json",
        config.metrics_output_dir / "prospective_monitoring_event_registry.csv",
        config.metrics_output_dir / "prospective_monitoring_event_order_reconciliation.csv",
        config.metrics_output_dir / "prospective_monitoring_event_order_integrity_summary.json",
        config.metrics_output_dir / "prospective_monitoring_preflight_summary.json",
        config.metrics_output_dir / "prospective_monitoring_preflight_checks.csv",
        config.metrics_output_dir / "prospective_monitoring_preflight_failures.csv",
        config.metrics_output_dir / "prospective_monitoring_forecasts.parquet",
        config.metrics_output_dir / "prospective_monitoring_settlements.parquet",
        config.metrics_output_dir / "prospective_monitoring_summary.json",
        config.metrics_output_dir / "monitoring_data_readiness_summary.json",
        config.metrics_output_dir / "prospective_monitoring_event_metrics.csv",
        config.metrics_output_dir / "prospective_monitoring_status_by_event.csv",
        config.metrics_output_dir / "prospective_monitoring_live_policy_summary.csv",
        config.metrics_output_dir / "prospective_monitoring_shadow_candidate_summary.csv",
        config.metrics_output_dir / "backtest_report.json",
        config.metrics_output_dir / "portfolio_summary.json",
        root / "reports/model_card.md",
    ]
    paths.extend(
        sorted((root / "data/processed/monitoring").glob("*/*/monitoring_event_manifest.json"))
    )
    paths.extend(
        sorted((root / "data/processed/monitoring").glob("*/*/monitoring_target_coverage.csv"))
    )
    artifacts = []
    for path in paths:
        available_on_disk = path.is_file()
        artifacts.append(
            SourceArtifact(
                path=_project_relative(path, root),
                available=available_on_disk,
                required=False,
                sha256=_sha256(path) if available_on_disk else None,
                reason=None if available_on_disk else "missing_optional_source_artifact",
            )
        )
    return tuple(artifacts)


def _read_sources(config: DataConfig) -> SourceReadResult:
    metrics = config.metrics_output_dir
    issues: list[str] = []
    protocol = _read_json(metrics / "prospective_monitoring_protocol.json", issues)
    registry = _read_csv(metrics / "prospective_monitoring_event_registry.csv", issues)
    reconciliation = _read_csv(
        metrics / "prospective_monitoring_event_order_reconciliation.csv", issues
    )
    event_order_integrity = _read_json(
        metrics / "prospective_monitoring_event_order_integrity_summary.json", issues
    )
    preflight_summary = _read_json(
        metrics / "prospective_monitoring_preflight_summary.json", issues
    )
    preflight_checks = _read_csv(metrics / "prospective_monitoring_preflight_checks.csv", issues)
    preflight_failures = _read_csv(
        metrics / "prospective_monitoring_preflight_failures.csv", issues
    )
    forecasts = _read_parquet(metrics / "prospective_monitoring_forecasts.parquet", issues)
    settlements = _read_parquet(metrics / "prospective_monitoring_settlements.parquet", issues)
    monitoring_summary = _read_json(metrics / "prospective_monitoring_summary.json", issues)
    readiness_summary = _read_json(metrics / "monitoring_data_readiness_summary.json", issues)
    backtest_report = _read_json(metrics / "backtest_report.json", issues)
    portfolio_summary = _read_json(metrics / "portfolio_summary.json", issues)
    event_metrics = _read_csv(metrics / "prospective_monitoring_event_metrics.csv", issues)
    status_by_event = _read_csv(metrics / "prospective_monitoring_status_by_event.csv", issues)
    live_policy_summary = _read_csv(
        metrics / "prospective_monitoring_live_policy_summary.csv", issues
    )
    shadow_candidate_summary = _read_csv(
        metrics / "prospective_monitoring_shadow_candidate_summary.csv", issues
    )
    manifests = _read_manifests(config.project_root / "data/processed/monitoring", issues)
    target_coverages = _read_target_coverages(
        config.project_root / "data/processed/monitoring",
        issues,
    )
    return SourceReadResult(
        protocol=protocol,
        registry=registry,
        reconciliation=reconciliation,
        event_order_integrity=event_order_integrity,
        preflight_summary=preflight_summary,
        preflight_checks=preflight_checks,
        preflight_failures=preflight_failures,
        forecasts=forecasts,
        settlements=settlements,
        monitoring_summary=monitoring_summary,
        readiness_summary=readiness_summary,
        backtest_report=backtest_report,
        portfolio_summary=portfolio_summary,
        event_metrics=event_metrics,
        status_by_event=status_by_event,
        live_policy_summary=live_policy_summary,
        shadow_candidate_summary=shadow_candidate_summary,
        manifests=manifests,
        target_coverages=target_coverages,
        issues=tuple(issues),
    )


def _read_json(path: Path, issues: list[str]) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(f"malformed_json:{path.name}:{exc.msg}")
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path, issues: list[str]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        issues.append(f"malformed_csv:{path.name}:{exc}")
        return pd.DataFrame()


def _read_parquet(path: Path, issues: list[str]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        issues.append(f"malformed_parquet:{path.name}:{exc}")
        return pd.DataFrame()


def _read_manifests(base_dir: Path, issues: list[str]) -> dict[tuple[int, str], dict[str, Any]]:
    manifests: dict[tuple[int, str], dict[str, Any]] = {}
    for path in sorted(base_dir.glob("*/*/monitoring_event_manifest.json")):
        payload = _read_json(path, issues)
        season = _optional_int(payload.get("season"))
        slug = _optional_str(payload.get("event_slug"))
        if season is not None and slug:
            manifests[(season, slug)] = payload
    return manifests


def _read_target_coverages(
    base_dir: Path,
    issues: list[str],
) -> dict[tuple[int, str], pd.DataFrame]:
    coverages: dict[tuple[int, str], pd.DataFrame] = {}
    for path in sorted(base_dir.glob("*/*/monitoring_target_coverage.csv")):
        frame = _read_csv(path, issues)
        if frame.empty:
            continue
        season = _optional_int(frame["season"].iloc[0]) if "season" in frame else None
        slug = _optional_str(frame["event_slug"].iloc[0]) if "event_slug" in frame else None
        if season is not None and slug:
            coverages[(season, slug)] = frame
    return coverages


def _forecast_leaderboard(
    context: EventContext,
    live_forecasts: pd.DataFrame,
) -> list[dict[str, Any]]:
    frame = live_forecasts.copy()
    frame["_predicted_gap"] = pd.to_numeric(frame.get("prediction_gap_sec"), errors="coerce")
    frame = frame.sort_values(["_predicted_gap", "driver"], na_position="last").reset_index(
        drop=True
    )
    actual_lookup = _settlement_actual_lookup(context.settlement_rows)
    rows = []
    for index, row in frame.iterrows():
        driver_key = _driver_key(row)
        actual = actual_lookup.get(driver_key, {})
        lower = _row_value(row, "prediction_interval_low_sec")
        upper = _row_value(row, "prediction_interval_high_sec")
        interval_available = lower is not None and upper is not None
        population = _driver_population(context, driver_key)
        rows.append(
            {
                "predicted_position": index + 1,
                "driver": _row_value(row, "driver"),
                "driver_code": _row_value(row, "driver"),
                "team": _row_value(row, "team"),
                "team_key": _row_value(row, "team_key"),
                "predicted_gap_to_pole_sec": _number_or_none(row.get("prediction_gap_sec")),
                "interval_lower_sec": _number_or_none(lower),
                "interval_upper_sec": _number_or_none(upper),
                "interval_available": interval_available,
                "selected_method": _selected_method(row),
                "provenance": {
                    "prediction_role": _row_value(row, "prediction_role"),
                    "forecast_id": _row_value(row, "forecast_id"),
                    "source_lineage_valid": _bool_or_none(row.get("source_lineage_valid")),
                    "live_policy_selected": _bool_or_none(row.get("live_policy_selected")),
                    "forecast_integrity_status": _row_value(row, "forecast_integrity_status"),
                },
                "actual_position": actual.get("actual_position"),
                "actual_gap_to_pole_sec": actual.get("actual_gap_to_pole_sec"),
                "absolute_gap_error_sec": actual.get("absolute_gap_error_sec"),
                "forecast_eligible_driver": population["forecast_eligible_driver"],
                "forecast_only_driver": population["forecast_only_driver"],
                "forecast_only_reason": population["forecast_only_reason"],
                "settlement_evaluable_driver": population["settlement_evaluable_driver"],
            }
        )
    return rows


def _forecast_population_rows(
    context: EventContext,
    live_forecasts: pd.DataFrame,
    population: str,
) -> pd.DataFrame:
    if live_forecasts.empty:
        return live_forecasts
    rows = []
    for _, row in live_forecasts.iterrows():
        driver_population = _driver_population(context, _driver_key(row))
        if population == "qualifying_eligible" and driver_population["forecast_eligible_driver"]:
            rows.append(row)
        elif population == "forecast_only" and driver_population["forecast_only_driver"]:
            rows.append(row)
        elif (
            population == "settlement_evaluable"
            and driver_population["settlement_evaluable_driver"]
        ):
            rows.append(row)
    return pd.DataFrame(rows, columns=live_forecasts.columns)


def _driver_population(context: EventContext, driver_key: str) -> dict[str, Any]:
    coverage = _coverage_lookup(context).get(driver_key)
    settlement = _settlement_lookup(context).get(driver_key)
    if coverage is not None:
        target_evaluable = _truthy(coverage.get("target_evaluable"))
        reason = _first_text(
            coverage.get("target_missing_reason"),
            coverage.get("settlement_exclusion_reason"),
        )
    elif settlement is not None:
        target_evaluable = _truthy(settlement.get("settlement_evaluable"))
        reason = _first_text(settlement.get("settlement_exclusion_reason"))
    else:
        target_evaluable = True
        reason = ""
    settlement_evaluable = bool(
        settlement is not None and _truthy(settlement.get("settlement_evaluable"))
    )
    return {
        "forecast_eligible_driver": bool(target_evaluable),
        "forecast_only_driver": bool(not target_evaluable),
        "forecast_only_reason": reason if not target_evaluable else "",
        "settlement_evaluable_driver": settlement_evaluable,
    }


def _coverage_lookup(context: EventContext) -> dict[str, dict[str, Any]]:
    frame = context.target_coverage_rows
    if frame.empty:
        return {}
    return {
        _normalized_driver_key(row.get("driver_key") or row.get("driver")): row.to_dict()
        for _, row in frame.iterrows()
    }


def _settlement_lookup(context: EventContext) -> dict[str, dict[str, Any]]:
    live = _live_rows(context.settlement_rows)
    if live.empty:
        return {}
    return {
        _normalized_driver_key(row.get("driver_key") or row.get("driver")): row.to_dict()
        for _, row in live.iterrows()
    }


def _settlement_evaluable_rows(live_settlements: pd.DataFrame) -> pd.DataFrame:
    if live_settlements.empty or "settlement_evaluable" not in live_settlements:
        return live_settlements
    mask = live_settlements["settlement_evaluable"].map(_truthy)
    return live_settlements[mask]


def _settlement_comparison(live_settlements: pd.DataFrame) -> list[dict[str, Any]]:
    frame = live_settlements.copy()
    frame["_predicted_gap"] = pd.to_numeric(frame.get("prediction_gap_sec"), errors="coerce")
    frame["_actual_gap"] = pd.to_numeric(frame.get("actual_gap_sec"), errors="coerce")
    frame["_predicted_position"] = _rank_series(frame["_predicted_gap"])
    frame["_actual_position"] = _rank_series(frame["_actual_gap"])
    frame = frame.sort_values(["_predicted_position", "driver"], na_position="last")
    rows = []
    for _, row in frame.iterrows():
        predicted = _optional_int(row.get("_predicted_position"))
        actual = _optional_int(row.get("_actual_position"))
        position_error = (
            abs(predicted - actual) if predicted is not None and actual is not None else None
        )
        rows.append(
            {
                "driver": _row_value(row, "driver"),
                "driver_code": _row_value(row, "driver"),
                "predicted_position": predicted,
                "actual_position": actual,
                "predicted_gap_to_pole_sec": _number_or_none(row.get("prediction_gap_sec")),
                "actual_gap_to_pole_sec": _number_or_none(row.get("actual_gap_sec")),
                "absolute_gap_error_sec": _number_or_none(row.get("absolute_error_sec")),
                "absolute_position_error": position_error,
                "included_in_metrics": _bool_or_none(row.get("included_in_metrics")),
                "settlement_evaluable": _bool_or_none(row.get("settlement_evaluable")),
                "settlement_exclusion_reason": _row_value(row, "settlement_exclusion_reason"),
            }
        )
    return rows


def _settlement_metrics(
    comparison: list[dict[str, Any]],
    *,
    total_rows: int | None = None,
) -> dict[str, Any]:
    scored = [row for row in comparison if row.get("included_in_metrics")]
    errors = [
        row["absolute_gap_error_sec"] for row in scored if row["absolute_gap_error_sec"] is not None
    ]
    position_errors = [
        row["absolute_position_error"]
        for row in scored
        if row["absolute_position_error"] is not None
    ]
    actual_pole = min(
        (row for row in scored if row.get("actual_position") is not None),
        key=lambda row: row["actual_position"],
        default={},
    )
    return {
        "driver_count": total_rows if total_rows is not None else len(comparison),
        "scored_driver_count": len(scored),
        "settlement_evaluable_driver_count": len(comparison),
        "excluded_driver_count": max((total_rows or len(comparison)) - len(comparison), 0),
        "mae_gap_sec": _mean(errors),
        "rmse_gap_sec": _rmse(errors),
        "median_absolute_gap_error_sec": _median(errors),
        "mean_absolute_position_error": _mean(position_errors),
        "top_3_agreement": _top_k_agreement(scored, 3),
        "top_5_agreement": _top_k_agreement(scored, 5),
        "top_10_agreement": _top_k_agreement(scored, 10),
        "actual_pole_driver": actual_pole.get("driver_code"),
    }


def _settlement_actual_lookup(settlement_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    live = _settlement_evaluable_rows(_live_rows(settlement_rows))
    if live.empty:
        return {}
    comparison = _settlement_comparison(live)
    return {
        _normalized_driver_key(row.get("driver_code")): {
            "actual_position": row.get("actual_position"),
            "actual_gap_to_pole_sec": row.get("actual_gap_to_pole_sec"),
            "absolute_gap_error_sec": row.get("absolute_gap_error_sec"),
        }
        for row in comparison
    }


def _forecast_metadata(context: EventContext, live: pd.DataFrame) -> dict[str, Any]:
    row = live.iloc[0]
    return {
        "forecast_timestamp": _first_non_null(live, "forecast_created_at_utc"),
        "protocol_name": _row_value(row, "protocol_name"),
        "protocol_fingerprint": _row_value(row, "protocol_fingerprint"),
        "preflight_status": _row_value(row, "preflight_status")
        or (context.preflight_summary or {}).get("status"),
        "preflight_run_id": _row_value(row, "preflight_run_id")
        or (context.preflight_summary or {}).get("preflight_run_id"),
        "preflight_summary_path": _row_value(row, "preflight_summary_path")
        or (context.preflight_summary or {}).get("preflight_summary_path"),
        "checkpoint": _row_value(row, "checkpoint"),
        "prediction_target": "quali_gap_to_pole_sec",
        "candidate_or_policy_identity": {
            "family": _row_value(row, "family"),
            "model_name": _row_value(row, "model_name"),
            "feature_group": _row_value(row, "feature_group"),
            "temporal_weighting_policy": _row_value(row, "temporal_weighting_policy"),
        },
        "uncertainty_method": _row_value(row, "uncertainty_method"),
    }


def _summary_kpis(context: EventContext) -> dict[str, Any]:
    live_forecasts = _live_rows(context.forecast_rows)
    live_settlements = _live_rows(context.settlement_rows)
    eligible_forecasts = _forecast_population_rows(
        context,
        live_forecasts,
        "qualifying_eligible",
    )
    leaderboard = (
        _forecast_leaderboard(context, eligible_forecasts) if not eligible_forecasts.empty else []
    )
    evaluable_settlements = _settlement_evaluable_rows(live_settlements)
    comparison = (
        _settlement_comparison(evaluable_settlements) if not evaluable_settlements.empty else []
    )
    settlement_metrics = _settlement_metrics(comparison) if comparison else {}
    return {
        "forecasted_driver_count": len(leaderboard) if leaderboard else None,
        "predicted_pole_driver": leaderboard[0]["driver_code"] if leaderboard else None,
        "forecast_checkpoint": _first_non_null(live_forecasts, "checkpoint"),
        "interval_availability_rate": _interval_availability_rate(live_forecasts)
        if not live_forecasts.empty
        else None,
        "settlement_mae_gap_sec": settlement_metrics.get("mae_gap_sec"),
        "actual_pole_driver": settlement_metrics.get("actual_pole_driver"),
    }


def _monitoring_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    if not protocol:
        return unavailable("protocol_not_available")
    return {
        "protocol_name": protocol.get("protocol_name"),
        "protocol_version": protocol.get("protocol_version"),
        "protocol_fingerprint": protocol.get("protocol_fingerprint"),
        "monitor_season": protocol.get("monitor_season"),
        "train_seasons": protocol.get("train_seasons"),
        "checkpoint": protocol.get("checkpoint"),
        "policy_recommendation": protocol.get("policy_recommendation"),
    }


def _registry_lineage(context: EventContext) -> dict[str, Any]:
    reconciliation = context.reconciliation_rows
    status = (
        _first_non_null(reconciliation, "event_order_lineage_status")
        if not reconciliation.empty
        else VALID_LINEAGE_STATUS
        if context.registry_row
        else None
    )
    return {
        "event_order": context.identity.event_order,
        "event_order_lineage_status": status,
        "legacy_noncanonical": context.legacy_noncanonical,
        "eligible_for_valid_prospective_evidence": context.eligible_for_valid_prospective_evidence,
        "reconciliation_action": _first_non_null(reconciliation, "reconciliation_action")
        if not reconciliation.empty
        else None,
        "reconciliation_reason": _first_non_null(reconciliation, "reconciliation_reason")
        if not reconciliation.empty
        else None,
    }


def _preflight_block(context: EventContext) -> dict[str, Any]:
    if not context.preflight_summary:
        return unavailable("matching_preflight_not_available")
    payload = context.preflight_summary
    return {
        "available": True,
        "status": payload.get("status"),
        "preflight_run_id": payload.get("preflight_run_id"),
        "forecast_allowed": payload.get("forecast_allowed"),
        "blocking_check_count": payload.get("blocking_check_count"),
        "warning_check_count": payload.get("warning_check_count"),
        "runbook_path": payload.get("prospective_monitoring_preflight_runbook_path"),
        "next_required_command": payload.get("next_required_command"),
    }


def _forecast_status_block(context: EventContext) -> dict[str, Any]:
    live = _live_rows(context.forecast_rows)
    if live.empty:
        return unavailable("forecast_not_available")
    eligible = _forecast_population_rows(context, live, "qualifying_eligible")
    forecast_only = _forecast_population_rows(context, live, "forecast_only")
    return {
        "available": True,
        "forecasted_driver_count": int(len(eligible)),
        "forecast_only_driver_count": int(len(forecast_only)),
        "forecast_created_at_utc": _first_non_null(live, "forecast_created_at_utc"),
        "checkpoint": _first_non_null(live, "checkpoint"),
    }


def _settlement_status_block(context: EventContext) -> dict[str, Any]:
    live = _live_rows(context.settlement_rows)
    if live.empty:
        return unavailable("settlement_not_available")
    included = live["included_in_metrics"].astype(bool) if "included_in_metrics" in live else []
    return {
        "available": True,
        "settled_at_utc": _first_non_null(live, "settled_at_utc"),
        "settlement_valid": _all_truthy(live, "settlement_valid"),
        "scored_driver_count": int(included.sum()) if len(live) else 0,
        "excluded_driver_count": int((~included).sum()) if len(live) else 0,
    }


def _legacy_status_block(context: EventContext) -> dict[str, Any]:
    return {
        "legacy_noncanonical": context.legacy_noncanonical,
        "eligible_for_valid_prospective_evidence": context.eligible_for_valid_prospective_evidence,
        "display_label": "Legacy descriptive only" if context.legacy_noncanonical else "Canonical",
        "reason": context.lifecycle.reason if context.legacy_noncanonical else "",
    }


def _freshness(context: EventContext, sources: SourceReadResult) -> dict[str, Any]:
    return {
        "dashboard_source_generated_at_utc": {
            "monitoring_summary": sources.monitoring_summary.get("generated_at_utc"),
            "monitoring_data_readiness": sources.readiness_summary.get("generated_at_utc"),
            "preflight": (context.preflight_summary or {}).get("generated_at_utc"),
            "forecast": _first_non_null(context.forecast_rows, "forecast_created_at_utc"),
            "settlement": _first_non_null(context.settlement_rows, "settled_at_utc"),
            "manifest": context.manifest.get("created_at_utc"),
            "target": context.manifest.get("target_created_at_utc"),
        },
        "absent_values_are_unavailable_not_zero": True,
    }


def _sessions(context: EventContext) -> list[dict[str, Any]]:
    manifest = context.manifest
    source_availability = manifest.get("source_availability", {})
    sessions = []
    for session in SESSION_NAMES:
        if session == "Q":
            available_flag = bool(
                context.registry_row.get("target_artifact_present")
                or manifest.get("target_artifact_path")
            )
            timestamp = manifest.get("target_created_at_utc")
            reason = (
                "target_artifact_present" if available_flag else "qualifying_target_not_available"
            )
        else:
            available_flag = bool(source_availability.get(session))
            timestamp = manifest.get("feature_created_at_utc") if available_flag else None
            reason = (
                "practice_lap_artifact_available" if available_flag else "practice_artifact_missing"
            )
        sessions.append(
            {
                "session": session,
                "available": available_flag,
                "status": "available" if available_flag else "unavailable",
                "artifact_available": available_flag,
                "last_known_timestamp": timestamp,
                "reason": reason,
            }
        )
    return sessions


def _empty_sessions() -> list[dict[str, Any]]:
    return [
        {
            "session": session,
            "available": False,
            "status": "unavailable",
            "artifact_available": False,
            "last_known_timestamp": None,
            "reason": "no_event_available",
        }
        for session in SESSION_NAMES
    ]


def _valid_aggregate_metrics(contexts: list[EventContext]) -> dict[str, Any]:
    live_frames = [_live_rows(context.settlement_rows) for context in contexts]
    live_frames = [frame for frame in live_frames if not frame.empty]
    if not live_frames:
        return unavailable("no_valid_prospective_settlements")
    frame = pd.concat(live_frames, ignore_index=True)
    included = frame[frame.get("included_in_metrics", pd.Series(dtype=bool)).astype(bool)]
    errors = pd.to_numeric(included.get("absolute_error_sec"), errors="coerce").dropna().tolist()
    return {
        "available": True,
        "event_count": int(included["event_slug"].nunique()) if "event_slug" in included else 0,
        "scored_rows": int(len(included)),
        "mae_gap_sec": _mean(errors),
    }


def _historical_event_row(context: EventContext) -> dict[str, Any]:
    return {
        "event_identity": context.identity.to_dict(),
        "lifecycle_state": context.lifecycle.state,
        "forecasted": _has_live_rows(context.forecast_rows),
        "settled": _has_live_rows(context.settlement_rows),
        "eligible_for_valid_prospective_evidence": True,
        "metrics": _event_metric_summary(context),
    }


def _legacy_event_row(context: EventContext) -> dict[str, Any]:
    return {
        "event_identity": context.identity.to_dict(),
        "lifecycle_state": "legacy_descriptive_only",
        "legacy_noncanonical": True,
        "eligible_for_valid_prospective_evidence": False,
        "exclusion_reason": _first_non_null(
            context.reconciliation_rows,
            "reconciliation_reason",
        )
        or "legacy_noncanonical_event_order",
        "descriptive_metrics": _event_metric_summary(context),
    }


def _event_metric_summary(context: EventContext) -> dict[str, Any]:
    live_metrics = context.event_metric_rows
    if not live_metrics.empty and "diagnostic_only" in live_metrics:
        live_metrics = live_metrics[~live_metrics["diagnostic_only"].astype(bool)]
    if live_metrics.empty:
        return unavailable("event_metrics_not_available")
    row = live_metrics.iloc[0]
    return {
        "available": True,
        "forecast_rows": _optional_int(row.get("forecast_rows")),
        "scored_rows": _optional_int(row.get("scored_rows")),
        "excluded_rows": _optional_int(row.get("excluded_rows")),
        "mae_gap_sec": _number_or_none(row.get("mae_gap_sec")),
    }


def _page_availability(current_event: EventContext | None, page: str) -> dict[str, Any]:
    if current_event is None:
        return unavailable("no_event_available")
    if page == "forecast" and not _has_live_rows(current_event.forecast_rows):
        return unavailable("forecast_not_available")
    if page == "settlement" and not _has_live_rows(current_event.settlement_rows):
        return unavailable("settlement_not_available")
    return available(f"event_{page}.json")


def _empty_event_data(reason: str) -> dict[str, Any]:
    lifecycle = _lifecycle("no_event_available", "No event available", reason)
    return {
        "event_identity": EventIdentity().to_dict(),
        "lifecycle": lifecycle.to_dict(),
        "freshness": {},
        "monitoring_protocol": unavailable("protocol_not_available"),
        "registry_lineage": unavailable("registry_not_available"),
        "preflight": unavailable("preflight_not_available"),
        "forecast_status": unavailable("forecast_not_available"),
        "settlement_status": unavailable("settlement_not_available"),
        "legacy_status": {"legacy_noncanonical": False},
        "summary_kpis": {},
    }


def _matching_rows(
    frame: pd.DataFrame,
    *,
    season: int | None,
    event_slug: str,
    season_column: str = "season",
) -> pd.DataFrame:
    if frame.empty or not event_slug or "event_slug" not in frame:
        return pd.DataFrame(columns=frame.columns)
    result = frame[frame["event_slug"].astype(str).eq(str(event_slug))].copy()
    if season is not None and season_column in result:
        result = result[pd.to_numeric(result[season_column], errors="coerce").eq(int(season))]
    return result


def _matching_row(
    frame: pd.DataFrame,
    *,
    season: int | None,
    event_slug: str,
    season_column: str = "season",
) -> dict[str, Any]:
    rows = _matching_rows(frame, season=season, event_slug=event_slug, season_column=season_column)
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _preflight_matches(payload: dict[str, Any], identity: EventIdentity) -> bool:
    if not payload or identity.event_slug is None:
        return False
    return (
        _optional_int(payload.get("season")) == identity.season
        and _optional_str(payload.get("event_slug")) == identity.event_slug
    )


def _matches_selection(identity: EventIdentity, *, season: int | None, event: str | None) -> bool:
    if season is not None and identity.season != season:
        return False
    if event is not None:
        slug = slugify(event)
        return identity.event_slug == slug or slugify(identity.event or "") == slug
    return True


def _merge_identity(primary: EventIdentity, secondary: EventIdentity) -> EventIdentity:
    return EventIdentity(
        season=primary.season or secondary.season,
        event=primary.event or secondary.event,
        event_slug=primary.event_slug or secondary.event_slug,
        event_order=primary.event_order or secondary.event_order,
    )


def _is_legacy_noncanonical(identity: EventIdentity, reconciliation: pd.DataFrame) -> bool:
    if identity.event_slug in {"australia", "great-britain"} and not reconciliation.empty:
        if reconciliation["event_order_lineage_status"].astype(str).eq(LEGACY_LINEAGE_STATUS).any():
            return True
    if reconciliation.empty or "event_order_lineage_status" not in reconciliation:
        return False
    return reconciliation["event_order_lineage_status"].astype(str).eq(LEGACY_LINEAGE_STATUS).any()


def _registry_blocked(row: dict[str, Any]) -> bool:
    if not row:
        return False
    blocker = _optional_str(row.get("readiness_blocking_reason")) or _optional_str(
        row.get("onboarding_blocking_reason")
    )
    return bool(blocker)


def _target_artifact_present(row: dict[str, Any], manifest: dict[str, Any]) -> bool:
    return bool(row.get("target_artifact_present") or manifest.get("target_artifact_path"))


def _has_live_rows(frame: pd.DataFrame) -> bool:
    return not _live_rows(frame).empty


def _live_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "diagnostic_only" in frame:
        return frame[~frame["diagnostic_only"].fillna(False).astype(bool)].copy()
    if "prediction_role" in frame:
        return frame[frame["prediction_role"].astype(str).eq(LIVE_POLICY_ROLE)].copy()
    return frame.copy()


def _forecast_has_intervals(frame: pd.DataFrame) -> bool:
    live = _live_rows(frame)
    return (
        not live.empty
        and {"prediction_interval_low_sec", "prediction_interval_high_sec"} <= set(live.columns)
        and live["prediction_interval_low_sec"].notna().any()
        and live["prediction_interval_high_sec"].notna().any()
    )


def _interval_availability_rate(frame: pd.DataFrame) -> float | None:
    if frame.empty or not {"prediction_interval_low_sec", "prediction_interval_high_sec"} <= set(
        frame.columns
    ):
        return 0.0 if not frame.empty else None
    available_count = (
        frame["prediction_interval_low_sec"].notna() & frame["prediction_interval_high_sec"].notna()
    ).sum()
    return float(available_count / len(frame)) if len(frame) else None


def _rank_series(values: pd.Series) -> pd.Series:
    return values.rank(method="first", na_option="bottom").where(values.notna()).astype("Int64")


def _top_k_agreement(rows: list[dict[str, Any]], k: int) -> float | None:
    evaluable = [
        row
        for row in rows
        if row.get("predicted_position") is not None and row.get("actual_position") is not None
    ]
    if not evaluable:
        return None
    denominator = min(k, len(evaluable))
    predicted = {
        row["driver_code"] for row in evaluable if int(row["predicted_position"]) <= denominator
    }
    actual = {row["driver_code"] for row in evaluable if int(row["actual_position"]) <= denominator}
    return float(len(predicted & actual) / denominator) if denominator else None


def _selected_method(row: pd.Series) -> dict[str, Any]:
    return {
        "family": _row_value(row, "family"),
        "model_name": _row_value(row, "model_name"),
        "feature_group": _row_value(row, "feature_group"),
        "temporal_weighting_policy": _row_value(row, "temporal_weighting_policy"),
    }


def _driver_key(row: pd.Series) -> str:
    return _normalized_driver_key(_row_value(row, "driver_key") or _row_value(row, "driver"))


def _normalized_driver_key(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def _row_value(row: pd.Series, column: str) -> Any:
    if column not in row:
        return None
    return _clean_scalar(row[column])


def _first_non_null(frame: pd.DataFrame, column: str) -> Any:
    if frame.empty or column not in frame:
        return None
    values = frame[column].dropna()
    return _clean_scalar(values.iloc[0]) if not values.empty else None


def _all_truthy(frame: pd.DataFrame, column: str) -> bool | None:
    if frame.empty or column not in frame:
        return None
    return bool(frame[column].map(_truthy).all())


def _optional_str(value: Any) -> str | None:
    value = _clean_scalar(value)
    if value is None:
        return None
    text = str(value)
    return text if text and text.lower() != "nan" else None


def _optional_int(value: Any) -> int | None:
    value = _clean_scalar(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _number_or_none(value: Any) -> float | None:
    value = _clean_scalar(value)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool_or_none(value: Any) -> bool | None:
    value = _clean_scalar(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    try:
        return bool(value)
    except TypeError:
        return None


def _truthy(value: Any) -> bool:
    parsed = _bool_or_none(value)
    return bool(parsed) if parsed is not None else False


def _first_text(*values: Any) -> str:
    for value in values:
        text = _optional_str(value)
        if text:
            return text
    return ""


def _clean_scalar(value: Any) -> Any:
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            return value
    return value


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _rmse(values: list[float]) -> float | None:
    return (
        float(math.sqrt(sum(value * value for value in values) / len(values))) if values else None
    )


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[midpoint])
    return float((sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2)


def _title_from_slug(slug: str | None) -> str | None:
    return slug.replace("-", " ").title() if slug else None


def _json_safe(value: Any) -> Any:
    value = _clean_scalar(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
