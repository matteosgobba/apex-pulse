"""Qualifying entry-list resolution and forecast-universe auditing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import fastf1
import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.data.fastf1_loader import build_lap_output_path
from f1_prediction.data.identity import normalize_team_key
from f1_prediction.data.ingest import build_metadata_output_path
from f1_prediction.utils.paths import ensure_directory, slugify

ENTRY_LIST_RESOLVED = "resolved"
ENTRY_LIST_UNRESOLVED = "unresolved"
ENTRY_LIST_PARITY_PASSED = "driver_set_parity_passed"
ENTRY_LIST_PARITY_FAILED = "driver_set_parity_failed"

LOCAL_PROCESSED_SOURCE = "processed_entry_list_artifact"
LOCAL_RAW_SOURCE = "official_local_entry_list"
AUTHORITATIVE_ROSTER_SOURCE = "authoritative_race_driver_roster"
LOCAL_Q_METADATA_SOURCE = "local_q_metadata"
LATEST_PRE_Q_SOURCE_PREFIX = "latest_completed_pre_qualifying_session"
FASTF1_Q_RESULTS_SOURCE = "fastf1_q_results"


class EntryDriverResolution(NamedTuple):
    drivers: pd.DataFrame
    source: str
    source_path: Path | None
    reason: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class QualifyingEntryListAudit:
    """Resolved entry-list audit payload and artifact paths."""

    status: str
    forecast_allowed: bool
    summary_path: Path
    drivers_path: Path
    exclusions_path: Path
    failures_path: Path
    summary: dict[str, Any]
    drivers: pd.DataFrame
    exclusions: pd.DataFrame
    failures: pd.DataFrame


def audit_qualifying_entry_list(
    config: DataConfig,
    *,
    season: int,
    event: str,
    event_order: int | None = None,
    feature_rows: pd.DataFrame | None = None,
    forecast_rows: pd.DataFrame | None = None,
    allow_fastf1: bool = True,
) -> QualifyingEntryListAudit:
    """Resolve and validate the qualifying-eligible forecast universe for one event."""
    event_slug = slugify(event)
    generated_at = _utc_now()
    resolution = _resolve_entry_drivers(
        config,
        season=season,
        event=event,
        allow_fastf1=allow_fastf1,
    )
    drivers = resolution.drivers
    practice = _practice_participants(config, season, event)
    features = _normalize_feature_rows(feature_rows)
    forecasts = _live_forecast_rows(forecast_rows)
    failures: list[dict[str, Any]] = []
    exclusions = _practice_exclusions(
        practice,
        entry_driver_keys=set(drivers["driver_key"]) if not drivers.empty else set(),
        entry_team_keys=set(drivers["team_key"].dropna().astype(str))
        if not drivers.empty and "team_key" in drivers
        else set(),
        latest_session=resolution.metadata.get("selected_source_session", ""),
        latest_session_source=bool(resolution.metadata.get("latest_session_source")),
    )
    if drivers.empty:
        failures.append(
            _failure(
                "entry_list_unresolved",
                True,
                resolution.reason or "No authoritative qualifying entry-list source was available.",
                "Provide or refresh an authoritative event entry-list artifact before forecasting.",
            )
        )
    identity_failures = [
        str(reason) for reason in resolution.metadata.get("identity_failure_reasons", [])
    ]
    for reason in identity_failures:
        failures.append(
            _failure(
                "identity_or_team_mismatch",
                True,
                reason,
                "Correct the source session driver/team identity metadata before forecasting.",
            )
        )
    duplicate_count = _duplicate_count(drivers, "driver_key")
    if duplicate_count:
        failures.append(
            _failure(
                "duplicate_eligible_driver",
                True,
                "The resolved qualifying entry list contains duplicate driver identifiers.",
                "Correct the upstream entry-list source and rerun the audit.",
            )
        )
    missing_features = _missing_driver_keys(drivers, features) if feature_rows is not None else []
    missing_checkpoint_features = (
        _missing_latest_checkpoint_feature_keys(drivers, features)
        if feature_rows is not None
        else []
    )
    if missing_features and feature_rows is not None:
        failures.append(
            _failure(
                "eligible_driver_missing_latest_checkpoint_features",
                True,
                f"Missing feature rows for eligible drivers: {', '.join(missing_features)}.",
                "Rebuild practice features after the correct entry list is available.",
            )
        )
    if missing_checkpoint_features:
        failures.append(
            _failure(
                "eligible_driver_missing_latest_checkpoint_features",
                True,
                "Eligible drivers lack compatible latest-checkpoint feature values: "
                f"{', '.join(missing_checkpoint_features)}.",
                "Rebuild leakage-safe latest-checkpoint features before forecasting.",
            )
        )
    feature_extra = _extra_driver_keys(features, drivers) if not features.empty else []
    team_mismatches = _team_mismatches(drivers, features) if not features.empty else []
    if team_mismatches:
        failures.append(
            _failure(
                "driver_team_mapping_mismatch",
                True,
                f"Driver-team mapping mismatch: {', '.join(team_mismatches)}.",
                "Correct the entry-list or feature metadata before forecasting.",
            )
        )
    forecast_missing = _missing_driver_keys(drivers, forecasts) if not forecasts.empty else []
    forecast_extra = _extra_driver_keys(forecasts, drivers) if not forecasts.empty else []
    forecast_duplicate_count = (
        _duplicate_count(forecasts, "driver_key") if not forecasts.empty else 0
    )
    if forecast_extra:
        failures.append(
            _failure(
                "extra_forecast_driver",
                True,
                f"Forecast contains non-entry-list drivers: {', '.join(forecast_extra)}.",
                (
                    "Quarantine the stale forecast and regenerate from the "
                    "entry-list-constrained path."
                ),
            )
        )
    if forecast_missing:
        failures.append(
            _failure(
                "missing_forecast_driver",
                True,
                f"Forecast is missing eligible drivers: {', '.join(forecast_missing)}.",
                "Regenerate the forecast after entry-list enforcement passes.",
            )
        )
    if forecast_duplicate_count:
        failures.append(
            _failure(
                "duplicate_forecast_driver",
                True,
                "Forecast contains duplicate live rows for at least one driver.",
                "Regenerate the forecast without duplicate driver rows.",
            )
        )
    blocking = [row for row in failures if row["blocking"]]
    resolution_status = ENTRY_LIST_RESOLVED if not drivers.empty else ENTRY_LIST_UNRESOLVED
    parity_status = (
        ENTRY_LIST_PARITY_PASSED
        if resolution_status == ENTRY_LIST_RESOLVED and not blocking
        else ENTRY_LIST_PARITY_FAILED
    )
    summary = {
        "season": int(season),
        "event": event,
        "event_slug": event_slug,
        "event_order": event_order,
        "entry_list_resolution_status": resolution_status,
        "driver_set_parity_status": parity_status,
        "resolution_source": resolution.source,
        "resolution_source_path": _project_relative(resolution.source_path, config.project_root)
        if resolution.source_path
        else "",
        "selected_source_session": resolution.metadata.get("selected_source_session", ""),
        "selected_source_session_datetime": resolution.metadata.get(
            "selected_source_session_datetime", ""
        ),
        "selected_source_session_completion_status": resolution.metadata.get(
            "selected_source_session_completion_status", ""
        ),
        "latest_session_participant_count": int(
            resolution.metadata.get("latest_session_participant_count", 0)
        ),
        "earlier_practice_only_exclusion_count": int(
            resolution.metadata.get("earlier_practice_only_exclusion_count", 0)
            or sum(
                str(reason).endswith("_not_in_latest_session")
                or str(reason) == "superseded_before_latest_session"
                for reason in exclusions.get("exclusion_reason", pd.Series(dtype=str))
            )
        ),
        "q_availability_status": resolution.metadata.get("q_availability_status", "unknown"),
        "q_data_available": bool(resolution.metadata.get("q_data_available", False)),
        "q_data_required": bool(resolution.metadata.get("q_data_required", False)),
        "source_precedence_decision_trace": resolution.metadata.get(
            "source_precedence_decision_trace", []
        ),
        "resolution_timestamp_utc": generated_at,
        "entry_list_driver_count": int(drivers["driver_key"].nunique()) if not drivers.empty else 0,
        "practice_participant_count": int(practice["driver_key"].nunique())
        if not practice.empty
        else 0,
        "forecast_driver_count": int(forecasts["driver_key"].nunique())
        if not forecasts.empty
        else 0,
        "excluded_practice_only_driver_count": int(len(exclusions)),
        "missing_eligible_driver_count": int(len(missing_features or forecast_missing)),
        "missing_latest_checkpoint_feature_count": int(len(missing_checkpoint_features)),
        "extra_feature_driver_count": int(len(feature_extra)),
        "extra_forecast_driver_count": int(len(forecast_extra)),
        "duplicate_count": int(duplicate_count + forecast_duplicate_count),
        "team_mapping_mismatch_count": int(len(team_mismatches)),
        "forecast_allowed": bool(resolution_status == ENTRY_LIST_RESOLVED and not blocking),
        "blocking_reasons": [str(row["reason"]) for row in blocking],
        "generated_at_utc": generated_at,
    }
    paths = qualifying_entry_list_artifact_paths(config, season, event)
    ensure_directory(paths["summary"].parent)
    _write_json(paths["summary"], summary)
    drivers.to_csv(paths["drivers"], index=False)
    exclusions.to_csv(paths["exclusions"], index=False)
    failure_frame = pd.DataFrame(failures, columns=_failure_columns())
    failure_frame.to_csv(paths["failures"], index=False)
    return QualifyingEntryListAudit(
        status=parity_status,
        forecast_allowed=bool(summary["forecast_allowed"]),
        summary_path=paths["summary"],
        drivers_path=paths["drivers"],
        exclusions_path=paths["exclusions"],
        failures_path=paths["failures"],
        summary=summary,
        drivers=drivers,
        exclusions=exclusions,
        failures=failure_frame,
    )


def constrain_features_to_entry_list(
    config: DataConfig,
    *,
    season: int,
    event: str,
    event_order: int | None,
    feature_rows: pd.DataFrame,
    allow_fastf1: bool = True,
) -> tuple[pd.DataFrame, QualifyingEntryListAudit]:
    """Return feature rows constrained exactly to qualifying entrants."""
    audit = audit_qualifying_entry_list(
        config,
        season=season,
        event=event,
        event_order=event_order,
        feature_rows=feature_rows,
        allow_fastf1=allow_fastf1,
    )
    if not audit.forecast_allowed:
        reasons = "; ".join(audit.summary.get("blocking_reasons", []))
        raise ValueError(f"Qualifying entry-list audit blocks forecast creation: {reasons}")
    accepted = set(audit.drivers["driver_key"].astype(str))
    features = _normalize_feature_rows(feature_rows)
    constrained = features[features["driver_key"].astype(str).isin(accepted)].copy()
    if constrained["driver_key"].nunique() != len(accepted):
        raise ValueError("Entry-list-constrained features do not match the eligible driver set.")
    return constrained, audit


def qualifying_entry_list_artifact_paths(
    config: DataConfig,
    season: int,
    event: str,
) -> dict[str, Path]:
    base = config.metrics_output_dir / "qualifying_entry_lists" / str(season) / slugify(event)
    return {
        "summary": base / "qualifying_entry_list_summary.json",
        "drivers": base / "qualifying_entry_list_drivers.csv",
        "exclusions": base / "qualifying_entry_list_exclusions.csv",
        "failures": base / "qualifying_entry_list_failures.csv",
    }


def local_entry_list_candidate_paths(
    config: DataConfig,
    season: int,
    event: str,
) -> tuple[Path, ...]:
    slug = slugify(event)
    return (
        config.project_root
        / "data/processed/monitoring"
        / str(season)
        / slug
        / "qualifying_entry_list.csv",
        config.project_root
        / "data/processed/monitoring"
        / str(season)
        / slug
        / "qualifying_entry_list.json",
        config.session_metadata_output_dir / str(season) / slug / "qualifying_entry_list.csv",
        config.session_metadata_output_dir / str(season) / slug / "qualifying_entry_list.json",
    )


def local_race_roster_candidate_paths(
    config: DataConfig,
    season: int,
    event: str,
) -> tuple[Path, ...]:
    slug = slugify(event)
    return (
        config.project_root
        / "data/processed/monitoring"
        / str(season)
        / slug
        / "race_driver_roster.csv",
        config.project_root
        / "data/processed/monitoring"
        / str(season)
        / slug
        / "race_driver_roster.json",
        config.project_root
        / "data/processed/monitoring"
        / str(season)
        / slug
        / "event_race_driver_roster.csv",
        config.project_root
        / "data/processed/monitoring"
        / str(season)
        / slug
        / "event_race_driver_roster.json",
        config.session_metadata_output_dir / str(season) / slug / "race_driver_roster.csv",
        config.session_metadata_output_dir / str(season) / slug / "race_driver_roster.json",
        config.session_metadata_output_dir / str(season) / slug / "event_race_driver_roster.csv",
        config.session_metadata_output_dir / str(season) / slug / "event_race_driver_roster.json",
    )


def _resolve_entry_drivers(
    config: DataConfig,
    *,
    season: int,
    event: str,
    allow_fastf1: bool,
) -> EntryDriverResolution:
    trace: list[dict[str, Any]] = []
    q_status = _q_availability_status(config, season, event)
    for path in local_entry_list_candidate_paths(config, season, event):
        if path.is_file():
            source = LOCAL_PROCESSED_SOURCE if "processed" in path.parts else LOCAL_RAW_SOURCE
            trace.append(_trace(source, "selected", _project_relative(path, config.project_root)))
            return EntryDriverResolution(
                _normalize_entry_rows(_read_entry_artifact(path), season, event, source),
                source,
                path,
                "",
                _resolution_metadata(
                    q_status=q_status,
                    q_required=False,
                    trace=trace,
                ),
            )
    for path in local_race_roster_candidate_paths(config, season, event):
        if path.is_file():
            trace.append(
                _trace(
                    AUTHORITATIVE_ROSTER_SOURCE,
                    "selected",
                    _project_relative(path, config.project_root),
                )
            )
            return EntryDriverResolution(
                _normalize_entry_rows(
                    _read_entry_artifact(path),
                    season,
                    event,
                    AUTHORITATIVE_ROSTER_SOURCE,
                ),
                AUTHORITATIVE_ROSTER_SOURCE,
                path,
                "",
                _resolution_metadata(q_status=q_status, q_required=False, trace=trace),
            )
    q_metadata = build_metadata_output_path(config.session_metadata_output_dir, season, event, "Q")
    if q_metadata.is_file():
        rows = _drivers_from_session_metadata(q_metadata, require_success=True)
        if rows:
            trace.append(
                _trace(
                    LOCAL_Q_METADATA_SOURCE,
                    "selected",
                    _project_relative(q_metadata, config.project_root),
                )
            )
            return EntryDriverResolution(
                _normalize_entry_rows(pd.DataFrame(rows), season, event, LOCAL_Q_METADATA_SOURCE),
                LOCAL_Q_METADATA_SOURCE,
                q_metadata,
                "",
                _resolution_metadata(
                    q_status=q_status,
                    q_required=False,
                    trace=trace,
                ),
            )
        trace.append(
            _trace(
                LOCAL_Q_METADATA_SOURCE,
                "skipped",
                "Q metadata is absent or does not contain successful driver metadata.",
            )
        )
    latest = _resolve_latest_pre_qualifying_session(
        config,
        season=season,
        event=event,
        allow_fastf1=allow_fastf1,
        trace=trace,
        q_status=q_status,
    )
    if not latest.drivers.empty or (
        latest.metadata.get("blocking_latest_session_failure")
        and not q_status.get("available", False)
    ):
        return latest
    q_lap_rows = _drivers_from_verified_q_lap_artifact(config, season, event)
    if q_lap_rows:
        source = LOCAL_Q_METADATA_SOURCE
        trace.append(_trace(source, "selected", "local Q metadata plus Q lap artifact"))
        return EntryDriverResolution(
            _normalize_entry_rows(pd.DataFrame(q_lap_rows), season, event, source),
            source,
            build_metadata_output_path(config.session_metadata_output_dir, season, event, "Q"),
            "",
            _resolution_metadata(q_status=q_status, q_required=True, trace=trace),
        )
    if allow_fastf1:
        try:
            rows = _drivers_from_fastf1_q_results(season, event)
        except Exception as exc:
            trace.append(_trace(FASTF1_Q_RESULTS_SOURCE, "failed", str(exc)))
            return EntryDriverResolution(
                pd.DataFrame(columns=_driver_columns()),
                "",
                None,
                f"FastF1 entry-list lookup failed: {exc}",
                _resolution_metadata(q_status=q_status, q_required=True, trace=trace),
            )
        if rows:
            trace.append(_trace(FASTF1_Q_RESULTS_SOURCE, "selected", "FastF1 Q results available."))
            return EntryDriverResolution(
                _normalize_entry_rows(pd.DataFrame(rows), season, event, FASTF1_Q_RESULTS_SOURCE),
                FASTF1_Q_RESULTS_SOURCE,
                None,
                "",
                _resolution_metadata(
                    q_status={"available": True, "status": "fastf1_q_results_available"},
                    q_required=True,
                    trace=trace,
                ),
            )
        trace.append(_trace(FASTF1_Q_RESULTS_SOURCE, "skipped", "FastF1 Q results unavailable."))
    return EntryDriverResolution(
        pd.DataFrame(columns=_driver_columns()),
        "",
        None,
        "No qualifying entry-list source found.",
        _resolution_metadata(q_status=q_status, q_required=True, trace=trace),
    )


def _read_entry_artifact(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("drivers", [])
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError(f"Qualifying entry-list artifact has invalid driver rows: {path}")
    return pd.DataFrame(rows)


def _drivers_from_session_metadata(
    path: Path,
    *,
    require_success: bool = False,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if require_success and payload.get("status") != "success":
        return []
    explicit = payload.get("qualifying_entry_list") or payload.get("entry_list")
    if isinstance(explicit, list):
        return [row if isinstance(row, dict) else {"driver": row} for row in explicit]
    drivers = payload.get("drivers")
    if not isinstance(drivers, list):
        return []
    return [{"driver": driver} for driver in drivers]


def _resolve_latest_pre_qualifying_session(
    config: DataConfig,
    *,
    season: int,
    event: str,
    allow_fastf1: bool,
    trace: list[dict[str, Any]],
    q_status: dict[str, Any],
) -> EntryDriverResolution:
    session_rows = _local_conventional_pre_q_sessions(config, season, event)
    if session_rows and session_rows[-1]["session_code"] == "FP3":
        trace.append(
            _trace(
                "local_conventional_pre_qualifying_sessions",
                "selected",
                "Completed local FP3 is the conventional pre-qualifying source.",
            )
        )
    else:
        try:
            session_rows = _scheduled_pre_q_sessions(season, event, allow_fastf1=allow_fastf1)
        except Exception as exc:
            session_rows = []
            trace.append(
                _trace(
                    "fastf1_event_schedule",
                    "failed",
                    f"FastF1 schedule lookup failed: {exc}",
                )
            )
        if not session_rows:
            session_rows = _local_conventional_pre_q_sessions(config, season, event)
            if session_rows:
                trace.append(
                    _trace(
                        "local_pre_qualifying_session_order",
                        "diagnostic_fallback",
                        "FastF1 schedule unavailable in offline mode; using local FP order.",
                    )
                )
    if not session_rows:
        trace.append(
            _trace(
                f"{LATEST_PRE_Q_SOURCE_PREFIX}:none",
                "skipped",
                "No scheduled pre-qualifying source session was available.",
            )
        )
        return EntryDriverResolution(
            pd.DataFrame(columns=_driver_columns()),
            "",
            None,
            "",
            _resolution_metadata(q_status=q_status, q_required=True, trace=trace),
        )
    selected = session_rows[-1]
    session_code = str(selected["session_code"])
    session_source = f"{LATEST_PRE_Q_SOURCE_PREFIX}:{session_code}"
    completion = _completed_local_session_drivers(config, season, event, session_code)
    if completion["status"] != "completed":
        reason = (
            f"Selected pre-qualifying source session {session_code} is not completed: "
            f"{completion['status']}."
        )
        trace.append(_trace(session_source, "blocked", reason))
        return EntryDriverResolution(
            pd.DataFrame(columns=_driver_columns()),
            session_source,
            completion.get("path"),
            reason,
            _resolution_metadata(
                q_status=q_status,
                q_required=False,
                trace=trace,
                selected_session=selected,
                completion=completion,
                latest_source=True,
                blocking_latest_session_failure=True,
            ),
        )
    rows = completion["rows"]
    normalized = _normalize_entry_rows(pd.DataFrame(rows), season, event, session_source)
    identity_failures = _source_identity_failures(normalized)
    if not _latest_session_consistent_with_prior_sessions(config, season, event, session_code):
        identity_failures.append(
            "Latest completed pre-qualifying session is not authoritative without a matching "
            "race-driver roster; earlier completed practice sessions contain a different "
            "entrant set."
        )
    status = "blocked" if identity_failures else "selected"
    trace.append(
        _trace(
            session_source,
            status,
            _project_relative(completion["path"], config.project_root),
        )
    )
    reason = "; ".join(identity_failures)
    return EntryDriverResolution(
        normalized,
        session_source,
        completion["path"],
        reason,
        _resolution_metadata(
            q_status=q_status,
            q_required=False,
            trace=trace,
            selected_session=selected,
            completion=completion,
            latest_source=True,
            identity_failure_reasons=identity_failures,
        ),
    )


def _scheduled_pre_q_sessions(
    season: int,
    event: str,
    *,
    allow_fastf1: bool,
) -> list[dict[str, Any]]:
    if not allow_fastf1:
        return []
    schedule = fastf1.get_event_schedule(season, include_testing=False).copy()
    row = _schedule_event_row(schedule, event)
    if row is None:
        return []
    sessions = _schedule_sessions(row)
    q_index = next(
        (index for index, session in enumerate(sessions) if _is_grand_prix_qualifying(session)),
        None,
    )
    if q_index is None:
        return []
    return sessions[:q_index]


def _schedule_event_row(schedule: pd.DataFrame, event: str) -> pd.Series | None:
    requested = slugify(event)
    for _, row in schedule.iterrows():
        aliases = {
            slugify(value)
            for value in (
                _first_text(row.get("EventName")),
                _first_text(row.get("OfficialEventName")),
            )
            if value
        }
        if requested in aliases:
            return row
    return None


def _schedule_sessions(row: pd.Series) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    for index in range(1, 6):
        name = _first_text(row.get(f"Session{index}"))
        if not name:
            continue
        when = _session_datetime(row, index)
        sessions.append(
            {
                "session_name": name,
                "session_code": _session_code(name),
                "session_datetime": when,
                "session_index": index,
            }
        )
    sessions.sort(key=lambda item: (item["session_datetime"] or "", int(item["session_index"])))
    return sessions


def _session_datetime(row: pd.Series, index: int) -> str:
    for column in (f"Session{index}DateUtc", f"Session{index}Date"):
        if column not in row:
            continue
        value = pd.to_datetime(row.get(column), errors="coerce")
        if pd.notna(value):
            return value.isoformat()
    return ""


def _is_grand_prix_qualifying(session: dict[str, Any]) -> bool:
    name = str(session.get("session_name", "")).strip().lower()
    return name == "qualifying"


def _session_code(session_name: str) -> str:
    normalized = slugify(session_name)
    mapping = {
        "practice-1": "FP1",
        "first-practice": "FP1",
        "fp1": "FP1",
        "practice-2": "FP2",
        "second-practice": "FP2",
        "fp2": "FP2",
        "practice-3": "FP3",
        "third-practice": "FP3",
        "fp3": "FP3",
        "sprint": "S",
        "sprint-shootout": "SQ",
        "sprint-qualifying": "SQ",
    }
    return mapping.get(normalized, session_name)


def _local_conventional_pre_q_sessions(
    config: DataConfig,
    season: int,
    event: str,
) -> list[dict[str, Any]]:
    rows = []
    for index, session_code in enumerate(("FP1", "FP2", "FP3"), start=1):
        if build_lap_output_path(config.lap_output_dir, season, event, session_code).is_file():
            rows.append(
                {
                    "session_name": session_code,
                    "session_code": session_code,
                    "session_datetime": "",
                    "session_index": index,
                }
            )
    return rows


def _completed_local_session_drivers(
    config: DataConfig,
    season: int,
    event: str,
    session_code: str,
) -> dict[str, Any]:
    laps_path = build_lap_output_path(config.lap_output_dir, season, event, session_code)
    metadata_path = build_metadata_output_path(
        config.session_metadata_output_dir,
        season,
        event,
        session_code,
    )
    if not laps_path.is_file():
        return {"status": "laps_missing", "path": laps_path, "rows": []}
    if not metadata_path.is_file():
        return {"status": "metadata_missing", "path": laps_path, "rows": []}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "metadata_malformed", "path": laps_path, "rows": []}
    if metadata.get("status") != "success":
        return {"status": "metadata_not_success", "path": laps_path, "rows": []}
    rows = _drivers_from_lap_artifact(laps_path)
    if not rows:
        return {"status": "driver_set_empty", "path": laps_path, "rows": []}
    expected = int(metadata.get("n_drivers") or 0)
    observed = pd.DataFrame(rows)["driver"].nunique()
    if expected and expected != observed:
        return {"status": "participant_count_mismatch", "path": laps_path, "rows": rows}
    return {"status": "completed", "path": laps_path, "rows": rows}


def _source_identity_failures(drivers: pd.DataFrame) -> list[str]:
    if drivers.empty:
        return ["Selected source session produced an empty driver set."]
    failures: list[str] = []
    duplicate_count = _duplicate_count(drivers, "driver_key")
    if duplicate_count:
        failures.append("Selected source session contains duplicate driver identities.")
    team_counts = drivers.groupby("driver_key")["team_key"].nunique(dropna=True)
    inconsistent = sorted(team_counts[team_counts.gt(1)].index.astype(str))
    if inconsistent:
        failures.append(
            "Selected source session contains inconsistent team mappings for: "
            + ", ".join(inconsistent)
            + "."
        )
    if drivers["team_key"].eq("").any():
        missing = sorted(drivers.loc[drivers["team_key"].eq(""), "driver"].astype(str))
        failures.append(
            "Selected source session is missing team identity for: " + ", ".join(missing)
        )
    return failures


def _q_availability_status(config: DataConfig, season: int, event: str) -> dict[str, Any]:
    metadata_path = build_metadata_output_path(
        config.session_metadata_output_dir,
        season,
        event,
        "Q",
    )
    laps_path = build_lap_output_path(config.lap_output_dir, season, event, "Q")
    if metadata_path.is_file():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"available": False, "status": "q_metadata_malformed"}
        if payload.get("status") == "success":
            return {"available": True, "status": "local_q_metadata_available"}
        return {"available": False, "status": "q_metadata_not_success"}
    if laps_path.is_file():
        return {"available": True, "status": "local_q_laps_without_metadata"}
    return {"available": False, "status": "q_not_available"}


def _drivers_from_verified_q_lap_artifact(
    config: DataConfig,
    season: int,
    event: str,
) -> list[dict[str, Any]]:
    metadata_path = build_metadata_output_path(
        config.session_metadata_output_dir,
        season,
        event,
        "Q",
    )
    if not metadata_path.is_file():
        return []
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if metadata.get("status") != "success":
        return []
    return _drivers_from_lap_artifact(
        build_lap_output_path(config.lap_output_dir, season, event, "Q")
    )


def _resolution_metadata(
    *,
    q_status: dict[str, Any],
    q_required: bool,
    trace: list[dict[str, Any]],
    selected_session: dict[str, Any] | None = None,
    completion: dict[str, Any] | None = None,
    latest_source: bool = False,
    blocking_latest_session_failure: bool = False,
    identity_failure_reasons: list[str] | None = None,
) -> dict[str, Any]:
    selected_session = selected_session or {}
    completion = completion or {}
    return {
        "selected_source_session": selected_session.get("session_code", ""),
        "selected_source_session_datetime": selected_session.get("session_datetime", ""),
        "selected_source_session_completion_status": completion.get("status", ""),
        "latest_session_participant_count": len(completion.get("rows", []) or []),
        "q_availability_status": q_status.get("status", "unknown"),
        "q_data_available": bool(q_status.get("available", False)),
        "q_data_required": bool(q_required),
        "source_precedence_decision_trace": trace,
        "latest_session_source": bool(latest_source),
        "blocking_latest_session_failure": bool(blocking_latest_session_failure),
        "identity_failure_reasons": identity_failure_reasons or [],
    }


def _trace(source: str, decision: str, reason: str) -> dict[str, str]:
    return {"source": source, "decision": decision, "reason": reason}


def _drivers_from_lap_artifact(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    laps = pd.read_parquet(path)
    if "Driver" not in laps:
        return []
    rows = []
    for _, row in laps.dropna(subset=["Driver"]).iterrows():
        rows.append(
            {
                "driver": row.get("Driver"),
                "driver_number": row.get("DriverNumber"),
                "full_name": _first_text(row.get("FullName"), row.get("BroadcastName")),
                "team": row.get("Team"),
            }
        )
    return pd.DataFrame(rows).drop_duplicates().to_dict("records") if rows else []


def _drivers_from_fastf1_q_results(season: int, event: str) -> list[dict[str, Any]]:
    session = fastf1.get_session(season, event, "Q")
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    results = getattr(session, "results", pd.DataFrame())
    if results is None or len(results) == 0:
        return []
    frame = pd.DataFrame(results)
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        driver = _first_text(row.get("Abbreviation"), row.get("Driver"), row.get("BroadcastName"))
        if not driver:
            continue
        rows.append(
            {
                "driver": driver,
                "driver_number": _first_text(row.get("DriverNumber"), row.get("Number")),
                "full_name": _first_text(row.get("FullName"), row.get("BroadcastName")),
                "team": _first_text(row.get("TeamName"), row.get("Team")),
            }
        )
    return rows


def _normalize_entry_rows(
    frame: pd.DataFrame,
    season: int,
    event: str,
    source: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_driver_columns())
    rows = []
    for _, row in frame.iterrows():
        driver = _first_text(row.get("driver"), row.get("Driver"), row.get("abbreviation"))
        if not driver:
            continue
        team = _first_text(
            row.get("team"),
            row.get("Team"),
            row.get("team_name"),
            row.get("TeamName"),
        )
        team_key = _first_text(row.get("team_key"), row.get("TeamKey"))
        driver_key = _first_text(row.get("driver_key"), row.get("DriverKey"))
        rows.append(
            {
                "season": int(season),
                "event": event,
                "event_slug": slugify(event),
                "driver": driver.upper(),
                "driver_key": _driver_key(driver_key or driver),
                "driver_number": _first_text(row.get("driver_number"), row.get("DriverNumber")),
                "full_name": _first_text(
                    row.get("full_name"),
                    row.get("FullName"),
                    row.get("name"),
                ),
                "team": team,
                "team_key": _team_key(team_key, team),
                "entry_list_resolution_source": source,
                "entry_classification": "latest_session_qualifying_entrant"
                if source.startswith(LATEST_PRE_Q_SOURCE_PREFIX)
                else "qualifying_entrant",
                "resolution_timestamp_utc": _utc_now(),
                "resolution_status": ENTRY_LIST_RESOLVED,
            }
        )
    return pd.DataFrame(rows, columns=_driver_columns())


def _normalize_feature_rows(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    if "driver_key" not in data:
        data["driver_key"] = data.get("driver", pd.Series(dtype=str)).map(_driver_key)
    else:
        data["driver_key"] = data["driver_key"].fillna(data.get("driver")).map(_driver_key)
    if "team_key" not in data:
        data["team_key"] = data.get("team", pd.Series(dtype=str)).map(
            lambda value: _team_key(value)
        )
    else:
        fallback = data.get("team", pd.Series(pd.NA, index=data.index))
        data["team_key"] = [
            _team_key(team_key, team)
            for team_key, team in zip(data["team_key"].fillna(fallback), fallback, strict=True)
        ]
    return data


def _live_forecast_rows(frame: pd.DataFrame | None) -> pd.DataFrame:
    data = _normalize_feature_rows(frame)
    if data.empty:
        return data
    if "diagnostic_only" in data:
        data = data[~data["diagnostic_only"].astype(bool)].copy()
    return data


def _practice_participants(config: DataConfig, season: int, event: str) -> pd.DataFrame:
    rows = []
    for session in ("FP1", "FP2", "FP3"):
        path = build_lap_output_path(config.lap_output_dir, season, event, session)
        if not path.is_file():
            continue
        laps = pd.read_parquet(path)
        if "Driver" not in laps:
            continue
        for _, row in laps.dropna(subset=["Driver"]).iterrows():
            driver = str(row["Driver"]).upper()
            team = _first_text(row.get("Team"))
            rows.append(
                {
                    "season": int(season),
                    "event": event,
                    "event_slug": slugify(event),
                    "session": session,
                    "driver": driver,
                    "driver_key": _driver_key(driver),
                    "team": team,
                    "team_key": _team_key(team),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=(
                "season",
                "event",
                "event_slug",
                "session",
                "driver",
                "driver_key",
                "team",
                "team_key",
            )
        )
    return pd.DataFrame(rows).drop_duplicates()


def _practice_exclusions(
    practice: pd.DataFrame,
    *,
    entry_driver_keys: set[str],
    entry_team_keys: set[str],
    latest_session: str,
    latest_session_source: bool,
) -> pd.DataFrame:
    if practice.empty:
        return pd.DataFrame(columns=_exclusion_columns())
    rows = []
    grouped = practice.groupby("driver_key", sort=True)
    for driver_key, group in grouped:
        if str(driver_key) in entry_driver_keys:
            continue
        sessions = sorted(set(group["session"].astype(str)))
        team_key = _first_text(
            group.get("team_key", pd.Series(dtype=str)).dropna().astype(str).head(1)
        )
        if latest_session_source:
            reason = "earlier_practice_only_not_in_latest_session"
            if len(sessions) == 1:
                reason = f"{sessions[0].lower()}_only_not_in_latest_session"
            elif team_key and team_key in entry_team_keys:
                reason = "superseded_before_latest_session"
        else:
            reason = "practice_participant_not_in_entry_list"
            if len(sessions) == 1:
                reason = f"{sessions[0].lower()}_only_not_qualifying_eligible"
            elif team_key and team_key in entry_team_keys:
                reason = "superseded_by_qualifying_entrant"
        rows.append(
            {
                "season": int(group["season"].iloc[0]),
                "event": str(group["event"].iloc[0]),
                "event_slug": str(group["event_slug"].iloc[0]),
                "driver": str(group["driver"].iloc[0]),
                "driver_key": str(driver_key),
                "team": _first_text(
                    group.get("team", pd.Series(dtype=str)).dropna().astype(str).head(1)
                ),
                "latest_session": latest_session,
                "sessions": ",".join(sessions),
                "exclusion_reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=_exclusion_columns())


def _missing_driver_keys(expected: pd.DataFrame, observed: pd.DataFrame) -> list[str]:
    if expected.empty:
        return []
    observed_keys = set(observed.get("driver_key", pd.Series(dtype=str)).dropna().astype(str))
    return sorted(set(expected["driver_key"].astype(str)) - observed_keys)


def _extra_driver_keys(observed: pd.DataFrame, expected: pd.DataFrame) -> list[str]:
    if observed.empty:
        return []
    expected_keys = set(expected.get("driver_key", pd.Series(dtype=str)).dropna().astype(str))
    return sorted(set(observed["driver_key"].dropna().astype(str)) - expected_keys)


def _missing_latest_checkpoint_feature_keys(
    expected: pd.DataFrame,
    features: pd.DataFrame,
) -> list[str]:
    if expected.empty or features.empty or "checkpoint" not in features:
        return []
    checkpoint = _first_text(features["checkpoint"].dropna().astype(str).tail(1))
    latest_prefix = {
        "after_fp1": "fp1_",
        "after_fp2": "fp2_",
        "after_fp3": "fp3_",
    }.get(checkpoint)
    if not latest_prefix:
        return []
    latest_columns = [
        column for column in features.columns if str(column).startswith(latest_prefix)
    ]
    if not latest_columns:
        return []
    missing: list[str] = []
    by_key = features.groupby("driver_key", sort=False)
    for _, row in expected.iterrows():
        key = str(row["driver_key"])
        if key not in by_key.groups:
            continue
        driver_features = by_key.get_group(key)
        if driver_features[latest_columns].notna().any(axis=None):
            continue
        missing.append(str(row["driver"]))
    return sorted(missing)


def _latest_session_consistent_with_prior_sessions(
    config: DataConfig,
    season: int,
    event: str,
    latest_session: str,
) -> bool:
    latest_path = build_lap_output_path(config.lap_output_dir, season, event, latest_session)
    latest_keys = {
        _driver_key(value)
        for value in pd.read_parquet(latest_path).get("Driver", pd.Series(dtype=str)).dropna()
    }
    if not latest_keys:
        return False
    for session in ("FP1", "FP2", "FP3", "SQ", "S"):
        if session == latest_session:
            continue
        path = build_lap_output_path(config.lap_output_dir, season, event, session)
        if not path.is_file():
            continue
        keys = {
            _driver_key(value)
            for value in pd.read_parquet(path).get("Driver", pd.Series(dtype=str)).dropna()
        }
        if keys and keys != latest_keys:
            return False
    return True


def _team_mismatches(entry: pd.DataFrame, features: pd.DataFrame) -> list[str]:
    if entry.empty or features.empty or "team_key" not in features:
        return []
    feature_team = features.dropna(subset=["driver_key"]).groupby("driver_key")["team_key"].first()
    mismatches = []
    for _, row in entry.iterrows():
        entry_team = str(row.get("team_key") or "")
        if not entry_team:
            continue
        observed = str(feature_team.get(row["driver_key"], ""))
        if observed and observed != entry_team:
            mismatches.append(f"{row['driver']}:{observed}!={entry_team}")
    return mismatches


def _duplicate_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int(frame[column].duplicated().sum())


def _failure(
    check_name: str,
    blocking: bool,
    reason: str,
    recommended_action: str,
) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "status": "fail" if blocking else "warning",
        "blocking": bool(blocking),
        "reason": reason,
        "recommended_action": recommended_action,
    }


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, pd.Series):
            if value.empty:
                continue
            value = value.iloc[0]
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _driver_key(value: Any) -> str:
    return str(value).strip().lower()


def _team_key(*values: Any) -> str:
    for value in values:
        key = normalize_team_key(value)
        if key:
            return key
    return ""


def _project_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return pd.Timestamp.utcnow().isoformat().replace("+00:00", "Z")


def _driver_columns() -> list[str]:
    return [
        "season",
        "event",
        "event_slug",
        "driver",
        "driver_key",
        "driver_number",
        "full_name",
        "team",
        "team_key",
        "entry_list_resolution_source",
        "entry_classification",
        "resolution_timestamp_utc",
        "resolution_status",
    ]


def _exclusion_columns() -> list[str]:
    return [
        "season",
        "event",
        "event_slug",
        "driver",
        "driver_key",
        "team",
        "latest_session",
        "sessions",
        "exclusion_reason",
    ]


def _failure_columns() -> list[str]:
    return ["check_name", "status", "blocking", "reason", "recommended_action"]
