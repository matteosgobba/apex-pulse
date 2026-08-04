"""One-shot, production-safe Formula 1 weekend orchestration."""

from __future__ import annotations

import fcntl
import json
import os
import socket
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import fastf1
import pandas as pd
import yaml

from f1_prediction.config import DataConfig, FeatureConfig, ModelConfig
from f1_prediction.data.cache import initialize_fastf1_cache
from f1_prediction.modeling.monitoring_operations import (
    DEFAULT_PROTOCOL_NAME,
    MonitoringWorkflowSummary,
    run_monitoring_after_qualifying,
    run_monitoring_before_qualifying,
)
from f1_prediction.utils.paths import slugify

AUTOPILOT_SCHEMA_VERSION = "1.0"
AUTOPILOT_ENABLED_ENV = "APEX_PULSE_AUTOPILOT_ENABLED"
CACHE_WARNING_ENV = "APEX_PULSE_RUNTIME_CACHE_WARNING_MB"
VOLUME_CAPACITY_ENV = "APEX_PULSE_RUNTIME_VOLUME_CAPACITY_MB"
STATUS_FILE = "autopilot_status.json"
RUNS_FILE = "autopilot_runs.jsonl"
LOCK_FILE = "autopilot.lock"
CALENDAR_SOURCE = "fastf1_event_schedule"
SUPPORTED_EVENT_FORMATS = {"conventional"}
REQUIRED_SESSIONS = ("FP1", "FP2", "FP3", "Q")


class OrchestratorState(str, Enum):
    """Explicit one-shot weekend states."""

    NO_EVENT_AVAILABLE = "NO_EVENT_AVAILABLE"
    WAITING_FOR_FP1 = "WAITING_FOR_FP1"
    FP1_TIME_ELAPSED_DATA_PENDING = "FP1_TIME_ELAPSED_DATA_PENDING"
    FP1_COMPLETE = "FP1_COMPLETE"
    WAITING_FOR_FP2 = "WAITING_FOR_FP2"
    FP2_TIME_ELAPSED_DATA_PENDING = "FP2_TIME_ELAPSED_DATA_PENDING"
    FP2_COMPLETE = "FP2_COMPLETE"
    WAITING_FOR_FP3 = "WAITING_FOR_FP3"
    FP3_INITIAL_GRACE = "FP3_INITIAL_GRACE"
    FP3_TIME_ELAPSED_DATA_PENDING = "FP3_TIME_ELAPSED_DATA_PENDING"
    READY_FOR_FORECAST = "READY_FOR_FORECAST"
    FORECAST_AVAILABLE = "FORECAST_AVAILABLE"
    WAITING_FOR_QUALIFYING = "WAITING_FOR_QUALIFYING"
    QUALIFYING_INITIAL_GRACE = "QUALIFYING_INITIAL_GRACE"
    QUALIFYING_TIME_ELAPSED_DATA_PENDING = "QUALIFYING_TIME_ELAPSED_DATA_PENDING"
    READY_FOR_SETTLEMENT = "READY_FOR_SETTLEMENT"
    SETTLED = "SETTLED"
    SETTLED_PARTIAL_COVERAGE = "SETTLED_PARTIAL_COVERAGE"
    UNSUPPORTED_WEEKEND_FORMAT = "UNSUPPORTED_WEEKEND_FORMAT"
    AUTOPILOT_DISABLED = "AUTOPILOT_DISABLED"
    LOCK_CONTENDED = "LOCK_CONTENDED"
    BLOCKED = "BLOCKED"
    TRANSIENT_ERROR = "TRANSIENT_ERROR"


class ProbeStatus(str, Enum):
    """Session data-readiness classifications."""

    READY = "ready"
    NOT_YET_AVAILABLE = "not_yet_available"
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"
    TRANSIENT_ERROR = "transient_error"
    PERMANENT_ERROR = "permanent_error"
    NOT_PROBED = "not_probed"


class ErrorClassification(str, Enum):
    """Operational retry classification."""

    NONE = "none"
    RETRYABLE = "retryable_transient"
    BLOCKING = "blocking_permanent"


@dataclass(frozen=True)
class AutopilotConfig:
    """Conservative one-shot timing and storage diagnostics configuration."""

    practice_initial_grace_minutes: int = 5
    fp3_initial_grace_minutes: int = 15
    qualifying_initial_grace_minutes: int = 15
    retry_interval_minutes: int = 10
    post_qualifying_event_hold_hours: int = 24
    settled_check_interval_minutes: int = 360
    cache_warning_mb: int = 400
    session_default_duration_minutes: Mapping[str, int] = field(
        default_factory=lambda: {"FP1": 60, "FP2": 60, "FP3": 60, "Q": 90}
    )


@dataclass(frozen=True)
class SessionWindow:
    """One normalized UTC session schedule window."""

    code: str
    name: str
    start_utc: datetime
    end_utc: datetime
    end_source: str


@dataclass(frozen=True)
class WeekendEvent:
    """Canonical FastF1 calendar identity and required session windows."""

    season: int
    event: str
    event_slug: str
    round_number: int
    event_format: str
    sessions: Mapping[str, SessionWindow]
    aliases: tuple[str, ...] = ()
    operational_sessions: tuple[SessionWindow, ...] = ()


@dataclass(frozen=True)
class ReadinessProbeResult:
    """Read-only public-session readiness result."""

    status: ProbeStatus
    reason: str
    retryable: bool = False
    lap_count: int | None = None
    driver_count: int | None = None


@dataclass(frozen=True)
class AuthoritativeEventState:
    """Forecast and settlement presence from canonical monitoring tables."""

    forecast_exists: bool
    settlement_exists: bool
    partial_coverage: bool


@dataclass(frozen=True)
class AutopilotTickResult:
    """Complete operational record for one one-shot evaluation."""

    schema_version: str
    run_id: str
    started_at_utc: str
    completed_at_utc: str
    duration_seconds: float
    dry_run: bool
    autopilot_enabled: bool
    season: int | None
    event: str | None
    event_slug: str | None
    round_number: int | None
    event_format: str | None
    orchestrator_state_before: str
    orchestrator_state_after: str
    calendar_source: str
    fp1_status: str
    fp2_status: str
    fp3_status: str
    qualifying_status: str
    forecast_exists: bool
    settlement_exists: bool
    action_considered: str
    action_taken: str
    action_result: str
    retryable: bool
    retry_reason: str | None
    next_recommended_check_at_utc: str | None
    lock_status: str
    error_classification: str
    error_message_safe: str | None
    current_dashboard_lifecycle: str | None
    fastf1_cache_bytes: int
    runtime_total_known_bytes: int
    volume_capacity_bytes: int | None
    cache_warning_status: str
    operational_event: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the stable API/audit representation."""
        return asdict(self)


class ScheduleProvider(Protocol):
    """Injectable calendar provider."""

    def events(self, season: int) -> Sequence[WeekendEvent]: ...


class ReadinessProvider(Protocol):
    """Injectable minimal session readiness provider."""

    def probe(self, event: WeekendEvent, session_code: str) -> ReadinessProbeResult: ...


class FastF1ScheduleProvider:
    """Normalize the public FastF1 event schedule without mutating monitoring state."""

    def __init__(self, config: AutopilotConfig, cache_dir: Path | None = None) -> None:
        self.config = config
        self.cache_dir = cache_dir

    def events(self, season: int) -> Sequence[WeekendEvent]:
        try:
            if self.cache_dir is not None:
                initialize_fastf1_cache(self.cache_dir)
            schedule = fastf1.get_event_schedule(int(season), include_testing=False)
        except Exception as exc:
            raise RuntimeError(f"FastF1 schedule unavailable: {_safe_message(exc)}") from exc
        events: list[WeekendEvent] = []
        for _, row in schedule.iterrows():
            round_number = _integer(row.get("RoundNumber"))
            if round_number is None or round_number < 1:
                continue
            event_name = str(row.get("EventName") or row.get("Location") or "").strip()
            if not event_name:
                continue
            operational_sessions = _operational_session_windows(row, self.config)
            sessions = {
                session.code: session
                for session in operational_sessions
                if session.code in REQUIRED_SESSIONS
            }
            raw_format = str(row.get("EventFormat") or "").strip().lower()
            event_format = raw_format or (
                "conventional" if set(REQUIRED_SESSIONS).issubset(sessions) else "unknown"
            )
            aliases = tuple(
                sorted(
                    {
                        slugify(str(value))
                        for value in (
                            row.get("EventName"),
                            row.get("Location"),
                            row.get("Country"),
                            row.get("OfficialEventName"),
                        )
                        if str(value or "").strip()
                    }
                )
            )
            events.append(
                WeekendEvent(
                    season=int(season),
                    event=event_name,
                    event_slug=slugify(event_name),
                    round_number=round_number,
                    event_format=event_format,
                    sessions=sessions,
                    aliases=aliases,
                    operational_sessions=operational_sessions,
                )
            )
        return tuple(sorted(events, key=lambda item: item.round_number))


class FastF1ReadinessProvider:
    """Probe only lap/results data needed to decide whether a guarded workflow may run."""

    def __init__(self, data_config: DataConfig) -> None:
        self.data_config = data_config

    def probe(self, event: WeekendEvent, session_code: str) -> ReadinessProbeResult:
        try:
            initialize_fastf1_cache(self.data_config.fastf1_cache_dir)
            session = fastf1.get_session(event.season, event.event, session_code)
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            laps = getattr(session, "laps", pd.DataFrame())
            if laps is None or len(laps) == 0:
                return ReadinessProbeResult(
                    ProbeStatus.INCOMPLETE,
                    f"FastF1 {session_code} has no published lap rows yet.",
                    retryable=True,
                    lap_count=0,
                    driver_count=0,
                )
            drivers = _driver_count(laps)
            if session_code == "Q":
                results = getattr(session, "results", pd.DataFrame())
                result_count = len(results) if results is not None else 0
                if result_count < 10:
                    return ReadinessProbeResult(
                        ProbeStatus.INCOMPLETE,
                        "FastF1 qualifying results are not complete enough for target ingestion.",
                        retryable=True,
                        lap_count=len(laps),
                        driver_count=drivers,
                    )
            return ReadinessProbeResult(
                ProbeStatus.READY,
                f"FastF1 {session_code} lap data is available.",
                lap_count=len(laps),
                driver_count=drivers,
            )
        except Exception as exc:
            message = _safe_message(exc)
            if _transient_message(message):
                return ReadinessProbeResult(
                    ProbeStatus.TRANSIENT_ERROR,
                    message,
                    retryable=True,
                )
            return ReadinessProbeResult(
                ProbeStatus.NOT_YET_AVAILABLE,
                message,
                retryable=True,
            )


class FilesystemWriterLock:
    """Process-safe advisory lock; stale metadata never overrides the OS lock state."""

    def __init__(self, path: Path, run_id: str, acquired_at: datetime) -> None:
        self.path = path
        self.run_id = run_id
        self.acquired_at = acquired_at
        self.handle: Any = None
        self.status = "not_attempted"

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.status = "contended"
            self.handle.close()
            self.handle = None
            return False
        previous = self._read_metadata()
        stale = bool(previous and previous.get("released_at_utc") in (None, ""))
        self.status = "acquired_after_stale_metadata" if stale else "acquired"
        self._write_metadata(
            {
                "schema_version": AUTOPILOT_SCHEMA_VERSION,
                "run_id": self.run_id,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "acquired_at_utc": _iso(self.acquired_at),
                "released_at_utc": None,
                "lock_status": self.status,
            }
        )
        return True

    def release(self, released_at: datetime) -> None:
        if self.handle is None:
            return
        self._write_metadata(
            {
                "schema_version": AUTOPILOT_SCHEMA_VERSION,
                "run_id": self.run_id,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "acquired_at_utc": _iso(self.acquired_at),
                "released_at_utc": _iso(released_at),
                "lock_status": "released",
            }
        )
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None

    def _read_metadata(self) -> dict[str, Any]:
        assert self.handle is not None
        self.handle.seek(0)
        try:
            value = json.loads(self.handle.read() or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def _write_metadata(self, value: dict[str, Any]) -> None:
        assert self.handle is not None
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps(value, sort_keys=True) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())


def load_autopilot_config(path: Path) -> AutopilotConfig:
    """Load conservative one-shot configuration from YAML."""
    if not path.is_file():
        raise FileNotFoundError(f"Autopilot configuration does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = payload.get("autopilot", {}) if isinstance(payload, dict) else {}
    durations = raw.get("session_default_duration_minutes", {})
    config = AutopilotConfig(
        practice_initial_grace_minutes=int(raw.get("practice_initial_grace_minutes", 5)),
        fp3_initial_grace_minutes=int(raw.get("fp3_initial_grace_minutes", 15)),
        qualifying_initial_grace_minutes=int(raw.get("qualifying_initial_grace_minutes", 15)),
        retry_interval_minutes=int(raw.get("retry_interval_minutes", 10)),
        post_qualifying_event_hold_hours=int(raw.get("post_qualifying_event_hold_hours", 24)),
        settled_check_interval_minutes=int(raw.get("settled_check_interval_minutes", 360)),
        cache_warning_mb=int(raw.get("cache_warning_mb", 400)),
        session_default_duration_minutes={
            code: int(durations.get(code, default))
            for code, default in {"FP1": 60, "FP2": 60, "FP3": 60, "Q": 90}.items()
        },
    )
    numeric = (
        config.practice_initial_grace_minutes,
        config.fp3_initial_grace_minutes,
        config.qualifying_initial_grace_minutes,
        config.retry_interval_minutes,
        config.post_qualifying_event_hold_hours,
        config.settled_check_interval_minutes,
        config.cache_warning_mb,
        *config.session_default_duration_minutes.values(),
    )
    if any(value <= 0 for value in numeric):
        raise ValueError("Autopilot timing and cache settings must be positive")
    return config


def run_autopilot_tick(
    data_config: DataConfig,
    model_config: ModelConfig,
    feature_config: FeatureConfig,
    *,
    autopilot_config: AutopilotConfig,
    now: datetime | None = None,
    season: int | None = None,
    event: str | None = None,
    dry_run: bool = False,
    protocol_name: str = DEFAULT_PROTOCOL_NAME,
    schedule_provider: ScheduleProvider | None = None,
    readiness_provider: ReadinessProvider | None = None,
    before_workflow: Callable[..., MonitoringWorkflowSummary] = run_monitoring_before_qualifying,
    after_workflow: Callable[..., MonitoringWorkflowSummary] = run_monitoring_after_qualifying,
    environ: Mapping[str, str] | None = None,
    allow_mutation_when_disabled: bool = False,
) -> AutopilotTickResult:
    """Inspect once, perform at most one canonical transition, record, and return."""
    environment = os.environ if environ is None else environ
    instant = _aware_utc(now or datetime.now(timezone.utc))
    started_monotonic = time.monotonic()
    run_id = uuid.uuid4().hex
    enabled = _parse_enabled(environment.get(AUTOPILOT_ENABLED_ENV))
    providers = schedule_provider or FastF1ScheduleProvider(
        autopilot_config,
        data_config.fastf1_cache_dir,
    )
    readiness = readiness_provider or FastF1ReadinessProvider(data_config)
    storage = runtime_storage_diagnostics(data_config, autopilot_config, environment)
    metrics_dir = data_config.metrics_output_dir
    lock = FilesystemWriterLock(metrics_dir / LOCK_FILE, run_id, instant)

    if not dry_run and not enabled and not allow_mutation_when_disabled:
        return _result(
            instant,
            started_monotonic,
            run_id,
            dry_run,
            enabled,
            state=OrchestratorState.AUTOPILOT_DISABLED,
            action_considered="none",
            action_taken="none",
            action_result="autopilot_disabled",
            error_classification=ErrorClassification.BLOCKING,
            error_message=(
                f"{AUTOPILOT_ENABLED_ENV} must be true before a mutating tick is allowed."
            ),
            lock_status="not_attempted",
            storage=storage,
        )

    if not dry_run and not lock.acquire():
        return _result(
            instant,
            started_monotonic,
            run_id,
            dry_run,
            enabled,
            state=OrchestratorState.LOCK_CONTENDED,
            action_considered="none",
            action_taken="none",
            action_result="lock_contended",
            retryable=True,
            retry_reason="another_autopilot_tick_holds_the_writer_lock",
            next_check=instant + timedelta(minutes=autopilot_config.retry_interval_minutes),
            lock_status="contended",
            error_classification=ErrorClassification.RETRYABLE,
            storage=storage,
        )

    try:
        try:
            selected = discover_relevant_event(
                providers,
                instant,
                autopilot_config,
                season_override=season,
                event_override=event,
            )
            result = _evaluate_selected_event(
                data_config,
                model_config,
                feature_config,
                selected,
                readiness,
                autopilot_config,
                instant,
                started_monotonic,
                run_id,
                dry_run,
                enabled,
                protocol_name,
                before_workflow,
                after_workflow,
                "not_required_dry_run" if dry_run else lock.status,
                storage,
            )
        except Exception as exc:
            message = _safe_message(exc)
            retryable = _transient_message(message)
            result = _result(
                instant,
                started_monotonic,
                run_id,
                dry_run,
                enabled,
                state=(
                    OrchestratorState.TRANSIENT_ERROR if retryable else OrchestratorState.BLOCKED
                ),
                action_considered="calendar_discovery",
                action_taken="none",
                action_result="failed",
                retryable=retryable,
                retry_reason=message if retryable else None,
                next_check=(
                    instant + timedelta(minutes=autopilot_config.retry_interval_minutes)
                    if retryable
                    else None
                ),
                lock_status="not_required_dry_run" if dry_run else lock.status,
                error_classification=(
                    ErrorClassification.RETRYABLE if retryable else ErrorClassification.BLOCKING
                ),
                error_message=message,
                storage=storage,
            )
        if not dry_run:
            _write_operational_record(metrics_dir, result)
        return result
    finally:
        if not dry_run:
            lock.release(
                instant + timedelta(seconds=max(0.0, time.monotonic() - started_monotonic))
            )


def discover_relevant_event(
    provider: ScheduleProvider,
    now: datetime,
    config: AutopilotConfig,
    *,
    season_override: int | None = None,
    event_override: str | None = None,
) -> WeekendEvent | None:
    """Select the current/next operational event without modifying the registry."""
    now = _aware_utc(now)
    season = int(season_override or now.year)
    events = tuple(provider.events(season))
    if event_override:
        requested = slugify(event_override)
        matches = [
            item
            for item in events
            if requested in {item.event_slug, *item.aliases}
            or _base_event_slug(requested) == _base_event_slug(item.event_slug)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Event override '{event_override}' did not resolve to exactly one calendar event."
            )
        return matches[0]
    hold = timedelta(hours=config.post_qualifying_event_hold_hours)
    for item in sorted(events, key=lambda value: value.round_number):
        q = item.sessions.get("Q")
        reference = q.end_utc if q else _latest_session_end(item)
        if reference is not None and now <= reference + hold:
            return item
    if season_override is None:
        next_events = tuple(provider.events(season + 1))
        return min(next_events, key=lambda item: item.round_number) if next_events else None
    return None


def inspect_authoritative_event_state(
    data_config: DataConfig,
    *,
    protocol_name: str,
    event: WeekendEvent,
) -> AuthoritativeEventState:
    """Inspect canonical monitoring tables, never dashboard presentation state."""
    metrics = data_config.metrics_output_dir
    forecasts = _read_parquet(metrics / "prospective_monitoring_forecasts.parquet")
    settlements = _read_parquet(metrics / "prospective_monitoring_settlements.parquet")
    registry = _read_csv(metrics / "prospective_monitoring_event_registry.csv")
    forecast_rows = _live_event_rows(forecasts, protocol_name, event.event_slug)
    settlement_rows = _live_event_rows(settlements, protocol_name, event.event_slug)
    if "settlement_valid" in settlement_rows:
        settlement_rows = settlement_rows[settlement_rows["settlement_valid"].astype(bool)]
    partial = False
    if not registry.empty and "event_slug" in registry:
        rows = registry[
            registry["event_slug"].astype(str).eq(event.event_slug)
            & registry.get("protocol_name", pd.Series(dtype=str)).astype(str).eq(protocol_name)
        ]
        if not rows.empty and "partial_target_coverage" in rows:
            partial = _truthy(rows.iloc[-1].get("partial_target_coverage"))
    return AuthoritativeEventState(
        forecast_exists=not forecast_rows.empty,
        settlement_exists=not settlement_rows.empty,
        partial_coverage=partial,
    )


def _evaluate_selected_event(
    data_config: DataConfig,
    model_config: ModelConfig,
    feature_config: FeatureConfig,
    selected: WeekendEvent | None,
    readiness: ReadinessProvider,
    config: AutopilotConfig,
    now: datetime,
    started_monotonic: float,
    run_id: str,
    dry_run: bool,
    enabled: bool,
    protocol_name: str,
    before_workflow: Callable[..., MonitoringWorkflowSummary],
    after_workflow: Callable[..., MonitoringWorkflowSummary],
    lock_status: str,
    storage: dict[str, Any],
) -> AutopilotTickResult:
    if selected is None:
        return _result(
            now,
            started_monotonic,
            run_id,
            dry_run,
            enabled,
            state=OrchestratorState.NO_EVENT_AVAILABLE,
            action_considered="none",
            action_taken="none",
            action_result="no_event_available",
            next_check=now + timedelta(minutes=config.settled_check_interval_minutes),
            lock_status=lock_status,
            storage=storage,
        )
    sessions = selected.sessions
    missing = sorted(set(REQUIRED_SESSIONS) - set(sessions))
    if selected.event_format not in SUPPORTED_EVENT_FORMATS or missing:
        reason = (
            f"event_format={selected.event_format}; missing required sessions={missing}. "
            "Only conventional FP1/FP2/FP3/Q weekends are supported."
        )
        return _event_result(
            now,
            started_monotonic,
            run_id,
            dry_run,
            enabled,
            selected,
            OrchestratorState.UNSUPPORTED_WEEKEND_FORMAT,
            "skip_unsupported_weekend",
            "none",
            "unsupported_weekend_format",
            lock_status,
            storage,
            error_classification=ErrorClassification.BLOCKING,
            error_message=reason,
        )
    authoritative = inspect_authoritative_event_state(
        data_config,
        protocol_name=protocol_name,
        event=selected,
    )
    statuses = _scheduled_statuses(selected, now)
    lifecycle = _dashboard_lifecycle(data_config)
    if authoritative.settlement_exists:
        state = (
            OrchestratorState.SETTLED_PARTIAL_COVERAGE
            if authoritative.partial_coverage
            else OrchestratorState.SETTLED
        )
        return _event_result(
            now,
            started_monotonic,
            run_id,
            dry_run,
            enabled,
            selected,
            state,
            "none",
            "none",
            "existing_settlement_reused",
            lock_status,
            storage,
            authoritative=authoritative,
            statuses=statuses,
            next_check=now + timedelta(minutes=config.settled_check_interval_minutes),
            lifecycle=lifecycle,
        )
    if authoritative.forecast_exists:
        return _evaluate_after_forecast(
            data_config,
            selected,
            authoritative,
            readiness,
            config,
            now,
            started_monotonic,
            run_id,
            dry_run,
            enabled,
            protocol_name,
            after_workflow,
            lock_status,
            storage,
            statuses,
            lifecycle,
        )
    return _evaluate_before_forecast(
        data_config,
        model_config,
        feature_config,
        selected,
        authoritative,
        readiness,
        config,
        now,
        started_monotonic,
        run_id,
        dry_run,
        enabled,
        protocol_name,
        before_workflow,
        lock_status,
        storage,
        statuses,
        lifecycle,
    )


def _evaluate_before_forecast(
    data_config: DataConfig,
    model_config: ModelConfig,
    feature_config: FeatureConfig,
    event: WeekendEvent,
    authoritative: AuthoritativeEventState,
    readiness: ReadinessProvider,
    config: AutopilotConfig,
    now: datetime,
    started_monotonic: float,
    run_id: str,
    dry_run: bool,
    enabled: bool,
    protocol_name: str,
    before_workflow: Callable[..., MonitoringWorkflowSummary],
    lock_status: str,
    storage: dict[str, Any],
    statuses: dict[str, str],
    lifecycle: str | None,
) -> AutopilotTickResult:
    fp1, fp2, fp3, qualifying = (event.sessions[code] for code in REQUIRED_SESSIONS)
    practice_grace = timedelta(minutes=config.practice_initial_grace_minutes)
    fp3_grace = timedelta(minutes=config.fp3_initial_grace_minutes)
    if now < fp1.start_utc:
        return _waiting_event_result(
            event,
            OrchestratorState.WAITING_FOR_FP1,
            fp1.start_utc,
            "wait_for_fp1",
            now,
            started_monotonic,
            run_id,
            dry_run,
            enabled,
            lock_status,
            storage,
            authoritative,
            statuses,
            lifecycle,
        )
    if now < fp2.start_utc:
        if now < fp1.end_utc + practice_grace:
            state = OrchestratorState.WAITING_FOR_FP2
            next_check = fp1.end_utc + practice_grace
        else:
            probe = readiness.probe(event, "FP1")
            statuses["FP1"] = probe.status.value
            if probe.status != ProbeStatus.READY:
                return _probe_pending_result(
                    event,
                    probe,
                    OrchestratorState.FP1_TIME_ELAPSED_DATA_PENDING,
                    "probe_fp1",
                    now,
                    started_monotonic,
                    run_id,
                    dry_run,
                    enabled,
                    lock_status,
                    storage,
                    authoritative,
                    statuses,
                    lifecycle,
                    config,
                )
            state = OrchestratorState.FP1_COMPLETE
            next_check = fp2.start_utc
        return _waiting_event_result(
            event,
            state,
            next_check,
            "wait_for_fp2",
            now,
            started_monotonic,
            run_id,
            dry_run,
            enabled,
            lock_status,
            storage,
            authoritative,
            statuses,
            lifecycle,
        )
    if now < fp3.start_utc:
        if now < fp2.end_utc + practice_grace:
            state = OrchestratorState.WAITING_FOR_FP3
            next_check = fp2.end_utc + practice_grace
        else:
            probe = readiness.probe(event, "FP2")
            statuses["FP2"] = probe.status.value
            if probe.status != ProbeStatus.READY:
                return _probe_pending_result(
                    event,
                    probe,
                    OrchestratorState.FP2_TIME_ELAPSED_DATA_PENDING,
                    "probe_fp2",
                    now,
                    started_monotonic,
                    run_id,
                    dry_run,
                    enabled,
                    lock_status,
                    storage,
                    authoritative,
                    statuses,
                    lifecycle,
                    config,
                )
            state = OrchestratorState.FP2_COMPLETE
            next_check = fp3.start_utc
        return _waiting_event_result(
            event,
            state,
            next_check,
            "wait_for_fp3",
            now,
            started_monotonic,
            run_id,
            dry_run,
            enabled,
            lock_status,
            storage,
            authoritative,
            statuses,
            lifecycle,
        )
    if now < fp3.end_utc + fp3_grace:
        state = (
            OrchestratorState.WAITING_FOR_FP3
            if now < fp3.end_utc
            else OrchestratorState.FP3_INITIAL_GRACE
        )
        return _waiting_event_result(
            event,
            state,
            fp3.end_utc + fp3_grace,
            "wait_for_fp3_data_grace",
            now,
            started_monotonic,
            run_id,
            dry_run,
            enabled,
            lock_status,
            storage,
            authoritative,
            statuses,
            lifecycle,
        )
    if now >= qualifying.start_utc:
        return _event_result(
            now,
            started_monotonic,
            run_id,
            dry_run,
            enabled,
            event,
            OrchestratorState.BLOCKED,
            "run_before_qualifying",
            "none",
            "forecast_window_missed",
            lock_status,
            storage,
            authoritative=authoritative,
            statuses=statuses,
            lifecycle=lifecycle,
            error_classification=ErrorClassification.BLOCKING,
            error_message=(
                "Qualifying has started and no immutable forecast exists; retrospective "
                "forecast creation is forbidden."
            ),
        )
    probe = readiness.probe(event, "FP3")
    statuses["FP3"] = probe.status.value
    if probe.status != ProbeStatus.READY:
        return _probe_pending_result(
            event,
            probe,
            OrchestratorState.FP3_TIME_ELAPSED_DATA_PENDING,
            "probe_fp3",
            now,
            started_monotonic,
            run_id,
            dry_run,
            enabled,
            lock_status,
            storage,
            authoritative,
            statuses,
            lifecycle,
            config,
        )
    if dry_run:
        return _event_result(
            now,
            started_monotonic,
            run_id,
            dry_run,
            enabled,
            event,
            OrchestratorState.READY_FOR_FORECAST,
            "run_before_qualifying",
            "none",
            "would_run_canonical_before_qualifying",
            lock_status,
            storage,
            authoritative=authoritative,
            statuses=statuses,
            lifecycle=lifecycle,
        )
    try:
        summary = before_workflow(
            data_config,
            model_config,
            feature_config,
            season=event.season,
            event=event.event,
            protocol_name=protocol_name,
        )
    except Exception as exc:
        return _workflow_exception_result(
            exc,
            event,
            authoritative,
            "run_before_qualifying",
            OrchestratorState.READY_FOR_FORECAST,
            now,
            started_monotonic,
            run_id,
            dry_run,
            enabled,
            lock_status,
            storage,
            statuses,
            lifecycle,
            config,
        )
    post_workflow_state = inspect_authoritative_event_state(
        data_config,
        protocol_name=protocol_name,
        event=event,
    )
    return _workflow_result(
        summary,
        event,
        "run_before_qualifying",
        OrchestratorState.FORECAST_AVAILABLE,
        OrchestratorState.READY_FOR_FORECAST,
        post_workflow_state,
        now,
        started_monotonic,
        run_id,
        dry_run,
        enabled,
        lock_status,
        storage,
        statuses,
        lifecycle,
        config,
    )


def _evaluate_after_forecast(
    data_config: DataConfig,
    event: WeekendEvent,
    authoritative: AuthoritativeEventState,
    readiness: ReadinessProvider,
    config: AutopilotConfig,
    now: datetime,
    started_monotonic: float,
    run_id: str,
    dry_run: bool,
    enabled: bool,
    protocol_name: str,
    after_workflow: Callable[..., MonitoringWorkflowSummary],
    lock_status: str,
    storage: dict[str, Any],
    statuses: dict[str, str],
    lifecycle: str | None,
) -> AutopilotTickResult:
    qualifying = event.sessions["Q"]
    grace_end = qualifying.end_utc + timedelta(minutes=config.qualifying_initial_grace_minutes)
    if now < grace_end:
        state = (
            OrchestratorState.WAITING_FOR_QUALIFYING
            if now < qualifying.end_utc
            else OrchestratorState.QUALIFYING_INITIAL_GRACE
        )
        return _waiting_event_result(
            event,
            state,
            grace_end,
            "wait_for_qualifying",
            now,
            started_monotonic,
            run_id,
            dry_run,
            enabled,
            lock_status,
            storage,
            authoritative,
            statuses,
            lifecycle,
        )
    probe = readiness.probe(event, "Q")
    statuses["Q"] = probe.status.value
    if probe.status != ProbeStatus.READY:
        return _probe_pending_result(
            event,
            probe,
            OrchestratorState.QUALIFYING_TIME_ELAPSED_DATA_PENDING,
            "probe_qualifying",
            now,
            started_monotonic,
            run_id,
            dry_run,
            enabled,
            lock_status,
            storage,
            authoritative,
            statuses,
            lifecycle,
            config,
        )
    if dry_run:
        return _event_result(
            now,
            started_monotonic,
            run_id,
            dry_run,
            enabled,
            event,
            OrchestratorState.READY_FOR_SETTLEMENT,
            "run_after_qualifying",
            "none",
            "would_run_canonical_after_qualifying",
            lock_status,
            storage,
            authoritative=authoritative,
            statuses=statuses,
            lifecycle=lifecycle,
        )
    try:
        summary = after_workflow(
            data_config,
            season=event.season,
            event=event.event,
            protocol_name=protocol_name,
        )
    except Exception as exc:
        return _workflow_exception_result(
            exc,
            event,
            authoritative,
            "run_after_qualifying",
            OrchestratorState.READY_FOR_SETTLEMENT,
            now,
            started_monotonic,
            run_id,
            dry_run,
            enabled,
            lock_status,
            storage,
            statuses,
            lifecycle,
            config,
        )
    post_workflow_state = inspect_authoritative_event_state(
        data_config,
        protocol_name=protocol_name,
        event=event,
    )
    return _workflow_result(
        summary,
        event,
        "run_after_qualifying",
        OrchestratorState.SETTLED,
        OrchestratorState.READY_FOR_SETTLEMENT,
        post_workflow_state,
        now,
        started_monotonic,
        run_id,
        dry_run,
        enabled,
        lock_status,
        storage,
        statuses,
        lifecycle,
        config,
    )


def _workflow_result(
    summary: MonitoringWorkflowSummary,
    event: WeekendEvent,
    action: str,
    success_state: OrchestratorState,
    state_before: OrchestratorState,
    authoritative: AuthoritativeEventState,
    now: datetime,
    started_monotonic: float,
    run_id: str,
    dry_run: bool,
    enabled: bool,
    lock_status: str,
    storage: dict[str, Any],
    statuses: dict[str, str],
    lifecycle: str | None,
    config: AutopilotConfig,
) -> AutopilotTickResult:
    if summary.completed and not summary.blocking_failure_count:
        final_state = (
            OrchestratorState.SETTLED_PARTIAL_COVERAGE
            if authoritative.partial_coverage
            else success_state
        )
        return _event_result(
            now,
            started_monotonic,
            run_id,
            dry_run,
            enabled,
            event,
            final_state,
            action,
            action,
            "completed",
            lock_status,
            storage,
            authoritative=authoritative,
            statuses=statuses,
            lifecycle=lifecycle,
            state_before=state_before,
            next_check=now + timedelta(minutes=config.settled_check_interval_minutes),
        )
    reason = _workflow_failure_reason(summary)
    retryable = _transient_message(reason)
    return _event_result(
        now,
        started_monotonic,
        run_id,
        dry_run,
        enabled,
        event,
        OrchestratorState.TRANSIENT_ERROR if retryable else OrchestratorState.BLOCKED,
        action,
        action,
        "blocked",
        lock_status,
        storage,
        statuses=statuses,
        lifecycle=lifecycle,
        authoritative=authoritative,
        state_before=state_before,
        retryable=retryable,
        retry_reason=reason if retryable else None,
        next_check=(now + timedelta(minutes=config.retry_interval_minutes) if retryable else None),
        error_classification=(
            ErrorClassification.RETRYABLE if retryable else ErrorClassification.BLOCKING
        ),
        error_message=reason,
    )


def _workflow_exception_result(
    exc: Exception,
    event: WeekendEvent,
    authoritative: AuthoritativeEventState,
    action: str,
    state_before: OrchestratorState,
    now: datetime,
    started_monotonic: float,
    run_id: str,
    dry_run: bool,
    enabled: bool,
    lock_status: str,
    storage: dict[str, Any],
    statuses: dict[str, str],
    lifecycle: str | None,
    config: AutopilotConfig,
) -> AutopilotTickResult:
    message = _safe_message(exc)
    retryable = _transient_message(message)
    return _event_result(
        now,
        started_monotonic,
        run_id,
        dry_run,
        enabled,
        event,
        OrchestratorState.TRANSIENT_ERROR if retryable else OrchestratorState.BLOCKED,
        action,
        action,
        "exception",
        lock_status,
        storage,
        authoritative=authoritative,
        statuses=statuses,
        lifecycle=lifecycle,
        state_before=state_before,
        retryable=retryable,
        retry_reason=message if retryable else None,
        next_check=(now + timedelta(minutes=config.retry_interval_minutes) if retryable else None),
        error_classification=(
            ErrorClassification.RETRYABLE if retryable else ErrorClassification.BLOCKING
        ),
        error_message=message,
    )


def _workflow_failure_reason(summary: MonitoringWorkflowSummary) -> str:
    try:
        payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"Canonical workflow status={summary.status}."
    return str(payload.get("blocking_reason") or payload.get("recommended_operator_action") or "")


def _probe_pending_result(
    event: WeekendEvent,
    probe: ReadinessProbeResult,
    pending_state: OrchestratorState,
    action: str,
    now: datetime,
    started_monotonic: float,
    run_id: str,
    dry_run: bool,
    enabled: bool,
    lock_status: str,
    storage: dict[str, Any],
    authoritative: AuthoritativeEventState,
    statuses: dict[str, str],
    lifecycle: str | None,
    config: AutopilotConfig,
) -> AutopilotTickResult:
    permanent = probe.status in {ProbeStatus.PERMANENT_ERROR, ProbeStatus.UNSUPPORTED}
    state = OrchestratorState.BLOCKED if permanent else pending_state
    return _event_result(
        now,
        started_monotonic,
        run_id,
        dry_run,
        enabled,
        event,
        state,
        action,
        "none",
        probe.status.value,
        lock_status,
        storage,
        authoritative=authoritative,
        statuses=statuses,
        lifecycle=lifecycle,
        retryable=not permanent,
        retry_reason=probe.reason if not permanent else None,
        next_check=(
            now + timedelta(minutes=config.retry_interval_minutes) if not permanent else None
        ),
        error_classification=(
            ErrorClassification.BLOCKING if permanent else ErrorClassification.RETRYABLE
        ),
        error_message=probe.reason,
    )


def _waiting_event_result(
    event: WeekendEvent,
    state: OrchestratorState,
    next_check: datetime,
    action: str,
    now: datetime,
    started_monotonic: float,
    run_id: str,
    dry_run: bool,
    enabled: bool,
    lock_status: str,
    storage: dict[str, Any],
    authoritative: AuthoritativeEventState,
    statuses: dict[str, str],
    lifecycle: str | None,
) -> AutopilotTickResult:
    return _event_result(
        now,
        started_monotonic,
        run_id,
        dry_run,
        enabled,
        event,
        state,
        action,
        "none",
        "waiting",
        lock_status,
        storage,
        authoritative=authoritative,
        statuses=statuses,
        lifecycle=lifecycle,
        next_check=next_check,
    )


def _event_result(
    now: datetime,
    started_monotonic: float,
    run_id: str,
    dry_run: bool,
    enabled: bool,
    event: WeekendEvent,
    state: OrchestratorState,
    action_considered: str,
    action_taken: str,
    action_result: str,
    lock_status: str,
    storage: dict[str, Any],
    *,
    authoritative: AuthoritativeEventState | None = None,
    statuses: Mapping[str, str] | None = None,
    next_check: datetime | None = None,
    retryable: bool = False,
    retry_reason: str | None = None,
    error_classification: ErrorClassification = ErrorClassification.NONE,
    error_message: str | None = None,
    lifecycle: str | None = None,
    state_before: OrchestratorState | None = None,
) -> AutopilotTickResult:
    return _result(
        now,
        started_monotonic,
        run_id,
        dry_run,
        enabled,
        state=state,
        event=event,
        authoritative=authoritative,
        statuses=statuses,
        action_considered=action_considered,
        action_taken=action_taken,
        action_result=action_result,
        retryable=retryable,
        retry_reason=retry_reason,
        next_check=next_check,
        lock_status=lock_status,
        error_classification=error_classification,
        error_message=error_message,
        lifecycle=lifecycle,
        state_before=state_before,
        storage=storage,
    )


def _result(
    now: datetime,
    started_monotonic: float,
    run_id: str,
    dry_run: bool,
    enabled: bool,
    *,
    state: OrchestratorState,
    action_considered: str,
    action_taken: str,
    action_result: str,
    lock_status: str,
    storage: Mapping[str, Any],
    event: WeekendEvent | None = None,
    authoritative: AuthoritativeEventState | None = None,
    statuses: Mapping[str, str] | None = None,
    retryable: bool = False,
    retry_reason: str | None = None,
    next_check: datetime | None = None,
    error_classification: ErrorClassification = ErrorClassification.NONE,
    error_message: str | None = None,
    lifecycle: str | None = None,
    state_before: OrchestratorState | None = None,
) -> AutopilotTickResult:
    duration = max(0.0, time.monotonic() - started_monotonic)
    completed = now + timedelta(seconds=duration)
    values = statuses or {}
    source = authoritative or AuthoritativeEventState(False, False, False)
    return AutopilotTickResult(
        schema_version=AUTOPILOT_SCHEMA_VERSION,
        run_id=run_id,
        started_at_utc=_iso(now),
        completed_at_utc=_iso(completed),
        duration_seconds=round(duration, 6),
        dry_run=dry_run,
        autopilot_enabled=enabled,
        season=event.season if event else None,
        event=event.event if event else None,
        event_slug=event.event_slug if event else None,
        round_number=event.round_number if event else None,
        event_format=event.event_format if event else None,
        orchestrator_state_before=(state_before or state).value,
        orchestrator_state_after=state.value,
        calendar_source=CALENDAR_SOURCE,
        fp1_status=values.get("FP1", ProbeStatus.NOT_PROBED.value),
        fp2_status=values.get("FP2", ProbeStatus.NOT_PROBED.value),
        fp3_status=values.get("FP3", ProbeStatus.NOT_PROBED.value),
        qualifying_status=values.get("Q", ProbeStatus.NOT_PROBED.value),
        forecast_exists=source.forecast_exists,
        settlement_exists=source.settlement_exists,
        action_considered=action_considered,
        action_taken=action_taken,
        action_result=action_result,
        retryable=retryable,
        retry_reason=retry_reason,
        next_recommended_check_at_utc=_iso(next_check) if next_check else None,
        lock_status=lock_status,
        error_classification=error_classification.value,
        error_message_safe=error_message,
        current_dashboard_lifecycle=lifecycle,
        fastf1_cache_bytes=int(storage["fastf1_cache_bytes"]),
        runtime_total_known_bytes=int(storage["runtime_total_known_bytes"]),
        volume_capacity_bytes=storage["volume_capacity_bytes"],
        cache_warning_status=str(storage["cache_warning_status"]),
        operational_event=_operational_event_snapshot(event, now) if event else None,
    )


def _write_operational_record(metrics_dir: Path, result: AutopilotTickResult) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    status_path = metrics_dir / STATUS_FILE
    temporary = status_path.with_name(f".{status_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, status_path)
    with (metrics_dir / RUNS_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_autopilot_status(payload: Any) -> dict[str, Any]:
    """Validate the separate operational status schema used by the read-only API."""
    required = set(AutopilotTickResult.__dataclass_fields__) - {"operational_event"}
    if not isinstance(payload, dict) or required - set(payload):
        raise ValueError("Autopilot status schema is incomplete")
    if payload.get("schema_version") != AUTOPILOT_SCHEMA_VERSION:
        raise ValueError("Unsupported autopilot status schema")
    if payload.get("orchestrator_state_after") not in {state.value for state in OrchestratorState}:
        raise ValueError("Invalid autopilot state")
    if payload.get("operational_event") is not None:
        _validate_operational_event(payload["operational_event"])
    return payload


def _operational_session_windows(
    row: pd.Series,
    config: AutopilotConfig,
) -> tuple[SessionWindow, ...]:
    result: list[SessionWindow] = []
    for index in range(1, 6):
        name = str(row.get(f"Session{index}") or "").strip()
        code = _operational_session_code(name)
        if code is None:
            continue
        start = _row_datetime(row, f"Session{index}DateUtc", f"Session{index}Date")
        if start is None:
            continue
        explicit_end = _row_datetime(
            row,
            f"Session{index}EndDateUtc",
            f"Session{index}EndDate",
        )
        if explicit_end is None:
            explicit_end = start + timedelta(minutes=_default_session_duration(code, config))
            source = "configured_default_duration"
        else:
            source = "fastf1_schedule"
        result.append(SessionWindow(code, name, start, explicit_end, source))
    return tuple(result)


def _operational_session_code(name: str) -> str | None:
    normalized = name.strip().lower().replace("free practice", "practice")
    return {
        "practice 1": "FP1",
        "practice 2": "FP2",
        "practice 3": "FP3",
        "qualifying": "Q",
        "sprint qualifying": "SQ",
        "sprint shootout": "SQ",
        "sprint": "S",
    }.get(normalized)


def _default_session_duration(code: str, config: AutopilotConfig) -> int:
    if code in config.session_default_duration_minutes:
        return int(config.session_default_duration_minutes[code])
    return 60


def _row_datetime(row: pd.Series, *columns: str) -> datetime | None:
    for column in columns:
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.notna(parsed):
            return parsed.to_pydatetime().astimezone(timezone.utc)
    return None


def _scheduled_statuses(event: WeekendEvent, now: datetime) -> dict[str, str]:
    result: dict[str, str] = {}
    for code, window in event.sessions.items():
        if now < window.start_utc:
            result[code] = "scheduled"
        elif now < window.end_utc:
            result[code] = "scheduled_window_open_data_not_proven"
        else:
            result[code] = "scheduled_time_elapsed_data_not_proven"
    return result


def _operational_event_snapshot(event: WeekendEvent, now: datetime) -> dict[str, Any]:
    sessions = event.operational_sessions or tuple(
        sorted(event.sessions.values(), key=lambda item: item.start_utc)
    )
    missing = sorted(set(REQUIRED_SESSIONS) - set(event.sessions))
    supported = event.event_format in SUPPORTED_EVENT_FORMATS and not missing
    reason = None
    if not supported:
        reason = (
            "Apex Pulse predictions currently require a conventional FP1, FP2, FP3 and "
            "qualifying weekend. This format remains visible but is not forecast-supported."
        )
    return {
        "season": event.season,
        "event": event.event,
        "event_slug": event.event_slug,
        "round_number": event.round_number,
        "event_format": event.event_format,
        "calendar_source": CALENDAR_SOURCE,
        "supported": supported,
        "prediction_support_reason": reason,
        "schedule_available": bool(sessions),
        "timezone": "UTC",
        "sessions": [
            {
                "sequence": sequence,
                "session": session.code,
                "display_name": session.name,
                "scheduled_start_utc": _iso(session.start_utc),
                "scheduled_end_utc": _iso(session.end_utc),
                "end_source": session.end_source,
                "schedule_status": _calendar_session_status(session, now),
            }
            for sequence, session in enumerate(sessions, start=1)
        ],
    }


def _calendar_session_status(session: SessionWindow, now: datetime) -> str:
    if now < session.start_utc:
        return "scheduled"
    if now < session.end_utc:
        return "calendar_window_open_data_not_proven"
    return "calendar_time_elapsed_data_not_proven"


def _validate_operational_event(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("Operational event snapshot must be an object")
    required = {
        "season",
        "event",
        "event_slug",
        "round_number",
        "event_format",
        "calendar_source",
        "supported",
        "schedule_available",
        "timezone",
        "sessions",
    }
    if required - set(value):
        raise ValueError("Operational event snapshot is incomplete")
    if not isinstance(value["supported"], bool) or not isinstance(
        value["schedule_available"], bool
    ):
        raise ValueError("Operational event support fields are invalid")
    if value["timezone"] != "UTC" or not isinstance(value["sessions"], list):
        raise ValueError("Operational event schedule fields are invalid")
    for session in value["sessions"]:
        if not isinstance(session, dict):
            raise ValueError("Operational session must be an object")
        session_required = {
            "sequence",
            "session",
            "display_name",
            "scheduled_start_utc",
            "scheduled_end_utc",
            "end_source",
            "schedule_status",
        }
        if session_required - set(session):
            raise ValueError("Operational session is incomplete")
        start = _parse_aware_datetime(session["scheduled_start_utc"])
        end = _parse_aware_datetime(session["scheduled_end_utc"])
        if end <= start:
            raise ValueError("Operational session end must follow its start")


def _parse_aware_datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Operational session timestamps must be strings")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Operational session timestamp is invalid") from exc
    return _aware_utc(parsed)


def runtime_storage_diagnostics(
    config: DataConfig,
    autopilot: AutopilotConfig,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    """Return read-only persistent-runtime and FastF1 cache size diagnostics."""
    cache_bytes = _tree_bytes(config.fastf1_cache_dir)
    runtime_root_value = environ.get("APEX_PULSE_RUNTIME_ROOT")
    runtime_root = Path(runtime_root_value) if runtime_root_value else config.project_root
    total = sum(_tree_bytes(runtime_root / name) for name in ("data", "reports", "models"))
    warning_mb = _positive_int(environ.get(CACHE_WARNING_ENV)) or autopilot.cache_warning_mb
    capacity_mb = _positive_int(environ.get(VOLUME_CAPACITY_ENV))
    capacity = capacity_mb * 1024 * 1024 if capacity_mb else None
    if capacity is not None and total >= capacity:
        status = "capacity_exceeded"
    elif cache_bytes >= warning_mb * 1024 * 1024:
        status = "warning_threshold_exceeded"
    elif capacity is None:
        status = "ok_capacity_unknown"
    else:
        status = "ok"
    return {
        "fastf1_cache_bytes": cache_bytes,
        "runtime_total_known_bytes": total,
        "volume_capacity_bytes": capacity,
        "cache_warning_status": status,
    }


def _tree_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _live_event_rows(frame: pd.DataFrame, protocol_name: str, event_slug: str) -> pd.DataFrame:
    if frame.empty or not {"protocol_name", "event_slug"} <= set(frame):
        return pd.DataFrame()
    result = frame[
        frame["protocol_name"].astype(str).eq(protocol_name)
        & frame["event_slug"].astype(str).eq(event_slug)
    ].copy()
    if "diagnostic_only" in result:
        result = result[~result["diagnostic_only"].astype(bool)]
    return result


def _dashboard_lifecycle(config: DataConfig) -> str | None:
    path = config.metrics_output_dir.parent / "dashboard/current_event.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("data", {}).get("lifecycle", {}).get("state")
    return str(value) if value else None


def _latest_session_end(event: WeekendEvent) -> datetime | None:
    return max((value.end_utc for value in event.sessions.values()), default=None)


def _driver_count(laps: Any) -> int:
    if isinstance(laps, pd.DataFrame) and "Driver" in laps:
        return int(laps["Driver"].dropna().astype(str).nunique())
    return 0


def _integer(value: Any) -> int | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return int(parsed) if pd.notna(parsed) else None


def _positive_int(value: str | None) -> int | None:
    try:
        parsed = int(value) if value is not None else 0
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _parse_enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Autopilot time must be timezone-aware")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _aware_utc(value).isoformat()


def _safe_message(exc: Exception) -> str:
    return " ".join(str(exc).split())[:500] or type(exc).__name__


def _transient_message(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in (
            "timeout",
            "temporar",
            "network",
            "connection",
            "unavailable",
            "not yet",
            "failed to fetch",
            "session could not be loaded",
            "ingestion failed",
        )
    )


def _base_event_slug(value: str) -> str:
    for suffix in ("-grand-prix", "-gp"):
        if value.endswith(suffix):
            return value.removesuffix(suffix)
    return value
