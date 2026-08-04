from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from f1_prediction.config import DataConfig, load_feature_config, load_model_config
from f1_prediction.modeling.monitoring_operations import MonitoringWorkflowSummary
from f1_prediction.modeling.weekend_orchestrator import (
    AUTOPILOT_ENABLED_ENV,
    RUNS_FILE,
    STATUS_FILE,
    AutopilotConfig,
    FastF1ReadinessProvider,
    FastF1ScheduleProvider,
    FilesystemWriterLock,
    ProbeStatus,
    ReadinessProbeResult,
    SessionWindow,
    WeekendEvent,
    discover_relevant_event,
    run_autopilot_tick,
    validate_autopilot_status,
)
from f1_prediction.modeling.weekend_orchestrator_rehearsal import (
    rehearse_autopilot_artifacts,
)

UTC = timezone.utc
PROTOCOL = "season_2026_v1"


class FakeScheduleProvider:
    def __init__(self, events: list[WeekendEvent] | None = None) -> None:
        self._events = events or []

    def events(self, season: int) -> tuple[WeekendEvent, ...]:
        return tuple(event for event in self._events if event.season == season)


class FakeReadinessProvider:
    def __init__(self, **statuses: ProbeStatus) -> None:
        self.statuses = statuses
        self.calls: list[str] = []

    def probe(self, event: WeekendEvent, session_code: str) -> ReadinessProbeResult:
        self.calls.append(session_code)
        status = self.statuses.get(session_code, ProbeStatus.READY)
        return ReadinessProbeResult(
            status,
            f"synthetic {session_code} {status.value}",
            retryable=status
            in {ProbeStatus.NOT_YET_AVAILABLE, ProbeStatus.INCOMPLETE, ProbeStatus.TRANSIENT_ERROR},
        )


def test_event_selection_chooses_current_then_next_and_preserves_timezone() -> None:
    first = _event("First Grand Prix", round_number=1)
    second = _event(
        "Second Grand Prix",
        round_number=2,
        base=datetime(2026, 6, 8, 10, tzinfo=UTC),
    )
    provider = FakeScheduleProvider([first, second])
    config = AutopilotConfig(post_qualifying_event_hold_hours=24)

    assert discover_relevant_event(provider, datetime(2026, 5, 31, tzinfo=UTC), config) == first
    assert discover_relevant_event(provider, datetime(2026, 6, 3, 20, tzinfo=UTC), config) == second
    with pytest.raises(ValueError, match="timezone-aware"):
        discover_relevant_event(provider, datetime(2026, 5, 31), config)


def test_fastf1_schedule_provider_normalizes_official_schedule_and_rejectable_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = pd.DataFrame(
        [
            {
                "RoundNumber": 7,
                "EventName": "Conventional Grand Prix",
                "Location": "Example",
                "Country": "Exampleland",
                "OfficialEventName": "FORMULA 1 EXAMPLE",
                "EventFormat": "conventional",
                "Session1": "Practice 1",
                "Session1DateUtc": "2026-06-01T10:00:00Z",
                "Session2": "Practice 2",
                "Session2DateUtc": "2026-06-01T14:00:00Z",
                "Session3": "Practice 3",
                "Session3DateUtc": "2026-06-02T10:00:00Z",
                "Session4": "Qualifying",
                "Session4DateUtc": "2026-06-02T14:00:00Z",
            },
            {
                "RoundNumber": 8,
                "EventName": "Sprint Grand Prix",
                "EventFormat": "sprint_qualifying",
                "Session1": "Practice 1",
                "Session1DateUtc": "2026-06-08T10:00:00Z",
                "Session2": "Sprint Qualifying",
                "Session2DateUtc": "2026-06-08T14:00:00Z",
                "Session3": "Sprint",
                "Session3DateUtc": "2026-06-09T10:00:00Z",
                "Session4": "Qualifying",
                "Session4DateUtc": "2026-06-09T14:00:00Z",
            },
        ]
    )
    monkeypatch.setattr("fastf1.get_event_schedule", lambda *args, **kwargs: schedule)

    events = FastF1ScheduleProvider(AutopilotConfig()).events(2026)

    assert events[0].event_slug == "conventional-grand-prix"
    assert set(events[0].sessions) == {"FP1", "FP2", "FP3", "Q"}
    assert events[0].sessions["FP1"].start_utc.tzinfo == UTC
    assert events[0].sessions["FP1"].end_source == "configured_default_duration"
    assert [session.code for session in events[0].operational_sessions] == [
        "FP1",
        "FP2",
        "FP3",
        "Q",
    ]
    assert events[1].event_format == "sprint_qualifying"
    assert set(events[1].sessions) == {"FP1", "Q"}
    assert [session.code for session in events[1].operational_sessions] == [
        "FP1",
        "SQ",
        "S",
        "Q",
    ]
    assert [session.name for session in events[1].operational_sessions] == [
        "Practice 1",
        "Sprint Qualifying",
        "Sprint",
        "Qualifying",
    ]


def test_fastf1_readiness_probe_loads_minimum_data_without_network_in_unit_test(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, bool]] = []

    class Session:
        laps = pd.DataFrame({"Driver": ["AAA", "BBB"]})
        results = pd.DataFrame(index=range(20))

        def load(self, **kwargs: bool) -> None:
            calls.append(kwargs)

    monkeypatch.setattr("fastf1.get_session", lambda *args, **kwargs: Session())
    monkeypatch.setattr("fastf1.Cache.enable_cache", lambda path: None)

    result = FastF1ReadinessProvider(_data_config(tmp_path)).probe(_event(), "Q")

    assert result.status == ProbeStatus.READY
    assert result.driver_count == 2
    assert calls == [{"laps": True, "telemetry": False, "weather": False, "messages": False}]


def test_event_override_must_resolve_exact_calendar_identity() -> None:
    event = _event("Hungarian Grand Prix", round_number=14)
    provider = FakeScheduleProvider([event])

    selected = discover_relevant_event(
        provider,
        datetime(2026, 6, 1, tzinfo=UTC),
        AutopilotConfig(),
        season_override=2026,
        event_override="Hungarian GP",
    )

    assert selected == event
    with pytest.raises(ValueError, match="exactly one"):
        discover_relevant_event(
            provider,
            datetime(2026, 6, 1, tzinfo=UTC),
            AutopilotConfig(),
            season_override=2026,
            event_override="invented race",
        )


@pytest.mark.parametrize(
    ("now", "expected_state", "expected_next"),
    [
        (datetime(2026, 5, 31, 12, tzinfo=UTC), "WAITING_FOR_FP1", "2026-06-01"),
        (datetime(2026, 6, 1, 12, tzinfo=UTC), "FP1_COMPLETE", "2026-06-01"),
        (datetime(2026, 6, 1, 16, tzinfo=UTC), "FP2_COMPLETE", "2026-06-02"),
        (datetime(2026, 6, 2, 10, 30, tzinfo=UTC), "WAITING_FOR_FP3", "2026-06-02"),
        (datetime(2026, 6, 2, 11, 10, tzinfo=UTC), "FP3_INITIAL_GRACE", "2026-06-02"),
    ],
)
def test_calendar_states_do_not_claim_data_readiness(
    tmp_path: Path,
    now: datetime,
    expected_state: str,
    expected_next: str,
) -> None:
    readiness = FakeReadinessProvider()

    result = _tick(tmp_path, now=now, readiness=readiness)

    assert result.orchestrator_state_after == expected_state
    assert result.next_recommended_check_at_utc.startswith(expected_next)
    if expected_state in {"WAITING_FOR_FP1", "WAITING_FOR_FP3", "FP3_INITIAL_GRACE"}:
        assert not readiness.calls


@pytest.mark.parametrize(
    ("probe_status", "state", "retryable", "classification"),
    [
        (
            ProbeStatus.NOT_YET_AVAILABLE,
            "FP3_TIME_ELAPSED_DATA_PENDING",
            True,
            "retryable_transient",
        ),
        (
            ProbeStatus.INCOMPLETE,
            "FP3_TIME_ELAPSED_DATA_PENDING",
            True,
            "retryable_transient",
        ),
        (ProbeStatus.TRANSIENT_ERROR, "FP3_TIME_ELAPSED_DATA_PENDING", True, "retryable_transient"),
        (ProbeStatus.PERMANENT_ERROR, "BLOCKED", False, "blocking_permanent"),
    ],
)
def test_fp3_readiness_classifies_pending_and_permanent_failures(
    tmp_path: Path,
    probe_status: ProbeStatus,
    state: str,
    retryable: bool,
    classification: str,
) -> None:
    result = _tick(
        tmp_path,
        now=datetime(2026, 6, 2, 11, 20, tzinfo=UTC),
        readiness=FakeReadinessProvider(FP3=probe_status),
    )

    assert result.orchestrator_state_after == state
    assert result.retryable is retryable
    assert result.error_classification == classification
    assert result.action_taken == "none"


def test_ready_fp3_dry_run_reports_forecast_without_invoking_workflow(tmp_path: Path) -> None:
    calls: list[Any] = []

    result = _tick(
        tmp_path,
        now=datetime(2026, 6, 2, 11, 20, tzinfo=UTC),
        readiness=FakeReadinessProvider(),
        before_workflow=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert result.orchestrator_state_after == "READY_FOR_FORECAST"
    assert result.action_result == "would_run_canonical_before_qualifying"
    assert calls == []
    assert not (tmp_path / "reports/metrics").exists()


def test_forecast_exists_skips_generation_and_waits_for_qualifying(tmp_path: Path) -> None:
    config = _data_config(tmp_path)
    _write_forecast(config, _event())
    before_calls: list[Any] = []

    result = _tick(
        tmp_path,
        now=datetime(2026, 6, 2, 12, tzinfo=UTC),
        before_workflow=lambda *args, **kwargs: before_calls.append((args, kwargs)),
    )

    assert result.forecast_exists is True
    assert result.orchestrator_state_after == "WAITING_FOR_QUALIFYING"
    assert result.action_considered == "wait_for_qualifying"
    assert before_calls == []


def test_qualifying_pending_then_ready_for_settlement(tmp_path: Path) -> None:
    config = _data_config(tmp_path)
    event = _event()
    _write_forecast(config, event)

    pending = _tick(
        tmp_path,
        now=datetime(2026, 6, 2, 15, 20, tzinfo=UTC),
        readiness=FakeReadinessProvider(Q=ProbeStatus.INCOMPLETE),
    )
    ready = _tick(
        tmp_path,
        now=datetime(2026, 6, 2, 15, 20, tzinfo=UTC),
        readiness=FakeReadinessProvider(Q=ProbeStatus.READY),
    )

    assert pending.orchestrator_state_after == "QUALIFYING_TIME_ELAPSED_DATA_PENDING"
    assert pending.retryable is True
    assert ready.orchestrator_state_after == "READY_FOR_SETTLEMENT"
    assert ready.action_result == "would_run_canonical_after_qualifying"


@pytest.mark.parametrize("partial", [False, True])
def test_existing_settlement_is_terminal_and_never_reinvokes_workflows(
    tmp_path: Path,
    partial: bool,
) -> None:
    config = _data_config(tmp_path)
    event = _event()
    _write_forecast(config, event)
    _write_settlement(config, event, partial=partial)
    calls: list[Any] = []

    result = _tick(
        tmp_path,
        now=datetime(2026, 6, 2, 15, 30, tzinfo=UTC),
        before_workflow=lambda *args, **kwargs: calls.append((args, kwargs)),
        after_workflow=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    expected = "SETTLED_PARTIAL_COVERAGE" if partial else "SETTLED"
    assert result.orchestrator_state_after == expected
    assert result.action_result == "existing_settlement_reused"
    assert calls == []


def test_sprint_weekend_is_explicitly_unsupported_without_probe_or_mutation(tmp_path: Path) -> None:
    event = _event(event_format="sprint", omit={"FP2", "FP3"})
    readiness = FakeReadinessProvider()

    result = _tick(tmp_path, event=event, readiness=readiness)

    assert result.orchestrator_state_after == "UNSUPPORTED_WEEKEND_FORMAT"
    assert result.error_classification == "blocking_permanent"
    assert "event_format=sprint" in str(result.error_message_safe)
    assert readiness.calls == []
    assert result.operational_event is not None
    assert result.operational_event["supported"] is False
    assert result.operational_event["schedule_available"] is True
    assert result.operational_event["timezone"] == "UTC"
    assert result.operational_event["sessions"][0]["scheduled_start_utc"].endswith("+00:00")


def test_operational_schedule_is_additive_and_old_status_remains_valid(tmp_path: Path) -> None:
    result = _tick(tmp_path, now=datetime(2026, 5, 31, 12, tzinfo=UTC))
    payload = result.to_dict()

    assert validate_autopilot_status(payload) == payload
    assert payload["operational_event"]["event"] == "Test Grand Prix"
    assert [session["session"] for session in payload["operational_event"]["sessions"]] == [
        "FP1",
        "FP2",
        "FP3",
        "Q",
    ]
    legacy_payload = dict(payload)
    legacy_payload.pop("operational_event")
    assert validate_autopilot_status(legacy_payload) == legacy_payload


def test_mutating_tick_requires_explicit_enable_flag(tmp_path: Path) -> None:
    result = _tick(tmp_path, dry_run=False, environ={})

    assert result.orchestrator_state_after == "AUTOPILOT_DISABLED"
    assert result.lock_status == "not_attempted"
    assert not (tmp_path / "reports/metrics").exists()


def test_enabled_waiting_tick_writes_valid_status_and_append_only_audit(tmp_path: Path) -> None:
    result = _tick(
        tmp_path,
        now=datetime(2026, 5, 31, 12, tzinfo=UTC),
        dry_run=False,
        environ={AUTOPILOT_ENABLED_ENV: "true"},
    )
    metrics = tmp_path / "reports/metrics"

    status = json.loads((metrics / STATUS_FILE).read_text(encoding="utf-8"))
    audit_lines = (metrics / RUNS_FILE).read_text(encoding="utf-8").splitlines()
    assert validate_autopilot_status(status) == status
    assert status["run_id"] == result.run_id
    assert len(audit_lines) == 1
    assert json.loads(audit_lines[0])["run_id"] == result.run_id


def test_lock_contention_and_release_are_process_safe(tmp_path: Path) -> None:
    path = tmp_path / "autopilot.lock"
    now = datetime(2026, 6, 1, tzinfo=UTC)
    first = FilesystemWriterLock(path, "first", now)
    second = FilesystemWriterLock(path, "second", now)

    assert first.acquire() is True
    assert second.acquire() is False
    first.release(now + timedelta(seconds=1))
    assert second.acquire() is True
    second.release(now + timedelta(seconds=2))


def test_lock_releases_when_readiness_raises(tmp_path: Path) -> None:
    class ExplodingReadiness:
        def probe(self, event: WeekendEvent, session_code: str) -> ReadinessProbeResult:
            raise RuntimeError("permanent synthetic integrity failure")

    _tick(
        tmp_path,
        now=datetime(2026, 6, 2, 11, 20, tzinfo=UTC),
        dry_run=False,
        environ={AUTOPILOT_ENABLED_ENV: "true"},
        readiness=ExplodingReadiness(),
    )
    path = tmp_path / "reports/metrics/autopilot.lock"
    replacement = FilesystemWriterLock(path, "replacement", datetime(2026, 6, 2, tzinfo=UTC))

    assert replacement.acquire() is True
    replacement.release(datetime(2026, 6, 2, 0, 1, tzinfo=UTC))


def test_canonical_before_workflow_runs_once_across_repeated_ticks(tmp_path: Path) -> None:
    config = _data_config(tmp_path)
    event = _event()
    calls: list[str] = []

    def before(*args: Any, **kwargs: Any) -> MonitoringWorkflowSummary:
        calls.append("before")
        _write_forecast(config, event)
        return _workflow_summary(tmp_path, "monitoring_before_qualifying", event)

    kwargs = {
        "now": datetime(2026, 6, 2, 11, 20, tzinfo=UTC),
        "dry_run": False,
        "environ": {AUTOPILOT_ENABLED_ENV: "true"},
        "before_workflow": before,
    }
    first = _tick(tmp_path, **kwargs)
    second = _tick(tmp_path, **kwargs)

    assert first.orchestrator_state_before == "READY_FOR_FORECAST"
    assert first.orchestrator_state_after == "FORECAST_AVAILABLE"
    assert second.orchestrator_state_after == "WAITING_FOR_QUALIFYING"
    assert calls == ["before"]


@pytest.mark.parametrize(
    ("message", "state", "retryable"),
    [
        ("temporary network timeout", "TRANSIENT_ERROR", True),
        ("protocol fingerprint mismatch", "BLOCKED", False),
    ],
)
def test_canonical_workflow_exceptions_keep_action_and_retry_classification(
    tmp_path: Path,
    message: str,
    state: str,
    retryable: bool,
) -> None:
    def before(*args: Any, **kwargs: Any) -> MonitoringWorkflowSummary:
        raise RuntimeError(message)

    result = _tick(
        tmp_path,
        now=datetime(2026, 6, 2, 11, 20, tzinfo=UTC),
        dry_run=False,
        environ={AUTOPILOT_ENABLED_ENV: "true"},
        before_workflow=before,
    )

    assert result.orchestrator_state_before == "READY_FOR_FORECAST"
    assert result.orchestrator_state_after == state
    assert result.action_considered == "run_before_qualifying"
    assert result.action_taken == "run_before_qualifying"
    assert result.retryable is retryable


def test_canonical_after_workflow_runs_once_and_preserves_partial_terminal_state(
    tmp_path: Path,
) -> None:
    config = _data_config(tmp_path)
    event = _event()
    _write_forecast(config, event)
    calls: list[str] = []

    def after(*args: Any, **kwargs: Any) -> MonitoringWorkflowSummary:
        calls.append("after")
        _write_settlement(config, event, partial=True)
        return _workflow_summary(tmp_path, "monitoring_after_qualifying", event, warnings=1)

    kwargs = {
        "now": datetime(2026, 6, 2, 15, 20, tzinfo=UTC),
        "dry_run": False,
        "environ": {AUTOPILOT_ENABLED_ENV: "true"},
        "after_workflow": after,
    }
    first = _tick(tmp_path, **kwargs)
    second = _tick(tmp_path, **kwargs)

    assert first.orchestrator_state_before == "READY_FOR_SETTLEMENT"
    assert first.orchestrator_state_after == "SETTLED_PARTIAL_COVERAGE"
    assert second.action_result == "existing_settlement_reused"
    assert calls == ["after"]


def test_storage_diagnostics_warn_without_deleting_cache(tmp_path: Path) -> None:
    cache = tmp_path / "data/raw/fastf1_cache/preserved.bin"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"x" * 1024)

    result = _tick(
        tmp_path,
        environ={"APEX_PULSE_RUNTIME_CACHE_WARNING_MB": "1"},
    )

    assert result.fastf1_cache_bytes == 1024
    assert result.cache_warning_status == "ok_capacity_unknown"
    assert cache.read_bytes() == b"x" * 1024


def test_artifact_rehearsal_uses_copies_and_preserves_sources(tmp_path: Path) -> None:
    config = _data_config(tmp_path)
    metrics = config.metrics_output_dir
    metrics.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "season": 2026,
                "event": "Belgian Grand Prix",
                "event_slug": "belgian-grand-prix",
                "event_order": 10,
                "protocol_name": PROTOCOL,
                "partial_target_coverage": False,
            },
            {
                "season": 2026,
                "event": "Hungarian Grand Prix",
                "event_slug": "hungarian-grand-prix",
                "event_order": 11,
                "protocol_name": PROTOCOL,
                "partial_target_coverage": True,
            },
        ]
    ).to_csv(metrics / "prospective_monitoring_event_registry.csv", index=False)
    pd.DataFrame(
        [
            {
                "protocol_name": PROTOCOL,
                "event_slug": slug,
                "diagnostic_only": False,
            }
            for slug in ("belgian-grand-prix", "hungarian-grand-prix")
        ]
    ).to_parquet(metrics / "prospective_monitoring_forecasts.parquet", index=False)
    pd.DataFrame(
        [
            {
                "protocol_name": PROTOCOL,
                "event_slug": "hungarian-grand-prix",
                "diagnostic_only": False,
                "settlement_valid": True,
            }
        ]
    ).to_parquet(metrics / "prospective_monitoring_settlements.parquet", index=False)
    before = {path.name: path.read_bytes() for path in metrics.iterdir()}
    project_root = Path(__file__).resolve().parents[1]

    summary = rehearse_autopilot_artifacts(
        config,
        load_model_config(project_root=project_root),
        load_feature_config(project_root=project_root),
        autopilot_config=AutopilotConfig(),
    )

    assert summary.status == "passed"
    assert summary.forecast_event == "Belgian Grand Prix"
    assert summary.settled_event == "Hungarian Grand Prix"
    assert summary.source_artifacts_unchanged is True
    assert summary.scenarios["post_fp3_pre_qualifying"]["action_result"] == (
        "would_run_canonical_before_qualifying"
    )
    assert summary.scenarios["post_qualifying"]["action_result"] == (
        "would_run_canonical_after_qualifying"
    )
    assert {path.name: path.read_bytes() for path in metrics.iterdir()} == before


def _tick(
    root: Path,
    *,
    now: datetime = datetime(2026, 6, 2, 11, 20, tzinfo=UTC),
    event: WeekendEvent | None = None,
    readiness: Any | None = None,
    dry_run: bool = True,
    environ: dict[str, str] | None = None,
    before_workflow: Any | None = None,
    after_workflow: Any | None = None,
):
    project_root = Path(__file__).resolve().parents[1]
    kwargs: dict[str, Any] = {}
    if before_workflow is not None:
        kwargs["before_workflow"] = before_workflow
    if after_workflow is not None:
        kwargs["after_workflow"] = after_workflow
    selected = event or _event()
    return run_autopilot_tick(
        _data_config(root),
        load_model_config(project_root=project_root),
        load_feature_config(project_root=project_root),
        autopilot_config=AutopilotConfig(),
        now=now,
        season=selected.season,
        event=selected.event,
        dry_run=dry_run,
        schedule_provider=FakeScheduleProvider([selected]),
        readiness_provider=readiness or FakeReadinessProvider(),
        environ=environ or {},
        **kwargs,
    )


def _event(
    name: str = "Test Grand Prix",
    *,
    round_number: int = 1,
    event_format: str = "conventional",
    base: datetime = datetime(2026, 6, 1, 10, tzinfo=UTC),
    omit: set[str] | None = None,
) -> WeekendEvent:
    starts = {
        "FP1": base,
        "FP2": base + timedelta(hours=4),
        "FP3": base + timedelta(days=1),
        "Q": base + timedelta(days=1, hours=4),
    }
    durations = {"FP1": 60, "FP2": 60, "FP3": 60, "Q": 60}
    sessions = {
        code: SessionWindow(
            code,
            "Qualifying" if code == "Q" else f"Practice {code[-1]}",
            start,
            start + timedelta(minutes=durations[code]),
            "synthetic_test",
        )
        for code, start in starts.items()
        if code not in (omit or set())
    }
    slug = name.lower().replace(" ", "-")
    operational_sessions = tuple(sorted(sessions.values(), key=lambda item: item.start_utc))
    return WeekendEvent(
        2026,
        name,
        slug,
        round_number,
        event_format,
        sessions,
        (slug,),
        operational_sessions,
    )


def _data_config(root: Path) -> DataConfig:
    return DataConfig(
        project_root=root,
        fastf1_cache_dir=root / "data/raw/fastf1_cache",
        lap_output_dir=root / "data/raw/laps",
        session_metadata_output_dir=root / "data/raw/session_metadata",
        clean_lap_output_dir=root / "data/interim/clean_laps",
        session_features_output_dir=root / "data/processed/session_features",
        modeling_output_dir=root / "data/processed/modeling",
        metrics_output_dir=root / "reports/metrics",
    )


def _write_forecast(config: DataConfig, event: WeekendEvent) -> None:
    config.metrics_output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "protocol_name": PROTOCOL,
                "event_slug": event.event_slug,
                "diagnostic_only": False,
                "driver_code": "AAA",
            }
        ]
    ).to_parquet(config.metrics_output_dir / "prospective_monitoring_forecasts.parquet")


def _write_settlement(config: DataConfig, event: WeekendEvent, *, partial: bool) -> None:
    config.metrics_output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "protocol_name": PROTOCOL,
                "event_slug": event.event_slug,
                "diagnostic_only": False,
                "settlement_valid": True,
                "driver_code": "AAA",
            }
        ]
    ).to_parquet(config.metrics_output_dir / "prospective_monitoring_settlements.parquet")
    pd.DataFrame(
        [
            {
                "protocol_name": PROTOCOL,
                "event_slug": event.event_slug,
                "partial_target_coverage": partial,
            }
        ]
    ).to_csv(config.metrics_output_dir / "prospective_monitoring_event_registry.csv", index=False)


def _workflow_summary(
    root: Path,
    workflow: str,
    event: WeekendEvent,
    *,
    warnings: int = 0,
) -> MonitoringWorkflowSummary:
    summary_path = root / f"{workflow}.json"
    summary_path.write_text('{"status":"completed"}\n', encoding="utf-8")
    return MonitoringWorkflowSummary(
        status="completed",
        workflow=workflow,
        summary_path=summary_path,
        stages_path=root / f"{workflow}.csv",
        completed=True,
        blocking_failure_count=0,
        warning_count=warnings,
        event=event.event,
        event_slug=event.event_slug,
        event_order=event.round_number,
        scheduled_event_date="2026-06-01",
        event_order_resolution_source="synthetic_test",
        dashboard_current_event=event.event,
    )
