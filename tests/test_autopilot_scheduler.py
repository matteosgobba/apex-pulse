from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from f1_prediction.autopilot_scheduler import (
    SCHEDULER_ENABLED_ENV,
    AutopilotScheduler,
    SchedulerSettings,
    validate_scheduler_status,
)
from f1_prediction.dashboard_api.app import create_dashboard_app
from f1_prediction.modeling.weekend_orchestrator import AutopilotTickResult

UTC = timezone.utc


def test_scheduler_defaults_disabled_and_switch_is_independent_from_mutation_flag() -> None:
    disabled = SchedulerSettings.from_environ({"APEX_PULSE_AUTOPILOT_ENABLED": "true"})
    enabled = SchedulerSettings.from_environ(
        {
            SCHEDULER_ENABLED_ENV: "true",
            "APEX_PULSE_AUTOPILOT_ENABLED": "false",
        }
    )

    assert disabled.enabled is False
    assert enabled.enabled is True
    assert enabled.interval_seconds == 300


def test_scheduler_rejects_unsafe_interval() -> None:
    with pytest.raises(ValueError, match="at least 60"):
        SchedulerSettings.from_environ(
            {SCHEDULER_ENABLED_ENV: "true", "APEX_PULSE_AUTOPILOT_INTERVAL_SECONDS": "5"}
        )


def test_disabled_scheduler_never_invokes_tick_or_writes_status(tmp_path: Path) -> None:
    calls: list[str] = []
    scheduler = AutopilotScheduler(
        SchedulerSettings(enabled=False),
        tmp_path / "status.json",
        lambda: calls.append("tick"),  # type: ignore[arg-type,return-value]
    )

    scheduler.start()

    assert scheduler.running is False
    assert calls == []
    assert not (tmp_path / "status.json").exists()


def test_repeated_ticks_are_sequential_and_persist_scheduler_origin(tmp_path: Path) -> None:
    calls = 0

    def tick() -> AutopilotTickResult:
        nonlocal calls
        calls += 1
        return _result("AUTOPILOT_DISABLED")

    scheduler = AutopilotScheduler(SchedulerSettings(enabled=True), tmp_path / "status.json", tick)

    assert asyncio.run(scheduler.run_once()) is True
    assert asyncio.run(scheduler.run_once()) is True

    payload = validate_scheduler_status(json.loads((tmp_path / "status.json").read_text()))
    assert calls == 2
    assert payload["scheduler_iteration_count"] == 2
    assert payload["last_scheduler_tick_state"] == "AUTOPILOT_DISABLED"
    assert payload["last_tick_origin"] == "scheduler"
    assert payload["last_tick_run_id"] == "synthetic-run"
    assert payload["consecutive_scheduler_failures"] == 0


def test_enabled_scheduler_can_relay_an_armed_mutating_tick_result(tmp_path: Path) -> None:
    scheduler = AutopilotScheduler(
        SchedulerSettings(enabled=True),
        tmp_path / "status.json",
        lambda: _result("UNSUPPORTED_WEEKEND_FORMAT", autopilot_enabled=True),
    )

    asyncio.run(scheduler.run_once())

    payload = json.loads((tmp_path / "status.json").read_text())
    assert payload["scheduler_enabled"] is True
    assert payload["last_tick_result"]["autopilot_enabled"] is True
    assert payload["last_tick_state"] == "UNSUPPORTED_WEEKEND_FORMAT"


def test_initial_delay_is_cancellable_without_invoking_tick(tmp_path: Path) -> None:
    calls: list[str] = []

    async def exercise() -> None:
        scheduler = AutopilotScheduler(
            SchedulerSettings(enabled=True, initial_delay_seconds=30),
            tmp_path / "status.json",
            lambda: calls.append("tick"),  # type: ignore[arg-type,return-value]
        )
        scheduler.start()
        await asyncio.sleep(0)
        assert scheduler.running is True
        await scheduler.stop()

    asyncio.run(exercise())
    assert calls == []
    assert json.loads((tmp_path / "status.json").read_text())["scheduler_running"] is False


def test_overlap_is_skipped_without_second_invocation(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def tick() -> AutopilotTickResult:
        nonlocal calls
        calls += 1
        entered.set()
        release.wait(timeout=2)
        return _result("UNSUPPORTED_WEEKEND_FORMAT")

    async def exercise() -> bool:
        scheduler = AutopilotScheduler(
            SchedulerSettings(enabled=True), tmp_path / "status.json", tick
        )
        first = asyncio.create_task(scheduler.run_once())
        await asyncio.to_thread(entered.wait, 1)
        second = await scheduler.run_once()
        release.set()
        await first
        return second

    assert asyncio.run(exercise()) is False
    assert calls == 1


def test_blocking_tick_runs_off_event_loop(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()

    def tick() -> AutopilotTickResult:
        entered.set()
        release.wait(timeout=2)
        return _result("SETTLED")

    async def exercise() -> None:
        scheduler = AutopilotScheduler(
            SchedulerSettings(enabled=True), tmp_path / "status.json", tick
        )
        task = asyncio.create_task(scheduler.run_once())
        await asyncio.to_thread(entered.wait, 1)
        heartbeat = False
        await asyncio.sleep(0)
        heartbeat = True
        assert heartbeat is True
        release.set()
        await task

    asyncio.run(exercise())


def test_tick_exception_is_recorded_and_later_tick_still_runs(tmp_path: Path) -> None:
    calls = 0

    def tick() -> AutopilotTickResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary synthetic failure")
        return _result("SETTLED")

    async def exercise() -> None:
        scheduler = AutopilotScheduler(
            SchedulerSettings(enabled=True), tmp_path / "status.json", tick
        )
        await scheduler.run_once()
        first = json.loads((tmp_path / "status.json").read_text())
        assert first["last_scheduler_tick_state"] == "SCHEDULER_TICK_ERROR"
        await scheduler.run_once()

    asyncio.run(exercise())
    final = json.loads((tmp_path / "status.json").read_text())
    assert calls == 2
    assert final["last_scheduler_tick_state"] == "SETTLED"
    assert final["last_scheduler_tick_error"] is None
    assert final["consecutive_scheduler_failures"] == 0


def test_fastapi_lifespan_starts_and_stops_injected_scheduler(tmp_path: Path) -> None:
    calls: list[str] = []

    class FakeScheduler:
        def start(self) -> None:
            calls.append("start")

        async def stop(self) -> None:
            calls.append("stop")

    app = create_dashboard_app(tmp_path / "dashboard", scheduler_factory=lambda: FakeScheduler())

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            assert calls == ["start"]

    asyncio.run(exercise())
    assert calls == ["start", "stop"]


def _result(state: str, *, autopilot_enabled: bool = False) -> AutopilotTickResult:
    instant = datetime(2026, 8, 4, tzinfo=UTC).isoformat()
    return AutopilotTickResult(
        schema_version="1.0",
        run_id="synthetic-run",
        started_at_utc=instant,
        completed_at_utc=instant,
        duration_seconds=0.0,
        dry_run=False,
        autopilot_enabled=autopilot_enabled,
        season=2026,
        event="Dutch Grand Prix",
        event_slug="dutch-grand-prix",
        round_number=15,
        event_format="sprint_qualifying",
        orchestrator_state_before=state,
        orchestrator_state_after=state,
        calendar_source="synthetic",
        fp1_status="not_probed",
        fp2_status="not_probed",
        fp3_status="not_probed",
        qualifying_status="not_probed",
        forecast_exists=False,
        settlement_exists=False,
        action_considered="none",
        action_taken="none",
        action_result="synthetic",
        retryable=False,
        retry_reason=None,
        next_recommended_check_at_utc=None,
        lock_status="not_attempted",
        error_classification="none",
        error_message_safe=None,
        current_dashboard_lifecycle="settled_partial_coverage",
        fastf1_cache_bytes=0,
        runtime_total_known_bytes=0,
        volume_capacity_bytes=None,
        cache_warning_status="ok_capacity_unknown",
    )
