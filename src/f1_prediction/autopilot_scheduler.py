"""Restart-safe in-process scheduling for the one-shot weekend orchestrator."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from f1_prediction.config import load_data_config, load_feature_config, load_model_config
from f1_prediction.modeling.weekend_orchestrator import (
    AutopilotTickResult,
    load_autopilot_config,
    run_autopilot_tick,
)

LOGGER = logging.getLogger(__name__)
SCHEDULER_SCHEMA_VERSION = "1.0"
SCHEDULER_STATUS_FILE = "autopilot_scheduler_status.json"
SCHEDULER_ENABLED_ENV = "APEX_PULSE_AUTOPILOT_SCHEDULER_ENABLED"
SCHEDULER_INTERVAL_ENV = "APEX_PULSE_AUTOPILOT_INTERVAL_SECONDS"
SCHEDULER_INITIAL_DELAY_ENV = "APEX_PULSE_AUTOPILOT_INITIAL_DELAY_SECONDS"
DEFAULT_INTERVAL_SECONDS = 300
MINIMUM_INTERVAL_SECONDS = 60
DEFAULT_INITIAL_DELAY_SECONDS = 10


@dataclass(frozen=True)
class SchedulerSettings:
    """Validated scheduler switches independent from mutation authorization."""

    enabled: bool = False
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS
    initial_delay_seconds: int = DEFAULT_INITIAL_DELAY_SECONDS

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None) -> SchedulerSettings:
        environment = os.environ if environ is None else environ
        enabled = _enabled(environment.get(SCHEDULER_ENABLED_ENV))
        interval = _integer(environment.get(SCHEDULER_INTERVAL_ENV), DEFAULT_INTERVAL_SECONDS)
        initial_delay = _integer(
            environment.get(SCHEDULER_INITIAL_DELAY_ENV), DEFAULT_INITIAL_DELAY_SECONDS
        )
        if interval < MINIMUM_INTERVAL_SECONDS:
            raise ValueError(
                f"{SCHEDULER_INTERVAL_ENV} must be at least {MINIMUM_INTERVAL_SECONDS}"
            )
        if initial_delay < 0:
            raise ValueError(f"{SCHEDULER_INITIAL_DELAY_ENV} must be zero or positive")
        return cls(enabled, interval, initial_delay)


@dataclass(frozen=True)
class SchedulerSnapshot:
    """Small operational snapshot merged into the read-only status endpoint."""

    schema_version: str
    scheduler_enabled: bool
    scheduler_running: bool
    scheduler_interval_seconds: int
    scheduler_initial_delay_seconds: int
    scheduler_started_at_utc: str | None
    scheduler_iteration_count: int
    scheduler_tick_in_progress: bool
    last_tick_started_at_utc: str | None
    last_tick_completed_at_utc: str | None
    last_tick_run_id: str | None
    last_tick_state: str | None
    next_scheduled_tick_at_utc: str | None
    consecutive_scheduler_failures: int
    last_scheduler_tick_started_at_utc: str | None
    last_scheduler_tick_completed_at_utc: str | None
    last_scheduler_tick_state: str | None
    last_scheduler_tick_error: str | None
    next_scheduled_run_at_utc: str | None
    last_tick_origin: str | None
    last_tick_result: dict[str, Any] | None


class AutopilotScheduler:
    """Run sequential orchestrator ticks on a worker thread inside one API process."""

    def __init__(
        self,
        settings: SchedulerSettings,
        status_path: Path,
        tick: Callable[[], AutopilotTickResult],
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.status_path = status_path
        self._tick = tick
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._tick_lock = asyncio.Lock()
        self._started_at: datetime | None = None
        self._iterations = 0
        self._last_started: datetime | None = None
        self._last_completed: datetime | None = None
        self._last_state: str | None = None
        self._last_error: str | None = None
        self._next_run: datetime | None = None
        self._last_result: dict[str, Any] | None = None
        self._consecutive_failures = 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start exactly one loop when enabled; repeated calls are harmless."""
        if not self.settings.enabled or self.running:
            return
        self._stop = asyncio.Event()
        self._started_at = self._utc_now()
        self._next_run = self._started_at + timedelta(seconds=self.settings.initial_delay_seconds)
        self._write_snapshot(running=True)
        self._task = asyncio.create_task(self._run_loop(), name="apex-pulse-autopilot")
        LOGGER.info(
            "Autopilot scheduler started interval_seconds=%s initial_delay_seconds=%s",
            self.settings.interval_seconds,
            self.settings.initial_delay_seconds,
        )

    async def stop(self) -> None:
        """Stop cleanly and wait for an in-flight worker-thread tick to finish."""
        if self._task is None:
            return
        self._stop.set()
        await self._task
        self._task = None
        self._next_run = None
        self._write_snapshot(running=False)
        LOGGER.info("Autopilot scheduler stopped")

    async def run_once(self) -> bool:
        """Run one non-overlapping tick; return false if a tick is already active."""
        if self._tick_lock.locked():
            LOGGER.warning("Autopilot scheduler skipped overlapping tick")
            return False
        async with self._tick_lock:
            self._last_started = self._utc_now()
            self._last_error = None
            self._write_snapshot(running=self.running, tick_in_progress=True)
            try:
                result = await asyncio.to_thread(self._tick)
                self._last_result = result.to_dict()
                self._last_state = result.orchestrator_state_after
                self._consecutive_failures = 0
                LOGGER.info(
                    "Autopilot scheduler tick completed run_id=%s state=%s action=%s",
                    result.run_id,
                    result.orchestrator_state_after,
                    result.action_taken,
                )
            except Exception as exc:  # the loop must survive one failed tick
                self._last_error = _safe_message(exc)
                self._last_state = "SCHEDULER_TICK_ERROR"
                self._consecutive_failures += 1
                LOGGER.exception("Autopilot scheduler tick failed")
            finally:
                self._iterations += 1
                self._last_completed = self._utc_now()
                self._write_snapshot(running=self.running, tick_in_progress=False)
            return True

    async def _run_loop(self) -> None:
        try:
            if await self._wait_or_stop(self.settings.initial_delay_seconds):
                return
            while not self._stop.is_set():
                await self.run_once()
                self._next_run = self._utc_now() + timedelta(seconds=self.settings.interval_seconds)
                self._write_snapshot(running=True)
                if await self._wait_or_stop(self.settings.interval_seconds):
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._last_error = _safe_message(exc)
            self._last_state = "SCHEDULER_LOOP_ERROR"
            self._consecutive_failures += 1
            self._next_run = None
            self._write_snapshot(running=False)
            LOGGER.exception("Autopilot scheduler loop failed unexpectedly")

    async def _wait_or_stop(self, seconds: int) -> bool:
        if seconds == 0:
            return self._stop.is_set()
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except TimeoutError:
            return False
        return True

    def _write_snapshot(self, *, running: bool, tick_in_progress: bool = False) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self.snapshot(running=running, tick_in_progress=tick_in_progress))
        temporary = self.status_path.with_name(f".{self.status_path.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.status_path)

    def snapshot(self, *, running: bool, tick_in_progress: bool = False) -> SchedulerSnapshot:
        return SchedulerSnapshot(
            schema_version=SCHEDULER_SCHEMA_VERSION,
            scheduler_enabled=self.settings.enabled,
            scheduler_running=running,
            scheduler_interval_seconds=self.settings.interval_seconds,
            scheduler_initial_delay_seconds=self.settings.initial_delay_seconds,
            scheduler_started_at_utc=_iso(self._started_at),
            scheduler_iteration_count=self._iterations,
            scheduler_tick_in_progress=tick_in_progress,
            last_tick_started_at_utc=_iso(self._last_started),
            last_tick_completed_at_utc=_iso(self._last_completed),
            last_tick_run_id=(str(self._last_result.get("run_id")) if self._last_result else None),
            last_tick_state=self._last_state,
            next_scheduled_tick_at_utc=_iso(self._next_run),
            consecutive_scheduler_failures=self._consecutive_failures,
            last_scheduler_tick_started_at_utc=_iso(self._last_started),
            last_scheduler_tick_completed_at_utc=_iso(self._last_completed),
            last_scheduler_tick_state=self._last_state,
            last_scheduler_tick_error=self._last_error,
            next_scheduled_run_at_utc=_iso(self._next_run),
            last_tick_origin="scheduler" if self._last_started else None,
            last_tick_result=self._last_result,
        )

    def _utc_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Scheduler clock must be timezone-aware")
        return value.astimezone(timezone.utc)


def build_production_scheduler(
    environ: Mapping[str, str] | None = None,
) -> AutopilotScheduler:
    """Build the production scheduler over the same direct operation used by the CLI."""
    environment = os.environ if environ is None else environ
    settings = SchedulerSettings.from_environ(environment)
    data_config = load_data_config()
    model_config = load_model_config(project_root=data_config.project_root)
    feature_config = load_feature_config(project_root=data_config.project_root)
    autopilot_config = load_autopilot_config(data_config.project_root / "configs/autopilot.yaml")

    def tick() -> AutopilotTickResult:
        return run_autopilot_tick(
            data_config,
            model_config,
            feature_config,
            autopilot_config=autopilot_config,
            environ=environment,
        )

    return AutopilotScheduler(
        settings,
        data_config.metrics_output_dir / SCHEDULER_STATUS_FILE,
        tick,
    )


def validate_scheduler_status(payload: Any) -> dict[str, Any]:
    required = set(SchedulerSnapshot.__dataclass_fields__)
    if not isinstance(payload, dict) or required - set(payload):
        raise ValueError("Autopilot scheduler status schema is incomplete")
    if payload.get("schema_version") != SCHEDULER_SCHEMA_VERSION:
        raise ValueError("Unsupported autopilot scheduler status schema")
    if payload.get("last_tick_result") is not None and not isinstance(
        payload["last_tick_result"], dict
    ):
        raise ValueError("Autopilot scheduler last tick result is invalid")
    return payload


def scheduler_environment_status(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    settings = SchedulerSettings.from_environ(environ)
    return asdict(
        SchedulerSnapshot(
            schema_version=SCHEDULER_SCHEMA_VERSION,
            scheduler_enabled=settings.enabled,
            scheduler_running=False,
            scheduler_interval_seconds=settings.interval_seconds,
            scheduler_initial_delay_seconds=settings.initial_delay_seconds,
            scheduler_started_at_utc=None,
            scheduler_iteration_count=0,
            scheduler_tick_in_progress=False,
            last_tick_started_at_utc=None,
            last_tick_completed_at_utc=None,
            last_tick_run_id=None,
            last_tick_state=None,
            next_scheduled_tick_at_utc=None,
            consecutive_scheduler_failures=0,
            last_scheduler_tick_started_at_utc=None,
            last_scheduler_tick_completed_at_utc=None,
            last_scheduler_tick_state=None,
            last_scheduler_tick_error=None,
            next_scheduled_run_at_utc=None,
            last_tick_origin=None,
            last_tick_result=None,
        )
    )


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _integer(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("Scheduler timing environment values must be integers") from exc


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _safe_message(exc: Exception) -> str:
    return " ".join(str(exc).split())[:500] or type(exc).__name__
