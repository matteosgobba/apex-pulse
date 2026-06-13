"""Batch ingestion orchestration for historical race weekends."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import fastf1

from f1_prediction.config import DataConfig
from f1_prediction.data.fastf1_loader import (
    FastF1SessionData,
    build_lap_output_path,
    load_fastf1_session_data,
    save_laps,
)
from f1_prediction.utils.paths import ensure_directory, slugify

DEFAULT_EVENT_SESSIONS: tuple[str, ...] = ("FP1", "FP2", "FP3", "Q")
ProgressCallback = Callable[[str], None]
IngestionStatus = Literal["success", "skipped", "failed"]


@dataclass(frozen=True)
class SessionIngestionResult:
    """Outcome for one requested session."""

    session: str
    status: IngestionStatus
    laps_path: Path
    metadata_path: Path
    n_laps: int = 0
    n_drivers: int = 0
    error_message: str | None = None


@dataclass(frozen=True)
class EventIngestionSummary:
    """Aggregate results for one event ingestion request."""

    season: int
    event: str
    results: tuple[SessionIngestionResult, ...]

    @property
    def success_count(self) -> int:
        return sum(result.status == "success" for result in self.results)

    @property
    def skipped_count(self) -> int:
        return sum(result.status == "skipped" for result in self.results)

    @property
    def failed_count(self) -> int:
        return sum(result.status == "failed" for result in self.results)


def ingest_event(
    season: int,
    event: str,
    config: DataConfig,
    sessions: Sequence[str] = DEFAULT_EVENT_SESSIONS,
    *,
    force: bool = False,
    fail_fast: bool = False,
    progress: ProgressCallback | None = None,
) -> EventIngestionSummary:
    """Ingest requested sessions and collect failures unless fail-fast is enabled."""
    requested_sessions = normalize_session_identifiers(sessions)
    results: list[SessionIngestionResult] = []

    for session_identifier in requested_sessions:
        laps_path = build_lap_output_path(
            config.lap_output_dir,
            season,
            event,
            session_identifier,
        )
        metadata_path = build_metadata_output_path(
            config.session_metadata_output_dir,
            season,
            event,
            session_identifier,
        )

        if not force and _outputs_are_complete(laps_path, metadata_path):
            result = SessionIngestionResult(
                session=session_identifier,
                status="skipped",
                laps_path=laps_path,
                metadata_path=metadata_path,
            )
            results.append(result)
            _report(progress, f"SKIP {session_identifier}: output files already exist")
            continue

        _report(progress, f"LOAD {session_identifier}: starting")
        try:
            session_data = load_fastf1_session_data(
                season=season,
                event=event,
                session_identifier=session_identifier,
                config=config,
            )
            save_laps(session_data.laps, laps_path)
            save_session_metadata(
                build_success_metadata(session_data, config, laps_path),
                metadata_path,
            )
            result = SessionIngestionResult(
                session=session_identifier,
                status="success",
                laps_path=laps_path,
                metadata_path=metadata_path,
                n_laps=len(session_data.laps),
                n_drivers=len(session_data.drivers),
            )
            _report(
                progress,
                f"OK   {session_identifier}: {result.n_laps} laps, {result.n_drivers} drivers",
            )
        except Exception as exc:
            error_message = concise_error(exc)
            try:
                save_session_metadata(
                    build_failed_metadata(
                        season=season,
                        event=event,
                        session_identifier=session_identifier,
                        config=config,
                        laps_path=laps_path,
                        error_message=error_message,
                    ),
                    metadata_path,
                )
            except Exception as metadata_exc:
                error_message = (
                    f"{error_message}; failed to save metadata: {concise_error(metadata_exc)}"
                )
            result = SessionIngestionResult(
                session=session_identifier,
                status="failed",
                laps_path=laps_path,
                metadata_path=metadata_path,
                error_message=error_message,
            )
            _report(progress, f"FAIL {session_identifier}: {error_message}")

        results.append(result)
        if result.status == "failed" and fail_fast:
            break

    return EventIngestionSummary(season=season, event=event, results=tuple(results))


def normalize_session_identifiers(sessions: Sequence[str]) -> tuple[str, ...]:
    """Normalize session identifiers while preserving order and removing duplicates."""
    normalized: list[str] = []
    for session in sessions:
        identifier = session.strip().upper()
        if not identifier:
            raise ValueError("Session identifiers cannot be empty")
        if identifier not in normalized:
            normalized.append(identifier)
    if not normalized:
        raise ValueError("At least one session must be requested")
    return tuple(normalized)


def build_metadata_output_path(
    output_dir: Path,
    season: int,
    event: str,
    session_identifier: str,
) -> Path:
    """Build the deterministic output path for session metadata."""
    event_dir = ensure_directory(output_dir / str(season) / slugify(event))
    return event_dir / f"{slugify(session_identifier)}_metadata.json"


def build_success_metadata(
    session_data: FastF1SessionData,
    config: DataConfig,
    laps_path: Path,
) -> dict[str, object]:
    """Build portable metadata for a successful session load."""
    return _base_metadata(
        season=session_data.season,
        event_input=session_data.event_input,
        event_name=session_data.event_name,
        session_input=session_data.session_input,
        session_name=session_data.session_name,
        config=config,
        laps_path=laps_path,
        n_laps=len(session_data.laps),
        drivers=list(session_data.drivers),
        status="success",
        error_message=None,
    )


def build_failed_metadata(
    season: int,
    event: str,
    session_identifier: str,
    config: DataConfig,
    laps_path: Path,
    error_message: str,
) -> dict[str, object]:
    """Build metadata describing a failed session load."""
    return _base_metadata(
        season=season,
        event_input=event,
        event_name=event,
        session_input=session_identifier,
        session_name=session_identifier,
        config=config,
        laps_path=laps_path,
        n_laps=0,
        drivers=[],
        status="failed",
        error_message=error_message,
    )


def save_session_metadata(metadata: dict[str, object], output_path: Path) -> None:
    """Write session metadata as readable UTF-8 JSON."""
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)
        metadata_file.write("\n")


def concise_error(exc: Exception) -> str:
    """Return a single-line exception summary suitable for metadata and CLI output."""
    message = " ".join(str(exc).split()) or "No error details were provided"
    return f"{type(exc).__name__}: {message}"


def _base_metadata(
    *,
    season: int,
    event_input: str,
    event_name: str,
    session_input: str,
    session_name: str,
    config: DataConfig,
    laps_path: Path,
    n_laps: int,
    drivers: list[str],
    status: Literal["success", "failed"],
    error_message: str | None,
) -> dict[str, object]:
    return {
        "season": season,
        "event_input": event_input,
        "event_name": event_name,
        "event_slug": slugify(event_input),
        "session_input": session_input,
        "session_name": session_name,
        "session_slug": slugify(session_input),
        "loaded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "n_laps": n_laps,
        "n_drivers": len(drivers),
        "drivers": drivers,
        "output_laps_path": _portable_path(laps_path, config.project_root),
        "fastf1_version": fastf1.__version__,
        "status": status,
        "error_message": error_message,
    }


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _outputs_are_complete(laps_path: Path, metadata_path: Path) -> bool:
    if not laps_path.is_file() or not metadata_path.is_file():
        return False
    try:
        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(metadata, dict) and metadata.get("status") == "success"


def _report(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
