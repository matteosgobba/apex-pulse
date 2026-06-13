"""Load a FastF1 session and persist its basic lap data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fastf1
import pandas as pd
from fastf1.exceptions import DataNotLoadedError

from f1_prediction.config import DataConfig
from f1_prediction.data.cache import initialize_fastf1_cache
from f1_prediction.data.schema import select_basic_lap_data, validate_lap_schema
from f1_prediction.utils.paths import ensure_directory, slugify

LOGGER = logging.getLogger(__name__)


class SessionDataUnavailableError(RuntimeError):
    """Raised when FastF1 cannot provide lap timing data for a session."""


@dataclass(frozen=True)
class FastF1SessionData:
    """Normalized lap data and identifiers loaded from one FastF1 session."""

    season: int
    event_input: str
    event_name: str
    session_input: str
    session_name: str
    laps: pd.DataFrame
    drivers: tuple[str, ...]


@dataclass(frozen=True)
class SessionLoadResult:
    """Summary of a loaded and persisted FastF1 session."""

    season: int
    event: str
    session: str
    driver_count: int
    lap_count: int
    output_path: Path


def load_fastf1_session(
    season: int,
    event: str,
    session_identifier: str,
    config: DataConfig,
) -> SessionLoadResult:
    """Load one historical session and save its basic laps as Parquet."""
    session_data = load_fastf1_session_data(
        season=season,
        event=event,
        session_identifier=session_identifier,
        config=config,
    )
    output_path = build_lap_output_path(
        output_dir=config.lap_output_dir,
        season=season,
        event=event,
        session_identifier=session_identifier,
    )
    save_laps(session_data.laps, output_path)
    LOGGER.info(
        "Saved %s laps for %s drivers to %s",
        len(session_data.laps),
        len(session_data.drivers),
        output_path,
    )

    return SessionLoadResult(
        season=season,
        event=session_data.event_name,
        session=_session_label(session_data.session_name, session_identifier),
        driver_count=len(session_data.drivers),
        lap_count=len(session_data.laps),
        output_path=output_path,
    )


def load_fastf1_session_data(
    season: int,
    event: str,
    session_identifier: str,
    config: DataConfig,
) -> FastF1SessionData:
    """Load and normalize one FastF1 session without persisting outputs."""
    cache_dir = initialize_fastf1_cache(config.fastf1_cache_dir)
    LOGGER.info("FastF1 cache initialized at %s", cache_dir)
    LOGGER.info("Loading %s %s %s", season, event, session_identifier)

    fastf1_session = fastf1.get_session(season, event, session_identifier)
    fastf1_session.load(
        laps=True,
        telemetry=config.load_telemetry,
        weather=config.load_weather,
        messages=config.load_messages,
    )

    try:
        raw_laps = fastf1_session.laps
    except DataNotLoadedError as exc:
        raise SessionDataUnavailableError(
            "FastF1 did not provide lap timing data for "
            f"{season} {event} {session_identifier}. "
            "Check network or FastF1 backend availability and retry."
        ) from exc

    laps = select_basic_lap_data(raw_laps)
    validate_lap_schema(laps)

    event_name = _event_name(fastf1_session, event)
    session_name = str(getattr(fastf1_session, "name", session_identifier))
    return FastF1SessionData(
        season=season,
        event_input=event,
        event_name=event_name,
        session_input=session_identifier,
        session_name=session_name,
        laps=laps,
        drivers=_driver_codes(laps),
    )


def build_lap_output_path(
    output_dir: Path,
    season: int,
    event: str,
    session_identifier: str,
) -> Path:
    """Build a deterministic project-relative layout for saved session laps."""
    event_dir = ensure_directory(output_dir / str(season) / slugify(event))
    return event_dir / f"{slugify(session_identifier)}_laps.parquet"


def save_laps(laps: pd.DataFrame, output_path: Path) -> None:
    """Persist lap data using the PyArrow Parquet engine."""
    ensure_directory(output_path.parent)
    laps.to_parquet(output_path, engine="pyarrow", index=False)


def _driver_codes(laps: pd.DataFrame) -> tuple[str, ...]:
    if "Driver" not in laps:
        return ()
    drivers = laps["Driver"].dropna().astype(str)
    return tuple(drivers[drivers.str.len() > 0].drop_duplicates())


def _event_name(session: Any, fallback: str) -> str:
    event = getattr(session, "event", None)
    if event is None:
        return fallback
    try:
        event_name = event.get("EventName")
    except AttributeError:
        return fallback
    return str(event_name) if event_name else fallback


def _session_label(session_name: str, session_identifier: str) -> str:
    identifier = session_identifier.upper()
    if session_name.upper() == identifier:
        return session_name
    return f"{session_name} ({identifier})"
