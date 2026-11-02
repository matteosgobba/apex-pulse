"""Normalize free, completed OpenF1 sessions for the existing lap pipeline.

The adapter's accuracy flag is a conservative sector-consistency proxy, not
FastF1's IsAccurate signal. Its version and limitations accompany every export.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import fastf1
import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.data.cache import initialize_fastf1_cache
from f1_prediction.data.fastf1_loader import FastF1SessionData
from f1_prediction.data.openf1_client import OpenF1Client, OpenF1UnavailableError
from f1_prediction.data.schema import validate_lap_schema

LOGGER = logging.getLogger(__name__)
ADAPTER_VERSION = "OPENF1_HISTORICAL_LAPS_V1"
SESSION_NAMES = {"FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3", "Q": "Qualifying"}
FREE_ACCESS_DELAY = timedelta(minutes=31)


def load_openf1_session_data(
    season: int,
    event: str,
    session_identifier: str,
    config: DataConfig,
    *,
    now: datetime | None = None,
    client: OpenF1Client | None = None,
) -> FastF1SessionData:
    """Load only the exact calendar session after its free historical window opens.

    Session matching uses year, session type, location and UTC date, never a
    'latest' alias. Practice loads never request qualifying laps or results.
    """
    code = session_identifier.upper()
    if season < 2023:
        raise ValueError(
            "OpenF1 historical coverage starts in 2023; older local caches are preserved."
        )
    if code not in SESSION_NAMES:
        raise ValueError(f"Unsupported OpenF1 session: {code}")
    if config.load_telemetry or config.load_weather or config.load_messages:
        raise ValueError(
            "OpenF1 lap adapter does not yet support optional telemetry/weather/messages"
        )
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None or clock.utcoffset() is None:
        raise ValueError("OpenF1 clock must be timezone-aware")
    initialize_fastf1_cache(config.fastf1_cache_dir)
    calendar = fastf1.get_event(season, event)
    expected_start = pd.Timestamp(calendar.get_session_date(code, utc=True))
    # FastF1 stores UTC session dates as timezone-naive pandas timestamps.
    if expected_start.tzinfo is None:
        expected_start = expected_start.tz_localize("UTC")
    api = client or OpenF1Client(config.fastf1_cache_dir.parent / "openf1_cache")
    sessions = api.get(
        "sessions",
        year=season,
        session_name=SESSION_NAMES[code],
        country_name=str(calendar["Country"]),
    )
    matches = [
        row
        for row in sessions
        if int(row["year"]) == season
        and row["session_name"] == SESSION_NAMES[code]
        and row.get("location") == calendar["Location"]
        and abs((pd.Timestamp(row["date_start"]) - expected_start).total_seconds()) <= 300
        and not row.get("is_cancelled", False)
    ]
    if len(matches) != 1:
        raise OpenF1UnavailableError(
            f"OpenF1 exact session match unavailable for {season} {event} {code}; "
            "year, location, type and UTC start must match the calendar."
        )
    session = matches[0]
    end = pd.Timestamp(session["date_end"]).to_pydatetime()
    if clock < end + FREE_ACCESS_DELAY:
        raise OpenF1UnavailableError(
            f"OpenF1 {season} {event} {code} is not historical yet; "
            f"free access opens after {(end + FREE_ACCESS_DELAY).isoformat()}."
        )
    key = int(session["session_key"])
    payload = {
        endpoint: api.get(endpoint, session_key=key, allow_empty_not_found=endpoint == "pit")
        for endpoint in ("laps", "drivers", "stints", "pit", "race_control")
    }
    if not payload["laps"] or not payload["drivers"] or not payload["stints"]:
        raise OpenF1UnavailableError(f"OpenF1 {key} has incomplete laps, drivers or tyre stints.")
    # Mini-sector pit markers also cover sessions without published pit rows.
    # Deleted laps require published race-control evidence.
    if not payload["race_control"]:
        raise OpenF1UnavailableError(f"OpenF1 {key} is missing race-control evidence.")
    if not any(
        row.get("message") == "SESSION FINISHED" or row.get("flag") == "CHEQUERED"
        for row in payload["race_control"]
    ):
        raise OpenF1UnavailableError(f"OpenF1 {key} has no session-completion evidence.")
    for rows in payload.values():
        if any(int(row.get("session_key", -1)) != key for row in rows):
            raise ValueError("OpenF1 response contains another session's data")
    identity_fallbacks = _complete_driver_identity(api, session, payload)
    laps = normalize_openf1_laps(payload)
    if not laps["IsAccurate"].any():
        raise OpenF1UnavailableError(f"OpenF1 {key} has no complete, sector-consistent laps yet.")
    if code == "Q":
        results = api.get("session_result", session_key=key)
        if len(results) < 10 or any(int(row.get("session_key", -1)) != key for row in results):
            raise OpenF1UnavailableError(f"OpenF1 {key} qualifying results are incomplete.")
    validate_lap_schema(laps)
    LOGGER.info("Loaded OpenF1 %s %s %s: %s laps", season, event, code, len(laps))
    return FastF1SessionData(
        season=season,
        event_input=event,
        event_name=str(calendar["EventName"]),
        session_input=code,
        session_name=SESSION_NAMES[code],
        laps=laps,
        drivers=tuple(laps["Driver"].drop_duplicates()),
        data_source="openf1",
        source_metadata={
            "adapter_version": ADAPTER_VERSION,
            "session_key": key,
            "meeting_key": int(session["meeting_key"]),
            "session_start_utc": session["date_start"],
            "session_end_utc": session["date_end"],
            "access": "free_historical",
            "accuracy_flag": "sector_consistency_proxy",
            "prior_practice_identity_sources": identity_fallbacks,
            "pit_evidence": "pit_and_mini_sectors" if payload["pit"] else "mini_sectors_only",
            "uncertain_driver_numbers": sorted(
                laps.loc[laps["LapQualityUncertain"], "DriverNumber"].unique().tolist()
            ),
            "limitations": [
                "IsAccurate is a sector-consistency and flag proxy, not FastF1's accuracy flag.",
                "Source-dependent lap counts and validity may differ from FastF1 training data.",
                "Telemetry, weather and live paid access are not used.",
            ],
        },
    )


def normalize_openf1_laps(payload: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    """Map observed timings/tyres and conservatively derive lap-quality flags.

    Tyre life counts the completed lap (initial age + lap offset + 1). In-laps
    use timestamped pit entry; stint boundaries alone cannot prove pit entry.
    A positive lap and all sectors must agree within 3 ms. Unknown start times,
    tyre stints, pit/unknown mini-sector codes and deleted laps cannot be accurate.
    Missing mini-sector samples alone do not invalidate complete sector timing.
    """
    drivers = {int(row["driver_number"]): row for row in payload["drivers"]}
    deleted, uncertain_drivers = _deleted_laps(payload["race_control"], payload["laps"])
    rows: list[dict[str, Any]] = []
    for lap in payload["laps"]:
        number, lap_number = int(lap["driver_number"]), int(lap["lap_number"])
        driver = drivers.get(number)
        if driver is None or not driver.get("name_acronym") or not driver.get("team_name"):
            raise ValueError(f"OpenF1 missing driver/team identity for car {number}")
        stints = [
            stint
            for stint in payload["stints"]
            if int(stint["driver_number"]) == number
            and int(stint["lap_start"]) <= lap_number <= int(stint["lap_end"])
        ]
        if len(stints) > 1:
            raise ValueError(f"OpenF1 overlapping tyre stints for car {number}, lap {lap_number}")
        stint = stints[0] if stints else {}
        start = pd.to_datetime(lap.get("date_start"), utc=True)
        duration = lap.get("lap_duration")
        end = start + pd.to_timedelta(duration, unit="s") if duration is not None else pd.NaT
        in_lap = any(
            int(pit["driver_number"]) == number
            and pd.notna(start)
            and pd.notna(end)
            and start <= pd.Timestamp(pit["date"]) <= end
            for pit in payload["pit"]
        )
        sectors = [lap.get(f"duration_sector_{i}") for i in (1, 2, 3)]
        consistent = (
            duration is not None
            and duration > 0
            and all(value is not None and value > 0 for value in sectors)
            and abs(sum(sectors) - duration) <= 0.0030001
        )
        # These colors describe relative pace, not track flags. A yellow
        # mini-sector (2048) is slower pace and remains a valid timing signal.
        segments = [value for i in (1, 2, 3) for value in (lap.get(f"segments_sector_{i}") or [])]
        in_lap = in_lap or 2064 in segments
        clean_segments = bool(segments) and all(
            value in {None, 0, 2048, 2049, 2051} for value in segments
        )
        is_deleted = (number, lap_number) in deleted
        accurate = bool(
            consistent
            and clean_segments
            and pd.notna(start)
            and stint
            and not _neutralized_lap(start, end, payload["race_control"])
            and not in_lap
            and not lap.get("is_pit_out_lap")
            and not is_deleted
            and number not in uncertain_drivers
        )
        rows.append(
            {
                "Driver": driver["name_acronym"],
                "DriverNumber": str(number),
                "FullName": driver.get("full_name"),
                "Team": driver["team_name"],
                "LapNumber": lap_number,
                "LapTime": duration,
                **{f"Sector{i}Time": lap.get(f"duration_sector_{i}") for i in (1, 2, 3)},
                "Stint": stint.get("stint_number"),
                "Compound": stint.get("compound"),
                "TyreLife": (
                    stint["tyre_age_at_start"] + lap_number - stint["lap_start"] + 1
                    if stint.get("tyre_age_at_start") is not None
                    else None
                ),
                "IsAccurate": accurate,
                "Deleted": is_deleted,
                "PitInTime": start if in_lap else pd.NaT,
                "PitOutTime": start if lap.get("is_pit_out_lap") else pd.NaT,
                "LapStartDate": start,
                "LapEndDate": end,
                "DataSource": "openf1",
                "SourceAdapterVersion": ADAPTER_VERSION,
                "LapQualityUncertain": number in uncertain_drivers,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty and frame.duplicated(["DriverNumber", "LapNumber"]).any():
        raise ValueError("OpenF1 duplicate driver/lap keys")
    return frame


def _neutralized_lap(start, end, messages) -> bool:
    """Exclude laps overlapping a red flag, safety car or virtual safety car."""
    if pd.isna(start) or pd.isna(end):
        return True
    active = False
    for row in sorted(messages, key=lambda item: item["date"]):
        timestamp = pd.Timestamp(row["date"])
        if timestamp > end:
            break
        message = str(row.get("message") or "").upper()
        if row.get("flag") == "RED" or message in {
            "SAFETY CAR DEPLOYED",
            "VIRTUAL SAFETY CAR DEPLOYED",
            "SESSION ABORTED",
        }:
            active = True
        elif message in {"TRACK CLEAR", "GREEN LIGHT - PIT EXIT OPEN", "SESSION RESUMED"} or (
            row.get("flag") == "GREEN" and row.get("scope") == "Track"
        ):
            if active and timestamp >= start:
                return True
            active = False
        if active and timestamp >= start:
            return True
    return active


def _complete_driver_identity(api, session, payload) -> list[dict[str, int]]:
    """Recover omitted identity only from earlier practice in the same meeting.

    Lap participation still defines the roster. This never adds a driver without
    observed laps or reads future qualifying/race participants.
    """
    missing = {int(row["driver_number"]) for row in payload["laps"]} - {
        int(row["driver_number"]) for row in payload["drivers"]
    }
    provenance = []
    if not missing:
        return provenance
    sessions = api.get("sessions", meeting_key=session["meeting_key"])
    previous = sorted(
        [
            row
            for row in sessions
            if row["session_name"] in {"Practice 1", "Practice 2", "Practice 3"}
            and int(row["meeting_key"]) == int(session["meeting_key"])
            and pd.Timestamp(row["date_start"]) < pd.Timestamp(session["date_start"])
        ],
        key=lambda row: row["date_start"],
        reverse=True,
    )
    for earlier in previous:
        for driver in api.get("drivers", session_key=earlier["session_key"]):
            number = int(driver["driver_number"])
            if number in missing and int(driver["session_key"]) == int(earlier["session_key"]):
                payload["drivers"].append(driver)
                missing.remove(number)
                provenance.append(
                    {"driver_number": number, "session_key": int(earlier["session_key"])}
                )
        if not missing:
            break
    if missing:
        raise OpenF1UnavailableError(
            f"OpenF1 missing observed driver identities: {sorted(missing)}"
        )
    return provenance


def _deleted_laps(
    messages: list[dict[str, Any]], laps: list[dict[str, Any]]
) -> tuple[set[tuple[int, int]], set[int]]:
    deleted: set[tuple[int, int]] = set()
    uncertain: set[int] = set()
    for row in sorted(messages, key=lambda item: item["date"]):
        message = str(row.get("message") or "").upper()
        if "DELETED" not in message and "REINSTATED" not in message:
            continue
        car = re.search(r"CAR\s+(\d+)", message)
        lap = re.search(r"\bLAP\s+(\d+)\b", message)
        number = row.get("driver_number") or (int(car[1]) if car else None)
        lap_number = row.get("lap_number") or (int(lap[1]) if lap else None)
        if number is None:
            raise OpenF1UnavailableError(
                "OpenF1 deleted/reinstated lap cannot be identified safely."
            )
        if lap_number is None:
            timed = re.search(r"\bTIME\s+(\d+):(\d+\.\d+)", message)
            matches = [
                lap
                for lap in laps
                if int(lap["driver_number"]) == int(number)
                and timed is not None
                and lap.get("lap_duration") is not None
                and abs(lap["lap_duration"] - (int(timed[1]) * 60 + float(timed[2]))) < 0.0005
            ]
            if len(matches) != 1:
                # Do not invent a deleted lap number. Preserve participation but
                # exclude this driver's ambiguous timing from valid/push pace.
                uncertain.add(int(number))
                continue
            lap_number = int(matches[0]["lap_number"])
        key = int(number), int(lap_number)
        if "REINSTATED" in message:
            deleted.discard(key)
        else:
            deleted.add(key)
            if "NEXT LAP" in message:
                deleted.add((int(number), int(lap_number) + 1))
    return deleted, uncertain
