from __future__ import annotations

import copy
import io
import json
from dataclasses import replace
from datetime import datetime, timezone
from urllib.error import HTTPError

import pandas as pd
import pytest

from f1_prediction.config import load_data_config
from f1_prediction.data.ingest import ingest_event
from f1_prediction.data.openf1_client import OpenF1Client, OpenF1UnavailableError
from f1_prediction.data.openf1_loader import load_openf1_session_data, normalize_openf1_laps
from f1_prediction.features.lap_cleaning import clean_session_laps

UTC = timezone.utc


def payload():
    return {
        "laps": [
            {
                "session_key": 9,
                "driver_number": 30,
                "lap_number": 2,
                "date_start": "2026-10-31T18:00:00+00:00",
                "lap_duration": 80.0,
                "duration_sector_1": 20.0,
                "duration_sector_2": 30.0,
                "duration_sector_3": 30.0,
                "is_pit_out_lap": False,
                "segments_sector_1": [2048, 2049],
                "segments_sector_2": [2051],
                "segments_sector_3": [2049],
            }
        ],
        "drivers": [
            {
                "session_key": 9,
                "driver_number": 30,
                "name_acronym": "LAW",
                "full_name": "Liam Lawson",
                "team_name": "Racing Bulls",
            }
        ],
        "stints": [
            {
                "session_key": 9,
                "driver_number": 30,
                "stint_number": 1,
                "lap_start": 1,
                "lap_end": 5,
                "compound": "SOFT",
                "tyre_age_at_start": 2,
            }
        ],
        "pit": [{"session_key": 9, "driver_number": 30, "date": "2026-10-31T18:09:00Z"}],
        "race_control": [
            {
                "session_key": 9,
                "date": "2026-10-31T18:30:00Z",
                "message": "SESSION FINISHED",
            }
        ],
    }


def test_adapter_preserves_observed_times_identity_and_tyre_age():
    raw = normalize_openf1_laps(payload())
    clean = clean_session_laps(raw, season=2026, event="Mexico", session="FP3")
    assert raw.loc[0, "Driver"] == "LAW"
    assert raw.loc[0, "TyreLife"] == 4
    assert raw.loc[0, "DataSource"] == "openf1"
    assert clean.loc[0, "lap_time_sec"] == 80
    # Yellow mini-sector means slower pace, not a yellow track flag.
    assert clean.loc[0, "is_valid_lap"]


@pytest.mark.parametrize(
    "change",
    [
        {"duration_sector_1": None},
        {"duration_sector_1": 20.1},
        {"is_pit_out_lap": True},
        {"segments_sector_3": [2064]},
        {"segments_sector_3": [2052]},
        {"date_start": None},
    ],
)
def test_incomplete_pit_and_inconsistent_laps_are_not_accurate(change):
    data = payload()
    data["laps"][0].update(change)
    assert not normalize_openf1_laps(data).loc[0, "IsAccurate"]


def test_timestamped_pit_entry_excludes_in_lap():
    data = payload()
    data["pit"][0]["date"] = "2026-10-31T18:01:00Z"
    raw = normalize_openf1_laps(data)
    assert pd.notna(raw.loc[0, "PitInTime"])
    assert not raw.loc[0, "IsAccurate"]


def test_deleted_laps_and_reinstatements_are_applied_in_order():
    data = payload()
    data["race_control"].append(
        {
            "date": "2026-10-31T18:02:00Z",
            "message": "CAR 30 (LAW) TIME 1:20.000 DELETED LAP 2",
        }
    )
    assert normalize_openf1_laps(data).loc[0, "Deleted"]
    data["race_control"].append(
        {
            "date": "2026-10-31T18:03:00Z",
            "message": "CAR 30 (LAW) LAP 2 REINSTATED",
        }
    )
    assert not normalize_openf1_laps(data).loc[0, "Deleted"]


def test_unidentifiable_deleted_lap_blocks_instead_of_silently_accepting():
    data = payload()
    data["race_control"].append({"date": "2026-10-31T18:02:00Z", "message": "LAP DELETED"})
    with pytest.raises(OpenF1UnavailableError, match="cannot be identified"):
        normalize_openf1_laps(data)


def test_red_flag_overlapping_lap_excludes_it_even_after_track_clears():
    data = payload()
    data["race_control"].extend(
        [
            {"date": "2026-10-31T17:59:00Z", "flag": "RED"},
            {"date": "2026-10-31T18:00:30Z", "message": "TRACK CLEAR"},
        ]
    )
    assert not normalize_openf1_laps(data).loc[0, "IsAccurate"]


def test_duplicate_lap_and_missing_identity_fail_closed():
    data = payload()
    data["laps"].append(copy.deepcopy(data["laps"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        normalize_openf1_laps(data)
    data = payload()
    data["drivers"] = []
    with pytest.raises(ValueError, match="identity"):
        normalize_openf1_laps(data)


def test_ambiguous_deletion_excludes_driver_validity_without_inventing_deleted_laps():
    data = payload()
    data["race_control"].append(
        {
            "date": "2026-10-31T18:02:00Z",
            "message": "CAR 30 (LAW) LAP DELETED - TRACK LIMITS AT TURN 14 (NEXT LAP PIT)",
        }
    )
    raw = normalize_openf1_laps(data)
    assert raw.loc[0, "LapQualityUncertain"]
    assert not raw.loc[0, "IsAccurate"]
    assert not raw.loc[0, "Deleted"]


def test_deleted_lap_can_be_matched_to_unique_observed_time():
    data = payload()
    data["race_control"].append(
        {
            "date": "2026-10-31T18:02:00Z",
            "message": "CAR 30 (LAW) TIME 1:20.000 DELETED - TRACK LIMITS (NEXT LAP)",
        }
    )
    assert normalize_openf1_laps(data).loc[0, "Deleted"]


def test_partial_mini_sector_samples_do_not_erase_complete_sector_timing():
    data = payload()
    data["laps"][0]["segments_sector_1"] = [None, 2048, 2049]
    assert normalize_openf1_laps(data).loc[0, "IsAccurate"]


class Calendar(dict):
    def __init__(self):
        super().__init__(
            EventName="Mexico City Grand Prix", Location="Mexico City", Country="Mexico"
        )

    def get_session_date(self, code, utc=True):
        return pd.Timestamp("2026-10-31T17:30:00")


class Client:
    def __init__(self):
        self.calls = []
        self.data = payload()
        self.session = {
            "session_key": 9,
            "meeting_key": 1,
            "year": 2026,
            "session_name": "Practice 3",
            "location": "Mexico City",
            "date_start": "2026-10-31T17:30:00Z",
            "date_end": "2026-10-31T18:30:00Z",
        }

    def get(self, endpoint, **filters):
        self.calls.append((endpoint, filters))
        return [self.session] if endpoint == "sessions" else copy.deepcopy(self.data[endpoint])


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr("fastf1.get_event", lambda *args: Calendar())
    monkeypatch.setattr("fastf1.Cache.enable_cache", lambda *args: None)
    return replace(
        load_data_config(),
        project_root=tmp_path,
        fastf1_cache_dir=tmp_path / "cache",
        lap_output_dir=tmp_path / "laps",
        session_metadata_output_dir=tmp_path / "metadata",
        session_data_source="openf1",
    )


def test_live_window_never_requests_paid_laps(config):
    client = Client()
    with pytest.raises(OpenF1UnavailableError, match="not historical yet"):
        load_openf1_session_data(
            2026,
            "Mexico",
            "FP3",
            config,
            client=client,
            now=datetime(2026, 10, 31, 18, 50, tzinfo=UTC),
        )
    assert [call[0] for call in client.calls] == ["sessions"]


def test_exact_session_match_rejects_other_location_and_cancelled_session(config):
    client = Client()
    client.session["location"] = "Austin"
    with pytest.raises(OpenF1UnavailableError, match="exact session match"):
        load_openf1_session_data(2026, "Mexico", "FP3", config, client=client)
    assert len(client.calls) == 1
    client.session["location"] = "Mexico City"
    client.session["is_cancelled"] = True
    with pytest.raises(OpenF1UnavailableError, match="exact session match"):
        load_openf1_session_data(2026, "Mexico", "FP3", config, client=client)


def test_practice_load_never_requests_qualifying_and_persists_source_metadata(config, monkeypatch):
    client = Client()
    loaded = load_openf1_session_data(2026, "Mexico", "FP3", config, client=client)
    assert "session_result" not in [call[0] for call in client.calls]
    assert loaded.source_metadata["access"] == "free_historical"
    monkeypatch.setattr(
        "f1_prediction.data.openf1_loader.load_openf1_session_data",
        lambda **kwargs: loaded,
    )
    summary = ingest_event(2026, "Mexico", config, sessions=["FP3"])
    assert summary.failed_count == 0
    metadata = json.loads(summary.results[0].metadata_path.read_text())
    assert metadata["data_source"] == "openf1"
    assert metadata["source_metadata"]["accuracy_flag"] == "sector_consistency_proxy"
    before = summary.results[0].laps_path.read_bytes()
    assert ingest_event(2026, "Mexico", config, sessions=["FP3"]).skipped_count == 1
    assert summary.results[0].laps_path.read_bytes() == before


def test_cross_session_payload_rejected(config):
    client = Client()
    client.data["laps"][0]["session_key"] = 8
    with pytest.raises(ValueError, match="another session"):
        load_openf1_session_data(2026, "Mexico", "FP3", config, client=client)


def test_missing_identity_uses_only_earlier_practice_and_preserves_lap_participation(config):
    client = Client()
    client.data["drivers"] = []
    original = client.get

    def get(endpoint, **filters):
        if endpoint == "sessions" and "meeting_key" in filters:
            return [
                {
                    **client.session,
                    "session_key": 8,
                    "session_name": "Practice 2",
                    "date_start": "2026-10-30T20:00:00Z",
                },
                {
                    **client.session,
                    "session_key": 10,
                    "session_name": "Qualifying",
                    "date_start": "2026-10-31T21:00:00Z",
                },
            ]
        if endpoint == "drivers" and filters["session_key"] == 8:
            client.calls.append((endpoint, filters))
            return [{**payload()["drivers"][0], "session_key": 8}]
        return original(endpoint, **filters)

    client.get = get
    # Keep a different known participant in the session metadata so the initial
    # drivers response is published, while LAW is omitted (observed Mexico case).
    client.data["drivers"] = [
        {**payload()["drivers"][0], "driver_number": 44, "name_acronym": "HAM"}
    ]
    loaded = load_openf1_session_data(2026, "Mexico", "FP3", config, client=client)
    assert loaded.drivers == ("LAW",)
    assert loaded.source_metadata["prior_practice_identity_sources"] == [
        {"driver_number": 30, "session_key": 8}
    ]
    assert all(filters.get("session_key") != 10 for _, filters in client.calls)


def test_readiness_uses_configured_free_source(config, monkeypatch):
    from types import SimpleNamespace

    from f1_prediction.modeling.weekend_orchestrator import FastF1ReadinessProvider, ProbeStatus

    loaded = load_openf1_session_data(2026, "Mexico", "FP3", config, client=Client())
    monkeypatch.setattr(
        "f1_prediction.data.openf1_loader.load_openf1_session_data",
        lambda *args: loaded,
    )
    result = FastF1ReadinessProvider(config).probe(
        SimpleNamespace(season=2026, event="Mexico"),
        "FP3",
    )
    assert result.status == ProbeStatus.READY
    assert "OpenF1 free historical" in result.reason


def test_incomplete_publication_does_not_pass_readiness(config):
    client = Client()
    client.data["race_control"] = []
    with pytest.raises(OpenF1UnavailableError, match="evidence"):
        load_openf1_session_data(2026, "Mexico", "FP3", config, client=client)


def test_client_does_not_cache_auth_errors_or_empty_rows(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise HTTPError("url", 401, "Unauthorized", {}, None)

    monkeypatch.setattr("f1_prediction.data.openf1_client.urlopen", denied)
    client = OpenF1Client(tmp_path)
    with pytest.raises(OpenF1UnavailableError, match="401"):
        client.get("laps", session_key=9)
    assert not list(tmp_path.glob("*.json"))
    monkeypatch.setattr("f1_prediction.data.openf1_client.time.sleep", lambda *args: None)
    monkeypatch.setattr(
        "f1_prediction.data.openf1_client.urlopen", lambda *a, **k: io.BytesIO(b"[]")
    )
    assert client.get("laps", session_key=9) == []
    assert not list(tmp_path.glob("*.json"))


def test_only_optional_pit_404_is_an_empty_publication(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise HTTPError("url", 404, "Not Found", {}, None)

    monkeypatch.setattr("f1_prediction.data.openf1_client.urlopen", missing)
    monkeypatch.setattr("f1_prediction.data.openf1_client.time.sleep", lambda *args: None)
    client = OpenF1Client(tmp_path)
    assert client.get("pit", session_key=9, allow_empty_not_found=True) == []
    with pytest.raises(OpenF1UnavailableError, match="404"):
        client.get("laps", session_key=9)


def test_client_uses_cache_without_repeating_http_and_spaces_requests(tmp_path, monkeypatch):
    calls = []
    sleeps = []
    monkeypatch.setattr(
        "f1_prediction.data.openf1_client.urlopen",
        lambda *a, **k: calls.append(a[0]) or io.BytesIO(b'[{"session_key": 9}]'),
    )
    monkeypatch.setattr("f1_prediction.data.openf1_client.time.sleep", sleeps.append)
    client = OpenF1Client(tmp_path)
    assert client.get("laps", session_key=9) == [{"session_key": 9}]
    assert client.get("laps", session_key=9) == [{"session_key": 9}]
    client.get("drivers", session_key=9)
    assert len(calls) == 2
    assert len(sleeps) == 1 and sleeps[0] > 2
