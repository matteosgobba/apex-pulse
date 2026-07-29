import json
from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.config import DataConfig
from f1_prediction.dashboard.export import export_dashboard_artifacts
from f1_prediction.data.fastf1_loader import build_lap_output_path
from f1_prediction.data.qualifying_entry_list import (
    ENTRY_LIST_PARITY_PASSED,
    ENTRY_LIST_RESOLVED,
    LATEST_PRE_Q_SOURCE_PREFIX,
    audit_qualifying_entry_list,
    constrain_features_to_entry_list,
)
from tests.test_dashboard_export import (
    _read_dashboard,
    _registry_event,
    _write_forecasts,
    _write_protocol,
    _write_registry,
)


def test_fp3_is_selected_before_q_on_conventional_weekend(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_practice(config, fp1=("ALO", "STR"), fp2=("ALO", "STR"), fp3=("ALO", "STR"))

    audit = audit_qualifying_entry_list(
        config,
        season=2026,
        event="Belgian Grand Prix",
        allow_fastf1=False,
    )

    assert audit.forecast_allowed
    assert audit.summary["resolution_source"] == f"{LATEST_PRE_Q_SOURCE_PREFIX}:FP3"
    assert audit.summary["selected_source_session"] == "FP3"
    assert audit.summary["q_data_available"] is False
    assert audit.summary["q_data_required"] is False
    assert set(audit.drivers["driver"]) == {"ALO", "STR"}
    assert audit.exclusions.empty


def test_latest_session_not_union_excludes_fp1_fp2_only_driver(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_practice(
        config,
        fp1=("VER", "NOR", "DEV"),
        fp2=("VER", "NOR", "DEV"),
        fp3=("VER", "NOR"),
    )

    audit = audit_qualifying_entry_list(
        config,
        season=2026,
        event="Belgian Grand Prix",
        allow_fastf1=False,
    )

    assert not audit.forecast_allowed
    assert "identity_or_team_mismatch" in set(audit.failures["check_name"])
    assert "DEV" not in set(audit.drivers["driver"])


def test_replacement_present_in_fp3_is_included_with_authoritative_roster(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_practice(config, fp1=("VER", "REG"), fp2=("VER", "REG"), fp3=("VER", "BEA"))
    _write_race_roster(config, [("VER", "Team VER"), ("BEA", "Team BEA")])
    features = _features([("VER", "Team VER"), ("REG", "Team REG"), ("BEA", "Team BEA")])

    constrained, audit = constrain_features_to_entry_list(
        config,
        season=2026,
        event="Belgian Grand Prix",
        event_order=12,
        feature_rows=features,
        allow_fastf1=False,
    )

    assert set(constrained["driver"]) == {"VER", "BEA"}
    assert "REG" not in set(constrained["driver"])
    assert audit.summary["resolution_source"] == "authoritative_race_driver_roster"


def test_regular_roster_driver_absent_from_fp3_is_retained(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_practice(config, fp1=("VER", "PER", "CRA"), fp2=("VER", "PER"), fp3=("VER",))
    _write_race_roster(config, [("VER", "Team VER"), ("PER", "Team PER")])

    audit = audit_qualifying_entry_list(
        config,
        season=2026,
        event="Belgian Grand Prix",
        allow_fastf1=False,
    )

    assert audit.forecast_allowed
    assert set(audit.drivers["driver"]) == {"VER", "PER"}
    assert set(audit.exclusions["driver"]) == {"CRA"}
    assert "PER" not in set(audit.exclusions["driver"])


def test_official_entrant_without_latest_checkpoint_features_blocks(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_race_roster(config, [("VER", "Team VER"), ("PER", "Team PER")])
    features = _features([("VER", "Team VER"), ("PER", "Team PER")])
    features.loc[features["driver"].eq("PER"), "fp3_best_push_lap_time_sec"] = pd.NA

    with pytest.raises(ValueError, match="latest-checkpoint feature values"):
        constrain_features_to_entry_list(
            config,
            season=2026,
            event="Belgian Grand Prix",
            event_order=12,
            feature_rows=features,
            allow_fastf1=False,
        )


def test_participant_count_is_not_hard_coded_for_latest_session(tmp_path: Path) -> None:
    config = _config(tmp_path)
    drivers = tuple(f"D{index:02d}" for index in range(17))
    _write_practice(config, fp1=drivers, fp2=drivers, fp3=drivers)

    audit = audit_qualifying_entry_list(
        config,
        season=2026,
        event="Belgian Grand Prix",
        allow_fastf1=False,
    )

    assert audit.forecast_allowed
    assert audit.summary["entry_list_driver_count"] == 17
    assert audit.summary["latest_session_participant_count"] == 17


def test_alternative_weekend_uses_latest_completed_session_before_q(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _write_session(config, "FP1", ("VER", "NOR"))
    _write_session(config, "SQ", ("VER", "NOR"))
    _write_session(config, "S", ("VER", "NOR"))
    monkeypatch.setattr(
        "f1_prediction.data.qualifying_entry_list.fastf1.get_event_schedule",
        lambda season, include_testing=False: pd.DataFrame(
            [
                {
                    "EventName": "Belgian Grand Prix",
                    "Session1": "Practice 1",
                    "Session1Date": "2026-07-17T12:00:00",
                    "Session2": "Sprint Qualifying",
                    "Session2Date": "2026-07-17T16:00:00",
                    "Session3": "Sprint",
                    "Session3Date": "2026-07-18T10:00:00",
                    "Session4": "Qualifying",
                    "Session4Date": "2026-07-18T14:00:00",
                }
            ]
        ),
    )

    audit = audit_qualifying_entry_list(config, season=2026, event="Belgian Grand Prix")

    assert audit.forecast_allowed
    assert audit.summary["resolution_source"] == f"{LATEST_PRE_Q_SOURCE_PREFIX}:S"
    assert set(audit.drivers["driver"]) == {"VER", "NOR"}
    assert audit.exclusions.empty


def test_empty_latest_session_blocks(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_practice(config, fp1=("VER", "NOR"), fp2=("VER", "NOR"), fp3=())

    audit = audit_qualifying_entry_list(
        config,
        season=2026,
        event="Belgian Grand Prix",
        allow_fastf1=False,
    )

    assert not audit.forecast_allowed
    assert audit.summary["selected_source_session"] == "FP3"
    assert audit.summary["selected_source_session_completion_status"] == "driver_set_empty"


def test_duplicate_latest_session_identity_blocks(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_practice(config, fp1=("VER", "NOR"), fp2=("VER", "NOR"), fp3=("VER", "VER"))
    fp3 = build_lap_output_path(config.lap_output_dir, 2026, "Belgian Grand Prix", "FP3")
    laps = pd.read_parquet(fp3)
    laps.loc[1, "Team"] = "Different Team"
    laps.to_parquet(fp3, index=False)

    audit = audit_qualifying_entry_list(
        config,
        season=2026,
        event="Belgian Grand Prix",
        allow_fastf1=False,
    )

    assert not audit.forecast_allowed
    assert "identity_or_team_mismatch" in set(audit.failures["check_name"])


def test_q_results_source_is_valid_after_q_when_no_pre_q_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)

    class _Session:
        results = pd.DataFrame(
            {
                "Abbreviation": ["VER", "NOR"],
                "TeamName": ["Red Bull", "McLaren"],
            }
        )

        def load(self, **_kwargs) -> None:
            return None

    monkeypatch.setattr(
        "f1_prediction.data.qualifying_entry_list.fastf1.get_event_schedule",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline schedule")),
    )
    monkeypatch.setattr(
        "f1_prediction.data.qualifying_entry_list.fastf1.get_session",
        lambda *_args, **_kwargs: _Session(),
    )

    audit = audit_qualifying_entry_list(config, season=2026, event="Belgian Grand Prix")

    assert audit.forecast_allowed
    assert audit.summary["resolution_source"] == "fastf1_q_results"
    assert audit.summary["q_data_available"] is True


def test_fp1_only_rookie_is_excluded_from_qualifying_entry_list(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_practice(config, fp1=("VER", "NOR", "CRA"), fp2=("VER", "NOR"), fp3=("VER", "NOR"))
    _write_entry_list(config, [("VER", "Red Bull"), ("NOR", "McLaren")])

    audit = audit_qualifying_entry_list(config, season=2026, event="Belgian Grand Prix")

    assert audit.forecast_allowed
    assert audit.summary["entry_list_resolution_status"] == ENTRY_LIST_RESOLVED
    assert audit.exclusions["driver"].tolist() == ["CRA"]
    assert audit.exclusions["exclusion_reason"].tolist() == ["fp1_only_not_qualifying_eligible"]


def test_cra_style_third_driver_is_filtered_before_forecast(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_practice(config, fp1=("ALO", "STR", "CRA"), fp2=("ALO", "STR"), fp3=("ALO", "STR"))
    _write_entry_list(config, [("ALO", "Aston Martin"), ("STR", "Aston Martin")])
    features = _features(
        [("ALO", "Aston Martin"), ("STR", "Aston Martin"), ("CRA", "Aston Martin")]
    )

    constrained, audit = constrain_features_to_entry_list(
        config,
        season=2026,
        event="Belgian Grand Prix",
        event_order=12,
        feature_rows=features,
    )

    assert audit.summary["excluded_practice_only_driver_count"] == 1
    assert set(constrained["driver"]) == {"ALO", "STR"}


def test_replacement_driver_is_included_and_regular_driver_excluded(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_practice(config, fp1=("VER", "REG"), fp2=("VER", "BEA"), fp3=("VER", "BEA"))
    _write_entry_list(config, [("VER", "Red Bull"), ("BEA", "Ferrari")])
    features = _features([("VER", "Red Bull"), ("REG", "Ferrari"), ("BEA", "Ferrari")])

    constrained, _audit = constrain_features_to_entry_list(
        config,
        season=2026,
        event="Belgian Grand Prix",
        event_order=12,
        feature_rows=features,
    )

    assert set(constrained["driver"]) == {"VER", "BEA"}
    assert "REG" not in set(constrained["driver"])


def test_fp2_or_fp3_only_non_entry_driver_is_excluded(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_practice(
        config,
        fp1=("VER", "NOR"),
        fp2=("VER", "NOR", "TST"),
        fp3=("VER", "NOR", "DEV"),
    )
    _write_entry_list(config, [("VER", "Red Bull"), ("NOR", "McLaren")])

    audit = audit_qualifying_entry_list(config, season=2026, event="Belgian Grand Prix")

    assert set(audit.exclusions["driver"]) == {"TST", "DEV"}
    assert set(audit.exclusions["exclusion_reason"]) == {
        "fp2_only_not_qualifying_eligible",
        "fp3_only_not_qualifying_eligible",
    }


def test_driver_and_team_counts_are_not_hard_coded(tmp_path: Path) -> None:
    config = _config(tmp_path)
    entrants = [(f"D{index}", f"Team {index % 7}") for index in range(17)]
    _write_entry_list(config, entrants)
    features = _features(entrants)

    constrained, audit = constrain_features_to_entry_list(
        config,
        season=2026,
        event="Belgian Grand Prix",
        event_order=12,
        feature_rows=features,
    )

    assert len(constrained) == 17
    assert audit.summary["entry_list_driver_count"] == 17
    assert constrained["team"].nunique() == 7


def test_missing_eligible_driver_features_block(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_entry_list(config, [("VER", "Red Bull"), ("NOR", "McLaren")])

    with pytest.raises(ValueError, match="Missing feature rows"):
        constrain_features_to_entry_list(
            config,
            season=2026,
            event="Belgian Grand Prix",
            event_order=12,
            feature_rows=_features([("VER", "Red Bull")]),
        )


def test_extra_predicted_driver_blocks_forecast_parity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_entry_list(config, [("VER", "Red Bull"), ("NOR", "McLaren")])
    forecasts = _features([("VER", "Red Bull"), ("NOR", "McLaren"), ("CRA", "Aston Martin")])
    forecasts["diagnostic_only"] = False

    audit = audit_qualifying_entry_list(
        config,
        season=2026,
        event="Belgian Grand Prix",
        forecast_rows=forecasts,
    )

    assert not audit.forecast_allowed
    assert "extra_forecast_driver" in set(audit.failures["check_name"])


def test_duplicate_entry_and_team_mismatch_block(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_entry_list(config, [("VER", "Red Bull"), ("VER", "Red Bull")])
    duplicate = audit_qualifying_entry_list(config, season=2026, event="Belgian Grand Prix")
    assert not duplicate.forecast_allowed
    assert "duplicate_eligible_driver" in set(duplicate.failures["check_name"])

    _write_entry_list(config, [("VER", "Ferrari")])
    mismatch = audit_qualifying_entry_list(
        config,
        season=2026,
        event="Belgian Grand Prix",
        feature_rows=_features([("VER", "Red Bull")]),
    )
    assert not mismatch.forecast_allowed
    assert "driver_team_mapping_mismatch" in set(mismatch.failures["check_name"])


def test_unresolved_entry_list_blocks_before_forecast(tmp_path: Path) -> None:
    config = _config(tmp_path)

    audit = audit_qualifying_entry_list(
        config,
        season=2026,
        event="Belgian Grand Prix",
        allow_fastf1=False,
    )

    assert not audit.forecast_allowed
    assert audit.summary["entry_list_resolution_status"] == "unresolved"
    assert "entry_list_unresolved" in set(audit.failures["check_name"])


def test_dashboard_blocks_old_forecast_with_entry_list_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_protocol(config)
    _write_registry(config, [_registry_event("Belgian Grand Prix", 12)])
    _write_entry_list(config, [("VER", "Red Bull"), ("NOR", "McLaren")])
    _write_forecasts(
        config,
        "Belgian Grand Prix",
        [("VER", "Red Bull", 0.1), ("NOR", "McLaren", 0.2), ("CRA", "Aston Martin", 0.4)],
    )
    old_forecasts = pd.read_parquet(
        config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    )
    audit_qualifying_entry_list(
        config,
        season=2026,
        event="Belgian Grand Prix",
        event_order=12,
        forecast_rows=old_forecasts,
        allow_fastf1=False,
    )

    export_dashboard_artifacts(config)
    current = _read_dashboard(config, "current_event.json")
    forecast = _read_dashboard(config, "event_forecast.json")

    assert current["data"]["lifecycle"]["state"] == "blocked"
    assert current["data"]["lifecycle"]["reason"] == "qualifying_entry_list_mismatch"
    assert forecast["data"]["leaderboard"]["available"] is False


def test_valid_forecast_entry_list_parity_passes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_entry_list(config, [("VER", "Red Bull"), ("NOR", "McLaren")])
    forecasts = _features([("VER", "Red Bull"), ("NOR", "McLaren")])
    forecasts["diagnostic_only"] = False

    audit = audit_qualifying_entry_list(
        config,
        season=2026,
        event="Belgian Grand Prix",
        forecast_rows=forecasts,
        allow_fastf1=False,
    )

    assert audit.forecast_allowed
    assert audit.summary["driver_set_parity_status"] == ENTRY_LIST_PARITY_PASSED


def _config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        project_root=tmp_path,
        fastf1_cache_dir=tmp_path / "cache",
        lap_output_dir=tmp_path / "data/raw/laps",
        session_metadata_output_dir=tmp_path / "data/raw/session_metadata",
        clean_lap_output_dir=tmp_path / "data/interim/clean_laps",
        session_features_output_dir=tmp_path / "data/processed/session_features",
        modeling_output_dir=tmp_path / "data/processed/modeling",
        metrics_output_dir=tmp_path / "reports/metrics",
    )


def _write_entry_list(config: DataConfig, entrants: list[tuple[str, str]]) -> None:
    path = (
        config.project_root
        / "data/processed/monitoring/2026/belgian-grand-prix/qualifying_entry_list.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "driver": [driver for driver, _team in entrants],
            "team": [team for _driver, team in entrants],
        }
    ).to_csv(path, index=False)


def _write_race_roster(config: DataConfig, entrants: list[tuple[str, str]]) -> None:
    path = (
        config.project_root
        / "data/processed/monitoring/2026/belgian-grand-prix/race_driver_roster.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "driver": [driver for driver, _team in entrants],
            "team": [team for _driver, team in entrants],
        }
    ).to_csv(path, index=False)


def _write_practice(
    config: DataConfig,
    *,
    fp1: tuple[str, ...],
    fp2: tuple[str, ...],
    fp3: tuple[str, ...],
) -> None:
    for session, drivers in {"FP1": fp1, "FP2": fp2, "FP3": fp3}.items():
        _write_session(config, session, drivers)


def _write_session(
    config: DataConfig,
    session: str,
    drivers: tuple[str, ...],
) -> None:
    path = build_lap_output_path(config.lap_output_dir, 2026, "Belgian Grand Prix", session)
    pd.DataFrame(
        {
            "Driver": list(drivers),
            "Team": [f"Team {driver}" for driver in drivers],
            "LapNumber": [1.0] * len(drivers),
            "LapTime": pd.to_timedelta([80.0] * len(drivers), unit="s"),
        }
    ).to_parquet(path, index=False)
    metadata_path = (
        config.session_metadata_output_dir
        / "2026"
        / "belgian-grand-prix"
        / f"{session.lower()}_metadata.json"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "season": 2026,
                "event_input": "Belgian Grand Prix",
                "event_name": "Belgian Grand Prix",
                "event_slug": "belgian-grand-prix",
                "session_input": session,
                "session_name": session,
                "session_slug": session.lower(),
                "status": "success",
                "n_laps": len(drivers),
                "n_drivers": len(set(drivers)),
                "drivers": list(dict.fromkeys(drivers)),
            }
        ),
        encoding="utf-8",
    )


def _features(drivers: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026] * len(drivers),
            "event": ["Belgian Grand Prix"] * len(drivers),
            "event_slug": ["belgian-grand-prix"] * len(drivers),
            "checkpoint": ["after_fp3"] * len(drivers),
            "driver": [driver for driver, _team in drivers],
            "driver_key": [driver.lower() for driver, _team in drivers],
            "team": [team for _driver, team in drivers],
            "team_key": [team.lower().replace(" ", "-") for _driver, team in drivers],
            "fp3_best_push_lap_time_sec": [80.0 + index for index in range(len(drivers))],
        }
    )
