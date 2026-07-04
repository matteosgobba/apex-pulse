import json
from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.config import DataConfig, FeatureConfig, PushLapConfig, load_model_config
from f1_prediction.data.fastf1_loader import build_lap_output_path
from f1_prediction.data.monitoring_onboarding import (
    FORBIDDEN_TARGET_COLUMNS,
    add_monitoring_targets,
    create_monitoring_data_readiness_report,
    feature_artifact_path,
    prepare_monitoring_event,
    register_monitoring_event,
    target_artifact_path,
)
from f1_prediction.modeling.prospective_monitoring import (
    create_prospective_monitoring_forecast,
    create_prospective_monitoring_protocol,
    create_prospective_monitoring_settlement,
)


def test_prepare_event_writes_fp3_safe_features_without_targets(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_practice_raw(config, 2026, "Bahrain")

    summary = prepare_monitoring_event(config, _features(), season=2026, event="Bahrain")
    features = pd.read_parquet(feature_artifact_path(config, 2026, "Bahrain"))
    manifest = json.loads(summary.summary_path.read_text())

    assert summary.status == "prepared"
    assert features["checkpoint"].eq("after_fp3").all()
    assert not set(FORBIDDEN_TARGET_COLUMNS).intersection(features.columns)
    assert not any(column.startswith("quali_") for column in features.columns)
    assert manifest["forbidden_target_column_count"] == 0
    assert manifest["driver_row_count"] == 4


def test_prepare_event_fails_when_local_practice_raw_is_missing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_practice_raw(config, 2026, "Bahrain", sessions=("FP1", "FP2"))

    with pytest.raises(FileNotFoundError, match="FP3"):
        prepare_monitoring_event(config, _features(), season=2026, event="Bahrain")

    manifest = json.loads(
        (
            tmp_path / "data/processed/monitoring/2026/bahrain/monitoring_event_manifest.json"
        ).read_text()
    )
    assert manifest["preparation_status"] == "failed"
    assert "missing_fp3_raw_laps" in manifest["readiness_blockers"]


def test_add_targets_writes_separate_artifact_without_mutating_features(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_practice_raw(config, 2026, "Bahrain")
    _write_q_raw(config, 2026, "Bahrain")
    prepare_monitoring_event(config, _features(), season=2026, event="Bahrain")
    feature_path = feature_artifact_path(config, 2026, "Bahrain")
    before = feature_path.read_bytes()

    summary = add_monitoring_targets(config, season=2026, event="Bahrain")
    targets = pd.read_parquet(target_artifact_path(config, 2026, "Bahrain"))

    assert summary.status == "targets_added"
    assert feature_path.read_bytes() == before
    assert target_artifact_path(config, 2026, "Bahrain") != feature_path
    assert set(FORBIDDEN_TARGET_COLUMNS).issubset(targets.columns)
    assert "target_created_at_utc" in targets.columns


def test_add_targets_requires_valid_prequalification_features(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_q_raw(config, 2026, "Bahrain")

    with pytest.raises(FileNotFoundError, match="manifest"):
        add_monitoring_targets(config, season=2026, event="Bahrain")


def test_register_validates_protocol_season_and_blocks_leaking_features(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_historical_dataset(config)
    create_prospective_monitoring_protocol(
        config,
        load_model_config(),
        protocol_name="season_2026_v1",
        monitor_season=2026,
        train_seasons=(2023,),
        dataset_path=dataset_path,
    )
    _write_practice_raw(config, 2026, "Bahrain")
    prepare_monitoring_event(config, _features(), season=2026, event="Bahrain")
    features_path = feature_artifact_path(config, 2026, "Bahrain")
    features = pd.read_parquet(features_path)
    features["quali_gap_to_pole_sec"] = 0.0
    features.to_parquet(features_path, index=False)

    with pytest.raises(ValueError, match="forbidden_target_columns_present"):
        register_monitoring_event(
            config,
            protocol_name="season_2026_v1",
            season=2026,
            event="Bahrain",
            event_order=2,
        )


def test_register_preserves_supplied_chronological_order_and_lifecycle(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_historical_dataset(config)
    create_prospective_monitoring_protocol(
        config,
        load_model_config(),
        protocol_name="season_2026_v1",
        monitor_season=2026,
        train_seasons=(2023,),
        dataset_path=dataset_path,
    )
    _write_practice_raw(config, 2026, "Bahrain")
    prepare_monitoring_event(config, _features(), season=2026, event="Bahrain")

    register_monitoring_event(
        config,
        protocol_name="season_2026_v1",
        season=2026,
        event="Bahrain",
        event_order=7,
    )
    registry = pd.read_csv(config.metrics_output_dir / "prospective_monitoring_event_registry.csv")

    assert registry.loc[0, "event_order"] == 7
    assert registry.loc[0, "onboarding_status"] == "registered_not_forecasted"
    assert bool(registry.loc[0, "forecastable"])


def test_forecast_and_settlement_use_separate_feature_and_target_artifacts(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    dataset_path = _write_historical_dataset(config)
    create_prospective_monitoring_protocol(
        config,
        load_model_config(),
        protocol_name="season_2026_v1",
        monitor_season=2026,
        train_seasons=(2023,),
        dataset_path=dataset_path,
    )
    _write_practice_raw(config, 2026, "Bahrain")
    _write_q_raw(config, 2026, "Bahrain")
    prepare_monitoring_event(config, _features(), season=2026, event="Bahrain")
    register_monitoring_event(
        config,
        protocol_name="season_2026_v1",
        season=2026,
        event="Bahrain",
        event_order=2,
    )

    create_prospective_monitoring_forecast(
        config,
        load_model_config(),
        _features(),
        protocol_name="season_2026_v1",
        event="Bahrain",
    )
    forecasts = pd.read_parquet(
        config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    )
    assert forecasts["current_event_target_accessed"].eq(False).all()
    assert forecasts["actual_gap_sec"].isna().all()
    with pytest.raises(ValueError, match="target artifact"):
        create_prospective_monitoring_settlement(
            config,
            protocol_name="season_2026_v1",
            event="Bahrain",
        )

    add_monitoring_targets(config, season=2026, event="Bahrain")
    create_prospective_monitoring_settlement(
        config,
        protocol_name="season_2026_v1",
        event="Bahrain",
    )
    settlements = pd.read_parquet(
        config.metrics_output_dir / "prospective_monitoring_settlements.parquet"
    )
    assert settlements["forecast_preexisted_settlement"].astype(bool).all()
    assert settlements["actual_gap_sec"].notna().all()


def test_readiness_report_handles_unavailable_monitored_season(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_historical_dataset(config)
    create_prospective_monitoring_protocol(
        config,
        load_model_config(),
        protocol_name="season_2026_v1",
        monitor_season=2026,
        train_seasons=(2023,),
        dataset_path=dataset_path,
    )

    summary = create_monitoring_data_readiness_report(config)
    payload = json.loads(summary.summary_path.read_text())

    assert payload["status"] == "not_ready"
    assert payload["forecastable_event_count"] == 0
    assert len(summary.figure_paths) == 4


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


def _features() -> FeatureConfig:
    return FeatureConfig(push_lap=PushLapConfig(1.03, 1.07, ("SOFT", "MEDIUM", "HARD")))


def _write_practice_raw(
    config: DataConfig,
    season: int,
    event: str,
    *,
    sessions: tuple[str, ...] = ("FP1", "FP2", "FP3"),
) -> None:
    for index, session in enumerate(sessions, start=1):
        path = build_lap_output_path(config.lap_output_dir, season, event, session)
        _raw_laps(base_time=80.0 + index).to_parquet(path, index=False)


def _write_q_raw(config: DataConfig, season: int, event: str) -> None:
    path = build_lap_output_path(config.lap_output_dir, season, event, "Q")
    _raw_laps(base_time=79.0).to_parquet(path, index=False)


def _raw_laps(*, base_time: float) -> pd.DataFrame:
    drivers = ("VER", "NOR", "LEC", "HAM")
    teams = ("Red Bull Racing", "McLaren", "Ferrari", "Mercedes")
    return pd.DataFrame(
        {
            "Driver": list(drivers),
            "Team": list(teams),
            "LapNumber": [1.0] * 4,
            "Stint": [1.0] * 4,
            "Compound": ["SOFT"] * 4,
            "TyreLife": [2.0] * 4,
            "LapTime": pd.to_timedelta([base_time + i * 0.2 for i in range(4)], unit="s"),
            "Sector1Time": pd.to_timedelta([25.0] * 4, unit="s"),
            "Sector2Time": pd.to_timedelta([29.0] * 4, unit="s"),
            "Sector3Time": pd.to_timedelta([25.0] * 4, unit="s"),
            "IsAccurate": [True] * 4,
            "Deleted": [False] * 4,
            "PitOutTime": [pd.NaT] * 4,
            "PitInTime": [pd.NaT] * 4,
        }
    )


def _write_historical_dataset(config: DataConfig) -> Path:
    rows = []
    for order, slug in enumerate(("australia", "canada")):
        for driver_index, driver in enumerate(("VER", "NOR", "LEC", "HAM")):
            gap = driver_index * 0.2
            rows.append(
                {
                    "season": 2023,
                    "event": slug.title(),
                    "event_slug": slug,
                    "event_order": order,
                    "checkpoint": "after_fp3",
                    "driver": driver,
                    "driver_key": driver.lower(),
                    "team": f"Team {driver_index}",
                    "team_key": f"team_{driver_index}",
                    "quali_gap_to_pole_sec": gap,
                    "quali_position": driver_index + 1,
                    "quali_best_lap_time_sec": 79.0 + gap,
                    "reached_q2": 1,
                    "reached_q3": 1,
                    "fp3_best_push_lap_time_sec": 80.0 + gap,
                    "fp3_best_valid_lap_time_sec": 80.1 + gap,
                    "fp3_theoretical_best_lap_time_sec": 79.9 + gap,
                    "fp3_best_push_gap_to_session_best_sec": gap,
                    "fp3_best_valid_gap_to_session_best_sec": gap + 0.1,
                    "fp3_theoretical_best_gap_to_session_best_sec": max(gap - 0.1, 0),
                    "practice_signal_quality_score": 6,
                }
            )
    dataset = pd.DataFrame(rows)
    path = config.modeling_output_dir / "combined" / "modeling_dataset.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(path, index=False)
    return path
