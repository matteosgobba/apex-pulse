import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.config import DataConfig, load_feature_config, load_model_config
from f1_prediction.modeling.backtest_report import build_backtest_report_payload
from f1_prediction.modeling.prospective_monitoring import (
    build_event_metrics,
    build_integrity_by_event,
    create_prospective_monitoring_forecast,
    create_prospective_monitoring_protocol,
    create_prospective_monitoring_report,
    create_prospective_monitoring_settlement,
)


def test_protocol_creation_stores_canonical_contract_and_fingerprint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)

    summary = create_prospective_monitoring_protocol(
        config,
        load_model_config(),
        protocol_name="season_2026_v1",
        monitor_season=2026,
        train_seasons=(2023,),
        dataset_path=dataset_path,
    )
    protocol = json.loads(
        (config.metrics_output_dir / "prospective_monitoring_protocol.json").read_text()
    )

    assert summary.status == "created"
    assert protocol["candidate_identity"]["temporal_weighting_policy"] == (
        "current_season_only_with_prior"
    )
    assert protocol["default_identity"]["temporal_weighting_policy"] == "uniform"
    assert protocol["frozen_gate_configuration"]["min_prior_candidate_predictions"] == 100
    assert protocol["protocol_fingerprint"]


def test_identical_protocol_rerun_validates_successfully(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    kwargs = {
        "protocol_name": "season_2026_v1",
        "monitor_season": 2026,
        "train_seasons": (2023,),
        "dataset_path": dataset_path,
    }

    create_prospective_monitoring_protocol(config, load_model_config(), **kwargs)
    summary = create_prospective_monitoring_protocol(config, load_model_config(), **kwargs)

    assert summary.status == "validated"


def test_protocol_mismatch_records_validation_artifact(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    model_config = load_model_config()
    create_prospective_monitoring_protocol(
        config,
        model_config,
        protocol_name="season_2026_v1",
        monitor_season=2026,
        train_seasons=(2023,),
        dataset_path=dataset_path,
    )
    changed_gate = replace(
        model_config.champion_policy.season_aware_nested_guarded,
        min_prior_candidate_predictions=50,
    )
    changed_policy = replace(
        model_config.champion_policy,
        season_aware_nested_guarded=changed_gate,
    )
    changed_model = replace(model_config, champion_policy=changed_policy)

    with pytest.raises(ValueError, match="protocol mismatch"):
        create_prospective_monitoring_protocol(
            config,
            changed_model,
            protocol_name="season_2026_v1",
            monitor_season=2026,
            train_seasons=(2023,),
            dataset_path=dataset_path,
        )
    validation = pd.read_csv(
        config.metrics_output_dir / "prospective_monitoring_protocol_validation.csv"
    )
    assert "frozen_gate_configuration" in set(validation["field"])


def test_event_registry_preserves_chronological_event_order(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)

    create_prospective_monitoring_protocol(
        config,
        load_model_config(),
        protocol_name="season_2026_v1",
        monitor_season=2026,
        train_seasons=(2023,),
        dataset_path=dataset_path,
    )
    registry = pd.read_csv(config.metrics_output_dir / "prospective_monitoring_event_registry.csv")

    assert list(registry["event_slug"]) == ["bahrain", "monza"]
    assert list(registry["event_order"]) == [2, 3]


def test_forecast_excludes_current_targets_and_future_events(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)

    create_prospective_monitoring_forecast(
        config,
        load_model_config(),
        load_feature_config(),
        protocol_name="season_2026_v1",
        event="Bahrain",
    )
    forecasts = pd.read_parquet(
        config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    )
    manifest = pd.read_csv(
        config.metrics_output_dir / "prospective_monitoring_training_manifest.csv"
    )

    assert forecasts["current_event_target_accessed"].eq(False).all()
    assert forecasts["actual_gap_sec"].isna().all()
    assert manifest["current_event_excluded_from_training"].astype(bool).all()
    assert manifest["future_same_season_events_excluded"].astype(bool).all()
    assert manifest["future_seasons_excluded"].astype(bool).all()
    assert not manifest["training_event_keys_used"].astype(str).str.contains("2026/monza").any()


def test_weighted_shadow_rows_are_diagnostic_only_and_not_live(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)

    create_prospective_monitoring_forecast(
        config,
        load_model_config(),
        load_feature_config(),
        protocol_name="season_2026_v1",
        event="Bahrain",
    )
    shadow = pd.read_parquet(
        config.metrics_output_dir / "prospective_monitoring_shadow_candidates.parquet"
    )
    weighted = shadow[shadow["prediction_role"].eq("season_aware_weighted_candidate_shadow")]

    assert not weighted.empty
    assert weighted["diagnostic_only"].astype(bool).all()
    assert not weighted["live_policy_selected"].astype(bool).any()
    assert not weighted["selection_is_live"].astype(bool).any()


def test_settlement_requires_preexisting_forecast(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)

    with pytest.raises(ValueError, match="No pre-existing forecast"):
        create_prospective_monitoring_settlement(
            config,
            protocol_name="season_2026_v1",
            event="Bahrain",
        )


def test_settlement_scores_exact_keys_and_keeps_live_shadow_metrics_separate(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)
    create_prospective_monitoring_forecast(
        config,
        load_model_config(),
        load_feature_config(),
        protocol_name="season_2026_v1",
        event="Bahrain",
    )

    create_prospective_monitoring_settlement(
        config,
        protocol_name="season_2026_v1",
        event="Bahrain",
    )
    settlements = pd.read_parquet(
        config.metrics_output_dir / "prospective_monitoring_settlements.parquet"
    )
    metrics = pd.read_csv(config.metrics_output_dir / "prospective_monitoring_event_metrics.csv")

    assert settlements["forecast_preexisted_settlement"].astype(bool).all()
    assert set(metrics["prediction_role"]) == {
        "observed_live_policy",
        "uniform_default_shadow",
        "season_aware_weighted_candidate_shadow",
    }
    live = build_event_metrics(settlements)
    assert not live[live["diagnostic_only"].eq(False)].empty


def test_forecast_snapshot_mutation_blocks_settlement(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)
    create_prospective_monitoring_forecast(
        config,
        load_model_config(),
        load_feature_config(),
        protocol_name="season_2026_v1",
        event="Bahrain",
    )
    forecasts_path = config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    forecasts = pd.read_parquet(forecasts_path)
    forecasts.loc[0, "prediction_gap_sec"] = 99.0
    forecasts.to_parquet(forecasts_path, index=False)

    with pytest.raises(ValueError, match="mutation"):
        create_prospective_monitoring_settlement(
            config,
            protocol_name="season_2026_v1",
            event="Bahrain",
        )


def test_only_settled_earlier_event_evidence_reaches_later_forecast(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)
    model_config = load_model_config()
    feature_config = load_feature_config()
    create_prospective_monitoring_forecast(
        config,
        model_config,
        feature_config,
        protocol_name="season_2026_v1",
        event="Bahrain",
    )
    create_prospective_monitoring_settlement(
        config,
        protocol_name="season_2026_v1",
        event="Bahrain",
    )
    create_prospective_monitoring_forecast(
        config,
        model_config,
        feature_config,
        protocol_name="season_2026_v1",
        event="Monza",
    )
    manifest = pd.read_csv(
        config.metrics_output_dir / "prospective_monitoring_training_manifest.csv"
    )
    monza = manifest[manifest["event_key"].astype(str).eq("2026/monza")]

    assert monza["training_event_keys_used"].astype(str).str.contains("2026/bahrain").any()
    assert not monza["training_event_keys_used"].astype(str).str.contains("2026/monza").any()


def test_report_handles_no_monitored_season_data_and_generates_figures(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config, include_monitor_season=False)
    create_prospective_monitoring_protocol(
        config,
        load_model_config(),
        protocol_name="season_2026_v1",
        monitor_season=2026,
        train_seasons=(2023,),
        dataset_path=dataset_path,
    )

    summary = create_prospective_monitoring_report(config)
    payload = json.loads(summary.summary_path.read_text())

    assert payload["status"] == "not_ready"
    assert payload["fresh_evidence_status"] == "not_collected"
    assert len(summary.figure_paths) == 5


def test_future_settlement_row_triggers_integrity_failure(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)
    create_prospective_monitoring_forecast(
        config,
        load_model_config(),
        load_feature_config(),
        protocol_name="season_2026_v1",
        event="Bahrain",
    )
    registry = pd.read_csv(config.metrics_output_dir / "prospective_monitoring_event_registry.csv")
    forecasts = pd.read_parquet(
        config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    )
    settlements = pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "event_slug": "monza",
                "event_order": 3,
                "settled_at_utc": "2026-01-01T00:00:00+00:00",
            }
        ]
    )
    by_event = build_integrity_by_event(
        json.loads(
            (config.metrics_output_dir / "prospective_monitoring_protocol.json").read_text()
        ),
        registry,
        forecasts,
        settlements,
        pd.read_csv(
            config.metrics_output_dir / "prospective_monitoring_forecast_integrity_audit.csv"
        ),
    )

    bahrain = by_event[by_event["event_slug"].eq("bahrain")].iloc[0]
    assert not bool(bahrain["future_settlement_not_used"])


def test_backtest_report_exposes_monitoring_fields() -> None:
    payload = build_backtest_report_payload(
        quality={"n_rows": 1, "n_events": 1, "checkpoints": ["after_fp3"]},
        baseline_metrics={},
        tabular_metrics=None,
        prospective_monitoring_summary={
            "status": "not_ready",
            "protocol_name": "season_2026_v1",
            "monitor_season": 2026,
            "integrity_status": "missing",
            "fresh_evidence_status": "not_collected",
            "policy_recommendation": "season_aware_candidate_requires_more_evidence",
        },
    )

    assert payload["prospective_monitoring_available"] is True
    assert payload["prospective_monitoring_protocol_name"] == "season_2026_v1"
    assert payload["prospective_monitoring_fresh_evidence_status"] == "not_collected"


def _init(config: DataConfig, dataset_path: Path) -> None:
    create_prospective_monitoring_protocol(
        config,
        load_model_config(),
        protocol_name="season_2026_v1",
        monitor_season=2026,
        train_seasons=(2023,),
        dataset_path=dataset_path,
    )


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


def _write_dataset(config: DataConfig, *, include_monitor_season: bool = True) -> Path:
    rows = []
    events = [
        (2023, "australia", "Australia", 0),
        (2023, "canada", "Canada", 1),
    ]
    if include_monitor_season:
        events.extend(
            [
                (2026, "bahrain", "Bahrain", 2),
                (2026, "monza", "Monza", 3),
            ]
        )
    for season, slug, event, order in events:
        for driver_index, driver in enumerate(("VER", "NOR", "LEC", "HAM")):
            gap = float(driver_index) * 0.2 + (0.05 if slug == "monza" else 0.0)
            rows.append(
                {
                    "season": season,
                    "event": event,
                    "event_slug": slug,
                    "event_order": order,
                    "checkpoint": "after_fp3",
                    "driver": driver,
                    "driver_key": driver,
                    "team": f"Team {driver_index}",
                    "team_key": f"T{driver_index}",
                    "quali_gap_to_pole_sec": gap,
                    "quali_position": driver_index + 1,
                    "quali_best_lap_time_sec": 80.0 + gap,
                    "reached_q2": 1,
                    "reached_q3": 1,
                    "fp3_best_push_lap_time_sec": 80.0 + gap,
                    "fp3_best_valid_lap_time_sec": 80.1 + gap,
                    "fp3_theoretical_best_lap_time_sec": 79.9 + gap,
                    "fp3_best_push_gap_to_session_best_sec": gap,
                    "fp3_best_valid_gap_to_session_best_sec": gap + 0.1,
                    "fp3_theoretical_best_gap_to_session_best_sec": max(gap - 0.1, 0),
                    "practice_base_feature": 1.0 + gap,
                    "team_gap_to_session_best": gap / 2,
                }
            )
    dataset = pd.DataFrame(rows)
    path = config.modeling_output_dir / "combined" / "modeling_dataset.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(path, index=False)
    return path
