import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.config import DataConfig, load_feature_config, load_model_config
from f1_prediction.modeling.backtest_report import build_backtest_report_payload
from f1_prediction.modeling.prospective_monitoring import (
    build_event_metrics,
    build_event_order_reconciliation,
    build_integrity_by_event,
    create_prospective_monitoring_forecast,
    create_prospective_monitoring_preflight,
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


def test_preflight_ready_for_valid_prepared_registered_event(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)

    summary = create_prospective_monitoring_preflight(
        config,
        protocol_name="season_2026_v1",
        season=2026,
        event="Bahrain",
    )
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    checks = pd.read_csv(config.metrics_output_dir / "prospective_monitoring_preflight_checks.csv")
    runbook = (config.metrics_output_dir / "prospective_monitoring_preflight_runbook.md").read_text(
        encoding="utf-8"
    )

    assert summary.status == "ready_to_forecast"
    assert payload["forecast_allowed"] is True
    assert payload["blocking_check_count"] == 0
    assert checks["check_name"].str.contains("feature_artifact_exists").any()
    assert "prospective-monitoring-forecast --protocol-name season_2026_v1" in runbook
    assert "A `ready_to_forecast` result is required" in runbook


def test_preflight_missing_protocol_returns_invalid_protocol(tmp_path: Path) -> None:
    config = _config(tmp_path)

    summary = create_prospective_monitoring_preflight(
        config,
        protocol_name="season_2026_v1",
        season=2026,
        event="Bahrain",
    )
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))

    assert summary.status == "invalid_protocol"
    assert payload["forecast_allowed"] is False


def test_preflight_missing_registry_row_blocks_forecast(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)

    summary = create_prospective_monitoring_preflight(
        config,
        protocol_name="season_2026_v1",
        season=2026,
        event="Singapore",
    )
    failures = pd.read_csv(
        config.metrics_output_dir / "prospective_monitoring_preflight_failures.csv"
    )

    assert summary.status == "invalid_registry_lineage"
    assert "event_exists_in_registry" in set(failures["check_name"])


def test_preflight_duplicate_event_order_blocks_forecast(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)
    registry_path = config.metrics_output_dir / "prospective_monitoring_event_registry.csv"
    registry = pd.read_csv(registry_path)
    registry.loc[registry["event_slug"].eq("monza"), "event_order"] = 2
    registry.to_csv(registry_path, index=False)

    summary = create_prospective_monitoring_preflight(
        config,
        protocol_name="season_2026_v1",
        season=2026,
        event="Bahrain",
    )
    failures = pd.read_csv(
        config.metrics_output_dir / "prospective_monitoring_preflight_failures.csv"
    )

    assert summary.status == "invalid_registry_lineage"
    assert "event_order_unique_within_protocol_and_season" in set(failures["check_name"])


@pytest.mark.parametrize("bad_value", [pd.NA, "bad"])
def test_preflight_missing_or_malformed_event_order_blocks_forecast(
    tmp_path: Path,
    bad_value: object,
) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)
    registry_path = config.metrics_output_dir / "prospective_monitoring_event_registry.csv"
    registry = pd.read_csv(registry_path).astype({"event_order": "object"})
    registry.loc[registry["event_slug"].eq("bahrain"), "event_order"] = bad_value
    registry.to_csv(registry_path, index=False)

    summary = create_prospective_monitoring_preflight(
        config,
        protocol_name="season_2026_v1",
        season=2026,
        event="Bahrain",
    )

    assert summary.status == "invalid_registry_lineage"


def test_preflight_blocks_feature_artifact_with_quali_columns(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)
    registry = pd.read_csv(config.metrics_output_dir / "prospective_monitoring_event_registry.csv")
    path = (
        config.project_root
        / registry.loc[
            registry["event_slug"].eq("bahrain"),
            "feature_artifact_path",
        ].iloc[0]
    )
    frame = pd.read_parquet(path)
    frame["quali_gap_to_pole_sec"] = 0.0
    frame.to_parquet(path, index=False)
    registry.loc[registry["event_slug"].eq("bahrain"), "feature_artifact_fingerprint"] = (
        _fingerprint(path)
    )
    registry.to_csv(
        config.metrics_output_dir / "prospective_monitoring_event_registry.csv", index=False
    )

    summary = create_prospective_monitoring_preflight(
        config,
        protocol_name="season_2026_v1",
        season=2026,
        event="Bahrain",
    )
    failures = pd.read_csv(
        config.metrics_output_dir / "prospective_monitoring_preflight_failures.csv"
    )

    assert summary.status == "blocked"
    assert "forbidden_target_columns_absent" in set(failures["check_name"])


def test_preflight_blocks_preexisting_target_artifact(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)
    _write_target_artifact(config, dataset_path, "Bahrain")

    summary = create_prospective_monitoring_preflight(
        config,
        protocol_name="season_2026_v1",
        season=2026,
        event="Bahrain",
    )
    failures = pd.read_csv(
        config.metrics_output_dir / "prospective_monitoring_preflight_failures.csv"
    )

    assert summary.status == "blocked"
    assert "no_existing_target_artifact_before_forecast" in set(failures["check_name"])


def test_preflight_existing_forecast_returns_already_forecasted(tmp_path: Path) -> None:
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

    summary = create_prospective_monitoring_preflight(
        config,
        protocol_name="season_2026_v1",
        season=2026,
        event="Bahrain",
    )

    assert summary.status == "already_forecasted"


def test_preflight_existing_settlement_blocks_when_no_forecast_snapshot(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)
    pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "event_slug": "bahrain",
                "forecast_id": "missing",
            }
        ]
    ).to_parquet(config.metrics_output_dir / "prospective_monitoring_settlements.parquet")

    summary = create_prospective_monitoring_preflight(
        config,
        protocol_name="season_2026_v1",
        season=2026,
        event="Bahrain",
    )

    assert summary.status == "blocked"


def test_preflight_legacy_current_event_blocks_forecast(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)
    pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "monitor_season": 2026,
                "event_slug": "bahrain",
                "forecast_id": "old",
                "artifact_event_order": 44,
                "registry_event_order": 2,
                "event_order_match": False,
                "event_order_lineage_status": "legacy_noncanonical_event_order",
                "eligible_for_future_prior_evidence_after_reconciliation": False,
            }
        ]
    ).to_csv(
        config.metrics_output_dir / "prospective_monitoring_event_order_reconciliation.csv",
        index=False,
    )

    summary = create_prospective_monitoring_preflight(
        config,
        protocol_name="season_2026_v1",
        season=2026,
        event="Bahrain",
    )

    assert summary.status == "invalid_registry_lineage"


def test_preflight_other_legacy_rows_do_not_block_clean_future_event(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)
    pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "monitor_season": 2026,
                "event_slug": "bahrain",
                "forecast_id": "old",
                "artifact_event_order": 44,
                "registry_event_order": 2,
                "event_order_match": False,
                "event_order_lineage_status": "legacy_noncanonical_event_order",
                "eligible_for_future_prior_evidence_after_reconciliation": False,
            }
        ]
    ).to_csv(
        config.metrics_output_dir / "prospective_monitoring_event_order_reconciliation.csv",
        index=False,
    )

    summary = create_prospective_monitoring_preflight(
        config,
        protocol_name="season_2026_v1",
        season=2026,
        event="Monza",
    )
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))

    assert summary.status == "ready_to_forecast"
    assert payload["legacy_exclusion_status"] == "valid"


def test_forecast_uses_registry_event_order_metadata_not_training_count(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)
    registry_path = config.metrics_output_dir / "prospective_monitoring_event_registry.csv"
    registry = pd.read_csv(registry_path)
    registry.loc[registry["event_slug"].eq("monza"), "event_order"] = 9
    registry.to_csv(registry_path, index=False)

    create_prospective_monitoring_forecast(
        config,
        load_model_config(),
        load_feature_config(),
        protocol_name="season_2026_v1",
        event="Monza",
    )
    forecasts = pd.read_parquet(
        config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    )
    monza = forecasts[forecasts["event_slug"].eq("monza")]

    assert monza["event_order"].eq(9).all()
    assert monza["event_order_source"].eq("registry").all()
    assert monza["event_order_registry_valid"].astype(bool).all()
    assert monza["event_order_lineage_status"].eq("valid_registry_lineage").all()
    assert not monza["event_order"].eq(monza["training_event_count"]).any()
    assert monza["preflight_status"].eq("ready_to_forecast").all()
    assert monza["preflight_run_id"].notna().all()
    assert (
        monza["preflight_summary_path"]
        .eq("reports/metrics/prospective_monitoring_preflight_summary.json")
        .all()
    )


@pytest.mark.parametrize("bad_value", [pd.NA, 0, "bad"])
def test_forecast_fails_when_registry_event_order_is_invalid(
    tmp_path: Path,
    bad_value: object,
) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)
    registry_path = config.metrics_output_dir / "prospective_monitoring_event_registry.csv"
    registry = pd.read_csv(registry_path)
    registry.loc[registry["event_slug"].eq("bahrain"), "event_order"] = bad_value
    registry.to_csv(registry_path, index=False)

    with pytest.raises(ValueError, match="preflight is not ready"):
        create_prospective_monitoring_forecast(
            config,
            load_model_config(),
            load_feature_config(),
            protocol_name="season_2026_v1",
            event="Bahrain",
        )


def test_forecast_fails_with_duplicate_registry_event_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)
    registry_path = config.metrics_output_dir / "prospective_monitoring_event_registry.csv"
    registry = pd.read_csv(registry_path)
    duplicate = registry[registry["event_slug"].eq("bahrain")]
    pd.concat([registry, duplicate], ignore_index=True).to_csv(registry_path, index=False)

    with pytest.raises(ValueError, match="preflight is not ready"):
        create_prospective_monitoring_forecast(
            config,
            load_model_config(),
            load_feature_config(),
            protocol_name="season_2026_v1",
            event="Bahrain",
        )


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
    _write_target_artifact(config, dataset_path, "Bahrain")

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


def test_settlement_blocks_when_raw_identity_validation_fails(tmp_path: Path) -> None:
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
    _write_target_artifact(config, dataset_path, "Bahrain")
    metadata_path = config.session_metadata_output_dir / "2026/bahrain/q_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["event_name"] = "Austrian Grand Prix"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="Raw Q identity verification"):
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
    _write_target_artifact(config, dataset_path, "Bahrain")
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
    forecasts = pd.read_parquet(
        config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    )
    monza_forecasts = forecasts[forecasts["event_slug"].eq("monza")]
    assert monza_forecasts["prior_monitoring_event_count"].eq(1).all()
    assert monza_forecasts["prior_monitoring_event_orders"].eq("[2]").all()


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


def test_reconciliation_flags_legacy_noncanonical_event_order() -> None:
    protocol = {
        "protocol_name": "season_2026_v1",
        "monitor_season": 2026,
        "protocol_fingerprint": "abc",
    }
    registry = pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "monitor_season": 2026,
                "event_slug": "great-britain",
                "event_order": 9,
            }
        ]
    )
    forecasts = pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "event_slug": "great-britain",
                "forecast_id": "gb",
                "event_order": 44,
                "forecast_created_at_utc": "2026-07-04T10:00:00+00:00",
            }
        ]
    )
    settlements = pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "event_slug": "great-britain",
                "forecast_id": "gb",
                "event_order": 44,
                "settled_at_utc": "2026-07-04T12:00:00+00:00",
                "eligible_for_future_prior_evidence": True,
            }
        ]
    )

    reconciliation = build_event_order_reconciliation(
        protocol,
        registry,
        forecasts,
        settlements,
        pd.DataFrame(),
    )
    row = reconciliation.iloc[0]

    assert row["artifact_event_order"] == 44
    assert row["registry_event_order"] == 9
    assert bool(row["event_order_match"]) is False
    assert row["event_order_lineage_status"] == "legacy_noncanonical_event_order"
    assert bool(row["affected_by_prior_evidence_lineage"]) is True
    assert bool(row["eligible_for_future_prior_evidence_after_reconciliation"]) is False
    assert row["reconciliation_action"] == "exclude_from_prior_monitoring_evidence"


def test_synthetic_rehearsal_settlement_is_excluded_from_future_prior_evidence() -> None:
    protocol = {
        "protocol_name": "season_2026_v1",
        "monitor_season": 2026,
        "protocol_fingerprint": "abc",
    }
    registry = pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "monitor_season": 2026,
                "event_slug": "synthetic-clean-gp",
                "event_order": 3,
            }
        ]
    )
    forecasts = pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "event_slug": "synthetic-clean-gp",
                "forecast_id": "synthetic",
                "event_order": 3,
                "forecast_created_at_utc": "2026-07-04T10:00:00+00:00",
            }
        ]
    )
    settlements = pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "event_slug": "synthetic-clean-gp",
                "forecast_id": "synthetic",
                "event_order": 3,
                "settled_at_utc": "2026-07-04T12:00:00+00:00",
                "eligible_for_future_prior_evidence": True,
            }
        ]
    )

    reconciliation = build_event_order_reconciliation(
        protocol,
        registry,
        forecasts,
        settlements,
        pd.DataFrame(),
    )
    row = reconciliation.iloc[0]

    assert row["event_order_lineage_status"] == "valid_registry_lineage"
    assert bool(row["eligible_for_future_prior_evidence_after_reconciliation"]) is False
    assert row["reconciliation_reason"] == "synthetic_rehearsal_excluded_from_prior_evidence"


def test_legacy_noncanonical_rows_are_excluded_without_mutating_forecast(
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
    forecast_path = config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    forecasts = pd.read_parquet(forecast_path)
    forecasts["event_order"] = 44
    forecasts.to_parquet(forecast_path, index=False)
    before = forecast_path.read_bytes()
    _write_target_artifact(config, dataset_path, "Bahrain")
    create_prospective_monitoring_settlement(
        config,
        protocol_name="season_2026_v1",
        event="Bahrain",
    )

    summary = create_prospective_monitoring_report(config)
    after = forecast_path.read_bytes()
    payload = json.loads(summary.summary_path.read_text())
    reconciliation = pd.read_csv(
        config.metrics_output_dir / "prospective_monitoring_event_order_reconciliation.csv"
    )
    ledger = pd.read_csv(
        config.metrics_output_dir / "prospective_monitoring_shadow_evidence_ledger.csv"
    )

    assert after == before
    assert payload["integrity_status"] == "valid_with_legacy_artifact_exclusion"
    assert payload["monitoring_legacy_event_order_exclusion_count"] == 1
    assert reconciliation["event_order_lineage_status"].eq("legacy_noncanonical_event_order").all()
    assert not ledger["eligible_for_future_prior_evidence"].astype(bool).any()


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
    _write_registered_feature_artifacts(config, dataset_path)


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


def _write_registered_feature_artifacts(config: DataConfig, dataset_path: Path) -> None:
    dataset = pd.read_parquet(dataset_path)
    registry_path = config.metrics_output_dir / "prospective_monitoring_event_registry.csv"
    registry = pd.read_csv(registry_path)
    target_columns = [
        "quali_gap_to_pole_sec",
        "quali_position",
        "quali_best_lap_time_sec",
        "reached_q2",
        "reached_q3",
    ]
    for slug in registry["event_slug"].astype(str).tolist():
        event_rows = dataset[
            dataset["season"].astype(int).eq(2026)
            & dataset["event_slug"].astype(str).eq(slug)
            & dataset["checkpoint"].astype(str).eq("after_fp3")
        ].copy()
        if event_rows.empty:
            continue
        features = event_rows.drop(columns=target_columns)
        event = str(event_rows["event"].iloc[0])
        event_dir = config.project_root / "data/processed/monitoring/2026" / slug
        event_dir.mkdir(parents=True, exist_ok=True)
        feature_path = event_dir / "monitoring_fp3_features.parquet"
        features.to_parquet(feature_path, index=False)
        manifest_path = event_dir / "monitoring_event_manifest.json"
        manifest = {
            "season": 2026,
            "event": event,
            "event_slug": slug,
            "feature_artifact_path": _portable(feature_path, config.project_root),
            "feature_artifact_fingerprint": _fingerprint(feature_path),
            "feature_driver_count": int(features["driver"].nunique()),
            "target_coverage_status": "target_not_available",
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        _write_entry_list(config, event, features)
        mask = registry["event_slug"].astype(str).eq(slug)
        registry.loc[mask, "feature_artifact_path"] = _portable(feature_path, config.project_root)
        registry.loc[mask, "feature_artifact_fingerprint"] = _fingerprint(feature_path)
        registry.loc[mask, "feature_artifact_valid"] = True
        registry.loc[mask, "prequalification_ready"] = True
        registry.loc[mask, "forecastable"] = True
        registry.loc[mask, "target_artifact_path"] = ""
        registry.loc[mask, "target_artifact_present"] = False
        registry.loc[mask, "target_artifact_valid"] = False
        registry.loc[mask, "settleable"] = False
        registry.loc[mask, "target_coverage_status"] = "target_not_available"
    registry.to_csv(registry_path, index=False)


def _write_entry_list(config: DataConfig, event: str, features: pd.DataFrame) -> None:
    slug = event.strip().lower().replace(" ", "-")
    path = (
        config.project_root
        / "data/processed/monitoring"
        / "2026"
        / slug
        / "qualifying_entry_list.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["driver", "team"]
    if "driver_key" in features:
        columns.append("driver_key")
    if "team_key" in features:
        columns.append("team_key")
    features[columns].drop_duplicates().to_csv(path, index=False)


def _write_target_artifact(config: DataConfig, dataset_path: Path, event: str) -> None:
    dataset = pd.read_parquet(dataset_path)
    slug = event.strip().lower().replace(" ", "-")
    event_rows = dataset[
        dataset["season"].astype(int).eq(2026)
        & dataset["event_slug"].astype(str).eq(slug)
        & dataset["checkpoint"].astype(str).eq("after_fp3")
    ].copy()
    event_dir = config.project_root / "data/processed/monitoring/2026" / slug
    target_path = event_dir / "monitoring_qualifying_targets.parquet"
    coverage_path = event_dir / "monitoring_target_coverage.csv"
    raw_q_path = config.lap_output_dir / "2026" / slug / "q_laps.parquet"
    raw_q_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Driver": event_rows["driver"].tolist(),
            "Team": event_rows["team"].tolist(),
            "LapNumber": [1.0] * len(event_rows),
            "Stint": [1.0] * len(event_rows),
            "Compound": ["SOFT"] * len(event_rows),
            "TyreLife": [2.0] * len(event_rows),
            "LapTime": pd.to_timedelta(
                event_rows["quali_best_lap_time_sec"].astype(float).tolist(),
                unit="s",
            ),
            "Sector1Time": pd.to_timedelta([25.0] * len(event_rows), unit="s"),
            "Sector2Time": pd.to_timedelta([29.0] * len(event_rows), unit="s"),
            "Sector3Time": pd.to_timedelta([25.0] * len(event_rows), unit="s"),
            "IsAccurate": [True] * len(event_rows),
            "Deleted": [False] * len(event_rows),
            "PitOutTime": [pd.NaT] * len(event_rows),
            "PitInTime": [pd.NaT] * len(event_rows),
        }
    ).to_parquet(raw_q_path, index=False)
    metadata_path = config.session_metadata_output_dir / "2026" / slug / "q_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "season": 2026,
                "event_input": event,
                "event_name": f"{event} Grand Prix",
                "event_slug": slug,
                "session_input": "Q",
                "session_name": "Qualifying",
                "session_slug": "q",
                "status": "success",
            }
        ),
        encoding="utf-8",
    )
    targets = event_rows[
        [
            "season",
            "event",
            "event_slug",
            "driver",
            "driver_key",
            "team",
            "team_key",
            "quali_gap_to_pole_sec",
            "quali_position",
            "quali_best_lap_time_sec",
            "reached_q2",
            "reached_q3",
        ]
    ].copy()
    targets["target_evaluable"] = True
    targets["target_coverage_status"] = "target_coverage_complete"
    targets["target_source_status"] = "synthetic"
    targets["target_created_at_utc"] = "2026-01-01T00:00:00+00:00"
    targets.to_parquet(target_path, index=False)
    coverage = event_rows[
        ["season", "event", "event_slug", "checkpoint", "driver", "driver_key", "team", "team_key"]
    ].copy()
    coverage["feature_row_present"] = True
    coverage["qualifying_target_present"] = True
    coverage["target_evaluable"] = True
    coverage["target_missing_reason"] = ""
    coverage["target_source_status"] = "synthetic"
    coverage["target_validation_status"] = "valid"
    coverage["included_in_settlement_metrics"] = True
    coverage["excluded_from_settlement_metrics"] = False
    coverage["settlement_exclusion_reason"] = ""
    coverage.to_csv(coverage_path, index=False)
    manifest_path = event_dir / "monitoring_event_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "target_artifact_path": _portable(target_path, config.project_root),
            "target_artifact_fingerprint": _fingerprint(target_path),
            "target_coverage_artifact_path": _portable(coverage_path, config.project_root),
            "target_coverage_artifact_fingerprint": _fingerprint(coverage_path),
            "feature_driver_count": int(event_rows["driver"].nunique()),
            "target_driver_count": int(event_rows["driver"].nunique()),
            "evaluable_driver_count": int(event_rows["driver"].nunique()),
            "non_evaluable_driver_count": 0,
            "feature_target_coverage_rate": 1.0,
            "target_coverage_rate": 1.0,
            "target_coverage_status": "target_coverage_complete",
            "partial_target_coverage": False,
            "settlement_metric_status": "scorable",
            "raw_session_identity_status": "identity_verified",
            "raw_session_identity_verified": True,
            "raw_session_identity_blocking": False,
            "raw_session_identity_reason": (
                "Raw session path and metadata identities match the requested event."
            ),
            "raw_session_identity_recommended_action": (
                "Target onboarding and settlement may proceed when other monitoring gates pass."
            ),
            "raw_session_identity_raw_laps_path": _portable(raw_q_path, config.project_root),
            "raw_session_identity_metadata_path": _portable(metadata_path, config.project_root),
            "raw_session_identity_metadata_event_name": f"{event} Grand Prix",
            "raw_session_identity_metadata_event_slug": slug,
            "raw_session_identity_metadata_official_event_name": f"{event} Grand Prix",
            "raw_session_identity_metadata_season": 2026,
            "raw_session_identity_metadata_session": "Q",
            "raw_session_identity_path_event_slug": slug,
            "quarantine_status": "clear",
            "quarantine_reason": "",
            "legacy_noncanonical": False,
            "quarantined_for_prospective_evidence": False,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    registry_path = config.metrics_output_dir / "prospective_monitoring_event_registry.csv"
    registry = pd.read_csv(registry_path)
    mask = registry["event_slug"].astype(str).eq(slug)
    registry.loc[mask, "target_artifact_path"] = _portable(target_path, config.project_root)
    registry.loc[mask, "target_artifact_present"] = True
    registry.loc[mask, "target_artifact_valid"] = True
    registry.loc[mask, "settleable"] = True
    registry.loc[mask, "target_coverage_status"] = "target_coverage_complete"
    registry.loc[mask, "target_coverage_rate"] = 1.0
    registry.loc[mask, "feature_driver_count"] = int(event_rows["driver"].nunique())
    registry.loc[mask, "target_driver_count"] = int(event_rows["driver"].nunique())
    registry.loc[mask, "evaluable_driver_count"] = int(event_rows["driver"].nunique())
    registry.loc[mask, "non_evaluable_driver_count"] = 0
    registry.to_csv(registry_path, index=False)


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _portable(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
