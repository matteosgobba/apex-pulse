from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.config import DataConfig
from f1_prediction.modeling.live_event_integrity import (
    LIVE_VALIDATION_STATUS_FILE,
    compare_live_integrity,
    create_live_integrity_baseline,
    create_live_validation_checkpoint,
    observe_live_validation_tick,
    validate_live_validation_status,
    write_live_integrity_baseline,
)
from f1_prediction.modeling.prospective_monitoring import forecast_snapshot_hash


def test_baseline_is_deterministic_and_event_scoped(tmp_path: Path) -> None:
    config = _runtime(tmp_path)

    first = create_live_integrity_baseline(config, generated_at_utc="2026-08-04T12:00:00+00:00")
    second = create_live_integrity_baseline(config, generated_at_utc="2026-08-04T12:00:00+00:00")

    assert first == second
    assert first["event_count"] == 2
    event = next(item for item in first["events"] if item["event_slug"] == "older-gp")
    assert event["forecast_fingerprint"]["row_count"] == 1
    assert event["settlement_fingerprint"]["row_count"] == 1
    assert event["event_table_fingerprints"]["training_manifest"]["row_count"] == 1
    assert "monitoring/monitoring_event_manifest.json" in event["evidence_fingerprints"]
    assert "historical_dashboard_event" in event["evidence_fingerprints"]


def test_compare_accepts_only_chronological_append(tmp_path: Path) -> None:
    config = _runtime(tmp_path)
    baseline = create_live_integrity_baseline(config)
    registry_path = config.metrics_output_dir / "prospective_monitoring_event_registry.csv"
    registry = pd.read_csv(registry_path)
    appended = registry.iloc[-1].copy()
    appended["event"] = "Future GP"
    appended["event_slug"] = "future-gp"
    appended["event_order"] = 3
    pd.concat([registry, appended.to_frame().T], ignore_index=True).to_csv(
        registry_path, index=False
    )

    result = compare_live_integrity(config, baseline)

    assert result["classification"] == "VALID_APPEND"
    assert result["success"] is True
    assert result["new_events"][0]["event_slug"] == "future-gp"


@pytest.mark.parametrize(
    ("mutation", "classification"),
    [
        ("forecast", "PREEXISTING_EVENT_MUTATED"),
        ("missing", "MISSING_PREEXISTING_EVENT"),
        ("static", "STATIC_INVARIANT_CHANGED"),
        ("chronology", "INVALID_CHRONOLOGY"),
    ],
)
def test_compare_blocks_historical_mutation(
    tmp_path: Path, mutation: str, classification: str
) -> None:
    config = _runtime(tmp_path)
    baseline = create_live_integrity_baseline(config)
    metrics = config.metrics_output_dir
    if mutation == "forecast":
        path = metrics / "prospective_monitoring_forecasts.parquet"
        frame = pd.read_parquet(path)
        frame.loc[frame["event_slug"].eq("older-gp"), "prediction_gap_sec"] = 99.0
        frame.to_parquet(path, index=False)
    elif mutation == "missing":
        path = metrics / "prospective_monitoring_event_registry.csv"
        frame = pd.read_csv(path)
        frame[~frame["event_slug"].eq("older-gp")].to_csv(path, index=False)
    elif mutation == "static":
        (config.modeling_output_dir / "combined/modeling_dataset.parquet").write_bytes(b"changed")
    else:
        path = metrics / "prospective_monitoring_event_registry.csv"
        frame = pd.read_csv(path)
        duplicate = frame.iloc[-1].copy()
        duplicate["event_slug"] = "colliding-gp"
        pd.concat([frame, duplicate.to_frame().T], ignore_index=True).to_csv(path, index=False)

    result = compare_live_integrity(config, baseline)

    assert result["classification"] == classification
    assert result["success"] is False


def test_checkpoint_write_refuses_overwrite_and_does_not_mutate_sources(tmp_path: Path) -> None:
    config = _runtime(tmp_path)
    forecast = config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    before = forecast.read_bytes()
    output = tmp_path / "pre_weekend_baseline.json"

    written = write_live_integrity_baseline(
        config, output, generated_at_utc="2026-08-04T12:00:00+00:00"
    )

    assert json.loads(output.read_text()) == written
    assert forecast.read_bytes() == before
    with pytest.raises(FileExistsError):
        write_live_integrity_baseline(config, output)


def test_checkpoint_embeds_reusable_current_manifest_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    config = _runtime(tmp_path)
    baseline = tmp_path / "baseline.json"
    checkpoint = tmp_path / "post_forecast.json"
    write_live_integrity_baseline(config, baseline)

    payload = create_live_validation_checkpoint(
        config, baseline, checkpoint, stage="post_forecast_validation"
    )

    assert payload["integrity"]["classification"] == "UNCHANGED"
    assert payload["current_state_manifest"]["event_count"] == 2
    assert compare_live_integrity(config, checkpoint)["classification"] == "UNCHANGED"
    with pytest.raises(FileExistsError):
        create_live_validation_checkpoint(
            config, baseline, checkpoint, stage="post_forecast_validation"
        )


def test_live_observer_records_trigger_timeline_and_high_watermark(tmp_path: Path) -> None:
    config = _runtime(tmp_path)
    baseline = config.metrics_output_dir / "live_validation/pre_weekend_baseline.json"
    write_live_integrity_baseline(config, baseline)
    tick = _tick()

    first = observe_live_validation_tick(
        config,
        tick,
        trigger_source="scheduler",
        scheduler_enabled=True,
        scheduler_running=True,
    )
    tick["fastf1_cache_bytes"] = 50
    tick["runtime_total_known_bytes"] = 500
    second = observe_live_validation_tick(
        config,
        tick,
        trigger_source="scheduler",
        scheduler_enabled=True,
        scheduler_running=True,
    )

    assert validate_live_validation_status(first)["trigger_source"] == "scheduler"
    assert first["timeline"]["first_fp3_readiness_success_at_utc"] == tick["completed_at_utc"]
    assert first["forecast_row_count"] == 1
    assert second["fastf1_cache_high_watermark_bytes"] == 100
    assert second["runtime_total_high_watermark_bytes"] == 1000
    assert second["historical_integrity_status"] == "UNCHANGED"
    assert second["operator_attention_category"] == "NONE"
    assert (config.metrics_output_dir / LIVE_VALIDATION_STATUS_FILE).is_file()


def test_live_observer_classifies_retry_and_never_touches_ledgers(tmp_path: Path) -> None:
    config = _runtime(tmp_path)
    forecast = config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    before = forecast.read_bytes()
    tick = _tick()
    tick.update(
        {
            "orchestrator_state_after": "FP3_TIME_ELAPSED_DATA_PENDING",
            "retryable": True,
            "retry_reason": "session not published",
            "fp3_status": "incomplete",
        }
    )

    status = observe_live_validation_tick(
        config,
        tick,
        trigger_source="manual",
        scheduler_enabled=True,
        scheduler_running=True,
    )

    assert status["operator_attention_category"] == "RETRYING_DATA_AVAILABILITY"
    assert forecast.read_bytes() == before


def _runtime(root: Path) -> DataConfig:
    metrics = root / "reports/metrics"
    modeling = root / "data/processed/modeling"
    metrics.mkdir(parents=True)
    (modeling / "combined").mkdir(parents=True)
    protocol = {
        "protocol_name": "season_2026_v1",
        "protocol_fingerprint": "frozen-protocol",
    }
    (metrics / "prospective_monitoring_protocol.json").write_text(json.dumps(protocol))
    registry = pd.DataFrame(
        [
            _registry_row("Older GP", "older-gp", 1, "complete"),
            _registry_row("Current GP", "current-gp", 2, "not_available"),
        ]
    )
    registry.to_csv(metrics / "prospective_monitoring_event_registry.csv", index=False)
    forecasts = pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "protocol_fingerprint": "frozen-protocol",
                "forecast_id": "forecast-older",
                "season": 2026,
                "event_slug": "older-gp",
                "driver": "AAA",
                "prediction_role": "observed_live_policy",
                "prediction_gap_sec": 0.1,
                "diagnostic_only": False,
                "forecast_created_at_utc": "2026-07-01T10:00:00+00:00",
            },
            {
                "protocol_name": "season_2026_v1",
                "protocol_fingerprint": "frozen-protocol",
                "forecast_id": "forecast-current",
                "season": 2026,
                "event_slug": "current-gp",
                "driver": "BBB",
                "prediction_role": "observed_live_policy",
                "prediction_gap_sec": 0.2,
                "diagnostic_only": False,
                "forecast_created_at_utc": "2026-08-01T10:00:00+00:00",
            },
        ]
    )
    forecasts.to_parquet(metrics / "prospective_monitoring_forecasts.parquet", index=False)
    pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "season": 2026,
                "event_slug": slug,
                "forecast_id": forecast_id,
                "forecast_snapshot_hash": forecast_snapshot_hash(
                    forecasts[forecasts["forecast_id"].eq(forecast_id)], pd.DataFrame()
                ),
            }
            for slug, forecast_id in (
                ("older-gp", "forecast-older"),
                ("current-gp", "forecast-current"),
            )
        ]
    ).to_csv(metrics / "prospective_monitoring_selection_log.csv", index=False)
    settlements = pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "protocol_fingerprint": "frozen-protocol",
                "forecast_id": "forecast-older",
                "settlement_id": "settlement-older-AAA",
                "season": 2026,
                "event_slug": "older-gp",
                "driver": "AAA",
                "prediction_role": "observed_live_policy",
                "diagnostic_only": False,
                "settlement_valid": True,
                "forecast_preexisted_settlement": True,
                "forecast_fingerprint_valid": True,
                "forecast_mutation_detected": False,
                "settled_at_utc": "2026-07-01T15:00:00+00:00",
            }
        ]
    )
    settlements.to_parquet(metrics / "prospective_monitoring_settlements.parquet", index=False)
    pd.DataFrame(
        [
            {
                "protocol_name": "season_2026_v1",
                "monitor_season": 2026,
                "event_key": "2026/older-gp",
                "test_event": "2026/older-gp",
                "training_event_count": 1,
            }
        ]
    ).to_csv(metrics / "prospective_monitoring_training_manifest.csv", index=False)
    (modeling / "combined/modeling_dataset.parquet").write_bytes(b"frozen-modeling-data")
    event_dir = root / "data/processed/monitoring/2026/older-gp"
    event_dir.mkdir(parents=True)
    (event_dir / "monitoring_event_manifest.json").write_text('{"event":"Older GP"}\n')
    dashboard = root / "reports/dashboard"
    dashboard.mkdir(parents=True)
    historical = {
        "data": {
            "legacy_descriptive_records": [
                {"event_identity": {"event_slug": "older-gp"}, "state": "settled"}
            ],
            "synthetic_rehearsal_records": [],
            "valid_prospective_monitoring": {"events": []},
        }
    }
    (dashboard / "historical_monitoring_summary.json").write_text(json.dumps(historical))
    return DataConfig(
        project_root=root,
        fastf1_cache_dir=root / "data/raw/fastf1_cache",
        lap_output_dir=root / "data/raw/laps",
        session_metadata_output_dir=root / "data/raw/session_metadata",
        clean_lap_output_dir=root / "data/interim/clean_laps",
        session_features_output_dir=root / "data/processed/session_features",
        modeling_output_dir=modeling,
        metrics_output_dir=metrics,
    )


def _registry_row(event: str, slug: str, order: int, coverage: str) -> dict[str, object]:
    return {
        "protocol_name": "season_2026_v1",
        "monitor_season": 2026,
        "event_order": order,
        "event": event,
        "event_slug": slug,
        "target_coverage_status": coverage,
        "partial_target_coverage": coverage == "partial",
    }


def _tick() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "started_at_utc": "2026-08-01T11:00:00+00:00",
        "completed_at_utc": "2026-08-01T11:00:02+00:00",
        "season": 2026,
        "event": "Current GP",
        "event_slug": "current-gp",
        "round_number": 2,
        "event_format": "conventional",
        "calendar_source": "fake",
        "orchestrator_state_after": "FORECAST_AVAILABLE",
        "fp1_status": "ready",
        "fp2_status": "ready",
        "fp3_status": "ready",
        "qualifying_status": "not_probed",
        "action_considered": "run_before_qualifying",
        "action_taken": "run_before_qualifying",
        "retryable": False,
        "retry_reason": None,
        "error_message_safe": None,
        "fastf1_cache_bytes": 100,
        "runtime_total_known_bytes": 1000,
        "volume_capacity_bytes": 5000,
        "cache_warning_status": "ok",
        "operational_event": {
            "supported": True,
            "sessions": [
                {
                    "session": "FP3",
                    "scheduled_start_utc": "2026-08-01T09:00:00+00:00",
                    "scheduled_end_utc": "2026-08-01T10:00:00+00:00",
                },
                {
                    "session": "Q",
                    "scheduled_start_utc": "2026-08-01T14:00:00+00:00",
                    "scheduled_end_utc": "2026-08-01T15:00:00+00:00",
                },
            ],
        },
    }
