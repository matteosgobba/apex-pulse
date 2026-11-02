import json
from dataclasses import replace
from pathlib import Path

from f1_prediction.config import load_data_config, load_feature_config, load_model_config
from f1_prediction.operational_diagnostics import (
    WEEKEND_HISTORY_FILE,
    prediction_policy_status,
    update_weekend_history,
)


def test_policy_status_checks_real_contract_and_detects_conflicts(monkeypatch, tmp_path):
    from f1_prediction.modeling import fp3_production_policy as fp3

    config = replace(load_data_config(), modeling_output_dir=tmp_path)
    dataset = tmp_path / "combined/modeling_dataset.parquet"
    dataset.parent.mkdir()
    dataset.write_bytes(b"fixture")
    expected = fp3.load_policy(config)["candidate_contract"]["historical_dataset_sha256"]
    actual_hash = fp3.file_hash
    monkeypatch.setattr(
        fp3, "file_hash", lambda path: expected if path == dataset else actual_hash(path)
    )
    monkeypatch.setattr(
        "f1_prediction.operational_diagnostics._validate_prior_feature_dependencies",
        lambda *args: None,
    )
    model, features = load_model_config(), load_feature_config()
    status = prediction_policy_status(config, model, features)
    assert status["status"] == "ready"
    assert status["production_policy_version"] == "FP3_CONDITIONAL_POLICY_V1"
    assert status["candidate_contract_valid"]
    assert status["prior_feature_dependencies_valid"]
    monkeypatch.setenv("APEX_FP3_POLICY_MODE", "uniform_only")
    assert prediction_policy_status(config, model, features)["production_policy_version"] == (
        "FP3_UNIFORM_ROLLBACK_V1"
    )
    assert prediction_policy_status(config, replace(model, random_state=99), features)[
        "status"
    ] == ("blocked")


def test_policy_missing_dataset_is_not_reported_ready(tmp_path):
    config = replace(load_data_config(), modeling_output_dir=tmp_path)
    status = prediction_policy_status(config, load_model_config(), load_feature_config())
    assert status["status"] == "blocked"
    assert status["historical_dataset_valid"] is False


def test_new_forecast_source_provenance_uses_original_ingestion_metadata(tmp_path):
    import pandas as pd

    from f1_prediction.modeling.fp3_production_policy import file_hash, practice_input_sources

    config = replace(
        load_data_config(),
        session_metadata_output_dir=tmp_path / "metadata",
        lap_output_dir=tmp_path / "laps",
        session_data_source="openf1",
    )
    metadata = config.session_metadata_output_dir / "2026/example/fp3_metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"data_source": "fastf1"}')
    laps = config.lap_output_dir / "2026/example/fp3_laps.parquet"
    laps.parent.mkdir(parents=True)
    laps.write_bytes(b"preserved source")
    before = metadata.read_bytes()
    source = practice_input_sources(config, "example", pd.DataFrame({"season": [2026]}))[0]
    assert source["data_source"] == "fastf1"
    assert source["metadata_sha256"] == file_hash(metadata)
    assert source["laps_sha256"] == file_hash(laps)
    assert metadata.read_bytes() == before


def test_policy_readiness_blocks_missing_prior_practice_dependency(monkeypatch, tmp_path):
    import pandas as pd
    import pytest

    from f1_prediction.modeling import prospective_monitoring as monitoring
    from f1_prediction.operational_diagnostics import _validate_prior_feature_dependencies

    config = replace(
        load_data_config(),
        metrics_output_dir=tmp_path,
        session_features_output_dir=tmp_path / "features",
    )
    (tmp_path / monitoring.PROTOCOL_FILE).write_text('{"protocol_name": "test"}')
    monkeypatch.setattr(monitoring, "load_protocol", lambda *args: {"protocol_name": "test"})
    monkeypatch.setattr(monitoring, "settled_event_keys", lambda *args, **kwargs: {"2026/prior"})
    pd.DataFrame([dict(monitor_season=2026, event_slug="prior", event_order=1)]).to_csv(
        tmp_path / "prospective_monitoring_event_registry.csv", index=False
    )
    with pytest.raises(ValueError, match="Missing prior practice aggregates for 2026/prior"):
        _validate_prior_feature_dependencies(config, {})


def test_history_recovers_missed_weekends_and_keeps_reason_across_rollover(tmp_path: Path):
    def tick(event, order, result, state, **extra):
        return dict(
            season=2026,
            event=event,
            event_slug=event,
            round_number=order,
            action_result=result,
            orchestrator_state_after=state,
            forecast_exists=False,
            **extra,
        )

    runs = [
        tick("singapore", 17, "unsupported_weekend_format", "UNSUPPORTED_WEEKEND_FORMAT"),
        tick(
            "usa",
            18,
            "not_yet_available",
            "FP3_TIME_ELAPSED_DATA_PENDING",
            retryable=True,
            error_message_safe="FastF1 timing not loaded",
        ),
        tick("usa", 18, "forecast_window_missed", "BLOCKED"),
    ]
    log = tmp_path / "autopilot_runs.jsonl"
    log.write_text("".join(json.dumps(row) + "\n" for row in runs))
    original = log.read_bytes()
    current = tick("mexico", 19, "waiting", "WAITING_FOR_FP1")
    update_weekend_history(tmp_path, current)
    update_weekend_history(tmp_path, current)

    history = json.loads((tmp_path / WEEKEND_HISTORY_FILE).read_text())
    assert len(history) == 3
    assert history[1]["action_result"] == "forecast_window_missed"
    assert history[1]["last_data_issue"] == "FastF1 timing not loaded"
    assert log.read_bytes() == original
    assert not (tmp_path / "prospective_monitoring_event_registry.csv").exists()
