"""Small operational views, kept separate from immutable predictive evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from f1_prediction.config import DataConfig, FeatureConfig, ModelConfig

WEEKEND_HISTORY_FILE = "autopilot_weekend_history.json"


def prediction_policy_status(
    config: DataConfig, model: ModelConfig, features: FeatureConfig
) -> dict[str, Any]:
    """Check the installed rule and candidate contract without fitting or writing."""
    from f1_prediction.modeling import fp3_production_policy as fp3

    result: dict[str, Any] = {
        "status": "not_installed",
        "production_policy_version": None,
        "mode": os.environ.get("APEX_FP3_POLICY_MODE", "conditional"),
        "transition_prior_events": None,
        "candidate_contract_valid": False,
        "historical_dataset_valid": False,
        "prior_feature_dependencies_valid": False,
        "session_data_source": config.session_data_source,
        "session_data_access": (
            "free_historical" if config.session_data_source == "openf1" else "fastf1_archive"
        ),
        "session_adapter_version": (
            "OPENF1_HISTORICAL_LAPS_V1" if config.session_data_source == "openf1" else None
        ),
    }
    try:
        policy = fp3.load_policy(config)
        if policy is None:
            return result
        result.update(
            production_policy_version=(
                fp3.ROLLBACK_VERSION if result["mode"] == "uniform_only" else fp3.POLICY_VERSION
            ),
            transition_prior_events=policy["transition_prior_events"],
        )
        fp3.validate_candidate(policy, model, features)
        result["candidate_contract_valid"] = True
        dataset = config.modeling_output_dir / "combined/modeling_dataset.parquet"
        result["historical_dataset_valid"] = (
            dataset.is_file()
            and fp3.file_hash(dataset) == policy["candidate_contract"]["historical_dataset_sha256"]
        )
        _validate_prior_feature_dependencies(config, policy)
        result["prior_feature_dependencies_valid"] = True
        result["status"] = "ready" if result["historical_dataset_valid"] else "blocked"
    except (OSError, ValueError, KeyError) as exc:
        result.update(status="blocked", reason=str(exc))
    return result


def _validate_prior_feature_dependencies(config: DataConfig, policy: dict[str, Any]) -> None:
    """Catch incomplete state transfers before the next pre-qualifying window."""
    import pandas as pd

    from f1_prediction.modeling import fp3_production_policy as fp3
    from f1_prediction.modeling import prospective_monitoring as monitoring

    metrics = config.metrics_output_dir
    protocol_name = str(
        json.loads((metrics / monitoring.PROTOCOL_FILE).read_text())["protocol_name"]
    )
    protocol = monitoring.load_protocol(metrics, protocol_name)
    registry = pd.read_csv(metrics / "prospective_monitoring_event_registry.csv")
    if "protocol_name" in registry:
        registry = registry[registry["protocol_name"].eq(protocol_name)]
    prior_keys = monitoring.settled_event_keys(
        metrics,
        protocol,
        registry,
        current_event_order=int(registry["event_order"].max()) + 1 if not registry.empty else 1,
    )
    for _, row in registry.iterrows():
        season, slug = int(row["monitor_season"]), str(row["event_slug"])
        if f"{season}/{slug}" not in prior_keys:
            continue
        snapshot = fp3.snapshot_path(config, protocol["protocol_name"], slug)
        if snapshot.is_file():
            metadata = json.loads(snapshot.read_text())
            inputs = snapshot.with_name("target_features.parquet")
            if fp3.file_hash(inputs) != metadata["target_features_file_sha256"]:
                raise ValueError(f"Frozen prior features changed for {season}/{slug}")
        else:
            practice = (
                config.session_features_output_dir
                / str(season)
                / slug
                / "practice_session_features.parquet"
            )
            if not practice.is_file():
                raise ValueError(
                    f"Missing prior practice aggregates for {season}/{slug}; "
                    "run build-session-features from the preserved raw practice laps."
                )
            fp3.registered_audited_features(config, row, policy)


def update_weekend_history(metrics: Path, tick: dict[str, Any]) -> None:
    """Keep one operational outcome per weekend, including those without forecasts.

    On upgrade, rebuild from the existing run log once. This view never creates
    registry entries or treats missed windows as prospective forecasts.
    """
    path = metrics / WEEKEND_HISTORY_FILE
    if path.is_file():
        records = json.loads(path.read_text())
    else:
        records = []
        runs = metrics / "autopilot_runs.jsonl"
        if runs.is_file():
            with runs.open() as stream:
                for line in stream:
                    if line.strip():
                        _merge_weekend(records, json.loads(line))
    _merge_weekend(records, tick)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _merge_weekend(records: list[dict[str, Any]], tick: dict[str, Any]) -> None:
    if not tick.get("event_slug") or tick.get("dry_run"):
        return
    key = (tick.get("season"), tick["event_slug"])
    previous = next((row for row in records if (row["season"], row["event_slug"]) == key), None)
    row = {
        name: tick.get(name)
        for name in (
            "season",
            "event",
            "event_slug",
            "round_number",
            "event_format",
            "orchestrator_state_after",
            "action_result",
            "forecast_exists",
            "settlement_exists",
            "completed_at_utc",
        )
    }
    row["last_data_issue"] = (
        tick.get("error_message_safe")
        if "TIME_ELAPSED_DATA_PENDING" in str(tick.get("orchestrator_state_after"))
        and tick.get("error_message_safe")
        else (previous or {}).get("last_data_issue")
    )
    if previous is not None:
        # Calendar failures without an event must not erase the last valid outcome.
        previous.update(row)
    else:
        records.append(row)
    records.sort(key=lambda item: (item["season"], item["round_number"] or 0))
