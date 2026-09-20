"""Versioned FP3 deployment rule and the historical candidate's input contract.

This module never imports Replay V2 or selects a rule using observed errors.
Historical settlements may supply training outcomes; only forecasts made after
the freeze belong to the new prospective evaluation.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import asdict
from functools import wraps
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_prediction.config import DataConfig, FeatureConfig, ModelConfig
from f1_prediction.data.identity import add_identity_columns
from f1_prediction.data.monitoring_onboarding import (
    forbidden_target_columns,
    latest_driver_identity,
)
from f1_prediction.features.build import build_session_features_output_path
from f1_prediction.features.modeling_dataset import SESSION_IDENTIFIER_COLUMNS
from f1_prediction.features.relative_features import add_relative_practice_features
from f1_prediction.modeling.prospective_replay import event_key_series

POLICY_VERSION = "FP3_CONDITIONAL_POLICY_V1"
ROLLBACK_VERSION = "FP3_UNIFORM_ROLLBACK_V1"
UNIFORM = "uniform_default"
SEASON_AWARE = "season_aware_weighted_candidate"
CONFIG_FILE = "fp3_production_policy_v1.json"
METADATA_COLUMNS = [
    "production_policy_version",
    "policy_freeze_at_utc",
    "candidate_version",
    "candidate_contract_fingerprint",
    "prior_current_season_eligible_events",
    "regime",
    "intended_candidate",
    "selected_candidate",
    "fallback_reason",
    "uniform_prediction_gap_sec",
    "season_aware_prediction_gap_sec",
    "training_cutoff",
    "training_dataset_fingerprint",
    "target_features_fingerprint",
    "prospective_evidence_class",
    "qualifying_start_utc",
    "runtime_versions",
]


def serialized_monitoring_write(function):
    """Serialize scheduler/CLI writers; immutable snapshots must have a single writer."""

    @wraps(function)
    def run(config, *args, **kwargs):
        if load_policy(config) is None:
            return function(config, *args, **kwargs)
        import fcntl

        config.metrics_output_dir.mkdir(parents=True, exist_ok=True)
        with (config.metrics_output_dir / ".fp3_policy.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return function(config, *args, **kwargs)

    return run


def signature(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    # JSON normalizes nullable/object dtypes across a Parquet round trip.
    normalized = frame.map(
        lambda value: (
            float(value) if isinstance(value, Real) and not isinstance(value, bool) else value
        )
    )
    return hashlib.sha256(
        normalized.to_json(orient="split", index=False, double_precision=15).encode()
    ).hexdigest()


def load_policy(config: DataConfig) -> dict[str, Any] | None:
    path = config.project_root / "configs" / CONFIG_FILE
    recorded = config.metrics_output_dir / "fp3_conditional_v1" / "policy_freeze.json"
    if not path.is_file():
        if recorded.is_file():
            raise ValueError("Missing configuration for an already frozen FP3 policy")
        return None  # Older protocols/test projects keep their original semantics.
    policy = json.loads(path.read_text())
    if recorded.is_file() and signature(json.loads(recorded.read_text())) != signature(policy):
        raise ValueError("Frozen FP3 policy/config conflict; use a new explicit version")
    if policy["policy_version"] != POLICY_VERSION or policy["transition_prior_events"] != 5:
        raise ValueError("FP3 V1 requires its frozen version and exactly five prior events")
    if signature(policy["candidate_contract"]) != policy["candidate_contract_fingerprint"]:
        raise ValueError("FP3 candidate contract fingerprint conflict")
    if os.environ.get("APEX_FP3_POLICY_MODE", "conditional") not in {"conditional", "uniform_only"}:
        raise ValueError("APEX_FP3_POLICY_MODE must be conditional or uniform_only")
    return policy


def validate_candidate(policy: dict, model: ModelConfig, features: FeatureConfig | None) -> None:
    contract = policy["candidate_contract"]
    observed = {
        "random_state": model.random_state,
        "random_forest": asdict(model.random_forest),
        "temporal_weighting": asdict(model.temporal_weighting),
    }
    if (
        signature(observed) != signature(contract["model"])
        or features is None
        or signature(asdict(features)) != signature(contract["features_config"])
    ):
        raise ValueError(
            "Frozen FP3 candidate/config conflict; a new candidate version is required"
        )
    source_root = Path(__file__).parents[1]
    for relative, expected in contract["source_hashes"].items():
        if file_hash(source_root / relative) != expected:
            raise ValueError(f"Frozen FP3 predictive source conflict: {relative}")
    from f1_prediction.modeling.prospective_monitoring import fit_monitoring_source_candidate

    functions = {
        f.__name__: f
        for f in (
            audited_feature_rows,
            registered_audited_features,
            legal_dataset,
            fit_monitoring_source_candidate,
        )
    }
    for name, expected in contract.get("function_hashes", {}).items():
        if hashlib.sha256(inspect.getsource(functions[name]).encode()).hexdigest() != expected:
            raise ValueError(f"Frozen FP3 predictive function conflict: {name}")


def selection(prior_count: int, *, mode: str = "conditional") -> dict[str, Any]:
    if prior_count < 0 or mode not in {"conditional", "uniform_only"}:
        raise ValueError("Invalid FP3 prior count or policy mode")
    established = prior_count >= 5
    candidate = SEASON_AWARE if established and mode == "conditional" else UNIFORM
    return {
        "production_policy_version": POLICY_VERSION if mode == "conditional" else ROLLBACK_VERSION,
        "prior_current_season_eligible_events": prior_count,
        "regime": "established" if established else "cold_start",
        "intended_candidate": candidate,
        "selected_candidate": candidate,
        "fallback_reason": "",
    }


def audited_feature_rows(practice: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Preserve the pre-f08c266 direct-driver joins used by the 126-feature candidate.

    Team features are present only with that driver's session row. The upstream
    roster/participation improvements remain available to other consumers.
    """
    if forbidden_target_columns(practice):
        raise ValueError("Qualifying target information in practice inputs")
    practice = add_relative_practice_features(add_identity_columns(practice))
    direct = [
        c
        for c in practice
        if c not in SESSION_IDENTIFIER_COLUMNS
        and not c.lower().startswith(("q_", "quali_"))
        and "target" not in c.lower()
    ]
    result = latest_driver_identity(practice)
    result["checkpoint"] = "after_fp3"
    for session in ("FP1", "FP2", "FP3"):
        rows = practice.loc[practice["session"].eq(session), ["driver_key", *direct]]
        result = result.merge(
            rows.rename(columns={c: f"{session.lower()}_{c}" for c in direct}),
            on="driver_key",
            how="left",
            validate="one_to_one",
        )
    identifiers = [c for c in result if c in SESSION_IDENTIFIER_COLUMNS or c == "checkpoint"]
    return result.reindex(columns=[*identifiers, *columns])


def registered_audited_features(config: DataConfig, row: pd.Series, policy: dict) -> pd.DataFrame:
    from f1_prediction.modeling import prospective_monitoring as monitoring

    path = monitoring.resolve_registered_path(config, row.get("feature_artifact_path"))
    if path is None or not path.is_file():
        raise ValueError("Missing registered targetless FP3 feature artifact")
    registered = pd.read_parquet(path)
    if forbidden_target_columns(registered):
        raise ValueError("Registered FP3 features contain targets")
    if monitoring.onboarding_artifact_fingerprint(path) != str(row["feature_artifact_fingerprint"]):
        raise ValueError("Registered FP3 feature fingerprint conflict")
    practice_path = build_session_features_output_path(
        config.session_features_output_dir, int(row["monitor_season"]), str(row["event"])
    )
    result = audited_feature_rows(
        pd.read_parquet(practice_path), policy["candidate_contract"]["feature_columns"]
    )
    # Use the registered roster; do not use qualifying to infer driver membership.
    ids = [c for c in registered if c in SESSION_IDENTIFIER_COLUMNS or c == "checkpoint"]
    result = registered[ids].merge(
        result[["driver_key", *policy["candidate_contract"]["feature_columns"]]],
        on="driver_key",
        how="left",
        validate="one_to_one",
    )
    if (
        not result["season"].eq(int(row["monitor_season"])).all()
        or not result["event_slug"].eq(str(row["event_slug"])).all()
    ):
        raise ValueError("FP3 feature event identity conflict")
    return result


def legal_dataset(
    config: DataConfig,
    protocol: dict,
    registry: pd.DataFrame,
    event_row: pd.Series,
    base: pd.DataFrame,
    policy: dict,
    *,
    forecast_time: str,
) -> tuple[pd.DataFrame, dict]:
    """Materialize settled prior events, then count actual eligible FP3 training events."""
    from f1_prediction.modeling import prospective_monitoring as monitoring

    season = int(protocol["monitor_season"])
    base = base[
        base["season"].isin(protocol["train_seasons"])
        & base["season"].lt(season)
        & base["checkpoint"].eq("after_fp3")
    ].dropna(subset=["quali_gap_to_pole_sec"])
    allowed = monitoring.settled_event_keys(
        config.metrics_output_dir,
        protocol,
        registry,
        current_event_order=int(event_row["event_order"]),
    )
    settlements = monitoring.read_parquet(
        config.metrics_output_dir / "prospective_monitoring_settlements.parquet"
    )
    prior_frames = []
    current_registry = registry[
        registry["protocol_name"].eq(protocol["protocol_name"])
        & registry["monitor_season"].eq(season)
    ]
    for _, row in current_registry.sort_values("event_order").iterrows():
        key = f"{season}/{row['event_slug']}"
        if key not in allowed or int(row["event_order"]) >= int(event_row["event_order"]):
            continue
        outcomes = settlements[
            settlements["protocol_name"].eq(protocol["protocol_name"])
            & settlements["event_slug"].eq(row["event_slug"])
            & settlements["settlement_evaluable"].fillna(False)
            & pd.to_datetime(settlements["settled_at_utc"], utc=True).lt(
                pd.Timestamp(forecast_time)
            )
        ].copy()
        if outcomes.empty:
            continue
        if outcomes.groupby("driver_key")["actual_gap_sec"].nunique().gt(1).any():
            raise ValueError(f"Conflicting settled outcomes for {key}")
        frozen_path = snapshot_path(config, protocol["protocol_name"], str(row["event_slug"]))
        if frozen_path.is_file():
            metadata = json.loads(frozen_path.read_text())
            inputs_path = frozen_path.with_name("target_features.parquet")
            if file_hash(inputs_path) != metadata["target_features_file_sha256"]:
                raise ValueError(f"Frozen prior features changed for {key}")
            features = pd.read_parquet(inputs_path).drop(
                columns=["quali_gap_to_pole_sec"], errors="ignore"
            )
        else:
            features = registered_audited_features(config, row, policy)
        outcomes = outcomes.drop_duplicates("driver_key")[["driver_key", "actual_gap_sec"]]
        settled = features.merge(outcomes, on="driver_key", validate="one_to_one")
        settled = settled.rename(columns={"actual_gap_sec": "quali_gap_to_pole_sec"})
        settled = settled[np.isfinite(pd.to_numeric(settled["quali_gap_to_pole_sec"]))]
        if not settled.empty:
            prior_frames.append(settled)
    train = pd.concat([base, *prior_frames], ignore_index=True, sort=False)
    target = registered_audited_features(config, event_row, policy)
    target["quali_gap_to_pole_sec"] = np.nan
    prior_keys = sorted(set(event_key_series(pd.concat(prior_frames))) if prior_frames else [])
    context = selection(len(prior_keys), mode=os.environ.get("APEX_FP3_POLICY_MODE", "conditional"))
    context.update(
        {
            "policy_freeze_at_utc": policy["freeze_at_utc"],
            "candidate_contract_fingerprint": policy["candidate_contract_fingerprint"],
            "training_cutoff": json.dumps(list(dict.fromkeys(event_key_series(train)))),
            "training_dataset_fingerprint": frame_hash(train),
            "target_features_fingerprint": frame_hash(target),
            "prospective_evidence_class": "post_freeze_prospective",
        }
    )
    import platform

    import sklearn

    context["runtime_versions"] = json.dumps(
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        sort_keys=True,
    )
    return pd.concat([train, target], ignore_index=True, sort=False), context


def qualifying_deadline(season: int, event_slug: str) -> str:
    """Use the same FastF1 calendar as the scheduler, including manual CLI calls."""
    import fastf1

    from f1_prediction.utils.paths import slugify

    schedule = fastf1.get_event_schedule(season, include_testing=False)
    rows = schedule[schedule["EventName"].map(slugify).eq(event_slug)]
    if len(rows) != 1:
        raise ValueError("No unique canonical qualifying deadline")
    row = rows.iloc[0]
    if str(row.get("EventFormat")) != "conventional":
        raise ValueError("FP3 conditional V1 supports standard conventional weekends only")
    for index in range(1, 6):
        if row.get(f"Session{index}") == "Qualifying":
            date = pd.to_datetime(row.get(f"Session{index}DateUtc"), utc=True)
            if not pd.isna(date):
                return date.isoformat()
    raise ValueError("Missing canonical qualifying deadline")


def ensure_forecast_window(policy: dict, now: str, deadline: str) -> None:
    if not pd.Timestamp(policy["freeze_at_utc"]) <= pd.Timestamp(now) < pd.Timestamp(deadline):
        raise ValueError("FP3 forecast must be frozen after policy freeze and before qualifying")


def snapshot_path(config: DataConfig, protocol: str, slug: str) -> Path:
    return config.metrics_output_dir / "fp3_conditional_v1" / protocol / slug / "snapshot.json"


def verify_existing(
    config: DataConfig,
    protocol: str,
    slug: str,
    policy: dict,
    model: ModelConfig | None,
    features: FeatureConfig | None,
) -> bool:
    """Reuse frozen output even after settlement/rollback; never regenerate it."""
    from f1_prediction.modeling import prospective_monitoring as monitoring

    if model is not None:
        validate_candidate(policy, model, features)
    forecasts = monitoring.read_parquet(
        config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    )
    if forecasts.empty:
        return False
    rows = forecasts[forecasts["protocol_name"].eq(protocol) & forecasts["event_slug"].eq(slug)]
    if rows.empty:
        return False
    path = snapshot_path(config, protocol, slug)
    if path.is_file():
        snapshot = json.loads(path.read_text())
        if snapshot["candidate_contract_fingerprint"] != policy["candidate_contract_fingerprint"]:
            raise ValueError("Frozen forecast candidate/config conflict")
        frozen = pd.read_parquet(path.with_name("forecasts.parquet"))
        if file_hash(path.with_name("forecasts.parquet")) != snapshot["forecast_file_sha256"]:
            raise ValueError("Frozen forecast mutation detected")
        actual = rows[rows["forecast_id"].eq(snapshot["forecast_id"])].reindex(
            columns=frozen.columns
        )
        if frame_hash(actual.reset_index(drop=True)) != frame_hash(frozen):
            raise ValueError("Canonical forecast differs from frozen FP3 snapshot")
    elif "production_policy_version" in rows and rows["production_policy_version"].notna().any():
        raise ValueError("Conditional forecast is missing its immutable provenance snapshot")
    expected = monitoring.expected_forecast_hash(config.metrics_output_dir, protocol, slug)
    shadow = monitoring.read_parquet(
        config.metrics_output_dir / "prospective_monitoring_shadow_candidates.parquet"
    )
    shadow = shadow[shadow["protocol_name"].eq(protocol) & shadow["event_slug"].eq(slug)]
    if not expected or monitoring.forecast_snapshot_hash(rows, shadow) != expected:
        raise ValueError("Existing forecast snapshot is not immutable or valid")
    return True


def freeze_snapshot(
    config: DataConfig,
    protocol: str,
    slug: str,
    forecasts: pd.DataFrame,
    context: dict,
    policy: dict,
    target_features: pd.DataFrame,
) -> None:
    path = snapshot_path(config, protocol, slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = config.metrics_output_dir / "fp3_conditional_v1" / "policy_freeze.json"
    if decision.is_file():
        if signature(json.loads(decision.read_text())) != signature(policy):
            raise ValueError("Frozen FP3 policy/config conflict")
    else:
        with decision.open("x") as stream:
            json.dump(policy, stream, indent=2)
    if path.exists() or path.with_name("forecasts.parquet").exists():
        raise ValueError("FP3 snapshot already exists; refusing regeneration")
    forecasts.to_parquet(path.with_name("forecasts.parquet"), index=False)
    safe_features = target_features.drop(
        columns=[
            c
            for c in target_features
            if c.lower().startswith(("q_", "quali_")) or c in {"reached_q2", "reached_q3"}
        ]
    )
    safe_features.to_parquet(path.with_name("target_features.parquet"), index=False)
    snapshot = {
        **context,
        "forecast_id": str(forecasts["forecast_id"].iloc[0]),
        "forecast_created_at_utc": str(forecasts["forecast_created_at_utc"].iloc[0]),
        "forecast_file_sha256": file_hash(path.with_name("forecasts.parquet")),
        "target_features_file_sha256": file_hash(path.with_name("target_features.parquet")),
        "candidate_contract": policy["candidate_contract"],
    }
    with path.open("x") as stream:
        json.dump(snapshot, stream, indent=2)


def decorate_predictions(forecasts: pd.DataFrame, context: dict, policy: dict) -> pd.DataFrame:
    result = forecasts.copy()
    live = result["prediction_role"].eq("observed_live_policy")
    uniform = result[result["prediction_role"].eq("uniform_default_shadow")].set_index("driver")
    weighted = result[result["prediction_role"].eq("season_aware_weighted_candidate_shadow")]
    weighted = weighted.set_index("driver")
    chosen = weighted if context["selected_candidate"] == SEASON_AWARE else uniform
    for column in (
        "prediction_gap_sec",
        "source_identity",
        "temporal_weighting_policy",
        "training_row_count",
        "training_event_count",
        "training_event_keys",
        "training_seasons",
        "training_effective_sample_size",
    ):
        result.loc[live, column] = result.loc[live, "driver"].map(chosen[column])
    for column, value in context.items():
        result[column] = value
    result["uniform_prediction_gap_sec"] = result["driver"].map(uniform["prediction_gap_sec"])
    result["season_aware_prediction_gap_sec"] = result["driver"].map(weighted["prediction_gap_sec"])
    result["candidate_version"] = result["temporal_weighting_policy"].map(
        policy["candidate_contract"]["candidate_versions"]
    )
    result["candidate_selection_reason"] = "explicit_five_event_policy"
    result["candidate_eligible_under_frozen_gates"] = pd.NA
    result["selection_is_counterfactual"] = False
    result["current_season_prior_event_count"] = context["prior_current_season_eligible_events"]
    result["training_completed"] = result["prediction_gap_sec"].notna()
    return result
