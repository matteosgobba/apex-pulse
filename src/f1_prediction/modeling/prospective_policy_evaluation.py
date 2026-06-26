"""Prospective season-held-out champion policy evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig, ModelConfig
from f1_prediction.modeling.season_aware_validation import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    current_season_evidence_regime,
    paired_bootstrap_mean_ci,
)
from f1_prediction.utils.paths import ensure_directory

CHECKPOINT_ORDER: tuple[str, ...] = ("after_fp1", "after_fp2", "after_fp3")
POLICY_PROFILE_TO_MODE: dict[str, str] = {
    "static_baseline": "static",
    "guarded_baseline": "stabilized_nested_guarded",
    "season_aware_frozen": "season_aware_nested_guarded",
}
EVALUATION_TYPE = "prospective_season_holdout"
PREDICTION_TOLERANCE = 1e-9


@dataclass(frozen=True)
class FrozenPolicyProfile:
    """Frozen policy metadata used for prospective season holdout scoring."""

    profile_name: str
    champion_selection_mode: str
    uncertainty_method: str
    candidate_family: str | None
    candidate_model_name: str | None
    candidate_feature_group: str | None
    candidate_temporal_weighting_policy: str | None
    cold_start_threshold: int | None
    min_prior_folds: int | None
    min_prior_predictions: int | None
    improvement_margin_sec: float | None
    guardrail_settings: dict[str, object]
    static_fp3_family: str | None
    static_fp3_model_name: str | None
    static_fp3_feature_group: str | None
    random_state: int | None
    config_signature: str

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-ready representation."""
        payload = asdict(self)
        payload["policy_signature"] = policy_signature(payload)
        return payload


@dataclass(frozen=True)
class ProspectivePolicyEvaluationSummary:
    """Paths and status for prospective season-held-out evaluation."""

    status: str
    summary_path: Path
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    missing_inputs: tuple[str, ...]
    generation_issues: tuple[str, ...]


def create_prospective_policy_evaluation_report(
    config: DataConfig,
    model_config: ModelConfig,
    *,
    train_seasons: tuple[int, ...],
    test_season: int,
    policy_profiles: tuple[str, ...] = (
        "static_baseline",
        "guarded_baseline",
        "season_aware_frozen",
    ),
    uncertainty: str = "conformal_predicted_gap_bucket",
    min_events: int = 10,
    min_train_events: int = 5,
) -> ProspectivePolicyEvaluationSummary:
    """Evaluate frozen champion-policy profiles on a held-out test season."""
    metrics_dir = config.metrics_output_dir
    figures_dir = metrics_dir.parent / "figures"
    ensure_directory(metrics_dir)
    ensure_directory(figures_dir)

    split_id = prospective_split_id(train_seasons, test_season)
    profiles = build_frozen_policy_profiles(
        model_config,
        profile_names=policy_profiles,
        uncertainty=uncertainty,
    )
    artifacts, missing_inputs = load_policy_artifacts(metrics_dir, policy_profiles)
    generation_issues: list[str] = []

    predictions = build_prospective_predictions(
        artifacts=artifacts,
        profiles=profiles,
        train_seasons=train_seasons,
        test_season=test_season,
        min_events=min_events,
        min_train_events=min_train_events,
    )
    selections = build_prospective_selection_log(
        artifacts=artifacts,
        profiles=profiles,
        train_seasons=train_seasons,
        test_season=test_season,
        min_events=min_events,
        min_train_events=min_train_events,
    )
    checkpoint = attach_selection_rates(
        build_checkpoint_comparison(predictions),
        selections,
    )
    event = build_event_comparison(predictions, selections)
    cold_start = build_cold_start_comparison(event)
    leakage = build_leakage_audit(
        selections=selections,
        profiles=profiles,
        train_seasons=train_seasons,
        test_season=test_season,
    )

    prediction_path = metrics_dir / f"{split_id}_predictions.parquet"
    selection_path = metrics_dir / f"{split_id}_selection_log.parquet"
    split_summary_path = metrics_dir / f"{split_id}_summary.json"
    predictions.to_parquet(prediction_path, index=False)
    selections.to_parquet(selection_path, index=False)

    table_paths = (
        metrics_dir / "prospective_policy_checkpoint_comparison.csv",
        metrics_dir / "prospective_policy_event_comparison.csv",
        metrics_dir / "prospective_policy_selection_log.csv",
        metrics_dir / "prospective_policy_cold_start_comparison.csv",
        metrics_dir / "prospective_policy_leakage_audit.csv",
    )
    combined_tables = merge_with_existing_prospective_tables(
        metrics_dir=metrics_dir,
        split_id=split_id,
        checkpoint=checkpoint,
        event=event,
        selections=selections,
        cold_start=cold_start,
        leakage=leakage,
    )
    for path, frame in zip(table_paths, combined_tables, strict=True):
        frame.to_csv(path, index=False)

    figure_paths, figure_issues = generate_prospective_figures(
        figures_dir=figures_dir,
        checkpoint=combined_tables[0],
        event=combined_tables[1],
        selections=combined_tables[2],
        cold_start=combined_tables[3],
    )
    generation_issues.extend(figure_issues)

    split_summary = build_summary_payload(
        split_id=split_id,
        train_seasons=train_seasons,
        test_season=test_season,
        profiles=profiles,
        checkpoint=checkpoint,
        event=event,
        selections=selections,
        cold_start=cold_start,
        leakage=leakage,
        missing_inputs=missing_inputs,
        generation_issues=generation_issues,
        prediction_path=prediction_path,
        selection_path=selection_path,
    )
    write_json(split_summary_path, split_summary)
    summary_path = metrics_dir / "prospective_policy_summary.json"
    summary_payload = merge_prospective_summary(summary_path, split_summary)
    write_json(summary_path, summary_payload)

    return ProspectivePolicyEvaluationSummary(
        status=str(summary_payload.get("status", "complete")),
        summary_path=summary_path,
        table_paths=(*table_paths, split_summary_path, prediction_path, selection_path),
        figure_paths=tuple(figure_paths),
        missing_inputs=tuple(missing_inputs),
        generation_issues=tuple(generation_issues),
    )


def build_frozen_policy_profiles(
    model_config: ModelConfig,
    *,
    profile_names: tuple[str, ...],
    uncertainty: str,
) -> dict[str, FrozenPolicyProfile]:
    """Build deterministic frozen policy profiles from current configuration."""
    config_payload = config_signature_payload(model_config)
    config_hash = signature(config_payload)
    static_fp3 = model_config.champion_policy.static.get("after_fp3")
    season_aware = model_config.champion_policy.season_aware_nested_guarded
    guarded = model_config.champion_policy.stabilized_nested_guarded
    profiles: dict[str, FrozenPolicyProfile] = {}
    for name in profile_names:
        if name not in POLICY_PROFILE_TO_MODE:
            raise ValueError(f"Unsupported prospective policy profile: {name}")
        candidate = season_aware.required_candidate if name == "season_aware_frozen" else None
        profiles[name] = FrozenPolicyProfile(
            profile_name=name,
            champion_selection_mode=POLICY_PROFILE_TO_MODE[name],
            uncertainty_method=uncertainty,
            candidate_family=candidate.family if candidate else None,
            candidate_model_name=candidate.model_name if candidate else None,
            candidate_feature_group=candidate.feature_group if candidate else None,
            candidate_temporal_weighting_policy=(
                candidate.temporal_weighting_policy if candidate else None
            ),
            cold_start_threshold=(
                season_aware.min_current_season_prior_events
                if name == "season_aware_frozen"
                else None
            ),
            min_prior_folds=(
                season_aware.min_prior_candidate_folds if name == "season_aware_frozen" else None
            ),
            min_prior_predictions=(
                season_aware.min_prior_candidate_predictions
                if name == "season_aware_frozen"
                else None
            ),
            improvement_margin_sec=(
                season_aware.improvement_margin_sec if name == "season_aware_frozen" else None
            ),
            guardrail_settings={
                "base_mode": guarded.base_mode,
                "fp3_no_baseline_switch": guarded.fp3_no_baseline_switch,
                "guarded_checkpoint": guarded.guarded_checkpoint,
                "guarded_default_family": guarded.guarded_default_family,
                "guarded_default_model_name": guarded.guarded_default_model_name,
                "guarded_default_feature_group": guarded.guarded_default_feature_group,
            },
            static_fp3_family=static_fp3.family if static_fp3 else None,
            static_fp3_model_name=static_fp3.model_name if static_fp3 else None,
            static_fp3_feature_group=static_fp3.feature_group if static_fp3 else None,
            random_state=model_config.random_state,
            config_signature=config_hash,
        )
    return profiles


def policy_signature(payload: dict[str, object]) -> str:
    """Hash a frozen policy payload without depending on dictionary order."""
    source = {key: value for key, value in payload.items() if key != "policy_signature"}
    return signature(source)


def signature(payload: object) -> str:
    """Return a stable short SHA-256 signature for JSON-compatible payloads."""
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()[:16]


def prospective_split_id(train_seasons: tuple[int, ...], test_season: int) -> str:
    """Return the path-safe split identifier."""
    train = "_".join(str(season) for season in train_seasons)
    return f"prospective_train_{train}_test_{test_season}"


def load_policy_artifacts(
    metrics_dir: Path,
    policy_profiles: tuple[str, ...],
) -> tuple[dict[str, dict[str, pd.DataFrame]], list[str]]:
    """Load mode-specific champion prediction and selection artifacts."""
    artifacts: dict[str, dict[str, pd.DataFrame]] = {}
    missing: list[str] = []
    for profile in policy_profiles:
        mode = POLICY_PROFILE_TO_MODE.get(profile)
        if not mode:
            continue
        prediction_path = metrics_dir / f"champion_{mode}_predictions.parquet"
        selection_path = metrics_dir / f"champion_{mode}_selection.parquet"
        prediction = read_parquet_if_exists(prediction_path)
        selection = read_parquet_if_exists(selection_path)
        if prediction is None:
            missing.append(prediction_path.name)
            prediction = pd.DataFrame()
        if selection is None:
            missing.append(selection_path.name)
            selection = pd.DataFrame()
        artifacts[profile] = {"predictions": prediction, "selection": selection}
    return artifacts, missing


def build_prospective_predictions(
    *,
    artifacts: dict[str, dict[str, pd.DataFrame]],
    profiles: dict[str, FrozenPolicyProfile],
    train_seasons: tuple[int, ...],
    test_season: int,
    min_events: int,
    min_train_events: int,
) -> pd.DataFrame:
    """Create prediction rows for requested held-out season from saved artifacts."""
    rows: list[pd.DataFrame] = []
    split_id = prospective_split_id(train_seasons, test_season)
    for profile_name, profile in profiles.items():
        frame = artifacts.get(profile_name, {}).get("predictions", pd.DataFrame()).copy()
        if frame.empty:
            continue
        if "season" not in frame:
            continue
        frame = frame[pd.to_numeric(frame["season"], errors="coerce").eq(test_season)].copy()
        if frame.empty:
            continue
        frame["evaluation_type"] = EVALUATION_TYPE
        frame["prospective_split"] = split_id
        frame["train_seasons"] = ",".join(str(season) for season in train_seasons)
        frame["test_season"] = test_season
        frame["policy_profile"] = profile_name
        profile_payload = profile.to_payload()
        frame["policy_signature"] = profile_payload["policy_signature"]
        frame["policy_frozen_at"] = "config_snapshot"
        frame["allowed_history_scope"] = frame.apply(
            lambda row: allowed_history_scope(row, train_seasons=train_seasons),
            axis=1,
        )
        frame["prospective_event_id"] = frame.apply(event_key_from_row, axis=1)
        frame["source_identity"] = frame.apply(source_identity_from_row, axis=1)
        frame["min_events"] = min_events
        frame["min_train_events"] = min_train_events
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_prospective_selection_log(
    *,
    artifacts: dict[str, dict[str, pd.DataFrame]],
    profiles: dict[str, FrozenPolicyProfile],
    train_seasons: tuple[int, ...],
    test_season: int,
    min_events: int,
    min_train_events: int,
) -> pd.DataFrame:
    """Create one selection metadata row per profile/fold/checkpoint."""
    rows: list[pd.DataFrame] = []
    split_id = prospective_split_id(train_seasons, test_season)
    for profile_name, profile in profiles.items():
        selection = artifacts.get(profile_name, {}).get("selection", pd.DataFrame()).copy()
        if selection.empty:
            predictions = artifacts.get(profile_name, {}).get("predictions", pd.DataFrame())
            selection = selection_from_predictions(predictions)
        if selection.empty or "fold_id" not in selection:
            continue
        selection = attach_event_metadata(
            selection,
            artifacts.get(profile_name, {}).get("predictions", pd.DataFrame()),
        )
        selection = selection[pd.to_numeric(selection["season"], errors="coerce").eq(test_season)]
        if selection.empty:
            continue
        profile_payload = profile.to_payload()
        selection["evaluation_type"] = EVALUATION_TYPE
        selection["prospective_split"] = split_id
        selection["train_seasons"] = ",".join(str(season) for season in train_seasons)
        selection["test_season"] = test_season
        selection["policy_profile"] = profile_name
        selection["policy_signature"] = profile_payload["policy_signature"]
        selection["policy_frozen_at"] = "config_snapshot"
        selection["allowed_history_scope"] = selection.apply(
            lambda row: allowed_history_scope(row, train_seasons=train_seasons),
            axis=1,
        )
        selection["current_test_season_prior_event_count"] = selection.apply(
            current_test_season_prior_event_count,
            axis=1,
        )
        selection["cold_start_regime"] = selection["current_test_season_prior_event_count"].map(
            current_season_evidence_regime
        )
        selection["min_events"] = min_events
        selection["min_train_events"] = min_train_events
        selection["candidate_selected"] = (
            selection.get("season_aware_selected", False).fillna(False).astype(bool)
            if "season_aware_selected" in selection
            else False
        )
        selection["candidate_selection_reason"] = selection.apply(selection_reason, axis=1)
        rows.append(selection)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_checkpoint_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize metrics by split/profile/checkpoint."""
    columns = [
        "prospective_split",
        "train_seasons",
        "test_season",
        "policy_profile",
        "checkpoint",
        "rows",
        "events",
        "mae_gap_sec",
        "rmse_gap_sec",
        "median_abs_error_gap_sec",
        "mean_position_error",
        "interval_availability_rate",
        "interval_coverage",
        "mean_interval_width_sec",
        "candidate_selection_rate",
        "fp3_delta_vs_static_baseline_sec",
        "fp3_delta_vs_guarded_baseline_sec",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    frame = predictions.copy()
    frame["abs_error_gap_sec"] = abs_error(frame)
    frame["squared_error_gap_sec"] = frame["abs_error_gap_sec"] ** 2
    frame["position_error"] = position_error(frame)
    rows: list[dict[str, object]] = []
    group_cols = [
        "prospective_split",
        "train_seasons",
        "test_season",
        "policy_profile",
        "checkpoint",
    ]
    for keys, group in frame.groupby(group_cols, dropna=False, sort=False):
        split, train, test_season, profile, checkpoint = keys
        interval = interval_metrics(group)
        rows.append(
            {
                "prospective_split": split,
                "train_seasons": train,
                "test_season": int(test_season),
                "policy_profile": profile,
                "checkpoint": checkpoint,
                "rows": int(len(group)),
                "events": int(group["prospective_event_id"].nunique()),
                "mae_gap_sec": mean_or_none(group["abs_error_gap_sec"]),
                "rmse_gap_sec": rmse_or_none(group["squared_error_gap_sec"]),
                "median_abs_error_gap_sec": median_or_none(group["abs_error_gap_sec"]),
                "mean_position_error": mean_or_none(group["position_error"]),
                **interval,
                "candidate_selection_rate": None,
                "fp3_delta_vs_static_baseline_sec": None,
                "fp3_delta_vs_guarded_baseline_sec": None,
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    result = add_checkpoint_deltas(result)
    return result


def attach_selection_rates(checkpoint: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    """Attach candidate selection rates by split/profile/checkpoint."""
    if checkpoint.empty or selections.empty:
        return checkpoint
    result = checkpoint.copy()
    group_cols = ["prospective_split", "policy_profile", "checkpoint"]
    if not set(group_cols) <= set(selections.columns):
        return result
    rates = (
        selections.groupby(group_cols, dropna=False)["candidate_selected"]
        .mean()
        .reset_index(name="candidate_selection_rate")
    )
    merged = result.drop(columns=["candidate_selection_rate"], errors="ignore").merge(
        rates,
        on=group_cols,
        how="left",
    )
    return merged[result.columns]


def build_event_comparison(
    predictions: pd.DataFrame,
    selections: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize metrics by split/profile/checkpoint/event."""
    columns = [
        "prospective_split",
        "train_seasons",
        "test_season",
        "policy_profile",
        "season",
        "event",
        "event_slug",
        "fold_id",
        "checkpoint",
        "cold_start_regime",
        "current_test_season_prior_event_count",
        "rows",
        "mae_gap_sec",
        "rmse_gap_sec",
        "median_abs_error_gap_sec",
        "mean_position_error",
        "interval_availability_rate",
        "interval_coverage",
        "mean_interval_width_sec",
        "season_aware_selected",
        "candidate_selection_reason",
        "fp3_delta_vs_static_baseline_sec",
        "fp3_delta_vs_guarded_baseline_sec",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    frame = predictions.copy()
    frame["abs_error_gap_sec"] = abs_error(frame)
    frame["squared_error_gap_sec"] = frame["abs_error_gap_sec"] ** 2
    frame["position_error"] = position_error(frame)
    selection_lookup = selection_metadata_lookup(selections)
    rows: list[dict[str, object]] = []
    group_cols = [
        "prospective_split",
        "train_seasons",
        "test_season",
        "policy_profile",
        "season",
        "event",
        "event_slug",
        "fold_id",
        "checkpoint",
    ]
    for keys, group in frame.groupby(group_cols, dropna=False, sort=False):
        (
            split,
            train,
            test_season,
            profile,
            season,
            event,
            event_slug,
            fold_id,
            checkpoint,
        ) = keys
        meta = selection_lookup.get((str(profile), int(fold_id), str(checkpoint)), {})
        prior_count = meta.get(
            "current_test_season_prior_event_count",
            current_test_season_prior_event_count(group.iloc[0]),
        )
        interval = interval_metrics(group)
        rows.append(
            {
                "prospective_split": split,
                "train_seasons": train,
                "test_season": int(test_season),
                "policy_profile": profile,
                "season": int(season),
                "event": event,
                "event_slug": event_slug,
                "fold_id": int(fold_id),
                "checkpoint": checkpoint,
                "cold_start_regime": current_season_evidence_regime(prior_count),
                "current_test_season_prior_event_count": int(prior_count),
                "rows": int(len(group)),
                "mae_gap_sec": mean_or_none(group["abs_error_gap_sec"]),
                "rmse_gap_sec": rmse_or_none(group["squared_error_gap_sec"]),
                "median_abs_error_gap_sec": median_or_none(group["abs_error_gap_sec"]),
                "mean_position_error": mean_or_none(group["position_error"]),
                **interval,
                "season_aware_selected": bool(meta.get("season_aware_selected", False)),
                "candidate_selection_reason": meta.get("candidate_selection_reason"),
                "fp3_delta_vs_static_baseline_sec": None,
                "fp3_delta_vs_guarded_baseline_sec": None,
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    return add_event_deltas(result)


def build_cold_start_comparison(event: pd.DataFrame) -> pd.DataFrame:
    """Summarize event-level metrics by evidence regime."""
    columns = [
        "prospective_split",
        "train_seasons",
        "test_season",
        "policy_profile",
        "checkpoint",
        "cold_start_regime",
        "events",
        "rows",
        "mae_gap_sec",
        "rmse_gap_sec",
        "median_abs_error_gap_sec",
        "mean_position_error",
        "mean_interval_width_sec",
        "interval_coverage",
        "candidate_selection_rate",
        "fp3_delta_vs_static_baseline_sec",
        "fp3_delta_vs_guarded_baseline_sec",
    ]
    if event.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    group_cols = [
        "prospective_split",
        "train_seasons",
        "test_season",
        "policy_profile",
        "checkpoint",
        "cold_start_regime",
    ]
    for keys, group in event.groupby(group_cols, dropna=False, sort=False):
        split, train, test_season, profile, checkpoint, regime = keys
        rows.append(
            {
                "prospective_split": split,
                "train_seasons": train,
                "test_season": int(test_season),
                "policy_profile": profile,
                "checkpoint": checkpoint,
                "cold_start_regime": regime,
                "events": int(len(group)),
                "rows": int(group["rows"].sum()),
                "mae_gap_sec": weighted_event_mean(group, "mae_gap_sec"),
                "rmse_gap_sec": weighted_event_mean(group, "rmse_gap_sec"),
                "median_abs_error_gap_sec": mean_or_none(group["median_abs_error_gap_sec"]),
                "mean_position_error": mean_or_none(group["mean_position_error"]),
                "mean_interval_width_sec": mean_or_none(group["mean_interval_width_sec"]),
                "interval_coverage": mean_or_none(group["interval_coverage"]),
                "candidate_selection_rate": mean_or_none(
                    group["season_aware_selected"].astype(float)
                ),
                "fp3_delta_vs_static_baseline_sec": mean_or_none(
                    group["fp3_delta_vs_static_baseline_sec"]
                ),
                "fp3_delta_vs_guarded_baseline_sec": mean_or_none(
                    group["fp3_delta_vs_guarded_baseline_sec"]
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_leakage_audit(
    *,
    selections: pd.DataFrame,
    profiles: dict[str, FrozenPolicyProfile],
    train_seasons: tuple[int, ...],
    test_season: int,
) -> pd.DataFrame:
    """Audit that frozen policy decisions use no future held-out-season evidence."""
    columns = [
        "test_season",
        "event",
        "checkpoint",
        "policy_profile",
        "maximum_allowed_history_event",
        "history_event_keys_used",
        "future_test_season_event_used",
        "future_event_used_anywhere",
        "test_season_outcome_used_to_set_thresholds",
        "frozen_policy_signature_match",
        "history_scope_valid",
        "leakage_status",
        "leakage_reason",
    ]
    if selections.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    event_fold_lookup = event_fold_lookup_from_selection(selections)
    for _, row in selections.iterrows():
        profile_name = str(row.get("policy_profile"))
        profile = profiles.get(profile_name)
        expected_signature = profile.to_payload()["policy_signature"] if profile else None
        allowed = allowed_history_event_keys(
            row,
            train_seasons=train_seasons,
            event_fold_lookup=event_fold_lookup,
        )
        used = parse_event_key_list(row.get("metric_scope_event_keys"))
        if not used:
            used = parse_event_key_list(row.get("selection_source_events"))
        future_test = [
            key for key in used if parse_event_key(key)[0] == test_season and key not in allowed
        ]
        future_any = [
            key
            for key in used
            if parse_event_key(key)[0] > test_season
            or (parse_event_key(key)[0] == test_season and key not in allowed)
        ]
        signature_match = str(row.get("policy_signature")) == str(expected_signature)
        history_valid = not future_any
        threshold_used = False
        status = "valid" if history_valid and signature_match and not threshold_used else "invalid"
        reason = "no_future_history_used"
        if future_any:
            reason = "future_held_out_event_used"
        elif not signature_match:
            reason = "frozen_policy_signature_mismatch"
        rows.append(
            {
                "test_season": test_season,
                "event": row.get("event") or row.get("test_event"),
                "checkpoint": row.get("checkpoint"),
                "policy_profile": profile_name,
                "maximum_allowed_history_event": max(allowed, default=None),
                "history_event_keys_used": json.dumps(used),
                "future_test_season_event_used": bool(future_test),
                "future_event_used_anywhere": bool(future_any),
                "test_season_outcome_used_to_set_thresholds": threshold_used,
                "frozen_policy_signature_match": signature_match,
                "history_scope_valid": history_valid,
                "leakage_status": status,
                "leakage_reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def merge_with_existing_prospective_tables(
    *,
    metrics_dir: Path,
    split_id: str,
    checkpoint: pd.DataFrame,
    event: pd.DataFrame,
    selections: pd.DataFrame,
    cold_start: pd.DataFrame,
    leakage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge current split tables into cumulative prospective report tables."""
    items = [
        ("prospective_policy_checkpoint_comparison.csv", checkpoint),
        ("prospective_policy_event_comparison.csv", event),
        ("prospective_policy_selection_log.csv", selections),
        ("prospective_policy_cold_start_comparison.csv", cold_start),
        ("prospective_policy_leakage_audit.csv", leakage),
    ]
    merged: list[pd.DataFrame] = []
    for filename, frame in items:
        path = metrics_dir / filename
        existing = pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=frame.columns)
        if not existing.empty and "prospective_split" in existing.columns:
            existing = existing[existing["prospective_split"].astype(str).ne(split_id)].copy()
        merged_frame = pd.concat([existing, frame], ignore_index=True)
        merged.append(merged_frame)
    return tuple(merged)  # type: ignore[return-value]


def build_summary_payload(
    *,
    split_id: str,
    train_seasons: tuple[int, ...],
    test_season: int,
    profiles: dict[str, FrozenPolicyProfile],
    checkpoint: pd.DataFrame,
    event: pd.DataFrame,
    selections: pd.DataFrame,
    cold_start: pd.DataFrame,
    leakage: pd.DataFrame,
    missing_inputs: list[str],
    generation_issues: list[str],
    prediction_path: Path,
    selection_path: Path,
) -> dict[str, object]:
    """Build summary JSON for one prospective split."""
    fp3 = (
        checkpoint[checkpoint["checkpoint"].eq("after_fp3")].copy()
        if not checkpoint.empty
        else checkpoint
    )
    bootstrap = bootstrap_summary(event)
    recommendation = prospective_recommendation(checkpoint, leakage)
    return {
        "status": "complete" if not missing_inputs and not checkpoint.empty else "partial",
        "evaluation_type": EVALUATION_TYPE,
        "prospective_split": split_id,
        "train_seasons": list(train_seasons),
        "test_season": test_season,
        "policy_profiles": {name: profile.to_payload() for name, profile in profiles.items()},
        "frozen_policy_profiles": {
            name: profile.to_payload() for name, profile in profiles.items()
        },
        "frozen_policy_principle": (
            "Profiles are configured before scoring the held-out season; test-season outcomes "
            "are not used to change thresholds, margins, candidates, or policy choice."
        ),
        "checkpoint_summary": records_for_json(checkpoint),
        "fp3_summary": records_for_json(fp3),
        "candidate_selection_summary": candidate_selection_summary(selections),
        "cold_start_summary": records_for_json(cold_start),
        "leakage_audit_summary": leakage_summary(leakage),
        "bootstrap_confidence_intervals": bootstrap,
        "recommendation": recommendation,
        "main_findings": main_findings(checkpoint, leakage, recommendation),
        "artifact_paths": {
            "predictions": str(prediction_path),
            "selection_log": str(selection_path),
            "checkpoint_comparison": "reports/metrics/prospective_policy_checkpoint_comparison.csv",
            "event_comparison": "reports/metrics/prospective_policy_event_comparison.csv",
            "selection_log_csv": "reports/metrics/prospective_policy_selection_log.csv",
            "cold_start_comparison": "reports/metrics/prospective_policy_cold_start_comparison.csv",
            "leakage_audit": "reports/metrics/prospective_policy_leakage_audit.csv",
        },
        "missing_inputs": missing_inputs,
        "generation_issues": generation_issues,
        "generated_at": utc_now(),
    }


def merge_prospective_summary(
    summary_path: Path,
    split_summary: dict[str, object],
) -> dict[str, object]:
    """Merge a split summary into the cumulative prospective summary."""
    existing = read_json_if_exists(summary_path) or {}
    splits = list(existing.get("splits", [])) if isinstance(existing.get("splits"), list) else []
    split_id = str(split_summary["prospective_split"])
    splits = [item for item in splits if item.get("prospective_split") != split_id]
    splits.append(split_summary)
    recommendation = aggregate_recommendation(splits)
    return {
        "status": "complete" if splits else "partial",
        "evaluation_type": EVALUATION_TYPE,
        "splits": splits,
        "prospective_splits_available": [item["prospective_split"] for item in splits],
        "frozen_policy_profiles_available": sorted(
            {name for item in splits for name in (item.get("policy_profiles") or {}).keys()}
        ),
        "leakage_audit_result": aggregate_leakage_result(splits),
        "recommendation": recommendation,
        "main_findings": aggregate_main_findings(splits, recommendation),
        "generated_at": utc_now(),
    }


def generate_prospective_figures(
    *,
    figures_dir: Path,
    checkpoint: pd.DataFrame,
    event: pd.DataFrame,
    selections: pd.DataFrame,
    cold_start: pd.DataFrame,
) -> tuple[list[Path], list[str]]:
    """Generate simple static figures for prospective policy evaluation."""
    ensure_directory(figures_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(figures_dir.parent / ".matplotlib-cache"))
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    figure_specs = [
        (
            "prospective_policy_fp3_mae_by_test_season.png",
            lambda path: plot_fp3_mae_by_test_season(plt, checkpoint, path),
        ),
        (
            "prospective_policy_fp3_delta_vs_static.png",
            lambda path: plot_fp3_delta(plt, checkpoint, path, "fp3_delta_vs_static_baseline_sec"),
        ),
        (
            "prospective_policy_fp3_delta_vs_guarded.png",
            lambda path: plot_fp3_delta(plt, checkpoint, path, "fp3_delta_vs_guarded_baseline_sec"),
        ),
        (
            "prospective_policy_selection_rate_by_regime.png",
            lambda path: plot_selection_rate_by_regime(plt, selections, path),
        ),
        (
            "prospective_policy_cold_start_performance.png",
            lambda path: plot_cold_start_performance(plt, cold_start, path),
        ),
        (
            "prospective_policy_interval_coverage_width.png",
            lambda path: plot_interval_coverage_width(plt, checkpoint, path),
        ),
    ]
    paths: list[Path] = []
    issues: list[str] = []
    for filename, callback in figure_specs:
        path = figures_dir / filename
        try:
            if callback(path):
                paths.append(path)
            else:
                issues.append(f"skipped figure {filename}: insufficient data")
        except Exception as exc:  # pragma: no cover - defensive reporting
            issues.append(f"skipped figure {filename}: {exc}")
        finally:
            plt.close("all")
    return paths, issues


def plot_fp3_mae_by_test_season(plt: Any, checkpoint: pd.DataFrame, path: Path) -> bool:
    frame = checkpoint[checkpoint["checkpoint"].eq("after_fp3")].copy()
    if frame.empty:
        return False
    frame["label"] = frame["test_season"].astype(str) + " " + frame["policy_profile"].astype(str)
    ax = frame.plot.bar(x="label", y="mae_gap_sec", legend=False, color="#356f9f", figsize=(9, 4))
    ax.set_ylabel("MAE gap (sec)")
    ax.set_xlabel("")
    ax.set_title("Prospective FP3 MAE by held-out season")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    return True


def plot_fp3_delta(plt: Any, checkpoint: pd.DataFrame, path: Path, column: str) -> bool:
    frame = checkpoint[checkpoint["checkpoint"].eq("after_fp3")].dropna(subset=[column]).copy()
    if frame.empty:
        return False
    frame["label"] = frame["test_season"].astype(str) + " " + frame["policy_profile"].astype(str)
    colors = ["#3b8f60" if value <= 0 else "#b44d4d" for value in frame[column]]
    ax = frame.plot.bar(x="label", y=column, legend=False, color=colors, figsize=(9, 4))
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Delta MAE (sec)")
    ax.set_xlabel("")
    ax.set_title(column.replace("_", " "))
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    return True


def plot_selection_rate_by_regime(plt: Any, selections: pd.DataFrame, path: Path) -> bool:
    if selections.empty or "cold_start_regime" not in selections:
        return False
    frame = selections[
        selections["checkpoint"].eq("after_fp3")
        & selections["policy_profile"].eq("season_aware_frozen")
    ].copy()
    if frame.empty:
        return False
    table = frame.pivot_table(
        index="cold_start_regime",
        columns="test_season",
        values="candidate_selected",
        aggfunc="mean",
        fill_value=0.0,
    )
    if table.empty:
        return False
    table = table.reindex(["cold_start", "early_season", "established_season"]).dropna(how="all")
    ax = table.plot.bar(
        figsize=(8, 4),
        color=["#356f9f", "#d98c3f", "#5f7f4f"][: len(table.columns)],
    )
    ax.set_ylabel("Selection rate")
    ax.set_xlabel("")
    ax.set_title("Season-aware candidate selection rate by regime")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    return True


def plot_cold_start_performance(plt: Any, cold_start: pd.DataFrame, path: Path) -> bool:
    frame = cold_start[cold_start["checkpoint"].eq("after_fp3")].copy()
    if frame.empty:
        return False
    frame["label"] = (
        frame["test_season"].astype(str)
        + " "
        + frame["policy_profile"].astype(str)
        + " "
        + frame["cold_start_regime"].astype(str)
    )
    ax = frame.plot.bar(x="label", y="mae_gap_sec", legend=False, color="#6c6c9d", figsize=(10, 4))
    ax.set_ylabel("MAE gap (sec)")
    ax.set_xlabel("")
    ax.set_title("Prospective FP3 performance by evidence regime")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    return True


def plot_interval_coverage_width(plt: Any, checkpoint: pd.DataFrame, path: Path) -> bool:
    frame = checkpoint[checkpoint["checkpoint"].eq("after_fp3")].copy()
    if frame.empty or frame["interval_coverage"].isna().all():
        return False
    frame["label"] = frame["test_season"].astype(str) + " " + frame["policy_profile"].astype(str)
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()
    ax1.bar(frame["label"], frame["interval_coverage"], color="#356f9f", alpha=0.75)
    ax2.plot(frame["label"], frame["mean_interval_width_sec"], color="#b44d4d", marker="o")
    ax1.set_ylabel("Coverage")
    ax2.set_ylabel("Mean width (sec)")
    ax1.set_xlabel("")
    ax1.set_title("Prospective FP3 interval coverage and width")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    return True


def add_checkpoint_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    for split in result["prospective_split"].dropna().unique():
        for checkpoint in result["checkpoint"].dropna().unique():
            mask = result["prospective_split"].eq(split) & result["checkpoint"].eq(checkpoint)
            static = metric_for_profile(result[mask], "static_baseline")
            guarded = metric_for_profile(result[mask], "guarded_baseline")
            if checkpoint == "after_fp3":
                if static is not None:
                    result.loc[mask, "fp3_delta_vs_static_baseline_sec"] = (
                        result.loc[mask, "mae_gap_sec"].astype(float) - static
                    )
                if guarded is not None:
                    result.loc[mask, "fp3_delta_vs_guarded_baseline_sec"] = (
                        result.loc[mask, "mae_gap_sec"].astype(float) - guarded
                    )
    return result


def add_event_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    keys = ["prospective_split", "fold_id", "checkpoint"]
    static_lookup = (
        result[result["policy_profile"].eq("static_baseline")]
        .set_index(keys)["mae_gap_sec"]
        .to_dict()
    )
    guarded_lookup = (
        result[result["policy_profile"].eq("guarded_baseline")]
        .set_index(keys)["mae_gap_sec"]
        .to_dict()
    )
    for index, row in result.iterrows():
        if row["checkpoint"] != "after_fp3":
            continue
        key = (row["prospective_split"], row["fold_id"], row["checkpoint"])
        if key in static_lookup and pd.notna(row["mae_gap_sec"]):
            result.at[index, "fp3_delta_vs_static_baseline_sec"] = float(
                row["mae_gap_sec"]
            ) - float(static_lookup[key])
        if key in guarded_lookup and pd.notna(row["mae_gap_sec"]):
            result.at[index, "fp3_delta_vs_guarded_baseline_sec"] = float(
                row["mae_gap_sec"]
            ) - float(guarded_lookup[key])
    return result


def bootstrap_summary(event: pd.DataFrame) -> dict[str, object]:
    if event.empty:
        return {}
    fp3 = event[event["checkpoint"].eq("after_fp3")].copy()
    rows: dict[str, object] = {}
    for column, label in (
        ("fp3_delta_vs_static_baseline_sec", "season_aware_frozen_vs_static_baseline"),
        ("fp3_delta_vs_guarded_baseline_sec", "season_aware_frozen_vs_guarded_baseline"),
    ):
        frame = fp3[fp3["policy_profile"].eq("season_aware_frozen")]
        rows[label] = paired_bootstrap_mean_ci(
            frame[column] if column in frame else [],
            seed=BOOTSTRAP_SEED,
            iterations=BOOTSTRAP_ITERATIONS,
        )
    return rows


def prospective_recommendation(checkpoint: pd.DataFrame, leakage: pd.DataFrame) -> str:
    if checkpoint.empty:
        return "season_aware_candidate_requires_more_evidence"
    if not leakage.empty and leakage["leakage_status"].astype(str).ne("valid").any():
        return "season_aware_candidate_requires_more_evidence"
    fp3 = checkpoint[checkpoint["checkpoint"].eq("after_fp3")].copy()
    season_aware = fp3[fp3["policy_profile"].eq("season_aware_frozen")]
    if season_aware.empty:
        return "season_aware_candidate_requires_more_evidence"
    static_deltas = pd.to_numeric(
        season_aware["fp3_delta_vs_static_baseline_sec"],
        errors="coerce",
    ).dropna()
    guarded_deltas = pd.to_numeric(
        season_aware["fp3_delta_vs_guarded_baseline_sec"],
        errors="coerce",
    ).dropna()
    if (
        not static_deltas.empty
        and not guarded_deltas.empty
        and (static_deltas < 0).all()
        and (guarded_deltas < 0).all()
    ):
        return "season_aware_candidate_eligible_for_default_consideration"
    guarded = fp3[fp3["policy_profile"].eq("guarded_baseline")]
    static = fp3[fp3["policy_profile"].eq("static_baseline")]
    if not guarded.empty and not static.empty:
        guarded_mae = float(guarded["mae_gap_sec"].iloc[0])
        static_mae = float(static["mae_gap_sec"].iloc[0])
        if guarded_mae <= static_mae:
            return "retain_guarded_policy"
    return "retain_static_policy"


def main_findings(
    checkpoint: pd.DataFrame,
    leakage: pd.DataFrame,
    recommendation: str,
) -> list[str]:
    findings: list[str] = []
    if not leakage.empty and leakage["leakage_status"].astype(str).eq("valid").all():
        findings.append("Prospective leakage audit found no future held-out-season events.")
    fp3 = (
        checkpoint[checkpoint["checkpoint"].eq("after_fp3")] if not checkpoint.empty else checkpoint
    )
    season_aware = fp3[fp3["policy_profile"].eq("season_aware_frozen")]
    if not season_aware.empty:
        static_delta = season_aware["fp3_delta_vs_static_baseline_sec"].iloc[0]
        guarded_delta = season_aware["fp3_delta_vs_guarded_baseline_sec"].iloc[0]
        findings.append(
            "Season-aware frozen FP3 delta was "
            f"{format_signed(static_delta)} versus static and {format_signed(guarded_delta)} "
            "versus guarded on this held-out split."
        )
    findings.append(f"Conservative recommendation: {recommendation}.")
    return findings


def aggregate_main_findings(splits: list[dict[str, object]], recommendation: str) -> list[str]:
    findings: list[str] = []
    valid = aggregate_leakage_result(splits)
    if valid.get("all_splits_valid"):
        findings.append("All available prospective leakage audits are valid.")
    for item in splits:
        for finding in item.get("main_findings", []):
            findings.append(str(finding))
    findings.append(f"Aggregate conservative recommendation: {recommendation}.")
    return findings


def aggregate_recommendation(splits: list[dict[str, object]]) -> str:
    if not splits:
        return "season_aware_candidate_requires_more_evidence"
    if not aggregate_leakage_result(splits).get("all_splits_valid"):
        return "season_aware_candidate_requires_more_evidence"
    recommendations = [str(item.get("recommendation")) for item in splits]
    if recommendations and all(
        value == "season_aware_candidate_eligible_for_default_consideration"
        for value in recommendations
    ):
        return "season_aware_candidate_eligible_for_default_consideration"
    if "retain_guarded_policy" in recommendations:
        return "retain_guarded_policy"
    if "retain_static_policy" in recommendations:
        return "retain_static_policy"
    return "season_aware_candidate_requires_more_evidence"


def aggregate_leakage_result(splits: list[dict[str, object]]) -> dict[str, object]:
    statuses = [
        item.get("leakage_audit_summary", {}).get("all_rows_valid")
        for item in splits
        if isinstance(item.get("leakage_audit_summary"), dict)
    ]
    return {
        "splits_checked": len(statuses),
        "all_splits_valid": bool(statuses) and all(bool(value) for value in statuses),
    }


def candidate_selection_summary(selections: pd.DataFrame) -> dict[str, object]:
    if selections.empty:
        return {}
    rows: dict[str, object] = {}
    fp3 = selections[selections["checkpoint"].eq("after_fp3")].copy()
    for profile, group in fp3.groupby("policy_profile", dropna=False):
        selected = group["candidate_selected"].fillna(False).astype(bool)
        rows[str(profile)] = {
            "folds": int(len(group)),
            "candidate_selection_rate": float(selected.mean()) if len(group) else None,
            "candidate_selected_folds": int(selected.sum()),
            "candidate_selection_reasons": group["candidate_selection_reason"]
            .astype(str)
            .value_counts()
            .to_dict(),
        }
    return rows


def leakage_summary(leakage: pd.DataFrame) -> dict[str, object]:
    if leakage.empty:
        return {"rows": 0, "all_rows_valid": False}
    invalid = leakage[leakage["leakage_status"].astype(str).ne("valid")]
    return {
        "rows": int(len(leakage)),
        "invalid_rows": int(len(invalid)),
        "all_rows_valid": invalid.empty,
        "future_event_used_anywhere": bool(
            leakage["future_event_used_anywhere"].astype(bool).any()
        ),
        "frozen_policy_signature_match_rate": float(
            leakage["frozen_policy_signature_match"].astype(bool).mean()
        ),
    }


def records_for_json(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    return json.loads(frame.where(pd.notna(frame), None).to_json(orient="records"))


def interval_metrics(group: pd.DataFrame) -> dict[str, float | None]:
    required = {"prediction_interval_low_sec", "prediction_interval_high_sec"}
    if not required <= set(group.columns):
        return {
            "interval_availability_rate": None,
            "interval_coverage": None,
            "mean_interval_width_sec": None,
        }
    available = group[
        group["prediction_interval_low_sec"].notna() & group["prediction_interval_high_sec"].notna()
    ].copy()
    if available.empty:
        return {
            "interval_availability_rate": 0.0 if len(group) else None,
            "interval_coverage": None,
            "mean_interval_width_sec": None,
        }
    widths = pd.to_numeric(
        available["prediction_interval_high_sec"], errors="coerce"
    ) - pd.to_numeric(available["prediction_interval_low_sec"], errors="coerce")
    coverage = None
    if "interval_contains_actual" in available:
        contains = available["interval_contains_actual"].dropna()
        coverage = float(contains.astype(bool).mean()) if not contains.empty else None
    return {
        "interval_availability_rate": float(len(available) / len(group)) if len(group) else None,
        "interval_coverage": coverage,
        "mean_interval_width_sec": mean_or_none(widths),
    }


def selection_metadata_lookup(
    selections: pd.DataFrame,
) -> dict[tuple[str, int, str], dict[str, object]]:
    lookup: dict[tuple[str, int, str], dict[str, object]] = {}
    if selections.empty:
        return lookup
    for _, row in selections.iterrows():
        if pd.isna(row.get("fold_id")) or pd.isna(row.get("checkpoint")):
            continue
        lookup[
            (
                str(row.get("policy_profile")),
                int(row.get("fold_id")),
                str(row.get("checkpoint")),
            )
        ] = row.to_dict()
    return lookup


def selection_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    columns = [
        col
        for col in [
            "fold_id",
            "season",
            "event",
            "event_slug",
            "checkpoint",
            "selection_mode",
            "selected_family",
            "selected_model_name",
            "selected_feature_group",
            "selected_temporal_weighting_policy",
        ]
        if col in predictions.columns
    ]
    if not columns:
        return pd.DataFrame()
    return predictions[columns].drop_duplicates().copy()


def attach_event_metadata(selection: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    result = selection.copy()
    if {"season", "event", "event_slug"} <= set(result.columns):
        return result
    if predictions.empty:
        return result
    meta_columns = [
        col
        for col in ["fold_id", "checkpoint", "season", "event", "event_slug"]
        if col in predictions.columns
    ]
    meta = predictions[meta_columns].drop_duplicates()
    return result.merge(
        meta,
        on=[col for col in ["fold_id", "checkpoint"] if col in result],
        how="left",
    )


def current_test_season_prior_event_count(row: pd.Series) -> int:
    if pd.notna(row.get("current_test_season_prior_event_count")):
        return int(row.get("current_test_season_prior_event_count"))
    if pd.notna(row.get("current_season_prior_event_count")):
        return int(row.get("current_season_prior_event_count"))
    fold = row.get("fold_id")
    return max(int(fold) - 1, 0) if pd.notna(fold) else 0


def selection_reason(row: pd.Series) -> str | None:
    for column in (
        "season_aware_selection_reason",
        "fallback_reason",
        "guardrail_reason",
        "selection_reason",
    ):
        value = row.get(column)
        if pd.notna(value) and str(value):
            return str(value)
    if bool(row.get("candidate_selected", False)):
        return "candidate_selected"
    return "default_retained"


def allowed_history_scope(row: pd.Series, *, train_seasons: tuple[int, ...]) -> str:
    keys = allowed_history_event_keys(row, train_seasons=train_seasons)
    return json.dumps(keys)


def allowed_history_event_keys(
    row: pd.Series,
    *,
    train_seasons: tuple[int, ...],
    event_fold_lookup: dict[str, int] | None = None,
) -> list[str]:
    current_season = int(row.get("season") or row.get("test_season"))
    current_fold = int(row.get("fold_id")) if pd.notna(row.get("fold_id")) else None
    used = parse_event_key_list(row.get("metric_scope_event_keys"))
    used.extend(parse_event_key_list(row.get("selection_source_events")))
    keys: set[str] = set()
    for key in used:
        season, _ = parse_event_key(key)
        if season in train_seasons:
            keys.add(key)
        elif season == current_season and current_fold is not None:
            if event_key_is_prior_to_fold(
                key,
                current_fold,
                event_fold_lookup=event_fold_lookup,
            ):
                keys.add(key)
    return sorted(keys)


def event_key_is_prior_to_fold(
    key: str,
    fold_id: int,
    *,
    event_fold_lookup: dict[str, int] | None = None,
) -> bool:
    # Saved fold ids increase chronologically in the project walk-forward artifacts.
    if event_fold_lookup is None:
        return bool(key) and fold_id > 0
    prior_fold = event_fold_lookup.get(key)
    return prior_fold is not None and prior_fold < fold_id


def event_fold_lookup_from_selection(selections: pd.DataFrame) -> dict[str, int]:
    if selections.empty or "fold_id" not in selections:
        return {}
    lookup: dict[str, int] = {}
    for _, row in selections.iterrows():
        if pd.isna(row.get("fold_id")):
            continue
        lookup[event_key_from_row(row)] = int(row.get("fold_id"))
    return lookup


def event_key_from_row(row: pd.Series) -> str:
    season = row.get("season") if pd.notna(row.get("season")) else row.get("test_season")
    slug = row.get("event_slug")
    if pd.isna(slug) or slug is None:
        slug = str(row.get("event", "")).strip().lower().replace(" ", "_")
    return f"{int(season)}/{slug}"


def parse_event_key_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value)
    if not text or text == "nan":
        return []
    try:
        parsed = json.loads(text.replace("'", '"'))
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item)]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in text.strip("[]").split(",") if item.strip()]


def parse_event_key(key: str) -> tuple[int, str]:
    parts = str(key).split("/", maxsplit=1)
    try:
        return int(parts[0]), parts[1] if len(parts) > 1 else ""
    except (ValueError, TypeError):
        return 0, str(key)


def source_identity_from_row(row: pd.Series) -> str:
    payload = {
        "source_artifact_kind": row.get("source_artifact_kind"),
        "source_artifact_path": row.get("source_artifact_path"),
        "source_family": row.get("source_family") or row.get("selected_family"),
        "source_model_name": row.get("source_model_name") or row.get("selected_model_name"),
        "source_feature_group": row.get("source_feature_group")
        or row.get("selected_feature_group"),
        "source_temporal_weighting_policy": row.get("source_temporal_weighting_policy")
        or row.get("selected_temporal_weighting_policy"),
        "source_prediction_signature": row.get("source_prediction_signature"),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def abs_error(frame: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(frame["predicted_quali_gap_to_pole_sec"], errors="coerce")
        - pd.to_numeric(frame["quali_gap_to_pole_sec"], errors="coerce")
    ).abs()


def position_error(frame: pd.DataFrame) -> pd.Series:
    if {"predicted_quali_position", "quali_position"} <= set(frame.columns):
        return (
            pd.to_numeric(frame["predicted_quali_position"], errors="coerce")
            - pd.to_numeric(frame["quali_position"], errors="coerce")
        ).abs()
    return pd.Series([pd.NA] * len(frame), index=frame.index)


def mean_or_none(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else None


def median_or_none(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else None


def rmse_or_none(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(math.sqrt(numeric.mean())) if not numeric.empty else None


def weighted_event_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    numeric = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame.get("rows", 1), errors="coerce").fillna(1.0)
    mask = numeric.notna()
    if not mask.any():
        return None
    return float((numeric[mask] * weights[mask]).sum() / weights[mask].sum())


def metric_for_profile(frame: pd.DataFrame, profile: str) -> float | None:
    rows = frame[frame["policy_profile"].eq(profile)]
    if rows.empty or pd.isna(rows["mae_gap_sec"].iloc[0]):
        return None
    return float(rows["mae_gap_sec"].iloc[0])


def format_signed(value: object) -> str:
    if value is None or pd.isna(value):
        return "unavailable"
    return f"{float(value):+.3f} sec"


def config_signature_payload(model_config: ModelConfig) -> dict[str, object]:
    champion = model_config.champion_policy
    season_aware = champion.season_aware_nested_guarded
    guarded = champion.stabilized_nested_guarded
    static = champion.static.get("after_fp3")
    return {
        "random_state": model_config.random_state,
        "static_after_fp3": asdict(static) if static else None,
        "guarded": asdict(guarded),
        "season_aware": asdict(season_aware),
    }


def read_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.is_file() else None


def read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
