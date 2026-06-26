"""True retrain-based prospective champion-policy replay."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig, FeatureConfig, ModelConfig
from f1_prediction.data.season_builder import build_combined_dataset_path
from f1_prediction.features.historical_features import (
    HistoricalFeatureSettings,
    add_historical_features,
)
from f1_prediction.modeling.ablation import _robust_threshold
from f1_prediction.modeling.baselines import generate_baseline_predictions
from f1_prediction.modeling.feature_groups import get_feature_columns_for_group
from f1_prediction.modeling.prospective_policy_evaluation import (
    FrozenPolicyProfile,
    build_cold_start_comparison,
    build_event_comparison,
    build_frozen_policy_profiles,
    current_season_evidence_regime,
    generate_prospective_figures,
    prospective_recommendation,
    records_for_json,
    write_json,
)
from f1_prediction.modeling.season_aware_validation import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    paired_bootstrap_mean_ci,
)
from f1_prediction.modeling.splits import ordered_event_keys
from f1_prediction.modeling.temporal_weighting import (
    TemporalWeightingPolicy,
    prepare_temporal_training_data,
)
from f1_prediction.modeling.train_tabular import fit_and_predict
from f1_prediction.utils.paths import ensure_directory

EVALUATION_TYPE = "true_prospective_replay"
POLICY_PROFILES: tuple[str, ...] = (
    "static_baseline",
    "guarded_baseline",
    "season_aware_frozen",
)
FP3_CHECKPOINT = "after_fp3"
CHECKPOINTS: tuple[str, ...] = ("after_fp1", "after_fp2", "after_fp3")


@dataclass(frozen=True)
class ProspectiveReplaySummary:
    """Paths and status produced by true prospective replay."""

    status: str
    summary_path: Path
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    generation_issues: tuple[str, ...]


def create_prospective_policy_replay_report(
    config: DataConfig,
    model_config: ModelConfig,
    feature_config: FeatureConfig | None = None,
    *,
    train_seasons: tuple[int, ...],
    test_season: int,
    policy_profiles: tuple[str, ...] = POLICY_PROFILES,
    uncertainty: str = "conformal_predicted_gap_bucket",
    dataset_path: Path | None = None,
    min_events: int = 10,
    min_train_events: int = 5,
) -> ProspectiveReplaySummary:
    """Run true retrain-based prospective season replay from modeling rows."""
    metrics_dir = config.metrics_output_dir
    figures_dir = metrics_dir.parent / "figures"
    ensure_directory(metrics_dir)
    ensure_directory(figures_dir)
    source_path = dataset_path or build_combined_dataset_path(config.modeling_output_dir)
    if not source_path.is_file():
        raise FileNotFoundError(f"Combined modeling dataset not found: {source_path}")
    dataset = pd.read_parquet(source_path).reset_index(drop=True)
    event_order = ordered_event_keys(dataset)
    if len(event_order) < min_events:
        raise ValueError(f"Dataset has {len(event_order)} events; at least {min_events} required")
    profiles = build_frozen_policy_profiles(
        model_config,
        profile_names=policy_profiles,
        uncertainty=uncertainty,
    )
    split_id = replay_split_id(train_seasons, test_season)
    replay = run_true_replay(
        dataset=dataset,
        event_order=event_order,
        profiles=profiles,
        model_config=model_config,
        feature_config=feature_config,
        train_seasons=train_seasons,
        test_season=test_season,
        min_train_events=min_train_events,
        uncertainty=uncertainty,
    )

    predictions = replay["predictions"]
    selection = replay["selection"]
    manifest = replay["manifest"]
    leakage = replay["leakage"]
    checkpoint = add_selection_rates(build_replay_checkpoint_comparison(predictions), selection)
    event = build_event_comparison(predictions, selection)
    cold_start = build_cold_start_comparison(event)
    comparison = compare_replay_to_artifact_driven(metrics_dir, checkpoint)

    prediction_path = metrics_dir / f"{split_id}_predictions.parquet"
    predictions.to_parquet(prediction_path, index=False)
    split_summary_path = metrics_dir / f"{split_id}_summary.json"
    table_paths = (
        metrics_dir / "prospective_replay_checkpoint_comparison.csv",
        metrics_dir / "prospective_replay_event_comparison.csv",
        metrics_dir / "prospective_replay_selection_log.csv",
        metrics_dir / "prospective_replay_training_manifest.csv",
        metrics_dir / "prospective_replay_leakage_audit.csv",
        metrics_dir / "prospective_replay_cold_start_comparison.csv",
        metrics_dir / "prospective_replay_vs_artifact_driven.csv",
    )
    combined_tables = merge_replay_tables(
        metrics_dir=metrics_dir,
        split_id=split_id,
        checkpoint=checkpoint,
        event=event,
        selection=selection,
        manifest=manifest,
        leakage=leakage,
        cold_start=cold_start,
        comparison=comparison,
    )
    for path, frame in zip(table_paths, combined_tables, strict=True):
        frame.to_csv(path, index=False)

    figure_paths, generation_issues = generate_replay_figures(
        figures_dir=figures_dir,
        checkpoint=combined_tables[0],
        event=combined_tables[1],
        selection=combined_tables[2],
        manifest=combined_tables[3],
        cold_start=combined_tables[5],
        comparison=combined_tables[6],
    )
    split_summary = build_replay_summary_payload(
        split_id=split_id,
        train_seasons=train_seasons,
        test_season=test_season,
        profiles=profiles,
        checkpoint=checkpoint,
        event=event,
        selection=selection,
        manifest=manifest,
        leakage=leakage,
        cold_start=cold_start,
        comparison=comparison,
        prediction_path=prediction_path,
        generation_issues=generation_issues,
    )
    write_json(split_summary_path, split_summary)
    summary_path = metrics_dir / "prospective_replay_summary.json"
    summary_payload = merge_replay_summary(summary_path, split_summary)
    write_json(summary_path, summary_payload)
    return ProspectiveReplaySummary(
        status=str(summary_payload.get("status", "complete")),
        summary_path=summary_path,
        table_paths=(*table_paths, split_summary_path, prediction_path),
        figure_paths=tuple(figure_paths),
        generation_issues=tuple(generation_issues),
    )


def run_true_replay(
    *,
    dataset: pd.DataFrame,
    event_order: list[str],
    profiles: dict[str, FrozenPolicyProfile],
    model_config: ModelConfig,
    feature_config: FeatureConfig | None,
    train_seasons: tuple[int, ...],
    test_season: int,
    min_train_events: int,
    uncertainty: str,
) -> dict[str, pd.DataFrame]:
    """Replay train-season evidence and held-out test-season events chronologically."""
    row_keys = event_key_series(dataset)
    test_events = [key for key in event_order if event_season(key) == test_season]
    history_predictions: list[pd.DataFrame] = []
    output_predictions: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []
    evidence_events = [
        key
        for key in event_order
        if event_season(key) in set(train_seasons)
        and len(prior_events_for(key, event_order, train_seasons, event_season(key)))
        >= min_train_events
    ]
    replay_events = [*evidence_events, *test_events]
    for event_key in replay_events:
        is_test_output = event_season(event_key) == test_season
        legal_train_events = prior_events_for(event_key, event_order, train_seasons, test_season)
        if len(legal_train_events) < min_train_events:
            continue
        source = train_event_sources(
            dataset=dataset,
            row_keys=row_keys,
            event_order=event_order,
            event_key=event_key,
            legal_train_events=legal_train_events,
            model_config=model_config,
            feature_config=feature_config,
            test_season=test_season,
        )
        manifest_rows.extend(source["manifest"])
        leakage_rows.extend(source["leakage"])
        profile_predictions, profile_selection = apply_profiles_for_event(
            source=source,
            profiles=profiles,
            history=concat_replay_history(history_predictions),
            train_seasons=train_seasons,
            test_season=test_season,
            uncertainty=uncertainty,
        )
        history_predictions.append(profile_predictions)
        if is_test_output:
            output_predictions.append(profile_predictions)
            selection_rows.extend(profile_selection)
    predictions = (
        pd.concat(output_predictions, ignore_index=True, sort=False)
        if output_predictions
        else pd.DataFrame()
    )
    selection = pd.DataFrame(selection_rows)
    return {
        "predictions": predictions,
        "selection": selection,
        "manifest": pd.DataFrame(manifest_rows),
        "leakage": pd.DataFrame(leakage_rows),
    }


def train_event_sources(
    *,
    dataset: pd.DataFrame,
    row_keys: pd.Series,
    event_order: list[str],
    event_key: str,
    legal_train_events: list[str],
    model_config: ModelConfig,
    feature_config: FeatureConfig | None,
    test_season: int,
) -> dict[str, Any]:
    """Fit static and weighted FP3 candidates for one prospective event."""
    fold_scope = dataset[row_keys.isin([*legal_train_events, event_key])].copy()
    fold_scope = add_historical_features(
        fold_scope,
        historical_settings(feature_config),
        excluded_target_events={event_key},
    )
    fold_keys = event_key_series(fold_scope)
    train = fold_scope[fold_keys.isin(legal_train_events)].copy()
    test = fold_scope[fold_keys.eq(event_key)].copy()
    if train.empty or test.empty:
        raise ValueError(f"Replay event {event_key} must have train and test rows")
    feature_columns = get_feature_columns_for_group(fold_scope, "base_plus_relative")
    static_predictions, static_fit = fit_source_candidate(
        train=train,
        test=test,
        event_order=event_order,
        event_key=event_key,
        model_config=model_config,
        feature_columns=feature_columns,
        temporal_policy=TemporalWeightingPolicy.uniform,
    )
    weighted_predictions, weighted_fit = fit_source_candidate(
        train=train,
        test=test,
        event_order=event_order,
        event_key=event_key,
        model_config=model_config,
        feature_columns=feature_columns,
        temporal_policy=TemporalWeightingPolicy.current_season_only_with_prior,
    )
    baseline = generate_baseline_predictions(
        test,
        robust_extreme_threshold_sec=_robust_threshold(feature_config),
    )
    manifest = [
        training_manifest_row(
            event_key=event_key,
            test=test,
            train=train,
            legal_train_events=legal_train_events,
            fit_payload=static_fit,
            model_config=model_config,
            temporal_policy="uniform",
            policy_profile="static_baseline",
            test_season=test_season,
        ),
        training_manifest_row(
            event_key=event_key,
            test=test,
            train=train,
            legal_train_events=legal_train_events,
            fit_payload=weighted_fit,
            model_config=model_config,
            temporal_policy="current_season_only_with_prior",
            policy_profile="season_aware_frozen",
            test_season=test_season,
        ),
    ]
    leakage = [leakage_row(row, event_order=event_order) for row in manifest]
    return {
        "event_key": event_key,
        "test": test,
        "static": static_predictions,
        "weighted": weighted_predictions,
        "baseline": baseline,
        "manifest": manifest,
        "leakage": leakage,
    }


def fit_source_candidate(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    event_order: list[str],
    event_key: str,
    model_config: ModelConfig,
    feature_columns: list[str],
    temporal_policy: TemporalWeightingPolicy,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Train one FP3 RF candidate and return predictions plus fit metadata."""
    temporal = prepare_temporal_training_data(
        train,
        test_event=event_key,
        event_order=event_order,
        config=model_config.temporal_weighting,
        policy=temporal_policy,
    )
    fp3_test = test[test["checkpoint"].eq(FP3_CHECKPOINT)].copy()
    predictions, fitted = fit_and_predict(
        temporal.train,
        fp3_test,
        model_config=model_config,
        candidate_features=feature_columns,
        sample_weights=temporal.sample_weights,
    )
    frame = predictions[predictions["model_name"].eq("random_forest")].copy()
    frame["feature_group"] = "base_plus_relative"
    frame["temporal_weighting_policy"] = temporal_policy.value
    frame["source_artifact_kind"] = "true_prospective_replay"
    frame["prediction_source_identity"] = json.dumps(
        {
            "family": "ablation",
            "model_name": "random_forest",
            "feature_group": "base_plus_relative",
            "temporal_weighting_policy": temporal_policy.value,
            "event_key": event_key,
        },
        sort_keys=True,
    )
    fit_info = fitted.get("random_forest", {}).get(FP3_CHECKPOINT, {})
    fit_payload = {
        "feature_columns": list(fit_info.get("feature_columns", [])),
        "sample_weight_summary": temporal.summary,
    }
    return frame, fit_payload


def apply_profiles_for_event(
    *,
    source: dict[str, Any],
    profiles: dict[str, FrozenPolicyProfile],
    history: pd.DataFrame,
    train_seasons: tuple[int, ...],
    test_season: int,
    uncertainty: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Apply frozen profiles to one event's freshly trained source predictions."""
    event_key = source["event_key"]
    static_fp3 = source["static"].copy()
    weighted_fp3 = source["weighted"].copy()
    baseline = source["baseline"].copy()
    rows: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    prior_count = same_season_prior_count(event_key, history)
    for profile_name, profile in profiles.items():
        selected_weighted = False
        reason = "default_retained"
        if profile_name == "season_aware_frozen":
            selected_weighted, reason = season_aware_decision(
                history,
                event_key=event_key,
                profile=profile,
            )
        profile_frames = []
        for checkpoint in CHECKPOINTS:
            if checkpoint == FP3_CHECKPOINT:
                frame = weighted_fp3.copy() if selected_weighted else static_fp3.copy()
                source_policy = "current_season_only_with_prior" if selected_weighted else "uniform"
                family = "ablation"
                model_name = "random_forest"
                feature_group = "base_plus_relative"
            else:
                model_name = (
                    "robust_best_push_lap"
                    if checkpoint == "after_fp1"
                    else "robust_theoretical_best_lap"
                )
                frame = baseline[
                    baseline["checkpoint"].eq(checkpoint) & baseline["baseline_name"].eq(model_name)
                ].copy()
                frame["model_name"] = model_name
                frame["feature_group"] = pd.NA
                source_policy = "uniform"
                family = "robust_baseline"
                feature_group = None
            frame = decorate_prediction_frame(
                frame,
                profile=profile,
                profile_name=profile_name,
                event_key=event_key,
                train_seasons=train_seasons,
                test_season=test_season,
                selected_weighted=selected_weighted,
                reason=reason if checkpoint == FP3_CHECKPOINT else "not_applicable_checkpoint",
                family=family,
                model_name=model_name,
                feature_group=feature_group,
                source_policy=source_policy,
                current_prior_count=prior_count,
            )
            if history.empty or "policy_profile" not in history:
                prior = pd.DataFrame()
            else:
                prior = history[
                    history["policy_profile"].eq(profile_name)
                    & history["checkpoint"].eq(checkpoint)
                ]
            frame = add_replay_intervals(frame, prior, uncertainty=uncertainty)
            profile_frames.append(frame)
            selection_rows.append(
                selection_row(
                    frame,
                    profile=profile,
                    profile_name=profile_name,
                    event_key=event_key,
                    checkpoint=checkpoint,
                    selected_weighted=selected_weighted if checkpoint == FP3_CHECKPOINT else False,
                    reason=reason if checkpoint == FP3_CHECKPOINT else "not_applicable_checkpoint",
                )
            )
        rows.append(pd.concat(profile_frames, ignore_index=True, sort=False))
    return pd.concat(rows, ignore_index=True, sort=False), selection_rows


def season_aware_decision(
    history: pd.DataFrame,
    *,
    event_key: str,
    profile: FrozenPolicyProfile,
) -> tuple[bool, str]:
    """Return whether to use weighted candidate using only prior replay evidence."""
    prior_count = same_season_prior_count(event_key, history)
    if prior_count < int(profile.cold_start_threshold or 0):
        return False, "season_aware_cold_start"
    fp3 = history[history["checkpoint"].eq(FP3_CHECKPOINT)].copy()
    candidate = fp3[fp3["source_temporal_weighting_policy"].eq("current_season_only_with_prior")]
    default = fp3[fp3["source_temporal_weighting_policy"].eq("uniform")]
    aligned = align_candidate_default(candidate, default)
    prior_folds = int(aligned["fold_id"].nunique()) if not aligned.empty else 0
    if prior_folds < int(profile.min_prior_folds or 0) or len(aligned) < int(
        profile.min_prior_predictions or 0
    ):
        return False, "insufficient_candidate_history"
    candidate_mae = float(aligned["candidate_abs_error"].mean())
    default_mae = float(aligned["default_abs_error"].mean())
    improvement = default_mae - candidate_mae
    if improvement >= float(profile.improvement_margin_sec or 0.0):
        return True, "season_aware_candidate_selected"
    return False, "margin_not_met"


def decorate_prediction_frame(
    frame: pd.DataFrame,
    *,
    profile: FrozenPolicyProfile,
    profile_name: str,
    event_key: str,
    train_seasons: tuple[int, ...],
    test_season: int,
    selected_weighted: bool,
    reason: str,
    family: str,
    model_name: str,
    feature_group: str | None,
    source_policy: str,
    current_prior_count: int,
) -> pd.DataFrame:
    """Attach replay metadata to prediction rows."""
    result = frame.copy()
    season, slug = parse_event_key(event_key)
    result["evaluation_type"] = EVALUATION_TYPE
    result["prospective_split"] = replay_split_id(train_seasons, test_season)
    result["train_seasons"] = ",".join(str(season) for season in train_seasons)
    result["test_season"] = test_season
    result["policy_profile"] = profile_name
    result["policy_signature"] = profile.to_payload()["policy_signature"]
    result["event"] = result.get("event", slug)
    result["event_slug"] = slug
    result["prospective_event_id"] = event_key
    result["fold_id"] = event_index_from_key(event_key)
    result["driver_key"] = result["driver"] if "driver_key" not in result else result["driver_key"]
    result["prediction_gap_sec"] = result["predicted_quali_gap_to_pole_sec"]
    result["actual_gap_sec"] = result["quali_gap_to_pole_sec"]
    result["candidate_selection_reason"] = reason
    result["current_test_season_prior_event_count"] = current_prior_count
    result["cold_start_regime"] = result["current_test_season_prior_event_count"].map(
        current_season_evidence_regime
    )
    result["history_scope_valid"] = True
    result["selected_family"] = family
    result["selected_model_name"] = model_name
    result["selected_feature_group"] = feature_group
    result["selected_temporal_weighting_policy"] = source_policy
    result["source_family"] = family
    result["source_model_name"] = model_name
    result["source_feature_group"] = feature_group
    result["source_temporal_weighting_policy"] = source_policy
    result["season_aware_selected"] = bool(selected_weighted)
    result["season"] = season
    return result


def selection_row(
    frame: pd.DataFrame,
    *,
    profile: FrozenPolicyProfile,
    profile_name: str,
    event_key: str,
    checkpoint: str,
    selected_weighted: bool,
    reason: str,
) -> dict[str, object]:
    season, slug = parse_event_key(event_key)
    return {
        "evaluation_type": EVALUATION_TYPE,
        "prospective_split": frame["prospective_split"].iloc[0],
        "train_seasons": frame["train_seasons"].iloc[0],
        "test_season": frame["test_season"].iloc[0],
        "policy_profile": profile_name,
        "policy_signature": profile.to_payload()["policy_signature"],
        "season": season,
        "event": frame["event"].iloc[0],
        "event_slug": slug,
        "fold_id": event_index_from_key(event_key),
        "checkpoint": checkpoint,
        "season_aware_selected": bool(selected_weighted),
        "candidate_selected": bool(selected_weighted),
        "candidate_selection_reason": reason,
        "current_test_season_prior_event_count": int(
            frame["current_test_season_prior_event_count"].iloc[0]
        ),
        "cold_start_regime": frame["cold_start_regime"].iloc[0],
        "history_scope_valid": True,
    }


def training_manifest_row(
    *,
    event_key: str,
    test: pd.DataFrame,
    train: pd.DataFrame,
    legal_train_events: list[str],
    fit_payload: dict[str, object],
    model_config: ModelConfig,
    temporal_policy: str,
    policy_profile: str,
    test_season: int,
) -> dict[str, object]:
    """Build the requested model-fit metadata row."""
    feature_columns = list(fit_payload.get("feature_columns", []))
    sample_weight_summary = fit_payload.get("sample_weight_summary", {})
    seasons = sorted({event_season(key) for key in legal_train_events})
    return {
        "evaluation_type": EVALUATION_TYPE,
        "test_event": event_key,
        "test_season": test_season,
        "checkpoint": FP3_CHECKPOINT,
        "policy_profile": policy_profile,
        "training_seasons_used": json.dumps(seasons),
        "training_event_keys_used": json.dumps(legal_train_events),
        "training_max_event_key": legal_train_events[-1] if legal_train_events else None,
        "same_test_season_prior_events_used": json.dumps(
            [key for key in legal_train_events if event_season(key) == test_season]
        ),
        "future_test_season_events_used": json.dumps([]),
        "feature_columns_signature": stable_signature(feature_columns),
        "model_configuration_signature": stable_signature(
            {
                "model": "random_forest",
                "random_forest": model_config.random_forest,
                "random_state": model_config.random_state,
            }
        ),
        "temporal_weighting_policy": temporal_policy,
        "sample_weight_summary": json.dumps(sample_weight_summary, sort_keys=True, default=str),
        "training_row_count": int(len(train)),
        "training_event_count": int(len(legal_train_events)),
        "random_state": model_config.random_state,
        "fit_timestamp": utc_now(),
        "current_event_in_training": event_key in legal_train_events,
        "target_leakage_columns_used": json.dumps(
            [
                col
                for col in feature_columns
                if col in {"quali_gap_to_pole_sec", "quali_position", "reached_q3"}
            ]
        ),
    }


def leakage_row(row: dict[str, object], *, event_order: list[str]) -> dict[str, object]:
    train_events = json.loads(str(row["training_event_keys_used"]))
    event_key = str(row["test_event"])
    test_season = int(row["test_season"])
    current_index = event_order.index(event_key)
    future_test = [
        key
        for key in train_events
        if event_season(key) == test_season and event_order.index(key) >= current_index
    ]
    future_any = [key for key in train_events if event_order.index(key) >= current_index]
    leakage = bool(
        row["current_event_in_training"]
        or future_test
        or future_any
        or json.loads(str(row["target_leakage_columns_used"]))
    )
    reason = "valid"
    if row["current_event_in_training"]:
        reason = "current_event_in_training"
    elif future_test or future_any:
        reason = "future_event_in_training"
    elif json.loads(str(row["target_leakage_columns_used"])):
        reason = "target_leakage_columns_used"
    return {
        "test_season": test_season,
        "test_event": event_key,
        "checkpoint": row["checkpoint"],
        "policy_profile": row["policy_profile"],
        "future_test_season_event_used": bool(future_test),
        "future_event_used_anywhere": bool(future_any),
        "current_event_used": bool(row["current_event_in_training"]),
        "history_scope_valid": not leakage,
        "leakage_status": "invalid" if leakage else "valid",
        "leakage_reason": reason,
    }


def add_replay_intervals(
    frame: pd.DataFrame,
    prior: pd.DataFrame,
    *,
    uncertainty: str,
) -> pd.DataFrame:
    """Add simple prior-residual conformal intervals from replay history."""
    result = frame.copy()
    result["uncertainty_method"] = uncertainty
    result["prediction_interval_low_sec"] = pd.NA
    result["prediction_interval_high_sec"] = pd.NA
    result["interval_contains_actual"] = pd.NA
    result["residual_quantile_sec"] = pd.NA
    if prior.empty or len(prior) < 20:
        result["uncertainty_method"] = "insufficient_history"
        return result
    residuals = (
        pd.to_numeric(prior["predicted_quali_gap_to_pole_sec"], errors="coerce")
        - pd.to_numeric(prior["quali_gap_to_pole_sec"], errors="coerce")
    ).abs()
    residuals = residuals.dropna()
    if len(residuals) < 20:
        result["uncertainty_method"] = "insufficient_history"
        return result
    q = float(residuals.quantile(0.90))
    pred = pd.to_numeric(result["predicted_quali_gap_to_pole_sec"], errors="coerce")
    actual = pd.to_numeric(result["quali_gap_to_pole_sec"], errors="coerce")
    result["prediction_interval_low_sec"] = pred - q
    result["prediction_interval_high_sec"] = pred + q
    result["interval_contains_actual"] = actual.between(pred - q, pred + q)
    result["residual_quantile_sec"] = q
    return result


def build_replay_checkpoint_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    from f1_prediction.modeling.prospective_policy_evaluation import build_checkpoint_comparison

    return build_checkpoint_comparison(predictions)


def add_selection_rates(checkpoint: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    if checkpoint.empty or selection.empty:
        return checkpoint
    result = checkpoint.copy()
    rates = (
        selection.groupby(["prospective_split", "policy_profile", "checkpoint"], dropna=False)[
            "candidate_selected"
        ]
        .mean()
        .reset_index(name="candidate_selection_rate")
    )
    merged = result.drop(columns=["candidate_selection_rate"], errors="ignore").merge(
        rates,
        on=["prospective_split", "policy_profile", "checkpoint"],
        how="left",
    )
    return merged[result.columns]


def compare_replay_to_artifact_driven(
    metrics_dir: Path,
    replay_checkpoint: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "prospective_split",
        "artifact_driven_split",
        "policy_profile",
        "checkpoint",
        "artifact_driven_mae",
        "replay_mae",
        "delta_replay_minus_artifact_driven",
        "artifact_driven_selection_rate",
        "replay_selection_rate",
        "comparison_interpretation",
    ]
    path = metrics_dir / "prospective_policy_checkpoint_comparison.csv"
    if not path.is_file() or replay_checkpoint.empty:
        return pd.DataFrame(columns=columns)
    artifact = pd.read_csv(path)
    artifact = artifact.copy()
    replay = replay_checkpoint.copy()
    artifact["comparison_split"] = artifact["prospective_split"].astype(str)
    replay["comparison_split"] = (
        replay["prospective_split"]
        .astype(str)
        .str.replace(
            "prospective_replay_",
            "prospective_",
            regex=False,
        )
    )
    merged = artifact.merge(
        replay,
        on=["comparison_split", "policy_profile", "checkpoint"],
        suffixes=("_artifact", "_replay"),
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, row in merged.iterrows():
        artifact_mae = row.get("mae_gap_sec_artifact")
        replay_mae = row.get("mae_gap_sec_replay")
        delta = float(replay_mae) - float(artifact_mae)
        rows.append(
            {
                "prospective_split": row["prospective_split_replay"],
                "artifact_driven_split": row["prospective_split_artifact"],
                "policy_profile": row["policy_profile"],
                "checkpoint": row["checkpoint"],
                "artifact_driven_mae": artifact_mae,
                "replay_mae": replay_mae,
                "delta_replay_minus_artifact_driven": delta,
                "artifact_driven_selection_rate": row.get("candidate_selection_rate_artifact"),
                "replay_selection_rate": row.get("candidate_selection_rate_replay"),
                "comparison_interpretation": (
                    "differs_due_to_retrain_history"
                    if abs(delta) > 1e-9
                    else "matches_artifact_driven"
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def merge_replay_tables(
    *,
    metrics_dir: Path,
    split_id: str,
    checkpoint: pd.DataFrame,
    event: pd.DataFrame,
    selection: pd.DataFrame,
    manifest: pd.DataFrame,
    leakage: pd.DataFrame,
    cold_start: pd.DataFrame,
    comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    items = [
        ("prospective_replay_checkpoint_comparison.csv", checkpoint),
        ("prospective_replay_event_comparison.csv", event),
        ("prospective_replay_selection_log.csv", selection),
        ("prospective_replay_training_manifest.csv", manifest),
        ("prospective_replay_leakage_audit.csv", leakage),
        ("prospective_replay_cold_start_comparison.csv", cold_start),
        ("prospective_replay_vs_artifact_driven.csv", comparison),
    ]
    merged = []
    for filename, frame in items:
        path = metrics_dir / filename
        existing = pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=frame.columns)
        if not existing.empty and "prospective_split" in existing.columns:
            existing = existing[existing["prospective_split"].astype(str).ne(split_id)]
        if not existing.empty:
            frame = pd.concat([existing, frame], ignore_index=True, sort=False)
        merged.append(frame)
    return tuple(merged)


def build_replay_summary_payload(
    *,
    split_id: str,
    train_seasons: tuple[int, ...],
    test_season: int,
    profiles: dict[str, FrozenPolicyProfile],
    checkpoint: pd.DataFrame,
    event: pd.DataFrame,
    selection: pd.DataFrame,
    manifest: pd.DataFrame,
    leakage: pd.DataFrame,
    cold_start: pd.DataFrame,
    comparison: pd.DataFrame,
    prediction_path: Path,
    generation_issues: list[str],
) -> dict[str, object]:
    bootstrap = replay_bootstrap_summary(event)
    recommendation = replay_recommendation(checkpoint, leakage)
    return {
        "status": "complete" if not checkpoint.empty else "partial",
        "evaluation_type": EVALUATION_TYPE,
        "prospective_split": split_id,
        "train_seasons": list(train_seasons),
        "test_season": test_season,
        "policy_profiles": {name: profile.to_payload() for name, profile in profiles.items()},
        "checkpoint_summary": records_for_json(checkpoint),
        "fp3_summary": records_for_json(checkpoint[checkpoint["checkpoint"].eq(FP3_CHECKPOINT)]),
        "selection_summary": selection_summary(selection),
        "training_manifest_summary": training_manifest_summary(manifest),
        "leakage_audit_summary": leakage_summary(leakage),
        "cold_start_summary": records_for_json(cold_start),
        "bootstrap_confidence_intervals": bootstrap,
        "artifact_driven_comparison": records_for_json(comparison),
        "recommendation": recommendation,
        "main_findings": replay_findings(checkpoint, leakage, comparison, recommendation),
        "artifact_paths": {
            "predictions": str(prediction_path),
            "checkpoint_comparison": "reports/metrics/prospective_replay_checkpoint_comparison.csv",
            "event_comparison": "reports/metrics/prospective_replay_event_comparison.csv",
            "selection_log": "reports/metrics/prospective_replay_selection_log.csv",
            "training_manifest": "reports/metrics/prospective_replay_training_manifest.csv",
            "leakage_audit": "reports/metrics/prospective_replay_leakage_audit.csv",
        },
        "generation_issues": generation_issues,
        "generated_at": utc_now(),
    }


def merge_replay_summary(summary_path: Path, split_summary: dict[str, object]) -> dict[str, object]:
    existing = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    splits = list(existing.get("splits", [])) if isinstance(existing.get("splits"), list) else []
    split_id = str(split_summary["prospective_split"])
    splits = [item for item in splits if item.get("prospective_split") != split_id]
    splits.append(split_summary)
    recommendation = aggregate_replay_recommendation(splits)
    return {
        "status": "complete" if splits else "partial",
        "evaluation_type": EVALUATION_TYPE,
        "splits": splits,
        "prospective_replay_splits_available": [item["prospective_split"] for item in splits],
        "leakage_audit_result": {
            "splits_checked": len(splits),
            "all_splits_valid": all(
                item.get("leakage_audit_summary", {}).get("all_rows_valid") for item in splits
            ),
        },
        "recommendation": recommendation,
        "main_findings": [finding for item in splits for finding in item.get("main_findings", [])],
        "generated_at": utc_now(),
    }


def generate_replay_figures(
    *,
    figures_dir: Path,
    checkpoint: pd.DataFrame,
    event: pd.DataFrame,
    selection: pd.DataFrame,
    manifest: pd.DataFrame,
    cold_start: pd.DataFrame,
    comparison: pd.DataFrame,
) -> tuple[list[Path], list[str]]:
    """Generate static replay figures."""
    matplotlib_cache = figures_dir / ".matplotlib-cache"
    ensure_directory(matplotlib_cache)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)
    os.environ["XDG_CACHE_HOME"] = str(matplotlib_cache)
    temp_figures_dir = figures_dir / ".prospective_replay_tmp"
    ensure_directory(temp_figures_dir)
    paths, issues = generate_prospective_figures(
        figures_dir=temp_figures_dir,
        checkpoint=checkpoint.rename(
            columns={
                "prospective_replay_split": "prospective_split",
            }
        ),
        event=event,
        selections=selection,
        cold_start=cold_start,
    )
    renamed = []
    for path in paths:
        new_path = figures_dir / path.name.replace("prospective_policy_", "prospective_replay_")
        path.replace(new_path)
        renamed.append(new_path)
    try:
        temp_figures_dir.rmdir()
    except OSError:
        pass
    os.environ.setdefault("MPLCONFIGDIR", str(figures_dir.parent / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extra_specs = [
        (
            figures_dir / "prospective_replay_training_history_growth.png",
            lambda p: plot_training_history(plt, manifest, p),
        ),
        (
            figures_dir / "prospective_replay_vs_artifact_driven.png",
            lambda p: plot_replay_vs_artifact(plt, comparison, p),
        ),
    ]
    for path, callback in extra_specs:
        try:
            if callback(path):
                renamed.append(path)
            else:
                issues.append(f"skipped figure {path.name}: insufficient data")
        except Exception as exc:  # pragma: no cover
            issues.append(f"skipped figure {path.name}: {exc}")
        finally:
            plt.close("all")
    return renamed, issues


def concat_replay_history(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate prior replay predictions without all-empty metadata columns."""
    non_empty = [frame.dropna(axis=1, how="all") for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True, sort=False)


def plot_training_history(plt: Any, manifest: pd.DataFrame, path: Path) -> bool:
    if manifest.empty:
        return False
    frame = manifest[manifest["test_season"].eq(manifest["test_season"].max())].copy()
    if frame.empty:
        frame = manifest.copy()
    ax = frame.plot(
        x="test_event",
        y="training_event_count",
        kind="bar",
        legend=False,
        color="#356f9f",
        figsize=(10, 4),
    )
    ax.set_title("Prospective replay training history growth")
    ax.set_ylabel("Training events")
    ax.set_xlabel("")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    return True


def plot_replay_vs_artifact(plt: Any, comparison: pd.DataFrame, path: Path) -> bool:
    frame = comparison[comparison["checkpoint"].eq(FP3_CHECKPOINT)].copy()
    if frame.empty:
        return False
    frame["label"] = (
        frame["prospective_split"].astype(str) + " " + frame["policy_profile"].astype(str)
    )
    ax = frame.plot.bar(
        x="label",
        y="delta_replay_minus_artifact_driven",
        legend=False,
        color="#6c6c9d",
        figsize=(10, 4),
    )
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("Replay minus artifact-driven FP3 MAE")
    ax.set_ylabel("Delta MAE (sec)")
    ax.set_xlabel("")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    return True


def replay_bootstrap_summary(event: pd.DataFrame) -> dict[str, object]:
    fp3 = event[event["checkpoint"].eq(FP3_CHECKPOINT)] if not event.empty else event
    rows = {}
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


def align_candidate_default(candidate: pd.DataFrame, default: pd.DataFrame) -> pd.DataFrame:
    keys = ["fold_id", "season", "event_slug", "checkpoint", "driver"]
    left = candidate[keys + ["quali_gap_to_pole_sec", "predicted_quali_gap_to_pole_sec"]].copy()
    right = default[keys + ["predicted_quali_gap_to_pole_sec"]].copy()
    merged = left.merge(right, on=keys, suffixes=("_candidate", "_default"), how="inner")
    if merged.empty:
        return merged
    actual = pd.to_numeric(merged["quali_gap_to_pole_sec"], errors="coerce")
    merged["candidate_abs_error"] = (
        pd.to_numeric(merged["predicted_quali_gap_to_pole_sec_candidate"], errors="coerce") - actual
    ).abs()
    merged["default_abs_error"] = (
        pd.to_numeric(merged["predicted_quali_gap_to_pole_sec_default"], errors="coerce") - actual
    ).abs()
    return merged


def selection_summary(selection: pd.DataFrame) -> dict[str, object]:
    if selection.empty:
        return {}
    fp3 = selection[selection["checkpoint"].eq(FP3_CHECKPOINT)]
    return {
        profile: {
            "folds": int(len(group)),
            "candidate_selection_rate": float(group["candidate_selected"].astype(bool).mean())
            if len(group)
            else None,
            "selection_reasons": group["candidate_selection_reason"]
            .astype(str)
            .value_counts()
            .to_dict(),
        }
        for profile, group in fp3.groupby("policy_profile", dropna=False)
    }


def training_manifest_summary(manifest: pd.DataFrame) -> dict[str, object]:
    if manifest.empty:
        return {}
    return {
        "fits": int(len(manifest)),
        "mean_training_events": float(manifest["training_event_count"].mean()),
        "mean_training_rows": float(manifest["training_row_count"].mean()),
    }


def leakage_summary(leakage: pd.DataFrame) -> dict[str, object]:
    if leakage.empty:
        return {"rows": 0, "all_rows_valid": False}
    invalid = leakage[leakage["leakage_status"].ne("valid")]
    return {
        "rows": int(len(leakage)),
        "invalid_rows": int(len(invalid)),
        "all_rows_valid": invalid.empty,
        "future_event_used_anywhere": bool(
            leakage["future_event_used_anywhere"].astype(bool).any()
        ),
    }


def replay_findings(
    checkpoint: pd.DataFrame,
    leakage: pd.DataFrame,
    comparison: pd.DataFrame,
    recommendation: str,
) -> list[str]:
    findings = []
    if not leakage.empty and leakage["leakage_status"].eq("valid").all():
        findings.append("True replay leakage audit found no current or future event in training.")
    fp3 = (
        checkpoint[checkpoint["checkpoint"].eq(FP3_CHECKPOINT)]
        if not checkpoint.empty
        else checkpoint
    )
    season_aware = fp3[fp3["policy_profile"].eq("season_aware_frozen")]
    if not season_aware.empty:
        row = season_aware.iloc[0]
        findings.append(
            "Season-aware replay FP3 delta was "
            f"{float(row['fp3_delta_vs_static_baseline_sec']):+.3f} versus static and "
            f"{float(row['fp3_delta_vs_guarded_baseline_sec']):+.3f} versus guarded."
        )
    if not comparison.empty:
        findings.append(
            "Replay and artifact-driven results differ when fold-specific retraining changes "
            "source predictions."
        )
    findings.append(f"Conservative recommendation: {recommendation}.")
    return findings


def aggregate_replay_recommendation(splits: list[dict[str, object]]) -> str:
    if not splits:
        return "season_aware_candidate_requires_more_evidence"
    if not all(item.get("leakage_audit_summary", {}).get("all_rows_valid") for item in splits):
        return "season_aware_candidate_requires_more_evidence"
    recommendations = [str(item.get("recommendation")) for item in splits]
    if recommendations and all(
        value == "season_aware_candidate_eligible_for_default_consideration"
        for value in recommendations
    ):
        return "season_aware_candidate_eligible_for_default_consideration"
    if "retain_static_policy" in recommendations:
        return "retain_static_policy"
    if "retain_guarded_policy" in recommendations:
        return "retain_guarded_policy"
    return "retain_static_policy"


def replay_recommendation(checkpoint: pd.DataFrame, leakage: pd.DataFrame) -> str:
    """Return conservative replay recommendation with static preferred on ties."""
    recommendation = prospective_recommendation(checkpoint, leakage)
    if recommendation != "retain_guarded_policy":
        return recommendation
    fp3 = (
        checkpoint[checkpoint["checkpoint"].eq(FP3_CHECKPOINT)]
        if not checkpoint.empty
        else checkpoint
    )
    static = fp3[fp3["policy_profile"].eq("static_baseline")]
    guarded = fp3[fp3["policy_profile"].eq("guarded_baseline")]
    if not static.empty and not guarded.empty:
        mae_delta = abs(
            float(static["mae_gap_sec"].iloc[0]) - float(guarded["mae_gap_sec"].iloc[0])
        )
        if mae_delta <= 1e-12:
            return "retain_static_policy"
    return recommendation


def prior_events_for(
    event_key: str,
    event_order: list[str],
    train_seasons: tuple[int, ...],
    test_season: int,
) -> list[str]:
    index = event_order.index(event_key)
    allowed = []
    for key in event_order[:index]:
        season = event_season(key)
        if season in train_seasons or (
            season == test_season and event_season(event_key) == test_season
        ):
            allowed.append(key)
    return allowed


def event_key_series(dataset: pd.DataFrame) -> pd.Series:
    return dataset["season"].astype(str) + "/" + dataset["event_slug"].astype(str)


def parse_event_key(event_key: str) -> tuple[int, str]:
    season, slug = event_key.split("/", maxsplit=1)
    return int(season), slug


def event_season(event_key: str) -> int:
    return parse_event_key(event_key)[0]


def event_index_from_key(event_key: str) -> int:
    digest = hashlib.sha256(event_key.encode("utf-8")).hexdigest()[:10]
    return int(digest, 16)


def same_season_prior_count(event_key: str, history: pd.DataFrame) -> int:
    if history.empty:
        return 0
    season = event_season(event_key)
    fp3 = history[history["checkpoint"].eq(FP3_CHECKPOINT)]
    return int(fp3.loc[fp3["season"].astype(int).eq(season), "event_slug"].nunique())


def historical_settings(config: FeatureConfig | None) -> HistoricalFeatureSettings:
    if config is None or config.historical_features is None:
        return HistoricalFeatureSettings()
    settings = config.historical_features
    return HistoricalFeatureSettings(settings.rolling_windows, settings.min_periods)


def stable_signature(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def replay_split_id(train_seasons: tuple[int, ...], test_season: int) -> str:
    train = "_".join(str(season) for season in train_seasons)
    return f"prospective_replay_train_{train}_test_{test_season}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
