"""Checkpoint-specific champion selection from prior out-of-sample predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd

from f1_prediction.config import (
    ChampionMethodConfig,
    DataConfig,
    ModelConfig,
    PredictedGapBucketUncertaintyConfig,
    SeasonAwareNestedGuardedChampionConfig,
    StabilizedNestedChampionConfig,
    StabilizedNestedGuardedChampionConfig,
    UncertaintyConfig,
)
from f1_prediction.data.season_builder import build_combined_dataset_path
from f1_prediction.modeling.backtest_tabular import (
    BacktestFold,
    BacktestStrategy,
    build_backtest_folds,
)
from f1_prediction.modeling.metrics import compute_prediction_metrics
from f1_prediction.modeling.splits import ordered_event_keys
from f1_prediction.modeling.temporal_weighting import temporal_artifact_stem
from f1_prediction.utils.paths import ensure_directory

CHECKPOINTS: tuple[str, ...] = ("after_fp1", "after_fp2", "after_fp3")
METHOD_COLUMNS: tuple[str, ...] = (
    "candidate_family",
    "model_name",
    "feature_group",
)
PREDICTION_KEY_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "season",
    "event_slug",
    "checkpoint",
    "driver",
)


class ChampionSelectionMode(str, Enum):
    """Supported champion selection modes."""

    static = "static"
    nested = "nested"
    stabilized_nested = "stabilized_nested"
    stabilized_nested_guarded = "stabilized_nested_guarded"
    season_aware_nested_guarded = "season_aware_nested_guarded"


class ChampionUncertaintyMethod(str, Enum):
    """Supported champion prediction interval methods."""

    residual_std = "residual_std"
    conformal = "conformal"
    conformal_predicted_gap_bucket = "conformal_predicted_gap_bucket"


@dataclass(frozen=True)
class ChampionBacktestSummary:
    """Counts and paths produced by champion backtesting."""

    status: str
    strategy: str
    selection_mode: str
    n_events: int
    n_folds_total: int
    n_folds_successful: int
    n_folds_failed: int
    prediction_rows: int
    metrics_path: Path
    predictions_path: Path | None
    selection_path: Path | None


@dataclass(frozen=True)
class ChampionSelectionDecision:
    """One checkpoint-level champion selection decision."""

    selected: ChampionMethodConfig
    selected_metric_value: float | None
    default_metric_value: float | None
    source_events: list[str]
    prior_folds_used: int
    prior_predictions_used: int
    fallback_used: bool
    fallback_reason: str | None


@dataclass(frozen=True)
class ChampionGuardrailDecision:
    """Selected method after any opt-in champion guardrail has been applied."""

    selected: ChampionMethodConfig
    pre_guardrail_selected: ChampionMethodConfig
    guardrail_applied: bool
    guardrail_name: str | None
    guardrail_reason: str | None


@dataclass(frozen=True)
class SeasonAwareChampionDecision:
    """Selected FP3 method after the season-aware weighted-candidate gate."""

    selected: ChampionMethodConfig
    season_aware_candidate_available: bool
    season_aware_candidate_eligible: bool
    current_season_prior_event_count: int
    prior_candidate_folds: int
    prior_candidate_predictions: int
    candidate_metric_value: float | None
    default_metric_value: float | None
    selected_candidate: bool
    selection_reason: str
    fallback_used: bool
    fallback_reason: str | None
    source_events: list[str]


def resolve_static_champion_policy(
    model_config: ModelConfig,
) -> dict[str, ChampionMethodConfig]:
    """Return the configured static method for each checkpoint."""
    policy = model_config.champion_policy.static
    missing = [checkpoint for checkpoint in CHECKPOINTS if checkpoint not in policy]
    if missing:
        raise ValueError(f"Static champion policy is missing: {', '.join(missing)}")
    return {checkpoint: policy[checkpoint] for checkpoint in CHECKPOINTS}


def load_champion_candidates(
    metrics_dir: Path,
    expected_folds: tuple[BacktestFold, ...],
    *,
    include_temporal_weighted: bool = False,
) -> pd.DataFrame:
    """Load and standardize available out-of-sample prediction families."""
    expected_by_event = {fold.test_event: fold.fold_id for fold in expected_folds}
    artifacts = [
        ("walk_forward", metrics_dir / "walk_forward_predictions.parquet", "uniform"),
        ("ablation", metrics_dir / "ablation_predictions.parquet", "uniform"),
        ("boosted", metrics_dir / "boosted_predictions.parquet", "uniform"),
    ]
    if include_temporal_weighted:
        artifacts.append(
            (
                "ablation",
                metrics_dir
                / f"{temporal_artifact_stem('ablation', 'current_season_only_with_prior')}"
                "_predictions.parquet",
                "current_season_only_with_prior",
            )
        )
    frames: list[pd.DataFrame] = []
    for source, path, temporal_policy in artifacts:
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        frame = _candidate_rows_for_source(frame, source)
        if frame.empty:
            continue
        frame["temporal_weighting_policy"] = temporal_policy
        frame = frame[frame["test_event"].isin(expected_by_event)].copy()
        frame["fold_id"] = frame["test_event"].map(expected_by_event).astype("int64")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            "No walk-forward, ablation, or boosted prediction artifacts are available"
        )
    candidates = pd.concat(frames, ignore_index=True, sort=False)
    required = {
        *PREDICTION_KEY_COLUMNS,
        "test_event",
        "event",
        "team",
        "quali_position",
        "quali_gap_to_pole_sec",
        "reached_q3",
        "predicted_quali_gap_to_pole_sec",
        "predicted_quali_position",
        "predicted_reached_q3",
        *METHOD_COLUMNS,
        "temporal_weighting_policy",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate predictions are missing columns: {', '.join(missing)}")
    candidates["feature_group"] = candidates["feature_group"].astype("string")
    candidates["temporal_weighting_policy"] = candidates["temporal_weighting_policy"].fillna(
        "uniform"
    )
    return candidates.drop_duplicates(
        [*PREDICTION_KEY_COLUMNS, *METHOD_COLUMNS, "temporal_weighting_policy"]
    ).reset_index(drop=True)


def select_nested_method(
    candidates: pd.DataFrame,
    *,
    fold_id: int,
    checkpoint: str,
    fallback: ChampionMethodConfig,
    selection_metric: str = "mae_gap_sec",
) -> tuple[ChampionMethodConfig, float | None, list[str], bool]:
    """Select the best method using only folds strictly before the test fold."""
    if selection_metric != "mae_gap_sec":
        raise ValueError("Only mae_gap_sec nested selection is currently supported")
    current = candidates[
        candidates["fold_id"].eq(fold_id) & candidates["checkpoint"].eq(checkpoint)
    ]
    prior = candidates[candidates["fold_id"].lt(fold_id) & candidates["checkpoint"].eq(checkpoint)]
    if prior.empty:
        return fallback, None, [], True

    available_methods = current.loc[:, METHOD_COLUMNS].drop_duplicates()
    history = prior.merge(available_methods, on=list(METHOD_COLUMNS), how="inner")
    if history.empty:
        return fallback, None, [], True
    history = history.dropna(
        subset=["quali_gap_to_pole_sec", "predicted_quali_gap_to_pole_sec"]
    ).copy()
    if history.empty:
        return fallback, None, [], True
    history["absolute_error"] = (
        history["predicted_quali_gap_to_pole_sec"] - history["quali_gap_to_pole_sec"]
    ).abs()
    metrics = (
        history.groupby(list(METHOD_COLUMNS), dropna=False, sort=False)
        .agg(
            selection_value=("absolute_error", "mean"),
            selection_source_events=("test_event", lambda values: sorted(set(values))),
        )
        .reset_index()
        .sort_values(
            ["selection_value", "candidate_family", "model_name", "feature_group"],
            kind="stable",
            na_position="last",
        )
    )
    if metrics.empty:
        return fallback, None, [], True
    best = metrics.iloc[0]
    method = ChampionMethodConfig(
        family=str(best["candidate_family"]),
        model_name=str(best["model_name"]),
        feature_group=_optional_string(best["feature_group"]),
    )
    return (
        method,
        float(best["selection_value"]),
        list(best["selection_source_events"]),
        False,
    )


def select_stabilized_nested_method(
    candidates: pd.DataFrame,
    *,
    fold_id: int,
    checkpoint: str,
    fallback: ChampionMethodConfig,
    settings: StabilizedNestedChampionConfig,
) -> ChampionSelectionDecision:
    """Select a candidate only when prior evidence clears history and margin gates."""
    if settings.selection_metric != "mae_gap_sec":
        raise ValueError("Only mae_gap_sec stabilized selection is currently supported")
    current = candidates[
        candidates["fold_id"].eq(fold_id) & candidates["checkpoint"].eq(checkpoint)
    ]
    prior = candidates[candidates["fold_id"].lt(fold_id) & candidates["checkpoint"].eq(checkpoint)]
    if current.empty or prior.empty:
        return _fallback_decision(fallback, "insufficient_history", 0, 0)

    available_methods = current.loc[:, METHOD_COLUMNS].drop_duplicates()
    history = prior.merge(available_methods, on=list(METHOD_COLUMNS), how="inner")
    history = history.dropna(
        subset=["quali_gap_to_pole_sec", "predicted_quali_gap_to_pole_sec"]
    ).copy()
    prior_folds = int(history["fold_id"].nunique()) if not history.empty else 0
    prior_predictions = int(len(history))
    if prior_folds < settings.min_prior_folds or prior_predictions < settings.min_prior_predictions:
        return _fallback_decision(
            fallback,
            "insufficient_history",
            prior_folds,
            prior_predictions,
        )

    history["absolute_error"] = (
        history["predicted_quali_gap_to_pole_sec"] - history["quali_gap_to_pole_sec"]
    ).abs()
    metrics = (
        history.groupby(list(METHOD_COLUMNS), dropna=False, sort=False)
        .agg(
            selected_metric_value=("absolute_error", "mean"),
            selection_source_events=("test_event", lambda values: sorted(set(values))),
        )
        .reset_index()
        .sort_values(
            ["selected_metric_value", "candidate_family", "model_name", "feature_group"],
            kind="stable",
            na_position="last",
        )
    )
    if metrics.empty:
        return _fallback_decision(
            fallback,
            "insufficient_history",
            prior_folds,
            prior_predictions,
        )

    default_history = _method_rows(history, fallback, checkpoint)
    default_folds = int(default_history["fold_id"].nunique()) if not default_history.empty else 0
    default_predictions = int(len(default_history))
    if (
        default_folds < settings.min_prior_folds
        or default_predictions < settings.min_prior_predictions
    ):
        return _fallback_decision(
            fallback,
            "insufficient_default_history",
            prior_folds,
            prior_predictions,
        )
    default_metric = float(default_history["absolute_error"].mean())

    best = metrics.iloc[0]
    candidate_metric = float(best["selected_metric_value"])
    candidate = ChampionMethodConfig(
        family=str(best["candidate_family"]),
        model_name=str(best["model_name"]),
        feature_group=_optional_string(best["feature_group"]),
    )
    if candidate_metric <= default_metric - settings.improvement_margin_sec:
        return ChampionSelectionDecision(
            selected=candidate,
            selected_metric_value=candidate_metric,
            default_metric_value=default_metric,
            source_events=list(best["selection_source_events"]),
            prior_folds_used=prior_folds,
            prior_predictions_used=prior_predictions,
            fallback_used=False,
            fallback_reason=None,
        )
    return ChampionSelectionDecision(
        selected=fallback,
        selected_metric_value=default_metric,
        default_metric_value=default_metric,
        source_events=sorted(default_history["test_event"].dropna().astype(str).unique().tolist()),
        prior_folds_used=prior_folds,
        prior_predictions_used=prior_predictions,
        fallback_used=True,
        fallback_reason="hysteresis_margin_not_met",
    )


def apply_stabilized_nested_guardrail(
    *,
    selected: ChampionMethodConfig,
    fallback: ChampionMethodConfig,
    checkpoint: str,
    settings: StabilizedNestedGuardedChampionConfig,
) -> ChampionGuardrailDecision:
    """Apply the opt-in FP3 no-baseline-switch guardrail to a stabilized decision."""
    if (
        not settings.fp3_no_baseline_switch
        or checkpoint != settings.guarded_checkpoint
        or not _matches_guarded_default(fallback, settings)
        or _same_method(selected, fallback)
        or not is_practice_baseline_method(selected)
    ):
        return ChampionGuardrailDecision(
            selected=selected,
            pre_guardrail_selected=selected,
            guardrail_applied=False,
            guardrail_name=None,
            guardrail_reason=None,
        )
    return ChampionGuardrailDecision(
        selected=fallback,
        pre_guardrail_selected=selected,
        guardrail_applied=True,
        guardrail_name="fp3_no_baseline_switch",
        guardrail_reason="prevent_fp3_baseline_switch_from_static_rf",
    )


def select_season_aware_guarded_method(
    candidates: pd.DataFrame,
    *,
    fold: BacktestFold,
    checkpoint: str,
    default_method: ChampionMethodConfig,
    settings: SeasonAwareNestedGuardedChampionConfig,
) -> SeasonAwareChampionDecision:
    """Gate the opt-in season-aware FP3 RF candidate using only prior folds."""
    prior_same_season_events = _current_season_prior_event_count(fold)
    if checkpoint != settings.eligible_checkpoint:
        return _season_aware_decision(
            selected=default_method,
            current_season_prior_event_count=prior_same_season_events,
            reason="not_applicable_checkpoint",
        )

    candidate = settings.required_candidate
    current_rows = _method_rows(
        candidates[candidates["fold_id"].eq(fold.fold_id)],
        candidate,
        checkpoint,
    )
    candidate_available = not current_rows.empty
    if not candidate_available:
        return _season_aware_decision(
            selected=default_method,
            current_season_prior_event_count=prior_same_season_events,
            candidate_available=False,
            reason="weighted_candidate_missing",
            fallback_reason="weighted_candidate_missing",
        )
    if prior_same_season_events < settings.min_current_season_prior_events:
        return _season_aware_decision(
            selected=default_method,
            current_season_prior_event_count=prior_same_season_events,
            candidate_available=True,
            reason="cold_start",
            fallback_reason="season_aware_cold_start",
        )

    prior = candidates[candidates["fold_id"].lt(fold.fold_id)].copy()
    candidate_history = _method_rows(prior, candidate, checkpoint).dropna(
        subset=["quali_gap_to_pole_sec", "predicted_quali_gap_to_pole_sec"]
    )
    candidate_folds = (
        int(candidate_history["fold_id"].nunique()) if not candidate_history.empty else 0
    )
    candidate_predictions = int(len(candidate_history))
    if (
        candidate_folds < settings.min_prior_candidate_folds
        or candidate_predictions < settings.min_prior_candidate_predictions
    ):
        return _season_aware_decision(
            selected=default_method,
            current_season_prior_event_count=prior_same_season_events,
            candidate_available=True,
            prior_candidate_folds=candidate_folds,
            prior_candidate_predictions=candidate_predictions,
            reason="insufficient_candidate_history",
            fallback_reason="insufficient_candidate_history",
        )

    default_history = _method_rows(prior, default_method, checkpoint).dropna(
        subset=["quali_gap_to_pole_sec", "predicted_quali_gap_to_pole_sec"]
    )
    if default_history.empty:
        return _season_aware_decision(
            selected=default_method,
            current_season_prior_event_count=prior_same_season_events,
            candidate_available=True,
            prior_candidate_folds=candidate_folds,
            prior_candidate_predictions=candidate_predictions,
            reason="default_retained",
            fallback_reason="insufficient_default_history",
        )

    candidate_metric = _mean_absolute_error(candidate_history)
    default_metric = _mean_absolute_error(default_history)
    if candidate_metric <= default_metric - settings.improvement_margin_sec:
        return _season_aware_decision(
            selected=candidate,
            current_season_prior_event_count=prior_same_season_events,
            candidate_available=True,
            candidate_eligible=True,
            prior_candidate_folds=candidate_folds,
            prior_candidate_predictions=candidate_predictions,
            candidate_metric_value=candidate_metric,
            default_metric_value=default_metric,
            selected_candidate=True,
            reason="selected_after_prior_evidence",
            fallback_used=False,
            source_events=sorted(
                candidate_history["test_event"].dropna().astype(str).unique().tolist()
            ),
        )
    return _season_aware_decision(
        selected=default_method,
        current_season_prior_event_count=prior_same_season_events,
        candidate_available=True,
        candidate_eligible=True,
        prior_candidate_folds=candidate_folds,
        prior_candidate_predictions=candidate_predictions,
        candidate_metric_value=candidate_metric,
        default_metric_value=default_metric,
        reason="margin_not_met",
        fallback_reason="margin_not_met",
    )


def _season_aware_decision(
    *,
    selected: ChampionMethodConfig,
    current_season_prior_event_count: int,
    reason: str,
    candidate_available: bool = True,
    candidate_eligible: bool = False,
    prior_candidate_folds: int = 0,
    prior_candidate_predictions: int = 0,
    candidate_metric_value: float | None = None,
    default_metric_value: float | None = None,
    selected_candidate: bool = False,
    fallback_used: bool = True,
    fallback_reason: str | None = None,
    source_events: list[str] | None = None,
) -> SeasonAwareChampionDecision:
    return SeasonAwareChampionDecision(
        selected=selected,
        season_aware_candidate_available=candidate_available,
        season_aware_candidate_eligible=candidate_eligible,
        current_season_prior_event_count=current_season_prior_event_count,
        prior_candidate_folds=prior_candidate_folds,
        prior_candidate_predictions=prior_candidate_predictions,
        candidate_metric_value=candidate_metric_value,
        default_metric_value=default_metric_value,
        selected_candidate=selected_candidate,
        selection_reason=reason,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        source_events=source_events or [],
    )


def is_practice_baseline_method(method: ChampionMethodConfig) -> bool:
    """Return whether a candidate identity represents a practice-lap baseline."""
    family = method.family.lower()
    model_name = method.model_name.lower()
    baseline_tokens = (
        "best_push_lap",
        "best_valid_lap",
        "theoretical_best_lap",
        "baseline",
    )
    return family in {"baseline", "robust_baseline"} or any(
        token in model_name for token in baseline_tokens
    )


def assign_predicted_gap_bucket(
    predicted_gap_sec: object,
    settings: PredictedGapBucketUncertaintyConfig | None = None,
) -> str | None:
    """Assign a prediction-only qualifying-gap regime bucket."""
    if predicted_gap_sec is None or pd.isna(predicted_gap_sec):
        return None
    thresholds = (
        settings.bucket_thresholds_sec
        if settings is not None
        else PredictedGapBucketUncertaintyConfig().bucket_thresholds_sec
    )
    gap = float(predicted_gap_sec)
    if gap <= thresholds["pole_contender"]:
        return "pole_contender"
    if gap <= thresholds["close_midfield"]:
        return "close_midfield"
    if gap <= thresholds["midfield"]:
        return "midfield"
    return "backmarker_or_outlier"


def add_prior_residual_uncertainty(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    config: UncertaintyConfig,
    *,
    method: ChampionUncertaintyMethod | str = ChampionUncertaintyMethod.residual_std,
) -> pd.DataFrame:
    """Attach intervals estimated only from earlier folds for the selected method."""
    method = ChampionUncertaintyMethod(method)
    if method is ChampionUncertaintyMethod.conformal_predicted_gap_bucket:
        return add_predicted_gap_bucket_conformal_uncertainty(predictions, candidates, config)
    result = predictions.copy()
    result["prediction_interval_low_sec"] = float("nan")
    result["prediction_interval_high_sec"] = float("nan")
    result["residual_std_sec"] = float("nan")
    result["residual_count"] = 0
    result["residual_quantile_sec"] = float("nan")
    result["uncertainty_confidence_level"] = config.confidence_level
    result["interval_contains_actual"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result["uncertainty_method"] = "insufficient_history"
    group_columns = [
        "fold_id",
        "checkpoint",
        "selected_family",
        "selected_model_name",
        "selected_feature_group",
    ]
    if "selected_temporal_weighting_policy" in result.columns:
        group_columns.append("selected_temporal_weighting_policy")
    for keys, rows in result.groupby(group_columns, dropna=False, sort=True):
        if len(group_columns) == 6:
            fold_id, checkpoint, family, model_name, feature_group, temporal_policy = keys
        else:
            fold_id, checkpoint, family, model_name, feature_group = keys
            temporal_policy = "uniform"
        history = _method_rows(
            candidates[candidates["fold_id"].lt(int(fold_id))],
            ChampionMethodConfig(
                family=str(family),
                model_name=str(model_name),
                feature_group=_optional_string(feature_group),
                temporal_weighting_policy=_optional_string(temporal_policy),
            ),
            str(checkpoint),
        ).dropna(subset=["quali_gap_to_pole_sec", "predicted_quali_gap_to_pole_sec"])
        if len(history) < config.min_residual_count:
            continue
        residuals = (
            history["quali_gap_to_pole_sec"] - history["predicted_quali_gap_to_pole_sec"]
        ).astype(float)
        residual_std = float(residuals.std(ddof=1))
        if pd.isna(residual_std):
            continue
        result.loc[rows.index, "residual_count"] = int(len(residuals))
        result.loc[rows.index, "residual_std_sec"] = residual_std
        if method is ChampionUncertaintyMethod.conformal:
            half_width = float(
                residuals.abs().quantile(config.confidence_level, interpolation="higher")
            )
            result.loc[rows.index, "residual_quantile_sec"] = half_width
            uncertainty_label = "conformal"
        else:
            half_width = config.interval_z * residual_std
            uncertainty_label = "residual_std"
        predicted = result.loc[rows.index, "predicted_quali_gap_to_pole_sec"].astype(float)
        low = predicted - half_width
        high = predicted + half_width
        result.loc[rows.index, "prediction_interval_low_sec"] = low
        result.loc[rows.index, "prediction_interval_high_sec"] = high
        result.loc[rows.index, "uncertainty_method"] = uncertainty_label
        actual = result.loc[rows.index, "quali_gap_to_pole_sec"].astype(float)
        result.loc[rows.index, "interval_contains_actual"] = (actual >= low) & (actual <= high)
    return result


def add_predicted_gap_bucket_conformal_uncertainty(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    config: UncertaintyConfig,
) -> pd.DataFrame:
    """Attach conformal intervals calibrated by prior predicted-gap regimes."""
    settings = config.predicted_gap_bucket
    result = predictions.copy()
    result["prediction_interval_low_sec"] = float("nan")
    result["prediction_interval_high_sec"] = float("nan")
    result["residual_std_sec"] = float("nan")
    result["residual_count"] = 0
    result["residual_quantile_sec"] = float("nan")
    result["uncertainty_confidence_level"] = settings.confidence_level
    result["interval_contains_actual"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result["uncertainty_method"] = "insufficient_history"
    result["predicted_gap_bucket"] = result["predicted_quali_gap_to_pole_sec"].map(
        lambda value: assign_predicted_gap_bucket(value, settings)
    )
    result["uncertainty_calibration_level"] = "insufficient_history"
    result["uncertainty_fallback_used"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result["uncertainty_fallback_reason"] = pd.NA
    result["uncertainty_calibration_group"] = pd.NA
    result["uncertainty_prior_group_count"] = 0

    history = _candidate_residual_history(candidates, settings)
    if history.empty:
        return result

    for index, row in result.iterrows():
        bucket = row["predicted_gap_bucket"]
        if bucket is None or pd.isna(bucket):
            continue
        prior = history[history["fold_id"].lt(int(row["fold_id"]))]
        if prior.empty:
            continue
        method_config = ChampionMethodConfig(
            family=str(row["selected_family"]),
            model_name=str(row["selected_model_name"]),
            feature_group=_optional_string(row["selected_feature_group"]),
            temporal_weighting_policy=_optional_string(
                row.get("selected_temporal_weighting_policy", "uniform")
            ),
        )
        selected = _select_predicted_bucket_quantile(
            prior,
            checkpoint=str(row["checkpoint"]),
            method=method_config,
            predicted_gap_bucket=str(bucket),
            settings=settings,
        )
        if selected is None:
            continue
        level, group, residuals = selected
        quantile = float(
            residuals.abs().quantile(settings.confidence_level, interpolation="higher")
        )
        residual_std = float(residuals.std(ddof=1))
        predicted = float(row["predicted_quali_gap_to_pole_sec"])
        low = predicted - quantile
        high = predicted + quantile
        actual = float(row["quali_gap_to_pole_sec"])
        result.loc[index, "prediction_interval_low_sec"] = low
        result.loc[index, "prediction_interval_high_sec"] = high
        result.loc[index, "residual_std_sec"] = None if pd.isna(residual_std) else residual_std
        result.loc[index, "residual_count"] = int(len(residuals))
        result.loc[index, "residual_quantile_sec"] = quantile
        result.loc[index, "interval_contains_actual"] = bool(low <= actual <= high)
        result.loc[index, "uncertainty_method"] = (
            ChampionUncertaintyMethod.conformal_predicted_gap_bucket.value
        )
        result.loc[index, "uncertainty_calibration_level"] = level
        result.loc[index, "uncertainty_fallback_used"] = level != settings.fallback_order[0]
        result.loc[index, "uncertainty_fallback_reason"] = (
            None if level == settings.fallback_order[0] else "fallback_to_coarser_group"
        )
        result.loc[index, "uncertainty_calibration_group"] = group
        result.loc[index, "uncertainty_prior_group_count"] = int(len(residuals))
    return result


def run_champion_backtest(
    config: DataConfig,
    *,
    strategy: BacktestStrategy | str = BacktestStrategy.walk_forward,
    selection_mode: ChampionSelectionMode | str = ChampionSelectionMode.nested,
    dataset_path: Path | None = None,
    min_events: int = 10,
    min_train_events: int = 5,
    model_config: ModelConfig,
    uncertainty_method: ChampionUncertaintyMethod | str = ChampionUncertaintyMethod.residual_std,
) -> ChampionBacktestSummary:
    """Evaluate static or nested checkpoint champions on walk-forward folds."""
    strategy = BacktestStrategy(strategy)
    if strategy is not BacktestStrategy.walk_forward:
        raise ValueError("Champion backtesting currently supports walk_forward only")
    selection_mode = ChampionSelectionMode(selection_mode)
    uncertainty_method = ChampionUncertaintyMethod(uncertainty_method)
    source_path = _resolve_dataset_path(config, dataset_path)
    dataset = pd.read_parquet(source_path).reset_index(drop=True)
    event_keys = ordered_event_keys(dataset)
    paths = _output_paths(config.metrics_output_dir, selection_mode)
    ensure_directory(config.metrics_output_dir)
    if len(event_keys) < min_events:
        reason = f"Dataset has {len(event_keys)} unique events; at least {min_events} are required"
        skipped = _skipped_payload(strategy, selection_mode, len(event_keys), reason)
        _write_mode_outputs(paths, skipped, None, None)
        return ChampionBacktestSummary(
            status="skipped",
            strategy=strategy.value,
            selection_mode=selection_mode.value,
            n_events=len(event_keys),
            n_folds_total=0,
            n_folds_successful=0,
            n_folds_failed=0,
            prediction_rows=0,
            metrics_path=paths["metrics"],
            predictions_path=None,
            selection_path=None,
        )
    folds = build_backtest_folds(
        dataset,
        strategy,
        min_train_events=min_train_events,
    )
    include_temporal_weighted = selection_mode is ChampionSelectionMode.season_aware_nested_guarded
    candidates = load_champion_candidates(
        config.metrics_output_dir,
        folds,
        include_temporal_weighted=include_temporal_weighted,
    )
    base_candidates = candidates[
        candidates.get("temporal_weighting_policy", "uniform").fillna("uniform").eq("uniform")
    ].copy()
    static_policy = resolve_static_champion_policy(model_config)
    prediction_frames: list[pd.DataFrame] = []
    selection_records: list[dict[str, object]] = []
    failed_folds = 0

    for fold in folds:
        fold_frames: list[pd.DataFrame] = []
        try:
            for checkpoint in CHECKPOINTS:
                fallback = static_policy[checkpoint]
                default_metric_value = None
                prior_folds_used = 0
                prior_predictions_used = 0
                fallback_reason = None
                season_aware_metadata = _default_season_aware_metadata(
                    fold=fold,
                    checkpoint=checkpoint,
                    settings=model_config.champion_policy.season_aware_nested_guarded,
                )
                if selection_mode is ChampionSelectionMode.static:
                    selected = fallback
                    selection_value = None
                    source_events: list[str] = []
                    fallback_used = False
                elif selection_mode in {
                    ChampionSelectionMode.stabilized_nested,
                    ChampionSelectionMode.stabilized_nested_guarded,
                    ChampionSelectionMode.season_aware_nested_guarded,
                }:
                    season_aware_settings = model_config.champion_policy.season_aware_nested_guarded
                    if (
                        selection_mode is ChampionSelectionMode.season_aware_nested_guarded
                        and checkpoint != season_aware_settings.eligible_checkpoint
                    ):
                        selected = fallback
                        selection_value = None
                        source_events = []
                        fallback_used = False
                    else:
                        decision = select_stabilized_nested_method(
                            base_candidates,
                            fold_id=fold.fold_id,
                            checkpoint=checkpoint,
                            fallback=fallback,
                            settings=model_config.champion_policy.stabilized_nested,
                        )
                        selected = decision.selected
                        selection_value = decision.selected_metric_value
                        default_metric_value = decision.default_metric_value
                        source_events = decision.source_events
                        prior_folds_used = decision.prior_folds_used
                        prior_predictions_used = decision.prior_predictions_used
                        fallback_used = decision.fallback_used
                        fallback_reason = decision.fallback_reason
                else:
                    selected, selection_value, source_events, fallback_used = select_nested_method(
                        base_candidates,
                        fold_id=fold.fold_id,
                        checkpoint=checkpoint,
                        fallback=fallback,
                        selection_metric=model_config.champion_policy.selection_metric,
                    )
                guardrail_decision = ChampionGuardrailDecision(
                    selected=selected,
                    pre_guardrail_selected=selected,
                    guardrail_applied=False,
                    guardrail_name=None,
                    guardrail_reason=None,
                )
                if selection_mode in {
                    ChampionSelectionMode.stabilized_nested_guarded,
                    ChampionSelectionMode.season_aware_nested_guarded,
                }:
                    guardrail_decision = apply_stabilized_nested_guardrail(
                        selected=selected,
                        fallback=fallback,
                        checkpoint=checkpoint,
                        settings=model_config.champion_policy.stabilized_nested_guarded,
                    )
                    selected = guardrail_decision.selected
                post_guardrail_selected = selected
                if selection_mode is ChampionSelectionMode.season_aware_nested_guarded:
                    season_decision = select_season_aware_guarded_method(
                        candidates,
                        fold=fold,
                        checkpoint=checkpoint,
                        default_method=selected,
                        settings=model_config.champion_policy.season_aware_nested_guarded,
                    )
                    selected = season_decision.selected
                    season_aware_metadata = _season_aware_metadata(
                        season_decision,
                        model_config.champion_policy.season_aware_nested_guarded,
                    )
                    if checkpoint == (
                        model_config.champion_policy.season_aware_nested_guarded.eligible_checkpoint
                    ):
                        selection_value = season_decision.candidate_metric_value
                        default_metric_value = season_decision.default_metric_value
                        source_events = season_decision.source_events
                        fallback_used = season_decision.fallback_used
                        fallback_reason = season_decision.fallback_reason
                rows = _method_rows(
                    candidates[candidates["fold_id"].eq(fold.fold_id)],
                    selected,
                    checkpoint,
                )
                if rows.empty:
                    raise ValueError(
                        f"Selected method unavailable for fold {fold.fold_id} {checkpoint}: "
                        f"{selected.family}/{selected.model_name}/{selected.feature_group}"
                    )
                fold_frames.append(_champion_prediction_rows(rows, selection_mode, selected))
                selection_records.append(
                    {
                        "selection_mode": selection_mode.value,
                        "fold_id": fold.fold_id,
                        "test_event": fold.test_event,
                        "checkpoint": checkpoint,
                        "selected_family": selected.family,
                        "selected_model_name": selected.model_name,
                        "selected_feature_group": selected.feature_group,
                        "selected_temporal_weighting_policy": (
                            selected.temporal_weighting_policy or "uniform"
                        ),
                        "default_family": fallback.family,
                        "default_model_name": fallback.model_name,
                        "default_feature_group": fallback.feature_group,
                        "selection_metric": model_config.champion_policy.selection_metric,
                        "selected_metric_value": selection_value,
                        "default_metric_value": default_metric_value,
                        "selection_value": selection_value,
                        "improvement_margin_sec": (
                            model_config.champion_policy.stabilized_nested.improvement_margin_sec
                        ),
                        "min_prior_folds": (
                            model_config.champion_policy.stabilized_nested.min_prior_folds
                        ),
                        "min_prior_predictions": (
                            model_config.champion_policy.stabilized_nested.min_prior_predictions
                        ),
                        "prior_folds_used": prior_folds_used,
                        "prior_predictions_used": prior_predictions_used,
                        "selection_source_events": source_events,
                        "fallback_used": fallback_used,
                        "fallback_reason": fallback_reason,
                        "guardrail_applied": guardrail_decision.guardrail_applied,
                        "guardrail_name": guardrail_decision.guardrail_name,
                        "guardrail_reason": guardrail_decision.guardrail_reason,
                        "pre_guardrail_selected_family": (
                            guardrail_decision.pre_guardrail_selected.family
                        ),
                        "pre_guardrail_selected_model_name": (
                            guardrail_decision.pre_guardrail_selected.model_name
                        ),
                        "pre_guardrail_selected_feature_group": (
                            guardrail_decision.pre_guardrail_selected.feature_group
                        ),
                        "post_guardrail_selected_family": post_guardrail_selected.family,
                        "post_guardrail_selected_model_name": post_guardrail_selected.model_name,
                        "post_guardrail_selected_feature_group": (
                            post_guardrail_selected.feature_group
                        ),
                        **season_aware_metadata,
                    }
                )
            prediction_frames.extend(fold_frames)
        except Exception:
            failed_folds += 1
            selection_records = [
                record for record in selection_records if record["fold_id"] != fold.fold_id
            ]

    if not prediction_frames:
        payload = {
            "status": "failed",
            "strategy": strategy.value,
            "selection_mode": selection_mode.value,
            "n_events": len(event_keys),
            "n_folds_total": len(folds),
            "n_folds_successful": 0,
            "n_folds_failed": failed_folds,
            "created_at_utc": _utc_now(),
        }
        _write_mode_outputs(paths, payload, None, None)
        return ChampionBacktestSummary(
            status="failed",
            strategy=strategy.value,
            selection_mode=selection_mode.value,
            n_events=len(event_keys),
            n_folds_total=len(folds),
            n_folds_successful=0,
            n_folds_failed=failed_folds,
            prediction_rows=0,
            metrics_path=paths["metrics"],
            predictions_path=None,
            selection_path=None,
        )

    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    successful_fold_ids = sorted(predictions["fold_id"].unique().tolist())
    predictions = add_prior_residual_uncertainty(
        predictions,
        candidates,
        model_config.uncertainty,
        method=uncertainty_method,
    )
    selection = pd.DataFrame(selection_records)
    payload = build_champion_metrics_payload(
        strategy,
        selection_mode,
        len(event_keys),
        len(folds),
        failed_folds,
        predictions,
        candidates[candidates["fold_id"].isin(successful_fold_ids)],
        uncertainty_method=uncertainty_method,
    )
    _write_mode_outputs(paths, payload, predictions, selection)
    return ChampionBacktestSummary(
        status=str(payload["status"]),
        strategy=strategy.value,
        selection_mode=selection_mode.value,
        n_events=len(event_keys),
        n_folds_total=len(folds),
        n_folds_successful=len(successful_fold_ids),
        n_folds_failed=failed_folds,
        prediction_rows=len(predictions),
        metrics_path=paths["metrics"],
        predictions_path=paths["predictions"],
        selection_path=paths["selection"],
    )


def build_champion_metrics_payload(
    strategy: BacktestStrategy | str,
    selection_mode: ChampionSelectionMode | str,
    n_events: int,
    n_folds_total: int,
    n_folds_failed: int,
    champion_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    *,
    uncertainty_method: ChampionUncertaintyMethod | str = ChampionUncertaintyMethod.residual_std,
) -> dict[str, object]:
    """Compute champion, baseline, and fixed-method comparisons."""
    strategy = BacktestStrategy(strategy)
    selection_mode = ChampionSelectionMode(selection_mode)
    uncertainty_method = ChampionUncertaintyMethod(uncertainty_method)
    checkpoints = champion_predictions["checkpoint"].drop_duplicates().astype(str).tolist()
    champion_metrics = {}
    for checkpoint, rows in champion_predictions.groupby("checkpoint", sort=False):
        metrics = compute_prediction_metrics(rows)
        metrics.update(_interval_metrics(rows))
        champion_metrics[str(checkpoint)] = metrics
    best_baselines = _best_candidate_by_checkpoint(
        candidate_predictions[
            candidate_predictions["candidate_family"].isin(["baseline", "robust_baseline"])
        ],
        checkpoints,
        champion_predictions,
    )
    best_single = _best_candidate_by_checkpoint(
        candidate_predictions,
        checkpoints,
        champion_predictions,
    )
    baseline_mae: dict[str, float | None] = {}
    baseline_position: dict[str, float | None] = {}
    single_mae: dict[str, float | None] = {}
    for checkpoint in checkpoints:
        champion = champion_metrics.get(checkpoint, {})
        baseline = best_baselines.get(checkpoint, {})
        single = best_single.get(checkpoint, {})
        baseline_mae[checkpoint] = _delta(champion.get("mae_gap_sec"), baseline.get("mae_gap_sec"))
        baseline_position[checkpoint] = _delta(
            champion.get("mean_abs_position_error"),
            baseline.get("mean_abs_position_error"),
        )
        single_mae[checkpoint] = _delta(champion.get("mae_gap_sec"), single.get("mae_gap_sec"))
    return {
        "status": "complete" if n_folds_failed == 0 else "partial",
        "strategy": strategy.value,
        "selection_mode": selection_mode.value,
        "uncertainty_method": uncertainty_method.value,
        "n_events": n_events,
        "n_folds_total": n_folds_total,
        "n_folds_successful": int(champion_predictions["fold_id"].nunique()),
        "n_folds_failed": n_folds_failed,
        "checkpoints": checkpoints,
        "metrics_by_checkpoint": champion_metrics,
        "best_baseline_by_checkpoint": best_baselines,
        "best_single_family_by_checkpoint": best_single,
        "champion_vs_best_baseline_delta_mae": baseline_mae,
        "champion_vs_best_baseline_delta_position_error": baseline_position,
        "champion_vs_best_single_family_delta_mae": single_mae,
        "interval_metrics_by_predicted_gap_bucket": _interval_metrics_by_predicted_gap_bucket(
            champion_predictions
        ),
        "created_at_utc": _utc_now(),
    }


def _candidate_rows_for_source(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    result = frame.copy()
    if source == "walk_forward":
        result = result[result["prediction_type"].isin(["tabular", "baseline"])].copy()
        result = result[
            result["prediction_type"].eq("baseline")
            | result["model_name"].isin(["ridge", "random_forest"])
        ].copy()
        result["candidate_family"] = result.apply(_walk_forward_family, axis=1)
        result["feature_group"] = pd.NA
    elif source == "ablation":
        result = result[result["prediction_type"].eq("tabular")].copy()
        result["candidate_family"] = "ablation"
    else:
        result = result[result["prediction_type"].eq("boosted")].copy()
        result["candidate_family"] = "boosted"
    return result


def _walk_forward_family(row: pd.Series) -> str:
    if row["prediction_type"] == "baseline":
        return "robust_baseline" if str(row["model_name"]).startswith("robust_") else "baseline"
    return "tabular"


def _method_rows(
    candidates: pd.DataFrame,
    method: ChampionMethodConfig,
    checkpoint: str,
) -> pd.DataFrame:
    feature_group = candidates["feature_group"].fillna("")
    expected_group = method.feature_group or ""
    rows = candidates[
        candidates["checkpoint"].eq(checkpoint)
        & candidates["candidate_family"].eq(method.family)
        & candidates["model_name"].eq(method.model_name)
        & feature_group.eq(expected_group)
    ].copy()
    if "temporal_weighting_policy" in rows.columns:
        expected_policy = method.temporal_weighting_policy or "uniform"
        rows = rows[rows["temporal_weighting_policy"].fillna("uniform").eq(expected_policy)]
    return rows


def _matches_guarded_default(
    method: ChampionMethodConfig,
    settings: StabilizedNestedGuardedChampionConfig,
) -> bool:
    return (
        method.family == settings.guarded_default_family
        and method.model_name == settings.guarded_default_model_name
        and (method.feature_group or "") == (settings.guarded_default_feature_group or "")
    )


def _same_method(first: ChampionMethodConfig, second: ChampionMethodConfig) -> bool:
    return (
        first.family == second.family
        and first.model_name == second.model_name
        and (first.feature_group or "") == (second.feature_group or "")
        and (first.temporal_weighting_policy or "uniform")
        == (second.temporal_weighting_policy or "uniform")
    )


def _champion_prediction_rows(
    rows: pd.DataFrame,
    selection_mode: ChampionSelectionMode,
    method: ChampionMethodConfig,
) -> pd.DataFrame:
    columns = [
        "strategy",
        "fold_id",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "driver",
        "team",
        "quali_position",
        "quali_gap_to_pole_sec",
        "reached_q3",
        "predicted_quali_gap_to_pole_sec",
        "predicted_quali_position",
        "predicted_reached_q3",
    ]
    result = rows.loc[:, columns].copy()
    result["selection_mode"] = selection_mode.value
    result["selected_family"] = method.family
    result["selected_model_name"] = method.model_name
    result["selected_feature_group"] = method.feature_group
    result["selected_temporal_weighting_policy"] = method.temporal_weighting_policy or "uniform"
    result["temporal_weighting_policy"] = method.temporal_weighting_policy or "uniform"
    return result


def _best_candidate_by_checkpoint(
    candidates: pd.DataFrame,
    checkpoints: list[str],
    reference_predictions: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for checkpoint in checkpoints:
        checkpoint_rows = candidates[candidates["checkpoint"].eq(checkpoint)]
        reference_keys = (
            reference_predictions[reference_predictions["checkpoint"].eq(checkpoint)]
            .loc[:, PREDICTION_KEY_COLUMNS]
            .drop_duplicates()
        )
        expected_keys = set(
            map(
                tuple,
                reference_keys.itertuples(index=False, name=None),
            )
        )
        choices: list[tuple[ChampionMethodConfig, dict[str, float | None]]] = []
        group_columns = list(METHOD_COLUMNS)
        if "temporal_weighting_policy" in checkpoint_rows.columns:
            group_columns.append("temporal_weighting_policy")
        for keys, rows in checkpoint_rows.groupby(group_columns, dropna=False, sort=False):
            method_keys = set(
                map(
                    tuple,
                    rows.loc[:, PREDICTION_KEY_COLUMNS]
                    .drop_duplicates()
                    .itertuples(index=False, name=None),
                )
            )
            if not expected_keys.issubset(method_keys):
                continue
            rows = rows.merge(
                reference_keys,
                on=list(PREDICTION_KEY_COLUMNS),
                how="inner",
            )
            if len(group_columns) == 4:
                family, model_name, feature_group, temporal_policy = keys
            else:
                family, model_name, feature_group = keys
                temporal_policy = "uniform"
            choices.append(
                (
                    ChampionMethodConfig(
                        family=str(family),
                        model_name=str(model_name),
                        feature_group=_optional_string(feature_group),
                        temporal_weighting_policy=_optional_string(temporal_policy),
                    ),
                    compute_prediction_metrics(rows),
                )
            )
        choices = [choice for choice in choices if choice[1].get("mae_gap_sec") is not None]
        if choices:
            method, metrics = min(
                choices,
                key=lambda choice: float(choice[1]["mae_gap_sec"]),
            )
            best[checkpoint] = {
                "family": method.family,
                "model_name": method.model_name,
                "feature_group": method.feature_group,
                "temporal_weighting_policy": method.temporal_weighting_policy or "uniform",
                "mae_gap_sec": metrics.get("mae_gap_sec"),
                "mean_abs_position_error": metrics.get("mean_abs_position_error"),
            }
    return best


def _fallback_decision(
    fallback: ChampionMethodConfig,
    reason: str,
    prior_folds_used: int,
    prior_predictions_used: int,
) -> ChampionSelectionDecision:
    return ChampionSelectionDecision(
        selected=fallback,
        selected_metric_value=None,
        default_metric_value=None,
        source_events=[],
        prior_folds_used=prior_folds_used,
        prior_predictions_used=prior_predictions_used,
        fallback_used=True,
        fallback_reason=reason,
    )


def _default_season_aware_metadata(
    *,
    fold: BacktestFold,
    checkpoint: str,
    settings: SeasonAwareNestedGuardedChampionConfig,
) -> dict[str, object]:
    reason = (
        "default_retained"
        if checkpoint == settings.eligible_checkpoint
        else "not_applicable_checkpoint"
    )
    return _season_aware_metadata(
        _season_aware_decision(
            selected=settings.required_candidate,
            current_season_prior_event_count=_current_season_prior_event_count(fold),
            candidate_available=False,
            reason=reason,
        ),
        settings,
    )


def _season_aware_metadata(
    decision: SeasonAwareChampionDecision,
    settings: SeasonAwareNestedGuardedChampionConfig,
) -> dict[str, object]:
    return {
        "current_season_prior_event_count": decision.current_season_prior_event_count,
        "season_aware_candidate_available": decision.season_aware_candidate_available,
        "season_aware_candidate_eligible": decision.season_aware_candidate_eligible,
        "season_aware_candidate_prior_folds": decision.prior_candidate_folds,
        "season_aware_candidate_prior_predictions": decision.prior_candidate_predictions,
        "season_aware_candidate_metric_value": decision.candidate_metric_value,
        "season_aware_default_metric_value": decision.default_metric_value,
        "season_aware_improvement_margin_sec": settings.improvement_margin_sec,
        "season_aware_selected": decision.selected_candidate,
        "season_aware_selection_reason": decision.selection_reason,
        "temporal_weighting_policy": decision.selected.temporal_weighting_policy or "uniform",
        "temporal_weighting_config_summary": _temporal_weighting_config_summary(settings),
    }


def _temporal_weighting_config_summary(
    settings: SeasonAwareNestedGuardedChampionConfig,
) -> str:
    candidate = settings.required_candidate
    return (
        f"{candidate.temporal_weighting_policy or 'uniform'};"
        f"min_current_season_prior_events={settings.min_current_season_prior_events};"
        f"min_prior_candidate_folds={settings.min_prior_candidate_folds};"
        f"min_prior_candidate_predictions={settings.min_prior_candidate_predictions};"
        f"improvement_margin_sec={settings.improvement_margin_sec}"
    )


def _interval_metrics(rows: pd.DataFrame) -> dict[str, float | None]:
    required = {
        "prediction_interval_low_sec",
        "prediction_interval_high_sec",
        "interval_contains_actual",
    }
    if not required <= set(rows.columns):
        return {
            "interval_coverage": None,
            "mean_interval_width_sec": None,
            "median_interval_width_sec": None,
            "interval_availability_rate": 0.0,
        }
    intervals = rows[
        rows["prediction_interval_low_sec"].notna() & rows["prediction_interval_high_sec"].notna()
    ].copy()
    availability = float(len(intervals) / len(rows)) if len(rows) else None
    if intervals.empty:
        return {
            "interval_coverage": None,
            "mean_interval_width_sec": None,
            "median_interval_width_sec": None,
            "interval_availability_rate": availability,
        }
    widths = intervals["prediction_interval_high_sec"].astype(float) - intervals[
        "prediction_interval_low_sec"
    ].astype(float)
    contains = intervals["interval_contains_actual"].dropna()
    coverage = float(contains.astype(bool).mean()) if not contains.empty else None
    return {
        "interval_coverage": coverage,
        "mean_interval_width_sec": float(widths.mean()),
        "median_interval_width_sec": float(widths.median()),
        "interval_availability_rate": availability,
    }


def _candidate_residual_history(
    candidates: pd.DataFrame,
    settings: PredictedGapBucketUncertaintyConfig,
) -> pd.DataFrame:
    required = {
        "fold_id",
        "checkpoint",
        "candidate_family",
        "model_name",
        "feature_group",
        "quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec",
    }
    if not required <= set(candidates.columns):
        return pd.DataFrame()
    history = candidates.dropna(
        subset=["quali_gap_to_pole_sec", "predicted_quali_gap_to_pole_sec"]
    ).copy()
    if history.empty:
        return history
    history["absolute_residual"] = (
        history["quali_gap_to_pole_sec"].astype(float)
        - history["predicted_quali_gap_to_pole_sec"].astype(float)
    ).abs()
    history["predicted_gap_bucket"] = history["predicted_quali_gap_to_pole_sec"].map(
        lambda value: assign_predicted_gap_bucket(value, settings)
    )
    history["_method_key"] = history.apply(
        lambda row: _method_key(
            ChampionMethodConfig(
                family=str(row["candidate_family"]),
                model_name=str(row["model_name"]),
                feature_group=_optional_string(row["feature_group"]),
                temporal_weighting_policy=_optional_string(
                    row.get("temporal_weighting_policy", "uniform")
                ),
            )
        ),
        axis=1,
    )
    return history[history["predicted_gap_bucket"].notna()].copy()


def _select_predicted_bucket_quantile(
    history: pd.DataFrame,
    *,
    checkpoint: str,
    method: ChampionMethodConfig,
    predicted_gap_bucket: str,
    settings: PredictedGapBucketUncertaintyConfig,
) -> tuple[str, str, pd.Series] | None:
    method_key = _method_key(method)
    for level in settings.fallback_order:
        rows = _predicted_bucket_history_subset(
            history,
            level=level,
            checkpoint=checkpoint,
            method_key=method_key,
            predicted_gap_bucket=predicted_gap_bucket,
        )
        if len(rows) >= settings.min_residual_count:
            group = _calibration_group_label(
                level=level,
                checkpoint=checkpoint,
                method_key=method_key,
                predicted_gap_bucket=predicted_gap_bucket,
            )
            return level, group, rows["absolute_residual"].astype(float)
    return None


def _predicted_bucket_history_subset(
    history: pd.DataFrame,
    *,
    level: str,
    checkpoint: str,
    method_key: str,
    predicted_gap_bucket: str,
) -> pd.DataFrame:
    if level == "checkpoint_method_bucket":
        return history[
            history["checkpoint"].eq(checkpoint)
            & history["_method_key"].eq(method_key)
            & history["predicted_gap_bucket"].eq(predicted_gap_bucket)
        ]
    if level == "checkpoint_bucket":
        return history[
            history["checkpoint"].eq(checkpoint)
            & history["predicted_gap_bucket"].eq(predicted_gap_bucket)
        ]
    if level == "checkpoint_method":
        return history[history["checkpoint"].eq(checkpoint) & history["_method_key"].eq(method_key)]
    if level == "checkpoint":
        return history[history["checkpoint"].eq(checkpoint)]
    if level == "global":
        return history
    return history.iloc[0:0]


def _calibration_group_label(
    *,
    level: str,
    checkpoint: str,
    method_key: str,
    predicted_gap_bucket: str,
) -> str:
    if level == "checkpoint_method_bucket":
        return f"{checkpoint}|{method_key}|{predicted_gap_bucket}"
    if level == "checkpoint_bucket":
        return f"{checkpoint}|{predicted_gap_bucket}"
    if level == "checkpoint_method":
        return f"{checkpoint}|{method_key}"
    if level == "checkpoint":
        return checkpoint
    return "global"


def _method_key(method: ChampionMethodConfig) -> str:
    return "|".join(
        [
            method.family,
            method.model_name,
            method.feature_group or "",
            method.temporal_weighting_policy or "uniform",
        ]
    )


def _interval_metrics_by_predicted_gap_bucket(
    rows: pd.DataFrame,
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    required = {
        "checkpoint",
        "predicted_gap_bucket",
        "prediction_interval_low_sec",
        "prediction_interval_high_sec",
        "interval_contains_actual",
        "quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec",
    }
    if not required <= set(rows.columns):
        return {}
    result: dict[str, dict[str, dict[str, float | int | None]]] = {}
    intervals = rows[
        rows["predicted_gap_bucket"].notna()
        & rows["prediction_interval_low_sec"].notna()
        & rows["prediction_interval_high_sec"].notna()
    ].copy()
    if intervals.empty:
        return result
    intervals["_interval_width_sec"] = intervals["prediction_interval_high_sec"].astype(
        float
    ) - intervals["prediction_interval_low_sec"].astype(float)
    intervals["_abs_error_gap_sec"] = (
        intervals["predicted_quali_gap_to_pole_sec"].astype(float)
        - intervals["quali_gap_to_pole_sec"].astype(float)
    ).abs()
    for (checkpoint, bucket), group in intervals.groupby(
        ["checkpoint", "predicted_gap_bucket"],
        dropna=False,
        sort=True,
    ):
        contains = group["interval_contains_actual"].dropna()
        coverage = float(contains.astype(bool).mean()) if not contains.empty else None
        miss_count = int((~contains.astype(bool)).sum()) if not contains.empty else None
        result.setdefault(str(checkpoint), {})[str(bucket)] = {
            "rows_with_interval": int(len(group)),
            "coverage": coverage,
            "mean_interval_width_sec": float(group["_interval_width_sec"].mean()),
            "median_interval_width_sec": float(group["_interval_width_sec"].median()),
            "mean_abs_error_gap_sec": float(group["_abs_error_gap_sec"].mean()),
            "miss_count": miss_count,
        }
    return result


def _optional_string(value: object) -> str | None:
    if value is None or pd.isna(value) or str(value) in {"", "<NA>", "nan"}:
        return None
    return str(value)


def _resolve_dataset_path(config: DataConfig, dataset_path: Path | None) -> Path:
    path = dataset_path or build_combined_dataset_path(config.modeling_output_dir)
    if not path.is_absolute():
        path = config.project_root / path
    if not path.is_file():
        raise FileNotFoundError(f"Modeling dataset does not exist: {path}")
    return path


def _output_paths(
    metrics_dir: Path,
    selection_mode: ChampionSelectionMode,
) -> dict[str, Path]:
    mode = selection_mode.value
    return {
        "metrics": metrics_dir / "champion_metrics.json",
        "predictions": metrics_dir / "champion_predictions.parquet",
        "selection": metrics_dir / "champion_selection.parquet",
        "mode_metrics": metrics_dir / f"champion_{mode}_metrics.json",
        "mode_predictions": metrics_dir / f"champion_{mode}_predictions.parquet",
        "mode_selection": metrics_dir / f"champion_{mode}_selection.parquet",
    }


def _write_mode_outputs(
    paths: dict[str, Path],
    payload: dict[str, object],
    predictions: pd.DataFrame | None,
    selection: pd.DataFrame | None,
) -> None:
    for key in ("metrics", "mode_metrics"):
        _write_json(paths[key], payload)
    for key in ("predictions", "mode_predictions"):
        if predictions is None:
            paths[key].unlink(missing_ok=True)
        else:
            predictions.to_parquet(paths[key], engine="pyarrow", index=False)
    for key in ("selection", "mode_selection"):
        if selection is None:
            paths[key].unlink(missing_ok=True)
        else:
            selection.to_parquet(paths[key], engine="pyarrow", index=False)


def _skipped_payload(
    strategy: BacktestStrategy,
    selection_mode: ChampionSelectionMode,
    n_events: int,
    reason: str,
) -> dict[str, object]:
    return {
        "status": "skipped",
        "strategy": strategy.value,
        "selection_mode": selection_mode.value,
        "reason": reason,
        "n_events": n_events,
        "n_folds_total": 0,
        "n_folds_successful": 0,
        "n_folds_failed": 0,
        "created_at_utc": _utc_now(),
    }


def _delta(first: object, second: object) -> float | None:
    if first is None or second is None:
        return None
    return float(first) - float(second)


def _mean_absolute_error(rows: pd.DataFrame) -> float:
    return float(
        (
            rows["predicted_quali_gap_to_pole_sec"].astype(float)
            - rows["quali_gap_to_pole_sec"].astype(float)
        )
        .abs()
        .mean()
    )


def _current_season_prior_event_count(fold: BacktestFold) -> int:
    test_season = str(fold.test_event).split("/", maxsplit=1)[0]
    return sum(1 for event_key in fold.train_events if str(event_key).startswith(f"{test_season}/"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
