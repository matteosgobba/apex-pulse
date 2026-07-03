"""Diagnostic stability analysis for the season-aware FP3 candidate."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.modeling.season_aware_governance import (
    CANDIDATE_TEMPORAL_POLICY,
    CANONICAL_FAMILY,
    CANONICAL_FEATURE_GROUP,
    CANONICAL_MODEL,
    DEFAULT_TEMPORAL_POLICY,
    FP3_CHECKPOINT,
    canonical_candidate_identity,
    canonical_default_identity,
)
from f1_prediction.utils.paths import ensure_directory

TOP_EVENT_COUNT = 3
DEFAULT_TAIL_THRESHOLD_SEC = 0.05


@dataclass(frozen=True)
class SeasonAwareStabilitySummary:
    """Paths and issue counts produced by the season-aware stability report."""

    status: str
    summary_path: Path
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    missing_inputs: tuple[str, ...]
    generation_issues: tuple[str, ...]


def create_season_aware_stability_report(
    config: DataConfig,
    *,
    tail_risk_threshold_sec: float = DEFAULT_TAIL_THRESHOLD_SEC,
) -> SeasonAwareStabilitySummary:
    """Create artifact-only season-aware stability tables, figures, and summary."""
    metrics_dir = config.metrics_output_dir
    figures_dir = metrics_dir.parent / "figures"
    ensure_directory(metrics_dir)
    ensure_directory(figures_dir)

    artifacts, missing_inputs, generation_issues = load_stability_artifacts(metrics_dir)
    identity = validate_stability_identity(artifacts)
    event_rows = canonical_event_rows(artifacts.get("season_aware_event_level_comparison"))
    row_errors = canonical_row_errors(
        artifacts.get("ablation_current_season_only_with_prior_predictions"),
        artifacts.get("ablation_uniform_predictions"),
        event_rows,
    )
    scope_valid = bool(identity["identity_valid"].all()) and not event_rows.empty
    if not scope_valid:
        event_rows = pd.DataFrame(columns=event_stability_columns())
        row_errors = pd.DataFrame(columns=row_error_columns())

    by_season = build_season_stability(event_rows, row_errors)
    by_event = build_event_order_stability(event_rows)
    by_regime = build_regime_stability(event_rows, row_errors)
    concentration, leave_one_out = build_event_concentration(event_rows)
    distribution = build_error_distribution(row_errors)
    tail_risk = build_tail_risk(row_errors, threshold_sec=tail_risk_threshold_sec)
    replay_shadow = build_replay_shadow_summary(artifacts)
    missing = build_missing_evidence(missing_inputs)
    summary_payload = build_stability_summary_payload(
        identity=identity,
        event_rows=event_rows,
        by_season=by_season,
        by_regime=by_regime,
        concentration=concentration,
        leave_one_out=leave_one_out,
        distribution=distribution,
        tail_risk=tail_risk,
        replay_shadow=replay_shadow,
        missing_inputs=missing_inputs,
        generation_issues=generation_issues,
        tail_risk_threshold_sec=tail_risk_threshold_sec,
    )

    table_paths = (
        metrics_dir / "season_aware_stability_by_season.csv",
        metrics_dir / "season_aware_stability_by_event.csv",
        metrics_dir / "season_aware_stability_by_regime.csv",
        metrics_dir / "season_aware_stability_event_concentration.csv",
        metrics_dir / "season_aware_stability_leave_one_event_out.csv",
        metrics_dir / "season_aware_stability_error_distribution.csv",
        metrics_dir / "season_aware_stability_tail_risk.csv",
        metrics_dir / "season_aware_stability_replay_shadow_summary.csv",
        metrics_dir / "season_aware_stability_missing_evidence.csv",
    )
    by_season.to_csv(table_paths[0], index=False)
    by_event.to_csv(table_paths[1], index=False)
    by_regime.to_csv(table_paths[2], index=False)
    concentration.to_csv(table_paths[3], index=False)
    leave_one_out.to_csv(table_paths[4], index=False)
    distribution.to_csv(table_paths[5], index=False)
    tail_risk.to_csv(table_paths[6], index=False)
    replay_shadow.to_csv(table_paths[7], index=False)
    missing.to_csv(table_paths[8], index=False)

    summary_path = metrics_dir / "season_aware_stability_summary.json"
    _write_json(summary_path, summary_payload)
    figure_paths, figure_issues = generate_stability_figures(
        figures_dir=figures_dir,
        by_season=by_season,
        by_event=by_event,
        by_regime=by_regime,
        concentration=concentration,
        row_errors=row_errors,
        replay_shadow=replay_shadow,
    )
    summary_payload["generated_outputs"]["figures"] = [
        _relative_report_path(path) for path in figure_paths
    ]
    summary_payload["generation_issues"] = [
        *summary_payload["generation_issues"],
        *figure_issues,
    ]
    _write_json(summary_path, summary_payload)

    return SeasonAwareStabilitySummary(
        status=str(summary_payload["status"]),
        summary_path=summary_path,
        table_paths=(summary_path, *table_paths),
        figure_paths=tuple(figure_paths),
        missing_inputs=tuple(missing_inputs),
        generation_issues=tuple(summary_payload["generation_issues"]),
    )


def load_stability_artifacts(
    metrics_dir: Path,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Load optional artifacts used by the stability report."""
    csv_specs = {
        "season_aware_event_level_comparison": "season_aware_event_level_comparison.csv",
        "season_aware_season_level_comparison": "season_aware_season_level_comparison.csv",
        "season_aware_cold_start_comparison": "season_aware_cold_start_comparison.csv",
        "season_aware_governance_matrix": "season_aware_governance_matrix.csv",
        "season_aware_governance_split_summary": "season_aware_governance_split_summary.csv",
        "season_aware_governance_regime_summary": "season_aware_governance_regime_summary.csv",
        "prospective_replay_shadow_event_comparison": (
            "prospective_replay_shadow_event_comparison.csv"
        ),
        "prospective_replay_shadow_gate_feasibility": (
            "prospective_replay_shadow_gate_feasibility.csv"
        ),
        "prospective_replay_eligibility_by_event": ("prospective_replay_eligibility_by_event.csv"),
        "prospective_replay_candidate_evidence_ledger": (
            "prospective_replay_candidate_evidence_ledger.csv"
        ),
    }
    json_specs = {
        "season_aware_governance_summary": "season_aware_governance_summary.json",
        "season_aware_validation_summary": "season_aware_validation_summary.json",
        "prospective_policy_summary": "prospective_policy_summary.json",
    }
    parquet_specs = {
        "ablation_current_season_only_with_prior_predictions": (
            "ablation_current_season_only_with_prior_predictions.parquet"
        ),
        "ablation_uniform_predictions": "ablation_uniform_predictions.parquet",
        "prospective_replay_shadow_candidates": "prospective_replay_shadow_candidates.parquet",
    }
    artifacts: dict[str, Any] = {"metrics_dir": metrics_dir}
    missing: list[str] = []
    issues: list[str] = []
    for key, name in csv_specs.items():
        path = metrics_dir / name
        artifacts[f"{key}_path"] = path
        if not path.is_file():
            artifacts[key] = pd.DataFrame()
            missing.append(name)
            continue
        try:
            artifacts[key] = pd.read_csv(path)
        except (OSError, ValueError) as exc:
            artifacts[key] = pd.DataFrame()
            missing.append(name)
            issues.append(f"{name}: {exc}")
    for key, name in json_specs.items():
        path = metrics_dir / name
        artifacts[f"{key}_path"] = path
        if not path.is_file():
            artifacts[key] = None
            missing.append(name)
            continue
        try:
            artifacts[key] = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            artifacts[key] = None
            missing.append(name)
            issues.append(f"{name}: {exc}")
    for key, name in parquet_specs.items():
        path = metrics_dir / name
        artifacts[f"{key}_path"] = path
        if not path.is_file():
            artifacts[key] = pd.DataFrame()
            missing.append(name)
            continue
        try:
            artifacts[key] = pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            artifacts[key] = pd.DataFrame()
            missing.append(name)
            issues.append(f"{name}: {exc}")
    return artifacts, missing, issues


def validate_stability_identity(artifacts: dict[str, Any]) -> pd.DataFrame:
    """Validate the canonical FP3 candidate/default comparison scope."""
    rows: list[dict[str, object]] = []
    event_rows = artifacts.get("season_aware_event_level_comparison")
    rows.append(
        _identity_row(
            artifact_name="season_aware_event_level_comparison.csv",
            role="candidate",
            valid=_has_canonical_event_rows(event_rows),
            validation_method="observed_event_level_fields",
            observed=canonical_candidate_identity()
            if _has_canonical_event_rows(event_rows)
            else _observed_identity(event_rows),
        )
    )
    rows.append(
        _identity_row(
            artifact_name="season_aware_event_level_comparison.csv",
            role="default",
            valid=_has_canonical_event_rows(event_rows),
            validation_method="inferred_static_uniform_contract",
            observed=canonical_default_identity(),
        )
    )
    current = artifacts.get("ablation_current_season_only_with_prior_predictions")
    uniform = artifacts.get("ablation_uniform_predictions")
    rows.append(
        _identity_row(
            artifact_name="ablation_current_season_only_with_prior_predictions.parquet",
            role="candidate",
            valid=_has_canonical_prediction_rows(current, CANDIDATE_TEMPORAL_POLICY),
            validation_method="observed_prediction_rows",
            observed=canonical_candidate_identity()
            if _has_canonical_prediction_rows(current, CANDIDATE_TEMPORAL_POLICY)
            else _observed_prediction_identity(current),
        )
    )
    rows.append(
        _identity_row(
            artifact_name="ablation_uniform_predictions.parquet",
            role="default",
            valid=_has_canonical_prediction_rows(uniform, DEFAULT_TEMPORAL_POLICY),
            validation_method="observed_prediction_rows",
            observed=canonical_default_identity()
            if _has_canonical_prediction_rows(uniform, DEFAULT_TEMPORAL_POLICY)
            else _observed_prediction_identity(uniform),
        )
    )
    return pd.DataFrame(rows)


def canonical_event_rows(frame: object) -> pd.DataFrame:
    """Return canonical event-level weighted-candidate versus static/default rows."""
    columns = event_stability_columns()
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=columns)
    required = {
        "candidate_family",
        "candidate_model_name",
        "candidate_feature_group",
        "training_policy",
        "season",
        "event",
        "event_slug",
        "fold_id",
        "current_season_prior_event_count",
        "current_season_evidence_regime",
        "rows",
        "static_mae_gap_sec",
        "candidate_mae_gap_sec",
        "delta_vs_static_sec",
    }
    if not required <= set(frame.columns):
        return pd.DataFrame(columns=columns)
    result = frame[
        frame["candidate_family"].astype(str).eq(CANONICAL_FAMILY)
        & frame["candidate_model_name"].astype(str).eq(CANONICAL_MODEL)
        & frame["candidate_feature_group"].astype(str).eq(CANONICAL_FEATURE_GROUP)
        & frame["training_policy"].astype(str).eq(CANDIDATE_TEMPORAL_POLICY)
    ].copy()
    if result.empty:
        return pd.DataFrame(columns=columns)
    result["season"] = pd.to_numeric(result["season"], errors="coerce").astype("Int64")
    result["event_order"] = pd.to_numeric(
        result["current_season_prior_event_count"],
        errors="coerce",
    ).astype("Int64")
    result["row_count"] = pd.to_numeric(result["rows"], errors="coerce").fillna(0).astype(int)
    result["candidate_mae"] = pd.to_numeric(
        result["candidate_mae_gap_sec"],
        errors="coerce",
    )
    result["default_mae"] = pd.to_numeric(result["static_mae_gap_sec"], errors="coerce")
    result["delta_mae"] = pd.to_numeric(result["delta_vs_static_sec"], errors="coerce")
    result["candidate_better"] = result["delta_mae"].lt(0)
    result["absolute_delta"] = result["delta_mae"].abs()
    result["regime"] = result["current_season_evidence_regime"].astype(str)
    return result.loc[:, columns].sort_values(["season", "event_order", "fold_id"], kind="stable")


def canonical_row_errors(
    current: object,
    uniform: object,
    event_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Align canonical row-level candidate/default predictions for distributional metrics."""
    columns = row_error_columns()
    if not isinstance(current, pd.DataFrame) or not isinstance(uniform, pd.DataFrame):
        return pd.DataFrame(columns=columns)
    if current.empty or uniform.empty:
        return pd.DataFrame(columns=columns)
    current_rows = _canonical_prediction_rows(current, CANDIDATE_TEMPORAL_POLICY)
    uniform_rows = _canonical_prediction_rows(uniform, DEFAULT_TEMPORAL_POLICY)
    if current_rows.empty or uniform_rows.empty:
        return pd.DataFrame(columns=columns)
    key_cols = ["fold_id", "season", "event_slug", "checkpoint", "driver"]
    current_rows = current_rows.rename(
        columns={"predicted_quali_gap_to_pole_sec": "candidate_prediction_gap_sec"}
    )
    uniform_rows = uniform_rows.rename(
        columns={"predicted_quali_gap_to_pole_sec": "default_prediction_gap_sec"}
    )
    merged = current_rows.merge(
        uniform_rows.loc[:, [*key_cols, "default_prediction_gap_sec"]],
        on=key_cols,
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged["actual_gap_sec"] = pd.to_numeric(merged["quali_gap_to_pole_sec"], errors="coerce")
    merged["candidate_absolute_error_sec"] = (
        pd.to_numeric(merged["candidate_prediction_gap_sec"], errors="coerce")
        - merged["actual_gap_sec"]
    ).abs()
    merged["default_absolute_error_sec"] = (
        pd.to_numeric(merged["default_prediction_gap_sec"], errors="coerce")
        - merged["actual_gap_sec"]
    ).abs()
    merged["error_delta_sec"] = (
        merged["candidate_absolute_error_sec"] - merged["default_absolute_error_sec"]
    )
    merged["candidate_better"] = merged["error_delta_sec"].lt(0)
    if not event_rows.empty:
        meta = event_rows.loc[
            :,
            ["season", "event_slug", "fold_id", "event_order", "regime"],
        ].drop_duplicates()
        merged = merged.merge(meta, on=["season", "event_slug", "fold_id"], how="left")
    if "event_order" not in merged:
        merged["event_order"] = pd.NA
    if "regime" not in merged:
        merged["regime"] = pd.NA
    return merged.loc[:, columns]


def build_season_stability(event_rows: pd.DataFrame, row_errors: pd.DataFrame) -> pd.DataFrame:
    """Summarize stability by season."""
    columns = [
        "season",
        "event_count",
        "row_count",
        "candidate_mae",
        "default_mae",
        "delta_mae",
        "median_event_delta",
        "share_events_improved",
        "share_rows_improved",
        "worst_event_delta",
        "best_event_delta",
    ]
    if event_rows.empty:
        return pd.DataFrame(columns=columns)
    row_share = _share_rows_by_group(row_errors, "season")
    rows: list[dict[str, object]] = []
    for season, group in event_rows.groupby("season", dropna=False, sort=True):
        rows.append(
            {
                "season": int(season),
                "event_count": int(group["event_slug"].nunique()),
                "row_count": int(group["row_count"].sum()),
                "candidate_mae": _weighted_mean(group, "candidate_mae", "row_count"),
                "default_mae": _weighted_mean(group, "default_mae", "row_count"),
                "delta_mae": _weighted_mean(group, "delta_mae", "row_count"),
                "median_event_delta": _number_or_none(group["delta_mae"].median()),
                "share_events_improved": float(group["candidate_better"].mean()),
                "share_rows_improved": row_share.get(season),
                "worst_event_delta": _number_or_none(group["delta_mae"].max()),
                "best_event_delta": _number_or_none(group["delta_mae"].min()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_event_order_stability(event_rows: pd.DataFrame) -> pd.DataFrame:
    """Return canonical event-order stability rows."""
    columns = [
        "season",
        "event_order",
        "event",
        "event_slug",
        "regime",
        "candidate_mae",
        "default_mae",
        "delta_mae",
        "candidate_better",
        "absolute_delta",
    ]
    if event_rows.empty:
        return pd.DataFrame(columns=columns)
    return event_rows.loc[:, columns].sort_values(["season", "event_order"], kind="stable")


def build_regime_stability(event_rows: pd.DataFrame, row_errors: pd.DataFrame) -> pd.DataFrame:
    """Summarize stability by documented current-season evidence regime."""
    columns = [
        "regime",
        "event_count",
        "row_count",
        "candidate_mae",
        "default_mae",
        "delta_mae",
        "share_events_improved",
        "share_rows_improved",
        "delta_dispersion",
        "worst_event_delta",
    ]
    if event_rows.empty:
        return pd.DataFrame(columns=columns)
    row_share = _share_rows_by_group(row_errors, "regime")
    rows: list[dict[str, object]] = []
    for regime, group in event_rows.groupby("regime", dropna=False, sort=False):
        rows.append(
            {
                "regime": str(regime),
                "event_count": int(group["event_slug"].nunique()),
                "row_count": int(group["row_count"].sum()),
                "candidate_mae": _weighted_mean(group, "candidate_mae", "row_count"),
                "default_mae": _weighted_mean(group, "default_mae", "row_count"),
                "delta_mae": _weighted_mean(group, "delta_mae", "row_count"),
                "share_events_improved": float(group["candidate_better"].mean()),
                "share_rows_improved": row_share.get(regime),
                "delta_dispersion": _number_or_none(group["delta_mae"].std(ddof=0)),
                "worst_event_delta": _number_or_none(group["delta_mae"].max()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_event_concentration(event_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe concentration and leave-one-event-out stability."""
    concentration_columns = [
        "top_k_beneficial_events",
        "top_k_harmful_events",
        "share_total_improvement_explained_by_top_beneficial_events",
        "share_total_harm_explained_by_top_harmful_events",
        "leave_one_event_out_delta_min",
        "leave_one_event_out_delta_max",
        "leave_one_event_out_delta_range",
        "sign_flip_when_event_removed_count",
    ]
    loo_columns = [
        "season",
        "event",
        "event_slug",
        "removed_event_delta",
        "aggregate_delta_without_event",
        "sign_flip_when_removed",
    ]
    if event_rows.empty:
        return pd.DataFrame(columns=concentration_columns), pd.DataFrame(columns=loo_columns)
    frame = event_rows.copy()
    frame["weighted_delta"] = frame["delta_mae"] * frame["row_count"]
    benefit = frame[frame["weighted_delta"].lt(0)].copy()
    harm = frame[frame["weighted_delta"].gt(0)].copy()
    total_benefit = float((-benefit["weighted_delta"]).sum())
    total_harm = float(harm["weighted_delta"].sum())
    top_benefit = benefit.sort_values("weighted_delta", kind="stable").head(TOP_EVENT_COUNT)
    top_harm = harm.sort_values("weighted_delta", ascending=False, kind="stable").head(
        TOP_EVENT_COUNT
    )
    overall_delta = _weighted_mean(frame, "delta_mae", "row_count")
    loo_rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        remaining = frame.drop(index=row.name)
        without = _weighted_mean(remaining, "delta_mae", "row_count")
        loo_rows.append(
            {
                "season": row["season"],
                "event": row["event"],
                "event_slug": row["event_slug"],
                "removed_event_delta": row["delta_mae"],
                "aggregate_delta_without_event": without,
                "sign_flip_when_removed": _sign_flip(overall_delta, without),
            }
        )
    loo = pd.DataFrame(loo_rows, columns=loo_columns)
    loo_values = pd.to_numeric(loo["aggregate_delta_without_event"], errors="coerce").dropna()
    concentration = pd.DataFrame(
        [
            {
                "top_k_beneficial_events": json.dumps(_event_records(top_benefit)),
                "top_k_harmful_events": json.dumps(_event_records(top_harm)),
                "share_total_improvement_explained_by_top_beneficial_events": (
                    float((-top_benefit["weighted_delta"]).sum() / total_benefit)
                    if total_benefit
                    else None
                ),
                "share_total_harm_explained_by_top_harmful_events": (
                    float(top_harm["weighted_delta"].sum() / total_harm) if total_harm else None
                ),
                "leave_one_event_out_delta_min": _number_or_none(loo_values.min()),
                "leave_one_event_out_delta_max": _number_or_none(loo_values.max()),
                "leave_one_event_out_delta_range": (
                    float(loo_values.max() - loo_values.min()) if not loo_values.empty else None
                ),
                "sign_flip_when_event_removed_count": int(
                    loo["sign_flip_when_removed"].astype(bool).sum()
                ),
            }
        ],
        columns=concentration_columns,
    )
    return concentration, loo


def build_error_distribution(row_errors: pd.DataFrame) -> pd.DataFrame:
    """Summarize candidate/default absolute-error distributions."""
    columns = [
        "model_role",
        "mean_absolute_error",
        "median_absolute_error",
        "p75_absolute_error",
        "p90_absolute_error",
        "p95_absolute_error",
        "maximum_absolute_error",
    ]
    if row_errors.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for role, column in (
        ("candidate", "candidate_absolute_error_sec"),
        ("default", "default_absolute_error_sec"),
    ):
        values = pd.to_numeric(row_errors[column], errors="coerce").dropna()
        rows.append(
            {
                "model_role": role,
                "mean_absolute_error": _number_or_none(values.mean()),
                "median_absolute_error": _number_or_none(values.median()),
                "p75_absolute_error": _number_or_none(values.quantile(0.75)),
                "p90_absolute_error": _number_or_none(values.quantile(0.90)),
                "p95_absolute_error": _number_or_none(values.quantile(0.95)),
                "maximum_absolute_error": _number_or_none(values.max()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_tail_risk(row_errors: pd.DataFrame, *, threshold_sec: float) -> pd.DataFrame:
    """Compute configured-threshold tail-risk diagnostics."""
    columns = [
        "threshold_sec",
        "rows",
        "candidate_tail_error_rate_above_threshold",
        "default_tail_error_rate_above_threshold",
        "candidate_tail_minus_default_tail_rate",
        "candidate_worse_by_more_than_threshold_rate",
        "candidate_worse_by_more_than_threshold_count",
        "maximum_candidate_minus_default_error_delta",
    ]
    if row_errors.empty:
        return pd.DataFrame(columns=columns)
    candidate_tail = row_errors["candidate_absolute_error_sec"].gt(threshold_sec)
    default_tail = row_errors["default_absolute_error_sec"].gt(threshold_sec)
    worse_by_threshold = row_errors["error_delta_sec"].gt(threshold_sec)
    return pd.DataFrame(
        [
            {
                "threshold_sec": threshold_sec,
                "rows": int(len(row_errors)),
                "candidate_tail_error_rate_above_threshold": float(candidate_tail.mean()),
                "default_tail_error_rate_above_threshold": float(default_tail.mean()),
                "candidate_tail_minus_default_tail_rate": float(
                    candidate_tail.mean() - default_tail.mean()
                ),
                "candidate_worse_by_more_than_threshold_rate": float(worse_by_threshold.mean()),
                "candidate_worse_by_more_than_threshold_count": int(worse_by_threshold.sum()),
                "maximum_candidate_minus_default_error_delta": _number_or_none(
                    row_errors["error_delta_sec"].max()
                ),
            }
        ],
        columns=columns,
    )


def build_replay_shadow_summary(artifacts: dict[str, Any]) -> pd.DataFrame:
    """Summarize true replay and shadow-history evidence separately by split."""
    columns = [
        "split_name",
        "evidence_type",
        "live_selected_events",
        "shadow_candidate_eligible_events",
        "shadow_counterfactual_selected_events",
        "max_prior_shadow_candidate_folds",
        "max_prior_shadow_candidate_prediction_rows",
        "max_prior_shadow_aligned_rows",
        "mean_shadow_candidate_prior_mae",
        "mean_shadow_default_prior_mae",
        "mean_shadow_prior_improvement_sec",
        "evidence_available",
    ]
    ledger = artifacts.get("prospective_replay_candidate_evidence_ledger")
    if not isinstance(ledger, pd.DataFrame) or ledger.empty:
        return pd.DataFrame(
            [
                {
                    "split_name": "unavailable",
                    "evidence_type": "replay_shadow_artifacts_missing",
                    "evidence_available": False,
                }
            ],
            columns=columns,
        )
    frame = ledger.copy()
    if "policy_profile" in frame:
        frame = frame[frame["policy_profile"].astype(str).eq("season_aware_frozen")].copy()
    rows: list[dict[str, object]] = []
    for split_name, group in frame.groupby("split_name", dropna=False, sort=False):
        rows.append(
            {
                "split_name": split_name,
                "evidence_type": "true_replay_live_and_shadow_history_diagnostic",
                "live_selected_events": _sum_bool(group, "season_aware_selected"),
                "shadow_candidate_eligible_events": _sum_bool(
                    group,
                    "shadow_season_aware_candidate_eligible_under_frozen_gates",
                ),
                "shadow_counterfactual_selected_events": int(
                    group.get("shadow_history_counterfactual_selection", pd.Series(dtype=object))
                    .astype(str)
                    .eq("season_aware_weighted_candidate")
                    .sum()
                ),
                "max_prior_shadow_candidate_folds": _max_int(
                    group,
                    "prior_shadow_candidate_events_available",
                ),
                "max_prior_shadow_candidate_prediction_rows": _max_int(
                    group,
                    "prior_shadow_candidate_prediction_rows_available",
                ),
                "max_prior_shadow_aligned_rows": _max_int(
                    group,
                    "prior_shadow_candidate_default_aligned_rows",
                ),
                "mean_shadow_candidate_prior_mae": _series_mean(
                    group,
                    "shadow_candidate_prior_mae",
                ),
                "mean_shadow_default_prior_mae": _series_mean(group, "shadow_default_prior_mae"),
                "mean_shadow_prior_improvement_sec": _series_mean(
                    group,
                    "shadow_prior_improvement_sec",
                ),
                "evidence_available": True,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_missing_evidence(missing_inputs: list[str]) -> pd.DataFrame:
    """Write missing optional input evidence explicitly."""
    return pd.DataFrame(
        [
            {
                "artifact_name": name,
                "evidence_status": "unavailable",
                "treatment": "not_treated_as_negative_evidence",
            }
            for name in missing_inputs
        ],
        columns=("artifact_name", "evidence_status", "treatment"),
    )


def build_stability_summary_payload(
    *,
    identity: pd.DataFrame,
    event_rows: pd.DataFrame,
    by_season: pd.DataFrame,
    by_regime: pd.DataFrame,
    concentration: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    distribution: pd.DataFrame,
    tail_risk: pd.DataFrame,
    replay_shadow: pd.DataFrame,
    missing_inputs: list[str],
    generation_issues: list[str],
    tail_risk_threshold_sec: float,
) -> dict[str, object]:
    """Create the stability summary JSON payload."""
    candidate_identity_valid = _identity_valid(identity, "candidate")
    default_identity_valid = _identity_valid(identity, "default")
    comparison_scope_valid = bool(candidate_identity_valid and default_identity_valid)
    season_class = classify_season_stability(by_season)
    regime_class = classify_regime_stability(by_regime)
    concentration_class = classify_event_concentration(concentration)
    tail_class = classify_tail_risk(tail_risk)
    overall = classify_overall_stability(
        comparison_scope_valid=comparison_scope_valid,
        season_classification=season_class,
        regime_classification=regime_class,
        concentration_classification=concentration_class,
        tail_risk_classification=tail_class,
        event_rows=event_rows,
    )
    return {
        "status": "partial" if missing_inputs else "complete",
        "candidate_identity_validation_status": "valid"
        if candidate_identity_valid
        else "identity_or_scope_invalid",
        "default_identity_validation_status": "valid"
        if default_identity_valid
        else "identity_or_scope_invalid",
        "comparison_scope_status": (
            "valid" if comparison_scope_valid else "identity_or_scope_invalid"
        ),
        "overall_stability_classification": overall,
        "season_stability_classification": season_class,
        "regime_stability_classification": regime_class,
        "event_concentration_classification": concentration_class,
        "tail_risk_classification": tail_class,
        "retrospective_stability_summary": retrospective_stability_summary(
            event_rows,
            by_season,
            by_regime,
        ),
        "replay_stability_summary": replay_stability_summary(replay_shadow),
        "shadow_history_stability_summary": shadow_history_stability_summary(replay_shadow),
        "primary_supporting_evidence": primary_supporting_evidence(by_season, by_regime),
        "primary_cautionary_evidence": primary_cautionary_evidence(
            by_season,
            by_regime,
            concentration,
            tail_risk,
        ),
        "event_concentration_summary": records_for_json(concentration),
        "tail_risk_summary": records_for_json(tail_risk),
        "tail_risk_threshold_sec": tail_risk_threshold_sec,
        "known_limitations": [
            "Stability analysis is artifact-driven and does not retrain or rerun replay.",
            "Retrospective error metrics, true replay live selections, and shadow-history "
            "counterfactual eligibility are reported separately.",
            "Circuit/event grouping is limited to safely inferable existing event names/slugs; "
            "no external metadata is fetched.",
        ],
        "governance_interpretation": (
            "The weighted candidate remains diagnostic evidence only; stability findings do not "
            "promote or alter any policy."
        ),
        "policy_recommendation": "season_aware_candidate_requires_more_evidence",
        "generated_outputs": {
            "metrics": [
                "reports/metrics/season_aware_stability_summary.json",
                "reports/metrics/season_aware_stability_by_season.csv",
                "reports/metrics/season_aware_stability_by_event.csv",
                "reports/metrics/season_aware_stability_by_regime.csv",
                "reports/metrics/season_aware_stability_event_concentration.csv",
                "reports/metrics/season_aware_stability_leave_one_event_out.csv",
                "reports/metrics/season_aware_stability_error_distribution.csv",
                "reports/metrics/season_aware_stability_tail_risk.csv",
                "reports/metrics/season_aware_stability_replay_shadow_summary.csv",
                "reports/metrics/season_aware_stability_missing_evidence.csv",
            ],
            "figures": [],
        },
        "missing_evidence": missing_inputs,
        "generation_issues": generation_issues,
        "generated_at": _utc_now(),
    }


def classify_season_stability(by_season: pd.DataFrame) -> str:
    if by_season.empty:
        return "insufficient_evidence"
    improved = by_season["delta_mae"].lt(0)
    harmed = by_season["delta_mae"].gt(0)
    if improved.sum() >= 2 and not harmed.any():
        return "broadly_stable_support"
    if improved.any() and harmed.any():
        return "mixed_support"
    if improved.any():
        return "mixed_support"
    return "insufficient_evidence"


def classify_regime_stability(by_regime: pd.DataFrame) -> str:
    if by_regime.empty:
        return "insufficient_evidence"
    deltas = by_regime.set_index("regime")["delta_mae"]
    if "cold_start" in deltas and deltas.get("cold_start", 0) >= 0 and (deltas < 0).any():
        return "regime_dependent_support"
    if (deltas < 0).all():
        return "broadly_stable_support"
    if (deltas < 0).any() and (deltas > 0).any():
        return "mixed_support"
    return "insufficient_evidence"


def classify_event_concentration(concentration: pd.DataFrame) -> str:
    if concentration.empty:
        return "insufficient_evidence"
    share = _number_or_none(
        concentration["share_total_improvement_explained_by_top_beneficial_events"].iloc[0]
    )
    if share is None:
        return "insufficient_evidence"
    if share >= 0.6:
        return "concentrated_support"
    return "broadly_stable_support"


def classify_tail_risk(tail_risk: pd.DataFrame) -> str:
    if tail_risk.empty:
        return "insufficient_evidence"
    row = tail_risk.iloc[0]
    worse_rate = _number_or_none(row.get("candidate_worse_by_more_than_threshold_rate"))
    max_delta = _number_or_none(row.get("maximum_candidate_minus_default_error_delta"))
    if worse_rate is not None and worse_rate > 0.25:
        return "tail_risk_concern"
    if max_delta is not None and max_delta > 1.0:
        return "tail_risk_concern"
    return "mixed_support" if worse_rate and worse_rate > 0 else "broadly_stable_support"


def classify_overall_stability(
    *,
    comparison_scope_valid: bool,
    season_classification: str,
    regime_classification: str,
    concentration_classification: str,
    tail_risk_classification: str,
    event_rows: pd.DataFrame,
) -> str:
    if not comparison_scope_valid:
        return "identity_or_scope_invalid"
    if event_rows.empty:
        return "insufficient_evidence"
    if tail_risk_classification == "tail_risk_concern":
        return "tail_risk_concern"
    if concentration_classification == "concentrated_support":
        return "concentrated_support"
    if regime_classification == "regime_dependent_support":
        return "regime_dependent_support"
    if season_classification == "mixed_support":
        return "mixed_support"
    return season_classification


def generate_stability_figures(
    *,
    figures_dir: Path,
    by_season: pd.DataFrame,
    by_event: pd.DataFrame,
    by_regime: pd.DataFrame,
    concentration: pd.DataFrame,
    row_errors: pd.DataFrame,
    replay_shadow: pd.DataFrame,
) -> tuple[list[Path], list[str]]:
    """Generate static Matplotlib stability figures."""
    ensure_directory(figures_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(figures_dir.parent / ".matplotlib-cache"))
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    specs = (
        (
            "season_aware_stability_delta_by_event_order.png",
            lambda path: plot_delta_by_event_order(plt, by_event, path),
        ),
        (
            "season_aware_stability_delta_by_season.png",
            lambda path: plot_delta_by_season(plt, by_season, path),
        ),
        (
            "season_aware_stability_delta_by_regime.png",
            lambda path: plot_delta_by_regime(plt, by_regime, path),
        ),
        (
            "season_aware_stability_event_concentration.png",
            lambda path: plot_event_concentration(plt, concentration, path),
        ),
        (
            "season_aware_stability_error_distribution.png",
            lambda path: plot_error_distribution(plt, row_errors, path),
        ),
        (
            "season_aware_stability_live_vs_shadow_status.png",
            lambda path: plot_live_vs_shadow_status(plt, replay_shadow, path),
        ),
    )
    paths: list[Path] = []
    issues: list[str] = []
    for filename, plotter in specs:
        path = figures_dir / filename
        try:
            if plotter(path):
                paths.append(path)
            else:
                issues.append(f"{filename}: insufficient data")
        except (OSError, ValueError, KeyError) as exc:
            issues.append(f"{filename}: {exc}")
            plt.close("all")
    return paths, issues


def plot_delta_by_event_order(plt: Any, by_event: pd.DataFrame, path: Path) -> bool:
    if by_event.empty:
        return _empty_plot(plt, path, "Event-order stability unavailable")
    fig, ax = plt.subplots(figsize=(11, 5))
    for season, group in by_event.groupby("season", sort=True):
        ax.plot(group["event_order"], group["delta_mae"], marker="o", label=str(season))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Weighted FP3 candidate delta by event order")
    ax.set_xlabel("Current-season prior event count")
    ax.set_ylabel("Candidate MAE minus default MAE")
    ax.legend(title="Season")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def plot_delta_by_season(plt: Any, by_season: pd.DataFrame, path: Path) -> bool:
    if by_season.empty:
        return _empty_plot(plt, path, "Season stability unavailable")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(by_season["season"].astype(str), by_season["delta_mae"], color="#3b6ea8")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Weighted FP3 candidate delta by season")
    ax.set_ylabel("Candidate MAE minus default MAE")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def plot_delta_by_regime(plt: Any, by_regime: pd.DataFrame, path: Path) -> bool:
    if by_regime.empty:
        return _empty_plot(plt, path, "Regime stability unavailable")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2d7f5e" if value < 0 else "#b55d4c" for value in by_regime["delta_mae"]]
    ax.bar(by_regime["regime"], by_regime["delta_mae"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Weighted FP3 candidate delta by regime")
    ax.set_ylabel("Candidate MAE minus default MAE")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def plot_event_concentration(plt: Any, concentration: pd.DataFrame, path: Path) -> bool:
    if concentration.empty:
        return _empty_plot(plt, path, "Event concentration unavailable")
    row = concentration.iloc[0]
    values = [
        row.get("share_total_improvement_explained_by_top_beneficial_events"),
        row.get("share_total_harm_explained_by_top_harmful_events"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["Top benefit share", "Top harm share"], values, color=["#2d7f5e", "#b55d4c"])
    ax.set_ylim(0, 1)
    ax.set_title("Event concentration of weighted-candidate signal")
    ax.set_ylabel("Share")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def plot_error_distribution(plt: Any, row_errors: pd.DataFrame, path: Path) -> bool:
    if row_errors.empty:
        return _empty_plot(plt, path, "Error distribution unavailable")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(
        row_errors["default_absolute_error_sec"].dropna(),
        bins=30,
        alpha=0.55,
        label="Uniform/default",
    )
    ax.hist(
        row_errors["candidate_absolute_error_sec"].dropna(),
        bins=30,
        alpha=0.55,
        label="Weighted candidate",
    )
    ax.set_title("Candidate/default absolute-error distribution")
    ax.set_xlabel("Absolute error (sec)")
    ax.set_ylabel("Rows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def plot_live_vs_shadow_status(plt: Any, replay_shadow: pd.DataFrame, path: Path) -> bool:
    if replay_shadow.empty:
        return _empty_plot(plt, path, "Replay and shadow evidence unavailable")
    labels = replay_shadow["split_name"].astype(str)
    live = pd.to_numeric(replay_shadow["live_selected_events"], errors="coerce").fillna(0)
    shadow = pd.to_numeric(
        replay_shadow["shadow_counterfactual_selected_events"],
        errors="coerce",
    ).fillna(0)
    x = range(len(replay_shadow))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([value - 0.18 for value in x], live, width=0.36, label="Observed live replay")
    ax.bar([value + 0.18 for value in x], shadow, width=0.36, label="Shadow counterfactual")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Events")
    ax.set_title("Live replay selections versus shadow-history diagnostics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def retrospective_stability_summary(
    event_rows: pd.DataFrame,
    by_season: pd.DataFrame,
    by_regime: pd.DataFrame,
) -> dict[str, object]:
    if event_rows.empty:
        return {"available": False}
    return {
        "available": True,
        "events": int(len(event_rows)),
        "overall_delta_mae": _weighted_mean(event_rows, "delta_mae", "row_count"),
        "share_events_improved": float(event_rows["candidate_better"].mean()),
        "seasons": records_for_json(by_season),
        "regimes": records_for_json(by_regime),
    }


def replay_stability_summary(replay_shadow: pd.DataFrame) -> dict[str, object]:
    evidence_available = replay_shadow["evidence_available"].fillna(False).astype(bool)
    if replay_shadow.empty or not evidence_available.any():
        return {"available": False}
    return {
        "available": True,
        "live_selected_events": int(
            pd.to_numeric(replay_shadow["live_selected_events"], errors="coerce").fillna(0).sum()
        ),
        "splits": records_for_json(replay_shadow),
    }


def shadow_history_stability_summary(replay_shadow: pd.DataFrame) -> dict[str, object]:
    evidence_available = replay_shadow["evidence_available"].fillna(False).astype(bool)
    if replay_shadow.empty or not evidence_available.any():
        return {"available": False}
    return {
        "available": True,
        "shadow_candidate_eligible_events": int(
            pd.to_numeric(
                replay_shadow["shadow_candidate_eligible_events"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "shadow_counterfactual_selected_events": int(
            pd.to_numeric(
                replay_shadow["shadow_counterfactual_selected_events"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "mean_shadow_prior_improvement_sec": _series_mean(
            replay_shadow,
            "mean_shadow_prior_improvement_sec",
        ),
    }


def primary_supporting_evidence(
    by_season: pd.DataFrame,
    by_regime: pd.DataFrame,
) -> list[str]:
    evidence: list[str] = []
    if not by_season.empty:
        improved = by_season[by_season["delta_mae"].lt(0)]
        evidence.append(f"Candidate improves in {len(improved)} of {len(by_season)} seasons.")
    if not by_regime.empty:
        best = by_regime.sort_values("delta_mae", kind="stable").iloc[0]
        evidence.append(
            f"Strongest regime is {best['regime']} with delta {float(best['delta_mae']):.3f} sec."
        )
    return evidence


def primary_cautionary_evidence(
    by_season: pd.DataFrame,
    by_regime: pd.DataFrame,
    concentration: pd.DataFrame,
    tail_risk: pd.DataFrame,
) -> list[str]:
    cautions: list[str] = []
    if not by_season.empty and by_season["delta_mae"].ge(0).any():
        neutral_or_harm = by_season[by_season["delta_mae"].ge(0)]
        seasons = ", ".join(neutral_or_harm["season"].astype(str))
        cautions.append(f"Candidate does not improve in seasons: {seasons}.")
    if not by_regime.empty and by_regime["delta_mae"].ge(0).any():
        regimes = by_regime[by_regime["delta_mae"].ge(0)]["regime"].astype(str)
        cautions.append(f"Non-improving regimes: {', '.join(regimes)}.")
    if not concentration.empty:
        share = concentration["share_total_improvement_explained_by_top_beneficial_events"].iloc[0]
        if pd.notna(share) and float(share) >= 0.6:
            cautions.append(
                f"Top {TOP_EVENT_COUNT} beneficial events explain "
                f"{float(share):.1%} of total improvement."
            )
    if not tail_risk.empty:
        worse = tail_risk["candidate_worse_by_more_than_threshold_rate"].iloc[0]
        cautions.append(f"Candidate worse-by-threshold row rate is {float(worse):.1%}.")
    return cautions


def event_stability_columns() -> list[str]:
    return [
        "season",
        "event_order",
        "event",
        "event_slug",
        "fold_id",
        "regime",
        "row_count",
        "candidate_mae",
        "default_mae",
        "delta_mae",
        "candidate_better",
        "absolute_delta",
    ]


def row_error_columns() -> list[str]:
    return [
        "season",
        "event",
        "event_slug",
        "fold_id",
        "event_order",
        "checkpoint",
        "driver",
        "regime",
        "candidate_prediction_gap_sec",
        "default_prediction_gap_sec",
        "actual_gap_sec",
        "candidate_absolute_error_sec",
        "default_absolute_error_sec",
        "error_delta_sec",
        "candidate_better",
    ]


def _canonical_prediction_rows(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    required = {
        "checkpoint",
        "model_name",
        "feature_group",
        "temporal_weighting_policy",
        "predicted_quali_gap_to_pole_sec",
        "quali_gap_to_pole_sec",
        "fold_id",
        "season",
        "event_slug",
        "driver",
    }
    if frame.empty or not required <= set(frame.columns):
        return pd.DataFrame()
    return frame[
        frame["checkpoint"].astype(str).eq(FP3_CHECKPOINT)
        & frame["model_name"].astype(str).eq(CANONICAL_MODEL)
        & frame["feature_group"].astype(str).eq(CANONICAL_FEATURE_GROUP)
        & frame["temporal_weighting_policy"].astype(str).eq(policy)
    ].copy()


def _has_canonical_event_rows(frame: object) -> bool:
    return not canonical_event_rows(frame).empty


def _has_canonical_prediction_rows(frame: object, policy: str) -> bool:
    return isinstance(frame, pd.DataFrame) and not _canonical_prediction_rows(frame, policy).empty


def _observed_identity(frame: object) -> dict[str, object]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    return {
        "family": _first_value(frame, "candidate_family"),
        "model_name": _first_value(frame, "candidate_model_name"),
        "feature_group": _first_value(frame, "candidate_feature_group"),
        "temporal_weighting_policy": _first_value(frame, "training_policy"),
    }


def _observed_prediction_identity(frame: object) -> dict[str, object]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    return {
        "family": CANONICAL_FAMILY,
        "model_name": _first_value(frame, "model_name"),
        "feature_group": _first_value(frame, "feature_group"),
        "temporal_weighting_policy": _first_value(frame, "temporal_weighting_policy"),
    }


def _identity_row(
    *,
    artifact_name: str,
    role: str,
    valid: bool,
    validation_method: str,
    observed: dict[str, object],
) -> dict[str, object]:
    expected = (
        canonical_candidate_identity() if role == "candidate" else canonical_default_identity()
    )
    mismatches = [
        key
        for key, expected_value in expected.items()
        if str(observed.get(key)) != str(expected_value)
    ]
    return {
        "artifact_name": artifact_name,
        "identity_role": role,
        "identity_valid": bool(valid and not mismatches),
        "validation_method": validation_method,
        "family": observed.get("family"),
        "model_name": observed.get("model_name"),
        "feature_group": observed.get("feature_group"),
        "temporal_weighting_policy": observed.get("temporal_weighting_policy"),
        "expected_family": expected["family"],
        "expected_model_name": expected["model_name"],
        "expected_feature_group": expected["feature_group"],
        "expected_temporal_weighting_policy": expected["temporal_weighting_policy"],
        "mismatch_reason": ";".join(mismatches) if mismatches else "",
    }


def _identity_valid(identity: pd.DataFrame, role: str) -> bool:
    if identity.empty:
        return False
    scoped = identity[identity["identity_role"].astype(str).eq(role)]
    return bool(not scoped.empty and scoped["identity_valid"].fillna(False).all())


def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float | None:
    if frame.empty or value_col not in frame or weight_col not in frame:
        return None
    values = pd.to_numeric(frame[value_col], errors="coerce")
    weights = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0)
    valid = values.notna() & weights.gt(0)
    if not valid.any():
        return None
    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def _share_rows_by_group(row_errors: pd.DataFrame, group_col: str) -> dict[object, float]:
    if row_errors.empty or group_col not in row_errors:
        return {}
    return {
        key: float(group["candidate_better"].mean())
        for key, group in row_errors.groupby(group_col, dropna=False)
    }


def _event_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "season": _json_clean(row["season"]),
                "event": row["event"],
                "event_slug": row["event_slug"],
                "delta_mae": _number_or_none(row["delta_mae"]),
                "weighted_delta": _number_or_none(row["weighted_delta"]),
            }
        )
    return rows


def _sign_flip(baseline: float | None, candidate: float | None) -> bool:
    if baseline is None or candidate is None:
        return False
    if baseline == 0 or candidate == 0:
        return False
    return (baseline < 0) != (candidate < 0)


def _sum_bool(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def _max_int(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return int(values.max()) if not values.empty else 0


def _series_mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _first_value(frame: pd.DataFrame, column: str) -> object:
    if frame.empty or column not in frame:
        return None
    values = frame[column].dropna()
    return None if values.empty else values.iloc[0]


def _number_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _empty_plot(plt: Any, path: Path, title: str) -> bool:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, title, ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def records_for_json(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert DataFrame records to JSON-safe Python objects."""
    if frame.empty:
        return []
    return json.loads(frame.where(pd.notna(frame), None).to_json(orient="records"))


def _json_clean(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


def _relative_report_path(path: Path) -> str:
    parts = path.parts
    if "reports" in parts:
        return str(Path(*parts[parts.index("reports") :]))
    return str(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
