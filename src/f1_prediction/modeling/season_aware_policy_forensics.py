"""Forensic diagnostics for the live season-aware champion policy."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.modeling.artifact_lineage import load_static_source_verification
from f1_prediction.modeling.season_aware_validation import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    paired_bootstrap_mean_ci,
)
from f1_prediction.utils.paths import ensure_directory

FP3_CHECKPOINT = "after_fp3"
JOIN_COLUMNS: tuple[str, ...] = ("fold_id", "season", "event_slug", "checkpoint", "driver")
PREDICTION_TOLERANCE = 1e-9
CURRENT_POLICY = "current_season_only_with_prior"
UNIFORM_POLICY = "uniform"


@dataclass(frozen=True)
class SeasonAwarePolicyForensicsSummary:
    """Paths and issue counts produced by season-aware policy forensics."""

    status: str
    summary_path: Path
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    missing_inputs: tuple[str, ...]
    generation_issues: tuple[str, ...]


def create_season_aware_policy_forensics_report(
    config: DataConfig,
    *,
    harmful_switch_tolerance_sec: float = 0.05,
) -> SeasonAwarePolicyForensicsSummary:
    """Create artifact-based season-aware policy forensic reports."""
    metrics_dir = config.metrics_output_dir
    figures_dir = metrics_dir.parent / "figures"
    ensure_directory(metrics_dir)
    ensure_directory(figures_dir)

    artifacts = load_policy_forensics_artifacts(metrics_dir)
    static = artifacts["static_predictions"]
    guarded = artifacts["guarded_predictions"]
    live = artifacts["season_aware_predictions"]
    selection = artifacts["season_aware_selection"]
    weighted = artifacts["weighted_candidate_predictions"]
    default = artifacts["default_candidate_predictions"]
    static_source_verification = load_static_source_verification(metrics_dir)

    reconstruction = build_policy_fold_reconstruction(
        season_aware_predictions=live,
        season_aware_selection=selection,
        weighted_candidate_predictions=weighted,
        default_predictions=default,
    )
    event_counterfactual = build_policy_event_counterfactual(
        static_predictions=static,
        guarded_predictions=guarded,
        season_aware_predictions=live,
        weighted_candidate_predictions=weighted,
        default_candidate_predictions=default,
        season_aware_selection=selection,
        static_source_verification=static_source_verification,
    )
    selected_analysis, switch_cases = build_selected_fold_analysis(
        event_counterfactual,
        static_predictions=static,
        season_aware_predictions=live,
        season_aware_selection=selection,
        tolerance_sec=harmful_switch_tolerance_sec,
        static_source_verification=static_source_verification,
    )
    guardrail_event_level, guardrail_summary = simulate_prior_only_guardrails(
        event_counterfactual,
        harmful_switch_tolerance_sec=harmful_switch_tolerance_sec,
    )

    table_frames = {
        "season_aware_policy_fold_reconstruction.csv": reconstruction,
        "season_aware_policy_event_counterfactual.csv": event_counterfactual,
        "season_aware_policy_selected_fold_analysis.csv": selected_analysis,
        "season_aware_policy_switch_cases.csv": switch_cases,
        "season_aware_policy_guardrail_event_level.csv": guardrail_event_level,
        "season_aware_policy_guardrail_simulation.csv": guardrail_summary,
    }
    table_paths: list[Path] = []
    for filename, frame in table_frames.items():
        path = metrics_dir / filename
        frame.to_csv(path, index=False)
        table_paths.append(path)

    guardrail_summary_payload = build_guardrail_summary_payload(guardrail_summary)
    guardrail_summary_path = metrics_dir / "season_aware_policy_guardrail_summary.json"
    _write_json(guardrail_summary_path, guardrail_summary_payload)
    table_paths.append(guardrail_summary_path)

    figure_paths, figure_issues = generate_policy_forensics_figures(
        figures_dir=figures_dir,
        event_counterfactual=event_counterfactual,
        selected_analysis=selected_analysis,
        reconstruction=reconstruction,
        guardrail_summary=guardrail_summary,
    )
    summary_payload = build_policy_forensics_summary_payload(
        inputs_available=artifacts["inputs_available"],
        missing_inputs=artifacts["missing_inputs"],
        reconstruction=reconstruction,
        event_counterfactual=event_counterfactual,
        selected_analysis=selected_analysis,
        switch_cases=switch_cases,
        guardrail_summary=guardrail_summary,
        static_source_verification=static_source_verification,
        generated_tables=table_paths,
        generated_figures=figure_paths,
        generation_issues=figure_issues,
        tolerance_sec=harmful_switch_tolerance_sec,
    )
    summary_path = metrics_dir / "season_aware_policy_forensics_summary.json"
    _write_json(summary_path, summary_payload)
    table_paths.insert(0, summary_path)

    return SeasonAwarePolicyForensicsSummary(
        status=str(summary_payload["status"]),
        summary_path=summary_path,
        table_paths=tuple(table_paths),
        figure_paths=tuple(figure_paths),
        missing_inputs=tuple(artifacts["missing_inputs"]),
        generation_issues=tuple(figure_issues),
    )


def load_policy_forensics_artifacts(metrics_dir: Path) -> dict[str, Any]:
    """Load saved artifacts for policy forensics without retraining."""
    files = {
        "champion_static_predictions.parquet": metrics_dir / "champion_static_predictions.parquet",
        "champion_stabilized_nested_guarded_predictions.parquet": (
            metrics_dir / "champion_stabilized_nested_guarded_predictions.parquet"
        ),
        "champion_season_aware_nested_guarded_predictions.parquet": (
            metrics_dir / "champion_season_aware_nested_guarded_predictions.parquet"
        ),
        "champion_season_aware_nested_guarded_selection.parquet": (
            metrics_dir / "champion_season_aware_nested_guarded_selection.parquet"
        ),
        "ablation_current_season_only_with_prior_predictions.parquet": (
            metrics_dir / "ablation_current_season_only_with_prior_predictions.parquet"
        ),
        "ablation_uniform_predictions.parquet": (
            metrics_dir / "ablation_uniform_predictions.parquet"
        ),
    }
    if not files["ablation_uniform_predictions.parquet"].is_file():
        files["ablation_uniform_predictions.parquet"] = metrics_dir / "ablation_predictions.parquet"
    missing = [name for name, path in files.items() if not path.is_file()]

    return {
        "static_predictions": _read_predictions(files["champion_static_predictions.parquet"]),
        "guarded_predictions": _read_predictions(
            files["champion_stabilized_nested_guarded_predictions.parquet"]
        ),
        "season_aware_predictions": _read_predictions(
            files["champion_season_aware_nested_guarded_predictions.parquet"]
        ),
        "season_aware_selection": _read_selection(
            files["champion_season_aware_nested_guarded_selection.parquet"]
        ),
        "weighted_candidate_predictions": _read_candidate_predictions(
            files["ablation_current_season_only_with_prior_predictions.parquet"],
            temporal_policy=CURRENT_POLICY,
        ),
        "default_candidate_predictions": _read_candidate_predictions(
            files["ablation_uniform_predictions.parquet"],
            temporal_policy=UNIFORM_POLICY,
        ),
        "inputs_available": {name: path.is_file() for name, path in files.items()},
        "missing_inputs": missing,
    }


def build_policy_fold_reconstruction(
    *,
    season_aware_predictions: pd.DataFrame,
    season_aware_selection: pd.DataFrame,
    weighted_candidate_predictions: pd.DataFrame,
    default_predictions: pd.DataFrame,
    tolerance: float = PREDICTION_TOLERANCE,
) -> pd.DataFrame:
    """Reconstruct saved live FP3 predictions from selected/default sources by fold."""
    columns = _reconstruction_columns()
    folds = _expected_folds(season_aware_predictions, season_aware_selection)
    if not folds:
        return pd.DataFrame(columns=columns)
    selection_lookup = _selection_lookup(season_aware_selection)
    rows: list[dict[str, object]] = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        selected = bool(selection_lookup.get(fold_id, {}).get("season_aware_selected", False))
        source = weighted_candidate_predictions if selected else default_predictions
        candidate_source = "weighted_candidate" if selected else "default_guarded"
        saved = _fold_fp3_rows(season_aware_predictions, fold_id)
        reconstructed = _fold_fp3_rows(source, fold_id)
        merged = _compare_prediction_frames(saved, reconstructed)
        prediction_diffs = (
            merged["prediction_difference_abs"].dropna()
            if "prediction_difference_abs" in merged
            else pd.Series(dtype=float)
        )
        prediction_match_rate = (
            float(prediction_diffs.le(tolerance).mean()) if not prediction_diffs.empty else None
        )
        max_diff = float(prediction_diffs.max()) if not prediction_diffs.empty else None
        saved_mae = _mae(saved)
        reconstructed_mae = _mae(reconstructed)
        row_count_match = len(saved) == len(reconstructed)
        predictions_match = bool(
            row_count_match and prediction_match_rate is not None and prediction_match_rate == 1.0
        )
        status = (
            "matched"
            if predictions_match
            else "row_count_mismatch"
            if not row_count_match
            else "prediction_mismatch"
        )
        selection = selection_lookup.get(fold_id, {})
        rows.append(
            {
                "fold_id": fold_id,
                "season": fold.get("season"),
                "event": fold.get("event"),
                "event_slug": fold.get("event_slug"),
                "checkpoint": FP3_CHECKPOINT,
                "current_season_prior_event_count": selection.get(
                    "current_season_prior_event_count"
                ),
                "selection_reason": selection.get("season_aware_selection_reason"),
                "season_aware_selected": selected,
                "candidate_source": candidate_source,
                "default_source": "guarded_or_static_default",
                "saved_prediction_rows": int(len(saved)),
                "reconstructed_prediction_rows": int(len(reconstructed)),
                "row_count_match": row_count_match,
                "prediction_match_rate": prediction_match_rate,
                "max_abs_prediction_difference": max_diff,
                "saved_fold_mae_gap_sec": saved_mae,
                "reconstructed_fold_mae_gap_sec": reconstructed_mae,
                "fold_mae_difference_sec": _delta(saved_mae, reconstructed_mae),
                "reconstruction_status": status,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_policy_event_counterfactual(
    *,
    static_predictions: pd.DataFrame,
    guarded_predictions: pd.DataFrame,
    season_aware_predictions: pd.DataFrame,
    weighted_candidate_predictions: pd.DataFrame,
    default_candidate_predictions: pd.DataFrame,
    season_aware_selection: pd.DataFrame,
    static_source_verification: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Compute FP3 event-level live/static/guarded/candidate counterfactual metrics."""
    columns = _event_counterfactual_columns()
    if season_aware_predictions.empty:
        return pd.DataFrame(columns=columns)
    aligned = _align_prediction_sources(
        static_predictions=static_predictions,
        guarded_predictions=guarded_predictions,
        season_aware_predictions=season_aware_predictions,
        weighted_candidate_predictions=weighted_candidate_predictions,
        default_candidate_predictions=default_candidate_predictions,
    )
    if aligned.empty:
        return pd.DataFrame(columns=columns)
    selection_lookup = _selection_lookup(season_aware_selection)
    verification = static_source_verification or {}
    source_verified = bool(verification.get("static_source_verified", False))
    comparison_valid = bool(verification.get("counterfactual_comparison_valid", source_verified))
    invalid_reason = verification.get("counterfactual_invalid_reason")
    rows: list[dict[str, object]] = []
    for keys, group in aligned.groupby(
        ["fold_id", "season", "event", "event_slug"], dropna=False, sort=False
    ):
        fold_id, season, event, event_slug = keys
        selection = selection_lookup.get(int(fold_id), {})
        static_mae = _mae_from_prediction(group, "static_prediction")
        guarded_mae = _mae_from_prediction(group, "guarded_prediction")
        live_mae = _mae_from_prediction(group, "live_prediction")
        weighted_mae = _mae_from_prediction(group, "weighted_candidate_prediction")
        default_mae = _mae_from_prediction(group, "default_candidate_prediction")
        prior_count = selection.get("current_season_prior_event_count")
        rows.append(
            {
                "fold_id": int(fold_id),
                "season": season,
                "event": event,
                "event_slug": event_slug,
                "checkpoint": FP3_CHECKPOINT,
                "rows": int(len(group)),
                "static_mae_gap_sec": static_mae,
                "guarded_mae_gap_sec": guarded_mae,
                "season_aware_live_mae_gap_sec": live_mae,
                "weighted_candidate_mae_gap_sec": weighted_mae,
                "default_candidate_mae_gap_sec": default_mae,
                "delta_live_vs_static_sec": _delta(live_mae, static_mae),
                "delta_weighted_candidate_vs_static_sec": _delta(weighted_mae, static_mae),
                "delta_live_vs_guarded_sec": _delta(live_mae, guarded_mae),
                "delta_weighted_candidate_vs_guarded_sec": _delta(weighted_mae, guarded_mae),
                "delta_weighted_candidate_vs_default_sec": _delta(weighted_mae, default_mae),
                "season_aware_selected": bool(selection.get("season_aware_selected", False)),
                "selection_reason": selection.get("season_aware_selection_reason"),
                "cold_start_regime": _current_season_regime(prior_count),
                "current_season_prior_event_count": prior_count,
                "candidate_prior_folds": selection.get("season_aware_candidate_prior_folds"),
                "candidate_prior_predictions": selection.get(
                    "season_aware_candidate_prior_predictions"
                ),
                "candidate_prior_mae": selection.get("metric_scope_candidate_mae"),
                "default_prior_mae": selection.get("metric_scope_default_mae"),
                "prior_improvement_sec": selection.get("metric_scope_improvement_sec"),
                "guardrail_applied": bool(selection.get("guardrail_applied", False)),
                "guardrail_reason": selection.get("guardrail_reason"),
                "static_source_verified": source_verified,
                "static_source_verification_reason": verification.get(
                    "static_source_verification_reason"
                ),
                "counterfactual_comparison_valid": comparison_valid,
                "counterfactual_invalid_reason": invalid_reason,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_selected_fold_analysis(
    event_counterfactual: pd.DataFrame,
    *,
    static_predictions: pd.DataFrame,
    season_aware_predictions: pd.DataFrame,
    season_aware_selection: pd.DataFrame,
    tolerance_sec: float,
    static_source_verification: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build selected-fold event summary and row-level switch cases."""
    verification = static_source_verification or {}
    comparison_valid = bool(verification.get("counterfactual_comparison_valid", False))
    switch_cases = build_policy_switch_cases(
        static_predictions,
        season_aware_predictions,
        season_aware_selection,
        tolerance_sec=tolerance_sec,
        counterfactual_comparison_valid=comparison_valid,
        counterfactual_invalid_reason=verification.get("counterfactual_invalid_reason"),
    )
    columns = _selected_analysis_columns()
    if event_counterfactual.empty:
        return pd.DataFrame(columns=columns), switch_cases
    rows: list[dict[str, object]] = []
    switch_summary = _switch_summary_lookup(switch_cases)
    for item in event_counterfactual.to_dict("records"):
        key = (int(item["fold_id"]), str(item["event_slug"]))
        switches = switch_summary.get(key, {})
        delta = item.get("delta_live_vs_static_sec")
        rows.append(
            {
                **{column: item.get(column) for column in _event_counterfactual_columns()},
                "fold_category": _fold_category(item),
                "harmful_driver_switches": switches.get("harmful_driver_switches", 0),
                "beneficial_driver_switches": switches.get("beneficial_driver_switches", 0),
                "neutral_driver_switches": switches.get("neutral_driver_switches", 0),
                "event_harmful_switch": (
                    bool(pd.notna(delta) and float(delta) > tolerance_sec)
                    if item.get("counterfactual_comparison_valid")
                    else pd.NA
                ),
                "event_beneficial_switch": (
                    bool(pd.notna(delta) and float(delta) < -tolerance_sec)
                    if item.get("counterfactual_comparison_valid")
                    else pd.NA
                ),
                "mean_driver_error_delta_sec": switches.get("mean_driver_error_delta_sec"),
                "median_driver_error_delta_sec": switches.get("median_driver_error_delta_sec"),
                "worst_driver_delta_sec": switches.get("worst_driver_delta_sec"),
                "event_delta_rank": None,
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty and "delta_live_vs_static_sec" in frame:
        frame = frame.sort_values(
            "delta_live_vs_static_sec", ascending=False, na_position="last"
        ).reset_index(drop=True)
        frame["event_delta_rank"] = range(1, len(frame) + 1)
    return frame.loc[:, columns], switch_cases


def build_policy_switch_cases(
    static_predictions: pd.DataFrame,
    season_aware_predictions: pd.DataFrame,
    season_aware_selection: pd.DataFrame,
    *,
    tolerance_sec: float,
    counterfactual_comparison_valid: bool = True,
    counterfactual_invalid_reason: object = None,
) -> pd.DataFrame:
    """Create driver-level live-vs-static FP3 switch cases for selected folds."""
    columns = _switch_case_columns()
    if static_predictions.empty or season_aware_predictions.empty or season_aware_selection.empty:
        return pd.DataFrame(columns=columns)
    selected_folds = set(
        pd.to_numeric(
            season_aware_selection[
                season_aware_selection.get("checkpoint", pd.Series(dtype=str))
                .astype(str)
                .eq(FP3_CHECKPOINT)
                & season_aware_selection.get("season_aware_selected", pd.Series(dtype=bool)).fillna(
                    False
                )
            ]["fold_id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )
    static = _normalize_predictions(static_predictions)
    live = _normalize_predictions(season_aware_predictions)
    static = static[static["checkpoint"].eq(FP3_CHECKPOINT)]
    live = live[live["checkpoint"].eq(FP3_CHECKPOINT) & live["fold_id"].isin(selected_folds)]
    if static.empty or live.empty:
        return pd.DataFrame(columns=columns)
    static_columns = [
        *JOIN_COLUMNS,
        "event",
        "team",
        "quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec",
    ]
    merged = static.loc[:, static_columns].merge(
        live.loc[:, [*JOIN_COLUMNS, "predicted_quali_gap_to_pole_sec"]],
        on=list(JOIN_COLUMNS),
        how="inner",
        suffixes=("_static", "_live"),
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)
    actual = pd.to_numeric(merged["quali_gap_to_pole_sec"], errors="coerce")
    static_error = (
        pd.to_numeric(merged["predicted_quali_gap_to_pole_sec_static"], errors="coerce") - actual
    ).abs()
    live_error = (
        pd.to_numeric(merged["predicted_quali_gap_to_pole_sec_live"], errors="coerce") - actual
    ).abs()
    delta = live_error - static_error
    output = pd.DataFrame(
        {
            "fold_id": merged["fold_id"],
            "season": merged["season"],
            "event": merged["event"],
            "event_slug": merged["event_slug"],
            "checkpoint": merged["checkpoint"],
            "driver": merged["driver"],
            "team": merged["team"],
            "actual_quali_gap_to_pole_sec": actual,
            "static_predicted_quali_gap_to_pole_sec": merged[
                "predicted_quali_gap_to_pole_sec_static"
            ],
            "live_predicted_quali_gap_to_pole_sec": merged["predicted_quali_gap_to_pole_sec_live"],
            "static_abs_error_gap_sec": static_error,
            "live_abs_error_gap_sec": live_error,
            "error_delta_vs_static_sec": delta,
            "harmful_switch": delta.gt(tolerance_sec)
            if counterfactual_comparison_valid
            else pd.Series(pd.NA, index=merged.index),
            "beneficial_switch": delta.lt(-tolerance_sec)
            if counterfactual_comparison_valid
            else pd.Series(pd.NA, index=merged.index),
            "counterfactual_comparison_valid": counterfactual_comparison_valid,
            "counterfactual_invalid_reason": counterfactual_invalid_reason,
        }
    )
    return output.loc[:, columns]


def simulate_prior_only_guardrails(
    event_counterfactual: pd.DataFrame,
    *,
    harmful_switch_tolerance_sec: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate deployable prior-only guardrails without changing live policy."""
    event_columns = _guardrail_event_columns()
    summary_columns = _guardrail_summary_columns()
    if event_counterfactual.empty:
        return pd.DataFrame(columns=event_columns), pd.DataFrame(columns=summary_columns)
    policies = [
        "current_live_policy",
        "static_lock",
        "higher_margin_guardrail_0.08",
        "higher_margin_guardrail_0.10",
        "higher_margin_guardrail_0.15",
        "prior_stability_guardrail",
        "recent_prior_events_guardrail_3",
        "recent_prior_events_guardrail_5",
    ]
    rows: list[dict[str, object]] = []
    frame = event_counterfactual.sort_values("fold_id").reset_index(drop=True)
    for policy in policies:
        for item in frame.to_dict("records"):
            use_candidate, reason = _guardrail_decision(policy, item, frame)
            if policy == "current_live_policy":
                simulated_mae = item["season_aware_live_mae_gap_sec"]
            else:
                simulated_mae = (
                    item["weighted_candidate_mae_gap_sec"]
                    if use_candidate
                    else item["static_mae_gap_sec"]
                )
            rows.append(
                {
                    "policy_name": policy,
                    "policy_type": "deployable_prior_only",
                    "fold_id": item.get("fold_id"),
                    "season": item.get("season"),
                    "event": item.get("event"),
                    "event_slug": item.get("event_slug"),
                    "checkpoint": FP3_CHECKPOINT,
                    "cold_start_regime": item.get("cold_start_regime"),
                    "rows": item.get("rows"),
                    "selected_weighted_candidate": use_candidate,
                    "selection_reason": reason,
                    "static_mae_gap_sec": item.get("static_mae_gap_sec"),
                    "current_live_mae_gap_sec": item.get("season_aware_live_mae_gap_sec"),
                    "weighted_candidate_mae_gap_sec": item.get("weighted_candidate_mae_gap_sec"),
                    "simulated_policy_mae_gap_sec": simulated_mae,
                    "delta_vs_static_sec": _delta(simulated_mae, item.get("static_mae_gap_sec")),
                    "delta_vs_current_live_policy_sec": _delta(
                        simulated_mae, item.get("season_aware_live_mae_gap_sec")
                    ),
                    "retrospective_policy_simulation": True,
                }
            )
    event_level = pd.DataFrame(rows, columns=event_columns)
    return event_level, _summarize_guardrail_events(event_level)


def build_guardrail_summary_payload(summary: pd.DataFrame) -> dict[str, object]:
    """Build a compact JSON summary for guardrail simulations."""
    if summary.empty:
        return {
            "status": "partial",
            "policies_tested": [],
            "best_policy_by_fp3_mae": None,
            "all_results_retrospective_simulation": True,
        }
    ordered = summary.sort_values("fp3_mae_gap_sec", na_position="last")
    best = ordered.iloc[0].to_dict()
    return {
        "status": "complete",
        "policies_tested": sorted(summary["policy_name"].dropna().astype(str).unique()),
        "best_policy_by_fp3_mae": best,
        "all_results_retrospective_simulation": True,
        "generated_at": _utc_now(),
    }


def build_policy_forensics_summary_payload(
    *,
    inputs_available: dict[str, bool],
    missing_inputs: list[str],
    reconstruction: pd.DataFrame,
    event_counterfactual: pd.DataFrame,
    selected_analysis: pd.DataFrame,
    switch_cases: pd.DataFrame,
    guardrail_summary: pd.DataFrame,
    static_source_verification: dict[str, object] | None,
    generated_tables: list[Path],
    generated_figures: list[Path],
    generation_issues: list[str],
    tolerance_sec: float,
) -> dict[str, object]:
    """Build the high-level forensics JSON payload."""
    reconstruction_complete = (
        not reconstruction.empty and reconstruction["reconstruction_status"].eq("matched").all()
    )
    aggregate = _aggregate_reconstruction_summary(reconstruction, event_counterfactual)
    if selected_analysis.empty or "season_aware_selected" not in selected_analysis:
        selected = pd.DataFrame(columns=_selected_analysis_columns())
    else:
        selected = selected_analysis[selected_analysis["season_aware_selected"].fillna(False)]
    selected_delta = (
        pd.to_numeric(selected["delta_live_vs_static_sec"], errors="coerce")
        if "delta_live_vs_static_sec" in selected
        else pd.Series(dtype=float)
    )
    verification = static_source_verification or {}
    comparison_valid = bool(verification.get("counterfactual_comparison_valid", False))
    selected_worse = int(selected_delta.gt(tolerance_sec).sum()) if comparison_valid else None
    selected_better = int(selected_delta.lt(-tolerance_sec).sum()) if comparison_valid else None
    selected_neutral = (
        int(len(selected) - selected_worse - selected_better)
        if comparison_valid and selected_worse is not None and selected_better is not None
        else None
    )
    worst_events = (
        selected.sort_values("delta_live_vs_static_sec", ascending=False)
        .head(5)
        .loc[:, ["fold_id", "season", "event", "delta_live_vs_static_sec"]]
        .to_dict("records")
        if not selected.empty
        else []
    )
    best_guardrail = (
        guardrail_summary.sort_values("fp3_mae_gap_sec", na_position="last").iloc[0].to_dict()
        if not guardrail_summary.empty
        else {}
    )
    return {
        "status": "complete" if not event_counterfactual.empty else "partial",
        "inputs_available": inputs_available,
        "missing_inputs": missing_inputs,
        "reconstruction_summary": {
            **aggregate,
            "all_folds_reconstructed": bool(reconstruction_complete),
            "matched_folds": int(reconstruction["reconstruction_status"].eq("matched").sum())
            if not reconstruction.empty
            else 0,
            "folds": int(len(reconstruction)),
        },
        "event_level_summary": _event_level_summary(event_counterfactual),
        "selected_fold_summary": {
            "selected_folds": int(len(selected)),
            "selected_folds_better_than_static": selected_better,
            "selected_folds_worse_than_static": selected_worse,
            "selected_folds_neutral_vs_static": selected_neutral,
            "definitive_switch_labels_valid": comparison_valid,
            "static_source_verified": bool(verification.get("static_source_verified", False)),
            "static_source_verification_reason": verification.get(
                "static_source_verification_reason"
            ),
            "counterfactual_invalid_reason": verification.get("counterfactual_invalid_reason"),
            "worst_selected_events": worst_events,
        },
        "harmful_switch_summary": _switch_case_summary(switch_cases),
        "static_source_verification": verification,
        "guardrail_simulation_summary": {
            "policies_tested": sorted(
                guardrail_summary["policy_name"].dropna().astype(str).unique()
            )
            if not guardrail_summary.empty
            else [],
            "best_policy_by_fp3_mae": best_guardrail,
            "all_results_retrospective_simulation": True,
        },
        "main_findings": _main_findings(
            reconstruction_complete=reconstruction_complete,
            selected_worse=selected_worse,
            selected_better=selected_better,
            event_counterfactual=event_counterfactual,
            guardrail_summary=guardrail_summary,
            counterfactual_comparison_valid=comparison_valid,
        ),
        "recommendation": "retain_static_policy",
        "generated_outputs": {
            "metrics": [_relative_report_path(path) for path in generated_tables],
            "figures": [_relative_report_path(path) for path in generated_figures],
        },
        "generation_issues": generation_issues,
        "generated_at": _utc_now(),
    }


def generate_policy_forensics_figures(
    *,
    figures_dir: Path,
    event_counterfactual: pd.DataFrame,
    selected_analysis: pd.DataFrame,
    reconstruction: pd.DataFrame,
    guardrail_summary: pd.DataFrame,
) -> tuple[list[Path], list[str]]:
    """Generate static matplotlib figures for policy forensics."""
    try:
        ensure_directory(figures_dir / ".matplotlib")
        ensure_directory(figures_dir / ".cache")
        os.environ.setdefault("MPLCONFIGDIR", str(figures_dir / ".matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(figures_dir / ".cache"))
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return [], [f"matplotlib_unavailable: {exc}"]

    specs = (
        (
            "season_aware_policy_selected_vs_static_event_delta.png",
            lambda path: _plot_selected_vs_static_delta(plt, selected_analysis, path),
        ),
        (
            "season_aware_policy_selected_switch_balance.png",
            lambda path: _plot_switch_balance(plt, selected_analysis, path),
        ),
        (
            "season_aware_policy_fold_reconstruction_status.png",
            lambda path: _plot_reconstruction_status(plt, reconstruction, path),
        ),
        (
            "season_aware_policy_guardrail_fp3_mae.png",
            lambda path: _plot_guardrail_mae(plt, guardrail_summary, path),
        ),
        (
            "season_aware_policy_guardrail_selection_rate.png",
            lambda path: _plot_guardrail_selection_rate(plt, guardrail_summary, path),
        ),
        (
            "season_aware_policy_prior_improvement_vs_realized_delta.png",
            lambda path: _plot_prior_vs_realized(plt, event_counterfactual, path),
        ),
    )
    paths: list[Path] = []
    issues: list[str] = []
    for filename, writer in specs:
        path = figures_dir / filename
        try:
            if writer(path):
                paths.append(path)
        except Exception as exc:
            issues.append(f"{filename}: {exc}")
    return paths, issues


def _align_prediction_sources(
    *,
    static_predictions: pd.DataFrame,
    guarded_predictions: pd.DataFrame,
    season_aware_predictions: pd.DataFrame,
    weighted_candidate_predictions: pd.DataFrame,
    default_candidate_predictions: pd.DataFrame,
) -> pd.DataFrame:
    base = _source_frame(static_predictions, "static_prediction")
    for frame, column in (
        (guarded_predictions, "guarded_prediction"),
        (season_aware_predictions, "live_prediction"),
        (weighted_candidate_predictions, "weighted_candidate_prediction"),
        (default_candidate_predictions, "default_candidate_prediction"),
    ):
        source = _source_frame(frame, column)
        if base.empty:
            base = source
            continue
        base = base.merge(
            source.loc[:, [*JOIN_COLUMNS, column]],
            on=list(JOIN_COLUMNS),
            how="inner",
        )
    return base


def _source_frame(frame: pd.DataFrame, prediction_name: str) -> pd.DataFrame:
    normalized = _normalize_predictions(frame)
    empty_columns = [*JOIN_COLUMNS, "event", "team", "quali_gap_to_pole_sec", prediction_name]
    if normalized.empty:
        return pd.DataFrame(columns=empty_columns)
    rows = normalized[normalized["checkpoint"].eq(FP3_CHECKPOINT)].copy()
    if rows.empty:
        return pd.DataFrame(columns=empty_columns)
    columns = [
        *JOIN_COLUMNS,
        "event",
        "team",
        "quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec",
    ]
    result = rows.loc[:, columns].rename(
        columns={"predicted_quali_gap_to_pole_sec": prediction_name}
    )
    return result.drop_duplicates(list(JOIN_COLUMNS), keep="last")


def _compare_prediction_frames(saved: pd.DataFrame, reconstructed: pd.DataFrame) -> pd.DataFrame:
    if saved.empty or reconstructed.empty:
        return pd.DataFrame(columns=[*JOIN_COLUMNS, "prediction_difference_abs"])
    saved_cols = [*JOIN_COLUMNS, "predicted_quali_gap_to_pole_sec"]
    recon_cols = [*JOIN_COLUMNS, "predicted_quali_gap_to_pole_sec"]
    merged = saved.loc[:, saved_cols].merge(
        reconstructed.loc[:, recon_cols],
        on=list(JOIN_COLUMNS),
        how="inner",
        suffixes=("_saved", "_reconstructed"),
    )
    if merged.empty:
        return pd.DataFrame(columns=[*JOIN_COLUMNS, "prediction_difference_abs"])
    merged["prediction_difference_abs"] = (
        pd.to_numeric(merged["predicted_quali_gap_to_pole_sec_saved"], errors="coerce")
        - pd.to_numeric(merged["predicted_quali_gap_to_pole_sec_reconstructed"], errors="coerce")
    ).abs()
    return merged


def _guardrail_decision(
    policy_name: str,
    item: dict[str, object],
    event_counterfactual: pd.DataFrame,
) -> tuple[bool, str]:
    if policy_name == "current_live_policy":
        return bool(item.get("season_aware_selected", False)), str(item.get("selection_reason"))
    if policy_name == "static_lock":
        return False, "static_lock"
    if _is_blocked_by_live_gates(item):
        return False, str(item.get("selection_reason") or "blocked_by_live_gate")
    prior_improvement = _float_or_none(item.get("prior_improvement_sec"))
    if policy_name.startswith("higher_margin_guardrail_"):
        margin = float(policy_name.rsplit("_", 1)[1])
        return (
            bool(prior_improvement is not None and prior_improvement >= margin),
            f"prior_improvement_margin_{margin}",
        )
    prior = _prior_event_history(event_counterfactual, int(item["fold_id"]))
    if policy_name == "prior_stability_guardrail":
        deltas = prior["delta_weighted_candidate_vs_default_sec"].dropna().astype(float)
        if len(deltas) < 3:
            return False, "insufficient_prior_event_history"
        improved = int(deltas.lt(0).sum())
        median_delta = float(deltas.median())
        harmful_share = float(deltas.gt(0.10).mean())
        use_candidate = bool(improved >= 3 and median_delta <= -0.05 and harmful_share <= 0.25)
        return use_candidate, "prior_stability_guardrail"
    if policy_name.startswith("recent_prior_events_guardrail_"):
        n_events = int(policy_name.rsplit("_", 1)[1])
        deltas = (
            prior.tail(n_events)["delta_weighted_candidate_vs_default_sec"].dropna().astype(float)
        )
        if len(deltas) < n_events:
            return False, f"insufficient_recent_prior_events_{n_events}"
        return bool(float(deltas.mean()) <= -0.05), f"recent_prior_events_{n_events}"
    return False, "unknown_policy"


def _is_blocked_by_live_gates(item: dict[str, object]) -> bool:
    reason = str(item.get("selection_reason"))
    if reason in {"cold_start", "season_aware_cold_start", "insufficient_candidate_history"}:
        return True
    prior_improvement = _float_or_none(item.get("prior_improvement_sec"))
    return prior_improvement is None or prior_improvement < 0.05


def _prior_event_history(event_counterfactual: pd.DataFrame, fold_id: int) -> pd.DataFrame:
    return event_counterfactual[
        pd.to_numeric(event_counterfactual["fold_id"], errors="coerce").lt(fold_id)
    ].sort_values("fold_id")


def _summarize_guardrail_events(events: pd.DataFrame) -> pd.DataFrame:
    columns = _guardrail_summary_columns()
    if events.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for policy, group in events.groupby("policy_name", sort=False):
        row_weight = pd.to_numeric(group["rows"], errors="coerce").fillna(0)
        simulated_mae = _weighted_mean(group["simulated_policy_mae_gap_sec"], row_weight)
        static_mae = _weighted_mean(group["static_mae_gap_sec"], row_weight)
        live_mae = _weighted_mean(group["current_live_mae_gap_sec"], row_weight)
        delta_vs_static = group["delta_vs_static_sec"].dropna().astype(float)
        delta_vs_live = group["delta_vs_current_live_policy_sec"].dropna().astype(float)
        bootstrap_static = paired_bootstrap_mean_ci(
            delta_vs_static.tolist(),
            seed=BOOTSTRAP_SEED,
            iterations=BOOTSTRAP_ITERATIONS,
        )
        bootstrap_live = paired_bootstrap_mean_ci(
            delta_vs_live.tolist(),
            seed=BOOTSTRAP_SEED,
            iterations=BOOTSTRAP_ITERATIONS,
        )
        selected_count = int(group["selected_weighted_candidate"].fillna(False).astype(bool).sum())
        rows.append(
            {
                "policy_name": policy,
                "policy_type": "deployable_prior_only",
                "events": int(len(group)),
                "rows": int(row_weight.sum()),
                "fp3_mae_gap_sec": simulated_mae,
                "delta_vs_static_sec": _delta(simulated_mae, static_mae),
                "delta_vs_current_live_policy_sec": _delta(simulated_mae, live_mae),
                "event_level_mean_delta_vs_static_sec": _mean_or_none(delta_vs_static),
                "event_level_median_delta_vs_static_sec": _median_or_none(delta_vs_static),
                "share_events_improved_vs_static": float(delta_vs_static.lt(0).mean())
                if not delta_vs_static.empty
                else None,
                "selected_event_count": selected_count,
                "selection_rate": float(selected_count / len(group)) if len(group) else None,
                "worst_event_delta_vs_static_sec": _number_or_none(delta_vs_static.max()),
                "bootstrap_vs_static_ci_low": bootstrap_static.get("ci_low"),
                "bootstrap_vs_static_ci_high": bootstrap_static.get("ci_high"),
                "bootstrap_vs_live_ci_low": bootstrap_live.get("ci_low"),
                "bootstrap_vs_live_ci_high": bootstrap_live.get("ci_high"),
                "cold_start_selected_events": int(
                    group[
                        group["cold_start_regime"].eq("cold_start")
                        & group["selected_weighted_candidate"].fillna(False).astype(bool)
                    ].shape[0]
                ),
                "early_season_selected_events": int(
                    group[
                        group["cold_start_regime"].eq("early_season")
                        & group["selected_weighted_candidate"].fillna(False).astype(bool)
                    ].shape[0]
                ),
                "established_season_selected_events": int(
                    group[
                        group["cold_start_regime"].eq("established_season")
                        & group["selected_weighted_candidate"].fillna(False).astype(bool)
                    ].shape[0]
                ),
                "retrospective_policy_simulation": True,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _switch_summary_lookup(switch_cases: pd.DataFrame) -> dict[tuple[int, str], dict[str, object]]:
    if switch_cases.empty:
        return {}
    lookup: dict[tuple[int, str], dict[str, object]] = {}
    for keys, group in switch_cases.groupby(["fold_id", "event_slug"], dropna=False):
        fold_id, event_slug = keys
        delta = group["error_delta_vs_static_sec"].dropna().astype(float)
        valid = bool(group["counterfactual_comparison_valid"].fillna(False).astype(bool).all())
        harmful = int(group["harmful_switch"].eq(True).sum()) if valid else None
        beneficial = int(group["beneficial_switch"].eq(True).sum()) if valid else None
        neutral = int(len(group) - harmful - beneficial) if valid else None
        lookup[(int(fold_id), str(event_slug))] = {
            "harmful_driver_switches": harmful,
            "beneficial_driver_switches": beneficial,
            "neutral_driver_switches": neutral,
            "mean_driver_error_delta_sec": _mean_or_none(delta),
            "median_driver_error_delta_sec": _median_or_none(delta),
            "worst_driver_delta_sec": _number_or_none(delta.max()),
        }
    return lookup


def _selection_lookup(selection: pd.DataFrame) -> dict[int, dict[str, object]]:
    if selection.empty or "fold_id" not in selection:
        return {}
    frame = selection.copy()
    if "checkpoint" in frame:
        frame = frame[frame["checkpoint"].astype(str).eq(FP3_CHECKPOINT)]
    lookup: dict[int, dict[str, object]] = {}
    for row in frame.to_dict("records"):
        fold_id = int(row["fold_id"])
        lookup[fold_id] = row
    return lookup


def _expected_folds(
    season_aware_predictions: pd.DataFrame,
    season_aware_selection: pd.DataFrame,
) -> list[dict[str, object]]:
    frames: list[pd.DataFrame] = []
    for frame in (season_aware_predictions, season_aware_selection):
        if frame.empty or "fold_id" not in frame:
            continue
        rows = frame.copy()
        if "checkpoint" in rows:
            rows = rows[rows["checkpoint"].astype(str).eq(FP3_CHECKPOINT)]
        for column in ("season", "event", "event_slug"):
            if column not in rows:
                rows[column] = pd.NA
        frames.append(rows.loc[:, ["fold_id", "season", "event", "event_slug"]])
    if not frames:
        return []
    folds = pd.concat(frames, ignore_index=True).drop_duplicates("fold_id", keep="first")
    folds["fold_id"] = pd.to_numeric(folds["fold_id"], errors="coerce")
    folds = folds[folds["fold_id"].notna()].copy()
    folds["fold_id"] = folds["fold_id"].astype(int)
    return folds.sort_values("fold_id").to_dict("records")


def _fold_fp3_rows(frame: pd.DataFrame, fold_id: int) -> pd.DataFrame:
    normalized = _normalize_predictions(frame)
    if normalized.empty:
        return normalized
    return normalized[
        normalized["checkpoint"].eq(FP3_CHECKPOINT) & normalized["fold_id"].eq(fold_id)
    ]


def _normalize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    columns = _prediction_columns()
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    result = frame.copy()
    if "event" not in result:
        result["event"] = result.get("event_slug", pd.NA)
    if "team" not in result:
        result["team"] = result.get("team_key", pd.NA)
    for column in columns:
        if column not in result:
            result[column] = pd.NA
    result = result.loc[:, columns].copy()
    result["fold_id"] = pd.to_numeric(result["fold_id"], errors="coerce")
    result = result[result["fold_id"].notna()].copy()
    result["fold_id"] = result["fold_id"].astype(int)
    return result


def _read_predictions(path: Path) -> pd.DataFrame:
    if path.is_file():
        return _normalize_predictions(pd.read_parquet(path))
    return _normalize_predictions(pd.DataFrame())


def _read_candidate_predictions(path: Path, *, temporal_policy: str) -> pd.DataFrame:
    frame = _read_predictions(path)
    if frame.empty:
        return frame
    frame["temporal_weighting_policy"] = temporal_policy
    source = pd.read_parquet(path) if path.is_file() else pd.DataFrame()
    if "model_name" in source:
        mask = source["model_name"].astype(str).eq("random_forest")
        if "feature_group" in source:
            mask &= source["feature_group"].astype(str).eq("base_plus_relative")
        source = source[mask].copy()
        frame = _normalize_predictions(source)
        frame["temporal_weighting_policy"] = temporal_policy
    return frame


def _read_selection(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame(columns=_selection_columns())


def _mae(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    actual = pd.to_numeric(frame["quali_gap_to_pole_sec"], errors="coerce")
    predicted = pd.to_numeric(frame["predicted_quali_gap_to_pole_sec"], errors="coerce")
    errors = (predicted - actual).abs().dropna()
    return float(errors.mean()) if not errors.empty else None


def _mae_from_prediction(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    actual = pd.to_numeric(frame["quali_gap_to_pole_sec"], errors="coerce")
    predicted = pd.to_numeric(frame[column], errors="coerce")
    errors = (predicted - actual).abs().dropna()
    return float(errors.mean()) if not errors.empty else None


def _aggregate_reconstruction_summary(
    reconstruction: pd.DataFrame,
    event_counterfactual: pd.DataFrame,
) -> dict[str, object]:
    if reconstruction.empty:
        return {
            "saved_fp3_mae_gap_sec": None,
            "reconstructed_fp3_mae_gap_sec": None,
            "aggregate_mae_difference_sec": None,
        }
    row_weight = pd.to_numeric(reconstruction["saved_prediction_rows"], errors="coerce").fillna(0)
    saved = _weighted_mean(reconstruction["saved_fold_mae_gap_sec"], row_weight)
    reconstructed = _weighted_mean(reconstruction["reconstructed_fold_mae_gap_sec"], row_weight)
    payload = {
        "saved_fp3_mae_gap_sec": saved,
        "reconstructed_fp3_mae_gap_sec": reconstructed,
        "aggregate_mae_difference_sec": _delta(saved, reconstructed),
    }
    if not event_counterfactual.empty:
        static = _weighted_mean(
            event_counterfactual["static_mae_gap_sec"],
            pd.to_numeric(event_counterfactual["rows"], errors="coerce").fillna(0),
        )
        live = _weighted_mean(
            event_counterfactual["season_aware_live_mae_gap_sec"],
            pd.to_numeric(event_counterfactual["rows"], errors="coerce").fillna(0),
        )
        guarded = _weighted_mean(
            event_counterfactual["guarded_mae_gap_sec"],
            pd.to_numeric(event_counterfactual["rows"], errors="coerce").fillna(0),
        )
        payload.update(
            {
                "static_fp3_mae_gap_sec": static,
                "guarded_fp3_mae_gap_sec": guarded,
                "delta_live_vs_static_sec": _delta(live, static),
                "delta_live_vs_guarded_sec": _delta(live, guarded),
            }
        )
    return payload


def _event_level_summary(event_counterfactual: pd.DataFrame) -> dict[str, object]:
    if event_counterfactual.empty:
        return {}
    selected = event_counterfactual[
        event_counterfactual["season_aware_selected"].fillna(False).astype(bool)
    ]
    delta = event_counterfactual["delta_live_vs_static_sec"].dropna().astype(float)
    bootstrap = paired_bootstrap_mean_ci(
        delta.tolist(),
        seed=BOOTSTRAP_SEED,
        iterations=BOOTSTRAP_ITERATIONS,
    )
    return {
        "events": int(len(event_counterfactual)),
        "selected_events": int(len(selected)),
        "mean_event_delta_vs_static_sec": _mean_or_none(delta),
        "median_event_delta_vs_static_sec": _median_or_none(delta),
        "share_events_improved_vs_static": float(delta.lt(0).mean()) if not delta.empty else None,
        "paired_bootstrap_delta_vs_static": bootstrap,
    }


def _switch_case_summary(switch_cases: pd.DataFrame) -> dict[str, object]:
    if switch_cases.empty:
        return {
            "rows": 0,
            "harmful_driver_switches": 0,
            "beneficial_driver_switches": 0,
            "definitive_switch_labels_valid": False,
        }
    valid = bool(switch_cases["counterfactual_comparison_valid"].fillna(False).all())
    return {
        "rows": int(len(switch_cases)),
        "harmful_driver_switches": int(switch_cases["harmful_switch"].fillna(False).sum())
        if valid
        else None,
        "beneficial_driver_switches": int(switch_cases["beneficial_switch"].fillna(False).sum())
        if valid
        else None,
        "definitive_switch_labels_valid": valid,
        "counterfactual_invalid_reason": None
        if valid
        else _first_non_null(switch_cases.get("counterfactual_invalid_reason")),
        "mean_error_delta_vs_static_sec": _mean_or_none(
            switch_cases["error_delta_vs_static_sec"].dropna().astype(float)
        ),
        "worst_driver_delta_sec": _number_or_none(
            switch_cases["error_delta_vs_static_sec"].dropna().astype(float).max()
        ),
    }


def _main_findings(
    *,
    reconstruction_complete: bool,
    selected_worse: int | None,
    selected_better: int | None,
    event_counterfactual: pd.DataFrame,
    guardrail_summary: pd.DataFrame,
    counterfactual_comparison_valid: bool,
) -> list[str]:
    findings: list[str] = []
    if reconstruction_complete:
        findings.append("Saved season-aware FP3 predictions are exactly reconstructable by fold.")
    if not event_counterfactual.empty:
        live = _weighted_mean(
            event_counterfactual["season_aware_live_mae_gap_sec"],
            pd.to_numeric(event_counterfactual["rows"], errors="coerce").fillna(0),
        )
        static = _weighted_mean(
            event_counterfactual["static_mae_gap_sec"],
            pd.to_numeric(event_counterfactual["rows"], errors="coerce").fillna(0),
        )
        findings.append(
            f"Live season-aware FP3 MAE is {live:.3f} sec versus static {static:.3f} sec."
        )
    if counterfactual_comparison_valid:
        findings.append(
            f"Among selected weighted-candidate folds, {selected_better} improved versus static "
            f"and {selected_worse} were harmful beyond the configured tolerance."
        )
    else:
        findings.append(
            "Selected-fold harmful/beneficial labels versus static are not definitive because "
            "static source lineage is not verified."
        )
    if not guardrail_summary.empty:
        best = guardrail_summary.sort_values("fp3_mae_gap_sec", na_position="last").iloc[0]
        findings.append(
            "Prior-only guardrail simulations are retrospective; the best tested policy reached "
            f"FP3 MAE {float(best['fp3_mae_gap_sec']):.3f} sec."
        )
    return findings


def _fold_category(item: dict[str, object]) -> str:
    if bool(item.get("season_aware_selected", False)):
        return "selected_weighted_candidate"
    reason = str(item.get("selection_reason"))
    if reason in {"cold_start", "season_aware_cold_start"}:
        return "cold_start"
    if reason == "insufficient_candidate_history":
        return "history_blocked"
    if reason == "margin_not_met":
        return "margin_blocked"
    return "default_retained"


def _current_season_regime(value: object) -> str:
    count = int(value) if pd.notna(value) else 0
    if count < 5:
        return "cold_start"
    if count <= 8:
        return "early_season"
    return "established_season"


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    weight = pd.to_numeric(weights, errors="coerce")
    valid = numeric.notna() & weight.notna() & weight.gt(0)
    if not valid.any():
        return None
    return float((numeric[valid] * weight[valid]).sum() / weight[valid].sum())


def _mean_or_none(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _median_or_none(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.median()) if not values.empty else None


def _delta(value: object, baseline: object) -> float | None:
    left = _float_or_none(value)
    right = _float_or_none(baseline)
    if left is None or right is None:
        return None
    return float(left - right)


def _float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _number_or_none(value: object) -> float | None:
    return _float_or_none(value)


def _first_non_null(series: pd.Series | None) -> object:
    if series is None:
        return None
    values = series.dropna()
    return values.iloc[0] if not values.empty else None


def _relative_report_path(path: Path) -> str:
    parts = path.parts
    if "reports" in parts:
        index = parts.index("reports")
        return str(Path(*parts[index:]))
    return str(path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _plot_selected_vs_static_delta(plt: Any, selected_analysis: pd.DataFrame, path: Path) -> bool:
    selected = selected_analysis[
        selected_analysis["season_aware_selected"].fillna(False).astype(bool)
    ].copy()
    if selected.empty:
        return False
    selected = selected.sort_values("delta_live_vs_static_sec", ascending=False)
    labels = selected["season"].astype(str) + " " + selected["event"].astype(str)
    plt.figure(figsize=(12, 5))
    plt.bar(labels, selected["delta_live_vs_static_sec"], color="#e45756")
    plt.axhline(0, color="#333333", linewidth=1)
    plt.title("Selected Season-Aware FP3 Event Delta vs Static")
    plt.ylabel("MAE delta (sec)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _plot_switch_balance(plt: Any, selected_analysis: pd.DataFrame, path: Path) -> bool:
    selected = selected_analysis[
        selected_analysis["season_aware_selected"].fillna(False).astype(bool)
    ].copy()
    if selected.empty:
        return False
    totals = {
        "harmful": int(selected["harmful_driver_switches"].fillna(0).sum()),
        "beneficial": int(selected["beneficial_driver_switches"].fillna(0).sum()),
        "neutral": int(selected["neutral_driver_switches"].fillna(0).sum()),
    }
    plt.figure(figsize=(6, 4))
    plt.bar(totals.keys(), totals.values(), color=["#e45756", "#54a24b", "#bab0ab"])
    plt.title("Selected Fold Driver-Level Switch Balance")
    plt.ylabel("Driver rows")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _plot_reconstruction_status(plt: Any, reconstruction: pd.DataFrame, path: Path) -> bool:
    if reconstruction.empty:
        return False
    counts = reconstruction["reconstruction_status"].astype(str).value_counts()
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index, counts.values, color="#4c78a8")
    plt.title("Season-Aware Fold Reconstruction Status")
    plt.ylabel("FP3 folds")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _plot_guardrail_mae(plt: Any, guardrail_summary: pd.DataFrame, path: Path) -> bool:
    if guardrail_summary.empty:
        return False
    frame = guardrail_summary.sort_values("fp3_mae_gap_sec")
    plt.figure(figsize=(10, 5))
    plt.bar(frame["policy_name"], frame["fp3_mae_gap_sec"], color="#4c78a8")
    plt.title("Prior-Only Guardrail Simulation FP3 MAE")
    plt.ylabel("MAE (sec)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _plot_guardrail_selection_rate(plt: Any, guardrail_summary: pd.DataFrame, path: Path) -> bool:
    if guardrail_summary.empty:
        return False
    frame = guardrail_summary.sort_values("selection_rate", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(frame["policy_name"], frame["selection_rate"], color="#72b7b2")
    plt.title("Prior-Only Guardrail Selection Rate")
    plt.ylabel("Share of FP3 events")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _plot_prior_vs_realized(plt: Any, event_counterfactual: pd.DataFrame, path: Path) -> bool:
    if event_counterfactual.empty:
        return False
    frame = event_counterfactual.dropna(
        subset=["prior_improvement_sec", "delta_weighted_candidate_vs_static_sec"]
    )
    if frame.empty:
        return False
    plt.figure(figsize=(6, 5))
    plt.scatter(
        frame["prior_improvement_sec"],
        frame["delta_weighted_candidate_vs_static_sec"],
        c=frame["season_aware_selected"].astype(bool).map({True: "#e45756", False: "#4c78a8"}),
        alpha=0.8,
    )
    plt.axhline(0, color="#333333", linewidth=1)
    plt.axvline(0.05, color="#888888", linestyle="--", linewidth=1)
    plt.title("Prior Improvement vs Realized Candidate Delta")
    plt.xlabel("Prior improvement vs default (sec)")
    plt.ylabel("Weighted candidate delta vs static (sec)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _prediction_columns() -> list[str]:
    return [
        "fold_id",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "driver",
        "team",
        "quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec",
    ]


def _selection_columns() -> list[str]:
    return [
        "fold_id",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "season_aware_selected",
        "season_aware_selection_reason",
    ]


def _reconstruction_columns() -> list[str]:
    return [
        "fold_id",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "current_season_prior_event_count",
        "selection_reason",
        "season_aware_selected",
        "candidate_source",
        "default_source",
        "saved_prediction_rows",
        "reconstructed_prediction_rows",
        "row_count_match",
        "prediction_match_rate",
        "max_abs_prediction_difference",
        "saved_fold_mae_gap_sec",
        "reconstructed_fold_mae_gap_sec",
        "fold_mae_difference_sec",
        "reconstruction_status",
    ]


def _event_counterfactual_columns() -> list[str]:
    return [
        "fold_id",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "rows",
        "static_mae_gap_sec",
        "guarded_mae_gap_sec",
        "season_aware_live_mae_gap_sec",
        "weighted_candidate_mae_gap_sec",
        "default_candidate_mae_gap_sec",
        "delta_live_vs_static_sec",
        "delta_weighted_candidate_vs_static_sec",
        "delta_live_vs_guarded_sec",
        "delta_weighted_candidate_vs_guarded_sec",
        "delta_weighted_candidate_vs_default_sec",
        "season_aware_selected",
        "selection_reason",
        "cold_start_regime",
        "current_season_prior_event_count",
        "candidate_prior_folds",
        "candidate_prior_predictions",
        "candidate_prior_mae",
        "default_prior_mae",
        "prior_improvement_sec",
        "guardrail_applied",
        "guardrail_reason",
        "static_source_verified",
        "static_source_verification_reason",
        "counterfactual_comparison_valid",
        "counterfactual_invalid_reason",
    ]


def _selected_analysis_columns() -> list[str]:
    return [
        *_event_counterfactual_columns(),
        "fold_category",
        "harmful_driver_switches",
        "beneficial_driver_switches",
        "neutral_driver_switches",
        "event_harmful_switch",
        "event_beneficial_switch",
        "mean_driver_error_delta_sec",
        "median_driver_error_delta_sec",
        "worst_driver_delta_sec",
        "event_delta_rank",
    ]


def _switch_case_columns() -> list[str]:
    return [
        "fold_id",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "driver",
        "team",
        "actual_quali_gap_to_pole_sec",
        "static_predicted_quali_gap_to_pole_sec",
        "live_predicted_quali_gap_to_pole_sec",
        "static_abs_error_gap_sec",
        "live_abs_error_gap_sec",
        "error_delta_vs_static_sec",
        "harmful_switch",
        "beneficial_switch",
        "counterfactual_comparison_valid",
        "counterfactual_invalid_reason",
    ]


def _guardrail_event_columns() -> list[str]:
    return [
        "policy_name",
        "policy_type",
        "fold_id",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "cold_start_regime",
        "rows",
        "selected_weighted_candidate",
        "selection_reason",
        "static_mae_gap_sec",
        "current_live_mae_gap_sec",
        "weighted_candidate_mae_gap_sec",
        "simulated_policy_mae_gap_sec",
        "delta_vs_static_sec",
        "delta_vs_current_live_policy_sec",
        "retrospective_policy_simulation",
    ]


def _guardrail_summary_columns() -> list[str]:
    return [
        "policy_name",
        "policy_type",
        "events",
        "rows",
        "fp3_mae_gap_sec",
        "delta_vs_static_sec",
        "delta_vs_current_live_policy_sec",
        "event_level_mean_delta_vs_static_sec",
        "event_level_median_delta_vs_static_sec",
        "share_events_improved_vs_static",
        "selected_event_count",
        "selection_rate",
        "worst_event_delta_vs_static_sec",
        "bootstrap_vs_static_ci_low",
        "bootstrap_vs_static_ci_high",
        "bootstrap_vs_live_ci_low",
        "bootstrap_vs_live_ci_high",
        "cold_start_selected_events",
        "early_season_selected_events",
        "established_season_selected_events",
        "retrospective_policy_simulation",
    ]
