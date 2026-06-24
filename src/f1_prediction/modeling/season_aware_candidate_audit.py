"""Artifact-based audit for the season-aware FP3 candidate eligibility path."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_prediction.config import DataConfig, SeasonAwareNestedGuardedChampionConfig
from f1_prediction.modeling.season_aware_validation import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    paired_bootstrap_mean_ci,
)
from f1_prediction.utils.paths import ensure_directory

KEY_COLUMNS: tuple[str, ...] = ("fold_id", "season", "event_slug", "checkpoint", "driver")
FP3_CHECKPOINT = "after_fp3"
CURRENT_POLICY = "current_season_only_with_prior"
UNIFORM_POLICY = "uniform"
WEIGHTED_PREDICTIONS = "ablation_current_season_only_with_prior_predictions.parquet"
WEIGHTED_METRICS = "ablation_current_season_only_with_prior_metrics.json"
UNIFORM_PREDICTIONS = "ablation_uniform_predictions.parquet"
UNIFORM_METRICS = "ablation_uniform_metrics.json"
CANONICAL_UNIFORM_PREDICTIONS = "ablation_predictions.parquet"
SEASON_AWARE_SELECTION = "champion_season_aware_nested_guarded_selection.parquet"
SEASON_AWARE_PREDICTIONS = "champion_season_aware_nested_guarded_predictions.parquet"
STATIC_CHAMPION_PREDICTIONS = "champion_static_predictions.parquet"
GUARDED_CHAMPION_PREDICTIONS = "champion_stabilized_nested_guarded_predictions.parquet"
TEMPORAL_WEIGHTING_SUMMARY = "temporal_weighting_summary.json"


@dataclass(frozen=True)
class SeasonAwareCandidateAuditSummary:
    """Paths and issue counts produced by the season-aware candidate audit."""

    status: str
    summary_path: Path
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    missing_inputs: tuple[str, ...]
    generation_issues: tuple[str, ...]


def create_season_aware_candidate_audit_report(
    config: DataConfig,
    settings: SeasonAwareNestedGuardedChampionConfig,
) -> SeasonAwareCandidateAuditSummary:
    """Create candidate eligibility, alignment, sensitivity, and summary artifacts."""
    metrics_dir = config.metrics_output_dir
    figures_dir = metrics_dir.parent / "figures"
    ensure_directory(metrics_dir)
    ensure_directory(figures_dir)

    artifacts = load_candidate_audit_artifacts(metrics_dir, settings)
    weighted = artifacts["weighted_candidate_predictions"]
    default = artifacts["default_candidate_predictions"]
    selection = artifacts["season_aware_selection"]
    composition = artifacts["training_composition"]

    alignment = build_candidate_alignment(
        weighted,
        default,
        selection,
        schema_issues=artifacts["schema_issues"],
    )
    history = build_candidate_history_by_fold(
        weighted,
        default,
        selection,
        composition,
        settings,
    )
    eligibility = build_candidate_eligibility_by_fold(history, selection, settings)
    gate_failures = build_candidate_gate_failures(eligibility)
    sensitivity, sensitivity_summary = build_candidate_gate_sensitivity(
        weighted,
        default,
        history,
        settings,
    )

    table_paths = (
        metrics_dir / "season_aware_candidate_eligibility_by_fold.csv",
        metrics_dir / "season_aware_candidate_history_by_fold.csv",
        metrics_dir / "season_aware_candidate_gate_failures.csv",
        metrics_dir / "season_aware_candidate_alignment.csv",
        metrics_dir / "season_aware_candidate_gate_sensitivity.csv",
        metrics_dir / "season_aware_candidate_gate_sensitivity_summary.csv",
    )
    eligibility.to_csv(table_paths[0], index=False)
    history.to_csv(table_paths[1], index=False)
    gate_failures.to_csv(table_paths[2], index=False)
    alignment.to_csv(table_paths[3], index=False)
    sensitivity.to_csv(table_paths[4], index=False)
    sensitivity_summary.to_csv(table_paths[5], index=False)

    figure_paths, figure_issues = generate_candidate_audit_figures(
        figures_dir=figures_dir,
        eligibility=eligibility,
        history=history,
        sensitivity_summary=sensitivity_summary,
    )
    summary_payload = build_candidate_audit_summary_payload(
        artifacts=artifacts,
        alignment=alignment,
        eligibility=eligibility,
        history=history,
        gate_failures=gate_failures,
        sensitivity_summary=sensitivity_summary,
        generated_figures=figure_paths,
        figure_issues=figure_issues,
    )
    summary_path = metrics_dir / "season_aware_candidate_audit_summary.json"
    _write_json(summary_path, summary_payload)

    return SeasonAwareCandidateAuditSummary(
        status=str(summary_payload["status"]),
        summary_path=summary_path,
        table_paths=table_paths,
        figure_paths=tuple(figure_paths),
        missing_inputs=tuple(summary_payload["missing_inputs"]),
        generation_issues=tuple(summary_payload["generation_issues"]),
    )


def load_candidate_audit_artifacts(
    metrics_dir: Path,
    settings: SeasonAwareNestedGuardedChampionConfig,
) -> dict[str, Any]:
    """Load saved artifacts used by the candidate audit without retraining."""
    required = {
        WEIGHTED_PREDICTIONS: metrics_dir / WEIGHTED_PREDICTIONS,
        WEIGHTED_METRICS: metrics_dir / WEIGHTED_METRICS,
        UNIFORM_METRICS: metrics_dir / UNIFORM_METRICS,
        SEASON_AWARE_SELECTION: metrics_dir / SEASON_AWARE_SELECTION,
    }
    uniform_prediction_path = metrics_dir / UNIFORM_PREDICTIONS
    if not uniform_prediction_path.is_file():
        uniform_prediction_path = metrics_dir / CANONICAL_UNIFORM_PREDICTIONS
    required[UNIFORM_PREDICTIONS] = uniform_prediction_path

    missing = [name for name, path in required.items() if not path.is_file()]
    optional_inputs = {
        SEASON_AWARE_PREDICTIONS: (metrics_dir / SEASON_AWARE_PREDICTIONS).is_file(),
        STATIC_CHAMPION_PREDICTIONS: (metrics_dir / STATIC_CHAMPION_PREDICTIONS).is_file(),
        GUARDED_CHAMPION_PREDICTIONS: (metrics_dir / GUARDED_CHAMPION_PREDICTIONS).is_file(),
        TEMPORAL_WEIGHTING_SUMMARY: (metrics_dir / TEMPORAL_WEIGHTING_SUMMARY).is_file(),
    }
    schema_issues: list[str] = []

    weighted = _read_predictions(
        metrics_dir / WEIGHTED_PREDICTIONS,
        family=settings.required_candidate.family,
        training_policy=CURRENT_POLICY,
        schema_issues=schema_issues,
    )
    default = _read_predictions(
        uniform_prediction_path,
        family=settings.required_candidate.family,
        training_policy=UNIFORM_POLICY,
        schema_issues=schema_issues,
    )
    selection = _read_parquet_if_exists(metrics_dir / SEASON_AWARE_SELECTION)
    if selection is None:
        selection = pd.DataFrame(columns=_selection_columns())
    weighted_metrics = _read_json_if_exists(metrics_dir / WEIGHTED_METRICS) or {}
    uniform_metrics = _read_json_if_exists(metrics_dir / UNIFORM_METRICS) or {}
    composition = _training_composition(
        {
            CURRENT_POLICY: weighted_metrics,
            UNIFORM_POLICY: uniform_metrics,
        }
    )

    return {
        "weighted_candidate_predictions": _filter_required_method(
            weighted,
            settings,
            CURRENT_POLICY,
        ),
        "default_candidate_predictions": _filter_required_method(
            default,
            settings,
            UNIFORM_POLICY,
        ),
        "season_aware_selection": selection,
        "training_composition": composition,
        "missing_inputs": missing,
        "inputs_available": {
            **{name: path.is_file() for name, path in required.items()},
            **optional_inputs,
        },
        "schema_issues": schema_issues,
        "settings": settings,
    }


def build_candidate_alignment(
    weighted: pd.DataFrame,
    default: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    schema_issues: list[str] | None = None,
) -> pd.DataFrame:
    """Report fold-level availability and row-key alignment for candidate artifacts."""
    columns = _alignment_columns()
    folds = _expected_fp3_folds(weighted, default, selection)
    if not folds:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        weighted_fold = _fold_rows(weighted, fold_id)
        default_fold = _fold_rows(default, fold_id)
        duplicate_weighted = _duplicate_key_count(weighted_fold)
        duplicate_default = _duplicate_key_count(default_fold)
        weighted_keys = _key_frame(weighted_fold)
        default_keys = _key_frame(default_fold)
        unmatched_weighted = _unmatched_count(weighted_keys, default_keys)
        unmatched_default = _unmatched_count(default_keys, weighted_keys)
        prior = weighted[
            weighted["checkpoint"].eq(FP3_CHECKPOINT)
            & pd.to_numeric(weighted["fold_id"], errors="coerce").lt(fold_id)
        ]
        rows.append(
            {
                "fold_id": fold_id,
                "season": fold.get("season"),
                "event": fold.get("event"),
                "event_slug": fold.get("event_slug"),
                "checkpoint": FP3_CHECKPOINT,
                "expected_fold": True,
                "weighted_candidate_fold_found": not weighted_fold.empty,
                "default_candidate_fold_found": not default_fold.empty,
                "weighted_candidate_rows": int(len(weighted_fold)),
                "default_candidate_rows": int(len(default_fold)),
                "weighted_candidate_duplicate_rows": duplicate_weighted,
                "default_candidate_duplicate_rows": duplicate_default,
                "unmatched_weighted_candidate_rows": unmatched_weighted,
                "unmatched_default_candidate_rows": unmatched_default,
                "historical_source_events": _event_list(prior),
                "current_event_in_history": _current_event_in_history(fold, prior),
                "schema_mismatch": "; ".join(schema_issues or []),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_candidate_history_by_fold(
    weighted: pd.DataFrame,
    default: pd.DataFrame,
    selection: pd.DataFrame,
    composition: pd.DataFrame,
    settings: SeasonAwareNestedGuardedChampionConfig,
) -> pd.DataFrame:
    """Compute prior-only candidate/default evidence available at each FP3 fold."""
    columns = _history_columns()
    folds = _expected_fp3_folds(weighted, default, selection)
    if not folds:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        weighted_current = _fold_rows(weighted, fold_id)
        default_current = _fold_rows(default, fold_id)
        weighted_prior = _prior_rows(weighted, fold_id)
        default_prior = _prior_rows(default, fold_id)
        weighted_metric = _mean_abs_error(weighted_prior)
        default_metric = _mean_abs_error(default_prior)
        composition_row = _composition_row(composition, CURRENT_POLICY, fold_id)
        prior_count = _prior_event_count(fold, selection, composition_row)
        rows.append(
            {
                "fold_id": fold_id,
                "season": fold.get("season"),
                "event": fold.get("event"),
                "event_slug": fold.get("event_slug"),
                "checkpoint": FP3_CHECKPOINT,
                "current_season_prior_event_count": prior_count,
                "weighted_candidate_artifact_available": not weighted.empty,
                "weighted_candidate_prediction_rows": int(len(weighted_current)),
                "default_candidate_prediction_rows": int(len(default_current)),
                "weighted_candidate_prior_folds": int(weighted_prior["fold_id"].nunique())
                if not weighted_prior.empty
                else 0,
                "weighted_candidate_prior_predictions": int(len(weighted_prior)),
                "default_prior_folds": int(default_prior["fold_id"].nunique())
                if not default_prior.empty
                else 0,
                "default_prior_predictions": int(len(default_prior)),
                "weighted_candidate_metric_value": weighted_metric,
                "default_metric_value": default_metric,
                "improvement_delta_sec": _safe_delta(weighted_metric, default_metric),
                "required_min_prior_candidate_folds": settings.min_prior_candidate_folds,
                "required_min_prior_candidate_predictions": (
                    settings.min_prior_candidate_predictions
                ),
                "required_min_current_season_prior_events": (
                    settings.min_current_season_prior_events
                ),
                "required_improvement_margin_sec": settings.improvement_margin_sec,
                "historical_source_events": _event_list(weighted_prior),
                "current_event_in_history": _current_event_in_history(fold, weighted_prior),
                "effective_sample_size": composition_row.get("effective_sample_size"),
                "same_season_weight_share": composition_row.get("same_season_weight_share"),
                "prior_season_weight_share": composition_row.get("prior_season_weight_share"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_candidate_eligibility_by_fold(
    history: pd.DataFrame,
    selection: pd.DataFrame,
    settings: SeasonAwareNestedGuardedChampionConfig,
) -> pd.DataFrame:
    """Apply the current live gates to fold-level candidate history."""
    columns = _eligibility_columns()
    if history.empty:
        return pd.DataFrame(columns=columns)
    selection_lookup = _selection_lookup(selection)
    rows: list[dict[str, object]] = []
    for item in history.to_dict("records"):
        cold = int(item.get("current_season_prior_event_count") or 0) >= (
            settings.min_current_season_prior_events
        )
        history_gate = (
            int(item.get("weighted_candidate_prior_folds") or 0)
            >= settings.min_prior_candidate_folds
            and int(item.get("weighted_candidate_prior_predictions") or 0)
            >= settings.min_prior_candidate_predictions
        )
        prediction_gate = bool(item.get("weighted_candidate_artifact_available")) and int(
            item.get("weighted_candidate_prediction_rows") or 0
        ) > 0
        delta = item.get("improvement_delta_sec")
        margin = pd.notna(delta) and float(delta) <= -settings.improvement_margin_sec
        eligible = bool(cold and history_gate and prediction_gate and margin)
        selected = bool(selection_lookup.get(int(item["fold_id"]), {}).get("selected", False))
        reason = _gate_reason(
            prediction_gate=prediction_gate,
            cold_start_gate=cold,
            history_gate=history_gate,
            margin_gate=margin,
            eligible=eligible,
            selected=selected,
            selection_reason=selection_lookup.get(int(item["fold_id"]), {}).get("reason"),
        )
        rows.append(
            {
                **{column: item.get(column) for column in _history_columns()},
                "cold_start_gate_passed": cold,
                "candidate_history_gate_passed": history_gate,
                "candidate_prediction_gate_passed": prediction_gate,
                "margin_gate_passed": margin,
                "candidate_eligible": eligible,
                "candidate_selected": selected,
                "selection_reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_candidate_gate_failures(eligibility: pd.DataFrame) -> pd.DataFrame:
    """Flatten failed gates into one row per fold and blocking reason."""
    columns = [
        "fold_id",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "gate_name",
        "failure_reason",
        "details",
    ]
    if eligibility.empty:
        return pd.DataFrame(columns=columns)
    gate_columns = {
        "candidate_prediction_gate_passed": "candidate_prediction_gate",
        "cold_start_gate_passed": "cold_start_gate",
        "candidate_history_gate_passed": "candidate_history_gate",
        "margin_gate_passed": "margin_gate",
    }
    rows: list[dict[str, object]] = []
    for item in eligibility.to_dict("records"):
        if bool(item.get("candidate_eligible")):
            continue
        for column, gate_name in gate_columns.items():
            if bool(item.get(column)):
                continue
            rows.append(
                {
                    "fold_id": item.get("fold_id"),
                    "season": item.get("season"),
                    "event": item.get("event"),
                    "event_slug": item.get("event_slug"),
                    "checkpoint": item.get("checkpoint"),
                    "gate_name": gate_name,
                    "failure_reason": item.get("selection_reason"),
                    "details": _failure_details(gate_name, item),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_candidate_gate_sensitivity(
    weighted: pd.DataFrame,
    default: pd.DataFrame,
    history: pd.DataFrame,
    settings: SeasonAwareNestedGuardedChampionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run retrospective gate sensitivity without changing the live policy."""
    detail_columns = _sensitivity_detail_columns()
    summary_columns = _sensitivity_summary_columns()
    aligned = _aligned_current_predictions(weighted, default)
    if aligned.empty or history.empty:
        return pd.DataFrame(columns=detail_columns), pd.DataFrame(columns=summary_columns)

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    grid = product((3, 4, 5), (3, 4, 5), (60, 80, 100), (0.0, 0.03, 0.05))
    for config_id, (min_events, min_folds, min_predictions, margin) in enumerate(grid, start=1):
        event_deltas: list[float] = []
        simulated_errors: list[pd.Series] = []
        static_errors: list[pd.Series] = []
        fold_rows: list[dict[str, object]] = []
        for item in history.to_dict("records"):
            fold_id = int(item["fold_id"])
            current = aligned[aligned["fold_id"].eq(fold_id)]
            if current.empty:
                continue
            cold = int(item.get("current_season_prior_event_count") or 0) >= min_events
            history_gate = (
                int(item.get("weighted_candidate_prior_folds") or 0) >= min_folds
                and int(item.get("weighted_candidate_prior_predictions") or 0) >= min_predictions
            )
            prediction_gate = int(item.get("weighted_candidate_prediction_rows") or 0) > 0
            delta = item.get("improvement_delta_sec")
            margin_gate = pd.notna(delta) and float(delta) <= -margin
            eligible = bool(cold and history_gate and prediction_gate and margin_gate)
            simulated = (
                current["candidate_abs_error_gap_sec"]
                if eligible
                else current["static_abs_error_gap_sec"]
            )
            static = current["static_abs_error_gap_sec"]
            event_delta = float(simulated.mean() - static.mean())
            event_deltas.append(event_delta)
            simulated_errors.append(simulated)
            static_errors.append(static)
            reason = _first_failed_gate(
                prediction_gate=prediction_gate,
                cold_start_gate=cold,
                history_gate=history_gate,
                margin_gate=margin_gate,
                eligible=eligible,
            )
            fold_rows.append(
                {
                    "config_id": config_id,
                    "min_current_season_prior_events": min_events,
                    "min_prior_candidate_folds": min_folds,
                    "min_prior_candidate_predictions": min_predictions,
                    "improvement_margin_sec": margin,
                    "fold_id": fold_id,
                    "season": item.get("season"),
                    "event": item.get("event"),
                    "event_slug": item.get("event_slug"),
                    "candidate_eligible": eligible,
                    "blocking_reason": reason,
                    "rows": int(len(current)),
                    "static_mae_gap_sec": float(static.mean()),
                    "simulated_policy_mae_gap_sec": float(simulated.mean()),
                    "delta_vs_static_sec": event_delta,
                    "retrospective_simulation": True,
                }
            )
        if not fold_rows:
            continue
        detail_rows.extend(fold_rows)
        simulated_all = pd.concat(simulated_errors, ignore_index=True)
        static_all = pd.concat(static_errors, ignore_index=True)
        bootstrap = paired_bootstrap_mean_ci(
            event_deltas,
            seed=BOOTSTRAP_SEED,
            iterations=BOOTSTRAP_ITERATIONS,
        )
        details = pd.DataFrame(fold_rows)
        eligible_folds = int(details["candidate_eligible"].sum())
        total_folds = int(len(details))
        summary_rows.append(
            {
                "config_id": config_id,
                "min_current_season_prior_events": min_events,
                "min_prior_candidate_folds": min_folds,
                "min_prior_candidate_predictions": min_predictions,
                "improvement_margin_sec": margin,
                "candidate_selection_rate": float(eligible_folds / total_folds)
                if total_folds
                else None,
                "candidate_eligible_folds": eligible_folds,
                "total_folds": total_folds,
                "cold_start_blocked_folds": int(
                    details["blocking_reason"].eq("cold_start_gate").sum()
                ),
                "history_blocked_folds": int(
                    details["blocking_reason"].eq("candidate_history_gate").sum()
                ),
                "margin_blocked_folds": int(details["blocking_reason"].eq("margin_gate").sum()),
                "fp3_retrospective_mae": float(simulated_all.mean()),
                "delta_vs_static": float(simulated_all.mean() - static_all.mean()),
                "share_fp3_events_improved": float(pd.Series(event_deltas).lt(0).mean()),
                "worst_event_delta": float(max(event_deltas)),
                "mean_event_level_delta": float(np.mean(event_deltas)),
                "median_event_level_delta": float(np.median(event_deltas)),
                "bootstrap_ci_low": bootstrap["ci_low"],
                "bootstrap_ci_high": bootstrap["ci_high"],
                "bootstrap_seed": BOOTSTRAP_SEED,
                "retrospective_simulation": True,
                "matches_live_configuration": (
                    min_events == settings.min_current_season_prior_events
                    and min_folds == settings.min_prior_candidate_folds
                    and min_predictions == settings.min_prior_candidate_predictions
                    and margin == settings.improvement_margin_sec
                ),
            }
        )
    return (
        pd.DataFrame(detail_rows, columns=detail_columns),
        pd.DataFrame(summary_rows, columns=summary_columns),
    )


def build_candidate_audit_summary_payload(
    *,
    artifacts: dict[str, Any],
    alignment: pd.DataFrame,
    eligibility: pd.DataFrame,
    history: pd.DataFrame,
    gate_failures: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    generated_figures: list[Path],
    figure_issues: list[str],
) -> dict[str, object]:
    """Build the high-level JSON payload for the audit."""
    missing = list(artifacts["missing_inputs"])
    schema_issues = list(artifacts["schema_issues"])
    available = not eligibility.empty and not artifacts["weighted_candidate_predictions"].empty
    gate_counts = (
        gate_failures["failure_reason"].astype(str).value_counts().to_dict()
        if not gate_failures.empty
        else {}
    )
    recommendation = _audit_recommendation(
        missing_inputs=missing,
        schema_issues=schema_issues,
        alignment=alignment,
        eligibility=eligibility,
        gate_counts=gate_counts,
    )
    live_selected = int(eligibility["candidate_selected"].sum()) if not eligibility.empty else 0
    eligible = int(eligibility["candidate_eligible"].sum()) if not eligibility.empty else 0
    return {
        "status": "complete" if available else "partial",
        "inputs_available": artifacts["inputs_available"],
        "missing_inputs": missing,
        "schema_issues": schema_issues,
        "candidate_availability": {
            "weighted_candidate_rows": int(len(artifacts["weighted_candidate_predictions"])),
            "default_candidate_rows": int(len(artifacts["default_candidate_predictions"])),
            "season_aware_selection_rows": int(len(artifacts["season_aware_selection"])),
        },
        "artifact_alignment_summary": _alignment_summary(alignment),
        "live_gate_summary": {
            "folds_evaluated": int(len(eligibility)),
            "candidate_eligible_folds": eligible,
            "candidate_selected_folds": live_selected,
            "weighted_candidate_selection_rate": float(live_selected / len(eligibility))
            if len(eligibility)
            else None,
            "gate_failure_reasons": gate_counts,
            "zero_selection_expected": bool(live_selected == 0 and not schema_issues and available),
        },
        "history_summary": _history_summary(history),
        "sensitivity_analysis_summary": _sensitivity_summary(sensitivity_summary),
        "main_findings": _main_findings(
            eligibility,
            alignment,
            gate_counts,
            missing,
            schema_issues,
            sensitivity_summary,
        ),
        "recommendation": recommendation,
        "generated_at": _utc_now(),
        "generated_outputs": {
            "metrics": [
                "reports/metrics/season_aware_candidate_audit_summary.json",
                "reports/metrics/season_aware_candidate_eligibility_by_fold.csv",
                "reports/metrics/season_aware_candidate_history_by_fold.csv",
                "reports/metrics/season_aware_candidate_gate_failures.csv",
                "reports/metrics/season_aware_candidate_alignment.csv",
                "reports/metrics/season_aware_candidate_gate_sensitivity.csv",
                "reports/metrics/season_aware_candidate_gate_sensitivity_summary.csv",
            ],
            "figures": [_relative_report_path(path) for path in generated_figures],
        },
        "generation_issues": [*schema_issues, *figure_issues],
        "table_row_counts": {
            "season_aware_candidate_eligibility_by_fold": int(len(eligibility)),
            "season_aware_candidate_history_by_fold": int(len(history)),
            "season_aware_candidate_gate_failures": int(len(gate_failures)),
            "season_aware_candidate_alignment": int(len(alignment)),
            "season_aware_candidate_gate_sensitivity_summary": int(len(sensitivity_summary)),
        },
    }


def generate_candidate_audit_figures(
    *,
    figures_dir: Path,
    eligibility: pd.DataFrame,
    history: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
) -> tuple[list[Path], list[str]]:
    """Generate static matplotlib figures for the candidate audit."""
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
            "season_aware_candidate_gate_failure_reasons.png",
            lambda path: _plot_gate_failures(plt, eligibility, path),
        ),
        (
            "season_aware_candidate_history_growth.png",
            lambda path: _plot_history_growth(plt, history, path),
        ),
        (
            "season_aware_candidate_metric_delta_over_folds.png",
            lambda path: _plot_metric_delta(plt, history, path),
        ),
        (
            "season_aware_candidate_sensitivity_selection_rate.png",
            lambda path: _plot_sensitivity(
                plt,
                sensitivity_summary,
                path,
                value_column="candidate_selection_rate",
                ylabel="Selection rate",
                title="Retrospective gate sensitivity selection rate",
            ),
        ),
        (
            "season_aware_candidate_sensitivity_fp3_mae.png",
            lambda path: _plot_sensitivity(
                plt,
                sensitivity_summary,
                path,
                value_column="fp3_retrospective_mae",
                ylabel="FP3 MAE (sec)",
                title="Retrospective gate sensitivity FP3 MAE",
            ),
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


def _read_predictions(
    path: Path,
    *,
    family: str,
    training_policy: str,
    schema_issues: list[str],
) -> pd.DataFrame:
    frame = _read_parquet_if_exists(path)
    if frame is None:
        return pd.DataFrame(columns=_prediction_columns())
    candidate = frame.copy()
    if "prediction_type" in candidate:
        candidate = candidate[candidate["prediction_type"].isin(["tabular", "boosted"])].copy()
    if "model_name" not in candidate and "candidate_model_name" in candidate:
        candidate["model_name"] = candidate["candidate_model_name"]
    if "feature_group" not in candidate and "candidate_feature_group" in candidate:
        candidate["feature_group"] = candidate["candidate_feature_group"]
    if "feature_group" not in candidate:
        candidate["feature_group"] = pd.NA
    if "event" not in candidate:
        candidate["event"] = candidate.get("event_slug", pd.NA)
    required = {
        *KEY_COLUMNS,
        "event",
        "model_name",
        "feature_group",
        "quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec",
    }
    missing = sorted(required - set(candidate.columns))
    if missing:
        schema_issues.append(f"{path.name}: missing columns {', '.join(missing)}")
        return pd.DataFrame(columns=_prediction_columns())
    candidate["candidate_family"] = family
    candidate["training_policy"] = training_policy
    candidate["candidate_model_name"] = candidate["model_name"].astype(str)
    candidate["candidate_feature_group"] = (
        candidate["feature_group"].astype("string").fillna("").astype(str)
    )
    if "predicted_quali_position" not in candidate:
        candidate["predicted_quali_position"] = pd.NA
    if "quali_position" not in candidate:
        candidate["quali_position"] = pd.NA
    if "team" not in candidate:
        candidate["team"] = pd.NA
    result = candidate.loc[:, _prediction_columns()].copy()
    result["fold_id"] = pd.to_numeric(result["fold_id"], errors="coerce").astype("Int64")
    result = result[result["fold_id"].notna()].copy()
    result["fold_id"] = result["fold_id"].astype(int)
    return result.drop_duplicates(
        [
            *KEY_COLUMNS,
            "candidate_family",
            "candidate_model_name",
            "candidate_feature_group",
            "training_policy",
        ],
        keep="last",
    ).reset_index(drop=True)


def _filter_required_method(
    frame: pd.DataFrame,
    settings: SeasonAwareNestedGuardedChampionConfig,
    policy: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    required = settings.required_candidate
    return frame[
        frame["checkpoint"].eq(settings.eligible_checkpoint)
        & frame["candidate_family"].eq(required.family)
        & frame["candidate_model_name"].eq(required.model_name)
        & frame["candidate_feature_group"].eq(required.feature_group or "")
        & frame["training_policy"].eq(policy)
    ].copy()


def _training_composition(metrics_by_policy: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy, payload in metrics_by_policy.items():
        for item in payload.get("training_weight_summary_by_fold", []):
            row = dict(item)
            row["training_policy"] = policy
            row["fold_id"] = int(row["fold_id"])
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=_composition_columns())
    frame = pd.DataFrame(rows)
    for column in _composition_columns():
        if column not in frame:
            frame[column] = pd.NA
    return frame.loc[:, _composition_columns()]


def _expected_fp3_folds(
    weighted: pd.DataFrame,
    default: pd.DataFrame,
    selection: pd.DataFrame,
) -> list[dict[str, object]]:
    frames: list[pd.DataFrame] = []
    for frame in (default, weighted):
        if frame.empty:
            continue
        cols = ["fold_id", "season", "event", "event_slug"]
        frames.append(frame[frame["checkpoint"].eq(FP3_CHECKPOINT)].loc[:, cols].drop_duplicates())
    if not selection.empty and {"fold_id", "season", "checkpoint"} <= set(selection.columns):
        selection_fp3 = selection[selection["checkpoint"].astype(str).eq(FP3_CHECKPOINT)].copy()
        if "event" not in selection_fp3:
            selection_fp3["event"] = selection_fp3.get("event_slug", pd.NA)
        if "event_slug" not in selection_fp3:
            selection_fp3["event_slug"] = selection_fp3.get("event", pd.NA)
        frames.append(selection_fp3.loc[:, ["fold_id", "season", "event", "event_slug"]])
    if not frames:
        return []
    folds = pd.concat(frames, ignore_index=True).drop_duplicates("fold_id", keep="first")
    folds["fold_id"] = pd.to_numeric(folds["fold_id"], errors="coerce")
    folds = folds[folds["fold_id"].notna()].copy()
    folds["fold_id"] = folds["fold_id"].astype(int)
    return folds.sort_values("fold_id").to_dict("records")


def _fold_rows(frame: pd.DataFrame, fold_id: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[frame["checkpoint"].eq(FP3_CHECKPOINT) & frame["fold_id"].eq(fold_id)].copy()


def _prior_rows(frame: pd.DataFrame, fold_id: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[frame["checkpoint"].eq(FP3_CHECKPOINT) & frame["fold_id"].lt(fold_id)].copy()


def _aligned_current_predictions(weighted: pd.DataFrame, default: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *KEY_COLUMNS,
        "season",
        "event",
        "event_slug",
        "quali_gap_to_pole_sec",
        "static_abs_error_gap_sec",
        "candidate_abs_error_gap_sec",
    ]
    if weighted.empty or default.empty:
        return pd.DataFrame(columns=columns)
    weighted_current = weighted.rename(
        columns={"predicted_quali_gap_to_pole_sec": "candidate_prediction"}
    )
    default_current = default.rename(columns={"predicted_quali_gap_to_pole_sec": "static_prediction"})
    merged = weighted_current.merge(
        default_current.loc[:, [*KEY_COLUMNS, "static_prediction"]],
        on=list(KEY_COLUMNS),
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged["static_abs_error_gap_sec"] = (
        merged["static_prediction"].astype(float) - merged["quali_gap_to_pole_sec"].astype(float)
    ).abs()
    merged["candidate_abs_error_gap_sec"] = (
        merged["candidate_prediction"].astype(float)
        - merged["quali_gap_to_pole_sec"].astype(float)
    ).abs()
    return merged.loc[:, columns]


def _selection_lookup(selection: pd.DataFrame) -> dict[int, dict[str, object]]:
    if selection.empty or "fold_id" not in selection:
        return {}
    frame = selection.copy()
    if "checkpoint" in frame:
        frame = frame[frame["checkpoint"].astype(str).eq(FP3_CHECKPOINT)]
    lookup: dict[int, dict[str, object]] = {}
    for row in frame.to_dict("records"):
        fold_id = int(row["fold_id"])
        selected = bool(row.get("season_aware_selected", False))
        if not selected and row.get("selected_temporal_weighting_policy") == CURRENT_POLICY:
            selected = True
        lookup[fold_id] = {
            "selected": selected,
            "reason": row.get("season_aware_selection_reason") or row.get("fallback_reason"),
        }
    return lookup


def _composition_row(
    composition: pd.DataFrame,
    policy: str,
    fold_id: int,
) -> dict[str, object]:
    if composition.empty:
        return {}
    rows = composition[
        composition["training_policy"].astype(str).eq(policy)
        & pd.to_numeric(composition["fold_id"], errors="coerce").eq(fold_id)
    ]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _prior_event_count(
    fold: dict[str, object],
    selection: pd.DataFrame,
    composition_row: dict[str, object],
) -> int:
    if not selection.empty and {"fold_id", "current_season_prior_event_count"} <= set(
        selection.columns
    ):
        rows = selection[
            selection["fold_id"].eq(fold["fold_id"])
            & selection["checkpoint"].astype(str).eq(FP3_CHECKPOINT)
        ]
        if not rows.empty:
            value = rows.iloc[0]["current_season_prior_event_count"]
            if pd.notna(value):
                return int(value)
    for key in ("same_season_training_events", "current_season_prior_event_count"):
        value = composition_row.get(key)
        if pd.notna(value):
            return int(value)
    return 0


def _mean_abs_error(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    actual = pd.to_numeric(frame["quali_gap_to_pole_sec"], errors="coerce")
    predicted = pd.to_numeric(frame["predicted_quali_gap_to_pole_sec"], errors="coerce")
    errors = (predicted - actual).abs().dropna()
    return float(errors.mean()) if not errors.empty else None


def _safe_delta(value: object, baseline: object) -> float | None:
    if value is None or baseline is None or pd.isna(value) or pd.isna(baseline):
        return None
    return float(value) - float(baseline)


def _gate_reason(
    *,
    prediction_gate: bool,
    cold_start_gate: bool,
    history_gate: bool,
    margin_gate: bool,
    eligible: bool,
    selected: bool,
    selection_reason: object,
) -> str:
    if selected and selection_reason:
        return str(selection_reason)
    if not prediction_gate:
        return "weighted_candidate_missing"
    if not cold_start_gate:
        return "season_aware_cold_start"
    if not history_gate:
        return "insufficient_candidate_history"
    if not margin_gate:
        return "margin_not_met"
    if eligible:
        return "eligible_for_selection"
    return str(selection_reason or "not_selected")


def _first_failed_gate(
    *,
    prediction_gate: bool,
    cold_start_gate: bool,
    history_gate: bool,
    margin_gate: bool,
    eligible: bool,
) -> str:
    if eligible:
        return "candidate_eligible"
    if not prediction_gate:
        return "candidate_prediction_gate"
    if not cold_start_gate:
        return "cold_start_gate"
    if not history_gate:
        return "candidate_history_gate"
    if not margin_gate:
        return "margin_gate"
    return "not_selected"


def _failure_details(gate_name: str, item: dict[str, object]) -> str:
    if gate_name == "cold_start_gate":
        return (
            f"current_season_prior_event_count={item.get('current_season_prior_event_count')}, "
            f"required={item.get('required_min_current_season_prior_events')}"
        )
    if gate_name == "candidate_history_gate":
        return (
            f"prior_folds={item.get('weighted_candidate_prior_folds')}, "
            f"prior_predictions={item.get('weighted_candidate_prior_predictions')}"
        )
    if gate_name == "margin_gate":
        return (
            f"improvement_delta_sec={item.get('improvement_delta_sec')}, "
            f"required_margin={item.get('required_improvement_margin_sec')}"
        )
    return f"weighted_candidate_prediction_rows={item.get('weighted_candidate_prediction_rows')}"


def _audit_recommendation(
    *,
    missing_inputs: list[str],
    schema_issues: list[str],
    alignment: pd.DataFrame,
    eligibility: pd.DataFrame,
    gate_counts: dict[str, int],
) -> str:
    required_missing = {WEIGHTED_PREDICTIONS, UNIFORM_PREDICTIONS, WEIGHTED_METRICS}
    if schema_issues or required_missing.intersection(missing_inputs):
        return "artifact_pipeline_fix_required"
    if not alignment.empty and (
        alignment["current_event_in_history"].fillna(False).astype(bool).any()
        or alignment["unmatched_default_candidate_rows"].fillna(0).astype(int).sum() > 0
    ):
        return "artifact_pipeline_fix_required"
    if eligibility.empty or int(eligibility["candidate_eligible"].sum()) == 0:
        if any(reason in gate_counts for reason in ("season_aware_cold_start", "insufficient_candidate_history")):
            return "retain_gates_and_collect_more_history"
        return "retain_static_policy"
    return "candidate_eligible_for_future_broader_validation"


def _alignment_summary(alignment: pd.DataFrame) -> dict[str, object]:
    if alignment.empty:
        return {
            "expected_folds": 0,
            "candidate_folds_found": 0,
            "missing_candidate_folds": [],
            "unmatched_rows": 0,
            "duplicate_rows": 0,
            "current_event_in_history": False,
        }
    return {
        "expected_folds": int(len(alignment)),
        "candidate_folds_found": int(alignment["weighted_candidate_fold_found"].sum()),
        "missing_candidate_folds": [
            int(value)
            for value in alignment.loc[
                ~alignment["weighted_candidate_fold_found"].fillna(False).astype(bool),
                "fold_id",
            ].tolist()
        ],
        "unmatched_rows": int(
            alignment["unmatched_weighted_candidate_rows"].fillna(0).astype(int).sum()
            + alignment["unmatched_default_candidate_rows"].fillna(0).astype(int).sum()
        ),
        "duplicate_rows": int(
            alignment["weighted_candidate_duplicate_rows"].fillna(0).astype(int).sum()
            + alignment["default_candidate_duplicate_rows"].fillna(0).astype(int).sum()
        ),
        "current_event_in_history": bool(
            alignment["current_event_in_history"].fillna(False).astype(bool).any()
        ),
    }


def _history_summary(history: pd.DataFrame) -> dict[str, object]:
    if history.empty:
        return {}
    return {
        "folds": int(len(history)),
        "max_prior_candidate_folds": int(
            history["weighted_candidate_prior_folds"].fillna(0).astype(int).max()
        ),
        "max_prior_candidate_predictions": int(
            history["weighted_candidate_prior_predictions"].fillna(0).astype(int).max()
        ),
        "mean_improvement_delta_sec": _mean_numeric(history["improvement_delta_sec"]),
    }


def _sensitivity_summary(summary: pd.DataFrame) -> dict[str, object]:
    if summary.empty:
        return {}
    non_oracle = summary.copy()
    best_mae = non_oracle.sort_values("fp3_retrospective_mae", na_position="last").iloc[0]
    best_rate = non_oracle.sort_values("candidate_selection_rate", ascending=False).iloc[0]
    live = non_oracle[non_oracle["matches_live_configuration"].astype(bool)]
    return {
        "configurations_tested": int(len(summary)),
        "best_retrospective_fp3_mae_config_id": int(best_mae["config_id"]),
        "best_retrospective_fp3_mae": float(best_mae["fp3_retrospective_mae"]),
        "highest_selection_rate_config_id": int(best_rate["config_id"]),
        "highest_selection_rate": float(best_rate["candidate_selection_rate"]),
        "live_configuration": live.iloc[0].to_dict() if not live.empty else {},
        "all_results_retrospective_simulation": True,
    }


def _main_findings(
    eligibility: pd.DataFrame,
    alignment: pd.DataFrame,
    gate_counts: dict[str, int],
    missing: list[str],
    schema_issues: list[str],
    sensitivity_summary: pd.DataFrame,
) -> list[str]:
    findings: list[str] = []
    if missing:
        findings.append("Candidate audit is partial because required saved artifacts are missing.")
    if schema_issues:
        findings.append("Candidate artifact schema mismatches prevent a complete eligibility audit.")
    if not alignment.empty and int(alignment["unmatched_default_candidate_rows"].sum()) == 0:
        findings.append("Weighted and default FP3 candidate rows align on identical fold/event/driver keys.")
    if not eligibility.empty:
        selected = int(eligibility["candidate_selected"].sum())
        eligible = int(eligibility["candidate_eligible"].sum())
        findings.append(
            f"Live season-aware gates selected the weighted candidate in {selected} of "
            f"{len(eligibility)} FP3 folds; {eligible} folds passed all audited gates."
        )
    if gate_counts:
        reason, count = max(gate_counts.items(), key=lambda item: item[1])
        findings.append(f"The most common audited blocking reason is `{reason}` ({count} rows).")
    if not sensitivity_summary.empty:
        best = sensitivity_summary.sort_values("fp3_retrospective_mae", na_position="last").iloc[0]
        findings.append(
            "Gate sensitivity results are retrospective simulations only; the best tested "
            f"configuration reached FP3 MAE {float(best['fp3_retrospective_mae']):.3f} sec."
        )
    if not findings:
        findings.append("Candidate audit could not evaluate the pathway from available artifacts.")
    return findings


def _read_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.is_file() else None


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


def _duplicate_key_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(frame.duplicated(list(KEY_COLUMNS), keep=False).sum())


def _key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=list(KEY_COLUMNS))
    return frame.loc[:, list(KEY_COLUMNS)].drop_duplicates()


def _unmatched_count(left: pd.DataFrame, right: pd.DataFrame) -> int:
    if left.empty:
        return 0
    if right.empty:
        return int(len(left))
    merged = left.merge(right, on=list(KEY_COLUMNS), how="left", indicator=True)
    return int(merged["_merge"].eq("left_only").sum())


def _event_list(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    label = frame["season"].astype(str) + "/" + frame["event_slug"].astype(str)
    return "; ".join(label.drop_duplicates().tolist())


def _current_event_in_history(fold: dict[str, object], prior: pd.DataFrame) -> bool:
    if prior.empty:
        return False
    same_season = prior["season"].astype(str).eq(str(fold.get("season")))
    same_event = prior["event_slug"].astype(str).eq(str(fold.get("event_slug")))
    return bool((same_season & same_event).any())


def _mean_numeric(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else None


def _plot_gate_failures(plt: Any, eligibility: pd.DataFrame, path: Path) -> bool:
    if eligibility.empty:
        return False
    counts = eligibility["selection_reason"].astype(str).value_counts()
    if counts.empty:
        return False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts.sort_values().plot(kind="barh", ax=ax, color="#5B8DEF")
    ax.set_title("Season-aware candidate gate failure reasons")
    ax.set_xlabel("FP3 folds")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_history_growth(plt: Any, history: pd.DataFrame, path: Path) -> bool:
    if history.empty:
        return False
    frame = history.sort_values("fold_id")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frame["fold_id"], frame["weighted_candidate_prior_predictions"], marker="o")
    ax.set_title("Weighted FP3 candidate prior prediction history")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Prior candidate predictions")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_metric_delta(plt: Any, history: pd.DataFrame, path: Path) -> bool:
    if history.empty or history["improvement_delta_sec"].dropna().empty:
        return False
    frame = history.sort_values("fold_id")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0.0, color="#555555", linewidth=1)
    ax.plot(frame["fold_id"], frame["improvement_delta_sec"], marker="o", color="#2E8B57")
    ax.set_title("Prior-only candidate metric delta over folds")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Candidate MAE minus default MAE (sec)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_sensitivity(
    plt: Any,
    summary: pd.DataFrame,
    path: Path,
    *,
    value_column: str,
    ylabel: str,
    title: str,
) -> bool:
    if summary.empty or value_column not in summary or summary[value_column].dropna().empty:
        return False
    frame = summary.sort_values("config_id")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frame["config_id"], frame[value_column], marker=".", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Sensitivity configuration")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _prediction_columns() -> list[str]:
    return [
        *KEY_COLUMNS,
        "event",
        "team",
        "quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec",
        "quali_position",
        "predicted_quali_position",
        "candidate_family",
        "candidate_model_name",
        "candidate_feature_group",
        "training_policy",
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


def _composition_columns() -> list[str]:
    return [
        "training_policy",
        "fold_id",
        "test_event",
        "test_season",
        "same_season_training_events",
        "prior_season_training_events",
        "effective_sample_size",
        "same_season_weight_share",
        "prior_season_weight_share",
    ]


def _alignment_columns() -> list[str]:
    return [
        "fold_id",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "expected_fold",
        "weighted_candidate_fold_found",
        "default_candidate_fold_found",
        "weighted_candidate_rows",
        "default_candidate_rows",
        "weighted_candidate_duplicate_rows",
        "default_candidate_duplicate_rows",
        "unmatched_weighted_candidate_rows",
        "unmatched_default_candidate_rows",
        "historical_source_events",
        "current_event_in_history",
        "schema_mismatch",
    ]


def _history_columns() -> list[str]:
    return [
        "fold_id",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "current_season_prior_event_count",
        "weighted_candidate_artifact_available",
        "weighted_candidate_prediction_rows",
        "default_candidate_prediction_rows",
        "weighted_candidate_prior_folds",
        "weighted_candidate_prior_predictions",
        "default_prior_folds",
        "default_prior_predictions",
        "weighted_candidate_metric_value",
        "default_metric_value",
        "improvement_delta_sec",
        "required_min_prior_candidate_folds",
        "required_min_prior_candidate_predictions",
        "required_min_current_season_prior_events",
        "required_improvement_margin_sec",
        "historical_source_events",
        "current_event_in_history",
        "effective_sample_size",
        "same_season_weight_share",
        "prior_season_weight_share",
    ]


def _eligibility_columns() -> list[str]:
    return [
        *_history_columns(),
        "cold_start_gate_passed",
        "candidate_history_gate_passed",
        "candidate_prediction_gate_passed",
        "margin_gate_passed",
        "candidate_eligible",
        "candidate_selected",
        "selection_reason",
    ]


def _sensitivity_detail_columns() -> list[str]:
    return [
        "config_id",
        "min_current_season_prior_events",
        "min_prior_candidate_folds",
        "min_prior_candidate_predictions",
        "improvement_margin_sec",
        "fold_id",
        "season",
        "event",
        "event_slug",
        "candidate_eligible",
        "blocking_reason",
        "rows",
        "static_mae_gap_sec",
        "simulated_policy_mae_gap_sec",
        "delta_vs_static_sec",
        "retrospective_simulation",
    ]


def _sensitivity_summary_columns() -> list[str]:
    return [
        "config_id",
        "min_current_season_prior_events",
        "min_prior_candidate_folds",
        "min_prior_candidate_predictions",
        "improvement_margin_sec",
        "candidate_selection_rate",
        "candidate_eligible_folds",
        "total_folds",
        "cold_start_blocked_folds",
        "history_blocked_folds",
        "margin_blocked_folds",
        "fp3_retrospective_mae",
        "delta_vs_static",
        "share_fp3_events_improved",
        "worst_event_delta",
        "mean_event_level_delta",
        "median_event_level_delta",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "bootstrap_seed",
        "retrospective_simulation",
        "matches_live_configuration",
    ]


def _relative_report_path(path: Path) -> str:
    parts = path.parts
    if "reports" in parts:
        index = parts.index("reports")
        return "/".join(parts[index:])
    return path.as_posix()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
