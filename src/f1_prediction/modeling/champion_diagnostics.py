"""Targeted champion-policy and conformal interval diagnostics."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_prediction.config import ChampionDiagnosticsConfig, DataConfig
from f1_prediction.utils.paths import ensure_directory

LOGGER = logging.getLogger(__name__)

CHECKPOINT_ORDER: tuple[str, ...] = ("after_fp1", "after_fp2", "after_fp3")
COMPARISON_MODES: tuple[str, ...] = (
    "nested",
    "stabilized_nested",
    "stabilized_nested_guarded",
    "season_aware_nested_guarded",
)
PREDICTION_FILES: dict[str, str] = {
    "static": "champion_static_predictions.parquet",
    "nested": "champion_nested_predictions.parquet",
    "stabilized_nested": "champion_stabilized_nested_predictions.parquet",
    "stabilized_nested_guarded": "champion_stabilized_nested_guarded_predictions.parquet",
    "season_aware_nested_guarded": ("champion_season_aware_nested_guarded_predictions.parquet"),
}
SELECTION_FILES: dict[str, str] = {
    "static": "champion_static_selection.parquet",
    "nested": "champion_nested_selection.parquet",
    "stabilized_nested": "champion_stabilized_nested_selection.parquet",
    "stabilized_nested_guarded": "champion_stabilized_nested_guarded_selection.parquet",
    "season_aware_nested_guarded": ("champion_season_aware_nested_guarded_selection.parquet"),
}
OPTIONAL_MODES: frozenset[str] = frozenset(
    {"stabilized_nested_guarded", "season_aware_nested_guarded"}
)
JOIN_COLUMNS: tuple[str, ...] = ("fold_id", "season", "event_slug", "checkpoint", "driver")
BOOTSTRAP_SEED = 20260422
BOOTSTRAP_ITERATIONS = 2000


@dataclass(frozen=True)
class ChampionDiagnosticsSummary:
    """Paths and issue counts produced by champion diagnostics generation."""

    status: str
    summary_path: Path
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    missing_inputs: tuple[str, ...]
    generation_issues: tuple[str, ...]


def create_champion_diagnostics_report(
    config: DataConfig,
    diagnostics_config: ChampionDiagnosticsConfig,
) -> ChampionDiagnosticsSummary:
    """Generate champion-policy and uncertainty diagnostics from saved artifacts."""
    metrics_dir = config.metrics_output_dir
    reports_dir = metrics_dir.parent
    figures_dir = reports_dir / "figures"
    ensure_directory(metrics_dir)
    ensure_directory(figures_dir)

    artifacts = _load_diagnostic_artifacts(metrics_dir)
    predictions = artifacts["predictions"]
    selections = artifacts["selection"]
    missing_inputs = list(artifacts["missing_inputs"])

    harmful_switches = build_harmful_switch_table(
        predictions,
        tolerance_sec=diagnostics_config.harmful_switch_tolerance_sec,
    )
    switch_summaries = build_harmful_switch_summaries(harmful_switches)
    fp3_failures = build_fp3_policy_failure_analysis(
        predictions.get("static"),
        predictions.get("stabilized_nested"),
        selections.get("stabilized_nested"),
        tolerance_sec=diagnostics_config.harmful_switch_tolerance_sec,
    )
    conformal_cases = build_conformal_miss_cases(_interval_diagnostic_predictions(predictions))
    conformal_summaries = build_conformal_miss_summaries(conformal_cases)
    coverage_by_regime = build_conformal_coverage_by_error_regime(conformal_cases)
    season_aware_tables = build_season_aware_champion_diagnostics(
        static_predictions=predictions.get("static"),
        guarded_predictions=predictions.get("stabilized_nested_guarded"),
        season_aware_predictions=predictions.get("season_aware_nested_guarded"),
        season_aware_selection=selections.get("season_aware_nested_guarded"),
        tolerance_sec=diagnostics_config.harmful_switch_tolerance_sec,
    )

    table_frames = {
        "champion_harmful_switches.csv": harmful_switches,
        "champion_switch_summary_by_checkpoint.csv": switch_summaries["checkpoint"],
        "champion_switch_summary_by_event.csv": switch_summaries["event"],
        "champion_switch_summary_by_method.csv": switch_summaries["method"],
        "fp3_policy_failure_analysis.csv": fp3_failures,
        "conformal_miss_summary_by_checkpoint.csv": conformal_summaries["checkpoint"],
        "conformal_miss_summary_by_event.csv": conformal_summaries["event"],
        "conformal_miss_summary_by_method.csv": conformal_summaries["method"],
        "conformal_miss_summary_by_driver.csv": conformal_summaries["driver"],
        "conformal_miss_cases.csv": conformal_cases,
        "conformal_coverage_by_error_regime.csv": coverage_by_regime,
        "season_aware_champion_event_comparison.csv": season_aware_tables["event"],
        "season_aware_champion_regime_comparison.csv": season_aware_tables["regime"],
    }
    table_paths: list[Path] = []
    for filename, frame in table_frames.items():
        path = metrics_dir / filename
        frame.to_csv(path, index=False)
        table_paths.append(path)

    figure_paths, figure_issues = generate_champion_diagnostics_figures(
        figures_dir=figures_dir,
        harmful_switches=harmful_switches,
        fp3_failures=fp3_failures,
        conformal_checkpoint_summary=conformal_summaries["checkpoint"],
        conformal_event_summary=conformal_summaries["event"],
        coverage_by_regime=coverage_by_regime,
        season_aware_event=season_aware_tables["event"],
        season_aware_regime=season_aware_tables["regime"],
    )
    season_aware_summary = build_season_aware_champion_summary_payload(
        event_comparison=season_aware_tables["event"],
        regime_comparison=season_aware_tables["regime"],
        selection=selections.get("season_aware_nested_guarded"),
        missing_inputs=missing_inputs,
    )
    season_aware_summary_path = metrics_dir / "season_aware_champion_summary.json"
    _write_json(season_aware_summary_path, season_aware_summary)
    table_paths.append(season_aware_summary_path)
    summary_payload = build_champion_diagnostics_summary_payload(
        inputs_available=artifacts["inputs_available"],
        missing_inputs=missing_inputs,
        harmful_switches=harmful_switches,
        switch_summaries=switch_summaries,
        fp3_failures=fp3_failures,
        conformal_cases=conformal_cases,
        conformal_summaries=conformal_summaries,
        coverage_by_regime=coverage_by_regime,
        generated_tables=table_paths,
        generated_figures=figure_paths,
        generation_issues=figure_issues,
        tolerance_sec=diagnostics_config.harmful_switch_tolerance_sec,
    )
    summary_path = metrics_dir / "champion_diagnostics_summary.json"
    _write_json(summary_path, summary_payload)

    return ChampionDiagnosticsSummary(
        status=str(summary_payload["status"]),
        summary_path=summary_path,
        table_paths=tuple(table_paths),
        figure_paths=tuple(figure_paths),
        missing_inputs=tuple(missing_inputs),
        generation_issues=tuple(figure_issues),
    )


def build_harmful_switch_table(
    predictions: dict[str, pd.DataFrame],
    tolerance_sec: float,
) -> pd.DataFrame:
    """Compare non-static champion rows against static rows on matching keys."""
    static = predictions.get("static")
    if static is None or static.empty:
        return pd.DataFrame(columns=_harmful_switch_columns())

    rows: list[pd.DataFrame] = []
    for mode in COMPARISON_MODES:
        comparison = predictions.get(mode)
        if comparison is None or comparison.empty:
            continue
        rows.append(
            build_harmful_switch_rows(
                static,
                comparison,
                selection_mode=mode,
                tolerance_sec=tolerance_sec,
            )
        )
    if not rows:
        return pd.DataFrame(columns=_harmful_switch_columns())
    return pd.concat(rows, ignore_index=True)[_harmful_switch_columns()]


def build_harmful_switch_rows(
    static_predictions: pd.DataFrame,
    comparison_predictions: pd.DataFrame,
    *,
    selection_mode: str,
    tolerance_sec: float,
) -> pd.DataFrame:
    """Return row-level static-vs-comparison error deltas."""
    missing_static = [
        column for column in _required_prediction_columns() if column not in static_predictions
    ]
    missing_comparison = [
        column for column in _required_prediction_columns() if column not in comparison_predictions
    ]
    if missing_static or missing_comparison:
        return pd.DataFrame(columns=_harmful_switch_columns())

    static = _normalize_prediction_frame(static_predictions)
    comparison = _normalize_prediction_frame(comparison_predictions)
    join_columns = [column for column in JOIN_COLUMNS if column in static and column in comparison]
    if len(join_columns) < 4:
        return pd.DataFrame(columns=_harmful_switch_columns())

    static_columns = [
        *join_columns,
        "event",
        "team",
        "quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec",
        "selected_family",
        "selected_model_name",
        "selected_feature_group",
    ]
    comparison_columns = [
        *join_columns,
        "predicted_quali_gap_to_pole_sec",
        "selected_family",
        "selected_model_name",
        "selected_feature_group",
    ]
    merged = static[static_columns].merge(
        comparison[comparison_columns],
        on=join_columns,
        how="inner",
        suffixes=("_static", "_comparison"),
    )
    if merged.empty:
        return pd.DataFrame(columns=_harmful_switch_columns())

    actual = pd.to_numeric(merged["quali_gap_to_pole_sec"], errors="coerce")
    static_predicted = pd.to_numeric(
        merged["predicted_quali_gap_to_pole_sec_static"], errors="coerce"
    )
    comparison_predicted = pd.to_numeric(
        merged["predicted_quali_gap_to_pole_sec_comparison"], errors="coerce"
    )
    output = pd.DataFrame(
        {
            "selection_mode": selection_mode,
            "season": merged["season"],
            "event": merged["event"],
            "fold_id": merged["fold_id"],
            "checkpoint": merged["checkpoint"],
            "driver": merged["driver"],
            "team": merged["team"],
            "static_selected_family": merged["selected_family_static"],
            "comparison_selected_family": merged["selected_family_comparison"],
            "static_selected_model_name": merged["selected_model_name_static"],
            "comparison_selected_model_name": merged["selected_model_name_comparison"],
            "static_selected_feature_group": merged["selected_feature_group_static"],
            "comparison_selected_feature_group": merged["selected_feature_group_comparison"],
            "actual_quali_gap_to_pole_sec": actual,
            "static_predicted_quali_gap_to_pole_sec": static_predicted,
            "comparison_predicted_quali_gap_to_pole_sec": comparison_predicted,
        }
    )
    output["static_abs_error_gap_sec"] = (static_predicted - actual).abs()
    output["comparison_abs_error_gap_sec"] = (comparison_predicted - actual).abs()
    output["error_delta_vs_static_sec"] = (
        output["comparison_abs_error_gap_sec"] - output["static_abs_error_gap_sec"]
    )
    output["harmful_switch"] = output["error_delta_vs_static_sec"] > tolerance_sec
    output["beneficial_switch"] = output["error_delta_vs_static_sec"] < -tolerance_sec
    return output[_harmful_switch_columns()]


def build_harmful_switch_summaries(harmful_switches: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build checkpoint, event, and selected-method switch summaries."""
    if harmful_switches.empty:
        return {
            "checkpoint": pd.DataFrame(columns=_switch_summary_columns("checkpoint")),
            "event": pd.DataFrame(columns=_switch_summary_columns("event")),
            "method": pd.DataFrame(columns=_switch_summary_columns("method")),
        }
    return {
        "checkpoint": _summarize_switches(
            harmful_switches,
            ["selection_mode", "checkpoint"],
            _switch_summary_columns("checkpoint"),
        ),
        "event": _summarize_switches(
            harmful_switches,
            ["selection_mode", "season", "event", "fold_id", "checkpoint"],
            _switch_summary_columns("event"),
        ),
        "method": _summarize_switches(
            harmful_switches,
            [
                "selection_mode",
                "checkpoint",
                "comparison_selected_family",
                "comparison_selected_model_name",
                "comparison_selected_feature_group",
            ],
            _switch_summary_columns("method"),
        ),
    }


def build_fp3_policy_failure_analysis(
    static_predictions: pd.DataFrame | None,
    stabilized_predictions: pd.DataFrame | None,
    stabilized_selection: pd.DataFrame | None,
    *,
    tolerance_sec: float = 0.05,
) -> pd.DataFrame:
    """Summarize FP3 folds where stabilized nested abandons the static RF policy."""
    if static_predictions is None or stabilized_predictions is None:
        return pd.DataFrame(columns=_fp3_failure_columns())
    rows = build_harmful_switch_rows(
        static_predictions,
        stabilized_predictions,
        selection_mode="stabilized_nested",
        tolerance_sec=tolerance_sec,
    )
    if rows.empty:
        return pd.DataFrame(columns=_fp3_failure_columns())
    fp3 = rows[rows["checkpoint"].eq("after_fp3")].copy()
    if fp3.empty:
        return pd.DataFrame(columns=_fp3_failure_columns())

    selection_lookup = _selection_lookup(stabilized_selection)
    output_rows: list[dict[str, object]] = []
    for keys, group in fp3.groupby(["season", "event", "fold_id"], dropna=False, sort=False):
        season, event, fold_id = keys
        static_method = _method_label(
            group["static_selected_family"],
            group["static_selected_model_name"],
            group["static_selected_feature_group"],
        )
        stabilized_method = _method_label(
            group["comparison_selected_family"],
            group["comparison_selected_model_name"],
            group["comparison_selected_feature_group"],
        )
        selection_values = selection_lookup.get((fold_id, "after_fp3"), {})
        fallback_rate = selection_values.get("fallback_rate")
        fallback_reason = selection_values.get("fallback_reason")
        output_rows.append(
            {
                "season": season,
                "event": event,
                "fold_id": fold_id,
                "static_method": static_method,
                "stabilized_method": stabilized_method,
                "static_mae_gap_sec": float(group["static_abs_error_gap_sec"].mean()),
                "stabilized_mae_gap_sec": float(group["comparison_abs_error_gap_sec"].mean()),
                "delta_vs_static_sec": float(group["error_delta_vs_static_sec"].mean()),
                "stabilized_fallback_rate": fallback_rate,
                "stabilized_fallback_reason": fallback_reason,
                "rows": int(len(group)),
                "drivers_affected": int(group.loc[group["harmful_switch"], "driver"].nunique()),
                "harmful_switches": int(group["harmful_switch"].sum()),
                "beneficial_switches": int(group["beneficial_switch"].sum()),
                "stabilized_abandoned_static_fp3_rf": _abandoned_static_fp3_rf(
                    group.iloc[0],
                ),
            }
        )
    return pd.DataFrame(output_rows, columns=_fp3_failure_columns())


def _interval_diagnostic_predictions(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """Choose the richest available interval artifact for conformal diagnostics."""
    guarded = predictions.get("stabilized_nested_guarded")
    if guarded is not None and not guarded.empty and "uncertainty_method" in guarded.columns:
        predicted_bucket = guarded[
            guarded["uncertainty_method"].eq("conformal_predicted_gap_bucket")
        ]
        if not predicted_bucket.empty:
            return guarded
        conformal = guarded[guarded["uncertainty_method"].eq("conformal")]
        if not conformal.empty:
            return guarded
    return predictions.get("stabilized_nested")


def build_conformal_miss_cases(predictions: pd.DataFrame | None) -> pd.DataFrame:
    """Create row-level conformal interval miss diagnostics."""
    if predictions is None or predictions.empty:
        return pd.DataFrame(columns=_conformal_case_columns())
    required = {
        "season",
        "event",
        "fold_id",
        "checkpoint",
        "driver",
        "quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec",
        "prediction_interval_low_sec",
        "prediction_interval_high_sec",
    }
    if not required <= set(predictions.columns):
        return pd.DataFrame(columns=_conformal_case_columns())

    frame = _normalize_prediction_frame(predictions)
    for column in (
        "selected_family",
        "selected_model_name",
        "selected_feature_group",
        "team",
        "residual_count",
        "residual_quantile_sec",
        "interval_contains_actual",
        "uncertainty_method",
        "predicted_gap_bucket",
        "uncertainty_calibration_level",
    ):
        if column not in frame:
            frame[column] = pd.NA

    available = frame[
        frame["prediction_interval_low_sec"].notna() & frame["prediction_interval_high_sec"].notna()
    ].copy()
    if available.empty:
        return pd.DataFrame(columns=_conformal_case_columns())

    actual = pd.to_numeric(available["quali_gap_to_pole_sec"], errors="coerce")
    predicted = pd.to_numeric(available["predicted_quali_gap_to_pole_sec"], errors="coerce")
    low = pd.to_numeric(available["prediction_interval_low_sec"], errors="coerce")
    high = pd.to_numeric(available["prediction_interval_high_sec"], errors="coerce")
    quantile = pd.to_numeric(available["residual_quantile_sec"], errors="coerce")
    width = high - low
    abs_error = (predicted - actual).abs()
    contained = _interval_contains_actual(available["interval_contains_actual"], actual, low, high)

    cases = pd.DataFrame(
        {
            "season": available["season"],
            "event": available["event"],
            "fold_id": available["fold_id"],
            "checkpoint": available["checkpoint"],
            "driver": available["driver"],
            "team": available["team"],
            "selected_family": available["selected_family"],
            "selected_model_name": available["selected_model_name"],
            "selected_feature_group": available["selected_feature_group"],
            "uncertainty_method": available["uncertainty_method"],
            "predicted_gap_bucket": available["predicted_gap_bucket"],
            "uncertainty_calibration_level": available["uncertainty_calibration_level"],
            "actual_quali_gap_to_pole_sec": actual,
            "predicted_quali_gap_to_pole_sec": predicted,
            "prediction_interval_low_sec": low,
            "prediction_interval_high_sec": high,
            "interval_width_sec": width,
            "residual_count": available["residual_count"],
            "residual_quantile_sec": quantile,
            "absolute_error_gap_sec": abs_error,
            "interval_miss": ~contained,
            "miss_side": _miss_side(actual, low, high, contained),
            "normalized_interval_error": _safe_divide(abs_error, quantile),
        }
    )
    return cases[_conformal_case_columns()]


def build_conformal_miss_summaries(cases: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize conformal misses by checkpoint, event, method, and driver."""
    if cases.empty:
        return {
            "checkpoint": pd.DataFrame(columns=_conformal_summary_columns("checkpoint")),
            "event": pd.DataFrame(columns=_conformal_summary_columns("event")),
            "method": pd.DataFrame(columns=_conformal_summary_columns("method")),
            "driver": pd.DataFrame(columns=_conformal_summary_columns("driver")),
        }
    return {
        "checkpoint": _summarize_conformal(
            cases,
            ["checkpoint"],
            _conformal_summary_columns("checkpoint"),
        ),
        "event": _summarize_conformal(
            cases,
            ["season", "event", "fold_id", "checkpoint"],
            _conformal_summary_columns("event"),
        ),
        "method": _summarize_conformal(
            cases,
            ["checkpoint", "selected_family", "selected_model_name", "selected_feature_group"],
            _conformal_summary_columns("method"),
        ),
        "driver": _summarize_conformal(
            cases,
            ["driver", "team", "checkpoint"],
            _conformal_summary_columns("driver"),
        ),
    }


def build_conformal_coverage_by_error_regime(cases: pd.DataFrame) -> pd.DataFrame:
    """Compute interval coverage in actual-gap and predicted-gap buckets."""
    columns = [
        "bucket_type",
        "checkpoint",
        "gap_bucket",
        "rows_with_interval",
        "coverage",
        "mean_interval_width_sec",
        "mean_abs_error_gap_sec",
        "miss_count",
    ]
    if cases.empty:
        return pd.DataFrame(columns=columns)
    frame = cases.copy()
    frame["actual_gap_bucket"] = frame["actual_quali_gap_to_pole_sec"].apply(_gap_bucket)
    frame["predicted_gap_bucket"] = frame["predicted_quali_gap_to_pole_sec"].apply(_gap_bucket)
    rows: list[dict[str, object]] = []
    for bucket_type, column in (
        ("actual_gap_bucket", "actual_gap_bucket"),
        ("predicted_gap_bucket", "predicted_gap_bucket"),
    ):
        for keys, group in frame.groupby(["checkpoint", column], dropna=False, sort=False):
            checkpoint, bucket = keys
            rows.append(
                {
                    "bucket_type": bucket_type,
                    "checkpoint": checkpoint,
                    "gap_bucket": bucket,
                    "rows_with_interval": int(len(group)),
                    "coverage": _coverage(group),
                    "mean_interval_width_sec": _mean_or_none(group["interval_width_sec"]),
                    "mean_abs_error_gap_sec": _mean_or_none(group["absolute_error_gap_sec"]),
                    "miss_count": int(group["interval_miss"].sum()),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_season_aware_champion_diagnostics(
    *,
    static_predictions: pd.DataFrame | None,
    guarded_predictions: pd.DataFrame | None,
    season_aware_predictions: pd.DataFrame | None,
    season_aware_selection: pd.DataFrame | None,
    tolerance_sec: float,
) -> dict[str, pd.DataFrame]:
    """Build FP3 event/regime comparisons for the opt-in season-aware champion."""
    event_columns = _season_aware_event_columns()
    regime_columns = _season_aware_regime_columns()
    if static_predictions is None or season_aware_predictions is None:
        return {
            "event": pd.DataFrame(columns=event_columns),
            "regime": pd.DataFrame(columns=regime_columns),
        }
    rows = build_harmful_switch_rows(
        static_predictions,
        season_aware_predictions,
        selection_mode="season_aware_nested_guarded",
        tolerance_sec=tolerance_sec,
    )
    rows = rows[rows["checkpoint"].eq("after_fp3")].copy()
    if rows.empty:
        return {
            "event": pd.DataFrame(columns=event_columns),
            "regime": pd.DataFrame(columns=regime_columns),
        }
    guarded_rows = (
        build_harmful_switch_rows(
            static_predictions,
            guarded_predictions,
            selection_mode="stabilized_nested_guarded",
            tolerance_sec=tolerance_sec,
        )
        if guarded_predictions is not None
        else pd.DataFrame()
    )
    guarded_lookup = _event_delta_lookup(guarded_rows)
    selection_lookup = _season_aware_selection_lookup(season_aware_selection)
    event_rows: list[dict[str, object]] = []
    for keys, group in rows.groupby(["season", "event", "fold_id"], dropna=False, sort=False):
        season, event, fold_id = keys
        selection_values = selection_lookup.get((fold_id, "after_fp3"), {})
        prior_count = int(selection_values.get("current_season_prior_event_count") or 0)
        event_rows.append(
            {
                "selection_mode": "season_aware_nested_guarded",
                "season": season,
                "event": event,
                "fold_id": fold_id,
                "checkpoint": "after_fp3",
                "current_season_prior_event_count": prior_count,
                "current_season_evidence_regime": _current_season_evidence_regime(prior_count),
                "rows": int(len(group)),
                "static_mae_gap_sec": float(group["static_abs_error_gap_sec"].mean()),
                "season_aware_mae_gap_sec": float(group["comparison_abs_error_gap_sec"].mean()),
                "delta_vs_static_sec": float(group["error_delta_vs_static_sec"].mean()),
                "delta_vs_guarded_sec": _delta(
                    float(group["comparison_abs_error_gap_sec"].mean()),
                    guarded_lookup.get((fold_id, "after_fp3")),
                ),
                "weighted_candidate_selected": bool(
                    selection_values.get("season_aware_selected", False)
                ),
                "season_aware_selection_reason": selection_values.get(
                    "season_aware_selection_reason"
                ),
                "guardrail_applied": bool(selection_values.get("guardrail_applied", False)),
                "harmful_switches_vs_static": int(group["harmful_switch"].sum()),
                "beneficial_switches_vs_static": int(group["beneficial_switch"].sum()),
            }
        )
    event = pd.DataFrame(event_rows, columns=event_columns)
    regime = _build_season_aware_regime_summary(event, regime_columns)
    return {"event": event, "regime": regime}


def build_season_aware_champion_summary_payload(
    *,
    event_comparison: pd.DataFrame,
    regime_comparison: pd.DataFrame,
    selection: pd.DataFrame | None,
    missing_inputs: list[str],
) -> dict[str, Any]:
    """Build the optional JSON summary for season-aware champion diagnostics."""
    required = {
        "champion_static_predictions.parquet",
        "champion_season_aware_nested_guarded_predictions.parquet",
    }
    missing_required = sorted(required.intersection(missing_inputs))
    bootstrap = _paired_bootstrap_ci(
        event_comparison["delta_vs_static_sec"] if not event_comparison.empty else []
    )
    fp3_summary = _season_aware_fp3_summary(event_comparison)
    selection_rate = _season_aware_selection_rate(selection)
    recommendation = _season_aware_champion_recommendation(fp3_summary, bootstrap)
    return {
        "status": "missing_inputs" if missing_required else "complete",
        "missing_inputs": missing_required,
        "fp3_summary": fp3_summary,
        "regime_summary": _records(regime_comparison),
        "weighted_candidate_selection_rate": selection_rate,
        "guardrail_application_rate": _guardrail_application_rate(selection),
        "bootstrap_ci": bootstrap,
        "promotion_recommendation": recommendation,
        "main_findings": _season_aware_champion_findings(
            fp3_summary,
            bootstrap,
            selection_rate,
        ),
        "generated_at": _utc_now(),
    }


def build_champion_diagnostics_summary_payload(
    *,
    inputs_available: dict[str, bool],
    missing_inputs: list[str],
    harmful_switches: pd.DataFrame,
    switch_summaries: dict[str, pd.DataFrame],
    fp3_failures: pd.DataFrame,
    conformal_cases: pd.DataFrame,
    conformal_summaries: dict[str, pd.DataFrame],
    coverage_by_regime: pd.DataFrame,
    generated_tables: list[Path],
    generated_figures: list[Path],
    generation_issues: list[str],
    tolerance_sec: float,
) -> dict[str, Any]:
    """Build the machine-readable champion diagnostics summary."""
    harmful_summary = _harmful_switch_summary(harmful_switches, switch_summaries)
    fp3_summary = _fp3_failure_summary(fp3_failures)
    conformal_summary = _conformal_undercoverage_summary(conformal_summaries)
    regime_summary = _regime_summary(coverage_by_regime)
    main_findings = _main_findings(
        harmful_summary=harmful_summary,
        fp3_summary=fp3_summary,
        conformal_summary=conformal_summary,
        regime_summary=regime_summary,
    )
    return {
        "status": _report_status(inputs_available, missing_inputs),
        "inputs_available": inputs_available,
        "missing_inputs": sorted(missing_inputs),
        "harmful_switch_tolerance_sec": tolerance_sec,
        "harmful_switch_summary": harmful_summary,
        "fp3_policy_failure_summary": fp3_summary,
        "conformal_undercoverage_summary": conformal_summary,
        "coverage_by_regime_summary": regime_summary,
        "main_findings": main_findings,
        "recommended_actions": _recommended_actions(main_findings, missing_inputs),
        "generated_tables": [str(path) for path in generated_tables],
        "generated_figures": [str(path) for path in generated_figures],
        "generation_issues": generation_issues,
        "generated_at": _utc_now(),
    }


def generate_champion_diagnostics_figures(
    *,
    figures_dir: Path,
    harmful_switches: pd.DataFrame,
    fp3_failures: pd.DataFrame,
    conformal_checkpoint_summary: pd.DataFrame,
    conformal_event_summary: pd.DataFrame,
    coverage_by_regime: pd.DataFrame,
    season_aware_event: pd.DataFrame | None = None,
    season_aware_regime: pd.DataFrame | None = None,
) -> tuple[list[Path], list[str]]:
    """Create non-interactive matplotlib diagnostic figures."""
    ensure_directory(figures_dir)
    paths: list[Path] = []
    issues: list[str] = []
    plot_specs = (
        (
            "harmful_switch_delta_by_checkpoint.png",
            lambda plt: _plot_switch_delta_by_checkpoint(plt, harmful_switches),
        ),
        (
            "fp3_static_vs_stabilized_mae_by_event.png",
            lambda plt: _plot_fp3_static_vs_stabilized(plt, fp3_failures),
        ),
        (
            "conformal_coverage_by_checkpoint.png",
            lambda plt: _plot_conformal_coverage_by_checkpoint(
                plt,
                conformal_checkpoint_summary,
            ),
        ),
        (
            "conformal_miss_count_by_event.png",
            lambda plt: _plot_conformal_miss_count_by_event(plt, conformal_event_summary),
        ),
        (
            "conformal_coverage_by_actual_gap_bucket.png",
            lambda plt: _plot_conformal_coverage_by_actual_gap_bucket(plt, coverage_by_regime),
        ),
        (
            "season_aware_champion_fp3_mae_by_event.png",
            lambda plt: _plot_season_aware_fp3_mae_by_event(plt, season_aware_event),
        ),
        (
            "season_aware_champion_fp3_delta_vs_static.png",
            lambda plt: _plot_season_aware_delta_vs_static(plt, season_aware_event),
        ),
        (
            "season_aware_champion_selection_by_regime.png",
            lambda plt: _plot_season_aware_selection_by_regime(plt, season_aware_regime),
        ),
        (
            "season_aware_champion_current_season_history.png",
            lambda plt: _plot_season_aware_current_history(plt, season_aware_event),
        ),
    )
    try:
        plt = _load_matplotlib()
    except Exception as exc:  # pragma: no cover - depends on local matplotlib install
        return [], [f"matplotlib unavailable: {exc}"]
    for filename, plotter in plot_specs:
        path = figures_dir / filename
        try:
            created = plotter(plt)
            if created:
                plt.savefig(path, dpi=160, bbox_inches="tight")
                paths.append(path)
            else:
                issues.append(f"Skipped {filename}: required data unavailable")
        except Exception as exc:  # pragma: no cover - defensive report generation
            issues.append(f"Skipped {filename}: {exc}")
        finally:
            plt.close()
    return paths, issues


def _load_diagnostic_artifacts(metrics_dir: Path) -> dict[str, object]:
    predictions: dict[str, pd.DataFrame] = {}
    selection: dict[str, pd.DataFrame] = {}
    inputs_available: dict[str, bool] = {}
    missing_inputs: list[str] = []
    for mode, filename in PREDICTION_FILES.items():
        path = metrics_dir / filename
        inputs_available[filename] = path.is_file()
        if path.is_file():
            predictions[mode] = pd.read_parquet(path)
        elif mode not in OPTIONAL_MODES:
            missing_inputs.append(filename)
    for mode, filename in SELECTION_FILES.items():
        path = metrics_dir / filename
        inputs_available[filename] = path.is_file()
        if path.is_file():
            selection[mode] = pd.read_parquet(path)
        elif mode not in OPTIONAL_MODES:
            missing_inputs.append(filename)
    return {
        "predictions": predictions,
        "selection": selection,
        "inputs_available": inputs_available,
        "missing_inputs": missing_inputs,
    }


def _normalize_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "driver" not in normalized and "driver_key" in normalized:
        normalized["driver"] = normalized["driver_key"]
    if "team" not in normalized and "team_key" in normalized:
        normalized["team"] = normalized["team_key"]
    if "team" not in normalized:
        normalized["team"] = pd.NA
    if "event_slug" not in normalized:
        normalized["event_slug"] = normalized["event"].astype(str)
    for column in ("selected_family", "selected_model_name", "selected_feature_group"):
        if column not in normalized:
            normalized[column] = pd.NA
    return normalized


def _required_prediction_columns() -> set[str]:
    return {
        "fold_id",
        "season",
        "event",
        "checkpoint",
        "quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec",
    }


def _summarize_switches(
    frame: pd.DataFrame,
    group_columns: list[str],
    columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "rows": int(len(group)),
                "harmful_switches": int(group["harmful_switch"].sum()),
                "beneficial_switches": int(group["beneficial_switch"].sum()),
                "neutral_switches": int(
                    len(group) - group["harmful_switch"].sum() - group["beneficial_switch"].sum()
                ),
                "harmful_rate": float(group["harmful_switch"].mean()),
                "beneficial_rate": float(group["beneficial_switch"].mean()),
                "mean_delta_vs_static_sec": _mean_or_none(group["error_delta_vs_static_sec"]),
                "median_delta_vs_static_sec": _median_or_none(group["error_delta_vs_static_sec"]),
                "static_mae_gap_sec": _mean_or_none(group["static_abs_error_gap_sec"]),
                "comparison_mae_gap_sec": _mean_or_none(group["comparison_abs_error_gap_sec"]),
                "delta_mae_vs_static_sec": _mean_or_none(group["error_delta_vs_static_sec"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _selection_lookup(
    selection: pd.DataFrame | None,
) -> dict[tuple[object, object], dict[str, object]]:
    if selection is None or selection.empty:
        return {}
    frame = selection.copy()
    for column in ("fallback_used", "fallback_reason"):
        if column not in frame:
            frame[column] = pd.NA
    lookup: dict[tuple[object, object], dict[str, object]] = {}
    for keys, group in frame.groupby(["fold_id", "checkpoint"], dropna=False, sort=False):
        fallback = group["fallback_used"].fillna(False).astype(bool)
        lookup[keys] = {
            "fallback_rate": float(fallback.mean()) if len(group) else None,
            "fallback_reason": _most_common_non_null(group["fallback_reason"]),
        }
    return lookup


def _abandoned_static_fp3_rf(row: pd.Series) -> bool:
    static_is_fp3_rf = (
        row.get("static_selected_family") == "ablation"
        and row.get("static_selected_model_name") == "random_forest"
        and row.get("static_selected_feature_group") == "base_plus_relative"
    )
    stabilized_matches = (
        row.get("comparison_selected_family") == "ablation"
        and row.get("comparison_selected_model_name") == "random_forest"
        and row.get("comparison_selected_feature_group") == "base_plus_relative"
    )
    return bool(static_is_fp3_rf and not stabilized_matches)


def _method_label(
    family: pd.Series,
    model_name: pd.Series,
    feature_group: pd.Series,
) -> str | None:
    values = pd.DataFrame(
        {
            "family": family,
            "model_name": model_name,
            "feature_group": feature_group,
        }
    )
    if values.empty:
        return None
    modes = values.mode(dropna=False)
    if modes.empty:
        row = values.iloc[0]
    else:
        row = modes.iloc[0]
    parts = [
        _display_optional(row["family"]),
        _display_optional(row["model_name"]),
        _display_optional(row["feature_group"]),
    ]
    return "/".join(part for part in parts if part)


def _summarize_conformal(
    frame: pd.DataFrame,
    group_columns: list[str],
    columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "rows_with_interval": int(len(group)),
                "miss_count": int(group["interval_miss"].sum()),
                "coverage": _coverage(group),
                "mean_interval_width_sec": _mean_or_none(group["interval_width_sec"]),
                "median_interval_width_sec": _median_or_none(group["interval_width_sec"]),
                "mean_abs_error_gap_sec": _mean_or_none(group["absolute_error_gap_sec"]),
                "mean_residual_quantile_sec": _mean_or_none(group["residual_quantile_sec"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _interval_contains_actual(
    raw_contains: pd.Series,
    actual: pd.Series,
    low: pd.Series,
    high: pd.Series,
) -> pd.Series:
    contains = raw_contains.map(_bool_or_none)
    computed = (actual >= low) & (actual <= high)
    return contains.where(contains.notna(), computed).fillna(False).astype(bool)


def _miss_side(
    actual: pd.Series,
    low: pd.Series,
    high: pd.Series,
    contained: pd.Series,
) -> pd.Series:
    side = pd.Series(pd.NA, index=actual.index, dtype="object")
    side.loc[~contained & (actual < low)] = "below_interval"
    side.loc[~contained & (actual > high)] = "above_interval"
    return side


def _gap_bucket(value: object) -> str | None:
    numeric = _number_or_none(value)
    if numeric is None:
        return None
    if numeric <= 0.5:
        return "pole_contender"
    if numeric <= 1.5:
        return "close_midfield"
    if numeric <= 3.0:
        return "midfield"
    return "backmarker_or_outlier"


def _harmful_switch_summary(
    harmful_switches: pd.DataFrame,
    switch_summaries: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    if harmful_switches.empty:
        return {
            "rows_compared": 0,
            "harmful_switches": 0,
            "beneficial_switches": 0,
            "mean_delta_vs_static_sec": None,
            "by_checkpoint": [],
        }
    return {
        "rows_compared": int(len(harmful_switches)),
        "harmful_switches": int(harmful_switches["harmful_switch"].sum()),
        "beneficial_switches": int(harmful_switches["beneficial_switch"].sum()),
        "mean_delta_vs_static_sec": _mean_or_none(harmful_switches["error_delta_vs_static_sec"]),
        "by_checkpoint": _records(switch_summaries["checkpoint"]),
    }


def _fp3_failure_summary(fp3_failures: pd.DataFrame) -> dict[str, Any]:
    if fp3_failures.empty:
        return {
            "rows": 0,
            "abandoned_static_fp3_rf_folds": 0,
            "mean_delta_vs_static_sec": None,
            "abandoned_mean_delta_vs_static_sec": None,
            "worst_events": [],
        }
    abandoned = fp3_failures[fp3_failures["stabilized_abandoned_static_fp3_rf"].astype(bool)]
    worst = fp3_failures.sort_values("delta_vs_static_sec", ascending=False).head(5)
    return {
        "rows": int(len(fp3_failures)),
        "abandoned_static_fp3_rf_folds": int(len(abandoned)),
        "mean_delta_vs_static_sec": _mean_or_none(fp3_failures["delta_vs_static_sec"]),
        "abandoned_mean_delta_vs_static_sec": _mean_or_none(abandoned["delta_vs_static_sec"]),
        "worst_events": _records(worst),
    }


def _conformal_undercoverage_summary(
    conformal_summaries: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    checkpoint = conformal_summaries["checkpoint"]
    if checkpoint.empty:
        return {"rows_with_interval": 0, "by_checkpoint": [], "fp3_coverage": None}
    fp3 = checkpoint[checkpoint["checkpoint"].eq("after_fp3")]
    return {
        "rows_with_interval": int(checkpoint["rows_with_interval"].sum()),
        "by_checkpoint": _records(checkpoint),
        "fp3_coverage": _number_or_none(fp3["coverage"].iloc[0]) if not fp3.empty else None,
    }


def _regime_summary(coverage_by_regime: pd.DataFrame) -> dict[str, Any]:
    if coverage_by_regime.empty:
        return {"rows": 0, "worst_actual_gap_bucket": None, "by_bucket": []}
    actual = coverage_by_regime[
        coverage_by_regime["bucket_type"].eq("actual_gap_bucket")
        & coverage_by_regime["gap_bucket"].notna()
    ]
    worst = actual.sort_values("coverage", ascending=True).head(1)
    return {
        "rows": int(len(coverage_by_regime)),
        "worst_actual_gap_bucket": _records(worst)[0] if not worst.empty else None,
        "by_bucket": _records(coverage_by_regime),
    }


def _main_findings(
    *,
    harmful_summary: dict[str, Any],
    fp3_summary: dict[str, Any],
    conformal_summary: dict[str, Any],
    regime_summary: dict[str, Any],
) -> list[str]:
    findings: list[str] = []
    if harmful_summary["rows_compared"]:
        harmful_count = int(harmful_summary["harmful_switches"])
        beneficial_count = int(harmful_summary["beneficial_switches"])
        findings.append(
            f"Harmful switches: {harmful_count} harmful vs {beneficial_count} beneficial "
            f"across {harmful_summary['rows_compared']} comparable rows."
        )
    if fp3_summary["abandoned_static_fp3_rf_folds"]:
        delta = fp3_summary.get("abandoned_mean_delta_vs_static_sec")
        if delta is not None and delta > 0:
            findings.append(
                "Stabilized nested underperforms static FP3 when it abandons the static "
                f"RF policy, with mean delta {delta:.3f} sec on abandoned folds."
            )
        else:
            findings.append(
                "Stabilized nested abandoned the static FP3 RF policy in "
                f"{fp3_summary['abandoned_static_fp3_rf_folds']} folds."
            )
    fp3_coverage = conformal_summary.get("fp3_coverage")
    if fp3_coverage is not None and fp3_coverage < 0.9:
        findings.append(f"FP3 conformal coverage is below nominal at {fp3_coverage:.1%}.")
    worst_bucket = regime_summary.get("worst_actual_gap_bucket")
    if worst_bucket and worst_bucket.get("coverage") is not None:
        coverage = worst_bucket["coverage"]
        if coverage < 0.9:
            findings.append(
                "Conformal intervals under-cover "
                f"{worst_bucket['gap_bucket']} actual-gap predictions at {coverage:.1%}."
            )
    if not findings:
        findings.append("Available artifacts did not show a concentrated diagnostic failure mode.")
    return findings


def _recommended_actions(findings: list[str], missing_inputs: list[str]) -> list[str]:
    actions: list[str] = []
    if missing_inputs:
        actions.append("Regenerate missing champion artifacts before interpreting all diagnostics.")
    if any("abandons the static RF policy" in finding for finding in findings):
        actions.append(
            "Inspect stabilized nested selection thresholds for FP3 RF/base_plus_relative."
        )
    if any("coverage is below nominal" in finding for finding in findings):
        actions.append(
            "Audit conformal residual pools by checkpoint, method, event, and gap regime."
        )
    if not actions:
        actions.append(
            "Review the generated diagnostic tables before changing champion policy logic."
        )
    return actions


def _report_status(inputs_available: dict[str, bool], missing_inputs: list[str]) -> str:
    if not any(inputs_available.values()):
        return "no_inputs"
    if missing_inputs:
        return "partial"
    return "complete"


def _plot_switch_delta_by_checkpoint(plt: Any, harmful_switches: pd.DataFrame) -> bool:
    if harmful_switches.empty:
        return False
    summary = (
        harmful_switches.groupby(["selection_mode", "checkpoint"], sort=False)[
            "error_delta_vs_static_sec"
        ]
        .mean()
        .reset_index()
    )
    _bar_by_checkpoint(
        plt,
        summary,
        value_column="error_delta_vs_static_sec",
        title="Mean Error Delta vs Static Champion",
        ylabel="Delta MAE (sec)",
    )
    return True


def _plot_fp3_static_vs_stabilized(plt: Any, fp3_failures: pd.DataFrame) -> bool:
    if fp3_failures.empty:
        return False
    frame = fp3_failures.sort_values(["season", "event", "fold_id"]).copy()
    labels = [f"{row.season} {row.event}" for row in frame.itertuples()]
    x_values = list(range(len(frame)))
    width = 0.38
    fig_width = max(8, len(frame) * 0.45)
    plt.figure(figsize=(fig_width, 4.5))
    plt.bar(
        [value - width / 2 for value in x_values],
        frame["static_mae_gap_sec"],
        width=width,
        label="static",
        color="#4c78a8",
    )
    plt.bar(
        [value + width / 2 for value in x_values],
        frame["stabilized_mae_gap_sec"],
        width=width,
        label="stabilized_nested",
        color="#f58518",
    )
    plt.xticks(x_values, labels, rotation=60, ha="right")
    plt.ylabel("MAE gap (sec)")
    plt.title("FP3 Static vs Stabilized Nested MAE by Event")
    plt.legend()
    _finish_plot(plt)
    return True


def _plot_conformal_coverage_by_checkpoint(plt: Any, summary: pd.DataFrame) -> bool:
    if summary.empty:
        return False
    ordered = _order_checkpoints(summary, "checkpoint")
    plt.figure(figsize=(7, 4))
    plt.bar(ordered["checkpoint"], ordered["coverage"], color="#54a24b")
    plt.axhline(0.9, color="#d62728", linestyle="--", linewidth=1, label="90% nominal")
    plt.ylim(0, 1.05)
    plt.ylabel("Coverage")
    plt.title("Conformal Coverage by Checkpoint")
    plt.legend()
    _finish_plot(plt)
    return True


def _plot_conformal_miss_count_by_event(plt: Any, summary: pd.DataFrame) -> bool:
    if summary.empty:
        return False
    frame = summary.sort_values("miss_count", ascending=False).head(12).copy()
    if frame["miss_count"].sum() == 0:
        return False
    labels = [f"{row.season} {row.event} {row.checkpoint}" for row in frame.itertuples()]
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, frame["miss_count"], color="#e45756")
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Miss count")
    plt.title("Conformal Miss Count by Event")
    _finish_plot(plt)
    return True


def _plot_conformal_coverage_by_actual_gap_bucket(plt: Any, regime: pd.DataFrame) -> bool:
    if regime.empty:
        return False
    frame = regime[regime["bucket_type"].eq("actual_gap_bucket")].copy()
    if frame.empty:
        return False
    pivot = frame.pivot_table(
        index="gap_bucket",
        columns="checkpoint",
        values="coverage",
        aggfunc="mean",
    )
    if pivot.empty:
        return False
    pivot = pivot.reindex(
        ["pole_contender", "close_midfield", "midfield", "backmarker_or_outlier"]
    ).dropna(how="all")
    checkpoints = [checkpoint for checkpoint in CHECKPOINT_ORDER if checkpoint in pivot.columns]
    x_values = list(range(len(pivot.index)))
    width = 0.75 / max(len(checkpoints), 1)
    plt.figure(figsize=(8, 4.5))
    for index, checkpoint in enumerate(checkpoints):
        offsets = [value - 0.375 + width / 2 + index * width for value in x_values]
        plt.bar(offsets, pivot[checkpoint], width=width, label=checkpoint)
    plt.axhline(0.9, color="#d62728", linestyle="--", linewidth=1, label="90% nominal")
    plt.xticks(x_values, pivot.index, rotation=30, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Coverage")
    plt.title("Conformal Coverage by Actual Gap Bucket")
    plt.legend()
    _finish_plot(plt)
    return True


def _plot_season_aware_fp3_mae_by_event(
    plt: Any,
    event: pd.DataFrame | None,
) -> bool:
    if event is None or event.empty:
        return False
    frame = event.sort_values(["season", "fold_id"]).copy()
    labels = [f"{row.season} {row.event}" for row in frame.itertuples()]
    x_values = list(range(len(frame)))
    width = 0.38
    plt.figure(figsize=(max(8, len(frame) * 0.45), 4.5))
    plt.bar(
        [value - width / 2 for value in x_values],
        frame["static_mae_gap_sec"],
        width=width,
        label="static",
        color="#4c78a8",
    )
    plt.bar(
        [value + width / 2 for value in x_values],
        frame["season_aware_mae_gap_sec"],
        width=width,
        label="season_aware_nested_guarded",
        color="#54a24b",
    )
    plt.xticks(x_values, labels, rotation=60, ha="right")
    plt.ylabel("FP3 MAE gap (sec)")
    plt.title("Season-Aware Champion FP3 MAE by Event")
    plt.legend()
    _finish_plot(plt)
    return True


def _plot_season_aware_delta_vs_static(plt: Any, event: pd.DataFrame | None) -> bool:
    if event is None or event.empty:
        return False
    frame = event.sort_values("delta_vs_static_sec").copy()
    labels = [f"{row.season} {row.event}" for row in frame.itertuples()]
    colors = ["#54a24b" if value <= 0 else "#e45756" for value in frame["delta_vs_static_sec"]]
    plt.figure(figsize=(max(8, len(frame) * 0.45), 4.5))
    plt.bar(labels, frame["delta_vs_static_sec"], color=colors)
    plt.axhline(0, color="#333333", linewidth=0.8)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Candidate minus static MAE (sec)")
    plt.title("Season-Aware Champion FP3 Delta vs Static")
    _finish_plot(plt)
    return True


def _plot_season_aware_selection_by_regime(
    plt: Any,
    regime: pd.DataFrame | None,
) -> bool:
    if regime is None or regime.empty:
        return False
    frame = regime.copy()
    order = ["cold_start", "early_season", "established_season"]
    frame["_order"] = frame["current_season_evidence_regime"].map(
        {value: index for index, value in enumerate(order)}
    )
    frame = frame.sort_values(["_order", "current_season_evidence_regime"])
    plt.figure(figsize=(7, 4))
    plt.bar(
        frame["current_season_evidence_regime"],
        frame["weighted_candidate_selection_rate"],
        color="#72b7b2",
    )
    plt.ylim(0, 1.05)
    plt.ylabel("Selection rate")
    plt.title("Season-Aware Candidate Selection by Regime")
    _finish_plot(plt)
    return True


def _plot_season_aware_current_history(plt: Any, event: pd.DataFrame | None) -> bool:
    if event is None or event.empty:
        return False
    frame = event.sort_values(["season", "fold_id"]).copy()
    labels = [f"{row.season} {row.event}" for row in frame.itertuples()]
    plt.figure(figsize=(max(8, len(frame) * 0.45), 4.5))
    plt.bar(labels, frame["current_season_prior_event_count"], color="#b279a2")
    plt.axhline(5, color="#333333", linestyle="--", linewidth=1, label="cold-start gate")
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Prior same-season events")
    plt.title("Current-Season History Available by FP3 Event")
    plt.legend()
    _finish_plot(plt)
    return True


def _bar_by_checkpoint(
    plt: Any,
    frame: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    ylabel: str,
) -> None:
    checkpoints = [
        checkpoint for checkpoint in CHECKPOINT_ORDER if checkpoint in set(frame["checkpoint"])
    ]
    modes = list(dict.fromkeys(frame["selection_mode"].dropna().astype(str).tolist()))
    x_values = list(range(len(checkpoints)))
    width = 0.75 / max(len(modes), 1)
    plt.figure(figsize=(7, 4))
    for index, mode in enumerate(modes):
        subset = frame[frame["selection_mode"].eq(mode)].set_index("checkpoint")
        heights = [subset[value_column].get(checkpoint, 0.0) for checkpoint in checkpoints]
        offsets = [value - 0.375 + width / 2 + index * width for value in x_values]
        plt.bar(offsets, heights, width=width, label=mode)
    plt.axhline(0, color="#333333", linewidth=0.8)
    plt.xticks(x_values, checkpoints)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    _finish_plot(plt)


def _load_matplotlib() -> Any:
    cache_root = Path(
        os.environ.get("MPLCONFIGDIR", Path(tempfile.gettempdir()) / "f1_prediction_matplotlib")
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("default")
    return plt


def _finish_plot(plt: Any) -> None:
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()


def _order_checkpoints(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["_checkpoint_order"] = ordered[column].map(
        {checkpoint: index for index, checkpoint in enumerate(CHECKPOINT_ORDER)}
    )
    return ordered.sort_values(["_checkpoint_order", column]).drop(columns="_checkpoint_order")


def _harmful_switch_columns() -> list[str]:
    return [
        "selection_mode",
        "season",
        "event",
        "fold_id",
        "checkpoint",
        "driver",
        "team",
        "static_selected_family",
        "comparison_selected_family",
        "static_selected_model_name",
        "comparison_selected_model_name",
        "static_selected_feature_group",
        "comparison_selected_feature_group",
        "actual_quali_gap_to_pole_sec",
        "static_predicted_quali_gap_to_pole_sec",
        "comparison_predicted_quali_gap_to_pole_sec",
        "static_abs_error_gap_sec",
        "comparison_abs_error_gap_sec",
        "error_delta_vs_static_sec",
        "harmful_switch",
        "beneficial_switch",
    ]


def _switch_summary_columns(summary_type: str) -> list[str]:
    prefixes = {
        "checkpoint": ["selection_mode", "checkpoint"],
        "event": ["selection_mode", "season", "event", "fold_id", "checkpoint"],
        "method": [
            "selection_mode",
            "checkpoint",
            "comparison_selected_family",
            "comparison_selected_model_name",
            "comparison_selected_feature_group",
        ],
    }[summary_type]
    return [
        *prefixes,
        "rows",
        "harmful_switches",
        "beneficial_switches",
        "neutral_switches",
        "harmful_rate",
        "beneficial_rate",
        "mean_delta_vs_static_sec",
        "median_delta_vs_static_sec",
        "static_mae_gap_sec",
        "comparison_mae_gap_sec",
        "delta_mae_vs_static_sec",
    ]


def _fp3_failure_columns() -> list[str]:
    return [
        "season",
        "event",
        "fold_id",
        "static_method",
        "stabilized_method",
        "static_mae_gap_sec",
        "stabilized_mae_gap_sec",
        "delta_vs_static_sec",
        "stabilized_fallback_rate",
        "stabilized_fallback_reason",
        "rows",
        "drivers_affected",
        "harmful_switches",
        "beneficial_switches",
        "stabilized_abandoned_static_fp3_rf",
    ]


def _conformal_case_columns() -> list[str]:
    return [
        "season",
        "event",
        "fold_id",
        "checkpoint",
        "driver",
        "team",
        "selected_family",
        "selected_model_name",
        "selected_feature_group",
        "uncertainty_method",
        "predicted_gap_bucket",
        "uncertainty_calibration_level",
        "actual_quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec",
        "prediction_interval_low_sec",
        "prediction_interval_high_sec",
        "interval_width_sec",
        "residual_count",
        "residual_quantile_sec",
        "absolute_error_gap_sec",
        "interval_miss",
        "miss_side",
        "normalized_interval_error",
    ]


def _conformal_summary_columns(summary_type: str) -> list[str]:
    prefixes = {
        "checkpoint": ["checkpoint"],
        "event": ["season", "event", "fold_id", "checkpoint"],
        "method": [
            "checkpoint",
            "selected_family",
            "selected_model_name",
            "selected_feature_group",
        ],
        "driver": ["driver", "team", "checkpoint"],
    }[summary_type]
    return [
        *prefixes,
        "rows_with_interval",
        "miss_count",
        "coverage",
        "mean_interval_width_sec",
        "median_interval_width_sec",
        "mean_abs_error_gap_sec",
        "mean_residual_quantile_sec",
    ]


def _season_aware_event_columns() -> list[str]:
    return [
        "selection_mode",
        "season",
        "event",
        "fold_id",
        "checkpoint",
        "current_season_prior_event_count",
        "current_season_evidence_regime",
        "rows",
        "static_mae_gap_sec",
        "season_aware_mae_gap_sec",
        "delta_vs_static_sec",
        "delta_vs_guarded_sec",
        "weighted_candidate_selected",
        "season_aware_selection_reason",
        "guardrail_applied",
        "harmful_switches_vs_static",
        "beneficial_switches_vs_static",
    ]


def _season_aware_regime_columns() -> list[str]:
    return [
        "current_season_evidence_regime",
        "events",
        "rows",
        "static_mae_gap_sec",
        "season_aware_mae_gap_sec",
        "mean_delta_vs_static_sec",
        "median_delta_vs_static_sec",
        "share_events_improved",
        "weighted_candidate_selection_rate",
        "guardrail_application_rate",
        "mean_current_season_prior_event_count",
    ]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.where(denominator > 0)
    return result.replace([math.inf, -math.inf], pd.NA)


def _season_aware_selection_lookup(
    selection: pd.DataFrame | None,
) -> dict[tuple[object, object], dict[str, object]]:
    if selection is None or selection.empty:
        return {}
    frame = selection.copy()
    for column in (
        "current_season_prior_event_count",
        "season_aware_selected",
        "season_aware_selection_reason",
        "guardrail_applied",
    ):
        if column not in frame:
            frame[column] = pd.NA
    lookup: dict[tuple[object, object], dict[str, object]] = {}
    for keys, group in frame.groupby(["fold_id", "checkpoint"], dropna=False, sort=False):
        row = group.iloc[0]
        lookup[keys] = {
            "current_season_prior_event_count": row.get("current_season_prior_event_count"),
            "season_aware_selected": _bool_or_none(row.get("season_aware_selected")),
            "season_aware_selection_reason": row.get("season_aware_selection_reason"),
            "guardrail_applied": _bool_or_none(row.get("guardrail_applied")),
        }
    return lookup


def _event_delta_lookup(rows: pd.DataFrame) -> dict[tuple[object, object], float]:
    if rows.empty:
        return {}
    fp3 = rows[rows["checkpoint"].eq("after_fp3")].copy()
    lookup: dict[tuple[object, object], float] = {}
    for keys, group in fp3.groupby(["fold_id", "checkpoint"], dropna=False, sort=False):
        lookup[keys] = float(group["comparison_abs_error_gap_sec"].mean())
    return lookup


def _build_season_aware_regime_summary(
    event: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    if event.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for regime, group in event.groupby(
        "current_season_evidence_regime",
        dropna=False,
        sort=False,
    ):
        rows.append(
            {
                "current_season_evidence_regime": regime,
                "events": int(len(group)),
                "rows": int(group["rows"].sum()),
                "static_mae_gap_sec": _mean_or_none(group["static_mae_gap_sec"]),
                "season_aware_mae_gap_sec": _mean_or_none(group["season_aware_mae_gap_sec"]),
                "mean_delta_vs_static_sec": _mean_or_none(group["delta_vs_static_sec"]),
                "median_delta_vs_static_sec": _median_or_none(group["delta_vs_static_sec"]),
                "share_events_improved": float(group["delta_vs_static_sec"].lt(0).mean()),
                "weighted_candidate_selection_rate": float(
                    group["weighted_candidate_selected"].fillna(False).astype(bool).mean()
                ),
                "guardrail_application_rate": float(
                    group["guardrail_applied"].fillna(False).astype(bool).mean()
                ),
                "mean_current_season_prior_event_count": _mean_or_none(
                    group["current_season_prior_event_count"]
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _current_season_evidence_regime(prior_event_count: int | float | object) -> str:
    count = int(prior_event_count) if pd.notna(prior_event_count) else 0
    if count < 5:
        return "cold_start"
    if count <= 8:
        return "early_season"
    return "established_season"


def _paired_bootstrap_ci(
    values: pd.Series | list[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype(float)
    if numeric.empty:
        return {"mean_delta": None, "ci_low": None, "ci_high": None, "events": 0, "seed": seed}
    data = numeric.to_numpy()
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(iterations, len(data)), replace=True).mean(axis=1)
    return {
        "mean_delta": float(data.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "events": int(len(data)),
        "seed": seed,
    }


def _season_aware_fp3_summary(event: pd.DataFrame) -> dict[str, object]:
    if event.empty:
        return {
            "events": 0,
            "rows": 0,
            "static_mae_gap_sec": None,
            "season_aware_mae_gap_sec": None,
            "delta_vs_static_sec": None,
            "delta_vs_guarded_sec": None,
            "share_events_improved": None,
        }
    return {
        "events": int(len(event)),
        "rows": int(event["rows"].sum()),
        "static_mae_gap_sec": _mean_or_none(event["static_mae_gap_sec"]),
        "season_aware_mae_gap_sec": _mean_or_none(event["season_aware_mae_gap_sec"]),
        "delta_vs_static_sec": _mean_or_none(event["delta_vs_static_sec"]),
        "delta_vs_guarded_sec": _mean_or_none(event["delta_vs_guarded_sec"]),
        "share_events_improved": float(event["delta_vs_static_sec"].lt(0).mean()),
    }


def _season_aware_selection_rate(selection: pd.DataFrame | None) -> float | None:
    if selection is None or selection.empty or "season_aware_selected" not in selection:
        return None
    fp3 = selection[selection["checkpoint"].eq("after_fp3")]
    if fp3.empty:
        return None
    return float(fp3["season_aware_selected"].fillna(False).astype(bool).mean())


def _guardrail_application_rate(selection: pd.DataFrame | None) -> float | None:
    if selection is None or selection.empty or "guardrail_applied" not in selection:
        return None
    fp3 = selection[selection["checkpoint"].eq("after_fp3")]
    if fp3.empty:
        return None
    return float(fp3["guardrail_applied"].fillna(False).astype(bool).mean())


def _season_aware_champion_recommendation(
    fp3_summary: dict[str, object],
    bootstrap: dict[str, float | int | None],
) -> str:
    delta = _number_or_none(fp3_summary.get("delta_vs_static_sec"))
    ci_high = _number_or_none(bootstrap.get("ci_high"))
    if delta is None:
        return "retain_static_policy"
    if delta < 0 and ci_high is not None and ci_high < 0:
        return "eligible_for_broader_validation"
    if delta < 0:
        return "season_aware_candidate_experimental"
    return "retain_static_policy"


def _season_aware_champion_findings(
    fp3_summary: dict[str, object],
    bootstrap: dict[str, float | int | None],
    selection_rate: float | None,
) -> list[str]:
    findings: list[str] = []
    delta = _number_or_none(fp3_summary.get("delta_vs_static_sec"))
    if delta is not None:
        findings.append(
            f"Season-aware nested guarded FP3 mean event delta vs static is {delta:.3f} sec."
        )
    if selection_rate is not None:
        findings.append(
            "Weighted FP3 candidate selection rate is "
            f"{selection_rate:.1%} across evaluated FP3 folds."
        )
    ci_low = _number_or_none(bootstrap.get("ci_low"))
    ci_high = _number_or_none(bootstrap.get("ci_high"))
    if ci_low is not None and ci_high is not None:
        findings.append(
            f"Paired event-level bootstrap CI for FP3 delta is [{ci_low:.3f}, {ci_high:.3f}] sec."
        )
    if not findings:
        findings.append("Season-aware champion artifacts are not available for comparison.")
    return findings


def _coverage(group: pd.DataFrame) -> float | None:
    if group.empty:
        return None
    return float(1.0 - group["interval_miss"].mean())


def _mean_or_none(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else None


def _delta(first: object, second: object) -> float | None:
    if first is None or second is None:
        return None
    return float(first) - float(second)


def _median_or_none(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else None


def _number_or_none(value: object) -> float | None:
    if value is None or value is pd.NA:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _bool_or_none(value: object) -> bool | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool | np.bool_):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        if math.isnan(float(value)):
            return None
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _display_optional(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _most_common_non_null(values: pd.Series) -> str | None:
    non_null = values.dropna()
    if non_null.empty:
        return None
    return str(non_null.astype(str).value_counts().idxmax())


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    clean = frame.copy()
    return [
        {key: _json_value(value) for key, value in row.items()}
        for row in clean.to_dict(orient="records")
    ]


def _json_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
