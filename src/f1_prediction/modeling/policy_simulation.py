"""Artifact-based simulations for champion guardrails and conformal calibration."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig, PolicySimulationConfig
from f1_prediction.utils.paths import ensure_directory

CHECKPOINT_ORDER: tuple[str, ...] = ("after_fp1", "after_fp2", "after_fp3")
GUARDRAIL_POLICIES: tuple[str, ...] = (
    "current_static",
    "current_stabilized_nested",
    "fp3_static_lock",
    "fp3_no_baseline_switch",
    "fp3_harmful_event_guardrail_oracle",
    "fp3_selection_confidence_guardrail",
)
CONFORMAL_STRATEGIES: tuple[str, ...] = (
    "global_conformal",
    "checkpoint_conformal",
    "checkpoint_method_conformal",
    "checkpoint_actual_gap_bucket_oracle",
    "checkpoint_predicted_gap_bucket",
)
JOIN_COLUMNS: tuple[str, ...] = ("fold_id", "season", "event_slug", "checkpoint", "driver")
STATIC_FP3_METHOD = ("ablation", "random_forest", "base_plus_relative")


@dataclass(frozen=True)
class PolicySimulationSummary:
    """Paths and issue counts produced by policy simulation generation."""

    status: str
    summary_path: Path
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    missing_inputs: tuple[str, ...]
    generation_issues: tuple[str, ...]


def create_policy_simulation_report(
    config: DataConfig,
    simulation_config: PolicySimulationConfig,
) -> PolicySimulationSummary:
    """Generate FP3 guardrail and regime-aware conformal simulation artifacts."""
    metrics_dir = config.metrics_output_dir
    figures_dir = metrics_dir.parent / "figures"
    ensure_directory(metrics_dir)
    ensure_directory(figures_dir)

    artifacts = _load_policy_simulation_artifacts(metrics_dir)
    predictions = artifacts["predictions"]
    selections = artifacts["selection"]
    missing_inputs = list(artifacts["missing_inputs"])

    guardrail_rows = simulate_guardrail_policies(
        predictions.get("static"),
        predictions.get("stabilized_nested"),
        selections.get("stabilized_nested"),
    )
    guardrail_summary = build_guardrail_simulation_table(guardrail_rows)
    guardrail_events = build_guardrail_event_level_table(guardrail_rows)

    conformal_rows = simulate_regime_conformal_intervals(
        predictions.get("stabilized_nested"),
        simulation_config,
    )
    conformal_summary = build_regime_conformal_simulation_table(conformal_rows)
    conformal_events = build_regime_conformal_event_level_table(conformal_rows)

    table_frames = {
        "fp3_guardrail_simulation_table.csv": guardrail_summary,
        "fp3_guardrail_event_level_table.csv": guardrail_events,
        "regime_conformal_simulation_table.csv": conformal_summary,
        "regime_conformal_event_level_table.csv": conformal_events,
    }
    table_paths: list[Path] = []
    for filename, frame in table_frames.items():
        path = metrics_dir / filename
        frame.to_csv(path, index=False)
        table_paths.append(path)

    figure_paths, figure_issues = generate_policy_simulation_figures(
        figures_dir=figures_dir,
        guardrail_summary=guardrail_summary,
        guardrail_events=guardrail_events,
        conformal_summary=conformal_summary,
    )
    payload = build_policy_simulation_summary_payload(
        inputs_available=artifacts["inputs_available"],
        missing_inputs=missing_inputs,
        guardrail_summary=guardrail_summary,
        guardrail_events=guardrail_events,
        conformal_summary=conformal_summary,
        guarded_mode_summary=build_guarded_mode_artifact_summary(
            predictions.get("static"),
            predictions.get("stabilized_nested"),
            predictions.get("stabilized_nested_guarded"),
            selections.get("stabilized_nested_guarded"),
            guardrail_summary,
        ),
        season_aware_guarded_mode_summary=build_guarded_mode_artifact_summary(
            predictions.get("static"),
            predictions.get("stabilized_nested_guarded"),
            predictions.get("season_aware_nested_guarded"),
            selections.get("season_aware_nested_guarded"),
            guardrail_summary,
        ),
        generated_tables=table_paths,
        generated_figures=figure_paths,
        generation_issues=figure_issues,
    )
    summary_path = metrics_dir / "policy_simulation_summary.json"
    _write_json(summary_path, payload)

    return PolicySimulationSummary(
        status=str(payload["status"]),
        summary_path=summary_path,
        table_paths=tuple(table_paths),
        figure_paths=tuple(figure_paths),
        missing_inputs=tuple(missing_inputs),
        generation_issues=tuple(figure_issues),
    )


def simulate_guardrail_policies(
    static_predictions: pd.DataFrame | None,
    stabilized_predictions: pd.DataFrame | None,
    stabilized_selection: pd.DataFrame | None,
    *,
    oracle_margin_sec: float = 0.05,
) -> pd.DataFrame:
    """Create row-level predictions for each simulated guardrail policy."""
    if static_predictions is None or stabilized_predictions is None:
        return pd.DataFrame(columns=_guardrail_row_columns())
    comparable = _join_static_stabilized_predictions(static_predictions, stabilized_predictions)
    if comparable.empty:
        return pd.DataFrame(columns=_guardrail_row_columns())
    selection_lookup = _selection_lookup(stabilized_selection)
    event_decisions = _oracle_event_decisions(comparable, oracle_margin_sec)

    frames = [
        _guardrail_rows_for_policy(comparable, selection_lookup, event_decisions, policy_name)
        for policy_name in GUARDRAIL_POLICIES
    ]
    return pd.concat(frames, ignore_index=True)[_guardrail_row_columns()]


def build_guardrail_simulation_table(guardrail_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize simulated guardrail metrics by policy and checkpoint."""
    columns = _guardrail_summary_columns()
    if guardrail_rows.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for policy_name, policy_rows in guardrail_rows.groupby("policy_name", sort=False):
        fp3_rows = policy_rows[policy_rows["checkpoint"].eq("after_fp3")]
        fp3_event_metrics = _fp3_event_metrics(policy_rows)
        fp3_mae = _mae(fp3_rows)
        static_by_checkpoint = _reference_mae_by_checkpoint(policy_rows, "static")
        stabilized_by_checkpoint = _reference_mae_by_checkpoint(policy_rows, "stabilized")
        for checkpoint, group in _checkpoint_and_overall_groups(policy_rows):
            metrics = _error_metrics(group)
            rows.append(
                {
                    "policy_name": policy_name,
                    "policy_type": _policy_type(policy_name),
                    "checkpoint": checkpoint,
                    "rows": int(len(group)),
                    "mae_gap_sec": metrics["mae_gap_sec"],
                    "rmse_gap_sec": metrics["rmse_gap_sec"],
                    "median_abs_error_gap_sec": metrics["median_abs_error_gap_sec"],
                    "fp3_mae_gap_sec": fp3_mae,
                    "delta_vs_static_mae_sec": _delta(
                        metrics["mae_gap_sec"],
                        static_by_checkpoint.get(checkpoint),
                    ),
                    "delta_vs_current_stabilized_nested_mae_sec": _delta(
                        metrics["mae_gap_sec"],
                        stabilized_by_checkpoint.get(checkpoint),
                    ),
                    "fp3_rows_replaced": int(
                        policy_rows["replaced_with_static"].fillna(False).sum()
                    ),
                    "fp3_events_affected": int(
                        policy_rows.loc[
                            policy_rows["replaced_with_static"].fillna(False),
                            ["season", "event", "fold_id"],
                        ]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    "worst_fp3_event_mae": _number_or_none(
                        fp3_event_metrics["simulated_policy_mae_gap_sec"].max()
                        if not fp3_event_metrics.empty
                        else None
                    ),
                    "best_fp3_event_mae": _number_or_none(
                        fp3_event_metrics["simulated_policy_mae_gap_sec"].min()
                        if not fp3_event_metrics.empty
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_guardrail_event_level_table(guardrail_rows: pd.DataFrame) -> pd.DataFrame:
    """Create event/fold/policy-level FP3 guardrail diagnostics."""
    columns = _guardrail_event_columns()
    if guardrail_rows.empty:
        return pd.DataFrame(columns=columns)
    fp3 = guardrail_rows[guardrail_rows["checkpoint"].eq("after_fp3")].copy()
    rows: list[dict[str, object]] = []
    for keys, group in fp3.groupby(
        ["policy_name", "season", "event", "fold_id"], dropna=False, sort=False
    ):
        policy_name, season, event, fold_id = keys
        replaced = group["replaced_with_static"].fillna(False)
        rows.append(
            {
                "policy_name": policy_name,
                "policy_type": _policy_type(str(policy_name)),
                "season": season,
                "event": event,
                "fold_id": fold_id,
                "rows": int(len(group)),
                "static_mae_gap_sec": _mae_from_columns(group, "static_predicted"),
                "stabilized_mae_gap_sec": _mae_from_columns(group, "stabilized_predicted"),
                "simulated_policy_mae_gap_sec": _mae_from_columns(group, "simulated_predicted"),
                "delta_vs_static_sec": _delta(
                    _mae_from_columns(group, "simulated_predicted"),
                    _mae_from_columns(group, "static_predicted"),
                ),
                "delta_vs_stabilized_sec": _delta(
                    _mae_from_columns(group, "simulated_predicted"),
                    _mae_from_columns(group, "stabilized_predicted"),
                ),
                "static_method": _method_label(
                    group["static_selected_family"],
                    group["static_selected_model_name"],
                    group["static_selected_feature_group"],
                ),
                "stabilized_method": _method_label(
                    group["stabilized_selected_family"],
                    group["stabilized_selected_model_name"],
                    group["stabilized_selected_feature_group"],
                ),
                "policy_action": _most_common_non_null(group["policy_action"]),
                "replaced_with_static": bool(replaced.any()),
                "replacement_reason": _most_common_non_null(group["replacement_reason"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def simulate_regime_conformal_intervals(
    predictions: pd.DataFrame | None,
    config: PolicySimulationConfig,
) -> pd.DataFrame:
    """Simulate leakage-safe conformal interval strategies from saved predictions."""
    if predictions is None or predictions.empty:
        return pd.DataFrame(columns=_conformal_row_columns())
    required = {
        "fold_id",
        "season",
        "event",
        "checkpoint",
        "driver",
        "quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec",
    }
    if not required <= set(predictions.columns):
        return pd.DataFrame(columns=_conformal_row_columns())

    frame = _normalize_prediction_frame(predictions)
    frame = frame.sort_values(["fold_id", "checkpoint", "driver"]).reset_index(drop=True)
    frame["absolute_residual_sec"] = (
        pd.to_numeric(frame["quali_gap_to_pole_sec"], errors="coerce")
        - pd.to_numeric(frame["predicted_quali_gap_to_pole_sec"], errors="coerce")
    ).abs()
    frame["actual_gap_bucket"] = frame["quali_gap_to_pole_sec"].apply(gap_bucket)
    frame["predicted_gap_bucket"] = frame["predicted_quali_gap_to_pole_sec"].apply(gap_bucket)
    frame["method_key"] = _method_key_series(frame)

    rows: list[dict[str, object]] = []
    conformal_config = config.conformal
    for strategy in CONFORMAL_STRATEGIES:
        is_oracle = strategy.endswith("_oracle")
        for row in frame.itertuples(index=False):
            current = row._asdict()
            history = frame[frame["fold_id"].lt(current["fold_id"])].copy()
            quantile = _select_residual_quantile(
                history,
                current,
                strategy,
                confidence_level=conformal_config.confidence_level,
                min_residual_count=conformal_config.min_residual_count,
                fallback_order=conformal_config.fallback_order,
            )
            predicted = _number_or_none(current["predicted_quali_gap_to_pole_sec"])
            actual = _number_or_none(current["quali_gap_to_pole_sec"])
            if quantile is None or predicted is None or actual is None:
                low = None
                high = None
                width = None
                contains = pd.NA
                miss = pd.NA
            else:
                low = predicted - quantile["residual_quantile_sec"]
                high = predicted + quantile["residual_quantile_sec"]
                width = high - low
                contains = bool(low <= actual <= high)
                miss = not contains
            rows.append(
                {
                    "strategy_name": strategy,
                    "strategy_type": "oracle" if is_oracle else "deployable",
                    "season": current["season"],
                    "event": current["event"],
                    "fold_id": current["fold_id"],
                    "checkpoint": current["checkpoint"],
                    "driver": current["driver"],
                    "team": current.get("team"),
                    "selected_family": current.get("selected_family"),
                    "selected_model_name": current.get("selected_model_name"),
                    "selected_feature_group": current.get("selected_feature_group"),
                    "actual_quali_gap_to_pole_sec": actual,
                    "predicted_quali_gap_to_pole_sec": predicted,
                    "actual_gap_bucket": current["actual_gap_bucket"],
                    "predicted_gap_bucket": current["predicted_gap_bucket"],
                    "residual_quantile_sec": (
                        quantile["residual_quantile_sec"] if quantile is not None else None
                    ),
                    "residual_count": quantile["residual_count"] if quantile is not None else 0,
                    "calibration_level": (
                        quantile["calibration_level"] if quantile is not None else None
                    ),
                    "prediction_interval_low_sec": low,
                    "prediction_interval_high_sec": high,
                    "interval_width_sec": width,
                    "interval_available": quantile is not None,
                    "interval_contains_actual": contains,
                    "interval_miss": miss,
                }
            )
    return pd.DataFrame(rows, columns=_conformal_row_columns())


def build_regime_conformal_simulation_table(conformal_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize conformal simulation metrics overall, by checkpoint, and by bucket."""
    columns = _conformal_summary_columns()
    if conformal_rows.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for keys, group in conformal_rows.groupby(
        ["strategy_name", "strategy_type"], dropna=False, sort=False
    ):
        strategy_name, strategy_type = keys
        rows.append(
            _conformal_summary_row(
                group,
                strategy_name=strategy_name,
                strategy_type=strategy_type,
                summary_scope="overall",
                checkpoint="overall",
                bucket_type="all",
                gap_bucket="all",
            )
        )
        for checkpoint, checkpoint_group in group.groupby("checkpoint", sort=False):
            rows.append(
                _conformal_summary_row(
                    checkpoint_group,
                    strategy_name=strategy_name,
                    strategy_type=strategy_type,
                    summary_scope="checkpoint",
                    checkpoint=checkpoint,
                    bucket_type="all",
                    gap_bucket="all",
                )
            )
        for bucket_type, bucket_column in (
            ("predicted_gap_bucket", "predicted_gap_bucket"),
            ("actual_gap_bucket", "actual_gap_bucket"),
        ):
            for bucket_keys, bucket_group in group.groupby(
                ["checkpoint", bucket_column], dropna=False, sort=False
            ):
                checkpoint, bucket = bucket_keys
                rows.append(
                    _conformal_summary_row(
                        bucket_group,
                        strategy_name=strategy_name,
                        strategy_type=strategy_type,
                        summary_scope=bucket_type,
                        checkpoint=checkpoint,
                        bucket_type=bucket_type,
                        gap_bucket=bucket,
                    )
                )
    return pd.DataFrame(rows, columns=columns)


def build_regime_conformal_event_level_table(conformal_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize conformal simulation metrics by event/fold/checkpoint."""
    columns = _conformal_event_columns()
    if conformal_rows.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for keys, group in conformal_rows.groupby(
        ["strategy_name", "strategy_type", "season", "event", "fold_id", "checkpoint"],
        dropna=False,
        sort=False,
    ):
        strategy_name, strategy_type, season, event, fold_id, checkpoint = keys
        base = _interval_metrics(group)
        base.update(
            {
                "strategy_name": strategy_name,
                "strategy_type": strategy_type,
                "season": season,
                "event": event,
                "fold_id": fold_id,
                "checkpoint": checkpoint,
            }
        )
        rows.append(base)
    return pd.DataFrame(rows, columns=columns)


def build_policy_simulation_summary_payload(
    *,
    inputs_available: dict[str, bool],
    missing_inputs: list[str],
    guardrail_summary: pd.DataFrame,
    guardrail_events: pd.DataFrame,
    conformal_summary: pd.DataFrame,
    guarded_mode_summary: dict[str, object] | None,
    season_aware_guarded_mode_summary: dict[str, object] | None,
    generated_tables: list[Path],
    generated_figures: list[Path],
    generation_issues: list[str],
) -> dict[str, Any]:
    """Build machine-readable simulation summary JSON."""
    best_guardrail_fp3 = _best_guardrail(guardrail_summary, checkpoint="after_fp3")
    best_guardrail_overall = _best_guardrail(guardrail_summary, checkpoint="overall")
    oracle_guardrail = _oracle_guardrail(guardrail_summary)
    best_conformal_fp3 = _best_conformal_by_coverage(conformal_summary, checkpoint="after_fp3")
    best_conformal_adjusted = _best_conformal_by_width_adjusted_coverage(
        conformal_summary,
        checkpoint="after_fp3",
    )
    main_findings = _main_findings(
        guardrail_summary=guardrail_summary,
        guardrail_events=guardrail_events,
        best_guardrail_fp3=best_guardrail_fp3,
        oracle_guardrail=oracle_guardrail,
        best_conformal_fp3=best_conformal_fp3,
    )
    return {
        "status": _report_status(inputs_available, missing_inputs),
        "inputs_available": inputs_available,
        "missing_inputs": sorted(missing_inputs),
        "guardrail_policies_tested": list(GUARDRAIL_POLICIES),
        "best_non_oracle_guardrail_by_fp3_mae": best_guardrail_fp3,
        "best_non_oracle_guardrail_by_overall_mae": best_guardrail_overall,
        "oracle_guardrail_upper_bound": oracle_guardrail,
        "guarded_mode_artifact_summary": guarded_mode_summary or {"available": False},
        "season_aware_guarded_mode_artifact_summary": (
            season_aware_guarded_mode_summary or {"available": False}
        ),
        "conformal_strategies_tested": list(CONFORMAL_STRATEGIES),
        "best_non_oracle_conformal_by_fp3_coverage": best_conformal_fp3,
        "best_non_oracle_conformal_by_fp3_width_adjusted_coverage": best_conformal_adjusted,
        "main_findings": main_findings,
        "recommended_actions": _recommended_actions(main_findings, missing_inputs),
        "generated_tables": [str(path) for path in generated_tables],
        "generated_figures": [str(path) for path in generated_figures],
        "generation_issues": generation_issues,
        "generated_at": _utc_now(),
    }


def build_guarded_mode_artifact_summary(
    static_predictions: pd.DataFrame | None,
    stabilized_predictions: pd.DataFrame | None,
    guarded_predictions: pd.DataFrame | None,
    guarded_selection: pd.DataFrame | None,
    guardrail_summary: pd.DataFrame,
) -> dict[str, object]:
    """Summarize real guarded-mode artifacts when they are available."""
    if guarded_predictions is None or guarded_predictions.empty:
        return {"available": False}
    summary: dict[str, object] = {
        "available": True,
        "mae_by_checkpoint": _mae_by_checkpoint(guarded_predictions),
        "fp3_mae_gap_sec": _checkpoint_mae(guarded_predictions, "after_fp3"),
        "guardrail_applied_count": 0,
        "fp3_guardrail_applied_count": 0,
    }
    if static_predictions is not None and not static_predictions.empty:
        summary["fp3_delta_vs_static_mae_sec"] = _delta(
            summary["fp3_mae_gap_sec"],
            _checkpoint_mae(static_predictions, "after_fp3"),
        )
    if stabilized_predictions is not None and not stabilized_predictions.empty:
        summary["fp3_delta_vs_stabilized_nested_mae_sec"] = _delta(
            summary["fp3_mae_gap_sec"],
            _checkpoint_mae(stabilized_predictions, "after_fp3"),
        )
    if guarded_selection is not None and "guardrail_applied" in guarded_selection:
        applied = guarded_selection[guarded_selection["guardrail_applied"].astype(bool)].copy()
        summary["guardrail_applied_count"] = int(len(applied))
        if "checkpoint" in applied:
            summary["fp3_guardrail_applied_count"] = int(
                applied["checkpoint"].astype(str).eq("after_fp3").sum()
            )
    simulated = _policy_checkpoint_row(
        guardrail_summary,
        "fp3_no_baseline_switch",
        "after_fp3",
    )
    if simulated:
        summary["simulated_fp3_no_baseline_switch_mae_gap_sec"] = simulated.get("mae_gap_sec")
        summary["matches_simulated_fp3_no_baseline_switch_mae"] = _close(
            summary["fp3_mae_gap_sec"], simulated.get("mae_gap_sec")
        )
    return summary


def generate_policy_simulation_figures(
    *,
    figures_dir: Path,
    guardrail_summary: pd.DataFrame,
    guardrail_events: pd.DataFrame,
    conformal_summary: pd.DataFrame,
) -> tuple[list[Path], list[str]]:
    """Create non-interactive matplotlib policy simulation figures."""
    ensure_directory(figures_dir)
    paths: list[Path] = []
    issues: list[str] = []
    plot_specs = (
        (
            "fp3_guardrail_policy_mae.png",
            lambda plt: _plot_fp3_guardrail_policy_mae(plt, guardrail_summary),
        ),
        (
            "fp3_guardrail_delta_vs_static.png",
            lambda plt: _plot_fp3_guardrail_delta_vs_static(plt, guardrail_summary),
        ),
        (
            "fp3_guardrail_event_mae.png",
            lambda plt: _plot_fp3_guardrail_event_mae(plt, guardrail_summary, guardrail_events),
        ),
        (
            "regime_conformal_fp3_coverage_width.png",
            lambda plt: _plot_regime_conformal_fp3_coverage_width(plt, conformal_summary),
        ),
        (
            "regime_conformal_coverage_by_bucket.png",
            lambda plt: _plot_regime_conformal_coverage_by_bucket(plt, conformal_summary),
        ),
    )
    try:
        plt = _load_matplotlib()
    except Exception as exc:  # pragma: no cover - depends on local matplotlib install
        return [], [f"matplotlib unavailable: {exc}"]
    for filename, plotter in plot_specs:
        path = figures_dir / filename
        try:
            if plotter(plt):
                plt.savefig(path, dpi=160, bbox_inches="tight")
                paths.append(path)
            else:
                issues.append(f"Skipped {filename}: required data unavailable")
        except Exception as exc:  # pragma: no cover - defensive report generation
            issues.append(f"Skipped {filename}: {exc}")
        finally:
            plt.close()
    return paths, issues


def gap_bucket(value: object) -> str | None:
    """Assign qualifying-gap regime bucket using fixed thresholds."""
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


def _load_policy_simulation_artifacts(metrics_dir: Path) -> dict[str, object]:
    required_predictions = {
        "static": "champion_static_predictions.parquet",
        "stabilized_nested": "champion_stabilized_nested_predictions.parquet",
    }
    required_selection = {
        "static": "champion_static_selection.parquet",
        "stabilized_nested": "champion_stabilized_nested_selection.parquet",
    }
    optional_inputs = (
        "champion_stabilized_nested_guarded_predictions.parquet",
        "champion_stabilized_nested_guarded_selection.parquet",
        "champion_season_aware_nested_guarded_predictions.parquet",
        "champion_season_aware_nested_guarded_selection.parquet",
        "fp3_policy_failure_analysis.csv",
        "champion_harmful_switches.csv",
    )
    predictions: dict[str, pd.DataFrame] = {}
    selection: dict[str, pd.DataFrame] = {}
    inputs_available: dict[str, bool] = {}
    missing_inputs: list[str] = []
    for mode, filename in required_predictions.items():
        path = metrics_dir / filename
        inputs_available[filename] = path.is_file()
        if path.is_file():
            predictions[mode] = pd.read_parquet(path)
        else:
            missing_inputs.append(filename)
    for mode, filename in required_selection.items():
        path = metrics_dir / filename
        inputs_available[filename] = path.is_file()
        if path.is_file():
            selection[mode] = pd.read_parquet(path)
        else:
            missing_inputs.append(filename)
    for filename in optional_inputs:
        path = metrics_dir / filename
        inputs_available[filename] = path.is_file()
        if filename == "champion_stabilized_nested_guarded_predictions.parquet" and path.is_file():
            predictions["stabilized_nested_guarded"] = pd.read_parquet(path)
        elif filename == "champion_stabilized_nested_guarded_selection.parquet" and path.is_file():
            selection["stabilized_nested_guarded"] = pd.read_parquet(path)
        elif (
            filename == "champion_season_aware_nested_guarded_predictions.parquet"
            and path.is_file()
        ):
            predictions["season_aware_nested_guarded"] = pd.read_parquet(path)
        elif (
            filename == "champion_season_aware_nested_guarded_selection.parquet" and path.is_file()
        ):
            selection["season_aware_nested_guarded"] = pd.read_parquet(path)
    return {
        "predictions": predictions,
        "selection": selection,
        "inputs_available": inputs_available,
        "missing_inputs": missing_inputs,
    }


def _join_static_stabilized_predictions(
    static_predictions: pd.DataFrame,
    stabilized_predictions: pd.DataFrame,
) -> pd.DataFrame:
    static = _normalize_prediction_frame(static_predictions)
    stabilized = _normalize_prediction_frame(stabilized_predictions)
    required = {
        *JOIN_COLUMNS,
        "event",
        "team",
        "quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec",
        "selected_family",
        "selected_model_name",
        "selected_feature_group",
    }
    if not required <= set(static.columns) or not required <= set(stabilized.columns):
        return pd.DataFrame()
    static_columns = [
        *JOIN_COLUMNS,
        "event",
        "team",
        "quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec",
        "selected_family",
        "selected_model_name",
        "selected_feature_group",
    ]
    stabilized_columns = [
        *JOIN_COLUMNS,
        "predicted_quali_gap_to_pole_sec",
        "selected_family",
        "selected_model_name",
        "selected_feature_group",
    ]
    joined = static[static_columns].merge(
        stabilized[stabilized_columns],
        on=list(JOIN_COLUMNS),
        how="inner",
        suffixes=("_static", "_stabilized"),
    )
    rename = {
        "quali_gap_to_pole_sec": "actual_quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec_static": "static_predicted",
        "predicted_quali_gap_to_pole_sec_stabilized": "stabilized_predicted",
        "selected_family_static": "static_selected_family",
        "selected_model_name_static": "static_selected_model_name",
        "selected_feature_group_static": "static_selected_feature_group",
        "selected_family_stabilized": "stabilized_selected_family",
        "selected_model_name_stabilized": "stabilized_selected_model_name",
        "selected_feature_group_stabilized": "stabilized_selected_feature_group",
    }
    return joined.rename(columns=rename)


def _guardrail_rows_for_policy(
    comparable: pd.DataFrame,
    selection_lookup: dict[tuple[object, object], dict[str, object]],
    oracle_event_decisions: dict[tuple[object, object, object], bool],
    policy_name: str,
) -> pd.DataFrame:
    frame = comparable.copy()
    frame["policy_name"] = policy_name
    frame["policy_type"] = _policy_type(policy_name)
    frame["policy_action"] = "use_stabilized_nested"
    frame["replacement_reason"] = pd.NA
    frame["replaced_with_static"] = False

    use_static = pd.Series(False, index=frame.index)
    if policy_name == "current_static":
        use_static = pd.Series(True, index=frame.index)
        frame["policy_action"] = "use_static"
    elif policy_name == "current_stabilized_nested":
        frame["policy_action"] = "use_stabilized_nested"
    elif policy_name == "fp3_static_lock":
        use_static = frame["checkpoint"].eq("after_fp3")
        frame.loc[use_static, "policy_action"] = "fp3_static_lock"
        frame.loc[use_static, "replacement_reason"] = "fp3_static_lock"
    elif policy_name == "fp3_no_baseline_switch":
        use_static = frame.apply(_is_fp3_baseline_switch, axis=1)
        frame.loc[use_static, "policy_action"] = "blocked_baseline_switch"
        frame.loc[use_static, "replacement_reason"] = "fp3_baseline_switch"
    elif policy_name == "fp3_harmful_event_guardrail_oracle":
        fp3 = frame["checkpoint"].eq("after_fp3")
        use_static = fp3 & ~frame.apply(
            lambda row: oracle_event_decisions.get(
                (row["season"], row["event"], row["fold_id"]),
                False,
            ),
            axis=1,
        )
        frame.loc[fp3 & ~use_static, "policy_action"] = "oracle_use_stabilized"
        frame.loc[use_static, "policy_action"] = "oracle_use_static"
        frame.loc[use_static, "replacement_reason"] = "oracle_event_mae_not_better"
    elif policy_name == "fp3_selection_confidence_guardrail":
        use_static = frame.apply(
            lambda row: _selection_confidence_uses_static(row, selection_lookup),
            axis=1,
        )
        frame.loc[use_static, "policy_action"] = "selection_confidence_use_static"
        frame.loc[use_static, "replacement_reason"] = frame.loc[use_static].apply(
            lambda row: _selection_confidence_reason(row, selection_lookup),
            axis=1,
        )

    frame["simulated_predicted"] = frame["stabilized_predicted"]
    frame.loc[use_static, "simulated_predicted"] = frame.loc[use_static, "static_predicted"]
    if policy_name in {"current_static", "current_stabilized_nested"}:
        frame["replaced_with_static"] = False
    else:
        frame["replaced_with_static"] = use_static & frame["checkpoint"].eq("after_fp3")
    return frame[_guardrail_row_columns()]


def _selection_confidence_uses_static(
    row: pd.Series,
    selection_lookup: dict[tuple[object, object], dict[str, object]],
) -> bool:
    if row["checkpoint"] != "after_fp3":
        return False
    values = selection_lookup.get((row["fold_id"], row["checkpoint"]))
    if not values:
        return True
    if values.get("fallback_used") is True:
        return True
    prior_predictions = _number_or_none(values.get("prior_predictions_used"))
    min_prior_predictions = _number_or_none(values.get("min_prior_predictions"))
    selected_metric = _number_or_none(values.get("selected_metric_value"))
    default_metric = _number_or_none(values.get("default_metric_value"))
    margin = _number_or_none(values.get("improvement_margin_sec")) or 0.0
    if (
        prior_predictions is None
        or min_prior_predictions is None
        or selected_metric is None
        or default_metric is None
    ):
        return True
    if prior_predictions < min_prior_predictions:
        return True
    return not (selected_metric <= default_metric - margin)


def _selection_confidence_reason(
    row: pd.Series,
    selection_lookup: dict[tuple[object, object], dict[str, object]],
) -> str:
    values = selection_lookup.get((row["fold_id"], row["checkpoint"]))
    if not values:
        return "selection_metadata_unavailable"
    if values.get("fallback_used") is True:
        return str(values.get("fallback_reason") or "fallback_used")
    prior_predictions = _number_or_none(values.get("prior_predictions_used"))
    min_prior_predictions = _number_or_none(values.get("min_prior_predictions"))
    selected_metric = _number_or_none(values.get("selected_metric_value"))
    default_metric = _number_or_none(values.get("default_metric_value"))
    if prior_predictions is None or min_prior_predictions is None:
        return "prior_prediction_metadata_unavailable"
    if prior_predictions < min_prior_predictions:
        return "insufficient_prior_predictions"
    if selected_metric is None or default_metric is None:
        return "metric_metadata_unavailable"
    return "improvement_margin_not_met"


def _is_fp3_baseline_switch(row: pd.Series) -> bool:
    return bool(
        row["checkpoint"] == "after_fp3"
        and _is_static_fp3_rf(row)
        and _is_baseline_like(
            row["stabilized_selected_family"],
            row["stabilized_selected_model_name"],
        )
        and not _stabilized_matches_static_fp3_rf(row)
    )


def _is_static_fp3_rf(row: pd.Series) -> bool:
    return (
        row.get("static_selected_family") == STATIC_FP3_METHOD[0]
        and row.get("static_selected_model_name") == STATIC_FP3_METHOD[1]
        and row.get("static_selected_feature_group") == STATIC_FP3_METHOD[2]
    )


def _stabilized_matches_static_fp3_rf(row: pd.Series) -> bool:
    return (
        row.get("stabilized_selected_family") == STATIC_FP3_METHOD[0]
        and row.get("stabilized_selected_model_name") == STATIC_FP3_METHOD[1]
        and row.get("stabilized_selected_feature_group") == STATIC_FP3_METHOD[2]
    )


def _is_baseline_like(family: object, model_name: object) -> bool:
    family_text = str(family).lower()
    model_text = str(model_name).lower()
    return family_text in {"baseline", "robust_baseline"} or any(
        token in model_text
        for token in (
            "best_valid_lap",
            "best_push_lap",
            "theoretical_best_lap",
            "baseline",
        )
    )


def _oracle_event_decisions(
    comparable: pd.DataFrame,
    margin_sec: float,
) -> dict[tuple[object, object, object], bool]:
    fp3 = comparable[comparable["checkpoint"].eq("after_fp3")]
    decisions: dict[tuple[object, object, object], bool] = {}
    for keys, group in fp3.groupby(["season", "event", "fold_id"], dropna=False, sort=False):
        static_mae = _mae_from_columns(group, "static_predicted")
        stabilized_mae = _mae_from_columns(group, "stabilized_predicted")
        decisions[keys] = bool(
            stabilized_mae is not None
            and static_mae is not None
            and stabilized_mae <= static_mae - margin_sec
        )
    return decisions


def _selection_lookup(
    selection: pd.DataFrame | None,
) -> dict[tuple[object, object], dict[str, object]]:
    if selection is None or selection.empty:
        return {}
    frame = selection.copy()
    lookup: dict[tuple[object, object], dict[str, object]] = {}
    for keys, group in frame.groupby(["fold_id", "checkpoint"], dropna=False, sort=False):
        row = group.iloc[0].to_dict()
        row["fallback_used"] = _bool_or_false(row.get("fallback_used"))
        lookup[keys] = row
    return lookup


def _select_residual_quantile(
    history: pd.DataFrame,
    current: dict[str, object],
    strategy_name: str,
    *,
    confidence_level: float,
    min_residual_count: int,
    fallback_order: tuple[str, ...],
) -> dict[str, object] | None:
    if history.empty:
        return None
    levels = _strategy_levels(strategy_name, fallback_order)
    for level in levels:
        subset = _history_subset(history, current, level, strategy_name)
        residuals = pd.to_numeric(subset["absolute_residual_sec"], errors="coerce").dropna()
        if len(residuals) >= min_residual_count:
            return {
                "residual_quantile_sec": float(
                    residuals.quantile(confidence_level, interpolation="higher")
                ),
                "residual_count": int(len(residuals)),
                "calibration_level": level,
            }
    return None


def _strategy_levels(strategy_name: str, fallback_order: tuple[str, ...]) -> tuple[str, ...]:
    if strategy_name == "global_conformal":
        return ("global",)
    if strategy_name == "checkpoint_conformal":
        return ("checkpoint", "global")
    if strategy_name == "checkpoint_method_conformal":
        return ("checkpoint_method", "checkpoint", "global")
    if strategy_name == "checkpoint_actual_gap_bucket_oracle":
        return tuple(fallback_order)
    if strategy_name == "checkpoint_predicted_gap_bucket":
        return tuple(fallback_order)
    return ("global",)


def _history_subset(
    history: pd.DataFrame,
    current: dict[str, object],
    level: str,
    strategy_name: str,
) -> pd.DataFrame:
    subset = history
    if level in {
        "checkpoint_method_bucket",
        "checkpoint_bucket",
        "checkpoint_method",
        "checkpoint",
    }:
        subset = subset[subset["checkpoint"].eq(current["checkpoint"])]
    if level in {"checkpoint_method_bucket", "checkpoint_method"}:
        subset = subset[subset["method_key"].eq(current["method_key"])]
    if level in {"checkpoint_method_bucket", "checkpoint_bucket"}:
        bucket_column = (
            "actual_gap_bucket"
            if strategy_name == "checkpoint_actual_gap_bucket_oracle"
            else "predicted_gap_bucket"
        )
        subset = subset[subset[bucket_column].eq(current[bucket_column])]
    return subset


def _conformal_summary_row(
    group: pd.DataFrame,
    *,
    strategy_name: object,
    strategy_type: object,
    summary_scope: str,
    checkpoint: object,
    bucket_type: object,
    gap_bucket: object,
) -> dict[str, object]:
    metrics = _interval_metrics(group)
    metrics.update(
        {
            "strategy_name": strategy_name,
            "strategy_type": strategy_type,
            "summary_scope": summary_scope,
            "checkpoint": checkpoint,
            "bucket_type": bucket_type,
            "gap_bucket": gap_bucket,
        }
    )
    return metrics


def _interval_metrics(group: pd.DataFrame) -> dict[str, object]:
    available = group[group["interval_available"].fillna(False)].copy()
    contains = available["interval_contains_actual"].dropna()
    miss = available["interval_miss"].dropna()
    return {
        "rows": int(len(group)),
        "interval_availability_rate": float(len(available) / len(group)) if len(group) else None,
        "coverage": float(contains.astype(bool).mean()) if not contains.empty else None,
        "mean_interval_width_sec": _mean_or_none(available["interval_width_sec"]),
        "median_interval_width_sec": _median_or_none(available["interval_width_sec"]),
        "miss_count": int(miss.astype(bool).sum()) if not miss.empty else 0,
        "miss_rate": float(miss.astype(bool).mean()) if not miss.empty else None,
        "mean_residual_quantile_sec": _mean_or_none(available["residual_quantile_sec"]),
    }


def _error_metrics(group: pd.DataFrame) -> dict[str, float | None]:
    actual = pd.to_numeric(group["actual_quali_gap_to_pole_sec"], errors="coerce")
    predicted = pd.to_numeric(group["simulated_predicted"], errors="coerce")
    error = predicted - actual
    abs_error = error.abs().dropna()
    if abs_error.empty:
        return {
            "mae_gap_sec": None,
            "rmse_gap_sec": None,
            "median_abs_error_gap_sec": None,
        }
    return {
        "mae_gap_sec": float(abs_error.mean()),
        "rmse_gap_sec": float(math.sqrt(float((error.dropna() ** 2).mean()))),
        "median_abs_error_gap_sec": float(abs_error.median()),
    }


def _checkpoint_and_overall_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups = [("overall", frame)]
    for checkpoint in CHECKPOINT_ORDER:
        subset = frame[frame["checkpoint"].eq(checkpoint)]
        if not subset.empty:
            groups.append((checkpoint, subset))
    return groups


def _reference_mae_by_checkpoint(frame: pd.DataFrame, source: str) -> dict[str, float | None]:
    predicted_column = f"{source}_predicted"
    values = {"overall": _mae_from_columns(frame, predicted_column)}
    for checkpoint in CHECKPOINT_ORDER:
        subset = frame[frame["checkpoint"].eq(checkpoint)]
        values[checkpoint] = _mae_from_columns(subset, predicted_column)
    return values


def _fp3_event_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        build_guardrail_event_level_table(frame[frame["checkpoint"].eq("after_fp3")].copy())
        if "policy_name" in frame
        else pd.DataFrame()
    )


def _mae(group: pd.DataFrame) -> float | None:
    if group.empty:
        return None
    return _mae_from_columns(group, "simulated_predicted")


def _mae_from_columns(group: pd.DataFrame, predicted_column: str) -> float | None:
    if group.empty or predicted_column not in group:
        return None
    actual = pd.to_numeric(group["actual_quali_gap_to_pole_sec"], errors="coerce")
    predicted = pd.to_numeric(group[predicted_column], errors="coerce")
    error = (predicted - actual).abs().dropna()
    return float(error.mean()) if not error.empty else None


def _normalize_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "driver" not in normalized and "driver_key" in normalized:
        normalized["driver"] = normalized["driver_key"]
    if "team" not in normalized and "team_key" in normalized:
        normalized["team"] = normalized["team_key"]
    if "team" not in normalized:
        normalized["team"] = pd.NA
    if "event_slug" not in normalized and "event" in normalized:
        normalized["event_slug"] = normalized["event"].astype(str)
    for column in ("selected_family", "selected_model_name", "selected_feature_group"):
        if column not in normalized:
            normalized[column] = pd.NA
    return normalized


def _method_key_series(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["selected_family"].fillna("").astype(str)
        + "/"
        + frame["selected_model_name"].fillna("").astype(str)
        + "/"
        + frame["selected_feature_group"].fillna("").astype(str)
    )


def _method_label(family: pd.Series, model_name: pd.Series, feature_group: pd.Series) -> str | None:
    values = pd.DataFrame(
        {"family": family, "model_name": model_name, "feature_group": feature_group}
    )
    if values.empty:
        return None
    row = values.mode(dropna=False).iloc[0]
    parts = [_display_optional(row[column]) for column in ("family", "model_name", "feature_group")]
    return "/".join(part for part in parts if part)


def _best_guardrail(
    guardrail_summary: pd.DataFrame,
    *,
    checkpoint: str,
) -> dict[str, object] | None:
    if guardrail_summary.empty:
        return None
    frame = guardrail_summary[
        guardrail_summary["checkpoint"].eq(checkpoint)
        & guardrail_summary["policy_type"].eq("deployable")
        & ~guardrail_summary["policy_name"].isin(["current_static", "current_stabilized_nested"])
    ].copy()
    if frame.empty:
        return None
    return _record(
        frame.sort_values(
            ["mae_gap_sec", "fp3_rows_replaced"],
            ascending=[True, True],
            na_position="last",
        ).head(1)
    )


def _oracle_guardrail(guardrail_summary: pd.DataFrame) -> dict[str, object] | None:
    if guardrail_summary.empty:
        return None
    frame = guardrail_summary[
        guardrail_summary["checkpoint"].eq("after_fp3")
        & guardrail_summary["policy_type"].eq("oracle")
    ]
    return _record(frame.sort_values("mae_gap_sec", ascending=True).head(1))


def _best_conformal_by_coverage(
    conformal_summary: pd.DataFrame,
    *,
    checkpoint: str,
) -> dict[str, object] | None:
    frame = _deployable_checkpoint_conformal(conformal_summary, checkpoint)
    if frame.empty:
        return None
    return _record(
        frame.sort_values(
            ["coverage", "mean_interval_width_sec"],
            ascending=[False, True],
            na_position="last",
        ).head(1)
    )


def _best_conformal_by_width_adjusted_coverage(
    conformal_summary: pd.DataFrame,
    *,
    checkpoint: str,
) -> dict[str, object] | None:
    frame = _deployable_checkpoint_conformal(conformal_summary, checkpoint)
    if frame.empty:
        return None
    frame = frame.copy()
    frame["width_adjusted_coverage_score"] = frame["coverage"] / frame[
        "mean_interval_width_sec"
    ].where(frame["mean_interval_width_sec"] > 0)
    return _record(
        frame.sort_values(
            ["width_adjusted_coverage_score", "coverage"],
            ascending=[False, False],
            na_position="last",
        ).head(1)
    )


def _deployable_checkpoint_conformal(
    conformal_summary: pd.DataFrame,
    checkpoint: str,
) -> pd.DataFrame:
    if conformal_summary.empty:
        return pd.DataFrame()
    return conformal_summary[
        conformal_summary["checkpoint"].eq(checkpoint)
        & conformal_summary["summary_scope"].eq("checkpoint")
        & conformal_summary["strategy_type"].eq("deployable")
        & conformal_summary["coverage"].notna()
    ].copy()


def _main_findings(
    *,
    guardrail_summary: pd.DataFrame,
    guardrail_events: pd.DataFrame,
    best_guardrail_fp3: dict[str, object] | None,
    oracle_guardrail: dict[str, object] | None,
    best_conformal_fp3: dict[str, object] | None,
) -> list[str]:
    findings: list[str] = []
    if not guardrail_summary.empty:
        stabilized = _policy_checkpoint_row(
            guardrail_summary,
            "current_stabilized_nested",
            "after_fp3",
        )
        no_baseline = _policy_checkpoint_row(
            guardrail_summary,
            "fp3_no_baseline_switch",
            "after_fp3",
        )
        if stabilized and no_baseline:
            delta = _delta(no_baseline.get("mae_gap_sec"), stabilized.get("mae_gap_sec"))
            if delta is not None and delta < 0:
                findings.append(
                    "fp3_no_baseline_switch improves FP3 MAE versus current "
                    f"stabilized_nested by {abs(delta):.3f} sec."
                )
        silverstone = guardrail_events[
            guardrail_events["event"].astype(str).str.lower().eq("silverstone")
            & guardrail_events["policy_name"].eq("fp3_no_baseline_switch")
            & guardrail_events["replaced_with_static"].astype(bool)
        ]
        if not silverstone.empty:
            findings.append(
                "fp3_no_baseline_switch prevents a Silverstone FP3 baseline switch "
                "in the saved artifacts."
            )
    if best_guardrail_fp3:
        findings.append(
            f"{best_guardrail_fp3['policy_name']} is the best non-oracle FP3 guardrail "
            f"by MAE at {best_guardrail_fp3['mae_gap_sec']:.3f} sec."
        )
    if oracle_guardrail:
        findings.append(
            "fp3_harmful_event_guardrail_oracle is an evaluation-only upper bound "
            f"with FP3 MAE {oracle_guardrail['mae_gap_sec']:.3f} sec."
        )
    if best_conformal_fp3:
        findings.append(
            f"{best_conformal_fp3['strategy_name']} has the best non-oracle FP3 "
            f"coverage at {best_conformal_fp3['coverage']:.1%}."
        )
    if not findings:
        findings.append("Available artifacts were insufficient for supported simulation findings.")
    return findings


def _recommended_actions(findings: list[str], missing_inputs: list[str]) -> list[str]:
    actions: list[str] = []
    if missing_inputs:
        actions.append("Regenerate missing champion artifacts before interpreting simulations.")
    if any("fp3_no_baseline_switch" in finding for finding in findings):
        actions.append("Promote fp3_no_baseline_switch to a candidate real policy in Milestone 19.")
    if any("coverage" in finding for finding in findings):
        actions.append("Inspect regime conformal intervals before changing live uncertainty logic.")
    if not actions:
        actions.append("Review simulation tables before modifying champion selection behavior.")
    return actions


def _policy_checkpoint_row(
    table: pd.DataFrame,
    policy_name: str,
    checkpoint: str,
) -> dict[str, object] | None:
    row = table[table["policy_name"].eq(policy_name) & table["checkpoint"].eq(checkpoint)]
    return _record(row.head(1)) if not row.empty else None


def _report_status(inputs_available: dict[str, bool], missing_inputs: list[str]) -> str:
    if not any(inputs_available.values()):
        return "no_inputs"
    if missing_inputs:
        return "partial"
    return "complete"


def _plot_fp3_guardrail_policy_mae(plt: Any, table: pd.DataFrame) -> bool:
    frame = table[table["checkpoint"].eq("after_fp3")].copy()
    if frame.empty:
        return False
    plt.figure(figsize=(8, 4.5))
    plt.bar(frame["policy_name"], frame["mae_gap_sec"], color="#4c78a8")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("FP3 MAE gap (sec)")
    plt.title("FP3 Guardrail Policy MAE")
    _finish_plot(plt)
    return True


def _plot_fp3_guardrail_delta_vs_static(plt: Any, table: pd.DataFrame) -> bool:
    frame = table[table["checkpoint"].eq("after_fp3")].copy()
    if frame.empty:
        return False
    plt.figure(figsize=(8, 4.5))
    plt.bar(frame["policy_name"], frame["delta_vs_static_mae_sec"], color="#f58518")
    plt.axhline(0, color="#333333", linewidth=0.8)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Delta vs static MAE (sec)")
    plt.title("FP3 Guardrail Delta vs Static")
    _finish_plot(plt)
    return True


def _plot_fp3_guardrail_event_mae(
    plt: Any,
    summary: pd.DataFrame,
    events: pd.DataFrame,
) -> bool:
    if events.empty:
        return False
    best = _best_guardrail(summary, checkpoint="after_fp3")
    if not best:
        return False
    best_policy = str(best["policy_name"])
    frame = events[
        events["policy_name"].isin(["current_static", "current_stabilized_nested", best_policy])
    ].copy()
    if frame.empty:
        return False
    frame["event_label"] = frame["season"].astype(str) + " " + frame["event"].astype(str)
    pivot = frame.pivot_table(
        index="event_label",
        columns="policy_name",
        values="simulated_policy_mae_gap_sec",
        aggfunc="mean",
    )
    if pivot.empty:
        return False
    pivot = pivot.sort_index()
    x_values = list(range(len(pivot.index)))
    policies = [
        column
        for column in ["current_static", "current_stabilized_nested", best_policy]
        if column in pivot
    ]
    width = 0.75 / max(len(policies), 1)
    plt.figure(figsize=(max(9, len(pivot.index) * 0.45), 4.8))
    for index, policy in enumerate(policies):
        offsets = [value - 0.375 + width / 2 + index * width for value in x_values]
        plt.bar(offsets, pivot[policy], width=width, label=policy)
    plt.xticks(x_values, pivot.index, rotation=60, ha="right")
    plt.ylabel("FP3 event MAE (sec)")
    plt.title("FP3 Event MAE by Guardrail")
    plt.legend()
    _finish_plot(plt)
    return True


def _plot_regime_conformal_fp3_coverage_width(plt: Any, table: pd.DataFrame) -> bool:
    frame = table[
        table["checkpoint"].eq("after_fp3") & table["summary_scope"].eq("checkpoint")
    ].copy()
    if frame.empty:
        return False
    x_values = list(range(len(frame)))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x_values, frame["coverage"], color="#54a24b", label="coverage")
    width_scaled = frame["mean_interval_width_sec"] / frame["mean_interval_width_sec"].max()
    plt.plot(x_values, width_scaled, color="#e45756", marker="o", label="mean width scaled")
    plt.axhline(0.9, color="#333333", linestyle="--", linewidth=1, label="90% nominal")
    plt.xticks(x_values, frame["strategy_name"], rotation=35, ha="right")
    plt.ylim(0, 1.1)
    plt.ylabel("Coverage / scaled width")
    plt.title("FP3 Conformal Coverage and Width")
    plt.legend()
    _finish_plot(plt)
    return True


def _plot_regime_conformal_coverage_by_bucket(plt: Any, table: pd.DataFrame) -> bool:
    frame = table[
        table["summary_scope"].eq("predicted_gap_bucket")
        & table["checkpoint"].eq("after_fp3")
        & table["strategy_type"].eq("deployable")
    ].copy()
    if frame.empty:
        return False
    pivot = frame.pivot_table(
        index="gap_bucket",
        columns="strategy_name",
        values="coverage",
        aggfunc="mean",
    )
    if pivot.empty:
        return False
    pivot = pivot.reindex(
        ["pole_contender", "close_midfield", "midfield", "backmarker_or_outlier"]
    ).dropna(how="all")
    x_values = list(range(len(pivot.index)))
    strategies = list(pivot.columns)
    width = 0.75 / max(len(strategies), 1)
    plt.figure(figsize=(9, 4.8))
    for index, strategy in enumerate(strategies):
        offsets = [value - 0.375 + width / 2 + index * width for value in x_values]
        plt.bar(offsets, pivot[strategy], width=width, label=strategy)
    plt.axhline(0.9, color="#333333", linestyle="--", linewidth=1)
    plt.xticks(x_values, pivot.index, rotation=30, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Coverage")
    plt.title("FP3 Coverage by Predicted Gap Bucket")
    plt.legend(fontsize=8)
    _finish_plot(plt)
    return True


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


def _guardrail_row_columns() -> list[str]:
    return [
        "policy_name",
        "policy_type",
        "season",
        "event",
        "event_slug",
        "fold_id",
        "checkpoint",
        "driver",
        "team",
        "actual_quali_gap_to_pole_sec",
        "static_predicted",
        "stabilized_predicted",
        "simulated_predicted",
        "static_selected_family",
        "static_selected_model_name",
        "static_selected_feature_group",
        "stabilized_selected_family",
        "stabilized_selected_model_name",
        "stabilized_selected_feature_group",
        "policy_action",
        "replaced_with_static",
        "replacement_reason",
    ]


def _guardrail_summary_columns() -> list[str]:
    return [
        "policy_name",
        "policy_type",
        "checkpoint",
        "rows",
        "mae_gap_sec",
        "rmse_gap_sec",
        "median_abs_error_gap_sec",
        "fp3_mae_gap_sec",
        "delta_vs_static_mae_sec",
        "delta_vs_current_stabilized_nested_mae_sec",
        "fp3_rows_replaced",
        "fp3_events_affected",
        "worst_fp3_event_mae",
        "best_fp3_event_mae",
    ]


def _guardrail_event_columns() -> list[str]:
    return [
        "policy_name",
        "policy_type",
        "season",
        "event",
        "fold_id",
        "rows",
        "static_mae_gap_sec",
        "stabilized_mae_gap_sec",
        "simulated_policy_mae_gap_sec",
        "delta_vs_static_sec",
        "delta_vs_stabilized_sec",
        "static_method",
        "stabilized_method",
        "policy_action",
        "replaced_with_static",
        "replacement_reason",
    ]


def _conformal_row_columns() -> list[str]:
    return [
        "strategy_name",
        "strategy_type",
        "season",
        "event",
        "fold_id",
        "checkpoint",
        "driver",
        "team",
        "selected_family",
        "selected_model_name",
        "selected_feature_group",
        "actual_quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec",
        "actual_gap_bucket",
        "predicted_gap_bucket",
        "residual_quantile_sec",
        "residual_count",
        "calibration_level",
        "prediction_interval_low_sec",
        "prediction_interval_high_sec",
        "interval_width_sec",
        "interval_available",
        "interval_contains_actual",
        "interval_miss",
    ]


def _conformal_summary_columns() -> list[str]:
    return [
        "strategy_name",
        "strategy_type",
        "summary_scope",
        "checkpoint",
        "bucket_type",
        "gap_bucket",
        "rows",
        "interval_availability_rate",
        "coverage",
        "mean_interval_width_sec",
        "median_interval_width_sec",
        "miss_count",
        "miss_rate",
        "mean_residual_quantile_sec",
    ]


def _conformal_event_columns() -> list[str]:
    return [
        "strategy_name",
        "strategy_type",
        "season",
        "event",
        "fold_id",
        "checkpoint",
        "rows",
        "interval_availability_rate",
        "coverage",
        "mean_interval_width_sec",
        "median_interval_width_sec",
        "miss_count",
        "miss_rate",
        "mean_residual_quantile_sec",
    ]


def _policy_type(policy_name: str) -> str:
    return "oracle" if policy_name.endswith("_oracle") else "deployable"


def _delta(value: object, baseline: object) -> float | None:
    numeric = _number_or_none(value)
    baseline_numeric = _number_or_none(baseline)
    if numeric is None or baseline_numeric is None:
        return None
    return numeric - baseline_numeric


def _checkpoint_mae(predictions: pd.DataFrame, checkpoint: str) -> float | None:
    if predictions.empty or not {"checkpoint", "quali_gap_to_pole_sec"} <= set(predictions.columns):
        return None
    rows = predictions[predictions["checkpoint"].astype(str).eq(checkpoint)]
    return _prediction_mae(rows)


def _mae_by_checkpoint(predictions: pd.DataFrame) -> dict[str, float | None]:
    if predictions.empty or "checkpoint" not in predictions:
        return {}
    return {
        str(checkpoint): _prediction_mae(rows)
        for checkpoint, rows in predictions.groupby("checkpoint", sort=False)
    }


def _prediction_mae(predictions: pd.DataFrame) -> float | None:
    required = {"quali_gap_to_pole_sec", "predicted_quali_gap_to_pole_sec"}
    if predictions.empty or not required <= set(predictions.columns):
        return None
    actual = pd.to_numeric(predictions["quali_gap_to_pole_sec"], errors="coerce")
    predicted = pd.to_numeric(predictions["predicted_quali_gap_to_pole_sec"], errors="coerce")
    errors = (predicted - actual).abs().dropna()
    return float(errors.mean()) if not errors.empty else None


def _close(value: object, other: object, tolerance: float = 1e-9) -> bool:
    numeric = _number_or_none(value)
    other_numeric = _number_or_none(other)
    return (
        numeric is not None
        and other_numeric is not None
        and abs(numeric - other_numeric) <= tolerance
    )


def _mean_or_none(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else None


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


def _bool_or_false(value: object) -> bool:
    if value is None or value is pd.NA:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if isinstance(value, int | float):
        if math.isnan(float(value)):
            return False
        return bool(value)
    return False


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


def _record(frame: pd.DataFrame) -> dict[str, object] | None:
    if frame.empty:
        return None
    row = frame.iloc[0].to_dict()
    return {key: _json_value(value) for key, value in row.items()}


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
