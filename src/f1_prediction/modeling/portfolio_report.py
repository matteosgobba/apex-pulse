"""Portfolio-ready reporting artifacts from saved backtest outputs."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.utils.paths import ensure_directory

CHECKPOINT_ORDER: tuple[str, ...] = ("after_fp1", "after_fp2", "after_fp3")
CHAMPION_MODES: tuple[str, ...] = (
    "static",
    "nested",
    "stabilized_nested",
    "stabilized_nested_guarded",
    "season_aware_nested_guarded",
)
CORE_CHAMPION_MODES: tuple[str, ...] = ("static", "nested", "stabilized_nested")
REQUIRED_PORTFOLIO_ARTIFACTS: tuple[str, ...] = (
    "champion_static_metrics.json",
    "champion_nested_metrics.json",
    "champion_stabilized_nested_metrics.json",
    "champion_static_predictions.parquet",
    "champion_nested_predictions.parquet",
    "champion_stabilized_nested_predictions.parquet",
    "champion_static_selection.parquet",
    "champion_nested_selection.parquet",
    "champion_stabilized_nested_selection.parquet",
    "backtest_report.json",
    "diagnostics_report.json",
    "event_error_summary.parquet",
    "driver_error_summary.parquet",
)


@dataclass(frozen=True)
class PortfolioReportSummary:
    """Paths and issue counts produced by portfolio report generation."""

    summary_path: Path
    model_card_path: Path
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    missing_artifacts: tuple[str, ...]
    generation_issues: tuple[str, ...]


def create_portfolio_report(config: DataConfig) -> PortfolioReportSummary:
    """Generate compact portfolio tables, figures, summary JSON, and model card."""
    metrics_dir = config.metrics_output_dir
    reports_dir = metrics_dir.parent
    figures_dir = reports_dir / "figures"
    ensure_directory(metrics_dir)
    ensure_directory(figures_dir)

    artifacts = _load_artifacts(metrics_dir)
    champion_summary = build_champion_summary_table(
        artifacts["champion_metrics"],
        artifacts["champion_predictions"],
    )
    selection_summary = build_champion_selection_summary_table(
        artifacts["champion_selection"],
    )
    interval_summary = build_champion_interval_summary_table(
        artifacts["champion_predictions"],
    )
    worst_events = build_worst_event_diagnostics_table(artifacts["event_error_summary"])
    worst_drivers = build_worst_driver_diagnostics_table(artifacts["driver_error_summary"])

    table_paths = (
        metrics_dir / "champion_summary_table.csv",
        metrics_dir / "champion_selection_summary_table.csv",
        metrics_dir / "champion_interval_summary_table.csv",
        metrics_dir / "worst_event_diagnostics_table.csv",
        metrics_dir / "worst_driver_diagnostics_table.csv",
    )
    champion_summary.to_csv(table_paths[0], index=False)
    selection_summary.to_csv(table_paths[1], index=False)
    interval_summary.to_csv(table_paths[2], index=False)
    worst_events.to_csv(table_paths[3], index=False)
    worst_drivers.to_csv(table_paths[4], index=False)

    figure_paths, figure_issues = generate_portfolio_figures(
        figures_dir=figures_dir,
        champion_summary=champion_summary,
        interval_summary=interval_summary,
        selection_summary=selection_summary,
        worst_events=worst_events,
    )
    summary_payload = build_portfolio_summary_payload(
        artifacts,
        champion_summary=champion_summary,
        interval_summary=interval_summary,
        selection_summary=selection_summary,
        generated_figures=figure_paths,
        generation_issues=figure_issues,
    )
    summary_path = metrics_dir / "portfolio_summary.json"
    _write_json(summary_path, summary_payload)
    model_card_path = reports_dir / "model_card.md"
    model_card_path.write_text(
        build_model_card(summary_payload, champion_summary, interval_summary),
        encoding="utf-8",
    )
    return PortfolioReportSummary(
        summary_path=summary_path,
        model_card_path=model_card_path,
        table_paths=table_paths,
        figure_paths=tuple(figure_paths),
        missing_artifacts=tuple(summary_payload["missing_artifacts"]),
        generation_issues=tuple(summary_payload["generation_issues"]),
    )


def build_champion_summary_table(
    champion_metrics: dict[str, dict[str, Any]],
    champion_predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create checkpoint-level champion metric rows for each available mode."""
    rows: list[dict[str, object]] = []
    for mode in CHAMPION_MODES:
        metrics_payload = champion_metrics.get(mode)
        if not metrics_payload:
            continue
        predictions = champion_predictions.get(mode)
        prediction_rows = _rows_by_checkpoint(predictions)
        metrics_by_checkpoint = metrics_payload.get("metrics_by_checkpoint", {})
        for checkpoint in _ordered_checkpoints(metrics_by_checkpoint):
            values = metrics_by_checkpoint.get(checkpoint, {})
            best_baseline = metrics_payload.get("best_baseline_by_checkpoint", {}).get(
                checkpoint, {}
            )
            rows.append(
                {
                    "selection_mode": mode,
                    "checkpoint": checkpoint,
                    "rows": prediction_rows.get(checkpoint),
                    "mae_gap_sec": _number_or_none(values.get("mae_gap_sec")),
                    "rmse_gap_sec": _number_or_none(values.get("rmse_gap_sec")),
                    "median_abs_error_gap_sec": _number_or_none(
                        values.get("median_abs_error_gap_sec")
                    ),
                    "mean_position_error": _number_or_none(values.get("mean_abs_position_error")),
                    "champion_vs_best_baseline_delta_mae": _number_or_none(
                        metrics_payload.get("champion_vs_best_baseline_delta_mae", {}).get(
                            checkpoint
                        )
                    ),
                    "best_baseline_mae_gap_sec": _number_or_none(best_baseline.get("mae_gap_sec")),
                    "selected_method_summary": _selected_method_summary(
                        predictions,
                        metrics_payload,
                        checkpoint,
                    ),
                }
            )
    return pd.DataFrame(rows, columns=_champion_summary_columns())


def build_champion_selection_summary_table(
    champion_selection: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize fixed and data-driven champion method selections."""
    rows: list[dict[str, object]] = []
    for mode in CHAMPION_MODES:
        selection = champion_selection.get(mode)
        if selection is None or selection.empty:
            continue
        frame = selection.copy()
        for column in ("selected_feature_group", "fallback_reason"):
            if column not in frame:
                frame[column] = pd.NA
        if "fallback_used" not in frame:
            frame["fallback_used"] = False
        group_columns = [
            "checkpoint",
            "selected_family",
            "selected_model_name",
            "selected_feature_group",
        ]
        checkpoint_counts = frame.groupby("checkpoint", dropna=False).size()
        for keys, group in frame.groupby(group_columns, dropna=False, sort=False):
            checkpoint, family, model_name, feature_group = keys
            total = int(checkpoint_counts.loc[checkpoint])
            fallback = group["fallback_used"].fillna(False).astype(bool)
            rows.append(
                {
                    "selection_mode": mode,
                    "checkpoint": checkpoint,
                    "selected_family": family,
                    "selected_model_name": model_name,
                    "selected_feature_group": _display_optional(feature_group),
                    "fold_count": int(len(group)),
                    "selection_share": float(len(group) / total) if total else None,
                    "fallback_rate": float(fallback.mean()) if len(group) else None,
                    "main_fallback_reason": _most_common_non_null(group["fallback_reason"]),
                }
            )
    return pd.DataFrame(rows, columns=_selection_summary_columns())


def build_champion_interval_summary_table(
    champion_predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute interval availability, coverage, and width by mode/checkpoint."""
    rows: list[dict[str, object]] = []
    for mode in CHAMPION_MODES:
        predictions = champion_predictions.get(mode)
        if predictions is None or predictions.empty:
            continue
        required = {
            "checkpoint",
            "uncertainty_method",
            "prediction_interval_low_sec",
            "prediction_interval_high_sec",
        }
        if not required <= set(predictions.columns):
            continue
        frame = predictions.copy()
        for column in (
            "interval_contains_actual",
            "residual_quantile_sec",
            "residual_std_sec",
        ):
            if column not in frame:
                frame[column] = pd.NA
        group_columns = ["checkpoint", "uncertainty_method"]
        for keys, group in frame.groupby(group_columns, dropna=False, sort=False):
            checkpoint, uncertainty_method = keys
            available = group[
                group["prediction_interval_low_sec"].notna()
                & group["prediction_interval_high_sec"].notna()
            ].copy()
            widths = (
                available["prediction_interval_high_sec"].astype(float)
                - available["prediction_interval_low_sec"].astype(float)
                if not available.empty
                else pd.Series(dtype=float)
            )
            contains = available["interval_contains_actual"].dropna()
            rows.append(
                {
                    "selection_mode": mode,
                    "checkpoint": checkpoint,
                    "uncertainty_method": uncertainty_method,
                    "interval_availability_rate": (
                        float(len(available) / len(group)) if len(group) else None
                    ),
                    "interval_coverage": (
                        float(contains.astype(bool).mean()) if not contains.empty else None
                    ),
                    "mean_interval_width_sec": (
                        _number_or_none(widths.mean()) if not widths.empty else None
                    ),
                    "median_interval_width_sec": (
                        _number_or_none(widths.median()) if not widths.empty else None
                    ),
                    "mean_residual_quantile_sec": _number_or_none(
                        available["residual_quantile_sec"].mean()
                    ),
                    "mean_residual_std_sec": _number_or_none(available["residual_std_sec"].mean()),
                }
            )
    return pd.DataFrame(rows, columns=_interval_summary_columns())


def build_worst_event_diagnostics_table(event_summary: pd.DataFrame | None) -> pd.DataFrame:
    """Create a compact hardest-event table from diagnostics artifacts."""
    columns = [
        "season",
        "event",
        "checkpoint",
        "model_or_method",
        "mae_gap_sec",
        "position_error",
        "rows",
        "notes_if_available",
    ]
    if event_summary is None or event_summary.empty:
        return pd.DataFrame(columns=columns)
    frame = event_summary.copy()
    if "model_name" not in frame and "model_or_method" in frame:
        frame["model_name"] = frame["model_or_method"]
    model_or_method = _source_method_label(frame)
    result = pd.DataFrame(
        {
            "season": frame.get("season"),
            "event": frame.get("event", frame.get("event_slug")),
            "checkpoint": frame.get("checkpoint"),
            "model_or_method": model_or_method,
            "mae_gap_sec": frame.get("mae_gap_sec"),
            "position_error": frame.get("mean_abs_position_error"),
            "rows": frame.get("n_rows"),
            "notes_if_available": frame.get("notes_if_available", ""),
        }
    )
    return result.sort_values("mae_gap_sec", ascending=False, na_position="last").head(30)


def build_worst_driver_diagnostics_table(driver_summary: pd.DataFrame | None) -> pd.DataFrame:
    """Create a compact hardest-driver table from diagnostics artifacts."""
    columns = [
        "driver_key",
        "driver",
        "team_key",
        "checkpoint",
        "model_or_method",
        "mae_gap_sec",
        "position_error",
        "rows",
        "notes_if_available",
    ]
    if driver_summary is None or driver_summary.empty:
        return pd.DataFrame(columns=columns)
    frame = driver_summary.copy()
    if "model_name" not in frame and "model_or_method" in frame:
        frame["model_name"] = frame["model_or_method"]
    model_or_method = _source_method_label(frame)
    result = pd.DataFrame(
        {
            "driver_key": frame.get("driver_key", ""),
            "driver": frame.get("driver"),
            "team_key": frame.get("team_key", ""),
            "checkpoint": frame.get("checkpoint"),
            "model_or_method": model_or_method,
            "mae_gap_sec": frame.get("mae_gap_sec"),
            "position_error": frame.get("mean_abs_position_error"),
            "rows": frame.get("n_rows"),
            "notes_if_available": frame.get("notes_if_available", ""),
        }
    )
    return result.sort_values("mae_gap_sec", ascending=False, na_position="last").head(30)


def build_portfolio_summary_payload(
    artifacts: dict[str, Any],
    *,
    champion_summary: pd.DataFrame,
    interval_summary: pd.DataFrame,
    selection_summary: pd.DataFrame,
    generated_figures: list[Path],
    generation_issues: list[str],
) -> dict[str, object]:
    """Build the high-level JSON summary used by the model card and README workflow."""
    champion_metrics = artifacts["champion_metrics"]
    backtest_report = artifacts["backtest_report"] or {}
    temporal_weighting_summary = artifacts["temporal_weighting_summary"] or {}
    season_aware_summary = artifacts["season_aware_validation_summary"] or {}
    season_aware_champion_summary = artifacts["season_aware_champion_summary"] or {}
    season_aware_candidate_audit = artifacts["season_aware_candidate_audit_summary"] or {}
    missing_artifacts = artifacts["missing_artifacts"]
    modes_available = [mode for mode in CHAMPION_MODES if mode in champion_metrics]
    best_by_checkpoint = _best_champion_mode_by_checkpoint(champion_metrics, backtest_report)
    best_overall = _best_champion_mode_overall(champion_metrics, backtest_report)
    key_results = _key_results(champion_metrics)
    dataset_summary = _dataset_summary_if_available(
        artifacts["dataset_build_report"],
        artifacts["dataset_quality_report"],
        backtest_report,
    )
    return {
        "project_status": _project_status(modes_available),
        "dataset_summary_if_available": dataset_summary,
        "champion_modes_available": modes_available,
        "best_champion_mode_by_checkpoint": best_by_checkpoint,
        "best_champion_mode_overall": best_overall,
        "key_results": key_results,
        "main_takeaways": _main_takeaways(
            champion_metrics,
            best_overall,
            temporal_weighting_summary,
            season_aware_summary,
            season_aware_candidate_audit,
        ),
        "limitations": _limitations(),
        "recommended_next_milestone": (
            "Milestone 23: decide whether the season-aware weighted FP3 candidate should enter "
            "future nested champion evaluation while keeping current champion defaults unchanged."
        ),
        "temporal_weighting_if_available": _temporal_weighting_portfolio_summary(
            temporal_weighting_summary
        ),
        "season_aware_validation_if_available": _season_aware_portfolio_summary(
            season_aware_summary
        ),
        "season_aware_champion_if_available": _season_aware_champion_portfolio_summary(
            season_aware_champion_summary
        ),
        "season_aware_candidate_audit_if_available": (
            _season_aware_candidate_audit_portfolio_summary(season_aware_candidate_audit)
        ),
        "generated_at": _utc_now(),
        "generated_outputs": {
            "metrics": [
                "reports/metrics/portfolio_summary.json",
                "reports/metrics/champion_summary_table.csv",
                "reports/metrics/champion_interval_summary_table.csv",
                "reports/metrics/champion_selection_summary_table.csv",
                "reports/metrics/worst_event_diagnostics_table.csv",
                "reports/metrics/worst_driver_diagnostics_table.csv",
            ],
            "figures": [_relative_report_path(path) for path in generated_figures],
            "model_card": "reports/model_card.md",
        },
        "missing_artifacts": missing_artifacts,
        "generation_issues": generation_issues,
        "table_row_counts": {
            "champion_summary_table": int(len(champion_summary)),
            "champion_interval_summary_table": int(len(interval_summary)),
            "champion_selection_summary_table": int(len(selection_summary)),
        },
    }


def generate_portfolio_figures(
    *,
    figures_dir: Path,
    champion_summary: pd.DataFrame,
    interval_summary: pd.DataFrame,
    selection_summary: pd.DataFrame,
    worst_events: pd.DataFrame,
) -> tuple[list[Path], list[str]]:
    """Write simple matplotlib figures and return paths plus non-fatal issues."""
    try:
        ensure_directory(figures_dir / ".matplotlib")
        ensure_directory(figures_dir / ".cache")
        os.environ.setdefault("MPLCONFIGDIR", str(figures_dir / ".matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(figures_dir / ".cache"))
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional runtime install
        return [], [f"matplotlib_unavailable: {exc}"]

    paths: list[Path] = []
    issues: list[str] = []
    figure_specs = (
        (
            "champion_mae_by_checkpoint.png",
            lambda path: _plot_grouped_bars(
                plt,
                champion_summary,
                path,
                value_column="mae_gap_sec",
                title="Champion MAE by checkpoint",
                ylabel="MAE gap (sec)",
            ),
        ),
        (
            "champion_vs_baseline_delta_by_checkpoint.png",
            lambda path: _plot_grouped_bars(
                plt,
                champion_summary,
                path,
                value_column="champion_vs_best_baseline_delta_mae",
                title="Champion delta vs best baseline",
                ylabel="Delta MAE (sec)",
            ),
        ),
        (
            "champion_interval_coverage_by_checkpoint.png",
            lambda path: _plot_grouped_bars(
                plt,
                interval_summary,
                path,
                value_column="interval_coverage",
                title="Champion interval coverage",
                ylabel="Coverage",
            ),
        ),
        (
            "champion_interval_width_by_checkpoint.png",
            lambda path: _plot_grouped_bars(
                plt,
                interval_summary,
                path,
                value_column="mean_interval_width_sec",
                title="Mean champion interval width",
                ylabel="Mean width (sec)",
            ),
        ),
        (
            "champion_selection_share_by_checkpoint.png",
            lambda path: _plot_selection_share(plt, selection_summary, path),
        ),
        (
            "worst_events_mae.png",
            lambda path: _plot_worst_events(plt, worst_events, path),
        ),
    )
    for filename, writer in figure_specs:
        path = figures_dir / filename
        try:
            if writer(path):
                paths.append(path)
        except Exception as exc:  # non-fatal by design for optional figures
            issues.append(f"{filename}: {exc}")
    return paths, issues


def build_model_card(
    summary: dict[str, object],
    champion_summary: pd.DataFrame,
    interval_summary: pd.DataFrame,
) -> str:
    """Render a concise Markdown model card from available portfolio artifacts."""
    lines = [
        "# Formula 1 Qualifying Prediction Model Card",
        "",
        "## Project overview",
        "This project predicts Formula 1 qualifying performance from historical free-practice "
        "data and leakage-safe historical context. It is an ML portfolio project, not a generic "
        "dashboard.",
        "",
        "## Prediction task",
        "The current task predicts qualifying gap to pole, qualifying position, and Q3 reach "
        "status at after-FP1, after-FP2, and after-FP3 checkpoints.",
        "",
        "## Data sources",
        "The project uses FastF1 public historical data only. It does not use private team data, "
        "paid APIs, fuel loads, engine modes, setup data, or internal energy-deployment signals.",
        "",
        "## Current dataset",
        _dataset_card_text(summary.get("dataset_summary_if_available")),
        "",
        "## Evaluation protocol",
        "The preferred evaluation is walk-forward backtesting: each test event is predicted using "
        "only earlier events for training or champion selection.",
        "",
        "## Temporal weighting",
        _temporal_weighting_card_text(summary.get("temporal_weighting_if_available")),
        "",
        "## Season-aware validation",
        _season_aware_card_text(summary.get("season_aware_validation_if_available")),
        "",
        "## Season-aware candidate audit",
        _season_aware_candidate_audit_card_text(
            summary.get("season_aware_candidate_audit_if_available")
        ),
        "",
        "## Baselines",
        "The report compares champion policies against practice-lap baselines, including robust "
        "baselines that fall back from weak or extreme latest-session signals.",
        "",
        "## Model families",
        "The current model families are Ridge regression, Random Forest, scikit-learn histogram "
        "gradient boosting, feature-ablation Random Forest variants, and non-ML practice "
        "baselines.",
        "",
        "## Champion policy",
        _champion_card_text(summary),
        "",
        "## Results summary",
        _results_card_table(champion_summary),
        "",
        "## Uncertainty estimates",
        _interval_card_text(interval_summary),
        "",
        "## Key limitations",
        "\n".join(f"- {item}" for item in summary.get("limitations", [])),
        "",
        "## Recommended next steps",
        f"- {summary.get('recommended_next_milestone')}",
        "",
    ]
    return "\n".join(lines)


def _load_artifacts(metrics_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_PORTFOLIO_ARTIFACTS if not (metrics_dir / name).is_file()]
    champion_metrics = {
        mode: payload
        for mode in CHAMPION_MODES
        if (payload := _read_json_if_exists(metrics_dir / f"champion_{mode}_metrics.json"))
        is not None
    }
    champion_predictions = {
        mode: frame
        for mode in CHAMPION_MODES
        if (frame := _read_parquet_if_exists(metrics_dir / f"champion_{mode}_predictions.parquet"))
        is not None
    }
    champion_selection = {
        mode: frame
        for mode in CHAMPION_MODES
        if (frame := _read_parquet_if_exists(metrics_dir / f"champion_{mode}_selection.parquet"))
        is not None
    }
    return {
        "champion_metrics": champion_metrics,
        "champion_predictions": champion_predictions,
        "champion_selection": champion_selection,
        "backtest_report": _read_json_if_exists(metrics_dir / "backtest_report.json"),
        "diagnostics_report": _read_json_if_exists(metrics_dir / "diagnostics_report.json"),
        "dataset_build_report": _read_json_if_exists(metrics_dir / "dataset_build_report.json"),
        "dataset_quality_report": _read_json_if_exists(metrics_dir / "dataset_quality_report.json"),
        "temporal_weighting_summary": _read_json_if_exists(
            metrics_dir / "temporal_weighting_summary.json"
        ),
        "season_aware_validation_summary": _read_json_if_exists(
            metrics_dir / "season_aware_validation_summary.json"
        ),
        "season_aware_champion_summary": _read_json_if_exists(
            metrics_dir / "season_aware_champion_summary.json"
        ),
        "season_aware_candidate_audit_summary": _read_json_if_exists(
            metrics_dir / "season_aware_candidate_audit_summary.json"
        ),
        "event_error_summary": _read_parquet_if_exists(metrics_dir / "event_error_summary.parquet"),
        "driver_error_summary": _read_parquet_if_exists(
            metrics_dir / "driver_error_summary.parquet"
        ),
        "missing_artifacts": missing,
    }


def _rows_by_checkpoint(predictions: pd.DataFrame | None) -> dict[str, int]:
    if predictions is None or predictions.empty or "checkpoint" not in predictions:
        return {}
    counts = predictions["checkpoint"].astype(str).value_counts()
    return {str(checkpoint): int(rows) for checkpoint, rows in counts.items()}


def _selected_method_summary(
    predictions: pd.DataFrame | None,
    metrics_payload: dict[str, Any],
    checkpoint: str,
) -> str:
    selected_columns = {
        "checkpoint",
        "selected_family",
        "selected_model_name",
        "selected_feature_group",
    }
    if predictions is not None and selected_columns <= set(predictions.columns):
        rows = predictions[predictions["checkpoint"].astype(str).eq(checkpoint)].copy()
        if not rows.empty:
            rows["method_label"] = rows.apply(
                lambda row: _method_label(
                    row["selected_family"],
                    row["selected_model_name"],
                    row["selected_feature_group"],
                ),
                axis=1,
            )
            shares = rows["method_label"].value_counts(normalize=True)
            if len(shares) == 1:
                return str(shares.index[0])
            return "; ".join(f"{method} ({share:.0%})" for method, share in shares.head(3).items())
    best_single = metrics_payload.get("best_single_family_by_checkpoint", {}).get(checkpoint, {})
    if not best_single:
        return ""
    family = best_single.get("family")
    model = best_single.get("model_name")
    feature_group = best_single.get("feature_group")
    parts = [str(value) for value in (family, model, feature_group) if value]
    return " / ".join(parts)


def _source_method_label(frame: pd.DataFrame) -> pd.Series:
    method = frame["model_name"].astype(str)
    if "prediction_source" not in frame:
        return method
    return frame["prediction_source"].astype(str) + "/" + method


def _key_results(champion_metrics: dict[str, dict[str, Any]]) -> dict[str, object]:
    results: dict[str, object] = {}
    for mode in CHAMPION_MODES:
        payload = champion_metrics.get(mode)
        if not payload:
            continue
        prefix = f"{mode}_champion"
        metrics_by_checkpoint = payload.get("metrics_by_checkpoint", {})
        results[f"{prefix}_mae_by_checkpoint"] = {
            checkpoint: _number_or_none(values.get("mae_gap_sec"))
            for checkpoint, values in metrics_by_checkpoint.items()
        }
        results[f"{prefix}_champion_vs_best_baseline_delta_mae_by_checkpoint"] = {
            checkpoint: _number_or_none(value)
            for checkpoint, value in payload.get("champion_vs_best_baseline_delta_mae", {}).items()
        }
        if mode == "stabilized_nested":
            results["stabilized_nested_interval_availability_by_checkpoint"] = {
                checkpoint: _number_or_none(values.get("interval_availability_rate"))
                for checkpoint, values in metrics_by_checkpoint.items()
            }
            results["stabilized_nested_interval_coverage_by_checkpoint"] = {
                checkpoint: _number_or_none(values.get("interval_coverage"))
                for checkpoint, values in metrics_by_checkpoint.items()
            }
            results["stabilized_nested_mean_interval_width_by_checkpoint"] = {
                checkpoint: _number_or_none(values.get("mean_interval_width_sec"))
                for checkpoint, values in metrics_by_checkpoint.items()
            }
    return results


def _best_champion_mode_by_checkpoint(
    champion_metrics: dict[str, dict[str, Any]],
    backtest_report: dict[str, Any],
) -> dict[str, object]:
    if backtest_report.get("best_champion_selection_mode_by_checkpoint"):
        return dict(backtest_report["best_champion_selection_mode_by_checkpoint"])
    best: dict[str, object] = {}
    checkpoints = {
        checkpoint
        for payload in champion_metrics.values()
        for checkpoint in payload.get("metrics_by_checkpoint", {})
    }
    for checkpoint in _ordered_checkpoints({checkpoint: {} for checkpoint in checkpoints}):
        candidates: list[tuple[str, float, dict[str, Any]]] = []
        for mode, payload in champion_metrics.items():
            values = payload.get("metrics_by_checkpoint", {}).get(checkpoint, {})
            mae = _number_or_none(values.get("mae_gap_sec"))
            if mae is not None:
                candidates.append((mode, mae, values))
        if candidates:
            mode, mae, values = min(candidates, key=lambda item: item[1])
            best[checkpoint] = {
                "selection_mode": mode,
                "mae_gap_sec": mae,
                "mean_abs_position_error": _number_or_none(values.get("mean_abs_position_error")),
            }
    return best


def _best_champion_mode_overall(
    champion_metrics: dict[str, dict[str, Any]],
    backtest_report: dict[str, Any],
) -> dict[str, object] | None:
    if backtest_report.get("best_champion_selection_mode_overall"):
        return dict(backtest_report["best_champion_selection_mode_overall"])
    candidates: list[tuple[str, float]] = []
    for mode, payload in champion_metrics.items():
        values = [
            float(metrics["mae_gap_sec"])
            for metrics in payload.get("metrics_by_checkpoint", {}).values()
            if metrics.get("mae_gap_sec") is not None
        ]
        if values:
            candidates.append((mode, sum(values) / len(values)))
    if not candidates:
        return None
    mode, mean_mae = min(candidates, key=lambda item: item[1])
    return {"selection_mode": mode, "mean_mae_gap_sec": mean_mae}


def _dataset_summary_if_available(
    dataset_build: dict[str, Any] | None,
    dataset_quality: dict[str, Any] | None,
    backtest_report: dict[str, Any],
) -> dict[str, object] | None:
    source = dataset_build or dataset_quality or backtest_report
    if not source:
        return None
    return {
        "rows": source.get("n_rows", source.get("dataset_rows")),
        "columns": source.get("n_columns"),
        "events": source.get("n_events_successful", source.get("n_events")),
        "drivers": source.get("n_drivers"),
        "teams": source.get("n_teams"),
        "seasons": source.get("requested_seasons", source.get("seasons")),
        "checkpoints": source.get("checkpoints"),
    }


def _project_status(modes_available: list[str]) -> str:
    if set(CORE_CHAMPION_MODES) <= set(modes_available):
        return (
            "Milestone 16 portfolio report generated from static, nested, and stabilized "
            "champion artifacts."
        )
    if modes_available:
        return "Milestone 16 portfolio report generated from partial champion artifacts."
    return (
        "Milestone 16 portfolio report generated without champion artifacts; outputs are partial."
    )


def _main_takeaways(
    champion_metrics: dict[str, dict[str, Any]],
    best_overall: dict[str, object] | None,
    temporal_weighting_summary: dict[str, Any] | None = None,
    season_aware_summary: dict[str, Any] | None = None,
    season_aware_candidate_audit: dict[str, Any] | None = None,
) -> list[str]:
    takeaways: list[str] = []
    static = champion_metrics.get("static", {})
    static_deltas = static.get("champion_vs_best_baseline_delta_mae", {})
    if static_deltas.get("after_fp1", 0) >= 0 and static_deltas.get("after_fp2", 0) >= 0:
        takeaways.append("FP1 and FP2 remain baseline-favored.")
    if static_deltas.get("after_fp3") is not None and static_deltas["after_fp3"] < 0:
        takeaways.append("FP3 is the clearest checkpoint where ML currently adds value.")
    if best_overall and best_overall.get("selection_mode") == "static":
        takeaways.append("Static champion remains the strongest current policy.")
    stabilized = champion_metrics.get("stabilized_nested", {})
    nested = champion_metrics.get("nested", {})
    if stabilized and nested:
        takeaways.append(
            "Stabilized nested reduces switching instability but remains conservative."
        )
    fp3_coverage = (
        stabilized.get("metrics_by_checkpoint", {}).get("after_fp3", {}).get("interval_coverage")
    )
    if fp3_coverage is not None and float(fp3_coverage) < 0.9:
        takeaways.append("Conformal intervals are leakage-safe but broad, with FP3 undercoverage.")
    if temporal_weighting_summary:
        takeaways.append(
            "Season-aware weighting is evaluated as an opt-in training policy; current champion "
            "defaults remain unchanged pending broader validation."
        )
    if season_aware_summary:
        recommendation = season_aware_summary.get(
            "season_aware_promotion_recommendation",
            "insufficient_evidence",
        )
        takeaways.append(
            "Season-aware candidate validation is retrospective; champion defaults remain "
            f"unchanged with recommendation `{recommendation}`."
        )
    if season_aware_candidate_audit:
        recommendation = season_aware_candidate_audit.get("recommendation", "retain_static_policy")
        takeaways.append(
            "Season-aware candidate audit is diagnostic; static champion remains the current "
            f"policy with recommendation `{recommendation}`."
        )
    if not takeaways:
        takeaways.append(
            "Portfolio outputs are partial because key champion artifacts are missing."
        )
    return takeaways


def _temporal_weighting_portfolio_summary(
    summary: dict[str, Any] | None,
) -> dict[str, object] | None:
    if not summary:
        return None
    return {
        "temporal_weighting_policies_available": summary.get(
            "temporal_weighting_policies_available", {}
        ),
        "best_temporal_weighting_policy_by_checkpoint": summary.get(
            "best_temporal_weighting_policy_by_checkpoint", {}
        ),
        "temporal_weighting_vs_uniform_delta_by_checkpoint": summary.get(
            "temporal_weighting_vs_uniform_delta_by_checkpoint", {}
        ),
        "main_findings": summary.get("main_findings", []),
    }


def _season_aware_portfolio_summary(
    summary: dict[str, Any] | None,
) -> dict[str, object] | None:
    if not summary:
        return None
    return {
        "season_aware_validation_available": summary.get(
            "season_aware_validation_available",
            False,
        ),
        "season_aware_fp3_candidate_summary": summary.get(
            "season_aware_fp3_candidate_summary",
            {},
        ),
        "season_aware_best_fixed_candidate": summary.get(
            "season_aware_best_fixed_candidate",
            {},
        ),
        "season_aware_promotion_recommendation": summary.get(
            "season_aware_promotion_recommendation",
            "insufficient_evidence",
        ),
        "bootstrap_robustness": summary.get("bootstrap_robustness", {}),
        "main_findings": summary.get("main_findings", []),
    }


def _season_aware_champion_portfolio_summary(
    summary: dict[str, Any] | None,
) -> dict[str, object] | None:
    if not summary:
        return None
    return {
        "season_aware_champion_available": True,
        "fp3_summary": summary.get("fp3_summary", {}),
        "regime_summary": summary.get("regime_summary", []),
        "bootstrap_ci": summary.get("bootstrap_ci", {}),
        "promotion_recommendation": summary.get(
            "promotion_recommendation",
            "retain_static_policy",
        ),
        "main_findings": summary.get("main_findings", []),
    }


def _season_aware_candidate_audit_portfolio_summary(
    summary: dict[str, Any] | None,
) -> dict[str, object] | None:
    if not summary:
        return None
    return {
        "season_aware_candidate_audit_available": summary.get("status") != "missing_inputs",
        "candidate_availability": summary.get("candidate_availability", {}),
        "artifact_alignment_summary": summary.get("artifact_alignment_summary", {}),
        "live_gate_summary": summary.get("live_gate_summary", {}),
        "live_audit_metric_consistency_rate": summary.get("live_audit_metric_consistency_rate"),
        "live_audit_selection_consistency_rate": summary.get(
            "live_audit_selection_consistency_rate"
        ),
        "comparator_scope_description": summary.get("comparator_scope_description"),
        "sensitivity_analysis_summary": summary.get("sensitivity_analysis_summary", {}),
        "recommendation": summary.get("recommendation", "retain_static_policy"),
        "main_findings": summary.get("main_findings", []),
    }


def _limitations() -> list[str]:
    return [
        "FastF1 public historical data only; no private team data or paid APIs.",
        "Current targets simplify qualifying classification and do not model penalties.",
        "Dataset coverage is limited to conventional FP1/FP2/FP3 weekends already built locally.",
        "FP1 and FP2 practice signals remain noisy and baseline-favored.",
        "Uncertainty intervals are diagnostic, broad, and not production-calibrated guarantees.",
        "Telemetry features, ranking-specific models, Q3 probability calibration, and dashboards "
        "are out of scope for this milestone.",
    ]


def _plot_grouped_bars(
    plt: Any,
    frame: pd.DataFrame,
    path: Path,
    *,
    value_column: str,
    title: str,
    ylabel: str,
) -> bool:
    required = {"selection_mode", "checkpoint", value_column}
    if frame.empty or not required <= set(frame.columns):
        return False
    pivot = (
        frame.dropna(subset=[value_column])
        .pivot_table(
            index="checkpoint",
            columns="selection_mode",
            values=value_column,
            aggfunc="mean",
        )
        .reindex(CHECKPOINT_ORDER)
    )
    pivot = pivot.dropna(how="all")
    if pivot.empty:
        return False
    ax = pivot.plot(kind="bar", figsize=(8, 4.5), width=0.78)
    ax.set_title(title)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.legend(title="Selection mode")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _plot_selection_share(plt: Any, selection_summary: pd.DataFrame, path: Path) -> bool:
    if selection_summary.empty:
        return False
    frame = selection_summary[selection_summary["selection_mode"].eq("stabilized_nested")].copy()
    if frame.empty:
        return False
    frame["method"] = frame.apply(
        lambda row: _method_label(
            row["selected_family"],
            row["selected_model_name"],
            row["selected_feature_group"],
        ),
        axis=1,
    )
    pivot = (
        frame.pivot_table(
            index="checkpoint",
            columns="method",
            values="selection_share",
            aggfunc="sum",
        )
        .reindex(CHECKPOINT_ORDER)
        .dropna(how="all")
    )
    if pivot.empty:
        return False
    ax = pivot.plot(kind="bar", stacked=True, figsize=(8, 4.5), width=0.78)
    ax.set_title("Stabilized nested selection share")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Selection share")
    ax.set_ylim(0, 1)
    ax.legend(title="Selected method", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _plot_worst_events(plt: Any, worst_events: pd.DataFrame, path: Path) -> bool:
    if worst_events.empty or "mae_gap_sec" not in worst_events:
        return False
    frame = worst_events.dropna(subset=["mae_gap_sec"]).head(12).copy()
    if frame.empty:
        return False
    frame["label"] = (
        frame["season"].astype(str)
        + " "
        + frame["event"].astype(str)
        + " "
        + frame["checkpoint"].astype(str)
    )
    ax = frame.sort_values("mae_gap_sec").plot.barh(
        x="label",
        y="mae_gap_sec",
        figsize=(8, 5),
        legend=False,
        color="#4c78a8",
    )
    ax.set_title("Worst event/checkpoint MAE")
    ax.set_xlabel("MAE gap (sec)")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _dataset_card_text(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "Current dataset summary was unavailable when this model card was generated."
    parts = []
    for label, key in (
        ("Rows", "rows"),
        ("Columns", "columns"),
        ("Events", "events"),
        ("Drivers", "drivers"),
        ("Teams", "teams"),
    ):
        if value.get(key) is not None:
            parts.append(f"{label}: {value[key]}")
    return "; ".join(parts) + "." if parts else "Dataset summary fields were unavailable."


def _champion_card_text(summary: dict[str, object]) -> str:
    best = summary.get("best_champion_mode_overall")
    if isinstance(best, dict) and best.get("selection_mode"):
        return (
            "Champion policies are evaluated in static, nested, stabilized nested, "
            "opt-in guarded stabilized, and opt-in season-aware guarded modes when the "
            "corresponding artifacts exist. The season-aware mode is gated by cold-start and "
            "prior-history checks and does not replace the current default. "
            f"The current best overall mode is `{best['selection_mode']}`."
        )
    return "Champion policies are evaluated when their saved artifacts are available."


def _temporal_weighting_card_text(value: object) -> str:
    intro = (
        "Temporal weighting is evaluated as an opt-in training policy for future live/current-"
        "season use. It emphasizes same-season evidence while preserving leakage-safe "
        "walk-forward splits, and it does not change current champion defaults."
    )
    if not isinstance(value, dict) or not value:
        return intro + " No temporal weighting comparison report was available."
    policies = value.get("temporal_weighting_policies_available", {})
    if not policies:
        return intro + " No temporal weighting policies had saved comparison artifacts."
    return (
        intro
        + " Available saved policy comparisons: "
        + ", ".join(f"{source}: {', '.join(names)}" for source, names in policies.items())
        + "."
    )


def _season_aware_card_text(value: object) -> str:
    intro = (
        "Season-aware validation directly compares the fixed FP3 static Random Forest "
        "candidate with temporally weighted candidates on aligned walk-forward rows. These "
        "are retrospective candidate checks and do not promote a new champion default."
    )
    if not isinstance(value, dict) or not value:
        return intro + " No season-aware validation report was available."
    recommendation = value.get("season_aware_promotion_recommendation", "insufficient_evidence")
    summary = value.get("season_aware_fp3_candidate_summary", {})
    if isinstance(summary, dict) and summary:
        candidate_mae = summary.get("candidate_mae_gap_sec")
        static_mae = summary.get("static_mae_gap_sec")
        return (
            intro
            + f" Fixed FP3 candidate MAE: {_format_metric(candidate_mae)}; "
            + f"static MAE: {_format_metric(static_mae)}; "
            + f"recommendation: `{recommendation}`."
        )
    return intro + f" Recommendation: `{recommendation}`."


def _season_aware_candidate_audit_card_text(value: object) -> str:
    intro = (
        "The candidate audit checks whether the season-aware weighted FP3 Random Forest path "
        "has valid artifacts, aligned prediction rows, prior-only history, and live gate "
        "evidence. Sensitivity results are retrospective diagnostics, not deployed policy."
    )
    if not isinstance(value, dict) or not value:
        return intro + " No season-aware candidate audit was available."
    recommendation = value.get("recommendation", "retain_static_policy")
    live = value.get("live_gate_summary", {})
    if isinstance(live, dict) and live:
        selection_rate = live.get("weighted_candidate_selection_rate")
        folds = live.get("folds_evaluated")
        consistency = value.get("live_audit_metric_consistency_rate")
        return (
            intro
            + f" Audited FP3 folds: {folds}; live candidate selection rate: "
            + f"{_format_metric(selection_rate)}; comparator consistency: "
            + f"{_format_metric(consistency)}; recommendation: `{recommendation}`."
        )
    return intro + f" Recommendation: `{recommendation}`."


def _results_card_table(champion_summary: pd.DataFrame) -> str:
    if champion_summary.empty:
        return "Champion summary metrics were unavailable."
    columns = ["selection_mode", "checkpoint", "mae_gap_sec", "champion_vs_best_baseline_delta_mae"]
    frame = champion_summary.loc[:, columns].copy()
    for column in ("mae_gap_sec", "champion_vs_best_baseline_delta_mae"):
        frame[column] = frame[column].map(_format_metric)
    return _markdown_table(frame)


def _interval_card_text(interval_summary: pd.DataFrame) -> str:
    preferred = interval_summary[
        interval_summary["selection_mode"].eq("stabilized_nested_guarded")
        & interval_summary["uncertainty_method"].eq("conformal_predicted_gap_bucket")
    ]
    intro = (
        "Predicted-gap-bucket conformal intervals use only prior-fold residuals and choose "
        "calibration buckets from the predicted gap, not the actual qualifying gap. They are "
        "leakage-safe diagnostics, but wider intervals may be needed for better coverage."
    )
    if preferred.empty:
        preferred = interval_summary[
            interval_summary["selection_mode"].eq("stabilized_nested")
            & interval_summary["uncertainty_method"].eq("conformal")
        ]
        intro = (
            "Conformal intervals use only prior-fold residuals. They are leakage-safe diagnostics, "
            "but broad and not guaranteed production-calibrated intervals."
        )
    if preferred.empty:
        return (
            "Prediction intervals are generated where prior-fold residual history is available. "
            "No stabilized conformal interval summary was available for this model card."
        )
    lines = [
        intro,
        "",
        preferred.loc[
            :,
            [
                "checkpoint",
                "interval_availability_rate",
                "interval_coverage",
                "mean_interval_width_sec",
            ],
        ]
        .assign(
            interval_availability_rate=lambda df: df["interval_availability_rate"].map(
                _format_metric
            ),
            interval_coverage=lambda df: df["interval_coverage"].map(_format_metric),
            mean_interval_width_sec=lambda df: df["mean_interval_width_sec"].map(_format_metric),
        )
        .pipe(_markdown_table),
    ]
    return "\n".join(lines)


def _ordered_checkpoints(metrics_by_checkpoint: dict[str, Any] | set[str]) -> list[str]:
    keys = set(metrics_by_checkpoint)
    ordered = [checkpoint for checkpoint in CHECKPOINT_ORDER if checkpoint in keys]
    ordered.extend(sorted(keys - set(ordered)))
    return ordered


def _champion_summary_columns() -> list[str]:
    return [
        "selection_mode",
        "checkpoint",
        "rows",
        "mae_gap_sec",
        "rmse_gap_sec",
        "median_abs_error_gap_sec",
        "mean_position_error",
        "champion_vs_best_baseline_delta_mae",
        "best_baseline_mae_gap_sec",
        "selected_method_summary",
    ]


def _selection_summary_columns() -> list[str]:
    return [
        "selection_mode",
        "checkpoint",
        "selected_family",
        "selected_model_name",
        "selected_feature_group",
        "fold_count",
        "selection_share",
        "fallback_rate",
        "main_fallback_reason",
    ]


def _interval_summary_columns() -> list[str]:
    return [
        "selection_mode",
        "checkpoint",
        "uncertainty_method",
        "interval_availability_rate",
        "interval_coverage",
        "mean_interval_width_sec",
        "median_interval_width_sec",
        "mean_residual_quantile_sec",
        "mean_residual_std_sec",
    ]


def _method_label(family: object, model_name: object, feature_group: object) -> str:
    parts = [
        str(value) for value in (family, model_name, feature_group) if _display_optional(value)
    ]
    return " / ".join(parts)


def _most_common_non_null(values: pd.Series) -> str | None:
    cleaned = values.dropna().astype(str)
    cleaned = cleaned[~cleaned.isin(["", "<NA>", "nan", "None"])]
    if cleaned.empty:
        return None
    return str(cleaned.value_counts().index[0])


def _display_optional(value: object) -> str:
    if value is None or pd.isna(value) or str(value) in {"", "<NA>", "nan", "None"}:
        return ""
    return str(value)


def _number_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _format_metric(value: object) -> str:
    numeric = _number_or_none(value)
    return "" if numeric is None else f"{numeric:.3f}"


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [
        ["" if pd.isna(value) else str(value) for value in record]
        for record in frame.itertuples(index=False, name=None)
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report root must be an object: {path}")
    return payload


def _read_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.is_file() else None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


def _relative_report_path(path: Path) -> str:
    parts = path.parts
    if "reports" in parts:
        index = parts.index("reports")
        return str(Path(*parts[index:]))
    return path.as_posix()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
