"""Compact model-versus-baseline backtesting report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.data.season_builder import build_combined_dataset_path
from f1_prediction.modeling.dataset_report import build_dataset_quality_report
from f1_prediction.utils.paths import ensure_directory

TABULAR_MODEL_NAMES: tuple[str, ...] = ("ridge", "random_forest")


@dataclass(frozen=True)
class BacktestReportSummary:
    """Key backtesting result counts and output path."""

    dataset_rows: int
    n_events: int
    training_status: str
    tabular_models: tuple[str, ...]
    output_path: Path


def create_backtest_report(
    config: DataConfig,
    *,
    dataset_path: Path | None = None,
    baseline_metrics_path: Path | None = None,
    tabular_metrics_path: Path | None = None,
    quality_report_path: Path | None = None,
    repeated_metrics_path: Path | None = None,
    walk_forward_metrics_path: Path | None = None,
    ablation_metrics_path: Path | None = None,
    boosted_metrics_path: Path | None = None,
    champion_metrics_path: Path | None = None,
    temporal_weighting_summary_path: Path | None = None,
    season_aware_validation_summary_path: Path | None = None,
    season_aware_candidate_audit_summary_path: Path | None = None,
    season_aware_policy_forensics_summary_path: Path | None = None,
    champion_source_lineage_summary_path: Path | None = None,
    season_aware_rebuild_summary_path: Path | None = None,
) -> BacktestReportSummary:
    """Read available evaluation artifacts and persist a compact summary."""
    source_path = _resolve_path(
        dataset_path or build_combined_dataset_path(config.modeling_output_dir),
        config.project_root,
        required=True,
    )
    baseline_path = _resolve_path(
        baseline_metrics_path or config.metrics_output_dir / "baseline_metrics.json",
        config.project_root,
    )
    tabular_path = _resolve_path(
        tabular_metrics_path or config.metrics_output_dir / "tabular_model_metrics.json",
        config.project_root,
    )
    quality_path = _resolve_path(
        quality_report_path or config.metrics_output_dir / "dataset_quality_report.json",
        config.project_root,
    )
    repeated_path = _resolve_path(
        repeated_metrics_path or config.metrics_output_dir / "repeated_event_holdout_metrics.json",
        config.project_root,
    )
    walk_forward_path = _resolve_path(
        walk_forward_metrics_path or config.metrics_output_dir / "walk_forward_metrics.json",
        config.project_root,
    )
    ablation_path = _resolve_path(
        ablation_metrics_path or config.metrics_output_dir / "ablation_metrics.json",
        config.project_root,
    )
    boosted_path = _resolve_path(
        boosted_metrics_path or config.metrics_output_dir / "boosted_metrics.json",
        config.project_root,
    )
    champion_path = _resolve_path(
        champion_metrics_path or config.metrics_output_dir / "champion_metrics.json",
        config.project_root,
    )
    temporal_weighting_path = _resolve_path(
        temporal_weighting_summary_path
        or config.metrics_output_dir / "temporal_weighting_summary.json",
        config.project_root,
    )
    season_aware_validation_path = _resolve_path(
        season_aware_validation_summary_path
        or config.metrics_output_dir / "season_aware_validation_summary.json",
        config.project_root,
    )
    season_aware_champion_path = _resolve_path(
        config.metrics_output_dir / "season_aware_champion_summary.json",
        config.project_root,
    )
    season_aware_candidate_audit_path = _resolve_path(
        season_aware_candidate_audit_summary_path
        or config.metrics_output_dir / "season_aware_candidate_audit_summary.json",
        config.project_root,
    )
    season_aware_policy_forensics_path = _resolve_path(
        season_aware_policy_forensics_summary_path
        or config.metrics_output_dir / "season_aware_policy_forensics_summary.json",
        config.project_root,
    )
    champion_source_lineage_path = _resolve_path(
        champion_source_lineage_summary_path
        or config.metrics_output_dir / "champion_source_lineage_manifest.json",
        config.project_root,
    )
    season_aware_rebuild_path = _resolve_path(
        season_aware_rebuild_summary_path
        or config.metrics_output_dir / "season_aware_rebuild_summary.json",
        config.project_root,
    )
    champion_mode_metrics = _read_champion_mode_metrics(config.metrics_output_dir)
    dataset = pd.read_parquet(source_path)
    quality = (
        _read_json(quality_path)
        if quality_path.is_file()
        else build_dataset_quality_report(dataset)
    )
    baseline_metrics = _read_json(baseline_path) if baseline_path.is_file() else {}
    tabular_metrics = _read_json(tabular_path) if tabular_path.is_file() else None
    repeated_metrics = _read_json(repeated_path) if repeated_path.is_file() else None
    walk_forward_metrics = _read_json(walk_forward_path) if walk_forward_path.is_file() else None
    ablation_metrics = _read_json(ablation_path) if ablation_path.is_file() else None
    boosted_metrics = _read_json(boosted_path) if boosted_path.is_file() else None
    champion_metrics = _read_json(champion_path) if champion_path.is_file() else None
    temporal_weighting_summary = (
        _read_json(temporal_weighting_path) if temporal_weighting_path.is_file() else None
    )
    season_aware_validation_summary = (
        _read_json(season_aware_validation_path) if season_aware_validation_path.is_file() else None
    )
    season_aware_champion_summary = (
        _read_json(season_aware_champion_path) if season_aware_champion_path.is_file() else None
    )
    season_aware_candidate_audit_summary = (
        _read_json(season_aware_candidate_audit_path)
        if season_aware_candidate_audit_path.is_file()
        else None
    )
    season_aware_policy_forensics_summary = (
        _read_json(season_aware_policy_forensics_path)
        if season_aware_policy_forensics_path.is_file()
        else None
    )
    champion_source_lineage_summary = (
        _read_json(champion_source_lineage_path) if champion_source_lineage_path.is_file() else None
    )
    season_aware_rebuild_summary = (
        _read_json(season_aware_rebuild_path) if season_aware_rebuild_path.is_file() else None
    )
    if champion_metrics is not None:
        mode = str(champion_metrics.get("selection_mode", ""))
        if mode:
            champion_mode_metrics.setdefault(mode, champion_metrics)
    payload = build_backtest_report_payload(
        quality,
        baseline_metrics,
        tabular_metrics,
        repeated_metrics=repeated_metrics,
        walk_forward_metrics=walk_forward_metrics,
        ablation_metrics=ablation_metrics,
        boosted_metrics=boosted_metrics,
        champion_metrics=champion_metrics,
        champion_mode_metrics=champion_mode_metrics,
        temporal_weighting_summary=temporal_weighting_summary,
        season_aware_validation_summary=season_aware_validation_summary,
        season_aware_champion_summary=season_aware_champion_summary,
        season_aware_candidate_audit_summary=season_aware_candidate_audit_summary,
        season_aware_policy_forensics_summary=season_aware_policy_forensics_summary,
        champion_source_lineage_summary=champion_source_lineage_summary,
        season_aware_rebuild_summary=season_aware_rebuild_summary,
    )

    output_path = config.metrics_output_dir / "backtest_report.json"
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, allow_nan=False)
        output_file.write("\n")
    return BacktestReportSummary(
        dataset_rows=int(payload["dataset_rows"]),
        n_events=int(payload["n_events"]),
        training_status=str(payload["training_status"]),
        tabular_models=tuple(payload["tabular_models_available"]),
        output_path=output_path,
    )


def build_backtest_report_payload(
    quality: dict[str, Any],
    baseline_metrics: dict[str, Any],
    tabular_metrics: dict[str, Any] | None,
    *,
    repeated_metrics: dict[str, Any] | None = None,
    walk_forward_metrics: dict[str, Any] | None = None,
    ablation_metrics: dict[str, Any] | None = None,
    boosted_metrics: dict[str, Any] | None = None,
    champion_metrics: dict[str, Any] | None = None,
    champion_mode_metrics: dict[str, dict[str, Any]] | None = None,
    temporal_weighting_summary: dict[str, Any] | None = None,
    season_aware_validation_summary: dict[str, Any] | None = None,
    season_aware_champion_summary: dict[str, Any] | None = None,
    season_aware_candidate_audit_summary: dict[str, Any] | None = None,
    season_aware_policy_forensics_summary: dict[str, Any] | None = None,
    champion_source_lineage_summary: dict[str, Any] | None = None,
    season_aware_rebuild_summary: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Compose comparable best-model and best-baseline metrics by checkpoint."""
    available_backtests = _available_backtests(
        walk_forward_metrics,
        repeated_metrics,
        tabular_metrics,
    )
    preferred_strategy, preferred_metrics = _preferred_backtest(
        walk_forward_metrics,
        repeated_metrics,
    )
    if preferred_metrics is not None:
        payload = _multi_fold_report_payload(
            quality,
            preferred_strategy,
            preferred_metrics,
            available_backtests,
        )
        payload.update(_ablation_summary(ablation_metrics))
        payload.update(_boosted_summary(payload, ablation_metrics, boosted_metrics))
        payload.update(_champion_summary(champion_metrics, champion_mode_metrics))
        payload.update(_temporal_weighting_summary(temporal_weighting_summary))
        payload.update(_season_aware_validation_summary(season_aware_validation_summary))
        payload.update(_season_aware_champion_summary(season_aware_champion_summary))
        payload.update(_season_aware_candidate_audit_summary(season_aware_candidate_audit_summary))
        payload.update(
            _season_aware_policy_forensics_summary(season_aware_policy_forensics_summary)
        )
        payload.update(_champion_source_lineage_summary(champion_source_lineage_summary))
        payload.update(_season_aware_rebuild_summary(season_aware_rebuild_summary))
        return payload

    training_status = (
        str(tabular_metrics.get("status", "unavailable")) if tabular_metrics else "unavailable"
    )
    model_metrics = tabular_metrics.get("models", {}) if training_status == "trained" else {}
    tabular_models = [name for name in TABULAR_MODEL_NAMES if name in model_metrics]
    checkpoints = [str(value) for value in quality.get("checkpoints", [])]
    if not checkpoints:
        checkpoints = _metric_checkpoints(baseline_metrics, model_metrics)

    holdout_baselines = (
        tabular_metrics.get("best_baseline_by_checkpoint", {})
        if training_status == "trained" and tabular_metrics
        else {}
    )
    best_baselines = _best_baselines(checkpoints, baseline_metrics, holdout_baselines)
    best_models = _best_tabular_models(checkpoints, model_metrics, tabular_models)
    mae_deltas, position_deltas = _comparison_deltas(
        checkpoints,
        best_baselines,
        best_models,
    )
    payload = {
        "dataset_rows": int(quality.get("n_rows", 0)),
        "n_events": int(quality.get("n_events", 0)),
        "events": list(quality.get("events", [])),
        "n_drivers": int(quality.get("n_drivers", 0)),
        "checkpoints": checkpoints,
        "best_baseline_by_checkpoint": best_baselines,
        "tabular_models_available": tabular_models,
        "best_tabular_model_by_checkpoint": best_models,
        "training_status": training_status,
        "available_backtests": available_backtests,
        "preferred_backtest_strategy": (
            "single_event_holdout" if training_status == "trained" else None
        ),
        "best_model_by_checkpoint": best_models,
        "model_vs_baseline_delta_mae_by_checkpoint": mae_deltas,
        "model_vs_baseline_delta_position_error_by_checkpoint": position_deltas,
        "n_folds_successful": 1 if training_status == "trained" else 0,
        "n_folds_failed": 0,
        "created_at_utc": _utc_now(),
    }
    payload.update(_ablation_summary(ablation_metrics))
    payload.update(_boosted_summary(payload, ablation_metrics, boosted_metrics))
    payload.update(_champion_summary(champion_metrics, champion_mode_metrics))
    payload.update(_temporal_weighting_summary(temporal_weighting_summary))
    payload.update(_season_aware_validation_summary(season_aware_validation_summary))
    payload.update(_season_aware_champion_summary(season_aware_champion_summary))
    payload.update(_season_aware_candidate_audit_summary(season_aware_candidate_audit_summary))
    payload.update(_season_aware_policy_forensics_summary(season_aware_policy_forensics_summary))
    payload.update(_champion_source_lineage_summary(champion_source_lineage_summary))
    payload.update(_season_aware_rebuild_summary(season_aware_rebuild_summary))
    return payload


def _multi_fold_report_payload(
    quality: dict[str, Any],
    strategy: str,
    metrics: dict[str, Any],
    available_backtests: list[str],
) -> dict[str, object]:
    best_models = metrics.get("best_model_by_checkpoint", {})
    best_baselines = metrics.get("best_baseline_by_checkpoint", {})
    checkpoints = [str(value) for value in quality.get("checkpoints", [])]
    mae_deltas: dict[str, float | None] = {}
    position_deltas: dict[str, float | None] = {}
    for checkpoint in checkpoints:
        model = best_models.get(checkpoint, {})
        baseline = best_baselines.get(checkpoint, {})
        mae_deltas[checkpoint] = _delta(model.get("mae_gap_sec"), baseline.get("mae_gap_sec"))
        position_deltas[checkpoint] = _delta(
            model.get("mean_abs_position_error"),
            baseline.get("mean_abs_position_error"),
        )
    tabular_models = list(metrics.get("tabular_models", []))
    return {
        "dataset_rows": int(quality.get("n_rows", 0)),
        "n_events": int(quality.get("n_events", 0)),
        "events": list(quality.get("events", [])),
        "n_drivers": int(quality.get("n_drivers", 0)),
        "checkpoints": checkpoints,
        "best_baseline_by_checkpoint": best_baselines,
        "tabular_models_available": tabular_models,
        "best_tabular_model_by_checkpoint": best_models,
        "best_model_by_checkpoint": best_models,
        "model_vs_baseline_delta_mae_by_checkpoint": mae_deltas,
        "model_vs_baseline_delta_position_error_by_checkpoint": position_deltas,
        "training_status": str(metrics.get("status", "unavailable")),
        "available_backtests": available_backtests,
        "preferred_backtest_strategy": strategy,
        "n_folds_successful": int(metrics.get("n_folds_successful", 0)),
        "n_folds_failed": int(metrics.get("n_folds_failed", 0)),
        "created_at_utc": _utc_now(),
    }


def _available_backtests(
    walk_forward: dict[str, Any] | None,
    repeated: dict[str, Any] | None,
    single: dict[str, Any] | None,
) -> list[str]:
    available: list[str] = []
    if walk_forward is not None:
        available.append("walk_forward")
    if repeated is not None:
        available.append("repeated_event_holdout")
    if single is not None:
        available.append("single_event_holdout")
    return available


def _preferred_backtest(
    walk_forward: dict[str, Any] | None,
    repeated: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any] | None]:
    usable_statuses = {"complete", "partial"}
    if walk_forward and walk_forward.get("status") in usable_statuses:
        return "walk_forward", walk_forward
    if repeated and repeated.get("status") in usable_statuses:
        return "repeated_event_holdout", repeated
    return None, None


def _best_baselines(
    checkpoints: list[str],
    baseline_metrics: dict[str, Any],
    holdout_baselines: dict[str, Any],
) -> dict[str, dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for checkpoint in checkpoints:
        holdout = holdout_baselines.get(checkpoint)
        if holdout and holdout.get("mae_gap_sec") is not None:
            best[checkpoint] = dict(holdout)
            continue
        candidates = [
            (name, metrics[checkpoint])
            for name, metrics in baseline_metrics.items()
            if checkpoint in metrics and metrics[checkpoint].get("mae_gap_sec") is not None
        ]
        if candidates:
            name, metrics = min(candidates, key=lambda item: float(item[1]["mae_gap_sec"]))
            best[checkpoint] = {
                "baseline_name": name,
                "mae_gap_sec": metrics.get("mae_gap_sec"),
                "mean_abs_position_error": metrics.get("mean_abs_position_error"),
            }
    return best


def _best_tabular_models(
    checkpoints: list[str],
    model_metrics: dict[str, Any],
    tabular_models: list[str],
) -> dict[str, dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for checkpoint in checkpoints:
        candidates = [
            (name, model_metrics[name][checkpoint])
            for name in tabular_models
            if checkpoint in model_metrics[name]
            and model_metrics[name][checkpoint].get("mae_gap_sec") is not None
        ]
        if candidates:
            name, metrics = min(candidates, key=lambda item: float(item[1]["mae_gap_sec"]))
            best[checkpoint] = {
                "model_name": name,
                "mae_gap_sec": metrics.get("mae_gap_sec"),
                "mean_abs_position_error": metrics.get("mean_abs_position_error"),
            }
    return best


def _comparison_deltas(
    checkpoints: list[str],
    baselines: dict[str, dict[str, object]],
    models: dict[str, dict[str, object]],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    mae: dict[str, float | None] = {}
    position: dict[str, float | None] = {}
    for checkpoint in checkpoints:
        baseline = baselines.get(checkpoint, {})
        model = models.get(checkpoint, {})
        mae[checkpoint] = _delta(model.get("mae_gap_sec"), baseline.get("mae_gap_sec"))
        position[checkpoint] = _delta(
            model.get("mean_abs_position_error"),
            baseline.get("mean_abs_position_error"),
        )
    return mae, position


def _ablation_summary(metrics: dict[str, Any] | None) -> dict[str, object]:
    usable = metrics is not None and metrics.get("status") in {"complete", "partial"}
    if not usable:
        return {
            "available_ablation_results": [],
            "preferred_feature_group_by_checkpoint": {},
            "best_ablation_model_by_checkpoint": {},
            "best_ablation_delta_vs_baseline_by_checkpoint": {},
        }
    best_overall = metrics.get("best_overall_by_checkpoint", {})
    best_baselines = metrics.get("best_baseline_by_checkpoint", {})
    preferred_groups: dict[str, str] = {}
    best_models: dict[str, dict[str, object]] = {}
    deltas: dict[str, float | None] = {}
    for checkpoint, values in best_overall.items():
        preferred_groups[str(checkpoint)] = str(values.get("feature_group"))
        best_models[str(checkpoint)] = dict(values)
        deltas[str(checkpoint)] = _delta(
            values.get("mae_gap_sec"),
            best_baselines.get(checkpoint, {}).get("mae_gap_sec"),
        )
    return {
        "available_ablation_results": list(metrics.get("feature_groups", [])),
        "preferred_feature_group_by_checkpoint": preferred_groups,
        "best_ablation_model_by_checkpoint": best_models,
        "best_ablation_delta_vs_baseline_by_checkpoint": deltas,
    }


def _boosted_summary(
    report: dict[str, object],
    ablation_metrics: dict[str, Any] | None,
    boosted_metrics: dict[str, Any] | None,
) -> dict[str, object]:
    usable = boosted_metrics is not None and boosted_metrics.get("status") in {
        "complete",
        "partial",
    }
    if not usable:
        return {
            "boosted_models_available": [],
            "best_boosted_model_by_checkpoint": {},
            "boosted_vs_best_baseline_delta_mae_by_checkpoint": {},
            "boosted_vs_best_tabular_delta_mae_by_checkpoint": {},
            "boosted_vs_best_ablation_delta_mae_by_checkpoint": {},
            "preferred_model_family_by_checkpoint": _preferred_model_families(
                report, ablation_metrics, None
            ),
        }
    best_boosted = boosted_metrics.get("best_model_by_checkpoint", {})
    best_tabular = report.get("best_tabular_model_by_checkpoint", {})
    best_ablation = (
        ablation_metrics.get("best_overall_by_checkpoint", {})
        if ablation_metrics and ablation_metrics.get("status") in {"complete", "partial"}
        else {}
    )
    best_baseline = boosted_metrics.get("best_baseline_by_checkpoint", {})
    checkpoints = set(best_boosted) | set(best_tabular) | set(best_ablation) | set(best_baseline)
    baseline_deltas: dict[str, float | None] = {}
    tabular_deltas: dict[str, float | None] = {}
    ablation_deltas: dict[str, float | None] = {}
    for checkpoint in sorted(checkpoints):
        boosted_mae = best_boosted.get(checkpoint, {}).get("mae_gap_sec")
        baseline_deltas[checkpoint] = _delta(
            boosted_mae, best_baseline.get(checkpoint, {}).get("mae_gap_sec")
        )
        tabular_deltas[checkpoint] = _delta(
            boosted_mae, best_tabular.get(checkpoint, {}).get("mae_gap_sec")
        )
        ablation_deltas[checkpoint] = _delta(
            boosted_mae, best_ablation.get(checkpoint, {}).get("mae_gap_sec")
        )
    return {
        "boosted_models_available": list(boosted_metrics.get("models", [])),
        "best_boosted_model_by_checkpoint": best_boosted,
        "boosted_vs_best_baseline_delta_mae_by_checkpoint": baseline_deltas,
        "boosted_vs_best_tabular_delta_mae_by_checkpoint": tabular_deltas,
        "boosted_vs_best_ablation_delta_mae_by_checkpoint": ablation_deltas,
        "preferred_model_family_by_checkpoint": _preferred_model_families(
            report, ablation_metrics, boosted_metrics
        ),
    }


def _preferred_model_families(
    report: dict[str, object],
    ablation_metrics: dict[str, Any] | None,
    boosted_metrics: dict[str, Any] | None,
) -> dict[str, str]:
    baselines = report.get("best_baseline_by_checkpoint", {})
    tabular = report.get("best_tabular_model_by_checkpoint", {})
    ablation = (
        ablation_metrics.get("best_overall_by_checkpoint", {})
        if ablation_metrics and ablation_metrics.get("status") in {"complete", "partial"}
        else {}
    )
    boosted = (
        boosted_metrics.get("best_model_by_checkpoint", {})
        if boosted_metrics and boosted_metrics.get("status") in {"complete", "partial"}
        else {}
    )
    checkpoints = set(baselines) | set(tabular) | set(ablation) | set(boosted)
    preferred: dict[str, str] = {}
    for checkpoint in sorted(checkpoints):
        candidates = {
            "baseline": baselines.get(checkpoint, {}).get("mae_gap_sec"),
            "tabular": tabular.get(checkpoint, {}).get("mae_gap_sec"),
            "ablation": ablation.get(checkpoint, {}).get("mae_gap_sec"),
            "boosted": boosted.get(checkpoint, {}).get("mae_gap_sec"),
        }
        numeric = {name: float(value) for name, value in candidates.items() if value is not None}
        if numeric:
            preferred[checkpoint] = min(numeric, key=numeric.get)
    return preferred


def _champion_summary(
    metrics: dict[str, Any] | None,
    mode_metrics: dict[str, dict[str, Any]] | None = None,
) -> dict[str, object]:
    usable_modes = {
        mode: payload
        for mode, payload in (mode_metrics or {}).items()
        if payload.get("status") in {"complete", "partial"}
    }
    usable = metrics is not None and metrics.get("status") in {"complete", "partial"}
    if not usable and not usable_modes:
        return {
            "champion_available": False,
            "champion_selection_mode": None,
            "champion_selection_modes_available": [],
            "champion_metrics_by_checkpoint": {},
            "champion_vs_best_baseline_delta_mae": {},
            "champion_vs_best_single_family_delta_mae": {},
            "best_champion_selection_mode_by_checkpoint": {},
            "best_champion_selection_mode_overall": None,
            "champion_interval_coverage_by_checkpoint": {},
            "champion_interval_width_by_checkpoint": {},
            "champion_interval_metrics_by_predicted_gap_bucket": {},
            "preferred_final_policy_by_checkpoint": {},
        }
    latest = metrics if usable else next(iter(usable_modes.values()))
    checkpoint_metrics = latest.get("metrics_by_checkpoint", {})
    selection_mode = str(latest.get("selection_mode", "unknown"))
    best_by_checkpoint = _best_champion_modes_by_checkpoint(usable_modes)
    best_overall = _best_champion_mode_overall(usable_modes)
    interval_coverage = {
        str(checkpoint): values.get("interval_coverage")
        for checkpoint, values in checkpoint_metrics.items()
    }
    interval_width = {
        str(checkpoint): {
            "mean_interval_width_sec": values.get("mean_interval_width_sec"),
            "median_interval_width_sec": values.get("median_interval_width_sec"),
            "interval_availability_rate": values.get("interval_availability_rate"),
        }
        for checkpoint, values in checkpoint_metrics.items()
    }
    preferred = {
        str(checkpoint): {
            "family": "champion",
            "selection_mode": selection_mode,
            "mae_gap_sec": values.get("mae_gap_sec"),
            "mean_abs_position_error": values.get("mean_abs_position_error"),
        }
        for checkpoint, values in checkpoint_metrics.items()
    }
    return {
        "champion_available": True,
        "champion_selection_mode": selection_mode,
        "champion_selection_modes_available": sorted(usable_modes),
        "champion_metrics_by_checkpoint": checkpoint_metrics,
        "champion_vs_best_baseline_delta_mae": latest.get(
            "champion_vs_best_baseline_delta_mae", {}
        ),
        "champion_vs_best_single_family_delta_mae": latest.get(
            "champion_vs_best_single_family_delta_mae", {}
        ),
        "best_champion_selection_mode_by_checkpoint": best_by_checkpoint,
        "best_champion_selection_mode_overall": best_overall,
        "champion_interval_coverage_by_checkpoint": interval_coverage,
        "champion_interval_width_by_checkpoint": interval_width,
        "champion_interval_metrics_by_predicted_gap_bucket": latest.get(
            "interval_metrics_by_predicted_gap_bucket", {}
        ),
        "preferred_final_policy_by_checkpoint": preferred,
    }


def _temporal_weighting_summary(summary: dict[str, Any] | None) -> dict[str, object]:
    if not summary:
        return {
            "temporal_weighting_policies_available": {},
            "best_temporal_weighting_policy_by_checkpoint": {},
            "temporal_weighting_vs_uniform_delta_by_checkpoint": {},
        }
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
    }


def _season_aware_validation_summary(summary: dict[str, Any] | None) -> dict[str, object]:
    if not summary:
        return {
            "season_aware_validation_available": False,
            "season_aware_fp3_candidate_summary": {},
            "season_aware_best_fixed_candidate": {},
            "season_aware_promotion_recommendation": "insufficient_evidence",
        }
    return {
        "season_aware_validation_available": bool(
            summary.get("season_aware_validation_available", False)
        ),
        "season_aware_fp3_candidate_summary": summary.get("season_aware_fp3_candidate_summary", {}),
        "season_aware_best_fixed_candidate": summary.get("season_aware_best_fixed_candidate", {}),
        "season_aware_promotion_recommendation": summary.get(
            "season_aware_promotion_recommendation",
            "insufficient_evidence",
        ),
    }


def _season_aware_champion_summary(summary: dict[str, Any] | None) -> dict[str, object]:
    if not summary:
        return {
            "season_aware_champion_available": False,
            "season_aware_champion_fp3_summary": {},
            "season_aware_champion_bootstrap_ci": {},
            "season_aware_champion_promotion_recommendation": "retain_static_policy",
        }
    return {
        "season_aware_champion_available": bool(summary.get("status") != "missing_inputs"),
        "season_aware_champion_fp3_summary": summary.get("fp3_summary", {}),
        "season_aware_champion_bootstrap_ci": summary.get("bootstrap_ci", {}),
        "season_aware_champion_promotion_recommendation": summary.get(
            "promotion_recommendation",
            "season_aware_candidate_experimental",
        ),
    }


def _season_aware_candidate_audit_summary(summary: dict[str, Any] | None) -> dict[str, object]:
    if not summary:
        return {
            "season_aware_candidate_audit_available": False,
            "season_aware_candidate_audit_recommendation": "retain_static_policy",
            "season_aware_candidate_gate_failure_summary": {},
            "season_aware_candidate_artifact_alignment_summary": {},
            "season_aware_candidate_comparator_consistency_rate": None,
            "season_aware_candidate_selection_consistency_rate": None,
            "season_aware_candidate_comparator_scope": None,
        }
    return {
        "season_aware_candidate_audit_available": bool(summary.get("status") != "missing_inputs"),
        "season_aware_candidate_audit_recommendation": summary.get(
            "recommendation",
            "retain_static_policy",
        ),
        "season_aware_candidate_gate_failure_summary": summary.get(
            "live_gate_summary",
            {},
        ),
        "season_aware_candidate_artifact_alignment_summary": summary.get(
            "artifact_alignment_summary",
            {},
        ),
        "season_aware_candidate_comparator_consistency_rate": summary.get(
            "live_audit_metric_consistency_rate"
        ),
        "season_aware_candidate_selection_consistency_rate": summary.get(
            "live_audit_selection_consistency_rate"
        ),
        "season_aware_candidate_comparator_scope": summary.get("comparator_scope_description"),
    }


def _season_aware_policy_forensics_summary(summary: dict[str, Any] | None) -> dict[str, object]:
    if not summary:
        return {
            "season_aware_policy_forensics_available": False,
            "season_aware_policy_reconstruction_summary": {},
            "season_aware_policy_selected_fold_summary": {},
            "season_aware_policy_guardrail_summary": {},
            "season_aware_policy_forensics_recommendation": "retain_static_policy",
        }
    return {
        "season_aware_policy_forensics_available": bool(summary.get("status") != "missing_inputs"),
        "season_aware_policy_reconstruction_summary": summary.get("reconstruction_summary", {}),
        "season_aware_policy_selected_fold_summary": summary.get("selected_fold_summary", {}),
        "season_aware_policy_guardrail_summary": summary.get("guardrail_simulation_summary", {}),
        "season_aware_policy_forensics_recommendation": summary.get(
            "recommendation",
            "retain_static_policy",
        ),
    }


def _champion_source_lineage_summary(summary: dict[str, Any] | None) -> dict[str, object]:
    if not summary:
        return {
            "champion_source_lineage_available": False,
            "static_source_lineage_verified": None,
            "season_aware_counterfactual_comparisons_valid": None,
            "champion_source_lineage_warning": None,
            "champion_source_lineage_rebuild_workflow": {},
        }
    verification = summary.get("static_source_verification", {})
    root_cause = summary.get("root_cause_classification")
    verified = bool(verification.get("static_source_verified", False))
    invalid_reason = verification.get("counterfactual_invalid_reason")
    warning = None
    if not verified:
        warning = (
            "Static-source lineage is unverified; season-aware counterfactual labels should be "
            f"treated as diagnostic only. Root cause: {root_cause}; reason: {invalid_reason}."
        )
    return {
        "champion_source_lineage_available": bool(summary.get("status") != "missing_inputs"),
        "static_source_lineage_verified": verified,
        "season_aware_counterfactual_comparisons_valid": bool(
            verification.get("counterfactual_comparison_valid", False)
        ),
        "champion_source_lineage_root_cause": root_cause,
        "champion_source_lineage_warning": warning,
        "champion_source_lineage_rebuild_workflow": summary.get("clean_rebuild_workflow", {}),
    }


def _season_aware_rebuild_summary(summary: dict[str, Any] | None) -> dict[str, object]:
    if not summary:
        return {
            "season_aware_rebuild_available": False,
            "season_aware_rebuild_status": None,
            "season_aware_rebuild_static_source_verified": None,
            "season_aware_rebuild_forensics_counterfactual_valid": None,
            "season_aware_rebuild_warning": None,
        }
    verified = bool(summary.get("static_source_verified", False))
    valid = bool(summary.get("forensics_counterfactual_valid", False))
    warning = None
    if not verified or not valid:
        warning = (
            "Season-aware source-contract rebuild is incomplete or unverified; keep "
            "season-aware conclusions diagnostic until the scoped rebuild passes."
        )
    return {
        "season_aware_rebuild_available": True,
        "season_aware_rebuild_status": summary.get("status"),
        "season_aware_rebuild_static_source_verified": verified,
        "season_aware_rebuild_forensics_counterfactual_valid": valid,
        "season_aware_rebuild_static_uniform_prediction_match_rate": summary.get(
            "static_uniform_prediction_match_rate"
        ),
        "season_aware_rebuild_guarded_static_prediction_match_rate": summary.get(
            "guarded_static_prediction_match_rate"
        ),
        "season_aware_rebuild_warning": warning,
    }


def _best_champion_modes_by_checkpoint(
    mode_metrics: dict[str, dict[str, Any]],
) -> dict[str, dict[str, object]]:
    checkpoints: set[str] = set()
    for payload in mode_metrics.values():
        checkpoints.update(
            str(checkpoint) for checkpoint in payload.get("metrics_by_checkpoint", {})
        )
    best: dict[str, dict[str, object]] = {}
    for checkpoint in sorted(checkpoints):
        choices: list[tuple[str, dict[str, Any]]] = []
        for mode, payload in mode_metrics.items():
            values = payload.get("metrics_by_checkpoint", {}).get(checkpoint, {})
            if values.get("mae_gap_sec") is not None:
                choices.append((mode, values))
        if choices:
            mode, values = min(choices, key=lambda item: float(item[1]["mae_gap_sec"]))
            best[checkpoint] = {
                "selection_mode": mode,
                "mae_gap_sec": values.get("mae_gap_sec"),
                "mean_abs_position_error": values.get("mean_abs_position_error"),
            }
    return best


def _best_champion_mode_overall(
    mode_metrics: dict[str, dict[str, Any]],
) -> dict[str, object] | None:
    choices: list[tuple[str, float]] = []
    for mode, payload in mode_metrics.items():
        values = [
            float(metrics["mae_gap_sec"])
            for metrics in payload.get("metrics_by_checkpoint", {}).values()
            if metrics.get("mae_gap_sec") is not None
        ]
        if values:
            choices.append((mode, sum(values) / len(values)))
    if not choices:
        return None
    mode, mean_mae = min(choices, key=lambda item: item[1])
    return {"selection_mode": mode, "mean_mae_gap_sec": mean_mae}


def _delta(model_value: object, baseline_value: object) -> float | None:
    if model_value is None or baseline_value is None:
        return None
    return float(model_value) - float(baseline_value)


def _metric_checkpoints(*metric_groups: dict[str, Any]) -> list[str]:
    checkpoints: list[str] = []
    for group in metric_groups:
        for metrics in group.values():
            for checkpoint in metrics:
                if checkpoint not in checkpoints:
                    checkpoints.append(checkpoint)
    return checkpoints


def _resolve_path(path: Path, project_root: Path, *, required: bool = False) -> Path:
    resolved = path if path.is_absolute() else project_root / path
    if required and not resolved.is_file():
        raise FileNotFoundError(f"Required backtest input does not exist: {resolved}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report root must be an object: {path}")
    return payload


def _read_champion_mode_metrics(metrics_dir: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for mode in (
        "static",
        "nested",
        "stabilized_nested",
        "stabilized_nested_guarded",
        "season_aware_nested_guarded",
    ):
        path = metrics_dir / f"champion_{mode}_metrics.json"
        if not path.is_file():
            continue
        payload = _read_json(path)
        payloads[str(payload.get("selection_mode", mode))] = payload
    return payloads


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
