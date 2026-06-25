"""Scoped rebuild workflow for season-aware source-contract validation."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig, FeatureConfig, ModelConfig
from f1_prediction.data.season_builder import (
    build_combined_dataset_path,
    build_season_dataset,
    resolve_event_selection,
)
from f1_prediction.modeling.ablation import run_ablation_backtest
from f1_prediction.modeling.artifact_lineage import (
    FP3_CHECKPOINT,
    PREDICTION_TOLERANCE,
    STATIC_FAMILY,
    STATIC_FEATURE_GROUP,
    STATIC_MODEL_NAME,
    STATIC_TEMPORAL_POLICY,
    WEIGHTED_TEMPORAL_POLICY,
    build_season_aware_rebuild_registry,
    compare_static_to_uniform_ablation,
    create_champion_source_lineage_report,
    season_aware_rebuild_artifact_paths,
)
from f1_prediction.modeling.backtest_report import create_backtest_report
from f1_prediction.modeling.backtest_tabular import BacktestStrategy
from f1_prediction.modeling.champion_policy import (
    ChampionSelectionMode,
    ChampionUncertaintyMethod,
    run_champion_backtest,
)
from f1_prediction.modeling.portfolio_report import create_portfolio_report
from f1_prediction.modeling.season_aware_candidate_audit import (
    create_season_aware_candidate_audit_report,
)
from f1_prediction.modeling.season_aware_policy_forensics import (
    create_season_aware_policy_forensics_report,
)
from f1_prediction.modeling.temporal_weighting import TemporalWeightingPolicy
from f1_prediction.utils.paths import ensure_directory

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeasonAwareRebuildSummary:
    """Paths and validation status from the scoped season-aware rebuild."""

    status: str
    summary_path: Path
    registry_path: Path | None
    validation_path: Path | None
    figure_paths: tuple[Path, ...]
    steps_completed: tuple[str, ...]
    steps_skipped: tuple[str, ...]
    warnings: tuple[str, ...]


def scoped_rebuild_artifacts(metrics_dir: Path) -> tuple[Path, ...]:
    """Return the exact files the rebuild command is allowed to refresh."""
    return tuple(season_aware_rebuild_artifact_paths(metrics_dir).values())


def cleanup_scoped_rebuild_artifacts(metrics_dir: Path, *, dry_run: bool) -> tuple[Path, ...]:
    """Remove only known rebuild artifacts when force-refreshing."""
    removed: list[Path] = []
    for path in scoped_rebuild_artifacts(metrics_dir):
        if path.exists():
            removed.append(path)
            if not dry_run:
                path.unlink()
    return tuple(removed)


def create_season_aware_rebuild_report(
    config: DataConfig,
    model_config: ModelConfig,
    feature_config: FeatureConfig,
    *,
    dataset_path: Path | None = None,
    seasons: tuple[int, ...] = (2023, 2024, 2025),
    include_dataset_rebuild: bool = False,
    dry_run: bool = False,
    force: bool = False,
    tolerance: float = PREDICTION_TOLERANCE,
) -> SeasonAwareRebuildSummary:
    """Run or describe the scoped rebuild workflow and write validation artifacts."""
    metrics_dir = config.metrics_output_dir
    figures_dir = metrics_dir.parent / "figures"
    ensure_directory(metrics_dir)
    ensure_directory(figures_dir)

    summary_path = metrics_dir / "season_aware_rebuild_summary.json"
    registry_path = metrics_dir / "season_aware_rebuild_artifact_registry.csv"
    validation_path = metrics_dir / "season_aware_rebuild_validation.csv"
    warnings: list[str] = []
    steps_completed: list[str] = []
    steps_skipped: list[str] = []

    scoped_paths = scoped_rebuild_artifacts(metrics_dir)
    if dry_run:
        payload = {
            "status": "dry_run",
            "static_source_verified": None,
            "uniform_weighted_artifacts_distinct": None,
            "static_uniform_prediction_match_rate": None,
            "static_uniform_max_prediction_delta": None,
            "guarded_static_prediction_match_rate": None,
            "season_aware_weighted_source_verified": None,
            "forensics_counterfactual_valid": None,
            "rebuild_steps_completed": [],
            "rebuild_steps_skipped": _workflow_step_names(include_dataset_rebuild),
            "artifact_paths": [_display_path(path, config.project_root) for path in scoped_paths],
            "warnings": ["dry_run_only_no_artifacts_refreshed"],
            "generated_at": _utc_now(),
        }
        return SeasonAwareRebuildSummary(
            status="dry_run",
            summary_path=summary_path,
            registry_path=None,
            validation_path=None,
            figure_paths=(),
            steps_completed=(),
            steps_skipped=tuple(payload["rebuild_steps_skipped"]),
            warnings=tuple(payload["warnings"]),
        )

    if force:
        removed = cleanup_scoped_rebuild_artifacts(metrics_dir, dry_run=False)
        steps_completed.append(f"cleanup_scoped_artifacts:{len(removed)}")

    resolved_dataset = dataset_path or build_combined_dataset_path(config.modeling_output_dir)
    if include_dataset_rebuild:
        requested_events = resolve_event_selection(list(seasons), None, "conventional")
        build_season_dataset(
            seasons=list(seasons),
            data_config=config,
            feature_config=feature_config,
            events=requested_events,
            preset="conventional",
            force=force,
        )
        steps_completed.append("dataset_rebuild")
    elif not resolved_dataset.is_file():
        raise FileNotFoundError(
            f"Combined modeling dataset is required for scoped rebuild: {resolved_dataset}"
        )
    else:
        steps_completed.append("dataset_verified")

    run_ablation_backtest(
        config,
        strategy=BacktestStrategy.walk_forward,
        dataset_path=resolved_dataset,
        min_events=10,
        min_train_events=5,
        model_config=model_config,
        feature_config=feature_config,
        temporal_weighting=TemporalWeightingPolicy.uniform,
    )
    steps_completed.append("ablation_uniform")
    run_ablation_backtest(
        config,
        strategy=BacktestStrategy.walk_forward,
        dataset_path=resolved_dataset,
        min_events=10,
        min_train_events=5,
        model_config=model_config,
        feature_config=feature_config,
        temporal_weighting=TemporalWeightingPolicy.current_season_only_with_prior,
    )
    steps_completed.append("ablation_current_season_only_with_prior")

    registry = build_season_aware_rebuild_registry(
        metrics_dir=metrics_dir,
        project_root=config.project_root,
        config_path=config.project_root / "configs" / "model.yaml",
    )
    registry.to_csv(registry_path, index=False)
    steps_completed.append("artifact_registry")

    run_champion_backtest(
        config,
        strategy=BacktestStrategy.walk_forward,
        selection_mode=ChampionSelectionMode.static,
        dataset_path=resolved_dataset,
        min_events=10,
        min_train_events=5,
        model_config=model_config,
    )
    steps_completed.append("champion_static")
    for mode in (
        ChampionSelectionMode.stabilized_nested_guarded,
        ChampionSelectionMode.season_aware_nested_guarded,
    ):
        run_champion_backtest(
            config,
            strategy=BacktestStrategy.walk_forward,
            selection_mode=mode,
            dataset_path=resolved_dataset,
            min_events=10,
            min_train_events=5,
            model_config=model_config,
            uncertainty_method=ChampionUncertaintyMethod.conformal_predicted_gap_bucket,
        )
        steps_completed.append(f"champion_{mode.value}")

    lineage = create_champion_source_lineage_report(config, model_config, tolerance=tolerance)
    steps_completed.append("champion_source_lineage")
    audit = create_season_aware_candidate_audit_report(
        config,
        model_config.champion_policy.season_aware_nested_guarded,
    )
    steps_completed.append("season_aware_candidate_audit")
    forensics = create_season_aware_policy_forensics_report(
        config,
        harmful_switch_tolerance_sec=model_config.champion_diagnostics.harmful_switch_tolerance_sec,
    )
    steps_completed.append("season_aware_policy_forensics")
    create_backtest_report(config, dataset_path=resolved_dataset)
    steps_completed.append("backtest_report")
    create_portfolio_report(config)
    steps_completed.append("portfolio_report")

    validation = build_rebuild_validation(metrics_dir=metrics_dir, tolerance=tolerance)
    validation.to_csv(validation_path, index=False)
    registry = build_season_aware_rebuild_registry(
        metrics_dir=metrics_dir,
        project_root=config.project_root,
        config_path=config.project_root / "configs" / "model.yaml",
    )
    registry.to_csv(registry_path, index=False)
    figure_paths, figure_issues = generate_rebuild_figures(
        figures_dir=figures_dir,
        validation=validation,
        registry=registry,
        tolerance=tolerance,
    )
    warnings.extend(figure_issues)

    payload = _build_summary_payload(
        metrics_dir=metrics_dir,
        project_root=config.project_root,
        validation=validation,
        registry=registry,
        lineage_summary_path=lineage.summary_path,
        forensics_summary_path=forensics.summary_path,
        audit_summary_path=audit.summary_path,
        steps_completed=steps_completed,
        steps_skipped=steps_skipped,
        artifact_paths=scoped_paths,
        warnings=warnings,
    )
    _write_json(summary_path, payload)
    return SeasonAwareRebuildSummary(
        status=str(payload["status"]),
        summary_path=summary_path,
        registry_path=registry_path,
        validation_path=validation_path,
        figure_paths=tuple(figure_paths),
        steps_completed=tuple(steps_completed),
        steps_skipped=tuple(steps_skipped),
        warnings=tuple(warnings),
    )


def build_rebuild_validation(
    *,
    metrics_dir: Path,
    tolerance: float = PREDICTION_TOLERANCE,
) -> pd.DataFrame:
    """Validate rebuilt source artifacts without retraining."""
    rows: list[dict[str, object]] = []
    uniform_path = metrics_dir / "ablation_uniform_predictions.parquet"
    weighted_path = metrics_dir / "ablation_current_season_only_with_prior_predictions.parquet"
    static_path = metrics_dir / "champion_static_predictions.parquet"
    guarded_path = metrics_dir / "champion_stabilized_nested_guarded_predictions.parquet"
    season_aware_path = metrics_dir / "champion_season_aware_nested_guarded_predictions.parquet"
    season_aware_selection_path = (
        metrics_dir / "champion_season_aware_nested_guarded_selection.parquet"
    )

    row_comparison = compare_static_to_uniform_ablation(metrics_dir, tolerance=tolerance)
    matched = row_comparison[row_comparison["row_match_status"].eq("matched")]
    static_uniform_match = _mean_bool(matched.get("prediction_tolerance_match"))
    rows.append(
        {
            "check_name": "static_uniform_prediction_match",
            "status": "passed" if static_uniform_match == 1.0 else "failed",
            "match_rate": static_uniform_match,
            "max_prediction_delta_sec": _max_numeric(matched.get("abs_prediction_delta_sec")),
            "details": "Static FP3 champion rows match uniform ablation FP3 candidate.",
        }
    )
    rows.append(
        {
            "check_name": "uniform_weighted_artifacts_distinct",
            "status": "passed"
            if uniform_path.is_file()
            and weighted_path.is_file()
            and uniform_path != weighted_path
            and _file_signature(uniform_path) != _file_signature(weighted_path)
            else "failed",
            "match_rate": None,
            "max_prediction_delta_sec": None,
            "details": "Uniform and weighted ablation snapshots are distinct files.",
        }
    )
    guarded_compare = _compare_guarded_static_source(static_path, guarded_path, tolerance=tolerance)
    rows.append(
        {
            "check_name": "guarded_static_prediction_match",
            "status": "passed" if guarded_compare["match_rate"] == 1.0 else "failed",
            "match_rate": guarded_compare["match_rate"],
            "max_prediction_delta_sec": guarded_compare["max_prediction_delta_sec"],
            "details": "Guarded FP3 rows match verified static source where aligned.",
        }
    )
    weighted_compare = _compare_season_aware_weighted_source(
        season_aware_predictions_path=season_aware_path,
        season_aware_selection_path=season_aware_selection_path,
        weighted_path=weighted_path,
        tolerance=tolerance,
    )
    rows.append(
        {
            "check_name": "season_aware_weighted_source_verified",
            "status": "passed" if weighted_compare["verified"] else "failed",
            "match_rate": weighted_compare["match_rate"],
            "max_prediction_delta_sec": weighted_compare["max_prediction_delta_sec"],
            "details": weighted_compare["details"],
        }
    )
    forensics_path = metrics_dir / "season_aware_policy_forensics_summary.json"
    forensics = _read_json_if_exists(forensics_path)
    valid = bool(
        forensics.get("static_source_verification", {}).get(
            "counterfactual_comparison_valid",
            False,
        )
    )
    rows.append(
        {
            "check_name": "forensics_counterfactual_valid",
            "status": "passed" if valid else "failed",
            "match_rate": 1.0 if valid else 0.0,
            "max_prediction_delta_sec": None,
            "details": "Season-aware forensics has verified static counterfactual source.",
        }
    )
    return pd.DataFrame(rows)


def generate_rebuild_figures(
    *,
    figures_dir: Path,
    validation: pd.DataFrame,
    registry: pd.DataFrame,
    tolerance: float,
) -> tuple[list[Path], list[str]]:
    """Generate simple rebuild-validation figures."""
    cache_dir = figures_dir.parent.parent / ".matplotlib-cache"
    ensure_directory(cache_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_directory(figures_dir)
    figures: list[Path] = []
    issues: list[str] = []
    specs = [
        (
            "season_aware_rebuild_artifact_contract_status.png",
            lambda path: _plot_validation_status(plt, validation, path),
        ),
        (
            "season_aware_rebuild_static_uniform_prediction_delta.png",
            lambda path: _plot_static_uniform_delta(plt, validation, path, tolerance),
        ),
        (
            "season_aware_rebuild_artifact_identity_matrix.png",
            lambda path: _plot_registry_matrix(plt, registry, path),
        ),
        (
            "season_aware_rebuild_fp3_mae_by_source.png",
            lambda path: _plot_fp3_mae_by_source(plt, validation, path),
        ),
        (
            "season_aware_rebuild_validation_status.png",
            lambda path: _plot_validation_status(plt, validation, path),
        ),
    ]
    for filename, plotter in specs:
        path = figures_dir / filename
        try:
            if plotter(path):
                figures.append(path)
            else:
                issues.append(f"skipped:{filename}")
        except Exception as exc:  # pragma: no cover - defensive figure isolation
            LOGGER.warning("Could not generate %s: %s", filename, exc)
            issues.append(f"{filename}:{exc}")
        finally:
            plt.close("all")
    return figures, issues


def _build_summary_payload(
    *,
    metrics_dir: Path,
    project_root: Path,
    validation: pd.DataFrame,
    registry: pd.DataFrame,
    lineage_summary_path: Path,
    forensics_summary_path: Path,
    audit_summary_path: Path,
    steps_completed: list[str],
    steps_skipped: list[str],
    artifact_paths: tuple[Path, ...],
    warnings: list[str],
) -> dict[str, object]:
    lineage = _read_json_if_exists(lineage_summary_path)
    forensics = _read_json_if_exists(forensics_summary_path)
    validation_by_name = {str(row["check_name"]): row.to_dict() for _, row in validation.iterrows()}
    static_uniform = validation_by_name.get("static_uniform_prediction_match", {})
    guarded_static = validation_by_name.get("guarded_static_prediction_match", {})
    weighted = validation_by_name.get("season_aware_weighted_source_verified", {})
    distinct = validation_by_name.get("uniform_weighted_artifacts_distinct", {})
    forensics_valid = validation_by_name.get("forensics_counterfactual_valid", {})
    static_verified = bool(
        lineage.get("static_source_verification", {}).get("static_source_verified", False)
    )
    status = "complete" if validation["status"].eq("passed").all() else "partial"
    return {
        "status": status,
        "static_source_verified": static_verified,
        "uniform_weighted_artifacts_distinct": distinct.get("status") == "passed",
        "static_uniform_prediction_match_rate": static_uniform.get("match_rate"),
        "static_uniform_max_prediction_delta": static_uniform.get("max_prediction_delta_sec"),
        "guarded_static_prediction_match_rate": guarded_static.get("match_rate"),
        "season_aware_weighted_source_verified": weighted.get("status") == "passed",
        "forensics_counterfactual_valid": forensics_valid.get("status") == "passed",
        "forensics_definitive_switch_labels_valid": forensics.get("definitive_switch_labels_valid"),
        "rebuild_steps_completed": steps_completed,
        "rebuild_steps_skipped": steps_skipped,
        "artifact_paths": [_display_path(path, project_root) for path in artifact_paths],
        "registry_rows": registry.to_dict("records"),
        "validation": validation.to_dict("records"),
        "source_lineage_summary_path": _display_path(lineage_summary_path, project_root),
        "candidate_audit_summary_path": _display_path(audit_summary_path, project_root),
        "policy_forensics_summary_path": _display_path(forensics_summary_path, project_root),
        "warnings": warnings,
        "generated_at": _utc_now(),
        "metrics_dir": _display_path(metrics_dir, project_root),
    }


def _workflow_step_names(include_dataset_rebuild: bool) -> list[str]:
    steps = [
        "dataset_rebuild" if include_dataset_rebuild else "dataset_verified",
        "ablation_uniform",
        "ablation_current_season_only_with_prior",
        "artifact_registry",
        "champion_static",
        "champion_stabilized_nested_guarded",
        "champion_season_aware_nested_guarded",
        "champion_source_lineage",
        "season_aware_candidate_audit",
        "season_aware_policy_forensics",
        "backtest_report",
        "portfolio_report",
    ]
    return steps


def _compare_prediction_sources(
    first_path: Path,
    second_path: Path,
    *,
    tolerance: float,
) -> dict[str, object]:
    if not first_path.is_file() or not second_path.is_file():
        return {"match_rate": None, "max_prediction_delta_sec": None, "matched_rows": 0}
    first = _filter_fp3(pd.read_parquet(first_path))
    second = _filter_fp3(pd.read_parquet(second_path))
    if first.empty or second.empty:
        return {"match_rate": None, "max_prediction_delta_sec": None, "matched_rows": 0}
    keys = _join_keys(first, second)
    merged = first.merge(second, on=keys, how="inner", suffixes=("_first", "_second"))
    if merged.empty:
        return {"match_rate": 0.0, "max_prediction_delta_sec": None, "matched_rows": 0}
    delta = (
        pd.to_numeric(merged["predicted_quali_gap_to_pole_sec_first"], errors="coerce")
        - pd.to_numeric(merged["predicted_quali_gap_to_pole_sec_second"], errors="coerce")
    ).abs()
    return {
        "match_rate": float(delta.le(tolerance).mean()),
        "max_prediction_delta_sec": float(delta.max()),
        "matched_rows": int(len(merged)),
    }


def _compare_guarded_static_source(
    static_path: Path,
    guarded_path: Path,
    *,
    tolerance: float,
) -> dict[str, object]:
    if not static_path.is_file() or not guarded_path.is_file():
        return {"match_rate": None, "max_prediction_delta_sec": None, "matched_rows": 0}
    static = _filter_fp3(pd.read_parquet(static_path))
    guarded = _filter_static_source_prediction_rows(_filter_fp3(pd.read_parquet(guarded_path)))
    if static.empty or guarded.empty:
        return {"match_rate": None, "max_prediction_delta_sec": None, "matched_rows": 0}
    return _compare_frames(static, guarded, tolerance=tolerance)


def _compare_season_aware_weighted_source(
    *,
    season_aware_predictions_path: Path,
    season_aware_selection_path: Path,
    weighted_path: Path,
    tolerance: float,
) -> dict[str, object]:
    if (
        not season_aware_predictions_path.is_file()
        or not season_aware_selection_path.is_file()
        or not weighted_path.is_file()
    ):
        return {
            "verified": False,
            "match_rate": None,
            "max_prediction_delta_sec": None,
            "details": "missing_season_aware_or_weighted_artifact",
        }
    season_aware = _filter_fp3(pd.read_parquet(season_aware_predictions_path))
    selection = pd.read_parquet(season_aware_selection_path)
    selected_folds = set(
        pd.to_numeric(
            selection[
                selection.get("season_aware_selected", pd.Series(False, index=selection.index))
                .fillna(False)
                .astype(bool)
                & selection["checkpoint"].astype(str).eq(FP3_CHECKPOINT)
            ]["fold_id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )
    if not selected_folds:
        return {
            "verified": True,
            "match_rate": 1.0,
            "max_prediction_delta_sec": 0.0,
            "details": "no_weighted_candidate_selected",
        }
    season_aware = season_aware[season_aware["fold_id"].isin(selected_folds)].copy()
    weighted = _filter_weighted_fp3(pd.read_parquet(weighted_path))
    comparison = _compare_frames(season_aware, weighted, tolerance=tolerance)
    return {
        "verified": comparison["match_rate"] == 1.0,
        "match_rate": comparison["match_rate"],
        "max_prediction_delta_sec": comparison["max_prediction_delta_sec"],
        "details": f"selected_weighted_folds={len(selected_folds)}",
    }


def _compare_frames(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    tolerance: float,
) -> dict[str, object]:
    if first.empty or second.empty:
        return {"match_rate": 0.0, "max_prediction_delta_sec": None}
    keys = _join_keys(first, second)
    merged = first.merge(second, on=keys, how="inner", suffixes=("_first", "_second"))
    if merged.empty:
        return {"match_rate": 0.0, "max_prediction_delta_sec": None}
    delta = (
        pd.to_numeric(merged["predicted_quali_gap_to_pole_sec_first"], errors="coerce")
        - pd.to_numeric(merged["predicted_quali_gap_to_pole_sec_second"], errors="coerce")
    ).abs()
    return {
        "match_rate": float(delta.le(tolerance).mean()),
        "max_prediction_delta_sec": float(delta.max()),
    }


def _filter_fp3(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "checkpoint" not in frame:
        return frame.iloc[0:0].copy()
    return frame[frame["checkpoint"].astype(str).eq(FP3_CHECKPOINT)].copy()


def _filter_weighted_fp3(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _filter_fp3(frame)
    if frame.empty:
        return frame
    policy = frame.get(
        "temporal_weighting_policy",
        pd.Series(STATIC_TEMPORAL_POLICY, index=frame.index),
    )
    return frame[
        frame.get("model_name", pd.Series(index=frame.index, dtype=str))
        .astype(str)
        .eq(STATIC_MODEL_NAME)
        & frame.get("feature_group", pd.Series(index=frame.index, dtype=str))
        .astype(str)
        .eq(STATIC_FEATURE_GROUP)
        & policy.fillna(STATIC_TEMPORAL_POLICY).astype(str).eq(WEIGHTED_TEMPORAL_POLICY)
    ].copy()


def _filter_static_source_prediction_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    policy = frame.get(
        "selected_temporal_weighting_policy",
        pd.Series(STATIC_TEMPORAL_POLICY, index=frame.index),
    )
    feature_group = frame.get("selected_feature_group", pd.Series("", index=frame.index))
    return frame[
        frame.get("selected_family", pd.Series(index=frame.index, dtype=str))
        .astype(str)
        .eq(STATIC_FAMILY)
        & frame.get("selected_model_name", pd.Series(index=frame.index, dtype=str))
        .astype(str)
        .eq(STATIC_MODEL_NAME)
        & feature_group.fillna("").astype(str).eq(STATIC_FEATURE_GROUP)
        & policy.fillna(STATIC_TEMPORAL_POLICY).astype(str).eq(STATIC_TEMPORAL_POLICY)
    ].copy()


def _join_keys(first: pd.DataFrame, second: pd.DataFrame) -> list[str]:
    driver = "driver" if "driver" in first and "driver" in second else "driver_key"
    event = "event_slug" if "event_slug" in first and "event_slug" in second else "event"
    return ["fold_id", "season", event, "checkpoint", driver]


def _mean_bool(series: Any) -> float | None:
    if series is None:
        return None
    values = pd.Series(series).dropna()
    if values.empty:
        return None
    return float(values.astype(bool).mean())


def _max_numeric(series: Any) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max())


def _file_signature(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plot_validation_status(plt: Any, validation: pd.DataFrame, path: Path) -> bool:
    if validation.empty:
        return False
    labels = validation["check_name"].astype(str).str.replace("_", "\n")
    values = validation["status"].eq("passed").astype(int)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, values, color=["#2c7fb8" if value else "#d95f0e" for value in values])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("passed")
    ax.set_title("Season-aware rebuild validation status")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return True


def _plot_static_uniform_delta(
    plt: Any,
    validation: pd.DataFrame,
    path: Path,
    tolerance: float,
) -> bool:
    row = validation[validation["check_name"].eq("static_uniform_prediction_match")]
    if row.empty:
        return False
    max_delta = row["max_prediction_delta_sec"].iloc[0]
    value = 0.0 if pd.isna(max_delta) else float(max_delta)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["max delta", "tolerance"], [value, tolerance], color=["#2c7fb8", "#636363"])
    ax.set_ylabel("seconds")
    ax.set_title("Static vs uniform prediction delta")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return True


def _plot_registry_matrix(plt: Any, registry: pd.DataFrame, path: Path) -> bool:
    if registry.empty:
        return False
    labels = (
        registry["artifact_kind"].astype(str)
        + "\n"
        + registry["temporal_weighting_policy"].astype(str)
    )
    values = registry["artifact_exists"].fillna(False).astype(int)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, values, color="#2c7fb8")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("exists")
    ax.set_title("Artifact identity matrix")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return True


def _plot_fp3_mae_by_source(plt: Any, validation: pd.DataFrame, path: Path) -> bool:
    rows = validation[validation["match_rate"].notna()]
    if rows.empty:
        return False
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(rows["check_name"].astype(str), rows["match_rate"].astype(float), color="#2c7fb8")
    ax.set_ylabel("match rate")
    ax.set_title("FP3 source validation match rates")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return True


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(_json_safe(payload), output_file, indent=2, allow_nan=False)
        output_file.write("\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if not isinstance(missing, pd.Series) and bool(missing):
        return None
    return value


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
