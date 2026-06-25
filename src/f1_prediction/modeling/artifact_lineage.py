"""Artifact lineage and source-contract diagnostics for champion predictions."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig, ModelConfig
from f1_prediction.utils.paths import ensure_directory

FP3_CHECKPOINT = "after_fp3"
STATIC_FAMILY = "ablation"
STATIC_MODEL_NAME = "random_forest"
STATIC_FEATURE_GROUP = "base_plus_relative"
STATIC_TEMPORAL_POLICY = "uniform"
WEIGHTED_TEMPORAL_POLICY = "current_season_only_with_prior"
PREDICTION_TOLERANCE = 1e-9

ARTIFACTS_TO_INSPECT: tuple[str, ...] = (
    "champion_static_predictions.parquet",
    "champion_static_metrics.json",
    "champion_static_selection.parquet",
    "champion_stabilized_nested_guarded_predictions.parquet",
    "champion_stabilized_nested_guarded_metrics.json",
    "champion_stabilized_nested_guarded_selection.parquet",
    "champion_season_aware_nested_guarded_predictions.parquet",
    "champion_season_aware_nested_guarded_metrics.json",
    "champion_season_aware_nested_guarded_selection.parquet",
    "ablation_uniform_predictions.parquet",
    "ablation_uniform_metrics.json",
    "ablation_uniform_feature_groups.json",
    "ablation_current_season_only_with_prior_predictions.parquet",
    "ablation_current_season_only_with_prior_metrics.json",
    "ablation_current_season_only_with_prior_feature_groups.json",
    "ablation_predictions.parquet",
    "ablation_metrics.json",
    "ablation_feature_groups.json",
)

EXPECTED_STATIC_SOURCE_CONTRACT: dict[str, object] = {
    "source_artifact_kind": "ablation_predictions_snapshot",
    "source_artifact_path": "reports/metrics/ablation_uniform_predictions.parquet",
    "source_family": STATIC_FAMILY,
    "source_model_name": STATIC_MODEL_NAME,
    "source_feature_group": STATIC_FEATURE_GROUP,
    "source_temporal_weighting_policy": STATIC_TEMPORAL_POLICY,
    "source_strategy": "walk_forward",
    "target_checkpoint": FP3_CHECKPOINT,
}


@dataclass(frozen=True)
class ChampionSourceLineageSummary:
    """Paths and status produced by champion source-lineage diagnostics."""

    status: str
    summary_path: Path
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    missing_inputs: tuple[str, ...]
    generation_issues: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactSourceIdentity:
    """Exact source identity for a generated prediction artifact."""

    artifact_kind: str
    model_family: str | None
    model_name: str | None
    feature_group: str | None
    temporal_weighting_policy: str
    strategy: str = "walk_forward"
    dataset_scope: str = "2023_2024_2025"


def season_aware_rebuild_artifact_paths(metrics_dir: Path) -> dict[str, Path]:
    """Return the scoped artifact set managed by the season-aware rebuild workflow."""
    names = [
        "ablation_metrics.json",
        "ablation_predictions.parquet",
        "ablation_feature_groups.json",
        "ablation_uniform_metrics.json",
        "ablation_uniform_predictions.parquet",
        "ablation_uniform_feature_groups.json",
        "ablation_current_season_only_with_prior_metrics.json",
        "ablation_current_season_only_with_prior_predictions.parquet",
        "ablation_current_season_only_with_prior_feature_groups.json",
        "champion_static_metrics.json",
        "champion_static_predictions.parquet",
        "champion_static_selection.parquet",
        "champion_stabilized_nested_guarded_metrics.json",
        "champion_stabilized_nested_guarded_predictions.parquet",
        "champion_stabilized_nested_guarded_selection.parquet",
        "champion_season_aware_nested_guarded_metrics.json",
        "champion_season_aware_nested_guarded_predictions.parquet",
        "champion_season_aware_nested_guarded_selection.parquet",
        "champion_source_lineage_manifest.json",
        "champion_source_lineage_artifact_summary.csv",
        "champion_source_lineage_fold_comparison.csv",
        "champion_source_lineage_row_comparison.csv",
        "season_aware_candidate_audit_summary.json",
        "season_aware_candidate_eligibility_by_fold.csv",
        "season_aware_candidate_history_by_fold.csv",
        "season_aware_candidate_gate_failures.csv",
        "season_aware_candidate_alignment.csv",
        "season_aware_candidate_comparator_consistency.csv",
        "season_aware_policy_forensics_summary.json",
        "season_aware_policy_fold_reconstruction.csv",
        "season_aware_policy_event_counterfactual.csv",
        "season_aware_policy_selected_fold_analysis.csv",
        "season_aware_policy_switch_cases.csv",
        "season_aware_policy_guardrail_simulation.csv",
        "season_aware_policy_guardrail_event_level.csv",
        "season_aware_policy_guardrail_summary.json",
        "season_aware_rebuild_summary.json",
        "season_aware_rebuild_artifact_registry.csv",
        "season_aware_rebuild_validation.csv",
        "backtest_report.json",
        "portfolio_summary.json",
    ]
    return {name: metrics_dir / name for name in names}


def resolve_artifact_source(metrics_dir: Path, identity: ArtifactSourceIdentity) -> Path:
    """Resolve an exact source identity to its deterministic artifact path."""
    if identity.artifact_kind == "ablation_predictions":
        if identity.temporal_weighting_policy == STATIC_TEMPORAL_POLICY:
            return metrics_dir / "ablation_uniform_predictions.parquet"
        if identity.temporal_weighting_policy == WEIGHTED_TEMPORAL_POLICY:
            return metrics_dir / "ablation_current_season_only_with_prior_predictions.parquet"
    if identity.artifact_kind == "champion_predictions":
        mode = identity.model_family or ""
        return metrics_dir / f"champion_{mode}_predictions.parquet"
    raise KeyError(f"Unsupported artifact source identity: {identity}")


def build_season_aware_rebuild_registry(
    *,
    metrics_dir: Path,
    project_root: Path,
    config_path: Path | None = None,
) -> pd.DataFrame:
    """Build a compact registry of source identities used by the rebuild workflow."""
    identities = [
        ArtifactSourceIdentity(
            "ablation_predictions",
            STATIC_FAMILY,
            STATIC_MODEL_NAME,
            STATIC_FEATURE_GROUP,
            STATIC_TEMPORAL_POLICY,
        ),
        ArtifactSourceIdentity(
            "ablation_predictions",
            STATIC_FAMILY,
            STATIC_MODEL_NAME,
            STATIC_FEATURE_GROUP,
            WEIGHTED_TEMPORAL_POLICY,
        ),
        ArtifactSourceIdentity(
            "champion_predictions",
            "static",
            None,
            None,
            STATIC_TEMPORAL_POLICY,
        ),
        ArtifactSourceIdentity(
            "champion_predictions",
            "stabilized_nested_guarded",
            None,
            None,
            STATIC_TEMPORAL_POLICY,
        ),
        ArtifactSourceIdentity(
            "champion_predictions",
            "season_aware_nested_guarded",
            None,
            None,
            WEIGHTED_TEMPORAL_POLICY,
        ),
    ]
    config_signature = _file_hash(config_path)
    rows: list[dict[str, object]] = []
    for identity in identities:
        path = resolve_artifact_source(metrics_dir, identity)
        rows.append(
            {
                "artifact_kind": identity.artifact_kind,
                "model_family": identity.model_family,
                "model_name": identity.model_name,
                "feature_group": identity.feature_group,
                "temporal_weighting_policy": identity.temporal_weighting_policy,
                "strategy": identity.strategy,
                "dataset_scope": identity.dataset_scope,
                "artifact_path": _relative_path(path, project_root),
                "artifact_exists": path.is_file(),
                "artifact_content_signature": _file_hash(path),
                "config_signature": config_signature,
            }
        )
    latest_path = metrics_dir / "ablation_predictions.parquet"
    rows.append(
        {
            "artifact_kind": "ablation_predictions_latest",
            "model_family": STATIC_FAMILY,
            "model_name": STATIC_MODEL_NAME,
            "feature_group": STATIC_FEATURE_GROUP,
            "temporal_weighting_policy": "latest_run",
            "strategy": "walk_forward",
            "dataset_scope": "2023_2024_2025",
            "artifact_path": _relative_path(latest_path, project_root),
            "artifact_exists": latest_path.is_file(),
            "artifact_content_signature": _file_hash(latest_path),
            "config_signature": config_signature,
        }
    )
    return pd.DataFrame(rows)


def create_champion_source_lineage_report(
    config: DataConfig,
    model_config: ModelConfig,
    *,
    tolerance: float = PREDICTION_TOLERANCE,
) -> ChampionSourceLineageSummary:
    """Inspect saved artifacts and compare static FP3 against the uniform source contract."""
    metrics_dir = config.metrics_output_dir
    figures_dir = metrics_dir.parent / "figures"
    ensure_directory(metrics_dir)
    ensure_directory(figures_dir)

    artifact_summary = build_artifact_summary(
        metrics_dir=metrics_dir,
        project_root=config.project_root,
        config_path=config.project_root / "configs" / "model.yaml",
    )
    row_comparison = compare_static_to_uniform_ablation(metrics_dir, tolerance=tolerance)
    fold_comparison = build_fold_comparison(row_comparison, tolerance=tolerance)
    source_contract = build_static_source_contract(config, model_config)
    root_cause = classify_static_uniform_gap(metrics_dir, row_comparison, tolerance=tolerance)
    verification = build_static_source_verification(
        row_comparison=row_comparison,
        fold_comparison=fold_comparison,
        root_cause=root_cause,
        tolerance=tolerance,
    )

    artifact_summary_path = metrics_dir / "champion_source_lineage_artifact_summary.csv"
    fold_comparison_path = metrics_dir / "champion_source_lineage_fold_comparison.csv"
    row_comparison_path = metrics_dir / "champion_source_lineage_row_comparison.csv"
    artifact_summary.to_csv(artifact_summary_path, index=False)
    fold_comparison.to_csv(fold_comparison_path, index=False)
    row_comparison.to_csv(row_comparison_path, index=False)

    figure_paths, figure_issues = generate_source_lineage_figures(
        figures_dir=figures_dir,
        row_comparison=row_comparison,
        fold_comparison=fold_comparison,
        artifact_summary=artifact_summary,
        verification=verification,
    )
    manifest = build_source_lineage_manifest(
        artifact_summary=artifact_summary,
        fold_comparison=fold_comparison,
        row_comparison=row_comparison,
        source_contract=source_contract,
        root_cause=root_cause,
        verification=verification,
        generated_tables=[
            artifact_summary_path,
            fold_comparison_path,
            row_comparison_path,
        ],
        generated_figures=figure_paths,
        generation_issues=figure_issues,
        tolerance=tolerance,
    )
    summary_path = metrics_dir / "champion_source_lineage_manifest.json"
    _write_json(summary_path, manifest)

    missing = tuple(
        artifact_summary.loc[
            artifact_summary["artifact_exists"].eq(False),
            "artifact_name",
        ]
        .astype(str)
        .tolist()
    )
    return ChampionSourceLineageSummary(
        status=str(manifest["status"]),
        summary_path=summary_path,
        table_paths=(
            summary_path,
            artifact_summary_path,
            fold_comparison_path,
            row_comparison_path,
        ),
        figure_paths=tuple(figure_paths),
        missing_inputs=missing,
        generation_issues=tuple(figure_issues),
    )


def build_artifact_summary(
    *,
    metrics_dir: Path,
    project_root: Path,
    config_path: Path | None = None,
) -> pd.DataFrame:
    """Create one manifest row for each relevant saved artifact."""
    config_signature = _file_hash(config_path) if config_path and config_path.is_file() else None
    rows = [
        inspect_artifact(
            metrics_dir / name,
            project_root=project_root,
            config_signature=config_signature,
        )
        for name in ARTIFACTS_TO_INSPECT
    ]
    return pd.DataFrame(rows)


def inspect_artifact(
    path: Path,
    *,
    project_root: Path,
    config_signature: str | None = None,
) -> dict[str, object]:
    """Inspect a parquet or JSON artifact without assuming a specific schema."""
    exists = path.is_file()
    row: dict[str, object] = {
        "artifact_name": path.name,
        "artifact_path": _relative_path(path, project_root),
        "artifact_exists": exists,
        "file_size_bytes": int(path.stat().st_size) if exists else None,
        "modified_timestamp": _modified_timestamp(path) if exists else None,
        "schema_columns": [],
        "row_count": None,
        "unique_folds": None,
        "unique_events": None,
        "unique_checkpoints": None,
        "unique_drivers": None,
        "model_family": None,
        "model_name": None,
        "feature_group": None,
        "temporal_weighting_policy": None,
        "strategy": None,
        "min_events": None,
        "min_train_events": None,
        "target_column": "quali_gap_to_pole_sec",
        "dataset_path_or_identity": None,
        "feature_column_signature": None,
        "config_signature": config_signature,
        "random_state": None,
        "artifact_generation_metadata": {},
    }
    if not exists:
        return row
    try:
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            row.update(_parquet_artifact_metadata(frame))
        elif path.suffix == ".json":
            payload = _read_json(path)
            row.update(_json_artifact_metadata(payload))
        row["artifact_content_signature"] = _file_hash(path)
    except Exception as exc:
        row["artifact_generation_metadata"] = {"inspection_error": str(exc)}
    return row


def compare_static_to_uniform_ablation(
    metrics_dir: Path,
    *,
    tolerance: float = PREDICTION_TOLERANCE,
) -> pd.DataFrame:
    """Compare saved static FP3 champion rows to the intended uniform ablation RF rows."""
    static_path = metrics_dir / "champion_static_predictions.parquet"
    uniform_path = metrics_dir / "ablation_uniform_predictions.parquet"
    columns = _row_comparison_columns()
    if not static_path.is_file() or not uniform_path.is_file():
        return pd.DataFrame(columns=columns)
    static = _filter_static_fp3(pd.read_parquet(static_path))
    uniform = _filter_uniform_fp3_candidate(pd.read_parquet(uniform_path))
    if static.empty and uniform.empty:
        return pd.DataFrame(columns=columns)
    key_columns = _join_columns(static, uniform)
    static = _deduplicate_keys(static, key_columns)
    uniform = _deduplicate_keys(uniform, key_columns)
    static_rows = static.loc[
        :,
        [
            *key_columns,
            "event",
            "team",
            "quali_gap_to_pole_sec",
            "predicted_quali_gap_to_pole_sec",
        ],
    ].rename(columns={"predicted_quali_gap_to_pole_sec": "static_prediction_gap_sec"})
    uniform_rows = uniform.loc[
        :,
        [*key_columns, "predicted_quali_gap_to_pole_sec"],
    ].rename(columns={"predicted_quali_gap_to_pole_sec": "uniform_ablation_prediction_gap_sec"})
    merged = static_rows.merge(
        uniform_rows,
        on=key_columns,
        how="outer",
        indicator=True,
    )
    merged["row_match_status"] = merged["_merge"].map(
        {
            "both": "matched",
            "left_only": "static_only",
            "right_only": "uniform_only",
        }
    )
    actual = pd.to_numeric(merged.get("quali_gap_to_pole_sec"), errors="coerce")
    static_prediction = pd.to_numeric(merged.get("static_prediction_gap_sec"), errors="coerce")
    uniform_prediction = pd.to_numeric(
        merged.get("uniform_ablation_prediction_gap_sec"), errors="coerce"
    )
    merged["prediction_delta_sec"] = static_prediction - uniform_prediction
    merged["abs_prediction_delta_sec"] = merged["prediction_delta_sec"].abs()
    merged["actual_gap_sec"] = actual
    merged["static_abs_error_sec"] = (static_prediction - actual).abs()
    merged["uniform_ablation_abs_error_sec"] = (uniform_prediction - actual).abs()
    merged["error_delta_sec"] = (
        merged["uniform_ablation_abs_error_sec"] - merged["static_abs_error_sec"]
    )
    merged["prediction_exact_match"] = merged["abs_prediction_delta_sec"].eq(0)
    merged["prediction_tolerance_match"] = merged["abs_prediction_delta_sec"].le(tolerance)
    merged["mismatch_cause"] = merged.apply(_row_mismatch_cause, axis=1)
    for column in columns:
        if column not in merged:
            merged[column] = pd.NA
    return merged.loc[:, columns]


def build_fold_comparison(
    row_comparison: pd.DataFrame,
    *,
    tolerance: float = PREDICTION_TOLERANCE,
) -> pd.DataFrame:
    """Aggregate row-level static/uniform comparison by FP3 fold."""
    columns = _fold_comparison_columns()
    if row_comparison.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for keys, group in row_comparison.groupby(
        ["fold_id", "season", "event", "event_slug", "checkpoint"],
        dropna=False,
        sort=True,
    ):
        fold_id, season, event, event_slug, checkpoint = keys
        matched = group[group["row_match_status"].eq("matched")]
        static_only = int(group["row_match_status"].eq("static_only").sum())
        uniform_only = int(group["row_match_status"].eq("uniform_only").sum())
        rows.append(
            {
                "fold_id": fold_id,
                "season": season,
                "event": event,
                "event_slug": event_slug,
                "checkpoint": checkpoint,
                "row_match_count": int(len(matched)),
                "unmatched_static_rows": static_only,
                "unmatched_uniform_rows": uniform_only,
                "exact_prediction_match_rate": _mean_bool(
                    matched.get("prediction_exact_match", pd.Series(dtype=bool))
                ),
                "tolerance_prediction_match_rate": _mean_bool(
                    matched.get("prediction_tolerance_match", pd.Series(dtype=bool))
                ),
                "maximum_absolute_prediction_delta_sec": _max_numeric(
                    matched.get("abs_prediction_delta_sec", pd.Series(dtype=float))
                ),
                "static_mae_gap_sec": _mean_numeric(
                    matched.get("static_abs_error_sec", pd.Series(dtype=float))
                ),
                "uniform_ablation_mae_gap_sec": _mean_numeric(
                    matched.get("uniform_ablation_abs_error_sec", pd.Series(dtype=float))
                ),
                "mae_difference_uniform_minus_static_sec": _mean_numeric(
                    matched.get("error_delta_sec", pd.Series(dtype=float))
                ),
                "event_level_mae_difference_sec": _mean_numeric(
                    matched.get("error_delta_sec", pd.Series(dtype=float))
                ),
                "mismatch_cause": _fold_mismatch_cause(
                    matched,
                    static_only=static_only,
                    uniform_only=uniform_only,
                    tolerance=tolerance,
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_static_source_contract(
    config: DataConfig,
    model_config: ModelConfig,
) -> dict[str, object]:
    """Return the expected source identity for the configured static FP3 champion."""
    method = model_config.champion_policy.static.get(FP3_CHECKPOINT)
    configured = {
        "configured_family": method.family if method else None,
        "configured_model_name": method.model_name if method else None,
        "configured_feature_group": method.feature_group if method else None,
        "configured_temporal_weighting_policy": (
            method.temporal_weighting_policy if method else STATIC_TEMPORAL_POLICY
        )
        or STATIC_TEMPORAL_POLICY,
    }
    expected = dict(EXPECTED_STATIC_SOURCE_CONTRACT)
    expected["source_artifact_path"] = _relative_path(
        config.metrics_output_dir / "ablation_uniform_predictions.parquet",
        config.project_root,
    )
    return {
        "contract_version": "static_fp3_uniform_ablation_rf_v1",
        "description": (
            "Static FP3 champion should be sourced from uniform ablation "
            "random_forest/base_plus_relative predictions."
        ),
        "expected": expected,
        "configured_static_fp3": configured,
        "configured_matches_expected": bool(
            configured["configured_family"] == STATIC_FAMILY
            and configured["configured_model_name"] == STATIC_MODEL_NAME
            and configured["configured_feature_group"] == STATIC_FEATURE_GROUP
            and configured["configured_temporal_weighting_policy"] == STATIC_TEMPORAL_POLICY
        ),
    }


def classify_static_uniform_gap(
    metrics_dir: Path,
    row_comparison: pd.DataFrame,
    *,
    tolerance: float = PREDICTION_TOLERANCE,
) -> dict[str, object]:
    """Classify the observed static/uniform gap using only saved-artifact evidence."""
    if row_comparison.empty:
        return {
            "root_cause_classification": "missing_artifacts",
            "evidence": ["Static or uniform ablation prediction artifact is missing."],
            "confidence": "high",
        }
    unmatched_static = int(row_comparison["row_match_status"].eq("static_only").sum())
    unmatched_uniform = int(row_comparison["row_match_status"].eq("uniform_only").sum())
    matched = row_comparison[row_comparison["row_match_status"].eq("matched")]
    tolerance_match_rate = _mean_bool(matched["prediction_tolerance_match"])
    if unmatched_static == 0 and unmatched_uniform == 0 and tolerance_match_rate == 1.0:
        return {
            "root_cause_classification": "none_verified",
            "evidence": ["Static FP3 and uniform ablation predictions match on all aligned rows."],
            "confidence": "high",
        }
    evidence = [
        f"matched_rows={len(matched)}",
        f"unmatched_static_rows={unmatched_static}",
        f"unmatched_uniform_rows={unmatched_uniform}",
        f"tolerance_match_rate={tolerance_match_rate}",
    ]
    latest_match = _compare_static_to_artifact(
        metrics_dir / "champion_static_predictions.parquet",
        metrics_dir / "ablation_predictions.parquet",
        tolerance=tolerance,
    )
    weighted_match = _compare_static_to_artifact(
        metrics_dir / "champion_static_predictions.parquet",
        metrics_dir / "ablation_current_season_only_with_prior_predictions.parquet",
        tolerance=tolerance,
    )
    if latest_match.get("tolerance_match_rate") == 1.0:
        evidence.append("Static FP3 rows exactly match ablation_predictions.parquet.")
        evidence.append(
            "ablation_predictions.parquet temporal policies: "
            f"{latest_match.get('temporal_weighting_policies')}"
        )
    if weighted_match.get("tolerance_match_rate") == 1.0:
        evidence.append(
            "Static FP3 rows exactly match "
            "ablation_current_season_only_with_prior_predictions.parquet."
        )
    if latest_match.get("tolerance_match_rate") == 1.0 and WEIGHTED_TEMPORAL_POLICY in set(
        latest_match.get("temporal_weighting_policies", [])
    ):
        classification = "stale_artifact_generation_order"
        evidence.append(
            "The canonical ablation_predictions.parquet artifact is a latest-run weighted "
            "snapshot, while the static contract expects ablation_uniform_predictions.parquet."
        )
    elif unmatched_static or unmatched_uniform:
        classification = "different_rows"
    else:
        classification = "different_preprocessing_or_model_configuration"
    return {
        "root_cause_classification": classification,
        "evidence": evidence,
        "confidence": "high" if classification == "stale_artifact_generation_order" else "medium",
        "latest_ablation_match": latest_match,
        "weighted_ablation_match": weighted_match,
    }


def build_static_source_verification(
    *,
    row_comparison: pd.DataFrame,
    fold_comparison: pd.DataFrame,
    root_cause: dict[str, object],
    tolerance: float = PREDICTION_TOLERANCE,
) -> dict[str, object]:
    """Summarize whether saved static FP3 rows satisfy the expected source contract."""
    if row_comparison.empty:
        return {
            "static_source_verified": False,
            "static_source_verification_reason": "missing_static_or_uniform_artifact",
            "counterfactual_comparison_valid": False,
            "counterfactual_invalid_reason": "missing_static_source_verification_inputs",
        }
    matched = row_comparison[row_comparison["row_match_status"].eq("matched")]
    unmatched_static = int(row_comparison["row_match_status"].eq("static_only").sum())
    unmatched_uniform = int(row_comparison["row_match_status"].eq("uniform_only").sum())
    tolerance_match_rate = _mean_bool(matched["prediction_tolerance_match"])
    max_delta = _max_numeric(matched["abs_prediction_delta_sec"])
    verified = bool(
        unmatched_static == 0 and unmatched_uniform == 0 and tolerance_match_rate == 1.0
    )
    reason = (
        "static_matches_uniform_ablation_contract"
        if verified
        else f"static_uniform_prediction_mismatch:{root_cause.get('root_cause_classification')}"
    )
    return {
        "static_source_verified": verified,
        "static_source_verification_reason": reason,
        "counterfactual_comparison_valid": verified,
        "counterfactual_invalid_reason": None
        if verified
        else "saved_static_predictions_do_not_match_uniform_ablation_source_contract",
        "row_match_count": int(len(matched)),
        "unmatched_static_rows": unmatched_static,
        "unmatched_uniform_rows": unmatched_uniform,
        "exact_prediction_match_rate": _mean_bool(matched["prediction_exact_match"]),
        "tolerance_prediction_match_rate": tolerance_match_rate,
        "maximum_absolute_prediction_delta_sec": max_delta,
        "static_mae_gap_sec": _mean_numeric(matched["static_abs_error_sec"]),
        "uniform_ablation_mae_gap_sec": _mean_numeric(matched["uniform_ablation_abs_error_sec"]),
        "mae_difference_uniform_minus_static_sec": _mean_numeric(matched["error_delta_sec"]),
        "folds_compared": int(len(fold_comparison)),
        "comparison_tolerance_sec": tolerance,
    }


def load_static_source_verification(metrics_dir: Path) -> dict[str, object]:
    """Load static source verification from the lineage manifest if available."""
    path = metrics_dir / "champion_source_lineage_manifest.json"
    if not path.is_file():
        return {
            "static_source_verified": False,
            "static_source_verification_reason": "source_lineage_manifest_missing",
            "counterfactual_comparison_valid": False,
            "counterfactual_invalid_reason": "source_lineage_manifest_missing",
        }
    payload = _read_json(path)
    verification = payload.get("static_source_verification", {})
    if isinstance(verification, dict):
        return verification
    return {
        "static_source_verified": False,
        "static_source_verification_reason": "source_lineage_manifest_invalid",
        "counterfactual_comparison_valid": False,
        "counterfactual_invalid_reason": "source_lineage_manifest_invalid",
    }


def build_source_lineage_manifest(
    *,
    artifact_summary: pd.DataFrame,
    fold_comparison: pd.DataFrame,
    row_comparison: pd.DataFrame,
    source_contract: dict[str, object],
    root_cause: dict[str, object],
    verification: dict[str, object],
    generated_tables: list[Path],
    generated_figures: list[Path],
    generation_issues: list[str],
    tolerance: float,
) -> dict[str, object]:
    """Build the JSON manifest for champion source lineage."""
    missing = artifact_summary.loc[
        artifact_summary["artifact_exists"].eq(False),
        "artifact_name",
    ].astype(str)
    return {
        "status": "complete" if not row_comparison.empty else "partial",
        "static_source_contract": source_contract,
        "static_source_verification": verification,
        "static_vs_uniform_artifact_comparison": {
            "row_count": int(len(row_comparison)),
            "fold_count": int(len(fold_comparison)),
            "tolerance_sec": tolerance,
            "root_cause": root_cause,
        },
        "root_cause_classification": root_cause.get("root_cause_classification"),
        "m26_counterfactual_conclusions_valid": bool(
            verification.get("counterfactual_comparison_valid")
        ),
        "m26_counterfactual_conclusion_note": (
            "M26 static counterfactual labels are valid."
            if verification.get("counterfactual_comparison_valid")
            else "M26 raw deltas remain useful for debugging, but definitive harmful/beneficial/"
            "neutral labels versus static are invalid until static source lineage is verified."
        ),
        "clean_rebuild_workflow": clean_rebuild_workflow(),
        "artifact_manifest": artifact_summary.to_dict("records"),
        "missing_inputs": sorted(missing.tolist()),
        "generated_outputs": {
            "metrics": [_relative_report_path(path) for path in generated_tables],
            "figures": [_relative_report_path(path) for path in generated_figures],
        },
        "generation_issues": generation_issues,
        "generated_at": _utc_now(),
    }


def clean_rebuild_workflow() -> dict[str, object]:
    """Return the documented deterministic rebuild workflow and scoped artifact set."""
    champion_base = "python -m f1_prediction.cli champion-backtest --strategy walk_forward"
    commands = [
        "python -m f1_prediction.cli build-season-dataset "
        "--seasons 2023 2024 2025 --preset conventional",
        "python -m f1_prediction.cli ablation-backtest --strategy walk_forward "
        "--temporal-weighting uniform --min-events 10 --min-train-events 5",
        "python -m f1_prediction.cli ablation-backtest --strategy walk_forward "
        "--temporal-weighting current_season_only_with_prior "
        "--min-events 10 --min-train-events 5",
        f"{champion_base} --selection-mode static --min-events 10 --min-train-events 5",
        f"{champion_base} --selection-mode stabilized_nested_guarded "
        "--uncertainty conformal_predicted_gap_bucket --min-events 10 --min-train-events 5",
        f"{champion_base} --selection-mode season_aware_nested_guarded "
        "--uncertainty conformal_predicted_gap_bucket --min-events 10 --min-train-events 5",
        "python -m f1_prediction.cli champion-source-lineage",
        "python -m f1_prediction.cli season-aware-policy-forensics",
        "python -m f1_prediction.cli backtest-report",
        "python -m f1_prediction.cli portfolio-report",
    ]
    scoped_artifacts = [
        "reports/metrics/ablation_metrics.json",
        "reports/metrics/ablation_predictions.parquet",
        "reports/metrics/ablation_uniform_metrics.json",
        "reports/metrics/ablation_uniform_predictions.parquet",
        "reports/metrics/ablation_current_season_only_with_prior_metrics.json",
        "reports/metrics/ablation_current_season_only_with_prior_predictions.parquet",
        "reports/metrics/champion_static_metrics.json",
        "reports/metrics/champion_static_predictions.parquet",
        "reports/metrics/champion_static_selection.parquet",
        "reports/metrics/champion_stabilized_nested_guarded_metrics.json",
        "reports/metrics/champion_stabilized_nested_guarded_predictions.parquet",
        "reports/metrics/champion_stabilized_nested_guarded_selection.parquet",
        "reports/metrics/champion_season_aware_nested_guarded_metrics.json",
        "reports/metrics/champion_season_aware_nested_guarded_predictions.parquet",
        "reports/metrics/champion_season_aware_nested_guarded_selection.parquet",
        "reports/metrics/champion_source_lineage_manifest.json",
        "reports/metrics/season_aware_policy_forensics_summary.json",
    ]
    return {
        "description": (
            "Deterministic rebuild path for source-lineage validation. Do not delete arbitrary "
            "files; clean only the scoped generated artifacts if a refresh is intentionally "
            "requested."
        ),
        "commands": commands,
        "scoped_generated_artifacts": scoped_artifacts,
        "preserve_generated_artifact_ignore_rules": True,
        "requires_fastf1_for_dataset_rebuild": True,
    }


def generate_source_lineage_figures(
    *,
    figures_dir: Path,
    row_comparison: pd.DataFrame,
    fold_comparison: pd.DataFrame,
    artifact_summary: pd.DataFrame,
    verification: dict[str, object],
) -> tuple[list[Path], list[str]]:
    """Generate static matplotlib figures for champion source-lineage diagnostics."""
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
            "champion_source_lineage_prediction_delta_distribution.png",
            lambda path: _plot_prediction_delta_distribution(plt, row_comparison, path),
        ),
        (
            "champion_source_lineage_mae_by_artifact.png",
            lambda path: _plot_mae_by_artifact(plt, fold_comparison, path),
        ),
        (
            "champion_source_lineage_fold_match_rate.png",
            lambda path: _plot_fold_match_rate(plt, fold_comparison, path),
        ),
        (
            "champion_source_lineage_event_mae_delta.png",
            lambda path: _plot_event_mae_delta(plt, fold_comparison, path),
        ),
        (
            "champion_source_lineage_contract_status.png",
            lambda path: _plot_contract_status(plt, artifact_summary, verification, path),
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


def _parquet_artifact_metadata(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "schema_columns": list(frame.columns),
        "row_count": int(len(frame)),
        "unique_folds": _nunique(frame, "fold_id"),
        "unique_events": _unique_event_count(frame),
        "unique_checkpoints": _nunique(frame, "checkpoint"),
        "unique_drivers": _driver_count(frame),
        "model_family": _joined_unique(
            frame,
            ("selected_family", "candidate_family", "prediction_type"),
        ),
        "model_name": _joined_unique(frame, ("selected_model_name", "model_name")),
        "feature_group": _joined_unique(frame, ("selected_feature_group", "feature_group")),
        "temporal_weighting_policy": _joined_unique(
            frame,
            ("selected_temporal_weighting_policy", "temporal_weighting_policy"),
        ),
        "strategy": _joined_unique(frame, ("strategy",)),
        "feature_column_signature": _column_signature(frame.columns),
        "random_state": None,
        "artifact_generation_metadata": {
            "prediction_types": _unique_values(frame, "prediction_type"),
            "selection_modes": _unique_values(frame, "selection_mode"),
        },
    }


def _json_artifact_metadata(payload: dict[str, Any]) -> dict[str, object]:
    return {
        "schema_columns": list(payload.keys()),
        "row_count": None,
        "unique_folds": payload.get("n_folds_successful"),
        "unique_events": payload.get("n_events"),
        "unique_checkpoints": len(payload.get("checkpoints", []))
        if isinstance(payload.get("checkpoints"), list)
        else None,
        "model_family": payload.get("selection_mode"),
        "model_name": _joined_payload_values(payload.get("models")),
        "feature_group": _joined_payload_values(payload.get("feature_groups")),
        "temporal_weighting_policy": payload.get("temporal_weighting_policy"),
        "strategy": payload.get("strategy"),
        "min_events": payload.get("min_events"),
        "min_train_events": payload.get("min_train_events"),
        "dataset_path_or_identity": payload.get("dataset_path") or payload.get("dataset_signature"),
        "random_state": _random_state_from_payload(payload),
        "artifact_generation_metadata": {
            "status": payload.get("status"),
            "created_at_utc": payload.get("created_at_utc"),
            "uncertainty_method": payload.get("uncertainty_method"),
        },
    }


def _filter_static_fp3(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    feature_group = frame.get("selected_feature_group", pd.Series(index=frame.index, dtype=str))
    return frame[
        frame["checkpoint"].astype(str).eq(FP3_CHECKPOINT)
        & frame.get("selected_family", pd.Series(index=frame.index, dtype=str))
        .astype(str)
        .eq(STATIC_FAMILY)
        & frame.get("selected_model_name", pd.Series(index=frame.index, dtype=str))
        .astype(str)
        .eq(STATIC_MODEL_NAME)
        & feature_group.fillna("").astype(str).eq(STATIC_FEATURE_GROUP)
    ].copy()


def _filter_uniform_fp3_candidate(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    policy = frame.get("temporal_weighting_policy", pd.Series("uniform", index=frame.index))
    return frame[
        frame["checkpoint"].astype(str).eq(FP3_CHECKPOINT)
        & frame.get("prediction_type", pd.Series(index=frame.index, dtype=str))
        .astype(str)
        .eq("tabular")
        & frame.get("model_name", pd.Series(index=frame.index, dtype=str))
        .astype(str)
        .eq(STATIC_MODEL_NAME)
        & frame.get("feature_group", pd.Series(index=frame.index, dtype=str))
        .astype(str)
        .eq(STATIC_FEATURE_GROUP)
        & policy.fillna("uniform").astype(str).eq(STATIC_TEMPORAL_POLICY)
    ].copy()


def _filter_fp3_candidate_any_policy(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[
        frame["checkpoint"].astype(str).eq(FP3_CHECKPOINT)
        & frame.get("prediction_type", pd.Series(index=frame.index, dtype=str))
        .astype(str)
        .eq("tabular")
        & frame.get("model_name", pd.Series(index=frame.index, dtype=str))
        .astype(str)
        .eq(STATIC_MODEL_NAME)
        & frame.get("feature_group", pd.Series(index=frame.index, dtype=str))
        .astype(str)
        .eq(STATIC_FEATURE_GROUP)
    ].copy()


def _compare_static_to_artifact(
    static_path: Path,
    artifact_path: Path,
    *,
    tolerance: float,
) -> dict[str, object]:
    if not static_path.is_file() or not artifact_path.is_file():
        return {"artifact_exists": artifact_path.is_file(), "tolerance_match_rate": None}
    static = _filter_static_fp3(pd.read_parquet(static_path))
    candidate = _filter_fp3_candidate_any_policy(pd.read_parquet(artifact_path))
    if static.empty or candidate.empty:
        return {
            "artifact_exists": True,
            "candidate_rows": int(len(candidate)),
            "tolerance_match_rate": None,
        }
    keys = _join_columns(static, candidate)
    merged = _deduplicate_keys(static, keys).merge(
        _deduplicate_keys(candidate, keys),
        on=keys,
        how="inner",
        suffixes=("_static", "_candidate"),
    )
    if merged.empty:
        return {"artifact_exists": True, "candidate_rows": int(len(candidate)), "matched_rows": 0}
    delta = (
        pd.to_numeric(merged["predicted_quali_gap_to_pole_sec_static"], errors="coerce")
        - pd.to_numeric(merged["predicted_quali_gap_to_pole_sec_candidate"], errors="coerce")
    ).abs()
    return {
        "artifact_exists": True,
        "candidate_rows": int(len(candidate)),
        "matched_rows": int(len(merged)),
        "tolerance_match_rate": float(delta.le(tolerance).mean()),
        "maximum_absolute_prediction_delta_sec": float(delta.max()),
        "temporal_weighting_policies": _unique_values(candidate, "temporal_weighting_policy"),
    }


def _join_columns(first: pd.DataFrame, second: pd.DataFrame) -> list[str]:
    driver_column = (
        "driver" if "driver" in first.columns and "driver" in second.columns else "driver_key"
    )
    return ["fold_id", "season", "event_slug", "checkpoint", driver_column]


def _deduplicate_keys(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return frame.drop_duplicates(keys, keep="last").copy()


def _row_mismatch_cause(row: pd.Series) -> str:
    status = str(row.get("row_match_status"))
    if status == "static_only":
        return "different_rows"
    if status == "uniform_only":
        return "different_rows"
    if bool(row.get("prediction_tolerance_match", False)):
        return "none"
    return "different_preprocessing_or_model_configuration"


def _fold_mismatch_cause(
    matched: pd.DataFrame,
    *,
    static_only: int,
    uniform_only: int,
    tolerance: float,
) -> str:
    if static_only or uniform_only:
        return "different_rows"
    if matched.empty:
        return "different_fold_assignment"
    if bool(matched["abs_prediction_delta_sec"].le(tolerance).all()):
        return "none"
    return "different_preprocessing_or_model_configuration"


def _plot_prediction_delta_distribution(plt: Any, frame: pd.DataFrame, path: Path) -> bool:
    matched = frame[frame["row_match_status"].eq("matched")] if not frame.empty else frame
    values = pd.to_numeric(matched.get("prediction_delta_sec"), errors="coerce").dropna()
    if values.empty:
        return False
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=30, color="#4c78a8", edgecolor="white")
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_title("Static vs Uniform Ablation Prediction Delta")
    ax.set_xlabel("Static prediction - uniform ablation prediction (sec)")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_mae_by_artifact(plt: Any, frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty:
        return False
    static = _mean_numeric(frame["static_mae_gap_sec"])
    uniform = _mean_numeric(frame["uniform_ablation_mae_gap_sec"])
    if static is None or uniform is None:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["static champion", "uniform ablation"], [static, uniform], color=["#4c78a8", "#f58518"])
    ax.set_title("FP3 MAE By Artifact")
    ax.set_ylabel("MAE gap (sec)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_fold_match_rate(plt: Any, frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty:
        return False
    plot = frame.sort_values("fold_id")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        plot["fold_id"].astype(str),
        pd.to_numeric(plot["tolerance_prediction_match_rate"], errors="coerce").fillna(0),
        color="#54a24b",
    )
    ax.set_title("Static Source Match Rate By Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Tolerance match rate")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_event_mae_delta(plt: Any, frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty:
        return False
    plot = frame.sort_values("mae_difference_uniform_minus_static_sec", ascending=False)
    values = pd.to_numeric(plot["mae_difference_uniform_minus_static_sec"], errors="coerce")
    if values.dropna().empty:
        return False
    labels = plot["season"].astype(str) + " " + plot["event"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, color="#e45756")
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_title("Uniform Ablation - Static MAE Delta By Event")
    ax.set_ylabel("Delta MAE (sec)")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_contract_status(
    plt: Any,
    artifact_summary: pd.DataFrame,
    verification: dict[str, object],
    path: Path,
) -> bool:
    if artifact_summary.empty:
        return False
    labels = ["artifact inputs", "static source verified"]
    inputs_present = float(artifact_summary["artifact_exists"].fillna(False).mean())
    verified = 1.0 if verification.get("static_source_verified") else 0.0
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, [inputs_present, verified], color=["#72b7b2", "#b279a2"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Source Contract Status")
    ax.set_ylabel("Status rate")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _row_comparison_columns() -> list[str]:
    return [
        "fold_id",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "driver",
        "driver_key",
        "team",
        "row_match_status",
        "static_prediction_gap_sec",
        "uniform_ablation_prediction_gap_sec",
        "prediction_delta_sec",
        "abs_prediction_delta_sec",
        "actual_gap_sec",
        "static_abs_error_sec",
        "uniform_ablation_abs_error_sec",
        "error_delta_sec",
        "prediction_exact_match",
        "prediction_tolerance_match",
        "mismatch_cause",
    ]


def _fold_comparison_columns() -> list[str]:
    return [
        "fold_id",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "row_match_count",
        "unmatched_static_rows",
        "unmatched_uniform_rows",
        "exact_prediction_match_rate",
        "tolerance_prediction_match_rate",
        "maximum_absolute_prediction_delta_sec",
        "static_mae_gap_sec",
        "uniform_ablation_mae_gap_sec",
        "mae_difference_uniform_minus_static_sec",
        "event_level_mae_difference_sec",
        "mismatch_cause",
    ]


def _nunique(frame: pd.DataFrame, column: str) -> int | None:
    return int(frame[column].nunique(dropna=True)) if column in frame else None


def _unique_event_count(frame: pd.DataFrame) -> int | None:
    if "test_event" in frame:
        return int(frame["test_event"].nunique(dropna=True))
    if {"season", "event_slug"} <= set(frame.columns):
        return int((frame["season"].astype(str) + "/" + frame["event_slug"].astype(str)).nunique())
    return _nunique(frame, "event")


def _driver_count(frame: pd.DataFrame) -> int | None:
    if "driver" in frame:
        return int(frame["driver"].nunique(dropna=True))
    if "driver_key" in frame:
        return int(frame["driver_key"].nunique(dropna=True))
    return None


def _joined_unique(frame: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    values: list[str] = []
    for column in columns:
        if column in frame:
            values.extend(frame[column].dropna().astype(str).unique().tolist())
    unique = sorted({value for value in values if value and value != "<NA>"})
    return "|".join(unique) if unique else None


def _unique_values(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame:
        return []
    return sorted(frame[column].dropna().astype(str).unique().tolist())


def _column_signature(columns: Any) -> str:
    return hashlib.sha256("\n".join(str(column) for column in columns).encode("utf-8")).hexdigest()


def _file_hash(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _modified_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _relative_report_path(path: Path) -> str:
    parts = path.parts
    if "reports" in parts:
        return Path(*parts[parts.index("reports") :]).as_posix()
    return path.as_posix()


def _joined_payload_values(value: object) -> str | None:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    if isinstance(value, dict):
        return "|".join(str(key) for key in value)
    return str(value) if value is not None else None


def _random_state_from_payload(payload: dict[str, Any]) -> object:
    if "random_state" in payload:
        return payload["random_state"]
    config = payload.get("temporal_weighting_config", {})
    if isinstance(config, dict):
        return config.get("random_state")
    return None


def _mean_bool(series: pd.Series) -> float | None:
    values = series.dropna()
    return float(values.astype(bool).mean()) if not values.empty else None


def _mean_numeric(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _max_numeric(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.max()) if not values.empty else None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(_json_ready(payload), output_file, indent=2, allow_nan=False)
        output_file.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value
