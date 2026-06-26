"""Artifact-based eligibility audit for true prospective replay."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig, ModelConfig
from f1_prediction.modeling.prospective_policy_evaluation import build_frozen_policy_profiles
from f1_prediction.modeling.prospective_replay import FP3_CHECKPOINT, align_candidate_default
from f1_prediction.utils.paths import ensure_directory

POLICY_PROFILES: tuple[str, ...] = (
    "static_baseline",
    "guarded_baseline",
    "season_aware_frozen",
)
CURRENT_POLICY = "current_season_only_with_prior"
UNIFORM_POLICY = "uniform"
JOIN_COLUMNS: tuple[str, ...] = ("fold_id", "season", "event_slug", "checkpoint", "driver")


@dataclass(frozen=True)
class ProspectiveReplayEligibilityAuditSummary:
    """Paths and issue counts produced by the replay eligibility audit."""

    status: str
    summary_path: Path
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    missing_inputs: tuple[str, ...]
    generation_issues: tuple[str, ...]


def create_prospective_replay_eligibility_audit_report(
    config: DataConfig,
    model_config: ModelConfig,
) -> ProspectiveReplayEligibilityAuditSummary:
    """Create artifact-only true replay eligibility audit outputs."""
    metrics_dir = config.metrics_output_dir
    figures_dir = metrics_dir.parent / "figures"
    ensure_directory(metrics_dir)
    ensure_directory(figures_dir)

    artifacts = load_replay_eligibility_artifacts(metrics_dir)
    profiles = build_frozen_policy_profiles(
        model_config,
        profile_names=POLICY_PROFILES,
        uncertainty="conformal_predicted_gap_bucket",
    )
    ledger = build_candidate_evidence_ledger(artifacts, profiles)
    by_event = build_eligibility_by_event(ledger)
    feasibility = build_gate_feasibility(ledger)
    failures = build_gate_failure_summary(ledger)
    availability = build_candidate_availability_comparison(ledger)
    artifact_comparison = build_artifact_driven_eligibility_comparison(
        ledger,
        artifacts["artifact_selection"],
    )
    consistency = build_live_selection_consistency(ledger)

    table_frames = {
        "prospective_replay_eligibility_by_event.csv": by_event,
        "prospective_replay_candidate_evidence_ledger.csv": ledger,
        "prospective_replay_gate_feasibility.csv": feasibility,
        "prospective_replay_gate_failure_summary.csv": failures,
        "prospective_replay_candidate_availability_comparison.csv": availability,
        "prospective_replay_vs_artifact_driven_eligibility.csv": artifact_comparison,
        "prospective_replay_live_selection_consistency.csv": consistency,
    }
    table_paths: list[Path] = []
    for filename, frame in table_frames.items():
        path = metrics_dir / filename
        frame.to_csv(path, index=False)
        table_paths.append(path)

    figure_paths, figure_issues = generate_eligibility_figures(
        figures_dir=figures_dir,
        ledger=ledger,
        feasibility=feasibility,
        failures=failures,
        availability=availability,
        artifact_comparison=artifact_comparison,
    )
    summary_payload = build_eligibility_summary_payload(
        artifacts=artifacts,
        ledger=ledger,
        feasibility=feasibility,
        failures=failures,
        artifact_comparison=artifact_comparison,
        consistency=consistency,
        figure_paths=figure_paths,
        figure_issues=figure_issues,
    )
    summary_path = metrics_dir / "prospective_replay_eligibility_audit_summary.json"
    _write_json(summary_path, summary_payload)
    return ProspectiveReplayEligibilityAuditSummary(
        status=str(summary_payload["status"]),
        summary_path=summary_path,
        table_paths=tuple(table_paths),
        figure_paths=tuple(figure_paths),
        missing_inputs=tuple(summary_payload["missing_inputs"]),
        generation_issues=tuple(summary_payload["generation_issues"]),
    )


def load_replay_eligibility_artifacts(metrics_dir: Path) -> dict[str, Any]:
    """Load saved replay/prospective artifacts without retraining."""
    required_files = {
        "prospective_replay_summary.json": metrics_dir / "prospective_replay_summary.json",
        "prospective_replay_selection_log.csv": (
            metrics_dir / "prospective_replay_selection_log.csv"
        ),
        "prospective_replay_training_manifest.csv": (
            metrics_dir / "prospective_replay_training_manifest.csv"
        ),
        "prospective_replay_leakage_audit.csv": (
            metrics_dir / "prospective_replay_leakage_audit.csv"
        ),
    }
    optional_files = {
        "prospective_replay_event_comparison.csv": (
            metrics_dir / "prospective_replay_event_comparison.csv"
        ),
        "prospective_replay_vs_artifact_driven.csv": (
            metrics_dir / "prospective_replay_vs_artifact_driven.csv"
        ),
        "prospective_policy_summary.json": metrics_dir / "prospective_policy_summary.json",
        "prospective_policy_selection_log.csv": (
            metrics_dir / "prospective_policy_selection_log.csv"
        ),
    }
    missing = [name for name, path in required_files.items() if not path.is_file()]
    replay_summary = _read_json_if_exists(required_files["prospective_replay_summary.json"]) or {}
    prediction_paths = prediction_paths_from_summary(metrics_dir, replay_summary)
    predictions = load_prediction_artifacts(prediction_paths)
    missing.extend(path.name for path in prediction_paths if not path.is_file())
    return {
        "summary": replay_summary,
        "selection": _read_csv_if_exists(required_files["prospective_replay_selection_log.csv"]),
        "manifest": _read_csv_if_exists(required_files["prospective_replay_training_manifest.csv"]),
        "leakage": _read_csv_if_exists(required_files["prospective_replay_leakage_audit.csv"]),
        "event_comparison": _read_csv_if_exists(
            optional_files["prospective_replay_event_comparison.csv"]
        ),
        "vs_artifact": _read_csv_if_exists(
            optional_files["prospective_replay_vs_artifact_driven.csv"]
        ),
        "artifact_summary": _read_json_if_exists(optional_files["prospective_policy_summary.json"])
        or {},
        "artifact_selection": _read_csv_if_exists(
            optional_files["prospective_policy_selection_log.csv"]
        ),
        "predictions": predictions,
        "prediction_paths": prediction_paths,
        "inputs_available": {
            **{name: path.is_file() for name, path in required_files.items()},
            **{name: path.is_file() for name, path in optional_files.items()},
            **{path.name: path.is_file() for path in prediction_paths},
        },
        "missing_inputs": missing,
    }


def prediction_paths_from_summary(metrics_dir: Path, summary: dict[str, Any]) -> list[Path]:
    """Return split-specific replay prediction paths from summary or glob fallback."""
    paths: list[Path] = []
    for split in summary.get("splits", []) if isinstance(summary.get("splits"), list) else []:
        artifact_paths = split.get("artifact_paths", {})
        prediction_path = (
            artifact_paths.get("predictions") if isinstance(artifact_paths, dict) else None
        )
        if prediction_path:
            path = Path(str(prediction_path))
            paths.append(path if path.is_absolute() else metrics_dir.parent.parent / path)
    if not paths:
        paths = sorted(metrics_dir.glob("prospective_replay_train_*_test_*_predictions.parquet"))
    return list(dict.fromkeys(paths))


def load_prediction_artifacts(paths: list[Path]) -> pd.DataFrame:
    """Load split-specific replay predictions."""
    frames = [pd.read_parquet(path) for path in paths if path.is_file()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_candidate_evidence_ledger(
    artifacts: dict[str, Any],
    profiles: dict[str, Any],
) -> pd.DataFrame:
    """Build the FP3 event/profile-level evidence ledger."""
    selection = artifacts["selection"]
    manifest = artifacts["manifest"]
    leakage = artifacts["leakage"]
    predictions = artifacts["predictions"]
    columns = _ledger_columns()
    if selection.empty:
        return pd.DataFrame(columns=columns)
    fp3_selection = selection[selection["checkpoint"].astype(str).eq(FP3_CHECKPOINT)].copy()
    rows: list[dict[str, object]] = []
    for split_name, split_rows in fp3_selection.groupby("prospective_split", sort=False):
        event_order = event_order_from_selection(split_rows)
        split_predictions = predictions[
            predictions.get("prospective_split", pd.Series(dtype=str))
            .astype(str)
            .eq(str(split_name))
        ].copy()
        split_manifest = manifest_for_split(manifest, split_rows)
        split_leakage = leakage_for_split(leakage, split_manifest)
        for _, selection_row in split_rows.iterrows():
            if str(selection_row.get("policy_profile")) not in POLICY_PROFILES:
                continue
            rows.append(
                ledger_row_for_selection(
                    selection_row=selection_row,
                    event_order=event_order,
                    predictions=split_predictions,
                    manifest=split_manifest,
                    leakage=split_leakage,
                    profile=profiles.get(str(selection_row.get("policy_profile"))),
                )
            )
    return pd.DataFrame(rows, columns=columns)


def ledger_row_for_selection(
    *,
    selection_row: pd.Series,
    event_order: list[str],
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    leakage: pd.DataFrame,
    profile: Any,
) -> dict[str, object]:
    """Build one event/profile evidence ledger row."""
    split_name = str(selection_row.get("prospective_split"))
    profile_name = str(selection_row.get("policy_profile"))
    event_key = event_key_from_values(selection_row.get("season"), selection_row.get("event_slug"))
    prior_keys = prior_event_keys(event_key, event_order)
    current_predictions = filter_event_predictions(predictions, selection_row)
    prior_predictions = predictions[
        predictions.get("prospective_event_id", pd.Series(dtype=str)).astype(str).isin(prior_keys)
    ].copy()
    current_manifest = manifest[
        manifest["test_event"].astype(str).eq(event_key)
        & manifest["policy_profile"]
        .astype(str)
        .eq("season_aware_frozen" if profile_name == "season_aware_frozen" else "static_baseline")
    ].copy()
    default_manifest = manifest[
        manifest["test_event"].astype(str).eq(event_key)
        & manifest["policy_profile"].astype(str).eq("static_baseline")
    ].copy()
    candidate_prior = prior_source_rows(prior_predictions, profile_name, CURRENT_POLICY)
    default_prior = prior_source_rows(prior_predictions, profile_name, UNIFORM_POLICY)
    aligned = (
        align_candidate_default(candidate_prior, default_prior)
        if not candidate_prior.empty and not default_prior.empty
        else pd.DataFrame()
    )
    current_candidate_persisted = current_source_rows(
        current_predictions,
        profile_name,
        CURRENT_POLICY,
    )
    current_default_persisted = current_source_rows(
        current_predictions,
        profile_name,
        UNIFORM_POLICY,
    )
    candidate_training_completed = not current_manifest.empty
    default_training_completed = not default_manifest.empty
    candidate_leakage = leakage_for_event(leakage, event_key, "season_aware_frozen")
    default_leakage = leakage_for_event(leakage, event_key, "static_baseline")
    candidate_mae, default_mae, improvement = prior_metrics(aligned)
    settings = profile_settings(profile)
    cold_start_passed = (
        int(selection_row.get("current_test_season_prior_event_count", 0))
        >= settings["min_current_season_prior_events_required"]
    )
    fold_gate = (
        aligned["fold_id"].nunique() >= settings["min_prior_candidate_folds_required"]
        if not aligned.empty
        else False
    )
    prediction_gate = len(aligned) >= settings["min_prior_candidate_predictions_required"]
    alignment_gate = not aligned.empty
    metric_available = improvement is not None
    margin_passed = (
        improvement is not None and improvement >= settings["improvement_margin_sec_required"]
    )
    eligible = bool(
        profile_name == "season_aware_frozen"
        and cold_start_passed
        and fold_gate
        and prediction_gate
        and alignment_gate
        and metric_available
        and margin_passed
    )
    selected = bool(selection_row.get("candidate_selected", False))
    blocking = blocking_reasons(
        profile_name=profile_name,
        candidate_training_completed=candidate_training_completed,
        candidate_persisted=not current_candidate_persisted.empty,
        prior_candidate_rows=len(candidate_prior),
        cold_start_passed=cold_start_passed,
        fold_gate=fold_gate,
        prediction_gate=prediction_gate,
        alignment_gate=alignment_gate,
        metric_available=metric_available,
        margin_passed=margin_passed,
        eligible=eligible,
        selected=selected,
    )
    return {
        "split_name": split_name,
        "train_seasons": selection_row.get("train_seasons"),
        "test_season": selection_row.get("test_season"),
        "fold_id_or_event_order": event_order.index(event_key)
        if event_key in event_order
        else None,
        "season": selection_row.get("season"),
        "event": selection_row.get("event"),
        "event_slug": selection_row.get("event_slug"),
        "checkpoint": selection_row.get("checkpoint"),
        "policy_profile": profile_name,
        "policy_signature": selection_row.get("policy_signature"),
        "current_test_season_prior_event_count": selection_row.get(
            "current_test_season_prior_event_count"
        ),
        "current_test_season_prior_event_keys": json.dumps(prior_keys),
        "candidate_prediction_available_for_current_event": candidate_training_completed,
        "candidate_prediction_persisted_for_current_event": bool(
            not current_candidate_persisted.empty
        ),
        "candidate_prediction_source_available": candidate_training_completed,
        "candidate_training_completed": candidate_training_completed,
        "candidate_training_row_count": number_from_manifest(
            current_manifest, "training_row_count"
        ),
        "candidate_training_event_count": number_from_manifest(
            current_manifest,
            "training_event_count",
        ),
        "candidate_temporal_weighting_policy": CURRENT_POLICY
        if profile_name == "season_aware_frozen"
        else None,
        "candidate_source_identity": source_identity_from_manifest(
            event_key,
            CURRENT_POLICY,
        )
        if candidate_training_completed
        else None,
        "default_prediction_available_for_current_event": default_training_completed,
        "default_prediction_persisted_for_current_event": bool(not current_default_persisted.empty),
        "default_prediction_source_available": default_training_completed,
        "default_training_completed": default_training_completed,
        "default_training_row_count": number_from_manifest(default_manifest, "training_row_count"),
        "default_training_event_count": number_from_manifest(
            default_manifest,
            "training_event_count",
        ),
        "default_source_identity": source_identity_from_manifest(event_key, UNIFORM_POLICY)
        if default_training_completed
        else None,
        "prior_candidate_events_available": int(candidate_prior["event_slug"].nunique())
        if not candidate_prior.empty
        else 0,
        "prior_candidate_prediction_rows_available": int(len(candidate_prior)),
        "prior_default_events_available": int(default_prior["event_slug"].nunique())
        if not default_prior.empty
        else 0,
        "prior_default_prediction_rows_available": int(len(default_prior)),
        "prior_candidate_default_aligned_events": int(aligned["event_slug"].nunique())
        if not aligned.empty
        else 0,
        "prior_candidate_default_aligned_rows": int(len(aligned)),
        "candidate_history_scope_valid": history_scope_valid(
            candidate_prior, event_key, event_order
        )
        and leakage_valid(candidate_leakage),
        "default_history_scope_valid": history_scope_valid(default_prior, event_key, event_order)
        and leakage_valid(default_leakage),
        "current_event_excluded_from_candidate_history": event_key
        not in event_keys_from_predictions(candidate_prior),
        "future_test_season_event_excluded_from_candidate_history": no_future_same_season(
            candidate_prior,
            event_key,
            event_order,
        ),
        "future_season_excluded_from_candidate_history": no_future_season(
            candidate_prior,
            int(selection_row.get("season")),
        ),
        **settings,
        "cold_start_gate_passed": bool(cold_start_passed),
        "candidate_fold_history_gate_passed": bool(fold_gate),
        "candidate_prediction_history_gate_passed": bool(prediction_gate),
        "candidate_default_alignment_gate_passed": bool(alignment_gate),
        "candidate_prior_metric_available": bool(metric_available),
        "candidate_prior_mae": candidate_mae,
        "default_prior_mae": default_mae,
        "prior_improvement_sec": improvement,
        "season_aware_candidate_eligible_under_frozen_gates": eligible,
        "season_aware_selected": selected,
        "season_aware_selection_reason": selection_row.get("candidate_selection_reason"),
        "primary_blocking_reason": blocking[0] if blocking else "candidate_selected",
        "all_blocking_reasons": ";".join(blocking),
    }


def build_eligibility_by_event(ledger: pd.DataFrame) -> pd.DataFrame:
    """Return compact event-level eligibility rows."""
    if ledger.empty:
        return pd.DataFrame(columns=_ledger_columns())
    return ledger.copy()


def build_gate_feasibility(ledger: pd.DataFrame) -> pd.DataFrame:
    """Summarize when frozen gates become satisfiable by split."""
    columns = [
        "split_name",
        "first_event_with_cold_start_gate_passed",
        "first_event_with_candidate_fold_history_gate_passed",
        "first_event_with_candidate_prediction_history_gate_passed",
        "first_event_with_candidate_default_alignment_gate_passed",
        "first_event_with_prior_metric_available",
        "first_event_with_all_non_margin_gates_passed",
        "first_event_with_margin_passed",
        "first_event_candidate_selected",
        "maximum_prior_candidate_folds_observed",
        "maximum_prior_candidate_prediction_rows_observed",
        "maximum_prior_aligned_rows_observed",
        "number_of_events_with_candidate_prediction_available",
        "number_of_events_with_usable_prior_candidate_evidence",
        "number_of_events_blocked_by_cold_start",
        "number_of_events_blocked_by_insufficient_candidate_history",
        "number_of_events_blocked_by_alignment",
        "number_of_events_blocked_by_margin",
        "number_of_events_with_unexpected_live_policy_disagreement",
        "min_prior_candidate_folds_feasible",
        "min_prior_candidate_predictions_feasible",
    ]
    if ledger.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    frame = ledger[ledger["policy_profile"].eq("season_aware_frozen")].copy()
    for split, group in frame.groupby("split_name", sort=False):
        group = group.sort_values("fold_id_or_event_order")
        non_margin = (
            group["cold_start_gate_passed"].astype(bool)
            & group["candidate_fold_history_gate_passed"].astype(bool)
            & group["candidate_prediction_history_gate_passed"].astype(bool)
            & group["candidate_default_alignment_gate_passed"].astype(bool)
            & group["candidate_prior_metric_available"].astype(bool)
        )
        mismatch = group[
            group["season_aware_candidate_eligible_under_frozen_gates"].astype(bool)
            != group["season_aware_selected"].astype(bool)
        ]
        max_folds = int(group["prior_candidate_default_aligned_events"].max()) if len(group) else 0
        max_rows = (
            int(group["prior_candidate_prediction_rows_available"].max()) if len(group) else 0
        )
        rows.append(
            {
                "split_name": split,
                "first_event_with_cold_start_gate_passed": first_event(
                    group,
                    "cold_start_gate_passed",
                ),
                "first_event_with_candidate_fold_history_gate_passed": first_event(
                    group,
                    "candidate_fold_history_gate_passed",
                ),
                "first_event_with_candidate_prediction_history_gate_passed": first_event(
                    group,
                    "candidate_prediction_history_gate_passed",
                ),
                "first_event_with_candidate_default_alignment_gate_passed": first_event(
                    group,
                    "candidate_default_alignment_gate_passed",
                ),
                "first_event_with_prior_metric_available": first_event(
                    group,
                    "candidate_prior_metric_available",
                ),
                "first_event_with_all_non_margin_gates_passed": first_event_from_mask(
                    group,
                    non_margin,
                ),
                "first_event_with_margin_passed": first_event(
                    group,
                    "season_aware_candidate_eligible_under_frozen_gates",
                ),
                "first_event_candidate_selected": first_event(group, "season_aware_selected"),
                "maximum_prior_candidate_folds_observed": max_folds,
                "maximum_prior_candidate_prediction_rows_observed": max_rows,
                "maximum_prior_aligned_rows_observed": int(
                    group["prior_candidate_default_aligned_rows"].max()
                )
                if len(group)
                else 0,
                "number_of_events_with_candidate_prediction_available": int(
                    group["candidate_prediction_available_for_current_event"].astype(bool).sum()
                ),
                "number_of_events_with_usable_prior_candidate_evidence": int(
                    group["candidate_prior_metric_available"].astype(bool).sum()
                ),
                "number_of_events_blocked_by_cold_start": contains_reason(group, "cold_start"),
                "number_of_events_blocked_by_insufficient_candidate_history": contains_reason(
                    group,
                    "insufficient_candidate_history",
                ),
                "number_of_events_blocked_by_alignment": contains_reason(group, "alignment"),
                "number_of_events_blocked_by_margin": contains_reason(group, "margin_not_met"),
                "number_of_events_with_unexpected_live_policy_disagreement": int(len(mismatch)),
                "min_prior_candidate_folds_feasible": bool(
                    max_folds >= int(group["min_prior_candidate_folds_required"].max())
                ),
                "min_prior_candidate_predictions_feasible": bool(
                    max_rows >= int(group["min_prior_candidate_predictions_required"].max())
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_gate_failure_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    """Count primary and all blocking reasons by split."""
    columns = ["split_name", "blocking_reason", "events", "reason_type"]
    if ledger.empty:
        return pd.DataFrame(columns=columns)
    frame = ledger[ledger["policy_profile"].eq("season_aware_frozen")]
    rows: list[dict[str, object]] = []
    for split, group in frame.groupby("split_name", sort=False):
        for reason, count in group["primary_blocking_reason"].value_counts().items():
            rows.append(
                {
                    "split_name": split,
                    "blocking_reason": reason,
                    "events": int(count),
                    "reason_type": "primary",
                }
            )
        exploded = (
            group.assign(reason=group["all_blocking_reasons"].astype(str).str.split(";"))
            .explode("reason")
            .query("reason != ''")
        )
        for reason, count in exploded["reason"].value_counts().items():
            rows.append(
                {
                    "split_name": split,
                    "blocking_reason": reason,
                    "events": int(count),
                    "reason_type": "all",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_candidate_availability_comparison(ledger: pd.DataFrame) -> pd.DataFrame:
    """Summarize training versus persisted candidate prediction availability."""
    columns = [
        "split_name",
        "events",
        "candidate_training_completed_events",
        "candidate_prediction_available_events",
        "candidate_prediction_persisted_events",
        "candidate_trained_but_not_persisted_events",
        "retention_status",
    ]
    if ledger.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    frame = ledger[ledger["policy_profile"].eq("season_aware_frozen")]
    for split, group in frame.groupby("split_name", sort=False):
        trained = group["candidate_training_completed"].astype(bool)
        persisted = group["candidate_prediction_persisted_for_current_event"].astype(bool)
        rows.append(
            {
                "split_name": split,
                "events": int(len(group)),
                "candidate_training_completed_events": int(trained.sum()),
                "candidate_prediction_available_events": int(
                    group["candidate_prediction_available_for_current_event"].astype(bool).sum()
                ),
                "candidate_prediction_persisted_events": int(persisted.sum()),
                "candidate_trained_but_not_persisted_events": int((trained & ~persisted).sum()),
                "retention_status": (
                    "trained_but_not_persisted_for_non_selected_events"
                    if (trained & ~persisted).any()
                    else "persisted"
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_artifact_driven_eligibility_comparison(
    ledger: pd.DataFrame,
    artifact_selection: pd.DataFrame,
) -> pd.DataFrame:
    """Compare M29 artifact-driven and M30 true replay eligibility evidence."""
    columns = [
        "split_name",
        "artifact_driven_split",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "artifact_driven_candidate_prediction_available",
        "artifact_driven_prior_candidate_folds",
        "artifact_driven_prior_candidate_prediction_rows",
        "artifact_driven_candidate_eligible",
        "artifact_driven_candidate_selected",
        "true_replay_candidate_prediction_available",
        "true_replay_prior_candidate_folds",
        "true_replay_prior_candidate_prediction_rows",
        "true_replay_candidate_eligible",
        "true_replay_candidate_selected",
        "availability_delta",
        "prior_fold_delta",
        "prior_prediction_row_delta",
        "eligibility_disagreement",
        "selection_disagreement",
        "likely_explanation",
    ]
    frame = ledger[ledger["policy_profile"].eq("season_aware_frozen")].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if artifact_selection.empty:
        result = frame.loc[:, ["split_name", "season", "event", "event_slug", "checkpoint"]].copy()
        result["artifact_driven_split"] = result["split_name"].map(artifact_split_name)
        result["likely_explanation"] = "missing artifact prevents definitive comparison"
        for column in columns:
            if column not in result:
                result[column] = pd.NA
        return result.loc[:, columns]
    artifact = artifact_selection[
        artifact_selection["policy_profile"].astype(str).eq("season_aware_frozen")
        & artifact_selection["checkpoint"].astype(str).eq(FP3_CHECKPOINT)
    ].copy()
    artifact["artifact_driven_split"] = artifact["prospective_split"].astype(str)
    frame["artifact_driven_split"] = frame["split_name"].map(artifact_split_name)
    merged = frame.merge(
        artifact,
        on=["artifact_driven_split", "season", "event_slug", "checkpoint"],
        how="left",
        suffixes=("_replay", "_artifact"),
    )
    rows = []
    for _, row in merged.iterrows():
        artifact_available = _bool(row.get("season_aware_candidate_available"))
        artifact_folds = _int_or_zero(row.get("season_aware_candidate_prior_folds"))
        artifact_rows = _int_or_zero(row.get("season_aware_candidate_prior_predictions"))
        artifact_eligible = _bool(row.get("season_aware_candidate_eligible"))
        artifact_selected = _bool(row.get("candidate_selected_artifact"))
        replay_available = _bool(row.get("candidate_prediction_available_for_current_event"))
        replay_folds = _int_or_zero(row.get("prior_candidate_default_aligned_events"))
        replay_rows = _int_or_zero(row.get("prior_candidate_prediction_rows_available"))
        replay_eligible = _bool(row.get("season_aware_candidate_eligible_under_frozen_gates"))
        replay_selected = _bool(row.get("season_aware_selected_replay"))
        rows.append(
            {
                "split_name": row.get("split_name"),
                "artifact_driven_split": row.get("artifact_driven_split"),
                "season": row.get("season"),
                "event": row.get("event_replay", row.get("event")),
                "event_slug": row.get("event_slug"),
                "checkpoint": row.get("checkpoint"),
                "artifact_driven_candidate_prediction_available": artifact_available,
                "artifact_driven_prior_candidate_folds": artifact_folds,
                "artifact_driven_prior_candidate_prediction_rows": artifact_rows,
                "artifact_driven_candidate_eligible": artifact_eligible,
                "artifact_driven_candidate_selected": artifact_selected,
                "true_replay_candidate_prediction_available": replay_available,
                "true_replay_prior_candidate_folds": replay_folds,
                "true_replay_prior_candidate_prediction_rows": replay_rows,
                "true_replay_candidate_eligible": replay_eligible,
                "true_replay_candidate_selected": replay_selected,
                "availability_delta": int(replay_available) - int(artifact_available),
                "prior_fold_delta": replay_folds - artifact_folds,
                "prior_prediction_row_delta": replay_rows - artifact_rows,
                "eligibility_disagreement": artifact_eligible != replay_eligible,
                "selection_disagreement": artifact_selected != replay_selected,
                "likely_explanation": comparison_explanation(
                    artifact_available=artifact_available,
                    replay_available=replay_available,
                    artifact_rows=artifact_rows,
                    replay_rows=replay_rows,
                    artifact_selected=artifact_selected,
                    replay_selected=replay_selected,
                    has_artifact_match=not pd.isna(row.get("policy_profile_artifact")),
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_live_selection_consistency(ledger: pd.DataFrame) -> pd.DataFrame:
    """Report whether ledger eligibility agrees with saved replay selection."""
    columns = [
        "split_name",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "ledger_candidate_eligible",
        "live_candidate_selected",
        "live_selection_reason",
        "selection_consistent",
        "consistency_issue",
    ]
    if ledger.empty:
        return pd.DataFrame(columns=columns)
    frame = ledger[ledger["policy_profile"].eq("season_aware_frozen")].copy()
    frame["selection_consistent"] = (
        frame["season_aware_candidate_eligible_under_frozen_gates"]
        .astype(bool)
        .eq(frame["season_aware_selected"].astype(bool))
    )
    frame["consistency_issue"] = frame["selection_consistent"].map(
        lambda value: "" if value else "ledger_live_selection_mismatch"
    )
    return frame.rename(
        columns={
            "season_aware_candidate_eligible_under_frozen_gates": "ledger_candidate_eligible",
            "season_aware_selected": "live_candidate_selected",
            "season_aware_selection_reason": "live_selection_reason",
        }
    ).loc[:, columns]


def build_eligibility_summary_payload(
    *,
    artifacts: dict[str, Any],
    ledger: pd.DataFrame,
    feasibility: pd.DataFrame,
    failures: pd.DataFrame,
    artifact_comparison: pd.DataFrame,
    consistency: pd.DataFrame,
    figure_paths: list[Path],
    figure_issues: list[str],
) -> dict[str, object]:
    """Build summary JSON for the audit."""
    season_aware = ledger[ledger["policy_profile"].eq("season_aware_frozen")]
    retention_status = candidate_evidence_retention_status(season_aware)
    consistency_ok = (
        bool(consistency["selection_consistent"].astype(bool).all())
        if not consistency.empty
        else False
    )
    future_violation = (
        bool(
            (
                season_aware["future_test_season_event_excluded_from_candidate_history"].eq(False)
            ).any()
        )
        if not season_aware.empty
        else False
    )
    current_violation = (
        bool((season_aware["current_event_excluded_from_candidate_history"].eq(False)).any())
        if not season_aware.empty
        else False
    )
    primary = primary_zero_selection_explanation(season_aware, retention_status)
    return {
        "status": "complete" if not ledger.empty else "missing_inputs",
        "inputs_available": artifacts["inputs_available"],
        "missing_inputs": artifacts["missing_inputs"],
        "replay_splits_available": sorted(ledger["split_name"].dropna().unique().tolist())
        if not ledger.empty
        else [],
        "replay_splits_complete": bool(not ledger.empty and not artifacts["missing_inputs"]),
        "candidate_evidence_retention_status": retention_status,
        "candidate_prediction_availability_status": candidate_prediction_availability_status(
            season_aware
        ),
        "live_selection_consistency_status": "consistent" if consistency_ok else "mismatch",
        "future_history_violation_detected": future_violation,
        "current_event_history_violation_detected": current_violation,
        "primary_explanation_for_zero_selection": primary,
        "secondary_explanations": secondary_explanations(season_aware, artifact_comparison),
        "true_replay_gate_feasibility_summary": records_for_json(feasibility),
        "artifact_driven_vs_true_replay_summary": artifact_comparison_summary(artifact_comparison),
        "candidate_evidence_pipeline_recommendation": pipeline_recommendation(
            retention_status,
            season_aware,
            future_violation,
            current_violation,
        ),
        "policy_recommendation": "retain_static_policy",
        "known_limitations": [
            "Audit is artifact-driven and does not retrain candidate models.",
            "Existing replay prediction artifacts persist only selected policy predictions.",
            "Weighted candidate predicted gaps for non-selected events are not available unless "
            "future replay runs add diagnostic-only shadow candidate persistence.",
        ],
        "generated_outputs": {
            "metrics": [
                "reports/metrics/prospective_replay_eligibility_audit_summary.json",
                "reports/metrics/prospective_replay_eligibility_by_event.csv",
                "reports/metrics/prospective_replay_candidate_evidence_ledger.csv",
                "reports/metrics/prospective_replay_gate_feasibility.csv",
                "reports/metrics/prospective_replay_gate_failure_summary.csv",
                "reports/metrics/prospective_replay_candidate_availability_comparison.csv",
                "reports/metrics/prospective_replay_vs_artifact_driven_eligibility.csv",
                "reports/metrics/prospective_replay_live_selection_consistency.csv",
            ],
            "figures": [_relative_report_path(path) for path in figure_paths],
        },
        "generation_issues": figure_issues,
        "generated_at": utc_now(),
    }


def generate_eligibility_figures(
    *,
    figures_dir: Path,
    ledger: pd.DataFrame,
    feasibility: pd.DataFrame,
    failures: pd.DataFrame,
    availability: pd.DataFrame,
    artifact_comparison: pd.DataFrame,
) -> tuple[list[Path], list[str]]:
    """Generate static Matplotlib audit figures."""
    ensure_directory(figures_dir)
    cache_dir = figures_dir / ".matplotlib-cache"
    ensure_directory(cache_dir)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [
        (
            "prospective_replay_eligibility_gate_pass_rate_by_event.png",
            lambda p: plot_gate_pass_rate(plt, ledger, p),
        ),
        (
            "prospective_replay_eligibility_history_growth.png",
            lambda p: plot_history_growth(plt, ledger, p),
        ),
        (
            "prospective_replay_eligibility_blocking_reasons.png",
            lambda p: plot_blocking_reasons(plt, failures, p),
        ),
        (
            "prospective_replay_eligibility_candidate_availability.png",
            lambda p: plot_candidate_availability(plt, availability, p),
        ),
        (
            "prospective_replay_eligibility_vs_artifact_driven.png",
            lambda p: plot_artifact_comparison(plt, artifact_comparison, p),
        ),
        (
            "prospective_replay_eligibility_feasibility_timeline.png",
            lambda p: plot_feasibility_timeline(plt, feasibility, p),
        ),
    ]
    paths: list[Path] = []
    issues: list[str] = []
    for filename, callback in specs:
        path = figures_dir / filename
        try:
            if callback(path):
                paths.append(path)
            else:
                issues.append(f"skipped figure {filename}: insufficient data")
        except Exception as exc:  # pragma: no cover
            issues.append(f"skipped figure {filename}: {exc}")
        finally:
            plt.close("all")
    return paths, issues


def event_order_from_selection(selection: pd.DataFrame) -> list[str]:
    """Return chronological test-season event keys from replay selection rows."""
    frame = selection.copy()
    frame["event_key"] = frame.apply(
        lambda row: event_key_from_values(row["season"], row["event_slug"]),
        axis=1,
    )
    frame["prior_count"] = pd.to_numeric(
        frame["current_test_season_prior_event_count"],
        errors="coerce",
    )
    ordered = frame.sort_values(["prior_count", "event_key"])["event_key"].drop_duplicates()
    return ordered.astype(str).tolist()


def manifest_for_split(manifest: pd.DataFrame, split_selection: pd.DataFrame) -> pd.DataFrame:
    """Filter manifest rows to the train/test season scope for one split."""
    if manifest.empty:
        return manifest
    test_season = int(split_selection["test_season"].iloc[0])
    train_text = str(split_selection["train_seasons"].iloc[0])
    train_seasons = {
        int(value)
        for value in train_text.replace("[", "").replace("]", "").replace(" ", "").split(",")
        if value
    }
    allowed = set(train_seasons) | {test_season}
    return manifest[
        manifest["test_season"].astype(int).eq(test_season)
        & manifest["test_event"].astype(str).map(lambda key: event_season(key) in allowed)
    ].copy()


def leakage_for_split(leakage: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Filter leakage rows to manifest event/profile pairs."""
    if leakage.empty or manifest.empty:
        return leakage.iloc[0:0].copy() if not leakage.empty else leakage
    keys = set(
        zip(
            manifest["test_event"].astype(str),
            manifest["policy_profile"].astype(str),
            strict=True,
        )
    )
    return leakage[
        leakage.apply(
            lambda row: (str(row.get("test_event")), str(row.get("policy_profile"))) in keys,
            axis=1,
        )
    ].copy()


def filter_event_predictions(predictions: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    mask = (
        predictions["policy_profile"].astype(str).eq(str(row.get("policy_profile")))
        & predictions["checkpoint"].astype(str).eq(FP3_CHECKPOINT)
        & predictions["season"].astype(int).eq(int(row.get("season")))
        & predictions["event_slug"].astype(str).eq(str(row.get("event_slug")))
    )
    return predictions[mask].copy()


def prior_source_rows(
    predictions: pd.DataFrame,
    profile_name: str,
    temporal_policy: str,
) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    mask = (
        predictions["policy_profile"].astype(str).eq(profile_name)
        & predictions["checkpoint"].astype(str).eq(FP3_CHECKPOINT)
        & predictions["source_temporal_weighting_policy"].astype(str).eq(temporal_policy)
    )
    return predictions[mask].copy()


def current_source_rows(
    predictions: pd.DataFrame,
    profile_name: str,
    temporal_policy: str,
) -> pd.DataFrame:
    return prior_source_rows(predictions, profile_name, temporal_policy)


def leakage_for_event(leakage: pd.DataFrame, event_key: str, policy_profile: str) -> pd.DataFrame:
    if leakage.empty:
        return leakage
    return leakage[
        leakage["test_event"].astype(str).eq(event_key)
        & leakage["policy_profile"].astype(str).eq(policy_profile)
    ].copy()


def number_from_manifest(manifest: pd.DataFrame, column: str) -> int | None:
    if manifest.empty or column not in manifest:
        return None
    value = pd.to_numeric(manifest[column], errors="coerce").dropna()
    return int(value.iloc[0]) if not value.empty else None


def source_identity_from_manifest(event_key: str, temporal_policy: str) -> str:
    return json.dumps(
        {
            "family": "ablation",
            "model_name": "random_forest",
            "feature_group": "base_plus_relative",
            "temporal_weighting_policy": temporal_policy,
            "event_key": event_key,
            "source": "true_replay_training_manifest",
        },
        sort_keys=True,
    )


def prior_metrics(aligned: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
    if aligned.empty:
        return None, None, None
    candidate = _number_or_none(aligned["candidate_abs_error"].mean())
    default = _number_or_none(aligned["default_abs_error"].mean())
    improvement = None if candidate is None or default is None else default - candidate
    return candidate, default, improvement


def profile_settings(profile: Any) -> dict[str, object]:
    return {
        "min_current_season_prior_events_required": int(
            getattr(profile, "cold_start_threshold", None) or 0
        ),
        "min_prior_candidate_folds_required": int(getattr(profile, "min_prior_folds", None) or 0),
        "min_prior_candidate_predictions_required": int(
            getattr(profile, "min_prior_predictions", None) or 0
        ),
        "improvement_margin_sec_required": float(
            getattr(profile, "improvement_margin_sec", None) or 0.0
        ),
    }


def blocking_reasons(
    *,
    profile_name: str,
    candidate_training_completed: bool,
    candidate_persisted: bool,
    prior_candidate_rows: int,
    cold_start_passed: bool,
    fold_gate: bool,
    prediction_gate: bool,
    alignment_gate: bool,
    metric_available: bool,
    margin_passed: bool,
    eligible: bool,
    selected: bool,
) -> list[str]:
    if profile_name != "season_aware_frozen":
        return ["not_season_aware_profile"]
    if selected:
        return ["candidate_selected"]
    reasons: list[str] = []
    if not candidate_training_completed:
        reasons.append("candidate_not_trained")
    if candidate_training_completed and not candidate_persisted:
        reasons.append("candidate_prediction_not_persisted")
    if not cold_start_passed:
        reasons.append("cold_start")
    if prior_candidate_rows == 0 and candidate_training_completed:
        reasons.append("candidate_predictions_not_retained_for_prior_events")
    if not fold_gate or not prediction_gate:
        reasons.append("insufficient_candidate_history")
    if not alignment_gate:
        reasons.append("candidate_default_alignment_unavailable")
    if not metric_available:
        reasons.append("prior_metric_unavailable")
    if metric_available and not margin_passed:
        reasons.append("margin_not_met")
    if eligible and not selected:
        reasons.append("eligible_but_not_selected")
    return list(dict.fromkeys(reasons))


def history_scope_valid(candidate: pd.DataFrame, event_key: str, event_order: list[str]) -> bool:
    if candidate.empty:
        return True
    current_index = event_order.index(event_key) if event_key in event_order else None
    if current_index is None:
        return False
    for key in event_keys_from_predictions(candidate):
        if key not in event_order or event_order.index(key) >= current_index:
            return False
    return True


def leakage_valid(leakage: pd.DataFrame) -> bool:
    if leakage.empty:
        return True
    return bool(leakage["leakage_status"].astype(str).eq("valid").all())


def event_keys_from_predictions(frame: pd.DataFrame) -> set[str]:
    if frame.empty:
        return set()
    return {
        event_key_from_values(row.season, row.event_slug)
        for row in frame.loc[:, ["season", "event_slug"]].drop_duplicates().itertuples()
    }


def no_future_same_season(candidate: pd.DataFrame, event_key: str, event_order: list[str]) -> bool:
    if candidate.empty:
        return True
    current_index = event_order.index(event_key) if event_key in event_order else None
    if current_index is None:
        return False
    current_season = event_season(event_key)
    for key in event_keys_from_predictions(candidate):
        if event_season(key) == current_season and (
            key not in event_order or event_order.index(key) >= current_index
        ):
            return False
    return True


def no_future_season(candidate: pd.DataFrame, current_season: int) -> bool:
    if candidate.empty:
        return True
    return all(
        event_season(key) <= current_season for key in event_keys_from_predictions(candidate)
    )


def prior_event_keys(event_key: str, event_order: list[str]) -> list[str]:
    if event_key not in event_order:
        return []
    return event_order[: event_order.index(event_key)]


def first_event(group: pd.DataFrame, column: str) -> str | None:
    if group.empty or column not in group:
        return None
    return first_event_from_mask(group, group[column].astype(bool))


def first_event_from_mask(group: pd.DataFrame, mask: pd.Series) -> str | None:
    matches = group.loc[mask.fillna(False)]
    if matches.empty:
        return None
    row = matches.sort_values("fold_id_or_event_order").iloc[0]
    return event_key_from_values(row["season"], row["event_slug"])


def contains_reason(group: pd.DataFrame, reason: str) -> int:
    return int(group["all_blocking_reasons"].astype(str).str.contains(reason, regex=False).sum())


def artifact_split_name(replay_split: object) -> str:
    return str(replay_split).replace("prospective_replay_", "prospective_", 1)


def comparison_explanation(
    *,
    artifact_available: bool,
    replay_available: bool,
    artifact_rows: int,
    replay_rows: int,
    artifact_selected: bool,
    replay_selected: bool,
    has_artifact_match: bool,
) -> str:
    if not has_artifact_match:
        return "missing artifact prevents definitive comparison"
    if artifact_rows > replay_rows:
        return (
            "artifact-driven history contains saved walk-forward candidate predictions unavailable "
            "in strict replay"
        )
    if replay_available and replay_rows == 0:
        return "true replay candidate predictions were not retained for non-selected events"
    if artifact_selected != replay_selected:
        return "candidate/default alignment or available history differs between pathways"
    if not artifact_available and not replay_available:
        return "candidate unavailable in both pathways"
    return "true replay history is genuinely insufficient under frozen gates"


def candidate_evidence_retention_status(ledger: pd.DataFrame) -> str:
    if ledger.empty:
        return "missing_replay_evidence"
    trained = ledger["candidate_training_completed"].astype(bool)
    persisted = ledger["candidate_prediction_persisted_for_current_event"].astype(bool)
    if (trained & ~persisted).any():
        return "trained_but_not_persisted_for_non_selected_events"
    if trained.all() and persisted.all():
        return "candidate_predictions_persisted"
    return "candidate_training_missing_for_some_events"


def candidate_prediction_availability_status(ledger: pd.DataFrame) -> str:
    if ledger.empty:
        return "missing"
    trained = int(ledger["candidate_training_completed"].astype(bool).sum())
    persisted = int(ledger["candidate_prediction_persisted_for_current_event"].astype(bool).sum())
    return f"trained_events={trained};persisted_candidate_prediction_events={persisted}"


def primary_zero_selection_explanation(ledger: pd.DataFrame, retention_status: str) -> str:
    if ledger.empty:
        return "missing replay artifacts"
    if retention_status == "trained_but_not_persisted_for_non_selected_events":
        return (
            "weighted candidates are trained in true replay but non-selected candidate predictions "
            "are not retained, so prior candidate evidence cannot accumulate under frozen gates"
        )
    if contains_reason(ledger, "cold_start") == len(ledger):
        return "current-season cold-start gate blocks all events"
    if contains_reason(ledger, "insufficient_candidate_history"):
        return "legal prior candidate history is insufficient under frozen gates"
    if contains_reason(ledger, "margin_not_met"):
        return "prior candidate evidence exists but does not beat the required margin"
    return "mixed gate failures require manual review"


def secondary_explanations(
    ledger: pd.DataFrame,
    artifact_comparison: pd.DataFrame,
) -> list[str]:
    explanations: list[str] = []
    if not ledger.empty:
        explanations.append(
            "Candidate model training completed for "
            f"{int(ledger['candidate_training_completed'].astype(bool).sum())} event/profile rows."
        )
        explanations.append(
            "Persisted weighted candidate prediction rows exist for "
            f"{int(ledger['candidate_prediction_persisted_for_current_event'].astype(bool).sum())} "
            "event/profile rows."
        )
    if not artifact_comparison.empty and artifact_comparison["selection_disagreement"].any():
        explanations.append(
            "Artifact-driven prospective selection differs from true replay because saved "
            "walk-forward candidate history is available outside strict replay history."
        )
    return explanations


def artifact_comparison_summary(comparison: pd.DataFrame) -> dict[str, object]:
    if comparison.empty:
        return {"available": False}
    return {
        "available": True,
        "rows": int(len(comparison)),
        "eligibility_disagreements": int(
            _bool_series(comparison["eligibility_disagreement"]).sum()
        ),
        "selection_disagreements": int(_bool_series(comparison["selection_disagreement"]).sum()),
        "mean_prior_prediction_row_delta": _number_or_none(
            pd.to_numeric(comparison["prior_prediction_row_delta"], errors="coerce").mean()
        ),
        "common_explanations": comparison["likely_explanation"]
        .astype(str)
        .value_counts()
        .to_dict(),
    }


def pipeline_recommendation(
    retention_status: str,
    ledger: pd.DataFrame,
    future_violation: bool,
    current_violation: bool,
) -> str:
    if future_violation or current_violation:
        return "replay_candidate_history_pipeline_fix_required"
    if retention_status == "trained_but_not_persisted_for_non_selected_events":
        return "diagnostic_evidence_persistence_needed"
    if ledger.empty:
        return "mixed_evidence_requires_manual_review"
    if not ledger["candidate_training_completed"].astype(bool).all():
        return "replay_candidate_history_pipeline_fix_required"
    if ledger["prior_candidate_prediction_rows_available"].max() == 0:
        return "insufficient_prospective_history_requires_more_seasons"
    return "no_pipeline_change_needed"


def plot_gate_pass_rate(plt: Any, ledger: pd.DataFrame, path: Path) -> bool:
    frame = ledger[ledger["policy_profile"].eq("season_aware_frozen")].copy()
    if frame.empty:
        return False
    frame = frame.sort_values(["split_name", "fold_id_or_event_order"])
    gate_columns = [
        "cold_start_gate_passed",
        "candidate_fold_history_gate_passed",
        "candidate_prediction_history_gate_passed",
        "candidate_default_alignment_gate_passed",
    ]
    plot_frame = frame.melt(
        id_vars=["split_name", "event_slug"],
        value_vars=gate_columns,
        var_name="gate",
        value_name="passed",
    )
    pivot = plot_frame.pivot_table(
        index="event_slug",
        columns="gate",
        values="passed",
        aggfunc="mean",
    )
    ax = pivot.plot(kind="line", marker="o", figsize=(10, 4))
    ax.set_title("Replay eligibility gate pass rate by event")
    ax.set_ylabel("Pass rate")
    ax.set_xlabel("")
    ax.set_ylim(-0.05, 1.05)
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    return True


def plot_history_growth(plt: Any, ledger: pd.DataFrame, path: Path) -> bool:
    frame = ledger[ledger["policy_profile"].eq("season_aware_frozen")].copy()
    if frame.empty:
        return False
    frame = frame.sort_values(["split_name", "fold_id_or_event_order"])
    ax = frame.plot(
        x="event_slug",
        y=[
            "prior_candidate_prediction_rows_available",
            "prior_default_prediction_rows_available",
            "prior_candidate_default_aligned_rows",
        ],
        kind="line",
        marker="o",
        figsize=(10, 4),
    )
    ax.set_title("Replay prior evidence growth")
    ax.set_ylabel("Prior rows")
    ax.set_xlabel("")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    return True


def plot_blocking_reasons(plt: Any, failures: pd.DataFrame, path: Path) -> bool:
    frame = failures[failures["reason_type"].eq("primary")].copy()
    if frame.empty:
        return False
    pivot = frame.pivot_table(
        index="blocking_reason",
        columns="split_name",
        values="events",
        aggfunc="sum",
        fill_value=0,
    )
    ax = pivot.plot(kind="barh", figsize=(9, 4))
    ax.set_title("Primary replay eligibility blocking reasons")
    ax.set_xlabel("Events")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    return True


def plot_candidate_availability(plt: Any, availability: pd.DataFrame, path: Path) -> bool:
    if availability.empty:
        return False
    frame = availability.set_index("split_name")[
        [
            "candidate_training_completed_events",
            "candidate_prediction_persisted_events",
            "candidate_trained_but_not_persisted_events",
        ]
    ]
    ax = frame.plot(kind="bar", figsize=(9, 4))
    ax.set_title("Candidate training vs persisted prediction availability")
    ax.set_ylabel("Events")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    return True


def plot_artifact_comparison(plt: Any, comparison: pd.DataFrame, path: Path) -> bool:
    if comparison.empty:
        return False
    frame = comparison.copy()
    frame["label"] = frame["split_name"].astype(str) + " " + frame["event_slug"].astype(str)
    ax = frame.plot(
        x="label",
        y=["prior_fold_delta", "prior_prediction_row_delta"],
        kind="line",
        marker="o",
        figsize=(10, 4),
    )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("True replay minus artifact-driven prior evidence")
    ax.set_ylabel("Delta")
    ax.set_xlabel("")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    return True


def plot_feasibility_timeline(plt: Any, feasibility: pd.DataFrame, path: Path) -> bool:
    if feasibility.empty:
        return False
    columns = [
        "maximum_prior_candidate_folds_observed",
        "maximum_prior_candidate_prediction_rows_observed",
        "maximum_prior_aligned_rows_observed",
    ]
    frame = feasibility.set_index("split_name")[columns]
    ax = frame.plot(kind="bar", figsize=(9, 4))
    ax.set_title("Replay eligibility feasibility maxima")
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    return True


def _ledger_columns() -> list[str]:
    return [
        "split_name",
        "train_seasons",
        "test_season",
        "fold_id_or_event_order",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "policy_profile",
        "policy_signature",
        "current_test_season_prior_event_count",
        "current_test_season_prior_event_keys",
        "candidate_prediction_available_for_current_event",
        "candidate_prediction_persisted_for_current_event",
        "candidate_prediction_source_available",
        "candidate_training_completed",
        "candidate_training_row_count",
        "candidate_training_event_count",
        "candidate_temporal_weighting_policy",
        "candidate_source_identity",
        "default_prediction_available_for_current_event",
        "default_prediction_persisted_for_current_event",
        "default_prediction_source_available",
        "default_training_completed",
        "default_training_row_count",
        "default_training_event_count",
        "default_source_identity",
        "prior_candidate_events_available",
        "prior_candidate_prediction_rows_available",
        "prior_default_events_available",
        "prior_default_prediction_rows_available",
        "prior_candidate_default_aligned_events",
        "prior_candidate_default_aligned_rows",
        "candidate_history_scope_valid",
        "default_history_scope_valid",
        "current_event_excluded_from_candidate_history",
        "future_test_season_event_excluded_from_candidate_history",
        "future_season_excluded_from_candidate_history",
        "min_current_season_prior_events_required",
        "min_prior_candidate_folds_required",
        "min_prior_candidate_predictions_required",
        "improvement_margin_sec_required",
        "cold_start_gate_passed",
        "candidate_fold_history_gate_passed",
        "candidate_prediction_history_gate_passed",
        "candidate_default_alignment_gate_passed",
        "candidate_prior_metric_available",
        "candidate_prior_mae",
        "default_prior_mae",
        "prior_improvement_sec",
        "season_aware_candidate_eligible_under_frozen_gates",
        "season_aware_selected",
        "season_aware_selection_reason",
        "primary_blocking_reason",
        "all_blocking_reasons",
    ]


def event_key_from_values(season: object, event_slug: object) -> str:
    return f"{int(season)}/{event_slug}"


def event_season(event_key: str) -> int:
    return int(str(event_key).split("/", maxsplit=1)[0])


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _bool_series(values: pd.Series) -> pd.Series:
    return values.map(_bool).astype(bool)


def _int_or_zero(value: object) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0 if pd.isna(numeric) else int(numeric)


def _number_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def records_for_json(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [_json_clean(record) for record in frame.to_dict(orient="records")]


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(_json_clean(payload), output_file, indent=2, allow_nan=False)
        output_file.write("\n")


def _relative_report_path(path: Path) -> str:
    parts = path.parts
    if "reports" in parts:
        index = parts.index("reports")
        return str(Path(*parts[index:]))
    return path.as_posix()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
