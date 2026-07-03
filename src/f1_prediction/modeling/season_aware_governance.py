"""Artifact-driven governance synthesis for the season-aware FP3 candidate."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.utils.paths import ensure_directory

FP3_CHECKPOINT = "after_fp3"
CANDIDATE_TEMPORAL_POLICY = "current_season_only_with_prior"
DEFAULT_TEMPORAL_POLICY = "uniform"
CANONICAL_FAMILY = "ablation"
CANONICAL_MODEL = "random_forest"
CANONICAL_FEATURE_GROUP = "base_plus_relative"
MIN_HISTORY_FOLDS = 5
MIN_HISTORY_PREDICTIONS = 100
IMPROVEMENT_MARGIN_SEC = 0.05

EVIDENCE_SOURCES: tuple[str, ...] = (
    "retrospective_aligned_walk_forward",
    "artifact_driven_prospective_evaluation",
    "true_retrain_based_prospective_replay",
    "shadow_history_counterfactual_frozen_gate_evaluation",
    "candidate_eligibility_audit",
    "policy_forensics_and_source_lineage",
)

EVIDENCE_COLUMNS: tuple[str, ...] = (
    "evidence_source",
    "evidence_type",
    "scope",
    "split_name",
    "train_seasons",
    "test_season",
    "checkpoint",
    "candidate_identity",
    "default_identity",
    "candidate_available",
    "candidate_selected",
    "candidate_eligible",
    "selection_is_live",
    "selection_is_counterfactual",
    "uses_retraining",
    "uses_prior_only_history",
    "uses_shadow_persistence",
    "leakage_audit_status",
    "source_lineage_status",
    "metric_name",
    "candidate_metric_value",
    "default_metric_value",
    "delta_vs_default",
    "confidence_or_uncertainty_note",
    "evidence_strength",
    "limitations",
    "artifact_path_or_status",
)

MATRIX_COLUMNS: tuple[str, ...] = (
    "evidence_source",
    "split_name",
    "checkpoint",
    "candidate_identity_valid",
    "default_identity_valid",
    "comparison_scope_valid",
    "candidate_available",
    "candidate_eligible",
    "candidate_selected_live",
    "candidate_selected_counterfactual",
    "candidate_metric",
    "default_metric",
    "delta_vs_default",
    "candidate_better_than_default",
    "selection_status",
    "methodological_interpretation",
    "governance_weight",
    "primary_limitation",
)

IDENTITY_COLUMNS: tuple[str, ...] = (
    "evidence_source",
    "artifact_name",
    "split_name",
    "checkpoint",
    "identity_role",
    "identity_valid",
    "validation_method",
    "family",
    "model_name",
    "feature_group",
    "temporal_weighting_policy",
    "expected_family",
    "expected_model_name",
    "expected_feature_group",
    "expected_temporal_weighting_policy",
    "mismatch_reason",
)


@dataclass(frozen=True)
class SeasonAwareGovernanceSummary:
    """Paths and issue counts produced by the season-aware governance report."""

    status: str
    summary_path: Path
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    missing_inputs: tuple[str, ...]
    generation_issues: tuple[str, ...]


def create_season_aware_governance_report(config: DataConfig) -> SeasonAwareGovernanceSummary:
    """Synthesize saved season-aware candidate evidence into governance artifacts."""
    metrics_dir = config.metrics_output_dir
    figures_dir = metrics_dir.parent / "figures"
    ensure_directory(metrics_dir)
    ensure_directory(figures_dir)

    artifacts, missing_inputs, generation_issues = load_governance_artifacts(metrics_dir)
    identity = build_identity_validation(artifacts, missing_inputs)
    evidence = build_evidence_inventory(artifacts, identity, missing_inputs)
    matrix = build_governance_matrix(evidence, identity)
    split_summary = build_split_summary(matrix, artifacts)
    regime_summary = build_regime_summary(artifacts)
    missing = build_missing_evidence(missing_inputs)
    decision_trace = build_decision_trace(matrix, split_summary, regime_summary, identity)
    summary_payload = build_governance_summary_payload(
        artifacts=artifacts,
        evidence=evidence,
        matrix=matrix,
        split_summary=split_summary,
        regime_summary=regime_summary,
        identity=identity,
        decision_trace=decision_trace,
        missing_inputs=missing_inputs,
        generation_issues=generation_issues,
    )

    table_paths = (
        metrics_dir / "season_aware_governance_matrix.csv",
        metrics_dir / "season_aware_governance_evidence_inventory.csv",
        metrics_dir / "season_aware_governance_split_summary.csv",
        metrics_dir / "season_aware_governance_regime_summary.csv",
        metrics_dir / "season_aware_governance_identity_validation.csv",
        metrics_dir / "season_aware_governance_decision_trace.csv",
        metrics_dir / "season_aware_governance_missing_evidence.csv",
    )
    matrix.to_csv(table_paths[0], index=False)
    evidence.to_csv(table_paths[1], index=False)
    split_summary.to_csv(table_paths[2], index=False)
    regime_summary.to_csv(table_paths[3], index=False)
    identity.to_csv(table_paths[4], index=False)
    decision_trace.to_csv(table_paths[5], index=False)
    missing.to_csv(table_paths[6], index=False)

    summary_path = metrics_dir / "season_aware_governance_summary.json"
    _write_json(summary_path, summary_payload)
    figure_paths, figure_issues = generate_governance_figures(
        figures_dir=figures_dir,
        matrix=matrix,
        split_summary=split_summary,
        regime_summary=regime_summary,
        decision_trace=decision_trace,
    )
    summary_payload["generated_outputs"]["figures"] = [
        _relative_report_path(path) for path in figure_paths
    ]
    summary_payload["generation_issues"] = [
        *summary_payload["generation_issues"],
        *figure_issues,
    ]
    _write_json(summary_path, summary_payload)

    return SeasonAwareGovernanceSummary(
        status=str(summary_payload["status"]),
        summary_path=summary_path,
        table_paths=(summary_path, *table_paths),
        figure_paths=tuple(figure_paths),
        missing_inputs=tuple(missing_inputs),
        generation_issues=tuple(summary_payload["generation_issues"]),
    )


def load_governance_artifacts(
    metrics_dir: Path,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Load optional saved artifacts needed for governance synthesis."""
    json_specs = {
        "prospective_replay_summary": "prospective_replay_summary.json",
        "prospective_replay_shadow_candidate_summary": (
            "prospective_replay_shadow_candidate_summary.json"
        ),
        "prospective_replay_eligibility_audit_summary": (
            "prospective_replay_eligibility_audit_summary.json"
        ),
        "prospective_policy_summary": "prospective_policy_summary.json",
        "season_aware_validation_summary": "season_aware_validation_summary.json",
        "season_aware_candidate_audit_summary": "season_aware_candidate_audit_summary.json",
        "season_aware_policy_forensics_summary": "season_aware_policy_forensics_summary.json",
    }
    csv_specs = {
        "prospective_replay_selection_log": "prospective_replay_selection_log.csv",
        "prospective_replay_event_comparison": "prospective_replay_event_comparison.csv",
        "prospective_replay_shadow_gate_feasibility": (
            "prospective_replay_shadow_gate_feasibility.csv"
        ),
        "prospective_replay_shadow_vs_live_selection": (
            "prospective_replay_shadow_vs_live_selection.csv"
        ),
        "prospective_replay_candidate_evidence_ledger": (
            "prospective_replay_candidate_evidence_ledger.csv"
        ),
    }
    parquet_specs = {
        "prospective_replay_shadow_candidates": "prospective_replay_shadow_candidates.parquet",
    }
    artifacts: dict[str, Any] = {"metrics_dir": metrics_dir}
    missing: list[str] = []
    issues: list[str] = []
    for key, name in json_specs.items():
        path = metrics_dir / name
        artifacts[f"{key}_path"] = path
        if not path.is_file():
            artifacts[key] = None
            missing.append(name)
            continue
        try:
            artifacts[key] = _read_json(path)
        except (json.JSONDecodeError, OSError) as exc:
            artifacts[key] = None
            missing.append(name)
            issues.append(f"{name}: {exc}")
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


def build_identity_validation(artifacts: dict[str, Any], missing_inputs: list[str]) -> pd.DataFrame:
    """Validate canonical weighted-candidate and uniform-default identities where observable."""
    rows: list[dict[str, object]] = []
    validation_summary = artifacts.get("season_aware_validation_summary") or {}
    if validation_summary:
        candidate = validation_summary.get("season_aware_fp3_candidate_summary", {})
        rows.append(
            _identity_row(
                evidence_source="retrospective_aligned_walk_forward",
                artifact_name="season_aware_validation_summary.json",
                split_name="all_walk_forward_folds",
                checkpoint=FP3_CHECKPOINT,
                role="candidate",
                family=candidate.get("candidate_family"),
                model_name=candidate.get("candidate_model_name"),
                feature_group=candidate.get("candidate_feature_group"),
                temporal_policy=candidate.get("training_policy"),
                validation_method="observed_summary_fields",
            )
        )
        rows.append(
            _identity_row(
                evidence_source="retrospective_aligned_walk_forward",
                artifact_name="season_aware_validation_summary.json",
                split_name="all_walk_forward_folds",
                checkpoint=FP3_CHECKPOINT,
                role="default",
                family=CANONICAL_FAMILY,
                model_name=CANONICAL_MODEL,
                feature_group=CANONICAL_FEATURE_GROUP,
                temporal_policy=DEFAULT_TEMPORAL_POLICY,
                validation_method="inferred_from_static_uniform_contract",
            )
        )

    for key, source, artifact_name in (
        (
            "prospective_policy_summary",
            "artifact_driven_prospective_evaluation",
            "prospective_policy_summary.json",
        ),
        (
            "prospective_replay_summary",
            "true_retrain_based_prospective_replay",
            "prospective_replay_summary.json",
        ),
    ):
        payload = artifacts.get(key) or {}
        for split in payload.get("splits", []) if isinstance(payload, dict) else []:
            if not isinstance(split, dict):
                continue
            profiles = split.get("frozen_policy_profiles") or split.get("policy_profiles") or {}
            profile = profiles.get("season_aware_frozen", {}) if isinstance(profiles, dict) else {}
            rows.append(
                _identity_row(
                    evidence_source=source,
                    artifact_name=artifact_name,
                    split_name=str(split.get("prospective_split")),
                    checkpoint=FP3_CHECKPOINT,
                    role="candidate",
                    family=profile.get("candidate_family"),
                    model_name=profile.get("candidate_model_name"),
                    feature_group=profile.get("candidate_feature_group"),
                    temporal_policy=profile.get("candidate_temporal_weighting_policy"),
                    validation_method="observed_frozen_policy_profile",
                )
            )
            rows.append(
                _identity_row(
                    evidence_source=source,
                    artifact_name=artifact_name,
                    split_name=str(split.get("prospective_split")),
                    checkpoint=FP3_CHECKPOINT,
                    role="default",
                    family=CANONICAL_FAMILY,
                    model_name=CANONICAL_MODEL,
                    feature_group=CANONICAL_FEATURE_GROUP,
                    temporal_policy=DEFAULT_TEMPORAL_POLICY,
                    validation_method="inferred_from_static_uniform_contract",
                )
            )

    shadow = artifacts.get("prospective_replay_shadow_candidates")
    if isinstance(shadow, pd.DataFrame) and not shadow.empty:
        for split_name, group in shadow.groupby("split_name", dropna=False, sort=False):
            for role, expected_policy in (
                ("uniform_default", DEFAULT_TEMPORAL_POLICY),
                ("season_aware_weighted_candidate", CANDIDATE_TEMPORAL_POLICY),
            ):
                rows.append(
                    _identity_row(
                        evidence_source="shadow_history_counterfactual_frozen_gate_evaluation",
                        artifact_name="prospective_replay_shadow_candidates.parquet",
                        split_name=str(split_name),
                        checkpoint=FP3_CHECKPOINT,
                        role="default" if role == "uniform_default" else "candidate",
                        family=_first_value(group[group["shadow_role"].eq(role)], "family"),
                        model_name=_first_value(group[group["shadow_role"].eq(role)], "model_name"),
                        feature_group=_first_value(
                            group[group["shadow_role"].eq(role)], "feature_group"
                        ),
                        temporal_policy=_first_value(
                            group[group["shadow_role"].eq(role)],
                            "temporal_weighting_policy",
                        ),
                        validation_method="observed_shadow_rows",
                        expected_temporal_policy=expected_policy,
                    )
                )

    audit_summary = artifacts.get("season_aware_candidate_audit_summary") or {}
    if audit_summary:
        rows.extend(
            [
                _identity_row(
                    evidence_source="candidate_eligibility_audit",
                    artifact_name="season_aware_candidate_audit_summary.json",
                    split_name="all_walk_forward_folds",
                    checkpoint=FP3_CHECKPOINT,
                    role="candidate",
                    family=CANONICAL_FAMILY,
                    model_name=CANONICAL_MODEL,
                    feature_group=CANONICAL_FEATURE_GROUP,
                    temporal_policy=CANDIDATE_TEMPORAL_POLICY,
                    validation_method="inferred_from_candidate_audit_contract",
                ),
                _identity_row(
                    evidence_source="candidate_eligibility_audit",
                    artifact_name="season_aware_candidate_audit_summary.json",
                    split_name="all_walk_forward_folds",
                    checkpoint=FP3_CHECKPOINT,
                    role="default",
                    family=CANONICAL_FAMILY,
                    model_name=CANONICAL_MODEL,
                    feature_group=CANONICAL_FEATURE_GROUP,
                    temporal_policy=DEFAULT_TEMPORAL_POLICY,
                    validation_method="inferred_from_candidate_audit_contract",
                ),
            ]
        )
    forensics_summary = artifacts.get("season_aware_policy_forensics_summary") or {}
    if forensics_summary:
        rows.extend(
            [
                _identity_row(
                    evidence_source="policy_forensics_and_source_lineage",
                    artifact_name="season_aware_policy_forensics_summary.json",
                    split_name="all_walk_forward_folds",
                    checkpoint=FP3_CHECKPOINT,
                    role="candidate",
                    family=CANONICAL_FAMILY,
                    model_name=CANONICAL_MODEL,
                    feature_group=CANONICAL_FEATURE_GROUP,
                    temporal_policy=CANDIDATE_TEMPORAL_POLICY,
                    validation_method="inferred_from_forensics_contract",
                ),
                _identity_row(
                    evidence_source="policy_forensics_and_source_lineage",
                    artifact_name="season_aware_policy_forensics_summary.json",
                    split_name="all_walk_forward_folds",
                    checkpoint=FP3_CHECKPOINT,
                    role="default",
                    family=CANONICAL_FAMILY,
                    model_name=CANONICAL_MODEL,
                    feature_group=CANONICAL_FEATURE_GROUP,
                    temporal_policy=DEFAULT_TEMPORAL_POLICY,
                    validation_method="inferred_from_forensics_contract",
                ),
            ]
        )
    for source in EVIDENCE_SOURCES:
        expected_artifact = _artifact_for_source(source)
        if expected_artifact in missing_inputs:
            rows.append(
                {
                    **{column: None for column in IDENTITY_COLUMNS},
                    "evidence_source": source,
                    "artifact_name": expected_artifact,
                    "split_name": "missing",
                    "checkpoint": FP3_CHECKPOINT,
                    "identity_role": "candidate",
                    "identity_valid": False,
                    "validation_method": "artifact_missing",
                    "expected_family": CANONICAL_FAMILY,
                    "expected_model_name": CANONICAL_MODEL,
                    "expected_feature_group": CANONICAL_FEATURE_GROUP,
                    "expected_temporal_weighting_policy": CANDIDATE_TEMPORAL_POLICY,
                    "mismatch_reason": "artifact_missing",
                }
            )
    return pd.DataFrame(rows, columns=IDENTITY_COLUMNS)


def build_evidence_inventory(
    artifacts: dict[str, Any],
    identity: pd.DataFrame,
    missing_inputs: list[str],
) -> pd.DataFrame:
    """Build one normalized evidence row per source and split/scope."""
    rows: list[dict[str, object]] = []
    rows.extend(_retrospective_evidence_rows(artifacts, identity))
    rows.extend(_prospective_policy_evidence_rows(artifacts, identity))
    rows.extend(_true_replay_evidence_rows(artifacts, identity))
    rows.extend(_shadow_history_evidence_rows(artifacts, identity))
    rows.extend(_candidate_audit_evidence_rows(artifacts, identity))
    rows.extend(_policy_forensics_evidence_rows(artifacts, identity))
    present_sources = {str(row.get("evidence_source")) for row in rows}
    for source in EVIDENCE_SOURCES:
        if source not in present_sources:
            artifact = _artifact_for_source(source)
            status = "missing" if artifact in missing_inputs else "unavailable"
            rows.append(
                _evidence_row(
                    evidence_source=source,
                    evidence_type="artifact_unavailable",
                    scope="artifact",
                    split_name="missing",
                    candidate_available=False,
                    candidate_selected=False,
                    candidate_eligible=False,
                    metric_name="mae_gap_sec",
                    evidence_strength="unavailable",
                    limitations="Optional artifact unavailable; not treated as negative evidence.",
                    artifact_path_or_status=status,
                    methodological_flags={
                        "selection_is_live": False,
                        "selection_is_counterfactual": False,
                    },
                )
            )
    return pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)


def build_governance_matrix(evidence: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """Create the primary governance matrix from normalized evidence rows."""
    rows: list[dict[str, object]] = []
    for record in evidence.to_dict(orient="records"):
        source = str(record.get("evidence_source"))
        split = str(record.get("split_name") or "")
        checkpoint = str(record.get("checkpoint") or FP3_CHECKPOINT)
        candidate_valid = _identity_valid(identity, source, split, "candidate")
        default_valid = _identity_valid(identity, source, split, "default")
        scope_valid = bool(candidate_valid and default_valid)
        if record.get("artifact_path_or_status") in {"missing", "unavailable"}:
            interpretation = "artifact_missing"
            limitation = str(record.get("limitations") or "artifact_missing")
        elif not scope_valid:
            interpretation = "identity_or_scope_invalid"
            limitation = "candidate/default identity or comparison scope is invalid"
        else:
            interpretation = _methodological_interpretation(record)
            limitation = str(record.get("limitations") or "")
        candidate_metric = _number_or_none(record.get("candidate_metric_value"))
        default_metric = _number_or_none(record.get("default_metric_value"))
        delta = _number_or_none(record.get("delta_vs_default"))
        rows.append(
            {
                "evidence_source": source,
                "split_name": split,
                "checkpoint": checkpoint,
                "candidate_identity_valid": bool(candidate_valid),
                "default_identity_valid": bool(default_valid),
                "comparison_scope_valid": scope_valid,
                "candidate_available": bool(record.get("candidate_available", False)),
                "candidate_eligible": bool(record.get("candidate_eligible", False)),
                "candidate_selected_live": bool(record.get("selection_is_live", False))
                and bool(record.get("candidate_selected", False)),
                "candidate_selected_counterfactual": bool(
                    record.get("selection_is_counterfactual", False)
                )
                and bool(record.get("candidate_selected", False)),
                "candidate_metric": candidate_metric,
                "default_metric": default_metric,
                "delta_vs_default": delta,
                "candidate_better_than_default": (bool(delta < 0) if delta is not None else False),
                "selection_status": _selection_status(record),
                "methodological_interpretation": interpretation,
                "governance_weight": _governance_weight(record, interpretation),
                "primary_limitation": limitation,
            }
        )
    return pd.DataFrame(rows, columns=MATRIX_COLUMNS)


def build_split_summary(matrix: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    """Summarize live and shadow eligibility status by split."""
    rows: list[dict[str, object]] = []
    split_names = sorted(
        {
            str(value)
            for value in matrix["split_name"].dropna().unique()
            if str(value) not in {"missing", "all_walk_forward_folds", ""}
        }
    )
    ledger = artifacts.get("prospective_replay_candidate_evidence_ledger")
    if isinstance(ledger, pd.DataFrame) and not ledger.empty and "split_name" in ledger:
        split_names = sorted(set(split_names) | set(ledger["split_name"].astype(str).unique()))
    for split in split_names:
        split_matrix = matrix[matrix["split_name"].astype(str).eq(split)]
        split_ledger = (
            ledger[ledger["split_name"].astype(str).eq(split)].copy()
            if isinstance(ledger, pd.DataFrame) and not ledger.empty and "split_name" in ledger
            else pd.DataFrame()
        )
        season_aware_ledger = (
            split_ledger[split_ledger["policy_profile"].astype(str).eq("season_aware_frozen")]
            if "policy_profile" in split_ledger
            else split_ledger
        )
        rows.append(
            {
                "split_name": split,
                "candidate_available_evidence_rows": int(
                    split_matrix["candidate_available"].astype(bool).sum()
                ),
                "candidate_eligible_evidence_rows": int(
                    split_matrix["candidate_eligible"].astype(bool).sum()
                ),
                "live_selected_events": _sum_bool(season_aware_ledger, "season_aware_selected"),
                "counterfactual_shadow_selected_events": _count_counterfactual_selected(
                    season_aware_ledger
                ),
                "shadow_candidate_eligible_events": _sum_bool(
                    season_aware_ledger,
                    "shadow_season_aware_candidate_eligible_under_frozen_gates",
                ),
                "max_prior_shadow_candidate_folds": _max_int(
                    season_aware_ledger,
                    "prior_shadow_candidate_events_available",
                ),
                "max_prior_shadow_candidate_prediction_rows": _max_int(
                    season_aware_ledger,
                    "prior_shadow_candidate_prediction_rows_available",
                ),
                "max_prior_shadow_aligned_rows": _max_int(
                    season_aware_ledger,
                    "prior_shadow_candidate_default_aligned_rows",
                ),
                "shadow_fold_gate_feasible": bool(
                    _max_int(season_aware_ledger, "prior_shadow_candidate_events_available")
                    >= MIN_HISTORY_FOLDS
                ),
                "shadow_prediction_gate_feasible": bool(
                    _max_int(
                        season_aware_ledger,
                        "prior_shadow_candidate_prediction_rows_available",
                    )
                    >= MIN_HISTORY_PREDICTIONS
                ),
                "live_vs_shadow_selection_disagreements": _sum_bool(
                    season_aware_ledger,
                    "live_vs_shadow_selection_disagreement",
                ),
            }
        )
    return pd.DataFrame(rows)


def build_regime_summary(artifacts: dict[str, Any]) -> pd.DataFrame:
    """Summarize cold-start versus established-season governance evidence."""
    ledger = artifacts.get("prospective_replay_candidate_evidence_ledger")
    columns = (
        "regime",
        "events",
        "live_selected_events",
        "shadow_candidate_eligible_events",
        "counterfactual_shadow_selected_events",
        "mean_shadow_candidate_prior_mae",
        "mean_shadow_default_prior_mae",
        "mean_shadow_prior_improvement_sec",
    )
    if not isinstance(ledger, pd.DataFrame) or ledger.empty:
        return pd.DataFrame(columns=columns)
    frame = ledger.copy()
    if "policy_profile" in frame:
        frame = frame[frame["policy_profile"].astype(str).eq("season_aware_frozen")].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    prior = pd.to_numeric(frame.get("current_test_season_prior_event_count"), errors="coerce")
    frame["regime"] = prior.map(lambda value: "cold_start" if value < 5 else "established_season")
    rows: list[dict[str, object]] = []
    for regime, group in frame.groupby("regime", dropna=False, sort=True):
        rows.append(
            {
                "regime": str(regime),
                "events": int(len(group)),
                "live_selected_events": _sum_bool(group, "season_aware_selected"),
                "shadow_candidate_eligible_events": _sum_bool(
                    group,
                    "shadow_season_aware_candidate_eligible_under_frozen_gates",
                ),
                "counterfactual_shadow_selected_events": _count_counterfactual_selected(group),
                "mean_shadow_candidate_prior_mae": _series_mean(
                    group, "shadow_candidate_prior_mae"
                ),
                "mean_shadow_default_prior_mae": _series_mean(group, "shadow_default_prior_mae"),
                "mean_shadow_prior_improvement_sec": _series_mean(
                    group, "shadow_prior_improvement_sec"
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_missing_evidence(missing_inputs: list[str]) -> pd.DataFrame:
    """Represent missing optional artifacts as explicit unavailable evidence."""
    rows = [
        {
            "artifact_name": name,
            "evidence_status": "unavailable",
            "treatment": "not_treated_as_negative_evidence",
        }
        for name in missing_inputs
    ]
    return pd.DataFrame(rows, columns=("artifact_name", "evidence_status", "treatment"))


def build_decision_trace(
    matrix: pd.DataFrame,
    split_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    identity: pd.DataFrame,
) -> pd.DataFrame:
    """Create a deterministic trace for the final governance state."""
    invalid_identity = not identity.empty and not identity["identity_valid"].fillna(False).all()
    missing_only = matrix["methodological_interpretation"].eq("artifact_missing").all()
    retrospective_support = _source_has_better_candidate(
        matrix, "retrospective_aligned_walk_forward"
    )
    artifact_support = _source_has_better_candidate(
        matrix, "artifact_driven_prospective_evaluation"
    )
    live_selected = _true_replay_live_selected(matrix)
    shadow_counterfactual = bool(matrix["candidate_selected_counterfactual"].astype(bool).any())
    shadow_gates_feasible = bool(
        not split_summary.empty
        and split_summary["shadow_fold_gate_feasible"].astype(bool).any()
        and split_summary["shadow_prediction_gate_feasible"].astype(bool).any()
    )
    established_regime = (
        not regime_summary.empty
        and regime_summary["regime"].astype(str).eq("established_season").any()
    )
    final_state = determine_governance_state(
        invalid_identity=invalid_identity,
        missing_only=missing_only,
        retrospective_support=retrospective_support,
        artifact_support=artifact_support,
        live_selected=live_selected,
        shadow_counterfactual=shadow_counterfactual,
        shadow_gates_feasible=shadow_gates_feasible,
        established_regime=established_regime,
    )
    rows = [
        _trace_row(
            "identity_and_scope_integrity",
            not invalid_identity,
            "Canonical candidate/default identities and comparison scope are valid.",
            "artifact_integrity_issue_requires_manual_review",
        ),
        _trace_row(
            "retrospective_aligned_validation_support",
            retrospective_support,
            "Retrospective aligned walk-forward evidence supports lower weighted-candidate MAE.",
            "candidate_evidence_inconclusive",
        ),
        _trace_row(
            "artifact_driven_prospective_support",
            artifact_support,
            (
                "Artifact-driven prospective evaluation supports the weighted candidate in at "
                "least one split."
            ),
            "candidate_evidence_inconclusive",
        ),
        _trace_row(
            "original_true_replay_live_selection_observed",
            live_selected,
            "Original true replay selected the weighted candidate live at least once.",
            "candidate_requires_more_live_prospective_evidence",
        ),
        _trace_row(
            "shadow_history_frozen_gates_feasible",
            shadow_gates_feasible,
            "Legal prior-only shadow history can evaluate frozen gates.",
            "candidate_requires_more_seasons",
        ),
        _trace_row(
            "shadow_history_counterfactual_eligibility_observed",
            shadow_counterfactual,
            (
                "Shadow history supports counterfactual eligibility without changing live replay "
                "behavior."
            ),
            "candidate_requires_more_live_prospective_evidence",
        ),
        _trace_row(
            "robust_deployable_superiority_established",
            False,
            "Counterfactual evidence is not observed live-policy deployment performance.",
            "candidate_requires_more_live_prospective_evidence",
        ),
        {
            "decision_step": "final_governance_state",
            "passed": True,
            "rationale": _primary_rationale(final_state),
            "controlled_recommendation_state": final_state,
        },
    ]
    return pd.DataFrame(rows)


def determine_governance_state(
    *,
    invalid_identity: bool,
    missing_only: bool,
    retrospective_support: bool,
    artifact_support: bool,
    live_selected: bool,
    shadow_counterfactual: bool,
    shadow_gates_feasible: bool,
    established_regime: bool,
) -> str:
    """Apply the conservative Milestone 33 governance decision framework."""
    if invalid_identity:
        return "artifact_integrity_issue_requires_manual_review"
    if missing_only:
        return "candidate_evidence_inconclusive"
    if shadow_counterfactual and not live_selected:
        return "candidate_requires_more_live_prospective_evidence"
    if (retrospective_support or artifact_support) and not shadow_gates_feasible:
        return "candidate_requires_more_seasons"
    if not established_regime:
        return "candidate_requires_more_seasons"
    if retrospective_support and artifact_support and live_selected:
        return "retain_guarded_policy"
    return "candidate_evidence_inconclusive"


def build_governance_summary_payload(
    *,
    artifacts: dict[str, Any],
    evidence: pd.DataFrame,
    matrix: pd.DataFrame,
    split_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    identity: pd.DataFrame,
    decision_trace: pd.DataFrame,
    missing_inputs: list[str],
    generation_issues: list[str],
) -> dict[str, object]:
    """Build the governance JSON summary."""
    final_state = str(
        decision_trace[decision_trace["decision_step"].eq("final_governance_state")][
            "controlled_recommendation_state"
        ].iloc[0]
    )
    shadow_quality = (
        (artifacts.get("prospective_replay_eligibility_audit_summary") or {}).get(
            "shadow_candidate_quality_summary",
            {},
        )
        if isinstance(artifacts.get("prospective_replay_eligibility_audit_summary"), dict)
        else {}
    )
    shadow_eligibility = (
        (artifacts.get("prospective_replay_eligibility_audit_summary") or {}).get(
            "shadow_candidate_eligibility_summary",
            {},
        )
        if isinstance(artifacts.get("prospective_replay_eligibility_audit_summary"), dict)
        else {}
    )
    true_replay_matrix = matrix[
        matrix["evidence_source"].astype(str).eq("true_retrain_based_prospective_replay")
    ]
    live_selected_count = int(true_replay_matrix["candidate_selected_live"].astype(bool).sum())
    counterfactual_selected = int(matrix["candidate_selected_counterfactual"].astype(bool).sum())
    return {
        "status": "partial" if missing_inputs else "complete",
        "season_aware_governance_available": True,
        "season_aware_governance_status": "partial" if missing_inputs else "complete",
        "final_governance_state": final_state,
        "final_recommendation": final_state,
        "primary_rationale": _primary_rationale(final_state),
        "evidence_taxonomy": list(EVIDENCE_SOURCES),
        "canonical_candidate_identity": canonical_candidate_identity(),
        "canonical_default_identity": canonical_default_identity(),
        "identity_validation_summary": {
            "rows": int(len(identity)),
            "all_observed_identities_valid": bool(
                not identity.empty and identity["identity_valid"].fillna(False).all()
            ),
            "mismatches": records_for_json(
                identity[identity["identity_valid"].fillna(False).eq(False)]
            ),
        },
        "evidence_strength_summary": evidence_strength_summary(matrix),
        "live_replay_status": {
            "weighted_candidate_live_selected": bool(live_selected_count),
            "live_selection_rows": live_selected_count,
            "interpretation": "live_replay_no_selection_observed"
            if live_selected_count == 0
            else "live_replay_selection_observed",
        },
        "shadow_history_status": {
            "shadow_persistence_enabled": _bool_from_summary(
                artifacts.get("prospective_replay_shadow_candidate_summary"),
                "shadow_candidate_persistence_enabled",
            ),
            "shadow_history_valid": _bool_from_summary(
                artifacts.get("prospective_replay_eligibility_audit_summary"),
                "shadow_history_valid",
            ),
            "counterfactual_selected_evidence_rows": counterfactual_selected,
            "counterfactual_selected_events": shadow_eligibility.get(
                "events_shadow_counterfactually_selected"
            ),
            "weighted_prior_mae": shadow_quality.get("mean_shadow_candidate_prior_mae"),
            "uniform_default_prior_mae": shadow_quality.get("mean_shadow_default_prior_mae"),
            "prior_improvement_sec": shadow_quality.get("mean_shadow_prior_improvement_sec"),
            "interpretation": "shadow_history_counterfactual_eligibility_supported"
            if counterfactual_selected
            else "candidate_evidence_not_sufficient",
        },
        "required_answers": required_answer_summary(artifacts, matrix),
        "governance_decision_trace": records_for_json(decision_trace),
        "split_summary": records_for_json(split_summary),
        "regime_summary": records_for_json(regime_summary),
        "evidence_inventory_rows": int(len(evidence)),
        "governance_matrix_rows": int(len(matrix)),
        "missing_evidence": missing_inputs,
        "known_limitations": [
            "Governance synthesis is artifact-driven and does not retrain or rerun replay.",
            (
                "Shadow-history eligibility is counterfactual diagnostic evidence, not live "
                "deployment behavior."
            ),
            (
                "The summary does not promote the weighted candidate or change thresholds, "
                "defaults, identities, or hyperparameters."
            ),
        ],
        "generated_outputs": {
            "metrics": [
                "reports/metrics/season_aware_governance_summary.json",
                "reports/metrics/season_aware_governance_matrix.csv",
                "reports/metrics/season_aware_governance_evidence_inventory.csv",
                "reports/metrics/season_aware_governance_split_summary.csv",
                "reports/metrics/season_aware_governance_regime_summary.csv",
                "reports/metrics/season_aware_governance_identity_validation.csv",
                "reports/metrics/season_aware_governance_decision_trace.csv",
                "reports/metrics/season_aware_governance_missing_evidence.csv",
            ],
            "figures": [],
        },
        "generation_issues": generation_issues,
        "generated_at": _utc_now(),
    }


def generate_governance_figures(
    *,
    figures_dir: Path,
    matrix: pd.DataFrame,
    split_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    decision_trace: pd.DataFrame,
) -> tuple[list[Path], list[str]]:
    """Generate static Matplotlib governance figures."""
    ensure_directory(figures_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(figures_dir.parent / ".matplotlib-cache"))
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    specs = (
        (
            "season_aware_governance_evidence_pathway.png",
            lambda path: plot_evidence_pathway(plt, matrix, path),
        ),
        (
            "season_aware_governance_candidate_vs_default.png",
            lambda path: plot_candidate_vs_default(plt, matrix, path),
        ),
        (
            "season_aware_governance_live_vs_shadow.png",
            lambda path: plot_live_vs_shadow(plt, split_summary, path),
        ),
        (
            "season_aware_governance_split_status.png",
            lambda path: plot_split_status(plt, split_summary, path),
        ),
        (
            "season_aware_governance_decision_trace.png",
            lambda path: plot_decision_trace(plt, decision_trace, path),
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


def plot_evidence_pathway(plt: Any, matrix: pd.DataFrame, path: Path) -> bool:
    if matrix.empty:
        return _empty_plot(plt, path, "Governance evidence pathways unavailable")
    counts = matrix["methodological_interpretation"].astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(counts)), counts.values, color="#3b6ea8")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=25, ha="right")
    ax.set_ylabel("Evidence rows")
    ax.set_title("Season-aware governance evidence pathways")
    ax.text(
        0.01,
        0.95,
        "Counterfactual shadow rows are diagnostic-only, not live selections.",
        transform=ax.transAxes,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def plot_candidate_vs_default(plt: Any, matrix: pd.DataFrame, path: Path) -> bool:
    frame = matrix[matrix["delta_vs_default"].notna()].copy()
    if frame.empty:
        return _empty_plot(plt, path, "Candidate/default metrics unavailable")
    labels = frame["evidence_source"].astype(str) + "\n" + frame["split_name"].astype(str)
    values = pd.to_numeric(frame["delta_vs_default"], errors="coerce")
    colors = ["#2d7f5e" if value < 0 else "#b55d4c" for value in values]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(frame)), values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(frame)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Candidate MAE minus default MAE")
    ax.set_title("Weighted FP3 candidate versus uniform/default evidence")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def plot_live_vs_shadow(plt: Any, split_summary: pd.DataFrame, path: Path) -> bool:
    if split_summary.empty:
        return _empty_plot(plt, path, "Live and shadow selection evidence unavailable")
    labels = split_summary["split_name"].astype(str)
    live = pd.to_numeric(split_summary["live_selected_events"], errors="coerce").fillna(0)
    shadow = pd.to_numeric(
        split_summary["counterfactual_shadow_selected_events"], errors="coerce"
    ).fillna(0)
    x = range(len(split_summary))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([value - 0.18 for value in x], live, width=0.36, label="Observed live replay")
    ax.bar(
        [value + 0.18 for value in x],
        shadow,
        width=0.36,
        label="Counterfactual shadow history",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Events")
    ax.set_title("Live replay selections versus shadow-history diagnostics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def plot_split_status(plt: Any, split_summary: pd.DataFrame, path: Path) -> bool:
    if split_summary.empty:
        return _empty_plot(plt, path, "Split governance status unavailable")
    labels = split_summary["split_name"].astype(str)
    eligible = pd.to_numeric(
        split_summary["shadow_candidate_eligible_events"], errors="coerce"
    ).fillna(0)
    available = pd.to_numeric(
        split_summary["candidate_available_evidence_rows"], errors="coerce"
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, available, label="Evidence rows available", color="#6c7a89")
    ax.bar(labels, eligible, label="Shadow eligible events", color="#2d7f5e")
    ax.set_ylabel("Count")
    ax.set_title("Season-aware governance status by split")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def plot_decision_trace(plt: Any, decision_trace: pd.DataFrame, path: Path) -> bool:
    if decision_trace.empty:
        return _empty_plot(plt, path, "Decision trace unavailable")
    frame = decision_trace[~decision_trace["decision_step"].eq("final_governance_state")].copy()
    if frame.empty:
        return _empty_plot(plt, path, "Decision trace unavailable")
    values = frame["passed"].fillna(False).astype(bool).map({True: 1, False: 0})
    colors = values.map({1: "#2d7f5e", 0: "#b55d4c"})
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(frame["decision_step"].astype(str), values, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Passed")
    ax.set_title("Conservative governance decision trace")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def canonical_candidate_identity() -> dict[str, str]:
    """Return the canonical weighted FP3 season-aware candidate identity."""
    return {
        "family": CANONICAL_FAMILY,
        "model_name": CANONICAL_MODEL,
        "feature_group": CANONICAL_FEATURE_GROUP,
        "temporal_weighting_policy": CANDIDATE_TEMPORAL_POLICY,
    }


def canonical_default_identity() -> dict[str, str]:
    """Return the canonical uniform FP3 default identity."""
    return {
        "family": CANONICAL_FAMILY,
        "model_name": CANONICAL_MODEL,
        "feature_group": CANONICAL_FEATURE_GROUP,
        "temporal_weighting_policy": DEFAULT_TEMPORAL_POLICY,
    }


def required_answer_summary(artifacts: dict[str, Any], matrix: pd.DataFrame) -> dict[str, object]:
    """Answer the Milestone 33 governance questions from saved artifacts."""
    shadow_summary = artifacts.get("prospective_replay_eligibility_audit_summary") or {}
    shadow_quality = shadow_summary.get("shadow_candidate_quality_summary", {})
    shadow_eligibility = shadow_summary.get("shadow_candidate_eligibility_summary", {})
    return {
        "retrospective_aligned_validation_supports_candidate": _source_has_better_candidate(
            matrix, "retrospective_aligned_walk_forward"
        ),
        "artifact_driven_prospective_evaluation_supports_candidate": _source_has_better_candidate(
            matrix, "artifact_driven_prospective_evaluation"
        ),
        "original_true_replay_live_weighted_selection_observed": bool(
            _true_replay_live_selected(matrix)
        ),
        "milestone_32_missing_retained_prior_evidence_explanation_supported": bool(
            shadow_eligibility.get("events_shadow_counterfactually_selected", 0)
        ),
        "frozen_gates_feasible_under_legal_shadow_history": bool(
            shadow_eligibility.get("events_shadow_candidate_eligible", 0)
        ),
        "shadow_counterfactual_eligible_events": shadow_eligibility.get(
            "events_shadow_candidate_eligible"
        ),
        "shadow_counterfactual_selected_events": shadow_eligibility.get(
            "events_shadow_counterfactually_selected"
        ),
        "weighted_prior_mae": shadow_quality.get("mean_shadow_candidate_prior_mae"),
        "uniform_default_prior_mae": shadow_quality.get("mean_shadow_default_prior_mae"),
        "shadow_prior_improvement_sec": shadow_quality.get("mean_shadow_prior_improvement_sec"),
        "shadow_margin_requirement_met_where_feasible": (
            _number_or_none(shadow_quality.get("mean_shadow_prior_improvement_sec")) is not None
            and float(shadow_quality.get("mean_shadow_prior_improvement_sec"))
            >= IMPROVEMENT_MARGIN_SEC
        ),
        "robust_deployable_superiority_claim_supported": False,
        "why_final_recommendation_remains_conservative": (
            "Original true replay live selections and shadow-history counterfactual eligibility "
            "are distinct facts; counterfactual diagnostics do not establish deployed live-policy "
            "superiority."
        ),
    }


def evidence_strength_summary(matrix: pd.DataFrame) -> dict[str, object]:
    if matrix.empty:
        return {}
    return {
        "evidence_rows": int(len(matrix)),
        "sources_available": sorted(
            matrix[~matrix["methodological_interpretation"].eq("artifact_missing")][
                "evidence_source"
            ]
            .astype(str)
            .unique()
            .tolist()
        ),
        "retrospective_support_rows": int(
            matrix["methodological_interpretation"].eq("retrospective_candidate_signal_only").sum()
        ),
        "artifact_driven_support_rows": int(
            matrix["methodological_interpretation"].eq("artifact_driven_prospective_support").sum()
        ),
        "live_replay_no_selection_rows": int(
            matrix["methodological_interpretation"].eq("live_replay_no_selection_observed").sum()
        ),
        "shadow_counterfactual_support_rows": int(
            matrix["methodological_interpretation"]
            .eq("shadow_history_counterfactual_eligibility_supported")
            .sum()
        ),
    }


def _retrospective_evidence_rows(
    artifacts: dict[str, Any], identity: pd.DataFrame
) -> list[dict[str, object]]:
    payload = artifacts.get("season_aware_validation_summary") or {}
    if not payload:
        return []
    summary = payload.get("season_aware_fp3_candidate_summary", {})
    if not isinstance(summary, dict):
        return []
    candidate = _number_or_none(summary.get("candidate_mae_gap_sec"))
    default = _number_or_none(summary.get("static_mae_gap_sec"))
    delta = candidate - default if candidate is not None and default is not None else None
    bootstrap = payload.get("bootstrap_robustness", {})
    note = ""
    if isinstance(bootstrap, dict) and bootstrap:
        note = (
            f"bootstrap_mean_delta={bootstrap.get('mean_delta')}; "
            f"ci=[{bootstrap.get('ci_low')}, {bootstrap.get('ci_high')}]"
        )
    return [
        _evidence_row(
            evidence_source="retrospective_aligned_walk_forward",
            evidence_type="retrospective_aligned_walk_forward",
            scope="all_walk_forward_folds",
            split_name="all_walk_forward_folds",
            candidate_available=True,
            candidate_selected=False,
            candidate_eligible=False,
            metric_name="mae_gap_sec",
            candidate_metric_value=candidate,
            default_metric_value=default,
            delta_vs_default=delta,
            confidence_or_uncertainty_note=note,
            evidence_strength="supportive" if delta is not None and delta < 0 else "mixed",
            limitations="Retrospective candidate signal only; not a live policy selection.",
            artifact_path_or_status="reports/metrics/season_aware_validation_summary.json",
            methodological_flags={
                "selection_is_live": False,
                "selection_is_counterfactual": False,
                "uses_retraining": False,
                "uses_prior_only_history": True,
                "uses_shadow_persistence": False,
                "leakage_audit_status": "not_applicable",
                "source_lineage_status": _source_lineage_status(identity),
            },
        )
    ]


def _prospective_policy_evidence_rows(
    artifacts: dict[str, Any], identity: pd.DataFrame
) -> list[dict[str, object]]:
    payload = artifacts.get("prospective_policy_summary") or {}
    if not payload:
        return []
    rows: list[dict[str, object]] = []
    for split in payload.get("splits", []):
        if not isinstance(split, dict):
            continue
        fp3 = _fp3_profile_rows(split)
        static = fp3.get("static_baseline", {})
        season = fp3.get("season_aware_frozen", {})
        selection_summary = split.get("candidate_selection_summary", {}).get(
            "season_aware_frozen",
            {},
        )
        candidate = _number_or_none(season.get("mae_gap_sec"))
        default = _number_or_none(static.get("mae_gap_sec"))
        rows.append(
            _evidence_row(
                evidence_source="artifact_driven_prospective_evaluation",
                evidence_type="prospective_policy_evaluation",
                scope="held_out_test_season_artifact_driven",
                split_name=str(split.get("prospective_split")),
                train_seasons=split.get("train_seasons"),
                test_season=split.get("test_season"),
                candidate_available=bool(season),
                candidate_selected=int(selection_summary.get("candidate_selected_folds", 0)) > 0,
                candidate_eligible=int(selection_summary.get("candidate_selected_folds", 0)) > 0,
                metric_name="mae_gap_sec",
                candidate_metric_value=candidate,
                default_metric_value=default,
                delta_vs_default=(
                    candidate - default if candidate is not None and default is not None else None
                ),
                evidence_strength="supportive"
                if candidate is not None and default is not None and candidate < default
                else "mixed",
                limitations=(
                    "Artifact-driven prospective evaluation uses saved walk-forward artifacts; "
                    "it is not true retrain-based replay."
                ),
                artifact_path_or_status="reports/metrics/prospective_policy_summary.json",
                methodological_flags={
                    "selection_is_live": True,
                    "selection_is_counterfactual": False,
                    "uses_retraining": False,
                    "uses_prior_only_history": True,
                    "uses_shadow_persistence": False,
                    "leakage_audit_status": _leakage_status(split.get("leakage_audit_summary")),
                    "source_lineage_status": _source_lineage_status(identity),
                },
            )
        )
    return rows


def _true_replay_evidence_rows(
    artifacts: dict[str, Any], identity: pd.DataFrame
) -> list[dict[str, object]]:
    payload = artifacts.get("prospective_replay_summary") or {}
    if not payload:
        return []
    rows: list[dict[str, object]] = []
    for split in payload.get("splits", []):
        if not isinstance(split, dict):
            continue
        fp3 = _fp3_profile_rows(split)
        static = fp3.get("static_baseline", {})
        season = fp3.get("season_aware_frozen", {})
        candidate = _number_or_none(season.get("mae_gap_sec"))
        default = _number_or_none(static.get("mae_gap_sec"))
        selection_summary = split.get("selection_summary", {}).get("season_aware_frozen", {})
        selected = int(selection_summary.get("candidate_selected_folds", 0) or 0) > 0
        rows.append(
            _evidence_row(
                evidence_source="true_retrain_based_prospective_replay",
                evidence_type="true_retrain_based_prospective_replay",
                scope="held_out_test_season_true_replay",
                split_name=str(split.get("prospective_split")),
                train_seasons=split.get("train_seasons"),
                test_season=split.get("test_season"),
                candidate_available=bool(season),
                candidate_selected=selected,
                candidate_eligible=selected,
                metric_name="mae_gap_sec",
                candidate_metric_value=candidate,
                default_metric_value=default,
                delta_vs_default=(
                    candidate - default if candidate is not None and default is not None else None
                ),
                evidence_strength="operational" if selected else "limited",
                limitations=(
                    "Original true replay live selection behavior; shadow diagnostics are excluded."
                ),
                artifact_path_or_status="reports/metrics/prospective_replay_summary.json",
                methodological_flags={
                    "selection_is_live": True,
                    "selection_is_counterfactual": False,
                    "uses_retraining": True,
                    "uses_prior_only_history": True,
                    "uses_shadow_persistence": False,
                    "leakage_audit_status": _leakage_status(split.get("leakage_audit_summary")),
                    "source_lineage_status": _source_lineage_status(identity),
                },
            )
        )
    return rows


def _shadow_history_evidence_rows(
    artifacts: dict[str, Any], identity: pd.DataFrame
) -> list[dict[str, object]]:
    ledger = artifacts.get("prospective_replay_candidate_evidence_ledger")
    if not isinstance(ledger, pd.DataFrame) or ledger.empty:
        return []
    frame = ledger.copy()
    if "policy_profile" in frame:
        frame = frame[frame["policy_profile"].astype(str).eq("season_aware_frozen")].copy()
    rows: list[dict[str, object]] = []
    for split, group in frame.groupby("split_name", dropna=False, sort=False):
        available = bool(
            group.get("shadow_candidate_prediction_available_for_current_event", pd.Series())
            .fillna(False)
            .astype(bool)
            .any()
        )
        eligible = bool(
            group.get(
                "shadow_season_aware_candidate_eligible_under_frozen_gates",
                pd.Series(),
            )
            .fillna(False)
            .astype(bool)
            .any()
        )
        selected = bool(
            group.get("shadow_history_counterfactual_selection", pd.Series(dtype=object))
            .astype(str)
            .eq("season_aware_weighted_candidate")
            .any()
        )
        candidate_metric = _series_mean(group, "shadow_candidate_prior_mae")
        default_metric = _series_mean(group, "shadow_default_prior_mae")
        rows.append(
            _evidence_row(
                evidence_source="shadow_history_counterfactual_frozen_gate_evaluation",
                evidence_type="counterfactual_frozen_gate_evaluation_from_legal_shadow_history",
                scope="prior_only_shadow_history",
                split_name=str(split),
                train_seasons=_first_value(group, "train_seasons"),
                test_season=_first_value(group, "test_season"),
                candidate_available=available,
                candidate_selected=selected,
                candidate_eligible=eligible,
                metric_name="prior_aligned_mae_gap_sec",
                candidate_metric_value=candidate_metric,
                default_metric_value=default_metric,
                delta_vs_default=(
                    candidate_metric - default_metric
                    if candidate_metric is not None and default_metric is not None
                    else None
                ),
                evidence_strength="diagnostic_counterfactual" if selected else "limited",
                limitations=(
                    "Legal prior-only shadow history; counterfactual eligibility is not original "
                    "live replay selection."
                ),
                artifact_path_or_status=(
                    "reports/metrics/prospective_replay_candidate_evidence_ledger.csv"
                ),
                methodological_flags={
                    "selection_is_live": False,
                    "selection_is_counterfactual": True,
                    "uses_retraining": True,
                    "uses_prior_only_history": True,
                    "uses_shadow_persistence": True,
                    "leakage_audit_status": "valid"
                    if group.get("shadow_history_scope_valid", pd.Series())
                    .fillna(False)
                    .astype(bool)
                    .all()
                    else "invalid_or_unavailable",
                    "source_lineage_status": _source_lineage_status(identity),
                },
            )
        )
    return rows


def _candidate_audit_evidence_rows(
    artifacts: dict[str, Any], identity: pd.DataFrame
) -> list[dict[str, object]]:
    payload = artifacts.get("season_aware_candidate_audit_summary") or {}
    if not payload:
        return []
    live = payload.get("live_gate_summary", {})
    history = payload.get("history_summary", {})
    selected = int(live.get("candidate_selected_folds", 0) or 0) > 0
    eligible = int(live.get("audited_candidate_eligible_folds", 0) or 0) > 0
    return [
        _evidence_row(
            evidence_source="candidate_eligibility_audit",
            evidence_type="candidate_eligibility_and_gate_audit",
            scope="all_walk_forward_folds",
            split_name="all_walk_forward_folds",
            candidate_available=int(
                payload.get("candidate_availability", {}).get("weighted_candidate_rows", 0) or 0
            )
            > 0,
            candidate_selected=selected,
            candidate_eligible=eligible,
            metric_name="mean_improvement_delta_sec",
            candidate_metric_value=None,
            default_metric_value=None,
            delta_vs_default=_number_or_none(history.get("mean_improvement_delta_sec")),
            evidence_strength="supportive" if eligible else "limited",
            limitations="Artifact audit of historical candidate gates; not a new policy rule.",
            artifact_path_or_status="reports/metrics/season_aware_candidate_audit_summary.json",
            methodological_flags={
                "selection_is_live": True,
                "selection_is_counterfactual": False,
                "uses_retraining": False,
                "uses_prior_only_history": True,
                "uses_shadow_persistence": False,
                "leakage_audit_status": "valid"
                if not payload.get("artifact_alignment_summary", {}).get(
                    "current_event_in_history",
                    True,
                )
                else "invalid",
                "source_lineage_status": _source_lineage_status(identity),
            },
        )
    ]


def _policy_forensics_evidence_rows(
    artifacts: dict[str, Any], identity: pd.DataFrame
) -> list[dict[str, object]]:
    payload = artifacts.get("season_aware_policy_forensics_summary") or {}
    if not payload:
        return []
    reconstruction = payload.get("reconstruction_summary", {})
    selected = int(payload.get("selected_fold_summary", {}).get("selected_folds", 0) or 0) > 0
    candidate = _number_or_none(reconstruction.get("saved_fp3_mae_gap_sec"))
    default = _number_or_none(reconstruction.get("static_fp3_mae_gap_sec"))
    source = payload.get("static_source_verification", {})
    return [
        _evidence_row(
            evidence_source="policy_forensics_and_source_lineage",
            evidence_type="policy_forensics_and_source_lineage",
            scope="all_walk_forward_folds",
            split_name="all_walk_forward_folds",
            candidate_available=bool(reconstruction),
            candidate_selected=selected,
            candidate_eligible=selected,
            metric_name="mae_gap_sec",
            candidate_metric_value=candidate,
            default_metric_value=default,
            delta_vs_default=(
                candidate - default if candidate is not None and default is not None else None
            ),
            evidence_strength="supportive" if selected else "limited",
            limitations=(
                "Forensics validates saved live season-aware artifacts and source lineage; "
                "counterfactual labels remain diagnostic."
            ),
            artifact_path_or_status="reports/metrics/season_aware_policy_forensics_summary.json",
            methodological_flags={
                "selection_is_live": True,
                "selection_is_counterfactual": False,
                "uses_retraining": False,
                "uses_prior_only_history": True,
                "uses_shadow_persistence": False,
                "leakage_audit_status": "valid",
                "source_lineage_status": "verified"
                if source.get("static_source_verified")
                else "unverified",
            },
        )
    ]


def _evidence_row(
    *,
    evidence_source: str,
    evidence_type: str,
    scope: str,
    split_name: str,
    candidate_available: bool,
    candidate_selected: bool,
    candidate_eligible: bool,
    metric_name: str,
    candidate_metric_value: object = None,
    default_metric_value: object = None,
    delta_vs_default: object = None,
    train_seasons: object = None,
    test_season: object = None,
    confidence_or_uncertainty_note: str = "",
    evidence_strength: str,
    limitations: str,
    artifact_path_or_status: str,
    methodological_flags: dict[str, object] | None = None,
) -> dict[str, object]:
    flags = methodological_flags or {}
    return {
        "evidence_source": evidence_source,
        "evidence_type": evidence_type,
        "scope": scope,
        "split_name": split_name,
        "train_seasons": _stringify(train_seasons),
        "test_season": test_season,
        "checkpoint": FP3_CHECKPOINT,
        "candidate_identity": json.dumps(canonical_candidate_identity(), sort_keys=True),
        "default_identity": json.dumps(canonical_default_identity(), sort_keys=True),
        "candidate_available": bool(candidate_available),
        "candidate_selected": bool(candidate_selected),
        "candidate_eligible": bool(candidate_eligible),
        "selection_is_live": bool(flags.get("selection_is_live", False)),
        "selection_is_counterfactual": bool(flags.get("selection_is_counterfactual", False)),
        "uses_retraining": bool(flags.get("uses_retraining", False)),
        "uses_prior_only_history": bool(flags.get("uses_prior_only_history", False)),
        "uses_shadow_persistence": bool(flags.get("uses_shadow_persistence", False)),
        "leakage_audit_status": str(flags.get("leakage_audit_status", "unavailable")),
        "source_lineage_status": str(flags.get("source_lineage_status", "unavailable")),
        "metric_name": metric_name,
        "candidate_metric_value": _number_or_none(candidate_metric_value),
        "default_metric_value": _number_or_none(default_metric_value),
        "delta_vs_default": _number_or_none(delta_vs_default),
        "confidence_or_uncertainty_note": confidence_or_uncertainty_note,
        "evidence_strength": evidence_strength,
        "limitations": limitations,
        "artifact_path_or_status": artifact_path_or_status,
    }


def _identity_row(
    *,
    evidence_source: str,
    artifact_name: str,
    split_name: str,
    checkpoint: str,
    role: str,
    family: object,
    model_name: object,
    feature_group: object,
    temporal_policy: object,
    validation_method: str,
    expected_temporal_policy: str | None = None,
) -> dict[str, object]:
    expected = (
        canonical_candidate_identity() if role == "candidate" else canonical_default_identity()
    )
    if expected_temporal_policy is not None:
        expected["temporal_weighting_policy"] = expected_temporal_policy
    observed = {
        "family": None if pd.isna(family) else str(family),
        "model_name": None if pd.isna(model_name) else str(model_name),
        "feature_group": None if pd.isna(feature_group) else str(feature_group),
        "temporal_weighting_policy": None if pd.isna(temporal_policy) else str(temporal_policy),
    }
    mismatches = [
        key for key, expected_value in expected.items() if observed.get(key) != expected_value
    ]
    return {
        "evidence_source": evidence_source,
        "artifact_name": artifact_name,
        "split_name": split_name,
        "checkpoint": checkpoint,
        "identity_role": role,
        "identity_valid": not mismatches,
        "validation_method": validation_method,
        "family": observed["family"],
        "model_name": observed["model_name"],
        "feature_group": observed["feature_group"],
        "temporal_weighting_policy": observed["temporal_weighting_policy"],
        "expected_family": expected["family"],
        "expected_model_name": expected["model_name"],
        "expected_feature_group": expected["feature_group"],
        "expected_temporal_weighting_policy": expected["temporal_weighting_policy"],
        "mismatch_reason": ";".join(mismatches) if mismatches else "",
    }


def _identity_valid(identity: pd.DataFrame, source: str, split: str, role: str) -> bool:
    if identity.empty:
        return False
    scoped = identity[
        identity["evidence_source"].astype(str).eq(source)
        & identity["identity_role"].astype(str).eq(role)
    ].copy()
    if scoped.empty:
        return False
    exact = scoped[scoped["split_name"].astype(str).eq(split)]
    chosen = exact if not exact.empty else scoped
    return bool(chosen["identity_valid"].fillna(False).all())


def _methodological_interpretation(record: dict[str, object]) -> str:
    source = str(record.get("evidence_source"))
    delta = _number_or_none(record.get("delta_vs_default"))
    selected = bool(record.get("candidate_selected", False))
    if source == "retrospective_aligned_walk_forward":
        return "retrospective_candidate_signal_only"
    if source == "artifact_driven_prospective_evaluation":
        return (
            "artifact_driven_prospective_support"
            if delta is not None and delta < 0
            else "candidate_evidence_not_sufficient"
        )
    if source == "true_retrain_based_prospective_replay":
        return "live_replay_selection_observed" if selected else "live_replay_no_selection_observed"
    if source == "shadow_history_counterfactual_frozen_gate_evaluation":
        return (
            "shadow_history_counterfactual_eligibility_supported"
            if selected
            else "candidate_evidence_not_sufficient"
        )
    if source in {"candidate_eligibility_audit", "policy_forensics_and_source_lineage"}:
        return (
            "artifact_driven_prospective_support"
            if selected
            else "candidate_evidence_not_sufficient"
        )
    return "candidate_evidence_not_sufficient"


def _governance_weight(record: dict[str, object], interpretation: str) -> str:
    if interpretation in {"artifact_missing", "identity_or_scope_invalid"}:
        return "none"
    if bool(record.get("selection_is_counterfactual", False)):
        return "diagnostic_counterfactual"
    if bool(record.get("selection_is_live", False)) and bool(record.get("uses_retraining", False)):
        return "high_operational"
    if bool(record.get("selection_is_live", False)):
        return "medium_artifact_driven"
    return "supporting_signal"


def _selection_status(record: dict[str, object]) -> str:
    if bool(record.get("selection_is_live", False)) and bool(
        record.get("candidate_selected", False)
    ):
        return "live_replay_selection"
    if bool(record.get("selection_is_counterfactual", False)) and bool(
        record.get("candidate_selected", False)
    ):
        return "shadow_history_counterfactual_selection"
    if bool(record.get("selection_is_live", False)):
        return "live_replay_default_or_no_candidate_selection"
    if bool(record.get("candidate_eligible", False)):
        return "eligible_diagnostic_only"
    return "not_selected_or_not_eligible"


def _artifact_for_source(source: str) -> str:
    return {
        "retrospective_aligned_walk_forward": "season_aware_validation_summary.json",
        "artifact_driven_prospective_evaluation": "prospective_policy_summary.json",
        "true_retrain_based_prospective_replay": "prospective_replay_summary.json",
        "shadow_history_counterfactual_frozen_gate_evaluation": (
            "prospective_replay_candidate_evidence_ledger.csv"
        ),
        "candidate_eligibility_audit": "season_aware_candidate_audit_summary.json",
        "policy_forensics_and_source_lineage": "season_aware_policy_forensics_summary.json",
    }[source]


def _fp3_profile_rows(split: dict[str, Any]) -> dict[str, dict[str, object]]:
    rows = split.get("fp3_summary", [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("policy_profile")): row
        for row in rows
        if isinstance(row, dict) and str(row.get("checkpoint", FP3_CHECKPOINT)) == FP3_CHECKPOINT
    }


def _source_has_better_candidate(matrix: pd.DataFrame, source: str) -> bool:
    if matrix.empty:
        return False
    scoped = matrix[matrix["evidence_source"].astype(str).eq(source)]
    if scoped.empty:
        return False
    return bool((pd.to_numeric(scoped["delta_vs_default"], errors="coerce") < 0).any())


def _true_replay_live_selected(matrix: pd.DataFrame) -> bool:
    if matrix.empty:
        return False
    scoped = matrix[
        matrix["evidence_source"].astype(str).eq("true_retrain_based_prospective_replay")
    ]
    if scoped.empty:
        return False
    return bool(scoped["candidate_selected_live"].fillna(False).astype(bool).any())


def _primary_rationale(final_state: str) -> str:
    if final_state == "artifact_integrity_issue_requires_manual_review":
        return "At least one observed canonical identity or comparison scope is invalid."
    if final_state == "candidate_requires_more_live_prospective_evidence":
        return (
            "The weighted candidate has supportive retrospective/artifact/shadow diagnostics, "
            "but shadow-history counterfactual eligibility is not original live replay selection."
        )
    if final_state == "candidate_requires_more_seasons":
        return "More legally prior prospective seasons are needed before robust governance."
    if final_state == "retain_guarded_policy":
        return "Evidence supports keeping opt-in guarded behavior without changing defaults."
    if final_state == "retain_static_policy":
        return "Available evidence does not justify moving beyond the static policy."
    return "Available artifact evidence is incomplete or mixed."


def _trace_row(
    decision_step: str,
    passed: bool,
    rationale: str,
    controlled_state: str,
) -> dict[str, object]:
    return {
        "decision_step": decision_step,
        "passed": bool(passed),
        "rationale": rationale,
        "controlled_recommendation_state": controlled_state,
    }


def _leakage_status(summary: object) -> str:
    if not isinstance(summary, dict) or not summary:
        return "unavailable"
    return "valid" if bool(summary.get("all_rows_valid", False)) else "invalid"


def _source_lineage_status(identity: pd.DataFrame) -> str:
    if identity.empty:
        return "unavailable"
    if identity["identity_valid"].fillna(False).all():
        return "verified_or_inferred_valid"
    return "identity_mismatch"


def _bool_from_summary(summary: object, key: str) -> bool | None:
    if not isinstance(summary, dict):
        return None
    value = summary.get(key)
    return bool(value) if value is not None else None


def _count_counterfactual_selected(frame: pd.DataFrame) -> int:
    if frame.empty or "shadow_history_counterfactual_selection" not in frame:
        return 0
    return int(
        frame["shadow_history_counterfactual_selection"]
        .astype(str)
        .eq("season_aware_weighted_candidate")
        .sum()
    )


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
    if values.empty:
        return None
    return values.iloc[0]


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


def _stringify(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


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
