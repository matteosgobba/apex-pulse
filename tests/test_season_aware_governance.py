import json
from pathlib import Path

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.modeling.backtest_report import build_backtest_report_payload
from f1_prediction.modeling.season_aware_governance import (
    canonical_candidate_identity,
    canonical_default_identity,
    create_season_aware_governance_report,
    determine_governance_state,
)


def test_governance_matrix_distinguishes_live_and_shadow_evidence(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_governance_artifacts(config.metrics_output_dir)

    summary = create_season_aware_governance_report(config)
    payload = json.loads(summary.summary_path.read_text())
    matrix = pd.read_csv(config.metrics_output_dir / "season_aware_governance_matrix.csv")

    assert payload["final_recommendation"] == "candidate_requires_more_live_prospective_evidence"
    assert (
        payload["required_answers"]["original_true_replay_live_weighted_selection_observed"]
        is False
    )
    assert payload["required_answers"]["frozen_gates_feasible_under_legal_shadow_history"] is True
    assert set(matrix["methodological_interpretation"]) >= {
        "retrospective_candidate_signal_only",
        "live_replay_no_selection_observed",
        "shadow_history_counterfactual_eligibility_supported",
    }
    assert not matrix[matrix["evidence_source"].eq("true_retrain_based_prospective_replay")][
        "candidate_selected_live"
    ].any()
    assert matrix[
        matrix["evidence_source"].eq("shadow_history_counterfactual_frozen_gate_evaluation")
    ]["candidate_selected_counterfactual"].any()


def test_canonical_identity_validation_accepts_expected_identity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_governance_artifacts(config.metrics_output_dir)

    create_season_aware_governance_report(config)
    identity = pd.read_csv(
        config.metrics_output_dir / "season_aware_governance_identity_validation.csv"
    )

    assert canonical_candidate_identity()["temporal_weighting_policy"] == (
        "current_season_only_with_prior"
    )
    assert canonical_default_identity()["temporal_weighting_policy"] == "uniform"
    assert identity["identity_valid"].all()


def test_identity_mismatch_is_recorded_and_blocks_aggregation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_governance_artifacts(config.metrics_output_dir, shadow_feature_group="wrong_group")

    create_season_aware_governance_report(config)
    identity = pd.read_csv(
        config.metrics_output_dir / "season_aware_governance_identity_validation.csv"
    )
    matrix = pd.read_csv(config.metrics_output_dir / "season_aware_governance_matrix.csv")

    mismatch = identity[~identity["identity_valid"]]
    assert not mismatch.empty
    assert "feature_group" in set(mismatch["mismatch_reason"])
    assert "identity_or_scope_invalid" in set(matrix["methodological_interpretation"])


def test_missing_optional_artifact_becomes_unavailable_evidence(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.metrics_output_dir.mkdir(parents=True)

    summary = create_season_aware_governance_report(config)
    matrix = pd.read_csv(config.metrics_output_dir / "season_aware_governance_matrix.csv")
    missing = pd.read_csv(
        config.metrics_output_dir / "season_aware_governance_missing_evidence.csv"
    )

    assert summary.status == "partial"
    assert set(matrix["methodological_interpretation"]) == {"artifact_missing"}
    assert "prospective_replay_summary.json" in set(missing["artifact_name"])


def test_split_and_regime_summary_count_live_and_counterfactual_separately(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_governance_artifacts(config.metrics_output_dir)

    create_season_aware_governance_report(config)
    split = pd.read_csv(config.metrics_output_dir / "season_aware_governance_split_summary.csv")
    regime = pd.read_csv(config.metrics_output_dir / "season_aware_governance_regime_summary.csv")

    replay_split = split[split["split_name"].eq("prospective_replay_train_2024_test_2025")]
    assert int(replay_split["live_selected_events"].iloc[0]) == 0
    assert int(replay_split["counterfactual_shadow_selected_events"].iloc[0]) == 1
    established = regime[regime["regime"].eq("established_season")]
    assert int(established["shadow_candidate_eligible_events"].iloc[0]) == 1
    assert int(established["live_selected_events"].iloc[0]) == 0


def test_margin_failure_prevents_counterfactual_selection_claim(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_governance_artifacts(config.metrics_output_dir, margin_pass=False)

    create_season_aware_governance_report(config)
    matrix = pd.read_csv(config.metrics_output_dir / "season_aware_governance_matrix.csv")
    shadow = matrix[
        matrix["evidence_source"].eq("shadow_history_counterfactual_frozen_gate_evaluation")
    ]

    assert not shadow["candidate_selected_counterfactual"].any()
    assert set(shadow["methodological_interpretation"]) == {"candidate_evidence_not_sufficient"}


def test_governance_figures_generate_from_minimal_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_governance_artifacts(config.metrics_output_dir)

    summary = create_season_aware_governance_report(config)

    assert len(summary.figure_paths) == 5
    for path in summary.figure_paths:
        assert path.is_file()


def test_backtest_report_exposes_governance_summary_fields() -> None:
    payload = build_backtest_report_payload(
        quality={"n_rows": 1, "n_events": 1, "checkpoints": ["after_fp3"]},
        baseline_metrics={},
        tabular_metrics=None,
        season_aware_governance_summary={
            "status": "complete",
            "final_recommendation": "candidate_requires_more_live_prospective_evidence",
            "primary_rationale": "counterfactual evidence is diagnostic",
            "live_replay_status": {"interpretation": "live_replay_no_selection_observed"},
            "shadow_history_status": {
                "interpretation": "shadow_history_counterfactual_eligibility_supported"
            },
            "evidence_strength_summary": {"shadow_counterfactual_support_rows": 1},
        },
    )

    assert payload["season_aware_governance_available"] is True
    assert (
        payload["season_aware_governance_final_recommendation"]
        == "candidate_requires_more_live_prospective_evidence"
    )


def test_governance_decision_logic_remains_conservative_for_counterfactual_only() -> None:
    state = determine_governance_state(
        invalid_identity=False,
        missing_only=False,
        retrospective_support=True,
        artifact_support=True,
        live_selected=False,
        shadow_counterfactual=True,
        shadow_gates_feasible=True,
        established_regime=True,
    )

    assert state == "candidate_requires_more_live_prospective_evidence"


def _config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        project_root=tmp_path,
        fastf1_cache_dir=tmp_path / "cache",
        lap_output_dir=tmp_path / "data/raw/laps",
        session_metadata_output_dir=tmp_path / "data/raw/session_metadata",
        clean_lap_output_dir=tmp_path / "data/interim/clean_laps",
        session_features_output_dir=tmp_path / "data/processed/session_features",
        modeling_output_dir=tmp_path / "data/processed/modeling",
        metrics_output_dir=tmp_path / "reports/metrics",
    )


def _write_governance_artifacts(
    metrics_dir: Path,
    *,
    shadow_feature_group: str = "base_plus_relative",
    margin_pass: bool = True,
) -> None:
    metrics_dir.mkdir(parents=True)
    _write_json(
        metrics_dir / "season_aware_validation_summary.json",
        {
            "status": "complete",
            "season_aware_fp3_candidate_summary": {
                "candidate_family": "ablation",
                "candidate_model_name": "random_forest",
                "candidate_feature_group": "base_plus_relative",
                "training_policy": "current_season_only_with_prior",
                "candidate_mae_gap_sec": 0.7,
                "static_mae_gap_sec": 0.9,
            },
            "bootstrap_robustness": {"mean_delta": -0.2, "ci_low": -0.3, "ci_high": -0.1},
        },
    )
    _write_json(
        metrics_dir / "prospective_policy_summary.json",
        {
            "status": "complete",
            "splits": [
                _policy_split(
                    "prospective_train_2024_test_2025",
                    season_aware_mae=0.75,
                    static_mae=0.9,
                    selected_folds=1,
                )
            ],
        },
    )
    _write_json(
        metrics_dir / "prospective_replay_summary.json",
        {
            "status": "complete",
            "splits": [
                _replay_split(
                    "prospective_replay_train_2024_test_2025",
                    season_aware_mae=0.9,
                    static_mae=0.9,
                    selected_folds=0,
                )
            ],
        },
    )
    _write_json(
        metrics_dir / "prospective_replay_shadow_candidate_summary.json",
        {
            "status": "complete",
            "shadow_candidate_persistence_enabled": True,
            "shadow_candidate_persistence_status": "complete",
        },
    )
    eligible = bool(margin_pass)
    _write_json(
        metrics_dir / "prospective_replay_eligibility_audit_summary.json",
        {
            "status": "complete",
            "shadow_history_valid": True,
            "shadow_candidate_eligibility_summary": {
                "events": 6,
                "events_shadow_candidate_eligible": 1 if eligible else 0,
                "events_shadow_counterfactually_selected": 1 if eligible else 0,
            },
            "shadow_candidate_quality_summary": {
                "available": True,
                "mean_shadow_candidate_prior_mae": 0.7 if margin_pass else 0.86,
                "mean_shadow_default_prior_mae": 0.9,
                "mean_shadow_prior_improvement_sec": 0.2 if margin_pass else 0.04,
            },
        },
    )
    _write_json(
        metrics_dir / "season_aware_candidate_audit_summary.json",
        {
            "status": "complete",
            "candidate_availability": {
                "weighted_candidate_rows": 120,
                "default_candidate_rows": 120,
            },
            "live_gate_summary": {
                "audited_candidate_eligible_folds": 1,
                "candidate_selected_folds": 1,
            },
            "history_summary": {"mean_improvement_delta_sec": -0.1},
            "artifact_alignment_summary": {"current_event_in_history": False},
        },
    )
    _write_json(
        metrics_dir / "season_aware_policy_forensics_summary.json",
        {
            "status": "complete",
            "reconstruction_summary": {
                "saved_fp3_mae_gap_sec": 0.8,
                "static_fp3_mae_gap_sec": 0.9,
            },
            "selected_fold_summary": {"selected_folds": 1},
            "static_source_verification": {"static_source_verified": True},
        },
    )
    _shadow_candidates(shadow_feature_group).to_parquet(
        metrics_dir / "prospective_replay_shadow_candidates.parquet"
    )
    _ledger(margin_pass=margin_pass).to_csv(
        metrics_dir / "prospective_replay_candidate_evidence_ledger.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "split_name": ["prospective_replay_train_2024_test_2025"],
            "live_replay_selection": ["uniform_default"],
            "shadow_weighted_available": [True],
            "selection_behavior_changed": [False],
        }
    ).to_csv(metrics_dir / "prospective_replay_shadow_vs_live_selection.csv", index=False)
    pd.DataFrame(
        {
            "split_name": ["prospective_replay_train_2024_test_2025"],
            "first_event_with_shadow_candidate_prediction_available": ["2025/a"],
            "maximum_prior_shadow_candidate_folds_observed": [5],
            "maximum_prior_shadow_candidate_prediction_rows_observed": [100],
            "maximum_prior_shadow_aligned_rows_observed": [100],
            "number_of_events_with_shadow_candidate_available": [6],
        }
    ).to_csv(metrics_dir / "prospective_replay_shadow_gate_feasibility.csv", index=False)
    pd.DataFrame().to_csv(metrics_dir / "prospective_replay_selection_log.csv", index=False)
    pd.DataFrame().to_csv(metrics_dir / "prospective_replay_event_comparison.csv", index=False)


def _policy_split(
    split_name: str,
    *,
    season_aware_mae: float,
    static_mae: float,
    selected_folds: int,
) -> dict[str, object]:
    return {
        "prospective_split": split_name,
        "train_seasons": [2024],
        "test_season": 2025,
        "frozen_policy_profiles": {"season_aware_frozen": _profile_identity()},
        "fp3_summary": [
            _fp3_row(split_name, "static_baseline", static_mae),
            _fp3_row(split_name, "season_aware_frozen", season_aware_mae),
        ],
        "candidate_selection_summary": {
            "season_aware_frozen": {"candidate_selected_folds": selected_folds}
        },
        "leakage_audit_summary": {"all_rows_valid": True},
    }


def _replay_split(
    split_name: str,
    *,
    season_aware_mae: float,
    static_mae: float,
    selected_folds: int,
) -> dict[str, object]:
    payload = _policy_split(
        split_name,
        season_aware_mae=season_aware_mae,
        static_mae=static_mae,
        selected_folds=selected_folds,
    )
    payload["policy_profiles"] = payload.pop("frozen_policy_profiles")
    payload["selection_summary"] = {
        "season_aware_frozen": {"candidate_selected_folds": selected_folds}
    }
    return payload


def _profile_identity() -> dict[str, str]:
    return {
        "candidate_family": "ablation",
        "candidate_model_name": "random_forest",
        "candidate_feature_group": "base_plus_relative",
        "candidate_temporal_weighting_policy": "current_season_only_with_prior",
    }


def _fp3_row(split_name: str, profile: str, mae: float) -> dict[str, object]:
    return {
        "prospective_split": split_name,
        "policy_profile": profile,
        "checkpoint": "after_fp3",
        "mae_gap_sec": mae,
    }


def _ledger(*, margin_pass: bool) -> pd.DataFrame:
    rows = []
    for event_order in range(6):
        eligible = margin_pass and event_order == 5
        rows.append(
            {
                "split_name": "prospective_replay_train_2024_test_2025",
                "train_seasons": "2024",
                "test_season": 2025,
                "season": 2025,
                "event_slug": chr(ord("a") + event_order),
                "checkpoint": "after_fp3",
                "policy_profile": "season_aware_frozen",
                "current_test_season_prior_event_count": event_order,
                "season_aware_selected": False,
                "shadow_candidate_prediction_available_for_current_event": True,
                "prior_shadow_candidate_events_available": event_order,
                "prior_shadow_candidate_prediction_rows_available": event_order * 20,
                "prior_shadow_candidate_default_aligned_rows": event_order * 20,
                "shadow_history_scope_valid": True,
                "shadow_candidate_prior_mae": 0.7 if margin_pass else 0.86,
                "shadow_default_prior_mae": 0.9,
                "shadow_prior_improvement_sec": 0.2 if margin_pass else 0.04,
                "shadow_season_aware_candidate_eligible_under_frozen_gates": eligible,
                "shadow_history_counterfactual_selection": (
                    "season_aware_weighted_candidate" if eligible else "uniform_default"
                ),
                "live_vs_shadow_selection_disagreement": eligible,
            }
        )
    return pd.DataFrame(rows)


def _shadow_candidates(feature_group: str) -> pd.DataFrame:
    rows = []
    for role, policy in [
        ("uniform_default", "uniform"),
        ("season_aware_weighted_candidate", "current_season_only_with_prior"),
    ]:
        rows.append(
            {
                "split_name": "prospective_replay_train_2024_test_2025",
                "event_slug": "a",
                "checkpoint": "after_fp3",
                "driver": "VER",
                "shadow_role": role,
                "diagnostic_only": True,
                "prediction_available": True,
                "absolute_error_sec": 0.1,
                "family": "ablation",
                "model_name": "random_forest",
                "feature_group": feature_group,
                "temporal_weighting_policy": policy,
                "source_lineage_valid": True,
                "shadow_eligible_for_prior_evidence": True,
            }
        )
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
