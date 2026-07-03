from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from f1_prediction.cli import app
from f1_prediction.config import DataConfig, load_model_config
from f1_prediction.modeling.champion_policy import ChampionSelectionMode
from f1_prediction.modeling.prospective_policy_evaluation import build_frozen_policy_profiles
from f1_prediction.modeling.prospective_replay import align_candidate_default
from f1_prediction.modeling.prospective_replay_eligibility_audit import (
    ProspectiveReplayEligibilityAuditSummary,
    build_artifact_driven_eligibility_comparison,
    build_candidate_evidence_ledger,
    build_gate_feasibility,
    build_live_selection_consistency,
    create_prospective_replay_eligibility_audit_report,
    history_scope_valid,
    load_replay_eligibility_artifacts,
    prior_metrics,
)


def test_prospective_replay_eligibility_audit_cli_registration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    model_config = load_model_config()
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)
    monkeypatch.setattr(
        "f1_prediction.cli.load_model_config",
        lambda config_path=None, project_root=None: model_config,
    )

    def fake_report(*args, **kwargs):
        return ProspectiveReplayEligibilityAuditSummary(
            status="complete",
            summary_path=tmp_path / "summary.json",
            table_paths=(),
            figure_paths=(),
            missing_inputs=(),
            generation_issues=(),
        )

    monkeypatch.setattr(
        "f1_prediction.cli.create_prospective_replay_eligibility_audit_report",
        fake_report,
    )

    result = CliRunner().invoke(app, ["prospective-replay-eligibility-audit"])

    assert result.exit_code == 0
    assert "Prospective replay eligibility audit complete" in result.output


def test_candidate_ledger_detects_trained_non_selected_candidate_not_persisted(
    tmp_path: Path,
) -> None:
    artifacts = _synthetic_artifacts(tmp_path)
    profiles = build_frozen_policy_profiles(
        load_model_config(),
        profile_names=("static_baseline", "guarded_baseline", "season_aware_frozen"),
        uncertainty="conformal_predicted_gap_bucket",
    )

    ledger = build_candidate_evidence_ledger(artifacts, profiles)
    season_aware = ledger[ledger["policy_profile"].eq("season_aware_frozen")]

    assert season_aware["candidate_training_completed"].all()
    assert season_aware["candidate_prediction_available_for_current_event"].all()
    assert not season_aware["candidate_prediction_persisted_for_current_event"].any()
    assert season_aware["primary_blocking_reason"].eq("candidate_prediction_not_persisted").all()


def test_candidate_ledger_distinguishes_missing_training_from_missing_persistence(
    tmp_path: Path,
) -> None:
    artifacts = _synthetic_artifacts(tmp_path)
    artifacts["manifest"] = artifacts["manifest"][
        ~artifacts["manifest"]["policy_profile"].eq("season_aware_frozen")
    ].copy()
    profiles = build_frozen_policy_profiles(
        load_model_config(),
        profile_names=("season_aware_frozen",),
        uncertainty="conformal_predicted_gap_bucket",
    )

    ledger = build_candidate_evidence_ledger(artifacts, profiles)
    row = ledger[ledger["policy_profile"].eq("season_aware_frozen")].iloc[0]

    assert not row["candidate_training_completed"]
    assert "candidate_not_trained" in row["all_blocking_reasons"]


def test_prior_history_filters_current_and_future_events(tmp_path: Path) -> None:
    artifacts = _synthetic_artifacts(tmp_path)
    profiles = build_frozen_policy_profiles(
        load_model_config(),
        profile_names=("season_aware_frozen",),
        uncertainty="conformal_predicted_gap_bucket",
    )

    ledger = build_candidate_evidence_ledger(artifacts, profiles)
    rows = ledger[ledger["event_slug"].eq("b")]

    assert rows["current_event_excluded_from_candidate_history"].all()
    assert rows["future_test_season_event_excluded_from_candidate_history"].all()
    assert rows["future_season_excluded_from_candidate_history"].all()


def test_cold_start_and_history_gates_are_computed(tmp_path: Path) -> None:
    artifacts = _synthetic_artifacts(tmp_path)
    profiles = build_frozen_policy_profiles(
        load_model_config(),
        profile_names=("season_aware_frozen",),
        uncertainty="conformal_predicted_gap_bucket",
    )

    ledger = build_candidate_evidence_ledger(artifacts, profiles)
    season_aware = ledger[ledger["policy_profile"].eq("season_aware_frozen")]
    first = season_aware[season_aware["event_slug"].eq("a")].iloc[0]
    later = season_aware[season_aware["event_slug"].eq("f")].iloc[0]

    assert not first["cold_start_gate_passed"]
    assert later["cold_start_gate_passed"]
    assert not later["candidate_fold_history_gate_passed"]
    assert not later["candidate_prediction_history_gate_passed"]


def test_alignment_and_margin_use_exact_prior_rows_only() -> None:
    candidate = pd.concat(
        [
            _prediction_rows("season_aware_frozen", "2025/a", "current_season_only_with_prior"),
            _prediction_rows("season_aware_frozen", "2025/b", "current_season_only_with_prior"),
        ],
        ignore_index=True,
    )
    default = _prediction_rows("season_aware_frozen", "2025/a", "uniform", prediction_offset=0.5)

    aligned = align_candidate_default(candidate, default)
    candidate_mae, default_mae, improvement = prior_metrics(aligned)

    assert len(aligned) == 2
    assert set(aligned["event_slug"]) == {"a"}
    assert candidate_mae == pytest.approx(0.0)
    assert default_mae == pytest.approx(0.5)
    assert improvement == pytest.approx(0.5)


def test_feasibility_timeline_identifies_first_gate_event(tmp_path: Path) -> None:
    artifacts = _synthetic_artifacts(tmp_path)
    profiles = build_frozen_policy_profiles(
        load_model_config(),
        profile_names=("season_aware_frozen",),
        uncertainty="conformal_predicted_gap_bucket",
    )
    ledger = build_candidate_evidence_ledger(artifacts, profiles)

    feasibility = build_gate_feasibility(ledger)

    assert feasibility["first_event_with_cold_start_gate_passed"].iloc[0] == "2025/f"
    assert not feasibility["min_prior_candidate_folds_feasible"].iloc[0]


def test_shadow_history_reaches_gates_and_excludes_current_event(tmp_path: Path) -> None:
    artifacts = _synthetic_artifacts(tmp_path)
    artifacts["shadow"] = _shadow_artifact(events=["a", "b", "c", "d", "e", "f"])
    profiles = build_frozen_policy_profiles(
        load_model_config(),
        profile_names=("season_aware_frozen",),
        uncertainty="conformal_predicted_gap_bucket",
    )

    ledger = build_candidate_evidence_ledger(artifacts, profiles)
    season_aware = ledger[ledger["policy_profile"].eq("season_aware_frozen")]
    row = season_aware[season_aware["event_slug"].eq("f")].iloc[0]

    assert row["shadow_current_event_excluded"]
    assert row["shadow_future_test_season_events_excluded"]
    assert row["prior_shadow_candidate_default_aligned_rows"] == 10
    assert not row["shadow_candidate_prediction_history_gate_passed"]


def test_shadow_counterfactual_selection_differs_from_live_when_gates_pass(
    tmp_path: Path,
) -> None:
    artifacts = _synthetic_artifacts(tmp_path)
    artifacts["shadow"] = _shadow_artifact(events=["a", "b", "c", "d", "e", "f"], rows_per_event=20)
    profiles = build_frozen_policy_profiles(
        load_model_config(),
        profile_names=("season_aware_frozen",),
        uncertainty="conformal_predicted_gap_bucket",
    )

    ledger = build_candidate_evidence_ledger(artifacts, profiles)
    row = ledger[
        ledger["policy_profile"].eq("season_aware_frozen") & ledger["event_slug"].eq("f")
    ].iloc[0]

    assert row["shadow_candidate_prediction_history_gate_passed"]
    assert row["shadow_candidate_fold_history_gate_passed"]
    assert row["shadow_season_aware_candidate_eligible_under_frozen_gates"]
    assert row["shadow_history_counterfactual_selection"] == "season_aware_weighted_candidate"
    assert row["live_vs_shadow_selection_disagreement"]
    assert not row["season_aware_selected"]


def test_shadow_margin_failure_prevents_counterfactual_selection(tmp_path: Path) -> None:
    artifacts = _synthetic_artifacts(tmp_path)
    artifacts["shadow"] = _shadow_artifact(
        events=["a", "b", "c", "d", "e", "f"],
        rows_per_event=20,
        weighted_offset=0.4,
        default_offset=0.1,
    )
    profiles = build_frozen_policy_profiles(
        load_model_config(),
        profile_names=("season_aware_frozen",),
        uncertainty="conformal_predicted_gap_bucket",
    )

    ledger = build_candidate_evidence_ledger(artifacts, profiles)
    row = ledger[
        ledger["policy_profile"].eq("season_aware_frozen") & ledger["event_slug"].eq("f")
    ].iloc[0]

    assert row["shadow_candidate_prior_metric_available"]
    assert not row["shadow_season_aware_candidate_eligible_under_frozen_gates"]
    assert row["shadow_history_counterfactual_selection"] == "uniform_default"
    assert "margin_requirement_not_met" in row["shadow_all_blocking_reasons"]


def test_live_selection_consistency_detects_mismatch(tmp_path: Path) -> None:
    artifacts = _synthetic_artifacts(tmp_path)
    profiles = build_frozen_policy_profiles(
        load_model_config(),
        profile_names=("season_aware_frozen",),
        uncertainty="conformal_predicted_gap_bucket",
    )
    ledger = build_candidate_evidence_ledger(artifacts, profiles)
    mask = ledger["event_slug"].eq("f")
    ledger.loc[mask, "season_aware_candidate_eligible_under_frozen_gates"] = True

    consistency = build_live_selection_consistency(ledger)

    assert not consistency[consistency["event_slug"].eq("f")]["selection_consistent"].iloc[0]


def test_artifact_driven_comparison_handles_missing_optional_artifacts(
    tmp_path: Path,
) -> None:
    artifacts = _synthetic_artifacts(tmp_path)
    profiles = build_frozen_policy_profiles(
        load_model_config(),
        profile_names=("season_aware_frozen",),
        uncertainty="conformal_predicted_gap_bucket",
    )
    ledger = build_candidate_evidence_ledger(artifacts, profiles)

    comparison = build_artifact_driven_eligibility_comparison(ledger, pd.DataFrame())

    assert not comparison.empty
    assert (
        comparison["likely_explanation"].eq("missing artifact prevents definitive comparison").all()
    )


def test_future_rows_make_history_scope_invalid() -> None:
    candidate = _prediction_rows("season_aware_frozen", "2025/c", "current_season_only_with_prior")

    assert not history_scope_valid(candidate, "2025/b", ["2025/a", "2025/b", "2025/c"])


def test_create_audit_report_writes_outputs_and_figures(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_synthetic_artifacts(config.metrics_output_dir)
    _shadow_artifact(events=["a", "b", "c", "d", "e", "f"]).to_parquet(
        config.metrics_output_dir / "prospective_replay_shadow_candidates.parquet",
        index=False,
    )

    summary = create_prospective_replay_eligibility_audit_report(config, load_model_config())

    assert summary.status == "complete"
    assert (
        config.metrics_output_dir / "prospective_replay_eligibility_audit_summary.json"
    ).is_file()
    assert len(summary.figure_paths) >= 1
    summary_path = config.metrics_output_dir / "prospective_replay_eligibility_audit_summary.json"
    payload = json.loads(summary_path.read_text())
    assert payload["shadow_candidate_persistence_enabled"]


def test_existing_champion_modes_unchanged() -> None:
    assert [mode.value for mode in ChampionSelectionMode] == [
        "static",
        "nested",
        "stabilized_nested",
        "stabilized_nested_guarded",
        "season_aware_nested_guarded",
    ]


def _synthetic_artifacts(tmp_path: Path) -> dict[str, pd.DataFrame]:
    metrics_dir = tmp_path / "metrics"
    _write_synthetic_artifacts(metrics_dir)
    return load_replay_eligibility_artifacts(metrics_dir)


def _write_synthetic_artifacts(metrics_dir: Path) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    split = "prospective_replay_train_2024_test_2025"
    events = ["a", "b", "c", "d", "e", "f"]
    selections = []
    manifests = []
    leakage = []
    predictions = []
    for index, slug in enumerate(events):
        event_key = f"2025/{slug}"
        for profile in ("static_baseline", "guarded_baseline", "season_aware_frozen"):
            selections.append(
                {
                    "evaluation_type": "true_prospective_replay",
                    "prospective_split": split,
                    "train_seasons": "2024",
                    "test_season": 2025,
                    "policy_profile": profile,
                    "policy_signature": f"sig-{profile}",
                    "season": 2025,
                    "event": slug.upper(),
                    "event_slug": slug,
                    "fold_id": index + 1,
                    "checkpoint": "after_fp3",
                    "season_aware_selected": False,
                    "candidate_selected": False,
                    "candidate_selection_reason": (
                        "season_aware_cold_start" if index < 5 else "insufficient_candidate_history"
                    )
                    if profile == "season_aware_frozen"
                    else "default_retained",
                    "current_test_season_prior_event_count": index,
                    "cold_start_regime": "cold_start" if index < 5 else "early_season",
                    "history_scope_valid": True,
                }
            )
            predictions.append(
                _prediction_rows(profile, event_key, "uniform").assign(
                    prospective_split=split,
                    train_seasons="2024",
                    test_season=2025,
                    policy_signature=f"sig-{profile}",
                    candidate_selection_reason="default_retained",
                    current_test_season_prior_event_count=index,
                )
            )
        for profile, policy in (
            ("static_baseline", "uniform"),
            ("season_aware_frozen", "current_season_only_with_prior"),
        ):
            manifests.append(
                {
                    "evaluation_type": "true_prospective_replay",
                    "test_event": event_key,
                    "test_season": 2025,
                    "checkpoint": "after_fp3",
                    "policy_profile": profile,
                    "training_seasons_used": "[2024, 2025]",
                    "training_event_keys_used": json.dumps(
                        [f"2025/{prior}" for prior in events[:index]]
                    ),
                    "training_max_event_key": f"2025/{events[index - 1]}" if index else "2024/z",
                    "same_test_season_prior_events_used": json.dumps(
                        [f"2025/{prior}" for prior in events[:index]]
                    ),
                    "future_test_season_events_used": "[]",
                    "feature_columns_signature": "features",
                    "model_configuration_signature": "model",
                    "temporal_weighting_policy": policy,
                    "sample_weight_summary": "{}",
                    "training_row_count": 20 * max(index, 1),
                    "training_event_count": max(index, 1),
                    "random_state": 42,
                    "fit_timestamp": "2026-01-01T00:00:00Z",
                    "current_event_in_training": False,
                    "target_leakage_columns_used": "[]",
                }
            )
            leakage.append(
                {
                    "test_season": 2025,
                    "test_event": event_key,
                    "checkpoint": "after_fp3",
                    "policy_profile": profile,
                    "future_test_season_event_used": False,
                    "future_event_used_anywhere": False,
                    "current_event_used": False,
                    "history_scope_valid": True,
                    "leakage_status": "valid",
                    "leakage_reason": "valid",
                }
            )
    prediction_path = metrics_dir / f"{split}_predictions.parquet"
    pd.concat(predictions, ignore_index=True).to_parquet(prediction_path, index=False)
    pd.DataFrame(selections).to_csv(
        metrics_dir / "prospective_replay_selection_log.csv",
        index=False,
    )
    pd.DataFrame(manifests).to_csv(
        metrics_dir / "prospective_replay_training_manifest.csv",
        index=False,
    )
    pd.DataFrame(leakage).to_csv(
        metrics_dir / "prospective_replay_leakage_audit.csv",
        index=False,
    )
    summary = {
        "status": "complete",
        "splits": [
            {
                "prospective_split": split,
                "artifact_paths": {"predictions": str(prediction_path)},
            }
        ],
    }
    (metrics_dir / "prospective_replay_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )


def _prediction_rows(
    profile: str,
    event_key: str,
    temporal_policy: str,
    *,
    prediction_offset: float = 0.0,
) -> pd.DataFrame:
    season_text, slug = event_key.split("/", maxsplit=1)
    actuals = [0.0, 1.0]
    predictions = [value + prediction_offset for value in actuals]
    return pd.DataFrame(
        {
            "season": [int(season_text), int(season_text)],
            "event": [slug.upper(), slug.upper()],
            "event_slug": [slug, slug],
            "checkpoint": ["after_fp3", "after_fp3"],
            "driver": ["AAA", "BBB"],
            "driver_key": ["AAA", "BBB"],
            "team": ["Team", "Team"],
            "quali_position": [1, 2],
            "quali_gap_to_pole_sec": actuals,
            "predicted_quali_gap_to_pole_sec": predictions,
            "predicted_quali_position": [1, 2],
            "evaluation_type": ["true_prospective_replay", "true_prospective_replay"],
            "prospective_event_id": [event_key, event_key],
            "fold_id": [hash(slug) % 10000, hash(slug) % 10000],
            "policy_profile": [profile, profile],
            "source_temporal_weighting_policy": [temporal_policy, temporal_policy],
            "source_family": ["ablation", "ablation"],
            "source_model_name": ["random_forest", "random_forest"],
            "source_feature_group": ["base_plus_relative", "base_plus_relative"],
            "selected_family": ["ablation", "ablation"],
            "selected_model_name": ["random_forest", "random_forest"],
            "selected_feature_group": ["base_plus_relative", "base_plus_relative"],
            "season_aware_selected": [False, False],
        }
    )


def _shadow_artifact(
    *,
    events: list[str],
    rows_per_event: int = 2,
    weighted_offset: float = 0.0,
    default_offset: float = 0.4,
) -> pd.DataFrame:
    frames = []
    split = "prospective_replay_train_2024_test_2025"
    for order, slug in enumerate(events):
        event_key = f"2025/{slug}"
        frames.append(
            _shadow_rows(
                split,
                "uniform_default",
                "uniform",
                event_key,
                order,
                rows_per_event,
                default_offset,
            )
        )
        frames.append(
            _shadow_rows(
                split,
                "season_aware_weighted_candidate",
                "current_season_only_with_prior",
                event_key,
                order,
                rows_per_event,
                weighted_offset,
            )
        )
    return pd.concat(frames, ignore_index=True)


def _shadow_rows(
    split: str,
    role: str,
    policy: str,
    event_key: str,
    event_order: int,
    rows_per_event: int,
    offset: float,
) -> pd.DataFrame:
    season_text, slug = event_key.split("/", maxsplit=1)
    actuals = [float(index) / 10.0 for index in range(rows_per_event)]
    drivers = [f"D{index:02d}" for index in range(rows_per_event)]
    predictions = [value + offset for value in actuals]
    return pd.DataFrame(
        {
            "split_name": [split] * rows_per_event,
            "train_seasons": ["2024"] * rows_per_event,
            "test_season": [2025] * rows_per_event,
            "fold_id": [event_order + 1] * rows_per_event,
            "event_order": [event_order] * rows_per_event,
            "season": [int(season_text)] * rows_per_event,
            "event": [slug.upper()] * rows_per_event,
            "event_slug": [slug] * rows_per_event,
            "checkpoint": ["after_fp3"] * rows_per_event,
            "driver": drivers,
            "driver_key": drivers,
            "team": ["Team"] * rows_per_event,
            "team_key": ["team"] * rows_per_event,
            "shadow_role": [role] * rows_per_event,
            "diagnostic_only": [True] * rows_per_event,
            "prediction_available": [True] * rows_per_event,
            "prediction_gap_sec": predictions,
            "actual_gap_sec": actuals,
            "absolute_error_sec": [abs(offset)] * rows_per_event,
            "family": ["ablation"] * rows_per_event,
            "model_name": ["random_forest"] * rows_per_event,
            "feature_group": ["base_plus_relative"] * rows_per_event,
            "temporal_weighting_policy": [policy] * rows_per_event,
            "source_identity": ["{}"] * rows_per_event,
            "source_lineage_valid": [True] * rows_per_event,
            "training_completed": [True] * rows_per_event,
            "training_row_count": [40] * rows_per_event,
            "training_event_count": [2] * rows_per_event,
            "training_event_keys": ["[]"] * rows_per_event,
            "training_seasons": ["[]"] * rows_per_event,
            "training_temporal_weighting_policy": [policy] * rows_per_event,
            "training_weight_summary": ["{}"] * rows_per_event,
            "training_effective_sample_size": [40.0] * rows_per_event,
            "training_current_season_prior_event_count": [event_order] * rows_per_event,
            "current_event_excluded_from_training": [True] * rows_per_event,
            "future_test_season_events_excluded_from_training": [True] * rows_per_event,
            "future_seasons_excluded_from_training": [True] * rows_per_event,
            "shadow_eligible_for_prior_evidence": [True] * rows_per_event,
            "shadow_persistence_status": ["persisted"] * rows_per_event,
            "missing_reason": [""] * rows_per_event,
        }
    )


def _config(project_root: Path) -> DataConfig:
    return DataConfig(
        project_root=project_root,
        fastf1_cache_dir=project_root / "cache",
        lap_output_dir=project_root / "laps",
        session_metadata_output_dir=project_root / "metadata",
        clean_lap_output_dir=project_root / "clean",
        session_features_output_dir=project_root / "features",
        modeling_output_dir=project_root / "modeling",
        metrics_output_dir=project_root / "metrics",
    )
