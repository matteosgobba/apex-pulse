from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from f1_prediction.cli import app
from f1_prediction.config import DataConfig, load_model_config
from f1_prediction.modeling.champion_policy import ChampionSelectionMode
from f1_prediction.modeling.prospective_policy_evaluation import build_frozen_policy_profiles
from f1_prediction.modeling.prospective_replay import (
    EVALUATION_TYPE,
    add_replay_intervals,
    align_candidate_default,
    apply_profiles_for_event,
    compare_replay_to_artifact_driven,
    fit_source_candidate,
    leakage_row,
    prior_events_for,
    replay_bootstrap_summary,
    replay_split_id,
    season_aware_decision,
    training_manifest_row,
)
from f1_prediction.modeling.temporal_weighting import TemporalWeightingPolicy


def test_prospective_replay_cli_registration(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    model_config = load_model_config()
    captured: dict[str, object] = {}
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)
    monkeypatch.setattr(
        "f1_prediction.cli.load_model_config",
        lambda config_path=None, project_root=None: model_config,
    )
    monkeypatch.setattr(
        "f1_prediction.cli.load_feature_config",
        lambda config_path=None, project_root=None: None,
    )

    def fake_report(*args, **kwargs):
        captured.update(kwargs)
        from f1_prediction.modeling.prospective_replay import ProspectiveReplaySummary

        return ProspectiveReplaySummary(
            status="complete",
            summary_path=tmp_path / "summary.json",
            table_paths=(),
            figure_paths=(),
            generation_issues=(),
        )

    monkeypatch.setattr("f1_prediction.cli.run_prospective_policy_replay_report", fake_report)

    result = CliRunner().invoke(
        app,
        [
            "prospective-policy-replay",
            "--train-seasons",
            "2023",
            "--train-seasons",
            "2024",
            "--test-season",
            "2025",
            "--policy-profiles",
            "static_baseline",
            "--policy-profiles",
            "season_aware_frozen",
            "--uncertainty",
            "conformal_predicted_gap_bucket",
            "--min-events",
            "2",
            "--min-train-events",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert captured["train_seasons"] == (2023, 2024)
    assert captured["test_season"] == 2025
    assert captured["policy_profiles"] == ("static_baseline", "season_aware_frozen")


def test_replay_history_excludes_current_and_future_test_events() -> None:
    event_order = ["2023/a", "2024/a", "2025/a", "2025/b", "2025/c"]

    history = prior_events_for(
        "2025/b",
        event_order,
        train_seasons=(2023, 2024),
        test_season=2025,
    )

    assert history == ["2023/a", "2024/a", "2025/a"]
    assert "2025/b" not in history
    assert "2025/c" not in history


def test_training_manifest_records_history_and_leakage_audit_flags_future_rows() -> None:
    model_config = load_model_config()
    train = pd.DataFrame(
        {
            "season": [2023, 2025],
            "event_slug": ["a", "a"],
            "checkpoint": ["after_fp3", "after_fp3"],
        }
    )
    test = pd.DataFrame(
        {
            "season": [2025],
            "event_slug": ["b"],
            "checkpoint": ["after_fp3"],
        }
    )
    row = training_manifest_row(
        event_key="2025/b",
        test=test,
        train=train,
        legal_train_events=["2023/a", "2025/a"],
        fit_payload={
            "feature_columns": ["practice_gap_feature"],
            "sample_weight_summary": {"weight_sum": 2.0},
        },
        model_config=model_config,
        temporal_policy="current_season_only_with_prior",
        policy_profile="season_aware_frozen",
        test_season=2025,
    )

    assert row["training_max_event_key"] == "2025/a"
    assert row["training_event_count"] == 2
    assert row["training_row_count"] == 2
    audit = leakage_row(row, event_order=["2023/a", "2025/a", "2025/b", "2025/c"])
    assert audit["leakage_status"] == "valid"

    bad_row = dict(row)
    bad_row["training_event_keys_used"] = '["2023/a", "2025/c"]'
    bad_audit = leakage_row(bad_row, event_order=["2023/a", "2025/a", "2025/b", "2025/c"])
    assert bad_audit["leakage_status"] == "invalid"
    assert bad_audit["future_test_season_event_used"]


def test_frozen_policy_signature_is_stable_for_replay_profiles() -> None:
    profiles = build_frozen_policy_profiles(
        load_model_config(),
        profile_names=("static_baseline", "guarded_baseline", "season_aware_frozen"),
        uncertainty="conformal_predicted_gap_bucket",
    )

    first = profiles["season_aware_frozen"].to_payload()["policy_signature"]
    second = profiles["season_aware_frozen"].to_payload()["policy_signature"]

    assert first == second


def test_static_guarded_and_season_aware_profiles_run_on_synthetic_sources() -> None:
    profiles = build_frozen_policy_profiles(
        load_model_config(),
        profile_names=("static_baseline", "guarded_baseline", "season_aware_frozen"),
        uncertainty="conformal_predicted_gap_bucket",
    )
    source = {
        "event_key": "2025/b",
        "static": _prediction_source("after_fp3", [0.0, 1.0]),
        "weighted": _prediction_source("after_fp3", [0.0, 0.9]),
        "baseline": pd.concat(
            [
                _baseline_source("after_fp1", "robust_best_push_lap", [0.0, 1.2]),
                _baseline_source("after_fp2", "robust_theoretical_best_lap", [0.0, 1.1]),
            ],
            ignore_index=True,
        ),
    }

    predictions, selections = apply_profiles_for_event(
        source=source,
        profiles=profiles,
        history=pd.DataFrame(),
        train_seasons=(2023, 2024),
        test_season=2025,
        uncertainty="conformal_predicted_gap_bucket",
    )

    assert set(predictions["policy_profile"]) == {
        "static_baseline",
        "guarded_baseline",
        "season_aware_frozen",
    }
    assert set(predictions["checkpoint"]) == {"after_fp1", "after_fp2", "after_fp3"}
    assert len(selections) == 9


def test_season_aware_decision_uses_prior_history_only_and_margin() -> None:
    profile = replace(
        build_frozen_policy_profiles(
            load_model_config(),
            profile_names=("season_aware_frozen",),
            uncertainty="conformal_predicted_gap_bucket",
        )["season_aware_frozen"],
        cold_start_threshold=1,
        min_prior_folds=1,
        min_prior_predictions=2,
        improvement_margin_sec=0.05,
    )
    history = pd.concat(
        [
            _history_rows("current_season_only_with_prior", "2025/a", 1, [0.05, 0.95]),
            _history_rows("uniform", "2025/a", 1, [0.4, 1.4]),
        ],
        ignore_index=True,
    )

    selected, reason = season_aware_decision(history, event_key="2025/b", profile=profile)

    assert selected
    assert reason == "season_aware_candidate_selected"

    no_margin_history = pd.concat(
        [
            _history_rows("current_season_only_with_prior", "2025/a", 1, [0.2, 1.2]),
            _history_rows("uniform", "2025/a", 1, [0.21, 1.21]),
        ],
        ignore_index=True,
    )
    selected, reason = season_aware_decision(
        no_margin_history,
        event_key="2025/b",
        profile=profile,
    )

    assert not selected
    assert reason == "margin_not_met"


def test_candidate_default_alignment_uses_identical_rows_only() -> None:
    candidate = pd.concat(
        [
            _history_rows("current_season_only_with_prior", "2025/a", 1, [0.0, 0.9]),
            _history_rows("current_season_only_with_prior", "2025/b", 2, [0.0, 0.8]),
        ],
        ignore_index=True,
    )
    default = _history_rows("uniform", "2025/a", 1, [0.0, 1.3])

    aligned = align_candidate_default(candidate, default)

    assert len(aligned) == 2
    assert set(aligned["event_slug"]) == {"a"}
    assert aligned["candidate_abs_error"].mean() < aligned["default_abs_error"].mean()


def test_replay_intervals_exclude_current_rows_and_require_history() -> None:
    current = _history_rows("uniform", "2025/b", 2, [0.0, 1.0])
    prior = pd.concat(
        [_history_rows("uniform", f"2025/p{i}", i, [0.0, 1.1]) for i in range(1, 12)],
        ignore_index=True,
    )

    with_interval = add_replay_intervals(
        current,
        prior,
        uncertainty="conformal_predicted_gap_bucket",
    )
    without_interval = add_replay_intervals(
        current,
        prior.head(10),
        uncertainty="conformal_predicted_gap_bucket",
    )

    assert with_interval["prediction_interval_low_sec"].notna().all()
    assert with_interval["uncertainty_method"].eq("conformal_predicted_gap_bucket").all()
    assert without_interval["prediction_interval_low_sec"].isna().all()
    assert without_interval["uncertainty_method"].eq("insufficient_history").all()


def test_replay_intervals_do_not_use_current_event_residuals_for_quantile() -> None:
    current = _history_rows("uniform", "2025/b", 2, [100.0, 101.0])
    prior = pd.concat(
        [_history_rows("uniform", f"2025/p{i}", i, [0.0, 1.1]) for i in range(1, 12)],
        ignore_index=True,
    )

    with_interval = add_replay_intervals(
        current,
        prior,
        uncertainty="conformal_predicted_gap_bucket",
    )

    assert with_interval["residual_quantile_sec"].iloc[0] == pytest.approx(0.1)


def test_weighted_fit_receives_sample_weights_from_legal_history(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_fit_and_predict(
        train,
        test,
        *,
        model_config,
        candidate_features,
        sample_weights=None,
    ):
        captured["training_events"] = sorted(train["event_slug"].unique())
        captured["weights"] = sample_weights.to_numpy().tolist()
        predictions = test.copy()
        predictions["model_name"] = "random_forest"
        predictions["predicted_quali_gap_to_pole_sec"] = predictions[
            "quali_gap_to_pole_sec"
        ].astype(float)
        predictions["predicted_quali_position"] = predictions["quali_position"]
        fitted = {"random_forest": {"after_fp3": {"feature_columns": candidate_features}}}
        return predictions, fitted

    monkeypatch.setattr(
        "f1_prediction.modeling.prospective_replay.fit_and_predict",
        fake_fit_and_predict,
    )
    train = pd.DataFrame(
        {
            "season": [2024, 2025],
            "event_slug": ["a", "a"],
            "event": ["A", "A"],
            "checkpoint": ["after_fp3", "after_fp3"],
            "driver": ["AAA", "AAA"],
            "quali_gap_to_pole_sec": [0.0, 0.2],
            "quali_position": [1, 1],
            "practice_feature": [1.0, 2.0],
        }
    )
    test = pd.DataFrame(
        {
            "season": [2025],
            "event_slug": ["b"],
            "event": ["B"],
            "checkpoint": ["after_fp3"],
            "driver": ["AAA"],
            "quali_gap_to_pole_sec": [0.1],
            "quali_position": [1],
            "practice_feature": [3.0],
        }
    )

    _, payload = fit_source_candidate(
        train=train,
        test=test,
        event_order=["2024/a", "2025/a", "2025/b", "2025/c"],
        event_key="2025/b",
        model_config=load_model_config(),
        feature_columns=["practice_feature"],
        temporal_policy=TemporalWeightingPolicy.current_season_only_with_prior,
    )

    assert captured["training_events"] == ["a"]
    assert captured["weights"] == pytest.approx([0.35, 1.0])
    assert payload["sample_weight_summary"]["training_events"] == 2


def test_replay_artifacts_are_path_distinct_from_artifact_driven_outputs() -> None:
    split = replay_split_id((2023, 2024), 2025)

    assert split == "prospective_replay_train_2023_2024_test_2025"
    assert split != "prospective_train_2023_2024_test_2025"


def test_compare_replay_to_artifact_driven_joins_matching_split_ids(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    pd.DataFrame(
        {
            "prospective_split": ["prospective_train_2023_2024_test_2025"],
            "policy_profile": ["season_aware_frozen"],
            "checkpoint": ["after_fp3"],
            "mae_gap_sec": [0.5],
            "candidate_selection_rate": [0.4],
        }
    ).to_csv(metrics_dir / "prospective_policy_checkpoint_comparison.csv", index=False)
    replay = pd.DataFrame(
        {
            "prospective_split": ["prospective_replay_train_2023_2024_test_2025"],
            "policy_profile": ["season_aware_frozen"],
            "checkpoint": ["after_fp3"],
            "mae_gap_sec": [0.6],
            "candidate_selection_rate": [0.2],
        }
    )

    comparison = compare_replay_to_artifact_driven(metrics_dir, replay)

    assert len(comparison) == 1
    assert comparison["delta_replay_minus_artifact_driven"].iloc[0] == pytest.approx(0.1)
    assert comparison["comparison_interpretation"].iloc[0] == "differs_due_to_retrain_history"


def test_bootstrap_summary_is_deterministic() -> None:
    event = pd.DataFrame(
        {
            "policy_profile": ["season_aware_frozen", "season_aware_frozen"],
            "checkpoint": ["after_fp3", "after_fp3"],
            "fp3_delta_vs_static_baseline_sec": [-0.1, 0.2],
            "fp3_delta_vs_guarded_baseline_sec": [0.0, 0.1],
        }
    )

    assert replay_bootstrap_summary(event) == replay_bootstrap_summary(event)


def test_existing_champion_selection_modes_are_unchanged() -> None:
    assert [mode.value for mode in ChampionSelectionMode] == [
        "static",
        "nested",
        "stabilized_nested",
        "stabilized_nested_guarded",
        "season_aware_nested_guarded",
    ]


def _prediction_source(checkpoint: str, predictions: list[float]) -> pd.DataFrame:
    frame = _history_rows("uniform", "2025/b", 2, predictions)
    frame["checkpoint"] = checkpoint
    frame["predicted_quali_position"] = [1, 2][: len(frame)]
    return frame


def _baseline_source(checkpoint: str, baseline_name: str, predictions: list[float]) -> pd.DataFrame:
    frame = _prediction_source(checkpoint, predictions)
    frame["baseline_name"] = baseline_name
    return frame


def _history_rows(
    temporal_policy: str,
    event_key: str,
    fold_id: int,
    predictions: list[float],
) -> pd.DataFrame:
    season_text, slug = event_key.split("/", maxsplit=1)
    actuals = [0.0, 1.0][: len(predictions)]
    return pd.DataFrame(
        {
            "evaluation_type": [EVALUATION_TYPE] * len(predictions),
            "policy_profile": ["season_aware_frozen"] * len(predictions),
            "source_temporal_weighting_policy": [temporal_policy] * len(predictions),
            "fold_id": [fold_id] * len(predictions),
            "season": [int(season_text)] * len(predictions),
            "event": [slug.upper()] * len(predictions),
            "event_slug": [slug] * len(predictions),
            "checkpoint": ["after_fp3"] * len(predictions),
            "driver": ["AAA", "BBB"][: len(predictions)],
            "driver_key": ["AAA", "BBB"][: len(predictions)],
            "quali_gap_to_pole_sec": actuals,
            "predicted_quali_gap_to_pole_sec": predictions,
            "predicted_quali_position": [1, 2][: len(predictions)],
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
