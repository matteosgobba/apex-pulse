from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from f1_prediction.cli import app
from f1_prediction.config import DataConfig, load_model_config
from f1_prediction.modeling.prospective_policy_evaluation import (
    build_checkpoint_comparison,
    build_event_comparison,
    build_frozen_policy_profiles,
    build_leakage_audit,
    create_prospective_policy_evaluation_report,
    prospective_split_id,
)


def test_frozen_policy_profile_signature_is_deterministic() -> None:
    model_config = load_model_config()
    first = build_frozen_policy_profiles(
        model_config,
        profile_names=("static_baseline", "guarded_baseline", "season_aware_frozen"),
        uncertainty="conformal_predicted_gap_bucket",
    )
    second = build_frozen_policy_profiles(
        model_config,
        profile_names=("static_baseline", "guarded_baseline", "season_aware_frozen"),
        uncertainty="conformal_predicted_gap_bucket",
    )

    assert first["season_aware_frozen"].to_payload() == second["season_aware_frozen"].to_payload()
    assert first["season_aware_frozen"].to_payload()["policy_signature"]


def test_prospective_split_paths_are_distinct_from_walk_forward_artifacts() -> None:
    split_id = prospective_split_id((2023, 2024), 2025)

    assert split_id == "prospective_train_2023_2024_test_2025"
    assert "walk_forward" not in split_id
    assert "champion_static_predictions" not in split_id


def test_leakage_audit_allows_only_prior_held_out_events() -> None:
    profiles = build_frozen_policy_profiles(
        load_model_config(),
        profile_names=("season_aware_frozen",),
        uncertainty="conformal_predicted_gap_bucket",
    )
    selections = pd.DataFrame(
        {
            "policy_profile": ["season_aware_frozen", "season_aware_frozen"],
            "policy_signature": [
                profiles["season_aware_frozen"].to_payload()["policy_signature"],
                profiles["season_aware_frozen"].to_payload()["policy_signature"],
            ],
            "season": [2025, 2025],
            "event": ["A", "B"],
            "event_slug": ["a", "b"],
            "checkpoint": ["after_fp3", "after_fp3"],
            "fold_id": [1, 2],
            "metric_scope_event_keys": [["2025/b"], ["2025/a"]],
        }
    )

    audit = build_leakage_audit(
        selections=selections,
        profiles=profiles,
        train_seasons=(2023, 2024),
        test_season=2025,
    )

    assert audit.loc[0, "leakage_status"] == "invalid"
    assert audit.loc[0, "future_test_season_event_used"]
    assert audit.loc[1, "leakage_status"] == "valid"


def test_event_metrics_and_bootstrap_inputs_are_deterministic() -> None:
    predictions = pd.concat(
        [
            _prediction_rows("static_baseline", 1, 2025, "A", [0.0, 1.0]),
            _prediction_rows("season_aware_frozen", 1, 2025, "A", [0.0, 0.8]),
            _prediction_rows("static_baseline", 2, 2025, "B", [0.0, 1.0]),
            _prediction_rows("season_aware_frozen", 2, 2025, "B", [0.0, 1.2]),
        ],
        ignore_index=True,
    )
    selections = pd.DataFrame(
        {
            "policy_profile": ["season_aware_frozen", "season_aware_frozen"],
            "fold_id": [1, 2],
            "checkpoint": ["after_fp3", "after_fp3"],
            "season_aware_selected": [True, False],
            "candidate_selection_reason": ["season_aware_candidate_selected", "margin_not_met"],
            "current_test_season_prior_event_count": [0, 1],
        }
    )

    checkpoint = build_checkpoint_comparison(predictions)
    event = build_event_comparison(predictions, selections)

    season_aware_fp3 = checkpoint[
        checkpoint["policy_profile"].eq("season_aware_frozen")
        & checkpoint["checkpoint"].eq("after_fp3")
    ].iloc[0]
    assert season_aware_fp3["mae_gap_sec"] == pytest.approx(0.1)
    assert season_aware_fp3["fp3_delta_vs_static_baseline_sec"] == pytest.approx(0.1)
    season_aware_events = event[event["policy_profile"].eq("season_aware_frozen")]
    assert season_aware_events["fp3_delta_vs_static_baseline_sec"].notna().sum() == 2


def test_create_prospective_report_runs_on_synthetic_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.metrics_output_dir.mkdir(parents=True)
    model_config = load_model_config()
    for mode, profile in {
        "static": "static_baseline",
        "stabilized_nested_guarded": "guarded_baseline",
        "season_aware_nested_guarded": "season_aware_frozen",
    }.items():
        predictions = pd.concat(
            [
                _prediction_rows(profile, 1, 2025, "A", [0.0, 1.0]),
                _prediction_rows(profile, 2, 2025, "B", [0.0, 0.9]),
            ],
            ignore_index=True,
        )
        predictions["selection_mode"] = mode
        predictions.to_parquet(
            config.metrics_output_dir / f"champion_{mode}_predictions.parquet",
            index=False,
        )
        _selection_rows(mode).to_parquet(
            config.metrics_output_dir / f"champion_{mode}_selection.parquet",
            index=False,
        )

    summary = create_prospective_policy_evaluation_report(
        config,
        model_config,
        train_seasons=(2023, 2024),
        test_season=2025,
        policy_profiles=("static_baseline", "guarded_baseline", "season_aware_frozen"),
    )

    assert summary.status == "complete"
    assert (config.metrics_output_dir / "prospective_policy_summary.json").is_file()
    assert (
        config.metrics_output_dir / "prospective_train_2023_2024_test_2025_predictions.parquet"
    ).is_file()
    leakage = pd.read_csv(config.metrics_output_dir / "prospective_policy_leakage_audit.csv")
    assert leakage["leakage_status"].eq("valid").all()


def test_prospective_policy_cli_registration(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    model_config = load_model_config()
    captured: dict[str, object] = {}
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)
    monkeypatch.setattr(
        "f1_prediction.cli.load_model_config",
        lambda config_path=None, project_root=None: model_config,
    )

    def fake_report(*args, **kwargs):
        captured.update(kwargs)
        from f1_prediction.modeling.prospective_policy_evaluation import (
            ProspectivePolicyEvaluationSummary,
        )

        return ProspectivePolicyEvaluationSummary(
            status="complete",
            summary_path=tmp_path / "summary.json",
            table_paths=(),
            figure_paths=(),
            missing_inputs=(),
            generation_issues=(),
        )

    monkeypatch.setattr(
        "f1_prediction.cli.run_prospective_policy_evaluation_report",
        fake_report,
    )

    result = CliRunner().invoke(
        app,
        [
            "prospective-policy-evaluation",
            "--train-seasons",
            "2023",
            "--test-season",
            "2024",
            "--policy-profiles",
            "static_baseline",
            "--uncertainty",
            "conformal_predicted_gap_bucket",
        ],
    )

    assert result.exit_code == 0
    assert captured["train_seasons"] == (2023,)
    assert captured["test_season"] == 2024
    assert captured["policy_profiles"] == ("static_baseline",)


def _prediction_rows(
    profile: str,
    fold_id: int,
    season: int,
    event: str,
    predictions: list[float],
) -> pd.DataFrame:
    actuals = [0.0, 1.0][: len(predictions)]
    return pd.DataFrame(
        {
            "strategy": ["walk_forward"] * len(predictions),
            "fold_id": [fold_id] * len(predictions),
            "season": [season] * len(predictions),
            "event": [event] * len(predictions),
            "event_slug": [event.lower()] * len(predictions),
            "checkpoint": ["after_fp3"] * len(predictions),
            "driver": ["AAA", "BBB"][: len(predictions)],
            "team": ["Team"] * len(predictions),
            "quali_position": [1, 2][: len(predictions)],
            "quali_gap_to_pole_sec": actuals,
            "predicted_quali_gap_to_pole_sec": predictions,
            "predicted_quali_position": [1, 2][: len(predictions)],
            "policy_profile": [profile] * len(predictions),
            "prospective_split": ["prospective_train_2023_2024_test_2025"] * len(predictions),
            "prospective_event_id": [f"{season}/{event.lower()}"] * len(predictions),
            "train_seasons": ["2023,2024"] * len(predictions),
            "test_season": [season] * len(predictions),
            "prediction_interval_low_sec": [prediction - 0.2 for prediction in predictions],
            "prediction_interval_high_sec": [prediction + 0.2 for prediction in predictions],
            "interval_contains_actual": [True] * len(predictions),
        }
    )


def _selection_rows(mode: str) -> pd.DataFrame:
    selected = mode == "season_aware_nested_guarded"
    return pd.DataFrame(
        {
            "selection_mode": [mode, mode],
            "fold_id": [1, 2],
            "season": [2025, 2025],
            "event": ["A", "B"],
            "event_slug": ["a", "b"],
            "checkpoint": ["after_fp3", "after_fp3"],
            "selected_family": ["ablation", "ablation"],
            "selected_model_name": ["random_forest", "random_forest"],
            "selected_feature_group": ["base_plus_relative", "base_plus_relative"],
            "season_aware_selected": [False, selected],
            "season_aware_selection_reason": ["cold_start", "season_aware_candidate_selected"],
            "current_season_prior_event_count": [0, 1],
            "metric_scope_event_keys": [[], ["2025/a"]],
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
