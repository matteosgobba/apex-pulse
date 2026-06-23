import json
from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.config import DataConfig, PolicySimulationConfig
from f1_prediction.modeling.policy_simulation import (
    build_guardrail_event_level_table,
    build_guardrail_simulation_table,
    build_regime_conformal_simulation_table,
    create_policy_simulation_report,
    gap_bucket,
    generate_policy_simulation_figures,
    simulate_guardrail_policies,
    simulate_regime_conformal_intervals,
)


def test_fp3_static_lock_uses_static_for_fp3_and_stabilized_for_fp1_fp2() -> None:
    rows = simulate_guardrail_policies(
        _static_predictions(),
        _stabilized_predictions(),
        _selection(),
    )

    policy = rows[rows["policy_name"].eq("fp3_static_lock")]
    fp1 = policy[policy["checkpoint"].eq("after_fp1")].iloc[0]
    fp3 = policy[policy["checkpoint"].eq("after_fp3")].iloc[0]

    assert fp1["simulated_predicted"] == pytest.approx(fp1["stabilized_predicted"])
    assert fp3["simulated_predicted"] == pytest.approx(fp3["static_predicted"])
    assert bool(fp3["replaced_with_static"])


def test_fp3_no_baseline_switch_replaces_stabilized_fp3_baseline_selection() -> None:
    rows = simulate_guardrail_policies(
        _static_predictions(),
        _stabilized_predictions(),
        _selection(),
    )

    policy = rows[
        rows["policy_name"].eq("fp3_no_baseline_switch") & rows["checkpoint"].eq("after_fp3")
    ].iloc[0]

    assert policy["simulated_predicted"] == pytest.approx(policy["static_predicted"])
    assert bool(policy["replaced_with_static"])
    assert policy["replacement_reason"] == "fp3_baseline_switch"


def test_oracle_guardrail_is_labeled_as_oracle() -> None:
    table = build_guardrail_simulation_table(
        simulate_guardrail_policies(_static_predictions(), _stabilized_predictions(), _selection())
    )

    row = table[table["policy_name"].eq("fp3_harmful_event_guardrail_oracle")].iloc[0]

    assert row["policy_type"] == "oracle"


def test_guardrail_metrics_compute_mae_delta_and_replaced_rows() -> None:
    table = build_guardrail_simulation_table(
        simulate_guardrail_policies(_static_predictions(), _stabilized_predictions(), _selection())
    )

    static = table[
        table["policy_name"].eq("current_static") & table["checkpoint"].eq("after_fp3")
    ].iloc[0]
    stabilized = table[
        table["policy_name"].eq("current_stabilized_nested") & table["checkpoint"].eq("after_fp3")
    ].iloc[0]
    lock = table[
        table["policy_name"].eq("fp3_static_lock") & table["checkpoint"].eq("after_fp3")
    ].iloc[0]

    assert static["mae_gap_sec"] == pytest.approx(0.0)
    assert stabilized["mae_gap_sec"] == pytest.approx(1.0)
    assert lock["delta_vs_current_stabilized_nested_mae_sec"] == pytest.approx(-1.0)
    assert lock["fp3_rows_replaced"] == 1


def test_event_level_guardrail_detects_silverstone_like_harmful_switch() -> None:
    events = build_guardrail_event_level_table(
        simulate_guardrail_policies(_static_predictions(), _stabilized_predictions(), _selection())
    )

    silverstone = events[
        events["policy_name"].eq("fp3_no_baseline_switch") & events["event"].eq("Silverstone")
    ].iloc[0]

    assert bool(silverstone["replaced_with_static"])
    assert silverstone["simulated_policy_mae_gap_sec"] == pytest.approx(0.0)
    assert silverstone["delta_vs_stabilized_sec"] == pytest.approx(-1.0)


def test_predicted_gap_bucket_threshold_boundaries() -> None:
    assert gap_bucket(0.5) == "pole_contender"
    assert gap_bucket(1.5) == "close_midfield"
    assert gap_bucket(3.0) == "midfield"
    assert gap_bucket(3.1) == "backmarker_or_outlier"


def test_regime_conformal_uses_prior_folds_only() -> None:
    rows = simulate_regime_conformal_intervals(_conformal_predictions(), _simulation_config(2))
    fold_2 = rows[
        rows["strategy_name"].eq("global_conformal")
        & rows["fold_id"].eq(2)
        & rows["driver"].eq("NOR")
    ].iloc[0]

    assert fold_2["residual_count"] == 2
    assert fold_2["residual_quantile_sec"] == pytest.approx(0.8)


def test_regime_conformal_falls_back_to_coarser_groups() -> None:
    config = _simulation_config(3)
    rows = simulate_regime_conformal_intervals(_conformal_predictions(), config)
    fold_3 = rows[
        rows["strategy_name"].eq("checkpoint_predicted_gap_bucket")
        & rows["fold_id"].eq(3)
        & rows["driver"].eq("NOR")
    ].iloc[0]

    assert fold_3["calibration_level"] == "checkpoint_method"
    assert fold_3["residual_count"] == 4


def test_interval_coverage_and_width_metrics_are_computed() -> None:
    rows = simulate_regime_conformal_intervals(_conformal_predictions(), _simulation_config(2))
    table = build_regime_conformal_simulation_table(rows)
    fp3 = table[
        table["strategy_name"].eq("global_conformal")
        & table["summary_scope"].eq("checkpoint")
        & table["checkpoint"].eq("after_fp3")
    ].iloc[0]

    assert fp3["rows"] == 6
    assert fp3["interval_availability_rate"] == pytest.approx(4 / 6)
    assert fp3["mean_interval_width_sec"] > 0
    assert 0 <= fp3["coverage"] <= 1


def test_summary_json_records_missing_inputs_gracefully(tmp_path: Path) -> None:
    config = _config(tmp_path)

    summary = create_policy_simulation_report(config, PolicySimulationConfig())
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))

    assert summary.status == "no_inputs"
    assert "champion_static_predictions.parquet" in payload["missing_inputs"]
    assert (config.metrics_output_dir / "fp3_guardrail_simulation_table.csv").is_file()


def test_policy_simulation_summarizes_guarded_artifacts_when_present(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.metrics_output_dir.mkdir(parents=True, exist_ok=True)
    _static_predictions().to_parquet(
        config.metrics_output_dir / "champion_static_predictions.parquet"
    )
    _stabilized_predictions().to_parquet(
        config.metrics_output_dir / "champion_stabilized_nested_predictions.parquet"
    )
    _static_predictions().assign(selection_mode="stabilized_nested_guarded").to_parquet(
        config.metrics_output_dir / "champion_stabilized_nested_guarded_predictions.parquet"
    )
    _selection().to_parquet(config.metrics_output_dir / "champion_static_selection.parquet")
    _selection().to_parquet(
        config.metrics_output_dir / "champion_stabilized_nested_selection.parquet"
    )
    guarded_selection = _selection().copy()
    guarded_selection["guardrail_applied"] = [True]
    guarded_selection.to_parquet(
        config.metrics_output_dir / "champion_stabilized_nested_guarded_selection.parquet"
    )

    summary = create_policy_simulation_report(config, PolicySimulationConfig())
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    guarded = payload["guarded_mode_artifact_summary"]

    assert summary.status == "complete"
    assert guarded["available"] is True
    assert guarded["fp3_mae_gap_sec"] == pytest.approx(0.0)
    assert guarded["guardrail_applied_count"] == 1
    assert guarded["matches_simulated_fp3_no_baseline_switch_mae"] is True


def test_figure_generation_does_not_crash_on_minimal_valid_inputs(tmp_path: Path) -> None:
    guardrail_rows = simulate_guardrail_policies(
        _static_predictions(),
        _stabilized_predictions(),
        _selection(),
    )
    guardrail_summary = build_guardrail_simulation_table(guardrail_rows)
    guardrail_events = build_guardrail_event_level_table(guardrail_rows)
    conformal_rows = simulate_regime_conformal_intervals(
        _conformal_predictions(),
        _simulation_config(2),
    )
    conformal_summary = build_regime_conformal_simulation_table(conformal_rows)

    paths, issues = generate_policy_simulation_figures(
        figures_dir=tmp_path,
        guardrail_summary=guardrail_summary,
        guardrail_events=guardrail_events,
        conformal_summary=conformal_summary,
    )

    assert isinstance(issues, list)
    assert all(path.is_file() for path in paths)


def _static_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": [1, 1, 1],
            "season": [2024, 2024, 2024],
            "event": ["Silverstone", "Silverstone", "Silverstone"],
            "event_slug": ["silverstone", "silverstone", "silverstone"],
            "checkpoint": ["after_fp1", "after_fp2", "after_fp3"],
            "driver": ["NOR", "NOR", "NOR"],
            "team": ["McLaren", "McLaren", "McLaren"],
            "quali_gap_to_pole_sec": [1.0, 1.0, 1.0],
            "predicted_quali_gap_to_pole_sec": [1.5, 1.4, 1.0],
            "selected_family": ["robust_baseline", "robust_baseline", "ablation"],
            "selected_model_name": [
                "robust_best_push_lap",
                "robust_theoretical_best_lap",
                "random_forest",
            ],
            "selected_feature_group": [pd.NA, pd.NA, "base_plus_relative"],
        }
    )


def _stabilized_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": [1, 1, 1],
            "season": [2024, 2024, 2024],
            "event": ["Silverstone", "Silverstone", "Silverstone"],
            "event_slug": ["silverstone", "silverstone", "silverstone"],
            "checkpoint": ["after_fp1", "after_fp2", "after_fp3"],
            "driver": ["NOR", "NOR", "NOR"],
            "team": ["McLaren", "McLaren", "McLaren"],
            "quali_gap_to_pole_sec": [1.0, 1.0, 1.0],
            "predicted_quali_gap_to_pole_sec": [1.2, 1.1, 2.0],
            "selected_family": ["robust_baseline", "robust_baseline", "baseline"],
            "selected_model_name": [
                "robust_best_push_lap",
                "robust_theoretical_best_lap",
                "best_valid_lap",
            ],
            "selected_feature_group": [pd.NA, pd.NA, pd.NA],
        }
    )


def _selection() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": [1],
            "checkpoint": ["after_fp3"],
            "fallback_used": [False],
            "prior_predictions_used": [100],
            "min_prior_predictions": [20],
            "selected_metric_value": [0.4],
            "default_metric_value": [0.6],
            "improvement_margin_sec": [0.05],
        }
    )


def _conformal_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": [1, 1, 2, 2, 3, 3],
            "season": [2024] * 6,
            "event": ["A", "A", "B", "B", "C", "C"],
            "event_slug": ["a", "a", "b", "b", "c", "c"],
            "checkpoint": ["after_fp3"] * 6,
            "driver": ["NOR", "LEC", "NOR", "LEC", "NOR", "LEC"],
            "team": ["McLaren", "Ferrari"] * 3,
            "selected_family": ["ablation"] * 6,
            "selected_model_name": ["random_forest"] * 6,
            "selected_feature_group": ["base_plus_relative"] * 6,
            "quali_gap_to_pole_sec": [1.0, 1.2, 10.0, 10.2, 1.1, 1.3],
            "predicted_quali_gap_to_pole_sec": [0.2, 1.1, 4.0, 1.1, 1.0, 1.2],
        }
    )


def _simulation_config(min_residual_count: int) -> PolicySimulationConfig:
    config = PolicySimulationConfig()
    return PolicySimulationConfig(
        conformal=type(config.conformal)(
            confidence_level=0.9,
            min_residual_count=min_residual_count,
            fallback_order=config.conformal.fallback_order,
        )
    )


def _config(project_root: Path) -> DataConfig:
    return DataConfig(
        project_root=project_root,
        fastf1_cache_dir=project_root / "cache",
        lap_output_dir=project_root / "laps",
        session_metadata_output_dir=project_root / "metadata",
        clean_lap_output_dir=project_root / "clean_laps",
        session_features_output_dir=project_root / "session_features",
        modeling_output_dir=project_root / "modeling",
        metrics_output_dir=project_root / "reports" / "metrics",
    )
