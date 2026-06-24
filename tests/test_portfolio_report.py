import json
from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.config import DataConfig
from f1_prediction.modeling.portfolio_report import (
    build_champion_interval_summary_table,
    build_champion_selection_summary_table,
    build_champion_summary_table,
    build_model_card,
    create_portfolio_report,
    generate_portfolio_figures,
)


def test_portfolio_summary_records_missing_inputs(tmp_path: Path) -> None:
    config = _config(tmp_path)

    summary = create_portfolio_report(config)
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))

    assert summary.summary_path.is_file()
    assert "champion_static_metrics.json" in payload["missing_artifacts"]
    assert payload["champion_modes_available"] == []
    assert (config.metrics_output_dir / "champion_summary_table.csv").is_file()


def test_champion_summary_table_from_synthetic_metrics_and_predictions() -> None:
    table = build_champion_summary_table(
        {"static": _metrics_payload("static", 0.8, 0.1)},
        {"static": _predictions("static")},
    )

    row = table.iloc[0]
    assert row["selection_mode"] == "static"
    assert row["checkpoint"] == "after_fp1"
    assert row["rows"] == 2
    assert row["mae_gap_sec"] == pytest.approx(0.8)
    assert row["champion_vs_best_baseline_delta_mae"] == pytest.approx(0.1)
    assert row["best_baseline_mae_gap_sec"] == pytest.approx(0.7)


def test_interval_summary_computes_availability_coverage_and_width() -> None:
    predictions = _predictions("stabilized_nested")
    predictions["uncertainty_method"] = "conformal"
    predictions["prediction_interval_low_sec"] = [-0.1, pd.NA]
    predictions["prediction_interval_high_sec"] = [0.3, pd.NA]
    predictions["interval_contains_actual"] = [True, pd.NA]
    predictions["residual_quantile_sec"] = [0.2, pd.NA]
    predictions["residual_std_sec"] = [0.15, pd.NA]

    table = build_champion_interval_summary_table({"stabilized_nested": predictions})

    row = table.iloc[0]
    assert row["interval_availability_rate"] == pytest.approx(0.5)
    assert row["interval_coverage"] == pytest.approx(1.0)
    assert row["mean_interval_width_sec"] == pytest.approx(0.4)
    assert row["mean_residual_quantile_sec"] == pytest.approx(0.2)


def test_selection_summary_computes_shares_and_fallback_rates() -> None:
    selection = pd.DataFrame(
        {
            "selection_mode": ["stabilized_nested"] * 4,
            "fold_id": [1, 2, 3, 4],
            "checkpoint": ["after_fp3"] * 4,
            "selected_family": ["ablation", "ablation", "ablation", "baseline"],
            "selected_model_name": [
                "random_forest",
                "random_forest",
                "random_forest",
                "best_valid_lap",
            ],
            "selected_feature_group": [
                "base_plus_relative",
                "base_plus_relative",
                "base_plus_relative",
                pd.NA,
            ],
            "fallback_used": [True, False, False, True],
            "fallback_reason": ["insufficient_history", pd.NA, pd.NA, "insufficient_history"],
        }
    )

    table = build_champion_selection_summary_table({"stabilized_nested": selection})

    forest = table[table["selected_model_name"].eq("random_forest")].iloc[0]
    baseline = table[table["selected_model_name"].eq("best_valid_lap")].iloc[0]
    assert forest["selection_share"] == pytest.approx(0.75)
    assert forest["fallback_rate"] == pytest.approx(1 / 3)
    assert forest["main_fallback_reason"] == "insufficient_history"
    assert baseline["selection_share"] == pytest.approx(0.25)


def test_model_card_generation_includes_required_sections() -> None:
    champion_summary = build_champion_summary_table(
        {"static": _metrics_payload("static", 0.8, 0.1)},
        {"static": _predictions("static")},
    )
    interval_summary = build_champion_interval_summary_table({"static": _predictions("static")})
    text = build_model_card(
        {
            "dataset_summary_if_available": {"rows": 12, "events": 2, "drivers": 4},
            "best_champion_mode_overall": {"selection_mode": "static"},
            "limitations": ["FastF1 public data only."],
            "recommended_next_milestone": "Milestone 17.",
        },
        champion_summary,
        interval_summary,
    )

    for section in (
        "Project overview",
        "Prediction task",
        "Data sources",
        "Current dataset",
        "Evaluation protocol",
        "Temporal weighting",
        "Season-aware validation",
        "Baselines",
        "Model families",
        "Champion policy",
        "Results summary",
        "Uncertainty estimates",
        "Key limitations",
        "Recommended next steps",
    ):
        assert f"## {section}" in text


def test_figure_generation_does_not_crash_on_minimal_inputs(tmp_path: Path) -> None:
    champion_summary = build_champion_summary_table(
        {"static": _metrics_payload("static", 0.8, 0.1)},
        {"static": _predictions("static")},
    )
    interval_summary = build_champion_interval_summary_table({"static": _predictions("static")})
    selection_summary = build_champion_selection_summary_table({"static": _selection("static")})

    paths, issues = generate_portfolio_figures(
        figures_dir=tmp_path,
        champion_summary=champion_summary,
        interval_summary=interval_summary,
        selection_summary=selection_summary,
        worst_events=pd.DataFrame(),
    )

    assert isinstance(issues, list)
    assert all(path.is_file() for path in paths)


def test_create_portfolio_report_writes_requested_outputs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_minimal_artifacts(config.metrics_output_dir)

    summary = create_portfolio_report(config)
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))

    assert payload["champion_modes_available"] == ["static", "stabilized_nested"]
    assert payload["key_results"]["static_champion_mae_by_checkpoint"]["after_fp1"] == 0.8
    assert (config.metrics_output_dir / "champion_interval_summary_table.csv").is_file()
    assert (config.metrics_output_dir / "worst_event_diagnostics_table.csv").is_file()
    assert summary.model_card_path.is_file()


def test_portfolio_report_includes_guarded_mode_when_artifacts_exist(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_minimal_artifacts(
        config.metrics_output_dir,
        include_guarded=True,
        include_season_aware=True,
    )

    summary = create_portfolio_report(config)
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    champion_summary = pd.read_csv(config.metrics_output_dir / "champion_summary_table.csv")

    assert summary.summary_path.is_file()
    assert "stabilized_nested_guarded" in payload["champion_modes_available"]
    assert "season_aware_nested_guarded" in payload["champion_modes_available"]
    assert (
        payload["key_results"]["stabilized_nested_guarded_champion_mae_by_checkpoint"]["after_fp1"]
        == 0.85
    )
    assert "stabilized_nested_guarded" in set(champion_summary["selection_mode"])
    assert "season_aware_nested_guarded" in set(champion_summary["selection_mode"])


def test_portfolio_report_includes_season_aware_champion_summary_when_available(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_minimal_artifacts(config.metrics_output_dir)
    (config.metrics_output_dir / "season_aware_champion_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "fp3_summary": {"delta_vs_static_sec": -0.03},
                "bootstrap_ci": {"ci_low": -0.08, "ci_high": 0.01},
                "promotion_recommendation": "season_aware_candidate_experimental",
                "main_findings": ["Season-aware champion remains experimental."],
            }
        ),
        encoding="utf-8",
    )

    summary = create_portfolio_report(config)
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))

    assert payload["season_aware_champion_if_available"]["season_aware_champion_available"] is True
    assert (
        payload["season_aware_champion_if_available"]["promotion_recommendation"]
        == "season_aware_candidate_experimental"
    )


def test_portfolio_report_includes_temporal_weighting_when_available(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_minimal_artifacts(config.metrics_output_dir)
    (config.metrics_output_dir / "temporal_weighting_summary.json").write_text(
        json.dumps(
            {
                "temporal_weighting_policies_available": {
                    "tabular": ["uniform", "season_priority"]
                },
                "best_temporal_weighting_policy_by_checkpoint": {
                    "after_fp3": {"temporal_weighting_policy": "season_priority"}
                },
                "temporal_weighting_vs_uniform_delta_by_checkpoint": {
                    "after_fp3": {"season_priority": -0.05}
                },
                "main_findings": ["Season-aware weighting is opt-in."],
            }
        ),
        encoding="utf-8",
    )

    summary = create_portfolio_report(config)
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    model_card = summary.model_card_path.read_text(encoding="utf-8")

    assert payload["temporal_weighting_if_available"]["temporal_weighting_policies_available"][
        "tabular"
    ] == ["uniform", "season_priority"]
    assert any("Season-aware weighting" in item for item in payload["main_takeaways"])
    assert "Temporal weighting" in model_card


def test_portfolio_report_includes_season_aware_validation_when_available(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_minimal_artifacts(config.metrics_output_dir)
    (config.metrics_output_dir / "season_aware_validation_summary.json").write_text(
        json.dumps(
            {
                "season_aware_validation_available": True,
                "season_aware_fp3_candidate_summary": {
                    "candidate_mae_gap_sec": 0.95,
                    "static_mae_gap_sec": 1.05,
                },
                "season_aware_best_fixed_candidate": {
                    "candidate_family": "ablation",
                    "candidate_model_name": "random_forest",
                },
                "season_aware_promotion_recommendation": "insufficient_evidence",
                "bootstrap_robustness": {
                    "mean_delta": -0.1,
                    "ci_low": -0.2,
                    "ci_high": 0.05,
                },
                "main_findings": ["Season-aware validation is retrospective."],
            }
        ),
        encoding="utf-8",
    )

    summary = create_portfolio_report(config)
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    model_card = summary.model_card_path.read_text(encoding="utf-8")

    assert (
        payload["season_aware_validation_if_available"]["season_aware_validation_available"] is True
    )
    assert any("Season-aware candidate validation" in item for item in payload["main_takeaways"])
    assert "Season-aware validation" in model_card


def test_portfolio_report_includes_season_aware_candidate_audit_when_available(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_minimal_artifacts(config.metrics_output_dir)
    (config.metrics_output_dir / "season_aware_candidate_audit_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "candidate_availability": {"weighted_candidate_rows": 20},
                "artifact_alignment_summary": {"unmatched_rows": 0},
                "live_gate_summary": {
                    "folds_evaluated": 4,
                    "weighted_candidate_selection_rate": 0.0,
                    "gate_failure_reasons": {"insufficient_candidate_history": 4},
                },
                "live_audit_metric_consistency_rate": 1.0,
                "live_audit_selection_consistency_rate": 1.0,
                "comparator_scope_description": "aligned_prior_rows_v1",
                "sensitivity_analysis_summary": {"all_results_retrospective_simulation": True},
                "recommendation": "retain_gates_and_collect_more_history",
                "main_findings": ["Current live gates selected zero weighted candidates."],
            }
        ),
        encoding="utf-8",
    )

    summary = create_portfolio_report(config)
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    model_card = summary.model_card_path.read_text(encoding="utf-8")

    audit = payload["season_aware_candidate_audit_if_available"]
    assert audit["live_gate_summary"]["weighted_candidate_selection_rate"] == 0.0
    assert audit["live_audit_metric_consistency_rate"] == 1.0
    assert audit["comparator_scope_description"] == "aligned_prior_rows_v1"
    assert audit["recommendation"] == "retain_gates_and_collect_more_history"
    assert any("Season-aware candidate audit" in item for item in payload["main_takeaways"])
    assert "Season-aware candidate audit" in model_card


def _metrics_payload(mode: str, mae: float, delta: float) -> dict[str, object]:
    return {
        "status": "complete",
        "selection_mode": mode,
        "uncertainty_method": "conformal" if mode == "stabilized_nested" else "residual_std",
        "metrics_by_checkpoint": {
            "after_fp1": {
                "mae_gap_sec": mae,
                "rmse_gap_sec": mae + 0.1,
                "median_abs_error_gap_sec": mae - 0.1,
                "mean_abs_position_error": 3.0,
                "interval_availability_rate": 1.0,
                "interval_coverage": 0.9,
                "mean_interval_width_sec": 1.2,
                "median_interval_width_sec": 1.0,
            }
        },
        "best_baseline_by_checkpoint": {
            "after_fp1": {"baseline_name": "robust", "mae_gap_sec": mae - delta}
        },
        "best_single_family_by_checkpoint": {
            "after_fp1": {
                "family": "robust_baseline",
                "model_name": "robust_best_push_lap",
                "feature_group": None,
            }
        },
        "champion_vs_best_baseline_delta_mae": {"after_fp1": delta},
        "champion_vs_best_single_family_delta_mae": {"after_fp1": 0.0},
    }


def _predictions(mode: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "selection_mode": [mode, mode],
            "checkpoint": ["after_fp1", "after_fp1"],
            "prediction_interval_low_sec": [-0.2, 0.1],
            "prediction_interval_high_sec": [0.2, 0.5],
            "interval_contains_actual": [True, False],
            "uncertainty_method": ["residual_std", "residual_std"],
            "residual_quantile_sec": [pd.NA, pd.NA],
            "residual_std_sec": [0.2, 0.2],
        }
    )


def _selection(mode: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "selection_mode": [mode],
            "fold_id": [1],
            "checkpoint": ["after_fp1"],
            "selected_family": ["robust_baseline"],
            "selected_model_name": ["robust_best_push_lap"],
            "selected_feature_group": [pd.NA],
            "fallback_used": [False],
            "fallback_reason": [pd.NA],
        }
    )


def _write_minimal_artifacts(
    metrics_dir: Path,
    *,
    include_guarded: bool = False,
    include_season_aware: bool = False,
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    modes = [("static", 0.8), ("stabilized_nested", 0.9)]
    if include_guarded:
        modes.append(("stabilized_nested_guarded", 0.85))
    if include_season_aware:
        modes.append(("season_aware_nested_guarded", 0.82))
    for mode, mae in modes:
        (metrics_dir / f"champion_{mode}_metrics.json").write_text(
            json.dumps(_metrics_payload(mode, mae, 0.1)),
            encoding="utf-8",
        )
        _predictions(mode).to_parquet(metrics_dir / f"champion_{mode}_predictions.parquet")
        _selection(mode).to_parquet(metrics_dir / f"champion_{mode}_selection.parquet")
    (metrics_dir / "backtest_report.json").write_text(
        json.dumps(
            {
                "dataset_rows": 12,
                "n_events": 2,
                "n_drivers": 4,
                "best_champion_selection_mode_overall": {
                    "selection_mode": "static",
                    "mean_mae_gap_sec": 0.8,
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "season": [2024],
            "event": ["Monza"],
            "checkpoint": ["after_fp1"],
            "model_name": ["robust"],
            "mae_gap_sec": [1.2],
            "mean_abs_position_error": [4.0],
            "n_rows": [2],
        }
    ).to_parquet(metrics_dir / "event_error_summary.parquet")
    pd.DataFrame(
        {
            "driver": ["NOR"],
            "checkpoint": ["after_fp1"],
            "model_name": ["robust"],
            "mae_gap_sec": [1.1],
            "mean_abs_position_error": [3.0],
            "n_rows": [2],
        }
    ).to_parquet(metrics_dir / "driver_error_summary.parquet")


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
