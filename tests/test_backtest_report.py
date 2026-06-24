import pytest

from f1_prediction.modeling.backtest_report import build_backtest_report_payload


def test_backtest_report_with_trained_tabular_metrics() -> None:
    report = build_backtest_report_payload(
        _quality(),
        _baseline_metrics(),
        _trained_metrics(),
    )

    assert report["training_status"] == "trained"
    assert report["tabular_models_available"] == ["ridge", "random_forest"]
    assert report["best_tabular_model_by_checkpoint"]["after_fp1"]["model_name"] == "ridge"
    assert report["model_vs_baseline_delta_mae_by_checkpoint"]["after_fp1"] == pytest.approx(-0.1)
    assert report["model_vs_baseline_delta_position_error_by_checkpoint"]["after_fp1"] == -1.0


def test_backtest_report_with_skipped_training() -> None:
    report = build_backtest_report_payload(
        _quality(),
        _baseline_metrics(),
        {"status": "skipped", "reason": "too small"},
    )

    assert report["training_status"] == "skipped"
    assert report["tabular_models_available"] == []
    assert report["best_tabular_model_by_checkpoint"] == {}
    assert report["model_vs_baseline_delta_mae_by_checkpoint"]["after_fp1"] is None


def test_backtest_report_without_tabular_metrics_keeps_baselines() -> None:
    report = build_backtest_report_payload(_quality(), _baseline_metrics(), None)

    assert report["training_status"] == "unavailable"
    assert report["best_baseline_by_checkpoint"]["after_fp1"]["baseline_name"] == "push"
    assert report["best_tabular_model_by_checkpoint"] == {}


def test_backtest_delta_uses_holdout_comparable_baseline() -> None:
    tabular = _trained_metrics()
    tabular["best_baseline_by_checkpoint"]["after_fp1"]["mae_gap_sec"] = 0.45

    report = build_backtest_report_payload(_quality(), _baseline_metrics(), tabular)

    assert report["model_vs_baseline_delta_mae_by_checkpoint"]["after_fp1"] == pytest.approx(-0.15)


def test_backtest_report_prefers_walk_forward_over_repeated_holdout() -> None:
    repeated = _multi_fold_metrics("repeated_event_holdout", 7)
    walk_forward = _multi_fold_metrics("walk_forward", 4)

    report = build_backtest_report_payload(
        _quality(),
        _baseline_metrics(),
        _trained_metrics(),
        repeated_metrics=repeated,
        walk_forward_metrics=walk_forward,
    )

    assert report["available_backtests"] == [
        "walk_forward",
        "repeated_event_holdout",
        "single_event_holdout",
    ]
    assert report["preferred_backtest_strategy"] == "walk_forward"
    assert report["n_folds_successful"] == 4
    assert report["best_model_by_checkpoint"]["after_fp1"]["model_name"] == "ridge"


def test_backtest_report_includes_ablation_summary() -> None:
    report = build_backtest_report_payload(
        _quality(),
        _baseline_metrics(),
        _trained_metrics(),
        walk_forward_metrics=_multi_fold_metrics("walk_forward", 4),
        ablation_metrics={
            "status": "complete",
            "feature_groups": ["base_lap_features", "all_features"],
            "best_overall_by_checkpoint": {
                "after_fp1": {
                    "feature_group": "all_features",
                    "model_name": "ridge",
                    "mae_gap_sec": 0.25,
                    "mean_abs_position_error": 1.8,
                }
            },
            "best_baseline_by_checkpoint": {
                "after_fp1": {"baseline_name": "push", "mae_gap_sec": 0.4}
            },
        },
    )

    assert report["available_ablation_results"] == ["base_lap_features", "all_features"]
    assert report["preferred_feature_group_by_checkpoint"]["after_fp1"] == "all_features"
    assert report["best_ablation_delta_vs_baseline_by_checkpoint"]["after_fp1"] == pytest.approx(
        -0.15
    )


def test_backtest_report_includes_boosted_comparisons() -> None:
    report = build_backtest_report_payload(
        _quality(),
        _baseline_metrics(),
        _trained_metrics(),
        walk_forward_metrics=_multi_fold_metrics("walk_forward", 4),
        ablation_metrics={
            "status": "complete",
            "feature_groups": ["base_lap_features"],
            "best_overall_by_checkpoint": {
                "after_fp1": {"model_name": "random_forest", "mae_gap_sec": 0.28}
            },
        },
        boosted_metrics={
            "status": "complete",
            "models": ["hist_gradient_boosting"],
            "best_model_by_checkpoint": {
                "after_fp1": {
                    "model_name": "hist_gradient_boosting",
                    "mae_gap_sec": 0.22,
                }
            },
            "best_baseline_by_checkpoint": {
                "after_fp1": {"baseline_name": "push", "mae_gap_sec": 0.4}
            },
        },
    )

    assert report["boosted_models_available"] == ["hist_gradient_boosting"]
    assert report["boosted_vs_best_baseline_delta_mae_by_checkpoint"]["after_fp1"] == pytest.approx(
        -0.18
    )
    assert report["boosted_vs_best_ablation_delta_mae_by_checkpoint"]["after_fp1"] == pytest.approx(
        -0.06
    )
    assert report["preferred_model_family_by_checkpoint"]["after_fp1"] == "boosted"


def test_backtest_report_includes_champion_fields() -> None:
    report = build_backtest_report_payload(
        _quality(),
        _baseline_metrics(),
        _trained_metrics(),
        champion_metrics={
            "status": "complete",
            "selection_mode": "nested",
            "metrics_by_checkpoint": {
                "after_fp1": {
                    "mae_gap_sec": 0.2,
                    "mean_abs_position_error": 1.5,
                }
            },
            "champion_vs_best_baseline_delta_mae": {"after_fp1": -0.2},
            "champion_vs_best_single_family_delta_mae": {"after_fp1": -0.05},
        },
    )

    assert report["champion_available"] is True
    assert report["champion_selection_mode"] == "nested"
    assert report["champion_vs_best_baseline_delta_mae"]["after_fp1"] == -0.2
    assert report["preferred_final_policy_by_checkpoint"]["after_fp1"]["family"] == "champion"


def test_backtest_report_includes_champion_mode_and_interval_diagnostics() -> None:
    report = build_backtest_report_payload(
        _quality(),
        _baseline_metrics(),
        _trained_metrics(),
        champion_metrics=_champion_metrics("stabilized_nested", 0.35),
        champion_mode_metrics={
            "static": _champion_metrics("static", 0.4),
            "nested": _champion_metrics("nested", 0.5),
            "stabilized_nested": _champion_metrics("stabilized_nested", 0.35),
            "stabilized_nested_guarded": _champion_metrics(
                "stabilized_nested_guarded",
                0.34,
            ),
            "season_aware_nested_guarded": _champion_metrics(
                "season_aware_nested_guarded",
                0.33,
            ),
        },
    )

    assert report["champion_selection_modes_available"] == [
        "nested",
        "season_aware_nested_guarded",
        "stabilized_nested",
        "stabilized_nested_guarded",
        "static",
    ]
    assert (
        report["best_champion_selection_mode_by_checkpoint"]["after_fp1"]["selection_mode"]
        == "season_aware_nested_guarded"
    )
    assert (
        report["best_champion_selection_mode_overall"]["selection_mode"]
        == "season_aware_nested_guarded"
    )
    assert report["champion_interval_coverage_by_checkpoint"]["after_fp1"] == pytest.approx(0.9)
    assert report["champion_interval_width_by_checkpoint"]["after_fp1"][
        "mean_interval_width_sec"
    ] == pytest.approx(1.2)
    assert report["champion_interval_metrics_by_predicted_gap_bucket"]["after_fp1"][
        "pole_contender"
    ]["coverage"] == pytest.approx(0.95)


def test_backtest_report_includes_temporal_weighting_summary() -> None:
    report = build_backtest_report_payload(
        _quality(),
        _baseline_metrics(),
        _trained_metrics(),
        temporal_weighting_summary={
            "temporal_weighting_policies_available": {"tabular": ["uniform", "season_priority"]},
            "best_temporal_weighting_policy_by_checkpoint": {
                "after_fp3": {
                    "temporal_weighting_policy": "season_priority",
                    "mae_gap_sec": 0.9,
                }
            },
            "temporal_weighting_vs_uniform_delta_by_checkpoint": {
                "after_fp3": {"season_priority": -0.1}
            },
        },
    )

    assert report["temporal_weighting_policies_available"]["tabular"] == [
        "uniform",
        "season_priority",
    ]
    assert (
        report["best_temporal_weighting_policy_by_checkpoint"]["after_fp3"][
            "temporal_weighting_policy"
        ]
        == "season_priority"
    )
    assert report["temporal_weighting_vs_uniform_delta_by_checkpoint"]["after_fp3"][
        "season_priority"
    ] == pytest.approx(-0.1)


def test_backtest_report_includes_season_aware_validation_summary() -> None:
    report = build_backtest_report_payload(
        _quality(),
        _baseline_metrics(),
        _trained_metrics(),
        season_aware_validation_summary={
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
        },
    )

    assert report["season_aware_validation_available"] is True
    assert report["season_aware_fp3_candidate_summary"]["candidate_mae_gap_sec"] == 0.95
    assert report["season_aware_promotion_recommendation"] == "insufficient_evidence"


def test_backtest_report_includes_season_aware_champion_summary() -> None:
    report = build_backtest_report_payload(
        _quality(),
        _baseline_metrics(),
        _trained_metrics(),
        season_aware_champion_summary={
            "status": "complete",
            "fp3_summary": {"delta_vs_static_sec": -0.05},
            "bootstrap_ci": {"ci_low": -0.1, "ci_high": 0.02},
            "promotion_recommendation": "season_aware_candidate_experimental",
        },
    )

    assert report["season_aware_champion_available"] is True
    assert report["season_aware_champion_fp3_summary"]["delta_vs_static_sec"] == -0.05
    assert (
        report["season_aware_champion_promotion_recommendation"]
        == "season_aware_candidate_experimental"
    )


def test_backtest_report_includes_season_aware_candidate_audit_summary() -> None:
    report = build_backtest_report_payload(
        _quality(),
        _baseline_metrics(),
        _trained_metrics(),
        season_aware_candidate_audit_summary={
            "status": "complete",
            "recommendation": "retain_gates_and_collect_more_history",
            "live_gate_summary": {
                "candidate_selected_folds": 0,
                "gate_failure_reasons": {"insufficient_candidate_history": 3},
            },
            "artifact_alignment_summary": {"unmatched_rows": 0},
            "live_audit_metric_consistency_rate": 1.0,
            "live_audit_selection_consistency_rate": 0.75,
            "comparator_scope_description": "aligned_prior_rows_v1",
        },
    )

    assert report["season_aware_candidate_audit_available"] is True
    assert (
        report["season_aware_candidate_audit_recommendation"]
        == "retain_gates_and_collect_more_history"
    )
    assert report["season_aware_candidate_gate_failure_summary"]["candidate_selected_folds"] == 0
    assert report["season_aware_candidate_comparator_consistency_rate"] == 1.0
    assert report["season_aware_candidate_selection_consistency_rate"] == 0.75
    assert report["season_aware_candidate_comparator_scope"] == "aligned_prior_rows_v1"


def _quality() -> dict[str, object]:
    return {
        "n_rows": 120,
        "n_events": 2,
        "events": ["2024/monza", "2024/suzuka"],
        "n_drivers": 20,
        "checkpoints": ["after_fp1"],
    }


def _baseline_metrics() -> dict[str, object]:
    return {
        "push": {
            "after_fp1": {
                "mae_gap_sec": 0.5,
                "mean_abs_position_error": 3.0,
            }
        },
        "valid": {
            "after_fp1": {
                "mae_gap_sec": 0.6,
                "mean_abs_position_error": 2.5,
            }
        },
    }


def _trained_metrics() -> dict[str, object]:
    return {
        "status": "trained",
        "models": {
            "ridge": {
                "after_fp1": {
                    "mae_gap_sec": 0.3,
                    "mean_abs_position_error": 2.0,
                }
            },
            "random_forest": {
                "after_fp1": {
                    "mae_gap_sec": 0.4,
                    "mean_abs_position_error": 2.2,
                }
            },
            "mean_target": {
                "after_fp1": {
                    "mae_gap_sec": 0.7,
                    "mean_abs_position_error": 5.0,
                }
            },
        },
        "best_baseline_by_checkpoint": {
            "after_fp1": {
                "baseline_name": "push",
                "mae_gap_sec": 0.4,
                "mean_abs_position_error": 3.0,
            }
        },
    }


def _multi_fold_metrics(strategy: str, folds: int) -> dict[str, object]:
    return {
        "status": "complete",
        "strategy": strategy,
        "n_folds_successful": folds,
        "n_folds_failed": 0,
        "tabular_models": ["ridge", "random_forest"],
        "best_model_by_checkpoint": {
            "after_fp1": {
                "model_name": "ridge",
                "mae_gap_sec": 0.3,
                "mean_abs_position_error": 2.0,
            }
        },
        "best_baseline_by_checkpoint": {
            "after_fp1": {
                "baseline_name": "push",
                "mae_gap_sec": 0.4,
                "mean_abs_position_error": 3.0,
            }
        },
    }


def _champion_metrics(selection_mode: str, mae: float) -> dict[str, object]:
    return {
        "status": "complete",
        "selection_mode": selection_mode,
        "metrics_by_checkpoint": {
            "after_fp1": {
                "mae_gap_sec": mae,
                "mean_abs_position_error": 2.0,
                "interval_coverage": 0.9,
                "mean_interval_width_sec": 1.2,
                "median_interval_width_sec": 1.0,
                "interval_availability_rate": 0.8,
            }
        },
        "champion_vs_best_baseline_delta_mae": {"after_fp1": mae - 0.4},
        "champion_vs_best_single_family_delta_mae": {"after_fp1": mae - 0.35},
        "interval_metrics_by_predicted_gap_bucket": {
            "after_fp1": {
                "pole_contender": {
                    "rows_with_interval": 20,
                    "coverage": 0.95,
                    "mean_interval_width_sec": 1.0,
                    "median_interval_width_sec": 0.9,
                    "mean_abs_error_gap_sec": 0.2,
                    "miss_count": 1,
                }
            }
        },
    }
