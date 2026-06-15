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
