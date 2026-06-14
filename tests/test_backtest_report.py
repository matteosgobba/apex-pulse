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
