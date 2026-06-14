import pandas as pd
import pytest

from f1_prediction.modeling.baselines import predict_baseline, select_checkpoint_metric
from f1_prediction.modeling.metrics import compute_prediction_metrics


def test_selected_metric_falls_back_to_earlier_session() -> None:
    rows = pd.DataFrame(
        {
            "fp1_best_push_lap_time_sec": [81.0, 82.0],
            "fp2_best_push_lap_time_sec": [80.0, pd.NA],
        }
    )

    metric, source = select_checkpoint_metric(
        rows,
        "after_fp2",
        "best_push_lap_time_sec",
    )

    assert metric.tolist() == [80.0, 82.0]
    assert source.tolist() == ["FP2", "FP1"]


def test_baseline_computes_event_gap_and_ranks_missing_last() -> None:
    dataset = _modeling_rows("after_fp2")
    dataset.loc[dataset["driver"].eq("BOT"), "fp1_best_push_lap_time_sec"] = pd.NA

    predictions = predict_baseline(dataset, "best_push_lap").set_index("driver")

    assert predictions.loc["NOR", "predicted_quali_gap_to_pole_sec"] == pytest.approx(0.0)
    assert predictions.loc["VER", "predicted_quali_gap_to_pole_sec"] == pytest.approx(0.2)
    assert pd.isna(predictions.loc["BOT", "predicted_quali_gap_to_pole_sec"])
    assert predictions.loc["NOR", "predicted_quali_position"] == 1
    assert predictions.loc["VER", "predicted_quali_position"] == 2
    assert predictions.loc["BOT", "predicted_quali_position"] == 3


def test_predicted_gap_resets_within_each_event_checkpoint() -> None:
    monza = _modeling_rows("after_fp1")
    spa = _modeling_rows("after_fp1").assign(
        event="Spa",
        event_slug="spa",
        fp1_best_push_lap_time_sec=[90.0, 90.3, 91.0],
    )

    predictions = predict_baseline(pd.concat([monza, spa], ignore_index=True), "best_push_lap")
    leaders = predictions.loc[
        predictions.groupby(["event_slug", "checkpoint"])["selected_practice_metric_sec"].idxmin()
    ]

    assert leaders["predicted_quali_gap_to_pole_sec"].eq(0.0).all()


def test_after_fp1_baseline_does_not_use_future_session_values() -> None:
    dataset = _modeling_rows("after_fp1")
    dataset["fp2_best_push_lap_time_sec"] = [1.0, 200.0, 300.0]
    dataset["fp3_best_push_lap_time_sec"] = [0.5, 400.0, 500.0]

    predictions = predict_baseline(dataset, "best_push_lap").set_index("driver")

    assert predictions.loc["NOR", "selected_practice_metric_sec"] == pytest.approx(80.0)
    assert predictions.loc["NOR", "selected_practice_session"] == "FP1"
    assert predictions.loc["NOR", "predicted_quali_position"] == 1


def test_metrics_computation_on_synthetic_predictions() -> None:
    predictions = pd.DataFrame(
        {
            "season": [2024] * 3,
            "event_slug": ["monza"] * 3,
            "driver": ["NOR", "VER", "BOT"],
            "quali_gap_to_pole_sec": [0.0, 0.2, 1.0],
            "predicted_quali_gap_to_pole_sec": [0.0, 0.4, 0.8],
            "quali_position": [1, 2, 3],
            "predicted_quali_position": [1, 3, 2],
            "reached_q3": [1, 1, 0],
            "predicted_reached_q3": [1, 1, 1],
        }
    )

    metrics = compute_prediction_metrics(predictions)

    assert metrics["mae_gap_sec"] == pytest.approx(0.1333333333)
    assert metrics["rmse_gap_sec"] == pytest.approx((0.08 / 3) ** 0.5)
    assert metrics["median_abs_error_gap_sec"] == pytest.approx(0.2)
    assert metrics["mean_abs_position_error"] == pytest.approx(2 / 3)
    assert metrics["spearman_corr"] == pytest.approx(0.5)
    assert metrics["top_3_accuracy"] == pytest.approx(1.0)
    assert metrics["q3_accuracy"] == pytest.approx(2 / 3)


def _modeling_rows(checkpoint: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024] * 3,
            "event": ["Monza"] * 3,
            "event_slug": ["monza"] * 3,
            "checkpoint": [checkpoint] * 3,
            "driver": ["NOR", "VER", "BOT"],
            "team": ["McLaren", "Red Bull Racing", "Kick Sauber"],
            "fp1_best_push_lap_time_sec": [80.0, 80.2, 81.0],
            "fp2_best_push_lap_time_sec": [80.0, 80.2, pd.NA],
            "fp3_best_push_lap_time_sec": [79.5, 79.7, 80.5],
            "quali_position": pd.Series([1, 2, 3], dtype="Int64"),
            "quali_best_lap_time_sec": [79.0, 79.2, 80.0],
            "quali_gap_to_pole_sec": [0.0, 0.2, 1.0],
            "reached_q2": [1, 1, 1],
            "reached_q3": [1, 1, 1],
        }
    )
