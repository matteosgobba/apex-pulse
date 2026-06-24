from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.config import DataConfig, TemporalWeightingConfig
from f1_prediction.modeling.temporal_weighting import (
    TemporalWeightingPolicy,
    effective_sample_size,
    prepare_temporal_training_data,
)
from f1_prediction.modeling.temporal_weighting_report import (
    build_checkpoint_comparison_table,
    create_temporal_weighting_report,
    generate_temporal_weighting_figures,
)


def test_uniform_weighting_produces_unit_weights() -> None:
    result = prepare_temporal_training_data(
        _training_rows(),
        test_event="2024/event-3",
        event_order=_event_order(),
        config=TemporalWeightingConfig(),
        policy=TemporalWeightingPolicy.uniform,
    )

    assert result.sample_weights.tolist() == [1.0] * len(result.train)
    assert result.summary["effective_sample_size"] == pytest.approx(len(result.train))
    assert result.summary["same_season_training_events"] == 3


def test_season_priority_weights_current_prior_and_older_seasons() -> None:
    result = prepare_temporal_training_data(
        _training_rows(),
        test_event="2024/event-3",
        event_order=_event_order(),
        config=TemporalWeightingConfig(
            current_season_weight=1.0,
            previous_season_weight=0.35,
            older_season_weight=0.1,
        ),
        policy=TemporalWeightingPolicy.season_priority,
    )

    by_event = dict(zip(_event_keys(result.train), result.sample_weights, strict=True))
    assert by_event["2022/event-1"] == pytest.approx(0.1)
    assert by_event["2023/event-1"] == pytest.approx(0.35)
    assert by_event["2024/event-1"] == pytest.approx(1.0)
    assert by_event["2024/event-2"] == pytest.approx(1.0)


def test_season_priority_filters_future_events() -> None:
    result = prepare_temporal_training_data(
        _training_rows(),
        test_event="2024/event-3",
        event_order=_event_order(),
        config=TemporalWeightingConfig(),
        policy=TemporalWeightingPolicy.season_priority,
    )

    assert "2024/event-4" not in set(_event_keys(result.train))
    assert result.summary["filtered_future_rows"] == 1


def test_exponential_recency_declines_with_event_distance() -> None:
    result = prepare_temporal_training_data(
        _training_rows(),
        test_event="2024/event-3",
        event_order=_event_order(),
        config=TemporalWeightingConfig(half_life_events=2),
        policy=TemporalWeightingPolicy.exponential_recency,
    )

    by_event = dict(zip(_event_keys(result.train), result.sample_weights, strict=True))
    assert by_event["2024/event-2"] > by_event["2024/event-1"] > by_event["2023/event-1"]


def test_current_season_only_with_prior_keeps_prior_before_threshold() -> None:
    result = prepare_temporal_training_data(
        _training_rows(),
        test_event="2024/event-3",
        event_order=_event_order(),
        config=TemporalWeightingConfig(min_current_season_events=5),
        policy=TemporalWeightingPolicy.current_season_only_with_prior,
    )

    assert "2023/event-1" in set(_event_keys(result.train))
    assert result.summary["prior_season_training_rows"] == 1
    assert result.summary["same_season_training_events"] == 2


def test_current_season_only_with_prior_excludes_prior_after_threshold() -> None:
    result = prepare_temporal_training_data(
        _training_rows(),
        test_event="2024/event-3",
        event_order=_event_order(),
        config=TemporalWeightingConfig(min_current_season_events=2),
        policy=TemporalWeightingPolicy.current_season_only_with_prior,
    )

    assert set(_event_keys(result.train)) == {"2024/event-1", "2024/event-2"}
    assert result.summary["prior_season_training_rows"] == 0


def test_effective_sample_size_formula() -> None:
    assert effective_sample_size(pd.Series([1.0, 1.0, 0.5])) == pytest.approx(2.7777777778)


def test_temporal_weighting_report_handles_missing_artifacts(tmp_path: Path) -> None:
    summary = create_temporal_weighting_report(_config(tmp_path))
    payload = pd.read_json(summary.summary_path, typ="series")

    assert summary.status == "partial"
    assert payload["missing_artifacts"]
    assert (tmp_path / "metrics/temporal_weighting_checkpoint_comparison.csv").is_file()


def test_temporal_weighting_report_computes_deltas_vs_uniform() -> None:
    frame = build_checkpoint_comparison_table(
        {
            "tabular": {
                "uniform": _metrics_payload("uniform", 1.0),
                "season_priority": _metrics_payload("season_priority", 0.8),
            }
        }
    )

    row = frame[frame["temporal_weighting_policy"].eq("season_priority")].iloc[0]
    assert row["delta_vs_uniform_mae_gap_sec"] == pytest.approx(-0.2)


def test_temporal_weighting_figure_generation_does_not_crash(tmp_path: Path) -> None:
    checkpoint = build_checkpoint_comparison_table(
        {
            "tabular": {
                "uniform": _metrics_payload("uniform", 1.0),
                "season_priority": _metrics_payload("season_priority", 0.8),
            }
        }
    )
    composition = pd.DataFrame(
        [
            {
                "source": "tabular",
                "temporal_weighting_policy": "uniform",
                "same_season_weight_share": 1.0,
                "effective_sample_size": 10.0,
            }
        ]
    )

    paths, issues = generate_temporal_weighting_figures(tmp_path, checkpoint, composition)

    assert not issues
    assert paths


def _training_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": 2022, "event_slug": "event-1"},
            {"season": 2023, "event_slug": "event-1"},
            {"season": 2024, "event_slug": "event-1"},
            {"season": 2024, "event_slug": "event-2"},
            {"season": 2024, "event_slug": "event-4"},
        ]
    )


def _event_order() -> list[str]:
    return [
        "2022/event-1",
        "2023/event-1",
        "2024/event-1",
        "2024/event-2",
        "2024/event-3",
        "2024/event-4",
    ]


def _event_keys(frame: pd.DataFrame) -> list[str]:
    return (frame["season"].astype(str) + "/" + frame["event_slug"].astype(str)).tolist()


def _metrics_payload(policy: str, mae: float) -> dict[str, object]:
    return {
        "status": "complete",
        "temporal_weighting_policy": policy,
        "best_model_by_checkpoint": {
            "after_fp3": {"model_name": "random_forest", "mae_gap_sec": mae}
        },
        "metrics_by_model_checkpoint": {
            "random_forest": {
                "after_fp3": {
                    "global": {
                        "mae_gap_sec": mae,
                        "rmse_gap_sec": mae + 0.1,
                        "median_abs_error_gap_sec": mae,
                        "mean_abs_position_error": 1.0,
                    }
                }
            }
        },
        "n_folds_successful": 2,
        "training_weight_summary_by_fold": [
            {
                "effective_sample_size": 5.0,
                "same_season_weight_share": 0.8,
                "prior_season_weight_share": 0.2,
            }
        ],
    }


def _config(project_root: Path) -> DataConfig:
    return DataConfig(
        project_root=project_root,
        fastf1_cache_dir=project_root / "cache",
        lap_output_dir=project_root / "laps",
        session_metadata_output_dir=project_root / "metadata",
        clean_lap_output_dir=project_root / "clean_laps",
        session_features_output_dir=project_root / "session_features",
        modeling_output_dir=project_root / "modeling",
        metrics_output_dir=project_root / "metrics",
    )
