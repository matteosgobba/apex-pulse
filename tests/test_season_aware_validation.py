import json
from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.config import DataConfig
from f1_prediction.modeling.season_aware_validation import (
    build_cold_start_comparison,
    build_event_level_comparison,
    build_fp3_candidate_comparison,
    build_policy_aligned_rows,
    build_season_level_comparison,
    create_season_aware_validation_report,
    current_season_evidence_regime,
    generate_season_aware_figures,
    load_training_composition,
    paired_bootstrap_mean_ci,
    standardize_candidate_predictions,
)


def test_static_and_candidate_predictions_join_only_identical_rows() -> None:
    candidates = standardize_candidate_predictions(
        [
            _predictions("ablation", "uniform", include_extra_static_driver=True),
            _predictions("ablation", "current_season_only_with_prior"),
        ]
    )
    composition = load_training_composition(_metrics_payloads())

    comparison = build_fp3_candidate_comparison(candidates, composition)

    assert len(comparison) == 2
    assert set(comparison["driver"]) == {"NOR", "PIA"}


def test_fp3_fixed_candidate_comparison_computes_deltas() -> None:
    candidates = standardize_candidate_predictions(
        [
            _predictions("ablation", "uniform"),
            _predictions("ablation", "current_season_only_with_prior"),
        ]
    )
    comparison = build_fp3_candidate_comparison(
        candidates, load_training_composition(_metrics_payloads())
    )

    nor = comparison[comparison["driver"].eq("NOR")].iloc[0]
    assert nor["static_abs_error_gap_sec"] == pytest.approx(0.2)
    assert nor["candidate_abs_error_gap_sec"] == pytest.approx(0.1)
    assert nor["error_delta_vs_static_sec"] == pytest.approx(-0.1)
    assert bool(nor["candidate_better_than_static"]) is True


def test_candidate_comparison_separates_test_seasons() -> None:
    candidates = standardize_candidate_predictions(
        [
            _predictions("ablation", "uniform", include_second_season=True),
            _predictions(
                "ablation",
                "current_season_only_with_prior",
                include_second_season=True,
            ),
        ]
    )
    composition = load_training_composition(_metrics_payloads(include_second_season=True))
    fp3 = build_fp3_candidate_comparison(candidates, composition)
    policy_rows = build_policy_aligned_rows(candidates, composition)
    season_level = build_season_level_comparison(policy_rows, fp3)

    assert set(season_level["season"]) == {2024, 2025}


def test_cold_start_bins_use_documented_boundaries() -> None:
    assert current_season_evidence_regime(4) == "cold_start"
    assert current_season_evidence_regime(5) == "early_season"
    assert current_season_evidence_regime(8) == "early_season"
    assert current_season_evidence_regime(9) == "established_season"


def test_training_composition_and_effective_sample_size_propagate() -> None:
    candidates = standardize_candidate_predictions(
        [
            _predictions("ablation", "uniform"),
            _predictions("ablation", "current_season_only_with_prior"),
        ]
    )
    comparison = build_fp3_candidate_comparison(
        candidates,
        load_training_composition(_metrics_payloads()),
    )
    event_level = build_event_level_comparison(comparison)

    row = event_level.iloc[0]
    assert row["effective_sample_size"] == pytest.approx(12.5)
    assert row["same_season_weight_share"] == pytest.approx(0.7)
    assert row["prior_season_weight_share"] == pytest.approx(0.2)


def test_event_level_bootstrap_confidence_interval_is_deterministic() -> None:
    first = paired_bootstrap_mean_ci([-0.1, 0.2, -0.3], seed=123, iterations=200)
    second = paired_bootstrap_mean_ci([-0.1, 0.2, -0.3], seed=123, iterations=200)

    assert first == second
    assert first["events"] == 3
    assert first["mean_delta"] == pytest.approx((-0.1 + 0.2 - 0.3) / 3)


def test_summary_json_handles_missing_optional_artifacts(tmp_path: Path) -> None:
    summary = create_season_aware_validation_report(_config(tmp_path))
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))

    assert summary.status == "partial"
    assert payload["season_aware_validation_available"] is False
    assert "ablation_uniform_predictions.parquet" in payload["missing_inputs"]
    assert (tmp_path / "reports/metrics/season_aware_fp3_candidate_comparison.csv").is_file()


def test_figure_generation_does_not_crash_on_minimal_synthetic_inputs(tmp_path: Path) -> None:
    candidates = standardize_candidate_predictions(
        [
            _predictions("ablation", "uniform"),
            _predictions("ablation", "current_season_only_with_prior"),
        ]
    )
    composition = load_training_composition(_metrics_payloads())
    fp3 = build_fp3_candidate_comparison(candidates, composition)
    event_level = build_event_level_comparison(fp3)
    policy_rows = build_policy_aligned_rows(candidates, composition)
    season_level = build_season_level_comparison(policy_rows, fp3)
    cold_start = build_cold_start_comparison(policy_rows, fp3)

    paths, issues = generate_season_aware_figures(
        figures_dir=tmp_path,
        fp3_comparison=fp3,
        event_level=event_level,
        season_level=season_level,
        cold_start=cold_start,
        composition=composition,
    )

    assert isinstance(issues, list)
    assert all(path.is_file() for path in paths)


def _predictions(
    family: str,
    policy: str,
    *,
    include_extra_static_driver: bool = False,
    include_second_season: bool = False,
) -> pd.DataFrame:
    rows = [
        _prediction_row(
            family, policy, 6, 2024, "Silverstone", "silverstone", "after_fp3", "NOR", 1.0, 1.2, 1.1
        ),
        _prediction_row(
            family, policy, 6, 2024, "Silverstone", "silverstone", "after_fp3", "PIA", 1.5, 1.4, 1.7
        ),
        _prediction_row(
            family, policy, 6, 2024, "Silverstone", "silverstone", "after_fp2", "NOR", 1.0, 1.3, 1.2
        ),
    ]
    if include_extra_static_driver and policy == "uniform":
        rows.append(
            _prediction_row(
                family,
                policy,
                6,
                2024,
                "Silverstone",
                "silverstone",
                "after_fp3",
                "HAM",
                0.8,
                0.9,
                0.9,
            )
        )
    if include_second_season:
        rows.extend(
            [
                _prediction_row(
                    family, policy, 7, 2025, "Monza", "monza", "after_fp3", "NOR", 1.2, 1.4, 1.1
                ),
                _prediction_row(
                    family, policy, 7, 2025, "Monza", "monza", "after_fp3", "PIA", 1.4, 1.3, 1.5
                ),
            ]
        )
    frame = pd.DataFrame(rows)
    frame["predicted_quali_gap_to_pole_sec"] = frame.pop(
        "current_predicted_quali_gap_to_pole_sec"
        if policy == "current_season_only_with_prior"
        else "uniform_predicted_quali_gap_to_pole_sec"
    )
    frame["candidate_family"] = family
    frame["training_policy"] = policy
    return frame


def _prediction_row(
    family: str,
    policy: str,
    fold_id: int,
    season: int,
    event: str,
    event_slug: str,
    checkpoint: str,
    driver: str,
    actual: float,
    uniform_prediction: float,
    current_prediction: float,
) -> dict[str, object]:
    return {
        "fold_id": fold_id,
        "season": season,
        "event": event,
        "event_slug": event_slug,
        "checkpoint": checkpoint,
        "driver": driver,
        "team": "McLaren",
        "prediction_type": "tabular" if family == "ablation" else "boosted",
        "model_name": "random_forest",
        "feature_group": "base_plus_relative",
        "quali_gap_to_pole_sec": actual,
        "uniform_predicted_quali_gap_to_pole_sec": uniform_prediction,
        "current_predicted_quali_gap_to_pole_sec": current_prediction,
        "quali_position": 2,
        "predicted_quali_position": 3,
    }


def _metrics_payloads(
    *, include_second_season: bool = False
) -> dict[tuple[str, str], dict[str, object]]:
    rows = [
        {
            "fold_id": 6,
            "test_event": "2024/silverstone",
            "test_season": 2024,
            "training_rows": 100,
            "training_events": 8,
            "same_season_training_events": 4,
            "prior_season_training_events": 4,
            "older_season_training_events": 0,
            "effective_sample_size": 12.5,
            "same_season_weight_share": 0.7,
            "prior_season_weight_share": 0.2,
            "older_season_weight_share": 0.1,
        }
    ]
    if include_second_season:
        rows.append(
            {
                "fold_id": 7,
                "test_event": "2025/monza",
                "test_season": 2025,
                "training_rows": 120,
                "training_events": 10,
                "same_season_training_events": 9,
                "prior_season_training_events": 1,
                "older_season_training_events": 0,
                "effective_sample_size": 15.0,
                "same_season_weight_share": 0.8,
                "prior_season_weight_share": 0.2,
                "older_season_weight_share": 0.0,
            }
        )
    return {
        ("ablation", "uniform"): {"training_weight_summary_by_fold": rows},
        ("ablation", "current_season_only_with_prior"): {"training_weight_summary_by_fold": rows},
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
        metrics_output_dir=project_root / "reports" / "metrics",
    )
