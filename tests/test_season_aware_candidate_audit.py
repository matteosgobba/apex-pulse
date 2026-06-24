import json
from pathlib import Path

import pandas as pd

from f1_prediction.config import DataConfig, SeasonAwareNestedGuardedChampionConfig
from f1_prediction.modeling.season_aware_candidate_audit import (
    build_candidate_alignment,
    build_candidate_eligibility_by_fold,
    build_candidate_gate_failures,
    build_candidate_gate_sensitivity,
    build_candidate_history_by_fold,
    build_comparator_consistency_report,
    create_season_aware_candidate_audit_report,
    generate_candidate_audit_figures,
)


def test_candidate_audit_identifies_absent_weighted_artifacts(tmp_path: Path) -> None:
    summary = create_season_aware_candidate_audit_report(_config(tmp_path), _settings())
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))

    assert summary.status == "partial"
    assert (
        "ablation_current_season_only_with_prior_predictions.parquet" in payload["missing_inputs"]
    )
    assert payload["recommendation"] == "artifact_pipeline_fix_required"
    assert (tmp_path / "reports/metrics/season_aware_candidate_eligibility_by_fold.csv").is_file()


def test_candidate_audit_identifies_missing_folds_and_schema_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    metrics_dir = config.metrics_output_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    _predictions("uniform", folds=[1, 2]).to_parquet(
        metrics_dir / "ablation_uniform_predictions.parquet"
    )
    broken = _predictions("current_season_only_with_prior", folds=[1]).drop(
        columns=["predicted_quali_gap_to_pole_sec"]
    )
    broken.to_parquet(metrics_dir / "ablation_current_season_only_with_prior_predictions.parquet")
    _metrics([1, 2]).to_json(metrics_dir / "ablation_uniform_metrics.json")
    _metrics([1]).to_json(metrics_dir / "ablation_current_season_only_with_prior_metrics.json")
    _selection([1, 2]).to_parquet(
        metrics_dir / "champion_season_aware_nested_guarded_selection.parquet"
    )

    summary = create_season_aware_candidate_audit_report(config, _settings())
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    alignment = pd.read_csv(metrics_dir / "season_aware_candidate_alignment.csv")

    assert payload["schema_issues"]
    missing_folds = alignment.loc[~alignment["weighted_candidate_fold_found"], "fold_id"]
    assert 1 in set(missing_folds)
    assert payload["recommendation"] == "artifact_pipeline_fix_required"


def test_candidate_history_excludes_current_test_fold_event() -> None:
    weighted = _predictions("current_season_only_with_prior", folds=[1, 2, 3, 4, 5, 6])
    default = _predictions("uniform", folds=[1, 2, 3, 4, 5, 6])
    selection = _selection([6], prior_events=5)
    history = build_candidate_history_by_fold(
        weighted,
        default,
        selection,
        _composition([6]),
        _settings(),
    )

    row = history[history["fold_id"].eq(6)].iloc[0]
    assert row["weighted_candidate_prior_folds"] == 5
    assert "2024/event_6" not in row["historical_source_events"]
    assert bool(row["current_event_in_history"]) is False


def test_each_gate_failure_reason_is_assigned_correctly() -> None:
    settings = _settings(min_events=5, min_folds=5, min_predictions=10, margin=0.05)
    history = pd.DataFrame(
        [
            _history_row(
                1,
                rows=0,
                prior_events=5,
                prior_folds=5,
                prior_predictions=10,
                delta=-0.2,
            ),
            _history_row(
                2,
                rows=2,
                prior_events=4,
                prior_folds=5,
                prior_predictions=10,
                delta=-0.2,
            ),
            _history_row(3, rows=2, prior_events=5, prior_folds=3, prior_predictions=6, delta=-0.2),
            _history_row(
                4,
                rows=2,
                prior_events=5,
                prior_folds=5,
                prior_predictions=10,
                delta=-0.01,
            ),
        ]
    )

    eligibility = build_candidate_eligibility_by_fold(history, _selection([1, 2, 3, 4]), settings)
    failures = build_candidate_gate_failures(eligibility)

    assert set(eligibility["selection_reason"]) == {
        "weighted_candidate_missing",
        "season_aware_cold_start",
        "insufficient_candidate_history",
        "margin_not_met",
    }
    assert set(failures["gate_name"]) == {
        "candidate_prediction_gate",
        "cold_start_gate",
        "candidate_history_gate",
        "margin_gate",
    }


def test_candidate_alignment_joins_only_identical_fold_event_checkpoint_driver_rows() -> None:
    weighted = _predictions("current_season_only_with_prior", folds=[6], drivers=("NOR", "PIA"))
    default = _predictions("uniform", folds=[6], drivers=("NOR", "HAM"))

    alignment = build_candidate_alignment(weighted, default, _selection([6]))

    row = alignment.iloc[0]
    assert row["unmatched_weighted_candidate_rows"] == 1
    assert row["unmatched_default_candidate_rows"] == 1


def test_sensitivity_simulation_is_deterministic_and_prior_only() -> None:
    weighted = _predictions("current_season_only_with_prior", folds=[1, 2, 3, 4, 5, 6])
    default = _predictions("uniform", folds=[1, 2, 3, 4, 5, 6])
    history = build_candidate_history_by_fold(
        weighted,
        default,
        _selection([6], prior_events=5),
        _composition([6]),
        _settings(min_folds=5, min_predictions=10),
    )
    weighted_bad_current = weighted.copy()
    weighted_bad_current.loc[weighted_bad_current["fold_id"].eq(6), "quali_gap_to_pole_sec"] = 50.0

    first_detail, first_summary = build_candidate_gate_sensitivity(
        weighted,
        default,
        history,
        _settings(min_folds=5, min_predictions=10),
    )
    second_detail, second_summary = build_candidate_gate_sensitivity(
        weighted,
        default,
        history,
        _settings(min_folds=5, min_predictions=10),
    )
    changed_detail, _ = build_candidate_gate_sensitivity(
        weighted_bad_current,
        default,
        history,
        _settings(min_folds=5, min_predictions=10),
    )

    pd.testing.assert_frame_equal(first_summary, second_summary)
    assert set(first_detail["candidate_eligible"]) == set(changed_detail["candidate_eligible"])
    assert first_summary["bootstrap_seed"].nunique() == 1


def test_live_and_audit_metrics_match_on_synthetic_artifacts() -> None:
    settings = _settings(min_folds=1, min_predictions=2, margin=0.05)
    weighted = _predictions("current_season_only_with_prior", folds=[1, 2])
    default = _predictions("uniform", folds=[1, 2])
    history = build_candidate_history_by_fold(
        weighted,
        default,
        _selection([2], prior_events=5),
        _composition([2]),
        settings,
    )
    selection = _selection([2], prior_events=5).assign(
        season_aware_candidate_metric_value=0.05,
        season_aware_default_metric_value=0.2,
        metric_scope_candidate_mae=0.05,
        metric_scope_default_mae=0.2,
        metric_scope_improvement_sec=0.15,
        season_aware_selection_reason="selected_after_prior_evidence",
        season_aware_selected=True,
    )
    eligibility = build_candidate_eligibility_by_fold(history, selection, settings)

    consistency = build_comparator_consistency_report(eligibility, selection)

    row = consistency[consistency["fold_id"].eq(2)].iloc[0]
    assert bool(row["candidate_metric_match"]) is True
    assert bool(row["default_metric_match"]) is True
    assert bool(row["improvement_match"]) is True
    assert bool(row["selection_reason_match"]) is True


def test_comparator_consistency_report_identifies_mismatches() -> None:
    settings = _settings(min_folds=1, min_predictions=2, margin=0.05)
    weighted = _predictions("current_season_only_with_prior", folds=[1, 2])
    default = _predictions("uniform", folds=[1, 2])
    history = build_candidate_history_by_fold(
        weighted,
        default,
        _selection([2], prior_events=5),
        _composition([2]),
        settings,
    )
    selection = _selection([2], prior_events=5).assign(
        metric_scope_candidate_mae=0.2,
        metric_scope_default_mae=0.2,
        metric_scope_improvement_sec=0.0,
        season_aware_selection_reason="margin_not_met",
    )
    eligibility = build_candidate_eligibility_by_fold(history, selection, settings)

    consistency = build_comparator_consistency_report(eligibility, selection)

    row = consistency[consistency["fold_id"].eq(2)].iloc[0]
    assert bool(row["candidate_metric_match"]) is False
    assert bool(row["improvement_match"]) is False
    assert bool(row["selection_reason_match"]) is False


def test_figure_generation_does_not_crash_on_minimal_valid_inputs(tmp_path: Path) -> None:
    settings = _settings(min_folds=1, min_predictions=2, margin=0.0)
    weighted = _predictions("current_season_only_with_prior", folds=[1, 2])
    default = _predictions("uniform", folds=[1, 2])
    history = build_candidate_history_by_fold(
        weighted,
        default,
        _selection([2], prior_events=5),
        _composition([2]),
        settings,
    )
    eligibility = build_candidate_eligibility_by_fold(history, _selection([2]), settings)
    consistency = build_comparator_consistency_report(eligibility, _selection([2]))
    _, sensitivity_summary = build_candidate_gate_sensitivity(weighted, default, history, settings)

    paths, issues = generate_candidate_audit_figures(
        figures_dir=tmp_path,
        eligibility=eligibility,
        history=history,
        consistency=consistency,
        sensitivity_summary=sensitivity_summary,
    )

    assert isinstance(issues, list)
    assert all(path.is_file() for path in paths)


def test_existing_static_selection_inputs_are_not_modified() -> None:
    settings = _settings(min_folds=1, min_predictions=2, margin=0.0)
    weighted = _predictions("current_season_only_with_prior", folds=[1, 2])
    default = _predictions("uniform", folds=[1, 2])
    default_before = default.copy(deep=True)

    build_candidate_history_by_fold(weighted, default, _selection([2]), _composition([2]), settings)

    pd.testing.assert_frame_equal(default, default_before)


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


def _settings(
    *,
    min_events: int = 5,
    min_folds: int = 5,
    min_predictions: int = 10,
    margin: float = 0.05,
) -> SeasonAwareNestedGuardedChampionConfig:
    return SeasonAwareNestedGuardedChampionConfig(
        min_current_season_prior_events=min_events,
        min_prior_candidate_folds=min_folds,
        min_prior_candidate_predictions=min_predictions,
        improvement_margin_sec=margin,
    )


def _predictions(
    policy: str,
    *,
    folds: list[int],
    drivers: tuple[str, ...] = ("NOR", "PIA"),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_id in folds:
        for index, driver in enumerate(drivers):
            actual = 1.0 + index * 0.2
            error = 0.05 if policy == "current_season_only_with_prior" else 0.2
            rows.append(
                {
                    "fold_id": fold_id,
                    "season": 2024,
                    "event": f"Event {fold_id}",
                    "event_slug": f"event_{fold_id}",
                    "checkpoint": "after_fp3",
                    "driver": driver,
                    "team": "McLaren",
                    "prediction_type": "tabular",
                    "model_name": "random_forest",
                    "feature_group": "base_plus_relative",
                    "quali_gap_to_pole_sec": actual,
                    "predicted_quali_gap_to_pole_sec": actual + error,
                    "quali_position": index + 1,
                    "predicted_quali_position": index + 1,
                    "candidate_family": "ablation",
                    "candidate_model_name": "random_forest",
                    "candidate_feature_group": "base_plus_relative",
                    "training_policy": policy,
                    "temporal_weighting_policy": policy,
                }
            )
    return pd.DataFrame(rows)


def _selection(folds: list[int], *, prior_events: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": folds,
            "season": [2024] * len(folds),
            "event": [f"Event {fold}" for fold in folds],
            "event_slug": [f"event_{fold}" for fold in folds],
            "checkpoint": ["after_fp3"] * len(folds),
            "current_season_prior_event_count": [prior_events] * len(folds),
            "season_aware_selected": [False] * len(folds),
            "season_aware_selection_reason": ["insufficient_candidate_history"] * len(folds),
        }
    )


def _composition(folds: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "training_policy": ["current_season_only_with_prior"] * len(folds),
            "fold_id": folds,
            "test_event": [f"2024/event_{fold}" for fold in folds],
            "test_season": [2024] * len(folds),
            "same_season_training_events": [5] * len(folds),
            "prior_season_training_events": [3] * len(folds),
            "effective_sample_size": [10.0] * len(folds),
            "same_season_weight_share": [0.7] * len(folds),
            "prior_season_weight_share": [0.3] * len(folds),
        }
    )


def _metrics(folds: list[int]) -> pd.DataFrame:
    return pd.Series(
        {
            "training_weight_summary_by_fold": [
                {
                    "fold_id": fold,
                    "test_event": f"2024/event_{fold}",
                    "test_season": 2024,
                    "same_season_training_events": 5,
                    "prior_season_training_events": 3,
                    "effective_sample_size": 10.0,
                    "same_season_weight_share": 0.7,
                    "prior_season_weight_share": 0.3,
                }
                for fold in folds
            ]
        }
    )


def _history_row(
    fold_id: int,
    *,
    rows: int,
    prior_events: int,
    prior_folds: int,
    prior_predictions: int,
    delta: float,
) -> dict[str, object]:
    return {
        "fold_id": fold_id,
        "season": 2024,
        "event": f"Event {fold_id}",
        "event_slug": f"event_{fold_id}",
        "checkpoint": "after_fp3",
        "current_season_prior_event_count": prior_events,
        "weighted_candidate_artifact_available": rows > 0,
        "weighted_candidate_prediction_rows": rows,
        "default_candidate_prediction_rows": rows,
        "weighted_candidate_prior_folds": prior_folds,
        "weighted_candidate_prior_predictions": prior_predictions,
        "default_prior_folds": prior_folds,
        "default_prior_predictions": prior_predictions,
        "weighted_candidate_metric_value": 0.5 + delta,
        "default_metric_value": 0.5,
        "improvement_delta_sec": delta,
        "metric_scope_candidate_mae": 0.5 + delta,
        "metric_scope_default_mae": 0.5,
        "metric_scope_improvement_sec": -delta,
        "required_min_prior_candidate_folds": 5,
        "required_min_prior_candidate_predictions": 10,
        "required_min_current_season_prior_events": 5,
        "required_improvement_margin_sec": 0.05,
        "historical_source_events": "",
        "current_event_in_history": False,
        "effective_sample_size": 10.0,
        "same_season_weight_share": 0.7,
        "prior_season_weight_share": 0.3,
    }
