import json
from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.config import (
    ChampionMethodConfig,
    ChampionPolicyConfig,
    DataConfig,
    ModelConfig,
    PredictedGapBucketUncertaintyConfig,
    RandomForestConfig,
    SeasonAwareNestedGuardedChampionConfig,
    StabilizedNestedChampionConfig,
    StabilizedNestedGuardedChampionConfig,
    UncertaintyConfig,
)
from f1_prediction.modeling.backtest_tabular import (
    BacktestFold,
    BacktestStrategy,
    build_backtest_folds,
)
from f1_prediction.modeling.champion_policy import (
    ChampionSelectionMode,
    ChampionUncertaintyMethod,
    add_predicted_gap_bucket_conformal_uncertainty,
    add_prior_residual_uncertainty,
    apply_stabilized_nested_guardrail,
    assign_predicted_gap_bucket,
    build_champion_metrics_payload,
    compare_prior_evidence,
    is_practice_baseline_method,
    resolve_static_champion_policy,
    run_champion_backtest,
    select_nested_method,
    select_season_aware_guarded_method,
    select_stabilized_nested_method,
)


def test_static_champion_policy_resolves_expected_methods() -> None:
    policy = resolve_static_champion_policy(_model_config())

    assert policy["after_fp1"].model_name == "robust_best_push_lap"
    assert policy["after_fp2"].model_name == "robust_theoretical_best_lap"
    assert policy["after_fp3"] == ChampionMethodConfig(
        family="ablation",
        model_name="random_forest",
        feature_group="base_plus_relative",
    )


def test_practice_baseline_detection_uses_family_and_model_name() -> None:
    assert is_practice_baseline_method(
        ChampionMethodConfig("robust_baseline", "robust_best_valid_lap")
    )
    assert is_practice_baseline_method(ChampionMethodConfig("baseline", "best_push_lap"))
    assert not is_practice_baseline_method(
        ChampionMethodConfig("ablation", "random_forest", "base_plus_relative")
    )


def test_guardrail_does_not_apply_to_fp1_or_fp2() -> None:
    fallback = _static_fp3_method()
    selected = ChampionMethodConfig("robust_baseline", "best_valid_lap")

    for checkpoint in ("after_fp1", "after_fp2"):
        decision = apply_stabilized_nested_guardrail(
            selected=selected,
            fallback=fallback,
            checkpoint=checkpoint,
            settings=StabilizedNestedGuardedChampionConfig(),
        )

        assert decision.selected == selected
        assert decision.guardrail_applied is False


def test_guardrail_applies_to_fp3_static_rf_to_baseline_switch() -> None:
    fallback = _static_fp3_method()
    selected = ChampionMethodConfig("robust_baseline", "best_valid_lap")

    decision = apply_stabilized_nested_guardrail(
        selected=selected,
        fallback=fallback,
        checkpoint="after_fp3",
        settings=StabilizedNestedGuardedChampionConfig(),
    )

    assert decision.selected == fallback
    assert decision.pre_guardrail_selected == selected
    assert decision.guardrail_applied is True
    assert decision.guardrail_name == "fp3_no_baseline_switch"
    assert decision.guardrail_reason == "prevent_fp3_baseline_switch_from_static_rf"


def test_guardrail_does_not_apply_when_fp3_selection_is_default_or_non_baseline_ml() -> None:
    fallback = _static_fp3_method()
    non_baseline = ChampionMethodConfig("boosted", "hist_gradient_boosting", "base_plus_relative")

    already_default = apply_stabilized_nested_guardrail(
        selected=fallback,
        fallback=fallback,
        checkpoint="after_fp3",
        settings=StabilizedNestedGuardedChampionConfig(),
    )
    ml_switch = apply_stabilized_nested_guardrail(
        selected=non_baseline,
        fallback=fallback,
        checkpoint="after_fp3",
        settings=StabilizedNestedGuardedChampionConfig(),
    )

    assert already_default.selected == fallback
    assert already_default.guardrail_applied is False
    assert ml_switch.selected == non_baseline
    assert ml_switch.guardrail_applied is False


def test_nested_selection_uses_only_prior_folds_and_ignores_current_event() -> None:
    candidates = pd.concat(
        [
            _candidate_rows(1, "event-1", "baseline", "method-a", [0.0, 0.1]),
            _candidate_rows(1, "event-1", "baseline", "method-b", [0.6, 0.7]),
            _candidate_rows(2, "event-2", "baseline", "method-a", [20.0, 20.0]),
            _candidate_rows(2, "event-2", "baseline", "method-b", [0.0, 0.0]),
        ],
        ignore_index=True,
    )

    selected, value, source_events, fallback = select_nested_method(
        candidates,
        fold_id=2,
        checkpoint="after_fp1",
        fallback=ChampionMethodConfig("robust_baseline", "fallback"),
    )

    assert selected.model_name == "method-a"
    assert value == pytest.approx(0.05)
    assert source_events == ["event-1"]
    assert fallback is False


def test_nested_selection_falls_back_without_prior_history() -> None:
    fallback = ChampionMethodConfig("robust_baseline", "robust_best_push_lap")
    selected, value, source_events, fallback_used = select_nested_method(
        _candidate_rows(1, "event-1", "baseline", "method-a", [0.1, 0.2]),
        fold_id=1,
        checkpoint="after_fp1",
        fallback=fallback,
    )

    assert selected == fallback
    assert value is None
    assert source_events == []
    assert fallback_used is True


def test_stabilized_nested_falls_back_when_prior_folds_are_too_few() -> None:
    fallback = ChampionMethodConfig("robust_baseline", "robust_best_push_lap")
    candidates = pd.concat(
        [
            _candidate_rows(1, "event-1", "baseline", "method-a", [0.1, 0.1]),
            _candidate_rows(2, "event-2", "baseline", "method-a", [0.1, 0.1]),
        ],
        ignore_index=True,
    )

    decision = select_stabilized_nested_method(
        candidates,
        fold_id=2,
        checkpoint="after_fp1",
        fallback=fallback,
        settings=StabilizedNestedChampionConfig(min_prior_folds=2, min_prior_predictions=1),
    )

    assert decision.selected == fallback
    assert decision.fallback_used is True
    assert decision.fallback_reason == "insufficient_history"
    assert decision.prior_folds_used == 1


def test_stabilized_nested_falls_back_when_prior_predictions_are_too_few() -> None:
    fallback = ChampionMethodConfig("robust_baseline", "robust_best_push_lap")
    candidates = pd.concat(
        [
            _candidate_rows(1, "event-1", "baseline", "method-a", [0.1, 0.1]),
            _candidate_rows(2, "event-2", "baseline", "method-a", [0.1, 0.1]),
        ],
        ignore_index=True,
    )

    decision = select_stabilized_nested_method(
        candidates,
        fold_id=2,
        checkpoint="after_fp1",
        fallback=fallback,
        settings=StabilizedNestedChampionConfig(min_prior_folds=1, min_prior_predictions=3),
    )

    assert decision.selected == fallback
    assert decision.fallback_reason == "insufficient_history"
    assert decision.prior_predictions_used == 2


def test_stabilized_nested_switches_only_when_margin_is_exceeded() -> None:
    fallback = ChampionMethodConfig("robust_baseline", "default")
    candidates = pd.concat(
        [
            _candidate_rows(1, "event-1", "robust_baseline", "default", [0.2, 0.2]),
            _candidate_rows(1, "event-1", "baseline", "method-a", [0.17, 0.17]),
            _candidate_rows(2, "event-2", "robust_baseline", "default", [0.2, 0.2]),
            _candidate_rows(2, "event-2", "baseline", "method-a", [0.17, 0.17]),
        ],
        ignore_index=True,
    )

    keep_default = select_stabilized_nested_method(
        candidates,
        fold_id=2,
        checkpoint="after_fp1",
        fallback=fallback,
        settings=StabilizedNestedChampionConfig(
            min_prior_folds=1,
            min_prior_predictions=1,
            improvement_margin_sec=0.05,
        ),
    )
    switch = select_stabilized_nested_method(
        candidates,
        fold_id=2,
        checkpoint="after_fp1",
        fallback=fallback,
        settings=StabilizedNestedChampionConfig(
            min_prior_folds=1,
            min_prior_predictions=1,
            improvement_margin_sec=0.02,
        ),
    )

    assert keep_default.selected == fallback
    assert keep_default.fallback_reason == "hysteresis_margin_not_met"
    assert switch.selected.model_name == "method-a"
    assert switch.fallback_used is False


def test_stabilized_nested_never_uses_current_test_event() -> None:
    fallback = ChampionMethodConfig("robust_baseline", "default")
    candidates = pd.concat(
        [
            _candidate_rows(1, "event-1", "robust_baseline", "default", [0.1, 0.1]),
            _candidate_rows(1, "event-1", "baseline", "method-a", [0.3, 0.3]),
            _candidate_rows(2, "event-2", "robust_baseline", "default", [10.0, 10.0]),
            _candidate_rows(2, "event-2", "baseline", "method-a", [0.0, 0.0]),
        ],
        ignore_index=True,
    )

    decision = select_stabilized_nested_method(
        candidates,
        fold_id=2,
        checkpoint="after_fp1",
        fallback=fallback,
        settings=StabilizedNestedChampionConfig(
            min_prior_folds=1,
            min_prior_predictions=1,
            improvement_margin_sec=0.0,
        ),
    )

    assert decision.selected == fallback
    assert decision.source_events == ["event-1"]


def test_season_aware_candidate_not_eligible_for_fp1_or_fp2() -> None:
    default = _static_fp3_method()
    candidates = _season_aware_candidates(prior_folds=5, candidate_error=0.0, default_error=0.2)
    fold = _season_aware_fold(fold_id=6, prior_same_season_events=5)

    for checkpoint in ("after_fp1", "after_fp2"):
        decision = select_season_aware_guarded_method(
            candidates,
            fold=fold,
            checkpoint=checkpoint,
            default_method=default,
            settings=_season_aware_settings(min_folds=5, min_predictions=10),
        )

        assert decision.selected == default
        assert decision.selection_reason == "not_applicable_checkpoint"
        assert decision.selected_candidate is False


def test_canonical_prior_evidence_comparator_uses_strictly_prior_aligned_rows() -> None:
    candidate = ChampionMethodConfig(
        "ablation",
        "random_forest",
        "base_plus_relative",
        "current_season_only_with_prior",
    )
    default = _static_fp3_method()
    candidates = pd.concat(
        [
            _season_aware_candidates(prior_folds=5, candidate_error=0.05, default_error=0.2),
            _season_aware_candidates(
                prior_folds=1,
                candidate_error=10.0,
                default_error=0.0,
                start_fold=6,
                event_prefix="current",
            ),
        ],
        ignore_index=True,
    )

    comparison = compare_prior_evidence(
        candidates,
        target_fold_id=6,
        checkpoint="after_fp3",
        candidate=candidate,
        default=default,
    )

    assert comparison.prior_fold_ids == [1, 2, 3, 4, 5]
    assert comparison.prior_rows_used == 10
    assert comparison.candidate_mae == pytest.approx(0.05)
    assert comparison.default_mae == pytest.approx(0.2)
    assert comparison.improvement_sec == pytest.approx(0.15)


def test_canonical_prior_evidence_comparator_counts_unmatched_rows() -> None:
    candidate = ChampionMethodConfig(
        "ablation",
        "random_forest",
        "base_plus_relative",
        "current_season_only_with_prior",
    )
    default = _static_fp3_method()
    candidates = _season_aware_candidates(prior_folds=5, candidate_error=0.05, default_error=0.2)
    default_ver = (
        candidates["temporal_weighting_policy"].eq("uniform")
        & candidates["fold_id"].eq(1)
        & candidates["driver"].eq("VER")
    )
    candidates = candidates[~default_ver].copy()

    comparison = compare_prior_evidence(
        candidates,
        target_fold_id=6,
        checkpoint="after_fp3",
        candidate=candidate,
        default=default,
    )

    assert comparison.candidate_rows_before_alignment == 10
    assert comparison.default_rows_before_alignment == 9
    assert comparison.prior_rows_used == 9
    assert comparison.dropped_candidate_rows == 1
    assert comparison.dropped_default_rows == 0


def test_season_aware_candidate_blocked_by_cold_start() -> None:
    default = _static_fp3_method()
    candidates = _season_aware_candidates(prior_folds=5, candidate_error=0.0, default_error=0.2)
    fold = _season_aware_fold(fold_id=6, prior_same_season_events=4)

    decision = select_season_aware_guarded_method(
        candidates,
        fold=fold,
        checkpoint="after_fp3",
        default_method=default,
        settings=_season_aware_settings(min_folds=5, min_predictions=10),
    )

    assert decision.selected == default
    assert decision.selection_reason == "cold_start"
    assert decision.fallback_reason == "season_aware_cold_start"


def test_season_aware_candidate_blocked_by_insufficient_history() -> None:
    default = _static_fp3_method()
    candidates = _season_aware_candidates(prior_folds=3, candidate_error=0.0, default_error=0.2)
    fold = _season_aware_fold(fold_id=6, prior_same_season_events=5)

    decision = select_season_aware_guarded_method(
        candidates,
        fold=fold,
        checkpoint="after_fp3",
        default_method=default,
        settings=_season_aware_settings(min_folds=5, min_predictions=10),
    )

    assert decision.selected == default
    assert decision.selection_reason == "insufficient_candidate_history"
    assert decision.prior_candidate_folds == 3


def test_season_aware_candidate_blocked_when_margin_not_met() -> None:
    default = _static_fp3_method()
    candidates = _season_aware_candidates(prior_folds=5, candidate_error=0.17, default_error=0.2)
    fold = _season_aware_fold(fold_id=6, prior_same_season_events=5)

    decision = select_season_aware_guarded_method(
        candidates,
        fold=fold,
        checkpoint="after_fp3",
        default_method=default,
        settings=_season_aware_settings(min_folds=5, min_predictions=10, margin=0.05),
    )

    assert decision.selected == default
    assert decision.selection_reason == "margin_not_met"
    assert decision.candidate_metric_value == pytest.approx(0.17)
    assert decision.metric_scope is not None
    assert decision.metric_scope.improvement_sec == pytest.approx(0.03)


def test_season_aware_candidate_selected_after_prior_evidence() -> None:
    default = _static_fp3_method()
    candidates = _season_aware_candidates(prior_folds=5, candidate_error=0.05, default_error=0.2)
    fold = _season_aware_fold(fold_id=6, prior_same_season_events=5)

    decision = select_season_aware_guarded_method(
        candidates,
        fold=fold,
        checkpoint="after_fp3",
        default_method=default,
        settings=_season_aware_settings(min_folds=5, min_predictions=10, margin=0.05),
    )

    assert decision.selected.temporal_weighting_policy == "current_season_only_with_prior"
    assert decision.selected_candidate is True
    assert decision.selection_reason == "selected_after_prior_evidence"
    assert decision.candidate_metric_value == pytest.approx(0.05)
    assert decision.default_metric_value == pytest.approx(0.2)
    assert decision.metric_scope is not None
    assert decision.metric_scope.prior_rows_used == 10


def test_season_aware_candidate_rejected_with_insufficient_aligned_comparator_history() -> None:
    default = _static_fp3_method()
    candidates = _season_aware_candidates(prior_folds=5, candidate_error=0.05, default_error=0.2)
    prior_default = candidates["temporal_weighting_policy"].eq("uniform") & candidates[
        "fold_id"
    ].lt(6)
    candidates = candidates[~prior_default].copy()
    fold = _season_aware_fold(fold_id=6, prior_same_season_events=5)

    decision = select_season_aware_guarded_method(
        candidates,
        fold=fold,
        checkpoint="after_fp3",
        default_method=default,
        settings=_season_aware_settings(min_folds=5, min_predictions=10, margin=0.05),
    )

    assert decision.selected == default
    assert decision.selection_reason == "insufficient_aligned_comparator_history"
    assert decision.metric_scope is not None
    assert decision.metric_scope.prior_rows_used == 0


def test_season_aware_candidate_uses_prior_folds_only() -> None:
    default = _static_fp3_method()
    candidates = pd.concat(
        [
            _season_aware_candidates(prior_folds=5, candidate_error=0.2, default_error=0.2),
            _season_aware_candidates(
                prior_folds=1,
                candidate_error=0.0,
                default_error=10.0,
                start_fold=6,
                event_prefix="current",
            ),
        ],
        ignore_index=True,
    )
    fold = _season_aware_fold(fold_id=6, prior_same_season_events=5)

    decision = select_season_aware_guarded_method(
        candidates,
        fold=fold,
        checkpoint="after_fp3",
        default_method=default,
        settings=_season_aware_settings(min_folds=5, min_predictions=10, margin=0.05),
    )

    assert decision.selected == default
    assert decision.selection_reason == "margin_not_met"
    assert decision.source_events == [f"2024/event-{fold_id}" for fold_id in range(1, 6)]
    assert "2024/current-6" not in decision.source_events


def test_season_aware_candidate_records_missing_weighted_artifact() -> None:
    default = _static_fp3_method()
    candidates = _season_aware_candidates(
        prior_folds=5,
        candidate_error=0.05,
        default_error=0.2,
        include_weighted=False,
    )
    fold = _season_aware_fold(fold_id=6, prior_same_season_events=5)

    decision = select_season_aware_guarded_method(
        candidates,
        fold=fold,
        checkpoint="after_fp3",
        default_method=default,
        settings=_season_aware_settings(min_folds=5, min_predictions=10, margin=0.05),
    )

    assert decision.selected == default
    assert decision.season_aware_candidate_available is False
    assert decision.selection_reason == "weighted_candidate_missing"


def test_uncertainty_uses_only_prior_residuals() -> None:
    candidates = pd.concat(
        [
            _candidate_rows(1, "event-1", "baseline", "method-a", [0.0, 2.0]),
            _candidate_rows(2, "event-2", "baseline", "method-a", [100.0, 100.0]),
        ],
        ignore_index=True,
    )
    predictions = _champion_rows(2, "method-a")

    result = add_prior_residual_uncertainty(
        predictions,
        candidates,
        UncertaintyConfig(interval_z=1.0, min_residual_count=2),
    )

    expected_std = pd.Series([0.0, -2.0]).std(ddof=1)
    assert result["residual_std_sec"].iloc[0] == pytest.approx(expected_std)
    assert set(result["uncertainty_method"]) == {"residual_std"}
    assert result["prediction_interval_low_sec"].notna().all()
    assert result["prediction_interval_high_sec"].notna().all()


def test_conformal_uncertainty_quantile_uses_only_prior_residuals() -> None:
    candidates = pd.concat(
        [
            _candidate_rows(1, "event-1", "baseline", "method-a", [0.0, 2.0]),
            _candidate_rows(2, "event-2", "baseline", "method-a", [100.0, 100.0]),
        ],
        ignore_index=True,
    )
    predictions = _champion_rows(2, "method-a")

    result = add_prior_residual_uncertainty(
        predictions,
        candidates,
        UncertaintyConfig(confidence_level=0.9, min_residual_count=2),
        method=ChampionUncertaintyMethod.conformal,
    )

    assert set(result["uncertainty_method"]) == {"conformal"}
    assert result["residual_quantile_sec"].iloc[0] == pytest.approx(2.0)
    assert result["residual_count"].iloc[0] == 2


def test_conformal_uncertainty_is_unavailable_with_insufficient_history() -> None:
    result = add_prior_residual_uncertainty(
        _champion_rows(1, "method-a"),
        _candidate_rows(1, "event-1", "baseline", "method-a", [0.1, 0.2]),
        UncertaintyConfig(confidence_level=0.9, min_residual_count=20),
        method=ChampionUncertaintyMethod.conformal,
    )

    assert result["prediction_interval_low_sec"].isna().all()
    assert set(result["uncertainty_method"]) == {"insufficient_history"}
    assert result["residual_count"].eq(0).all()


def test_uncertainty_is_null_when_history_is_insufficient() -> None:
    result = add_prior_residual_uncertainty(
        _champion_rows(1, "method-a"),
        _candidate_rows(1, "event-1", "baseline", "method-a", [0.1, 0.2]),
        UncertaintyConfig(interval_z=1.64, min_residual_count=20),
    )

    assert result["prediction_interval_low_sec"].isna().all()
    assert result["prediction_interval_high_sec"].isna().all()
    assert set(result["uncertainty_method"]) == {"insufficient_history"}


def test_predicted_gap_bucket_assignment_thresholds() -> None:
    settings = PredictedGapBucketUncertaintyConfig()

    assert assign_predicted_gap_bucket(0.5, settings) == "pole_contender"
    assert assign_predicted_gap_bucket(0.5001, settings) == "close_midfield"
    assert assign_predicted_gap_bucket(1.5, settings) == "close_midfield"
    assert assign_predicted_gap_bucket(1.5001, settings) == "midfield"
    assert assign_predicted_gap_bucket(3.0, settings) == "midfield"
    assert assign_predicted_gap_bucket(3.0001, settings) == "backmarker_or_outlier"


def test_predicted_gap_bucket_conformal_uses_predicted_gap_not_actual_gap() -> None:
    candidates = _bucket_candidates(
        fold_id=1,
        predicted=[0.2, 0.4],
        actual=[0.0, 0.0],
    )
    predictions = _bucket_predictions(
        fold_id=2,
        predicted=[0.4],
        actual=[10.0],
    )

    result = add_predicted_gap_bucket_conformal_uncertainty(
        predictions,
        candidates,
        _bucket_uncertainty(min_residual_count=2),
    )

    assert result["predicted_gap_bucket"].iloc[0] == "pole_contender"
    assert result["uncertainty_method"].iloc[0] == "conformal_predicted_gap_bucket"


def test_predicted_gap_bucket_conformal_uses_prior_folds_only() -> None:
    candidates = pd.concat(
        [
            _bucket_candidates(1, predicted=[0.3, 0.4], actual=[0.0, 0.0]),
            _bucket_candidates(2, predicted=[0.3, 0.4], actual=[100.0, 100.0]),
        ],
        ignore_index=True,
    )
    predictions = _bucket_predictions(2, predicted=[0.3], actual=[0.0])

    result = add_predicted_gap_bucket_conformal_uncertainty(
        predictions,
        candidates,
        _bucket_uncertainty(min_residual_count=2),
    )

    assert result["residual_quantile_sec"].iloc[0] == pytest.approx(0.4)
    assert result["uncertainty_prior_group_count"].iloc[0] == 2


def test_predicted_gap_bucket_conformal_uses_method_bucket_when_available() -> None:
    candidates = _bucket_candidates(1, predicted=[0.2, 0.4], actual=[0.0, 0.0])
    predictions = _bucket_predictions(2, predicted=[0.3], actual=[0.0])

    result = add_prior_residual_uncertainty(
        predictions,
        candidates,
        _bucket_uncertainty(min_residual_count=2),
        method=ChampionUncertaintyMethod.conformal_predicted_gap_bucket,
    )

    assert result["uncertainty_calibration_level"].iloc[0] == "checkpoint_method_bucket"
    assert bool(result["uncertainty_fallback_used"].iloc[0]) is False


def test_predicted_gap_bucket_conformal_falls_back_to_checkpoint_bucket() -> None:
    candidates = pd.concat(
        [
            _bucket_candidates(1, predicted=[0.2], actual=[0.0]),
            _bucket_candidates(
                1,
                predicted=[0.4, 0.45],
                actual=[0.0, 0.0],
                model_name="method-b",
            ),
        ],
        ignore_index=True,
    )
    predictions = _bucket_predictions(2, predicted=[0.3], actual=[0.0])

    result = add_predicted_gap_bucket_conformal_uncertainty(
        predictions,
        candidates,
        _bucket_uncertainty(min_residual_count=2),
    )

    assert result["uncertainty_calibration_level"].iloc[0] == "checkpoint_bucket"
    assert bool(result["uncertainty_fallback_used"].iloc[0]) is True
    assert result["uncertainty_prior_group_count"].iloc[0] == 3


def test_predicted_gap_bucket_conformal_is_null_with_insufficient_global_history() -> None:
    result = add_predicted_gap_bucket_conformal_uncertainty(
        _bucket_predictions(2, predicted=[0.3], actual=[0.0]),
        _bucket_candidates(1, predicted=[0.2], actual=[0.0]),
        _bucket_uncertainty(min_residual_count=2),
    )

    assert result["prediction_interval_low_sec"].isna().all()
    assert result["uncertainty_method"].iloc[0] == "insufficient_history"
    assert result["uncertainty_calibration_level"].iloc[0] == "insufficient_history"


def test_predicted_gap_bucket_metrics_are_included() -> None:
    champion = _champion_metric_rows([0.1, 0.2])
    champion["predicted_gap_bucket"] = ["pole_contender", "pole_contender"]
    champion["prediction_interval_low_sec"] = [-0.2, 0.15]
    champion["prediction_interval_high_sec"] = [0.2, 0.25]
    champion["interval_contains_actual"] = [True, False]
    candidates = _metric_candidate_rows("robust_baseline", "baseline", [0.3, 0.5])

    payload = build_champion_metrics_payload(
        BacktestStrategy.walk_forward,
        ChampionSelectionMode.nested,
        2,
        2,
        0,
        champion,
        candidates,
        uncertainty_method=ChampionUncertaintyMethod.conformal_predicted_gap_bucket,
    )

    bucket = payload["interval_metrics_by_predicted_gap_bucket"]["after_fp1"]["pole_contender"]
    assert bucket["rows_with_interval"] == 2
    assert bucket["coverage"] == pytest.approx(0.5)
    assert bucket["miss_count"] == 1


def test_champion_metrics_and_delta_sign() -> None:
    champion = _champion_metric_rows([0.1, 0.2])
    candidates = pd.concat(
        [
            _metric_candidate_rows("robust_baseline", "baseline", [0.3, 0.5]),
            _metric_candidate_rows("ablation", "random_forest", [0.2, 0.3]),
        ],
        ignore_index=True,
    )
    payload = build_champion_metrics_payload(
        BacktestStrategy.walk_forward,
        ChampionSelectionMode.nested,
        2,
        2,
        0,
        champion,
        candidates,
    )

    assert payload["metrics_by_checkpoint"]["after_fp1"]["mae_gap_sec"] == pytest.approx(0.15)
    assert payload["champion_vs_best_baseline_delta_mae"]["after_fp1"] < 0
    assert payload["champion_vs_best_single_family_delta_mae"]["after_fp1"] < 0


def test_champion_interval_metrics_are_computed() -> None:
    champion = _champion_metric_rows([0.1, 0.2])
    champion["prediction_interval_low_sec"] = [-0.2, 0.15]
    champion["prediction_interval_high_sec"] = [0.2, 0.25]
    champion["interval_contains_actual"] = [True, False]
    candidates = _metric_candidate_rows("robust_baseline", "baseline", [0.3, 0.5])

    payload = build_champion_metrics_payload(
        BacktestStrategy.walk_forward,
        ChampionSelectionMode.nested,
        2,
        2,
        0,
        champion,
        candidates,
    )
    metrics = payload["metrics_by_checkpoint"]["after_fp1"]

    assert metrics["interval_coverage"] == pytest.approx(0.5)
    assert metrics["mean_interval_width_sec"] == pytest.approx(0.25)
    assert metrics["median_interval_width_sec"] == pytest.approx(0.25)
    assert metrics["interval_availability_rate"] == pytest.approx(1.0)


def test_static_champion_backtest_writes_prediction_and_selection_schemas(
    tmp_path: Path,
) -> None:
    dataset = _dataset(6)
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    config = _config(tmp_path)
    folds = build_backtest_folds(dataset, BacktestStrategy.walk_forward, min_train_events=3)
    _write_prediction_artifacts(config.metrics_output_dir, dataset, folds)

    summary = run_champion_backtest(
        config,
        selection_mode=ChampionSelectionMode.static,
        dataset_path=dataset_path,
        min_events=5,
        min_train_events=3,
        model_config=_model_config(),
    )

    predictions = pd.read_parquet(summary.predictions_path)
    selection = pd.read_parquet(summary.selection_path)
    prediction_columns = {
        "selected_family",
        "selected_model_name",
        "selected_feature_group",
        "residual_count",
        "residual_quantile_sec",
        "prediction_interval_low_sec",
        "prediction_interval_high_sec",
        "interval_contains_actual",
        "uncertainty_method",
    }
    assert summary.n_folds_successful == len(folds)
    assert prediction_columns <= set(predictions.columns)
    assert set(selection["checkpoint"]) == {"after_fp1", "after_fp2", "after_fp3"}
    assert {
        "fallback_reason",
        "prior_folds_used",
        "prior_predictions_used",
        "default_model_name",
        "guardrail_applied",
        "pre_guardrail_selected_model_name",
        "post_guardrail_selected_model_name",
    } <= set(selection.columns)


def test_champion_mode_specific_outputs_coexist(tmp_path: Path) -> None:
    dataset = _dataset(6)
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    config = _config(tmp_path)
    folds = build_backtest_folds(dataset, BacktestStrategy.walk_forward, min_train_events=3)
    _write_prediction_artifacts(config.metrics_output_dir, dataset, folds)

    for mode in (
        ChampionSelectionMode.static,
        ChampionSelectionMode.nested,
        ChampionSelectionMode.stabilized_nested,
        ChampionSelectionMode.stabilized_nested_guarded,
        ChampionSelectionMode.season_aware_nested_guarded,
    ):
        run_champion_backtest(
            config,
            selection_mode=mode,
            dataset_path=dataset_path,
            min_events=5,
            min_train_events=3,
            model_config=_model_config(),
        )

    assert (config.metrics_output_dir / "champion_static_metrics.json").is_file()
    assert (config.metrics_output_dir / "champion_nested_metrics.json").is_file()
    assert (config.metrics_output_dir / "champion_stabilized_nested_metrics.json").is_file()
    assert (config.metrics_output_dir / "champion_stabilized_nested_guarded_metrics.json").is_file()
    assert (
        config.metrics_output_dir / "champion_season_aware_nested_guarded_metrics.json"
    ).is_file()


def test_guarded_champion_backtest_records_guardrail_metadata(tmp_path: Path) -> None:
    dataset = _dataset(8)
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    config = _config(tmp_path)
    folds = build_backtest_folds(dataset, BacktestStrategy.walk_forward, min_train_events=3)
    _write_prediction_artifacts(
        config.metrics_output_dir,
        dataset,
        folds,
        include_best_valid_switch=True,
    )

    summary = run_champion_backtest(
        config,
        selection_mode=ChampionSelectionMode.stabilized_nested_guarded,
        dataset_path=dataset_path,
        min_events=5,
        min_train_events=3,
        model_config=_guarded_model_config(),
    )

    predictions = pd.read_parquet(summary.predictions_path)
    selection = pd.read_parquet(summary.selection_path)
    guarded = selection[selection["guardrail_applied"].astype(bool)]

    assert (config.metrics_output_dir / "champion_stabilized_nested_guarded_metrics.json").is_file()
    assert not guarded.empty
    assert set(guarded["checkpoint"]) == {"after_fp3"}
    assert set(guarded["guardrail_name"]) == {"fp3_no_baseline_switch"}
    assert set(guarded["guardrail_reason"]) == {"prevent_fp3_baseline_switch_from_static_rf"}
    assert set(guarded["pre_guardrail_selected_model_name"]) == {"best_valid_lap"}
    assert set(guarded["post_guardrail_selected_model_name"]) == {"random_forest"}
    fp3_predictions = predictions[predictions["checkpoint"].eq("after_fp3")]
    assert set(fp3_predictions["selected_model_name"]) == {"random_forest"}


def test_season_aware_champion_backtest_records_metadata_and_keeps_guardrail(
    tmp_path: Path,
) -> None:
    dataset = _dataset(8)
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    config = _config(tmp_path)
    folds = build_backtest_folds(dataset, BacktestStrategy.walk_forward, min_train_events=3)
    _write_prediction_artifacts(
        config.metrics_output_dir,
        dataset,
        folds,
        include_best_valid_switch=True,
    )

    summary = run_champion_backtest(
        config,
        selection_mode=ChampionSelectionMode.season_aware_nested_guarded,
        dataset_path=dataset_path,
        min_events=5,
        min_train_events=3,
        model_config=_guarded_model_config(),
    )

    predictions = pd.read_parquet(summary.predictions_path)
    selection = pd.read_parquet(summary.selection_path)
    fp3_selection = selection[selection["checkpoint"].eq("after_fp3")]

    assert (
        config.metrics_output_dir / "champion_season_aware_nested_guarded_metrics.json"
    ).is_file()
    assert set(predictions[predictions["checkpoint"].eq("after_fp3")]["selected_model_name"]) == {
        "random_forest"
    }
    assert "season_aware_selection_reason" in selection.columns
    assert "weighted_candidate_missing" in set(fp3_selection["season_aware_selection_reason"])
    assert fp3_selection["guardrail_applied"].astype(bool).any()


def test_champion_backtest_skips_when_dataset_is_too_small(tmp_path: Path) -> None:
    dataset_path = tmp_path / "small.parquet"
    _dataset(3).to_parquet(dataset_path, index=False)

    summary = run_champion_backtest(
        _config(tmp_path),
        dataset_path=dataset_path,
        min_events=5,
        model_config=_model_config(),
    )
    metrics = json.loads(summary.metrics_path.read_text(encoding="utf-8"))

    assert summary.status == "skipped"
    assert metrics["status"] == "skipped"


def _candidate_rows(
    fold_id: int,
    event: str,
    family: str,
    model_name: str,
    predicted: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": [fold_id, fold_id],
            "test_event": [event, event],
            "checkpoint": ["after_fp1", "after_fp1"],
            "candidate_family": [family, family],
            "model_name": [model_name, model_name],
            "feature_group": pd.Series([pd.NA, pd.NA], dtype="string"),
            "driver": ["NOR", "VER"],
            "quali_gap_to_pole_sec": [0.0, 0.0],
            "predicted_quali_gap_to_pole_sec": predicted,
        }
    )


def _champion_rows(fold_id: int, model_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": [fold_id, fold_id],
            "checkpoint": ["after_fp1", "after_fp1"],
            "selected_family": ["baseline", "baseline"],
            "selected_model_name": [model_name, model_name],
            "selected_feature_group": pd.Series([pd.NA, pd.NA], dtype="string"),
            "quali_gap_to_pole_sec": [0.0, 0.0],
            "predicted_quali_gap_to_pole_sec": [0.4, 0.6],
        }
    )


def _bucket_candidates(
    fold_id: int,
    *,
    predicted: list[float],
    actual: list[float],
    model_name: str = "method-a",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": [fold_id] * len(predicted),
            "test_event": [f"event-{fold_id}"] * len(predicted),
            "checkpoint": ["after_fp3"] * len(predicted),
            "candidate_family": ["baseline"] * len(predicted),
            "model_name": [model_name] * len(predicted),
            "feature_group": pd.Series([pd.NA] * len(predicted), dtype="string"),
            "driver": [f"DRV{i}" for i in range(len(predicted))],
            "quali_gap_to_pole_sec": actual,
            "predicted_quali_gap_to_pole_sec": predicted,
        }
    )


def _bucket_predictions(
    fold_id: int,
    *,
    predicted: list[float],
    actual: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": [fold_id] * len(predicted),
            "checkpoint": ["after_fp3"] * len(predicted),
            "selected_family": ["baseline"] * len(predicted),
            "selected_model_name": ["method-a"] * len(predicted),
            "selected_feature_group": pd.Series([pd.NA] * len(predicted), dtype="string"),
            "quali_gap_to_pole_sec": actual,
            "predicted_quali_gap_to_pole_sec": predicted,
        }
    )


def _bucket_uncertainty(min_residual_count: int) -> UncertaintyConfig:
    return UncertaintyConfig(
        predicted_gap_bucket=PredictedGapBucketUncertaintyConfig(
            min_residual_count=min_residual_count
        )
    )


def _season_aware_candidates(
    *,
    prior_folds: int,
    candidate_error: float,
    default_error: float,
    start_fold: int = 1,
    event_prefix: str = "event",
    include_weighted: bool = True,
) -> pd.DataFrame:
    frames = []
    fold_ids = [start_fold + offset for offset in range(prior_folds)]
    if 6 not in fold_ids:
        fold_ids.append(6)
    for fold_id in fold_ids:
        event = f"2024/{event_prefix}-{fold_id}"
        for temporal_policy, error in (
            ("uniform", default_error),
            ("current_season_only_with_prior", candidate_error),
        ):
            if temporal_policy != "uniform" and not include_weighted:
                continue
            frame = pd.DataFrame(
                {
                    "fold_id": [fold_id, fold_id],
                    "test_event": [event, event],
                    "season": [2024, 2024],
                    "event": [event, event],
                    "event_slug": [f"{event_prefix}-{fold_id}", f"{event_prefix}-{fold_id}"],
                    "checkpoint": ["after_fp3", "after_fp3"],
                    "driver": ["NOR", "VER"],
                    "team": ["McLaren", "Red Bull"],
                    "quali_position": [1, 2],
                    "quali_gap_to_pole_sec": [0.0, 1.0],
                    "reached_q3": [1, 1],
                    "predicted_quali_gap_to_pole_sec": [error, 1.0 + error],
                    "predicted_quali_position": [1, 2],
                    "predicted_reached_q3": [1, 1],
                    "candidate_family": ["ablation", "ablation"],
                    "model_name": ["random_forest", "random_forest"],
                    "feature_group": ["base_plus_relative", "base_plus_relative"],
                    "temporal_weighting_policy": [temporal_policy, temporal_policy],
                }
            )
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _season_aware_fold(fold_id: int, prior_same_season_events: int) -> BacktestFold:
    return BacktestFold(
        fold_id=fold_id,
        strategy="walk_forward",
        test_event=f"2024/event-{fold_id}",
        train_events=tuple(
            f"2024/event-{index}" for index in range(1, prior_same_season_events + 1)
        ),
        train_rows=prior_same_season_events * 6,
        test_rows=6,
    )


def _season_aware_settings(
    *,
    min_folds: int,
    min_predictions: int,
    margin: float = 0.05,
) -> SeasonAwareNestedGuardedChampionConfig:
    return SeasonAwareNestedGuardedChampionConfig(
        min_current_season_prior_events=5,
        min_prior_candidate_folds=min_folds,
        min_prior_candidate_predictions=min_predictions,
        improvement_margin_sec=margin,
    )


def _champion_metric_rows(predicted: list[float]) -> pd.DataFrame:
    frame = _metric_rows(predicted)
    frame["fold_id"] = [1, 2]
    return frame


def _metric_candidate_rows(
    family: str,
    model_name: str,
    predicted: list[float],
) -> pd.DataFrame:
    frame = _metric_rows(predicted)
    frame["fold_id"] = [1, 2]
    frame["candidate_family"] = family
    frame["model_name"] = model_name
    frame["feature_group"] = pd.Series([pd.NA, pd.NA], dtype="string")
    return frame


def _metric_rows(predicted: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024, 2024],
            "event_slug": ["event-1", "event-2"],
            "checkpoint": ["after_fp1", "after_fp1"],
            "driver": ["NOR", "NOR"],
            "quali_gap_to_pole_sec": [0.0, 0.0],
            "predicted_quali_gap_to_pole_sec": predicted,
            "quali_position": [1, 1],
            "predicted_quali_position": [1, 1],
            "reached_q3": [1, 1],
            "predicted_reached_q3": [1, 1],
        }
    )


def _dataset(n_events: int) -> pd.DataFrame:
    rows = []
    for event_index in range(1, n_events + 1):
        for checkpoint in ("after_fp1", "after_fp2", "after_fp3"):
            for position, driver in enumerate(("NOR", "VER"), start=1):
                rows.append(
                    {
                        "season": 2024,
                        "event": f"Event {event_index}",
                        "event_slug": f"event-{event_index}",
                        "event_order": event_index,
                        "checkpoint": checkpoint,
                        "driver": driver,
                        "team": f"Team {position}",
                        "quali_position": position,
                        "quali_gap_to_pole_sec": float(position - 1),
                        "reached_q3": 1,
                    }
                )
    return pd.DataFrame(rows)


def _write_prediction_artifacts(
    metrics_dir: Path,
    dataset: pd.DataFrame,
    folds,
    *,
    include_best_valid_switch: bool = False,
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    event_to_fold = {fold.test_event: fold.fold_id for fold in folds}
    event_keys = dataset["season"].astype(str) + "/" + dataset["event_slug"]
    test = dataset[event_keys.isin(event_to_fold)].copy()
    test["test_event"] = event_keys[test.index]
    test["fold_id"] = test["test_event"].map(event_to_fold)
    base_columns = list(test.columns)
    baseline_frames = []
    baseline_names = ["robust_best_push_lap", "robust_theoretical_best_lap"]
    if include_best_valid_switch:
        baseline_names.append("best_valid_lap")
    for model_name in baseline_names:
        frame = test.loc[:, base_columns].copy()
        frame["model_name"] = model_name
        frame["baseline_name"] = model_name
        frame["prediction_type"] = "baseline"
        frame["strategy"] = "walk_forward"
        if model_name == "best_valid_lap":
            frame["predicted_quali_gap_to_pole_sec"] = frame["quali_gap_to_pole_sec"]
        else:
            frame["predicted_quali_gap_to_pole_sec"] = frame["quali_gap_to_pole_sec"] + 0.2
        frame["predicted_quali_position"] = frame["quali_position"]
        frame["predicted_reached_q3"] = 1
        baseline_frames.append(frame)
    pd.concat(baseline_frames, ignore_index=True).to_parquet(
        metrics_dir / "walk_forward_predictions.parquet", index=False
    )
    ablation = test.copy()
    ablation["model_name"] = "random_forest"
    ablation["feature_group"] = "base_plus_relative"
    ablation["prediction_type"] = "tabular"
    ablation["strategy"] = "walk_forward"
    ablation["predicted_quali_gap_to_pole_sec"] = ablation["quali_gap_to_pole_sec"] + 0.1
    ablation["predicted_quali_position"] = ablation["quali_position"]
    ablation["predicted_reached_q3"] = 1
    ablation.to_parquet(metrics_dir / "ablation_predictions.parquet", index=False)


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


def _static_fp3_method() -> ChampionMethodConfig:
    return ChampionMethodConfig(
        family="ablation",
        model_name="random_forest",
        feature_group="base_plus_relative",
    )


def _model_config() -> ModelConfig:
    return ModelConfig(
        min_events=5,
        random_state=42,
        ridge_alpha=1.0,
        random_forest=RandomForestConfig(5, 3, 1),
        champion_policy=ChampionPolicyConfig(),
        uncertainty=UncertaintyConfig(interval_z=1.64, min_residual_count=2),
    )


def _guarded_model_config() -> ModelConfig:
    return ModelConfig(
        min_events=5,
        random_state=42,
        ridge_alpha=1.0,
        random_forest=RandomForestConfig(5, 3, 1),
        champion_policy=ChampionPolicyConfig(
            stabilized_nested=StabilizedNestedChampionConfig(
                min_prior_folds=1,
                min_prior_predictions=2,
                improvement_margin_sec=0.0,
            ),
            stabilized_nested_guarded=StabilizedNestedGuardedChampionConfig(),
        ),
        uncertainty=UncertaintyConfig(interval_z=1.64, min_residual_count=2),
    )
