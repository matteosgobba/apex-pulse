import json
from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.config import DataConfig
from f1_prediction.modeling.season_aware_policy_forensics import (
    build_policy_event_counterfactual,
    build_policy_fold_reconstruction,
    build_policy_switch_cases,
    build_selected_fold_analysis,
    create_season_aware_policy_forensics_report,
    generate_policy_forensics_figures,
    simulate_prior_only_guardrails,
)


def test_reconstruction_uses_selected_and_default_sources() -> None:
    static, default, weighted, live, selection = _forensic_frames()

    reconstruction = build_policy_fold_reconstruction(
        season_aware_predictions=live,
        season_aware_selection=selection,
        weighted_candidate_predictions=weighted,
        default_predictions=default,
    )

    assert reconstruction["reconstruction_status"].tolist() == ["matched", "matched", "matched"]
    assert reconstruction.loc[reconstruction["fold_id"].eq(2), "candidate_source"].iloc[0] == (
        "weighted_candidate"
    )
    assert reconstruction.loc[reconstruction["fold_id"].eq(3), "candidate_source"].iloc[0] == (
        "default_guarded"
    )


def test_reconstruction_detects_row_count_mismatch() -> None:
    _, default, weighted, live, selection = _forensic_frames()
    weighted = weighted[~((weighted["fold_id"].eq(2)) & (weighted["driver"].eq("PIA")))]

    reconstruction = build_policy_fold_reconstruction(
        season_aware_predictions=live,
        season_aware_selection=selection,
        weighted_candidate_predictions=weighted,
        default_predictions=default,
    )

    row = reconstruction[reconstruction["fold_id"].eq(2)].iloc[0]
    assert row["reconstruction_status"] == "row_count_mismatch"
    assert not bool(row["row_count_match"])


def test_reconstruction_detects_prediction_mismatch() -> None:
    _, default, weighted, live, selection = _forensic_frames()
    live = live.copy()
    mismatch = live["fold_id"].eq(2) & live["driver"].eq("NOR")
    live.loc[mismatch, "predicted_quali_gap_to_pole_sec"] = 5.0

    reconstruction = build_policy_fold_reconstruction(
        season_aware_predictions=live,
        season_aware_selection=selection,
        weighted_candidate_predictions=weighted,
        default_predictions=default,
    )

    row = reconstruction[reconstruction["fold_id"].eq(2)].iloc[0]
    assert row["reconstruction_status"] == "prediction_mismatch"
    assert row["max_abs_prediction_difference"] > 0


def test_aggregate_mae_reconstruction_matches_saved_predictions() -> None:
    _, default, weighted, live, selection = _forensic_frames()

    reconstruction = build_policy_fold_reconstruction(
        season_aware_predictions=live,
        season_aware_selection=selection,
        weighted_candidate_predictions=weighted,
        default_predictions=default,
    )

    weighted_saved = (
        reconstruction["saved_fold_mae_gap_sec"] * reconstruction["saved_prediction_rows"]
    ).sum() / reconstruction["saved_prediction_rows"].sum()
    weighted_reconstructed = (
        reconstruction["reconstructed_fold_mae_gap_sec"]
        * reconstruction["reconstructed_prediction_rows"]
    ).sum() / reconstruction["reconstructed_prediction_rows"].sum()
    assert weighted_saved == pytest.approx(weighted_reconstructed)


def test_counterfactual_event_level_mae_deltas_are_computed() -> None:
    static, default, weighted, live, selection = _forensic_frames()

    table = build_policy_event_counterfactual(
        static_predictions=static,
        guarded_predictions=default,
        season_aware_predictions=live,
        weighted_candidate_predictions=weighted,
        default_candidate_predictions=default,
        season_aware_selection=selection,
    )

    selected = table[table["fold_id"].eq(2)].iloc[0]
    assert selected["weighted_candidate_mae_gap_sec"] == pytest.approx(0.05)
    assert selected["static_mae_gap_sec"] == pytest.approx(0.2)
    assert selected["delta_live_vs_static_sec"] == pytest.approx(-0.15)


def test_harmful_and_beneficial_switch_labels_are_assigned() -> None:
    static = _predictions([1], static_error=0.2)
    live = _predictions([1], static_error=0.0)
    live.loc[live["driver"].eq("NOR"), "predicted_quali_gap_to_pole_sec"] = 1.5
    selection = _selection([1], selected={1}, prior_improvements={1: 0.1})

    cases = build_policy_switch_cases(static, live, selection, tolerance_sec=0.05)

    assert bool(cases[cases["driver"].eq("NOR")]["harmful_switch"].iloc[0])
    assert bool(cases[cases["driver"].eq("PIA")]["beneficial_switch"].iloc[0])


def test_selected_fold_analysis_separates_fold_categories() -> None:
    static, default, weighted, live, selection = _forensic_frames()
    events = build_policy_event_counterfactual(
        static_predictions=static,
        guarded_predictions=default,
        season_aware_predictions=live,
        weighted_candidate_predictions=weighted,
        default_candidate_predictions=default,
        season_aware_selection=selection,
    )
    analysis, _ = build_selected_fold_analysis(
        events,
        static_predictions=static,
        season_aware_predictions=live,
        season_aware_selection=selection,
        tolerance_sec=0.05,
    )

    assert set(analysis["fold_category"]) == {
        "history_blocked",
        "selected_weighted_candidate",
        "margin_blocked",
    }


def test_higher_margin_guardrail_blocks_candidates_below_margin() -> None:
    static, default, weighted, live, selection = _forensic_frames()
    events = build_policy_event_counterfactual(
        static_predictions=static,
        guarded_predictions=default,
        season_aware_predictions=live,
        weighted_candidate_predictions=weighted,
        default_candidate_predictions=default,
        season_aware_selection=selection,
    )

    event_level, summary = simulate_prior_only_guardrails(events)

    high = event_level[
        event_level["policy_name"].eq("higher_margin_guardrail_0.15") & event_level["fold_id"].eq(2)
    ].iloc[0]
    assert not bool(high["selected_weighted_candidate"])
    live = summary[summary["policy_name"].eq("current_live_policy")].iloc[0]
    assert live["selected_event_count"] == 1


def test_prior_stability_guardrail_uses_strictly_prior_events() -> None:
    events = _guardrail_history_events()

    event_level, _ = simulate_prior_only_guardrails(events)

    fold_3 = event_level[
        event_level["policy_name"].eq("prior_stability_guardrail") & event_level["fold_id"].eq(3)
    ].iloc[0]
    fold_4 = event_level[
        event_level["policy_name"].eq("prior_stability_guardrail") & event_level["fold_id"].eq(4)
    ].iloc[0]
    assert not bool(fold_3["selected_weighted_candidate"])
    assert bool(fold_4["selected_weighted_candidate"])


def test_recent_prior_events_guardrail_uses_latest_requested_events() -> None:
    events = _guardrail_history_events()
    events.loc[events["fold_id"].eq(4), "delta_weighted_candidate_vs_default_sec"] = 0.4

    event_level, _ = simulate_prior_only_guardrails(events)

    recent_3 = event_level[
        event_level["policy_name"].eq("recent_prior_events_guardrail_3")
        & event_level["fold_id"].eq(5)
    ].iloc[0]
    recent_5 = event_level[
        event_level["policy_name"].eq("recent_prior_events_guardrail_5")
        & event_level["fold_id"].eq(5)
    ].iloc[0]
    assert not bool(recent_3["selected_weighted_candidate"])
    assert not bool(recent_5["selected_weighted_candidate"])


def test_guardrail_bootstrap_is_deterministic() -> None:
    events = _guardrail_history_events()

    _, first = simulate_prior_only_guardrails(events)
    _, second = simulate_prior_only_guardrails(events)

    pd.testing.assert_frame_equal(first, second)


def test_missing_artifact_handling_is_graceful(tmp_path: Path) -> None:
    config = _config(tmp_path)

    summary = create_season_aware_policy_forensics_report(config)
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))

    assert summary.status == "partial"
    assert "champion_static_predictions.parquet" in payload["missing_inputs"]
    assert (config.metrics_output_dir / "season_aware_policy_fold_reconstruction.csv").is_file()


def test_figure_generation_does_not_crash_on_minimal_valid_inputs(tmp_path: Path) -> None:
    static, default, weighted, live, selection = _forensic_frames()
    reconstruction = build_policy_fold_reconstruction(
        season_aware_predictions=live,
        season_aware_selection=selection,
        weighted_candidate_predictions=weighted,
        default_predictions=default,
    )
    events = build_policy_event_counterfactual(
        static_predictions=static,
        guarded_predictions=default,
        season_aware_predictions=live,
        weighted_candidate_predictions=weighted,
        default_candidate_predictions=default,
        season_aware_selection=selection,
    )
    analysis, _ = build_selected_fold_analysis(
        events,
        static_predictions=static,
        season_aware_predictions=live,
        season_aware_selection=selection,
        tolerance_sec=0.05,
    )
    _, guardrail = simulate_prior_only_guardrails(events)

    paths, issues = generate_policy_forensics_figures(
        figures_dir=tmp_path,
        event_counterfactual=events,
        selected_analysis=analysis,
        reconstruction=reconstruction,
        guardrail_summary=guardrail,
    )

    assert isinstance(issues, list)
    assert all(path.is_file() for path in paths)


def test_existing_champion_policy_behavior_is_unchanged() -> None:
    static, default, weighted, live, selection = _forensic_frames()
    before = live.copy(deep=True)

    build_policy_fold_reconstruction(
        season_aware_predictions=live,
        season_aware_selection=selection,
        weighted_candidate_predictions=weighted,
        default_predictions=default,
    )

    pd.testing.assert_frame_equal(live, before)
    assert not static.empty


def _forensic_frames() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    folds = [1, 2, 3]
    static = _predictions(folds, static_error=0.2)
    default = _predictions(folds, static_error=0.2)
    weighted = _predictions(folds, static_error=0.05)
    live = pd.concat(
        [
            default[default["fold_id"].isin([1, 3])],
            weighted[weighted["fold_id"].eq(2)],
        ],
        ignore_index=True,
    )
    selection = _selection(
        folds,
        selected={2},
        reasons={
            1: "insufficient_candidate_history",
            2: "selected_after_prior_evidence",
            3: "margin_not_met",
        },
        prior_improvements={1: None, 2: 0.10, 3: 0.02},
    )
    return static, default, weighted, live, selection


def _predictions(folds: list[int], *, static_error: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_id in folds:
        for driver, actual in (("NOR", 1.0), ("PIA", 1.4)):
            rows.append(
                {
                    "fold_id": fold_id,
                    "season": 2025,
                    "event": f"Event {fold_id}",
                    "event_slug": f"event_{fold_id}",
                    "checkpoint": "after_fp3",
                    "driver": driver,
                    "team": "McLaren",
                    "quali_gap_to_pole_sec": actual,
                    "predicted_quali_gap_to_pole_sec": actual + static_error,
                    "model_name": "random_forest",
                    "feature_group": "base_plus_relative",
                }
            )
    return pd.DataFrame(rows)


def _selection(
    folds: list[int],
    *,
    selected: set[int],
    reasons: dict[int, str] | None = None,
    prior_improvements: dict[int, float | None] | None = None,
) -> pd.DataFrame:
    reasons = reasons or {}
    prior_improvements = prior_improvements or {}
    rows: list[dict[str, object]] = []
    for fold_id in folds:
        rows.append(
            {
                "fold_id": fold_id,
                "season": 2025,
                "event": f"Event {fold_id}",
                "event_slug": f"event_{fold_id}",
                "checkpoint": "after_fp3",
                "current_season_prior_event_count": 6,
                "season_aware_selected": fold_id in selected,
                "season_aware_selection_reason": reasons.get(
                    fold_id,
                    "selected_after_prior_evidence" if fold_id in selected else "margin_not_met",
                ),
                "season_aware_candidate_prior_folds": 5,
                "season_aware_candidate_prior_predictions": 100,
                "metric_scope_candidate_mae": 0.5,
                "metric_scope_default_mae": (
                    0.5 + prior_improvements[fold_id]
                    if prior_improvements.get(fold_id) is not None
                    else pd.NA
                ),
                "metric_scope_improvement_sec": prior_improvements.get(fold_id),
                "guardrail_applied": False,
                "guardrail_reason": pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _guardrail_history_events() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    deltas = [-0.10, -0.08, -0.06, -0.07, -0.09]
    for fold_id, delta in enumerate(deltas, start=1):
        rows.append(
            {
                "fold_id": fold_id,
                "season": 2025,
                "event": f"Event {fold_id}",
                "event_slug": f"event_{fold_id}",
                "checkpoint": "after_fp3",
                "rows": 2,
                "static_mae_gap_sec": 0.5,
                "guarded_mae_gap_sec": 0.5,
                "season_aware_live_mae_gap_sec": 0.5,
                "weighted_candidate_mae_gap_sec": 0.4,
                "default_candidate_mae_gap_sec": 0.5,
                "delta_live_vs_static_sec": 0.0,
                "delta_weighted_candidate_vs_static_sec": -0.1,
                "delta_live_vs_guarded_sec": 0.0,
                "delta_weighted_candidate_vs_guarded_sec": -0.1,
                "delta_weighted_candidate_vs_default_sec": delta,
                "season_aware_selected": False,
                "selection_reason": "margin_not_met",
                "cold_start_regime": "established_season",
                "current_season_prior_event_count": 10,
                "candidate_prior_folds": 5,
                "candidate_prior_predictions": 100,
                "candidate_prior_mae": 0.4,
                "default_prior_mae": 0.5,
                "prior_improvement_sec": 0.1,
                "guardrail_applied": False,
                "guardrail_reason": pd.NA,
            }
        )
    return pd.DataFrame(rows)


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
