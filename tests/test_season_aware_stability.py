import json
from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.config import DataConfig
from f1_prediction.modeling.backtest_report import build_backtest_report_payload
from f1_prediction.modeling.season_aware_stability import (
    build_event_concentration,
    build_regime_stability,
    build_season_stability,
    build_tail_risk,
    canonical_event_rows,
    canonical_row_errors,
    create_season_aware_stability_report,
    validate_stability_identity,
)


def test_stability_report_generates_expected_artifacts_and_figures(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_stability_artifacts(config.metrics_output_dir)

    summary = create_season_aware_stability_report(config)
    payload = json.loads(summary.summary_path.read_text())

    assert summary.status == "partial"
    assert payload["candidate_identity_validation_status"] == "valid"
    assert payload["default_identity_validation_status"] == "valid"
    assert len(summary.table_paths) == 10
    assert len(summary.figure_paths) == 6
    for path in summary.figure_paths:
        assert path.is_file()


def test_canonical_identity_validation_accepts_valid_candidate_and_default(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_stability_artifacts(config.metrics_output_dir)
    artifacts = {
        "season_aware_event_level_comparison": pd.read_csv(
            config.metrics_output_dir / "season_aware_event_level_comparison.csv"
        ),
        "ablation_current_season_only_with_prior_predictions": pd.read_parquet(
            config.metrics_output_dir
            / "ablation_current_season_only_with_prior_predictions.parquet"
        ),
        "ablation_uniform_predictions": pd.read_parquet(
            config.metrics_output_dir / "ablation_uniform_predictions.parquet"
        ),
    }

    identity = validate_stability_identity(artifacts)

    assert identity["identity_valid"].all()


def test_identity_mismatch_blocks_invalid_aggregation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_stability_artifacts(config.metrics_output_dir, feature_group="wrong_group")

    summary = create_season_aware_stability_report(config)
    payload = json.loads(summary.summary_path.read_text())
    season = pd.read_csv(config.metrics_output_dir / "season_aware_stability_by_season.csv")

    assert payload["comparison_scope_status"] == "identity_or_scope_invalid"
    assert payload["overall_stability_classification"] == "identity_or_scope_invalid"
    assert season.empty


def test_season_level_mae_and_delta_calculation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_stability_artifacts(config.metrics_output_dir)
    event_rows, row_errors = _canonical_frames(config.metrics_output_dir)

    season = build_season_stability(event_rows, row_errors)
    row_2024 = season[season["season"].eq(2024)].iloc[0]

    assert row_2024["event_count"] == 2
    assert row_2024["candidate_mae"] == pytest.approx(0.2)
    assert row_2024["default_mae"] == pytest.approx(0.85)
    assert row_2024["delta_mae"] == pytest.approx(-0.65)


def test_event_order_output_preserves_supplied_chronology(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_stability_artifacts(config.metrics_output_dir)

    create_season_aware_stability_report(config)
    by_event = pd.read_csv(config.metrics_output_dir / "season_aware_stability_by_event.csv")
    season_2024 = by_event[by_event["season"].eq(2024)]

    assert list(season_2024["event_slug"]) == ["bahrain", "zandvoort"]
    assert list(season_2024["event_order"]) == [0, 6]


def test_regime_analysis_separates_cold_start_and_established_rows(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_stability_artifacts(config.metrics_output_dir)
    event_rows, row_errors = _canonical_frames(config.metrics_output_dir)

    regime = build_regime_stability(event_rows, row_errors)

    cold = regime[regime["regime"].eq("cold_start")].iloc[0]
    established = regime[regime["regime"].eq("established_season")].iloc[0]
    assert cold["delta_mae"] > 0
    assert established["delta_mae"] < 0


def test_event_concentration_identifies_dominated_benefit(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_stability_artifacts(config.metrics_output_dir)
    event_rows, _ = _canonical_frames(config.metrics_output_dir)

    concentration, _ = build_event_concentration(event_rows)

    top_events = json.loads(concentration["top_k_beneficial_events"].iloc[0])
    assert top_events[0]["event_slug"] == "zandvoort"
    assert concentration["share_total_improvement_explained_by_top_beneficial_events"].iloc[0] > 0.9


def test_leave_one_event_out_detects_sign_flip() -> None:
    event_rows = pd.DataFrame(
        [
            _event_row(2024, "Dominant", "dominant", 0, "established_season", 0.2, 0.5),
            _event_row(2024, "Small Harm A", "small-harm-a", 1, "established_season", 0.6, 0.5),
            _event_row(2024, "Small Harm B", "small-harm-b", 2, "established_season", 0.6, 0.5),
        ]
    )

    _, leave_one_out = build_event_concentration(event_rows)

    assert int(leave_one_out["sign_flip_when_removed"].sum()) == 1


def test_tail_metrics_and_candidate_worse_threshold_rates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_stability_artifacts(config.metrics_output_dir)
    _, row_errors = _canonical_frames(config.metrics_output_dir)

    tail = build_tail_risk(row_errors, threshold_sec=0.05)

    assert tail["rows"].iloc[0] == 8
    assert tail["candidate_worse_by_more_than_threshold_rate"].iloc[0] == pytest.approx(0.25)
    assert tail["candidate_worse_by_more_than_threshold_count"].iloc[0] == 2


def test_live_and_shadow_replay_status_remain_separate(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_stability_artifacts(config.metrics_output_dir, include_replay=True)

    create_season_aware_stability_report(config)
    replay = pd.read_csv(
        config.metrics_output_dir / "season_aware_stability_replay_shadow_summary.csv"
    )

    assert int(replay["live_selected_events"].iloc[0]) == 0
    assert int(replay["shadow_counterfactual_selected_events"].iloc[0]) == 1


def test_missing_optional_replay_artifacts_are_unavailable_not_negative(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_stability_artifacts(config.metrics_output_dir, include_replay=False)

    create_season_aware_stability_report(config)
    replay = pd.read_csv(
        config.metrics_output_dir / "season_aware_stability_replay_shadow_summary.csv"
    )
    missing = pd.read_csv(config.metrics_output_dir / "season_aware_stability_missing_evidence.csv")

    assert replay["evidence_available"].iloc[0] is False or not bool(
        replay["evidence_available"].iloc[0]
    )
    assert "prospective_replay_candidate_evidence_ledger.csv" in set(missing["artifact_name"])


def test_stability_classification_is_conservative_for_concentrated_evidence(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_stability_artifacts(config.metrics_output_dir)

    create_season_aware_stability_report(config)
    payload = json.loads(
        (config.metrics_output_dir / "season_aware_stability_summary.json").read_text()
    )

    assert payload["event_concentration_classification"] == "concentrated_support"
    assert payload["policy_recommendation"] == "season_aware_candidate_requires_more_evidence"


def test_backtest_report_exposes_stability_summary_fields() -> None:
    payload = build_backtest_report_payload(
        quality={"n_rows": 1, "n_events": 1, "checkpoints": ["after_fp3"]},
        baseline_metrics={},
        tabular_metrics=None,
        season_aware_stability_summary={
            "status": "complete",
            "overall_stability_classification": "tail_risk_concern",
            "primary_cautionary_evidence": ["Candidate worse-by-threshold row rate is 25.0%."],
            "policy_recommendation": "season_aware_candidate_requires_more_evidence",
        },
    )

    assert payload["season_aware_stability_available"] is True
    assert payload["season_aware_stability_classification"] == "tail_risk_concern"
    assert payload["season_aware_stability_primary_caution"].startswith("Candidate")


def _config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        project_root=tmp_path,
        fastf1_cache_dir=tmp_path / "cache",
        lap_output_dir=tmp_path / "data/raw/laps",
        session_metadata_output_dir=tmp_path / "data/raw/session_metadata",
        clean_lap_output_dir=tmp_path / "data/interim/clean_laps",
        session_features_output_dir=tmp_path / "data/processed/session_features",
        modeling_output_dir=tmp_path / "data/processed/modeling",
        metrics_output_dir=tmp_path / "reports/metrics",
    )


def _canonical_frames(metrics_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows = canonical_event_rows(
        pd.read_csv(metrics_dir / "season_aware_event_level_comparison.csv")
    )
    row_errors = canonical_row_errors(
        pd.read_parquet(
            metrics_dir / "ablation_current_season_only_with_prior_predictions.parquet"
        ),
        pd.read_parquet(metrics_dir / "ablation_uniform_predictions.parquet"),
        event_rows,
    )
    return event_rows, row_errors


def _write_stability_artifacts(
    metrics_dir: Path,
    *,
    feature_group: str = "base_plus_relative",
    include_replay: bool = False,
) -> None:
    metrics_dir.mkdir(parents=True)
    event_rows = [
        _artifact_event(2024, "Bahrain", "bahrain", 0, "cold_start", 0.3, 0.2, feature_group),
        _artifact_event(
            2024,
            "Zandvoort",
            "zandvoort",
            6,
            "established_season",
            0.1,
            1.5,
            feature_group,
        ),
        _artifact_event(2025, "Monaco", "monaco", 1, "early_season", 0.45, 0.5, feature_group),
        _artifact_event(
            2025,
            "Baku",
            "baku",
            7,
            "established_season",
            0.1,
            0.7,
            feature_group,
        ),
    ]
    pd.DataFrame(event_rows).to_csv(
        metrics_dir / "season_aware_event_level_comparison.csv",
        index=False,
    )
    current_rows: list[dict[str, object]] = []
    uniform_rows: list[dict[str, object]] = []
    for row in event_rows:
        for driver_index, driver in enumerate(("VER", "NOR")):
            actual = float(driver_index)
            candidate_error = float(row["candidate_mae_gap_sec"])
            default_error = float(row["static_mae_gap_sec"])
            common = {
                "season": row["season"],
                "event": row["event"],
                "event_slug": row["event_slug"],
                "fold_id": row["fold_id"],
                "checkpoint": "after_fp3",
                "driver": driver,
                "quali_gap_to_pole_sec": actual,
                "model_name": "random_forest",
                "feature_group": feature_group,
                "family": "ablation",
            }
            current_rows.append(
                {
                    **common,
                    "temporal_weighting_policy": "current_season_only_with_prior",
                    "predicted_quali_gap_to_pole_sec": actual + candidate_error,
                }
            )
            uniform_rows.append(
                {
                    **common,
                    "temporal_weighting_policy": "uniform",
                    "predicted_quali_gap_to_pole_sec": actual + default_error,
                }
            )
    pd.DataFrame(current_rows).to_parquet(
        metrics_dir / "ablation_current_season_only_with_prior_predictions.parquet",
        index=False,
    )
    pd.DataFrame(uniform_rows).to_parquet(
        metrics_dir / "ablation_uniform_predictions.parquet",
        index=False,
    )
    if include_replay:
        pd.DataFrame(
            [
                {
                    "split_name": "prospective_replay_train_2024_test_2025",
                    "policy_profile": "season_aware_frozen",
                    "season_aware_selected": False,
                    "shadow_season_aware_candidate_eligible_under_frozen_gates": True,
                    "shadow_history_counterfactual_selection": "season_aware_weighted_candidate",
                    "prior_shadow_candidate_events_available": 5,
                    "prior_shadow_candidate_prediction_rows_available": 100,
                    "prior_shadow_candidate_default_aligned_rows": 100,
                    "shadow_candidate_prior_mae": 0.5,
                    "shadow_default_prior_mae": 0.7,
                    "shadow_prior_improvement_sec": 0.2,
                }
            ]
        ).to_csv(metrics_dir / "prospective_replay_candidate_evidence_ledger.csv", index=False)


def _artifact_event(
    season: int,
    event: str,
    event_slug: str,
    event_order: int,
    regime: str,
    candidate_mae: float,
    default_mae: float,
    feature_group: str,
) -> dict[str, object]:
    return {
        "candidate_family": "ablation",
        "candidate_model_name": "random_forest",
        "candidate_feature_group": feature_group,
        "training_policy": "current_season_only_with_prior",
        "season": season,
        "event": event,
        "event_slug": event_slug,
        "fold_id": f"{season}_{event_slug}",
        "current_season_prior_event_count": event_order,
        "current_season_evidence_regime": regime,
        "rows": 2,
        "static_mae_gap_sec": default_mae,
        "candidate_mae_gap_sec": candidate_mae,
        "delta_vs_static_sec": candidate_mae - default_mae,
    }


def _event_row(
    season: int,
    event: str,
    event_slug: str,
    event_order: int,
    regime: str,
    candidate_mae: float,
    default_mae: float,
) -> dict[str, object]:
    return {
        "season": season,
        "event_order": event_order,
        "event": event,
        "event_slug": event_slug,
        "fold_id": f"{season}_{event_slug}",
        "regime": regime,
        "row_count": 1,
        "candidate_mae": candidate_mae,
        "default_mae": default_mae,
        "delta_mae": candidate_mae - default_mae,
        "candidate_better": candidate_mae < default_mae,
        "absolute_delta": abs(candidate_mae - default_mae),
    }
