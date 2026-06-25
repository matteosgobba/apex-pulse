from pathlib import Path

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.modeling.season_aware_rebuild import (
    build_rebuild_validation,
    cleanup_scoped_rebuild_artifacts,
    create_season_aware_rebuild_report,
    scoped_rebuild_artifacts,
)


def test_scoped_rebuild_cleanup_does_not_remove_unrelated_artifacts(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    known = metrics_dir / "season_aware_rebuild_summary.json"
    unrelated = metrics_dir / "user_notes.json"
    known.write_text("{}", encoding="utf-8")
    unrelated.write_text("keep", encoding="utf-8")

    removed = cleanup_scoped_rebuild_artifacts(metrics_dir, dry_run=False)

    assert known in removed
    assert not known.exists()
    assert unrelated.exists()


def test_scoped_rebuild_dry_run_lists_only_known_workflow_artifacts(tmp_path: Path) -> None:
    artifacts = scoped_rebuild_artifacts(tmp_path / "metrics")

    assert artifacts
    assert all(path.parent == tmp_path / "metrics" for path in artifacts)
    assert tmp_path / "metrics" / "user_notes.json" not in artifacts


def test_scoped_rebuild_dry_run_does_not_overwrite_existing_summary(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.metrics_output_dir.mkdir(parents=True)
    summary_path = config.metrics_output_dir / "season_aware_rebuild_summary.json"
    summary_path.write_text('{"status": "complete"}\n', encoding="utf-8")

    summary = create_season_aware_rebuild_report(
        config,
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        dry_run=True,
    )

    assert summary.status == "dry_run"
    assert summary_path.read_text(encoding="utf-8") == '{"status": "complete"}\n'


def test_clean_rebuild_validation_detects_identical_static_uniform_predictions(
    tmp_path: Path,
) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    static = _static_predictions([0.1, 0.3])
    uniform = _ablation_predictions([0.1, 0.3], "uniform")
    static.to_parquet(metrics_dir / "champion_static_predictions.parquet", index=False)
    uniform.to_parquet(metrics_dir / "ablation_uniform_predictions.parquet", index=False)
    uniform.to_parquet(
        metrics_dir / "ablation_current_season_only_with_prior_predictions.parquet",
        index=False,
    )
    static.to_parquet(
        metrics_dir / "champion_stabilized_nested_guarded_predictions.parquet",
        index=False,
    )
    _forensics_summary(True).to_json(
        metrics_dir / "season_aware_policy_forensics_summary.json",
        orient="records",
    )
    (metrics_dir / "season_aware_policy_forensics_summary.json").write_text(
        '{"counterfactual_comparison_valid": true}',
        encoding="utf-8",
    )

    validation = build_rebuild_validation(metrics_dir=metrics_dir)
    static_check = validation[validation["check_name"].eq("static_uniform_prediction_match")].iloc[
        0
    ]

    assert static_check["status"] == "passed"
    assert static_check["match_rate"] == 1.0


def test_clean_rebuild_validation_detects_mismatched_predictions(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    _static_predictions([0.1, 0.3]).to_parquet(
        metrics_dir / "champion_static_predictions.parquet",
        index=False,
    )
    _ablation_predictions([9.0, 9.0], "uniform").to_parquet(
        metrics_dir / "ablation_uniform_predictions.parquet",
        index=False,
    )

    validation = build_rebuild_validation(metrics_dir=metrics_dir)
    static_check = validation[validation["check_name"].eq("static_uniform_prediction_match")].iloc[
        0
    ]

    assert static_check["status"] == "failed"
    assert static_check["match_rate"] == 0.0


def test_guarded_source_verification_inherits_verified_static_source(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    static = _static_predictions([0.1, 0.3])
    static.to_parquet(metrics_dir / "champion_static_predictions.parquet", index=False)
    static.to_parquet(
        metrics_dir / "champion_stabilized_nested_guarded_predictions.parquet",
        index=False,
    )
    _ablation_predictions([0.1, 0.3], "uniform").to_parquet(
        metrics_dir / "ablation_uniform_predictions.parquet",
        index=False,
    )

    validation = build_rebuild_validation(metrics_dir=metrics_dir)
    guarded = validation[validation["check_name"].eq("guarded_static_prediction_match")].iloc[0]

    assert guarded["status"] == "passed"
    assert guarded["match_rate"] == 1.0


def test_season_aware_weighted_source_verification_remains_separate(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    weighted = _ablation_predictions([0.0, 0.2], "current_season_only_with_prior")
    live = _static_predictions([0.0, 0.2])
    live["selection_mode"] = "season_aware_nested_guarded"
    selection = pd.DataFrame(
        {
            "fold_id": [1],
            "checkpoint": ["after_fp3"],
            "season_aware_selected": [True],
        }
    )
    live.to_parquet(
        metrics_dir / "champion_season_aware_nested_guarded_predictions.parquet",
        index=False,
    )
    selection.to_parquet(
        metrics_dir / "champion_season_aware_nested_guarded_selection.parquet",
        index=False,
    )
    weighted.to_parquet(
        metrics_dir / "ablation_current_season_only_with_prior_predictions.parquet",
        index=False,
    )

    validation = build_rebuild_validation(metrics_dir=metrics_dir)
    weighted_check = validation[
        validation["check_name"].eq("season_aware_weighted_source_verified")
    ].iloc[0]

    assert weighted_check["status"] == "passed"
    assert weighted_check["match_rate"] == 1.0


def _static_predictions(predictions: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": ["walk_forward"] * len(predictions),
            "fold_id": [1] * len(predictions),
            "season": [2025] * len(predictions),
            "event": ["Silverstone"] * len(predictions),
            "event_slug": ["silverstone"] * len(predictions),
            "checkpoint": ["after_fp3"] * len(predictions),
            "driver": ["NOR", "PIA"][: len(predictions)],
            "team": ["McLaren"] * len(predictions),
            "quali_position": [1, 2][: len(predictions)],
            "quali_gap_to_pole_sec": [0.0, 0.2][: len(predictions)],
            "reached_q3": [1] * len(predictions),
            "predicted_quali_gap_to_pole_sec": predictions,
            "predicted_quali_position": [1, 2][: len(predictions)],
            "predicted_reached_q3": [1] * len(predictions),
            "selection_mode": ["static"] * len(predictions),
            "selected_family": ["ablation"] * len(predictions),
            "selected_model_name": ["random_forest"] * len(predictions),
            "selected_feature_group": ["base_plus_relative"] * len(predictions),
            "selected_temporal_weighting_policy": ["uniform"] * len(predictions),
            "temporal_weighting_policy": ["uniform"] * len(predictions),
        }
    )


def _ablation_predictions(predictions: list[float], policy: str) -> pd.DataFrame:
    static = _static_predictions(predictions)
    return pd.DataFrame(
        {
            "strategy": static["strategy"],
            "fold_id": static["fold_id"],
            "season": static["season"],
            "event": static["event"],
            "event_slug": static["event_slug"],
            "checkpoint": static["checkpoint"],
            "driver": static["driver"],
            "team": static["team"],
            "quali_position": static["quali_position"],
            "quali_gap_to_pole_sec": static["quali_gap_to_pole_sec"],
            "reached_q3": static["reached_q3"],
            "model_name": ["random_forest"] * len(static),
            "feature_group": ["base_plus_relative"] * len(static),
            "prediction_type": ["tabular"] * len(static),
            "temporal_weighting_policy": [policy] * len(static),
            "predicted_quali_gap_to_pole_sec": predictions,
            "predicted_quali_position": static["predicted_quali_position"],
            "predicted_reached_q3": static["predicted_reached_q3"],
            "test_event": static["season"].astype(str) + "/" + static["event_slug"],
        }
    )


def _forensics_summary(valid: bool) -> pd.DataFrame:
    return pd.DataFrame({"counterfactual_comparison_valid": [valid]})


def _config(project_root: Path) -> DataConfig:
    return DataConfig(
        project_root=project_root,
        fastf1_cache_dir=project_root / "cache",
        lap_output_dir=project_root / "laps",
        session_metadata_output_dir=project_root / "metadata",
        clean_lap_output_dir=project_root / "clean",
        session_features_output_dir=project_root / "features",
        modeling_output_dir=project_root / "modeling",
        metrics_output_dir=project_root / "metrics",
    )
