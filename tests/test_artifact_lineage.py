import json
from pathlib import Path

import pandas as pd

from f1_prediction.config import DataConfig, load_model_config
from f1_prediction.modeling.artifact_lineage import (
    ArtifactSourceIdentity,
    build_artifact_summary,
    build_fold_comparison,
    build_season_aware_rebuild_registry,
    build_static_source_contract,
    clean_rebuild_workflow,
    compare_static_to_uniform_ablation,
    create_champion_source_lineage_report,
    inspect_artifact,
    resolve_artifact_source,
)
from f1_prediction.modeling.champion_policy import (
    ChampionSelectionMode,
    _champion_prediction_rows,
)
from f1_prediction.modeling.season_aware_policy_forensics import (
    build_policy_event_counterfactual,
    build_selected_fold_analysis,
)


def test_manifest_captures_artifact_identity_metadata(tmp_path: Path) -> None:
    path = tmp_path / "artifact.parquet"
    _static_predictions().to_parquet(path, index=False)

    row = inspect_artifact(path, project_root=tmp_path, config_signature="abc")

    assert row["artifact_exists"] is True
    assert row["row_count"] == 4
    assert row["unique_folds"] == 2
    assert row["unique_checkpoints"] == 1
    assert row["model_name"] == "random_forest"
    assert row["feature_group"] == "base_plus_relative"
    assert row["config_signature"] == "abc"


def test_exact_row_alignment_works_on_matching_artifacts(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    static = _static_predictions()
    uniform = _uniform_predictions(static["predicted_quali_gap_to_pole_sec"].tolist())
    static.to_parquet(metrics_dir / "champion_static_predictions.parquet", index=False)
    uniform.to_parquet(metrics_dir / "ablation_uniform_predictions.parquet", index=False)

    rows = compare_static_to_uniform_ablation(metrics_dir)
    folds = build_fold_comparison(rows)

    assert rows["row_match_status"].eq("matched").all()
    assert rows["prediction_tolerance_match"].all()
    assert folds["tolerance_prediction_match_rate"].tolist() == [1.0, 1.0]


def test_row_comparison_detects_prediction_mismatches(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    static = _static_predictions()
    uniform = _uniform_predictions([9.0, 9.0, 9.0, 9.0])
    static.to_parquet(metrics_dir / "champion_static_predictions.parquet", index=False)
    uniform.to_parquet(metrics_dir / "ablation_uniform_predictions.parquet", index=False)

    rows = compare_static_to_uniform_ablation(metrics_dir)

    assert not rows["prediction_tolerance_match"].any()
    assert rows["abs_prediction_delta_sec"].max() > 0
    assert set(rows["mismatch_cause"]) == {"different_preprocessing_or_model_configuration"}


def test_fold_comparison_detects_differing_scopes(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    static = _static_predictions()
    uniform = _uniform_predictions(static["predicted_quali_gap_to_pole_sec"].tolist()).iloc[:-1]
    static.to_parquet(metrics_dir / "champion_static_predictions.parquet", index=False)
    uniform.to_parquet(metrics_dir / "ablation_uniform_predictions.parquet", index=False)

    rows = compare_static_to_uniform_ablation(metrics_dir)
    folds = build_fold_comparison(rows)

    assert rows["row_match_status"].value_counts().to_dict()["static_only"] == 1
    assert "different_rows" in set(folds["mismatch_cause"])


def test_static_source_contract_includes_expected_uniform_rf_identity(tmp_path: Path) -> None:
    config = _data_config(tmp_path)
    model_config = load_model_config(project_root=Path.cwd())

    contract = build_static_source_contract(config, model_config)

    assert contract["configured_matches_expected"] is True
    expected = contract["expected"]
    assert expected["source_family"] == "ablation"
    assert expected["source_model_name"] == "random_forest"
    assert expected["source_feature_group"] == "base_plus_relative"
    assert expected["source_temporal_weighting_policy"] == "uniform"


def test_source_verification_fails_safely_for_mismatched_artifacts(tmp_path: Path) -> None:
    config = _data_config(tmp_path)
    model_config = load_model_config(project_root=Path.cwd())
    metrics_dir = config.metrics_output_dir
    metrics_dir.mkdir(parents=True)
    static = _static_predictions()
    uniform = _uniform_predictions([5.0, 5.0, 5.0, 5.0])
    latest = _uniform_predictions(static["predicted_quali_gap_to_pole_sec"].tolist())
    latest["temporal_weighting_policy"] = "current_season_only_with_prior"
    static.to_parquet(metrics_dir / "champion_static_predictions.parquet", index=False)
    uniform.to_parquet(metrics_dir / "ablation_uniform_predictions.parquet", index=False)
    latest.to_parquet(metrics_dir / "ablation_predictions.parquet", index=False)

    summary = create_champion_source_lineage_report(config, model_config)
    payload = json.loads(summary.summary_path.read_text())

    assert payload["static_source_verification"]["static_source_verified"] is False
    assert payload["root_cause_classification"] == "stale_artifact_generation_order"
    assert payload["m26_counterfactual_conclusions_valid"] is False


def test_forensics_does_not_report_definitive_labels_when_source_verification_fails() -> None:
    static = _static_predictions()
    live = static.copy()
    selection = pd.DataFrame(
        {
            "fold_id": [1, 2],
            "checkpoint": ["after_fp3", "after_fp3"],
            "season_aware_selected": [True, False],
            "season_aware_selection_reason": ["selected_after_prior_evidence", "margin_not_met"],
        }
    )
    verification = {
        "static_source_verified": False,
        "counterfactual_comparison_valid": False,
        "counterfactual_invalid_reason": "source_mismatch",
    }

    events = build_policy_event_counterfactual(
        static_predictions=static,
        guarded_predictions=static,
        season_aware_predictions=live,
        weighted_candidate_predictions=live,
        default_candidate_predictions=static,
        season_aware_selection=selection,
        static_source_verification=verification,
    )
    analysis, cases = build_selected_fold_analysis(
        events,
        static_predictions=static,
        season_aware_predictions=live,
        season_aware_selection=selection,
        tolerance_sec=0.05,
        static_source_verification=verification,
    )

    assert analysis["counterfactual_comparison_valid"].eq(False).all()
    assert analysis["event_harmful_switch"].isna().all()
    assert cases["counterfactual_comparison_valid"].eq(False).all()
    assert cases["harmful_switch"].isna().all()


def test_clean_rebuild_workflow_is_deterministic_and_scoped() -> None:
    first = clean_rebuild_workflow()
    second = clean_rebuild_workflow()

    assert first == second
    assert first["preserve_generated_artifact_ignore_rules"] is True
    assert all(path.startswith("reports/metrics/") for path in first["scoped_generated_artifacts"])
    assert any("champion-source-lineage" in command for command in first["commands"])


def test_uniform_and_weighted_artifact_paths_are_distinct(tmp_path: Path) -> None:
    uniform = resolve_artifact_source(
        tmp_path,
        ArtifactSourceIdentity(
            "ablation_predictions",
            "ablation",
            "random_forest",
            "base_plus_relative",
            "uniform",
        ),
    )
    weighted = resolve_artifact_source(
        tmp_path,
        ArtifactSourceIdentity(
            "ablation_predictions",
            "ablation",
            "random_forest",
            "base_plus_relative",
            "current_season_only_with_prior",
        ),
    )

    assert uniform.name == "ablation_uniform_predictions.parquet"
    assert weighted.name == "ablation_current_season_only_with_prior_predictions.parquet"
    assert uniform != weighted


def test_artifact_registry_resolves_exact_requested_source_identity(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    _uniform_predictions([0.1, 0.2, 0.3, 0.4]).to_parquet(
        metrics_dir / "ablation_uniform_predictions.parquet",
        index=False,
    )

    registry = build_season_aware_rebuild_registry(
        metrics_dir=metrics_dir,
        project_root=tmp_path,
        config_path=None,
    )
    uniform = registry[
        registry["artifact_kind"].eq("ablation_predictions")
        & registry["temporal_weighting_policy"].eq("uniform")
    ].iloc[0]

    assert uniform["artifact_path"] == "metrics/ablation_uniform_predictions.parquet"
    assert bool(uniform["artifact_exists"]) is True


def test_existing_champion_policy_rows_keep_selection_identity_and_source_metadata() -> None:
    rows = _uniform_predictions([0.1, 0.2, 0.3, 0.4]).copy()
    rows["candidate_family"] = "ablation"
    rows["source_artifact_kind"] = "ablation"
    rows["source_artifact_path"] = "reports/metrics/ablation_uniform_predictions.parquet"
    rows["source_family"] = "ablation"
    rows["source_model_name"] = "random_forest"
    rows["source_feature_group"] = "base_plus_relative"
    rows["source_temporal_weighting_policy"] = "uniform"
    rows["source_strategy"] = "walk_forward"
    rows["source_dataset_signature"] = pd.NA
    rows["source_config_signature"] = pd.NA
    rows["source_prediction_signature"] = "sig"

    output = _champion_prediction_rows(
        rows,
        ChampionSelectionMode.static,
        load_model_config(project_root=Path.cwd()).champion_policy.static["after_fp3"],
    )

    assert output["selection_mode"].eq("static").all()
    assert output["selected_temporal_weighting_policy"].eq("uniform").all()
    assert (
        output["source_artifact_path"]
        .eq("reports/metrics/ablation_uniform_predictions.parquet")
        .all()
    )


def test_artifact_summary_marks_absent_artifacts(tmp_path: Path) -> None:
    summary = build_artifact_summary(
        metrics_dir=tmp_path / "metrics",
        project_root=tmp_path,
        config_path=None,
    )

    assert "champion_static_predictions.parquet" in set(summary["artifact_name"])
    assert not summary["artifact_exists"].any()


def _static_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": ["walk_forward"] * 4,
            "fold_id": [1, 1, 2, 2],
            "season": [2024, 2024, 2024, 2024],
            "event": ["A", "A", "B", "B"],
            "event_slug": ["a", "a", "b", "b"],
            "checkpoint": ["after_fp3"] * 4,
            "driver": ["NOR", "PIA", "NOR", "PIA"],
            "team": ["McLaren", "McLaren", "McLaren", "McLaren"],
            "quali_position": [1, 2, 1, 2],
            "quali_gap_to_pole_sec": [0.0, 0.2, 0.0, 0.4],
            "reached_q3": [1, 1, 1, 1],
            "predicted_quali_gap_to_pole_sec": [0.1, 0.3, 0.2, 0.5],
            "predicted_quali_position": [1, 2, 1, 2],
            "predicted_reached_q3": [1, 1, 1, 1],
            "selection_mode": ["static"] * 4,
            "selected_family": ["ablation"] * 4,
            "selected_model_name": ["random_forest"] * 4,
            "selected_feature_group": ["base_plus_relative"] * 4,
            "selected_temporal_weighting_policy": ["uniform"] * 4,
            "temporal_weighting_policy": ["uniform"] * 4,
        }
    )


def _uniform_predictions(predictions: list[float]) -> pd.DataFrame:
    static = _static_predictions()
    return pd.DataFrame(
        {
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
            "predicted_quali_gap_to_pole_sec": predictions,
            "predicted_quali_position": static["predicted_quali_position"],
            "predicted_reached_q3": static["predicted_reached_q3"],
            "feature_group": ["base_plus_relative"] * len(static),
            "fold_id": static["fold_id"],
            "strategy": static["strategy"],
            "test_event": static["season"].astype(str) + "/" + static["event_slug"],
            "prediction_type": ["tabular"] * len(static),
            "temporal_weighting_policy": ["uniform"] * len(static),
        }
    )


def _data_config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        project_root=tmp_path,
        fastf1_cache_dir=tmp_path / "cache",
        lap_output_dir=tmp_path / "laps",
        session_metadata_output_dir=tmp_path / "metadata",
        clean_lap_output_dir=tmp_path / "clean",
        session_features_output_dir=tmp_path / "features",
        modeling_output_dir=tmp_path / "modeling",
        metrics_output_dir=tmp_path / "metrics",
    )
