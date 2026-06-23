import json
from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.config import ChampionDiagnosticsConfig, DataConfig
from f1_prediction.modeling.champion_diagnostics import (
    build_conformal_coverage_by_error_regime,
    build_conformal_miss_cases,
    build_conformal_miss_summaries,
    build_fp3_policy_failure_analysis,
    build_harmful_switch_rows,
    build_harmful_switch_summaries,
    create_champion_diagnostics_report,
    generate_champion_diagnostics_figures,
)


def test_harmful_switch_labels_harmful_beneficial_and_neutral() -> None:
    static = _predictions(
        "static",
        actual=[1.0, 1.0, 1.0],
        predicted=[1.0, 1.0, 1.0],
    )
    comparison = _predictions(
        "stabilized_nested",
        actual=[1.0, 1.0, 1.0],
        predicted=[1.2, 0.7, 1.03],
    )

    rows = build_harmful_switch_rows(
        static,
        comparison,
        selection_mode="stabilized_nested",
        tolerance_sec=0.05,
    )

    assert rows["harmful_switch"].tolist() == [True, True, False]
    assert rows["beneficial_switch"].tolist() == [False, False, False]
    assert rows["error_delta_vs_static_sec"].iloc[0] == pytest.approx(0.2)

    better_static = _predictions(
        "static",
        actual=[1.0],
        predicted=[1.3],
        drivers=["PIA"],
    )
    better_comparison = _predictions(
        "nested",
        actual=[1.0],
        predicted=[1.0],
        drivers=["PIA"],
    )
    better = build_harmful_switch_rows(
        better_static,
        better_comparison,
        selection_mode="nested",
        tolerance_sec=0.05,
    )

    assert bool(better["beneficial_switch"].iloc[0])
    assert not bool(better["harmful_switch"].iloc[0])


def test_harmful_switch_summaries_by_checkpoint_event_and_method() -> None:
    rows = build_harmful_switch_rows(
        _predictions("static", actual=[1.0, 1.0], predicted=[1.0, 1.0]),
        _predictions("nested", actual=[1.0, 1.0], predicted=[1.2, 1.02]),
        selection_mode="nested",
        tolerance_sec=0.05,
    )

    summaries = build_harmful_switch_summaries(rows)

    checkpoint = summaries["checkpoint"].iloc[0]
    event = summaries["event"].iloc[0]
    method = summaries["method"].iloc[0]
    assert checkpoint["rows"] == 2
    assert checkpoint["harmful_switches"] == 1
    assert checkpoint["neutral_switches"] == 1
    assert event["event"] == "Monza"
    assert method["comparison_selected_model_name"] == "random_forest"


def test_fp3_policy_failure_detects_abandoned_static_rf_policy() -> None:
    static = _predictions(
        "static",
        actual=[1.0, 2.0],
        predicted=[1.0, 2.0],
        checkpoints=["after_fp3", "after_fp3"],
        selected_family="ablation",
        selected_model_name="random_forest",
        selected_feature_group="base_plus_relative",
    )
    stabilized = _predictions(
        "stabilized_nested",
        actual=[1.0, 2.0],
        predicted=[1.3, 2.1],
        checkpoints=["after_fp3", "after_fp3"],
        selected_family="robust_baseline",
        selected_model_name="best_valid_lap",
        selected_feature_group=pd.NA,
    )
    selection = pd.DataFrame(
        {
            "fold_id": [1],
            "checkpoint": ["after_fp3"],
            "fallback_used": [True],
            "fallback_reason": ["insufficient_history"],
        }
    )

    table = build_fp3_policy_failure_analysis(static, stabilized, selection, tolerance_sec=0.05)

    row = table.iloc[0]
    assert bool(row["stabilized_abandoned_static_fp3_rf"])
    assert row["stabilized_method"] == "robust_baseline/best_valid_lap"
    assert row["stabilized_fallback_rate"] == pytest.approx(1.0)
    assert row["harmful_switches"] == 2


def test_conformal_miss_side_for_covered_below_and_above() -> None:
    cases = build_conformal_miss_cases(
        _interval_predictions(
            actual=[1.0, 0.2, 2.5],
            predicted=[1.0, 1.0, 1.0],
            low=[0.5, 0.5, 0.5],
            high=[1.5, 1.5, 1.5],
            contains=[True, False, False],
        )
    )

    assert cases["interval_miss"].tolist() == [False, True, True]
    assert pd.isna(cases["miss_side"].iloc[0])
    assert cases["miss_side"].iloc[1] == "below_interval"
    assert cases["miss_side"].iloc[2] == "above_interval"


def test_conformal_miss_cases_keep_predicted_bucket_metadata() -> None:
    predictions = _interval_predictions(
        actual=[1.0],
        predicted=[1.0],
        low=[0.5],
        high=[1.5],
        contains=[True],
    )
    predictions["uncertainty_method"] = "conformal_predicted_gap_bucket"
    predictions["predicted_gap_bucket"] = "close_midfield"
    predictions["uncertainty_calibration_level"] = "checkpoint_method_bucket"

    cases = build_conformal_miss_cases(predictions)

    assert cases["uncertainty_method"].iloc[0] == "conformal_predicted_gap_bucket"
    assert cases["predicted_gap_bucket"].iloc[0] == "close_midfield"
    assert cases["uncertainty_calibration_level"].iloc[0] == "checkpoint_method_bucket"


def test_conformal_miss_summaries_compute_coverage_and_miss_count() -> None:
    cases = build_conformal_miss_cases(
        _interval_predictions(
            actual=[1.0, 0.2, 2.5],
            predicted=[1.0, 1.0, 1.0],
            low=[0.5, 0.5, 0.5],
            high=[1.5, 1.5, 1.5],
            contains=[True, False, False],
        )
    )

    summaries = build_conformal_miss_summaries(cases)
    checkpoint = summaries["checkpoint"].iloc[0]

    assert checkpoint["rows_with_interval"] == 3
    assert checkpoint["miss_count"] == 2
    assert checkpoint["coverage"] == pytest.approx(1 / 3)
    assert checkpoint["mean_interval_width_sec"] == pytest.approx(1.0)


def test_coverage_by_actual_gap_bucket_threshold_boundaries() -> None:
    cases = build_conformal_miss_cases(
        _interval_predictions(
            actual=[0.5, 1.5, 3.0, 3.1],
            predicted=[0.5, 1.5, 3.0, 3.1],
            low=[0.0, 1.0, 2.5, 3.0],
            high=[1.0, 2.0, 3.5, 4.0],
            contains=[True, True, True, True],
        )
    )

    table = build_conformal_coverage_by_error_regime(cases)
    actual = table[table["bucket_type"].eq("actual_gap_bucket")]

    assert set(actual["gap_bucket"]) == {
        "pole_contender",
        "close_midfield",
        "midfield",
        "backmarker_or_outlier",
    }
    assert actual["coverage"].min() == pytest.approx(1.0)


def test_diagnostics_summary_records_missing_inputs_gracefully(tmp_path: Path) -> None:
    config = _config(tmp_path)

    summary = create_champion_diagnostics_report(config, ChampionDiagnosticsConfig())
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))

    assert summary.status == "no_inputs"
    assert "champion_static_predictions.parquet" in payload["missing_inputs"]
    assert (config.metrics_output_dir / "champion_harmful_switches.csv").is_file()
    assert payload["status"] == "no_inputs"


def test_champion_diagnostics_compares_guarded_mode_when_artifacts_exist(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.metrics_output_dir.mkdir(parents=True, exist_ok=True)
    static = _predictions("static", actual=[1.0], predicted=[1.0], drivers=["NOR"])
    nested = _predictions("nested", actual=[1.0], predicted=[1.1], drivers=["NOR"])
    stabilized = _predictions(
        "stabilized_nested",
        actual=[1.0],
        predicted=[1.2],
        drivers=["NOR"],
    )
    guarded = _predictions(
        "stabilized_nested_guarded",
        actual=[1.0],
        predicted=[1.0],
        drivers=["NOR"],
    )
    for mode, frame in (
        ("static", static),
        ("nested", nested),
        ("stabilized_nested", stabilized),
        ("stabilized_nested_guarded", guarded),
    ):
        frame.to_parquet(config.metrics_output_dir / f"champion_{mode}_predictions.parquet")
        pd.DataFrame(
            {
                "fold_id": [1],
                "checkpoint": ["after_fp1"],
                "selected_family": ["ablation"],
                "selected_model_name": ["random_forest"],
                "selected_feature_group": ["base_plus_relative"],
            }
        ).to_parquet(config.metrics_output_dir / f"champion_{mode}_selection.parquet")

    summary = create_champion_diagnostics_report(config, ChampionDiagnosticsConfig())
    switches = pd.read_csv(config.metrics_output_dir / "champion_harmful_switches.csv")

    assert summary.status == "complete"
    assert "stabilized_nested_guarded" in set(switches["selection_mode"])


def test_figure_generation_does_not_crash_on_minimal_inputs(tmp_path: Path) -> None:
    rows = build_harmful_switch_rows(
        _predictions("static", actual=[1.0], predicted=[1.0]),
        _predictions("stabilized_nested", actual=[1.0], predicted=[1.2]),
        selection_mode="stabilized_nested",
        tolerance_sec=0.05,
    )
    switch_summaries = build_harmful_switch_summaries(rows)
    fp3 = build_fp3_policy_failure_analysis(
        _predictions(
            "static",
            actual=[1.0],
            predicted=[1.0],
            checkpoints=["after_fp3"],
            selected_family="ablation",
            selected_model_name="random_forest",
            selected_feature_group="base_plus_relative",
        ),
        _predictions(
            "stabilized_nested",
            actual=[1.0],
            predicted=[1.2],
            checkpoints=["after_fp3"],
            selected_family="robust_baseline",
            selected_model_name="best_valid_lap",
            selected_feature_group=pd.NA,
        ),
        pd.DataFrame(),
    )
    cases = build_conformal_miss_cases(
        _interval_predictions(
            actual=[1.0],
            predicted=[1.0],
            low=[0.5],
            high=[1.5],
            contains=[True],
        )
    )
    conformal = build_conformal_miss_summaries(cases)
    regime = build_conformal_coverage_by_error_regime(cases)

    paths, issues = generate_champion_diagnostics_figures(
        figures_dir=tmp_path,
        harmful_switches=rows,
        fp3_failures=fp3,
        conformal_checkpoint_summary=conformal["checkpoint"],
        conformal_event_summary=conformal["event"],
        coverage_by_regime=regime,
    )

    assert isinstance(issues, list)
    assert switch_summaries["checkpoint"]["harmful_switches"].iloc[0] == 1
    assert all(path.is_file() for path in paths)


def _predictions(
    mode: str,
    *,
    actual: list[float],
    predicted: list[float],
    drivers: list[str] | None = None,
    checkpoints: list[str] | None = None,
    selected_family: str = "ablation",
    selected_model_name: str = "random_forest",
    selected_feature_group: str | object = "base_plus_relative",
) -> pd.DataFrame:
    count = len(actual)
    return pd.DataFrame(
        {
            "selection_mode": [mode] * count,
            "fold_id": [1] * count,
            "season": [2024] * count,
            "event": ["Monza"] * count,
            "event_slug": ["2024_monza"] * count,
            "checkpoint": checkpoints or ["after_fp1"] * count,
            "driver": drivers or ["NOR", "LEC", "HAM"][:count],
            "team": ["McLaren", "Ferrari", "Mercedes"][:count],
            "quali_gap_to_pole_sec": actual,
            "predicted_quali_gap_to_pole_sec": predicted,
            "selected_family": [selected_family] * count,
            "selected_model_name": [selected_model_name] * count,
            "selected_feature_group": [selected_feature_group] * count,
        }
    )


def _interval_predictions(
    *,
    actual: list[float],
    predicted: list[float],
    low: list[float],
    high: list[float],
    contains: list[bool],
) -> pd.DataFrame:
    count = len(actual)
    return pd.DataFrame(
        {
            "season": [2024] * count,
            "event": ["Monza"] * count,
            "fold_id": [1] * count,
            "checkpoint": ["after_fp3"] * count,
            "driver": ["NOR", "LEC", "HAM", "PIA"][:count],
            "team": ["McLaren", "Ferrari", "Mercedes", "McLaren"][:count],
            "selected_family": ["ablation"] * count,
            "selected_model_name": ["random_forest"] * count,
            "selected_feature_group": ["base_plus_relative"] * count,
            "quali_gap_to_pole_sec": actual,
            "predicted_quali_gap_to_pole_sec": predicted,
            "prediction_interval_low_sec": low,
            "prediction_interval_high_sec": high,
            "interval_contains_actual": contains,
            "residual_count": [20] * count,
            "residual_quantile_sec": [0.5] * count,
        }
    )


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
