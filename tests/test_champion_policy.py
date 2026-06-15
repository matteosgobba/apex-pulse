import json
from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.config import (
    ChampionMethodConfig,
    ChampionPolicyConfig,
    DataConfig,
    ModelConfig,
    RandomForestConfig,
    UncertaintyConfig,
)
from f1_prediction.modeling.backtest_tabular import BacktestStrategy, build_backtest_folds
from f1_prediction.modeling.champion_policy import (
    ChampionSelectionMode,
    add_prior_residual_uncertainty,
    build_champion_metrics_payload,
    resolve_static_champion_policy,
    run_champion_backtest,
    select_nested_method,
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
    assert set(result["uncertainty_method"]) == {"prior_residual_std"}
    assert result["prediction_interval_low_sec"].notna().all()
    assert result["prediction_interval_high_sec"].notna().all()


def test_uncertainty_is_null_when_history_is_insufficient() -> None:
    result = add_prior_residual_uncertainty(
        _champion_rows(1, "method-a"),
        _candidate_rows(1, "event-1", "baseline", "method-a", [0.1, 0.2]),
        UncertaintyConfig(interval_z=1.64, min_residual_count=20),
    )

    assert result["prediction_interval_low_sec"].isna().all()
    assert result["prediction_interval_high_sec"].isna().all()
    assert set(result["uncertainty_method"]) == {"insufficient_history"}


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
        "prediction_interval_low_sec",
        "prediction_interval_high_sec",
        "uncertainty_method",
    }
    assert summary.n_folds_successful == len(folds)
    assert prediction_columns <= set(predictions.columns)
    assert set(selection["checkpoint"]) == {"after_fp1", "after_fp2", "after_fp3"}


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
            "predicted_quali_gap_to_pole_sec": [0.4, 0.6],
        }
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
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    event_to_fold = {fold.test_event: fold.fold_id for fold in folds}
    event_keys = dataset["season"].astype(str) + "/" + dataset["event_slug"]
    test = dataset[event_keys.isin(event_to_fold)].copy()
    test["test_event"] = event_keys[test.index]
    test["fold_id"] = test["test_event"].map(event_to_fold)
    base_columns = list(test.columns)
    baseline_frames = []
    for model_name in ("robust_best_push_lap", "robust_theoretical_best_lap"):
        frame = test.loc[:, base_columns].copy()
        frame["model_name"] = model_name
        frame["baseline_name"] = model_name
        frame["prediction_type"] = "baseline"
        frame["strategy"] = "walk_forward"
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


def _model_config() -> ModelConfig:
    return ModelConfig(
        min_events=5,
        random_state=42,
        ridge_alpha=1.0,
        random_forest=RandomForestConfig(5, 3, 1),
        champion_policy=ChampionPolicyConfig(),
        uncertainty=UncertaintyConfig(interval_z=1.64, min_residual_count=2),
    )
