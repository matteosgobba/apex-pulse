from pathlib import Path

import pytest

from f1_prediction.config import (
    ConfigError,
    load_data_config,
    load_feature_config,
    load_model_config,
    load_yaml_config,
)


def test_load_data_config_resolves_project_relative_paths(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "data.yaml"
    config_path.write_text(
        """
paths:
  fastf1_cache_dir: data/raw/cache
  lap_output_dir: data/raw/laps
  session_metadata_output_dir: data/raw/metadata
  clean_lap_output_dir: data/interim/clean_laps
  session_features_output_dir: data/processed/session_features
  modeling_output_dir: data/processed/modeling
  metrics_output_dir: reports/metrics
fastf1:
  load_telemetry: false
  load_weather: true
  load_messages: false
""".strip(),
        encoding="utf-8",
    )

    config = load_data_config(config_path=config_path, project_root=tmp_path)

    assert config.fastf1_cache_dir == (tmp_path / "data/raw/cache").resolve()
    assert config.lap_output_dir == (tmp_path / "data/raw/laps").resolve()
    assert config.session_metadata_output_dir == (tmp_path / "data/raw/metadata").resolve()
    assert config.clean_lap_output_dir == (tmp_path / "data/interim/clean_laps").resolve()
    assert (
        config.session_features_output_dir
        == (tmp_path / "data/processed/session_features").resolve()
    )
    assert config.modeling_output_dir == (tmp_path / "data/processed/modeling").resolve()
    assert config.metrics_output_dir == (tmp_path / "reports/metrics").resolve()
    assert config.load_telemetry is False
    assert config.load_weather is True
    assert config.load_messages is False


def test_load_yaml_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="root must be a mapping"):
        load_yaml_config(config_path)


def test_load_feature_config(tmp_path: Path) -> None:
    config_path = tmp_path / "features.yaml"
    config_path.write_text(
        """
push_lap:
  driver_best_pct_threshold: 1.03
  session_best_pct_threshold: 1.07
  allowed_compounds: [soft, medium]
""".strip(),
        encoding="utf-8",
    )

    config = load_feature_config(config_path=config_path, project_root=tmp_path)

    assert config.push_lap.driver_best_pct_threshold == 1.03
    assert config.push_lap.session_best_pct_threshold == 1.07
    assert config.push_lap.allowed_compounds == ("SOFT", "MEDIUM")


def test_load_model_config_reads_tabular_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        """
model:
  min_events: 6
  random_state: 7
  ridge_alpha: 2.0
  random_forest:
    n_estimators: 50
    max_depth: 4
    min_samples_leaf: 3
  models:
    hist_gradient_boosting:
      enabled: true
      max_iter: 75
      learning_rate: 0.08
      max_leaf_nodes: 15
      l2_regularization: 0.2
      random_state: 9
  feature_group_policy:
    after_fp1: base_lap_features
    after_fp2: base_plus_quality
    after_fp3: base_plus_relative
  champion_policy:
    selection_metric: mae_gap_sec
    stabilized_nested:
      min_prior_folds: 4
      min_prior_predictions: 80
      selection_metric: mae_gap_sec
      improvement_margin_sec: 0.07
    stabilized_nested_guarded:
      base_mode: stabilized_nested
      fp3_no_baseline_switch: true
      guarded_checkpoint: after_fp3
      guarded_default_family: ablation
      guarded_default_model_name: random_forest
      guarded_default_feature_group: base_plus_relative
    static:
      after_fp1:
        family: robust_baseline
        model_name: robust_best_push_lap
  uncertainty:
    interval_z: 1.5
    confidence_level: 0.85
    min_residual_count: 12
    predicted_gap_bucket:
      confidence_level: 0.91
      min_residual_count: 14
      bucket_thresholds_sec:
        pole_contender: 0.4
        close_midfield: 1.4
        midfield: 2.8
      fallback_order:
        - checkpoint_method_bucket
        - checkpoint_bucket
        - global
  champion_diagnostics:
    harmful_switch_tolerance_sec: 0.08
  policy_simulation:
    conformal:
      confidence_level: 0.88
      min_residual_count: 9
      fallback_order:
        - checkpoint_method_bucket
        - checkpoint
        - global
""".strip(),
        encoding="utf-8",
    )

    config = load_model_config(config_path=config_path, project_root=tmp_path)

    assert config.min_events == 6
    assert config.random_state == 7
    assert config.ridge_alpha == 2.0
    assert config.random_forest.n_estimators == 50
    assert config.hist_gradient_boosting.max_iter == 75
    assert config.hist_gradient_boosting.learning_rate == 0.08
    assert config.hist_gradient_boosting.random_state == 9
    assert config.feature_group_policy.after_fp2 == "base_plus_quality"
    assert config.champion_policy.static["after_fp1"].model_name == "robust_best_push_lap"
    assert config.champion_policy.static["after_fp3"].feature_group == "base_plus_relative"
    assert config.champion_policy.stabilized_nested.min_prior_folds == 4
    assert config.champion_policy.stabilized_nested.min_prior_predictions == 80
    assert config.champion_policy.stabilized_nested.improvement_margin_sec == 0.07
    guarded = config.champion_policy.stabilized_nested_guarded
    assert guarded.base_mode == "stabilized_nested"
    assert guarded.fp3_no_baseline_switch is True
    assert guarded.guarded_checkpoint == "after_fp3"
    assert guarded.guarded_default_family == "ablation"
    assert guarded.guarded_default_model_name == "random_forest"
    assert guarded.guarded_default_feature_group == "base_plus_relative"
    assert config.uncertainty.interval_z == 1.5
    assert config.uncertainty.confidence_level == 0.85
    assert config.uncertainty.min_residual_count == 12
    bucket_uncertainty = config.uncertainty.predicted_gap_bucket
    assert bucket_uncertainty.confidence_level == 0.91
    assert bucket_uncertainty.min_residual_count == 14
    assert bucket_uncertainty.bucket_thresholds_sec == {
        "pole_contender": 0.4,
        "close_midfield": 1.4,
        "midfield": 2.8,
    }
    assert bucket_uncertainty.fallback_order == (
        "checkpoint_method_bucket",
        "checkpoint_bucket",
        "global",
    )
    assert config.champion_diagnostics.harmful_switch_tolerance_sec == 0.08
    assert config.policy_simulation.conformal.confidence_level == 0.88
    assert config.policy_simulation.conformal.min_residual_count == 9
    assert config.policy_simulation.conformal.fallback_order == (
        "checkpoint_method_bucket",
        "checkpoint",
        "global",
    )
