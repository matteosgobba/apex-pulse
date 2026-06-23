"""Configuration loading for the data pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from f1_prediction.utils.paths import get_project_root, resolve_project_path


class ConfigError(ValueError):
    """Raised when a project configuration file is missing or invalid."""


@dataclass(frozen=True)
class DataConfig:
    """Resolved paths and FastF1 loading options."""

    project_root: Path
    fastf1_cache_dir: Path
    lap_output_dir: Path
    session_metadata_output_dir: Path
    clean_lap_output_dir: Path
    session_features_output_dir: Path
    modeling_output_dir: Path
    metrics_output_dir: Path
    load_telemetry: bool = False
    load_weather: bool = False
    load_messages: bool = False


@dataclass(frozen=True)
class PushLapConfig:
    """Thresholds used by rule-based push-lap detection."""

    driver_best_pct_threshold: float
    session_best_pct_threshold: float
    allowed_compounds: tuple[str, ...]


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Thresholds and output limits for prediction diagnostics."""

    extreme_mae_gap_threshold_sec: float
    top_n: int


@dataclass(frozen=True)
class HistoricalFeaturesConfig:
    """Leakage-safe historical rolling feature settings."""

    rolling_windows: tuple[int, ...]
    min_periods: int


@dataclass(frozen=True)
class DataQualityConfig:
    """Practice signal quality thresholds."""

    extreme_gap_to_session_best_sec: float
    min_push_laps_latest_session: int
    min_valid_laps_latest_session: int


@dataclass(frozen=True)
class BaselineConfig:
    """Robust practice baseline settings."""

    robust_extreme_gap_to_session_best_sec: float


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering configuration."""

    push_lap: PushLapConfig
    diagnostics: DiagnosticsConfig | None = None
    historical_features: HistoricalFeaturesConfig | None = None
    data_quality: DataQualityConfig | None = None
    baselines: BaselineConfig | None = None


@dataclass(frozen=True)
class RandomForestConfig:
    """Conservative Random Forest settings for the first tabular model."""

    n_estimators: int
    max_depth: int
    min_samples_leaf: int


@dataclass(frozen=True)
class HistGradientBoostingConfig:
    """Conservative histogram gradient boosting settings."""

    enabled: bool = True
    max_iter: int = 200
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    l2_regularization: float = 0.1
    random_state: int = 42


@dataclass(frozen=True)
class FeatureGroupPolicyConfig:
    """Default feature group selected for each prediction checkpoint."""

    after_fp1: str = "base_lap_features"
    after_fp2: str = "base_plus_quality"
    after_fp3: str = "base_plus_relative"


@dataclass(frozen=True)
class ChampionMethodConfig:
    """One configured checkpoint champion method."""

    family: str
    model_name: str
    feature_group: str | None = None


@dataclass(frozen=True)
class StabilizedNestedChampionConfig:
    """Stabilized nested champion selection guardrails."""

    min_prior_folds: int = 5
    min_prior_predictions: int = 100
    selection_metric: str = "mae_gap_sec"
    improvement_margin_sec: float = 0.05


@dataclass(frozen=True)
class StabilizedNestedGuardedChampionConfig:
    """Opt-in FP3 guardrail settings layered on stabilized nested selection."""

    base_mode: str = "stabilized_nested"
    fp3_no_baseline_switch: bool = True
    guarded_checkpoint: str = "after_fp3"
    guarded_default_family: str = "ablation"
    guarded_default_model_name: str = "random_forest"
    guarded_default_feature_group: str | None = "base_plus_relative"


def _default_champion_static_policy() -> dict[str, ChampionMethodConfig]:
    return {
        "after_fp1": ChampionMethodConfig(
            family="robust_baseline",
            model_name="robust_best_push_lap",
        ),
        "after_fp2": ChampionMethodConfig(
            family="robust_baseline",
            model_name="robust_theoretical_best_lap",
        ),
        "after_fp3": ChampionMethodConfig(
            family="ablation",
            model_name="random_forest",
            feature_group="base_plus_relative",
        ),
    }


@dataclass(frozen=True)
class ChampionPolicyConfig:
    """Static fallback methods and nested selection criterion."""

    static: dict[str, ChampionMethodConfig] = field(default_factory=_default_champion_static_policy)
    selection_metric: str = "mae_gap_sec"
    stabilized_nested: StabilizedNestedChampionConfig = field(
        default_factory=StabilizedNestedChampionConfig
    )
    stabilized_nested_guarded: StabilizedNestedGuardedChampionConfig = field(
        default_factory=StabilizedNestedGuardedChampionConfig
    )


@dataclass(frozen=True)
class UncertaintyConfig:
    """Prior-residual prediction interval settings."""

    interval_z: float = 1.64
    confidence_level: float = 0.90
    min_residual_count: int = 20


@dataclass(frozen=True)
class ChampionDiagnosticsConfig:
    """Champion policy diagnostic thresholds."""

    harmful_switch_tolerance_sec: float = 0.05


@dataclass(frozen=True)
class PolicySimulationConformalConfig:
    """Regime-aware conformal simulation settings."""

    confidence_level: float = 0.90
    min_residual_count: int = 20
    fallback_order: tuple[str, ...] = (
        "checkpoint_method_bucket",
        "checkpoint_bucket",
        "checkpoint_method",
        "checkpoint",
        "global",
    )


@dataclass(frozen=True)
class PolicySimulationConfig:
    """Artifact-based policy simulation settings."""

    conformal: PolicySimulationConformalConfig = field(
        default_factory=PolicySimulationConformalConfig
    )


@dataclass(frozen=True)
class ModelConfig:
    """Simple tabular model and dataset-size settings."""

    min_events: int
    random_state: int
    ridge_alpha: float
    random_forest: RandomForestConfig
    hist_gradient_boosting: HistGradientBoostingConfig = field(
        default_factory=HistGradientBoostingConfig
    )
    feature_group_policy: FeatureGroupPolicyConfig = field(default_factory=FeatureGroupPolicyConfig)
    champion_policy: ChampionPolicyConfig = field(default_factory=ChampionPolicyConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    champion_diagnostics: ChampionDiagnosticsConfig = field(
        default_factory=ChampionDiagnosticsConfig
    )
    policy_simulation: PolicySimulationConfig = field(default_factory=PolicySimulationConfig)


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping at its root."""
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")

    with path.open(encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file)

    if not isinstance(data, dict):
        raise ConfigError(f"Configuration root must be a mapping: {path}")

    return data


def load_data_config(
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> DataConfig:
    """Load and resolve the data configuration relative to the project root."""
    root = (project_root or get_project_root()).resolve()
    path = config_path or root / "configs" / "data.yaml"
    if not path.is_absolute():
        path = root / path

    raw_config = load_yaml_config(path)
    paths = _required_mapping(raw_config, "paths", path)
    fastf1_options = raw_config.get("fastf1", {})
    if not isinstance(fastf1_options, dict):
        raise ConfigError(f"'fastf1' must be a mapping in {path}")

    cache_value = _required_string(paths, "fastf1_cache_dir", path)
    output_value = _required_string(paths, "lap_output_dir", path)
    metadata_output_value = _required_string(paths, "session_metadata_output_dir", path)
    clean_output_value = _required_string(paths, "clean_lap_output_dir", path)
    features_output_value = _required_string(paths, "session_features_output_dir", path)
    modeling_output_value = _required_string(paths, "modeling_output_dir", path)
    metrics_output_value = _required_string(paths, "metrics_output_dir", path)

    return DataConfig(
        project_root=root,
        fastf1_cache_dir=resolve_project_path(cache_value, root),
        lap_output_dir=resolve_project_path(output_value, root),
        session_metadata_output_dir=resolve_project_path(metadata_output_value, root),
        clean_lap_output_dir=resolve_project_path(clean_output_value, root),
        session_features_output_dir=resolve_project_path(features_output_value, root),
        modeling_output_dir=resolve_project_path(modeling_output_value, root),
        metrics_output_dir=resolve_project_path(metrics_output_value, root),
        load_telemetry=_boolean_option(fastf1_options, "load_telemetry", path),
        load_weather=_boolean_option(fastf1_options, "load_weather", path),
        load_messages=_boolean_option(fastf1_options, "load_messages", path),
    )


def load_feature_config(
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> FeatureConfig:
    """Load feature engineering thresholds from YAML."""
    root = (project_root or get_project_root()).resolve()
    path = config_path or root / "configs" / "features.yaml"
    if not path.is_absolute():
        path = root / path

    raw_config = load_yaml_config(path)
    push_lap = _required_mapping(raw_config, "push_lap", path)
    driver_threshold = _required_number(push_lap, "driver_best_pct_threshold", path)
    session_threshold = _required_number(push_lap, "session_best_pct_threshold", path)
    allowed_compounds = _required_string_list(push_lap, "allowed_compounds", path)

    if driver_threshold < 1.0 or session_threshold < 1.0:
        raise ConfigError(f"Push-lap percentage thresholds must be at least 1.0 in {path}")

    diagnostics_raw = raw_config.get("diagnostics", {})
    if not isinstance(diagnostics_raw, dict):
        raise ConfigError(f"'diagnostics' must be a mapping in {path}")
    diagnostics = DiagnosticsConfig(
        extreme_mae_gap_threshold_sec=float(
            diagnostics_raw.get("extreme_mae_gap_threshold_sec", 1.5)
        ),
        top_n=int(diagnostics_raw.get("top_n", 10)),
    )
    if diagnostics.extreme_mae_gap_threshold_sec <= 0 or diagnostics.top_n < 1:
        raise ConfigError(f"Diagnostics threshold and top_n must be positive in {path}")

    historical_raw = raw_config.get("historical_features", {})
    if not isinstance(historical_raw, dict):
        raise ConfigError(f"'historical_features' must be a mapping in {path}")
    rolling_windows_raw = historical_raw.get("rolling_windows", [3, 5])
    if (
        not isinstance(rolling_windows_raw, list)
        or not rolling_windows_raw
        or not all(isinstance(window, int) and window > 0 for window in rolling_windows_raw)
    ):
        raise ConfigError(f"Historical rolling_windows must be positive integers in {path}")
    historical = HistoricalFeaturesConfig(
        rolling_windows=tuple(rolling_windows_raw),
        min_periods=int(historical_raw.get("min_periods", 1)),
    )
    if historical.min_periods < 1:
        raise ConfigError(f"Historical min_periods must be positive in {path}")

    quality_raw = raw_config.get("data_quality", {})
    if not isinstance(quality_raw, dict):
        raise ConfigError(f"'data_quality' must be a mapping in {path}")
    data_quality = DataQualityConfig(
        extreme_gap_to_session_best_sec=float(
            quality_raw.get("extreme_gap_to_session_best_sec", 3.0)
        ),
        min_push_laps_latest_session=int(quality_raw.get("min_push_laps_latest_session", 2)),
        min_valid_laps_latest_session=int(quality_raw.get("min_valid_laps_latest_session", 5)),
    )
    if (
        data_quality.extreme_gap_to_session_best_sec <= 0
        or data_quality.min_push_laps_latest_session < 1
        or data_quality.min_valid_laps_latest_session < 1
    ):
        raise ConfigError(f"Data-quality thresholds must be positive in {path}")

    baselines_raw = raw_config.get("baselines", {})
    if not isinstance(baselines_raw, dict):
        raise ConfigError(f"'baselines' must be a mapping in {path}")
    baselines = BaselineConfig(
        robust_extreme_gap_to_session_best_sec=float(
            baselines_raw.get("robust_extreme_gap_to_session_best_sec", 3.0)
        )
    )
    if baselines.robust_extreme_gap_to_session_best_sec <= 0:
        raise ConfigError(f"Robust baseline threshold must be positive in {path}")

    return FeatureConfig(
        push_lap=PushLapConfig(
            driver_best_pct_threshold=driver_threshold,
            session_best_pct_threshold=session_threshold,
            allowed_compounds=tuple(compound.upper() for compound in allowed_compounds),
        ),
        diagnostics=diagnostics,
        historical_features=historical,
        data_quality=data_quality,
        baselines=baselines,
    )


def load_model_config(
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> ModelConfig:
    """Load conservative first-model settings from YAML."""
    root = (project_root or get_project_root()).resolve()
    path = config_path or root / "configs" / "model.yaml"
    if not path.is_absolute():
        path = root / path
    raw_config = load_yaml_config(path)
    model = _required_mapping(raw_config, "model", path)
    random_forest = _required_mapping(model, "random_forest", path)
    models = model.get("models", {})
    if not isinstance(models, dict):
        raise ConfigError(f"'models' must be a mapping in {path}")
    hist_gradient_boosting = models.get("hist_gradient_boosting", {})
    if not isinstance(hist_gradient_boosting, dict):
        raise ConfigError(f"'hist_gradient_boosting' must be a mapping in {path}")
    feature_group_policy = model.get("feature_group_policy", {})
    if not isinstance(feature_group_policy, dict):
        raise ConfigError(f"'feature_group_policy' must be a mapping in {path}")
    champion_policy = model.get("champion_policy", {})
    if not isinstance(champion_policy, dict):
        raise ConfigError(f"'champion_policy' must be a mapping in {path}")
    static_policy = champion_policy.get("static", {})
    if not isinstance(static_policy, dict):
        raise ConfigError(f"'champion_policy.static' must be a mapping in {path}")
    stabilized_nested = champion_policy.get("stabilized_nested", {})
    if not isinstance(stabilized_nested, dict):
        raise ConfigError(f"'champion_policy.stabilized_nested' must be a mapping in {path}")
    stabilized_nested_guarded = champion_policy.get("stabilized_nested_guarded", {})
    if not isinstance(stabilized_nested_guarded, dict):
        raise ConfigError(
            f"'champion_policy.stabilized_nested_guarded' must be a mapping in {path}"
        )
    uncertainty = model.get("uncertainty", {})
    if not isinstance(uncertainty, dict):
        raise ConfigError(f"'uncertainty' must be a mapping in {path}")
    champion_diagnostics = model.get("champion_diagnostics", {})
    if not isinstance(champion_diagnostics, dict):
        raise ConfigError(f"'champion_diagnostics' must be a mapping in {path}")
    policy_simulation = model.get("policy_simulation", {})
    if not isinstance(policy_simulation, dict):
        raise ConfigError(f"'policy_simulation' must be a mapping in {path}")
    policy_simulation_conformal = policy_simulation.get("conformal", {})
    if not isinstance(policy_simulation_conformal, dict):
        raise ConfigError(f"'policy_simulation.conformal' must be a mapping in {path}")
    default_boosting = HistGradientBoostingConfig(
        random_state=_required_integer(model, "random_state", path)
    )
    default_policy = FeatureGroupPolicyConfig()
    default_champion = ChampionPolicyConfig()
    default_guarded = default_champion.stabilized_nested_guarded
    default_uncertainty = UncertaintyConfig()
    default_champion_diagnostics = ChampionDiagnosticsConfig()
    default_policy_simulation = PolicySimulationConfig()
    fallback_order_raw = policy_simulation_conformal.get(
        "fallback_order",
        list(default_policy_simulation.conformal.fallback_order),
    )
    if (
        not isinstance(fallback_order_raw, list)
        or not fallback_order_raw
        or not all(isinstance(value, str) and value.strip() for value in fallback_order_raw)
    ):
        raise ConfigError(f"'policy_simulation.conformal.fallback_order' is invalid in {path}")
    config = ModelConfig(
        min_events=_required_integer(model, "min_events", path),
        random_state=_required_integer(model, "random_state", path),
        ridge_alpha=_required_number(model, "ridge_alpha", path),
        random_forest=RandomForestConfig(
            n_estimators=_required_integer(random_forest, "n_estimators", path),
            max_depth=_required_integer(random_forest, "max_depth", path),
            min_samples_leaf=_required_integer(random_forest, "min_samples_leaf", path),
        ),
        hist_gradient_boosting=HistGradientBoostingConfig(
            enabled=bool(hist_gradient_boosting.get("enabled", default_boosting.enabled)),
            max_iter=int(hist_gradient_boosting.get("max_iter", default_boosting.max_iter)),
            learning_rate=float(
                hist_gradient_boosting.get("learning_rate", default_boosting.learning_rate)
            ),
            max_leaf_nodes=int(
                hist_gradient_boosting.get("max_leaf_nodes", default_boosting.max_leaf_nodes)
            ),
            l2_regularization=float(
                hist_gradient_boosting.get("l2_regularization", default_boosting.l2_regularization)
            ),
            random_state=int(
                hist_gradient_boosting.get("random_state", default_boosting.random_state)
            ),
        ),
        feature_group_policy=FeatureGroupPolicyConfig(
            after_fp1=str(feature_group_policy.get("after_fp1", default_policy.after_fp1)),
            after_fp2=str(feature_group_policy.get("after_fp2", default_policy.after_fp2)),
            after_fp3=str(feature_group_policy.get("after_fp3", default_policy.after_fp3)),
        ),
        champion_policy=ChampionPolicyConfig(
            static={
                checkpoint: _load_champion_method(
                    static_policy.get(checkpoint),
                    default_champion.static[checkpoint],
                    checkpoint,
                    path,
                )
                for checkpoint in ("after_fp1", "after_fp2", "after_fp3")
            },
            selection_metric=str(
                champion_policy.get("selection_metric", default_champion.selection_metric)
            ),
            stabilized_nested=StabilizedNestedChampionConfig(
                min_prior_folds=int(
                    stabilized_nested.get(
                        "min_prior_folds",
                        default_champion.stabilized_nested.min_prior_folds,
                    )
                ),
                min_prior_predictions=int(
                    stabilized_nested.get(
                        "min_prior_predictions",
                        default_champion.stabilized_nested.min_prior_predictions,
                    )
                ),
                selection_metric=str(
                    stabilized_nested.get(
                        "selection_metric",
                        default_champion.stabilized_nested.selection_metric,
                    )
                ),
                improvement_margin_sec=float(
                    stabilized_nested.get(
                        "improvement_margin_sec",
                        default_champion.stabilized_nested.improvement_margin_sec,
                    )
                ),
            ),
            stabilized_nested_guarded=StabilizedNestedGuardedChampionConfig(
                base_mode=str(
                    stabilized_nested_guarded.get("base_mode", default_guarded.base_mode)
                ),
                fp3_no_baseline_switch=bool(
                    stabilized_nested_guarded.get(
                        "fp3_no_baseline_switch",
                        default_guarded.fp3_no_baseline_switch,
                    )
                ),
                guarded_checkpoint=str(
                    stabilized_nested_guarded.get(
                        "guarded_checkpoint",
                        default_guarded.guarded_checkpoint,
                    )
                ),
                guarded_default_family=str(
                    stabilized_nested_guarded.get(
                        "guarded_default_family",
                        default_guarded.guarded_default_family,
                    )
                ),
                guarded_default_model_name=str(
                    stabilized_nested_guarded.get(
                        "guarded_default_model_name",
                        default_guarded.guarded_default_model_name,
                    )
                ),
                guarded_default_feature_group=_optional_config_string(
                    stabilized_nested_guarded.get(
                        "guarded_default_feature_group",
                        default_guarded.guarded_default_feature_group,
                    )
                ),
            ),
        ),
        uncertainty=UncertaintyConfig(
            interval_z=float(uncertainty.get("interval_z", default_uncertainty.interval_z)),
            confidence_level=float(
                uncertainty.get("confidence_level", default_uncertainty.confidence_level)
            ),
            min_residual_count=int(
                uncertainty.get("min_residual_count", default_uncertainty.min_residual_count)
            ),
        ),
        champion_diagnostics=ChampionDiagnosticsConfig(
            harmful_switch_tolerance_sec=float(
                champion_diagnostics.get(
                    "harmful_switch_tolerance_sec",
                    default_champion_diagnostics.harmful_switch_tolerance_sec,
                )
            )
        ),
        policy_simulation=PolicySimulationConfig(
            conformal=PolicySimulationConformalConfig(
                confidence_level=float(
                    policy_simulation_conformal.get(
                        "confidence_level",
                        default_policy_simulation.conformal.confidence_level,
                    )
                ),
                min_residual_count=int(
                    policy_simulation_conformal.get(
                        "min_residual_count",
                        default_policy_simulation.conformal.min_residual_count,
                    )
                ),
                fallback_order=tuple(fallback_order_raw),
            )
        ),
    )
    if config.min_events < 2 or config.ridge_alpha <= 0:
        raise ConfigError(f"Model min_events and ridge_alpha must be positive in {path}")
    if (
        config.random_forest.n_estimators < 1
        or config.random_forest.max_depth < 1
        or config.random_forest.min_samples_leaf < 1
    ):
        raise ConfigError(f"Random Forest settings must be positive in {path}")
    boosting = config.hist_gradient_boosting
    if (
        boosting.max_iter < 1
        or boosting.learning_rate <= 0
        or boosting.max_leaf_nodes < 2
        or boosting.l2_regularization < 0
    ):
        raise ConfigError(f"HistGradientBoosting settings are invalid in {path}")
    if config.champion_policy.selection_metric != "mae_gap_sec":
        raise ConfigError(f"Only champion selection_metric 'mae_gap_sec' is supported in {path}")
    stabilized = config.champion_policy.stabilized_nested
    if (
        stabilized.selection_metric != "mae_gap_sec"
        or stabilized.min_prior_folds < 1
        or stabilized.min_prior_predictions < 1
        or stabilized.improvement_margin_sec < 0
    ):
        raise ConfigError(f"Stabilized nested champion settings are invalid in {path}")
    guarded = config.champion_policy.stabilized_nested_guarded
    if (
        guarded.base_mode != "stabilized_nested"
        or guarded.guarded_checkpoint not in {"after_fp1", "after_fp2", "after_fp3"}
        or not guarded.guarded_default_family
        or not guarded.guarded_default_model_name
    ):
        raise ConfigError(f"Guarded stabilized nested champion settings are invalid in {path}")
    if (
        config.uncertainty.interval_z <= 0
        or not 0 < config.uncertainty.confidence_level < 1
        or config.uncertainty.min_residual_count < 2
    ):
        raise ConfigError(f"Uncertainty settings are invalid in {path}")
    if config.champion_diagnostics.harmful_switch_tolerance_sec < 0:
        raise ConfigError(f"Champion diagnostics tolerance must be non-negative in {path}")
    if (
        not 0 < config.policy_simulation.conformal.confidence_level < 1
        or config.policy_simulation.conformal.min_residual_count < 2
    ):
        raise ConfigError(f"Policy simulation conformal settings are invalid in {path}")
    return config


def _load_champion_method(
    raw: object,
    default: ChampionMethodConfig,
    checkpoint: str,
    path: Path,
) -> ChampionMethodConfig:
    if raw is None:
        return default
    if not isinstance(raw, dict):
        raise ConfigError(f"Champion method for {checkpoint} must be a mapping in {path}")
    family = raw.get("family", default.family)
    model_name = raw.get("model_name", default.model_name)
    feature_group = raw.get("feature_group", default.feature_group)
    if not isinstance(family, str) or not family.strip():
        raise ConfigError(f"Champion family for {checkpoint} must be non-empty in {path}")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ConfigError(f"Champion model_name for {checkpoint} must be non-empty in {path}")
    if feature_group is not None and not isinstance(feature_group, str):
        raise ConfigError(f"Champion feature_group for {checkpoint} must be a string in {path}")
    return ChampionMethodConfig(
        family=family,
        model_name=model_name,
        feature_group=feature_group,
    )


def _optional_config_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return str(value)
    return value if value.strip() else None


def _required_mapping(config: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping in {path}")
    return value


def _required_string(config: dict[str, Any], key: str, path: Path) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{key}' must be a non-empty string in {path}")
    return value


def _boolean_option(config: dict[str, Any], key: str, path: Path) -> bool:
    value = config.get(key, False)
    if not isinstance(value, bool):
        raise ConfigError(f"'{key}' must be a boolean in {path}")
    return value


def _required_number(config: dict[str, Any], key: str, path: Path) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"'{key}' must be a number in {path}")
    return float(value)


def _required_integer(config: dict[str, Any], key: str, path: Path) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{key}' must be an integer in {path}")
    return value


def _required_string_list(config: dict[str, Any], key: str, path: Path) -> list[str]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"'{key}' must be a non-empty list in {path}")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ConfigError(f"'{key}' must contain only non-empty strings in {path}")
    return value
