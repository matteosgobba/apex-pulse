"""Configuration loading for the data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
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
class FeatureConfig:
    """Feature engineering configuration."""

    push_lap: PushLapConfig


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

    return DataConfig(
        project_root=root,
        fastf1_cache_dir=resolve_project_path(cache_value, root),
        lap_output_dir=resolve_project_path(output_value, root),
        session_metadata_output_dir=resolve_project_path(metadata_output_value, root),
        clean_lap_output_dir=resolve_project_path(clean_output_value, root),
        session_features_output_dir=resolve_project_path(features_output_value, root),
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

    return FeatureConfig(
        push_lap=PushLapConfig(
            driver_best_pct_threshold=driver_threshold,
            session_best_pct_threshold=session_threshold,
            allowed_compounds=tuple(compound.upper() for compound in allowed_compounds),
        )
    )


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


def _required_string_list(config: dict[str, Any], key: str, path: Path) -> list[str]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"'{key}' must be a non-empty list in {path}")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ConfigError(f"'{key}' must contain only non-empty strings in {path}")
    return value
