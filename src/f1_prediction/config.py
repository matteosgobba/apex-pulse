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

    fastf1_cache_dir: Path
    lap_output_dir: Path
    load_telemetry: bool = False
    load_weather: bool = False
    load_messages: bool = False


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

    return DataConfig(
        fastf1_cache_dir=resolve_project_path(cache_value, root),
        lap_output_dir=resolve_project_path(output_value, root),
        load_telemetry=_boolean_option(fastf1_options, "load_telemetry", path),
        load_weather=_boolean_option(fastf1_options, "load_weather", path),
        load_messages=_boolean_option(fastf1_options, "load_messages", path),
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
