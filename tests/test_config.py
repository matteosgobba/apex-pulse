from pathlib import Path

import pytest

from f1_prediction.config import (
    ConfigError,
    load_data_config,
    load_feature_config,
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
