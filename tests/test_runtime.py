from pathlib import Path

import pytest
from typer.testing import CliRunner

from f1_prediction.cli import app
from f1_prediction.config import load_data_config
from f1_prediction.dashboard_api.app import create_dashboard_app
from f1_prediction.data.cache import initialize_fastf1_cache
from f1_prediction.runtime import (
    RuntimeConfigurationError,
    create_production_runtime_report,
    initialize_runtime_layout,
    resolve_runtime_layout,
)


def test_local_runtime_paths_and_initialization_remain_unchanged(tmp_path: Path) -> None:
    project_root = _write_project(tmp_path / "project")
    before = sorted(project_root.rglob("*"))

    layout = initialize_runtime_layout(project_root=project_root, environ={})

    assert layout.runtime_mode == "local"
    assert layout.runtime_root is None
    assert layout.data_path == project_root / "data"
    assert layout.reports_path == project_root / "reports"
    assert layout.models_path == project_root / "models"
    assert sorted(project_root.rglob("*")) == before


def test_configured_runtime_root_links_mutable_trees_idempotently_and_preserves_files(
    tmp_path: Path,
) -> None:
    project_root = _write_project(tmp_path / "project", mutable_trees=True)
    runtime_root = tmp_path / "runtime"
    preserved = runtime_root / "reports/metrics/preserved.json"
    preserved.parent.mkdir(parents=True)
    preserved.write_text('{"immutable": true}\n', encoding="utf-8")

    first = initialize_runtime_layout(project_root=project_root, runtime_root=runtime_root)
    second = initialize_runtime_layout(project_root=project_root, runtime_root=runtime_root)

    assert first == second
    assert first.runtime_mode == "persistent"
    for tree in ("data", "reports", "models"):
        link = project_root / tree
        assert link.is_symlink()
        assert link.resolve() == (runtime_root / tree).resolve()
    assert preserved.read_text(encoding="utf-8") == '{"immutable": true}\n'


def test_runtime_initialization_refuses_to_replace_nonempty_application_tree(
    tmp_path: Path,
) -> None:
    project_root = _write_project(tmp_path / "project", mutable_trees=True)
    local_artifact = project_root / "data/local.parquet"
    local_artifact.write_bytes(b"preserve-me")

    with pytest.raises(RuntimeConfigurationError, match="does not migrate local artifacts"):
        initialize_runtime_layout(
            project_root=project_root,
            runtime_root=tmp_path / "runtime",
        )

    assert local_artifact.read_bytes() == b"preserve-me"


def test_fastf1_cache_resolves_under_persistent_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root = _write_project(tmp_path / "project", mutable_trees=True)
    runtime_root = tmp_path / "runtime"
    initialize_runtime_layout(project_root=project_root, runtime_root=runtime_root)

    config = load_data_config(
        project_root / "configs/data.yaml",
        project_root=project_root,
    )
    enabled_paths: list[str] = []
    monkeypatch.setattr("fastf1.Cache.enable_cache", enabled_paths.append)
    initialized_path = initialize_fastf1_cache(config.fastf1_cache_dir)

    assert config.fastf1_cache_dir == (runtime_root / "data/raw/fastf1_cache").resolve()
    assert initialized_path == config.fastf1_cache_dir
    assert enabled_paths == [str(config.fastf1_cache_dir)]


def test_production_runtime_report_is_read_only_and_reports_unseeded_runtime(
    tmp_path: Path,
) -> None:
    project_root = _write_project(tmp_path / "project")
    runtime_root = tmp_path / "runtime"

    report = create_production_runtime_report(
        project_root=project_root,
        runtime_root=runtime_root,
        environ={},
    )

    assert report.runtime_mode == "persistent"
    assert report.runtime_root == str(runtime_root.resolve())
    assert report.fastf1_cache_path == str((runtime_root / "data/raw/fastf1_cache").resolve())
    assert report.dashboard_path == str((runtime_root / "reports/dashboard").resolve())
    assert report.runtime_root_exists is False
    assert report.runtime_root_writable is None
    assert report.dashboard_artifacts_present is False
    assert report.monitoring_protocol_present is False
    assert report.historical_modeling_dataset_present is False
    assert report.status == "not_seeded"
    assert not runtime_root.exists()


def test_production_runtime_report_does_not_treat_unreceipted_files_as_seeded(
    tmp_path: Path,
) -> None:
    project_root = _write_project(tmp_path / "project")
    runtime_root = tmp_path / "runtime"
    _seed_runtime(runtime_root)

    report = create_production_runtime_report(
        project_root=project_root,
        runtime_root=runtime_root,
        environ={},
    )

    assert report.runtime_root_exists is True
    assert report.runtime_root_writable is True
    assert report.dashboard_artifacts_present is True
    assert report.monitoring_protocol_present is True
    assert report.historical_modeling_dataset_present is True
    assert report.bootstrap_receipt_present is False
    assert report.ready_for_api is False
    assert report.ready_for_future_monitoring is False
    assert report.status == "not_seeded"


def test_production_runtime_check_cli_reports_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root = _write_project(tmp_path / "project")
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("F1_PREDICTION_PROJECT_ROOT", str(project_root))
    monkeypatch.setenv("APEX_PULSE_RUNTIME_ROOT", str(runtime_root))

    result = CliRunner().invoke(app, ["production-runtime-check"])

    assert result.exit_code == 0
    assert "runtime_mode=persistent" in result.stdout
    assert "status=not_seeded" in result.stdout
    assert not runtime_root.exists()


def test_production_runtime_check_cli_reports_unreceipted_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root = _write_project(tmp_path / "project")
    runtime_root = tmp_path / "runtime"
    _seed_runtime(runtime_root)
    monkeypatch.setenv("F1_PREDICTION_PROJECT_ROOT", str(project_root))
    monkeypatch.setenv("APEX_PULSE_RUNTIME_ROOT", str(runtime_root))

    result = CliRunner().invoke(app, ["production-runtime-check"])

    assert result.exit_code == 0
    assert "dashboard_artifacts_present=true" in result.stdout
    assert "monitoring_protocol_present=true" in result.stdout
    assert "historical_modeling_dataset_present=true" in result.stdout
    assert "bootstrap_receipt_present=false" in result.stdout
    assert "ready_for_api=false" in result.stdout
    assert "status=not_seeded" in result.stdout


def test_runtime_initialization_and_health_do_not_mutate_monitoring_artifacts(
    tmp_path: Path,
) -> None:
    project_root = _write_project(tmp_path / "project", mutable_trees=True)
    runtime_root = tmp_path / "runtime"
    monitoring_path = runtime_root / "reports/metrics/prospective_monitoring_forecasts.parquet"
    monitoring_path.parent.mkdir(parents=True)
    monitoring_path.write_bytes(b"immutable-forecast-fixture")
    before = monitoring_path.read_bytes()

    initialize_runtime_layout(project_root=project_root, runtime_root=runtime_root)
    application = create_dashboard_app(runtime_root / "reports/dashboard")
    health_route = next(route for route in application.routes if route.path == "/api/v1/health")
    response = health_route.endpoint()

    assert response.status == "ok"
    assert response.dashboard_artifact_status == "unavailable"
    assert monitoring_path.read_bytes() == before


def test_runtime_root_must_be_absolute_and_separate_from_application(tmp_path: Path) -> None:
    project_root = _write_project(tmp_path / "project")

    with pytest.raises(RuntimeConfigurationError, match="absolute path"):
        resolve_runtime_layout(project_root=project_root, runtime_root="runtime")
    with pytest.raises(RuntimeConfigurationError, match="application directory"):
        resolve_runtime_layout(project_root=project_root, runtime_root=tmp_path)


def _write_project(project_root: Path, *, mutable_trees: bool = False) -> Path:
    project_root.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text(
        '[project]\nname = "runtime-test"\nversion = "0.0.0"\n',
        encoding="utf-8",
    )
    config_dir = project_root / "configs"
    config_dir.mkdir()
    (config_dir / "data.yaml").write_text(
        """paths:
  fastf1_cache_dir: data/raw/fastf1_cache
  lap_output_dir: data/raw/laps
  session_metadata_output_dir: data/raw/session_metadata
  clean_lap_output_dir: data/interim/clean_laps
  session_features_output_dir: data/processed/session_features
  modeling_output_dir: data/processed/modeling
  metrics_output_dir: reports/metrics
fastf1:
  load_telemetry: false
  load_weather: false
  load_messages: false
""",
        encoding="utf-8",
    )
    if mutable_trees:
        for tree in ("data", "reports", "models"):
            (project_root / tree).mkdir()
    return project_root.resolve()


def _seed_runtime(runtime_root: Path) -> None:
    dashboard_dir = runtime_root / "reports/dashboard"
    dashboard_dir.mkdir(parents=True)
    for filename in (
        "dashboard_manifest.json",
        "current_event.json",
        "event_forecast.json",
        "event_settlement.json",
        "event_practice_status.json",
        "historical_monitoring_summary.json",
        "model_summary.json",
    ):
        (dashboard_dir / filename).write_text("{}\n", encoding="utf-8")
    protocol_path = runtime_root / "reports/metrics/prospective_monitoring_protocol.json"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("{}\n", encoding="utf-8")
    dataset_path = runtime_root / "data/processed/modeling/combined/modeling_dataset.parquet"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_bytes(b"synthetic-test-placeholder")
