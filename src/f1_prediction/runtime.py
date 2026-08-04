"""Production runtime layout initialization and read-only diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from f1_prediction.config import load_data_config
from f1_prediction.dashboard_api.service import (
    DASHBOARD_ARTIFACTS,
    DEFAULT_DASHBOARD_DIR,
)
from f1_prediction.utils.paths import get_project_root, resolve_project_path

RUNTIME_ROOT_ENV = "APEX_PULSE_RUNTIME_ROOT"
DASHBOARD_DIR_ENV = "APEX_PULSE_DASHBOARD_DIR"
MUTABLE_TREES = ("data", "reports", "models")
MONITORING_PROTOCOL_RELATIVE_PATH = Path("reports/metrics/prospective_monitoring_protocol.json")
HISTORICAL_DATASET_RELATIVE_PATH = Path("data/processed/modeling/combined/modeling_dataset.parquet")


class RuntimeConfigurationError(ValueError):
    """Raised when a configured persistent runtime cannot be initialized safely."""


@dataclass(frozen=True)
class RuntimeLayout:
    """Resolved mutable-tree locations for local or persistent execution."""

    runtime_mode: str
    project_root: Path
    runtime_root: Path | None
    data_path: Path
    reports_path: Path
    models_path: Path


@dataclass(frozen=True)
class ProductionRuntimeReport:
    """Read-only deployment readiness facts for the configured runtime."""

    runtime_mode: str
    runtime_root: str | None
    data_path: str
    reports_path: str
    models_path: str
    dashboard_path: str
    fastf1_cache_path: str
    runtime_root_exists: bool
    runtime_root_writable: bool | None
    dashboard_artifacts_present: bool
    monitoring_protocol_present: bool
    historical_modeling_dataset_present: bool
    bootstrap_receipt_present: bool
    bootstrap_status: str
    bundle_fingerprint: str | None
    manifest_fingerprint: str | None
    production_state_file_count: int
    production_state_integrity_status: str
    bootstrap_seed_status: str
    static_seed_invariants_status: str
    live_operational_state_status: str
    critical_artifact_integrity_status: str
    protocol_fingerprint_status: str
    dashboard_bundle_status: str
    forecast_artifact_status: str
    settlement_artifact_status: str
    modeling_dataset_status: str
    ready_for_api: bool
    ready_for_future_monitoring: bool
    fastf1_cache_bytes: int
    runtime_total_known_bytes: int
    volume_capacity_bytes: int | None
    cache_warning_status: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        """Return a stable serialization for CLI output and tests."""
        return asdict(self)


def resolve_runtime_layout(
    project_root: Path | None = None,
    runtime_root: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> RuntimeLayout:
    """Resolve local or persistent mutable-tree locations without changing the filesystem."""
    root = (project_root or get_project_root()).resolve()
    environment = os.environ if environ is None else environ
    configured_value = runtime_root
    if configured_value is None:
        configured_value = environment.get(RUNTIME_ROOT_ENV)
    if configured_value is None or not str(configured_value).strip():
        return RuntimeLayout(
            runtime_mode="local",
            project_root=root,
            runtime_root=None,
            data_path=root / "data",
            reports_path=root / "reports",
            models_path=root / "models",
        )

    configured_path = Path(configured_value).expanduser()
    if not configured_path.is_absolute():
        raise RuntimeConfigurationError(f"{RUNTIME_ROOT_ENV} must be an absolute path")
    persistent_root = configured_path.resolve()
    if persistent_root == root or root.is_relative_to(persistent_root):
        raise RuntimeConfigurationError(
            f"{RUNTIME_ROOT_ENV} must not be the application directory or one of its parents"
        )
    return RuntimeLayout(
        runtime_mode="persistent",
        project_root=root,
        runtime_root=persistent_root,
        data_path=persistent_root / "data",
        reports_path=persistent_root / "reports",
        models_path=persistent_root / "models",
    )


def initialize_runtime_layout(
    project_root: Path | None = None,
    runtime_root: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> RuntimeLayout:
    """Idempotently link project-relative mutable trees to persistent storage when configured."""
    layout = resolve_runtime_layout(
        project_root=project_root,
        runtime_root=runtime_root,
        environ=environ,
    )
    if layout.runtime_root is None:
        return layout

    layout.runtime_root.mkdir(parents=True, exist_ok=True)
    targets = {
        "data": layout.data_path,
        "reports": layout.reports_path,
        "models": layout.models_path,
    }
    for target in targets.values():
        target.mkdir(parents=True, exist_ok=True)

    for tree_name, target in targets.items():
        application_path = layout.project_root / tree_name
        _validate_application_tree(application_path, target)

    for tree_name, target in targets.items():
        application_path = layout.project_root / tree_name
        if application_path.is_symlink():
            continue
        if application_path.is_dir():
            application_path.rmdir()
        application_path.symlink_to(target, target_is_directory=True)
    return layout


def create_production_runtime_report(
    project_root: Path | None = None,
    runtime_root: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ProductionRuntimeReport:
    """Inspect deployment paths and seed artifacts without creating or modifying files."""
    environment = os.environ if environ is None else environ
    layout = resolve_runtime_layout(
        project_root=project_root,
        runtime_root=runtime_root,
        environ=environment,
    )
    data_config = load_data_config(
        layout.project_root / "configs/data.yaml",
        project_root=layout.project_root,
    )
    fastf1_cache_path = _physical_runtime_path(data_config.fastf1_cache_dir, layout)
    dashboard_config = environment.get(DASHBOARD_DIR_ENV, DEFAULT_DASHBOARD_DIR)
    dashboard_path = _configured_runtime_path(dashboard_config, layout)
    protocol_path = _physical_runtime_path(
        layout.project_root / MONITORING_PROTOCOL_RELATIVE_PATH,
        layout,
    )
    dataset_path = _physical_runtime_path(
        layout.project_root / HISTORICAL_DATASET_RELATIVE_PATH,
        layout,
    )
    dashboard_present = all(
        (dashboard_path / filename).is_file() for filename in DASHBOARD_ARTIFACTS.values()
    )
    protocol_present = protocol_path.is_file()
    dataset_present = dataset_path.is_file()
    from f1_prediction.modeling.weekend_orchestrator import (
        AutopilotConfig,
        runtime_storage_diagnostics,
    )
    from f1_prediction.production_state import verify_seeded_production_state

    physical_runtime_root = layout.runtime_root or layout.project_root
    production_state = verify_seeded_production_state(physical_runtime_root)
    storage = runtime_storage_diagnostics(data_config, AutopilotConfig(), environment)
    if production_state["ready_for_future_monitoring"]:
        status = "seeded_monitoring_ready"
    elif production_state["ready_for_api"]:
        status = "seeded_api_ready"
    elif production_state["production_state_integrity_status"] == "failed":
        status = "corrupt_or_conflicting"
    else:
        status = "not_seeded"

    root_exists = bool(layout.runtime_root and layout.runtime_root.is_dir())
    root_writable = (
        os.access(layout.runtime_root, os.W_OK)
        if layout.runtime_root is not None and root_exists
        else None
    )
    return ProductionRuntimeReport(
        runtime_mode=layout.runtime_mode,
        runtime_root=str(layout.runtime_root) if layout.runtime_root is not None else None,
        data_path=str(layout.data_path),
        reports_path=str(layout.reports_path),
        models_path=str(layout.models_path),
        dashboard_path=str(dashboard_path),
        fastf1_cache_path=str(fastf1_cache_path),
        runtime_root_exists=root_exists,
        runtime_root_writable=root_writable,
        dashboard_artifacts_present=dashboard_present,
        monitoring_protocol_present=protocol_present,
        historical_modeling_dataset_present=dataset_present,
        bootstrap_receipt_present=bool(production_state["bootstrap_receipt_present"]),
        bootstrap_status=str(production_state["bootstrap_status"]),
        bundle_fingerprint=production_state["bundle_fingerprint"],
        manifest_fingerprint=production_state["manifest_fingerprint"],
        production_state_file_count=int(production_state["production_state_file_count"]),
        production_state_integrity_status=str(
            production_state["production_state_integrity_status"]
        ),
        bootstrap_seed_status=str(production_state["bootstrap_seed_status"]),
        static_seed_invariants_status=str(production_state["static_seed_invariants_status"]),
        live_operational_state_status=str(production_state["live_operational_state_status"]),
        critical_artifact_integrity_status=str(
            production_state["critical_artifact_integrity_status"]
        ),
        protocol_fingerprint_status=str(production_state["protocol_fingerprint_status"]),
        dashboard_bundle_status=str(production_state["dashboard_bundle_status"]),
        forecast_artifact_status=str(production_state["forecast_artifact_status"]),
        settlement_artifact_status=str(production_state["settlement_artifact_status"]),
        modeling_dataset_status=str(production_state["modeling_dataset_status"]),
        ready_for_api=bool(production_state["ready_for_api"]),
        ready_for_future_monitoring=bool(production_state["ready_for_future_monitoring"]),
        fastf1_cache_bytes=int(storage["fastf1_cache_bytes"]),
        runtime_total_known_bytes=int(storage["runtime_total_known_bytes"]),
        volume_capacity_bytes=storage["volume_capacity_bytes"],
        cache_warning_status=str(storage["cache_warning_status"]),
        status=status,
    )


def _validate_application_tree(application_path: Path, target: Path) -> None:
    if application_path.is_symlink():
        if application_path.resolve(strict=False) != target.resolve(strict=False):
            raise RuntimeConfigurationError(
                f"Refusing to replace existing symlink {application_path}; "
                f"it does not point to {target}"
            )
        return
    if not application_path.exists():
        return
    if not application_path.is_dir():
        raise RuntimeConfigurationError(
            f"Refusing to replace non-directory runtime path: {application_path}"
        )
    if any(application_path.iterdir()):
        raise RuntimeConfigurationError(
            f"Refusing to replace non-empty application tree {application_path}. "
            "Milestone 49A does not migrate local artifacts; seed the persistent runtime "
            "explicitly."
        )


def _configured_runtime_path(value: str | Path, layout: RuntimeLayout) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return _physical_runtime_path(resolve_project_path(candidate, layout.project_root), layout)


def _physical_runtime_path(path: Path, layout: RuntimeLayout) -> Path:
    resolved = path.resolve()
    if layout.runtime_root is None:
        return resolved
    try:
        relative = resolved.relative_to(layout.project_root)
    except ValueError:
        return resolved
    if not relative.parts or relative.parts[0] not in MUTABLE_TREES:
        return resolved
    tree_root = getattr(layout, f"{relative.parts[0]}_path")
    return tree_root.joinpath(*relative.parts[1:]).resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apex Pulse production runtime utilities")
    parser.add_argument(
        "command",
        choices=("initialize",),
        help="Initialize persistent runtime links when APEX_PULSE_RUNTIME_ROOT is configured.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the container-facing runtime initializer."""
    _build_parser().parse_args(argv)
    try:
        layout = initialize_runtime_layout()
    except (OSError, RuntimeConfigurationError) as exc:
        print(f"Production runtime initialization failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"runtime_mode": layout.runtime_mode, "status": "initialized"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
