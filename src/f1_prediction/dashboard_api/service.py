"""Safe read-only access to exported dashboard artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from f1_prediction.dashboard.schema import (
    DashboardSchemaError,
    validate_dashboard_artifact_file,
)
from f1_prediction.utils.paths import get_project_root, resolve_project_path

DASHBOARD_ARTIFACTS: dict[str, str] = {
    "dashboard_manifest": "dashboard_manifest.json",
    "current_event": "current_event.json",
    "event_forecast": "event_forecast.json",
    "event_settlement": "event_settlement.json",
    "event_practice_status": "event_practice_status.json",
    "historical_monitoring_summary": "historical_monitoring_summary.json",
    "model_summary": "model_summary.json",
}

DEFAULT_DASHBOARD_DIR = "reports/dashboard"


class DashboardApiError(Exception):
    """Base exception converted to stable API errors by the app layer."""

    status_code: int = 500
    code: str = "dashboard_artifact_error"
    message: str = "Dashboard artifact error."

    def __init__(self, *, artifact_type: str | None = None) -> None:
        self.artifact_type = artifact_type
        super().__init__(self.message)


class DashboardArtifactsUnavailableError(DashboardApiError):
    """Raised when the dashboard artifact directory has not been exported yet."""

    status_code = 503
    code = "dashboard_artifacts_unavailable"
    message = "Dashboard artifacts have not been exported yet. Run dashboard-export first."


class DashboardArtifactNotFoundError(DashboardApiError):
    """Raised when one dashboard artifact file is missing."""

    status_code = 404
    code = "dashboard_artifact_not_found"
    message = "The requested dashboard artifact is not available."


class DashboardArtifactInvalidError(DashboardApiError):
    """Raised when an exported dashboard artifact fails validation."""

    status_code = 500
    code = "dashboard_artifact_invalid"
    message = "The requested dashboard artifact failed validation."


@dataclass(frozen=True)
class DashboardArtifact:
    """Loaded and validated dashboard artifact plus response metadata."""

    artifact_type: str
    payload: dict[str, Any]
    generated_at_utc: str


class DashboardArtifactService:
    """Read and validate dashboard JSON artifacts from a fixed directory."""

    def __init__(self, dashboard_dir: Path | str | None = None) -> None:
        self.project_root = get_project_root()
        configured = dashboard_dir or DEFAULT_DASHBOARD_DIR
        self.dashboard_dir = resolve_project_path(configured, self.project_root)

    def health_status(self) -> str:
        """Return the exported dashboard manifest status without raising."""
        if not self.dashboard_dir.is_dir():
            return "unavailable"
        manifest_path = self.dashboard_dir / DASHBOARD_ARTIFACTS["dashboard_manifest"]
        if not manifest_path.is_file():
            return "unavailable"
        try:
            payload = validate_dashboard_artifact_file(manifest_path)
        except (OSError, json.JSONDecodeError, DashboardSchemaError):
            return "invalid"
        status = payload.get("status")
        return status if status in {"complete", "partial", "empty", "invalid"} else "invalid"

    def load_artifact(self, artifact_type: str) -> DashboardArtifact:
        """Load one validated dashboard artifact."""
        filename = DASHBOARD_ARTIFACTS[artifact_type]
        if not self.dashboard_dir.is_dir() or not self._export_manifest_exists():
            raise DashboardArtifactsUnavailableError()
        path = self.dashboard_dir / filename
        if not path.is_file():
            raise DashboardArtifactNotFoundError(artifact_type=artifact_type)
        try:
            payload = validate_dashboard_artifact_file(path)
        except (OSError, json.JSONDecodeError, DashboardSchemaError) as exc:
            raise DashboardArtifactInvalidError(artifact_type=artifact_type) from exc
        return DashboardArtifact(
            artifact_type=artifact_type,
            payload=payload,
            generated_at_utc=str(payload["generated_at_utc"]),
        )

    def load_bundle(self) -> dict[str, dict[str, Any]]:
        """Load every dashboard artifact using the same validation path as individual routes."""
        return {
            artifact_type: self.load_artifact(artifact_type).payload
            for artifact_type in DASHBOARD_ARTIFACTS
        }

    def _export_manifest_exists(self) -> bool:
        return (self.dashboard_dir / DASHBOARD_ARTIFACTS["dashboard_manifest"]).is_file()
