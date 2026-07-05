"""Stable dashboard artifact contract and validation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "1.0"

EnvelopeStatus = Literal["complete", "partial", "empty", "invalid"]
LifecycleStateName = Literal[
    "no_event_available",
    "practice_in_progress",
    "ready_to_forecast",
    "forecast_available",
    "awaiting_qualifying_targets",
    "settled",
    "blocked",
    "legacy_descriptive_only",
]

ENVELOPE_STATUSES: set[str] = {"complete", "partial", "empty", "invalid"}
LIFECYCLE_STATES: set[str] = {
    "no_event_available",
    "practice_in_progress",
    "ready_to_forecast",
    "forecast_available",
    "awaiting_qualifying_targets",
    "settled",
    "blocked",
    "legacy_descriptive_only",
}


class DashboardSchemaError(ValueError):
    """Raised when a dashboard-facing document violates the public contract."""


@dataclass(frozen=True)
class AvailabilityValue:
    """Null-safe optional value wrapper used by dashboard artifacts."""

    available: bool
    reason: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EventIdentity:
    """Stable public event identity."""

    season: int | None = None
    event: str | None = None
    event_slug: str | None = None
    event_order: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LifecycleState:
    """Dashboard lifecycle state with display text and reason."""

    state: LifecycleStateName
    display_label: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceArtifact:
    """One source artifact referenced by the dashboard export."""

    path: str
    available: bool
    required: bool = False
    sha256: str | None = None
    reason: str | None = None

    def to_fingerprint_entry(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "required": self.required,
            "sha256": self.sha256,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DashboardEnvelope:
    """Common envelope shared by all dashboard JSON artifacts."""

    artifact_type: str
    generated_at_utc: str
    source_artifacts: tuple[str, ...]
    source_fingerprints: dict[str, dict[str, Any]]
    status: EnvelopeStatus
    data: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "generated_at_utc": self.generated_at_utc,
            "source_artifacts": list(self.source_artifacts),
            "source_fingerprints": self.source_fingerprints,
            "status": self.status,
            "data": self.data,
        }
        validate_dashboard_document(payload)
        return payload


@dataclass(frozen=True)
class DashboardManifest:
    """Top-level dashboard discovery document."""

    envelope: DashboardEnvelope


@dataclass(frozen=True)
class CurrentEventDashboard:
    """Default current-event dashboard document."""

    envelope: DashboardEnvelope


@dataclass(frozen=True)
class ForecastDashboard:
    """Normalized forecast leaderboard document."""

    envelope: DashboardEnvelope


@dataclass(frozen=True)
class SettlementDashboard:
    """Normalized post-qualifying settlement document."""

    envelope: DashboardEnvelope


@dataclass(frozen=True)
class PracticeStatusDashboard:
    """Normalized practice availability and readiness document."""

    envelope: DashboardEnvelope


@dataclass(frozen=True)
class HistoricalMonitoringDashboard:
    """Historical monitoring summary document."""

    envelope: DashboardEnvelope


@dataclass(frozen=True)
class ModelSummaryDashboard:
    """Compact public model and methodology summary document."""

    envelope: DashboardEnvelope


def unavailable(reason: str) -> dict[str, Any]:
    """Return the standard unavailable-value structure."""
    return AvailabilityValue(available=False, reason=reason, value=None).to_dict()


def available(value: Any) -> dict[str, Any]:
    """Return the standard available-value structure."""
    return AvailabilityValue(available=True, reason="", value=value).to_dict()


def validate_dashboard_document(payload: dict[str, Any]) -> None:
    """Validate the shared dashboard envelope."""
    required_keys = {
        "schema_version",
        "artifact_type",
        "generated_at_utc",
        "source_artifacts",
        "source_fingerprints",
        "status",
        "data",
    }
    missing = required_keys - set(payload)
    if missing:
        raise DashboardSchemaError(f"Dashboard envelope missing keys: {sorted(missing)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise DashboardSchemaError(f"Unsupported schema_version: {payload['schema_version']}")
    if not isinstance(payload["artifact_type"], str) or not payload["artifact_type"]:
        raise DashboardSchemaError("artifact_type must be a non-empty string")
    if payload["status"] not in ENVELOPE_STATUSES:
        raise DashboardSchemaError(f"Invalid dashboard status: {payload['status']}")
    _validate_generated_at(str(payload["generated_at_utc"]))
    _validate_source_artifacts(payload["source_artifacts"])
    if not isinstance(payload["source_fingerprints"], dict):
        raise DashboardSchemaError("source_fingerprints must be a mapping")
    if not isinstance(payload["data"], dict):
        raise DashboardSchemaError("data must be an object")


def validate_lifecycle_state(value: str) -> None:
    """Validate a lifecycle state name."""
    if value not in LIFECYCLE_STATES:
        raise DashboardSchemaError(f"Invalid lifecycle state: {value}")


def validate_dashboard_artifact_file(path: Path) -> dict[str, Any]:
    """Read and validate one exported dashboard JSON file."""
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_dashboard_document(payload)
    return payload


def _validate_generated_at(value: str) -> None:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise DashboardSchemaError("generated_at_utc must be timezone-aware")


def _validate_source_artifacts(value: Any) -> None:
    if not isinstance(value, list):
        raise DashboardSchemaError("source_artifacts must be a list")
    for item in value:
        if not isinstance(item, str):
            raise DashboardSchemaError("source_artifacts entries must be strings")
        path = Path(item)
        if path.is_absolute() or ".." in path.parts:
            raise DashboardSchemaError(f"source artifact is not project-relative: {item}")
