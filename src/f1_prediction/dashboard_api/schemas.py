"""API response models for the read-only dashboard service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

API_VERSION = "v1"
SERVICE_NAME = "apex-pulse-dashboard-api"

DashboardArtifactStatus = Literal[
    "complete",
    "partial",
    "empty",
    "invalid",
    "unavailable",
]


class HealthResponse(BaseModel):
    """Compact health response for the dashboard API."""

    status: Literal["ok"]
    service: Literal["apex-pulse-dashboard-api"]
    api_version: Literal["v1"]
    dashboard_artifact_status: DashboardArtifactStatus


class DashboardErrorDetail(BaseModel):
    """Stable API error detail body."""

    code: str
    message: str
    artifact_type: str | None = None


class DashboardErrorResponse(BaseModel):
    """Stable API error response body."""

    detail: DashboardErrorDetail
