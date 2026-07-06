"""FastAPI app factory for serving dashboard artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from f1_prediction.dashboard_api.schemas import (
    API_VERSION,
    SERVICE_NAME,
    DashboardErrorDetail,
    HealthResponse,
)
from f1_prediction.dashboard_api.service import (
    DEFAULT_DASHBOARD_DIR,
    DashboardApiError,
    DashboardArtifactService,
)

DASHBOARD_DIR_ENV = "APEX_PULSE_DASHBOARD_DIR"
CORS_ORIGINS_ENV = "APEX_PULSE_DASHBOARD_API_CORS_ORIGINS"
DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"


def create_dashboard_app(dashboard_dir: Path | None = None) -> FastAPI:
    """Create the read-only dashboard API app."""
    configured_dir = dashboard_dir or Path(os.getenv(DASHBOARD_DIR_ENV, DEFAULT_DASHBOARD_DIR))
    service = DashboardArtifactService(configured_dir)
    app = FastAPI(
        title="Apex Pulse Dashboard API",
        version="1.0.0",
        description="Read-only API over validated dashboard JSON artifacts.",
    )
    app.state.dashboard_service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.exception_handler(DashboardApiError)
    async def dashboard_api_error_handler(
        _request: Any,
        exc: DashboardApiError,
    ) -> JSONResponse:
        detail = DashboardErrorDetail(
            code=exc.code,
            message=exc.message,
            artifact_type=exc.artifact_type,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": detail.model_dump(exclude_none=True)},
        )

    @app.get("/api/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=SERVICE_NAME,
            api_version=API_VERSION,
            dashboard_artifact_status=service.health_status(),
        )

    @app.get("/api/v1/dashboard/manifest")
    def manifest() -> JSONResponse:
        return _artifact_response(service, "dashboard_manifest")

    @app.get("/api/v1/dashboard/current-event")
    def current_event() -> JSONResponse:
        return _artifact_response(service, "current_event")

    @app.get("/api/v1/dashboard/current-event/forecast")
    def forecast() -> JSONResponse:
        return _artifact_response(service, "event_forecast")

    @app.get("/api/v1/dashboard/current-event/settlement")
    def settlement() -> JSONResponse:
        return _artifact_response(service, "event_settlement")

    @app.get("/api/v1/dashboard/current-event/practice-status")
    def practice_status() -> JSONResponse:
        return _artifact_response(service, "event_practice_status")

    @app.get("/api/v1/dashboard/historical-monitoring")
    def historical_monitoring() -> JSONResponse:
        return _artifact_response(service, "historical_monitoring_summary")

    @app.get("/api/v1/dashboard/model-summary")
    def model_summary() -> JSONResponse:
        return _artifact_response(service, "model_summary")

    @app.get("/api/v1/dashboard/bundle")
    def bundle() -> JSONResponse:
        body = service.load_bundle()
        return _json_response(body)

    return app


def _artifact_response(service: DashboardArtifactService, artifact_type: str) -> JSONResponse:
    artifact = service.load_artifact(artifact_type)
    return _json_response(
        artifact.payload,
        headers={"X-Apex-Pulse-Dashboard-Generated-At": artifact.generated_at_utc},
    )


def _json_response(content: Any, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(
        content=content,
        headers=headers,
    )


def _cors_origins() -> list[str]:
    raw = os.getenv(CORS_ORIGINS_ENV, DEFAULT_CORS_ORIGINS)
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if "*" in origins:
        return ["*"] if len(origins) == 1 else [origin for origin in origins if origin != "*"]
    return origins
