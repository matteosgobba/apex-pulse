import asyncio
import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from f1_prediction.cli import app
from f1_prediction.dashboard.schema import SCHEMA_VERSION
from f1_prediction.dashboard_api.app import create_dashboard_app
from f1_prediction.dashboard_api.service import (
    DEFAULT_STALE_AFTER_MINUTES,
    parse_stale_after_minutes,
)

GENERATED_AT = "2026-01-01T12:00:00+00:00"
ARTIFACT_FILES = {
    "dashboard_manifest": "dashboard_manifest.json",
    "current_event": "current_event.json",
    "event_forecast": "event_forecast.json",
    "event_settlement": "event_settlement.json",
    "event_practice_status": "event_practice_status.json",
    "historical_monitoring_summary": "historical_monitoring_summary.json",
    "model_summary": "model_summary.json",
}


def test_health_works_with_no_dashboard_directory(tmp_path: Path) -> None:
    application = create_dashboard_app(tmp_path / "missing")

    response = _request(application, "GET", "/api/v1/health")

    assert response.status_code == 200
    assert response.json == {
        "status": "ok",
        "service": "apex-pulse-dashboard-api",
        "api_version": "v1",
        "dashboard_artifact_status": "unavailable",
    }


def test_health_reports_unavailable_before_export(tmp_path: Path) -> None:
    dashboard_dir = tmp_path / "dashboard"
    dashboard_dir.mkdir()
    application = create_dashboard_app(dashboard_dir)

    response = _request(application, "GET", "/api/v1/health")

    assert response.status_code == 200
    assert response.json["dashboard_artifact_status"] == "unavailable"


def test_valid_manifest_returns_exact_artifact_content(tmp_path: Path) -> None:
    dashboard_dir = _write_dashboard_bundle(tmp_path)
    expected = _read_json(dashboard_dir / "dashboard_manifest.json")
    application = create_dashboard_app(dashboard_dir)

    response = _request(application, "GET", "/api/v1/dashboard/manifest")

    assert response.status_code == 200
    assert response.json == expected


def test_current_event_endpoint_returns_valid_current_event(tmp_path: Path) -> None:
    dashboard_dir = _write_dashboard_bundle(tmp_path)
    application = create_dashboard_app(dashboard_dir)

    response = _request(application, "GET", "/api/v1/dashboard/current-event")

    assert response.status_code == 200
    assert response.json["artifact_type"] == "current_event"


def test_dashboard_endpoints_map_to_correct_artifacts(tmp_path: Path) -> None:
    dashboard_dir = _write_dashboard_bundle(tmp_path)
    application = create_dashboard_app(dashboard_dir)

    endpoint_artifacts = {
        "/api/v1/dashboard/current-event/forecast": "event_forecast",
        "/api/v1/dashboard/current-event/settlement": "event_settlement",
        "/api/v1/dashboard/current-event/practice-status": "event_practice_status",
        "/api/v1/dashboard/historical-monitoring": "historical_monitoring_summary",
        "/api/v1/dashboard/model-summary": "model_summary",
    }

    for endpoint, artifact_type in endpoint_artifacts.items():
        response = _request(application, "GET", endpoint)
        assert response.status_code == 200
        assert response.json["artifact_type"] == artifact_type


def test_valid_responses_include_generated_at_header(tmp_path: Path) -> None:
    dashboard_dir = _write_dashboard_bundle(tmp_path)
    application = create_dashboard_app(dashboard_dir)

    response = _request(application, "GET", "/api/v1/dashboard/current-event/forecast")

    assert response.status_code == 200
    assert response.headers["x-apex-pulse-dashboard-generated-at"] == GENERATED_AT


def test_valid_artifact_responses_include_safe_freshness_headers(tmp_path: Path) -> None:
    dashboard_dir = _write_dashboard_bundle(tmp_path)
    application = create_dashboard_app(dashboard_dir)

    response = _request(application, "GET", "/api/v1/dashboard/current-event")

    assert response.status_code == 200
    assert response.headers["x-apex-pulse-dashboard-generated-at"] == GENERATED_AT
    assert response.headers["x-apex-pulse-dashboard-artifact-type"] == "current_event"
    assert response.headers["x-apex-pulse-dashboard-status"] == "complete"
    assert response.headers["cache-control"] == "no-cache"
    assert "immutable" not in response.headers["cache-control"]
    assert str(tmp_path) not in json.dumps(dict(response.headers))


def test_missing_optional_artifact_returns_stable_404(tmp_path: Path) -> None:
    dashboard_dir = _write_dashboard_bundle(tmp_path)
    (dashboard_dir / "event_forecast.json").unlink()
    application = create_dashboard_app(dashboard_dir)

    response = _request(application, "GET", "/api/v1/dashboard/current-event/forecast")

    assert response.status_code == 404
    assert response.json == {
        "detail": {
            "code": "dashboard_artifact_not_found",
            "message": "The requested dashboard artifact is not available.",
            "artifact_type": "event_forecast",
        }
    }


def test_missing_dashboard_directory_returns_stable_503(tmp_path: Path) -> None:
    application = create_dashboard_app(tmp_path / "missing")

    response = _request(application, "GET", "/api/v1/dashboard/current-event")

    assert response.status_code == 503
    assert response.json == {
        "detail": {
            "code": "dashboard_artifacts_unavailable",
            "message": (
                "Dashboard artifacts have not been exported yet. Run dashboard-export first."
            ),
        }
    }


def test_empty_dashboard_directory_returns_stable_503(tmp_path: Path) -> None:
    dashboard_dir = tmp_path / "dashboard"
    dashboard_dir.mkdir()
    application = create_dashboard_app(dashboard_dir)

    response = _request(application, "GET", "/api/v1/dashboard/current-event")

    assert response.status_code == 503
    assert response.json["detail"]["code"] == "dashboard_artifacts_unavailable"


def test_malformed_json_returns_safe_500(tmp_path: Path) -> None:
    dashboard_dir = _write_dashboard_bundle(tmp_path)
    (dashboard_dir / "current_event.json").write_text("{bad", encoding="utf-8")
    application = create_dashboard_app(dashboard_dir)

    response = _request(application, "GET", "/api/v1/dashboard/current-event")

    assert response.status_code == 500
    assert response.json == {
        "detail": {
            "code": "dashboard_artifact_invalid",
            "message": "The requested dashboard artifact failed validation.",
            "artifact_type": "current_event",
        }
    }
    assert "Expecting" not in response.text
    assert str(tmp_path) not in response.text


def test_schema_invalid_json_returns_safe_500(tmp_path: Path) -> None:
    dashboard_dir = _write_dashboard_bundle(tmp_path)
    payload = _artifact("current_event", {"example": True})
    payload["source_artifacts"] = [str(tmp_path / "absolute.json")]
    _write_json(dashboard_dir / "current_event.json", payload)
    application = create_dashboard_app(dashboard_dir)

    response = _request(application, "GET", "/api/v1/dashboard/current-event")

    assert response.status_code == 500
    assert response.json["detail"]["code"] == "dashboard_artifact_invalid"
    assert str(tmp_path) not in response.text


def test_absolute_paths_never_appear_in_successful_or_error_responses(tmp_path: Path) -> None:
    dashboard_dir = _write_dashboard_bundle(tmp_path)
    application = create_dashboard_app(dashboard_dir)

    success = _request(application, "GET", "/api/v1/dashboard/current-event")
    (dashboard_dir / "event_forecast.json").unlink()
    missing = _request(application, "GET", "/api/v1/dashboard/current-event/forecast")

    assert str(tmp_path) not in success.text
    assert str(tmp_path) not in missing.text


def test_api_reads_artifacts_only_and_does_not_modify_them(tmp_path: Path) -> None:
    dashboard_dir = _write_dashboard_bundle(tmp_path)
    path = dashboard_dir / "current_event.json"
    before = path.read_bytes()
    application = create_dashboard_app(dashboard_dir)

    response = _request(application, "GET", "/api/v1/dashboard/current-event")

    assert response.status_code == 200
    assert path.read_bytes() == before


def test_configured_cors_origin_is_allowed(monkeypatch, tmp_path: Path) -> None:
    dashboard_dir = _write_dashboard_bundle(tmp_path)
    monkeypatch.setenv("APEX_PULSE_DASHBOARD_API_CORS_ORIGINS", "http://localhost:3000")
    application = create_dashboard_app(dashboard_dir)

    response = _request(
        application,
        "OPTIONS",
        "/api/v1/dashboard/current-event",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-credentials" not in response.headers


def test_staleness_configuration_parsing_is_robust(monkeypatch, tmp_path: Path) -> None:
    assert parse_stale_after_minutes("45") == 45
    assert parse_stale_after_minutes("bad") == DEFAULT_STALE_AFTER_MINUTES
    assert parse_stale_after_minutes("-10") == DEFAULT_STALE_AFTER_MINUTES
    assert parse_stale_after_minutes(None) == DEFAULT_STALE_AFTER_MINUTES

    monkeypatch.setenv("APEX_PULSE_DASHBOARD_STALE_AFTER_MINUTES", "not-a-number")
    application = create_dashboard_app(tmp_path / "missing")

    assert application.state.dashboard_stale_after_minutes == DEFAULT_STALE_AFTER_MINUTES


def test_bundle_returns_same_validated_documents_without_second_schema(tmp_path: Path) -> None:
    dashboard_dir = _write_dashboard_bundle(tmp_path)
    application = create_dashboard_app(dashboard_dir)

    response = _request(application, "GET", "/api/v1/dashboard/bundle")

    assert response.status_code == 200
    assert response.json["current_event"] == _read_json(dashboard_dir / "current_event.json")
    assert set(response.json) == set(ARTIFACT_FILES)


def test_dashboard_api_cli_registration(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(application, *, host, port):
        captured["application"] = application
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr("uvicorn.run", fake_run)

    result = CliRunner().invoke(
        app,
        [
            "dashboard-api",
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
            "--dashboard-dir",
            str(tmp_path / "dashboard"),
        ],
    )

    assert result.exit_code == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8765


class AsgiResponse:
    def __init__(self, status_code: int, headers: dict[str, str], body: bytes) -> None:
        self.status_code = status_code
        self.headers = headers
        self.body = body
        self.text = body.decode("utf-8")
        try:
            self.json = json.loads(self.text) if self.text else None
        except json.JSONDecodeError:
            self.json = None


def _request(
    application: Any,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
) -> AsgiResponse:
    return asyncio.run(_request_async(application, method, path, headers=headers or {}))


async def _request_async(
    application: Any,
    method: str,
    path: str,
    *,
    headers: dict[str, str],
) -> AsgiResponse:
    response_start: dict[str, Any] = {}
    body_parts: list[bytes] = []
    request_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if request_sent:
            return {"type": "http.disconnect"}
        request_sent = True
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            response_start.update(message)
        elif message["type"] == "http.response.body":
            body_parts.append(message.get("body", b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [
            (key.lower().encode("latin-1"), value.encode("latin-1"))
            for key, value in headers.items()
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    await application(scope, receive, send)
    response_headers = {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in response_start.get("headers", [])
    }
    return AsgiResponse(
        status_code=int(response_start["status"]),
        headers=response_headers,
        body=b"".join(body_parts),
    )


def _write_dashboard_bundle(tmp_path: Path) -> Path:
    dashboard_dir = tmp_path / "dashboard"
    dashboard_dir.mkdir()
    for artifact_type, filename in ARTIFACT_FILES.items():
        _write_json(dashboard_dir / filename, _artifact(artifact_type, _data_for(artifact_type)))
    return dashboard_dir


def _artifact(artifact_type: str, data: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "generated_at_utc": GENERATED_AT,
        "source_artifacts": ["reports/metrics/source.json"],
        "source_fingerprints": {
            "reports/metrics/source.json": {
                "available": True,
                "required": False,
                "sha256": "abc",
                "reason": None,
            }
        },
        "status": "complete",
        "data": data,
    }


def _data_for(artifact_type: str) -> dict:
    if artifact_type == "dashboard_manifest":
        return {
            "current_event_reference": {
                "artifact": "current_event.json",
                "lifecycle_state": "forecast_available",
            }
        }
    if artifact_type == "current_event":
        return {
            "event_identity": {"season": 2026, "event": "Bahrain", "event_slug": "bahrain"},
            "lifecycle": {
                "state": "forecast_available",
                "display_label": "Forecast available",
                "reason": "forecast_snapshot_exists",
            },
        }
    if artifact_type == "event_forecast":
        return {
            "event_identity": {"season": 2026, "event_slug": "bahrain"},
            "lifecycle_state": "forecast_available",
            "leaderboard": [],
        }
    if artifact_type == "event_settlement":
        return {
            "event_identity": {"season": 2026, "event_slug": "bahrain"},
            "lifecycle_state": "forecast_available",
            "driver_comparison": [],
        }
    if artifact_type == "event_practice_status":
        return {
            "event_identity": {"season": 2026, "event_slug": "bahrain"},
            "lifecycle_state": "forecast_available",
            "sessions": [],
        }
    if artifact_type == "historical_monitoring_summary":
        return {
            "valid_prospective_monitoring": {"event_count": 0},
            "legacy_descriptive_records": [],
        }
    if artifact_type == "model_summary":
        return {"model_status": {"training_status": "complete"}}
    raise AssertionError(f"Unknown artifact_type: {artifact_type}")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
