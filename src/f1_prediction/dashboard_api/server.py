"""Dashboard API server binding configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

PORT_ENV = "PORT"
DEFAULT_LOCAL_HOST = "127.0.0.1"
DEFAULT_LOCAL_PORT = 8000
PRODUCTION_HOST = "0.0.0.0"


@dataclass(frozen=True)
class DashboardServerBinding:
    """Resolved host and port for the dashboard API process."""

    host: str
    port: int


def resolve_dashboard_server_binding(
    host: str | None = None,
    port: int | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> DashboardServerBinding:
    """Use local defaults unless a deployment ``PORT`` is configured."""
    environment = os.environ if environ is None else environ
    raw_environment_port = environment.get(PORT_ENV)
    deployment_port_configured = bool(
        raw_environment_port is not None and raw_environment_port.strip()
    )
    if port is None:
        resolved_port = (
            _parse_port(raw_environment_port) if deployment_port_configured else DEFAULT_LOCAL_PORT
        )
    else:
        resolved_port = _validate_port(port)
    resolved_host = host or (PRODUCTION_HOST if deployment_port_configured else DEFAULT_LOCAL_HOST)
    return DashboardServerBinding(host=resolved_host, port=resolved_port)


def _parse_port(value: str | None) -> int:
    try:
        parsed = int(value or "")
    except ValueError as exc:
        raise ValueError("PORT must be an integer between 1 and 65535") from exc
    return _validate_port(parsed)


def _validate_port(port: int) -> int:
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be an integer between 1 and 65535")
    return port
