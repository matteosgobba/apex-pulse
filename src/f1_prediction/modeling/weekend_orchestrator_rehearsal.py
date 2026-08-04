"""Artifact-copy rehearsal for the autonomous weekend orchestrator."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig, FeatureConfig, ModelConfig
from f1_prediction.modeling.monitoring_operations import DEFAULT_PROTOCOL_NAME
from f1_prediction.modeling.weekend_orchestrator import (
    AutopilotConfig,
    ProbeStatus,
    ReadinessProbeResult,
    SessionWindow,
    WeekendEvent,
    run_autopilot_tick,
)

REHEARSAL_SCHEMA_VERSION = "1.0"
COPIED_TABLES = (
    "prospective_monitoring_event_registry.csv",
    "prospective_monitoring_forecasts.parquet",
    "prospective_monitoring_settlements.parquet",
)


@dataclass(frozen=True)
class AutopilotRehearsalSummary:
    """Read-only result from four historical artifact-copy scenarios."""

    schema_version: str
    source_artifact_hashes: dict[str, str]
    source_artifacts_unchanged: bool
    forecast_event: str
    settled_event: str
    scenarios: dict[str, dict[str, Any]]
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _SingleEventSchedule:
    def __init__(self, event: WeekendEvent) -> None:
        self.event = event

    def events(self, season: int) -> tuple[WeekendEvent, ...]:
        return (self.event,) if season == self.event.season else ()


class _ReadySessions:
    def probe(self, event: WeekendEvent, session_code: str) -> ReadinessProbeResult:
        return ReadinessProbeResult(
            ProbeStatus.READY,
            f"Copied-artifact rehearsal marks {session_code} data ready.",
        )


def rehearse_autopilot_artifacts(
    data_config: DataConfig,
    model_config: ModelConfig,
    feature_config: FeatureConfig,
    *,
    autopilot_config: AutopilotConfig,
    protocol_name: str = DEFAULT_PROTOCOL_NAME,
) -> AutopilotRehearsalSummary:
    """Evaluate historical transitions using temporary copies and no canonical mutation."""
    source_paths = {name: data_config.metrics_output_dir / name for name in COPIED_TABLES}
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Autopilot rehearsal source artifacts are missing: {missing}")
    before_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    registry = pd.read_csv(source_paths["prospective_monitoring_event_registry.csv"])
    forecasts = pd.read_parquet(source_paths["prospective_monitoring_forecasts.parquet"])
    settlements = pd.read_parquet(source_paths["prospective_monitoring_settlements.parquet"])
    live_forecasts = _live_rows(forecasts, protocol_name)
    live_settlements = _live_rows(settlements, protocol_name)
    real_registry = registry[
        registry["protocol_name"].astype(str).eq(protocol_name)
        & ~registry["event_slug"].astype(str).str.contains("synthetic", case=False)
    ].copy()
    forecast_only = set(live_forecasts["event_slug"].astype(str)) - set(
        live_settlements["event_slug"].astype(str)
    )
    forecast_row = _select_registry_event(real_registry, forecast_only)
    settled_candidates = set(live_settlements["event_slug"].astype(str))
    settled_row = _select_registry_event(real_registry, settled_candidates, prefer_latest=True)

    with tempfile.TemporaryDirectory(prefix="apex-pulse-autopilot-rehearsal-") as temp_value:
        temp_root = Path(temp_value)
        metrics = temp_root / "reports/metrics"
        metrics.mkdir(parents=True)
        for name, source in source_paths.items():
            shutil.copy2(source, metrics / name)
        temp_config = _temporary_data_config(data_config, temp_root)
        forecast_event = _rehearsal_event(forecast_row)
        settled_event = _rehearsal_event(settled_row)
        _remove_event_rows(metrics, forecast_event.event_slug, remove_forecast=True)
        scenarios = {
            "pre_fp3": _run_case(
                temp_config,
                model_config,
                feature_config,
                autopilot_config,
                forecast_event,
                forecast_event.sessions["FP3"].start_utc - timedelta(minutes=30),
                protocol_name,
            ),
            "post_fp3_pre_qualifying": _run_case(
                temp_config,
                model_config,
                feature_config,
                autopilot_config,
                forecast_event,
                forecast_event.sessions["FP3"].end_utc
                + timedelta(minutes=autopilot_config.fp3_initial_grace_minutes + 1),
                protocol_name,
            ),
        }
        shutil.copy2(
            source_paths["prospective_monitoring_forecasts.parquet"], metrics / COPIED_TABLES[1]
        )
        scenarios["post_qualifying"] = _run_case(
            temp_config,
            model_config,
            feature_config,
            autopilot_config,
            forecast_event,
            forecast_event.sessions["Q"].end_utc
            + timedelta(minutes=autopilot_config.qualifying_initial_grace_minutes + 1),
            protocol_name,
        )
        scenarios["already_settled"] = _run_case(
            temp_config,
            model_config,
            feature_config,
            autopilot_config,
            settled_event,
            settled_event.sessions["Q"].end_utc + timedelta(minutes=30),
            protocol_name,
        )

    after_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    expected_states = {
        "pre_fp3": {"FP2_COMPLETE", "WAITING_FOR_FP3"},
        "post_fp3_pre_qualifying": {"READY_FOR_FORECAST"},
        "post_qualifying": {"READY_FOR_SETTLEMENT"},
        "already_settled": {"SETTLED", "SETTLED_PARTIAL_COVERAGE"},
    }
    passed = all(
        scenarios[name]["orchestrator_state_after"] in allowed
        for name, allowed in expected_states.items()
    )
    unchanged = before_hashes == after_hashes
    return AutopilotRehearsalSummary(
        schema_version=REHEARSAL_SCHEMA_VERSION,
        source_artifact_hashes=before_hashes,
        source_artifacts_unchanged=unchanged,
        forecast_event=str(forecast_row["event"]),
        settled_event=str(settled_row["event"]),
        scenarios=scenarios,
        status="passed" if passed and unchanged else "failed",
    )


def _run_case(
    data_config: DataConfig,
    model_config: ModelConfig,
    feature_config: FeatureConfig,
    autopilot_config: AutopilotConfig,
    event: WeekendEvent,
    now: datetime,
    protocol_name: str,
) -> dict[str, Any]:
    result = run_autopilot_tick(
        data_config,
        model_config,
        feature_config,
        autopilot_config=autopilot_config,
        now=now,
        season=event.season,
        event=event.event,
        dry_run=True,
        protocol_name=protocol_name,
        schedule_provider=_SingleEventSchedule(event),
        readiness_provider=_ReadySessions(),
        before_workflow=_forbidden_workflow,
        after_workflow=_forbidden_workflow,
        environ={},
    )
    return result.to_dict()


def _forbidden_workflow(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("Dry-run rehearsal must never invoke a canonical mutation workflow")


def _select_registry_event(
    registry: pd.DataFrame,
    slugs: set[str],
    *,
    prefer_latest: bool = False,
) -> pd.Series:
    rows = registry[registry["event_slug"].astype(str).isin(slugs)].copy()
    if rows.empty:
        raise ValueError("No real artifact-backed event satisfies the rehearsal scenario")
    order_column = "event_order" if "event_order" in rows else None
    if order_column:
        rows = rows.sort_values(order_column)
    return rows.iloc[-1 if prefer_latest else 0]


def _live_rows(frame: pd.DataFrame, protocol_name: str) -> pd.DataFrame:
    rows = frame[frame["protocol_name"].astype(str).eq(protocol_name)].copy()
    if "diagnostic_only" in rows:
        rows = rows[~rows["diagnostic_only"].astype(bool)]
    if "settlement_valid" in rows:
        rows = rows[rows["settlement_valid"].astype(bool)]
    return rows


def _remove_event_rows(metrics: Path, event_slug: str, *, remove_forecast: bool) -> None:
    if remove_forecast:
        path = metrics / "prospective_monitoring_forecasts.parquet"
        frame = pd.read_parquet(path)
        frame[~frame["event_slug"].astype(str).eq(event_slug)].to_parquet(path, index=False)
    settlement_path = metrics / "prospective_monitoring_settlements.parquet"
    settlements = pd.read_parquet(settlement_path)
    settlements[~settlements["event_slug"].astype(str).eq(event_slug)].to_parquet(
        settlement_path,
        index=False,
    )


def _rehearsal_event(row: pd.Series) -> WeekendEvent:
    season = int(row.get("season", 2026))
    event = str(row["event"])
    event_slug = str(row["event_slug"])
    round_number = int(row.get("event_order", 1))
    base = datetime(season, 1, 10, 10, tzinfo=timezone.utc) + timedelta(days=round_number)
    starts = {
        "FP1": base,
        "FP2": base + timedelta(hours=4),
        "FP3": base + timedelta(days=1),
        "Q": base + timedelta(days=1, hours=4),
    }
    sessions = {
        code: SessionWindow(
            code,
            "Qualifying" if code == "Q" else f"Practice {code[-1]}",
            start,
            start + timedelta(minutes=60),
            "artifact_rehearsal_fixture",
        )
        for code, start in starts.items()
    }
    return WeekendEvent(
        season,
        event,
        event_slug,
        round_number,
        "conventional",
        sessions,
        (event_slug,),
    )


def _temporary_data_config(source: DataConfig, root: Path) -> DataConfig:
    return replace(
        source,
        project_root=root,
        fastf1_cache_dir=root / "data/raw/fastf1_cache",
        lap_output_dir=root / "data/raw/laps",
        session_metadata_output_dir=root / "data/raw/session_metadata",
        clean_lap_output_dir=root / "data/interim/clean_laps",
        session_features_output_dir=root / "data/processed/session_features",
        modeling_output_dir=root / "data/processed/modeling",
        metrics_output_dir=root / "reports/metrics",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
