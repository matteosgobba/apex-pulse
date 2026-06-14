"""Multi-event dataset construction using the existing event pipeline."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import fastf1
import pandas as pd

from f1_prediction.config import DataConfig, FeatureConfig
from f1_prediction.data.cache import initialize_fastf1_cache
from f1_prediction.data.ingest import ingest_event
from f1_prediction.features.build import build_session_features
from f1_prediction.features.modeling_dataset import (
    CHECKPOINT_SESSIONS,
    build_modeling_dataset_files,
)
from f1_prediction.utils.paths import ensure_directory, slugify

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class EventReference:
    """One scheduled event selected for dataset construction."""

    season: int
    event: str
    event_name: str
    round_number: int


@dataclass(frozen=True)
class SuccessfulEventBuild:
    """A successfully produced event-level modeling dataset."""

    season: int
    event: str
    event_name: str
    rows: int
    output_path: str


@dataclass(frozen=True)
class FailedEventBuild:
    """An event that could not complete the existing pipeline."""

    season: int
    event: str
    event_name: str
    error_message: str


@dataclass(frozen=True)
class SeasonDatasetBuildSummary:
    """Summary and report locations for a multi-event dataset build."""

    requested_seasons: tuple[int, ...]
    n_events_requested: int
    successful_events: tuple[SuccessfulEventBuild, ...]
    failed_events: tuple[FailedEventBuild, ...]
    n_rows: int
    n_drivers: int
    checkpoints: tuple[str, ...]
    output_path: Path
    report_path: Path

    @property
    def n_events_successful(self) -> int:
        return len(self.successful_events)

    @property
    def n_events_failed(self) -> int:
        return len(self.failed_events)


def discover_season_events(
    season: int,
    requested_events: Sequence[str] | None = None,
) -> tuple[EventReference, ...]:
    """Discover championship events from the FastF1 schedule."""
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    schedule = schedule[schedule["RoundNumber"].fillna(0).astype(int).gt(0)]
    references = tuple(_event_reference(season, row) for _, row in schedule.iterrows())
    if not requested_events:
        return references

    selected: list[EventReference] = []
    for requested in requested_events:
        requested_slug = slugify(requested)
        matches = [
            reference for reference in references if requested_slug in _event_aliases(reference)
        ]
        if not matches:
            raise ValueError(f"Event '{requested}' was not found in the {season} FastF1 schedule")
        if matches[0] not in selected:
            selected.append(matches[0])
    return tuple(selected)


def build_season_dataset(
    seasons: Sequence[int],
    data_config: DataConfig,
    feature_config: FeatureConfig,
    *,
    events: Sequence[str] | None = None,
    force: bool = False,
    fail_fast: bool = False,
    progress: ProgressCallback | None = None,
) -> SeasonDatasetBuildSummary:
    """Run the existing event pipeline and concatenate successful datasets."""
    requested_seasons = tuple(dict.fromkeys(seasons))
    if not requested_seasons:
        raise ValueError("At least one season must be requested")

    initialize_fastf1_cache(data_config.fastf1_cache_dir)
    event_references = tuple(
        reference
        for season in requested_seasons
        for reference in discover_season_events(season, events)
    )
    successful: list[SuccessfulEventBuild] = []
    failed: list[FailedEventBuild] = []
    event_frames: list[pd.DataFrame] = []

    for reference in event_references:
        _report(
            progress,
            f"EVENT {reference.season} {reference.event}: starting pipeline",
        )
        try:
            ingestion = ingest_event(
                reference.season,
                reference.event,
                data_config,
                force=force,
                fail_fast=True,
                progress=progress,
            )
            if ingestion.failed_count:
                failure = next(result for result in ingestion.results if result.status == "failed")
                raise RuntimeError(failure.error_message or "FastF1 session ingestion failed")

            build_session_features(
                reference.season,
                reference.event,
                data_config,
                feature_config,
                force=force,
                progress=progress,
            )
            modeling = build_modeling_dataset_files(
                reference.season,
                reference.event,
                data_config,
                force=force,
                progress=progress,
            )
            frame = pd.read_parquet(modeling.output_path)
            frame["event_order"] = reference.round_number
            event_frames.append(frame)
            successful.append(
                SuccessfulEventBuild(
                    season=reference.season,
                    event=reference.event,
                    event_name=reference.event_name,
                    rows=len(frame),
                    output_path=_portable_path(modeling.output_path, data_config.project_root),
                )
            )
            _report(progress, f"OK    {reference.season} {reference.event}: {len(frame)} rows")
        except Exception as exc:
            error_message = _concise_error(exc)
            failed.append(
                FailedEventBuild(
                    season=reference.season,
                    event=reference.event,
                    event_name=reference.event_name,
                    error_message=error_message,
                )
            )
            _report(progress, f"FAIL  {reference.season} {reference.event}: {error_message}")
            if fail_fast:
                break

    combined = combine_event_datasets(event_frames)
    output_path = build_combined_dataset_path(data_config.modeling_output_dir)
    if not combined.empty:
        ensure_directory(output_path.parent)
        combined.to_parquet(output_path, engine="pyarrow", index=False)

    report_path = data_config.metrics_output_dir / "dataset_build_report.json"
    summary = SeasonDatasetBuildSummary(
        requested_seasons=requested_seasons,
        n_events_requested=len(event_references),
        successful_events=tuple(successful),
        failed_events=tuple(failed),
        n_rows=len(combined),
        n_drivers=combined["driver"].nunique() if not combined.empty else 0,
        checkpoints=tuple(CHECKPOINT_SESSIONS),
        output_path=output_path,
        report_path=report_path,
    )
    save_dataset_build_report(summary, data_config.project_root)
    return summary


def combine_event_datasets(event_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate event datasets into a deterministic combined dataset."""
    if not event_frames:
        return pd.DataFrame()
    combined = pd.concat(event_frames, ignore_index=True, sort=False)
    checkpoint_order = {name: index for index, name in enumerate(CHECKPOINT_SESSIONS)}
    combined["_checkpoint_order"] = combined["checkpoint"].map(checkpoint_order)
    event_sort_column = "event_order" if "event_order" in combined else "_event_appearance"
    combined["_event_appearance"] = combined.groupby(["season", "event_slug"], sort=False).ngroup()
    combined = combined.sort_values(
        ["season", event_sort_column, "_checkpoint_order", "quali_position", "driver"],
        kind="stable",
    ).drop(columns=["_checkpoint_order", "_event_appearance"])
    return combined.reset_index(drop=True)


def combine_event_dataset_files(event_paths: Sequence[Path]) -> pd.DataFrame:
    """Read event-level Parquet files and concatenate them."""
    return combine_event_datasets([pd.read_parquet(path) for path in event_paths])


def build_combined_dataset_path(modeling_output_dir: Path) -> Path:
    """Return the fixed combined modeling dataset path."""
    return modeling_output_dir / "combined" / "modeling_dataset.parquet"


def build_dataset_report_payload(
    summary: SeasonDatasetBuildSummary,
    project_root: Path,
) -> dict[str, object]:
    """Build the serializable dataset report required by the milestone."""
    return {
        "requested_seasons": list(summary.requested_seasons),
        "n_events_requested": summary.n_events_requested,
        "n_events_successful": summary.n_events_successful,
        "n_events_failed": summary.n_events_failed,
        "successful_events": [asdict(event) for event in summary.successful_events],
        "failed_events": [asdict(event) for event in summary.failed_events],
        "n_rows": summary.n_rows,
        "n_drivers": summary.n_drivers,
        "checkpoints": list(summary.checkpoints),
        "output_path": _portable_path(summary.output_path, project_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def save_dataset_build_report(
    summary: SeasonDatasetBuildSummary,
    project_root: Path,
) -> None:
    """Persist the multi-event build report as readable JSON."""
    ensure_directory(summary.report_path.parent)
    payload = build_dataset_report_payload(summary, project_root)
    with summary.report_path.open("w", encoding="utf-8") as report_file:
        json.dump(payload, report_file, indent=2, ensure_ascii=False)
        report_file.write("\n")


def _event_reference(season: int, row: pd.Series) -> EventReference:
    location = str(row.get("Location") or "").strip()
    event_name = str(row.get("EventName") or location).strip()
    event = location or event_name
    return EventReference(
        season=season,
        event=event,
        event_name=event_name,
        round_number=int(row["RoundNumber"]),
    )


def _event_aliases(reference: EventReference) -> set[str]:
    return {slugify(reference.event), slugify(reference.event_name)}


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _concise_error(exc: Exception) -> str:
    message = " ".join(str(exc).split()) or "No error details were provided"
    return f"{type(exc).__name__}: {message}"


def _report(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
