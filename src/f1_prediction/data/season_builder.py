"""Multi-event dataset construction using the existing event pipeline."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import fastf1
import pandas as pd

from f1_prediction.config import DataConfig, FeatureConfig
from f1_prediction.data.cache import initialize_fastf1_cache
from f1_prediction.data.identity import add_identity_columns
from f1_prediction.data.ingest import ingest_event
from f1_prediction.features.build import build_session_features
from f1_prediction.features.historical_features import (
    HistoricalFeatureSettings,
    add_historical_features,
)
from f1_prediction.features.modeling_dataset import (
    CHECKPOINT_SESSIONS,
    build_modeling_dataset_files,
)
from f1_prediction.utils.paths import ensure_directory, slugify

ProgressCallback = Callable[[str], None]
CONVENTIONAL_2024_EVENTS: tuple[str, ...] = (
    "Bahrain",
    "Australia",
    "Japan",
    "Imola",
    "Monaco",
    "Canada",
    "Spain",
    "Silverstone",
    "Hungary",
    "Netherlands",
    "Monza",
    "Abu Dhabi",
)
CONVENTIONAL_2023_EVENTS: tuple[str, ...] = (
    "Bahrain",
    "Saudi Arabia",
    "Australia",
    "Miami",
    "Monaco",
    "Spain",
    "Canada",
    "Great Britain",
    "Hungary",
    "Netherlands",
    "Italy",
    "Singapore",
    "Japan",
    "Abu Dhabi",
)
EVENT_PRESETS: dict[str, tuple[str, ...]] = {
    "conventional_2023": CONVENTIONAL_2023_EVENTS,
    "conventional_2024": CONVENTIONAL_2024_EVENTS,
}
CONVENTIONAL_EVENTS_BY_SEASON: dict[int, tuple[str, ...]] = {
    2023: CONVENTIONAL_2023_EVENTS,
    2024: CONVENTIONAL_2024_EVENTS,
}
EventSelection = tuple[str, ...] | dict[int, tuple[str, ...]] | None
COMMON_EVENT_ALIASES: dict[str, tuple[str, ...]] = {
    "abu-dhabi": ("yas-island", "united-arab-emirates"),
    "emilia-romagna": ("imola",),
}


@dataclass(frozen=True)
class EventReference:
    """One scheduled event selected for dataset construction."""

    season: int
    event: str
    event_name: str
    round_number: int
    country: str = ""
    official_event_name: str = ""


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
    n_teams: int = 0
    n_columns: int = 0
    rows_by_checkpoint: tuple[tuple[str, int], ...] = ()
    events_by_checkpoint: tuple[tuple[str, int], ...] = ()
    rows_by_season: tuple[tuple[str, int], ...] = ()
    rows_by_event: tuple[tuple[str, int], ...] = ()
    preset: str | None = None

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
            reference
            for reference in references
            if requested_slug in _event_aliases(reference)
            or _common_alias_matches(requested_slug, reference)
        ]
        if not matches:
            raise ValueError(f"Event '{requested}' was not found in the {season} FastF1 schedule")
        if matches[0] not in selected:
            selected.append(matches[0])
    return tuple(selected)


def resolve_event_selection(
    seasons: Sequence[int],
    events: Sequence[str] | None = None,
    preset: str | None = None,
) -> EventSelection:
    """Resolve explicit events or a documented convenience preset."""
    if events and preset:
        raise ValueError("Use either explicit --events values or --preset, not both")
    if not preset:
        return tuple(events) if events else None
    if preset == "conventional":
        unsupported = sorted(set(seasons) - set(CONVENTIONAL_EVENTS_BY_SEASON))
        if unsupported:
            values = ", ".join(str(season) for season in unsupported)
            raise ValueError(f"The conventional preset is unavailable for seasons: {values}")
        return {season: CONVENTIONAL_EVENTS_BY_SEASON[season] for season in dict.fromkeys(seasons)}
    if preset not in EVENT_PRESETS:
        available = ", ".join([*sorted(EVENT_PRESETS), "conventional"])
        raise ValueError(f"Unknown event preset '{preset}'. Available presets: {available}")
    unique_seasons = tuple(dict.fromkeys(seasons))
    expected_season = int(preset.rsplit("_", maxsplit=1)[-1])
    if unique_seasons != (expected_season,):
        raise ValueError(f"The {preset} preset requires exactly --season {expected_season}")
    return EVENT_PRESETS[preset]


def build_season_dataset(
    seasons: Sequence[int],
    data_config: DataConfig,
    feature_config: FeatureConfig,
    *,
    events: Sequence[str] | Mapping[int, Sequence[str]] | None = None,
    preset: str | None = None,
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
        for reference in _discover_or_fallback_events(
            season,
            _events_for_season(events, season),
            progress,
        )
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
                force=False,
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
                feature_config=feature_config,
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

    historical_config = feature_config.historical_features
    historical_settings = (
        HistoricalFeatureSettings(
            rolling_windows=historical_config.rolling_windows,
            min_periods=historical_config.min_periods,
        )
        if historical_config is not None
        else HistoricalFeatureSettings()
    )
    combined = combine_event_datasets(event_frames, historical_settings=historical_settings)
    output_path = build_combined_dataset_path(data_config.modeling_output_dir)
    if not combined.empty:
        ensure_directory(output_path.parent)
        combined.to_parquet(output_path, engine="pyarrow", index=False)
    else:
        output_path.unlink(missing_ok=True)

    report_path = data_config.metrics_output_dir / "dataset_build_report.json"
    summary = SeasonDatasetBuildSummary(
        requested_seasons=requested_seasons,
        n_events_requested=len(event_references),
        successful_events=tuple(successful),
        failed_events=tuple(failed),
        n_rows=len(combined),
        n_drivers=combined["driver"].nunique() if not combined.empty else 0,
        n_teams=(
            combined["team_key"].nunique() if not combined.empty and "team_key" in combined else 0
        ),
        checkpoints=tuple(CHECKPOINT_SESSIONS),
        output_path=output_path,
        report_path=report_path,
        n_columns=len(combined.columns),
        rows_by_checkpoint=_checkpoint_counts(combined),
        events_by_checkpoint=_checkpoint_event_counts(combined),
        rows_by_season=_season_counts(combined),
        rows_by_event=_event_counts(combined),
        preset=preset,
    )
    save_dataset_build_report(summary, data_config.project_root)
    return summary


def combine_event_datasets(
    event_frames: Sequence[pd.DataFrame],
    *,
    historical_settings: HistoricalFeatureSettings | None = None,
) -> pd.DataFrame:
    """Concatenate event datasets into a deterministic combined dataset."""
    if not event_frames:
        return pd.DataFrame()
    combined = pd.concat(event_frames, ignore_index=True, sort=False)
    combined = add_identity_columns(combined)
    checkpoint_order = {name: index for index, name in enumerate(CHECKPOINT_SESSIONS)}
    combined["_checkpoint_order"] = combined["checkpoint"].map(checkpoint_order)
    event_sort_column = "event_order" if "event_order" in combined else "_event_appearance"
    combined["_event_appearance"] = combined.groupby(["season", "event_slug"], sort=False).ngroup()
    combined = combined.sort_values(
        ["season", event_sort_column, "_checkpoint_order", "quali_position", "driver"],
        kind="stable",
    ).drop(columns=["_checkpoint_order", "_event_appearance"])
    combined = combined.reset_index(drop=True)
    if {"team", "quali_gap_to_pole_sec", "reached_q3"} <= set(combined.columns):
        combined = add_historical_features(combined, historical_settings)
    return combined


def combine_event_dataset_files(
    event_paths: Sequence[Path],
    *,
    historical_settings: HistoricalFeatureSettings | None = None,
) -> pd.DataFrame:
    """Read event-level Parquet files and concatenate them."""
    return combine_event_datasets(
        [pd.read_parquet(path) for path in event_paths],
        historical_settings=historical_settings,
    )


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
        "preset": summary.preset,
        "n_events_requested": summary.n_events_requested,
        "n_events_successful": summary.n_events_successful,
        "n_events_failed": summary.n_events_failed,
        "successful_events": [asdict(event) for event in summary.successful_events],
        "failed_events": [asdict(event) for event in summary.failed_events],
        "n_rows": summary.n_rows,
        "n_columns": summary.n_columns,
        "n_drivers": summary.n_drivers,
        "n_teams": summary.n_teams,
        "checkpoints": list(summary.checkpoints),
        "rows_by_checkpoint": dict(summary.rows_by_checkpoint),
        "rows_by_season": dict(summary.rows_by_season),
        "rows_by_event": dict(summary.rows_by_event),
        "events_by_checkpoint": dict(summary.events_by_checkpoint),
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
    country = str(row.get("Country") or "").strip()
    official_event_name = str(row.get("OfficialEventName") or "").strip()
    event = location or event_name
    return EventReference(
        season=season,
        event=event,
        event_name=event_name,
        round_number=int(row["RoundNumber"]),
        country=country,
        official_event_name=official_event_name,
    )


def _event_aliases(reference: EventReference) -> set[str]:
    aliases = {
        slugify(value)
        for value in (
            reference.event,
            reference.event_name,
            reference.country,
            reference.official_event_name,
        )
        if value
    }
    aliases.update(_event_name_bases(aliases))
    return aliases


def _event_name_bases(aliases: set[str]) -> set[str]:
    suffixes = ("-grand-prix", "-gp")
    bases: set[str] = set()
    for alias in aliases:
        for suffix in suffixes:
            if alias.endswith(suffix):
                bases.add(alias.removesuffix(suffix))
    return bases


def _common_alias_matches(requested_slug: str, reference: EventReference) -> bool:
    reference_aliases = _event_aliases(reference)
    for common_name, aliases in COMMON_EVENT_ALIASES.items():
        alias_group = {common_name, *aliases}
        if requested_slug in alias_group and reference_aliases & alias_group:
            return True
    return False


def _checkpoint_counts(dataset: pd.DataFrame) -> tuple[tuple[str, int], ...]:
    if dataset.empty or "checkpoint" not in dataset:
        return ()
    return tuple(
        (str(checkpoint), int(count))
        for checkpoint, count in dataset.groupby("checkpoint", sort=False).size().items()
    )


def _checkpoint_event_counts(dataset: pd.DataFrame) -> tuple[tuple[str, int], ...]:
    if dataset.empty or "checkpoint" not in dataset:
        return ()
    counts = dataset.groupby("checkpoint", sort=False)[["season", "event_slug"]].apply(
        lambda rows: len(rows.drop_duplicates())
    )
    return tuple((str(checkpoint), int(count)) for checkpoint, count in counts.items())


def _season_counts(dataset: pd.DataFrame) -> tuple[tuple[str, int], ...]:
    if dataset.empty or "season" not in dataset:
        return ()
    return tuple(
        (str(season), int(count))
        for season, count in dataset.groupby("season", sort=True).size().items()
    )


def _event_counts(dataset: pd.DataFrame) -> tuple[tuple[str, int], ...]:
    if dataset.empty or not {"season", "event_slug"} <= set(dataset.columns):
        return ()
    keys = dataset["season"].astype(str) + "/" + dataset["event_slug"].astype(str)
    return tuple((str(event), int(count)) for event, count in keys.value_counts(sort=False).items())


def _events_for_season(
    events: Sequence[str] | Mapping[int, Sequence[str]] | None,
    season: int,
) -> Sequence[str] | None:
    if isinstance(events, Mapping):
        return events.get(season)
    return events


def _discover_or_fallback_events(
    season: int,
    events: Sequence[str] | None,
    progress: ProgressCallback | None,
) -> tuple[EventReference, ...]:
    try:
        return discover_season_events(season, events)
    except Exception as exc:
        if not events:
            raise
        _report(
            progress,
            f"WARN  {season}: schedule lookup failed; using requested event names "
            f"({_concise_error(exc)})",
        )
        return tuple(
            EventReference(
                season=season,
                event=event,
                event_name=event,
                round_number=index,
            )
            for index, event in enumerate(events, start=1)
        )


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
