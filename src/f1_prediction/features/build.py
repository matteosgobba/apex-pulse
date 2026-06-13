"""File orchestration for cleaned laps and practice-session features."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from f1_prediction.config import DataConfig, FeatureConfig
from f1_prediction.data.fastf1_loader import build_lap_output_path
from f1_prediction.data.ingest import normalize_session_identifiers
from f1_prediction.features.lap_cleaning import (
    build_clean_lap_output_path,
    clean_session_laps,
)
from f1_prediction.features.push_laps import add_push_lap_flags
from f1_prediction.features.session_aggregates import aggregate_session_features
from f1_prediction.utils.paths import ensure_directory, slugify

DEFAULT_PRACTICE_SESSIONS: tuple[str, ...] = ("FP1", "FP2", "FP3")
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class SessionFeatureBuildSummary:
    """Outcome of one event-level practice feature build."""

    season: int
    event: str
    sessions: tuple[str, ...]
    clean_lap_paths: tuple[Path, ...]
    clean_lap_files_written: int
    aggregate_rows: int
    output_path: Path


def build_session_features(
    season: int,
    event: str,
    data_config: DataConfig,
    feature_config: FeatureConfig,
    sessions: Sequence[str] = DEFAULT_PRACTICE_SESSIONS,
    *,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> SessionFeatureBuildSummary:
    """Build cleaned laps and driver/session aggregates from raw Parquet files."""
    requested_sessions = normalize_session_identifiers(sessions)
    clean_paths = tuple(
        build_clean_lap_output_path(
            data_config.clean_lap_output_dir,
            season,
            event,
            session,
        )
        for session in requested_sessions
    )
    output_path = build_session_features_output_path(
        data_config.session_features_output_dir,
        season,
        event,
    )
    existing_features = _read_existing_features(output_path)

    if (
        not force
        and all(path.is_file() for path in clean_paths)
        and _contains_sessions(existing_features, season, event, requested_sessions)
    ):
        _report(progress, "SKIP: cleaned laps and aggregate features already exist")
        return SessionFeatureBuildSummary(
            season=season,
            event=event,
            sessions=requested_sessions,
            clean_lap_paths=clean_paths,
            clean_lap_files_written=0,
            aggregate_rows=len(existing_features),
            output_path=output_path,
        )

    cleaned_frames: list[pd.DataFrame] = []
    written = 0
    for session, clean_path in zip(requested_sessions, clean_paths, strict=True):
        if clean_path.is_file() and not force:
            _report(progress, f"SKIP {session}: cleaned laps already exist")
            cleaned = pd.read_parquet(clean_path)
        else:
            raw_path = build_lap_output_path(
                data_config.lap_output_dir,
                season,
                event,
                session,
            )
            if not raw_path.is_file():
                raise FileNotFoundError(
                    f"Raw lap file does not exist for {session}: {raw_path}. "
                    "Run ingest-event first."
                )
            _report(progress, f"BUILD {session}: cleaning raw laps")
            raw_laps = pd.read_parquet(raw_path)
            cleaned = clean_session_laps(
                raw_laps,
                season=season,
                event=event,
                session=session,
            )
            cleaned = add_push_lap_flags(cleaned, feature_config.push_lap)
            cleaned.to_parquet(clean_path, engine="pyarrow", index=False)
            written += 1
            _report(progress, f"OK    {session}: {len(cleaned)} cleaned laps")
        cleaned_frames.append(cleaned)

    requested_features = aggregate_session_features(pd.concat(cleaned_frames, ignore_index=True))
    merged_features = _replace_requested_features(
        existing_features,
        requested_features,
        season,
        event,
        requested_sessions,
    )
    ensure_directory(output_path.parent)
    merged_features.to_parquet(output_path, engine="pyarrow", index=False)
    _report(progress, f"OK    aggregate: {len(merged_features)} rows")

    return SessionFeatureBuildSummary(
        season=season,
        event=event,
        sessions=requested_sessions,
        clean_lap_paths=clean_paths,
        clean_lap_files_written=written,
        aggregate_rows=len(merged_features),
        output_path=output_path,
    )


def build_session_features_output_path(output_dir: Path, season: int, event: str) -> Path:
    """Build the deterministic path for event-level practice features."""
    event_dir = ensure_directory(output_dir / str(season) / slugify(event))
    return event_dir / "practice_session_features.parquet"


def _read_existing_features(output_path: Path) -> pd.DataFrame:
    if not output_path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(output_path)


def _contains_sessions(
    features: pd.DataFrame,
    season: int,
    event: str,
    sessions: tuple[str, ...],
) -> bool:
    if features.empty:
        return False
    event_rows = features[features["season"].eq(season) & features["event_slug"].eq(slugify(event))]
    return set(sessions).issubset(set(event_rows["session"]))


def _replace_requested_features(
    existing: pd.DataFrame,
    requested: pd.DataFrame,
    season: int,
    event: str,
    sessions: tuple[str, ...],
) -> pd.DataFrame:
    if existing.empty:
        return requested.reset_index(drop=True)
    replace_mask = (
        existing["season"].eq(season)
        & existing["event_slug"].eq(slugify(event))
        & existing["session"].isin(sessions)
    )
    preserved = existing.loc[~replace_mask]
    return pd.concat([preserved, requested], ignore_index=True)


def _report(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
