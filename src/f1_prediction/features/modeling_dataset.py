"""Checkpoint-safe modeling rows built from practice features and Q targets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from f1_prediction.config import DataConfig, DataQualityConfig, FeatureConfig
from f1_prediction.data.fastf1_loader import build_lap_output_path
from f1_prediction.features.build import build_session_features_output_path
from f1_prediction.features.data_quality import (
    DATA_QUALITY_FEATURE_COLUMNS,
    DataQualitySettings,
    add_data_quality_features,
)
from f1_prediction.features.historical_features import HISTORICAL_FEATURE_COLUMNS
from f1_prediction.features.qualifying_targets import TARGET_COLUMNS, build_qualifying_targets
from f1_prediction.features.relative_features import add_relative_practice_features
from f1_prediction.utils.paths import ensure_directory, slugify

CHECKPOINT_SESSIONS: dict[str, tuple[str, ...]] = {
    "after_fp1": ("FP1",),
    "after_fp2": ("FP1", "FP2"),
    "after_fp3": ("FP1", "FP2", "FP3"),
}
IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "season",
    "event",
    "event_slug",
    "checkpoint",
    "driver",
    "team",
)
SESSION_IDENTIFIER_COLUMNS: frozenset[str] = frozenset(
    {"season", "event", "event_slug", "session", "session_slug", "driver", "team"}
)
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class ModelingDatasetBuildSummary:
    """Outcome of building one event's checkpoint modeling dataset."""

    season: int
    event: str
    rows: int
    drivers: int
    checkpoints: tuple[str, ...]
    qualifying_only_drivers: tuple[str, ...]
    practice_only_drivers: tuple[str, ...]
    output_path: Path
    skipped: bool = False


def build_checkpoint_modeling_dataset(
    practice_features: pd.DataFrame,
    qualifying_targets: pd.DataFrame,
    data_quality_config: DataQualityConfig | None = None,
) -> pd.DataFrame:
    """Build after-FP1/FP2/FP3 rows without exposing future practice values."""
    _validate_target_columns(qualifying_targets)
    _validate_unique_session_driver_rows(practice_features)
    practice_features = add_relative_practice_features(practice_features)
    source_feature_columns = [
        column
        for column in practice_features.columns
        if column not in SESSION_IDENTIFIER_COLUMNS
        and not column.lower().startswith(("q_", "quali_"))
    ]
    team_map = _driver_team_map(practice_features)
    checkpoint_frames: list[pd.DataFrame] = []

    for checkpoint, available_sessions in CHECKPOINT_SESSIONS.items():
        frame = qualifying_targets.copy()
        frame["checkpoint"] = checkpoint
        frame["team"] = frame["driver"].map(team_map)

        for session in available_sessions:
            session_rows = practice_features[practice_features["session"].eq(session)]
            renamed = session_rows.loc[:, ["driver", *source_feature_columns]].rename(
                columns={column: f"{session.lower()}_{column}" for column in source_feature_columns}
            )
            frame = frame.merge(renamed, on="driver", how="left", validate="one_to_one")

        checkpoint_frames.append(frame)

    dataset = pd.concat(checkpoint_frames, ignore_index=True, sort=False)
    dataset = add_data_quality_features(dataset, _quality_settings(data_quality_config))
    feature_columns = _ordered_feature_columns(dataset)
    ordered_columns = [*IDENTIFIER_COLUMNS, *feature_columns, *TARGET_COLUMNS]
    return dataset.loc[:, ordered_columns]


def get_feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Return predictor columns while excluding identifiers and qualifying targets."""
    return [
        column
        for column in dataset.columns
        if (
            column.startswith(("fp1_", "fp2_", "fp3_"))
            or column in HISTORICAL_FEATURE_COLUMNS
            or column in DATA_QUALITY_FEATURE_COLUMNS
        )
        and column not in TARGET_COLUMNS
    ]


def build_modeling_dataset_files(
    season: int,
    event: str,
    config: DataConfig,
    *,
    feature_config: FeatureConfig | None = None,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> ModelingDatasetBuildSummary:
    """Read existing pipeline outputs and persist one event modeling dataset."""
    output_path = build_modeling_output_path(config.modeling_output_dir, season, event)
    practice_path = build_session_features_output_path(
        config.session_features_output_dir,
        season,
        event,
    )
    if output_path.is_file() and not force:
        existing = pd.read_parquet(output_path)
        practice_features = pd.read_parquet(practice_path) if practice_path.is_file() else None
        _report(progress, "SKIP: modeling dataset already exists")
        return _build_summary(
            existing,
            season,
            event,
            output_path,
            practice_features=practice_features,
            skipped=True,
        )

    qualifying_path = build_lap_output_path(config.lap_output_dir, season, event, "Q")
    if not practice_path.is_file():
        raise FileNotFoundError(
            f"Practice feature file does not exist: {practice_path}. "
            "Run build-session-features first."
        )
    if not qualifying_path.is_file():
        raise FileNotFoundError(
            f"Raw qualifying lap file does not exist: {qualifying_path}. Run ingest-event first."
        )

    _report(progress, "BUILD: qualifying targets")
    practice_features = pd.read_parquet(practice_path)
    raw_qualifying_laps = pd.read_parquet(qualifying_path)
    targets = build_qualifying_targets(
        raw_qualifying_laps,
        season=season,
        event=event,
    )
    _report(progress, "BUILD: after_fp1, after_fp2, and after_fp3 rows")
    quality_config = feature_config.data_quality if feature_config is not None else None
    dataset = build_checkpoint_modeling_dataset(
        practice_features,
        targets,
        data_quality_config=quality_config,
    )
    ensure_directory(output_path.parent)
    dataset.to_parquet(output_path, engine="pyarrow", index=False)
    _report(progress, f"OK: {len(dataset)} modeling rows")
    return _build_summary(dataset, season, event, output_path, practice_features=practice_features)


def build_modeling_output_path(output_dir: Path, season: int, event: str) -> Path:
    """Build the deterministic event modeling dataset path."""
    event_dir = ensure_directory(output_dir / str(season) / slugify(event))
    return event_dir / "modeling_dataset.parquet"


def _build_summary(
    dataset: pd.DataFrame,
    season: int,
    event: str,
    output_path: Path,
    *,
    practice_features: pd.DataFrame | None = None,
    skipped: bool = False,
) -> ModelingDatasetBuildSummary:
    target_drivers = set(dataset["driver"].dropna().astype(str))
    practice_drivers = (
        set(practice_features["driver"].dropna().astype(str))
        if practice_features is not None
        else target_drivers
    )
    return ModelingDatasetBuildSummary(
        season=season,
        event=event,
        rows=len(dataset),
        drivers=dataset["driver"].nunique(),
        checkpoints=tuple(CHECKPOINT_SESSIONS),
        qualifying_only_drivers=tuple(sorted(target_drivers - practice_drivers)),
        practice_only_drivers=tuple(sorted(practice_drivers - target_drivers)),
        output_path=output_path,
        skipped=skipped,
    )


def _driver_team_map(practice_features: pd.DataFrame) -> dict[str, object]:
    session_order = {"FP1": 1, "FP2": 2, "FP3": 3}
    teams = practice_features.loc[
        practice_features["driver"].notna() & practice_features["team"].notna(),
        ["driver", "team", "session"],
    ].copy()
    teams["session_order"] = teams["session"].map(session_order).fillna(0)
    latest = teams.sort_values("session_order").drop_duplicates("driver", keep="last")
    return dict(zip(latest["driver"], latest["team"], strict=True))


def _ordered_feature_columns(dataset: pd.DataFrame) -> list[str]:
    available = set(get_feature_columns(dataset))
    ordered: list[str] = []
    for session in ("fp1", "fp2", "fp3"):
        ordered.extend(sorted(column for column in available if column.startswith(f"{session}_")))
    ordered.extend(column for column in DATA_QUALITY_FEATURE_COLUMNS if column in available)
    ordered.extend(column for column in HISTORICAL_FEATURE_COLUMNS if column in available)
    return ordered


def _quality_settings(config: DataQualityConfig | None) -> DataQualitySettings:
    if config is None:
        return DataQualitySettings()
    return DataQualitySettings(
        extreme_gap_to_session_best_sec=config.extreme_gap_to_session_best_sec,
        min_push_laps_latest_session=config.min_push_laps_latest_session,
        min_valid_laps_latest_session=config.min_valid_laps_latest_session,
    )


def _validate_target_columns(targets: pd.DataFrame) -> None:
    required = {"season", "event", "event_slug", "driver", *TARGET_COLUMNS}
    missing = sorted(required - set(targets.columns))
    if missing:
        raise ValueError(f"Qualifying targets are missing columns: {', '.join(missing)}")


def _validate_unique_session_driver_rows(practice_features: pd.DataFrame) -> None:
    required = {"session", "driver", "team"}
    missing = sorted(required - set(practice_features.columns))
    if missing:
        raise ValueError(f"Practice features are missing columns: {', '.join(missing)}")
    duplicates = practice_features.duplicated(["season", "event_slug", "session", "driver"])
    if duplicates.any():
        raise ValueError("Practice features contain duplicate session-driver rows")


def _report(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
