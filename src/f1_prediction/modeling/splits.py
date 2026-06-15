"""Leakage-safe event, season, and walk-forward dataset splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.data.season_builder import build_combined_dataset_path
from f1_prediction.features.data_quality import DATA_QUALITY_FEATURE_COLUMNS
from f1_prediction.features.historical_features import HISTORICAL_FEATURE_COLUMNS
from f1_prediction.features.modeling_dataset import get_feature_columns
from f1_prediction.features.qualifying_targets import TARGET_COLUMNS
from f1_prediction.utils.paths import ensure_directory, slugify


class SplitStrategy(str, Enum):
    """Supported time-aware split strategies."""

    event_holdout = "event_holdout"
    season_holdout = "season_holdout"
    walk_forward = "walk_forward"


@dataclass(frozen=True)
class DatasetSplit:
    """Row indices and serializable metadata for one dataset split."""

    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DatasetSplitSummary:
    """CLI-facing split report summary."""

    strategy: str
    train_rows: int
    test_rows: int
    folds: int
    report_path: Path


def create_dataset_split(
    dataset: pd.DataFrame,
    strategy: SplitStrategy | str,
    *,
    test_events: list[str] | None = None,
    test_seasons: list[int] | None = None,
    min_train_events: int = 3,
) -> DatasetSplit:
    """Create one leakage-safe split or walk-forward split collection."""
    frame = dataset.reset_index(drop=True)
    _validate_dataset(frame)
    strategy = SplitStrategy(strategy)
    if strategy is SplitStrategy.event_holdout:
        return _event_holdout(frame, test_events or [])
    if strategy is SplitStrategy.season_holdout:
        return _season_holdout(frame, test_seasons or [])
    return _walk_forward(frame, min_train_events=min_train_events)


def write_dataset_split_report(
    config: DataConfig,
    *,
    strategy: SplitStrategy | str,
    dataset_path: Path | None = None,
    test_events: list[str] | None = None,
    test_seasons: list[int] | None = None,
    min_train_events: int = 3,
) -> DatasetSplitSummary:
    """Read a dataset, create split metadata, and save it as JSON."""
    source_path = _resolve_dataset_path(config, dataset_path)
    split = create_dataset_split(
        pd.read_parquet(source_path),
        strategy,
        test_events=test_events,
        test_seasons=test_seasons,
        min_train_events=min_train_events,
    )
    payload = {
        **split.metadata,
        "dataset_path": _portable_path(source_path, config.project_root),
        "created_at_utc": _utc_now(),
    }
    report_path = config.metrics_output_dir / "dataset_splits.json"
    ensure_directory(report_path.parent)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(payload, report_file, indent=2, allow_nan=False)
        report_file.write("\n")
    return DatasetSplitSummary(
        strategy=str(payload["strategy"]),
        train_rows=int(payload["train_rows"]),
        test_rows=int(payload["test_rows"]),
        folds=len(payload.get("folds", [])),
        report_path=report_path,
    )


def ordered_event_keys(dataset: pd.DataFrame) -> list[str]:
    """Return events in chronological order using round order when available."""
    events = dataset.loc[:, ["season", "event_slug"]].copy()
    events["_appearance"] = range(len(events))
    if "event_order" in dataset:
        events["event_order"] = pd.to_numeric(dataset["event_order"], errors="coerce")
    else:
        events["event_order"] = events.groupby("season", sort=False)["event_slug"].transform(
            lambda values: pd.factorize(values, sort=False)[0] + 1
        )
    events = events.drop_duplicates(["season", "event_slug"])
    events = events.sort_values(["season", "event_order", "_appearance"], kind="stable")
    return [f"{row.season}/{row.event_slug}" for row in events.itertuples(index=False)]


def latest_event_holdout(dataset: pd.DataFrame) -> DatasetSplit:
    """Hold out the chronologically latest event."""
    event_key = ordered_event_keys(dataset)[-1]
    return _event_holdout(dataset.reset_index(drop=True), [event_key])


def get_numeric_feature_columns(
    dataset: pd.DataFrame,
    checkpoint: str | None = None,
) -> list[str]:
    """Select numeric practice predictors allowed at a checkpoint."""
    allowed_prefixes = {
        "after_fp1": ("fp1_",),
        "after_fp2": ("fp1_", "fp2_"),
        "after_fp3": ("fp1_", "fp2_", "fp3_"),
    }
    if checkpoint is not None and checkpoint not in allowed_prefixes:
        raise ValueError(f"Unsupported checkpoint: {checkpoint}")
    columns = get_feature_columns(dataset)
    if checkpoint is not None:
        always_available = set(HISTORICAL_FEATURE_COLUMNS) | set(DATA_QUALITY_FEATURE_COLUMNS)
        columns = [
            column
            for column in columns
            if column.startswith(allowed_prefixes[checkpoint]) or column in always_available
        ]
    return [column for column in columns if pd.api.types.is_numeric_dtype(dataset[column])]


def _event_holdout(dataset: pd.DataFrame, requested_events: list[str]) -> DatasetSplit:
    if not requested_events:
        raise ValueError("event_holdout requires at least one --test-events value")
    event_keys = _event_key_series(dataset)
    available = event_keys.drop_duplicates().tolist()
    selected = {
        key
        for key in available
        if any(_event_matches(key, requested) for requested in requested_events)
    }
    if not selected:
        raise ValueError(f"No requested test events were found: {', '.join(requested_events)}")
    test_mask = event_keys.isin(selected)
    split = _basic_split(dataset, ~test_mask, test_mask, SplitStrategy.event_holdout.value)
    if set(split.metadata["train_events"]) & set(split.metadata["test_events"]):
        raise ValueError("Event leakage detected between train and test rows")
    return split


def _season_holdout(dataset: pd.DataFrame, test_seasons: list[int]) -> DatasetSplit:
    if not test_seasons:
        raise ValueError("season_holdout requires at least one --test-seasons value")
    test_mask = dataset["season"].isin(test_seasons)
    if not test_mask.any():
        raise ValueError("None of the requested test seasons exist in the dataset")
    split = _basic_split(dataset, ~test_mask, test_mask, SplitStrategy.season_holdout.value)
    if set(split.metadata["train_seasons"]) & set(split.metadata["test_seasons"]):
        raise ValueError("Season leakage detected between train and test rows")
    return split


def _walk_forward(dataset: pd.DataFrame, *, min_train_events: int) -> DatasetSplit:
    if min_train_events < 1:
        raise ValueError("min_train_events must be at least 1")
    event_keys = ordered_event_keys(dataset)
    if len(event_keys) <= min_train_events:
        raise ValueError(
            f"walk_forward needs more than {min_train_events} events; found {len(event_keys)}"
        )
    row_event_keys = _event_key_series(dataset)
    folds: list[dict[str, object]] = []
    final_train = pd.Series(False, index=dataset.index)
    final_test = pd.Series(False, index=dataset.index)
    for test_position in range(min_train_events, len(event_keys)):
        train_events = event_keys[:test_position]
        test_event = event_keys[test_position]
        train_mask = row_event_keys.isin(train_events)
        test_mask = row_event_keys.eq(test_event)
        folds.append(
            {
                "fold_id": len(folds) + 1,
                "train_events": train_events,
                "test_event": test_event,
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
            }
        )
        final_train, final_test = train_mask, test_mask
    split = _basic_split(dataset, final_train, final_test, SplitStrategy.walk_forward.value)
    split.metadata["folds"] = folds
    split.metadata["ordering"] = "season_then_event_order_or_dataset_appearance"
    return split


def _basic_split(
    dataset: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    strategy: str,
) -> DatasetSplit:
    if not test_mask.any():
        raise ValueError("The test split must contain rows")
    event_keys = _event_key_series(dataset)
    metadata = {
        "strategy": strategy,
        "n_rows": len(dataset),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_events": event_keys[train_mask].drop_duplicates().tolist(),
        "test_events": event_keys[test_mask].drop_duplicates().tolist(),
        "train_seasons": sorted(_native_ints(dataset.loc[train_mask, "season"].unique())),
        "test_seasons": sorted(_native_ints(dataset.loc[test_mask, "season"].unique())),
    }
    return DatasetSplit(
        train_indices=tuple(dataset.index[train_mask].astype(int)),
        test_indices=tuple(dataset.index[test_mask].astype(int)),
        metadata=metadata,
    )


def _validate_dataset(dataset: pd.DataFrame) -> None:
    required = {"season", "event", "event_slug", "checkpoint", "driver"}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"Modeling dataset is missing columns: {', '.join(missing)}")
    leaked = [column for column in get_feature_columns(dataset) if column in TARGET_COLUMNS]
    leaked.extend(column for column in get_feature_columns(dataset) if column.startswith("quali_"))
    if leaked:
        raise ValueError(f"Target leakage detected in feature columns: {', '.join(leaked)}")


def _event_key_series(dataset: pd.DataFrame) -> pd.Series:
    return dataset["season"].astype(str) + "/" + dataset["event_slug"].astype(str)


def _event_matches(event_key: str, requested: str) -> bool:
    requested_slug = slugify(requested)
    season, event_slug = event_key.split("/", maxsplit=1)
    return requested_slug in {event_slug, slugify(event_key), slugify(f"{season}-{event_slug}")}


def _resolve_dataset_path(config: DataConfig, dataset_path: Path | None) -> Path:
    path = dataset_path or build_combined_dataset_path(config.modeling_output_dir)
    if not path.is_absolute():
        path = config.project_root / path
    if not path.is_file():
        raise FileNotFoundError(f"Modeling dataset does not exist: {path}")
    return path


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _native_ints(values: object) -> list[int]:
    return [int(value) for value in values]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
