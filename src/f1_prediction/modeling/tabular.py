"""Simple checkpoint-safe sklearn regression models."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from f1_prediction.config import ModelConfig
from f1_prediction.modeling.splits import get_numeric_feature_columns

TARGET_COLUMN = "quali_gap_to_pole_sec"
MODEL_NAMES: tuple[str, ...] = (
    "mean_target",
    "median_target",
    "ridge",
    "random_forest",
)


def build_regressors(config: ModelConfig) -> dict[str, object]:
    """Create conservative Ridge and Random Forest pipelines."""
    ridge_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=config.ridge_alpha)),
        ]
    )
    random_forest_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=config.random_forest.n_estimators,
                    max_depth=config.random_forest.max_depth,
                    min_samples_leaf=config.random_forest.min_samples_leaf,
                    random_state=config.random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )
    return {"ridge": ridge_pipeline, "random_forest": random_forest_pipeline}


def usable_checkpoint_features(dataset: pd.DataFrame, checkpoint: str) -> list[str]:
    """Return allowed numeric features that contain training data."""
    return [
        column
        for column in get_numeric_feature_columns(dataset, checkpoint)
        if dataset[column].notna().any()
    ]


def rank_gap_predictions(predictions: pd.DataFrame) -> pd.Series:
    """Convert predicted gaps into ordinal positions within event/checkpoint."""
    required = {
        "season",
        "event_slug",
        "checkpoint",
        "driver",
        "predicted_quali_gap_to_pole_sec",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing columns: {', '.join(missing)}")
    positions = pd.Series(index=predictions.index, dtype="Int64")
    for _, group in predictions.groupby(["season", "event_slug", "checkpoint"], sort=False):
        ordered = group.assign(
            _missing=group["predicted_quali_gap_to_pole_sec"].isna()
        ).sort_values(
            ["_missing", "predicted_quali_gap_to_pole_sec", "driver"],
            kind="stable",
            na_position="last",
        )
        positions.loc[ordered.index] = range(1, len(ordered) + 1)
    return positions


def checkpoint_feature_manifest(models: Mapping[str, object]) -> dict[str, list[str]]:
    """Extract persisted feature lists from a checkpoint model bundle."""
    return {checkpoint: list(bundle["feature_columns"]) for checkpoint, bundle in models.items()}
