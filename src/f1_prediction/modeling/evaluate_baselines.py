"""File orchestration for non-ML baseline evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from f1_prediction.config import DataConfig, FeatureConfig
from f1_prediction.data.season_builder import build_combined_dataset_path
from f1_prediction.modeling.baselines import generate_baseline_predictions
from f1_prediction.modeling.metrics import compute_baseline_metrics
from f1_prediction.utils.paths import ensure_directory


@dataclass(frozen=True)
class BaselineEvaluationSummary:
    """Paths and counts produced by baseline evaluation."""

    dataset_path: Path
    prediction_rows: int
    baselines: tuple[str, ...]
    checkpoints: tuple[str, ...]
    metrics_path: Path
    predictions_path: Path


def evaluate_baselines(
    config: DataConfig,
    dataset_path: Path | None = None,
    *,
    feature_config: FeatureConfig | None = None,
) -> BaselineEvaluationSummary:
    """Evaluate all baselines and persist predictions plus nested metrics."""
    source_path = dataset_path or build_combined_dataset_path(config.modeling_output_dir)
    if not source_path.is_absolute():
        source_path = config.project_root / source_path
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Combined modeling dataset does not exist: {source_path}. "
            "Run build-season-dataset first."
        )

    dataset = pd.read_parquet(source_path)
    robust_threshold = (
        feature_config.baselines.robust_extreme_gap_to_session_best_sec
        if feature_config is not None and feature_config.baselines is not None
        else 3.0
    )
    predictions = generate_baseline_predictions(
        dataset,
        robust_extreme_threshold_sec=robust_threshold,
    )
    metrics = compute_baseline_metrics(predictions)
    metrics_path = config.metrics_output_dir / "baseline_metrics.json"
    predictions_path = config.metrics_output_dir / "baseline_predictions.parquet"
    ensure_directory(config.metrics_output_dir)
    predictions.to_parquet(predictions_path, engine="pyarrow", index=False)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, allow_nan=False)
        metrics_file.write("\n")

    return BaselineEvaluationSummary(
        dataset_path=source_path,
        prediction_rows=len(predictions),
        baselines=tuple(metrics),
        checkpoints=tuple(dict.fromkeys(predictions["checkpoint"].astype(str))),
        metrics_path=metrics_path,
        predictions_path=predictions_path,
    )
