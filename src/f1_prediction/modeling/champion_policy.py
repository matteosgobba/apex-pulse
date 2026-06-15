"""Checkpoint-specific champion selection from prior out-of-sample predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd

from f1_prediction.config import (
    ChampionMethodConfig,
    DataConfig,
    ModelConfig,
    UncertaintyConfig,
)
from f1_prediction.data.season_builder import build_combined_dataset_path
from f1_prediction.modeling.backtest_tabular import (
    BacktestFold,
    BacktestStrategy,
    build_backtest_folds,
)
from f1_prediction.modeling.metrics import compute_prediction_metrics
from f1_prediction.modeling.splits import ordered_event_keys
from f1_prediction.utils.paths import ensure_directory

CHECKPOINTS: tuple[str, ...] = ("after_fp1", "after_fp2", "after_fp3")
METHOD_COLUMNS: tuple[str, ...] = (
    "candidate_family",
    "model_name",
    "feature_group",
)
PREDICTION_KEY_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "season",
    "event_slug",
    "checkpoint",
    "driver",
)


class ChampionSelectionMode(str, Enum):
    """Supported champion selection modes."""

    static = "static"
    nested = "nested"


@dataclass(frozen=True)
class ChampionBacktestSummary:
    """Counts and paths produced by champion backtesting."""

    status: str
    strategy: str
    selection_mode: str
    n_events: int
    n_folds_total: int
    n_folds_successful: int
    n_folds_failed: int
    prediction_rows: int
    metrics_path: Path
    predictions_path: Path | None
    selection_path: Path | None


def resolve_static_champion_policy(
    model_config: ModelConfig,
) -> dict[str, ChampionMethodConfig]:
    """Return the configured static method for each checkpoint."""
    policy = model_config.champion_policy.static
    missing = [checkpoint for checkpoint in CHECKPOINTS if checkpoint not in policy]
    if missing:
        raise ValueError(f"Static champion policy is missing: {', '.join(missing)}")
    return {checkpoint: policy[checkpoint] for checkpoint in CHECKPOINTS}


def load_champion_candidates(
    metrics_dir: Path,
    expected_folds: tuple[BacktestFold, ...],
) -> pd.DataFrame:
    """Load and standardize available out-of-sample prediction families."""
    expected_by_event = {fold.test_event: fold.fold_id for fold in expected_folds}
    artifacts = (
        ("walk_forward", metrics_dir / "walk_forward_predictions.parquet"),
        ("ablation", metrics_dir / "ablation_predictions.parquet"),
        ("boosted", metrics_dir / "boosted_predictions.parquet"),
    )
    frames: list[pd.DataFrame] = []
    for source, path in artifacts:
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        frame = _candidate_rows_for_source(frame, source)
        if frame.empty:
            continue
        frame = frame[frame["test_event"].isin(expected_by_event)].copy()
        frame["fold_id"] = frame["test_event"].map(expected_by_event).astype("int64")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            "No walk-forward, ablation, or boosted prediction artifacts are available"
        )
    candidates = pd.concat(frames, ignore_index=True, sort=False)
    required = {
        *PREDICTION_KEY_COLUMNS,
        "test_event",
        "event",
        "team",
        "quali_position",
        "quali_gap_to_pole_sec",
        "reached_q3",
        "predicted_quali_gap_to_pole_sec",
        "predicted_quali_position",
        "predicted_reached_q3",
        *METHOD_COLUMNS,
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate predictions are missing columns: {', '.join(missing)}")
    candidates["feature_group"] = candidates["feature_group"].astype("string")
    return candidates.drop_duplicates([*PREDICTION_KEY_COLUMNS, *METHOD_COLUMNS]).reset_index(
        drop=True
    )


def select_nested_method(
    candidates: pd.DataFrame,
    *,
    fold_id: int,
    checkpoint: str,
    fallback: ChampionMethodConfig,
    selection_metric: str = "mae_gap_sec",
) -> tuple[ChampionMethodConfig, float | None, list[str], bool]:
    """Select the best method using only folds strictly before the test fold."""
    if selection_metric != "mae_gap_sec":
        raise ValueError("Only mae_gap_sec nested selection is currently supported")
    current = candidates[
        candidates["fold_id"].eq(fold_id) & candidates["checkpoint"].eq(checkpoint)
    ]
    prior = candidates[candidates["fold_id"].lt(fold_id) & candidates["checkpoint"].eq(checkpoint)]
    if prior.empty:
        return fallback, None, [], True

    available_methods = current.loc[:, METHOD_COLUMNS].drop_duplicates()
    history = prior.merge(available_methods, on=list(METHOD_COLUMNS), how="inner")
    if history.empty:
        return fallback, None, [], True
    history = history.dropna(
        subset=["quali_gap_to_pole_sec", "predicted_quali_gap_to_pole_sec"]
    ).copy()
    if history.empty:
        return fallback, None, [], True
    history["absolute_error"] = (
        history["predicted_quali_gap_to_pole_sec"] - history["quali_gap_to_pole_sec"]
    ).abs()
    metrics = (
        history.groupby(list(METHOD_COLUMNS), dropna=False, sort=False)
        .agg(
            selection_value=("absolute_error", "mean"),
            selection_source_events=("test_event", lambda values: sorted(set(values))),
        )
        .reset_index()
        .sort_values(
            ["selection_value", "candidate_family", "model_name", "feature_group"],
            kind="stable",
            na_position="last",
        )
    )
    if metrics.empty:
        return fallback, None, [], True
    best = metrics.iloc[0]
    method = ChampionMethodConfig(
        family=str(best["candidate_family"]),
        model_name=str(best["model_name"]),
        feature_group=_optional_string(best["feature_group"]),
    )
    return (
        method,
        float(best["selection_value"]),
        list(best["selection_source_events"]),
        False,
    )


def add_prior_residual_uncertainty(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    config: UncertaintyConfig,
) -> pd.DataFrame:
    """Attach intervals estimated only from earlier folds for the selected method."""
    result = predictions.copy()
    result["prediction_interval_low_sec"] = float("nan")
    result["prediction_interval_high_sec"] = float("nan")
    result["residual_std_sec"] = float("nan")
    result["uncertainty_method"] = "insufficient_history"
    group_columns = [
        "fold_id",
        "checkpoint",
        "selected_family",
        "selected_model_name",
        "selected_feature_group",
    ]
    for keys, rows in result.groupby(group_columns, dropna=False, sort=True):
        fold_id, checkpoint, family, model_name, feature_group = keys
        history = _method_rows(
            candidates[candidates["fold_id"].lt(int(fold_id))],
            ChampionMethodConfig(
                family=str(family),
                model_name=str(model_name),
                feature_group=_optional_string(feature_group),
            ),
            str(checkpoint),
        ).dropna(subset=["quali_gap_to_pole_sec", "predicted_quali_gap_to_pole_sec"])
        if len(history) < config.min_residual_count:
            continue
        residuals = (
            history["quali_gap_to_pole_sec"] - history["predicted_quali_gap_to_pole_sec"]
        ).astype(float)
        residual_std = float(residuals.std(ddof=1))
        if pd.isna(residual_std):
            continue
        half_width = config.interval_z * residual_std
        predicted = result.loc[rows.index, "predicted_quali_gap_to_pole_sec"].astype(float)
        result.loc[rows.index, "prediction_interval_low_sec"] = predicted - half_width
        result.loc[rows.index, "prediction_interval_high_sec"] = predicted + half_width
        result.loc[rows.index, "residual_std_sec"] = residual_std
        result.loc[rows.index, "uncertainty_method"] = "prior_residual_std"
    return result


def run_champion_backtest(
    config: DataConfig,
    *,
    strategy: BacktestStrategy | str = BacktestStrategy.walk_forward,
    selection_mode: ChampionSelectionMode | str = ChampionSelectionMode.nested,
    dataset_path: Path | None = None,
    min_events: int = 10,
    min_train_events: int = 5,
    model_config: ModelConfig,
) -> ChampionBacktestSummary:
    """Evaluate static or nested checkpoint champions on walk-forward folds."""
    strategy = BacktestStrategy(strategy)
    if strategy is not BacktestStrategy.walk_forward:
        raise ValueError("Champion backtesting currently supports walk_forward only")
    selection_mode = ChampionSelectionMode(selection_mode)
    source_path = _resolve_dataset_path(config, dataset_path)
    dataset = pd.read_parquet(source_path).reset_index(drop=True)
    event_keys = ordered_event_keys(dataset)
    paths = _output_paths(config.metrics_output_dir)
    ensure_directory(config.metrics_output_dir)
    if len(event_keys) < min_events:
        reason = f"Dataset has {len(event_keys)} unique events; at least {min_events} are required"
        _write_json(
            paths["metrics"],
            _skipped_payload(strategy, selection_mode, len(event_keys), reason),
        )
        paths["predictions"].unlink(missing_ok=True)
        paths["selection"].unlink(missing_ok=True)
        return ChampionBacktestSummary(
            status="skipped",
            strategy=strategy.value,
            selection_mode=selection_mode.value,
            n_events=len(event_keys),
            n_folds_total=0,
            n_folds_successful=0,
            n_folds_failed=0,
            prediction_rows=0,
            metrics_path=paths["metrics"],
            predictions_path=None,
            selection_path=None,
        )
    folds = build_backtest_folds(
        dataset,
        strategy,
        min_train_events=min_train_events,
    )
    candidates = load_champion_candidates(config.metrics_output_dir, folds)
    static_policy = resolve_static_champion_policy(model_config)
    prediction_frames: list[pd.DataFrame] = []
    selection_records: list[dict[str, object]] = []
    failed_folds = 0

    for fold in folds:
        fold_frames: list[pd.DataFrame] = []
        try:
            for checkpoint in CHECKPOINTS:
                fallback = static_policy[checkpoint]
                if selection_mode is ChampionSelectionMode.static:
                    selected = fallback
                    selection_value = None
                    source_events: list[str] = []
                    fallback_used = False
                else:
                    selected, selection_value, source_events, fallback_used = select_nested_method(
                        candidates,
                        fold_id=fold.fold_id,
                        checkpoint=checkpoint,
                        fallback=fallback,
                        selection_metric=model_config.champion_policy.selection_metric,
                    )
                rows = _method_rows(
                    candidates[candidates["fold_id"].eq(fold.fold_id)],
                    selected,
                    checkpoint,
                )
                if rows.empty:
                    raise ValueError(
                        f"Selected method unavailable for fold {fold.fold_id} {checkpoint}: "
                        f"{selected.family}/{selected.model_name}/{selected.feature_group}"
                    )
                fold_frames.append(_champion_prediction_rows(rows, selection_mode, selected))
                selection_records.append(
                    {
                        "fold_id": fold.fold_id,
                        "test_event": fold.test_event,
                        "checkpoint": checkpoint,
                        "selected_family": selected.family,
                        "selected_model_name": selected.model_name,
                        "selected_feature_group": selected.feature_group,
                        "selection_metric": model_config.champion_policy.selection_metric,
                        "selection_value": selection_value,
                        "selection_source_events": source_events,
                        "fallback_used": fallback_used,
                    }
                )
            prediction_frames.extend(fold_frames)
        except Exception:
            failed_folds += 1
            selection_records = [
                record for record in selection_records if record["fold_id"] != fold.fold_id
            ]

    if not prediction_frames:
        payload = {
            "status": "failed",
            "strategy": strategy.value,
            "selection_mode": selection_mode.value,
            "n_events": len(event_keys),
            "n_folds_total": len(folds),
            "n_folds_successful": 0,
            "n_folds_failed": failed_folds,
            "created_at_utc": _utc_now(),
        }
        _write_json(paths["metrics"], payload)
        paths["predictions"].unlink(missing_ok=True)
        paths["selection"].unlink(missing_ok=True)
        return ChampionBacktestSummary(
            status="failed",
            strategy=strategy.value,
            selection_mode=selection_mode.value,
            n_events=len(event_keys),
            n_folds_total=len(folds),
            n_folds_successful=0,
            n_folds_failed=failed_folds,
            prediction_rows=0,
            metrics_path=paths["metrics"],
            predictions_path=None,
            selection_path=None,
        )

    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    successful_fold_ids = sorted(predictions["fold_id"].unique().tolist())
    predictions = add_prior_residual_uncertainty(
        predictions,
        candidates,
        model_config.uncertainty,
    )
    selection = pd.DataFrame(selection_records)
    predictions.to_parquet(paths["predictions"], engine="pyarrow", index=False)
    selection.to_parquet(paths["selection"], engine="pyarrow", index=False)
    payload = build_champion_metrics_payload(
        strategy,
        selection_mode,
        len(event_keys),
        len(folds),
        failed_folds,
        predictions,
        candidates[candidates["fold_id"].isin(successful_fold_ids)],
    )
    _write_json(paths["metrics"], payload)
    return ChampionBacktestSummary(
        status=str(payload["status"]),
        strategy=strategy.value,
        selection_mode=selection_mode.value,
        n_events=len(event_keys),
        n_folds_total=len(folds),
        n_folds_successful=len(successful_fold_ids),
        n_folds_failed=failed_folds,
        prediction_rows=len(predictions),
        metrics_path=paths["metrics"],
        predictions_path=paths["predictions"],
        selection_path=paths["selection"],
    )


def build_champion_metrics_payload(
    strategy: BacktestStrategy | str,
    selection_mode: ChampionSelectionMode | str,
    n_events: int,
    n_folds_total: int,
    n_folds_failed: int,
    champion_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
) -> dict[str, object]:
    """Compute champion, baseline, and fixed-method comparisons."""
    strategy = BacktestStrategy(strategy)
    selection_mode = ChampionSelectionMode(selection_mode)
    checkpoints = champion_predictions["checkpoint"].drop_duplicates().astype(str).tolist()
    champion_metrics = {
        str(checkpoint): compute_prediction_metrics(rows)
        for checkpoint, rows in champion_predictions.groupby("checkpoint", sort=False)
    }
    best_baselines = _best_candidate_by_checkpoint(
        candidate_predictions[
            candidate_predictions["candidate_family"].isin(["baseline", "robust_baseline"])
        ],
        checkpoints,
        champion_predictions,
    )
    best_single = _best_candidate_by_checkpoint(
        candidate_predictions,
        checkpoints,
        champion_predictions,
    )
    baseline_mae: dict[str, float | None] = {}
    baseline_position: dict[str, float | None] = {}
    single_mae: dict[str, float | None] = {}
    for checkpoint in checkpoints:
        champion = champion_metrics.get(checkpoint, {})
        baseline = best_baselines.get(checkpoint, {})
        single = best_single.get(checkpoint, {})
        baseline_mae[checkpoint] = _delta(champion.get("mae_gap_sec"), baseline.get("mae_gap_sec"))
        baseline_position[checkpoint] = _delta(
            champion.get("mean_abs_position_error"),
            baseline.get("mean_abs_position_error"),
        )
        single_mae[checkpoint] = _delta(champion.get("mae_gap_sec"), single.get("mae_gap_sec"))
    return {
        "status": "complete" if n_folds_failed == 0 else "partial",
        "strategy": strategy.value,
        "selection_mode": selection_mode.value,
        "n_events": n_events,
        "n_folds_total": n_folds_total,
        "n_folds_successful": int(champion_predictions["fold_id"].nunique()),
        "n_folds_failed": n_folds_failed,
        "checkpoints": checkpoints,
        "metrics_by_checkpoint": champion_metrics,
        "best_baseline_by_checkpoint": best_baselines,
        "best_single_family_by_checkpoint": best_single,
        "champion_vs_best_baseline_delta_mae": baseline_mae,
        "champion_vs_best_baseline_delta_position_error": baseline_position,
        "champion_vs_best_single_family_delta_mae": single_mae,
        "created_at_utc": _utc_now(),
    }


def _candidate_rows_for_source(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    result = frame.copy()
    if source == "walk_forward":
        result = result[result["prediction_type"].isin(["tabular", "baseline"])].copy()
        result = result[
            result["prediction_type"].eq("baseline")
            | result["model_name"].isin(["ridge", "random_forest"])
        ].copy()
        result["candidate_family"] = result.apply(_walk_forward_family, axis=1)
        result["feature_group"] = pd.NA
    elif source == "ablation":
        result = result[result["prediction_type"].eq("tabular")].copy()
        result["candidate_family"] = "ablation"
    else:
        result = result[result["prediction_type"].eq("boosted")].copy()
        result["candidate_family"] = "boosted"
    return result


def _walk_forward_family(row: pd.Series) -> str:
    if row["prediction_type"] == "baseline":
        return "robust_baseline" if str(row["model_name"]).startswith("robust_") else "baseline"
    return "tabular"


def _method_rows(
    candidates: pd.DataFrame,
    method: ChampionMethodConfig,
    checkpoint: str,
) -> pd.DataFrame:
    feature_group = candidates["feature_group"].fillna("")
    expected_group = method.feature_group or ""
    return candidates[
        candidates["checkpoint"].eq(checkpoint)
        & candidates["candidate_family"].eq(method.family)
        & candidates["model_name"].eq(method.model_name)
        & feature_group.eq(expected_group)
    ].copy()


def _champion_prediction_rows(
    rows: pd.DataFrame,
    selection_mode: ChampionSelectionMode,
    method: ChampionMethodConfig,
) -> pd.DataFrame:
    columns = [
        "strategy",
        "fold_id",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "driver",
        "team",
        "quali_position",
        "quali_gap_to_pole_sec",
        "reached_q3",
        "predicted_quali_gap_to_pole_sec",
        "predicted_quali_position",
        "predicted_reached_q3",
    ]
    result = rows.loc[:, columns].copy()
    result["selection_mode"] = selection_mode.value
    result["selected_family"] = method.family
    result["selected_model_name"] = method.model_name
    result["selected_feature_group"] = method.feature_group
    return result


def _best_candidate_by_checkpoint(
    candidates: pd.DataFrame,
    checkpoints: list[str],
    reference_predictions: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for checkpoint in checkpoints:
        checkpoint_rows = candidates[candidates["checkpoint"].eq(checkpoint)]
        reference_keys = (
            reference_predictions[reference_predictions["checkpoint"].eq(checkpoint)]
            .loc[:, PREDICTION_KEY_COLUMNS]
            .drop_duplicates()
        )
        expected_keys = set(
            map(
                tuple,
                reference_keys.itertuples(index=False, name=None),
            )
        )
        choices: list[tuple[ChampionMethodConfig, dict[str, float | None]]] = []
        for keys, rows in checkpoint_rows.groupby(list(METHOD_COLUMNS), dropna=False, sort=False):
            method_keys = set(
                map(
                    tuple,
                    rows.loc[:, PREDICTION_KEY_COLUMNS]
                    .drop_duplicates()
                    .itertuples(index=False, name=None),
                )
            )
            if not expected_keys.issubset(method_keys):
                continue
            rows = rows.merge(
                reference_keys,
                on=list(PREDICTION_KEY_COLUMNS),
                how="inner",
            )
            family, model_name, feature_group = keys
            choices.append(
                (
                    ChampionMethodConfig(
                        family=str(family),
                        model_name=str(model_name),
                        feature_group=_optional_string(feature_group),
                    ),
                    compute_prediction_metrics(rows),
                )
            )
        choices = [choice for choice in choices if choice[1].get("mae_gap_sec") is not None]
        if choices:
            method, metrics = min(
                choices,
                key=lambda choice: float(choice[1]["mae_gap_sec"]),
            )
            best[checkpoint] = {
                "family": method.family,
                "model_name": method.model_name,
                "feature_group": method.feature_group,
                "mae_gap_sec": metrics.get("mae_gap_sec"),
                "mean_abs_position_error": metrics.get("mean_abs_position_error"),
            }
    return best


def _optional_string(value: object) -> str | None:
    if value is None or pd.isna(value) or str(value) in {"", "<NA>", "nan"}:
        return None
    return str(value)


def _resolve_dataset_path(config: DataConfig, dataset_path: Path | None) -> Path:
    path = dataset_path or build_combined_dataset_path(config.modeling_output_dir)
    if not path.is_absolute():
        path = config.project_root / path
    if not path.is_file():
        raise FileNotFoundError(f"Modeling dataset does not exist: {path}")
    return path


def _output_paths(metrics_dir: Path) -> dict[str, Path]:
    return {
        "metrics": metrics_dir / "champion_metrics.json",
        "predictions": metrics_dir / "champion_predictions.parquet",
        "selection": metrics_dir / "champion_selection.parquet",
    }


def _skipped_payload(
    strategy: BacktestStrategy,
    selection_mode: ChampionSelectionMode,
    n_events: int,
    reason: str,
) -> dict[str, object]:
    return {
        "status": "skipped",
        "strategy": strategy.value,
        "selection_mode": selection_mode.value,
        "reason": reason,
        "n_events": n_events,
        "n_folds_total": 0,
        "n_folds_successful": 0,
        "n_folds_failed": 0,
        "created_at_utc": _utc_now(),
    }


def _delta(first: object, second: object) -> float | None:
    if first is None or second is None:
        return None
    return float(first) - float(second)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
