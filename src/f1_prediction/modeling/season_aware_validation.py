"""Artifact-based validation for season-aware training candidates."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.utils.paths import ensure_directory

CHECKPOINT_ORDER: tuple[str, ...] = ("after_fp1", "after_fp2", "after_fp3")
ROW_KEY_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "season",
    "event_slug",
    "checkpoint",
    "driver",
)
CURRENT_POLICY = "current_season_only_with_prior"
UNIFORM_POLICY = "uniform"
STATIC_FAMILY = "ablation"
STATIC_MODEL = "random_forest"
STATIC_FEATURE_GROUP = "base_plus_relative"
BOOTSTRAP_SEED = 2026022
BOOTSTRAP_ITERATIONS = 2000


@dataclass(frozen=True)
class SeasonAwareValidationSummary:
    """Paths and issue counts produced by season-aware validation."""

    status: str
    summary_path: Path
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    missing_inputs: tuple[str, ...]
    generation_issues: tuple[str, ...]


def create_season_aware_validation_report(config: DataConfig) -> SeasonAwareValidationSummary:
    """Create validation tables and figures from saved backtest artifacts."""
    metrics_dir = config.metrics_output_dir
    figures_dir = metrics_dir.parent / "figures"
    ensure_directory(metrics_dir)
    ensure_directory(figures_dir)
    artifacts, missing_inputs = load_candidate_artifacts(metrics_dir)
    composition = load_training_composition(artifacts["metrics"])
    candidate_rows = standardize_candidate_predictions(artifacts["predictions"])
    fp3_comparison = build_fp3_candidate_comparison(candidate_rows, composition)
    event_level = build_event_level_comparison(fp3_comparison)
    policy_rows = build_policy_aligned_rows(candidate_rows, composition)
    season_level = build_season_level_comparison(policy_rows, fp3_comparison)
    cold_start = build_cold_start_comparison(policy_rows, fp3_comparison)
    summary_payload = build_season_aware_summary_payload(
        fp3_comparison=fp3_comparison,
        event_level=event_level,
        season_level=season_level,
        cold_start=cold_start,
        composition=composition,
        missing_inputs=missing_inputs,
    )

    table_paths = (
        metrics_dir / "season_aware_fp3_candidate_comparison.csv",
        metrics_dir / "season_aware_event_level_comparison.csv",
        metrics_dir / "season_aware_season_level_comparison.csv",
        metrics_dir / "season_aware_cold_start_comparison.csv",
    )
    fp3_comparison.to_csv(table_paths[0], index=False)
    event_level.to_csv(table_paths[1], index=False)
    season_level.to_csv(table_paths[2], index=False)
    cold_start.to_csv(table_paths[3], index=False)

    figure_paths, figure_issues = generate_season_aware_figures(
        figures_dir=figures_dir,
        fp3_comparison=fp3_comparison,
        event_level=event_level,
        season_level=season_level,
        cold_start=cold_start,
        composition=composition,
    )
    summary_payload["generated_figures"] = [_relative_report_path(path) for path in figure_paths]
    summary_payload["generation_issues"] = [
        *summary_payload["generation_issues"],
        *figure_issues,
    ]
    summary_path = metrics_dir / "season_aware_validation_summary.json"
    _write_json(summary_path, summary_payload)
    return SeasonAwareValidationSummary(
        status=str(summary_payload["status"]),
        summary_path=summary_path,
        table_paths=table_paths,
        figure_paths=tuple(figure_paths),
        missing_inputs=tuple(missing_inputs),
        generation_issues=tuple(summary_payload["generation_issues"]),
    )


def load_candidate_artifacts(metrics_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Load available prediction and metrics artifacts used by validation."""
    prediction_specs = (
        ("tabular", UNIFORM_POLICY, "walk_forward_uniform_predictions.parquet"),
        (
            "tabular",
            CURRENT_POLICY,
            "walk_forward_current_season_only_with_prior_predictions.parquet",
        ),
        ("ablation", UNIFORM_POLICY, "ablation_uniform_predictions.parquet"),
        ("ablation", CURRENT_POLICY, "ablation_current_season_only_with_prior_predictions.parquet"),
        ("boosted", UNIFORM_POLICY, "boosted_uniform_predictions.parquet"),
        ("boosted", CURRENT_POLICY, "boosted_current_season_only_with_prior_predictions.parquet"),
    )
    metric_specs = (
        ("tabular", UNIFORM_POLICY, "walk_forward_uniform_metrics.json"),
        ("tabular", CURRENT_POLICY, "walk_forward_current_season_only_with_prior_metrics.json"),
        ("ablation", UNIFORM_POLICY, "ablation_uniform_metrics.json"),
        ("ablation", CURRENT_POLICY, "ablation_current_season_only_with_prior_metrics.json"),
        ("boosted", UNIFORM_POLICY, "boosted_uniform_metrics.json"),
        ("boosted", CURRENT_POLICY, "boosted_current_season_only_with_prior_metrics.json"),
    )
    missing: list[str] = []
    predictions: list[pd.DataFrame] = []
    metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for family, policy, filename in prediction_specs:
        path = metrics_dir / filename
        if not path.is_file():
            missing.append(filename)
            continue
        frame = pd.read_parquet(path)
        frame["candidate_family"] = family
        frame["training_policy"] = policy
        predictions.append(frame)
    for family, policy, filename in metric_specs:
        path = metrics_dir / filename
        if not path.is_file():
            missing.append(filename)
            continue
        metrics[(family, policy)] = _read_json(path)
    return {"predictions": predictions, "metrics": metrics}, missing


def standardize_candidate_predictions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Normalize saved prediction artifacts into one candidate table."""
    rows: list[pd.DataFrame] = []
    for frame in frames:
        if frame.empty:
            continue
        candidate = frame.copy()
        if "prediction_type" in candidate:
            candidate = candidate[candidate["prediction_type"].isin(["tabular", "boosted"])].copy()
        if candidate.empty:
            continue
        candidate["candidate_model_name"] = candidate["model_name"].astype(str)
        if "feature_group" not in candidate:
            candidate["feature_group"] = pd.NA
        if "team" not in candidate:
            candidate["team"] = pd.NA
        candidate["candidate_feature_group"] = candidate["feature_group"].astype("string")
        candidate["candidate_feature_group"] = candidate["candidate_feature_group"].fillna("")
        required = {
            *ROW_KEY_COLUMNS,
            "event",
            "quali_gap_to_pole_sec",
            "predicted_quali_gap_to_pole_sec",
            "quali_position",
            "predicted_quali_position",
            "candidate_family",
            "candidate_model_name",
            "candidate_feature_group",
            "training_policy",
        }
        missing = sorted(required - set(candidate.columns))
        if missing:
            raise ValueError(f"Prediction artifact is missing columns: {', '.join(missing)}")
        rows.append(candidate)
    if not rows:
        return pd.DataFrame(columns=_candidate_columns())
    result = pd.concat(rows, ignore_index=True, sort=False)
    result = result.loc[:, [column for column in _candidate_columns() if column in result]]
    return result.drop_duplicates(
        [
            *ROW_KEY_COLUMNS,
            "candidate_family",
            "candidate_model_name",
            "candidate_feature_group",
            "training_policy",
        ]
    ).reset_index(drop=True)


def load_training_composition(
    metrics: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    """Flatten fold-level training-weight summaries from metrics payloads."""
    rows: list[dict[str, object]] = []
    for (family, policy), payload in metrics.items():
        for item in payload.get("training_weight_summary_by_fold", []):
            row = dict(item)
            row["candidate_family"] = family
            row["training_policy"] = policy
            row["fold_id"] = int(row["fold_id"])
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=_composition_columns())
    frame = pd.DataFrame(rows)
    for column in _composition_columns():
        if column not in frame:
            frame[column] = pd.NA
    return frame.loc[:, _composition_columns()]


def build_fp3_candidate_comparison(
    candidates: pd.DataFrame,
    composition: pd.DataFrame,
) -> pd.DataFrame:
    """Compare FP3 static uniform RF rows against current-season weighted candidates."""
    columns = _fp3_comparison_columns()
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    fp3 = candidates[candidates["checkpoint"].eq("after_fp3")].copy()
    static = fp3[
        fp3["candidate_family"].eq(STATIC_FAMILY)
        & fp3["candidate_model_name"].eq(STATIC_MODEL)
        & fp3["candidate_feature_group"].eq(STATIC_FEATURE_GROUP)
        & fp3["training_policy"].eq(UNIFORM_POLICY)
    ].copy()
    candidate = fp3[fp3["training_policy"].eq(CURRENT_POLICY)].copy()
    if static.empty or candidate.empty:
        return pd.DataFrame(columns=columns)

    static = static.rename(
        columns={
            "predicted_quali_gap_to_pole_sec": "static_predicted_quali_gap_to_pole_sec",
            "predicted_quali_position": "static_predicted_quali_position",
        }
    )
    static_columns = [
        *ROW_KEY_COLUMNS,
        "event",
        "quali_gap_to_pole_sec",
        "quali_position",
        "static_predicted_quali_gap_to_pole_sec",
        "static_predicted_quali_position",
    ]
    merged = candidate.merge(
        static.loc[:, static_columns],
        on=list(ROW_KEY_COLUMNS),
        how="inner",
        suffixes=("", "_static"),
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged = _add_candidate_composition(merged, composition)
    merged["static_abs_error_gap_sec"] = (
        merged["static_predicted_quali_gap_to_pole_sec"] - merged["quali_gap_to_pole_sec"]
    ).abs()
    merged["candidate_abs_error_gap_sec"] = (
        merged["predicted_quali_gap_to_pole_sec"] - merged["quali_gap_to_pole_sec"]
    ).abs()
    merged["error_delta_vs_static_sec"] = (
        merged["candidate_abs_error_gap_sec"] - merged["static_abs_error_gap_sec"]
    )
    merged["candidate_better_than_static"] = merged["error_delta_vs_static_sec"].lt(0)
    merged["candidate_worse_than_static"] = merged["error_delta_vs_static_sec"].gt(0)
    merged["current_season_prior_event_count"] = (
        pd.to_numeric(merged["same_season_training_events"], errors="coerce").fillna(0).astype(int)
    )
    merged["current_season_evidence_regime"] = merged["current_season_prior_event_count"].map(
        current_season_evidence_regime
    )
    merged = merged.rename(
        columns={
            "predicted_quali_gap_to_pole_sec": "candidate_predicted_quali_gap_to_pole_sec",
            "predicted_quali_position": "candidate_predicted_quali_position",
        }
    )
    return merged.loc[:, columns]


def build_policy_aligned_rows(
    candidates: pd.DataFrame,
    composition: pd.DataFrame,
) -> pd.DataFrame:
    """Align current-policy candidates with matching uniform candidate rows."""
    columns = _policy_aligned_columns()
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    method_keys = [
        *ROW_KEY_COLUMNS,
        "candidate_family",
        "candidate_model_name",
        "candidate_feature_group",
    ]
    uniform = candidates[candidates["training_policy"].eq(UNIFORM_POLICY)].copy()
    current = candidates[candidates["training_policy"].eq(CURRENT_POLICY)].copy()
    if uniform.empty or current.empty:
        return pd.DataFrame(columns=columns)
    uniform = uniform.rename(
        columns={
            "predicted_quali_gap_to_pole_sec": "uniform_predicted_quali_gap_to_pole_sec",
            "predicted_quali_position": "uniform_predicted_quali_position",
        }
    )
    merged = current.merge(
        uniform.loc[
            :,
            [
                *method_keys,
                "uniform_predicted_quali_gap_to_pole_sec",
                "uniform_predicted_quali_position",
            ],
        ],
        on=method_keys,
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged = _add_candidate_composition(merged, composition)
    merged["uniform_abs_error_gap_sec"] = (
        merged["uniform_predicted_quali_gap_to_pole_sec"] - merged["quali_gap_to_pole_sec"]
    ).abs()
    merged["candidate_abs_error_gap_sec"] = (
        merged["predicted_quali_gap_to_pole_sec"] - merged["quali_gap_to_pole_sec"]
    ).abs()
    merged["error_delta_vs_uniform_sec"] = (
        merged["candidate_abs_error_gap_sec"] - merged["uniform_abs_error_gap_sec"]
    )
    merged["current_season_prior_event_count"] = (
        pd.to_numeric(merged["same_season_training_events"], errors="coerce").fillna(0).astype(int)
    )
    merged["current_season_evidence_regime"] = merged["current_season_prior_event_count"].map(
        current_season_evidence_regime
    )
    merged = merged.rename(
        columns={
            "predicted_quali_gap_to_pole_sec": "candidate_predicted_quali_gap_to_pole_sec",
            "predicted_quali_position": "candidate_predicted_quali_position",
        }
    )
    return merged.loc[:, columns]


def build_event_level_comparison(fp3_comparison: pd.DataFrame) -> pd.DataFrame:
    """Aggregate FP3 static-vs-candidate comparisons by event/fold."""
    columns = _event_level_columns()
    if fp3_comparison.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    group_columns = [
        "candidate_family",
        "candidate_model_name",
        "candidate_feature_group",
        "training_policy",
        "season",
        "event",
        "event_slug",
        "fold_id",
    ]
    for keys, group in fp3_comparison.groupby(group_columns, dropna=False, sort=False):
        (
            family,
            model,
            feature_group,
            policy,
            season,
            event,
            event_slug,
            fold_id,
        ) = keys
        rows.append(
            {
                "candidate_family": family,
                "candidate_model_name": model,
                "candidate_feature_group": feature_group,
                "training_policy": policy,
                "season": season,
                "event": event,
                "event_slug": event_slug,
                "fold_id": fold_id,
                "current_season_prior_event_count": int(
                    group["current_season_prior_event_count"].max()
                ),
                "current_season_evidence_regime": group["current_season_evidence_regime"].iloc[0],
                "rows": int(len(group)),
                "static_mae_gap_sec": float(group["static_abs_error_gap_sec"].mean()),
                "candidate_mae_gap_sec": float(group["candidate_abs_error_gap_sec"].mean()),
                "delta_vs_static_sec": float(group["error_delta_vs_static_sec"].mean()),
                "candidate_median_abs_error_gap_sec": float(
                    group["candidate_abs_error_gap_sec"].median()
                ),
                "candidate_mean_position_error": _mean_position_error(
                    group["candidate_predicted_quali_position"],
                    group["quali_position"],
                ),
                "effective_sample_size": _mean_numeric(group["effective_sample_size"]),
                "same_season_weight_share": _mean_numeric(group["same_season_weight_share"]),
                "prior_season_weight_share": _mean_numeric(group["prior_season_weight_share"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_season_level_comparison(
    policy_rows: pd.DataFrame,
    fp3_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize current-policy candidates by season and checkpoint."""
    columns = _season_level_columns()
    if policy_rows.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    group_columns = [
        "candidate_family",
        "candidate_model_name",
        "candidate_feature_group",
        "training_policy",
        "checkpoint",
        "season",
    ]
    static_lookup = _static_delta_lookup(fp3_comparison, ["candidate_key", "season"])
    frame = policy_rows.copy()
    frame["candidate_key"] = _candidate_key_series(frame)
    for keys, group in frame.groupby(group_columns, dropna=False, sort=False):
        family, model, feature_group, policy, checkpoint, season = keys
        metrics = _metrics_from_abs_errors(group["candidate_abs_error_gap_sec"])
        rows.append(
            {
                "candidate_family": family,
                "candidate_model_name": model,
                "candidate_feature_group": feature_group,
                "training_policy": policy,
                "checkpoint": checkpoint,
                "season": season,
                **metrics,
                "delta_vs_uniform_sec": float(group["error_delta_vs_uniform_sec"].mean()),
                "delta_vs_static_sec": (
                    static_lookup.get((group["candidate_key"].iloc[0], season))
                    if checkpoint == "after_fp3"
                    else None
                ),
                "events": int(group["fold_id"].nunique()),
                "prediction_rows": int(len(group)),
                "effective_sample_size": _mean_numeric(group["effective_sample_size"]),
                "same_season_weight_share": _mean_numeric(group["same_season_weight_share"]),
                "prior_season_weight_share": _mean_numeric(group["prior_season_weight_share"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_cold_start_comparison(
    policy_rows: pd.DataFrame,
    fp3_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize current-policy candidates by cold-start regime and checkpoint."""
    columns = _cold_start_columns()
    if policy_rows.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    group_columns = [
        "candidate_family",
        "candidate_model_name",
        "candidate_feature_group",
        "training_policy",
        "checkpoint",
        "current_season_evidence_regime",
    ]
    static_lookup = _static_delta_lookup(
        fp3_comparison,
        ["candidate_key", "current_season_evidence_regime"],
    )
    frame = policy_rows.copy()
    frame["candidate_key"] = _candidate_key_series(frame)
    for keys, group in frame.groupby(group_columns, dropna=False, sort=False):
        family, model, feature_group, policy, checkpoint, regime = keys
        metrics = _metrics_from_abs_errors(group["candidate_abs_error_gap_sec"])
        rows.append(
            {
                "candidate_family": family,
                "candidate_model_name": model,
                "candidate_feature_group": feature_group,
                "training_policy": policy,
                "checkpoint": checkpoint,
                "current_season_evidence_regime": regime,
                **metrics,
                "delta_vs_uniform_sec": float(group["error_delta_vs_uniform_sec"].mean()),
                "delta_vs_static_sec": (
                    static_lookup.get((group["candidate_key"].iloc[0], regime))
                    if checkpoint == "after_fp3"
                    else None
                ),
                "events": int(group["fold_id"].nunique()),
                "prediction_rows": int(len(group)),
                "effective_sample_size": _mean_numeric(group["effective_sample_size"]),
                "same_season_weight_share": _mean_numeric(group["same_season_weight_share"]),
                "prior_season_weight_share": _mean_numeric(group["prior_season_weight_share"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def current_season_evidence_regime(prior_event_count: int | float | object) -> str:
    """Return the documented current-season evidence regime."""
    count = int(prior_event_count) if pd.notna(prior_event_count) else 0
    if count < 5:
        return "cold_start"
    if count <= 8:
        return "early_season"
    return "established_season"


def paired_bootstrap_mean_ci(
    values: pd.Series | list[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, float | int | None]:
    """Return a deterministic paired-bootstrap CI for event-level mean deltas."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype(float)
    if numeric.empty:
        return {"mean_delta": None, "ci_low": None, "ci_high": None, "events": 0, "seed": seed}
    rng = np.random.default_rng(seed)
    data = numeric.to_numpy()
    samples = rng.choice(data, size=(iterations, len(data)), replace=True).mean(axis=1)
    return {
        "mean_delta": float(data.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "events": int(len(data)),
        "seed": seed,
    }


def build_season_aware_summary_payload(
    *,
    fp3_comparison: pd.DataFrame,
    event_level: pd.DataFrame,
    season_level: pd.DataFrame,
    cold_start: pd.DataFrame,
    composition: pd.DataFrame,
    missing_inputs: list[str],
) -> dict[str, object]:
    """Build the JSON summary for season-aware validation."""
    fixed_event_rows = _fixed_candidate_rows(event_level)
    bootstrap = paired_bootstrap_mean_ci(
        fixed_event_rows["delta_vs_static_sec"] if not fixed_event_rows.empty else []
    )
    fixed_summary = _fixed_candidate_summary(fixed_event_rows)
    best_candidate = _best_fixed_candidate(event_level)
    recommendation = _promotion_recommendation(fixed_summary, bootstrap)
    return {
        "status": "complete" if not fp3_comparison.empty else "partial",
        "inputs_available": _inputs_available(missing_inputs),
        "missing_inputs": missing_inputs,
        "season_aware_validation_available": not fp3_comparison.empty,
        "fixed_fp3_candidate_summary": fixed_summary,
        "season_aware_fp3_candidate_summary": fixed_summary,
        "season_aware_best_fixed_candidate": best_candidate,
        "season_aware_promotion_recommendation": recommendation,
        "retrospective_best_candidate_summary": _retrospective_best_candidate(event_level),
        "future_eligible_candidate_configurations": _future_eligible_candidates(event_level),
        "bootstrap_robustness": bootstrap,
        "test_season_summary": _compact_group_summary(season_level, "season"),
        "cold_start_summary": _compact_group_summary(
            cold_start,
            "current_season_evidence_regime",
        ),
        "training_composition_summary": _composition_summary(composition),
        "main_findings": _main_findings(fixed_summary, bootstrap, recommendation),
        "generated_at": _utc_now(),
        "generation_issues": [],
        "table_row_counts": {
            "season_aware_fp3_candidate_comparison": int(len(fp3_comparison)),
            "season_aware_event_level_comparison": int(len(event_level)),
            "season_aware_season_level_comparison": int(len(season_level)),
            "season_aware_cold_start_comparison": int(len(cold_start)),
        },
    }


def generate_season_aware_figures(
    *,
    figures_dir: Path,
    fp3_comparison: pd.DataFrame,
    event_level: pd.DataFrame,
    season_level: pd.DataFrame,
    cold_start: pd.DataFrame,
    composition: pd.DataFrame,
) -> tuple[list[Path], list[str]]:
    """Generate static matplotlib figures for season-aware validation."""
    try:
        ensure_directory(figures_dir / ".matplotlib")
        ensure_directory(figures_dir / ".cache")
        os.environ.setdefault("MPLCONFIGDIR", str(figures_dir / ".matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(figures_dir / ".cache"))
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return [], [f"matplotlib_unavailable: {exc}"]
    specs = (
        (
            "season_aware_fp3_candidate_vs_static_mae.png",
            lambda path: _plot_fp3_static_vs_candidate(plt, event_level, path),
        ),
        (
            "season_aware_fp3_delta_by_event.png",
            lambda path: _plot_fp3_delta_by_event(plt, event_level, path),
        ),
        (
            "season_aware_policy_by_test_season.png",
            lambda path: _plot_policy_by_season(plt, season_level, path),
        ),
        (
            "season_aware_cold_start_vs_established.png",
            lambda path: _plot_cold_start(plt, cold_start, path),
        ),
        (
            "season_aware_training_weight_composition.png",
            lambda path: _plot_training_composition(plt, composition, path),
        ),
    )
    paths: list[Path] = []
    issues: list[str] = []
    for filename, writer in specs:
        path = figures_dir / filename
        try:
            if writer(path):
                paths.append(path)
        except Exception as exc:
            issues.append(f"{filename}: {exc}")
    return paths, issues


def _add_candidate_composition(frame: pd.DataFrame, composition: pd.DataFrame) -> pd.DataFrame:
    if composition.empty:
        result = frame.copy()
        for column in _composition_value_columns():
            result[column] = pd.NA
        return result
    result = frame.merge(
        composition,
        on=["candidate_family", "training_policy", "fold_id"],
        how="left",
    )
    for column in _composition_value_columns():
        if column not in result:
            result[column] = pd.NA
    return result


def _metrics_from_abs_errors(errors: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(errors, errors="coerce").dropna().astype(float)
    return {
        "mae_gap_sec": float(numeric.mean()),
        "rmse_gap_sec": float(np.sqrt((numeric**2).mean())),
        "median_abs_error_gap_sec": float(numeric.median()),
    }


def _mean_position_error(predicted: pd.Series, actual: pd.Series) -> float | None:
    frame = pd.DataFrame({"predicted": predicted, "actual": actual}).dropna()
    if frame.empty:
        return None
    return float((frame["predicted"].astype(float) - frame["actual"].astype(float)).abs().mean())


def _mean_numeric(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _candidate_key_series(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["candidate_family"].astype(str)
        + "/"
        + frame["candidate_model_name"].astype(str)
        + "/"
        + frame["candidate_feature_group"].fillna("").astype(str)
    )


def _static_delta_lookup(
    fp3_comparison: pd.DataFrame,
    keys: list[str],
) -> dict[tuple[object, ...], float]:
    if fp3_comparison.empty:
        return {}
    frame = fp3_comparison.copy()
    frame["candidate_key"] = _candidate_key_series(frame)
    return {
        tuple(row[key] for key in keys): float(row["error_delta_vs_static_sec"])
        for row in frame.groupby(keys, dropna=False, sort=False)["error_delta_vs_static_sec"]
        .mean()
        .reset_index()
        .to_dict("records")
    }


def _fixed_candidate_rows(event_level: pd.DataFrame) -> pd.DataFrame:
    if event_level.empty:
        return event_level
    return event_level[
        event_level["candidate_family"].eq(STATIC_FAMILY)
        & event_level["candidate_model_name"].eq(STATIC_MODEL)
        & event_level["candidate_feature_group"].eq(STATIC_FEATURE_GROUP)
        & event_level["training_policy"].eq(CURRENT_POLICY)
    ].copy()


def _fixed_candidate_summary(event_level: pd.DataFrame) -> dict[str, object]:
    if event_level.empty:
        return {}
    return {
        "candidate_family": STATIC_FAMILY,
        "candidate_model_name": STATIC_MODEL,
        "candidate_feature_group": STATIC_FEATURE_GROUP,
        "training_policy": CURRENT_POLICY,
        "events": int(len(event_level)),
        "rows": int(event_level["rows"].sum()),
        "static_mae_gap_sec": float(
            np.average(event_level["static_mae_gap_sec"], weights=event_level["rows"])
        ),
        "candidate_mae_gap_sec": float(
            np.average(event_level["candidate_mae_gap_sec"], weights=event_level["rows"])
        ),
        "mean_event_delta_vs_static_sec": float(event_level["delta_vs_static_sec"].mean()),
        "median_event_delta_vs_static_sec": float(event_level["delta_vs_static_sec"].median()),
        "share_events_improved": float(event_level["delta_vs_static_sec"].lt(0).mean()),
    }


def _best_fixed_candidate(event_level: pd.DataFrame) -> dict[str, object]:
    if event_level.empty:
        return {}
    group_columns = [
        "candidate_family",
        "candidate_model_name",
        "candidate_feature_group",
        "training_policy",
    ]
    grouped = (
        event_level.groupby(group_columns, dropna=False, sort=False)
        .agg(
            candidate_mae_gap_sec=("candidate_mae_gap_sec", "mean"),
            delta_vs_static_sec=("delta_vs_static_sec", "mean"),
            events=("fold_id", "nunique"),
        )
        .reset_index()
        .sort_values(["candidate_mae_gap_sec", "delta_vs_static_sec"], kind="stable")
    )
    if grouped.empty:
        return {}
    return dict(grouped.iloc[0])


def _retrospective_best_candidate(event_level: pd.DataFrame) -> dict[str, object]:
    best = _best_fixed_candidate(event_level)
    if not best:
        return {}
    best["note"] = "Retrospective summary only; not a live champion-policy selection rule."
    return best


def _future_eligible_candidates(event_level: pd.DataFrame) -> list[dict[str, object]]:
    if event_level.empty:
        return []
    grouped = (
        event_level.groupby(
            [
                "candidate_family",
                "candidate_model_name",
                "candidate_feature_group",
                "training_policy",
            ],
            dropna=False,
            sort=False,
        )["delta_vs_static_sec"]
        .mean()
        .reset_index()
    )
    improved = grouped[grouped["delta_vs_static_sec"].lt(0)].copy()
    return improved.to_dict("records")


def _promotion_recommendation(
    fixed_summary: dict[str, object],
    bootstrap: dict[str, float | int | None],
) -> str:
    if not fixed_summary or bootstrap.get("mean_delta") is None:
        return "insufficient_evidence"
    mean_delta = float(bootstrap["mean_delta"])
    ci_high = bootstrap.get("ci_high")
    if mean_delta > 0:
        return "retain_static_policy"
    if ci_high is not None and float(ci_high) < 0 and int(bootstrap.get("events", 0)) >= 8:
        return "eligible_for_future_nested_candidate_evaluation"
    return "insufficient_evidence"


def _main_findings(
    fixed_summary: dict[str, object],
    bootstrap: dict[str, float | int | None],
    recommendation: str,
) -> list[str]:
    if not fixed_summary:
        return ["Season-aware validation artifacts were incomplete; results are partial."]
    delta = fixed_summary.get("mean_event_delta_vs_static_sec")
    findings = ["Season-aware validation is retrospective; champion defaults remain unchanged."]
    if delta is not None:
        if float(delta) < 0:
            findings.append(
                "The fixed FP3 weighted RF candidate improves mean event-level MAE versus "
                f"static by {abs(float(delta)):.3f} sec."
            )
        else:
            findings.append(
                "The fixed FP3 weighted RF candidate does not improve mean event-level MAE "
                "versus static."
            )
    if bootstrap.get("ci_low") is not None and bootstrap.get("ci_high") is not None:
        findings.append(
            "Bootstrap CI for mean event-level delta: "
            f"[{float(bootstrap['ci_low']):.3f}, {float(bootstrap['ci_high']):.3f}] sec."
        )
    findings.append(f"Promotion recommendation: {recommendation}.")
    return findings


def _compact_group_summary(frame: pd.DataFrame, group_column: str) -> dict[str, dict[str, object]]:
    if frame.empty or group_column not in frame:
        return {}
    return {
        str(key): {
            "rows": int(group["prediction_rows"].sum()),
            "mean_mae_gap_sec": _mean_numeric(group["mae_gap_sec"]),
            "mean_delta_vs_uniform_sec": _mean_numeric(group["delta_vs_uniform_sec"]),
            "mean_delta_vs_static_sec": _mean_numeric(group["delta_vs_static_sec"]),
        }
        for key, group in frame.groupby(group_column, dropna=False, sort=False)
    }


def _composition_summary(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    if frame.empty:
        return {}
    result: dict[str, dict[str, float | None]] = {}
    for keys, group in frame.groupby(["candidate_family", "training_policy"], sort=False):
        family, policy = keys
        result[f"{family}/{policy}"] = {
            "mean_effective_sample_size": _mean_numeric(group["effective_sample_size"]),
            "mean_same_season_weight_share": _mean_numeric(group["same_season_weight_share"]),
            "mean_prior_season_weight_share": _mean_numeric(group["prior_season_weight_share"]),
        }
    return result


def _inputs_available(missing: list[str]) -> dict[str, bool]:
    expected = (
        "ablation_uniform_predictions.parquet",
        "ablation_current_season_only_with_prior_predictions.parquet",
    )
    return {name: name not in missing for name in expected}


def _plot_fp3_static_vs_candidate(plt: Any, event_level: pd.DataFrame, path: Path) -> bool:
    frame = _fixed_candidate_rows(event_level)
    if frame.empty:
        return False
    data = frame.sort_values(["season", "fold_id"]).copy()
    data["label"] = data["season"].astype(str) + " " + data["event"].astype(str)
    ax = data.plot(
        x="label",
        y=["static_mae_gap_sec", "candidate_mae_gap_sec"],
        kind="bar",
        figsize=(10, 4.5),
        width=0.8,
    )
    ax.set_title("FP3 static vs weighted RF candidate MAE")
    ax.set_xlabel("Test event")
    ax.set_ylabel("MAE gap (sec)")
    ax.legend(["Static RF", "Weighted RF"])
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _plot_fp3_delta_by_event(plt: Any, event_level: pd.DataFrame, path: Path) -> bool:
    frame = _fixed_candidate_rows(event_level)
    if frame.empty:
        return False
    data = frame.sort_values(["season", "fold_id"]).copy()
    data["label"] = data["season"].astype(str) + " " + data["event"].astype(str)
    ax = data.plot.bar(
        x="label",
        y="delta_vs_static_sec",
        figsize=(10, 4.5),
        width=0.8,
        legend=False,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("FP3 weighted RF delta vs static by event")
    ax.set_xlabel("Test event")
    ax.set_ylabel("Candidate - static MAE (sec)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _plot_policy_by_season(plt: Any, season_level: pd.DataFrame, path: Path) -> bool:
    if season_level.empty:
        return False
    frame = season_level[season_level["checkpoint"].eq("after_fp3")].copy()
    if frame.empty:
        return False
    pivot = frame.pivot_table(
        index="season",
        columns="candidate_family",
        values="delta_vs_uniform_sec",
        aggfunc="mean",
    )
    if pivot.empty:
        return False
    ax = pivot.plot(kind="bar", figsize=(8, 4), width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("FP3 temporal policy delta vs uniform by season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Delta MAE (sec)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _plot_cold_start(plt: Any, cold_start: pd.DataFrame, path: Path) -> bool:
    if cold_start.empty:
        return False
    frame = cold_start[cold_start["checkpoint"].eq("after_fp3")].copy()
    if frame.empty:
        return False
    pivot = frame.pivot_table(
        index="current_season_evidence_regime",
        columns="candidate_family",
        values="delta_vs_uniform_sec",
        aggfunc="mean",
    )
    pivot = pivot.reindex(["cold_start", "early_season", "established_season"]).dropna(how="all")
    if pivot.empty:
        return False
    ax = pivot.plot(kind="bar", figsize=(8, 4), width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("FP3 temporal policy delta by current-season evidence")
    ax.set_xlabel("Regime")
    ax.set_ylabel("Delta MAE (sec)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _plot_training_composition(plt: Any, composition: pd.DataFrame, path: Path) -> bool:
    if composition.empty:
        return False
    frame = composition[composition["training_policy"].eq(CURRENT_POLICY)].copy()
    if frame.empty:
        return False
    grouped = frame.groupby("candidate_family", sort=False)[
        ["same_season_weight_share", "prior_season_weight_share"]
    ].mean()
    if grouped.empty:
        return False
    ax = grouped.plot(kind="bar", stacked=True, figsize=(8, 4), width=0.8)
    ax.set_title("Season-aware training weight composition")
    ax.set_xlabel("Candidate family")
    ax.set_ylabel("Mean weight share")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _candidate_columns() -> list[str]:
    return [
        "fold_id",
        "season",
        "event",
        "event_slug",
        "checkpoint",
        "driver",
        "team",
        "candidate_family",
        "candidate_model_name",
        "candidate_feature_group",
        "training_policy",
        "quali_gap_to_pole_sec",
        "predicted_quali_gap_to_pole_sec",
        "quali_position",
        "predicted_quali_position",
    ]


def _composition_columns() -> list[str]:
    return [
        "candidate_family",
        "training_policy",
        "fold_id",
        "test_event",
        "test_season",
        "training_rows",
        "training_events",
        "same_season_training_events",
        "prior_season_training_events",
        "older_season_training_events",
        "effective_sample_size",
        "same_season_weight_share",
        "prior_season_weight_share",
        "older_season_weight_share",
    ]


def _composition_value_columns() -> list[str]:
    return [
        column
        for column in _composition_columns()
        if column
        not in {
            "candidate_family",
            "training_policy",
            "fold_id",
        }
    ]


def _fp3_comparison_columns() -> list[str]:
    return [
        "candidate_family",
        "candidate_model_name",
        "candidate_feature_group",
        "training_policy",
        "season",
        "event",
        "event_slug",
        "fold_id",
        "checkpoint",
        "driver",
        "team",
        "current_season_prior_event_count",
        "current_season_evidence_regime",
        "quali_gap_to_pole_sec",
        "static_predicted_quali_gap_to_pole_sec",
        "candidate_predicted_quali_gap_to_pole_sec",
        "quali_position",
        "static_predicted_quali_position",
        "candidate_predicted_quali_position",
        "static_abs_error_gap_sec",
        "candidate_abs_error_gap_sec",
        "error_delta_vs_static_sec",
        "candidate_better_than_static",
        "candidate_worse_than_static",
        "effective_sample_size",
        "same_season_weight_share",
        "prior_season_weight_share",
    ]


def _policy_aligned_columns() -> list[str]:
    return [
        "candidate_family",
        "candidate_model_name",
        "candidate_feature_group",
        "training_policy",
        "season",
        "event",
        "event_slug",
        "fold_id",
        "checkpoint",
        "driver",
        "current_season_prior_event_count",
        "current_season_evidence_regime",
        "quali_gap_to_pole_sec",
        "uniform_predicted_quali_gap_to_pole_sec",
        "candidate_predicted_quali_gap_to_pole_sec",
        "uniform_abs_error_gap_sec",
        "candidate_abs_error_gap_sec",
        "error_delta_vs_uniform_sec",
        "quali_position",
        "uniform_predicted_quali_position",
        "candidate_predicted_quali_position",
        "effective_sample_size",
        "same_season_weight_share",
        "prior_season_weight_share",
    ]


def _event_level_columns() -> list[str]:
    return [
        "candidate_family",
        "candidate_model_name",
        "candidate_feature_group",
        "training_policy",
        "season",
        "event",
        "event_slug",
        "fold_id",
        "current_season_prior_event_count",
        "current_season_evidence_regime",
        "rows",
        "static_mae_gap_sec",
        "candidate_mae_gap_sec",
        "delta_vs_static_sec",
        "candidate_median_abs_error_gap_sec",
        "candidate_mean_position_error",
        "effective_sample_size",
        "same_season_weight_share",
        "prior_season_weight_share",
    ]


def _season_level_columns() -> list[str]:
    return [
        "candidate_family",
        "candidate_model_name",
        "candidate_feature_group",
        "training_policy",
        "checkpoint",
        "season",
        "mae_gap_sec",
        "rmse_gap_sec",
        "median_abs_error_gap_sec",
        "delta_vs_uniform_sec",
        "delta_vs_static_sec",
        "events",
        "prediction_rows",
        "effective_sample_size",
        "same_season_weight_share",
        "prior_season_weight_share",
    ]


def _cold_start_columns() -> list[str]:
    return [
        "candidate_family",
        "candidate_model_name",
        "candidate_feature_group",
        "training_policy",
        "checkpoint",
        "current_season_evidence_regime",
        "mae_gap_sec",
        "rmse_gap_sec",
        "median_abs_error_gap_sec",
        "delta_vs_uniform_sec",
        "delta_vs_static_sec",
        "events",
        "prediction_rows",
        "effective_sample_size",
        "same_season_weight_share",
        "prior_season_weight_share",
    ]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(_json_ready(payload), output_file, indent=2, allow_nan=False)
        output_file.write("\n")


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _relative_report_path(path: Path) -> str:
    parts = path.parts
    if "reports" in parts:
        index = parts.index("reports")
        return str(Path(*parts[index:]))
    return path.as_posix()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
