"""Artifact-based diagnostics for temporal weighting backtests."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.modeling.temporal_weighting import TemporalWeightingPolicy
from f1_prediction.utils.paths import ensure_directory

POLICIES: tuple[str, ...] = tuple(policy.value for policy in TemporalWeightingPolicy)
SOURCES: tuple[str, ...] = ("tabular", "ablation", "boosted")
CHECKPOINT_ORDER: tuple[str, ...] = ("after_fp1", "after_fp2", "after_fp3")


@dataclass(frozen=True)
class TemporalWeightingReportSummary:
    """Paths and counts produced by temporal weighting reporting."""

    status: str
    summary_path: Path
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    missing_artifacts: tuple[str, ...]
    generation_issues: tuple[str, ...]


def create_temporal_weighting_report(config: DataConfig) -> TemporalWeightingReportSummary:
    """Read saved temporal-weighted artifacts and create comparison outputs."""
    metrics_dir = config.metrics_output_dir
    figures_dir = metrics_dir.parent / "figures"
    ensure_directory(metrics_dir)
    ensure_directory(figures_dir)
    artifacts, missing = _load_temporal_artifacts(metrics_dir)
    checkpoint = build_checkpoint_comparison_table(artifacts)
    fold = build_fold_comparison_table(artifacts)
    composition = build_training_composition_table(artifacts)
    summary_payload = build_temporal_summary_payload(
        artifacts,
        checkpoint,
        fold,
        composition,
        missing,
    )

    table_paths = (
        metrics_dir / "temporal_weighting_checkpoint_comparison.csv",
        metrics_dir / "temporal_weighting_fold_comparison.csv",
        metrics_dir / "temporal_weighting_training_composition.csv",
    )
    checkpoint.to_csv(table_paths[0], index=False)
    fold.to_csv(table_paths[1], index=False)
    composition.to_csv(table_paths[2], index=False)

    figure_paths, figure_issues = generate_temporal_weighting_figures(
        figures_dir,
        checkpoint,
        composition,
    )
    summary_payload["generated_figures"] = [_relative_report_path(path) for path in figure_paths]
    summary_payload["generation_issues"] = [*summary_payload["generation_issues"], *figure_issues]
    summary_path = metrics_dir / "temporal_weighting_summary.json"
    _write_json(summary_path, summary_payload)
    return TemporalWeightingReportSummary(
        status=str(summary_payload["status"]),
        summary_path=summary_path,
        table_paths=table_paths,
        figure_paths=tuple(figure_paths),
        missing_artifacts=tuple(missing),
        generation_issues=tuple(summary_payload["generation_issues"]),
    )


def build_checkpoint_comparison_table(
    artifacts: dict[str, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    """Build policy/checkpoint metric rows from available artifacts."""
    rows: list[dict[str, object]] = []
    for source, policies in artifacts.items():
        for policy, payload in policies.items():
            for row in _checkpoint_rows(source, policy, payload):
                rows.append(row)
    frame = pd.DataFrame(rows, columns=_checkpoint_columns())
    if frame.empty:
        return frame
    frame = _add_delta_vs_uniform(frame)
    return frame.sort_values(["source", "checkpoint", "temporal_weighting_policy"])


def build_fold_comparison_table(
    artifacts: dict[str, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    """Build fold-level comparison rows for available tabular and boosted artifacts."""
    rows: list[dict[str, object]] = []
    for source in ("tabular", "boosted"):
        for policy, payload in artifacts.get(source, {}).items():
            rows.extend(_fold_rows(source, policy, payload))
    frame = pd.DataFrame(rows, columns=_fold_columns())
    if frame.empty:
        return frame
    frame = _add_delta_vs_uniform(frame, keys=["source", "fold_id", "checkpoint"])
    return frame.sort_values(["source", "fold_id", "checkpoint", "temporal_weighting_policy"])


def build_training_composition_table(
    artifacts: dict[str, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    """Flatten per-fold training-weight summaries from metrics JSON."""
    rows: list[dict[str, object]] = []
    for source, policies in artifacts.items():
        for policy, payload in policies.items():
            for item in payload.get("training_weight_summary_by_fold", []):
                row = dict(item)
                row["source"] = source
                row["temporal_weighting_policy"] = policy
                rows.append(row)
    return pd.DataFrame(rows, columns=_composition_columns())


def build_temporal_summary_payload(
    artifacts: dict[str, dict[str, dict[str, Any]]],
    checkpoint: pd.DataFrame,
    fold: pd.DataFrame,
    composition: pd.DataFrame,
    missing: list[str],
) -> dict[str, object]:
    """Build a concise JSON summary of temporal weighting diagnostics."""
    policies_available = {
        source: sorted(policies) for source, policies in artifacts.items() if policies
    }
    best_by_checkpoint = _best_policy_by_checkpoint(checkpoint)
    deltas = _delta_by_checkpoint(checkpoint)
    return {
        "status": "complete" if policies_available else "partial",
        "temporal_weighting_policies_available": policies_available,
        "best_temporal_weighting_policy_by_checkpoint": best_by_checkpoint,
        "temporal_weighting_vs_uniform_delta_by_checkpoint": deltas,
        "main_findings": _main_findings(checkpoint, best_by_checkpoint),
        "training_composition_summary": _composition_summary(composition),
        "missing_artifacts": missing,
        "generation_issues": [],
        "generated_at": _utc_now(),
        "table_row_counts": {
            "temporal_weighting_checkpoint_comparison": int(len(checkpoint)),
            "temporal_weighting_fold_comparison": int(len(fold)),
            "temporal_weighting_training_composition": int(len(composition)),
        },
    }


def generate_temporal_weighting_figures(
    figures_dir: Path,
    checkpoint: pd.DataFrame,
    composition: pd.DataFrame,
) -> tuple[list[Path], list[str]]:
    """Generate simple static temporal weighting figures."""
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

    figure_specs = (
        (
            "temporal_weighting_mae_by_checkpoint.png",
            lambda path: _plot_checkpoint_bars(
                plt,
                checkpoint,
                path,
                "mae_gap_sec",
                "Temporal weighting MAE by checkpoint",
                "MAE gap (sec)",
            ),
        ),
        (
            "temporal_weighting_delta_vs_uniform.png",
            lambda path: _plot_checkpoint_bars(
                plt,
                checkpoint,
                path,
                "delta_vs_uniform_mae_gap_sec",
                "Temporal weighting delta vs uniform",
                "Delta MAE (sec)",
            ),
        ),
        (
            "temporal_weighting_training_composition.png",
            lambda path: _plot_composition(
                plt,
                composition,
                path,
                "same_season_weight_share",
                "Same-season training weight share",
            ),
        ),
        (
            "temporal_weighting_effective_sample_size.png",
            lambda path: _plot_composition(
                plt,
                composition,
                path,
                "effective_sample_size",
                "Effective sample size by policy",
            ),
        ),
    )
    paths: list[Path] = []
    issues: list[str] = []
    for filename, writer in figure_specs:
        path = figures_dir / filename
        try:
            if writer(path):
                paths.append(path)
        except Exception as exc:
            issues.append(f"{filename}: {exc}")
    return paths, issues


def _load_temporal_artifacts(
    metrics_dir: Path,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str]]:
    specs = {
        "tabular": "walk_forward_{policy}_metrics.json",
        "ablation": "ablation_{policy}_metrics.json",
        "boosted": "boosted_{policy}_metrics.json",
    }
    artifacts: dict[str, dict[str, dict[str, Any]]] = {source: {} for source in SOURCES}
    missing: list[str] = []
    for source, template in specs.items():
        for policy in POLICIES:
            path = metrics_dir / template.format(policy=policy)
            if not path.is_file():
                missing.append(path.name)
                continue
            artifacts[source][policy] = _read_json(path)
    return artifacts, missing


def _checkpoint_rows(source: str, policy: str, payload: dict[str, Any]) -> list[dict[str, object]]:
    if payload.get("status") not in {"complete", "partial"}:
        return []
    rows: list[dict[str, object]] = []
    best = payload.get("best_model_by_checkpoint") or payload.get("best_overall_by_checkpoint", {})
    for checkpoint, selected in best.items():
        metrics = _selected_checkpoint_metrics(source, payload, str(checkpoint), selected)
        if not metrics:
            continue
        composition = _mean_composition(payload.get("training_weight_summary_by_fold", []))
        rows.append(
            {
                "source": source,
                "temporal_weighting_policy": policy,
                "checkpoint": checkpoint,
                "model_name": selected.get("model_name"),
                "feature_group": selected.get("feature_group"),
                "mae_gap_sec": metrics.get("mae_gap_sec"),
                "rmse_gap_sec": metrics.get("rmse_gap_sec"),
                "median_abs_error_gap_sec": metrics.get("median_abs_error_gap_sec"),
                "mean_abs_position_error": metrics.get("mean_abs_position_error"),
                "delta_vs_uniform_mae_gap_sec": None,
                "n_folds_successful": payload.get("n_folds_successful"),
                **composition,
            }
        )
    return rows


def _selected_checkpoint_metrics(
    source: str,
    payload: dict[str, Any],
    checkpoint: str,
    selected: dict[str, Any],
) -> dict[str, Any]:
    if source == "ablation":
        group = selected.get("feature_group")
        model = selected.get("model_name")
        return (
            payload.get("metrics_by_feature_group_model_checkpoint", {})
            .get(group, {})
            .get(model, {})
            .get(checkpoint, {})
        )
    model = selected.get("model_name")
    values = payload.get("metrics_by_model_checkpoint", {}).get(model, {}).get(checkpoint, {})
    return values.get("global", values)


def _fold_rows(source: str, policy: str, payload: dict[str, Any]) -> list[dict[str, object]]:
    best = payload.get("best_model_by_checkpoint", {})
    fold_metrics = payload.get("metrics_by_fold_model_checkpoint", {})
    rows: list[dict[str, object]] = []
    for fold_id, model_values in fold_metrics.items():
        for checkpoint, selected in best.items():
            model = selected.get("model_name")
            values = model_values.get(model, {}).get(checkpoint, {})
            if values.get("mae_gap_sec") is None:
                continue
            rows.append(
                {
                    "source": source,
                    "temporal_weighting_policy": policy,
                    "fold_id": int(fold_id),
                    "checkpoint": checkpoint,
                    "model_name": model,
                    "mae_gap_sec": values.get("mae_gap_sec"),
                    "delta_vs_uniform_mae_gap_sec": None,
                }
            )
    return rows


def _add_delta_vs_uniform(
    frame: pd.DataFrame,
    *,
    keys: list[str] | None = None,
) -> pd.DataFrame:
    keys = keys or ["source", "checkpoint"]
    result = frame.copy()
    uniform = result[result["temporal_weighting_policy"].eq("uniform")]
    lookup = {
        tuple(row[key] for key in keys): row["mae_gap_sec"] for row in uniform.to_dict("records")
    }
    result["delta_vs_uniform_mae_gap_sec"] = [
        _delta(row["mae_gap_sec"], lookup.get(tuple(row[key] for key in keys)))
        for row in result.to_dict("records")
    ]
    return result


def _best_policy_by_checkpoint(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    if frame.empty:
        return {}
    primary = frame[frame["source"].eq("tabular")].copy()
    if primary.empty:
        primary = frame.copy()
    best: dict[str, dict[str, object]] = {}
    for checkpoint, group in primary.groupby("checkpoint", sort=False):
        candidates = group.dropna(subset=["mae_gap_sec"])
        if candidates.empty:
            continue
        row = candidates.sort_values("mae_gap_sec", kind="stable").iloc[0]
        best[str(checkpoint)] = {
            "source": row["source"],
            "temporal_weighting_policy": row["temporal_weighting_policy"],
            "mae_gap_sec": _number_or_none(row["mae_gap_sec"]),
            "delta_vs_uniform_mae_gap_sec": _number_or_none(row["delta_vs_uniform_mae_gap_sec"]),
        }
    return best


def _delta_by_checkpoint(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    if frame.empty:
        return {}
    primary = frame[frame["source"].eq("tabular")].copy()
    if primary.empty:
        primary = frame.copy()
    result: dict[str, dict[str, float | None]] = {}
    for checkpoint, group in primary.groupby("checkpoint", sort=False):
        result[str(checkpoint)] = {
            str(row["temporal_weighting_policy"]): _number_or_none(
                row["delta_vs_uniform_mae_gap_sec"]
            )
            for row in group.to_dict("records")
        }
    return result


def _composition_summary(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    if frame.empty:
        return {}
    result: dict[str, dict[str, float | None]] = {}
    for policy, group in frame.groupby("temporal_weighting_policy", sort=False):
        result[str(policy)] = {
            "mean_effective_sample_size": _number_or_none(group["effective_sample_size"].mean()),
            "mean_same_season_weight_share": _number_or_none(
                group["same_season_weight_share"].mean()
            ),
            "mean_prior_season_weight_share": _number_or_none(
                group["prior_season_weight_share"].mean()
            ),
        }
    return result


def _main_findings(
    frame: pd.DataFrame,
    best_by_checkpoint: dict[str, dict[str, object]],
) -> list[str]:
    if frame.empty:
        return ["Temporal weighting artifacts were not available for comparison."]
    findings = [
        "Season-aware weighting is evaluated as an opt-in training policy; "
        "champion defaults remain unchanged."
    ]
    for checkpoint in CHECKPOINT_ORDER:
        best = best_by_checkpoint.get(checkpoint)
        if not best:
            continue
        delta = best.get("delta_vs_uniform_mae_gap_sec")
        if delta is None:
            continue
        if abs(float(delta)) < 0.01:
            findings.append(f"{checkpoint} differences versus uniform are small.")
        elif float(delta) < 0:
            findings.append(
                f"{checkpoint} best temporal policy improves MAE versus uniform by "
                f"{abs(float(delta)):.3f} sec."
            )
        else:
            findings.append(
                f"{checkpoint} best available temporal policy does not beat uniform in "
                "current artifacts."
            )
    return findings


def _mean_composition(rows: list[dict[str, object]]) -> dict[str, float | None]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "mean_effective_sample_size": None,
            "mean_same_season_training_weight_share": None,
            "mean_prior_season_training_weight_share": None,
        }
    return {
        "mean_effective_sample_size": _number_or_none(frame["effective_sample_size"].mean()),
        "mean_same_season_training_weight_share": _number_or_none(
            frame["same_season_weight_share"].mean()
        ),
        "mean_prior_season_training_weight_share": _number_or_none(
            frame["prior_season_weight_share"].mean()
        ),
    }


def _plot_checkpoint_bars(
    plt: Any,
    frame: pd.DataFrame,
    path: Path,
    value_column: str,
    title: str,
    ylabel: str,
) -> bool:
    data = frame[frame["source"].eq("tabular")].copy()
    if data.empty or value_column not in data:
        return False
    pivot = data.pivot_table(
        index="checkpoint",
        columns="temporal_weighting_policy",
        values=value_column,
        aggfunc="first",
    ).reindex(CHECKPOINT_ORDER)
    pivot = pivot.dropna(axis=0, how="all")
    if pivot.empty:
        return False
    ax = pivot.plot(kind="bar", figsize=(8, 4), width=0.8)
    ax.set_title(title)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(title="Policy", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _plot_composition(
    plt: Any,
    frame: pd.DataFrame,
    path: Path,
    value_column: str,
    title: str,
) -> bool:
    if frame.empty or value_column not in frame:
        return False
    data = frame[frame["source"].eq("tabular")].copy()
    if data.empty:
        data = frame.copy()
    grouped = data.groupby("temporal_weighting_policy", sort=False)[value_column].mean()
    grouped = grouped.dropna()
    if grouped.empty:
        return False
    ax = grouped.plot(kind="bar", figsize=(7, 4))
    ax.set_title(title)
    ax.set_xlabel("Policy")
    ax.set_ylabel(value_column)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _checkpoint_columns() -> list[str]:
    return [
        "source",
        "temporal_weighting_policy",
        "checkpoint",
        "model_name",
        "feature_group",
        "mae_gap_sec",
        "rmse_gap_sec",
        "median_abs_error_gap_sec",
        "mean_abs_position_error",
        "delta_vs_uniform_mae_gap_sec",
        "n_folds_successful",
        "mean_effective_sample_size",
        "mean_same_season_training_weight_share",
        "mean_prior_season_training_weight_share",
    ]


def _fold_columns() -> list[str]:
    return [
        "source",
        "temporal_weighting_policy",
        "fold_id",
        "checkpoint",
        "model_name",
        "mae_gap_sec",
        "delta_vs_uniform_mae_gap_sec",
    ]


def _composition_columns() -> list[str]:
    return [
        "source",
        "temporal_weighting_policy",
        "fold_id",
        "test_event",
        "test_season",
        "training_rows",
        "training_events",
        "same_season_training_rows",
        "prior_season_training_rows",
        "older_season_training_rows",
        "same_season_training_events",
        "prior_season_training_events",
        "older_season_training_events",
        "weight_min",
        "weight_max",
        "weight_mean",
        "weight_sum",
        "effective_sample_size",
        "same_season_weight_sum",
        "prior_season_weight_sum",
        "older_season_weight_sum",
        "same_season_weight_share",
        "prior_season_weight_share",
        "older_season_weight_share",
        "filtered_future_rows",
    ]


def _delta(value: object, baseline: object) -> float | None:
    if value is None or pd.isna(value) or baseline is None or pd.isna(baseline):
        return None
    return float(value) - float(baseline)


def _number_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _relative_report_path(path: Path) -> str:
    try:
        return f"reports/{path.relative_to(path.parents[1]).as_posix()}"
    except ValueError:
        return path.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
