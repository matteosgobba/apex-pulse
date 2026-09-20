"""Reproduction, aligned diagnostics and predeclared governance sensitivity for V2."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from f1_prediction.config import (
    ModelConfig,
    load_data_config,
    load_feature_config,
    load_model_config,
)
from f1_prediction.data.season_builder import build_combined_dataset_path
from f1_prediction.modeling.prospective_replay_v2 import (
    KEYS,
    PRED,
    TARGET,
    UNIFORM,
    WEIGHTED,
)
from f1_prediction.modeling.season_aware_validation import current_season_evidence_regime

SEED = 2026022
ITERATIONS = 10_000


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_protected_artifacts(output: Path) -> None:
    """Fail if the invocation changed a previously fingerprinted production artifact."""
    manifest = output / "protected_artifacts_before.json"
    if not manifest.exists():
        return
    before = json.loads(manifest.read_text())
    changed = [
        name
        for name, expected in before.items()
        if not Path(name).is_file()
        or hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected
    ]
    save_json(
        output / "protected_artifacts_verification.json",
        {
            "checked_files": len(before),
            "changed_files": changed,
            "all_unchanged": not changed,
        },
    )
    if changed:
        raise ValueError(f"Protected artifacts changed: {changed}")


def align_settlements(settlements: pd.DataFrame) -> pd.DataFrame:
    """Inner alignment with target equality and unique candidate-key enforcement."""
    frame = settlements[settlements.is_evaluation_event & settlements.evaluable].copy()
    keys = ["test_season", *KEYS, "event", "event_key", "fold_id"]
    if frame.duplicated([*keys, "role"]).any():
        raise ValueError("Duplicate aligned evaluation key")
    if frame.groupby(keys)[TARGET].nunique().gt(1).any():
        raise ValueError("Candidate outcomes disagree")
    counts = frame.groupby(keys).role.nunique()
    if not counts.eq(3).all():
        raise ValueError("Evaluation rows differ across policies")
    prediction = frame.pivot(index=keys, columns="role", values=PRED).reset_index()
    actual = frame.drop_duplicates(keys)[[*keys, TARGET]]
    aligned = prediction.merge(actual, on=keys, validate="one_to_one")
    for role, name in [(UNIFORM, "uniform"), (WEIGHTED, "weighted"), ("selected", "selected")]:
        aligned[f"{name}_prediction"] = aligned.pop(role)
        aligned[f"{name}_absolute_error"] = (aligned[f"{name}_prediction"] - aligned[TARGET]).abs()
    aligned["d_uniform_minus_weighted"] = (
        aligned.uniform_absolute_error - aligned.weighted_absolute_error
    )
    return aligned.sort_values(["test_season", "fold_id", "driver"]).reset_index(drop=True)


def distribution(values: pd.Series) -> dict[str, float | int]:
    x = values.to_numpy(dtype=float)
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "min": float(x.min()),
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
        "max": float(x.max()),
        "fraction_positive": float((x > 0).mean()),
        "fraction_negative": float((x < 0).mean()),
        "fraction_zero": float((x == 0).mean()),
    }


def bootstrap_events(events: pd.DataFrame) -> dict[str, object]:
    """Paired whole-event bootstrap, stratified by season when pooling seasons.

    Report both equal-weekend and row-weighted estimands. Temporal dependence,
    overlapping fitted histories and development reuse preclude formal proof.
    """
    rng = np.random.default_rng(SEED)
    event_sums = np.zeros(ITERATIONS)
    row_sums = np.zeros(ITERATIONS)
    row_counts = np.zeros(ITERATIONS)
    for _, group in events.groupby("test_season", sort=True):
        sample = rng.integers(0, len(group), size=(ITERATIONS, len(group)))
        delta = group.d_uniform_minus_weighted.to_numpy()
        n = group.rows.to_numpy()
        event_sums += delta[sample].sum(axis=1)
        row_sums += (delta * n)[sample].sum(axis=1)
        row_counts += n[sample].sum(axis=1)
    return {
        "diagnostic_only": True,
        "seed": SEED,
        "iterations": ITERATIONS,
        "resampling_unit": "whole_event",
        "stratified_by_season": True,
        "event_mean": float(events.d_uniform_minus_weighted.mean()),
        "event_mean_ci95": np.quantile(event_sums / len(events), [0.025, 0.975]).tolist(),
        "row_mae_difference": float(
            np.average(events.d_uniform_minus_weighted, weights=events.rows)
        ),
        "row_mae_difference_ci95": np.quantile(row_sums / row_counts, [0.025, 0.975]).tolist(),
    }


def event_metrics(aligned: pd.DataFrame, gates: pd.DataFrame) -> pd.DataFrame:
    grouped = aligned.groupby(["test_season", "event_key", "event", "fold_id"], sort=False)
    events = grouped.agg(
        rows=("driver", "size"),
        uniform_mae=("uniform_absolute_error", "mean"),
        weighted_mae=("weighted_absolute_error", "mean"),
        selected_mae=("selected_absolute_error", "mean"),
        d_uniform_minus_weighted=("d_uniform_minus_weighted", "mean"),
    ).reset_index()
    events = events.merge(
        gates[
            [
                "test_season",
                "event_key",
                "current_season_prior_events",
                "selected_policy",
                "all_gates_pass",
                "history_eligible",
                "selection_reason",
            ]
        ],
        on=["test_season", "event_key"],
        validate="one_to_one",
    )
    events["regime"] = events.current_season_prior_events.map(current_season_evidence_regime)
    for _, group in events.groupby("test_season", sort=False):
        for role in ("uniform", "weighted", "selected"):
            events.loc[group.index, f"cumulative_{role}_mae"] = (
                group[f"{role}_mae"] * group.rows
            ).cumsum() / group.rows.cumsum()
        changes = group.selected_policy.ne(group.selected_policy.shift())
        changes.iloc[0] = False
        events.loc[group.index, "switch"] = changes
    return events


def selection_stats(events: pd.DataFrame, selected: np.ndarray) -> dict[str, object]:
    errors = np.where(selected, events.weighted_mae, events.uniform_mae)
    return {
        "weighted_events": int(selected.sum()),
        "weighted_fraction": float(selected.mean()),
        "selected_mae": float(np.average(errors, weights=events.rows)),
        "switches": int(np.count_nonzero(np.diff(selected.astype(int)))),
        "selection_sequence": "".join("W" if flag else "U" for flag in selected),
        "selected_events": ",".join(events.loc[selected, "event_key"]),
    }


def sensitivity_analysis(
    events: pd.DataFrame, gates: pd.DataFrame, model_config: ModelConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All predeclared neighbors, evaluated only after the completed primary replay."""
    summaries, timelines = [], []
    frozen = model_config.champion_policy.season_aware_nested_guarded
    for season, group in events.groupby("test_season", sort=True):
        state = gates.set_index("event_key").loc[group.event_key]
        for current, folds, predictions, margin in itertools.product(
            (4, 5, 6), (4, 5, 6), (80, 100, 120), (0.025, 0.05, 0.075)
        ):
            selected = (
                state.current_season_prior_events.ge(current)
                & state.prior_candidate_folds.ge(folds)
                & state.prior_candidate_predictions.ge(predictions)
                & state.aligned_prior_folds.ge(folds)
                & state.aligned_prior_rows.ge(predictions)
                & state.prior_improvement_sec.ge(margin)
            ).to_numpy()
            is_primary = (current, folds, predictions, margin) == (
                frozen.min_current_season_prior_events,
                frozen.min_prior_candidate_folds,
                frozen.min_prior_candidate_predictions,
                frozen.improvement_margin_sec,
            )
            if is_primary and not np.array_equal(selected, group.selected_policy.eq(WEIGHTED)):
                raise AssertionError("Sensitivity's frozen setting differs from primary selector")
            meta = {
                "test_season": int(season),
                "min_current_season_prior_events": current,
                "min_prior_folds": folds,
                "min_prior_predictions": predictions,
                "improvement_margin_sec": margin,
                "is_primary_setting": is_primary,
            }
            summaries.append({**meta, **selection_stats(group, selected)})
            for event, flag in zip(group.event_key, selected, strict=True):
                timelines.append({**meta, "event_key": event, "selected_weighted": bool(flag)})
        no_margin = (
            state.current_season_prior_events.ge(frozen.min_current_season_prior_events)
            & state.prior_candidate_folds.ge(frozen.min_prior_candidate_folds)
            & state.prior_candidate_predictions.ge(frozen.min_prior_candidate_predictions)
            & state.aligned_prior_folds.ge(frozen.min_prior_candidate_folds)
            & state.aligned_prior_rows.ge(frozen.min_prior_candidate_predictions)
            & state.prior_improvement_sec.ge(0)
        ).to_numpy()
        summaries.append(
            {
                "test_season": int(season),
                "diagnostic": "zero_margin_churn_check",
                "improvement_margin_sec": 0.0,
                **selection_stats(group, no_margin),
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(timelines)


def reset_evidence_sensitivity(events: pd.DataFrame, model_config: ModelConfig) -> pd.DataFrame:
    """Start target-season candidate history at zero; keep fits and all gates fixed."""
    settings = model_config.champion_policy.season_aware_nested_guarded
    rows = []
    for season, group in events.groupby("test_season", sort=True):
        n = 0
        uniform_sum = weighted_sum = 0.0
        for i, event in enumerate(group.itertuples(index=False)):
            improvement = (uniform_sum - weighted_sum) / n if n else None
            history_pass = (
                event.current_season_prior_events >= settings.min_current_season_prior_events
                and i >= settings.min_prior_candidate_folds
                and n >= settings.min_prior_candidate_predictions
            )
            selected = (
                history_pass
                and improvement is not None
                and improvement >= settings.improvement_margin_sec
            )
            rows.append(
                {
                    "test_season": int(season),
                    "event_key": event.event_key,
                    "prior_folds": i,
                    "prior_rows": n,
                    "prior_improvement_sec": improvement,
                    "history_gate": history_pass,
                    "weighted_selected": selected,
                    "selected_mae": event.weighted_mae if selected else event.uniform_mae,
                    "rows": event.rows,
                }
            )
            n += event.rows
            uniform_sum += event.rows * event.uniform_mae
            weighted_sum += event.rows * event.weighted_mae
    return pd.DataFrame(rows)


def analyze_replay(output: Path, *, model_config: ModelConfig) -> None:
    """Write aligned evidence, uncertainty and governance diagnostics separately."""
    settlements = pd.read_csv(output / "settlements.csv", float_precision="round_trip")
    gates = pd.read_csv(output / "gates.csv", float_precision="round_trip")
    eval_gates = gates[gates.is_evaluation_event].copy()
    aligned = align_settlements(settlements)
    events = event_metrics(aligned, eval_gates)
    aligned = aligned.merge(
        events[["test_season", "event_key", "current_season_prior_events", "regime"]],
        on=["test_season", "event_key"],
        validate="many_to_one",
    )
    aligned.to_csv(output / "aligned_predictions_errors.csv", index=False)
    events.to_csv(output / "per_event_metrics.csv", index=False)
    eval_gates.to_csv(output / "per_event_gate_states.csv", index=False)
    events[["test_season", "event_key", "selected_policy", "switch", "selection_reason"]].to_csv(
        output / "selected_policy_by_event.csv", index=False
    )
    summaries, diagnostics, regimes, uncertainty = [], {}, [], []
    for season, group in events.groupby("test_season", sort=True):
        selected = group.selected_policy.eq(WEIGHTED).to_numpy()
        eligible = group[group.all_gates_pass]
        eligible_history = group[group.history_eligible]
        item = {
            "season": int(season),
            "events": len(group),
            "rows": int(group.rows.sum()),
            "cold_start_events": int(group.current_season_prior_events.lt(5).sum()),
            "first_history_eligible_event": eligible_history.event_key.iloc[0]
            if len(eligible_history)
            else None,
            "first_all_gates_pass_event": eligible.event_key.iloc[0] if len(eligible) else None,
            "uniform_mae": float(np.average(group.uniform_mae, weights=group.rows)),
            "weighted_mae": float(np.average(group.weighted_mae, weights=group.rows)),
            "uniform_events": int((~selected).sum()),
            "uniform_fraction": float((~selected).mean()),
            **selection_stats(group, selected),
        }
        for label, mask in [
            (
                "before_first_selection",
                group.fold_id.lt(eligible.fold_id.iloc[0])
                if len(eligible)
                else pd.Series(True, index=group.index),
            ),
            (
                "after_first_selection",
                group.fold_id.ge(eligible.fold_id.iloc[0])
                if len(eligible)
                else pd.Series(False, index=group.index),
            ),
        ]:
            part = group[mask]
            item[label] = {
                "events": len(part),
                "rows": int(part.rows.sum()),
                **{
                    f"{role}_mae": float(np.average(part[f"{role}_mae"], weights=part.rows))
                    if len(part)
                    else None
                    for role in ("uniform", "weighted", "selected")
                },
            }
        summaries.append(item)
        part = aligned[aligned.test_season.eq(season)]
        diagnostics[str(season)] = {
            "rows": distribution(part.d_uniform_minus_weighted),
            "events": distribution(group.d_uniform_minus_weighted),
            "bootstrap": bootstrap_events(group),
            "selected_vs_uniform_bootstrap": bootstrap_events(
                group.assign(d_uniform_minus_weighted=group.uniform_mae - group.selected_mae)
            ),
            "leave_one_event_out_event_mean_min": float(
                min(group.drop(i).d_uniform_minus_weighted.mean() for i in group.index)
            ),
            "leave_one_event_out_event_mean_max": float(
                max(group.drop(i).d_uniform_minus_weighted.mean() for i in group.index)
            ),
            "top_three_positive_share": float(
                group.d_uniform_minus_weighted.nlargest(3).clip(lower=0).sum()
                / group.d_uniform_minus_weighted.clip(lower=0).sum()
            ),
        }
        for name, mask in [
            ("cold_start", group.current_season_prior_events.lt(5)),
            ("post_cold_start", group.current_season_prior_events.ge(5)),
            ("early_season", group.regime.eq("early_season")),
            ("established_season", group.regime.eq("established_season")),
        ]:
            sub = group[mask]
            if sub.empty:
                continue
            regimes.append(
                {
                    "test_season": int(season),
                    "regime": name,
                    "events": len(sub),
                    "rows": int(sub.rows.sum()),
                    **{
                        f"{role}_mae": float(np.average(sub[f"{role}_mae"], weights=sub.rows))
                        for role in ("uniform", "weighted", "selected")
                    },
                    "event_mean_difference": float(sub.d_uniform_minus_weighted.mean()),
                    "improved_event_fraction": float(sub.d_uniform_minus_weighted.gt(0).mean()),
                    "bootstrap_event_ci95": json.dumps(bootstrap_events(sub)["event_mean_ci95"]),
                }
            )
    for (season, role), group in settlements[
        settlements.is_evaluation_event & settlements.evaluable
    ].groupby(["test_season", "role"], sort=False):
        valid = group.dropna(subset=["prediction_interval_low_sec"])
        uncertainty.append(
            {
                "test_season": int(season),
                "role": role,
                "interval_rows": len(valid),
                "evaluable_rows": len(group),
                "coverage": float(valid.interval_contains_actual.astype(bool).mean()),
                "mean_width": float(
                    (valid.prediction_interval_high_sec - valid.prediction_interval_low_sec).mean()
                ),
            }
        )
    diagnostics["pooled"] = {
        "rows": distribution(aligned.d_uniform_minus_weighted),
        "events": distribution(events.d_uniform_minus_weighted),
        "bootstrap": bootstrap_events(events),
    }
    save_json(
        output / "summary.json",
        {
            "evaluation_type": "prospective_replay_v2",
            "evidence_class": "causal_historical_replay_not_untouched_prospective",
            "production_default_changed": False,
            "seasons": summaries,
        },
    )
    save_json(output / "paired_uncertainty.json", diagnostics)
    pd.DataFrame(regimes).to_csv(output / "regime_comparison.csv", index=False)
    pd.DataFrame(uncertainty).to_csv(output / "interval_diagnostics.csv", index=False)
    sensitivity, timelines = sensitivity_analysis(events, eval_gates, model_config)
    sensitivity.to_csv(output / "sensitivity_summary.csv", index=False)
    timelines.to_csv(output / "sensitivity_event_selections.csv", index=False)
    reset_evidence_sensitivity(events, model_config).to_csv(
        output / "season_reset_sensitivity.csv", index=False
    )
    fits = pd.read_csv(output / "training_manifest.csv")
    features = json.loads(fits.feature_columns.iloc[0])
    save_json(
        output / "feature_manifest.json",
        {
            "feature_group": "base_plus_relative",
            "count": len(features),
            "columns": features,
            "same_order_all_fits": bool(fits.feature_columns.nunique() == 1),
        },
    )
    write_results_appendix(output, events, eval_gates, diagnostics, summaries, sensitivity)
    plot_results(events, output)
    verify_protected_artifacts(output)


def markdown_table(frame: pd.DataFrame) -> str:
    """Format small report tables without an optional tabulate dependency."""

    def cell(value: object) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "/")

    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_results_appendix(
    output: Path,
    events: pd.DataFrame,
    gates: pd.DataFrame,
    diagnostics: dict,
    summaries: list[dict],
    sensitivity: pd.DataFrame,
) -> None:
    """Regenerate every numeric table in the audit from the actual aligned outputs."""
    distribution_rows = []
    for season, payload in diagnostics.items():
        for unit in ("rows", "events"):
            distribution_rows.append({"season": season, "unit": unit, **payload[unit]})
    sections = [
        "# Replay V2 — generated numeric appendix",
        "All errors are seconds. Positive d = uniform absolute error "
        "minus weighted absolute error. "
        "Read report.md for scope, contamination and interpretation.",
        "## Aligned primary results",
        markdown_table(
            pd.DataFrame(summaries)[
                [
                    "season",
                    "events",
                    "rows",
                    "cold_start_events",
                    "uniform_mae",
                    "weighted_mae",
                    "selected_mae",
                    "uniform_events",
                    "weighted_events",
                    "weighted_fraction",
                    "switches",
                ]
            ]
        ),
        "## Per-event and cumulative errors",
        markdown_table(
            events[
                [
                    "event_key",
                    "rows",
                    "current_season_prior_events",
                    "uniform_mae",
                    "weighted_mae",
                    "selected_mae",
                    "d_uniform_minus_weighted",
                    "cumulative_uniform_mae",
                    "cumulative_weighted_mae",
                    "cumulative_selected_mae",
                ]
            ]
        ),
        "## Gate-by-gate timeline",
        "Available/checkpoint/history-scope gates pass at all evaluation events. "
        "Folds require >=5; candidate and aligned predictions require >=100; "
        "current-season events require >=5; margin requires >=0.05. "
        "The CSV additionally records observed/required values "
        "and a pass/fail reason for every gate.",
        markdown_table(
            gates[
                [
                    "event_key",
                    "current_season_prior_events",
                    "prior_candidate_folds",
                    "prior_candidate_predictions",
                    "aligned_prior_rows",
                    "gate_current_season",
                    "gate_prior_folds",
                    "gate_prior_predictions",
                    "gate_aligned_folds",
                    "gate_aligned_rows",
                    "gate_finite_metrics",
                    "prior_improvement_sec",
                    "gate_margin",
                    "selection_reason",
                ]
            ]
        ),
        "## Cold-start and established-season regimes",
        markdown_table(pd.read_csv(output / "regime_comparison.csv")),
        "## Before/after first all-gates eligibility",
        markdown_table(
            pd.DataFrame(
                [
                    {"season": s["season"], "period": period, **s[period]}
                    for s in summaries
                    for period in ("before_first_selection", "after_first_selection")
                ]
            )
        ),
        "## Paired error distributions",
        markdown_table(pd.DataFrame(distribution_rows)),
        "## Diagnostic paired whole-event bootstrap",
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "season": season,
                        **{
                            k: json.dumps(v) if isinstance(v, list) else v
                            for k, v in payload["bootstrap"].items()
                        },
                    }
                    for season, payload in diagnostics.items()
                ]
            )
        ),
        "## Governance sensitivity — separate from primary replay",
        markdown_table(
            sensitivity[sensitivity.diagnostic.isna()]
            .groupby("test_season")
            .agg(
                settings=("weighted_events", "size"),
                min_selected=("weighted_events", "min"),
                max_selected=("weighted_events", "max"),
                min_mae=("selected_mae", "min"),
                max_mae=("selected_mae", "max"),
                min_switches=("switches", "min"),
                max_switches=("switches", "max"),
            )
            .reset_index()
        ),
        "All 81 settings per season and their event timelines are in sensitivity_summary.csv "
        "and sensitivity_event_selections.csv. No setting was selected or deployed.",
        "## Zero candidate history at season start — initialization sensitivity",
        markdown_table(pd.read_csv(output / "season_reset_sensitivity.csv")),
        "## Intervals (not promotion gates)",
        markdown_table(pd.read_csv(output / "interval_diagnostics.csv")),
    ]
    (output / "results.md").write_text("\n\n".join(sections) + "\n")


def plot_results(events: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for season, group in events.groupby("test_season", sort=True):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        x = np.arange(len(group))
        for role in ("uniform", "weighted", "selected"):
            axes[0, 0].plot(
                x,
                group[f"cumulative_{role}_mae"],
                label=role,
                marker=".",
                linestyle="--" if role == "selected" else "-",
            )
        axes[0, 0].set(title="Cumulative aligned MAE", ylabel="Seconds")
        axes[0, 0].legend(loc="upper left")
        axes[0, 1].bar(
            x,
            group.d_uniform_minus_weighted,
            color=np.where(group.d_uniform_minus_weighted.ge(0), "#19806b", "#ba5347"),
        )
        axes[0, 1].axhline(0, color="black", lw=0.7)
        axes[0, 1].set(
            title="Uniform minus weighted event MAE", ylabel="Seconds (positive favors weighted)"
        )
        axes[1, 0].step(x, group.selected_policy.eq(WEIGHTED).astype(int), where="mid")
        axes[1, 0].set(
            title="Frozen-gate simulated selection",
            yticks=[0, 1],
            yticklabels=["Uniform", "Weighted"],
            ylim=(-0.15, 1.15),
        )
        axes[1, 1].scatter(group.current_season_prior_events, group.d_uniform_minus_weighted)
        axes[1, 1].axvline(5, color="gray", ls="--", label="Frozen history threshold")
        axes[1, 1].axhline(0, color="black", lw=0.7)
        axes[1, 1].set(
            title="Candidate difference versus history",
            xlabel="Prior represented current-season events",
            ylabel="Seconds (positive favors weighted)",
        )
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.set_xticks(x, group.event, rotation=75, fontsize=8)
        for ax in axes.flat:
            ax.title.set_fontsize(11)
        fig.suptitle(
            f"{season}: causal historical simulation; development-contaminated seasons", fontsize=12
        )
        fig.savefig(output / f"results_{season}.png", dpi=150)
        plt.close(fig)


def reproduce_original(output: Path) -> None:
    """Retrain the unchanged original replay in isolation, then compare saved artifacts."""
    from f1_prediction.modeling.prospective_policy_evaluation import build_frozen_policy_profiles
    from f1_prediction.modeling.prospective_replay import POLICY_PROFILES, run_true_replay
    from f1_prediction.modeling.splits import ordered_event_keys

    config, model, feature = load_data_config(), load_model_config(), load_feature_config()
    dataset = pd.read_parquet(build_combined_dataset_path(config.modeling_output_dir))
    profiles = build_frozen_policy_profiles(
        model, profile_names=POLICY_PROFILES, uncertainty="conformal_predicted_gap_bucket"
    )
    directory = output / "original_reproduction"
    directory.mkdir(parents=True, exist_ok=True)
    for season, train in [(2024, (2023,)), (2025, (2023, 2024))]:
        print(f"Reproducing original {season}", flush=True)
        result = run_true_replay(
            dataset=dataset,
            event_order=ordered_event_keys(dataset),
            profiles=profiles,
            model_config=model,
            feature_config=feature,
            train_seasons=train,
            test_season=season,
            min_train_events=5,
            uncertainty="conformal_predicted_gap_bucket",
        )
        for name, frame in result.items():
            frame.to_parquet(directory / f"{season}_{name}.parquet", index=False)
    verify_previous_results(output)


def verify_previous_results(output: Path) -> None:
    """Recompute previous metrics from rows and test fresh replay prediction parity."""
    metrics = load_data_config().metrics_output_dir
    rows, parity = [], []
    names = [
        "ablation_uniform_predictions.parquet",
        "ablation_current_season_only_with_prior_predictions.parquet",
        "prospective_train_2023_test_2024_predictions.parquet",
        "prospective_train_2023_2024_test_2025_predictions.parquet",
        "prospective_replay_train_2023_test_2024_predictions.parquet",
        "prospective_replay_train_2023_2024_test_2025_predictions.parquet",
        "prospective_replay_shadow_candidates.parquet",
    ]
    for name in names:
        frame = pd.read_parquet(metrics / name)
        frame = frame[frame.checkpoint.eq("after_fp3")].copy()
        if name.startswith("ablation"):
            frame = frame[
                frame.feature_group.eq("base_plus_relative") & frame.model_name.eq("random_forest")
            ]
        pred = "prediction_gap_sec" if "prediction_gap_sec" in frame else PRED
        actual = "actual_gap_sec" if "actual_gap_sec" in frame else TARGET
        frame["error"] = (frame[pred] - frame[actual]).abs()
        groups = [c for c in ("policy_profile", "shadow_role", "test_season") if c in frame]
        grouped = frame.groupby(groups, sort=False) if groups else [("all", frame)]
        for key, part in grouped:
            rows.append(
                {
                    "artifact": name,
                    "group": str(key),
                    "rows": int(part.error.count()),
                    "events": len(part[["season", "event_slug"]].drop_duplicates()),
                    "mae": float(part.error.mean()),
                    "scope": "includes_warmup" if "shadow" in name else "evaluation",
                }
            )
        if "shadow" in name:
            for (season, role), part in frame[frame.season.eq(frame.test_season)].groupby(
                ["test_season", "shadow_role"]
            ):
                rows.append(
                    {
                        "artifact": name,
                        "group": f"{season}/{role}",
                        "rows": len(part),
                        "events": part.event_slug.nunique(),
                        "mae": float(part.error.mean()),
                        "scope": "evaluation_only",
                    }
                )
    for season, train in [(2024, "2023"), (2025, "2023_2024")]:
        fresh = pd.read_parquet(output / "original_reproduction" / f"{season}_predictions.parquet")
        old = pd.read_parquet(
            metrics / f"prospective_replay_train_{train}_test_{season}_predictions.parquet"
        )
        keys = [*KEYS, "policy_profile"]
        merged = fresh.merge(old, on=keys, suffixes=("_fresh", "_old"), validate="one_to_one")
        delta = (merged[f"{PRED}_fresh"] - merged[f"{PRED}_old"]).abs()
        parity.append(
            {
                "comparison": f"original replay {season}, all checkpoints",
                "rows": len(merged),
                "max_prediction_difference": float(delta.max()),
                "all_keys_equal": len(merged) == len(old) == len(fresh),
            }
        )
    # The freshly fitted 2025 replay shadows span every retrospective evaluation fold.
    fresh_shadow = pd.read_parquet(output / "original_reproduction" / "2025_shadow.parquet")
    for policy, role in [
        (UNIFORM, "uniform_default"),
        (WEIGHTED, "season_aware_weighted_candidate"),
    ]:
        artifact = pd.read_parquet(metrics / f"ablation_{policy}_predictions.parquet")
        artifact = artifact[
            artifact.checkpoint.eq("after_fp3")
            & artifact.feature_group.eq("base_plus_relative")
            & artifact.model_name.eq("random_forest")
        ]
        source = fresh_shadow[fresh_shadow.shadow_role.eq(role)]
        merged = artifact.merge(source, on=KEYS, validate="one_to_one", suffixes=("_old", "_fresh"))
        parity.append(
            {
                "comparison": f"retrospective {policy} versus fresh fits",
                "rows": len(merged),
                "max_prediction_difference": float(
                    (merged[PRED] - merged.prediction_gap_sec).abs().max()
                ),
                "all_keys_equal": len(merged) == len(source) == len(artifact),
            }
        )
    pd.DataFrame(rows).to_csv(output / "previous_results_recomputed.csv", index=False)
    save_json(output / "reproduction_parity.json", parity)
    monitoring_evidence_inventory(metrics, output)
    verify_protected_artifacts(output)


def monitoring_evidence_inventory(metrics: Path, output: Path) -> None:
    """Inventory later monitoring separately; stored integrity flags are not certification."""
    path = metrics / "prospective_monitoring_settlements.parquet"
    if not path.is_file():
        return
    frame = pd.read_parquet(path)
    rows = []
    for (season, slug, role), part in frame.groupby(
        ["season", "event_slug", "prediction_role"], sort=False
    ):
        valid = part[part.included_in_metrics & part.settlement_valid].dropna(
            subset=["prediction_gap_sec", "actual_gap_sec"]
        )
        classification = "stored_monitoring_record_not_independently_recertified_here"
        if str(slug).startswith("synthetic"):
            classification = "excluded_synthetic"
        elif str(slug) in {"australia", "great-britain"}:
            classification = "excluded_legacy_per_project_identity_audit"
        rows.append(
            {
                "season": int(season),
                "event_slug": slug,
                "role": role,
                "classification": classification,
                "rows": len(valid),
                "mae_recomputed": float(
                    (valid.prediction_gap_sec - valid.actual_gap_sec).abs().mean()
                )
                if len(valid)
                else None,
                "included_in_v2": False,
            }
        )
    pd.DataFrame(rows).to_csv(output / "supplemental_monitoring_inventory.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/prospective_replay_v2"))
    parser.add_argument("--reproduce-original", action="store_true")
    parser.add_argument("--verify-previous", action="store_true")
    args = parser.parse_args()
    if args.reproduce_original:
        reproduce_original(args.output_dir)
    elif args.verify_previous:
        verify_previous_results(args.output_dir)
    else:
        analyze_replay(args.output_dir, model_config=load_model_config())


if __name__ == "__main__":
    main()
