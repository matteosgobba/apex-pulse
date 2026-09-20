"""Isolated, target-blind FP3 replay with immutable forecasts and prior shadow evidence.

This module is evaluation-only. It is never imported by production monitoring and
never reads or writes its prediction/evidence artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_prediction.config import (
    FeatureConfig,
    ModelConfig,
    SeasonAwareNestedGuardedChampionConfig,
    load_data_config,
    load_feature_config,
    load_model_config,
)
from f1_prediction.data.season_builder import build_combined_dataset_path
from f1_prediction.features.qualifying_targets import TARGET_COLUMNS
from f1_prediction.modeling.backtest_tabular import BacktestFold
from f1_prediction.modeling.champion_policy import select_season_aware_guarded_method
from f1_prediction.modeling.feature_groups import get_feature_columns_for_group
from f1_prediction.modeling.prospective_monitoring import train_monitoring_event_sources
from f1_prediction.modeling.prospective_replay import (
    FP3_CHECKPOINT,
    add_replay_intervals,
    event_key_series,
    event_season,
    prior_events_for,
)
from f1_prediction.modeling.splits import ordered_event_keys
from f1_prediction.modeling.tabular import usable_checkpoint_features
from f1_prediction.modeling.temporal_weighting import prepare_temporal_training_data

VERSION = "prospective_replay_v2"
UNIFORM = "uniform"
WEIGHTED = "current_season_only_with_prior"
KEYS = ["season", "event_slug", "checkpoint", "driver"]
PRED = "predicted_quali_gap_to_pole_sec"
TARGET = "quali_gap_to_pole_sec"
LOGGER = logging.getLogger(__name__)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize missing values as null and retain full float round-trip precision."""
    return [
        {key: None if pd.isna(value) else value for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def encode(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(payload: object) -> str:
    return hashlib.sha256(encode(payload)).hexdigest()


def frame_digest(frame: pd.DataFrame) -> str:
    """Fingerprint values, row/column order and dtypes, without the pandas index."""
    hashes = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    schema = encode([(str(c), str(t)) for c, t in frame.dtypes.items()])
    return hashlib.sha256(schema + hashes).hexdigest()


def write_immutable(path: Path, payload: object) -> None:
    """Create a record once; accept only byte-identical reruns, never replace it."""
    content = encode(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError(f"Immutable replay record differs: {path}") from None


def freeze_legal_information(
    dataset: pd.DataFrame, event_key: str, legal_events: list[str]
) -> pd.DataFrame:
    """Expose completed history and targetless current rows; exclude every future row."""
    keys = event_key_series(dataset)
    scope = dataset[keys.isin([*legal_events, event_key])].copy()
    current = event_key_series(scope).eq(event_key)
    for column in TARGET_COLUMNS:
        if column in scope:
            scope.loc[current, column] = np.nan
    return scope


def evaluate_gates(
    history: pd.DataFrame,
    current: pd.DataFrame,
    *,
    event_key: str,
    step: int,
    legal_events: list[str],
    model_config: ModelConfig,
    settings: SeasonAwareNestedGuardedChampionConfig | None = None,
) -> dict[str, Any]:
    """Use the unchanged canonical selector, with independently logged prior gates."""
    settings = settings or model_config.champion_policy.season_aware_nested_guarded
    if not history.empty:
        if not history["fold_id"].lt(step).all():
            raise ValueError("Current/future event supplied as prior candidate evidence")
        if not history["available_from_step"].le(step).all():
            raise ValueError("Shadow residual is not settled/available yet")
        if not set(event_key_series(history)).issubset(legal_events):
            raise ValueError("Candidate evidence is outside legal training history")
    if current[TARGET].notna().any():
        raise ValueError("Current outcomes must be hidden during selection")
    if current.duplicated([*KEYS, "temporal_weighting_policy"]).any():
        raise ValueError("Duplicate candidate prediction key")
    candidates = pd.concat([history, current], ignore_index=True)
    default = model_config.champion_policy.static[FP3_CHECKPOINT]
    fold = BacktestFold(
        fold_id=step,
        strategy="walk_forward",
        test_event=event_key,
        train_events=tuple(legal_events),
        train_rows=0,
        test_rows=len(current) // 2,
    )
    decision = select_season_aware_guarded_method(
        candidates,
        fold=fold,
        checkpoint=FP3_CHECKPOINT,
        default_method=default,
        settings=settings,
    )
    comparison = decision.metric_scope
    if comparison is None:
        raise ValueError("Both canonical candidates must be available for V2")
    count = decision.current_season_prior_event_count
    gates = {
        "checkpoint": (
            FP3_CHECKPOINT == settings.eligible_checkpoint,
            FP3_CHECKPOINT,
            settings.eligible_checkpoint,
        ),
        "candidate_available": (decision.season_aware_candidate_available, True, True),
        "current_season": (
            count >= settings.min_current_season_prior_events,
            count,
            settings.min_current_season_prior_events,
        ),
        "prior_folds": (
            decision.prior_candidate_folds >= settings.min_prior_candidate_folds,
            decision.prior_candidate_folds,
            settings.min_prior_candidate_folds,
        ),
        "prior_predictions": (
            decision.prior_candidate_predictions >= settings.min_prior_candidate_predictions,
            decision.prior_candidate_predictions,
            settings.min_prior_candidate_predictions,
        ),
        "aligned_folds": (
            comparison.prior_folds_used >= settings.min_prior_candidate_folds,
            comparison.prior_folds_used,
            settings.min_prior_candidate_folds,
        ),
        "aligned_rows": (
            comparison.prior_rows_used >= settings.min_prior_candidate_predictions,
            comparison.prior_rows_used,
            settings.min_prior_candidate_predictions,
        ),
        "finite_metrics": (
            comparison.improvement_sec is not None,
            comparison.improvement_sec is not None,
            True,
        ),
        "margin": (
            comparison.improvement_sec is not None
            and comparison.improvement_sec >= settings.improvement_margin_sec,
            comparison.improvement_sec,
            settings.improvement_margin_sec,
        ),
    }
    result = {
        "event_key": event_key,
        "season": event_season(event_key),
        "fold_id": step,
        "current_season_prior_events": count,
        "prior_candidate_folds": decision.prior_candidate_folds,
        "prior_candidate_predictions": decision.prior_candidate_predictions,
        "aligned_prior_folds": comparison.prior_folds_used,
        "aligned_prior_rows": comparison.prior_rows_used,
        "prior_uniform_mae": comparison.default_mae,
        "prior_weighted_mae": comparison.candidate_mae,
        "prior_improvement_sec": comparison.improvement_sec,
        "history_event_keys": json.dumps(comparison.prior_events_used),
        "dropped_candidate_rows": comparison.dropped_candidate_rows,
        "dropped_default_rows": comparison.dropped_default_rows,
        "history_eligible": decision.season_aware_candidate_eligible,
        "all_gates_pass": decision.selected_candidate,
        "selected_policy": WEIGHTED if decision.selected_candidate else UNIFORM,
        "selection_reason": decision.selection_reason,
        "history_scope_gate": True,
        "history_scope_reason": "strictly_prior_and_settled",
    }
    for name, (passed, observed, required) in gates.items():
        result[f"gate_{name}"] = bool(passed)
        result[f"gate_{name}_observed"] = observed
        result[f"gate_{name}_required"] = required
        result[f"gate_{name}_reason"] = "pass" if passed else "missing_or_below_requirement"
    if decision.selected_candidate != all(bool(g[0]) for g in gates.values()):
        raise AssertionError("Logged gates disagree with the canonical selector")
    return result


def candidate_frames(source: dict[str, Any], *, step: int) -> pd.DataFrame:
    """Create one targetless, canonical row per candidate and driver."""
    frames = []
    for name, policy in [("static", UNIFORM), ("weighted", WEIGHTED)]:
        frame = source[name].copy()
        if frame[list(TARGET_COLUMNS)].notna().any().any():
            raise ValueError("The fitting boundary exposed current qualifying outcomes")
        frame = frame[[*KEYS, "event", PRED]].copy()
        frame["candidate_family"] = "ablation"
        frame["model_name"] = "random_forest"
        frame["feature_group"] = "base_plus_relative"
        frame["temporal_weighting_policy"] = policy
        frame["fold_id"] = step
        frame[TARGET] = np.nan
        frames.append(frame)
    left, right = [set(f[KEYS].itertuples(index=False, name=None)) for f in frames]
    if left != right or any(f.duplicated(KEYS).any() for f in frames):
        raise ValueError("Both candidates must forecast identical unique target rows")
    result = pd.concat(frames, ignore_index=True)
    if not np.isfinite(result[PRED]).all():
        raise ValueError("Nonfinite candidate prediction")
    return result


def freeze_forecast(
    current: pd.DataFrame,
    history: pd.DataFrame,
    live_history: pd.DataFrame,
    *,
    gate: dict[str, Any],
    event_key: str,
    step: int,
) -> pd.DataFrame:
    """Freeze both shadows and the chosen live simulation before outcome access."""
    frames = []
    for policy, rows in current.groupby("temporal_weighting_policy", sort=False):
        prior = (
            history[history["temporal_weighting_policy"].eq(policy)] if len(history) else history
        )
        frame = add_replay_intervals(rows, prior, uncertainty="conformal_predicted_gap_bucket")
        frame["role"] = policy
        frames.append(frame)
    selected = current[current["temporal_weighting_policy"].eq(gate["selected_policy"])].copy()
    selected = add_replay_intervals(
        selected, live_history, uncertainty="conformal_predicted_gap_bucket"
    )
    selected["role"] = "selected"
    result = pd.concat([*frames, selected], ignore_index=True)
    result = result.drop(columns=[TARGET, "interval_contains_actual"])
    result["event_key"] = event_key
    result["creation_step"] = step
    result["evaluation_type"] = VERSION
    result["simulation_only"] = True
    result["diagnostic_only"] = result["role"].ne("selected")
    result["eligible_before_margin"] = gate["history_eligible"]
    result["all_promotion_gates_pass"] = gate["all_gates_pass"]
    return result


def settle_forecast(
    forecast: pd.DataFrame, outcomes: pd.DataFrame, *, expected_hash: str, step: int
) -> pd.DataFrame:
    """Reveal outcomes only for a frozen prediction; evidence starts next event."""
    if digest(records(forecast)) != expected_hash:
        raise ValueError("Forecast changed before settlement")
    if outcomes.duplicated(KEYS).any():
        raise ValueError("Duplicate settlement outcome key")
    settled = forecast.merge(outcomes[[*KEYS, TARGET]], on=KEYS, how="left", validate="many_to_one")
    settled["absolute_error_sec"] = (settled[PRED] - settled[TARGET]).abs()
    settled["evaluable"] = np.isfinite(settled[TARGET]) & np.isfinite(settled[PRED])
    settled["settlement_step"] = step
    settled["available_from_step"] = step + 1
    settled["forecast_hash"] = expected_hash
    low = pd.to_numeric(settled["prediction_interval_low_sec"], errors="coerce")
    high = pd.to_numeric(settled["prediction_interval_high_sec"], errors="coerce")
    settled["interval_contains_actual"] = settled[TARGET].between(low, high).where(low.notna())
    return settled


def fit_manifests(
    source: dict[str, Any],
    scope: pd.DataFrame,
    *,
    event_key: str,
    step: int,
    legal_events: list[str],
    event_order: list[str],
    model_config: ModelConfig,
) -> list[dict[str, Any]]:
    """Distinguish permitted input history from rows actually fitted after weighting."""
    train = scope[event_key_series(scope).isin(legal_events)]
    rows = []
    for policy, manifest in zip((UNIFORM, WEIGHTED), source["manifest"], strict=True):
        temporal = prepare_temporal_training_data(
            train,
            test_event=event_key,
            event_order=event_order,
            config=model_config.temporal_weighting,
            policy=policy,
        )
        actual = temporal.train[temporal.train.checkpoint.eq(FP3_CHECKPOINT)].dropna(
            subset=[TARGET]
        )
        actual_keys = set(event_key_series(actual))
        allowed = set(get_feature_columns_for_group(scope, "base_plus_relative"))
        features = [c for c in usable_checkpoint_features(actual, FP3_CHECKPOINT) if c in allowed]
        manifest = {k: v for k, v in manifest.items() if k != "fit_timestamp"}
        rows.append(
            {
                "event_key": event_key,
                "step": step,
                "policy": policy,
                "legal_history_cutoff": legal_events[-1],
                "legal_history_events": json.dumps(legal_events),
                "legal_history_event_count": len(legal_events),
                "actual_fit_events": json.dumps([e for e in legal_events if e in actual_keys]),
                "actual_fit_event_count": len(actual_keys),
                "actual_fit_rows": len(actual),
                "feature_columns": json.dumps(features),
                "feature_count": len(features),
                "training_data_fingerprint": frame_digest(actual),
                "training_weights_fingerprint": frame_digest(
                    temporal.sample_weights.loc[actual.index].rename("weight").to_frame()
                ),
                "current_season_prior_events": sum(
                    event_season(e) == event_season(event_key) for e in legal_events
                ),
                "model_config_fingerprint": digest(asdict(model_config)),
                "source_manifest": json.dumps(manifest, sort_keys=True),
            }
        )
    return rows


def run_replay_v2(
    dataset: pd.DataFrame,
    *,
    model_config: ModelConfig,
    feature_config: FeatureConfig | None,
    train_seasons: tuple[int, ...],
    test_season: int,
    output_dir: Path | None = None,
    min_train_events: int = 5,
) -> dict[str, pd.DataFrame]:
    """Freshly replay an independent split, including legal prior-season OOF warmup."""
    if any(s >= test_season for s in train_seasons):
        raise ValueError("Training seasons must be strictly earlier than evaluation season")
    order = ordered_event_keys(dataset)
    if dataset.duplicated(KEYS).any():
        raise ValueError("Duplicate dataset prediction key")
    if test_season not in set(dataset.season):
        raise ValueError("Evaluation season missing")
    first = next(e for e in order if event_season(e) == test_season)
    if len(prior_events_for(first, order, train_seasons, test_season)) < min_train_events:
        raise ValueError("Insufficient history before the evaluation season starts")
    history = pd.DataFrame()
    live_history = pd.DataFrame()
    forecasts, settlements, gates, manifests, stages = [], [], [], [], []
    for index, event_key in enumerate(order):
        if event_season(event_key) not in (*train_seasons, test_season):
            continue
        legal = prior_events_for(event_key, order, train_seasons, test_season)
        if len(legal) < min_train_events:
            continue
        step = index + 1
        LOGGER.info("%s split %s: %s, %s legal events", VERSION, test_season, event_key, len(legal))
        # This is the only input given to model fitting. Current labels are absent.
        scope = freeze_legal_information(dataset, event_key, legal)
        source = train_monitoring_event_sources(
            dataset=scope,
            row_keys=event_key_series(scope),
            event_order=order[:step],
            event_key=event_key,
            legal_train_events=legal,
            model_config=model_config,
            feature_config=feature_config,
            test_season=test_season,
        )
        if any(row["leakage_status"] != "valid" for row in source["leakage"]):
            raise ValueError("Source leakage audit failed")
        current = candidate_frames(source, step=step)
        current_features = scope[event_key_series(scope).eq(event_key)].drop(
            columns=list(TARGET_COLUMNS), errors="ignore"
        )
        fit_rows = fit_manifests(
            source,
            scope,
            event_key=event_key,
            step=step,
            legal_events=legal,
            event_order=order[:step],
            model_config=model_config,
        )
        common = {
            "version": VERSION,
            "test_season": test_season,
            "event_key": event_key,
            "step": step,
            "legal_information_fingerprint": frame_digest(scope),
            "targetless_input_fingerprint": frame_digest(current_features),
            "model_config_fingerprint": digest(asdict(model_config)),
        }
        event_dir = (
            output_dir / str(test_season) / event_key.replace("/", "_") if output_dir else None
        )
        shadow_payload = {
            **common,
            "stage": "shadow_generated",
            "fits": fit_rows,
            "predictions": records(current.drop(columns=TARGET)),
        }
        if event_dir:
            write_immutable(event_dir / "01_shadow.json", shadow_payload)
        gate = evaluate_gates(
            history,
            current,
            event_key=event_key,
            step=step,
            legal_events=legal,
            model_config=model_config,
        )
        forecast = freeze_forecast(
            current,
            history,
            live_history,
            gate=gate,
            event_key=event_key,
            step=step,
        )
        forecast_hash = digest(records(forecast))
        forecast_payload = {
            **common,
            "stage": "selected_forecast_frozen",
            "gate": gate,
            "forecast_hash": forecast_hash,
            "predictions": records(forecast),
        }
        if event_dir:
            write_immutable(event_dir / "02_forecast.json", forecast_payload)
        # The outcome table is deliberately accessed only after both files are frozen.
        outcomes = dataset[
            event_key_series(dataset).eq(event_key) & dataset.checkpoint.eq(FP3_CHECKPOINT)
        ][[*KEYS, TARGET]].copy()
        settled = settle_forecast(forecast, outcomes, expected_hash=forecast_hash, step=step)
        if event_dir:
            write_immutable(
                event_dir / "03_settlement.json",
                {
                    **common,
                    "stage": "outcome_revealed_and_settled",
                    "forecast_hash": forecast_hash,
                    "available_from_step": step + 1,
                    "settlements": records(settled),
                },
            )
        settled["test_season"] = test_season
        settled["is_evaluation_event"] = event_season(event_key) == test_season
        forecast["test_season"] = test_season
        gate["test_season"] = test_season
        gate["is_evaluation_event"] = event_season(event_key) == test_season
        gate["forecast_hash"] = forecast_hash
        for fit_row in fit_rows:
            fit_row["test_season"] = test_season
        for stage_index, name in enumerate(
            (
                "legal_information_frozen",
                "both_candidates_fitted",
                "shadows_frozen",
                "prior_gates_evaluated",
                "selected_forecast_frozen",
                "outcome_revealed",
                "settled_for_next_event",
            )
        ):
            stages.append(
                {
                    "test_season": test_season,
                    "event_key": event_key,
                    "step": step,
                    "stage_index": stage_index,
                    "stage": name,
                    "forecast_hash": forecast_hash,
                }
            )
        forecasts.append(forecast)
        settlements.append(settled)
        gates.append(gate)
        manifests.extend(fit_rows)
        valid = settled[settled.evaluable].copy()
        # No unsettled/current record enters either history until this point.
        history = pd.concat([history, valid[valid.role.ne("selected")]], ignore_index=True)
        live_history = pd.concat(
            [live_history, valid[valid.role.eq("selected")]], ignore_index=True
        )
    return {
        "forecasts": pd.concat(forecasts, ignore_index=True),
        "settlements": pd.concat(settlements, ignore_index=True),
        "gates": pd.DataFrame(gates),
        "training_manifest": pd.DataFrame(manifests),
        "lifecycle": pd.DataFrame(stages),
    }


def main() -> None:
    """Run all complete historical windows, with optional separate analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/prospective_replay_v2"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_data_config()
    source = args.dataset or build_combined_dataset_path(config.modeling_output_dir)
    dataset = pd.read_parquet(source)
    model, feature = load_model_config(), load_feature_config()
    seasons = sorted(int(v) for v in dataset.season.unique())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from f1_prediction.modeling.prospective_replay_v2_analysis import (
        analyze_replay,
        verify_protected_artifacts,
    )

    verify_protected_artifacts(args.output_dir)
    write_immutable(
        args.output_dir / "frozen_protocol.json",
        {
            "version": VERSION,
            "model_config": asdict(model),
            "feature_config": asdict(feature),
            "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "dataset_fingerprint": frame_digest(dataset),
            "seasons": seasons,
            "evidence_initialization": "empty ledger; freshly replay legal preseason OOF history",
            "min_train_events": 5,
            "uncertainty_numeric_rule": (
                "original replay helper: 0.90 quantile; minimum 20 residuals"
            ),
            "sensitivity": {
                "current_season": [4, 5, 6],
                "folds": [4, 5, 6],
                "predictions": [80, 100, 120],
                "margin": [0.025, 0.05, 0.075],
            },
        },
    )
    outputs: dict[str, list[pd.DataFrame]] = {}
    for season in seasons:
        earlier = tuple(s for s in seasons if s < season)
        if (
            dataset[dataset.season.isin(earlier)][["season", "event_slug"]]
            .drop_duplicates()
            .shape[0]
            < 5
        ):
            LOGGER.info("Excluded full-season window %s: insufficient preseason history", season)
            continue
        result = run_replay_v2(
            dataset,
            model_config=model,
            feature_config=feature,
            train_seasons=earlier,
            test_season=season,
            output_dir=args.output_dir / "events",
        )
        for name, frame in result.items():
            outputs.setdefault(name, []).append(frame)
    for name, frames in outputs.items():
        pd.concat(frames, ignore_index=True).to_csv(args.output_dir / f"{name}.csv", index=False)
    verify_protected_artifacts(args.output_dir)
    if args.analyze:
        analyze_replay(args.output_dir, model_config=model)


if __name__ == "__main__":
    main()
