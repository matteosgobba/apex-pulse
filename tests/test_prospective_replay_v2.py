"""Causality and isolation tests for the separately versioned FP3 experiment."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from f1_prediction.config import load_model_config
from f1_prediction.features.qualifying_targets import TARGET_COLUMNS
from f1_prediction.modeling import prospective_replay_v2 as replay
from f1_prediction.modeling.prospective_policy_evaluation import build_frozen_policy_profiles
from f1_prediction.modeling.prospective_replay import POLICY_PROFILES, run_true_replay
from f1_prediction.modeling.prospective_replay_v2_analysis import (
    align_settlements,
    bootstrap_events,
    verify_protected_artifacts,
)
from f1_prediction.modeling.splits import ordered_event_keys


@pytest.fixture
def model():
    config = load_model_config()
    # Small synthetic test forests only; production run loads the untouched YAML.
    return replace(config, random_forest=replace(config.random_forest, n_estimators=3))


@pytest.fixture
def dataset():
    rows = []
    for season, count in [(2023, 6), (2024, 7)]:
        for event in range(count):
            for driver in range(20):
                for checkpoint in ("after_fp1", "after_fp2", "after_fp3"):
                    rows.append(
                        {
                            "season": season,
                            "event_slug": f"event-{event}",
                            "event": f"Event {event}",
                            "event_order": event + 1,
                            "checkpoint": checkpoint,
                            "driver": f"D{driver}",
                            "team": f"Team{driver // 2}",
                            "quali_gap_to_pole_sec": driver / 10 + event / 40,
                            "quali_position": driver + 1,
                            "quali_best_lap_time_sec": 80 + driver / 10,
                            "reached_q2": int(driver < 15),
                            "reached_q3": int(driver < 10),
                            "fp1_best_valid_lap_time_sec": 80 + driver / 8,
                            "fp2_best_valid_lap_time_sec": 80 + driver / 9,
                            "fp3_best_valid_lap_time_sec": 80 + driver / 10,
                            "fp3_best_push_gap_to_session_best_sec": driver / 10,
                        }
                    )
    return pd.DataFrame(rows)


def run(dataset, model, output=None):
    return replay.run_replay_v2(
        dataset,
        model_config=model,
        feature_config=None,
        train_seasons=(2023,),
        test_season=2024,
        output_dir=output,
    )


def test_future_rows_and_current_outcomes_never_reach_fitter(dataset, model, monkeypatch):
    real = replay.train_monitoring_event_sources
    observed = []

    def inspect(**kwargs):
        scope = kwargs["dataset"]
        key = kwargs["event_key"]
        events = set(replay.event_key_series(scope))
        assert events == set([*kwargs["legal_train_events"], key])
        current = scope[replay.event_key_series(scope).eq(key)]
        assert current[list(TARGET_COLUMNS)].isna().all().all()
        assert kwargs["event_order"][-1] == key
        observed.append(key)
        return real(**kwargs)

    monkeypatch.setattr(replay, "train_monitoring_event_sources", inspect)
    result = run(dataset, model)
    assert len(observed) == 8
    state = result["gates"]
    assert state.prior_candidate_folds.tolist() == list(range(8))
    assert state.prior_candidate_predictions.tolist() == [20 * i for i in range(8)]
    assert state.aligned_prior_rows.equals(state.prior_candidate_predictions)


def test_outcome_and_future_perturbations_leave_current_forecast_and_gates_identical(
    dataset, model
):
    initial = run(dataset, model)
    changed = dataset.copy()
    target = (changed.season == 2024) & (changed.event_order >= 6)
    changed.loc[target, list(TARGET_COLUMNS)] = 9999
    changed.loc[
        (changed.season == 2024) & (changed.event_order > 6), "fp3_best_valid_lap_time_sec"
    ] = -9999
    rerun = run(changed, model)
    for table in ("forecasts", "gates"):
        left = initial[table]
        right = rerun[table]
        # event-5 has changed actuals but the same legal pre-event information.
        mask = left.event_key.ne("2024/event-6")
        pd.testing.assert_frame_equal(
            left[mask].reset_index(drop=True), right[mask].reset_index(drop=True)
        )


def test_prediction_precedes_settlement_and_missing_targets_do_not_filter_forecast(
    dataset,
    model,
    monkeypatch,
    tmp_path,
):
    current = (
        (dataset.season == 2024) & (dataset.event_slug == "event-0") & (dataset.driver == "D0")
    )
    dataset.loc[current, replay.TARGET] = np.nan
    real = replay.settle_forecast

    def inspect(forecast, outcomes, **kwargs):
        key = forecast.event_key.iloc[0]
        folder = tmp_path / "2024" / key.replace("/", "_")
        shadow = json.loads((folder / "01_shadow.json").read_text())
        selected = json.loads((folder / "02_forecast.json").read_text())
        assert not (folder / "03_settlement.json").exists()
        assert all(replay.TARGET not in row for row in shadow["predictions"])
        assert all(replay.TARGET not in row for row in selected["predictions"])
        return real(forecast, outcomes, **kwargs)

    monkeypatch.setattr(replay, "settle_forecast", inspect)
    result = run(dataset, model, tmp_path)
    forecast = result["forecasts"]
    assert forecast[forecast.event_key.eq("2024/event-0")].groupby("role").size().eq(20).all()
    settled = result["settlements"]
    assert settled.available_from_step.eq(settled.settlement_step + 1).all()
    assert (
        settled[settled.event_key.eq("2024/event-0")].groupby("role").evaluable.sum().eq(19).all()
    )
    aligned = align_settlements(settled)
    assert len(aligned[aligned.event_key.eq("2024/event-0")]) == 19


def test_rerun_is_identical_and_snapshots_immutable(dataset, model, tmp_path):
    first = run(dataset, model, tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    second = run(dataset, model, tmp_path)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
    assert all(p.read_bytes() == value for p, value in before.items())
    sample = next(iter(before))
    with pytest.raises(ValueError, match="Immutable"):
        replay.write_immutable(sample, {"modified": True})
    forecast = first["forecasts"].iloc[:1].drop(columns="test_season").copy()
    expected = replay.digest(replay.records(forecast))
    forecast.loc[forecast.index[0], replay.PRED] += 1
    with pytest.raises(ValueError, match="changed before settlement"):
        replay.settle_forecast(forecast, pd.DataFrame(), expected_hash=expected, step=1)


def controlled_rows(step, *, settled):
    rows = []
    for policy in (replay.UNIFORM, replay.WEIGHTED):
        for driver in range(20):
            rows.append(
                {
                    "season": 2024,
                    "event_slug": f"e{step}",
                    "checkpoint": "after_fp3",
                    "driver": f"D{driver}",
                    "fold_id": step,
                    "candidate_family": "ablation",
                    "model_name": "random_forest",
                    "feature_group": "base_plus_relative",
                    "temporal_weighting_policy": policy,
                    replay.TARGET: 0.0 if settled else np.nan,
                    replay.PRED: 1.0 if policy == replay.UNIFORM else 0.8,
                    "available_from_step": step + 1,
                }
            )
    return pd.DataFrame(rows)


def test_gate_uses_only_prior_settled_evidence_and_exact_frozen_thresholds():
    model = load_model_config()
    history = pd.concat([controlled_rows(i, settled=True) for i in range(1, 6)])
    current = controlled_rows(6, settled=False)
    kwargs = dict(
        event_key="2024/e6",
        step=6,
        legal_events=[f"2024/e{i}" for i in range(1, 6)],
        model_config=model,
    )
    gate = replay.evaluate_gates(history, current, **kwargs)
    assert gate["selected_policy"] == replay.WEIGHTED
    assert gate["prior_candidate_folds"] == 5
    assert gate["aligned_prior_rows"] == 100
    assert gate["gate_margin_required"] == 0.05
    assert gate["prior_improvement_sec"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="Current/future"):
        replay.evaluate_gates(
            pd.concat([history, controlled_rows(6, settled=True)]), current, **kwargs
        )
    late = history.copy()
    late["available_from_step"] = 7
    with pytest.raises(ValueError, match="not settled/available"):
        replay.evaluate_gates(late, current, **kwargs)
    leaked = current.copy()
    leaked[replay.TARGET] = 1
    with pytest.raises(ValueError, match="hidden"):
        replay.evaluate_gates(history, leaked, **kwargs)
    # Drop a single candidate row without dropping all identical index labels.
    missing = history.reset_index(drop=True).iloc[:-1]
    gate = replay.evaluate_gates(missing, current, **kwargs)
    assert not gate["all_gates_pass"]
    assert not gate["gate_prior_predictions"]
    assert not gate["gate_aligned_rows"]
    assert gate["gate_margin"]


def test_existing_original_replay_still_reproduces_zero_selection(dataset, model):
    profiles = build_frozen_policy_profiles(
        model, profile_names=POLICY_PROFILES, uncertainty="conformal_predicted_gap_bucket"
    )
    kwargs = dict(
        dataset=dataset,
        event_order=ordered_event_keys(dataset),
        profiles=profiles,
        model_config=model,
        feature_config=None,
        train_seasons=(2023,),
        test_season=2024,
        min_train_events=5,
        uncertainty="conformal_predicted_gap_bucket",
    )
    first, second = run_true_replay(**kwargs), run_true_replay(**kwargs)
    pd.testing.assert_frame_equal(first["predictions"], second["predictions"])
    assert not first["selection"].season_aware_selected.any()
    assert not first["shadow"].empty


def test_artifact_preservation_verifier_detects_mutation(tmp_path):
    import hashlib

    old = tmp_path / "production.json"
    old.write_text("original")
    (tmp_path / "protected_artifacts_before.json").write_text(
        json.dumps(
            {
                str(old): hashlib.sha256(old.read_bytes()).hexdigest(),
            }
        )
    )
    verify_protected_artifacts(tmp_path)
    old.write_text("modified")
    with pytest.raises(ValueError, match="Protected artifacts changed"):
        verify_protected_artifacts(tmp_path)


def test_diagnostic_directory_cannot_redirect_production_live_policy(monkeypatch, tmp_path):
    from f1_prediction.modeling import prospective_monitoring as production

    # Even when candidate gates pass, the production role still receives static rows.
    seen = []

    def role(frame, **kwargs):
        seen.append((kwargs["role"], frame.copy(), kwargs["selection_is_live"]))
        return pd.DataFrame({"diagnostic_only": [kwargs["diagnostic_only"]]})

    monkeypatch.setattr(production, "role_frame", role)
    monkeypatch.setattr(production, "forecast_columns", lambda: ["diagnostic_only"])
    monkeypatch.setattr(
        production,
        "build_shadow_candidate_rows",
        lambda **kwargs: pd.DataFrame(
            {
                "shadow_role": ["uniform_default", "season_aware_weighted_candidate"],
            }
        ),
    )
    (tmp_path / "prospective_replay_v2").mkdir()
    (tmp_path / "prospective_replay_v2" / "summary.json").write_text(
        '{"selected_policy":"weighted"}'
    )
    monkeypatch.chdir(tmp_path)
    source = {
        "static": pd.DataFrame({"sentinel": [1]}),
        "weighted": pd.DataFrame({"sentinel": [2]}),
    }
    production.monitoring_prediction_rows(
        source=source,
        protocol={"train_seasons": [2023], "monitor_season": 2024},
        forecast_id="x",
        forecast_created="x",
        event_key="2024/a",
        event_order=["2024/a"],
        event_order_lineage={},
        prior_monitoring={},
        preflight={},
        candidate_eligible=True,
        selection_reason="selected_after_prior_evidence",
    )
    assert seen[0][0] == "observed_live_policy"
    assert seen[0][1].sentinel.tolist() == [1]
    assert seen[0][2] is True
    assert all(not item[2] for item in seen[1:])


def test_event_bootstrap_keeps_weekend_clusters_and_is_deterministic():
    events = pd.DataFrame(
        {"test_season": [2024, 2024], "rows": [1, 20], "d_uniform_minus_weighted": [-1.0, 1.0]}
    )
    result = bootstrap_events(events)
    assert result == bootstrap_events(events)
    assert result["event_mean"] == 0
    assert result["row_mae_difference"] == pytest.approx(19 / 21)
    assert result["diagnostic_only"]


def test_reject_future_training_season_and_incomplete_season(dataset, model):
    with pytest.raises(ValueError, match="strictly earlier"):
        replay.run_replay_v2(
            dataset,
            model_config=model,
            feature_config=None,
            train_seasons=(2025,),
            test_season=2024,
        )
    with pytest.raises(ValueError, match="Insufficient history"):
        replay.run_replay_v2(
            dataset, model_config=model, feature_config=None, train_seasons=(), test_season=2023
        )
