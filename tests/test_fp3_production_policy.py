"""FP3 V1 correctness tests use synthetic outcomes, never future race outcomes."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_prospective_monitoring import _config, _init, _write_dataset, _write_target_artifact

from f1_prediction.config import load_data_config, load_feature_config, load_model_config
from f1_prediction.features.build import build_session_features_output_path
from f1_prediction.modeling import fp3_production_policy as policy
from f1_prediction.modeling import prospective_monitoring as monitoring
from f1_prediction.modeling.fp3_policy_report import aligned_evidence, build_report
from f1_prediction.modeling.prospective_replay import event_key_series

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "count,expected",
    [
        (0, policy.UNIFORM),
        (1, policy.UNIFORM),
        (2, policy.UNIFORM),
        (3, policy.UNIFORM),
        (4, policy.UNIFORM),
        (5, policy.SEASON_AWARE),
        (6, policy.SEASON_AWARE),
        (10, policy.SEASON_AWARE),
    ],
)
def test_frozen_transition(count, expected):
    assert policy.selection(count)["selected_candidate"] == expected
    assert policy.selection(count, mode="uniform_only")["selected_candidate"] == policy.UNIFORM


@pytest.fixture
def project(tmp_path, monkeypatch):
    config = _config(tmp_path)
    dataset_path = _write_dataset(config)
    _init(config, dataset_path)
    outcomes_path = tmp_path / "synthetic_outcomes.parquet"
    original = pd.read_parquet(dataset_path)
    original.to_parquet(outcomes_path, index=False)
    # Match production: monitoring events live in SEPARATE feature/target artifacts.
    original[original.season.lt(2026)].to_parquet(dataset_path, index=False)
    frozen = json.loads((ROOT / "configs" / policy.CONFIG_FILE).read_text())
    frozen["candidate_contract"]["historical_dataset_sha256"] = policy.file_hash(dataset_path)
    frozen["candidate_contract"]["function_hashes"] = {}  # Synthetic adapter below is injected.
    frozen["candidate_contract_fingerprint"] = policy.signature(frozen["candidate_contract"])
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / policy.CONFIG_FILE).write_text(json.dumps(frozen))

    # Minimal fixture has already-built numeric features. Adapter semantics are
    # tested separately against all 44 real historical events below.
    def features(config, row, spec):
        path = monitoring.resolve_registered_path(config, row["feature_artifact_path"])
        frame = pd.read_parquet(path)
        assert not any(c.startswith("quali_") for c in frame)
        return frame

    monkeypatch.setattr(policy, "registered_audited_features", features)
    monkeypatch.setattr(policy, "qualifying_deadline", lambda *args: "2099-01-01T00:00:00+00:00")
    return config, outcomes_path, frozen


def forecast(config, event="Bahrain", model=None):
    return monitoring.create_prospective_monitoring_forecast(
        config,
        model or load_model_config(),
        load_feature_config(),
        protocol_name="season_2026_v1",
        event=event,
    )


def rows(config):
    return pd.read_parquet(config.metrics_output_dir / "prospective_monitoring_forecasts.parquet")


def test_full_lifecycle_separate_history_idempotence_and_readonly_report(project, monkeypatch):
    config, outcomes, frozen = project
    assert forecast(config).status == "forecast_created"
    before = {
        str(p): policy.file_hash(p) for p in config.metrics_output_dir.rglob("*") if p.is_file()
    }
    data = rows(config)
    assert data.production_policy_version.eq(policy.POLICY_VERSION).all()
    assert data.prior_current_season_eligible_events.eq(0).all()
    assert data.actual_gap_sec.isna().all()
    assert data.absolute_error_sec.isna().all()
    assert set(data.prediction_role) == set(monitoring.FORECAST_ROLES)
    assert (
        data.groupby("prediction_role")
        .driver.apply(set)
        .apply(lambda x: x == set(data.driver))
        .all()
    )
    live = data[data.prediction_role.eq("observed_live_policy")]
    assert live.prediction_gap_sec.equals(live.uniform_prediction_gap_sec)
    assert forecast(config).status == "forecast_reused"
    assert before == {str(p): policy.file_hash(Path(p)) for p in before}
    with pytest.raises(ValueError, match="config conflict"):
        forecast(config, model=replace(load_model_config(), random_state=7))
    _write_target_artifact(config, outcomes, "Bahrain")
    monitoring.create_prospective_monitoring_settlement(
        config, protocol_name="season_2026_v1", event="Bahrain"
    )
    assert forecast(config).status == "forecast_reused"
    assert (
        policy.file_hash(config.metrics_output_dir / "prospective_monitoring_forecasts.parquet")
        == before[str(config.metrics_output_dir / "prospective_monitoring_forecasts.parquet")]
    )
    assert (
        monitoring.create_prospective_monitoring_settlement(
            config, protocol_name="season_2026_v1", event="Bahrain"
        ).status
        == "settlement_reused"
    )
    # Past settled outcomes now really enter the second model fit; they were absent
    # from the frozen 2023-only base dataset.
    forecast(config, "Monza")
    later = rows(config).query("event_slug == 'monza'")
    assert later.prior_current_season_eligible_events.eq(1).all()
    assert later.training_event_keys.str.contains("2026/bahrain").all()
    assert not later.training_event_keys.str.contains("2026/monza").any()
    before_report = {
        str(p): policy.file_hash(p) for p in config.metrics_output_dir.rglob("*") if p.is_file()
    }
    report = build_report(config, event="Monza")
    assert report["overall"]["events"] == 1
    assert report["overall"]["aligned_rows"] == 4
    assert before_report == {str(p): policy.file_hash(Path(p)) for p in before_report}
    monkeypatch.setenv("APEX_FP3_POLICY_MODE", "uniform_only")
    assert forecast(config, "Monza").status == "forecast_reused"
    assert later.production_policy_version.eq(policy.POLICY_VERSION).all()


def test_snapshot_and_config_mutation_block_regeneration(project):
    config, _, _ = project
    forecast(config)
    path = config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "prediction_gap_sec"] += 1
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="differs from frozen"):
        forecast(config)


def test_forecast_window_does_not_allow_historical_backfill(project, monkeypatch):
    config, _, frozen = project
    monkeypatch.setattr(policy, "qualifying_deadline", lambda *args: "2025-01-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="before qualifying"):
        forecast(config)
    with pytest.raises(ValueError):
        policy.ensure_forecast_window(
            frozen, "2025-01-01T00:00:00+00:00", "2099-01-01T00:00:00+00:00"
        )
    assert aligned_evidence(pd.DataFrame({"season": [2025]}), frozen)[0].empty


@pytest.mark.parametrize("prior_count", [0, 4, 5, 10])
def test_training_and_public_selection_with_frozen_candidates(prior_count, tmp_path):
    config = _config(tmp_path)
    path = _write_dataset(config)
    base = pd.read_parquet(path).query("season == 2023").copy()
    template = pd.read_parquet(path).query("event_slug == 'bahrain'")
    frames = [base]
    order = ["2023/australia", "2023/canada"]
    for index in range(prior_count + 2):
        frame = template.copy()
        frame["event_slug"] = f"event-{index}"
        frame["event"] = f"Event {index}"
        frame["event_order"] = index + 3
        frame["fp3_best_push_gap_to_session_best_sec"] += index * 0.01
        frames.append(frame)
        order.append(f"2026/event-{index}")
    dataset = pd.concat(frames, ignore_index=True)
    event = f"2026/event-{prior_count}"

    def fit(frame):
        return monitoring.train_monitoring_event_sources(
            dataset=frame,
            row_keys=event_key_series(frame),
            event_order=order,
            event_key=event,
            legal_train_events=order[: 2 + prior_count],
            model_config=load_model_config(),
            feature_config=load_feature_config(),
            test_season=2026,
            frozen_feature_columns=["fp3_best_push_gap_to_session_best_sec"],
        )

    source = fit(dataset)
    altered = dataset.copy()
    altered.loc[
        event_key_series(altered).isin(order[2 + prior_count :]), "quali_gap_to_pole_sec"
    ] = 999
    altered["current_residual"] = -999
    again = fit(altered)
    for key in ("static", "weighted"):
        pd.testing.assert_frame_equal(source[key], again[key])
    weighted = source["manifest"][1]
    used = json.loads(weighted["training_event_keys_used"])
    assert event not in used and order[-1] not in used
    if prior_count >= 5:
        assert len(used) == prior_count and all(x.startswith("2026/") for x in used)
    context = policy.selection(prior_count)
    assert context["selected_candidate"] == (
        policy.SEASON_AWARE if prior_count >= 5 else policy.UNIFORM
    )
    frame, _ = monitoring.monitoring_prediction_rows(
        source=source,
        protocol={
            "protocol_name": "test",
            "protocol_fingerprint": "test",
            "monitor_season": 2026,
            "train_seasons": [2023],
            "checkpoint": "after_fp3",
        },
        forecast_id="test",
        forecast_created="2026-10-01T00:00:00Z",
        event_key=event,
        event_order=order,
        event_order_lineage={"event_order": prior_count + 3},
        prior_monitoring=monitoring.prior_monitoring_evidence_summary(
            pd.DataFrame(), event_order_lineage_valid=True
        ),
        preflight={},
        candidate_eligible=False,
        selection_reason="test",
    )
    spec = json.loads((ROOT / "configs" / policy.CONFIG_FILE).read_text())
    decorated = policy.decorate_predictions(frame, context, spec)
    live = decorated.query("prediction_role == 'observed_live_policy'")
    shadow_column = (
        "season_aware_prediction_gap_sec" if prior_count >= 5 else "uniform_prediction_gap_sec"
    )
    assert live.prediction_gap_sec.equals(live[shadow_column])
    assert decorated.query("diagnostic_only").live_policy_selected.eq(False).all()
    from f1_prediction.dashboard.export import _live_rows

    public = _live_rows(decorated)
    assert set(public.prediction_role) == {"observed_live_policy"}
    unchanged = public.prediction_gap_sec.copy()
    unselected = (
        "uniform_default_shadow" if prior_count >= 5 else "season_aware_weighted_candidate_shadow"
    )
    frame.loc[frame.prediction_role.eq(unselected), "prediction_gap_sec"] += 1000
    changed_shadow = policy.decorate_predictions(frame, context, spec)
    pd.testing.assert_series_equal(_live_rows(changed_shadow).prediction_gap_sec, unchanged)


def test_weighted_failure_falls_back_and_uniform_failure_blocks(project, monkeypatch):
    config, _, _ = project
    original = monitoring.fit_monitoring_source_candidate

    def fail_weighted(**kwargs):
        if kwargs["temporal_policy"].value == "current_season_only_with_prior":
            raise ValueError("synthetic missing prerequisite")
        return original(**kwargs)

    monkeypatch.setattr(monitoring, "fit_monitoring_source_candidate", fail_weighted)
    real_dataset = policy.legal_dataset

    def established(*args, **kwargs):
        data, context = real_dataset(*args, **kwargs)
        context.update(policy.selection(5))
        return data, context

    monkeypatch.setattr(policy, "legal_dataset", established)
    forecast(config)
    data = rows(config)
    assert data.intended_candidate.eq(policy.SEASON_AWARE).all()
    assert data.selected_candidate.eq(policy.UNIFORM).all()
    assert data.fallback_reason.str.contains("synthetic missing prerequisite").all()
    assert (
        data.query("prediction_role == 'season_aware_weighted_candidate_shadow'")
        .prediction_gap_sec.isna()
        .all()
    )
    monkeypatch.setattr(
        monitoring,
        "fit_monitoring_source_candidate",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("Uniform failed")),
    )
    with pytest.raises(ValueError, match="Uniform failed"):
        forecast(config, "Monza")


@pytest.mark.skipif(
    not (ROOT / "data/processed/modeling/combined/modeling_dataset.parquet").is_file(),
    reason="Canonical historical dataset is a local, untracked audit input",
)
def test_audited_adapter_matches_all_canonical_rows_and_excludes_nine_new_features():
    config = load_data_config(project_root=ROOT)
    spec = policy.load_policy(config)
    columns = spec["candidate_contract"]["feature_columns"]
    assert len(columns) == 126
    canonical = pd.read_parquet(config.modeling_output_dir / "combined/modeling_dataset.parquet")
    for (season, event), old in canonical.query("checkpoint == 'after_fp3'").groupby(
        ["season", "event"]
    ):
        practice_path = build_session_features_output_path(
            config.session_features_output_dir, int(season), event
        )
        if not practice_path.is_file():
            pytest.skip("Canonical practice aggregates are local, untracked audit inputs")
        practice = pd.read_parquet(practice_path)
        new = (
            policy.audited_feature_rows(practice, columns)
            .set_index("driver_key")
            .reindex(old.driver_key)
        )
        np.testing.assert_allclose(
            old[columns].to_numpy(dtype=float), new[columns].to_numpy(dtype=float), equal_nan=True
        )
    assert not any(
        "participation" in c or "observed_driver_count" in c or "evidence_from_other" in c
        for c in columns
    )


def test_prior_count_uses_only_settled_earlier_eligible_rows(project, monkeypatch):
    config, outcomes, frozen = project
    protocol = monitoring.load_protocol(config.metrics_output_dir, "season_2026_v1")
    registry = monitoring.read_csv(
        config.metrics_output_dir / "prospective_monitoring_event_registry.csv"
    )
    target = registry.query("event_slug == 'monza'").iloc[0]
    original = pd.read_parquet(outcomes)
    base = original.query("season == 2023")
    for settled_time, expected_count in [("2099-01-01T00:00:00Z", 0), ("2026-09-20T09:00:00Z", 1)]:
        fake = []
        for slug in ["bahrain", "monza"]:
            for driver in ["VER", "NOR", "LEC", "HAM"]:
                fake.append(
                    dict(
                        protocol_name="season_2026_v1",
                        event_slug=slug,
                        driver_key=driver,
                        actual_gap_sec=0.2,
                        settlement_evaluable=True,
                        settled_at_utc=settled_time,
                    )
                )
        pd.DataFrame(fake).to_parquet(
            config.metrics_output_dir / "prospective_monitoring_settlements.parquet"
        )
        # Even an over-permissive eligibility source cannot make current/future
        # events or not-yet-available outcomes enter legal training/counts.
        monkeypatch.setattr(
            monitoring, "settled_event_keys", lambda *a, **k: {"2026/bahrain", "2026/monza"}
        )
        data, state = policy.legal_dataset(
            config,
            protocol,
            registry,
            target,
            base,
            frozen,
            forecast_time="2026-09-20T10:00:00+00:00",
        )
        assert state["prior_current_season_eligible_events"] == expected_count
        current = data.query("event_slug == 'monza'")
        assert current.quali_gap_to_pole_sec.isna().all()
        assert "2026/monza" not in json.loads(state["training_cutoff"])


@pytest.mark.skipif(
    not all(
        (ROOT / "reports/metrics" / f"ablation_{name}_predictions.parquet").is_file()
        for name in ("uniform", "current_season_only_with_prior")
    ),
    reason="Canonical prediction artifacts are local, untracked audit inputs",
)
def test_real_reference_metrics_are_unchanged_and_exactly_aligned():
    frames = []
    for temporal in ["uniform", "current_season_only_with_prior"]:
        data = pd.read_parquet(
            ROOT / "reports/metrics" / f"ablation_{temporal}_predictions.parquet"
        )
        data = data.query(
            "model_name == 'random_forest' and feature_group == 'base_plus_relative' "
            "and checkpoint == 'after_fp3' and season in [2024,2025]"
        )
        frames.append(data)
    keys = ["season", "event_slug", "checkpoint", "driver"]
    aligned = frames[0].merge(frames[1], on=keys, validate="one_to_one", suffixes=("_u", "_s"))
    assert len(aligned) == 595
    np.testing.assert_array_equal(aligned.quali_gap_to_pole_sec_u, aligned.quali_gap_to_pole_sec_s)
    for season, count, uniform, weighted in [
        (2024, 238, 0.9459500048586944, 0.7164713729299674),
        (2025, 357, 0.7882732422310238, 0.5275664993017083),
    ]:
        rows = aligned[aligned.season.eq(season)]
        assert len(rows) == count
        assert (
            rows.predicted_quali_gap_to_pole_sec_u - rows.quali_gap_to_pole_sec_u
        ).abs().mean() == pytest.approx(uniform, abs=1e-12)
        assert (
            rows.predicted_quali_gap_to_pole_sec_s - rows.quali_gap_to_pole_sec_s
        ).abs().mean() == pytest.approx(weighted, abs=1e-12)


def test_snapshot_survives_append_to_legacy_nullable_schema(project):
    config, _, _ = project
    forecast(config)
    path = config.metrics_output_dir / "prospective_monitoring_forecasts.parquet"
    actual = pd.read_parquet(path)
    legacy = actual.drop(columns=policy.METADATA_COLUMNS).copy()
    legacy["event_slug"] = "legacy-event"
    legacy["forecast_id"] = "legacy-id"
    pd.concat([legacy, actual], ignore_index=True, sort=False).to_parquet(path, index=False)
    assert forecast(config).status == "forecast_reused"


def test_existing_policy_bypasses_diagnostic_selector(project, monkeypatch):
    config, _, _ = project
    monkeypatch.setattr(
        monitoring,
        "season_aware_decision",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("selector called")),
    )
    forecast(config)


def test_freeze_cannot_be_retimed_or_removed_after_first_forecast(project):
    config, _, frozen = project
    forecast(config)
    path = config.project_root / "configs" / policy.CONFIG_FILE
    frozen["freeze_at_utc"] = "2026-01-01T00:00:00Z"
    path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="policy/config conflict"):
        forecast(config, "Monza")
    path.unlink()
    with pytest.raises(ValueError, match="Missing configuration"):
        forecast(config, "Monza")
