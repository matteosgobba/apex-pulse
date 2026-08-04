from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.modeling.prospective_monitoring import forecast_snapshot_hash
from f1_prediction.production_state import (
    BOOTSTRAP_RECEIPT_RELATIVE_PATH,
    MANIFEST_NAME,
    ProductionStateConflictError,
    ProductionStateError,
    dashboard_fingerprint,
    export_production_state,
    import_production_state,
    inventory_production_state,
    sha256_file,
)
from f1_prediction.runtime import create_production_runtime_report

FIXED_CREATED_AT = "2026-08-03T12:00:00+00:00"


def test_inventory_is_deterministic_and_selects_dependency_closure(tmp_path: Path) -> None:
    root = _write_source_runtime(tmp_path / "source")

    first = inventory_production_state(root)
    second = inventory_production_state(root)

    assert first == second
    selected = {item.source_relative_path: item for item in first if item.included}
    assert "reports/metrics/prospective_monitoring_protocol.json" in selected
    assert "reports/metrics/prospective_monitoring_forecasts.parquet" in selected
    assert "data/processed/modeling/combined/modeling_dataset.parquet" in selected
    assert "data/processed/monitoring/2026/example-gp/monitoring_fp3_features.parquet" in selected
    assert "reports/dashboard/current_event.json" in selected
    assert "data/raw/fastf1_cache/2026/example/session_info.ff1pkl" in selected
    excluded = {item.source_relative_path: item for item in first if not item.included}
    assert excluded["models/ridge.joblib"].classification == "development_only"
    assert (
        excluded["data/raw/fastf1_cache/2025/old/response.ff1pkl"].classification
        == "reconstructable_optional"
    )
    assert excluded["reports/metrics/unrelated.csv"].classification == "unrelated_generated"


def test_export_manifest_checksums_are_correct_and_source_is_unchanged(
    tmp_path: Path,
) -> None:
    root = _write_source_runtime(tmp_path / "source")
    critical = _critical_source_paths(root)
    before = {path: path.read_bytes() for path in critical}

    result = export_production_state(
        root,
        tmp_path / "dist",
        created_at_utc=FIXED_CREATED_AT,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["bundle_schema_version"] == "1.0"
    assert manifest["file_count"] == len(manifest["files"])
    assert manifest["total_bytes"] == sum(item["size_bytes"] for item in manifest["files"])
    for item in manifest["files"]:
        source = root / item["source_relative_path"]
        assert sha256_file(source) == item["sha256"]
        assert source.stat().st_size == item["size_bytes"]
    assert {path: path.read_bytes() for path in critical} == before


def test_export_is_byte_deterministic_given_fixed_creation_time(tmp_path: Path) -> None:
    root = _write_source_runtime(tmp_path / "source")

    first = export_production_state(root, tmp_path / "one", created_at_utc=FIXED_CREATED_AT)
    second = export_production_state(root, tmp_path / "two", created_at_utc=FIXED_CREATED_AT)

    assert first.manifest_fingerprint == second.manifest_fingerprint
    assert first.bundle_fingerprint == second.bundle_fingerprint
    assert first.inventory_path.read_bytes() == second.inventory_path.read_bytes()


def test_dashboard_fingerprint_is_validated_deterministic_and_reports_parity(
    tmp_path: Path,
) -> None:
    root = _write_source_runtime(tmp_path / "source")

    first = dashboard_fingerprint(root / "reports/dashboard")
    second = dashboard_fingerprint(root / "reports/dashboard")

    assert first == second
    assert first["current_event"] == "Example Grand Prix"
    assert first["lifecycle_state"] == "settled_partial_coverage"
    assert first["forecast_row_count"] == 2
    assert first["settlement_comparison_row_count"] == 1
    assert first["forecast_coverage_status"] == "partial_coverage"


def test_fresh_import_and_idempotent_reimport_preserve_dashboard_parity(
    tmp_path: Path,
) -> None:
    source = _write_source_runtime(tmp_path / "source")
    exported = export_production_state(source, tmp_path / "dist", created_at_utc=FIXED_CREATED_AT)
    runtime = tmp_path / "runtime"

    first = import_production_state(exported.bundle_path, runtime)
    receipt_before = (runtime / BOOTSTRAP_RECEIPT_RELATIVE_PATH).read_bytes()
    second = import_production_state(exported.bundle_path, runtime)

    assert first.status == "imported"
    assert first.imported_count == first.file_count
    assert second.status == "already_present_identical"
    assert second.imported_count == 0
    assert second.already_present_identical_count == second.file_count
    assert (runtime / BOOTSTRAP_RECEIPT_RELATIVE_PATH).read_bytes() == receipt_before
    assert dashboard_fingerprint(runtime / "reports/dashboard") == dashboard_fingerprint(
        source / "reports/dashboard"
    )
    report = create_production_runtime_report(
        project_root=_write_project_shell(tmp_path / "application"),
        runtime_root=runtime,
        environ={"APEX_PULSE_DASHBOARD_DIR": str(runtime / "reports/dashboard")},
    )
    assert report.status == "seeded_monitoring_ready"
    assert report.ready_for_api is True
    assert report.ready_for_future_monitoring is True


def test_differing_critical_file_conflicts_without_overwrite_or_receipt(
    tmp_path: Path,
) -> None:
    source = _write_source_runtime(tmp_path / "source")
    exported = export_production_state(source, tmp_path / "dist", created_at_utc=FIXED_CREATED_AT)
    runtime = tmp_path / "runtime"
    conflict = runtime / "reports/metrics/prospective_monitoring_forecasts.parquet"
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"different-immutable-forecast")
    before = conflict.read_bytes()

    with pytest.raises(ProductionStateConflictError, match="status=conflict"):
        import_production_state(exported.bundle_path, runtime)

    assert conflict.read_bytes() == before
    assert not (runtime / BOOTSTRAP_RECEIPT_RELATIVE_PATH).exists()
    assert not (runtime / "reports/dashboard/current_event.json").exists()


@pytest.mark.parametrize("case", ["path_traversal", "undeclared_file", "symlink"])
def test_unsafe_or_undeclared_archive_members_are_rejected(tmp_path: Path, case: str) -> None:
    source = _write_source_runtime(tmp_path / "source")
    exported = export_production_state(source, tmp_path / "dist", created_at_utc=FIXED_CREATED_AT)
    malicious = tmp_path / f"{case}.tar.gz"
    _copy_archive_with_extra(exported.bundle_path, malicious, case)

    with pytest.raises(ProductionStateError):
        import_production_state(malicious, tmp_path / "runtime")

    assert not (tmp_path / "runtime" / BOOTSTRAP_RECEIPT_RELATIVE_PATH).exists()
    assert not (tmp_path / "escaped").exists()


def test_corrupted_archive_and_manifest_are_rejected_without_receipt(tmp_path: Path) -> None:
    source = _write_source_runtime(tmp_path / "source")
    exported = export_production_state(source, tmp_path / "dist", created_at_utc=FIXED_CREATED_AT)
    corrupted = tmp_path / "corrupted.tar.gz"
    corrupted.write_bytes(exported.bundle_path.read_bytes()[:100])
    bad_manifest = tmp_path / "bad-manifest.tar.gz"
    _rewrite_manifest(exported.bundle_path, bad_manifest, {"bundle_schema_version": "1.0"})

    with pytest.raises(ProductionStateError):
        import_production_state(corrupted, tmp_path / "corrupt-runtime")
    with pytest.raises(ProductionStateError, match="manifest schema"):
        import_production_state(bad_manifest, tmp_path / "manifest-runtime")
    assert not (tmp_path / "corrupt-runtime" / BOOTSTRAP_RECEIPT_RELATIVE_PATH).exists()
    assert not (tmp_path / "manifest-runtime" / BOOTSTRAP_RECEIPT_RELATIVE_PATH).exists()


def test_modified_seeded_state_is_reported_corrupt(tmp_path: Path) -> None:
    source = _write_source_runtime(tmp_path / "source")
    exported = export_production_state(source, tmp_path / "dist", created_at_utc=FIXED_CREATED_AT)
    runtime = tmp_path / "runtime"
    import_production_state(exported.bundle_path, runtime)
    (runtime / "reports/dashboard/current_event.json").write_text("{}\n", encoding="utf-8")

    report = create_production_runtime_report(
        project_root=_write_project_shell(tmp_path / "application"),
        runtime_root=runtime,
        environ={"APEX_PULSE_DASHBOARD_DIR": str(runtime / "reports/dashboard")},
    )

    assert report.status == "corrupt_or_conflicting"
    assert report.ready_for_api is False
    assert report.production_state_integrity_status == "failed"


def test_valid_future_event_append_preserves_seed_provenance_and_runtime_readiness(
    tmp_path: Path,
) -> None:
    source = _write_source_runtime(tmp_path / "source")
    exported = export_production_state(source, tmp_path / "dist", created_at_utc=FIXED_CREATED_AT)
    runtime = tmp_path / "runtime"
    import_production_state(exported.bundle_path, runtime)
    seed_protocol = (runtime / "reports/metrics/prospective_monitoring_protocol.json").read_bytes()
    seed_dataset = (
        runtime / "data/processed/modeling/combined/modeling_dataset.parquet"
    ).read_bytes()
    seed_forecast_rows = pd.read_parquet(
        runtime / "reports/metrics/prospective_monitoring_forecasts.parquet"
    )

    _append_valid_future_event(runtime)
    report = create_production_runtime_report(
        project_root=_write_project_shell(tmp_path / "application"),
        runtime_root=runtime,
        environ={"APEX_PULSE_DASHBOARD_DIR": str(runtime / "reports/dashboard")},
    )

    assert report.status == "seeded_monitoring_ready"
    assert report.bootstrap_seed_status == "verified"
    assert report.static_seed_invariants_status == "verified"
    assert report.live_operational_state_status == "verified"
    assert report.production_state_integrity_status == "verified_live_evolved"
    assert report.dashboard_bundle_status == "verified_current"
    assert report.forecast_artifact_status == "verified_current"
    assert report.settlement_artifact_status == "verified_current"
    assert (
        runtime / "reports/metrics/prospective_monitoring_protocol.json"
    ).read_bytes() == seed_protocol
    assert (
        runtime / "data/processed/modeling/combined/modeling_dataset.parquet"
    ).read_bytes() == seed_dataset
    current_forecasts = pd.read_parquet(
        runtime / "reports/metrics/prospective_monitoring_forecasts.parquet"
    )
    pd.testing.assert_frame_equal(
        current_forecasts[current_forecasts["forecast_id"].eq("forecast-example")].reset_index(
            drop=True
        ),
        seed_forecast_rows.reset_index(drop=True),
    )


def test_state_operations_do_not_call_monitoring_workflows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _write_source_runtime(tmp_path / "source")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("monitoring workflow was invoked")

    monkeypatch.setattr(
        "f1_prediction.modeling.monitoring_operations.run_monitoring_before_qualifying",
        forbidden,
    )
    monkeypatch.setattr(
        "f1_prediction.modeling.monitoring_operations.run_monitoring_after_qualifying",
        forbidden,
    )
    exported = export_production_state(source, tmp_path / "dist", created_at_utc=FIXED_CREATED_AT)
    import_production_state(exported.bundle_path, tmp_path / "runtime")
    dashboard_fingerprint(tmp_path / "runtime/reports/dashboard")


def _write_source_runtime(root: Path) -> Path:
    protocol = {
        "protocol_name": "test_protocol",
        "protocol_version": "1.0",
        "protocol_fingerprint": "abc123",
        "dataset_path": "data/processed/modeling/combined/modeling_dataset.parquet",
    }
    _write_json(root / "reports/metrics/prospective_monitoring_protocol.json", protocol)
    registry = pd.DataFrame(
        [
            {
                "protocol_name": "test_protocol",
                "monitor_season": 2026,
                "event_order": 1,
                "event_slug": "example-gp",
                "feature_artifact_path": (
                    "data/processed/monitoring/2026/example-gp/monitoring_fp3_features.parquet"
                ),
                "target_artifact_path": (
                    "data/processed/monitoring/2026/example-gp/"
                    "monitoring_qualifying_targets.parquet"
                ),
            }
        ]
    )
    registry_path = root / "reports/metrics/prospective_monitoring_event_registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(registry_path, index=False)
    forecasts, shadow, settlements, selection = _monitoring_tables(
        event_slug="example-gp", event_order=1, forecast_id="forecast-example"
    )
    forecasts.to_parquet(root / "reports/metrics/prospective_monitoring_forecasts.parquet")
    shadow.to_parquet(root / "reports/metrics/prospective_monitoring_shadow_candidates.parquet")
    settlements.to_parquet(root / "reports/metrics/prospective_monitoring_settlements.parquet")
    selection.to_csv(root / "reports/metrics/prospective_monitoring_selection_log.csv", index=False)
    dataset = root / "data/processed/modeling/combined/modeling_dataset.parquet"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"frozen-modeling-dataset")
    event_dir = root / "data/processed/monitoring/2026/example-gp"
    event_dir.mkdir(parents=True)
    for name, contents in {
        "monitoring_event_manifest.json": b'{"fingerprint":"event"}\n',
        "monitoring_fp3_features.parquet": b"features",
        "monitoring_qualifying_targets.parquet": b"targets",
        "monitoring_target_coverage.csv": b"driver,covered\nAAA,true\n",
    }.items():
        (event_dir / name).write_bytes(contents)
    raw_laps = root / "data/raw/laps/2026/example-gp/FP3_laps.parquet"
    raw_laps.parent.mkdir(parents=True)
    raw_laps.write_bytes(b"raw-laps")
    raw_metadata = root / "data/raw/session_metadata/2026/example-gp/FP3.json"
    _write_json(raw_metadata, {"event": "Example Grand Prix"})
    entry = root / "reports/metrics/qualifying_entry_lists/2026/example-gp/summary.json"
    _write_json(entry, {"status": "verified"})
    source_metric = root / "reports/metrics/backtest_report.json"
    _write_json(source_metric, {"status": "frozen"})
    cache = root / "data/raw/fastf1_cache/2026/example/session_info.ff1pkl"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"session-info")
    old_cache = root / "data/raw/fastf1_cache/2025/old/response.ff1pkl"
    old_cache.parent.mkdir(parents=True)
    old_cache.write_bytes(b"reconstructable")
    model = root / "models/ridge.joblib"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"unused-model")
    unrelated = root / "reports/metrics/unrelated.csv"
    unrelated.write_text("not,required\n", encoding="utf-8")
    _write_dashboard(root / "reports/dashboard", cache.relative_to(root).as_posix())
    return root.resolve()


def _write_dashboard(directory: Path, cache_relative: str) -> None:
    directory.mkdir(parents=True)
    artifacts = {
        "dashboard_manifest.json": {},
        "current_event.json": {
            "event_identity": {
                "event": "Example Grand Prix",
                "event_slug": "example-gp",
                "season": 2026,
            },
            "lifecycle": {"state": "settled_partial_coverage"},
            "forecast_status": {"forecast_coverage_status": "partial_coverage"},
            "settlement_status": {"forecast_coverage_status": "partial_coverage"},
        },
        "event_forecast.json": {"leaderboard": [{"driver": "AAA"}, {"driver": "BBB"}]},
        "event_settlement.json": {"driver_comparison": [{"driver": "AAA"}]},
        "event_practice_status.json": {},
        "historical_monitoring_summary.json": {},
        "model_summary.json": {},
    }
    for filename, data in artifacts.items():
        payload = {
            "schema_version": "1.0",
            "artifact_type": filename.removesuffix(".json"),
            "generated_at_utc": FIXED_CREATED_AT,
            "source_artifacts": ["reports/metrics/backtest_report.json"],
            "source_fingerprints": {
                "reports/metrics/backtest_report.json": "frozen",
                cache_relative: "cached",
            },
            "status": "complete",
            "data": data,
        }
        _write_json(directory / filename, payload)


def _monitoring_tables(
    *, event_slug: str, event_order: int, forecast_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common = {
        "protocol_name": "test_protocol",
        "protocol_fingerprint": "abc123",
        "forecast_id": forecast_id,
        "season": 2026,
        "event_slug": event_slug,
        "checkpoint": "after_fp3",
        "diagnostic_only": False,
        "family": "ablation",
        "model_name": "random_forest",
        "feature_group": "base_plus_relative",
        "temporal_weighting_policy": "uniform",
        "event_order": event_order,
    }
    forecasts = pd.DataFrame(
        [
            {
                **common,
                "driver": "AAA",
                "prediction_role": "observed_live_policy",
                "prediction_gap_sec": 0.1,
            },
            {
                **common,
                "driver": "BBB",
                "prediction_role": "observed_live_policy",
                "prediction_gap_sec": 0.2,
            },
        ]
    )
    shadow = pd.DataFrame(
        [
            {
                **common,
                "driver": "AAA",
                "prediction_role": "shadow_candidate",
                "diagnostic_only": True,
                "prediction_gap_sec": 0.15,
            }
        ]
    )
    snapshot = forecast_snapshot_hash(forecasts, shadow)
    selection = pd.DataFrame(
        [
            {
                "protocol_name": "test_protocol",
                "forecast_id": forecast_id,
                "event_slug": event_slug,
                "forecast_snapshot_hash": snapshot,
            }
        ]
    )
    settlements = pd.DataFrame(
        [
            {
                "protocol_name": "test_protocol",
                "protocol_fingerprint": "abc123",
                "forecast_id": forecast_id,
                "settlement_id": f"settlement-{event_slug}-AAA",
                "event_slug": event_slug,
                "driver": "AAA",
                "settlement_valid": True,
                "forecast_preexisted_settlement": True,
                "forecast_fingerprint_valid": True,
                "forecast_mutation_detected": False,
            }
        ]
    )
    return forecasts, shadow, settlements, selection


def _append_valid_future_event(root: Path) -> None:
    metrics = root / "reports/metrics"
    forecasts = pd.read_parquet(metrics / "prospective_monitoring_forecasts.parquet")
    shadow = pd.read_parquet(metrics / "prospective_monitoring_shadow_candidates.parquet")
    settlements = pd.read_parquet(metrics / "prospective_monitoring_settlements.parquet")
    selection = pd.read_csv(metrics / "prospective_monitoring_selection_log.csv")
    new_forecasts, new_shadow, new_settlements, new_selection = _monitoring_tables(
        event_slug="future-gp", event_order=2, forecast_id="forecast-future"
    )
    pd.concat([forecasts, new_forecasts], ignore_index=True).to_parquet(
        metrics / "prospective_monitoring_forecasts.parquet"
    )
    pd.concat([shadow, new_shadow], ignore_index=True).to_parquet(
        metrics / "prospective_monitoring_shadow_candidates.parquet"
    )
    pd.concat([settlements, new_settlements], ignore_index=True).to_parquet(
        metrics / "prospective_monitoring_settlements.parquet"
    )
    pd.concat([selection, new_selection], ignore_index=True).to_csv(
        metrics / "prospective_monitoring_selection_log.csv", index=False
    )
    registry_path = metrics / "prospective_monitoring_event_registry.csv"
    registry = pd.read_csv(registry_path)
    future = registry.iloc[-1].copy()
    future["event_slug"] = "future-gp"
    future["event_order"] = 2
    pd.concat([registry, future.to_frame().T], ignore_index=True).to_csv(registry_path, index=False)
    for path in (root / "reports/dashboard").glob("*.json"):
        payload = json.loads(path.read_text())
        payload["generated_at_utc"] = "2026-08-04T12:00:00+00:00"
        _write_json(path, payload)


def _write_project_shell(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname="test"\nversion="0"\n')
    config = root / "configs/data.yaml"
    config.parent.mkdir()
    config.write_text(
        "paths:\n"
        "  fastf1_cache_dir: data/raw/fastf1_cache\n"
        "  lap_output_dir: data/raw/laps\n"
        "  session_metadata_output_dir: data/raw/session_metadata\n"
        "  clean_lap_output_dir: data/interim/clean_laps\n"
        "  session_features_output_dir: data/processed/session_features\n"
        "  modeling_output_dir: data/processed/modeling\n"
        "  metrics_output_dir: reports/metrics\n"
        "fastf1:\n"
        "  load_telemetry: false\n"
        "  load_weather: false\n"
        "  load_messages: false\n",
        encoding="utf-8",
    )
    return root.resolve()


def _critical_source_paths(root: Path) -> list[Path]:
    return [
        root / "reports/metrics/prospective_monitoring_protocol.json",
        root / "reports/metrics/prospective_monitoring_event_registry.csv",
        root / "reports/metrics/prospective_monitoring_forecasts.parquet",
        root / "reports/metrics/prospective_monitoring_settlements.parquet",
        root / "reports/dashboard/current_event.json",
        root / "data/processed/modeling/combined/modeling_dataset.parquet",
    ]


def _copy_archive_with_extra(source: Path, destination: Path, case: str) -> None:
    with tarfile.open(source, "r:gz") as original, tarfile.open(destination, "w:gz") as output:
        for member in original.getmembers():
            handle = original.extractfile(member)
            output.addfile(member, handle)
        if case == "symlink":
            extra = tarfile.TarInfo("runtime/data/link")
            extra.type = tarfile.SYMTYPE
            extra.linkname = "/tmp/target"
            output.addfile(extra)
        else:
            name = "../escaped" if case == "path_traversal" else "runtime/data/undeclared"
            data = b"unsafe"
            extra = tarfile.TarInfo(name)
            extra.size = len(data)
            output.addfile(extra, io.BytesIO(data))


def _rewrite_manifest(source: Path, destination: Path, manifest: dict[str, object]) -> None:
    manifest_bytes = (json.dumps(manifest) + "\n").encode()
    with tarfile.open(source, "r:gz") as original, tarfile.open(destination, "w:gz") as output:
        for member in original.getmembers():
            if member.name == MANIFEST_NAME:
                replacement = tarfile.TarInfo(MANIFEST_NAME)
                replacement.size = len(manifest_bytes)
                output.addfile(replacement, io.BytesIO(manifest_bytes))
            else:
                output.addfile(member, original.extractfile(member))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
