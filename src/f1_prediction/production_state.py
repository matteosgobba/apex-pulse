"""Deterministic export, safe import, and parity checks for production state."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from f1_prediction.dashboard.schema import validate_dashboard_artifact_file
from f1_prediction.dashboard_api.service import DASHBOARD_ARTIFACTS
from f1_prediction.utils.paths import get_project_root

BUNDLE_SCHEMA_VERSION = "1.0"
MANIFEST_NAME = "production_state_manifest.json"
INVENTORY_NAME = "production_state_inventory.csv"
SUMMARY_NAME = "production_state_summary.json"
BUNDLE_NAME = "apex-pulse-production-state.tar.gz"
STORED_MANIFEST_RELATIVE_PATH = Path("reports/metrics/production_state_manifest.json")
BOOTSTRAP_RECEIPT_RELATIVE_PATH = Path("reports/metrics/production_bootstrap_receipt.json")
RUNTIME_TREES = ("data", "reports", "models")

CORE_MONITORING_FILES = {
    "prospective_monitoring_protocol.json",
    "prospective_monitoring_event_registry.csv",
    "prospective_monitoring_forecasts.parquet",
    "prospective_monitoring_shadow_candidates.parquet",
    "prospective_monitoring_selection_log.csv",
    "prospective_monitoring_training_manifest.csv",
    "prospective_monitoring_forecast_integrity_audit.csv",
    "prospective_monitoring_settlements.parquet",
    "prospective_monitoring_event_metrics.csv",
    "prospective_monitoring_shadow_evidence_ledger.csv",
    "prospective_monitoring_settlement_integrity_audit.csv",
    "prospective_monitoring_integrity_summary.json",
    "prospective_monitoring_integrity_by_event.csv",
    "prospective_monitoring_integrity_failures.csv",
    "prospective_monitoring_event_order_reconciliation.csv",
    "prospective_monitoring_event_order_integrity_summary.json",
    "prospective_monitoring_event_order_integrity_by_event.csv",
    "prospective_monitoring_event_order_integrity_failures.csv",
}
CRITICAL_MONITORING_FILES = {
    "prospective_monitoring_protocol.json",
    "prospective_monitoring_event_registry.csv",
    "prospective_monitoring_forecasts.parquet",
    "prospective_monitoring_settlements.parquet",
    "prospective_monitoring_shadow_candidates.parquet",
}

# These files are immutable at the row/event level, but the canonical workflows append
# future events to them. The bootstrap hashes remain provenance for the seed snapshot;
# they are not a permanent whole-file checksum for live operation.
LIVE_EVOLVING_PATHS = {
    "reports/metrics/prospective_monitoring_event_registry.csv",
    "reports/metrics/prospective_monitoring_forecasts.parquet",
    "reports/metrics/prospective_monitoring_shadow_candidates.parquet",
    "reports/metrics/prospective_monitoring_selection_log.csv",
    "reports/metrics/prospective_monitoring_training_manifest.csv",
    "reports/metrics/prospective_monitoring_forecast_integrity_audit.csv",
    "reports/metrics/prospective_monitoring_settlements.parquet",
    "reports/metrics/prospective_monitoring_event_metrics.csv",
    "reports/metrics/prospective_monitoring_shadow_evidence_ledger.csv",
    "reports/metrics/prospective_monitoring_settlement_integrity_audit.csv",
    "reports/metrics/prospective_monitoring_integrity_summary.json",
    "reports/metrics/prospective_monitoring_integrity_by_event.csv",
    "reports/metrics/prospective_monitoring_integrity_failures.csv",
    "reports/metrics/prospective_monitoring_event_order_reconciliation.csv",
    "reports/metrics/prospective_monitoring_event_order_integrity_summary.json",
    "reports/metrics/prospective_monitoring_event_order_integrity_by_event.csv",
    "reports/metrics/prospective_monitoring_event_order_integrity_failures.csv",
}
LIVE_EVOLVING_PREFIXES = ("reports/dashboard/",)


class ProductionStateError(RuntimeError):
    """Raised when state export or bootstrap cannot be completed safely."""


class ProductionStateConflictError(ProductionStateError):
    """Raised before import when an existing runtime file differs."""


@dataclass(frozen=True)
class InventoryRecord:
    """One generated file and its production-state disposition."""

    source_relative_path: str
    destination_relative_path: str
    classification: str
    included: bool
    immutable: bool
    required_for_dashboard: bool
    required_for_future_forecast: bool
    required_for_future_settlement: bool
    size_bytes: int
    sha256: str
    reason: str


@dataclass(frozen=True)
class ProductionStateExportResult:
    """Files and counters produced by one export."""

    bundle_path: Path
    manifest_path: Path
    inventory_path: Path
    summary_path: Path
    bundle_fingerprint: str
    manifest_fingerprint: str
    file_count: int
    total_bytes: int
    dashboard_fingerprint: str


@dataclass(frozen=True)
class ProductionStateImportResult:
    """Safe bootstrap outcome."""

    status: str
    receipt_path: Path
    bundle_fingerprint: str
    manifest_fingerprint: str
    file_count: int
    imported_count: int
    already_present_identical_count: int


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one file without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dashboard_fingerprint(dashboard_dir: Path) -> dict[str, Any]:
    """Validate dashboard JSON and return byte-level deterministic parity facts."""
    files: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for artifact_type, filename in sorted(DASHBOARD_ARTIFACTS.items()):
        path = dashboard_dir / filename
        if not path.is_file():
            raise ProductionStateError(f"Dashboard artifact is missing: {path}")
        payloads[artifact_type] = validate_dashboard_artifact_file(path)
        files.append(
            {
                "artifact_type": artifact_type,
                "filename": filename,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    aggregate = hashlib.sha256(
        b"".join(f"{item['filename']}\0{item['sha256']}\n".encode() for item in files)
    ).hexdigest()
    current = payloads["current_event"].get("data", {})
    event = current.get("event_identity", {})
    lifecycle = current.get("lifecycle", {})
    forecast = payloads["event_forecast"].get("data", {})
    settlement = payloads["event_settlement"].get("data", {})
    forecast_rows = _first_list_length(forecast, "leaderboard", "predictions", "drivers")
    comparison_rows = _first_list_length(settlement, "driver_comparison", "comparison", "drivers")
    forecast_status = current.get("forecast_status", {})
    settlement_status = current.get("settlement_status", {})
    return {
        "dashboard_schema_version": payloads["current_event"]["schema_version"],
        "dashboard_fingerprint": aggregate,
        "files": files,
        "current_event": event.get("event"),
        "current_event_slug": event.get("event_slug"),
        "current_season": event.get("season"),
        "lifecycle_state": lifecycle.get("state"),
        "forecast_row_count": forecast_rows,
        "settlement_comparison_row_count": comparison_rows,
        "forecast_coverage_status": forecast_status.get("forecast_coverage_status"),
        "settlement_coverage_status": settlement_status.get("forecast_coverage_status"),
    }


def inventory_production_state(project_root: Path | None = None) -> list[InventoryRecord]:
    """Classify every generated runtime file and select the portable dependency closure."""
    root = (project_root or get_project_root()).resolve()
    selections = _build_selection(root)
    records: list[InventoryRecord] = []
    for tree in RUNTIME_TREES:
        tree_path = root / tree
        if not tree_path.is_dir():
            continue
        for path in sorted(candidate for candidate in tree_path.rglob("*") if candidate.is_file()):
            relative = path.relative_to(root).as_posix()
            disposition = selections.get(relative) or _excluded_disposition(relative)
            records.append(
                InventoryRecord(
                    source_relative_path=relative,
                    destination_relative_path=relative,
                    size_bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                    **disposition,
                )
            )
    missing = sorted(set(selections) - {item.source_relative_path for item in records})
    if missing:
        raise ProductionStateError(
            "Required production-state files are missing: " + ", ".join(missing)
        )
    return records


def export_production_state(
    project_root: Path | None = None,
    output_dir: Path | None = None,
    *,
    created_at_utc: str | None = None,
) -> ProductionStateExportResult:
    """Create a checksummed portable bundle without mutating source artifacts."""
    root = (project_root or get_project_root()).resolve()
    destination = (output_dir or root / "dist/production-state").resolve()
    records = inventory_production_state(root)
    included = [record for record in records if record.included]
    dashboard = dashboard_fingerprint(root / "reports/dashboard")
    protocol = json.loads(
        (root / "reports/metrics/prospective_monitoring_protocol.json").read_text(encoding="utf-8")
    )
    created = created_at_utc or datetime.now(UTC).isoformat()
    manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "created_at_utc": created,
        "source_runtime_mode": "local",
        "source_project_root": ".",
        "file_count": len(included),
        "total_bytes": sum(record.size_bytes for record in included),
        "protocol_name": protocol.get("protocol_name"),
        "protocol_fingerprint": protocol.get("protocol_fingerprint"),
        "current_dashboard_event": dashboard["current_event"],
        "current_dashboard_lifecycle": dashboard["lifecycle_state"],
        "dashboard_fingerprint": dashboard["dashboard_fingerprint"],
        "fastf1_cache_bundle_policy": "referenced_dashboard_session_info_only",
        "files": [_manifest_file(record) for record in included],
    }
    manifest_bytes = _json_bytes(manifest)
    manifest_fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / MANIFEST_NAME
    inventory_path = destination / INVENTORY_NAME
    summary_path = destination / SUMMARY_NAME
    bundle_path = destination / BUNDLE_NAME
    _atomic_write(manifest_path, manifest_bytes)
    _write_inventory(inventory_path, records)
    fastf1 = [
        record for record in records if "data/raw/fastf1_cache/" in record.source_relative_path
    ]
    summary = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "created_at_utc": created,
        "bundle_path": BUNDLE_NAME,
        "manifest_fingerprint": manifest_fingerprint,
        "file_count": len(included),
        "total_bytes": manifest["total_bytes"],
        "dashboard_fingerprint": dashboard["dashboard_fingerprint"],
        "classification_counts": _classification_counts(records),
        "included_classification_counts": _classification_counts(included),
        "runtime_tree_total_sizes": {
            tree: sum(
                record.size_bytes
                for record in records
                if record.source_relative_path.startswith(f"{tree}/")
            )
            for tree in RUNTIME_TREES
        },
        "fastf1_cache_bundle_policy": manifest["fastf1_cache_bundle_policy"],
        "fastf1_cache_total_size": sum(record.size_bytes for record in fastf1),
        "fastf1_cache_included_size": sum(
            record.size_bytes for record in fastf1 if record.included
        ),
        "fastf1_cache_excluded_reconstructable_size": sum(
            record.size_bytes for record in fastf1 if not record.included
        ),
    }
    _atomic_write(summary_path, _json_bytes(summary))
    _write_deterministic_archive(bundle_path, root, included, manifest_bytes)
    bundle_fingerprint = sha256_file(bundle_path)
    summary["bundle_fingerprint"] = bundle_fingerprint
    summary["bundle_compressed_bytes"] = bundle_path.stat().st_size
    _atomic_write(summary_path, _json_bytes(summary))
    return ProductionStateExportResult(
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        inventory_path=inventory_path,
        summary_path=summary_path,
        bundle_fingerprint=bundle_fingerprint,
        manifest_fingerprint=manifest_fingerprint,
        file_count=len(included),
        total_bytes=int(manifest["total_bytes"]),
        dashboard_fingerprint=str(dashboard["dashboard_fingerprint"]),
    )


def import_production_state(bundle_path: Path, runtime_root: Path) -> ProductionStateImportResult:
    """Verify and import one bundle, rejecting every differing destination file."""
    bundle = bundle_path.resolve()
    root = runtime_root.expanduser().resolve()
    if not bundle.is_file():
        raise ProductionStateError(f"Bundle does not exist: {bundle}")
    bundle_fingerprint = sha256_file(bundle)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="apex-pulse-bootstrap-", dir=root.parent) as name:
        staging = Path(name)
        manifest_bytes, staged_files = _verify_archive_to_staging(bundle, staging)
        manifest = json.loads(manifest_bytes)
        manifest_fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
        bootstrap_status = (
            "seeded_monitoring_ready"
            if _manifest_monitoring_ready(manifest)
            else "seeded_api_ready"
        )
        conflicts: list[str] = []
        identical: list[str] = []
        missing: list[str] = []
        for item in manifest["files"]:
            relative = _validated_relative_path(item["destination_relative_path"])
            destination = root.joinpath(*relative.parts)
            if _has_symlink_component(root, relative):
                conflicts.append(relative.as_posix())
            elif destination.exists() and not destination.is_file():
                conflicts.append(relative.as_posix())
            elif destination.is_file():
                if sha256_file(destination) == item["sha256"]:
                    identical.append(relative.as_posix())
                else:
                    conflicts.append(relative.as_posix())
            else:
                missing.append(relative.as_posix())
        if conflicts:
            raise ProductionStateConflictError(
                "status=conflict; refusing to overwrite differing runtime files: "
                + ", ".join(conflicts)
            )

        stored_manifest = root / STORED_MANIFEST_RELATIVE_PATH
        if stored_manifest.is_file() and stored_manifest.read_bytes() != manifest_bytes:
            raise ProductionStateConflictError(
                "status=conflict; stored production manifest differs"
            )
        receipt_path = root / BOOTSTRAP_RECEIPT_RELATIVE_PATH
        existing_receipt: dict[str, Any] | None = None
        if receipt_path.is_file():
            try:
                existing_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ProductionStateConflictError(
                    "status=conflict; existing bootstrap receipt is invalid"
                ) from exc
            if (
                existing_receipt.get("bundle_fingerprint") != bundle_fingerprint
                or existing_receipt.get("manifest_fingerprint") != manifest_fingerprint
                or existing_receipt.get("status") != bootstrap_status
            ):
                raise ProductionStateConflictError(
                    "status=conflict; existing bootstrap receipt identifies different state"
                )

        created_files: list[Path] = []
        try:
            root.mkdir(parents=True, exist_ok=True)
            for relative_text in missing:
                relative = PurePosixPath(relative_text)
                source = staged_files[relative_text]
                destination = root.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temp_path = destination.with_name(f".{destination.name}.bootstrap.tmp")
                shutil.copyfile(source, temp_path)
                os.replace(temp_path, destination)
                created_files.append(destination)
            for item in manifest["files"]:
                relative = _validated_relative_path(item["destination_relative_path"])
                destination = root.joinpath(*relative.parts)
                if sha256_file(destination) != item["sha256"]:
                    raise ProductionStateError(
                        f"Destination checksum verification failed: {relative.as_posix()}"
                    )
            if not stored_manifest.exists():
                stored_manifest.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write(stored_manifest, manifest_bytes)
                created_files.append(stored_manifest)
            dashboard = dashboard_fingerprint(root / "reports/dashboard")
            receipt = {
                "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
                "bundle_fingerprint": bundle_fingerprint,
                "manifest_fingerprint": manifest_fingerprint,
                "imported_at_utc": datetime.now(UTC).isoformat(),
                "runtime_root": str(root),
                "file_count": len(manifest["files"]),
                "imported_count": len(missing),
                "already_present_identical_count": len(identical),
                "critical_artifact_verification": {
                    item["destination_relative_path"]: item["sha256"]
                    for item in manifest["files"]
                    if item["immutable"]
                },
                "protocol_fingerprint": manifest.get("protocol_fingerprint"),
                "dashboard_current_event": dashboard["current_event"],
                "dashboard_lifecycle": dashboard["lifecycle_state"],
                "dashboard_fingerprint": dashboard["dashboard_fingerprint"],
                "status": bootstrap_status,
            }
            if existing_receipt is None:
                _atomic_write(receipt_path, _json_bytes(receipt))
        except Exception:
            for path in reversed(created_files):
                if path.is_file():
                    path.unlink()
            raise
    return ProductionStateImportResult(
        status=("already_present_identical" if not missing else "imported"),
        receipt_path=root / BOOTSTRAP_RECEIPT_RELATIVE_PATH,
        bundle_fingerprint=bundle_fingerprint,
        manifest_fingerprint=manifest_fingerprint,
        file_count=len(manifest["files"]),
        imported_count=len(missing),
        already_present_identical_count=len(identical),
    )


def verify_seeded_production_state(runtime_root: Path) -> dict[str, Any]:
    """Verify bootstrap provenance, static invariants, and current live state separately."""
    root = runtime_root.resolve()
    receipt_path = root / BOOTSTRAP_RECEIPT_RELATIVE_PATH
    manifest_path = root / STORED_MANIFEST_RELATIVE_PATH
    base = {
        "bootstrap_receipt_present": receipt_path.is_file(),
        "bootstrap_status": "not_seeded",
        "bundle_fingerprint": None,
        "manifest_fingerprint": None,
        "production_state_file_count": 0,
        "production_state_integrity_status": "not_seeded",
        "bootstrap_seed_status": "not_seeded",
        "static_seed_invariants_status": "not_seeded",
        "live_operational_state_status": "not_seeded",
        "critical_artifact_integrity_status": "not_seeded",
        "protocol_fingerprint_status": "not_seeded",
        "dashboard_bundle_status": "not_seeded",
        "forecast_artifact_status": "not_seeded",
        "settlement_artifact_status": "not_seeded",
        "modeling_dataset_status": "not_seeded",
        "ready_for_api": False,
        "ready_for_future_monitoring": False,
    }
    if not receipt_path.exists() and not manifest_path.exists():
        return base
    if not receipt_path.is_file() or not manifest_path.is_file():
        base.update(
            {
                "bootstrap_status": "corrupt_or_conflicting",
                "production_state_integrity_status": "failed",
                "bootstrap_seed_status": "failed",
                "static_seed_invariants_status": "failed",
                "live_operational_state_status": "failed",
                "critical_artifact_integrity_status": "failed",
            }
        )
        return base
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        _validate_manifest(manifest)
        manifest_fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
        static_failures: list[str] = []
        live_changed: list[str] = []
        for item in manifest["files"]:
            relative = _validated_relative_path(item["destination_relative_path"])
            path = root.joinpath(*relative.parts)
            matches = path.is_file() and sha256_file(path) == item["sha256"]
            if not matches:
                relative_text = relative.as_posix()
                if _is_live_evolving_path(relative_text):
                    live_changed.append(relative_text)
                else:
                    static_failures.append(relative_text)
        dashboard = dashboard_fingerprint(root / "reports/dashboard")
        protocol_path = root / "reports/metrics/prospective_monitoring_protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol_ok = protocol.get("protocol_fingerprint") == manifest.get("protocol_fingerprint")
        dashboard_matches_seed = dashboard["dashboard_fingerprint"] == manifest.get(
            "dashboard_fingerprint"
        )
        receipt_status = receipt.get("status")
        receipt_ok = receipt.get(
            "manifest_fingerprint"
        ) == manifest_fingerprint and receipt_status in {
            "seeded_api_ready",
            "seeded_monitoring_ready",
        }
        live_ok = _verify_live_operational_state(root, manifest)
        static_ok = not static_failures and protocol_ok
        ready_api = static_ok and live_ok and receipt_ok
        ready_monitoring = ready_api and receipt_status == "seeded_monitoring_ready"
        base.update(
            {
                "bootstrap_status": receipt.get("status", "invalid"),
                "bundle_fingerprint": receipt.get("bundle_fingerprint"),
                "manifest_fingerprint": manifest_fingerprint,
                "production_state_file_count": len(manifest["files"]),
                "production_state_integrity_status": (
                    ("verified_live_evolved" if live_changed else "verified")
                    if static_ok and live_ok and receipt_ok
                    else "failed"
                ),
                "bootstrap_seed_status": "verified" if receipt_ok else "failed",
                "static_seed_invariants_status": "verified" if static_ok else "failed",
                "live_operational_state_status": "verified" if live_ok else "failed",
                "critical_artifact_integrity_status": (
                    "verified" if static_ok and live_ok else "failed"
                ),
                "protocol_fingerprint_status": "verified" if protocol_ok else "failed",
                "dashboard_bundle_status": (
                    "verified_seed" if dashboard_matches_seed else "verified_current"
                ),
                "forecast_artifact_status": _live_manifest_path_status(
                    root,
                    manifest,
                    "reports/metrics/prospective_monitoring_forecasts.parquet",
                    live_ok,
                ),
                "settlement_artifact_status": _live_manifest_path_status(
                    root,
                    manifest,
                    "reports/metrics/prospective_monitoring_settlements.parquet",
                    live_ok,
                ),
                "modeling_dataset_status": _manifest_path_status(
                    root,
                    manifest,
                    "data/processed/modeling/combined/modeling_dataset.parquet",
                ),
                "ready_for_api": ready_api,
                "ready_for_future_monitoring": ready_monitoring,
            }
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, ProductionStateError):
        base.update(
            {
                "bootstrap_status": "corrupt_or_conflicting",
                "production_state_integrity_status": "failed",
                "bootstrap_seed_status": "failed",
                "static_seed_invariants_status": "failed",
                "live_operational_state_status": "failed",
                "critical_artifact_integrity_status": "failed",
                "ready_for_api": False,
                "ready_for_future_monitoring": False,
            }
        )
    return base


def _build_selection(root: Path) -> dict[str, dict[str, Any]]:
    selections: dict[str, dict[str, Any]] = {}

    def add(
        relative: str,
        classification: str,
        reason: str,
        *,
        immutable: bool = False,
        dashboard: bool = False,
        forecast: bool = False,
        settlement: bool = False,
    ) -> None:
        normalized = _validated_relative_path(relative).as_posix()
        existing = selections.get(normalized)
        value = {
            "classification": classification,
            "included": True,
            "immutable": immutable,
            "required_for_dashboard": dashboard,
            "required_for_future_forecast": forecast,
            "required_for_future_settlement": settlement,
            "reason": reason,
        }
        if existing:
            value["immutable"] = existing["immutable"] or immutable
            value["required_for_dashboard"] = existing["required_for_dashboard"] or dashboard
            value["required_for_future_forecast"] = (
                existing["required_for_future_forecast"] or forecast
            )
            value["required_for_future_settlement"] = (
                existing["required_for_future_settlement"] or settlement
            )
            if existing["classification"] == "authoritative_required":
                value["classification"] = existing["classification"]
        selections[normalized] = value

    metrics_dir = root / "reports/metrics"
    for filename in sorted(CORE_MONITORING_FILES):
        if (metrics_dir / filename).is_file():
            add(
                f"reports/metrics/{filename}",
                "authoritative_required",
                "Canonical frozen prospective-monitoring state.",
                immutable=filename in CRITICAL_MONITORING_FILES,
                dashboard=True,
                forecast=True,
                settlement=True,
            )

    protocol_path = metrics_dir / "prospective_monitoring_protocol.json"
    registry_path = metrics_dir / "prospective_monitoring_event_registry.csv"
    if not protocol_path.is_file() or not registry_path.is_file():
        raise ProductionStateError("Protocol and registry are required for production export")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    dataset_path = str(protocol.get("dataset_path", ""))
    add(
        dataset_path,
        "operational_required",
        "Frozen protocol dataset used to fit future monitored-event forecasts.",
        immutable=True,
        forecast=True,
        settlement=True,
    )
    registry = pd.read_csv(registry_path)
    for row in registry.to_dict(orient="records"):
        season = int(row["monitor_season"])
        slug = str(row["event_slug"])
        event_dir = f"data/processed/monitoring/{season}/{slug}"
        for filename in (
            "monitoring_event_manifest.json",
            "monitoring_fp3_features.parquet",
            "monitoring_qualifying_targets.parquet",
            "monitoring_target_coverage.csv",
        ):
            relative = f"{event_dir}/{filename}"
            if (root / relative).is_file():
                add(
                    relative,
                    "authoritative_required",
                    "Registered per-event monitoring source-of-truth artifact.",
                    immutable=True,
                    dashboard=filename != "monitoring_fp3_features.parquet",
                    forecast=True,
                    settlement=True,
                )
        for base in ("data/raw/laps", "data/raw/session_metadata"):
            directory = root / base / str(season) / slug
            if directory.is_dir():
                raw_files = sorted(
                    candidate for candidate in directory.rglob("*") if candidate.is_file()
                )
                for path in raw_files:
                    add(
                        path.relative_to(root).as_posix(),
                        "operational_required",
                        "Raw registered-event provenance required for identity validation.",
                        settlement=True,
                    )
        entry_dir = metrics_dir / "qualifying_entry_lists" / str(season) / slug
        if entry_dir.is_dir():
            entry_files = sorted(
                candidate for candidate in entry_dir.rglob("*") if candidate.is_file()
            )
            for path in entry_files:
                add(
                    path.relative_to(root).as_posix(),
                    "operational_required",
                    "Preserved pre-qualifying entry-list evidence.",
                    immutable=True,
                    dashboard=True,
                    forecast=True,
                )

    dashboard_dir = root / "reports/dashboard"
    dashboard_sources: set[str] = set()
    for filename in sorted(DASHBOARD_ARTIFACTS.values()):
        relative = f"reports/dashboard/{filename}"
        add(
            relative,
            "dashboard_only",
            "Exact validated public dashboard snapshot.",
            immutable=True,
            dashboard=True,
        )
        payload = validate_dashboard_artifact_file(dashboard_dir / filename)
        dashboard_sources.update(str(item) for item in payload.get("source_artifacts", []))
        dashboard_sources.update(str(item) for item in payload.get("source_fingerprints", {}))
    for relative in sorted(dashboard_sources):
        if (root / relative).is_file():
            add(
                relative,
                "dashboard_only",
                "Existing source-fingerprint closure for the copied dashboard snapshot.",
                dashboard=True,
            )
    return selections


def _excluded_disposition(relative: str) -> dict[str, Any]:
    if relative.startswith("data/raw/fastf1_cache/"):
        classification = "reconstructable_optional"
        reason = "FastF1 response cache is network-reconstructable and not authoritative."
    elif relative.startswith(("data/interim/", "data/processed/session_features/")):
        classification = "reconstructable_optional"
        reason = "Intermediate feature material can be rebuilt from preserved raw inputs."
    elif relative.startswith("models/") or relative.startswith("reports/figures/"):
        classification = "development_only"
        reason = "Development model/figure output is not loaded by the prospective workflow."
    elif relative.endswith((".DS_Store", ".Rhistory", ".gitkeep")):
        classification = "unrelated_generated"
        reason = "Filesystem or repository placeholder metadata is not runtime state."
    else:
        classification = "unrelated_generated"
        reason = "Generated artifact is outside the inspected production dependency closure."
    return {
        "classification": classification,
        "included": False,
        "immutable": False,
        "required_for_dashboard": False,
        "required_for_future_forecast": False,
        "required_for_future_settlement": False,
        "reason": reason,
    }


def _manifest_file(record: InventoryRecord) -> dict[str, Any]:
    value = asdict(record)
    value.pop("included")
    value.pop("reason")
    return value


def _write_inventory(path: Path, records: list[InventoryRecord]) -> None:
    buffer = io.StringIO(newline="")
    fieldnames = list(asdict(records[0])) if records else list(InventoryRecord.__annotations__)
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(asdict(record))
    _atomic_write(path, buffer.getvalue().encode())


def _write_deterministic_archive(
    path: Path,
    root: Path,
    records: list[InventoryRecord],
    manifest_bytes: bytes,
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                _add_bytes_to_tar(archive, MANIFEST_NAME, manifest_bytes)
                for record in sorted(records, key=lambda item: item.destination_relative_path):
                    data = (root / record.source_relative_path).read_bytes()
                    _add_bytes_to_tar(
                        archive,
                        f"runtime/{record.destination_relative_path}",
                        data,
                    )
    os.replace(temporary, path)


def _add_bytes_to_tar(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def _verify_archive_to_staging(bundle: Path, staging: Path) -> tuple[bytes, dict[str, Path]]:
    try:
        with tarfile.open(bundle, mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise ProductionStateError("Archive contains duplicate members")
            for member in members:
                _validated_archive_member(member)
            manifest_member = next(
                (member for member in members if member.name == MANIFEST_NAME), None
            )
            if manifest_member is None:
                raise ProductionStateError("Archive manifest is missing")
            manifest_handle = archive.extractfile(manifest_member)
            if manifest_handle is None:
                raise ProductionStateError("Archive manifest cannot be read")
            manifest_bytes = manifest_handle.read()
            manifest = json.loads(manifest_bytes)
            _validate_manifest(manifest)
            declared = {
                f"runtime/{item['destination_relative_path']}": item for item in manifest["files"]
            }
            actual = {name for name in names if name != MANIFEST_NAME}
            if actual != set(declared):
                extra = sorted(actual - set(declared))
                missing = sorted(set(declared) - actual)
                raise ProductionStateError(
                    f"Archive file declaration mismatch; extra={extra}, missing={missing}"
                )
            staged_files: dict[str, Path] = {}
            for name, item in declared.items():
                member = archive.getmember(name)
                handle = archive.extractfile(member)
                if handle is None:
                    raise ProductionStateError(f"Archive member cannot be read: {name}")
                relative = _validated_relative_path(item["destination_relative_path"])
                destination = staging.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with destination.open("wb") as output:
                    while chunk := handle.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                if size != item["size_bytes"] or digest.hexdigest() != item["sha256"]:
                    raise ProductionStateError(
                        f"Archive checksum verification failed: {relative.as_posix()}"
                    )
                staged_files[relative.as_posix()] = destination
    except (tarfile.TarError, EOFError, OSError, json.JSONDecodeError) as exc:
        raise ProductionStateError(f"Invalid or corrupted production-state bundle: {exc}") from exc
    return manifest_bytes, staged_files


def _validate_manifest(manifest: Any) -> None:
    required = {
        "bundle_schema_version",
        "created_at_utc",
        "source_runtime_mode",
        "source_project_root",
        "file_count",
        "total_bytes",
        "protocol_name",
        "protocol_fingerprint",
        "current_dashboard_event",
        "current_dashboard_lifecycle",
        "dashboard_fingerprint",
        "files",
    }
    if not isinstance(manifest, dict) or required - set(manifest):
        raise ProductionStateError("Production-state manifest schema is incomplete")
    if manifest["bundle_schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise ProductionStateError("Unsupported production-state bundle schema")
    if not isinstance(manifest["files"], list):
        raise ProductionStateError("Manifest files must be a list")
    seen: set[str] = set()
    total = 0
    for item in manifest["files"]:
        required_file = {
            "source_relative_path",
            "destination_relative_path",
            "classification",
            "immutable",
            "required_for_dashboard",
            "required_for_future_forecast",
            "required_for_future_settlement",
            "size_bytes",
            "sha256",
        }
        if not isinstance(item, dict) or required_file - set(item):
            raise ProductionStateError("Manifest file entry schema is incomplete")
        relative = _validated_relative_path(item["destination_relative_path"]).as_posix()
        if relative in seen:
            raise ProductionStateError("Manifest contains duplicate destinations")
        seen.add(relative)
        if not isinstance(item["size_bytes"], int) or item["size_bytes"] < 0:
            raise ProductionStateError("Manifest file size is invalid")
        if not isinstance(item["sha256"], str) or len(item["sha256"]) != 64:
            raise ProductionStateError("Manifest SHA-256 is invalid")
        total += item["size_bytes"]
    if manifest["file_count"] != len(manifest["files"]) or manifest["total_bytes"] != total:
        raise ProductionStateError("Manifest aggregate counts are inconsistent")


def _validated_archive_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if member.name.startswith("/") or ".." in path.parts or not member.isfile():
        raise ProductionStateError(f"Unsafe or unsupported archive member: {member.name}")
    if member.name != MANIFEST_NAME and (not path.parts or path.parts[0] != "runtime"):
        raise ProductionStateError(f"Unexpected archive member location: {member.name}")


def _validated_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ProductionStateError("Runtime destination must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.parts[0] not in RUNTIME_TREES:
        raise ProductionStateError(f"Unsafe runtime destination path: {value}")
    return path


def _manifest_path_status(root: Path, manifest: dict[str, Any], relative: str) -> str:
    item = next(
        (entry for entry in manifest["files"] if entry["destination_relative_path"] == relative),
        None,
    )
    if item is None:
        return "not_declared"
    path = root / relative
    return "verified" if path.is_file() and sha256_file(path) == item["sha256"] else "failed"


def _live_manifest_path_status(
    root: Path,
    manifest: dict[str, Any],
    relative: str,
    live_ok: bool,
) -> str:
    """Report seed parity separately from a valid current append-only ledger."""
    seed_status = _manifest_path_status(root, manifest, relative)
    if seed_status == "verified":
        return "verified_seed"
    if seed_status == "not_declared":
        return seed_status
    return "verified_current" if live_ok else "failed"


def _is_live_evolving_path(relative: str) -> bool:
    return relative in LIVE_EVOLVING_PATHS or relative.startswith(LIVE_EVOLVING_PREFIXES)


def _verify_live_operational_state(root: Path, manifest: dict[str, Any]) -> bool:
    """Validate current append-only ledgers after they diverge from their seed hashes."""
    live_items = [
        item
        for item in manifest["files"]
        if _is_live_evolving_path(item["destination_relative_path"])
    ]
    if all(
        (root / item["destination_relative_path"]).is_file()
        and sha256_file(root / item["destination_relative_path"]) == item["sha256"]
        for item in live_items
    ):
        return True
    try:
        dashboard_fingerprint(root / "reports/dashboard")
        metrics = root / "reports/metrics"
        registry = pd.read_csv(metrics / "prospective_monitoring_event_registry.csv")
        forecasts = pd.read_parquet(metrics / "prospective_monitoring_forecasts.parquet")
        shadow = pd.read_parquet(metrics / "prospective_monitoring_shadow_candidates.parquet")
        settlements = pd.read_parquet(metrics / "prospective_monitoring_settlements.parquet")
        selection = pd.read_csv(metrics / "prospective_monitoring_selection_log.csv")
    except (OSError, ValueError, KeyError):
        return False

    registry_key = {"protocol_name", "monitor_season", "event_slug", "event_order"}
    if registry_key - set(registry) or registry.duplicated(list(registry_key)).any():
        return False
    order_view = registry.loc[:, list(registry_key)].drop_duplicates()
    if order_view.duplicated(["protocol_name", "monitor_season", "event_order"]).any():
        return False

    forecast_required = {
        "protocol_name",
        "protocol_fingerprint",
        "forecast_id",
        "event_slug",
        "driver",
    }
    settlement_required = {
        "protocol_name",
        "protocol_fingerprint",
        "forecast_id",
        "settlement_id",
        "event_slug",
        "driver",
        "settlement_valid",
        "forecast_preexisted_settlement",
        "forecast_fingerprint_valid",
        "forecast_mutation_detected",
    }
    selection_required = {
        "protocol_name",
        "forecast_id",
        "event_slug",
        "forecast_snapshot_hash",
    }
    if (
        forecast_required - set(forecasts)
        or settlement_required - set(settlements)
        or selection_required - set(selection)
    ):
        return False
    if forecasts.duplicated(["forecast_id", "prediction_role", "driver"]).any():
        return False
    if settlements.duplicated(["settlement_id"]).any():
        return False
    if not settlements.empty:
        valid_flags = (
            settlements["settlement_valid"].astype(bool)
            & settlements["forecast_preexisted_settlement"].astype(bool)
            & settlements["forecast_fingerprint_valid"].astype(bool)
            & ~settlements["forecast_mutation_detected"].astype(bool)
        )
        if not valid_flags.all() or not set(settlements["forecast_id"]).issubset(
            set(forecasts["forecast_id"])
        ):
            return False

    from f1_prediction.modeling.prospective_monitoring import forecast_snapshot_hash

    for _, row in selection.iterrows():
        expected = str(row.get("forecast_snapshot_hash") or "").strip()
        forecast_id = str(row.get("forecast_id") or "").strip()
        if not expected or not forecast_id:
            return False
        forecast_group = forecasts[forecasts["forecast_id"].astype(str).eq(forecast_id)].copy()
        shadow_group = (
            shadow[shadow["forecast_id"].astype(str).eq(forecast_id)].copy()
            if "forecast_id" in shadow
            else pd.DataFrame()
        )
        if forecast_group.empty or forecast_snapshot_hash(forecast_group, shadow_group) != expected:
            return False
    return True


def _manifest_monitoring_ready(manifest: dict[str, Any]) -> bool:
    declared = {item["destination_relative_path"] for item in manifest["files"]}
    return {
        "reports/metrics/prospective_monitoring_protocol.json",
        "reports/metrics/prospective_monitoring_event_registry.csv",
        "data/processed/modeling/combined/modeling_dataset.parquet",
    }.issubset(declared)


def _has_symlink_component(root: Path, relative: PurePosixPath) -> bool:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _classification_counts(records: list[InventoryRecord]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for record in records:
        bucket = result.setdefault(record.classification, {"file_count": 0, "total_bytes": 0})
        bucket["file_count"] += 1
        bucket["total_bytes"] += record.size_bytes
    return result


def _first_list_length(payload: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    for value in payload.values():
        if isinstance(value, dict):
            nested = _first_list_length(value, *keys)
            if nested:
                return nested
    return 0


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
