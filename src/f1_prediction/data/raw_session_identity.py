"""Raw session identity validation for monitored qualifying targets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.data.fastf1_loader import build_lap_output_path
from f1_prediction.data.ingest import build_metadata_output_path
from f1_prediction.utils.paths import ensure_directory, slugify

IDENTITY_VERIFIED = "identity_verified"
IDENTITY_MISMATCH = "identity_mismatch"
METADATA_MISSING = "metadata_missing"
METADATA_MALFORMED = "metadata_malformed"
RAW_ARTIFACT_MISSING = "raw_artifact_missing"
SESSION_MISMATCH = "session_mismatch"
SEASON_MISMATCH = "season_mismatch"
EVENT_MISMATCH = "event_mismatch"
LEGACY_KNOWN_MISMATCH = "legacy_known_mismatch"

BLOCKING_STATUSES = {
    IDENTITY_MISMATCH,
    METADATA_MISSING,
    METADATA_MALFORMED,
    RAW_ARTIFACT_MISSING,
    SESSION_MISMATCH,
    SEASON_MISMATCH,
    EVENT_MISMATCH,
    LEGACY_KNOWN_MISMATCH,
}

SESSION_ALIASES: dict[str, set[str]] = {
    "fp1": {"fp1", "practice-1", "practice-one", "free-practice-1"},
    "fp2": {"fp2", "practice-2", "practice-two", "free-practice-2"},
    "fp3": {"fp3", "practice-3", "practice-three", "free-practice-3"},
    "q": {"q", "qualifying", "qualifying-session"},
}

EVENT_ALIASES: dict[str, set[str]] = {
    "australia": {"australia", "australian"},
    "austria": {"austria", "austrian"},
    "bahrain": {"bahrain", "sakhir"},
    "great-britain": {"great-britain", "britain", "british", "united-kingdom", "silverstone"},
    "italy": {"italy", "italian", "monza"},
    "saudi-arabia": {"saudi-arabia", "saudi-arabian", "jeddah"},
    "united-states": {"united-states", "united-states-of-america", "usa", "us", "austin"},
}

EVENT_ALIAS_LOOKUP = {
    alias: canonical for canonical, aliases in EVENT_ALIASES.items() for alias in aliases
}

LEGACY_NONCANONICAL_EVENTS = {(2026, "australia"), (2026, "great-britain")}
LEGACY_RAW_MISMATCH_EVENTS = {(2026, "great-britain")}


@dataclass(frozen=True)
class RawSessionIdentityResult:
    """Structured result for one raw session identity validation."""

    season: int
    requested_event: str
    requested_event_slug: str
    requested_session: str
    raw_laps_path: Path
    raw_metadata_path: Path
    metadata_event_name: str | None
    metadata_event_slug: str | None
    metadata_official_event_name: str | None
    metadata_season: int | None
    metadata_session: str | None
    path_event_slug: str | None
    identity_status: str
    identity_match: bool
    blocking: bool
    reason: str
    recommended_action: str
    legacy_noncanonical: bool = False
    quarantined: bool = False
    quarantined_for_prospective_evidence: bool = False

    def to_record(self, project_root: Path | None = None) -> dict[str, object]:
        """Return a JSON/CSV-safe record with portable paths."""
        return {
            "season": self.season,
            "requested_event": self.requested_event,
            "requested_event_slug": self.requested_event_slug,
            "requested_session": self.requested_session,
            "raw_laps_path": _portable_path(self.raw_laps_path, project_root),
            "raw_metadata_path": _portable_path(self.raw_metadata_path, project_root),
            "metadata_event_name": self.metadata_event_name,
            "metadata_event_slug": self.metadata_event_slug,
            "metadata_official_event_name": self.metadata_official_event_name,
            "metadata_season": self.metadata_season,
            "metadata_session": self.metadata_session,
            "path_event_slug": self.path_event_slug,
            "identity_status": self.identity_status,
            "identity_match": self.identity_match,
            "blocking": self.blocking,
            "reason": self.reason,
            "recommended_action": self.recommended_action,
            "legacy_noncanonical": self.legacy_noncanonical,
            "quarantined": self.quarantined,
            "quarantined_for_prospective_evidence": self.quarantined_for_prospective_evidence,
        }


def validate_raw_session_identity(
    config: DataConfig,
    *,
    season: int,
    event: str,
    session: str = "Q",
    raw_laps_path: Path | None = None,
    raw_metadata_path: Path | None = None,
) -> RawSessionIdentityResult:
    """Validate local raw session path and metadata identity without network access."""
    requested_session = session.strip().upper()
    requested_slug = slugify(event)
    laps_path = raw_laps_path or build_lap_output_path(
        config.lap_output_dir,
        season,
        event,
        requested_session,
    )
    metadata_path = raw_metadata_path or build_metadata_output_path(
        config.session_metadata_output_dir,
        season,
        event,
        requested_session,
    )
    path_event_slug = _path_event_slug(laps_path, config.lap_output_dir)
    legacy_noncanonical = (int(season), requested_slug) in LEGACY_NONCANONICAL_EVENTS

    if not laps_path.is_file():
        return _result(
            season=season,
            event=event,
            session=requested_session,
            laps_path=laps_path,
            metadata_path=metadata_path,
            path_event_slug=path_event_slug,
            status=RAW_ARTIFACT_MISSING,
            reason="Raw lap artifact is missing for the requested session.",
            legacy_noncanonical=legacy_noncanonical,
        )
    if not metadata_path.is_file():
        return _result(
            season=season,
            event=event,
            session=requested_session,
            laps_path=laps_path,
            metadata_path=metadata_path,
            path_event_slug=path_event_slug,
            status=METADATA_MISSING,
            reason="Raw session metadata JSON is missing.",
            legacy_noncanonical=legacy_noncanonical,
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _result(
            season=season,
            event=event,
            session=requested_session,
            laps_path=laps_path,
            metadata_path=metadata_path,
            path_event_slug=path_event_slug,
            status=METADATA_MALFORMED,
            reason=f"Raw session metadata JSON is malformed: {type(exc).__name__}.",
            legacy_noncanonical=legacy_noncanonical,
        )
    if not isinstance(metadata, dict):
        return _result(
            season=season,
            event=event,
            session=requested_session,
            laps_path=laps_path,
            metadata_path=metadata_path,
            path_event_slug=path_event_slug,
            status=METADATA_MALFORMED,
            reason="Raw session metadata JSON must contain an object.",
            legacy_noncanonical=legacy_noncanonical,
        )

    metadata_season = _int_or_none(metadata.get("season"))
    metadata_event_name = _text_or_none(metadata.get("event_name"))
    metadata_event_slug = _text_or_none(metadata.get("event_slug"))
    metadata_official_event_name = _text_or_none(
        metadata.get("official_event_name") or metadata.get("event_name")
    )
    metadata_session = _text_or_none(metadata.get("session_input") or metadata.get("session_name"))
    metadata_session_slug = _text_or_none(metadata.get("session_slug"))

    if metadata_season != int(season):
        return _metadata_result(
            season,
            event,
            requested_session,
            laps_path,
            metadata_path,
            path_event_slug,
            metadata,
            SEASON_MISMATCH,
            "Metadata season does not match the requested season.",
            legacy_noncanonical=legacy_noncanonical,
        )
    if path_event_slug != requested_slug:
        return _metadata_result(
            season,
            event,
            requested_session,
            laps_path,
            metadata_path,
            path_event_slug,
            metadata,
            EVENT_MISMATCH,
            "Raw lap artifact path event slug does not match the requested event.",
            legacy_noncanonical=legacy_noncanonical,
        )
    if not _sessions_match(requested_session, metadata_session, metadata_session_slug):
        return _metadata_result(
            season,
            event,
            requested_session,
            laps_path,
            metadata_path,
            path_event_slug,
            metadata,
            SESSION_MISMATCH,
            "Metadata session identity does not match the requested session.",
            legacy_noncanonical=legacy_noncanonical,
        )

    metadata_event_matches = all(
        _event_matches(event, candidate)
        for candidate in (
            metadata_event_slug,
            metadata.get("event_input"),
            metadata_event_name,
            metadata_official_event_name,
        )
        if _text_or_none(candidate) is not None
    )
    if not metadata_event_matches:
        status = (
            LEGACY_KNOWN_MISMATCH
            if (int(season), requested_slug) in LEGACY_RAW_MISMATCH_EVENTS
            else IDENTITY_MISMATCH
        )
        return _metadata_result(
            season,
            event,
            requested_session,
            laps_path,
            metadata_path,
            path_event_slug,
            metadata,
            status,
            "Metadata event identity does not match the requested/path event identity.",
            legacy_noncanonical=legacy_noncanonical,
        )

    quarantine_for_legacy = legacy_noncanonical
    reason = (
        "Raw session identity is verified, but this event is a legacy noncanonical "
        "monitoring snapshot and remains quarantined from prospective evidence."
        if quarantine_for_legacy
        else "Raw session path and metadata identities match the requested event."
    )
    recommended = (
        "Preserve the legacy snapshot for descriptive history only; do not overwrite or "
        "reinterpret it as prospective evidence."
        if quarantine_for_legacy
        else "Target onboarding and settlement may proceed when other monitoring gates pass."
    )
    return RawSessionIdentityResult(
        season=int(season),
        requested_event=event,
        requested_event_slug=requested_slug,
        requested_session=requested_session,
        raw_laps_path=laps_path,
        raw_metadata_path=metadata_path,
        metadata_event_name=metadata_event_name,
        metadata_event_slug=metadata_event_slug,
        metadata_official_event_name=metadata_official_event_name,
        metadata_season=metadata_season,
        metadata_session=metadata_session,
        path_event_slug=path_event_slug,
        identity_status=IDENTITY_VERIFIED,
        identity_match=True,
        blocking=False,
        reason=reason,
        recommended_action=recommended,
        legacy_noncanonical=legacy_noncanonical,
        quarantined=quarantine_for_legacy,
        quarantined_for_prospective_evidence=quarantine_for_legacy,
    )


def create_raw_session_identity_validation_report(
    config: DataConfig,
    *,
    season: int | None = None,
    event: str | None = None,
    session: str = "Q",
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    """Write raw session identity validation, failure, quarantine, and runbook artifacts."""
    ensure_directory(config.metrics_output_dir)
    results = validate_raw_session_identity_events(
        config,
        season=season,
        event=event,
        session=session,
    )
    checks = pd.DataFrame([result.to_record(config.project_root) for result in results])
    if checks.empty:
        checks = pd.DataFrame(columns=list(RawSessionIdentityResult.__dataclass_fields__))
    failures = checks[
        checks.get("blocking", pd.Series(dtype=bool)).astype(bool)
        | checks.get("quarantined", pd.Series(dtype=bool)).astype(bool)
    ].copy()
    quarantine = build_raw_session_identity_quarantine(checks)
    event_allowed = (
        bool(
            not checks["blocking"].astype(bool).any()
            and not checks["quarantined_for_prospective_evidence"].astype(bool).any()
        )
        if event
        else None
    )
    summary = {
        "status": "fail" if checks["blocking"].astype(bool).any() else "pass",
        "identity_validation_available": True,
        "requested_season": season,
        "requested_event": event,
        "requested_event_slug": slugify(event) if event else None,
        "requested_session": session.strip().upper(),
        "events_checked": int(len(checks)),
        "identity_verified_event_count": int(
            checks["identity_status"].astype(str).eq(IDENTITY_VERIFIED).sum()
        ),
        "blocking_event_count": int(checks["blocking"].astype(bool).sum()),
        "quarantined_event_count": int(quarantine["quarantined"].astype(bool).sum())
        if not quarantine.empty
        else 0,
        "target_onboarding_allowed": event_allowed,
        "settlement_allowed": event_allowed,
        "recommended_operator_action": _summary_recommendation(checks),
    }
    summary_path = config.metrics_output_dir / "raw_session_identity_validation_summary.json"
    checks_path = config.metrics_output_dir / "raw_session_identity_validation_checks.csv"
    failures_path = config.metrics_output_dir / "raw_session_identity_validation_failures.csv"
    quarantine_path = config.metrics_output_dir / "raw_session_identity_quarantine.csv"
    runbook_path = config.metrics_output_dir / "raw_session_identity_runbook.md"
    _write_json(summary_path, summary)
    checks.to_csv(checks_path, index=False)
    failures.to_csv(failures_path, index=False)
    quarantine.to_csv(quarantine_path, index=False)
    runbook_path.write_text(_runbook(summary, quarantine), encoding="utf-8")
    return summary, checks, failures, quarantine, runbook_path


def validate_raw_session_identity_events(
    config: DataConfig,
    *,
    season: int | None = None,
    event: str | None = None,
    session: str = "Q",
) -> list[RawSessionIdentityResult]:
    """Validate requested event plus known legacy raw-Q events when present."""
    requests: dict[tuple[int, str], str] = {}
    if season is not None and event is not None:
        requests[(int(season), slugify(event))] = event
    for legacy_season, legacy_slug in LEGACY_NONCANONICAL_EVENTS:
        laps_path = config.lap_output_dir / str(legacy_season) / legacy_slug / "q_laps.parquet"
        metadata_path = (
            config.session_metadata_output_dir
            / str(legacy_season)
            / legacy_slug
            / "q_metadata.json"
        )
        if laps_path.is_file() or metadata_path.is_file():
            requests.setdefault((legacy_season, legacy_slug), legacy_slug.replace("-", " ").title())
    if not requests:
        metadata_paths = sorted(config.session_metadata_output_dir.glob("*/**/q_metadata.json"))
        for metadata_path in metadata_paths:
            metadata_season = _int_or_none(metadata_path.parent.parent.name)
            if metadata_season is None:
                continue
            slug = metadata_path.parent.name
            requests[(metadata_season, slug)] = slug.replace("-", " ").title()
    return [
        validate_raw_session_identity(
            config,
            season=request_season,
            event=request_event,
            session=session,
        )
        for (request_season, _), request_event in sorted(requests.items())
    ]


def build_raw_session_identity_quarantine(checks: pd.DataFrame) -> pd.DataFrame:
    """Build the operator-facing raw identity quarantine table."""
    columns = [
        "season",
        "event",
        "event_slug",
        "session",
        "identity_status",
        "quarantined",
        "legacy_noncanonical",
        "quarantined_for_prospective_evidence",
        "raw_laps_path",
        "raw_metadata_path",
        "metadata_event_name",
        "expected_event",
        "reason",
        "recommended_action",
    ]
    rows: list[dict[str, object]] = []
    for _, row in checks.iterrows():
        legacy = bool(row.get("legacy_noncanonical", False))
        blocking = bool(row.get("blocking", False))
        quarantined = bool(row.get("quarantined", False) or blocking or legacy)
        rows.append(
            {
                "season": row.get("season"),
                "event": row.get("requested_event"),
                "event_slug": row.get("requested_event_slug"),
                "session": row.get("requested_session"),
                "identity_status": row.get("identity_status"),
                "quarantined": quarantined,
                "legacy_noncanonical": legacy,
                "quarantined_for_prospective_evidence": bool(
                    row.get("quarantined_for_prospective_evidence", False) or legacy or blocking
                ),
                "raw_laps_path": row.get("raw_laps_path"),
                "raw_metadata_path": row.get("raw_metadata_path"),
                "metadata_event_name": row.get("metadata_event_name"),
                "expected_event": row.get("requested_event"),
                "reason": row.get("reason"),
                "recommended_action": row.get("recommended_action"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def target_identity_manifest_fields(
    result: RawSessionIdentityResult,
    *,
    project_root: Path,
) -> dict[str, object]:
    """Fields persisted in monitoring manifests when target identity validation passes."""
    return {
        "raw_session_identity_status": result.identity_status,
        "raw_session_identity_verified": result.identity_match and not result.blocking,
        "raw_session_identity_blocking": result.blocking,
        "raw_session_identity_reason": result.reason,
        "raw_session_identity_recommended_action": result.recommended_action,
        "raw_session_identity_validated_at_utc": _utc_now(),
        "raw_session_identity_raw_laps_path": _portable_path(result.raw_laps_path, project_root),
        "raw_session_identity_metadata_path": _portable_path(
            result.raw_metadata_path,
            project_root,
        ),
        "raw_session_identity_metadata_event_name": result.metadata_event_name,
        "raw_session_identity_metadata_event_slug": result.metadata_event_slug,
        "raw_session_identity_metadata_official_event_name": (result.metadata_official_event_name),
        "raw_session_identity_metadata_season": result.metadata_season,
        "raw_session_identity_metadata_session": result.metadata_session,
        "raw_session_identity_path_event_slug": result.path_event_slug,
        "quarantine_status": "quarantined" if result.quarantined else "clear",
        "quarantine_reason": result.reason if result.quarantined else "",
        "legacy_noncanonical": result.legacy_noncanonical,
        "quarantined_for_prospective_evidence": result.quarantined_for_prospective_evidence,
    }


def write_raw_identity_target_block(
    config: DataConfig,
    result: RawSessionIdentityResult,
) -> Path:
    """Write a per-event structured failure artifact without touching targets."""
    path = (
        config.project_root
        / "data/processed/monitoring"
        / str(result.season)
        / result.requested_event_slug
        / "raw_session_identity_target_block.json"
    )
    ensure_directory(path.parent)
    _write_json(
        path,
        {
            "status": "blocked",
            "target_onboarding_allowed": False,
            "settlement_allowed": False,
            "validation": result.to_record(config.project_root),
        },
    )
    return path


def _metadata_result(
    season: int,
    event: str,
    session: str,
    laps_path: Path,
    metadata_path: Path,
    path_event_slug: str | None,
    metadata: dict[str, Any],
    status: str,
    reason: str,
    *,
    legacy_noncanonical: bool,
) -> RawSessionIdentityResult:
    return RawSessionIdentityResult(
        season=int(season),
        requested_event=event,
        requested_event_slug=slugify(event),
        requested_session=session,
        raw_laps_path=laps_path,
        raw_metadata_path=metadata_path,
        metadata_event_name=_text_or_none(metadata.get("event_name")),
        metadata_event_slug=_text_or_none(metadata.get("event_slug")),
        metadata_official_event_name=_text_or_none(
            metadata.get("official_event_name") or metadata.get("event_name")
        ),
        metadata_season=_int_or_none(metadata.get("season")),
        metadata_session=_text_or_none(
            metadata.get("session_input") or metadata.get("session_name")
        ),
        path_event_slug=path_event_slug,
        identity_status=status,
        identity_match=False,
        blocking=status in BLOCKING_STATUSES,
        reason=reason
        + (
            " This is a known legacy raw-source mismatch."
            if status == LEGACY_KNOWN_MISMATCH
            else ""
        ),
        recommended_action=(
            "Inspect metadata and re-ingest the correct qualifying session before retrying "
            "target onboarding or settlement. Never overwrite or reinterpret legacy Australia "
            "or Great Britain snapshots."
            if legacy_noncanonical
            else "Inspect metadata and re-ingest the correct qualifying session before retrying."
        ),
        legacy_noncanonical=legacy_noncanonical,
        quarantined=True,
        quarantined_for_prospective_evidence=True,
    )


def _result(
    *,
    season: int,
    event: str,
    session: str,
    laps_path: Path,
    metadata_path: Path,
    path_event_slug: str | None,
    status: str,
    reason: str,
    legacy_noncanonical: bool,
) -> RawSessionIdentityResult:
    return RawSessionIdentityResult(
        season=int(season),
        requested_event=event,
        requested_event_slug=slugify(event),
        requested_session=session,
        raw_laps_path=laps_path,
        raw_metadata_path=metadata_path,
        metadata_event_name=None,
        metadata_event_slug=None,
        metadata_official_event_name=None,
        metadata_season=None,
        metadata_session=None,
        path_event_slug=path_event_slug,
        identity_status=status,
        identity_match=False,
        blocking=status in BLOCKING_STATUSES,
        reason=reason,
        recommended_action=(
            "Inspect metadata and re-ingest the correct qualifying session before retrying."
        ),
        legacy_noncanonical=legacy_noncanonical,
        quarantined=True,
        quarantined_for_prospective_evidence=True,
    )


def _event_matches(expected: object, observed: object) -> bool:
    expected_key = canonical_event_key(expected)
    observed_key = canonical_event_key(observed)
    return expected_key is not None and expected_key == observed_key


def canonical_event_key(value: object) -> str | None:
    text = _text_or_none(value)
    if text is None:
        return None
    slug = slugify(text)
    for suffix in ("-grand-prix", "-gp"):
        if slug.endswith(suffix):
            slug = slug[: -len(suffix)]
    return EVENT_ALIAS_LOOKUP.get(slug, slug)


def _sessions_match(
    requested: str,
    metadata_session: object,
    metadata_session_slug: object,
) -> bool:
    expected = slugify(requested)
    aliases = SESSION_ALIASES.get(expected, {expected})
    candidates = {
        slugify(candidate)
        for candidate in (metadata_session, metadata_session_slug)
        if _text_or_none(candidate) is not None
    }
    return bool(candidates & aliases)


def _path_event_slug(path: Path, lap_output_dir: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(lap_output_dir.resolve())
    except ValueError:
        parts = path.parts
        return parts[-2] if len(parts) >= 2 else None
    return relative.parts[1] if len(relative.parts) >= 3 else None


def _summary_recommendation(checks: pd.DataFrame) -> str:
    if checks.empty:
        return "No raw qualifying session artifacts were discovered."
    blocking = checks[checks["blocking"].astype(bool)]
    if not blocking.empty:
        return str(blocking["recommended_action"].iloc[0])
    if checks["quarantined"].astype(bool).any():
        return "Legacy quarantined records may remain descriptive only; clean events can proceed."
    return "Raw session identity validation passed for the checked events."


def _runbook(summary: dict[str, object], quarantine: pd.DataFrame) -> str:
    blocked = int(summary.get("blocking_event_count") or 0)
    quarantined = int(summary.get("quarantined_event_count") or 0)
    return "\n".join(
        [
            "# Raw Session Identity Validation Runbook",
            "",
            f"- Status: {summary.get('status')}",
            f"- Blocking events: {blocked}",
            f"- Quarantined events: {quarantined}",
            "",
            "When validation fails:",
            "",
            "1. Inspect metadata for the requested Q raw artifact.",
            "2. Verify event, season, session, and path identity.",
            "3. Re-ingest the correct Q session with explicit season, event, and session.",
            "4. Rerun target onboarding.",
            "5. Rerun settlement only after identity validation passes.",
            "",
            "Never overwrite or reinterpret legacy Australia or Great Britain snapshots.",
            "",
            "Quarantined events:",
            *[
                f"- {row.event_slug}: {row.identity_status} - {row.reason}"
                for row in quarantine.itertuples(index=False)
                if bool(row.quarantined)
            ],
            "",
        ]
    )


def _portable_path(path: Path, project_root: Path | None) -> str:
    if project_root is None:
        return path.as_posix()
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _text_or_none(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _int_or_none(value: object) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
