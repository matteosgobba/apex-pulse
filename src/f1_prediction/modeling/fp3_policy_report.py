"""Read-only reporting for genuine post-freeze conditional-policy forecasts."""

from __future__ import annotations

import json

import pandas as pd

from f1_prediction.config import DataConfig
from f1_prediction.modeling import fp3_production_policy as fp3
from f1_prediction.modeling import prospective_monitoring as monitoring


def summarize(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {
            "events": 0,
            "aligned_rows": 0,
            "uniform_mae": None,
            "season_aware_mae": None,
            "production_mae": None,
            "delta": None,
            "season_aware_wins": 0,
            "uniform_wins": 0,
            "ties": 0,
        }
    event = rows.groupby(["season", "event_slug"])[["uniform_error", "season_aware_error"]].mean()
    differences = event["uniform_error"] - event["season_aware_error"]
    return {
        "events": len(event),
        "aligned_rows": len(rows),
        "uniform_mae": float(rows["uniform_error"].mean()),
        "season_aware_mae": float(rows["season_aware_error"].mean()),
        "production_mae": float(rows["production_error"].mean()),
        "delta": float((rows["uniform_error"] - rows["season_aware_error"]).mean()),
        "season_aware_wins": int(differences.gt(0).sum()),
        "uniform_wins": int(differences.lt(0).sum()),
        "ties": int(differences.eq(0).sum()),
    }


def aligned_evidence(settlements: pd.DataFrame, policy: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Never initialize this evidence with historical replay or old monitoring rows."""
    if settlements.empty or "production_policy_version" not in settlements:
        return pd.DataFrame(), pd.DataFrame()
    valid = settlements[
        settlements["production_policy_version"].isin([fp3.POLICY_VERSION, fp3.ROLLBACK_VERSION])
        & settlements["candidate_contract_fingerprint"].eq(policy["candidate_contract_fingerprint"])
        & settlements["prospective_evidence_class"].eq("post_freeze_prospective")
        & pd.to_datetime(settlements["forecast_created_at_utc"], utc=True).ge(
            pd.Timestamp(policy["freeze_at_utc"])
        )
        & pd.to_datetime(settlements["forecast_created_at_utc"], utc=True).lt(
            pd.to_datetime(settlements["qualifying_start_utc"], utc=True)
        )
        & pd.to_datetime(settlements["forecast_created_at_utc"], utc=True).lt(
            pd.to_datetime(settlements["settled_at_utc"], utc=True)
        )
        & settlements["settlement_valid"].fillna(False)
        & settlements["settlement_evaluable"].fillna(False)
        & ~settlements["event_slug"].str.startswith("synthetic")
    ].copy()
    live = valid[valid["prediction_role"].eq("observed_live_policy")].copy()
    if live.empty:
        return live, live
    keys = ["forecast_id", "season", "event_slug", "checkpoint", "driver"]
    if valid.duplicated([*keys, "prediction_role"]).any():
        raise ValueError("Duplicate prospective settlement rows")
    live["production_error"] = live["absolute_error_sec"]
    result = live.copy()
    for role, name in [
        ("uniform_default_shadow", "uniform"),
        ("season_aware_weighted_candidate_shadow", "season_aware"),
    ]:
        shadow = valid[valid["prediction_role"].eq(role)][
            [*keys, "absolute_error_sec", "actual_gap_sec"]
        ]
        shadow = shadow.rename(
            columns={"absolute_error_sec": f"{name}_error", "actual_gap_sec": f"{name}_actual"}
        )
        result = result.merge(shadow, on=keys, how="inner", validate="one_to_one")
        if not result[f"{name}_actual"].eq(result["actual_gap_sec"]).all():
            raise ValueError("Candidate outcome alignment conflict")
    return result.dropna(subset=["uniform_error", "season_aware_error", "production_error"]), live


def build_report(config: DataConfig, *, event: str | None = None) -> dict:
    policy = fp3.load_policy(config)
    if policy is None:
        raise ValueError("FP3 conditional policy is not configured")
    protocol = monitoring._read_json(config.metrics_output_dir / monitoring.PROTOCOL_FILE)
    registry = monitoring.read_csv(
        config.metrics_output_dir / "prospective_monitoring_event_registry.csv"
    )
    settlements = monitoring.read_parquet(
        config.metrics_output_dir / "prospective_monitoring_settlements.parquet"
    )
    aligned, live = aligned_evidence(settlements, policy)
    report = {
        "policy_version": policy["policy_version"],
        "freeze_at_utc": policy["freeze_at_utc"],
        "current_season": protocol["monitor_season"],
        "historical_evidence": "excluded; see historical aligned comparison and Replay V2",
        "overall": summarize(aligned),
        "by_regime": {},
        "events": [],
        "settled_production_rows": len(live),
        "completed_prospective_events": (
            len(live[["season", "event_slug"]].drop_duplicates()) if not live.empty else 0
        ),
        "production_mae_all_available": (
            float(live["production_error"].mean()) if not live.empty else None
        ),
        "fallback_events": (
            live.loc[live["fallback_reason"].fillna("").ne(""), ["event_slug", "fallback_reason"]]
            .drop_duplicates()
            .to_dict("records")
            if not live.empty
            else []
        ),
    }
    for regime in ("cold_start", "established"):
        report["by_regime"][regime] = summarize(
            aligned[aligned["regime"].eq(regime)] if not aligned.empty else aligned
        )
    cumulative = []
    if not live.empty:
        for (season, slug), group in live.sort_values(["season", "event_order"]).groupby(
            ["season", "event_slug"], sort=False
        ):
            paired = aligned[aligned["season"].eq(season) & aligned["event_slug"].eq(slug)]
            cumulative.append(paired)
            first = group.iloc[0]
            metrics = summarize(paired)
            delta = metrics["delta"]
            report["events"].append(
                {
                    "season": int(season),
                    "event": slug,
                    **{
                        c: first[c]
                        for c in [
                            "forecast_created_at_utc",
                            "settled_at_utc",
                            "regime",
                            "selected_candidate",
                            "intended_candidate",
                            "fallback_reason",
                            "production_policy_version",
                        ]
                    },
                    "prior_count": int(first["prior_current_season_eligible_events"]),
                    **metrics,
                    "production_mae_all_available": float(group["production_error"].mean()),
                    "winner": "unavailable"
                    if delta is None
                    else ("season_aware" if delta > 0 else "uniform" if delta < 0 else "tie"),
                    "cumulative": summarize(pd.concat(cumulative, ignore_index=True)),
                }
            )
    if event is None:
        completed = set(live["event_slug"]) if not live.empty else set()
        pending = registry[
            ~registry["event_slug"].isin(completed)
            & ~registry["event_slug"].str.startswith("synthetic")
        ].sort_values("event_order")
        event = str(pending.iloc[-1]["event_slug"]) if not pending.empty else None
    report["current_or_requested_event"] = event
    if event:
        row = monitoring.resolve_registry_event(registry, event)
        base, _ = monitoring.read_monitoring_dataset(
            monitoring.resolve_protocol_dataset_path(config, protocol)
        )
        try:
            _, state = fp3.legal_dataset(
                config, protocol, registry, row, base, policy, forecast_time=monitoring.utc_now()
            )
            report["selection_status"] = state
        except (ValueError, OSError) as exc:
            report["selection_status"] = {"status": "unavailable", "reason": str(exc)}
    return json.loads(json.dumps(report, default=str))
