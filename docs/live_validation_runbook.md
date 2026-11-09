# Apex Pulse live-validation runbook

This runbook is for the single Railway API/writer service. It preserves the
append-only monitoring ledgers. Never delete, reseed, or force-overwrite an
immutable artifact as routine recovery.

## Read-only triage

```bash
curl -fsS https://apex-pulse-production.up.railway.app/api/v1/health
curl -fsS https://apex-pulse-production.up.railway.app/api/v1/autopilot-status
curl -fsS https://apex-pulse-production.up.railway.app/api/v1/live-validation-status
railway logs --service apex-pulse --lines 300
railway logs --service apex-pulse --lines 300 --filter '@level:error OR @level:warn'
```

Inside the running service, use only read-only diagnostics first:

```bash
railway ssh --service apex-pulse python -m f1_prediction.cli production-runtime-check
railway ssh --service apex-pulse python -m f1_prediction.cli live-integrity-compare --baseline /runtime/reports/metrics/live_validation/pre_weekend_baseline.json
railway ssh --service apex-pulse python -m f1_prediction.cli autopilot-tick --dry-run --json
```

`UNCHANGED` and `VALID_APPEND` are passing integrity outcomes. Every other
classification requires operator investigation before mutation is re-enabled.
Derived dashboard history is a filtered view and is not immutable event evidence;
its disappearance alone must not be interpreted as a historical forecast rewrite.
Existing baseline files remain valid without replacement.

The autopilot endpoint also reports `prediction_policy`: verify `status=ready`,
`production_policy_version=FP3_CONDITIONAL_POLICY_V1`, `mode=conditional`, and
all three contract/dependency checks true. This describes the installed rule; the forecast's own
metadata describes the model actually used for its immutable prediction. The
`weekend_history` list retains skipped/missed outcomes after event rollover.
For the subscription-free ingestion installed on November 2, also confirm
`session_data_source=openf1` and `session_data_access=free_historical`. The scheduler
waits 35 minutes after practice ends, then retries free historical data before
qualifying. The model being ready does not prove that future session data is published.
See the [November 2 source migration](prediction_incident_2026_11_02.md) and
[October 27 incident audit](prediction_incident_2026_10_27.md).

## Qualifying roster versus practice evidence

Roster resolution and feature evidence are intentionally different. The forecast roster contains
only drivers eligible for the weekend qualifying prediction. The practice-evidence population keeps
every driver observed in FP1, FP2 or FP3, including test and substitute drivers, so their laps may
contribute to team-level features without being copied into another driver's direct features.

The OpenF1 workflow checks FIA entry-list documents and revisions with a publication timestamp before
the forecast. An explicit event roster remains the operator override. The latest verified practice
session is the fallback when FIA data is unavailable. A newer FIA driver-change decision with no
revised list yet is retryable. The successful forecast snapshot records the official document URL
and timestamp when available, alongside the entrants and their actual model inputs.

Inspect the event audit under
`reports/metrics/qualifying_entry_lists/<season>/<event>/`. The summary and CSV files identify the
selected roster source, eligible drivers, all practice participants and their sessions, safe
practice-only exclusions, current weekend team mapping, and any typed retryable ambiguity. Different
FP1 and FP3 participant sets are normal and are not a blocker by themselves. Duplicate identities,
contradictory team mappings in the selected source, missing teams, empty/incomplete source sessions,
or an otherwise indeterminate current roster remain blockers. A verified entrant can miss practice
and still be scored when the team has usable pace from an earlier or the latest session. The
registered practice artifact retains the full practice population and the frozen scoring input
contains only entrants. Settlement uses the frozen roster and records actual-only entrants as
unforecasted.

Before qualifying, retryable roster ambiguity is left to the scheduler so newly published public
data can resolve it. Once qualifying starts without an immutable forecast, expect
`action_result=forecast_window_missed`: do not invoke a manual backfill, do not edit the append-only
registry, and do not regenerate historical forecast artifacts.

## Transient FastF1 or network failure

Timeouts, unpublished sessions, incomplete public results, and temporary
Railway network failures remain retryable. The scheduler loop survives, no
partial forecast or settlement should be committed, and the next scheduled tick
retries. Successful local practice exports with matching event identity and counts are reused
during OpenF1 access interruptions. Human intervention is needed only when retries persist beyond the
normal publication delay, the scheduler stops, capacity warnings appear, or a
blocking integrity/preflight condition is reported.

## Break glass: stop writes without deleting state

Disable mutation permission first. This variable change deploys a new instance;
the persistent `/runtime` volume remains attached.

```bash
railway variable set APEX_PULSE_AUTOPILOT_ENABLED=false --service apex-pulse
```

If all automatic ticks must also stop, disable the scheduler separately:

```bash
railway variable set APEX_PULSE_AUTOPILOT_SCHEDULER_ENABLED=false --service apex-pulse
```

Re-run the health, runtime, autopilot-status, live-validation-status, and
integrity commands above. Preserve logs and checkpoint metadata. Do not run
`monitoring-before-qualifying`, `monitoring-after-qualifying`, `dashboard-export`,
or any force/reseed operation as diagnosis.

Re-enable only after the blocking cause is understood and integrity comparison
passes:

```bash
railway variable set APEX_PULSE_AUTOPILOT_ENABLED=true --service apex-pulse
railway variable set APEX_PULSE_AUTOPILOT_SCHEDULER_ENABLED=true --service apex-pulse
```

Verify exactly one scheduler is running and that the next tick has
`trigger_source=scheduler`. Keep the deployment at one replica/single writer.

## Live checkpoints

The pre-weekend baseline is written once and never silently replaced:

```bash
railway ssh --service apex-pulse python -m f1_prediction.cli live-integrity-baseline --output /runtime/reports/metrics/live_validation/pre_weekend_baseline.json
```

After the naturally occurring forecast and settlement transitions, record small
metadata checkpoints (no FastF1 cache or Parquet copies):

```bash
railway ssh --service apex-pulse python -m f1_prediction.cli live-validation-checkpoint --stage post_forecast_validation
railway ssh --service apex-pulse python -m f1_prediction.cli live-validation-checkpoint --stage post_settlement_validation
```

If a checkpoint already exists, the command stops instead of overwriting it.
