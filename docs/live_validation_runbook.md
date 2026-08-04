# Apex Pulse live-validation runbook

This runbook is for the single Railway API/writer service. It preserves the
append-only monitoring ledgers. Never delete, reseed, or force-overwrite an
immutable artifact as routine recovery.

## Read-only triage

```bash
curl -fsS https://apex-pulse-production.up.railway.app/api/v1/health
curl -fsS https://apex-pulse-production.up.railway.app/api/v1/autopilot-status
curl -fsS https://apex-pulse-production.up.railway.app/api/v1/live-validation-status
railway logs --service apex-pulse-production --lines 300
railway logs --service apex-pulse-production --lines 300 --filter '@level:error OR @level:warn'
```

Inside the running service, use only read-only diagnostics first:

```bash
railway ssh --service apex-pulse-production python -m f1_prediction.cli production-runtime-check
railway ssh --service apex-pulse-production python -m f1_prediction.cli live-integrity-compare --baseline /runtime/reports/metrics/live_validation/pre_weekend_baseline.json
railway ssh --service apex-pulse-production python -m f1_prediction.cli autopilot-tick --dry-run --json
```

`UNCHANGED` and `VALID_APPEND` are passing integrity outcomes. Every other
classification requires operator investigation before mutation is re-enabled.

## Transient FastF1 or network failure

Timeouts, unpublished sessions, incomplete public results, and temporary
Railway network failures remain retryable. The scheduler loop survives, no
partial forecast or settlement should be committed, and the next scheduled tick
retries. Human intervention is needed only when retries persist beyond the
normal publication delay, the scheduler stops, capacity warnings appear, or a
blocking integrity/preflight condition is reported.

## Break glass: stop writes without deleting state

Disable mutation permission first. This variable change deploys a new instance;
the persistent `/runtime` volume remains attached.

```bash
railway variable set APEX_PULSE_AUTOPILOT_ENABLED=false --service apex-pulse-production
```

If all automatic ticks must also stop, disable the scheduler separately:

```bash
railway variable set APEX_PULSE_AUTOPILOT_SCHEDULER_ENABLED=false --service apex-pulse-production
```

Re-run the health, runtime, autopilot-status, live-validation-status, and
integrity commands above. Preserve logs and checkpoint metadata. Do not run
`monitoring-before-qualifying`, `monitoring-after-qualifying`, `dashboard-export`,
or any force/reseed operation as diagnosis.

Re-enable only after the blocking cause is understood and integrity comparison
passes:

```bash
railway variable set APEX_PULSE_AUTOPILOT_ENABLED=true --service apex-pulse-production
railway variable set APEX_PULSE_AUTOPILOT_SCHEDULER_ENABLED=true --service apex-pulse-production
```

Verify exactly one scheduler is running and that the next tick has
`trigger_source=scheduler`. Keep the deployment at one replica/single writer.

## Live checkpoints

The pre-weekend baseline is written once and never silently replaced:

```bash
railway ssh --service apex-pulse-production python -m f1_prediction.cli live-integrity-baseline --output /runtime/reports/metrics/live_validation/pre_weekend_baseline.json
```

After the naturally occurring forecast and settlement transitions, record small
metadata checkpoints (no FastF1 cache or Parquet copies):

```bash
railway ssh --service apex-pulse-production python -m f1_prediction.cli live-validation-checkpoint --stage post_forecast_validation
railway ssh --service apex-pulse-production python -m f1_prediction.cli live-validation-checkpoint --stage post_settlement_validation
```

If a checkpoint already exists, the command stops instead of overwriting it.
