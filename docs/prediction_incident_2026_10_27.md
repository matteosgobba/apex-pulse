# Prediction availability and FP3 policy audit — October 27, 2026

## Findings from the running Railway service

The September 5 deployment `31d7841c-1718-4198-845a-efa635579ac7` was still
running. It did **not** contain `configs/fp3_production_policy_v1.json` or the
conditional production implementation committed on September 20. The last
published Bahrain forecast (October 3, 06:01 UTC, 22 drivers) used Uniform.
The scheduler was enabled and running every five minutes.

The persisted `autopilot_runs.jsonl` distinguishes the subsequent weekends:

| Weekend | Observed outcome | Explanation |
| --- | --- | --- |
| Singapore | `UNSUPPORTED_WEEKEND_FORMAT` | Calendar format `sprint_qualifying`; FP2/FP3 absent. The FP3 model intentionally supports conventional weekends only. |
| United States | `forecast_window_missed` | FP1/FP2/FP3 timing could not be loaded throughout their available pre-qualifying probe windows. FP3 retried 26 times between October 24 18:49 and 20:56 UTC; qualifying began at 21:00 UTC. |
| Mexico City | `WAITING_FOR_FP1` | Next conventional weekend; FP1 scheduled October 30 at 18:30 UTC. |

A fresh isolated-cache USA FP3 load on the server reproduced
`SessionNotAvailableError` from the public timing backend/mirror, followed by
FastF1's generic `DataNotLoadedError` on `session.laps`. This is an upstream
availability/access limitation, not evidence that a stored model stopped
predicting. No alternative event's data or post-qualifying backfill is valid.

The simultaneous `PREEXISTING_EVENT_MUTATED` warning was a separate false
alarm. All seven differences were **only** `historical_dashboard_event`, a
derived history projection that the dashboard had filtered out. Original
registry, forecast, settlement and event evidence hashes were unchanged;
static invariants, chronology and forecast snapshot checks passed.

A final conditional-policy materialization check found an additional deployment
dependency: the initial production-state bundle had omitted Hungary's derived
`practice_session_features.parquet`. The original practice laps were present.
The aggregates were rebuilt solely from those preserved laps (545 FP1, 572 FP2,
392 FP3 rows), leaving all immutable evidence untouched. No FastF1 download or
retrospective prediction was used. The report then successfully reconstructed
legal training history. Before Mexico, four settled events qualify for history:
Hungary, Spain, Azerbaijan and Bahrain. Four means Uniform remains the intended
V1 selection; the season-aware switch requires a fifth eligible event, regardless
of calendar round number.

## Changes

- Integrity comparisons ignore the derived dashboard projection, including in
  existing baseline files. Original inputs, forecasts, settlements, registry,
  chronology and static invariants remain protected. No baseline is replaced.
- A compact operational weekend history is reconstructed once from the existing
  run log and subsequently updated per tick. The public homepage retains the
  reasons for missed/unsupported weekends after rollover, separately from
  actual forecasts and prospective evaluation evidence.
- Scheduler status exposes the installed policy version/mode and validates the
  frozen predictive source/config contract, historical dataset and prior feature
  dependencies at startup.
  A missing or conflicting contract is never advertised as ready.
- Exported forecast metadata and technical details expose the policy and actual
  training method recorded on that forecast. Installing a newer rule does not
  relabel an existing prediction.
- Readiness errors identify season, event, session and FastF1 API path rather
  than merely repeating `See Session.load`.
- Production-state export now preserves registered practice aggregates and V1
  provenance snapshots instead of excluding them as reconstructable extras.

The predictive specification is unchanged: `FP3_CONDITIONAL_POLICY_V1` selects
Uniform for 0–4 eligible settled same-season training events and the season-aware
candidate from 5 onward. Round number, synthetic events, invalid legacy events
and events without qualifying outcomes cannot substitute for this count.
The persistent historical dataset matched the frozen SHA-256
`dca8fb564d953f699a37405ccbef75eb2b77ea4282ed4dc81b14d802a01a2165`.

Validation: 694 Python tests and 101 frontend tests passed; lint, TypeScript and
the production frontend build passed. In the actual new Railway container,
60 candidate refits across the 30 canonical 2024/2025 weekends reproduced all
1,190 aligned candidate predictions exactly (maximum difference 0.0). The
deployed environment used Python 3.12.16, FastF1 3.9.0, NumPy 2.5.4, pandas
2.3.3 and scikit-learn 1.9.1. The four legal current-season events materialized
21/20/22/22 training rows respectively. Hash comparison confirmed all 25
protected production ledger/protocol/dataset/baseline files unchanged.

## Deployment verification

Verified live on October 27:

- Railway deployment `3d6582ed-6e1f-420d-9013-4eb55ca73cc7`: `SUCCESS`.
- Vercel deployment `dpl_Lbo7zEdaGSKvXrvfDnQXuReRE3pY`: `READY`, aliased to
  `https://apex-pulse-ten.vercel.app`.
- Installed V1: `status=ready`, `mode=conditional`, predictive contract,
  historical dataset and prior feature dependencies all valid.
- Scheduler running, Mexico `WAITING_FOR_FP1`, integrity `VALID_APPEND`,
  operator attention `NONE`.
- Public browser verification showed the conditional policy notice and retained
  USA/Singapore availability explanations, with Bahrain still the last completed
  immutable forecast.

Implementation locations:

- `src/f1_prediction/operational_diagnostics.py`, `autopilot_scheduler.py` and
  `modeling/weekend_orchestrator.py`: installed contract checks, session diagnostics
  and persistent operational outcomes.
- `src/f1_prediction/modeling/live_event_integrity.py`: canonical evidence comparison.
- `src/f1_prediction/production_state.py`: practice inputs and V1 provenance in exports.
- `src/f1_prediction/dashboard/export.py` and `dashboard_api/service.py`: public
  forecast metadata and operational history.
- `web/components/operational-weekend-card.tsx`, `web/lib/public-view-model.ts`
  and `web/lib/dashboard-types.ts`: public notices and per-forecast details.
- Corresponding Python/frontend regression tests, `.vercelignore`, README and runbook.

After deploying, check:

```bash
curl -fsS https://apex-pulse-production.up.railway.app/api/v1/autopilot-status
curl -fsS https://apex-pulse-production.up.railway.app/api/v1/live-validation-status
railway ssh --service apex-pulse -- python -m f1_prediction.cli live-integrity-compare --baseline /runtime/reports/metrics/live_validation/pre_weekend_baseline.json
railway ssh --service apex-pulse -- python -m f1_prediction.cli fp3-policy-report
```

Expect `prediction_policy.status=ready`, version `FP3_CONDITIONAL_POLICY_V1`,
mode `conditional`, all three contract/dependency checks true, a running scheduler, and
`VALID_APPEND` against the existing pre-weekend baseline. Bahrain remains an
immutable Uniform forecast. There are no new-policy prospective results until
a supported future weekend produces a forecast before qualifying.

Deployment uses the current local source (`railway up --service apex-pulse`
and `vercel deploy --project apex-pulse --scope matteosgobbas-projects --prod`
from the repository root). Vercel already has `web` configured as its root;
`.vercelignore` limits uploads to frontend files and excludes environments,
data, models, reports and credentials. The inspected Railway
service had no GitHub source configured: a Git commit/push alone does not update
that backend. Keep one writer/replica and retain the existing `/runtime` volume.
