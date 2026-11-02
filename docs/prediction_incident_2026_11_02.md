# Missing forecasts and free OpenF1 ingestion — November 2, 2026

## Root cause confirmed in production

The running Railway service was healthy, with the scheduler enabled at five-minute
intervals and `FP3_CONDITIONAL_POLICY_V1` ready. Mexico City had no forecast:
practice probes failed until qualifying began, then the orchestrator correctly
recorded `forecast_window_missed`. Bahrain remained the most recent preserved
forecast. This was not a frontend cache problem or a missing model deployment.

An HTTP check from the deployed container against Mexico FP3 returned:

- F1 timing archive `SessionInfo.json`: **401**.
- FastF1 timing mirror for the same session: **404**.

The [FastF1 maintainer confirms](https://github.com/theOehrly/Fast-F1/issues/978)
that the archive now requires an F1 TV Access subscription or higher. Repeatedly
retrying an unauthenticated FastF1 session cannot solve this restriction.
The October 27 investigation established availability failures but did not identify
this authentication requirement. Singapore remains a separate unsupported Sprint
format case.

The user explicitly chose to keep all production access free and use OpenF1.
[OpenF1 documents free historical access](https://openf1.org/) from 30 minutes
after the session ends. Its live service is not used.

## Implementation

- `configs/data.yaml` explicitly selects `session_data_source: openf1`.
  Configurations omitting the option retain FastF1 behavior for compatibility.
- `data/openf1_client.py` adds bounded unauthenticated HTTP requests, local JSON
  caching and request spacing below the free 30 requests/minute limit. Empty rows,
  authentication failures, 404s and rate limits cannot poison the cache.
- `data/openf1_loader.py` matches the FastF1 calendar's year, location, session type
  and UTC start exactly. It waits until 31 minutes after the published session end
  and requires session-completion evidence. No `latest` alias is used.
- Ingestion, `load-session` and readiness probes use the configured source.
  Production probe grace periods are 35 minutes to accommodate free historical
  access. FastF1 remains the calendar provider and source of preserved prior data.
- The adapter maps observed lap/sector timing, driver/team identities, compounds
  and stint tyre age. It excludes out/in laps, deleted laps, neutralized laps,
  incomplete sector timing and sums differing by more than 3 ms. A missing driver
  identity can be recovered only from an earlier practice in the same meeting.
  It cannot introduce a participant without observed laps or use future Q/race data.
- An unidentifiable deleted lap with a known car excludes that driver's ambiguous
  pace from valid/push aggregates. It does not invent a lap number or a deletion
  flag. Missing mini-sector samples alone do not invalidate complete sectors;
  mini-sector colors express relative pace, not track flags.
- Per-session metadata records source, adapter version, access mode, session keys,
  identity recovery sources, uncertain identities and limitations. New forecast
  snapshots freeze this provenance plus input and metadata hashes. Existing
  forecasts are not relabeled. The frozen model/config/feature contract is unchanged.
- The public homepage and methodology explain the archive restriction, free source
  migration, access delay and source-dependent validity proxy.

## Validation

An isolated ingestion of Mexico City 2026 succeeded for all required sessions:

| Session | Raw laps | Observed drivers |
| --- | ---: | ---: |
| FP1 | 589 | 22 |
| FP2 | 656 | 22 |
| FP3 | 469 | 22 |
| Qualifying | 366 | 22 |

The existing feature builder ran against these Parquet files. The existing entry-list
audit selected 22 drivers from completed FP3 and excluded ten earlier-practice-only
participants. It passed without Q data or a FastF1 results request.
The frozen 126-feature candidate produced 22 finite outputs for the 22 FP3-roster
drivers, with no completely missing feature columns and no qualifying input columns,
in an isolated compatibility check of the final persisted practice features.
No retrospective forecast or registry
entry was written; this is not prospective evidence or an accuracy evaluation.

A second comparison used preserved Hungary 2026 FP3 FastF1 laps:

- 392 shared driver/lap keys; maximum lap-time difference **0.0 seconds**.
- Maximum tyre-age difference on shared rows: **0.0 laps**.
- The 20 drivers with comparable valid timing had identical best valid laps.
- The proxy retained 203 valid laps versus FastF1's 216. One driver's ambiguous
  deletion message required conservative exclusion from valid pace.

These checks establish source accessibility and mechanical compatibility, not equal
model accuracy. The new source has a different validity/coverage distribution;
historical FastF1 backtests do not validate that shift. Monitor future prospective
outcomes separately. Optional telemetry/weather/message exports are not supported
by this lap adapter, and Sprint weekends remain unsupported by the FP3 model.

Before deployment, checksums were saved for the production protocol, event registry,
forecast/settlement/shadow ledgers, historical dataset and both integrity baselines.
All 723 Python tests and 102 frontend tests passed. Ruff, ESLint, TypeScript and
the production frontend build passed. The frozen predictive contract still validates.

## Production verification

Verified after deployment on November 2:

- Railway deployment `28ff39e9-3de0-4e88-a43b-4e11fd3ba48a`: `SUCCESS`.
- Vercel deployment `dpl_5AeSKSRqcpU9ab1Hm9CJLm37aZts`: `READY`, production alias
  [apex-pulse-ten.vercel.app](https://apex-pulse-ten.vercel.app/).
- Public API: scheduler running, policy ready, all predictive contract/dependency
  checks true, `session_data_source=openf1`, `session_data_access=free_historical`,
  adapter `OPENF1_HISTORICAL_LAPS_V1`. São Paulo is `WAITING_FOR_FP1`.
- In the actual deployed container, isolated free-source loads succeeded for
  Mexico FP3 (469 laps, 22 drivers, 246 proxy-valid laps) and Q (366 laps,
  22 drivers, 144 proxy-valid laps). No canonical retrospective forecast was written.
- All eight saved protocol/registry/ledger/dataset/baseline checksums are unchanged.
  Integrity comparison is `VALID_APPEND`, with no missing/mutated preexisting events,
  chronology errors or static invariant changes.
- Public browser verification shows the OpenF1 source/delay/proxy notice, retained
  Mexico/USA/Singapore outcomes, and Bahrain labeled as the last completed forecast.

The deployed runtime uses FastF1 3.9.1 for the calendar, pandas 2.3.3, NumPy 2.5.4
and scikit-learn 1.9.1. Ingestion no longer accesses the authenticated F1 timing archive.
The existing `/runtime` volume and single writer/replica were retained.

Deployment commands (from the repository root):

```bash
railway up --service apex-pulse --detach
npx --yes vercel deploy --project apex-pulse --scope matteosgobbas-projects --prod --yes
```

## Operator checks

```bash
curl -fsS https://apex-pulse-production.up.railway.app/api/v1/autopilot-status
railway ssh --service apex-pulse -- python -m f1_prediction.cli fp3-policy-report
railway ssh --service apex-pulse -- python -m f1_prediction.cli live-integrity-compare --baseline /runtime/reports/metrics/live_validation/pre_weekend_baseline.json
```

Expect the scheduler to be running, the policy to be ready, and
`prediction_policy.session_data_source=openf1` with
`session_data_access=free_historical`. The next forecast can only be created after
free FP3 data is published and before qualifying starts. Mexico's missed forecast
cannot be recovered as prospective evidence.
