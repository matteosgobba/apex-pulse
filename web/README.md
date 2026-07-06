# Apex Pulse Web

Read-only Next.js frontend for the Apex Pulse dashboard contract.

## Local development

Terminal 1, from the repository root:

```bash
.venv/bin/python -m f1_prediction.cli dashboard-export
.venv/bin/python -m f1_prediction.cli dashboard-api
```

Terminal 2, for the frontend:

```bash
cd web
npm install
cp .env.example .env.local
npm run dev
```

By default the frontend expects the API at:

```text
http://127.0.0.1:8000
```

Override it with:

```text
NEXT_PUBLIC_APEX_PULSE_API_BASE_URL=http://127.0.0.1:8000
NEXT_PUBLIC_APEX_PULSE_DASHBOARD_STALE_AFTER_MINUTES=180
```

The stale threshold controls only visual freshness treatment. Old but valid artifacts are still
served by the API and rendered by the frontend.

## Routes

```text
/
/forecast
/practice
/settlement
/monitoring-history
/methodology
```

`/` shows the current monitored event with compact forecast and practice previews. `/forecast`
shows the exported qualifying forecast leaderboard. `/practice` shows FP1, FP2, FP3, and Q
artifact availability plus workflow readiness. `/settlement` compares an exported pre-qualifying
forecast against qualifying outcomes when settlement data exists. `/monitoring-history` keeps valid
prospective monitoring evidence, legacy descriptive records, and historical backtest context
visually and analytically separate. `/methodology` explains the prediction lifecycle and public-data
limits.

## Scope

This app is a read-only consumer of validated dashboard JSON. It does not trigger ingestion,
preflight, forecast generation, target ingestion, settlement, model training, FastF1 calls, or any
other ML workflow operation.

Dashboard freshness depends on regenerated dashboard artifacts and the separately running read-only
Python API. The frontend does not poll, schedule refreshes, or claim live telemetry. Freshness labels
use the artifact `generated_at_utc` timestamp and classify data as fresh, aging, stale, or unknown.

Milestone 40F includes local startup documentation, deployment-readiness notes, conservative
freshness UX, safe API-unavailable states, and production build hardening. Automatic deployment,
polling, websockets, and live updates are intentionally out of scope.

## Deployment notes

For a portfolio demo, host the Next.js frontend on Vercel or an equivalent platform and host the
FastAPI dashboard API separately on Render, Railway, Fly.io, or an equivalent Python service. The
API runtime must have access to a validated `reports/dashboard/` artifact bundle generated before
publication. Do not publish raw FastF1 cache data, raw lap exports, full modeling Parquets, or model
binaries for the dashboard demo.
