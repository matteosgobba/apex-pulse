# Apex Pulse Web

Read-only Next.js frontend for the Apex Pulse dashboard contract.

## Local development

```bash
cd web
npm install
cp .env.example .env.local
npm run dev
```

The Python dashboard artifacts and API must be prepared separately from the repository root:

```bash
.venv/bin/python -m f1_prediction.cli dashboard-export
.venv/bin/python -m f1_prediction.cli dashboard-api
```

By default the frontend expects the API at:

```text
http://127.0.0.1:8000
```

Override it with:

```text
NEXT_PUBLIC_APEX_PULSE_API_BASE_URL=http://127.0.0.1:8000
```

## Routes

```text
/
/forecast
/practice
/methodology
```

`/` shows the current monitored event with compact forecast and practice previews. `/forecast`
shows the exported qualifying forecast leaderboard. `/practice` shows FP1, FP2, FP3, and Q
artifact availability plus workflow readiness. `/methodology` explains the prediction lifecycle and
public-data limits.

## Scope

This app is a read-only consumer of validated dashboard JSON. It does not trigger ingestion,
preflight, forecast generation, target ingestion, settlement, model training, FastF1 calls, or any
other ML workflow operation.

Dashboard freshness depends on regenerated dashboard artifacts and the separately running read-only
Python API. The frontend does not poll, schedule refreshes, or claim live telemetry.

Milestone 40D includes the dashboard shell, Current Event page, Forecast leaderboard page, Practice
Status page, and static Methodology page. Settlement comparison, historical monitoring, deployment,
polling, websockets, and live updates are intentionally out of scope.
