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

## Scope

This app is a read-only consumer of validated dashboard JSON. It does not trigger ingestion,
preflight, forecast generation, target ingestion, settlement, model training, FastF1 calls, or any
other ML workflow operation.

Milestone 40C includes only the dashboard shell, the Current Event page, and the static Methodology
page. Forecast leaderboards, practice evolution charts, settlement comparison, historical monitoring,
deployment, polling, and live updates are intentionally out of scope.
