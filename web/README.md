# Apex Pulse Web

Public-facing, read-only Next.js interface for Apex Pulse qualifying predictions.

## Local development

Terminal 1, from the repository root:

```bash
.venv/bin/python -m f1_prediction.cli dashboard-export
.venv/bin/python -m f1_prediction.cli dashboard-api
```

Terminal 2:

```bash
cd web
npm install
cp .env.example .env.local
npm run dev
```

The frontend expects the API at `http://127.0.0.1:8000` unless
`NEXT_PUBLIC_APEX_PULSE_API_BASE_URL` is set.

## Public information architecture

- `/` — coherent current/latest-event page with event context, session timing, weekend progression,
  predicted ranking, official comparison, metrics, methodology preview, technical disclosure, and
  contact.
- `/history` — expandable past predictions with prospective evidence kept separate from legacy
  descriptive records and historical backtests.
- `/methodology` — public methodology, data limits, evidence policy, and secondary technical details.
- `/monitoring-history` — backward-compatible alias for the public history experience.
- `/forecast`, `/practice`, and `/settlement` — backward-compatible redirects to the relevant
  homepage sections.

The old permanent operator sidebar is not part of the public interface.

## Data and countdown

The app consumes the existing version `1.0` dashboard JSON envelopes through the read-only FastAPI
service. The optional additive `event_schedule` block is exported from already-cached FastF1
`session_info.ff1pkl` metadata. The frontend never calls FastF1 or another calendar service while
rendering.

The countdown chooses the first verified future session in `FP1 → FP2 → FP3 → Qualifying` order,
updates once per second in the browser, uses the visitor's local time, clamps at zero, and shows a
safe unavailable state when schedule data is absent. Settled events show `Qualifying complete`.
The timeline reflects exported artifact availability, not lap-by-lap live telemetry.

## Team identity and assets

`lib/team-identity.ts` is the single source of truth for team names, aliases, accessible colors, and
optional logo paths. No team-logo files were present or added for Milestone 47, so the UI uses
polished local color/monogram marks. Unknown teams receive a deterministic, high-contrast fallback.
Future suitable local team logos can be registered through `logoPath`; missing files never block
the ranking.

The supplied Apex Pulse logos are copied unchanged into `public/brand/` for web delivery.

## Contact configuration

`lib/site-config.ts` centralizes contact values. The repository URL and author email were verified
from repository metadata and Git history. LinkedIn was not verifiable, so no LinkedIn link is
rendered by default. Configure a verified profile with:

```text
NEXT_PUBLIC_APEX_PULSE_LINKEDIN_URL=https://www.linkedin.com/in/verified-profile
```

Invalid or empty values are ignored.

## Read-only scope

The website does not trigger ingestion, preflight, forecasting, settlement, training, FastF1
downloads, or artifact mutation. It does not claim live telemetry. Partial coverage is explicit:
only comparable preserved forecast rows are evaluated, and official entrants missing from the
original forecast are listed without a retrospective prediction.

Freshness comes from the artifact envelope's `generated_at_utc`. The stale threshold only changes
the public notice; old but valid artifacts remain renderable.

## Verification

```bash
cd web
npm test
npm run lint
npm run typecheck
npm run build
```

The tests are deterministic and network-independent. They cover countdown selection and zero
clamping, missing schedules, ordering, team fallbacks, comparison semantics, partial coverage,
missing entrants, stale/unavailable states, contact filtering, technical disclosures, and public
history separation.
