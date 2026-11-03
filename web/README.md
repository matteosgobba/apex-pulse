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

## Theme system

Apex Pulse defaults to dark mode on a visitor's first load. The sun/moon control in the shared
header switches immediately between dark and light without reloading, remains visible at mobile
sizes, and stores only `dark` or `light` under the browser-storage key `apex-pulse-theme`. Missing,
invalid, or unavailable storage falls back safely to dark.

The root layout ships `data-theme="dark"` as the no-JavaScript fallback and runs a small inline
initializer in the document head. That initializer restores a valid preference, `color-scheme`,
and `theme-color` before React hydrates, preventing a light-theme flash. The client theme provider
is the only runtime owner of theme state and browser storage.

`app/globals.css` defines the shared semantic color contract for background, surfaces, text,
borders, brand, success, warning, danger, and shadows. `tailwind.config.ts` exposes those variables
to components through the existing `apex-*` utilities, so the public routes share one component
implementation rather than separate dark and light trees. Theme transitions are limited to color
changes and reduced to effectively instant when `prefers-reduced-motion` is enabled.

## Racing homepage and calendar

The approved racing design is implemented in the real Next.js application, not loaded from
`output/redesign-preview`. The ranking, driver detail, official comparison, coverage and metrics
still come exclusively from the existing version `1.0` dashboard envelopes through the FastAPI
service. The leading predicted gap is preserved, including when it is non-zero. Search, full-field
expansion and driver selection never generate or modify predictions. Official results remain
unavailable until exported by the backend. A full comparison can be expanded below the ranking.

`GET /api/calendar` fetches the current and following season from the public FastF1 schedule
repository. Browser checks run every five minutes, on focus, on reconnect and when returning to
a visible tab. The server deduplicates concurrent requests and retains responses for five minutes.
Source requests time out after ten seconds. This endpoint requires a Next.js server runtime;
it is not compatible with a static-only export.

Schedule ingestion discovers the actual session slots, including Sprint formats, and converts
explicit local offsets or UTC timestamps. Corrections, removed/cancelled events, reordered rounds
and next-season publication are incorporated on refresh. Event selection uses a stable identity
rather than a round number. Postponed or unconfirmed sessions do not receive invented countdowns.
The animation and countdown describe the public schedule; they do not indicate live session status.

The counter ticks once a second, uses the visitor's timezone, reaches zero for 30 seconds at each
scheduled start, then advances to the following session and eventually the next Grand Prix.
Five red lights illuminate at 7 days, 24 hours, 6 hours, 1 hour and 5 minutes, and go out at zero.
The clearly labelled eight-second light demo changes only the light animation. Selecting another
upcoming event changes the circuit and its countdown; Auto follows the next Grand Prix again.

The initial fallback is `data/racing-bootstrap.json`, containing public calendar metadata and
circuit coordinates, never forecast rows. Last verified calendars are cached in browser storage
when available. Upstream failure keeps the last verified schedule with a visible warning and
verification time. A cold server fallback cannot overwrite a newer saved browser calendar.
Dates remain dependent on the public provider's own updates; refresh is not an official live feed.

Real circuit geometry comes from [f1-circuits](https://github.com/bacinger/f1-circuits), with a local
projection that preserves proportions and can rotate tall tracks. Unknown venues show an explicit
unavailable outline rather than a substitute circuit. The MIT attribution is included in
`public/CIRCUITS-LICENSE.md` and linked in the footer. The source may not yet contain newly added venues.

`lib/racing/race-engine.mjs` and `calendar-source.mjs` are framework-independent modules shared
with the tested preview logic; `.d.mts` files define their TypeScript interfaces. The server cache
and endpoint live in `calendar-service.ts` and `app/api/calendar/route.ts`. The React integration
is in `racing-weekend.tsx`. These modules do not call the prediction pipeline.

`app/racing.css` supplies the approved charcoal/cream palette and responsive layout. Barlow
Condensed, DM Sans and IBM Plex Mono are served locally from `public/fonts` with their SIL OFL
licenses, so building the app does not require Google Fonts access. The shared header includes
the repository link, persisted theme switch and animation pause. Reduced-motion preferences and
coarse pointers disable decorative cursor effects. Pausing motion does not stop the real clock.

The existing artifact-derived session countdown remains available in operational components when
used outside the landing page. The artifact timeline reports loaded data availability separately
from the refreshed public calendar. The page never assumes calendar activity means a forecast exists.

## Review and publish the integration

No backend model or API schema migration is required. Keep the existing
`NEXT_PUBLIC_APEX_PULSE_API_BASE_URL` configured to the deployed dashboard API. From `web/`:

```bash
npm test
npm run lint
npm run typecheck
npm run build
npm run dev -- --port 3001
```

The app is available at `http://localhost:3001`. The Python API must also be running to show real
prediction exports. For a visual-only local review, start it with
`APEX_PULSE_AUTOPILOT_SCHEDULER_ENABLED=false .venv/bin/python -m f1_prediction.cli dashboard-api`
from the repository root; this explicitly disables automatic prediction operations.

Commit the changes under `web/` and push the redesign branch for review. A connected Vercel project
with `web` as its root can build this branch as a preview; production publication follows merging
into the project's configured production branch. This integration does not change that deployment
configuration or publish automatically from this working tree.

## Team identity and assets

`lib/team-identity.ts` is the single source of truth for team names, aliases, accessible colors, and
logo paths. Mixed PNG, SVG, and WebP assets live under `public/teams/` and render through the shared
`TeamMark` component on a theme-independent `#2b2b30` backing square. Unknown teams keep the
deterministic, high-contrast monogram fallback, so a missing mapping never blocks the ranking.

The supplied Apex Pulse logos are copied unchanged into `public/brand/` for web delivery. The
header uses the black monochrome mark in light mode and the white monochrome mark in dark mode;
CSS follows the root `data-theme` value without introducing separate theme state or a logo card.

## Contact configuration

`lib/site-config.ts` centralizes contact values. GitHub, email, and Matteo Sgobba's LinkedIn profile
are rendered in the About and footer contact surfaces. A deployment may override the default
LinkedIn profile with:

```text
NEXT_PUBLIC_APEX_PULSE_LINKEDIN_URL=https://www.linkedin.com/in/profile
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
history separation. Theme coverage includes dark-default initialization, valid and invalid stored
preferences, both toggle directions, root attribute updates, storage persistence, accessible labels,
hydration behavior, shared desktop/mobile navigation, and all three public routes in both themes.
