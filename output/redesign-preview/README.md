# Apex Pulse — design preview 01

Isolated, local, responsive concept for review. The production Next.js application and ML pipeline have not been changed.

From the repository root:

```sh
python3 -m http.server 4321 --bind 127.0.0.1 --directory output/redesign-preview
```

Open http://127.0.0.1:4321. The folder is under the existing ignored `output/` directory; it is not a production route or deployment.

## Reviewable behavior

- Calendar-driven circuit outline with an illustrative animated lap and pointer parallax; native cursor with a small follower and short spark trail.
- Motion pause control, system reduced-motion support, keyboard focus and mobile layout.
- Light/dark theme, sticky navigation and scroll progress.
- Historical forecast table, driver/team search, full-field expansion and driver inspection.
- Prediction/result comparison, checkpoint availability and archived metrics.
- Expandable archive and methodology details.

## Data boundary

`snapshot.json` contains a minimal, read-only subset of the existing Hungarian GP 2026 dashboard exports generated on 30 July 2026. It does not include source paths or operational metadata. Historical figures are copied without alteration. It covers 21 of 22 entrants; missing predictions are not invented. FP1/FP2 forecasts, Q3 probabilities and confidence intervals are not available. Animation and decorative graphics are illustrative, not telemetry.

The prototype does not call the production API, run ingestion, train models, save predictions, commit, push or deploy. Google Fonts is used for typography with system fallbacks. The independent upcoming-weekend panel boots from `data/weekends.json`, then checks the public calendar online. Public cross-origin requests use no credentials. The latest verified version is cached in browser storage when available.

After design approval, port the agreed layout to the existing Next.js components, keeping its current data adapters, availability states, theme persistence and public route compatibility. Connect the main hero to the monitored weekend and show the most recent available forecast as its own explicit event. Run the existing frontend checks after integration.

## Visual refinements

- Team marks have an explicit centered image box with bounded padding, preventing portrait logos such as Ferrari from overflowing. All marks retain their original proportions.
- Cadillac uses a white CSS silhouette of its existing transparent asset, preserving the original file and shape in both themes.
- The methodology chart baseline ends at the last column's right edge; the three diagram icons share rounded stroke joins and caps.
- Filtered tables hide the full-field toggle correctly.

## Upcoming weekend, countdown and starting lights

The hero automatically follows the next scheduled GP and its next session, including the race. It counts down to FP1, FP2, FP3, qualifying and race on conventional weekends; Sprint weekends use their actual listed sessions. The displayed flag and circuit geometry follow the selected event. The archived Hungarian forecast remains a separate, explicitly labelled historical example.

Times are converted from the public calendar's track-local timestamps and explicit UTC offsets into UTC once, then shown in the visitor's local timezone. The display catches up immediately when a background tab becomes visible. At a scheduled start it shows zero and lights out for 30 seconds, then selects the next session. After a race's scheduled start plus that grace period, it follows the next GP. This indicates scheduled times, not verified live session status.

One additional red light turns on at each threshold: 7 days, 24 hours, 6 hours, 1 hour, 5 minutes. These are an interface metaphor, not the official starting sequence. **Preview the lights** runs an eight-second, clearly labelled compressed demonstration and returns to the real countdown automatically. **Follow the next Grand Prix** can also select the remaining upcoming weekends for exploration; **Back to next GP** restores automatic selection.

Sources:

- Calendar: [FastF1 public schedule provider](https://github.com/theOehrly/f1schedule). The São Paulo times were checked against the [official F1 weekend schedule](https://www.formula1.com/en/racing/2026/brazil) on 2 November 2026.
- Geometry: [Tomislav Bacinger's f1-circuits](https://github.com/bacinger/f1-circuits), with the original [MIT license](data/CIRCUITS-LICENSE.md). Coordinates are projected locally without stretching; tall outlines receive a rigid 90° rotation to fit the illustration. Moving dots do not represent telemetry or real lap timing. Geometry is a community outline, not a guarantee of the latest corner configuration.

The bootstrap bundle contains 23 calendar events and 40 circuit outlines, including all four remaining 2026 weekends. Raw bootstrap source hashes and retrieval time are recorded. Runtime refreshes replace the calendar with the newly verified source, rather than retaining removed races. Unknown geometry, missing timestamps and an exhausted calendar have explicit unavailable states; no substitute schedule or circuit is invented. In particular, the source's Bahrain/Kuala Lumpur location cannot be matched safely and has no outline mapping. No paid API is used.

Refresh the cached public data (also supports multiple `--season` arguments when those calendars are available):

```sh
python3 output/redesign-preview/scripts/update-weekend-data.py --season 2026
```

The preview checks the public schedule every five minutes, on returning to the tab (with a 15-second deduplication window), on reconnect and with the refresh control. Requests time out after ten seconds and overlapping refreshes share one request. The current UTC year and the following year are derived from the clock; there is no permanent 2026 selection. An unpublished following-year calendar is a normal unavailable state. A following-year outage does not prevent a corrected current-year schedule from loading.

Source additions, removals, explicit cancellations, revised session times, UTC offsets and changed formats replace the old event list. Session slots are discovered from the source fields instead of assuming exactly five. A postponed event or a future event awaiting confirmed times has no invented countdown. Event identities use season/name/location, preserving the selected GP if round numbering changes. Venue changes or a removed selected event return the explorer safely to the next available GP. Existing geometry is refreshed daily; an unknown future venue prompts a refresh and stays explicitly unavailable if no matching outline is published. Geometry matching uses location metadata and country, with documented aliases for Miami Gardens, Monte Carlo and Marina Bay.

If network, source schema or storage fails, the last verified or bundled version remains visible with freshness/degradation information. No unavailable calendar is presented as freshly verified. The previous successful browser cache is preferred over an older bootstrap file. Calendar changes become visible after the source publishes them and a refresh succeeds; a valid but incorrect upstream schedule cannot be detected automatically.

For final integration, the same source loader should be served through the existing read-only backend/export workflow to centralize requests across visitors. This prototype implements browser-side public reads and does not change the production backend, forecasts or automation.

Verification:

```sh
node --test output/redesign-preview/tests/*.test.mjs
```

The tests cover published UTC conversion, next-event selection, every session boundary through the race, race-to-next-GP rollover, Sprint sessions, missing or timezone-less timestamps, end of season, countdown clamping, light thresholds and distinct bounded circuit paths.


## Header and layout

The desktop hero uses separate 1.2:1 columns with a gap up to 112 px; tablet and mobile sizes use appropriate smaller horizontal or larger vertical spacing. A theme-aware GitHub icon in the header links to the project repository, opens in a new tab and has an accessible name.

The schedule tests also cover moved times, round reordering, source deletion, cancellation, postponed dates, additional session slots, changed UTC offsets, new circuit metadata, next-season discovery, partial source outages, last-verified fallback, incompatible schemas, corrupt or unavailable browser storage, request deduplication and timeouts.
