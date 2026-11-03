import { test } from "vitest";
import assert from "node:assert/strict";
import {
  CalendarClient,
  normalizeSchedule,
  parseCircuits,
  fetchCalendarUpdate,
  CACHE_KEY,
} from "../lib/racing/calendar-source.mjs";
import { selectWeekend } from "../lib/racing/race-engine.mjs";

const now = Date.parse("2026-11-02T20:00:00Z");
const geo = {
  features: [
    {
      properties: { id: "br-1940", Name: "Interlagos", Location: "Sao Paulo" },
      geometry: {
        type: "LineString",
        coordinates: [
          [-46.7, -23.7],
          [-46.69, -23.7],
          [-46.69, -23.69],
          [-46.7, -23.7],
        ],
      },
    },
  ],
};
const circuits = parseCircuits(geo);
const row = (changes = {}) => ({
  round_number: 20,
  event_name: "São Paulo Grand Prix",
  location: "São Paulo",
  country: "Brazil",
  event_format: "conventional",
  event_date: "2026-11-08T00:00:00",
  gmt_offset: "-03:00",
  session1: "Practice 1",
  session1_date: "2026-11-06T12:30:00",
  session2: "Practice 2",
  session2_date: "2026-11-06T16:00:00",
  session3: "Practice 3",
  session3_date: "2026-11-07T11:30:00",
  session4: "Qualifying",
  session4_date: "2026-11-07T15:00:00",
  session5: "Race",
  session5_date: "2026-11-08T14:00:00",
  ...changes,
});
const columns = (rows) =>
  Object.fromEntries(
    [...new Set(rows.flatMap(Object.keys))].map((key) => [
      key,
      Object.fromEntries(rows.map((r, i) => [String(i), r[key] ?? null])),
    ]),
  );
const bundle = () => ({
  schemaVersion: 1,
  retrievedAtUtc: "2026-11-01T00:00:00Z",
  geometryRetrievedAtUtc: new Date(now).toISOString(),
  circuits,
  events: normalizeSchedule(columns([row()]), 2026, circuits),
});
const response = (payload, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => payload,
});
const source =
  (current, next = null) =>
  async (url) =>
    url.includes("f1-circuits.geojson")
      ? response(geo)
      : url.includes("schedule_2027")
        ? response(next, next === null ? 404 : 200)
        : response(current);

test("a moved session uses its new time without changing event identity when rounds reorder", () => {
  const before = normalizeSchedule(columns([row()]), 2026, circuits)[0];
  const after = normalizeSchedule(
    columns([row({ round_number: 22, session1_date: "2026-11-06T14:30:00" })]),
    2026,
    circuits,
  )[0];
  assert.equal(before.id, after.id);
  assert.equal(
    selectWeekend([after], now).session.startUtc,
    "2026-11-06T17:30:00.000Z",
  );
  assert.notEqual(before.sessions[0].startUtc, after.sessions[0].startUtc);
});
test("source deletion and explicit cancellation remove a race from future selection", async () => {
  const vegas = row({
    round_number: 21,
    event_name: "Las Vegas Grand Prix",
    location: "Las Vegas",
    country: "United States",
    session5_date: "2026-11-21T20:00:00",
    gmt_offset: "-08:00",
  });
  const payload = columns([row({ status: "cancelled" }), vegas]);
  const events = normalizeSchedule(payload, 2026, circuits);
  assert.equal(events.length, 1);
  assert.equal(events[0].name, "Las Vegas Grand Prix");
  const updated = await fetchCalendarUpdate({
    current: bundle(),
    now,
    fetchImpl: source(columns([vegas])),
  });
  assert.ok(
    updated.events.every((event) => event.name !== "São Paulo Grand Prix"),
  );
});
test("postponed or unconfirmed sessions never count down to an invented time", () => {
  const postponed = normalizeSchedule(
    columns([row({ status: "postponed" })]),
    2026,
    circuits,
  );
  assert.equal(selectWeekend(postponed, now).status, "awaiting_schedule");
  const missing = normalizeSchedule(
    columns([row({ gmt_offset: null })]),
    2026,
    circuits,
  );
  assert.equal(missing[0].sessions.length, 0);
  assert.equal(selectWeekend(missing, now).status, "awaiting_schedule");
});
test("session slots and timezone changes come from the source rather than a five-slot template", () => {
  const event = normalizeSchedule(
    columns([
      row({
        gmt_offset: "-04:00",
        session6: "Additional Practice",
        session6_date: "2026-11-09T09:00:00",
        session1_date_utc: "2026-11-06T17:00:00",
      }),
    ]),
    2026,
    circuits,
  )[0];
  assert.equal(event.sessions.length, 6);
  assert.equal(event.sessions[0].startUtc, "2026-11-06T17:00:00.000Z");
  assert.equal(event.sessions[1].startUtc, "2026-11-06T20:00:00.000Z");
  assert.equal(event.sessions[5].name, "Additional Practice");
});
test("a newly added venue is matched from refreshed GeoJSON metadata; unknown tracks stay unavailable", () => {
  const newGeo = structuredClone(geo);
  newGeo.features[0].properties = {
    id: "za-1961",
    Name: "Kyalami",
    Location: "Johannesburg",
  };
  const event = normalizeSchedule(
    columns([
      row({
        event_name: "South African Grand Prix",
        country: "South Africa",
        location: "Johannesburg",
      }),
    ]),
    2027,
    parseCircuits(newGeo),
  )[0];
  assert.equal(event.countryCode, "ZA");
  assert.equal(event.circuitId, "za-1961");
  const unknown = normalizeSchedule(
    columns([row({ location: "Unannounced venue" })]),
    2026,
    circuits,
  )[0];
  assert.equal(unknown.circuitId, null);
  const mismatchedCountry = normalizeSchedule(
    columns([row({ country: "United States" })]),
    2026,
    circuits,
  )[0];
  assert.equal(mismatchedCountry.circuitId, null);
});
test("a published next season is discovered automatically, and an unpublished 404 is normal", async () => {
  const next = columns([
    row({
      event_date: "2027-03-08",
      session1_date: "2027-03-06T12:00:00",
      session2_date: "2027-03-06T16:00:00",
      session3_date: "2027-03-07T11:00:00",
      session4_date: "2027-03-07T15:00:00",
      session5_date: "2027-03-08T14:00:00",
    }),
  ]);
  const updated = await fetchCalendarUpdate({
    current: bundle(),
    now,
    fetchImpl: source(columns([row()]), next),
  });
  assert.equal(updated.nextSeasonAvailable, true);
  assert.equal(
    selectWeekend(updated.events, Date.parse("2027-01-01T00:00:00Z")).event
      .season,
    2027,
  );
  const pending = await fetchCalendarUpdate({
    current: bundle(),
    now,
    fetchImpl: source(columns([row()])),
  });
  assert.equal(pending.nextSeasonAvailable, false);
  assert.equal(pending.nextSeasonWarning, false);
});
test("a next-season outage does not block a corrected current-season start", async () => {
  const updated = await fetchCalendarUpdate({
    current: bundle(),
    now,
    fetchImpl: async (url) =>
      url.includes("2027")
        ? response(null, 503)
        : response(columns([row({ session1_date: "2026-11-06T14:30:00" })])),
  });
  assert.equal(updated.nextSeasonWarning, true);
  assert.equal(
    updated.events[0].sessions[0].startUtc,
    "2026-11-06T17:30:00.000Z",
  );
});
test("offline and incompatible schemas keep the last verified calendar and report degradation", async () => {
  for (const fetchImpl of [
    async () => {
      throw new Error("offline");
    },
    async () => response({ unsupported: true }),
  ]) {
    const initial = bundle(),
      updates = [],
      statuses = [];
    const client = new CalendarClient({
      initial,
      fetchImpl,
      now: () => now,
      onUpdate: (v) => updates.push(v),
      onStatus: (s) => statuses.push(s),
    });
    client.restore();
    assert.equal(await client.refresh(), false);
    assert.equal(client.current, initial);
    assert.equal(updates.length, 1);
    assert.equal(statuses.at(-1).state, "offline");
  }
});
test("invalid stored data is discarded, while the bundled fallback and online updates remain usable", async () => {
  const updates = [],
    statuses = [];
  const client = new CalendarClient({
    initial: bundle(),
    storage: {
      getItem: () => "{bad json",
      setItem: () => {
        throw new Error("quota");
      },
    },
    now: () => now,
    fetchImpl: source(columns([row()])),
    onUpdate: (v) => updates.push(v),
    onStatus: (s) => statuses.push(s),
  });
  client.restore();
  assert.ok(statuses.some((s) => s.state === "cache_unavailable"));
  assert.equal(await client.refresh(), true);
  assert.equal(updates.length, 2);
  assert.equal(statuses.at(-1).cacheAvailable, false);
  assert.equal(statuses.at(-1).state, "fresh");
});
test("a successful refresh is cached and overlapping refresh requests are deduplicated", async () => {
  let calls = 0,
    release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const saved = new Map();
  const fetchImpl = async (url) => {
    calls++;
    await gate;
    return source(columns([row()]))(url);
  };
  const client = new CalendarClient({
    initial: bundle(),
    fetchImpl,
    now: () => now,
    storage: {
      getItem: (key) => saved.get(key),
      setItem: (key, value) => saved.set(key, value),
    },
    onUpdate: () => {},
    onStatus: () => {},
  });
  const first = client.refresh(),
    second = client.refresh();
  assert.equal(first, second);
  release();
  assert.equal(await first, true);
  assert.equal(calls, 2);
  assert.equal(
    JSON.parse(saved.get(CACHE_KEY)).events[0].name,
    "São Paulo Grand Prix",
  );
});
test("a timed-out source request releases the client for the next retry", async () => {
  const client = new CalendarClient({
    initial: bundle(),
    now: () => now,
    timeoutMs: 5,
    fetchImpl: async (_url, { signal }) =>
      new Promise((_, reject) =>
        signal.addEventListener("abort", () => reject(new Error("timeout"))),
      ),
    onUpdate: () => {},
    onStatus: () => {},
  });
  assert.equal(await client.refresh(), false);
  assert.equal(client.inflight, null);
});
