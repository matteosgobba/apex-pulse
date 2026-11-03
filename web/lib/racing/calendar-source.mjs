/** Public calendar ingestion. Shared with the tested preview; no prediction side effects. */
export const REFRESH_INTERVAL_MS = 5 * 60_000;
export const SOURCE_TIMEOUT_MS = 10_000;
export const GEOMETRY_REFRESH_MS = 24 * 60 * 60_000;
export const CACHE_KEY = "apex-pulse-calendar-v2";
const SCHEDULE_URL = (year) =>
  `https://raw.githubusercontent.com/theOehrly/f1schedule/master/schedule_${year}.json`;
const GEOMETRY_URL =
  "https://raw.githubusercontent.com/bacinger/f1-circuits/master/f1-circuits.geojson";
const CODES = {
  "practice 1": "FP1",
  "practice 2": "FP2",
  "practice 3": "FP3",
  qualifying: "Q",
  race: "R",
  sprint: "SPRINT",
  "sprint qualifying": "SQ",
  "sprint shootout": "SQ",
};
const COUNTRIES = {
  australia: "AU",
  bahrain: "BH",
  china: "CN",
  japan: "JP",
  "united-states": "US",
  canada: "CA",
  monaco: "MC",
  spain: "ES",
  austria: "AT",
  "united-kingdom": "GB",
  belgium: "BE",
  hungary: "HU",
  netherlands: "NL",
  italy: "IT",
  azerbaijan: "AZ",
  singapore: "SG",
  mexico: "MX",
  brazil: "BR",
  qatar: "QA",
  "united-arab-emirates": "AE",
  "saudi-arabia": "SA",
  portugal: "PT",
  turkey: "TR",
  "south-africa": "ZA",
  argentina: "AR",
};
const LOCATION_ALIASES = {
  "miami-gardens": "miami",
  "monte-carlo": "monaco",
  "marina-bay": "singapore",
};

export function normalizeKey(value) {
  return String(value ?? "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

export function stableEventId(event) {
  return `${event.season}-${normalizeKey(event.name)}-${normalizeKey(event.location)}`;
}

export function parseCircuits(geojson) {
  if (!Array.isArray(geojson?.features))
    throw new Error("Circuit source schema unavailable");
  const circuits = {};
  for (const feature of geojson.features) {
    const props = feature.properties;
    if (
      feature.geometry?.type !== "LineString" ||
      !props?.id ||
      !props.Name ||
      !Array.isArray(feature.geometry.coordinates)
    )
      continue;
    circuits[props.id] = {
      name: props.Name,
      location: props.Location ?? "",
      countryCode: props.id.slice(0, 2).toUpperCase(),
      points: feature.geometry.coordinates.map((p) => [p[0], p[1]]),
    };
  }
  if (!Object.keys(circuits).length)
    throw new Error("Circuit source contains no usable outlines");
  return circuits;
}

export function normalizeSchedule(payload, season, circuits = {}) {
  if (!Number.isInteger(season)) throw new Error("Invalid calendar season");
  let rows;
  if (Array.isArray(payload)) rows = payload;
  else if (
    payload &&
    typeof payload.round_number === "object" &&
    payload.round_number !== null
  ) {
    rows = Object.keys(payload.round_number).map((index) =>
      Object.fromEntries(
        Object.entries(payload).map(([key, values]) => [key, values?.[index]]),
      ),
    );
  } else throw new Error("Unsupported schedule source schema");
  const events = [];
  for (const row of rows) {
    if (!Number.isFinite(Number(row.round_number)))
      throw new Error("Invalid calendar round");
    if (Number(row.round_number) <= 0) continue;
    if (
      typeof row.event_name !== "string" ||
      typeof row.location !== "string" ||
      typeof row.country !== "string"
    )
      throw new Error("Calendar event identity is incomplete");
    const sourceStatus = normalizeKey(
      row.status ?? row.event_status ?? "scheduled",
    );
    if (
      row.cancelled === true ||
      row.canceled === true ||
      ["cancelled", "canceled"].includes(sourceStatus)
    )
      continue;
    const location = normalizeKey(row.location);
    const lookup = LOCATION_ALIASES[location] ?? location;
    const expectedCountry = COUNTRIES[normalizeKey(row.country)];
    const matchingCircuit = Object.entries(circuits).find(
      ([id, circuit]) =>
        normalizeKey(circuit.location) === lookup &&
        (!expectedCountry ||
          expectedCountry ===
            (circuit.countryCode ?? id.slice(0, 2).toUpperCase())),
    );
    const circuitId = matchingCircuit?.[0] ?? null;
    const countryCode =
      COUNTRIES[normalizeKey(row.country)] ??
      matchingCircuit?.[1].countryCode ??
      null;
    const sessions = [];
    // Discover slots from the actual payload, including formats with more than five sessions.
    for (const key of Object.keys(row)
      .filter((key) => /^session\d+$/.test(key))
      .sort((a, b) => Number(a.slice(7)) - Number(b.slice(7)))) {
      const name = row[key];
      if (typeof name !== "string" || !name.trim()) continue;
      const utc = row[`${key}_date_utc`];
      const local = row[`${key}_date`];
      const value = typeof utc === "string" && utc ? utc : local;
      if (typeof value !== "string" || !value) continue;
      const explicitTimezone = /(?:Z|[+-]\d{2}:\d{2})$/i.test(value);
      const offset =
        typeof row[`${key}_gmt_offset`] === "string"
          ? row[`${key}_gmt_offset`]
          : row.gmt_offset;
      const suffix = explicitTimezone
        ? ""
        : utc
          ? "Z"
          : /^(?:Z|[+-]\d{2}:\d{2})$/.test(offset ?? "")
            ? offset
            : null;
      if (suffix === null) continue;
      const ms = Date.parse(value + suffix);
      if (!Number.isFinite(ms)) continue;
      sessions.push({
        code: CODES[name.trim().toLowerCase()] ?? name.trim(),
        name: name.trim(),
        startUtc: new Date(ms).toISOString(),
      });
    }
    const event = {
      season,
      round: Number(row.round_number),
      name: row.event_name.trim(),
      location: row.location.trim(),
      country: row.country.trim(),
      countryCode,
      circuitId,
      format:
        typeof row.event_format === "string" ? row.event_format : "unknown",
      status: sourceStatus,
      date:
        typeof row.event_date === "string" ? row.event_date.slice(0, 10) : null,
      sessions: sessions.sort(
        (a, b) => Date.parse(a.startUtc) - Date.parse(b.startUtc),
      ),
    };
    event.id = stableEventId(event);
    if (events.some((existing) => existing.id === event.id))
      throw new Error("Duplicate calendar identity");
    events.push(event);
  }
  return events;
}

export function validateBundle(bundle) {
  if (
    bundle?.schemaVersion !== 1 ||
    !Array.isArray(bundle.events) ||
    !bundle.circuits ||
    typeof bundle.circuits !== "object" ||
    Array.isArray(bundle.circuits) ||
    !Number.isFinite(Date.parse(bundle.retrievedAtUtc))
  )
    throw new Error("Invalid calendar bundle");
  const ids = new Set();
  for (const event of bundle.events) {
    if (
      !event ||
      typeof event.id !== "string" ||
      typeof event.name !== "string" ||
      typeof event.country !== "string" ||
      typeof event.location !== "string" ||
      !Number.isInteger(event.season) ||
      !Array.isArray(event.sessions) ||
      ids.has(event.id) ||
      event.sessions.some(
        (session) =>
          !session ||
          typeof session.name !== "string" ||
          typeof session.code !== "string" ||
          typeof session.startUtc !== "string",
      )
    )
      throw new Error("Invalid cached calendar event");
    ids.add(event.id);
  }
  return bundle;
}

async function fetchJson(
  url,
  { fetchImpl, optional = false, timeoutMs = SOURCE_TIMEOUT_MS },
) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetchImpl(url, {
      cache: "no-store",
      credentials: "omit",
      signal: controller.signal,
    });
    if (optional && response.status === 404) return null;
    if (!response.ok)
      throw new Error(`Public calendar source HTTP ${response.status}`);
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

export async function fetchCalendarUpdate({
  current,
  now = Date.now(),
  fetchImpl = fetch,
  timeoutMs = SOURCE_TIMEOUT_MS,
}) {
  const year = new Date(now).getUTCFullYear();
  const [currentResult, nextResult] = await Promise.allSettled([
    fetchJson(SCHEDULE_URL(year), { fetchImpl, timeoutMs }),
    fetchJson(SCHEDULE_URL(year + 1), { fetchImpl, optional: true, timeoutMs }),
  ]);
  if (currentResult.status === "rejected") throw currentResult.reason;
  const currentRaw = currentResult.value;
  let nextRaw = nextResult.status === "fulfilled" ? nextResult.value : null;
  let nextSeasonWarning = nextResult.status === "rejected";
  const savedNext = (current?.events ?? []).filter(
    (event) => event.season === year + 1,
  );
  if (nextRaw === null && savedNext.length) nextSeasonWarning = true;
  let circuits = Object.fromEntries(
    Object.entries(current?.circuits ?? {}).map(([id, circuit]) => [
      id,
      {
        ...circuit,
        location:
          circuit.location ??
          current?.events?.find((event) => event.circuitId === id)?.location ??
          "",
        countryCode: circuit.countryCode ?? id.slice(0, 2).toUpperCase(),
      },
    ]),
  );
  let geometryRetrievedAtUtc = current?.geometryRetrievedAtUtc;
  const needsGeometry =
    !Object.values(circuits).some((circuit) => circuit.location) ||
    !geometryRetrievedAtUtc ||
    now - Date.parse(geometryRetrievedAtUtc) >= GEOMETRY_REFRESH_MS;
  let geometryWarning = false;
  if (needsGeometry) {
    try {
      circuits = parseCircuits(
        await fetchJson(GEOMETRY_URL, { fetchImpl, timeoutMs }),
      );
      geometryRetrievedAtUtc = new Date(now).toISOString();
    } catch (error) {
      if (!Object.keys(circuits).length) throw error;
      geometryWarning = true;
    }
  }
  let events = normalizeSchedule(currentRaw, year, circuits);
  if (nextRaw !== null) {
    try {
      events.push(...normalizeSchedule(nextRaw, year + 1, circuits));
    } catch {
      nextRaw = null;
      nextSeasonWarning = true;
    }
  }
  if (nextRaw === null && nextSeasonWarning) events.push(...savedNext);
  // A newly announced venue prompts an immediate geometry refresh, even within the daily TTL.
  if (
    !needsGeometry &&
    events.some(
      (event) =>
        !event.circuitId &&
        event.sessions.some((session) => Date.parse(session.startUtc) > now),
    )
  ) {
    try {
      circuits = parseCircuits(
        await fetchJson(GEOMETRY_URL, { fetchImpl, timeoutMs }),
      );
      geometryRetrievedAtUtc = new Date(now).toISOString();
      events = normalizeSchedule(currentRaw, year, circuits);
      if (nextRaw !== null)
        events.push(...normalizeSchedule(nextRaw, year + 1, circuits));
      else if (nextSeasonWarning) events.push(...savedNext);
    } catch {
      geometryWarning = true;
    }
  }
  return validateBundle({
    schemaVersion: 1,
    retrievedAtUtc: new Date(now).toISOString(),
    geometryRetrievedAtUtc,
    geometryWarning,
    nextSeasonWarning,
    source: "fastf1-public-calendar",
    nextSeasonAvailable: nextRaw !== null,
    sources: [
      { kind: "calendar", url: SCHEDULE_URL(year) },
      ...(nextRaw ? [{ kind: "calendar", url: SCHEDULE_URL(year + 1) }] : []),
    ],
    events,
    circuits,
  });
}

export class CalendarClient {
  constructor({
    onUpdate,
    onStatus,
    initial,
    fetchImpl = fetch,
    storage,
    now = Date.now,
    timeoutMs = SOURCE_TIMEOUT_MS,
  }) {
    this.onUpdate = onUpdate;
    this.onStatus = onStatus;
    this.current = initial;
    this.fetchImpl = fetchImpl;
    this.storage = storage;
    this.now = now;
    this.timeoutMs = timeoutMs;
    this.inflight = null;
  }
  restore() {
    try {
      const raw = this.storage?.getItem(CACHE_KEY);
      if (raw) {
        const cached = validateBundle(JSON.parse(raw));
        if (
          !this.current ||
          Date.parse(cached.retrievedAtUtc) >
            Date.parse(this.current.retrievedAtUtc)
        )
          this.current = cached;
      }
    } catch {
      this.onStatus({ state: "cache_unavailable" });
    }
    if (this.current) this.onUpdate(this.current);
    this.onStatus({ state: "cached", checkedAt: this.current?.retrievedAtUtc });
  }
  refresh() {
    if (this.inflight) return this.inflight;
    this.inflight = (async () => {
      this.onStatus({
        state: "refreshing",
        checkedAt: this.current?.retrievedAtUtc,
      });
      try {
        const updated = await fetchCalendarUpdate({
          current: this.current,
          now: this.now(),
          fetchImpl: this.fetchImpl,
          timeoutMs: this.timeoutMs,
        });
        this.current = updated;
        this.onUpdate(updated);
        let cacheAvailable = Boolean(this.storage);
        try {
          this.storage?.setItem(CACHE_KEY, JSON.stringify(updated));
        } catch {
          cacheAvailable = false;
        }
        this.onStatus({
          state: updated.geometryWarning
            ? "geometry_cached"
            : updated.nextSeasonWarning
              ? "next_season_cached"
              : "fresh",
          checkedAt: updated.retrievedAtUtc,
          cacheAvailable,
        });
        return true;
      } catch (error) {
        this.onStatus({
          state: "offline",
          checkedAt: this.current?.retrievedAtUtc,
          message: error.message,
        });
        return false;
      } finally {
        this.inflight = null;
      }
    })();
    return this.inflight;
  }
}
