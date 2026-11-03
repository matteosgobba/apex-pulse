/** Pure scheduling and geometry logic. No forecast, ingestion or browser side effects. */
export const START_GRACE_MS = 30_000;
export const LIGHT_THRESHOLDS_MS = [
  7 * 86400,
  86400,
  6 * 3600,
  3600,
  5 * 60,
].map((s) => s * 1000);

export function lightCount(remainingMs) {
  if (!Number.isFinite(remainingMs) || remainingMs <= 0) return 0;
  return LIGHT_THRESHOLDS_MS.filter((threshold) => remainingMs <= threshold)
    .length;
}

export function countdownParts(remainingMs) {
  const seconds = Number.isFinite(remainingMs)
    ? Math.max(0, Math.ceil(remainingMs / 1000))
    : 0;
  return {
    days: Math.floor(seconds / 86400),
    hours: Math.floor((seconds % 86400) / 3600),
    minutes: Math.floor((seconds % 3600) / 60),
    seconds: seconds % 60,
  };
}

export function validSessions(event) {
  return (Array.isArray(event?.sessions) ? event.sessions : [])
    .filter(
      (session) =>
        session &&
        typeof session.code === "string" &&
        typeof session.name === "string" &&
        typeof session.startUtc === "string" &&
        /(?:Z|[+-]\d{2}:\d{2})$/.test(session.startUtc),
    )
    .map((session) => ({ ...session, startMs: Date.parse(session.startUtc) }))
    .filter((session) => Number.isFinite(session.startMs))
    .sort((a, b) => a.startMs - b.startMs);
}

export function selectWeekend(events, nowMs) {
  if (!Array.isArray(events) || !Number.isFinite(nowMs))
    return { status: "unavailable" };
  const ordered = events
    .filter(
      (event) => event && !["cancelled", "canceled"].includes(event.status),
    )
    .map((event) => {
      const sessions = validSessions(event);
      const race = sessions.find((session) => session.code === "R");
      const date = /^\d{4}-\d{2}-\d{2}$/.test(event.date ?? "")
        ? Date.parse(`${event.date}T00:00:00Z`)
        : NaN;
      return { event, sessions, race, anchor: race?.startMs ?? date };
    })
    .filter((item) => Number.isFinite(item.anchor))
    .sort((a, b) => a.anchor - b.anchor);
  for (const { event, sessions, race, anchor } of ordered) {
    if (race && nowMs >= race.startMs + START_GRACE_MS) continue;
    if (!race && nowMs >= anchor + 86400_000) continue;
    if (event.status === "postponed")
      return { status: "awaiting_schedule", event, sessions };
    const starting = sessions.findLast(
      (session) =>
        nowMs >= session.startMs && nowMs < session.startMs + START_GRACE_MS,
    );
    const session =
      starting ?? sessions.find((session) => session.startMs > nowMs);
    if (!session) return { status: "awaiting_schedule", event, sessions };
    return {
      status: starting ? "scheduled_start" : "upcoming",
      event,
      session,
      sessions,
      remainingMs: Math.max(0, session.startMs - nowMs),
      lights: lightCount(session.startMs - nowMs),
    };
  }
  return { status: ordered.length ? "complete" : "unavailable" };
}

/** Local projection with an optional rigid 90° rotation to fit tall circuits horizontally. */
export function projectCircuit(
  points,
  bounds = { x: 95, y: 135, width: 470, height: 280 },
) {
  if (
    !Array.isArray(points) ||
    points.length < 4 ||
    points.some(
      (point) =>
        !Array.isArray(point) ||
        point.length < 2 ||
        !Number.isFinite(point[0]) ||
        !Number.isFinite(point[1]) ||
        Math.abs(point[0]) > 180 ||
        Math.abs(point[1]) > 90,
    )
  )
    return null;
  const latitude =
    points.reduce((sum, point) => sum + point[1], 0) / points.length;
  const cosLat = Math.cos((latitude * Math.PI) / 180);
  let projected = points.map(([lon, lat]) => [lon * cosLat, -lat]);
  const spanX =
    Math.max(...projected.map((p) => p[0])) -
    Math.min(...projected.map((p) => p[0]));
  const spanY =
    Math.max(...projected.map((p) => p[1])) -
    Math.min(...projected.map((p) => p[1]));
  if (spanY > spanX) projected = projected.map(([x, y]) => [-y, x]);
  const xs = projected.map((point) => point[0]),
    ys = projected.map((point) => point[1]);
  const minX = Math.min(...xs),
    minY = Math.min(...ys);
  const width = Math.max(...xs) - minX,
    height = Math.max(...ys) - minY;
  if (width <= 0 || height <= 0) return null;
  const scale = Math.min(bounds.width / width, bounds.height / height);
  const left = bounds.x + (bounds.width - width * scale) / 2;
  const top = bounds.y + (bounds.height - height * scale) / 2;
  const coordinates = projected.map(([x, y]) => [
    +(left + (x - minX) * scale).toFixed(2),
    +(top + (y - minY) * scale).toFixed(2),
  ]);
  return {
    path:
      coordinates
        .map(([x, y], index) => `${index ? "L" : "M"}${x} ${y}`)
        .join(" ") + " Z",
    coordinates,
  };
}

export function countryFlag(code) {
  return typeof code === "string" && /^[A-Z]{2}$/.test(code)
    ? [...code]
        .map((letter) => String.fromCodePoint(127397 + letter.charCodeAt(0)))
        .join("")
    : "";
}
