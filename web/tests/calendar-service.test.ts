import { expect, test, vi } from "vitest";
import { createCalendarService } from "@/lib/racing/calendar-service";

const schedule = [
  {
    round_number: 20,
    event_name: "São Paulo Grand Prix",
    country: "Brazil",
    location: "São Paulo",
    event_date: "2026-11-08",
    gmt_offset: "-03:00",
    session1: "Practice 1",
    session1_date: "2026-11-06T14:30:00",
    session2: "Race",
    session2_date: "2026-11-08T14:00:00",
  },
];
const geo = {
  features: [
    {
      properties: { id: "br-test", Name: "Interlagos", Location: "Sao Paulo" },
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
const now = Date.parse("2026-11-02T20:00:00Z");

test("server deduplicates concurrent requests, caches success, then accepts rescheduled sessions", async () => {
  let clock = now;
  let start = "2026-11-06T14:30:00";
  const upstream = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("2027")) return new Response(null, { status: 404 });
    return Response.json(
      url.includes("geojson")
        ? geo
        : [{ ...schedule[0], session1_date: start }],
    );
  });
  const read = createCalendarService(upstream, () => clock);
  const first = read();
  expect(read()).toBe(first);
  const response = await first;
  expect(response.status).toBe("fresh");
  expect(response.bundle.events[0].sessions[0].startUtc).toBe(
    "2026-11-06T17:30:00.000Z",
  );
  const requests = upstream.mock.calls.length;
  await read();
  expect(upstream).toHaveBeenCalledTimes(requests);
  clock += 300_001;
  start = "2026-11-06T15:30:00";
  expect((await read()).bundle.events[0].sessions[0].startUtc).toBe(
    "2026-11-06T18:30:00.000Z",
  );
});

test("server labels an outage, preserves its last calendar, and recovers after the retry interval", async () => {
  const warning = vi.spyOn(console, "warn").mockImplementation(() => {});
  let clock = now;
  let offline = true;
  const read = createCalendarService(
    async (input) => {
      if (offline) throw new Error("Source offline");
      return String(input).includes("2027")
        ? new Response(null, { status: 404 })
        : Response.json(String(input).includes("geojson") ? geo : schedule);
    },
    () => clock,
  );
  const saved = await read();
  expect(saved.status).toBe("cached");
  expect(saved.bundle.events.length).toBeGreaterThan(0);
  expect(saved.warning).toMatch(/last verified/i);
  offline = false;
  clock += 300_001;
  expect((await read()).status).toBe("fresh");
  warning.mockRestore();
});
