import bootstrap from "@/lib/racing/racing-bootstrap.json";
import {
  fetchCalendarUpdate,
  REFRESH_INTERVAL_MS,
  validateBundle,
} from "./calendar-source.mjs";
import type { CalendarResponse } from "./types";

/** One refresh per server instance; upstream failures preserve the last verified schedule. */
export function createCalendarService(
  fetchImpl: typeof fetch = fetch,
  now = Date.now,
) {
  let bundle = validateBundle(bootstrap);
  let response: CalendarResponse | null = null;
  let expiresAt = 0;
  let inflight: Promise<CalendarResponse> | null = null;
  return function readCalendar(): Promise<CalendarResponse> {
    if (response && now() < expiresAt) return Promise.resolve(response);
    if (inflight) return inflight;
    inflight = (async () => {
      const checkedAtUtc = new Date(now()).toISOString();
      try {
        bundle = await fetchCalendarUpdate({
          current: bundle,
          now: now(),
          fetchImpl,
        });
        response = { bundle, status: "fresh", checkedAtUtc };
      } catch (error) {
        console.warn(
          "Public calendar refresh failed; retaining verified schedule",
          error instanceof Error ? error.message : "Unknown source error",
        );
        response = {
          bundle,
          status: "cached",
          checkedAtUtc,
          warning:
            "Calendar source unavailable. Showing the last verified schedule.",
        };
      }
      expiresAt = now() + REFRESH_INTERVAL_MS;
      return response;
    })().finally(() => {
      inflight = null;
    });
    return inflight;
  };
}

export const readCalendar = createCalendarService();
