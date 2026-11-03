import type { CalendarBundle, Circuit, RacingEvent } from "./types";
export const REFRESH_INTERVAL_MS: number;
export const SOURCE_TIMEOUT_MS: number;
export const GEOMETRY_REFRESH_MS: number;
export const CACHE_KEY: string;
export function normalizeKey(value: unknown): string;
export function stableEventId(
  event: Pick<RacingEvent, "season" | "name" | "location">,
): string;
export function parseCircuits(geojson: unknown): Record<string, Circuit>;
export function normalizeSchedule(
  payload: unknown,
  season: number,
  circuits?: Record<string, Circuit>,
): RacingEvent[];
export function validateBundle(bundle: unknown): CalendarBundle;
export function fetchCalendarUpdate(options: {
  current?: CalendarBundle;
  now?: number;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
}): Promise<CalendarBundle>;
